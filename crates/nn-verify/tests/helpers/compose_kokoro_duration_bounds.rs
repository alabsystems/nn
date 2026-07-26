// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! IBP compose tests for Kokoro duration predictor bounds.
//!
//! The Kokoro TTS duration predictor estimates phoneme durations. Architecture:
//!   - Linear projection from hidden dim to intermediate
//!   - Conv1d layers with LayerNorm for temporal smoothing
//!   - Duration projection to scalar per phoneme
//!   - ReLU/Softplus activation ensuring non-negative durations
//!   - Regulate step expanding phoneme features to variable-length frame features
//!   - Speed scaling multiplying durations by a speed factor
//!
//! This file verifies 8 IBP properties of the duration predictor pipeline:
//!
//! 1. **Linear projection bounds** -- hidden to intermediate preserves IBP bounds.
//! 2. **Conv1d with LayerNorm bounds** -- Conv+LN stack maintains bounded output.
//! 3. **Duration projection bounds** -- projection to scalar produces bounded durations.
//! 4. **ReLU/Softplus activation bounds** -- non-negative activation output bounded.
//! 5. **Regulate expansion bounds** -- expanding phoneme features preserves per-frame bounds.
//! 6. **Speed scaling bounds** -- duration * speed_factor produces bounded output.
//! 7. **Full duration predictor** -- end-to-end from hidden features to frame durations.
//! 8. **Variable-length handling** -- different sequence lengths produce consistent bounds.
//!
//! All tests use small dims (D<=16, T<=8) and IBP propagation through proxy graphs
//! built with TensorBlockBuilder.
//!
//! Part of #3351: Epic -- Absolutely Best Kokoro.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

use super::common::{
    assert_bounds_valid, assert_bounds_width, assert_norm_spatial_non_degenerate, bounds_min_max,
    uniform_bounds,
};

// ===========================================================================
// Constants
// ===========================================================================

/// Hidden dimension (d_model in production: 512; toy scale for verification).
const D_HIDDEN: usize = 16;

/// Intermediate dimension for conv layers.
const D_INTER: usize = 8;

/// Sequence length (number of phonemes).
const SEQ_LEN: usize = 8;

/// Maximum duration bins (production: 50).
const MAX_DUR: usize = 10;

/// Conv1d kernel size for temporal smoothing.
const KERNEL_SIZE: usize = 3;

/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.01;

/// Vacuous width threshold -- bounds wider than this are meaningless.
const VACUOUS_THRESHOLD: f32 = 500.0;

// ===========================================================================
// Builder helpers
// ===========================================================================

/// Build a linear projection graph: hidden -> intermediate.
///
/// Input: `[SEQ_LEN, D_HIDDEN]` (Variable).
/// Output: `[SEQ_LEN, D_INTER]`.
fn build_linear_projection(seq_len: usize, d_in: usize, d_out: usize) -> TensorKernelDef {
    let in_shape = [seq_len, d_in];
    let out_shape = [seq_len, d_out];
    let mut b = TensorBlockBuilder::new("duration_linear_proj");

    let x = b.add_input("x", &in_shape);
    let w = b.add_input("w", &[d_out, d_in]);
    let bias = b.add_input("bias", &[d_out]);
    let out = b.add_linear(x, w, Some(bias), &out_shape);

    b.build(out).expect("valid linear projection graph")
}

/// Bindings for a linear projection with given weight magnitude.
fn linear_proj_bindings(d_in: usize, d_out: usize, weight_mag: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_out, d_in]), weight_mag)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_out]), 0.0f32)),
    ]
}

/// Build a Conv1d + LayerNorm block for temporal smoothing.
///
/// Input: `[D_INTER, T]` (Variable).
/// Output: `[D_INTER, T]`.
///
/// Architecture: Conv1d(same padding) -> LayerNorm(axis=0) over channels.
fn build_conv_layernorm(channels: usize, time_len: usize) -> TensorKernelDef {
    assert_norm_spatial_non_degenerate(time_len, "conv_layernorm");
    let shape = [channels, time_len];
    let padding = (KERNEL_SIZE - 1) / 2; // same padding for k=3

    let mut b = TensorBlockBuilder::new("duration_conv_layernorm");

    let x = b.add_input("x", &shape);
    let conv_w = b.add_input("conv_w", &[channels, channels, KERNEL_SIZE]);
    let conv_b = b.add_input("conv_b", &[channels]);
    let conv_out = b.add_conv1d(x, conv_w, Some(conv_b), 1, padding, &shape);

    // LayerNorm normalizes over the last axis (PyTorch normalized_shape convention).
    // To normalize over channels while keeping the public [C, T] contract, transpose
    // to [T, C] so channels are last, LayerNorm over channels, then transpose back.
    let tc_shape = [time_len, channels];
    let conv_tc = b.add_transpose(conv_out, &[1, 0], &tc_shape);

    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[channels]);
    let ln_b = b.add_input("ln_b", &[channels]);
    let normed_tc = b.add_layer_norm(conv_tc, eps, 1, ln_w, ln_b, &tc_shape);

    // Transpose back to [C, T].
    let out = b.add_transpose(normed_tc, &[1, 0], &shape);

    b.build(out).expect("valid conv+layernorm graph")
}

/// Bindings for Conv1d + LayerNorm.
fn conv_layernorm_bindings(channels: usize, weight_mag: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels, channels, KERNEL_SIZE]),
            weight_mag,
        )), // conv_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // conv_b
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // ln_b
    ]
}

/// Build a duration projection graph: intermediate -> scalar per phoneme.
///
/// Input: `[T, D_INTER]` (Variable).
/// Output: `[T, MAX_DUR]`.
fn build_duration_projection(seq_len: usize, d_in: usize, max_dur: usize) -> TensorKernelDef {
    let in_shape = [seq_len, d_in];
    let out_shape = [seq_len, max_dur];
    let mut b = TensorBlockBuilder::new("duration_projection");

    let x = b.add_input("x", &in_shape);
    let w = b.add_input("w", &[max_dur, d_in]);
    let bias = b.add_input("bias", &[max_dur]);
    let out = b.add_linear(x, w, Some(bias), &out_shape);

    b.build(out).expect("valid duration projection graph")
}

/// Bindings for duration projection.
fn duration_proj_bindings(d_in: usize, max_dur: usize, weight_mag: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[max_dur, d_in]), weight_mag)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[max_dur]), 0.0f32)),
    ]
}

/// Build a ReLU + Softplus activation graph.
///
/// Input: `[T, 1]` (Variable).
/// Output: `[T, 1]` -- non-negative durations.
///
/// Architecture: ReLU ensures non-negative, Softplus smooths near zero.
fn build_activation_block(seq_len: usize) -> TensorKernelDef {
    let shape = [seq_len, 1];
    let mut b = TensorBlockBuilder::new("duration_activation");

    let x = b.add_input("x", &shape);
    let relu = b.add_relu(x, &shape);
    let out = b.add_softplus(relu, &shape);

    b.build(out).expect("valid activation graph")
}

/// Bindings for activation block (input only, no parameters).
fn activation_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

/// Build a regulate expansion proxy graph.
///
/// Simulates length_regulate by repeating each phoneme feature by a fixed
/// expansion factor. Since true length_regulate is data-dependent (segment
/// boundary), we model it as a matmul with a binary expansion matrix.
///
/// Input: `[D, T_in]` (Variable) -- phoneme features.
/// Output: `[D, T_out]` -- frame features.
///
/// The expansion matrix has shape `[T_in, T_out]` where each row has
/// `expansion_factor` ones, spreading each phoneme across multiple frames.
fn build_regulate_expansion(
    channels: usize,
    t_in: usize,
    expansion_factor: usize,
) -> TensorKernelDef {
    let t_out = t_in * expansion_factor;
    let in_shape = [channels, t_in];
    let out_shape = [channels, t_out];
    // Expansion matrix: [T_in, T_out]. We transpose the features and multiply.
    // Actually: features [D, T_in] -> transpose [T_in, D] -> matmul with expand [T_in, T_out]
    // That gives [T_in, T_out] which is wrong.
    // Better: use features [D, T_in] and expand matrix [T_in, T_out]:
    //   result = features @ expand_matrix = [D, T_in] @ [T_in, T_out] = [D, T_out]
    let mut b = TensorBlockBuilder::new("duration_regulate_expansion");

    let x = b.add_input("x", &in_shape);
    let expand_mat = b.add_input("expand_mat", &[t_in, t_out]);
    let out = b.add_matmul(x, expand_mat, false, None, &out_shape);

    b.build(out).expect("valid regulate expansion graph")
}

/// Bindings for regulate expansion.
///
/// The expansion matrix is binary: each phoneme maps to `expansion_factor`
/// consecutive frames.
fn regulate_expansion_bindings(t_in: usize, expansion_factor: usize) -> Vec<TensorParamBinding> {
    let t_out = t_in * expansion_factor;
    let mut expand_data = vec![0.0f32; t_in * t_out];
    for i in 0..t_in {
        for k in 0..expansion_factor {
            let col = i * expansion_factor + k;
            if col < t_out {
                expand_data[i * t_out + col] = 1.0;
            }
        }
    }
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[t_in, t_out]), expand_data)
                .expect("valid expansion matrix"),
        ),
    ]
}

/// Build a speed scaling graph: duration * speed_factor.
///
/// Input: `[T, 1]` (Variable) -- durations.
/// Output: `[T, 1]` -- scaled durations.
fn build_speed_scaling(seq_len: usize) -> TensorKernelDef {
    let shape = [seq_len, 1];
    let mut b = TensorBlockBuilder::new("duration_speed_scaling");

    let x = b.add_input("x", &shape);
    let speed = b.add_input("speed", &[1]);
    let speed_bc = b.add_broadcast(speed, &shape);
    let out = b.add_binary_mul(x, speed_bc, &shape);

    b.build(out).expect("valid speed scaling graph")
}

/// Bindings for speed scaling with a given factor.
fn speed_scaling_bindings(speed_factor: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(speed_factor),
    ]
}

/// Build a full duration predictor proxy graph.
///
/// Architecture: Linear(hidden->inter) -> Conv1d+LayerNorm -> Linear(inter->max_dur) -> Softplus.
///
/// Input: `[T, D_HIDDEN]` (Variable).
/// Output: `[T, MAX_DUR]`.
fn build_full_duration_predictor(
    seq_len: usize,
    d_hidden: usize,
    d_inter: usize,
    max_dur: usize,
) -> TensorKernelDef {
    assert_norm_spatial_non_degenerate(seq_len, "full_duration_predictor");
    let in_shape = [seq_len, d_hidden];
    let proj1_shape = [seq_len, d_inter];
    // For Conv+LN, we need [C, T] layout -> transpose to [d_inter, seq_len]
    let conv_shape = [d_inter, seq_len];
    let out_shape = [seq_len, max_dur];

    let mut b = TensorBlockBuilder::new("kokoro_full_duration_predictor");

    // Input
    let x = b.add_input("x", &in_shape);

    // Stage 1: Linear projection hidden -> intermediate
    let w1 = b.add_input("w1", &[d_inter, d_hidden]);
    let b1 = b.add_input("b1", &[d_inter]);
    let proj1 = b.add_linear(x, w1, Some(b1), &proj1_shape);

    // Stage 2: Transpose to [C, T] for conv (axes [1, 0] swaps dims)
    let proj1_t = b.add_transpose(proj1, &[1, 0], &conv_shape);

    // Stage 3: Conv1d with same padding
    let padding = (KERNEL_SIZE - 1) / 2;
    let conv_w = b.add_input("conv_w", &[d_inter, d_inter, KERNEL_SIZE]);
    let conv_b = b.add_input("conv_b", &[d_inter]);
    let conv_out = b.add_conv1d(proj1_t, conv_w, Some(conv_b), 1, padding, &conv_shape);

    // Stage 4: Transpose back to [T, C] so the channel/feature dimension is last.
    // LayerNorm normalizes over the last axis (PyTorch normalized_shape convention),
    // so the per-channel normalization must be expressed with channels trailing.
    let conv_t = b.add_transpose(conv_out, &[1, 0], &proj1_shape);

    // Stage 5: LayerNorm over the channel dimension (now the last axis).
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[d_inter]);
    let ln_b = b.add_input("ln_b", &[d_inter]);
    let normed_t = b.add_layer_norm(conv_t, eps, 1, ln_w, ln_b, &proj1_shape);

    // Stage 6: Duration projection intermediate -> max_dur
    let w2 = b.add_input("w2", &[max_dur, d_inter]);
    let b2 = b.add_input("b2", &[max_dur]);
    let dur_logits = b.add_linear(normed_t, w2, Some(b2), &out_shape);

    // Stage 7: Softplus activation for non-negative durations
    let out = b.add_softplus(dur_logits, &out_shape);

    b.build(out).expect("valid full duration predictor graph")
}

/// Bindings for the full duration predictor.
fn full_predictor_bindings(
    d_hidden: usize,
    d_inter: usize,
    max_dur: usize,
    weight_mag: f32,
) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d_inter, d_hidden]),
            weight_mag,
        )), // w1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_inter]), 0.0f32)), // b1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[d_inter, d_inter, KERNEL_SIZE]),
            weight_mag,
        )), // conv_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_inter]), 0.0f32)), // conv_b
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_inter]), 1.0f32)), // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d_inter]), 0.0f32)), // ln_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[max_dur, d_inter]),
            weight_mag,
        )), // w2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[max_dur]), 0.0f32)), // b2
    ]
}

// ===========================================================================
// Test 1: Linear projection bounds
// ===========================================================================

/// Linear projection from hidden to intermediate preserves IBP bounds.
///
/// A Linear layer computes `y = x @ W^T + b`. For IBP, interval matrix-vector
/// multiply produces output bounds proportional to `input_range * ||W||_1`.
/// With small weights (0.01), the output bounds should remain tight.
#[test]
fn test_duration_linear_projection_bounds() {
    let def = build_linear_projection(SEQ_LEN, D_HIDDEN, D_INTER);
    def.validate().expect("linear projection def validates");

    let bindings = linear_proj_bindings(D_HIDDEN, D_INTER, WEIGHT_MAG);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through linear projection");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "linear projection bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, D_INTER],
        "linear projection must produce [T, D_INTER] output"
    );

    // With weight_mag=0.01 and D_HIDDEN=16 inputs in [-1, 1]:
    // Max output per element: D_HIDDEN * WEIGHT_MAG * 1.0 = 0.16. Plus zero bias.
    // Width should be at most 2 * 0.16 = 0.32 per element.
    assert!(
        width < 5.0,
        "linear projection with small weights should have tight bounds, got width={width}"
    );

    eprintln!("Duration linear projection: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 2: Conv1d with LayerNorm bounds
// ===========================================================================

/// Conv1d + LayerNorm stack maintains bounded output through IBP.
///
/// Conv1d with same-padding preserves temporal dimension. LayerNorm normalizes
/// the channel dimension, keeping activations centered. With small weights,
/// the composition should produce tight bounds.
#[test]
fn test_duration_conv_layernorm_bounds() {
    let def = build_conv_layernorm(D_INTER, SEQ_LEN);
    def.validate().expect("conv+layernorm def validates");

    let bindings = conv_layernorm_bindings(D_INTER, WEIGHT_MAG);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[D_INTER, SEQ_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv+layernorm");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "conv+layernorm bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[D_INTER, SEQ_LEN],
        "conv+layernorm must preserve [C, T] shape"
    );

    // LayerNorm normalizes output to near zero-mean/unit-variance.
    // With small conv weights, output bounds should be reasonable.
    assert!(
        width < VACUOUS_THRESHOLD,
        "conv+layernorm bounds width {width} exceeds vacuous threshold {VACUOUS_THRESHOLD}"
    );

    eprintln!("Duration Conv1d+LayerNorm: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 3: Duration projection bounds
// ===========================================================================

/// Projection to max_dur scalar bins produces bounded duration logits.
///
/// This is the final linear layer mapping intermediate features to duration
/// bins. Each output element represents a logit for a duration class.
/// With small weights, output bounds should be proportional to
/// D_INTER * WEIGHT_MAG.
#[test]
fn test_duration_projection_bounds() {
    let def = build_duration_projection(SEQ_LEN, D_INTER, MAX_DUR);
    def.validate().expect("duration projection def validates");

    let bindings = duration_proj_bindings(D_INTER, MAX_DUR, WEIGHT_MAG);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_INTER], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through duration projection");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "duration projection bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, MAX_DUR],
        "duration projection must produce [T, max_dur] output"
    );

    // With D_INTER=8 inputs in [-1, 1] and weight_mag=0.01:
    // output range ~ D_INTER * WEIGHT_MAG = 0.08 per element.
    assert!(
        width < 5.0,
        "duration projection with small weights should have tight bounds, got width={width}"
    );

    eprintln!("Duration projection: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 4: ReLU/Softplus activation bounds
// ===========================================================================

/// ReLU followed by Softplus produces non-negative bounded output.
///
/// ReLU clips negative values to 0, then Softplus = log(1 + exp(x)) smooths
/// near zero. For non-negative input x:
///   - ReLU(x) = x for x >= 0
///   - Softplus(x) = log(1 + exp(x)) in [log(2), x + log(2)] for small x
///
/// Key property: the composition is monotonic and bounded below by log(2).
/// IBP propagates this correctly because both activations are monotonic.
#[test]
fn test_duration_relu_softplus_bounds() {
    let def = build_activation_block(SEQ_LEN);
    def.validate().expect("activation block def validates");

    let bindings = activation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Input includes negative values to exercise ReLU clipping.
    let input = uniform_bounds(&[SEQ_LEN, 1], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ReLU+Softplus");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "activation bounds must be finite: [{lo_min}, {hi_max}]"
    );

    // Softplus is always positive: log(1 + exp(x)) >= log(1) = 0 for ReLU output >= 0.
    // Actually Softplus(0) = log(2) ~ 0.693.
    assert!(
        lo_min >= -1e-4,
        "ReLU+Softplus lower bound {lo_min} should be non-negative (Softplus >= log(2))"
    );

    // For input in [-2, 2], after ReLU: [0, 2], after Softplus: [log(2), log(1+e^2)] ~ [0.693, 2.13].
    assert!(
        hi_max < 10.0,
        "ReLU+Softplus upper bound {hi_max} should be bounded for input in [-2, 2]"
    );

    assert!(
        width > 0.0,
        "activation bounds should have non-zero width, got {width}"
    );
    assert!(
        width < VACUOUS_THRESHOLD,
        "activation bounds width {width} exceeds vacuous threshold {VACUOUS_THRESHOLD}"
    );

    eprintln!("Duration ReLU+Softplus: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

// ===========================================================================
// Test 5: Regulate expansion bounds
// ===========================================================================

/// Expanding phoneme features to frame features preserves per-frame bounds.
///
/// The regulate step repeats each phoneme feature by its predicted duration.
/// Here we model this as a matmul with a binary expansion matrix where each
/// phoneme maps to `expansion_factor` consecutive frames.
///
/// Key property: the binary expansion matrix has entries in {0, 1}, so the
/// output bounds are at most equal to the input bounds (each output frame
/// copies exactly one input phoneme).
#[test]
fn test_duration_regulate_expansion_bounds() {
    let expansion_factor = 3;
    let t_out = SEQ_LEN * expansion_factor;
    let def = build_regulate_expansion(D_INTER, SEQ_LEN, expansion_factor);
    def.validate().expect("regulate expansion def validates");

    let bindings = regulate_expansion_bindings(SEQ_LEN, expansion_factor);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[D_INTER, SEQ_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through regulate expansion");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "regulate expansion bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[D_INTER, t_out],
        "regulate expansion must produce [D, T_out] output"
    );

    // Binary expansion matrix with {0, 1} entries and each row summing to
    // expansion_factor. IBP matmul sums absolute-value contributions:
    // width <= input_width * expansion_factor = 2.0 * 3 = 6.0.
    assert!(
        width < 20.0,
        "regulate expansion width {width} should be bounded by input_width * expansion_factor"
    );

    // Lower bound should be negative (input includes [-1, 1] range).
    assert!(
        lo_min <= 0.0,
        "regulate expansion lower bound {lo_min} should be <= 0 for input in [-1, 1]"
    );

    eprintln!(
        "Duration regulate expansion (factor={expansion_factor}): \
         bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}, \
         output_shape=[{D_INTER}, {t_out}]"
    );
}

// ===========================================================================
// Test 6: Speed scaling bounds
// ===========================================================================

/// Duration * speed_factor produces bounded output for various speed factors.
///
/// Speed scaling is a simple element-wise multiply by a constant factor.
/// IBP handles this exactly: if input in [lo, hi] and speed > 0, then
/// output in [lo * speed, hi * speed].
///
/// Test with speed_factor=1.5 (faster speech, shorter durations in relative terms).
#[test]
fn test_duration_speed_scaling_bounds() {
    let def = build_speed_scaling(SEQ_LEN);
    def.validate().expect("speed scaling def validates");

    let speed_factor = 1.5;
    let bindings = speed_scaling_bindings(speed_factor);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Duration values in [0.5, 3.0] (positive, typical range after softmax+sum).
    let input = nn_verify::BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, 1]), 0.5f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, 1]), 3.0f32),
    )
    .expect("valid duration bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through speed scaling");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "speed scaling bounds must be finite: [{lo_min}, {hi_max}]"
    );

    // Exact interval arithmetic: [0.5, 3.0] * 1.5 = [0.75, 4.5].
    let expected_lo = 0.5 * speed_factor;
    let expected_hi = 3.0 * speed_factor;
    assert!(
        lo_min >= expected_lo - 1e-4,
        "speed scaling lower {lo_min} should be >= {expected_lo}"
    );
    assert!(
        hi_max <= expected_hi + 1e-4,
        "speed scaling upper {hi_max} should be <= {expected_hi}"
    );

    // Width should scale proportionally: (3.0 - 0.5) * 1.5 = 3.75.
    let expected_width = (3.0 - 0.5) * speed_factor;
    assert!(
        (width - expected_width).abs() < 0.01,
        "speed scaling width {width} should be approximately {expected_width}"
    );

    // Test with speed_factor < 1 (slower speech)
    let slow_bindings = speed_scaling_bindings(0.5);
    let slow_graph = tensor_kernel_to_graph(&def, &slow_bindings).expect("slow speed graph");
    let slow_output = slow_graph
        .propagate_ibp(&input)
        .expect("IBP through slow speed scaling");
    assert_bounds_valid(&slow_output);
    let (slow_lo, slow_hi) = bounds_min_max(&slow_output);
    let slow_width = slow_hi - slow_lo;

    // Slower speed should produce tighter bounds (smaller width).
    assert!(
        slow_width < width,
        "slower speed (0.5) width {slow_width} should be < faster speed (1.5) width {width}"
    );

    eprintln!(
        "Duration speed scaling: fast(1.5x)=[{lo_min:.4}, {hi_max:.4}] width={width:.4}, \
         slow(0.5x)=[{slow_lo:.4}, {slow_hi:.4}] width={slow_width:.4}"
    );
}

// ===========================================================================
// Test 7: Full duration predictor
// ===========================================================================

/// End-to-end duration predictor: hidden features -> duration logits.
///
/// Architecture: Linear -> Transpose -> Conv1d+LayerNorm -> Transpose -> Linear -> Softplus.
/// Tests that all stages compose correctly and bounds remain finite and non-vacuous
/// through the full pipeline.
#[test]
fn test_duration_full_predictor_bounds() {
    let def = build_full_duration_predictor(SEQ_LEN, D_HIDDEN, D_INTER, MAX_DUR);
    def.validate()
        .expect("full duration predictor def validates");

    let bindings = full_predictor_bindings(D_HIDDEN, D_INTER, MAX_DUR, WEIGHT_MAG);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, D_HIDDEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full duration predictor");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "full predictor bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, MAX_DUR],
        "full predictor must produce [T, max_dur] output"
    );

    // Softplus at the end ensures non-negative output.
    assert!(
        lo_min >= -1e-4,
        "full predictor lower bound {lo_min} should be non-negative (Softplus output)"
    );

    // With small weights through multiple stages, bounds should stay bounded.
    assert_bounds_width(&output, VACUOUS_THRESHOLD, "full_duration_predictor");

    // Graph structure: the pipeline is 7 op stages — Linear, Transpose, Conv1d,
    // Transpose, LayerNorm, Linear, Softplus. Constant weight/bias inputs fold
    // and do not become graph nodes, so the real node count is 7. (The previous
    // `>= 8` was stale: it predated the LayerNorm-axis layout fix in 9ba87f4d,
    // which relocated a transpose rather than adding a node, leaving the count
    // at 7. That bulk node-count update missed this assertion.)
    assert!(
        graph.num_nodes() >= 7,
        "full predictor graph should have >= 7 nodes, got {}",
        graph.num_nodes()
    );

    eprintln!(
        "Full duration predictor: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}, \
         graph_nodes={}, output_shape={:?}",
        graph.num_nodes(),
        output.lower_upper().0.shape()
    );
}

// ===========================================================================
// Test 8: Variable-length handling
// ===========================================================================

/// Different sequence lengths produce consistent IBP bound characteristics.
///
/// The duration predictor must handle varying phoneme counts (different
/// utterance lengths). We test three lengths (4, 8, 16) and verify:
///   - All produce valid finite bounds
///   - Output widths are consistent (no blow-up at longer sequences)
///   - Softplus non-negativity holds regardless of length
///
/// This property is critical because production utterances range from
/// a few phonemes (short words) to 100+ (full sentences).
#[test]
fn test_duration_variable_length_bounds() {
    let seq_lengths = [4, 8, 16];
    let mut widths = Vec::with_capacity(seq_lengths.len());

    for &seq_len in &seq_lengths {
        let def = build_full_duration_predictor(seq_len, D_HIDDEN, D_INTER, MAX_DUR);
        def.validate()
            .unwrap_or_else(|e| panic!("full predictor T={seq_len} def: {e}"));

        let bindings = full_predictor_bindings(D_HIDDEN, D_INTER, MAX_DUR, WEIGHT_MAG);
        let graph = tensor_kernel_to_graph(&def, &bindings)
            .unwrap_or_else(|e| panic!("full predictor T={seq_len} graph: {e}"));
        let input = uniform_bounds(&[seq_len, D_HIDDEN], 1.0);

        let output = graph
            .propagate_ibp(&input)
            .unwrap_or_else(|e| panic!("IBP through full predictor T={seq_len}: {e}"));
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;

        assert!(
            lo_min.is_finite() && hi_max.is_finite(),
            "T={seq_len}: bounds must be finite [{lo_min}, {hi_max}]"
        );
        assert_eq!(
            output.lower_upper().0.shape(),
            &[seq_len, MAX_DUR],
            "T={seq_len}: output shape must be [T, max_dur]"
        );
        assert!(
            lo_min >= -1e-4,
            "T={seq_len}: lower bound {lo_min} should be non-negative (Softplus)"
        );
        assert!(
            width < VACUOUS_THRESHOLD,
            "T={seq_len}: width {width} exceeds vacuous threshold {VACUOUS_THRESHOLD}"
        );

        eprintln!("  T={seq_len}: bounds=[{lo_min:.4}, {hi_max:.4}], width={width:.4}");
        widths.push((seq_len, width));
    }

    // Consistency check: widths should be similar across lengths.
    // The conv + layernorm pipeline processes each position similarly,
    // so width should not dramatically change with sequence length.
    let min_width = widths.iter().map(|&(_, w)| w).fold(f32::INFINITY, f32::min);
    let max_width = widths
        .iter()
        .map(|&(_, w)| w)
        .fold(f32::NEG_INFINITY, f32::max);

    // Allow up to 10x variation (conv padding effects may cause some differences).
    if min_width > 1e-6 {
        let ratio = max_width / min_width;
        assert!(
            ratio < 10.0,
            "width ratio across lengths {ratio:.2}x exceeds 10x: widths={widths:?}"
        );
        eprintln!(
            "Variable-length consistency: min_width={min_width:.4}, max_width={max_width:.4}, \
             ratio={ratio:.2}x"
        );
    }
}
