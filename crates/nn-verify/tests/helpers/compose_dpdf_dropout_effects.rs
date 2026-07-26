// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for dropout and stochastic depth effects on bounds.
//!
//! Verifies IBP and CROWN bound propagation through dropout-related patterns
//! used across dpdf models. At inference (eval mode), dropout is disabled and
//! stochastic depth survival alpha = 1.0, so the graph is deterministic. These
//! tests verify that bounds propagation correctly handles the inference-time
//! identity semantics and the compositional effects on output width.
//!
//! 1.  **Dropout disabled (eval mode) preserves bounds**: identity at inference (IBP)
//! 2.  **Stochastic depth alpha=0 preserves input**: skip branch only (IBP)
//! 3.  **Stochastic depth alpha=1 is identity**: full residual passthrough (IBP)
//! 4.  **Dropout mask scaling at inference**: 1/(1-p) scaling preserved (IBP)
//! 5.  **Attention dropout disabled at eval**: MHA without dropout (IBP + CROWN)
//! 6.  **FFN dropout disabled at eval**: Linear->ReLU->Linear identity scale (IBP)
//! 7.  **Layer drop (skip entire layer) at eval**: layer always active (IBP)
//! 8.  **Bounds with vs without dropout comparison**: scale factor effect (IBP)
//! 9.  **Residual with stochastic depth**: x + alpha * FFN(x) (IBP)
//! 10. **CROWN tightness without dropout**: CROWN vs IBP on clean path (CROWN)
//! 11. **Monotone tightening**: smaller input -> tighter output without dropout (IBP)
//! 12. **Dropout probability effect analysis**: different scale factors (IBP)
//! 13. **Deep model without dropout**: 4-layer FFN, no stochastic scaling (IBP)
//! 14. **Multi-head attention dropout**: MHA with identity dropout scale (IBP)
//! 15. **Full block eval mode pipeline**: Attn->Norm->FFN->Norm all clean (IBP + CROWN)
//!
//! Architecture references:
//! - Dropout (Srivastava et al., 2014): inverted dropout scaling at training
//! - Stochastic Depth (Huang et al., 2016): layer-wise skip probability
//! - DropPath (Larsson et al., 2017): drop entire residual branch
//! - Pre-norm transformer: dropout after attention and FFN sub-layers
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, FFN_DIM=128
//!
//! Part of #4060: Compose tests for dropout and stochastic depth effects on bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 4;
const HIDDEN_DIM: usize = 64;
const FFN_DIM: usize = 128;
const WEIGHT_MAG: f32 = 0.02;
const NUM_HEADS: usize = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a simple FFN block: Linear -> ReLU -> Linear.
///
/// Input shape: `[seq_len, hidden_dim]`.
/// Output shape: `[seq_len, hidden_dim]`.
fn build_ffn_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> nn_dsl::TensorNodeId {
    let ffn_shape = [seq_len, ffn_dim];
    let out_shape = [seq_len, hidden_dim];

    let w1 = b.add_input(&format!("{prefix}_w1"), &[ffn_dim, hidden_dim]);
    let w2 = b.add_input(&format!("{prefix}_w2"), &[hidden_dim, ffn_dim]);

    let h = b.add_linear(input, w1, None, &ffn_shape);
    let h = b.add_relu(h, &ffn_shape);
    b.add_linear(h, w2, None, &out_shape)
}

/// Push FFN weight bindings (w1, w2) for given dimensions.
fn push_ffn_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    hidden_dim: usize,
    ffn_dim: usize,
    weight_mag: f32,
) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ffn_dim, hidden_dim]),
        weight_mag,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[hidden_dim, ffn_dim]),
        weight_mag,
    )));
}

/// Apply a scalar scale factor (broadcast multiply) to a tensor.
///
/// Models the effect of dropout scaling (1/(1-p)) or stochastic depth alpha.
fn apply_scale(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    scale_name: &str,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let scale = b.add_input(scale_name, &[1]);
    let scale_bc = b.add_broadcast(scale, shape);
    b.add_binary_mul(input, scale_bc, shape)
}

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. Dropout disabled (eval mode) preserves bounds
// ===========================================================================

/// At eval mode, dropout is an identity (scale = 1.0). Verify that
/// passing through a scale=1.0 multiply preserves input bounds exactly.
#[test]
fn test_dropout_disabled_eval_preserves_bounds_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_dropout_eval_identity");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Dropout at eval = identity = multiply by 1.0
    let out = apply_scale(&mut b, input, "dropout_scale", &shape);
    let def = b.build(out).expect("valid dropout eval kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        // scale = 1.0 (dropout disabled)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1.0f32)),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Dropout eval identity IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // scale=1.0 should preserve bounds: output ≈ input
    let eps = 1e-5;
    assert!(
        (lo_min - (-1.0)).abs() < eps,
        "dropout eval lower should be ≈ -1.0, got {lo_min}"
    );
    assert!(
        (hi_max - 1.0).abs() < eps,
        "dropout eval upper should be ≈ 1.0, got {hi_max}"
    );
}

// ===========================================================================
// 2. Stochastic depth alpha=0 preserves input
// ===========================================================================

/// Stochastic depth with alpha=0 means the FFN branch is dropped:
/// output = x + 0 * FFN(x) = x. Verify bounds equal input bounds.
#[test]
fn test_stochastic_depth_alpha0_preserves_input_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_stoch_depth_alpha0");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // FFN path
    let ffn_out = build_ffn_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Scale by alpha=0 (drop the FFN branch entirely)
    let scaled = apply_scale(&mut b, ffn_out, "alpha", &shape);

    // Residual: x + 0 * FFN(x) = x
    let out = b.add_binary_add(input, scaled, &shape);
    let def = b.build(out).expect("valid stochastic depth alpha=0 kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // alpha = 0.0 (fully dropped)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Stochastic depth alpha=0 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // alpha=0 -> output=input, so bounds should be ≈ [-1, 1]
    let eps = 1e-4;
    assert!(
        (lo_min - (-1.0)).abs() < eps,
        "alpha=0 lower should be ≈ -1.0, got {lo_min}"
    );
    assert!(
        (hi_max - 1.0).abs() < eps,
        "alpha=0 upper should be ≈ 1.0, got {hi_max}"
    );
}

// ===========================================================================
// 3. Stochastic depth alpha=1 is identity (full residual passthrough)
// ===========================================================================

/// Stochastic depth with alpha=1 means full FFN is active:
/// output = x + 1 * FFN(x). Same as a normal residual block.
#[test]
fn test_stochastic_depth_alpha1_full_residual_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_stoch_depth_alpha1");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // FFN path
    let ffn_out = build_ffn_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Scale by alpha=1 (no drop)
    let scaled = apply_scale(&mut b, ffn_out, "alpha", &shape);

    // Residual: x + 1 * FFN(x)
    let out = b.add_binary_add(input, scaled, &shape);
    let def = b.build(out).expect("valid stochastic depth alpha=1 kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // alpha = 1.0 (fully active)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        1.0f32,
    )));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Stochastic depth alpha=1 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // With alpha=1 the FFN contributes, so bounds should be wider than input
    let width = hi_max - lo_min;
    assert!(
        width >= 2.0,
        "residual should widen bounds beyond input ±1, got width={width}"
    );
}

// ===========================================================================
// 4. Dropout mask scaling at inference: 1/(1-p) scaling preserved
// ===========================================================================

/// Inverted dropout scales activations by 1/(1-p) during training so that
/// at inference time no scaling is needed (scale=1). This test verifies
/// that different scaling factors correctly propagate through bounds.
#[test]
fn test_dropout_mask_scaling_inference_ibp() {
    // At inference: scale=1.0 (identity). At training with p=0.5: scale=2.0.
    // We verify the scale factor linearly widens bounds.
    let mut b = TensorBlockBuilder::new("dpdf_dropout_scaling");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let out = apply_scale(&mut b, input, "scale", &shape);
    let def = b.build(out).expect("valid dropout scaling kernel");

    // Test with scale = 1/(1-0.5) = 2.0 (training-time inverted dropout)
    let bindings_train = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 2.0f32)),
    ];
    let graph_train = tensor_kernel_to_graph(&def, &bindings_train).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output_train = graph_train.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output_train);

    let (lo_train, hi_train) = bounds_min_max(&output_train);
    eprintln!("Dropout scale=2.0 IBP: bounds=[{lo_train:.6}, {hi_train:.6}]");

    // scale=2 on [-1,1] should give [-2,2]
    let eps = 1e-4;
    assert!(
        (lo_train - (-2.0)).abs() < eps,
        "scale=2 lower should be ≈ -2.0, got {lo_train}"
    );
    assert!(
        (hi_train - 2.0).abs() < eps,
        "scale=2 upper should be ≈ 2.0, got {hi_train}"
    );
}

// ===========================================================================
// 5. Attention dropout disabled at eval: MHA without dropout
// ===========================================================================

/// At eval, attention dropout is disabled. Verify that MHA produces
/// finite, valid bounds without any dropout scaling.
#[test]
fn test_attention_dropout_disabled_eval_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_attn_dropout_eval");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    // MHA with no dropout (eval mode)
    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");
    let def = b.build(attn_out).expect("valid attention kernel");

    let w = |shape: &[usize]| {
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
    };
    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // q_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // k_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // v_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // o_w
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Attention dropout eval IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_attention_dropout_disabled_eval_crown() {
    let mut b = TensorBlockBuilder::new("dpdf_attn_dropout_eval_crown");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");
    let def = b.build(attn_out).expect("valid attention kernel");

    let w = |shape: &[usize]| {
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
    };
    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[HIDDEN_DIM, HIDDEN_DIM]),
        w(&[HIDDEN_DIM, HIDDEN_DIM]),
        w(&[HIDDEN_DIM, HIDDEN_DIM]),
        w(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Attention dropout eval CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 6. FFN dropout disabled at eval: Linear->ReLU->Linear identity scale
// ===========================================================================

/// At eval, FFN dropout is disabled. Verify that an FFN block with a
/// trailing scale=1.0 (representing disabled dropout) produces valid bounds.
#[test]
fn test_ffn_dropout_disabled_eval_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_ffn_dropout_eval");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // FFN + identity dropout scale
    let ffn_out = build_ffn_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let out = apply_scale(&mut b, ffn_out, "dropout_scale", &shape);
    let def = b.build(out).expect("valid FFN dropout kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // dropout_scale = 1.0 (eval mode)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        1.0f32,
    )));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("FFN dropout eval IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

// ===========================================================================
// 7. Layer drop (skip entire layer) at eval: layer always active
// ===========================================================================

/// Layer drop (DropPath) skips entire residual blocks during training.
/// At eval, all layers are active. Model: x + layer(x) for each of 2 layers.
#[test]
fn test_layer_drop_eval_all_active_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_layer_drop_eval");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Layer 0: x + FFN_0(x) (no drop at eval)
    let ffn0 = build_ffn_block(&mut b, input, "ffn0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let h = b.add_binary_add(input, ffn0, &shape);

    // Layer 1: h + FFN_1(h) (no drop at eval)
    let ffn1 = build_ffn_block(&mut b, h, "ffn1", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let out = b.add_binary_add(h, ffn1, &shape);

    let def = b.build(out).expect("valid layer drop eval kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    push_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Layer drop eval IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. Bounds with vs without dropout comparison: scale factor effect
// ===========================================================================

/// Compare bounds of an FFN block with scale=1.0 vs scale=0.5.
/// Scale < 1.0 should produce tighter or equal bounds.
#[test]
fn test_dropout_scale_comparison_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_dropout_scale_compare");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let ffn_out = build_ffn_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let out = apply_scale(&mut b, ffn_out, "scale", &shape);
    let def = b.build(out).expect("valid scale comparison kernel");

    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // scale = 1.0 (no dropout effect)
    let mut bindings_full = vec![TensorParamBinding::Variable];
    push_ffn_bindings(&mut bindings_full, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings_full.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        1.0f32,
    )));
    let graph_full = tensor_kernel_to_graph(&def, &bindings_full).expect("graph full");
    let output_full = graph_full.propagate_ibp(&input_bounds).expect("IBP full");
    assert_bounds_valid(&output_full);
    let width_full = bound_width(&output_full);

    // scale = 0.5 (simulate partial dropout effect)
    let mut bindings_half = vec![TensorParamBinding::Variable];
    push_ffn_bindings(&mut bindings_half, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings_half.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.5f32,
    )));
    let graph_half = tensor_kernel_to_graph(&def, &bindings_half).expect("graph half");
    let output_half = graph_half.propagate_ibp(&input_bounds).expect("IBP half");
    assert_bounds_valid(&output_half);
    let width_half = bound_width(&output_half);

    eprintln!(
        "Dropout scale comparison IBP: full_width={width_full:.6}, half_width={width_half:.6}"
    );
    // scale=0.5 should produce tighter bounds than scale=1.0
    assert!(
        width_half <= width_full + 1e-4,
        "scale=0.5 should be tighter: half={width_half}, full={width_full}"
    );
}

// ===========================================================================
// 9. Residual with stochastic depth: x + alpha * FFN(x)
// ===========================================================================

/// Standard stochastic depth residual: x + alpha * FFN(x).
/// At eval, alpha=1.0 (survival probability). Verify finite bounds.
#[test]
fn test_residual_stochastic_depth_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_residual_stoch_depth");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // FFN path
    let ffn_out = build_ffn_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Scale by survival probability
    let scaled = apply_scale(&mut b, ffn_out, "alpha", &shape);

    // Residual: x + alpha * FFN(x)
    let out = b.add_binary_add(input, scaled, &shape);
    let def = b
        .build(out)
        .expect("valid residual stochastic depth kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // alpha = 0.8 (80% survival)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.8f32,
    )));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Residual stochastic depth IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. CROWN tightness without dropout
// ===========================================================================

/// Verify CROWN produces tighter-than-IBP bounds on an FFN block
/// with no dropout (clean inference path).
#[test]
fn test_crown_tightness_no_dropout() {
    let mut b = TensorBlockBuilder::new("dpdf_crown_no_dropout");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let ffn_out = build_ffn_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let def = b.build(ffn_out).expect("valid FFN kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("CROWN no dropout: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 11. Monotone tightening: smaller input -> tighter output without dropout
// ===========================================================================

/// Smaller input bounds should produce tighter output bounds through
/// a dropout-free FFN. This is a fundamental soundness property.
#[test]
fn test_monotone_tightening_no_dropout_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_monotone_no_dropout");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let ffn_out = build_ffn_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let def = b.build(ffn_out).expect("valid FFN kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input: ±1.0
    let wide_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);
    let wide_width = bound_width(&wide_output);

    // Tight input: ±0.1
    let tight_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let tight_output = graph.propagate_ibp(&tight_input).expect("IBP tight");
    assert_bounds_valid(&tight_output);
    let tight_width = bound_width(&tight_output);

    eprintln!("Monotone tightening: wide_width={wide_width:.6}, tight_width={tight_width:.6}");
    assert!(
        tight_width <= wide_width + 1e-6,
        "tight input should produce tighter output: wide={wide_width}, tight={tight_width}"
    );
}

// ===========================================================================
// 12. Dropout probability effect analysis: different scale factors
// ===========================================================================

/// Verify that higher dropout scale factors (corresponding to higher
/// dropout probabilities during training) produce proportionally wider bounds.
#[test]
fn test_dropout_probability_effect_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_dropout_prob_effect");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let ffn_out = build_ffn_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let out = apply_scale(&mut b, ffn_out, "scale", &shape);
    let def = b.build(out).expect("valid dropout prob kernel");

    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Test three scale factors: 1.0 (p=0), 1.25 (p=0.2), 2.0 (p=0.5)
    let scales = [1.0f32, 1.25, 2.0];
    let mut widths = Vec::new();

    for &scale in &scales {
        let mut bindings = vec![TensorParamBinding::Variable];
        push_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[1]),
            scale,
        )));

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let output = graph.propagate_ibp(&input_bounds).expect("IBP");
        assert_bounds_valid(&output);
        let width = bound_width(&output);
        eprintln!("Dropout scale={scale:.2} IBP: width={width:.6}");
        widths.push(width);
    }

    // Widths should be monotonically non-decreasing with scale
    for i in 1..widths.len() {
        assert!(
            widths[i] >= widths[i - 1] - 1e-4,
            "scale {} width {} should be >= scale {} width {}",
            scales[i],
            widths[i],
            scales[i - 1],
            widths[i - 1]
        );
    }
}

// ===========================================================================
// 13. Deep model without dropout: 4-layer FFN, no stochastic scaling
// ===========================================================================

/// Verify that a 4-layer deep FFN stack with residual connections
/// and no dropout produces finite bounds. This tests the clean
/// inference path through deep models.
#[test]
fn test_deep_model_no_dropout_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_deep_no_dropout");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut h = input;
    for i in 0..4 {
        let ffn = build_ffn_block(&mut b, h, &format!("ffn{i}"), SEQ_LEN, HIDDEN_DIM, FFN_DIM);
        h = b.add_binary_add(h, ffn, &shape);
    }

    let def = b.build(h).expect("valid deep no-dropout kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..4 {
        push_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Deep 4-layer no dropout IBP: width={width:.6}");
    assert!(width.is_finite(), "depth-4 output width must be finite");
}

// ===========================================================================
// 14. Multi-head attention dropout: MHA with identity dropout scale
// ===========================================================================

/// In a full transformer block, attention dropout is applied after
/// softmax in MHA. At eval, this is identity. Verify MHA + residual
/// with scale=1.0 (no dropout) produces valid bounds.
#[test]
fn test_mha_dropout_identity_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mha_dropout_identity");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    // MHA (no internal dropout at eval)
    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // Post-attention dropout scale = 1.0 (identity at eval)
    let scaled = apply_scale(&mut b, attn_out, "attn_dropout_scale", &shape);

    // Residual: x + dropout(attn(x))
    let out = b.add_binary_add(input, scaled, &shape);
    let def = b.build(out).expect("valid MHA dropout kernel");

    let w =
        |s: &[usize]| TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(s), WEIGHT_MAG));
    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // q_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // k_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // v_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // o_w
        // attn_dropout_scale = 1.0
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1.0f32)),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MHA dropout identity IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. Full block eval mode pipeline: Attn->Norm->FFN->Norm all clean
// ===========================================================================

/// Full transformer decoder block at eval mode: no dropout anywhere.
/// MHA -> residual -> RMSNorm -> FFN -> residual -> RMSNorm.
/// All dropout scales are 1.0 (identity).
fn build_full_block_eval_kernel() -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_full_block_eval");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Self-attention
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // Post-attention dropout (identity at eval)
    let attn_scaled = apply_scale(&mut b, attn_out, "attn_drop", &shape);

    // First residual
    let h = b.add_binary_add(input, attn_scaled, &shape);

    // RMSNorm before FFN
    let eps1 = b.add_input("eps1", &[1]);
    let norm_w1 = b.add_input("norm_w1", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(h, eps1, 1, norm_w1, &shape);

    // FFN
    let ffn_out = build_ffn_block(&mut b, normed, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Post-FFN dropout (identity at eval)
    let ffn_scaled = apply_scale(&mut b, ffn_out, "ffn_drop", &shape);

    // Second residual
    let h2 = b.add_binary_add(h, ffn_scaled, &shape);

    // Final RMSNorm
    let eps2 = b.add_input("eps2", &[1]);
    let norm_w2 = b.add_input("norm_w2", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(h2, eps2, 1, norm_w2, &shape);

    b.build(out).expect("valid full block eval kernel")
}

fn full_block_eval_bindings() -> Vec<TensorParamBinding> {
    let w =
        |s: &[usize]| TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(s), WEIGHT_MAG));
    let mut bindings = vec![
        TensorParamBinding::Variable,
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // q_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // k_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // v_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // o_w
        // attn_drop = 1.0
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1.0f32)),
        // eps1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        // norm_w1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
    ];
    push_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // ffn_drop = 1.0
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        1.0f32,
    )));
    // eps2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        1e-5f32,
    )));
    // norm_w2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        1.0f32,
    )));
    bindings
}

#[test]
fn test_full_block_eval_pipeline_ibp() {
    let def = build_full_block_eval_kernel();
    let bindings = full_block_eval_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full block eval IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_full_block_eval_pipeline_crown() {
    let def = build_full_block_eval_kernel();
    let bindings = full_block_eval_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full block eval CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
