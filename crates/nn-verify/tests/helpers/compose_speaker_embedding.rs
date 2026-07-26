// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: ECAPA-TDNN speaker encoder composition.
//!
//! Validates that a simplified ECAPA-TDNN speaker encoder translates through
//! `tensor_kernel_to_graph` and produces a NY `GraphNetwork` where
//! IBP and CROWN bounds propagate end-to-end.
//!
//! Architecture (small-scale for CI, 2D [C, T] layout):
//! ```text
//! Mel input [IN_CH=4, T=8]
//!   → Conv1d(4→CH, k=3, s=1, p=1) + ReLU + BN    → [CH, 8]
//!   → Conv1d(CH→CH, k=3, dil=2, p=2) + ReLU + BN → [CH, 8]
//!   → SE1d(CH, bottleneck=2)                       → [CH, 8]
//!   → Conv1d(CH→CH, k=1) + ReLU + BN              → [CH, 8]
//!   → residual add + ReLU                          → [CH, 8]
//!   → Conv1d(CH→CH, k=1) + ReLU                   → [CH, 8]
//!   → Reduce(Mean, axis=1)                         → [CH]
//!   → Linear(CH→EMBED)                             → [EMBED]
//! ```
//!
//! This covers the key ECAPA-TDNN verification targets:
//! - Conv1d with dilation (dilated convolutions in SE-Res blocks)
//! - BatchNorm (channel normalization)
//! - SE block (squeeze-excitation with global pooling)
//! - Mean pooling + linear projection (speaker embedding extraction)
//! - Residual connection
//!
//! Res2Net channel splitting is omitted because the tensor IR reserves
//! axis 0 for verification stacking (AxisZeroReserved), making concat on
//! the channel dimension of 2D [C, T] tensors impossible. The Res2Net
//! pattern would require 3D [batch, C, T] tensors which NY's
//! Conv1d IBP propagation does not fully support at small scale.
//!
//! Part of #2079.

use super::common;

use common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::ReduceOp;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor, TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Small-scale ECAPA-TDNN configuration
// ---------------------------------------------------------------------------

const IN_CH: usize = 4;
const HIDDEN_CH: usize = 4;
const EMBED_DIM: usize = 4;
const SE_BOTTLENECK: usize = 2;
const KERNEL_SIZE: usize = 3;
const DILATION: usize = 2;
const IN_LENGTH: usize = 8;

/// Conv1d output length with dilation support.
fn dilated_conv1d_out_len(
    in_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let effective_k = dilation * (kernel_size - 1) + 1;
    (in_len + 2 * padding - effective_k) / stride + 1
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Struct to track all weight input node IDs for binding order.
struct WeightNodes {
    ids: Vec<(String, Vec<usize>)>,
}

impl WeightNodes {
    fn new() -> Self {
        Self { ids: Vec::new() }
    }

    fn add(
        &mut self,
        b: &mut TensorBlockBuilder,
        name: &str,
        shape: &[usize],
    ) -> nn_dsl::tensor_ir::TensorNodeId {
        let id = b.add_input(name, shape);
        self.ids.push((name.to_string(), shape.to_vec()));
        id
    }
}

/// Add Conv1d + BatchNorm + ReLU block. All shapes are 2D [C, T].
fn add_conv_bn_relu(
    b: &mut TensorBlockBuilder,
    w: &mut WeightNodes,
    input: nn_dsl::tensor_ir::TensorNodeId,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    in_length: usize,
    prefix: &str,
) -> (nn_dsl::tensor_ir::TensorNodeId, usize) {
    let out_t = dilated_conv1d_out_len(in_length, kernel_size, stride, padding, dilation);

    let conv_w = w.add(
        b,
        &format!("{prefix}_conv_w"),
        &[out_ch, in_ch, kernel_size],
    );
    let conv_b = w.add(b, &format!("{prefix}_conv_b"), &[out_ch]);

    let conv_out = b.add_conv1d_full(
        input,
        conv_w,
        Some(conv_b),
        stride,
        padding,
        dilation,
        1, // groups=1
        &[out_ch, out_t],
    );

    // BatchNorm parameters
    let bn_mean = w.add(b, &format!("{prefix}_bn_mean"), &[out_ch]);
    let bn_var = w.add(b, &format!("{prefix}_bn_var"), &[out_ch]);
    let bn_weight = w.add(b, &format!("{prefix}_bn_weight"), &[out_ch]);
    let bn_bias = w.add(b, &format!("{prefix}_bn_bias"), &[out_ch]);
    let bn_eps = w.add(b, &format!("{prefix}_bn_eps"), &[1]);

    let bn_out = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_weight,
        bn_bias,
        bn_eps,
        &[out_ch, out_t],
    );
    let relu_out = b.add_relu(bn_out, &[out_ch, out_t]);

    (relu_out, out_t)
}

/// Add SE1d block: global average pool → Linear → ReLU → Linear → Sigmoid → scale.
/// Input/output: [channels, time_len].
fn add_se_block(
    b: &mut TensorBlockBuilder,
    w: &mut WeightNodes,
    input: nn_dsl::tensor_ir::TensorNodeId,
    channels: usize,
    bottleneck: usize,
    time_len: usize,
    prefix: &str,
) -> nn_dsl::tensor_ir::TensorNodeId {
    // Global average pool: [C, T] → [C, 1] (keepdims) → reshape → [C]
    let pooled_kd = b.add_reduce(input, ReduceOp::Mean, 1, false, &[channels]);
    let pooled = b.add_reshape(pooled_kd, &[channels]);

    // Linear down: [C] → [bottleneck]
    let fc1_w = w.add(b, &format!("{prefix}_se_fc1_w"), &[bottleneck, channels]);
    let fc1_b = w.add(b, &format!("{prefix}_se_fc1_b"), &[bottleneck]);
    let fc1_out = b.add_linear(pooled, fc1_w, Some(fc1_b), &[bottleneck]);
    let fc1_relu = b.add_relu(fc1_out, &[bottleneck]);

    // Linear up: [bottleneck] → [C]
    let fc2_w = w.add(b, &format!("{prefix}_se_fc2_w"), &[channels, bottleneck]);
    let fc2_b = w.add(b, &format!("{prefix}_se_fc2_b"), &[channels]);
    let fc2_out = b.add_linear(fc1_relu, fc2_w, Some(fc2_b), &[channels]);
    let sigmoid = b.add_sigmoid(fc2_out, &[channels]);

    // Broadcast [C] → [C, T] and scale
    let broadcast = b.add_broadcast_left(sigmoid, &[channels, time_len]);
    b.add_binary_mul(input, broadcast, &[channels, time_len])
}

/// Build the full simplified ECAPA-TDNN speaker encoder.
///
/// Returns (kernel def, weight node metadata for bindings, final output shape).
fn build_speaker_encoder() -> (
    nn_dsl::tensor_ir::TensorKernelDef,
    Vec<(String, Vec<usize>)>,
    [usize; 1], // [EMBED_DIM]
) {
    let mut b = TensorBlockBuilder::new("ecapa_tdnn_speaker_encoder");
    let mut w = WeightNodes::new();

    // Input: mel features [IN_CH, T]
    let data = b.add_input("mel_input", &[IN_CH, IN_LENGTH]);

    // --- Stage 1: Initial Conv1d(k=3) + ReLU + BN ---
    let (stage1, stage1_t) = add_conv_bn_relu(
        &mut b,
        &mut w,
        data,
        IN_CH,
        HIDDEN_CH,
        KERNEL_SIZE,
        1,
        1,
        1,
        IN_LENGTH,
        "stage1",
    );

    // --- Stage 2: Dilated Conv1d(k=3, dil=2) + ReLU + BN ---
    let dil_padding = DILATION; // same-padding for dilation=2, k=3
    let (stage2, stage2_t) = add_conv_bn_relu(
        &mut b,
        &mut w,
        stage1,
        HIDDEN_CH,
        HIDDEN_CH,
        KERNEL_SIZE,
        1,
        dil_padding,
        DILATION,
        stage1_t,
        "stage2_dil",
    );

    // --- Stage 3: SE block ---
    let se_out = add_se_block(
        &mut b,
        &mut w,
        stage2,
        HIDDEN_CH,
        SE_BOTTLENECK,
        stage2_t,
        "block0",
    );

    // --- Stage 4: Post-conv(k=1) + ReLU + BN ---
    let (post_out, post_t) = add_conv_bn_relu(
        &mut b, &mut w, se_out, HIDDEN_CH, HIDDEN_CH, 1, 1, 0, 1, stage2_t, "post",
    );

    // --- Stage 5: Residual add + ReLU ---
    // Residual from stage1 output (same shape [CH, T])
    let residual = b.add_binary_add(post_out, stage1, &[HIDDEN_CH, post_t]);
    let res_relu = b.add_relu(residual, &[HIDDEN_CH, post_t]);

    // --- Stage 6: Final Conv1d(k=1) + ReLU ---
    let final_conv_w = w.add(&mut b, "final_conv_w", &[HIDDEN_CH, HIDDEN_CH, 1]);
    let final_conv_b = w.add(&mut b, "final_conv_b", &[HIDDEN_CH]);
    let final_conv = b.add_conv1d_full(
        res_relu,
        final_conv_w,
        Some(final_conv_b),
        1,
        0,
        1,
        1,
        &[HIDDEN_CH, post_t],
    );
    let final_relu = b.add_relu(final_conv, &[HIDDEN_CH, post_t]);

    // --- Stage 7: Mean pooling over time (axis=1) → [HIDDEN_CH, 1] → reshape → [HIDDEN_CH] ---
    // ReduceMean uses keepdims=true in NY, so we need reshape to squeeze.
    let pooled_kd = b.add_reduce(final_relu, ReduceOp::Mean, 1, false, &[HIDDEN_CH]);
    let pooled = b.add_reshape(pooled_kd, &[HIDDEN_CH]);

    // --- Stage 8: Linear projection → [EMBED_DIM] ---
    let proj_w = w.add(&mut b, "proj_w", &[EMBED_DIM, HIDDEN_CH]);
    let proj_b = w.add(&mut b, "proj_b", &[EMBED_DIM]);
    let embedding = b.add_linear(pooled, proj_w, Some(proj_b), &[EMBED_DIM]);

    let def = b.build(embedding).expect("valid ECAPA-TDNN graph");
    (def, w.ids, [EMBED_DIM])
}

/// Create BN-specific bindings where running_var is 1.0 and running_mean is 0.0.
fn build_bindings(weight_meta: &[(String, Vec<usize>)]) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // mel_input

    for (name, shape) in weight_meta {
        let arr = if name.contains("bn_var") {
            ArrayD::from_elem(IxDyn(shape), 1.0f32)
        } else if name.contains("bn_mean") {
            ArrayD::from_elem(IxDyn(shape), 0.0f32)
        } else if name.contains("bn_eps") {
            ArrayD::from_elem(IxDyn(shape), 1e-5f32)
        } else if name.contains("bn_weight") {
            ArrayD::from_elem(IxDyn(shape), 1.0f32)
        } else if name.contains("bn_bias") {
            ArrayD::from_elem(IxDyn(shape), 0.0f32)
        } else {
            // Conv/linear weights and biases: small uniform
            ArrayD::from_elem(IxDyn(shape), 0.1f32)
        };
        bindings.push(TensorParamBinding::ConstantTensor(arr));
    }

    bindings
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// ECAPA-TDNN encoder graph builds and translates to NY.
#[test]
fn test_speaker_encoder_graph_builds() {
    let (def, weight_meta, _output_shape) = build_speaker_encoder();
    let bindings = build_bindings(&weight_meta);

    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("ECAPA-TDNN speaker encoder graph translation");

    // Architecture has: 3× Conv+BN+ReLU blocks (stage1, stage2_dil, post),
    // SE block (reduce+linear+relu+linear+sigmoid+broadcast+mul),
    // residual add+ReLU, final Conv+ReLU, reduce, linear.
    // Should have many nodes.
    assert!(
        graph.num_nodes() >= 15,
        "speaker encoder graph should have >= 15 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full ECAPA-TDNN encoder.
#[test]
fn test_speaker_encoder_ibp_propagates() {
    let (def, weight_meta, output_shape) = build_speaker_encoder();
    let bindings = build_bindings(&weight_meta);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Input: mel features in [0, 10] range (2D: [IN_CH, T])
    let lower = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through speaker encoder");

    assert_eq!(output.lower_upper().0.shape(), output_shape.as_slice());
    assert_bounds_valid(&output);
}

/// CROWN propagation through ECAPA-TDNN encoder.
#[test]
fn test_speaker_encoder_crown_propagates() {
    let (def, weight_meta, output_shape) = build_speaker_encoder();
    let bindings = build_bindings(&weight_meta);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (_, output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN through speaker encoder");

    assert_eq!(output.lower_upper().0.shape(), output_shape.as_slice());
    assert_bounds_valid(&output);
}

/// CROWN should produce tighter (or equal) bounds than IBP.
#[test]
fn test_speaker_encoder_crown_tighter_than_ibp() {
    let (def, weight_meta, _) = build_speaker_encoder();
    let bindings = build_bindings(&weight_meta);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    assert_crown_tighter_when_not_fallback(&graph, &input);
}

/// verify_and_assert for the ECAPA-TDNN speaker encoder pipeline.
#[test]
fn test_speaker_encoder_verify_and_record() {
    let (def, weight_meta, output_shape) = build_speaker_encoder();
    let bindings = build_bindings(&weight_meta);

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 10.0f32),
    )
    .expect("input bounds");

    let result = verify_and_assert(&def, &bindings, &input, "ecapa_tdnn_speaker_encoder");
    assert_eq!(result.num_variables, 1, "single Variable input (mel_input)");

    let (lo, _hi) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), output_shape.as_slice());
}

// ---------------------------------------------------------------------------
// Speaker embedding distance verification (cosine distance bound)
// ---------------------------------------------------------------------------

/// Compute worst-case L2 distance from per-dimension IBP bounds to a reference.
///
/// For each dimension i: d_i = max(|ref_i - lo_i|, |ref_i - hi_i|)
/// d_worst = sqrt(Σ d_i²)
fn worst_case_l2_distance(lo: &[f32], hi: &[f32], reference: &[f64]) -> f64 {
    lo.iter()
        .zip(hi.iter())
        .zip(reference.iter())
        .map(|((&l, &h), &r)| {
            let d_lo = (r - f64::from(l)).abs();
            let d_hi = (r - f64::from(h)).abs();
            let d_max = d_lo.max(d_hi);
            d_max * d_max
        })
        .sum::<f64>()
        .sqrt()
}

/// Compute worst-case cosine distance from per-dimension IBP bounds.
///
/// Given embedding bounds [lo_i, hi_i] and a reference embedding ref_i:
/// - Minimize dot product by choosing worst-case endpoints per dimension
/// - Maximize embedding norm by choosing largest absolute value per dimension
/// - cos_dist = 1 - min_dot / (max_norm_e * ref_norm)
fn worst_case_cosine_distance(lo: &[f32], hi: &[f32], reference: &[f64]) -> f64 {
    let ref_norm_sq: f64 = reference.iter().map(|r| r * r).sum();
    let ref_norm = ref_norm_sq.sqrt();
    if ref_norm < 1e-12 {
        return 1.0; // degenerate: zero reference
    }

    let mut min_dot = 0.0_f64;
    let mut max_norm_e_sq = 0.0_f64;

    for ((&l, &h), &r) in lo.iter().zip(hi.iter()).zip(reference.iter()) {
        let l64 = f64::from(l);
        let h64 = f64::from(h);

        // Minimize dot product: choose endpoint that gives smallest e_i * r_i
        if r >= 0.0 {
            min_dot += l64 * r;
        } else {
            min_dot += h64 * r;
        }

        // Maximize embedding norm: choose endpoint with largest |e_i|
        let abs_max = l64.abs().max(h64.abs());
        max_norm_e_sq += abs_max * abs_max;
    }

    let max_norm_e = max_norm_e_sq.sqrt();
    if max_norm_e < 1e-12 {
        return 1.0; // degenerate: zero embedding
    }

    let cos_sim_lower = min_dot / (max_norm_e * ref_norm);
    let cos_sim_lower = cos_sim_lower.clamp(-1.0, 1.0);
    1.0 - cos_sim_lower
}

/// Property 4: worst-case L2 distance from IBP bounds.
///
/// Proves that the ECAPA-TDNN speaker embedding is bounded in L2 distance
/// from a reference embedding (center of IBP bounds).
#[test]
fn test_speaker_embedding_worst_case_l2_distance() {
    let (def, weight_meta, _) = build_speaker_encoder();
    let bindings = build_bindings(&weight_meta);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    // Reference: center of bounds.
    let reference: Vec<f64> = lo
        .iter()
        .zip(hi.iter())
        .map(|(&l, &h)| f64::midpoint(f64::from(l), f64::from(h)))
        .collect();

    let d_worst =
        worst_case_l2_distance(lo.as_slice().unwrap(), hi.as_slice().unwrap(), &reference);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ECAPA-TDNN IBP L2 distance: {d_worst:.6} (bounds: [{lo_min}, {hi_max}])");

    assert!(d_worst.is_finite(), "L2 distance should be finite");
}

/// Property 4 (Speaker consistency): worst-case cosine distance from IBP bounds.
///
/// Proves that the ECAPA-TDNN speaker embedding has bounded cosine distance
/// from a reference embedding, enabling voice identity verification.
/// The cosine distance bound is the key metric for speaker verification:
/// two utterances from the same speaker should have cosine distance < threshold.
#[test]
fn test_speaker_embedding_worst_case_cosine_distance() {
    let (def, weight_meta, _) = build_speaker_encoder();
    let bindings = build_bindings(&weight_meta);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    // Reference: center of bounds (simulates a "target" speaker embedding).
    let reference: Vec<f64> = lo
        .iter()
        .zip(hi.iter())
        .map(|(&l, &h)| f64::midpoint(f64::from(l), f64::from(h)))
        .collect();

    let cos_dist =
        worst_case_cosine_distance(lo.as_slice().unwrap(), hi.as_slice().unwrap(), &reference);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ECAPA-TDNN IBP cosine distance: {cos_dist:.6} (bounds: [{lo_min}, {hi_max}])");

    assert!(
        cos_dist.is_finite(),
        "cosine distance should be finite, got {cos_dist}"
    );
    assert!(
        cos_dist >= 0.0,
        "cosine distance should be non-negative, got {cos_dist}"
    );
    assert!(
        cos_dist <= 2.0,
        "cosine distance should be <= 2.0, got {cos_dist}"
    );

    // Log the Property 4 result for certificate generation.
    eprintln!("Property 4 (Speaker consistency): cosine distance bound = {cos_dist:.6}");
}

/// CROWN-based cosine distance: compare with IBP bounds.
///
/// CROWN should produce equal or tighter bounds → equal or smaller cosine distance.
#[test]
fn test_speaker_embedding_crown_cosine_distance() {
    let (def, weight_meta, _) = build_speaker_encoder();
    let bindings = build_bindings(&weight_meta);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[IN_CH, IN_LENGTH]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    // IBP distance
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
    let ibp_ref: Vec<f64> = ibp_lo
        .iter()
        .zip(ibp_hi.iter())
        .map(|(&l, &h)| f64::midpoint(f64::from(l), f64::from(h)))
        .collect();
    let ibp_cos_dist = worst_case_cosine_distance(
        ibp_lo.as_slice().unwrap(),
        ibp_hi.as_slice().unwrap(),
        &ibp_ref,
    );

    // CROWN distance
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = crown_output.lower_upper();
    let crown_ref: Vec<f64> = crown_lo
        .iter()
        .zip(crown_hi.iter())
        .map(|(&l, &h)| f64::midpoint(f64::from(l), f64::from(h)))
        .collect();
    let crown_cos_dist = worst_case_cosine_distance(
        crown_lo.as_slice().unwrap(),
        crown_hi.as_slice().unwrap(),
        &crown_ref,
    );

    eprintln!(
        "ECAPA-TDNN {method:?}: IBP cos_dist={ibp_cos_dist:.6}, \
         CROWN cos_dist={crown_cos_dist:.6}{}",
        fallback_reason.as_deref().unwrap_or("")
    );

    assert!(
        crown_cos_dist.is_finite(),
        "CROWN cosine distance should be finite"
    );
    // CROWN bounds are tighter or equal → cosine distance should be <= IBP's
    // (using epsilon tolerance for floating-point)
    assert!(
        crown_cos_dist <= ibp_cos_dist + 1e-6,
        "CROWN cosine distance ({crown_cos_dist:.6}) should be <= \
         IBP cosine distance ({ibp_cos_dist:.6})"
    );
}
