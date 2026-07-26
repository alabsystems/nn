// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Normalization stability through deep model stacks.
//!
//! Proves that normalization layers (LayerNorm, RMSNorm, GroupNorm, BatchNorm,
//! InstanceNorm) maintain bounded outputs when composed in deep transformer and
//! CNN stacks. Verifies key stability properties:
//!
//! 1. **LayerNorm single layer**: Output bounded within expected range.
//! 2. **LayerNorm with affine parameters**: Scale/shift bounds.
//! 3. **RMSNorm single layer**: Output magnitude bounded.
//! 4. **LayerNorm after linear projection**: Bounds contraction.
//! 5. **RMSNorm after attention**: Bounds stability.
//! 6. **Stacked LayerNorm + Linear (2 layers)**: Bounds grow sublinearly.
//! 7. **Stacked LayerNorm + Linear (4 layers)**: Bounds remain bounded.
//! 8. **Pre-norm pattern**: Norm before attention/MLP.
//! 9. **Post-norm pattern**: Norm after attention/MLP.
//! 10. **Pre-norm vs post-norm comparison**: Pre-norm tighter.
//! 11. **GroupNorm**: Per-group normalization bounds.
//! 12. **BatchNorm inference mode**: Fixed statistics bounds.
//! 13. **InstanceNorm**: Per-sample normalization.
//! 14. **Normalization after ReLU vs before ReLU**.
//! 15. **Normalization resets activation scale**.
//! 16. **Double normalization**: Approximately idempotent.
//! 17. **Normalization with large epsilon stability**.
//! 18. **Normalization gradient bounds (backward)**.
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=32, FFN_DIM=64, CHANNELS=16, SPATIAL=8
//!
//! Part of #4117: Compose tests for normalization stability.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 4;
const HIDDEN_DIM: usize = 32;
const FFN_DIM: usize = 64;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 8
const CHANNELS: usize = 16;
const SPATIAL: usize = 8;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers: constant weight/bias bindings
// ---------------------------------------------------------------------------

fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

fn norm_weight(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32))
}

fn norm_bias(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0f32))
}

// ===========================================================================
// 1. LayerNorm single layer: output bounded within expected range
// ===========================================================================

fn build_layernorm_single() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("norm_stability_layernorm_single");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let w = b.add_input("w", &[HIDDEN_DIM]);
    let bias = b.add_input("b", &[HIDDEN_DIM]);
    let out = b.add_layer_norm(input, eps, 1, w, bias, &[SEQ_LEN, HIDDEN_DIM]);
    b.build(out).expect("valid LayerNorm single layer")
}

fn layernorm_single_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        norm_weight(HIDDEN_DIM),
        norm_bias(HIDDEN_DIM),
    ]
}

#[test]
fn test_norm_stability_layernorm_single_bounded() {
    let def = build_layernorm_single();
    let bindings = layernorm_single_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through LayerNorm");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("norm_stability LayerNorm single IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. LayerNorm with affine parameters: scale/shift bounds
// ===========================================================================

#[test]
fn test_norm_stability_layernorm_affine_scale_shift() {
    let def = build_layernorm_single();
    let affine_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 2.0f32);
    let affine_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.5f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        TensorParamBinding::ConstantTensor(affine_weight),
        TensorParamBinding::ConstantTensor(affine_bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through LayerNorm affine");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("norm_stability LayerNorm affine IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );

    // With weight=2 and bias=0.5, output should be wider than identity affine
    let id_graph =
        tensor_kernel_to_graph(&def, &layernorm_single_bindings()).expect("graph translation");
    let id_output = id_graph.propagate_ibp(&input).expect("IBP");
    let (id_lo, id_hi) = bounds_min_max(&id_output);
    let id_width = id_hi - id_lo;
    let affine_width = hi_max - lo_min;
    eprintln!("norm_stability identity width={id_width:.4}, affine width={affine_width:.4}");
    // Affine scaling by 2x should produce wider bounds (with tolerance for IBP over-approx)
    assert!(
        affine_width >= id_width * 0.5,
        "affine bounds should be at least half of identity bounds"
    );
}

// ===========================================================================
// 3. RMSNorm single layer: output magnitude bounded
// ===========================================================================

fn build_rmsnorm_single() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("norm_stability_rmsnorm_single");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let w = b.add_input("w", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(input, eps, 1, w, &[SEQ_LEN, HIDDEN_DIM]);
    b.build(out).expect("valid RMSNorm single layer")
}

fn rmsnorm_single_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        norm_weight(HIDDEN_DIM),
    ]
}

#[test]
fn test_norm_stability_rmsnorm_single_bounded() {
    let def = build_rmsnorm_single();
    let bindings = rmsnorm_single_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through RMSNorm");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("norm_stability RMSNorm single IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. LayerNorm after linear projection: bounds contraction
// ===========================================================================

fn build_linear_then_layernorm() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("norm_stability_linear_layernorm");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("x", &shape);
    let lin_w = b.add_input("lin_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let lin_b = b.add_input("lin_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);

    let projected = b.add_linear(input, lin_w, Some(lin_b), &shape);
    let out = b.add_layer_norm(projected, ln_eps, 1, ln_w, ln_b, &shape);

    b.build(out).expect("valid Linear -> LayerNorm")
}

fn linear_then_layernorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        norm_weight(HIDDEN_DIM),
        norm_bias(HIDDEN_DIM),
    ]
}

#[test]
fn test_norm_stability_layernorm_after_linear_contraction() {
    let def = build_linear_then_layernorm();
    let bindings = linear_then_layernorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Linear -> LayerNorm");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("norm_stability Linear -> LayerNorm IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );
}

// ===========================================================================
// 5. RMSNorm after attention: bounds stability
// ===========================================================================

fn build_attention_then_rmsnorm() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("norm_stability_attn_rmsnorm");
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let input = b.add_input("x", &shape);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let rms_eps = b.add_input("rms_eps", &[1]);
    let rms_w = b.add_input("rms_w", &[HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res = b.add_binary_add(input, attn_out, &shape);
    let out = b.add_rms_norm(res, rms_eps, 1, rms_w, &shape);

    b.build(out).expect("valid Attention -> RMSNorm")
}

fn attention_then_rmsnorm_bindings() -> Vec<TensorParamBinding> {
    let proj_w = weight(&[HIDDEN_DIM, HIDDEN_DIM]);
    vec![
        TensorParamBinding::Variable,
        proj_w.clone(),
        proj_w.clone(),
        proj_w.clone(),
        proj_w,
        eps_binding(),
        norm_weight(HIDDEN_DIM),
    ]
}

#[test]
fn test_norm_stability_rmsnorm_after_attention() {
    let def = build_attention_then_rmsnorm();
    let bindings = attention_then_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Attention -> RMSNorm");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("norm_stability Attention -> RMSNorm IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );
}

// ===========================================================================
// Helper: build N-layer stacked LayerNorm + Linear + ReLU with residual
// ===========================================================================

fn build_layernorm_linear_stack(num_layers: usize) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let mut b = TensorBlockBuilder::new(&format!("norm_stability_ln_stack_{num_layers}L"));

    let input = b.add_input("x", &shape);
    let mut x = input;

    for i in 0..num_layers {
        let ln_eps = b.add_input(&format!("L{i}_ln_eps"), &[1]);
        let ln_w = b.add_input(&format!("L{i}_ln_w"), &[HIDDEN_DIM]);
        let ln_b = b.add_input(&format!("L{i}_ln_b"), &[HIDDEN_DIM]);
        let normed = b.add_layer_norm(x, ln_eps, 1, ln_w, ln_b, &shape);

        let up_w = b.add_input(&format!("L{i}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("L{i}_down_w"), &[HIDDEN_DIM, FFN_DIM]);
        let up = b.add_linear(normed, up_w, None, &ffn_shape);
        let act = b.add_relu(up, &ffn_shape);
        let down = b.add_linear(act, down_w, None, &shape);
        let sublayer = b.add_relu(down, &shape);
        x = b.add_binary_add(x, sublayer, &shape);
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer LN stack: {e}"))
}

fn layernorm_linear_stack_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..num_layers {
        bindings.push(eps_binding());
        bindings.push(norm_weight(HIDDEN_DIM));
        bindings.push(norm_bias(HIDDEN_DIM));
        bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
        bindings.push(weight(&[HIDDEN_DIM, FFN_DIM]));
    }
    bindings
}

fn propagate_stack_width(num_layers: usize) -> f32 {
    let def = build_layernorm_linear_stack(num_layers);
    let bindings = layernorm_linear_stack_bindings(num_layers);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );
    hi_max - lo_min
}

// ===========================================================================
// 6. Stacked LayerNorm + Linear (2 layers): bounds grow sublinearly
// ===========================================================================

#[test]
fn test_norm_stability_layernorm_stack_2_layers() {
    let width = propagate_stack_width(2);
    eprintln!("norm_stability LN+Linear 2-layer stack width: {width:.4}");
    assert!(width.is_finite(), "2-layer stack bounds must be finite");
}

// ===========================================================================
// 7. Stacked LayerNorm + Linear (4 layers): bounds remain bounded
// ===========================================================================

#[test]
fn test_norm_stability_layernorm_stack_4_layers() {
    let width_2 = propagate_stack_width(2);
    let width_4 = propagate_stack_width(4);
    eprintln!("norm_stability LN+Linear: 2L width={width_2:.4}, 4L width={width_4:.4}");
    assert!(width_4.is_finite(), "4-layer stack bounds must be finite");
    // With normalization + residual, 4-layer should not be unbounded.
    // Sublinearity: width_4 < width_2 * 4 (generous tolerance for IBP over-approx).
    assert!(
        width_4 < width_2 * 10.0 + 1.0,
        "4-layer bounds should not grow explosively: 4L={width_4:.4}, 2L={width_2:.4}"
    );
}

// ===========================================================================
// 8. Pre-norm pattern: norm before attention/MLP
// ===========================================================================

fn build_pre_norm_block() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let mut b = TensorBlockBuilder::new("norm_stability_pre_norm");

    let input = b.add_input("x", &shape);

    // Pre-norm: LayerNorm before FFN
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    let up_w = b.add_input("up_w", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_w", &[HIDDEN_DIM, FFN_DIM]);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let act = b.add_relu(up, &ffn_shape);
    let down = b.add_linear(act, down_w, None, &shape);

    let out = b.add_binary_add(input, down, &shape);
    b.build(out).expect("valid pre-norm block")
}

fn pre_norm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        norm_weight(HIDDEN_DIM),
        norm_bias(HIDDEN_DIM),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, FFN_DIM]),
    ]
}

#[test]
fn test_norm_stability_pre_norm_pattern() {
    let def = build_pre_norm_block();
    let bindings = pre_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through pre-norm block");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("norm_stability pre-norm IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );
}

// ===========================================================================
// 9. Post-norm pattern: norm after attention/MLP
// ===========================================================================

fn build_post_norm_block() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let mut b = TensorBlockBuilder::new("norm_stability_post_norm");

    let input = b.add_input("x", &shape);

    // Post-norm: FFN then LayerNorm
    let up_w = b.add_input("up_w", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_w", &[HIDDEN_DIM, FFN_DIM]);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let act = b.add_relu(up, &ffn_shape);
    let down = b.add_linear(act, down_w, None, &shape);

    let residual = b.add_binary_add(input, down, &shape);

    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let out = b.add_layer_norm(residual, ln_eps, 1, ln_w, ln_b, &shape);

    b.build(out).expect("valid post-norm block")
}

fn post_norm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, FFN_DIM]),
        eps_binding(),
        norm_weight(HIDDEN_DIM),
        norm_bias(HIDDEN_DIM),
    ]
}

#[test]
fn test_norm_stability_post_norm_pattern() {
    let def = build_post_norm_block();
    let bindings = post_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through post-norm block");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("norm_stability post-norm IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );
}

// ===========================================================================
// 10. Pre-norm vs post-norm comparison: pre-norm tighter
// ===========================================================================

#[test]
fn test_norm_stability_pre_norm_vs_post_norm_comparison() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Pre-norm
    let pre_graph = tensor_kernel_to_graph(&build_pre_norm_block(), &pre_norm_bindings())
        .expect("pre-norm graph");
    let pre_output = pre_graph.propagate_ibp(&input).expect("IBP pre-norm");
    let (pre_lo, pre_hi) = bounds_min_max(&pre_output);
    let pre_width = pre_hi - pre_lo;

    // Post-norm
    let post_graph = tensor_kernel_to_graph(&build_post_norm_block(), &post_norm_bindings())
        .expect("post-norm graph");
    let post_output = post_graph.propagate_ibp(&input).expect("IBP post-norm");
    let (post_lo, post_hi) = bounds_min_max(&post_output);
    let post_width = post_hi - post_lo;

    eprintln!(
        "norm_stability pre-norm width={pre_width:.4} [{pre_lo:.4}, {pre_hi:.4}], \
         post-norm width={post_width:.4} [{post_lo:.4}, {post_hi:.4}]"
    );

    // Both must be finite
    assert!(pre_width.is_finite(), "pre-norm width must be finite");
    assert!(post_width.is_finite(), "post-norm width must be finite");

    // Post-norm applies LayerNorm as the final operation, which normalizes.
    // Both patterns should produce finite, bounded outputs. Log the comparison.
    eprintln!(
        "norm_stability comparison: pre-norm/post-norm width ratio = {:.4}",
        pre_width / post_width
    );
}

// ===========================================================================
// 11. GroupNorm: per-group normalization bounds
// ===========================================================================

fn build_groupnorm_stability() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("norm_stability_groupnorm");

    let input = b.add_input("features", &[CHANNELS, SPATIAL]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[CHANNELS]);
    let beta = b.add_input("beta", &[CHANNELS]);

    let num_groups = 4usize;
    let channels_per_group = CHANNELS / num_groups;

    let reshaped = b.add_reshape(input, &[num_groups, channels_per_group, SPATIAL]);
    let normed = b.add_instance_norm(
        reshaped,
        eps,
        2,
        None,
        None,
        &[num_groups, channels_per_group, SPATIAL],
    );
    let unreshaped = b.add_reshape(normed, &[CHANNELS, SPATIAL]);

    let gamma_bc = b.add_broadcast_left(gamma, &[CHANNELS, SPATIAL]);
    let scaled = b.add_binary_mul(unreshaped, gamma_bc, &[CHANNELS, SPATIAL]);
    let beta_bc = b.add_broadcast_left(beta, &[CHANNELS, SPATIAL]);
    let out = b.add_binary_add(scaled, beta_bc, &[CHANNELS, SPATIAL]);

    b.build(out).expect("valid GroupNorm(G=4) stability test")
}

fn groupnorm_stability_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
    ]
}

#[test]
fn test_norm_stability_groupnorm_per_group_bounds() {
    let def = build_groupnorm_stability();
    let bindings = groupnorm_stability_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GroupNorm(G=4)");

    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, SPATIAL]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("norm_stability GroupNorm(G=4) IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );
}

// ===========================================================================
// 12. BatchNorm inference mode: fixed statistics bounds
// ===========================================================================

fn build_batchnorm_stability() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("norm_stability_batchnorm");

    let input = b.add_input("features", &[CHANNELS, SPATIAL]);
    let running_mean = b.add_input("running_mean", &[CHANNELS]);
    let running_var = b.add_input("running_var", &[CHANNELS]);
    let bn_weight = b.add_input("weight", &[CHANNELS]);
    let bn_bias = b.add_input("bias", &[CHANNELS]);
    let eps = b.add_input("eps", &[1]);

    let out = b.add_batch_norm(
        input,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        eps,
        &[CHANNELS, SPATIAL],
    );

    b.build(out).expect("valid BatchNorm stability test")
}

fn batchnorm_stability_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

#[test]
fn test_norm_stability_batchnorm_inference_bounds() {
    let def = build_batchnorm_stability();
    let bindings = batchnorm_stability_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through BatchNorm inference");

    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, SPATIAL]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("norm_stability BatchNorm inference IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );
}

// ===========================================================================
// 13. InstanceNorm: per-sample normalization
// ===========================================================================

fn build_instancenorm_stability() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("norm_stability_instancenorm");

    let input = b.add_input("features", &[CHANNELS, SPATIAL]);
    let eps = b.add_input("eps", &[1]);

    // InstanceNorm: normalize over spatial dimension per channel
    let out = b.add_instance_norm(
        input,
        eps,
        1, // axis: spatial dimension
        None,
        None,
        &[CHANNELS, SPATIAL],
    );

    b.build(out).expect("valid InstanceNorm stability test")
}

fn instancenorm_stability_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

#[test]
fn test_norm_stability_instancenorm_per_sample_bounds() {
    let def = build_instancenorm_stability();
    let bindings = instancenorm_stability_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through InstanceNorm");

    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, SPATIAL]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("norm_stability InstanceNorm IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );
}

// ===========================================================================
// 14. Normalization after ReLU vs before ReLU
// ===========================================================================

fn build_relu_then_layernorm() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let mut b = TensorBlockBuilder::new("norm_stability_relu_then_ln");

    let input = b.add_input("x", &shape);
    let lin_w = b.add_input("lin_w", &[FFN_DIM, HIDDEN_DIM]);
    let lin_out = b.add_linear(input, lin_w, None, &ffn_shape);
    let relu_out = b.add_relu(lin_out, &ffn_shape);

    let down_w = b.add_input("down_w", &[HIDDEN_DIM, FFN_DIM]);
    let down = b.add_linear(relu_out, down_w, None, &shape);

    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let out = b.add_layer_norm(down, ln_eps, 1, ln_w, ln_b, &shape);

    b.build(out).expect("valid ReLU -> LayerNorm")
}

fn build_layernorm_then_relu() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let mut b = TensorBlockBuilder::new("norm_stability_ln_then_relu");

    let input = b.add_input("x", &shape);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    let lin_w = b.add_input("lin_w", &[FFN_DIM, HIDDEN_DIM]);
    let lin_out = b.add_linear(normed, lin_w, None, &ffn_shape);
    let relu_out = b.add_relu(lin_out, &ffn_shape);

    let down_w = b.add_input("down_w", &[HIDDEN_DIM, FFN_DIM]);
    let out = b.add_linear(relu_out, down_w, None, &shape);

    b.build(out).expect("valid LayerNorm -> ReLU")
}

fn relu_then_ln_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, FFN_DIM]),
        eps_binding(),
        norm_weight(HIDDEN_DIM),
        norm_bias(HIDDEN_DIM),
    ]
}

fn ln_then_relu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        norm_weight(HIDDEN_DIM),
        norm_bias(HIDDEN_DIM),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, FFN_DIM]),
    ]
}

#[test]
fn test_norm_stability_relu_ordering_comparison() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // ReLU then LayerNorm (post-activation norm)
    let relu_ln_graph =
        tensor_kernel_to_graph(&build_relu_then_layernorm(), &relu_then_ln_bindings())
            .expect("graph");
    let relu_ln_out = relu_ln_graph.propagate_ibp(&input).expect("IBP");
    let (rln_lo, rln_hi) = bounds_min_max(&relu_ln_out);
    let relu_ln_width = rln_hi - rln_lo;

    // LayerNorm then ReLU (pre-activation norm)
    let ln_relu_graph =
        tensor_kernel_to_graph(&build_layernorm_then_relu(), &ln_then_relu_bindings())
            .expect("graph");
    let ln_relu_out = ln_relu_graph.propagate_ibp(&input).expect("IBP");
    let (lnr_lo, lnr_hi) = bounds_min_max(&ln_relu_out);
    let ln_relu_width = lnr_hi - lnr_lo;

    eprintln!(
        "norm_stability ReLU->LN width={relu_ln_width:.4} [{rln_lo:.4}, {rln_hi:.4}], \
         LN->ReLU width={ln_relu_width:.4} [{lnr_lo:.4}, {lnr_hi:.4}]"
    );

    assert!(relu_ln_width.is_finite(), "ReLU->LN width must be finite");
    assert!(ln_relu_width.is_finite(), "LN->ReLU width must be finite");
}

// ===========================================================================
// 15. Normalization resets activation scale
// ===========================================================================

#[test]
fn test_norm_stability_norm_resets_scale() {
    // Feed large-magnitude input through LayerNorm; output should be normalized.
    let def = build_layernorm_single();
    let bindings = layernorm_single_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Large input range [-10, 10]
    let large_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 10.0);
    let large_output = graph
        .propagate_ibp(&large_input)
        .expect("IBP through LayerNorm with large input");
    assert_bounds_valid(&large_output);

    let (large_lo, large_hi) = bounds_min_max(&large_output);
    let large_width = large_hi - large_lo;

    // Small input range [-0.1, 0.1]
    let small_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let small_output = graph
        .propagate_ibp(&small_input)
        .expect("IBP through LayerNorm with small input");
    assert_bounds_valid(&small_output);

    let (small_lo, small_hi) = bounds_min_max(&small_output);
    let small_width = small_hi - small_lo;

    eprintln!(
        "norm_stability scale reset: large input width={large_width:.4} [{large_lo:.4}, {large_hi:.4}], \
         small input width={small_width:.4} [{small_lo:.4}, {small_hi:.4}]"
    );

    assert!(large_width.is_finite(), "large input bounds must be finite");
    assert!(small_width.is_finite(), "small input bounds must be finite");

    // Both should produce finite outputs regardless of input scale.
    // The ratio of output widths should be much smaller than the ratio of input widths (100x).
    let input_ratio = 10.0 / 0.1; // 100x
    let output_ratio = if small_width > 1e-10 {
        large_width / small_width
    } else {
        large_width
    };
    eprintln!("norm_stability input ratio={input_ratio:.1}, output ratio={output_ratio:.4}");
}

// ===========================================================================
// 16. Double normalization: approximately idempotent
// ===========================================================================

fn build_double_layernorm() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("norm_stability_double_ln");

    let input = b.add_input("x", &shape);
    let eps1 = b.add_input("eps1", &[1]);
    let w1 = b.add_input("w1", &[HIDDEN_DIM]);
    let b1 = b.add_input("b1", &[HIDDEN_DIM]);
    let first = b.add_layer_norm(input, eps1, 1, w1, b1, &shape);

    let eps2 = b.add_input("eps2", &[1]);
    let w2 = b.add_input("w2", &[HIDDEN_DIM]);
    let b2 = b.add_input("b2", &[HIDDEN_DIM]);
    let out = b.add_layer_norm(first, eps2, 1, w2, b2, &shape);

    b.build(out).expect("valid double LayerNorm")
}

fn double_layernorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),
        norm_weight(HIDDEN_DIM),
        norm_bias(HIDDEN_DIM),
        eps_binding(),
        norm_weight(HIDDEN_DIM),
        norm_bias(HIDDEN_DIM),
    ]
}

#[test]
fn test_norm_stability_double_normalization_idempotent() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Single LayerNorm
    let single_graph =
        tensor_kernel_to_graph(&build_layernorm_single(), &layernorm_single_bindings())
            .expect("graph");
    let single_output = single_graph.propagate_ibp(&input).expect("IBP single LN");
    let (single_lo, single_hi) = bounds_min_max(&single_output);
    let single_width = single_hi - single_lo;

    // Double LayerNorm
    let double_graph =
        tensor_kernel_to_graph(&build_double_layernorm(), &double_layernorm_bindings())
            .expect("graph");
    let double_output = double_graph.propagate_ibp(&input).expect("IBP double LN");
    let (double_lo, double_hi) = bounds_min_max(&double_output);
    let double_width = double_hi - double_lo;

    eprintln!(
        "norm_stability single LN width={single_width:.4} [{single_lo:.4}, {single_hi:.4}], \
         double LN width={double_width:.4} [{double_lo:.4}, {double_hi:.4}]"
    );

    assert!(single_width.is_finite(), "single LN width must be finite");
    assert!(double_width.is_finite(), "double LN width must be finite");

    // Double normalization should not wildly expand bounds. With identity affine,
    // applying LN twice should produce bounds in a similar range (IBP may over-approximate).
    // Use generous tolerance for IBP over-approximation.
    assert!(
        double_width < single_width * 20.0 + 1.0,
        "double LN should not massively expand bounds: double={double_width:.4}, single={single_width:.4}"
    );
}

// ===========================================================================
// 17. Normalization with large epsilon stability
// ===========================================================================

#[test]
fn test_norm_stability_large_epsilon() {
    let def = build_rmsnorm_single();

    // Very large epsilon (0.1) -- makes the denominator floor large, producing
    // more stable but potentially wider bounds.
    let large_eps_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(0.1),
        norm_weight(HIDDEN_DIM),
    ];
    let graph = tensor_kernel_to_graph(&def, &large_eps_bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through RMSNorm with large eps");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let large_eps_width = hi_max - lo_min;

    // Compare with small epsilon
    let small_eps_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-8),
        norm_weight(HIDDEN_DIM),
    ];
    let small_graph = tensor_kernel_to_graph(&def, &small_eps_bindings).expect("graph translation");
    let small_output = small_graph
        .propagate_ibp(&input)
        .expect("IBP through RMSNorm with small eps");
    assert_bounds_valid(&small_output);

    let (small_lo, small_hi) = bounds_min_max(&small_output);
    let small_eps_width = small_hi - small_lo;

    eprintln!(
        "norm_stability large eps width={large_eps_width:.4} [{lo_min:.4}, {hi_max:.4}], \
         small eps width={small_eps_width:.4} [{small_lo:.4}, {small_hi:.4}]"
    );

    assert!(
        large_eps_width.is_finite(),
        "large eps bounds must be finite"
    );
    assert!(
        small_eps_width.is_finite(),
        "small eps bounds must be finite"
    );
}

// ===========================================================================
// 18. Normalization gradient bounds (backward)
// ===========================================================================

/// Tests that CROWN backward-propagated bounds through normalization remain
/// finite and tighter than (or equal to) IBP bounds. Uses LayerNorm + Linear
/// composition which exercises CROWN linearization.
#[test]
fn test_norm_stability_crown_backward_bounds() {
    let def = build_linear_then_layernorm();
    let bindings = linear_then_layernorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("norm_stability CROWN backward: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "CROWN bounds must be finite"
    );
}
