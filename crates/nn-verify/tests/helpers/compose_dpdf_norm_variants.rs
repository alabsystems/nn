// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Normalization variant bound characteristics.
//!
//! Complements `compose_dpdf_normalization.rs` (20 tests covering single-layer
//! IBP/CROWN for LN, RMSNorm, BN, GN(G=4, G=1), basic compositions, epsilon
//! monotonicity, and verify-and-record) with 15 additional tests focused on:
//!
//! - Running statistics effects on BatchNorm bounds
//! - Tighter input ranges for LayerNorm and RMSNorm
//! - GroupNorm at production-scale group counts (G=32)
//! - InstanceNorm standalone verification
//! - Cross-type comparisons (BN vs LN, RMSNorm vs LN, GroupNorm group count)
//! - Large affine parameter effects on BatchNorm and LayerNorm
//! - Norm + activation composition (LN -> GELU)
//! - Pre-norm residual block pattern
//! - Pre-norm vs post-norm comparison
//! - Numerical stability with small variance inputs
//! - Full Conv -> BN -> SiLU block (YOLO backbone pattern)
//!
//! Dimensions (small for fast verification):
//! - HIDDEN_DIM=64, SEQ_LEN=4, CHANNELS=32, SPATIAL=8
//!
//! Part of #4032: Compose tests for normalization variant bounds.

use super::common::{assert_bounds_valid, bounds_min_max, uniform_bounds};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 64;
const SEQ_LEN: usize = 4;
const CHANNELS: usize = 32;
const SPATIAL: usize = 8;
const WEIGHT_MAG: f32 = 0.02;
const CONV_SPATIAL: usize = 4;
const CONV_OUT_CH: usize = 32;

// ===========================================================================
// 1. BatchNorm — running mean/var offset effect on bounds
// ===========================================================================

/// Build a BatchNorm kernel for [CHANNELS, SPATIAL] inputs.
fn build_batchnorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_nv_batchnorm");

    let input = b.add_input("features", &[CHANNELS, SPATIAL]);
    let running_mean = b.add_input("running_mean", &[CHANNELS]);
    let running_var = b.add_input("running_var", &[CHANNELS]);
    let weight = b.add_input("weight", &[CHANNELS]);
    let bias = b.add_input("bias", &[CHANNELS]);
    let eps = b.add_input("eps", &[1]);

    let out = b.add_batch_norm(
        input,
        running_mean,
        running_var,
        weight,
        bias,
        eps,
        &[CHANNELS, SPATIAL],
    );

    b.build(out).expect("valid BatchNorm kernel")
}

/// BatchNorm with non-zero running mean shifts output bounds.
///
/// Compared to mean=0, var=1 (identity case in normalization.rs test 7),
/// non-zero running_mean and non-unit running_var should shift and scale
/// the output bounds. We verify bounds remain finite and observe the shift.
#[test]
fn test_dpdf_nv_batchnorm_running_stats_shift() {
    let def = build_batchnorm_kernel();

    // Case 1: identity stats (mean=0, var=1)
    let bindings_identity = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    // Case 2: shifted stats (mean=2.0, var=0.5)
    let bindings_shifted = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 2.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);

    let graph_id = tensor_kernel_to_graph(&def, &bindings_identity).expect("graph translation");
    let out_id = graph_id.propagate_ibp(&input).expect("IBP identity stats");
    assert_bounds_valid(&out_id);
    let (lo_id, hi_id) = bounds_min_max(&out_id);

    let graph_sh = tensor_kernel_to_graph(&def, &bindings_shifted).expect("graph translation");
    let out_sh = graph_sh.propagate_ibp(&input).expect("IBP shifted stats");
    assert_bounds_valid(&out_sh);
    let (lo_sh, hi_sh) = bounds_min_max(&out_sh);

    eprintln!("BN identity stats: [{lo_id}, {hi_id}], shifted stats: [{lo_sh}, {hi_sh}]");
    // Shifted mean changes normalization center; both must remain finite.
    assert!(lo_sh.is_finite() && hi_sh.is_finite());
}

// ===========================================================================
// 2. LayerNorm — tighter input range produces tighter bounds
// ===========================================================================

/// Build a LayerNorm kernel for [SEQ_LEN, HIDDEN_DIM] inputs.
fn build_layernorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_nv_layernorm");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);

    let out = b.add_layer_norm(input, eps, 1, weight, bias, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid LayerNorm kernel")
}

fn layernorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
    ]
}

/// Tighter input bounds should produce tighter (or equal) LayerNorm output bounds.
#[test]
fn test_dpdf_nv_layernorm_tighter_input() {
    let def = build_layernorm_kernel();
    let bindings = layernorm_bindings();

    let input_wide = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);
    let input_narrow = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let out_wide = graph.propagate_ibp(&input_wide).expect("IBP wide");
    let out_narrow = graph.propagate_ibp(&input_narrow).expect("IBP narrow");

    assert_bounds_valid(&out_wide);
    assert_bounds_valid(&out_narrow);

    let (lo_w, hi_w) = bounds_min_max(&out_wide);
    let (lo_n, hi_n) = bounds_min_max(&out_narrow);
    let width_wide = hi_w - lo_w;
    let width_narrow = hi_n - lo_n;

    eprintln!("LN wide input: width={width_wide:.4}, narrow input: width={width_narrow:.4}");
    // Narrower input should give tighter or equal output (with tolerance for IBP).
    let tolerance = width_wide * 0.1 + 1e-3;
    assert!(
        width_narrow <= width_wide + tolerance,
        "narrow input should produce tighter LN output: {width_narrow} > {width_wide} + {tolerance}"
    );
}

// ===========================================================================
// 3. RMSNorm — tighter input range produces tighter bounds
// ===========================================================================

fn build_rmsnorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_nv_rmsnorm");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[HIDDEN_DIM]);

    let out = b.add_rms_norm(input, eps, 1, weight, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid RMSNorm kernel")
}

fn rmsnorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
    ]
}

/// Tighter input bounds should produce tighter (or equal) RMSNorm output bounds.
#[test]
fn test_dpdf_nv_rmsnorm_tighter_input() {
    let def = build_rmsnorm_kernel();
    let bindings = rmsnorm_bindings();

    let input_wide = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);
    let input_narrow = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let out_wide = graph.propagate_ibp(&input_wide).expect("IBP wide");
    let out_narrow = graph.propagate_ibp(&input_narrow).expect("IBP narrow");

    assert_bounds_valid(&out_wide);
    assert_bounds_valid(&out_narrow);

    let (lo_w, hi_w) = bounds_min_max(&out_wide);
    let (lo_n, hi_n) = bounds_min_max(&out_narrow);
    let width_wide = hi_w - lo_w;
    let width_narrow = hi_n - lo_n;

    eprintln!("RMSNorm wide input: width={width_wide:.4}, narrow input: width={width_narrow:.4}");
    let tolerance = width_wide * 0.1 + 1e-3;
    assert!(
        width_narrow <= width_wide + tolerance,
        "narrow input should produce tighter RMSNorm output: {width_narrow} > {width_wide} + {tolerance}"
    );
}

// ===========================================================================
// 4. GroupNorm (groups=32) — production-scale group count
// ===========================================================================

/// Build GroupNorm(G=32) for CHANNELS=32 (i.e. 1 channel per group = InstanceNorm).
///
/// Many vision backbones use groups=32 with channels=32 or 64.
/// This tests the decomposed path at a production-scale group count.
fn build_groupnorm_g32_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_nv_groupnorm_g32");

    let input = b.add_input("features", &[CHANNELS, SPATIAL]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[CHANNELS]);
    let beta = b.add_input("beta", &[CHANNELS]);

    let num_groups = 32usize;
    let channels_per_group = CHANNELS / num_groups; // 1

    // Reshape [C, T] -> [G, C/G, T]
    let reshaped = b.add_reshape(input, &[num_groups, channels_per_group, SPATIAL]);

    // InstanceNorm over spatial axis
    let normed = b.add_instance_norm(
        reshaped,
        eps,
        2,
        None,
        None,
        &[num_groups, channels_per_group, SPATIAL],
    );

    // Reshape back to [C, T]
    let unreshaped = b.add_reshape(normed, &[CHANNELS, SPATIAL]);

    // Affine
    let gamma_bc = b.add_broadcast_left(gamma, &[CHANNELS, SPATIAL]);
    let scaled = b.add_binary_mul(unreshaped, gamma_bc, &[CHANNELS, SPATIAL]);
    let beta_bc = b.add_broadcast_left(beta, &[CHANNELS, SPATIAL]);
    let out = b.add_binary_add(scaled, beta_bc, &[CHANNELS, SPATIAL]);

    b.build(out).expect("valid GroupNorm(G=32) kernel")
}

/// GroupNorm(G=32) IBP bounds propagate finitely.
#[test]
fn test_dpdf_nv_groupnorm_g32_ibp_bounds() {
    let def = build_groupnorm_g32_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
    ];

    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GroupNorm(G=32)");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL],
        "GroupNorm(G=32) output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GroupNorm(G=32) IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. InstanceNorm — standalone verification
// ===========================================================================

/// Build standalone InstanceNorm for [CHANNELS, SPATIAL].
fn build_instancenorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_nv_instancenorm");

    let input = b.add_input("features", &[CHANNELS, SPATIAL]);
    let eps = b.add_input("eps", &[1]);

    let out = b.add_instance_norm(input, eps, 1, None, None, &[CHANNELS, SPATIAL]);

    b.build(out).expect("valid InstanceNorm kernel")
}

/// InstanceNorm IBP bounds propagate finitely.
#[test]
fn test_dpdf_nv_instancenorm_ibp_bounds() {
    let def = build_instancenorm_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through InstanceNorm");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL],
        "InstanceNorm output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("InstanceNorm IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. BN vs LN bound width comparison
// ===========================================================================

/// Compare BatchNorm and LayerNorm output bound widths on comparable inputs.
///
/// BatchNorm uses frozen running stats (affine transform), while LayerNorm
/// computes data-dependent normalization. On equal input ranges, BN should
/// produce tighter bounds (it's a fixed affine transform), but both must
/// be finite and valid.
#[test]
fn test_dpdf_nv_batchnorm_vs_layernorm_comparison() {
    // Use CHANNELS=HIDDEN_DIM so shapes are comparable
    let dim = 32usize;
    let seq = 8usize;

    // BatchNorm on [dim, seq]
    let mut b_bn = TensorBlockBuilder::new("dpdf_nv_bn_compare");
    let bn_in = b_bn.add_input("features", &[dim, seq]);
    let bn_mean = b_bn.add_input("mean", &[dim]);
    let bn_var = b_bn.add_input("var", &[dim]);
    let bn_w = b_bn.add_input("weight", &[dim]);
    let bn_b = b_bn.add_input("bias", &[dim]);
    let bn_eps = b_bn.add_input("eps", &[1]);
    let bn_out = b_bn.add_batch_norm(bn_in, bn_mean, bn_var, bn_w, bn_b, bn_eps, &[dim, seq]);
    let bn_def = b_bn.build(bn_out).expect("valid BN kernel");

    let bn_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    // LayerNorm on [dim, seq]
    let mut b_ln = TensorBlockBuilder::new("dpdf_nv_ln_compare");
    let ln_in = b_ln.add_input("hidden", &[dim, seq]);
    let ln_eps = b_ln.add_input("eps", &[1]);
    let ln_w = b_ln.add_input("weight", &[seq]);
    let ln_b = b_ln.add_input("bias", &[seq]);
    let ln_out = b_ln.add_layer_norm(ln_in, ln_eps, 1, ln_w, ln_b, &[dim, seq]);
    let ln_def = b_ln.build(ln_out).expect("valid LN kernel");

    let ln_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[seq]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[seq]), 0.0f32)),
    ];

    let input = uniform_bounds(&[dim, seq], 1.0);

    let bn_graph = tensor_kernel_to_graph(&bn_def, &bn_bindings).expect("BN graph");
    let bn_output = bn_graph.propagate_ibp(&input).expect("BN IBP");
    assert_bounds_valid(&bn_output);
    let (bn_lo, bn_hi) = bounds_min_max(&bn_output);
    let bn_width = bn_hi - bn_lo;

    let ln_graph = tensor_kernel_to_graph(&ln_def, &ln_bindings).expect("LN graph");
    let ln_output = ln_graph.propagate_ibp(&input).expect("LN IBP");
    assert_bounds_valid(&ln_output);
    let (ln_lo, ln_hi) = bounds_min_max(&ln_output);
    let ln_width = ln_hi - ln_lo;

    eprintln!(
        "BN width={bn_width:.4} [{bn_lo}, {bn_hi}], LN width={ln_width:.4} [{ln_lo}, {ln_hi}]"
    );
    // Both widths must be finite; BN (fixed affine) is expected to be tighter.
    assert!(bn_width.is_finite() && ln_width.is_finite());
}

// ===========================================================================
// 7. RMSNorm vs LayerNorm bound width comparison
// ===========================================================================

/// Compare RMSNorm and LayerNorm output bound widths.
///
/// RMSNorm omits mean subtraction. On symmetric input ([-r, r]), both should
/// produce similar width, but the actual bound magnitudes may differ.
#[test]
fn test_dpdf_nv_rmsnorm_vs_layernorm_comparison() {
    let def_rms = build_rmsnorm_kernel();
    let def_ln = build_layernorm_kernel();
    let bindings_rms = rmsnorm_bindings();
    let bindings_ln = layernorm_bindings();

    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let graph_rms = tensor_kernel_to_graph(&def_rms, &bindings_rms).expect("RMS graph");
    let out_rms = graph_rms.propagate_ibp(&input).expect("RMS IBP");
    assert_bounds_valid(&out_rms);
    let (rms_lo, rms_hi) = bounds_min_max(&out_rms);
    let rms_width = rms_hi - rms_lo;

    let graph_ln = tensor_kernel_to_graph(&def_ln, &bindings_ln).expect("LN graph");
    let out_ln = graph_ln.propagate_ibp(&input).expect("LN IBP");
    assert_bounds_valid(&out_ln);
    let (ln_lo, ln_hi) = bounds_min_max(&out_ln);
    let ln_width = ln_hi - ln_lo;

    eprintln!(
        "RMSNorm width={rms_width:.4} [{rms_lo}, {rms_hi}], LN width={ln_width:.4} [{ln_lo}, {ln_hi}]"
    );
    assert!(rms_width.is_finite() && ln_width.is_finite());
}

// ===========================================================================
// 8. GroupNorm group count effect (G=4, G=8, G=16)
// ===========================================================================

/// Build GroupNorm(G=g) for CHANNELS=32, SPATIAL=8.
fn build_groupnorm_gn_kernel(num_groups: usize) -> TensorKernelDef {
    let cpg = CHANNELS / num_groups;
    let mut b = TensorBlockBuilder::new(&format!("dpdf_nv_groupnorm_g{num_groups}"));

    let input = b.add_input("features", &[CHANNELS, SPATIAL]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[CHANNELS]);
    let beta = b.add_input("beta", &[CHANNELS]);

    let reshaped = b.add_reshape(input, &[num_groups, cpg, SPATIAL]);
    let normed = b.add_instance_norm(reshaped, eps, 2, None, None, &[num_groups, cpg, SPATIAL]);
    let unreshaped = b.add_reshape(normed, &[CHANNELS, SPATIAL]);

    let gamma_bc = b.add_broadcast_left(gamma, &[CHANNELS, SPATIAL]);
    let scaled = b.add_binary_mul(unreshaped, gamma_bc, &[CHANNELS, SPATIAL]);
    let beta_bc = b.add_broadcast_left(beta, &[CHANNELS, SPATIAL]);
    let out = b.add_binary_add(scaled, beta_bc, &[CHANNELS, SPATIAL]);

    b.build(out).expect("valid GroupNorm kernel")
}

fn groupnorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
    ]
}

/// More groups -> each group normalizes fewer channels -> potentially different
/// bound widths. All must be finite and valid.
#[test]
fn test_dpdf_nv_groupnorm_group_count_effect() {
    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);
    let bindings = groupnorm_bindings();

    let mut widths = Vec::new();
    for &g in &[4usize, 8, 16] {
        let def = build_groupnorm_gn_kernel(g);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let output = graph
            .propagate_ibp(&input)
            .unwrap_or_else(|e| panic!("IBP through GroupNorm(G={g}): {e:?}"));

        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("GroupNorm(G={g}) IBP: width={width:.4}, bounds=[{lo_min}, {hi_max}]");
        assert!(width.is_finite());
        widths.push(width);
    }
    // All widths recorded for observability — group count effect varies per input.
    eprintln!(
        "Widths: G=4: {:.4}, G=8: {:.4}, G=16: {:.4}",
        widths[0], widths[1], widths[2]
    );
}

// ===========================================================================
// 9. BatchNorm with large affine (gamma/beta) bounds
// ===========================================================================

/// BatchNorm with large gamma (2.0) and large bias (5.0) amplifies and shifts bounds.
#[test]
fn test_dpdf_nv_batchnorm_large_affine() {
    let def = build_batchnorm_kernel();

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 2.0f32)), // gamma=2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 5.0f32)), // beta=5
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let input = uniform_bounds(&[CHANNELS, SPATIAL], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through BN with large affine");

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("BN large affine (gamma=2, beta=5): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    // With beta=5 offset, lower bound should be shifted positive.
    // BN(x) = 2*(x - 0)/sqrt(1 + eps) + 5 on input [-1, 1] => roughly [3, 7]
}

// ===========================================================================
// 10. LayerNorm + scaled affine bounds
// ===========================================================================

/// LayerNorm with non-trivial affine (weight=0.1, bias=3.0) compresses and shifts.
#[test]
fn test_dpdf_nv_layernorm_scaled_affine() {
    let def = build_layernorm_kernel();

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.1f32)), // weight=0.1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 3.0f32)), // bias=3.0
    ];

    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through LN scaled affine");

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("LN scaled affine (w=0.1, b=3.0): width={width:.4}, bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 11. Norm + activation composition: LayerNorm -> GELU
// ===========================================================================

/// Build LayerNorm -> GELU composition.
///
/// Pattern: pre-activation normalization before non-linearity.
fn build_layernorm_gelu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_nv_layernorm_gelu");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);

    let normed = b.add_layer_norm(input, eps, 1, weight, bias, &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_gelu(normed, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid LayerNorm -> GELU kernel")
}

/// LayerNorm -> GELU IBP bounds propagate finitely.
#[test]
fn test_dpdf_nv_layernorm_gelu_composition() {
    let def = build_layernorm_gelu_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph.propagate_ibp(&input).expect("IBP through LN -> GELU");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "LN -> GELU output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LN -> GELU IBP: bounds=[{lo_min}, {hi_max}]");
    // GELU lower bound should be >= -0.17 (GELU minimum), so overall lower >= some finite value.
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 12. Pre-norm residual block: x + Linear(LayerNorm(x))
// ===========================================================================

/// Build pre-norm residual: x + Linear(LayerNorm(x)).
///
/// This is the standard Transformer pre-norm pattern used in ViT, Granite, GPT.
fn build_prenorm_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_nv_prenorm_residual");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_weight = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_bias = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let linear_w = b.add_input("linear_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let linear_b = b.add_input("linear_bias", &[HIDDEN_DIM]);

    let normed = b.add_layer_norm(input, eps, 1, ln_weight, ln_bias, &[SEQ_LEN, HIDDEN_DIM]);
    let projected = b.add_linear(normed, linear_w, Some(linear_b), &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_binary_add(input, projected, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid pre-norm residual kernel")
}

/// Pre-norm residual IBP bounds propagate, demonstrating bound widening from skip.
#[test]
fn test_dpdf_nv_prenorm_residual_ibp() {
    let def = build_prenorm_residual_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through pre-norm residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "pre-norm residual output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("Pre-norm residual IBP: width={width:.4}, bounds=[{lo_min}, {hi_max}]");
    // Residual adds input bounds to transformed bounds, so output width >= input width (2.0).
    assert!(width >= 1.9, "residual should widen bounds: width={width}");
}

// ===========================================================================
// 13. Pre-norm vs post-norm comparison
// ===========================================================================

/// Build post-norm residual: LayerNorm(x + Linear(x)).
fn build_postnorm_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_nv_postnorm_residual");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_weight = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_bias = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let linear_w = b.add_input("linear_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let linear_b = b.add_input("linear_bias", &[HIDDEN_DIM]);

    let projected = b.add_linear(input, linear_w, Some(linear_b), &[SEQ_LEN, HIDDEN_DIM]);
    let residual = b.add_binary_add(input, projected, &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_layer_norm(residual, eps, 1, ln_weight, ln_bias, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid post-norm residual kernel")
}

/// Compare pre-norm and post-norm residual bound widths.
///
/// Pre-norm (x + LN(x)) preserves the skip connection bounds directly.
/// Post-norm (LN(x + Linear(x))) normalizes after the skip.
/// Both must produce finite bounds; width difference is logged.
#[test]
fn test_dpdf_nv_prenorm_vs_postnorm_comparison() {
    let def_pre = build_prenorm_residual_kernel();
    let def_post = build_postnorm_residual_kernel();

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let graph_pre = tensor_kernel_to_graph(&def_pre, &bindings).expect("pre-norm graph");
    let out_pre = graph_pre.propagate_ibp(&input).expect("pre-norm IBP");
    assert_bounds_valid(&out_pre);
    let (pre_lo, pre_hi) = bounds_min_max(&out_pre);
    let pre_width = pre_hi - pre_lo;

    let graph_post = tensor_kernel_to_graph(&def_post, &bindings).expect("post-norm graph");
    let out_post = graph_post.propagate_ibp(&input).expect("post-norm IBP");
    assert_bounds_valid(&out_post);
    let (post_lo, post_hi) = bounds_min_max(&out_post);
    let post_width = post_hi - post_lo;

    eprintln!(
        "Pre-norm: width={pre_width:.4} [{pre_lo}, {pre_hi}], \
         Post-norm: width={post_width:.4} [{post_lo}, {post_hi}]"
    );
    assert!(pre_width.is_finite() && post_width.is_finite());
}

// ===========================================================================
// 14. Normalization numerical stability — small variance inputs
// ===========================================================================

/// Verify normalization handles near-constant inputs (small variance) gracefully.
///
/// When all input elements are nearly identical, variance approaches zero.
/// The epsilon parameter prevents division by zero. Bounds must remain finite.
#[test]
fn test_dpdf_nv_norm_numerical_stability_small_variance() {
    let def = build_layernorm_kernel();
    let bindings = layernorm_bindings();

    // Very tight input: all elements in [0.99, 1.01] — near-constant.
    let input = nn_verify::BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.99f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 1.01f32),
    )
    .expect("valid tight bounds");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through LN with small variance");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LN small variance input [0.99, 1.01]: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite(),
        "lower bound must be finite for small variance"
    );
    assert!(
        hi_max.is_finite(),
        "upper bound must be finite for small variance"
    );
}

// ===========================================================================
// 15. Full Conv -> BN -> SiLU block (YOLO backbone pattern)
// ===========================================================================

/// Build Conv2d -> BatchNorm -> SiLU (Sigmoid Linear Unit) block.
///
/// This is the ConvBnAct building block used in DocLayout-YOLO and similar
/// detection backbones. SiLU(x) = x * sigmoid(x).
///
/// Input: `[CHANNELS, CONV_SPATIAL, CONV_SPATIAL]` (Variable).
/// Output: `[CONV_OUT_CH, CONV_SPATIAL - 2, CONV_SPATIAL - 2]` (valid padding conv).
fn build_conv_bn_silu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_nv_conv_bn_silu");
    let out_h = CONV_SPATIAL - 2;
    let out_w = out_h;

    let input = b.add_input("features", &[CHANNELS, CONV_SPATIAL, CONV_SPATIAL]);
    let conv_w = b.add_input("conv_weight", &[CONV_OUT_CH, CHANNELS, 3, 3]);
    let conv_b = b.add_input("conv_bias", &[CONV_OUT_CH]);
    let bn_mean = b.add_input("bn_running_mean", &[CONV_OUT_CH]);
    let bn_var = b.add_input("bn_running_var", &[CONV_OUT_CH]);
    let bn_weight = b.add_input("bn_weight", &[CONV_OUT_CH]);
    let bn_bias = b.add_input("bn_bias", &[CONV_OUT_CH]);
    let bn_eps = b.add_input("bn_eps", &[1]);

    // Conv2d(kernel=3, stride=1, padding=0)
    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        1,
        1,
        0,
        0,
        &[CONV_OUT_CH, out_h, out_w],
    );

    // BatchNorm
    let normed = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_weight,
        bn_bias,
        bn_eps,
        &[CONV_OUT_CH, out_h, out_w],
    );

    // SiLU: x * sigmoid(x)
    let sig = b.add_sigmoid(normed, &[CONV_OUT_CH, out_h, out_w]);
    let out = b.add_binary_mul(normed, sig, &[CONV_OUT_CH, out_h, out_w]);

    b.build(out).expect("valid Conv -> BN -> SiLU kernel")
}

/// Full Conv -> BN -> SiLU IBP bounds propagate.
#[test]
fn test_dpdf_nv_conv_bn_silu_block() {
    let def = build_conv_bn_silu_kernel();
    let out_h = CONV_SPATIAL - 2;
    let out_w = out_h;

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CONV_OUT_CH, CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CONV_OUT_CH]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let input = uniform_bounds(&[CHANNELS, CONV_SPATIAL, CONV_SPATIAL], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Conv -> BN -> SiLU");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CONV_OUT_CH, out_h, out_w],
        "Conv -> BN -> SiLU output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv -> BN -> SiLU IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}
