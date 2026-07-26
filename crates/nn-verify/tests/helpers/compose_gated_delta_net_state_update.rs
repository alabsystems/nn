// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification for GatedDeltaNet linear attention state update.
//!
//! Verifies that the recurrent state update S' = diag(gate)*S + outer(k, beta*v - beta*vr)
//! has bounded state evolution. The gate (in [0,1]) is the key mechanism preventing
//! unbounded growth: when gate < 1, the recurrence is contractive.
//!
//! Design: Only the state input is Variable; q/k/v/gate/beta are Constant.
//! This isolates state evolution bounds from multi-variable shape mismatch
//! (known issue: heterogeneous shapes [H,K] vs [H,K,V] in IBP stacking).
//! Single-variable propagation produces clean, meaningful bounds.
//!
//! Test categories:
//!   1. Single-step state update (state=Variable, all else Constant)
//!   2. Multi-step state evolution (4 steps, chained state)
//!   3. Output computation: o = scale * q @ new_state
//!   4. Contractivity: gate < 1 implies state bounds converge
//!
//! Part of #3578 — GatedDeltaNet state update composition verification.

use super::common::{assert_bounds_valid, assert_bounds_width, bounds_min_max};
use nn_dsl::gated_delta_net::decompose_gated_delta_net;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor, TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

// ===========================================================================
// Constants: small dims as specified in the issue (d=16 would be large;
// we use d=4 for tractable IBP/CROWN propagation)
// ===========================================================================

const H: usize = 2; // num_heads
const K: usize = 4; // key_dim
const V: usize = 4; // value_dim (= key_dim for simplicity)
const SCALE: f32 = 0.5; // 1/sqrt(K) for K=4

// ===========================================================================
// Helper: state-only Variable bindings
// ===========================================================================

/// Bindings where only state is Variable; q/k/v/gate/beta are Constant.
///
/// This avoids the known multi-variable shape mismatch issue ([H,K] vs [H,K,V])
/// and produces clean single-variable bounds through the state evolution.
fn state_only_bindings(
    h: usize,
    k: usize,
    v: usize,
    gate_val: f32,
    beta_val: f32,
) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, k]), 0.1)), // q
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, k]), 0.2)), // k
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, v]), 0.3)), // v
        TensorParamBinding::Variable,                                               // state
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1, 1]), gate_val)), // gate
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1]), beta_val)), // beta
    ]
}

/// Create state bounds: lower=-range, upper=+range, shape [H, K, V].
fn state_bounds(h: usize, k: usize, v: usize, range: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[h, k, v]), -range),
        ArrayD::from_elem(IxDyn(&[h, k, v]), range),
    )
    .expect("valid state bounds")
}

// ===========================================================================
// 1. Single-step state update: S' = gate*S + outer(k, beta*v - beta*vr)
// ===========================================================================

/// Build a single-step state update graph with the new_state as output.
fn build_single_step_state_update(h: usize, k: usize, v: usize, scale: f32) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gdn_single_step_state_update");

    let q = b.add_input("q", &[h, k]);
    let ki = b.add_input("k", &[h, k]);
    let vi = b.add_input("v", &[h, v]);
    let state = b.add_input("state", &[h, k, v]);
    let gate = b.add_input("gate", &[h, 1, 1]);
    let beta = b.add_input("beta", &[h, 1]);

    let outputs = decompose_gated_delta_net(&mut b, q, ki, vi, state, gate, beta, scale, h, k, v);

    // Output the new_state, not the output vector
    b.build(outputs.new_state)
        .expect("valid single-step state update")
}

/// Single-step state update graph builds and validates.
#[test]
fn test_single_step_state_update_builds() {
    let def = build_single_step_state_update(H, K, V, SCALE);
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    assert_eq!(def.nodes[def.output.index()].shape, vec![H, K, V]);
}

/// Single-step state update: IBP propagation produces finite bounds.
#[test]
fn test_single_step_state_update_ibp() {
    let def = build_single_step_state_update(H, K, V, SCALE);
    let bindings = state_only_bindings(H, K, V, 0.9, 0.5);
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("single-step state update graph build");

    let input = state_bounds(H, K, V, 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Single-step state update IBP: [{lo_min:.4}, {hi_max:.4}], width={:.4}",
        hi_max - lo_min
    );
}

/// Single-step state update: CROWN propagation.
#[test]
fn test_single_step_state_update_crown() {
    let def = build_single_step_state_update(H, K, V, SCALE);
    let bindings = state_only_bindings(H, K, V, 0.9, 0.5);
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("single-step state update graph build");

    let input = state_bounds(H, K, V, 1.0);
    let (method, output, fallback) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN must succeed");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Single-step state update CROWN: method={method:?}, fallback={fallback:?}, \
         bounds=[{lo_min:.4}, {hi_max:.4}]"
    );
}

// ===========================================================================
// 2. Multi-step state evolution (4 steps)
// ===========================================================================

/// Build a 4-step state evolution graph.
///
/// Chains 4 GDN steps where the new_state from each step feeds into the next.
/// All steps share constant q/k/v/gate/beta. Only state0 is Variable.
/// The final output is the new_state after 4 steps.
fn build_four_step_state_evolution(
    h: usize,
    k: usize,
    v: usize,
    scale: f32,
    gate_val: f32,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let beta_val = 0.5f32;

    let mut b = TensorBlockBuilder::new("gdn_four_step_state_evolution");

    // Shared constant inputs for all steps
    let q = b.add_input("q", &[h, k]);
    let ki = b.add_input("k", &[h, k]);
    let vi = b.add_input("v", &[h, v]);
    let state0 = b.add_input("state0", &[h, k, v]);
    let gate = b.add_input("gate", &[h, 1, 1]);
    let beta = b.add_input("beta", &[h, 1]);

    let mut current_state = state0;
    for _step in 0..4 {
        let outputs =
            decompose_gated_delta_net(&mut b, q, ki, vi, current_state, gate, beta, scale, h, k, v);
        current_state = outputs.new_state;
    }

    let def = b
        .build(current_state)
        .expect("valid 4-step state evolution");

    // Only state0 is Variable
    let bindings = vec![
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, k]), 0.1)), // q
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, k]), 0.2)), // k
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, v]), 0.3)), // v
        TensorParamBinding::Variable,                                               // state0
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1, 1]), gate_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1]), beta_val)),
    ];

    (def, bindings)
}

/// 4-step state evolution graph builds and validates.
#[test]
fn test_four_step_state_evolution_builds() {
    let (def, _bindings) = build_four_step_state_evolution(H, K, V, SCALE, 0.9);
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    assert_eq!(def.nodes[def.output.index()].shape, vec![H, K, V]);
    // 4 chained decompositions reuse q/k/v/gate/beta inputs, creating many
    // intermediate nodes
    assert!(
        def.nodes.len() > 40,
        "expected >40 nodes for 4-step chain, got {}",
        def.nodes.len()
    );
}

/// 4-step state evolution: IBP propagation produces finite, bounded output.
///
/// The key verification: after 4 recurrence steps with gate=0.9 and
/// bounded initial state, state bounds must remain finite and not blow up.
#[test]
fn test_four_step_state_evolution_ibp() {
    let (def, bindings) = build_four_step_state_evolution(H, K, V, SCALE, 0.9);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("4-step state graph build");

    let input = state_bounds(H, K, V, 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("4-step IBP must succeed");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    eprintln!("4-step state evolution IBP: [{lo_min:.4}, {hi_max:.4}], width={width:.4}");

    // State bounds must be finite and reasonably bounded — the contractive
    // gate prevents exponential blowup.
    assert!(
        width < 1e6,
        "4-step state bounds width {width} exceeds 1e6 — possible unbounded growth"
    );
}

/// 4-step state evolution: CROWN propagation.
#[test]
fn test_four_step_state_evolution_crown() {
    let (def, bindings) = build_four_step_state_evolution(H, K, V, SCALE, 0.9);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("4-step state graph build");

    let input = state_bounds(H, K, V, 1.0);
    let (method, output, fallback) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN must succeed");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    eprintln!(
        "4-step state evolution CROWN: method={method:?}, fallback={fallback:?}, \
         bounds=[{lo_min:.4}, {hi_max:.4}], width={width:.4}"
    );
}

/// 4-step vs 1-step state bounds: verify growth is sub-exponential.
///
/// If the recurrence were unbounded, 4-step bounds would be exponentially
/// wider than 1-step. With gate=0.9, we expect at most moderate growth.
#[test]
fn test_state_bounds_growth_rate() {
    // Build and propagate 1-step
    let def1 = build_single_step_state_update(H, K, V, SCALE);
    let bindings1 = state_only_bindings(H, K, V, 0.9, 0.5);
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("1-step graph");
    let input1 = state_bounds(H, K, V, 1.0);
    let output1 = graph1.propagate_ibp(&input1).expect("1-step IBP");
    let (lo1, hi1) = bounds_min_max(&output1);
    let width1 = hi1 - lo1;

    // Build and propagate 4-step
    let (def4, bindings4) = build_four_step_state_evolution(H, K, V, SCALE, 0.9);
    let graph4 = tensor_kernel_to_graph(&def4, &bindings4).expect("4-step graph");
    let input4 = state_bounds(H, K, V, 1.0);
    let output4 = graph4.propagate_ibp(&input4).expect("4-step IBP");
    let (lo4, hi4) = bounds_min_max(&output4);
    let width4 = hi4 - lo4;

    let growth_ratio = width4 / width1.max(1e-10);
    eprintln!(
        "State bounds growth: 1-step width={width1:.4}, 4-step width={width4:.4}, \
         ratio={growth_ratio:.2}x"
    );

    // With gate=0.9 and IBP, bounds grow due to wrapping error but should
    // not be exponentially unbounded. Conservative threshold: 1000x.
    assert!(
        growth_ratio < 1000.0,
        "state bounds grew {growth_ratio:.1}x from 1 to 4 steps — \
         possible unbounded growth (expected sub-exponential with gate=0.9)"
    );
}

// ===========================================================================
// 3. Output computation: o = scale * q @ new_state
// ===========================================================================

/// Build a graph for the full single-step GDN output computation.
fn build_output_computation(h: usize, k: usize, v: usize, scale: f32) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gdn_output_computation");

    let q = b.add_input("q", &[h, k]);
    let ki = b.add_input("k", &[h, k]);
    let vi = b.add_input("v", &[h, v]);
    let state = b.add_input("state", &[h, k, v]);
    let gate = b.add_input("gate", &[h, 1, 1]);
    let beta = b.add_input("beta", &[h, 1]);

    let outputs = decompose_gated_delta_net(&mut b, q, ki, vi, state, gate, beta, scale, h, k, v);

    b.build(outputs.output).expect("valid output computation")
}

/// Output computation graph builds with correct shape [H, V].
#[test]
fn test_output_computation_builds() {
    let def = build_output_computation(H, K, V, SCALE);
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    assert_eq!(def.nodes[def.output.index()].shape, vec![H, V]);
}

/// Output computation: IBP with state-only Variable.
#[test]
fn test_output_computation_ibp() {
    let def = build_output_computation(H, K, V, SCALE);
    let bindings = state_only_bindings(H, K, V, 0.9, 0.5);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("output graph build");

    let input = state_bounds(H, K, V, 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Output computation IBP: [{lo_min:.4}, {hi_max:.4}], width={:.4}",
        hi_max - lo_min
    );
}

/// Output computation: CROWN propagation.
#[test]
fn test_output_computation_crown() {
    let def = build_output_computation(H, K, V, SCALE);
    let bindings = state_only_bindings(H, K, V, 0.9, 0.5);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("output graph build");

    let input = state_bounds(H, K, V, 1.0);
    let (method, output, fallback) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN must succeed");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Output computation CROWN: method={method:?}, fallback={fallback:?}, \
         bounds=[{lo_min:.4}, {hi_max:.4}]"
    );
}

/// Output scales linearly with scale parameter.
///
/// Doubling the scale should approximately double the output bounds width
/// (within IBP wrapping tolerance).
#[test]
fn test_output_scales_with_scale_parameter() {
    let bindings = state_only_bindings(H, K, V, 0.9, 0.5);
    let input = state_bounds(H, K, V, 1.0);

    let def_half = build_output_computation(H, K, V, 0.25);
    let graph_half = tensor_kernel_to_graph(&def_half, &bindings).expect("half-scale graph");
    let out_half = graph_half.propagate_ibp(&input).expect("IBP half");
    let (lo_h, hi_h) = bounds_min_max(&out_half);
    let width_half = hi_h - lo_h;

    let def_full = build_output_computation(H, K, V, 0.5);
    let graph_full = tensor_kernel_to_graph(&def_full, &bindings).expect("full-scale graph");
    let out_full = graph_full.propagate_ibp(&input).expect("IBP full");
    let (lo_f, hi_f) = bounds_min_max(&out_full);
    let width_full = hi_f - lo_f;

    let ratio = width_full / width_half.max(1e-10);
    eprintln!(
        "Output scale test: half-scale width={width_half:.4}, full-scale width={width_full:.4}, \
         ratio={ratio:.2}x (expected ~2.0)"
    );

    // IBP through matmul is linear in scale, so ratio should be close to 2.0
    assert!(
        ratio > 1.5 && ratio < 3.0,
        "scale ratio {ratio:.2} outside expected range [1.5, 3.0]"
    );
}

// ===========================================================================
// 4. Contractivity: gate < 1 implies state bounds converge
// ===========================================================================

/// Compare state bounds with different gate values (single-step).
///
/// Stronger decay (gate closer to 0) should produce tighter state bounds
/// because the recurrence discards more of the old state per step.
#[test]
fn test_state_contractivity_gate_strength() {
    let mut results = Vec::new();

    for &gate_val in &[0.5f32, 0.7, 0.9, 0.99] {
        let def = build_single_step_state_update(H, K, V, SCALE);
        let bindings = state_only_bindings(H, K, V, gate_val, 0.5);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");
        let input = state_bounds(H, K, V, 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP");
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("gate={gate_val:.2}: state width={width:.4}");
        results.push((gate_val, width));
    }

    // Stronger decay (lower gate) should produce tighter state bounds.
    if results.len() >= 2 {
        let (g_low, w_low) = results[0];
        let (g_high, w_high) = results[results.len() - 1];
        assert!(
            w_low <= w_high + 1e-3,
            "gate={g_low} (width={w_low:.4}) should be tighter than \
             gate={g_high} (width={w_high:.4})"
        );
        eprintln!(
            "Contractivity confirmed: gate={g_low} width={w_low:.4} <= \
             gate={g_high} width={w_high:.4}"
        );
    }
}

/// Multi-step contractivity: gate=0.5 bounds should be tighter than gate=0.9.
#[test]
fn test_multi_step_contractivity() {
    let mut results = Vec::new();

    for &gate_val in &[0.5f32, 0.9] {
        let (def, bindings) = build_four_step_state_evolution(H, K, V, SCALE, gate_val);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");
        let input = state_bounds(H, K, V, 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP");
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("4-step gate={gate_val:.2}: state width={width:.4}");
        results.push((gate_val, width));
    }

    if results.len() == 2 {
        let (g_low, w_low) = results[0];
        let (g_high, w_high) = results[1];
        eprintln!(
            "4-step contractivity: gate={g_low} width={w_low:.4}, \
             gate={g_high} width={w_high:.4}"
        );
        assert!(
            w_low <= w_high + 1e-2,
            "4-step gate={g_low} (width={w_low:.4}) should be tighter than \
             gate={g_high} (width={w_high:.4})"
        );
    }
}

/// Narrow input bounds produce tighter state evolution bounds.
#[test]
fn test_state_update_narrow_vs_wide_inputs() {
    let def = build_single_step_state_update(H, K, V, SCALE);
    let bindings = state_only_bindings(H, K, V, 0.9, 0.5);

    let graph_wide = tensor_kernel_to_graph(&def, &bindings).expect("wide graph");
    let graph_narrow = tensor_kernel_to_graph(&def, &bindings).expect("narrow graph");

    let wide_input = state_bounds(H, K, V, 1.0);
    let narrow_input = state_bounds(H, K, V, 0.5);

    let wide_output = graph_wide.propagate_ibp(&wide_input).expect("IBP wide");
    let narrow_output = graph_narrow
        .propagate_ibp(&narrow_input)
        .expect("IBP narrow");

    let (w_lo, w_hi) = bounds_min_max(&wide_output);
    let wide_width = w_hi - w_lo;
    let (n_lo, n_hi) = bounds_min_max(&narrow_output);
    let narrow_width = n_hi - n_lo;

    eprintln!(
        "State input sensitivity: wide width={wide_width:.4}, narrow width={narrow_width:.4}"
    );

    assert!(
        narrow_width <= wide_width + 1e-3,
        "narrow inputs ({narrow_width:.4}) should produce tighter bounds \
         than wide inputs ({wide_width:.4})"
    );
}

// ===========================================================================
// 5. verify_tensor_and_record integration
// ===========================================================================

/// Single-step state update records verification in status.
#[test]
fn test_single_step_state_verify_and_record() {
    use nn_verify::VerifyStatus;

    let def = build_single_step_state_update(H, K, V, SCALE);
    let bindings = state_only_bindings(H, K, V, 0.9, 0.5);
    let input = state_bounds(H, K, V, 1.0);

    let mut status = VerifyStatus::default();
    let result = nn_verify::verify_tensor_and_record(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("gdn_state_update_single_step"),
    )
    .expect("verify_tensor_and_record must succeed");

    assert!(result.verification.is_finite, "bounds must be finite");
    assert_eq!(result.num_variables, 1, "expected 1 variable input (state)");
    assert_bounds_valid(&result.output_bounds);
}

/// 4-step state evolution records verification in status.
#[test]
fn test_four_step_state_verify_and_record() {
    use nn_verify::VerifyStatus;

    let (def, bindings) = build_four_step_state_evolution(H, K, V, SCALE, 0.9);
    let input = state_bounds(H, K, V, 1.0);

    let mut status = VerifyStatus::default();
    let result = nn_verify::verify_tensor_and_record(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("gdn_state_update_four_step"),
    )
    .expect("verify_tensor_and_record must succeed");

    assert!(result.verification.is_finite, "bounds must be finite");
    assert_eq!(
        result.num_variables, 1,
        "expected 1 variable input (state0)"
    );
    assert_bounds_valid(&result.output_bounds);
    assert_bounds_width(&result.output_bounds, 1e6, "gdn_state_update_four_step");
}
