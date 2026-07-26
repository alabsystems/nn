// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: zonotope tightening and parallel verification for MHA.
//!
//! Tests that NY's automatic zonotope tightening activates on
//! nn-generated graphs with Q@K^T patterns, and that parallel position
//! verification produces consistent results.
//!
//! Part of #813.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{
    parallel_verify_positions, parallel_verify_with_method, tensor_kernel_to_graph,
    verify_tensor_and_record, BoundedTensor, ParallelVerifyConfig, PropMethod, TensorParamBinding,
    VerifyStatus,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Helpers (shared with compose_multi_head_attention.rs)
// ---------------------------------------------------------------------------

fn build_mha_kernel(
    name: &str,
    seq_len: usize,
    model_dim: usize,
    num_heads: usize,
    mask: AttentionMask,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("input", &[seq_len, model_dim]);
    let q_w = b.add_input("q_weight", &[model_dim, model_dim]);
    let k_w = b.add_input("k_weight", &[model_dim, model_dim]);
    let v_w = b.add_input("v_weight", &[model_dim, model_dim]);
    let out_w = b.add_input("out_weight", &[model_dim, model_dim]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            num_heads,
            mask,
            &[seq_len, model_dim],
        )
        .expect("valid MHA");
    b.build(out).expect("valid kernel")
}

fn mha_bindings(model_dim: usize) -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[model_dim, model_dim]), 0.02f32);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w),
    ]
}

fn mha_input_bounds(seq_len: usize, model_dim: usize) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(&[seq_len, model_dim]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[seq_len, model_dim]), 1.0f32);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

// ---------------------------------------------------------------------------
// Zonotope tightening verification
// ---------------------------------------------------------------------------

/// Verify that IBP through nn-generated MHA produces finite, reasonable bounds.
///
/// The MHA graph uses monolithic SelfAttentionLayer (not decomposed MatMul),
/// so NY's automatic zonotope tightening for Q@K^T does NOT activate
/// in this configuration. This test documents the baseline behavior.
///
/// To enable zonotope tightening, the attention would need to be decomposed
/// into explicit Linear → MatMul(transpose_b=true) → Softmax → MatMul nodes
/// in the NY graph, so the graph-level zonotope detector can trace
/// Q and K back to shared Linear projections.
#[test]
fn test_mha_ibp_baseline_without_zonotope() {
    let (t, d, h) = (4, 8, 2);
    let def = build_mha_kernel("mha_zonotope_baseline", t, d, h, AttentionMask::Standard);
    let bindings = mha_bindings(d);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("MHA graph");

    let input = mha_input_bounds(t, d);
    let output = graph.propagate_ibp(&input).expect("IBP through MHA");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[t, d], "output shape [T, D]");

    // Verify finite and valid bounds
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite(), "lower finite: {l}");
        assert!(u.is_finite(), "upper finite: {u}");
        assert!(l <= u, "lower <= upper: {l} <= {u}");
    }

    // Record max width for comparison with future zonotope-enabled path
    let max_width = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, u)| u - l)
        .fold(0.0f32, f32::max);

    // With small weights (0.02) and input [-1, 1], bounds should be moderate
    assert!(
        max_width < 1e6,
        "IBP bounds should not be vacuously wide (max_width={max_width})"
    );
}

/// Verify that CROWN on MHA produces a SOUND enclosure of the IBP bounds.
///
/// SelfAttentionLayer does not implement a tightening CROWN backward pass, so
/// `GraphNetwork::propagate_crown` runs a conservative backward relaxation
/// across the attention boundary. The result is NOT identical to pure IBP:
/// the CROWN relaxation of the attention sub-graph is honestly *looser* than
/// IBP here (observed crown_lo ≈ -0.17 vs ibp_lo ≈ -0.028), because the
/// linear relaxation over-approximates the softmax/value product more than the
/// element-wise IBP interval does. This is mathematically sound — CROWN is not
/// guaranteed to be tighter than IBP for every graph — just not as tight.
///
/// The original test asserted crown == ibp ("both are IBP"). That premise was
/// already false before ny 5de589a6 (verified by checking out ny@dced3db2: the
/// same crown_lo ≈ -0.17 appears, ~1 ULP away), so this is NOT a 5de589a6
/// regression — the test's equality expectation was simply stale. The sound,
/// load-bearing property is *enclosure*: because IBP is sound, the true output
/// range ⊆ [ibp_lo, ibp_hi]; CROWN here is looser, so CROWN must be a SUPERSET
/// of the IBP interval (crown_lo ≤ ibp_lo and crown_hi ≥ ibp_hi), and is
/// therefore itself a sound enclosure of the true range. We assert exactly
/// that (plus finiteness and lo ≤ hi). If CROWN ever returned a bound INSIDE
/// the IBP interval without a documented tightening pass, that would indicate
/// an unsound CROWN and this assertion would (correctly) fail.
#[test]
fn test_mha_crown_encloses_ibp_soundly() {
    let (t, d, h) = (4, 8, 2);
    let def = build_mha_kernel("mha_crown_tight", t, d, h, AttentionMask::Standard);
    let bindings = mha_bindings(d);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("MHA graph");

    let input = mha_input_bounds(t, d);

    // IBP baseline (sound; true range ⊆ [ibp_lo, ibp_hi]).
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();

    // CROWN over the MHA graph: conservative backward relaxation across the
    // SelfAttention boundary — looser than IBP but still a sound enclosure.
    let crown_output = graph
        .propagate_crown(&input)
        .expect("CROWN should return Ok over MHA graph");
    let (crown_lo, crown_hi) = crown_output.lower_upper();

    // Soundness: CROWN must ENCLOSE the (sound) IBP interval — i.e. be a
    // superset, never a tighter interval carved inside it. Tolerance 1e-6
    // absorbs f32 accumulation drift in the equal-when-exact case.
    for ((&cl, &cu), (&il, &iu)) in crown_lo
        .iter()
        .zip(crown_hi.iter())
        .zip(ibp_lo.iter().zip(ibp_hi.iter()))
    {
        assert!(cl.is_finite(), "CROWN lower must be finite: {cl}");
        assert!(cu.is_finite(), "CROWN upper must be finite: {cu}");
        assert!(cl <= cu, "CROWN lower <= upper: {cl} <= {cu}");
        // Enclosure is the load-bearing soundness property: CROWN must be a SUPERSET
        // of the sound IBP interval (crown_lo <= ibp_lo, crown_hi >= ibp_hi). The slack
        // must only absorb genuine f32 rounding (a few ULPs at the value's magnitude),
        // NOT a meaningful CROWN-tighter-than-IBP violation — so it is ULP-scaled, not a
        // fixed 1e-6 (which previously admitted up to a 1e-6 enclosure violation in the
        // UNSOUND direction).
        let lo_tol = 8.0 * f32::EPSILON * cl.abs().max(il.abs());
        let hi_tol = 8.0 * f32::EPSILON * cu.abs().max(iu.abs());
        assert!(
            cl <= il + lo_tol,
            "CROWN lower must enclose (be <=) IBP lower: crown={cl}, ibp={il}, tol={lo_tol}"
        );
        assert!(
            cu >= iu - hi_tol,
            "CROWN upper must enclose (be >=) IBP upper: crown={cu}, ibp={iu}, tol={hi_tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// Parallel position verification
// ---------------------------------------------------------------------------
//
// MHA graphs are NOT compatible with per-position parallel verification
// because SelfAttentionLayer's softmax operates across the full sequence
// dimension. Slicing individual positions along axis=0 creates shape
// mismatches (the layer expects [T, D] but receives [1, D]).
//
// Parallel verification works for position-independent graphs (e.g., Linear,
// element-wise ops). The tests below use Linear graphs built through the
// nn-dsl → NY translation pipeline to exercise the parallel API
// at the integration level.
// ---------------------------------------------------------------------------

/// Build a simple Linear kernel through nn-dsl for parallel testing.
fn build_linear_kernel(name: &str, seq_len: usize, dim: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("input", &[seq_len, dim]);
    let weight = b.add_input("weight", &[dim, dim]);
    let out = b.add_linear(input, weight, None, &[seq_len, dim]);
    b.build(out).expect("valid linear kernel")
}

fn linear_bindings(dim: usize) -> Vec<TensorParamBinding> {
    let eye = ArrayD::from_elem(IxDyn(&[dim, dim]), 0.0f32);
    let mut eye_mut = eye;
    for i in 0..dim {
        eye_mut[IxDyn(&[i, i])] = 1.0;
    }
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(eye_mut),
    ]
}

fn linear_input_bounds(seq_len: usize, dim: usize) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(&[seq_len, dim]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[seq_len, dim]), 1.0f32);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

/// Parallel IBP through nn-translated Linear graph matches serial IBP.
#[test]
fn test_linear_parallel_ibp_matches_serial() {
    let (t, d) = (6, 4);
    let def = build_linear_kernel("lin_parallel", t, d);
    let bindings = linear_bindings(d);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("Linear graph");

    let input = linear_input_bounds(t, d);

    // Serial IBP
    let serial_output = graph.propagate_ibp(&input).expect("serial IBP");

    // Parallel IBP across sequence positions (axis=0)
    let parallel_result = parallel_verify_positions(&graph, &input, 0, None).expect("parallel IBP");

    assert_eq!(
        parallel_result.num_positions, t,
        "should verify {t} positions"
    );

    let (s_lo, s_hi) = serial_output.lower_upper();
    let (p_lo, p_hi) = parallel_result.output_bounds.lower_upper();
    assert_eq!(s_lo.shape(), p_lo.shape(), "shapes must match");

    // Identity linear: parallel and serial should produce identical bounds
    for ((&sl, &sh), (&pl, &ph)) in s_lo
        .iter()
        .zip(s_hi.iter())
        .zip(p_lo.iter().zip(p_hi.iter()))
    {
        assert!(pl.is_finite(), "parallel lower finite");
        assert!(ph.is_finite(), "parallel upper finite");
        assert!(pl <= ph, "parallel lower <= upper");
        assert!(
            (sl - pl).abs() < 1e-6,
            "lower mismatch: serial={sl}, parallel={pl}"
        );
        assert!(
            (sh - ph).abs() < 1e-6,
            "upper mismatch: serial={sh}, parallel={ph}"
        );
    }
}

/// Parallel CROWN verification through nn-translated Linear graph.
#[test]
fn test_linear_parallel_crown() {
    let (t, d) = (4, 4);
    let def = build_linear_kernel("lin_par_crown", t, d);
    let bindings = linear_bindings(d);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("Linear graph");

    let input = linear_input_bounds(t, d);

    let config = ParallelVerifyConfig::crown().with_max_threads(2);
    let result =
        parallel_verify_positions(&graph, &input, 0, Some(&config)).expect("parallel CROWN");

    assert_eq!(result.num_positions, t);
    let (lo, hi) = result.output_bounds.lower_upper();
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite(), "lower finite: {l}");
        assert!(u.is_finite(), "upper finite: {u}");
        assert!(l <= u, "lower <= upper: {l} <= {u}");
    }
}

/// Convenience function: parallel_verify_with_method through nn pipeline.
#[test]
fn test_linear_parallel_verify_with_method() {
    let t = 4;
    let d = 4;
    let def = build_linear_kernel("lin_method", t, d);
    let bindings = linear_bindings(d);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("Linear graph");

    let input = linear_input_bounds(t, d);

    let output =
        parallel_verify_with_method(&graph, &input, 0, PropMethod::Ibp).expect("parallel IBP");

    let (lo, hi) = output.lower_upper();
    let expected_shape = [t, d];
    assert_eq!(lo.shape(), &expected_shape);
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite());
        assert!(u.is_finite());
        assert!(l <= u);
    }
}

/// MHA graphs reject per-position parallel verification due to cross-position
/// attention (softmax over sequence dimension). This documents the expected
/// behavior — parallel verification is for position-independent graphs.
#[test]
fn test_mha_parallel_rejects_position_slicing() {
    let (t, d, h) = (4, 8, 2);
    let def = build_mha_kernel("mha_par_reject", t, d, h, AttentionMask::Standard);
    let bindings = mha_bindings(d);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("MHA graph");

    let input = mha_input_bounds(t, d);

    // Per-position slicing should fail because SelfAttentionLayer expects
    // the full sequence dimension for softmax computation.
    let result = parallel_verify_positions(&graph, &input, 0, None);
    assert!(
        result.is_err(),
        "MHA parallel per-position should fail (attention is position-dependent)"
    );
}

// ---------------------------------------------------------------------------
// verify_tensor_and_record integration (AC6)
// ---------------------------------------------------------------------------

/// Record parallel Linear verification in VerifyStatus under "parallel_linear".
///
/// Exercises the full pipeline: nn-dsl → NY translation → IBP
/// propagation → verify_tensor_and_record. Parallel bounds validated against
/// serial for consistency.
#[test]
fn test_parallel_linear_verify_and_record() {
    let (t, d) = (6, 4);
    let def = build_linear_kernel("lin_record", t, d);
    let bindings = linear_bindings(d);
    let input = linear_input_bounds(t, d);

    // Record via verify_tensor_and_record (serial path)
    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("parallel_linear"),
    )
    .expect("verify_tensor_and_record pipeline");

    // Output bounds should be finite and valid
    let (lo, hi) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[6, 4], "recorded output shape");
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite(), "recorded lower finite: {l}");
        assert!(u.is_finite(), "recorded upper finite: {u}");
        assert!(l <= u, "recorded lower <= upper: {l} <= {u}");
    }

    // Parallel verification should produce matching bounds
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let parallel_result = parallel_verify_positions(&graph, &input, 0, None).expect("parallel IBP");

    let (p_lo, p_hi) = parallel_result.output_bounds.lower_upper();
    for ((&rl, &ru), (&pl, &pu)) in lo.iter().zip(hi.iter()).zip(p_lo.iter().zip(p_hi.iter())) {
        assert!(
            (rl - pl).abs() < 1e-6,
            "serial vs parallel lower: {rl} vs {pl}"
        );
        assert!(
            (ru - pu).abs() < 1e-6,
            "serial vs parallel upper: {ru} vs {pu}"
        );
    }
}

/// Record MHA serial verification in VerifyStatus under "mha_zonotope".
///
/// MHA cannot use parallel per-position verification (softmax is
/// position-dependent), so we verify the full graph serially and record.
#[test]
fn test_mha_serial_verify_and_record() {
    let (t, d, h) = (4, 8, 2);
    let def = build_mha_kernel("mha_record", t, d, h, AttentionMask::Standard);
    let bindings = mha_bindings(d);
    let input = mha_input_bounds(t, d);

    let mut status = VerifyStatus::default();
    let result =
        verify_tensor_and_record(&mut status, &def, &bindings, &input, Some("mha_zonotope"))
            .expect("verify_tensor_and_record pipeline for MHA");

    let (lo, hi) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t, d], "MHA output shape [T, D]");

    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite(), "MHA lower finite: {l}");
        assert!(u.is_finite(), "MHA upper finite: {u}");
        assert!(l <= u, "MHA lower <= upper: {l} <= {u}");
    }

    // Bounds should stay moderate with small weights (0.02)
    let max_width = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, u)| u - l)
        .fold(0.0f32, f32::max);
    assert!(
        max_width < 1e3,
        "MHA IBP bounds should not be vacuously wide (max_width={max_width})"
    );
}
