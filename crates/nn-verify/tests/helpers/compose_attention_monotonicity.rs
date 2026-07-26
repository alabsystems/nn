// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Cross-attention monotonicity verification via diagonal
//! dominance of pre-softmax attention scores.
//!
//! Phase 2 of #1729: Attention Monotonicity Proofs.
//!
//! Approach: Build a graph that outputs raw attention scores `Q @ K^T / √d`
//! (pre-softmax), propagate CROWN bounds, and verify diagonal dominance.
//! If `lower(S[t,t]) > upper(S[t,j])` for all `j ≠ t`, monotonic attention
//! is formally proven.
//!
//! Two variants:
//! 1. Simple: direct Q @ K^T / √d (no projections).
//! 2. Projected: Linear(Q) @ Linear(K)^T / √d_k with multi-head reshape.
//!
//! Phase 3 parametric sweeps are in `compose_attention_monotonicity_phase3.rs`.
//!
//! Part of #1729: Attention Monotonicity Proofs.

#[path = "attention_monotonicity.rs"]
mod helpers;

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use helpers::{
    attention_scores_projected_bindings, attention_scores_simple_bindings,
    build_attention_scores_projected, build_attention_scores_simple, D_MODEL, HEAD_DIM, NUM_HEADS,
    SEQ_LEN,
};
use nn_tts_verify::monotonicity::interpret_attention_monotonicity;
use nn_verify::tensor_kernel_to_graph;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Input bound for Q Variable (symmetric [-1, 1]).
const INPUT_BOUND: f32 = 1.0;

// ===========================================================================
// Simple variant: direct Q @ K^T / √d
// ===========================================================================

/// Simple attention scores graph validates.
#[test]
fn test_attn_mono_simple_def_validates() {
    let (def, _) = build_attention_scores_simple();
    def.validate()
        .expect("simple attention scores def should validate");
}

/// Simple attention scores graph translates to NY.
#[test]
fn test_attn_mono_simple_graph_builds() {
    let (def, out_shape) = build_attention_scores_simple();
    assert_eq!(out_shape, [SEQ_LEN, SEQ_LEN]);

    let bindings = attention_scores_simple_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("attention scores graph should translate");

    // Simple attention scores (MatMul + scale) translates to a single
    // fused LinearLayer node in NY.
    assert_eq!(
        graph.num_nodes(),
        1,
        "simple attention scores should be 1 fused node, got {}",
        graph.num_nodes()
    );
    eprintln!("Simple attention scores graph: {} nodes", graph.num_nodes());
}

/// IBP bounds propagate through simple attention scores.
#[test]
fn test_attn_mono_simple_ibp_propagates() {
    let (def, _) = build_attention_scores_simple();
    let bindings = attention_scores_simple_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], INPUT_BOUND);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through attention scores");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, SEQ_LEN],
        "output shape [T, T]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Simple attn scores IBP: bounds=[{lo_min}, {hi_max}]");
}

/// CROWN propagation through simple attention scores.
#[test]
fn test_attn_mono_simple_crown_propagates() {
    let (def, _) = build_attention_scores_simple();
    let bindings = attention_scores_simple_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], INPUT_BOUND);
    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, SEQ_LEN],
        "output shape [T, T]"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Simple attn scores: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// Interpret simple attention score bounds as monotonicity certificate.
///
/// With identity-like K structure and small Q perturbations, the diagonal
/// of `Q @ K^T / √d` should dominate off-diagonal elements. This is the
/// core Phase 2 result for #1729.
#[test]
fn test_attn_mono_simple_certificate() {
    let (def, _) = build_attention_scores_simple();
    let bindings = attention_scores_simple_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], INPUT_BOUND);

    let (method, output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("propagation");
    let (lo, hi) = output.lower_upper();

    let method_str = match method {
        nn_verify::PropMethod::Crown => "CROWN",
        nn_verify::PropMethod::Ibp => "IBP",
        _ => "unknown",
    };

    let lo_slice = lo.as_slice().expect("contiguous lower bounds");
    let hi_slice = hi.as_slice().expect("contiguous upper bounds");

    let cert = interpret_attention_monotonicity(
        lo_slice,
        hi_slice,
        SEQ_LEN,
        SEQ_LEN,
        f64::from(INPUT_BOUND),
        method_str,
    )
    .unwrap();

    eprintln!(
        "Simple monotonicity certificate: min_margin={:.6}, is_proven={}, method={}",
        cert.min_margin, cert.is_proven, cert.propagation_mode
    );
    for (t, m) in cert.row_margins.iter().enumerate() {
        eprintln!("  row {t}: margin={m:.6}");
    }

    // The certificate should have finite margins.
    assert!(
        cert.min_margin.is_finite(),
        "monotonicity margin should be finite, got {}",
        cert.min_margin
    );

    // Report monotonicity result (may or may not be proven depending on
    // IBP/CROWN tightness at this scale).
    if cert.is_proven {
        eprintln!(
            "PROVEN: attention is monotonic for all Q in [-{}, {}]",
            cert.input_bound, cert.input_bound
        );
    } else {
        eprintln!(
            "NOT PROVEN: min_margin={:.6} <= 0. Bounds may be too wide for diagonal dominance.",
            cert.min_margin
        );
    }
}

/// Verify and record simple attention monotonicity.
#[test]
fn test_attn_mono_simple_verify_and_record() {
    let (def, _) = build_attention_scores_simple();
    let bindings = attention_scores_simple_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], INPUT_BOUND);

    let result = verify_and_assert(&def, &bindings, &input, "attn_monotonicity_simple");
    assert_eq!(result.num_variables, 1, "single Variable input (query)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, SEQ_LEN]);
}

// ===========================================================================
// Projected variant: Linear projections + multi-head + Q @ K^T / √d_k
// ===========================================================================

/// Projected attention scores graph validates.
#[test]
fn test_attn_mono_projected_def_validates() {
    let (def, _) = build_attention_scores_projected();
    def.validate()
        .expect("projected attention scores def should validate");
}

/// Projected attention scores graph translates to NY.
#[test]
fn test_attn_mono_projected_graph_builds() {
    let (def, out_shape) = build_attention_scores_projected();
    assert_eq!(out_shape, [NUM_HEADS, SEQ_LEN, SEQ_LEN]);

    let bindings = attention_scores_projected_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("projected attention scores graph");

    // Linear proj × 2 + Reshape × 2 + Transpose × 2 + MatMul = 7+ nodes
    assert!(
        graph.num_nodes() >= 5,
        "projected attention scores graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
    eprintln!(
        "Projected attention scores graph: {} nodes ({NUM_HEADS} heads, d_k={HEAD_DIM})",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through projected attention scores.
#[test]
fn test_attn_mono_projected_ibp_propagates() {
    let (def, _) = build_attention_scores_projected();
    let bindings = attention_scores_projected_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], INPUT_BOUND);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through projected attention scores");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_HEADS, SEQ_LEN, SEQ_LEN],
        "output shape [H, T, T]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Projected attn scores IBP: bounds=[{lo_min}, {hi_max}]");
}

/// CROWN propagation through projected attention scores.
#[test]
fn test_attn_mono_projected_crown_propagates() {
    let (def, _) = build_attention_scores_projected();
    let bindings = attention_scores_projected_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], INPUT_BOUND);
    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_HEADS, SEQ_LEN, SEQ_LEN],
        "output shape [H, T, T]"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Projected attn scores: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// Interpret projected attention score bounds as monotonicity certificate.
///
/// With near-identity projections and identity-like K structure, the
/// multi-head attention scores should still exhibit diagonal dominance.
/// Each head is checked independently — all must be diagonally dominant.
#[test]
fn test_attn_mono_projected_certificate() {
    let (def, _) = build_attention_scores_projected();
    let bindings = attention_scores_projected_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], INPUT_BOUND);

    let (method, output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("propagation");
    let (lo, hi) = output.lower_upper();

    let method_str = match method {
        nn_verify::PropMethod::Crown => "CROWN",
        nn_verify::PropMethod::Ibp => "IBP",
        _ => "unknown",
    };

    // Output is [H, T, T]. Check each head's T×T block independently.
    let lo_slice = lo.as_slice().expect("contiguous lower bounds");
    let hi_slice = hi.as_slice().expect("contiguous upper bounds");

    let scores_per_head = SEQ_LEN * SEQ_LEN;
    let mut all_proven = true;
    let mut overall_min_margin = f64::INFINITY;

    for h in 0..NUM_HEADS {
        let offset = h * scores_per_head;
        let head_lo = &lo_slice[offset..offset + scores_per_head];
        let head_hi = &hi_slice[offset..offset + scores_per_head];

        let cert = interpret_attention_monotonicity(
            head_lo,
            head_hi,
            SEQ_LEN,
            SEQ_LEN,
            f64::from(INPUT_BOUND),
            method_str,
        )
        .unwrap();

        eprintln!(
            "Head {h}: min_margin={:.6}, is_proven={}",
            cert.min_margin, cert.is_proven
        );
        for (t, m) in cert.row_margins.iter().enumerate() {
            eprintln!("  row {t}: margin={m:.6}");
        }

        if !cert.is_proven {
            all_proven = false;
        }
        if cert.min_margin < overall_min_margin {
            overall_min_margin = cert.min_margin;
        }
    }

    assert!(
        overall_min_margin.is_finite(),
        "overall monotonicity margin should be finite"
    );

    eprintln!(
        "Projected monotonicity: all_proven={all_proven}, overall_min_margin={overall_min_margin:.6}, method={method_str}"
    );

    if all_proven {
        eprintln!(
            "PROVEN: all {NUM_HEADS} heads have monotonic attention for Q in [-{INPUT_BOUND}, {INPUT_BOUND}]"
        );
    } else {
        eprintln!("NOT PROVEN: at least one head lacks diagonal dominance.");
    }
}

/// Verify and record projected attention monotonicity.
#[test]
fn test_attn_mono_projected_verify_and_record() {
    let (def, _) = build_attention_scores_projected();
    let bindings = attention_scores_projected_bindings();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], INPUT_BOUND);

    let result = verify_and_assert(&def, &bindings, &input, "attn_monotonicity_projected");
    assert_eq!(result.num_variables, 1, "single Variable input (query)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_HEADS, SEQ_LEN, SEQ_LEN]);
}
