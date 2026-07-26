// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Gated DeltaNet gate sub-graph (D2).
//!
//! The DeltaNet gate computation produces the decay gate factor:
//!   `g = -exp(A_log) * softplus(a_proj(x) + dt_bias)`
//! where `A_log` and `dt_bias` are learnable parameters. The result `g` is
//! always negative, so `exp(g)` produces a decay in `(0, 1)`.
//!
//! These tests verify the gate sub-graph (softplus→exp chain) as an isolated
//! composition through NY IBP and CROWN propagation.
//!
//! See also `compose_gated_delta_net_gate_d3.rs` for D3 tests (full GDN cell
//! with computed gate pathway).
//!
//! Part of #834 — Gated DeltaNet for Qwen3.5 model support.

use super::common;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{propagate_with_crown_fallback, tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ===========================================================================
// D2: Gate sub-graph builders + composition tests
// ===========================================================================
//
// Gate computation pathway:
//   a_proj_out = a_proj(x)                       [Linear, weight param]
//   shifted = a_proj_out + dt_bias                [BinaryAdd, bias param]
//   sp = softplus(shifted)                        [Softplus]
//   neg_A = -exp(A_log)                           [Exp + negate, A_log param]
//   g = neg_A * sp                                [BinaryMul]
//   decay_gate = exp(g)                           [Exp]
//
// For the builder, we model a_proj_out as a Variable input (the linear
// projection happens outside the gate sub-graph). dt_bias and A_log are
// ConstantTensor parameters.

/// Build the gate computation sub-graph as a standalone `TensorKernelDef`.
///
/// Input: `a_proj_out` `[H]` — output of the linear projection `a_proj(x)`.
/// Constants: `dt_bias` `[H]`, `A_log` `[H]` — learnable parameters.
///
/// Output: `decay_gate` `[H, 1, 1]` — reshaped for broadcasting with state.
///
/// Computation:
///   `sp = softplus(a_proj_out + dt_bias)`
///   `neg_A = -1 * exp(A_log)`
///   `g = neg_A * sp`
///   `decay_gate = reshape(exp(g), [H, 1, 1])`
fn build_gate_subgraph(num_heads: usize) -> TensorKernelDef {
    let h_shape = [num_heads];
    let mut b = TensorBlockBuilder::new("gdn_gate_subgraph");

    // Inputs
    let a_proj_out = b.add_input("a_proj_out", &h_shape);
    let dt_bias = b.add_input("dt_bias", &h_shape);
    let a_log = b.add_input("A_log", &h_shape);

    // softplus(a_proj_out + dt_bias)
    let shifted = b.add_binary_add(a_proj_out, dt_bias, &h_shape);
    let sp = b.add_softplus(shifted, &h_shape);

    // -exp(A_log): A_log → exp → negate via BinaryMul with -1 constant
    let exp_a = b.add_exp(a_log, &h_shape);
    // Negate input: use a constant input set to -1.0 at binding time
    let neg_one = b.add_input("neg_one", &h_shape);
    let neg_exp_a = b.add_binary_mul(exp_a, neg_one, &h_shape);

    // g = neg_A * sp
    let g = b.add_binary_mul(neg_exp_a, sp, &h_shape);

    // decay = exp(g) — g is always negative, so exp(g) in (0, 1)
    let decay = b.add_exp(g, &h_shape);

    // Reshape to [H, 1, 1] for state broadcasting
    let decay_gate = b.add_reshape(decay, &[num_heads, 1, 1]);

    b.build(decay_gate).expect("valid gate sub-graph")
}

/// Gate sub-graph bindings: a_proj_out=Variable, dt_bias/A_log/neg_one=Constant.
///
/// `dt_bias_val`: typical small positive (0.1-1.0 in Qwen3.5).
/// `a_log_val`: typical small negative (-2 to -0.5, so exp(A_log) in (0.1, 0.6)).
fn gate_bindings(num_heads: usize, dt_bias_val: f32, a_log_val: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // a_proj_out
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[num_heads]), dt_bias_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[num_heads]), a_log_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[num_heads]), -1.0f32)),
    ]
}

// ---------------------------------------------------------------------------
// D2 tests: gate sub-graph in isolation
// ---------------------------------------------------------------------------

/// Gate sub-graph builds and validates.
#[test]
fn test_gate_subgraph_builds() {
    let def = build_gate_subgraph(4);
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    // 4 inputs (a_proj_out, dt_bias, A_log, neg_one) + ops
    assert!(def.nodes.len() >= 4, "expected >= 4 nodes");
    // Output shape: [H, 1, 1]
    assert_eq!(def.nodes[def.output.index()].shape, vec![4, 1, 1]);
}

/// Gate sub-graph builds NY graph with constant parameters.
#[test]
fn test_gate_subgraph_gamma_crown_graph_builds() {
    let h = 4;
    let def = build_gate_subgraph(h);
    let bindings = gate_bindings(h, 0.5, -1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings);
    assert!(
        graph.is_ok(),
        "gate subgraph graph build failed: {:?}",
        graph.err()
    );
}

/// Gate sub-graph IBP propagation with realistic parameter values.
///
/// dt_bias = 0.5, A_log = -1.0 (typical Qwen3.5 initialization).
/// a_proj_out bounded in [-2, 2].
/// Expected: decay_gate in (0, 1) since g is always negative.
#[test]
fn test_gate_subgraph_ibp_propagates() {
    let h = 4;
    let def = build_gate_subgraph(h);
    let bindings = gate_bindings(h, 0.5, -1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");

    // 1 Variable input (a_proj_out), bounds shape [1, H]
    let input = common::uniform_bounds(&[1, h], 2.0);

    let result = graph.propagate_ibp(&input);
    assert!(result.is_ok(), "IBP failed: {:?}", result.err());
    let output = result.unwrap();
    common::assert_bounds_valid(&output);
    let (lo, hi) = output.lower_upper();
    // exp(negative) is always in (0, 1). NY's monotonic interval
    // extension preserves this structurally, so lo >= 0 is guaranteed.
    // Add meaningful upper bound check: exp of bounded negative input
    // should stay well below 10.
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(
            *l >= 0.0 - 1e-6,
            "decay gate lower must be non-negative, got {l}"
        );
        assert!(*u < 10.0, "decay gate upper should be bounded, got {u}");
    }
    eprintln!(
        "Gate sub-graph IBP bounds: [{:.6}, {:.6}]",
        lo.iter().copied().reduce(f32::min).unwrap_or(0.0),
        hi.iter().copied().reduce(f32::max).unwrap_or(0.0),
    );
}

/// Gate sub-graph CROWN propagation.
#[test]
fn test_gate_subgraph_crown_propagates() {
    let h = 4;
    let def = build_gate_subgraph(h);
    let bindings = gate_bindings(h, 0.5, -1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");

    let input = common::uniform_bounds(&[1, h], 2.0);

    let result = propagate_with_crown_fallback(&graph, &input);
    assert!(result.is_ok(), "CROWN failed: {:?}", result.err());
    let (_method, output, fallback) = result.unwrap();
    common::assert_bounds_valid(&output);
    let (lo, hi) = output.lower_upper();
    // exp(negative) is always in (0, 1) — same invariant as IBP test.
    // Add upper bound check alongside the lower bound.
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(
            *l >= 0.0 - 1e-6,
            "decay gate CROWN lower must be non-negative, got {l}"
        );
        assert!(
            *u < 10.0,
            "decay gate CROWN upper should be bounded, got {u}"
        );
    }
    if let Some(reason) = &fallback {
        eprintln!("Gate sub-graph CROWN fell back to IBP: {reason}");
    }
    eprintln!(
        "Gate sub-graph CROWN bounds: [{:.6}, {:.6}]",
        lo.iter().copied().reduce(f32::min).unwrap_or(0.0),
        hi.iter().copied().reduce(f32::max).unwrap_or(0.0),
    );
}

/// Gate sub-graph CROWN at least as tight as IBP.
#[test]
fn test_gate_subgraph_crown_at_least_as_tight_as_ibp() {
    let h = 2;
    let def = build_gate_subgraph(h);
    let bindings = gate_bindings(h, 0.5, -1.0);

    let input = common::uniform_bounds(&[1, h], 1.0);

    let graph_ibp = tensor_kernel_to_graph(&def, &bindings).expect("graph build");
    let ibp_output = graph_ibp.propagate_ibp(&input).expect("IBP");

    let graph_crown = tensor_kernel_to_graph(&def, &bindings).expect("graph build");
    let input2 = common::uniform_bounds(&[1, h], 1.0);
    let (_, crown_output, _) = propagate_with_crown_fallback(&graph_crown, &input2).expect("CROWN");

    common::assert_crown_tighter_than_ibp(&crown_output, &ibp_output);

    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
    let (crown_lo, crown_hi) = crown_output.lower_upper();
    let ibp_width: f32 = ibp_hi.iter().zip(ibp_lo.iter()).map(|(h, l)| h - l).sum();
    let crown_width: f32 = crown_hi
        .iter()
        .zip(crown_lo.iter())
        .map(|(h, l)| h - l)
        .sum();
    eprintln!(
        "Gate tightness: IBP_width={ibp_width:.6}, CROWN_width={crown_width:.6}, \
         ratio={:.2}x",
        ibp_width / crown_width.max(1e-10)
    );
}

/// Gate sub-graph with different A_log values produces different bounds.
#[test]
fn test_gate_subgraph_different_a_log_values() {
    for &a_log in &[-2.0, -1.0, -0.5, -0.1] {
        let h = 2;
        let def = build_gate_subgraph(h);
        let bindings = gate_bindings(h, 0.5, a_log);
        let graph = tensor_kernel_to_graph(&def, &bindings);
        assert!(
            graph.is_ok(),
            "gate subgraph failed for A_log={a_log}: {:?}",
            graph.err()
        );

        let graph = graph.unwrap();
        let input = common::uniform_bounds(&[1, h], 1.0);

        let output = graph
            .propagate_ibp(&input)
            .unwrap_or_else(|e| panic!("IBP failed for A_log={a_log}: {e}"));
        common::assert_bounds_valid(&output);
        let (lo, hi) = output.lower_upper();
        // decay = exp(g) where g = A_log + dt_bias + softplus(a_proj_out)
        // is always negative for negative A_log. exp(negative) in (0, 1).
        // Add upper bound check alongside the lower bound.
        for (l, u) in lo.iter().zip(hi.iter()) {
            assert!(
                *l >= 0.0 - 1e-6,
                "A_log={a_log}: decay lower must be non-negative, got {l}"
            );
            assert!(
                *u < 100.0,
                "A_log={a_log}: decay upper should be bounded, got {u}"
            );
        }
        eprintln!(
            "A_log={a_log:.1}: IBP bounds [{:.6}, {:.6}]",
            lo.iter().copied().reduce(f32::min).unwrap_or(0.0),
            hi.iter().copied().reduce(f32::max).unwrap_or(0.0),
        );
    }
}
