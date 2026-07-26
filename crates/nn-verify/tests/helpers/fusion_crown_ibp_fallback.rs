// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the CROWN→IBP fallback path in fusion verification (#486).
//!
//! `propagate_with_crown_fallback` tries CROWN first, falls back to IBP on
//! CROWN failure. This module tests the reachable code paths:
//!
//! - **CROWN success**: well-formed graph → `(Crown, bounds, None)`
//! - **Both fail**: structurally invalid graph → `Err` (IBP error propagated)
//!
//! ## Defense-in-depth note
//!
//! The CROWN-fails-IBP-succeeds branch (fusion.rs:212-215) is defense-in-depth
//! code that is **unreachable with current NY** (Feb 2026). NY's
//! CROWN backward catches most errors internally via IBP fallback, returning
//! `Ok(ForwardFallback)` — not `Err`. The remaining `Err` cases (e.g.,
//! `NumericalInstability` from `non_finite_domain_guard`, `verify_split_path_bias_zero`)
//! require non-finite intermediate bounds, but NY's IBP layers use
//! `BoundedTensor::new()` which rejects Inf/NaN, preventing non-finite
//! intermediates from forming.
//!
//! If NY relaxes its Inf rejection (e.g., adds `new_allow_infinite`
//! paths to arithmetic layers), the fallback branch would become reachable:
//! a nonlinear layer (Sin, SiLU, etc.) receiving Inf pre-activation bounds
//! would trigger `NumericalInstability` in CROWN backward while IBP handles
//! Inf gracefully with conservative bounds.

use ny_propagate::layers::MulConstantLayer;
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};
use ndarray::{ArrayD, IxDyn};

/// Scalar input bounds as a 1-D BoundedTensor.
fn scalar_input_bounds(lower: f32, upper: f32) -> BoundedTensor {
    let lo = ArrayD::from_elem(IxDyn(&[1]), lower);
    let hi = ArrayD::from_elem(IxDyn(&[1]), upper);
    BoundedTensor::new(lo, hi).expect("valid scalar bounds")
}

// ---------------------------------------------------------------------------
// CROWN success path
// ---------------------------------------------------------------------------

#[test]
fn test_crown_succeeds_on_well_formed_graph() {
    // Baseline: a simple linear graph where CROWN succeeds.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "mul".to_string(),
        Layer::MulConstant(MulConstantLayer::scalar(2.0)),
    ));
    graph.set_output("mul".to_string());

    let input = scalar_input_bounds(-1.0, 1.0);

    let (method, output_bounds, crown_fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("should succeed on well-formed graph");

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for a simple linear graph"
    );
    assert!(
        crown_fallback_reason.is_none(),
        "no fallback reason when CROWN succeeds"
    );

    // Verify output bounds are correct: 2 * [-1, 1] = [-2, 2]
    let lower = output_bounds.lower();
    let upper = output_bounds.upper();
    assert!(
        (lower[[0]] - (-2.0)).abs() < 1e-6,
        "lower bound should be -2.0, got {}",
        lower[[0]]
    );
    assert!(
        (upper[[0]] - 2.0).abs() < 1e-6,
        "upper bound should be 2.0, got {}",
        upper[[0]]
    );
}

#[test]
fn test_crown_succeeds_method_and_no_fallback_reason() {
    // AC1 positive case: when CROWN succeeds, method is Crown and
    // crown_fallback_reason is None.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "identity".to_string(),
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
    ));
    graph.set_output("identity".to_string());

    let input = scalar_input_bounds(-5.0, 5.0);

    let (method, _output_bounds, crown_fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("should succeed");

    assert_eq!(method, PropMethod::Crown);
    assert!(crown_fallback_reason.is_none());
}

// ---------------------------------------------------------------------------
// Both-fail error propagation
// ---------------------------------------------------------------------------

#[test]
fn test_both_fail_propagates_ibp_error() {
    // When the graph is structurally invalid (dangling node reference),
    // both CROWN and IBP fail. The IBP error should propagate through `?`.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "broken".to_string(),
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
        vec!["nonexistent_node".to_string()],
    ));
    graph.set_output("broken".to_string());

    let input = scalar_input_bounds(-1.0, 1.0);

    let result = propagate_with_crown_fallback(&graph, &input);
    assert!(
        result.is_err(),
        "should fail when graph has dangling reference"
    );

    // The error message should indicate the structural problem.
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("nonexistent")
            || err_msg.contains("not found")
            || err_msg.contains("node"),
        "error should mention the missing node, got: {err_msg}"
    );
}

#[test]
fn test_both_fail_cycle_in_graph() {
    // A graph with a self-referencing node creates a cycle that topological_sort
    // rejects — both CROWN and IBP fail.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "cycle".to_string(),
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
        vec!["cycle".to_string()],
    ));
    graph.set_output("cycle".to_string());

    let input = scalar_input_bounds(-1.0, 1.0);

    let result = propagate_with_crown_fallback(&graph, &input);
    assert!(result.is_err(), "graph with cycle should fail");
}

// ---------------------------------------------------------------------------
// Contract tests for the helper function
// ---------------------------------------------------------------------------

#[test]
fn test_crown_success_returns_crown_method() {
    // Verify the contract: on CROWN success, the returned method is Crown.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "scale".to_string(),
        Layer::MulConstant(MulConstantLayer::scalar(0.5)),
    ));
    graph.set_output("scale".to_string());

    let input = scalar_input_bounds(-10.0, 10.0);

    let (method, output_bounds, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("should succeed");

    assert_eq!(method, PropMethod::Crown);
    assert!(fallback_reason.is_none());

    // Output bounds should be sound: 0.5 * [-10, 10] = [-5, 5]
    let lower = output_bounds.lower();
    let upper = output_bounds.upper();
    assert!(lower[[0]] <= -5.0 + 1e-6);
    assert!(upper[[0]] >= 5.0 - 1e-6);
}

#[test]
fn test_is_conclusive_invariant_crown_true() {
    // AC2 prerequisite: CROWN method maps to is_conclusive() == true.
    // This is tested directly in fusion_spec::tests::test_is_conclusive_crown_returns_true
    // but we also verify that the helper returns Crown for a valid graph,
    // which feeds into is_conclusive() via the FusionVerification.method field.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "pass".to_string(),
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
    ));
    graph.set_output("pass".to_string());

    let input = scalar_input_bounds(-1.0, 1.0);

    let (method, _, _) = propagate_with_crown_fallback(&graph, &input).expect("should succeed");
    // PropMethod::Crown → is_conclusive() == true
    assert_eq!(method, PropMethod::Crown);
    // PropMethod::Ibp → is_conclusive() == false (defense-in-depth branch)
    // This mapping is tested in fusion_spec::tests::test_is_conclusive_ibp_returns_false
}
