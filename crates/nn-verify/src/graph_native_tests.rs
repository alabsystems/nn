// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `graph_native.rs` — native NY layer dispatch.

use super::*;
use crate::graph::ParamBinding;
use ny_propagate::Layer;
use nn_dsl::ir::KernelDef;
use nn_dsl::test_kernels::{parse_kernel, snake_kernel};

/// Build a minimal silu_mul kernel for testing (name matching only).
fn silu_mul_kernel() -> KernelDef {
    parse_kernel("fn silu_mul(x: f32, up: f32) -> f32 { x * (1.0 / (1.0 + (-x).exp())) * up }")
}

/// Build a kernel with a non-matching name.
fn unrelated_kernel() -> KernelDef {
    parse_kernel("fn relu(x: f32) -> f32 { x.max(0.0) }")
}

// ---------------------------------------------------------------------------
// Snake native layer dispatch
// ---------------------------------------------------------------------------

/// Verify a snake graph has the correct structure: 1 node, Snake layer with expected a.
fn assert_snake_graph(graph: &GraphNetwork, expected_alpha: f32) {
    assert_eq!(
        graph.num_nodes(),
        1,
        "snake graph should have exactly 1 node"
    );
    assert_eq!(graph.output_name(), "snake_native");
    let node = graph.node("snake_native").expect("output node must exist");
    match node.layer() {
        Layer::Snake(snake) => {
            let alpha_val = snake.alpha()[0];
            assert_eq!(
                alpha_val.to_bits(),
                expected_alpha.to_bits(),
                "snake alpha should be {expected_alpha}, got {alpha_val}",
            );
        }
        other => panic!("expected Layer::Snake, got {}", other.layer_type()),
    }
    assert_eq!(
        node.inputs(),
        &["_input"],
        "snake node should take network input"
    );
}

#[test]
fn test_snake_native_valid_alpha() {
    let kernel = snake_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let graph = try_native_layer(&kernel, &bindings)
        .unwrap()
        .expect("snake with valid alpha should produce native layer");
    assert_snake_graph(&graph, 1.0);
}

#[test]
fn test_snake_native_large_alpha() {
    let kernel = snake_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(100.0)];
    let graph = try_native_layer(&kernel, &bindings)
        .unwrap()
        .expect("snake with large alpha should produce native layer");
    assert_snake_graph(&graph, 100.0);
}

#[test]
fn test_snake_native_small_alpha() {
    let kernel = snake_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(0.001)];
    let graph = try_native_layer(&kernel, &bindings)
        .unwrap()
        .expect("snake with small positive alpha should produce native layer");
    assert_snake_graph(&graph, 0.001);
}

#[test]
fn test_snake_native_zero_alpha_falls_through() {
    // alpha=0.0 should cause SnakeLayer::new to fail, falling through to decomposition
    let kernel = snake_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(0.0)];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "snake with alpha=0.0 should fall through to decomposition"
    );
}

#[test]
fn test_snake_native_negative_alpha_falls_through() {
    let kernel = snake_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(-1.0)];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "snake with negative alpha should fall through to decomposition"
    );
}

#[test]
fn test_snake_native_nan_alpha_falls_through() {
    let kernel = snake_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(f32::NAN)];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "snake with NaN alpha should fall through to decomposition"
    );
}

#[test]
fn test_snake_native_inf_alpha_falls_through() {
    let kernel = snake_kernel();
    let bindings = [
        ParamBinding::Variable,
        ParamBinding::Constant(f32::INFINITY),
    ];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "snake with Inf alpha should fall through to decomposition"
    );
}

#[test]
fn test_snake_native_both_variable_falls_through() {
    // Multi-variable: both x and alpha are Variable — should fall through.
    let kernel = snake_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "snake with both variable params should fall through"
    );
}

#[test]
fn test_snake_native_wrong_param_count_falls_through() {
    let kernel = snake_kernel();
    let bindings = [ParamBinding::Variable]; // only 1 param
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "snake with wrong param count should fall through"
    );
}

#[test]
fn test_snake_native_first_param_constant_falls_through() {
    // First param is Constant (not Variable) — should fall through.
    let kernel = snake_kernel();
    let bindings = [ParamBinding::Constant(1.0), ParamBinding::Constant(1.0)];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "snake with first param constant should fall through"
    );
}

// ---------------------------------------------------------------------------
// SiLU-Mul native layer dispatch
// ---------------------------------------------------------------------------

/// Verify a silu_mul graph with up != 1.0: 2 nodes (SiLU + MulConstant), correct scalar.
fn assert_silu_mul_graph_with_mul(graph: &GraphNetwork, expected_up: f32) {
    assert_eq!(
        graph.num_nodes(),
        2,
        "silu_mul with up!={expected_up} should have 2 nodes"
    );
    assert_eq!(graph.output_name(), "silu_mul_native");
    // Verify SiLU node
    let silu_node = graph.node("silu_native").expect("silu node must exist");
    assert!(
        matches!(silu_node.layer(), Layer::SiLU(_)),
        "first node should be SiLU, got {}",
        silu_node.layer().layer_type()
    );
    assert_eq!(
        silu_node.inputs(),
        &["_input"],
        "silu node should take network input"
    );
    // Verify MulConstant node
    let mul_node = graph.node("silu_mul_native").expect("mul node must exist");
    match mul_node.layer() {
        Layer::MulConstant(mc) => {
            let scalar_val = mc
                .constant()
                .iter()
                .next()
                .expect("scalar must have a value");
            assert_eq!(
                scalar_val.to_bits(),
                expected_up.to_bits(),
                "MulConstant scalar should be {expected_up}, got {scalar_val}"
            );
        }
        other => panic!("expected Layer::MulConstant, got {}", other.layer_type()),
    }
    assert_eq!(
        mul_node.inputs(),
        &["silu_native"],
        "mul node should chain from silu"
    );
}

#[test]
fn test_silu_mul_native_with_scalar() {
    let kernel = silu_mul_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(2.0)];
    let graph = try_native_layer(&kernel, &bindings)
        .unwrap()
        .expect("silu_mul with constant up should produce native layer");
    assert_silu_mul_graph_with_mul(&graph, 2.0);
}

#[test]
fn test_silu_mul_native_up_equals_one() {
    // When up ≈ 1.0, MulConstant is skipped — graph has only 1 SiLU node.
    let kernel = silu_mul_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let graph = try_native_layer(&kernel, &bindings)
        .unwrap()
        .expect("silu_mul with up=1.0 should produce native layer");
    assert_eq!(
        graph.num_nodes(),
        1,
        "up=1.0 optimization: should have only 1 node"
    );
    assert_eq!(graph.output_name(), "silu_native");
    let node = graph.node("silu_native").expect("silu node must exist");
    assert!(
        matches!(node.layer(), Layer::SiLU(_)),
        "single node should be SiLU, got {}",
        node.layer().layer_type()
    );
    assert!(
        graph.node("silu_mul_native").is_none(),
        "MulConstant node should be absent when up=1.0"
    );
}

#[test]
fn test_silu_mul_native_up_zero() {
    let kernel = silu_mul_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(0.0)];
    let graph = try_native_layer(&kernel, &bindings)
        .unwrap()
        .expect("silu_mul with up=0.0 should produce native layer");
    assert_silu_mul_graph_with_mul(&graph, 0.0);
}

#[test]
fn test_silu_mul_native_up_negative() {
    let kernel = silu_mul_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(-3.0)];
    let graph = try_native_layer(&kernel, &bindings)
        .unwrap()
        .expect("silu_mul with negative up should produce native layer");
    assert_silu_mul_graph_with_mul(&graph, -3.0);
}

#[test]
fn test_silu_mul_native_both_variable_falls_through() {
    let kernel = silu_mul_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "silu_mul with both variable params should fall through"
    );
}

#[test]
fn test_silu_mul_native_first_constant_falls_through() {
    let kernel = silu_mul_kernel();
    let bindings = [ParamBinding::Constant(1.0), ParamBinding::Constant(2.0)];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "silu_mul with first param constant should fall through"
    );
}

// ---------------------------------------------------------------------------
// Sigmoid native layer dispatch
// ---------------------------------------------------------------------------

/// Build a minimal sigmoid kernel for testing.
fn sigmoid_kernel() -> KernelDef {
    parse_kernel("fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }")
}

#[test]
fn test_sigmoid_native_dispatches() {
    let kernel = sigmoid_kernel();
    let bindings = [ParamBinding::Variable];
    let graph = try_native_layer(&kernel, &bindings)
        .unwrap()
        .expect("sigmoid with variable x should produce native layer");
    assert_eq!(
        graph.num_nodes(),
        1,
        "sigmoid graph should have exactly 1 node"
    );
    assert_eq!(graph.output_name(), "sigmoid_native");
    let node = graph
        .node("sigmoid_native")
        .expect("output node must exist");
    assert_eq!(
        node.layer().layer_type(),
        "Sigmoid",
        "expected Sigmoid layer"
    );
}

#[test]
fn test_sigmoid_native_bounds_correct() {
    // Uses propagate_single from lib.rs which calls kernel_to_graph →
    // try_native_layer → SigmoidLayer → propagate_ibp.
    let (lo, hi) = crate::test_helpers::propagate_single(
        "fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }",
        &[],
        -2.0,
        2.0,
    );

    // sigmoid is monotone: bounds should tightly wrap the endpoints.
    let expected_lo = 1.0 / (1.0 + 2.0_f32.exp()); // sigmoid(-2) ≈ 0.1192
    let expected_hi = 1.0 / (1.0 + (-2.0_f32).exp()); // sigmoid(2) ≈ 0.8808

    assert!(
        (lo - expected_lo).abs() < 0.01,
        "lower bound {lo} should be near {expected_lo}"
    );
    assert!(
        (hi - expected_hi).abs() < 0.01,
        "upper bound {hi} should be near {expected_hi}"
    );
}

#[test]
fn test_sigmoid_native_wrong_binding_count_falls_through() {
    let kernel = sigmoid_kernel();
    // Sigmoid expects 1 binding, passing 2 should fall through
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "sigmoid with 2 bindings should fall through"
    );
}

#[test]
fn test_sigmoid_native_constant_binding_falls_through() {
    let kernel = sigmoid_kernel();
    let bindings = [ParamBinding::Constant(0.5)];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "sigmoid with constant binding should fall through"
    );
}

// ---------------------------------------------------------------------------
// GELU native layer dispatch
// ---------------------------------------------------------------------------

fn gelu_tanh_kernel() -> KernelDef {
    // Body is irrelevant for native dispatch — only kernel.name is checked.
    parse_kernel("fn gelu(x: f32) -> f32 { x }")
}

fn gelu_erf_kernel() -> KernelDef {
    parse_kernel("fn gelu_erf(x: f32) -> f32 { x }")
}

#[test]
fn test_gelu_tanh_native_dispatches() {
    let kernel = gelu_tanh_kernel();
    let bindings = [ParamBinding::Variable];
    let graph = try_native_layer(&kernel, &bindings)
        .unwrap()
        .expect("gelu with variable x should produce native layer");
    assert_eq!(graph.num_nodes(), 1);
    assert_eq!(graph.output_name(), "gelu_native");
    let node = graph.node("gelu_native").expect("output node must exist");
    match node.layer() {
        Layer::GELU(g) => assert_eq!(g.approximation, GeluApproximation::Tanh),
        other => panic!("expected GELU layer, got {other:?}"),
    }
}

#[test]
fn test_gelu_erf_native_dispatches() {
    let kernel = gelu_erf_kernel();
    let bindings = [ParamBinding::Variable];
    let graph = try_native_layer(&kernel, &bindings)
        .unwrap()
        .expect("gelu_erf with variable x should produce native layer");
    assert_eq!(graph.num_nodes(), 1);
    assert_eq!(graph.output_name(), "gelu_erf_native");
    let node = graph
        .node("gelu_erf_native")
        .expect("output node must exist");
    match node.layer() {
        Layer::GELU(g) => assert_eq!(g.approximation, GeluApproximation::Erf),
        other => panic!("expected GELU layer, got {other:?}"),
    }
}

#[test]
fn test_gelu_erf_wrong_binding_count_falls_through() {
    let kernel = gelu_erf_kernel();
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "gelu_erf with 2 bindings should fall through"
    );
}

// ---------------------------------------------------------------------------
// Non-matching kernels
// ---------------------------------------------------------------------------

#[test]
fn test_non_matching_kernel_falls_through() {
    let kernel = unrelated_kernel();
    let bindings = [ParamBinding::Variable];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(result.is_none(), "unrelated kernel should return None");
}

#[test]
fn test_empty_bindings_falls_through() {
    let kernel = snake_kernel();
    let bindings: Vec<ParamBinding> = vec![];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(result.is_none(), "empty bindings should return None");
}

#[test]
fn test_confusable_snake_name_does_not_match() {
    // A kernel whose name contains "snake" as substring but is NOT "snake"
    // must NOT get the snake native layer (#561).
    let kernel = parse_kernel("fn not_a_snake(x: f32, alpha: f32) -> f32 { x + alpha }");
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "kernel named 'not_a_snake' must not match snake native path"
    );
}

#[test]
fn test_confusable_silu_mul_name_does_not_match() {
    // A kernel whose name contains "silu_mul" as substring but is NOT "silu_mul"
    // must NOT get the silu_mul native layer (#561).
    let kernel = parse_kernel("fn nn_silu_mul_variant(x: f32, up: f32) -> f32 { x * up }");
    let bindings = [ParamBinding::Variable, ParamBinding::Constant(2.0)];
    let result = try_native_layer(&kernel, &bindings).unwrap();
    assert!(
        result.is_none(),
        "kernel named 'nn_silu_mul_variant' must not match silu_mul native path"
    );
}
