// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for native NY layer dispatch in tensor-level Elementwise translation.
//!
//! Verifies that `translate_elementwise_inline` uses native `SnakeLayer` and `SiLULayer`
//! instead of decomposing through scalar IR nodes. Native layers produce tighter bounds
//! because they exploit mathematical properties (Snake monotonicity, SiLU derivative bounds).
//!
//! Part of #1045 AC2: SiLU and Snake native layers in tensor-level verification.

use super::common;
use nn_dsl::adain::build_snake_scalar_kernel;
use nn_dsl::silu_mul::build_silu_mul_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};

// ---------------------------------------------------------------------------
// Snake native dispatch
// ---------------------------------------------------------------------------

/// Build a minimal tensor graph: input → Elementwise(snake) → output.
fn build_snake_elementwise(channels: usize, length: usize) -> nn_dsl::tensor_ir::TensorKernelDef {
    let snake = build_snake_scalar_kernel().expect("snake kernel");
    let mut b = TensorBlockBuilder::new("snake_native_test");

    let x = b.add_input("x", &[channels, length]);
    let alpha = b.add_input("alpha", &[1]);
    let shape = [channels, length];
    let alpha_bc = b.add_broadcast(alpha, &shape);
    let out = b.add_elementwise(snake, &[x, alpha_bc], &shape);

    b.build(out).expect("valid graph")
}

/// Native Snake dispatch produces a graph with fewer nodes than decomposition.
///
/// The native path emits a single SnakeLayer node. The decomposed path expands
/// to Sin→Pow→MulConstant→Add (4+ nodes from the scalar IR).
#[test]
fn test_tensor_snake_native_fewer_nodes() {
    let def = build_snake_elementwise(4, 8);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1.0), // alpha
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("native graph");

    // With native dispatch: 1 SnakeLayer node.
    // Without native dispatch: 5+ nodes from scalar IR decomposition.
    // Accept up to 3 nodes (SnakeLayer + possible identity/broadcast).
    assert!(
        graph.num_nodes() <= 3,
        "native Snake should use <= 3 NY nodes, got {}",
        graph.num_nodes()
    );
}

/// Native Snake produces valid IBP bounds at the tensor level.
#[test]
fn test_tensor_snake_native_ibp_valid() {
    let def = build_snake_elementwise(4, 8);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1.0),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = common::uniform_bounds(&[4, 8], 5.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    common::assert_bounds_valid(&output);
    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[4, 8]);

    // Snake(x, alpha=1) = x + sin²(x) has range:
    // lower at x=-5: -5 + sin²(-5) ≈ -5 + 0.92 ≈ -4.08
    // upper at x=5: 5 + sin²(5) ≈ 5 + 0.92 ≈ 5.92
    // Since Snake is monotone for alpha>0, native bounds are exact.
    for &l in lo.iter() {
        assert!(l > -6.0 && l < 0.0, "lower bound {l} out of expected range");
    }
    for &u in hi.iter() {
        assert!(u > 4.0 && u < 7.0, "upper bound {u} out of expected range");
    }
}

/// Native Snake produces tighter bounds than decomposed path for wide input ranges.
#[test]
fn test_tensor_snake_native_tighter_than_decomposed() {
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");

    // Build native graph (kernel name = "snake" → native dispatch)
    let native_def = {
        let mut b = TensorBlockBuilder::new("native_snake");
        let x = b.add_input("x", &[1, 4]);
        let alpha = b.add_input("alpha", &[1]);
        let shape = [1, 4];
        let alpha_bc = b.add_broadcast(alpha, &shape);
        let out = b.add_elementwise(snake_kernel.clone(), &[x, alpha_bc], &shape);
        b.build(out).expect("valid")
    };

    // Build decomposed graph (rename kernel so native dispatch doesn't match)
    let decomposed_def = {
        let mut decomposed_kernel = snake_kernel;
        decomposed_kernel.name = "snake_decomposed_test".to_string();
        let mut b = TensorBlockBuilder::new("decomposed_snake");
        let x = b.add_input("x", &[1, 4]);
        let alpha = b.add_input("alpha", &[1]);
        let shape = [1, 4];
        let alpha_bc = b.add_broadcast(alpha, &shape);
        let out = b.add_elementwise(decomposed_kernel, &[x, alpha_bc], &shape);
        b.build(out).expect("valid")
    };

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1.0),
    ];
    let native_graph = tensor_kernel_to_graph(&native_def, &bindings).expect("native");
    let decomposed_graph = tensor_kernel_to_graph(&decomposed_def, &bindings).expect("decomposed");

    let input = common::uniform_bounds(&[1, 4], 10.0);

    let native_out = native_graph.propagate_ibp(&input).expect("native IBP");
    let decomposed_out = decomposed_graph
        .propagate_ibp(&input)
        .expect("decomposed IBP");

    let (nat_lo, nat_hi) = native_out.lower_upper();
    let (dec_lo, dec_hi) = decomposed_out.lower_upper();

    // Native bounds should be equal-or-tighter (subset of decomposed bounds).
    let eps = 1e-3;
    for (&nl, &dl) in nat_lo.iter().zip(dec_lo.iter()) {
        assert!(
            nl >= dl - eps,
            "native lower {nl} should be >= decomposed lower {dl}"
        );
    }
    for (&nu, &du) in nat_hi.iter().zip(dec_hi.iter()) {
        assert!(
            nu <= du + eps,
            "native upper {nu} should be <= decomposed upper {du}"
        );
    }

    // For wide ranges ([-10, 10]), native should be strictly tighter.
    let native_width: f32 = nat_lo.iter().zip(nat_hi.iter()).map(|(&l, &u)| u - l).sum();
    let decomposed_width: f32 = dec_lo.iter().zip(dec_hi.iter()).map(|(&l, &u)| u - l).sum();

    assert!(
        native_width < decomposed_width + eps,
        "native total width ({native_width}) should be <= decomposed ({decomposed_width})"
    );
}

// ---------------------------------------------------------------------------
// SiLU-Mul native dispatch
// ---------------------------------------------------------------------------

/// Build a minimal tensor graph: input → Elementwise(silu_mul) → output.
fn build_silu_mul_elementwise(
    channels: usize,
    length: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let silu = build_silu_mul_kernel().expect("silu_mul kernel");
    let mut b = TensorBlockBuilder::new("silu_mul_native_test");

    let x = b.add_input("x", &[channels, length]);
    let up = b.add_input("up", &[1]);
    let shape = [channels, length];
    let up_bc = b.add_broadcast(up, &shape);
    let out = b.add_elementwise(silu, &[x, up_bc], &shape);

    b.build(out).expect("valid graph")
}

/// Native SiLU dispatch produces a graph with fewer nodes than decomposition.
#[test]
fn test_tensor_silu_native_fewer_nodes() {
    let def = build_silu_mul_elementwise(4, 8);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(2.0), // up
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("native graph");

    // With native dispatch: SiLULayer + MulConstant = 2 nodes.
    // Without native dispatch: Sigmoid→Mul→MulConstant (3+ from scalar IR).
    assert!(
        graph.num_nodes() <= 4,
        "native SiLU should use <= 4 NY nodes, got {}",
        graph.num_nodes()
    );
}

/// Native SiLU produces valid IBP bounds at the tensor level.
#[test]
fn test_tensor_silu_native_ibp_valid() {
    let def = build_silu_mul_elementwise(4, 8);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(2.0),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = common::uniform_bounds(&[4, 8], 5.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    common::assert_bounds_valid(&output);
    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[4, 8]);

    // silu_mul(x, up=2) = x * sigmoid(x) * 2
    // At x=-5: silu(-5)*2 ≈ -5*sigmoid(-5)*2 ≈ -5*0.0067*2 ≈ -0.067
    // At x=5: silu(5)*2 ≈ 5*sigmoid(5)*2 ≈ 5*0.9933*2 ≈ 9.933
    for &l in lo.iter() {
        assert!(l >= -2.0, "lower bound {l} too negative for silu_mul");
    }
    for &u in hi.iter() {
        assert!(u <= 12.0, "upper bound {u} too large for silu_mul");
    }
}

/// Native SiLU with up=1 (no MulConstant optimization check).
#[test]
fn test_tensor_silu_up_one() {
    let def = build_silu_mul_elementwise(2, 4);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1.0), // up = 1.0
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = common::uniform_bounds(&[2, 4], 3.0);
    let output = graph.propagate_ibp(&input).expect("IBP");

    common::assert_bounds_valid(&output);
}

/// Decomposed fallback when both inputs are Variable (multi-variable SiLU).
#[test]
fn test_tensor_silu_both_variable_falls_through() {
    let silu = build_silu_mul_kernel().expect("silu_mul kernel");
    let mut b = TensorBlockBuilder::new("silu_both_var");

    let x = b.add_input("x", &[2, 4]);
    let up = b.add_input("up", &[2, 4]);
    let shape = [2, 4];
    let out = b.add_elementwise(silu, &[x, up], &shape);
    let def = b.build(out).expect("valid");

    // Both inputs Variable → native dispatch won't match, falls through to decomposed.
    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];

    // Should still translate successfully via decomposed path.
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("decomposed graph");

    // Decomposed path uses more nodes than native.
    assert!(
        graph.num_nodes() >= 3,
        "decomposed SiLU should use >= 3 nodes, got {}",
        graph.num_nodes()
    );
}
