// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Softplus and Exp NY translation.
//!
//! Verifies that Softplus and Exp ops translate to NY
//! `SoftplusLayer` and `ExpLayer` with correct IBP and CROWN propagation.
//!
//! Part of #834 — Gated DeltaNet gate computation pathway.

use super::common;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{propagate_with_crown_fallback, tensor_kernel_to_graph, TensorParamBinding};

/// Helper: build a 1-input graph with a softplus activation op.
fn build_softplus_kernel() -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("softplus_test");
    let x = b.add_input("x", &[4]);
    let sp = b.add_softplus(x, &[4]);
    b.build(sp).expect("valid graph")
}

/// Helper: build a 1-input graph with an exp activation op.
fn build_exp_kernel() -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("exp_test");
    let x = b.add_input("x", &[4]);
    let e = b.add_exp(x, &[4]);
    b.build(e).expect("valid graph")
}

/// Helper: build a 2-op chain: softplus → exp (gate computation pattern).
fn build_softplus_then_exp_kernel() -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("softplus_exp_chain");
    let x = b.add_input("x", &[4]);
    let sp = b.add_softplus(x, &[4]);
    let e = b.add_exp(sp, &[4]);
    b.build(e).expect("valid graph")
}

#[test]
fn test_softplus_graph_builds_and_validates() {
    let def = build_softplus_kernel();
    assert!(def.validate().is_ok());
    assert_eq!(def.nodes.len(), 2); // 1 input + 1 softplus
}

#[test]
fn test_exp_graph_builds_and_validates() {
    let def = build_exp_kernel();
    assert!(def.validate().is_ok());
    assert_eq!(def.nodes.len(), 2); // 1 input + 1 exp
}

#[test]
fn test_softplus_gamma_crown_graph_builds() {
    let def = build_softplus_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings);
    assert!(
        graph.is_ok(),
        "softplus graph build failed: {:?}",
        graph.err()
    );
}

#[test]
fn test_exp_gamma_crown_graph_builds() {
    let def = build_exp_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings);
    assert!(graph.is_ok(), "exp graph build failed: {:?}", graph.err());
}

#[test]
fn test_softplus_ibp_propagates() {
    let def = build_softplus_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");

    let input = common::uniform_bounds(&[1, 4], 2.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    common::assert_bounds_valid(&output);
    // softplus is monotonically increasing and always positive:
    // softplus(-2) ≈ 0.1269, softplus(2) ≈ 2.1269
    // IBP uses monotonic interval extension for softplus, so lo >= 0 is
    // structurally guaranteed. Add meaningful upper bound check instead.
    let (lo, hi) = output.lower_upper();
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(
            l.is_finite() && *l >= 0.0,
            "softplus lower must be non-negative, got {l}"
        );
        assert!(
            *u < 10.0,
            "softplus upper with input [-2,2] should be bounded, got {u}"
        );
    }
}

#[test]
fn test_exp_ibp_propagates() {
    let def = build_exp_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");

    let input = common::uniform_bounds(&[1, 4], 2.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    common::assert_bounds_valid(&output);
    // exp is monotonically increasing and always positive:
    // exp(-2) ≈ 0.1353, exp(2) ≈ 7.389
    // IBP uses monotonic interval extension for exp, so lo > 0 is
    // structurally guaranteed. Add meaningful upper bound check.
    let (lo, hi) = output.lower_upper();
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(
            l.is_finite() && *l > 0.0,
            "exp lower must be positive, got {l}"
        );
        assert!(
            *u < 10.0,
            "exp upper with input [-2,2] should be bounded, got {u}"
        );
    }
}

#[test]
fn test_softplus_crown_propagates() {
    let def = build_softplus_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");

    let input = common::uniform_bounds(&[1, 4], 1.0);
    let (_method, output, _fallback) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN");
    common::assert_bounds_valid(&output);
    let (lo, _hi) = output.lower_upper();
    for l in lo.iter() {
        assert!(*l > 0.0, "softplus is always positive");
    }
}

#[test]
fn test_exp_crown_propagates() {
    let def = build_exp_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");

    let input = common::uniform_bounds(&[1, 4], 1.0);
    let (_method, output, _fallback) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN");
    common::assert_bounds_valid(&output);
    let (lo, _hi) = output.lower_upper();
    for l in lo.iter() {
        assert!(*l > 0.0, "exp is always positive");
    }
}

#[test]
fn test_softplus_exp_chain_ibp() {
    let def = build_softplus_then_exp_kernel();
    assert!(def.validate().is_ok());
    assert_eq!(def.nodes.len(), 3); // 1 input + 1 softplus + 1 exp

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");

    let input = common::uniform_bounds(&[1, 4], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    common::assert_bounds_valid(&output);
    // exp(softplus(x)) >= exp(0) = 1 for all finite x (softplus(x) > 0)
    let (lo, _hi) = output.lower_upper();
    for l in lo.iter() {
        assert!(*l >= 1.0 - 1e-4, "exp(softplus(x)) >= 1, got {l}");
    }
}

#[test]
fn test_softplus_exp_chain_crown() {
    let def = build_softplus_then_exp_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph build");

    let input = common::uniform_bounds(&[1, 4], 1.0);
    let (_method, output, _fallback) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN");
    common::assert_bounds_valid(&output);
    // exp(softplus(x)) >= exp(0) = 1 for all finite x (softplus(x) > 0)
    let (lo, _hi) = output.lower_upper();
    for l in lo.iter() {
        assert!(*l >= 1.0 - 1e-4, "exp(softplus(x)) >= 1");
    }
}

#[test]
fn test_softplus_crown_at_least_as_tight_as_ibp() {
    let def = build_softplus_kernel();
    let bindings = vec![TensorParamBinding::Variable];

    // IBP
    let graph_ibp = tensor_kernel_to_graph(&def, &bindings).expect("graph build");
    let input = common::uniform_bounds(&[1, 4], 2.0);
    let ibp_output = graph_ibp.propagate_ibp(&input).expect("IBP");

    // CROWN
    let graph_crown = tensor_kernel_to_graph(&def, &bindings).expect("graph build");
    let input2 = common::uniform_bounds(&[1, 4], 2.0);
    let (_method, crown_output, _fallback) =
        propagate_with_crown_fallback(&graph_crown, &input2).expect("CROWN");

    // CROWN bounds should be at least as tight as IBP
    common::assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
}
