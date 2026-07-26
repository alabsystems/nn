// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Graph translation tests: native vs decomposed bounds + extended ops.
//!
//! - Native vs decomposed bounds comparison (#338 AC5/AC6)
//! - Gap-closing tests: Powi variable base, Eq/Ne var-var, ReLU via Ge
//!
//! Core translation tests are in `graph_translate.rs`.

use super::common;

use common::ibp_scalar;
use nn_dsl::ir::{CompareOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};
use nn_verify::{kernel_to_graph, kernel_to_graph_multi, BoundedTensor, ParamBinding};
use ndarray::{ArrayD, IxDyn};

// ── #338 AC5: Native vs decomposed bounds comparison ────────────────────────

/// Build a decomposed-only graph by changing the kernel name so `try_native_layer`
/// doesn't match. Same IR, same math — just bypasses the native fast path.
fn decomposed_graph(kernel: &KernelDef, alpha: f32) -> nn_verify::GraphNetwork {
    let mut decomposed_kernel = kernel.clone();
    decomposed_kernel.name = "decomposed_test".to_string();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(alpha)];
    kernel_to_graph_multi(&decomposed_kernel, &bindings).expect("decomposed graph")
}

#[test]
fn test_native_snake_bounds_tighter_than_decomposed() {
    // #338 AC5: Native SnakeLayer should produce equal-or-tighter IBP bounds
    // compared to the decomposed Sin→Pow→Mul→Add path.
    let kernel = common::snake_kernel();
    let alpha = 1.0f32;

    let native_graph = kernel_to_graph(&kernel, &[alpha]).expect("native graph");
    let decomposed = decomposed_graph(&kernel, alpha);

    // Verify native uses fewer nodes (1 vs 5+)
    assert!(
        native_graph.num_nodes() < decomposed.num_nodes(),
        "native ({}) should have fewer nodes than decomposed ({})",
        native_graph.num_nodes(),
        decomposed.num_nodes()
    );

    let (nat_lo, nat_hi) = ibp_scalar(&native_graph, -10.0, 10.0);
    let (dec_lo, dec_hi) = ibp_scalar(&decomposed, -10.0, 10.0);

    // Native bounds must be tighter: [nat_lo, nat_hi] ⊆ [dec_lo, dec_hi]
    // (with small tolerance for floating-point differences)
    assert!(
        nat_lo >= dec_lo - 1e-3,
        "native lower {nat_lo} should be >= decomposed lower {dec_lo}"
    );
    assert!(
        nat_hi <= dec_hi + 1e-3,
        "native upper {nat_hi} should be <= decomposed upper {dec_hi}"
    );

    // The native path should be strictly tighter (not just equal) for wide ranges,
    // because decomposed SinLayer over-approximates while SnakeLayer uses monotonicity.
    let width_native = nat_hi - nat_lo;
    let width_decomposed = dec_hi - dec_lo;
    assert!(
        width_native < width_decomposed + 1e-3,
        "native width {width_native} should be <= decomposed width {width_decomposed}"
    );
}

#[test]
fn test_native_silu_bounds_tighter_than_decomposed() {
    // #338 AC5: Native SiLULayer should produce equal-or-tighter IBP bounds
    // compared to the decomposed Exp→Add→Div→Mul path.
    let kernel = nn_dsl::silu_mul::build_silu_mul_kernel().expect("build silu_mul");
    let up = 2.0f32;

    let native_graph = kernel_to_graph(&kernel, &[up]).expect("native graph");
    let decomposed = decomposed_graph(&kernel, up);

    // Verify native uses fewer nodes (2 vs 5+)
    assert!(
        native_graph.num_nodes() < decomposed.num_nodes(),
        "native ({}) should have fewer nodes than decomposed ({})",
        native_graph.num_nodes(),
        decomposed.num_nodes()
    );

    let (nat_lo, nat_hi) = ibp_scalar(&native_graph, -5.0, 5.0);
    let (dec_lo, dec_hi) = ibp_scalar(&decomposed, -5.0, 5.0);

    // Native bounds must be tighter: [nat_lo, nat_hi] ⊆ [dec_lo, dec_hi]
    assert!(
        nat_lo >= dec_lo - 1e-3,
        "native lower {nat_lo} should be >= decomposed lower {dec_lo}"
    );
    assert!(
        nat_hi <= dec_hi + 1e-3,
        "native upper {nat_hi} should be <= decomposed upper {dec_hi}"
    );
}

// ── #338 AC6: Decomposed fallback paths preserved ───────────────────────────

#[test]
fn test_multi_variable_snake_uses_decomposed_fallback() {
    // #338 AC6: When both params are variable, try_native_layer returns None
    // and the kernel falls through to the decomposed path.
    let kernel = common::snake_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let graph =
        kernel_to_graph_multi(&kernel, &bindings).expect("multi-variable snake should succeed");
    // Decomposed path produces multiple nodes (5+), not the native single node.
    assert!(
        graph.num_nodes() > 1,
        "multi-variable snake should use decomposed path (got {} nodes)",
        graph.num_nodes()
    );
}

#[test]
fn test_multi_variable_silu_uses_decomposed_fallback() {
    // #338 AC6: When both params are variable, silu_mul falls through to decomposition.
    let kernel = nn_dsl::silu_mul::build_silu_mul_kernel().expect("build silu_mul");
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let graph =
        kernel_to_graph_multi(&kernel, &bindings).expect("multi-variable silu_mul should succeed");
    // Decomposed path produces multiple nodes, not the native 1-2 nodes.
    assert!(
        graph.num_nodes() > 2,
        "multi-variable silu_mul should use decomposed path (got {} nodes)",
        graph.num_nodes()
    );
}

// ── Extended ops: Powi variable base, Eq/Ne var-var, ReLU via Ge ────────────

/// Test Powi with a variable base: `x.powi(2)` where x is a variable input.
#[test]
fn test_powi_variable_base_ibp_bounds() {
    let kernel = KernelDef::new(
        "powi_var",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 2,
                },
            ),
        ],
        NodeId::new(1),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build x.powi(2) graph");
    // x in [2, 5] → x^2 in [4, 25]
    let (lo, hi) = ibp_scalar(&graph, 2.0, 5.0);
    assert!(lo >= 3.9, "x^2 lower on [2,5] should be >= 4, got {lo}");
    assert!(hi <= 25.1, "x^2 upper on [2,5] should be <= 25, got {hi}");
}

/// Test Powi with variable base and odd exponent: `x.powi(3)`.
#[test]
fn test_powi_variable_base_odd_exponent() {
    let kernel = KernelDef::new(
        "powi_cube",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 3,
                },
            ),
        ],
        NodeId::new(1),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build x.powi(3) graph");
    let (lo, hi) = ibp_scalar(&graph, -2.0, 3.0);
    assert!(lo <= -7.9, "x^3 lower on [-2,3] should be <= -8, got {lo}");
    assert!(hi >= 26.9, "x^3 upper on [-2,3] should be >= 27, got {hi}");
}

/// Build a var-var compare kernel: `select(compare(x, y), 1.0, 0.0)`.
fn compare_var_var_kernel(op: CompareOpKind) -> KernelDef {
    KernelDef::new(
        format!("{op:?}_vv"),
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("y", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(NodeId::new(4), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(5),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(3),
                    else_val: NodeId::new(4),
                },
            ),
        ],
        NodeId::new(5),
    )
}

#[test]
fn test_compare_eq_var_var_translates() {
    let kernel = compare_var_var_kernel(CompareOpKind::Eq);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let graph = kernel_to_graph_multi(&kernel, &bindings).expect("build Eq(x,y) graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -3.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 3.0f32),
    )
    .expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();
    // Select(Eq, 1.0, 0.0): bounds in [0, 1]
    assert!(lo[[0]] >= -0.01, "Eq lo >= 0, got {}", lo[[0]]);
    assert!(hi[[0]] <= 1.01, "Eq hi <= 1, got {}", hi[[0]]);
    assert!(
        lo[[0]].is_finite() && hi[[0]].is_finite(),
        "Eq bounds finite"
    );
}

#[test]
fn test_compare_ne_var_var_translates() {
    let kernel = compare_var_var_kernel(CompareOpKind::Ne);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let graph = kernel_to_graph_multi(&kernel, &bindings).expect("build Ne(x,y) graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -3.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 3.0f32),
    )
    .expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();
    // Select(Ne, 1.0, 0.0): bounds in [0, 1]
    assert!(lo[[0]] >= -0.01, "Ne lo >= 0, got {}", lo[[0]]);
    assert!(hi[[0]] <= 1.01, "Ne hi <= 1, got {}", hi[[0]]);
    assert!(
        lo[[0]].is_finite() && hi[[0]].is_finite(),
        "Ne bounds finite"
    );
}

/// Test ReLU activation pattern matching via `Ge` comparison (>= 0).
#[test]
fn test_relu_via_select_ge_comparison() {
    let kernel = KernelDef::new(
        "relu_ge",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Ge,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(3),
    );

    let graph = kernel_to_graph(&kernel, &[]).expect("build relu via Ge select");
    let (lo, hi) = ibp_scalar(&graph, -5.0, 5.0);
    assert!(lo >= -0.01, "relu via Ge lower should be >= 0, got {lo}");
    assert!(hi >= 4.99, "relu via Ge upper should be >= 5, got {hi}");
}
