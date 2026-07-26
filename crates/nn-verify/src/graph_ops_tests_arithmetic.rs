// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for arithmetic graph_ops translation: binop, unary, clamp, powi.
//!
//! Split from graph_ops_tests.rs (#478). Strategy: build KernelDef via Lowerer,
//! translate to NY GraphNetwork, propagate IBP bounds, verify output
//! bounds are mathematically correct.

use crate::graph::{kernel_to_graph, ParamBinding};
use crate::test_helpers::{parse_kernel, propagate_multi, propagate_single};

// ---------------------------------------------------------------------------
// BinOp translation tests (binop.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_binop_add_var_const() {
    // f(x) = x + 3.0, x in [1, 5] → output in [4, 8]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x + 3.0 }", &[], 1.0, 5.0);
    assert!((lo - 4.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 8.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_binop_sub_var_const() {
    // f(x) = x - 2.0, x in [3, 7] → output in [1, 5]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x - 2.0 }", &[], 3.0, 7.0);
    assert!((lo - 1.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 5.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_binop_mul_var_const() {
    // f(x) = x * 2.0, x in [1, 4] → output in [2, 8]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x * 2.0 }", &[], 1.0, 4.0);
    assert!((lo - 2.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 8.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_binop_mul_negative_const_swaps_bounds() {
    // f(x) = x * (-3.0), x in [1, 4] → output in [-12, -3]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x * (-3.0) }", &[], 1.0, 4.0);
    assert!((lo - (-12.0)).abs() < 1e-5, "lower: {lo}");
    assert!((hi - (-3.0)).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_binop_div_var_const() {
    // f(x) = x / 2.0, x in [4, 8] → output in [2, 4]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x / 2.0 }", &[], 4.0, 8.0);
    assert!((lo - 2.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 4.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_binop_div_by_zero_const_errors() {
    let kernel = parse_kernel("fn f(x: f32) -> f32 { x / 0.0 }");
    let result = kernel_to_graph(&kernel, &[]);
    assert!(result.is_err(), "division by constant zero should error");
}

#[test]
fn test_binop_const_div_var() {
    // f(x) = 6.0 / x, x in [2, 3] → output in [2, 3] (recip(x) * 6)
    // IBP: recip([2,3]) = [1/3, 1/2] (sound), * 6 = [2, 3]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { 6.0 / x }", &[], 2.0, 3.0);
    assert!((lo - 2.0).abs() < 1e-4, "lower: {lo}");
    assert!((hi - 3.0).abs() < 1e-4, "upper: {hi}");
}

#[test]
fn test_binop_const_sub_var() {
    // f(x) = 10.0 - x, x in [3, 7] → output in [3, 7]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { 10.0 - x }", &[], 3.0, 7.0);
    assert!((lo - 3.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 7.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_binop_add_two_variables() {
    // f(x, y) = x + y, x in [1,3], y in [2,5] → output in [3, 8]
    let (lo, hi) = propagate_multi(
        "fn f(x: f32, y: f32) -> f32 { x + y }",
        &[ParamBinding::Variable, ParamBinding::Variable],
        &[(1.0, 3.0), (2.0, 5.0)],
    );
    assert!((lo - 3.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 8.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_binop_sub_two_variables() {
    // f(x, y) = x - y, x in [5,10], y in [1,3] → output in [2, 9]
    let (lo, hi) = propagate_multi(
        "fn f(x: f32, y: f32) -> f32 { x - y }",
        &[ParamBinding::Variable, ParamBinding::Variable],
        &[(5.0, 10.0), (1.0, 3.0)],
    );
    assert!((lo - 2.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 9.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_binop_constant_folding() {
    // f(x) = x + (2.0 + 3.0) = x + 5.0, x in [0, 1] → [5, 6]
    // The 2.0 + 3.0 is folded at translation time via constant-constant binop.
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32, b: f32) -> f32 { x + (a + b) }",
        &[2.0, 3.0],
        0.0,
        1.0,
    );
    assert!((lo - 5.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 6.0).abs() < 1e-5, "upper: {hi}");
}

// ---------------------------------------------------------------------------
// UnaryFn translation tests (unary.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_unary_abs() {
    // f(x) = x.abs(), x in [-3, -1] → output in [1, 3]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.abs() }", &[], -3.0, -1.0);
    assert!((lo - 1.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 3.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_unary_abs_crosses_zero() {
    // f(x) = x.abs(), x in [-2, 3] → output in [0, 3]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.abs() }", &[], -2.0, 3.0);
    assert!((lo - 0.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 3.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_unary_exp() {
    // f(x) = x.exp(), x in [0, 1] → output in [1, e]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.exp() }", &[], 0.0, 1.0);
    assert!((lo - 1.0).abs() < 1e-4, "lower: {lo}");
    assert!((hi - std::f32::consts::E).abs() < 1e-4, "upper: {hi}");
}

#[test]
fn test_unary_sqrt() {
    // f(x) = x.sqrt(), x in [4, 9] → output in [2, 3]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.sqrt() }", &[], 4.0, 9.0);
    assert!((lo - 2.0).abs() < 1e-4, "lower: {lo}");
    assert!((hi - 3.0).abs() < 1e-4, "upper: {hi}");
}

#[test]
fn test_unary_recip() {
    // f(x) = x.recip(), x in [2, 5] → output in [0.2, 0.5]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.recip() }", &[], 2.0, 5.0);
    assert!((lo - 0.2).abs() < 1e-4, "lower: {lo}");
    assert!((hi - 0.5).abs() < 1e-4, "upper: {hi}");
}

#[test]
fn test_unary_sin() {
    // f(x) = x.sin(), x in [0, PI/2] → output in [0, 1]
    let half_pi = std::f32::consts::FRAC_PI_2;
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.sin() }", &[], 0.0, half_pi);
    assert!(lo >= -0.01, "lower should be >= ~0: {lo}");
    assert!((hi - 1.0).abs() < 0.1, "upper should be ~1: {hi}");
}

#[test]
fn test_unary_cos() {
    // f(x) = x.cos(), x in [0, PI/2] → output in [0, 1]
    let half_pi = std::f32::consts::FRAC_PI_2;
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.cos() }", &[], 0.0, half_pi);
    assert!(lo >= -0.01, "lower should be >= ~0: {lo}");
    assert!((hi - 1.0).abs() < 0.1, "upper should be ~1: {hi}");
}

#[test]
fn test_unary_rsqrt() {
    // f(x) = 1/sqrt(x), x in [4, 16] → output in [0.25, 0.5]
    // Rsqrt is translated as Sqrt then Reciprocal
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.rsqrt() }", &[], 4.0, 16.0);
    assert!((lo - 0.25).abs() < 1e-4, "lower: {lo}");
    assert!((hi - 0.5).abs() < 1e-4, "upper: {hi}");
}

#[test]
fn test_unary_constant_folding() {
    // f(x) = x + alpha.exp(), alpha=0.0 → x + 1.0, x in [0,1] → [1,2]
    let (lo, hi) = propagate_single(
        "fn f(x: f32, alpha: f32) -> f32 { x + alpha.exp() }",
        &[0.0],
        0.0,
        1.0,
    );
    assert!((lo - 1.0).abs() < 1e-4, "lower: {lo}");
    assert!((hi - 2.0).abs() < 1e-4, "upper: {hi}");
}

// ---------------------------------------------------------------------------
// Clamp translation tests (clamp.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_clamp_clips_range() {
    // f(x) = x.clamp(2.0, 8.0), x in [0, 10] → output in [2, 8]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.clamp(2.0, 8.0) }", &[], 0.0, 10.0);
    assert!((lo - 2.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 8.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_clamp_no_clip_needed() {
    // f(x) = x.clamp(0.0, 100.0), x in [5, 10] → output in [5, 10]
    let (lo, hi) = propagate_single(
        "fn f(x: f32) -> f32 { x.clamp(0.0, 100.0) }",
        &[],
        5.0,
        10.0,
    );
    assert!((lo - 5.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 10.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_clamp_constant_fold() {
    // f(x) = x + a.clamp(1.0, 5.0), a=3.0 → x + 3.0, x in [0,1] → [3,4]
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32) -> f32 { x + a.clamp(1.0, 5.0) }",
        &[3.0],
        0.0,
        1.0,
    );
    assert!((lo - 3.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 4.0).abs() < 1e-5, "upper: {hi}");
}

// ---------------------------------------------------------------------------
// Powi translation tests (from graph.rs translate_node)
// ---------------------------------------------------------------------------

#[test]
fn test_powi_squared() {
    // f(x) = x.powi(2), x in [2, 3] → output in [4, 9]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.powi(2) }", &[], 2.0, 3.0);
    assert!((lo - 4.0).abs() < 1e-4, "lower: {lo}");
    assert!((hi - 9.0).abs() < 1e-4, "upper: {hi}");
}

#[test]
fn test_powi_constant_fold() {
    // f(x) = x + a.powi(3), a=2.0 → x + 8.0, x in [0,1] → [8,9]
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32) -> f32 { x + a.powi(3) }",
        &[2.0],
        0.0,
        1.0,
    );
    assert!((lo - 8.0).abs() < 1e-4, "lower: {lo}");
    assert!((hi - 9.0).abs() < 1e-4, "upper: {hi}");
}

// ---------------------------------------------------------------------------
// BinaryFn translation tests (binop.rs — translate_binary_fn)
// ---------------------------------------------------------------------------

#[test]
fn test_binary_fn_atan2_first_quadrant() {
    // atan2(y, x) with y in [1, 2], x in [3, 4] → first quadrant, result in (0, π/2).
    // Corners: atan2(1,4)≈0.245, atan2(1,3)≈0.322, atan2(2,4)≈0.464, atan2(2,3)≈0.588
    // Expected bounds: [≈0.245, ≈0.588]
    use nn_dsl::ir::{BinaryFnKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};

    let kernel = KernelDef::new(
        "atan2_test",
        vec![
            Param::new("y", ScalarType::F32),
            Param::new("x", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinaryFn {
                    op: BinaryFnKind::Atan2,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );

    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let graph =
        crate::graph::kernel_to_graph_multi(&kernel, &bindings).expect("Atan2 graph translation");

    let input = crate::verify_input::multi_scalar_input_bounds(&[(1.0, 2.0), (3.0, 4.0)])
        .expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP through Atan2");

    let lo = output.lower().as_slice().expect("lower")[0];
    let hi = output.upper().as_slice().expect("upper")[0];

    let expected_lo = (1.0_f64).atan2(4.0_f64) as f32;
    let expected_hi = (2.0_f64).atan2(3.0_f64) as f32;
    assert!(lo <= expected_lo, "lower {lo} should be <= {expected_lo}");
    assert!(hi >= expected_hi, "upper {hi} should be >= {expected_hi}");
    assert!(lo > 0.0, "first quadrant: lower must be > 0");
    assert!(
        hi < std::f32::consts::FRAC_PI_2,
        "first quadrant: upper must be < π/2"
    );
}

#[test]
fn test_binary_fn_atan2_constant_fold() {
    // atan2(1.0, 1.0) = π/4 when both inputs are constant.
    use nn_dsl::ir::{BinaryFnKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};

    let kernel = KernelDef::new(
        "atan2_const",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.0)),
            IRNode::new(NodeId::new(2), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::BinaryFn {
                    op: BinaryFnKind::Atan2,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(2),
                },
            ),
            // x + atan2(1, 1) = x + π/4
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::BinOp {
                    op: nn_dsl::ir::BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(3),
                },
            ),
        ],
        NodeId::new(4),
    );

    let bindings = vec![ParamBinding::Variable];
    let graph =
        crate::graph::kernel_to_graph_multi(&kernel, &bindings).expect("constant-fold atan2 graph");

    let input = crate::verify_input::scalar_input_bounds(0.0, 1.0).expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");

    let lo = output.lower().as_slice().expect("lower")[0];
    let hi = output.upper().as_slice().expect("upper")[0];

    let pi_4 = std::f32::consts::FRAC_PI_4;
    assert!((lo - pi_4).abs() < 1e-5, "lower should be ≈π/4, got {lo}");
    assert!(
        (hi - (1.0 + pi_4)).abs() < 1e-5,
        "upper should be ≈1+π/4, got {hi}"
    );
}
