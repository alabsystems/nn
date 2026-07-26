// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for piecewise graph_ops translation: compare, minmax, select.
//!
//! Split from graph_ops_tests.rs (#478). Strategy: build KernelDef via Lowerer,
//! translate to NY GraphNetwork, propagate IBP bounds, verify output
//! bounds are mathematically correct.

use crate::graph::{kernel_to_graph, ParamBinding};
use crate::test_helpers::{parse_kernel, propagate_multi, propagate_single};
use crate::verify_input::scalar_input_bounds;

// ---------------------------------------------------------------------------
// Compare translation tests (compare.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_compare_gt_constant_fold_true() {
    // if 5.0 > 3.0 { x + 1.0 } else { x * 2.0 } → x + 1.0, x in [1,2] → [2,3]
    // Note: the then-branch must not be bare `x` (identity), because when constant
    // condition folds to select the Variable input directly, the graph output
    // references NETWORK_INPUT which NY IBP cannot resolve. See #477.
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32, b: f32) -> f32 { if a > b { x + 1.0 } else { x * 2.0 } }",
        &[5.0, 3.0],
        1.0,
        2.0,
    );
    assert!((lo - 2.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 3.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_compare_gt_constant_fold_false() {
    // if 1.0 > 3.0 { x } else { x * 2.0 } → x*2, x in [1,2] → [2,4]
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32, b: f32) -> f32 { if a > b { x } else { x * 2.0 } }",
        &[1.0, 3.0],
        1.0,
        2.0,
    );
    assert!((lo - 2.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 4.0).abs() < 1e-5, "upper: {hi}");
}

// ---------------------------------------------------------------------------
// MinMax translation tests (minmax.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_max_var_zero_is_relu() {
    // f(x) = x.max(0.0) = ReLU(x), x in [-2, 3] → [0, 3]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.max(0.0) }", &[], -2.0, 3.0);
    assert!((lo - 0.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 3.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_min_var_zero() {
    // f(x) = x.min(0.0) = -ReLU(-x), x in [-2, 3] → [-2, 0]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.min(0.0) }", &[], -2.0, 3.0);
    assert!((lo - (-2.0)).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 0.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_max_var_nonzero_const() {
    // f(x) = x.max(2.0), x in [0, 5] → [2, 5]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.max(2.0) }", &[], 0.0, 5.0);
    assert!((lo - 2.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 5.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_min_var_nonzero_const() {
    // f(x) = x.min(3.0), x in [0, 5] → [0, 3]
    let (lo, hi) = propagate_single("fn f(x: f32) -> f32 { x.min(3.0) }", &[], 0.0, 5.0);
    assert!((lo - 0.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 3.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_max_constant_fold() {
    // f(x) = x + a.max(b), a=3.0, b=5.0 → x + 5.0, x in [0,1] → [5,6]
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32, b: f32) -> f32 { x + a.max(b) }",
        &[3.0, 5.0],
        0.0,
        1.0,
    );
    assert!((lo - 5.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 6.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_min_constant_fold() {
    // f(x) = x + a.min(b), a=3.0, b=5.0 → x + 3.0, x in [0,1] → [3,4]
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32, b: f32) -> f32 { x + a.min(b) }",
        &[3.0, 5.0],
        0.0,
        1.0,
    );
    assert!((lo - 3.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 4.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_max_two_variables() {
    // f(x, y) = x.max(y), x in [1,3], y in [2,5] → [2, 5]
    // IBP: max([1,3], [2,5]) → [max(1,2), max(3,5)] = [2, 5]
    let (lo, hi) = propagate_multi(
        "fn f(x: f32, y: f32) -> f32 { x.max(y) }",
        &[ParamBinding::Variable, ParamBinding::Variable],
        &[(1.0, 3.0), (2.0, 5.0)],
    );
    assert!((1.0..=2.5).contains(&lo), "lower: {lo}");
    assert!((4.5..=5.5).contains(&hi), "upper: {hi}");
}

// ---------------------------------------------------------------------------
// Select / pattern-matching tests (select.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_select_relu_pattern() {
    // if x > 0.0 { x } else { 0.0 } → ReLU(x)
    // x in [-2, 5] → [0, 5]
    let (lo, hi) = propagate_single(
        "fn f(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }",
        &[],
        -2.0,
        5.0,
    );
    assert!((lo - 0.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 5.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_select_leaky_relu_pattern() {
    // if x > 0.0 { x } else { 0.01 * x } → LeakyReLU(0.01)
    // x in [-10, 10] → [-0.1, 10]
    let (lo, hi) = propagate_single(
        "fn f(x: f32) -> f32 { if x > 0.0 { x } else { 0.01 * x } }",
        &[],
        -10.0,
        10.0,
    );
    assert!((lo - (-0.1)).abs() < 1e-4, "lower: {lo}");
    assert!((hi - 10.0).abs() < 1e-4, "upper: {hi}");
}

#[test]
fn test_select_constant_condition_true() {
    // if a > 0 { x + 1 } else { x * 2 } with a=1.0 (constant true) → x + 1
    // Note: then-branch must not be bare `x` (see #477 — identity output
    // references NETWORK_INPUT which NY cannot resolve as output).
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32) -> f32 { if a > 0.0 { x + 1.0 } else { x * 2.0 } }",
        &[1.0],
        3.0,
        7.0,
    );
    assert!((lo - 4.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 8.0).abs() < 1e-5, "upper: {hi}");
}

#[test]
fn test_select_constant_condition_false() {
    // if a > 0 { x } else { x * 2 } with a=-1.0 (constant false) → x*2
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32) -> f32 { if a > 0.0 { x } else { x * 2.0 } }",
        &[-1.0],
        3.0,
        7.0,
    );
    assert!((lo - 6.0).abs() < 1e-5, "lower: {lo}");
    assert!((hi - 14.0).abs() < 1e-5, "upper: {hi}");
}

// ---------------------------------------------------------------------------
// Compare equality boundary tests (#559): Ge/Le select then-branch, Gt/Lt
// select else-branch when operands are exactly equal.
// ---------------------------------------------------------------------------

#[test]
fn test_select_ge_equality_boundary_selects_then() {
    // AC3: if 5.0 >= 5.0 { x + 1.0 } else { x * 2.0 } → x + 1.0 (then-branch)
    // Ge at equality boundary must select the then-branch.
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32, b: f32) -> f32 { if a >= b { x + 1.0 } else { x * 2.0 } }",
        &[5.0, 5.0],
        1.0,
        2.0,
    );
    // then-branch: x + 1.0 with x in [1,2] → [2, 3]
    assert!(
        (lo - 2.0).abs() < 1e-5,
        "Ge(5,5) should select then-branch, lower: {lo}"
    );
    assert!(
        (hi - 3.0).abs() < 1e-5,
        "Ge(5,5) should select then-branch, upper: {hi}"
    );
}

#[test]
fn test_select_gt_equality_boundary_selects_else() {
    // AC4: if 5.0 > 5.0 { x + 1.0 } else { x * 2.0 } → x * 2.0 (else-branch)
    // Gt at equality boundary must select the else-branch.
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32, b: f32) -> f32 { if a > b { x + 1.0 } else { x * 2.0 } }",
        &[5.0, 5.0],
        1.0,
        2.0,
    );
    // else-branch: x * 2.0 with x in [1,2] → [2, 4]
    assert!(
        (lo - 2.0).abs() < 1e-5,
        "Gt(5,5) should select else-branch, lower: {lo}"
    );
    assert!(
        (hi - 4.0).abs() < 1e-5,
        "Gt(5,5) should select else-branch, upper: {hi}"
    );
}

#[test]
fn test_select_le_equality_boundary_selects_then() {
    // AC1 (Le variant): if 5.0 <= 5.0 { x + 1.0 } else { x * 2.0 } → x + 1.0
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32, b: f32) -> f32 { if a <= b { x + 1.0 } else { x * 2.0 } }",
        &[5.0, 5.0],
        1.0,
        2.0,
    );
    // then-branch: x + 1.0 with x in [1,2] → [2, 3]
    assert!(
        (lo - 2.0).abs() < 1e-5,
        "Le(5,5) should select then-branch, lower: {lo}"
    );
    assert!(
        (hi - 3.0).abs() < 1e-5,
        "Le(5,5) should select then-branch, upper: {hi}"
    );
}

#[test]
fn test_select_lt_equality_boundary_selects_else() {
    // AC2 (Lt variant): if 5.0 < 5.0 { x + 1.0 } else { x * 2.0 } → x * 2.0
    let (lo, hi) = propagate_single(
        "fn f(x: f32, a: f32, b: f32) -> f32 { if a < b { x + 1.0 } else { x * 2.0 } }",
        &[5.0, 5.0],
        1.0,
        2.0,
    );
    // else-branch: x * 2.0 with x in [1,2] → [2, 4]
    assert!(
        (lo - 2.0).abs() < 1e-5,
        "Lt(5,5) should select else-branch, lower: {lo}"
    );
    assert!(
        (hi - 4.0).abs() < 1e-5,
        "Lt(5,5) should select else-branch, upper: {hi}"
    );
}

// ---------------------------------------------------------------------------
// Identity-output regression: constant Select folds to bare Variable (#477)
// ---------------------------------------------------------------------------

#[test]
fn test_constant_select_identity_output_propagates() {
    // When constant condition folds and selects the bare input variable,
    // an identity layer is inserted so IBP propagation succeeds. (#477)
    let kernel = parse_kernel("fn f(x: f32, a: f32) -> f32 { if a > 0.0 { x } else { x * 2.0 } }");
    let graph = kernel_to_graph(&kernel, &[1.0]).expect("translate");
    let input = scalar_input_bounds(1.0, 5.0).expect("bounds");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP should succeed for identity output");
    let lo = output.lower().as_slice().expect("lower slice")[0];
    let hi = output.upper().as_slice().expect("upper slice")[0];
    // Identity passthrough: output bounds == input bounds [1.0, 5.0]
    assert!((lo - 1.0).abs() < 1e-6, "lower: {lo}");
    assert!((hi - 5.0).abs() < 1e-6, "upper: {hi}");
}
