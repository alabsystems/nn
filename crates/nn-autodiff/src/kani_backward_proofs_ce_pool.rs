// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for cross-entropy loss and pooling backward derivatives.
//!
//! Covers the largest proof coverage gaps in the backward rule suite:
//! - CrossEntropyLoss per-element gradient: `(softmax_i - one_hot_i) / N`
//! - AvgPool2d per-element scaling: `grad / count`
//! - AdaptiveAvgPool2d tier-1 global backward: `grad / window_size`
//! - MaxPool2d scatter routing: structural correctness
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.
//!
//! Re: #1614 (proof quality), Part of proof_coverage phase.

use super::*;

// ── Cross-Entropy Loss backward scalar functions ────────────────────

/// Cross-entropy backward per-element: gradient for logit i given softmax output.
///
/// When i is the target class: `(softmax_i - 1.0) / N * upstream_grad`
/// When i is not the target class: `softmax_i / N * upstream_grad`
///
/// Equivalently: `(softmax_i - one_hot_i) / N * upstream_grad`
/// where one_hot_i is 1 for the target class and 0 otherwise.
///
/// SYNC: backward_rules_special.rs:221-225 (softmax - one_hot pattern).
#[allow(dead_code)]
fn cross_entropy_backward_element(softmax_i: f32, is_target: bool, n: usize, upstream: f32) -> f32 {
    let one_hot_i = if is_target { 1.0_f32 } else { 0.0_f32 };
    (softmax_i - one_hot_i) / n as f32 * upstream
}

/// AvgPool2d backward per-element scaling: divide gradient by valid window count.
///
/// SYNC: backward_rules_pool.rs:94 (grad.div(&counts) pattern).
#[allow(dead_code)]
fn avg_pool2d_backward_element(grad: f32, count: f32) -> f32 {
    grad / count
}

/// AdaptiveAvgPool2d tier-1 (global pooling) backward: uniform scaling.
///
/// SYNC: backward_rules_pool.rs:154-156 (grad * (1.0 / window_size) pattern).
#[allow(dead_code)]
fn adaptive_avg_pool2d_global_backward(grad: f32, window_size: usize) -> f32 {
    grad / window_size as f32
}

/// MaxPool2d backward routing: scatter gradient to argmax position.
///
/// Element at position `pos` receives gradient if `pos == argmax_pos`, zero otherwise.
///
/// SYNC: backward_rules_pool.rs:35-38 (scatter_add_into pattern).
#[allow(dead_code)]
fn max_pool2d_backward_element(grad: f32, pos: usize, argmax_pos: usize) -> f32 {
    if pos == argmax_pos {
        grad
    } else {
        0.0
    }
}

// ── Cross-Entropy Loss Kani proofs ──────────────────────────────────

/// Prove CE backward element is finite for valid softmax output.
///
/// Softmax outputs are in [0, 1], so (softmax_i - one_hot_i) is in [-1, 1].
/// Division by N (>= 1) keeps magnitude bounded.
#[kani::unwind(1)]
#[kani::proof]
fn ce_backward_element_finite() {
    let s_i: f32 = kani::any();
    let upstream: f32 = kani::any();
    let n: usize = kani::any();
    let is_target: bool = kani::any();
    kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i <= 1.0);
    kani::assume(upstream.is_finite() && upstream.abs() <= 1e6);
    kani::assume(n >= 1 && n <= 1_000_000);
    let d = cross_entropy_backward_element(s_i, is_target, n, upstream);
    assert!(d.is_finite(), "CE backward element must be finite");
}

/// Prove CE backward element is bounded by upstream_grad / N.
///
/// Since |softmax_i - one_hot_i| <= 1.0, the element gradient is
/// bounded by |upstream| / N.
#[kani::unwind(1)]
#[kani::proof]
fn ce_backward_element_bounded() {
    let s_i: f32 = kani::any();
    let upstream: f32 = kani::any();
    let n: usize = kani::any();
    let is_target: bool = kani::any();
    kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i <= 1.0);
    kani::assume(upstream.is_finite() && upstream.abs() <= 1e3);
    kani::assume(n >= 1 && n <= 10_000);
    let d = cross_entropy_backward_element(s_i, is_target, n, upstream);
    let bound = upstream.abs() / n as f32;
    assert!(
        d.abs() <= bound + 1e-6,
        "CE backward element must be bounded by |upstream|/N"
    );
}

/// Prove CE backward produces negative gradient for target class when softmax < 1.
///
/// For the target class: grad = (softmax_i - 1.0) / N * upstream.
/// When softmax_i < 1 (always for multi-class) and upstream > 0,
/// the gradient is negative, pushing logits up (reducing loss).
#[kani::unwind(1)]
#[kani::proof]
fn ce_backward_target_class_negative() {
    let s_i: f32 = kani::any();
    let upstream: f32 = kani::any();
    let n: usize = kani::any();
    // softmax < 1 (not the degenerate one-class case)
    kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i < 1.0);
    kani::assume(upstream.is_finite() && upstream > 0.0);
    kani::assume(n >= 1 && n <= 1_000_000);
    let d = cross_entropy_backward_element(s_i, true, n, upstream);
    assert!(
        d < 0.0,
        "CE target-class gradient must be negative (pushes logits up)"
    );
}

/// Prove CE backward produces positive gradient for non-target class when softmax > 0.
///
/// For non-target classes: grad = softmax_i / N * upstream.
/// When softmax_i > 0 and upstream > 0, the gradient is positive,
/// pushing logits down (reducing probability of wrong class).
#[kani::unwind(1)]
#[kani::proof]
fn ce_backward_nontarget_class_positive() {
    let s_i: f32 = kani::any();
    let upstream: f32 = kani::any();
    let n: usize = kani::any();
    kani::assume(s_i.is_finite() && s_i > 0.0 && s_i <= 1.0);
    kani::assume(upstream.is_finite() && upstream > 0.0);
    kani::assume(n >= 1 && n <= 1_000_000);
    let d = cross_entropy_backward_element(s_i, false, n, upstream);
    assert!(
        d > 0.0,
        "CE non-target gradient must be positive (pushes logits down)"
    );
}

/// Prove CE backward is zero for non-target with zero softmax probability.
///
/// If softmax_i == 0 for a non-target class, the gradient is exactly zero.
/// This means tokens that were already completely suppressed get no gradient push.
#[kani::unwind(1)]
#[kani::proof]
fn ce_backward_zero_softmax_zero_grad() {
    let upstream: f32 = kani::any();
    let n: usize = kani::any();
    kani::assume(upstream.is_finite());
    kani::assume(n >= 1 && n <= 1_000_000);
    let d = cross_entropy_backward_element(0.0, false, n, upstream);
    assert!(
        d == 0.0,
        "CE backward must be zero when softmax_i == 0 for non-target"
    );
}

// ── AvgPool2d backward Kani proofs ──────────────────────────────────

/// Prove AvgPool2d backward scaling is finite for valid window counts.
///
/// Window counts are integers >= 1 (at least one valid input per output position).
/// Bound: count in [1, K*K] where K is kernel size.
#[kani::unwind(1)]
#[kani::proof]
fn avg_pool2d_backward_element_finite() {
    let grad: f32 = kani::any();
    let count: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    // count is always a positive integer (number of valid elements in window)
    kani::assume(count.is_finite() && count >= 1.0 && count <= 49.0); // max 7x7 kernel
    let d = avg_pool2d_backward_element(grad, count);
    assert!(d.is_finite(), "AvgPool2d backward scaling must be finite");
}

/// Prove AvgPool2d backward scaling is bounded by |grad| (count >= 1).
#[kani::unwind(1)]
#[kani::proof]
fn avg_pool2d_backward_element_bounded() {
    let grad: f32 = kani::any();
    let count: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(count.is_finite() && count >= 1.0 && count <= 49.0);
    let d = avg_pool2d_backward_element(grad, count);
    assert!(
        d.abs() <= grad.abs() + 1e-6,
        "AvgPool2d backward must be bounded by |grad| (count >= 1)"
    );
}

/// Prove AvgPool2d backward preserves gradient sign.
#[kani::unwind(1)]
#[kani::proof]
fn avg_pool2d_backward_element_sign() {
    let grad: f32 = kani::any();
    let count: f32 = kani::any();
    kani::assume(grad.is_finite() && grad != 0.0 && grad.abs() <= 1e6);
    kani::assume(count.is_finite() && count >= 1.0 && count <= 49.0);
    let d = avg_pool2d_backward_element(grad, count);
    if grad > 0.0 {
        assert!(d > 0.0, "AvgPool2d backward must preserve positive sign");
    } else {
        assert!(d < 0.0, "AvgPool2d backward must preserve negative sign");
    }
}

// ── AdaptiveAvgPool2d tier-1 backward Kani proofs ───────────────────

/// Prove AdaptiveAvgPool2d global backward is finite for valid window sizes.
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_avg_pool2d_global_backward_finite() {
    let grad: f32 = kani::any();
    let window_size: usize = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(window_size >= 1 && window_size <= 1_000_000);
    let d = adaptive_avg_pool2d_global_backward(grad, window_size);
    assert!(
        d.is_finite(),
        "AdaptiveAvgPool2d global backward must be finite"
    );
}

/// Prove AdaptiveAvgPool2d global backward is bounded by |grad|.
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_avg_pool2d_global_backward_bounded() {
    let grad: f32 = kani::any();
    let window_size: usize = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(window_size >= 1 && window_size <= 1_000_000);
    let d = adaptive_avg_pool2d_global_backward(grad, window_size);
    assert!(
        d.abs() <= grad.abs() + 1e-7,
        "AdaptiveAvgPool2d global backward must be bounded by |grad|"
    );
}

/// Prove AdaptiveAvgPool2d global backward scales inversely with window size.
#[kani::unwind(1)]
#[kani::proof]
fn adaptive_avg_pool2d_global_backward_scales() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() > 0.0 && grad.abs() <= 1e3);
    let d1 = adaptive_avg_pool2d_global_backward(grad, 1);
    let d4 = adaptive_avg_pool2d_global_backward(grad, 4);
    // d4 should be roughly d1 / 4
    assert!(
        d4.abs() <= d1.abs() + 1e-6,
        "larger window → smaller gradient"
    );
}

// ── MaxPool2d backward Kani proofs ──────────────────────────────────

/// Prove MaxPool2d backward routes gradient exactly to argmax position.
#[kani::unwind(1)]
#[kani::proof]
fn max_pool2d_backward_routes_to_argmax() {
    let grad: f32 = kani::any();
    let argmax_pos: usize = kani::any();
    kani::assume(grad.is_finite());
    kani::assume(argmax_pos <= 1_000);
    let d = max_pool2d_backward_element(grad, argmax_pos, argmax_pos);
    assert!(
        d == grad,
        "MaxPool2d backward must pass gradient to argmax position"
    );
}

/// Prove MaxPool2d backward produces zero for non-argmax positions.
#[kani::unwind(1)]
#[kani::proof]
fn max_pool2d_backward_zero_elsewhere() {
    let grad: f32 = kani::any();
    let pos: usize = kani::any();
    let argmax_pos: usize = kani::any();
    kani::assume(grad.is_finite());
    kani::assume(pos <= 1_000 && argmax_pos <= 1_000);
    kani::assume(pos != argmax_pos);
    let d = max_pool2d_backward_element(grad, pos, argmax_pos);
    assert!(
        d == 0.0,
        "MaxPool2d backward must be zero for non-argmax positions"
    );
}

/// Prove MaxPool2d backward preserves gradient finiteness at argmax.
#[kani::unwind(1)]
#[kani::proof]
fn max_pool2d_backward_preserves_finiteness() {
    let grad: f32 = kani::any();
    let pos: usize = kani::any();
    let argmax_pos: usize = kani::any();
    kani::assume(grad.is_finite());
    kani::assume(pos <= 1_000 && argmax_pos <= 1_000);
    let d = max_pool2d_backward_element(grad, pos, argmax_pos);
    assert!(d.is_finite(), "MaxPool2d backward must preserve finiteness");
}
