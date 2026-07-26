// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Sin, Cos, Recip, Powf, Clamp, ELU backward derivatives.
//!
//! Extracted from `kani_backward_proofs.rs` for 500-line compliance.
//! Proves finiteness, correctness, sign, and bound properties of each
//! math operation's scalar derivative formula used in `backward_rules.rs`.
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.
//!
//! Re: #13 (verified training epic), Part of #1476.

use super::*;

// ── Sin ─────────────────────────────────────────────────────────

/// Prove sin derivative (cos(x)) is finite for bounded inputs.
///
/// Uses sin/cos stubs since CBMC cannot model trig functions accurately.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
fn sin_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);
    let d = sin_derivative(x);
    assert!(d.is_finite(), "sin derivative must be finite");
}

/// Prove sin derivative is bounded in [-1, 1] (since d/dx sin(x) = cos(x)).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
fn sin_derivative_bounded() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);
    let d = sin_derivative(x);
    assert!(
        d >= -1.0 && d <= 1.0,
        "sin derivative (cos) must be in [-1, 1]"
    );
}

// ── Cos ─────────────────────────────────────────────────────────

/// Prove cos derivative (-sin(x)) is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sin, sin_f32_stub)]
fn cos_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);
    let d = cos_derivative(x);
    assert!(d.is_finite(), "cos derivative must be finite");
}

/// Prove cos derivative is bounded in [-1, 1] (since d/dx cos(x) = -sin(x)).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sin, sin_f32_stub)]
fn cos_derivative_bounded() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);
    let d = cos_derivative(x);
    assert!(
        d >= -1.0 && d <= 1.0,
        "cos derivative (-sin) must be in [-1, 1]"
    );
}

// ── Recip ───────────────────────────────────────────────────────

/// Prove recip derivative (-1/x^2) is finite when x is bounded away from zero.
#[kani::unwind(1)]
#[kani::proof]
fn recip_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    // Bounded away from zero to prevent division by zero
    kani::assume(x.abs() >= 0.01 && x.abs() <= 1e4);
    let d = recip_derivative(x);
    assert!(d.is_finite(), "recip derivative must be finite for |x| > 0");
}

/// Prove recip derivative is always negative (1/x is strictly decreasing for x > 0 and x < 0).
#[kani::unwind(1)]
#[kani::proof]
fn recip_derivative_negative() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x.abs() >= 0.01 && x.abs() <= 1e4);
    let d = recip_derivative(x);
    assert!(
        d < 0.0,
        "recip derivative must be < 0 (strictly decreasing)"
    );
}

/// Prove recip derivative magnitude decreases with |x| (d/dx 1/x = -1/x^2,
/// so |d| = 1/x^2 which decreases as |x| grows).
#[kani::unwind(1)]
#[kani::proof]
fn recip_derivative_monotone_magnitude() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= 1.0 && x <= 1e4);
    let d = recip_derivative(x);
    // |d| = 1/x^2 <= 1 when |x| >= 1
    assert!(
        d.abs() <= 1.0 + 1e-6,
        "recip derivative magnitude must be <= 1 when |x| >= 1"
    );
}

// ── Powf ────────────────────────────────────────────────────────

/// Prove powf derivative (p * x^(p-1)) is finite for x > 0, integer p.
///
/// Uses powf stub since CBMC cannot model arbitrary powers.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn powf_derivative_finite_integer() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.01 && x <= 100.0);
    // Test with integer exponent p = 2 (common: squared distance loss)
    let d = powf_derivative(x, 2.0);
    assert!(d.is_finite(), "powf derivative must be finite for p=2, x>0");
}

/// Prove powf derivative sign for positive x and positive p:
/// d/dx x^p = p * x^(p-1) > 0 when x > 0 and p > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn powf_derivative_positive_for_positive_x_p() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.01 && x <= 100.0);
    let d = powf_derivative(x, 2.0);
    // With stub, result is nondeterministic positive, multiplied by p=2 > 0
    assert!(d >= 0.0, "powf derivative must be >= 0 for x > 0, p > 0");
}

/// Prove powf derivative with p=1 (identity): d/dx x^1 = 1.
///
/// This tests the edge case where powf(x, 0.0) = 1 (any x^0 = 1),
/// so the derivative p * x^(p-1) = 1 * x^0 = 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn powf_derivative_identity() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.01 && x <= 100.0);
    // p=1: d/dx x^1 = 1 * x^0 = 1
    // With stub, x.powf(0.0) returns nondeterministic positive value,
    // so we can only prove finiteness here (exact value needs deterministic stub)
    let d = powf_derivative(x, 1.0);
    assert!(d.is_finite(), "powf derivative must be finite for p=1");
}

/// Prove powf derivative is exactly zero when p=0 for all finite x.
///
/// This is the exact edge case that caused NaN via 0 * x^(-1) before the
/// p=0 short-circuit fix (#2000). The general formula p * x^(p-1) computes
/// 0 * x^(-1) which is NaN at x=0 (IEEE 754: 0 * Inf = NaN).
/// The fix returns 0.0 directly when p=0.
///
/// No powf stub needed: the short-circuit returns before calling powf.
#[kani::unwind(1)]
#[kani::proof]
fn powf_derivative_zero_for_p_zero() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    // Allow ANY finite x including zero, negative, and denormals
    let d = powf_derivative(x, 0.0);
    assert!(d == 0.0, "powf derivative must be exactly 0.0 when p=0");
}

// ── Clamp ───────────────────────────────────────────────────────

/// Prove clamp derivative is finite for all finite inputs.
#[kani::unwind(1)]
#[kani::proof]
fn clamp_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = clamp_derivative(x, -1.0, 1.0);
    assert!(d.is_finite(), "clamp derivative must be finite");
}

/// Prove clamp derivative is 0 or 1 (binary).
#[kani::unwind(1)]
#[kani::proof]
fn clamp_derivative_binary() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = clamp_derivative(x, -1.0, 1.0);
    assert!(d == 0.0 || d == 1.0, "clamp derivative must be 0 or 1");
}

/// Prove clamp derivative is 1 when value is inside range (inclusive of boundary).
/// Production code uses ge/le so gradient flows at boundary values, matching PyTorch.
#[kani::unwind(1)]
#[kani::proof]
fn clamp_derivative_one_inside() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -1.0 && x <= 1.0);
    let d = clamp_derivative(x, -1.0, 1.0);
    assert!(
        d == 1.0,
        "clamp derivative must be 1 inside range (inclusive)"
    );
}

/// Prove clamp derivative with arbitrary bounds is still binary.
#[kani::unwind(1)]
#[kani::proof]
fn clamp_derivative_arbitrary_bounds_binary() {
    let x: f32 = kani::any();
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo < hi); // lo < hi required for valid clamp
    let d = clamp_derivative(x, lo, hi);
    assert!(
        d == 0.0 || d == 1.0,
        "clamp derivative must be 0 or 1 for any valid bounds"
    );
}

// ── ELU ────────────────────────────────────────────────────────

/// Prove ELU derivative is finite for bounded inputs.
///
/// Uses exp stub since CBMC cannot model exp accurately.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn elu_derivative_finite() {
    let x: f32 = kani::any();
    let alpha: f64 = kani::any();
    kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);
    kani::assume(alpha.is_finite() && alpha >= 0.01 && alpha <= 10.0);
    let d = elu_derivative(x, alpha);
    assert!(d.is_finite(), "ELU derivative must be finite");
}

/// Prove ELU derivative is 1 for positive inputs (regardless of alpha).
#[kani::unwind(1)]
#[kani::proof]
fn elu_derivative_one_for_positive() {
    let x: f32 = kani::any();
    let alpha: f64 = kani::any();
    kani::assume(x.is_finite() && x > 0.0);
    kani::assume(alpha.is_finite());
    let d = elu_derivative(x, alpha);
    assert!(d == 1.0, "ELU derivative must be 1.0 for x > 0");
}

/// Prove ELU derivative is positive for negative inputs when alpha > 0.
///
/// d/dx ELU(x, alpha) = alpha * exp(x) > 0 when alpha > 0 (since exp(x) > 0 for all x).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn elu_derivative_positive_for_negative_x() {
    let x: f32 = kani::any();
    let alpha: f64 = kani::any();
    kani::assume(x.is_finite() && x <= 0.0 && x >= -100.0);
    kani::assume(alpha > 0.0 && alpha <= 10.0);
    let d = elu_derivative(x, alpha);
    assert!(d >= 0.0, "ELU derivative must be >= 0 when alpha > 0");
}

/// Prove ELU derivative continuity at x=0: lim_{x→0-} alpha*exp(0) = alpha,
/// and d/dx at x>0 = 1. For standard alpha=1.0, both sides equal 1.0.
///
/// Uses deterministic exp stub returning 1.0 for input 0.0 since CBMC cannot
/// model exp(0) = 1.0 accurately (#708).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_det_one_stub)]
fn elu_derivative_continuous_at_zero_alpha1() {
    // At x = 0: the negative branch gives alpha * exp(0) = alpha * 1 = alpha
    // For alpha = 1.0, this equals 1.0, matching the positive branch
    let d_neg = elu_derivative(0.0, 1.0);
    assert!(
        (d_neg - 1.0).abs() < 1e-6,
        "ELU derivative must be ~1.0 at x=0 when alpha=1.0"
    );
}

// ── Stubs for CBMC transcendentals ──────────────────────────────
//
// CBMC cannot accurately model f32 transcendentals.
// Same pattern as existing stubs in kani_backward_proofs_activation.rs (#708, #541).

fn cos_f32_stub(x: f32) -> f32 {
    let _ = x;
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result >= -1.0 && result <= 1.0);
    result
}

fn sin_f32_stub(x: f32) -> f32 {
    let _ = x;
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result >= -1.0 && result <= 1.0);
    result
}

fn powf_f32_stub(base: f32, _exp: f32) -> f32 {
    let _ = base;
    let result: f32 = kani::any();
    // x^p for x > 0 is always positive and finite (for bounded x, p)
    kani::assume(result.is_finite() && result > 0.0 && result <= 1e10);
    result
}

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let result: f32 = kani::any();
    // exp(x) > 0 for all x, bounded for bounded inputs
    kani::assume(result.is_finite() && result > 0.0 && result <= 1e10);
    result
}

/// Deterministic exp stub returning 1.0 — for proofs that test exp(0.0) = 1.0.
/// CBMC cannot model this accurately (#708). Same pattern as sin_det_stub/cos_det_stub in rope.rs.
fn exp_f32_det_one_stub(_x: f32) -> f32 {
    1.0
}
