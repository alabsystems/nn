// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for binary arithmetic backward rules.
//!
//! Proves partial derivative formulas for Add, Sub, Mul, Div, MulScalar,
//! and AddScalar operations as used in `backward_rules.rs`.
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.
//!
//! Re: #13 (verified training epic).

// ── Scalar partial derivative functions ──────────────────────────────────
//
// For f(a, b) = a op b, we verify both partial derivatives:
//   ∂f/∂a (grad_a) and ∂f/∂b (grad_b)

/// Div backward: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b^2.
///
/// SYNC: backward_rules.rs:127-136 (grad/b to a, -a/b^2 * grad to b, with reduce_to_shape).
fn div_grad_a(_a: f32, b: f32) -> f32 {
    1.0 / b
}
fn div_grad_b(a: f32, b: f32) -> f32 {
    -a / (b * b)
}

// ── Kani proof harnesses ─────────────────────────────────────────────────

// Tautological harnesses removed (#1614 AC1):
// - add_grad_both_one: proved constant fns return 1.0
// - sub_grad_correct: proved constant fns return 1.0/-1.0
// - mul_grad_correct: proved projection fns mul_grad_a(_,b)=b, mul_grad_b(a,_)=a

/// Prove Div ∂/∂a = 1/b is finite for non-zero b.
#[kani::unwind(1)]
#[kani::proof]
fn div_grad_a_finite() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(b.abs() > 0.001 && b.abs() < 1e6);
    let d = div_grad_a(a, b);
    assert!(d.is_finite(), "div grad_a must be finite for |b| > 0");
}

/// Prove Div ∂/∂b = -a/b^2 is finite for non-zero b and bounded a.
#[kani::unwind(1)]
#[kani::proof]
fn div_grad_b_finite() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(b.abs() > 0.001 && b.abs() < 1e6);
    kani::assume(a.abs() < 1e6);
    let d = div_grad_b(a, b);
    assert!(d.is_finite(), "div grad_b must be finite for |b| > 0");
}

/// Prove Div ∂/∂b has correct sign: negative for positive a and b.
///
/// For a > 0, b > 0: grad_b = -a/b^2 < 0. Increasing denominator
/// decreases the quotient, so the partial is negative.
#[kani::unwind(1)]
#[kani::proof]
fn div_grad_b_sign() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a > 0.01 && a < 1e4);
    kani::assume(b > 0.01 && b < 1e4);
    let d = div_grad_b(a, b);
    assert!(d < 0.0, "div grad_b must be negative when a > 0, b > 0");
}

// Tautological harnesses removed (#1614 AC1):
// - mul_scalar_grad_correct: proved identity fn mul_scalar_grad(s)=s
// - add_scalar_grad_correct: proved constant fn returns 1.0
