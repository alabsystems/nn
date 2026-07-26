// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for unary activation backward derivatives.
//!
//! Extracted from `kani_backward_proofs.rs` for 500-line compliance.
//! Proves finiteness, correctness, sign, and bound properties of each
//! activation's scalar derivative formula used in `backward_rules.rs`.
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.
//!
//! Re: #13 (verified training epic).

use super::*;

// ── ReLU ─────────────────────────────────────────────────────────

/// Prove ReLU derivative is finite for all finite inputs.
#[kani::unwind(1)]
#[kani::proof]
fn relu_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = relu_derivative(x);
    assert!(d.is_finite(), "relu derivative must be finite");
}

/// Prove ReLU derivative is 0 or 1 (binary).
#[kani::unwind(1)]
#[kani::proof]
fn relu_derivative_binary() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = relu_derivative(x);
    assert!(d == 0.0 || d == 1.0, "relu derivative must be 0 or 1");
}

/// Prove ReLU derivative is non-negative.
#[kani::unwind(1)]
#[kani::proof]
fn relu_derivative_nonneg() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = relu_derivative(x);
    assert!(d >= 0.0, "relu derivative must be >= 0");
}

// ── Tanh ─────────────────────────────────────────────────────────

/// Prove tanh derivative is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn tanh_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -10.0 && x <= 10.0);
    let d = tanh_derivative(x);
    assert!(d.is_finite(), "tanh derivative must be finite");
}

/// Prove tanh derivative is non-negative (tanh is monotonically increasing).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn tanh_derivative_nonneg() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -10.0 && x <= 10.0);
    let d = tanh_derivative(x);
    assert!(
        d >= 0.0,
        "tanh derivative must be >= 0 (monotone increasing)"
    );
}

/// Prove tanh derivative is at most 1 (maximum at x=0).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn tanh_derivative_at_most_one() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -10.0 && x <= 10.0);
    let d = tanh_derivative(x);
    assert!(d <= 1.0 + 1e-6, "tanh derivative must be <= 1");
}

// ── Sigmoid ──────────────────────────────────────────────────────

/// Prove sigmoid derivative is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn sigmoid_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -80.0 && x <= 80.0);
    let d = sigmoid_derivative(x);
    assert!(d.is_finite(), "sigmoid derivative must be finite");
}

/// Prove sigmoid derivative is non-negative (sigmoid is monotone).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn sigmoid_derivative_nonneg() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -80.0 && x <= 80.0);
    let d = sigmoid_derivative(x);
    assert!(d >= 0.0, "sigmoid derivative must be >= 0 (monotone)");
}

// ── Exp ──────────────────────────────────────────────────────────

/// Prove exp derivative is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn exp_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -80.0 && x <= 80.0);
    let d = exp_derivative(x);
    assert!(d.is_finite(), "exp derivative must be finite");
}

/// Prove exp derivative is strictly positive.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn exp_derivative_positive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -80.0 && x <= 80.0);
    let d = exp_derivative(x);
    assert!(d > 0.0, "exp derivative must be > 0");
}

// ── Log ──────────────────────────────────────────────────────────

/// Prove log derivative is finite for strictly positive inputs.
#[kani::unwind(1)]
#[kani::proof]
fn log_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.01 && x <= 1e6);
    let d = log_derivative(x);
    assert!(d.is_finite(), "log derivative must be finite for x > 0");
}

/// Prove log derivative is positive for positive inputs.
#[kani::unwind(1)]
#[kani::proof]
fn log_derivative_positive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.01 && x <= 1e6);
    let d = log_derivative(x);
    assert!(d > 0.0, "log derivative must be > 0 for x > 0");
}

// ── Sqrt ─────────────────────────────────────────────────────────

/// Prove sqrt derivative is finite for strictly positive inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn sqrt_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.01 && x <= 1e6);
    let d = sqrt_derivative(x);
    assert!(d.is_finite(), "sqrt derivative must be finite for x > 0");
}

/// Prove sqrt derivative is positive for positive inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn sqrt_derivative_positive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.01 && x <= 1e6);
    let d = sqrt_derivative(x);
    assert!(d > 0.0, "sqrt derivative must be > 0 for x > 0");
}

/// Prove sqrt derivative returns zero at x=0 boundary (subderivative convention).
///
/// Production backward rule (#2002) masks x<=0 to zero gradient.
/// This harness proves the Kani model matches that convention.
#[kani::unwind(1)]
#[kani::proof]
fn sqrt_derivative_zero_at_boundary() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x <= 0.0 && x >= -1e6);
    let d = sqrt_derivative(x);
    assert!(
        d == 0.0,
        "sqrt derivative must be 0 for x <= 0 (subderivative)"
    );
}

/// Prove sqrt derivative is finite for ALL finite non-negative inputs,
/// including x=0 (the boundary that caused #2002).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn sqrt_derivative_finite_including_zero() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= 0.0 && x <= 1e6);
    let d = sqrt_derivative(x);
    assert!(d.is_finite(), "sqrt derivative must be finite for x >= 0");
}

// ── Sqr ──────────────────────────────────────────────────────────

/// Prove sqr derivative is finite for all finite inputs.
#[kani::unwind(1)]
#[kani::proof]
fn sqr_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -1e18 && x <= 1e18);
    let d = sqr_derivative(x);
    assert!(d.is_finite(), "sqr derivative must be finite");
}

// Tautological harnesses removed (#1614 AC1):
// - sqr_derivative_exact: proved sqr_derivative(x)=2*x where body IS 2*x
// - neg_derivative_constant: proved neg_derivative(_)=-1.0 where body IS -1.0

// ── SiLU ─────────────────────────────────────────────────────────

/// Prove SiLU derivative is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn silu_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -80.0 && x <= 80.0);
    let d = silu_derivative(x);
    assert!(d.is_finite(), "silu derivative must be finite");
}

/// Prove SiLU derivative is non-negative for x >= 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn silu_derivative_nonneg_for_positive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= 0.0 && x <= 80.0);
    let d = silu_derivative(x);
    assert!(d >= 0.0, "silu derivative must be >= 0 for x >= 0");
}

/// SiLU derivative is bounded below by -0.28 (true minimum is -0.2784 at x ≈ -1.278).
///
/// NOTE: This tight bound CANNOT be proven with nondeterministic exp stub because
/// the stub allows sigmoid(x) values inconsistent with x (e.g., s=0.5 when x=-80).
/// The derivative formula s*(1+x*(1-s)) produces values << -0.28 when s is
/// unrealistically large for negative x. Requires a deterministic exp stub or
/// NY interval propagation. The finiteness and non-negativity proofs
/// above are sound because they only need structural properties of exp (positive, finite).
///
/// Weakened to -41.0 bound (worst case: s=0.5, x=-80 gives 0.5*(1-40)=-19.5;
/// actual provable bound with stub is ~-40.5 due to s just above 0.5).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn silu_derivative_lower_bound_with_stub() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -80.0 && x <= 80.0);
    let d = silu_derivative(x);
    // Provable with nondeterministic stub: s in (0,1), x in [-80,80]
    // Worst case: s*(1 + x*(1-s)) with s=0.5, x=-80 -> 0.5*(-39) = -19.5
    // Upper bound with s->1: 1*(1+80*0) = 1, lower with s->0.5, x=-80: -19.5
    // Use -41 as sound upper bound accounting for float rounding
    assert!(
        d >= -41.0,
        "silu derivative must be >= -41 (stub-provable bound)"
    );
}

// ── Abs ──────────────────────────────────────────────────────────

/// Prove abs derivative is finite for all finite inputs.
#[kani::unwind(1)]
#[kani::proof]
fn abs_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = abs_derivative(x);
    assert!(d.is_finite(), "abs derivative must be finite");
}

/// Prove abs derivative is in {-1, 0, 1}.
#[kani::unwind(1)]
#[kani::proof]
fn abs_derivative_ternary() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = abs_derivative(x);
    assert!(
        d == -1.0 || d == 0.0 || d == 1.0,
        "abs derivative must be -1, 0, or 1"
    );
}

/// Prove abs derivative sign agrees with input sign.
///
/// For x > 0: d = 1 > 0. For x < 0: d = -1 < 0.
#[kani::unwind(1)]
#[kani::proof]
fn abs_derivative_sign() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x != 0.0);
    let d = abs_derivative(x);
    if x > 0.0 {
        assert!(d > 0.0, "abs'(x) > 0 when x > 0");
    } else {
        assert!(d < 0.0, "abs'(x) < 0 when x < 0");
    }
}

// ── GELU ─────────────────────────────────────────────────────────

/// Prove GELU derivative is finite for bounded inputs.
///
/// Uses tanh stub because CBMC cannot model tanh accurately.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn gelu_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -5.0 && x <= 5.0);
    let d = gelu_derivative(x);
    assert!(d.is_finite(), "gelu derivative must be finite");
}

/// Prove GELU derivative is non-negative for x >= 0.
///
/// GELU is monotonically increasing for x >= 0, so its
/// derivative is non-negative in this region.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn gelu_derivative_nonneg_for_positive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= 0.0 && x <= 5.0);
    let d = gelu_derivative(x);
    assert!(d >= -1e-6, "gelu derivative must be >= 0 for x >= 0");
}

/// GELU derivative bounded below (true minimum ~-0.170 at x ≈ -1.545).
///
/// NOTE: Tight -0.18 bound cannot be proven with nondeterministic tanh stub.
/// The stub allows tanh(s) values inconsistent with s, making sech2 and
/// term2 = 0.5*x*sech2*s' unbounded in the over-approximate model.
/// Worst case with stub: term1 >= 0, term2 = 0.5*(-5)*1.0*s'_max.
/// s' = sqrt(2/pi)*(1+3*0.044715*25) ≈ 3.47, so term2 >= -8.7.
/// Use -10.0 as sound bound accounting for float rounding.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn gelu_derivative_lower_bound_with_stub() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -5.0 && x <= 5.0);
    let d = gelu_derivative(x);
    assert!(
        d >= -10.0,
        "gelu derivative must be >= -10.0 (stub-provable bound)"
    );
}

// ── Stubs for CBMC transcendentals ───────────────────────────────
//
// CBMC cannot accurately model f32 transcendentals (exp, tanh, sqrt).
// Same pattern as nn-dsl (#708, #541).

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result > 0.0);
    result
}

fn tanh_f32_stub(x: f32) -> f32 {
    let _ = x;
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result >= -1.0 && result <= 1.0);
    result
}

fn sqrt_f32_stub(x: f32) -> f32 {
    let result: f32 = kani::any();
    // Must be strictly positive and bounded: sqrt(x) > 0 for x > 0.
    // Lower bound (1e-19) prevents 1/(2*result) overflow in sqrt_derivative.
    // Sound: harness constrains x > 0.01, so real sqrt(x) > 0.1 >> 1e-19.
    // Upper bound prevents 2.0 * result overflow in sqrt_derivative.
    kani::assume(result.is_finite() && result >= 1e-19 && result <= 1e18);
    if x > 0.0 {
        kani::assume(result > 0.0);
    }
    result
}
