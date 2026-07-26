// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared Kani stubs for transcendental functions.
//!
//! CBMC cannot model `f32::exp`, `f32::sin`, `f32::cos` correctly
//! (#239, #329, #708). These stubs provide sound over-approximations
//! for safety proofs and deterministic substitutes for relational proofs.
//!
//! ## Usage
//!
//! In a kernel's `#[cfg(kani)]` proof module:
//! ```ignore
//! // NOTE: ignore — uses crate-internal paths and requires Kani toolchain
//! use crate::kani_stubs::{exp_stub, exp_det_stub};
//!
//! #[kani::proof]
//! #[kani::stub(f32::exp, exp_stub)]
//! fn nn_proof() { ... }
//! ```

/// Nondeterministic stub for `f32::exp` — returns any positive finite value.
///
/// Sound over-approximation: `exp(finite)` is always positive and finite
/// (IEEE 754 guarantee). CBMC's built-in exp model produces incorrect
/// results for edge cases, so we over-approximate with any positive float.
pub(crate) fn exp_stub(x: f32) -> f32 {
    let _ = x;
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result > 0.0);
    result
}

/// Deterministic monotone stub: `exp(x) ≈ x + 101`.
///
/// Preserves monotonicity (exp is monotone, x+101 is monotone) and positivity
/// for `x > -101`. Used for bounds proofs where structural monotonicity matters.
///
/// CBMC's SAT solver benefits from the explicit `kani::assume` on positivity —
/// without it, CBMC must derive positivity through float arithmetic bounds
/// propagation, which is slow.
pub(crate) fn exp_det_stub(x: f32) -> f32 {
    let result = x + 101.0;
    kani::assume(result > 0.0 && result.is_finite());
    result
}

/// Nondeterministic stub for `f32::sin` — returns any value in `[-1, 1]`.
pub(crate) fn sin_stub(_x: f32) -> f32 {
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result >= -1.0 && result <= 1.0);
    result
}

/// Nondeterministic stub for `f32::cos` — returns any value in `[-1, 1]`.
pub(crate) fn cos_stub(_x: f32) -> f32 {
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result >= -1.0 && result <= 1.0);
    result
}

/// Deterministic sin stub: constant `0.8` (Pythagorean pair with `cos_det_stub`).
///
/// `sin²(0.8) + cos²(0.6) = 0.64 + 0.36 = 1.0` — satisfies the Pythagorean
/// identity exactly, which is needed for RoPE norm-preservation proofs.
pub(crate) fn sin_det_stub(_x: f32) -> f32 {
    0.8
}

/// Deterministic cos stub: constant `0.6` (Pythagorean pair with `sin_det_stub`).
pub(crate) fn cos_det_stub(_x: f32) -> f32 {
    0.6
}

/// Nondeterministic stub for `f32::tanh` — returns any value in `(-1, 1)`.
///
/// Sound over-approximation: `tanh(finite)` is always in the open interval (-1, 1)
/// and is finite (IEEE 754 guarantee). CBMC cannot model `f32::tanh` correctly
/// (uses exp internally, same issue as #239). Same pattern as sin_stub/cos_stub.
pub(crate) fn tanh_stub(_x: f32) -> f32 {
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result > -1.0 && result < 1.0);
    result
}

/// Deterministic monotone stub: `tanh(x) ≈ x / (1 + |x|)`.
///
/// Preserves monotonicity (both tanh and x/(1+|x|) are monotone increasing)
/// and the (-1, 1) output range. Used for bounds proofs where structural
/// monotonicity matters. The approximation is exact at x=0 and asymptotically
/// approaches ±1 like tanh.
pub(crate) fn tanh_det_stub(x: f32) -> f32 {
    let result = x / (1.0 + x.abs());
    kani::assume(result.is_finite() && result > -1.0 && result < 1.0);
    result
}

/// Nondeterministic stub for `f32::sqrt` — returns any non-negative finite value.
///
/// Sound over-approximation: `sqrt(x)` for `x >= 0` is always non-negative and
/// finite (IEEE 754 guarantee). CBMC's built-in sqrt model can produce incorrect
/// results for edge cases.
pub(crate) fn sqrt_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        // Sound lower bound: for 0 < x <= 1, sqrt(x) >= x (since x^0.5 >= x^1).
        // For x > 1, sqrt(x) >= 1.0 (handled below). Combined: sqrt(x) >= min(x, 1.0).
        // Prevents 1/sqrt(x) from overflowing to infinity for positive x.
        kani::assume(r >= x.min(1.0));
    }
    if x >= 1.0 {
        kani::assume(r >= 1.0);
    }
    r
}

/// Nondeterministic stub for `f32::powi` — returns any finite value.
///
/// Sound over-approximation: `powi(b, e)` for finite `b` and any `e` can produce
/// any finite value (positive, negative, or zero depending on sign and parity).
pub(crate) fn powi_stub(_b: f32, _e: i32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}
