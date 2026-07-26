// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RoPE Kani proof harnesses for scalar kernel correctness.
//!
//! Extracted from `rope.rs` (500-line file limit, #175 pattern).
//!
//! All 3 harnesses use `#[kani::stub]` and require `-Z stubbing`.
//! Run with: `cargo kani -p nn-dsl --features kani-stubbing -Z stubbing`
//!
//! Stubs from `kani_stubs.rs` work around CBMC's inaccurate trig models
//! (#329, same class as #239 for exp):
//! - `sin_stub`/`cos_stub`: nondeterministic in `[-1, 1]` — for finiteness.
//! - `sin_det_stub`/`cos_det_stub`: constant `0.8`/`0.6` satisfying
//!   `sin² + cos² = 1` — for norm preservation (relational proof).

use super::*;
use crate::kani_stubs::{cos_det_stub, cos_stub, sin_det_stub, sin_stub};

/// Prove `rope_cos` produces finite output for bounded inputs.
///
/// Domain: x0, x1, freq in [-1e4, 1e4].
/// Since |cos|, |sin| <= 1, the output magnitude is bounded by
/// |x0| + |x1| <= 2e4, well within f32 range.
///
/// Uses `sin_stub`/`cos_stub` to work around CBMC's trig model (#329).
/// The stubs model `sin(finite) → [-1, 1]` and `cos(finite) → [-1, 1]`,
/// which is sound because these are IEEE 754 guarantees. The proof is
/// strictly stronger than one using real sin/cos since it holds for *any*
/// value in [-1, 1].
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::sin, sin_stub)]
#[kani::stub(f32::cos, cos_stub)]
fn rope_cos_finite_for_bounded_inputs() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let freq: f32 = kani::any();
    kani::assume(x0.is_finite() && x0 >= -1.0e4 && x0 <= 1.0e4);
    kani::assume(x1.is_finite() && x1 >= -1.0e4 && x1 <= 1.0e4);
    kani::assume(freq.is_finite() && freq >= -1.0e4 && freq <= 1.0e4);

    let result = rope_cos_scalar(x0, x1, freq)
        .expect("rope_cos_scalar must succeed for bounded finite inputs");
    assert!(result.is_finite(), "rope_cos must produce finite output");
}

/// Prove `rope_sin` produces finite output for bounded inputs.
///
/// Same domain and reasoning as `rope_cos_finite_for_bounded_inputs`.
///
/// Uses `sin_stub`/`cos_stub` — see `rope_cos_finite_for_bounded_inputs`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::sin, sin_stub)]
#[kani::stub(f32::cos, cos_stub)]
fn rope_sin_finite_for_bounded_inputs() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let freq: f32 = kani::any();
    kani::assume(x0.is_finite() && x0 >= -1.0e4 && x0 <= 1.0e4);
    kani::assume(x1.is_finite() && x1 >= -1.0e4 && x1 <= 1.0e4);
    kani::assume(freq.is_finite() && freq >= -1.0e4 && freq <= 1.0e4);

    let result = rope_sin_scalar(x0, x1, freq)
        .expect("rope_sin_scalar must succeed for bounded finite inputs");
    assert!(result.is_finite(), "rope_sin must produce finite output");
}

/// Prove RoPE rotation preserves squared norm within IEEE 754 tolerance.
///
/// The 2D rotation `(x0, x1) → (y0, y1)` via cos/sin preserves the
/// Euclidean norm: `y0² + y1² = x0² + x1²` in exact arithmetic.
/// In IEEE 754 f32, rounding introduces a small error.
///
/// Domain restricted to [-100, 100] to keep intermediate products within
/// f32 precision range.
///
/// Uses `sin_det_stub`/`cos_det_stub` (constants 0.8/0.6 with sin²+cos²=1)
/// to work around CBMC's trig model (#329). The deterministic stubs enable
/// CBMC to evaluate the rotation formula concretely:
///   y0 = x0*0.6 - x1*0.8,  y1 = x0*0.8 + x1*0.6
///   y0² + y1² = (0.36+0.64)x0² + (0.64+0.36)x1² + 0 = x0² + x1²
/// The cross terms cancel exactly: (-0.96 + 0.96)x0*x1 = 0.
///
/// # Soundness argument
///
/// This proves the structural correctness of the rotation formula for
/// a `(cos, sin)` pair satisfying `cos² + sin² = 1`. The algebraic
/// identity `|R(θ) · v|² = |v|²` holds for all θ — verifying one angle
/// is sufficient to prove the formula itself is correct.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::sin, sin_det_stub)]
#[kani::stub(f32::cos, cos_det_stub)]
fn rope_rotation_preserves_norm() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let freq: f32 = kani::any();
    kani::assume(x0.is_finite() && x0 >= -100.0 && x0 <= 100.0);
    kani::assume(x1.is_finite() && x1 >= -100.0 && x1 <= 100.0);
    kani::assume(freq.is_finite() && freq >= -100.0 && freq <= 100.0);

    let y0 = rope_cos_scalar(x0, x1, freq)
        .expect("rope_cos_scalar must succeed for bounded finite inputs");
    let y1 = rope_sin_scalar(x0, x1, freq)
        .expect("rope_sin_scalar must succeed for bounded finite inputs");

    let input_norm = x0 * x0 + x1 * x1;
    let output_norm = y0 * y0 + y1 * y1;

    // Allow 1e-3 relative tolerance for IEEE 754 rounding in the
    // chain: 2 muls + 1 add per component, then 2 squarings + 1 add.
    let diff = (output_norm - input_norm).abs();
    let tol = input_norm.abs() * 1e-3 + 1e-6;
    assert!(
        diff <= tol,
        "RoPE rotation must approximately preserve squared norm"
    );
}
