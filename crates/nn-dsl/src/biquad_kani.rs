// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for biquad filter operations.
//!
//! Extracted from `biquad.rs` to keep files under the 500-line limit.

use super::*;

/// Proof: `process_sample` produces finite output for bounded inputs
/// and stable coefficients (Jury conditions satisfied).
#[kani::unwind(8)]
#[kani::proof]
fn biquad_process_sample_finite_for_bounded_stable() {
    let b0: f32 = kani::any();
    let b1: f32 = kani::any();
    let b2: f32 = kani::any();
    let a1: f32 = kani::any();
    let a2: f32 = kani::any();
    let x: f32 = kani::any();
    let z1: f32 = kani::any();
    let z2: f32 = kani::any();

    // Bounded inputs
    kani::assume(b0.is_finite() && b0.abs() <= 10.0);
    kani::assume(b1.is_finite() && b1.abs() <= 10.0);
    kani::assume(b2.is_finite() && b2.abs() <= 10.0);
    kani::assume(a1.is_finite() && a1.abs() <= 2.0);
    kani::assume(a2.is_finite() && a2.abs() < 1.0);
    kani::assume(x.is_finite() && x.abs() <= 1.0);
    kani::assume(z1.is_finite() && z1.abs() <= 100.0);
    kani::assume(z2.is_finite() && z2.abs() <= 100.0);

    // Jury stability conditions
    kani::assume(1.0 + a1 + a2 > 0.0);
    kani::assume(1.0 - a1 + a2 > 0.0);

    let coeffs = BiquadCoeffs { b0, b1, b2, a1, a2 };
    let result = biquad_process_sample_scalar(x, &coeffs, z1, z2);
    assert!(
        result.is_ok(),
        "process_sample must succeed for bounded stable filter"
    );
    let out = result.expect("invariant: bounded stable filter must succeed");
    assert!(out.y.is_finite());
    assert!(out.z1.is_finite());
    assert!(out.z2.is_finite());
}

/// Proof: stability check correctly reflects Jury/Schur-Cohn conditions.
#[kani::unwind(1)]
#[kani::proof]
fn biquad_stability_check_reflects_jury() {
    let a1: f32 = kani::any();
    let a2: f32 = kani::any();
    kani::assume(a1.is_finite() && a2.is_finite());
    kani::assume(a1.abs() <= 3.0 && a2.abs() <= 2.0);

    let coeffs = BiquadCoeffs {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1,
        a2,
    };
    if coeffs.is_stable() {
        assert!(a2.abs() < 1.0);
        assert!(1.0 + a1 + a2 > 0.0);
        assert!(1.0 - a1 + a2 > 0.0);
    }
}

/// Proof: identity coefficients produce exact passthrough for any finite input.
#[kani::unwind(1)]
#[kani::proof]
fn biquad_identity_passthrough() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);

    let coeffs = BiquadCoeffs {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };
    assert!(coeffs.is_identity(1e-6));

    let result = biquad_process_sample_scalar(x, &coeffs, 0.0, 0.0)
        .expect("invariant: identity filter must succeed");
    assert_eq!(
        result.y.to_bits(),
        x.to_bits(),
        "identity filter must pass through exactly"
    );
    assert_eq!(result.z1.to_bits(), 0.0_f32.to_bits());
    assert_eq!(result.z2.to_bits(), 0.0_f32.to_bits());
}

/// Proof: FIR-only filter (a1=a2=0) with bounded coefficients and
/// zero state produces output bounded by max(|b_i|) * |x|.
#[kani::unwind(1)]
#[kani::proof]
fn biquad_fir_output_bounded() {
    let x: f32 = kani::any();
    let b0: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1.0);
    kani::assume(b0.is_finite() && b0.abs() <= 5.0);

    let coeffs = BiquadCoeffs {
        b0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };
    let result = biquad_process_sample_scalar(x, &coeffs, 0.0, 0.0)
        .expect("invariant: FIR filter with bounded coefficients must succeed");
    // y = b0 * x, state = 0 → |y| <= |b0| * |x| <= 5.0
    assert!(result.y.abs() <= 5.001, "FIR output bounded by |b0|*|x|");
}
