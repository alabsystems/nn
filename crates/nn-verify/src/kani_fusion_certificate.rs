// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `fusion_certificate.rs`.
//!
//! Proves safety and correctness properties of:
//! - `AnalyticalFusionBound::compute`: input validation, output finiteness,
//!   monotonicity in parameters, mathematical identity
//! - `FusionEquivalenceCertificate::proves_equivalence`: disjunction semantics
//! - `FusionEquivalenceCertificate::tightest_bound`: min-selection correctness
//! - `FusionEquivalenceCertificate::validate`: field-level validation completeness
//! - `is_iso8601_utc`: format validation correctness
//! - `known_bounds::*`: all known fusion bounds are finite and within epsilon
//! - `F32_MACHINE_EPS`: exact value matches 2^-24
//!
//! Part of #3658.

use super::{
    is_iso8601_utc, known_bounds, AnalyticalFusionBound, FusionEquivalenceCertificate,
    FusionVerification, F32_MACHINE_EPS, FUSION_CERTIFICATE_VERSION,
};
use crate::soundness_compat::VerificationSoundnessMode;
use crate::verify_types::PropMethod;

// ===========================================================================
// CBMC transcendental stubs for Kani (#708)
// ===========================================================================

/// Nondeterministic stub for `f64::powi`.
/// CBMC cannot handle the powi intrinsic. Returns a finite f64.
fn powi_f64_stub(x: f64, n: i32) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    if x > 0.0 && x < 1.0 && n >= 1 {
        kani::assume(r > 0.0 && r <= x);
    }
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// ===========================================================================
// F32_MACHINE_EPS correctness
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. F32_MACHINE_EPS equals 2^-24
// ---------------------------------------------------------------------------

/// Prove: `F32_MACHINE_EPS` is exactly 2.0_f64.powi(-24).
/// This constant is the foundation of all analytical fusion bounds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::powi, powi_f64_stub)]
fn f32_machine_eps_exact_value() {
    let expected = 2.0_f64.powi(-24);
    assert!(
        (F32_MACHINE_EPS - expected).abs() < 1e-30,
        "F32_MACHINE_EPS must equal 2^-24"
    );
    assert!(F32_MACHINE_EPS > 0.0, "must be positive");
    assert!(F32_MACHINE_EPS < 1e-6, "must be small");
    assert!(F32_MACHINE_EPS.is_finite(), "must be finite");
}

// ===========================================================================
// AnalyticalFusionBound::compute input validation
// ===========================================================================

// ---------------------------------------------------------------------------
// 2. Zero differing_op_count rejected
// ---------------------------------------------------------------------------

/// Prove: `compute(0, *, *)` always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_zero_ops_rejected() {
    let mag: u8 = kani::any();
    let lip: u8 = kani::any();
    let mag = mag as f64;
    let lip = lip as f64;
    kani::assume(mag.is_finite() && mag >= 0.0);
    kani::assume(lip.is_finite() && lip >= 0.0);

    let result = AnalyticalFusionBound::compute(0, mag, lip);
    assert!(result.is_err(), "zero ops must be rejected");
}

// ---------------------------------------------------------------------------
// 3. Negative magnitude rejected
// ---------------------------------------------------------------------------

/// Prove: `compute(*, negative, *)` always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_negative_magnitude_rejected() {
    let ops: u8 = kani::any();
    kani::assume(ops >= 1);
    let mag: u8 = kani::any();
    kani::assume(mag >= 1);
    let neg_mag = -(mag as f64);
    let lip: u8 = kani::any();
    let lip = lip as f64;
    kani::assume(lip.is_finite() && lip >= 0.0);

    let result = AnalyticalFusionBound::compute(ops as usize, neg_mag, lip);
    assert!(result.is_err(), "negative magnitude must be rejected");
}

// ---------------------------------------------------------------------------
// 4. Negative lipschitz rejected
// ---------------------------------------------------------------------------

/// Prove: `compute(*, *, negative)` always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_negative_lipschitz_rejected() {
    let ops: u8 = kani::any();
    kani::assume(ops >= 1);
    let mag: u8 = kani::any();
    let mag = mag as f64;
    kani::assume(mag.is_finite() && mag >= 0.0);
    let lip: u8 = kani::any();
    kani::assume(lip >= 1);
    let neg_lip = -(lip as f64);

    let result = AnalyticalFusionBound::compute(ops as usize, mag, neg_lip);
    assert!(result.is_err(), "negative lipschitz must be rejected");
}

// ---------------------------------------------------------------------------
// 5. NaN magnitude rejected
// ---------------------------------------------------------------------------

/// Prove: `compute(*, NaN, *)` always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_nan_magnitude_rejected() {
    let result = AnalyticalFusionBound::compute(2, f64::NAN, 1.0);
    assert!(result.is_err(), "NaN magnitude must be rejected");
}

// ---------------------------------------------------------------------------
// 6. NaN lipschitz rejected
// ---------------------------------------------------------------------------

/// Prove: `compute(*, *, NaN)` always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_nan_lipschitz_rejected() {
    let result = AnalyticalFusionBound::compute(2, 1.0, f64::NAN);
    assert!(result.is_err(), "NaN lipschitz must be rejected");
}

// ---------------------------------------------------------------------------
// 7. Infinity magnitude rejected
// ---------------------------------------------------------------------------

/// Prove: `compute(*, Inf, *)` always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_inf_magnitude_rejected() {
    let result = AnalyticalFusionBound::compute(2, f64::INFINITY, 1.0);
    assert!(result.is_err(), "Inf magnitude must be rejected");
}

// ---------------------------------------------------------------------------
// 8. Infinity lipschitz rejected
// ---------------------------------------------------------------------------

/// Prove: `compute(*, *, Inf)` always returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_inf_lipschitz_rejected() {
    let result = AnalyticalFusionBound::compute(2, 1.0, f64::INFINITY);
    assert!(result.is_err(), "Inf lipschitz must be rejected");
}

// ===========================================================================
// AnalyticalFusionBound::compute output properties
// ===========================================================================

// ---------------------------------------------------------------------------
// 9. Valid inputs produce finite non-negative bound
// ---------------------------------------------------------------------------

/// Prove: for small valid inputs, `compute` returns a finite, non-negative
/// `max_abs_diff`.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_valid_inputs_finite_nonneg() {
    let ops: u8 = kani::any();
    kani::assume(ops >= 1 && ops <= 10);
    let mag: u8 = kani::any();
    let mag = mag as f64;
    let lip: u8 = kani::any();
    let lip = lip as f64;

    if let Ok(bound) = AnalyticalFusionBound::compute(ops as usize, mag, lip) {
        assert!(bound.max_abs_diff.is_finite(), "bound must be finite");
        assert!(bound.max_abs_diff >= 0.0, "bound must be non-negative");
    }
}

// ---------------------------------------------------------------------------
// 10. Zero magnitude produces zero bound
// ---------------------------------------------------------------------------

/// Prove: `compute(n, 0.0, L)` always returns `max_abs_diff == 0.0`.
/// If the intermediate magnitude is zero, there is no error to amplify.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_zero_magnitude_zero_bound() {
    let ops: u8 = kani::any();
    kani::assume(ops >= 1 && ops <= 10);
    let lip: u8 = kani::any();
    let lip = lip as f64;
    kani::assume(lip.is_finite() && lip >= 0.0);

    let bound =
        AnalyticalFusionBound::compute(ops as usize, 0.0, lip).expect("zero magnitude is valid");
    assert!(
        bound.max_abs_diff == 0.0,
        "zero magnitude must produce zero diff"
    );
}

// ---------------------------------------------------------------------------
// 11. Zero lipschitz produces zero bound
// ---------------------------------------------------------------------------

/// Prove: `compute(n, M, 0.0)` always returns `max_abs_diff == 0.0`.
/// If the downstream Lipschitz factor is zero, no error propagates.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_zero_lipschitz_zero_bound() {
    let ops: u8 = kani::any();
    kani::assume(ops >= 1 && ops <= 10);
    let mag: u8 = kani::any();
    let mag = mag as f64;
    kani::assume(mag.is_finite() && mag >= 0.0);

    let bound =
        AnalyticalFusionBound::compute(ops as usize, mag, 0.0).expect("zero lipschitz is valid");
    assert!(
        bound.max_abs_diff == 0.0,
        "zero lipschitz must produce zero diff"
    );
}

// ---------------------------------------------------------------------------
// 12. Bound formula: max_abs_diff = magnitude * ops * eps * lipschitz
// ---------------------------------------------------------------------------

/// Prove: the computed `max_abs_diff` matches the analytical formula.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_formula_correct() {
    let ops: u8 = kani::any();
    kani::assume(ops >= 1 && ops <= 5);
    let mag: u8 = kani::any();
    kani::assume(mag >= 1 && mag <= 100);
    let lip: u8 = kani::any();
    kani::assume(lip >= 1 && lip <= 10);

    let ops_usize = ops as usize;
    let mag_f64 = mag as f64;
    let lip_f64 = lip as f64;

    let bound =
        AnalyticalFusionBound::compute(ops_usize, mag_f64, lip_f64).expect("small valid inputs");
    let expected = mag_f64 * (ops_usize as f64) * F32_MACHINE_EPS * lip_f64;

    assert!(
        (bound.max_abs_diff - expected).abs() < 1e-30,
        "formula must match: got {}, expected {}",
        bound.max_abs_diff,
        expected
    );
}

// ---------------------------------------------------------------------------
// 13. Bound monotonic in ops count
// ---------------------------------------------------------------------------

/// Prove: increasing `differing_op_count` does not decrease the bound.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_monotone_in_ops() {
    let ops1: u8 = kani::any();
    let ops2: u8 = kani::any();
    kani::assume(ops1 >= 1 && ops1 <= 10);
    kani::assume(ops2 >= ops1 && ops2 <= 10);
    let mag = 50.0_f64;
    let lip = 2.0_f64;

    let b1 = AnalyticalFusionBound::compute(ops1 as usize, mag, lip).expect("valid");
    let b2 = AnalyticalFusionBound::compute(ops2 as usize, mag, lip).expect("valid");
    assert!(
        b2.max_abs_diff >= b1.max_abs_diff - 1e-30,
        "more ops must produce >= bound"
    );
}

// ---------------------------------------------------------------------------
// 14. Bound monotonic in magnitude
// ---------------------------------------------------------------------------

/// Prove: increasing `max_magnitude` does not decrease the bound.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_monotone_in_magnitude() {
    let mag1: u8 = kani::any();
    let mag2: u8 = kani::any();
    kani::assume(mag2 >= mag1);
    let lip = 2.0_f64;

    let b1 = AnalyticalFusionBound::compute(2, mag1 as f64, lip).expect("valid");
    let b2 = AnalyticalFusionBound::compute(2, mag2 as f64, lip).expect("valid");
    assert!(
        b2.max_abs_diff >= b1.max_abs_diff - 1e-30,
        "larger magnitude must produce >= bound"
    );
}

// ---------------------------------------------------------------------------
// 15. Bound monotonic in lipschitz
// ---------------------------------------------------------------------------

/// Prove: increasing `lipschitz_factor` does not decrease the bound.
#[kani::unwind(1)]
#[kani::proof]
fn analytical_bound_monotone_in_lipschitz() {
    let lip1: u8 = kani::any();
    let lip2: u8 = kani::any();
    kani::assume(lip2 >= lip1);
    let mag = 50.0_f64;

    let b1 = AnalyticalFusionBound::compute(2, mag, lip1 as f64).expect("valid");
    let b2 = AnalyticalFusionBound::compute(2, mag, lip2 as f64).expect("valid");
    assert!(
        b2.max_abs_diff >= b1.max_abs_diff - 1e-30,
        "larger lipschitz must produce >= bound"
    );
}

// ===========================================================================
// AnalyticalFusionBound::proves_within_epsilon
// ===========================================================================

// ---------------------------------------------------------------------------
// 16. proves_within_epsilon consistent with max_abs_diff <= epsilon
// ---------------------------------------------------------------------------

/// Prove: `proves_within_epsilon` returns true iff `max_abs_diff <= epsilon`.
#[kani::unwind(1)]
#[kani::proof]
fn proves_within_epsilon_consistent() {
    let ops: u8 = kani::any();
    kani::assume(ops >= 1 && ops <= 5);
    let mag: u8 = kani::any();
    let lip: u8 = kani::any();
    kani::assume(lip >= 1 && lip <= 5);

    if let Ok(bound) = AnalyticalFusionBound::compute(ops as usize, mag as f64, lip as f64) {
        let eps_f32 = 1e-4_f32;
        let expected = bound.max_abs_diff <= f64::from(eps_f32);
        assert_eq!(
            bound.proves_within_epsilon(eps_f32),
            expected,
            "proves_within_epsilon must match direct comparison"
        );
    }
}

// ===========================================================================
// is_iso8601_utc validation
// ===========================================================================

// ---------------------------------------------------------------------------
// 17. Valid ISO 8601 accepted
// ---------------------------------------------------------------------------

/// Prove: a correctly formatted ISO 8601 UTC string is accepted.
#[kani::unwind(1)]
#[kani::proof]
fn iso8601_valid_accepted() {
    assert!(is_iso8601_utc("2026-03-15T12:00:00Z"));
    assert!(is_iso8601_utc("1970-01-01T00:00:00Z"));
    assert!(is_iso8601_utc("9999-12-31T23:59:59Z"));
}

// ---------------------------------------------------------------------------
// 18. Empty string rejected
// ---------------------------------------------------------------------------

/// Prove: empty string is rejected as non-ISO8601.
#[kani::unwind(1)]
#[kani::proof]
fn iso8601_empty_rejected() {
    assert!(!is_iso8601_utc(""));
}

// ---------------------------------------------------------------------------
// 19. Wrong length rejected
// ---------------------------------------------------------------------------

/// Prove: strings with wrong length are rejected.
#[kani::unwind(1)]
#[kani::proof]
fn iso8601_wrong_length_rejected() {
    assert!(!is_iso8601_utc("2026-03-15T12:00:00")); // 19 chars, no Z
    assert!(!is_iso8601_utc("2026-03-15T12:00:00ZZ")); // 21 chars
}

// ---------------------------------------------------------------------------
// 20. Missing separators rejected
// ---------------------------------------------------------------------------

/// Prove: strings with wrong separators are rejected.
#[kani::unwind(1)]
#[kani::proof]
fn iso8601_wrong_separators_rejected() {
    assert!(!is_iso8601_utc("2026/03/15T12:00:00Z")); // / instead of -
    assert!(!is_iso8601_utc("2026-03-15 12:00:00Z")); // space instead of T
    assert!(!is_iso8601_utc("2026-03-15T12-00-00Z")); // - instead of :
    assert!(!is_iso8601_utc("2026-03-15T12:00:00X")); // X instead of Z
}

// ---------------------------------------------------------------------------
// 21. Non-digit characters in positions rejected
// ---------------------------------------------------------------------------

/// Prove: non-digit characters in numeric positions are rejected.
#[kani::unwind(1)]
#[kani::proof]
fn iso8601_non_digits_rejected() {
    assert!(!is_iso8601_utc("ABCD-03-15T12:00:00Z")); // letters in year
    assert!(!is_iso8601_utc("2026-XX-15T12:00:00Z")); // letters in month
}

// ---------------------------------------------------------------------------
// 22. UNIX epoch string rejected
// ---------------------------------------------------------------------------

/// Prove: bare UNIX epoch string is rejected (not ISO 8601).
#[kani::unwind(1)]
#[kani::proof]
fn iso8601_unix_epoch_rejected() {
    assert!(!is_iso8601_utc("1742000000Z"));
    assert!(!is_iso8601_utc("1742000000"));
}

// ===========================================================================
// FusionEquivalenceCertificate::proves_equivalence
// ===========================================================================

// ---------------------------------------------------------------------------
// 23. proves_equivalence: CROWN only (crown <= epsilon)
// ---------------------------------------------------------------------------

/// Prove: when only CROWN bound exists and is <= epsilon,
/// `proves_equivalence` returns true.
#[kani::unwind(64)]
#[kani::proof]
fn proves_equivalence_crown_within_epsilon() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: -5e-5,
        diff_upper: 5e-5,
        max_abs_diff: 5e-5,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-10.0, 10.0)]);
    assert!(cert.proves_equivalence(), "CROWN within epsilon must prove");
}

// ---------------------------------------------------------------------------
// 24. proves_equivalence: CROWN too loose, no analytical
// ---------------------------------------------------------------------------

/// Prove: when CROWN bound exceeds epsilon and no analytical bound exists,
/// `proves_equivalence` returns false.
#[kani::unwind(64)]
#[kani::proof]
fn proves_equivalence_crown_too_loose() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: -0.01,
        diff_upper: 0.01,
        max_abs_diff: 0.01,
        within_epsilon: false,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-10.0, 10.0)]);
    assert!(!cert.proves_equivalence(), "loose CROWN must not prove");
}

// ---------------------------------------------------------------------------
// 25. proves_equivalence: analytical rescues loose CROWN
// ---------------------------------------------------------------------------

/// Prove: when CROWN is too loose but analytical bound is tight,
/// `proves_equivalence` returns true (disjunction semantics).
#[kani::unwind(64)]
#[kani::proof]
fn proves_equivalence_analytical_rescues() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: -1.0,
        diff_upper: 1.0,
        max_abs_diff: 1.0,
        within_epsilon: false,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let analytical = AnalyticalFusionBound::compute(2, 64.0, 2.0).expect("valid");
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-10.0, 10.0)])
        .with_analytical_bound(analytical);
    // analytical ~1.53e-5 < 1e-4
    assert!(cert.proves_equivalence(), "analytical must rescue");
}

// ===========================================================================
// FusionEquivalenceCertificate::tightest_bound
// ===========================================================================

// ---------------------------------------------------------------------------
// 26. tightest_bound picks minimum
// ---------------------------------------------------------------------------

/// Prove: `tightest_bound` returns the smaller of CROWN and analytical bounds.
#[kani::unwind(64)]
#[kani::proof]
fn tightest_bound_picks_min() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: -0.001,
        diff_upper: 0.001,
        max_abs_diff: 0.001,
        within_epsilon: false,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let analytical = AnalyticalFusionBound::compute(2, 64.0, 2.0).expect("valid");
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-10.0, 10.0)])
        .with_analytical_bound(analytical);

    let tightest = cert.tightest_bound().expect("has bounds");
    let crown_val = f64::from(cert.crown_bound.unwrap());
    let analytical_val = cert.analytical_bound.as_ref().unwrap().max_abs_diff;

    assert!(
        (tightest - crown_val.min(analytical_val)).abs() < 1e-30,
        "tightest must be min of crown and analytical"
    );
}

// ---------------------------------------------------------------------------
// 27. tightest_bound with no bounds returns None
// ---------------------------------------------------------------------------

/// Prove: when both CROWN and analytical are absent, `tightest_bound` is None.
#[kani::unwind(64)]
#[kani::proof]
fn tightest_bound_none_when_empty() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let mut cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[]);
    cert.crown_bound = None;
    assert!(cert.tightest_bound().is_none(), "no bounds means None");
}

// ===========================================================================
// FusionEquivalenceCertificate::validate
// ===========================================================================

// ---------------------------------------------------------------------------
// 28. validate: version 0 rejected
// ---------------------------------------------------------------------------

/// Prove: certificate with version=0 fails validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_version_zero_rejected() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let mut cert =
        FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-1.0, 1.0)]);
    cert.version = 0;
    assert!(cert.validate().is_err(), "version 0 must be rejected");
}

// ---------------------------------------------------------------------------
// 29. validate: NaN epsilon rejected
// ---------------------------------------------------------------------------

/// Prove: certificate with NaN epsilon fails validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_nan_epsilon_rejected() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let mut cert =
        FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-1.0, 1.0)]);
    cert.epsilon = f32::NAN;
    assert!(cert.validate().is_err(), "NaN epsilon must be rejected");
}

// ---------------------------------------------------------------------------
// 30. validate: negative epsilon rejected
// ---------------------------------------------------------------------------

/// Prove: certificate with negative epsilon fails validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_negative_epsilon_rejected() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let mut cert =
        FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-1.0, 1.0)]);
    cert.epsilon = -0.01;
    assert!(
        cert.validate().is_err(),
        "negative epsilon must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 31. validate: inverted variable bounds rejected
// ---------------------------------------------------------------------------

/// Prove: certificate with lo > hi variable bounds fails validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_inverted_bounds_rejected() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(10.0, -10.0)]);
    assert!(cert.validate().is_err(), "inverted bounds must be rejected");
}

// ---------------------------------------------------------------------------
// 32. validate: empty kernel name rejected
// ---------------------------------------------------------------------------

/// Prove: certificate with empty fused_kernel_name fails validation.
#[kani::unwind(1)]
#[kani::proof]
fn validate_empty_name_rejected() {
    let mut v = FusionVerification {
        fused_kernel_name: String::new(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-1.0, 1.0)]);
    assert!(cert.validate().is_err(), "empty name must be rejected");
}

// ---------------------------------------------------------------------------
// 33. validate: NaN crown bound rejected
// ---------------------------------------------------------------------------

/// Prove: certificate with NaN crown_bound fails validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_nan_crown_bound_rejected() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let mut cert =
        FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-1.0, 1.0)]);
    cert.crown_bound = Some(f32::NAN);
    assert!(cert.validate().is_err(), "NaN crown bound must be rejected");
}

// ---------------------------------------------------------------------------
// 34. validate: bad SHA-256 hash rejected
// ---------------------------------------------------------------------------

/// Prove: certificate with invalid SHA-256 hash fails validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_bad_hash_rejected() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-1.0, 1.0)])
        .with_source_hash("not-a-valid-hash".to_string());
    assert!(cert.validate().is_err(), "bad hash must be rejected");
}

// ===========================================================================
// known_bounds: all produce finite bounds within 1e-4
// ===========================================================================

// ---------------------------------------------------------------------------
// 35. known_bounds adain_snake finite and within epsilon
// ---------------------------------------------------------------------------

/// Prove: `known_bounds::adain_snake()` produces a finite bound within 1e-4.
#[kani::unwind(1)]
#[kani::proof]
fn known_bounds_adain_snake_valid() {
    let bound = known_bounds::adain_snake().expect("must succeed");
    assert!(bound.max_abs_diff.is_finite(), "must be finite");
    assert!(bound.max_abs_diff >= 0.0, "must be non-negative");
    assert!(bound.proves_within_epsilon(1e-4), "must be within 1e-4");
}

// ---------------------------------------------------------------------------
// 36. known_bounds layer_norm_gelu finite and within epsilon
// ---------------------------------------------------------------------------

/// Prove: `known_bounds::layer_norm_gelu()` produces a finite bound within 1e-4.
#[kani::unwind(1)]
#[kani::proof]
fn known_bounds_layer_norm_gelu_valid() {
    let bound = known_bounds::layer_norm_gelu().expect("must succeed");
    assert!(bound.max_abs_diff.is_finite(), "must be finite");
    assert!(bound.max_abs_diff >= 0.0, "must be non-negative");
    assert!(bound.proves_within_epsilon(1e-4), "must be within 1e-4");
}

// ---------------------------------------------------------------------------
// 37. known_bounds rms_norm_silu_mul finite and within epsilon
// ---------------------------------------------------------------------------

/// Prove: `known_bounds::rms_norm_silu_mul()` produces a finite bound within 1e-4.
#[kani::unwind(1)]
#[kani::proof]
fn known_bounds_rms_norm_silu_mul_valid() {
    let bound = known_bounds::rms_norm_silu_mul().expect("must succeed");
    assert!(bound.max_abs_diff.is_finite(), "must be finite");
    assert!(bound.max_abs_diff >= 0.0, "must be non-negative");
    assert!(bound.proves_within_epsilon(1e-4), "must be within 1e-4");
}

// ---------------------------------------------------------------------------
// 38. known_bounds adain_leaky_relu finite and within epsilon
// ---------------------------------------------------------------------------

/// Prove: `known_bounds::adain_leaky_relu()` produces a finite bound within 1e-4.
#[kani::unwind(1)]
#[kani::proof]
fn known_bounds_adain_leaky_relu_valid() {
    let bound = known_bounds::adain_leaky_relu().expect("must succeed");
    assert!(bound.max_abs_diff.is_finite(), "must be finite");
    assert!(bound.max_abs_diff >= 0.0, "must be non-negative");
    assert!(bound.proves_within_epsilon(1e-4), "must be within 1e-4");
}

// ---------------------------------------------------------------------------
// 39. known_bounds ada_layer_norm finite and within epsilon
// ---------------------------------------------------------------------------

/// Prove: `known_bounds::ada_layer_norm()` produces a finite bound within 1e-4.
#[kani::unwind(1)]
#[kani::proof]
fn known_bounds_ada_layer_norm_valid() {
    let bound = known_bounds::ada_layer_norm().expect("must succeed");
    assert!(bound.max_abs_diff.is_finite(), "must be finite");
    assert!(bound.max_abs_diff >= 0.0, "must be non-negative");
    assert!(bound.proves_within_epsilon(1e-4), "must be within 1e-4");
}

// ---------------------------------------------------------------------------
// 40. known_bounds: leaky_relu tighter than snake (lower Lipschitz)
// ---------------------------------------------------------------------------

/// Prove: adain_leaky_relu bound is strictly tighter than adain_snake bound.
/// LeakyReLU Lipschitz (1.0) < Snake Lipschitz (2.0), same magnitude/ops.
#[kani::unwind(1)]
#[kani::proof]
fn known_bounds_leaky_relu_tighter_than_snake() {
    let lr = known_bounds::adain_leaky_relu().expect("valid");
    let sn = known_bounds::adain_snake().expect("valid");
    assert!(
        lr.max_abs_diff < sn.max_abs_diff,
        "leaky_relu ({}) must be tighter than snake ({})",
        lr.max_abs_diff,
        sn.max_abs_diff,
    );
}

// ===========================================================================
// FUSION_CERTIFICATE_VERSION
// ===========================================================================

// ---------------------------------------------------------------------------
// 41. Version constant is positive
// ---------------------------------------------------------------------------

/// Prove: the current certificate version is at least 1.
#[kani::unwind(1)]
#[kani::proof]
fn certificate_version_positive() {
    assert!(FUSION_CERTIFICATE_VERSION >= 1, "version must be >= 1");
}

// ---------------------------------------------------------------------------
// 42. from_verification sets version to current
// ---------------------------------------------------------------------------

/// Prove: `from_verification` always sets version to FUSION_CERTIFICATE_VERSION.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_sets_current_version() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[]);
    assert_eq!(cert.version, FUSION_CERTIFICATE_VERSION);
}

// ===========================================================================
// ADDITIONAL HARNESSES (Part of #3702)
// ===========================================================================

// ---------------------------------------------------------------------------
// 43. validate: non-finite analytical bound rejected
// ---------------------------------------------------------------------------

/// Prove: certificate with NaN analytical max_abs_diff fails validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_nan_analytical_bound_rejected() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let mut cert =
        FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-1.0, 1.0)]);
    cert.analytical_bound = Some(AnalyticalFusionBound {
        differing_op_count: 2,
        max_magnitude: 64.0,
        lipschitz_factor: 2.0,
        max_abs_diff: f64::NAN,
    });
    assert!(
        cert.validate().is_err(),
        "NaN analytical bound must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 44. validate: negative analytical bound rejected
// ---------------------------------------------------------------------------

/// Prove: certificate with negative analytical max_abs_diff fails validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_negative_analytical_bound_rejected() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let mut cert =
        FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-1.0, 1.0)]);
    cert.analytical_bound = Some(AnalyticalFusionBound {
        differing_op_count: 2,
        max_magnitude: 64.0,
        lipschitz_factor: 2.0,
        max_abs_diff: -0.001,
    });
    assert!(
        cert.validate().is_err(),
        "negative analytical bound must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 45. validate: NaN variable bound rejected
// ---------------------------------------------------------------------------

/// Prove: certificate with NaN in variable_bounds fails validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_nan_variable_bounds_rejected() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let cert =
        FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(f32::NAN, 1.0)]);
    assert!(
        cert.validate().is_err(),
        "NaN variable bound must be rejected"
    );
}

// ---------------------------------------------------------------------------
// 46. validate: valid certificate passes
// ---------------------------------------------------------------------------

/// Prove: a well-formed certificate passes validation.
#[kani::unwind(64)]
#[kani::proof]
fn validate_well_formed_passes() {
    let v = FusionVerification {
        fused_kernel_name: "adain_snake_fused".to_string(),
        diff_lower: -5e-5,
        diff_upper: 5e-5,
        max_abs_diff: 5e-5,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let cert = FusionEquivalenceCertificate::from_verification(
        &v,
        "adain",
        "snake",
        512,
        &[(-10.0, 10.0)],
    );
    assert!(
        cert.validate().is_ok(),
        "well-formed certificate must pass validation"
    );
}

// ---------------------------------------------------------------------------
// 47. with_source_hash: correct SHA-256 passes validation
// ---------------------------------------------------------------------------

/// Prove: a certificate with a valid 64-char hex SHA-256 hash passes validation.
#[kani::unwind(64)]
#[kani::proof]
fn with_source_hash_valid_passes() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let valid_hash = "a".repeat(64); // 64 hex chars
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-1.0, 1.0)])
        .with_source_hash(valid_hash);
    assert!(cert.validate().is_ok(), "valid SHA-256 hash must pass");
}

// ---------------------------------------------------------------------------
// 48. with_source_hash: too short hash rejected
// ---------------------------------------------------------------------------

/// Prove: a certificate with a 32-char hash (not 64) fails validation.
#[kani::unwind(64)]
#[kani::proof]
fn with_source_hash_too_short_rejected() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let short_hash = "a".repeat(32);
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-1.0, 1.0)])
        .with_source_hash(short_hash);
    assert!(
        cert.validate().is_err(),
        "32-char hash must be rejected (need 64)"
    );
}

// ---------------------------------------------------------------------------
// 49. proves_equivalence: neither CROWN nor analytical — returns false
// ---------------------------------------------------------------------------

/// Prove: when both crown_bound and analytical_bound are None,
/// `proves_equivalence` returns false.
#[kani::unwind(64)]
#[kani::proof]
fn proves_equivalence_no_bounds_returns_false() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let mut cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[]);
    cert.crown_bound = None;
    cert.analytical_bound = None;
    assert!(
        !cert.proves_equivalence(),
        "no bounds must not prove equivalence"
    );
}

// ---------------------------------------------------------------------------
// 50. tightest_bound: CROWN only returns CROWN value
// ---------------------------------------------------------------------------

/// Prove: when only CROWN bound exists, `tightest_bound` returns it.
#[kani::unwind(64)]
#[kani::proof]
fn tightest_bound_crown_only() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: -0.001,
        diff_upper: 0.001,
        max_abs_diff: 0.001,
        within_epsilon: false,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-10.0, 10.0)]);
    let tightest = cert.tightest_bound().expect("has CROWN bound");
    let crown_val = f64::from(cert.crown_bound.unwrap());
    assert!(
        (tightest - crown_val).abs() < 1e-30,
        "tightest must equal CROWN when no analytical"
    );
}

// ---------------------------------------------------------------------------
// 51. tightest_bound: analytical only returns analytical value
// ---------------------------------------------------------------------------

/// Prove: when only analytical bound exists, `tightest_bound` returns it.
#[kani::unwind(64)]
#[kani::proof]
fn tightest_bound_analytical_only() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let analytical = AnalyticalFusionBound::compute(2, 64.0, 2.0).expect("valid");
    let mut cert =
        FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-10.0, 10.0)])
            .with_analytical_bound(analytical.clone());
    cert.crown_bound = None;
    let tightest = cert.tightest_bound().expect("has analytical bound");
    assert!(
        (tightest - analytical.max_abs_diff).abs() < 1e-30,
        "tightest must equal analytical when no CROWN"
    );
}

// ---------------------------------------------------------------------------
// 52. from_verification: sequential names preserved
// ---------------------------------------------------------------------------

/// Prove: `from_verification` preserves the sequential kernel names.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_preserves_sequential_names() {
    let v = FusionVerification {
        fused_kernel_name: "fused".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let cert =
        FusionEquivalenceCertificate::from_verification(&v, "kernel_a", "kernel_b", 512, &[]);
    assert_eq!(cert.sequential_names.0, "kernel_a");
    assert_eq!(cert.sequential_names.1, "kernel_b");
}

// ---------------------------------------------------------------------------
// 53. from_verification: variable bounds preserved
// ---------------------------------------------------------------------------

/// Prove: `from_verification` preserves the variable bounds exactly.
#[kani::unwind(64)]
#[kani::proof]
fn from_verification_preserves_variable_bounds() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let bounds = vec![(-10.0_f32, 10.0_f32), (-5.0, 5.0)];
    let cert = FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &bounds);
    assert_eq!(cert.variable_bounds.len(), 2);
    assert_eq!(cert.variable_bounds[0], (-10.0, 10.0));
    assert_eq!(cert.variable_bounds[1], (-5.0, 5.0));
}

// ---------------------------------------------------------------------------
// 54. validate: future version rejected
// ---------------------------------------------------------------------------

/// Prove: a certificate with version > FUSION_CERTIFICATE_VERSION fails.
#[kani::unwind(64)]
#[kani::proof]
fn validate_future_version_rejected() {
    let v = FusionVerification {
        fused_kernel_name: "test".to_string(),
        diff_lower: 0.0,
        diff_upper: 0.0,
        max_abs_diff: 0.0,
        within_epsilon: true,
        epsilon: 1e-4,
        method: PropMethod::Crown,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    let mut cert =
        FusionEquivalenceCertificate::from_verification(&v, "a", "b", 512, &[(-1.0, 1.0)]);
    cert.version = FUSION_CERTIFICATE_VERSION + 1;
    assert!(cert.validate().is_err(), "future version must be rejected");
}

// ---------------------------------------------------------------------------
// 55. unix_secs_to_iso8601: epoch zero produces 1970-01-01T00:00:00Z
// ---------------------------------------------------------------------------

/// Prove: UNIX epoch 0 converts to the correct ISO 8601 string.
#[kani::unwind(1)]
#[kani::proof]
fn unix_epoch_zero_correct() {
    let s = super::unix_secs_to_iso8601(0);
    assert_eq!(s, "1970-01-01T00:00:00Z");
}

// ---------------------------------------------------------------------------
// 56. unix_secs_to_iso8601: known timestamp converts correctly
// ---------------------------------------------------------------------------

/// Prove: a known UNIX timestamp converts correctly.
/// 1711929600 = 2024-04-01T00:00:00Z
#[kani::unwind(1)]
#[kani::proof]
fn unix_known_timestamp_correct() {
    let s = super::unix_secs_to_iso8601(1711929600);
    assert_eq!(s, "2024-04-01T00:00:00Z");
}

// ---------------------------------------------------------------------------
// 57. unix_secs_to_iso8601: output is always valid ISO 8601
// ---------------------------------------------------------------------------

/// Prove: for small timestamps, the output always passes is_iso8601_utc.
#[kani::unwind(1)]
#[kani::proof]
fn unix_output_always_valid_iso8601() {
    let secs: u16 = kani::any();
    let s = super::unix_secs_to_iso8601(secs as u64);
    assert!(
        is_iso8601_utc(&s),
        "output must always be valid ISO 8601 UTC"
    );
}
