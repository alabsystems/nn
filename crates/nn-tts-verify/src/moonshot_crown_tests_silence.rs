// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! P1 (non-silence) and P2 (non-clipping) property tests.

use super::*;

#[test]
fn test_non_silence_proven() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.8; 8], true);
    let result = check_non_silence(&cert, 0.01);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.bound_value > 0.01);
    assert!(result.explanation.contains("PROVEN"));
}

#[test]
fn test_non_silence_fails_near_zero() {
    let cert = bounded_pipeline(vec![-0.005; 8], vec![0.005; 8], true);
    let result = check_non_silence(&cert, 0.01);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
    assert!(result.bound_value < 0.01);
}

#[test]
fn test_non_silence_ibp_fallback() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.8; 8], false);
    let result = check_non_silence(&cert, 0.01);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

#[test]
fn test_non_clipping_proven() {
    let cert = bounded_pipeline(vec![-0.8; 8], vec![0.9; 8], true);
    let result = check_non_clipping(&cert);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.bound_value <= 1.0);
}

#[test]
fn test_non_clipping_fails_exceeds_range() {
    let cert = bounded_pipeline(vec![-1.2; 8], vec![1.5; 8], true);
    let result = check_non_clipping(&cert);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
    assert!(result.explanation.contains("NOT PROVEN"));
}

#[test]
fn test_non_clipping_exactly_at_boundary() {
    let cert = bounded_pipeline(vec![-1.0; 8], vec![1.0; 8], true);
    let result = check_non_clipping(&cert);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownProven);
}

#[test]
fn test_non_clipping_ibp_fallback() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], false);
    let result = check_non_clipping(&cert);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

/// NaN in output bounds must not produce a "proven" non-clipping result.
///
/// P1-234: Without the finiteness guard, f64::max(NEG_INFINITY, NaN) returns
/// NEG_INFINITY, so max_upper stays at NEG_INFINITY (≤ 1.0), and
/// f64::min(INFINITY, NaN) returns INFINITY (≥ -1.0), falsely proving
/// the output is within [-1, 1].
#[test]
fn test_non_clipping_nan_bounds_not_proven() {
    let mut upper = vec![0.5; 8];
    upper[2] = f64::NAN;
    let cert = bounded_pipeline(vec![-0.5; 8], upper, true);
    let result = check_non_clipping(&cert);
    assert!(
        !result.proven,
        "NaN in output bounds must not prove non-clipping"
    );
    assert_eq!(result.level, VerificationLevel::Empirical);
}
