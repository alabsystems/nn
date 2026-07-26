// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_precision_tier_parse() {
    assert_eq!(
        PrecisionTier::parse("strict").expect("strict should parse"),
        PrecisionTier::Strict
    );
    assert_eq!(
        PrecisionTier::parse("normal").expect("normal should parse"),
        PrecisionTier::Normal
    );
    assert_eq!(
        PrecisionTier::parse("relaxed").expect("relaxed should parse"),
        PrecisionTier::Relaxed
    );
}

#[test]
fn test_precision_tier_parse_rejects_unknown() {
    let err = PrecisionTier::parse("fast").expect_err("unknown precision should fail");
    assert_eq!(err, PrecisionParseError::Unsupported("fast".to_string()));
}

#[test]
fn test_bootstrap_budget_relaxed_is_10x_normal() {
    let (normal_abs, _) = bootstrap_budget(ScalarType::F32, PrecisionTier::Normal);
    let (relaxed_abs, _) = bootstrap_budget(ScalarType::F32, PrecisionTier::Relaxed);
    assert!((relaxed_abs - (normal_abs * 10.0)).abs() < f32::EPSILON);
}

#[test]
fn test_within_differential_budget_passing() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    assert!(within_differential_budget(10.0, 10.0001, contract));
}

#[test]
fn test_within_differential_budget_failing() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    // Strict f32 abs budget is 1e-6 and rel is 1e-6, so tolerance at ref=1.0
    // is ~2e-6. A difference of 0.01 must exceed that.
    assert!(!within_differential_budget(1.0, 1.01, contract));
}

#[test]
fn test_within_differential_budget_exact_match() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    assert!(within_differential_budget(42.0, 42.0, contract));
}

#[test]
fn test_precision_tier_as_str() {
    assert_eq!(PrecisionTier::Strict.as_str(), "strict");
    assert_eq!(PrecisionTier::Normal.as_str(), "normal");
    assert_eq!(PrecisionTier::Relaxed.as_str(), "relaxed");
}

#[test]
fn test_precision_tier_fast_math() {
    assert!(!PrecisionTier::Strict.fast_math());
    assert!(!PrecisionTier::Normal.fast_math());
    assert!(PrecisionTier::Relaxed.fast_math());
}

#[test]
fn test_bootstrap_budget_f16_all_tiers() {
    let (strict_abs, _) = bootstrap_budget(ScalarType::F16, PrecisionTier::Strict);
    let (normal_abs, _) = bootstrap_budget(ScalarType::F16, PrecisionTier::Normal);
    let (relaxed_abs, _) = bootstrap_budget(ScalarType::F16, PrecisionTier::Relaxed);
    // f16 strict < f16 normal < f16 relaxed
    assert!(
        strict_abs < normal_abs,
        "f16 strict should be tighter than normal"
    );
    assert!(
        normal_abs < relaxed_abs,
        "f16 normal should be tighter than relaxed"
    );
}

#[test]
fn test_bootstrap_budget_f32_strict_tightest() {
    let (strict_abs, _) = bootstrap_budget(ScalarType::F32, PrecisionTier::Strict);
    let (normal_abs, _) = bootstrap_budget(ScalarType::F32, PrecisionTier::Normal);
    assert!(
        strict_abs < normal_abs,
        "f32 strict should be tighter than normal"
    );
}

#[test]
fn test_differential_tolerance_grows_with_reference() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let tol_small = differential_tolerance(1.0, contract);
    let tol_large = differential_tolerance(1000.0, contract);
    assert!(
        tol_large > tol_small,
        "tolerance should grow with reference magnitude"
    );
}

#[test]
fn test_precision_contract_fields() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Relaxed, ScalarType::F32);
    assert_eq!(contract.tier, PrecisionTier::Relaxed);
    assert!(contract.fast_math);
    assert!(contract.differential_abs_budget > 0.0);
    assert!(contract.differential_rel_budget > 0.0);
}

#[test]
fn test_input_bound_parse_basic() {
    let b = InputBound::parse("-1e4..1e4").expect("should parse");
    assert!((b.lo() - (-1e4)).abs() < f64::EPSILON);
    assert!((b.hi() - 1e4).abs() < f64::EPSILON);
}

#[test]
fn test_input_bound_parse_scientific() {
    let b = InputBound::parse("1e-8..1e3").expect("should parse");
    assert!((b.lo() - 1e-8).abs() < f64::EPSILON);
    assert!((b.hi() - 1e3).abs() < f64::EPSILON);
}

#[test]
fn test_input_bound_parse_negative() {
    let b = InputBound::parse("-100.5..200.0").expect("should parse");
    assert!((b.lo() - (-100.5)).abs() < f64::EPSILON);
    assert!((b.hi() - 200.0).abs() < f64::EPSILON);
}

#[test]
fn test_input_bound_parse_bad_format() {
    let err = InputBound::parse("1e4").expect_err("missing ..");
    assert!(matches!(err, InputBoundParseError::BadFormat(_)));
}

#[test]
fn test_input_bound_parse_bad_float() {
    let err = InputBound::parse("abc..1e4").expect_err("bad float");
    assert!(matches!(err, InputBoundParseError::BadFloat(_)));
}

#[test]
fn test_input_bound_parse_inverted() {
    let err = InputBound::parse("100.0..-100.0").expect_err("inverted");
    assert!(matches!(err, InputBoundParseError::Inverted { .. }));
}

#[test]
fn test_input_bounds_default_fallback() {
    let bounds = InputBounds::new();
    let b = bounds.get("x", ScalarType::F32);
    assert!((b.lo() - (-1e6)).abs() < f64::EPSILON);
    assert!((b.hi() - 1e6).abs() < f64::EPSILON);
}

#[test]
fn test_input_bounds_explicit_override() {
    let mut bounds = InputBounds::new();
    bounds.insert(
        "alpha",
        InputBound::new(1e-8, 1e3).expect("valid test bound"),
    );
    let b = bounds.get("alpha", ScalarType::F32);
    assert!((b.lo() - 1e-8).abs() < f64::EPSILON);
    assert!((b.hi() - 1e3).abs() < f64::EPSILON);
}

#[test]
fn test_input_bound_default_for_f16() {
    let b = InputBound::default_for(ScalarType::F16);
    assert!((b.lo() - (-65504.0)).abs() < f64::EPSILON);
    assert!((b.hi() - 65504.0).abs() < f64::EPSILON);
}

#[test]
fn test_within_differential_budget_both_nan() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    assert!(
        within_differential_budget(f32::NAN, f32::NAN, contract),
        "both NaN should match (same domain error)"
    );
}

#[test]
fn test_within_differential_budget_both_pos_inf() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    assert!(
        within_differential_budget(f32::INFINITY, f32::INFINITY, contract),
        "both +inf should match"
    );
}

#[test]
fn test_within_differential_budget_both_neg_inf() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    assert!(
        within_differential_budget(f32::NEG_INFINITY, f32::NEG_INFINITY, contract),
        "both -inf should match"
    );
}

#[test]
fn test_within_differential_budget_mixed_inf() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    assert!(
        !within_differential_budget(f32::INFINITY, f32::NEG_INFINITY, contract),
        "+inf vs -inf should NOT match"
    );
}

#[test]
fn test_within_differential_budget_nan_vs_finite() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    assert!(
        !within_differential_budget(f32::NAN, 0.0, contract),
        "NaN vs finite should NOT match"
    );
    assert!(
        !within_differential_budget(0.0, f32::NAN, contract),
        "finite vs NaN should NOT match"
    );
}

#[test]
fn test_within_differential_budget_inf_vs_finite() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    assert!(
        !within_differential_budget(f32::INFINITY, 1e30, contract),
        "+inf vs large finite should NOT match"
    );
}

#[test]
fn test_input_bound_new_validates_inputs() {
    // NaN rejected
    assert!(matches!(
        InputBound::new(f64::NAN, 1.0),
        Err(InputBoundParseError::NonFinite(_))
    ));
    assert!(matches!(
        InputBound::new(0.0, f64::NAN),
        Err(InputBoundParseError::NonFinite(_))
    ));
    // Infinity rejected
    assert!(matches!(
        InputBound::new(f64::NEG_INFINITY, 1.0),
        Err(InputBoundParseError::NonFinite(_))
    ));
    assert!(matches!(
        InputBound::new(0.0, f64::INFINITY),
        Err(InputBoundParseError::NonFinite(_))
    ));
    // Inverted rejected
    assert!(matches!(
        InputBound::new(100.0, -100.0),
        Err(InputBoundParseError::Inverted { .. })
    ));
    // Valid accepted
    let b = InputBound::new(-1.0, 1.0).expect("valid bound");
    assert!((b.lo() - (-1.0)).abs() < f64::EPSILON);
    assert!((b.hi() - 1.0).abs() < f64::EPSILON);
}
