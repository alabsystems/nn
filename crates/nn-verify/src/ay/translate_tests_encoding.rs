// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Numeric encoding tests: real_from_f64 (#161), constant param finiteness (#235).
//!
//! Adaptive denominator and param convention tests extracted to
//! `translate_tests_encoding_adaptive.rs`.

use super::*;

// --- real_from_f64 tests (AC1-AC3 of #161) ---

#[test]
fn test_real_from_f64_normal_integer() {
    // Integer-valued f64 should produce an integer Real constant.
    let expr = real_from_f64(42.0).expect("42.0 should encode");
    let smt2 = format!("{}", expr);
    assert_eq!(
        smt2, "42.0",
        "integer 42.0 should encode as Real constant 42.0"
    );
}

#[test]
fn test_real_from_f64_normal_fractional() {
    // Fractional value within safe range: 1.23456 → (/ 1234560 1000000)
    let expr = real_from_f64(1.23456).expect("1.23456 should encode");
    let smt2 = format!("{}", expr);
    assert!(
        smt2.contains("1234560") && smt2.contains("1000000"),
        "1.23456 should encode as (/ 1234560.0 1000000.0), got: {smt2}"
    );
}

#[test]
fn test_real_from_f64_negative_integer() {
    let expr = real_from_f64(-7.0).expect("-7.0 should encode");
    let smt2 = format!("{}", expr);
    assert_eq!(smt2, "-7.0", "negative integer should encode directly");
}

#[test]
fn test_real_from_f64_zero() {
    let expr = real_from_f64(0.0).expect("0.0 should encode");
    let smt2 = format!("{}", expr);
    assert_eq!(smt2, "0.0", "zero should encode as 0.0");
}

#[test]
fn test_real_from_f64_half() {
    // 1.5 → numer = (1.5 * 1e6).round() = 1500000, denom = 1000000
    let expr = real_from_f64(1.5).expect("1.5 should encode");
    let smt2 = format!("{}", expr);
    assert!(
        smt2.contains("1500000") && smt2.contains("1000000"),
        "1.5 should encode as (/ 1500000.0 1000000.0), got: {smt2}"
    );
}

#[test]
fn test_real_from_f64_negative_fractional() {
    // -2.5 → numer = (-2.5 * 1e6).round() = -2500000
    let expr = real_from_f64(-2.5).expect("-2.5 should encode");
    let smt2 = format!("{}", expr);
    assert!(
        smt2.contains("-2500000") && smt2.contains("1000000"),
        "-2.5 should encode with numer -2500000 / denom 1000000, got: {smt2}"
    );
}

#[test]
fn test_real_from_f64_nan_rejected() {
    let err = real_from_f64(f64::NAN).unwrap_err();
    assert!(
        matches!(err, SmtError::NonFiniteLiteral(v) if v.is_nan()),
        "NaN should produce NonFiniteLiteral, got: {err}"
    );
}

#[test]
fn test_real_from_f64_inf_rejected() {
    let err = real_from_f64(f64::INFINITY).unwrap_err();
    assert!(
        matches!(err, SmtError::NonFiniteLiteral(v) if v.is_infinite()),
        "Inf should produce NonFiniteLiteral, got: {err}"
    );
}

#[test]
fn test_real_from_f64_neg_inf_rejected() {
    let err = real_from_f64(f64::NEG_INFINITY).unwrap_err();
    assert!(
        matches!(err, SmtError::NonFiniteLiteral(v) if v.is_infinite()),
        "NEG_INFINITY should produce NonFiniteLiteral, got: {err}"
    );
}

#[test]
fn test_real_from_f64_large_fractional_overflow_rejected() {
    // AC3: A fractional value where val * 1_000_000 exceeds i64::MAX.
    // 1e13 + 0.5 is non-integer, so it takes the fractional path.
    // (1e13 + 0.5) * 1_000_000 = ~1e19, which exceeds i64::MAX (~9.22e18).
    let val = 1e13_f64 + 0.5;
    let err = real_from_f64(val).unwrap_err();
    assert!(
        matches!(err, SmtError::ValueTooLargeForRealEncoding(_)),
        "large fractional value should produce ValueTooLargeForRealEncoding, got: {err}"
    );
}

#[test]
fn test_real_from_f64_negative_large_fractional_overflow_rejected() {
    let val = -(1e13_f64 + 0.5);
    let err = real_from_f64(val).unwrap_err();
    assert!(
        matches!(err, SmtError::ValueTooLargeForRealEncoding(_)),
        "negative large fractional should produce ValueTooLargeForRealEncoding, got: {err}"
    );
}

#[test]
fn test_real_from_f64_boundary_safe_fractional() {
    // 9.0e12 + 0.5 is fractional. (9.0e12 + 0.5) * 1e6 = ~9.0e18,
    // which is just under i64::MAX (~9.22e18). Should succeed.
    let val = 9.0e12_f64 + 0.5;
    let expr = real_from_f64(val).expect("boundary-safe fractional should encode");
    let smt2 = format!("{}", expr);
    // Verify it encodes as a ratio, not an integer
    assert!(
        smt2.contains("1000000"),
        "boundary-safe fractional should use denominator 1000000, got: {smt2}"
    );
}

#[test]
fn test_real_from_f64_boundary_overflow_fractional() {
    // 9.3e12 + 0.5 is fractional. (9.3e12 + 0.5) * 1e6 = ~9.3e18,
    // which exceeds i64::MAX (~9.22e18).
    let val = 9.3e12_f64 + 0.5;
    let err = real_from_f64(val).unwrap_err();
    assert!(
        matches!(err, SmtError::ValueTooLargeForRealEncoding(_)),
        "boundary overflow should be caught, got: {err}"
    );
}

/// Precise boundary test for the `>=` guard (#398 AC5).
///
/// `9223372036854.775_f64 * 1e6` rounds to exactly `i64::MAX as f64`
/// (9223372036854775808.0). The old `>` guard missed this because
/// `i64_max_f64 > i64_max_f64` is false, causing `as i64` to saturate.
/// The `>=` guard catches it.
///
/// This test distinguishes `>` from `>=` — it fails with the old code.
#[test]
fn test_real_from_f64_exact_i64_max_boundary_rejected() {
    // This fractional value produces numer_f64 == i64::MAX as f64 exactly.
    let val = 9_223_372_036_854.775_f64;
    assert_ne!(
        val,
        val.floor(),
        "must be fractional to take the ratio path"
    );

    let err = real_from_f64(val).unwrap_err();
    assert!(
        matches!(err, SmtError::ValueTooLargeForRealEncoding(_)),
        "exact i64::MAX boundary should be rejected by >= guard, got: {err}"
    );

    // Negative counterpart must also be rejected.
    let err_neg = real_from_f64(-val).unwrap_err();
    assert!(
        matches!(err_neg, SmtError::ValueTooLargeForRealEncoding(_)),
        "negative exact boundary should also be rejected, got: {err_neg}"
    );
}

/// Companion: a value just below the boundary should succeed.
#[test]
fn test_real_from_f64_just_below_i64_max_boundary_succeeds() {
    // One unit less in the integer part → numer_f64 < i64::MAX as f64.
    let val = 9_223_372_036_853.775_f64;
    assert_ne!(val, val.floor(), "must be fractional");
    real_from_f64(val).expect("just below i64::MAX boundary should encode");
}

#[test]
fn test_real_from_f64_large_integer_succeeds() {
    // Large integer-valued f64 that fits in i64 takes the integer path.
    // 1e15 == (1e15).floor(), so this uses Expr::real(val as i64) directly.
    let expr = real_from_f64(1e15).expect("1e15 should encode");
    let smt2 = format!("{}", expr);
    // Should encode as the exact integer, not a fraction
    assert_eq!(
        smt2, "1000000000000000.0",
        "1e15 should encode as integer Real 1000000000000000.0"
    );
}

#[test]
fn test_real_from_f64_issue_161_ac3() {
    // AC3 from #161: real_from_f64(1e15) should not produce i64::MAX/1_000_000.
    // 1e15 is integer-valued so it takes the integer path (no overflow risk).
    // Verify it succeeds without overflow.
    let expr = real_from_f64(1e15).expect("1e15 should encode without overflow");
    // The expression should be a clean integer Real, not a saturated fraction.
    let debug = format!("{:?}", expr);
    assert!(
        !debug.contains("9223372036854"),
        "should not contain i64::MAX prefix, got: {debug}"
    );
}

// --- constant_params finiteness validation (#235) ---

#[test]
fn test_translate_nan_constant_param_rejected() {
    let kernel = KernelDef::new(
        "scale",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("alpha", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let err = translate_kernel(
        &kernel,
        &[ParamBinding::Constant(f32::NAN), ParamBinding::Variable],
    )
    .unwrap_err();
    assert!(
        matches!(err, SmtError::NonFiniteConstantParam { index: 0, .. }),
        "NaN constant param should produce NonFiniteConstantParam at index 0, got: {err}"
    );
}

#[test]
fn test_translate_inf_constant_param_rejected() {
    let kernel = KernelDef::new(
        "scale",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("alpha", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let err = translate_kernel(
        &kernel,
        &[
            ParamBinding::Constant(f32::INFINITY),
            ParamBinding::Variable,
        ],
    )
    .unwrap_err();
    assert!(
        matches!(err, SmtError::NonFiniteConstantParam { index: 0, .. }),
        "Inf constant param should produce NonFiniteConstantParam at index 0, got: {err}"
    );
}

#[test]
fn test_translate_neg_inf_constant_param_rejected() {
    let kernel = KernelDef::new(
        "scale",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("alpha", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let err = translate_kernel(
        &kernel,
        &[
            ParamBinding::Constant(f32::NEG_INFINITY),
            ParamBinding::Variable,
        ],
    )
    .unwrap_err();
    assert!(
        matches!(err, SmtError::NonFiniteConstantParam { index: 0, .. }),
        "NEG_INFINITY constant param should produce NonFiniteConstantParam at index 0, got: {err}"
    );
}

#[test]
fn test_translate_second_constant_param_nan_rejected() {
    let kernel = KernelDef::new(
        "two_const",
        vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
            Param::new("x", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(NodeId::new(2), IRNodeKind::Param(2)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(2),
                },
            ),
        ],
        NodeId::new(3),
    );
    // First constant is finite, second is NaN — should catch at index 1.
    let err = translate_kernel(
        &kernel,
        &[
            ParamBinding::Constant(1.0),
            ParamBinding::Constant(f32::NAN),
            ParamBinding::Variable,
        ],
    )
    .unwrap_err();
    assert!(
        matches!(err, SmtError::NonFiniteConstantParam { index: 1, .. }),
        "NaN at index 1 should produce NonFiniteConstantParam {{ index: 1 }}, got: {err}"
    );
}
