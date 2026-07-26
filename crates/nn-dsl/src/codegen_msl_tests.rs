// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for MSL code generation from KernelIR.

use super::*;
use crate::test_kernels::parse_kernel as lower;

#[test]
fn test_snake_msl_contains_sin() {
    let kernel = lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
            x + (1.0 / alpha) * (alpha * x).sin().powi(2)
        }",
    );
    let msl = emit_msl(&kernel).expect("emit");
    assert!(msl.contains("metal::precise::sin"), "MSL:\n{msl}");
    assert!(msl.contains("[[kernel]]"), "MSL:\n{msl}");
    assert!(msl.contains("snake_kernel"), "MSL:\n{msl}");
    assert!(msl.contains("#include <metal_stdlib>"), "MSL:\n{msl}");
}

#[test]
fn test_snake_msl_scalar_fn() {
    let kernel = lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
            x + (1.0 / alpha) * (alpha * x).sin().powi(2)
        }",
    );
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    assert!(
        scalar.starts_with("float snake(float x, float alpha)"),
        "scalar:\n{scalar}"
    );
    assert!(scalar.contains("return "), "scalar:\n{scalar}");
}

#[test]
fn test_relu_msl() {
    let kernel = lower("fn relu(x: f32) -> f32 { x.max(0.0) }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(msl.contains("max("), "MSL:\n{msl}");
}

#[test]
fn test_clamp_msl() {
    let kernel = lower("fn clamped(x: f32, lo: f32, hi: f32) -> f32 { x.clamp(lo, hi) }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(msl.contains("clamp("), "MSL:\n{msl}");
}

#[test]
fn test_f16_kernel_emits_half_signature_with_float_acc() {
    let kernel = lower("fn half_add(x: f16, y: f16) -> f16 { x + y }");
    let msl = emit_msl(&kernel).expect("emit");
    // Function signature uses half (buffer types).
    assert!(
        msl.contains("half _nn_half_add(half x, half y)"),
        "MSL:\n{msl}"
    );
    // Float-accumulator: params promoted to float, intermediates in float.
    assert!(
        msl.contains("float x_f = float(x);"),
        "F16 should promote x to float, MSL:\n{msl}"
    );
    assert!(
        msl.contains("float y_f = float(y);"),
        "F16 should promote y to float, MSL:\n{msl}"
    );
    // Result demoted back to half.
    assert!(
        msl.contains("return half("),
        "F16 should demote result to half, MSL:\n{msl}"
    );
}

#[test]
fn test_powi_3_emits_multiplication() {
    let kernel = lower("fn cube(x: f32) -> f32 { x.powi(3) }");
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    // powi(3) must expand to x * x * x, not metal::pow which is wrong for negative bases
    assert!(
        scalar.contains("x * x * x"),
        "powi(3) should expand to repeated multiplication, scalar:\n{scalar}"
    );
    assert!(
        !scalar.contains("metal::pow") && !scalar.contains("metal::precise::pow"),
        "powi should never use metal::pow (broken for negative bases), scalar:\n{scalar}"
    );
}

#[test]
fn test_powi_neg1_emits_reciprocal() {
    let kernel = lower("fn inv(x: f32) -> f32 { x.powi(-1) }");
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    assert!(
        scalar.contains("float(1) / x"),
        "powi(-1) should emit T(1) / b, scalar:\n{scalar}"
    );
}

#[test]
fn test_powi_neg2_emits_reciprocal_square() {
    let kernel = lower("fn inv_sq(x: f32) -> f32 { x.powi(-2) }");
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    assert!(
        scalar.contains("float(1) / (x * x)"),
        "powi(-2) should emit T(1) / (b * b), scalar:\n{scalar}"
    );
}

#[test]
fn test_recip_emits_division() {
    let kernel = lower("fn inv(x: f32) -> f32 { x.recip() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("float(1) /"),
        "recip should emit 1/x pattern, MSL:\n{msl}"
    );
}

#[test]
fn test_relaxed_precision_uses_non_precise_intrinsics() {
    let kernel = lower("fn relaxed_math(x: f32) -> f32 { x.sin().exp() }");
    let contract = PrecisionContract::bootstrap(PrecisionTier::Relaxed, ScalarType::F32);
    let msl = emit_msl_with_contract(&kernel, contract).expect("emit");
    assert!(msl.contains("metal::sin("), "MSL:\n{msl}");
    assert!(msl.contains("metal::exp("), "MSL:\n{msl}");
    assert!(!msl.contains("metal::precise::sin"), "MSL:\n{msl}");
    assert!(!msl.contains("metal::precise::exp"), "MSL:\n{msl}");
}

#[test]
fn test_cos_msl_emits_precise_cos() {
    let kernel = lower("fn cosine(x: f32) -> f32 { x.cos() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("metal::precise::cos("),
        "cos should emit metal::precise::cos, MSL:\n{msl}"
    );
}

#[test]
fn test_sqrt_msl_emits_precise_sqrt() {
    let kernel = lower("fn root(x: f32) -> f32 { x.sqrt() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("metal::precise::sqrt("),
        "sqrt should emit metal::precise::sqrt, MSL:\n{msl}"
    );
}

#[test]
fn test_rsqrt_msl_emits_precise_rsqrt() {
    let kernel = lower("fn inv_sqrt(x: f32) -> f32 { x.rsqrt() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("metal::precise::rsqrt("),
        "rsqrt should emit metal::precise::rsqrt, MSL:\n{msl}"
    );
}

#[test]
fn test_exp_msl_emits_precise_exp() {
    let kernel = lower("fn exponential(x: f32) -> f32 { x.exp() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("metal::precise::exp("),
        "exp should emit metal::precise::exp, MSL:\n{msl}"
    );
}

#[test]
fn test_abs_msl_emits_metal_abs() {
    let kernel = lower("fn absolute(x: f32) -> f32 { x.abs() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("metal::abs("),
        "abs should emit metal::abs (no precise:: prefix), MSL:\n{msl}"
    );
    // abs must NOT use the precise:: prefix
    assert!(
        !msl.contains("metal::precise::abs"),
        "abs should not use precise:: prefix, MSL:\n{msl}"
    );
}

#[test]
fn test_min_msl_emission() {
    let kernel = lower("fn capped(x: f32) -> f32 { x.min(1.0) }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(msl.contains("min("), "min should emit min(), MSL:\n{msl}");
}

#[test]
fn test_sum_reduce_msl_emits_add_chain() {
    let kernel = lower(
        "fn sum3(a: f32, b: f32, c: f32) -> f32 {
            nn_dsl::sum_reduce([a, b, c])
        }",
    );
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("float t3 = a + b + c;"),
        "sum_reduce should emit explicit add chain, MSL:\n{msl}"
    );
}

#[test]
fn test_strict_precision_uses_precise_intrinsics() {
    let kernel = lower("fn strict_math(x: f32) -> f32 { x.sin().cos() }");
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    let msl = emit_msl_with_contract(&kernel, contract).expect("emit");
    assert!(
        msl.contains("metal::precise::sin("),
        "strict tier should use metal::precise::sin, MSL:\n{msl}"
    );
    assert!(
        msl.contains("metal::precise::cos("),
        "strict tier should use metal::precise::cos, MSL:\n{msl}"
    );
}

#[test]
fn test_format_float_zero() {
    assert_eq!(format_float(0.0), "0.0");
}

#[test]
fn test_format_float_integer_value() {
    assert_eq!(format_float(42.0), "42.0");
}

#[test]
fn test_format_float_fractional() {
    let s = format_float(1.23456);
    assert!(
        s.contains("1.23456"),
        "expected fractional format, got: {s}"
    );
}

#[test]
fn test_format_float_negative() {
    let s = format_float(-1.0);
    assert_eq!(s, "-1.0");
}

// ======================== format_float boundary conditions ========================

#[test]
fn test_format_float_negative_zero_preserves_sign() {
    // IEEE 754: -0.0 == 0.0 is true, but -0.0 and 0.0 have different
    // semantics (e.g., 1.0 / -0.0 = -Inf). MSL must preserve the sign.
    let s = format_float(-0.0_f64);
    assert_eq!(s, "-0.0", "format_float(-0.0) must preserve sign for MSL");
    let s = format_float(0.0_f64);
    assert_eq!(s, "0.0", "format_float(+0.0) should not have negative sign");
}

#[test]
fn test_format_float_nan_produces_valid_msl() {
    let s = format_float(f64::NAN);
    assert_eq!(s, "NAN", "NaN must emit MSL-valid NAN macro, got: {s}");
}

#[test]
fn test_format_float_infinity_produces_valid_msl() {
    let s = format_float(f64::INFINITY);
    assert_eq!(s, "INFINITY", "+Inf must emit MSL-valid INFINITY, got: {s}");
}

#[test]
fn test_format_float_neg_infinity_produces_valid_msl() {
    let s = format_float(f64::NEG_INFINITY);
    assert_eq!(
        s, "(-INFINITY)",
        "-Inf must emit parenthesized MSL-valid (-INFINITY), got: {s}"
    );
}

#[test]
fn test_format_float_very_large_integer() {
    // 1e15 is the boundary in the floor check
    let s = format_float(1e15);
    // 1e15 == 1e15.floor() is true, but 1e15.abs() < 1e15 is false,
    // so it falls through to the general format.
    assert!(
        !s.is_empty(),
        "very large integer should produce valid output"
    );
}

#[test]
fn test_format_float_subnormal() {
    let s = format_float(5e-324_f64);
    assert!(!s.is_empty(), "subnormal should produce output: {s}");
}

// ======================== float-accumulator mode ========================

#[test]
fn test_f16_exp_uses_float_intermediates() {
    let kernel = lower("fn f16_exp(x: f16) -> f16 { x.exp() }");
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    // intermediates must be float, not half
    assert!(
        scalar.contains("float x_f = float(x);"),
        "F16 exp should promote param, scalar:\n{scalar}"
    );
    assert!(
        scalar.contains("float t1 = metal::precise::exp(x_f);"),
        "F16 exp should use float intermediate, scalar:\n{scalar}"
    );
    assert!(
        scalar.contains("return half("),
        "F16 exp should demote output, scalar:\n{scalar}"
    );
}

#[test]
fn test_f32_kernel_no_accumulator_promotion() {
    let kernel = lower("fn f32_exp(x: f32) -> f32 { x.exp() }");
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    // F32 kernels should NOT have accumulator promotion
    assert!(
        !scalar.contains("_f = float("),
        "F32 kernel should not promote params, scalar:\n{scalar}"
    );
    assert!(
        !scalar.contains("return float("),
        "F32 kernel should not wrap return, scalar:\n{scalar}"
    );
    assert!(
        scalar.contains("return t1;"),
        "F32 kernel should return directly, scalar:\n{scalar}"
    );
}

#[test]
fn test_f16_recip_uses_float_division() {
    let kernel = lower("fn f16_inv(x: f16) -> f16 { x.recip() }");
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    assert!(
        scalar.contains("float(1) / x_f"),
        "F16 recip should use float(1) / promoted param, scalar:\n{scalar}"
    );
}

// -- Validation tests extracted to codegen_msl_tests_validation.rs (#1565) --
#[path = "codegen_msl_tests_validation.rs"]
mod validation;
