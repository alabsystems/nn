// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! K6 RoPE ay SMT verification tests (#19, #253, #275).

use super::prove_dispatch::rope_output_bounds;
use super::*;
use crate::test_helpers::bounds;

/// Helper: build rope_cos kernel via the builder.
fn rope_cos_kernel() -> KernelDef {
    nn_dsl::build_rope_cos_kernel().expect("rope_cos kernel must build")
}

/// Helper: build rope_sin kernel via the builder.
fn rope_sin_kernel() -> KernelDef {
    nn_dsl::build_rope_sin_kernel().expect("rope_sin kernel must build")
}

#[test]
fn test_verify_rope_cos_translates() {
    // rope_cos(x0, x1, freq) = x0 * cos(freq) - x1 * sin(freq)
    // #448: x0 is variable in [-10, 10], constant_params = [x1=1.0, freq=0.5].
    // cos(0.5) and sin(0.5) operate on constants → ground-folded (#376).
    // Yields Exact encoding → ay direct execution attempted.
    //
    // Analytical bounds are widened by SMT_QUANTIZATION_MARGIN (#539) to account
    // for real_from_f64 encoding error in ground-folded trig constants (cos(0.5),
    // sin(0.5)). The margin eliminates spurious Counterexample results.
    // ay#5605 fixed: real_mul with fractional coefficients now works.
    let kernel = rope_cos_kernel();
    let result = verify_kernel_smt(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.property, "output_bounded");
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: ground-folded rope_cos must reach Proven, got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_verify_rope_sin_translates() {
    // rope_sin(x0, x1, freq) = x0 * sin(freq) + x1 * cos(freq)
    // #448: x0 is variable in [-10, 10], constant_params = [x1=1.0, freq=0.5].
    // sin(0.5) and cos(0.5) operate on constants → ground-folded (#376).
    //
    // Analytical bounds are widened by SMT_QUANTIZATION_MARGIN (#539),
    // same fix as rope_cos — see that test's comment.
    let kernel = rope_sin_kernel();
    let result = verify_kernel_smt(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.property, "output_bounded");
    assert_eq!(result.solver, "ay-direct");
    // ay#5605 fixed: real_mul with fractional sin(0.5)/cos(0.5) coefficients now works.
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: ground-folded rope_sin must reach Proven, got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_rope_cos_smt2_output() {
    // With ground-folding (#376), cos(0.5)/sin(0.5) are folded to Real literals.
    let kernel = rope_cos_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    assert!(smt2.contains("set-logic"));
    assert!(smt2.contains("check-sat"));
    // Ground-folded: trig on constants → no UF declarations.
    assert!(
        !smt2.contains("cos_approx") && !smt2.contains("sin_approx"),
        "ground-folded rope_cos should NOT contain cos_approx/sin_approx UF"
    );
    // Input bounds for x0 variable.
    assert!(smt2.contains("-10"), "should contain input lower bound");
    assert!(smt2.contains("10"), "should contain input upper bound");
}

#[test]
fn test_rope_sin_smt2_output() {
    // With ground-folding (#376), sin(0.5)/cos(0.5) are folded to Real literals.
    let kernel = rope_sin_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    assert!(smt2.contains("set-logic"));
    assert!(smt2.contains("check-sat"));
    // Ground-folded: trig on constants → no UF declarations.
    assert!(
        !smt2.contains("cos_approx") && !smt2.contains("sin_approx"),
        "ground-folded rope_sin should NOT contain cos_approx/sin_approx UF"
    );
}

#[test]
fn test_rope_cos_with_explicit_bounds() {
    // Ground-folded (#376): cos(0.5)/sin(0.5) on constants → Exact encoding.
    // rope_cos(x0, x1=1.0, freq=0.5) = x0*cos(0.5) - sin(0.5)
    //   = x0*0.87758 - 0.47943
    // With x0 in [-pi, pi]: range ≈ [-3.24, 2.28].
    // Use bounds (-4.0, 3.0) to comfortably contain the output.
    // (Original (-2.0, 2.0) was too tight — max output ≈ 2.28 exceeds 2.0.)
    let kernel = rope_cos_kernel();
    let result = verify_kernel_smt_with_bounds(
        &kernel,
        &[1.0, 0.5],
        bounds(-std::f32::consts::PI, std::f32::consts::PI),
        Some((-4.0, 3.0)),
    )
    .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    // ay#5605 fixed: real_mul with fractional cos(0.5)/sin(0.5) coefficients now works.
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: rope_cos with explicit bounds must reach Proven, got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_rope_nan_constant_params_rejected() {
    let kernel = rope_cos_kernel();
    let err = verify_kernel_smt(&kernel, &[f32::NAN, 0.5], bounds(-10.0, 10.0)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN constant param should be rejected, got: {msg}"
    );
}

#[test]
fn test_rope_cos_uses_tight_analytical_bounds() {
    // rope_cos(x0=1.0, x1=0.5, freq) = cos(freq) - 0.5*sin(freq).
    // With analytical bounds, the output range is [-1.5, 1.5].
    // With ±1e6 fallback, the bounds would be (-1000010, 1000010).
    // real_from_f64 encodes with 1e6 denominator, so check for the fallback
    // numerator "1000010000000" (±1e6 + input_range scaled by 1e6).
    let kernel = rope_cos_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    assert!(
        !smt2.contains("1000010000000"),
        "rope_cos should use analytical bounds, not ±1e6 fallback. SMT2 excerpt: {}",
        &smt2[..smt2.len().min(500)]
    );
}

#[test]
fn test_rope_sin_uses_tight_analytical_bounds() {
    let kernel = rope_sin_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    assert!(
        !smt2.contains("1000010000000"),
        "rope_sin should use analytical bounds, not ±1e6 fallback. SMT2 excerpt: {}",
        &smt2[..smt2.len().min(500)]
    );
}

// --- RoPE ay structural and numerical correctness tests (P1 #275) ---

#[test]
fn test_rope_cos_smt2_declares_variable_no_uf() {
    // rope_cos has 3 params: (x0, x1, freq). With constant_params=[x1, freq],
    // only x0 is the symbolic variable → exactly 1 `declare-const`.
    // Ground-folding (#376) eliminates trig UFs → 0 declare-fun.
    let kernel = rope_cos_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    let declare_const_count = smt2.matches("declare-const").count();
    assert_eq!(
        declare_const_count, 1,
        "expected exactly 1 declare-const (symbolic x0 variable), got {declare_const_count}. SMT2:\n{smt2}"
    );
    // Ground-folded: no UF declarations.
    let declare_fun_count = smt2.matches("declare-fun").count();
    assert_eq!(
        declare_fun_count, 0,
        "expected 0 declare-fun (trig ground-folded), got {declare_fun_count}"
    );
}

#[test]
fn test_rope_cos_analytical_bounds_match_dsl_computation() {
    // #459: Verify analytical bounds match SMT-LIB2 for #448 variable-first convention.
    // rope_cos(x0, x1, freq): x0 is variable in [-10, 10], x1=1.0, freq=0.5 are constants.
    // rope_cos = x0*cos(0.5) - 1.0*sin(0.5) = x0*0.87758 - 0.47943
    // lower = -10*0.87758 - 0.47943 ≈ -9.25523
    // upper =  10*0.87758 - 0.47943 ≈  8.29637
    let (lo, hi) = nn_dsl::rope_cos_scalar_bounds(-10.0, 10.0, 1.0, 1.0, 0.5, 0.5)
        .expect("analytical bounds");
    assert!(
        lo < -9.0 && lo > -10.0,
        "rope_cos lower bound should be ~-9.26, got {lo}"
    );
    assert!(
        hi > 8.0 && hi < 9.0,
        "rope_cos upper bound should be ~8.30, got {hi}"
    );

    // Verify the heuristic also produces these same values by checking the
    // SMT-LIB2 does not contain the fallback ±1e6 pattern.
    let kernel = rope_cos_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    assert!(!smt2.contains("1000010000000"));

    // Verify the output bound assertion in SMT-LIB2 contains the
    // numerically-encoded analytical bounds (encoded as Real via 1e6 denom).
    // lower ≈ -9.2552 → numerator ≈ -9255251
    // upper ≈  8.2964 → numerator ≈  8296400
    assert!(
        smt2.contains("9255"),
        "SMT-LIB2 should contain 9255... (analytical lower bound * 1e6 denom), got excerpt: {}",
        &smt2[..smt2.len().min(800)]
    );
}

#[test]
fn test_rope_sin_analytical_bounds_match_dsl_computation() {
    // rope_sin(x0=1.0, x1=0.5, freq ∈ [-10, 10]):
    //   sin(freq) ∈ [-1, 1], cos(freq) ∈ [-1, 1]
    //   term1 = 1.0 * sin(freq) ∈ [-1, 1]
    //   term2 = 0.5 * cos(freq) ∈ [-0.5, 0.5]
    //   output ∈ [-1.5, 1.5]
    let (lo, hi) = nn_dsl::rope_sin_scalar_bounds(1.0, 1.0, 0.5, 0.5, -10.0, 10.0)
        .expect("analytical bounds");
    assert!(
        (lo - (-1.5)).abs() < 1e-5,
        "rope_sin lower bound should be -1.5, got {lo}"
    );
    assert!(
        (hi - 1.5).abs() < 1e-5,
        "rope_sin upper bound should be 1.5, got {hi}"
    );
}

#[test]
fn test_rope_cos_smt2_ground_folded_no_trig_uf() {
    // With ground-folding (#376): cos(0.5)/sin(0.5) are constants,
    // so no UF approximations should be emitted.
    let kernel = rope_cos_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    assert!(
        !smt2.contains("cos_approx"),
        "ground-folded rope_cos should NOT declare cos_approx UF"
    );
    assert!(
        !smt2.contains("sin_approx"),
        "ground-folded rope_cos should NOT declare sin_approx UF"
    );
}

#[test]
fn test_rope_cos_asymmetric_constants_produce_asymmetric_bounds() {
    // With x0=5.0, x1=1.0: cos term dominates → bounds should be asymmetric
    // vs x0=1.0, x1=5.0 where sin term dominates.
    let (lo_a, hi_a) =
        nn_dsl::rope_cos_scalar_bounds(5.0, 5.0, 1.0, 1.0, 0.0, 1.0).expect("bounds A");
    let (lo_b, hi_b) =
        nn_dsl::rope_cos_scalar_bounds(1.0, 1.0, 5.0, 5.0, 0.0, 1.0).expect("bounds B");
    // Both should be valid (lo <= hi).
    assert!(lo_a <= hi_a, "bounds A inverted: [{lo_a}, {hi_a}]");
    assert!(lo_b <= hi_b, "bounds B inverted: [{lo_b}, {hi_b}]");
    // They should differ — different constant magnitudes produce different widths.
    let width_a = hi_a - lo_a;
    let width_b = hi_b - lo_b;
    assert!(
        (width_a - width_b).abs() > 0.01,
        "asymmetric constants should produce different bound widths: A={width_a}, B={width_b}"
    );
}

// --- rope_output_bounds finiteness guard tests (#384) ---

#[test]
fn test_rope_output_bounds_rejects_infinite_lower() {
    // Synthetic bounds_fn that returns infinity in the lower bound.
    // The finiteness guard in rope_output_bounds must catch this.
    fn bad_bounds_fn(
        _x0_lo: f32,
        _x0_hi: f32,
        _x1_lo: f32,
        _x1_hi: f32,
        _freq_lo: f32,
        _freq_hi: f32,
    ) -> Result<(f32, f32), nn_dsl::kernel_error::KernelError> {
        Ok((f32::INFINITY, 1.0))
    }
    let err = rope_output_bounds(1.0, 0.5, -10.0, 10.0, bad_bounds_fn).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite"),
        "infinite lower bound should be rejected, got: {msg}"
    );
}

#[test]
fn test_rope_output_bounds_rejects_nan_upper() {
    // Synthetic bounds_fn that returns NaN in the upper bound.
    fn bad_bounds_fn(
        _x0_lo: f32,
        _x0_hi: f32,
        _x1_lo: f32,
        _x1_hi: f32,
        _freq_lo: f32,
        _freq_hi: f32,
    ) -> Result<(f32, f32), nn_dsl::kernel_error::KernelError> {
        Ok((-1.0, f32::NAN))
    }
    let err = rope_output_bounds(1.0, 0.5, -10.0, 10.0, bad_bounds_fn).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite"),
        "NaN upper bound should be rejected, got: {msg}"
    );
}

#[test]
fn test_rope_output_bounds_accepts_finite() {
    // Synthetic bounds_fn returning valid finite bounds.
    fn good_bounds_fn(
        _x0_lo: f32,
        _x0_hi: f32,
        _x1_lo: f32,
        _x1_hi: f32,
        _freq_lo: f32,
        _freq_hi: f32,
    ) -> Result<(f32, f32), nn_dsl::kernel_error::KernelError> {
        Ok((-1.5, 1.5))
    }
    let result = rope_output_bounds(1.0, 0.5, -10.0, 10.0, good_bounds_fn);
    assert!(result.is_ok(), "finite bounds should be accepted");
    let (lo, hi) = result.unwrap();
    assert!((lo - (-1.5_f64)).abs() < 1e-6);
    assert!((hi - 1.5_f64).abs() < 1e-6);
}
