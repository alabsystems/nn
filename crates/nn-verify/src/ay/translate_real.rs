// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! f64 → ay Real encoding with adaptive denominator precision.
//!
//! Extracted from `translate.rs` (#504). Contains `real_from_f64` and its
//! Kani proof harnesses. The canonical implementation per design doc
//! (`ay real_from_f64 numeric encoding` rule) — do not duplicate.
//!
//! # Precision model (#539)
//!
//! `real_from_f64` encodes f64 values as `round(val * denom) / denom` where
//! `denom` defaults to 1e6. This yields ~6 decimal digits of precision with
//! a maximum absolute rounding error of `0.5 / denom = 5e-7` per encoding.
//!
//! When ground-folded kernel constants (e.g., `cos(0.5)`, `rsqrt(eps)`) pass
//! through `real_from_f64`, the SMT kernel body operates on quantized values.
//! If the kernel multiplies a quantized constant by a symbolic variable with
//! magnitude M, the output error is up to `M * 5e-7`. For a kernel with N
//! ground-folded constants, the accumulated output error is at most
//! `Σ |∂f/∂ci| * 5e-7`, bounded by `N * M_max * 5e-7`.
//!
//! Analytical output bounds (in `prove_dispatch.rs`) are computed with exact
//! f64 arithmetic. To prevent spurious counterexamples from this precision
//! gap, `finalize_query` widens analytical bounds by `SMT_QUANTIZATION_MARGIN`
//! (1e-4) before encoding them into the SMT assertion.

use ay_bindings::Expr;

use super::error::SmtError;

/// Convert an f64 to a ay Real expression with fractional precision.
///
/// Returns `Err(NonFiniteLiteral)` for NaN/Inf inputs.
/// Returns `Err(ValueTooLargeForRealEncoding)` when the numerator (val * denom)
/// would overflow i64, which silently saturates to i64::MAX/MIN on Rust >= 1.45.
///
/// Uses an adaptive denominator for tiny values (< 5e-7) to avoid silently
/// quantizing subnormals and small epsilons to zero (#398).
pub(crate) fn real_from_f64(val: f64) -> Result<Expr, SmtError> {
    if !val.is_finite() {
        return Err(SmtError::NonFiniteLiteral(val));
    }
    // Integer-valued f64s that fit in i64 can be encoded directly.
    if val == val.floor() && val.abs() < (i64::MAX as f64) {
        return Ok(Expr::real(val as i64));
    }
    // Encode as numerator/denominator with ~6 decimal digits of precision.
    // Start with 1e6; if the value is too small, increase the denominator
    // adaptively to avoid zero-quantization of tiny values like epsilon.
    let denom = real_from_f64_denominator(val);
    let numer_f64 = (val * denom as f64).round();
    if numer_f64.abs() >= i64::MAX as f64 {
        return Err(SmtError::ValueTooLargeForRealEncoding(val));
    }
    let numer = numer_f64 as i64;
    Ok(Expr::real(numer).real_div(Expr::real(denom)))
}

/// Choose an adaptive denominator for `real_from_f64`.
///
/// For values where `|val * 1e6| < 1` (i.e., `|val| < 1e-6`), the default
/// 1e6 denominator would round to numerator 0, silently replacing the value
/// with zero. This function scales the denominator to maintain ~6 significant
/// digits, capping at 1e15 to stay safely within i64 range.
fn real_from_f64_denominator(val: f64) -> i64 {
    const DEFAULT_DENOM: i64 = 1_000_000;
    const MAX_DENOM: i64 = 1_000_000_000_000_000; // 1e15

    if val == 0.0 {
        return DEFAULT_DENOM;
    }

    let abs_val = val.abs();
    // If default denominator produces a non-zero numerator, use it.
    if abs_val * (DEFAULT_DENOM as f64) >= 0.5 {
        return DEFAULT_DENOM;
    }

    // Adaptive: find the power of 10 where |val * 10^n| >= 1, then add 6
    // digits of precision. Equivalent to 10^(ceil(-log10(|val|)) + 6).
    let log_scale = (-abs_val.log10()).ceil() as u32 + 6;
    let denom = 10i64.saturating_pow(log_scale.min(15));
    denom.min(MAX_DENOM)
}

#[cfg(test)]
#[path = "translate_real_tests.rs"]
mod tests;

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// floor stub: returns a finite value <= x (CBMC cannot model f64::floor).
    fn floor_f64_stub(x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite());
        kani::assume(r <= x);
        kani::assume(r >= x - 1.0);
        r
    }

    /// log10 stub: returns a finite value (CBMC cannot model f64::log10).
    fn log10_f64_stub(_x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r >= -20.0 && r <= 20.0);
        r
    }

    /// ceil stub: returns a finite value >= x (CBMC cannot model f64::ceil).
    fn ceil_f64_stub(x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite());
        kani::assume(r >= x);
        kani::assume(r <= x + 1.0);
        r
    }

    /// Proves `real_from_f64` rejects all non-finite inputs (NaN, +Inf, -Inf).
    #[kani::unwind(1)]
    #[kani::proof]
    fn real_from_f64_rejects_non_finite() {
        let val: f64 = kani::any();
        kani::assume(!val.is_finite());

        let result = real_from_f64(val);
        assert!(result.is_err(), "non-finite input must return Err");
    }

    /// Proves `real_from_f64` accepts all finite f64 within the safe magnitude range
    /// (|val| <= 9.2e12) and rejects inputs that would overflow the i64 numerator.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::floor, floor_f64_stub)]
    #[kani::stub(f64::log10, log10_f64_stub)]
    #[kani::stub(f64::ceil, ceil_f64_stub)]
    fn real_from_f64_safe_range_accepted() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());
        // Safe range: magnitude small enough that numer_f64 stays in i64 range.
        // The adaptive denominator is at most 1e15, so |val * 1e15| < i64::MAX
        // requires |val| < ~9223. Use a tighter bound for the common denominator (1e6).
        kani::assume(val.abs() <= 9.0e12);

        let result = real_from_f64(val);
        assert!(
            result.is_ok(),
            "finite values in safe range must be accepted"
        );
    }

    /// Proves `real_from_f64_denominator` always returns a positive value >= DEFAULT_DENOM
    /// for any finite f64 input.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::log10, log10_f64_stub)]
    #[kani::stub(f64::ceil, ceil_f64_stub)]
    fn real_from_f64_denominator_always_positive() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());

        let denom = real_from_f64_denominator(val);
        assert!(denom >= 1_000_000, "denominator must be at least 1e6");
        assert!(denom > 0, "denominator must be positive");
    }

    /// Proves `real_from_f64` rejects values whose numerator would overflow i64.
    /// The guard at line 35 uses `>=` (not `>`) per issue #398: `i64::MAX as f64`
    /// rounds up, so equality means the value is already out of i64 range.
    ///
    /// This harness picks values in the overflow band and asserts rejection.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::floor, floor_f64_stub)]
    #[kani::stub(f64::log10, log10_f64_stub)]
    #[kani::stub(f64::ceil, ceil_f64_stub)]
    fn real_from_f64_overflow_guard_boundary() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());
        // Focus on values large enough that numer_f64 = (val * denom).round() >= i64::MAX as f64.
        // With default denom = 1e6, that's |val| >= ~9.223e12.
        kani::assume(val.abs() >= 9.3e12);

        let result = real_from_f64(val);
        // Large-magnitude values must either be rejected (Err) or safely handled
        // through the integer fast-path (which has its own < guard).
        // The critical invariant: no silent i64 saturation.
        if let Ok(_) = result {
            // If Ok, it went through the integer fast-path: val == val.floor()
            // and val.abs() < i64::MAX as f64. Verify that condition held.
            assert!(
                val == val.floor() && val.abs() < i64::MAX as f64,
                "Ok for large value must come from safe integer fast-path"
            );
        }
        // Err is the expected outcome for overflow-range fractional values.
    }
}
