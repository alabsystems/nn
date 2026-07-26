// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Analytical output bounds for Snake activation.
//!
//! Extracted from `ay/snake_uf.rs` (#859). Only the pure-Rust bounds function
//! is here; the ay-bindings-dependent `assert_input_bounds` stays in `ay/snake_uf.rs`.

use crate::smt_error::SmtError;

/// Compute output bounds for Snake activation: `snake(x, alpha) = x + (1/alpha) * sin(alpha*x)^2`.
///
/// For alpha > 0:
///   sin(alpha*x)^2 ∈ [0, 1]
///   (1/alpha) * sin(alpha*x)^2 ∈ [0, 1/alpha]
///   snake(x, alpha) ∈ [x, x + 1/alpha]
///
/// So global bounds over x ∈ [x_lo, x_hi]:
///   output_lower = x_lo       (minimum of x + non-negative term)
///   output_upper = x_hi + 1/alpha  (maximum of x + bounded term)
pub(crate) fn snake_output_bounds(
    x_lo: f64,
    x_hi: f64,
    alpha: f64,
) -> Result<(f64, f64), SmtError> {
    if alpha <= 0.0 || !alpha.is_finite() {
        return Err(SmtError::InvalidSnakeAlpha(alpha));
    }
    if !x_lo.is_finite() || !x_hi.is_finite() {
        if !x_lo.is_finite() {
            return Err(SmtError::NonFiniteLiteral(x_lo));
        }
        return Err(SmtError::NonFiniteLiteral(x_hi));
    }
    if x_lo > x_hi {
        return Err(SmtError::InvertedBounds {
            lower: x_lo,
            upper: x_hi,
        });
    }
    let output_lower = x_lo;
    let output_upper = x_hi + 1.0 / alpha;
    if !output_lower.is_finite() || !output_upper.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: output_lower,
            upper: output_upper,
        });
    }
    Ok((output_lower, output_upper))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_bounds_alpha_1() {
        let (lo, hi) = snake_output_bounds(-10.0, 10.0, 1.0).unwrap();
        assert_eq!(lo, -10.0);
        assert_eq!(hi, 11.0); // 10 + 1/1
    }

    #[test]
    fn test_snake_bounds_alpha_2() {
        let (lo, hi) = snake_output_bounds(-5.0, 5.0, 2.0).unwrap();
        assert_eq!(lo, -5.0);
        assert_eq!(hi, 5.5); // 5 + 1/2
    }

    #[test]
    fn test_snake_bounds_unit_interval() {
        let (lo, hi) = snake_output_bounds(0.0, 1.0, 4.0).unwrap();
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 1.25); // 1 + 1/4
    }

    #[test]
    fn test_snake_bounds_zero_alpha_rejected() {
        let err = snake_output_bounds(-1.0, 1.0, 0.0).unwrap_err();
        assert!(
            matches!(err, SmtError::InvalidSnakeAlpha(a) if a == 0.0),
            "alpha=0 should be rejected, got: {err}"
        );
    }

    #[test]
    fn test_snake_bounds_negative_alpha_rejected() {
        let err = snake_output_bounds(-1.0, 1.0, -2.0).unwrap_err();
        assert!(
            matches!(err, SmtError::InvalidSnakeAlpha(a) if a == -2.0),
            "negative alpha should be rejected, got: {err}"
        );
    }

    #[test]
    fn test_snake_bounds_inverted_x_rejected() {
        let err = snake_output_bounds(10.0, -10.0, 1.0).unwrap_err();
        assert!(
            matches!(err, SmtError::InvertedBounds { lower, upper } if lower == 10.0 && upper == -10.0),
            "inverted x bounds should be rejected, got: {err}"
        );
    }

    #[test]
    fn test_snake_bounds_small_alpha_large_x_overflow() {
        // 1.0 / 5e-324 (smallest positive f64) overflows to Inf.
        let err = snake_output_bounds(0.0, 1.0, 5e-324).unwrap_err();
        assert!(
            matches!(err, SmtError::NonFiniteBound { upper, .. } if upper.is_infinite()),
            "non-finite output should be rejected, got: {err}"
        );
    }
}
