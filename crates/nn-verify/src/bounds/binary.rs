// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Analytical output bounds for binary operations (add, mul).
//!
//! Binary add is a multi-variable kernel (`x + y`) where both inputs are
//! symbolic. The output bounds are exact: `[x_lo + y_lo, x_hi + y_hi]`.
//! This is trivially sound because addition is monotonically increasing
//! in both arguments.

use crate::error::VerifyError;
use crate::smt_error::SmtError;

/// Compute analytical output bounds for element-wise addition.
///
/// `binary_add(x, y) = x + y`
///
/// Addition is monotonically increasing in both arguments. Output bounds
/// are exact: `(x_lower + y_lower, x_upper + y_upper)`.
pub(crate) fn binary_add_output_bounds(
    x_lower: f64,
    x_upper: f64,
    y_lower: f64,
    y_upper: f64,
) -> Result<(f64, f64), VerifyError> {
    if !x_lower.is_finite() || !x_upper.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: x_lower,
            upper: x_upper,
        }
        .into());
    }
    if !y_lower.is_finite() || !y_upper.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: y_lower,
            upper: y_upper,
        }
        .into());
    }
    if x_lower > x_upper {
        return Err(SmtError::InvertedBounds {
            lower: x_lower,
            upper: x_upper,
        }
        .into());
    }
    if y_lower > y_upper {
        return Err(SmtError::InvertedBounds {
            lower: y_lower,
            upper: y_upper,
        }
        .into());
    }

    let out_lower = x_lower + y_lower;
    let out_upper = x_upper + y_upper;

    if !out_lower.is_finite() || !out_upper.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: out_lower,
            upper: out_upper,
        }
        .into());
    }

    Ok((out_lower, out_upper))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_add_bounds_symmetric() {
        let (lo, hi) = binary_add_output_bounds(-1.0, 1.0, -1.0, 1.0).unwrap();
        assert!((lo - (-2.0)).abs() < 1e-10);
        assert!((hi - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_binary_add_bounds_asymmetric() {
        let (lo, hi) = binary_add_output_bounds(0.0, 5.0, -3.0, 2.0).unwrap();
        assert!((lo - (-3.0)).abs() < 1e-10);
        assert!((hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_binary_add_bounds_point() {
        let (lo, hi) = binary_add_output_bounds(2.0, 2.0, 3.0, 3.0).unwrap();
        assert!((lo - 5.0).abs() < 1e-10);
        assert!((hi - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_binary_add_bounds_non_finite_x_rejects() {
        let result = binary_add_output_bounds(f64::NAN, 1.0, -1.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_add_bounds_non_finite_y_rejects() {
        let result = binary_add_output_bounds(-1.0, 1.0, f64::INFINITY, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_add_bounds_inverted_x_rejects() {
        let result = binary_add_output_bounds(1.0, -1.0, -1.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_add_bounds_inverted_y_rejects() {
        let result = binary_add_output_bounds(-1.0, 1.0, 1.0, -1.0);
        assert!(result.is_err());
    }
}
