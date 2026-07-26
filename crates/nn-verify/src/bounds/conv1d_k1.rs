// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Analytical output bounds for Conv1d kernel_size=1 scalar representation.
//!
//! Conv1d with kernel_size=1 reduces to a per-position linear transformation.
//! For a single (input_channel, output_channel) path, the scalar kernel is:
//!   `out = x * weight + bias`
//! where `weight` and `bias` are constants (model parameters).
//!
//! This is a linear function of `x`, so output bounds are exact:
//!   `out ∈ [min(lo*w+b, hi*w+b), max(lo*w+b, hi*w+b)]`

use crate::error::VerifyError;
use crate::smt_error::SmtError;

/// Compute analytical output bounds for conv1d_k1_scalar.
///
/// `conv1d_k1_scalar(x, weight, bias) = x * weight + bias`
///
/// **#448 convention:** param 0 (x) is the symbolic variable bounded by
/// `[x_lower, x_upper]`, param 1 (weight) and param 2 (bias) are constants.
/// `constant_params[0]` = weight, `constant_params[1]` = bias.
///
/// Linear in x: output is monotonically increasing if weight > 0,
/// decreasing if weight < 0. Bounds are exact.
pub(crate) fn conv1d_k1_scalar_output_bounds(
    weight: f64,
    bias: f64,
    x_lower: f64,
    x_upper: f64,
) -> Result<(f64, f64), VerifyError> {
    if !weight.is_finite() {
        return Err(SmtError::NonFiniteConstantParam {
            index: 1,
            value: weight,
        }
        .into());
    }
    if !bias.is_finite() {
        return Err(SmtError::NonFiniteConstantParam {
            index: 2,
            value: bias,
        }
        .into());
    }
    if !x_lower.is_finite() || !x_upper.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: x_lower,
            upper: x_upper,
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

    let a = x_lower * weight + bias;
    let b = x_upper * weight + bias;
    let (out_lower, out_upper) = if a <= b { (a, b) } else { (b, a) };

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
    fn test_conv1d_k1_positive_weight() {
        // out = x * 2.0 + 1.0, x in [-1, 1] → out in [-1, 3]
        let (lo, hi) = conv1d_k1_scalar_output_bounds(2.0, 1.0, -1.0, 1.0).unwrap();
        assert!((lo - (-1.0)).abs() < 1e-10);
        assert!((hi - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv1d_k1_negative_weight() {
        // out = x * (-3.0) + 0.0, x in [0, 2] → out in [-6, 0]
        let (lo, hi) = conv1d_k1_scalar_output_bounds(-3.0, 0.0, 0.0, 2.0).unwrap();
        assert!((lo - (-6.0)).abs() < 1e-10);
        assert!((hi - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv1d_k1_zero_weight() {
        // out = x * 0.0 + 5.0, x in [-100, 100] → out = 5.0
        let (lo, hi) = conv1d_k1_scalar_output_bounds(0.0, 5.0, -100.0, 100.0).unwrap();
        assert!((lo - 5.0).abs() < 1e-10);
        assert!((hi - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_conv1d_k1_non_finite_weight_rejects() {
        let result = conv1d_k1_scalar_output_bounds(f64::NAN, 0.0, -1.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv1d_k1_non_finite_bias_rejects() {
        let result = conv1d_k1_scalar_output_bounds(1.0, f64::INFINITY, -1.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv1d_k1_inverted_bounds_rejects() {
        let result = conv1d_k1_scalar_output_bounds(1.0, 0.0, 1.0, -1.0);
        assert!(result.is_err());
    }
}
