// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Analytical output bounds for normalization kernels.
//!
//! Covers: RMSNorm, LayerNorm/InstanceNormAffine (shared via `norm_affine`),
//! AdaIN, and InstanceNorm.
//!
//! Extracted from `ay/prove_bounds_norm.rs` (#859) to be always-available
//! without the `ay-smt` feature flag. Pure Rust math — no ay-bindings dependency.

use crate::error::VerifyError;
use crate::smt_error::SmtError;

/// Shared helper for normalization bounds functions that are linear in x.
///
/// All norm kernels compute `output(x) = slope * x + intercept` where
/// slope and intercept depend on constant params (mean, var, eps, gamma, beta).
///
/// Caller is responsible for computing `slope` and `intercept` from their
/// specific constant params. This function handles:
/// - Input bounds validation (finiteness, ordering)
/// - Linear evaluation at both endpoints
/// - Output finiteness guard
///
/// Defense-in-depth: this function validates inputs even though
/// `compute_output_bounds_heuristic` validates at the dispatch level (#394).
fn linear_norm_bounds(
    x_lower: f64,
    x_upper: f64,
    slope: f64,
    intercept: f64,
) -> Result<(f64, f64), VerifyError> {
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
    let a = slope * x_lower + intercept;
    let b = slope * x_upper + intercept;
    let (out_lo, out_hi) = if a <= b { (a, b) } else { (b, a) };
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: out_lo,
            upper: out_hi,
        }
        .into());
    }
    Ok((out_lo, out_hi))
}

/// Compute analytical output bounds for RMSNorm scalar: `x * rms_inv * weight`.
///
/// **#448 convention:** param 0 (x) is the symbolic variable bounded by
/// `[x_lower, x_upper]`, params 1-2 (rms_inv, weight) are constants.
/// Output = `x * coeff` where `coeff = rms_inv_const * weight_const`. Linear in x.
pub(crate) fn rms_norm_scalar_output_bounds(
    rms_inv_const: f32,
    weight_const: f32,
    x_lower: f64,
    x_upper: f64,
) -> Result<(f64, f64), VerifyError> {
    if !rms_inv_const.is_finite() {
        return Err(SmtError::NonFiniteConstantParam {
            index: 1,
            value: f64::from(rms_inv_const),
        }
        .into());
    }
    if !weight_const.is_finite() {
        return Err(SmtError::NonFiniteConstantParam {
            index: 2,
            value: f64::from(weight_const),
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

    let coeff = f64::from(rms_inv_const) * f64::from(weight_const);

    let a = coeff * x_lower;
    let b = coeff * x_upper;
    let (out_lo, out_hi) = if a <= b { (a, b) } else { (b, a) };

    if !out_lo.is_finite() || !out_hi.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: out_lo,
            upper: out_hi,
        }
        .into());
    }

    Ok((out_lo, out_hi))
}

/// Compute analytical output bounds for norm kernels with affine transform.
///
/// Shared by `layer_norm_scalar` and `instance_norm_affine_scalar`, which have
/// the same formula: `(x - mean) * rsqrt(var_val + eps) * gamma + beta`.
///
/// **#448/#459 convention:** param 0 (x) is the symbolic variable bounded by
/// `[x_lower, x_upper]`, params 1-5 (mean, var, eps, gamma, beta) are constants.
/// Output = `(x - mean) * inv_std * gamma + beta` — linear in x.
pub(crate) fn norm_affine_output_bounds(
    mean_const: f32,
    var_const: f32,
    eps_const: f32,
    gamma_const: f32,
    beta_const: f32,
    x_lower: f64,
    x_upper: f64,
) -> Result<(f64, f64), VerifyError> {
    for (index, val) in [
        (1, mean_const),
        (2, var_const),
        (3, eps_const),
        (4, gamma_const),
        (5, beta_const),
    ] {
        if !val.is_finite() {
            return Err(SmtError::NonFiniteConstantParam {
                index,
                value: f64::from(val),
            }
            .into());
        }
    }

    let mean = f64::from(mean_const);
    let var = f64::from(var_const);
    let eps = f64::from(eps_const);
    let gamma = f64::from(gamma_const);
    let beta = f64::from(beta_const);

    // Guard: var + eps must be positive for sqrt to produce a finite result.
    let denom = var + eps;
    if denom <= 0.0 || !denom.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: f64::NAN,
            upper: f64::NAN,
        }
        .into());
    }

    let inv_std = 1.0 / denom.sqrt();
    // output(x) = (x - mean) * inv_std * gamma + beta — linear in x
    let slope = inv_std * gamma;
    let intercept = -mean * inv_std * gamma + beta;
    linear_norm_bounds(x_lower, x_upper, slope, intercept)
}

/// Compute analytical output bounds for AdaIN: `gamma * (x - mu) * rsqrt(var_val + eps) + beta`.
///
/// **#448/#459 convention:** param 0 (x) is the symbolic variable bounded by
/// `[x_lower, x_upper]`, params 1-5 (mu, var_val, gamma, beta, eps) are constants.
/// AdaIN param order: `(x, mu, var_val, gamma, beta, eps)`.
/// Output = `gamma * (x - mu) / sqrt(var + eps) + beta` — linear in x.
pub(crate) fn adain_output_bounds(
    mu_const: f32,
    var_const: f32,
    gamma_const: f32,
    beta_const: f32,
    eps_const: f32,
    x_lower: f64,
    x_upper: f64,
) -> Result<(f64, f64), VerifyError> {
    for (index, val) in [
        (1, mu_const),
        (2, var_const),
        (3, gamma_const),
        (4, beta_const),
        (5, eps_const),
    ] {
        if !val.is_finite() {
            return Err(SmtError::NonFiniteConstantParam {
                index,
                value: f64::from(val),
            }
            .into());
        }
    }

    let mu = f64::from(mu_const);
    let var = f64::from(var_const);
    let gamma = f64::from(gamma_const);
    let beta = f64::from(beta_const);
    let eps = f64::from(eps_const);

    // Guard: var + eps must be positive for sqrt to be real.
    let denom = var + eps;
    if denom <= 0.0 || !denom.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: f64::NAN,
            upper: f64::NAN,
        }
        .into());
    }

    let inv_std = 1.0 / denom.sqrt();
    // output(x) = gamma * (x - mu) * inv_std + beta — linear in x
    let slope = gamma * inv_std;
    let intercept = -mu * gamma * inv_std + beta;
    linear_norm_bounds(x_lower, x_upper, slope, intercept)
}

/// Compute analytical output bounds for InstanceNorm: `(x - mean) * rsqrt(var_val + eps)`.
///
/// **#448/#459 convention:** param 0 (x) is the symbolic variable bounded by
/// `[x_lower, x_upper]`, params 1-3 (mean, var_val, eps) are constants.
/// Output = `(x - mean) / sqrt(var + eps)` — linear in x.
pub(crate) fn instance_norm_output_bounds(
    mean_const: f32,
    var_const: f32,
    eps_const: f32,
    x_lower: f64,
    x_upper: f64,
) -> Result<(f64, f64), VerifyError> {
    for (index, val) in [(1, mean_const), (2, var_const), (3, eps_const)] {
        if !val.is_finite() {
            return Err(SmtError::NonFiniteConstantParam {
                index,
                value: f64::from(val),
            }
            .into());
        }
    }

    let mean = f64::from(mean_const);
    let var = f64::from(var_const);
    let eps = f64::from(eps_const);

    // Guard: var + eps must be positive for sqrt to be real.
    let denom = var + eps;
    if denom <= 0.0 || !denom.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: f64::NAN,
            upper: f64::NAN,
        }
        .into());
    }

    let inv_std = 1.0 / denom.sqrt();
    // output(x) = (x - mean) * inv_std — linear in x
    let slope = inv_std;
    let intercept = -mean * inv_std;
    linear_norm_bounds(x_lower, x_upper, slope, intercept)
}

#[cfg(test)]
#[path = "norm_tests.rs"]
mod tests;
