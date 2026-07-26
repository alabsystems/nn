// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Analytical output bounds for activation kernels.
//!
//! Covers: SiLU-Mul, Sigmoid, ReLU, Tanh, GELU, LeakyReLU, Exp, Softplus.
//!
//! Extracted from `ay/prove_bounds_activation.rs` (#859) to be always-available
//! without the `ay-smt` feature flag. Pure Rust math — no ay-bindings dependency.

use crate::error::VerifyError;
use crate::smt_error::SmtError;

/// Compute analytical output bounds for SiLU-Mul: `silu(x) * up`.
///
/// **#448 convention:** param 0 (x) is the symbolic variable bounded by
/// `[x_lower, x_upper]`, param 1 (up) is constant.
/// Output = `silu(x) * up_const`.
///
/// **silu is NOT monotonically increasing.** It has a global minimum at
/// `x ≈ -1.278` where `silu ≈ -0.278`. For `x < -1.278`, silu decreases
/// toward 0 as x → -∞. We must evaluate at both endpoints AND at the
/// global minimum when the input range spans it. Matches the algorithm
/// in `nn_dsl::silu_mul_scalar_bounds`.
pub(crate) fn silu_mul_output_bounds(
    up_const: f64,
    x_lower: f64,
    x_upper: f64,
) -> Result<(f64, f64), VerifyError> {
    if !up_const.is_finite() {
        return Err(SmtError::NonFiniteConstantParam {
            index: 1,
            value: up_const,
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

    fn silu_f64(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    // x-coordinate of the SiLU global minimum (matches `nn_dsl::silu_mul::SILU_ARGMIN`).
    const SILU_ARGMIN_F64: f64 = -1.278_464_5_f64;

    // Evaluate silu at both endpoints.
    let silu_at_lo = silu_f64(x_lower);
    let silu_at_hi = silu_f64(x_upper);

    // Find the range of silu values over [x_lower, x_upper].
    let mut silu_min = silu_at_lo.min(silu_at_hi);
    let mut silu_max = silu_at_lo.max(silu_at_hi);

    // If the input range spans the global minimum, include it.
    if x_lower < SILU_ARGMIN_F64 && x_upper > SILU_ARGMIN_F64 {
        let silu_at_argmin = silu_f64(SILU_ARGMIN_F64);
        silu_min = silu_min.min(silu_at_argmin);
        silu_max = silu_max.max(silu_at_argmin);
    }

    // output = silu(x) * up_const. Since up_const is a single value,
    // multiply both extrema and take min/max (sign flip if up_const < 0).
    let a = silu_min * up_const;
    let b = silu_max * up_const;
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

/// Compute analytical output bounds for sigmoid.
///
/// `sigmoid(x) = 1 / (1 + exp(-x))`
///
/// Sigmoid is **monotonically increasing** — bounds are simply
/// `(sigmoid(x_lower), sigmoid(x_upper))`. No global minimum handling needed.
/// Matches the algorithm in `nn_dsl::sigmoid_scalar_bounds`.
pub(crate) fn sigmoid_output_bounds(x_lower: f64, x_upper: f64) -> Result<(f64, f64), VerifyError> {
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

    fn sigmoid_f64(x: f64) -> f64 {
        let raw = 1.0 / (1.0 + (-x).exp());
        // Sigmoid is strictly in (0, 1) for all finite x. Clamp to avoid
        // f64 boundary collapse where (-x).exp() underflows to 0.0 for
        // |x| > ~36.7, producing exactly 0.0 or 1.0. (#685)
        raw.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    }

    // Sigmoid is monotonically increasing: lower bound = sigmoid(x_lower),
    // upper bound = sigmoid(x_upper).
    let lower = sigmoid_f64(x_lower);
    let upper = sigmoid_f64(x_upper);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(SmtError::NonFiniteBound { lower, upper }.into());
    }

    Ok((lower, upper))
}

/// Compute analytical output bounds for ReLU.
///
/// `relu(x) = max(x, 0)`
///
/// ReLU is **monotonically increasing** — bounds are simply
/// `(max(x_lower, 0), max(x_upper, 0))`. No global minimum handling needed.
/// Matches the algorithm in `nn_dsl::relu_scalar_bounds`.
pub(crate) fn relu_output_bounds(x_lower: f64, x_upper: f64) -> Result<(f64, f64), VerifyError> {
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

    let lower = x_lower.max(0.0);
    let upper = x_upper.max(0.0);

    Ok((lower, upper))
}

/// Compute analytical output bounds for tanh.
///
/// `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)`
///
/// Tanh is **monotonically increasing** with output ∈ (-1, 1) — bounds are
/// simply `(tanh(x_lower), tanh(x_upper))`. No global minimum handling needed.
/// Matches the algorithm in `nn_dsl::tanh_scalar_bounds`.
pub(crate) fn tanh_output_bounds(x_lower: f64, x_upper: f64) -> Result<(f64, f64), VerifyError> {
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

    let lower = x_lower.tanh();
    let upper = x_upper.tanh();

    if !lower.is_finite() || !upper.is_finite() {
        return Err(SmtError::NonFiniteBound { lower, upper }.into());
    }

    Ok((lower, upper))
}

/// Compute analytical output bounds for GELU (tanh approximation).
///
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// GELU is **not** monotonically increasing — it has a global minimum at
/// `x ≈ -0.7523` where `gelu ≈ -0.1700`. For `x < GELU_ARGMIN`, gelu
/// increases back toward 0 as x → -∞. To get sound bounds, we evaluate at
/// both endpoints AND at the global minimum when the input range spans it.
/// Matches the algorithm in `nn_dsl::gelu_scalar_bounds`.
pub(crate) fn gelu_output_bounds(x_lower: f64, x_upper: f64) -> Result<(f64, f64), VerifyError> {
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

    fn gelu_f64(x: f64) -> f64 {
        let k: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)
        let inner = k * (x + 0.044715 * x * x * x);
        let e2 = (2.0 * inner).exp();
        0.5 * x * (2.0 - 2.0 / (e2 + 1.0))
    }

    // x-coordinate of the GELU global minimum (matches `nn_dsl::gelu::GELU_ARGMIN`).
    const GELU_ARGMIN_F64: f64 = -0.752_252_6_f64;

    let g_lo = gelu_f64(x_lower);
    let g_hi = gelu_f64(x_upper);

    let mut lower = g_lo.min(g_hi);
    let mut upper = g_lo.max(g_hi);

    if x_lower < GELU_ARGMIN_F64 && x_upper > GELU_ARGMIN_F64 {
        let g_min = gelu_f64(GELU_ARGMIN_F64);
        lower = lower.min(g_min);
        upper = upper.max(g_min);
    }

    if !lower.is_finite() || !upper.is_finite() {
        return Err(SmtError::NonFiniteBound { lower, upper }.into());
    }

    Ok((lower, upper))
}

/// Compute analytical output bounds for LeakyReLU.
///
/// `leaky_relu(x, alpha) = x if x >= 0, else alpha * x`
///
/// LeakyReLU is **piecewise linear and monotone** for `alpha >= 0`.
/// For alpha in [0, 1], both pieces are non-decreasing. For negative alpha
/// the function is NOT monotone — we evaluate all candidate extrema including
/// the kink at x=0.
pub(crate) fn leaky_relu_output_bounds(
    alpha: f64,
    x_lower: f64,
    x_upper: f64,
) -> Result<(f64, f64), VerifyError> {
    if !alpha.is_finite() {
        return Err(SmtError::NonFiniteConstantParam {
            index: 1,
            value: alpha,
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

    fn leaky_relu_f64(x: f64, alpha: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            alpha * x
        }
    }

    let a = leaky_relu_f64(x_lower, alpha);
    let b = leaky_relu_f64(x_upper, alpha);
    let mut out_lo = a.min(b);
    let mut out_hi = a.max(b);

    // For negative alpha, the kink at x=0 is a local extremum.
    if alpha < 0.0 && x_lower < 0.0 && x_upper > 0.0 {
        out_lo = out_lo.min(0.0);
        out_hi = out_hi.max(0.0);
    }

    if !out_lo.is_finite() || !out_hi.is_finite() {
        return Err(SmtError::NonFiniteBound {
            lower: out_lo,
            upper: out_hi,
        }
        .into());
    }

    Ok((out_lo, out_hi))
}

/// Compute analytical output bounds for exp.
///
/// `exp(x)` is **monotonically increasing**: `exp(x_lower)` to `exp(x_upper)`.
/// Output is always positive. For large inputs, exp overflows to infinity
/// — the output finiteness guard catches this.
pub(crate) fn exp_output_bounds(x_lower: f64, x_upper: f64) -> Result<(f64, f64), VerifyError> {
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

    let lower = x_lower.exp();
    let upper = x_upper.exp();

    if !lower.is_finite() || !upper.is_finite() {
        return Err(SmtError::NonFiniteBound { lower, upper }.into());
    }

    Ok((lower, upper))
}

/// Compute analytical output bounds for softplus.
///
/// `softplus(x) = ln(1 + exp(x))`
///
/// Softplus is **monotonically increasing**: bounds are
/// `(softplus(lower), softplus(upper))`. Output is always positive.
/// For large positive x, `softplus(x) ≈ x`. For large negative x,
/// `softplus(x) ≈ exp(x) ≈ 0`.
pub(crate) fn softplus_output_bounds(
    x_lower: f64,
    x_upper: f64,
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

    fn softplus_f64(x: f64) -> f64 {
        // Numerically stable: for large x, ln(1 + exp(x)) ≈ x.
        if x > 20.0 {
            x
        } else {
            x.exp().ln_1p()
        }
    }

    let lower = softplus_f64(x_lower);
    let upper = softplus_f64(x_upper);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(SmtError::NonFiniteBound { lower, upper }.into());
    }

    Ok((lower, upper))
}

#[cfg(test)]
#[path = "activation_tests.rs"]
mod tests;
