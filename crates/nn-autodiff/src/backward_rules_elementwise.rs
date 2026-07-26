// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward rules for elementwise activation and math operations.
//!
//! Split from `backward_rules.rs` (#1544 D9) for 500-line compliance.
//! Contains backward_activation (Relu, Tanh, Sigmoid, Exp, Log, Sqrt, Sqr,
//! Gelu, Silu, Neg, Abs) and backward_elementwise_math (Sin, Cos, Recip,
//! Powf, Clamp, Elu).

use nn_core::dyn_tensor::DynTensor;

use crate::error::Result;
use crate::grad::GradStore;
use crate::op::Op;

use super::accumulate;

/// Backward rules for unary activation functions.
pub(super) fn backward_activation(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::Relu(x) => {
            let mask = x.tensor().ge(0.0)?;
            let zeros = DynTensor::zeros(grad.dims(), grad.dtype(), &grad.device())?;
            accumulate(x, &mask.where_cond(grad, &zeros)?, grads)
        }
        Op::Tanh(x) => {
            let one_minus_sq = x.tensor().tanh()?.sqr()?.affine(-1.0, 1.0)?;
            accumulate(x, &grad.mul(&one_minus_sq)?, grads)
        }
        Op::Sigmoid(x) => {
            let sig = x.tensor().sigmoid()?;
            let dsig = sig.mul(&sig.affine(-1.0, 1.0)?)?;
            accumulate(x, &grad.mul(&dsig)?, grads)
        }
        Op::Exp(x) => accumulate(x, &grad.mul(&x.tensor().exp()?)?, grads),
        Op::Log(x) => accumulate(x, &grad.div(x.tensor())?, grads),
        Op::Sqrt(x) => {
            // d/dx sqrt(x) = 1 / (2 * sqrt(x)).
            // At x=0: sqrt(0)=0, so 1/(2*0) = Inf — DynTensor::div rejects
            // this.  Clamp denominator away from zero before division, then
            // mask the result to zero for x<=0 (subderivative convention).
            let two_sqrt = x.tensor().sqrt()?.affine(2.0, 0.0)?;
            let safe_denom = two_sqrt.clamp_min(1e-30)?;
            let raw_grad = grad.div(&safe_denom)?;
            let mask = x.tensor().gt(0.0)?;
            let zeros = DynTensor::zeros(grad.dims(), grad.dtype(), &grad.device())?;
            accumulate(x, &mask.where_cond(&raw_grad, &zeros)?, grads)
        }
        Op::Sqr(x) => accumulate(x, &grad.mul(&x.tensor().affine(2.0, 0.0)?)?, grads),
        Op::Gelu(x) => {
            // GELU (tanh approx): g(x) = 0.5 * x * (1 + tanh(s))
            // where s = sqrt(2/pi) * (x + 0.044715 * x^3)
            // d/dx = 0.5*(1+tanh(s)) + 0.5*x*(1-tanh(s)^2)*s'
            // s' = sqrt(2/pi) * (1 + 3*0.044715*x^2)
            let x_val = x.tensor();
            let sqrt_2_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
            let x_cubed = x_val.mul(&x_val.sqr()?)?;
            let inner = x_val.add(&x_cubed.affine(0.044715, 0.0)?)?;
            let s = inner.affine(sqrt_2_pi, 0.0)?;
            let tanh_s = s.tanh()?;
            let term1 = tanh_s.affine(0.5, 0.5)?;
            let s_prime = x_val
                .sqr()?
                .affine(3.0 * 0.044715, 1.0)?
                .affine(sqrt_2_pi, 0.0)?;
            let sech2 = tanh_s.sqr()?.affine(-1.0, 1.0)?;
            let term2 = x_val.mul(&sech2)?.mul(&s_prime)?.affine(0.5, 0.0)?;
            let dgelu = term1.add(&term2)?;
            accumulate(x, &grad.mul(&dgelu)?, grads)
        }
        Op::GeluErf(x) => {
            // Exact GELU (erf-based): gelu_erf(x) = x * 0.5 * (1 + erf(x/sqrt(2)))
            // d/dx = 0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
            //      = Phi(x) + x * phi(x)
            // where Phi = standard normal CDF, phi = standard normal PDF.
            //
            // Phi(x) = gelu_erf(x) / x, but x=0 gives 0/0.
            // Strategy: use where_cond to select safe denominator for |x| < eps,
            // then override Phi result to 0.5 at the origin.
            let x_val = x.tensor();
            let inv_sqrt_2pi = 1.0_f64 / (2.0 * std::f64::consts::PI).sqrt();

            // phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)  (standard normal PDF)
            let neg_half_x_sq = x_val.sqr()?.affine(-0.5, 0.0)?;
            let phi = neg_half_x_sq.exp()?.mul_scalar(inv_sqrt_2pi)?;

            // Compute Phi(x) = gelu_erf(x) / x with zero-safe division.
            // For |x| < eps, substitute denominator with 1.0 (arbitrary nonzero)
            // then override the result with Phi(0) = 0.5 via masking.
            let eps = 1e-7_f64;
            let abs_x = x_val.abs()?;
            let is_near_zero = abs_x.lt(eps)?;
            let ones = DynTensor::ones(x_val.dims(), grad.dtype(), &grad.device())?;
            let safe_denom = is_near_zero.where_cond(&ones, x_val)?;
            let gelu_erf_val = x_val.gelu_erf()?;
            let phi_cdf_raw = gelu_erf_val.div(&safe_denom)?;
            let half = DynTensor::full(x_val.dims(), 0.5, grad.dtype(), &grad.device())?;
            let phi_cdf = is_near_zero.where_cond(&half, &phi_cdf_raw)?;

            // d/dx = Phi(x) + x * phi(x)
            let x_phi = x_val.mul(&phi)?;
            let dgelu_erf = phi_cdf.add(&x_phi)?;
            accumulate(x, &grad.mul(&dgelu_erf)?, grads)
        }
        Op::Silu(x) => {
            // SiLU(x) = x * sigmoid(x)
            // d/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            let sig = x.tensor().sigmoid()?;
            let one_minus_sig = sig.affine(-1.0, 1.0)?;
            let x_oms = x.tensor().mul(&one_minus_sig)?;
            let dsilu = sig.mul(&x_oms.affine(1.0, 1.0)?)?;
            accumulate(x, &grad.mul(&dsilu)?, grads)
        }
        Op::Neg(x) => accumulate(x, &grad.neg()?, grads),
        Op::Abs(x) => {
            // Use gt/lt (not ge/lt) so abs'(0) = 0, matching PyTorch sign(0)=0.
            let pos = x.tensor().gt(0.0)?;
            let neg = x.tensor().lt(0.0)?;
            let zeros = DynTensor::zeros(grad.dims(), grad.dtype(), &grad.device())?;
            let g = pos
                .where_cond(grad, &zeros)?
                .add(&neg.where_cond(&grad.neg()?, &zeros)?)?;
            accumulate(x, &g, grads)
        }
        other => Err(super::unsupported(other)),
    }
}

/// Backward rules for new activation functions.
///
/// HardSigmoid, HardSwish, Mish, Selu, Softplus, Celu.
pub(super) fn backward_new_activations(
    op: &Op,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    match op {
        Op::HardSigmoid(x) => {
            // d/dx HardSigmoid = 1/6 for x in (-3, 3), 0 otherwise
            let in_range_lo = x.tensor().gt(-3.0)?;
            let in_range_hi = x.tensor().lt(3.0)?;
            let zeros = DynTensor::zeros(grad.dims(), grad.dtype(), &grad.device())?;
            let sixth = DynTensor::full(grad.dims(), 1.0 / 6.0, grad.dtype(), &grad.device())?;
            let deriv = in_range_lo.where_cond(&in_range_hi.where_cond(&sixth, &zeros)?, &zeros)?;
            accumulate(x, &grad.mul(&deriv)?, grads)
        }
        Op::HardSwish(x) => {
            // HardSwish(x) = x * HardSigmoid(x) = x * clamp(x/6 + 0.5, 0, 1)
            // For x <= -3: f(x) = 0, f'(x) = 0
            // For -3 < x < 3: f(x) = x*(x/6+0.5) = x^2/6 + x/2, f'(x) = x/3 + 0.5
            // For x >= 3: f(x) = x, f'(x) = 1
            let x_val = x.tensor();
            let below = x_val.le(-3.0)?;
            let above = x_val.ge(3.0)?;
            let zeros = DynTensor::zeros(grad.dims(), grad.dtype(), &grad.device())?;
            let ones = DynTensor::ones(grad.dims(), grad.dtype(), &grad.device())?;
            // In-range derivative: x/3 + 0.5
            let mid_deriv = x_val.affine(1.0 / 3.0, 0.5)?;
            // below -> 0, above -> 1, else -> x/3 + 0.5
            let deriv = below.where_cond(&zeros, &above.where_cond(&ones, &mid_deriv)?)?;
            accumulate(x, &grad.mul(&deriv)?, grads)
        }
        Op::Mish(x) => {
            // Mish(x) = x * tanh(softplus(x))
            // d(mish)/dx = t + x * (1 - t^2) * sigmoid(x)
            // where t = tanh(softplus(x)), sigmoid(x) = d(softplus)/dx
            let x_val = x.tensor();
            let sp = x_val.softplus()?;
            let t = sp.tanh()?;
            let sig = x_val.sigmoid()?;
            let t_sq = t.sqr()?;
            let one_minus_t_sq = t_sq.affine(-1.0, 1.0)?;
            let second_term = x_val.mul(&one_minus_t_sq)?.mul(&sig)?;
            let deriv = t.add(&second_term)?;
            accumulate(x, &grad.mul(&deriv)?, grads)
        }
        Op::Selu(x) => {
            // d/dx = lambda for x >= 0, lambda * alpha * exp(x) for x < 0
            const SELU_ALPHA: f64 = 1.6732632423543772;
            const SELU_LAMBDA: f64 = 1.0507009873554805;
            let pos_mask = x.tensor().ge(0.0)?;
            let lambda_t = DynTensor::full(grad.dims(), SELU_LAMBDA, grad.dtype(), &grad.device())?;
            let neg_deriv = x.tensor().exp()?.mul_scalar(SELU_LAMBDA * SELU_ALPHA)?;
            let deriv = pos_mask.where_cond(&lambda_t, &neg_deriv)?;
            accumulate(x, &grad.mul(&deriv)?, grads)
        }
        Op::Softplus(x) => {
            // d/dx softplus(x) = sigmoid(x) = exp(x) / (1 + exp(x))
            let sig = x.tensor().sigmoid()?;
            accumulate(x, &grad.mul(&sig)?, grads)
        }
        Op::Celu(x, alpha) => {
            // d/dx = 1 for x >= 0, exp(x/alpha) for x < 0
            let pos_mask = x.tensor().ge(0.0)?;
            let ones = DynTensor::ones(grad.dims(), grad.dtype(), &grad.device())?;
            let neg_deriv = x.tensor().mul_scalar(1.0 / alpha)?.exp()?;
            let deriv = pos_mask.where_cond(&ones, &neg_deriv)?;
            accumulate(x, &grad.mul(&deriv)?, grads)
        }
        other => Err(super::unsupported(other)),
    }
}

/// Backward rules for elementwise math ops (sin, cos, recip, powf, clamp, elu).
pub(super) fn backward_elementwise_math(
    op: &Op,
    grad: &DynTensor,
    grads: &mut GradStore,
) -> Result<()> {
    match op {
        Op::Sin(x) => accumulate(x, &grad.mul(&x.tensor().cos()?)?, grads),
        Op::Cos(x) => accumulate(x, &grad.mul(&x.tensor().sin()?.neg()?)?, grads),
        Op::Recip(x) => {
            let neg_inv_sq = x.tensor().sqr()?.recip()?.neg()?;
            accumulate(x, &grad.mul(&neg_inv_sq)?, grads)
        }
        Op::Powf(x, p) => {
            if *p == 0.0 {
                // x^0 = 1 (constant), gradient is zero everywhere.
                // The general formula p * x^(p-1) computes 0 * x^(-1) which is
                // NaN at x=0 (0 * Inf = NaN). Short-circuit to avoid this.
                let zeros = DynTensor::zeros(grad.dims(), grad.dtype(), &grad.device())?;
                accumulate(x, &zeros, grads)
            } else {
                let dpower = x.tensor().powf(*p - 1.0)?.mul_scalar(*p)?;
                accumulate(x, &grad.mul(&dpower)?, grads)
            }
        }
        Op::Clamp(x, lo, hi) => {
            // Use ge/le (not gt/lt) so gradient flows at the boundary values.
            // Matches PyTorch: clamp'(x) = 1 when lo <= x <= hi, 0 otherwise.
            let above_lo = x.tensor().ge(*lo)?;
            let below_hi = x.tensor().le(*hi)?;
            let zeros = DynTensor::zeros(grad.dims(), grad.dtype(), &grad.device())?;
            let mask = above_lo.where_cond(&below_hi.where_cond(grad, &zeros)?, &zeros)?;
            accumulate(x, &mask, grads)
        }
        Op::Elu(x, alpha) => {
            // ELU'(x) = 1 if x > 0, alpha * exp(x) if x <= 0
            let pos_mask = x.tensor().gt(0.0)?;
            let ones = DynTensor::ones(grad.dims(), grad.dtype(), &grad.device())?;
            let neg_deriv = x.tensor().exp()?.mul_scalar(*alpha)?;
            let deriv = pos_mask.where_cond(&ones, &neg_deriv)?;
            accumulate(x, &grad.mul(&deriv)?, grads)
        }
        other => Err(super::unsupported(other)),
    }
}
