// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-kernel analytical bounds wrapper functions.
//!
//! Extracted from `dispatch.rs` to stay under the 450-line limit (#2218).
//! Each function takes `(constant_params, input_lower, input_upper)` and returns
//! `Ok(Some((lo, hi)))` or `Ok(None)` to fall through to the heuristic.

use crate::error::VerifyError;

// `super` = the `dispatch` module; grandparent `bounds` holds sibling modules.
use crate::bounds::activation::{
    exp_output_bounds, gelu_output_bounds, leaky_relu_output_bounds, relu_output_bounds,
    sigmoid_output_bounds, silu_mul_output_bounds, softplus_output_bounds, tanh_output_bounds,
};
use crate::bounds::conv1d_k1::conv1d_k1_scalar_output_bounds;
use crate::bounds::norm::{
    adain_output_bounds, instance_norm_output_bounds, norm_affine_output_bounds,
    rms_norm_scalar_output_bounds,
};
use crate::bounds::rope::rope_output_bounds;
use crate::bounds::snake::snake_output_bounds;

/// Defense-in-depth: verify `cp` has at least `required` elements before
/// direct indexing. The caller (`compute_output_bounds_heuristic`) already
/// checks `min_constant_params`, but a local guard catches registry
/// misconfigurations at the function boundary rather than mid-computation.
pub(crate) fn require_params(
    cp: &[f32],
    required: usize,
    fn_name: &str,
) -> Result<(), VerifyError> {
    if cp.len() < required {
        return Err(VerifyError::InternalTranslationError {
            context: format!(
                "{fn_name}: expected >= {required} constant params, got {}",
                cp.len()
            ),
        });
    }
    Ok(())
}

pub(crate) fn bounds_snake(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 1, "bounds_snake")?;
    let alpha = f64::from(cp[0]);
    if alpha > 0.0 {
        let (out_lo, out_hi) = snake_output_bounds(f64::from(lo), f64::from(hi), alpha)?;
        Ok(Some((out_lo, out_hi)))
    } else {
        Ok(None) // Non-positive alpha: fall through to heuristic.
    }
}

pub(crate) fn bounds_silu_mul(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 1, "bounds_silu_mul")?;
    let up_const = f64::from(cp[0]);
    silu_mul_output_bounds(up_const, f64::from(lo), f64::from(hi)).map(Some)
}

pub(crate) fn bounds_rope_cos(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 2, "bounds_rope_cos")?;
    rope_output_bounds(cp[0], cp[1], lo, hi, nn_dsl::rope_cos_scalar_bounds).map(Some)
}

pub(crate) fn bounds_rope_sin(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 2, "bounds_rope_sin")?;
    rope_output_bounds(cp[0], cp[1], lo, hi, nn_dsl::rope_sin_scalar_bounds).map(Some)
}

pub(crate) fn bounds_rms_norm_scalar(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 2, "bounds_rms_norm_scalar")?;
    rms_norm_scalar_output_bounds(cp[0], cp[1], f64::from(lo), f64::from(hi)).map(Some)
}

pub(crate) fn bounds_norm_affine(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 5, "bounds_norm_affine")?;
    norm_affine_output_bounds(
        cp[0],
        cp[1],
        cp[2],
        cp[3],
        cp[4],
        f64::from(lo),
        f64::from(hi),
    )
    .map(Some)
}

pub(crate) fn bounds_instance_norm(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 3, "bounds_instance_norm")?;
    instance_norm_output_bounds(cp[0], cp[1], cp[2], f64::from(lo), f64::from(hi)).map(Some)
}

pub(crate) fn bounds_adain(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 5, "bounds_adain")?;
    adain_output_bounds(
        cp[0],
        cp[1],
        cp[2],
        cp[3],
        cp[4],
        f64::from(lo),
        f64::from(hi),
    )
    .map(Some)
}

pub(crate) fn bounds_adain_snake(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 6, "bounds_adain_snake")?;
    let lo_f64 = f64::from(lo);
    let hi_f64 = f64::from(hi);
    // adain_output_bounds expects (mu, var, gamma, beta, eps):
    let (adain_lo, adain_hi) = adain_output_bounds(
        cp[0], // mu
        cp[1], // var_val
        cp[2], // gamma
        cp[3], // beta
        cp[5], // eps (param 6 = cp[5], not cp[4])
        lo_f64, hi_f64,
    )?;
    let alpha = f64::from(cp[4]); // alpha (param 5 = cp[4])
    let alpha_clamped = alpha.max(f64::from(nn_dsl::snake::SNAKE_MIN_ALPHA));
    let (out_lo, out_hi) = snake_output_bounds(adain_lo, adain_hi, alpha_clamped)?;
    Ok(Some((out_lo, out_hi)))
}

pub(crate) fn bounds_gelu(
    _cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    gelu_output_bounds(f64::from(lo), f64::from(hi)).map(Some)
}

pub(crate) fn bounds_sigmoid(
    _cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    sigmoid_output_bounds(f64::from(lo), f64::from(hi)).map(Some)
}

pub(crate) fn bounds_relu(
    _cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    relu_output_bounds(f64::from(lo), f64::from(hi)).map(Some)
}

pub(crate) fn bounds_tanh_act(
    _cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    tanh_output_bounds(f64::from(lo), f64::from(hi)).map(Some)
}

pub(crate) fn bounds_leaky_relu(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 1, "bounds_leaky_relu")?;
    leaky_relu_output_bounds(f64::from(cp[0]), f64::from(lo), f64::from(hi)).map(Some)
}

pub(crate) fn bounds_exp(_cp: &[f32], lo: f32, hi: f32) -> Result<Option<(f64, f64)>, VerifyError> {
    exp_output_bounds(f64::from(lo), f64::from(hi)).map(Some)
}

pub(crate) fn bounds_softplus(
    _cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    softplus_output_bounds(f64::from(lo), f64::from(hi)).map(Some)
}

/// Binary add: `f(x) = x + c`. Bounds: `[lo + c, hi + c]`.
/// Part of #2917: ay coverage for elementwise add ops.
pub(crate) fn bounds_binary_add(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 1, "bounds_binary_add")?;
    let c = f64::from(cp[0]);
    let lo_f64 = f64::from(lo);
    let hi_f64 = f64::from(hi);
    let out_lo = lo_f64 + c;
    let out_hi = hi_f64 + c;
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return Err(VerifyError::UnsupportedOp(
            "bounds_binary_add: output bounds overflow".into(),
        ));
    }
    Ok(Some((out_lo, out_hi)))
}

/// Binary mul: `f(x) = x * c`. Bounds depend on sign of `c`:
///  - `c >= 0`: `[lo * c, hi * c]`
///  - `c < 0`: `[hi * c, lo * c]` (multiplication flips interval)
///
/// Part of #2917: ay coverage for elementwise mul and conv1d-k1.
pub(crate) fn bounds_binary_mul(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 1, "bounds_binary_mul")?;
    let c = f64::from(cp[0]);
    let lo_f64 = f64::from(lo);
    let hi_f64 = f64::from(hi);
    let (out_lo, out_hi) = if c >= 0.0 {
        (lo_f64 * c, hi_f64 * c)
    } else {
        (hi_f64 * c, lo_f64 * c)
    };
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return Err(VerifyError::UnsupportedOp(
            "bounds_binary_mul: output bounds overflow".into(),
        ));
    }
    Ok(Some((out_lo, out_hi)))
}

/// Conv1d kernel_size=1 scalar: `f(x) = x * weight + bias`.
/// Delegates to `conv1d_k1_scalar_output_bounds`.
/// Part of #2917: ay coverage for conv1d-k1 layers in Kokoro.
pub(crate) fn bounds_conv1d_k1_scalar(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 2, "bounds_conv1d_k1_scalar")?;
    let weight = f64::from(cp[0]);
    let bias = f64::from(cp[1]);
    conv1d_k1_scalar_output_bounds(weight, bias, f64::from(lo), f64::from(hi)).map(Some)
}

/// Fused AdaIN+LeakyReLU: `leaky_relu(adain(x, mu, var, gamma, beta, eps), slope)`.
///
/// **#448 convention:** param 0 (x) is the symbolic variable. constant_params:
/// `[mu, var_val, gamma, beta, slope, eps]` (6 params).
/// Composes `adain_output_bounds` → `leaky_relu_output_bounds`.
/// Part of #2218: BOUNDS_REGISTRY coverage for Kokoro fused kernels.
pub(crate) fn bounds_adain_leaky_relu(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 6, "bounds_adain_leaky_relu")?;
    // adain params: mu=cp[0], var_val=cp[1], gamma=cp[2], beta=cp[3], eps=cp[5]
    let (adain_lo, adain_hi) = adain_output_bounds(
        cp[0], // mu
        cp[1], // var_val
        cp[2], // gamma
        cp[3], // beta
        cp[5], // eps (param 7 = cp[5], same order as adain_snake)
        f64::from(lo),
        f64::from(hi),
    )?;
    // leaky_relu on the adain output interval
    let slope = f64::from(cp[4]); // slope (param 6 = cp[4])
    leaky_relu_output_bounds(slope, adain_lo, adain_hi).map(Some)
}

/// Fused AdaLayerNorm: `(1 + gamma) * layer_norm(x) + beta`.
///
/// **#448 convention:** param 0 (x) is the symbolic variable. constant_params:
/// `[mean, var_val, eps, norm_weight, norm_bias, gamma, beta]` (7 params).
/// Composes `norm_affine_output_bounds` (LayerNorm) → adaptive affine (linear).
/// Both stages are linear in x, so the composition is exact (no over-approximation).
/// Part of #2218: BOUNDS_REGISTRY coverage for Kokoro fused kernels.
pub(crate) fn bounds_ada_layer_norm(
    cp: &[f32],
    lo: f32,
    hi: f32,
) -> Result<Option<(f64, f64)>, VerifyError> {
    require_params(cp, 7, "bounds_ada_layer_norm")?;
    // norm_affine params: mean=cp[0], var=cp[1], eps=cp[2], gamma=cp[3], beta=cp[4]
    let (norm_lo, norm_hi) = norm_affine_output_bounds(
        cp[0], // mean
        cp[1], // var_val
        cp[2], // eps
        cp[3], // norm_weight
        cp[4], // norm_bias
        f64::from(lo),
        f64::from(hi),
    )?;
    // adaptive affine: output = (1 + gamma) * normed + beta — linear in normed
    let gamma = f64::from(cp[5]);
    let beta = f64::from(cp[6]);
    let scale = 1.0 + gamma;
    let a = scale * norm_lo + beta;
    let b = scale * norm_hi + beta;
    let (out_lo, out_hi) = if a <= b { (a, b) } else { (b, a) };
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return Err(VerifyError::UnsupportedOp(
            "bounds_ada_layer_norm: output bounds overflow".into(),
        ));
    }
    Ok(Some((out_lo, out_hi)))
}
