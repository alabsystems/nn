// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Gradient clipping utilities for training stability.
//!
//! Provides two standard clipping strategies matching PyTorch's API:
//! - [`clip_grad_norm`]: Clips total gradient L2 norm across all parameters.
//! - [`clip_grad_value`]: Clamps each gradient element to a symmetric range.
//!
//! # Example
//! ```no_run
//! use nn_optim::grad_clip::{clip_grad_norm, clip_grad_value};
//! # use nn_autodiff::GradStore;
//!
//! # fn example(grads: &mut GradStore) -> nn_optim::Result<()> {
//! // Clip total gradient norm to 1.0 (prevents gradient explosion)
//! let total_norm = clip_grad_norm(grads, 1.0)?;
//! // total_norm contains the original (unclipped) L2 norm
//! let _ = total_norm;
//!
//! // Or clip each gradient element to [-0.5, 0.5]
//! clip_grad_value(grads, 0.5)?;
//! # Ok(())
//! # }
//! ```

use crate::error::{OptimError, Result};
use nn_autodiff::GradStore;

/// Clips total gradient L2 norm across all variable parameters.
///
/// Computes the total L2 norm of all variable gradients in `grads`. If the
/// total norm exceeds `max_norm`, all gradients are scaled down by
/// `max_norm / total_norm` so the total norm becomes `max_norm`.
///
/// Returns the original (unclipped) total norm.
///
/// Matches PyTorch's `torch.nn.utils.clip_grad_norm_` behavior.
///
/// # Errors
///
/// Returns `InvalidParam` if `max_norm` is not positive or not finite.
pub fn clip_grad_norm(grads: &mut GradStore, max_norm: f64) -> Result<f64> {
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Err(OptimError::InvalidParam {
            param: "max_norm",
            reason: format!("must be positive and finite, got {max_norm}"),
        });
    }

    // Compute total L2 norm: sqrt(sum of squared elements across all gradients)
    // Uses device-agnostic ops (sqr + sum_all) so GPU gradients stay on device.
    let mut total_norm_sq: f64 = 0.0;
    for (_id, grad) in grads.var_grads() {
        let grad_norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
        total_norm_sq += f64::from(grad_norm_sq);
    }
    let total_norm = total_norm_sq.sqrt();

    // Scale gradients if total norm exceeds max_norm
    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for (_id, grad) in grads.var_grads_mut() {
            *grad = grad.mul_scalar(scale)?;
        }
    }

    Ok(total_norm)
}

/// Clamps each gradient element to the symmetric range `[-clip_value, clip_value]`.
///
/// Matches PyTorch's `torch.nn.utils.clip_grad_value_` behavior.
///
/// # Errors
///
/// Returns `InvalidParam` if `clip_value` is not positive or not finite.
pub fn clip_grad_value(grads: &mut GradStore, clip_value: f64) -> Result<()> {
    if !clip_value.is_finite() || clip_value <= 0.0 {
        return Err(OptimError::InvalidParam {
            param: "clip_value",
            reason: format!("must be positive and finite, got {clip_value}"),
        });
    }

    for (_id, grad) in grads.var_grads_mut() {
        *grad = grad.clamp(-clip_value, clip_value)?;
    }

    Ok(())
}

#[cfg(test)]
#[path = "grad_clip_tests.rs"]
mod tests;
