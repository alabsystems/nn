// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Gradient scaler for mixed-precision training.
//!
//! In mixed-precision training (bf16/f16 forward, f32 backward), small
//! gradients can underflow to zero. [`GradScaler`] multiplies the loss by a
//! large scale factor before `backward()`, then divides gradients by the
//! same factor before the optimizer step. If gradients contain inf/NaN
//! (from overflow), the step is skipped and the scale is reduced.
//!
//! # Example
//!
//! ```no_run
//! use nn_optim::{GradScaler, GradScalerConfig, AdamW, AdamConfig, Optimizer};
//! use nn_autodiff::{Var, TrackedTensor, backward};
//! use std::sync::Arc;
//!
//! # fn main() -> nn_optim::Result<()> {
//! # let var = Var::new(nn_core::dyn_tensor::DynTensor::from_vec(vec![1.0], &[1], &nn_core::Device::Cpu)?);
//! # let tracked = Arc::new(TrackedTensor::from_var(&var)?);
//! # let loss = tracked.sqr()?;
//! let mut scaler = GradScaler::new(GradScalerConfig::default())?;
//! let mut optimizer = AdamW::new(vec![var], AdamConfig::default())?;
//!
//! // Scale loss, backward, unscale, step
//! let scaled_loss = scaler.scale_loss(&loss)?;
//! let mut grads = backward(&scaled_loss)?;
//! if scaler.unscale_and_check(&mut grads)? {
//!     optimizer.step(&grads)?;
//! }
//! scaler.update();
//! # Ok(())
//! # }
//! ```

use crate::error::Result;
use nn_autodiff::{GradStore, TrackedTensor};
use std::sync::Arc;

/// Configuration for [`GradScaler`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GradScalerConfig {
    /// Initial scale factor. Default: 65536.0 (2^16).
    pub init_scale: f64,
    /// Factor to multiply scale by when no inf/NaN found. Default: 2.0.
    pub growth_factor: f64,
    /// Factor to multiply scale by when inf/NaN found. Default: 0.5.
    pub backoff_factor: f64,
    /// Number of consecutive non-inf/NaN steps before growing scale. Default: 2000.
    pub growth_interval: usize,
    /// Minimum allowed scale. Default: 1.0.
    pub min_scale: f64,
    /// Maximum allowed scale. Default: 2^24 = 16777216.0.
    pub max_scale: f64,
}

impl Default for GradScalerConfig {
    fn default() -> Self {
        Self {
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            min_scale: 1.0,
            max_scale: 16_777_216.0,
        }
    }
}

/// Gradient scaler for mixed-precision training.
///
/// Prevents gradient underflow by scaling the loss before backward pass,
/// then unscaling gradients before the optimizer step. Automatically
/// adjusts the scale factor: grows when gradients are healthy, shrinks
/// when inf/NaN is detected.
#[derive(Debug)]
pub struct GradScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    min_scale: f64,
    max_scale: f64,
    /// Consecutive steps without inf/NaN.
    growth_tracker: usize,
    /// Whether the last step found inf/NaN.
    found_inf: bool,
}

impl GradScaler {
    /// Create a new gradient scaler with the given configuration.
    ///
    /// Returns an error if the configuration is invalid (e.g., non-positive
    /// scale, growth_factor ≤ 1, backoff_factor ≥ 1, min > max).
    #[must_use = "scaler must be stored for mixed-precision training"]
    pub fn new(config: GradScalerConfig) -> Result<Self> {
        if config.init_scale <= 0.0 || !config.init_scale.is_finite() {
            return Err(crate::OptimError::InvalidParam {
                param: "init_scale",
                reason: format!("must be positive and finite, got {}", config.init_scale),
            });
        }
        if config.growth_factor <= 1.0 || !config.growth_factor.is_finite() {
            return Err(crate::OptimError::InvalidParam {
                param: "growth_factor",
                reason: format!("must be > 1.0 and finite, got {}", config.growth_factor),
            });
        }
        if config.backoff_factor <= 0.0
            || config.backoff_factor >= 1.0
            || !config.backoff_factor.is_finite()
        {
            return Err(crate::OptimError::InvalidParam {
                param: "backoff_factor",
                reason: format!(
                    "must be in (0, 1) and finite, got {}",
                    config.backoff_factor
                ),
            });
        }
        if config.min_scale <= 0.0 || !config.min_scale.is_finite() {
            return Err(crate::OptimError::InvalidParam {
                param: "min_scale",
                reason: format!("must be positive and finite, got {}", config.min_scale),
            });
        }
        if config.max_scale < config.min_scale || !config.max_scale.is_finite() {
            return Err(crate::OptimError::InvalidParam {
                param: "max_scale",
                reason: format!(
                    "must be >= min_scale and finite, got max={} min={}",
                    config.max_scale, config.min_scale
                ),
            });
        }
        if config.init_scale < config.min_scale || config.init_scale > config.max_scale {
            return Err(crate::OptimError::InvalidParam {
                param: "init_scale",
                reason: format!(
                    "must be in [min_scale, max_scale], got init={} min={} max={}",
                    config.init_scale, config.min_scale, config.max_scale
                ),
            });
        }
        if config.growth_interval == 0 {
            return Err(crate::OptimError::InvalidParam {
                param: "growth_interval",
                reason: "must be > 0".into(),
            });
        }
        Ok(Self {
            scale: config.init_scale,
            growth_factor: config.growth_factor,
            backoff_factor: config.backoff_factor,
            growth_interval: config.growth_interval,
            min_scale: config.min_scale,
            max_scale: config.max_scale,
            growth_tracker: 0,
            found_inf: false,
        })
    }

    /// Current scale factor.
    #[must_use]
    pub fn scale_factor(&self) -> f64 {
        self.scale
    }

    /// Whether the last `unscale_and_check` found inf/NaN.
    #[must_use]
    pub fn found_inf(&self) -> bool {
        self.found_inf
    }

    /// Export current state for checkpoint persistence.
    ///
    /// Returns a [`GradScalerState`] containing the current scale factor and
    /// growth tracker. Use with [`TrainingMetadata::grad_scaler`] for
    /// checkpoint save/load.
    #[must_use]
    pub fn save_state(&self) -> crate::checkpoint::GradScalerState {
        crate::checkpoint::GradScalerState {
            scale: self.scale,
            growth_tracker: self.growth_tracker,
        }
    }

    /// Restore state from a checkpoint.
    ///
    /// Validates that the restored scale is within `[min_scale, max_scale]`
    /// and is finite. The growth tracker is restored as-is (capped at
    /// `growth_interval` to prevent immediate growth on resume).
    ///
    /// # Errors
    ///
    /// Returns `InvalidParam` if the scale is not finite or outside bounds.
    pub fn load_state(&mut self, state: &crate::checkpoint::GradScalerState) -> Result<()> {
        if !state.scale.is_finite() || state.scale <= 0.0 {
            return Err(crate::OptimError::InvalidParam {
                param: "checkpoint_scale",
                reason: format!("must be positive and finite, got {}", state.scale),
            });
        }
        // Clamp to current min/max bounds (config may have changed between save and load)
        self.scale = state.scale.clamp(self.min_scale, self.max_scale);
        // Cap growth_tracker below growth_interval to prevent immediate growth on resume.
        // Using saturating_sub(1) ensures at least one clean step is required after load.
        self.growth_tracker = state
            .growth_tracker
            .min(self.growth_interval.saturating_sub(1));
        Ok(())
    }

    /// Scale a loss tensor by the current scale factor.
    ///
    /// Returns a new tracked tensor that is `loss * scale`. Call `backward()`
    /// on this scaled loss to get scaled gradients.
    pub fn scale_loss(&self, loss: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        Ok(loss.mul_scalar(self.scale)?)
    }

    /// Unscale gradients and check for inf/NaN.
    ///
    /// Divides all variable gradients by the scale factor. Returns `true` if
    /// gradients are finite (safe to call `optimizer.step()`), or `false` if
    /// any gradient contains inf/NaN (skip the step).
    pub fn unscale_and_check(&mut self, grads: &mut GradStore) -> Result<bool> {
        let inv_scale = 1.0 / self.scale;
        let mut has_inf_nan = false;

        for (_var_id, grad) in grads.var_grads_mut() {
            // Divide by scale factor
            *grad = grad.mul_scalar(inv_scale)?;

            // Check for inf/NaN in the unscaled gradient.
            // Uses any_non_finite() which scans the CPU ndarray slice or GPU
            // unified memory directly — no Vec allocation or GPU→CPU transfer.
            if !has_inf_nan && grad.any_non_finite()? {
                has_inf_nan = true;
            }
        }

        self.found_inf = has_inf_nan;
        Ok(!has_inf_nan)
    }

    /// Update the scale factor after a training step.
    ///
    /// Call this after every training step (regardless of whether the optimizer
    /// step was taken). If inf/NaN was found, the scale is reduced. If enough
    /// consecutive clean steps have passed, the scale is increased.
    pub fn update(&mut self) {
        if self.found_inf {
            self.scale = (self.scale * self.backoff_factor).max(self.min_scale);
            self.growth_tracker = 0;
        } else {
            self.growth_tracker += 1;
            if self.growth_tracker >= self.growth_interval {
                self.scale = (self.scale * self.growth_factor).min(self.max_scale);
                self.growth_tracker = 0;
            }
        }
    }
}

#[cfg(test)]
#[path = "grad_scaler_tests.rs"]
mod tests;
