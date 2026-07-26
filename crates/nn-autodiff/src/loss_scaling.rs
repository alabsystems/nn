// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dynamic loss scaling for mixed-precision training.
//!
//! In mixed-precision training (BF16/F16 forward, F32 backward), small
//! gradients can underflow to zero in reduced-precision representations.
//! [`DynamicLossScaler`] multiplies the loss by a large scale factor before
//! `backward()`, then divides gradients by the same factor. If gradients
//! overflow to inf/NaN, the scale is reduced; if training proceeds without
//! overflow for enough steps, the scale is increased.
//!
//! This complements the optimizer-side [`GradScaler`](nn_optim::GradScaler)
//! by providing a standalone, autodiff-crate-local implementation that does
//! not depend on `nn-optim`. The two can be used interchangeably; prefer
//! [`GradScaler`](nn_optim::GradScaler) when using nn-optim optimizers.
//!
//! # Example
//!
//! ```no_run
//! use nn_autodiff::loss_scaling::DynamicLossScaler;
//! use nn_core::dyn_tensor::DynTensor;
//! use nn_core::Device;
//!
//! let scaler = DynamicLossScaler::default();
//! let loss = DynTensor::from_vec(vec![0.5], &[1], &Device::Cpu).unwrap();
//! let scaled = scaler.scale_loss(&loss).unwrap();
//! // scaled == loss * 65536.0
//! ```

use crate::error::{AutodiffError, Result};
use nn_core::dyn_tensor::DynTensor;
use nn_core::DType;

/// Configuration for [`DynamicLossScaler`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MixedPrecisionConfig {
    /// Initial loss scale factor. Default: 65536.0 (2^16).
    pub loss_scale: f32,

    /// Dtype for gradient accumulation. Default: F32.
    ///
    /// Gradients are always accumulated in this dtype regardless of the
    /// forward pass dtype. This prevents precision loss when summing many
    /// small BF16/F16 gradient contributions.
    pub grad_dtype: DType,

    /// Factor to multiply scale by when no inf/NaN found. Default: 2.0.
    pub growth_factor: f32,

    /// Factor to multiply scale by when inf/NaN found. Default: 0.5.
    pub backoff_factor: f32,

    /// Number of consecutive non-inf/NaN steps before growing scale. Default: 2000.
    pub growth_interval: usize,
}

impl MixedPrecisionConfig {
    /// Create a new config with default settings (F32 grad accumulation).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config for BF16 forward / F32 backward training.
    #[must_use]
    pub fn bf16_training() -> Self {
        Self {
            loss_scale: 65536.0,
            grad_dtype: DType::F32,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
        }
    }

    /// Create a config for F16 forward / F32 backward training.
    #[must_use]
    pub fn f16_training() -> Self {
        // F16 has smaller dynamic range than BF16; start with higher scale
        // and reduce backoff interval to recover faster from overflows.
        Self {
            loss_scale: 131072.0, // 2^17
            grad_dtype: DType::F32,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 1000,
        }
    }
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            loss_scale: 65536.0,
            grad_dtype: DType::F32,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
        }
    }
}

/// Dynamic loss scaler for mixed-precision training.
///
/// Prevents gradient underflow by scaling the loss before the backward pass,
/// then unscaling gradients before the optimizer step. Automatically adjusts
/// the scale factor based on gradient health.
///
/// # Algorithm
///
/// 1. `scale_loss(loss)` → `loss * scale`
/// 2. Run `backward()` on the scaled loss → scaled gradients
/// 3. `unscale_gradients(grads)` → divides each gradient by `scale`
/// 4. `found_inf(grads)` → checks for NaN/Inf in unscaled gradients
/// 5. `update(found_inf)` → adjusts scale:
///    - If `found_inf`: `scale *= backoff_factor`, reset growth counter
///    - If not: increment growth counter; if it reaches `growth_interval`,
///      `scale *= growth_factor` and reset counter
#[derive(Debug)]
pub struct DynamicLossScaler {
    /// Current scale factor.
    scale: f32,
    /// Factor to grow scale by on consecutive good steps.
    growth_factor: f32,
    /// Factor to shrink scale by when inf/NaN detected.
    backoff_factor: f32,
    /// Steps without inf/NaN needed before growing.
    growth_interval: usize,
    /// Counter of consecutive good steps.
    consecutive_good_steps: usize,
    /// Target dtype for gradient accumulation.
    grad_dtype: DType,
}

impl DynamicLossScaler {
    /// Create a new dynamic loss scaler from configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid (non-positive scale,
    /// growth_factor <= 1, backoff_factor outside (0, 1), growth_interval == 0,
    /// or non-float grad_dtype).
    pub fn new(config: MixedPrecisionConfig) -> Result<Self> {
        if config.loss_scale <= 0.0 || !config.loss_scale.is_finite() {
            return Err(AutodiffError::InvalidConfig {
                op: "DynamicLossScaler",
                reason: format!(
                    "loss_scale must be positive and finite, got {}",
                    config.loss_scale
                ),
            });
        }
        if config.growth_factor <= 1.0 || !config.growth_factor.is_finite() {
            return Err(AutodiffError::InvalidConfig {
                op: "DynamicLossScaler",
                reason: format!(
                    "growth_factor must be > 1.0 and finite, got {}",
                    config.growth_factor
                ),
            });
        }
        if config.backoff_factor <= 0.0
            || config.backoff_factor >= 1.0
            || !config.backoff_factor.is_finite()
        {
            return Err(AutodiffError::InvalidConfig {
                op: "DynamicLossScaler",
                reason: format!(
                    "backoff_factor must be in (0, 1) and finite, got {}",
                    config.backoff_factor
                ),
            });
        }
        if config.growth_interval == 0 {
            return Err(AutodiffError::InvalidConfig {
                op: "DynamicLossScaler",
                reason: "growth_interval must be > 0".into(),
            });
        }
        if !config.grad_dtype.is_float() {
            return Err(AutodiffError::InvalidConfig {
                op: "DynamicLossScaler",
                reason: format!(
                    "grad_dtype must be a float type, got {:?}",
                    config.grad_dtype
                ),
            });
        }
        Ok(Self {
            scale: config.loss_scale,
            growth_factor: config.growth_factor,
            backoff_factor: config.backoff_factor,
            growth_interval: config.growth_interval,
            consecutive_good_steps: 0,
            grad_dtype: config.grad_dtype,
        })
    }

    /// Current scale factor.
    #[must_use]
    pub fn scale_factor(&self) -> f32 {
        self.scale
    }

    /// Target dtype for gradient accumulation.
    #[must_use]
    pub fn grad_dtype(&self) -> DType {
        self.grad_dtype
    }

    /// Number of consecutive good steps (no inf/NaN).
    #[must_use]
    pub fn consecutive_good_steps(&self) -> usize {
        self.consecutive_good_steps
    }

    /// Scale a loss tensor by the current scale factor.
    ///
    /// Returns a new tensor: `loss * scale`.
    pub fn scale_loss(&self, loss: &DynTensor) -> Result<DynTensor> {
        Ok(loss.mul_scalar(f64::from(self.scale))?)
    }

    /// Unscale gradients by dividing each by the current scale factor.
    ///
    /// After calling `backward()` on a scaled loss, call this to restore
    /// gradients to their true magnitude before passing to the optimizer.
    pub fn unscale_gradients(&self, grads: &mut [DynTensor]) -> Result<()> {
        let inv_scale = 1.0 / f64::from(self.scale);
        for grad in grads.iter_mut() {
            *grad = grad.mul_scalar(inv_scale)?;
        }
        Ok(())
    }

    /// Check whether any gradient tensor contains inf or NaN.
    ///
    /// Should be called after `unscale_gradients`. If this returns `true`,
    /// skip the optimizer step and call `update(true)`.
    pub fn found_inf(grads: &[DynTensor]) -> Result<bool> {
        for grad in grads {
            if grad.any_non_finite()? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Update the scale factor based on whether inf/NaN was found.
    ///
    /// Call this after every training step, regardless of whether the
    /// optimizer step was taken.
    ///
    /// - `found_inf = true`: scale is reduced by `backoff_factor`, counter reset
    /// - `found_inf = false`: counter incremented; if it reaches `growth_interval`,
    ///   scale is increased by `growth_factor` and counter resets
    pub fn update(&mut self, found_inf: bool) {
        if found_inf {
            self.scale *= self.backoff_factor;
            self.consecutive_good_steps = 0;
        } else {
            self.consecutive_good_steps += 1;
            if self.consecutive_good_steps >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.consecutive_good_steps = 0;
            }
        }
    }
}

impl Default for DynamicLossScaler {
    fn default() -> Self {
        Self::new(MixedPrecisionConfig::default())
            .expect("default MixedPrecisionConfig should be valid")
    }
}

/// Cast a gradient tensor to F32 if it is in a reduced-precision float dtype.
///
/// BF16 and F16 gradients are upcast to F32 for accumulation. F32 and F64
/// gradients are returned as-is. Non-float dtypes return an error.
///
/// This is the core helper for dtype-aware gradient accumulation: all backward
/// rules should use `grad.dtype()` (not hardcoded `DType::F32`) per the
/// engineering rule, and this function ensures accumulation always happens
/// in full precision.
pub fn cast_grad_to_f32(grad: &DynTensor) -> Result<DynTensor> {
    match grad.dtype() {
        DType::F32 | DType::F64 => Ok(grad.clone()),
        DType::BF16 | DType::F16 => Ok(grad.to_dtype(DType::F32)?),
        other => Err(AutodiffError::InvalidConfig {
            op: "cast_grad_to_f32",
            reason: format!("expected float dtype for gradient, got {other:?}"),
        }),
    }
}

#[cfg(test)]
#[path = "mixed_precision_tests.rs"]
mod tests;
