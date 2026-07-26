// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! AdaFactor optimizer with factored second moments.
//!
//! Algorithm (Shazeer & Stern, 2018):
//!
//! For matrices (rank >= 2), second moments are factored into row and column
//! vectors, reducing memory from O(mn) to O(m+n). For vectors (rank < 2),
//! full second moments are stored (like Adam).
//!
//! Key features:
//! - Memory-efficient via factored second moments
//! - Optional relative step size (learning-rate-free mode)
//! - Adaptive beta2 that increases with training steps

use crate::error::{OptimError, Result};
use crate::optimizer::Optimizer;
use nn_autodiff::{GradStore, Var};
use nn_core::dyn_tensor::DynTensor;

/// Configuration for the AdaFactor optimizer.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaFactorConfig {
    /// Learning rate. Ignored when `relative_step` is true.
    pub lr: f64,
    /// Whether to use relative step size: `lr_t = max(eps_rms, rms(param)) * rho_t`.
    pub relative_step: bool,
    /// Epsilon for RMS computation when using relative step.
    pub eps_rms: f64,
    /// Epsilon for denominator numerical stability.
    pub eps_denom: f64,
    /// Decay rate for second moment estimation (used in beta2 schedule).
    pub decay_rate: f64,
    /// Whether to use first moment (momentum). Disabling saves memory.
    pub beta1: Option<f64>,
    /// Weight decay coefficient (decoupled, like AdamW).
    pub weight_decay: f64,
}

impl Default for AdaFactorConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            relative_step: false,
            eps_rms: 1e-3,
            eps_denom: 1e-30,
            decay_rate: -0.8,
            beta1: None,
            weight_decay: 0.0,
        }
    }
}

/// Per-variable state for AdaFactor.
#[derive(Debug)]
struct VarState {
    var: Var,
    /// First moment estimate (optional, only if beta1 is Some).
    first_moment: Option<DynTensor>,
    /// Full second moment for rank < 2 variables.
    second_moment_full: Option<DynTensor>,
    /// Row factor of second moment (for rank >= 2).
    row_factor: Option<DynTensor>,
    /// Column factor of second moment (for rank >= 2).
    col_factor: Option<DynTensor>,
}

/// AdaFactor optimizer with factored second moments.
///
/// For matrices (rank >= 2), the second moment accumulator is factored into
/// row and column vectors. For vectors (rank < 2), full second moments are
/// stored.
#[derive(Debug)]
pub struct AdaFactor {
    states: Vec<VarState>,
    config: AdaFactorConfig,
    step_t: usize,
}

impl AdaFactor {
    /// Create a new AdaFactor optimizer for the given trainable variables.
    ///
    /// # Errors
    ///
    /// Returns `InvalidParam` if:
    /// - `lr` is negative or not finite
    /// - `beta1` (if Some) is outside `[0, 1)`
    /// - `eps_denom` or `eps_rms` is not positive or not finite
    /// - `decay_rate` is not finite
    /// - `weight_decay` is negative or not finite
    #[must_use = "optimizer must be stored to perform training steps"]
    pub fn new(vars: Vec<Var>, config: AdaFactorConfig) -> Result<Self> {
        Self::validate_config(&config)?;

        let mut states = Vec::with_capacity(vars.len());
        for var in vars {
            let dims = var.dims()?;
            let dtype = var.dtype()?;
            let device = var.device()?;
            let use_factored = dims.len() >= 2;

            let first_moment = if config.beta1.is_some() {
                Some(DynTensor::zeros(&dims, dtype, &device)?)
            } else {
                None
            };

            let (second_moment_full, row_factor, col_factor) = if use_factored {
                // Row factor: all dims except last
                let mut row_shape = dims.clone();
                let last = row_shape.len() - 1;
                row_shape[last] = 1;
                // Col factor: all dims except second-to-last
                let mut col_shape = dims.clone();
                let second_last = col_shape.len() - 2;
                col_shape[second_last] = 1;
                (
                    None,
                    Some(DynTensor::zeros(&row_shape, dtype, &device)?),
                    Some(DynTensor::zeros(&col_shape, dtype, &device)?),
                )
            } else {
                (Some(DynTensor::zeros(&dims, dtype, &device)?), None, None)
            };

            states.push(VarState {
                var,
                first_moment,
                second_moment_full,
                row_factor,
                col_factor,
            });
        }
        Ok(Self {
            states,
            config,
            step_t: 0,
        })
    }

    /// Validate all configuration parameters.
    ///
    /// Extracted from `new()` for function-size compliance.
    fn validate_config(config: &AdaFactorConfig) -> Result<()> {
        crate::error::validate_lr(config.lr)?;
        if !config.decay_rate.is_finite() {
            return Err(OptimError::InvalidParam {
                param: "decay_rate",
                reason: format!("must be finite, got {}", config.decay_rate),
            });
        }
        if config.decay_rate > 0.0 {
            return Err(OptimError::InvalidParam {
                param: "decay_rate",
                reason: format!(
                    "must be negative for proper beta2 schedule (paper default: -0.8), got {}",
                    config.decay_rate
                ),
            });
        }
        crate::error::validate_weight_decay(config.weight_decay)?;
        if let Some(b1) = config.beta1 {
            if !(0.0..1.0).contains(&b1) {
                return Err(OptimError::InvalidParam {
                    param: "beta1",
                    reason: format!("must be in [0, 1), got {b1}"),
                });
            }
        }
        if config.eps_denom <= 0.0 || !config.eps_denom.is_finite() {
            return Err(OptimError::InvalidParam {
                param: "eps_denom",
                reason: format!("must be positive and finite, got {}", config.eps_denom),
            });
        }
        if config.eps_rms <= 0.0 || !config.eps_rms.is_finite() {
            return Err(OptimError::InvalidParam {
                param: "eps_rms",
                reason: format!("must be positive and finite, got {}", config.eps_rms),
            });
        }
        Ok(())
    }

    /// Current step count.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.step_t
    }

    /// Configuration reference.
    #[must_use]
    pub fn config(&self) -> &AdaFactorConfig {
        &self.config
    }

    /// Compute beta2 schedule: `1 - t^decay_rate`.
    ///
    /// `step_t` is already incremented by `step()` before this is called, so
    /// `step_t == 1` on the first optimizer step.  No additional offset is
    /// needed — the previous `saturating_add(1)` caused an off-by-one where
    /// step 1 used `t=2` (#1993).
    ///
    /// Caps step at `i32::MAX` before `as f64` cast to prevent precision loss
    /// on 64-bit targets and overflow on 32-bit targets, matching Adam's pattern.
    /// Clamps result to `[0.0, 1.0 - 1e-8]` to prevent negative rho when
    /// decay_rate is positive (which would make second moments negative → NaN).
    fn rho_t(&self) -> f64 {
        let t = self.step_t.min(i32::MAX as usize) as f64;
        (1.0 - t.powf(self.config.decay_rate)).clamp(0.0, 1.0 - 1e-8)
    }
}

/// Update factored second moment estimates (rank >= 2).
///
/// Returns the reconstructed approximate second moment `v_t`.
fn update_factored_moment(
    state: &mut VarState,
    grad: &DynTensor,
    rho: f64,
    eps_denom: f64,
) -> Result<DynTensor> {
    let rf = state.row_factor.as_ref().ok_or(OptimError::MissingState {
        optimizer: "AdaFactor",
        state: "row_factor",
    })?;
    let cf = state.col_factor.as_ref().ok_or(OptimError::MissingState {
        optimizer: "AdaFactor",
        state: "col_factor",
    })?;
    let g_sqr = grad.sqr()?;
    let dims = grad.dims();
    let last_dim = dims.len() - 1;
    let second_last_dim = dims.len() - 2;

    // Row factor: mean over last dim of g^2
    let new_row = rf
        .mul_scalar(rho)?
        .add(&g_sqr.mean_keepdim(last_dim)?.mul_scalar(1.0 - rho)?)?;
    // Col factor: mean over second-to-last dim of g^2
    let new_col = cf
        .mul_scalar(rho)?
        .add(&g_sqr.mean_keepdim(second_last_dim)?.mul_scalar(1.0 - rho)?)?;

    // Reconstruct: v_t = row_factor * col_factor / mean(row_factor)
    // Stays on-device: mean_all() returns a scalar tensor that broadcasts
    // through div(), avoiding a GPU→CPU readback per variable.
    // add_scalar(eps_denom) prevents division by zero when all gradients are
    // exactly zero (row_mean = 0).  eps_denom is 1e-30 by default — it does
    // not shift the result for non-degenerate inputs.
    let row_mean = new_row.mean_all()?.add_scalar(eps_denom)?;
    let v_approx = new_row.mul(&new_col.expand(grad.dims())?)?.div(&row_mean)?;

    state.row_factor = Some(new_row);
    state.col_factor = Some(new_col);
    Ok(v_approx)
}

impl Optimizer for AdaFactor {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_t += 1;
        let rho = self.rho_t();
        // Cache config values to avoid borrowing self inside the mutable loop.
        let config_lr = self.config.lr;
        let relative_step = self.config.relative_step;
        let eps_rms = self.config.eps_rms;
        let eps_denom = self.config.eps_denom;
        let beta1 = self.config.beta1;
        let weight_decay = self.config.weight_decay;
        let step_t = self.step_t;

        for state in &mut self.states {
            let grad = match grads.get(&state.var) {
                Some(g) => g.clone(),
                None => continue,
            };
            crate::error::validate_gradient(&grad)?;

            let param = state.var.data()?;

            // Update second moment estimates
            let v_t = if state.row_factor.is_some() && state.col_factor.is_some() {
                update_factored_moment(state, &grad, rho, eps_denom)?
            } else if let Some(ref v) = state.second_moment_full {
                // Full second moment (rank < 2)
                let new_v = v
                    .mul_scalar(rho)?
                    .add(&grad.sqr()?.mul_scalar(1.0 - rho)?)?;
                state.second_moment_full = Some(new_v.clone());
                new_v
            } else {
                return Err(OptimError::CorruptedState {
                    optimizer: "AdaFactor",
                    reason: "neither factored nor full second moment exists",
                });
            };

            // Compute update: u_t = grad / sqrt(v_t + eps)
            let u_t = grad.div(&v_t.add_scalar(eps_denom)?.sqrt()?)?;

            // Apply first moment (momentum) if enabled
            let u_t = if let (Some(b1), Some(ref m)) = (beta1, &state.first_moment) {
                let new_m = m.mul_scalar(b1)?.add(&u_t.mul_scalar(1.0 - b1)?)?;
                state.first_moment = Some(new_m.clone());
                new_m
            } else {
                u_t
            };

            // Apply lr, weight decay, and parameter update.
            // When relative_step is true, lr is a per-variable on-device scalar
            // tensor derived from parameter RMS — avoids GPU→CPU pipeline stall.
            let new_theta = if relative_step {
                let rho_lr = 1.0 / (step_t.min(i32::MAX as usize) as f64).sqrt();
                let lr_t = param
                    .sqr()?
                    .mean_all()?
                    .sqrt()?
                    .clamp_min(eps_rms)?
                    .mul_scalar(rho_lr)?;
                let theta = if weight_decay > 0.0 {
                    param.sub(&param.mul(&lr_t)?.mul_scalar(weight_decay)?)?
                } else {
                    param
                };
                theta.sub(&u_t.mul(&lr_t)?)?
            } else {
                let theta = if weight_decay > 0.0 {
                    param.mul_scalar(1.0 - config_lr * weight_decay)?
                } else {
                    param
                };
                theta.sub(&u_t.mul_scalar(config_lr)?)?
            };
            crate::error::validate_update(&new_theta)?;
            state.var.set(&new_theta)?;
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) -> Result<()> {
        crate::error::validate_lr(lr)?;
        self.config.lr = lr;
        Ok(())
    }
}

#[path = "adafactor_checkpoint.rs"]
mod checkpoint_impl;

#[cfg(test)]
#[path = "adafactor_tests.rs"]
mod tests;
