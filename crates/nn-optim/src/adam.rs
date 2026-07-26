// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! AdamW optimizer with decoupled weight decay.
//!
//! Algorithm (Loshchilov & Hutter, 2017):
//! ```text
//! m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
//! v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
//! m_hat = m_t / (1 - beta1^t)
//! v_hat = v_t / (1 - beta2^t)
//! theta = theta * (1 - lr * lambda) - lr * m_hat / (sqrt(v_hat) + eps)
//! ```

use crate::error::{OptimError, Result};
use crate::optimizer::Optimizer;
use nn_autodiff::{GradStore, Var};
use nn_core::device::Device;
use nn_core::dyn_tensor::DynTensor;

/// Configuration for the AdamW optimizer.
///
/// Defaults match candle and PyTorch:
/// - lr: 1e-3
/// - beta1: 0.9
/// - beta2: 0.999
/// - eps: 1e-8
/// - weight_decay: 0.01
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdamConfig {
    /// Learning rate.
    pub lr: f64,
    /// Exponential decay rate for first moment estimates.
    pub beta1: f64,
    /// Exponential decay rate for second moment estimates.
    pub beta2: f64,
    /// Small constant for numerical stability in denominator.
    pub eps: f64,
    /// Decoupled weight decay coefficient (AdamW).
    pub weight_decay: f64,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// Per-variable optimizer state: first and second moment estimates.
#[derive(Debug)]
struct VarState {
    var: Var,
    first_moment: DynTensor,
    second_moment: DynTensor,
}

/// AdamW optimizer with decoupled weight decay.
///
/// Implements the AdamW algorithm (Loshchilov & Hutter, 2017) with bias
/// correction for the moment estimates.
#[derive(Debug)]
pub struct AdamW {
    states: Vec<VarState>,
    config: AdamConfig,
    step_t: usize,
}

impl AdamW {
    /// Create a new AdamW optimizer for the given trainable variables.
    ///
    /// Initializes first and second moment estimates to zero.
    ///
    /// # Errors
    ///
    /// Returns `InvalidParam` if:
    /// - `lr` is negative or not finite
    /// - `beta1` or `beta2` is outside `[0, 1)` (1.0 causes division by zero in bias correction)
    /// - `eps` is not positive (zero eps causes division by zero in adaptive step)
    /// - `weight_decay` is negative or not finite
    #[must_use = "optimizer must be stored to perform training steps"]
    pub fn new(vars: Vec<Var>, config: AdamConfig) -> Result<Self> {
        crate::error::validate_lr(config.lr)?;
        if !(0.0..1.0).contains(&config.beta1) {
            return Err(OptimError::InvalidParam {
                param: "beta1",
                reason: format!("must be in [0, 1), got {}", config.beta1),
            });
        }
        if !(0.0..1.0).contains(&config.beta2) {
            return Err(OptimError::InvalidParam {
                param: "beta2",
                reason: format!("must be in [0, 1), got {}", config.beta2),
            });
        }
        if config.eps <= 0.0 || !config.eps.is_finite() {
            return Err(OptimError::InvalidParam {
                param: "eps",
                reason: format!("must be finite and positive, got {}", config.eps),
            });
        }
        crate::error::validate_weight_decay(config.weight_decay)?;
        let mut states = Vec::with_capacity(vars.len());
        for var in vars {
            let dims = var.dims()?;
            let dtype = var.dtype()?;
            let device = var.device()?;
            let zeros = DynTensor::zeros(&dims, dtype, &device)?;
            states.push(VarState {
                first_moment: zeros.clone(),
                second_moment: zeros,
                var,
            });
        }
        Ok(Self {
            states,
            config,
            step_t: 0,
        })
    }

    /// Current step count.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.step_t
    }

    /// Configuration reference.
    #[must_use]
    pub fn config(&self) -> &AdamConfig {
        &self.config
    }
}

/// Pre-computed f32 hyperparameters for the fused Adam kernel.
struct AdamHyperparams {
    beta1: f32,
    beta2: f32,
    lr: f32,
    eps: f32,
    wd: f32,
    bc1: f32,
    bc2: f32,
    decay_factor: f32,
}

/// Move a tensor to CPU if it is on a GPU device.
fn ensure_cpu(t: &DynTensor) -> std::result::Result<DynTensor, nn_core::TensorError> {
    if t.device().is_gpu() {
        t.to_device(&Device::Cpu)
    } else {
        Ok(t.clone())
    }
}

/// Fused Adam update: compute m, v, and theta in a single pass over raw f32
/// slices. Avoids ~14 intermediate DynTensor allocations per variable that
/// the element-wise API would create.
///
/// GPU-resident tensors are automatically moved to CPU for the fused update,
/// then results are transferred back to the original device.
fn fused_adam_update(state: &mut VarState, grad: &DynTensor, hp: &AdamHyperparams) -> Result<()> {
    // Capture original device so we can transfer results back after CPU math.
    let original_device = state.var.device()?;

    // Move all inputs to CPU if they are on GPU.
    let grad_cpu = ensure_cpu(grad).map_err(OptimError::Tensor)?;
    let m_cpu = ensure_cpu(&state.first_moment).map_err(OptimError::Tensor)?;
    let v_cpu = ensure_cpu(&state.second_moment).map_err(OptimError::Tensor)?;
    let theta_cpu = ensure_cpu(&state.var.data()?).map_err(OptimError::Tensor)?;

    let grad_arr = grad_cpu.to_f32_array()?.as_standard_layout().to_owned();
    let mut m_arr = m_cpu.to_f32_array()?;
    let mut v_arr = v_cpu.to_f32_array()?;
    let mut theta_arr = theta_cpu.to_f32_array()?;

    // to_f32_array() returns owned standard-layout arrays.
    // as_standard_layout() + to_owned() guarantees contiguity for
    // the grad (which may come from a non-contiguous intermediate).
    let not_contiguous = || OptimError::InvalidParam {
        param: "tensor",
        reason: "non-contiguous array in fused Adam step".into(),
    };
    let grad_slice = grad_arr.as_slice().ok_or_else(not_contiguous)?;
    let m_slice = m_arr.as_slice_mut().ok_or_else(not_contiguous)?;
    let v_slice = v_arr.as_slice_mut().ok_or_else(not_contiguous)?;
    let theta_slice = theta_arr.as_slice_mut().ok_or_else(not_contiguous)?;

    let mut non_finite_count = 0usize;
    for i in 0..theta_slice.len() {
        let g = grad_slice[i];
        m_slice[i] = beta1_ema(m_slice[i], g, hp.beta1);
        v_slice[i] = beta2_ema(v_slice[i], g, hp.beta2);
        let m_hat = m_slice[i] * hp.bc1;
        let v_hat = v_slice[i] * hp.bc2;
        let step = hp.lr * m_hat / (v_hat.sqrt() + hp.eps);
        let new_val = if hp.wd > 0.0 {
            theta_slice[i] * hp.decay_factor - step
        } else {
            theta_slice[i] - step
        };
        if !new_val.is_finite() {
            non_finite_count += 1;
        }
        theta_slice[i] = new_val;
    }

    if non_finite_count > 0 {
        return Err(OptimError::NonFiniteUpdate {
            count: non_finite_count,
        });
    }

    let input_dtype = state.first_moment.dtype();
    let mut new_m = DynTensor::from_f32_result(m_arr, input_dtype)?;
    let mut new_v = DynTensor::from_f32_result(v_arr, input_dtype)?;
    let theta_dtype = state.var.dtype()?;
    let mut new_theta = DynTensor::from_f32_result(theta_arr, theta_dtype)?;

    // Transfer results back to the original device if needed.
    if original_device.is_gpu() {
        new_m = new_m
            .to_device(&original_device)
            .map_err(OptimError::Tensor)?;
        new_v = new_v
            .to_device(&original_device)
            .map_err(OptimError::Tensor)?;
        new_theta = new_theta
            .to_device(&original_device)
            .map_err(OptimError::Tensor)?;
    }

    state.first_moment = new_m;
    state.second_moment = new_v;
    state.var.set(&new_theta)?;
    Ok(())
}

/// First-moment EMA: m = beta1 * m + (1 - beta1) * g
#[inline]
fn beta1_ema(m: f32, g: f32, beta1: f32) -> f32 {
    beta1 * m + (1.0 - beta1) * g
}

/// Second-moment EMA: v = beta2 * v + (1 - beta2) * g²
#[inline]
fn beta2_ema(v: f32, g: f32, beta2: f32) -> f32 {
    beta2 * v + (1.0 - beta2) * g * g
}

impl Optimizer for AdamW {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_t += 1;
        // Cap at i32::MAX for powi() — bias correction is 1.0 for any
        // t > ~7000 (beta1=0.9) or ~7M (beta2=0.999), so capping is
        // numerically invisible but prevents unsound `as i32` overflow.
        let t = self.step_t.min(i32::MAX as usize) as i32;

        let hp = AdamHyperparams {
            beta1: self.config.beta1 as f32,
            beta2: self.config.beta2 as f32,
            lr: self.config.lr as f32,
            eps: self.config.eps as f32,
            wd: self.config.weight_decay as f32,
            bc1: 1.0f32 / (1.0 - (self.config.beta1.powi(t) as f32)),
            bc2: 1.0f32 / (1.0 - (self.config.beta2.powi(t) as f32)),
            decay_factor: 1.0f32 - (self.config.lr as f32) * (self.config.weight_decay as f32),
        };

        for state in &mut self.states {
            let grad = match grads.get(&state.var) {
                Some(g) => g,
                None => continue,
            };
            crate::error::validate_gradient(grad)?;
            fused_adam_update(state, grad, &hp)?;
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

// Checkpoint save/load (OptimizerCheckpoint impl + restore_adam_config)
// extracted to adam_checkpoint.rs via #[path] submodule.
#[path = "adam_checkpoint.rs"]
mod checkpoint_impl;

#[cfg(test)]
#[path = "adam_tests.rs"]
mod tests;
