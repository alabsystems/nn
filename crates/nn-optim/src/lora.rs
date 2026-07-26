// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
//!
//! Replaces a frozen `Linear` layer's weight `W` with `W + (alpha/r) * B @ A`,
//! where only the low-rank matrices `A` (rank × in) and `B` (out × rank) are
//! trainable. This adds `r * (in + out)` parameters instead of `in * out`.
//!
//! # Example
//!
//! ```ignore
//! // NOTE: ignore — requires constructing weight/bias DynTensors and input x
//! let linear = Linear::new(weight, Some(bias));
//! let lora = LoraLinear::from_linear(&linear, 8, 8.0)?;
//! let y = lora.forward(&x)?;  // efficient two-matmul path
//! let merged = lora.merge()?; // W + scaling * B @ A for deployment
//! ```

use nn_autodiff::Var;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::{DType, Device};

use crate::error::{OptimError, Result};

/// Shared LoRA initialization: validates params, creates A/B matrices, computes scaling.
///
/// Returns `(frozen_weight, frozen_bias, lora_a, lora_b, scaling)`.
fn init_lora(
    linear: &Linear,
    rank: usize,
    alpha: f64,
) -> Result<(DynTensor, Option<DynTensor>, Var, Var, f64)> {
    if rank == 0 {
        return Err(OptimError::InvalidParam {
            param: "rank",
            reason: "LoRA rank must be > 0".into(),
        });
    }
    if !alpha.is_finite() {
        return Err(OptimError::InvalidParam {
            param: "alpha",
            reason: format!("LoRA alpha must be finite, got {alpha}"),
        });
    }

    let (out_features, in_features) = linear.weight().dims2()?;
    let device = linear.weight().device();

    // A: [rank, in_features] — random normal init.
    // Compute on CPU (randn uses host RNG), then move to target device.
    let a_data =
        DynTensor::randn(0.0, 1.0, &[rank, in_features], &Device::Cpu)?.to_device(&device)?;
    let lora_a = Var::new(a_data);

    // B: [out_features, rank] — zero init (so initial output == original).
    let b_data = DynTensor::zeros(&[out_features, rank], DType::F32, &device)?;
    let lora_b = Var::new(b_data);

    let scaling = alpha / rank as f64;

    // Reject scaling values that overflow f32 (used in forward/merge as `scaling as f32`)
    if (scaling as f32).is_infinite() {
        return Err(OptimError::InvalidParam {
            param: "alpha",
            reason: format!("LoRA scaling alpha/rank = {scaling} overflows f32"),
        });
    }

    Ok((
        linear.weight().clone(),
        linear.bias().cloned(),
        lora_a,
        lora_b,
        scaling,
    ))
}

/// LoRA adapter wrapping a frozen `Linear` layer with trainable low-rank matrices.
///
/// Forward: `y = x @ W^T + (x @ A^T) @ B^T * scaling + bias`
///
/// Only `lora_a` and `lora_b` are trainable; the original weight is frozen.
#[derive(Debug)]
pub struct LoraLinear {
    /// Frozen original weight, shape `[out_features, in_features]`.
    frozen_weight: DynTensor,
    /// Frozen original bias, shape `[out_features]` (optional).
    frozen_bias: Option<DynTensor>,
    /// Low-rank A matrix, shape `[rank, in_features]`. Initialized random normal.
    lora_a: Var,
    /// Low-rank B matrix, shape `[out_features, rank]`. Initialized to zero.
    lora_b: Var,
    /// Scaling factor: `alpha / rank`.
    scaling: f64,
}

impl LoraLinear {
    /// Create a LoRA adapter from an existing `Linear` layer.
    ///
    /// - `rank`: low-rank dimension (typical: 4, 8, 16)
    /// - `alpha`: scaling factor (typical: equal to rank)
    ///
    /// Initializes `A` with random normal values and `B` with zeros, so the
    /// initial output is identical to the original `Linear` layer.
    #[must_use = "LoRA adapter must be stored for forward/merge"]
    pub fn from_linear(linear: &Linear, rank: usize, alpha: f64) -> Result<Self> {
        let (frozen_weight, frozen_bias, lora_a, lora_b, scaling) = init_lora(linear, rank, alpha)?;
        Ok(Self {
            frozen_weight,
            frozen_bias,
            lora_a,
            lora_b,
            scaling,
        })
    }

    /// Returns references to the trainable variables `[A, B]` for passing to an optimizer.
    ///
    /// Matches `TrainableModule::vars()` signature (returns `Vec<&Var>`).
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<&Var> {
        vec![&self.lora_a, &self.lora_b]
    }

    /// Merge LoRA weights into the frozen weight for deployment.
    ///
    /// Returns `W_merged = W + scaling * B @ A`, eliminating LoRA runtime overhead.
    pub fn merge(&self) -> Result<DynTensor> {
        let b = self.lora_b.data()?;
        let a = self.lora_a.data()?;

        // B @ A: [out_features, rank] @ [rank, in_features] -> [out_features, in_features]
        let ba = b.matmul(&a)?;

        // scaling * (B @ A)
        let scaled_ba = ba.mul_scalar(self.scaling)?;

        // W + scaling * B @ A
        let merged = self.frozen_weight.add(&scaled_ba)?;

        // Defense-in-depth: verify merged weight is finite
        crate::error::validate_update(&merged)?;

        Ok(merged)
    }

    /// Frozen weight reference.
    #[must_use]
    pub fn frozen_weight(&self) -> &DynTensor {
        &self.frozen_weight
    }

    /// LoRA A matrix (trainable).
    #[must_use]
    pub fn lora_a(&self) -> &Var {
        &self.lora_a
    }

    /// LoRA B matrix (trainable).
    #[must_use]
    pub fn lora_b(&self) -> &Var {
        &self.lora_b
    }

    /// Scaling factor (alpha / rank).
    #[must_use]
    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    /// Frozen bias reference (if the original `Linear` had a bias).
    #[must_use]
    pub fn frozen_bias(&self) -> Option<&DynTensor> {
        self.frozen_bias.as_ref()
    }
}

impl Module for LoraLinear {
    /// Forward pass using efficient two-matmul path.
    ///
    /// `y = x @ W^T + (x @ A^T) @ B^T * scaling + bias`
    ///
    /// This avoids materializing the full `[out, in]` LoRA delta matrix.
    fn forward(&self, x: &DynTensor) -> nn_core::Result<DynTensor> {
        // Base: x @ W^T
        let wt = self.frozen_weight.transpose(0, 1)?;
        let base = x.matmul(&wt)?;

        // LoRA: (x @ A^T) @ B^T * scaling
        let a = self
            .lora_a
            .data()
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        let b = self
            .lora_b
            .data()
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        let at = a.transpose(0, 1)?;
        let bt = b.transpose(0, 1)?;
        let lora_out = x.matmul(&at)?.matmul(&bt)?;

        // Scale LoRA contribution
        let scaled_lora = lora_out.mul_scalar(self.scaling)?;

        // Combine
        let y = base.add(&scaled_lora)?;

        // Add bias if present
        match &self.frozen_bias {
            Some(bias) => y.broadcast_add(bias),
            None => Ok(y),
        }
    }
}

// TrainableLoraLinear extracted to lora_trainable.rs (#1575).
#[path = "lora_trainable.rs"]
mod trainable;
pub use trainable::TrainableLoraLinear;

/// Configuration for LoRA injection into a model.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LoraConfig {
    /// Low-rank dimension (typical: 4, 8, 16).
    pub rank: usize,
    /// Scaling factor (typical: equal to rank).
    pub alpha: f64,
    /// Target layer name patterns for LoRA injection (e.g., `["q_proj", "v_proj"]`).
    pub targets: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 8.0,
            targets: vec!["q_proj".into(), "v_proj".into()],
        }
    }
}

#[cfg(test)]
#[path = "lora_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "lora_trainable_tests.rs"]
mod trainable_tests;

#[cfg(test)]
#[path = "lora_validation_tests.rs"]
mod validation_tests;
