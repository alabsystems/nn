// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trainable LoRA adapter for gradient-tracked training.
//!
//! Extracted from `lora.rs` (#1575) to keep files under 400 lines.

use std::sync::Arc;

use nn_autodiff::{TrackedTensor, TrainableModule, Var};
use nn_core::dyn_tensor::DynTensor;

use super::{init_lora, LoraLinear};
use crate::error::{OptimError, Result};
use nn_core::layers::Linear;

/// A LoRA adapter that participates in the autodiff backward pass.
///
/// Unlike [`LoraLinear`] (which returns `DynTensor` for inference), this type
/// returns `Arc<TrackedTensor>` for gradient-tracked training. Only `lora_a`
/// and `lora_b` receive gradients; the frozen weight is a constant in the graph.
///
/// Forward: `y = x @ W_frozen^T + (x @ A^T) @ B^T * scaling + bias_frozen`
///
/// # Example
///
/// ```ignore
/// // NOTE: ignore — requires TrackedTensor input and backward setup
/// let linear = Linear::new(weight, Some(bias));
/// let lora = TrainableLoraLinear::from_linear(&linear, 8, 8.0)?;
/// let y = lora.forward(&x_tracked)?;  // tracked output for backward()
/// let grads = backward(&y)?;          // gradients for A and B only
/// ```
#[derive(Debug)]
pub struct TrainableLoraLinear {
    /// Frozen original weight, shape `[out_features, in_features]`.
    frozen_weight: DynTensor,
    /// Frozen original bias, shape `[out_features]` (optional).
    frozen_bias: Option<DynTensor>,
    /// Low-rank A matrix, shape `[rank, in_features]`. Trainable.
    lora_a: Var,
    /// Low-rank B matrix, shape `[out_features, rank]`. Trainable.
    lora_b: Var,
    /// Scaling factor: `alpha / rank`.
    scaling: f64,
}

impl TrainableLoraLinear {
    /// Create a trainable LoRA adapter from an existing `Linear` layer.
    ///
    /// - `rank`: low-rank dimension (typical: 4, 8, 16)
    /// - `alpha`: scaling factor (typical: equal to rank)
    ///
    /// Initializes `A` with random normal values and `B` with zeros, so the
    /// initial output is identical to the original `Linear` layer.
    #[must_use = "LoRA adapter must be stored for training"]
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

    /// Create from an existing (inference-only) [`LoraLinear`], sharing its `Var`s.
    ///
    /// This is useful when switching from inference to training mode.
    /// Returns an error if the source `LoraLinear` has a non-finite scaling factor.
    #[must_use = "LoRA adapter must be stored for training"]
    pub fn from_lora_linear(lora: &LoraLinear) -> Result<Self> {
        if !lora.scaling().is_finite() {
            return Err(OptimError::InvalidParam {
                param: "scaling",
                reason: format!("LoRA scaling must be finite, got {}", lora.scaling()),
            });
        }
        Ok(Self {
            frozen_weight: lora.frozen_weight().clone(),
            frozen_bias: lora.frozen_bias().cloned(),
            lora_a: lora.lora_a().clone(),
            lora_b: lora.lora_b().clone(),
            scaling: lora.scaling(),
        })
    }

    /// Returns references to trainable variables `[A, B]` for optimizer.
    #[must_use]
    pub fn vars(&self) -> Vec<&Var> {
        vec![&self.lora_a, &self.lora_b]
    }

    /// Gradient-tracked forward pass.
    ///
    /// The frozen weight contributes to the output but does not receive gradients.
    /// Only `lora_a` and `lora_b` are tracked as `Var`s in the computation graph.
    pub fn forward(
        &self,
        x: &Arc<TrackedTensor>,
    ) -> nn_autodiff::error::Result<Arc<TrackedTensor>> {
        // Frozen base: x @ W^T (constant, no gradients)
        let wt = self.frozen_weight.transpose(0, 1)?;
        let wt_tracked = Arc::new(TrackedTensor::from_tensor(wt));
        let base = x.matmul(&wt_tracked)?;

        // LoRA path: (x @ A^T) @ B^T * scaling (tracked, receives gradients)
        let a_tracked = Arc::new(TrackedTensor::from_var(&self.lora_a)?);
        let b_tracked = Arc::new(TrackedTensor::from_var(&self.lora_b)?);
        let at = a_tracked.transpose(0, 1)?;
        let bt = b_tracked.transpose(0, 1)?;
        let lora_out = x.matmul(&at)?.matmul(&bt)?;
        let scaled_lora = lora_out.mul_scalar(self.scaling)?;

        // Combine base + LoRA
        let y = base.add(&scaled_lora)?;

        // Add frozen bias if present (constant, no gradients)
        match &self.frozen_bias {
            Some(bias) => {
                let bias_tracked = Arc::new(TrackedTensor::from_tensor(bias.clone()));
                y.add(&bias_tracked)
            }
            None => Ok(y),
        }
    }

    /// Merge LoRA weights into the frozen weight for deployment.
    ///
    /// Returns `W_merged = W + scaling * B @ A`.
    pub fn merge(&self) -> Result<DynTensor> {
        let b = self.lora_b.data()?;
        let a = self.lora_a.data()?;
        let ba = b.matmul(&a)?;
        let scaled_ba = ba.mul_scalar(self.scaling)?;
        let merged = self.frozen_weight.add(&scaled_ba)?;

        // Defense-in-depth: verify merged weight is finite
        crate::error::validate_update(&merged)?;

        Ok(merged)
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
}

impl TrainableModule for TrainableLoraLinear {
    fn forward(&self, x: &Arc<TrackedTensor>) -> nn_autodiff::error::Result<Arc<TrackedTensor>> {
        self.forward(x)
    }

    fn vars(&self) -> Vec<&Var> {
        self.vars()
    }
}
