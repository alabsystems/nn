// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.
//!
//! Wraps a frozen [`Linear`] layer with trainable low-rank A/B matrices.
//! Forward: `y = frozen_linear(x) + (alpha/rank) * x @ A^T @ B^T`.
//!
//! This is the core nn-level LoRA implementation using pure [`DynTensor`].
//! For gradient-tracked training with [`Var`](nn_autodiff::Var), see
//! `nn-optim`'s `LoraLinear` and `TrainableLoraLinear`.
//!
//! # Example
//!
//! ```ignore
//! use nn_core::layers::{Linear, LoraLinear, LoraConfig};
//! let config = LoraConfig { rank: 8, alpha: 8.0, dropout: None };
//! let lora = LoraLinear::from_linear(&linear, &config)?;
//! let y = lora.forward(&x)?;
//! let merged = lora.merge()?;  // W + scaling * B @ A
//! ```

use super::{Linear, Module};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device, Result, TensorError};

/// Configuration for LoRA (Low-Rank Adaptation).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LoraConfig {
    /// Low-rank dimension (typical: 4, 8, 16).
    pub rank: usize,
    /// Scaling factor (typical: equal to rank). Effective scaling = alpha / rank.
    pub alpha: f32,
    /// Optional dropout rate for LoRA path (not applied during inference).
    pub dropout: Option<f32>,
}

impl LoraConfig {
    /// Create a new LoRA config.
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self {
            rank,
            alpha,
            dropout: None,
        }
    }

    /// Create a LoRA config with dropout.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = Some(dropout);
        self
    }
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 8.0,
            dropout: None,
        }
    }
}

/// LoRA adapter wrapping a frozen [`Linear`] layer with trainable low-rank matrices.
///
/// Forward: `y = x @ W^T + (alpha/rank) * (x @ A^T) @ B^T + bias`
///
/// Only `lora_a` and `lora_b` are trainable; the original weight and bias are frozen.
/// `B` is zero-initialized so the initial output matches the original `Linear` exactly.
///
/// # Merge for deployment
///
/// Call [`merge()`](LoraLinear::merge) to fold LoRA into the frozen weight:
/// `W_merged = W + (alpha/rank) * B @ A`. This eliminates LoRA runtime overhead.
#[derive(Debug, Clone)]
pub struct LoraLinear {
    /// Frozen original weight, shape `[out_features, in_features]`.
    frozen_weight: DynTensor,
    /// Frozen original bias, shape `[out_features]` (optional).
    frozen_bias: Option<DynTensor>,
    /// Low-rank A matrix, shape `[rank, in_features]`. Initialized with random normal.
    lora_a: DynTensor,
    /// Low-rank B matrix, shape `[out_features, rank]`. Initialized to zero.
    lora_b: DynTensor,
    /// Scaling factor: `alpha / rank`.
    scaling: f64,
}

impl LoraLinear {
    /// Create a LoRA adapter from an existing [`Linear`] layer and config.
    ///
    /// Initializes `A` with random normal values (std=1.0) and `B` with zeros,
    /// so the initial output is identical to the original `Linear`.
    pub fn from_linear(linear: &Linear, config: &LoraConfig) -> Result<Self> {
        if config.rank == 0 {
            return Err(TensorError::InvalidShape("LoRA rank must be > 0".into()));
        }
        if !config.alpha.is_finite() {
            return Err(TensorError::InvalidShape(format!(
                "LoRA alpha must be finite, got {}",
                config.alpha
            )));
        }

        let out_features = linear.out_features();
        let in_features = linear.in_features();
        let rank = config.rank;

        // A: [rank, in_features] -- random normal init (Kaiming-like)
        let lora_a = DynTensor::randn(0.0, 1.0, &[rank, in_features], &Device::Cpu)?;

        // B: [out_features, rank] -- zero init (initial output == original)
        let lora_b = DynTensor::zeros(&[out_features, rank], DType::F32, &Device::Cpu)?;

        let scaling = f64::from(config.alpha) / rank as f64;

        // Defense-in-depth: reject scaling that overflows f32
        if (scaling as f32).is_infinite() {
            return Err(TensorError::InvalidShape(format!(
                "LoRA scaling alpha/rank = {scaling} overflows f32"
            )));
        }

        Ok(Self {
            frozen_weight: linear.weight().clone(),
            frozen_bias: linear.bias().cloned(),
            lora_a,
            lora_b,
            scaling,
        })
    }

    /// Create a LoRA adapter from raw tensors (for deserialization / weight loading).
    ///
    /// - `frozen_weight`: shape `[out_features, in_features]`
    /// - `frozen_bias`: optional, shape `[out_features]`
    /// - `lora_a`: shape `[rank, in_features]`
    /// - `lora_b`: shape `[out_features, rank]`
    /// - `scaling`: alpha / rank
    pub fn from_parts(
        frozen_weight: DynTensor,
        frozen_bias: Option<DynTensor>,
        lora_a: DynTensor,
        lora_b: DynTensor,
        scaling: f64,
    ) -> Result<Self> {
        if frozen_weight.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: frozen_weight.rank(),
            });
        }
        if lora_a.rank() != 2 || lora_b.rank() != 2 {
            return Err(TensorError::InvalidShape("LoRA A and B must be 2D".into()));
        }
        // Validate dimensions: A=[rank, in], B=[out, rank], W=[out, in]
        let out = frozen_weight.dims()[0];
        let in_f = frozen_weight.dims()[1];
        let rank = lora_a.dims()[0];
        if lora_a.dims()[1] != in_f {
            return Err(TensorError::shape_mismatch(
                vec![rank, in_f],
                lora_a.dims().to_vec(),
            ));
        }
        if lora_b.dims() != [out, rank] {
            return Err(TensorError::shape_mismatch(
                vec![out, rank],
                lora_b.dims().to_vec(),
            ));
        }
        Ok(Self {
            frozen_weight,
            frozen_bias,
            lora_a,
            lora_b,
            scaling,
        })
    }

    /// Returns references to the trainable parameters `[A, B]`.
    #[must_use]
    pub fn trainable_params(&self) -> Vec<&DynTensor> {
        vec![&self.lora_a, &self.lora_b]
    }

    /// Returns mutable references to the trainable parameters `[A, B]`.
    pub fn trainable_params_mut(&mut self) -> Vec<&mut DynTensor> {
        vec![&mut self.lora_a, &mut self.lora_b]
    }

    /// Merge LoRA weights into the frozen weight for deployment.
    ///
    /// Returns a new `Linear` with `W_merged = W + (alpha/rank) * B @ A`.
    pub fn merge(&self) -> Result<Linear> {
        let merged_weight = self.merged_weight()?;
        Linear::new(merged_weight, self.frozen_bias.clone())
    }

    /// Compute merged weight: `W + scaling * B @ A`.
    pub fn merged_weight(&self) -> Result<DynTensor> {
        // B @ A: [out, rank] @ [rank, in] -> [out, in]
        let ba = self.lora_b.matmul(&self.lora_a)?;
        let scaled_ba = ba.mul_scalar(self.scaling)?;
        let merged = self.frozen_weight.add(&scaled_ba)?;
        Ok(merged)
    }

    /// Frozen weight reference.
    #[must_use]
    pub fn frozen_weight(&self) -> &DynTensor {
        &self.frozen_weight
    }

    /// Frozen bias reference (if present).
    #[must_use]
    pub fn frozen_bias(&self) -> Option<&DynTensor> {
        self.frozen_bias.as_ref()
    }

    /// LoRA A matrix reference.
    #[must_use]
    pub fn lora_a(&self) -> &DynTensor {
        &self.lora_a
    }

    /// LoRA B matrix reference.
    #[must_use]
    pub fn lora_b(&self) -> &DynTensor {
        &self.lora_b
    }

    /// Set LoRA A matrix (for weight loading).
    pub fn set_lora_a(&mut self, a: DynTensor) -> Result<()> {
        if a.dims() != self.lora_a.dims() {
            return Err(TensorError::shape_mismatch(
                self.lora_a.dims().to_vec(),
                a.dims().to_vec(),
            ));
        }
        self.lora_a = a;
        Ok(())
    }

    /// Set LoRA B matrix (for weight loading).
    pub fn set_lora_b(&mut self, b: DynTensor) -> Result<()> {
        if b.dims() != self.lora_b.dims() {
            return Err(TensorError::shape_mismatch(
                self.lora_b.dims().to_vec(),
                b.dims().to_vec(),
            ));
        }
        self.lora_b = b;
        Ok(())
    }

    /// Scaling factor (alpha / rank).
    #[must_use]
    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    /// LoRA rank (number of columns in A / columns in B).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.lora_a.dims()[0]
    }
}

impl Module for LoraLinear {
    /// Forward pass using efficient two-matmul path.
    ///
    /// `y = x @ W^T + (x @ A^T) @ B^T * scaling + bias`
    ///
    /// This avoids materializing the full `[out, in]` LoRA delta matrix.
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        // Base: x @ W^T
        let wt = self.frozen_weight.transpose(0, 1)?;
        let base = x.matmul(&wt)?;

        // LoRA: (x @ A^T) @ B^T * scaling
        let at = self.lora_a.transpose(0, 1)?;
        let bt = self.lora_b.transpose(0, 1)?;
        let lora_out = x.matmul(&at)?.matmul(&bt)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear(out: usize, inp: usize) -> Linear {
        let w = DynTensor::randn(0.0, 0.1, &[out, inp], &Device::Cpu).unwrap();
        let b = DynTensor::zeros(&[out], DType::F32, &Device::Cpu).unwrap();
        Linear::new(w, Some(b)).unwrap()
    }

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 8);
        assert!((config.alpha - 8.0).abs() < 1e-6);
        assert!(config.dropout.is_none());
    }

    #[test]
    fn test_lora_config_with_dropout() {
        let config = LoraConfig::new(4, 4.0).with_dropout(0.1);
        assert_eq!(config.rank, 4);
        assert!((config.dropout.unwrap() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_lora_linear_construction() {
        let linear = make_linear(16, 8);
        let config = LoraConfig::new(4, 4.0);
        let lora = LoraLinear::from_linear(&linear, &config).unwrap();
        assert_eq!(lora.rank(), 4);
        assert!((lora.scaling() - 1.0).abs() < 1e-10);
        assert_eq!(lora.trainable_params().len(), 2);
        assert_eq!(lora.lora_a().dims(), &[4, 8]);
        assert_eq!(lora.lora_b().dims(), &[16, 4]);
    }

    #[test]
    fn test_lora_zero_rank_error() {
        let linear = make_linear(16, 8);
        let config = LoraConfig::new(0, 4.0);
        assert!(LoraLinear::from_linear(&linear, &config).is_err());
    }

    #[test]
    fn test_lora_nan_alpha_error() {
        let linear = make_linear(16, 8);
        let config = LoraConfig::new(4, f32::NAN);
        assert!(LoraLinear::from_linear(&linear, &config).is_err());
    }

    #[test]
    fn test_lora_initial_output_matches_linear() {
        let linear = make_linear(16, 8);
        let config = LoraConfig::new(4, 4.0);
        let lora = LoraLinear::from_linear(&linear, &config).unwrap();

        let x = DynTensor::randn(0.0, 1.0, &[2, 8], &Device::Cpu).unwrap();
        let y_linear = linear.forward(&x).unwrap();
        let y_lora = lora.forward(&x).unwrap();

        // B is zero-initialized, so LoRA contribution is zero
        let diff = y_linear
            .sub(&y_lora)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "initial LoRA output should match Linear, got max diff {diff}"
        );
    }

    #[test]
    fn test_lora_merge_initial_matches_frozen() {
        let linear = make_linear(16, 8);
        let w_original = linear.weight().clone();
        let config = LoraConfig::new(4, 4.0);
        let lora = LoraLinear::from_linear(&linear, &config).unwrap();

        let merged_w = lora.merged_weight().unwrap();
        let diff = merged_w
            .sub(&w_original)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-6,
            "initial merged weight should match frozen, got max diff {diff}"
        );
    }

    #[test]
    fn test_lora_merge_returns_linear() {
        let linear = make_linear(16, 8);
        let config = LoraConfig::new(4, 4.0);
        let lora = LoraLinear::from_linear(&linear, &config).unwrap();
        let merged = lora.merge().unwrap();
        assert_eq!(merged.out_features(), 16);
        assert_eq!(merged.in_features(), 8);
    }

    #[test]
    fn test_lora_forward_shape() {
        let linear = make_linear(16, 8);
        let config = LoraConfig::new(4, 4.0);
        let lora = LoraLinear::from_linear(&linear, &config).unwrap();

        // 2D input
        let x = DynTensor::randn(0.0, 1.0, &[3, 8], &Device::Cpu).unwrap();
        let y = lora.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 16]);

        // 3D input (batched)
        let x3 = DynTensor::randn(0.0, 1.0, &[2, 3, 8], &Device::Cpu).unwrap();
        let y3 = lora.forward(&x3).unwrap();
        assert_eq!(y3.dims(), &[2, 3, 16]);
    }

    #[test]
    fn test_lora_from_parts() {
        let w = DynTensor::randn(0.0, 0.1, &[16, 8], &Device::Cpu).unwrap();
        let a = DynTensor::randn(0.0, 1.0, &[4, 8], &Device::Cpu).unwrap();
        let b = DynTensor::zeros(&[16, 4], DType::F32, &Device::Cpu).unwrap();
        let lora = LoraLinear::from_parts(w, None, a, b, 1.0).unwrap();
        assert_eq!(lora.rank(), 4);
    }

    #[test]
    fn test_lora_from_parts_shape_mismatch() {
        let w = DynTensor::randn(0.0, 0.1, &[16, 8], &Device::Cpu).unwrap();
        let a = DynTensor::randn(0.0, 1.0, &[4, 7], &Device::Cpu).unwrap(); // wrong in_features
        let b = DynTensor::zeros(&[16, 4], DType::F32, &Device::Cpu).unwrap();
        assert!(LoraLinear::from_parts(w, None, a, b, 1.0).is_err());
    }

    #[test]
    fn test_lora_set_weights() {
        let linear = make_linear(16, 8);
        let config = LoraConfig::new(4, 4.0);
        let mut lora = LoraLinear::from_linear(&linear, &config).unwrap();

        let new_a = DynTensor::randn(0.0, 0.5, &[4, 8], &Device::Cpu).unwrap();
        let new_b = DynTensor::randn(0.0, 0.5, &[16, 4], &Device::Cpu).unwrap();
        lora.set_lora_a(new_a.clone()).unwrap();
        lora.set_lora_b(new_b).unwrap();

        let diff_a = lora
            .lora_a()
            .sub(&new_a)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff_a < 1e-7);
    }

    #[test]
    fn test_lora_set_wrong_shape_error() {
        let linear = make_linear(16, 8);
        let config = LoraConfig::new(4, 4.0);
        let mut lora = LoraLinear::from_linear(&linear, &config).unwrap();

        let bad_a = DynTensor::randn(0.0, 1.0, &[8, 8], &Device::Cpu).unwrap();
        assert!(lora.set_lora_a(bad_a).is_err());
    }
}
