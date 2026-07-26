// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trainable normalization layer wrappers: LayerNorm, RmsNorm, GroupNorm,
//! BatchNorm, InstanceNorm.
//!
//! Extracted from `trainable_extra.rs` for 500-line compliance.

use crate::error::Result;
use crate::tracked::TrackedTensor;
use crate::var::Var;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use std::sync::Arc;

use super::TrainableModule;

/// A layer normalization layer with trainable weight and bias.
///
/// Normalizes over the last dimension, then applies affine transform:
/// `y = (x - mean) / sqrt(var + eps) * weight + bias`
///
/// Weight and bias are both `[normalized_shape]` (the size of the last dimension).
/// Matches `layers::LayerNorm` semantics for training.
#[derive(Debug, Clone)]
pub struct TrainableLayerNorm {
    weight: Var, // [normalized_shape]
    bias: Var,   // [normalized_shape]
    eps: f64,
}

impl TrainableLayerNorm {
    /// Create a new layer norm with weight=1, bias=0 initialization.
    ///
    /// `normalized_shape` is the size of the last dimension to normalize over.
    pub fn new(normalized_shape: usize, eps: f64) -> Result<Self> {
        let weight = Var::new(DynTensor::from_vec(
            vec![1.0f32; normalized_shape],
            &[normalized_shape],
            &Device::Cpu,
        )?);
        let bias = Var::zeros(&[normalized_shape], DType::F32, &Device::Cpu)?;
        Ok(Self { weight, bias, eps })
    }

    /// Create from existing `Var`s.
    pub fn from_vars(weight: Var, bias: Var, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    /// Create from `DynTensor` weight and bias (wraps each in a new `Var`).
    pub fn from_tensors(weight: DynTensor, bias: DynTensor, eps: f64) -> Self {
        Self {
            weight: Var::new(weight),
            bias: Var::new(bias),
            eps,
        }
    }

    /// Reference to the weight `Var`.
    #[must_use]
    pub fn weight(&self) -> &Var {
        &self.weight
    }

    /// Reference to the bias `Var`.
    #[must_use]
    pub fn bias(&self) -> &Var {
        &self.bias
    }
}

impl TrainableModule for TrainableLayerNorm {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let w = Arc::new(TrackedTensor::from_var(&self.weight)?);
        let b = Arc::new(TrackedTensor::from_var(&self.bias)?);
        x.layer_norm(&w, &b, self.eps)
    }

    fn vars(&self) -> Vec<&Var> {
        vec![&self.weight, &self.bias]
    }
}

/// RMS normalization with a trainable scale weight.
///
/// Normalizes by root-mean-square of the input (no centering), then scales:
/// `y = x / rms(x) * weight`
///
/// Weight is `[normalized_shape]` (last dimension). No bias. Matches LLaMA/Qwen usage.
#[derive(Debug, Clone)]
pub struct TrainableRmsNorm {
    weight: Var, // [normalized_shape]
    eps: f64,
}

impl TrainableRmsNorm {
    /// Create with weight=1 initialization.
    pub fn new(normalized_shape: usize, eps: f64) -> Result<Self> {
        let weight = Var::new(DynTensor::from_vec(
            vec![1.0f32; normalized_shape],
            &[normalized_shape],
            &Device::Cpu,
        )?);
        Ok(Self { weight, eps })
    }

    /// Create from an existing `Var`.
    pub fn from_var(weight: Var, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Reference to the weight `Var`.
    #[must_use]
    pub fn weight(&self) -> &Var {
        &self.weight
    }
}

impl TrainableModule for TrainableRmsNorm {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let w = Arc::new(TrackedTensor::from_var(&self.weight)?);
        x.rms_norm(&w, self.eps)
    }

    fn vars(&self) -> Vec<&Var> {
        vec![&self.weight]
    }
}

/// Group normalization with trainable weight and bias.
///
/// Input: `[N, C, *]`. Divides C channels into `num_groups` groups.
/// Weight and bias are `[C]`.
#[derive(Debug, Clone)]
pub struct TrainableGroupNorm {
    weight: Var, // [C]
    bias: Var,   // [C]
    num_groups: usize,
    eps: f64,
}

impl TrainableGroupNorm {
    /// Create with weight=1, bias=0 initialization.
    pub fn new(num_channels: usize, num_groups: usize, eps: f64) -> Result<Self> {
        let weight = Var::new(DynTensor::from_vec(
            vec![1.0f32; num_channels],
            &[num_channels],
            &Device::Cpu,
        )?);
        let bias = Var::zeros(&[num_channels], DType::F32, &Device::Cpu)?;
        Ok(Self {
            weight,
            bias,
            num_groups,
            eps,
        })
    }

    /// Create from existing `Var`s.
    pub fn from_vars(weight: Var, bias: Var, num_groups: usize, eps: f64) -> Self {
        Self {
            weight,
            bias,
            num_groups,
            eps,
        }
    }

    /// Reference to the weight `Var`.
    #[must_use]
    pub fn weight(&self) -> &Var {
        &self.weight
    }

    /// Reference to the bias `Var`.
    #[must_use]
    pub fn bias(&self) -> &Var {
        &self.bias
    }
}

impl TrainableModule for TrainableGroupNorm {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let w = Arc::new(TrackedTensor::from_var(&self.weight)?);
        let b = Arc::new(TrackedTensor::from_var(&self.bias)?);
        x.group_norm(&w, &b, self.num_groups, self.eps)
    }

    fn vars(&self) -> Vec<&Var> {
        vec![&self.weight, &self.bias]
    }
}

/// Batch normalization (training mode) with trainable weight and bias.
///
/// Input: `[N, C, *]`. Computes mean/variance over batch and spatial dims.
/// Weight and bias are `[C]`.
#[derive(Debug, Clone)]
pub struct TrainableBatchNorm {
    weight: Var, // [C]
    bias: Var,   // [C]
    eps: f64,
}

impl TrainableBatchNorm {
    /// Create with weight=1, bias=0 initialization.
    pub fn new(num_channels: usize, eps: f64) -> Result<Self> {
        let weight = Var::new(DynTensor::from_vec(
            vec![1.0f32; num_channels],
            &[num_channels],
            &Device::Cpu,
        )?);
        let bias = Var::zeros(&[num_channels], DType::F32, &Device::Cpu)?;
        Ok(Self { weight, bias, eps })
    }

    /// Create from existing `Var`s.
    pub fn from_vars(weight: Var, bias: Var, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    /// Reference to the weight `Var`.
    #[must_use]
    pub fn weight(&self) -> &Var {
        &self.weight
    }

    /// Reference to the bias `Var`.
    #[must_use]
    pub fn bias(&self) -> &Var {
        &self.bias
    }
}

impl TrainableModule for TrainableBatchNorm {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let w = Arc::new(TrackedTensor::from_var(&self.weight)?);
        let b = Arc::new(TrackedTensor::from_var(&self.bias)?);
        x.batch_norm(&w, &b, self.eps)
    }

    fn vars(&self) -> Vec<&Var> {
        vec![&self.weight, &self.bias]
    }
}

/// Instance normalization with trainable weight and bias.
///
/// Input: `[N, C, *]`. Normalizes each (N, C) slice independently.
/// Weight and bias are `[C]`.
#[derive(Debug, Clone)]
pub struct TrainableInstanceNorm {
    weight: Var, // [C]
    bias: Var,   // [C]
    eps: f64,
}

impl TrainableInstanceNorm {
    /// Create with weight=1, bias=0 initialization.
    pub fn new(num_channels: usize, eps: f64) -> Result<Self> {
        let weight = Var::new(DynTensor::from_vec(
            vec![1.0f32; num_channels],
            &[num_channels],
            &Device::Cpu,
        )?);
        let bias = Var::zeros(&[num_channels], DType::F32, &Device::Cpu)?;
        Ok(Self { weight, bias, eps })
    }

    /// Create from existing `Var`s.
    pub fn from_vars(weight: Var, bias: Var, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    /// Reference to the weight `Var`.
    #[must_use]
    pub fn weight(&self) -> &Var {
        &self.weight
    }

    /// Reference to the bias `Var`.
    #[must_use]
    pub fn bias(&self) -> &Var {
        &self.bias
    }
}

impl TrainableModule for TrainableInstanceNorm {
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let w = Arc::new(TrackedTensor::from_var(&self.weight)?);
        let b = Arc::new(TrackedTensor::from_var(&self.bias)?);
        x.instance_norm(&w, &b, self.eps)
    }

    fn vars(&self) -> Vec<&Var> {
        vec![&self.weight, &self.bias]
    }
}
