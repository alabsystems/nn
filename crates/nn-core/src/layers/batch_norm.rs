// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Batch normalization layers using frozen running statistics.
//!
//! - [`BatchNorm`]: General batch normalization for any rank >= 2 input.
//! - [`BatchNorm2d`]: 4D-specific (`[B, C, H, W]`) batch normalization matching
//!   PyTorch's `nn.BatchNorm2d`. Required by DocLayout-YOLO and every CNN
//!   detection model.

use super::{check_output_finite, validate_eps, Module};
use crate::dyn_tensor::trace::{TraceOp, WeightRef};
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

// -- BatchNormConfig ----------------------------------------------------------

/// Configuration for batch normalization.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct BatchNormConfig {
    /// Small constant for numerical stability. Default: 1e-5.
    pub eps: f64,
    /// Whether to subtract the mean (true = full BN, false = variance-only). Default: true.
    pub remove_mean: bool,
    /// Whether the layer has learnable affine parameters. Default: true.
    pub affine: bool,
    /// Momentum for running statistics update (unused in inference). Default: 0.1.
    pub momentum: f64,
}

impl BatchNormConfig {
    /// Create config with custom epsilon (most common customization point).
    ///
    /// Other fields use defaults: `remove_mean = true`, `affine = true`, `momentum = 0.1`.
    #[must_use]
    pub fn new(eps: f64) -> Self {
        Self {
            eps,
            ..Default::default()
        }
    }

    /// Set epsilon for numerical stability.
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set whether to subtract the mean.
    #[must_use]
    pub fn with_remove_mean(mut self, remove_mean: bool) -> Self {
        self.remove_mean = remove_mean;
        self
    }

    /// Set whether the layer has learnable affine parameters.
    #[must_use]
    pub fn with_affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    /// Set momentum for running statistics update.
    #[must_use]
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Default for BatchNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        }
    }
}

// -- BatchNorm ----------------------------------------------------------------

/// Batch normalization layer using frozen running statistics.
///
/// Matches candle-nn `BatchNorm` in inference mode. Input: `[B, C, ...]`.
/// Normalizes per-channel using pre-computed running mean/var from training.
///
/// dvoice calls `forward_t(x, false)` exclusively — inference only.
/// Training-mode forward is not yet implemented.
#[derive(Debug, Clone)]
pub struct BatchNorm {
    running_mean: DynTensor,
    running_var: DynTensor,
    weight: Option<DynTensor>,
    bias: Option<DynTensor>,
    remove_mean: bool,
    eps: f64,
}

impl BatchNorm {
    /// Create from pre-loaded tensors (running stats + optional affine).
    ///
    /// Returns an error if `eps` is not finite or is negative.
    pub fn new(
        running_mean: DynTensor,
        running_var: DynTensor,
        weight: Option<DynTensor>,
        bias: Option<DynTensor>,
        eps: f64,
    ) -> Result<Self> {
        validate_eps(eps, "BatchNorm")?;
        Ok(Self {
            running_mean,
            running_var,
            weight,
            bias,
            remove_mean: true,
            eps,
        })
    }

    /// Create with full config (including remove_mean control).
    ///
    /// Returns an error if `config.eps` is not finite or is negative.
    pub fn with_config(
        running_mean: DynTensor,
        running_var: DynTensor,
        weight: Option<DynTensor>,
        bias: Option<DynTensor>,
        config: BatchNormConfig,
    ) -> Result<Self> {
        validate_eps(config.eps, "BatchNorm")?;
        Ok(Self {
            running_mean,
            running_var,
            weight,
            bias,
            remove_mean: config.remove_mean,
            eps: config.eps,
        })
    }

    /// Running mean tensor (shape `[C]`).
    #[must_use]
    pub fn running_mean(&self) -> &DynTensor {
        &self.running_mean
    }

    /// Running variance tensor (shape `[C]`).
    #[must_use]
    pub fn running_var(&self) -> &DynTensor {
        &self.running_var
    }

    /// Affine weight (gamma), if present.
    #[must_use]
    pub fn weight(&self) -> Option<&DynTensor> {
        self.weight.as_ref()
    }

    /// Affine bias (beta), if present.
    #[must_use]
    pub fn bias(&self) -> Option<&DynTensor> {
        self.bias.as_ref()
    }

    /// Inference forward: normalize using frozen running statistics.
    fn forward_eval(&self, x: &DynTensor) -> Result<DynTensor> {
        let rank = x.rank();
        if rank < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: rank,
            });
        }

        // Try fused GPU dispatch first (#4324). Single kernel replaces ~6
        // separate GPU dispatches in the decomposed path below.
        // Skip GPU path when remove_mean is false: the fused kernel always
        // subtracts running_mean, which is incorrect for variance-only
        // normalization. Fall through to the CPU path instead.
        if self.remove_mean && x.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| {
                b.batch_norm(
                    x,
                    &self.running_mean,
                    &self.running_var,
                    self.weight.as_ref(),
                    self.bias.as_ref(),
                    self.eps,
                )
            }) {
                return result;
            }
        }

        // CPU path (and GPU fallback when fused kernel returns None).

        // Broadcast shape: [1, C, 1, 1, ...] for [B, C, ...]
        let num_features = x.dim(1)?;
        let mut broadcast_shape = vec![1usize; rank];
        broadcast_shape[1] = num_features;

        let mean = self.running_mean.reshape(&broadcast_shape)?;
        let var = self.running_var.reshape(&broadcast_shape)?;

        // x_normalized = (x - mean) / sqrt(var + eps)
        let x = if self.remove_mean {
            x.broadcast_sub(&mean)?
        } else {
            x.clone()
        };
        let inv_std = var.add_scalar(self.eps)?.sqrt()?.recip()?;
        let x_norm = x.broadcast_mul(&inv_std)?;

        // Affine: y = x_norm * weight + bias
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => {
                let w = w.reshape(&broadcast_shape)?;
                let b = b.reshape(&broadcast_shape)?;
                x_norm.broadcast_mul(&w)?.broadcast_add(&b)
            }
            (Some(w), None) => {
                let w = w.reshape(&broadcast_shape)?;
                x_norm.broadcast_mul(&w)
            }
            (None, Some(b)) => {
                let b = b.reshape(&broadcast_shape)?;
                x_norm.broadcast_add(&b)
            }
            (None, None) => Ok(x_norm),
        }
    }
}

impl Module for BatchNorm {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        super::traced_forward(
            &[x],
            || {
                Ok(TraceOp::BatchNorm {
                    eps: self.eps,
                    weight: self
                        .weight
                        .as_ref()
                        .map(DynTensor::to_weight_ref)
                        .transpose()?
                        .unwrap_or_else(|| WeightRef::from_shape(&[])),
                    bias: self
                        .bias
                        .as_ref()
                        .map(DynTensor::to_weight_ref)
                        .transpose()?
                        .unwrap_or_else(|| WeightRef::from_shape(&[])),
                    running_mean: self.running_mean.to_weight_ref()?,
                    running_var: self.running_var.to_weight_ref()?,
                })
            },
            || {
                let result = self.forward_eval(x)?;
                check_output_finite(&result, "BatchNorm")?;
                Ok(result)
            },
        )
    }
}

// -- BatchNorm2d --------------------------------------------------------------

/// 2D batch normalization for `[B, C, H, W]` inputs.
///
/// Matches PyTorch's `nn.BatchNorm2d` API: constructs from `num_features`
/// (number of channels), loads `weight`, `bias`, `running_mean`, `running_var`
/// from a [`VarBuilder`], and enforces that the input is exactly 4D.
///
/// Internally delegates to [`BatchNorm`] for the normalization math.
///
/// # Forward
///
/// `y = (x - running_mean) / sqrt(running_var + eps) * weight + bias`
///
/// where statistics are broadcast to `[1, C, 1, 1]`.
///
/// # Example
///
/// ```ignore
/// let bn2d = BatchNorm2d::load(&vb.pp("backbone.bn1"), 64, BatchNormConfig::default())?;
/// let y = bn2d.forward(&x)?; // x: [B, 64, H, W]
/// ```
#[derive(Debug, Clone)]
pub struct BatchNorm2d {
    inner: BatchNorm,
    num_features: usize,
}

impl BatchNorm2d {
    /// Create from pre-loaded tensors.
    ///
    /// - `num_features`: number of channels (C dimension).
    /// - `running_mean`, `running_var`: shape `[num_features]`.
    /// - `weight`, `bias`: optional affine parameters, shape `[num_features]`.
    /// - `eps`: small constant for numerical stability.
    pub fn new(
        num_features: usize,
        running_mean: DynTensor,
        running_var: DynTensor,
        weight: Option<DynTensor>,
        bias: Option<DynTensor>,
        eps: f64,
    ) -> Result<Self> {
        let inner = BatchNorm::new(running_mean, running_var, weight, bias, eps)?;
        Ok(Self {
            inner,
            num_features,
        })
    }

    /// Create with full config.
    pub fn with_config(
        num_features: usize,
        running_mean: DynTensor,
        running_var: DynTensor,
        weight: Option<DynTensor>,
        bias: Option<DynTensor>,
        config: BatchNormConfig,
    ) -> Result<Self> {
        let inner = BatchNorm::with_config(running_mean, running_var, weight, bias, config)?;
        Ok(Self {
            inner,
            num_features,
        })
    }

    /// Load from a [`VarBuilder`].
    ///
    /// Loads PyTorch-standard names:
    /// - `"running_mean"`: `[num_features]`
    /// - `"running_var"`: `[num_features]`
    /// - `"weight"` (gamma): `[num_features]` (if `config.affine`)
    /// - `"bias"` (beta): `[num_features]` (if `config.affine`)
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        num_features: usize,
        config: BatchNormConfig,
    ) -> Result<Self> {
        let inner = BatchNorm::load(vb, num_features, config)?;
        Ok(Self {
            inner,
            num_features,
        })
    }

    /// Number of channels this layer normalizes over.
    #[must_use]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Access the underlying [`BatchNorm`] layer.
    #[must_use]
    pub fn inner(&self) -> &BatchNorm {
        &self.inner
    }

    /// Running mean tensor (shape `[C]`).
    #[must_use]
    pub fn running_mean(&self) -> &DynTensor {
        self.inner.running_mean()
    }

    /// Running variance tensor (shape `[C]`).
    #[must_use]
    pub fn running_var(&self) -> &DynTensor {
        self.inner.running_var()
    }

    /// Affine weight (gamma), if present.
    #[must_use]
    pub fn weight(&self) -> Option<&DynTensor> {
        self.inner.weight()
    }

    /// Affine bias (beta), if present.
    #[must_use]
    pub fn bias(&self) -> Option<&DynTensor> {
        self.inner.bias()
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let rank = x.rank();
        if rank != 4 {
            return Err(TensorError::RankMismatch {
                expected: 4,
                actual: rank,
            });
        }
        let channels = x.dim(1)?;
        if channels != self.num_features {
            return Err(TensorError::shape_mismatch(
                vec![0, self.num_features, 0, 0],
                x.dims().to_vec(),
            ));
        }
        self.inner.forward(x)
    }
}

#[cfg(kani)]
#[path = "kani_norm_proofs.rs"]
mod kani_norm_proofs;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device};

    fn make_bn(channels: usize) -> BatchNorm {
        let mean = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
        let var = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
        BatchNorm::new(mean, var, None, None, 1e-5).unwrap()
    }

    fn make_bn2d(channels: usize) -> BatchNorm2d {
        let mean = DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap();
        let var = DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap();
        BatchNorm2d::new(channels, mean, var, None, None, 1e-5).unwrap()
    }

    #[test]
    fn test_batch_norm_normal_input() {
        let bn = make_bn(2);
        let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &Device::Cpu)
            .unwrap();
        let result = bn.forward(&x);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batch_norm_nan_input_returns_error() {
        let bn = make_bn(2);
        let mut data = vec![1.0f32; 6];
        data[0] = f32::NAN;
        let x = DynTensor::from_vec(data, &[1, 2, 3], &Device::Cpu).unwrap();
        let result = bn.forward(&x);
        assert!(result.is_err(), "NaN input should produce an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Non-finite") || msg.contains("NaN"),
            "error should mention non-finite: {msg}"
        );
    }

    #[test]
    fn test_batch_norm_inf_input_returns_error() {
        let bn = make_bn(2);
        let mut data = vec![1.0f32; 6];
        data[0] = f32::INFINITY;
        let x = DynTensor::from_vec(data, &[1, 2, 3], &Device::Cpu).unwrap();
        let result = bn.forward(&x);
        assert!(result.is_err(), "Inf input should produce an error");
    }

    // -- BatchNorm2d tests -------------------------------------------------------

    #[test]
    fn test_batch_norm_2d_known_values() {
        // 2 channels, running_mean=[2.0, 5.0], running_var=[1.0, 4.0], eps=0
        // Input [B=1, C=2, H=1, W=3]:
        //   ch0: [1.0, 2.0, 3.0], ch1: [4.0, 5.0, 6.0]
        // Expected (no affine):
        //   ch0: (x-2)/sqrt(1) = [-1, 0, 1]
        //   ch1: (x-5)/sqrt(4) = [-0.5, 0, 0.5]
        let running_mean = DynTensor::new(&[2.0, 5.0], &[2], &Device::Cpu).unwrap();
        let running_var = DynTensor::new(&[1.0, 4.0], &[2], &Device::Cpu).unwrap();
        let bn = BatchNorm2d::new(2, running_mean, running_var, None, None, 0.0).unwrap();
        let x =
            DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 1, 3], &Device::Cpu).unwrap();
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2, 1, 3]);
        let vals = y.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - (-1.0)).abs() < 1e-5, "ch0[0]: {}", vals[0]);
        assert!(vals[1].abs() < 1e-5, "ch0[1]: {}", vals[1]);
        assert!((vals[2] - 1.0).abs() < 1e-5, "ch0[2]: {}", vals[2]);
        assert!((vals[3] - (-0.5)).abs() < 1e-5, "ch1[0]: {}", vals[3]);
        assert!(vals[4].abs() < 1e-5, "ch1[1]: {}", vals[4]);
        assert!((vals[5] - 0.5).abs() < 1e-5, "ch1[2]: {}", vals[5]);
    }

    #[test]
    fn test_batch_norm_2d_with_affine() {
        // running_mean=0, running_var=1, weight=2, bias=10, eps=0
        // x = [1.0] at [1,1,1,1] → normalized = 1.0 → *2 + 10 = 12.0
        let running_mean = DynTensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
        let running_var = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
        let weight = DynTensor::full(&[1], 2.0, DType::F32, &Device::Cpu).unwrap();
        let bias = DynTensor::full(&[1], 10.0, DType::F32, &Device::Cpu).unwrap();
        let bn =
            BatchNorm2d::new(1, running_mean, running_var, Some(weight), Some(bias), 0.0).unwrap();
        let x = DynTensor::new(&[1.0], &[1, 1, 1, 1], &Device::Cpu).unwrap();
        let y = bn.forward(&x).unwrap();
        let vals = y.to_flat_vec::<f32>().unwrap();
        assert!((vals[0] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_norm_2d_batched() {
        // Verify batch dimension is handled correctly
        let bn = make_bn2d(2);
        let x = DynTensor::ones(&[3, 2, 4, 4], DType::F32, &Device::Cpu).unwrap();
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 2, 4, 4]);
    }

    #[test]
    fn test_batch_norm_2d_rejects_3d() {
        let bn = make_bn2d(2);
        let x = DynTensor::ones(&[1, 2, 4], DType::F32, &Device::Cpu).unwrap();
        let err = bn.forward(&x).unwrap_err();
        assert!(
            matches!(
                err,
                TensorError::RankMismatch {
                    expected: 4,
                    actual: 3
                }
            ),
            "expected RankMismatch {{4, 3}}, got: {err:?}"
        );
    }

    #[test]
    fn test_batch_norm_2d_rejects_5d() {
        let bn = make_bn2d(2);
        let x = DynTensor::ones(&[1, 2, 3, 4, 5], DType::F32, &Device::Cpu).unwrap();
        let err = bn.forward(&x).unwrap_err();
        assert!(
            matches!(
                err,
                TensorError::RankMismatch {
                    expected: 4,
                    actual: 5
                }
            ),
            "expected RankMismatch {{4, 5}}, got: {err:?}"
        );
    }

    #[test]
    fn test_batch_norm_2d_channel_mismatch() {
        let bn = make_bn2d(3);
        let x = DynTensor::ones(&[1, 2, 4, 4], DType::F32, &Device::Cpu).unwrap();
        let err = bn.forward(&x).unwrap_err();
        assert!(
            matches!(err, TensorError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got: {err:?}"
        );
    }

    #[test]
    fn test_batch_norm_2d_accessors() {
        let rm = DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap();
        let rv = DynTensor::new(&[3.0, 4.0], &[2], &Device::Cpu).unwrap();
        let w = DynTensor::new(&[5.0, 6.0], &[2], &Device::Cpu).unwrap();
        let b = DynTensor::new(&[7.0, 8.0], &[2], &Device::Cpu).unwrap();
        let bn = BatchNorm2d::new(2, rm, rv, Some(w), Some(b), 1e-5).unwrap();
        assert_eq!(bn.num_features(), 2);
        assert_eq!(bn.running_mean().dims(), &[2]);
        assert_eq!(bn.running_var().dims(), &[2]);
        assert!(bn.weight().is_some());
        assert!(bn.bias().is_some());
        assert_eq!(bn.inner().running_mean().dims(), &[2]);
    }

    #[test]
    fn test_batch_norm_2d_nan_input_returns_error() {
        let bn = make_bn2d(1);
        let x = DynTensor::new(&[f32::NAN], &[1, 1, 1, 1], &Device::Cpu).unwrap();
        assert!(bn.forward(&x).is_err(), "NaN input should return error");
    }

    #[test]
    fn test_batch_norm_2d_pytorch_parity() {
        // PyTorch reference:
        //   bn = nn.BatchNorm2d(3, eps=1e-5)
        //   bn.running_mean = tensor([0.5, -0.5, 1.0])
        //   bn.running_var  = tensor([2.0, 0.5, 4.0])
        //   bn.weight       = tensor([1.0, 2.0, 0.5])
        //   bn.bias         = tensor([0.0, 1.0, -1.0])
        //   x[0,0,:,:] = 1.5 → normalized = (1.5-0.5)/sqrt(2+1e-5) = 0.70710...
        //   → affine = 0.70710 * 1.0 + 0.0 = 0.70710
        let running_mean = DynTensor::new(&[0.5, -0.5, 1.0], &[3], &Device::Cpu).unwrap();
        let running_var = DynTensor::new(&[2.0, 0.5, 4.0], &[3], &Device::Cpu).unwrap();
        let weight = DynTensor::new(&[1.0, 2.0, 0.5], &[3], &Device::Cpu).unwrap();
        let bias = DynTensor::new(&[0.0, 1.0, -1.0], &[3], &Device::Cpu).unwrap();
        let bn =
            BatchNorm2d::new(3, running_mean, running_var, Some(weight), Some(bias), 1e-5).unwrap();
        // x: [1, 3, 1, 1] with values [1.5, 0.5, 3.0]
        let x = DynTensor::new(&[1.5, 0.5, 3.0], &[1, 3, 1, 1], &Device::Cpu).unwrap();
        let y = bn.forward(&x).unwrap();
        let vals = y.to_flat_vec::<f32>().unwrap();
        // ch0: (1.5-0.5)/sqrt(2+1e-5) * 1.0 + 0.0 = 1.0/1.41421... = 0.70711
        let expected_ch0 = 1.0 / (2.0_f32 + 1e-5).sqrt() * 1.0 + 0.0;
        assert!(
            (vals[0] - expected_ch0).abs() < 1e-4,
            "ch0: expected {expected_ch0}, got {}",
            vals[0]
        );
        // ch1: (0.5-(-0.5))/sqrt(0.5+1e-5) * 2.0 + 1.0 = 1.0/0.70711*2.0+1.0 = 3.82842...
        let expected_ch1 = 1.0 / (0.5_f32 + 1e-5).sqrt() * 2.0 + 1.0;
        assert!(
            (vals[1] - expected_ch1).abs() < 1e-4,
            "ch1: expected {expected_ch1}, got {}",
            vals[1]
        );
        // ch2: (3.0-1.0)/sqrt(4.0+1e-5) * 0.5 + (-1.0) = 2.0/2.0*0.5-1.0 = -0.5
        let expected_ch2 = 2.0 / (4.0_f32 + 1e-5).sqrt() * 0.5 + (-1.0);
        assert!(
            (vals[2] - expected_ch2).abs() < 1e-4,
            "ch2: expected {expected_ch2}, got {}",
            vals[2]
        );
    }
}
