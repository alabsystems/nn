// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Instance normalization for [`DynTensor`].
//!
//! [`InstanceNorm`]: normalizes per-channel per-sample over spatial dims.
//! [`AdaIn`] (style-conditioned affine) is in the sibling `adain` module.

use super::{check_output_finite, validate_eps, Module};
use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::{Result, TensorError};
use ndarray::{Axis, IxDyn};

/// Precision mode for InstanceNorm CPU computation.
///
/// Controls whether the CPU path uses F64 accumulation (mathematical accuracy)
/// or F32 throughout (matching PyTorch ATen CPU behavior).
///
/// - [`F64`](InstanceNormPrecision::F64) (default): F64 accumulation, single rounding to F32.
///   Validated for dvoice (<22 chained operations, #1121).
/// - [`MatchPyTorchCpu`](InstanceNormPrecision::MatchPyTorchCpu): F32 throughout, matching
///   PyTorch ATen CPU behavior. Use when AC1 parity with PyTorch is the metric and the
///   model chains 20+ InstanceNorm operations (e.g., Kokoro's 58).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InstanceNormPrecision {
    #[default]
    F64,
    MatchPyTorchCpu,
}

/// Instance normalization: per-channel per-sample normalization over spatial dims.
///
/// Input: `[B, C, T]` (or higher-rank `[B, C, *spatial]`).
/// Normalizes over all spatial dims for each `(batch, channel)` pair.
///
/// `y = (x - mean) / sqrt(var + eps)`
///
/// No learnable affine parameters — use [`AdaIn`] for style-conditioned affine.
#[derive(Debug, Clone, Copy)]
pub struct InstanceNorm {
    eps: f64,
    precision: InstanceNormPrecision,
}

impl InstanceNorm {
    /// Create with specified epsilon for numerical stability.
    ///
    /// Uses [`InstanceNormPrecision::F64`] accumulation by default.
    /// Returns an error if `eps` is not finite or is negative.
    pub fn new(eps: f64) -> Result<Self> {
        validate_eps(eps, "InstanceNorm")?;
        Ok(Self {
            eps,
            precision: InstanceNormPrecision::F64,
        })
    }

    /// Create with specified epsilon and precision mode.
    ///
    /// Use [`InstanceNormPrecision::MatchPyTorchCpu`] when parity with PyTorch
    /// CPU is the metric and the model chains 20+ InstanceNorm operations.
    pub fn with_precision(eps: f64, precision: InstanceNormPrecision) -> Result<Self> {
        validate_eps(eps, "InstanceNorm")?;
        Ok(Self { eps, precision })
    }

    /// Epsilon used for numerical stability.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Normalize input `[B, C, *spatial]` per-channel per-sample.
    pub(crate) fn forward_norm(&self, x: &DynTensor) -> Result<DynTensor> {
        let dims = x.dims();
        if dims.len() < 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: dims.len(),
            });
        }

        // GPU path: fused kernel (#2040)
        if x.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.instance_norm(x, self.eps)) {
                return result;
            }
        }

        let batch = dims[0];
        let channels = dims[1];
        let spatial = crate::tensor::checked_dim_product(&dims[2..])?;
        let input_dtype = x.dtype();
        let input_device = x.device();

        // GPU fallback: move to CPU for ndarray path (to_f32_array requires CPU).
        // contiguous() ensures standard layout so into_shape_with_order succeeds
        // on tensors from narrow/transpose. (#2683)
        let x_f32 = if input_device.is_gpu() {
            x.to_device(&crate::Device::Cpu)?
                .contiguous()?
                .to_f32_array()?
        } else {
            x.contiguous()?.to_f32_array()?
        };

        let result_arr = match self.precision {
            InstanceNormPrecision::MatchPyTorchCpu => {
                // F32-only path matching PyTorch ATen CPU behavior.
                // No type conversion, identical computation to PyTorch. (#2691)
                let eps_f32 = self.eps as f32;
                let x_flat = x_f32
                    .into_shape_with_order(IxDyn(&[batch, channels, spatial]))
                    .map_err(|e| TensorError::InvalidShape(format!("InstanceNorm reshape: {e}")))?;
                let mean = x_flat
                    .mean_axis(Axis(2))
                    .ok_or_else(|| TensorError::InvalidShape("empty spatial dim".into()))?
                    .insert_axis(Axis(2));
                let centered = &x_flat - &mean;
                let var = centered
                    .mapv(|v| v * v)
                    .mean_axis(Axis(2))
                    .ok_or_else(|| TensorError::InvalidShape("empty spatial dim".into()))?
                    .insert_axis(Axis(2));
                let std_inv = (&var + eps_f32).mapv(|v| 1.0 / v.sqrt());
                let normed = &centered * &std_inv;
                normed
                    .into_shape_with_order(IxDyn(dims))
                    .map_err(|e| TensorError::InvalidShape(format!("InstanceNorm reshape: {e}")))?
            }
            InstanceNormPrecision::F64 => {
                // F64 accumulation: essential for 22+ chained InstanceNorm ops
                // matching PyTorch cuDNN precision. dvoice validated in #1121. (#2688)
                let x_f64 = x_f32.mapv(f64::from);
                let x_flat = x_f64
                    .into_shape_with_order(IxDyn(&[batch, channels, spatial]))
                    .map_err(|e| TensorError::InvalidShape(format!("InstanceNorm reshape: {e}")))?;
                let mean = x_flat
                    .mean_axis(Axis(2))
                    .ok_or_else(|| TensorError::InvalidShape("empty spatial dim".into()))?
                    .insert_axis(Axis(2));
                let centered = &x_flat - &mean;
                let var = centered
                    .mapv(|v| v * v)
                    .mean_axis(Axis(2))
                    .ok_or_else(|| TensorError::InvalidShape("empty spatial dim".into()))?
                    .insert_axis(Axis(2));
                let std_inv = (&var + self.eps).mapv(|v| 1.0 / v.sqrt());
                let normed_f64 = &centered * &std_inv;
                normed_f64
                    .mapv(|v| v as f32)
                    .into_shape_with_order(IxDyn(dims))
                    .map_err(|e| TensorError::InvalidShape(format!("InstanceNorm reshape: {e}")))?
            }
        };

        let mut result = DynTensor::from_cpu_f32(result_arr)?;
        if input_dtype != crate::DType::F32 {
            result = result.to_dtype(input_dtype)?;
        }
        if input_device.is_gpu() {
            result.to_device(&input_device)
        } else {
            Ok(result)
        }
    }
}

impl Module for InstanceNorm {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        super::traced_forward(
            &[x],
            || Ok(TraceOp::InstanceNorm { eps: self.eps }),
            || {
                let result = self.forward_norm(x)?;
                check_output_finite(&result, "InstanceNorm")?;
                Ok(result)
            },
        )
    }
}

#[cfg(test)]
#[path = "instance_norm_tests.rs"]
mod tests;
