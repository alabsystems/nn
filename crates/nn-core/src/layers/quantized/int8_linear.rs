// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! INT8 quantized linear layer (W8A16): INT8 weights, FP32 activations.
//!
//! CPU path: dequantize weights to F32, then standard matmul.
//! GPU path (future): W8A16 Metal kernel with on-the-fly dequantization.
//!
//! Memory savings: 4x vs F32 weights (1 byte/weight + 4 bytes/channel scale).
//! For a 768x768 linear layer: 589 KB (INT8) vs 2,359 KB (F32).
//!
//! Part of #3522

use super::int8::{dequantize_per_channel, quantize_per_channel, Int8Mode, Int8QuantParams};
use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, Linear, Module};
use crate::{Result, TensorError};

/// INT8 quantized linear layer.
///
/// Stores weights as INT8 (U8 tensor) with per-channel F32 scales.
/// Forward pass dequantizes to F32 then computes `y = x @ W^T + bias`.
///
/// # Construction
///
/// ```ignore
/// // From an existing Linear layer (symmetric quantization)
/// let int8_linear = Int8Linear::from_linear(&linear, Int8Mode::Symmetric)?;
///
/// // Forward works identically
/// let output = int8_linear.forward(&input)?;
/// ```
///
/// # Memory layout
///
/// - `weight_int8`: `[out_features, in_features]` as DType::U8
///   (stores i8 values reinterpreted as u8)
/// - `scale`: per-channel F32 scales, length = out_features
/// - `zero_point`: per-channel i8 zero points, length = out_features
///   (all zeros for symmetric mode)
/// - `bias`: optional `[out_features]` F32 tensor
#[derive(Debug, Clone)]
pub struct Int8Linear {
    /// Quantized weight tensor [out_features, in_features] as U8.
    weight_int8: DynTensor,
    /// Per-channel quantization parameters (scale + zero_point).
    params: Int8QuantParams,
    /// Optional bias [out_features] as F32.
    bias: Option<DynTensor>,
    /// Original weight shape for validation.
    out_features: usize,
    in_features: usize,
}

impl Int8Linear {
    /// Create an Int8Linear from pre-quantized components.
    ///
    /// # Arguments
    /// - `weight_int8`: quantized weights `[out_features, in_features]` as U8
    /// - `params`: per-channel scale and zero_point
    /// - `bias`: optional bias `[out_features]`
    ///
    /// # Errors
    /// Returns error if shapes are inconsistent.
    pub fn new(
        weight_int8: DynTensor,
        params: Int8QuantParams,
        bias: Option<DynTensor>,
    ) -> Result<Self> {
        let (out_features, in_features) = weight_int8.dims2()?;

        if params.scale.len() != out_features {
            return Err(TensorError::DataLengthMismatch {
                expected: out_features,
                actual: params.scale.len(),
            });
        }
        if params.zero_point.len() != out_features {
            return Err(TensorError::DataLengthMismatch {
                expected: out_features,
                actual: params.zero_point.len(),
            });
        }

        if let Some(ref b) = bias {
            if b.dims() != [out_features] {
                return Err(TensorError::shape_mismatch(
                    vec![out_features],
                    b.dims().to_vec(),
                ));
            }
        }

        Ok(Self {
            weight_int8,
            params,
            bias,
            out_features,
            in_features,
        })
    }

    /// Quantize an existing `Linear` layer to INT8.
    ///
    /// Extracts the weight tensor, quantizes per-channel, preserves the bias.
    pub fn from_linear(linear: &Linear, mode: Int8Mode) -> Result<Self> {
        let (quantized, params) = quantize_per_channel(linear.weight(), mode)?;
        Self::new(quantized, params, linear.bias().cloned())
    }

    /// Dequantize back to a standard `Linear` layer (for verification/comparison).
    pub fn dequantize(&self) -> Result<Linear> {
        let weight_f32 = dequantize_per_channel(&self.weight_int8, &self.params)?;
        Linear::new(weight_f32, self.bias.clone())
    }

    /// Number of output features.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Number of input features.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Reference to quantization parameters.
    #[must_use]
    pub fn quant_params(&self) -> &Int8QuantParams {
        &self.params
    }

    /// Reference to the INT8 weight tensor.
    #[must_use]
    pub fn weight_int8(&self) -> &DynTensor {
        &self.weight_int8
    }

    /// Reference to the bias tensor (if present).
    #[must_use]
    pub fn bias(&self) -> Option<&DynTensor> {
        self.bias.as_ref()
    }

    /// Memory footprint in bytes (INT8 weights + scales + zero_points + optional bias).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let weight_bytes = self.out_features * self.in_features; // 1 byte per weight
        let scale_bytes = self.out_features * 4; // f32 per channel
        let zp_bytes = self.out_features; // i8 per channel
        let bias_bytes = self.bias.as_ref().map_or(0, |_| self.out_features * 4);
        weight_bytes + scale_bytes + zp_bytes + bias_bytes
    }

    /// Equivalent F32 memory footprint (for comparison).
    #[must_use]
    pub fn f32_memory_bytes(&self) -> usize {
        let weight_bytes = self.out_features * self.in_features * 4;
        let bias_bytes = self.bias.as_ref().map_or(0, |_| self.out_features * 4);
        weight_bytes + bias_bytes
    }

    /// Compression ratio vs F32.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        self.f32_memory_bytes() as f32 / self.memory_bytes() as f32
    }
}

impl Module for Int8Linear {
    /// Forward pass: dequantize INT8 weights to F32, then matmul.
    ///
    /// Input shape: `[*, in_features]` (any number of batch dimensions).
    /// Output shape: `[*, out_features]`.
    ///
    /// CPU path (Phase 1): Full dequantize → standard matmul.
    /// GPU path (Phase 2, future): W8A16 Metal kernel with tile-level dequant.
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        // Validate input last dimension matches in_features
        let x_dims = x.dims();
        if x_dims.is_empty() {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: 0,
            });
        }
        let x_last = x_dims[x_dims.len() - 1];
        if x_last != self.in_features {
            return Err(TensorError::shape_mismatch(
                vec![self.in_features],
                vec![x_last],
            ));
        }

        // CPU path: dequantize then standard matmul
        // Future: GPU path dispatches W8A16 kernel directly
        let weight_f32 = dequantize_per_channel(&self.weight_int8, &self.params)?;

        // Move weight to same device as input
        let weight_on_device = weight_f32.to_device(&x.device())?;

        let out = x.matmul(&weight_on_device.t()?)?;
        let out = match &self.bias {
            Some(bias) => {
                let bias_on_device = bias.to_device(&x.device())?;
                out.broadcast_add(&bias_on_device)?
            }
            None => out,
        };

        check_output_finite(&out, "Int8Linear")?;
        Ok(out)
    }
}
