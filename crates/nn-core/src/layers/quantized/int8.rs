// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! INT8 per-channel quantization utilities for weight-only quantization (W8A16).
//!
//! Supports symmetric (zero_point = 0) and asymmetric per-channel quantization.
//! Symmetric is the default and sufficient for most models — weights are centered
//! near zero after training normalization.
//!
//! # W8A16 scheme
//!
//! - **Weights:** INT8 per-channel (one scale per output channel)
//! - **Activations:** FP16/FP32 (unquantized)
//! - **Compute:** Dequantize weight on-the-fly during matmul: `w_f32 = w_i8 * scale`
//! - **Output:** Same dtype as activation input
//!
//! This gives ~4x memory reduction vs F32 weights with minimal accuracy loss
//! for inference workloads. The CPU reference path dequantizes fully before
//! matmul. The GPU path (future Metal kernel) dequantizes per-tile inside the
//! GEMM for bandwidth efficiency.
//!
//! # Per-channel vs per-tensor
//!
//! Per-channel (one scale per output row of weight matrix) is strictly better
//! than per-tensor for linear layers because different output channels can have
//! very different weight magnitudes. The overhead is negligible: N extra f32
//! values for an [N, K] weight matrix.
//!
//! Part of #3522

use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

/// Per-channel INT8 quantization parameters.
///
/// Each output channel has its own scale (and optionally zero_point).
/// Dequantization formula: `w_f32[i] = (w_i8[i] - zero_point) * scale`
///
/// For symmetric quantization, `zero_point` is all zeros and the formula
/// simplifies to: `w_f32[i] = w_i8[i] * scale`
///
/// `zero_point` is stored as `i32` (not `i8`) because asymmetric quantization
/// can produce zero_point values outside [-128, 127] when the data range does
/// not straddle zero. This matches PyTorch's `qint8` convention (int64 zp).
/// The zero_point is only used during dequantization arithmetic, not stored
/// in the quantized tensor itself.
#[derive(Debug, Clone)]
pub struct Int8QuantParams {
    /// Per-channel scale factors. Length = out_features (one per output channel).
    /// `w_f32 = (w_i8 - zero_point) * scale`
    pub scale: Vec<f32>,
    /// Per-channel zero points. Length = out_features.
    /// For symmetric quantization, all zeros.
    /// Stored as i32 to support asymmetric ranges where zp may exceed i8 bounds.
    pub zero_point: Vec<i32>,
}

/// Quantization mode for INT8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8Mode {
    /// Symmetric: zero_point = 0, range [-127, 127].
    /// Simpler, faster, sufficient for most trained weights.
    Symmetric,
    /// Asymmetric: zero_point calibrated per-channel, range [-128, 127].
    /// Better for weights with non-zero mean (rare after training).
    Asymmetric,
}

/// Quantize a 2D weight matrix from f32 to INT8 with per-channel scaling.
///
/// Input: `weights` with shape `[out_features, in_features]` (F32 on CPU).
/// Output: `(quantized_u8, params)` where `quantized_u8` has shape
/// `[out_features, in_features]` with dtype U8 (storing i8 values as u8).
///
/// For symmetric mode: maps `[-abs_max, abs_max]` to `[-127, 127]`.
/// For asymmetric mode: maps `[min, max]` to `[-128, 127]`.
///
/// # Errors
///
/// Returns error if `weights` is not 2D, not F32, or contains non-finite values.
pub fn quantize_per_channel(
    weights: &DynTensor,
    mode: Int8Mode,
) -> Result<(DynTensor, Int8QuantParams)> {
    let (out_features, in_features) = weights.dims2()?;
    let weight_cpu = weights.to_device(&Device::Cpu)?;
    let flat = weight_cpu.to_flat_vec::<f32>()?;

    // Validate all values are finite
    for (i, &v) in flat.iter().enumerate() {
        if !v.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "quantize_per_channel: non-finite weight value",
            });
        }
        let _ = i; // suppress unused warning
    }

    let mut scales = Vec::with_capacity(out_features);
    let mut zero_points = Vec::with_capacity(out_features);
    let mut quantized = Vec::with_capacity(out_features * in_features);

    for row in 0..out_features {
        let row_start = row * in_features;
        let row_end = row_start + in_features;
        let row_data = &flat[row_start..row_end];

        let (scale, zero_point) = compute_channel_params(row_data, mode);
        scales.push(scale);
        zero_points.push(zero_point);

        // Quantize each element in this row
        for &val in row_data {
            let q_i8 = if scale == 0.0 {
                // All values in this channel are zero (or constant for asymmetric).
                // Pick a q that dequantizes to 0: (q - zp) * 0 = 0 for any q.
                0_i8
            } else {
                let q_f32 = val / scale + zero_point as f32;
                q_f32.round().clamp(-128.0, 127.0) as i8
            };
            // Store i8 as u8 (reinterpret cast)
            quantized.push(q_i8 as u8);
        }
    }

    let quantized_tensor =
        DynTensor::from_vec_u8(quantized, &[out_features, in_features], &Device::Cpu)?;

    let params = Int8QuantParams {
        scale: scales,
        zero_point: zero_points,
    };

    Ok((quantized_tensor, params))
}

/// Dequantize an INT8 weight tensor back to F32.
///
/// Input: `quantized` with shape `[out_features, in_features]` (U8 on CPU),
/// `params` with matching out_features.
/// Output: F32 tensor with shape `[out_features, in_features]`.
///
/// Formula: `w_f32[row][col] = (w_i8[row][col] - zero_point[row]) * scale[row]`
pub fn dequantize_per_channel(
    quantized: &DynTensor,
    params: &Int8QuantParams,
) -> Result<DynTensor> {
    let (out_features, in_features) = quantized.dims2()?;

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

    let q_cpu = quantized.to_device(&Device::Cpu)?;
    let q_data = q_cpu.to_flat_vec::<u8>()?;

    let mut result = Vec::with_capacity(out_features * in_features);

    for row in 0..out_features {
        let scale = params.scale[row];
        let zp = params.zero_point[row];
        let row_start = row * in_features;
        let row_end = row_start + in_features;

        for &q_u8 in &q_data[row_start..row_end] {
            let q_i8 = q_u8 as i8;
            let val = (f32::from(q_i8) - zp as f32) * scale;
            result.push(val);
        }
    }

    DynTensor::from_vec(result, &[out_features, in_features], &Device::Cpu)
}

/// Compute per-channel quantization scale and zero_point for a single row.
///
/// Symmetric: scale = abs_max / 127, zero_point = 0.
/// Asymmetric: scale = (max - min) / 255, zero_point = round(-128 - min / scale).
///
/// The dequantization formula is: `val = (q_i8 - zero_point) * scale`.
/// For asymmetric mode, zero_point is stored as i32 because it can exceed
/// the i8 range when the data does not straddle zero (e.g., all-negative weights).
fn compute_channel_params(row: &[f32], mode: Int8Mode) -> (f32, i32) {
    if row.is_empty() {
        return (0.0, 0);
    }

    match mode {
        Int8Mode::Symmetric => {
            let abs_max = row.iter().copied().fold(0.0_f32, |acc, v| acc.max(v.abs()));
            if abs_max == 0.0 {
                return (0.0, 0);
            }
            let scale = abs_max / 127.0;
            (scale, 0)
        }
        Int8Mode::Asymmetric => {
            let min_val = row.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max_val - min_val;

            if range == 0.0 {
                // Constant channel — scale=0, zero_point maps to constant
                return (0.0, 0);
            }

            let scale = range / 255.0;
            // zero_point: the i32 value such that min maps to q=-128.
            // min = (-128 - zp) * scale  =>  zp = -128 - min/scale
            let zp = (-128.0 - min_val / scale).round() as i32;
            (scale, zp)
        }
    }
}

/// Compute the maximum absolute quantization error for a round-trip.
///
/// Useful for testing: `max_error(original, dequantized)`.
#[must_use]
pub fn max_quantization_error(original: &[f32], dequantized: &[f32]) -> f32 {
    original
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

#[cfg(test)]
#[path = "int8_tests.rs"]
mod tests;
