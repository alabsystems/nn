// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Quantized matrix multiplication for [`DynTensor`].
//!
//! Provides fused dequantize-and-multiply operations that dequantize quantized
//! weight blocks on the fly during matrix multiplication, avoiding full
//! materialization of the dequantized weight matrix. This is the reference CPU
//! implementation for verification; GPU backends can implement optimized
//! fused kernels that match the numerical behavior defined here.
//!
//! Supported formats:
//! - **Q4_0**: 4-bit symmetric with f16 block scale (18 bytes / 32 elements)
//! - **Q8_0**: 8-bit symmetric with f16 block scale (34 bytes / 32 elements)
//!
//! ## Operations
//!
//! - [`quantized_matmul_q4_0`]: `x @ W^T` where `W` is Q4_0 quantized
//! - [`quantized_matmul_q8_0`]: `x @ W^T` where `W` is Q8_0 quantized
//! - [`quantized_linear`]: `x @ W^T + bias` (linear layer with quantized weights)

use crate::dyn_tensor::quantized::{QuantType, QuantizedStorage};
use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

/// Errors specific to quantized matrix multiplication.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum QuantizedMatmulError {
    /// The weight tensor is not quantized.
    #[error("weight tensor is not quantized (expected {expected})")]
    NotQuantized {
        /// The expected quantization type.
        expected: String,
    },

    /// The weight tensor has the wrong quantization type.
    #[error("weight quantization type mismatch: expected {expected}, got {actual}")]
    WrongQuantType {
        /// Expected quantization type.
        expected: QuantType,
        /// Actual quantization type.
        actual: QuantType,
    },

    /// The input tensor dimensions are incompatible with the quantized weight.
    #[error(
        "quantized matmul shape mismatch: input last dim {input_k} \
         != weight in_features {weight_k}"
    )]
    ShapeMismatch {
        /// Input tensor's last dimension (K).
        input_k: usize,
        /// Weight tensor's in_features dimension (K).
        weight_k: usize,
    },

    /// The weight tensor must be 2-D `[out_features, in_features]`.
    #[error("quantized weight must be 2-D [out_features, in_features], got rank {rank}")]
    WeightRank {
        /// Actual rank of the weight tensor.
        rank: usize,
    },

    /// The bias shape does not match the output features.
    #[error("bias shape mismatch: expected [{expected}], got {actual:?}")]
    BiasMismatch {
        /// Expected bias size (out_features).
        expected: usize,
        /// Actual bias dimensions.
        actual: Vec<usize>,
    },
}

impl From<QuantizedMatmulError> for TensorError {
    fn from(e: QuantizedMatmulError) -> Self {
        Self::Unsupported(e.to_string())
    }
}

/// Validate and extract quantized weight metadata.
///
/// Returns `(quantized_storage, out_features, in_features)`.
fn validate_quantized_weight(
    weight: &DynTensor,
    expected_quant: QuantType,
) -> Result<(Arc<QuantizedStorage>, usize, usize)> {
    let qs = match &weight.storage {
        crate::dyn_tensor::TensorStorage::Quantized(qs) => Arc::clone(qs),
        _ => {
            return Err(QuantizedMatmulError::NotQuantized {
                expected: expected_quant.to_string(),
            }
            .into());
        }
    };
    if qs.quant_type() != expected_quant {
        return Err(QuantizedMatmulError::WrongQuantType {
            expected: expected_quant,
            actual: qs.quant_type(),
        }
        .into());
    }
    let shape = qs.shape();
    if shape.len() != 2 {
        return Err(QuantizedMatmulError::WeightRank { rank: shape.len() }.into());
    }
    let out_features = shape[0];
    let in_features = shape[1];
    Ok((qs, out_features, in_features))
}

/// Validate the input tensor and return its f32 data and batch dimensions.
///
/// The input tensor can be 2-D `[M, K]` or N-D `[..batch, M, K]`.
/// Returns `(f32_array, batch_shape, m, k)` where `batch_shape` is the
/// leading dimensions (empty for 2-D input).
fn validate_input(
    input: &DynTensor,
    in_features: usize,
) -> Result<(ArrayD<f32>, Vec<usize>, usize, usize)> {
    if input.rank() < 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: input.rank(),
        });
    }
    let dims = input.dims();
    let k = dims[dims.len() - 1];
    let m = dims[dims.len() - 2];
    if k != in_features {
        return Err(QuantizedMatmulError::ShapeMismatch {
            input_k: k,
            weight_k: in_features,
        }
        .into());
    }
    let batch_shape: Vec<usize> = dims[..dims.len() - 2].to_vec();
    let input_deq = input.auto_dequantize()?;
    let arr = input_deq.to_f32_array()?;
    Ok((arr, batch_shape, m, k))
}

/// Dequantize a single Q4_0 block and accumulate dot product with an input row.
///
/// This is the inner kernel for quantized matmul: for each block of 32 weight
/// elements, dequantize and multiply with the corresponding 32 input elements.
#[inline]
fn dot_q4_0_block(
    weight_data: &[u8],
    block_offset: usize,
    input_row: &[f32],
    input_offset: usize,
) -> f32 {
    let b = block_offset;
    let scale = half::f16::from_le_bytes([weight_data[b], weight_data[b + 1]]).to_f32();
    let mut acc = 0.0_f32;
    for j in 0..16 {
        let byte = weight_data[b + 2 + j];
        let lo = i32::from(byte & 0x0F) - 8;
        let hi = i32::from((byte >> 4) & 0x0F) - 8;
        acc += input_row[input_offset + 2 * j] * (scale * lo as f32);
        acc += input_row[input_offset + 2 * j + 1] * (scale * hi as f32);
    }
    acc
}

/// Dequantize a single Q8_0 block and accumulate dot product with an input row.
#[inline]
fn dot_q8_0_block(
    weight_data: &[u8],
    block_offset: usize,
    input_row: &[f32],
    input_offset: usize,
) -> f32 {
    let b = block_offset;
    let scale = half::f16::from_le_bytes([weight_data[b], weight_data[b + 1]]).to_f32();
    let mut acc = 0.0_f32;
    for j in 0..32 {
        let q = weight_data[b + 2 + j] as i8;
        acc += input_row[input_offset + j] * (scale * f32::from(q));
    }
    acc
}

/// Compute a single output element: dot product of input row with a quantized
/// weight row, dequantizing block by block.
fn quantized_dot_row(
    weight_data: &[u8],
    quant_type: QuantType,
    row_idx: usize,
    in_features: usize,
    input_row: &[f32],
) -> f32 {
    let block_size = quant_type.block_size();
    let block_bytes = quant_type.block_bytes();
    let num_blocks = in_features / block_size;
    let blocks_per_row = num_blocks;
    let row_byte_offset = row_idx * blocks_per_row * block_bytes;

    let mut acc = 0.0_f32;
    for blk in 0..blocks_per_row {
        let block_offset = row_byte_offset + blk * block_bytes;
        let input_offset = blk * block_size;
        acc += match quant_type {
            QuantType::Q4_0 => dot_q4_0_block(weight_data, block_offset, input_row, input_offset),
            QuantType::Q8_0 => dot_q8_0_block(weight_data, block_offset, input_row, input_offset),
            QuantType::Q4_1 => {
                // Q4_1 fused matmul not yet supported; callers should
                // dequantize first. This branch should not be reached
                // because public entry points only accept Q4_0 and Q8_0.
                0.0
            }
        };
    }
    acc
}

/// Core quantized matmul: `input @ W^T` with on-the-fly block dequantization.
///
/// - `input`: f32 array with shape `[..batch, M, K]`
/// - `weight_data`: raw quantized bytes for weight `[out_features, in_features]`
/// - `quant_type`: Q4_0 or Q8_0
/// - `out_features`, `in_features`: weight dimensions
/// - `batch_shape`: leading batch dimensions (may be empty)
/// - `m`: rows in the matmul (M dimension)
///
/// Returns f32 array with shape `[..batch, M, out_features]`.
fn quantized_matmul_core(
    input: &ArrayD<f32>,
    weight_data: &[u8],
    quant_type: QuantType,
    out_features: usize,
    in_features: usize,
    batch_shape: &[usize],
    m: usize,
) -> Result<ArrayD<f32>> {
    // Compute total batch size.
    let batch_size: usize = batch_shape.iter().copied().product::<usize>().max(1);

    // Output shape: [...batch, M, out_features]
    let mut out_shape: Vec<usize> = batch_shape.to_vec();
    out_shape.push(m);
    out_shape.push(out_features);

    let mut output = ArrayD::<f32>::zeros(IxDyn(&out_shape));
    let input_flat = input.as_slice().ok_or_else(|| {
        TensorError::Unsupported("quantized_matmul requires contiguous input tensor".into())
    })?;
    let output_flat = output
        .as_slice_mut()
        .ok_or_else(|| TensorError::Unsupported("quantized_matmul output not contiguous".into()))?;

    // For each batch element and row, compute the dot product with each
    // output feature's quantized weight row.
    for b in 0..batch_size {
        for row in 0..m {
            let input_start = (b * m + row) * in_features;
            let input_row = &input_flat[input_start..input_start + in_features];
            let output_start = (b * m + row) * out_features;
            for oc in 0..out_features {
                output_flat[output_start + oc] =
                    quantized_dot_row(weight_data, quant_type, oc, in_features, input_row);
            }
        }
    }

    Ok(output)
}

/// Quantized matrix multiplication with Q4_0 weights.
///
/// Computes `input @ weight^T` where `weight` is stored in Q4_0 format.
/// Weight blocks are dequantized on the fly during the dot product, avoiding
/// full materialization of the dequantized weight matrix.
///
/// # Arguments
///
/// - `input`: f32 tensor with shape `[..batch, M, K]`
/// - `weight`: Q4_0 quantized tensor with shape `[N, K]` (out_features, in_features)
///
/// # Returns
///
/// f32 tensor with shape `[..batch, M, N]`
///
/// # Errors
///
/// Returns an error if:
/// - `weight` is not Q4_0 quantized
/// - `weight` is not 2-D
/// - Input last dimension does not match weight in_features
/// - `in_features` is not a multiple of 32 (Q4_0 block size)
pub fn quantized_matmul_q4_0(input: &DynTensor, weight: &DynTensor) -> Result<DynTensor> {
    let (qs, out_features, in_features) = validate_quantized_weight(weight, QuantType::Q4_0)?;
    let (input_arr, batch_shape, m, _k) = validate_input(input, in_features)?;

    let result = quantized_matmul_core(
        &input_arr,
        qs.raw_data(),
        QuantType::Q4_0,
        out_features,
        in_features,
        &batch_shape,
        m,
    )?;

    DynTensor::from_cpu_f32(result)
}

/// Quantized matrix multiplication with Q8_0 weights.
///
/// Computes `input @ weight^T` where `weight` is stored in Q8_0 format.
/// Weight blocks are dequantized on the fly during the dot product, avoiding
/// full materialization of the dequantized weight matrix.
///
/// # Arguments
///
/// - `input`: f32 tensor with shape `[..batch, M, K]`
/// - `weight`: Q8_0 quantized tensor with shape `[N, K]` (out_features, in_features)
///
/// # Returns
///
/// f32 tensor with shape `[..batch, M, N]`
///
/// # Errors
///
/// Returns an error if:
/// - `weight` is not Q8_0 quantized
/// - `weight` is not 2-D
/// - Input last dimension does not match weight in_features
/// - `in_features` is not a multiple of 32 (Q8_0 block size)
pub fn quantized_matmul_q8_0(input: &DynTensor, weight: &DynTensor) -> Result<DynTensor> {
    let (qs, out_features, in_features) = validate_quantized_weight(weight, QuantType::Q8_0)?;
    let (input_arr, batch_shape, m, _k) = validate_input(input, in_features)?;

    let result = quantized_matmul_core(
        &input_arr,
        qs.raw_data(),
        QuantType::Q8_0,
        out_features,
        in_features,
        &batch_shape,
        m,
    )?;

    DynTensor::from_cpu_f32(result)
}

/// Quantized linear layer: `y = input @ weight^T + bias`.
///
/// Computes a linear transformation with quantized weights (Q4_0 or Q8_0).
/// The weight format is determined from the tensor's quantization metadata.
/// Bias is optional and must be an f32 1-D tensor of length `out_features`.
///
/// # Arguments
///
/// - `input`: f32 tensor with shape `[..batch, M, K]`
/// - `weight`: quantized tensor (Q4_0 or Q8_0) with shape `[N, K]`
/// - `bias`: optional f32 tensor with shape `[N]`
///
/// # Returns
///
/// f32 tensor with shape `[..batch, M, N]`
pub fn quantized_linear(
    input: &DynTensor,
    weight: &DynTensor,
    bias: Option<&DynTensor>,
) -> Result<DynTensor> {
    let qs = weight.quantized_storage().ok_or_else(|| {
        TensorError::Unsupported("quantized_linear: weight is not quantized".into())
    })?;
    let quant_type = qs.quant_type();

    // Dispatch to the appropriate quantized matmul.
    let mut result = match quant_type {
        QuantType::Q4_0 => quantized_matmul_q4_0(input, weight)?,
        QuantType::Q8_0 => quantized_matmul_q8_0(input, weight)?,
        QuantType::Q4_1 => {
            // Q4_1 does not have a fused matmul implementation.
            // Fall back to dequantize + standard matmul.
            let weight_deq = weight.dequantize()?;
            let weight_t = weight_deq.t()?;
            input.matmul(&weight_t)?
        }
    };

    // Add bias if present.
    if let Some(b) = bias {
        let b_dims = b.dims();
        let out_features = weight.dims()[0];
        // Bias must be 1-D with length == out_features.
        if b_dims.len() != 1 || b_dims[0] != out_features {
            return Err(QuantizedMatmulError::BiasMismatch {
                expected: out_features,
                actual: b_dims.to_vec(),
            }
            .into());
        }
        result = result.add(b)?;
    }

    Ok(result)
}

#[cfg(test)]
#[path = "quantized_matmul_tests.rs"]
mod tests;
