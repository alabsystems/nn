// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPTQ quantized weight loader for INT4 group-quantized models.
//!
//! GPTQ (Frantar et al., 2022) stores weights as packed uint32 tensors where
//! each uint32 contains 8 INT4 values (4 bits each). Scales and zero-points
//! are stored per-group (e.g., every 128 input channels share one scale/zp).
//!
//! # Format layout (AutoGPTQ / HuggingFace convention)
//!
//! - `qweight`: `[in_features / 8, out_features]` packed uint32
//!   Each uint32 packs 8 consecutive INT4 values along the in_features axis.
//! - `qzeros`: `[groups, out_features / 8]` packed uint32
//!   Each uint32 packs 8 INT4 zero-point values.
//! - `scales`: `[groups, out_features]` float32
//! - `g_idx`: optional `[in_features]` permutation (act_order=True)
//!
//! Dequantization: `w_f32 = (q_int4 - zero_point) * scale`
//!
//! Part of #3863

use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, Module};
use crate::{DType, Result, TensorError};

/// GPTQ format configuration.
#[derive(Debug, Clone)]
pub struct GptqFormat {
    /// Number of input features sharing one scale/zero-point.
    /// Typical values: 32, 64, 128.
    pub group_size: usize,
    /// Quantization bit width (currently only 4 is supported).
    pub bits: u8,
    /// Whether activation reordering was used during quantization.
    /// When true, weights are permuted by a `g_idx` tensor.
    pub act_order: bool,
}

impl Default for GptqFormat {
    fn default() -> Self {
        Self {
            group_size: 128,
            bits: 4,
            act_order: false,
        }
    }
}

/// INT4 values packed in a uint32 in GPTQ format.
const INT4_PER_U32: usize = 8;
/// Bit-width of a single INT4 value.
const INT4_BITS: u32 = 4;
/// Mask for extracting one INT4 value.
const INT4_MASK: u32 = 0xF;

/// Unpack GPTQ qweight from packed uint32 to individual INT4 values as f32.
///
/// Input: `[in_features / 8, out_features]` uint32 packed tensor.
/// Output: `[in_features, out_features]` f32 tensor with values in [0, 15].
///
/// Each uint32 contains 8 INT4 values: bits [3:0] → first value,
/// bits [7:4] → second value, ..., bits [31:28] → eighth value.
///
/// # Errors
///
/// Returns error if input tensor is not 2D or not U32 dtype.
pub fn unpack_gptq_qweight(packed: &DynTensor) -> Result<DynTensor> {
    let (packed_rows, out_features) = packed.dims2()?;
    if packed.dtype() != DType::U32 {
        return Err(TensorError::dtype_mismatch(DType::U32, packed.dtype()));
    }

    let packed_data = packed.to_flat_vec::<u32>()?;
    let in_features = packed_rows * INT4_PER_U32;
    let mut unpacked = vec![0.0_f32; in_features * out_features];

    for row in 0..packed_rows {
        for col in 0..out_features {
            let packed_val = packed_data[row * out_features + col];
            for bit_idx in 0..INT4_PER_U32 {
                let int4_val = (packed_val >> (bit_idx as u32 * INT4_BITS)) & INT4_MASK;
                let out_row = row * INT4_PER_U32 + bit_idx;
                unpacked[out_row * out_features + col] = int4_val as f32;
            }
        }
    }

    DynTensor::from_vec(unpacked, &[in_features, out_features], &packed.device())
}

/// Unpack GPTQ qzeros from packed uint32 to individual zero-point values as f32.
///
/// Input: `[groups, out_features / 8]` uint32 packed tensor.
/// Output: `[groups, out_features]` f32 tensor with values in [0, 15].
///
/// # Errors
///
/// Returns error if input tensor is not 2D or not U32 dtype.
pub fn unpack_gptq_qzeros(packed: &DynTensor) -> Result<DynTensor> {
    let (groups, packed_cols) = packed.dims2()?;
    if packed.dtype() != DType::U32 {
        return Err(TensorError::dtype_mismatch(DType::U32, packed.dtype()));
    }

    let packed_data = packed.to_flat_vec::<u32>()?;
    let out_features = packed_cols * INT4_PER_U32;
    let mut unpacked = vec![0.0_f32; groups * out_features];

    for group in 0..groups {
        for packed_col in 0..packed_cols {
            let packed_val = packed_data[group * packed_cols + packed_col];
            for bit_idx in 0..INT4_PER_U32 {
                let zp_val = (packed_val >> (bit_idx as u32 * INT4_BITS)) & INT4_MASK;
                let out_col = packed_col * INT4_PER_U32 + bit_idx;
                unpacked[group * out_features + out_col] = zp_val as f32;
            }
        }
    }

    DynTensor::from_vec(unpacked, &[groups, out_features], &packed.device())
}

/// Dequantize INT4 weight values using per-group scales and zero-points.
///
/// Formula: `w_f32[i, j] = (q_int4[i, j] - zeros[group, j]) * scales[group, j]`
/// where `group = i / group_size`.
///
/// # Arguments
/// - `q_weight`: `[in_features, out_features]` unpacked INT4 values as f32
/// - `scales`: `[groups, out_features]` per-group scale factors
/// - `zeros`: `[groups, out_features]` per-group zero-points
/// - `group_size`: number of input features per group
///
/// # Errors
///
/// Returns error on shape mismatches.
pub(crate) fn dequantize_gptq(
    q_weight: &DynTensor,
    scales: &DynTensor,
    zeros: &DynTensor,
    group_size: usize,
) -> Result<DynTensor> {
    let (in_features, out_features) = q_weight.dims2()?;
    let (num_groups, scale_out) = scales.dims2()?;
    let (zero_groups, zero_out) = zeros.dims2()?;

    if scale_out != out_features {
        return Err(TensorError::shape_mismatch(
            vec![num_groups, out_features],
            vec![num_groups, scale_out],
        ));
    }
    if zero_groups != num_groups || zero_out != out_features {
        return Err(TensorError::shape_mismatch(
            vec![num_groups, out_features],
            vec![zero_groups, zero_out],
        ));
    }

    let expected_groups = in_features.div_ceil(group_size);
    if num_groups != expected_groups {
        return Err(TensorError::DataLengthMismatch {
            expected: expected_groups,
            actual: num_groups,
        });
    }

    let q_data = q_weight.to_flat_vec::<f32>()?;
    let scale_data = scales.to_flat_vec::<f32>()?;
    let zero_data = zeros.to_flat_vec::<f32>()?;

    let mut result = vec![0.0_f32; in_features * out_features];

    for i in 0..in_features {
        let group = i / group_size;
        for j in 0..out_features {
            let q_val = q_data[i * out_features + j];
            let scale = scale_data[group * out_features + j];
            let zp = zero_data[group * out_features + j];
            result[i * out_features + j] = (q_val - zp) * scale;
        }
    }

    DynTensor::from_vec(result, &[in_features, out_features], &q_weight.device())
}

/// GPTQ/AWQ quantized linear layer.
///
/// Stores dequantized weights as F32 for inference. The dequantization
/// happens at load time (Phase 1). Future Phase 2 will perform on-the-fly
/// dequantization during matmul for memory savings.
///
/// # Memory
///
/// Phase 1 (current): Full F32 materialization at load. Same runtime memory
/// as unquantized, but loads from 4-bit safetensors format.
///
/// Phase 2 (future): Keep packed INT4 weights, dequantize per-tile in GPU
/// kernel. ~8x memory savings vs F32.
#[derive(Debug, Clone)]
pub struct GptqLinear {
    /// Dequantized weight tensor `[out_features, in_features]` as F32.
    weight: DynTensor,
    /// Optional bias `[out_features]` as F32.
    bias: Option<DynTensor>,
    /// Original quantization format metadata.
    format: GptqFormat,
    /// Output feature count (for validation).
    out_features: usize,
    /// Input feature count (for validation).
    in_features: usize,
}

impl GptqLinear {
    /// Create a `GptqLinear` from pre-dequantized weight and optional bias.
    ///
    /// # Arguments
    /// - `weight`: dequantized weight `[out_features, in_features]` as F32
    /// - `bias`: optional bias `[out_features]` as F32
    /// - `format`: GPTQ format metadata
    ///
    /// # Errors
    /// Returns error if weight is not 2D or bias shape is inconsistent.
    pub fn new(weight: DynTensor, bias: Option<DynTensor>, format: GptqFormat) -> Result<Self> {
        let (out_features, in_features) = weight.dims2()?;

        if let Some(ref b) = bias {
            if b.dims() != [out_features] {
                return Err(TensorError::shape_mismatch(
                    vec![out_features],
                    b.dims().to_vec(),
                ));
            }
        }

        Ok(Self {
            weight,
            bias,
            format,
            out_features,
            in_features,
        })
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

    /// Reference to the GPTQ format configuration.
    #[must_use]
    pub fn format(&self) -> &GptqFormat {
        &self.format
    }

    /// Reference to the dequantized weight tensor.
    #[must_use]
    pub fn weight(&self) -> &DynTensor {
        &self.weight
    }

    /// Reference to the bias tensor (if present).
    #[must_use]
    pub fn bias(&self) -> Option<&DynTensor> {
        self.bias.as_ref()
    }
}

impl Module for GptqLinear {
    /// Forward pass: `y = x @ W^T + bias`.
    ///
    /// Input shape: `[*, in_features]` (any number of batch dimensions).
    /// Output shape: `[*, out_features]`.
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
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

        let weight_on_device = self.weight.to_device(&x.device())?;
        let out = x.matmul(&weight_on_device.t()?)?;
        let out = match &self.bias {
            Some(bias) => {
                let bias_on_device = bias.to_device(&x.device())?;
                out.broadcast_add(&bias_on_device)?
            }
            None => out,
        };

        check_output_finite(&out, "GptqLinear")?;
        Ok(out)
    }
}

/// Load a GPTQ quantized linear layer from packed weight components.
///
/// Unpacks qweight and qzeros, dequantizes using scales, and constructs
/// a `GptqLinear` ready for inference.
///
/// # Arguments
/// - `qweight`: `[in_features / 8, out_features]` packed uint32 tensor
/// - `scales`: `[groups, out_features]` float32 scale factors
/// - `qzeros`: `[groups, out_features / 8]` packed uint32 zero-points
/// - `bias`: optional `[out_features]` float32 bias
/// - `group_size`: features per group (e.g. 128)
///
/// # Errors
///
/// Returns error on shape mismatches or invalid packed data.
pub fn load_gptq_linear(
    qweight: &DynTensor,
    scales: &DynTensor,
    qzeros: &DynTensor,
    bias: Option<DynTensor>,
    group_size: usize,
) -> Result<GptqLinear> {
    let unpacked_weight = unpack_gptq_qweight(qweight)?;
    let unpacked_zeros = unpack_gptq_qzeros(qzeros)?;

    let dequantized = dequantize_gptq(&unpacked_weight, scales, &unpacked_zeros, group_size)?;

    // Transpose from [in_features, out_features] to [out_features, in_features]
    let weight = dequantized.t()?;

    let format = GptqFormat {
        group_size,
        bits: 4,
        act_order: false,
    };

    GptqLinear::new(weight, bias, format)
}
