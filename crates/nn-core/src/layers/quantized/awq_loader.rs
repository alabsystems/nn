// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! AWQ (Activation-aware Weight Quantization) weight loader for INT4 models.
//!
//! AWQ (Lin et al., 2023) uses the same packed uint32 storage format as GPTQ
//! but applies activation-aware channel scaling before quantization.
//! The weight packing/unpacking is identical to GPTQ — both store 8 INT4
//! values per uint32.
//!
//! # Differences from GPTQ
//!
//! - AWQ never uses activation reordering (`act_order` is always false).
//! - AWQ applies per-channel scaling factors to activations at runtime
//!   (handled externally, not in the weight loader).
//! - The packed format is bit-compatible with GPTQ.
//!
//! Part of #3863

use super::gptq_loader::{dequantize_gptq, unpack_gptq_qweight, unpack_gptq_qzeros, GptqLinear};
use crate::dyn_tensor::DynTensor;
use crate::Result;

/// AWQ format configuration.
#[derive(Debug, Clone)]
pub struct AwqFormat {
    /// Number of input features sharing one scale/zero-point.
    /// Typical values: 64, 128.
    pub group_size: usize,
    /// Quantization bit width (currently only 4 is supported).
    pub bits: u8,
}

impl Default for AwqFormat {
    fn default() -> Self {
        Self {
            group_size: 128,
            bits: 4,
        }
    }
}

/// Unpack AWQ qweight from packed uint32 to individual INT4 values as f32.
///
/// AWQ uses the same packing format as GPTQ — delegates directly.
///
/// Input: `[in_features / 8, out_features]` uint32 packed tensor.
/// Output: `[in_features, out_features]` f32 tensor with values in [0, 15].
///
/// # Errors
///
/// Returns error if input tensor is not 2D or not U32 dtype.
pub fn unpack_awq_qweight(packed: &DynTensor) -> Result<DynTensor> {
    unpack_gptq_qweight(packed)
}

/// Load an AWQ quantized linear layer from packed weight components.
///
/// Uses the same unpacking and dequantization as GPTQ (formats are
/// bit-compatible). Returns a `GptqLinear` since the inference path
/// is identical after dequantization.
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
pub fn load_awq_linear(
    qweight: &DynTensor,
    scales: &DynTensor,
    qzeros: &DynTensor,
    bias: Option<DynTensor>,
    group_size: usize,
) -> Result<GptqLinear> {
    let unpacked_weight = unpack_awq_qweight(qweight)?;
    let unpacked_zeros = unpack_gptq_qzeros(qzeros)?;

    let dequantized = dequantize_gptq(&unpacked_weight, scales, &unpacked_zeros, group_size)?;

    // Transpose from [in_features, out_features] to [out_features, in_features]
    let weight = dequantized.t()?;

    let format = super::gptq_loader::GptqFormat {
        group_size,
        bits: 4,
        act_order: false,
    };

    GptqLinear::new(weight, bias, format)
}
