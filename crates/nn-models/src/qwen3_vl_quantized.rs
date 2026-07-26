// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Quantized Qwen3-VL-30B-A3B weight loading via GPTQ/AWQ.
//!
//! Extends [`Qwen3VLConfig`] with quantization parameters for INT4
//! group-quantized MoE models. The 30B-A3B variant uses 60 experts with
//! top-2 routing, yielding ~3B active parameters per token despite 30B
//! total parameters.
//!
//! # Supported formats
//!
//! - **GPTQ** (Frantar et al., 2022): Post-training quantization with
//!   activation-order permutation. Uses `GptqLinear` from
//!   `nn_core::layers::quantized`.
//! - **AWQ** (Lin et al., 2023): Activation-aware weight quantization.
//!   Bit-compatible with GPTQ storage. Uses `AwqFormat` from
//!   `nn_core::layers::quantized`.
//!
//! # MoE configuration
//!
//! - Total parameters: ~30B
//! - Active parameters per token: ~3B
//! - Experts: 60 total, top-2 routing
//! - Decoder layers: 48
//!
//! Part of #3897

use super::qwen3_vl::Qwen3VLConfig;
use nn_core::layers::quantized::{AwqFormat, GptqFormat};
use nn_core::{Result, TensorError};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Quantization method enum
// ---------------------------------------------------------------------------

/// Quantization method used for weight compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuantMethod {
    /// GPTQ (Frantar et al., 2022): post-training quantization with
    /// optional activation reordering.
    Gptq,
    /// AWQ (Lin et al., 2023): activation-aware weight quantization.
    /// Bit-compatible with GPTQ storage format.
    Awq,
}

impl std::fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gptq => write!(f, "GPTQ"),
            Self::Awq => write!(f, "AWQ"),
        }
    }
}

// ---------------------------------------------------------------------------
// Quantized config
// ---------------------------------------------------------------------------

/// Configuration for a quantized Qwen3-VL model.
///
/// Extends [`Qwen3VLConfig`] with quantization-specific parameters that
/// describe how linear layer weights are compressed for memory-efficient
/// inference.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Qwen3VLQuantConfig {
    /// Base model configuration (architecture, dimensions, MoE).
    pub base: Qwen3VLConfig,
    /// Quantization method (GPTQ or AWQ).
    pub quant_method: QuantMethod,
    /// Quantization bit width. Currently only 4 is supported.
    pub bits: u8,
    /// Number of input features sharing one scale/zero-point pair.
    /// Typical values: 32, 64, 128.
    pub group_size: usize,
    /// Whether activation reordering (desc_act) was used during
    /// quantization. Always false for AWQ; may be true for GPTQ.
    pub desc_act: bool,
}

impl Qwen3VLQuantConfig {
    /// Create a new quantized config from base config and quant params.
    #[must_use]
    pub fn new(
        base: Qwen3VLConfig,
        quant_method: QuantMethod,
        bits: u8,
        group_size: usize,
        desc_act: bool,
    ) -> Self {
        Self {
            base,
            quant_method,
            bits,
            group_size,
            desc_act,
        }
    }

    /// Create a GPTQ-quantized 30B-A3B MoE preset.
    ///
    /// - 60 experts, top-2 routing (~3B active params)
    /// - INT4 with group_size=128, desc_act=true
    /// - 48 decoder layers
    #[must_use]
    pub fn preset_30b_a3b_gptq() -> Self {
        Self {
            base: base_30b_a3b_moe(),
            quant_method: QuantMethod::Gptq,
            bits: 4,
            group_size: 128,
            desc_act: true,
        }
    }

    /// Create an AWQ-quantized 30B-A3B MoE preset.
    ///
    /// - 60 experts, top-2 routing (~3B active params)
    /// - INT4 with group_size=128, desc_act=false (AWQ never reorders)
    /// - 48 decoder layers
    #[must_use]
    pub fn preset_30b_a3b_awq() -> Self {
        Self {
            base: base_30b_a3b_moe(),
            quant_method: QuantMethod::Awq,
            bits: 4,
            group_size: 128,
            desc_act: false,
        }
    }

    /// Validate the full quantized configuration for consistency.
    ///
    /// Checks both the base model config and quantization parameters:
    /// - Base config passes [`Qwen3VLConfig::validate`]
    /// - Bit width is 4 (only supported value)
    /// - Group size is non-zero and a power of two
    /// - AWQ never uses desc_act
    /// - MoE model has valid expert counts
    /// - Hidden/intermediate sizes are divisible by group_size
    pub fn validate(&self) -> Result<()> {
        // Validate base config first
        self.base.validate()?;

        // Bit width must be 4
        if self.bits != 4 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLQuantConfig: only 4-bit quantization is supported",
            });
        }

        // Group size must be non-zero
        if self.group_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLQuantConfig: group_size must be > 0",
            });
        }

        // Group size must be a power of two
        if !self.group_size.is_power_of_two() {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLQuantConfig: group_size must be a power of two",
            });
        }

        // AWQ never uses activation reordering
        if self.quant_method == QuantMethod::Awq && self.desc_act {
            return Err(TensorError::ValueOutOfRange {
                description:
                    "Qwen3VLQuantConfig: AWQ does not support desc_act (activation reordering)",
            });
        }

        // Hidden size should be divisible by group_size for clean grouping
        if !self.base.hidden_size.is_multiple_of(self.group_size) {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLQuantConfig: hidden_size must be divisible by group_size",
            });
        }

        // Intermediate size should be divisible by group_size
        if !self.base.intermediate_size.is_multiple_of(self.group_size) {
            return Err(TensorError::ValueOutOfRange {
                description:
                    "Qwen3VLQuantConfig: intermediate_size must be divisible by group_size",
            });
        }

        Ok(())
    }

    /// Build a [`GptqFormat`] from the quantization parameters.
    ///
    /// # Errors
    ///
    /// Returns error if `quant_method` is not GPTQ.
    pub fn to_gptq_format(&self) -> Result<GptqFormat> {
        if self.quant_method != QuantMethod::Gptq {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLQuantConfig: to_gptq_format requires quant_method == Gptq",
            });
        }
        Ok(GptqFormat {
            group_size: self.group_size,
            bits: self.bits,
            act_order: self.desc_act,
        })
    }

    /// Build an [`AwqFormat`] from the quantization parameters.
    ///
    /// # Errors
    ///
    /// Returns error if `quant_method` is not AWQ.
    pub fn to_awq_format(&self) -> Result<AwqFormat> {
        if self.quant_method != QuantMethod::Awq {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLQuantConfig: to_awq_format requires quant_method == Awq",
            });
        }
        Ok(AwqFormat {
            group_size: self.group_size,
            bits: self.bits,
        })
    }

    /// Whether this is a Mixture-of-Experts model.
    #[must_use]
    pub fn is_moe(&self) -> bool {
        self.base.is_moe()
    }

    /// Total number of experts (0 = dense model).
    #[must_use]
    pub fn num_experts(&self) -> usize {
        self.base.num_experts
    }

    /// Number of active experts per token.
    #[must_use]
    pub fn active_experts(&self) -> usize {
        self.base.active_experts
    }

    /// Estimated model memory in bytes (INT4 quantized linear layers).
    ///
    /// Approximate: counts only linear layer weights as INT4 + per-group
    /// scale/zero overhead. Embeddings and norms are kept in F32.
    #[must_use]
    pub fn estimated_memory_bytes(&self) -> usize {
        let h = self.base.hidden_size;
        let i = self.base.intermediate_size;
        let n_layers = self.base.num_layers;
        let vocab = self.base.vocab_size;
        let groups_per_group_size = self.group_size;

        // Per decoder layer: attention (Q, K, V, O) + MLP (gate, up, down)
        // Q: [h, h], K: [kv_dim, h], V: [kv_dim, h], O: [h, h]
        let kv_dim = self.base.num_kv_heads * self.base.head_dim();
        // Q: [h,h] + K: [kv_dim,h] + V: [kv_dim,h] + O: [h,h]
        #[allow(clippy::suspicious_operation_groupings)]
        let attn_params = h * h + kv_dim * h + kv_dim * h + h * h;
        let mlp_params = if self.base.is_moe() {
            // MoE: each expert has gate+up+down; router is small
            let per_expert = i * h + i * h + h * i; // gate + up + down
            let router = h * self.base.num_experts; // routing matrix (F32)
            per_expert * self.base.num_experts + router
        } else {
            i * h + i * h + h * i // gate + up + down
        };

        let linear_params = (attn_params + mlp_params) * n_layers;

        // INT4: 0.5 bytes per param + scale/zp overhead
        let int4_bytes = linear_params / 2;
        let num_groups = linear_params / groups_per_group_size;
        let overhead = num_groups * (4 + 4); // f32 scale + i32 zero_point

        // Embeddings (F32) + LM head (F32) + norms (F32)
        let embed_bytes = vocab * h * 4;
        let lm_head_bytes = vocab * h * 4;
        let norm_bytes = h * 4 * 2 * n_layers; // 2 norms per layer

        int4_bytes + overhead + embed_bytes + lm_head_bytes + norm_bytes
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Base 30B-A3B MoE architecture config with 60 experts, top-2 routing.
fn base_30b_a3b_moe() -> Qwen3VLConfig {
    Qwen3VLConfig {
        hidden_size: 3584,
        num_heads: 28,
        num_kv_heads: 4,
        intermediate_size: 2560,
        num_layers: 48,
        vocab_size: 152064,
        vision_hidden: 1280,
        vision_heads: 16,
        vision_layers: 32,
        vision_patch_size: 14,
        vision_temporal_patch: 2,
        rms_norm_eps: 1e-6,
        num_experts: 60,
        active_experts: 2,
    }
}

// ---------------------------------------------------------------------------
// Error type for quantized layer operations
// ---------------------------------------------------------------------------

/// Errors specific to GPTQ INT4 quantized linear layer operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum QuantizedLayerError {
    /// Group size must be a positive power of two.
    #[error("invalid group_size {group_size}: must be > 0 and a power of two")]
    InvalidGroupSize {
        /// The invalid group size that was provided.
        group_size: usize,
    },

    /// Total weight elements must be divisible by group_size.
    #[error(
        "in_features * out_features ({in_features} * {out_features}) \
         must be divisible by group_size ({group_size})"
    )]
    GroupSizeNotDivisible {
        /// Input feature count.
        in_features: usize,
        /// Output feature count.
        out_features: usize,
        /// Group size that does not evenly divide total elements.
        group_size: usize,
    },

    /// Packed weight length does not match expected for given dimensions.
    #[error(
        "packed_weights length {actual} does not match expected {expected} \
         for in_features={in_features}, out_features={out_features}"
    )]
    PackedWeightLengthMismatch {
        /// Expected packed length.
        expected: usize,
        /// Actual packed length.
        actual: usize,
        /// Input feature count.
        in_features: usize,
        /// Output feature count.
        out_features: usize,
    },

    /// Scale/zero vector length does not match expected group count.
    #[error("scales/zeros length {actual} does not match expected {expected} groups")]
    ScaleZeroLengthMismatch {
        /// Expected number of groups.
        expected: usize,
        /// Actual vector length.
        actual: usize,
    },

    /// Input vector length does not match in_features.
    #[error("input length {actual} does not match in_features {expected}")]
    InputLengthMismatch {
        /// Expected input length (in_features).
        expected: usize,
        /// Actual input length.
        actual: usize,
    },

    /// Non-finite value detected in dequantized output.
    #[error("non-finite value detected during dequantization at index {index}")]
    NonFiniteValue {
        /// Flat index of the non-finite element.
        index: usize,
    },
}

// ---------------------------------------------------------------------------
// GPTQ INT4 quantized linear layer
// ---------------------------------------------------------------------------

/// Number of INT4 values packed per `u32`.
const INT4_PER_U32: usize = 8;

/// Bit mask for a single INT4 nibble.
const INT4_MASK: u32 = 0xF;

/// GPTQ INT4 quantized linear layer.
///
/// Stores weights as packed `u32` (8 INT4 values per `u32`) with per-group
/// scales and zero points. Dequantization formula per weight element:
///
/// ```text
/// w_f32[idx] = (int4_val - zeros[group]) * scales[group]
/// ```
///
/// where `group = idx / group_size`, and `int4_val` is unpacked from
/// `packed_weights`.
///
/// # Memory layout
///
/// - `packed_weights`: `[ceil(in_features * out_features / 8)]` u32 values.
///   Each u32 stores 8 consecutive INT4 weights in little-endian nibble order
///   (bits [3:0] = first, bits [7:4] = second, ..., bits [31:28] = eighth).
/// - `scales`: `[num_groups]` f32 -- one scale per group.
/// - `zeros`: `[num_groups]` f32 -- one zero point per group.
///
/// Groups are formed by flattening the `[in_features, out_features]` weight
/// matrix in row-major order and partitioning into chunks of `group_size`.
#[derive(Debug, Clone)]
pub struct QuantizedLinearLayer {
    /// INT4 packed weights (8 values per u32).
    packed_weights: Vec<u32>,
    /// Per-group quantization scales.
    scales: Vec<f32>,
    /// Per-group zero points.
    zeros: Vec<f32>,
    /// Number of input features.
    in_features: usize,
    /// Number of output features.
    out_features: usize,
    /// Number of elements sharing one scale/zero pair.
    group_size: usize,
}

impl QuantizedLinearLayer {
    /// Construct a new quantized linear layer.
    ///
    /// # Arguments
    ///
    /// - `packed_weights`: INT4 packed u32 values (8 INT4 per u32).
    /// - `scales`: per-group scale factors.
    /// - `zeros`: per-group zero points.
    /// - `in_features`: input dimension.
    /// - `out_features`: output dimension.
    /// - `group_size`: elements per quantization group.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizedLayerError`] if dimensions are inconsistent.
    pub fn new(
        packed_weights: Vec<u32>,
        scales: Vec<f32>,
        zeros: Vec<f32>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> std::result::Result<Self, QuantizedLayerError> {
        if group_size == 0 || !group_size.is_power_of_two() {
            return Err(QuantizedLayerError::InvalidGroupSize { group_size });
        }

        let total_elements = in_features * out_features;
        if !total_elements.is_multiple_of(group_size) {
            return Err(QuantizedLayerError::GroupSizeNotDivisible {
                in_features,
                out_features,
                group_size,
            });
        }

        let expected_packed = total_elements.div_ceil(INT4_PER_U32);
        if packed_weights.len() != expected_packed {
            return Err(QuantizedLayerError::PackedWeightLengthMismatch {
                expected: expected_packed,
                actual: packed_weights.len(),
                in_features,
                out_features,
            });
        }

        let num_groups = total_elements / group_size;
        if scales.len() != num_groups {
            return Err(QuantizedLayerError::ScaleZeroLengthMismatch {
                expected: num_groups,
                actual: scales.len(),
            });
        }
        if zeros.len() != num_groups {
            return Err(QuantizedLayerError::ScaleZeroLengthMismatch {
                expected: num_groups,
                actual: zeros.len(),
            });
        }

        Ok(Self {
            packed_weights,
            scales,
            zeros,
            in_features,
            out_features,
            group_size,
        })
    }

    /// Input feature count.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output feature count.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Group size used for quantization.
    #[must_use]
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Dequantize all weights to f32.
    ///
    /// Returns a flat `Vec<f32>` of length `in_features * out_features`
    /// in row-major order (`[in_features, out_features]`).
    ///
    /// Each INT4 value is extracted from the packed u32, then:
    /// ```text
    /// w_f32 = (int4_val as f32 - zeros[group]) * scales[group]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`QuantizedLayerError::NonFiniteValue`] if any dequantized
    /// weight is NaN or infinite.
    pub fn dequantize_weights(&self) -> std::result::Result<Vec<f32>, QuantizedLayerError> {
        let total = self.in_features * self.out_features;
        let mut result = Vec::with_capacity(total);

        for elem_idx in 0..total {
            let int4_val = unpack_int4(&self.packed_weights, elem_idx);
            let group_idx = elem_idx / self.group_size;
            let scale = self.scales[group_idx];
            let zero = self.zeros[group_idx];
            let val = (int4_val as f32 - zero) * scale;

            if !val.is_finite() {
                return Err(QuantizedLayerError::NonFiniteValue { index: elem_idx });
            }

            result.push(val);
        }

        Ok(result)
    }

    /// Forward pass: dequantize then matmul.
    ///
    /// Computes `y = input @ W^T` where `W` is stored as
    /// `[in_features, out_features]` in packed form.
    ///
    /// `y[j] = sum_i input[i] * W[i, j]`
    ///
    /// # Arguments
    ///
    /// - `input`: flat f32 slice of length `in_features` (single vector,
    ///   no batch dimension).
    ///
    /// # Returns
    ///
    /// `Vec<f32>` of length `out_features`.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizedLayerError`] on input length mismatch or
    /// non-finite dequantized weights.
    pub fn forward_quantized_linear(
        &self,
        input: &[f32],
    ) -> std::result::Result<Vec<f32>, QuantizedLayerError> {
        if input.len() != self.in_features {
            return Err(QuantizedLayerError::InputLengthMismatch {
                expected: self.in_features,
                actual: input.len(),
            });
        }

        let weights = self.dequantize_weights()?;
        // weights is [in_features, out_features] row-major.
        // y[j] = sum_i input[i] * weights[i * out_features + j]
        let mut output = vec![0.0_f32; self.out_features];

        for (i, &x_i) in input.iter().enumerate().take(self.in_features) {
            let row_offset = i * self.out_features;
            for j in 0..self.out_features {
                output[j] += x_i * weights[row_offset + j];
            }
        }

        Ok(output)
    }
}

/// Unpack a single INT4 value from packed u32 storage.
///
/// Each u32 holds 8 nibbles in little-endian order:
/// bits [3:0] = index 0, bits [7:4] = index 1, ..., bits [31:28] = index 7.
///
/// Returns unsigned value in `[0, 15]`.
fn unpack_int4(packed: &[u32], element_index: usize) -> u32 {
    let word_idx = element_index / INT4_PER_U32;
    let nibble_idx = element_index % INT4_PER_U32;
    let shift = nibble_idx as u32 * 4;
    (packed[word_idx] >> shift) & INT4_MASK
}

// ---------------------------------------------------------------------------
// Standalone memory estimation
// ---------------------------------------------------------------------------

/// Estimate total memory in bytes for a quantized Qwen3-VL model.
///
/// Delegates to [`Qwen3VLQuantConfig::estimated_memory_bytes`]. This
/// standalone function is convenient when you have a config reference
/// but don't need to call through the method.
#[must_use]
pub fn estimate_memory_bytes(config: &Qwen3VLQuantConfig) -> usize {
    config.estimated_memory_bytes()
}

#[cfg(test)]
#[path = "qwen3_vl_quantized_tests.rs"]
mod tests;
