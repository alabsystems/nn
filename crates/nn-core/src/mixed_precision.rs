// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Graph-level mixed-precision policy for weight, compute, and accumulate dtypes.
//!
//! Models declare a [`MixedPrecisionPolicy`] at construction time. The policy
//! flows through [`VarBuilder`](crate::VarBuilder) (weight loading),
//! [`DynTensor`](crate::DynTensor) ops (compute), and NY
//! (verification bounds).
//!
//! Op classification follows PyTorch's autocast allowlist/denylist, with static
//! resolution matching ONNX's graph-level approach. This is necessary because
//! nn uses static graphs for NY verification — runtime dtype decisions
//! would break bounds propagation.
//!
//! # Example
//!
//! ```
//! use nn_core::mixed_precision::{MixedPrecisionPolicy, OpDTypeCategory, default_op_category};
//! use nn_core::DType;
//!
//! let policy = MixedPrecisionPolicy::apple_silicon_default();
//! assert_eq!(policy.compute_dtype, DType::F16);
//! assert_eq!(policy.accumulate_dtype, DType::F32);
//!
//! // MatMul uses compute dtype (F16 on Apple Silicon)
//! let dt = policy.dtype_for_op(OpDTypeCategory::Compute);
//! assert_eq!(dt, DType::F16);
//!
//! // Softmax uses accumulate dtype (always F32)
//! assert_eq!(default_op_category("softmax"), OpDTypeCategory::Accumulate);
//! ```

use crate::DType;

/// Mixed-precision policy declaring weight, compute, and accumulate dtypes.
///
/// Models declare a policy at construction time. The policy flows through
/// VarBuilder (weight loading), DynTensor ops (compute), and NY
/// (verification bounds).
///
/// # Presets
///
/// - [`f32_only()`](Self::f32_only) — safe default, no precision risk
/// - [`apple_silicon_default()`](Self::apple_silicon_default) — BF16 weights,
///   F16 compute, F32 accumulate (standard dvoice config)
/// - [`cuda_bf16()`](Self::cuda_bf16) — BF16 weights and compute on sm_80+
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MixedPrecisionPolicy {
    /// Dtype for weight storage (what safetensors contains).
    /// Weights are loaded in this dtype, then cast to `compute_dtype` for ops.
    pub weight_dtype: DType,

    /// Dtype for FLOPS-dominated ops (matmul, conv, linear, embedding lookup).
    /// On Apple Silicon: F16 (hardware auto-converts BF16->F16).
    /// On CUDA sm_80+: BF16 natively. On older CUDA: F32.
    pub compute_dtype: DType,

    /// Dtype for numerically sensitive ops (softmax, norms, reductions).
    /// Always F32 for correctness. Downgrading this requires NY
    /// proof that bounds are preserved.
    pub accumulate_dtype: DType,
}

impl MixedPrecisionPolicy {
    /// F32 everywhere — safe default, no precision risk.
    #[must_use]
    pub fn f32_only() -> Self {
        Self {
            weight_dtype: DType::F32,
            compute_dtype: DType::F32,
            accumulate_dtype: DType::F32,
        }
    }

    /// BF16 weights, F16 compute on Apple Silicon, F32 accumulate.
    /// The standard dvoice configuration.
    #[must_use]
    pub fn apple_silicon_default() -> Self {
        Self {
            weight_dtype: DType::BF16,
            compute_dtype: DType::F16,
            accumulate_dtype: DType::F32,
        }
    }

    /// BF16 weights, BF16 compute on CUDA sm_80+, F32 accumulate.
    #[must_use]
    pub fn cuda_bf16() -> Self {
        Self {
            weight_dtype: DType::BF16,
            compute_dtype: DType::BF16,
            accumulate_dtype: DType::F32,
        }
    }

    /// Resolve the compute dtype for a given op category.
    #[must_use]
    pub fn dtype_for_op(&self, category: OpDTypeCategory) -> DType {
        match category {
            OpDTypeCategory::Compute => self.compute_dtype,
            OpDTypeCategory::Accumulate => self.accumulate_dtype,
            OpDTypeCategory::Inherit => self.compute_dtype,
        }
    }
}

impl Default for MixedPrecisionPolicy {
    fn default() -> Self {
        Self::f32_only()
    }
}

/// Classification of ops by their numerical sensitivity.
///
/// Mirrors PyTorch's autocast op lists. The classification determines which
/// dtype from [`MixedPrecisionPolicy`] is used for each op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpDTypeCategory {
    /// FLOPS-dominated ops: safe in reduced precision.
    /// MatMul, Conv1d, ConvTranspose1d, Linear, Embedding lookup.
    /// Uses [`MixedPrecisionPolicy::compute_dtype`].
    Compute,

    /// Numerically sensitive ops: require full precision.
    /// Softmax, LayerNorm, GroupNorm, InstanceNorm, RmsNorm, BatchNorm,
    /// log, pow, reductions (sum, mean).
    /// Uses [`MixedPrecisionPolicy::accumulate_dtype`].
    Accumulate,

    /// Element-wise ops: inherit dtype from input.
    /// ReLU, GELU, SiLU, Snake, Tanh, Sigmoid, element-wise add/mul.
    /// Uses whatever dtype the input tensor has.
    Inherit,
}

/// Get the default dtype category for a DynTensor op.
///
/// This classification is based on PyTorch's autocast allowlist/denylist,
/// validated for correctness on production models (Qwen3, Whisper, Demucs).
///
/// Unknown ops default to [`OpDTypeCategory::Inherit`] (element-wise passthrough),
/// which is safe — the op runs in whatever dtype the input already has.
#[must_use]
pub fn default_op_category(op_name: &str) -> OpDTypeCategory {
    match op_name {
        // Compute (safe in reduced precision — FLOPS-dominated)
        "matmul"
        | "conv1d"
        | "conv2d"
        | "conv_transpose1d"
        | "conv_transpose2d"
        | "linear"
        | "embedding"
        | "lstm_gates"
        | "attention"
        | "flash_attention"
        | "norm_activ_conv1d"
        | "fused_res_block"
        | "norm_linear"
        | "batched_linear_projection"
        | "adain_snake"
        | "adain_leaky_relu" => OpDTypeCategory::Compute,

        // Accumulate (require full precision — numerically sensitive)
        "softmax" | "log_softmax" | "layer_norm" | "group_norm" | "instance_norm" | "rms_norm"
        | "batch_norm" | "sum" | "mean" | "log" | "pow" | "cumsum" => OpDTypeCategory::Accumulate,

        // Inherit (element-wise, dtype-transparent)
        _ => OpDTypeCategory::Inherit,
    }
}

#[cfg(test)]
#[path = "mixed_precision_tests.rs"]
mod tests;
