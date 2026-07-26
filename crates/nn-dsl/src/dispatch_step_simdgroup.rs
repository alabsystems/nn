// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Simdgroup-tiled GEMM dispatch step parameter types.
//!
//! When matmul/linear shapes conform to simdgroup requirements (all dims % 8,
//! M×N ≥ 16384, K ≥ 128), the compiled pipeline uses Apple Silicon's
//! `simdgroup_matrix<T, 8, 8>` hardware cooperative multiply-accumulate
//! instead of the naive per-element dot product.
//!
//! Part of #2275.

use crate::ir::ScalarType;
use crate::tensor_ir::TensorNodeId;

/// Parameters for a simdgroup-tiled linear layer dispatch step.
///
/// Weight is stored as `[out_features, in_features]` (row-major), read
/// transposed by the simdgroup kernel. Bias is added in the output write
/// phase when present.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimdgroupLinearParams {
    /// Name of the generated MSL kernel function.
    pub kernel_name: String,
    /// Scalar type (f32 or f16).
    pub dtype: ScalarType,
    /// Input data tensor node (shape `[batch_size, in_features]`).
    pub input: TensorNodeId,
    /// Weight tensor node (shape `[out_features, in_features]`).
    pub weight: TensorNodeId,
    /// Optional bias vector (shape `[out_features]`).
    pub bias: Option<TensorNodeId>,
    /// Output tensor node (shape `[batch_size, out_features]`).
    pub output: TensorNodeId,
    /// Contracted dimension (in_features = K).
    pub in_features: usize,
    /// Output feature dimension (out_features = N).
    pub out_features: usize,
    /// Product of all leading (batch) dimensions (= M).
    pub batch_size: usize,
}

/// Parameters for a simdgroup-tiled matrix multiplication dispatch step.
///
/// Uses `simdgroup_matrix<T, 8, 8>` with 32×32 output tiles, 128 threads
/// per threadgroup (4 simdgroups of 32). Supports transposed right operand,
/// broadcast right, and optional post-multiply scaling.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimdgroupMatMulParams {
    /// Name of the generated MSL kernel function.
    pub kernel_name: String,
    /// Scalar type (f32 or f16).
    pub dtype: ScalarType,
    /// Left input tensor node (shape `[*, M, K]`).
    pub left: TensorNodeId,
    /// Right input tensor node (shape `[*, K, N]` or `[*, N, K]`).
    pub right: TensorNodeId,
    /// Output tensor node (shape `[*, M, N]`).
    pub output: TensorNodeId,
    /// Rows in left matrix (M).
    pub m: usize,
    /// Contracted dimension (K).
    pub k: usize,
    /// Columns in output (N).
    pub n: usize,
    /// Product of leading batch dimensions.
    pub batch_size: usize,
    /// Whether right is transposed before multiplication.
    pub transpose_right: bool,
    /// Whether right has fewer batch dims (broadcast across batches).
    pub broadcast_right: bool,
    /// Optional scaling factor applied post-multiply.
    pub scale: Option<f32>,
}
