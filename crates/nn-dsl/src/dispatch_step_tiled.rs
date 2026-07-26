// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tiled GEMM dispatch step parameter types.
//!
//! Middle-tier GEMM using threadgroup (shared) memory tiling. Targets shapes
//! that are too small for simdgroup hardware matrix units (M*N < 16384 or
//! K < 128) but large enough to fill at least one 16×16 tile (M >= 16,
//! N >= 16, K >= 8).
//!
//! Part of #3230 (Gap 1).

use crate::ir::ScalarType;
use crate::tensor_ir::TensorNodeId;

/// Tile size for tiled GEMM threadgroup memory.
pub const TILED_GEMM_TILE: usize = 16;

/// Parameters for a tiled linear layer dispatch step.
///
/// Weight is stored as `[out_features, in_features]` (row-major).
/// Each threadgroup computes one TILE_M × TILE_N output block.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TiledLinearParams {
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

/// Parameters for a tiled matrix multiplication dispatch step.
///
/// Uses threadgroup memory tiling with 16×16 tiles. Supports transposed
/// right operand, broadcast right, and optional post-multiply scaling.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TiledMatMulParams {
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
