// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Query methods for [`DispatchStep`].
//!
//! Extracted from `dispatch_step.rs` for 450-line compliance (#3184).

use crate::tensor_ir::TensorNodeId;

use super::DispatchStep;

/// Tile size for shared-memory transpose (16×16 is standard for GPU).
pub const TILED_TRANSPOSE_TILE_SIZE: usize = 16;

impl DispatchStep {
    /// Returns `Some((batch, rows, cols))` if this is a Transpose that can use
    /// the tiled shared-memory implementation (2-5× faster than naive).
    ///
    /// Qualifies when all leading axes are identity and only the last two are
    /// swapped (e.g., axes `[0, 2, 1]` for rank 3). Both `rows` and `cols`
    /// must be >= `TILED_TRANSPOSE_TILE_SIZE`.
    pub fn tiled_transpose_params(&self) -> Option<(usize, usize, usize)> {
        match self {
            Self::Transpose {
                input_shape, axes, ..
            } => tiled_transpose_2d_params(input_shape, axes),
            _ => None,
        }
    }

    /// Returns `true` if this step reads from the given tensor node ID.
    pub fn uses_input(&self, id: TensorNodeId) -> bool {
        match self {
            Self::Reduce { input, .. }
            | Self::Sigmoid { input, .. }
            | Self::Gelu { input, .. }
            | Self::GeluErf { input, .. }
            | Self::Relu { input, .. }
            | Self::Tanh { input, .. }
            | Self::LeakyRelu { input, .. }
            | Self::Elu { input, .. }
            | Self::Exp { input, .. }
            | Self::Softplus { input, .. }
            | Self::Reshape { input, .. }
            | Self::Narrow { input, .. }
            | Self::AxisSelect { input, .. }
            | Self::Softmax { input, .. }
            | Self::ZeroPad1d { input, .. }
            | Self::Transpose { input, .. } => *input == id,
            Self::Broadcast { input, .. } => *input == id,
            Self::BinaryAdd { left, right, .. } | Self::BinaryMul { left, right, .. } => {
                *left == id || *right == id
            }
            Self::MatMul { left, right, .. } => *left == id || *right == id,
            Self::Linear {
                input,
                weight,
                bias,
                ..
            } => *input == id || *weight == id || *bias == Some(id),
            Self::Embedding { input, weight, .. } => *input == id || *weight == id,
            Self::IndexSelect { input, indices, .. } | Self::Gather { input, indices, .. } => {
                *input == id || *indices == id
            }
            Self::Elementwise { inputs, .. }
            | Self::Stack { inputs, .. }
            | Self::Concat { inputs, .. } => inputs.contains(&id),
            Self::Conv1d(p) => p.input == id || p.weight == id || p.bias == Some(id),
            Self::Conv2d(p) => p.input == id || p.weight == id || p.bias == Some(id),
            Self::ConvTranspose1d(p) => p.input == id || p.weight == id || p.bias == Some(id),
            Self::SimdgroupLinear(p) => p.input == id || p.weight == id || p.bias == Some(id),
            Self::SimdgroupMatMul(p) => p.left == id || p.right == id,
            Self::TiledLinear(p) => p.input == id || p.weight == id || p.bias == Some(id),
            Self::TiledMatMul(p) => p.left == id || p.right == id,
        }
    }
}

/// Check if a transpose can use the tiled shared-memory 2D implementation.
///
/// Returns `Some((batch, rows, cols))` when all leading axes are identity
/// and only the last two are swapped. Both rows and cols must be >= tile size.
pub fn tiled_transpose_2d_params(
    input_shape: &[usize],
    axes: &[usize],
) -> Option<(usize, usize, usize)> {
    let rank = input_shape.len();
    if rank < 2 {
        return None;
    }
    // All axes except last two must be identity.
    for (i, &ax) in axes[..rank - 2].iter().enumerate() {
        if ax != i {
            return None;
        }
    }
    // Last two must be swapped.
    if axes[rank - 2] != rank - 1 || axes[rank - 1] != rank - 2 {
        return None;
    }
    let rows = input_shape[rank - 2];
    let cols = input_shape[rank - 1];
    if rows < TILED_TRANSPOSE_TILE_SIZE || cols < TILED_TRANSPOSE_TILE_SIZE {
        return None;
    }
    let batch: usize = input_shape[..rank - 2].iter().product();
    Some((batch.max(1), rows, cols))
}
