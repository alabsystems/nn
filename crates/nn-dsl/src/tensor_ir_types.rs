// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Supporting types for the tensor IR — node IDs, broadcast alignment, reductions.
//!
//! Extracted from `tensor_ir_ops.rs` to keep that file focused on `TensorOpKind`
//! as the number of operation variants grows. All types are re-exported from
//! `tensor_ir`, so downstream callers are unchanged.

/// Unique identifier for a node in the tensor-level IR graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TensorNodeId(usize);

impl TensorNodeId {
    /// Create a new `TensorNodeId` from a raw index.
    pub fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// Get the underlying index.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Broadcast alignment strategy.
///
/// When broadcasting a lower-rank tensor to a higher-rank target, alignment
/// determines which dimensions of the target correspond to the input dims.
///
/// For shapes where both left and right alignment are valid (e.g., `[2] -> [2, 2]`),
/// the alignment must be specified explicitly — the IR rejects ambiguous cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum BroadcastAlignment {
    /// Input dims align to the prefix (left) of the target shape.
    ///
    /// Example: `[B, C]` broadcast to `[B, C, T]` — reduce→broadcast pattern.
    Left,
    /// Input dims align to the suffix (right) of the target shape.
    ///
    /// Example: `[D]` broadcast to `[B, T, D]` — NumPy-style.
    Right,
}

/// Reduction operations supported by the tensor IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ReduceOp {
    /// Sum reduction: output[i] = sum(input[i, :]) along the reduction axis.
    Sum,
    /// Mean reduction: output[i] = mean(input[i, :]) along the reduction axis.
    Mean,
    /// Max reduction: output[i] = max(input[i, :]) along the reduction axis.
    Max,
    /// Min reduction: output[i] = min(input[i, :]) along the reduction axis.
    Min,
}

/// Shared 2D pooling parameters for `AvgPool2d` and `MaxPool2d`.
///
/// Both pooling variants use identical spatial parameters. This struct
/// eliminates the duplicated 6-field parameter list.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Pool2dParams {
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Stride height.
    pub stride_h: usize,
    /// Stride width.
    pub stride_w: usize,
    /// Zero-padding height (applied to both sides).
    pub padding_h: usize,
    /// Zero-padding width (applied to both sides).
    pub padding_w: usize,
}

impl Pool2dParams {
    pub fn new(
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Self {
        Self {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        }
    }
}

/// Attention masking mode for the `Attention` tensor op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum AttentionMask {
    /// Standard (bidirectional) attention — no mask applied.
    Standard,
    /// Causal (autoregressive) attention — position j attends only to positions <= j.
    Causal,
}
