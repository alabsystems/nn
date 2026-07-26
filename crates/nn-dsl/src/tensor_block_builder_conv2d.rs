// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv2d builder methods for `TensorBlockBuilder`.
//!
//! Extracted from `tensor_block_builder_ops.rs` to keep files under 500 lines.

use super::*;

impl TensorBlockBuilder {
    /// Add a Conv2d op with dilation=1, groups=1. Returns output node ID.
    ///
    /// Input shape: `[C_in, H, W]`. Weight shape: `[C_out, C_in, kH, kW]`.
    /// Bias shape: `[C_out]` (optional). Output: `[C_out, out_h, out_w]`.
    pub fn add_conv2d(
        &mut self,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        self.add_conv2d_full(
            input, weight, bias, stride_h, stride_w, padding_h, padding_w, 1, 1, 1, out_shape,
        )
    }

    /// Add a Conv2d op with explicit dilation and groups. Returns output node ID.
    ///
    /// Input shape: `[C_in, H, W]`. Weight shape: `[C_out, C_in/groups, kH, kW]`.
    /// Bias shape: `[C_out]` (optional). Output: `[C_out, out_h, out_w]`.
    pub fn add_conv2d_full(
        &mut self,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        groups: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Conv2d {
                input,
                weight,
                bias,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                groups,
            },
            out_shape.to_vec(),
        ));
        id
    }
}
