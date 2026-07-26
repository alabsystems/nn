// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convolution and linear builder methods for `TensorBlockBuilder`.
//!
//! Extracted from `tensor_block_builder.rs` to stay under the 500-line limit
//! (Part of #1575).

use crate::tensor_ir::{Pool2dParams, TensorNode, TensorNodeId, TensorOpKind};

use super::TensorBlockBuilder;

impl TensorBlockBuilder {
    /// Add a Conv1d op with dilation=1, groups=1. Returns output node ID.
    pub fn add_conv1d(
        &mut self,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride: usize,
        padding: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        self.add_conv1d_full(input, weight, bias, stride, padding, 1, 1, out_shape)
    }

    /// Add a Conv1d op with explicit dilation and groups. Returns output node ID.
    pub fn add_conv1d_full(
        &mut self,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Conv1d {
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a ConvTranspose1d (transposed convolution / upsampling) op. Returns output node ID.
    #[allow(clippy::too_many_arguments)]
    pub fn add_conv_transpose_1d(
        &mut self,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        output_padding: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::ConvTranspose1d {
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
                output_padding,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a ConvTranspose2d (2D transposed convolution / upsampling) op. Returns output node ID.
    #[allow(clippy::too_many_arguments)]
    pub fn add_conv_transpose_2d(
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
        output_padding_h: usize,
        output_padding_w: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::ConvTranspose2d {
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
                output_padding_h,
                output_padding_w,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add an AvgPool2d op. Returns output node ID.
    #[allow(clippy::too_many_arguments)]
    pub fn add_avg_pool_2d(
        &mut self,
        input: TensorNodeId,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::AvgPool2d {
                input,
                params: Pool2dParams::new(
                    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
                ),
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a MaxPool2d op. Returns output node ID.
    #[allow(clippy::too_many_arguments)]
    pub fn add_max_pool_2d(
        &mut self,
        input: TensorNodeId,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::MaxPool2d {
                input,
                params: Pool2dParams::new(
                    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
                ),
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a Linear (fully-connected) layer. Returns output node ID.
    ///
    /// Input shape: `[*, in_features]`. Weight shape: `[out_features, in_features]`.
    /// Bias shape: `[out_features]` (optional). Output shape: `[*, out_features]`.
    pub fn add_linear(
        &mut self,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Linear {
                input,
                weight,
                bias,
            },
            out_shape.to_vec(),
        ));
        id
    }
}
