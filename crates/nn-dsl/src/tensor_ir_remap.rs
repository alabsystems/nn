// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `TensorOpKind::remap_ids` — remap all node-id references in a tensor op.

use super::{TensorNodeId, TensorOpKind};
use std::collections::HashMap;

impl TensorOpKind {
    /// Remap all `TensorNodeId` references using the given mapping.
    ///
    /// Returns a new `TensorOpKind` with every node-id reference replaced via
    /// `id_map`. Panics if a referenced id is missing from the map.
    ///
    /// Used by norm expansion (`codegen_msl_tensor_expand`) to rewrite node
    /// references after inserting decomposed ops.
    #[must_use]
    pub(crate) fn remap_ids(&self, id_map: &HashMap<usize, usize>) -> Self {
        let remap = |id: &TensorNodeId| TensorNodeId::new(id_map[&id.index()]);

        match self {
            Self::Input { name, shape } => Self::Input {
                name: name.clone(),
                shape: shape.clone(),
            },
            Self::Reshape {
                input,
                target_shape,
            } => Self::Reshape {
                input: remap(input),
                target_shape: target_shape.clone(),
            },
            Self::AxisSelect { input, axis, index } => Self::AxisSelect {
                input: remap(input),
                axis: *axis,
                index: *index,
            },
            Self::Stack { inputs, axis } => Self::Stack {
                inputs: inputs.iter().map(&remap).collect(),
                axis: *axis,
            },
            Self::Concat { inputs, axis } => Self::Concat {
                inputs: inputs.iter().map(&remap).collect(),
                axis: *axis,
            },
            Self::Reduce {
                op,
                input,
                axis,
                keepdim,
            } => Self::Reduce {
                op: *op,
                input: remap(input),
                axis: *axis,
                keepdim: *keepdim,
            },
            Self::Elementwise { kernel, inputs } => Self::Elementwise {
                kernel: kernel.clone(),
                inputs: inputs.iter().map(&remap).collect(),
            },
            Self::Broadcast {
                input,
                target_shape,
                alignment,
            } => Self::Broadcast {
                input: remap(input),
                target_shape: target_shape.clone(),
                alignment: *alignment,
            },
            Self::Conv1d {
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
            } => Self::Conv1d {
                input: remap(input),
                weight: remap(weight),
                bias: bias.as_ref().map(&remap),
                stride: *stride,
                padding: *padding,
                dilation: *dilation,
                groups: *groups,
            },
            Self::Conv2d {
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
            } => Self::Conv2d {
                input: remap(input),
                weight: remap(weight),
                bias: bias.as_ref().map(&remap),
                stride_h: *stride_h,
                stride_w: *stride_w,
                padding_h: *padding_h,
                padding_w: *padding_w,
                dilation_h: *dilation_h,
                dilation_w: *dilation_w,
                groups: *groups,
            },
            Self::ConvTranspose1d {
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
                output_padding,
            } => Self::ConvTranspose1d {
                input: remap(input),
                weight: remap(weight),
                bias: bias.as_ref().map(&remap),
                stride: *stride,
                padding: *padding,
                dilation: *dilation,
                groups: *groups,
                output_padding: *output_padding,
            },
            Self::ConvTranspose2d {
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
            } => Self::ConvTranspose2d {
                input: remap(input),
                weight: remap(weight),
                bias: bias.as_ref().map(&remap),
                stride_h: *stride_h,
                stride_w: *stride_w,
                padding_h: *padding_h,
                padding_w: *padding_w,
                dilation_h: *dilation_h,
                dilation_w: *dilation_w,
                groups: *groups,
                output_padding_h: *output_padding_h,
                output_padding_w: *output_padding_w,
            },
            Self::BinaryAdd { left, right } => Self::BinaryAdd {
                left: remap(left),
                right: remap(right),
            },
            Self::BinaryMul { left, right } => Self::BinaryMul {
                left: remap(left),
                right: remap(right),
            },
            Self::Softmax { input, axis } => Self::Softmax {
                input: remap(input),
                axis: *axis,
            },
            Self::LogSoftmax { input, axis } => Self::LogSoftmax {
                input: remap(input),
                axis: *axis,
            },
            Self::Sigmoid { input } => Self::Sigmoid {
                input: remap(input),
            },
            Self::Silu { input } => Self::Silu {
                input: remap(input),
            },
            Self::Gelu { input } => Self::Gelu {
                input: remap(input),
            },
            Self::GeluErf { input } => Self::GeluErf {
                input: remap(input),
            },
            Self::Relu { input } => Self::Relu {
                input: remap(input),
            },
            Self::LeakyRelu {
                input,
                negative_slope,
            } => Self::LeakyRelu {
                input: remap(input),
                negative_slope: *negative_slope,
            },
            Self::Elu { input, alpha } => Self::Elu {
                input: remap(input),
                alpha: *alpha,
            },
            Self::Tanh { input } => Self::Tanh {
                input: remap(input),
            },
            Self::Softplus { input } => Self::Softplus {
                input: remap(input),
            },
            Self::Exp { input } => Self::Exp {
                input: remap(input),
            },
            Self::Narrow {
                input,
                axis,
                start,
                length,
            } => Self::Narrow {
                input: remap(input),
                axis: *axis,
                start: *start,
                length: *length,
            },
            Self::Linear {
                input,
                weight,
                bias,
            } => Self::Linear {
                input: remap(input),
                weight: remap(weight),
                bias: bias.as_ref().map(&remap),
            },
            Self::MatMul {
                left,
                right,
                transpose_right,
                scale,
            } => Self::MatMul {
                left: remap(left),
                right: remap(right),
                transpose_right: *transpose_right,
                scale: *scale,
            },
            Self::InstanceNorm1d {
                input,
                eps,
                axis,
                gamma,
                beta,
            } => Self::InstanceNorm1d {
                input: remap(input),
                eps: remap(eps),
                axis: *axis,
                gamma: gamma.as_ref().map(&remap),
                beta: beta.as_ref().map(&remap),
            },
            Self::RmsNorm {
                input,
                eps,
                axis,
                weight,
            } => Self::RmsNorm {
                input: remap(input),
                eps: remap(eps),
                axis: *axis,
                weight: remap(weight),
            },
            Self::AdaIN1d {
                input,
                eps,
                axis,
                style_gamma,
                style_beta,
            } => Self::AdaIN1d {
                input: remap(input),
                eps: remap(eps),
                axis: *axis,
                style_gamma: remap(style_gamma),
                style_beta: remap(style_beta),
            },
            Self::ZeroPad1d {
                input,
                pad_left,
                pad_right,
            } => Self::ZeroPad1d {
                input: remap(input),
                pad_left: *pad_left,
                pad_right: *pad_right,
            },
            Self::Embedding { input, weight } => Self::Embedding {
                input: remap(input),
                weight: remap(weight),
            },
            Self::LayerNorm {
                input,
                eps,
                axis,
                weight,
                bias,
            } => Self::LayerNorm {
                input: remap(input),
                eps: remap(eps),
                axis: *axis,
                weight: remap(weight),
                bias: remap(bias),
            },
            Self::Attention {
                q,
                k,
                v,
                mask,
                scale,
            } => Self::Attention {
                q: remap(q),
                k: remap(k),
                v: remap(v),
                mask: *mask,
                scale: *scale,
            },
            Self::Transpose { input, axes } => Self::Transpose {
                input: remap(input),
                axes: axes.clone(),
            },
            Self::Lstm {
                input,
                hidden_state,
                cell_state,
                weight_ih,
                weight_hh,
                bias,
            } => Self::Lstm {
                input: remap(input),
                hidden_state: remap(hidden_state),
                cell_state: remap(cell_state),
                weight_ih: remap(weight_ih),
                weight_hh: remap(weight_hh),
                bias: bias.as_ref().map(&remap),
            },
            Self::GatedDeltaNet {
                q,
                k,
                v,
                state,
                gate,
                beta,
                scale,
            } => Self::GatedDeltaNet {
                q: remap(q),
                k: remap(k),
                v: remap(v),
                state: remap(state),
                gate: remap(gate),
                beta: remap(beta),
                scale: *scale,
            },
            Self::AvgPool2d { input, params } => Self::AvgPool2d {
                input: remap(input),
                params: *params,
            },
            Self::MaxPool2d { input, params } => Self::MaxPool2d {
                input: remap(input),
                params: *params,
            },
            Self::BatchNorm {
                input,
                running_mean,
                running_var,
                weight,
                bias,
                eps,
            } => Self::BatchNorm {
                input: remap(input),
                running_mean: remap(running_mean),
                running_var: remap(running_var),
                weight: remap(weight),
                bias: remap(bias),
                eps: remap(eps),
            },
            Self::IndexSelect {
                input,
                indices,
                dim,
            } => Self::IndexSelect {
                input: remap(input),
                indices: remap(indices),
                dim: *dim,
            },
            Self::Gather {
                input,
                indices,
                dim,
            } => Self::Gather {
                input: remap(input),
                indices: remap(indices),
                dim: *dim,
            },
        }
    }
}
