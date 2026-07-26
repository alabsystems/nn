// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dispatch function for tensor-level IR → NY translation.
//!
//! Extracted from `graph_tensor.rs` to stay within the 500-line file limit.
//! Each `TensorOpKind` variant dispatches to a per-op submodule.

use ny_propagate::GraphNetwork;
use nn_dsl::tensor_ir::{TensorNodeId, TensorOpKind};

use super::{
    attention::translate_attention,
    binary::{translate_binary_add, translate_binary_mul},
    elementwise::translate_elementwise_inline,
    embedding::translate_embedding,
    exp::translate_exp,
    gated_delta_net::translate_gated_delta_net,
    gelu::{translate_gelu, translate_gelu_erf},
    leaky_relu::translate_leaky_relu,
    linear::translate_linear,
    lstm::translate_lstm,
    matmul::translate_matmul,
    reduce::translate_reduce_op,
    relu::translate_relu,
    sigmoid::translate_sigmoid,
    silu::translate_silu,
    softmax::translate_softmax,
    softplus::translate_softplus,
    structural::{
        translate_axis_select, translate_concat, translate_narrow, translate_reshape,
        translate_stack,
    },
    tanh::translate_tanh,
    transpose::translate_transpose,
    zero_pad::translate_zero_pad_1d,
    TensorNodeValue, TensorTranslationContext,
};
use crate::error::VerifyError;

#[path = "graph_tensor_helpers.rs"]
mod helpers;
use helpers::{translate_broadcast, translate_input};

#[path = "graph_tensor_dispatch_conv.rs"]
mod conv_norm;

#[path = "graph_tensor_pool.rs"]
mod pool;

pub(crate) fn translate_tensor_node(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    kind: &TensorOpKind,
    node_values: &[TensorNodeValue],
    input_idx: &mut usize,
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    match kind {
        TensorOpKind::Input { .. } => translate_input(ctx, node_values, input_idx),

        TensorOpKind::Reduce {
            op,
            input,
            axis,
            keepdim,
        } => translate_reduce_op(ctx, node_id, op, input, *axis, *keepdim, node_values, graph),

        TensorOpKind::Elementwise {
            kernel: scalar_kernel,
            inputs,
        } => translate_elementwise_inline(ctx, node_id, scalar_kernel, inputs, node_values, graph),

        TensorOpKind::Broadcast {
            input,
            target_shape,
            alignment,
        } => translate_broadcast(
            node_id,
            input,
            target_shape,
            *alignment,
            node_values,
            ctx,
            graph,
        ),

        // Conv-like ops and normalization: delegated to graph_tensor_dispatch_conv.rs.
        // Conv2d is split out below so grouped (groups>1) convs route to the
        // dedicated grouped translator instead of the single-group path.
        TensorOpKind::Conv1d { .. }
        | TensorOpKind::ConvTranspose1d { .. }
        | TensorOpKind::RmsNorm { .. }
        | TensorOpKind::InstanceNorm1d { .. }
        | TensorOpKind::AdaIN1d { .. }
        | TensorOpKind::LayerNorm { .. }
        | TensorOpKind::BatchNorm { .. } => {
            conv_norm::translate_conv_or_norm(ctx, node_id, kind, node_values, graph)
                .ok_or_else(|| VerifyError::UnsupportedOp(format!("{kind:?}")))?
        }

        // Conv2d: groups==1 keeps the existing single-group translator (incl. its
        // dilated-kernel expansion); groups>1 routes to the native grouped path.
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
        } => {
            if *groups <= 1 {
                conv_norm::translate_conv_or_norm(ctx, node_id, kind, node_values, graph)
                    .ok_or_else(|| VerifyError::UnsupportedOp(format!("{kind:?}")))?
            } else {
                pool::translate_conv2d_grouped(
                    ctx,
                    node_id,
                    input,
                    weight,
                    bias,
                    *stride_h,
                    *stride_w,
                    *padding_h,
                    *padding_w,
                    *dilation_h,
                    *dilation_w,
                    *groups,
                    node_values,
                    graph,
                )
            }
        }

        // 2D pooling, LogSoftmax, and ConvTranspose2d: NY has confirmed-sound
        // IBP+CROWN layers for all of these (graph_tensor_pool.rs).
        TensorOpKind::AvgPool2d { input, params } => {
            pool::translate_avg_pool2d(node_id, input, params, node_values, graph)
        }

        TensorOpKind::MaxPool2d { input, params } => {
            pool::translate_max_pool2d(node_id, input, params, node_values, graph)
        }

        TensorOpKind::LogSoftmax { input, axis } => {
            pool::translate_log_softmax(node_id, input, *axis, node_values, graph)
        }

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
        } => pool::translate_conv_transpose_2d(
            ctx,
            node_id,
            input,
            weight,
            bias,
            *stride_h,
            *stride_w,
            *padding_h,
            *padding_w,
            *dilation_h,
            *dilation_w,
            *groups,
            *output_padding_h,
            *output_padding_w,
            node_values,
            graph,
        ),

        TensorOpKind::Reshape {
            input,
            target_shape,
        } => translate_reshape(ctx, node_id, input, target_shape, node_values, graph),

        TensorOpKind::AxisSelect { input, axis, index } => {
            translate_axis_select(ctx, node_id, input, *axis, *index, node_values, graph)
        }

        TensorOpKind::Stack { inputs, axis } => {
            translate_stack(ctx, node_id, inputs, *axis, node_values, graph)
        }

        TensorOpKind::Concat { inputs, axis } => {
            translate_concat(ctx, node_id, inputs, *axis, node_values, graph)
        }

        TensorOpKind::Transpose { input, axes } => {
            translate_transpose(ctx, node_id, input, axes, node_values, graph)
        }

        TensorOpKind::Narrow {
            input,
            axis,
            start,
            length,
        } => translate_narrow(
            ctx,
            node_id,
            input,
            *axis,
            *start,
            *length,
            node_values,
            graph,
        ),

        TensorOpKind::Sigmoid { input } => translate_sigmoid(node_id, input, node_values, graph),

        TensorOpKind::Silu { input } => translate_silu(node_id, input, node_values, graph),

        TensorOpKind::Gelu { input } => translate_gelu(node_id, input, node_values, graph),

        TensorOpKind::GeluErf { input } => translate_gelu_erf(node_id, input, node_values, graph),

        TensorOpKind::Relu { input } => translate_relu(node_id, input, node_values, graph),

        TensorOpKind::LeakyRelu {
            input,
            negative_slope,
        } => translate_leaky_relu(node_id, input, *negative_slope, node_values, graph),

        TensorOpKind::Tanh { input } => translate_tanh(node_id, input, node_values, graph),

        TensorOpKind::Softplus { input } => translate_softplus(node_id, input, node_values, graph),

        TensorOpKind::Exp { input } => translate_exp(node_id, input, node_values, graph),

        TensorOpKind::BinaryAdd { left, right } => {
            translate_binary_add(node_id, *left, *right, node_values, graph)
        }

        TensorOpKind::BinaryMul { left, right } => {
            translate_binary_mul(node_id, *left, *right, node_values, graph)
        }

        TensorOpKind::Linear {
            input,
            weight,
            bias,
        } => translate_linear(node_id, *input, *weight, bias.as_ref(), node_values, graph),

        TensorOpKind::Softmax { input, axis } => {
            translate_softmax(ctx, node_id, input, *axis, node_values, graph)
        }

        TensorOpKind::ZeroPad1d {
            input,
            pad_left,
            pad_right,
        } => {
            let output_shape = ctx
                .all_nodes
                .get(node_id.index())
                .map(|n| n.shape.as_slice())
                .ok_or_else(|| {
                    VerifyError::UnsupportedOp(format!(
                        "ZeroPad1d node {} out of bounds",
                        node_id.index(),
                    ))
                })?;
            translate_zero_pad_1d(
                ctx,
                node_id,
                input,
                *pad_left,
                *pad_right,
                output_shape,
                node_values,
                graph,
            )
        }
        TensorOpKind::MatMul {
            left,
            right,
            transpose_right,
            scale,
        } => translate_matmul(
            node_id,
            *left,
            *right,
            *transpose_right,
            *scale,
            node_values,
            graph,
        ),
        TensorOpKind::Embedding { input, weight } => {
            // Plumb the declared output shape `[*index_dims, embedding_dim]`
            // (like the ZeroPad1d arm above) so the embedding translation can
            // emit a node whose OUTPUT bounds carry the correct rank/shape,
            // independent of the [*index_dims] index tensor's shape.
            let output_shape = ctx
                .all_nodes
                .get(node_id.index())
                .map(|n| n.shape.as_slice())
                .ok_or_else(|| {
                    VerifyError::UnsupportedOp(format!(
                        "Embedding node {} out of bounds",
                        node_id.index(),
                    ))
                })?;
            translate_embedding(node_id, *input, *weight, output_shape, node_values, graph)
        }
        TensorOpKind::Attention {
            q,
            k,
            v,
            mask,
            scale,
        } => translate_attention(
            node_id,
            q,
            k,
            v,
            mask,
            *scale,
            node_values,
            ctx.all_nodes,
            graph,
        ),
        TensorOpKind::Lstm {
            input,
            hidden_state,
            cell_state,
            weight_ih,
            weight_hh,
            bias,
        } => translate_lstm(
            node_id,
            *input,
            *hidden_state,
            *cell_state,
            *weight_ih,
            *weight_hh,
            *bias,
            node_values,
            graph,
        ),
        TensorOpKind::GatedDeltaNet {
            q,
            k,
            v,
            state,
            gate,
            beta,
            scale,
        } => translate_gated_delta_net(
            node_id,
            *q,
            *k,
            *v,
            *state,
            *gate,
            *beta,
            *scale,
            ctx.all_nodes,
            node_values,
            graph,
        ),
        _ => Err(VerifyError::UnsupportedOp(format!("TensorOpKind {kind:?}"))),
    }
}
