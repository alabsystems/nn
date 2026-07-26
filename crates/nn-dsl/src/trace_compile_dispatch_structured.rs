// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structured-op dispatch: linear, matmul, conv, norm, pool, embedding, LSTM,
//! softmax.
//!
//! Part of the category-dispatch refactor (#2305). Workers adding a new
//! structured op only touch this file, not the shared `compile_node` hub.

use nn_core::dyn_tensor::trace::{ComputationGraph, KokoroFusedOp, TraceNode, TraceOp};

use crate::tensor_ir::TensorIRError;

use super::super::trace_compile_ops::{
    compile_ada_layer_norm, compile_avg_pool2d, compile_batch_norm, compile_conv1d, compile_conv2d,
    compile_conv_transpose1d, compile_conv_transpose2d, compile_embedding, compile_group_norm,
    compile_instance_norm, compile_layer_norm, compile_linear, compile_log_softmax, compile_lstm,
    compile_matmul, compile_max_pool1d, compile_max_pool2d, compile_rms_norm, compile_softmax,
};
use super::super::CompiledStep;

/// Try to compile a structured trace op. Returns `None` for non-structured ops.
pub(in crate::trace_compile) fn try_compile(
    node: &TraceNode,
    graph: &ComputationGraph,
) -> Option<Result<CompiledStep, TensorIRError>> {
    match node.op() {
        // -- Matrix multiply --------------------------------------------------
        TraceOp::MatMul => Some(compile_matmul(node, graph)),

        // -- Linear -----------------------------------------------------------
        TraceOp::Linear { weight, bias } | TraceOp::QLinear { weight, bias } => {
            Some(compile_linear(node, graph, weight, bias))
        }

        // -- Convolutions -----------------------------------------------------
        TraceOp::Conv1d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => Some(compile_conv1d(
            node, graph, weight, bias, *padding, *stride, *dilation, *groups,
        )),
        TraceOp::Conv2d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => Some(compile_conv2d(
            node, graph, weight, bias, *padding, *stride, *dilation, *groups,
        )),
        TraceOp::ConvTranspose1d {
            weight,
            bias,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        } => Some(compile_conv_transpose1d(
            node,
            graph,
            weight,
            bias,
            *padding,
            *output_padding,
            *stride,
            *dilation,
            *groups,
        )),
        TraceOp::ConvTranspose2d {
            weight,
            bias,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        } => Some(compile_conv_transpose2d(
            node,
            graph,
            weight,
            bias,
            *padding,
            *output_padding,
            *stride,
            *dilation,
            *groups,
        )),

        // -- Pooling ----------------------------------------------------------
        TraceOp::MaxPool1d {
            kernel_size,
            stride,
            padding,
        } => Some(compile_max_pool1d(
            node,
            graph,
            *kernel_size,
            *stride,
            *padding,
        )),
        TraceOp::AvgPool2d {
            kernel_size,
            stride,
            padding,
        } => Some(compile_avg_pool2d(
            node,
            graph,
            kernel_size,
            stride,
            padding,
        )),
        TraceOp::MaxPool2d {
            kernel_size,
            stride,
            padding,
        } => Some(compile_max_pool2d(
            node,
            graph,
            kernel_size,
            stride,
            padding,
        )),

        // -- Normalization ----------------------------------------------------
        TraceOp::LayerNorm { eps, weight, bias } => {
            Some(compile_layer_norm(node, graph, *eps, weight, bias))
        }
        TraceOp::RmsNorm { eps, weight } => Some(compile_rms_norm(node, graph, *eps, weight)),
        TraceOp::InstanceNorm { eps } => Some(compile_instance_norm(node, graph, *eps)),
        TraceOp::KokoroFused(KokoroFusedOp::AdaLayerNorm {
            norm_weight,
            norm_bias,
            eps,
        }) => Some(compile_ada_layer_norm(
            node,
            graph,
            *eps,
            norm_weight,
            norm_bias,
        )),
        TraceOp::BatchNorm {
            eps,
            weight,
            bias,
            running_mean,
            running_var,
        } => Some(compile_batch_norm(
            node,
            graph,
            *eps,
            weight,
            bias,
            running_mean,
            running_var,
        )),
        TraceOp::GroupNorm {
            num_groups,
            eps,
            weight,
            bias,
        } => Some(compile_group_norm(
            node,
            graph,
            *num_groups,
            *eps,
            weight,
            bias,
        )),

        // -- Softmax / LogSoftmax ---------------------------------------------
        TraceOp::Softmax { dim } => Some(compile_softmax(node, graph, *dim)),
        TraceOp::LogSoftmax { dim } => Some(compile_log_softmax(node, graph, *dim)),

        // -- Embedding --------------------------------------------------------
        TraceOp::Embedding { weight } => Some(compile_embedding(node, graph, weight)),

        // -- LSTM -------------------------------------------------------------
        TraceOp::Lstm {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
            ..
        } => Some(compile_lstm(
            node,
            graph,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            *hidden_size,
        )),

        _ => None,
    }
}
