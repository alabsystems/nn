// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Composite-op dispatch: attention, selection, unfold, upsample, pixel
//! shuffle, accumulation.
//!
//! Part of the category-dispatch refactor (#2305). Workers adding a new
//! composite op only touch this file, not the shared `compile_node` hub.

use nn_core::dyn_tensor::trace::{ComputationGraph, KokoroFusedOp, TraceNode, TraceOp};

use crate::tensor_ir::TensorIRError;

use super::super::trace_compile_attention::{
    compile_index_add, compile_index_put, compile_mha, compile_rope, compile_scatter_add,
    compile_sdpa, compile_sdpa_causal, compile_swiglu,
};
use super::super::trace_compile_resblock::compile_fused_adain_resblock;
use super::super::trace_compile_selection::{compile_gather, compile_index_select};
use super::super::trace_compile_unfold::{
    compile_adaptive_avg_pool2d, compile_pixel_shuffle, compile_pixel_unshuffle, compile_unfold,
    compile_upsample1d, compile_upsample2d,
};
use super::super::CompiledStep;

/// Try to compile a composite trace op. Returns `None` for non-composite ops.
pub(in crate::trace_compile) fn try_compile(
    node: &TraceNode,
    graph: &ComputationGraph,
) -> Option<Result<CompiledStep, TensorIRError>> {
    match node.op() {
        // -- Attention --------------------------------------------------------
        TraceOp::Sdpa { scale } => Some(compile_sdpa(node, graph, *scale)),
        TraceOp::SdpaCausal { scale } => Some(compile_sdpa_causal(node, graph, *scale)),
        TraceOp::RotaryEmbedding {
            head_dim,
            cos_cache,
            sin_cache,
            ..
        } => Some(compile_rope(node, graph, *head_dim, cos_cache, sin_cache)),
        TraceOp::MultiHeadAttention { .. } => Some(compile_mha(node, graph)),

        // -- Composite (inner ops traced individually) ------------------------
        TraceOp::SwiGlu => Some(compile_swiglu(node, graph)),

        // -- Accumulation (needs atomic GPU ops) ------------------------------
        TraceOp::ScatterAdd { dim } => Some(compile_scatter_add(node, graph, *dim)),
        TraceOp::IndexAdd { dim } => Some(compile_index_add(node, graph, *dim)),
        TraceOp::IndexPut { dim } => Some(compile_index_put(node, graph, *dim)),

        // -- Selection / indexing ---------------------------------------------
        TraceOp::IndexSelect { dim } => Some(compile_index_select(node, graph, *dim)),
        TraceOp::Gather { dim } => Some(compile_gather(node, graph, *dim)),

        // -- Unfold -----------------------------------------------------------
        TraceOp::Unfold { dim, size, step } => {
            Some(compile_unfold(node, graph, *dim, *size, *step))
        }

        // -- Adaptive pooling (decomposed to AvgPool2d) -----------------------
        TraceOp::AdaptiveAvgPool2d { output_size } => {
            Some(compile_adaptive_avg_pool2d(node, graph, output_size))
        }

        // -- Pixel shuffle / unshuffle ----------------------------------------
        TraceOp::PixelShuffle { upscale_factor } => {
            Some(compile_pixel_shuffle(node, graph, *upscale_factor))
        }
        TraceOp::PixelUnshuffle { downscale_factor } => {
            Some(compile_pixel_unshuffle(node, graph, *downscale_factor))
        }

        // -- Upsample ---------------------------------------------------------
        TraceOp::Upsample1d { factor } => Some(compile_upsample1d(node, graph, *factor)),
        TraceOp::Upsample2d {
            mode,
            scale_h,
            scale_w,
        } => Some(compile_upsample2d(
            node,
            graph,
            mode.as_str(),
            *scale_h,
            *scale_w,
        )),

        // -- Fused AdaIN residual block (#2459) -------------------------------
        TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock {
            activation,
            adain1_weight,
            adain1_bias,
            adain2_weight,
            adain2_bias,
            conv1_weight,
            conv1_bias,
            conv1_dilation,
            conv1_padding,
            conv2_weight,
            conv2_bias,
            conv2_padding,
            eps,
            residual_scale,
        }) => Some(compile_fused_adain_resblock(
            node,
            graph,
            activation,
            adain1_weight,
            adain1_bias,
            adain2_weight,
            adain2_bias,
            conv1_weight,
            conv1_bias,
            *conv1_dilation,
            *conv1_padding,
            conv2_weight,
            conv2_bias,
            *conv2_padding,
            *eps,
            *residual_scale,
        )),

        _ => None,
    }
}
