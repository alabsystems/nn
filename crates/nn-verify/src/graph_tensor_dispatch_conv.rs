// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv and norm dispatch arms for tensor-level IR → NY translation.
//!
//! Extracted from `graph_tensor_dispatch.rs` (#1575) to keep files under 400 lines.
//! Contains Conv1d, Conv2d, ConvTranspose1d, RmsNorm, InstanceNorm1d, AdaIN1d,
//! LayerNorm, and BatchNorm dispatch arms.

use ny_propagate::GraphNetwork;
use nn_dsl::tensor_ir::{TensorNodeId, TensorOpKind};

use super::super::{
    adain::translate_adain1d, batch_norm::translate_batch_norm, conv1d::translate_conv1d,
    conv2d::translate_conv2d, conv_transpose_1d::translate_conv_transpose_1d,
    instance_norm::translate_instance_norm_1d, layer_norm::translate_layer_norm,
    rms_norm::translate_rms_norm, TensorNodeValue, TensorTranslationContext,
};
use crate::error::VerifyError;

/// Dispatch conv-like and normalization tensor ops to NY translation.
///
/// Returns `Some(Ok(value))` if the op was handled, `Some(Err(...))` on error,
/// or `None` if the op is not a conv/norm variant.
pub(super) fn translate_conv_or_norm(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    kind: &TensorOpKind,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Option<Result<TensorNodeValue, VerifyError>> {
    match kind {
        TensorOpKind::Conv1d {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        } => Some(translate_conv1d(
            ctx,
            node_id,
            input,
            weight,
            bias,
            *stride,
            *padding,
            *dilation,
            *groups,
            node_values,
            graph,
        )),

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
        } => Some(translate_conv2d(
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
        )),

        TensorOpKind::ConvTranspose1d {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            output_padding,
        } => Some(translate_conv_transpose_1d(
            ctx,
            node_id,
            input,
            weight,
            bias,
            *stride,
            *padding,
            *dilation,
            *groups,
            *output_padding,
            node_values,
            graph,
        )),

        TensorOpKind::RmsNorm {
            input,
            eps,
            axis,
            weight,
        } => Some(translate_rms_norm(
            ctx,
            node_id,
            input,
            eps,
            *axis,
            weight,
            node_values,
            ctx.all_nodes,
            graph,
        )),

        TensorOpKind::InstanceNorm1d {
            input,
            eps,
            axis,
            gamma,
            beta,
        } => Some(translate_instance_norm_1d(
            ctx,
            node_id,
            input,
            eps,
            *axis,
            gamma.as_ref(),
            beta.as_ref(),
            node_values,
            ctx.all_nodes,
            graph,
        )),

        TensorOpKind::AdaIN1d {
            input,
            eps,
            axis,
            style_gamma,
            style_beta,
        } => Some(translate_adain1d(
            ctx,
            node_id,
            input,
            eps,
            *axis,
            style_gamma,
            style_beta,
            node_values,
            ctx.all_nodes,
            graph,
        )),

        TensorOpKind::LayerNorm {
            input,
            eps,
            axis,
            weight,
            bias,
        } => Some(translate_layer_norm(
            ctx,
            node_id,
            input,
            eps,
            *axis,
            weight,
            bias,
            node_values,
            ctx.all_nodes,
            graph,
        )),

        TensorOpKind::BatchNorm {
            input,
            running_mean,
            running_var,
            weight,
            bias,
            eps,
        } => Some(translate_batch_norm(
            node_id,
            input,
            running_mean,
            running_var,
            weight,
            bias,
            eps,
            node_values,
            ctx.all_nodes,
            graph,
        )),

        // SAFETY: TensorOpKind is #[non_exhaustive]. None means "not a conv/norm
        // op" — the parent translate_tensor_node() tries the next dispatcher and
        // ultimately returns UnsupportedOp if no dispatcher handles the variant.
        _ => None,
    }
}
