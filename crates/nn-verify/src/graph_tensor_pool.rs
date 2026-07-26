// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pooling / LogSoftmax / ConvTranspose2d / grouped-Conv2d tensor-level IR → NY
//! translation helpers.
//!
//! Created to close the verifier op-coverage gap for `TensorOpKind::AvgPool2d`,
//! `MaxPool2d`, `LogSoftmax`, `ConvTranspose2d`, and grouped `Conv2d`
//! (`groups > 1`) — ops that previously fell through the dispatch catch-all and
//! were rejected as `UnsupportedOp`, even though NY ships fully-implemented,
//! sound IBP+CROWN layers for them.
//!
//! Declared as a child module of `graph_tensor_dispatch.rs`, so paths up to the
//! `graph_tensor` parent use `super::super::` (matching `graph_tensor_dispatch_conv.rs`).
//!
//! # Soundness
//! Every helper here builds a NY `Layer` whose IBP and CROWN relaxations have
//! been confirmed sound for the op in question:
//! - `AveragePoolLayer` — IBP is exact (avg-pool is linear); CROWN distributes
//!   `1/divisor` to in-bounds positions exactly. `count_include_pad = true`
//!   matches PyTorch `nn.AvgPool2d`'s default (and the existing TraceOp
//!   translator), which is the only semantics representable by `Pool2dParams`.
//! - `MaxPool2dLayer` — IBP is exact (max is monotone); CROWN routes the
//!   gradient through a definite winner or falls back to sound constant IBP
//!   bounds. `MaxPool2dLayer::new` uses `use_negative_inf_padding = true`
//!   (correct PyTorch semantics: padding never raises the max).
//! - `LogSoftmaxLayer::new` sets `sound = true`, so CROWN always takes the
//!   LSE-based affine (sound) path, never the heuristic sampling path.
//! - `ConvTranspose2dLayer` — IBP via W⁺/W⁻ interval splitting; CROWN backward is
//!   a regular conv. NY enforces `output_padding < stride` for sound backward.
//!   NY has **no** grouped transpose-conv, so `groups > 1` is rejected here.
//! - `Conv2dLayer` — IBP/CROWN thread `groups` and `dilation` natively
//!   (`conv2d_ibp_forward_grouped` / grouped transposed-conv backward), exactly.

use ny_propagate::layers::{
    AveragePoolLayer, Conv2dLayer, ConvTranspose2dLayer, LogSoftmaxLayer, MaxPool2dLayer,
};
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::{Pool2dParams, TensorNodeId};
use ndarray::Array1;

use super::super::{TensorNodeValue, TensorTranslationContext};
use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

/// Resolve the input of a unary spatial op to the NY `Variable` name it must be.
///
/// The data tensor under verification is always a `Variable`; a `Constant`
/// scalar or `WeightTensor` in this position indicates a malformed graph.
fn require_variable_input(
    node_values: &[TensorNodeValue],
    input: &TensorNodeId,
    op: &'static str,
) -> Result<String, VerifyError> {
    match get_value(node_values, input.index(), op)? {
        TensorNodeValue::Variable(name) => Ok(name.clone()),
        TensorNodeValue::Constant(_) => Err(VerifyError::UnsupportedOp(format!(
            "{op} input must be a variable tensor, not a constant scalar"
        ))),
        TensorNodeValue::WeightTensor(_) => Err(VerifyError::UnsupportedOp(format!(
            "{op} input must be a variable tensor, not a weight tensor"
        ))),
    }
}

/// Translate `TensorOpKind::AvgPool2d` to a NY `Layer::AveragePool` node.
///
/// # Soundness
/// `AveragePoolLayer` IBP is exact and CROWN is exact (avg-pool is a fixed
/// linear map). `count_include_pad = true` matches PyTorch `nn.AvgPool2d`'s
/// default — the only semantics `Pool2dParams` can express (it carries no
/// `count_include_pad`/`ceil_mode` fields), so this is a faithful, sound match.
pub(super) fn translate_avg_pool2d(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    params: &Pool2dParams,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_name = require_variable_input(node_values, input, "AvgPool2d")?;

    // PyTorch nn.AvgPool2d defaults count_include_pad=true. Pool2dParams cannot
    // express count_include_pad=false / ceil_mode, so true is the faithful match.
    let layer = AveragePoolLayer::new(
        (params.kernel_h, params.kernel_w),
        (params.stride_h, params.stride_w),
        (params.padding_h, params.padding_w),
        /* count_include_pad = */ true,
    );

    let node_name = format!("t{}", node_id.index());
    add_unary_node(&node_name, Layer::AveragePool(layer), &input_name, graph);
    Ok(TensorNodeValue::Variable(node_name))
}

/// Translate `TensorOpKind::MaxPool2d` to a NY `Layer::MaxPool2d` node.
///
/// # Soundness
/// `MaxPool2dLayer` IBP is exact (max is monotone increasing). CROWN routes the
/// gradient through a definite winner when one exists, else falls back to sound
/// constant IBP bounds. `MaxPool2dLayer::new` sets `use_negative_inf_padding =
/// true`, i.e. padding contributes `-inf` and never raises the max — the correct
/// PyTorch `nn.MaxPool2d` semantics.
pub(super) fn translate_max_pool2d(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    params: &Pool2dParams,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_name = require_variable_input(node_values, input, "MaxPool2d")?;

    let layer = MaxPool2dLayer::new(
        (params.kernel_h, params.kernel_w),
        (params.stride_h, params.stride_w),
        (params.padding_h, params.padding_w),
    );

    let node_name = format!("t{}", node_id.index());
    add_unary_node(&node_name, Layer::MaxPool2d(layer), &input_name, graph);
    Ok(TensorNodeValue::Variable(node_name))
}

/// Translate `TensorOpKind::LogSoftmax` to a NY `Layer::LogSoftmax` node.
///
/// # Soundness
/// `LogSoftmaxLayer::new` sets `sound = true`, so both IBP (LSE with directed
/// f64 rounding) and CROWN (LSE-based affine bounds) take their sound paths; the
/// heuristic sampling relaxation is never engaged. The NY soundness checker only
/// flags `LogSoftmax` as heuristic when `sound == false`, which is not the case
/// here. The IR axis is `i32` (Python-style negative indexing), matching the
/// `LogSoftmaxLayer::axis` convention exactly.
pub(super) fn translate_log_softmax(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    axis: i32,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_name = require_variable_input(node_values, input, "LogSoftmax")?;

    let layer = LogSoftmaxLayer::new(axis);

    let node_name = format!("t{}", node_id.index());
    add_unary_node(&node_name, Layer::LogSoftmax(layer), &input_name, graph);
    Ok(TensorNodeValue::Variable(node_name))
}

/// Translate `TensorOpKind::ConvTranspose2d` to a NY `Layer::ConvTranspose2d` node.
///
/// Weight layout: IR `[C_in, C_out/groups, kH, kW]`. With `groups == 1` this is
/// `[C_in, C_out, kH, kW]`, exactly NY's `ConvTranspose2dLayer` kernel layout
/// (`out_channels == kernel.shape()[1]`).
///
/// # Soundness
/// NY's `ConvTranspose2dLayer` is sound (IBP via W⁺/W⁻ interval split; CROWN
/// backward is a regular strided conv). `new_full` enforces `output_padding <
/// stride` per dimension, which is required for the CROWN backward conv to
/// recover the exact input size — violations are rejected (returned as an
/// error), never silently approximated.
///
/// NY's `ConvTranspose2dLayer` has **no groups support** (no `groups` field on
/// the struct or any constructor). Grouped transposed conv (`groups > 1`) is
/// therefore rejected: forwarding only some of the kernel while treating it as
/// `groups = 1` would be unsound, and there is no sound NY path. The other half
/// of the failing-case set (`groups == 1`) is fully covered.
pub(super) fn translate_conv_transpose_2d(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    weight: &TensorNodeId,
    bias: &Option<TensorNodeId>,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    output_padding_h: usize,
    output_padding_w: usize,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    // NY's ConvTranspose2dLayer has no groups support; treating a grouped kernel
    // as groups=1 would be unsound. Reject rather than approximate.
    if groups != 1 {
        return Err(VerifyError::UnsupportedOp(format!(
            "ConvTranspose2d groups={groups} not supported by NY ConvTranspose2dLayer \
             (only groups=1)"
        )));
    }

    let input_name = require_variable_input(node_values, input, "ConvTranspose2d")?;

    // Weight must be a fixed model parameter: kernel [C_in, C_out, kH, kW].
    let kernel_array = match get_value(node_values, weight.index(), "ConvTranspose2d weight")? {
        TensorNodeValue::WeightTensor(arr) => arr.clone(),
        _ => {
            return Err(VerifyError::WeightValidation {
                op: "ConvTranspose2d",
                reason: "weight must be a ConstantTensor binding".into(),
            });
        }
    };

    // Bias extraction (optional).
    let bias_array = if let Some(bias_id) = bias {
        match get_value(node_values, bias_id.index(), "ConvTranspose2d bias")? {
            TensorNodeValue::WeightTensor(arr) => {
                let flat: Vec<f32> = arr.iter().copied().collect();
                Some(Array1::from_vec(flat))
            }
            _ => {
                return Err(VerifyError::WeightValidation {
                    op: "ConvTranspose2d",
                    reason: "bias must be a ConstantTensor binding".into(),
                });
            }
        }
    } else {
        None
    };

    // Input spatial dims (H, W) for CROWN backward propagation.
    let input_node =
        ctx.all_nodes
            .get(input.index())
            .ok_or_else(|| VerifyError::InternalTranslationError {
                context: format!(
                    "ConvTranspose2d input node index {} out of bounds (len {})",
                    input.index(),
                    ctx.all_nodes.len()
                ),
            })?;
    let input_shape = &input_node.shape;
    // Input shape is [C, H, W] or [B, C, H, W]; take last two dims.
    if input_shape.len() < 2 {
        return Err(VerifyError::UnsupportedOp(
            "ConvTranspose2d input shape must have at least 2 dimensions (H, W)".into(),
        ));
    }
    let in_height = input_shape[input_shape.len() - 2];
    let in_width = input_shape[input_shape.len() - 1];

    // new_full validates output_padding < stride per dim (sound CROWN backward).
    let mut conv_layer = ConvTranspose2dLayer::new_full(
        kernel_array,
        bias_array,
        (stride_h, stride_w),
        (padding_h, padding_w),
        (dilation_h, dilation_w),
        (output_padding_h, output_padding_w),
    )
    .map_err(|e| VerifyError::WeightValidation {
        op: "ConvTranspose2d",
        reason: format!("layer construction failed: {e}"),
    })?;
    conv_layer.set_input_shape(in_height, in_width);

    let node_name = format!("t{}", node_id.index());
    add_unary_node(
        &node_name,
        Layer::ConvTranspose2d(conv_layer),
        &input_name,
        graph,
    );
    Ok(TensorNodeValue::Variable(node_name))
}

/// Translate a grouped `TensorOpKind::Conv2d` (`groups > 1`) to a NY
/// `Layer::Conv2d` node, forwarding `groups` (and `dilation`) natively.
///
/// `groups == 1` is left to the existing `graph_tensor_conv2d.rs` translator;
/// this path handles only the previously-rejected `groups > 1` case so that the
/// well-tested single-group translator (incl. its dilated-kernel expansion) is
/// untouched.
///
/// # Soundness
/// NY's `Conv2dLayer` threads `groups` and `dilation` straight into
/// `conv2d_ibp_forward_grouped` (IBP, exact W⁺/W⁻ interval split) and the
/// grouped transposed-conv CROWN backward — both sound. Kernel layout is `[C_out,
/// C_in/groups, kH, kW]`, identical to the IR's grouped-Conv2d weight layout, so
/// `groups` is forwarded, never dropped. `new_dilated` validates `out_channels %
/// groups == 0`.
pub(super) fn translate_conv2d_grouped(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    weight: &TensorNodeId,
    bias: &Option<TensorNodeId>,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_name = require_variable_input(node_values, input, "Conv2d")?;

    // Weight must be a fixed model parameter: kernel [C_out, C_in/groups, kH, kW].
    let kernel_array = match get_value(node_values, weight.index(), "Conv2d weight")? {
        TensorNodeValue::WeightTensor(arr) => arr.clone(),
        _ => {
            return Err(VerifyError::WeightValidation {
                op: "Conv2d",
                reason: "weight must be a ConstantTensor binding".into(),
            });
        }
    };

    // Bias extraction (optional).
    let bias_array = if let Some(bias_id) = bias {
        match get_value(node_values, bias_id.index(), "Conv2d bias")? {
            TensorNodeValue::WeightTensor(arr) => {
                let flat: Vec<f32> = arr.iter().copied().collect();
                Some(Array1::from_vec(flat))
            }
            _ => {
                return Err(VerifyError::WeightValidation {
                    op: "Conv2d",
                    reason: "bias must be a ConstantTensor binding".into(),
                });
            }
        }
    } else {
        None
    };

    // Input spatial dims (H, W) for CROWN backward propagation.
    let input_node =
        ctx.all_nodes
            .get(input.index())
            .ok_or_else(|| VerifyError::InternalTranslationError {
                context: format!(
                    "Conv2d input node index {} out of bounds (len {})",
                    input.index(),
                    ctx.all_nodes.len()
                ),
            })?;
    let input_shape = &input_node.shape;
    if input_shape.len() < 2 {
        return Err(VerifyError::UnsupportedOp(
            "Conv2d input shape must have at least 2 dimensions (H, W)".into(),
        ));
    }
    let in_height = input_shape[input_shape.len() - 2];
    let in_width = input_shape[input_shape.len() - 1];

    // new_dilated forwards both dilation and groups natively (sound, exact);
    // validates out_channels % groups == 0. set_input_shape supplies the spatial
    // dims CROWN backward needs.
    let mut conv_layer = Conv2dLayer::new_dilated(
        kernel_array,
        bias_array,
        (stride_h, stride_w),
        (padding_h, padding_w),
        (dilation_h, dilation_w),
        groups,
    )
    .map_err(|e| VerifyError::WeightValidation {
        op: "Conv2d",
        reason: format!("grouped layer construction failed: {e}"),
    })?;
    conv_layer.set_input_shape(in_height, in_width);

    let node_name = format!("t{}", node_id.index());
    add_unary_node(&node_name, Layer::Conv2d(conv_layer), &input_name, graph);
    Ok(TensorNodeValue::Variable(node_name))
}
