// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ConvTranspose1d tensor-level IR → NY `ConvTranspose1dLayer` translation.
//!
//! Follows the exact pattern of `graph_tensor_conv1d.rs`:
//! - Input data must be a `Variable` (verified tensor).
//! - Weight and bias must be `WeightTensor` (fixed model parameters).
//! - Creates `Layer::ConvTranspose1d(ConvTranspose1dLayer)` node.
//!
//! Key differences from Conv1d:
//! - Uses `ConvTranspose1dLayer::with_input_length_full()` (same API shape as Conv1d).
//! - Kernel layout: `[in_channels, out_channels/groups, kernel_size]` (in/out swapped).
//! - Supports dilation and groups via NY's `ConvTranspose1dLayer::new_full()`.

use ny_propagate::layers::{ConvTranspose1dLayer, LinearLayer};
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;
use ndarray::{Array1, Array2};

use super::{TensorNodeValue, TensorTranslationContext};
use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

/// Translate a ConvTranspose1d tensor operation to a NY graph node.
///
/// The input data must be a `Variable` (the tensor being verified).
/// Weight and bias must be `WeightTensor` (fixed model parameters).
///
/// Creates a `Layer::ConvTranspose1d(ConvTranspose1dLayer)` node with the
/// weight kernel, optional bias, stride, padding, and input_length set for
/// CROWN backward propagation.
///
/// When `output_padding != 0`, decomposes into ConvTranspose1d(output_padding=0)
/// followed by a LinearLayer zero-pad. NY's `ConvTranspose1dLayer` has
/// no `output_padding` field (#2558).
pub(super) fn translate_conv_transpose_1d(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    weight: &TensorNodeId,
    bias: &Option<TensorNodeId>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    output_padding: usize,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    // Input must be a Variable (data tensor being verified).
    let input_name = match get_value(node_values, input.index(), "ConvTranspose1d input")? {
        TensorNodeValue::Variable(name) => name.clone(),
        TensorNodeValue::Constant(_) => {
            return Err(VerifyError::UnsupportedOp(
                "ConvTranspose1d input must be a variable tensor, not a constant scalar".into(),
            ));
        }
        TensorNodeValue::WeightTensor(_) => {
            return Err(VerifyError::UnsupportedOp(
                "ConvTranspose1d input must be a variable tensor, not a weight tensor".into(),
            ));
        }
    };

    // Weight must be a WeightTensor (constant kernel parameters).
    let kernel_array = match get_value(node_values, weight.index(), "ConvTranspose1d weight")? {
        TensorNodeValue::WeightTensor(arr) => arr.clone(),
        _ => {
            return Err(VerifyError::WeightValidation {
                op: "ConvTranspose1d",
                reason: "weight must be a ConstantTensor binding".into(),
            });
        }
    };

    // Bias extraction (optional).
    let bias_array = if let Some(bias_id) = bias {
        match get_value(node_values, bias_id.index(), "ConvTranspose1d bias")? {
            TensorNodeValue::WeightTensor(arr) => {
                let flat: Vec<f32> = arr.iter().copied().collect();
                Some(Array1::from_vec(flat))
            }
            _ => {
                return Err(VerifyError::WeightValidation {
                    op: "ConvTranspose1d",
                    reason: "bias must be a ConstantTensor binding".into(),
                });
            }
        }
    } else {
        None
    };

    // Get input spatial length for CROWN backward propagation (bounds-checked).
    let input_node =
        ctx.all_nodes
            .get(input.index())
            .ok_or_else(|| VerifyError::InternalTranslationError {
                context: format!(
                    "ConvTranspose1d input node index {} out of bounds (len {})",
                    input.index(),
                    ctx.all_nodes.len()
                ),
            })?;
    let in_length =
        *input_node
            .shape
            .last()
            .ok_or_else(|| VerifyError::InternalTranslationError {
                context: "ConvTranspose1d input shape is empty".into(),
            })?;

    // Build NY ConvTranspose1dLayer with dilation, groups, and input_length.
    let conv_layer = ConvTranspose1dLayer::with_input_length_full(
        kernel_array,
        bias_array,
        stride,
        padding,
        dilation,
        groups,
        in_length,
    )
    .map_err(|e| VerifyError::WeightValidation {
        op: "ConvTranspose1d",
        reason: format!("layer construction failed: {e}"),
    })?;

    if output_padding == 0 {
        // Standard path: single ConvTranspose1d node.
        let node_name = format!("t{}", node_id.index());
        add_unary_node(
            &node_name,
            Layer::ConvTranspose1d(conv_layer),
            &input_name,
            graph,
        );
        Ok(TensorNodeValue::Variable(node_name))
    } else {
        // Decompose: ConvTranspose1d(output_padding=0) + LinearLayer zero-pad.
        // NY's ConvTranspose1dLayer has no output_padding field (#2558).
        let conv_name = format!("t{}_conv", node_id.index());
        add_unary_node(
            &conv_name,
            Layer::ConvTranspose1d(conv_layer),
            &input_name,
            graph,
        );

        // Compute T_mid (output length without output_padding):
        // T_mid = (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
        let kernel_size = *ctx
            .all_nodes
            .get(weight.index())
            .and_then(|n| n.shape.last())
            .ok_or_else(|| VerifyError::InternalTranslationError {
                context: "ConvTranspose1d output_padding: cannot determine kernel_size".into(),
            })?;

        let t_mid = in_length
            .checked_sub(1)
            .and_then(|v| v.checked_mul(stride))
            .and_then(|v| {
                dilation
                    .checked_mul(kernel_size.checked_sub(1)?)
                    .and_then(|dk| v.checked_add(dk))
            })
            .and_then(|v| v.checked_add(1))
            .and_then(|v| v.checked_sub(2usize.checked_mul(padding)?))
            .ok_or_else(|| VerifyError::DimensionOverflow {
                op: "ConvTranspose1d",
                context: format!(
                    "T_mid overflow (in_len={in_length}, stride={stride}, \
                     padding={padding}, kernel_size={kernel_size})"
                ),
            })?;
        let t_out =
            t_mid
                .checked_add(output_padding)
                .ok_or_else(|| VerifyError::DimensionOverflow {
                    op: "ConvTranspose1d",
                    context: format!("T_out overflow ({t_mid} + {output_padding})"),
                })?;

        // Build [T_out, T_mid] weight: identity at [0..T_mid], zero at [T_mid..T_out].
        // Same approach as ZeroPad1d (graph_tensor_zero_pad.rs).
        let mut pad_weight = Array2::<f32>::zeros((t_out, t_mid));
        for t in 0..t_mid {
            pad_weight[[t, t]] = 1.0;
        }

        let linear =
            LinearLayer::new(pad_weight, None).map_err(|e| VerifyError::WeightValidation {
                op: "ConvTranspose1d",
                reason: format!("output_padding LinearLayer failed: {e}"),
            })?;

        let node_name = format!("t{}", node_id.index());
        add_unary_node(&node_name, Layer::Linear(linear), &conv_name, graph);
        Ok(TensorNodeValue::Variable(node_name))
    }
}
