// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! BiLSTM decomposition: expands a bidirectional LSTM into per-layer
//! forward + backward LSTM pairs with flip and cat nodes.

use nn_core::dyn_tensor::trace::TraceOp;

use super::super::{
    get_arg, optional_bool, optional_int, require_tensor_name, resolve_weight, ExpandedNode,
    ImportError, Node, OpMapContext,
};

/// Decompose a bidirectional LSTM into per-layer forward + backward passes.
///
/// Each layer produces 9 nodes:
///   4 zero-constant (h0f, c0f, h0b, c0b) + fwd LSTM + flip_in + bwd LSTM +
///   flip_out + cat(fwd, bwd).
///
/// Multi-layer BiLSTMs chain: layer i+1 reads the cat output of layer i.
/// The final cat node is named `output_name` for seamless graph integration.
pub(crate) fn expand_bilstm(
    node: &Node,
    ctx: &OpMapContext<'_>,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Vec<ExpandedNode>, ImportError> {
    use nn_core::DType;

    let input = require_tensor_name(node, "input")?;
    let num_layers = optional_int(node, "num_layers").unwrap_or(1) as usize;
    let batch_first = optional_bool(node, "batch_first", true);
    let has_biases = optional_bool(node, "has_biases", true);

    // Weights per direction: w_ih, w_hh, [b_ih, b_hh]
    let weights_per_dir: usize = if has_biases { 4 } else { 2 };

    let params_arg = get_arg(node, "params")?;
    let param_names =
        params_arg
            .as_tensor_names()
            .ok_or_else(|| ImportError::WrongArgumentType {
                op_target: node.target.clone(),
                arg_name: "params".to_string(),
                expected: "tensor list",
                actual: "non-tensor-list".to_string(),
            })?;

    let expected_total = weights_per_dir * 2 * num_layers;
    if param_names.len() < expected_total {
        return Err(ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: "params".to_string(),
            expected: "sufficient params for all BiLSTM layers",
            actual: format!(
                "got {} params, need {} ({} layers × 2 dirs × {} per dir)",
                param_names.len(),
                expected_total,
                num_layers,
                weights_per_dir
            ),
        });
    }

    // Derive hidden_size from first w_hh: shape [4*H, H].
    let first_whh = resolve_weight(param_names[1], ctx)?;
    let hidden_size = *first_whh
        .shape()
        .get(1)
        .ok_or_else(|| ImportError::WrongArgumentType {
            op_target: node.target.clone(),
            arg_name: "params (w_hh)".to_string(),
            expected: "2D weight [4*H, H]",
            actual: format!("shape {:?}", first_whh.shape()),
        })?;

    // Sequence dimension: dim 1 if batch_first, dim 0 otherwise.
    let seq_dim = if batch_first { 1 } else { 0 };

    let mut expanded = Vec::new();
    let mut prev_output = input;

    for layer in 0..num_layers {
        let prefix = format!("{output_name}_L{layer}");

        // Input shape for this layer.
        let layer_in_shape: Vec<usize> = if layer == 0 {
            input_shape.to_vec()
        } else {
            // After first layer, feature dim doubles (forward + backward).
            let mut s = input_shape.to_vec();
            if let Some(last) = s.last_mut() {
                *last = 2 * hidden_size;
            }
            s
        };

        // LSTM output shape: same as input but with hidden_size as last dim.
        let lstm_out_shape: Vec<usize> = {
            let mut s = layer_in_shape.clone();
            if let Some(last) = s.last_mut() {
                *last = hidden_size;
            }
            s
        };

        // Constant zero nodes for initial states.
        // Shape: [1, 1, hidden_size] for batch_first, [1, 1, hidden_size] otherwise.
        let state_shape = vec![1, 1, hidden_size];

        for (suffix, is_hidden) in [("h0f", true), ("c0f", true), ("h0b", false), ("c0b", false)] {
            let _ = is_hidden; // both use same shape
            expanded.push(ExpandedNode {
                name: format!("{prefix}_{suffix}"),
                op: TraceOp::Constant { value: 0.0 },
                input_names: vec![],
                output_shape: state_shape.clone(),
                output_dtype: DType::F32,
            });
        }

        // Resolve weights for this layer.
        let base = layer * weights_per_dir * 2;
        let fwd_wih = resolve_weight(param_names[base], ctx)?;
        let fwd_whh = resolve_weight(param_names[base + 1], ctx)?;
        let (fwd_bih, fwd_bhh) = if has_biases {
            (
                Some(resolve_weight(param_names[base + 2], ctx)?),
                Some(resolve_weight(param_names[base + 3], ctx)?),
            )
        } else {
            (None, None)
        };

        let bwd_base = base + weights_per_dir;
        let bwd_wih = resolve_weight(param_names[bwd_base], ctx)?;
        let bwd_whh = resolve_weight(param_names[bwd_base + 1], ctx)?;
        let (bwd_bih, bwd_bhh) = if has_biases {
            (
                Some(resolve_weight(param_names[bwd_base + 2], ctx)?),
                Some(resolve_weight(param_names[bwd_base + 3], ctx)?),
            )
        } else {
            (None, None)
        };

        // Forward LSTM.
        expanded.push(ExpandedNode {
            name: format!("{prefix}_fwd"),
            op: TraceOp::Lstm {
                weight_ih: fwd_wih,
                weight_hh: fwd_whh,
                bias_ih: fwd_bih,
                bias_hh: fwd_bhh,
                hidden_size,
                initial_hidden: None,
                initial_cell: None,
            },
            input_names: vec![
                prev_output.clone(),
                format!("{prefix}_h0f"),
                format!("{prefix}_c0f"),
            ],
            output_shape: lstm_out_shape.clone(),
            output_dtype: DType::F32,
        });

        // Flip input for backward pass (reverse along seq_dim).
        expanded.push(ExpandedNode {
            name: format!("{prefix}_flip_in"),
            op: TraceOp::Flip { dim: seq_dim },
            input_names: vec![prev_output.clone()],
            output_shape: layer_in_shape.clone(),
            output_dtype: DType::F32,
        });

        // Backward LSTM (on flipped input).
        expanded.push(ExpandedNode {
            name: format!("{prefix}_bwd"),
            op: TraceOp::Lstm {
                weight_ih: bwd_wih,
                weight_hh: bwd_whh,
                bias_ih: bwd_bih,
                bias_hh: bwd_bhh,
                hidden_size,
                initial_hidden: None,
                initial_cell: None,
            },
            input_names: vec![
                format!("{prefix}_flip_in"),
                format!("{prefix}_h0b"),
                format!("{prefix}_c0b"),
            ],
            output_shape: lstm_out_shape.clone(),
            output_dtype: DType::F32,
        });

        // Flip backward output to restore original time order.
        expanded.push(ExpandedNode {
            name: format!("{prefix}_flip_out"),
            op: TraceOp::Flip { dim: seq_dim },
            input_names: vec![format!("{prefix}_bwd")],
            output_shape: lstm_out_shape.clone(),
            output_dtype: DType::F32,
        });

        // Cat forward + backward along feature dim.
        let cat_shape: Vec<usize> = {
            let mut s = layer_in_shape;
            if let Some(last) = s.last_mut() {
                *last = 2 * hidden_size;
            }
            s
        };
        let is_last_layer = layer == num_layers - 1;
        let cat_name = if is_last_layer {
            output_name.to_string()
        } else {
            format!("{prefix}_cat")
        };

        let last_dim = cat_shape.len().saturating_sub(1);
        expanded.push(ExpandedNode {
            name: cat_name.clone(),
            op: TraceOp::Cat {
                dim: last_dim,
                num_inputs: 2,
            },
            input_names: vec![format!("{prefix}_fwd"), format!("{prefix}_flip_out")],
            output_shape: cat_shape,
            output_dtype: DType::F32,
        });

        prev_output = cat_name;
    }

    Ok(expanded)
}
