// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM cell tensor op translation to NY decomposed layers.
//!
//! NY has no native LSTM layer, so we decompose the single-step LSTM cell
//! into primitive operations that NY CAN verify:
//!
//! Standard LSTM cell (PyTorch convention):
//!   gates_ih = weight_ih @ input    (+ bias split)
//!   gates_hh = weight_hh @ hidden   (+ bias split)
//!   i, f, g, o = chunk(gates_ih + gates_hh, 4)
//!   c_new = sigmoid(f) * cell_state + sigmoid(i) * tanh(g)
//!   h_new = sigmoid(o) * tanh(c_new)
//!
//! Weight matrices are split at translation time into 4 gate sub-matrices.
//! Each gate gets its own LinearLayer node. This produces a graph of:
//!   4 Linear (ih) + 4 Linear (hh) + 4 Add + 3 Sigmoid + 2 Tanh +
//!   3 MulBinary + 1 Add = 21 NY nodes.
//!
//! Used by Kokoro text encoder and Silero VAD in dvoice (#729).

use ny_propagate::layers::{AddLayer, LinearLayer, MulBinaryLayer, SigmoidLayer, TanhLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::tensor_ir::TensorNodeId;
use ndarray::{Array1, Array2, ArrayD};

use super::TensorNodeValue;
use crate::error::VerifyError;
use crate::util::get_value;

/// Gate names for the 4 LSTM gates (PyTorch order: i, f, g, o).
const GATE_NAMES: [&str; 4] = ["i", "f", "g", "o"];

/// Extract a weight sub-matrix for one gate from the combined [4*H, K] matrix.
///
/// Gate index 0..3 maps to rows [idx*H .. (idx+1)*H].
fn extract_gate_weight(combined: &Array2<f32>, gate_idx: usize, hidden_size: usize) -> Array2<f32> {
    let start = gate_idx * hidden_size;
    combined
        .slice(ndarray::s![start..start + hidden_size, ..])
        .to_owned()
}

/// Extract a bias sub-vector for one gate from the combined [4*H] vector.
fn extract_gate_bias(combined: &Array1<f32>, gate_idx: usize, hidden_size: usize) -> Array1<f32> {
    let start = gate_idx * hidden_size;
    combined
        .slice(ndarray::s![start..start + hidden_size])
        .to_owned()
}

/// Inject a constant state tensor as a NY graph node.
///
/// Creates a `LinearLayer(zeros[state_size, input_dim], const_bias)` node
/// that takes the LSTM input as its parent and produces a constant output
/// regardless of input values. This enables model-level verification with
/// fixed initial LSTM states (e.g., `SileroVadState::zero()`).
fn inject_constant_state(
    prefix: &str,
    state_label: &str,
    const_arr: &ArrayD<f32>,
    input_name: &str,
    input_dim: usize,
    graph: &mut GraphNetwork,
) -> Result<String, VerifyError> {
    let flat: Vec<f32> = const_arr.iter().copied().collect();
    let state_size = flat.len();
    // Zero weight [state_size, input_dim] ensures output is always bias.
    let zeros_w = Array2::zeros((state_size, input_dim));
    let const_bias = Array1::from_vec(flat);
    let node_name = format!("{prefix}_{state_label}_const");
    let layer = LinearLayer::new(zeros_w, Some(const_bias)).map_err(|e| {
        VerifyError::InternalTranslationError {
            context: format!("LSTM {state_label} constant injection failed: {e}"),
        }
    })?;
    // Connect to input_name so it's a valid graph node. The zero weight
    // means the output is always equal to const_bias regardless of input.
    graph.add_node(GraphNode::new(
        node_name.clone(),
        Layer::Linear(layer),
        vec![input_name.to_string()],
    ));
    Ok(node_name)
}

/// Translate a `TensorOpKind::Lstm` node to a NY decomposed graph.
///
/// The `input` data must be `Variable` (a graph node). `hidden_state` and
/// `cell_state` can be either `Variable` (graph nodes) or `WeightTensor`
/// (constant initial states, e.g., zero-initialized for first chunk).
/// Weight tensors (weight_ih, weight_hh) must be `WeightTensor`. Bias is optional.
///
/// The decomposition splits the [4*H, K] weight matrices into per-gate sub-matrices
/// at translation time, then builds a 21-node NY graph representing one
/// LSTM cell step.
pub(super) fn translate_lstm(
    node_id: TensorNodeId,
    input_id: TensorNodeId,
    hidden_id: TensorNodeId,
    cell_id: TensorNodeId,
    weight_ih_id: TensorNodeId,
    weight_hh_id: TensorNodeId,
    bias_id: Option<TensorNodeId>,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let prefix = format!("t{}_lstm", node_id.index());

    // --- Extract weight matrices first (needed for input_dim) ---
    let weight_ih = extract_weight_2d(node_values, weight_ih_id.index(), "LSTM weight_ih")?;
    let weight_hh = extract_weight_2d(node_values, weight_hh_id.index(), "LSTM weight_hh")?;

    let four_h = weight_ih.nrows();
    if four_h % 4 != 0 {
        return Err(VerifyError::WeightValidation {
            op: "LSTM",
            reason: format!("weight_ih rows ({four_h}) must be divisible by 4"),
        });
    }
    let hidden_size = four_h / 4;
    let input_dim = weight_ih.ncols(); // I in weight_ih [4*H, I]

    // Defense-in-depth: weight_hh must have the same 4*H row count.
    if weight_hh.nrows() != four_h {
        return Err(VerifyError::WeightValidation {
            op: "LSTM",
            reason: format!(
                "weight_hh rows ({}) must match weight_ih rows ({four_h})",
                weight_hh.nrows()
            ),
        });
    }

    // --- Extract input variable name ---
    let input_name = match get_value(node_values, input_id.index(), "LSTM input")? {
        TensorNodeValue::Variable(name) => name.clone(),
        other => {
            return Err(VerifyError::UnsupportedOp(format!(
                "LSTM input must be Variable, got {other:?}"
            )));
        }
    };

    // --- Extract hidden/cell state names (Variable or constant WeightTensor) ---
    let hidden_name = match get_value(node_values, hidden_id.index(), "LSTM hidden_state")? {
        TensorNodeValue::Variable(name) => name.clone(),
        TensorNodeValue::WeightTensor(arr) => {
            inject_constant_state(&prefix, "hidden", arr, &input_name, input_dim, graph)?
        }
        other => {
            return Err(VerifyError::UnsupportedOp(format!(
                "LSTM hidden_state must be Variable or WeightTensor, got {other:?}"
            )));
        }
    };

    let cell_name = match get_value(node_values, cell_id.index(), "LSTM cell_state")? {
        TensorNodeValue::Variable(name) => name.clone(),
        TensorNodeValue::WeightTensor(arr) => {
            inject_constant_state(&prefix, "cell", arr, &input_name, input_dim, graph)?
        }
        other => {
            return Err(VerifyError::UnsupportedOp(format!(
                "LSTM cell_state must be Variable or WeightTensor, got {other:?}"
            )));
        }
    };

    // --- Extract bias (optional, split into per-gate sub-vectors) ---
    let bias_vec: Option<Array1<f32>> = if let Some(bid) = bias_id {
        match get_value(node_values, bid.index(), "LSTM bias")? {
            TensorNodeValue::WeightTensor(arr) => {
                let flat: Vec<f32> = arr.iter().copied().collect();
                let bias = Array1::from_vec(flat);
                // Defense-in-depth: bias length must be 4*H.
                if bias.len() != four_h {
                    return Err(VerifyError::WeightValidation {
                        op: "LSTM",
                        reason: format!("bias length ({}) must be 4*H={four_h}", bias.len()),
                    });
                }
                Some(bias)
            }
            other => {
                return Err(VerifyError::WeightValidation {
                    op: "LSTM",
                    reason: format!("bias must be WeightTensor, got {other:?}"),
                });
            }
        }
    } else {
        None
    };

    // --- Build per-gate LinearLayer nodes ---
    // For each gate g in {i, f, g, o}:
    //   {prefix}_{gate}_ih = LinearLayer(W_ih[gate], bias[gate]/2) @ input
    //   {prefix}_{gate}_hh = LinearLayer(W_hh[gate], bias[gate]/2) @ hidden
    //   {prefix}_{gate}_sum = AddLayer({ih}, {hh})
    //
    // Bias is split evenly between ih and hh (PyTorch uses separate bias_ih and bias_hh;
    // when combined into a single bias, we split 50/50 for numerical equivalence with
    // the additive decomposition).

    let mut gate_sum_names = Vec::with_capacity(4);

    for (gate_idx, gate_name) in GATE_NAMES.iter().enumerate() {
        let w_ih_gate = extract_gate_weight(&weight_ih, gate_idx, hidden_size);
        let w_hh_gate = extract_gate_weight(&weight_hh, gate_idx, hidden_size);

        // Split bias evenly between ih and hh paths.
        let (bias_ih, bias_hh) = if let Some(ref bias) = bias_vec {
            let gate_bias = extract_gate_bias(bias, gate_idx, hidden_size);
            let half_bias: Array1<f32> = gate_bias.mapv(|v| v * 0.5);
            (Some(half_bias.clone()), Some(half_bias))
        } else {
            (None, None)
        };

        // ih linear
        let ih_name = format!("{prefix}_{gate_name}_ih");
        let ih_layer =
            LinearLayer::new(w_ih_gate, bias_ih).map_err(|e| VerifyError::WeightValidation {
                op: "LSTM",
                reason: format!("{gate_name}_ih LinearLayer failed: {e}"),
            })?;
        graph.add_node(GraphNode::new(
            ih_name.clone(),
            Layer::Linear(ih_layer),
            vec![input_name.clone()],
        ));

        // hh linear
        let hh_name = format!("{prefix}_{gate_name}_hh");
        let hh_layer =
            LinearLayer::new(w_hh_gate, bias_hh).map_err(|e| VerifyError::WeightValidation {
                op: "LSTM",
                reason: format!("{gate_name}_hh LinearLayer failed: {e}"),
            })?;
        graph.add_node(GraphNode::new(
            hh_name.clone(),
            Layer::Linear(hh_layer),
            vec![hidden_name.clone()],
        ));

        // Sum ih + hh
        let sum_name = format!("{prefix}_{gate_name}_sum");
        graph.add_node(GraphNode::binary(
            sum_name.clone(),
            Layer::Add(AddLayer),
            ih_name,
            hh_name,
        ));

        gate_sum_names.push(sum_name);
    }

    // --- Apply gate activations ---
    // i_gate = sigmoid(i_sum)
    let i_gate = format!("{prefix}_i_gate");
    graph.add_node(GraphNode::new(
        i_gate.clone(),
        Layer::Sigmoid(SigmoidLayer::new()),
        vec![gate_sum_names[0].clone()],
    ));

    // f_gate = sigmoid(f_sum)
    let f_gate = format!("{prefix}_f_gate");
    graph.add_node(GraphNode::new(
        f_gate.clone(),
        Layer::Sigmoid(SigmoidLayer::new()),
        vec![gate_sum_names[1].clone()],
    ));

    // g_candidate = tanh(g_sum)
    let g_candidate = format!("{prefix}_g_cand");
    graph.add_node(GraphNode::new(
        g_candidate.clone(),
        Layer::Tanh(TanhLayer),
        vec![gate_sum_names[2].clone()],
    ));

    // o_gate = sigmoid(o_sum)
    let o_gate = format!("{prefix}_o_gate");
    graph.add_node(GraphNode::new(
        o_gate.clone(),
        Layer::Sigmoid(SigmoidLayer::new()),
        vec![gate_sum_names[3].clone()],
    ));

    // --- Cell state update ---
    // f_cell = f_gate * cell_state
    let f_cell = format!("{prefix}_f_cell");
    graph.add_node(GraphNode::binary(
        f_cell.clone(),
        Layer::MulBinary(MulBinaryLayer),
        f_gate,
        cell_name,
    ));

    // i_g = i_gate * g_candidate
    let i_g = format!("{prefix}_i_g");
    graph.add_node(GraphNode::binary(
        i_g.clone(),
        Layer::MulBinary(MulBinaryLayer),
        i_gate,
        g_candidate,
    ));

    // c_new = f_cell + i_g
    let c_new = format!("{prefix}_c_new");
    graph.add_node(GraphNode::binary(
        c_new.clone(),
        Layer::Add(AddLayer),
        f_cell,
        i_g,
    ));

    // --- Hidden state output ---
    // tanh_c = tanh(c_new)
    let tanh_c = format!("{prefix}_tanh_c");
    graph.add_node(GraphNode::new(
        tanh_c.clone(),
        Layer::Tanh(TanhLayer),
        vec![c_new],
    ));

    // h_new = o_gate * tanh(c_new)
    let h_new = format!("{prefix}_h_new");
    graph.add_node(GraphNode::binary(
        h_new.clone(),
        Layer::MulBinary(MulBinaryLayer),
        o_gate,
        tanh_c,
    ));

    Ok(TensorNodeValue::Variable(h_new))
}

/// Extract a 2-D weight tensor from node values.
fn extract_weight_2d(
    node_values: &[TensorNodeValue],
    idx: usize,
    context: &str,
) -> Result<Array2<f32>, VerifyError> {
    match get_value(node_values, idx, context)? {
        TensorNodeValue::WeightTensor(arr) => {
            let shape = arr.shape();
            if shape.len() != 2 {
                return Err(VerifyError::WeightValidation {
                    op: "LSTM",
                    reason: format!("{context} must be 2-D, got {}-D", shape.len()),
                });
            }
            arr.clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| VerifyError::InternalTranslationError {
                    context: format!("{context} conversion to Array2 failed: {e}"),
                })
        }
        other => Err(VerifyError::WeightValidation {
            op: "LSTM",
            reason: format!("{context} must be WeightTensor, got {other:?}"),
        }),
    }
}

#[cfg(test)]
#[path = "graph_tensor_lstm_tests.rs"]
mod tests;
