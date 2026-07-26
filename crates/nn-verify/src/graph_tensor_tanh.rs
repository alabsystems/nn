// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tanh tensor op translation to NY `TanhLayer`.
//!
//! Part of #761 Direction 1.

use ny_propagate::layers::TanhLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{checked_tensor_constant, TensorNodeValue};

/// Translate a `TensorOpKind::Tanh` node to a NY `TanhLayer`.
///
/// Constant-folds when the input is a known constant: `tanh(c)`.
/// For variable inputs, emits a `TanhLayer` node in the graph.
pub(crate) fn translate_tanh(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_val = get_value(node_values, input.index(), "Tanh input")?;
    match input_val {
        TensorNodeValue::Constant(c) => {
            let val = c.get().tanh();
            checked_tensor_constant(val, &format!("tanh constant fold t{}", node_id.index()))
        }
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            add_unary_node(&node_name, Layer::Tanh(TanhLayer), input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        // Constant-fold path: apply `tanh(x)` element-wise to the weight array.
        // The input is a known finite constant, so the output is a deterministic
        // constant (lower == upper) — exact, hence sound.
        TensorNodeValue::WeightTensor(arr) => {
            let folded = arr.mapv(|x| x.tanh());
            // Reject non-finite outputs, exactly as the Linear/LayerNorm folds do.
            for &val in folded.iter() {
                if !val.is_finite() {
                    return Err(VerifyError::NonFiniteConstant {
                        value: val,
                        context: format!("tanh constant fold t{}", node_id.index()),
                    });
                }
            }
            Ok(TensorNodeValue::WeightTensor(folded))
        }
    }
}
