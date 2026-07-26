// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ReLU tensor op translation to NY `ReLULayer`.
//!
//! Part of #761 Direction 1.

use ny_propagate::layers::ReLULayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{checked_tensor_constant, TensorNodeValue};

/// Translate a `TensorOpKind::Relu` node to a NY `ReLULayer`.
///
/// Constant-folds when the input is a known constant: `relu(c) = max(c, 0)`.
/// For variable inputs, emits a `ReLULayer` node in the graph.
pub(crate) fn translate_relu(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_val = get_value(node_values, input.index(), "Relu input")?;
    match input_val {
        TensorNodeValue::Constant(c) => {
            let val = c.get().max(0.0);
            checked_tensor_constant(val, &format!("relu constant fold t{}", node_id.index()))
        }
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            add_unary_node(&node_name, Layer::ReLU(ReLULayer::new()), input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        // Constant-fold path: apply `relu(x) = max(x, 0)` element-wise to the
        // weight array. The input is a known finite constant, so the output is a
        // deterministic constant (lower == upper) — exact, hence sound.
        TensorNodeValue::WeightTensor(arr) => {
            let folded = arr.mapv(|x| x.max(0.0));
            // Reject non-finite outputs, exactly as the Linear/LayerNorm folds do.
            for &val in folded.iter() {
                if !val.is_finite() {
                    return Err(VerifyError::NonFiniteConstant {
                        value: val,
                        context: format!("relu constant fold t{}", node_id.index()),
                    });
                }
            }
            Ok(TensorNodeValue::WeightTensor(folded))
        }
    }
}
