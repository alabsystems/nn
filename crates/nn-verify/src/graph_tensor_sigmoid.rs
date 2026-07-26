// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sigmoid tensor op translation to NY `SigmoidLayer`.

use ny_propagate::layers::SigmoidLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{checked_tensor_constant, TensorNodeValue};

/// Translate a `TensorOpKind::Sigmoid` node to a NY `SigmoidLayer`.
///
/// Constant-folds when the input is a known constant: `sigmoid(c) = 1/(1+exp(-c))`.
/// For variable inputs, emits a `SigmoidLayer` node in the graph.
pub(crate) fn translate_sigmoid(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_val = get_value(node_values, input.index(), "Sigmoid input")?;
    match input_val {
        TensorNodeValue::Constant(c) => {
            let val = 1.0 / (1.0 + (-c.get()).exp());
            checked_tensor_constant(val, &format!("sigmoid constant fold t{}", node_id.index()))
        }
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            add_unary_node(
                &node_name,
                Layer::Sigmoid(SigmoidLayer::new()),
                input_name,
                graph,
            );
            Ok(TensorNodeValue::Variable(node_name))
        }
        // Constant-fold path: apply `sigmoid(x) = 1/(1+exp(-x))` element-wise to
        // the weight array, matching the scalar `Constant` arm above exactly. The
        // input is a known finite constant, so the output is a deterministic
        // constant (lower == upper) — exact, hence sound.
        TensorNodeValue::WeightTensor(arr) => {
            let folded = arr.mapv(|x| 1.0 / (1.0 + (-x).exp()));
            // Reject non-finite outputs, exactly as the Linear/LayerNorm folds do.
            for &val in folded.iter() {
                if !val.is_finite() {
                    return Err(VerifyError::NonFiniteConstant {
                        value: val,
                        context: format!("sigmoid constant fold t{}", node_id.index()),
                    });
                }
            }
            Ok(TensorNodeValue::WeightTensor(folded))
        }
    }
}
