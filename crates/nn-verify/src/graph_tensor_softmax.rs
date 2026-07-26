// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Softmax tensor op translation to NY `SoftmaxLayer`.
//!
//! Softmax is a multi-element operation that normalizes along an axis so
//! outputs sum to 1.0. Unlike element-wise activations (Sigmoid, GELU),
//! softmax depends on all elements along the reduction axis.
//!
//! NY's `SoftmaxLayer` supports IBP bound propagation (sound mode
//! with LSE-based affine bounds) and takes `axis: i32` with negative indexing.

use ny_propagate::layers::SoftmaxLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{checked_tensor_constant, TensorNodeValue, TensorTranslationContext};

/// Translate a `TensorOpKind::Softmax` node to a NY `SoftmaxLayer`.
///
/// The IR stores `axis: i32` which is forwarded directly to NY
/// (both use Python-style negative indexing). In the multi-variable scheme each
/// variable enters at its TRUE rank (no stacking dimension), so the declared
/// axis needs no offset.
///
/// Constant-fold: softmax of a single scalar constant is always 1.0
/// (single-element normalization).
///
/// Weight tensors as softmax input are rejected — softmax operates on
/// bounded variables (attention logits), not fixed parameters.
pub(super) fn translate_softmax(
    _ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    axis: i32,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_val = get_value(node_values, input.index(), "Softmax input")?;
    match input_val {
        TensorNodeValue::Constant(_c) => {
            // Softmax of a single scalar: exp(c) / exp(c) = 1.0
            checked_tensor_constant(1.0, &format!("softmax constant fold t{}", node_id.index()))
        }
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            // Each variable enters at its TRUE rank (see `setup_multi_variable_inputs`),
            // so the user-declared softmax axis is forwarded directly with no offset.
            let layer = Layer::Softmax(SoftmaxLayer::new(axis));
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::WeightTensor(_) => Err(VerifyError::UnsupportedOp(
            "Softmax on WeightTensor not supported".into(),
        )),
    }
}
