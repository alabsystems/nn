// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SiLU (swish) tensor op translation to NY `SiLULayer`.
//!
//! `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
//!
//! Emitting a fused `Layer::SiLU` node (rather than a `Sigmoid` + `MulBinary`
//! decomposition) is what lets ny recognize the SwiGLU `MulBinary(SiLU(gate),
//! up)` pattern and apply its correlation-aware zonotope tightening of the
//! gate*up product, instead of decorrelating the two terms.

use ny_propagate::layers::SiLULayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{checked_tensor_constant, TensorNodeValue};

/// Apply the scalar SiLU function `silu(x) = x / (1 + exp(-x))`.
fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Translate a `TensorOpKind::Silu` node to a NY `SiLULayer`.
///
/// Constant-folds when the input is a known constant: `silu(c) = c/(1+exp(-c))`,
/// matching ny's `SiLULayer` exactly (exact, hence sound). For variable inputs,
/// emits a fused `Layer::SiLU` node in the graph so the downstream SwiGLU
/// `MulBinary` can trigger ny's up/gate-correlation zonotope tightening.
pub(crate) fn translate_silu(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_val = get_value(node_values, input.index(), "SiLU input")?;
    match input_val {
        TensorNodeValue::Constant(c) => {
            let val = silu_scalar(c.get());
            checked_tensor_constant(val, &format!("silu constant fold t{}", node_id.index()))
        }
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            add_unary_node(&node_name, Layer::SiLU(SiLULayer::new()), input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        // Constant-fold path: apply `silu(x) = x/(1+exp(-x))` element-wise to the
        // weight array, matching the scalar `Constant` arm above exactly. The
        // input is a known finite constant, so the output is a deterministic
        // constant (lower == upper) — exact, hence sound.
        TensorNodeValue::WeightTensor(arr) => {
            let folded = arr.mapv(silu_scalar);
            // Reject non-finite outputs, exactly as the Sigmoid/Linear folds do.
            for &val in folded.iter() {
                if !val.is_finite() {
                    return Err(VerifyError::NonFiniteConstant {
                        value: val,
                        context: format!("silu constant fold t{}", node_id.index()),
                    });
                }
            }
            Ok(TensorNodeValue::WeightTensor(folded))
        }
    }
}
