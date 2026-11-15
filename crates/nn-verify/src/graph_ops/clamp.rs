// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Clamp translation: constant-bounds ClipLayer.

use ny_propagate::layers::ClipLayer;
use ny_propagate::{GraphNetwork, Layer};

use crate::error::VerifyError;
use crate::graph::{add_unary_node, checked_constant, NodeValue};

/// Evaluate constant clamp: returns `x.clamp(lo, hi)` or error for inverted/NaN bounds.
fn evaluate_constant_clamp(x: f32, lo: f32, hi: f32) -> Result<f32, VerifyError> {
    // IEEE 754: `lo > hi` returns false when either is NaN. Check finiteness first.
    if !lo.is_finite() || !hi.is_finite() {
        return Err(VerifyError::InternalTranslationError {
            context: format!("Clamp bounds non-finite: lo={lo}, hi={hi}"),
        });
    }
    if lo > hi {
        return Err(VerifyError::InternalTranslationError {
            context: "Clamp bounds inverted".to_string(),
        });
    }
    Ok(x.clamp(lo, hi))
}

/// Translate a Clamp node with constant bounds to ClipLayer.
pub(crate) fn translate_clamp(
    name: &str,
    input_val: &NodeValue,
    min_val: &NodeValue,
    max_val: &NodeValue,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    match (min_val, max_val) {
        (NodeValue::Constant(lo), NodeValue::Constant(hi)) => {
            let (lo, hi) = (lo.get(), hi.get());
            match input_val {
                NodeValue::Constant(v) => {
                    let result = evaluate_constant_clamp(v.get(), lo, hi)?;
                    checked_constant(result, "Clamp constant fold")
                }
                NodeValue::Variable(var_name) => {
                    // IEEE 754: `lo > hi` returns false when either is NaN.
                    // Defense-in-depth: reject NaN bounds before NY ClipLayer.
                    if !lo.is_finite() || !hi.is_finite() {
                        return Err(VerifyError::InternalTranslationError {
                            context: format!(
                                "Clamp bounds non-finite: lo={lo}, hi={hi} in node `{name}`"
                            ),
                        });
                    }
                    if lo > hi {
                        return Err(VerifyError::InternalTranslationError {
                            context: format!(
                                "Clamp bounds inverted: lo={lo} > hi={hi} in node `{name}`"
                            ),
                        });
                    }
                    let layer = Layer::Clip(ClipLayer::new(lo, hi));
                    add_unary_node(name, layer, var_name, graph);
                    Ok(NodeValue::Variable(name.to_string()))
                }
            }
        }
        _ => Err(VerifyError::UnsupportedOp(
            "Clamp with variable bounds (constant min/max required)".into(),
        )),
    }
}
