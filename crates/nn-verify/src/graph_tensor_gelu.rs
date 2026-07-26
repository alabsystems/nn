// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GELU tensor op translation: TensorOpKind::Gelu/GeluErf → NY GELULayer.
//!
//! Extracted to its own file following the per-op module pattern (#640).
//! GELU maps directly to NY's native GELULayer (tanh or erf
//! approximation, sound mode) for tighter bounds than the decomposed scalar
//! Elementwise path.

use ny_propagate::layers::{GELULayer, GeluApproximation};
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::TensorNodeValue;

/// Translate `TensorOpKind::Gelu` to a NY `GELULayer` node.
///
/// Variable inputs produce a GELULayer node (using `add_unary_node` to handle
/// the `NETWORK_INPUT` special case). Constant inputs are evaluated directly
/// using the exp-based form that matches the scalar reference and MSL emission.
pub(super) fn translate_gelu(
    node_id: TensorNodeId,
    input_id: &TensorNodeId,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input = get_value(node_values, input_id.index(), "Gelu input")?;
    let node_name = format!("t{}_gelu", node_id.index());

    match input {
        TensorNodeValue::Variable(input_name) => {
            let layer = Layer::GELU(GELULayer::new(GeluApproximation::Tanh));
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::Constant(c) => {
            // Constant-fold: evaluate GELU(c) via exp form (matches scalar
            // reference in gelu.rs and MSL emission, #679).
            let x = f64::from(c.get());
            let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
            let e2 = (2.0 * inner).exp();
            let result = 0.5 * x * (2.0 - 2.0 / (e2 + 1.0));
            super::checked_tensor_constant(
                result as f32,
                &format!("Gelu constant fold gelu({c:?})"),
            )
        }
        TensorNodeValue::WeightTensor(_) => Err(VerifyError::UnsupportedOp(
            "Gelu does not support weight tensor operands".into(),
        )),
    }
}

/// Translate `TensorOpKind::GeluErf` to a NY `GELULayer` node (exact erf).
///
/// Same structure as `translate_gelu` but uses `GeluApproximation::Erf`.
/// Constant inputs use the exact erf formula for folding.
pub(super) fn translate_gelu_erf(
    node_id: TensorNodeId,
    input_id: &TensorNodeId,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input = get_value(node_values, input_id.index(), "GeluErf input")?;
    let node_name = format!("t{}_gelu_erf", node_id.index());

    match input {
        TensorNodeValue::Variable(input_name) => {
            let layer = Layer::GELU(GELULayer::new(GeluApproximation::Erf));
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::Constant(c) => {
            // Constant-fold: evaluate GeluErf(c) via erf form.
            // gelu_erf(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            let x = f64::from(c.get());
            let erf_val = erf_f64(x * std::f64::consts::FRAC_1_SQRT_2);
            let result = 0.5 * x * (1.0 + erf_val);
            super::checked_tensor_constant(
                result as f32,
                &format!("GeluErf constant fold gelu_erf({c:?})"),
            )
        }
        TensorNodeValue::WeightTensor(_) => Err(VerifyError::UnsupportedOp(
            "GeluErf does not support weight tensor operands".into(),
        )),
    }
}

/// Abramowitz & Stegun approximation of erf(x) in f64, max error ~1.5e-7.
///
/// Same algorithm as `erf_f32` in nn-core (formula 7.1.26) but computed in
/// f64 for constant-fold precision. Avoids adding a `libm` dependency.
fn erf_f64(x: f64) -> f64 {
    let a1: f64 = 0.254_829_592;
    let a2: f64 = -0.284_496_736;
    let a3: f64 = 1.421_413_741;
    let a4: f64 = -1.453_152_027;
    let a5: f64 = 1.061_405_429;
    let p: f64 = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}
