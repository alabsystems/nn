// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LeakyReLU tensor op translation via decomposition: α*x + (1-α)*ReLU(x).
//!
//! Used by Kokoro decoder (ISTFTNet vocoder): `LeakyReLU(0.1)` per upsample
//! stage, `LeakyReLU(0.01)` before `conv_post`.
//!
//! Decomposed form avoids NY's `LeakyReLULayer` which returns
//! IBP-wide CROWN bounds (#2977). ReLU has correct CROWN linearization,
//! and MulConstant/Add are exact linear layers.
//!
//! Part of #1741 — extending moonshot proofs to real Kokoro model graph.

use ny_propagate::layers::{AddLayer, MulConstantLayer, ReLULayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{checked_tensor_constant, TensorNodeValue};

/// Translate a `TensorOpKind::LeakyRelu` node to a NY `LeakyReLULayer`.
///
/// Constant-folds when the input is a known constant:
/// `leaky_relu(c, alpha) = c if c >= 0, else alpha * c`.
/// For variable inputs, emits a `LeakyReLULayer` node in the graph.
pub(crate) fn translate_leaky_relu(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    negative_slope: f32,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    if !negative_slope.is_finite() {
        return Err(VerifyError::UnsupportedOp(format!(
            "LeakyRelu negative_slope is non-finite ({negative_slope})"
        )));
    }
    let input_val = get_value(node_values, input.index(), "LeakyRelu input")?;
    match input_val {
        TensorNodeValue::Constant(c) => {
            let val = if c.get() >= 0.0 {
                c.get()
            } else {
                negative_slope * c.get()
            };
            checked_tensor_constant(
                val,
                &format!("leaky_relu constant fold t{}", node_id.index()),
            )
        }
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            // Decompose LeakyReLU(x, alpha) = alpha*x + (1-alpha)*ReLU(x).
            // NY's LeakyReLULayer returns IBP-wide bounds. The
            // decomposed form uses ReLU (correct CROWN) + linear layers. (#2977)
            let scale_name = format!("{node_name}_alpha_x");
            add_unary_node(
                &scale_name,
                Layer::MulConstant(MulConstantLayer::scalar(negative_slope)),
                input_name,
                graph,
            );
            let relu_name = format!("{node_name}_relu");
            add_unary_node(&relu_name, Layer::ReLU(ReLULayer::new()), input_name, graph);
            let relu_scaled_name = format!("{node_name}_relu_scaled");
            add_unary_node(
                &relu_scaled_name,
                Layer::MulConstant(MulConstantLayer::scalar(1.0 - negative_slope)),
                &relu_name,
                graph,
            );
            graph.add_node(GraphNode::new(
                node_name.clone(),
                Layer::Add(AddLayer),
                vec![scale_name, relu_scaled_name],
            ));
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::WeightTensor(_) => Err(VerifyError::UnsupportedOp(
            "LeakyRelu on WeightTensor not supported".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::FiniteF32;

    #[test]
    fn test_translate_leaky_relu_variable() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Variable("x".to_string())];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            0.01,
            &node_values,
            &mut graph,
        );
        let val = result.expect("variable path should succeed");
        let TensorNodeValue::Variable(name) = val else {
            panic!("expected Variable, got {val:?}");
        };
        assert_eq!(name, "t1");
    }

    #[test]
    fn test_translate_leaky_relu_constant_positive() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(5.0).unwrap())];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            0.01,
            &node_values,
            &mut graph,
        );
        let val = result.expect("positive constant should succeed");
        let TensorNodeValue::Constant(c) = val else {
            panic!("expected Constant, got {val:?}");
        };
        assert!((c.get() - 5.0).abs() < 1e-6, "leaky_relu(5) = 5");
    }

    #[test]
    fn test_translate_leaky_relu_constant_negative() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(-5.0).unwrap())];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            0.1,
            &node_values,
            &mut graph,
        );
        let val = result.expect("negative constant should succeed");
        let TensorNodeValue::Constant(c) = val else {
            panic!("expected Constant, got {val:?}");
        };
        assert!(
            (c.get() - (-0.5)).abs() < 1e-6,
            "leaky_relu(-5, 0.1) = -0.5"
        );
    }

    #[test]
    fn test_translate_leaky_relu_nan_slope_rejected() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Variable("x".to_string())];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            f32::NAN,
            &node_values,
            &mut graph,
        );
        assert!(result.is_err(), "NaN slope must be rejected");
    }

    #[test]
    fn test_translate_leaky_relu_inf_slope_rejected() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Variable("x".to_string())];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            f32::INFINITY,
            &node_values,
            &mut graph,
        );
        assert!(result.is_err(), "Inf slope must be rejected");
    }

    #[test]
    fn test_translate_leaky_relu_weight_tensor_unsupported() {
        let mut graph = GraphNetwork::new();
        let weight =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, -1.0]).unwrap();
        let node_values = vec![TensorNodeValue::WeightTensor(weight)];
        let result = translate_leaky_relu(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            0.01,
            &node_values,
            &mut graph,
        );
        assert!(result.is_err(), "WeightTensor should be unsupported");
    }
}
