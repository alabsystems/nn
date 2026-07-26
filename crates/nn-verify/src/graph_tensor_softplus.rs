// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Softplus tensor op translation to NY `SoftplusLayer`.
//!
//! Part of #834 — Gated DeltaNet gate computation pathway.

use ny_propagate::layers::SoftplusLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{checked_tensor_constant, TensorNodeValue};

/// Translate a `TensorOpKind::Softplus` node to a NY `SoftplusLayer`.
///
/// Constant-folds when the input is a known constant: `softplus(c) = ln(1 + exp(c))`.
/// For variable inputs, emits a `SoftplusLayer` node in the graph.
pub(crate) fn translate_softplus(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_val = get_value(node_values, input.index(), "Softplus input")?;
    match input_val {
        TensorNodeValue::Constant(c) => {
            let x = c.get();
            // Numerically stable: for large x, exp(x) overflows but softplus(x) ≈ x
            let val = if x > 20.0 { x } else { x.exp().ln_1p() };
            checked_tensor_constant(val, &format!("softplus constant fold t{}", node_id.index()))
        }
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            add_unary_node(
                &node_name,
                Layer::Softplus(SoftplusLayer),
                input_name,
                graph,
            );
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::WeightTensor(arr) => {
            let result = arr.mapv(|x| if x > 20.0 { x } else { x.exp().ln_1p() });
            for &val in result.iter() {
                if !val.is_finite() {
                    return Err(VerifyError::NonFiniteConstant {
                        value: val,
                        context: format!("softplus WeightTensor fold t{}", node_id.index()),
                    });
                }
            }
            Ok(TensorNodeValue::WeightTensor(result))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::FiniteF32;

    #[test]
    fn test_translate_softplus_variable() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Variable("x".to_string())];
        let result = translate_softplus(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("softplus variable should succeed");
        let TensorNodeValue::Variable(name) = val else {
            panic!("expected Variable, got {val:?}");
        };
        assert_eq!(name, "t1");
    }

    #[test]
    fn test_translate_softplus_constant_folds() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(0.0).unwrap())];
        let result = translate_softplus(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("constant fold should succeed");
        let TensorNodeValue::Constant(c) = val else {
            panic!("expected Constant, got {val:?}");
        };
        // softplus(0) = ln(1 + exp(0)) = ln(2) ≈ 0.6931
        let expected = 2.0_f32.ln();
        assert!((c.get() - expected).abs() < 1e-6, "got {}", c.get());
    }

    #[test]
    fn test_translate_softplus_constant_large_input() {
        let mut graph = GraphNetwork::new();
        // For large x, softplus(x) ≈ x
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(10.0).unwrap())];
        let result = translate_softplus(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("constant fold should succeed");
        let TensorNodeValue::Constant(c) = val else {
            panic!("expected Constant, got {val:?}");
        };
        assert!(
            (c.get() - 10.0).abs() < 0.001,
            "softplus(10) ≈ 10, got {}",
            c.get()
        );
    }

    #[test]
    fn test_translate_softplus_constant_negative_input() {
        let mut graph = GraphNetwork::new();
        // For large negative x, softplus(x) ≈ exp(x) ≈ 0
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(-10.0).unwrap())];
        let result = translate_softplus(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("constant fold should succeed");
        let TensorNodeValue::Constant(c) = val else {
            panic!("expected Constant, got {val:?}");
        };
        assert!(c.get() > 0.0, "softplus is always positive");
        assert!(c.get() < 0.001, "softplus(-10) ≈ 0, got {}", c.get());
    }

    #[test]
    fn test_translate_softplus_constant_very_large_no_overflow() {
        // Regression: softplus(100) used to overflow via exp(100) -> Inf
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(100.0).unwrap())];
        let result = translate_softplus(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("softplus(100) must not overflow");
        let TensorNodeValue::Constant(c) = val else {
            panic!("expected Constant, got {val:?}");
        };
        assert!(c.get().is_finite(), "softplus(100) must be finite");
        assert!(
            (c.get() - 100.0).abs() < 0.001,
            "softplus(100) ≈ 100, got {}",
            c.get()
        );
    }

    #[test]
    fn test_translate_softplus_weight_tensor_large_no_overflow() {
        // Regression: WeightTensor path also had overflow for large values
        let mut graph = GraphNetwork::new();
        let weight =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![100.0, 0.0]).unwrap();
        let node_values = vec![TensorNodeValue::WeightTensor(weight)];
        let result = translate_softplus(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("WeightTensor softplus must not overflow");
        let TensorNodeValue::WeightTensor(arr) = val else {
            panic!("expected WeightTensor, got {val:?}");
        };
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert!(vals[0].is_finite(), "softplus(100) must be finite");
        assert!((vals[0] - 100.0).abs() < 0.001, "softplus(100) ≈ 100");
    }

    #[test]
    fn test_translate_softplus_weight_tensor_folds() {
        let mut graph = GraphNetwork::new();
        let weight =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.0, -10.0]).unwrap();
        let node_values = vec![TensorNodeValue::WeightTensor(weight)];
        let result = translate_softplus(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("WeightTensor softplus should fold");
        let TensorNodeValue::WeightTensor(arr) = val else {
            panic!("expected WeightTensor, got {val:?}");
        };
        let vals: Vec<f32> = arr.iter().copied().collect();
        let expected_0 = 2.0_f32.ln(); // softplus(0) = ln(2)
        assert!((vals[0] - expected_0).abs() < 1e-5, "softplus(0) ≈ 0.693");
        assert!(vals[1] > 0.0, "softplus is always positive");
        assert!(vals[1] < 0.001, "softplus(-10) ≈ 0");
    }
}
