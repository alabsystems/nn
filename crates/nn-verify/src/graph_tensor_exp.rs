// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exp tensor op translation to NY `ExpLayer`.
//!
//! Part of #834 — Gated DeltaNet decay gate computation.

use ny_propagate::layers::ExpLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{checked_tensor_constant, TensorNodeValue};

/// Translate a `TensorOpKind::Exp` node to a NY `ExpLayer`.
///
/// Constant-folds when the input is a known constant: `exp(c)`.
/// For variable inputs, emits an `ExpLayer` node in the graph.
pub(crate) fn translate_exp(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_val = get_value(node_values, input.index(), "Exp input")?;
    match input_val {
        TensorNodeValue::Constant(c) => {
            let val = c.get().exp();
            checked_tensor_constant(val, &format!("exp constant fold t{}", node_id.index()))
        }
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            add_unary_node(&node_name, Layer::Exp(ExpLayer), input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::WeightTensor(arr) => {
            let result = arr.mapv(f32::exp);
            for &val in result.iter() {
                if !val.is_finite() {
                    return Err(VerifyError::NonFiniteConstant {
                        value: val,
                        context: format!("exp WeightTensor fold t{}", node_id.index()),
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
    fn test_translate_exp_variable() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Variable("x".to_string())];
        let result = translate_exp(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("exp variable should succeed");
        let TensorNodeValue::Variable(name) = val else {
            panic!("expected Variable, got {val:?}");
        };
        assert_eq!(name, "t1");
    }

    #[test]
    fn test_translate_exp_constant_zero() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(0.0).unwrap())];
        let result = translate_exp(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("constant fold should succeed");
        let TensorNodeValue::Constant(c) = val else {
            panic!("expected Constant, got {val:?}");
        };
        assert!((c.get() - 1.0).abs() < 1e-6, "exp(0) = 1, got {}", c.get());
    }

    #[test]
    fn test_translate_exp_constant_one() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(1.0).unwrap())];
        let result = translate_exp(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("constant fold should succeed");
        let TensorNodeValue::Constant(c) = val else {
            panic!("expected Constant, got {val:?}");
        };
        let expected = 1.0_f32.exp();
        assert!(
            (c.get() - expected).abs() < 1e-5,
            "exp(1) ≈ {expected}, got {}",
            c.get()
        );
    }

    #[test]
    fn test_translate_exp_constant_negative() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(-2.0).unwrap())];
        let result = translate_exp(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("constant fold should succeed");
        let TensorNodeValue::Constant(c) = val else {
            panic!("expected Constant, got {val:?}");
        };
        let expected = (-2.0_f32).exp();
        assert!(
            (c.get() - expected).abs() < 1e-6,
            "exp(-2) ≈ {expected}, got {}",
            c.get()
        );
    }

    #[test]
    fn test_translate_exp_large_input_rejects_inf() {
        let mut graph = GraphNetwork::new();
        // exp(1000) = +inf, which should be rejected by checked_tensor_constant
        let node_values = vec![TensorNodeValue::Constant(FiniteF32::new(1000.0).unwrap())];
        let result = translate_exp(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        assert!(
            result.is_err(),
            "exp(1000) overflows to inf and must be rejected"
        );
    }

    #[test]
    fn test_translate_exp_weight_tensor_folds() {
        let mut graph = GraphNetwork::new();
        let weight =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, -1.0]).unwrap();
        let node_values = vec![TensorNodeValue::WeightTensor(weight)];
        let result = translate_exp(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        let val = result.expect("WeightTensor exp should fold");
        let TensorNodeValue::WeightTensor(arr) = val else {
            panic!("expected WeightTensor, got {val:?}");
        };
        let vals: Vec<f32> = arr.iter().copied().collect();
        assert!((vals[0] - 1.0_f32.exp()).abs() < 1e-5, "exp(1) ≈ 2.718");
        assert!((vals[1] - (-1.0_f32).exp()).abs() < 1e-5, "exp(-1) ≈ 0.368");
    }

    #[test]
    fn test_translate_exp_weight_tensor_rejects_overflow() {
        let mut graph = GraphNetwork::new();
        let weight =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 1000.0]).unwrap();
        let node_values = vec![TensorNodeValue::WeightTensor(weight)];
        let result = translate_exp(
            TensorNodeId::new(1),
            &TensorNodeId::new(0),
            &node_values,
            &mut graph,
        );
        assert!(result.is_err(), "exp(1000) overflows to inf");
    }
}
