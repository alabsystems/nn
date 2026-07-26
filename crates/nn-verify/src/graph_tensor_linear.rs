// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Linear (fully-connected) tensor-level IR → NY `LinearLayer` translation.
//!
//! Maps `TensorOpKind::Linear` to `Layer::Linear(LinearLayer)`, extracting
//! weight and bias tensors from `ConstantTensor` bindings.
//!
//! NY's `LinearLayer` is a unary operation: the weight matrix is fixed
//! (constant), and bounds propagate through the single variable input. This
//! matches the standard `y = Wx + b` fully-connected layer used throughout
//! dvoice (18 files, all 9 models).
//!
//! **Constant-fold path:** When the Linear input is itself a constant tensor
//! (e.g., KV projections in cross-attention where KV is `ConstantTensor`),
//! the result is computed eagerly: `output = input @ weight^T + bias`. The
//! result is returned as a `WeightTensor`, allowing downstream nodes (Reshape,
//! Transpose, Attention) to consume it. This enables single-variable
//! cross-attention verification where Q is Variable and K/V are constant.

use ny_propagate::layers::LinearLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;
use ndarray::Array1;

use super::TensorNodeValue;
use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

/// Translate a Linear tensor operation to a NY graph node.
///
/// The input data must be a `Variable` (the tensor being verified).
/// Weight must be a `WeightTensor` (fixed model parameters, shape `[out_features, in_features]`).
/// Bias (if present) must be a `WeightTensor` (shape `[out_features]`).
///
/// Creates a `Layer::Linear(LinearLayer)` node using NY's native
/// IBP/CROWN-aware linear layer, which analytically separates positive and
/// negative weight contributions for tight bounds.
pub(super) fn translate_linear(
    node_id: TensorNodeId,
    input: TensorNodeId,
    weight: TensorNodeId,
    bias: Option<&TensorNodeId>,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_val = get_value(node_values, input.index(), "Linear input")?;

    // Weight must be a WeightTensor (constant weight matrix).
    let weight_array_2d = match get_value(node_values, weight.index(), "Linear weight")? {
        TensorNodeValue::WeightTensor(arr) => {
            let shape = arr.shape();
            if shape.len() != 2 {
                return Err(VerifyError::WeightValidation {
                    op: "Linear",
                    reason: format!("weight must be 2-D, got {}-D", shape.len()),
                });
            }
            arr.clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| VerifyError::WeightValidation {
                    op: "Linear",
                    reason: format!("weight conversion to Array2 failed: {e}"),
                })?
        }
        _ => {
            return Err(VerifyError::WeightValidation {
                op: "Linear",
                reason: "weight must be a ConstantTensor binding".into(),
            });
        }
    };

    // Bias extraction (optional, shared by both paths).
    let bias_array: Option<Array1<f32>> = if let Some(bias_id) = bias {
        match get_value(node_values, bias_id.index(), "Linear bias")? {
            TensorNodeValue::WeightTensor(arr) => {
                let flat: Vec<f32> = arr.iter().copied().collect();
                Some(Array1::from_vec(flat))
            }
            _ => {
                return Err(VerifyError::WeightValidation {
                    op: "Linear",
                    reason: "bias must be a ConstantTensor binding".into(),
                });
            }
        }
    } else {
        None
    };

    match input_val {
        // Standard path: input is Variable, create NY LinearLayer node.
        TensorNodeValue::Variable(input_name) => {
            let linear_layer = LinearLayer::new(weight_array_2d, bias_array).map_err(|e| {
                VerifyError::UnsupportedOp(format!("LinearLayer construction failed: {e}"))
            })?;
            let node_name = format!("t{}", node_id.index());
            let layer = Layer::Linear(linear_layer);
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }

        // Constant-fold path: input is a constant tensor (e.g., KV in cross-attention).
        // Compute output = input @ weight^T + bias eagerly.
        TensorNodeValue::WeightTensor(input_arr) => {
            let input_2d = input_arr
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| {
                    VerifyError::UnsupportedOp(format!(
                        "Linear constant-fold: input must be 2-D, got: {e}"
                    ))
                })?;

            // output = input @ weight^T: [T, D_in] @ [D_in, D_out] = [T, D_out]
            let mut result = input_2d.dot(&weight_array_2d.t());

            // Add bias if present: broadcast [D_out] across rows.
            if let Some(ref b) = bias_array {
                result += b;
            }

            // Validate all output values are finite (checked_constant pattern).
            for &val in result.iter() {
                if !val.is_finite() {
                    return Err(VerifyError::NonFiniteConstant {
                        value: val,
                        context: format!("Linear constant-fold t{}", node_id.index()),
                    });
                }
            }

            Ok(TensorNodeValue::WeightTensor(result.into_dyn()))
        }

        // Scalar constant cannot be a Linear input.
        TensorNodeValue::Constant(_) => Err(VerifyError::UnsupportedOp(
            "Linear input must be a variable tensor, not a constant scalar".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array2, ArrayD, IxDyn};

    /// Helper: build a WeightTensor node value from a 2-D array.
    fn weight_tensor_2d(data: Array2<f32>) -> TensorNodeValue {
        TensorNodeValue::WeightTensor(data.into_dyn())
    }

    /// Helper: build a WeightTensor node value from a 1-D array.
    fn weight_tensor_1d(data: Array1<f32>) -> TensorNodeValue {
        TensorNodeValue::WeightTensor(data.into_dyn())
    }

    #[test]
    fn test_translate_linear_no_bias() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![
            TensorNodeValue::Variable("_input".to_string()),
            weight_tensor_2d(arr2(&[[1.0, -1.0], [0.5, 0.5]])),
        ];
        let result = translate_linear(
            TensorNodeId::new(2),
            TensorNodeId::new(0),
            TensorNodeId::new(1),
            None,
            &node_values,
            &mut graph,
        );
        let val = result.expect("translate_linear should succeed");
        let TensorNodeValue::Variable(name) = val else {
            panic!("expected Variable, got {val:?}");
        };
        assert_eq!(name, "t2");
    }

    #[test]
    fn test_translate_linear_with_bias() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![
            TensorNodeValue::Variable("_input".to_string()),
            weight_tensor_2d(arr2(&[[1.0, 0.0], [0.0, 1.0]])),
            weight_tensor_1d(arr1(&[0.1, -0.2])),
        ];
        let result = translate_linear(
            TensorNodeId::new(3),
            TensorNodeId::new(0),
            TensorNodeId::new(1),
            Some(&TensorNodeId::new(2)),
            &node_values,
            &mut graph,
        );
        let val = result.expect("translate_linear should succeed");
        let TensorNodeValue::Variable(name) = val else {
            panic!("expected Variable, got {val:?}");
        };
        assert_eq!(name, "t3");
    }

    #[test]
    fn test_translate_linear_rejects_constant_input() {
        use crate::graph::FiniteF32;
        let mut graph = GraphNetwork::new();
        let node_values = vec![
            TensorNodeValue::Constant(FiniteF32::new(1.0).unwrap()),
            weight_tensor_2d(arr2(&[[1.0]])),
        ];
        let result = translate_linear(
            TensorNodeId::new(2),
            TensorNodeId::new(0),
            TensorNodeId::new(1),
            None,
            &node_values,
            &mut graph,
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("constant scalar"), "error: {err}");
    }

    #[test]
    fn test_translate_linear_rejects_non_weight_weight() {
        let mut graph = GraphNetwork::new();
        let node_values = vec![
            TensorNodeValue::Variable("_input".to_string()),
            TensorNodeValue::Variable("bad_weight".to_string()),
        ];
        let result = translate_linear(
            TensorNodeId::new(2),
            TensorNodeId::new(0),
            TensorNodeId::new(1),
            None,
            &node_values,
            &mut graph,
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("ConstantTensor"), "error: {err}");
    }

    #[test]
    fn test_translate_linear_rejects_3d_weight() {
        let mut graph = GraphNetwork::new();
        let weight_3d = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0; 8]).unwrap();
        let node_values = vec![
            TensorNodeValue::Variable("_input".to_string()),
            TensorNodeValue::WeightTensor(weight_3d),
        ];
        let result = translate_linear(
            TensorNodeId::new(2),
            TensorNodeId::new(0),
            TensorNodeId::new(1),
            None,
            &node_values,
            &mut graph,
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("2-D"), "error: {err}");
    }
}
