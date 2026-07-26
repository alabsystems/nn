// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Binary tensor op translations: BinaryAdd → AddLayer, BinaryMul → MulBinaryLayer.
//!
//! Extracted from `graph_tensor.rs` to stay under the 500-line file limit (#640).

use ny_propagate::layers::{AddConstantLayer, AddLayer, MulBinaryLayer, MulConstantLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::tensor_ir::TensorNodeId;
use ndarray::ArrayD;

use crate::error::VerifyError;
use crate::graph::{scalar_array, FiniteF32};
use crate::util::get_value;

use super::TensorNodeValue;

/// Validate that all elements in a weight tensor are finite.
fn validate_weight_finite(arr: &ArrayD<f32>, context: &str) -> Result<(), VerifyError> {
    if let Some(v) = arr.iter().find(|v| !v.is_finite()) {
        return Err(VerifyError::NonFiniteConstant {
            value: *v,
            context: context.to_string(),
        });
    }
    Ok(())
}

/// Translate `BinaryAdd` to NY graph: handles all combinations of
/// Variable/Constant inputs. Two Variables produce a binary `AddLayer` node;
/// mixed cases use `AddConstantLayer`; two constants fold to a new constant.
pub(super) fn translate_binary_add(
    node_id: TensorNodeId,
    left_id: TensorNodeId,
    right_id: TensorNodeId,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let left = get_value(node_values, left_id.index(), "BinaryAdd left")?;
    let right = get_value(node_values, right_id.index(), "BinaryAdd right")?;
    let name = format!("t{}_binary_add", node_id.index());

    match (left, right) {
        (TensorNodeValue::Variable(lhs_name), TensorNodeValue::Variable(rhs_name)) => {
            graph.add_node(GraphNode::binary(
                name.clone(),
                Layer::Add(AddLayer),
                lhs_name.clone(),
                rhs_name.clone(),
            ));
            Ok(TensorNodeValue::Variable(name))
        }
        (TensorNodeValue::Variable(var_name), TensorNodeValue::Constant(c))
        | (TensorNodeValue::Constant(c), TensorNodeValue::Variable(var_name)) => {
            let layer = Layer::AddConstant(AddConstantLayer::new(scalar_array(c.get())?));
            graph.add_node(GraphNode::new(name.clone(), layer, vec![var_name.clone()]));
            Ok(TensorNodeValue::Variable(name))
        }
        (TensorNodeValue::Constant(a), TensorNodeValue::Constant(b)) => {
            let sum = a.get() + b.get();
            let finite = FiniteF32::new(sum).map_err(|_| VerifyError::NonFiniteConstant {
                value: sum,
                context: format!("BinaryAdd constant fold {a:?} + {b:?}"),
            })?;
            Ok(TensorNodeValue::Constant(finite))
        }
        // Variable + WeightTensor: use AddConstantLayer with the full weight array.
        // NY handles broadcasting in IBP propagation.
        (TensorNodeValue::Variable(var_name), TensorNodeValue::WeightTensor(arr))
        | (TensorNodeValue::WeightTensor(arr), TensorNodeValue::Variable(var_name)) => {
            validate_weight_finite(arr, "BinaryAdd weight tensor")?;
            let layer = Layer::AddConstant(AddConstantLayer::new(arr.clone()));
            graph.add_node(GraphNode::new(name.clone(), layer, vec![var_name.clone()]));
            Ok(TensorNodeValue::Variable(name))
        }
        // WeightTensor + WeightTensor: element-wise constant fold via ndarray.
        (TensorNodeValue::WeightTensor(lhs), TensorNodeValue::WeightTensor(rhs)) => {
            validate_weight_finite(lhs, "BinaryAdd left weight")?;
            validate_weight_finite(rhs, "BinaryAdd right weight")?;
            let sum = lhs + rhs;
            validate_weight_finite(&sum, "BinaryAdd weight fold")?;
            Ok(TensorNodeValue::WeightTensor(sum))
        }
        // Constant + WeightTensor: add scalar to all elements.
        (TensorNodeValue::Constant(c), TensorNodeValue::WeightTensor(arr))
        | (TensorNodeValue::WeightTensor(arr), TensorNodeValue::Constant(c)) => {
            validate_weight_finite(arr, "BinaryAdd weight tensor")?;
            let sum = arr.mapv(|v| v + c.get());
            validate_weight_finite(&sum, "BinaryAdd scalar+weight fold")?;
            Ok(TensorNodeValue::WeightTensor(sum))
        }
    }
}

/// Translate `BinaryMul` to NY graph: handles all combinations of
/// Variable/Constant/WeightTensor inputs. Two Variables produce a binary
/// `MulBinaryLayer` node; mixed cases use `MulConstantLayer`; two constants
/// fold to a new constant. WeightTensor operands are treated as constant
/// arrays for `MulConstantLayer`.
pub(super) fn translate_binary_mul(
    node_id: TensorNodeId,
    left_id: TensorNodeId,
    right_id: TensorNodeId,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let left = get_value(node_values, left_id.index(), "BinaryMul left")?;
    let right = get_value(node_values, right_id.index(), "BinaryMul right")?;
    let name = format!("t{}_binary_mul", node_id.index());

    match (left, right) {
        (TensorNodeValue::Variable(lhs_name), TensorNodeValue::Variable(rhs_name)) => {
            graph.add_node(GraphNode::binary(
                name.clone(),
                Layer::MulBinary(MulBinaryLayer),
                lhs_name.clone(),
                rhs_name.clone(),
            ));
            Ok(TensorNodeValue::Variable(name))
        }
        (TensorNodeValue::Variable(var_name), TensorNodeValue::Constant(c))
        | (TensorNodeValue::Constant(c), TensorNodeValue::Variable(var_name)) => {
            let layer = Layer::MulConstant(MulConstantLayer::new(scalar_array(c.get())?));
            graph.add_node(GraphNode::new(name.clone(), layer, vec![var_name.clone()]));
            Ok(TensorNodeValue::Variable(name))
        }
        (TensorNodeValue::Constant(a), TensorNodeValue::Constant(b)) => {
            let product = a.get() * b.get();
            let finite = FiniteF32::new(product).map_err(|_| VerifyError::NonFiniteConstant {
                value: product,
                context: format!("BinaryMul constant fold {a:?} * {b:?}"),
            })?;
            Ok(TensorNodeValue::Constant(finite))
        }
        // Variable * WeightTensor: use MulConstantLayer with the full weight array.
        // NY handles broadcasting in IBP propagation. This enables
        // per-channel affine (gamma * x) in GroupNorm with ConstantTensor bindings.
        (TensorNodeValue::Variable(var_name), TensorNodeValue::WeightTensor(arr))
        | (TensorNodeValue::WeightTensor(arr), TensorNodeValue::Variable(var_name)) => {
            validate_weight_finite(arr, "BinaryMul weight tensor")?;
            let layer = Layer::MulConstant(MulConstantLayer::new(arr.clone()));
            graph.add_node(GraphNode::new(name.clone(), layer, vec![var_name.clone()]));
            Ok(TensorNodeValue::Variable(name))
        }
        // WeightTensor * WeightTensor: element-wise constant fold via ndarray.
        (TensorNodeValue::WeightTensor(lhs), TensorNodeValue::WeightTensor(rhs)) => {
            validate_weight_finite(lhs, "BinaryMul left weight")?;
            validate_weight_finite(rhs, "BinaryMul right weight")?;
            let product = lhs * rhs;
            validate_weight_finite(&product, "BinaryMul weight fold")?;
            Ok(TensorNodeValue::WeightTensor(product))
        }
        // Constant * WeightTensor: scale all elements by the scalar.
        (TensorNodeValue::Constant(c), TensorNodeValue::WeightTensor(arr))
        | (TensorNodeValue::WeightTensor(arr), TensorNodeValue::Constant(c)) => {
            validate_weight_finite(arr, "BinaryMul weight tensor")?;
            let product = arr.mapv(|v| v * c.get());
            validate_weight_finite(&product, "BinaryMul scalar*weight fold")?;
            Ok(TensorNodeValue::WeightTensor(product))
        }
    }
}
