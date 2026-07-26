// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for normalization graph_tensor translations.
//!
//! Extracted from `graph_tensor_adain.rs`, `graph_tensor_rms_norm.rs`, and
//! `graph_tensor_instance_norm.rs` to deduplicate shape validation, eps
//! extraction, and parameter extraction patterns (#673).

use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::{StructuralError, VerifyError};
use crate::util::get_value;

use super::TensorNodeValue;

/// Validate normalization input shape (rank >= 2, last-axis) and extract eps value.
///
/// Returns `(input_val, eps_f32, input_shape)` on success.
/// `op_name` is used in error messages (e.g., "RmsNorm", "AdaIN1d").
pub(super) fn validate_norm_shape_and_eps<'a, 'b>(
    node_values: &'a [TensorNodeValue],
    all_nodes: &'b [nn_dsl::tensor_ir::TensorNode],
    input: &TensorNodeId,
    eps: &TensorNodeId,
    axis: usize,
    op_name: &str,
) -> Result<(&'a TensorNodeValue, f32, &'b [usize]), VerifyError> {
    let input_shape =
        &get_value(all_nodes, input.index(), &format!("{op_name} input shape"))?.shape;
    if input_shape.len() < 2 {
        return Err(StructuralError::ShapeConstraint {
            context: format!(
                "{op_name} requires rank >= 2 (got rank {}); \
                 need at least [channels, features] dimensions",
                input_shape.len()
            ),
        }
        .into());
    }
    if axis + 1 != input_shape.len() {
        return Err(StructuralError::ShapeConstraint {
            context: format!(
                "{op_name} axis {axis} is not the last axis (rank {}); \
                 NY only supports last-axis normalization",
                input_shape.len()
            ),
        }
        .into());
    }

    let input_val = get_value(node_values, input.index(), &format!("{op_name} input"))?;
    let eps_val = match get_value(node_values, eps.index(), &format!("{op_name} eps"))? {
        TensorNodeValue::Constant(v) => v.get(),
        // A single-element `WeightTensor` is a compile-time-known constant: the
        // documented eps contract is a "[1] (scalar constant)" tensor, and a
        // `ConstantTensor` binding becomes a `WeightTensor` only after every
        // element is verified finite (graph_tensor_helpers.rs:50-56). Reading
        // its lone element treats it as the fixed scalar it is — semantically
        // identical to the `Constant` path above, so this is sound. Multi-element
        // tensors and `Variable` eps (genuinely-unbounded/symbolic) stay rejected.
        TensorNodeValue::WeightTensor(arr) if arr.len() == 1 => {
            arr.iter().next().copied().expect("len()==1 has one element")
        }
        TensorNodeValue::Variable(_) | TensorNodeValue::WeightTensor(_) => {
            return Err(VerifyError::WeightValidation {
                op: "Normalization",
                reason: format!("{op_name} eps must be constant"),
            });
        }
    };

    Ok((input_val, eps_val, input_shape))
}

/// Extract a normalization parameter (gamma, beta, weight) from node values.
///
/// Supports `Constant` (broadcast to `expected_len`) and `WeightTensor` (validate
/// 1-D shape). Rejects `Variable` inputs since NY requires constant
/// affine parameters.
pub(super) fn extract_norm_param(
    node_values: &[TensorNodeValue],
    param_id: &TensorNodeId,
    expected_len: usize,
    op_name: &str,
    param_name: &str,
) -> Result<ndarray::Array1<f32>, VerifyError> {
    match get_value(
        node_values,
        param_id.index(),
        &format!("{op_name} {param_name}"),
    )? {
        TensorNodeValue::Constant(v) => Ok(ndarray::Array1::from_elem(expected_len, v.get())),
        TensorNodeValue::WeightTensor(arr) => {
            if arr.ndim() != 1 || arr.len() != expected_len {
                return Err(VerifyError::WeightValidation {
                    op: "Normalization",
                    reason: format!(
                        "{op_name} {param_name} shape mismatch: expected [{expected_len}], got {:?}",
                        arr.shape()
                    ),
                });
            }
            arr.to_owned()
                .into_dimensionality()
                .map_err(|e| VerifyError::WeightValidation {
                    op: "Normalization",
                    reason: format!("{op_name} {param_name} reshape: {e}"),
                })
        }
        TensorNodeValue::Variable(_) => Err(VerifyError::WeightValidation {
            op: "Normalization",
            reason: format!("{op_name} {param_name} must be constant or weight tensor"),
        }),
    }
}
