// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ZeroPad1d tensor-level IR → NY translation.
//!
//! ZeroPad1d adds known-zero elements to the last (time) axis. The padded
//! positions have exact bounds [0.0, 0.0] and original elements are unchanged.
//!
//! For NY propagation, we represent zero-padding as a `LinearLayer`
//! with weight `[T_out, T_in]` where `T_out = T_in + pad_left + pad_right`.
//! The weight is an identity matrix with zero rows at padding positions.
//! NY's N-D LinearLayer propagation applies this per-channel
//! automatically (last dim = `in_features`, all other dims = batch).
//!
//! For verification dimensions (T~4, pad~4), the matrix is small (~8×4).

use ny_propagate::layers::LinearLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;
use ndarray::Array2;

use super::{checked_tensor_constant, TensorNodeValue, TensorTranslationContext};
use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

/// Translate ZeroPad1d: for variable inputs, insert a LinearLayer that extends
/// the temporal dimension with zero-padded rows. For constants, pass through.
pub(super) fn translate_zero_pad_1d(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    pad_left: usize,
    pad_right: usize,
    _output_shape: &[usize],
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_val = get_value(node_values, input.index(), "ZeroPad1d input")?;
    match input_val {
        TensorNodeValue::Variable(var_name) => {
            let input_ir = ctx.all_nodes.get(input.index()).ok_or_else(|| {
                VerifyError::UnsupportedOp(format!(
                    "ZeroPad1d input node {} out of bounds",
                    input.index(),
                ))
            })?;
            let in_shape = &input_ir.shape;
            if in_shape.is_empty() {
                return Err(VerifyError::UnsupportedOp(
                    "ZeroPad1d requires at least 1D tensor".into(),
                ));
            }

            let t_in = *in_shape
                .last()
                .ok_or_else(|| VerifyError::InternalTranslationError {
                    context: "ZeroPad1d: empty shape".into(),
                })?;
            let t_out = t_in + pad_left + pad_right;

            // Build [T_out, T_in] weight: identity at rows [pad_left..pad_left+T_in],
            // zero rows at padding positions. NY's N-D LinearLayer applies
            // this per-channel (last dim = in_features, other dims = batch).
            let mut weight = Array2::<f32>::zeros((t_out, t_in));
            for t in 0..t_in {
                weight[[pad_left + t, t]] = 1.0;
            }

            let linear = LinearLayer::new(weight, None).map_err(|e| {
                VerifyError::UnsupportedOp(format!(
                    "ZeroPad1d LinearLayer construction failed: {e}"
                ))
            })?;

            let node_name = format!("t{}", node_id.index());
            add_unary_node(&node_name, Layer::Linear(linear), var_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::WeightTensor(_) => {
            // Weight tensors are constants — padding doesn't affect variable bounds.
            Ok(input_val.clone())
        }
        TensorNodeValue::Constant(val) => {
            checked_tensor_constant(val.get(), "ZeroPad1d constant input")
        }
    }
}
