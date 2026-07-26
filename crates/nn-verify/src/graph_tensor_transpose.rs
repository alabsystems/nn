// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Transpose tensor op translation: axis permutation → `TransposeLayer`.
//!
//! Maps `TensorOpKind::Transpose { input, axes }` to NY's
//! `Layer::Transpose(TransposeLayer::new(adjusted_axes))`.
//!
//! When multi-variable stacking adds a leading dimension (`axis_offset=1`),
//! all axes in the permutation are incremented by 1 and the leading stacking
//! axis (0) is prepended. For example, axes `[1, 0, 2]` becomes `[0, 2, 1, 3]`.
//!
//! Part of #809.

use ny_propagate::layers::TransposeLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{TensorNodeValue, TensorTranslationContext};

/// Translate `Transpose` — permute tensor dimensions.
/// Constant inputs pass through unchanged (same scalar, different shape).
pub(super) fn translate_transpose(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    axes: &[usize],
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    match get_value(node_values, input.index(), "Transpose input")? {
        TensorNodeValue::Constant(val) => Ok(TensorNodeValue::Constant(*val)),
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            // Adjust axes for multi-variable stacking: prepend axis 0 (stacking dim)
            // and increment all user axes by axis_offset.
            let adjusted_axes: Vec<usize> = if ctx.axis_offset > 0 {
                let mut adj = vec![0usize]; // preserve leading stacking dim
                adj.extend(axes.iter().map(|&a| a + ctx.axis_offset));
                adj
            } else {
                axes.to_vec()
            };
            let layer = Layer::Transpose(TransposeLayer::new(adjusted_axes));
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        // Constant-fold: permute the ndarray axes directly (e.g., KV projections
        // in cross-attention where reshaping+transposing a constant tensor).
        TensorNodeValue::WeightTensor(arr) => {
            let permuted = arr.clone().permuted_axes(axes.to_vec());
            // Make contiguous after permutation (into_dyn preserves data order).
            let contiguous = permuted.as_standard_layout().into_owned().into_dyn();
            Ok(TensorNodeValue::WeightTensor(contiguous))
        }
    }
}
