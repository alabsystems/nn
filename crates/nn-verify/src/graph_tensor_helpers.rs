// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Helper functions for tensor dispatch translation.
//!
//! Extracted from `graph_tensor_dispatch.rs` to stay within the 500-line
//! file limit. Contains `translate_input` and `translate_broadcast`.

use ny_propagate::layers::{ReshapeLayer, TileLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::tensor_ir::{BroadcastAlignment, TensorNodeId};
use ndarray::IxDyn;

use super::super::{
    axis_as_i32, dim_as_i64, TensorNodeValue, TensorParamBinding, TensorTranslationContext,
};
use crate::error::{StructuralError, VerifyError};
use crate::graph::FiniteF32;
use crate::util::get_value;

/// Handle `TensorOpKind::Input` — extracted from the dispatch match for clarity.
pub(crate) fn translate_input(
    ctx: &TensorTranslationContext<'_>,
    node_values: &[TensorNodeValue],
    input_idx: &mut usize,
) -> Result<TensorNodeValue, VerifyError> {
    let _ = node_values; // present for consistency with other translate fns
    let current_idx = *input_idx;
    let binding = &ctx.input_bindings[current_idx];
    let node_name = &ctx.input_node_names[current_idx];
    *input_idx += 1;
    match binding {
        TensorParamBinding::Variable => Ok(TensorNodeValue::Variable(
            node_name
                .as_ref()
                .ok_or(VerifyError::from(StructuralError::MissingNodeName {
                    input_idx: current_idx,
                }))?
                .clone(),
        )),
        TensorParamBinding::ConstantScalar(val) => {
            let finite = FiniteF32::new(*val).map_err(|_| VerifyError::NonFiniteConstant {
                value: *val,
                context: format!("tensor input binding {current_idx}"),
            })?;
            Ok(TensorNodeValue::Constant(finite))
        }
        TensorParamBinding::ConstantTensor(arr) => {
            for (idx, &val) in arr.iter().enumerate() {
                if !val.is_finite() {
                    return Err(VerifyError::NonFiniteConstant {
                        value: val,
                        context: format!("tensor input binding {current_idx} element {idx}"),
                    });
                }
            }
            Ok(TensorNodeValue::WeightTensor(arr.clone()))
        }
    }
}

/// Handle `TensorOpKind::Broadcast` — reshape `WeightTensor` for left-aligned broadcast,
/// and insert `TileLayer` graph nodes for `Variable` broadcasts that expand dimensions.
///
/// ndarray uses right-aligned (NumPy-style) broadcasting. When the tensor IR requests
/// left-aligned broadcast (e.g., `[C]` → `[C, T]`), we must reshape the weight array
/// by appending trailing size-1 dimensions so ndarray can broadcast correctly.
/// For right-aligned broadcast, no reshape is needed.
///
/// For `Variable` nodes: NY does NOT auto-broadcast during IBP/CROWN
/// propagation. When a Variable input has shape `[1, D]` and the target is `[T, D]`,
/// we insert `TileLayer(axis=0, reps=T)` nodes to explicitly expand the dimensions.
/// Without this, shape mismatches occur during bound propagation.
///
/// `Constant` scalar nodes pass through unchanged — they are broadcast naturally by
/// NY arithmetic layers.
pub(crate) fn translate_broadcast(
    node_id: TensorNodeId,
    input: &TensorNodeId,
    target_shape: &[usize],
    alignment: BroadcastAlignment,
    node_values: &[TensorNodeValue],
    ctx: &TensorTranslationContext<'_>,
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let value = get_value(node_values, input.index(), "Broadcast input")?;
    match value {
        TensorNodeValue::WeightTensor(arr) => {
            let src_ndim = arr.ndim();
            let tgt_ndim = target_shape.len();
            if src_ndim == tgt_ndim {
                // Already same rank — no reshape needed.
                return Ok(value.clone());
            }
            match alignment {
                BroadcastAlignment::Left => {
                    // Append trailing size-1 dims: [C] → [C, 1] for target [C, T].
                    let mut new_shape = arr.shape().to_vec();
                    new_shape.resize(tgt_ndim, 1);
                    let reshaped = arr
                        .clone()
                        .into_shape_with_order(IxDyn(&new_shape))
                        .map_err(|e| {
                            VerifyError::UnsupportedOp(format!(
                                "Broadcast left reshape {:?} → {:?}: {e}",
                                arr.shape(),
                                new_shape,
                            ))
                        })?;
                    Ok(TensorNodeValue::WeightTensor(reshaped))
                }
                BroadcastAlignment::Right => {
                    // ndarray uses right-aligned broadcast by default — no reshape needed.
                    Ok(value.clone())
                }
                // SAFETY: BroadcastAlignment is #[non_exhaustive] from nn-dsl.
                // New variants require explicit handling; returning an error is
                // conservative (blocks unsupported ops rather than silently passing).
                _ => Err(VerifyError::UnsupportedOp(format!(
                    "unsupported BroadcastAlignment variant: {alignment:?}"
                ))),
            }
        }
        TensorNodeValue::Variable(var_name) => {
            // Look up the source shape from the IR node that produced this value.
            let src_shape = ctx
                .all_nodes
                .get(input.index())
                .map(|n| n.shape.as_slice())
                .ok_or_else(|| VerifyError::InternalTranslationError {
                    context: format!(
                        "Broadcast: input node {} out of bounds (len {})",
                        input.index(),
                        ctx.all_nodes.len()
                    ),
                })?;

            let src_ndim = src_shape.len();
            let tgt_ndim = target_shape.len();

            // When ranks differ, insert a ReshapeLayer to match target rank
            // before applying tile operations. For BroadcastLeft, append trailing
            // size-1 dims: [C] → [C, 1]. For BroadcastRight, prepend leading
            // size-1 dims: [T] → [1, T]. This matches the WeightTensor path.
            let mut current_name = var_name.clone();
            let effective_src_shape: Vec<usize> = if src_ndim < tgt_ndim {
                let reshaped: Vec<usize> = match alignment {
                    BroadcastAlignment::Left => {
                        let mut s = src_shape.to_vec();
                        s.resize(tgt_ndim, 1);
                        s
                    }
                    _ => {
                        let pad = tgt_ndim - src_ndim;
                        let mut s = vec![1; pad];
                        s.extend_from_slice(src_shape);
                        s
                    }
                };
                let reshape_name = format!("t{}_broadcast_reshape", node_id.index());
                let reshape_dims: Vec<i64> = reshaped
                    .iter()
                    .map(|&d| dim_as_i64(d, "Broadcast reshape"))
                    .collect::<Result<_, _>>()?;
                let layer = Layer::Reshape(ReshapeLayer::new(reshape_dims));
                graph.add_node(GraphNode::new(
                    reshape_name.clone(),
                    layer,
                    vec![current_name],
                ));
                current_name = reshape_name;
                reshaped
            } else {
                src_shape.to_vec()
            };

            // Insert TileLayer nodes for each axis where src_dim == 1 and
            // target_dim > 1 (dimension expansion via tiling).
            for (axis, (&src_dim, &tgt_dim)) in effective_src_shape
                .iter()
                .zip(target_shape.iter())
                .enumerate()
            {
                if src_dim == 1 && tgt_dim > 1 {
                    let tile_name = format!("t{}_broadcast_tile_ax{}", node_id.index(), axis);
                    let axis_i32 = axis_as_i32(axis, "Broadcast TileLayer")?;
                    let layer = Layer::Tile(TileLayer::new(axis_i32, tgt_dim));
                    graph.add_node(GraphNode::new(tile_name.clone(), layer, vec![current_name]));
                    current_name = tile_name;
                }
            }
            Ok(TensorNodeValue::Variable(current_name))
        }
        // Constant scalars pass through — broadcast naturally by NY
        // arithmetic layers (AddConstant, MulConstant).
        TensorNodeValue::Constant(_) => Ok(value.clone()),
    }
}
