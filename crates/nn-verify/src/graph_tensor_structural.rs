// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural tensor op translations: Reshape, AxisSelect, Stack.
//!
//! Extracted from `graph_tensor.rs` to stay under the 500-line file limit.

use ny_propagate::layers::{
    ConcatLayer, LinearLayer, ReshapeLayer, SliceLayer, SqueezeLayer, UnsqueezeLayer,
};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::tensor_ir::TensorNodeId;
use ndarray::{Array1, Array2, ArrayD};

use crate::error::VerifyError;
use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{axis_as_i32, dim_as_i64, TensorNodeValue, TensorTranslationContext};

/// Translate `Reshape` — shape change with no data reordering.
/// Constant inputs pass through unchanged (same scalar, different shape).
pub(super) fn translate_reshape(
    _ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    target_shape: &[usize],
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    match get_value(node_values, input.index(), "Reshape input")? {
        TensorNodeValue::Constant(val) => Ok(TensorNodeValue::Constant(*val)),
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            // Each variable enters at its TRUE rank (see `setup_multi_variable_inputs`),
            // so the user-declared target shape is used directly with no leading
            // stacking dimension.
            let adjusted: Vec<i64> = target_shape
                .iter()
                .map(|&d| dim_as_i64(d, "Reshape target_shape"))
                .collect::<Result<_, _>>()?;
            let layer = Layer::Reshape(ReshapeLayer::new(adjusted));
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        // Constant-fold: reshape the ndarray directly (e.g., KV projections in
        // cross-attention where KV Linear output is a constant WeightTensor).
        TensorNodeValue::WeightTensor(arr) => {
            let new_shape: Vec<usize> = target_shape.to_vec();
            let reshaped = arr
                .clone()
                .into_shape_with_order(ndarray::IxDyn(&new_shape))
                .map_err(|e| VerifyError::InternalTranslationError {
                    context: format!("Reshape constant-fold failed: {e}"),
                })?;
            Ok(TensorNodeValue::WeightTensor(reshaped))
        }
    }
}

/// Translate `AxisSelect` — select a single index along an axis.
/// Decomposed as `SliceLayer(axis, index, index+1)` + `SqueezeLayer(axis)`.
pub(super) fn translate_axis_select(
    _ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    axis: usize,
    index: usize,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    match get_value(node_values, input.index(), "AxisSelect input")? {
        TensorNodeValue::Constant(val) => Ok(TensorNodeValue::Constant(*val)),
        TensorNodeValue::Variable(input_name) => {
            let adjusted_axis = axis_as_i32(axis, "AxisSelect axis")?;
            let slice_name = format!("t{}_slice", node_id.index());
            let squeeze_name = format!("t{}", node_id.index());
            let slice_layer = Layer::Slice(SliceLayer::new(adjusted_axis, index, index + 1));
            add_unary_node(&slice_name, slice_layer, input_name, graph);
            let squeeze_layer = Layer::Squeeze(SqueezeLayer::new(adjusted_axis));
            add_unary_node(&squeeze_name, squeeze_layer, &slice_name, graph);
            Ok(TensorNodeValue::Variable(squeeze_name))
        }
        TensorNodeValue::WeightTensor(_) => Err(VerifyError::UnsupportedOp(
            "weight tensor cannot be used as AxisSelect input".into(),
        )),
    }
}

/// Translate `Narrow` — extract a contiguous slice along one axis.
/// Maps directly to `SliceLayer(axis, start, start+length)`. Unlike AxisSelect,
/// no SqueezeLayer is needed because the axis dimension is preserved.
pub(super) fn translate_narrow(
    _ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    input: &TensorNodeId,
    axis: usize,
    start: usize,
    length: usize,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    match get_value(node_values, input.index(), "Narrow input")? {
        TensorNodeValue::Constant(val) => Ok(TensorNodeValue::Constant(*val)),
        TensorNodeValue::Variable(input_name) => {
            let adjusted_axis = axis_as_i32(axis, "Narrow axis")?;
            let node_name = format!("t{}", node_id.index());
            let layer = Layer::Slice(SliceLayer::new(adjusted_axis, start, start + length));
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::WeightTensor(arr) => {
            // Constant-fold: slice the ndarray along the given axis.
            let sliced = arr
                .slice_axis(
                    ndarray::Axis(axis),
                    ndarray::Slice::from(start..start + length),
                )
                .to_owned();
            Ok(TensorNodeValue::WeightTensor(sliced))
        }
    }
}

/// Translate `Stack` — join tensors along a new axis.
/// Each input gets `UnsqueezeLayer(axis)`, then pairwise `ConcatLayer(axis)`.
pub(super) fn translate_stack(
    _ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    inputs: &[TensorNodeId],
    axis: usize,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    if inputs.is_empty() {
        return Err(VerifyError::UnsupportedOp("Stack with 0 inputs".into()));
    }

    // Reject constant inputs early (#270): the pairwise concat loop cannot
    // meaningfully concat a constant with a variable graph node. Constant
    // Stack inputs would be silently dropped by the `continue` below.
    for (i, tid) in inputs.iter().enumerate() {
        if matches!(
            get_value(node_values, tid.index(), "Stack constant check")?,
            TensorNodeValue::Constant(_) | TensorNodeValue::WeightTensor(_)
        ) {
            return Err(VerifyError::UnsupportedOp(format!(
                "Stack input {i} is constant/weight — constant inputs in Stack are not supported"
            )));
        }
    }

    let axis_i32 = axis_as_i32(axis, "Stack axis")?;
    let axis_i64 = dim_as_i64(axis, "Stack axis")?;

    // Unsqueeze each input to insert the new axis dimension.
    let mut unsqueezed: Vec<TensorNodeValue> = Vec::with_capacity(inputs.len());
    for (i, tid) in inputs.iter().enumerate() {
        match get_value(node_values, tid.index(), "Stack unsqueeze input")? {
            TensorNodeValue::Constant(val) => unsqueezed.push(TensorNodeValue::Constant(*val)),
            TensorNodeValue::Variable(input_name) => {
                let name = format!("t{}_unsq{i}", node_id.index());
                let layer = Layer::Unsqueeze(UnsqueezeLayer::new(axis_i32));
                add_unary_node(&name, layer, input_name, graph);
                unsqueezed.push(TensorNodeValue::Variable(name));
            }
            TensorNodeValue::WeightTensor(_) => {
                return Err(VerifyError::UnsupportedOp(
                    "weight tensor cannot be used as Stack input".into(),
                ));
            }
        }
    }

    // Pairwise concat: fold left with ConcatLayer.
    // All inputs are guaranteed Variable by the early rejection guard above.
    let mut iter = unsqueezed.into_iter();
    let mut acc = iter
        .next()
        .ok_or_else(|| VerifyError::InternalTranslationError {
            context: "Stack: unsqueezed list unexpectedly empty".into(),
        })?;
    for (i, next) in iter.enumerate() {
        let i = i + 1; // offset by 1 since we consumed the first element
        match (&acc, &next) {
            (TensorNodeValue::Variable(a), TensorNodeValue::Variable(b)) => {
                let concat_name = if i + 1 == inputs.len() {
                    format!("t{}", node_id.index())
                } else {
                    format!("t{}_cat{i}", node_id.index())
                };
                let layer = Layer::Concat(ConcatLayer::new(axis_i64));
                graph.add_node(GraphNode::binary(
                    concat_name.clone(),
                    layer,
                    a.clone(),
                    b.clone(),
                ));
                acc = TensorNodeValue::Variable(concat_name);
            }
            _ => {
                // Unreachable due to the early rejection guard, but fail-closed.
                return Err(VerifyError::InternalTranslationError {
                    context: "Stack with mixed constant/variable inputs".into(),
                });
            }
        }
    }
    Ok(acc)
}

/// Inject a constant tensor as a graph node with degenerate bounds
/// (`lower == upper == constant`), decoupled from any variable.
///
/// Mirrors `graph_tensor_attention.rs::inject_constant_via_zero_mul`: it hangs a
/// chain off a variable `parent_name` purely to seed valid graph edges, then
/// erases the parent's values with zero weights so the produced node's bounds are
/// exactly `constant` regardless of the parent. The four steps are:
///   1. Reshape the parent to 1-D `[-1]` (seeds an edge; values discarded next).
///   2. Zero-weight `Linear` → scalar `[1]` (always `0.0`, fully decoupling the
///      constant from the variable parent).
///   3. Zero-weight `Linear` + constant bias → flattened constant (the bias *is*
///      the constant; the zero weight contributes nothing).
///   4. Reshape back to the constant's N-D shape.
///
/// Soundness: the zero weights make the output independent of the parent's
/// interval, so the node's bounds are the degenerate point `constant` — exact.
/// Unlike the attention variant we do not require leading-axis agreement with the
/// parent: the parent is only an edge seed (its values are erased), and Concat
/// operands can legitimately differ along inner axes. We keep the non-empty-shape
/// guard so the zero-weight Linears are well-formed.
fn inject_constant_via_zero_mul(
    name: &str,
    constant: &ArrayD<f32>,
    parent_name: &str,
    parent_shape: &[usize],
    graph: &mut GraphNetwork,
) -> Result<String, VerifyError> {
    let const_shape = constant.shape();
    let parent_total: usize = parent_shape.iter().product();
    let const_flat: Vec<f32> = constant.iter().copied().collect();
    let const_total = const_flat.len();

    // Guard against degenerate (empty) shapes that would make the zero-weight
    // Linear ill-formed.
    if parent_total == 0 || const_total == 0 {
        return Err(VerifyError::UnsupportedOp(format!(
            "Concat constant '{name}' or anchor '{parent_name}' has an empty shape \
             (anchor {parent_shape:?}, constant {const_shape:?})"
        )));
    }

    // 1. Flatten parent to 1-D [parent_total] so the collapse Linear sees a known
    //    `in_features`. Reshape only seeds a valid edge; its values are discarded
    //    by the zero weight below.
    let flat_name = format!("{name}_flat");
    graph.add_node(GraphNode::new(
        flat_name.clone(),
        Layer::Reshape(ReshapeLayer::new(vec![-1])),
        vec![parent_name.to_string()],
    ));

    // 2. Collapse to scalar [1] with a zero weight: output is always 0.0,
    //    decoupling the constant from the variable parent entirely.
    let collapse = LinearLayer::new(Array2::zeros((1, parent_total)), Some(Array1::zeros(1)))
        .map_err(|e| VerifyError::InternalTranslationError {
            context: format!("Concat constant '{name}' collapse Linear: {e}"),
        })?;
    let scalar_name = format!("{name}_scalar");
    graph.add_node(GraphNode::new(
        scalar_name.clone(),
        Layer::Linear(collapse),
        vec![flat_name],
    ));

    // 3. Expand the scalar to the flattened constant via a zero weight + constant
    //    bias: output is always exactly `const_flat` (degenerate bounds).
    let expand = LinearLayer::new(
        Array2::zeros((const_total, 1)),
        Some(Array1::from_vec(const_flat)),
    )
    .map_err(|e| VerifyError::InternalTranslationError {
        context: format!("Concat constant '{name}' expand Linear: {e}"),
    })?;
    let const_1d_name = format!("{name}_1d");
    graph.add_node(GraphNode::new(
        const_1d_name.clone(),
        Layer::Linear(expand),
        vec![scalar_name],
    ));

    // 4. Restore the constant's N-D shape.
    let target: Vec<i64> = const_shape.iter().map(|&d| d as i64).collect();
    graph.add_node(GraphNode::new(
        name.to_string(),
        Layer::Reshape(ReshapeLayer::new(target)),
        vec![const_1d_name],
    ));
    Ok(name.to_string())
}

/// Translate `Concat` — join tensors along an existing axis.
///
/// Unlike `Stack`, no `UnsqueezeLayer` is needed because the axis already exists.
/// Pairwise fold with `ConcatLayer(axis)`. A constant/weight operand (e.g. a
/// learned CLS token concatenated with variable patch embeddings) is injected as
/// a graph node with degenerate bounds via `inject_constant_via_zero_mul` and then
/// participates in the concat alongside the variable operands.
pub(super) fn translate_concat(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    inputs: &[TensorNodeId],
    axis: usize,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    if inputs.len() < 2 {
        return Err(VerifyError::UnsupportedOp(
            "Concat with fewer than 2 inputs".into(),
        ));
    }

    // Pick a variable "anchor" — the first Variable operand. Constant/weight
    // operands (e.g. a learned CLS token concatenated with variable patch
    // embeddings) are injected as graph nodes with degenerate bounds via
    // `inject_constant_via_zero_mul`, which needs a variable parent to seed valid
    // graph edges. Its values are erased by a zero weight, so the anchor choice
    // does not affect the injected constants' bounds. Every real concat has at
    // least one variable operand; if ALL operands are constant the concat result
    // is itself a precisely-known constant — we keep a clear error here rather
    // than fold (the all-constant case never arises from a real model and folding
    // would need shape-aware `ndarray::concatenate` plumbing not present here).
    let anchor: Option<(String, Vec<usize>)> = inputs
        .iter()
        .find_map(|tid| match get_value(node_values, tid.index(), "Concat anchor") {
            Ok(TensorNodeValue::Variable(name)) => {
                // Fetch the anchor's IR shape for the zero-weight Linear seed.
                match get_value(ctx.all_nodes, tid.index(), "Concat anchor shape") {
                    Ok(node) => Some(Ok((name.clone(), node.shape.clone()))),
                    Err(e) => Some(Err(e)),
                }
            }
            Ok(_) => None,
            Err(e) => Some(Err(e)),
        })
        .transpose()?;
    let (anchor_name, anchor_shape) = match anchor {
        Some(a) => a,
        None => {
            return Err(VerifyError::UnsupportedOp(
                "Concat with all-constant inputs is not supported (no variable operand to \
                 anchor; a real concat always has at least one variable operand)"
                    .into(),
            ));
        }
    };

    let axis_i64 = dim_as_i64(axis, "Concat axis")?;

    // Resolve each operand to a graph node name: Variable operands use their own
    // name; constant/weight operands are injected with degenerate bounds equal to
    // the constant. A scalar `Constant` cannot be laid out into an N-D concat, so
    // it is still rejected as a genuine spec error.
    let mut operand_names: Vec<String> = Vec::with_capacity(inputs.len());
    for (i, tid) in inputs.iter().enumerate() {
        match get_value(node_values, tid.index(), "Concat input")? {
            TensorNodeValue::Variable(name) => operand_names.push(name.clone()),
            TensorNodeValue::WeightTensor(arr) => {
                let name = format!("t{}_const{i}", node_id.index());
                let injected =
                    inject_constant_via_zero_mul(&name, arr, &anchor_name, &anchor_shape, graph)?;
                operand_names.push(injected);
            }
            TensorNodeValue::Constant(_) => {
                return Err(VerifyError::UnsupportedOp(format!(
                    "Concat input {i} is a scalar constant — only variable or constant-tensor \
                     operands are supported"
                )));
            }
        }
    }

    // Pairwise concat: fold left with ConcatLayer over the resolved node names.
    let mut iter = operand_names.into_iter();
    let mut acc = iter
        .next()
        .ok_or_else(|| VerifyError::InternalTranslationError {
            context: "Concat: empty input list".into(),
        })?;
    for (i, next) in iter.enumerate() {
        let i = i + 1; // offset by 1 since we consumed the first element
        let concat_name = if i + 1 == inputs.len() {
            format!("t{}", node_id.index())
        } else {
            format!("t{}_cat{i}", node_id.index())
        };
        let layer = Layer::Concat(ConcatLayer::new(axis_i64));
        graph.add_node(GraphNode::binary(concat_name.clone(), layer, acc, next));
        acc = concat_name;
    }
    Ok(TensorNodeValue::Variable(acc))
}

#[cfg(test)]
#[path = "graph_tensor_structural_tests.rs"]
mod tests;
