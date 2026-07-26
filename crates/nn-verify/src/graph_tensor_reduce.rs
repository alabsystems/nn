// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reduce operations and multi-variable input setup for tensor-level translation.
//!
//! Extracted from `graph_tensor.rs` (#358) to keep the main dispatcher under 300 lines.

use ny_propagate::layers::{
    ReduceMaxLayer, ReduceMeanLayer, ReduceMinLayer, ReduceSumLayer, ReshapeLayer, SliceLayer,
};
use ny_propagate::{GraphNetwork, GraphNode, Layer, NETWORK_INPUT};
use nn_dsl::tensor_ir::{ReduceOp, TensorNode, TensorNodeId, TensorOpKind};

use crate::error::{StructuralError, VerifyError};
use crate::graph::{add_unary_node, checked_constant, NodeValue};
use crate::util::get_value;

use super::{dim_as_i64, TensorNodeValue, TensorParamBinding, TensorTranslationContext};

/// Split a single flat network input into per-variable subgraph entry points.
///
/// When `num_variables > 1`, all Variable inputs share ONE network input that is
/// a flat per-variable element concatenation. (Tests feed this either as a 1-D
/// `[total_flat]` tensor — e.g. gpt-oss attention/kv-cache — or, for
/// equal-shaped variables, as a higher-rank `[num_variables, ...shared]` tensor;
/// both flatten row-major to the same element buffer.) We:
///   1. `Reshape([-1])` the network input to a flat 1-D tensor (`t_input_flat`),
///      so the per-variable slices are layout-agnostic w.r.t. the fed tensor's
///      declared rank.
///   2. For each Variable, `Slice(axis=0, elem_offset, elem_offset+flat_i)` peels
///      off its `flat_i = product(true_shape)` elements at the running byte/elem
///      offset, then `Reshape(true_shape)` restores that variable's TRUE shape.
///
/// Each variable therefore enters its subgraph at its declared rank with NATURAL
/// axes — no stacking dimension — so downstream ops use no axis offset
/// (`axis_offset = 0`). Slice + Reshape are exact layout ops, so this only
/// corrects shapes and never loosens bounds.
///
/// `variable_shapes[i]` is the declared shape of the i-th Variable input, in the
/// order the Variable bindings appear. Returns `(num_variables, input_node_names)`
/// where `input_node_names[i]` is `Some(node_name)` for Variable bindings and
/// `None` for constants.
pub(crate) fn setup_multi_variable_inputs(
    input_bindings: &[TensorParamBinding],
    variable_shapes: &[Vec<usize>],
    graph: &mut GraphNetwork,
) -> Result<(usize, Vec<Option<String>>), VerifyError> {
    let num_variables = input_bindings
        .iter()
        .filter(|b| matches!(b, TensorParamBinding::Variable))
        .count();

    let mut variable_node_names: Vec<String> = Vec::new();
    if num_variables > 1 {
        // 1. Flatten the single network input to 1-D so per-variable slices are
        //    independent of the fed tensor's declared rank (flat or stacked).
        let flat_name = "t_input_flat".to_string();
        graph.add_node(GraphNode::from_input(
            flat_name.clone(),
            Layer::Reshape(ReshapeLayer::new(vec![-1])),
        ));

        // 2. Slice each variable's flat element range, then reshape to its true
        //    shape. Offsets are the running sum of preceding variables' element
        //    counts (row-major flat concatenation order).
        let mut elem_offset: usize = 0;
        for (var_idx, shape) in variable_shapes.iter().enumerate() {
            let flat_size: usize = shape.iter().product();
            let end = elem_offset.checked_add(flat_size).ok_or_else(|| {
                VerifyError::Structural(StructuralError::ShapeConstraint {
                    context: format!(
                        "multi-variable input v{var_idx}: element offset overflow"
                    ),
                })
            })?;

            let slice_name = format!("t_input_v{var_idx}_slice");
            graph.add_node(GraphNode::new(
                slice_name.clone(),
                Layer::Slice(SliceLayer::new(0, elem_offset, end)),
                vec![flat_name.clone()],
            ));

            let var_name = format!("t_input_v{var_idx}");
            let true_shape: Vec<i64> = shape
                .iter()
                .map(|&d| dim_as_i64(d, "multi-variable reshape"))
                .collect::<Result<_, _>>()?;
            graph.add_node(GraphNode::new(
                var_name.clone(),
                Layer::Reshape(ReshapeLayer::new(true_shape)),
                vec![slice_name],
            ));
            variable_node_names.push(var_name);

            elem_offset = end;
        }
    }

    let input_node_names = {
        let mut var_counter = 0usize;
        input_bindings
            .iter()
            .map(|b| match b {
                TensorParamBinding::Variable => {
                    let name = if num_variables > 1 {
                        let n = variable_node_names[var_counter].clone();
                        var_counter += 1;
                        n
                    } else {
                        var_counter += 1;
                        NETWORK_INPUT.to_string()
                    };
                    Some(name)
                }
                TensorParamBinding::ConstantScalar(_) | TensorParamBinding::ConstantTensor(_) => {
                    None
                }
            })
            .collect()
    };

    Ok((num_variables, input_node_names))
}

/// Collect the declared shape of each Variable input, in binding order.
///
/// Bindings are positional over the kernel's `Input` nodes (same order). For each
/// `TensorParamBinding::Variable` we record the corresponding `Input` node's
/// declared shape so [`setup_multi_variable_inputs`] can reshape each flat slice
/// back to that variable's true rank.
pub(crate) fn variable_input_shapes(
    input_bindings: &[TensorParamBinding],
    all_nodes: &[TensorNode],
) -> Result<Vec<Vec<usize>>, VerifyError> {
    let mut input_shapes = all_nodes.iter().filter_map(|n| match &n.kind {
        TensorOpKind::Input { shape, .. } => Some(shape.clone()),
        _ => None,
    });
    let mut shapes = Vec::new();
    for (idx, binding) in input_bindings.iter().enumerate() {
        let shape = input_shapes.next().ok_or_else(|| {
            VerifyError::Structural(StructuralError::ShapeConstraint {
                context: format!(
                    "multi-variable input: binding {idx} has no matching Input node"
                ),
            })
        })?;
        if matches!(binding, TensorParamBinding::Variable) {
            shapes.push(shape);
        }
    }
    Ok(shapes)
}

/// Translate a Reduce tensor operation (Mean or Sum) on constant or variable input.
pub(crate) fn translate_reduce_op(
    ctx: &TensorTranslationContext<'_>,
    node_id: TensorNodeId,
    op: &ReduceOp,
    input: &TensorNodeId,
    axis: usize,
    keepdim: bool,
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    let input_val = get_value(node_values, input.index(), "Reduce input")?;
    match input_val {
        TensorNodeValue::Constant(val) => {
            // Reducing a constant tensor:
            // - Mean of identical values = the value
            // - Max/Min of identical values = the value
            // - Sum of N identical values = N * value (need compile-time dim size)
            match op {
                ReduceOp::Mean | ReduceOp::Max | ReduceOp::Min => {
                    Ok(TensorNodeValue::Constant(*val))
                }
                ReduceOp::Sum => {
                    let shape = &get_value(ctx.all_nodes, input.index(), "Reduce all_nodes")?.shape;
                    let dim = *shape.get(axis).ok_or_else(|| {
                        VerifyError::Structural(StructuralError::ShapeConstraint {
                            context: format!(
                                "ReduceSum axis {axis} out of bounds for shape with {} dims",
                                shape.len()
                            ),
                        })
                    })?;
                    let v = val.get();
                    // Compute in f64 to avoid precision loss for dim > 2^24.
                    let product = f64::from(v) * (dim as f64);
                    checked_constant(
                        product as f32,
                        &format!("ReduceSum(constant={v}, dim={dim})"),
                    )
                    .map(|nv| match nv {
                        NodeValue::Constant(fv) => TensorNodeValue::Constant(fv),
                        NodeValue::Variable(n) => TensorNodeValue::Variable(n),
                    })
                }
                _ => Err(VerifyError::UnsupportedOp(format!(
                    "ReduceOp {op:?} on constant tensor"
                ))),
            }
        }
        TensorNodeValue::Variable(input_name) => {
            let node_name = format!("t{}", node_id.index());
            // Each variable enters at its TRUE rank (see `setup_multi_variable_inputs`),
            // so the user-declared axis is used directly with no offset.
            let adjusted_axis = dim_as_i64(axis, "ReduceOp axis")?;
            // keepdims preserves the reduced axis as size 1 for broadcasting.
            let layer = match op {
                ReduceOp::Mean => {
                    Layer::ReduceMean(ReduceMeanLayer::new(vec![adjusted_axis], keepdim))
                }
                ReduceOp::Sum => {
                    Layer::ReduceSum(ReduceSumLayer::new(vec![adjusted_axis], keepdim))
                }
                ReduceOp::Max => {
                    Layer::ReduceMax(ReduceMaxLayer::new(vec![adjusted_axis], keepdim))
                }
                ReduceOp::Min => {
                    Layer::ReduceMin(ReduceMinLayer::new(vec![adjusted_axis], keepdim))
                }
                _ => {
                    return Err(VerifyError::UnsupportedOp(format!(
                        "ReduceOp {op:?} on variable tensor"
                    )))
                }
            };
            add_unary_node(&node_name, layer, input_name, graph);
            Ok(TensorNodeValue::Variable(node_name))
        }
        TensorNodeValue::WeightTensor(_) => Err(VerifyError::UnsupportedOp(
            "weight tensor cannot be used as reduce input".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::FiniteF32;
    use crate::graph_tensor::TensorParamBinding;
    use nn_dsl::tensor_ir::{TensorNode, TensorNodeId, TensorOpKind};

    /// AC3 regression: ReduceSum on a constant with axis beyond shape dimensions
    /// returns StructuralError instead of panicking (#562).
    #[test]
    fn test_reduce_sum_constant_out_of_bounds_axis_returns_error() {
        // Input with shape [2, 3] — valid axes are 0 and 1.
        let nodes = vec![TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".into(),
                shape: vec![2, 3],
            },
            vec![2, 3],
        )];
        let bindings = [TensorParamBinding::ConstantScalar(1.0)];
        let names: Vec<Option<String>> = vec![None];
        let ctx = TensorTranslationContext {
            input_bindings: &bindings,
            input_node_names: &names,
            axis_offset: 0,
            all_nodes: &nodes,
            norm_mode: crate::verify_types::NormBoundsMode::Conservative,
        };

        let node_values = vec![TensorNodeValue::Constant(
            FiniteF32::new(1.0).expect("finite"),
        )];
        let mut graph = GraphNetwork::new();

        // axis=5 is out of bounds for a 2-dim shape.
        let result = translate_reduce_op(
            &ctx,
            TensorNodeId::new(1),
            &ReduceOp::Sum,
            &TensorNodeId::new(0),
            5, // out-of-bounds axis
            false,
            &node_values,
            &mut graph,
        );

        assert!(
            result.is_err(),
            "ReduceSum with out-of-bounds axis must fail"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("out of bounds"),
            "error should mention 'out of bounds', got: {err_msg}"
        );
    }

    /// ReduceMean with keepdim=true on a variable produces a ReduceMean NY
    /// node with keepdims=true. Part of #2203.
    #[test]
    fn test_reduce_mean_keepdim_true_variable_produces_keepdim_layer() {
        let nodes = vec![TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".into(),
                shape: vec![2, 4],
            },
            vec![2, 4],
        )];
        let bindings = [TensorParamBinding::Variable];
        let names: Vec<Option<String>> = vec![Some(NETWORK_INPUT.to_string())];
        let ctx = TensorTranslationContext {
            input_bindings: &bindings,
            input_node_names: &names,
            axis_offset: 0,
            all_nodes: &nodes,
            norm_mode: crate::verify_types::NormBoundsMode::Conservative,
        };

        let node_values = vec![TensorNodeValue::Variable(NETWORK_INPUT.to_string())];
        let mut graph = GraphNetwork::new();

        let result = translate_reduce_op(
            &ctx,
            TensorNodeId::new(1),
            &ReduceOp::Mean,
            &TensorNodeId::new(0),
            1,    // reduce last axis
            true, // keepdim=true
            &node_values,
            &mut graph,
        );

        assert!(
            result.is_ok(),
            "keepdim=true ReduceMean should succeed: {result:?}"
        );
        match result.unwrap() {
            TensorNodeValue::Variable(name) => {
                assert_eq!(name, "t1", "output node should be named t1");
            }
            other => unreachable!("expected Variable, got {other:?}"),
        }
    }

    /// ReduceSum with keepdim=true on a variable produces correct NY layer.
    #[test]
    fn test_reduce_sum_keepdim_true_variable() {
        let nodes = vec![TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".into(),
                shape: vec![3, 5],
            },
            vec![3, 5],
        )];
        let bindings = [TensorParamBinding::Variable];
        let names: Vec<Option<String>> = vec![Some(NETWORK_INPUT.to_string())];
        let ctx = TensorTranslationContext {
            input_bindings: &bindings,
            input_node_names: &names,
            axis_offset: 0,
            all_nodes: &nodes,
            norm_mode: crate::verify_types::NormBoundsMode::Conservative,
        };

        let node_values = vec![TensorNodeValue::Variable(NETWORK_INPUT.to_string())];
        let mut graph = GraphNetwork::new();

        let result = translate_reduce_op(
            &ctx,
            TensorNodeId::new(1),
            &ReduceOp::Sum,
            &TensorNodeId::new(0),
            1,    // reduce axis 1
            true, // keepdim=true
            &node_values,
            &mut graph,
        );

        assert!(
            result.is_ok(),
            "keepdim=true ReduceSum should succeed: {result:?}"
        );
        match result.unwrap() {
            TensorNodeValue::Variable(name) => {
                assert_eq!(name, "t1");
            }
            other => unreachable!("expected Variable, got {other:?}"),
        }
    }

    /// ReduceMean with keepdim=true on a constant folds to the same constant
    /// value (mean of identical values is identity regardless of keepdim).
    #[test]
    fn test_reduce_mean_keepdim_true_constant_folds() {
        let nodes = vec![TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".into(),
                shape: vec![2, 3],
            },
            vec![2, 3],
        )];
        let bindings = [TensorParamBinding::ConstantScalar(5.0)];
        let names: Vec<Option<String>> = vec![None];
        let ctx = TensorTranslationContext {
            input_bindings: &bindings,
            input_node_names: &names,
            axis_offset: 0,
            all_nodes: &nodes,
            norm_mode: crate::verify_types::NormBoundsMode::Conservative,
        };

        let node_values = vec![TensorNodeValue::Constant(
            FiniteF32::new(5.0).expect("finite"),
        )];
        let mut graph = GraphNetwork::new();

        let result = translate_reduce_op(
            &ctx,
            TensorNodeId::new(1),
            &ReduceOp::Mean,
            &TensorNodeId::new(0),
            1,
            true, // keepdim=true — should not affect constant folding
            &node_values,
            &mut graph,
        );

        assert!(
            result.is_ok(),
            "keepdim=true ReduceMean on constant should succeed"
        );
        match result.unwrap() {
            TensorNodeValue::Constant(v) => {
                assert!(
                    (v.get() - 5.0).abs() < 1e-6,
                    "ReduceMean(constant=5.0) should be 5.0 regardless of keepdim, got {}",
                    v.get()
                );
            }
            other => unreachable!("expected Constant, got {other:?}"),
        }
    }

    /// ReduceSum on a constant with valid axis succeeds.
    #[test]
    fn test_reduce_sum_constant_valid_axis_succeeds() {
        let nodes = vec![TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".into(),
                shape: vec![2, 3],
            },
            vec![2, 3],
        )];
        let bindings = [TensorParamBinding::ConstantScalar(2.0)];
        let names: Vec<Option<String>> = vec![None];
        let ctx = TensorTranslationContext {
            input_bindings: &bindings,
            input_node_names: &names,
            axis_offset: 0,
            all_nodes: &nodes,
            norm_mode: crate::verify_types::NormBoundsMode::Conservative,
        };

        let node_values = vec![TensorNodeValue::Constant(
            FiniteF32::new(2.0).expect("finite"),
        )];
        let mut graph = GraphNetwork::new();

        // axis=1 is valid for shape [2, 3] — reduces dim of size 3.
        let result = translate_reduce_op(
            &ctx,
            TensorNodeId::new(1),
            &ReduceOp::Sum,
            &TensorNodeId::new(0),
            1, // valid axis
            false,
            &node_values,
            &mut graph,
        );

        assert!(
            result.is_ok(),
            "ReduceSum with valid axis should succeed: {result:?}"
        );
        // Sum of 3 copies of 2.0 = 6.0
        match result.unwrap() {
            TensorNodeValue::Constant(v) => {
                assert!(
                    (v.get() - 6.0).abs() < 1e-6,
                    "ReduceSum(constant=2.0, dim=3) should be 6.0, got {}",
                    v.get()
                );
            }
            other => unreachable!("expected Constant, got {other:?}"),
        }
    }
}
