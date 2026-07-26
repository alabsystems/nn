// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared translation helpers for KernelIR → NY graph conversion.
//!
//! Extracted from `graph.rs` to reduce hub coupling (#508). All items are
//! re-exported from `crate::graph` so consumer imports are unchanged.

use ny_propagate::{GraphNetwork, GraphNode, Layer, NETWORK_INPUT};
use nn_dsl::ir::{IRNodeKind, KernelDef};
use ndarray::{ArrayD, IxDyn};

use crate::error::VerifyError;
use crate::graph_ops;
use crate::util::get_value;

use super::{FiniteF32, NodeValue, ParamBinding};

/// Immutable context shared across all node translations in a single kernel.
///
/// Bundles the parameters that remain constant throughout a translation pass,
/// reducing `translate_node` from 8 parameters to 4.
pub(crate) struct TranslationContext<'a> {
    pub prefix: &'a str,
    pub bindings: &'a [ParamBinding],
    pub num_variables: usize,
    pub param_node_names: &'a [Option<String>],
    pub all_nodes: &'a [nn_dsl::ir::IRNode],
}

pub(crate) fn translate_node(
    ctx: &TranslationContext<'_>,
    node_idx: usize,
    values: &[NodeValue],
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    let name = format!("{}n{node_idx}", ctx.prefix);
    let kind = &get_value(ctx.all_nodes, node_idx, "translate_node")?.kind;

    match kind {
        IRNodeKind::Param(idx) => match get_value(ctx.bindings, *idx, "Param binding")? {
            ParamBinding::Constant(val) => Ok(NodeValue::Constant(FiniteF32::new(*val)?)),
            ParamBinding::Variable => {
                // Check param_node_names first (used by tensor inline translation
                // and multi-variable scalar translation).
                if let Some(ref node_name) =
                    get_value(ctx.param_node_names, *idx, "param_node_names")?
                {
                    Ok(NodeValue::Variable(node_name.clone()))
                } else if ctx.num_variables == 1 {
                    Ok(NodeValue::Variable(NETWORK_INPUT.to_string()))
                } else {
                    Err(VerifyError::InternalTranslationError {
                        context: format!(
                            "variable param index {idx} missing slice node during graph translation"
                        ),
                    })
                }
            }
        },

        IRNodeKind::Literal(val) => {
            // f64→f32 cast: precision loss for values with >24 significant bits.
            // checked_constant rejects Inf/NaN, so overflow is caught.
            checked_constant(*val as f32, "Literal")
        }

        IRNodeKind::BinOp { op, lhs, rhs } => graph_ops::translate_binop(
            &name,
            *op,
            get_value(values, lhs.index(), "BinOp lhs")?,
            get_value(values, rhs.index(), "BinOp rhs")?,
            graph,
        ),

        IRNodeKind::UnaryFn { op, input } => graph_ops::translate_unary(
            &name,
            *op,
            get_value(values, input.index(), "UnaryFn input")?,
            graph,
        ),

        IRNodeKind::BinaryFn { op, lhs, rhs } => graph_ops::translate_binary_fn(
            &name,
            *op,
            get_value(values, lhs.index(), "BinaryFn lhs")?,
            get_value(values, rhs.index(), "BinaryFn rhs")?,
            graph,
        ),

        IRNodeKind::Powi { base, exp } => match get_value(values, base.index(), "Powi base")? {
            NodeValue::Constant(v) => {
                let val = v.get();
                checked_constant(val.powi(*exp), &format!("{val}.powi({exp})"))
            }
            NodeValue::Variable(base_name) => {
                use ny_propagate::layers::PowConstantLayer;
                // i32→f32 cast is lossless for |exp| <= 2^24 (16_777_216).
                // Beyond that, f32 mantissa (24 bits) cannot represent all i32 values.
                const POWI_F32_PRECISION_LIMIT: i32 = 1 << 24;
                if exp.unsigned_abs() > POWI_F32_PRECISION_LIMIT as u32 {
                    return Err(VerifyError::InternalTranslationError {
                        context: format!(
                            "Powi exponent {exp} exceeds i32→f32 precision limit ({POWI_F32_PRECISION_LIMIT})"
                        ),
                    });
                }
                let layer = Layer::PowConstant(PowConstantLayer::new(*exp as f32));
                graph.add_node(GraphNode::new(name.clone(), layer, vec![base_name.clone()]));
                Ok(NodeValue::Variable(name))
            }
        },

        IRNodeKind::Clamp { input, min, max } => graph_ops::translate_clamp(
            &name,
            get_value(values, input.index(), "Clamp input")?,
            get_value(values, min.index(), "Clamp min")?,
            get_value(values, max.index(), "Clamp max")?,
            graph,
        ),

        IRNodeKind::MinMax { op, lhs, rhs } => graph_ops::translate_minmax(
            &name,
            *op,
            get_value(values, lhs.index(), "MinMax lhs")?,
            get_value(values, rhs.index(), "MinMax rhs")?,
            graph,
        ),

        IRNodeKind::SumReduce { inputs } => {
            graph_ops::translate_sum_reduce(&name, inputs, values, graph)
        }

        IRNodeKind::Compare { op, lhs, rhs } => graph_ops::translate_compare(
            &name,
            *op,
            get_value(values, lhs.index(), "Compare lhs")?,
            get_value(values, rhs.index(), "Compare rhs")?,
            graph,
        ),

        IRNodeKind::Select {
            cond,
            then_val,
            else_val,
        } => graph_ops::translate_select(
            &name,
            cond.index(),
            then_val.index(),
            else_val.index(),
            ctx.all_nodes,
            values,
            graph,
        ),

        _ => Err(VerifyError::UnsupportedOp(format!("{kind:?}"))),
    }
}

/// Helper: add a unary (single-input) node to the graph.
pub(crate) fn add_unary_node(name: &str, layer: Layer, input_name: &str, graph: &mut GraphNetwork) {
    if input_name == NETWORK_INPUT {
        graph.add_node(GraphNode::from_input(name.to_string(), layer));
    } else {
        graph.add_node(GraphNode::new(
            name.to_string(),
            layer,
            vec![input_name.to_string()],
        ));
    }
}

/// Check whether any `Compare` node in the kernel IR has variable operands,
/// indicating the NY graph uses a continuous approximation (e.g.,
/// `lhs - rhs` for Gt/Ge) that may produce looser bounds.
pub(crate) fn has_variable_comparison(kernel: &KernelDef, bindings: &[ParamBinding]) -> bool {
    let has_compare = kernel
        .nodes
        .iter()
        .any(|n| matches!(n.kind, IRNodeKind::Compare { .. }));
    if !has_compare {
        return false;
    }

    // Forward data-flow: mark nodes that depend on a Variable parameter.
    let mut depends_on_variable = vec![false; kernel.nodes.len()];
    // Bounds-checked lookup: returns false for out-of-range NodeIds.
    // Malformed IR (invalid NodeId references) is caught later in translate_node().
    macro_rules! dep {
        ($id:expr) => {
            depends_on_variable
                .get($id.index())
                .copied()
                .unwrap_or(false)
        };
    }
    for (i, node) in kernel.nodes.iter().enumerate() {
        depends_on_variable[i] = match &node.kind {
            IRNodeKind::Param(idx) => matches!(bindings.get(*idx), Some(ParamBinding::Variable)),
            IRNodeKind::Literal(_) => false,
            IRNodeKind::BinOp { lhs, rhs, .. }
            | IRNodeKind::Compare { lhs, rhs, .. }
            | IRNodeKind::MinMax { lhs, rhs, .. } => dep!(lhs) || dep!(rhs),
            IRNodeKind::UnaryFn { input, .. } | IRNodeKind::Powi { base: input, .. } => {
                dep!(input)
            }
            IRNodeKind::Clamp { input, min, max } => dep!(input) || dep!(min) || dep!(max),
            IRNodeKind::Select {
                cond,
                then_val,
                else_val,
            } => dep!(cond) || dep!(then_val) || dep!(else_val),
            IRNodeKind::SumReduce { inputs } => inputs.iter().any(|id| dep!(id)),
            // SAFETY: IRNodeKind is #[non_exhaustive]. Unknown variants are assumed
            // to depend on variables (conservative). This may produce unnecessary
            // diagnostics but won't silently miss real variable comparisons.
            _ => true,
        };
    }

    kernel
        .nodes
        .iter()
        .enumerate()
        .any(|(i, node)| matches!(node.kind, IRNodeKind::Compare { .. }) && depends_on_variable[i])
}

/// Wrap a constant value, rejecting NaN and Inf from constant folding.
///
/// Preferred entry point for creating `NodeValue::Constant` — provides
/// a descriptive context string in the error message.
pub(crate) fn checked_constant(value: f32, context: &str) -> Result<NodeValue, VerifyError> {
    if !value.is_finite() {
        return Err(VerifyError::NonFiniteConstant {
            value,
            context: context.to_string(),
        });
    }
    // value is validated finite above; construct FiniteF32 directly.
    Ok(NodeValue::Constant(FiniteF32(value)))
}

/// Create a 0-dimensional scalar ArrayD<f32>.
///
/// Returns `Err(NonFiniteConstant)` if `value` is NaN or Inf. All current
/// callers source values from `FiniteF32::get()` or validated paths, so the
/// error path is defense-in-depth.
pub(crate) fn scalar_array(value: f32) -> Result<ArrayD<f32>, VerifyError> {
    if !value.is_finite() {
        return Err(VerifyError::NonFiniteConstant {
            value,
            context: "scalar_array".to_string(),
        });
    }
    Ok(ArrayD::from_elem(IxDyn(&[]), value))
}

#[cfg(test)]
#[path = "graph_translate_tests.rs"]
mod tests;
