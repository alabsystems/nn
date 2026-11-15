// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! KernelIR → NY `GraphNetwork` translation (scalar path).
//! For tensor-level translation, see [`super::graph_tensor`].

use ny_propagate::layers::{AddConstantLayer, SliceLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer, NETWORK_INPUT};
use nn_dsl::ir::KernelDef;

use crate::error::VerifyError;
use crate::util::get_value;

#[path = "graph_native.rs"]
mod native;
use native::try_native_layer;

#[path = "graph_translate.rs"]
mod translate;
pub(crate) use translate::{
    add_unary_node, checked_constant, has_variable_comparison, scalar_array, translate_node,
    TranslationContext,
};

/// How a kernel parameter is treated during verification.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ParamBinding {
    /// Variable input — bounds are provided via BoundedTensor.
    Variable,
    /// Fixed constant value at verification time.
    Constant(f32),
}

/// A finite f32 value — guaranteed to be neither NaN nor Inf.
///
/// Prevents construction of `NodeValue::Constant` with non-finite values
/// at the type level. Raw `NodeValue::Constant(raw_f32)` is a compile error;
/// use [`checked_constant`] or `FiniteF32::new()` instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FiniteF32(f32);

impl FiniteF32 {
    /// Create a new `FiniteF32`, rejecting NaN and Inf.
    pub(crate) fn new(val: f32) -> Result<Self, VerifyError> {
        if !val.is_finite() {
            return Err(VerifyError::NonFiniteConstant {
                value: val,
                context: String::new(),
            });
        }
        Ok(Self(val))
    }

    /// Extract the inner f32 value.
    pub(crate) fn get(self) -> f32 {
        self.0
    }
}

/// Result of translating each IR node: either a known constant or a
/// reference to a NY graph node.
#[derive(Clone, Debug)]
pub(crate) enum NodeValue {
    /// Known constant value (fully evaluated at translation time).
    /// Wrapped in [`FiniteF32`] to prevent NaN/Inf at the type level.
    Constant(FiniteF32),
    /// Variable node in the NY graph (name of the GraphNode).
    Variable(String),
}

/// Translate a `KernelDef` IR into a NY `GraphNetwork` (single-variable legacy API).
///
/// Param 0 is the variable input; params 1..N are constants.
/// For kernels with multiple variable inputs, use [`kernel_to_graph_multi`].
///
/// # Errors
///
/// Returns `VerifyError::ParamCountMismatch` if `constant_params.len() != kernel.params.len() - 1`.
#[must_use = "graph should be used for verification"]
pub fn kernel_to_graph(
    kernel: &KernelDef,
    constant_params: &[f32],
) -> Result<GraphNetwork, VerifyError> {
    // Build bindings: param 0 = Variable, rest = Constant
    let expected_constants = kernel.params.len().saturating_sub(1);
    if constant_params.len() != expected_constants {
        return Err(VerifyError::ParamCountMismatch {
            ir_count: kernel.params.len(),
            provided: constant_params.len(),
        });
    }

    let mut bindings = vec![ParamBinding::Variable];
    for &val in constant_params {
        bindings.push(ParamBinding::Constant(val));
    }

    kernel_to_graph_multi(kernel, &bindings)
}

/// Translate `KernelDef` IR to NY `GraphNetwork` with per-parameter bindings.
///
/// # Errors
///
/// Returns `VerifyError::ParamCountMismatch` if `bindings.len() != kernel.params.len()`.
#[must_use = "graph should be used for verification"]
pub fn kernel_to_graph_multi(
    kernel: &KernelDef,
    bindings: &[ParamBinding],
) -> Result<GraphNetwork, VerifyError> {
    kernel.validate()?;

    if bindings.len() != kernel.params.len() {
        return Err(VerifyError::ParamCountMismatch {
            ir_count: kernel.params.len(),
            provided: bindings.len(),
        });
    }

    // Validate constant bindings are finite
    for (i, binding) in bindings.iter().enumerate() {
        if let ParamBinding::Constant(val) = binding {
            if !val.is_finite() {
                return Err(VerifyError::NonFiniteConstant {
                    value: *val,
                    context: format!("bindings[{i}]"),
                });
            }
        }
    }

    // Fast path: native NY layer for known activations (tighter bounds).
    if let Some(graph) = try_native_layer(kernel, bindings)? {
        return Ok(graph);
    }

    // Count variable params and assign each a position in the input vector
    let var_positions: Vec<(usize, usize)> = bindings
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b, ParamBinding::Variable))
        .enumerate()
        .map(|(var_idx, (param_idx, _))| (param_idx, var_idx))
        .collect();
    let num_variables = var_positions.len();

    let mut graph = GraphNetwork::new();

    // Multi-variable: SliceLayer per variable param; single: direct NETWORK_INPUT.
    let mut param_node_names: Vec<Option<String>> = vec![None; kernel.params.len()];
    if num_variables > 1 {
        for &(param_idx, var_idx) in &var_positions {
            let slice_name = format!("input_p{param_idx}");
            let layer = Layer::Slice(SliceLayer::new(0, var_idx, var_idx + 1));
            graph.add_node(GraphNode::from_input(slice_name.clone(), layer));
            param_node_names[param_idx] = Some(slice_name);
        }
    }

    let mut node_values: Vec<NodeValue> = Vec::with_capacity(kernel.nodes.len());
    let ctx = TranslationContext {
        prefix: "",
        bindings,
        num_variables,
        param_node_names: &param_node_names,
        all_nodes: &kernel.nodes,
    };

    for node in &kernel.nodes {
        let value = translate_node(&ctx, node.id.index(), &node_values, &mut graph)?;
        node_values.push(value);
    }

    // Set output. Wrap bare NETWORK_INPUT in identity layer (see #477).
    match get_value(&node_values, kernel.output.index(), "kernel output")? {
        NodeValue::Variable(name) if name == NETWORK_INPUT => {
            let identity_name = format!("n{}_identity", kernel.output.index());
            graph.add_node(GraphNode::from_input(
                identity_name.clone(),
                Layer::AddConstant(AddConstantLayer::new(scalar_array(0.0)?)),
            ));
            graph.set_output(identity_name);
        }
        NodeValue::Variable(name) => {
            graph.set_output(name.clone());
        }
        NodeValue::Constant(val) => {
            let name = format!("n{}", kernel.output.index());
            graph.add_node(GraphNode::from_input(
                name.clone(),
                Layer::AddConstant(AddConstantLayer::new(scalar_array(val.get())?)),
            ));
            graph.set_output(name);
        }
    }

    Ok(graph)
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod tests;
