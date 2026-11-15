// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SumReduce translation: pairwise AddLayer fold with constant accumulation.

use ny_propagate::layers::{AddConstantLayer, AddLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::ir::NodeId;

use crate::error::VerifyError;
use crate::graph::{add_unary_node, checked_constant, scalar_array, NodeValue};
use crate::util::get_value;

/// Translate a `SumReduce` node: fold constants separately, pairwise-add variables.
pub(crate) fn translate_sum_reduce(
    name: &str,
    inputs: &[NodeId],
    values: &[NodeValue],
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    let mut const_sum: f32 = 0.0;
    let mut var_refs: Vec<String> = Vec::new();
    for input_id in inputs {
        match get_value(values, input_id.index(), "SumReduce input")? {
            NodeValue::Constant(v) => const_sum += v.get(),
            NodeValue::Variable(var_name) => var_refs.push(var_name.clone()),
        }
    }

    if var_refs.is_empty() {
        return checked_constant(const_sum, "SumReduce(all constants)");
    }

    let mut acc_name = var_refs[0].clone();
    for (i, var_name) in var_refs.iter().enumerate().skip(1) {
        let step_name = format!("{name}_sum{i}");
        graph.add_node(GraphNode::binary(
            step_name.clone(),
            Layer::Add(AddLayer),
            acc_name,
            var_name.clone(),
        ));
        acc_name = step_name;
    }

    if const_sum != 0.0 {
        if !const_sum.is_finite() {
            return Err(VerifyError::NonFiniteConstant {
                value: const_sum,
                context: "SumReduce(mixed constant accumulation)".to_string(),
            });
        }
        let final_name = name.to_string();
        let layer = Layer::AddConstant(AddConstantLayer::new(scalar_array(const_sum)?));
        add_unary_node(&final_name, layer, &acc_name, graph);
        Ok(NodeValue::Variable(final_name))
    } else {
        Ok(NodeValue::Variable(acc_name))
    }
}
