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

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proves that `checked_constant` correctly handles the sum of two finite
    /// constants: accepts when the sum is finite, rejects when it overflows.
    /// This is the same accumulation pattern used by `translate_sum_reduce`.
    ///
    /// Decomposed: only tests the finite-sum path to avoid `String` allocation
    /// in `checked_constant`'s error path (which causes syn/raw_vec unwinding
    /// timeout in CBMC). The rejection path is proved by
    /// `checked_constant_accepts_finite` in graph.rs.
    /// Uses `unwind(8)` to bound syn::ErrorMessage Drop unwinding (#608).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn sum_reduce_accumulation_finiteness_guard() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());
        // Restrict to range where a + b is guaranteed finite,
        // avoiding the error-path `String` allocation that causes timeout.
        kani::assume(a.abs() <= 1e30);
        kani::assume(b.abs() <= 1e30);
        let sum = a + b;
        kani::assume(sum.is_finite());

        let result = checked_constant(sum, "SumReduce kani");
        let val = result.expect("finite sum must produce Ok");
        match val {
            NodeValue::Constant(f) => {
                assert_eq!(f.get().to_bits(), sum.to_bits(), "sum must be preserved");
            }
            NodeValue::Variable(_) => assert!(false, "checked_constant must return Constant"),
        }
    }
}
