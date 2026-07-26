// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::graph::FiniteF32;
use ndarray::{ArrayD, IxDyn};

/// resolve_output with a constant value must create a two-node subgraph
/// (MulConstant(0) → AddConstant(c)) that produces bounds [c, c] when
/// IBP-propagated. (#585 AC3)
#[test]
fn test_resolve_output_constant_produces_correct_bounds() {
    let mut graph = GraphNetwork::new();
    let c = FiniteF32::new(7.5).expect("finite");
    let val = NodeValue::Constant(c);

    let name =
        resolve_output(&val, "const_out", &mut graph).expect("resolve_output should succeed");
    graph.set_output(name);

    let lower = ArrayD::from_elem(IxDyn(&[]), -100.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[]), 100.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through constant subgraph");
    let (lo, hi) = output.lower_upper();

    // Constant output: regardless of input, bounds should be [c, c] = [7.5, 7.5]
    let lo_val = lo.iter().next().copied().unwrap();
    let hi_val = hi.iter().next().copied().unwrap();
    assert!(
        (lo_val - 7.5).abs() < 0.01,
        "constant output lower should be 7.5, got {lo_val}"
    );
    assert!(
        (hi_val - 7.5).abs() < 0.01,
        "constant output upper should be 7.5, got {hi_val}"
    );
}

/// resolve_output with a Variable just returns the name (no graph modification).
#[test]
fn test_resolve_output_variable_passthrough() {
    let mut graph = GraphNetwork::new();
    let val = NodeValue::Variable("nn_node".to_string());

    let name =
        resolve_output(&val, "const_out", &mut graph).expect("resolve_output should succeed");
    assert_eq!(name, "nn_node", "Variable should pass through unchanged");
    assert_eq!(
        graph.num_nodes(),
        0,
        "no nodes should be added for Variable"
    );
}
