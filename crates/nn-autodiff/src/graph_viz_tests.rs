// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DOT/Graphviz computation graph export.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::graph_viz;
use crate::tracked::TrackedTensor;
use crate::var::Var;

/// Build a simple linear graph: x (var) -> mul_scalar(3.0) -> add_scalar(1.0) -> output
fn build_linear_graph() -> (Var, Arc<TrackedTensor>) {
    let x = Var::new(DynTensor::from_vec(vec![2.0, 3.0], &[2], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let mul = t.mul_scalar(3.0).unwrap();
    let add = mul.add_scalar(1.0).unwrap();
    (x, add)
}

#[test]
fn test_graph_to_dot_contains_digraph() {
    let (_x, output) = build_linear_graph();
    let dot = graph_viz::graph_to_dot(&output);
    assert!(
        dot.contains("digraph"),
        "DOT output must contain 'digraph' keyword"
    );
}

#[test]
fn test_graph_to_dot_linear_node_count() {
    // x -> mul_scalar -> add_scalar = 3 nodes
    let (_x, output) = build_linear_graph();
    let count = graph_viz::node_count(&output);
    assert_eq!(count, 3, "Linear graph x -> mul -> add should have 3 nodes");
}

#[test]
fn test_graph_to_dot_linear_edge_count() {
    // x -> mul_scalar -> add_scalar = 2 forward edges
    let (_x, output) = build_linear_graph();
    let count = graph_viz::edge_count(&output);
    assert_eq!(
        count, 2,
        "Linear graph x -> mul -> add should have 2 forward edges"
    );
}

#[test]
fn test_graph_to_dot_edge_direction_parent_to_child() {
    let (_x, output) = build_linear_graph();
    let dot = graph_viz::graph_to_dot(&output);

    // The DOT output should contain forward edges (parent_id -> child_id).
    // Find all "nX -> nY;" lines before the backward section.
    let forward_section = dot.split("// Backward gradient flow").next().unwrap();
    let forward_edges: Vec<&str> = forward_section
        .lines()
        .filter(|l| l.contains(" -> ") && l.trim().starts_with('n'))
        .collect();

    assert_eq!(
        forward_edges.len(),
        2,
        "Should have 2 forward edges in the forward section"
    );

    // Each edge should be "    nP -> nC;" format
    for edge in &forward_edges {
        assert!(
            edge.trim().starts_with('n') && edge.contains(" -> n"),
            "Edge should be in 'nP -> nC' format: {edge}"
        );
    }
}

#[test]
fn test_graph_to_dot_color_coding() {
    let (_x, output) = build_linear_graph();
    let dot = graph_viz::graph_to_dot(&output);

    // Green for Var (input)
    assert!(
        dot.contains("#90EE90"),
        "DOT should contain green color for variable inputs"
    );
    // Blue for operations
    assert!(
        dot.contains("#87CEEB"),
        "DOT should contain blue color for operations"
    );
    // Red/pink for output
    assert!(
        dot.contains("#FFB6C1"),
        "DOT should contain red/pink color for output node"
    );
}

#[test]
fn test_graph_to_dot_shape_info_present() {
    let (_x, output) = build_linear_graph();
    let dot = graph_viz::graph_to_dot(&output);

    // Full mode should show shape information
    assert!(
        dot.contains("[2]"),
        "Full DOT should contain shape [2] for the tensor dimensions"
    );
    assert!(
        dot.contains("grad="),
        "Full DOT should contain grad= annotations"
    );
}

#[test]
fn test_graph_to_dot_minimal_omits_shape() {
    let (_x, output) = build_linear_graph();
    let dot_full = graph_viz::graph_to_dot(&output);
    let dot_minimal = graph_viz::graph_to_dot_minimal(&output);

    // Minimal mode should not contain shape info
    assert!(
        !dot_minimal.contains("grad="),
        "Minimal DOT should not contain grad= annotations"
    );

    // But full mode does
    assert!(
        dot_full.contains("grad="),
        "Full DOT should contain grad= annotations"
    );

    // Minimal should still be valid DOT
    assert!(
        dot_minimal.contains("digraph"),
        "Minimal DOT must still be valid with 'digraph'"
    );
}

#[test]
fn test_graph_to_dot_no_grad_constant() {
    // Build a graph with a constant (no-grad) node: const + var
    let x = Var::new(DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let c = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![5.0], &[1], &Device::Cpu).unwrap(),
    ));
    let sum = t.add(&c).unwrap();

    let dot = graph_viz::graph_to_dot(&sum);

    // Should contain gray for no-grad constant
    assert!(
        dot.contains("#D3D3D3"),
        "DOT should contain gray color for no-grad constant"
    );
    assert!(
        dot.contains("Const"),
        "DOT should label constant nodes as 'Const'"
    );
}

#[test]
fn test_graph_to_dot_diamond_graph() {
    // Diamond: x -> sqr, x -> neg, sqr + neg -> add
    // Tests that shared nodes are not duplicated.
    let x = Var::new(DynTensor::from_vec(vec![3.0], &[1], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sqr = t.sqr().unwrap();
    let neg = t.neg().unwrap();
    let sum = sqr.add(&neg).unwrap();

    let count = graph_viz::node_count(&sum);
    // x, sqr, neg, add = 4 nodes
    assert_eq!(count, 4, "Diamond graph should have 4 unique nodes");

    let edges = graph_viz::edge_count(&sum);
    // x->sqr, x->neg, sqr->add, neg->add = 4 edges
    assert_eq!(edges, 4, "Diamond graph should have 4 forward edges");
}

#[test]
fn test_graph_to_dot_op_names_present() {
    let (_x, output) = build_linear_graph();
    let dot = graph_viz::graph_to_dot(&output);

    assert!(
        dot.contains("Var"),
        "DOT should contain 'Var' label for the variable node"
    );
    assert!(
        dot.contains("MulScalar"),
        "DOT should contain 'MulScalar' for the multiply operation"
    );
    assert!(
        dot.contains("AddScalar"),
        "DOT should contain 'AddScalar' for the add operation"
    );
}

#[test]
fn test_graph_to_dot_backward_edges() {
    let (_x, output) = build_linear_graph();
    let dot = graph_viz::graph_to_dot(&output);

    // After the backward section header, there should be dashed backward edges.
    let backward_section = dot.split("// Backward gradient flow").nth(1).unwrap_or("");
    let backward_edges: Vec<&str> = backward_section
        .lines()
        .filter(|l| l.contains(" -> ") && l.trim().starts_with('n'))
        .collect();

    assert_eq!(
        backward_edges.len(),
        2,
        "Should have 2 backward edges matching the 2 forward edges"
    );
}

#[test]
fn test_write_dot_file() {
    let (_x, output) = build_linear_graph();
    let tmp = std::env::temp_dir().join("nn_graph_viz_test.dot");
    graph_viz::write_dot_file(&output, &tmp).expect("write_dot_file should succeed");

    let content = std::fs::read_to_string(&tmp).expect("should read written file");
    assert!(
        content.contains("digraph"),
        "Written file should contain valid DOT"
    );

    // Cleanup
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_graph_to_dot_single_node() {
    // Single leaf node (no ops)
    let x = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());

    let dot = graph_viz::graph_to_dot(&t);
    assert!(dot.contains("digraph"));

    let count = graph_viz::node_count(&t);
    assert_eq!(count, 1, "Single node graph should have 1 node");

    let edges = graph_viz::edge_count(&t);
    assert_eq!(edges, 0, "Single node graph should have 0 edges");
}
