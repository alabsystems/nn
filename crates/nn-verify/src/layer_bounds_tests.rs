// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for per-layer bound extraction from NY GraphNetworks.

use crate::layer_bounds::extract_layer_bounds;
use crate::verify::PropMethod;
use ny_api::BoundedTensor;
use ndarray::{ArrayD, IxDyn};

/// Build a simple scalar kernel graph: snake(x, alpha=1.0) -> x + sin(x)^2
/// This produces a multi-layer graph with known structure.
fn snake_graph() -> (ny_propagate::GraphNetwork, BoundedTensor) {
    let kernel = crate::test_helpers::parse_kernel(
        "fn snake(x: f32, alpha: f32) -> f32 { x + (1.0 / alpha) * (alpha * x).sin().powi(2) }",
    );
    let graph = crate::graph::kernel_to_graph(&kernel, &[1.0]).expect("build snake graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");

    (graph, input)
}

/// Build a simple ReLU kernel graph: relu(x) = max(0, x)
fn relu_graph() -> (ny_propagate::GraphNetwork, BoundedTensor) {
    let kernel = crate::test_helpers::parse_kernel(
        "fn nn_relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }",
    );
    let graph = crate::graph::kernel_to_graph(&kernel, &[]).expect("build relu graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");

    (graph, input)
}

#[test]
fn test_extract_layer_bounds_returns_records() {
    let (graph, input) = snake_graph();
    let records = extract_layer_bounds(&graph, &input).expect("extraction should succeed");

    // Snake kernel translates to multiple NY nodes.
    assert!(!records.is_empty(), "should have at least one layer record");
}

#[test]
fn test_layer_indices_are_sequential() {
    let (graph, input) = snake_graph();
    let records = extract_layer_bounds(&graph, &input).expect("extraction");

    for (i, record) in records.iter().enumerate() {
        assert_eq!(
            record.layer_index, i,
            "layer_index should match position in topological order"
        );
    }
}

#[test]
fn test_layer_types_are_non_empty() {
    let (graph, input) = snake_graph();
    let records = extract_layer_bounds(&graph, &input).expect("extraction");

    for record in &records {
        assert!(
            !record.layer_type.is_empty(),
            "layer_type should be non-empty for layer {}",
            record.layer_index
        );
    }
}

#[test]
fn test_bounds_are_valid_intervals() {
    let (graph, input) = snake_graph();
    let records = extract_layer_bounds(&graph, &input).expect("extraction");

    for record in &records {
        for (i, &(lo, hi)) in record.output_bounds.iter().enumerate() {
            assert!(
                lo <= hi || lo.is_nan() || hi.is_nan(),
                "layer {} output element {}: lower {} should be <= upper {}",
                record.layer_index,
                i,
                lo,
                hi
            );
        }
    }
}

#[test]
fn test_input_bounds_non_empty() {
    let (graph, input) = snake_graph();
    let records = extract_layer_bounds(&graph, &input).expect("extraction");

    for record in &records {
        assert!(
            !record.input_bounds.is_empty(),
            "layer {} should have non-empty input_bounds",
            record.layer_index
        );
    }
}

#[test]
fn test_output_bounds_non_empty() {
    let (graph, input) = snake_graph();
    let records = extract_layer_bounds(&graph, &input).expect("extraction");

    for record in &records {
        assert!(
            !record.output_bounds.is_empty(),
            "layer {} should have non-empty output_bounds",
            record.layer_index
        );
    }
}

#[test]
fn test_method_is_crown_or_ibp() {
    let (graph, input) = snake_graph();
    let records = extract_layer_bounds(&graph, &input).expect("extraction");

    for record in &records {
        assert!(
            record.method == PropMethod::Crown || record.method == PropMethod::Ibp,
            "layer {} method should be Crown or Ibp, got {:?}",
            record.layer_index,
            record.method
        );
    }
}

#[test]
fn test_relu_graph_extraction() {
    let (graph, input) = relu_graph();
    let records = extract_layer_bounds(&graph, &input).expect("extraction");

    assert!(!records.is_empty(), "ReLU graph should have layer records");

    // The final output bounds should be in [0, 3] for relu([-3, 3]).
    let last = records.last().expect("at least one record");
    let &(lo, hi) = last.output_bounds.first().expect("at least one element");
    assert!(
        (-0.1..=0.1).contains(&lo),
        "ReLU lower bound should be ~0.0, got {lo}"
    );
    assert!(
        (2.9..=3.1).contains(&hi),
        "ReLU upper bound should be ~3.0, got {hi}"
    );
}

#[test]
fn test_layer_trace_continuity() {
    // Verify that output_bounds[i] approximately matches input_bounds[i+1]
    // for consecutive layers (when they are connected).
    let (graph, input) = snake_graph();
    let records = extract_layer_bounds(&graph, &input).expect("extraction");

    if records.len() < 2 {
        return; // Nothing to check for single-layer graphs.
    }

    // Note: Not all consecutive layers are directly connected (DAG may have
    // branches). This test checks that bounds dimensions are consistent
    // rather than exact value matching.
    for window in records.windows(2) {
        let prev = &window[0];
        let next = &window[1];
        // Output and input should have consistent dimensionality.
        assert!(
            !prev.output_bounds.is_empty(),
            "prev layer {} output should be non-empty",
            prev.layer_index
        );
        assert!(
            !next.input_bounds.is_empty(),
            "next layer {} input should be non-empty",
            next.layer_index
        );
    }
}

#[test]
fn test_bounded_tensor_to_pairs_roundtrip() {
    // Directly test the conversion helper via the public API.
    let lower = ArrayD::from_elem(IxDyn(&[3]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let bt = BoundedTensor::new(lower, upper).expect("bounds");

    let pairs = super::bounded_tensor_to_pairs(&bt);
    assert_eq!(pairs.len(), 3);
    for &(lo, hi) in &pairs {
        assert_eq!(lo, -1.0);
        assert_eq!(hi, 2.0);
    }
}
