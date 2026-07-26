// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the tightening loop orchestrator.

use super::*;

use ny_api::BoundedTensor;
use ny_propagate::GraphNetwork;
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::Device;
use ndarray::{ArrayD, IxDyn};

use crate::trace_to_graph::trace_to_graph_model;

/// Trace a Linear(2->2) + ReLU model and return (GraphNetwork, BoundedTensor).
fn build_traced_linear_relu() -> (GraphNetwork, BoundedTensor) {
    let weight =
        DynTensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[2, 2], &Device::Cpu).expect("weight");
    let linear = Linear::new(weight, None).expect("linear");
    let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).expect("input");

    let (_output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let h = linear.forward(&traced)?;
        h.relu()
    })
    .expect("trace");

    let network = trace_to_graph_model(&graph).expect("translate").graph;

    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
    let bounds = BoundedTensor::new(lower, upper).expect("bounds");

    (network, bounds)
}

#[test]
fn test_tightening_config_defaults() {
    let config = TighteningConfig::default();
    assert_eq!(config.max_iterations, 5);
    assert!((config.convergence_threshold - 0.01).abs() < 1e-6);
    assert!(config.ay_candidates_enabled);
    assert_eq!(config.model_name, "unknown");
}

#[test]
fn test_tightening_config_builder() {
    let config = TighteningConfig::new("test_model")
        .with_max_iterations(3)
        .with_convergence_threshold(0.05)
        .with_ay_candidates(false);
    assert_eq!(config.max_iterations, 3);
    assert!((config.convergence_threshold - 0.05).abs() < 1e-6);
    assert!(!config.ay_candidates_enabled);
    assert_eq!(config.model_name, "test_model");
}

#[test]
fn test_tightening_loop_simple_graph() {
    let (network, bounds) = build_traced_linear_relu();
    let config = TighteningConfig::new("linear_relu")
        .with_max_iterations(3)
        .with_ay_candidates(false);

    let result = run_tightening_loop(&network, &bounds, &config).expect("tightening loop");

    // At least one iteration should have run.
    assert!(result.iterations_run >= 1);
    assert!(!result.improvement_history.is_empty());

    // Output should be finite for this simple graph.
    assert!(result.achieved_finite_bounds());

    // Final width should be finite and non-negative.
    let width = result.final_max_width();
    assert!(width.is_finite());
    assert!(width >= 0.0);

    // For Linear(2->2) + ReLU with input [-1, 1], max output width
    // should be bounded.
    assert!(
        width < 100.0,
        "width {width} should be bounded for simple graph"
    );
}

#[test]
fn test_tightening_loop_converges_on_simple_graph() {
    let (network, bounds) = build_traced_linear_relu();
    let config = TighteningConfig::new("tight_model")
        .with_max_iterations(5)
        .with_convergence_threshold(0.5);

    let result = run_tightening_loop(&network, &bounds, &config).expect("tightening loop");

    // With a simple graph, should converge quickly because IBP alone
    // produces tight bounds (no explosion points, output is finite).
    assert!(result.converged, "should converge on a simple graph");
}

#[test]
fn test_tightening_loop_max_iterations_respected() {
    let (network, bounds) = build_traced_linear_relu();
    let config = TighteningConfig::new("bounded_iters").with_max_iterations(2);

    let result = run_tightening_loop(&network, &bounds, &config).expect("tightening loop");
    assert!(
        result.iterations_run <= 2,
        "should not exceed max_iterations"
    );
}

#[test]
fn test_tightening_loop_zero_iterations_errors() {
    let (network, bounds) = build_traced_linear_relu();
    let config = TighteningConfig::new("zero_iters").with_max_iterations(0);

    let result = run_tightening_loop(&network, &bounds, &config);
    assert!(
        result.is_err(),
        "zero max_iterations should produce an error"
    );
}

#[test]
fn test_tightening_step_iteration_one_has_no_improvement() {
    let (network, bounds) = build_traced_linear_relu();
    let config = TighteningConfig::new("step_check").with_max_iterations(1);

    let result = run_tightening_loop(&network, &bounds, &config).expect("tightening loop");
    assert_eq!(result.improvement_history.len(), 1);

    let step = &result.improvement_history[0];
    assert_eq!(step.iteration, 1);
    assert!(
        step.improvement_ratio.is_none(),
        "iteration 1 has no baseline"
    );
}

#[test]
fn test_tightening_result_best_improvement_single_iter() {
    let (network, bounds) = build_traced_linear_relu();
    let config = TighteningConfig::new("best_improvement").with_max_iterations(1);

    let result = run_tightening_loop(&network, &bounds, &config).expect("tightening loop");
    // Only 1 iteration -- no improvement to compare.
    assert!(result.best_improvement().is_none());
}

#[test]
fn test_ay_candidates_disabled() {
    let (network, bounds) = build_traced_linear_relu();
    let config = TighteningConfig::new("no_ay")
        .with_max_iterations(1)
        .with_ay_candidates(false);

    let result = run_tightening_loop(&network, &bounds, &config).expect("tightening loop");
    assert!(
        result.ay_candidate_ranges.is_empty(),
        "ay candidates should be empty when disabled"
    );
}

#[test]
fn test_tightening_loop_records_layer_bounds() {
    let (network, bounds) = build_traced_linear_relu();
    let config = TighteningConfig::new("layer_bounds").with_max_iterations(1);

    let result = run_tightening_loop(&network, &bounds, &config).expect("tightening loop");
    assert!(
        !result.final_layer_bounds.is_empty(),
        "should have per-layer bound records"
    );
}

#[test]
fn test_tightening_step_fields_populated() {
    let (network, bounds) = build_traced_linear_relu();
    let config = TighteningConfig::new("field_check").with_max_iterations(2);

    let result = run_tightening_loop(&network, &bounds, &config).expect("tightening loop");

    for step in &result.improvement_history {
        assert!(step.iteration >= 1);
        // max_output_width should be finite for this simple graph.
        assert!(step.max_output_width.is_finite());
        // crown_coverage is between 0.0 and 1.0.
        assert!(step.crown_coverage >= 0.0 && step.crown_coverage <= 1.0);
    }
}
