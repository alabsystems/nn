// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Boundary-condition regression tests for the trace-to-graph model translator.
//!
//! These tests exercise edge cases identified during algorithm audit:
//! - Clamp with normal f64 bounds produces valid IBP output
//! - Clamp with extreme f64 values (f64::MAX) that overflow f32
//! - Clamp with NaN/Inf bounds
//!
//! The extreme-value tests document current behavior and will serve as
//! regression tests after the checked_f64_to_f32 fix lands.
//!
//! Part of #2080 (Prover algorithm audit, trace-translator boundary checks).

use super::common::{assert_bounds_valid, uniform_bounds};
use ny_build::{
    build_graph_network, GraphBuildInputs, GraphNetworkOptions, MissingOutputPolicy, TensorSpec,
};
use ny_core::LayerType;
use nn_core::dyn_tensor::trace::{
    record_input, trace_graph, ComputationGraph, TraceNode, TraceOp,
};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_verify::{
    trace_to_graph_model, trace_to_graph_model_multi_input_with_boundary,
    trace_to_graph_model_with_boundary,
};

fn cpu() -> Device {
    Device::Cpu
}

fn strict_output_options() -> GraphNetworkOptions {
    GraphNetworkOptions {
        missing_output_policy: MissingOutputPolicy::Error,
        ..GraphNetworkOptions::default()
    }
}

// ---------------------------------------------------------------------------
// Clamp with normal f64 bounds — happy path
// ---------------------------------------------------------------------------

/// Clamp with normal bounds translates correctly and produces valid IBP output.
///
/// Regression: verifies that f64→f32 cast for normal values (-1.5, 2.5)
/// produces correct Clip layer attributes.
#[test]
fn test_clamp_normal_bounds_succeeds() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.clamp(-1.5, 2.5)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("normal clamp should translate")
        .graph;

    let input_bounds = uniform_bounds(&[2, 2], 5.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -1.51, "clamped lower should be near -1.5, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 2.51, "clamped upper should be near 2.5, got {v}");
    }
}

// ---------------------------------------------------------------------------
// Clamp with f64::MAX — documents f32 overflow behavior
// ---------------------------------------------------------------------------

/// Clamp with f64::MAX min: documents behavior for f64→f32 overflow.
///
/// f64::MAX cast to f32 produces f32::INFINITY. Depending on whether
/// checked_f64_to_f32 is present, this either:
/// - Succeeds with Inf attribute (unchecked path — current HEAD)
/// - Fails with overflow error (checked path — after fix)
///
/// This test accepts either outcome, documenting the boundary.
#[test]
fn test_clamp_f64_max_min_boundary() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2, 3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "clamp_0".into(),
            TraceOp::Clamp {
                min: Some(f64::MAX),
                max: None,
            },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);

    // Accept either error (checked path) or success (unchecked path).
    // After checked_f64_to_f32 lands, change to expect_err.
    match trace_to_graph_model(&graph).map(|r| r.graph) {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("overflows f32") || msg.contains("non-finite"),
                "error should mention overflow, got: {msg}"
            );
        }
        Ok(gn) => {
            // Unchecked path: translation succeeded but the Clip attribute
            // contains f32::INFINITY. IBP may or may not handle this gracefully.
            // Just verify the graph was built without panic.
            assert!(gn.num_nodes() > 0, "graph should have at least one node");
        }
    }
}

/// Clamp with NaN min: must fail or produce detectable corruption.
///
/// NaN in Clip attributes is always wrong — NY comparison with NaN
/// returns false, making bounds vacuous.
#[test]
fn test_clamp_nan_min_boundary() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "clamp_0".into(),
            TraceOp::Clamp {
                min: Some(f64::NAN),
                max: Some(1.0),
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);

    // Accept either error (checked path) or success (unchecked path).
    // After checked_f64_to_f32 lands, change to expect_err.
    match trace_to_graph_model(&graph).map(|r| r.graph) {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("non-finite") || msg.contains("NaN"),
                "error should mention non-finite, got: {msg}"
            );
        }
        Ok(gn) => {
            // Unchecked path: NaN passed through to Clip attribute.
            // NaN as f32 is still NaN. The graph builds but bounds may
            // be corrupted. Document this as accepted (pre-fix behavior).
            assert!(gn.num_nodes() > 0);
        }
    }
}

/// Clamp with Inf max: must fail or produce detectable corruption.
#[test]
fn test_clamp_inf_max_boundary() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "clamp_0".into(),
            TraceOp::Clamp {
                min: Some(0.0),
                max: Some(f64::INFINITY),
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);

    match trace_to_graph_model(&graph).map(|r| r.graph) {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("non-finite"),
                "error should mention non-finite, got: {msg}"
            );
        }
        Ok(gn) => {
            assert!(gn.num_nodes() > 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Clamp with negative overflow
// ---------------------------------------------------------------------------

/// Clamp with -f64::MAX min: overflows to -f32::INFINITY.
#[test]
fn test_clamp_neg_f64_max_boundary() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![2],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "clamp_0".into(),
            TraceOp::Clamp {
                min: Some(-f64::MAX),
                max: Some(1.0),
            },
            vec![0],
            vec![2],
            DType::F32,
        ),
    ]);

    match trace_to_graph_model(&graph).map(|r| r.graph) {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("overflows f32") || msg.contains("non-finite"),
                "error should mention overflow, got: {msg}"
            );
        }
        Ok(gn) => {
            assert!(gn.num_nodes() > 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Clamp with None bounds (one-sided clamp)
// ---------------------------------------------------------------------------

/// One-sided clamp (min only, no max) translates correctly.
#[test]
fn test_clamp_min_only_succeeds() {
    let x = DynTensor::new(&[-2.0, -1.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        // clamp_min records Clamp { min: Some(0.0), max: None }
        let y = x.clamp_min(0.0)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("one-sided clamp_min should translate")
        .graph;

    let input_bounds = uniform_bounds(&[2, 2], 3.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, _hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(
            v >= -0.01,
            "clamp_min(0) lower bound should be >= 0, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// Owned producer boundary
// ---------------------------------------------------------------------------

#[test]
fn test_trace_to_graph_boundary_exposes_owned_graph_model() {
    let x = DynTensor::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.relu()?;
        Ok(y)
    })
    .unwrap();

    let boundary =
        trace_to_graph_model_with_boundary(&graph).expect("boundary translation should succeed");
    let build_inputs = boundary.graph_build_inputs();

    assert!(
        boundary
            .graph_model
            .network
            .name
            .starts_with("trace_graph_"),
        "owned graph model should retain a trace-derived network name"
    );
    assert_eq!(
        build_inputs.inputs.len(),
        1,
        "single-input translation should expose one input spec"
    );
    assert_eq!(
        build_inputs.inputs[0].shape,
        vec![2, 2],
        "input spec shape should match the traced input"
    );
    assert_eq!(
        build_inputs.layers.len(),
        boundary.graph_model.network.layers.len(),
        "borrowed GraphBuildInputs should expose the owned layer list"
    );
    assert!(
        boundary
            .graph_model
            .tensor_shapes
            .contains_key(build_inputs.outputs[0].name.as_str()),
        "owned graph model should retain output tensor shape metadata"
    );

    let rebuilt = boundary
        .build_graph_network()
        .expect("owned graph model should rebuild a graph network");
    assert_eq!(
        rebuilt.num_nodes(),
        boundary.graph.num_nodes(),
        "rebuilding from the owned boundary should match the eagerly built graph"
    );
}

#[test]
fn test_trace_to_graph_boundary_borrowed_inputs_support_manual_gamma_build_rebuild() {
    let x = DynTensor::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.relu()?;
        Ok(y)
    })
    .unwrap();

    let boundary =
        trace_to_graph_model_with_boundary(&graph).expect("boundary translation should succeed");
    let build_inputs = boundary.graph_build_inputs();

    let rebuilt = build_graph_network(&build_inputs, strict_output_options())
        .expect("borrowed GraphBuildInputs should be enough for manual rebuild");
    assert_eq!(
        rebuilt.num_nodes(),
        boundary.graph.num_nodes(),
        "manual gamma-build rebuild should match the eager graph network"
    );
}

#[test]
fn test_trace_to_graph_boundary_borrowed_inputs_fail_closed_on_wrong_output_name() {
    let x = DynTensor::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.relu()?;
        Ok(y)
    })
    .unwrap();

    let boundary =
        trace_to_graph_model_with_boundary(&graph).expect("boundary translation should succeed");
    let build_inputs = boundary.graph_build_inputs();

    let wrong_outputs = vec![TensorSpec {
        name: "definitely_missing_output".to_string(),
        shape: build_inputs.outputs[0].shape.clone(),
        dtype: build_inputs.outputs[0].dtype,
    }];
    let wrong_build_inputs = GraphBuildInputs {
        layers: build_inputs.layers,
        inputs: build_inputs.inputs,
        outputs: &wrong_outputs,
        weights: build_inputs.weights,
        tensor_producer: build_inputs.tensor_producer,
        constant_tensors: build_inputs.constant_tensors,
        tensor_shapes: build_inputs.tensor_shapes,
    };

    let err = build_graph_network(&wrong_build_inputs, strict_output_options())
        .expect_err("borrowed boundary inputs should fail closed on wrong output names");
    let msg = err.to_string();
    assert!(
        msg.contains("output") || msg.contains("resolution") || msg.contains("missing"),
        "strict output resolution error should mention the missing output, got: {msg}"
    );
}

#[test]
fn test_trace_to_graph_multi_input_boundary_preserves_stacked_input_contract() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[0.5, 0.6, 0.7, 0.8], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[2, 2], DType::F32).unwrap();
        a.set_trace_id(id_a);

        let mut b = b.clone();
        let id_b = record_input(&[2, 2], DType::F32).unwrap();
        b.set_trace_id(id_b);

        let y = a.add(&b)?;
        Ok(y)
    })
    .unwrap();

    let boundary = trace_to_graph_model_multi_input_with_boundary(&graph)
        .expect("multi-input boundary translation should succeed");
    let build_inputs = boundary.graph_build_inputs();

    assert_eq!(
        build_inputs.inputs.len(),
        1,
        "stacked multi-input translation should still expose one graph input"
    );
    assert_eq!(
        build_inputs.inputs[0].name, "multi_in",
        "the owned producer contract should retain the stacked input tensor"
    );
    assert_eq!(
        build_inputs.inputs[0].shape,
        vec![8],
        "two 2x2 inputs should stack into one flat 8-element producer input"
    );
    assert_eq!(
        boundary.graph_model.tensor_shapes.get("multi_in"),
        Some(&vec![8]),
        "tensor shape metadata should retain the stacked producer input"
    );

    let slice_count = boundary
        .graph_model
        .network
        .layers
        .iter()
        .filter(|layer| layer.layer_type == LayerType::Slice)
        .count();
    let reshape_count = boundary
        .graph_model
        .network
        .layers
        .iter()
        .filter(|layer| layer.layer_type == LayerType::Reshape)
        .count();
    assert_eq!(
        slice_count, 2,
        "multi-input boundary should retain one Slice per variable input"
    );
    assert_eq!(
        reshape_count, 2,
        "multi-input boundary should retain one Reshape per variable input"
    );

    let rebuilt_from_borrowed = build_graph_network(&build_inputs, strict_output_options())
        .expect("borrowed multi-input GraphBuildInputs should rebuild");
    assert_eq!(
        rebuilt_from_borrowed.num_nodes(),
        boundary.graph.num_nodes(),
        "borrowed multi-input producer data should rebuild the eager graph"
    );

    let rebuilt = boundary
        .build_graph_network()
        .expect("owned multi-input graph model should rebuild");
    assert_eq!(
        rebuilt.num_nodes(),
        boundary.graph.num_nodes(),
        "rebuilding from the owned multi-input boundary should match the eager graph"
    );
}
