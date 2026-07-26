// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "verify")]

//! Public API surface tests for the reusable traced producer boundary.
//!
//! These tests prove downstream callers can import and use the owned
//! NY producer boundary via `nn::verify` without depending on
//! `nn-verify` directly. The surface is intentionally limited to reusable
//! graph-build artifacts over a traced `ComputationGraph`; it is not presented
//! as a proof-powered compiler.

use nn::trace::{record_input, trace_graph, ComputationGraph};
use nn::verify::{
    trace_to_graph_model, trace_to_graph_model_multi_input_with_boundary,
    trace_to_graph_model_with_boundary, GraphBuildInputs, GraphModel, TraceGraphBoundaryResult,
    TraceTranslateResult, VerifyError,
};
use nn::{DType, Device, DynTensor};

fn cpu() -> Device {
    Device::Cpu
}

type BoundaryHelper = fn(&ComputationGraph) -> Result<TraceGraphBoundaryResult, VerifyError>;
type NarrowHelper = fn(&ComputationGraph) -> Result<TraceTranslateResult, VerifyError>;

#[test]
fn verify_root_reexports_narrow_and_boundary_helpers() {
    let _: NarrowHelper = trace_to_graph_model;
    let _: BoundaryHelper = trace_to_graph_model_with_boundary;
    let _: BoundaryHelper = trace_to_graph_model_multi_input_with_boundary;

    let _ = GraphModel::graph_build_inputs;
    let _ = GraphModel::build_graph_network;
    let _ = TraceGraphBoundaryResult::graph_build_inputs;
    let _ = TraceGraphBoundaryResult::build_graph_network;
}

#[test]
fn single_input_boundary_is_usable_from_nn_verify() {
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
        trace_to_graph_model_with_boundary(&graph).expect("single-input boundary should build");
    let _: &GraphModel = &boundary.graph_model;
    let build_inputs: GraphBuildInputs<'_> = boundary.graph_build_inputs();

    assert_eq!(build_inputs.inputs.len(), 1);
    assert_eq!(build_inputs.inputs[0].shape, vec![2, 2]);
    assert_eq!(
        build_inputs.outputs[0].name, boundary.graph_model.network.outputs[0].name,
        "borrowed outputs should line up with the owned producer contract"
    );
    assert_eq!(
        build_inputs.layers.len(),
        boundary.graph_model.network.layers.len(),
        "borrowed layer view should expose the full owned layer list"
    );
    assert!(
        boundary
            .graph_model
            .network
            .name
            .starts_with("trace_graph_"),
        "boundary should expose the owned producer graph model"
    );

    let rebuilt = boundary
        .build_graph_network()
        .expect("owned producer boundary should rebuild without retracing");
    assert_eq!(rebuilt.num_nodes(), boundary.graph.num_nodes());
}

#[test]
fn single_input_verify_boundary_rejects_multi_input_graph_from_root_api() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[0.5, 0.6, 0.7], &[1, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[1, 3], DType::F32).unwrap();
        a.set_trace_id(id_a);

        let mut b = b.clone();
        let id_b = record_input(&[1, 3], DType::F32).unwrap();
        b.set_trace_id(id_b);

        let y = a.add(&b)?;
        Ok(y)
    })
    .unwrap();

    let err = trace_to_graph_model_with_boundary(&graph)
        .expect_err("single-input boundary helper should reject independent inputs");
    let msg = err.to_string();
    assert!(
        msg.contains("multiple variable inputs")
            || msg.contains("trace_to_graph_model_multi_input"),
        "root verify boundary should point callers at the multi-input helper, got: {msg}"
    );
}

#[test]
fn multi_input_boundary_is_usable_from_nn_verify() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[0.5, 0.6, 0.7], &[1, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[1, 3], DType::F32).unwrap();
        a.set_trace_id(id_a);

        let mut b = b.clone();
        let id_b = record_input(&[1, 3], DType::F32).unwrap();
        b.set_trace_id(id_b);

        let y = a.add(&b)?;
        Ok(y)
    })
    .unwrap();

    let boundary = trace_to_graph_model_multi_input_with_boundary(&graph)
        .expect("multi-input boundary should build");
    let build_inputs: GraphBuildInputs<'_> = boundary.graph_build_inputs();

    assert_eq!(build_inputs.inputs.len(), 1);
    assert_eq!(build_inputs.inputs[0].name, "multi_in");
    assert_eq!(build_inputs.inputs[0].shape, vec![6]);
    assert_eq!(
        boundary.graph_model.tensor_shapes.get("multi_in"),
        Some(&vec![6]),
        "stacked producer input should remain visible through the public boundary"
    );
    assert_eq!(
        build_inputs.outputs[0].name, boundary.graph_model.network.outputs[0].name,
        "borrowed multi-input outputs should line up with the owned producer contract"
    );
}
