// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `traced_forward()` — the 3-step composite op tracing helper
//! used by all nn layers (Linear, Conv, Embedding, normalization, etc.).
//!
//! `traced_forward(inputs, op, compute)`:
//! 1. Suppresses tracing during `compute` (prevents sub-op double-recording)
//! 2. Executes `compute` to get the result tensor
//! 3. Records the composite `op` in the trace graph with input IDs
//!
//! These tests verify each step independently and in combination.

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

fn t1d(data: &[f32]) -> DynTensor {
    DynTensor::new(data, &[data.len()], &cpu()).expect("valid")
}

// -- Step 1: Trace suppression during compute ---------------------------------

/// During `traced_forward`, the `compute` closure runs with trace suppression.
/// Any ops executed inside `compute` must NOT be recorded as separate nodes.
#[test]
fn test_traced_forward_suppresses_sub_ops() {
    let a = t1d(&[1.0, 2.0, 3.0]);

    let (_, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[3], DType::F32).expect("valid");
        a.set_trace_id(id_a);

        // traced_forward should suppress the internal add from being recorded
        let result = traced_forward(
            &[&a],
            || Ok(TraceOp::Relu), // composite op
            || {
                // This add is a "sub-op" that should be suppressed
                let inner = a.add(&t1d(&[10.0, 10.0, 10.0]))?;
                Ok(inner)
            },
        )?;
        Ok(result)
    })
    .expect("valid");

    // Graph should have: 1 Input + 1 Relu (the composite op)
    // The inner Add should NOT appear
    assert_eq!(
        graph.len(),
        2,
        "expected 2 nodes (Input + Relu composite), got {}. Sub-op Add leaked through suppression.",
        graph.len()
    );

    let last = graph.output_node().expect("valid");
    assert!(
        matches!(last.op(), TraceOp::Relu),
        "last node should be Relu composite, got {:?}",
        last.op()
    );
}

// -- Step 2: Compute result is correct ----------------------------------------

/// `traced_forward` must return the actual computation result, not a placeholder.
#[test]
fn test_traced_forward_returns_compute_result() {
    let a = t1d(&[2.0, 4.0, 6.0]);

    let (output, _) = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[3], DType::F32).expect("valid");
        a.set_trace_id(id_a);

        let result = traced_forward(
            &[&a],
            || Ok(TraceOp::Relu),
            || {
                // Double each element
                let doubled = a.mul(&t1d(&[2.0, 2.0, 2.0]))?;
                Ok(doubled)
            },
        )?;
        Ok(result)
    })
    .expect("valid");

    let values = output.to_flat_vec::<f32>().expect("valid");
    assert_eq!(values, vec![4.0, 8.0, 12.0], "compute result incorrect");
}

// -- Step 3: Op recording and ID assignment -----------------------------------

/// After compute, `traced_forward` records the composite op and assigns
/// a trace ID to the result tensor.
#[test]
fn test_traced_forward_assigns_trace_id() {
    let a = t1d(&[1.0, 2.0]);

    let (output, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[2], DType::F32).expect("valid");
        a.set_trace_id(id_a);

        let result = traced_forward(&[&a], || Ok(TraceOp::Sigmoid), || Ok(t1d(&[0.5, 0.5])))?;
        Ok(result)
    })
    .expect("valid");

    // The output should have a trace ID
    assert!(
        output.trace_id().is_some(),
        "traced_forward must assign trace ID to result"
    );

    // The trace ID should match the output node in the graph
    let output_id = output.trace_id().expect("valid");
    let output_node = graph.node(output_id);
    assert!(
        output_node.is_some(),
        "trace ID on result must exist in graph"
    );
    assert!(
        matches!(output_node.expect("valid").op(), TraceOp::Sigmoid),
        "recorded op should be Sigmoid"
    );
}

/// The recorded composite op should list the correct input node IDs.
#[test]
fn test_traced_forward_records_correct_input_ids() {
    let a = t1d(&[1.0]);
    let b = t1d(&[2.0]);

    let (_, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[1], DType::F32).expect("valid");
        a.set_trace_id(id_a);
        let id_b = record_input(&[1], DType::F32).expect("valid");
        b.set_trace_id(id_b);

        let result = traced_forward(&[&a, &b], || Ok(TraceOp::Add), || a.add(&b))?;
        Ok(result)
    })
    .expect("valid");

    assert_eq!(graph.len(), 3, "expected Input, Input, Add");
    let add_node = graph.output_node().expect("valid");
    assert_eq!(
        add_node.inputs().len(),
        2,
        "composite Add should have 2 input references"
    );

    // Input IDs should reference the two Input nodes
    let input_nodes = graph.input_nodes();
    assert_eq!(input_nodes.len(), 2);
    let expected_ids: Vec<NodeId> = input_nodes.iter().map(|n| n.id()).collect();
    assert_eq!(add_node.inputs(), &expected_ids);
}

// -- No-tracing path ----------------------------------------------------------

/// When tracing is NOT active, `traced_forward` runs compute directly
/// without suppression, recording, or ID assignment.
#[test]
fn test_traced_forward_noop_without_tracing() {
    let a = t1d(&[1.0, 2.0, 3.0]);

    // Call traced_forward outside of trace_graph — should behave like plain compute
    let result = traced_forward(
        &[&a],
        || Ok(TraceOp::Relu),
        || {
            let r = a.add(&t1d(&[10.0, 10.0, 10.0]))?;
            Ok(r)
        },
    )
    .expect("valid");

    // Result should be correct
    let values = result.to_flat_vec::<f32>().expect("valid");
    assert_eq!(values, vec![11.0, 12.0, 13.0]);

    // No trace ID should be set
    assert!(
        result.trace_id().is_none(),
        "no trace ID should be assigned outside trace_graph"
    );
}

// -- Error propagation --------------------------------------------------------

/// When `compute` returns Err, `traced_forward` propagates the error
/// and does NOT record anything in the graph.
#[test]
fn test_traced_forward_propagates_compute_error() {
    let a = t1d(&[1.0]);

    let result = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[1], DType::F32).expect("valid");
        a.set_trace_id(id_a);

        let err_result = traced_forward(
            &[&a],
            || Ok(TraceOp::Relu),
            || Err(TensorError::Unsupported("test error".into())),
        );
        assert!(err_result.is_err(), "error should propagate");

        // Return a valid tensor so trace_graph doesn't fail
        Ok(a)
    });
    assert!(result.is_ok());
}

// -- Missing input trace IDs --------------------------------------------------

/// During active tracing, if an input tensor is missing its trace ID,
/// `traced_forward` auto-registers it as ConstantWeight (since [U]131).
/// Verify the auto-registration succeeds and produces a traced output.
#[test]
fn test_traced_forward_auto_registers_untraced_input_as_constant_weight() {
    let a = t1d(&[1.0]); // No trace ID set

    let result = trace_graph(|| {
        // a has no trace ID — auto-registered as ConstantWeight
        let traced_result = traced_forward(&[&a], || Ok(TraceOp::Relu), || Ok(t1d(&[1.0])));

        // Should succeed: untraced input auto-registered as ConstantWeight.
        let output = traced_result
            .expect("traced_forward should succeed with auto-registered ConstantWeight input");
        assert!(
            output.trace_id().is_some(),
            "output should have a trace ID after traced_forward"
        );
        Ok(output)
    });
    assert!(result.is_ok());
}

// -- Output shape recording ---------------------------------------------------

/// The recorded node should capture the output tensor's shape and dtype.
#[test]
fn test_traced_forward_records_output_shape_and_dtype() {
    let a = t1d(&[1.0, 2.0, 3.0, 4.0]);

    let (_, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[4], DType::F32).expect("valid");
        a.set_trace_id(id_a);

        let result = traced_forward(
            &[&a],
            || Ok(TraceOp::Relu),
            || {
                // Return a 2x2 tensor (different shape from input)
                Ok(DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).expect("valid"))
            },
        )?;
        Ok(result)
    })
    .expect("valid");

    let relu_node = graph.output_node().expect("valid");
    assert_eq!(
        relu_node.output_shape(),
        &[2, 2],
        "recorded shape should match output tensor"
    );
    assert_eq!(relu_node.output_dtype(), DType::F32);
}
