// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace tests for compound/parametric DynTensor ops (elu, leaky_relu).
//!
//! Regression tests for #2346: elu() and leaky_relu() CPU paths did not
//! record trace nodes. Now both record composite TraceOp nodes on all
//! code paths (CPU and GPU decomposed).

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

fn t1d(data: &[f32]) -> DynTensor {
    DynTensor::new(data, &[data.len()], &cpu()).expect("valid 1D tensor")
}

#[test]
fn test_elu_trace_records_composite_node() {
    let x = t1d(&[-1.0, 0.0, 1.0, 2.0]);

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[4], DType::F32).expect("record input");
        x.set_trace_id(id);
        let out = x.elu(1.0)?;
        Ok(out)
    })
    .expect("trace_graph should succeed");

    // Output node should be the composite Elu, not a decomposed primitive.
    let output = graph.output_node().expect("graph should have output");
    assert!(
        matches!(output.op(), TraceOp::Elu { alpha } if (*alpha - 1.0).abs() < 1e-12),
        "expected TraceOp::Elu with alpha=1.0, got {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[4]);
    assert_eq!(output.inputs().len(), 1);
}

#[test]
fn test_leaky_relu_trace_records_composite_node() {
    let x = t1d(&[-2.0, -1.0, 0.0, 1.0]);

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[4], DType::F32).expect("record input");
        x.set_trace_id(id);
        let out = x.leaky_relu(0.01)?;
        Ok(out)
    })
    .expect("trace_graph should succeed");

    let output = graph.output_node().expect("graph should have output");
    assert!(
        matches!(output.op(), TraceOp::LeakyRelu { slope } if (*slope - 0.01).abs() < 1e-12),
        "expected TraceOp::LeakyRelu with slope=0.01, got {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[4]);
    assert_eq!(output.inputs().len(), 1);
}

// -- contiguous() trace propagation (#2357) -----------------------------------

#[test]
fn test_contiguous_propagates_trace_id() {
    let x = t1d(&[1.0, 2.0, 3.0]);

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[3], DType::F32).expect("record input");
        x.set_trace_id(id);
        // contiguous() is identity — should preserve trace_node_id, not drop it.
        let y = x.contiguous()?;
        // Verify the trace ID survived by using y in a downstream op.
        let z = y.relu()?;
        Ok(z)
    })
    .expect("trace_graph should succeed");

    // If contiguous() dropped the trace ID, relu() would fail with
    // "input 0 has no trace ID during active trace".
    assert_eq!(result.dims(), &[3]);
    let output = graph.output_node().expect("graph should have output");
    assert!(
        matches!(output.op(), TraceOp::Relu),
        "expected TraceOp::Relu, got {:?}",
        output.op()
    );
    // Graph: input -> relu (contiguous is invisible, just ID propagation)
    assert_eq!(graph.len(), 2);
}
