// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DynTensor computation graph tracing.

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

fn t1d(data: &[f32]) -> DynTensor {
    DynTensor::new(data, &[data.len()], &cpu()).unwrap()
}

fn t2d(data: &[f32], rows: usize, cols: usize) -> DynTensor {
    DynTensor::new(data, &[rows, cols], &cpu()).unwrap()
}

// -- Basic tracing lifecycle --------------------------------------------------

#[test]
fn test_trace_empty_closure() {
    let ((), graph) = trace_graph(|| Ok(())).unwrap();
    assert!(graph.is_empty());
    assert!(graph.output_node().is_none());
}

#[test]
fn test_tracing_inactive_by_default() {
    assert!(!is_tracing());
}

#[test]
fn test_nested_tracing_returns_error() {
    let result = trace_graph(|| {
        // Attempt nested tracing
        let inner = trace_graph(|| Ok(()));
        assert!(inner.is_err());
        Ok(())
    });
    assert!(result.is_ok());
}

// -- Input recording ----------------------------------------------------------

#[test]
fn test_record_input() {
    let ((), graph) = trace_graph(|| {
        let id = record_input(&[3, 4], DType::F32);
        assert!(id.is_some());
        Ok(())
    })
    .unwrap();
    assert_eq!(graph.len(), 1);
    let node = &graph.nodes()[0];
    assert!(matches!(node.op(), TraceOp::Input));
    assert_eq!(node.output_shape(), &[3, 4]);
    assert_eq!(node.output_dtype(), DType::F32);
    assert!(node.name().starts_with("input_"));
}

// -- Binary ops tracing -------------------------------------------------------

#[test]
fn test_trace_add() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[3], DType::F32).unwrap();
        a.set_trace_id(id_a);
        let id_b = record_input(&[3], DType::F32).unwrap();
        b.set_trace_id(id_b);
        let c = a.add(&b)?;
        Ok(c)
    })
    .unwrap();

    // 2 inputs + 1 add = 3 nodes
    assert_eq!(graph.len(), 3);

    // Verify the add node
    let output = graph.output_node().unwrap();
    assert!(matches!(output.op(), TraceOp::Add));
    assert_eq!(output.output_shape(), &[3]);
    assert_eq!(output.inputs().len(), 2);

    // Verify result is correct (tracing does not change computation)
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_trace_mul() {
    let a = t1d(&[2.0, 3.0]);
    let b = t1d(&[4.0, 5.0]);

    let ((), graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        let id_b = record_input(&[2], DType::F32).unwrap();
        b.set_trace_id(id_b);
        let _c = a.mul(&b)?;
        Ok(())
    })
    .unwrap();

    assert_eq!(graph.len(), 3);
    let output = graph.output_node().unwrap();
    assert!(matches!(output.op(), TraceOp::Mul));
}

// -- Unary ops tracing --------------------------------------------------------

#[test]
fn test_trace_relu() {
    let a = t1d(&[-1.0, 0.0, 1.0]);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.relu()?;
        Ok(b)
    })
    .unwrap();

    // 1 input + 1 relu = 2 nodes
    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    assert!(matches!(output.op(), TraceOp::Relu));
    assert_eq!(output.output_shape(), &[3]);
    assert_eq!(output.inputs().len(), 1);

    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_trace_exp() {
    let a = t1d(&[0.0, 1.0]);

    let ((), graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[2], DType::F32).unwrap();
        a.set_trace_id(id);
        let _b = a.exp()?;
        Ok(())
    })
    .unwrap();

    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    assert!(matches!(output.op(), TraceOp::Exp));
}

#[test]
fn test_trace_neg() {
    let a = t1d(&[1.0, -2.0, 3.0]);

    let ((), graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[3], DType::F32).unwrap();
        a.set_trace_id(id);
        let _b = a.neg()?;
        Ok(())
    })
    .unwrap();

    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    assert!(matches!(output.op(), TraceOp::Neg));
}

// -- MatMul tracing -----------------------------------------------------------

#[test]
fn test_trace_matmul() {
    // [2, 3] x [3, 2] -> [2, 2]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2, 3], DType::F32).unwrap();
        a.set_trace_id(id_a);
        let id_b = record_input(&[3, 2], DType::F32).unwrap();
        b.set_trace_id(id_b);
        let c = a.matmul(&b)?;
        Ok(c)
    })
    .unwrap();

    // 2 inputs + 1 matmul = 3 nodes
    assert_eq!(graph.len(), 3);
    let output = graph.output_node().unwrap();
    assert!(matches!(output.op(), TraceOp::MatMul));
    assert_eq!(output.output_shape(), &[2, 2]);
    assert_eq!(output.inputs().len(), 2);

    // Verify computation is correct
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![22.0, 28.0, 49.0, 64.0]);
}

// -- Reduction tracing --------------------------------------------------------

#[test]
fn test_trace_sum_keepdim() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.sum_keepdim(1)?;
        Ok(b)
    })
    .unwrap();

    // 1 input + 1 reduce = 2 nodes
    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::ReduceSum { dim, keepdim } => {
            assert_eq!(*dim, 1);
            assert!(*keepdim);
        }
        other => panic!("expected ReduceSum, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[2, 1]);

    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![6.0, 15.0]);
}

#[test]
fn test_trace_mean_keepdim() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

    let ((), graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        a.set_trace_id(id);
        let _b = a.mean_keepdim(0)?;
        Ok(())
    })
    .unwrap();

    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::ReduceMean { dim, keepdim } => {
            assert_eq!(*dim, 0);
            assert!(*keepdim);
        }
        other => panic!("expected ReduceMean, got {other:?}"),
    }
}

// -- Multi-op graph -----------------------------------------------------------

#[test]
fn test_trace_multi_op_graph() {
    // Simulate a simple linear + relu: relu(x @ w + b)
    let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let w = t2d(&[0.5, 0.5, 0.5, 0.5], 2, 2);
    let b = t1d(&[-1.0, 1.0]);

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let mut w = w.clone();
        let mut b = b.clone();
        let id_x = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id_x);
        let id_w = record_input(&[2, 2], DType::F32).unwrap();
        w.set_trace_id(id_w);
        let id_b = record_input(&[2], DType::F32).unwrap();
        b.set_trace_id(id_b);

        // y = x @ w
        let y = x.matmul(&w)?;
        // z = y + b (broadcast)
        let z = y.add(&b)?;
        // out = relu(z)
        let out = z.relu()?;
        Ok(out)
    })
    .unwrap();

    // 3 inputs + matmul + add + relu = 6 nodes
    assert_eq!(graph.len(), 6);

    // Check node types in order
    let ops: Vec<&str> = graph
        .nodes()
        .iter()
        .map(|n| match n.op() {
            TraceOp::Input => "input",
            TraceOp::MatMul => "matmul",
            TraceOp::Add => "add",
            TraceOp::Relu => "relu",
            other => panic!("unexpected op: {other:?}"),
        })
        .collect();
    assert_eq!(
        ops,
        vec!["input", "input", "input", "matmul", "add", "relu"]
    );

    // Output is the relu node
    let output = graph.output_node().unwrap();
    assert!(matches!(output.op(), TraceOp::Relu));
}

// -- No tracing when inactive -------------------------------------------------

#[test]
fn test_no_trace_ids_outside_tracing() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c = a.add(&b).unwrap();
    // Outside tracing, no trace IDs are set
    assert!(c.trace_id().is_none());
}

// -- Graph inspection ---------------------------------------------------------

#[test]
fn test_graph_input_nodes() {
    let ((), graph) = trace_graph(|| {
        let id1 = record_input(&[3], DType::F32);
        let id2 = record_input(&[4], DType::F32);
        assert!(id1.is_some());
        assert!(id2.is_some());
        Ok(())
    })
    .unwrap();

    let inputs = graph.input_nodes();
    assert_eq!(inputs.len(), 2);
    assert!(inputs.iter().all(|n| matches!(n.op(), TraceOp::Input)));
}

#[test]
fn test_graph_node_by_id() {
    let (_, graph) = trace_graph(|| {
        let id = record_input(&[5], DType::BF16).unwrap();
        // Verify we can look up the node
        Ok(id)
    })
    .unwrap();

    let id = graph.nodes()[0].id();
    let node = graph.node(id).unwrap();
    assert_eq!(node.output_shape(), &[5]);
    assert_eq!(node.output_dtype(), DType::BF16);
}

// -- Name generation ----------------------------------------------------------

#[test]
fn test_name_generation_unique() {
    let a = t1d(&[1.0, 2.0]);

    let (_, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[2], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.relu()?;
        let c = b.relu()?;
        Ok(c)
    })
    .unwrap();

    let names: Vec<&str> = graph.nodes().iter().map(TraceNode::name).collect();
    // Each relu should get a unique name
    assert!(names.contains(&"relu_0"));
    assert!(names.contains(&"relu_1"));
}

// -- Error propagation --------------------------------------------------------

#[test]
fn test_trace_error_propagation() {
    let result: Result<((), _)> =
        trace_graph(|| Err(TensorError::InvalidShape("test error".into())));
    assert!(result.is_err());
    // Tracing should be cleaned up after error
    assert!(!is_tracing());
}

// -- Cleanup on panic ---------------------------------------------------------

#[test]
fn test_trace_cleanup_on_panic() {
    let result = std::panic::catch_unwind(|| {
        let _ = trace_graph(|| -> Result<()> { panic!("test panic") });
    });
    assert!(result.is_err()); // Panic propagated
    assert!(!is_tracing()); // But tracing was cleaned up
}

// -- validate_topology ---------------------------------------------------------

#[test]
fn test_validate_topology_valid_graph() {
    let ((), graph) = trace_graph(|| {
        let a = t1d(&[1.0, 2.0]);
        let b = t1d(&[3.0, 4.0]);
        let mut a = a;
        let mut b = b;
        let id_a = record_input(&[2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        let id_b = record_input(&[2], DType::F32).unwrap();
        b.set_trace_id(id_b);
        let _c = a.add(&b)?;
        Ok(())
    })
    .unwrap();
    assert!(graph.validate_topology().is_ok());
}

#[test]
fn test_validate_topology_bad_order() {
    // Build a graph manually where a node references a non-existent node.
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            1,
            "in_0".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "bad_op".into(),
            TraceOp::Relu,
            vec![999],
            vec![4],
            DType::F32,
        ),
    ]);
    let err = graph.validate_topology().unwrap_err();
    match &err {
        TensorError::TopologyError {
            node_name,
            index,
            missing_input,
        } => {
            assert_eq!(node_name, "bad_op");
            assert_eq!(*index, 1);
            assert_eq!(*missing_input, 999);
        }
        other => panic!("expected TopologyError, got: {other:?}"),
    }
}

// -- trace_input_ids missing trace ID detection (#2087 G1) ---------------------

#[test]
fn test_trace_input_ids_errors_on_untraced_input() {
    // Verifies G1 fix from #2087: trace_input_ids now returns Err when an
    // input tensor lacks a trace ID during active tracing, instead of
    // Since [U]131 (34d4af603), untraced tensors are auto-registered as
    // ConstantWeight nodes instead of causing an error. Verify the graph
    // contains the auto-registered ConstantWeight for tensor b.
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0]);

    let result = trace_graph(|| {
        let mut a = a.clone();
        let b = b.clone(); // b is NOT given a trace ID — auto-registered as ConstantWeight
        let id_a = record_input(&[2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        let c = a.add(&b)?;
        Ok(c)
    });

    // trace_graph succeeds: b is auto-registered as ConstantWeight.
    let (output, graph) = result.expect("trace should succeed with auto-registered ConstantWeight");
    assert_eq!(output.dims(), &[2]);

    // Verify the graph contains a ConstantWeight node for the untraced tensor.
    let has_constant_weight = graph
        .nodes()
        .iter()
        .any(|n| matches!(n.op(), TraceOp::ConstantWeight { .. }));
    assert!(
        has_constant_weight,
        "untraced tensor should be auto-registered as ConstantWeight in the graph"
    );
}

// Shape ops, cat, and softmax trace tests extracted to
// `trace_tests_shape_ops.rs` for 500-line compliance.
