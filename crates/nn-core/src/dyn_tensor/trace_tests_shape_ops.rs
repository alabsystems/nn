// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shape and structural op trace tests extracted from `trace_tests.rs`
//! for 500-line compliance.
//!
//! Covers: reshape, transpose, narrow, unsqueeze, squeeze, permute, cat,
//! softmax, log_softmax.

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

// -- Shape ops tracing --------------------------------------------------------

#[test]
fn test_trace_reshape() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.reshape([3, 2])?;
        Ok(b)
    })
    .unwrap();

    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Reshape { target_shape } => {
            assert_eq!(target_shape, &[3, 2]);
        }
        other => panic!("expected Reshape, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[3, 2]);
    assert_eq!(output.inputs().len(), 1);
    assert_eq!(result.dims(), &[3, 2]);
}

#[test]
fn test_trace_transpose() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.transpose(0, 1)?;
        Ok(b)
    })
    .unwrap();

    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Transpose { dim0, dim1 } => {
            assert_eq!(*dim0, 0);
            assert_eq!(*dim1, 1);
        }
        other => panic!("expected Transpose, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[3, 2]);
    assert_eq!(result.dims(), &[3, 2]);
}

#[test]
fn test_trace_narrow() {
    let a = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[5], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.narrow(0, 1, 3)?;
        Ok(b)
    })
    .unwrap();

    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Narrow { dim, start, length } => {
            assert_eq!(*dim, 0);
            assert_eq!(*start, 1);
            assert_eq!(*length, 3);
        }
        other => panic!("expected Narrow, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_trace_unsqueeze() {
    let a = t1d(&[1.0, 2.0, 3.0]);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.unsqueeze(0)?;
        Ok(b)
    })
    .unwrap();

    // 1 input + reshape (from unsqueeze internals) + unsqueeze = 3 nodes.
    // unsqueeze delegates to reshape which records Reshape, then unsqueeze
    // records Unsqueeze. The override via set_trace_id only changes which node
    // the result tensor points to — it does NOT remove the Reshape node.
    assert_eq!(graph.len(), 3);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Unsqueeze { dim } => {
            assert_eq!(*dim, 0);
        }
        other => panic!("expected Unsqueeze, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[1, 3]);
    assert_eq!(result.dims(), &[1, 3]);
}

#[test]
fn test_trace_squeeze() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[1, 3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.squeeze(0)?;
        Ok(b)
    })
    .unwrap();

    // 3 nodes: input + reshape (from squeeze internals) + squeeze.
    // Same delegation pattern as unsqueeze — reshape node remains in graph.
    assert_eq!(graph.len(), 3);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Squeeze { dim } => {
            assert_eq!(*dim, 0);
        }
        other => panic!("expected Squeeze, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[3]);
    assert_eq!(result.dims(), &[3]);
}

#[test]
fn test_trace_permute() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[1, 2, 3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.permute([2, 0, 1])?;
        Ok(b)
    })
    .unwrap();

    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Permute { axes } => {
            assert_eq!(axes, &[2, 0, 1]);
        }
        other => panic!("expected Permute, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[3, 1, 2]);
    assert_eq!(result.dims(), &[3, 1, 2]);
}

// -- Cat tracing --------------------------------------------------------------

#[test]
fn test_trace_cat() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0, 5.0]);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        let id_b = record_input(&[3], DType::F32).unwrap();
        b.set_trace_id(id_b);
        let c = DynTensor::cat(&[&a, &b], 0)?;
        Ok(c)
    })
    .unwrap();

    // 2 inputs + 1 cat = 3 nodes
    assert_eq!(graph.len(), 3);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Cat { dim, num_inputs } => {
            assert_eq!(*dim, 0);
            assert_eq!(*num_inputs, 2);
        }
        other => panic!("expected Cat, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[5]);
    assert_eq!(output.inputs().len(), 2);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

// -- Softmax tracing ----------------------------------------------------------

#[test]
fn test_trace_softmax() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.softmax(1)?;
        Ok(b)
    })
    .unwrap();

    // 1 input + 1 softmax = 2 nodes
    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Softmax { dim } => {
            assert_eq!(*dim, 1);
        }
        other => panic!("expected Softmax, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[2, 3]);
    assert_eq!(output.inputs().len(), 1);

    // Verify softmax rows sum to 1.0
    let vals = result.to_flat_vec::<f32>().unwrap();
    let row0_sum: f32 = vals[0..3].iter().sum();
    let row1_sum: f32 = vals[3..6].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-6);
    assert!((row1_sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_trace_log_softmax() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.log_softmax(1)?;
        Ok(b)
    })
    .unwrap();

    // 1 input + 1 log_softmax = 2 nodes
    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::LogSoftmax { dim } => {
            assert_eq!(*dim, 1);
        }
        other => panic!("expected LogSoftmax, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[2, 3]);
    assert_eq!(output.inputs().len(), 1);

    // Verify all log_softmax values are <= 0
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| v <= 0.0));
}

// -- Type conversion tracing --------------------------------------------------

#[test]
fn test_trace_to_dtype() {
    let a = t1d(&[1.0, 2.0, 3.0]);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.to_dtype(DType::BF16)?;
        Ok(b)
    })
    .unwrap();

    // 1 input + 1 to_dtype = 2 nodes
    assert_eq!(graph.len(), 2);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::ToDtype { target_dtype } => {
            assert_eq!(*target_dtype, DType::BF16);
        }
        other => panic!("expected ToDtype, got {other:?}"),
    }
    assert_eq!(output.inputs().len(), 1);
    assert_eq!(result.dtype(), DType::BF16);
}

// -- Flip tracing -------------------------------------------------------------

#[test]
fn test_trace_flip() {
    let a = t1d(&[1.0, 2.0, 3.0]);

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.flip(0)?;
        Ok(b)
    })
    .unwrap();

    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::Flip { dim } => {
            assert_eq!(*dim, 0);
        }
        other => panic!("expected Flip, got {other:?}"),
    }
    assert_eq!(output.output_shape(), &[3]);
    assert_eq!(output.inputs().len(), 1);

    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![3.0, 2.0, 1.0]);
}

// -- Accumulation tracing -----------------------------------------------------

#[test]
fn test_trace_scatter_add() {
    // dst: [0,0,0], index: [2,0,1], src: [10,20,30]
    // result: [20, 30, 10]
    let dst = t1d(&[0.0, 0.0, 0.0]);
    let idx = DynTensor::from_vec_u32(vec![2, 0, 1], &[3], &cpu()).unwrap();
    let src = t1d(&[10.0, 20.0, 30.0]);

    let (result, graph) = trace_graph(|| {
        let mut dst = dst.clone();
        let mut idx = idx.clone();
        let mut src = src.clone();
        let id_dst = record_input(&[3], DType::F32).unwrap();
        dst.set_trace_id(id_dst);
        let id_idx = record_input(&[3], DType::U32).unwrap();
        idx.set_trace_id(id_idx);
        let id_src = record_input(&[3], DType::F32).unwrap();
        src.set_trace_id(id_src);
        let out = dst.scatter_add(0, &idx, &src)?;
        Ok(out)
    })
    .unwrap();

    // 3 inputs + 1 scatter_add = 4 nodes
    assert_eq!(graph.len(), 4);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::ScatterAdd { dim } => {
            assert_eq!(*dim, 0);
        }
        other => panic!("expected ScatterAdd, got {other:?}"),
    }
    assert_eq!(output.inputs().len(), 3);

    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![20.0, 30.0, 10.0]);
}

#[test]
fn test_trace_index_add() {
    // dst: [0,0,0], index: [2,0], src: [10,20]
    // result: [20, 0, 10]
    let dst = t1d(&[0.0, 0.0, 0.0]);
    let idx = DynTensor::from_vec_u32(vec![2, 0], &[2], &cpu()).unwrap();
    let src = t1d(&[10.0, 20.0]);

    let (result, graph) = trace_graph(|| {
        let mut dst = dst.clone();
        let mut idx = idx.clone();
        let mut src = src.clone();
        let id_dst = record_input(&[3], DType::F32).unwrap();
        dst.set_trace_id(id_dst);
        let id_idx = record_input(&[2], DType::U32).unwrap();
        idx.set_trace_id(id_idx);
        let id_src = record_input(&[2], DType::F32).unwrap();
        src.set_trace_id(id_src);
        let out = dst.index_add(0, &idx, &src)?;
        Ok(out)
    })
    .unwrap();

    // 3 inputs + 1 index_add = 4 nodes
    assert_eq!(graph.len(), 4);
    let output = graph.output_node().unwrap();
    match output.op() {
        TraceOp::IndexAdd { dim } => {
            assert_eq!(*dim, 0);
        }
        other => panic!("expected IndexAdd, got {other:?}"),
    }
    assert_eq!(output.inputs().len(), 3);

    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![20.0, 0.0, 10.0]);
}

// -- Multi-output graph tests -------------------------------------------------

#[test]
fn test_multi_output_mark_output() {
    // Build a graph manually: input → add → relu, mark both add and relu as outputs.
    use super::*;
    let input = TraceNode::new(
        0,
        "input_0".into(),
        TraceOp::Input,
        vec![],
        vec![2, 4],
        DType::F32,
    );
    let add = TraceNode::new(
        1,
        "add_0".into(),
        TraceOp::Add,
        vec![0, 0],
        vec![2, 4],
        DType::F32,
    );
    let relu = TraceNode::new(
        2,
        "relu_0".into(),
        TraceOp::Relu,
        vec![1],
        vec![2, 4],
        DType::F32,
    );

    let mut graph = ComputationGraph::from_nodes(vec![input, add, relu]);
    // Default: single output = last node (relu)
    assert_eq!(graph.output_nodes().len(), 1);
    assert_eq!(graph.output_node().unwrap().name(), "relu_0");

    // Mark add as an additional output
    let _ = graph.mark_output(0); // input
    let _ = graph.mark_output(1); // add
    assert_eq!(graph.output_nodes().len(), 3); // relu (default) + input + add

    // output_node() returns the last marked = add
    assert_eq!(graph.output_node().unwrap().name(), "add_0");

    // Duplicate mark is ignored
    let _ = graph.mark_output(1);
    assert_eq!(graph.output_nodes().len(), 3);

    // Non-existent node is ignored
    let _ = graph.mark_output(999);
    assert_eq!(graph.output_nodes().len(), 3);
}

// -- Upsample1d tracing (#2222) -----------------------------------------------

#[test]
fn test_trace_upsample_nearest_1d() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[1, 1, 4], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.upsample_nearest_1d(2)?;
        Ok(b)
    })
    .unwrap();

    assert_eq!(result.shape().dims(), &[1, 1, 8]);
    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 2);
    assert!(matches!(nodes[1].op(), TraceOp::Upsample1d { factor: 2 }));
    assert_eq!(nodes[1].output_shape(), &[1, 1, 8]);
}
