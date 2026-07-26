// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Node-count verification tests for trace suppression (#2121).
//!
//! Each nn Module::forward() wraps computation in `with_trace_suppressed()`,
//! so the trace graph must contain exactly 2 nodes: 1 Input + 1 composite op.
//! Anything more means primitive ops leaked through suppression.

use nn_core::dyn_tensor::trace::{record_input, trace_graph, TraceOp};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

#[test]
fn test_trace_suppression_conv1d_exact_node_count() {
    use nn_core::layers::{Conv1d, Conv1dConfig, Module};

    let weight = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();
    let conv = Conv1d::new(weight, None, Conv1dConfig::default()).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 5], DType::F32).unwrap();
        x.set_trace_id(id);
        conv.forward(&x)
    })
    .unwrap();

    let ops: Vec<_> = graph
        .nodes()
        .iter()
        .map(|n| format!("{:?}", n.op()))
        .collect();
    assert_eq!(
        graph.len(),
        2,
        "Conv1d Module forward should produce exactly 2 nodes (1 Input + 1 Conv1d), \
         but got {} nodes: {:?}",
        graph.len(),
        ops,
    );
    assert!(matches!(graph.nodes()[0].op(), TraceOp::Input));
    assert!(matches!(graph.nodes()[1].op(), TraceOp::Conv1d { .. }));
}

#[test]
fn test_trace_suppression_conv2d_exact_node_count() {
    use nn_core::layers::{Conv2d, Conv2dConfig, Module};

    let weight = DynTensor::new(&[1.0; 4], &[1, 1, 2, 2], &cpu()).unwrap();
    let conv = Conv2d::new(weight, None, Conv2dConfig::default()).unwrap();
    let x = DynTensor::new(&[1.0; 9], &[1, 1, 3, 3], &cpu()).unwrap();

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 3, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        conv.forward(&x)
    })
    .unwrap();

    let ops: Vec<_> = graph
        .nodes()
        .iter()
        .map(|n| format!("{:?}", n.op()))
        .collect();
    assert_eq!(
        graph.len(),
        2,
        "Conv2d Module forward should produce exactly 2 nodes (1 Input + 1 Conv2d), \
         but got {} nodes: {:?}",
        graph.len(),
        ops,
    );
    assert!(matches!(graph.nodes()[0].op(), TraceOp::Input));
    assert!(matches!(graph.nodes()[1].op(), TraceOp::Conv2d { .. }));
}

#[test]
fn test_trace_suppression_linear_exact_node_count() {
    use nn_core::layers::{Linear, Module};

    let weight = DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap();
    let bias = DynTensor::new(&[0.5, -0.5], &[2], &cpu()).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        linear.forward(&x)
    })
    .unwrap();

    let ops: Vec<_> = graph
        .nodes()
        .iter()
        .map(|n| format!("{:?}", n.op()))
        .collect();
    assert_eq!(
        graph.len(),
        2,
        "Linear Module forward should produce exactly 2 nodes (1 Input + 1 Linear), \
         but got {} nodes: {:?}",
        graph.len(),
        ops,
    );
    assert!(matches!(graph.nodes()[0].op(), TraceOp::Input));
    assert!(matches!(graph.nodes()[1].op(), TraceOp::Linear { .. }));
}

#[test]
fn test_trace_suppression_layer_norm_exact_node_count() {
    use nn_core::layers::{LayerNorm, Module};

    let weight = DynTensor::ones(&[4], DType::F32, &cpu()).unwrap();
    let bias = DynTensor::zeros(&[4], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5).unwrap();
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let (_, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        ln.forward(&x)
    })
    .unwrap();

    let ops: Vec<_> = graph
        .nodes()
        .iter()
        .map(|n| format!("{:?}", n.op()))
        .collect();
    assert_eq!(
        graph.len(),
        2,
        "LayerNorm Module forward should produce exactly 2 nodes (1 Input + 1 LayerNorm), \
         but got {} nodes: {:?}",
        graph.len(),
        ops,
    );
    assert!(matches!(graph.nodes()[0].op(), TraceOp::Input));
    assert!(matches!(graph.nodes()[1].op(), TraceOp::LayerNorm { .. }));
}

#[test]
fn test_trace_suppression_embedding_exact_node_count() {
    use nn_core::layers::{Embedding, Module};

    let embed_weight = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap();
    let emb = Embedding::new(embed_weight).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 2, 1], &[3], &cpu()).unwrap();

    let (_, graph) = trace_graph(|| {
        let mut ids = ids.clone();
        let id = record_input(&[3], DType::U32).unwrap();
        ids.set_trace_id(id);
        emb.forward(&ids)
    })
    .unwrap();

    let ops: Vec<_> = graph
        .nodes()
        .iter()
        .map(|n| format!("{:?}", n.op()))
        .collect();
    assert_eq!(
        graph.len(),
        2,
        "Embedding Module forward should produce exactly 2 nodes (1 Input + 1 Embedding), \
         but got {} nodes: {:?}",
        graph.len(),
        ops,
    );
    assert!(matches!(graph.nodes()[0].op(), TraceOp::Input));
    assert!(matches!(graph.nodes()[1].op(), TraceOp::Embedding { .. }));
}

/// Flip on CPU uses ndarray slice (no decomposition), but the trace contract
/// must still produce exactly 2 nodes. On GPU, flip decomposes into
/// `index_select` which is suppressed via `with_trace_suppressed` (#2414).
/// This test validates the CPU path contract; GPU path requires Metal hardware.
#[test]
fn test_trace_suppression_flip_exact_node_count() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        x.flip(0)
    })
    .unwrap();

    let ops: Vec<_> = graph
        .nodes()
        .iter()
        .map(|n| format!("{:?}", n.op()))
        .collect();
    assert_eq!(
        graph.len(),
        2,
        "flip should produce exactly 2 nodes (1 Input + 1 Flip), \
         but got {} nodes: {:?}",
        graph.len(),
        ops,
    );
    assert!(matches!(graph.nodes()[0].op(), TraceOp::Input));
    assert!(matches!(graph.nodes()[1].op(), TraceOp::Flip { dim: 0 }));

    // Verify flip correctness: rows reversed
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![3.0, 4.0, 1.0, 2.0]);
}
