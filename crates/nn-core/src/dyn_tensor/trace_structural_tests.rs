// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural correctness tests for computation graph tracing.
//!
//! These tests verify graph structure properties that the basic op-type tests
//! do not cover: edge wiring (input reference correctness), fan-out (one tensor
//! consumed by multiple ops), `with_trace_suppressed`, and topological ordering.

use std::collections::HashSet;

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

// -- Edge wiring correctness --------------------------------------------------

/// Verify that input reference IDs on traced nodes actually point to the
/// correct predecessor nodes. The basic op tests check `inputs().len()` but
/// not that the IDs resolve to the expected inputs.
#[test]
fn test_trace_edge_wiring_add() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0]);

    let (_, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        let id_b = record_input(&[2], DType::F32).unwrap();
        b.set_trace_id(id_b);
        let c = a.add(&b)?;
        Ok(c)
    })
    .unwrap();

    assert_eq!(graph.len(), 3);
    let add_node = graph.output_node().unwrap();
    assert!(matches!(add_node.op(), TraceOp::Add));

    // The add node's inputs must resolve to the two Input nodes.
    let input_ids = add_node.inputs();
    assert_eq!(input_ids.len(), 2);

    let input_0 = graph
        .node(input_ids[0])
        .expect("first input should exist in graph");
    let input_1 = graph
        .node(input_ids[1])
        .expect("second input should exist in graph");

    assert!(
        matches!(input_0.op(), TraceOp::Input),
        "first input should be Input"
    );
    assert!(
        matches!(input_1.op(), TraceOp::Input),
        "second input should be Input"
    );

    // Verify the inputs are actually the two different input nodes, not the same
    assert_ne!(
        input_ids[0], input_ids[1],
        "add should reference two distinct inputs"
    );
}

/// Verify edge wiring in a multi-op chain: matmul -> add -> relu.
/// Each op's inputs must point to the correct predecessors.
#[test]
fn test_trace_edge_wiring_chain() {
    let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let w = t2d(&[0.5, 0.5, 0.5, 0.5], 2, 2);
    let b = t1d(&[1.0, -1.0]);

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

        let y = x.matmul(&w)?; // matmul(x, w)
        let z = y.add(&b)?; // add(matmul_result, b)
        let out = z.relu()?; // relu(add_result)
        Ok(out)
    })
    .unwrap();

    // 3 inputs + matmul + add + relu = 6
    assert_eq!(graph.len(), 6);
    let nodes = graph.nodes();

    // Node 3 = matmul: inputs should be nodes 0 (x) and 1 (w)
    let matmul = &nodes[3];
    assert!(matches!(matmul.op(), TraceOp::MatMul));
    assert_eq!(matmul.inputs().len(), 2);
    let mm_in0 = graph.node(matmul.inputs()[0]).unwrap();
    let mm_in1 = graph.node(matmul.inputs()[1]).unwrap();
    assert!(matches!(mm_in0.op(), TraceOp::Input));
    assert!(matches!(mm_in1.op(), TraceOp::Input));

    // Node 4 = add: inputs should be matmul (node 3) and bias input (node 2)
    let add = &nodes[4];
    assert!(matches!(add.op(), TraceOp::Add));
    assert_eq!(add.inputs().len(), 2);
    // One input is the matmul, the other is the bias
    let add_in_ops: Vec<_> = add
        .inputs()
        .iter()
        .map(|&id| graph.node(id).unwrap().op().clone())
        .collect();
    assert!(
        add_in_ops.iter().any(|op| matches!(op, TraceOp::MatMul)),
        "add should have matmul as input"
    );
    assert!(
        add_in_ops.iter().any(|op| matches!(op, TraceOp::Input)),
        "add should have bias (Input) as input"
    );

    // Node 5 = relu: input should be the add node
    let relu = &nodes[5];
    assert!(matches!(relu.op(), TraceOp::Relu));
    assert_eq!(relu.inputs().len(), 1);
    let relu_input = graph.node(relu.inputs()[0]).unwrap();
    assert!(matches!(relu_input.op(), TraceOp::Add));
}

// -- Fan-out (diamond) graph --------------------------------------------------

/// Verify that a tensor consumed by multiple operations produces correct
/// graph structure with shared input references (fan-out).
#[test]
fn test_trace_fan_out_diamond() {
    let x = t1d(&[1.0, 2.0, 3.0]);

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[3], DType::F32).unwrap();
        x.set_trace_id(id);

        // Fan-out: x feeds both relu and neg
        let a = x.relu()?; // relu(x)
        let b = x.neg()?; // neg(x)
                          // Join: add(relu(x), neg(x))
        let c = a.add(&b)?;
        Ok(c)
    })
    .unwrap();

    // 1 input + relu + neg + add = 4
    assert_eq!(graph.len(), 4);

    let nodes = graph.nodes();
    let input_node = &nodes[0];
    assert!(matches!(input_node.op(), TraceOp::Input));
    let input_id = input_node.id();

    // Both relu and neg should reference the same input
    let relu_node = &nodes[1];
    assert!(matches!(relu_node.op(), TraceOp::Relu));
    assert_eq!(
        relu_node.inputs(),
        &[input_id],
        "relu should reference the shared input"
    );

    let neg_node = &nodes[2];
    assert!(matches!(neg_node.op(), TraceOp::Neg));
    assert_eq!(
        neg_node.inputs(),
        &[input_id],
        "neg should reference the shared input"
    );

    // The add node should reference both relu and neg
    let add_node = &nodes[3];
    assert!(matches!(add_node.op(), TraceOp::Add));
    assert_eq!(add_node.inputs().len(), 2);
    let add_input_ids: HashSet<_> = add_node.inputs().iter().copied().collect();
    assert!(add_input_ids.contains(&relu_node.id()));
    assert!(add_input_ids.contains(&neg_node.id()));

    // Verify computation: relu([1,2,3]) + neg([1,2,3]) = [1,2,3] + [-1,-2,-3] = [0,0,0]
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 0.0, 0.0]);
}

// -- with_trace_suppressed ----------------------------------------------------

/// Verify that `with_trace_suppressed` prevents internal ops from being
/// recorded. This is the mechanism that composite layers (Linear, Conv1d)
/// use to record only their composite op, not the decomposed primitives.
#[test]
fn test_with_trace_suppressed_prevents_recording() {
    let a = t1d(&[1.0, 2.0, 3.0]);

    let (_, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[3], DType::F32).unwrap();
        a.set_trace_id(id);

        // This relu should NOT be recorded (suppressed)
        let suppressed_result = with_trace_suppressed(|| a.relu());
        let _ = suppressed_result?;

        // Recording is active again — verify by recording an explicit op
        let unsuppressed = a.exp()?;
        Ok(unsuppressed)
    })
    .unwrap();

    // Only input + exp should be recorded. The relu is suppressed.
    // Note: the suppressed relu still executes (eager eval), just not recorded.
    let ops: Vec<_> = graph
        .nodes()
        .iter()
        .map(|n| format!("{:?}", std::mem::discriminant(n.op())))
        .collect();

    // Should have exactly 2 nodes: Input and Exp (no Relu)
    assert_eq!(
        graph.len(),
        2,
        "suppressed relu should not appear: ops = {ops:?}"
    );
    assert!(matches!(graph.nodes()[0].op(), TraceOp::Input));
    assert!(matches!(graph.nodes()[1].op(), TraceOp::Exp));
}

/// Verify that nested `with_trace_suppressed` restores correctly.
/// Inner suppression should not accidentally re-enable recording.
#[test]
fn test_with_trace_suppressed_nested_restore() {
    let a = t1d(&[1.0, 2.0]);

    let (_, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[2], DType::F32).unwrap();
        a.set_trace_id(id);

        with_trace_suppressed(|| {
            // Outer suppression: relu should NOT be recorded
            let _ = a.relu();

            // Inner suppression: neg should also NOT be recorded
            with_trace_suppressed(|| {
                let _ = a.neg();
            });

            // After inner returns, still suppressed (outer scope)
            let _ = a.exp();
        });

        // After outer suppression, recording should be active again
        let result = a.tanh()?;
        Ok(result)
    })
    .unwrap();

    // Only input + tanh should be recorded. All 3 ops inside suppression are hidden.
    assert_eq!(graph.len(), 2, "all suppressed ops should be hidden");
    assert!(matches!(graph.nodes()[0].op(), TraceOp::Input));
    assert!(matches!(graph.nodes()[1].op(), TraceOp::Tanh));
}

// -- Topological order validation ---------------------------------------------

/// Verify that `validate_topology` rejects a graph where a valid node ID
/// appears after its consumer (topological order violation, not missing node).
#[test]
fn test_validate_topology_rejects_out_of_order_references() {
    // Node 2 (relu) references node 1 (input), but node 1 appears AFTER node 2.
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![1], // references node 1, which hasn't appeared yet
            vec![3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![3],
            DType::F32,
        ),
    ]);

    let err = graph
        .validate_topology()
        .expect_err("should reject out-of-order references");
    match &err {
        TensorError::TopologyError {
            node_name,
            missing_input,
            ..
        } => {
            assert_eq!(node_name, "relu_0");
            assert_eq!(*missing_input, 1);
        }
        other => panic!("expected TopologyError, got: {other:?}"),
    }
}

// -- trace_input_ids with 3 inputs -------------------------------------------

/// Verify `trace_input_ids` works correctly with 3 inputs (not just 1-2).
/// This covers ops like WhereCond that take condition + true_branch + false_branch.
#[test]
fn test_trace_input_ids_three_inputs() {
    let ((), graph) = trace_graph(|| {
        let a = t1d(&[1.0]);
        let b = t1d(&[2.0]);
        let c = t1d(&[3.0]);
        let mut a = a;
        let mut b = b;
        let mut c = c;
        let id_a = record_input(&[1], DType::F32).unwrap();
        a.set_trace_id(id_a);
        let id_b = record_input(&[1], DType::F32).unwrap();
        b.set_trace_id(id_b);
        let id_c = record_input(&[1], DType::F32).unwrap();
        c.set_trace_id(id_c);

        // Collect all three trace IDs
        let ids = DynTensor::trace_input_ids(&[&a, &b, &c])?;
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], id_a);
        assert_eq!(ids[1], id_b);
        assert_eq!(ids[2], id_c);

        Ok(())
    })
    .unwrap();

    assert_eq!(graph.len(), 3); // 3 inputs
}

// -- to_weight_ref fallback paths ---------------------------------------------

/// Verify `to_weight_ref` returns an error for I64 tensors.
/// I64 CPU tensors cannot be converted to f32 arrays, so the method
/// returns `Err(WeightConversionFailed)` instead of silently falling
/// back to shape-only capture.
#[test]
fn test_to_weight_ref_i64_tensor_returns_error() {
    let t = DynTensor::from_vec_i64(vec![10i64, 20, 30], &[3], &cpu()).unwrap();
    let result = t.to_weight_ref();

    assert!(
        result.is_err(),
        "I64 tensor should return WeightConversionFailed"
    );
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("Weight conversion failed"),
        "Expected WeightConversionFailed, got: {err}"
    );
}

/// Verify `to_weight_ref` captures actual data for BF16 tensors.
/// BF16 tensors have `to_f32_array()` available, so the fast path should work.
#[test]
fn test_to_weight_ref_bf16_tensor_captures_data() {
    let f32_tensor = DynTensor::new(&[1.0f32, 2.0, 3.0], &[3], &cpu()).unwrap();
    let bf16_tensor = f32_tensor.to_dtype(DType::BF16).unwrap();
    let wref = bf16_tensor.to_weight_ref().unwrap();

    assert_eq!(wref.shape, &[3]);
    // BF16 should convert to f32 successfully — data should be non-empty.
    assert!(
        !wref.data.is_empty(),
        "BF16 tensor should have data captured via to_f32_array()"
    );
    assert_eq!(wref.data.len(), 3);
    // Check approximate values (BF16 has lower precision).
    for (i, &val) in wref.data.iter().enumerate() {
        let expected = (i + 1) as f32;
        assert!(
            (val - expected).abs() < 0.1,
            "BF16 weight[{i}] = {val}, expected ~{expected}"
        );
    }
}

// -- WeightRef::new() validation ----------------------------------------------

/// `WeightRef::new()` succeeds when data length matches shape product.
#[test]
fn test_weight_ref_new_valid_data_shape() {
    let wref = WeightRef::new(vec![1.0; 6], vec![2, 3]);
    assert!(wref.is_ok());
    let wref = wref.unwrap();
    assert_eq!(wref.data(), &[1.0; 6]);
    assert_eq!(wref.shape(), &[2, 3]);
}

/// `WeightRef::new()` succeeds with empty data (shape-only ref).
#[test]
fn test_weight_ref_new_empty_data_allowed() {
    let wref = WeightRef::new(vec![], vec![4, 5]);
    assert!(
        wref.is_ok(),
        "empty data should be allowed for shape-only refs"
    );
    assert!(wref.unwrap().data().is_empty());
}

/// `WeightRef::from_shape()` creates shape-only ref with empty data.
#[test]
fn test_weight_ref_from_shape_creates_shape_only() {
    let wref = WeightRef::from_shape(&[4, 5]);
    assert!(wref.data().is_empty());
    assert_eq!(wref.shape(), &[4, 5]);
}

/// `WeightRef::new()` returns error when data length does not match shape product.
#[test]
fn test_weight_ref_new_data_shape_mismatch() {
    // data has 4 elements, shape requires 6
    let result = WeightRef::new(vec![1.0; 4], vec![2, 3]);
    assert!(result.is_err(), "data/shape mismatch should return error");
}

/// `WeightRef::new()` returns error for scalar shape with wrong data length.
#[test]
fn test_weight_ref_new_scalar_shape_mismatch() {
    // shape=[] means product=1, but data has 3 elements
    let result = WeightRef::new(vec![1.0, 2.0, 3.0], vec![]);
    assert!(result.is_err(), "3 elements for scalar shape should fail");
}

/// `WeightRef::new()` succeeds for scalar shape with exactly 1 element.
#[test]
fn test_weight_ref_new_scalar_shape_valid() {
    let wref = WeightRef::new(vec![42.0], vec![]);
    assert!(wref.is_ok(), "1 element for scalar shape should succeed");
    assert_eq!(wref.unwrap().data(), &[42.0]);
}

/// `WeightRef::new()` with zero-dim shape (product=0) requires empty data.
#[test]
fn test_weight_ref_new_zero_dim_empty_data() {
    let wref = WeightRef::new(vec![], vec![2, 0, 3]);
    assert!(
        wref.is_ok(),
        "empty data with zero-dim shape should succeed"
    );
    assert_eq!(wref.unwrap().shape(), &[2, 0, 3]);
}

/// `WeightRef::new()` rejects non-empty data with zero-dim shape.
#[test]
fn test_weight_ref_new_zero_dim_nonempty_data() {
    let result = WeightRef::new(vec![1.0], vec![0]);
    assert!(
        result.is_err(),
        "non-empty data with zero-dim shape should fail"
    );
}

/// `WeightRef::from_shape()` preserves zero-dim shapes.
#[test]
fn test_weight_ref_from_shape_zero_dim() {
    let wref = WeightRef::from_shape(&[3, 0, 5]);
    assert_eq!(wref.shape(), &[3, 0, 5]);
    assert!(wref.data().is_empty());
}

/// `WeightRef` accessors return correct borrowed slices.
#[test]
fn test_weight_ref_accessors_borrow() {
    let wref = WeightRef::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert_eq!(wref.data().len(), 3);
    assert_eq!(wref.shape().len(), 1);
    assert_eq!(wref.data()[2], 3.0);
    assert_eq!(wref.shape()[0], 3);
}

// -- Performance proofs -------------------------------------------------------

/// Prove that `NameCounter::next_name` allocates a String key on every
/// `HashMap::entry()` call via `prefix.to_string()`, even on cache hit.
#[test]
fn test_name_counter_generates_sequential_names_many_ops() {
    let a = t1d(&[1.0, 2.0, 3.0]);

    let (_, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[3], DType::F32).unwrap();
        a.set_trace_id(id);

        let mut x = a;
        for _ in 0..50 {
            x = x.relu()?;
        }
        Ok(x)
    })
    .unwrap();

    // 1 input + 50 relu = 51 nodes
    assert_eq!(graph.len(), 51);

    // Verify sequential naming: relu_0 through relu_49
    let relu_names: Vec<&str> = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Relu))
        .map(TraceNode::name)
        .collect();
    assert_eq!(relu_names.len(), 50);
    for (i, &name) in relu_names.iter().enumerate() {
        let expected = format!("relu_{i}");
        assert_eq!(name, expected, "relu name at index {i}");
    }

    assert!(graph.validate_topology().is_ok());
}

/// Prove that `input_nodes()` performs a full linear scan per call.
#[test]
fn test_input_nodes_linear_scan_per_call() {
    let (_, graph) = trace_graph(|| {
        let mut tensors = Vec::new();
        for _ in 0..5 {
            let a = t1d(&[1.0, 2.0]);
            let mut a = a;
            let id = record_input(&[2], DType::F32).unwrap();
            a.set_trace_id(id);
            tensors.push(a);
        }
        let mut acc = tensors[0].relu()?;
        for t in &tensors[1..] {
            acc = acc.add(t)?;
        }
        Ok(acc)
    })
    .unwrap();

    assert_eq!(graph.len(), 10);

    let inputs = graph.input_nodes();
    assert_eq!(inputs.len(), 5);
    assert!(inputs.iter().all(|n| matches!(n.op(), TraceOp::Input)));

    let inputs2 = graph.input_nodes();
    assert_eq!(inputs2.len(), 5);
    assert_ne!(
        inputs.as_ptr() as usize,
        inputs2.as_ptr() as usize,
        "each call allocates a new Vec"
    );
}

// -- is_placeholder (#2190) ---------------------------------------------------

/// `is_placeholder()` is true for non-zero shape with empty data.
#[test]
fn test_weight_ref_is_placeholder_nonzero_shape() {
    let wref = WeightRef::from_shape(&[3, 4]);
    assert!(
        wref.is_placeholder(),
        "from_shape with non-zero dims must be placeholder"
    );
}

/// `is_placeholder()` is false for empty shape (absent optional param).
#[test]
fn test_weight_ref_is_placeholder_empty_shape() {
    let wref = WeightRef::from_shape(&[]);
    assert!(!wref.is_placeholder(), "empty shape is not a placeholder");
}

/// `is_placeholder()` is false for zero-dim shape (product is 0).
#[test]
fn test_weight_ref_is_placeholder_zero_dim_shape() {
    let wref = WeightRef::from_shape(&[3, 0, 5]);
    assert!(
        !wref.is_placeholder(),
        "zero-dim shape is not a placeholder"
    );
}

/// `is_placeholder()` is false when WeightRef has actual data.
#[test]
fn test_weight_ref_is_placeholder_with_data() {
    let wref = WeightRef::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert!(
        !wref.is_placeholder(),
        "real data should not be placeholder"
    );
}

// -- ComputationGraph accessor coverage ---------------------------------------

/// `is_empty()` returns true for empty graph.
#[test]
fn test_computation_graph_is_empty() {
    let graph = ComputationGraph::from_nodes(vec![]);
    assert!(graph.is_empty());
    assert_eq!(graph.len(), 0);
    assert!(graph.output_node().is_none());
}

/// `node()` returns None for missing ID and Some for valid ID.
#[test]
fn test_computation_graph_node_lookup() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            10,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![3],
            DType::F32,
        ),
        TraceNode::new(
            20,
            "relu".into(),
            TraceOp::Relu,
            vec![10],
            vec![3],
            DType::F32,
        ),
    ]);
    let node = graph.node(10);
    assert!(node.is_some());
    assert_eq!(node.unwrap().name(), "input");
    assert!(graph.node(999).is_none());
}

/// `input_nodes()` returns only Input-typed nodes.
#[test]
fn test_computation_graph_input_nodes() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(1, "in0".into(), TraceOp::Input, vec![], vec![2], DType::F32),
        TraceNode::new(2, "in1".into(), TraceOp::Input, vec![], vec![3], DType::F32),
        TraceNode::new(
            3,
            "add".into(),
            TraceOp::Add,
            vec![1, 2],
            vec![3],
            DType::F32,
        ),
    ]);
    let inputs = graph.input_nodes();
    assert_eq!(inputs.len(), 2);
    assert!(inputs.iter().all(|n| matches!(n.op(), TraceOp::Input)));
}

/// `input_nodes()` on empty graph returns empty vec.
#[test]
fn test_computation_graph_input_nodes_empty() {
    let graph = ComputationGraph::from_nodes(vec![]);
    assert!(graph.input_nodes().is_empty());
}

/// `output_node()` returns last node.
#[test]
fn test_computation_graph_output_node_is_last() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(1, "in".into(), TraceOp::Input, vec![], vec![4], DType::F32),
        TraceNode::new(2, "exp".into(), TraceOp::Exp, vec![1], vec![4], DType::F32),
    ]);
    let out = graph.output_node().expect("should have output");
    assert_eq!(out.name(), "exp");
    assert_eq!(out.id(), 2);
}
