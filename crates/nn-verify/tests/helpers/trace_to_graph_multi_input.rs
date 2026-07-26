// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for multi-input trace-to-graph translation (#2377).
//!
//! When a traced computation has >1 `TraceOp::Input` node, gamma-build maps
//! all input TensorSpecs to the same `_input` sentinel. The multi-input path
//! stacks all inputs into a single 1D tensor and uses Slice+Reshape LayerSpecs
//! to split them back per variable.

use super::common::assert_bounds_valid;
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_verify::{trace_to_graph_model, trace_to_graph_model_multi_input, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

fn cpu() -> Device {
    Device::Cpu
}

// -- Test 1: Two same-shape inputs added together ----------------------------

#[test]
fn test_multi_input_add_same_shape() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[2, 3], DType::F32).unwrap();
        a.set_trace_id(id_a);

        let mut b = b.clone();
        let id_b = record_input(&[2, 3], DType::F32).unwrap();
        b.set_trace_id(id_b);

        let y = a.add(&b)?;
        Ok(y)
    })
    .unwrap();

    // Translate via multi-input path
    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("multi-input should succeed")
        .graph;
    assert!(gn.num_nodes() > 0, "GraphNetwork should have nodes");

    // IBP propagation: stacked input is [12] (6 elements for a + 6 for b).
    // a bounds: [-1, 1] for all 6 elements
    // b bounds: [-0.5, 0.5] for all 6 elements
    let total_flat = 6 + 6;
    let mut lower = vec![-1.0_f32; 6];
    lower.extend_from_slice(&[-0.5_f32; 6]);
    let mut upper = vec![1.0_f32; 6];
    upper.extend_from_slice(&[0.5_f32; 6]);

    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total_flat]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total_flat]), upper).unwrap(),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // Add bounds: lower = -1.0 + (-0.5) = -1.5, upper = 1.0 + 0.5 = 1.5
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -1.5 - 1e-4, "lower bound {v} should be >= -1.5");
    }
    for &v in hi.iter() {
        assert!(v <= 1.5 + 1e-4, "upper bound {v} should be <= 1.5");
    }
}

// -- Test 2: Two different-shape inputs --------------------------------------

#[test]
fn test_multi_input_different_shapes() {
    // a: [1, 4], b: [1, 2] — different shapes, both fed to a simple op chain.
    // We'll use Narrow on a to [1, 2] then Add with b.
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();
    let b = DynTensor::new(&[0.1, 0.2], &[1, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[1, 4], DType::F32).unwrap();
        a.set_trace_id(id_a);

        let mut b = b.clone();
        let id_b = record_input(&[1, 2], DType::F32).unwrap();
        b.set_trace_id(id_b);

        // Narrow a from [1,4] to [1,2] then add b
        let a_narrow = a.narrow(1, 0, 2)?;
        let y = a_narrow.add(&b)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("multi-input should succeed")
        .graph;
    assert!(gn.num_nodes() > 0, "GraphNetwork should have nodes");

    // Stacked input: [4 + 2] = [6]
    // a bounds: [-2, 2] for 4 elements, b bounds: [-1, 1] for 2 elements
    let total_flat = 4 + 2;
    let mut lower = vec![-2.0_f32; 4];
    lower.extend_from_slice(&[-1.0_f32; 2]);
    let mut upper = vec![2.0_f32; 4];
    upper.extend_from_slice(&[1.0_f32; 2]);

    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total_flat]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total_flat]), upper).unwrap(),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // Narrow(a, dim=1, start=0, len=2): bounds [-2, 2] for 2 elements
    // Add: [-2, 2] + [-1, 1] = [-3, 3]
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -3.0 - 1e-4, "lower bound {v} should be >= -3.0");
    }
    for &v in hi.iter() {
        assert!(v <= 3.0 + 1e-4, "upper bound {v} should be <= 3.0");
    }
}

// -- Test 3: Single input still works (regression test) ----------------------

#[test]
fn test_single_input_unchanged() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.relu()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("single input should work")
        .graph;
    assert!(gn.num_nodes() > 0);

    // Single input: standard shape [1, 3] (NOT stacked)
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 3]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 3]), 1.0_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // ReLU: lower = max(0, -1) = 0, upper = max(0, 1) = 1
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -1e-4, "ReLU lower bound {v} should be >= 0");
    }
    for &v in hi.iter() {
        assert!(v <= 1.0 + 1e-4, "ReLU upper bound {v} should be <= 1.0");
    }
}

// -- Test 4: trace_to_graph_model rejects multi-input graphs (#2425) ----------

#[test]
fn test_single_input_guard_rejects_multi_input() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[4.0, 5.0, 6.0], &[1, 3], &cpu()).unwrap();

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

    // trace_to_graph_model (single-input) should reject with MultipleVariableInputs
    let err = trace_to_graph_model(&graph).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("variable inputs"),
        "expected MultipleVariableInputs error, got: {msg}"
    );
    assert!(
        msg.contains("trace_to_graph_model_multi_input"),
        "error should suggest multi_input variant, got: {msg}"
    );
}

// -- Test 5: Three same-shape inputs added together --------------------------

#[test]
fn test_three_inputs_same_shape() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[0.1, 0.2, 0.3], &[1, 3], &cpu()).unwrap();
    let c = DynTensor::new(&[0.01, 0.02, 0.03], &[1, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id_a = record_input(&[1, 3], DType::F32).unwrap();
        a.set_trace_id(id_a);

        let mut b = b.clone();
        let id_b = record_input(&[1, 3], DType::F32).unwrap();
        b.set_trace_id(id_b);

        let mut c = c.clone();
        let id_c = record_input(&[1, 3], DType::F32).unwrap();
        c.set_trace_id(id_c);

        // a + b + c: all [1, 3]
        let ab = a.add(&b)?;
        let y = ab.add(&c)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("three inputs should work")
        .graph;
    assert!(gn.num_nodes() > 0);

    // Stacked: [3 + 3 + 3] = [9]
    let total_flat = 3 + 3 + 3;
    let lower = vec![-1.0_f32; total_flat];
    let upper = vec![1.0_f32; total_flat];

    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total_flat]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total_flat]), upper).unwrap(),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // a+b+c bounds: [-1, 1] + [-1, 1] + [-1, 1] = [-3, 3]
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -3.0 - 1e-4, "lower bound {v} should be >= -3.0");
    }
    for &v in hi.iter() {
        assert!(v <= 3.0 + 1e-4, "upper bound {v} should be <= 3.0");
    }
}
