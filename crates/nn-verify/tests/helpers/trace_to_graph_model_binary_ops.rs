// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for binary op and reduction trace-to-graph translation
//! via the `trace_to_graph_model` (LayerSpec → build_graph_network) path.
//!
//! Mirrors `trace_to_graph_binary_ops.rs` (old `trace_to_graph_network` path)
//! to ensure equivalent coverage on the new path.

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_verify::{
    propagate_with_crown_fallback, trace_to_graph_model_multi_input, BoundedTensor, PropMethod,
};
use ndarray::{ArrayD, IxDyn};

fn cpu() -> Device {
    Device::Cpu
}

// -- Add IBP ------------------------------------------------------------------

#[test]
fn test_model_trace_add_ibp() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2, 2], DType::F32).unwrap();
        let id_b = record_input(&[2, 2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        b.set_trace_id(id_b);
        let y = a.add(&b)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("Add translation")
        .graph;
    // Multi-input: two [2,2] inputs stacked = [8]
    let input_bounds = uniform_bounds(&[8], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -2.0 - 0.1, "add lower bound should be >= -2, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 2.0 + 0.1, "add upper bound should be <= 2, got {v}");
    }
}

// -- Add CROWN ----------------------------------------------------------------

#[test]
fn test_model_trace_add_crown() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2, 2], DType::F32).unwrap();
        let id_b = record_input(&[2, 2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        b.set_trace_id(id_b);
        let y = a.add(&b)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("translation")
        .graph;
    // Multi-input: two [2,2] inputs stacked = [8]
    let input_bounds = uniform_bounds(&[8], 1.0);

    let (method, output, crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    assert_bounds_valid(&output);

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for Add. Error: {crown_err:?}"
    );
}

// -- Sub IBP ------------------------------------------------------------------

#[test]
fn test_model_trace_sub_ibp() {
    let a = DynTensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2, 2], DType::F32).unwrap();
        let id_b = record_input(&[2, 2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        b.set_trace_id(id_b);
        let y = a.sub(&b)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("Sub translation")
        .graph;
    // Multi-input: two [2,2] inputs stacked = [8]
    let input_bounds = uniform_bounds(&[8], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l <= 0.01, "sub lower bound should contain 0, got {l}");
        assert!(u >= -0.01, "sub upper bound should contain 0, got {u}");
    }
}

// -- Mul IBP ------------------------------------------------------------------

#[test]
fn test_model_trace_mul_ibp() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2, 2], DType::F32).unwrap();
        let id_b = record_input(&[2, 2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        b.set_trace_id(id_b);
        let y = a.mul(&b)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("Mul translation")
        .graph;
    // Multi-input: two [2,2] inputs stacked = [8]
    let input_bounds = uniform_bounds(&[8], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (_lo, hi) = output.lower_upper();
    for &v in hi.iter() {
        assert!(
            v <= 1.0 + 0.1,
            "mul(x,x) upper bound should be <= 1 for x in [-1,1], got {v}"
        );
    }
}

// -- Div IBP ------------------------------------------------------------------

#[test]
fn test_model_trace_div_ibp() {
    let a = DynTensor::new(&[2.0, 4.0, 6.0, 8.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[2.0, 2.0, 2.0, 2.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2, 2], DType::F32).unwrap();
        let id_b = record_input(&[2, 2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        b.set_trace_id(id_b);
        let y = a.div(&b)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("Div translation")
        .graph;
    // Multi-input: two [2,2] inputs stacked = [8]
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[8]), 0.5f32),
        ArrayD::from_elem(IxDyn(&[8]), 2.0f32),
    )
    .expect("positive bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -0.01, "div of positives should be >= 0, got lo={l}");
        assert!(l <= 1.01, "div bounds should contain 1.0, got lo={l}");
        assert!(u >= 0.99, "div bounds should contain 1.0, got hi={u}");
    }
}

// -- Maximum IBP --------------------------------------------------------------

#[test]
fn test_model_trace_maximum_ibp() {
    let a = DynTensor::new(&[1.0, 4.0, 2.0, 5.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[3.0, 2.0, 4.0, 1.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2, 2], DType::F32).unwrap();
        let id_b = record_input(&[2, 2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        b.set_trace_id(id_b);
        let y = a.maximum(&b)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("Maximum translation")
        .graph;
    // Multi-input: two [2,2] inputs stacked = [8]
    let input_bounds = uniform_bounds(&[8], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -2.0 - 0.1, "maximum lower bound too low: {l}");
        assert!(u <= 2.0 + 0.1, "maximum upper bound too high: {u}");
    }
}

// -- Minimum IBP --------------------------------------------------------------

#[test]
fn test_model_trace_minimum_ibp() {
    let a = DynTensor::new(&[1.0, 4.0, 2.0, 5.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[3.0, 2.0, 4.0, 1.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2, 2], DType::F32).unwrap();
        let id_b = record_input(&[2, 2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        b.set_trace_id(id_b);
        let y = a.minimum(&b)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("Minimum translation")
        .graph;
    // Multi-input: two [2,2] inputs stacked = [8]
    let input_bounds = uniform_bounds(&[8], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -2.0 - 0.1, "minimum lower bound too low: {l}");
        assert!(u <= 2.0 + 0.1, "minimum upper bound too high: {u}");
    }
}

// -- ReduceMax IBP ------------------------------------------------------------

#[test]
fn test_model_trace_reduce_max_ibp() {
    let x = DynTensor::new(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.max_keepdim(1)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 1]);

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("ReduceMax translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (_lo, hi) = output.lower_upper();
    for &v in hi.iter() {
        assert!(v <= 1.0 + 1e-5, "max upper bound too high: {v}");
    }
}

// -- ReduceMin IBP ------------------------------------------------------------

#[test]
fn test_model_trace_reduce_min_ibp() {
    let x = DynTensor::new(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.min_keepdim(1)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 1]);

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("ReduceMin translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, _hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -1.0 - 1e-5, "min lower bound too low: {v}");
    }
}

// -- MatMul IBP (Part of #2329) -----------------------------------------------
//
// Verifies that TraceOp::MatMul → LayerType::MatMul wiring (commit 01ece04)
// produces a valid GraphNetwork and IBP propagation succeeds.

#[test]
fn test_model_trace_matmul_ibp() {
    // Two [2, 2] variable inputs: a @ b
    let a = DynTensor::new(&[1.0, 0.5, -0.3, 0.8], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[0.2, -0.1, 0.4, 0.6], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2, 2], DType::F32).unwrap();
        let id_b = record_input(&[2, 2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        b.set_trace_id(id_b);
        let y = a.matmul(&b)?;
        Ok(y)
    })
    .unwrap();

    // Verify graph translates (was UnsupportedOp before commit 01ece04)
    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("MatMul translation should succeed")
        .graph;
    assert!(gn.num_nodes() > 0, "GraphNetwork should have nodes");

    // Multi-input: two [2,2] inputs stacked = [8], all in [-1, 1]
    let input_bounds = uniform_bounds(&[8], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // MatMul of two [-1,1]^{2x2} matrices: each output element is sum of 2 products.
    // Each product has range [-1, 1], so each output element has range [-2, 2].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(
            v >= -2.0 - 0.5,
            "matmul lower bound should be >= -2.5, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v <= 2.0 + 0.5,
            "matmul upper bound should be <= 2.5, got {v}"
        );
    }
}

// -- MatMul trace_node_id verification (Part of #2329) ------------------------

#[test]
fn test_model_trace_matmul_has_trace_id() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2], &cpu()).unwrap();

    let (result, _graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[2, 2], DType::F32).unwrap();
        let id_b = record_input(&[2, 2], DType::F32).unwrap();
        a.set_trace_id(id_a);
        b.set_trace_id(id_b);
        let y = a.matmul(&b)?;
        Ok(y)
    })
    .unwrap();

    assert!(
        result.trace_id().is_some(),
        "MatMul result should have trace_node_id during tracing"
    );
}
