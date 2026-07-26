// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for normalization op trace-to-graph translation
//! via the `trace_to_graph_model` (LayerSpec → build_graph_network) path.
//!
//! Covers: LayerNorm (IBP, CROWN), RmsNorm (IBP, CROWN),
//! BatchNorm (IBP, CROWN).
//!
//! Extracted from `trace_to_graph_model_misc_ops.rs` for 500-line compliance.

use super::common::assert_bounds_valid;
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{BatchNorm, LayerNorm, Module, RmsNorm};
use nn_core::{DType, Device};
use nn_verify::{propagate_with_crown_fallback, trace_to_graph_model, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

fn cpu() -> Device {
    Device::Cpu
}

// -- LayerNorm IBP ------------------------------------------------------------

#[test]
fn test_model_trace_layer_norm_ibp() {
    let weight = DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[4], &cpu()).unwrap();
    let bias = DynTensor::new(&[0.0, 0.0, 0.0, 0.0], &[4], &cpu()).unwrap();
    let layer_norm = LayerNorm::new(weight, bias, 1e-5).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = layer_norm.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("LayerNorm translation should succeed")
        .graph;
    assert!(gn.num_nodes() > 0);

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 4]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 4]), 10.0_f32),
    )
    .expect("valid bounds");

    let output = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP should succeed for LayerNorm");
    assert_bounds_valid(&output);
}

// -- LayerNorm CROWN ----------------------------------------------------------

#[test]
fn test_model_trace_layer_norm_crown() {
    let weight = DynTensor::new(&[2.0, 0.5, 1.0, 1.5], &[4], &cpu()).unwrap();
    let bias = DynTensor::new(&[0.1, -0.1, 0.0, 0.2], &[4], &cpu()).unwrap();
    let layer_norm = LayerNorm::new(weight, bias, 1e-5).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = layer_norm.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("LayerNorm translation should succeed")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 4]), -5.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 4]), 5.0_f32),
    )
    .expect("valid bounds");

    let (_method, crown_output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    assert_bounds_valid(&crown_output);
}

// -- RmsNorm IBP --------------------------------------------------------------

#[test]
fn test_model_trace_rms_norm_ibp() {
    let weight = DynTensor::new(&[1.0, 1.0, 1.0, 1.0], &[4], &cpu()).unwrap();
    let rms_norm = RmsNorm::new(weight, 1e-5).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = rms_norm.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("RmsNorm translation should succeed")
        .graph;
    assert!(gn.num_nodes() > 0);

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 4]), 0.1_f32),
        ArrayD::from_elem(IxDyn(&[2, 4]), 10.0_f32),
    )
    .expect("valid bounds");

    let output = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP should succeed for RmsNorm");
    assert_bounds_valid(&output);
}

// -- RmsNorm CROWN ------------------------------------------------------------

#[test]
fn test_model_trace_rms_norm_crown() {
    let weight = DynTensor::new(&[2.0, 0.5, 1.0, 1.5], &[4], &cpu()).unwrap();
    let rms_norm = RmsNorm::new(weight, 1e-6).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = rms_norm.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("RmsNorm translation should succeed")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 4]), -5.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 4]), 5.0_f32),
    )
    .expect("valid bounds");

    let (_method, crown_output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    assert_bounds_valid(&crown_output);
}

// -- BatchNorm IBP ------------------------------------------------------------

#[test]
fn test_model_trace_batch_norm_ibp() {
    let running_mean = DynTensor::new(&[0.0, 1.0, -1.0], &[3], &cpu()).unwrap();
    let running_var = DynTensor::new(&[1.0, 2.0, 0.5], &[3], &cpu()).unwrap();
    let gamma = DynTensor::new(&[1.0, 1.0, 1.0], &[3], &cpu()).unwrap();
    let beta = DynTensor::new(&[0.0, 0.0, 0.0], &[3], &cpu()).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, Some(gamma), Some(beta), 1e-5).unwrap();

    let x_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let x = DynTensor::new(&x_data, &[1, 3, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 3, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = bn.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("BatchNorm translation should succeed")
        .graph;
    assert!(gn.num_nodes() > 0);

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 3, 4]), -5.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 3, 4]), 5.0_f32),
    )
    .expect("valid bounds");

    let output = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP should succeed for BatchNorm");
    assert_bounds_valid(&output);
}

// -- BatchNorm CROWN ----------------------------------------------------------

#[test]
fn test_model_trace_batch_norm_crown() {
    let running_mean = DynTensor::new(&[0.5, -0.5], &[2], &cpu()).unwrap();
    let running_var = DynTensor::new(&[1.0, 0.5], &[2], &cpu()).unwrap();
    let gamma = DynTensor::new(&[2.0, 0.5], &[2], &cpu()).unwrap();
    let beta = DynTensor::new(&[0.1, -0.1], &[2], &cpu()).unwrap();
    let bn = BatchNorm::new(running_mean, running_var, Some(gamma), Some(beta), 1e-5).unwrap();

    let x_data: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.5).collect();
    let x = DynTensor::new(&x_data, &[2, 2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = bn.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("BatchNorm translation should succeed")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 2, 3]), -3.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 2, 3]), 3.0_f32),
    )
    .expect("valid bounds");

    let (_method, crown_output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    assert_bounds_valid(&crown_output);
}
