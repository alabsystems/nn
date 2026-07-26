// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for pool-op trace-to-graph translation via the
//! `trace_to_graph_model` (LayerSpec → build_graph_network) path.
//!
//! Covers AvgPool2d and MaxPool2d — previously untested in both old
//! and new translation paths (P1-294 proof coverage finding #1).

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_verify::trace_to_graph_model;

fn cpu() -> Device {
    Device::Cpu
}

// -- AvgPool2d IBP ------------------------------------------------------------

#[test]
fn test_model_trace_avg_pool2d_ibp() {
    // Input: [1, 1, 4, 4] — single-channel 4×4 image.
    let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let x = DynTensor::new(&data, &[1, 1, 4, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 4, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.avg_pool2d(2, 2, 0)?;
        Ok(y)
    })
    .unwrap();

    // Output shape: [1, 1, 2, 2]
    assert_eq!(_result.dims(), &[1, 1, 2, 2]);

    let gn = trace_to_graph_model(&graph)
        .expect("AvgPool2d translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 1, 4, 4], 16.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // Bounds should be within [-16, 16] range (input bounds).
    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -16.0 - 1e-3, "avg_pool lower bound out of range: {l}");
        assert!(u <= 16.0 + 1e-3, "avg_pool upper bound out of range: {u}");
    }
}

#[test]
fn test_model_trace_avg_pool2d_with_padding_ibp() {
    // Input: [1, 1, 4, 4] with padding=1, kernel=3, stride=1.
    let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let x = DynTensor::new(&data, &[1, 1, 4, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 4, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.avg_pool2d(3, 1, 1)?;
        Ok(y)
    })
    .unwrap();

    // With padding=1, stride=1, kernel=3 on 4×4 → output 4×4.
    assert_eq!(_result.dims(), &[1, 1, 4, 4]);

    let gn = trace_to_graph_model(&graph)
        .expect("AvgPool2d+pad translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 1, 4, 4], 16.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
}

// -- MaxPool2d IBP ------------------------------------------------------------

#[test]
fn test_model_trace_max_pool2d_ibp() {
    // Input: [1, 1, 4, 4].
    let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let x = DynTensor::new(&data, &[1, 1, 4, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 4, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.max_pool2d(2, 2, 0)?;
        Ok(y)
    })
    .unwrap();

    // Output shape: [1, 1, 2, 2].
    assert_eq!(_result.dims(), &[1, 1, 2, 2]);

    let gn = trace_to_graph_model(&graph)
        .expect("MaxPool2d translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 1, 4, 4], 16.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -16.0 - 1e-3, "max_pool lower bound out of range: {l}");
        assert!(u <= 16.0 + 1e-3, "max_pool upper bound out of range: {u}");
    }
}

#[test]
fn test_model_trace_max_pool2d_with_padding_ibp() {
    // Input: [1, 2, 6, 6] — multi-channel with padding.
    let data: Vec<f32> = (0..72).map(|v| v as f32 * 0.1).collect();
    let x = DynTensor::new(&data, &[1, 2, 6, 6], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 6, 6], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.max_pool2d(3, 2, 1)?;
        Ok(y)
    })
    .unwrap();

    // With kernel=3, stride=2, padding=1 on 6×6 → output 3×3.
    assert_eq!(_result.dims(), &[1, 2, 3, 3]);

    let gn = trace_to_graph_model(&graph)
        .expect("MaxPool2d+pad translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 2, 6, 6], 7.2);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
}

// -- Multi-layer with pooling -------------------------------------------------

#[test]
fn test_model_trace_relu_avg_pool_chain() {
    // Input: [1, 1, 4, 4] → Relu → AvgPool2d → output [1, 1, 2, 2].
    let data: Vec<f32> = (-8..8).map(|v| v as f32).collect();
    let x = DynTensor::new(&data, &[1, 1, 4, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 4, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.relu()?;
        let z = y.avg_pool2d(2, 2, 0)?;
        Ok(z)
    })
    .unwrap();

    assert_eq!(_result.dims(), &[1, 1, 2, 2]);

    let gn = trace_to_graph_model(&graph)
        .expect("Relu+AvgPool2d chain")
        .graph;
    let input_bounds = uniform_bounds(&[1, 1, 4, 4], 8.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // After Relu, lower bound should be >= 0 (when input lb >= 0).
    // With uniform_bounds [-8, 8], Relu clips to [0, 8].
    // AvgPool2d preserves non-negative lower bound.
    let (lo, _hi) = output.lower_upper();
    for &l in lo.iter() {
        assert!(l >= -1e-3, "relu+avg_pool lower bound should be >= 0: {l}");
    }
}
