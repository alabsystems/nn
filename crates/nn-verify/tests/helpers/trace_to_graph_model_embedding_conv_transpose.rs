// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Embedding and ConvTranspose2d trace-to-graph translation
//! via the `trace_to_graph_model` (LayerSpec → build_graph_network) path.
//!
//! Mirrors `trace_to_graph_embedding_conv_transpose.rs` (old `trace_to_graph_network` path)
//! to ensure equivalent coverage on the new path.

use super::common::assert_bounds_valid;
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    ConvTranspose1d, ConvTranspose1dConfig, ConvTranspose2d, ConvTranspose2dConfig, Embedding,
    Module,
};
use nn_core::{DType, Device};
use nn_verify::{propagate_with_crown_fallback, trace_to_graph_model, BoundedTensor, PropMethod};
use ndarray::{ArrayD, IxDyn};

fn cpu() -> Device {
    Device::Cpu
}

// -- Embedding trace → graph translation with IBP bounds -----------------------

#[test]
fn test_model_trace_embedding_ibp() {
    // 3 vocabulary entries × 4 embedding dims with known values
    let weight_data = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
        3.0, 4.0, 5.0, 6.0, // row 2
    ];
    let weight = DynTensor::new(&weight_data, &[3, 4], &cpu()).unwrap();
    let embedding = Embedding::new(weight).unwrap();

    // Input: single token index [1] (batch=1, seq=1)
    let x = DynTensor::from_vec_u32(vec![1], &[1, 1], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1], DType::U32).unwrap();
        x.set_trace_id(id);
        let y = embedding.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Embedding translation should succeed")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1]), 1.0_f32),
    )
    .expect("valid bounds");

    let (_method, output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("propagation");

    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.len(), 4, "expected 4 embedding dims in output");

    for &v in lo.iter() {
        assert!(v >= -10.0, "embedding lower bound unreasonably low: {v}");
        assert!(v.is_finite(), "lower bound must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 20.0, "embedding upper bound unreasonably high: {v}");
        assert!(v.is_finite(), "upper bound must be finite, got {v}");
    }
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

// -- Embedding non-finite weight rejection ------------------------------------

#[test]
fn test_model_trace_embedding_non_finite_weight_rejected() {
    let weight_data = vec![
        1.0,
        2.0,
        3.0,
        f32::NAN, // row 0 has NaN
        5.0,
        6.0,
        7.0,
        8.0, // row 1
    ];
    let weight = DynTensor::new(&weight_data, &[2, 4], &cpu()).unwrap();
    let embedding = Embedding::new(weight).unwrap();

    let x = DynTensor::from_vec_u32(vec![0], &[1, 1], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1], DType::U32).unwrap();
        x.set_trace_id(id);
        let y = embedding.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let err = trace_to_graph_model(&graph).unwrap_err();
    let err_str = err.to_string();
    assert!(
        err_str.contains("non-finite") || err_str.contains("NaN"),
        "Expected non-finite weight rejection, got: {err_str}"
    );
}

// -- ConvTranspose2d IBP propagation ------------------------------------------

#[test]
fn test_model_trace_conv_transpose2d_ibp() {
    let weight_data = vec![1.0, 0.5, -0.5, 0.3];
    let weight = DynTensor::new(&weight_data, &[1, 1, 2, 2], &cpu()).unwrap();
    let config = ConvTranspose2dConfig::default()
        .with_stride(1)
        .with_padding(0);
    let conv_t = ConvTranspose2d::new(weight, None, config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("ConvTranspose2d translation")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 1.0_f32),
    )
    .expect("valid bounds");

    let (_method, output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("propagation");

    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -10.0, "conv_t lower bound unreasonably low: {v}");
        assert!(v.is_finite(), "lower bound must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 10.0, "conv_t upper bound unreasonably high: {v}");
        assert!(v.is_finite(), "upper bound must be finite, got {v}");
    }
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
        assert!(l <= 0.01, "lower bound should contain 0, got {l}");
        assert!(u >= -0.01, "upper bound should contain 0, got {u}");
    }
}

// -- ConvTranspose2d CROWN propagation ----------------------------------------

#[test]
fn test_model_trace_conv_transpose2d_crown_succeeds() {
    let weight_data = vec![1.0, 0.5, -0.5, 0.3];
    let weight = DynTensor::new(&weight_data, &[1, 1, 2, 2], &cpu()).unwrap();
    let config = ConvTranspose2dConfig::default()
        .with_stride(1)
        .with_padding(0);
    let conv_t = ConvTranspose2d::new(weight, None, config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("translation").graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 1.0_f32),
    )
    .expect("valid bounds");

    let (method, output, crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("propagation");

    assert_eq!(
        method,
        PropMethod::Crown,
        "Expected CROWN success, but got IBP fallback. CROWN error: {crown_err:?}"
    );

    assert_bounds_valid(&output);
}

// -- ConvTranspose2d unsupported parameters -----------------------------------

/// Grouped ConvTranspose2d is supported: NY builder handles groups natively.
#[test]
fn test_model_trace_conv_transpose2d_groups_succeeds() {
    let weight_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
    let weight = DynTensor::new(&weight_data, &[2, 1, 2, 2], &cpu()).unwrap();
    let config = ConvTranspose2dConfig::default().with_groups(2);
    let conv_t = ConvTranspose2d::new(weight, None, config).unwrap();

    let x_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let x = DynTensor::new(&x_data, &[1, 2, 2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let network = trace_to_graph_model(&graph)
        .expect("grouped ConvTranspose2d should build")
        .graph;
    assert!(
        network.num_nodes() > 0,
        "graph should contain at least one node"
    );
}

#[test]
fn test_model_trace_conv_transpose2d_dilation_rejected() {
    let weight = DynTensor::new(&[1.0, 0.5, -0.5, 0.3], &[1, 1, 2, 2], &cpu()).unwrap();
    let config = ConvTranspose2dConfig::default().with_dilation(2);
    let conv_t = ConvTranspose2d::new(weight, None, config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let err = trace_to_graph_model(&graph).unwrap_err();
    let err_str = err.to_string();
    assert!(
        err_str.contains("dilation") || err_str.contains("not supported"),
        "Expected dilation rejection, got: {err_str}"
    );
}

#[test]
fn test_model_trace_conv_transpose2d_output_padding_rejected() {
    let weight = DynTensor::new(&[1.0, 0.5, -0.5, 0.3], &[1, 1, 2, 2], &cpu()).unwrap();
    let config = ConvTranspose2dConfig::default()
        .with_stride(2)
        .with_output_padding(1);
    let conv_t = ConvTranspose2d::new(weight, None, config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let err = trace_to_graph_model(&graph).unwrap_err();
    let err_str = err.to_string();
    assert!(
        err_str.contains("output_padding") || err_str.contains("not supported"),
        "Expected output_padding rejection, got: {err_str}"
    );
}

// -- ConvTranspose1d with output_padding decomposition (#2558) ----------------

#[test]
fn test_model_trace_conv_transpose1d_output_padding_decomposed() {
    // ConvTranspose1d: in_ch=1, out_ch=1, kernel=3, stride=2, output_padding=1
    // Input: [1, 1, 4] → T_mid = (4-1)*2 + 3 - 0 = 9, T_out = 9 + 1 = 10
    let weight_data = vec![0.5, 1.0, -0.5];
    let weight = DynTensor::new(&weight_data, &[1, 1, 3], &cpu()).unwrap();
    let bias_data = vec![0.1];
    let bias = DynTensor::new(&bias_data, &[1], &cpu()).unwrap();

    let config = ConvTranspose1dConfig::new(0, 2, 1).with_output_padding(1);
    let conv_t = ConvTranspose1d::new(weight, Some(bias), config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Translation should succeed (no longer rejected).
    let gn = trace_to_graph_model(&graph)
        .expect("ConvTranspose1d with output_padding should translate (#2558).graph")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 4]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 4]), 1.0_f32),
    )
    .expect("valid bounds");

    let (_method, output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("propagation");

    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    // T_out = 10 elements
    assert_eq!(
        lo.len(),
        10,
        "expected 10 output elements (T_mid=9 + output_padding=1), got {}",
        lo.len()
    );

    for &v in lo.iter() {
        assert!(v.is_finite(), "lower bound must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "upper bound must be finite, got {v}");
    }
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }

    // The last element (from output_padding) should have bounds [0, 0]
    // since the zero-pad linear layer maps it from a zero row.
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();
    let last_lo = lo_flat[lo_flat.len() - 1];
    let last_hi = hi_flat[hi_flat.len() - 1];
    assert!(
        last_lo.abs() < 0.2 && last_hi.abs() < 0.2,
        "output_padding element should have near-zero bounds, got [{last_lo}, {last_hi}]"
    );
}
