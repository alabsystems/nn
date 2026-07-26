// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for misc op trace-to-graph translation
//! via the `trace_to_graph_model` (LayerSpec → build_graph_network) path.
//!
//! Covers: Floor, Transpose, GroupNorm, ConvTranspose1d (IBP, CROWN,
//! groups/dilation/output_padding rejection), and a multi-layer
//! Linear→ReLU→Linear→Sigmoid end-to-end model.
//!
//! Normalization tests (LayerNorm, RmsNorm, BatchNorm) extracted to
//! `trace_to_graph_model_norm_ops.rs` for 500-line compliance.
//!
//! Mirrors corresponding tests in `trace_to_graph_tests.rs` (old
//! `trace_to_graph_network` path) to ensure equivalent coverage on the new path.

use super::common::assert_bounds_valid;
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{ConvTranspose1d, ConvTranspose1dConfig, GroupNorm, Linear, Module};
use nn_core::{DType, Device};
use nn_verify::{propagate_with_crown_fallback, trace_to_graph_model, BoundedTensor, PropMethod};
use ndarray::{ArrayD, IxDyn};

fn cpu() -> Device {
    Device::Cpu
}

// -- Multi-layer model: Linear→ReLU→Linear→Sigmoid IBP propagation -----------

#[test]
fn test_model_trace_ibp_propagation() {
    let w1: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.1 - 1.6) * 0.5).collect();
    let b1: Vec<f32> = (0..8).map(|i| (i as f32) * 0.05 - 0.2).collect();
    let linear1 = Linear::new(
        DynTensor::new(&w1, &[8, 4], &cpu()).unwrap(),
        Some(DynTensor::new(&b1, &[8], &cpu()).unwrap()),
    )
    .unwrap();

    let w2: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.15 - 1.2) * 0.3).collect();
    let linear2 = Linear::new(
        DynTensor::new(&w2, &[2, 8], &cpu()).unwrap(),
        Some(DynTensor::new(&[0.1, -0.1], &[2], &cpu()).unwrap()),
    )
    .unwrap();

    let x = DynTensor::new(&[0.5, -0.3, 0.8, -0.1], &[1, 4], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let h = linear1.forward(&x)?;
        let h = h.relu()?;
        let h = linear2.forward(&h)?;
        let y = h.sigmoid()?;
        Ok(y)
    })
    .unwrap();

    // Forward pass sanity: sigmoid output must be in (0, 1)
    assert_eq!(result.dims(), &[1, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(v > 0.0 && v < 1.0, "sigmoid output in (0,1), got {v}");
    }

    let gn = trace_to_graph_model(&graph)
        .expect("multi-layer translation")
        .graph;
    assert!(
        gn.num_nodes() >= 4,
        "expected >=4 nodes, got {}",
        gn.num_nodes()
    );

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 4]), 1.0_f32),
    )
    .expect("valid bounds");

    let output = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP on multi-layer model");
    assert_bounds_valid(&output);

    // Sigmoid output bounds must be in [0, 1] (mathematically guaranteed)
    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[1, 2]);
    for &v in lo.iter() {
        assert!(v >= -1e-6, "sigmoid lo >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.0 + 1e-6, "sigmoid hi <= 1, got {v}");
    }
}

// -- Floor IBP ----------------------------------------------------------------

#[test]
fn test_model_trace_floor_ibp() {
    let x = DynTensor::new(&[1.5, -0.3, 2.7, 0.0], &[2, 2], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.floor()?;
        Ok(y)
    })
    .unwrap();

    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, -1.0, 2.0, 0.0]);

    let gn = trace_to_graph_model(&graph)
        .expect("translation should succeed")
        .graph;
    assert!(gn.num_nodes() > 0);

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 2]), -2.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 2]), 3.0_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v.is_finite(), "lower bound must be finite");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "upper bound must be finite");
    }
}

// -- Transpose IBP ------------------------------------------------------------

#[test]
fn test_model_trace_transpose_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.transpose(1, 2)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[1, 3, 2]);

    let gn = trace_to_graph_model(&graph)
        .expect("translation should succeed")
        .graph;
    assert!(gn.num_nodes() > 0);

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 3]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 2, 3]), 1.0_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v.is_finite(), "lower bound must be finite");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "upper bound must be finite");
    }
}

// -- GroupNorm IBP (decomposed) -----------------------------------------------

#[test]
fn test_model_trace_group_norm_ibp() {
    let num_groups = 2;
    let num_channels = 4;
    let eps: f64 = 1e-5;

    let weight = DynTensor::new(&[1.0, 0.5, 2.0, 1.5], &[num_channels], &cpu()).unwrap();
    let bias = DynTensor::new(&[0.1, -0.1, 0.2, -0.2], &[num_channels], &cpu()).unwrap();
    let gn_layer = GroupNorm::new(num_groups, num_channels, weight, bias, eps).unwrap();

    let x_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.3 - 1.0).collect();
    let x = DynTensor::new(&x_data, &[1, 4, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = gn_layer.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[1, 4, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(v.is_finite(), "GroupNorm output must be finite, got {v}");
    }

    let network = trace_to_graph_model(&graph)
        .expect("GroupNorm translation should succeed")
        .graph;
    assert!(
        network.num_nodes() >= 5,
        "expected >=5 nodes for GroupNorm decomposition, got {}",
        network.num_nodes()
    );

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 3]), -2.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 3]), 2.0_f32),
    )
    .expect("valid bounds");

    let output = network
        .propagate_ibp(&input_bounds)
        .expect("IBP propagation through GroupNorm");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[1, 4, 3]);
    for &v in lo.iter() {
        assert!(v.is_finite(), "lower bound must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "upper bound must be finite, got {v}");
    }
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

// -- ConvTranspose1d IBP ------------------------------------------------------

#[test]
fn test_model_trace_conv_transpose1d_ibp() {
    let weight = DynTensor::new(&[1.0, 0.5, -0.5, 0.3, -0.3, 0.7], &[2, 1, 3], &cpu()).unwrap();
    let bias = DynTensor::new(&[0.1], &[1], &cpu()).unwrap();
    let config = ConvTranspose1dConfig::default()
        .with_stride(1)
        .with_padding(0);
    let conv_t = ConvTranspose1d::new(weight, Some(bias), config).unwrap();

    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, -2.0],
        &[1, 2, 4],
        &cpu(),
    )
    .unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[1, 1, 6]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(
            v.is_finite(),
            "ConvTranspose1d output must be finite, got {v}"
        );
    }

    let gn = trace_to_graph_model(&graph)
        .expect("ConvTranspose1d translation")
        .graph;
    assert!(gn.num_nodes() > 0, "GraphNetwork should have nodes");

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 4]), -2.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 2, 4]), 2.0_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[1, 1, 6], "output shape mismatch");
    for &v in lo.iter() {
        assert!(v.is_finite(), "lower bound must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "upper bound must be finite, got {v}");
    }
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

// -- ConvTranspose1d CROWN ----------------------------------------------------

#[test]
fn test_model_trace_conv_transpose1d_crown() {
    let weight = DynTensor::new(&[1.0, 0.5, -0.5, 0.3, -0.3, 0.7], &[2, 1, 3], &cpu()).unwrap();
    let config = ConvTranspose1dConfig::default()
        .with_stride(1)
        .with_padding(0);
    let conv_t = ConvTranspose1d::new(weight, None, config).unwrap();

    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, -2.0],
        &[1, 2, 4],
        &cpu(),
    )
    .unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("translation").graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 4]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 2, 4]), 1.0_f32),
    )
    .expect("valid bounds");

    let (method, output, crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("propagation");

    assert_eq!(
        method,
        PropMethod::Crown,
        "Expected CROWN to succeed, but got IBP fallback. CROWN error: {crown_err:?}"
    );

    assert_bounds_valid(&output);
}

// -- ConvTranspose1d grouped/dilation parameters ------------------------------

/// Grouped ConvTranspose1d is supported (#2989): NY handles groups natively.
#[test]
fn test_model_trace_conv_transpose1d_groups_succeeds() {
    let weight = DynTensor::new(&[1.0, 0.5, -0.5, 0.3, -0.3, 0.7], &[2, 1, 3], &cpu()).unwrap();
    let config = ConvTranspose1dConfig::default().with_groups(2);
    let conv_t = ConvTranspose1d::new(weight, None, config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Graph should build successfully with grouped ConvTranspose1d.
    let network = trace_to_graph_model(&graph)
        .expect("grouped ConvTranspose1d should build")
        .graph;
    assert!(
        network.num_nodes() > 0,
        "graph should contain at least one node"
    );
}

#[test]
fn test_model_trace_conv_transpose1d_dilation_rejected() {
    let weight = DynTensor::new(&[1.0, 0.5, -0.5], &[1, 1, 3], &cpu()).unwrap();
    let config = ConvTranspose1dConfig::default().with_dilation(2);
    let conv_t = ConvTranspose1d::new(weight, None, config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 4], DType::F32).unwrap();
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

/// Output padding is handled via decomposition: ConvTranspose1d(output_padding=0)
/// followed by a right-side zero-pad via Linear (#2558).
#[test]
fn test_model_trace_conv_transpose1d_output_padding_decomposes() {
    let weight = DynTensor::new(&[1.0, 0.5, -0.5], &[1, 1, 3], &cpu()).unwrap();
    let config = ConvTranspose1dConfig::default()
        .with_stride(2)
        .with_output_padding(1);
    let conv_t = ConvTranspose1d::new(weight, None, config).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = conv_t.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Graph should build successfully via output_padding decomposition.
    let network = trace_to_graph_model(&graph)
        .expect("output_padding ConvTranspose1d should decompose successfully")
        .graph;
    assert!(
        network.num_nodes() > 0,
        "graph should contain at least one node"
    );
}

// -- Constant node: verify path translates TraceOp::Constant correctly --------

/// Regression test for #2399: Constant nodes (from `DynTensor::full()` /
/// `scalar_like()`) must be translated as weight tensors, not network inputs.
/// Before the fix, `translate_input` was called for Constants, creating a
/// dangling `"{name}_in"` reference (no matching input_spec).
#[test]
fn test_model_trace_constant_node_ibp() {
    // Build a simple graph: x + 2.0, where 2.0 is a Constant node.
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        // full() during tracing creates a TraceOp::Constant node.
        let c = DynTensor::full(&[2, 2], 2.0, DType::F32, &cpu())?;
        let y = x.add(&c)?;
        Ok(y)
    })
    .unwrap();

    // Translate: should succeed (no dangling tensor reference).
    let gn = trace_to_graph_model(&graph)
        .expect("Constant node translation should succeed")
        .graph;
    assert!(gn.num_nodes() > 0, "GraphNetwork should have nodes");

    // IBP propagation: x in [-1, 1], constant = 2.0 → output in [1, 3].
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 2]), 1.0_f32),
    )
    .expect("valid bounds");

    let output = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP should propagate through constant");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v.is_finite(), "lower bound must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "upper bound must be finite, got {v}");
    }
}
