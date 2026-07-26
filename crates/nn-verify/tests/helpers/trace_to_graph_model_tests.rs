// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for ComputationGraph → GraphNetwork via gamma-build's
//! `build_graph_network(GraphBuildInputs)` API.
//!
//! Exercises the `trace_to_graph_model` path (LayerSpec → build_graph_network)
//! covering Linear, Conv1d, ReLU, Sigmoid, InstanceNorm, sin/cos, and
//! multi-layer composition.
//!
//! ConvTranspose1d, RmsNorm, and BatchNorm tests live in
//! `trace_to_graph_model_misc_ops.rs` (deduplicated by P1-294 audit).

use super::common::assert_bounds_valid;
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{InstanceNorm, Linear, Module};
use nn_core::{DType, Device};
use nn_verify::{trace_to_graph_model, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

fn cpu() -> Device {
    Device::Cpu
}

// -- Test 1: Single Linear layer via build_graph_network ----------------------

#[test]
fn test_model_trace_linear() {
    let weight = DynTensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &cpu()).unwrap();
    let bias = DynTensor::new(&[0.5, -0.5], &[2], &cpu()).unwrap();
    let linear = Linear::new(weight, Some(bias)).unwrap();

    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = linear.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    // Translate via gamma-build path
    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should succeed")
        .graph;
    assert!(gn.num_nodes() > 0, "GraphNetwork should have nodes");

    // Propagate IBP bounds
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2, 2]), 1.0_f32),
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

// -- Test 2: Conv1d + ReLU chain via build_graph_network ----------------------

#[test]
fn test_model_trace_conv_relu() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0, 1.0], &[1, 1, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let mut k = k.clone();
        let id_x = record_input(&[1, 1, 5], DType::F32).unwrap();
        x.set_trace_id(id_x);
        let id_k = record_input(&[1, 1, 3], DType::F32).unwrap();
        k.set_trace_id(id_k);
        let conv_out = x.conv1d(&k, 0, 1, 1, 1)?;
        let y = conv_out.relu()?;
        Ok(y)
    })
    .unwrap();

    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![6.0, 9.0, 12.0]);

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should succeed")
        .graph;
    assert!(gn.num_nodes() > 0);

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 5]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 5]), 5.0_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "lower >= 0 after relu, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 15.01, "upper <= 15, got {v}");
    }
}

// -- Test 3: Multi-layer model e2e (Linear + Sigmoid + Linear) ----------------

#[test]
fn test_model_trace_multi_layer_e2e() {
    let w1 = DynTensor::new(&[0.5, 0.3, 0.2, 0.8], &[2, 2], &cpu()).unwrap();
    let b1 = DynTensor::new(&[0.1, -0.1], &[2], &cpu()).unwrap();
    let layer1 = Linear::new(w1, Some(b1)).unwrap();

    let w2 = DynTensor::new(&[1.0, -1.0, 0.5, 0.5], &[2, 2], &cpu()).unwrap();
    let b2 = DynTensor::new(&[0.0, 0.0], &[2], &cpu()).unwrap();
    let layer2 = Linear::new(w2, Some(b2)).unwrap();

    let x = DynTensor::new(&[1.0, 2.0], &[1, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let h = layer1.forward(&x)?;
        let h = h.sigmoid()?;
        let y = layer2.forward(&h)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should succeed")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2]), -2.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 2]), 2.0_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // Sigmoid output is in (0, 1), so layer2 output is bounded
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v.is_finite(), "lower bound finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "upper bound finite, got {v}");
    }
}

// -- Test 4: InstanceNorm via build_graph_network -----------------------------

#[test]
fn test_model_trace_instance_norm() {
    let eps = 1e-5;
    let instance_norm = InstanceNorm::new(eps).unwrap();

    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 2, 4],
        &cpu(),
    )
    .unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = instance_norm.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should succeed")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 4]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 2, 4]), 10.0_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);
}

// -- Test 5: Unary ops (sin, cos) via build_graph_network ---------------------

#[test]
fn test_model_trace_sin_cos() {
    let x = DynTensor::new(&[0.0, 1.0, 2.0, 3.0], &[1, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let s = x.sin()?;
        let y = s.cos()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should succeed")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4]), -3.15_f32),
        ArrayD::from_elem(IxDyn(&[1, 4]), 3.15_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // cos(sin(x)): sin output in [-1, 1], cos of [-1, 1] in [cos(1), 1]
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -1.01, "cos output >= -1, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.01, "cos output <= 1, got {v}");
    }
}

// Tests for Floor, Transpose, GroupNorm, ConvTranspose1d, LayerNorm, RmsNorm,
// BatchNorm live in trace_to_graph_model_misc_ops.rs and
// trace_to_graph_model_norm_ops.rs — deduplicated by P1-295 and W3-76 audits.

// -- LSTM forward_seq trace_node_id verification (Part of #2369) ---------------
//
// Verifies that Lstm::forward_seq records trace_node_ids on the output tensor
// when running inside trace_graph(). The fix at commit 42a9d0a ensures the
// fused GPU path is skipped during tracing so per-timestep TraceOp::Lstm
// nodes are recorded.

#[test]
fn test_lstm_forward_seq_trace_ids() {
    use nn_core::layers::Lstm;

    let input_size = 4;
    let hidden_size = 3;
    let batch = 1;
    let seq_len = 3;

    let four_h = 4 * hidden_size;
    // Small deterministic weights
    let w_ih = DynTensor::new(
        &vec![0.1f32; four_h * input_size],
        &[four_h, input_size],
        &cpu(),
    )
    .unwrap();
    let w_hh = DynTensor::new(
        &vec![0.1f32; four_h * hidden_size],
        &[four_h, hidden_size],
        &cpu(),
    )
    .unwrap();
    let b_ih = DynTensor::new(&vec![0.0f32; four_h], &[four_h], &cpu()).unwrap();
    let b_hh = DynTensor::new(&vec![0.0f32; four_h], &[four_h], &cpu()).unwrap();

    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden_size).unwrap();

    let x = DynTensor::new(
        &vec![0.5f32; seq_len * batch * input_size],
        &[seq_len, batch, input_size],
        &cpu(),
    )
    .unwrap();

    let ((output, final_state), graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[seq_len, batch, input_size], DType::F32).unwrap();
        x.set_trace_id(id);
        let (out, state) = lstm.forward_seq(&x, None)?;
        Ok((out, state))
    })
    .unwrap();

    // Output should have a trace_node_id (from the final cat of per-timestep outputs)
    assert!(
        output.trace_id().is_some(),
        "forward_seq output should have trace_node_id during tracing"
    );

    // Output shape should be [seq_len, batch, hidden_size]
    assert_eq!(output.dims(), &[seq_len, batch, hidden_size]);

    // Graph should have recorded at least some nodes (input + LSTM).
    // NOTE: The current trace records only 2 nodes (input + cat output),
    // not per-timestep LSTM ops. Per-timestep trace recording requires
    // graph builder changes for sequence models. (#2329 known limitation)
    assert!(
        graph.nodes().len() >= 2,
        "graph should have at least 2 nodes: got {}",
        graph.nodes().len()
    );

    // Verify final_state has shape [batch, hidden_size]
    assert_eq!(final_state.h.dims(), &[batch, hidden_size]);
    assert_eq!(final_state.c.dims(), &[batch, hidden_size]);

    // Note: full trace_to_graph_model translation of multi-timestep forward_seq
    // is not tested here — the Cat node aggregating per-timestep outputs
    // has dangling internal references that the graph builder can't resolve.
    // The single-cell LSTM trace-to-graph path is tested in
    // trace_to_graph_model_silero_vad_synthetic.rs.
}

// -- AdaIn forward_style trace_node_id verification (Part of #2370) ------------
//
// Verifies that AdaIn::forward_style records trace_node_id on the output tensor
// when running inside trace_graph(). The fix at commit fdfaf84 wraps the
// inner InstanceNorm with traced_forward to ensure trace chain continuity.

#[test]
fn test_adain_forward_style_trace_id() {
    use nn_core::layers::AdaIn;

    let num_channels = 2;
    let style_dim = 4;
    let t = 3;
    let batch = 1;

    // AdaIn projects style -> [2*C] (gamma and beta per channel)
    let proj_w = DynTensor::new(
        &vec![0.1f32; (2 * num_channels) * style_dim],
        &[2 * num_channels, style_dim],
        &cpu(),
    )
    .unwrap();
    let style_linear = Linear::new(proj_w, None).unwrap();
    let adain = AdaIn::new(style_linear, 1e-5).unwrap();

    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[batch, num_channels, t],
        &cpu(),
    )
    .unwrap();
    let style = DynTensor::new(&[0.1, 0.2, 0.3, 0.4], &[batch, style_dim], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let mut style = style.clone();
        let id_x = record_input(&[batch, num_channels, t], DType::F32).unwrap();
        let id_s = record_input(&[batch, style_dim], DType::F32).unwrap();
        x.set_trace_id(id_x);
        style.set_trace_id(id_s);
        let y = adain.forward_style(&x, &style)?;
        Ok(y)
    })
    .unwrap();

    // Output should have a trace_node_id (from the affine transform after norm)
    assert!(
        result.trace_id().is_some(),
        "AdaIn forward_style output should have trace_node_id during tracing"
    );

    // Output shape should match input spatial shape
    assert_eq!(result.dims(), &[batch, num_channels, t]);

    // Graph should have recorded nodes
    assert!(
        graph.nodes().len() > 2,
        "graph should have more than just input nodes: got {}",
        graph.nodes().len()
    );

    // Note: full trace_to_graph_model translation of AdaIn's composed ops
    // (InstanceNorm + style projection + affine transform) has dangling internal
    // references that the graph builder can't resolve yet. The basic InstanceNorm
    // trace-to-graph path is tested in test_model_trace_instance_norm above.
    // This test verifies the trace_node_id propagation (#2370 fix), which is the
    // prerequisite for graph translation to work once the builder is extended.
}
