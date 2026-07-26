// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for LSTM-based model trace-to-graph translation
//! via the `trace_to_graph_model` (LayerSpec → build_graph_network) path.
//!
//! Mirrors `trace_silero_vad_synthetic.rs` (old `trace_to_graph_network` path)
//! to ensure equivalent coverage on the new path. Exercises both standalone
//! LSTM cells and the synthetic Silero VAD architecture
//! (4×Conv1d+ReLU → Reshape → LSTM → Linear → Sigmoid).

use super::common::{assert_bounds_valid, assert_crown_tighter_when_not_fallback, conv1d_out_len};
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv1d, Conv1dConfig, Linear, Lstm, Module};
use nn_core::{DType, Device};
use nn_verify::{trace_to_graph_model, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

fn cpu() -> Device {
    Device::Cpu
}

/// Deterministic pseudo-random weight generation for reproducible tests.
fn seeded_weights(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let t = ((state >> 33) as f32) / (u32::MAX as f32);
            t * 0.6 - 0.3
        })
        .collect()
}

// -- Test 1: Single LSTM cell trace → graph → IBP -------------------------

#[test]
fn test_model_trace_lstm_cell_to_graph() {
    let input_size = 4;
    let hidden_size = 3;
    let batch = 1;

    let four_h = 4 * hidden_size;
    let w_ih_data = seeded_weights(42, four_h * input_size);
    let w_hh_data = seeded_weights(43, four_h * hidden_size);
    let b_ih_data = seeded_weights(44, four_h);
    let b_hh_data = seeded_weights(45, four_h);

    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], &cpu()).unwrap();
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden_size], &cpu()).unwrap();
    let b_ih = DynTensor::new(&b_ih_data, &[four_h], &cpu()).unwrap();
    let b_hh = DynTensor::new(&b_hh_data, &[four_h], &cpu()).unwrap();

    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden_size).unwrap();

    let x_data = seeded_weights(99, batch * input_size);
    let x = DynTensor::new(&x_data, &[batch, input_size], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[batch, input_size], DType::F32).unwrap();
        x.set_trace_id(id);
        let (output, _state) = lstm.forward(&x, None)?;
        Ok(output)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should succeed")
        .graph;
    assert!(gn.num_nodes() > 0, "GraphNetwork should have nodes");

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, input_size]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[batch, input_size]), 1.0_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // LSTM output goes through tanh, so bounds should be in (-1, 1)
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(
            v >= -1.01,
            "LSTM output lower >= -1 (tanh bounded), got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(v <= 1.01, "LSTM output upper <= 1 (tanh bounded), got {v}");
    }
}

// -- Test 2: Full synthetic Silero VAD architecture → IBP ------------------

fn build_seeded_conv1d(
    seed_w: u64,
    seed_b: u64,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
) -> Conv1d {
    let w = DynTensor::new(
        &seeded_weights(seed_w, out_ch * in_ch * kernel_size),
        &[out_ch, in_ch, kernel_size],
        &cpu(),
    )
    .unwrap();
    let b = DynTensor::new(&seeded_weights(seed_b, out_ch), &[out_ch], &cpu()).unwrap();
    Conv1d::new(w, Some(b), Conv1dConfig::default()).unwrap()
}

fn build_silero_vad_layers(t4: usize) -> (Conv1d, Conv1d, Conv1d, Conv1d, Lstm, Linear) {
    let conv1 = build_seeded_conv1d(100, 101, 1, 48, 3);
    let conv2 = build_seeded_conv1d(102, 103, 48, 96, 3);
    let conv3 = build_seeded_conv1d(104, 105, 96, 192, 3);
    let conv4 = build_seeded_conv1d(106, 107, 192, 256, 3);

    let lstm_input_size = 256 * t4;
    let lstm_hidden_size = 64;
    let four_h = 4 * lstm_hidden_size;
    let w_ih = DynTensor::new(
        &seeded_weights(200, four_h * lstm_input_size),
        &[four_h, lstm_input_size],
        &cpu(),
    )
    .unwrap();
    let w_hh = DynTensor::new(
        &seeded_weights(201, four_h * lstm_hidden_size),
        &[four_h, lstm_hidden_size],
        &cpu(),
    )
    .unwrap();
    let b_ih = DynTensor::new(&seeded_weights(202, four_h), &[four_h], &cpu()).unwrap();
    let b_hh = DynTensor::new(&seeded_weights(203, four_h), &[four_h], &cpu()).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), lstm_hidden_size).unwrap();

    let lin_w = DynTensor::new(
        &seeded_weights(300, lstm_hidden_size),
        &[1, lstm_hidden_size],
        &cpu(),
    )
    .unwrap();
    let lin_b = DynTensor::new(&seeded_weights(301, 1), &[1], &cpu()).unwrap();
    let linear = Linear::new(lin_w, Some(lin_b)).unwrap();

    (conv1, conv2, conv3, conv4, lstm, linear)
}

#[test]
fn test_model_trace_silero_vad_architecture_ibp() {
    let batch = 1;
    let time_steps = 16;
    let in_channels = 1;

    let t1 = conv1d_out_len(time_steps, 3, 1, 0);
    let t2 = conv1d_out_len(t1, 3, 1, 0);
    let t3 = conv1d_out_len(t2, 3, 1, 0);
    let t4 = conv1d_out_len(t3, 3, 1, 0);
    assert!(t4 > 0, "temporal dim must be > 0 after 4 convolutions");

    let (conv1, conv2, conv3, conv4, lstm, linear) = build_silero_vad_layers(t4);

    let x_data = seeded_weights(999, batch * in_channels * time_steps);
    let x = DynTensor::new(&x_data, &[batch, in_channels, time_steps], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[batch, in_channels, time_steps], DType::F32).unwrap();
        x.set_trace_id(id);

        let h = conv1.forward(&x)?;
        let h = h.relu()?;
        let h = conv2.forward(&h)?;
        let h = h.relu()?;
        let h = conv3.forward(&h)?;
        let h = h.relu()?;
        let h = conv4.forward(&h)?;
        let h = h.relu()?;

        let flat_dim = 256 * t4;
        let h = h.reshape([batch, flat_dim])?;

        let (h, _state) = lstm.forward(&h, None)?;
        let h = linear.forward(&h)?;
        let y = h.sigmoid()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should succeed")
        .graph;
    assert!(gn.num_nodes() > 0, "GraphNetwork should have nodes");

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, in_channels, time_steps]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[batch, in_channels, time_steps]), 1.0_f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "sigmoid output lower >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.01, "sigmoid output upper <= 1, got {v}");
    }
}

// -- Test 3: Full synthetic Silero VAD architecture → CROWN ----------------

#[test]
fn test_model_trace_silero_vad_architecture_crown() {
    let batch = 1;
    let time_steps = 16;
    let in_channels = 1;

    let t1 = conv1d_out_len(time_steps, 3, 1, 0);
    let t2 = conv1d_out_len(t1, 3, 1, 0);
    let t3 = conv1d_out_len(t2, 3, 1, 0);
    let t4 = conv1d_out_len(t3, 3, 1, 0);

    let (conv1, conv2, conv3, conv4, lstm, linear) = build_silero_vad_layers(t4);

    let x_data = seeded_weights(999, batch * in_channels * time_steps);
    let x = DynTensor::new(&x_data, &[batch, in_channels, time_steps], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[batch, in_channels, time_steps], DType::F32).unwrap();
        x.set_trace_id(id);

        let h = conv1.forward(&x)?;
        let h = h.relu()?;
        let h = conv2.forward(&h)?;
        let h = h.relu()?;
        let h = conv3.forward(&h)?;
        let h = h.relu()?;
        let h = conv4.forward(&h)?;
        let h = h.relu()?;

        let flat_dim = 256 * t4;
        let h = h.reshape([batch, flat_dim])?;

        let (h, _state) = lstm.forward(&h, None)?;
        let h = linear.forward(&h)?;
        let y = h.sigmoid()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should succeed")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, in_channels, time_steps]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[batch, in_channels, time_steps]), 1.0_f32),
    )
    .expect("valid bounds");

    let (method, output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&gn, &input_bounds);

    assert_bounds_valid(&output);

    eprintln!(
        "Silero VAD CROWN test (model path): method={method:?}, fallback={:?}",
        fallback_reason.as_deref().unwrap_or("none")
    );

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "sigmoid output lower >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.01, "sigmoid output upper <= 1, got {v}");
    }
}
