#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`Lstm`] and [`LstmState`].

use super::*;
use crate::{DType, Device, TensorError};

/// Helper: create a DynTensor from flat data and shape.
fn tensor(data: &[f32], shape: &[usize]) -> DynTensor {
    DynTensor::from_vec(data.to_vec(), shape, &Device::Cpu).expect("valid tensor")
}

/// Helper: create zero-filled tensor.
fn zeros(shape: &[usize]) -> DynTensor {
    DynTensor::zeros(shape, DType::F32, &Device::Cpu).expect("valid zeros")
}

#[test]
fn test_lstm_basic_forward() {
    // hidden=2, input=3, batch=1
    let h = 2;
    let input_size = 3;
    // w_ih: [4*H, input_size] = [8, 3], all 0.1
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    // w_hh: [4*H, H] = [8, 2], all 0.1
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();

    let lstm = Lstm::new(w_ih, w_hh, None, None, h).expect("valid LSTM");

    let input = DynTensor::full(&[1, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let (output, state) = lstm.forward(&input, None).expect("forward succeeds");

    // Check output shapes.
    assert_eq!(output.dims(), &[1, h]);
    assert_eq!(state.h.dims(), &[1, h]);
    assert_eq!(state.c.dims(), &[1, h]);

    // Values should be finite and non-zero (gates are non-trivial).
    let h_vals = state.h.to_flat_vec::<f32>().unwrap();
    let c_vals = state.c.to_flat_vec::<f32>().unwrap();
    for &v in h_vals.iter().chain(c_vals.iter()) {
        assert!(v.is_finite(), "all outputs must be finite");
    }
    assert!(h_vals[0].abs() > 1e-10, "h should be non-zero");
    assert!(c_vals[0].abs() > 1e-10, "c should be non-zero");
}

#[test]
fn test_lstm_with_bias() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let b_ih = Some(DynTensor::full(&[4 * h], 0.1, DType::F32, &Device::Cpu).unwrap());
    let b_hh = Some(DynTensor::full(&[4 * h], 0.05, DType::F32, &Device::Cpu).unwrap());

    let lstm = Lstm::new(w_ih, w_hh, b_ih, b_hh, h).expect("valid LSTM");
    let input = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let (_output, state) = lstm.forward(&input, None).unwrap();

    assert_eq!(state.h.dims(), &[1, h]);
    assert_eq!(state.c.dims(), &[1, h]);
    let h_vals = state.h.to_flat_vec::<f32>().unwrap();
    for &v in &h_vals {
        assert!(v.is_finite());
        assert!(v.abs() > 1e-10);
    }
}

#[test]
fn test_lstm_with_initial_state() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();

    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();
    let input = tensor(&[1.0, 2.0, 3.0], &[1, 3]);

    let init_state = LstmState::new(
        DynTensor::full(&[1, h], 0.5, DType::F32, &Device::Cpu).unwrap(),
        DynTensor::full(&[1, h], 0.3, DType::F32, &Device::Cpu).unwrap(),
    )
    .unwrap();

    let (_, result_with_state) = lstm.forward(&input, Some(&init_state)).unwrap();
    let (_, result_without_state) = lstm.forward(&input, None).unwrap();

    // Results should differ since state is non-zero vs zero-initialized.
    let h_with = result_with_state.h.to_flat_vec::<f32>().unwrap();
    let h_without = result_without_state.h.to_flat_vec::<f32>().unwrap();
    assert!(
        (h_with[0] - h_without[0]).abs() > 1e-6,
        "non-zero initial state should produce different output"
    );
}

#[test]
fn test_lstm_batch_size_2() {
    let h = 3;
    let input_size = 2;
    let batch = 2;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();

    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();
    // [batch=2, input_size=2]
    let input = tensor(&[1.0, 2.0, 3.0, 4.0], &[batch, input_size]);
    let (_output, state) = lstm.forward(&input, None).unwrap();

    assert_eq!(state.h.dims(), &[batch, h]);
    assert_eq!(state.c.dims(), &[batch, h]);

    // Batch items should have different outputs since inputs differ.
    let h_vals = state.h.to_flat_vec::<f32>().unwrap();
    assert!(
        (h_vals[0] - h_vals[h]).abs() > 1e-6,
        "different batch inputs should produce different outputs"
    );
}

#[test]
fn test_lstm_forward_seq() {
    let h = 2;
    let input_size = 3;
    let seq_len = 4;
    let batch = 1;

    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    // [seq_len=4, batch=1, input_size=3]
    let input =
        DynTensor::full(&[seq_len, batch, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();

    let (outputs, final_state) = lstm.forward_seq(&input, None).unwrap();

    assert_eq!(outputs.dims(), &[seq_len, batch, h]);
    assert_eq!(final_state.h.dims(), &[batch, h]);
    assert_eq!(final_state.c.dims(), &[batch, h]);

    // Final state h should match the last output frame.
    let last_output = outputs
        .narrow(0, seq_len - 1, 1)
        .unwrap()
        .squeeze(0)
        .unwrap();
    let last_h = last_output.to_flat_vec::<f32>().unwrap();
    let final_h = final_state.h.to_flat_vec::<f32>().unwrap();
    for i in 0..h {
        assert!(
            (last_h[i] - final_h[i]).abs() < 1e-7,
            "final h should match last output: {} vs {}",
            last_h[i],
            final_h[i]
        );
    }
}

#[test]
fn test_lstm_step_by_step_matches_seq() {
    let h = 2;
    let input_size = 3;
    let seq_len = 3;
    let batch = 1;

    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.05, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    let input_seq =
        DynTensor::full(&[seq_len, batch, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();

    // Forward via forward_seq.
    let (_seq_outputs, seq_final) = lstm.forward_seq(&input_seq, None).unwrap();

    // Forward step-by-step.
    let mut state: Option<LstmState> = None;
    for t in 0..seq_len {
        let x_t = input_seq.narrow(0, t, 1).unwrap().squeeze(0).unwrap();
        let (_, new_state) = lstm.forward(&x_t, state.as_ref()).unwrap();
        state = Some(new_state);
    }
    let step_final = state.unwrap();

    // Final states should match exactly.
    let seq_h = seq_final.h.to_flat_vec::<f32>().unwrap();
    let step_h = step_final.h.to_flat_vec::<f32>().unwrap();
    for i in 0..h {
        assert!(
            (seq_h[i] - step_h[i]).abs() < 1e-7,
            "seq vs step h mismatch at {i}: {} vs {}",
            seq_h[i],
            step_h[i]
        );
    }
}

/// Comprehensive parity test for batched input matmul (#2679).
///
/// Covers: biases, batch>1, non-uniform inputs, and initial state — all
/// code paths modified by the CPU batched input-to-gate optimization.
#[test]
fn test_lstm_batched_matmul_parity_with_bias_and_batch() {
    let h = 4;
    let input_size = 6;
    let seq_len = 5;
    let batch = 3;
    let four_h = 4 * h;

    // Non-uniform weights for realistic gate behavior.
    let w_ih_data: Vec<f32> = (0..four_h * input_size)
        .map(|i| ((i as f32) * 0.017).sin() * 0.3)
        .collect();
    let w_hh_data: Vec<f32> = (0..four_h * h)
        .map(|i| ((i as f32) * 0.031).cos() * 0.2)
        .collect();
    let b_ih_data: Vec<f32> = (0..four_h).map(|i| (i as f32) * 0.01 - 0.08).collect();
    let b_hh_data: Vec<f32> = (0..four_h).map(|i| (i as f32) * -0.005 + 0.04).collect();

    let w_ih = DynTensor::from_vec(w_ih_data, &[four_h, input_size], &Device::Cpu).unwrap();
    let w_hh = DynTensor::from_vec(w_hh_data, &[four_h, h], &Device::Cpu).unwrap();
    let b_ih = Some(DynTensor::from_vec(b_ih_data, &[four_h], &Device::Cpu).unwrap());
    let b_hh = Some(DynTensor::from_vec(b_hh_data, &[four_h], &Device::Cpu).unwrap());

    let lstm = Lstm::new(w_ih, w_hh, b_ih, b_hh, h).unwrap();

    // Non-uniform input: different values per timestep and batch element.
    let input_data: Vec<f32> = (0..seq_len * batch * input_size)
        .map(|i| ((i as f32) * 0.13).sin())
        .collect();
    let input_seq =
        DynTensor::from_vec(input_data, &[seq_len, batch, input_size], &Device::Cpu).unwrap();

    // Non-zero initial state.
    let init_h_data: Vec<f32> = (0..batch * h).map(|i| (i as f32) * 0.1 - 0.2).collect();
    let init_c_data: Vec<f32> = (0..batch * h).map(|i| (i as f32) * -0.05 + 0.15).collect();
    let init_state = LstmState::new(
        DynTensor::from_vec(init_h_data, &[batch, h], &Device::Cpu).unwrap(),
        DynTensor::from_vec(init_c_data, &[batch, h], &Device::Cpu).unwrap(),
    )
    .unwrap();

    // forward_seq uses the batched input matmul optimization.
    let (seq_outputs, seq_final) = lstm.forward_seq(&input_seq, Some(&init_state)).unwrap();

    // Step-by-step uses the original forward path (no batching).
    let mut state: Option<LstmState> = Some(init_state);
    let mut step_outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let x_t = input_seq.narrow(0, t, 1).unwrap().squeeze(0).unwrap();
        let (h_out, new_state) = lstm.forward(&x_t, state.as_ref()).unwrap();
        step_outputs.push(h_out.unsqueeze(0).unwrap());
        state = Some(new_state);
    }
    let step_final = state.unwrap();
    let step_refs: Vec<&DynTensor> = step_outputs.iter().collect();
    let step_stacked = DynTensor::cat(&step_refs, 0).unwrap();

    // Compare all outputs (every timestep, every batch element).
    let seq_vals = seq_outputs.to_flat_vec::<f32>().unwrap();
    let step_vals = step_stacked.to_flat_vec::<f32>().unwrap();
    assert_eq!(seq_vals.len(), step_vals.len());
    for (i, (&s, &r)) in seq_vals.iter().zip(step_vals.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-5,
            "output mismatch at flat index {i}: seq={s}, step={r}, diff={}",
            (s - r).abs()
        );
    }

    // Compare final hidden and cell states.
    let seq_h = seq_final.h.to_flat_vec::<f32>().unwrap();
    let step_h = step_final.h.to_flat_vec::<f32>().unwrap();
    let seq_c = seq_final.c.to_flat_vec::<f32>().unwrap();
    let step_c = step_final.c.to_flat_vec::<f32>().unwrap();
    for i in 0..seq_h.len() {
        assert!(
            (seq_h[i] - step_h[i]).abs() < 1e-5,
            "final h mismatch at {i}: {} vs {}",
            seq_h[i],
            step_h[i]
        );
        assert!(
            (seq_c[i] - step_c[i]).abs() < 1e-5,
            "final c mismatch at {i}: {} vs {}",
            seq_c[i],
            step_c[i]
        );
    }
}

#[test]
fn test_lstm_invalid_w_ih_shape() {
    let h = 2;
    // Wrong: w_ih should be [8, 3] not [3, 3]
    let w_ih = zeros(&[3, 3]);
    let w_hh = zeros(&[8, 2]);
    let err = Lstm::new(w_ih, w_hh, None, None, h).unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected shape mismatch for w_ih, got: {err}"
    );
}

#[test]
fn test_lstm_invalid_w_hh_shape() {
    let h = 2;
    let w_ih = zeros(&[8, 3]);
    // Wrong: w_hh should be [8, 2] not [8, 5]
    let w_hh = zeros(&[8, 5]);
    let err = Lstm::new(w_ih, w_hh, None, None, h).unwrap_err();
    assert!(
        matches!(err, TensorError::ShapeMismatch { .. }),
        "expected shape mismatch for w_hh, got: {err}"
    );
}

#[test]
fn test_lstm_invalid_input_rank() {
    let h = 2;
    let input_size = 3;
    let w_ih = zeros(&[8, input_size]);
    let w_hh = zeros(&[8, h]);
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    // 1D input should fail (needs [batch, input_size]).
    let input_1d = tensor(&[1.0, 2.0, 3.0], &[3]);
    let err = lstm.forward(&input_1d, None).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 2, .. }),
        "expected rank mismatch for input, got: {err}"
    );
}

#[test]
fn test_lstm_zero_input_produces_output() {
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    // Zero input with zero state: gates are all sigmoid(0)=0.5, tanh(0)=0
    // i=0.5, f=0.5, g=0, o=0.5 → c_new = 0.5*0 + 0.5*0 = 0, h_new = 0.5*tanh(0) = 0
    let input = zeros(&[1, input_size]);
    let (_output, state) = lstm.forward(&input, None).unwrap();
    let h_vals = state.h.to_flat_vec::<f32>().unwrap();
    let c_vals = state.c.to_flat_vec::<f32>().unwrap();
    for &v in h_vals.iter().chain(c_vals.iter()) {
        assert!(
            v.abs() < 1e-7,
            "zero input/state should produce zero output, got {v}"
        );
    }
}

#[test]
fn test_lstm_accessors() {
    let h = 2;
    let w_ih = zeros(&[8, 3]);
    let w_hh = zeros(&[8, 2]);
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    assert_eq!(lstm.hidden_size(), 2);
    assert_eq!(lstm.w_ih().dims(), &[8, 3]);
    assert_eq!(lstm.w_hh().dims(), &[8, 2]);
}

#[test]
fn test_lstm_output_finiteness_validation() {
    // All standard forward paths should produce finite output.
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    let input = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let (_output, state) = lstm.forward(&input, None).unwrap();

    // Verify that outputs are validated as finite (AC5).
    let h_vals = state.h.to_flat_vec::<f32>().unwrap();
    let c_vals = state.c.to_flat_vec::<f32>().unwrap();
    assert!(h_vals.iter().all(|v| v.is_finite()));
    assert!(c_vals.iter().all(|v| v.is_finite()));
}

#[test]
fn test_lstm_cell_alias_works() {
    // LstmCell is a type alias for Lstm (candle-nn compat).
    let h = 2;
    let w_ih = zeros(&[8, 3]);
    let w_hh = zeros(&[8, 2]);
    let cell = LstmCell::new(w_ih, w_hh, None, None, h).unwrap();
    assert_eq!(cell.hidden_size(), 2);
}

#[test]
fn test_lstm_forward_seq_empty_sequence() {
    // seq_len=0 should return an error, not panic (#979 audit finding).
    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    // [0, 1, 3] — seq_len=0
    let input = DynTensor::zeros(&[0, 1, input_size], DType::F32, &Device::Cpu).unwrap();
    let result = lstm.forward_seq(&input, None);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("seq_len=0 should return error, not panic"),
    };
    assert!(
        matches!(err, TensorError::ZeroLengthDimension { axis: 0, .. }),
        "expected zero-length dimension error, got: {err}"
    );
}

/// Multi-step LSTM cell state boundedness test.
///
/// The Kani proof `lstm_cell_state_bounded` proves single-step: if |c| < 100
/// then |c_new| < 101. This test empirically verifies the inductive claim
/// over 1000 timesteps — cell state should not grow unboundedly when weights
/// are small and inputs are bounded.
///
/// Exercises the production `forward_seq` path, not a synthetic gate formula.
#[test]
fn test_lstm_cell_state_bounded_multi_step() {
    let input_size = 4;
    let hidden_size = 8;
    let seq_len = 1000;
    let batch = 1;

    // Small weights → sigmoid outputs near 0.5, tanh outputs near 0.
    let four_h = 4 * hidden_size;
    let w_ih = tensor(
        &(0..four_h * input_size)
            .map(|i| ((i as f32) * 0.003).sin() * 0.05)
            .collect::<Vec<_>>(),
        &[four_h, input_size],
    );
    let w_hh = tensor(
        &(0..four_h * hidden_size)
            .map(|i| ((i as f32) * 0.007).cos() * 0.05)
            .collect::<Vec<_>>(),
        &[four_h, hidden_size],
    );
    let lstm = Lstm::new(w_ih, w_hh, None, None, hidden_size).unwrap();

    let input_data: Vec<f32> = (0..seq_len * batch * input_size)
        .map(|i| ((i as f32) * 0.1).sin() * 0.5)
        .collect();
    let input =
        DynTensor::from_vec(input_data, &[seq_len, batch, input_size], &Device::Cpu).unwrap();

    let (_, final_state) = lstm.forward_seq(&input, None).unwrap();

    // Cell state should remain bounded after 1000 steps.
    let c_vals = final_state.c.to_flat_vec::<f32>().unwrap();
    for (i, &c) in c_vals.iter().enumerate() {
        assert!(
            c.is_finite(),
            "cell state[{i}] is not finite after {seq_len} steps: {c}"
        );
        assert!(
            c.abs() < 100.0,
            "cell state[{i}] grew unboundedly after {seq_len} steps: {c}"
        );
    }

    let h_vals = final_state.h.to_flat_vec::<f32>().unwrap();
    for (i, &h) in h_vals.iter().enumerate() {
        assert!(
            h.is_finite(),
            "hidden state[{i}] is not finite after {seq_len} steps: {h}"
        );
        // tanh output is always in [-1, 1]
        assert!(
            h.abs() <= 1.0 + 1e-6,
            "hidden state[{i}] outside tanh range after {seq_len} steps: {h}"
        );
    }
}

#[test]
fn test_lstm_trace_records_3_inputs() {
    // Regression test for #2189: LSTM trace must record x, h, c as 3 inputs.
    use crate::dyn_tensor::trace::{trace_graph, TraceOp};

    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    let mut input = DynTensor::full(&[1, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let ((output, _state), graph) = trace_graph(|| {
        // Register input as a graph input node so it gets a trace ID.
        if let Some(id) = trace::record_input(input.dims(), input.dtype()) {
            input.set_trace_id(id);
        }
        lstm.forward(&input, None)
    })
    .unwrap();

    // Graph should have: 1 input (x) + 2 inputs (h, c from LSTM) + 1 LSTM node = 4
    assert!(
        graph.nodes().len() >= 4,
        "expected at least 4 nodes (x, h, c, lstm), got {}",
        graph.nodes().len()
    );

    // Find the LSTM node and verify it has 3 inputs.
    let lstm_node = graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Lstm { .. }))
        .expect("graph should contain an Lstm node");
    assert_eq!(
        lstm_node.inputs().len(),
        3,
        "LSTM node must have 3 inputs (x, h, c), got {}",
        lstm_node.inputs().len()
    );

    // Output should have a trace ID.
    assert!(output.trace_id().is_some(), "output should have a trace ID");
}

#[test]
fn test_lstm_trace_with_initial_state_records_3_inputs() {
    // When caller provides initial state, h and c still appear as 3 inputs.
    use crate::dyn_tensor::trace::{trace_graph, TraceOp};

    let h = 2;
    let input_size = 3;
    let w_ih = DynTensor::full(&[4 * h, input_size], 0.1, DType::F32, &Device::Cpu).unwrap();
    let w_hh = DynTensor::full(&[4 * h, h], 0.1, DType::F32, &Device::Cpu).unwrap();
    let lstm = Lstm::new(w_ih, w_hh, None, None, h).unwrap();

    let mut input = DynTensor::full(&[1, input_size], 1.0, DType::F32, &Device::Cpu).unwrap();
    let init_state = LstmState::new(
        DynTensor::full(&[1, h], 0.5, DType::F32, &Device::Cpu).unwrap(),
        DynTensor::full(&[1, h], 0.3, DType::F32, &Device::Cpu).unwrap(),
    )
    .unwrap();

    let (_, graph) = trace_graph(|| {
        if let Some(id) = trace::record_input(input.dims(), input.dtype()) {
            input.set_trace_id(id);
        }
        let (output, state) = lstm.forward(&input, Some(&init_state))?;
        Ok((output, state))
    })
    .unwrap();

    let lstm_node = graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Lstm { .. }))
        .expect("graph should contain an Lstm node");
    assert_eq!(
        lstm_node.inputs().len(),
        3,
        "LSTM node with initial state must have 3 inputs (x, h, c), got {}",
        lstm_node.inputs().len()
    );
}

#[path = "lstm_tests_validation.rs"]
mod validation;
