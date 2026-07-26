#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Validation error-path and boundary-condition tests for fused GPU LSTM cell.
//!
//! Tests exercise GPU LSTM through the public `layers::Lstm` API since
//! `gpu_lstm_cell` is `pub(super)`.
//!
//! Gaps identified by P1 proof_coverage audit:
//! - hidden_size=1 boundary condition (narrow gate split edge case)
//! - 3D input rejection propagation through layers::Lstm
//! - Shape mismatch between hidden and input batch dimensions
//! - Larger batch (8) exercising GPU threadgroup scheduling
//!
//! These complement the 6 parity tests in `dyn_tensor_metal_lstm_tests.rs`.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Lstm, LstmState};
use nn_core::{Device, TensorError};

use crate::test_common::{assert_close, init};

/// Build a GPU LSTM with known small weights.
fn build_gpu_lstm(input_size: usize, hidden_size: usize, with_bias: bool) -> (Lstm, Lstm) {
    let four_h = 4 * hidden_size;
    let w_ih_data: Vec<f32> = (0..four_h * input_size)
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();
    let w_hh_data: Vec<f32> = (0..four_h * hidden_size)
        .map(|i| (i as f32 + 1.0) * 0.01)
        .collect();

    let cpu_w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], &Device::Cpu).unwrap();
    let cpu_w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden_size], &Device::Cpu).unwrap();
    let gpu_w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], &Device::metal()).unwrap();
    let gpu_w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden_size], &Device::metal()).unwrap();

    let (cpu_b_ih, cpu_b_hh, gpu_b_ih, gpu_b_hh) = if with_bias {
        let b_data: Vec<f32> = vec![0.01; four_h];
        (
            Some(DynTensor::new(&b_data, &[four_h], &Device::Cpu).unwrap()),
            Some(DynTensor::new(&b_data, &[four_h], &Device::Cpu).unwrap()),
            Some(DynTensor::new(&b_data, &[four_h], &Device::metal()).unwrap()),
            Some(DynTensor::new(&b_data, &[four_h], &Device::metal()).unwrap()),
        )
    } else {
        (None, None, None, None)
    };

    let cpu_lstm = Lstm::new(cpu_w_ih, cpu_w_hh, cpu_b_ih, cpu_b_hh, hidden_size).unwrap();
    let gpu_lstm = Lstm::new(gpu_w_ih, gpu_w_hh, gpu_b_ih, gpu_b_hh, hidden_size).unwrap();
    (cpu_lstm, gpu_lstm)
}

/// GPU LSTM with hidden_size=1 boundary condition.
/// Exercises narrow gate split with minimal dimension — the 4 gates each
/// have width 1, testing that the narrow(0,1), narrow(1,1), narrow(2,1),
/// narrow(3,1) splits work at the minimum size.
#[test]
fn test_gpu_lstm_cell_hidden_size_1() {
    init();
    let batch = 2;
    let input_size = 3;
    let hidden_size = 1;

    let (cpu_lstm, gpu_lstm) = build_gpu_lstm(input_size, hidden_size, false);

    let input_data: Vec<f32> = (0..batch * input_size)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();

    let cpu_input = DynTensor::new(&input_data, &[batch, input_size], &Device::Cpu).unwrap();
    let gpu_input = DynTensor::new(&input_data, &[batch, input_size], &Device::metal()).unwrap();

    let (cpu_out, cpu_st) = cpu_lstm.forward(&cpu_input, None).unwrap();
    let (gpu_out, gpu_st) = gpu_lstm.forward(&gpu_input, None).unwrap();

    let cpu_h_vals = cpu_st.h.to_flat_vec::<f32>().unwrap();
    let gpu_h_vals = gpu_st
        .h
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_h_vals, &cpu_h_vals, 1e-4, "lstm_h1_hidden");

    let cpu_c_vals = cpu_st.c.to_flat_vec::<f32>().unwrap();
    let gpu_c_vals = gpu_st
        .c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_c_vals, &cpu_c_vals, 1e-4, "lstm_h1_cell");

    // Also check output (should be same as h_new)
    let cpu_out_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    let gpu_out_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_out_vals, &cpu_out_vals, 1e-4, "lstm_h1_output");
}

/// 3D input to layers::Lstm should propagate an error.
#[test]
fn test_gpu_lstm_rejects_3d_input() {
    init();
    let input_size = 4;
    let hidden_size = 3;

    let (_, gpu_lstm) = build_gpu_lstm(input_size, hidden_size, false);

    // 3D input — should fail
    let input_3d = DynTensor::new(
        &vec![1.0; 2 * input_size],
        &[2, 1, input_size],
        &Device::metal(),
    )
    .unwrap();

    let result = gpu_lstm.forward(&input_3d, None);
    assert!(result.is_err(), "Expected error for 3D input to GPU LSTM");
}

/// Hidden state with wrong batch size should return ShapeMismatch.
#[test]
fn test_gpu_lstm_rejects_hidden_batch_mismatch() {
    init();
    let batch = 2;
    let hidden_size = 3;

    // Hidden with wrong batch (1 instead of 2)
    let hidden_wrong =
        DynTensor::new(&vec![0.0; hidden_size], &[1, hidden_size], &Device::metal()).unwrap();
    let cell = DynTensor::new(
        &vec![0.0; batch * hidden_size],
        &[batch, hidden_size],
        &Device::metal(),
    )
    .unwrap();

    let result = LstmState::new(hidden_wrong, cell);
    assert!(
        result.is_err(),
        "Expected ShapeMismatch for hidden batch mismatch at construction"
    );
}

/// GPU LSTM with NaN in w_ih is rejected at construction (#2064).
///
/// `Lstm::new()` validates weight finiteness via `validate_weight_finite()`.
/// NaN in w_ih is detected and returns `NonFiniteData` before any forward call.
#[test]
fn test_gpu_lstm_nan_weights_rejected_at_construction() {
    init();
    let hidden_size = 4;
    let input_size = 3;
    let four_h = 4 * hidden_size;

    let mut w_ih_data: Vec<f32> = vec![0.01; four_h * input_size];
    w_ih_data[5] = f32::NAN;

    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], &Device::metal()).unwrap();
    let w_hh_data: Vec<f32> = vec![0.01; four_h * hidden_size];
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden_size], &Device::metal()).unwrap();

    let result = Lstm::new(w_ih, w_hh, None, None, hidden_size);
    assert!(
        matches!(result, Err(TensorError::NonFiniteData { count: 1, .. })),
        "Expected NonFiniteData with count=1, got: {result:?}"
    );
}

/// GPU LSTM with Inf in w_ih is rejected at construction (#2064).
///
/// `Lstm::new()` validates weight finiteness. Inf in w_ih returns
/// `NonFiniteData` before any forward call.
#[test]
fn test_gpu_lstm_inf_wih_rejected_at_construction() {
    init();
    let hidden_size = 4;
    let input_size = 3;
    let four_h = 4 * hidden_size;

    let mut w_ih_data: Vec<f32> = vec![0.01; four_h * input_size];
    w_ih_data[0] = f32::INFINITY;

    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], &Device::metal()).unwrap();
    let w_hh_data: Vec<f32> = vec![0.01; four_h * hidden_size];
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden_size], &Device::metal()).unwrap();

    let result = Lstm::new(w_ih, w_hh, None, None, hidden_size);
    assert!(
        matches!(result, Err(TensorError::NonFiniteData { count: 1, .. })),
        "Expected NonFiniteData with count=1, got: {result:?}"
    );
}

/// GPU LSTM with -Inf in w_hh is rejected at construction (#2064).
#[test]
fn test_gpu_lstm_inf_whh_rejected_at_construction() {
    init();
    let hidden_size = 4;
    let input_size = 3;
    let four_h = 4 * hidden_size;

    let w_ih_data: Vec<f32> = vec![0.01; four_h * input_size];
    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], &Device::metal()).unwrap();
    let mut w_hh_data: Vec<f32> = vec![0.01; four_h * hidden_size];
    w_hh_data[3] = f32::NEG_INFINITY;

    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden_size], &Device::metal()).unwrap();

    let result = Lstm::new(w_ih, w_hh, None, None, hidden_size);
    assert!(
        matches!(result, Err(TensorError::NonFiniteData { count: 1, .. })),
        "Expected NonFiniteData with count=1, got: {result:?}"
    );
}

/// GPU LSTM with Inf in b_ih is rejected at construction (#2064).
///
/// `Lstm::new()` validates bias finiteness via `validate_weight_finite()`.
/// Inf in b_ih is detected and returns `NonFiniteData` before any forward call.
#[test]
fn test_gpu_lstm_inf_bias_rejected_at_construction() {
    init();
    let hidden_size = 4;
    let input_size = 3;
    let four_h = 4 * hidden_size;

    let w_ih_data: Vec<f32> = vec![0.01; four_h * input_size];
    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], &Device::metal()).unwrap();
    let w_hh_data: Vec<f32> = vec![0.01; four_h * hidden_size];
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden_size], &Device::metal()).unwrap();

    let mut b_ih_data: Vec<f32> = vec![0.01; four_h];
    b_ih_data[0] = f32::INFINITY;
    let b_ih = DynTensor::new(&b_ih_data, &[four_h], &Device::metal()).unwrap();
    let b_hh = DynTensor::new(&vec![0.01; four_h], &[four_h], &Device::metal()).unwrap();

    let result = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden_size);
    assert!(
        matches!(result, Err(TensorError::NonFiniteData { count: 1, .. })),
        "Expected NonFiniteData with count=1, got: {result:?}"
    );
}

/// GPU LSTM sequence with hidden_size > 512 should fall back to per-timestep
/// loop (cell kernel path) because the sequence kernel exceeds threadgroup
/// memory limits. This tests the fallback boundary.
///
/// The per-timestep GPU loop internally uses `NanCheckPolicy::Skip` to prevent
/// intermediate `flush()` calls from advancing the arena generation past the
/// stale-read threshold (#2328). NaN/Inf validation defers to model-boundary
/// checks (#941, #958).
#[test]
fn test_gpu_lstm_seq_hidden_513_fallback() {
    init();
    let hidden_size = 513; // just over the 512 boundary
    let input_size = 8;
    let seq_len = 3;
    let batch = 1;

    let cpu_lstm = build_gpu_lstm_for_seq(input_size, hidden_size, &Device::Cpu);
    let gpu_lstm = build_gpu_lstm_for_seq(input_size, hidden_size, &Device::metal());

    let input_data: Vec<f32> = (0..seq_len * batch * input_size)
        .map(|i| ((i as f32) * 0.1 + 0.5).sin() * 0.1)
        .collect();
    let cpu_input =
        DynTensor::new(&input_data, &[seq_len, batch, input_size], &Device::Cpu).unwrap();
    let gpu_input =
        DynTensor::new(&input_data, &[seq_len, batch, input_size], &Device::metal()).unwrap();

    let (cpu_out, cpu_st) = cpu_lstm.forward_seq(&cpu_input, None).unwrap();
    // No NanCheckPolicy::Skip wrapper needed — forward_seq now handles it internally.
    let (gpu_out, gpu_st) = gpu_lstm.forward_seq(&gpu_input, None).unwrap();

    let cpu_h = cpu_st.h.to_flat_vec::<f32>().unwrap();
    let gpu_h = gpu_st
        .h
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_h, &cpu_h, 1e-3, "h513_fallback_h");

    let cpu_c = cpu_st.c.to_flat_vec::<f32>().unwrap();
    let gpu_c = gpu_st
        .c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_c, &cpu_c, 1e-3, "h513_fallback_c");

    let cpu_out_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    let gpu_out_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_out_vals, &cpu_out_vals, 1e-3, "h513_fallback_output");
}

/// Helper for sequence tests: build LSTM on given device.
fn build_gpu_lstm_for_seq(input_size: usize, hidden_size: usize, device: &Device) -> Lstm {
    let four_h = 4 * hidden_size;
    let w_ih_data: Vec<f32> = (0..four_h * input_size)
        .map(|i| ((i as f32) * 0.01 - 0.05).sin() * 0.05)
        .collect();
    let w_hh_data: Vec<f32> = (0..four_h * hidden_size)
        .map(|i| ((i as f32) * 0.02 + 0.1).cos() * 0.05)
        .collect();
    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], device).unwrap();
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden_size], device).unwrap();
    Lstm::new(w_ih, w_hh, None, None, hidden_size).unwrap()
}

/// GPU LSTM with larger batch (8) to exercise threadgroup scheduling.
#[test]
fn test_gpu_lstm_cell_batch_8() {
    init();
    let batch = 8;
    let input_size = 16;
    let hidden_size = 8;

    let (cpu_lstm, gpu_lstm) = build_gpu_lstm(input_size, hidden_size, true);

    let input_data: Vec<f32> = (0..batch * input_size)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();

    let cpu_input = DynTensor::new(&input_data, &[batch, input_size], &Device::Cpu).unwrap();
    let gpu_input = DynTensor::new(&input_data, &[batch, input_size], &Device::metal()).unwrap();

    // With initial state
    let h_data: Vec<f32> = (0..batch * hidden_size)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let c_data: Vec<f32> = (0..batch * hidden_size)
        .map(|i| (i as f32) * -0.005)
        .collect();

    let cpu_state = LstmState::new(
        DynTensor::new(&h_data, &[batch, hidden_size], &Device::Cpu).unwrap(),
        DynTensor::new(&c_data, &[batch, hidden_size], &Device::Cpu).unwrap(),
    )
    .unwrap();
    let gpu_state = LstmState::new(
        DynTensor::new(&h_data, &[batch, hidden_size], &Device::metal()).unwrap(),
        DynTensor::new(&c_data, &[batch, hidden_size], &Device::metal()).unwrap(),
    )
    .unwrap();

    let (_, cpu_st) = cpu_lstm.forward(&cpu_input, Some(&cpu_state)).unwrap();
    let (_, gpu_st) = gpu_lstm.forward(&gpu_input, Some(&gpu_state)).unwrap();

    let cpu_h_vals = cpu_st.h.to_flat_vec::<f32>().unwrap();
    let gpu_h_vals = gpu_st
        .h
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_h_vals, &cpu_h_vals, 1e-4, "lstm_batch8_hidden");

    let cpu_c_vals = cpu_st.c.to_flat_vec::<f32>().unwrap();
    let gpu_c_vals = gpu_st
        .c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_c_vals, &cpu_c_vals, 1e-4, "lstm_batch8_cell");
}
