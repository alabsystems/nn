#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU LSTM cell parity tests — verifies fused GPU LSTM cell produces
//! identical output to the CPU decomposed path.
//!
//! Part of #1373 (fused GPU LSTM cell).

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::Lstm;
use nn_core::Device;

use crate::test_common::{assert_close, init};

/// Build an LSTM on the specified device with small known weights.
fn build_lstm(input_size: usize, hidden_size: usize, with_bias: bool, device: &Device) -> Lstm {
    let four_h = 4 * hidden_size;
    // Use small but non-uniform weights for numerical variety.
    let w_ih_data: Vec<f32> = (0..four_h * input_size)
        .map(|i| ((i as f32) * 0.01 - 0.05).sin() * 0.1)
        .collect();
    let w_hh_data: Vec<f32> = (0..four_h * hidden_size)
        .map(|i| ((i as f32) * 0.02 + 0.1).cos() * 0.1)
        .collect();

    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], device).unwrap();
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden_size], device).unwrap();

    let (b_ih, b_hh) = if with_bias {
        let bih_data: Vec<f32> = (0..four_h).map(|i| (i as f32) * 0.001).collect();
        let bhh_data: Vec<f32> = (0..four_h).map(|i| (i as f32) * -0.001).collect();
        (
            Some(DynTensor::new(&bih_data, &[four_h], device).unwrap()),
            Some(DynTensor::new(&bhh_data, &[four_h], device).unwrap()),
        )
    } else {
        (None, None)
    };

    Lstm::new(w_ih, w_hh, b_ih, b_hh, hidden_size).unwrap()
}

/// Run LSTM forward on both CPU and GPU, compare results.
fn assert_lstm_gpu_cpu_parity(
    input_size: usize,
    hidden_size: usize,
    batch: usize,
    with_bias: bool,
    with_state: bool,
    tol: f32,
    label: &str,
) {
    init();

    let cpu_lstm = build_lstm(input_size, hidden_size, with_bias, &Device::Cpu);
    let gpu_lstm = build_lstm(input_size, hidden_size, with_bias, &Device::metal());

    // Input data.
    let input_data: Vec<f32> = (0..batch * input_size)
        .map(|i| ((i as f32) * 0.1 + 0.5).sin())
        .collect();
    let cpu_input = DynTensor::new(&input_data, &[batch, input_size], &Device::Cpu).unwrap();
    let gpu_input = DynTensor::new(&input_data, &[batch, input_size], &Device::metal()).unwrap();

    // Optional initial state.
    let (cpu_state, gpu_state) = if with_state {
        let h_data: Vec<f32> = (0..batch * hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let c_data: Vec<f32> = (0..batch * hidden_size)
            .map(|i| (i as f32) * -0.005)
            .collect();

        let cpu_h = DynTensor::new(&h_data, &[batch, hidden_size], &Device::Cpu).unwrap();
        let cpu_c = DynTensor::new(&c_data, &[batch, hidden_size], &Device::Cpu).unwrap();
        let gpu_h = DynTensor::new(&h_data, &[batch, hidden_size], &Device::metal()).unwrap();
        let gpu_c = DynTensor::new(&c_data, &[batch, hidden_size], &Device::metal()).unwrap();

        (
            Some(nn_core::layers::LstmState::new(cpu_h, cpu_c).unwrap()),
            Some(nn_core::layers::LstmState::new(gpu_h, gpu_c).unwrap()),
        )
    } else {
        (None, None)
    };

    // CPU forward.
    let (cpu_out, cpu_st) = cpu_lstm.forward(&cpu_input, cpu_state.as_ref()).unwrap();
    let cpu_out_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    let cpu_h_vals = cpu_st.h.to_flat_vec::<f32>().unwrap();
    let cpu_c_vals = cpu_st.c.to_flat_vec::<f32>().unwrap();

    // GPU forward (uses fused path).
    let (gpu_out, gpu_st) = gpu_lstm.forward(&gpu_input, gpu_state.as_ref()).unwrap();

    assert_eq!(
        gpu_out.device(),
        Device::metal(),
        "{label}: output should stay on GPU"
    );

    let gpu_out_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let gpu_h_vals = gpu_st
        .h
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let gpu_c_vals = gpu_st
        .c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_close(
        &gpu_out_vals,
        &cpu_out_vals,
        tol,
        &format!("{label} output"),
    );
    assert_close(&gpu_h_vals, &cpu_h_vals, tol, &format!("{label} h_new"));
    assert_close(&gpu_c_vals, &cpu_c_vals, tol, &format!("{label} c_new"));
}

// -- AC4: GPU vs CPU parity tests --

#[test]
fn test_fused_lstm_gpu_cpu_parity_no_bias_no_state() {
    assert_lstm_gpu_cpu_parity(4, 3, 1, false, false, 1e-5, "lstm_no_bias_no_state");
}

#[test]
fn test_fused_lstm_gpu_cpu_parity_with_bias() {
    assert_lstm_gpu_cpu_parity(4, 3, 2, true, false, 1e-5, "lstm_with_bias");
}

#[test]
fn test_fused_lstm_gpu_cpu_parity_with_state() {
    assert_lstm_gpu_cpu_parity(4, 3, 2, true, true, 1e-5, "lstm_with_state");
}

#[test]
fn test_fused_lstm_gpu_cpu_parity_batch1() {
    // Silero VAD typical: batch=1, small hidden size.
    assert_lstm_gpu_cpu_parity(64, 64, 1, true, true, 1e-4, "lstm_batch1_h64");
}

#[test]
fn test_fused_lstm_gpu_cpu_parity_larger() {
    // Larger hidden size typical of production models.
    assert_lstm_gpu_cpu_parity(32, 128, 2, true, true, 1e-4, "lstm_h128_b2");
}

// -- forward_seq GPU parity --

#[test]
fn test_fused_lstm_seq_gpu_cpu_parity() {
    init();

    let input_size = 4;
    let hidden_size = 3;
    let batch = 2;
    let seq_len = 3;
    let tol = 1e-5;

    let cpu_lstm = build_lstm(input_size, hidden_size, true, &Device::Cpu);
    let gpu_lstm = build_lstm(input_size, hidden_size, true, &Device::metal());

    let input_data: Vec<f32> = (0..seq_len * batch * input_size)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();

    let cpu_input =
        DynTensor::new(&input_data, &[seq_len, batch, input_size], &Device::Cpu).unwrap();
    let gpu_input =
        DynTensor::new(&input_data, &[seq_len, batch, input_size], &Device::metal()).unwrap();

    let (cpu_out, cpu_st) = cpu_lstm.forward_seq(&cpu_input, None).unwrap();
    let (gpu_out, gpu_st) = gpu_lstm.forward_seq(&gpu_input, None).unwrap();

    assert_eq!(cpu_out.dims(), gpu_out.dims(), "seq output shape mismatch");
    assert_eq!(
        gpu_out.device(),
        Device::metal(),
        "seq output should stay on GPU"
    );

    let cpu_vals = cpu_out.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, tol, "lstm_seq_output");

    let cpu_h = cpu_st.h.to_flat_vec::<f32>().unwrap();
    let gpu_h = gpu_st
        .h
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_h, &cpu_h, tol, "lstm_seq_h_final");
}
