// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM GPU forward tests.
//!
//! Extracted from `nn_gpu_forward.rs` for file size compliance.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::Lstm;
use nn_core::Device;

const TOL: f32 = 1e-4;

fn init() {
    gpu_init();
}

fn assert_close(gpu_result: &DynTensor, cpu_result: &DynTensor, label: &str) {
    assert_gpu_cpu_close(gpu_result, cpu_result, TOL, label);
}

// -- LSTM GPU forward (#1287) -------------------------------------------------

#[test]
fn test_lstm_gpu() {
    init();
    let input_size = 3;
    let hidden_size = 2;
    let four_h = 4 * hidden_size;

    // Deterministic weights: small values
    let w_ih_data: Vec<f32> = (0..four_h * input_size)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let w_hh_data: Vec<f32> = (0..four_h * hidden_size)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let b_ih_data: Vec<f32> = vec![0.1; four_h];
    let b_hh_data: Vec<f32> = vec![0.0; four_h];
    let x_data = vec![0.5, -0.3, 0.8];

    // CPU reference
    let w_ih_cpu = DynTensor::new(&w_ih_data, &[four_h, input_size], &Device::Cpu).unwrap();
    let w_hh_cpu = DynTensor::new(&w_hh_data, &[four_h, hidden_size], &Device::Cpu).unwrap();
    let b_ih_cpu = DynTensor::new(&b_ih_data, &[four_h], &Device::Cpu).unwrap();
    let b_hh_cpu = DynTensor::new(&b_hh_data, &[four_h], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[1, input_size], &Device::Cpu).unwrap();
    let lstm_cpu = Lstm::new(
        w_ih_cpu,
        w_hh_cpu,
        Some(b_ih_cpu),
        Some(b_hh_cpu),
        hidden_size,
    )
    .unwrap();
    let (h_cpu, state_cpu) = lstm_cpu.forward(&x_cpu, None).unwrap();

    // GPU
    let w_ih_gpu = DynTensor::new(&w_ih_data, &[four_h, input_size], &Device::metal()).unwrap();
    let w_hh_gpu = DynTensor::new(&w_hh_data, &[four_h, hidden_size], &Device::metal()).unwrap();
    let b_ih_gpu = DynTensor::new(&b_ih_data, &[four_h], &Device::metal()).unwrap();
    let b_hh_gpu = DynTensor::new(&b_hh_data, &[four_h], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[1, input_size], &Device::metal()).unwrap();
    let lstm_gpu = Lstm::new(
        w_ih_gpu,
        w_hh_gpu,
        Some(b_ih_gpu),
        Some(b_hh_gpu),
        hidden_size,
    )
    .unwrap();
    let (h_gpu, state_gpu) = lstm_gpu.forward(&x_gpu, None).unwrap();

    assert_eq!(h_gpu.dims(), &[1, hidden_size]);
    assert_close(&h_gpu, &h_cpu, "lstm_h");
    assert_close(&state_gpu.h, &state_cpu.h, "lstm_state_h");
    assert_close(&state_gpu.c, &state_cpu.c, "lstm_state_c");
}

#[test]
fn test_lstm_with_state_gpu() {
    init();
    let input_size = 2;
    let hidden_size = 2;
    let four_h = 4 * hidden_size;

    let w_ih_data: Vec<f32> = (0..four_h * input_size)
        .map(|i| (i as f32) * 0.02)
        .collect();
    let w_hh_data: Vec<f32> = (0..four_h * hidden_size)
        .map(|i| (i as f32) * 0.02)
        .collect();
    let x_data = vec![1.0, -1.0];

    // CPU: step 1 then step 2
    let w_ih_cpu = DynTensor::new(&w_ih_data, &[four_h, input_size], &Device::Cpu).unwrap();
    let w_hh_cpu = DynTensor::new(&w_hh_data, &[four_h, hidden_size], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[1, input_size], &Device::Cpu).unwrap();
    let lstm_cpu = Lstm::new(w_ih_cpu, w_hh_cpu, None, None, hidden_size).unwrap();
    let (_, state1_cpu) = lstm_cpu.forward(&x_cpu, None).unwrap();
    let (h2_cpu, _) = lstm_cpu.forward(&x_cpu, Some(&state1_cpu)).unwrap();

    // GPU: same two steps
    let w_ih_gpu = DynTensor::new(&w_ih_data, &[four_h, input_size], &Device::metal()).unwrap();
    let w_hh_gpu = DynTensor::new(&w_hh_data, &[four_h, hidden_size], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[1, input_size], &Device::metal()).unwrap();
    let lstm_gpu = Lstm::new(w_ih_gpu, w_hh_gpu, None, None, hidden_size).unwrap();
    let (_, state1_gpu) = lstm_gpu.forward(&x_gpu, None).unwrap();
    let (h2_gpu, _) = lstm_gpu.forward(&x_gpu, Some(&state1_gpu)).unwrap();

    assert_eq!(h2_gpu.dims(), &[1, hidden_size]);
    assert_close(&h2_gpu, &h2_cpu, "lstm_with_state_h2");
}
