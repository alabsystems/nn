// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Production-scale GPU LSTM parity test for #2083.
//!
//! Verifies that the fused GPU LSTM sequence kernel produces output matching
//! the CPU per-timestep path at Kokoro production dimensions:
//! input_size=640, hidden_size=256, seq_len=20, batch=1.
//!
//! Root cause of #2083: before commit 75d36ae7, the LSTM sequence kernel's
//! encode macro used `set_buffer` without byte_offset. When weights are loaded
//! from mmapped safetensors (narrow views with byte_offset > 0), the kernel
//! read from byte 0 of the backing buffer instead of the actual weight data,
//! producing garbage output. The fix wired `set_buffer_with_offset` for all
//! 6 input buffers in the encode_lstm macro.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Lstm, LstmState};
use nn_core::Device;

/// Build an LSTM with PyTorch-scale Xavier uniform weights.
/// PyTorch LSTM default init: U(-1/sqrt(H), 1/sqrt(H)).
fn build_lstm_pytorch_scale(input_size: usize, hidden_size: usize, device: &Device) -> Lstm {
    let four_h = 4 * hidden_size;
    let bound = 1.0 / (hidden_size as f32).sqrt();

    let w_ih_data: Vec<f32> = (0..four_h * input_size)
        .map(|i| {
            let x = ((i as f32) * 0.618_034).fract();
            (x * 2.0 - 1.0) * bound
        })
        .collect();
    let w_hh_data: Vec<f32> = (0..four_h * hidden_size)
        .map(|i| {
            let x = ((i as f32) * 0.618_034 + 0.3).fract();
            (x * 2.0 - 1.0) * bound
        })
        .collect();

    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], device).unwrap();
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden_size], device).unwrap();

    let bias_data: Vec<f32> = vec![0.0; four_h];
    let b_ih = DynTensor::new(&bias_data, &[four_h], device).unwrap();
    let b_hh = DynTensor::new(&bias_data, &[four_h], device).unwrap();

    Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden_size).unwrap()
}

/// Build an LSTM with all-positive weights that maximize accumulation magnitude.
/// Tests worst-case FP32 accumulation: no sign cancellation over 640 products.
fn build_lstm_large_accum(input_size: usize, hidden_size: usize, device: &Device) -> Lstm {
    let four_h = 4 * hidden_size;
    let bound = 1.0 / (hidden_size as f32).sqrt();

    let w_ih_data: Vec<f32> = (0..four_h * input_size)
        .map(|i| {
            let x = ((i as f32) * 0.618_034).fract();
            x.abs() * bound
        })
        .collect();
    let w_hh_data: Vec<f32> = (0..four_h * hidden_size)
        .map(|i| {
            let x = ((i as f32) * 0.618_034 + 0.3).fract();
            x.abs() * bound
        })
        .collect();

    let w_ih = DynTensor::new(&w_ih_data, &[four_h, input_size], device).unwrap();
    let w_hh = DynTensor::new(&w_hh_data, &[four_h, hidden_size], device).unwrap();

    let mut bias_data: Vec<f32> = vec![0.0; four_h];
    // Set forget-gate bias to 1.0 (common in production).
    bias_data[hidden_size..2 * hidden_size].fill(1.0);
    let b_ih = DynTensor::new(&bias_data, &[four_h], device).unwrap();
    let b_hh = DynTensor::new(&vec![0.0f32; four_h], &[four_h], device).unwrap();

    Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden_size).unwrap()
}

fn run_parity_test(
    build_fn: fn(usize, usize, &Device) -> Lstm,
    input_size: usize,
    hidden_size: usize,
    seq_len: usize,
    tol: f32,
    label: &str,
) {
    gpu_init();
    let batch = 1;

    let cpu_lstm = build_fn(input_size, hidden_size, &Device::Cpu);
    let gpu_lstm = build_fn(input_size, hidden_size, &Device::metal());

    let data: Vec<f32> = (0..seq_len * batch * input_size)
        .map(|i| ((i as f32) * 0.1 + 0.5).sin() * 0.5)
        .collect();
    let cpu_input = DynTensor::new(&data, &[seq_len, batch, input_size], &Device::Cpu).unwrap();
    let gpu_input = DynTensor::new(&data, &[seq_len, batch, input_size], &Device::metal()).unwrap();

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

    let cpu_state = LstmState::new(cpu_h, cpu_c).unwrap();
    let gpu_state = LstmState::new(gpu_h, gpu_c).unwrap();

    let (cpu_out, cpu_final) = cpu_lstm.forward_seq(&cpu_input, Some(&cpu_state)).unwrap();
    let (gpu_out, gpu_final) = gpu_lstm.forward_seq(&gpu_input, Some(&gpu_state)).unwrap();

    let gpu_out_cpu = gpu_out.to_device(&Device::Cpu).unwrap();
    let gpu_h_cpu = gpu_final.h.to_device(&Device::Cpu).unwrap();

    assert_gpu_cpu_close(&gpu_out_cpu, &cpu_out, tol, &format!("{label}_output"));
    assert_gpu_cpu_close(&gpu_h_cpu, &cpu_final.h, tol, &format!("{label}_h_n"));
}

/// Compare fused GPU sequence kernel vs per-timestep GPU cell dispatch.
/// Verifies the fused MSL kernel matches the decomposed cell dispatch graph
/// when both run on GPU with the same weights.
fn run_fused_vs_pertimestep_test(
    build_fn: fn(usize, usize, &Device) -> Lstm,
    input_size: usize,
    hidden_size: usize,
    seq_len: usize,
    tol: f32,
    label: &str,
) {
    gpu_init();
    let batch = 1;
    let device = Device::metal();

    let lstm = build_fn(input_size, hidden_size, &device);

    let data: Vec<f32> = (0..seq_len * batch * input_size)
        .map(|i| ((i as f32) * 0.1 + 0.5).sin() * 0.5)
        .collect();
    let input = DynTensor::new(&data, &[seq_len, batch, input_size], &device).unwrap();

    let h_data: Vec<f32> = (0..batch * hidden_size)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let c_data: Vec<f32> = (0..batch * hidden_size)
        .map(|i| (i as f32) * -0.005)
        .collect();
    let h0 = DynTensor::new(&h_data, &[batch, hidden_size], &device).unwrap();
    let c0 = DynTensor::new(&c_data, &[batch, hidden_size], &device).unwrap();
    let state = LstmState::new(h0, c0).unwrap();

    // Fused sequence path (single dispatch).
    let (fused_out, fused_final) = lstm.forward_seq(&input, Some(&state)).unwrap();

    // Per-timestep path (cell dispatch per timestep).
    let mut outputs = Vec::with_capacity(seq_len);
    let mut current_state = Some(state);
    for t in 0..seq_len {
        let x_t = input.narrow(0, t, 1).unwrap().squeeze(0).unwrap();
        let (h_out, new_state) = lstm.forward(&x_t, current_state.as_ref()).unwrap();
        outputs.push(h_out.unsqueeze(0).unwrap());
        current_state = Some(new_state);
    }
    let out_refs: Vec<&DynTensor> = outputs.iter().collect();
    let pt_out = DynTensor::cat(&out_refs, 0).unwrap();
    let pt_final = current_state.unwrap();

    let fused_cpu = fused_out.to_device(&Device::Cpu).unwrap();
    let pt_cpu = pt_out.to_device(&Device::Cpu).unwrap();
    let fused_h_cpu = fused_final.h.to_device(&Device::Cpu).unwrap();
    let pt_h_cpu = pt_final.h.to_device(&Device::Cpu).unwrap();

    assert_gpu_cpu_close(
        &fused_cpu,
        &pt_cpu,
        tol,
        &format!("{label}_fused_vs_pt_output"),
    );
    assert_gpu_cpu_close(
        &fused_h_cpu,
        &pt_h_cpu,
        tol,
        &format!("{label}_fused_vs_pt_h_n"),
    );
}

// -- GPU vs CPU parity at production dimensions --

#[test]
fn test_lstm_production_640_pytorch_scale() {
    run_parity_test(build_lstm_pytorch_scale, 640, 256, 20, 1e-2, "pytorch_640");
}

#[test]
fn test_lstm_production_640_large_accum() {
    run_parity_test(
        build_lstm_large_accum,
        640,
        256,
        20,
        1e-2,
        "large_accum_640",
    );
}

// -- Fused kernel vs per-timestep cell dispatch at production dimensions --

#[test]
fn test_lstm_fused_vs_pertimestep_640_pytorch() {
    run_fused_vs_pertimestep_test(
        build_lstm_pytorch_scale,
        640,
        256,
        20,
        1e-2,
        "fvp_pytorch_640",
    );
}

#[test]
fn test_lstm_fused_vs_pertimestep_640_large_accum() {
    run_fused_vs_pertimestep_test(build_lstm_large_accum, 640, 256, 20, 1e-2, "fvp_large_640");
}
