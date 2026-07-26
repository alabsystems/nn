// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM parity tests at real model dimensions.
//!
//! Tests LSTM cell forward with shapes from production models:
//! - Silero VAD: input_size=128, hidden_size=128
//! - WeSpeaker: input_size=256, hidden_size=256
//! - Multi-step LSTM: verifies state propagation over multiple timesteps
//!
//! LSTM precision is critical: gate errors (sigmoid, tanh) compound through
//! recurrent timesteps, so single-step parity does not guarantee multi-step.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Lstm, LstmState};
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

/// Tolerance for single LSTM step.
const TOL: f32 = 1e-5;

/// Tolerance for multi-step LSTM (error compounds across steps).
const MULTI_STEP_TOL: f32 = 1e-4;

/// Helper: create LSTM weights on a device and run one forward step.
fn run_lstm_step(
    x_data: &[f32],
    w_ih_data: &[f32],
    w_hh_data: &[f32],
    b_ih_data: &[f32],
    b_hh_data: &[f32],
    batch: usize,
    input_size: usize,
    hidden_size: usize,
    device: &Device,
    initial_state: Option<(&[f32], &[f32])>,
) -> (DynTensor, DynTensor, DynTensor) {
    let x = DynTensor::new(x_data, &[batch, input_size], device).unwrap();
    let w_ih = DynTensor::new(w_ih_data, &[4 * hidden_size, input_size], device).unwrap();
    let w_hh = DynTensor::new(w_hh_data, &[4 * hidden_size, hidden_size], device).unwrap();
    let b_ih = DynTensor::new(b_ih_data, &[4 * hidden_size], device).unwrap();
    let b_hh = DynTensor::new(b_hh_data, &[4 * hidden_size], device).unwrap();

    let lstm = Lstm::new(w_ih, w_hh, Some(b_ih), Some(b_hh), hidden_size).unwrap();

    let state = initial_state.map(|(h_data, c_data)| {
        let h = DynTensor::new(h_data, &[batch, hidden_size], device).unwrap();
        let c = DynTensor::new(c_data, &[batch, hidden_size], device).unwrap();
        LstmState::new(h, c).unwrap()
    });

    let (output, new_state) = lstm.forward(&x, state.as_ref()).unwrap();
    (output, new_state.h, new_state.c)
}

// -- Silero VAD LSTM shape (input=128, hidden=128) ---------------------------

/// Silero VAD LSTM cell: single step, zero initial state.
/// input_size=128, hidden_size=128 — exact production dimensions.
#[test]
fn test_lstm_silero_vad_zero_state() {
    gpu_init();
    let batch = 1;
    let input_size = 128;
    let hidden_size = 128;

    let x_data = rand_f32_vec(6000, batch * input_size, -1.0, 1.0);
    let w_ih = rand_f32_vec(6001, 4 * hidden_size * input_size, -0.1, 0.1);
    let w_hh = rand_f32_vec(6002, 4 * hidden_size * hidden_size, -0.1, 0.1);
    let b_ih = rand_f32_vec(6003, 4 * hidden_size, -0.1, 0.1);
    let b_hh = rand_f32_vec(6004, 4 * hidden_size, -0.1, 0.1);

    let (cpu_out, cpu_h, cpu_c) = run_lstm_step(
        &x_data,
        &w_ih,
        &w_hh,
        &b_ih,
        &b_hh,
        batch,
        input_size,
        hidden_size,
        &Device::Cpu,
        None,
    );
    let (gpu_out, gpu_h, gpu_c) = run_lstm_step(
        &x_data,
        &w_ih,
        &w_hh,
        &b_ih,
        &b_hh,
        batch,
        input_size,
        hidden_size,
        &Device::metal(),
        None,
    );

    assert_eq!(cpu_out.dims(), &[batch, hidden_size]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "lstm_silero_output");
    assert_gpu_cpu_close(&gpu_h, &cpu_h, TOL, "lstm_silero_h");
    assert_gpu_cpu_close(&gpu_c, &cpu_c, TOL, "lstm_silero_c");
}

/// Silero VAD LSTM cell: single step with non-zero initial state.
#[test]
fn test_lstm_silero_vad_with_state() {
    gpu_init();
    let batch = 1;
    let input_size = 128;
    let hidden_size = 128;

    let x_data = rand_f32_vec(6010, batch * input_size, -1.0, 1.0);
    let w_ih = rand_f32_vec(6011, 4 * hidden_size * input_size, -0.1, 0.1);
    let w_hh = rand_f32_vec(6012, 4 * hidden_size * hidden_size, -0.1, 0.1);
    let b_ih = rand_f32_vec(6013, 4 * hidden_size, -0.1, 0.1);
    let b_hh = rand_f32_vec(6014, 4 * hidden_size, -0.1, 0.1);
    let h0_data = rand_f32_vec(6015, batch * hidden_size, -0.5, 0.5);
    let c0_data = rand_f32_vec(6016, batch * hidden_size, -0.5, 0.5);

    let (cpu_out, cpu_h, cpu_c) = run_lstm_step(
        &x_data,
        &w_ih,
        &w_hh,
        &b_ih,
        &b_hh,
        batch,
        input_size,
        hidden_size,
        &Device::Cpu,
        Some((&h0_data, &c0_data)),
    );
    let (gpu_out, gpu_h, gpu_c) = run_lstm_step(
        &x_data,
        &w_ih,
        &w_hh,
        &b_ih,
        &b_hh,
        batch,
        input_size,
        hidden_size,
        &Device::metal(),
        Some((&h0_data, &c0_data)),
    );

    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "lstm_silero_state_output");
    assert_gpu_cpu_close(&gpu_h, &cpu_h, TOL, "lstm_silero_state_h");
    assert_gpu_cpu_close(&gpu_c, &cpu_c, TOL, "lstm_silero_state_c");
}

/// Multi-step LSTM: 5 timesteps with state propagation.
/// Verifies that error does not compound excessively through recurrence.
#[test]
fn test_lstm_silero_multi_step() {
    gpu_init();
    let batch = 1;
    let input_size = 128;
    let hidden_size = 128;
    let num_steps = 5;

    let w_ih = rand_f32_vec(6020, 4 * hidden_size * input_size, -0.1, 0.1);
    let w_hh = rand_f32_vec(6021, 4 * hidden_size * hidden_size, -0.1, 0.1);
    let b_ih = rand_f32_vec(6022, 4 * hidden_size, -0.1, 0.1);
    let b_hh = rand_f32_vec(6023, 4 * hidden_size, -0.1, 0.1);

    // Build LSTM on each device.
    let build_lstm = |device: &Device| {
        let wi = DynTensor::new(&w_ih, &[4 * hidden_size, input_size], device).unwrap();
        let wh = DynTensor::new(&w_hh, &[4 * hidden_size, hidden_size], device).unwrap();
        let bi = DynTensor::new(&b_ih, &[4 * hidden_size], device).unwrap();
        let bh = DynTensor::new(&b_hh, &[4 * hidden_size], device).unwrap();
        Lstm::new(wi, wh, Some(bi), Some(bh), hidden_size).unwrap()
    };

    let cpu_lstm = build_lstm(&Device::Cpu);
    let gpu_lstm = build_lstm(&Device::metal());

    let mut cpu_state: Option<LstmState> = None;
    let mut gpu_state: Option<LstmState> = None;

    for step in 0..num_steps {
        let x_data = rand_f32_vec(6030 + step as u64, batch * input_size, -1.0, 1.0);

        let x_cpu = DynTensor::new(&x_data, &[batch, input_size], &Device::Cpu).unwrap();
        let x_gpu = DynTensor::new(&x_data, &[batch, input_size], &Device::metal()).unwrap();

        let (cpu_out, cpu_new) = cpu_lstm.forward(&x_cpu, cpu_state.as_ref()).unwrap();
        let (gpu_out, gpu_new) = gpu_lstm.forward(&x_gpu, gpu_state.as_ref()).unwrap();

        assert_gpu_cpu_close(
            &gpu_out,
            &cpu_out,
            MULTI_STEP_TOL,
            &format!("lstm_multi_step_{step}_out"),
        );
        assert_gpu_cpu_close(
            &gpu_new.h,
            &cpu_new.h,
            MULTI_STEP_TOL,
            &format!("lstm_multi_step_{step}_h"),
        );
        assert_gpu_cpu_close(
            &gpu_new.c,
            &cpu_new.c,
            MULTI_STEP_TOL,
            &format!("lstm_multi_step_{step}_c"),
        );

        cpu_state = Some(cpu_new);
        gpu_state = Some(gpu_new);
    }
}

// -- Larger LSTM (WeSpeaker-like) --------------------------------------------

/// WeSpeaker-like LSTM: input_size=256, hidden_size=256.
/// Larger than Silero VAD, tests LSTM dispatch at scale.
#[test]
fn test_lstm_wespeaker_shape() {
    gpu_init();
    let batch = 1;
    let input_size = 256;
    let hidden_size = 256;

    let x_data = rand_f32_vec(6040, batch * input_size, -1.0, 1.0);
    let w_ih = rand_f32_vec(6041, 4 * hidden_size * input_size, -0.05, 0.05);
    let w_hh = rand_f32_vec(6042, 4 * hidden_size * hidden_size, -0.05, 0.05);
    let b_ih = rand_f32_vec(6043, 4 * hidden_size, -0.1, 0.1);
    let b_hh = rand_f32_vec(6044, 4 * hidden_size, -0.1, 0.1);

    let (cpu_out, cpu_h, cpu_c) = run_lstm_step(
        &x_data,
        &w_ih,
        &w_hh,
        &b_ih,
        &b_hh,
        batch,
        input_size,
        hidden_size,
        &Device::Cpu,
        None,
    );
    let (gpu_out, gpu_h, gpu_c) = run_lstm_step(
        &x_data,
        &w_ih,
        &w_hh,
        &b_ih,
        &b_hh,
        batch,
        input_size,
        hidden_size,
        &Device::metal(),
        None,
    );

    assert_eq!(cpu_out.dims(), &[batch, hidden_size]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "lstm_wespeaker_output");
    assert_gpu_cpu_close(&gpu_h, &cpu_h, TOL, "lstm_wespeaker_h");
    assert_gpu_cpu_close(&gpu_c, &cpu_c, TOL, "lstm_wespeaker_c");
}

// -- Batched LSTM ------------------------------------------------------------

/// Batched LSTM: batch=4, input_size=128, hidden_size=128.
/// Tests GPU LSTM dispatch with multiple samples.
#[test]
fn test_lstm_batched() {
    gpu_init();
    let batch = 4;
    let input_size = 128;
    let hidden_size = 128;

    let x_data = rand_f32_vec(6050, batch * input_size, -1.0, 1.0);
    let w_ih = rand_f32_vec(6051, 4 * hidden_size * input_size, -0.1, 0.1);
    let w_hh = rand_f32_vec(6052, 4 * hidden_size * hidden_size, -0.1, 0.1);
    let b_ih = rand_f32_vec(6053, 4 * hidden_size, -0.1, 0.1);
    let b_hh = rand_f32_vec(6054, 4 * hidden_size, -0.1, 0.1);

    let (cpu_out, cpu_h, cpu_c) = run_lstm_step(
        &x_data,
        &w_ih,
        &w_hh,
        &b_ih,
        &b_hh,
        batch,
        input_size,
        hidden_size,
        &Device::Cpu,
        None,
    );
    let (gpu_out, gpu_h, gpu_c) = run_lstm_step(
        &x_data,
        &w_ih,
        &w_hh,
        &b_ih,
        &b_hh,
        batch,
        input_size,
        hidden_size,
        &Device::metal(),
        None,
    );

    assert_eq!(cpu_out.dims(), &[batch, hidden_size]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "lstm_batched_output");
    assert_gpu_cpu_close(&gpu_h, &cpu_h, TOL, "lstm_batched_h");
    assert_gpu_cpu_close(&gpu_c, &cpu_c, TOL, "lstm_batched_c");
}
