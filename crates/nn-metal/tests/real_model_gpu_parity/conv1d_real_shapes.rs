// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d parity tests at real model dimensions.
//!
//! Tests Conv1d with shapes drawn from production audio models:
//! - Silero VAD encoder: Conv1d(129->128, k=3), Conv1d(128->64, k=3, s=2)
//! - Kokoro encoder: Conv1d(1->48, k=8, s=4), Conv1d(48->96, k=8, s=4)
//! - HTDemucs: Conv1d(48->96, k=8, s=4) with long sequences
//!
//! These shapes exercise the GPU conv dispatch path at production dimensions,
//! catching precision and correctness issues that hide at small test sizes.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv1d, Conv1dConfig, Module};
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

/// Tolerance for conv1d at production dimensions.
const TOL: f32 = 1e-4;

/// Helper: run Conv1d on both CPU and GPU with the same data and compare.
fn run_conv1d_parity(
    seed: u64,
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    length: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    label: &str,
) {
    let x_data = rand_f32_vec(seed, batch * in_ch * length, -1.0, 1.0);
    let w_data = rand_f32_vec(seed + 1, out_ch * in_ch * kernel, -0.3, 0.3);
    let b_data = rand_f32_vec(seed + 2, out_ch, -0.1, 0.1);

    let config = Conv1dConfig::new(padding, stride, dilation);

    // CPU
    let w_cpu = DynTensor::new(&w_data, &[out_ch, in_ch, kernel], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[out_ch], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[batch, in_ch, length], &Device::Cpu).unwrap();
    let conv_cpu = Conv1d::new(w_cpu, Some(b_cpu), config).unwrap();
    let y_cpu = conv_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[out_ch, in_ch, kernel], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[out_ch], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[batch, in_ch, length], &Device::metal()).unwrap();
    let conv_gpu = Conv1d::new(w_gpu, Some(b_gpu), config).unwrap();
    let y_gpu = conv_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.dims(), y_cpu.dims(), "{label}: shape mismatch");
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, label);
}

// -- Silero VAD encoder shapes -----------------------------------------------

/// Silero VAD encoder block 0: Conv1d(129, 128, kernel=3, stride=1, padding=1).
/// Input: [1, 129, 4] (STFT magnitude output).
#[test]
fn test_conv1d_silero_encoder_0() {
    gpu_init();
    run_conv1d_parity(
        3000,
        1,   // batch
        129, // in_channels (STFT bins)
        128, // out_channels
        3,   // kernel_size
        4,   // length (STFT time frames)
        1,   // padding
        1,   // stride
        1,   // dilation
        "silero_enc0_129to128_k3",
    );
}

/// Silero VAD encoder block 1: Conv1d(128, 64, kernel=3, stride=2, padding=1).
/// Input: [1, 128, 4].
#[test]
fn test_conv1d_silero_encoder_1() {
    gpu_init();
    run_conv1d_parity(3001, 1, 128, 64, 3, 4, 1, 2, 1, "silero_enc1_128to64_k3s2");
}

/// Silero VAD encoder block 2: Conv1d(64, 64, kernel=3, stride=2, padding=1).
/// Input: [1, 64, 2].
#[test]
fn test_conv1d_silero_encoder_2() {
    gpu_init();
    run_conv1d_parity(3002, 1, 64, 64, 3, 2, 1, 2, 1, "silero_enc2_64to64_k3s2");
}

/// Silero VAD encoder block 3: Conv1d(64, 128, kernel=3, stride=1, padding=1).
/// Input: [1, 64, 1].
#[test]
fn test_conv1d_silero_encoder_3() {
    gpu_init();
    run_conv1d_parity(3003, 1, 64, 128, 3, 1, 1, 1, 1, "silero_enc3_64to128_k3");
}

// -- Kokoro encoder shapes ---------------------------------------------------

/// Kokoro first encoder stage: Conv1d(1, 48, kernel=8, stride=4).
/// Input: [1, 1, 24000] (1 second of 24kHz audio).
#[test]
fn test_conv1d_kokoro_encoder_stage1() {
    gpu_init();
    // Use shorter input for test speed while maintaining real channel/kernel dims.
    run_conv1d_parity(
        3010,
        1,    // batch
        1,    // in_channels (mono audio)
        48,   // out_channels
        8,    // kernel_size
        4000, // length (shorter than production for speed)
        0,    // padding
        4,    // stride
        1,    // dilation
        "kokoro_enc1_1to48_k8s4",
    );
}

/// Kokoro second encoder stage: Conv1d(48, 96, kernel=8, stride=4).
/// Input: [1, 48, 1000] (output of first stage).
#[test]
fn test_conv1d_kokoro_encoder_stage2() {
    gpu_init();
    run_conv1d_parity(
        3011,
        1,    // batch
        48,   // in_channels
        96,   // out_channels
        8,    // kernel_size
        1000, // length
        0,    // padding
        4,    // stride
        1,    // dilation
        "kokoro_enc2_48to96_k8s4",
    );
}

/// Kokoro third encoder stage: Conv1d(96, 192, kernel=8, stride=4).
/// Input: [1, 96, 250].
#[test]
fn test_conv1d_kokoro_encoder_stage3() {
    gpu_init();
    run_conv1d_parity(
        3012,
        1,
        96,
        192,
        8,
        250,
        0,
        4,
        1,
        "kokoro_enc3_96to192_k8s4",
    );
}

// -- HTDemucs shapes ---------------------------------------------------------

/// HTDemucs temporal encoder: Conv1d(48, 96, kernel=8, stride=4) with long seq.
/// Input: [1, 48, 16000] (long audio segment).
#[test]
fn test_conv1d_htdemucs_temporal_encoder() {
    gpu_init();
    run_conv1d_parity(
        3020,
        1,     // batch
        48,    // in_channels
        96,    // out_channels
        8,     // kernel_size
        16000, // length (production-scale)
        0,     // padding
        4,     // stride
        1,     // dilation
        "htdemucs_temporal_48to96_k8s4",
    );
}

/// HTDemucs with padding and dilation: Conv1d(64, 64, kernel=3, padding=2, dilation=2).
/// Common in dilated conv blocks for large receptive fields.
#[test]
fn test_conv1d_htdemucs_dilated() {
    gpu_init();
    run_conv1d_parity(
        3021,
        1,    // batch
        64,   // in_channels
        64,   // out_channels
        3,    // kernel_size
        1024, // length
        2,    // padding
        1,    // stride
        2,    // dilation
        "htdemucs_dilated_64to64_k3d2",
    );
}

// -- Batched conv (multi-sample inference) -----------------------------------

/// Batched Silero VAD encoder: batch=4, Conv1d(129, 128, k=3, p=1).
#[test]
fn test_conv1d_silero_batched() {
    gpu_init();
    run_conv1d_parity(
        3030,
        4,   // batch
        129, // in_channels
        128, // out_channels
        3,   // kernel
        4,   // length
        1,   // padding
        1,   // stride
        1,   // dilation
        "silero_enc0_batched_4",
    );
}
