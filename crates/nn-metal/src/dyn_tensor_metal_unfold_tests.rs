// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU/CPU parity tests for `DynTensor::unfold()` (#1945).
//!
//! Verifies the Metal GPU unfold kernel produces identical output to the
//! CPU reference implementation across multiple tensor ranks, dimensions,
//! and step sizes — including the exact STFT framing pattern that replaces
//! 87K narrow() dispatches with a single GPU kernel.

#![allow(deprecated)]

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_close, init};

/// 1D unfold: [8].unfold(0, 4, 2) -> [3, 4]
#[test]
fn test_gpu_unfold_1d_basic() {
    init();
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[8], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = cpu.unfold(0, 4, 2).unwrap();
    let gpu_result = gpu.unfold(0, 4, 2).unwrap();

    assert_eq!(cpu_result.dims(), &[3, 4]);
    assert_eq!(gpu_result.dims(), &[3, 4]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "unfold_1d_basic");
}

/// 2D unfold along dim=1: [2, 8].unfold(1, 3, 2) -> [2, 3, 3]
#[test]
fn test_gpu_unfold_2d_dim1() {
    init();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[2, 8], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = cpu.unfold(1, 3, 2).unwrap();
    let gpu_result = gpu.unfold(1, 3, 2).unwrap();

    assert_eq!(cpu_result.dims(), &[2, 3, 3]);
    assert_eq!(gpu_result.dims(), &[2, 3, 3]);

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "unfold_2d_dim1");
}

/// 2D unfold along dim=0: [6, 4].unfold(0, 3, 1) -> [4, 4, 3]
#[test]
fn test_gpu_unfold_2d_dim0() {
    init();
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[6, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = cpu.unfold(0, 3, 1).unwrap();
    let gpu_result = gpu.unfold(0, 3, 1).unwrap();

    assert_eq!(cpu_result.dims(), &[4, 4, 3]);
    assert_eq!(gpu_result.dims(), &[4, 4, 3]);

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "unfold_2d_dim0");
}

/// 3D STFT framing pattern: [1, 1, T].unfold(2, fft_size, hop_size) -> [1, 1, n_frames, fft_size]
/// This is the exact pattern that replaces 87K narrow() dispatches in Kokoro TTS.
#[test]
fn test_gpu_unfold_3d_stft_pattern() {
    init();
    // Simulate audio: [batch=1, channels=1, samples=256]
    let n_samples = 256;
    let fft_size = 20;
    let hop_size = 5;
    let data: Vec<f32> = (0..n_samples).map(|i| (i as f32) * 0.01).collect();
    let cpu = DynTensor::new(&data, &[1, 1, n_samples], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = cpu.unfold(2, fft_size, hop_size).unwrap();
    let gpu_result = gpu.unfold(2, fft_size, hop_size).unwrap();

    let n_frames = (n_samples - fft_size) / hop_size + 1;
    assert_eq!(cpu_result.dims(), &[1, 1, n_frames, fft_size]);
    assert_eq!(gpu_result.dims(), &[1, 1, n_frames, fft_size]);

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "unfold_3d_stft");
}

/// Larger STFT-like pattern approaching production scale.
/// Tests [1, 1, 2048].unfold(2, 256, 128) -> [1, 1, 14, 256]
#[test]
fn test_gpu_unfold_stft_production_scale() {
    init();
    let n_samples = 2048;
    let fft_size = 256;
    let hop_size = 128;
    let data: Vec<f32> = (0..n_samples).map(|i| ((i as f32) * 0.001).sin()).collect();
    let cpu = DynTensor::new(&data, &[1, 1, n_samples], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = cpu.unfold(2, fft_size, hop_size).unwrap();
    let gpu_result = gpu.unfold(2, fft_size, hop_size).unwrap();

    let n_frames = (n_samples - fft_size) / hop_size + 1;
    assert_eq!(cpu_result.dims(), &[1, 1, n_frames, fft_size]);
    assert_eq!(gpu_result.dims(), &[1, 1, n_frames, fft_size]);

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    // Allow tiny floating-point tolerance since sin() values are not exact integers.
    assert_close(&gpu_vals, &cpu_vals, 0.0, "unfold_stft_production");
}

/// Unfold on a narrow-view GPU tensor (byte_offset > 0).
/// Verifies byte_offset-aware dispatch from narrow GPU → unfold GPU.
#[test]
fn test_gpu_unfold_on_narrow_view() {
    init();
    // Create a large tensor, narrow it, then unfold the narrow view.
    let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 48], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Narrow dim=2 start=8 len=32: GPU narrow uses byte_offset.
    let cpu_narrow = cpu.narrow(2, 8, 32).unwrap();
    let gpu_narrow = gpu.narrow(2, 8, 32).unwrap();

    // Unfold the narrowed view: [1, 1, 32].unfold(2, 8, 4) -> [1, 1, 7, 8]
    let cpu_result = cpu_narrow.unfold(2, 8, 4).unwrap();
    let gpu_result = gpu_narrow.unfold(2, 8, 4).unwrap();

    assert_eq!(cpu_result.dims(), &[1, 1, 7, 8]);
    assert_eq!(gpu_result.dims(), &[1, 1, 7, 8]);

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "unfold_on_narrow_view");
}

/// Step > size (non-overlapping windows).
#[test]
fn test_gpu_unfold_non_overlapping() {
    init();
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[20], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Non-overlapping: step=5 > size=4 means a gap of 1 element between windows.
    let cpu_result = cpu.unfold(0, 4, 5).unwrap();
    let gpu_result = gpu.unfold(0, 4, 5).unwrap();

    assert_eq!(cpu_result.dims(), &[4, 4]);
    assert_eq!(gpu_result.dims(), &[4, 4]);

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "unfold_non_overlapping");
}

/// Step = 1 (maximum overlap, every position).
#[test]
fn test_gpu_unfold_step1() {
    init();
    let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[10], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = cpu.unfold(0, 3, 1).unwrap();
    let gpu_result = gpu.unfold(0, 3, 1).unwrap();

    // (10 - 3) / 1 + 1 = 8 windows of size 3
    assert_eq!(cpu_result.dims(), &[8, 3]);
    assert_eq!(gpu_result.dims(), &[8, 3]);

    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "unfold_step1");
}

/// Production-scale Kokoro TTS STFT framing: 22,050 samples at standard FFT
/// sizes 512, 1024, 2048 (hop = fft_size / 4).
///
/// Before unfold, this pattern dispatched O(n_frames) narrow() calls per
/// resolution — up to 87K total GPU kernel dispatches for multi-resolution
/// STFT. With unfold, each resolution is a single GPU dispatch (3 total).
///
/// This test verifies bit-exact GPU/CPU parity at Kokoro's actual production
/// audio length across all three standard FFT resolutions.
#[test]
fn test_gpu_unfold_kokoro_multi_res_stft() {
    init();
    let n_samples = 22_050; // ~1 second at 22.05 kHz (Kokoro sample rate)
    let data: Vec<f32> = (0..n_samples).map(|i| ((i as f32) * 0.01).sin()).collect();
    let cpu = DynTensor::new(&data, &[n_samples], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Standard multi-resolution STFT FFT sizes used in HiFi-GAN / Kokoro
    for &fft_size in &[512, 1024, 2048] {
        let hop_size = fft_size / 4;
        let n_frames = (n_samples - fft_size) / hop_size + 1;

        let cpu_result = cpu.unfold(0, fft_size, hop_size).unwrap();
        let gpu_result = gpu.unfold(0, fft_size, hop_size).unwrap();

        assert_eq!(
            cpu_result.dims(),
            &[n_frames, fft_size],
            "fft_size={fft_size}: shape mismatch"
        );
        assert_eq!(
            gpu_result.dims(),
            &[n_frames, fft_size],
            "fft_size={fft_size}: GPU shape mismatch"
        );

        let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
        let gpu_vals = gpu_result
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();
        // Data is sin() of small floats — bit-exact parity expected (copy, not compute).
        assert_close(
            &gpu_vals,
            &cpu_vals,
            0.0,
            &format!("kokoro_stft_fft{fft_size}"),
        );
    }
}
