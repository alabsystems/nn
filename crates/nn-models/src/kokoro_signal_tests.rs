// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `kokoro_signal.rs` — harmonic source, har_source building,
//! iSTFT preparation.
//!
//! Part of #2218 (Kokoro epic).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, TensorError};

use super::*;

fn cpu() -> Device {
    Device::Cpu
}

// -- harmonic_source ----------------------------------------------------------

#[test]
fn test_harmonic_source_shape() {
    let f0 = DynTensor::full(&[1, 1, 10], 440.0, DType::F32, &cpu()).unwrap();
    let result = harmonic_source(&f0, 24000.0).unwrap();
    assert_eq!(result.dims(), &[1, 1, 10]);
}

#[test]
fn test_harmonic_source_zero_f0_produces_zero() {
    let f0 = DynTensor::zeros(&[1, 1, 8], DType::F32, &cpu()).unwrap();
    let result = harmonic_source(&f0, 24000.0).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(
            v.abs() < 1e-6,
            "zero F0 should produce zero signal, got {v}"
        );
    }
}

#[test]
fn test_harmonic_source_output_in_sinusoidal_range() {
    let f0 = DynTensor::full(&[1, 1, 100], 1000.0, DType::F32, &cpu()).unwrap();
    let result = harmonic_source(&f0, 24000.0).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.is_finite(), "output must be finite, got {v}");
        assert!(
            *v >= -1.0 && *v <= 1.0,
            "sin() output must be in [-1, 1], got {v}"
        );
    }
}

#[test]
fn test_harmonic_source_batch_preservation() {
    let f0 = DynTensor::full(&[2, 1, 5], 880.0, DType::F32, &cpu()).unwrap();
    let result = harmonic_source(&f0, 24000.0).unwrap();
    assert_eq!(result.dims(), &[2, 1, 5]);
}

// -- build_har_source ---------------------------------------------------------

#[test]
fn test_build_har_source_shape() {
    let f0 = DynTensor::full(&[1, 1, 10], 440.0, DType::F32, &cpu()).unwrap();
    let energy = DynTensor::full(&[1, 1, 10], 0.5, DType::F32, &cpu()).unwrap();
    let result = build_har_source(&f0, &energy, 3, 10, 24000.0).unwrap();
    // Output: [B, 2*n_bins, total_samples] = [1, 6, 10]
    assert_eq!(result.dims(), &[1, 6, 10]);
}

#[test]
fn test_build_har_source_with_padding() {
    // F0/energy have 5 samples but total_samples=10: should zero-pad the tail.
    let f0 = DynTensor::full(&[1, 1, 5], 440.0, DType::F32, &cpu()).unwrap();
    let energy = DynTensor::full(&[1, 1, 5], 0.5, DType::F32, &cpu()).unwrap();
    let result = build_har_source(&f0, &energy, 3, 10, 24000.0).unwrap();
    assert_eq!(result.dims(), &[1, 6, 10]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "all values must be finite"
    );
    // Padded region (indices 5..10 for each channel) should be zero.
    // Layout: [B=1, C=6, T=10]. channel c, time t => index c*10 + t.
    for c in 0..6 {
        for t in 5..10 {
            let idx = c * 10 + t;
            assert!(
                vals[idx].abs() < 1e-6,
                "padded region should be zero: c={c}, t={t}, val={}",
                vals[idx]
            );
        }
    }
}

#[test]
fn test_build_har_source_trimming() {
    // F0/energy have 20 samples but total_samples=10: should trim to 10.
    let f0 = DynTensor::full(&[1, 1, 20], 440.0, DType::F32, &cpu()).unwrap();
    let energy = DynTensor::full(&[1, 1, 20], 0.5, DType::F32, &cpu()).unwrap();
    let result = build_har_source(&f0, &energy, 3, 10, 24000.0).unwrap();
    assert_eq!(result.dims(), &[1, 6, 10]);
}

// -- build_har_from_source ----------------------------------------------------

#[test]
fn test_build_har_from_source_shape() {
    let source = DynTensor::full(&[1, 1, 16], 0.3, DType::F32, &cpu()).unwrap();
    let energy = DynTensor::full(&[1, 1, 16], 0.5, DType::F32, &cpu()).unwrap();
    let result = build_har_from_source(&source, &energy, 3, 16).unwrap();
    assert_eq!(result.dims(), &[1, 6, 16]);
}

#[test]
fn test_build_har_from_source_energy_shorter_than_source() {
    // Source: 16 samples, Energy: 8 samples (lower rate).
    let source = DynTensor::full(&[1, 1, 16], 0.3, DType::F32, &cpu()).unwrap();
    let energy = DynTensor::full(&[1, 1, 8], 0.5, DType::F32, &cpu()).unwrap();
    let result = build_har_from_source(&source, &energy, 3, 16).unwrap();
    assert_eq!(result.dims(), &[1, 6, 16]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "all values must be finite"
    );
}

#[test]
fn test_build_har_from_source_with_tail_padding() {
    // Source: 8 samples, total_samples=16: should zero-pad tail.
    let source = DynTensor::full(&[1, 1, 8], 0.3, DType::F32, &cpu()).unwrap();
    let energy = DynTensor::full(&[1, 1, 8], 0.5, DType::F32, &cpu()).unwrap();
    let result = build_har_from_source(&source, &energy, 3, 16).unwrap();
    assert_eq!(result.dims(), &[1, 6, 16]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Padded region (t=8..16) should be zero for all channels.
    for c in 0..6 {
        for t in 8..16 {
            let idx = c * 16 + t;
            assert!(
                vals[idx].abs() < 1e-6,
                "padded region should be zero: c={c}, t={t}, val={}",
                vals[idx]
            );
        }
    }
}

// -- prepare_istft_input ------------------------------------------------------

#[test]
fn test_prepare_istft_input_basic() {
    let n_fft = 4;
    let n_frames = 3;
    let data: Vec<f32> = (0..n_fft * n_frames).map(|i| i as f32).collect();
    let tensor = DynTensor::new(&data, &[1, n_fft, n_frames], &cpu()).unwrap();
    let (real, imag, frames) = prepare_istft_input(&tensor).unwrap();
    assert_eq!(frames, n_frames);
    // n_bins = n_fft/2 + 1 = 3
    let n_bins = n_fft / 2 + 1;
    assert_eq!(real.len(), n_bins * n_frames);
    assert_eq!(imag.len(), n_bins * n_frames);
}

#[test]
fn test_prepare_istft_input_real_imag_split() {
    // n_fft=4, n_frames=2. Data layout [1, 4, 2]:
    // ch0=[0,1], ch1=[2,3], ch2=[4,5], ch3=[6,7]
    // Real: ch0,ch1 + Nyquist pad (zeros). Imag: ch2,ch3 + DC pad (zeros).
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let tensor = DynTensor::new(&data, &[1, 4, 2], &cpu()).unwrap();
    let (real, imag, frames) = prepare_istft_input(&tensor).unwrap();
    assert_eq!(frames, 2);
    // n_bins = 3, so real = [ch0(0,1), ch1(2,3), nyquist(0,0)]
    assert_eq!(real, vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    // imag = [ch2(4,5), ch3(6,7), dc(0,0)]
    assert_eq!(imag, vec![4.0, 5.0, 6.0, 7.0, 0.0, 0.0]);
}

#[test]
fn test_prepare_istft_input_rank_error() {
    let tensor = DynTensor::zeros(&[4, 3], DType::F32, &cpu()).unwrap();
    let err = prepare_istft_input(&tensor).unwrap_err();
    match err {
        TensorError::RankMismatch {
            expected: 3,
            actual: 2,
        } => {} // expected
        other => panic!("expected RankMismatch, got {other:?}"),
    }
}

#[test]
fn test_prepare_istft_input_batch_not_1_error() {
    let tensor = DynTensor::zeros(&[2, 4, 3], DType::F32, &cpu()).unwrap();
    let err = prepare_istft_input(&tensor).unwrap_err();
    match err {
        TensorError::ShapeMismatch { .. } => {} // expected
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

// -- Constants ----------------------------------------------------------------

#[test]
fn test_kokoro_constants_consistency() {
    assert_eq!(KOKORO_N_BINS, KOKORO_N_FFT / 2 + 1);
    const { assert!(KOKORO_N_FFT > 0) };
    const { assert!(KOKORO_HOP_LENGTH > 0) };
    const { assert!(KOKORO_SAMPLE_RATE > 0) };
}
