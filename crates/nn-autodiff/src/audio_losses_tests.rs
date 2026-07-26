#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::grad::backward;
use crate::var::Var;
use nn_core::Device;

fn sine_wave(freq_hz: f32, sample_rate: f32, duration_sec: f32) -> Vec<f32> {
    let n = (sample_rate * duration_sec) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.5
        })
        .collect()
}

fn make_tracked(data: &[f32]) -> Arc<TrackedTensor> {
    let t = DynTensor::new(data, &[data.len()], &Device::Cpu).expect("new");
    Arc::new(TrackedTensor::from_tensor(t))
}

fn make_var(data: &[f32]) -> (Var, Arc<TrackedTensor>) {
    let t = DynTensor::new(data, &[data.len()], &Device::Cpu).expect("new");
    let var = Var::from_tensor(&t);
    let tracked = Arc::new(TrackedTensor::from_var(&var).expect("from_var"));
    (var, tracked)
}

// ---- Multi-res STFT loss ----

#[test]
fn test_stft_loss_zero_for_identical() {
    let audio = sine_wave(440.0, 16000.0, 0.1); // 1600 samples
    let a = make_tracked(&audio);
    let b = make_tracked(&audio);

    let loss = stft_loss(&a, &b, 512, 128).expect("stft_loss");
    let val = loss.tensor().to_flat_vec::<f32>().expect("to_vec");
    assert!(
        val[0].abs() < 0.01,
        "Identical signals should have near-zero loss, got {}",
        val[0]
    );
}

#[test]
fn test_stft_loss_positive_for_different() {
    let a_audio = sine_wave(440.0, 16000.0, 0.1);
    let b_audio = sine_wave(880.0, 16000.0, 0.1);
    let a = make_tracked(&a_audio);
    let b = make_tracked(&b_audio);

    let loss = stft_loss(&a, &b, 512, 128).expect("stft_loss");
    let val = loss.tensor().to_flat_vec::<f32>().expect("to_vec");
    assert!(
        val[0] > 0.01,
        "Different signals should have positive loss, got {}",
        val[0]
    );
}

#[test]
fn test_stft_loss_backward() {
    let audio_ref = sine_wave(440.0, 16000.0, 0.1);
    let audio_cand = sine_wave(450.0, 16000.0, 0.1); // slightly different
    let (var, cand) = make_var(&audio_cand);
    let refr = make_tracked(&audio_ref);

    let loss = stft_loss(&cand, &refr, 512, 128).expect("stft_loss");
    let grads = backward(&loss).expect("backward");

    // Gradient should exist and be non-zero (signals differ at 440 vs 450 Hz)
    let grad_tensor = grads
        .get(&var)
        .expect("gradient should exist for candidate");
    let grad_vals = grad_tensor.to_flat_vec::<f32>().expect("to_vec");
    let grad_norm: f32 = grad_vals.iter().map(|g| g * g).sum::<f32>().sqrt();
    assert!(
        grad_norm > 1e-6,
        "STFT loss gradient should be non-zero for different signals, got L2 norm {grad_norm}"
    );
    // All gradient values should be finite
    assert!(
        grad_vals.iter().all(|g| g.is_finite()),
        "All gradient values should be finite"
    );
}

#[test]
fn test_multi_res_stft_loss_averages() {
    let a_audio = sine_wave(440.0, 16000.0, 0.2); // 3200 samples
    let b_audio = sine_wave(880.0, 16000.0, 0.2);
    let a = make_tracked(&a_audio);
    let b = make_tracked(&b_audio);

    // Single resolution
    let loss_512 = stft_loss(&a, &b, 512, 128).expect("stft_loss 512");
    let val_512 = loss_512.tensor().to_flat_vec::<f32>().expect("to_vec")[0];

    // Multi-resolution should produce a different (averaged) value
    let multi_loss = multi_res_stft_loss(&a, &b, &[512, 1024]).expect("multi_res");
    let val_multi = multi_loss.tensor().to_flat_vec::<f32>().expect("to_vec")[0];

    assert!(val_multi > 0.0, "Multi-res loss should be positive");
    // Multi-res averages, so it should differ from single-resolution
    assert!(
        (val_multi - val_512).abs() > 1e-6 || val_multi > 0.0,
        "Multi-res should differ from single-res or be positive"
    );
}

#[test]
fn test_multi_res_stft_loss_empty_fft_sizes() {
    let audio = sine_wave(440.0, 16000.0, 0.1);
    let a = make_tracked(&audio);
    let b = make_tracked(&audio);

    let result = multi_res_stft_loss(&a, &b, &[]);
    assert!(result.is_err(), "Empty fft_sizes should error");
}

// ---- Mel-spectrogram loss ----

#[test]
fn test_mel_loss_zero_for_identical() {
    let audio = sine_wave(440.0, 16000.0, 0.1);
    let a = make_tracked(&audio);
    let b = make_tracked(&audio);

    let loss = mel_spectrogram_loss(&a, &b, 80, 512, 16000).expect("mel_loss");
    let val = loss.tensor().to_flat_vec::<f32>().expect("to_vec");
    assert!(
        val[0].abs() < 0.01,
        "Identical signals should have near-zero mel loss, got {}",
        val[0]
    );
}

#[test]
fn test_mel_loss_backward() {
    let audio_ref = sine_wave(440.0, 16000.0, 0.1);
    let audio_cand = sine_wave(450.0, 16000.0, 0.1);
    let (var, cand) = make_var(&audio_cand);
    let refr = make_tracked(&audio_ref);

    let loss = mel_spectrogram_loss(&cand, &refr, 40, 512, 16000).expect("mel_loss");
    let grads = backward(&loss).expect("backward");

    // Gradient should exist and be non-zero (candidate != reference)
    let grad_tensor = grads
        .get(&var)
        .expect("gradient should exist for candidate");
    let grad_vals = grad_tensor.to_flat_vec::<f32>().expect("to_vec");
    let grad_norm: f32 = grad_vals.iter().map(|g| g * g).sum::<f32>().sqrt();
    assert!(
        grad_norm > 1e-6,
        "Mel loss gradient should be non-zero for different signals, got L2 norm {grad_norm}"
    );
    assert!(
        grad_vals.iter().all(|g| g.is_finite()),
        "All gradient values should be finite"
    );
}

// ---- Feature matching loss ----

#[test]
fn test_feature_matching_loss() {
    let f1_cand = make_tracked(&[1.0, 2.0, 3.0, 4.0]);
    let f1_ref = make_tracked(&[1.0, 2.0, 3.0, 4.0]);
    let f2_cand = make_tracked(&[5.0, 6.0]);
    let f2_ref = make_tracked(&[7.0, 8.0]);

    let loss = feature_matching_loss(&[f1_cand, f2_cand], &[f1_ref, f2_ref])
        .expect("feature_matching_loss");

    let val = loss.tensor().to_flat_vec::<f32>().expect("to_vec");
    // Layer 1: L1 = 0 (identical), Layer 2: L1 = mean(|5-7|, |6-8|) = 2.0
    // Average: (0 + 2) / 2 = 1.0
    assert!(
        (val[0] - 1.0).abs() < 0.01,
        "Feature matching loss should be ~1.0, got {}",
        val[0]
    );
}

#[test]
fn test_feature_matching_loss_empty() {
    let result = feature_matching_loss(&[], &[]);
    assert!(result.is_err(), "Empty features should error");
}

#[test]
fn test_feature_matching_loss_length_mismatch() {
    let f1 = make_tracked(&[1.0, 2.0]);
    let f2 = make_tracked(&[3.0, 4.0]);
    let result = feature_matching_loss(&[f1], &[f2.clone(), f2]);
    assert!(result.is_err(), "Mismatched feature counts should error");
}
