// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DSP utilities: RMS, DC offset, click detection, FFT, YIN F0.

use nn_tts_verify::dsp;

fn sine_wave(freq: f64, sample_rate: u32, duration_sec: f64, amplitude: f32) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            amplitude * (2.0 * std::f64::consts::PI * freq * t).sin() as f32
        })
        .collect()
}

#[test]
fn test_rms_silence() {
    let silence = vec![0.0_f32; 1000];
    assert!((dsp::rms(&silence) - 0.0).abs() < 1e-10);
}

#[test]
fn test_rms_sine() {
    let signal = sine_wave(440.0, 16000, 1.0, 1.0);
    let r = dsp::rms(&signal);
    // RMS of a sine wave = amplitude / sqrt(2) ≈ 0.707.
    assert!(
        (r - 0.707).abs() < 0.01,
        "RMS of unit sine should be ~0.707, got {r}"
    );
}

#[test]
fn test_rms_empty() {
    assert!((dsp::rms(&[]) - 0.0).abs() < 1e-10);
}

#[test]
fn test_dc_offset_zero_mean() {
    let signal = sine_wave(440.0, 16000, 1.0, 0.5);
    let dc = dsp::dc_offset(&signal);
    assert!(
        dc.abs() < 0.01,
        "Sine wave should have near-zero DC offset, got {dc}"
    );
}

#[test]
fn test_dc_offset_with_bias() {
    let mut signal = vec![0.5_f32; 1000];
    signal.extend(vec![0.5_f32; 1000]);
    let dc = dsp::dc_offset(&signal);
    assert!((dc - 0.5).abs() < 1e-6);
}

#[test]
fn test_max_sample_diff_smooth() {
    let signal = sine_wave(100.0, 16000, 0.1, 0.3);
    let diff = dsp::max_sample_diff(&signal);
    assert!(
        diff < 0.05,
        "Smooth low-frequency sine should have small diff, got {diff}"
    );
}

#[test]
fn test_max_sample_diff_click() {
    let mut signal = vec![0.0_f32; 100];
    signal[50] = 1.0; // Sharp click.
    let diff = dsp::max_sample_diff(&signal);
    assert!(
        (diff - 1.0).abs() < 1e-6,
        "Click should produce diff=1.0, got {diff}"
    );
}

#[test]
fn test_max_sample_diff_single_sample() {
    assert!((dsp::max_sample_diff(&[0.5]) - 0.0).abs() < 1e-10);
}

#[test]
fn test_spectral_band_energy_sine() {
    let signal = sine_wave(1000.0, 16000, 0.5, 0.5);
    let energy = dsp::spectral_band_energy(&signal, 16000, 8).unwrap();
    assert_eq!(energy.len(), 8);
    // Energy at 1 kHz should be highest (band 1 of 8 = 1-2 kHz range).
    let max_band = energy
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert!(
        max_band <= 2,
        "1 kHz tone energy should be in low bands, got band {max_band}"
    );
}

#[test]
fn test_spectral_band_energy_empty() {
    let result = dsp::spectral_band_energy(&[], 16000, 8);
    assert!(result.is_err());
}

#[test]
fn test_spectral_band_energy_zero_bands_returns_error() {
    let signal = sine_wave(440.0, 16000, 0.5, 0.3);
    let result = dsp::spectral_band_energy(&signal, 16000, 0);
    assert!(result.is_err(), "n_bands=0 should return error");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("n_bands"),
        "error should mention n_bands: {msg}"
    );
}

#[test]
fn test_spectral_band_energy_zero_sample_rate_returns_error() {
    let signal = sine_wave(440.0, 16000, 0.5, 0.3);
    let result = dsp::spectral_band_energy(&signal, 0, 8);
    assert!(result.is_err(), "sample_rate=0 should return error");
}

#[test]
fn test_autocorrelation_periodic() {
    let signal = sine_wave(440.0, 16000, 0.1, 0.5);
    let ac = dsp::autocorrelation(&signal, 100);
    // Autocorrelation at lag 0 = energy.
    assert!(ac[0] > 0.0, "Autocorrelation at lag 0 should be positive");
    // For a periodic signal, autocorrelation should peak near the period.
    let period = (16000.0_f64 / 440.0).round() as usize; // ~36 samples.
    if period < ac.len() {
        // The AC at the period should be close to the AC at lag 0.
        let ratio = ac[period] / ac[0];
        assert!(
            ratio > 0.5,
            "AC at period should be high for periodic signal, got {ratio}"
        );
    }
}

#[test]
fn test_yin_f0_pure_tone() {
    let signal = sine_wave(220.0, 16000, 0.5, 0.5);
    let f0s = dsp::yin_f0(&signal, 16000, 640, 160, 0.15).unwrap();
    assert!(!f0s.is_empty());
    let voiced: Vec<f64> = f0s.iter().copied().filter(|&f| f > 0.0).collect();
    assert!(
        !voiced.is_empty(),
        "Should detect voiced frames in 220 Hz tone"
    );
    let mean_f0: f64 = voiced.iter().sum::<f64>() / voiced.len() as f64;
    assert!(
        (mean_f0 - 220.0).abs() < 20.0,
        "Mean F0 should be near 220 Hz, got {mean_f0}"
    );
}

#[test]
fn test_yin_f0_empty() {
    let result = dsp::yin_f0(&[], 16000, 640, 160, 0.15);
    assert!(result.is_err());
}

#[test]
fn test_mel_filterbank_shape() {
    let fb = dsp::mel_filterbank(16000, 1024, 40);
    assert_eq!(fb.len(), 40, "Should have 40 mel filters");
    assert_eq!(
        fb[0].len(),
        513,
        "Each filter should span n_fft/2+1=513 bins"
    );
}

#[test]
fn test_mel_filterbank_non_negative() {
    let fb = dsp::mel_filterbank(16000, 1024, 40);
    for filter in &fb {
        for &val in filter {
            assert!(val >= 0.0, "Mel filter weights should be non-negative");
        }
    }
}

#[test]
fn test_spectral_tilt_speech_like() {
    // Speech-like signal with harmonics (natural falloff).
    let n = 16000; // 1 second at 16 kHz.
    let mut signal = vec![0.0_f32; n];
    for (i, sample) in signal.iter_mut().enumerate() {
        let t = i as f64 / 16000.0;
        *sample = (0.5 * (2.0 * std::f64::consts::PI * 200.0 * t).sin()
            + 0.25 * (2.0 * std::f64::consts::PI * 400.0 * t).sin()
            + 0.12 * (2.0 * std::f64::consts::PI * 800.0 * t).sin()
            + 0.06 * (2.0 * std::f64::consts::PI * 1600.0 * t).sin()) as f32;
    }
    let tilt = dsp::spectral_tilt(&signal, 16000).unwrap();
    // Speech-like harmonic series should have negative tilt.
    assert!(
        tilt < 0.0,
        "Harmonic signal should have negative tilt, got {tilt}"
    );
}

#[test]
fn test_hnr_periodic_signal() {
    let signal = sine_wave(200.0, 16000, 0.5, 0.5);
    let hnr_val = dsp::hnr(&signal, 16000).unwrap();
    // Pure tone should have high HNR.
    assert!(
        hnr_val > 10.0,
        "Pure tone should have high HNR, got {hnr_val}"
    );
}

#[test]
fn test_hnr_noise() {
    // White noise (pseudo-random).
    let n = 16000;
    let mut signal = vec![0.0_f32; n];
    let mut seed: u64 = 42;
    for s in &mut signal {
        // Simple LCG PRNG.
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        *s = ((seed >> 33) as f64 / f64::from(u32::MAX) * 2.0 - 1.0) as f32 * 0.3;
    }
    let hnr_val = dsp::hnr(&signal, 16000).unwrap();
    // Noise should have low HNR.
    assert!(hnr_val < 10.0, "Noise should have low HNR, got {hnr_val}");
}
