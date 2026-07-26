// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for iSTFT: construction, error cases, DFT basis, and window correctness.
//!
//! Round-trip, COLA, and model-parameter tests: `istft_tests_roundtrip.rs`

use super::*;

#[path = "istft_tests_roundtrip.rs"]
mod roundtrip;

fn default_small_params() -> IstftParams {
    IstftParams {
        n_fft: 8,
        hop_length: 4,
        normalized: false,
        center: false,
    }
}

// ---- Construction tests ----

#[test]
fn test_istft_basis_construction_small() {
    let params = default_small_params();
    let basis = IstftBasis::new(params).unwrap();
    assert_eq!(basis.n_bins(), 5); // n_fft/2 + 1
    assert_eq!(basis.cos_basis.len(), 5 * 8);
    assert_eq!(basis.sin_basis.len(), 5 * 8);
    assert_eq!(basis.window.len(), 8);
}

#[test]
fn test_istft_basis_construction_htdemucs() {
    let params = IstftParams {
        n_fft: 4096,
        hop_length: 1024,
        normalized: true,
        center: false,
    };
    let basis = IstftBasis::new(params).unwrap();
    assert_eq!(basis.n_bins(), 2049);
    assert_eq!(basis.cos_basis.len(), 2049 * 4096);
}

#[test]
fn test_istft_basis_construction_kokoro() {
    let params = IstftParams {
        n_fft: 20,
        hop_length: 5,
        normalized: false,
        center: true,
    };
    let basis = IstftBasis::new(params).unwrap();
    assert_eq!(basis.n_bins(), 11);
}

// ---- Error case tests ----

#[test]
fn test_istft_odd_nfft_rejected() {
    let result = IstftBasis::new(IstftParams {
        n_fft: 7,
        hop_length: 3,
        normalized: false,
        center: false,
    });
    assert!(matches!(result, Err(IstftError::OddNfft { n_fft: 7 })));
}

#[test]
fn test_istft_zero_nfft_rejected() {
    let result = IstftBasis::new(IstftParams {
        n_fft: 0,
        hop_length: 1,
        normalized: false,
        center: false,
    });
    assert!(matches!(result, Err(IstftError::OddNfft { n_fft: 0 })));
}

#[test]
fn test_istft_zero_hop_rejected() {
    let result = IstftBasis::new(IstftParams {
        n_fft: 8,
        hop_length: 0,
        normalized: false,
        center: false,
    });
    assert!(matches!(result, Err(IstftError::ZeroHopLength)));
}

#[test]
fn test_istft_length_mismatch() {
    let basis = IstftBasis::new(default_small_params()).unwrap();
    let real = vec![0.0f32; 10]; // 5 bins * 2 frames
    let imag = vec![0.0f32; 15]; // wrong length
    let result = basis.istft(&real, &imag, 2, 16);
    assert!(matches!(
        result,
        Err(IstftError::LengthMismatch {
            real_len: 10,
            imag_len: 15
        })
    ));
}

#[test]
fn test_istft_shape_mismatch() {
    let basis = IstftBasis::new(default_small_params()).unwrap();
    let real = vec![0.0f32; 7]; // doesn't match 5 bins * any integer
    let imag = vec![0.0f32; 7];
    let result = basis.istft(&real, &imag, 2, 16);
    assert!(matches!(result, Err(IstftError::ShapeMismatch { .. })));
}

#[test]
fn test_istft_non_finite_input_rejected() {
    let basis = IstftBasis::new(default_small_params()).unwrap();
    let n_bins = basis.n_bins();
    let n_frames = 2;
    let mut real = vec![0.0f32; n_bins * n_frames];
    real[0] = f32::NAN;
    let imag = vec![0.0f32; n_bins * n_frames];
    let result = basis.istft(&real, &imag, n_frames, 16);
    assert!(matches!(result, Err(IstftError::NonFiniteInput)));
}

#[test]
fn test_istft_non_finite_imag_rejected() {
    let basis = IstftBasis::new(default_small_params()).unwrap();
    let n_bins = basis.n_bins();
    let n_frames = 2;
    let real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];
    imag[3] = f32::INFINITY;
    let result = basis.istft(&real, &imag, n_frames, 16);
    assert!(matches!(result, Err(IstftError::NonFiniteInput)));
}

// ---- DFT basis correctness tests ----

#[test]
fn test_dft_basis_dc_component() {
    let basis = IstftBasis::new(default_small_params()).unwrap();
    let n_fft = basis.params.n_fft;
    for k in 0..n_fft {
        assert!(
            (basis.cos_basis[k] - 1.0).abs() < 1e-6,
            "cos[0,{k}] should be 1.0, got {}",
            basis.cos_basis[k]
        );
        assert!(
            basis.sin_basis[k].abs() < 1e-6,
            "sin[0,{k}] should be 0.0, got {}",
            basis.sin_basis[k]
        );
    }
}

#[test]
fn test_dft_basis_nyquist_component() {
    // Nyquist (f=n_fft/2): cos(pi*k) = (-1)^k, sin(pi*k) = 0.
    let basis = IstftBasis::new(default_small_params()).unwrap();
    let n_fft = basis.params.n_fft;
    let f = n_fft / 2; // = 4 for n_fft=8
    for k in 0..n_fft {
        let expected_cos = if k % 2 == 0 { 1.0 } else { -1.0 };
        assert!(
            (basis.cos_basis[f * n_fft + k] - expected_cos).abs() < 1e-6,
            "cos[{f},{k}] should be {expected_cos}, got {}",
            basis.cos_basis[f * n_fft + k]
        );
        assert!(
            basis.sin_basis[f * n_fft + k].abs() < 1e-5,
            "sin[{f},{k}] should be ~0.0, got {}",
            basis.sin_basis[f * n_fft + k]
        );
    }
}

// ---- Window tests ----

#[test]
fn test_hann_window_endpoints() {
    // Hann window: w[0] = 0, w[n_fft/2] = 1, w[n_fft-1] ~= 0
    let basis = IstftBasis::new(default_small_params()).unwrap();
    assert!(basis.window[0].abs() < 1e-6, "window[0] should be ~0");
    assert!(
        (basis.window[4] - 1.0).abs() < 1e-6,
        "window[n/2] should be ~1"
    );
}

#[test]
fn test_hann_window_values_in_range() {
    let basis = IstftBasis::new(default_small_params()).unwrap();
    for (k, &w) in basis.window.iter().enumerate() {
        assert!((0.0..=1.0).contains(&w), "window[{k}] = {w} outside [0, 1]");
    }
}
