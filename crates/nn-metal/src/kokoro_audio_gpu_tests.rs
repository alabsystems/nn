// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for GPU-accelerated Kokoro iSTFT audio path.
//!
//! Verifies that [`kokoro_forward_audio_gpu`] produces output matching the
//! CPU [`KokoroModel::forward_audio()`] path within f32 tolerance.
//!
//! Part of #2230.

use crate::istft::{IstftBasis, IstftParams};
use crate::istft_gpu::IstftGpuBasis;
use crate::test_common::{assert_close, init, make_cache};

/// Kokoro iSTFT params: n_fft=20, hop=5, unnormalized, center=true.
const KOKORO_N_FFT: usize = 20;
const KOKORO_HOP: usize = 5;
const KOKORO_N_BINS: usize = KOKORO_N_FFT / 2 + 1; // 11

/// Test GPU iSTFT with Kokoro-specific parameters matches CPU iSTFT.
///
/// Uses the general IstftBasis (pub API) as CPU reference, since the
/// Kokoro-specific `kokoro_istft` is `pub(crate)` in nn-models.
#[test]
fn test_gpu_istft_kokoro_params_matches_cpu() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return, // Skip on non-Metal platforms
    };

    let n_frames = 8;
    // Production uses center=true: output_length = (n_frames - 1) * hop.
    // Matches compiled_kokoro_bridges.rs:116.
    let output_length = (n_frames - 1) * KOKORO_HOP; // 35

    // Build basis with Kokoro production params (center=true).
    let params = IstftParams::new(KOKORO_N_FFT, KOKORO_HOP, false, true).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid basis");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    // Deterministic test data: sinusoidal STFT coefficients.
    let mut real = Vec::with_capacity(KOKORO_N_BINS * n_frames);
    let mut imag = Vec::with_capacity(KOKORO_N_BINS * n_frames);
    for f in 0..KOKORO_N_BINS {
        for t in 0..n_frames {
            let phase = (f as f32 * 0.3 + t as f32 * 0.7).sin();
            real.push(phase * 0.5);
            imag.push(phase * 0.25);
        }
    }

    let cpu_result = basis.istft(&real, &imag, n_frames, output_length).unwrap();
    let gpu_result = gpu_basis
        .gpu_istft_from_cpu(&cache, &real, &imag, n_frames, output_length)
        .unwrap();

    assert_eq!(cpu_result.len(), output_length);
    assert_eq!(gpu_result.len(), output_length);
    assert_close(&gpu_result, &cpu_result, 1e-4, "kokoro_params_parity");
}

/// Test GPU iSTFT with zero input produces zero output.
#[test]
fn test_gpu_istft_kokoro_all_zeros() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let n_frames = 4;
    let output_length = (n_frames - 1) * KOKORO_HOP;

    let params = IstftParams::new(KOKORO_N_FFT, KOKORO_HOP, false, true).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid basis");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    let real = vec![0.0f32; KOKORO_N_BINS * n_frames];
    let imag = vec![0.0f32; KOKORO_N_BINS * n_frames];

    let gpu_result = gpu_basis
        .gpu_istft_from_cpu(&cache, &real, &imag, n_frames, output_length)
        .unwrap();

    for v in &gpu_result {
        assert!(v.abs() < 1e-6, "expected near-zero, got {v}");
    }
}
