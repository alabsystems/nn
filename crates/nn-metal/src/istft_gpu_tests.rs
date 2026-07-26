// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU/CPU parity tests for iSTFT.
//!
//! Verifies that [`IstftGpuBasis::gpu_istft_from_cpu`] produces output matching
//! [`IstftBasis::istft`] within f32 tolerance for various configurations.
//!
//! Part of #1393, Stage 5 of #1370.

use crate::istft::{IstftBasis, IstftParams};
use crate::istft_gpu::IstftGpuBasis;
use crate::test_common::{assert_close, init, make_cache};

/// Small test configuration (fast, exercises core algorithm).
const SMALL_N_FFT: usize = 8;
const SMALL_HOP: usize = 2;
const SMALL_N_BINS: usize = SMALL_N_FFT / 2 + 1; // 5
const SMALL_N_FRAMES: usize = 4;

/// Generate deterministic test data: sinusoidal STFT coefficients.
fn generate_test_stft(n_bins: usize, n_frames: usize) -> (Vec<f32>, Vec<f32>) {
    let mut real = Vec::with_capacity(n_bins * n_frames);
    let mut imag = Vec::with_capacity(n_bins * n_frames);
    for f in 0..n_bins {
        for t in 0..n_frames {
            let phase = (f as f32 * 0.3 + t as f32 * 0.7).sin();
            real.push(phase * 0.5);
            imag.push(phase * 0.25);
        }
    }
    (real, imag)
}

#[test]
fn test_gpu_istft_small_normalized_center() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return, // Skip on non-Metal platforms
    };

    let params = IstftParams::new(SMALL_N_FFT, SMALL_HOP, true, true).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid params");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    let (real, imag) = generate_test_stft(SMALL_N_BINS, SMALL_N_FRAMES);
    let output_len = 16;

    let cpu_result = basis
        .istft(&real, &imag, SMALL_N_FRAMES, output_len)
        .unwrap();
    let gpu_result = gpu_basis
        .gpu_istft_from_cpu(&cache, &real, &imag, SMALL_N_FRAMES, output_len)
        .unwrap();

    assert_close(&gpu_result, &cpu_result, 1e-4, "small_normalized_center");
}

#[test]
fn test_gpu_istft_small_unnormalized_no_center() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let params = IstftParams::new(SMALL_N_FFT, SMALL_HOP, false, false).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid params");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    let (real, imag) = generate_test_stft(SMALL_N_BINS, SMALL_N_FRAMES);
    let full_len = SMALL_N_FFT + (SMALL_N_FRAMES - 1) * SMALL_HOP;

    let cpu_result = basis.istft(&real, &imag, SMALL_N_FRAMES, full_len).unwrap();
    let gpu_result = gpu_basis
        .gpu_istft_from_cpu(&cache, &real, &imag, SMALL_N_FRAMES, full_len)
        .unwrap();

    assert_close(
        &gpu_result,
        &cpu_result,
        1e-4,
        "small_unnormalized_no_center",
    );
}

#[test]
fn test_gpu_istft_medium_htdemucs_params() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    // HTDemucs-like parameters (but smaller n_fft for test speed).
    let n_fft = 64;
    let hop = n_fft / 4; // 16
    let n_bins = n_fft / 2 + 1; // 33
    let n_frames = 8;

    let params = IstftParams::new(n_fft, hop, true, true).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid params");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    let (real, imag) = generate_test_stft(n_bins, n_frames);
    let output_len = 128;

    let cpu_result = basis.istft(&real, &imag, n_frames, output_len).unwrap();
    let gpu_result = gpu_basis
        .gpu_istft_from_cpu(&cache, &real, &imag, n_frames, output_len)
        .unwrap();

    assert_close(&gpu_result, &cpu_result, 1e-4, "medium_htdemucs_params");
}

#[test]
fn test_gpu_istft_kokoro_params() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    // Kokoro-82M parameters: n_fft=20, hop=5, normalized=false, center=false.
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1; // 11
    let n_frames = 10;

    let params = IstftParams::new(n_fft, hop, false, false).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid params");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    let (real, imag) = generate_test_stft(n_bins, n_frames);
    let full_len = n_fft + (n_frames - 1) * hop;

    let cpu_result = basis.istft(&real, &imag, n_frames, full_len).unwrap();
    let gpu_result = gpu_basis
        .gpu_istft_from_cpu(&cache, &real, &imag, n_frames, full_len)
        .unwrap();

    assert_close(&gpu_result, &cpu_result, 1e-4, "kokoro_params");
}

#[test]
fn test_gpu_istft_single_frame() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let n_fft = 16;
    let hop = 4;
    let n_bins = n_fft / 2 + 1; // 9
    let n_frames = 1;

    let params = IstftParams::new(n_fft, hop, true, false).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid params");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    let (real, imag) = generate_test_stft(n_bins, n_frames);
    let full_len = n_fft;

    let cpu_result = basis.istft(&real, &imag, n_frames, full_len).unwrap();
    let gpu_result = gpu_basis
        .gpu_istft_from_cpu(&cache, &real, &imag, n_frames, full_len)
        .unwrap();

    assert_close(&gpu_result, &cpu_result, 1e-4, "single_frame");
}

#[test]
fn test_gpu_istft_output_padding() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    // Request output_length longer than the natural signal length.
    let params = IstftParams::new(SMALL_N_FFT, SMALL_HOP, true, false).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid params");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    let (real, imag) = generate_test_stft(SMALL_N_BINS, SMALL_N_FRAMES);
    let natural_len = SMALL_N_FFT + (SMALL_N_FRAMES - 1) * SMALL_HOP;
    let padded_len = natural_len + 10;

    let cpu_result = basis
        .istft(&real, &imag, SMALL_N_FRAMES, padded_len)
        .unwrap();
    let gpu_result = gpu_basis
        .gpu_istft_from_cpu(&cache, &real, &imag, SMALL_N_FRAMES, padded_len)
        .unwrap();

    assert_eq!(gpu_result.len(), padded_len);
    assert_close(&gpu_result, &cpu_result, 1e-4, "output_padding");
    // Verify the padded region is zeros.
    for &v in &gpu_result[natural_len..] {
        assert_eq!(v, 0.0, "padded region should be zero");
    }
}

#[test]
fn test_gpu_istft_input_validation() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let params = IstftParams::new(SMALL_N_FFT, SMALL_HOP, true, false).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid params");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    // Wrong-length input should error.
    let result = gpu_basis.gpu_istft_from_cpu(&cache, &[1.0, 2.0], &[3.0], 1, 8);
    assert!(result.is_err(), "mismatched real/imag lengths should error");

    // Non-finite input should error.
    let n_bins = SMALL_N_FFT / 2 + 1;
    let mut real = vec![0.0; n_bins];
    real[0] = f32::NAN;
    let imag = vec![0.0; n_bins];
    let result = gpu_basis.gpu_istft_from_cpu(&cache, &real, &imag, 1, 8);
    assert!(result.is_err(), "NaN input should error");
}

/// Regression test for #1912: gpu_istft must produce correct output even when
/// called within with_gpu_scope. Before the fix, GpuScope deferred commit_and_wait
/// causing CPU readback to return stale zeros.
#[test]
fn test_gpu_istft_within_gpu_scope_matches_without() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let params = IstftParams::new(SMALL_N_FFT, SMALL_HOP, true, true).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid params");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    let (real, imag) = generate_test_stft(SMALL_N_BINS, SMALL_N_FRAMES);
    let output_len = 16;

    // Without scope.
    let result_no_scope = gpu_basis
        .gpu_istft_from_cpu(&cache, &real, &imag, SMALL_N_FRAMES, output_len)
        .unwrap();

    // With scope — must produce identical output.
    let result_with_scope = crate::gpu_scope::with_gpu_scope(|| {
        gpu_basis.gpu_istft_from_cpu(&cache, &real, &imag, SMALL_N_FRAMES, output_len)
    })
    .unwrap();

    assert_close(
        &result_with_scope,
        &result_no_scope,
        0.0, // bit-exact match expected
        "gpu_istft_scope_vs_no_scope",
    );
    // Verify output is non-zero (not stale).
    let sum: f32 = result_with_scope.iter().map(|v| v.abs()).sum();
    assert!(
        sum > 0.0,
        "gpu_istft output should not be all-zeros (regression: stale readback)"
    );
}

#[test]
fn test_gpu_istft_larger_config() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    // Larger configuration closer to production scale.
    let n_fft = 256;
    let hop = 64;
    let n_bins = n_fft / 2 + 1; // 129
    let n_frames = 16;

    let params = IstftParams::new(n_fft, hop, true, true).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid params");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    let (real, imag) = generate_test_stft(n_bins, n_frames);
    let output_len = 512;

    let cpu_result = basis.istft(&real, &imag, n_frames, output_len).unwrap();
    let gpu_result = gpu_basis
        .gpu_istft_from_cpu(&cache, &real, &imag, n_frames, output_len)
        .unwrap();

    assert_close(&gpu_result, &cpu_result, 1e-3, "larger_config");
}

/// End-to-end cross-validation: (magnitude, phase) → polar-to-rect → iSTFT → PCM.
///
/// CPU path uses decomposed cos/sin/mul ops; GPU path uses fused sincos kernel.
/// This tests the composition that existing tests cover individually, catching
/// any shape/layout mismatch between polar-to-rect output and iSTFT input.
///
/// Part of #2545.
#[test]
fn test_end_to_end_magnitude_phase_to_pcm_cpu_vs_gpu() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    // Kokoro-82M dimensions: n_fft=20, hop=5, n_bins=11.
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1; // 11
    let n_frames = 10;
    let total = n_bins * n_frames;
    let output_length = n_fft + (n_frames - 1) * hop; // 65

    // Generate deterministic (magnitude, phase) test data.
    let mag_data = nn_core::test_prng::rand_f32_vec(42, total, 0.0, 5.0);
    let phase_data =
        nn_core::test_prng::rand_f32_vec(99, total, -std::f32::consts::PI, std::f32::consts::PI);

    // -- CPU path: decomposed polar-to-rect + CPU iSTFT --
    let cpu_real: Vec<f32> = mag_data
        .iter()
        .zip(phase_data.iter())
        .map(|(m, p)| m * p.cos())
        .collect();
    let cpu_imag: Vec<f32> = mag_data
        .iter()
        .zip(phase_data.iter())
        .map(|(m, p)| m * p.sin())
        .collect();
    let params = IstftParams::new(n_fft, hop, false, false).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid basis");
    let cpu_pcm = basis
        .istft(&cpu_real, &cpu_imag, n_frames, output_length)
        .unwrap();

    // -- GPU path: fused gpu_polar_to_rect + GPU iSTFT --
    let gpu_device = nn_core::Device::metal();
    let mag_tensor =
        nn_core::dyn_tensor::DynTensor::from_vec(mag_data, &[1, n_bins, n_frames], &gpu_device)
            .unwrap();
    let phase_tensor =
        nn_core::dyn_tensor::DynTensor::from_vec(phase_data, &[1, n_bins, n_frames], &gpu_device)
            .unwrap();
    let (real_tensor, imag_tensor) =
        crate::dyn_tensor_metal::gpu_polar_to_rect(&mag_tensor, &phase_tensor).unwrap();

    // Flatten GPU real/imag to slices for gpu_istft_from_cpu.
    let gpu_real = real_tensor
        .to_device(&nn_core::Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let gpu_imag = imag_tensor
        .to_device(&nn_core::Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");
    let gpu_pcm = gpu_basis
        .gpu_istft_from_cpu(&cache, &gpu_real, &gpu_imag, n_frames, output_length)
        .unwrap();

    // -- Compare --
    assert_eq!(cpu_pcm.len(), gpu_pcm.len());
    assert_close(
        &gpu_pcm,
        &cpu_pcm,
        1e-3,
        "end_to_end_magnitude_phase_to_pcm",
    );
}

/// Fused polar→iSTFT single-dispatch kernel vs separate polar-to-rect + IDFT + OLA.
///
/// The fused kernel (`gpu_istft_from_polar`) combines all three operations into
/// one Metal dispatch. This test validates numerical equivalence against the
/// two-kernel path (`gpu_polar_to_rect` + `gpu_istft`).
///
/// Part of iSTFT fusion (#3351).
#[test]
fn test_fused_polar_istft_vs_separate_kernels() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    // Kokoro dimensions: n_fft=20, hop=5, n_bins=11.
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 40;
    let total = n_bins * n_frames;
    let output_length = n_fft + (n_frames - 1) * hop;

    let mag_data = nn_core::test_prng::rand_f32_vec(42, total, 0.0, 5.0);
    let phase_data =
        nn_core::test_prng::rand_f32_vec(99, total, -std::f32::consts::PI, std::f32::consts::PI);

    let params = IstftParams::new(n_fft, hop, false, false).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid basis");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    // -- Separate-kernel path: polar_to_rect + gpu_istft --
    let cpu_real: Vec<f32> = mag_data
        .iter()
        .zip(phase_data.iter())
        .map(|(m, p)| m * p.cos())
        .collect();
    let cpu_imag: Vec<f32> = mag_data
        .iter()
        .zip(phase_data.iter())
        .map(|(m, p)| m * p.sin())
        .collect();
    let separate_pcm = gpu_basis
        .gpu_istft_from_cpu(&cache, &cpu_real, &cpu_imag, n_frames, output_length)
        .unwrap();

    // -- Fused-kernel path: gpu_istft_from_polar --
    let ctx = crate::metal_backend::global_metal_context().expect("Metal context");
    let mag_buf = ctx.create_buffer(&mag_data).expect("mag buffer");
    let phase_buf = ctx.create_buffer(&phase_data).expect("phase buffer");
    let fused_pcm = gpu_basis
        .gpu_istft_from_polar(&cache, &mag_buf, 0, &phase_buf, 0, n_frames, output_length)
        .unwrap();

    // -- Compare: fused vs separate must match within f32 tolerance --
    assert_eq!(separate_pcm.len(), fused_pcm.len());
    assert_close(
        &fused_pcm,
        &separate_pcm,
        1e-3,
        "fused_polar_istft_vs_separate",
    );

    // Verify output is non-trivial.
    let max_abs = fused_pcm.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs > 1e-6,
        "fused iSTFT output should not be all-zeros (max_abs={max_abs:.6e})"
    );
}

/// Fused polar→iSTFT with center=true (Kokoro production config).
///
/// Validates center-trim logic in the fused path matches the separate path.
#[test]
fn test_fused_polar_istft_center_trim() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let n_fft: usize = 20;
    let hop: usize = 5;
    let n_bins = n_fft / 2 + 1;
    let n_frames: usize = 20;
    let total = n_bins * n_frames;
    // Center-trimmed output length (Kokoro convention).
    let output_length = n_frames.saturating_sub(1) * hop;

    let mag_data = nn_core::test_prng::rand_f32_vec(17, total, 0.0, 3.0);
    let phase_data =
        nn_core::test_prng::rand_f32_vec(23, total, -std::f32::consts::PI, std::f32::consts::PI);

    // center=true: IstftParams(n_fft=20, hop=5, normalized=false, center=true)
    let params = IstftParams::new(n_fft, hop, false, true).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid basis");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    // CPU reference: manual polar_to_rect + CPU iSTFT.
    let cpu_real: Vec<f32> = mag_data
        .iter()
        .zip(phase_data.iter())
        .map(|(m, p)| m * p.cos())
        .collect();
    let cpu_imag: Vec<f32> = mag_data
        .iter()
        .zip(phase_data.iter())
        .map(|(m, p)| m * p.sin())
        .collect();
    let cpu_pcm = basis
        .istft(&cpu_real, &cpu_imag, n_frames, output_length)
        .unwrap();

    // Fused GPU path.
    let ctx = crate::metal_backend::global_metal_context().expect("Metal context");
    let mag_buf = ctx.create_buffer(&mag_data).expect("mag buffer");
    let phase_buf = ctx.create_buffer(&phase_data).expect("phase buffer");
    let fused_pcm = gpu_basis
        .gpu_istft_from_polar(&cache, &mag_buf, 0, &phase_buf, 0, n_frames, output_length)
        .unwrap();

    assert_eq!(cpu_pcm.len(), fused_pcm.len());
    assert_close(
        &fused_pcm,
        &cpu_pcm,
        1e-3,
        "fused_polar_istft_center_trim",
    );
}

/// Fused polar→iSTFT with miniaturized gate config (n_fft=4, center=true).
///
/// Reproduces the production failure: n_bins=3, n_frames=301, hop=1.
/// The small n_fft and large n_frames expose edge cases in the fused kernel.
#[test]
fn test_fused_polar_istft_miniaturized_n_fft4() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let n_fft: usize = 4;
    let hop: usize = n_fft / 4; // 1
    let n_bins = n_fft / 2 + 1; // 3
    let n_frames: usize = 301;
    let total = n_bins * n_frames;
    let output_length = n_frames.saturating_sub(1) * hop; // 300

    let mag_data = nn_core::test_prng::rand_f32_vec(42, total, 0.0, 5.0);
    let phase_data =
        nn_core::test_prng::rand_f32_vec(99, total, -std::f32::consts::PI, std::f32::consts::PI);

    let params = IstftParams::new(n_fft, hop, false, true).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid basis");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    // CPU reference.
    let cpu_real: Vec<f32> = mag_data
        .iter()
        .zip(phase_data.iter())
        .map(|(m, p)| m * p.cos())
        .collect();
    let cpu_imag: Vec<f32> = mag_data
        .iter()
        .zip(phase_data.iter())
        .map(|(m, p)| m * p.sin())
        .collect();
    let cpu_pcm = basis
        .istft(&cpu_real, &cpu_imag, n_frames, output_length)
        .unwrap();

    // Fused GPU path.
    let ctx = crate::metal_backend::global_metal_context().expect("Metal context");
    let mag_buf = ctx.create_buffer(&mag_data).expect("mag buffer");
    let phase_buf = ctx.create_buffer(&phase_data).expect("phase buffer");
    let fused_pcm = gpu_basis
        .gpu_istft_from_polar(&cache, &mag_buf, 0, &phase_buf, 0, n_frames, output_length)
        .unwrap();

    assert_eq!(cpu_pcm.len(), fused_pcm.len());
    assert_close(
        &fused_pcm,
        &cpu_pcm,
        1e-3,
        "fused_polar_istft_miniaturized_n_fft4",
    );
}

/// Fused polar→iSTFT with zero magnitude and NaN phase.
///
/// Reproduces the production failure: miniaturized Kokoro model (D=8, N_FFT=4)
/// produces all-zero magnitude with all-NaN phase from the generator. The fused
/// kernel must produce all-zero output (not NaN) because zero magnitude means
/// zero spectral energy regardless of phase.
///
/// Before the fix, `sincos(NaN)` propagated NaN through the IDFT sum.
#[test]
fn test_fused_polar_istft_zero_mag_nan_phase() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let n_fft: usize = 20;
    let hop: usize = 5;
    let n_bins = n_fft / 2 + 1; // 11
    let n_frames: usize = 40;
    let total = n_bins * n_frames;
    let output_length = n_frames.saturating_sub(1) * hop; // 195

    // All-zero magnitude, all-NaN phase — the exact production failure case.
    let mag_data = vec![0.0f32; total];
    let phase_data = vec![f32::NAN; total];

    let params = IstftParams::new(n_fft, hop, false, true).expect("valid params");
    let basis = IstftBasis::new(params).expect("valid basis");
    let gpu_basis = IstftGpuBasis::from_basis(&basis).expect("GPU basis upload");

    let ctx = crate::metal_backend::global_metal_context().expect("Metal context");
    let mag_buf = ctx.create_buffer(&mag_data).expect("mag buffer");
    let phase_buf = ctx.create_buffer(&phase_data).expect("phase buffer");
    let fused_pcm = gpu_basis
        .gpu_istft_from_polar(&cache, &mag_buf, 0, &phase_buf, 0, n_frames, output_length)
        .unwrap();

    assert_eq!(fused_pcm.len(), output_length);
    // All output must be exactly zero (no NaN propagation).
    for (i, &v) in fused_pcm.iter().enumerate() {
        assert!(
            v == 0.0,
            "output[{i}] = {v}, expected 0.0 (NaN propagation from zero-mag path)"
        );
    }
}
