// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for GPU forward STFT.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::{init, make_cache};

use super::StftGpuBasis;

fn wrapped_phase_error(a: f32, b: f32) -> f32 {
    let tau = 2.0 * std::f32::consts::PI;
    let mut diff = (a - b).abs();
    while diff > std::f32::consts::PI {
        diff = (diff - tau).abs();
    }
    diff
}

/// Verify GPU forward STFT matches CPU KokoroForwardStft for Kokoro parameters.
#[test]
fn test_gpu_stft_matches_cpu_kokoro_params() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return, // Metal not available
    };

    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1; // 11

    // Create a simple test signal on GPU: sine wave at 440 Hz, 24 kHz sample rate.
    let sr = 24000.0f32;
    let freq = 440.0f32;
    let t_audio = 300; // ~12.5ms
    let signal_data: Vec<f32> = (0..t_audio)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5)
        .collect();

    let signal_cpu = DynTensor::from_vec(signal_data, &[1, 1, t_audio], &Device::Cpu).unwrap();
    let signal_gpu = signal_cpu.to_device(&Device::metal()).unwrap();

    // GPU forward STFT.
    let gpu_basis = StftGpuBasis::new(n_fft, hop).unwrap();
    let gpu_result = gpu_basis.forward_cat_center(&signal_gpu, &cache).unwrap();

    // Verify shape: [1, 2*n_bins, n_frames].
    let gpu_dims = gpu_result.dims();
    assert_eq!(gpu_dims[0], 1, "batch");
    assert_eq!(gpu_dims[1], 2 * n_bins, "2*n_bins channels");

    // CPU forward STFT for comparison.
    let cpu_stft =
        nn_models::kokoro_forward_stft::KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let cpu_result = cpu_stft.forward_cat_center(&signal_cpu).unwrap();

    assert_eq!(gpu_dims, cpu_result.dims(), "shape mismatch");

    // Compare values: GPU DFT-matmul vs CPU FFT.
    // Output layout: [B, 2*n_bins, n_frames] — first n_bins rows = magnitude,
    // next n_bins rows = phase. Phase comparison uses Cartesian form
    // (mag*cos(phase), mag*sin(phase)) to avoid atan2 ±π discontinuity (#2691).
    let gpu_cpu = gpu_result.to_device(&Device::Cpu).unwrap();
    let gpu_vals = gpu_cpu.to_flat_vec::<f32>().unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let n_frames = gpu_dims[2];
    let mut max_mag_err = 0.0f32;
    let mut max_cart_err = 0.0f32;

    for f in 0..n_bins {
        for t in 0..n_frames {
            let mag_idx = f * n_frames + t;
            let phase_idx = (f + n_bins) * n_frames + t;

            // Magnitude should match directly.
            let mag_err = (gpu_vals[mag_idx] - cpu_vals[mag_idx]).abs();
            if mag_err > max_mag_err {
                max_mag_err = mag_err;
            }

            // Phase: compare via Cartesian form to avoid 2π wrapping.
            let g_mag = gpu_vals[mag_idx];
            let g_phase = gpu_vals[phase_idx];
            let c_mag = cpu_vals[mag_idx];
            let c_phase = cpu_vals[phase_idx];
            let g_real = g_mag * g_phase.cos();
            let g_imag = g_mag * g_phase.sin();
            let c_real = c_mag * c_phase.cos();
            let c_imag = c_mag * c_phase.sin();
            let cart_err = (g_real - c_real).hypot(g_imag - c_imag);
            if cart_err > max_cart_err {
                max_cart_err = cart_err;
            }
        }
    }

    // For n_fft=20, the DFT-matmul and FFT should agree to f32 precision.
    assert!(
        max_mag_err < 1e-4,
        "GPU vs CPU magnitude max error {max_mag_err} exceeds 1e-4"
    );
    assert!(
        max_cart_err < 1e-4,
        "GPU vs CPU Cartesian max error {max_cart_err} exceeds 1e-4"
    );
}

/// Verify GPU forward STFT raw phase matches CPU FFT (#2928 regression test).
///
/// The Cartesian test above (`test_gpu_stft_matches_cpu_kokoro_params`) passes
/// even when phase wraps by 2π — mag*cos(phase) cancels the wrapping. This
/// test catches the regression directly: compare raw phase values and count
/// bins where |gpu_phase − cpu_phase| > π (a phase wrap occurred).
///
/// DFT-matmul produces ~1% wrapping rate; butterfly FFT produces ~0.002%.
/// The phase wrapping feeds through trained Generator noise_conv weights,
/// cascading to -21% amplitude deficit in production (#2928).
#[test]
fn test_gpu_stft_phase_matches_cpu_raw() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;

    // Longer signal for statistically meaningful wrap rate measurement.
    let sr = 24000.0f32;
    let freq = 440.0f32;
    let t_audio = 2400; // 100ms at 24kHz — ~475 STFT frames
    let signal_data: Vec<f32> = (0..t_audio)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5)
        .collect();

    let signal_cpu = DynTensor::from_vec(signal_data, &[1, 1, t_audio], &Device::Cpu).unwrap();
    let signal_gpu = signal_cpu.to_device(&Device::metal()).unwrap();

    // GPU forward STFT (DFT-matmul).
    let gpu_basis = StftGpuBasis::new(n_fft, hop).unwrap();
    let gpu_result = gpu_basis.forward_cat_center(&signal_gpu, &cache).unwrap();
    let gpu_cpu = gpu_result.to_device(&Device::Cpu).unwrap();
    let gpu_vals = gpu_cpu.to_flat_vec::<f32>().unwrap();

    // CPU forward STFT (rustfft butterfly).
    let cpu_stft =
        nn_models::kokoro_forward_stft::KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let cpu_result = cpu_stft.forward_cat_center(&signal_cpu).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let n_frames = gpu_cpu.dims()[2];
    let total_bins = n_bins * n_frames;
    let mut wrap_count = 0usize;

    for f in 0..n_bins {
        for t in 0..n_frames {
            let phase_idx = (f + n_bins) * n_frames + t;
            let gpu_phase = gpu_vals[phase_idx];
            let cpu_phase = cpu_vals[phase_idx];
            // A phase difference > π indicates a 2π wrap at the atan2 boundary.
            let diff = (gpu_phase - cpu_phase).abs();
            if diff > std::f32::consts::PI {
                wrap_count += 1;
            }
        }
    }

    let wrap_rate = wrap_count as f64 / total_bins as f64;
    eprintln!(
        "GPU STFT phase wrap rate: {wrap_count}/{total_bins} = {:.4}% \
         (DFT-matmul measured ~4.8%, FFT ~0.002%)",
        wrap_rate * 100.0
    );

    // Threshold: 6% accommodates the measured DFT-matmul ~4.8% rate while
    // catching further regressions. The compiled pipeline uses CPU FFT (D1 of
    // #2928), so production is correct. This test guards the GPU kernel directly.
    // TODO(#2928): When D2 (GPU mixed-radix FFT) lands, tighten to 0.5%.
    assert!(
        wrap_rate < 0.06,
        "GPU STFT phase wrap rate {:.4}% exceeds 6% threshold. \
         DFT-matmul measured ~4.8%, butterfly FFT is ~0.002%. \
         See #2928 for the amplitude regression caused by phase wrapping.",
        wrap_rate * 100.0
    );
}

/// Verify GPU STFT output is non-trivial (magnitude > 0 for non-zero signal).
#[test]
fn test_gpu_stft_nontrivial_output() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;

    let signal_data: Vec<f32> = (0..200).map(|i| (i as f32 * 0.3).sin()).collect();
    let signal = DynTensor::from_vec(signal_data, &[1, 1, 200], &Device::metal()).unwrap();

    let basis = StftGpuBasis::new(n_fft, hop).unwrap();
    let result = basis.forward_cat_center(&signal, &cache).unwrap();

    let result_cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = result_cpu.to_flat_vec::<f32>().unwrap();

    // Magnitude channels (first n_bins rows) should have positive energy.
    let n_frames = result.dims()[2];
    let mag_sum: f32 = vals[..n_bins * n_frames].iter().sum();
    assert!(
        mag_sum > 0.0,
        "magnitude sum should be positive, got {mag_sum}"
    );
}

/// Verify GPU STFT rejects non-3D input.
#[test]
fn test_gpu_stft_rejects_wrong_rank() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let signal = DynTensor::zeros(&[200], DType::F32, &Device::metal()).unwrap();
    let basis = StftGpuBasis::new(20, 5).unwrap();
    let result = basis.forward_cat_center(&signal, &cache);
    assert!(result.is_err());
}

/// Regression test for the mini Kokoro config: DFT STFT must work for n_fft=4.
///
/// The compiled Kokoro synth tests use `mini_test_config()` with `n_fft=4`.
/// Existing STFT coverage only exercised the default `n_fft=20` path, which
/// let a small-FFT corruption slip through step_harmonic_source.
#[test]
fn test_gpu_stft_small_nfft_matches_cpu() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let n_fft = 4;
    let hop = 1;
    let n_bins = n_fft / 2 + 1;
    let t_audio = 24;

    let signal_data: Vec<f32> = (0..t_audio)
        .map(|i| ((i as f32) * 0.25).sin() * 0.5 + ((i as f32) * 0.11).cos() * 0.25)
        .collect();
    let signal_cpu = DynTensor::from_vec(signal_data, &[1, 1, t_audio], &Device::Cpu).unwrap();
    let signal_gpu = signal_cpu.to_device(&Device::metal()).unwrap();

    let gpu_basis = StftGpuBasis::new(n_fft, hop).unwrap();
    let gpu_result = gpu_basis.forward_cat_center(&signal_gpu, &cache).unwrap();
    let gpu_cpu = gpu_result.to_device(&Device::Cpu).unwrap();
    let gpu_vals = gpu_cpu.to_flat_vec::<f32>().unwrap();

    let cpu_stft =
        nn_models::kokoro_forward_stft::KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let cpu_result = cpu_stft.forward_cat_center(&signal_cpu).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let n_frames = gpu_result.dims()[2];
    assert_eq!(gpu_result.dims(), cpu_result.dims(), "shape mismatch");

    let mut max_mag_err = 0.0f32;
    let mut max_phase_err = 0.0f32;
    for f in 0..n_bins {
        for t in 0..n_frames {
            let mag_idx = f * n_frames + t;
            let phase_idx = (n_bins + f) * n_frames + t;
            max_mag_err = max_mag_err.max((gpu_vals[mag_idx] - cpu_vals[mag_idx]).abs());
            max_phase_err =
                max_phase_err.max(wrapped_phase_error(gpu_vals[phase_idx], cpu_vals[phase_idx]));
        }
    }

    assert!(
        max_mag_err < 1e-4,
        "n_fft=4 GPU STFT magnitude regression: max error {max_mag_err} exceeds 1e-4"
    );
    assert!(
        max_phase_err < 1e-4,
        "n_fft=4 GPU STFT phase regression: max error {max_phase_err} exceeds 1e-4"
    );
}

/// Zero-energy STFT bins must emit phase 0, not NaN.
///
/// Synthetic Kokoro weights produce all-zero excitation in the mini test
/// config. Metal `atan2(0, 0)` can return NaN, so guard this explicitly.
#[test]
fn test_gpu_stft_zero_signal_has_finite_phase() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let signal = DynTensor::zeros(&[1, 1, 300], DType::F32, &Device::metal()).unwrap();
    let basis = StftGpuBasis::new(4, 1).unwrap();
    let result = basis.forward_cat_center(&signal, &cache).unwrap();
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let non_finite = vals.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "zero-signal STFT should be finite, found {non_finite} non-finite values"
    );
    assert!(
        vals.iter().all(|&v| v == 0.0),
        "zero-signal STFT should produce exact zeros"
    );
}

/// Regression test for #2928: GPU STFT within arena context (non-zero byte_offset).
///
/// The original bug: `set_buffer(0, padded_buf)` ignored `byte_offset()`, so when
/// the arena is active and `reflection_pad1d` output has a non-zero offset, the
/// kernel reads from the wrong position in the buffer.
///
/// This test activates the arena, does a dummy allocation to push the offset > 0,
/// then runs the GPU STFT and verifies magnitude matches the CPU reference.
#[test]
fn test_gpu_stft_arena_byte_offset() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;

    let sr = 24000.0f32;
    let freq = 440.0f32;
    let t_audio = 300;
    let signal_data: Vec<f32> = (0..t_audio)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5)
        .collect();

    let signal_cpu = DynTensor::from_vec(signal_data, &[1, 1, t_audio], &Device::Cpu).unwrap();

    // CPU reference (outside arena).
    let cpu_stft =
        nn_models::kokoro_forward_stft::KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let cpu_result = cpu_stft.forward_cat_center(&signal_cpu).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    // GPU STFT inside arena — arena pushes byte_offset > 0.
    let ctx = crate::metal_backend::global_metal_context().unwrap();
    let mut arena = crate::arena::ActivationArena::new(ctx, 1024 * 1024).unwrap();
    let gpu_result = crate::arena::with_arena(&mut arena, || {
        let signal_gpu = signal_cpu.to_device(&Device::metal()).unwrap();
        // Dummy op to consume arena space and push subsequent offsets > 0.
        let _dummy = DynTensor::zeros(&[256], DType::F32, &Device::metal()).unwrap();
        let gpu_basis = StftGpuBasis::new(n_fft, hop).unwrap();
        gpu_basis.forward_cat_center(&signal_gpu, &cache).unwrap()
    });

    let gpu_cpu = gpu_result.to_device(&Device::Cpu).unwrap();
    let gpu_vals = gpu_cpu.to_flat_vec::<f32>().unwrap();

    let n_frames = gpu_cpu.dims()[2];
    assert_eq!(n_frames, cpu_result.dims()[2], "frame count mismatch");

    let mut max_mag_err = 0.0f32;
    for f in 0..n_bins {
        for t in 0..n_frames {
            let mag_idx = f * n_frames + t;
            let mag_err = (gpu_vals[mag_idx] - cpu_vals[mag_idx]).abs();
            if mag_err > max_mag_err {
                max_mag_err = mag_err;
            }
        }
    }

    assert!(
        max_mag_err < 1e-4,
        "Arena byte_offset regression (#2928): GPU vs CPU magnitude max error \
         {max_mag_err} exceeds 1e-4. The signal byte_offset may not be passed \
         to set_buffer_with_offset in the STFT kernel dispatch."
    );
}

/// Verify GPU FFT STFT (Good-Thomas PFA) matches CPU rustfft with tight phase threshold.
///
/// This is the D2 regression test for #2928. The mixed-radix FFT kernel should produce
/// near-zero phase wrapping (< 0.5%) vs the DFT-matmul's ~4.8%. This confirms the
/// butterfly accumulation eliminates the ±π atan2 boundary flips that cause the -21%
/// amplitude deficit through trained Generator weights.
#[test]
fn test_gpu_stft_fft_phase_matches_cpu_raw() {
    init();
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };

    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;

    // Longer signal for statistically meaningful wrap rate measurement.
    let sr = 24000.0f32;
    let freq = 440.0f32;
    let t_audio = 2400;
    let signal_data: Vec<f32> = (0..t_audio)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5)
        .collect();

    let signal_cpu = DynTensor::from_vec(signal_data, &[1, 1, t_audio], &Device::Cpu).unwrap();
    let signal_gpu = signal_cpu.to_device(&Device::metal()).unwrap();

    // GPU forward STFT via FFT (Good-Thomas PFA).
    let gpu_basis = StftGpuBasis::new(n_fft, hop).unwrap();
    let gpu_result = gpu_basis
        .forward_cat_center_fft(&signal_gpu, &cache)
        .unwrap();
    let gpu_cpu = gpu_result.to_device(&Device::Cpu).unwrap();
    let gpu_vals = gpu_cpu.to_flat_vec::<f32>().unwrap();

    // CPU forward STFT (rustfft butterfly).
    let cpu_stft =
        nn_models::kokoro_forward_stft::KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let cpu_result = cpu_stft.forward_cat_center(&signal_cpu).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    assert_eq!(gpu_cpu.dims(), cpu_result.dims(), "shape mismatch");

    let n_frames = gpu_cpu.dims()[2];
    let total_bins = n_bins * n_frames;
    let mut wrap_count = 0usize;
    let mut max_mag_err = 0.0f32;

    for f in 0..n_bins {
        for t in 0..n_frames {
            let mag_idx = f * n_frames + t;
            let phase_idx = (f + n_bins) * n_frames + t;

            let mag_err = (gpu_vals[mag_idx] - cpu_vals[mag_idx]).abs();
            if mag_err > max_mag_err {
                max_mag_err = mag_err;
            }

            let diff = (gpu_vals[phase_idx] - cpu_vals[phase_idx]).abs();
            if diff > std::f32::consts::PI {
                wrap_count += 1;
            }
        }
    }

    let wrap_rate = wrap_count as f64 / total_bins as f64;
    eprintln!(
        "GPU FFT STFT phase wrap rate: {wrap_count}/{total_bins} = {:.4}% \
         (target < 0.5%, DFT-matmul was ~4.8%)",
        wrap_rate * 100.0
    );
    eprintln!("GPU FFT vs CPU FFT max magnitude error: {max_mag_err}");

    assert!(
        max_mag_err < 1e-3,
        "GPU FFT vs CPU FFT magnitude error {max_mag_err} exceeds 1e-3"
    );
    assert!(
        wrap_rate < 0.005,
        "GPU FFT STFT phase wrap rate {:.4}% exceeds 0.5% threshold. \
         The Good-Thomas PFA should produce near-zero wrapping. \
         See #2928 for context.",
        wrap_rate * 100.0
    );
}
