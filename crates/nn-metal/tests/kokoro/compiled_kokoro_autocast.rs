// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Autocast (per-op F16) tests for CompiledKokoro.
//!
//! Validates that `with_autocast()` produces finite audio output with both
//! synthetic and production weights. Autocast keeps intermediate buffers F32
//! (no overflow) while using F16 for compute-dominant kernels (Linear, Conv,
//! FlashAttention) for 2x ALU throughput on Apple Silicon.
//!
//! Part of #2981, #3085.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_metal::compiled_kokoro::CompiledKokoro;

use super::kokoro_test_weights::{build_kokoro_mini, mini_test_config};

fn cpu() -> Device {
    Device::Cpu
}

/// Autocast synthesize with synthetic weights — no NaN/Inf.
///
/// Same pipeline as test_compiled_kokoro_synthesize_synthetic but with
/// `with_autocast()`. All segments compile and execute with per-op F16
/// for compute-dominant steps, F32 for everything else.
#[test]
fn test_autocast_synthesize_synthetic() {
    let (kokoro, cache) = build_kokoro_mini();
    let mut kokoro = kokoro.with_autocast();
    let cfg = mini_test_config();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(200, 2 * cfg.style_dim, -0.1, 0.1),
        &[1, 2 * cfg.style_dim],
        &cpu(),
    )
    .unwrap();

    let result = kokoro.synthesize(&input_ids, &style, 1.0, &cache);
    assert!(
        result.is_ok(),
        "autocast synthesize() failed: {:?}",
        result.err()
    );

    let (audio, _certificate) = result.unwrap();

    // Audio shape: [1, 1, T_audio].
    assert_eq!(audio.dims().len(), 3, "audio should be rank-3");
    assert!(audio.dims()[2] > 0, "audio should have non-zero samples");

    // All output samples must be finite (the core #3085 property).
    let samples = audio.to_flat_vec::<f32>().expect("read audio samples");
    let non_finite = samples.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite,
        0,
        "autocast audio has {non_finite} non-finite samples out of {}",
        samples.len()
    );
}

/// Autocast vs F32 baseline — output shapes must match.
///
/// Autocast should produce the same audio shape as F32 (segment compilation
/// produces identical step plans, only kernel dtype changes).
#[test]
fn test_autocast_vs_f32_shape_match() {
    let (kokoro_f32, cache) = build_kokoro_mini();
    let cfg = mini_test_config();

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(301, 2 * cfg.style_dim, -0.1, 0.1),
        &[1, 2 * cfg.style_dim],
        &cpu(),
    )
    .unwrap();

    // F32 baseline.
    let mut kokoro_f32 = kokoro_f32;
    let (audio_f32, _) = kokoro_f32
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("f32 synthesize");

    // Autocast — rebuild from same config to get fresh segment caches.
    let (kokoro_ac, _) = build_kokoro_mini();
    let mut kokoro_ac = kokoro_ac.with_autocast();
    let (audio_ac, _) = kokoro_ac
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("autocast synthesize");

    // Shapes must match — same model architecture, same input.
    assert_eq!(
        audio_f32.dims(),
        audio_ac.dims(),
        "autocast and F32 audio shapes differ"
    );
}

/// Production-weight autocast smoke test.
///
/// Requires KOKORO_WEIGHTS env var pointing to kokoro_v1_0.safetensors.
/// Validates no NaN/Inf in output with real weights — the core #3085 fix.
#[test]
fn test_autocast_production_weights_no_nan() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "production autocast test not enforced.",
    ) {
        Some(path) => path,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use Warn policy: test tokens [0..7] produce click artifacts with
    // production weights that fail the no_clicks hard bound. Part of #4262.
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;
    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb).expect("load Kokoro weights")
    }
    .with_autocast();

    let input_ids =
        DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();

    let result = kokoro.synthesize(&input_ids, &style, 1.0, &cache);
    // #2567 fix: sdpa guards CPU tensors, extract_gpu_slices auto-transfers.
    let (audio, _cert) = result.expect("autocast production synthesize()");
    let samples = audio.to_flat_vec::<f32>().expect("read audio");
    let non_finite = samples.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite,
        0,
        "autocast production audio has {non_finite} non-finite out of {} samples",
        samples.len()
    );
    eprintln!(
        "autocast production: {} samples, all finite, shape {:?}",
        samples.len(),
        audio.dims()
    );
}
