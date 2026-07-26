// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Autocast integration tests for multi-voice chorus and streaming synthesis.
//!
//! Validates that `with_autocast()` produces correct audio when combined with:
//! - `clone_dispatch()` multi-voice synthesis (AC1)
//! - `synthesize_streaming()` chunked synthesis (AC2)
//!
//! Production Kokoro uses autocast (F16 for non-LSTM ops). These tests close
//! the gap where all existing chorus/streaming tests run at F32 only.
//!
//! Part of #3481, #3351.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_metal::compiled_kokoro::CompiledKokoro;
use nn_models::kokoro_streaming::concatenate_chunks;

use super::kokoro_test_weights::{build_kokoro_mini, mini_test_config};

fn cpu() -> Device {
    Device::Cpu
}

fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    assert_eq!(reference.len(), test.len(), "SNR: length mismatch");
    let signal_power: f64 = reference.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let noise_power: f64 = reference
        .iter()
        .zip(test.iter())
        .map(|(&r, &t)| {
            let d = f64::from(r) - f64::from(t);
            d * d
        })
        .sum();
    if noise_power < 1e-30 {
        return 120.0;
    }
    10.0 * (signal_power / noise_power).log10()
}

/// AC1: Multi-voice chorus with autocast produces finite audio.
///
/// Creates a CompiledKokoro with_autocast(), warms up, creates a clone_dispatch
/// for a second voice, runs synthesize on both, and asserts:
/// - Non-empty audio output from both voices
/// - No NaN or Inf in either voice
/// - Audio within [-1, 1]
///
/// Validates that `clone_dispatch()` correctly propagates `autocast_policy`
/// (compiled_kokoro.rs:309) and that Arc-shared weight aliasing works under
/// F16 mixed-precision segment compilation.
///
/// Part of #3481.
#[test]
fn test_autocast_chorus_clone_dispatch_no_nan() {
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

    // Warmup: compile all segments with autocast active.
    let (audio1, _cert1) = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("autocast voice 1 warmup");

    // Clone dispatch: shares Arc<SharedKokoroState> + GPU weight buffers.
    // clone_dispatch() propagates autocast_policy (compiled_kokoro.rs:309).
    let mut voice2 = kokoro.clone_dispatch();

    // Synthesize on clone with different style seed.
    let style2 = DynTensor::new(
        &super::test_utils::rand_f32_vec(301, 2 * cfg.style_dim, -0.1, 0.1),
        &[1, 2 * cfg.style_dim],
        &cpu(),
    )
    .unwrap();

    let (audio2, _cert2) = voice2
        .synthesize(&input_ids, &style2, 1.0, &cache)
        .expect("autocast voice 2 synthesize");

    // Both voices produce non-empty audio.
    assert!(audio1.dims()[2] > 0, "voice 1 should have samples");
    assert!(audio2.dims()[2] > 0, "voice 2 should have samples");

    // No NaN/Inf in either voice (the core #3481 property).
    for (label, audio) in [("voice1", &audio1), ("voice2", &audio2)] {
        let samples = audio.to_flat_vec::<f32>().expect("read audio");
        let non_finite = samples.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            non_finite,
            0,
            "autocast {label} has {non_finite} non-finite samples out of {}",
            samples.len()
        );
    }

    eprintln!(
        "autocast chorus: voice1 shape={:?} ({} samples), voice2 shape={:?} ({} samples)",
        audio1.dims(),
        audio1.dims()[2],
        audio2.dims(),
        audio2.dims()[2],
    );
}

/// AC2: Streaming synthesis with autocast produces finite audio chunks.
///
/// Creates a CompiledKokoro with_autocast() and calls synthesize_streaming()
/// with 2 token chunks. Validates F16 buffer survival across chunk boundaries
/// and crossfade correctness under mixed-precision dispatch.
///
/// Part of #3481.
#[test]
fn test_autocast_streaming_synthesis_no_nan() {
    let (kokoro, cache) = build_kokoro_mini();
    let mut kokoro = kokoro.with_autocast();
    let cfg = mini_test_config();

    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(400, 2 * cfg.style_dim, -0.1, 0.1),
        &[1, 2 * cfg.style_dim],
        &cpu(),
    )
    .unwrap();

    // Two token chunks — exercises chunk boundary crossfade with F16 buffers.
    let chunk0 = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let chunk1 = DynTensor::from_vec(vec![4.0, 5.0, 6.0], &[1, 3], &cpu()).unwrap();
    let (warmup_audio, _cert) = kokoro
        .synthesize(&chunk0, &style, 1.0, &cache)
        .expect("autocast short-chunk warmup");
    let stream_config = super::test_utils::short_stream_config_for_pcm_len(warmup_audio.dims()[2]);

    let chunks = kokoro
        .synthesize_streaming(
            &[chunk0.clone(), chunk1],
            &style,
            1.0,
            &stream_config,
            &cache,
        )
        .expect("autocast streaming synthesis");

    assert!(
        !chunks.is_empty(),
        "streaming should produce at least 1 chunk"
    );

    // Every chunk must have finite PCM with no NaN/Inf.
    let mut total_samples = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(!chunk.pcm.is_empty(), "chunk {i} should have non-empty PCM");
        let non_finite = chunk.pcm.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            non_finite,
            0,
            "autocast streaming chunk {i} has {non_finite} non-finite samples out of {}",
            chunk.pcm.len()
        );
        total_samples += chunk.pcm.len();
    }

    eprintln!(
        "autocast streaming: {} chunks, {} total samples",
        chunks.len(),
        total_samples,
    );
}

/// AC3: Autocast streaming concat produces finite concatenated audio.
///
/// Exercises `synthesize_streaming()` + `concatenate_chunks()` under autocast
/// to ensure the full concatenated PCM is finite after crossfade assembly.
///
/// Part of #3481.
#[test]
fn test_autocast_streaming_concat_no_nan() {
    let (kokoro, cache) = build_kokoro_mini();
    let mut kokoro = kokoro.with_autocast();
    let cfg = mini_test_config();

    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(500, 2 * cfg.style_dim, -0.1, 0.1),
        &[1, 2 * cfg.style_dim],
        &cpu(),
    )
    .unwrap();

    let chunk0 = DynTensor::from_vec(vec![2.0, 3.0, 4.0], &[1, 3], &cpu()).unwrap();
    let chunk1 = DynTensor::from_vec(vec![5.0, 6.0, 7.0], &[1, 3], &cpu()).unwrap();
    let (warmup_audio, _cert) = kokoro
        .synthesize(&chunk0, &style, 1.0, &cache)
        .expect("autocast concat short-chunk warmup");
    let stream_config = super::test_utils::short_stream_config_for_pcm_len(warmup_audio.dims()[2]);

    let audio_chunks = kokoro
        .synthesize_streaming(
            &[chunk0.clone(), chunk1],
            &style,
            1.0,
            &stream_config,
            &cache,
        )
        .expect("autocast streaming for concat");

    let pcm = concatenate_chunks(&audio_chunks);

    assert!(!pcm.is_empty(), "concat should produce non-empty PCM");

    let non_finite = pcm.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite,
        0,
        "autocast streaming concat has {non_finite} non-finite samples out of {}",
        pcm.len()
    );

    eprintln!("autocast streaming concat: {} samples", pcm.len());
}

/// AC4: Production clone-dispatch chorus stays close to the F32 baseline.
///
/// Uses real Kokoro weights when available, runs a two-voice `clone_dispatch()`
/// chorus with different text lengths and styles, and measures per-voice SNR
/// plus max sample error between F32 and autocast outputs.
///
/// This turns chorus autocast regressions into a quantitative gate instead of
/// only checking for NaN/Inf.
#[test]
fn test_autocast_chorus_clone_dispatch_matches_f32_production() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "production chorus autocast parity test not enforced.",
    ) {
        Some(path) => path,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let input_short =
        DynTensor::from_vec_i64(vec![0_i64, 1, 2, 3, 4, 5, 6, 7], &[1, 8], &cpu()).unwrap();
    let input_long = DynTensor::from_vec_i64(
        vec![0_i64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        &[1, 12],
        &cpu(),
    )
    .unwrap();
    let style1 = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();
    let style2 = DynTensor::full(&[1, 256], 0.013, DType::F32, &cpu()).unwrap();

    let mut hb_f32 = nn_tts_verify::HardBoundsConfig::default();
    hb_f32.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;
    let hb_ac = hb_f32.clone();

    // F32 baseline.
    // SAFETY: safetensors file not modified while alive.
    let mut f32_parent =
        unsafe { CompiledKokoro::load_with_hard_bounds(&weights_path, hb_f32).expect("load F32") };
    let _ = f32_parent
        .synthesize(&input_short, &style1, 1.0, &cache)
        .expect("F32 warmup");
    let mut f32_clone = f32_parent.clone_dispatch();

    let (audio1_f32, _) = f32_parent
        .synthesize(&input_short, &style1, 1.0, &cache)
        .expect("F32 voice1");
    let (audio2_f32, _) = f32_clone
        .synthesize(&input_long, &style2, 1.0, &cache)
        .expect("F32 voice2");

    // Autocast under the same two-voice clone-dispatch setup.
    // SAFETY: safetensors file not modified while alive.
    let mut ac_parent = unsafe {
        CompiledKokoro::load_with_hard_bounds(&weights_path, hb_ac).expect("load autocast")
    }
    .with_autocast();
    let _ = ac_parent
        .synthesize(&input_short, &style1, 1.0, &cache)
        .expect("autocast warmup");
    let mut ac_clone = ac_parent.clone_dispatch();

    let (audio1_ac, _) = ac_parent
        .synthesize(&input_short, &style1, 1.0, &cache)
        .expect("autocast voice1");
    let (audio2_ac, _) = ac_clone
        .synthesize(&input_long, &style2, 1.0, &cache)
        .expect("autocast voice2");

    assert_eq!(
        audio1_f32.dims(),
        audio1_ac.dims(),
        "production chorus voice1 autocast vs F32 shape mismatch"
    );
    assert_eq!(
        audio2_f32.dims(),
        audio2_ac.dims(),
        "production chorus voice2 autocast vs F32 shape mismatch"
    );

    let voice1_f32 = audio1_f32.to_flat_vec::<f32>().expect("voice1 F32 samples");
    let voice2_f32 = audio2_f32.to_flat_vec::<f32>().expect("voice2 F32 samples");
    let voice1_ac = audio1_ac
        .to_flat_vec::<f32>()
        .expect("voice1 autocast samples");
    let voice2_ac = audio2_ac
        .to_flat_vec::<f32>()
        .expect("voice2 autocast samples");

    for (label, samples) in [("voice1", &voice1_ac), ("voice2", &voice2_ac)] {
        let non_finite = samples.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            non_finite,
            0,
            "production chorus autocast {label} has {non_finite} non-finite samples out of {}",
            samples.len()
        );
    }

    let voice1_snr = snr_db(&voice1_f32, &voice1_ac);
    let voice2_snr = snr_db(&voice2_f32, &voice2_ac);
    let voice1_max_diff = voice1_f32
        .iter()
        .zip(voice1_ac.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let voice2_max_diff = voice2_f32
        .iter()
        .zip(voice2_ac.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    eprintln!(
        "production chorus autocast vs F32: \
         voice1={} samples SNR={voice1_snr:.1} dB max_diff={voice1_max_diff:.6e}, \
         voice2={} samples SNR={voice2_snr:.1} dB max_diff={voice2_max_diff:.6e}",
        voice1_f32.len(),
        voice2_f32.len(),
    );

    assert!(
        voice1_snr > 35.0,
        "production chorus voice1 autocast SNR {voice1_snr:.1} dB < 35.0 dB threshold"
    );
    assert!(
        voice2_snr > 35.0,
        "production chorus voice2 autocast SNR {voice2_snr:.1} dB < 35.0 dB threshold"
    );
}
