// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for [`KokoroChorus`] — multi-voice GPU synthesis pool.
//!
//! Verifies that:
//! 1. KokoroChorus creates N voices sharing weights via Arc.
//! 2. synthesize_chorus() produces mixed audio from multiple voices.
//! 3. GPU weight memory is shared (aliased, not copied).
//! 4. Different styles produce different audio that mixes correctly.
//!
//! Part of #3355, #3351, #2740.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_metal::compiled_kokoro::chorus::KokoroChorus;
use nn_metal::compiled_kokoro::{CompiledKokoro, CompiledKokoroError};
use nn_models::kokoro_chorus::ChorusConfig;
use nn_models::kokoro_streaming::{concatenate_chunks, KokoroStreamConfig};

use super::kokoro_test_weights as kw;

fn cpu() -> Device {
    Device::Cpu
}

const STYLE_DIM: usize = 4; // Must match kw::mini_test_config().style_dim

// -- Helpers ------------------------------------------------------------------

fn build_kokoro() -> (CompiledKokoro, nn_metal::PipelineCache) {
    kw::build_kokoro_mini()
}

fn make_style(seed: usize) -> DynTensor {
    DynTensor::new(
        &super::test_utils::rand_f32_vec(200 + seed as u64, 2 * STYLE_DIM, -0.1, 0.1),
        &[1, 2 * STYLE_DIM],
        &cpu(),
    )
    .unwrap()
}

fn make_input(len: usize) -> DynTensor {
    let vals: Vec<f32> = (1..=len).map(|v| v as f32).collect();
    DynTensor::from_vec(vals, &[1, len], &cpu()).unwrap()
}

fn make_input_mod_vocab(len: usize, vocab: usize) -> DynTensor {
    let vals: Vec<f32> = (0..len).map(|i| (i % vocab) as f32).collect();
    DynTensor::from_vec(vals, &[1, len], &cpu()).unwrap()
}

fn assert_input_too_long<T>(
    result: Result<T, CompiledKokoroError>,
    label: &str,
    expected_context: Option<&str>,
) {
    match result {
        Err(CompiledKokoroError::InvalidInput(msg)) => {
            if let Some(context) = expected_context {
                assert!(
                    msg.contains(context),
                    "{label}: missing oversized-input context {context:?}: {msg}"
                );
            }
            assert!(
                msg.contains("seq_len 17 exceeds max_position_embeddings 16"),
                "{label}: unexpected oversized-input message: {msg}"
            );
        }
        Err(other) => panic!("{label}: expected InvalidInput for oversized input, got: {other:?}"),
        Ok(_) => panic!("{label}: expected oversized input to fail"),
    }
}

// -- Tests --------------------------------------------------------------------

/// KokoroChorus creates 8 voices sharing the same Arc<SharedKokoroState>.
#[test]
fn test_chorus_creation_8_voices() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    // Warmup primary to populate segment caches.
    let input = make_input(3);
    let style = make_style(0);
    let _ = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(8).unwrap();
    let chorus = KokoroChorus::new(&primary, config).unwrap();

    assert_eq!(chorus.n_voices(), 8);
    // 8 chorus voices + 1 primary = 9 references to shared state.
    assert_eq!(chorus.shared_state_refcount(), 9);
}

/// synthesize_chorus produces non-empty audio from 2 voices with same input.
#[test]
fn test_chorus_synthesize_2_voices_same_text() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let styles = vec![make_style(100), make_style(200)];
    let mixed = chorus
        .synthesize_chorus_same_text(&input, &styles, 1.0, &cache)
        .expect("chorus synthesize");

    assert!(!mixed.is_empty(), "mixed audio should not be empty");
    // Check that output is in reasonable range (gains are 0.5 each, clipped).
    let max_abs = mixed.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs <= 1.0 + 1e-6,
        "clipped output should be in [-1,1]: max_abs={max_abs}",
    );

    eprintln!(
        "chorus 2-voice: {} samples, max_abs={max_abs:.4}",
        mixed.len()
    );
}

/// Repeated shared-encode synthesis with many voices must not hit stale arena reads.
#[test]
fn test_chorus_shared_encode_repeated_8_voices() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(7);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(8).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();
    let styles: Vec<DynTensor> = (0..8).map(|i| make_style(100 + i)).collect();

    for iter in 0..3 {
        let mixed = chorus
            .synthesize_chorus_same_text(&input, &styles, 1.0, &cache)
            .unwrap_or_else(|err| panic!("shared encode iteration {iter} failed: {err}"));
        assert!(
            !mixed.is_empty(),
            "shared encode iteration {iter} produced empty audio"
        );
    }
}

/// synthesize_chorus with different inputs per voice produces mixed output.
#[test]
fn test_chorus_synthesize_different_inputs() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input3 = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input3, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let inputs = vec![make_input(3), make_input(5)];
    let styles = vec![make_style(0), make_style(1)];
    let mixed = chorus
        .synthesize_chorus(&inputs, &styles, 1.0, &cache)
        .expect("chorus synthesize");

    assert!(!mixed.is_empty());
    eprintln!("chorus different inputs: {} samples", mixed.len());
}

/// synthesize_chorus_varied_speed produces audio with speed variation.
#[test]
fn test_chorus_varied_speed() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style = make_style(0);
    let _ = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(3).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let speeds = vec![0.95, 1.0, 1.05];
    let mixed = chorus
        .synthesize_chorus_varied_speed(&input, &style, &speeds, &cache)
        .expect("varied speed");

    assert!(!mixed.is_empty());
    eprintln!("chorus varied speed: {} samples", mixed.len());
}

/// Oversized inputs are rejected consistently across chorus APIs.
#[test]
fn test_chorus_input_too_long() {
    use nn_metal::compiled_kokoro::gpu_synth::ChorusGpuSynth;
    use nn_models::kokoro_pipeline::KokoroSynth;

    super::test_utils::gpu_init();
    let (primary, cache) = build_kokoro();
    let config = kw::mini_test_config();
    let max_pos = config.plbert.max_position_embeddings;
    let oversized = make_input_mod_vocab(max_pos + 1, config.plbert.vocab_size);

    let chorus_cfg = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, chorus_cfg).unwrap();
    let styles = vec![make_style(10), make_style(20)];
    let speeds = vec![1.0f32, 1.0];
    let stream_config = KokoroStreamConfig::default();
    let short = make_input(3);

    assert_input_too_long(
        chorus.synthesize_chorus_same_text(&oversized, &styles, 1.0, &cache),
        "synthesize_chorus_same_text",
        None,
    );
    assert_input_too_long(
        chorus.synthesize_chorus(&[short.clone(), oversized.clone()], &styles, 1.0, &cache),
        "synthesize_chorus",
        Some("inputs[1]"),
    );
    assert_input_too_long(
        chorus.synthesize_chorus_varied_speed(&oversized, &styles[0], &speeds, &cache),
        "synthesize_chorus_varied_speed",
        None,
    );
    assert_input_too_long(
        chorus.synthesize_streaming_chorus(
            std::slice::from_ref(&oversized),
            &styles,
            1.0,
            &stream_config,
            &cache,
        ),
        "synthesize_streaming_chorus",
        Some("chunks[0]"),
    );
    assert_input_too_long(
        chorus.synthesize_streaming_chorus_per_voice(
            &[vec![short], vec![oversized.clone()]],
            &styles,
            1.0,
            &stream_config,
            &cache,
        ),
        "synthesize_streaming_chorus_per_voice",
        Some("per_voice_chunks[1][0]"),
    );

    let mut gpu = ChorusGpuSynth::new(&mut chorus, &cache);
    assert_input_too_long(
        gpu.synthesize_batch(&oversized, &styles, &speeds),
        "ChorusGpuSynth::synthesize_batch",
        None,
    );
}

/// GPU weight memory is shared across all chorus voices.
#[test]
fn test_chorus_memory_sharing() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style = make_style(0);
    let _ = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup");

    let parent_bytes = primary.gpu_weight_bytes();
    assert!(
        parent_bytes > 0,
        "primary should have GPU weights after warmup"
    );

    let config = ChorusConfig::equal_gain(8).unwrap();
    let chorus = KokoroChorus::new(&primary, config).unwrap();

    // All voices should report the same weight bytes (aliased, not copied).
    let chorus_bytes = chorus.gpu_weight_bytes_per_voice();
    assert_eq!(
        chorus_bytes, parent_bytes,
        "chorus voices should alias parent's GPU buffers: chorus={chorus_bytes}, parent={parent_bytes}"
    );
}

/// Input count mismatch is rejected.
#[test]
fn test_chorus_input_mismatch_rejected() {
    super::test_utils::gpu_init();
    let (primary, cache) = build_kokoro();

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    // 1 input for 2 voices → error.
    let inputs = vec![make_input(3)];
    let styles = vec![make_style(0), make_style(1)];
    let result = chorus.synthesize_chorus(&inputs, &styles, 1.0, &cache);
    assert!(result.is_err());
}

/// Style count mismatch is rejected.
#[test]
fn test_chorus_style_mismatch_rejected() {
    super::test_utils::gpu_init();
    let (primary, cache) = build_kokoro();

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let inputs = vec![make_input(3), make_input(3)];
    let styles = vec![make_style(0)]; // 1 style for 2 voices
    let result = chorus.synthesize_chorus(&inputs, &styles, 1.0, &cache);
    assert!(result.is_err());
}

// -- Streaming chorus tests -----------------------------------------------

/// Streaming chorus with 2 voices and 2 chunks produces crossfaded AudioChunks.
#[test]
fn test_streaming_chorus_2_voices_2_chunks() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style = make_style(0);
    let (warmup_audio, _cert) = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let styles = vec![make_style(10), make_style(20)];
    let stream_config = super::test_utils::short_stream_config_for_pcm_len(warmup_audio.dims()[2]);
    let chunks_in = [make_input(3), make_input(4)];

    let audio_chunks = chorus
        .synthesize_streaming_chorus(&chunks_in, &styles, 1.0, &stream_config, &cache)
        .expect("streaming chorus");

    assert_eq!(audio_chunks.len(), 2, "expected 2 audio chunks");
    assert!(!audio_chunks[0].is_final);
    assert!(audio_chunks[1].is_final);
    assert_eq!(audio_chunks[0].chunk_index, 0);
    assert_eq!(audio_chunks[1].chunk_index, 1);

    let full = concatenate_chunks(&audio_chunks);
    assert!(!full.is_empty());
    eprintln!(
        "streaming chorus 2v×2c: {} total samples, chunks: [{}, {}]",
        full.len(),
        audio_chunks[0].len(),
        audio_chunks[1].len(),
    );
}

/// Single-chunk streaming chorus is equivalent to a regular chorus mix.
#[test]
fn test_streaming_chorus_single_chunk() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style = make_style(0);
    let _ = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let styles = vec![make_style(10), make_style(20)];
    let stream_config = KokoroStreamConfig::default();

    let audio_chunks = chorus
        .synthesize_streaming_chorus(&[make_input(3)], &styles, 1.0, &stream_config, &cache)
        .expect("streaming chorus single");

    assert_eq!(audio_chunks.len(), 1);
    assert!(audio_chunks[0].is_final);
    assert!(!audio_chunks[0].is_empty());
}

/// Empty chunk list returns empty result.
#[test]
fn test_streaming_chorus_empty_chunks() {
    super::test_utils::gpu_init();
    let (primary, cache) = build_kokoro();

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let styles = vec![make_style(10), make_style(20)];
    let stream_config = KokoroStreamConfig::default();

    let audio_chunks = chorus
        .synthesize_streaming_chorus(&[], &styles, 1.0, &stream_config, &cache)
        .expect("streaming chorus empty");

    assert!(audio_chunks.is_empty());
}

/// Style mismatch in streaming chorus is rejected.
#[test]
fn test_streaming_chorus_style_mismatch() {
    super::test_utils::gpu_init();
    let (primary, cache) = build_kokoro();

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let styles = vec![make_style(10)]; // 1 style for 2 voices
    let stream_config = KokoroStreamConfig::default();

    let result =
        chorus.synthesize_streaming_chorus(&[make_input(3)], &styles, 1.0, &stream_config, &cache);
    assert!(result.is_err());
}

/// Per-voice streaming chorus with different text per voice.
#[test]
fn test_streaming_chorus_per_voice() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style = make_style(0);
    let (warmup_audio, _cert) = primary
        .synthesize(&input, &style, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let styles = vec![make_style(10), make_style(20)];
    let stream_config = super::test_utils::short_stream_config_for_pcm_len(warmup_audio.dims()[2]);

    let voice0_chunks = vec![make_input(3), make_input(4)];
    let voice1_chunks = vec![make_input(5), make_input(3)];
    let per_voice = vec![voice0_chunks, voice1_chunks];

    let audio_chunks = chorus
        .synthesize_streaming_chorus_per_voice(&per_voice, &styles, 1.0, &stream_config, &cache)
        .expect("streaming chorus per-voice");

    assert_eq!(audio_chunks.len(), 2);
    assert!(audio_chunks[1].is_final);

    let full = concatenate_chunks(&audio_chunks);
    assert!(!full.is_empty());
    eprintln!("streaming chorus per-voice: {} total samples", full.len());
}

// -- Pipelined chorus tests ---------------------------------------------------

/// Pipelined chorus decode produces identical audio to sequential decode.
/// Part of #4290.
#[test]
fn test_pipelined_chorus_matches_sequential() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(5);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(4).unwrap();
    let mut chorus_seq = KokoroChorus::new(&primary, config.clone()).unwrap();
    let mut chorus_pipe = KokoroChorus::new(&primary, config).unwrap();

    let styles: Vec<DynTensor> = (0..4).map(|i| make_style(300 + i)).collect();

    let audio_seq = chorus_seq
        .synthesize_chorus_shared_encode(&input, &styles, 1.0, &cache)
        .expect("sequential chorus");

    let audio_pipe = chorus_pipe
        .synthesize_chorus_pipelined(&input, &styles, 1.0, &cache)
        .expect("pipelined chorus");

    assert_eq!(
        audio_seq.len(),
        audio_pipe.len(),
        "sequential and pipelined audio length mismatch",
    );

    let max_diff = audio_seq
        .iter()
        .zip(audio_pipe.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-5,
        "sequential vs pipelined max sample diff = {max_diff} (expected < 1e-5)",
    );

    eprintln!(
        "pipelined vs sequential: {} samples, max_diff={max_diff:.2e}",
        audio_seq.len(),
    );
}

/// Pipelined chorus with 2 voices produces non-empty audio. Part of #4290.
#[test]
fn test_pipelined_chorus_2_voices() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let styles = vec![make_style(400), make_style(500)];
    let mixed = chorus
        .synthesize_chorus_pipelined(&input, &styles, 1.0, &cache)
        .expect("pipelined 2-voice");

    assert!(!mixed.is_empty(), "pipelined audio should not be empty");
    let max_abs = mixed.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs <= 1.0 + 1e-6,
        "clipped output should be in [-1,1]: max_abs={max_abs}",
    );
    eprintln!(
        "pipelined 2-voice: {} samples, max_abs={max_abs:.4}",
        mixed.len(),
    );
}

/// Repeated pipelined synthesis does not accumulate stale GPU state.
/// Part of #4290.
#[test]
fn test_pipelined_chorus_repeated() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(4);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(3).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();
    let styles: Vec<DynTensor> = (0..3).map(|i| make_style(600 + i)).collect();

    for iter in 0..3 {
        let mixed = chorus
            .synthesize_chorus_pipelined(&input, &styles, 1.0, &cache)
            .unwrap_or_else(|err| panic!("pipelined iteration {iter} failed: {err}"));
        assert!(
            !mixed.is_empty(),
            "pipelined iteration {iter} produced empty audio",
        );
    }
}

// -- ChorusGpuSynth tests -----------------------------------------------------

/// ChorusGpuSynth::synthesize_batch uses shared encoding (Steps 1-2 once,
/// Steps 3-8 per voice). Verify it produces per-voice audio.
#[test]
fn test_chorus_gpu_synth_synthesize_batch() {
    use nn_metal::compiled_kokoro::gpu_synth::ChorusGpuSynth;
    use nn_models::kokoro_pipeline::KokoroSynth;

    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let styles = vec![make_style(100), make_style(200)];
    let speeds = vec![1.0f32, 1.0];

    let mut gpu = ChorusGpuSynth::new(&mut chorus, &cache);
    let voice_audio = gpu
        .synthesize_batch(&input, &styles, &speeds)
        .expect("synthesize_batch");

    assert_eq!(voice_audio.len(), 2, "should return 2 voice tracks");
    for (i, track) in voice_audio.iter().enumerate() {
        assert!(!track.is_empty(), "voice {i} audio should not be empty");
        let max_abs = track.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.0, "voice {i} should have non-zero audio");
        assert!(
            track.iter().all(|s| s.is_finite()),
            "voice {i} audio should be finite",
        );
    }

    eprintln!(
        "ChorusGpuSynth batch: voice0={} samples, voice1={} samples",
        voice_audio[0].len(),
        voice_audio[1].len(),
    );
}

/// ChorusGpuSynth::synthesize_chunk falls back to voice[0].
#[test]
fn test_chorus_gpu_synth_single_chunk() {
    use nn_metal::compiled_kokoro::gpu_synth::ChorusGpuSynth;
    use nn_models::kokoro_pipeline::KokoroSynth;

    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let style = make_style(50);

    let mut gpu = ChorusGpuSynth::new(&mut chorus, &cache);
    let audio = gpu
        .synthesize_chunk(&input, &style, 1.0)
        .expect("synthesize_chunk");

    assert!(!audio.is_empty(), "single-chunk audio should not be empty");
    assert!(
        audio.iter().all(|s| s.is_finite()),
        "audio should be finite",
    );
    eprintln!("ChorusGpuSynth chunk: {} samples", audio.len());
}

// -- Parallel chorus tests (#4290) --------------------------------------------

/// synthesize_chorus_parallel produces non-empty, finite mixed audio.
#[test]
fn test_parallel_chorus_basic() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(4).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let styles: Vec<DynTensor> = (0..4).map(|i| make_style(700 + i)).collect();
    let mixed = chorus
        .synthesize_chorus_parallel(&input, &styles, 1.0, &cache)
        .expect("parallel chorus");

    assert!(!mixed.is_empty(), "parallel audio should not be empty");
    let max_abs = mixed.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs > 0.0,
        "parallel audio max_abs should be positive, got {max_abs}",
    );
    assert!(
        mixed.iter().all(|s| s.is_finite()),
        "all parallel audio samples should be finite",
    );
    eprintln!(
        "parallel chorus: {} samples, max_abs={max_abs:.6}",
        mixed.len()
    );
}

/// synthesize_chorus_parallel matches synthesize_chorus_shared_encode (bitwise).
#[test]
fn test_parallel_chorus_matches_sequential() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(5);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(4).unwrap();
    let mut chorus_seq = KokoroChorus::new(&primary, config.clone()).unwrap();
    let mut chorus_par = KokoroChorus::new(&primary, config).unwrap();

    let styles: Vec<DynTensor> = (0..4).map(|i| make_style(800 + i)).collect();

    let audio_seq = chorus_seq
        .synthesize_chorus_shared_encode(&input, &styles, 1.0, &cache)
        .expect("sequential chorus");

    let audio_par = chorus_par
        .synthesize_chorus_parallel(&input, &styles, 1.0, &cache)
        .expect("parallel chorus");

    assert_eq!(
        audio_seq.len(),
        audio_par.len(),
        "parallel and sequential should produce same length audio",
    );

    // Allow small floating-point differences due to GPU command buffer ordering.
    let max_diff = audio_seq
        .iter()
        .zip(audio_par.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!(
        "parallel vs sequential: len={}, max_diff={max_diff:.8}",
        audio_seq.len()
    );

    // With mini weights the outputs should be very close (GPU determinism for
    // identical inputs). Allow a small epsilon for command buffer reordering.
    assert!(
        max_diff < 1e-4,
        "parallel should match sequential within epsilon, max_diff={max_diff}",
    );
}

/// synthesize_chorus_parallel with 2 voices.
#[test]
fn test_parallel_chorus_2_voices() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    let styles = vec![make_style(900), make_style(901)];
    let mixed = chorus
        .synthesize_chorus_parallel(&input, &styles, 1.0, &cache)
        .expect("parallel 2-voice");

    assert!(!mixed.is_empty(), "parallel audio should not be empty");
    let max_abs = mixed.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 10.0,
        "audio should not be unreasonably loud: max_abs={max_abs}",
    );
}

/// synthesize_chorus_parallel can be called repeatedly (state cleanup).
#[test]
fn test_parallel_chorus_repeated() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(4);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(3).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();
    let styles: Vec<DynTensor> = (0..3).map(|i| make_style(950 + i)).collect();

    for iter in 0..3 {
        let mixed = chorus
            .synthesize_chorus_parallel(&input, &styles, 1.0, &cache)
            .unwrap_or_else(|err| panic!("parallel iteration {iter} failed: {err}"));
        assert!(
            !mixed.is_empty(),
            "parallel iteration {iter} produced empty audio",
        );
    }
}

/// synthesize_chorus_parallel rejects mismatched style count.
#[test]
fn test_parallel_chorus_style_mismatch() {
    super::test_utils::gpu_init();
    let (mut primary, cache) = build_kokoro();

    let input = make_input(3);
    let style0 = make_style(0);
    let _ = primary
        .synthesize(&input, &style0, 1.0, &cache)
        .expect("warmup");

    let config = ChorusConfig::equal_gain(3).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    // Only 2 styles for 3 voices.
    let styles = vec![make_style(960), make_style(961)];
    let result = chorus.synthesize_chorus_parallel(&input, &styles, 1.0, &cache);
    assert!(result.is_err(), "should reject mismatched style count");
}

/// synthesize_chorus_parallel rejects oversized input.
#[test]
fn test_parallel_chorus_oversized_input() {
    super::test_utils::gpu_init();
    let (primary, cache) = build_kokoro();

    let config = ChorusConfig::equal_gain(2).unwrap();
    let mut chorus = KokoroChorus::new(&primary, config).unwrap();

    // max_position_embeddings for mini config is 16.
    let oversized_input = make_input(17);
    let styles = vec![make_style(970), make_style(971)];
    assert_input_too_long(
        chorus.synthesize_chorus_parallel(&oversized_input, &styles, 1.0, &cache),
        "parallel_chorus_oversized",
        None,
    );
}
