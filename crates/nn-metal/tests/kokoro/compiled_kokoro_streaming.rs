// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `CompiledKokoro::synthesize_streaming()`.
//!
//! Verifies that chunked synthesis produces crossfaded AudioChunks.
//!
//! Part of #3355, #3351, #2918.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_models::kokoro_streaming::{concatenate_chunks, KokoroStreamConfig};

use super::kokoro_test_weights as kw;

fn cpu() -> Device {
    Device::Cpu
}

const STYLE_DIM: usize = 4; // Must match kw::mini_test_config().style_dim

fn build_kokoro() -> (
    nn_metal::compiled_kokoro::CompiledKokoro,
    nn_metal::PipelineCache,
) {
    kw::build_kokoro_mini()
}

fn make_style(seed: usize) -> DynTensor {
    DynTensor::new(
        &super::test_utils::rand_f32_vec(300 + seed as u64, 2 * STYLE_DIM, -0.1, 0.1),
        &[1, 2 * STYLE_DIM],
        &cpu(),
    )
    .unwrap()
}

fn make_input(len: usize) -> DynTensor {
    let vals: Vec<f32> = (1..=len).map(|v| v as f32).collect();
    DynTensor::from_vec(vals, &[1, len], &cpu()).unwrap()
}

/// Single chunk streaming is equivalent to a single synthesize call.
#[test]
fn test_streaming_single_chunk() {
    super::test_utils::gpu_init();
    let (mut kokoro, cache) = build_kokoro();

    let input = make_input(3);
    let style = make_style(0);
    let config = KokoroStreamConfig::default();

    let chunks = kokoro
        .synthesize_streaming(&[input], &style, 1.0, &config, &cache)
        .expect("streaming single chunk");

    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].is_final);
    assert_eq!(chunks[0].chunk_index, 0);
    assert_eq!(chunks[0].sample_offset, 0);
    assert!(!chunks[0].is_empty(), "chunk should have audio");
}

/// Two-chunk streaming produces crossfaded output.
#[test]
fn test_streaming_two_chunks() {
    super::test_utils::gpu_init();
    let (mut kokoro, cache) = build_kokoro();

    let style = make_style(0);

    // Warmup with first chunk shape
    let chunk0 = make_input(3);
    let (warmup_audio, _cert) = kokoro
        .synthesize(&chunk0, &style, 1.0, &cache)
        .expect("warmup");
    let config = super::test_utils::short_stream_config_for_pcm_len(warmup_audio.dims()[2]);

    let chunk1 = make_input(4);
    let chunks = kokoro
        .synthesize_streaming(&[chunk0, chunk1], &style, 1.0, &config, &cache)
        .expect("streaming two chunks");

    assert_eq!(chunks.len(), 2);
    assert!(!chunks[0].is_final);
    assert!(chunks[1].is_final);
    assert_eq!(chunks[0].chunk_index, 0);
    assert_eq!(chunks[1].chunk_index, 1);
    assert_eq!(chunks[0].total_chunks, 2);
    assert_eq!(chunks[1].total_chunks, 2);

    // Second chunk starts after first chunk's playable portion.
    assert_eq!(chunks[1].sample_offset, chunks[0].pcm.len());

    // Concatenated audio should be non-empty.
    let full = concatenate_chunks(&chunks);
    assert!(!full.is_empty());
    eprintln!("streaming 2 chunks: {} total samples", full.len());
}

/// Regression for #3507: a single CompiledKokoro instance can synthesize
/// multiple shape variants, then stream mixed chunk lengths, without stale
/// shared-weight aliasing across recompiles.
#[test]
fn test_streaming_mixed_lengths_after_shape_recompiles() {
    super::test_utils::gpu_init();
    let (mut kokoro, cache) = build_kokoro();

    let style = make_style(1);
    let seed_short = make_input(3);
    let (warmup_audio, _cert) = kokoro
        .synthesize(&seed_short, &style, 1.0, &cache)
        .expect("warmup");
    let config = super::test_utils::short_stream_config_for_pcm_len(warmup_audio.dims()[2]);

    // Force additional shape/speed recompiles before streaming.
    let _ = kokoro
        .synthesize(&make_input(5), &style, 1.0, &cache)
        .expect("shape recompile seq=5");
    let _ = kokoro
        .synthesize(&make_input(6), &style, 0.9, &cache)
        .expect("shape+speed variant seq=6");

    let chunk0 = make_input(4);
    let chunk1 = make_input(7);
    let chunks = kokoro
        .synthesize_streaming(&[chunk0, chunk1], &style, 1.0, &config, &cache)
        .expect("streaming mixed lengths after shape recompiles");

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[1].sample_offset, chunks[0].pcm.len());

    let full = concatenate_chunks(&chunks);
    assert!(!full.is_empty());
    eprintln!(
        "streaming mixed lengths after shape recompiles: {} total samples",
        full.len()
    );
}

/// Empty chunk list produces empty result.
#[test]
fn test_streaming_empty() {
    super::test_utils::gpu_init();
    let (mut kokoro, cache) = build_kokoro();

    let style = make_style(0);
    let config = KokoroStreamConfig::default();

    let chunks = kokoro
        .synthesize_streaming(&[], &style, 1.0, &config, &cache)
        .expect("streaming empty");

    assert!(chunks.is_empty());
}

/// synthesize_with_intermediates returns durations, F0, and energy alongside audio.
#[test]
fn test_synthesize_with_intermediates() {
    super::test_utils::gpu_init();
    let (mut kokoro, cache) = build_kokoro();

    let input = make_input(3);
    let style = make_style(0);

    let (audio, certificate, intermediates) = kokoro
        .synthesize_with_intermediates(&input, &style, 1.0, &cache)
        .expect("synthesize_with_intermediates");

    // Audio should be non-empty [1, 1, T_audio].
    assert_eq!(audio.dims().len(), 3);
    assert!(audio.dims()[2] > 0, "audio should have samples");

    // Certificate should exist (pass or fail, but present).
    let _ = certificate.overall_passed;

    // Intermediates: durations [B, T] where T = input seq_len.
    assert_eq!(intermediates.durations.dims().len(), 2);
    assert_eq!(intermediates.durations.dims()[0], 1); // batch=1

    // Intermediates: F0 [B, 1, 2*T_mel].
    assert_eq!(intermediates.f0.dims().len(), 3);
    assert_eq!(intermediates.f0.dims()[0], 1);
    assert_eq!(intermediates.f0.dims()[1], 1);
    assert!(intermediates.f0.dims()[2] > 0, "F0 should be non-empty");

    // Intermediates: energy [B, 1, 2*T_mel].
    assert_eq!(intermediates.energy.dims().len(), 3);
    assert_eq!(intermediates.energy.dims()[0], 1);
    assert_eq!(intermediates.energy.dims()[1], 1);
    assert_eq!(intermediates.energy.dims()[2], intermediates.f0.dims()[2]);

    // t_mel should be positive.
    assert!(intermediates.t_mel > 0, "t_mel should be > 0");

    // F0 and energy time dimension should be 2*t_mel.
    assert_eq!(intermediates.f0.dims()[2], 2 * intermediates.t_mel);

    eprintln!(
        "intermediates: durations={:?}, f0={:?}, energy={:?}, t_mel={}",
        intermediates.durations.dims(),
        intermediates.f0.dims(),
        intermediates.energy.dims(),
        intermediates.t_mel,
    );
}
