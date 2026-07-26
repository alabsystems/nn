// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for streaming reverb and ReverbConfig presets.

use crate::kokoro_chorus_reverb::ReverbConfig;
use crate::kokoro_chorus_reverb_streaming::StreamingReverb;
use crate::kokoro_streaming::AudioChunk;

// ---------------------------------------------------------------------------
// ReverbConfig preset tests
// ---------------------------------------------------------------------------

#[test]
fn test_preset_small_room_valid() {
    let config = ReverbConfig::small_room();
    config.validate().unwrap();
    assert!((config.reverb_mix - 0.10).abs() < 1e-6);
    assert!((config.room_size - 0.15).abs() < 1e-6);
    assert!(config.early_reflections);
    assert!((config.damping - 0.3).abs() < 1e-6);
}

#[test]
fn test_preset_medium_hall_valid() {
    let config = ReverbConfig::medium_hall();
    config.validate().unwrap();
    assert!((config.reverb_mix - 0.20).abs() < 1e-6);
    assert!((config.room_size - 0.45).abs() < 1e-6);
    assert!(config.early_reflections);
    assert!((config.damping - 0.5).abs() < 1e-6);
}

#[test]
fn test_preset_large_church_valid() {
    let config = ReverbConfig::large_church();
    config.validate().unwrap();
    assert!((config.reverb_mix - 0.30).abs() < 1e-6);
    assert!((config.room_size - 0.70).abs() < 1e-6);
    assert!(config.early_reflections);
    assert!((config.damping - 0.6).abs() < 1e-6);
}

#[test]
fn test_preset_cathedral_valid() {
    let config = ReverbConfig::cathedral();
    config.validate().unwrap();
    assert!((config.reverb_mix - 0.35).abs() < 1e-6);
    assert!((config.room_size - 0.90).abs() < 1e-6);
    assert!(config.early_reflections);
    assert!((config.damping - 0.7).abs() < 1e-6);
}

#[test]
fn test_presets_increasing_room_size() {
    let small = ReverbConfig::small_room();
    let medium = ReverbConfig::medium_hall();
    let church = ReverbConfig::large_church();
    let cathedral = ReverbConfig::cathedral();
    assert!(small.room_size < medium.room_size);
    assert!(medium.room_size < church.room_size);
    assert!(church.room_size < cathedral.room_size);
}

#[test]
fn test_presets_increasing_reverb_mix() {
    let small = ReverbConfig::small_room();
    let medium = ReverbConfig::medium_hall();
    let church = ReverbConfig::large_church();
    let cathedral = ReverbConfig::cathedral();
    assert!(small.reverb_mix < medium.reverb_mix);
    assert!(medium.reverb_mix < church.reverb_mix);
    assert!(church.reverb_mix < cathedral.reverb_mix);
}

// ---------------------------------------------------------------------------
// StreamingReverb construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_streaming_reverb_new_valid() {
    let config = ReverbConfig::medium_hall();
    let reverb = StreamingReverb::new(config.clone(), true);
    assert!(reverb.is_ok());
    let reverb = reverb.unwrap();
    assert!(reverb.is_stereo());
    assert!((reverb.config().reverb_mix - config.reverb_mix).abs() < 1e-6);
}

#[test]
fn test_streaming_reverb_new_mono() {
    let reverb = StreamingReverb::new(ReverbConfig::small_room(), false).unwrap();
    assert!(!reverb.is_stereo());
}

#[test]
fn test_streaming_reverb_invalid_config() {
    let bad = ReverbConfig::new().with_reverb_mix(2.0);
    assert!(StreamingReverb::new(bad, true).is_err());
}

// ---------------------------------------------------------------------------
// StreamingReverb state persistence tests
// ---------------------------------------------------------------------------

// Note: Schroeder comb filter delays are 1116, 1188, 1277, 1356 samples.
// Buffers must be longer than the longest delay (1356) to see reverb output.
// For stereo, each frame is 2 floats, so we need 2 * N floats for N frames.

/// Stereo buffer size: 3000 frames * 2 channels = 6000 floats.
const STEREO_BUF_LEN: usize = 6000;
/// Mono buffer size: 3000 samples.
const MONO_BUF_LEN: usize = 3000;

#[test]
fn test_streaming_reverb_modifies_audio_stereo() {
    let mut reverb = StreamingReverb::new(ReverbConfig::medium_hall(), true).unwrap();
    // Create a stereo impulse: [1.0, 1.0, 0.0, 0.0, ...]
    let mut buffer = vec![0.0f32; STEREO_BUF_LEN];
    buffer[0] = 1.0;
    buffer[1] = 1.0;
    reverb.process_chunk(&mut buffer);
    // After reverb, the reverb tail should contain energy beyond the
    // comb filter delay length (samples 2400+ in interleaved stereo).
    let tail_energy: f32 = buffer[2400..].iter().map(|x| x * x).sum();
    assert!(
        tail_energy > 1e-6,
        "reverb tail should have nonzero energy, got {tail_energy}"
    );
}

#[test]
fn test_streaming_reverb_modifies_audio_mono() {
    let mut reverb = StreamingReverb::new(ReverbConfig::large_church(), false).unwrap();
    let mut buffer = vec![0.0f32; MONO_BUF_LEN];
    buffer[0] = 1.0;
    reverb.process_chunk(&mut buffer);
    let tail_energy: f32 = buffer[1400..].iter().map(|x| x * x).sum();
    assert!(
        tail_energy > 1e-6,
        "mono reverb tail should have nonzero energy, got {tail_energy}"
    );
}

#[test]
fn test_streaming_reverb_state_persists_across_chunks() {
    let mut reverb = StreamingReverb::new(ReverbConfig::cathedral(), true).unwrap();

    // Chunk 1: stereo impulse fills the delay lines.
    let mut chunk1 = vec![0.0f32; STEREO_BUF_LEN];
    chunk1[0] = 1.0;
    chunk1[1] = 1.0;
    reverb.process_chunk(&mut chunk1);

    // Chunk 2: silence. If reverb state persists, the delay lines should
    // produce nonzero output (the reverb tail from chunk 1 continuing).
    let mut chunk2 = vec![0.0f32; STEREO_BUF_LEN];
    reverb.process_chunk(&mut chunk2);

    let chunk2_energy: f32 = chunk2.iter().map(|x| x * x).sum();
    assert!(
        chunk2_energy > 1e-8,
        "chunk 2 should carry reverb tail from chunk 1, energy = {chunk2_energy}"
    );
}

#[test]
fn test_streaming_reverb_reset_clears_state() {
    let mut reverb = StreamingReverb::new(ReverbConfig::cathedral(), true).unwrap();

    // Process an impulse to fill delay lines.
    let mut impulse = vec![0.0f32; STEREO_BUF_LEN];
    impulse[0] = 1.0;
    impulse[1] = 1.0;
    reverb.process_chunk(&mut impulse);

    // Reset clears delay lines.
    reverb.reset();

    // Process silence -- should produce zero output since state was reset.
    let mut silence = vec![0.0f32; STEREO_BUF_LEN];
    reverb.process_chunk(&mut silence);

    let energy: f32 = silence.iter().map(|x| x * x).sum();
    assert!(
        energy < 1e-12,
        "after reset, silence input should produce zero output, energy = {energy}"
    );
}

#[test]
fn test_streaming_reverb_dry_mix_passthrough() {
    let config = ReverbConfig::new().with_reverb_mix(0.0);
    let mut reverb = StreamingReverb::new(config, true).unwrap();

    let original = vec![0.5f32, -0.3, 0.1, 0.2, 0.0, 0.0, 0.7, -0.4];
    let mut buffer = original.clone();
    reverb.process_chunk(&mut buffer);

    // With mix = 0.0, output should be identical to input (dry passthrough).
    for (i, (&orig, &processed)) in original.iter().zip(buffer.iter()).enumerate() {
        assert!(
            (orig - processed).abs() < 1e-10,
            "sample {i}: expected {orig}, got {processed}"
        );
    }
}

// ---------------------------------------------------------------------------
// apply_to_chunks tests
// ---------------------------------------------------------------------------

#[test]
fn test_apply_to_chunks_processes_all_chunks() {
    let mut reverb = StreamingReverb::new(ReverbConfig::medium_hall(), false).unwrap();

    let mut impulse_pcm = vec![0.0f32; MONO_BUF_LEN];
    impulse_pcm[0] = 1.0;
    let mut chunks = vec![
        AudioChunk::new(impulse_pcm, 1, 0, 0, 2, false),
        AudioChunk::new(vec![0.0f32; MONO_BUF_LEN], 1, MONO_BUF_LEN, 1, 2, true),
    ];

    reverb.apply_to_chunks(&mut chunks);

    // Chunk 0 should have reverb energy past the comb filter delay.
    let c0_tail: f32 = chunks[0].pcm[1400..].iter().map(|x| x * x).sum();
    assert!(c0_tail > 1e-6, "chunk 0 tail should have reverb energy");

    // Chunk 1 (silence input) should carry reverb tail from chunk 0.
    let c1_energy: f32 = chunks[1].pcm.iter().map(|x| x * x).sum();
    assert!(
        c1_energy > 1e-8,
        "chunk 1 should carry reverb tail from chunk 0"
    );
}

#[test]
fn test_apply_to_chunks_empty() {
    let mut reverb = StreamingReverb::new(ReverbConfig::small_room(), true).unwrap();
    let mut chunks: Vec<AudioChunk> = Vec::new();
    reverb.apply_to_chunks(&mut chunks);
    // No panic.
}

// ---------------------------------------------------------------------------
// Larger room = more reverb energy tests
// ---------------------------------------------------------------------------

#[test]
fn test_larger_room_produces_more_reverb_energy() {
    // Process the same impulse through small room and cathedral,
    // then compare the reverb tail energy.
    let small = ReverbConfig::small_room();
    let cathedral = ReverbConfig::cathedral();

    // Use buffers longer than the longest comb delay (1356 samples)
    // so the reverb tail is measurable.
    let make_impulse = || -> Vec<f32> {
        let mut buf = vec![0.0f32; MONO_BUF_LEN];
        buf[0] = 1.0;
        buf
    };

    let mut small_reverb = StreamingReverb::new(small, false).unwrap();
    let mut small_buf = make_impulse();
    small_reverb.process_chunk(&mut small_buf);

    let mut cathedral_reverb = StreamingReverb::new(cathedral, false).unwrap();
    let mut cathedral_buf = make_impulse();
    cathedral_reverb.process_chunk(&mut cathedral_buf);

    // Compare energy past the longest comb delay.
    let small_energy: f32 = small_buf[1400..].iter().map(|x| x * x).sum();
    let cathedral_energy: f32 = cathedral_buf[1400..].iter().map(|x| x * x).sum();

    assert!(
        cathedral_energy > small_energy,
        "cathedral ({cathedral_energy}) should have more reverb energy than small room ({small_energy})"
    );
}
