// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`StreamingKokoroSession`].

use crate::kokoro_error::KokoroError;
use crate::kokoro_pipeline::KokoroSynth;
use crate::kokoro_streaming::{KokoroStreamConfig, StreamingKokoroSession};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

// ---------------------------------------------------------------------------
// Mock synthesizer for testing
// ---------------------------------------------------------------------------

/// Simple mock synth that produces constant-valued PCM of deterministic length.
///
/// For each `synthesize_chunk` call, produces `output_len` samples of value
/// `(call_index + 1) as f32 * 0.1`. This lets tests verify ordering and
/// crossfade behavior without real model weights.
struct MockSynth {
    output_len: usize,
    call_count: usize,
}

impl MockSynth {
    fn new(output_len: usize) -> Self {
        Self {
            output_len,
            call_count: 0,
        }
    }
}

impl KokoroSynth for MockSynth {
    type Error = KokoroError;

    fn synthesize_chunk(
        &mut self,
        _input_ids: &DynTensor,
        _style: &DynTensor,
        _speed: f32,
    ) -> Result<Vec<f32>, KokoroError> {
        let val = (self.call_count + 1) as f32 * 0.1;
        self.call_count += 1;
        Ok(vec![val; self.output_len])
    }
}

/// Mock synth that records the speed parameter from each call.
///
/// Used to verify that `StreamingKokoroSession::next_chunk()` faithfully
/// forwards the speed argument to the underlying synth backend.
struct SpeedCapturingSynth {
    output_len: usize,
    captured_speeds: Vec<f32>,
}

impl SpeedCapturingSynth {
    fn new(output_len: usize) -> Self {
        Self {
            output_len,
            captured_speeds: Vec::new(),
        }
    }
}

impl KokoroSynth for SpeedCapturingSynth {
    type Error = KokoroError;

    fn synthesize_chunk(
        &mut self,
        _input_ids: &DynTensor,
        _style: &DynTensor,
        speed: f32,
    ) -> Result<Vec<f32>, KokoroError> {
        self.captured_speeds.push(speed);
        Ok(vec![0.5; self.output_len])
    }
}

/// Mock synth that fails on the Nth call.
struct FailingSynth {
    fail_on: usize,
    call_count: usize,
}

impl KokoroSynth for FailingSynth {
    type Error = KokoroError;

    fn synthesize_chunk(
        &mut self,
        _input_ids: &DynTensor,
        _style: &DynTensor,
        _speed: f32,
    ) -> Result<Vec<f32>, KokoroError> {
        self.call_count += 1;
        if self.call_count == self.fail_on {
            return Err(KokoroError::InvalidInput("mock synthesis failure".into()));
        }
        Ok(vec![0.5; 2000])
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_dummy_tensor() -> DynTensor {
    DynTensor::from_vec_u32(vec![1, 2, 3, 4], &[1, 4], &Device::Cpu)
        .expect("should create dummy tensor")
}

fn make_dummy_style() -> DynTensor {
    DynTensor::zeros(&[1, 512], nn_core::DType::F32, &Device::Cpu)
        .expect("should create style tensor")
}

fn default_config() -> KokoroStreamConfig {
    KokoroStreamConfig::new(480).expect("valid config")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_session_new_empty_chunks_returns_error() {
    let result = StreamingKokoroSession::new(vec![], default_config());
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        KokoroError::InvalidConfig { field, .. } => {
            assert_eq!(field, "chunks");
        }
        other => panic!("expected InvalidConfig, got: {other:?}"),
    }
}

#[test]
fn test_session_initial_state() {
    let chunks = vec![
        make_dummy_tensor(),
        make_dummy_tensor(),
        make_dummy_tensor(),
    ];
    let session =
        StreamingKokoroSession::new(chunks, default_config()).expect("should create session");

    assert_eq!(session.total_chunks(), 3);
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.cursor(), 0);
    assert!(!session.is_complete());
}

#[test]
fn test_session_single_chunk_yields_one_then_none() {
    let chunks = vec![make_dummy_tensor()];
    let mut session =
        StreamingKokoroSession::new(chunks, default_config()).expect("should create session");
    let mut synth = MockSynth::new(2000);
    let style = make_dummy_style();

    // First call: should get Some(AudioChunk)
    let chunk = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("should synthesize")
        .expect("should have chunk");
    assert!(chunk.is_final);
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.total_chunks, 1);
    assert!(!chunk.pcm.is_empty());

    // Second call: should get None
    let none = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("should succeed");
    assert!(none.is_none());
    assert!(session.is_complete());
    assert_eq!(session.remaining(), 0);
}

#[test]
fn test_session_multiple_chunks_progress() {
    let n = 4;
    let chunks: Vec<DynTensor> = (0..n).map(|_| make_dummy_tensor()).collect();
    let mut session =
        StreamingKokoroSession::new(chunks, default_config()).expect("should create session");
    let mut synth = MockSynth::new(2000);
    let style = make_dummy_style();

    for i in 0..n {
        assert_eq!(session.remaining(), n - i);
        assert_eq!(session.cursor(), i);
        assert!(!session.is_complete());

        let chunk = session
            .next_chunk(&mut synth, &style, 1.0)
            .expect("should synthesize")
            .expect("should have chunk");
        assert_eq!(chunk.chunk_index, i);
        assert_eq!(chunk.total_chunks, n);
        assert_eq!(chunk.is_final, i == n - 1);
    }

    assert!(session.is_complete());
    assert_eq!(session.remaining(), 0);

    // After completion, returns None
    let none = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("should succeed");
    assert!(none.is_none());
}

#[test]
fn test_session_synth_error_propagates() {
    let chunks = vec![make_dummy_tensor(), make_dummy_tensor()];
    let mut session =
        StreamingKokoroSession::new(chunks, default_config()).expect("should create session");
    let mut synth = FailingSynth {
        fail_on: 2,
        call_count: 0,
    };
    let style = make_dummy_style();

    // First chunk succeeds
    let first = session.next_chunk(&mut synth, &style, 1.0);
    assert!(first.is_ok());

    // Second chunk fails
    let second = session.next_chunk(&mut synth, &style, 1.0);
    assert!(second.is_err());
}

#[test]
fn test_session_crossfade_produces_non_empty_audio() {
    let chunks = vec![make_dummy_tensor(), make_dummy_tensor()];
    let config = KokoroStreamConfig::new(100).expect("valid config");
    let mut session = StreamingKokoroSession::new(chunks, config).expect("should create session");
    let mut synth = MockSynth::new(2000);
    let style = make_dummy_style();

    let first = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("ok")
        .expect("some");
    assert!(!first.pcm.is_empty());
    assert_eq!(first.chunk_index, 0);

    let second = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("ok")
        .expect("some");
    assert!(!second.pcm.is_empty());
    assert_eq!(second.chunk_index, 1);
    assert!(second.is_final);
}

#[test]
fn test_session_sample_offsets_increase() {
    let n = 3;
    let chunks: Vec<DynTensor> = (0..n).map(|_| make_dummy_tensor()).collect();
    let config = KokoroStreamConfig::new(100).expect("valid config");
    let mut session = StreamingKokoroSession::new(chunks, config).expect("should create session");
    let mut synth = MockSynth::new(2000);
    let style = make_dummy_style();

    let mut prev_offset = 0;
    for i in 0..n {
        let chunk = session
            .next_chunk(&mut synth, &style, 1.0)
            .expect("ok")
            .expect("some");
        if i > 0 {
            assert!(
                chunk.sample_offset > prev_offset,
                "sample_offset should increase: {} <= {}",
                chunk.sample_offset,
                prev_offset,
            );
        }
        prev_offset = chunk.sample_offset;
    }
}

// ---------------------------------------------------------------------------
// cancel() tests
// ---------------------------------------------------------------------------

#[test]
fn test_session_cancel_before_any_synthesis_remaining_zero() {
    let chunks = vec![
        make_dummy_tensor(),
        make_dummy_tensor(),
        make_dummy_tensor(),
    ];
    let mut session =
        StreamingKokoroSession::new(chunks, default_config()).expect("should create session");

    assert_eq!(session.remaining(), 3);

    session.cancel();

    assert_eq!(session.remaining(), 0);
    assert!(session.is_complete());
    assert_eq!(session.cursor(), 3);
}

#[test]
fn test_session_cancel_mid_stream_stops_synthesis() {
    let chunks = vec![
        make_dummy_tensor(),
        make_dummy_tensor(),
        make_dummy_tensor(),
        make_dummy_tensor(),
    ];
    let mut session =
        StreamingKokoroSession::new(chunks, default_config()).expect("should create session");
    let mut synth = MockSynth::new(2000);
    let style = make_dummy_style();

    // Synthesize first two chunks.
    let _ = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("ok")
        .expect("some");
    let _ = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("ok")
        .expect("some");
    assert_eq!(session.remaining(), 2);
    assert_eq!(synth.call_count, 2);

    // Cancel mid-stream.
    session.cancel();

    assert_eq!(session.remaining(), 0);
    assert!(session.is_complete());

    // Subsequent next_chunk returns None without calling the synth.
    let none = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("should succeed");
    assert!(none.is_none());
    // Synth was NOT called again after cancel.
    assert_eq!(synth.call_count, 2);
}

#[test]
fn test_session_cancel_after_completion_is_idempotent() {
    let chunks = vec![make_dummy_tensor()];
    let mut session =
        StreamingKokoroSession::new(chunks, default_config()).expect("should create session");
    let mut synth = MockSynth::new(2000);
    let style = make_dummy_style();

    let _ = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("ok")
        .expect("some");
    assert!(session.is_complete());

    // Cancel after already complete: no-op, no panic.
    session.cancel();

    assert!(session.is_complete());
    assert_eq!(session.remaining(), 0);
}

// ---------------------------------------------------------------------------
// Speed forwarding tests
// ---------------------------------------------------------------------------

#[test]
fn test_session_next_chunk_forwards_speed_half() {
    let chunks = vec![make_dummy_tensor(), make_dummy_tensor()];
    let mut session =
        StreamingKokoroSession::new(chunks, default_config()).expect("should create session");
    let mut synth = SpeedCapturingSynth::new(2000);
    let style = make_dummy_style();

    let _ = session
        .next_chunk(&mut synth, &style, 0.5)
        .expect("ok")
        .expect("some");
    let _ = session
        .next_chunk(&mut synth, &style, 0.5)
        .expect("ok")
        .expect("some");

    assert_eq!(synth.captured_speeds.len(), 2);
    assert!((synth.captured_speeds[0] - 0.5).abs() < f32::EPSILON);
    assert!((synth.captured_speeds[1] - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_session_next_chunk_forwards_speed_double() {
    let chunks = vec![make_dummy_tensor()];
    let mut session =
        StreamingKokoroSession::new(chunks, default_config()).expect("should create session");
    let mut synth = SpeedCapturingSynth::new(2000);
    let style = make_dummy_style();

    let _ = session
        .next_chunk(&mut synth, &style, 2.0)
        .expect("ok")
        .expect("some");

    assert_eq!(synth.captured_speeds.len(), 1);
    assert!((synth.captured_speeds[0] - 2.0).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// Exhaustion / idempotency tests
// ---------------------------------------------------------------------------

#[test]
fn test_session_repeated_next_chunk_after_exhaustion_returns_none() {
    let chunks = vec![make_dummy_tensor()];
    let mut session =
        StreamingKokoroSession::new(chunks, default_config()).expect("should create session");
    let mut synth = MockSynth::new(2000);
    let style = make_dummy_style();

    // Consume the only chunk.
    let _ = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("ok")
        .expect("some");

    // Call next_chunk 5 more times: all should return None.
    for _ in 0..5 {
        let result = session
            .next_chunk(&mut synth, &style, 1.0)
            .expect("should not error");
        assert!(result.is_none(), "expected None after exhaustion");
    }

    // Synth was called exactly once (for the single chunk).
    assert_eq!(synth.call_count, 1);
}

// ---------------------------------------------------------------------------
// Error does not advance cursor
// ---------------------------------------------------------------------------

#[test]
fn test_session_synth_error_does_not_advance_cursor() {
    let chunks = vec![make_dummy_tensor(), make_dummy_tensor()];
    let mut session =
        StreamingKokoroSession::new(chunks, default_config()).expect("should create session");
    let style = make_dummy_style();

    // Synth that fails on the very first call.
    let mut failing = FailingSynth {
        fail_on: 1,
        call_count: 0,
    };

    let result = session.next_chunk(&mut failing, &style, 1.0);
    assert!(result.is_err());

    // Cursor should NOT have advanced: still at 0, remaining still 2.
    assert_eq!(session.cursor(), 0);
    assert_eq!(session.remaining(), 2);
    assert!(!session.is_complete());
}

// ---------------------------------------------------------------------------
// Large number of chunks (stress test)
// ---------------------------------------------------------------------------

#[test]
fn test_session_many_chunks_exhaust_stream() {
    let n = 50;
    let chunks: Vec<DynTensor> = (0..n).map(|_| make_dummy_tensor()).collect();
    let config = KokoroStreamConfig::new(100).expect("valid config");
    let mut session = StreamingKokoroSession::new(chunks, config).expect("should create session");
    let mut synth = MockSynth::new(2000);
    let style = make_dummy_style();

    let mut audio_chunks = Vec::new();
    while let Some(chunk) = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("should not error")
    {
        audio_chunks.push(chunk);
    }

    assert_eq!(audio_chunks.len(), n);
    assert!(session.is_complete());
    assert_eq!(session.remaining(), 0);

    // Verify final flag only on last chunk.
    for (i, chunk) in audio_chunks.iter().enumerate() {
        assert_eq!(chunk.is_final, i == n - 1, "chunk {i} is_final mismatch");
        assert_eq!(chunk.chunk_index, i);
        assert_eq!(chunk.total_chunks, n);
        assert!(!chunk.pcm.is_empty());
    }

    // Verify synth was called exactly n times.
    assert_eq!(synth.call_count, n);
}

// ---------------------------------------------------------------------------
// Concatenated streaming output continuity
// ---------------------------------------------------------------------------

#[test]
fn test_session_streaming_output_matches_batch_assembly() {
    use crate::kokoro_streaming::{assemble_streaming_chunks, concatenate_chunks};

    let n = 5;
    let output_len = 2000;
    let cf = 100;
    let config = KokoroStreamConfig::new(cf).expect("valid config");

    // Generate deterministic raw PCM for each chunk (same as MockSynth would).
    let raw_chunks: Vec<Vec<f32>> = (0..n)
        .map(|i| vec![(i + 1) as f32 * 0.1; output_len])
        .collect();

    // Batch assembly path.
    let batch_chunks = assemble_streaming_chunks(&raw_chunks, &config).expect("batch ok");
    let batch_pcm = concatenate_chunks(&batch_chunks);

    // Streaming session path with same mock data.
    let chunks: Vec<DynTensor> = (0..n).map(|_| make_dummy_tensor()).collect();
    let mut session = StreamingKokoroSession::new(chunks, config).expect("should create session");
    let mut synth = MockSynth::new(output_len);
    let style = make_dummy_style();

    let mut stream_audio_chunks = Vec::new();
    while let Some(chunk) = session.next_chunk(&mut synth, &style, 1.0).expect("ok") {
        stream_audio_chunks.push(chunk);
    }
    let stream_pcm = concatenate_chunks(&stream_audio_chunks);

    // Verify lengths match.
    assert_eq!(
        stream_pcm.len(),
        batch_pcm.len(),
        "total PCM length mismatch: streaming={}, batch={}",
        stream_pcm.len(),
        batch_pcm.len(),
    );

    // Verify sample values match within epsilon.
    for (i, (&s, &b)) in stream_pcm.iter().zip(batch_pcm.iter()).enumerate() {
        assert!(
            (s - b).abs() < 1e-6,
            "PCM mismatch at sample {i}: streaming={s}, batch={b}",
        );
    }
}

// ---------------------------------------------------------------------------
// Small crossfade (edge case: crossfade_samples = 1)
// ---------------------------------------------------------------------------

#[test]
fn test_session_crossfade_single_sample() {
    let chunks = vec![make_dummy_tensor(), make_dummy_tensor()];
    let config = KokoroStreamConfig::new(1).expect("valid config");
    let mut session = StreamingKokoroSession::new(chunks, config).expect("should create session");
    let mut synth = MockSynth::new(2000);
    let style = make_dummy_style();

    let first = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("ok")
        .expect("some");
    let second = session
        .next_chunk(&mut synth, &style, 1.0)
        .expect("ok")
        .expect("some");

    // Both chunks produced non-empty audio.
    assert!(!first.pcm.is_empty());
    assert!(!second.pcm.is_empty());
    assert!(second.is_final);
    assert!(session.is_complete());
}
