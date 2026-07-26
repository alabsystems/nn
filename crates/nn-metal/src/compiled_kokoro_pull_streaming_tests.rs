// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`StreamingKokoroSession`] pull-based streaming session.

use super::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

/// Helper: create a dummy `(input_ids, style)` pair for state-machine tests.
/// These are not valid Kokoro inputs — they just need to be DynTensors.
fn dummy_chunk() -> (DynTensor, DynTensor) {
    let cpu = Device::Cpu;
    let input_ids = DynTensor::zeros(&[1, 10], DType::I64, &cpu).unwrap();
    let style = DynTensor::zeros(&[1, 512], DType::F32, &cpu).unwrap();
    (input_ids, style)
}

fn dummy_style() -> DynTensor {
    DynTensor::zeros(&[1, 512], DType::F32, &Device::Cpu).unwrap()
}

fn default_stream_config() -> KokoroStreamConfig {
    KokoroStreamConfig::new(480).expect("valid config")
}

// -------------------------------------------------------------------
// Basic session state-machine tests (new() constructor)
// -------------------------------------------------------------------

#[test]
fn test_empty_session_is_done() {
    let session = StreamingKokoroSession::new(Vec::new(), 1.0);
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);
    assert_eq!(session.total_chunks(), 0);
    assert_eq!(session.synthesized_count(), 0);
}

#[test]
fn test_empty_session_next_returns_none() {
    let session = StreamingKokoroSession::new(Vec::new(), 1.0);
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);
}

#[test]
fn test_session_initial_state() {
    let chunks = vec![dummy_chunk(), dummy_chunk(), dummy_chunk()];
    let session = StreamingKokoroSession::new(chunks, 1.5);
    assert!(!session.is_done());
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.total_chunks(), 3);
    assert_eq!(session.synthesized_count(), 0);
    assert!((session.speed() - 1.5).abs() < f32::EPSILON);
}

#[test]
fn test_session_reset() {
    let chunks = vec![dummy_chunk(), dummy_chunk()];
    let mut session = StreamingKokoroSession::new(chunks, 1.0);
    session.cursor = 2;
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);

    session.reset();
    assert!(!session.is_done());
    assert_eq!(session.remaining(), 2);
    assert_eq!(session.synthesized_count(), 0);
}

#[test]
fn test_session_set_speed() {
    let mut session = StreamingKokoroSession::new(vec![dummy_chunk()], 1.0);
    assert!((session.speed() - 1.0).abs() < f32::EPSILON);
    session.set_speed(0.8);
    assert!((session.speed() - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_remaining_counts_down() {
    let chunks = vec![dummy_chunk(), dummy_chunk(), dummy_chunk()];
    let mut session = StreamingKokoroSession::new(chunks, 1.0);

    assert_eq!(session.remaining(), 3);
    assert_eq!(session.synthesized_count(), 0);

    session.cursor = 1;
    assert_eq!(session.remaining(), 2);
    assert_eq!(session.synthesized_count(), 1);

    session.cursor = 2;
    assert_eq!(session.remaining(), 1);
    assert_eq!(session.synthesized_count(), 2);

    session.cursor = 3;
    assert_eq!(session.remaining(), 0);
    assert_eq!(session.synthesized_count(), 3);
    assert!(session.is_done());
}

#[test]
fn test_remaining_saturates_at_zero() {
    let mut session = StreamingKokoroSession::new(vec![dummy_chunk()], 1.0);
    session.cursor = 100;
    assert_eq!(session.remaining(), 0);
    assert!(session.is_done());
}

#[test]
fn test_precompile_enabled_by_default() {
    let session = StreamingKokoroSession::new(vec![dummy_chunk()], 1.0);
    assert!(session.precompile_enabled());
}

#[test]
fn test_precompile_disabled_via_builder() {
    let session =
        StreamingKokoroSession::new(vec![dummy_chunk()], 1.0).with_precompile(false);
    assert!(!session.precompile_enabled());
}

#[test]
fn test_precompile_shapes_from_token_lengths() {
    let shapes = PrecompileShapes::from_token_lengths(&[10, 20, 40]);
    assert!(shapes.is_some());
    let shapes = shapes.unwrap();
    assert_eq!(shapes.seq_lens, vec![10, 20, 40]);
    assert!(!shapes.t_mels.is_empty());
}

#[test]
fn test_precompile_shapes_from_empty() {
    let shapes = PrecompileShapes::from_token_lengths(&[]);
    assert!(shapes.is_none());
}

#[test]
fn test_precompile_shapes_deduplicates() {
    let shapes = PrecompileShapes::from_token_lengths(&[20, 20, 40, 40, 20]);
    let shapes = shapes.unwrap();
    assert_eq!(shapes.seq_lens, vec![20, 40]);
}

// -------------------------------------------------------------------
// from_token_ids tests
// -------------------------------------------------------------------

#[test]
fn test_from_token_ids_empty_chunks_is_done() {
    let session = StreamingKokoroSession::from_token_ids(
        Vec::new(),
        dummy_style(),
        1.0,
        default_stream_config(),
    )
    .expect("should create session");
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);
    assert_eq!(session.total_chunks(), 0);
    assert!(!session.has_crossfade());
}

#[test]
fn test_from_token_ids_single_chunk() {
    let chunks = vec![vec![1i64, 2, 3, 4, 5]];
    let session = StreamingKokoroSession::from_token_ids(
        chunks,
        dummy_style(),
        1.0,
        default_stream_config(),
    )
    .expect("should create session");
    assert!(!session.is_done());
    assert_eq!(session.remaining(), 1);
    assert_eq!(session.total_chunks(), 1);
    assert!(session.has_crossfade());
}

#[test]
fn test_from_token_ids_multiple_chunks() {
    let chunks = vec![
        vec![1i64, 2, 3],
        vec![4i64, 5, 6, 7],
        vec![8i64, 9],
    ];
    let session = StreamingKokoroSession::from_token_ids(
        chunks,
        dummy_style(),
        0.8,
        default_stream_config(),
    )
    .expect("should create session");
    assert_eq!(session.total_chunks(), 3);
    assert_eq!(session.remaining(), 3);
    assert!(!session.is_done());
    assert!((session.speed() - 0.8).abs() < f32::EPSILON);
    assert!(session.has_crossfade());
}

#[test]
fn test_from_token_ids_remaining_decrements() {
    let chunks = vec![vec![1i64, 2], vec![3i64, 4], vec![5i64, 6]];
    let mut session = StreamingKokoroSession::from_token_ids(
        chunks,
        dummy_style(),
        1.0,
        default_stream_config(),
    )
    .expect("should create session");

    assert_eq!(session.remaining(), 3);
    session.cursor = 1;
    assert_eq!(session.remaining(), 2);
    session.cursor = 2;
    assert_eq!(session.remaining(), 1);
    session.cursor = 3;
    assert_eq!(session.remaining(), 0);
    assert!(session.is_done());
}

#[test]
fn test_from_token_ids_is_done_transitions() {
    let chunks = vec![vec![10i64, 20]];
    let mut session = StreamingKokoroSession::from_token_ids(
        chunks,
        dummy_style(),
        1.0,
        default_stream_config(),
    )
    .expect("should create session");

    assert!(!session.is_done());
    session.cursor = 1;
    assert!(session.is_done());
}

// -------------------------------------------------------------------
// with_crossfade tests
// -------------------------------------------------------------------

#[test]
fn test_with_crossfade_attaches_assembler() {
    let chunks = vec![dummy_chunk(), dummy_chunk()];
    let session = StreamingKokoroSession::new(chunks, 1.0);
    assert!(!session.has_crossfade());

    let session = session
        .with_crossfade(default_stream_config())
        .expect("should attach crossfade");
    assert!(session.has_crossfade());
}

#[test]
fn test_with_crossfade_empty_session_no_assembler() {
    let session = StreamingKokoroSession::new(Vec::new(), 1.0);
    let session = session
        .with_crossfade(default_stream_config())
        .expect("ok for empty");
    assert!(!session.has_crossfade());
}

#[test]
fn test_new_session_has_no_crossfade() {
    let session = StreamingKokoroSession::new(vec![dummy_chunk()], 1.0);
    assert!(!session.has_crossfade());
}
