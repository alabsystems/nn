// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `StreamingChorusSession` state machine and crossfade logic.
//!
//! These tests verify the state machine invariants without requiring a real
//! `KokoroChorus` (which needs GPU). GPU-dependent integration tests are
//! in `tests/kokoro/`.

use super::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_models::kokoro_streaming::KokoroStreamConfig;

/// Helper: create a dummy `DynTensor` for state-machine tests.
fn dummy_chunk_ids() -> DynTensor {
    DynTensor::zeros(&[1, 10], DType::I64, &Device::Cpu).unwrap()
}

/// Helper: create a dummy `DynTensor` with a specific token length.
fn dummy_chunk_ids_with_len(len: usize) -> DynTensor {
    DynTensor::zeros(&[1, len], DType::I64, &Device::Cpu).unwrap()
}

/// Helper: create a dummy style tensor.
fn dummy_style() -> DynTensor {
    DynTensor::zeros(&[1, 512], DType::F32, &Device::Cpu).unwrap()
}

// -- Construction tests -------------------------------------------------------

#[test]
fn test_new_empty_session() {
    let session = StreamingChorusSession::new(
        Vec::new(),
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);
    assert_eq!(session.total_chunks(), 0);
    assert_eq!(session.synthesized_count(), 0);
    assert!(!session.is_cancelled());
}

#[test]
fn test_new_with_chunks() {
    let chunks = vec![dummy_chunk_ids(), dummy_chunk_ids(), dummy_chunk_ids()];
    let styles = vec![dummy_style(), dummy_style()];
    let session =
        StreamingChorusSession::new(chunks, styles, 1.5, KokoroStreamConfig::default()).unwrap();
    assert!(!session.is_done());
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.total_chunks(), 3);
    assert_eq!(session.synthesized_count(), 0);
    assert!((session.speed() - 1.5).abs() < f32::EPSILON);
}

#[test]
fn test_new_invalid_stream_config() {
    // KokoroStreamConfig::new(0) returns Err because crossfade_samples must be > 0.
    let bad_config = KokoroStreamConfig::new(0);
    assert!(
        bad_config.is_err(),
        "config with crossfade_samples=0 should fail"
    );

    // Even if we somehow get a bad config past validation, StreamingChorusSession::new
    // would catch it. Verify by using a valid config that works.
    let ok_result = StreamingChorusSession::new(
        vec![dummy_chunk_ids()],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    );
    assert!(ok_result.is_ok());
}

// -- Cancel tests -------------------------------------------------------------

#[test]
fn test_cancel_stops_iteration() {
    let chunks = vec![dummy_chunk_ids(), dummy_chunk_ids()];
    let mut session = StreamingChorusSession::new(
        chunks,
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    assert!(!session.is_cancelled());
    assert_eq!(session.remaining(), 2);

    session.cancel();
    assert!(session.is_cancelled());
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);
}

// -- Reset tests --------------------------------------------------------------

#[test]
fn test_reset_restores_initial_state() {
    let chunks = vec![dummy_chunk_ids(), dummy_chunk_ids()];
    let mut session = StreamingChorusSession::new(
        chunks,
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    // Simulate advancing the cursor.
    session.cursor = 2;
    assert!(session.is_done());

    session.reset();
    assert!(!session.is_done());
    assert_eq!(session.remaining(), 2);
    assert_eq!(session.synthesized_count(), 0);
    assert!(session.assembler.is_none(), "reset clears the assembler");
    assert!(!session.is_cancelled());
}

#[test]
fn test_reset_after_cancel() {
    let chunks = vec![dummy_chunk_ids()];
    let mut session = StreamingChorusSession::new(
        chunks,
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    session.cancel();
    assert!(session.is_cancelled());
    assert!(session.is_done());

    session.reset();
    assert!(!session.is_cancelled());
    assert!(!session.is_done());
    assert_eq!(session.remaining(), 1);
}

// -- Speed tests --------------------------------------------------------------

#[test]
fn test_set_speed() {
    let mut session = StreamingChorusSession::new(
        vec![dummy_chunk_ids()],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    assert!((session.speed() - 1.0).abs() < f32::EPSILON);
    session.set_speed(0.8);
    assert!((session.speed() - 0.8).abs() < f32::EPSILON);
}

// -- Precompile tests ---------------------------------------------------------

#[test]
fn test_precompile_enabled_by_default() {
    let session = StreamingChorusSession::new(
        vec![dummy_chunk_ids()],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    assert!(session.precompile_enabled());
    assert!(session.precompile_pending());
    assert!(!session.precompile_consumed());
}

#[test]
fn test_precompile_disabled_via_builder() {
    let session = StreamingChorusSession::new(
        vec![dummy_chunk_ids()],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap()
    .with_precompile(false);
    assert!(!session.precompile_enabled());
    assert!(!session.precompile_pending());
    assert!(session.pending_precompile_shapes().is_none());
}

#[test]
fn test_pending_precompile_shapes_shared_text() {
    let session = StreamingChorusSession::new(
        vec![
            dummy_chunk_ids_with_len(10),
            dummy_chunk_ids_with_len(20),
            dummy_chunk_ids_with_len(40),
        ],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    let shapes = session
        .pending_precompile_shapes()
        .expect("shared-text chunks should plan warmup shapes");
    assert_eq!(shapes.seq_lens, vec![10, 20, 40]);
    assert!(!shapes.t_mels.is_empty());
}

#[test]
fn test_pending_precompile_shapes_shared_text_skips_consumed_chunks() {
    let mut session = StreamingChorusSession::new(
        vec![
            dummy_chunk_ids_with_len(10),
            dummy_chunk_ids_with_len(20),
            dummy_chunk_ids_with_len(40),
        ],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    session.cursor = 1;

    let shapes = session
        .pending_precompile_shapes()
        .expect("remaining shared-text chunks should still plan warmup shapes");
    assert_eq!(shapes.seq_lens, vec![20, 40]);
    assert!(!shapes.t_mels.is_empty());
}

#[test]
fn test_pending_precompile_shapes_per_voice_deduplicates_lengths() {
    let session = StreamingChorusSession::new_varied_text(
        vec![
            vec![dummy_chunk_ids_with_len(20), dummy_chunk_ids_with_len(40)],
            vec![dummy_chunk_ids_with_len(40), dummy_chunk_ids_with_len(20)],
        ],
        vec![dummy_style(), dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    let shapes = session
        .pending_precompile_shapes()
        .expect("per-voice chunks should plan deduplicated warmup shapes");
    assert_eq!(shapes.seq_lens, vec![20, 40]);
    assert!(!shapes.t_mels.is_empty());
}

#[test]
fn test_pending_precompile_shapes_per_voice_skips_consumed_chunks() {
    let mut session = StreamingChorusSession::new_varied_text(
        vec![
            vec![
                dummy_chunk_ids_with_len(10),
                dummy_chunk_ids_with_len(20),
                dummy_chunk_ids_with_len(40),
            ],
            vec![
                dummy_chunk_ids_with_len(20),
                dummy_chunk_ids_with_len(40),
                dummy_chunk_ids_with_len(80),
            ],
        ],
        vec![dummy_style(), dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    session.cursor = 1;

    let shapes = session
        .pending_precompile_shapes()
        .expect("remaining per-voice chunks should still plan warmup shapes");
    assert_eq!(shapes.seq_lens, vec![20, 40, 80]);
    assert!(!shapes.t_mels.is_empty());
}

#[test]
fn test_pending_precompile_shapes_none_after_consumed() {
    let mut session = StreamingChorusSession::new(
        vec![dummy_chunk_ids_with_len(10)],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    session.precompiled = true;

    assert!(!session.precompile_pending());
    assert!(session.pending_precompile_shapes().is_none());
}

#[test]
fn test_pending_precompile_shapes_none_after_all_chunks_consumed() {
    let mut session = StreamingChorusSession::new(
        vec![dummy_chunk_ids_with_len(10), dummy_chunk_ids_with_len(20)],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    session.cursor = 2;

    assert!(!session.precompile_pending());
    assert!(session.pending_precompile_shapes().is_none());
}

#[test]
fn test_precompile_consumed_persists_across_reset() {
    let mut session = StreamingChorusSession::new(
        vec![dummy_chunk_ids_with_len(10)],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    session.precompiled = true;

    assert!(session.precompile_consumed());

    session.reset();

    assert!(session.precompile_consumed());
    assert!(!session.precompile_pending());
    assert!(session.pending_precompile_shapes().is_none());
}

// -- Remaining count tests ----------------------------------------------------

#[test]
fn test_remaining_counts_down_with_cursor() {
    let chunks = vec![dummy_chunk_ids(), dummy_chunk_ids(), dummy_chunk_ids()];
    let mut session = StreamingChorusSession::new(
        chunks,
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    assert_eq!(session.remaining(), 3);
    assert_eq!(session.synthesized_count(), 0);

    session.cursor = 1;
    assert_eq!(session.remaining(), 2);
    assert_eq!(session.synthesized_count(), 1);

    session.cursor = 3;
    assert_eq!(session.remaining(), 0);
    assert_eq!(session.synthesized_count(), 3);
    assert!(session.is_done());
}

#[test]
fn test_remaining_saturates_at_zero() {
    let mut session = StreamingChorusSession::new(
        vec![dummy_chunk_ids()],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    session.cursor = 100;
    assert_eq!(session.remaining(), 0);
    assert!(session.is_done());
}

// -- Cancel + reset cycle -------------------------------------------------------

#[test]
fn test_cancel_reset_cycle() {
    let chunks = vec![dummy_chunk_ids(), dummy_chunk_ids(), dummy_chunk_ids()];
    let mut session = StreamingChorusSession::new(
        chunks,
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    // Advance cursor partway.
    session.cursor = 1;
    assert_eq!(session.remaining(), 2);
    assert_eq!(session.synthesized_count(), 1);

    // Cancel mid-stream.
    session.cancel();
    assert!(session.is_cancelled());
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);
    assert!(session.assembler.is_none(), "cancel clears assembler");

    // Reset restores everything.
    session.reset();
    assert!(!session.is_cancelled());
    assert!(!session.is_done());
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.synthesized_count(), 0);
    assert!(session.assembler.is_none(), "reset clears assembler");
}

#[test]
fn test_double_cancel_is_idempotent() {
    let mut session = StreamingChorusSession::new(
        vec![dummy_chunk_ids()],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    session.cancel();
    assert!(session.is_cancelled());

    // Second cancel should not panic or change state.
    session.cancel();
    assert!(session.is_cancelled());
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);
}

#[test]
fn test_double_reset_is_idempotent() {
    let mut session = StreamingChorusSession::new(
        vec![dummy_chunk_ids(), dummy_chunk_ids()],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    session.cursor = 2;
    session.reset();
    assert_eq!(session.remaining(), 2);

    // Second reset should not panic or change state.
    session.reset();
    assert_eq!(session.remaining(), 2);
    assert!(!session.is_cancelled());
}

#[test]
fn test_cancel_reset_cancel_cycle() {
    let mut session = StreamingChorusSession::new(
        vec![dummy_chunk_ids(), dummy_chunk_ids()],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    session.cancel();
    assert!(session.is_cancelled());

    session.reset();
    assert!(!session.is_cancelled());
    assert_eq!(session.remaining(), 2);

    // Cancel again after reset.
    session.cancel();
    assert!(session.is_cancelled());
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);

    // And reset again.
    session.reset();
    assert!(!session.is_cancelled());
    assert_eq!(session.remaining(), 2);
}

#[test]
fn test_speed_preserved_across_reset() {
    let mut session = StreamingChorusSession::new(
        vec![dummy_chunk_ids()],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    session.set_speed(2.0);
    session.cancel();
    session.reset();

    // Speed should be preserved (reset does not touch speed).
    assert!((session.speed() - 2.0).abs() < f32::EPSILON);
}

// -- ChorusChunkMode tests ---------------------------------------------------

#[test]
fn test_chunk_mode_shared_text_len() {
    let mode = ChorusChunkMode::SharedText(vec![
        dummy_chunk_ids(),
        dummy_chunk_ids(),
        dummy_chunk_ids(),
    ]);
    assert_eq!(mode.len(), 3);
    assert!(!mode.is_empty());
}

#[test]
fn test_chunk_mode_shared_text_empty() {
    let mode = ChorusChunkMode::SharedText(Vec::new());
    assert_eq!(mode.len(), 0);
    assert!(mode.is_empty());
}

#[test]
fn test_chunk_mode_per_voice_len() {
    let mode = ChorusChunkMode::PerVoice(vec![
        vec![dummy_chunk_ids(), dummy_chunk_ids()],
        vec![dummy_chunk_ids(), dummy_chunk_ids()],
    ]);
    assert_eq!(mode.len(), 2);
    assert!(!mode.is_empty());
}

#[test]
fn test_chunk_mode_per_voice_empty_outer() {
    let mode = ChorusChunkMode::PerVoice(Vec::new());
    assert_eq!(mode.len(), 0);
    assert!(mode.is_empty());
}

#[test]
fn test_chunk_mode_shared_text_token_lengths() {
    let mode = ChorusChunkMode::SharedText(vec![
        dummy_chunk_ids_with_len(10),
        dummy_chunk_ids_with_len(20),
        dummy_chunk_ids_with_len(40),
    ]);
    assert_eq!(mode.token_lengths(), vec![10, 20, 40]);
}

#[test]
fn test_chunk_mode_per_voice_token_lengths_flattens_all_voices() {
    let mode = ChorusChunkMode::PerVoice(vec![
        vec![dummy_chunk_ids_with_len(10), dummy_chunk_ids_with_len(20)],
        vec![dummy_chunk_ids_with_len(20), dummy_chunk_ids_with_len(40)],
    ]);
    assert_eq!(mode.token_lengths(), vec![10, 20, 20, 40]);
}

#[test]
fn test_precompile_shapes_from_per_voice_chunk_lengths_deduplicates() {
    let mode = ChorusChunkMode::PerVoice(vec![
        vec![dummy_chunk_ids_with_len(20), dummy_chunk_ids_with_len(40)],
        vec![dummy_chunk_ids_with_len(40), dummy_chunk_ids_with_len(20)],
    ]);
    let shapes = PrecompileShapes::from_token_lengths(&mode.token_lengths())
        .expect("per-voice lengths should produce precompile shapes");
    assert_eq!(shapes.seq_lens, vec![20, 40]);
    assert!(!shapes.t_mels.is_empty());
}

// -- new_varied_text tests ---------------------------------------------------

#[test]
fn test_new_varied_text_two_voices_three_chunks() {
    let per_voice = vec![
        vec![dummy_chunk_ids(), dummy_chunk_ids(), dummy_chunk_ids()],
        vec![dummy_chunk_ids(), dummy_chunk_ids(), dummy_chunk_ids()],
    ];
    let styles = vec![dummy_style(), dummy_style()];
    let session = StreamingChorusSession::new_varied_text(
        per_voice,
        styles,
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    assert!(session.is_varied_text());
    assert!(!session.is_done());
    assert_eq!(session.total_chunks(), 3);
    assert_eq!(session.remaining(), 3);
    assert_eq!(session.synthesized_count(), 0);
}

#[test]
fn test_new_varied_text_mismatched_chunk_counts() {
    let per_voice = vec![
        vec![dummy_chunk_ids(), dummy_chunk_ids(), dummy_chunk_ids()],
        vec![dummy_chunk_ids(), dummy_chunk_ids()], // only 2 chunks
    ];
    let styles = vec![dummy_style(), dummy_style()];
    let result = StreamingChorusSession::new_varied_text(
        per_voice,
        styles,
        1.0,
        KokoroStreamConfig::default(),
    );
    assert!(result.is_err());
    let err_msg = result.err().expect("expected error").to_string();
    assert!(
        err_msg.contains("equal chunk counts"),
        "expected chunk count mismatch error, got: {err_msg}"
    );
}

#[test]
fn test_new_varied_text_mismatched_styles_length() {
    let per_voice = vec![vec![dummy_chunk_ids()], vec![dummy_chunk_ids()]];
    let styles = vec![dummy_style()]; // only 1 style for 2 voices
    let result = StreamingChorusSession::new_varied_text(
        per_voice,
        styles,
        1.0,
        KokoroStreamConfig::default(),
    );
    assert!(result.is_err());
    let err_msg = result.err().expect("expected error").to_string();
    assert!(
        err_msg.contains("styles.len()"),
        "expected styles length mismatch error, got: {err_msg}"
    );
}

#[test]
fn test_new_varied_text_empty_per_voice() {
    let result = StreamingChorusSession::new_varied_text(
        Vec::new(),
        Vec::new(),
        1.0,
        KokoroStreamConfig::default(),
    );
    assert!(result.is_err());
    let err_msg = result.err().expect("expected error").to_string();
    assert!(
        err_msg.contains("must not be empty"),
        "expected empty error, got: {err_msg}"
    );
}

#[test]
fn test_is_varied_text_shared() {
    let session = StreamingChorusSession::new(
        vec![dummy_chunk_ids()],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    assert!(!session.is_varied_text());
}

#[test]
fn test_is_varied_text_per_voice() {
    let session = StreamingChorusSession::new_varied_text(
        vec![vec![dummy_chunk_ids()]],
        vec![dummy_style()],
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();
    assert!(session.is_varied_text());
}

// -- Varied-text state machine tests -----------------------------------------

#[test]
fn test_varied_text_remaining_counts_down() {
    let per_voice = vec![vec![
        dummy_chunk_ids(),
        dummy_chunk_ids(),
        dummy_chunk_ids(),
    ]];
    let styles = vec![dummy_style()];
    let mut session = StreamingChorusSession::new_varied_text(
        per_voice,
        styles,
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    assert_eq!(session.remaining(), 3);
    session.cursor = 1;
    assert_eq!(session.remaining(), 2);
    session.cursor = 3;
    assert_eq!(session.remaining(), 0);
    assert!(session.is_done());
}

#[test]
fn test_varied_text_cancel_and_reset() {
    let per_voice = vec![
        vec![dummy_chunk_ids(), dummy_chunk_ids()],
        vec![dummy_chunk_ids(), dummy_chunk_ids()],
    ];
    let styles = vec![dummy_style(), dummy_style()];
    let mut session = StreamingChorusSession::new_varied_text(
        per_voice,
        styles,
        1.0,
        KokoroStreamConfig::default(),
    )
    .unwrap();

    session.cursor = 1;
    session.cancel();
    assert!(session.is_cancelled());
    assert!(session.is_done());
    assert_eq!(session.remaining(), 0);

    session.reset();
    assert!(!session.is_cancelled());
    assert!(!session.is_done());
    assert_eq!(session.remaining(), 2);
    assert!(session.is_varied_text());
}
