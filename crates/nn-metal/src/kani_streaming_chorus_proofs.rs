// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`StreamingChorusSession`] state machine properties.
//!
//! Since `StreamingChorusSession` depends on `KokoroChorus`, `PipelineCache`,
//! and real GPU synthesis, these harnesses model the session as an abstract
//! state machine operating on symbolic inputs. We prove 10 properties:
//!
//! 1. **Cursor monotonicity**: cursor never decreases during `next_chunk()`.
//! 2. **Termination**: after `chunks.len()` calls, `next_chunk()` returns `None`.
//! 3. **Cancel terminates**: after `cancel()`, all subsequent calls return `None`.
//! 4. **Reset restores**: after `reset()`, cursor==0 and cancelled==false.
//! 5. **Remaining consistency**: `remaining() == chunks.len() - cursor` (uncancelled).
//! 6. **is_done correctness**: `is_done()` iff cursor >= len or cancelled.
//! 7. **Sample offset monotonicity**: sample_offset never decreases.
//! 8. **Crossfade state management**: prev_tail lifecycle is correct.
//! 9. **Speed mutation**: `set_speed` only affects future chunks (state flag).
//! 10. **Session uniqueness**: each `next_chunk()` advances cursor by exactly 1.

/// Abstract model of `StreamingChorusSession` for Kani proofs.
///
/// Strips out GPU synthesis, `DynTensor`, `KokoroChorus`, etc.
/// Retains only the state machine fields that govern control flow.
struct SessionModel {
    num_chunks: usize,
    cursor: usize,
    sample_offset: usize,
    prev_tail_len: usize, // models Vec<f32>.len()
    cancelled: bool,
    speed: f32,
}

impl SessionModel {
    /// Models `StreamingChorusSession::new()`.
    fn new(num_chunks: usize, speed: f32) -> Self {
        Self {
            num_chunks,
            cursor: 0,
            sample_offset: 0,
            prev_tail_len: 0,
            cancelled: false,
            speed,
        }
    }

    /// Models `next_chunk()` — returns `true` if a chunk was produced.
    ///
    /// The real method synthesizes audio and applies crossfade. We model
    /// only the state transitions: cursor advance, sample_offset growth,
    /// and prev_tail lifecycle.
    fn next_chunk(&mut self, chunk_pcm_len: usize) -> bool {
        if self.cancelled || self.cursor >= self.num_chunks {
            return false;
        }

        let idx = self.cursor;
        self.cursor += 1;
        let is_first = idx == 0;
        let is_last = idx == self.num_chunks - 1;

        // Model crossfade emit: non-last chunks exclude a crossfade tail.
        // We use a symbolic crossfade size of 0 for simplicity — the key
        // property is that sample_offset grows by the emitted length.
        let emit_len = chunk_pcm_len;
        self.sample_offset += emit_len;

        // Model prev_tail lifecycle.
        self.prev_tail_len = if is_last { 0 } else { chunk_pcm_len.min(64) };

        // First chunk has no incoming crossfade (prev_tail was empty).
        // Between chunks, prev_tail is non-empty (set above for !is_last).
        // After last chunk, prev_tail is empty (set to 0 above).
        let _ = is_first; // used for documentation clarity

        true
    }

    /// Models `remaining()`.
    fn remaining(&self) -> usize {
        if self.cancelled {
            return 0;
        }
        self.num_chunks.saturating_sub(self.cursor)
    }

    /// Models `is_done()`.
    fn is_done(&self) -> bool {
        self.cancelled || self.cursor >= self.num_chunks
    }

    /// Models `cancel()`.
    fn cancel(&mut self) {
        self.cancelled = true;
        self.prev_tail_len = 0;
    }

    /// Models `reset()`.
    fn reset(&mut self) {
        self.cursor = 0;
        self.sample_offset = 0;
        self.prev_tail_len = 0;
        self.cancelled = false;
    }

    /// Models `set_speed()`.
    fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }
}

// ============================================================================
// 1. Cursor monotonicity: cursor never decreases during next_chunk()
// ============================================================================

/// Prove: each `next_chunk()` call either leaves the cursor unchanged
/// (when returning None) or increments it by 1. The cursor never decreases.
#[kani::proof]
#[kani::unwind(7)]
fn proof_cursor_monotonicity() {
    let num_chunks: usize = kani::any();
    kani::assume(num_chunks > 0 && num_chunks <= 5);

    let mut session = SessionModel::new(num_chunks, 1.0);
    let mut prev_cursor: usize = 0;

    let steps: usize = kani::any();
    kani::assume(steps <= num_chunks + 1); // allow one extra call past end

    let mut i: usize = 0;
    while i < steps {
        prev_cursor = session.cursor;
        let pcm_len: usize = kani::any();
        kani::assume(pcm_len > 0 && pcm_len <= 100);
        session.next_chunk(pcm_len);
        assert!(
            session.cursor >= prev_cursor,
            "cursor must never decrease after next_chunk()"
        );
        i += 1;
    }
}

// ============================================================================
// 2. Termination: after chunks.len() calls, next_chunk() returns None
// ============================================================================

/// Prove: after exactly `num_chunks` calls to `next_chunk()`, the session
/// is done and subsequent calls return `false` (None).
#[kani::proof]
#[kani::unwind(7)]
fn proof_termination_after_all_chunks() {
    let num_chunks: usize = kani::any();
    kani::assume(num_chunks > 0 && num_chunks <= 5);

    let mut session = SessionModel::new(num_chunks, 1.0);

    // Consume all chunks.
    let mut i: usize = 0;
    while i < num_chunks {
        let produced = session.next_chunk(100);
        assert!(produced, "chunk {i} must be produced");
        i += 1;
    }

    // Session should now be done.
    assert!(session.is_done(), "session must be done after all chunks consumed");
    assert_eq!(session.cursor, num_chunks, "cursor must equal num_chunks");

    // Next call must return false (None).
    let extra = session.next_chunk(100);
    assert!(!extra, "next_chunk after exhaustion must return None");
}

// ============================================================================
// 3. Cancel terminates: after cancel(), all subsequent calls return None
// ============================================================================

/// Prove: once `cancel()` is called, every subsequent `next_chunk()` returns
/// `false` (None), regardless of how many chunks remain.
#[kani::proof]
#[kani::unwind(7)]
fn proof_cancel_terminates_session() {
    let num_chunks: usize = kani::any();
    kani::assume(num_chunks > 0 && num_chunks <= 5);

    let mut session = SessionModel::new(num_chunks, 1.0);

    // Consume some chunks before cancelling.
    let consume_before: usize = kani::any();
    kani::assume(consume_before < num_chunks);
    let mut i: usize = 0;
    while i < consume_before {
        session.next_chunk(100);
        i += 1;
    }

    // Cancel.
    session.cancel();
    assert!(session.cancelled, "session must be cancelled");
    assert!(session.is_done(), "cancelled session must report is_done");

    // All subsequent calls return None.
    let mut j: usize = 0;
    while j < 3 {
        let produced = session.next_chunk(100);
        assert!(!produced, "next_chunk after cancel must return None");
        j += 1;
    }

    // Cursor is frozen at cancellation point.
    assert_eq!(
        session.cursor, consume_before,
        "cursor must not advance after cancel"
    );
}

// ============================================================================
// 4. Reset restores: after reset(), cursor==0 and cancelled==false
// ============================================================================

/// Prove: `reset()` restores the session to its initial state regardless
/// of how many chunks were consumed or whether it was cancelled.
#[kani::proof]
#[kani::unwind(7)]
fn proof_reset_restores_initial_state() {
    let num_chunks: usize = kani::any();
    kani::assume(num_chunks > 0 && num_chunks <= 5);

    let mut session = SessionModel::new(num_chunks, 1.0);

    // Drive session to an arbitrary state.
    let consume: usize = kani::any();
    kani::assume(consume <= num_chunks);
    let mut i: usize = 0;
    while i < consume {
        session.next_chunk(100);
        i += 1;
    }

    let do_cancel: bool = kani::any();
    if do_cancel {
        session.cancel();
    }

    // Reset.
    session.reset();

    // Verify initial state.
    assert_eq!(session.cursor, 0, "cursor must be 0 after reset");
    assert_eq!(session.sample_offset, 0, "sample_offset must be 0 after reset");
    assert_eq!(session.prev_tail_len, 0, "prev_tail must be empty after reset");
    assert!(!session.cancelled, "cancelled must be false after reset");

    // Session should be usable again.
    assert!(!session.is_done(), "reset session must not be done");
    assert_eq!(session.remaining(), num_chunks, "remaining must equal total after reset");
}

// ============================================================================
// 5. Remaining consistency: remaining() == chunks.len() - cursor (uncancelled)
// ============================================================================

/// Prove: `remaining()` always equals `num_chunks - cursor` when uncancelled,
/// and 0 when cancelled.
#[kani::proof]
#[kani::unwind(7)]
fn proof_remaining_consistency() {
    let num_chunks: usize = kani::any();
    kani::assume(num_chunks > 0 && num_chunks <= 5);

    let mut session = SessionModel::new(num_chunks, 1.0);

    // Check at initial state.
    assert_eq!(
        session.remaining(),
        num_chunks,
        "remaining must be num_chunks at start"
    );

    // Step through and verify at each point.
    let mut i: usize = 0;
    while i < num_chunks {
        session.next_chunk(100);
        let expected = num_chunks - session.cursor;
        assert_eq!(
            session.remaining(),
            expected,
            "remaining must equal num_chunks - cursor"
        );
        i += 1;
    }

    // After all consumed.
    assert_eq!(session.remaining(), 0, "remaining must be 0 when exhausted");

    // After cancel at any point.
    session.reset();
    let consume_some: usize = kani::any();
    kani::assume(consume_some <= num_chunks);
    let mut j: usize = 0;
    while j < consume_some {
        session.next_chunk(100);
        j += 1;
    }
    session.cancel();
    assert_eq!(session.remaining(), 0, "remaining must be 0 when cancelled");
}

// ============================================================================
// 6. is_done correctness: is_done() iff cursor >= len or cancelled
// ============================================================================

/// Prove: `is_done()` returns `true` if and only if the cursor has reached
/// the end or the session was cancelled. No other condition makes it true.
#[kani::proof]
#[kani::unwind(7)]
fn proof_is_done_correctness() {
    let num_chunks: usize = kani::any();
    kani::assume(num_chunks > 0 && num_chunks <= 5);

    let mut session = SessionModel::new(num_chunks, 1.0);

    // Not done at start.
    assert!(!session.is_done(), "fresh session must not be done");

    // Step through.
    let mut i: usize = 0;
    while i < num_chunks {
        let expected_done = session.cursor >= session.num_chunks || session.cancelled;
        assert_eq!(session.is_done(), expected_done, "is_done must match predicate");
        session.next_chunk(100);
        i += 1;
    }

    // Done after all chunks consumed.
    assert!(session.is_done(), "session must be done after all chunks");

    // Reset and cancel mid-way.
    session.reset();
    let partial: usize = kani::any();
    kani::assume(partial < num_chunks);
    let mut j: usize = 0;
    while j < partial {
        session.next_chunk(100);
        j += 1;
    }
    assert!(!session.is_done(), "partially consumed session is not done");
    session.cancel();
    assert!(session.is_done(), "cancelled session is done");
}

// ============================================================================
// 7. Sample offset monotonicity: sample_offset never decreases
// ============================================================================

/// Prove: `sample_offset` is monotonically non-decreasing across all
/// `next_chunk()` calls. Each call adds a positive emit length.
#[kani::proof]
#[kani::unwind(7)]
fn proof_sample_offset_monotonicity() {
    let num_chunks: usize = kani::any();
    kani::assume(num_chunks > 0 && num_chunks <= 5);

    let mut session = SessionModel::new(num_chunks, 1.0);

    let mut i: usize = 0;
    while i < num_chunks {
        let prev_offset = session.sample_offset;
        let pcm_len: usize = kani::any();
        kani::assume(pcm_len > 0 && pcm_len <= 1000);
        session.next_chunk(pcm_len);
        assert!(
            session.sample_offset >= prev_offset,
            "sample_offset must never decrease"
        );
        assert!(
            session.sample_offset > prev_offset,
            "sample_offset must strictly increase when a chunk is produced"
        );
        i += 1;
    }

    // After exhaustion, sample_offset stays constant.
    let final_offset = session.sample_offset;
    session.next_chunk(100); // returns false
    assert_eq!(
        session.sample_offset, final_offset,
        "sample_offset must not change when next_chunk returns None"
    );
}

// ============================================================================
// 8. Crossfade state management: prev_tail lifecycle
// ============================================================================

/// Prove: `prev_tail` follows the lifecycle:
/// - Empty before the first chunk (initial state).
/// - Non-empty between chunks (after producing a non-last chunk).
/// - Empty after the last chunk.
#[kani::proof]
#[kani::unwind(7)]
fn proof_crossfade_state_lifecycle() {
    let num_chunks: usize = kani::any();
    kani::assume(num_chunks >= 2 && num_chunks <= 5);

    let mut session = SessionModel::new(num_chunks, 1.0);

    // Before first chunk: prev_tail is empty.
    assert_eq!(session.prev_tail_len, 0, "prev_tail must be empty initially");

    // After first chunk (not last since num_chunks >= 2): prev_tail non-empty.
    session.next_chunk(100);
    assert!(
        session.prev_tail_len > 0,
        "prev_tail must be non-empty after first (non-last) chunk"
    );

    // Middle chunks: prev_tail stays non-empty.
    let mut i: usize = 1;
    while i < num_chunks - 1 {
        session.next_chunk(100);
        assert!(
            session.prev_tail_len > 0,
            "prev_tail must be non-empty between chunks"
        );
        i += 1;
    }

    // After last chunk: prev_tail is empty.
    session.next_chunk(100);
    assert_eq!(
        session.prev_tail_len, 0,
        "prev_tail must be empty after last chunk"
    );
}

// ============================================================================
// 9. Speed mutation: set_speed only affects future state (modeled as flag)
// ============================================================================

/// Prove: `set_speed()` changes the speed field but does not affect any
/// other state machine field (cursor, sample_offset, prev_tail, cancelled).
#[kani::proof]
#[kani::unwind(7)]
fn proof_set_speed_isolation() {
    let num_chunks: usize = kani::any();
    kani::assume(num_chunks > 0 && num_chunks <= 5);

    let initial_speed: f32 = kani::any();
    kani::assume(initial_speed.is_finite() && initial_speed > 0.0);
    let mut session = SessionModel::new(num_chunks, initial_speed);

    // Consume some chunks.
    let consume: usize = kani::any();
    kani::assume(consume <= num_chunks);
    let mut i: usize = 0;
    while i < consume {
        session.next_chunk(100);
        i += 1;
    }

    // Snapshot state before speed change.
    let cursor_before = session.cursor;
    let sample_offset_before = session.sample_offset;
    let prev_tail_before = session.prev_tail_len;
    let cancelled_before = session.cancelled;

    // Change speed.
    let new_speed: f32 = kani::any();
    kani::assume(new_speed.is_finite() && new_speed > 0.0);
    session.set_speed(new_speed);

    // Speed changed.
    assert_eq!(session.speed, new_speed, "speed must be updated");

    // All other state unchanged.
    assert_eq!(session.cursor, cursor_before, "cursor must not change on set_speed");
    assert_eq!(
        session.sample_offset, sample_offset_before,
        "sample_offset must not change on set_speed"
    );
    assert_eq!(
        session.prev_tail_len, prev_tail_before,
        "prev_tail must not change on set_speed"
    );
    assert_eq!(
        session.cancelled, cancelled_before,
        "cancelled must not change on set_speed"
    );
}

// ============================================================================
// 10. Session uniqueness: each next_chunk() advances cursor by exactly 1
// ============================================================================

/// Prove: every successful `next_chunk()` call advances the cursor by
/// exactly 1. Failed calls (returning None) do not advance the cursor.
#[kani::proof]
#[kani::unwind(9)]
fn proof_cursor_advances_by_exactly_one() {
    let num_chunks: usize = kani::any();
    kani::assume(num_chunks > 0 && num_chunks <= 5);

    let mut session = SessionModel::new(num_chunks, 1.0);

    // Test all chunks + 2 extra calls past the end.
    let total_calls = num_chunks + 2;
    let mut i: usize = 0;
    while i < total_calls {
        let cursor_before = session.cursor;
        let produced = session.next_chunk(100);

        if produced {
            assert_eq!(
                session.cursor,
                cursor_before + 1,
                "successful next_chunk must advance cursor by exactly 1"
            );
        } else {
            assert_eq!(
                session.cursor, cursor_before,
                "failed next_chunk must not advance cursor"
            );
        }
        i += 1;
    }

    // Final cursor must equal num_chunks (one increment per chunk).
    assert_eq!(
        session.cursor, num_chunks,
        "cursor must equal num_chunks after exhaustion"
    );
}
