// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for streaming Kokoro synthesis session invariants.
//!
//! Models a streaming synthesis session as an abstract state machine and proves
//! 8 properties about its behavior:
//!
//! 1. **Chunk index monotonicity** — chunk index only increases.
//! 2. **Termination guarantee** — session completes after processing all chunks.
//! 3. **Sample count consistency** — total samples = sum of per-chunk samples.
//! 4. **Cancel halts progress** — cancellation prevents further chunk processing.
//! 5. **Completed state is absorbing** — once completed, no more chunks process.
//! 6. **Chunk size bounds** — each chunk produces samples within [min, max].
//! 7. **Session progress fraction** — progress = chunks_done / total is in [0, 1].
//! 8. **Empty input handling** — zero-length text produces zero chunks.
//!
//! Since `StreamingKokoroSession` depends on `DynTensor` and GPU synthesis for
//! actual audio generation, these harnesses model the session as an abstract
//! state machine operating on symbolic inputs. The model captures the cursor,
//! sample accumulation, cancellation, and completion semantics that any
//! correct streaming session must satisfy.
//!
//! Part of #3351.

/// Abstract model of a streaming Kokoro synthesis session for Kani proofs.
///
/// Strips out `DynTensor`, `CompiledKokoro`, `PipelineCache`, and GPU
/// synthesis. Retains only the state machine fields that govern streaming
/// control flow and sample accounting.
struct StreamingSessionModel {
    /// Total number of chunks to synthesize.
    total_chunks: usize,
    /// Current chunk cursor. When `cursor >= total_chunks`, the session is done.
    cursor: usize,
    /// Accumulated sample count across all synthesized chunks.
    total_samples: usize,
    /// Per-chunk sample counts (recorded for consistency verification).
    per_chunk_samples: [usize; 8],
    /// Whether the session has been cancelled.
    cancelled: bool,
}

impl StreamingSessionModel {
    /// Create a new session model with `n` chunks.
    fn new(n: usize) -> Self {
        Self {
            total_chunks: n,
            cursor: 0,
            total_samples: 0,
            per_chunk_samples: [0; 8],
            cancelled: false,
        }
    }

    /// Model `next_chunk()`: returns the number of samples produced, or 0 if
    /// the session is done or cancelled.
    fn next_chunk(&mut self, chunk_samples: usize) -> usize {
        if self.cancelled || self.cursor >= self.total_chunks {
            return 0;
        }
        let idx = self.cursor;
        self.per_chunk_samples[idx] = chunk_samples;
        self.total_samples += chunk_samples;
        self.cursor += 1;
        chunk_samples
    }

    /// Whether all chunks have been consumed or the session was cancelled.
    fn is_done(&self) -> bool {
        self.cancelled || self.cursor >= self.total_chunks
    }

    /// Number of remaining chunks.
    fn remaining(&self) -> usize {
        if self.cancelled {
            return 0;
        }
        self.total_chunks.saturating_sub(self.cursor)
    }

    /// Cancel the session, preventing further chunk processing.
    fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Compute progress as a fraction: chunks_done / total_chunks.
    /// Returns (numerator, denominator) to avoid floating-point in Kani.
    fn progress(&self) -> (usize, usize) {
        (self.cursor, self.total_chunks)
    }
}

// ============================================================================
// 1. Chunk index monotonicity — chunk index only increases
// ============================================================================

/// Prove: the cursor (chunk index) never decreases across `next_chunk()` calls.
/// Each call either increments the cursor by 1 or leaves it unchanged.
#[kani::proof]
#[kani::unwind(9)]
fn proof_chunk_index_monotonicity() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 6);

    let mut session = StreamingSessionModel::new(n);

    let steps: usize = kani::any();
    kani::assume(steps <= n + 2);

    let mut i: usize = 0;
    while i < steps {
        let prev_cursor = session.cursor;
        let samples: usize = kani::any();
        kani::assume(samples > 0 && samples <= 48000);
        session.next_chunk(samples);
        assert!(
            session.cursor >= prev_cursor,
            "chunk index must never decrease"
        );
        i += 1;
    }
}

// ============================================================================
// 2. Termination guarantee — session completes after processing all chunks
// ============================================================================

/// Prove: after exactly `total_chunks` calls to `next_chunk()`, the session
/// reports `is_done() == true` and subsequent calls produce 0 samples.
#[kani::proof]
#[kani::unwind(9)]
fn proof_termination_after_all_chunks() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 6);

    let mut session = StreamingSessionModel::new(n);

    // Consume all chunks.
    let mut i: usize = 0;
    while i < n {
        let produced = session.next_chunk(100);
        assert!(produced > 0, "chunk must produce samples before exhaustion");
        i += 1;
    }

    // Session must now be done.
    assert!(session.is_done(), "session must be done after all chunks");
    assert_eq!(session.cursor, n, "cursor must equal total after exhaustion");
    assert_eq!(session.remaining(), 0, "remaining must be 0 after exhaustion");

    // Subsequent calls produce 0.
    let extra = session.next_chunk(100);
    assert_eq!(extra, 0, "next_chunk after exhaustion must produce 0 samples");

    // Cursor does not advance past total.
    assert_eq!(session.cursor, n, "cursor must not advance past total");
}

// ============================================================================
// 3. Sample count consistency — total samples = sum of per-chunk samples
// ============================================================================

/// Prove: `total_samples` always equals the sum of `per_chunk_samples[0..cursor]`.
/// No samples are lost or double-counted during streaming synthesis.
#[kani::proof]
#[kani::unwind(9)]
fn proof_sample_count_consistency() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 6);

    let mut session = StreamingSessionModel::new(n);

    let mut i: usize = 0;
    while i < n {
        let samples: usize = kani::any();
        kani::assume(samples > 0 && samples <= 48000);
        session.next_chunk(samples);

        // Verify running total equals sum of per-chunk samples.
        let mut sum: usize = 0;
        let mut j: usize = 0;
        while j < session.cursor {
            sum += session.per_chunk_samples[j];
            j += 1;
        }
        assert_eq!(
            session.total_samples, sum,
            "total_samples must equal sum of per-chunk samples"
        );
        i += 1;
    }
}

// ============================================================================
// 4. Cancel halts progress — cancellation prevents further chunk processing
// ============================================================================

/// Prove: once `cancel()` is called, every subsequent `next_chunk()` returns 0
/// and the cursor does not advance.
#[kani::proof]
#[kani::unwind(9)]
fn proof_cancel_halts_progress() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 6);

    let mut session = StreamingSessionModel::new(n);

    // Consume some chunks before cancelling.
    let consume_before: usize = kani::any();
    kani::assume(consume_before < n);
    let mut i: usize = 0;
    while i < consume_before {
        session.next_chunk(100);
        i += 1;
    }

    let cursor_at_cancel = session.cursor;
    let samples_at_cancel = session.total_samples;

    // Cancel.
    session.cancel();
    assert!(session.cancelled, "session must be cancelled");
    assert!(session.is_done(), "cancelled session must report is_done");
    assert_eq!(session.remaining(), 0, "cancelled session must have 0 remaining");

    // All subsequent calls produce 0 and do not advance cursor.
    let mut j: usize = 0;
    while j < 3 {
        let produced = session.next_chunk(100);
        assert_eq!(produced, 0, "next_chunk after cancel must produce 0");
        assert_eq!(
            session.cursor, cursor_at_cancel,
            "cursor must not advance after cancel"
        );
        assert_eq!(
            session.total_samples, samples_at_cancel,
            "total_samples must not change after cancel"
        );
        j += 1;
    }
}

// ============================================================================
// 5. Completed state is absorbing — once done, no more chunks can be processed
// ============================================================================

/// Prove: once `is_done()` returns true (whether by exhaustion or cancellation),
/// it remains true for all subsequent operations. The completed state is
/// absorbing — there is no way to leave it except through `reset`-like
/// operations (not modeled here, as the session has no un-cancel path).
#[kani::proof]
#[kani::unwind(9)]
fn proof_completed_state_is_absorbing() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 6);

    let mut session = StreamingSessionModel::new(n);

    // Drive to completion (either by exhaustion or cancellation).
    let exhaust: bool = kani::any();
    if exhaust {
        let mut i: usize = 0;
        while i < n {
            session.next_chunk(100);
            i += 1;
        }
    } else {
        let consume: usize = kani::any();
        kani::assume(consume < n);
        let mut i: usize = 0;
        while i < consume {
            session.next_chunk(100);
            i += 1;
        }
        session.cancel();
    }

    // Session is now done.
    assert!(session.is_done(), "session must be done at this point");
    let cursor_frozen = session.cursor;
    let samples_frozen = session.total_samples;

    // Attempt multiple further operations — is_done remains true, state frozen.
    let mut k: usize = 0;
    while k < 4 {
        let produced = session.next_chunk(100);
        assert_eq!(produced, 0, "no samples produced once done");
        assert!(session.is_done(), "is_done must remain true (absorbing)");
        assert_eq!(session.cursor, cursor_frozen, "cursor must not change once done");
        assert_eq!(
            session.total_samples, samples_frozen,
            "total_samples must not change once done"
        );
        k += 1;
    }
}

// ============================================================================
// 6. Chunk size bounds — each chunk produces samples within [min, max]
// ============================================================================

/// Prove: when each chunk's sample count is constrained to [min_chunk, max_chunk],
/// the total samples after synthesizing all chunks is within
/// [n * min_chunk, n * max_chunk].
#[kani::proof]
#[kani::unwind(9)]
fn proof_chunk_size_bounds() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 6);

    let min_chunk: usize = 100;
    let max_chunk: usize = 4800;

    let mut session = StreamingSessionModel::new(n);

    let mut i: usize = 0;
    while i < n {
        let samples: usize = kani::any();
        kani::assume(samples >= min_chunk && samples <= max_chunk);
        session.next_chunk(samples);

        // Each recorded chunk is within bounds.
        assert!(
            session.per_chunk_samples[i] >= min_chunk,
            "chunk samples must be >= min_chunk"
        );
        assert!(
            session.per_chunk_samples[i] <= max_chunk,
            "chunk samples must be <= max_chunk"
        );
        i += 1;
    }

    // Total samples within aggregate bounds.
    assert!(
        session.total_samples >= n * min_chunk,
        "total samples must be >= n * min_chunk"
    );
    assert!(
        session.total_samples <= n * max_chunk,
        "total samples must be <= n * max_chunk"
    );
}

// ============================================================================
// 7. Session progress fraction — progress = chunks_done / total is in [0, 1]
// ============================================================================

/// Prove: the progress fraction (cursor / total_chunks) is always in [0, 1]
/// at every step of the session. Specifically, `cursor <= total_chunks` holds
/// as an invariant, since `next_chunk` only advances when `cursor < total`.
#[kani::proof]
#[kani::unwind(9)]
fn proof_progress_fraction_in_unit_interval() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 6);

    let mut session = StreamingSessionModel::new(n);

    // Check at initial state.
    let (num, den) = session.progress();
    assert_eq!(num, 0, "progress numerator must be 0 at start");
    assert_eq!(den, n, "progress denominator must equal total_chunks");

    // Step through all chunks + extra calls.
    let steps: usize = kani::any();
    kani::assume(steps <= n + 2);

    let mut i: usize = 0;
    while i < steps {
        let samples: usize = kani::any();
        kani::assume(samples > 0 && samples <= 48000);
        session.next_chunk(samples);

        let (num, den) = session.progress();
        assert!(
            num <= den,
            "progress numerator must not exceed denominator"
        );
        assert_eq!(
            den, n,
            "progress denominator must remain constant"
        );
        i += 1;
    }

    // After exhaustion, progress is exactly 1 (n/n).
    // Consume any remaining.
    while !session.is_done() {
        session.next_chunk(100);
    }
    let (num, den) = session.progress();
    assert_eq!(num, n, "progress must be 1 (n/n) when done");
    assert_eq!(den, n, "denominator must be n");
}

// ============================================================================
// 8. Empty input handling — zero-length text produces zero chunks
// ============================================================================

/// Prove: a session created with 0 chunks is immediately done, produces
/// no samples, and all subsequent operations are no-ops.
#[kani::proof]
#[kani::unwind(1)]
fn proof_empty_input_produces_zero_chunks() {
    let mut session = StreamingSessionModel::new(0);

    // Immediately done.
    assert!(session.is_done(), "empty session must be done immediately");
    assert_eq!(session.remaining(), 0, "empty session must have 0 remaining");
    assert_eq!(session.total_chunks, 0, "empty session must have 0 total chunks");
    assert_eq!(session.cursor, 0, "empty session cursor must be 0");
    assert_eq!(session.total_samples, 0, "empty session must have 0 total samples");

    // Progress is (0, 0) — degenerate case.
    let (num, den) = session.progress();
    assert_eq!(num, 0, "empty progress numerator must be 0");
    assert_eq!(den, 0, "empty progress denominator must be 0");

    // next_chunk produces nothing.
    let produced = session.next_chunk(100);
    assert_eq!(produced, 0, "next_chunk on empty session must produce 0");
    assert_eq!(session.cursor, 0, "cursor must not advance on empty session");
    assert_eq!(session.total_samples, 0, "total_samples must stay 0 on empty session");

    // Cancel is also a no-op (already done).
    session.cancel();
    assert!(session.is_done(), "empty cancelled session must be done");
    assert_eq!(session.remaining(), 0, "empty cancelled session remaining must be 0");
}
