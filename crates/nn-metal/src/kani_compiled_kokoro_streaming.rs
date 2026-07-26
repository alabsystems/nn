// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `StreamingKokoroSession` state machine invariants.
//!
//! These harnesses prove that the pull-based streaming session's cursor-based
//! state machine satisfies key invariants for all reachable states:
//!
//! 1. `remaining() + synthesized_count() == total_chunks()`
//! 2. `is_done() <=> remaining() == 0`
//! 3. `reset()` restores initial state
//! 4. Cursor monotonicity (only increases or resets to 0)
//! 5. `remaining() <= total_chunks()` always
//!
//! Since DynTensor cannot be constructed in Kani (Metal/GPU dependencies),
//! we use `StreamingKokoroSession::kani_from_len_cursor()` which creates
//! a session with a given chunk count and cursor position using CPU-only
//! dummy tensors.
//!
//! Part of #3351.

use super::StreamingKokoroSession;

// ============================================================================
// 1. remaining + synthesized == total (conservation law)
// ============================================================================

/// Prove: for any session state (cursor in 0..=len), the conservation law
/// `remaining() + synthesized_count() == total_chunks()` holds.
///
/// This is the fundamental accounting invariant: every chunk is either
/// "already synthesized" or "remaining." None are lost or double-counted.
#[kani::proof]
#[kani::unwind(1)]
fn proof_remaining_plus_synthesized_equals_total() {
    let len: usize = kani::any();
    let cursor: usize = kani::any();
    kani::assume(len <= 8);
    kani::assume(cursor <= len + 2); // allow cursor past end (saturating_sub handles it)

    let session = StreamingKokoroSession::kani_from_len_cursor(len, cursor);

    let remaining = session.remaining();
    let synthesized = session.synthesized_count();
    let total = session.total_chunks();

    assert_eq!(total, len, "total_chunks must equal chunk vec length");

    // The key invariant: when cursor <= len, remaining + synthesized == total.
    // When cursor > len, remaining saturates to 0 and synthesized == cursor (not len).
    // The real invariant is: min(cursor, len) + remaining == len.
    if cursor <= len {
        assert_eq!(
            remaining + synthesized,
            total,
            "remaining + synthesized must equal total when cursor <= len"
        );
    } else {
        // cursor > len: remaining saturates to 0, synthesized_count = cursor
        assert_eq!(remaining, 0, "remaining must be 0 when cursor > len");
    }
}

// ============================================================================
// 2. is_done iff remaining == 0
// ============================================================================

/// Prove: `is_done()` is true if and only if `remaining() == 0`.
///
/// This ensures the session's "done" flag is consistent with the remaining
/// count. A caller checking either condition gets the same answer.
#[kani::proof]
#[kani::unwind(1)]
fn proof_is_done_iff_remaining_zero() {
    let len: usize = kani::any();
    let cursor: usize = kani::any();
    kani::assume(len <= 8);
    kani::assume(cursor <= len + 2);

    let session = StreamingKokoroSession::kani_from_len_cursor(len, cursor);

    let done = session.is_done();
    let remaining = session.remaining();

    // is_done() uses cursor >= len; remaining() uses len.saturating_sub(cursor).
    // Both should agree on the "nothing left" condition.
    assert_eq!(
        done,
        remaining == 0,
        "is_done() must be equivalent to remaining() == 0"
    );
}

// ============================================================================
// 3. reset restores initial state
// ============================================================================

/// Prove: after `reset()`, the session is in its initial state:
/// `remaining() == total_chunks()`, `synthesized_count() == 0`, `!is_done()`
/// (for non-empty sessions).
///
/// Reset is used for re-synthesizing with a different voice without
/// re-tokenizing. The invariant ensures no chunks are skipped or lost.
#[kani::proof]
#[kani::unwind(1)]
fn proof_reset_restores_initial_state() {
    let len: usize = kani::any();
    let cursor: usize = kani::any();
    kani::assume(len <= 8);
    kani::assume(len > 0); // non-empty session (empty is trivially done)
    kani::assume(cursor <= len + 2);

    let mut session = StreamingKokoroSession::kani_from_len_cursor(len, cursor);

    session.reset();

    assert_eq!(
        session.remaining(),
        session.total_chunks(),
        "after reset, remaining must equal total"
    );
    assert_eq!(
        session.synthesized_count(),
        0,
        "after reset, synthesized must be 0"
    );
    assert!(
        !session.is_done(),
        "after reset, non-empty session must not be done"
    );
}

// ============================================================================
// 4. cursor monotonicity (advance or reset)
// ============================================================================

/// Prove: the cursor only increases (simulated `next_chunk`) or resets to 0.
///
/// We model the two state transitions:
/// - `advance()`: cursor increments by 1 (when not done)
/// - `reset()`: cursor goes to 0
///
/// After either transition, the cursor is >= 0 (trivial for usize) and
/// the new cursor is either old_cursor + 1 or 0.
#[kani::proof]
#[kani::unwind(1)]
fn proof_cursor_monotonic() {
    let len: usize = kani::any();
    let cursor: usize = kani::any();
    kani::assume(len <= 8);
    kani::assume(cursor <= len); // valid pre-advance state

    let mut session = StreamingKokoroSession::kani_from_len_cursor(len, cursor);
    let old_synthesized = session.synthesized_count();

    // Choose a transition: advance or reset.
    let do_reset: bool = kani::any();

    if do_reset {
        session.reset();
        assert_eq!(
            session.synthesized_count(),
            0,
            "reset must set cursor to 0"
        );
    } else if !session.is_done() {
        // Simulate advance: cursor += 1 (what next_chunk does internally).
        session.kani_advance_cursor();
        let new_synthesized = session.synthesized_count();
        assert_eq!(
            new_synthesized,
            old_synthesized + 1,
            "advance must increment cursor by exactly 1"
        );
    }
    // If is_done() and not resetting, no state change — cursor stays.
}

// ============================================================================
// 5. remaining never exceeds total
// ============================================================================

/// Prove: `remaining() <= total_chunks()` for all reachable states.
///
/// Since remaining uses `saturating_sub`, it cannot underflow. And since
/// cursor >= 0 (usize), len - cursor <= len. This harness verifies it
/// for all len/cursor combinations.
#[kani::proof]
#[kani::unwind(1)]
fn proof_remaining_never_exceeds_total() {
    let len: usize = kani::any();
    let cursor: usize = kani::any();
    kani::assume(len <= 8);
    kani::assume(cursor <= len + 2); // allow past-end cursor

    let session = StreamingKokoroSession::kani_from_len_cursor(len, cursor);

    assert!(
        session.remaining() <= session.total_chunks(),
        "remaining must never exceed total_chunks"
    );
}
