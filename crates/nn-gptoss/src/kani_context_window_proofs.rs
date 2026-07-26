// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for context window management invariants.
//!
//! Covers:
//! - Generated positions are strictly sequential
//! - remaining_capacity() <= max_context_length
//! - chunk_positions covers all positions without gaps or overlaps
//! - Sliding window range contains the current position
//! - advance(n) increases current_position by exactly n
//!
//! Part of gpt-oss context window management (131K YaRN RoPE).

use super::context_window::*;

// ============================================================================
// Harness 1: Generated positions are strictly sequential
// ============================================================================

/// Proves that `positions_for_tokens(n)` returns a strictly sequential array
/// starting at `current_position`.
///
/// For any starting position and any token count within bounds, positions
/// must be `[pos, pos+1, pos+2, ..., pos+n-1]` with no gaps or repeats.
#[kani::unwind(17)]
#[kani::proof]
fn proof_positions_sequential() {
    let start_pos: usize = kani::any();
    let n_tokens: usize = kani::any();

    kani::assume(start_pos <= 131_072);
    kani::assume(n_tokens >= 1 && n_tokens <= 16);
    // Prevent overflow in position computation
    kani::assume(start_pos.checked_add(n_tokens).is_some());

    let cfg = ContextWindowConfig::new(200_000, 4096, 128, 4096, true);
    let mut cw = ContextWindow::new(cfg);
    cw.advance(start_pos);

    let positions = cw.positions_for_tokens(n_tokens);

    assert_eq!(
        positions.len(),
        n_tokens,
        "positions length must equal n_tokens"
    );

    // First position must be current_position
    assert_eq!(
        positions[0], start_pos,
        "first position must equal current_position"
    );

    // All positions must be strictly sequential
    let mut i: usize = 1;
    while i < positions.len() {
        assert_eq!(
            positions[i],
            positions[i - 1] + 1,
            "positions must be strictly sequential"
        );
        i += 1;
    }
}

// ============================================================================
// Harness 2: remaining_capacity() <= max_context_length
// ============================================================================

/// Proves that `remaining_capacity()` is always bounded by `max_context_length`.
///
/// For any advance amount, remaining capacity can never exceed the configured
/// maximum. This is a safety invariant: remaining_capacity is used to guard
/// against over-generation.
#[kani::unwind(1)]
#[kani::proof]
fn proof_remaining_capacity_bounded() {
    let max_ctx: usize = kani::any();
    let advance_amount: usize = kani::any();

    kani::assume(max_ctx >= 1 && max_ctx <= 262_144);
    kani::assume(advance_amount <= 262_144);

    let cfg = ContextWindowConfig::new(max_ctx, 4096, 128, 4096, true);
    let mut cw = ContextWindow::new(cfg);

    // Before any advance, remaining == max
    assert_eq!(
        cw.remaining_capacity(),
        max_ctx,
        "initial remaining must equal max_context_length"
    );
    assert!(
        cw.remaining_capacity() <= max_ctx,
        "remaining must be <= max_context_length initially"
    );

    cw.advance(advance_amount);

    assert!(
        cw.remaining_capacity() <= max_ctx,
        "remaining must be <= max_context_length after advance"
    );

    // remaining + current_position >= max_ctx (they sum to at least max when pos <= max)
    if cw.current_position() <= max_ctx {
        assert_eq!(
            cw.remaining_capacity() + cw.current_position(),
            max_ctx,
            "remaining + position must equal max_context_length"
        );
    }
}

// ============================================================================
// Harness 3: chunk_positions covers all positions without gaps or overlaps
// ============================================================================

/// Proves that `chunk_positions` produces contiguous, non-overlapping chunks
/// that together cover exactly `[0, total_len)`.
///
/// This ensures that chunked prefill processes every token exactly once with
/// no missed or duplicated positions.
#[kani::unwind(18)]
#[kani::proof]
fn proof_chunks_cover_full_range() {
    let total_len: usize = kani::any();
    let chunk_size: usize = kani::any();

    kani::assume(total_len >= 1 && total_len <= 64);
    kani::assume(chunk_size >= 1 && chunk_size <= 16);

    let chunks = chunk_positions(total_len, chunk_size);

    // Chunks must be non-empty for non-zero total_len
    assert!(
        !chunks.is_empty(),
        "chunks must be non-empty when total_len > 0"
    );

    // Verify contiguous coverage: each chunk starts where the previous ended
    let mut covered: usize = 0;
    let mut i: usize = 0;
    while i < chunks.len() {
        let (start, len) = chunks[i];
        assert_eq!(start, covered, "chunk must start at end of previous chunk");
        assert!(len >= 1, "each chunk must have non-zero length");
        assert!(len <= chunk_size, "each chunk length must be <= chunk_size");
        covered += len;
        i += 1;
    }

    // Total coverage must equal total_len
    assert_eq!(
        covered, total_len,
        "chunks must cover exactly total_len positions"
    );
}

// ============================================================================
// Harness 4: Sliding window range contains the current position
// ============================================================================

/// Proves that `sliding_window_range(position, window_size)` always returns
/// a range `(start, end)` where `start <= position` and `end > position`
/// (i.e., the position itself is within the range).
///
/// This is critical for correctness: a token must always be able to attend
/// to itself in sliding window attention.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sliding_window_range_valid() {
    let position: usize = kani::any();
    let window_size: usize = kani::any();

    kani::assume(position <= 262_144);
    kani::assume(window_size >= 1 && window_size <= 131_072);

    let (start, end) = sliding_window_range(position, window_size);

    assert!(
        start <= position,
        "sliding window start must be <= position"
    );
    assert!(
        end > position,
        "sliding window end must be > position (position is included)"
    );
    assert!(end >= start, "sliding window end must be >= start");

    // Range width is at most window_size + 1
    assert!(
        end - start <= window_size + 1,
        "sliding window range must not exceed window_size + 1"
    );
}

// ============================================================================
// Harness 5: advance(n) increases current_position by exactly n
// ============================================================================

/// Proves that `advance(n)` increases `current_position` by exactly `n`
/// when the resulting position is within `max_context_length`.
///
/// When the result would exceed `max_context_length`, position saturates
/// at the maximum. This ensures no overflow and predictable position tracking.
#[kani::unwind(1)]
#[kani::proof]
fn proof_advance_monotonic() {
    let max_ctx: usize = kani::any();
    let initial_pos: usize = kani::any();
    let advance_n: usize = kani::any();

    kani::assume(max_ctx >= 1 && max_ctx <= 262_144);
    kani::assume(initial_pos <= max_ctx);
    kani::assume(advance_n <= 262_144);

    let cfg = ContextWindowConfig::new(max_ctx, 4096, 128, 4096, true);
    let mut cw = ContextWindow::new(cfg);

    // Advance to initial position first
    cw.advance(initial_pos);
    let pos_before = cw.current_position();

    cw.advance(advance_n);
    let pos_after = cw.current_position();

    // Position must not decrease (monotonic)
    assert!(
        pos_after >= pos_before,
        "advance must not decrease current_position"
    );

    // If no saturation, advance is exact
    if initial_pos.checked_add(advance_n).is_some() && initial_pos + advance_n <= max_ctx {
        assert_eq!(
            pos_after,
            initial_pos + advance_n,
            "advance(n) must increase position by exactly n when within bounds"
        );
    }

    // Position never exceeds max
    assert!(
        pos_after <= max_ctx,
        "position must never exceed max_context_length"
    );
}
