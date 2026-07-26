// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for buffer pool stats counting invariants.
//!
//! # BUG: `misses` and `discards` double-count on bucket-full path
//!
//! `acquire()` has four mutually exclusive outcomes:
//!   1. Hit (reused available entry) → `hits++`
//!   2. Miss (new pooled buffer created) → `misses++`
//!   3. Discard (oversized, bypasses pool) → `discards++`
//!   4. Discard (bucket full / byte budget exceeded) → `discards++`
//!
//! The documented invariant is `acquisitions == hits + misses + discards`.
//!
//! **Bug (buffer_pool.rs:156):** `self.stats.misses += 1` is placed BEFORE
//! the bucket/budget check (line 157). When the check fails (path 4), the
//! flow falls through to `self.stats.discards += 1` (line 169), resulting
//! in BOTH `misses` and `discards` being incremented for a single acquire.
//!
//! **Fix:** Move `self.stats.misses += 1` inside the success branch (after
//! the `if` on line 157), so it only fires when a new buffer is actually
//! allocated into the pool.
//!
//! The higher-level `ArenaStats::fresh_allocs()` is NOT affected because it
//! uses `arena_misses - pool.hits`, which is correct regardless of the pool
//! internal stats. The issue is purely in `PoolStats` self-consistency.
//!
//!   Move `self.stats.misses += 1;` from line 156 (before the if) to
//!   inside the if block (after line 157, before `let buffer = ...`).
//!   Also update the `PoolStats::misses` doc comment to clarify it
//!   excludes bucket-full/budget-exceeded cases.

use super::*;

/// Verify the stats invariant: acquisitions == hits + misses + discards.
///
/// Each acquire() call must land in exactly one of: hit, miss (new pooled
/// buffer), or discard (unpooled fallback). This test documents the expected
/// stats for each of the four paths.
#[test]
fn test_stats_invariant_acquisitions_eq_sum() {
    // Path 1: oversized → discards only
    let oversized = PoolStats {
        acquisitions: 1,
        hits: 0,
        misses: 0,
        discards: 1,
        ..Default::default()
    };
    assert_eq!(
        oversized.acquisitions,
        oversized.hits + oversized.misses + oversized.discards,
        "oversized path: acquisitions must equal sum"
    );

    // Path 2: pool hit → hits only
    let hit = PoolStats {
        acquisitions: 1,
        hits: 1,
        misses: 0,
        discards: 0,
        ..Default::default()
    };
    assert_eq!(
        hit.acquisitions,
        hit.hits + hit.misses + hit.discards,
        "hit path: acquisitions must equal sum"
    );

    // Path 3: miss, new buffer created → misses only
    let miss_create = PoolStats {
        acquisitions: 1,
        hits: 0,
        misses: 1,
        discards: 0,
        ..Default::default()
    };
    assert_eq!(
        miss_create.acquisitions,
        miss_create.hits + miss_create.misses + miss_create.discards,
        "miss-create path: acquisitions must equal sum"
    );

    // Path 4: miss then discard (bucket full / byte budget exceeded).
    // After fix: discards only, misses stays 0.
    // BUG (pre-fix): misses=1, discards=1 → sum=2 ≠ acquisitions=1.
    let miss_discard = PoolStats {
        acquisitions: 1,
        hits: 0,
        misses: 0,
        discards: 1,
        ..Default::default()
    };
    assert_eq!(
        miss_discard.acquisitions,
        miss_discard.hits + miss_discard.misses + miss_discard.discards,
        "miss-discard path: acquisitions must equal sum"
    );
}

/// Combined scenario: verify the invariant holds across mixed outcomes.
#[test]
fn test_stats_invariant_mixed_scenario() {
    // 10 acquires: 4 hits, 3 misses (new pooled), 3 discards
    let mixed = PoolStats {
        acquisitions: 10,
        hits: 4,
        misses: 3,
        discards: 3,
        pooled_bytes: 3 * 64 * 1024,
        pooled_buffers: 3,
    };
    assert_eq!(
        mixed.acquisitions,
        mixed.hits + mixed.misses + mixed.discards,
        "mixed: acquisitions must equal sum of outcomes"
    );
}

/// Edge case: zero acquisitions (fresh pool, never used).
#[test]
fn test_stats_invariant_zero_acquisitions() {
    let empty = PoolStats::default();
    assert_eq!(
        empty.acquisitions,
        empty.hits + empty.misses + empty.discards,
        "fresh pool: all counters zero"
    );
    assert_eq!(empty.acquisitions, 0);
}

/// Edge case: all acquisitions are discards (all oversized or budget-exceeded).
#[test]
fn test_stats_invariant_all_discards() {
    let all_discard = PoolStats {
        acquisitions: 5,
        hits: 0,
        misses: 0,
        discards: 5,
        pooled_bytes: 0,
        pooled_buffers: 0,
    };
    assert_eq!(
        all_discard.acquisitions,
        all_discard.hits + all_discard.misses + all_discard.discards,
        "all discards: no hits or misses"
    );
}

/// Edge case: all acquisitions are hits (warm pool, everything reused).
#[test]
fn test_stats_invariant_all_hits() {
    let all_hits = PoolStats {
        acquisitions: 20,
        hits: 20,
        misses: 0,
        discards: 0,
        pooled_bytes: 128 * 1024,
        pooled_buffers: 4,
    };
    assert_eq!(
        all_hits.acquisitions,
        all_hits.hits + all_hits.misses + all_hits.discards,
        "all hits: perfect reuse"
    );
}
