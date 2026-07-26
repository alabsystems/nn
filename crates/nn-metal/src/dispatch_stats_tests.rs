// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`DispatchStats`], counter read/reset, and `record_gpu_event`.

use std::cell::Cell;

use super::*;

// ---------------------------------------------------------------------------
// DispatchStats field access and equality
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_stats_initial_all_zero() {
    reset_counters();
    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 0);
    assert_eq!(stats.blits, 0);
    assert_eq!(stats.flushes, 0);
    assert_eq!(stats.submits, 0);
    assert_eq!(stats.blits_eliminated, 0);
    assert_eq!(stats.arena.hits, 0);
    assert_eq!(stats.arena.misses, 0);
}

#[test]
fn test_dispatch_stats_debug_impl() {
    reset_counters();
    let stats = dispatch_stats();
    let dbg = format!("{stats:?}");
    assert!(dbg.contains("DispatchStats"), "Debug should include type name");
    assert!(dbg.contains("compute_encodings"), "Debug should include field names");
}

#[test]
fn test_dispatch_stats_clone_eq() {
    reset_counters();
    let a = dispatch_stats();
    let b = a;
    assert_eq!(a, b, "Copy semantics must produce identical values");
}

// ---------------------------------------------------------------------------
// Counter increment via thread-local manipulation
// ---------------------------------------------------------------------------

#[test]
fn test_counter_increment_encodings() {
    reset_counters();
    TOTAL_ENCODINGS.with(|c| c.set(42));
    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 42);
}

#[test]
fn test_counter_increment_blits() {
    reset_counters();
    TOTAL_BLITS.with(|c| c.set(7));
    let stats = dispatch_stats();
    assert_eq!(stats.blits, 7);
}

#[test]
fn test_counter_increment_flushes() {
    reset_counters();
    TOTAL_FLUSHES.with(|c| c.set(3));
    let stats = dispatch_stats();
    assert_eq!(stats.flushes, 3);
}

#[test]
fn test_counter_increment_submits() {
    reset_counters();
    TOTAL_SUBMITS.with(|c| c.set(11));
    let stats = dispatch_stats();
    assert_eq!(stats.submits, 11);
}

#[test]
fn test_reset_counters_clears_all() {
    TOTAL_ENCODINGS.with(|c| c.set(100));
    TOTAL_BLITS.with(|c| c.set(50));
    TOTAL_FLUSHES.with(|c| c.set(25));
    TOTAL_SUBMITS.with(|c| c.set(12));
    TOTAL_BLITS_ELIMINATED.with(|c| c.set(8));
    reset_counters();
    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 0);
    assert_eq!(stats.blits, 0);
    assert_eq!(stats.flushes, 0);
    assert_eq!(stats.submits, 0);
    assert_eq!(stats.blits_eliminated, 0);
}

#[test]
fn test_reset_counters_idempotent() {
    reset_counters();
    reset_counters();
    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 0);
    assert_eq!(stats.flushes, 0);
}

#[test]
fn test_counter_increment_blits_eliminated() {
    reset_counters();
    TOTAL_BLITS_ELIMINATED.with(|c| c.set(15));
    let stats = dispatch_stats();
    assert_eq!(stats.blits_eliminated, 15);
}

// ---------------------------------------------------------------------------
// record_gpu_event
// ---------------------------------------------------------------------------

#[test]
fn test_record_gpu_event_increments_and_returns() {
    let counter = Cell::new(0);
    let n1 = record_gpu_event(&counter, "test", 0);
    assert_eq!(n1, 1);
    assert_eq!(counter.get(), 1);

    let n2 = record_gpu_event(&counter, "test", 0);
    assert_eq!(n2, 2);
    assert_eq!(counter.get(), 2);
}

#[test]
fn test_record_gpu_event_preserves_existing_count() {
    let counter = Cell::new(99);
    let n = record_gpu_event(&counter, "flush", 10);
    assert_eq!(n, 100);
    assert_eq!(counter.get(), 100);
}

// ---------------------------------------------------------------------------
// DispatchStats non_exhaustive: constructed via dispatch_stats() only
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_stats_non_exhaustive_via_public_api() {
    // DispatchStats is #[non_exhaustive] — cannot construct via struct literal
    // from outside the crate. This test verifies we can read all fields via
    // the public dispatch_stats() function.
    reset_counters();
    TOTAL_ENCODINGS.with(|c| c.set(5));
    TOTAL_BLITS.with(|c| c.set(3));
    TOTAL_FLUSHES.with(|c| c.set(1));
    TOTAL_SUBMITS.with(|c| c.set(2));

    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 5);
    assert_eq!(stats.blits, 3);
    assert_eq!(stats.flushes, 1);
    assert_eq!(stats.submits, 2);

    // Total Metal encodings = compute + blits (documented invariant).
    assert_eq!(
        stats.compute_encodings + stats.blits,
        8,
        "total Metal encodings = compute + blits"
    );
}
