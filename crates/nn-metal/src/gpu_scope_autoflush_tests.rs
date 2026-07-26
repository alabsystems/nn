#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MAX_LAZY_ENCODINGS auto-flush tests (P1-294 Finding 3).
//!
//! Safety valve preventing Metal command buffer exhaustion. Tests exercise
//! `get_or_create_batch()` auto-flush at MAX_LAZY_ENCODINGS (`gpu_scope.rs`).
//!
//! Extracted from `gpu_scope_lazy_tests.rs` for 500-line compliance.

use super::*;
use crate::dispatch_stats::{dispatch_stats, reset_counters};
use crate::test_common::init;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

/// After MAX_LAZY_ENCODINGS calls, the next get_or_create_batch() auto-flushes.
///
/// Verifies the auto-flush path increments TOTAL_FLUSHES and resets the
/// encoding count, then creates a fresh batch for continued dispatch.
#[test]
fn test_auto_flush_triggers_at_max_lazy_encodings() {
    init();

    // Clear any pending state.
    flush().unwrap();
    reset_counters();

    // Accumulate exactly MAX_LAZY_ENCODINGS calls.
    // Each get_or_create_batch() increments ENCODING_COUNT by 1.
    // At count < MAX_LAZY_ENCODINGS, no auto-flush occurs.
    for _ in 0..MAX_LAZY_ENCODINGS {
        get_or_create_batch().unwrap();
    }

    // Verify: batch is active, count == MAX_LAZY_ENCODINGS, no auto-flush yet.
    assert!(
        is_lazy_batch_active(),
        "batch should be active after encoding"
    );
    assert_eq!(
        pending_encoding_count(),
        MAX_LAZY_ENCODINGS,
        "encoding count should be exactly MAX_LAZY_ENCODINGS"
    );

    let stats_before = dispatch_stats();
    // Initial flush() above does NOT count (we reset_counters after it).
    // The only flushes counted are those since reset_counters().
    assert_eq!(
        stats_before.flushes, 0,
        "no auto-flush should have occurred yet"
    );

    // The (MAX_LAZY_ENCODINGS + 1)th call triggers auto-flush.
    get_or_create_batch().unwrap();

    let stats_after = dispatch_stats();
    assert_eq!(
        stats_after.flushes, 1,
        "auto-flush should have fired exactly once"
    );

    // After auto-flush: encoding count is reset to 0 then incremented to 1.
    assert_eq!(
        pending_encoding_count(),
        1,
        "encoding count should be 1 after auto-flush + new encoding"
    );

    // A new batch was created after the auto-flush.
    assert!(
        is_lazy_batch_active(),
        "new batch should be active after auto-flush"
    );

    // Total encodings = MAX_LAZY_ENCODINGS + 1 (the triggering call).
    assert_eq!(
        stats_after.compute_encodings,
        MAX_LAZY_ENCODINGS + 1,
        "total encodings should be MAX_LAZY_ENCODINGS + 1"
    );

    // Clean up.
    flush().unwrap();
}

/// Auto-flush produces correct GPU results for ops spanning the boundary.
///
/// Encodes MAX_LAZY_ENCODINGS real GPU ops, triggers auto-flush, then
/// continues encoding. All results must be correct after final readback.
#[test]
fn test_auto_flush_preserves_gpu_correctness() {
    init();
    flush().unwrap();
    reset_counters();

    let device = Device::metal();
    let mut x = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();

    // Accumulate MAX_LAZY_ENCODINGS ops using add_scalar.
    // Each add_scalar is 1.0, so after N ops: x = 1.0 + N.
    for _ in 0..MAX_LAZY_ENCODINGS {
        x = x.add_scalar(1.0).unwrap();
    }

    // At this point, auto-flush should have triggered internally
    // (DynTensor ops call get_or_create_batch, which checks the count).
    // The lazy batch was committed and a new one started.

    // Add a few more ops after the auto-flush boundary.
    for _ in 0..10 {
        x = x.add_scalar(1.0).unwrap();
    }

    // Read back to CPU — triggers final flush.
    let vals = x
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Expected: 1.0 + (MAX_LAZY_ENCODINGS + 10) * 1.0
    let expected = 1.0 + (MAX_LAZY_ENCODINGS + 10) as f32;
    for &v in &vals {
        assert!(
            (v - expected).abs() < 1.0,
            "auto-flush correctness: got {v}, expected {expected}"
        );
    }

    let stats = dispatch_stats();
    // At least 1 auto-flush + 1 final flush from to_device(Cpu).
    assert!(
        stats.flushes >= 2,
        "expected >=2 flushes (auto + readback), got {}",
        stats.flushes
    );
}
