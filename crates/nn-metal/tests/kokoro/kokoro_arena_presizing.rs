// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Arena pre-sizing verification for Kokoro synthesis pipeline.
//!
//! Verifies that after a warmup synthesis call, `estimate_arena_bytes()` returns
//! a non-zero estimate and that subsequent synthesis calls produce zero arena
//! overflow events when the arena is pre-sized using that estimate.
//!
//! Part of #4289.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use super::kokoro_test_weights as kw;

fn cpu() -> Device {
    Device::Cpu
}

/// After warmup, `estimate_arena_bytes()` returns a non-zero estimate that
/// prevents arena overflows on subsequent synthesis calls.
///
/// Strategy:
/// 1. Build a miniaturized Kokoro and run a warmup synthesis (populates caches).
/// 2. Verify `estimate_arena_bytes() > 0` (segments are compiled, estimate works).
/// 3. Reset arena stats.
/// 4. Run synthesis again — arena is already pre-sized from the first call.
/// 5. Verify zero overflow events via `ArenaStats::total_overflow_count`.
///
/// Part of #4289.
#[test]
fn test_arena_presizing_eliminates_overflows_on_warm_call() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(4289, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    // Warmup: compile all segments and populate caches.
    let _ = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("warmup synthesize");

    // After warmup, estimate_arena_bytes() should be non-zero (segments compiled).
    let estimate = kokoro.estimate_arena_bytes();
    assert!(
        estimate > 0,
        "estimate_arena_bytes() should be non-zero after warmup, got 0. \
         Segment caches may not have been populated.",
    );
    eprintln!(
        "Arena estimate after warmup: {} bytes ({:.1} MB)",
        estimate,
        estimate as f64 / (1024.0 * 1024.0)
    );

    // Pre-size the arena to the estimated capacity. This is what
    // synthesize_gpu() does internally on subsequent calls.
    nn_metal::ensure_default_arena_capacity(cache.context(), estimate)
        .expect("ensure_default_arena_capacity");

    // Reset arena stats to measure only the second call.
    nn_metal::reset_arena_stats();

    // Second synthesis: arena is pre-sized — should have zero overflows.
    let _ = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("warm synthesize");

    let stats = nn_metal::arena_stats();
    eprintln!(
        "Warm call arena stats: hits={}, misses={}, growth_count={}, overflow_count={}, \
         total_overflow_count={}, overflow_bytes={}",
        stats.hits,
        stats.misses,
        stats.growth_count,
        stats.overflow_count,
        stats.total_overflow_count,
        stats.overflow_bytes,
    );

    // The key invariant: zero overflow events on the warm call.
    // With auto-grow enabled, overflow_count tracks slab growth events
    // that happened because the slab was too small. Pre-sizing eliminates these.
    assert_eq!(
        stats.overflow_count, 0,
        "Arena should have zero overflow events after pre-sizing. \
         Got {} overflows ({} overflow bytes). The estimate_arena_bytes() \
         result ({} bytes) may be insufficient — increase the headroom multiplier.",
        stats.overflow_count, stats.overflow_bytes, estimate,
    );
}

/// `estimate_arena_bytes()` returns 0 before any synthesis (no compiled segments).
///
/// Part of #4289.
#[test]
fn test_arena_estimate_zero_before_warmup() {
    let (kokoro, _cache) = kw::build_kokoro_mini();
    let estimate = kokoro.estimate_arena_bytes();
    assert_eq!(
        estimate, 0,
        "estimate_arena_bytes() should be 0 before any synthesis, got {estimate}",
    );
}
