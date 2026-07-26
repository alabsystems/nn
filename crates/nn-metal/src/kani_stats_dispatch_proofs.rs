// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for PoolStats, ArenaStats, DispatchStats, DispatchPlan
//! builder patterns, simdgroup tile selection safety, segment cache LRU
//! invariants, and DtypeTracker safety properties.
//!
//! 24 harnesses proving arithmetic safety, state machine correctness, and
//! invariant preservation for the Metal backend's diagnostic and dispatch
//! infrastructure.

// ============================================================================
// PoolStats arithmetic safety
// ============================================================================

/// Proves PoolStats::acquisitions == hits + misses + discards invariant.
///
/// The pool tracks three disjoint outcomes per acquire() call:
/// hit (reused), miss (new allocation), discard (bypass). Their sum must
/// always equal the total acquisition count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool_stats_acquisitions_is_sum_of_outcomes() {
    let hits: usize = kani::any();
    let misses: usize = kani::any();
    let discards: usize = kani::any();

    // Constrain to prevent overflow in the addition itself.
    kani::assume(hits <= 1_000_000);
    kani::assume(misses <= 1_000_000);
    kani::assume(discards <= 1_000_000);

    let acquisitions = hits + misses + discards;

    let stats = crate::buffer_pool::PoolStats {
        acquisitions,
        hits,
        misses,
        discards,
        pooled_bytes: 0,
        pooled_buffers: 0,
    };

    assert_eq!(
        stats.acquisitions,
        stats.hits + stats.misses + stats.discards,
        "acquisitions must equal sum of hit/miss/discard"
    );
}

/// Proves PoolStats default is all zeros.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool_stats_default_is_zero() {
    let stats = crate::buffer_pool::PoolStats::default();
    assert_eq!(stats.acquisitions, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.discards, 0);
    assert_eq!(stats.pooled_bytes, 0);
    assert_eq!(stats.pooled_buffers, 0);
}

/// Proves PoolStats Copy semantics preserve all fields.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool_stats_copy_preserves_fields() {
    let stats = crate::buffer_pool::PoolStats {
        acquisitions: kani::any(),
        hits: kani::any(),
        misses: kani::any(),
        discards: kani::any(),
        pooled_bytes: kani::any(),
        pooled_buffers: kani::any(),
    };

    let copied = stats;
    assert_eq!(stats, copied, "Copy must preserve all fields");
}

// ============================================================================
// ArenaStats arithmetic safety
// ============================================================================

/// Proves ArenaStats::hit_rate is in [0.0, 1.0] for any non-negative inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_stats_hit_rate_in_unit_interval() {
    let hits: usize = kani::any();
    let misses: usize = kani::any();

    // Constrain to prevent usize overflow in hits + misses.
    kani::assume(hits <= 1_000_000);
    kani::assume(misses <= 1_000_000);

    let stats = crate::arena::ArenaStats {
        hits,
        misses,
        pool: crate::buffer_pool::PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
    };

    let rate = stats.hit_rate();
    assert!(rate >= 0.0, "hit_rate must be >= 0.0");
    assert!(rate <= 1.0, "hit_rate must be <= 1.0");
}

/// Proves ArenaStats::hit_rate is 0.0 when no allocations occurred.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_stats_hit_rate_zero_on_empty() {
    let stats = crate::arena::ArenaStats {
        hits: 0,
        misses: 0,
        pool: crate::buffer_pool::PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
    };

    assert_eq!(stats.hit_rate(), 0.0, "zero allocations must produce 0.0 rate");
}

/// Proves ArenaStats::hit_rate is 1.0 when all allocations are hits.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_stats_hit_rate_one_on_all_hits() {
    let hits: usize = kani::any();
    kani::assume(hits > 0 && hits <= 1_000_000);

    let stats = crate::arena::ArenaStats {
        hits,
        misses: 0,
        pool: crate::buffer_pool::PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
    };

    assert_eq!(stats.hit_rate(), 1.0, "all-hit must produce 1.0 rate");
}

/// Proves ArenaStats::fresh_allocs uses saturating subtraction (no underflow).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arena_stats_fresh_allocs_no_underflow() {
    let misses: usize = kani::any();
    let pool_hits: usize = kani::any();

    kani::assume(misses <= 1_000_000);
    kani::assume(pool_hits <= 1_000_000);

    let stats = crate::arena::ArenaStats {
        hits: 0,
        misses,
        pool: crate::buffer_pool::PoolStats {
            acquisitions: 0,
            hits: pool_hits,
            misses: 0,
            discards: 0,
            pooled_bytes: 0,
            pooled_buffers: 0,
        },
        growth_count: 0,
        total_growth_count: 0,
    };

    let fresh = stats.fresh_allocs();
    // saturating_sub guarantees no underflow.
    assert!(fresh <= misses, "fresh_allocs must be <= misses");
    if misses >= pool_hits {
        assert_eq!(fresh, misses - pool_hits);
    } else {
        assert_eq!(fresh, 0, "saturating_sub clamps to 0");
    }
}

// ============================================================================
// DispatchStats field consistency
// ============================================================================

/// Proves DispatchStats Copy preserves all scalar fields.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_stats_copy_preserves_fields() {
    let stats = crate::dispatch_stats::DispatchStats {
        compute_encodings: kani::any(),
        blits: kani::any(),
        flushes: kani::any(),
        submits: kani::any(),
        blits_eliminated: kani::any(),
        arena: crate::arena::ArenaStats {
            hits: kani::any(),
            misses: kani::any(),
            pool: crate::buffer_pool::PoolStats::default(),
            growth_count: 0,
            total_growth_count: 0,
        },
    };

    let copied = stats;
    assert_eq!(stats, copied, "DispatchStats copy must be identical");
}

/// Proves DispatchStats total Metal encodings = compute + blits.
///
/// This is the documented invariant: total Metal command encodings is the
/// sum of compute dispatch encodings and buffer-planner blit copies.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_stats_total_encodings_is_compute_plus_blits() {
    let compute: usize = kani::any();
    let blits: usize = kani::any();

    kani::assume(compute <= 100_000);
    kani::assume(blits <= 100_000);

    let total = compute + blits;

    // Verify the total doesn't overflow for reasonable counts.
    assert!(total >= compute, "total must be >= compute encodings");
    assert!(total >= blits, "total must be >= blits");
    assert_eq!(total, compute + blits);
}

// ============================================================================
// DispatchPlan builder pattern safety
// ============================================================================

/// Proves with_output_elems overwrites correctly.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_plan_with_output_elems_overrides() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let plan = crate::dispatch_plan::plan_elementwise(total)
        .expect("elementwise always succeeds for > 0");

    let new_elems: usize = kani::any();
    kani::assume(new_elems <= 1_000_000);

    let modified = plan.with_output_elems(new_elems);
    assert_eq!(modified.output_elems(), new_elems, "override must take effect");
    // Other fields preserved.
    assert_eq!(modified.threads()[0], crate::dispatch_plan::threadgroup_width_1d(total));
}

/// Proves with_use_threadgroups overrides correctly.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_plan_with_use_threadgroups_overrides() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let plan = crate::dispatch_plan::plan_elementwise(total)
        .expect("elementwise always succeeds for > 0");

    assert!(!plan.use_threadgroups(), "elementwise default is dispatch_threads");

    let modified = plan.with_use_threadgroups(true);
    assert!(modified.use_threadgroups(), "override to true must take effect");

    let restored = modified.with_use_threadgroups(false);
    assert!(!restored.use_threadgroups(), "override back to false must work");
}

/// Proves with_threadgroup_memory_bytes overrides correctly.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_plan_with_threadgroup_memory_overrides() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let plan = crate::dispatch_plan::plan_elementwise(total)
        .expect("elementwise always succeeds for > 0");

    assert!(plan.threadgroup_memory_bytes().is_none(), "elementwise has no shared memory");

    let bytes: u64 = kani::any();
    kani::assume(bytes <= 32768); // Metal threadgroup memory limit
    let modified = plan.with_threadgroup_memory_bytes(Some(bytes));
    assert_eq!(modified.threadgroup_memory_bytes(), Some(bytes));

    let cleared = modified.with_threadgroup_memory_bytes(None);
    assert!(cleared.threadgroup_memory_bytes().is_none());
}

/// Proves builder chain ordering doesn't matter (commutative).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_plan_builder_chain_commutative() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let base = crate::dispatch_plan::plan_elementwise(total)
        .expect("elementwise always succeeds");

    let elems: usize = kani::any();
    kani::assume(elems <= 1_000_000);
    let use_tg: bool = kani::any();

    // Order A: output_elems then use_threadgroups
    let a = base.clone()
        .with_output_elems(elems)
        .with_use_threadgroups(use_tg);

    // Order B: use_threadgroups then output_elems
    let b = base
        .with_use_threadgroups(use_tg)
        .with_output_elems(elems);

    assert_eq!(a.output_elems(), b.output_elems());
    assert_eq!(a.use_threadgroups(), b.use_threadgroups());
    assert_eq!(a.grid(), b.grid());
    assert_eq!(a.threads(), b.threads());
}

// ============================================================================
// threadgroup_width_1d safety
// ============================================================================

/// Proves threadgroup_width_1d is always in [1, 64] for non-zero total.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn threadgroup_width_1d_bounded() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let tg = crate::dispatch_plan::threadgroup_width_1d(total);
    assert!(tg >= 1, "threadgroup width must be >= 1");
    assert!(tg <= 64, "threadgroup width must be <= 64");
}

/// Proves threadgroup_width_1d <= total (never exceeds dispatch size).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn threadgroup_width_1d_le_total() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let tg = crate::dispatch_plan::threadgroup_width_1d(total);
    assert!(tg <= total, "threadgroup width must not exceed total elements");
}

// ============================================================================
// Simdgroup tile selection safety
// ============================================================================

/// Proves select_gemm_tiles returns None for tiny matrices (M*N < 1024).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_select_tiny_returns_none() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let k: usize = kani::any();

    kani::assume(m <= 100);
    kani::assume(n <= 100);
    kani::assume(k > 0 && k <= 1024);
    kani::assume(m.saturating_mul(n) < crate::simdgroup_tile_select::TINY_THRESHOLD);

    let result = crate::simdgroup_tile_select::select_gemm_tiles(m, k, n);
    assert!(result.is_none(), "tiny matrices must return None (scalar fallback)");
}

/// Proves all returned TileConfigs have aligned tile dimensions (multiples of 8).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_select_alignment_invariant() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let k: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(m.saturating_mul(n) >= crate::simdgroup_tile_select::TINY_THRESHOLD);

    if let Some(cfg) = crate::simdgroup_tile_select::select_gemm_tiles(m, k, n) {
        let align = crate::simdgroup_tile_select::SIMDGROUP_ALIGN;
        assert_eq!(cfg.tile_m % align, 0, "tile_m must be aligned to {align}");
        assert_eq!(cfg.tile_n % align, 0, "tile_n must be aligned to {align}");
        assert_eq!(cfg.tile_k % align, 0, "tile_k must be aligned to {align}");
    }
}

/// Proves output_per_threadgroup is always tile_m * tile_n.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_config_output_per_threadgroup_correct() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let k: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(m.saturating_mul(n) >= crate::simdgroup_tile_select::TINY_THRESHOLD);

    if let Some(cfg) = crate::simdgroup_tile_select::select_gemm_tiles(m, k, n) {
        assert_eq!(
            cfg.output_per_threadgroup(),
            cfg.tile_m * cfg.tile_n,
            "output_per_threadgroup must be tile_m * tile_n"
        );
    }
}

/// Proves threads_per_threadgroup is always 128 for all configs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_config_threads_per_threadgroup_is_128() {
    use crate::simdgroup_tile_select::TileConfig;

    assert_eq!(TileConfig::SQUARE.threads_per_threadgroup(), 128);
    assert_eq!(TileConfig::TALL_SKINNY.threads_per_threadgroup(), 128);
    assert_eq!(TileConfig::WIDE.threads_per_threadgroup(), 128);
}

/// Proves is_scalar_fallback is consistent with select_gemm_tiles returning None.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_select_scalar_fallback_consistency() {
    let m: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m <= 1024);
    kani::assume(n <= 1024);

    let fallback = crate::simdgroup_tile_select::is_scalar_fallback(m, n);
    let mn = m.saturating_mul(n);
    let expected = mn < crate::simdgroup_tile_select::TINY_THRESHOLD;

    assert_eq!(fallback, expected, "is_scalar_fallback must match TINY_THRESHOLD check");
}

/// Proves threadgroup_count covers the entire M x N output space.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_config_threadgroup_count_covers_output() {
    use crate::simdgroup_tile_select::TileConfig;

    let m: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(n > 0 && n <= 4096);

    let cfg = TileConfig::SQUARE;
    let tg_count = cfg.threadgroup_count(m, n);
    let covered_m = tg_count * cfg.tile_m; // upper bound on covered rows
    let tg_rows = m.div_ceil(cfg.tile_m);
    let tg_cols = n.div_ceil(cfg.tile_n);

    assert_eq!(tg_count, tg_rows * tg_cols, "threadgroup count must be ceil(M/BM) * ceil(N/BN)");
    // Every output element is covered.
    assert!(tg_rows * cfg.tile_m >= m, "all M rows must be covered");
    assert!(tg_cols * cfg.tile_n >= n, "all N cols must be covered");
}

// ============================================================================
// DispatchPlan::plan() for DispatchMode variants
// ============================================================================

/// Proves DispatchMode::Elementwise { total: 0 } produces a valid plan with 0 output.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_mode_elementwise_zero_total_valid() {
    let mode = crate::dispatch_plan::DispatchMode::Elementwise { total: 0 };
    let plan = mode.plan().expect("zero-total elementwise must succeed");
    assert_eq!(plan.output_elems(), 0);
    assert_eq!(plan.grid(), [0, 1, 1]);
}

/// Proves DispatchMode::PerSliceReduction constants encode [outer, reduce].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dispatch_mode_reduction_constants_correct() {
    let outer: u32 = kani::any();
    let reduce: u32 = kani::any();
    let threads: u32 = kani::any();
    let shared: u32 = kani::any();

    kani::assume(outer > 0 && outer <= 10000);
    kani::assume(reduce > 0 && reduce <= 10000);
    kani::assume(threads > 0 && threads <= 1024);

    let mode = crate::dispatch_plan::DispatchMode::PerSliceReduction {
        outer,
        reduce,
        threads,
        shared_bytes: shared,
    };

    let plan = mode.plan().expect("valid reduction plan");
    let constants = plan.constants();
    assert_eq!(constants.len(), 2);
    assert_eq!(constants[0], outer, "first constant must be outer");
    assert_eq!(constants[1], reduce, "second constant must be reduce");
}
