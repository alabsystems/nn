// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`TileConfig`] and [`select_gemm_tiles`].
//!
//! Proves:
//! - TileConfig computation correctness (output_per_threadgroup, threadgroup_count)
//! - All preset tile configs (SQUARE, TALL_SKINNY, WIDE) have consistent invariants
//! - select_gemm_tiles routing properties (tiny rejection, alignment)
//! - is_scalar_fallback consistency with select_gemm_tiles

use crate::simdgroup_tile_select::*;

/// Proves: output_per_threadgroup equals tile_m * tile_n for SQUARE config.
#[kani::unwind(1)]
#[kani::proof]
fn tile_config_square_output_correct() {
    let cfg = TileConfig::SQUARE;
    assert_eq!(cfg.output_per_threadgroup(), cfg.tile_m * cfg.tile_n);
    assert_eq!(cfg.output_per_threadgroup(), 32 * 32);
}

/// Proves: output_per_threadgroup equals tile_m * tile_n for TALL_SKINNY config.
#[kani::unwind(1)]
#[kani::proof]
fn tile_config_tall_skinny_output_correct() {
    let cfg = TileConfig::TALL_SKINNY;
    assert_eq!(cfg.output_per_threadgroup(), cfg.tile_m * cfg.tile_n);
    assert_eq!(cfg.output_per_threadgroup(), 64 * 16);
}

/// Proves: output_per_threadgroup equals tile_m * tile_n for WIDE config.
#[kani::unwind(1)]
#[kani::proof]
fn tile_config_wide_output_correct() {
    let cfg = TileConfig::WIDE;
    assert_eq!(cfg.output_per_threadgroup(), cfg.tile_m * cfg.tile_n);
    assert_eq!(cfg.output_per_threadgroup(), 16 * 64);
}

/// Proves: All three preset configs produce 1024 output elements per TG.
#[kani::unwind(1)]
#[kani::proof]
fn tile_config_presets_same_output_count() {
    assert_eq!(TileConfig::SQUARE.output_per_threadgroup(), 1024);
    assert_eq!(TileConfig::TALL_SKINNY.output_per_threadgroup(), 1024);
    assert_eq!(TileConfig::WIDE.output_per_threadgroup(), 1024);
}

/// Proves: threads_per_threadgroup equals product of threadgroup_size for all presets.
#[kani::unwind(1)]
#[kani::proof]
fn tile_config_threads_per_tg_correct() {
    let sq = TileConfig::SQUARE;
    assert_eq!(
        sq.threads_per_threadgroup(),
        sq.threadgroup_size[0] * sq.threadgroup_size[1] * sq.threadgroup_size[2]
    );

    let ts = TileConfig::TALL_SKINNY;
    assert_eq!(
        ts.threads_per_threadgroup(),
        ts.threadgroup_size[0] * ts.threadgroup_size[1] * ts.threadgroup_size[2]
    );

    let wide = TileConfig::WIDE;
    assert_eq!(
        wide.threads_per_threadgroup(),
        wide.threadgroup_size[0] * wide.threadgroup_size[1] * wide.threadgroup_size[2]
    );
}

/// Proves: threadgroup_count * output_per_threadgroup >= m * n for valid m, n.
///
/// This is a coverage property: the tile grid covers the entire M x N output.
#[kani::unwind(1)]
#[kani::proof]
fn tile_config_threadgroup_count_covers_output() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(m > 0 && m <= 4096);
    kani::assume(n > 0 && n <= 4096);

    let cfg = TileConfig::SQUARE;
    let tg_count = cfg.threadgroup_count(m, n);
    let total_output = tg_count * cfg.output_per_threadgroup();
    assert!(total_output >= m * n, "threadgroups must cover all M*N elements");
}

/// Proves: select_gemm_tiles returns None iff m*n < TINY_THRESHOLD,
/// and is_scalar_fallback returns true in exactly the same case.
#[kani::unwind(1)]
#[kani::proof]
fn select_gemm_tiles_consistent_with_scalar_fallback() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let k: usize = kani::any();
    kani::assume(m <= 256 && n <= 256 && k <= 256);
    kani::assume(k > 0); // k=0 is degenerate

    let tiles = select_gemm_tiles(m, k, n);
    let scalar = is_scalar_fallback(m, n);

    if m.saturating_mul(n) < TINY_THRESHOLD {
        assert!(tiles.is_none(), "tiny M*N must return None");
        assert!(scalar, "tiny M*N must be scalar fallback");
    } else {
        // Above tiny threshold, tiles should be Some (kernel is worthwhile).
        assert!(tiles.is_some(), "non-tiny M*N should return a tile config");
        assert!(!scalar, "non-tiny M*N should not be scalar fallback");
    }
}

/// Proves: All tile dimensions in returned configs are multiples of SIMDGROUP_ALIGN.
#[kani::unwind(1)]
#[kani::proof]
fn select_gemm_tiles_alignment_invariant() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let k: usize = kani::any();
    kani::assume(m <= 512 && n <= 512 && k <= 512);
    kani::assume(k > 0);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        assert_eq!(cfg.tile_m % SIMDGROUP_ALIGN, 0, "tile_m must be aligned");
        assert_eq!(cfg.tile_n % SIMDGROUP_ALIGN, 0, "tile_n must be aligned");
        assert_eq!(cfg.tile_k % SIMDGROUP_ALIGN, 0, "tile_k must be aligned");
    }
}
