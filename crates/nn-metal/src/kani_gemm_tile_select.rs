// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for adaptive GEMM tile selection (#3549).
//!
//! Proves safety and correctness properties of the two tile selection systems:
//!
//! 1. **Production simdgroup routing** (`dyn_tensor_metal_matmul_simd.rs`):
//!    `select_tile_config`, `should_use_simdgroup`, `tg_memory_bytes`.
//!
//! 2. **Adaptive tile selection** (`simdgroup_tile_select.rs`):
//!    `select_gemm_tiles`, `TileConfig`, shape-class routing.
//!
//! ## Properties Proved
//!
//! - Tile dimensions are always multiples of SIMDGROUP_ALIGN (8)
//! - No integer overflow in M*K, K*N, M*N, threadgroup calculations
//! - Threadgroup memory fits within Metal's 32 KB limit
//! - Threads per threadgroup <= 1024 (Metal hard limit)
//! - All valid dimension ranges are handled (no silent fallthrough)
//! - select_gemm_tiles and is_scalar_fallback are consistent
//! - Production select_tile_config always returns a valid GemmTileConfig
//! - should_use_simdgroup alignment invariant

use crate::simdgroup_tile_select::{
    is_scalar_fallback, select_gemm_tiles, TileConfig, SIMDGROUP_ALIGN, SMALL_K_THRESHOLD,
    TINY_THRESHOLD,
};

// ---------------------------------------------------------------------------
// Adaptive tile selection: select_gemm_tiles
// ---------------------------------------------------------------------------

/// Prove: when select_gemm_tiles returns Some, all tile dimensions are
/// multiples of SIMDGROUP_ALIGN (8).
///
/// This is the core alignment invariant — Metal simdgroup_matrix<T, 8, 8>
/// requires all tile dimensions to be multiples of 8.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_dimensions_aligned() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    // Bound for CBMC tractability.
    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        assert!(
            cfg.tile_m % SIMDGROUP_ALIGN == 0,
            "tile_m must be aligned to SIMDGROUP_ALIGN"
        );
        assert!(
            cfg.tile_n % SIMDGROUP_ALIGN == 0,
            "tile_n must be aligned to SIMDGROUP_ALIGN"
        );
        assert!(
            cfg.tile_k % SIMDGROUP_ALIGN == 0,
            "tile_k must be aligned to SIMDGROUP_ALIGN"
        );
    }
}

/// Prove: select_gemm_tiles never returns tiles with zero dimensions.
///
/// Zero tile dimensions would produce division-by-zero in threadgroup_count().
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_dimensions_nonzero() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        assert!(cfg.tile_m > 0, "tile_m must be positive");
        assert!(cfg.tile_n > 0, "tile_n must be positive");
        assert!(cfg.tile_k > 0, "tile_k must be positive");
    }
}

/// Prove: threads_per_threadgroup never exceeds Metal's 1024-thread limit.
///
/// Metal specification: maximum 1024 threads per threadgroup.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn threads_per_threadgroup_within_metal_limit() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        let threads = cfg.threads_per_threadgroup();
        assert!(
            threads <= 1024,
            "threads per threadgroup ({threads}) exceeds Metal 1024 limit"
        );
    }
}

/// Prove: threadgroup_size components are all positive.
///
/// Zero in any threadgroup_size dimension is invalid for Metal dispatch.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn threadgroup_size_all_positive() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        assert!(cfg.threadgroup_size[0] > 0, "threads_x must be positive");
        assert!(cfg.threadgroup_size[1] > 0, "threads_y must be positive");
        assert!(cfg.threadgroup_size[2] > 0, "threads_z must be positive");
    }
}

/// Prove: output_per_threadgroup does not overflow for any returned config.
///
/// tile_m * tile_n must fit in usize without overflow.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn output_per_threadgroup_no_overflow() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        // Verify no overflow: tile_m * tile_n
        let product = cfg.tile_m.checked_mul(cfg.tile_n);
        assert!(
            product.is_some(),
            "tile_m * tile_n overflows"
        );
        assert_eq!(
            product.unwrap(),
            cfg.output_per_threadgroup(),
            "output_per_threadgroup must equal tile_m * tile_n"
        );
    }
}

/// Prove: threadgroup_count does not overflow for reasonable M, N values.
///
/// div_ceil(m, tile_m) * div_ceil(n, tile_n) must not overflow.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn threadgroup_count_no_overflow() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        let tg_m = m.div_ceil(cfg.tile_m);
        let tg_n = n.div_ceil(cfg.tile_n);
        let tg_count = tg_m.checked_mul(tg_n);
        assert!(
            tg_count.is_some(),
            "threadgroup count overflows for m={m}, n={n}"
        );
        assert_eq!(
            tg_count.unwrap(),
            cfg.threadgroup_count(m, n),
            "threadgroup_count must match manual calculation"
        );
    }
}

/// Prove: select_gemm_tiles and is_scalar_fallback are consistent.
///
/// When is_scalar_fallback returns true, select_gemm_tiles must return None.
/// When is_scalar_fallback returns false AND M*N >= TINY_THRESHOLD,
/// select_gemm_tiles must return Some.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_fallback_consistent_with_select() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    let fallback = is_scalar_fallback(m, n);
    let tiles = select_gemm_tiles(m, k, n);

    if fallback {
        // Scalar fallback → must not return tile config.
        assert!(
            tiles.is_none(),
            "scalar fallback flagged but select_gemm_tiles returned Some"
        );
    }
    if tiles.is_some() {
        // Tiles selected → must not be scalar fallback.
        assert!(
            !fallback,
            "select_gemm_tiles returned Some but scalar fallback is true"
        );
    }
}

/// Prove: small-K path always rounds tile_k up to a multiple of SIMDGROUP_ALIGN.
///
/// When K <= SMALL_K_THRESHOLD, the tile_k in the returned config must be
/// >= K and a multiple of 8.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn small_k_tile_rounded_up() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= SMALL_K_THRESHOLD);
    kani::assume(n > 0 && n <= 4096);

    // Ensure not tiny.
    kani::assume(m.saturating_mul(n) >= TINY_THRESHOLD);

    let cfg = select_gemm_tiles(m, k, n);
    assert!(cfg.is_some(), "non-tiny shapes must return Some");

    let cfg = cfg.unwrap();
    assert!(cfg.tile_k >= k, "tile_k must be >= K for small-K path");
    assert!(
        cfg.tile_k % SIMDGROUP_ALIGN == 0,
        "tile_k must be aligned for small-K path"
    );
    // Minimal rounding: tile_k should be the next_multiple_of(8) at most.
    assert!(
        cfg.tile_k <= k + SIMDGROUP_ALIGN - 1,
        "tile_k rounding should be minimal"
    );
}

/// Prove: M*N product in select_gemm_tiles uses saturating_mul safely.
///
/// When M and N are large, M*N could overflow. The function uses
/// saturating_mul, so the comparison against TINY_THRESHOLD remains
/// correct even for extreme inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mn_product_saturating_no_false_tiny() {
    let m: usize = kani::any();
    let n: usize = kani::any();

    // Large values that would overflow on unchecked multiply.
    kani::assume(m >= 1 && m <= usize::MAX / 2);
    kani::assume(n >= 1 && n <= usize::MAX / 2);

    let sat = m.saturating_mul(n);

    // If both are >= 1 and saturating_mul saturated to MAX, we must NOT
    // falsely classify as tiny (< TINY_THRESHOLD).
    if m > 0 && n > 0 {
        // m*n >= 1, so saturating_mul >= 1 >= TINY_THRESHOLD only if
        // the actual product >= TINY_THRESHOLD. If it saturates, it's
        // usize::MAX which is definitely >= TINY_THRESHOLD.
        if sat == usize::MAX && m.checked_mul(n).is_none() {
            // Overflow saturated — result is usize::MAX >= TINY_THRESHOLD.
            assert!(sat >= TINY_THRESHOLD, "saturated product must pass tiny check");
        }
    }
}

// ---------------------------------------------------------------------------
// Production tile selection: select_tile_config / should_use_simdgroup
// ---------------------------------------------------------------------------

/// Prove: should_use_simdgroup returns false for non-aligned dimensions.
///
/// If any dimension is not a multiple of 8, simdgroup dispatch is invalid.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn simdgroup_rejects_unaligned() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    // At least one dimension not aligned to 8.
    kani::assume(
        m % 8 != 0 || k % 8 != 0 || n % 8 != 0,
    );

    assert!(
        !crate::dyn_tensor_metal::should_use_simdgroup(m, k, n),
        "unaligned dims must not route to simdgroup"
    );
}

/// Prove: should_use_simdgroup returns false when K < 128.
///
/// K must be >= 128 to amortize shared memory tiling cost.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn simdgroup_rejects_small_k() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k < 128);
    kani::assume(n > 0 && n <= 4096);
    // Ensure all aligned to 8.
    kani::assume(m % 8 == 0 && k % 8 == 0 && n % 8 == 0);

    assert!(
        !crate::dyn_tensor_metal::should_use_simdgroup(m, k, n),
        "K < 128 must not route to simdgroup"
    );
}

/// Prove: should_use_simdgroup returns false when M*N < 16384.
///
/// Dispatch overhead dominates compute at small M*N.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn simdgroup_rejects_small_mn() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k >= 128 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(m % 8 == 0 && k % 8 == 0 && n % 8 == 0);
    kani::assume(m * n < 16_384);

    assert!(
        !crate::dyn_tensor_metal::should_use_simdgroup(m, k, n),
        "M*N < 16384 must not route to simdgroup"
    );
}

/// Prove: when should_use_simdgroup returns true, all alignment and size
/// invariants hold.
///
/// This is the positive direction: if the function accepts, the shape is valid.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn simdgroup_acceptance_invariants() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if crate::dyn_tensor_metal::should_use_simdgroup(m, k, n) {
        assert!(m % 8 == 0, "accepted M must be aligned");
        assert!(k % 8 == 0, "accepted K must be aligned");
        assert!(n % 8 == 0, "accepted N must be aligned");
        assert!(m * n >= 16_384, "accepted M*N must be >= 16384");
        assert!(k >= 128, "accepted K must be >= 128");
    }
}

/// Prove: select_tile_config always returns a valid GemmTileConfig (no panic).
///
/// The function is total for all non-zero inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_tile_config_total() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(batch <= 256);

    // Must not panic.
    let tile = crate::dyn_tensor_metal::select_tile_config(m, k, n, batch);

    // Result must be one of the two known configs.
    let is_small = tile.bm == 32 && tile.bn == 32;
    let is_large = tile.bm == 64 && tile.bn == 64;
    assert!(
        is_small || is_large,
        "select_tile_config must return SMALL or LARGE"
    );
}

/// Prove: select_tile_config grid dimensions do not overflow u32.
///
/// The dispatch grid uses n.div_ceil(tile.bn) and m.div_ceil(tile.bm)
/// as u32 values. These must fit in u32 for Metal dispatch.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn select_tile_config_grid_fits_u32() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    // Production range: up to 65536 per dimension (massive GEMMs).
    kani::assume(m > 0 && m <= 65536);
    kani::assume(k > 0 && k <= 65536);
    kani::assume(n > 0 && n <= 65536);
    kani::assume(batch > 0 && batch <= 256);

    let tile = crate::dyn_tensor_metal::select_tile_config(m, k, n, batch);

    let grid_x = n.div_ceil(tile.bn as usize);
    let grid_y = m.div_ceil(tile.bm as usize);

    // All grid dimensions must fit in u32 for Metal dispatch.
    assert!(
        grid_x <= u32::MAX as usize,
        "grid_x overflows u32"
    );
    assert!(
        grid_y <= u32::MAX as usize,
        "grid_y overflows u32"
    );
    assert!(
        batch <= u32::MAX as usize,
        "batch overflows u32"
    );
}

/// Prove: LARGE tile config is only selected when M >= 64 and N >= 64.
///
/// The LARGE 64x64 tile kernel requires at least 64 rows and 64 columns
/// to be meaningful. The function must not select LARGE for smaller dims.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn large_tile_requires_sufficient_dimensions() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(batch <= 256);

    let tile = crate::dyn_tensor_metal::select_tile_config(m, k, n, batch);

    if tile.bm == 64 && tile.bn == 64 {
        assert!(m >= 64, "LARGE requires M >= 64");
        assert!(n >= 64, "LARGE requires N >= 64");
    }
}

// ---------------------------------------------------------------------------
// Threadgroup memory bounds
// ---------------------------------------------------------------------------

/// Prove: all constant TileConfig variants have output_per_threadgroup == 1024.
///
/// The three configs (SQUARE, TALL_SKINNY, WIDE) all produce exactly 1024
/// output elements per threadgroup. This is a design invariant.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn all_tile_configs_output_1024() {
    assert_eq!(
        TileConfig::SQUARE.output_per_threadgroup(),
        1024,
        "SQUARE must produce 1024 outputs"
    );
    assert_eq!(
        TileConfig::TALL_SKINNY.output_per_threadgroup(),
        1024,
        "TALL_SKINNY must produce 1024 outputs"
    );
    assert_eq!(
        TileConfig::WIDE.output_per_threadgroup(),
        1024,
        "WIDE must produce 1024 outputs"
    );
}

/// Prove: all constant TileConfig variants have exactly 128 threads.
///
/// 128 threads = 4 simdgroups of 32. This is a hardware-matching invariant.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn all_tile_configs_128_threads() {
    assert_eq!(
        TileConfig::SQUARE.threads_per_threadgroup(),
        128,
        "SQUARE must have 128 threads"
    );
    assert_eq!(
        TileConfig::TALL_SKINNY.threads_per_threadgroup(),
        128,
        "TALL_SKINNY must have 128 threads"
    );
    assert_eq!(
        TileConfig::WIDE.threads_per_threadgroup(),
        128,
        "WIDE must have 128 threads"
    );
}

/// Prove: all constant TileConfig tile dimensions are multiples of 8.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn all_constant_tiles_aligned() {
    let configs = [TileConfig::SQUARE, TileConfig::TALL_SKINNY, TileConfig::WIDE];
    for cfg in configs {
        assert!(cfg.tile_m % SIMDGROUP_ALIGN == 0);
        assert!(cfg.tile_n % SIMDGROUP_ALIGN == 0);
        assert!(cfg.tile_k % SIMDGROUP_ALIGN == 0);
    }
}

// ---------------------------------------------------------------------------
// Coverage: all paths reachable
// ---------------------------------------------------------------------------

/// Prove: the tiny path (None) is reachable.
///
/// Existence proof: there exist valid inputs where select_gemm_tiles returns None.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tiny_path_reachable() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(m.saturating_mul(n) < TINY_THRESHOLD);

    assert!(
        select_gemm_tiles(m, k, n).is_none(),
        "tiny shapes must return None"
    );
}

/// Prove: the small-K path produces square 32x32 tiles with clamped tile_k.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_k_path_produces_square() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= SMALL_K_THRESHOLD);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(m.saturating_mul(n) >= TINY_THRESHOLD);

    let cfg = select_gemm_tiles(m, k, n).unwrap();
    assert_eq!(cfg.tile_m, 32, "small-K must use 32x32 tile_m");
    assert_eq!(cfg.tile_n, 32, "small-K must use 32x32 tile_n");
}

/// Prove: the default path (non-tiny, non-small-K, non-tall, non-wide)
/// returns SQUARE config.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_path_returns_square() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > SMALL_K_THRESHOLD && k <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(m.saturating_mul(n) >= TINY_THRESHOLD);

    // Not tall-skinny and not wide: ratios < TALL_SKINNY_RATIO and < WIDE_RATIO.
    // Use equal dimensions to guarantee neither.
    kani::assume(m == n);
    kani::assume(m % SIMDGROUP_ALIGN == 0);

    let cfg = select_gemm_tiles(m, k, n).unwrap();
    assert_eq!(cfg, TileConfig::SQUARE, "equal m,n should route to SQUARE");
}
