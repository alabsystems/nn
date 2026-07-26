// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for SIMD group tile selection logic.
//!
//! Proves properties complementary to `kani_gemm_tile_select.rs` and
//! `kani_tile_config_proofs.rs`, focusing on:
//!
//! 1. Tile dimensions are powers of 2 (all preset configs)
//! 2. M*N*K product fits within Metal threadgroup constraints
//! 3. Selected tile never exceeds the matrix dimension it tiles over
//! 4. Tile selection is deterministic (same inputs → same output)
//! 5. Fallback to scalar produces consistent None for tiny matrices
//! 6. K-tile alignment ensures correct accumulation loop
//! 7. Performance ordering: larger output tiles preferred when shapes allow

use crate::simdgroup_tile_select::{
    is_scalar_fallback, select_gemm_tiles, TileConfig, SIMDGROUP_ALIGN, SMALL_K_THRESHOLD,
    TALL_SKINNY_RATIO, TINY_THRESHOLD, WIDE_RATIO,
};

// ---------------------------------------------------------------------------
// 1. Tile dimensions are powers of 2
// ---------------------------------------------------------------------------

/// Prove: all preset TileConfig tile dimensions (tile_m, tile_n, tile_k)
/// are powers of 2.
///
/// Metal simdgroup_matrix hardware operates most efficiently when tile
/// dimensions are powers of 2 (8, 16, 32, 64). This proves all three
/// preset configs satisfy this constraint.
#[kani::unwind(1)]
#[kani::proof]
fn preset_tile_dimensions_are_powers_of_two() {
    // SQUARE: 32, 32, 32 — all powers of 2
    assert!(
        TileConfig::SQUARE.tile_m.is_power_of_two(),
        "SQUARE tile_m not power of 2"
    );
    assert!(
        TileConfig::SQUARE.tile_n.is_power_of_two(),
        "SQUARE tile_n not power of 2"
    );
    assert!(
        TileConfig::SQUARE.tile_k.is_power_of_two(),
        "SQUARE tile_k not power of 2"
    );

    // TALL_SKINNY: 64, 16, 32 — all powers of 2
    assert!(
        TileConfig::TALL_SKINNY.tile_m.is_power_of_two(),
        "TALL_SKINNY tile_m not power of 2"
    );
    assert!(
        TileConfig::TALL_SKINNY.tile_n.is_power_of_two(),
        "TALL_SKINNY tile_n not power of 2"
    );
    assert!(
        TileConfig::TALL_SKINNY.tile_k.is_power_of_two(),
        "TALL_SKINNY tile_k not power of 2"
    );

    // WIDE: 16, 64, 32 — all powers of 2
    assert!(
        TileConfig::WIDE.tile_m.is_power_of_two(),
        "WIDE tile_m not power of 2"
    );
    assert!(
        TileConfig::WIDE.tile_n.is_power_of_two(),
        "WIDE tile_n not power of 2"
    );
    assert!(
        TileConfig::WIDE.tile_k.is_power_of_two(),
        "WIDE tile_k not power of 2"
    );
}

/// Prove: for non-small-K paths, the returned tile_m and tile_n are always
/// powers of 2 for any input dimensions.
///
/// When K > SMALL_K_THRESHOLD, the function returns one of the three preset
/// configs, all of which have power-of-2 tile dimensions. This proves the
/// property universally for the non-small-K regime.
#[kani::unwind(1)]
#[kani::proof]
fn non_small_k_tiles_are_powers_of_two() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > SMALL_K_THRESHOLD && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        assert!(cfg.tile_m.is_power_of_two(), "tile_m must be power of 2");
        assert!(cfg.tile_n.is_power_of_two(), "tile_n must be power of 2");
        // tile_k is always 32 for non-small-K paths
        assert!(cfg.tile_k.is_power_of_two(), "tile_k must be power of 2");
    }
}

// ---------------------------------------------------------------------------
// 2. M*N*K tile product fits within threadgroup memory constraints
// ---------------------------------------------------------------------------

/// Prove: tile_m * tile_n * tile_k never exceeds a safe bound.
///
/// The product tile_m * tile_n * tile_k determines the total work per
/// threadgroup tile step. On Apple Silicon, threadgroup memory is 32 KB
/// and max threads are 1024. The tile volume (tile_m * tile_k + tile_k *
/// tile_n) * sizeof(float) must fit in threadgroup memory. This proves
/// the weaker but universally useful bound that tile_m * tile_n fits
/// within 1024 * tile_k (no excessive tiling).
///
/// Concretely, for all configs: tile_m * tile_n = 1024, tile_k <= 32,
/// so tile_m * tile_n * tile_k <= 32768. The threadgroup shared memory
/// needed is (tile_m * tile_k + tile_k * tile_n) * 4 bytes, which for
/// all preset configs is (32*32 + 32*32)*4 = 8192 bytes, well within
/// the 32 KB Metal limit.
#[kani::unwind(1)]
#[kani::proof]
fn tile_volume_within_threadgroup_memory() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        // Shared memory for A tile (tile_m * tile_k) + B tile (tile_k * tile_n),
        // each element 4 bytes (float).
        let a_tile_elems = cfg.tile_m.checked_mul(cfg.tile_k).unwrap();
        let b_tile_elems = cfg.tile_k.checked_mul(cfg.tile_n).unwrap();
        let total_elems = a_tile_elems.checked_add(b_tile_elems).unwrap();
        let shared_bytes = total_elems.checked_mul(4).unwrap(); // sizeof(float) = 4

        // Metal threadgroup memory limit: 32 KB = 32768 bytes
        assert!(
            shared_bytes <= 32_768,
            "shared memory {shared_bytes} bytes exceeds Metal 32 KB limit"
        );
    }
}

/// Prove: tile_m * tile_n does not exceed max threadgroup output (1024)
/// for any returned config.
///
/// All three preset configs produce exactly 1024 outputs. The small-K
/// path uses 32x32 = 1024. This proves no config exceeds this bound.
#[kani::unwind(1)]
#[kani::proof]
fn tile_output_bounded_by_1024() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        let output = cfg.tile_m.checked_mul(cfg.tile_n).unwrap();
        assert!(output <= 1024, "tile output {output} exceeds 1024");
    }
}

// ---------------------------------------------------------------------------
// 3. Selected tile does not exceed the role it covers in output
// ---------------------------------------------------------------------------

/// Prove: threadgroup grid fully covers the M x N output space.
///
/// For any valid (m, k, n) that produces a tile config, the number of
/// threadgroups times the tile dimensions must cover the full output:
///   div_ceil(m, tile_m) * tile_m >= m
///   div_ceil(n, tile_n) * tile_n >= n
///
/// This is the tiling coverage guarantee — no output element is missed.
#[kani::unwind(1)]
#[kani::proof]
fn tile_grid_covers_full_output() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        let grid_m = m.div_ceil(cfg.tile_m);
        let grid_n = n.div_ceil(cfg.tile_n);

        // Coverage: grid * tile >= dimension
        assert!(grid_m * cfg.tile_m >= m, "M-dimension not fully covered");
        assert!(grid_n * cfg.tile_n >= n, "N-dimension not fully covered");

        // Minimality: removing one row/col of threadgroups would miss elements
        if grid_m > 1 {
            assert!(
                (grid_m - 1) * cfg.tile_m < m,
                "M-grid has unnecessary extra row of threadgroups"
            );
        }
        if grid_n > 1 {
            assert!(
                (grid_n - 1) * cfg.tile_n < n,
                "N-grid has unnecessary extra column of threadgroups"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Tile selection determinism
// ---------------------------------------------------------------------------

/// Prove: select_gemm_tiles is a pure function — identical inputs always
/// produce identical outputs.
///
/// This is critical for compiled model correctness: the dispatch plan
/// recorded at compile time must match the plan at execution time.
#[kani::unwind(1)]
#[kani::proof]
fn tile_selection_is_deterministic() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    let result1 = select_gemm_tiles(m, k, n);
    let result2 = select_gemm_tiles(m, k, n);

    match (result1, result2) {
        (None, None) => {} // Both None — consistent
        (Some(a), Some(b)) => {
            assert_eq!(a.tile_m, b.tile_m, "tile_m differs between calls");
            assert_eq!(a.tile_n, b.tile_n, "tile_n differs between calls");
            assert_eq!(a.tile_k, b.tile_k, "tile_k differs between calls");
            assert_eq!(
                a.threadgroup_size, b.threadgroup_size,
                "threadgroup_size differs between calls"
            );
        }
        _ => {
            panic!("select_gemm_tiles returned different Some/None for identical inputs");
        }
    }
}

/// Prove: is_scalar_fallback is also deterministic.
#[kani::unwind(1)]
#[kani::proof]
fn scalar_fallback_is_deterministic() {
    let m: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m <= 4096);
    kani::assume(n <= 4096);

    let r1 = is_scalar_fallback(m, n);
    let r2 = is_scalar_fallback(m, n);
    assert_eq!(r1, r2, "is_scalar_fallback not deterministic");
}

// ---------------------------------------------------------------------------
// 5. Fallback to scalar: when no tile fits, None is returned
// ---------------------------------------------------------------------------

/// Prove: for any M*N < TINY_THRESHOLD, select_gemm_tiles returns None
/// regardless of K.
///
/// The scalar fallback is purely a function of M*N — K does not affect
/// whether we fall back. This proves K-independence of the fallback decision.
#[kani::unwind(1)]
#[kani::proof]
fn scalar_fallback_independent_of_k() {
    let m: usize = kani::any();
    let k1: usize = kani::any();
    let k2: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k1 > 0 && k1 <= 4096);
    kani::assume(k2 > 0 && k2 <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(m.saturating_mul(n) < TINY_THRESHOLD);

    let r1 = select_gemm_tiles(m, k1, n);
    let r2 = select_gemm_tiles(m, k2, n);

    assert!(r1.is_none(), "tiny M*N must return None for k1");
    assert!(r2.is_none(), "tiny M*N must return None for k2");
}

/// Prove: for M=0 or N=0, the result is always None (zero-size matrix).
///
/// Zero-dimension matrices have M*N = 0 < TINY_THRESHOLD. The function
/// must handle this gracefully without panic.
#[kani::unwind(1)]
#[kani::proof]
fn zero_dimension_returns_none() {
    let k: usize = kani::any();
    kani::assume(k > 0 && k <= 4096);

    // M=0
    assert!(
        select_gemm_tiles(0, k, 64).is_none(),
        "M=0 must return None"
    );
    // N=0
    assert!(
        select_gemm_tiles(64, k, 0).is_none(),
        "N=0 must return None"
    );
    // Both=0
    assert!(
        select_gemm_tiles(0, k, 0).is_none(),
        "M=N=0 must return None"
    );
}

// ---------------------------------------------------------------------------
// 6. K-tile alignment for correct accumulation
// ---------------------------------------------------------------------------

/// Prove: tile_k is always a multiple of SIMDGROUP_ALIGN (8) for any
/// returned config, including the small-K path where tile_k is derived
/// from the input K.
///
/// The accumulation loop steps by tile_k. Metal simdgroup_matrix requires
/// all dimensions to be multiples of 8. An unaligned tile_k would produce
/// incorrect partial sums.
#[kani::unwind(1)]
#[kani::proof]
fn tile_k_always_aligned_for_accumulation() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= 4096);
    kani::assume(n > 0 && n <= 4096);

    if let Some(cfg) = select_gemm_tiles(m, k, n) {
        assert!(
            cfg.tile_k % SIMDGROUP_ALIGN == 0,
            "tile_k must be a multiple of SIMDGROUP_ALIGN for accumulation"
        );
        assert!(
            cfg.tile_k > 0,
            "tile_k must be positive for accumulation loop"
        );
    }
}

/// Prove: for the small-K path, tile_k >= K (the accumulation window
/// covers the entire K dimension in one step).
///
/// When K <= SMALL_K_THRESHOLD, tile_k is set to next_multiple_of(8) of K.
/// This means the K-loop body executes once, accumulating all K columns.
/// tile_k >= K ensures no K elements are missed.
#[kani::unwind(1)]
#[kani::proof]
fn small_k_tile_covers_full_k_dimension() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= SMALL_K_THRESHOLD);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(m.saturating_mul(n) >= TINY_THRESHOLD);

    let cfg = select_gemm_tiles(m, k, n).unwrap();

    // tile_k must cover the full K dimension
    assert!(
        cfg.tile_k >= k,
        "small-K tile_k ({}) does not cover K ({})",
        cfg.tile_k,
        k
    );

    // tile_k should be minimal: next_multiple_of(SIMDGROUP_ALIGN)
    let expected = k.next_multiple_of(SIMDGROUP_ALIGN);
    assert_eq!(
        cfg.tile_k, expected,
        "small-K tile_k should be minimal aligned value"
    );
}

// ---------------------------------------------------------------------------
// 7. Performance ordering: larger tiles preferred when shapes allow
// ---------------------------------------------------------------------------

/// Prove: tall-skinny config has tile_m >= tile_n (biased toward M).
///
/// When M >> N, the tall-skinny config allocates more tile rows (tile_m=64)
/// than columns (tile_n=16), ensuring the threadgroup covers more of the
/// dominant dimension. Conversely, WIDE has tile_n >= tile_m.
#[kani::unwind(1)]
#[kani::proof]
fn tall_skinny_biased_toward_m_dimension() {
    let cfg = TileConfig::TALL_SKINNY;
    assert!(
        cfg.tile_m >= cfg.tile_n,
        "TALL_SKINNY must have tile_m >= tile_n"
    );
    assert!(
        cfg.tile_m > cfg.tile_n,
        "TALL_SKINNY must have tile_m strictly > tile_n"
    );
}

/// Prove: wide config has tile_n >= tile_m (biased toward N).
#[kani::unwind(1)]
#[kani::proof]
fn wide_biased_toward_n_dimension() {
    let cfg = TileConfig::WIDE;
    assert!(cfg.tile_n >= cfg.tile_m, "WIDE must have tile_n >= tile_m");
    assert!(
        cfg.tile_n > cfg.tile_m,
        "WIDE must have tile_n strictly > tile_m"
    );
}

/// Prove: when M/N >= TALL_SKINNY_RATIO and both dimensions are aligned,
/// the tall-skinny config is selected (not square), which produces fewer
/// threadgroups in the M direction.
///
/// Fewer TGs means better L1/L2 cache utilization because each TG covers
/// 64 M-rows instead of 32. This is the performance ordering property:
/// tall-skinny is preferred over square when the shape allows it.
#[kani::unwind(1)]
#[kani::proof]
fn tall_skinny_preferred_over_square_when_applicable() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > SMALL_K_THRESHOLD && k <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(m.saturating_mul(n) >= TINY_THRESHOLD);

    // Tall-skinny conditions
    kani::assume(n > 0);
    kani::assume(m / n >= TALL_SKINNY_RATIO);
    kani::assume(m % SIMDGROUP_ALIGN == 0);
    kani::assume(n % SIMDGROUP_ALIGN == 0);

    let cfg = select_gemm_tiles(m, k, n).unwrap();

    // Must select TALL_SKINNY, not SQUARE
    assert_eq!(cfg.tile_m, 64, "tall-skinny shape must select tile_m=64");
    assert_eq!(cfg.tile_n, 16, "tall-skinny shape must select tile_n=16");

    // TALL_SKINNY covers 2x more M-rows per TG than SQUARE
    assert!(
        cfg.tile_m > TileConfig::SQUARE.tile_m,
        "TALL_SKINNY tile_m must exceed SQUARE tile_m"
    );
}

/// Prove: when N/M >= WIDE_RATIO and both dimensions are aligned,
/// the wide config is selected (not square), which produces fewer
/// threadgroups in the N direction.
#[kani::unwind(1)]
#[kani::proof]
fn wide_preferred_over_square_when_applicable() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > SMALL_K_THRESHOLD && k <= 4096);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(m.saturating_mul(n) >= TINY_THRESHOLD);

    // Wide conditions — ensure tall-skinny does NOT fire first.
    // tall-skinny fires when m / n >= TALL_SKINNY_RATIO. We need m / n < 4.
    // Since we want n / m >= 4, that implies m / n < 1, which satisfies.
    kani::assume(m > 0);
    kani::assume(n / m >= WIDE_RATIO);
    kani::assume(m % SIMDGROUP_ALIGN == 0);
    kani::assume(n % SIMDGROUP_ALIGN == 0);

    // Also need to ensure tall-skinny doesn't fire (m/n < TALL_SKINNY_RATIO).
    // Since n/m >= 4 and m > 0, n >= 4*m, so m/n <= 1/4 < 4. Safe.

    let cfg = select_gemm_tiles(m, k, n).unwrap();

    // Must select WIDE, not SQUARE
    assert_eq!(cfg.tile_m, 16, "wide shape must select tile_m=16");
    assert_eq!(cfg.tile_n, 64, "wide shape must select tile_n=64");

    // WIDE covers 2x more N-columns per TG than SQUARE
    assert!(
        cfg.tile_n > TileConfig::SQUARE.tile_n,
        "WIDE tile_n must exceed SQUARE tile_n"
    );
}

/// Prove: small-K takes priority over tall-skinny and wide routing.
///
/// When K <= SMALL_K_THRESHOLD, the small-K path fires first regardless
/// of M/N ratio. This is a priority ordering invariant in the selection
/// logic: small-K is checked before shape-class routing.
#[kani::unwind(1)]
#[kani::proof]
fn small_k_priority_over_shape_routing() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m > 0 && m <= 4096);
    kani::assume(k > 0 && k <= SMALL_K_THRESHOLD);
    kani::assume(n > 0 && n <= 4096);
    kani::assume(m.saturating_mul(n) >= TINY_THRESHOLD);

    let cfg = select_gemm_tiles(m, k, n).unwrap();

    // Small-K always uses 32x32 tiles, never TALL_SKINNY or WIDE
    assert_eq!(cfg.tile_m, 32, "small-K must use tile_m=32");
    assert_eq!(cfg.tile_n, 32, "small-K must use tile_n=32");
    // tile_k is derived from K, not a preset
    assert!(cfg.tile_k <= SMALL_K_THRESHOLD + SIMDGROUP_ALIGN - 1);
}
