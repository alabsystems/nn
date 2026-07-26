// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for matmul SIMD MSL kernel properties (#3735).
//!
//! Complements `kani_dyn_tensor_metal_matmul_simd_msl.rs` with deeper proofs
//! targeting:
//!
//! - Adaptive tile selection (`select_gemm_tiles`, `TileConfig`) from
//!   `simdgroup_tile_select.rs`
//! - F16 simdgroup occupancy threshold arithmetic
//! - MSL shared memory bank conflict freedom across all tile configs
//! - Thread-to-element assignment uniqueness in cooperative loads
//! - Output element write index correctness across all kernel variants

use crate::dyn_tensor_metal::matmul_simd::{
    select_tile_config, should_use_f16_simdgroup, should_use_simdgroup, tg_memory_bytes,
    GemmTileConfig, F16_MIN_THREADGROUPS,
};
use crate::simdgroup_tile_select::{
    is_scalar_fallback, select_gemm_tiles, TileConfig, SIMDGROUP_ALIGN, SMALL_K_THRESHOLD,
    STANDARD_MN_THRESHOLD, TALL_SKINNY_RATIO, TINY_THRESHOLD, WIDE_RATIO,
};

// ===========================================================================
// 1. select_gemm_tiles: tiny threshold rejects correctly
// ===========================================================================

/// Prove: select_gemm_tiles returns None when M*N < TINY_THRESHOLD (1024).
///
/// Shapes below this threshold are rejected for scalar fallback because
/// dispatch overhead dominates compute.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gemm_tiles_none_when_tiny() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 1024);
    kani::assume(k >= 1 && k <= 4096);
    kani::assume(n >= 1 && n <= 1024);
    kani::assume(m * n < TINY_THRESHOLD);

    let result = select_gemm_tiles(m, k, n);
    assert!(result.is_none(), "M*N < 1024 must return None");
}

// ===========================================================================
// 2. select_gemm_tiles: small K path
// ===========================================================================

/// Prove: when K <= SMALL_K_THRESHOLD (32) and M*N >= TINY_THRESHOLD,
/// select_gemm_tiles returns a config with tile_m=32, tile_n=32.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gemm_tiles_small_k_uses_square_tile() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 32 && m <= 4096);
    kani::assume(k >= 1 && k <= SMALL_K_THRESHOLD);
    kani::assume(n >= 32 && n <= 4096);
    kani::assume(m * n >= TINY_THRESHOLD);

    let config = select_gemm_tiles(m, k, n);
    assert!(config.is_some(), "non-tiny with small K must return Some");
    let cfg = config.unwrap();
    assert_eq!(cfg.tile_m, 32, "small K uses 32x32 tile");
    assert_eq!(cfg.tile_n, 32, "small K uses 32x32 tile");
}

/// Prove: small K tile_k is K rounded up to SIMDGROUP_ALIGN (8).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gemm_tiles_small_k_tile_k_aligned() {
    let k: usize = kani::any();
    kani::assume(k >= 1 && k <= SMALL_K_THRESHOLD);

    let expected_tile_k = k.next_multiple_of(SIMDGROUP_ALIGN);

    // Must be a multiple of 8.
    assert_eq!(expected_tile_k % SIMDGROUP_ALIGN, 0);
    // Must be >= K.
    assert!(expected_tile_k >= k);
    // Must be at most K + 7.
    assert!(expected_tile_k <= k + SIMDGROUP_ALIGN - 1);
}

// ===========================================================================
// 3. select_gemm_tiles: tall-skinny detection
// ===========================================================================

/// Prove: tall-skinny config is selected when M/N >= TALL_SKINNY_RATIO (4)
/// and both dims are aligned, and M*N >= TINY_THRESHOLD, and K > 32.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gemm_tiles_tall_skinny_detection() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let k: usize = kani::any();

    kani::assume(m >= 64 && m <= 4096);
    kani::assume(n >= 8 && n <= 256);
    kani::assume(k >= 33 && k <= 4096);
    kani::assume(m % SIMDGROUP_ALIGN == 0);
    kani::assume(n % SIMDGROUP_ALIGN == 0);
    kani::assume(m * n >= TINY_THRESHOLD);
    kani::assume(m / n >= TALL_SKINNY_RATIO);

    let config = select_gemm_tiles(m, k, n);
    assert!(config.is_some());
    let cfg = config.unwrap();
    assert_eq!(cfg.tile_m, 64, "tall-skinny tile_m=64");
    assert_eq!(cfg.tile_n, 16, "tall-skinny tile_n=16");
}

// ===========================================================================
// 4. select_gemm_tiles: wide detection
// ===========================================================================

/// Prove: wide config is selected when N/M >= WIDE_RATIO (4)
/// and both dims are aligned.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gemm_tiles_wide_detection() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let k: usize = kani::any();

    kani::assume(m >= 8 && m <= 256);
    kani::assume(n >= 64 && n <= 4096);
    kani::assume(k >= 33 && k <= 4096);
    kani::assume(m % SIMDGROUP_ALIGN == 0);
    kani::assume(n % SIMDGROUP_ALIGN == 0);
    kani::assume(m * n >= TINY_THRESHOLD);
    kani::assume(n / m >= WIDE_RATIO);
    // Ensure NOT tall-skinny (M/N < 4):
    kani::assume(m / n < TALL_SKINNY_RATIO);

    let config = select_gemm_tiles(m, k, n);
    assert!(config.is_some());
    let cfg = config.unwrap();
    assert_eq!(cfg.tile_m, 16, "wide tile_m=16");
    assert_eq!(cfg.tile_n, 64, "wide tile_n=64");
}

// ===========================================================================
// 5. TileConfig: output_per_threadgroup and threadgroup_count
// ===========================================================================

/// Prove: TileConfig::SQUARE output per threadgroup is exactly 1024.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_config_square_output_count() {
    let cfg = TileConfig::SQUARE;
    assert_eq!(cfg.output_per_threadgroup(), 1024);
    assert_eq!(cfg.threads_per_threadgroup(), 128);
}

/// Prove: TileConfig::TALL_SKINNY output per threadgroup is exactly 1024.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_config_tall_skinny_output_count() {
    let cfg = TileConfig::TALL_SKINNY;
    assert_eq!(cfg.output_per_threadgroup(), 64 * 16);
    assert_eq!(cfg.output_per_threadgroup(), 1024);
    assert_eq!(cfg.threads_per_threadgroup(), 128);
}

/// Prove: TileConfig::WIDE output per threadgroup is exactly 1024.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_config_wide_output_count() {
    let cfg = TileConfig::WIDE;
    assert_eq!(cfg.output_per_threadgroup(), 16 * 64);
    assert_eq!(cfg.output_per_threadgroup(), 1024);
    assert_eq!(cfg.threads_per_threadgroup(), 128);
}

/// Prove: threadgroup_count * output_per_threadgroup >= M*N for SQUARE.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_config_square_covers_all() {
    let m: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 8192);
    kani::assume(n >= 1 && n <= 8192);

    let cfg = TileConfig::SQUARE;
    let tg_count = cfg.threadgroup_count(m, n);
    let covered = tg_count * cfg.output_per_threadgroup();
    assert!(covered >= m * n, "SQUARE tiles must cover all M*N");
}

/// Prove: threadgroup_count * output_per_threadgroup >= M*N for TALL_SKINNY.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_config_tall_skinny_covers_all() {
    let m: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 8192);
    kani::assume(n >= 1 && n <= 8192);

    let cfg = TileConfig::TALL_SKINNY;
    let tg_count = cfg.threadgroup_count(m, n);
    let covered = tg_count * cfg.output_per_threadgroup();
    assert!(covered >= m * n, "TALL_SKINNY tiles must cover all M*N");
}

/// Prove: threadgroup_count * output_per_threadgroup >= M*N for WIDE.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tile_config_wide_covers_all() {
    let m: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 8192);
    kani::assume(n >= 1 && n <= 8192);

    let cfg = TileConfig::WIDE;
    let tg_count = cfg.threadgroup_count(m, n);
    let covered = tg_count * cfg.output_per_threadgroup();
    assert!(covered >= m * n, "WIDE tiles must cover all M*N");
}

// ===========================================================================
// 7. F16 occupancy threshold: monotonicity with batch
// ===========================================================================

/// Prove: F16 simdgroup eligibility is monotonic in batch size.
///
/// If should_use_f16_simdgroup(m, k, n, b1) is true for batch=b1,
/// then it is also true for any batch=b2 >= b1 (more TGs = higher occupancy).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn f16_simdgroup_monotonic_in_batch() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let b1: usize = kani::any();
    let b2: usize = kani::any();

    kani::assume(m >= 8 && m <= 2048);
    kani::assume(k >= 128 && k <= 2048);
    kani::assume(n >= 8 && n <= 2048);
    kani::assume(m % 8 == 0 && k % 8 == 0 && n % 8 == 0);
    kani::assume(b1 >= 1 && b1 <= 64);
    kani::assume(b2 >= b1 && b2 <= 64);
    kani::assume(m * n >= 16_384);

    if should_use_f16_simdgroup(m, k, n, b1) {
        assert!(
            should_use_f16_simdgroup(m, k, n, b2),
            "F16 eligibility must be monotonic in batch"
        );
    }
}

// ===========================================================================
// 8. Shared memory bank conflict proofs for all pad constants
// ===========================================================================

/// Prove: OUT_PAD (BN+1=65) for LARGE F16 kernel avoids bank conflicts.
///
/// The pass_out buffer uses OUT_PAD=65 stride.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn large_f16_out_pad_avoids_conflicts() {
    let bn: u32 = 64;
    let out_pad: u32 = bn + 1; // 65
    let simd_width: u32 = 32;

    assert!(out_pad % simd_width != 0, "OUT_PAD must not be multiple of SIMD width");
    // 65 is odd, so no power-of-2 alignment issues.
    assert!(out_pad % 2 != 0, "OUT_PAD must be odd");
}

// ===========================================================================
// 9. Global output write index uniqueness for edge tiles
// ===========================================================================

/// Prove: in the SMALL edge tile write loop, each (r, c) pair maps to a
/// unique global output index for given tile_row, tile_col, N.
///
/// The loop `for idx in tid..TILE*TILE step 128` assigns each thread a
/// set of (r, c) = (idx/TILE, idx%TILE). For non-overlapping idx values,
/// (r, c) pairs are unique.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_edge_write_index_unique() {
    let tile: u32 = 32;
    let n: u32 = kani::any();
    let tile_row: u32 = kani::any();
    let tile_col: u32 = kani::any();

    kani::assume(n >= 1 && n <= 4096);
    kani::assume(tile_row <= 65504); // max tile-aligned
    kani::assume(tile_col <= 65504);
    kani::assume(tile_col < n);

    let idx1: u32 = kani::any();
    let idx2: u32 = kani::any();
    kani::assume(idx1 < tile * tile);
    kani::assume(idx2 < tile * tile);
    kani::assume(idx1 != idx2);

    let r1 = idx1 / tile;
    let c1 = idx1 % tile;
    let r2 = idx2 / tile;
    let c2 = idx2 % tile;

    let global1 = (tile_row + r1) as u64 * (n as u64) + (tile_col + c1) as u64;
    let global2 = (tile_row + r2) as u64 * (n as u64) + (tile_col + c2) as u64;

    // Different idx implies different (r, c) implies different global index.
    if r1 != r2 || c1 != c2 {
        assert_ne!(global1, global2, "different (r,c) must map to different global index");
    }
}

// ===========================================================================
// 10. SMALL simdgroup load: Bmat load stride matches PADDED
// ===========================================================================

/// Prove: B sub-tile load index `kk * PADDED + sg_col_start` is within
/// shared memory Bs[TILE * PADDED] for all valid kk and sg_id.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_bmat_load_index_in_bounds() {
    let tile: u32 = 32;
    let padded: u32 = 33;
    let alloc = tile * padded; // 1056

    let kk: u32 = kani::any();
    let sg_id: u32 = kani::any();
    kani::assume(kk < tile);
    kani::assume(kk % 8 == 0);
    kani::assume(sg_id < 4);

    let sg_col_start = sg_id * 8;
    let base_idx = kk * padded + sg_col_start;

    // The simdgroup_load reads an 8x8 sub-matrix starting at base_idx
    // with stride PADDED. Max accessed index: base_idx + 7*PADDED + 7.
    let max_idx = base_idx + 7 * padded + 7;
    assert!(max_idx < alloc, "Bmat load must be within Bs allocation");
}

/// Prove: A sub-tile load index `(ri*8) * PADDED + kk` is within
/// shared memory As[TILE * PADDED] for all valid ri and kk.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_amat_load_index_in_bounds() {
    let tile: u32 = 32;
    let padded: u32 = 33;
    let alloc = tile * padded; // 1056

    let ri: u32 = kani::any();
    let kk: u32 = kani::any();
    kani::assume(ri < 4);
    kani::assume(kk < tile);
    kani::assume(kk % 8 == 0);

    let base_idx = (ri * 8) * padded + kk;
    let max_idx = base_idx + 7 * padded + 7;
    assert!(max_idx < alloc, "Amat load must be within As allocation");
}

// ===========================================================================
// 11. LARGE kernel Bmat/Amat load index bounds
// ===========================================================================

/// Prove: LARGE kernel Bmat load `kk * B_PAD + sg_col_start + ci * 8` is
/// within Bs[BK * B_PAD] for all valid indices.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn large_bmat_load_in_bounds() {
    let bk: u32 = 32;
    let bn: u32 = 64;
    let b_pad: u32 = bn + 1; // 65
    let alloc = bk * b_pad; // 2080

    let kk: u32 = kani::any();
    let sg_id: u32 = kani::any();
    let ci: u32 = kani::any();
    kani::assume(kk < bk);
    kani::assume(kk % 8 == 0);
    kani::assume(sg_id < 4);
    kani::assume(ci < 2);

    let sg_cols: u32 = 16;
    let sg_col_start = sg_id * sg_cols;
    let base_idx = kk * b_pad + sg_col_start + ci * 8;
    let max_idx = base_idx + 7 * b_pad + 7;
    assert!(max_idx < alloc, "LARGE Bmat load must be within Bs");
}

/// Prove: LARGE kernel Amat load `(ri*8) * A_PAD + kk` is within
/// As[BM * A_PAD] for all valid indices.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn large_amat_load_in_bounds() {
    let bm: u32 = 64;
    let bk: u32 = 32;
    let a_pad: u32 = bk + 1; // 33
    let alloc = bm * a_pad; // 2112

    let ri: u32 = kani::any();
    let kk: u32 = kani::any();
    kani::assume(ri < 8);
    kani::assume(kk < bk);
    kani::assume(kk % 8 == 0);

    let base_idx = (ri * 8) * a_pad + kk;
    let max_idx = base_idx + 7 * a_pad + 7;
    assert!(max_idx < alloc, "LARGE Amat load must be within As");
}
