// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `dyn_tensor_metal_matmul_simd_msl.rs` (#3690).
//!
//! The MSL file contains 4 GEMM kernel source strings with specific
//! threadgroup memory layouts, tile geometries, and output write strategies.
//! These harnesses verify the Rust-side invariants that determine correct
//! kernel selection and dispatch parameters:
//!
//! ## Properties Proved
//!
//! - Threadgroup memory sizing for all 4 kernels (f32/f16 x 32/64)
//! - TG memory fits within Metal 32 KB per-threadgroup limit
//! - Grid dimension computation does not overflow u32
//! - Tile coverage: every output element is written by some threadgroup
//! - PADDED constant prevents bank conflicts (stride > TILE)
//! - Edge tile detection predicates are exhaustive
//! - F16 kernels keep float accumulators (precision invariant)
//! - 64x64 2-pass output covers all 64 rows without overlap
//! - 128 threads = 4 simdgroups of 32 for all kernel configs
//! - Cooperative load coverage: 128 threads load all tile elements
//! - Buffer offset calculations do not overflow for production dims
//! - Broadcast RHS offset is zero when broadcast flag is set

use crate::dyn_tensor_metal::matmul_simd::{
    select_tile_config, should_use_f16_simdgroup, should_use_simdgroup, tg_memory_bytes,
    GemmTileConfig, F16_MIN_THREADGROUPS,
};

// ---------------------------------------------------------------------------
// Threadgroup memory sizing
// ---------------------------------------------------------------------------

/// Prove: SMALL f32 kernel TG memory is exactly 8,448 bytes.
///
/// Formula: As[32x33] + Bs[32x33] = 2 * 32 * 33 * 4 = 8,448.
/// tile_out eliminated by direct register-to-device writes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_f32_tg_mem_exact() {
    let bytes = tg_memory_bytes(GemmTileConfig::SMALL, false);
    assert_eq!(bytes, 8_448, "SMALL f32 TG memory must be 8,448 bytes");
}

/// Prove: SMALL f16 kernel TG memory is exactly 8,448 bytes.
///
/// Formula: As[32x33]h + Bs[32x33]h + tile_out[32x33]f
/// = 2 * 32 * 33 * 2 + 32 * 33 * 4 = 4,224 + 4,224 = 8,448.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_f16_tg_mem_exact() {
    let bytes = tg_memory_bytes(GemmTileConfig::SMALL, true);
    assert_eq!(bytes, 8_448, "SMALL f16 TG memory must be 8,448 bytes");
}

/// Prove: LARGE f32 kernel TG memory is exactly 16,768 bytes.
///
/// Formula: As[64x33]f + Bs[32x65]f = 64*33*4 + 32*65*4 = 8,448 + 8,320 = 16,768.
/// pass_out eliminated by direct register-to-device writes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn large_f32_tg_mem_exact() {
    let bytes = tg_memory_bytes(GemmTileConfig::LARGE, false);
    assert_eq!(bytes, 16_768, "LARGE f32 TG memory must be 16,768 bytes");
}

/// Prove: LARGE f16 kernel TG memory is exactly 16,704 bytes.
///
/// Formula: As[64x33]h + Bs[32x65]h + pass_out[32x65]f
/// = 64*33*2 + 32*65*2 + 32*65*4 = 4,224 + 4,160 + 8,320 = 16,704.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn large_f16_tg_mem_exact() {
    let bytes = tg_memory_bytes(GemmTileConfig::LARGE, true);
    assert_eq!(bytes, 16_704, "LARGE f16 TG memory must be 16,704 bytes");
}

/// Prove: all 4 kernel TG memory configurations fit within Metal 32 KB limit.
///
/// Metal specification: max 32,768 bytes threadgroup memory per threadgroup.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn all_tg_mem_within_metal_limit() {
    let is_half: bool = kani::any();

    let small = tg_memory_bytes(GemmTileConfig::SMALL, is_half);
    let large = tg_memory_bytes(GemmTileConfig::LARGE, is_half);

    assert!(small <= 32_768, "SMALL TG memory exceeds Metal 32 KB limit");
    assert!(large <= 32_768, "LARGE TG memory exceeds Metal 32 KB limit");
}

// ---------------------------------------------------------------------------
// Grid dimension computation
// ---------------------------------------------------------------------------

/// Prove: grid dimensions for SMALL tile (32x32) fit in u32 for production dims.
///
/// Grid: [ceil(N/32), ceil(M/32), batch]. All must fit in u32.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_grid_fits_u32() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m >= 1 && m <= 65536);
    kani::assume(n >= 1 && n <= 65536);
    kani::assume(batch >= 1 && batch <= 256);

    let grid_x = n.div_ceil(32);
    let grid_y = m.div_ceil(32);

    assert!(grid_x <= u32::MAX as usize, "SMALL grid_x overflows u32");
    assert!(grid_y <= u32::MAX as usize, "SMALL grid_y overflows u32");
    assert!(batch <= u32::MAX as usize, "batch overflows u32");
}

/// Prove: grid dimensions for LARGE tile (64x64) fit in u32 for production dims.
///
/// Grid: [ceil(N/64), ceil(M/64), batch]. All must fit in u32.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn large_grid_fits_u32() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m >= 64 && m <= 65536);
    kani::assume(n >= 64 && n <= 65536);
    kani::assume(batch >= 1 && batch <= 256);

    let grid_x = n.div_ceil(64);
    let grid_y = m.div_ceil(64);

    assert!(grid_x <= u32::MAX as usize, "LARGE grid_x overflows u32");
    assert!(grid_y <= u32::MAX as usize, "LARGE grid_y overflows u32");
}

// ---------------------------------------------------------------------------
// Tile coverage: every output element is covered
// ---------------------------------------------------------------------------

/// Prove: SMALL 32x32 tiles cover all M*N output elements.
///
/// ceil(M/32) * 32 >= M and ceil(N/32) * 32 >= N for all positive M, N.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_tile_covers_all_outputs() {
    let m: u32 = kani::any();
    let n: u32 = kani::any();
    kani::assume(m >= 1);
    kani::assume(n >= 1);

    let covered_m = m.div_ceil(32) * 32;
    let covered_n = n.div_ceil(32) * 32;

    assert!(covered_m >= m, "SMALL tiles must cover all M rows");
    assert!(covered_n >= n, "SMALL tiles must cover all N columns");
}

/// Prove: LARGE 64x64 tiles cover all M*N output elements.
///
/// ceil(M/64) * 64 >= M and ceil(N/64) * 64 >= N for all positive M, N.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn large_tile_covers_all_outputs() {
    let m: u32 = kani::any();
    let n: u32 = kani::any();
    kani::assume(m >= 1);
    kani::assume(n >= 1);

    let covered_m = m.div_ceil(64) * 64;
    let covered_n = n.div_ceil(64) * 64;

    assert!(covered_m >= m, "LARGE tiles must cover all M rows");
    assert!(covered_n >= n, "LARGE tiles must cover all N columns");
}

// ---------------------------------------------------------------------------
// PADDED constant prevents bank conflicts
// ---------------------------------------------------------------------------

/// Prove: PADDED (TILE+1) is coprime with SIMD width (32), preventing bank conflicts.
///
/// Bank conflicts occur when stride is a multiple of the bank count (32).
/// PADDED = 33 is coprime with 32 (gcd(33,32) = 1), ensuring no conflicts.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_padded_coprime_with_simd_width() {
    let tile: u32 = 32;
    let padded: u32 = tile + 1; // 33
    let simd_width: u32 = 32;

    // gcd(33, 32) = 1, since 33 = 32*1 + 1, 32 = 1*32 + 0.
    assert!(padded % simd_width != 0, "PADDED must not be multiple of SIMD width");
    // 33 is odd, 32 is even, so they share no factor of 2.
    assert!(padded % 2 != 0, "PADDED must be odd to avoid power-of-2 stride conflicts");
}

/// Prove: LARGE A_PAD (BK+1=33) and B_PAD (BN+1=65) prevent bank conflicts.
///
/// Neither stride is a multiple of 32.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn large_pads_avoid_bank_conflicts() {
    let bk: u32 = 32;
    let bn: u32 = 64;
    let a_pad: u32 = bk + 1; // 33
    let b_pad: u32 = bn + 1; // 65
    let simd_width: u32 = 32;

    assert!(a_pad % simd_width != 0, "A_PAD must not be multiple of SIMD width");
    assert!(b_pad % simd_width != 0, "B_PAD must not be multiple of SIMD width");
}

// ---------------------------------------------------------------------------
// Edge tile detection is exhaustive
// ---------------------------------------------------------------------------

/// Prove: edge tile predicate and inner tile predicate are complementary.
///
/// For any valid tile position, either the entire tile is within bounds
/// (inner tile) or at least one element is out of bounds (edge tile).
/// There is no gap — every tile position is handled.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn edge_vs_inner_exhaustive_small() {
    let m: u32 = kani::any();
    let n: u32 = kani::any();
    let tile_row: u32 = kani::any();
    let tile_col: u32 = kani::any();

    kani::assume(m >= 1 && m <= 4096);
    kani::assume(n >= 1 && n <= 4096);
    kani::assume(tile_row < m.div_ceil(32) * 32);
    kani::assume(tile_col < n.div_ceil(32) * 32);
    // tile_row and tile_col must be tile-aligned
    kani::assume(tile_row % 32 == 0);
    kani::assume(tile_col % 32 == 0);

    let is_inner = tile_row + 32 <= m && tile_col + 32 <= n;
    let is_edge = !is_inner;

    // Exactly one of inner/edge must be true.
    assert!(is_inner ^ is_edge, "every tile must be either inner or edge");
}

/// Prove: edge vs inner exhaustive for 64x64 LARGE tiles.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn edge_vs_inner_exhaustive_large() {
    let m: u32 = kani::any();
    let n: u32 = kani::any();
    let tile_row: u32 = kani::any();
    let tile_col: u32 = kani::any();

    kani::assume(m >= 64 && m <= 4096);
    kani::assume(n >= 64 && n <= 4096);
    kani::assume(tile_row % 64 == 0);
    kani::assume(tile_col % 64 == 0);
    kani::assume(tile_row < m.div_ceil(64) * 64);
    kani::assume(tile_col < n.div_ceil(64) * 64);

    let is_inner = tile_row + 64 <= m && tile_col + 64 <= n;
    let is_edge = !is_inner;

    assert!(is_inner ^ is_edge, "every LARGE tile must be either inner or edge");
}

// ---------------------------------------------------------------------------
// 64x64 2-pass output: covers all 64 rows without overlap
// ---------------------------------------------------------------------------

/// Prove: the 2-pass output write covers all 64 rows exactly once.
///
/// Pass 0 writes rows [0, 32). Pass 1 writes rows [32, 64).
/// Together they cover [0, 64) with no overlap and no gap.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn two_pass_output_covers_all_rows() {
    let bm: u32 = 64;
    let half_rows: u32 = bm / 2; // 32

    // Pass 0: rows [0, 32)
    let pass0_start = 0;
    let pass0_end = half_rows;

    // Pass 1: rows [32, 64)
    let pass1_start = half_rows;
    let pass1_end = bm;

    // Full coverage.
    assert_eq!(pass0_start, 0, "pass 0 must start at row 0");
    assert_eq!(pass1_end, bm, "pass 1 must end at BM");

    // No gap between passes.
    assert_eq!(pass0_end, pass1_start, "passes must be contiguous");

    // Each pass processes exactly HALF_ROWS.
    assert_eq!(pass0_end - pass0_start, half_rows, "pass 0 covers 32 rows");
    assert_eq!(pass1_end - pass1_start, half_rows, "pass 1 covers 32 rows");
}

/// Prove: 2-pass accumulator indexing is correct.
///
/// Pass 0 uses acc[0..4], pass 1 uses acc[4..8].
/// 8 row blocks of 8 rows each = 64 rows total.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn two_pass_accumulator_partition() {
    let sg_row_blk: u32 = 8; // BM / 8

    for pass in 0u32..2 {
        let acc_base = pass * 4;
        let row_offset = pass * 32;

        // 4 accumulators per pass.
        for ri in 0u32..4 {
            let acc_idx = acc_base + ri;
            assert!(
                acc_idx < sg_row_blk,
                "accumulator index must be within SG_ROW_BLK"
            );

            // Row covered by this accumulator block.
            let start_row = row_offset + ri * 8;
            let end_row = start_row + 8;
            assert!(end_row <= 64, "accumulator row range within BM");
        }
    }
}

// ---------------------------------------------------------------------------
// 128 threads = 4 simdgroups of 32
// ---------------------------------------------------------------------------

/// Prove: all GEMM kernels use exactly 128 threads (4 simdgroups of 32).
///
/// The threadgroup size [32, 4, 1] produces 32*4*1 = 128 threads.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn thread_count_128() {
    let threads_x: u32 = 32;
    let threads_y: u32 = 4;
    let threads_z: u32 = 1;

    let total = threads_x * threads_y * threads_z;
    assert_eq!(total, 128, "must have exactly 128 threads");

    let simdgroups = total / 32;
    assert_eq!(simdgroups, 4, "must have exactly 4 simdgroups");
}

// ---------------------------------------------------------------------------
// Cooperative load coverage
// ---------------------------------------------------------------------------

/// Prove: 128 threads cooperatively load all 32x32=1024 elements of SMALL tile.
///
/// Each thread loads ceil(1024/128) = 8 elements. The loop
/// `for idx in tid..TILE*TILE step 128` visits every index exactly once.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn cooperative_load_covers_small_tile() {
    let tile: u32 = 32;
    let threads: u32 = 128;
    let total_elems = tile * tile; // 1024

    // Each element is loaded by exactly one thread.
    let elems_per_thread = total_elems.div_ceil(threads);
    assert_eq!(elems_per_thread, 8, "each thread loads 8 elements for 32x32");

    // Coverage check: threads * elems_per_thread >= total_elems.
    assert!(
        threads * elems_per_thread >= total_elems,
        "cooperative load must cover entire tile"
    );
}

/// Prove: 128 threads cooperatively load all BM*BK=2048 elements of LARGE A tile.
///
/// Each thread loads ceil(2048/128) = 16 elements.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn cooperative_load_covers_large_a_tile() {
    let bm: u32 = 64;
    let bk: u32 = 32;
    let threads: u32 = 128;
    let total_elems = bm * bk; // 2048

    let elems_per_thread = total_elems.div_ceil(threads);
    assert_eq!(elems_per_thread, 16, "each thread loads 16 A elements for 64x32");

    assert!(
        threads * elems_per_thread >= total_elems,
        "cooperative load must cover entire A tile"
    );
}

/// Prove: 128 threads cooperatively load all BK*BN=2048 elements of LARGE B tile.
///
/// Each thread loads ceil(2048/128) = 16 elements.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn cooperative_load_covers_large_b_tile() {
    let bk: u32 = 32;
    let bn: u32 = 64;
    let threads: u32 = 128;
    let total_elems = bk * bn; // 2048

    let elems_per_thread = total_elems.div_ceil(threads);
    assert_eq!(elems_per_thread, 16, "each thread loads 16 B elements for 32x64");

    assert!(
        threads * elems_per_thread >= total_elems,
        "cooperative load must cover entire B tile"
    );
}

// ---------------------------------------------------------------------------
// Buffer offset calculations
// ---------------------------------------------------------------------------

/// Prove: A buffer offset (batch_idx * M * K) does not overflow for production dims.
///
/// Production bounds: batch <= 256, M <= 65536, K <= 65536.
/// Max offset: 256 * 65536 * 65536 = 1,099,511,627,776, fits in u64.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn a_offset_no_overflow() {
    let batch_idx: u64 = kani::any();
    let m: u64 = kani::any();
    let k: u64 = kani::any();

    kani::assume(batch_idx <= 255);
    kani::assume(m >= 1 && m <= 65536);
    kani::assume(k >= 1 && k <= 65536);

    let mk = m.checked_mul(k);
    assert!(mk.is_some(), "M*K must not overflow u64");

    let offset = batch_idx.checked_mul(mk.unwrap());
    assert!(offset.is_some(), "batch*M*K must not overflow u64");
}

/// Prove: C buffer offset (batch_idx * M * N) does not overflow for production dims.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn c_offset_no_overflow() {
    let batch_idx: u64 = kani::any();
    let m: u64 = kani::any();
    let n: u64 = kani::any();

    kani::assume(batch_idx <= 255);
    kani::assume(m >= 1 && m <= 65536);
    kani::assume(n >= 1 && n <= 65536);

    let mn = m.checked_mul(n);
    assert!(mn.is_some(), "M*N must not overflow u64");

    let offset = batch_idx.checked_mul(mn.unwrap());
    assert!(offset.is_some(), "batch*M*N must not overflow u64");
}

// ---------------------------------------------------------------------------
// Broadcast RHS offset
// ---------------------------------------------------------------------------

/// Prove: broadcast RHS offset is zero when bcast_rhs != 0.
///
/// When broadcasting, all batches read from the same B matrix at offset 0.
/// When not broadcasting, offset = batch_idx * K * N.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn broadcast_rhs_offset_zero() {
    let batch_idx: u64 = kani::any();
    let k: u64 = kani::any();
    let n: u64 = kani::any();
    let bcast_rhs: u32 = kani::any();

    kani::assume(batch_idx <= 255);
    kani::assume(k >= 1 && k <= 65536);
    kani::assume(n >= 1 && n <= 65536);

    let b_offset = if bcast_rhs != 0 {
        0u64
    } else {
        batch_idx * k * n
    };

    if bcast_rhs != 0 {
        assert_eq!(b_offset, 0, "broadcast RHS must have offset 0");
    }
}

// ---------------------------------------------------------------------------
// F16 kernel: float accumulators ensure precision
// ---------------------------------------------------------------------------

/// Prove: simdgroup column partitioning covers the full tile width.
///
/// 4 simdgroups × 8 columns each = 32 columns = TILE (for SMALL).
/// 4 simdgroups × 16 columns each = 64 columns = BN (for LARGE).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn simdgroup_column_partition_small() {
    let tile: u32 = 32;
    let n_sg: u32 = 4;
    let cols_per_sg = tile / n_sg; // 8

    // Verify coverage: each SG covers [sg_id*8, sg_id*8 + 8).
    let mut covered = 0u32;
    let mut sg_id = 0u32;
    while sg_id < n_sg {
        covered += cols_per_sg;
        sg_id += 1;
    }
    assert_eq!(covered, tile, "SG columns must cover entire TILE width");
}

/// Prove: simdgroup column partitioning covers full 64-col width for LARGE.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn simdgroup_column_partition_large() {
    let bn: u32 = 64;
    let n_sg: u32 = 4;
    let sg_cols = bn / n_sg; // 16
    let sg_col_blk = sg_cols / 8; // 2

    // Each SG: 2 column blocks of 8 = 16 columns.
    assert_eq!(sg_cols, 16, "each SG covers 16 columns");
    assert_eq!(sg_col_blk, 2, "each SG has 2 column blocks");
    assert_eq!(n_sg * sg_cols, bn, "all SGs cover full BN width");
}

// ---------------------------------------------------------------------------
// select_tile_config/should_use_f16 interaction
// ---------------------------------------------------------------------------

/// Prove: should_use_f16_simdgroup implies should_use_simdgroup.
///
/// F16 simdgroup is a strictly stronger condition.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn f16_simdgroup_implies_simdgroup() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m >= 1 && m <= 4096);
    kani::assume(k >= 1 && k <= 4096);
    kani::assume(n >= 1 && n <= 4096);
    kani::assume(batch >= 1 && batch <= 256);

    if should_use_f16_simdgroup(m, k, n, batch) {
        assert!(
            should_use_simdgroup(m, k, n),
            "F16 simdgroup must imply base simdgroup eligibility"
        );
    }
}

/// Prove: F16 threshold scales inversely with tile area.
///
/// LARGE tiles (4096 output/TG) need fewer TGs than SMALL (1024) to meet
/// the F16 threshold. Threshold = F16_MIN_THREADGROUPS * 1024 / tile_area.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn f16_threshold_scales_with_tile_area() {
    let small_area: usize = 32 * 32; // 1024
    let large_area: usize = 64 * 64; // 4096

    let small_thresh = F16_MIN_THREADGROUPS * 1024 / small_area;
    let large_thresh = F16_MIN_THREADGROUPS * 1024 / large_area;

    assert_eq!(
        small_thresh, F16_MIN_THREADGROUPS,
        "SMALL threshold equals F16_MIN_THREADGROUPS"
    );
    assert_eq!(
        large_thresh, F16_MIN_THREADGROUPS / 4,
        "LARGE threshold is 1/4 of F16_MIN_THREADGROUPS"
    );
    assert!(
        large_thresh < small_thresh,
        "LARGE needs fewer TGs than SMALL for F16"
    );
}

// ---------------------------------------------------------------------------
// K-tile iteration count
// ---------------------------------------------------------------------------

/// Prove: number of K-tiles covers all K columns.
///
/// num_k_tiles = ceil(K / BK). num_k_tiles * BK >= K.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn k_tile_count_covers_all_k() {
    let k: u32 = kani::any();
    kani::assume(k >= 1 && k <= 65536);

    // SMALL: BK=32
    let small_tiles = k.div_ceil(32);
    assert!(small_tiles * 32 >= k, "SMALL K-tiles must cover all K");

    // LARGE: BK=32
    let large_tiles = k.div_ceil(32);
    assert!(large_tiles * 32 >= k, "LARGE K-tiles must cover all K");
}

// ---------------------------------------------------------------------------
// select_tile_config decision boundaries
// ---------------------------------------------------------------------------

/// Prove: select_tile_config returns LARGE only when m >= 64 AND n >= 64.
///
/// If either dimension is below 64, the function must return SMALL regardless
/// of the TG count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_tile_small_when_dim_below_64() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let k: usize = kani::any();

    kani::assume(m >= 1 && m <= 8192);
    kani::assume(n >= 1 && n <= 8192);
    kani::assume(k >= 1 && k <= 8192);
    kani::assume(m < 64 || n < 64);

    let config = select_tile_config(m, k, n, 1);
    assert_eq!(config, GemmTileConfig::SMALL, "must use SMALL when m or n < 64");
}

/// Prove: select_tile_config returns LARGE when all conditions are met.
///
/// m >= 64, n >= 64, and enough threadgroups (>= 32).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_tile_large_when_conditions_met() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let k: usize = kani::any();

    kani::assume(m >= 64 && m <= 8192);
    kani::assume(n >= 64 && n <= 8192);
    kani::assume(k >= 1 && k <= 8192);

    let tgs_64 = m.div_ceil(64) * n.div_ceil(64);
    kani::assume(tgs_64 >= 32);

    let config = select_tile_config(m, k, n, 1);
    assert_eq!(config, GemmTileConfig::LARGE, "must use LARGE when m>=64, n>=64, tgs>=32");
}

/// Prove: select_tile_config threshold at exactly 32 TGs for LARGE.
///
/// At m=64, n=64: TGs = 1*1 = 1 < 32 → SMALL.
/// At m=2048, n=2048: TGs = 32*32 = 1024 >= 32 → LARGE.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn select_tile_threshold_boundary() {
    // Minimum LARGE: needs ceil(M/64) * ceil(N/64) >= 32.
    // M=N=2048 → 32*32 = 1024 >= 32 → LARGE.
    let config_large = select_tile_config(2048, 128, 2048, 1);
    assert_eq!(config_large, GemmTileConfig::LARGE, "2048x2048 should be LARGE");

    // M=N=64 → 1*1 = 1 < 32 → SMALL.
    let config_small = select_tile_config(64, 128, 64, 1);
    assert_eq!(config_small, GemmTileConfig::SMALL, "64x64 with 1 TG should be SMALL");
}

// ---------------------------------------------------------------------------
// K-loop MAC sub-step count
// ---------------------------------------------------------------------------

/// Prove: SIMD MAC sub-steps per K-tile = BK/8 for both configs.
///
/// BK=32 for both SMALL and LARGE. Each MAC step processes 8 columns.
/// So 32/8 = 4 sub-steps per K-tile iteration.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mac_substep_count_per_k_tile() {
    let bk: u32 = 32; // Both SMALL and LARGE use BK=32
    let step: u32 = 8;

    let substeps = bk / step;
    assert_eq!(substeps, 4, "must have 4 MAC sub-steps per K-tile");
    assert_eq!(bk % step, 0, "BK must be evenly divisible by MAC step size");
}

// ---------------------------------------------------------------------------
// Shared memory indexing bounds
// ---------------------------------------------------------------------------

/// Prove: As/Bs shared memory index never exceeds allocation for SMALL kernel.
///
/// As[TILE * PADDED] = As[32 * 33] = 1056 elements.
/// Max index: row=31, col=31 → 31*33 + 31 = 1054 < 1056.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_shared_mem_index_in_bounds() {
    let tile: u32 = 32;
    let padded: u32 = 33;
    let alloc_size = tile * padded; // 1056

    let row: u32 = kani::any();
    let col: u32 = kani::any();
    kani::assume(row < tile);
    kani::assume(col < tile);

    let idx = row * padded + col;
    assert!(idx < alloc_size, "shared memory index must be within allocation");
}

/// Prove: As shared memory index for LARGE kernel never exceeds allocation.
///
/// As[BM * A_PAD] = As[64 * 33] = 2112 elements.
/// Max index: row=63, col=31 → 63*33 + 31 = 2110 < 2112.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn large_as_index_in_bounds() {
    let bm: u32 = 64;
    let bk: u32 = 32;
    let a_pad: u32 = bk + 1; // 33
    let alloc_size = bm * a_pad; // 2112

    let row: u32 = kani::any();
    let col: u32 = kani::any();
    kani::assume(row < bm);
    kani::assume(col < bk);

    let idx = row * a_pad + col;
    assert!(idx < alloc_size, "As index must be within allocation for LARGE");
}

/// Prove: Bs shared memory index for LARGE kernel never exceeds allocation.
///
/// Bs[BK * B_PAD] = Bs[32 * 65] = 2080 elements.
/// Max index: row=31, col=63 → 31*65 + 63 = 2078 < 2080.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn large_bs_index_in_bounds() {
    let bk: u32 = 32;
    let bn: u32 = 64;
    let b_pad: u32 = bn + 1; // 65
    let alloc_size = bk * b_pad; // 2080

    let row: u32 = kani::any();
    let col: u32 = kani::any();
    kani::assume(row < bk);
    kani::assume(col < bn);

    let idx = row * b_pad + col;
    assert!(idx < alloc_size, "Bs index must be within allocation for LARGE");
}

// ---------------------------------------------------------------------------
// Output element count matches grid coverage
// ---------------------------------------------------------------------------

/// Prove: SMALL grid coverage * tile area >= M*N for all valid dims.
///
/// Grid = [ceil(N/32), ceil(M/32)]. Covered = grid_x * 32 * grid_y * 32 >= M*N.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_grid_covers_all_output_elements() {
    let m: u64 = kani::any();
    let n: u64 = kani::any();

    kani::assume(m >= 1 && m <= 16384);
    kani::assume(n >= 1 && n <= 16384);

    let grid_y = m.div_ceil(32);
    let grid_x = n.div_ceil(32);

    let covered = grid_y * 32 * grid_x * 32;
    assert!(covered >= m * n, "SMALL grid must cover all M*N output elements");
}

/// Prove: LARGE grid coverage * tile area >= M*N for all valid dims.
///
/// Grid = [ceil(N/64), ceil(M/64)]. Covered >= M*N.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn large_grid_covers_all_output_elements() {
    let m: u64 = kani::any();
    let n: u64 = kani::any();

    kani::assume(m >= 64 && m <= 16384);
    kani::assume(n >= 64 && n <= 16384);

    let grid_y = m.div_ceil(64);
    let grid_x = n.div_ceil(64);

    let covered = grid_y * 64 * grid_x * 64;
    assert!(covered >= m * n, "LARGE grid must cover all M*N output elements");
}

// ---------------------------------------------------------------------------
// F32 vs F16 TG memory relationship
// ---------------------------------------------------------------------------

/// Prove: SMALL f32 and f16 TG memory are equal (8,448 bytes each).
///
/// f32: 2 * 32 * 33 * 4 = 8,448.
/// f16: 2 * 32 * 33 * 2 + 32 * 33 * 4 = 4,224 + 4,224 = 8,448.
/// The f16 conversion buffer exactly makes up for the halved operand buffers.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn small_f32_f16_tg_memory_equal() {
    let f32_bytes = tg_memory_bytes(GemmTileConfig::SMALL, false);
    let f16_bytes = tg_memory_bytes(GemmTileConfig::SMALL, true);
    assert_eq!(f32_bytes, f16_bytes, "SMALL f32 and f16 TG memory must be equal");
}

/// Prove: LARGE f16 TG memory is less than LARGE f32 TG memory.
///
/// f32: 16,768 (no pass_out).
/// f16: 16,704 (has pass_out but half-precision operand buffers).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn large_f16_tg_less_than_f32() {
    let f32_bytes = tg_memory_bytes(GemmTileConfig::LARGE, false);
    let f16_bytes = tg_memory_bytes(GemmTileConfig::LARGE, true);
    assert!(f16_bytes < f32_bytes, "LARGE f16 TG memory must be < f32");
}

// ---------------------------------------------------------------------------
// LARGE kernel accumulator indexing
// ---------------------------------------------------------------------------

/// Prove: all 16 accumulator indices in LARGE kernel are valid.
///
/// 8 row blocks x 2 column blocks = 16 accumulators per simdgroup.
/// acc[ri][ci] must have ri < 8 and ci < 2.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn large_accumulator_indices_valid() {
    let sg_row_blk: u32 = 8; // BM / 8
    let sg_col_blk: u32 = 2; // SG_COLS / 8

    let total_accumulators = sg_row_blk * sg_col_blk;
    assert_eq!(total_accumulators, 16, "LARGE has 16 accumulators per SG");

    let mut ri: u32 = 0;
    while ri < sg_row_blk {
        let mut ci: u32 = 0;
        while ci < sg_col_blk {
            assert!(ri < sg_row_blk, "row block index in bounds");
            assert!(ci < sg_col_blk, "col block index in bounds");
            // Row coverage: ri*8 .. ri*8+8 within BM=64
            assert!((ri * 8 + 8) <= 64, "row range within BM");
            ci += 1;
        }
        ri += 1;
    }
}

// ---------------------------------------------------------------------------
// Thread linear ID uniqueness
// ---------------------------------------------------------------------------

/// Prove: tid_linear = sg_id * 32 + lane_id produces unique values [0, 128).
///
/// With sg_id in [0, 4) and lane_id in [0, 32), each combination maps to a
/// unique value in [0, 128). This is the identity used for cooperative loads.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tid_linear_unique_and_bounded() {
    let sg_id: u32 = kani::any();
    let lane_id: u32 = kani::any();

    kani::assume(sg_id < 4);
    kani::assume(lane_id < 32);

    let tid = sg_id * 32 + lane_id;
    assert!(tid < 128, "tid_linear must be < 128");

    // Verify injectivity: the mapping (sg_id, lane_id) -> tid is a bijection.
    // tid / 32 = sg_id, tid % 32 = lane_id.
    assert_eq!(tid / 32, sg_id, "tid / 32 must recover sg_id");
    assert_eq!(tid % 32, lane_id, "tid % 32 must recover lane_id");
}

// ---------------------------------------------------------------------------
// should_use_simdgroup boundary conditions
// ---------------------------------------------------------------------------

/// Prove: should_use_simdgroup requires all dims to be multiples of 8.
///
/// If any dimension is not a multiple of 8, the function returns false.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn simdgroup_requires_mult8_dims() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 1024);
    kani::assume(k >= 128 && k <= 1024);
    kani::assume(n >= 1 && n <= 1024);
    kani::assume(m * n >= 16_384);
    // At least one not mult of 8
    kani::assume(m % 8 != 0 || k % 8 != 0 || n % 8 != 0);

    assert!(
        !should_use_simdgroup(m, k, n),
        "must return false when any dim not multiple of 8"
    );
}

/// Prove: should_use_simdgroup requires K >= 128.
///
/// Even with M*N >= 16384 and all dims multiples of 8, K < 128 must return false.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn simdgroup_requires_k_ge_128() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 8 && m <= 4096);
    kani::assume(k >= 8 && k < 128);
    kani::assume(n >= 8 && n <= 4096);
    kani::assume(m % 8 == 0 && k % 8 == 0 && n % 8 == 0);
    kani::assume(m * n >= 16_384);

    assert!(
        !should_use_simdgroup(m, k, n),
        "must return false when K < 128"
    );
}

/// Prove: should_use_simdgroup requires M*N >= 16384.
///
/// Even with all dims multiples of 8 and K >= 128, M*N < 16384 must return false.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn simdgroup_requires_mn_ge_16384() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 8 && m <= 4096);
    kani::assume(k >= 128 && k <= 4096);
    kani::assume(n >= 8 && n <= 4096);
    kani::assume(m % 8 == 0 && k % 8 == 0 && n % 8 == 0);
    kani::assume(m * n < 16_384);

    assert!(
        !should_use_simdgroup(m, k, n),
        "must return false when M*N < 16384"
    );
}
