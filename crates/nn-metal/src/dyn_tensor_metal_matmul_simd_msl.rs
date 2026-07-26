// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL kernel source strings for simdgroup_matrix GEMM.
//!
//! Extracted from `dyn_tensor_metal_matmul_simd.rs` to keep the parent file
//! under 500 lines. Contains the f32 and f16/bf16 kernel variants for both
//! 32×32 (small) and 64×64 (large) tile configurations.
//!
//! ## Tile configs (#3479)
//!
//! | Config | BM | BN | BK | Threads | Output/TG | TG memory (f32) |
//! |--------|----|----|----|---------|-----------|-----------------|
//! | Small  | 32 | 32 | 32 | 128     | 1,024     | 8,448 bytes     |
//! | Large  | 64 | 64 | 32 | 128     | 4,096     | 16,768 bytes    |
//!
//! Direct register-to-device output writes eliminate tile_out/pass_out from
//! f32 kernels. Edge tiles reuse As (32×32) or Bs (64×64) for bounds checking.
//! F16 kernels keep a float conversion buffer (can't simdgroup_store float→half).

/// MSL source for the corrected simdgroup_matrix GEMM kernel.
///
/// Buffer layout:
/// - buffer(0): A [batch * M * K] (row-major)
/// - buffer(1): B [batch * K * N] or [1 * K * N] when broadcast (row-major)
/// - buffer(2): output [batch * M * N] (row-major)
/// - buffer(3): M (uint constant)
/// - buffer(4): N (uint constant)
/// - buffer(5): K (uint constant)
/// - buffer(6): batch_count (uint constant)
/// - buffer(7): broadcast_rhs (uint constant, 0=no, 1=yes)
///
/// Grid: [ceil(N/32), ceil(M/32), batch_count] threadgroups
/// Threads per threadgroup: [32, 4, 1] (= 128 threads, 4 simdgroups)
///
/// Key difference from deleted kernel: output write phase uses a single
/// shared tile_out[32*33] buffer with cooperative 128-thread writes to global,
/// eliminating the 4-iteration per-simdgroup temp→global copy loop.
pub(super) const SIMD_GEMM_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint TILE = 32;
constant uint SIMD_SIZE = 32;
// +1 column padding to avoid shared memory bank conflicts.
constant uint PADDED = TILE + 1;

kernel void simd_gemm_f32(
    device const float* A            [[buffer(0)]],
    device const float* B            [[buffer(1)]],
    device float*       C            [[buffer(2)]],
    device const uint&  M_val        [[buffer(3)]],
    device const uint&  N_val        [[buffer(4)]],
    device const uint&  K_val        [[buffer(5)]],
    device const uint&  batch_ct     [[buffer(6)]],
    device const uint&  bcast_rhs    [[buffer(7)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]]
) {
    uint batch_idx = tgid.z;
    if (batch_idx >= batch_ct) return;

    uint M = M_val;
    uint N = N_val;
    uint K = K_val;

    uint a_offset = batch_idx * M * K;
    uint b_offset = (bcast_rhs != 0u) ? 0u : (batch_idx * K * N);
    uint c_offset = batch_idx * M * N;

    // Tile origin in global output.
    uint tile_row = tgid.y * TILE;
    uint tile_col = tgid.x * TILE;

    // Shared memory for one K-strip of A and B (32×32 each, padded).
    // No tile_out: inner tiles write directly from registers to device memory.
    // Edge tiles reuse As (dead after K-loop) for bounds-checked writes.
    threadgroup float As[TILE * PADDED];
    threadgroup float Bs[TILE * PADDED];

    // Each simdgroup accumulates a 32×8 strip of the output tile.
    // sg_id selects which 8-column strip: [0..8), [8..16), [16..24), [24..32).
    uint sg_col_start = sg_id * 8;

    // Accumulator: 4 rows × 1 column of 8×8 simdgroup matrices = 32×8 strip.
    simdgroup_matrix<float, 8, 8> acc[4];
    for (uint i = 0; i < 4; i++) {
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }

    // Thread's linear index within the threadgroup for cooperative loading.
    uint tid_linear = sg_id * SIMD_SIZE + lane_id;

    uint num_k_tiles = (K + TILE - 1) / TILE;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * TILE;

        // -- Cooperative load of A tile [TILE × TILE] into shared memory ------
        // 128 threads load 32×32 = 1024 elements → 8 elements per thread.
        for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {
            uint row = idx / TILE;
            uint col = idx % TILE;
            uint gr = tile_row + row;
            uint gc = k_start + col;
            float val = (gr < M && gc < K) ? A[a_offset + gr * K + gc] : 0.0f;
            As[row * PADDED + col] = val;
        }

        // -- Cooperative load of B tile [TILE × TILE] into shared memory ------
        for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {
            uint row = idx / TILE;
            uint col = idx % TILE;
            uint gr = k_start + row;
            uint gc = tile_col + col;
            float val = (gr < K && gc < N) ? B[b_offset + gr * N + gc] : 0.0f;
            Bs[row * PADDED + col] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- SIMD multiply-accumulate over K=32 in 8-wide steps ---------------
        for (uint kk = 0; kk < TILE; kk += 8) {
            // Load B sub-tile: B_shared[kk..kk+8, sg_col_start..sg_col_start+8]
            simdgroup_matrix<float, 8, 8> Bmat;
            simdgroup_load(Bmat, &Bs[kk * PADDED + sg_col_start], PADDED);

            // For each 8-row strip of A, multiply and accumulate.
            for (uint ri = 0; ri < 4; ri++) {
                simdgroup_matrix<float, 8, 8> Amat;
                simdgroup_load(Amat, &As[(ri * 8) * PADDED + kk], PADDED);
                simdgroup_multiply_accumulate(acc[ri], Amat, Bmat, acc[ri]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Write results to global memory ----------------------------------------
    // Direct register-to-device write for inner tiles (no TG buffer needed).
    // Edge tiles reuse As (dead after K-loop) for bounds-checked writes.
    // Eliminates tile_out from TG memory: 12,672 -> 8,448 bytes (3 TGs/core).

    if (tile_row + TILE <= M && tile_col + TILE <= N) {
        // Fast path: entire tile within bounds. simdgroup_store to device memory.
        for (uint ri = 0; ri < 4; ri++) {
            simdgroup_store(acc[ri],
                &C[c_offset + (tile_row + ri * 8) * N + tile_col + sg_col_start], N);
        }
    } else {
        // Edge tile: store to As (reused, dead after K-loop), then bounds-checked write.
        for (uint ri = 0; ri < 4; ri++) {
            simdgroup_store(acc[ri], &As[(ri * 8) * PADDED + sg_col_start], PADDED);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {
            uint r = idx / TILE;
            uint c = idx % TILE;
            uint gr = tile_row + r;
            uint gc = tile_col + c;
            if (gr < M && gc < N) {
                C[c_offset + gr * N + gc] = As[r * PADDED + c];
            }
        }
    }
}
"#;

/// MSL source for the half-precision simdgroup_matrix GEMM kernel.
///
/// Same tiling strategy as `simd_gemm_f32` but with `half*` buffer types.
/// Shared memory uses `half` (halves threadgroup memory vs f32).
/// Operand matrices are `simdgroup_matrix<half, 8, 8>` (loaded from half shared memory).
/// Accumulators are `simdgroup_matrix<float, 8, 8>` for precision.
/// `simdgroup_multiply_accumulate(float, half, half, float)` is the native mixed-precision API.
/// Results are downcast to `half` during the cooperative global write phase.
///
/// Issue: #1670 (bf16 simdgroup matmul)
pub(super) const SIMD_GEMM_F16_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint TILE = 32;
constant uint SIMD_SIZE = 32;
// +1 column padding to avoid shared memory bank conflicts.
constant uint PADDED = TILE + 1;

kernel void simd_gemm_f16(
    device const half* A            [[buffer(0)]],
    device const half* B            [[buffer(1)]],
    device half*       C            [[buffer(2)]],
    device const uint&  M_val       [[buffer(3)]],
    device const uint&  N_val       [[buffer(4)]],
    device const uint&  K_val       [[buffer(5)]],
    device const uint&  batch_ct    [[buffer(6)]],
    device const uint&  bcast_rhs   [[buffer(7)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]]
) {
    uint batch_idx = tgid.z;
    if (batch_idx >= batch_ct) return;

    uint M = M_val;
    uint N = N_val;
    uint K = K_val;

    uint a_offset = batch_idx * M * K;
    uint b_offset = (bcast_rhs != 0u) ? 0u : (batch_idx * K * N);
    uint c_offset = batch_idx * M * N;

    uint tile_row = tgid.y * TILE;
    uint tile_col = tgid.x * TILE;

    // Shared memory in half — halves threadgroup memory vs f32 kernel.
    threadgroup half As[TILE * PADDED];
    threadgroup half Bs[TILE * PADDED];

    // Output tile in float for precision during store phase.
    threadgroup float tile_out[TILE * PADDED];

    uint sg_col_start = sg_id * 8;

    // Float accumulators for precision.
    simdgroup_matrix<float, 8, 8> acc[4];
    for (uint i = 0; i < 4; i++) {
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }

    uint tid_linear = sg_id * SIMD_SIZE + lane_id;
    uint num_k_tiles = (K + TILE - 1) / TILE;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * TILE;

        // Cooperative load A tile into half shared memory.
        for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {
            uint row = idx / TILE;
            uint col = idx % TILE;
            uint gr = tile_row + row;
            uint gc = k_start + col;
            half val = (gr < M && gc < K) ? A[a_offset + gr * K + gc] : half(0.0h);
            As[row * PADDED + col] = val;
        }

        // Cooperative load B tile into half shared memory.
        for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {
            uint row = idx / TILE;
            uint col = idx % TILE;
            uint gr = k_start + row;
            uint gc = tile_col + col;
            half val = (gr < K && gc < N) ? B[b_offset + gr * N + gc] : half(0.0h);
            Bs[row * PADDED + col] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // SIMD multiply-accumulate: half operands, float accumulators.
        // simdgroup_load requires matrix element type to match source pointer type,
        // so operand matrices are half. simdgroup_multiply_accumulate(float, half, half, float)
        // is the native mixed-precision API on Apple Silicon.
        for (uint kk = 0; kk < TILE; kk += 8) {
            simdgroup_matrix<half, 8, 8> Bmat;
            simdgroup_load(Bmat, &Bs[kk * PADDED + sg_col_start], PADDED);

            for (uint ri = 0; ri < 4; ri++) {
                simdgroup_matrix<half, 8, 8> Amat;
                simdgroup_load(Amat, &As[(ri * 8) * PADDED + kk], PADDED);
                simdgroup_multiply_accumulate(acc[ri], Amat, Bmat, acc[ri]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store float accumulators to float tile_out, then downcast to half on global write.
    for (uint ri = 0; ri < 4; ri++) {
        simdgroup_store(acc[ri], &tile_out[(ri * 8) * PADDED + sg_col_start], PADDED);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative global write with float-to-half conversion.
    for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {
        uint r = idx / TILE;
        uint c = idx % TILE;
        uint gr = tile_row + r;
        uint gc = tile_col + c;
        if (gr < M && gc < N) {
            C[c_offset + gr * N + gc] = half(tile_out[r * PADDED + c]);
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// 64×64 tile kernels (#3479)
// ---------------------------------------------------------------------------

/// MSL source for the 64×64 tile simdgroup GEMM kernel (f32), BK=32.
///
/// 4x output per threadgroup vs the 32×32 kernel. Direct register-to-device
/// output writes eliminate pass_out from TG memory.
///
/// TG memory: `As[64×33] + Bs[32×65]` = 16,768 bytes (was 25,088 with pass_out).
///
/// Each of the 4 SIMD groups computes a 64×16 strip of the output tile
/// (8 row blocks × 2 column blocks of 8×8 = 16 accumulators per SG).
///
/// Grid: [ceil(N/64), ceil(M/64), batch_count] threadgroups
/// Threads per threadgroup: [32, 4, 1] (= 128 threads, 4 simdgroups)
///
/// Issue: #3479 (adaptive GEMM tile selection)
pub(super) const SIMD_GEMM_64_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint BM = 64;
constant uint BN = 64;
constant uint BK = 32;
constant uint N_SG = 4;
constant uint SIMD_SIZE = 32;
constant uint THREADS = N_SG * SIMD_SIZE;
// Padding to avoid shared memory bank conflicts.
constant uint A_PAD = BK + 1;   // 33
constant uint B_PAD = BN + 1;   // 65
// Per-SG layout: 16 columns, 8 row blocks.
constant uint SG_COLS = BN / N_SG;       // 16
constant uint SG_COL_BLK = SG_COLS / 8;  // 2
constant uint SG_ROW_BLK = BM / 8;       // 8

kernel void simd_gemm_64_f32(
    device const float* A            [[buffer(0)]],
    device const float* B            [[buffer(1)]],
    device float*       C            [[buffer(2)]],
    device const uint&  M_val        [[buffer(3)]],
    device const uint&  N_val        [[buffer(4)]],
    device const uint&  K_val        [[buffer(5)]],
    device const uint&  batch_ct     [[buffer(6)]],
    device const uint&  bcast_rhs    [[buffer(7)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]]
) {
    uint batch_idx = tgid.z;
    if (batch_idx >= batch_ct) return;

    uint M = M_val;
    uint N = N_val;
    uint K = K_val;

    uint a_offset = batch_idx * M * K;
    uint b_offset = (bcast_rhs != 0u) ? 0u : (batch_idx * K * N);
    uint c_offset = batch_idx * M * N;

    uint tile_row = tgid.y * BM;
    uint tile_col = tgid.x * BN;

    // Shared memory: A[64×32] and B[32×64] with padding.
    // No pass_out: inner tiles write directly from registers to device memory.
    // Edge tiles reuse Bs (dead after K-loop) for bounds-checked 2-pass writes.
    threadgroup float As[BM * A_PAD];            // 64×33×4 = 8,448 bytes
    threadgroup float Bs[BK * B_PAD];            // 32×65×4 = 8,320 bytes
    // Total: 16,768 bytes (was 25,088 with pass_out)

    uint sg_col_start = sg_id * SG_COLS;
    uint tid_linear = sg_id * SIMD_SIZE + lane_id;

    // 16 accumulators per SG: 8 row blocks × 2 column blocks of 8×8.
    simdgroup_matrix<float, 8, 8> acc[SG_ROW_BLK][SG_COL_BLK];
    for (uint ri = 0; ri < SG_ROW_BLK; ri++)
        for (uint ci = 0; ci < SG_COL_BLK; ci++)
            acc[ri][ci] = simdgroup_matrix<float, 8, 8>(0.0f);

    uint num_k_tiles = (K + BK - 1) / BK;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        // Cooperative load A[BM×BK] = 64×32 = 2048 elements, 16 per thread.
        for (uint idx = tid_linear; idx < BM * BK; idx += THREADS) {
            uint row = idx / BK;
            uint col = idx % BK;
            uint gr = tile_row + row;
            uint gc = k_start + col;
            float val = (gr < M && gc < K) ? A[a_offset + gr * K + gc] : 0.0f;
            As[row * A_PAD + col] = val;
        }

        // Cooperative load B[BK×BN] = 32×64 = 2048 elements, 16 per thread.
        for (uint idx = tid_linear; idx < BK * BN; idx += THREADS) {
            uint row = idx / BN;
            uint col = idx % BN;
            uint gr = k_start + row;
            uint gc = tile_col + col;
            float val = (gr < K && gc < N) ? B[b_offset + gr * N + gc] : 0.0f;
            Bs[row * B_PAD + col] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // SIMD MAC: BK/8 = 4 sub-steps of 8 (same count as 32×32 BK=32).
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<float, 8, 8> Bmat[SG_COL_BLK];
            for (uint ci = 0; ci < SG_COL_BLK; ci++) {
                simdgroup_load(Bmat[ci], &Bs[kk * B_PAD + sg_col_start + ci * 8], B_PAD);
            }

            for (uint ri = 0; ri < SG_ROW_BLK; ri++) {
                simdgroup_matrix<float, 8, 8> Amat;
                simdgroup_load(Amat, &As[(ri * 8) * A_PAD + kk], A_PAD);
                for (uint ci = 0; ci < SG_COL_BLK; ci++) {
                    simdgroup_multiply_accumulate(acc[ri][ci], Amat, Bmat[ci], acc[ri][ci]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Output phase: direct register-to-device write for inner tiles.
    // Edge tiles use 2-pass bounds-checked write through Bs (dead after K-loop).
    // Eliminates pass_out: 25,088 -> 16,768 bytes TG memory.

    if (tile_row + BM <= M && tile_col + BN <= N) {
        // Fast path: entire 64x64 tile within bounds. Direct device write.
        for (uint ri = 0; ri < SG_ROW_BLK; ri++) {
            for (uint ci = 0; ci < SG_COL_BLK; ci++) {
                simdgroup_store(acc[ri][ci],
                    &C[c_offset + (tile_row + ri * 8) * N + tile_col + sg_col_start + ci * 8],
                    N);
            }
        }
    } else {
        // Edge tile: 2-pass write through Bs (reused, dead after K-loop).
        // Bs[BK * B_PAD] = Bs[32x65] has same layout as old pass_out[32x65].
        for (uint pass = 0; pass < 2; pass++) {
            uint acc_base = pass * 4;
            uint row_offset = pass * (BM / 2);

            for (uint ri = 0; ri < 4; ri++) {
                for (uint ci = 0; ci < SG_COL_BLK; ci++) {
                    simdgroup_store(acc[acc_base + ri][ci],
                        &Bs[(ri * 8) * B_PAD + sg_col_start + ci * 8], B_PAD);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint idx = tid_linear; idx < (BM / 2) * BN; idx += THREADS) {
                uint r = idx / BN;
                uint c = idx % BN;
                uint gr = tile_row + row_offset + r;
                uint gc = tile_col + c;
                if (gr < M && gc < N) {
                    C[c_offset + gr * N + gc] = Bs[r * B_PAD + c];
                }
            }

            if (pass == 0) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
"#;

/// MSL source for the 64×64 tile simdgroup GEMM kernel (f16/bf16), BK=32.
///
/// Same tiling as `simd_gemm_64_f32` but with half buffers and float
/// accumulators for mixed precision. Keeps pass_out for float→half conversion
/// (can't simdgroup_store float to device half directly).
/// TG memory: `As[64×33]h + Bs[32×65]h + pass_out[32×65]f` = 16,704 bytes.
///
/// Issue: #3479
pub(super) const SIMD_GEMM_64_F16_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint BM = 64;
constant uint BN = 64;
constant uint BK = 32;
constant uint N_SG = 4;
constant uint SIMD_SIZE = 32;
constant uint THREADS = N_SG * SIMD_SIZE;
constant uint A_PAD = BK + 1;   // 33
constant uint B_PAD = BN + 1;   // 65
constant uint OUT_PAD = BN + 1; // 65
constant uint SG_COLS = BN / N_SG;       // 16
constant uint SG_COL_BLK = SG_COLS / 8;  // 2
constant uint SG_ROW_BLK = BM / 8;       // 8
constant uint HALF_ROWS = BM / 2;        // 32

kernel void simd_gemm_64_f16(
    device const half* A            [[buffer(0)]],
    device const half* B            [[buffer(1)]],
    device half*       C            [[buffer(2)]],
    device const uint&  M_val       [[buffer(3)]],
    device const uint&  N_val       [[buffer(4)]],
    device const uint&  K_val       [[buffer(5)]],
    device const uint&  batch_ct    [[buffer(6)]],
    device const uint&  bcast_rhs   [[buffer(7)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]]
) {
    uint batch_idx = tgid.z;
    if (batch_idx >= batch_ct) return;

    uint M = M_val;
    uint N = N_val;
    uint K = K_val;

    uint a_offset = batch_idx * M * K;
    uint b_offset = (bcast_rhs != 0u) ? 0u : (batch_idx * K * N);
    uint c_offset = batch_idx * M * N;

    uint tile_row = tgid.y * BM;
    uint tile_col = tgid.x * BN;

    // Shared memory: half A/B, float pass_out for 2-pass output.
    threadgroup half As[BM * A_PAD];             // 64×33×2 = 4,224 bytes
    threadgroup half Bs[BK * B_PAD];             // 32×65×2 = 4,160 bytes
    threadgroup float pass_out[HALF_ROWS * OUT_PAD]; // 32×65×4 = 8,320 bytes
    // Total: 16,704 bytes

    uint sg_col_start = sg_id * SG_COLS;
    uint tid_linear = sg_id * SIMD_SIZE + lane_id;

    // Float accumulators for precision.
    simdgroup_matrix<float, 8, 8> acc[SG_ROW_BLK][SG_COL_BLK];
    for (uint ri = 0; ri < SG_ROW_BLK; ri++)
        for (uint ci = 0; ci < SG_COL_BLK; ci++)
            acc[ri][ci] = simdgroup_matrix<float, 8, 8>(0.0f);

    uint num_k_tiles = (K + BK - 1) / BK;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        // Cooperative load A[BM×BK] = 64×32 into half shared memory.
        for (uint idx = tid_linear; idx < BM * BK; idx += THREADS) {
            uint row = idx / BK;
            uint col = idx % BK;
            uint gr = tile_row + row;
            uint gc = k_start + col;
            half val = (gr < M && gc < K) ? A[a_offset + gr * K + gc] : half(0.0h);
            As[row * A_PAD + col] = val;
        }

        // Cooperative load B[BK×BN] = 32×64 into half shared memory.
        for (uint idx = tid_linear; idx < BK * BN; idx += THREADS) {
            uint row = idx / BN;
            uint col = idx % BN;
            uint gr = k_start + row;
            uint gc = tile_col + col;
            half val = (gr < K && gc < N) ? B[b_offset + gr * N + gc] : half(0.0h);
            Bs[row * B_PAD + col] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Half operands, float accumulators (native mixed-precision API).
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<half, 8, 8> Bmat[SG_COL_BLK];
            for (uint ci = 0; ci < SG_COL_BLK; ci++) {
                simdgroup_load(Bmat[ci], &Bs[kk * B_PAD + sg_col_start + ci * 8], B_PAD);
            }

            for (uint ri = 0; ri < SG_ROW_BLK; ri++) {
                simdgroup_matrix<half, 8, 8> Amat;
                simdgroup_load(Amat, &As[(ri * 8) * A_PAD + kk], A_PAD);
                for (uint ci = 0; ci < SG_COL_BLK; ci++) {
                    simdgroup_multiply_accumulate(acc[ri][ci], Amat, Bmat[ci], acc[ri][ci]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Output phase: 2-pass cooperative write through pass_out[32×65].
    for (uint pass = 0; pass < 2; pass++) {
        uint acc_base = pass * 4;
        uint row_offset = pass * HALF_ROWS;

        // Store float accumulators to pass_out.
        for (uint ri = 0; ri < 4; ri++) {
            for (uint ci = 0; ci < SG_COL_BLK; ci++) {
                simdgroup_store(acc[acc_base + ri][ci],
                    &pass_out[(ri * 8) * OUT_PAD + sg_col_start + ci * 8], OUT_PAD);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative global write with float-to-half conversion.
        for (uint idx = tid_linear; idx < HALF_ROWS * BN; idx += THREADS) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint gr = tile_row + row_offset + r;
            uint gc = tile_col + c;
            if (gr < M && gc < N) {
                C[c_offset + gr * N + gc] = half(pass_out[r * OUT_PAD + c]);
            }
        }

        if (pass == 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"#;
