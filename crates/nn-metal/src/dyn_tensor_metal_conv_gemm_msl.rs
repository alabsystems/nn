// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL kernel source for Conv1d im2col transformation.
//!
//! The im2col kernel unfolds a 1D input tensor into a column matrix suitable
//! for matrix multiplication with the convolution weight tensor:
//!
//!   weight `[C_out, C_in*K]` × im2col `[C_in*K, L_out]` → `[C_out, L_out]`
//!
//! This converts Conv1d into a standard GEMM that uses the existing optimized
//! simdgroup_matrix kernel for ~1.3x speedup on ALU-bound Conv1d operations.
//!
//! Part of #3002 (Conv1d im2col + simdgroup GEMM).

/// F32 im2col_1d kernel for single-batch Conv1d.
///
/// Unfolds input `[C_in, L_in]` → `[C_in * K, L_out]`.
///
/// Buffer layout (KernelPipeline convention, param_count=1):
/// - buffer(0): input `[C_in, L_in]` (row-major, float)
/// - buffer(1): output `[C_in * K, L_out]` (row-major, float)
/// - buffer(2): total — total output elements (uint constant)
/// - buffer(3): C_in (uint constant)
/// - buffer(4): K_sz — kernel size (uint constant)
/// - buffer(5): L_in — input spatial length (uint constant)
/// - buffer(6): L_out — output spatial length (uint constant)
/// - buffer(7): stride (uint constant)
/// - buffer(8): padding (uint constant)
/// - buffer(9): dilation (uint constant)
///
/// Grid: Elementwise with `total = C_in * K_sz * L_out` threads.
pub(super) const IM2COL_1D_F32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void im2col_1d_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    device const uint&  total  [[buffer(2)]],
    device const uint&  C_IN   [[buffer(3)]],
    device const uint&  K_SZ   [[buffer(4)]],
    device const uint&  L_IN   [[buffer(5)]],
    device const uint&  L_OUT  [[buffer(6)]],
    device const uint&  S_VAL  [[buffer(7)]],
    device const uint&  P_VAL  [[buffer(8)]],
    device const uint&  D_VAL  [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total) return;

    // Output layout: [C_in * K_sz, L_out], row-major.
    // tid = ck * L_out + t, where ck = c * K_sz + k.
    uint ck = tid / L_OUT;
    uint t  = tid % L_OUT;
    uint c  = ck / K_SZ;
    uint k  = ck % K_SZ;

    // Source position: t_out * stride + k * dilation - padding.
    // Signed arithmetic handles the padding region.
    int pos = int(t * S_VAL) + int(k * D_VAL) - int(P_VAL);

    float val = 0.0f;
    if (pos >= 0 && uint(pos) < L_IN) {
        val = input[c * L_IN + uint(pos)];
    }

    output[tid] = val;
}
"#;

/// F16 im2col_1d kernel for mixed-precision Conv1d.
/// Used when input dtype is F16 or BF16 (both stored as half on Metal).
pub(super) const IM2COL_1D_F16_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void im2col_1d_f16(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    device const uint& total  [[buffer(2)]],
    device const uint& C_IN   [[buffer(3)]],
    device const uint& K_SZ   [[buffer(4)]],
    device const uint& L_IN   [[buffer(5)]],
    device const uint& L_OUT  [[buffer(6)]],
    device const uint& S_VAL  [[buffer(7)]],
    device const uint& P_VAL  [[buffer(8)]],
    device const uint& D_VAL  [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total) return;

    uint ck = tid / L_OUT;
    uint t  = tid % L_OUT;
    uint c  = ck / K_SZ;
    uint k  = ck % K_SZ;

    int pos = int(t * S_VAL) + int(k * D_VAL) - int(P_VAL);

    half val = half(0.0h);
    if (pos >= 0 && uint(pos) < L_IN) {
        val = input[c * L_IN + uint(pos)];
    }

    output[tid] = val;
}
"#;

// ---------------------------------------------------------------------------
// Direct sliding-window Conv1d GEMM for K=3, stride=1, dilation=1 (#4264)
// ---------------------------------------------------------------------------
//
// Instead of im2col (gather input → [C_in*3, L_out] temporary buffer → GEMM),
// this kernel reads the 3 input positions directly in the GEMM K-loop.
//
// For each output element [c_out, t]:
//   out[c_out][t] = sum_{c_in=0..C_in} (
//       weight[c_out][c_in*3+0] * input[c_in][t-1+pad] +
//       weight[c_out][c_in*3+1] * input[c_in][t+pad]   +
//       weight[c_out][c_in*3+2] * input[c_in][t+1+pad]
//   )
//
// Benefits vs im2col:
// - Eliminates C_in*3*L_out temporary buffer allocation
// - Eliminates im2col dispatch (saves 1 dispatch per Conv1d)
// - Better L1 cache utilization: reads input sequentially per channel
//
// The weight layout [C_out, C_in*3] is the same as for im2col GEMM.

/// F32 direct sliding-window Conv1d GEMM kernel for K=3, stride=1, dilation=1.
///
/// Buffer layout:
/// - buffer(0): input `[C_in, L_in]` (row-major, float)
/// - buffer(1): weight `[C_out, C_in * 3]` (row-major, float)
/// - buffer(2): output `[C_out, L_out]` (row-major, float)
/// - buffer(3): C_out (uint constant)
/// - buffer(4): C_in (uint constant)
/// - buffer(5): L_in (uint constant)
/// - buffer(6): L_out (uint constant)
/// - buffer(7): padding (uint constant)
///
/// Grid: [ceil(L_out/TN), ceil(C_out/TM), 1] threadgroups
/// Threads: [32, 4, 1] (128 threads, 4 simdgroups)
///
/// Each threadgroup computes a TM x TN tile of the output. The K-loop
/// iterates over C_in, loading 3 input values per channel and 3 weight
/// values per channel, accumulating into float registers.
///
/// Issue: #4264
pub(super) const DIRECT_CONV1D_K3_F32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TM = 32;
constant uint TN = 32;

kernel void direct_conv1d_k3_f32(
    device const float* input    [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device float*       output   [[buffer(2)]],
    device const uint&  C_OUT    [[buffer(3)]],
    device const uint&  C_IN     [[buffer(4)]],
    device const uint&  L_IN     [[buffer(5)]],
    device const uint&  L_OUT    [[buffer(6)]],
    device const uint&  PAD      [[buffer(7)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]]
) {
    // Tile origin in output space.
    uint tile_row = tgid.y * TM;  // C_out dimension
    uint tile_col = tgid.x * TN;  // L_out (time) dimension

    // Each thread handles multiple (row, col) pairs within the tile.
    // 128 threads cover TM*TN = 1024 elements → 8 elements per thread.
    uint thread_row = tid / TN;    // 0..3 (4 rows per pass)
    uint thread_col = tid % TN;    // 0..31

    // Accumulate in registers for 8 output elements (4 rows, done in 8 passes).
    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;

    uint K3 = C_IN * 3;  // weight row length

    // K-loop: iterate over input channels.
    for (uint c = 0; c < C_IN; c++) {
        // Global time positions for this thread's column.
        uint t = tile_col + thread_col;
        // The 3 input positions for kernel taps 0, 1, 2 at time t.
        int pos0 = int(t) - int(PAD);
        int pos1 = pos0 + 1;
        int pos2 = pos0 + 2;

        // Load 3 input values (with boundary check).
        float in0 = (pos0 >= 0 && uint(pos0) < L_IN) ? input[c * L_IN + uint(pos0)] : 0.0f;
        float in1 = (pos1 >= 0 && uint(pos1) < L_IN) ? input[c * L_IN + uint(pos1)] : 0.0f;
        float in2 = (pos2 >= 0 && uint(pos2) < L_IN) ? input[c * L_IN + uint(pos2)] : 0.0f;

        // Weight offset for channel c: 3 consecutive values per c_out row.
        uint w_c_offset = c * 3;

        // Accumulate across 8 output rows (TM=32, 4 per pass, 8 passes).
        for (uint p = 0; p < 8; p++) {
            uint row = tile_row + p * 4 + thread_row;
            if (row < C_OUT) {
                uint w_base = row * K3 + w_c_offset;
                float w0 = weight[w_base];
                float w1 = weight[w_base + 1];
                float w2 = weight[w_base + 2];
                acc[p] += w0 * in0 + w1 * in1 + w2 * in2;
            }
        }
    }

    // Write accumulated results to output.
    for (uint p = 0; p < 8; p++) {
        uint row = tile_row + p * 4 + thread_row;
        uint col = tile_col + thread_col;
        if (row < C_OUT && col < L_OUT) {
            output[row * L_OUT + col] = acc[p];
        }
    }
}
"#;

/// F16 direct sliding-window Conv1d GEMM kernel for K=3, stride=1, dilation=1.
/// Float accumulation with half I/O for mixed-precision.
///
/// Issue: #4264
pub(super) const DIRECT_CONV1D_K3_F16_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TM = 32;
constant uint TN = 32;

kernel void direct_conv1d_k3_f16(
    device const half*  input    [[buffer(0)]],
    device const half*  weight   [[buffer(1)]],
    device half*        output   [[buffer(2)]],
    device const uint&  C_OUT    [[buffer(3)]],
    device const uint&  C_IN     [[buffer(4)]],
    device const uint&  L_IN     [[buffer(5)]],
    device const uint&  L_OUT    [[buffer(6)]],
    device const uint&  PAD      [[buffer(7)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]]
) {
    uint tile_row = tgid.y * TM;
    uint tile_col = tgid.x * TN;

    uint thread_row = tid / TN;
    uint thread_col = tid % TN;

    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;

    uint K3 = C_IN * 3;

    for (uint c = 0; c < C_IN; c++) {
        uint t = tile_col + thread_col;
        int pos0 = int(t) - int(PAD);
        int pos1 = pos0 + 1;
        int pos2 = pos0 + 2;

        float in0 = (pos0 >= 0 && uint(pos0) < L_IN) ? float(input[c * L_IN + uint(pos0)]) : 0.0f;
        float in1 = (pos1 >= 0 && uint(pos1) < L_IN) ? float(input[c * L_IN + uint(pos1)]) : 0.0f;
        float in2 = (pos2 >= 0 && uint(pos2) < L_IN) ? float(input[c * L_IN + uint(pos2)]) : 0.0f;

        uint w_c_offset = c * 3;

        for (uint p = 0; p < 8; p++) {
            uint row = tile_row + p * 4 + thread_row;
            if (row < C_OUT) {
                uint w_base = row * K3 + w_c_offset;
                float w0 = float(weight[w_base]);
                float w1 = float(weight[w_base + 1]);
                float w2 = float(weight[w_base + 2]);
                acc[p] += w0 * in0 + w1 * in1 + w2 * in2;
            }
        }
    }

    for (uint p = 0; p < 8; p++) {
        uint row = tile_row + p * 4 + thread_row;
        uint col = tile_col + thread_col;
        if (row < C_OUT && col < L_OUT) {
            output[row * L_OUT + col] = half(acc[p]);
        }
    }
}
"#;
