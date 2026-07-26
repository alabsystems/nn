// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL generation for INT8 W8A16 dequantizing GEMM.
//!
//! Generates Metal Shading Language kernels where weights are INT8 (stored as
//! `uchar`, reinterpreted as signed `char`), activations are F32, and output
//! is F32. Per-channel scale and zero_point are applied during the tile load
//! phase: each INT8 weight element is dequantized to `half` before loading
//! into threadgroup memory.
//!
//! Uses the same 32x32 simdgroup-tiled GEMM skeleton as `mixed_gemm_msl.rs`,
//! with the weight load phase modified for INT8 dequantization.
//!
//! Part of #3522 (INT8/FP8 quantized matmul Metal kernels).

/// Parameters for INT8 GEMM MSL code generation.
#[derive(Debug, Clone)]
pub(crate) struct Int8GemmInfo {
    /// Rows in the activation matrix (product of leading batch dims).
    pub m: usize,
    /// Contracted dimension (in_features).
    pub k: usize,
    /// Output features.
    pub n: usize,
    /// Whether the linear has a bias vector.
    pub has_bias: bool,
}

/// Generate MSL source for INT8 W8A16 dequantizing GEMM with compile-time dimensions.
///
/// Buffer binding layout:
/// - `buffer(0)`: A — F32 activations `[M * K]`
/// - `buffer(1)`: W — INT8 weights `[N * K]` as `uchar` (transposed: weight\[n\]\[k\])
/// - `buffer(2)`: scale — F32 per-channel scales `[N]`
/// - `buffer(3)`: zero_point — I32 per-channel zero points `[N]`
/// - `buffer(4)`: bias — F32 `[N]` (only when `has_bias`)
/// - `buffer(4 or 5)`: C — F32 output `[M * N]`
///
/// The kernel reads INT8 weights in transposed layout (`W[n][k]`) because
/// PyTorch Linear stores weights as `[out_features, in_features]`. Each
/// element is dequantized on-the-fly:
/// ```text
/// w_f32 = float(int(char(W[n * K + k])) - zp[n]) * scale[n]
/// ```
/// Then cast to `half` for simdgroup MAC operations with F32 accumulation.
pub(crate) fn generate_int8_gemm_msl(info: &Int8GemmInfo) -> String {
    let Int8GemmInfo { m, k, n, has_bias } = *info;

    let bias_param = if has_bias {
        "    device const float* bias           [[buffer(4)]],\n"
    } else {
        ""
    };
    let out_buf_idx = if has_bias { 5 } else { 4 };

    let bias_add = if has_bias {
        "            val += bias[gc];\n"
    } else {
        ""
    };

    format!(
        r#"#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint TILE = 32;
constant uint SIMD_SIZE = 32;
constant uint PADDED = TILE + 1;
constant uint M_DIM = {m}u;
constant uint K_DIM = {k}u;
constant uint N_DIM = {n}u;

kernel void int8_matmul_dequant(
    device const float*  A              [[buffer(0)]],
    device const uchar*  W              [[buffer(1)]],
    device const float*  scale          [[buffer(2)]],
    device const int*    zero_point     [[buffer(3)]],
{bias_param}    device float*        C              [[buffer({out_buf_idx})]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]]
) {{
    uint tile_row = tgid.y * TILE;
    uint tile_col = tgid.x * TILE;

    threadgroup half  As[TILE * PADDED];
    threadgroup half  Ws[TILE * PADDED];
    threadgroup float tile_out[TILE * PADDED];

    uint sg_col_start = sg_id * 8;

    simdgroup_matrix<float, 8, 8> acc[4];
    for (uint i = 0; i < 4; i++) {{
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }}

    uint tid_linear = sg_id * SIMD_SIZE + lane_id;
    uint num_k_tiles = (K_DIM + TILE - 1) / TILE;

    for (uint kt = 0; kt < num_k_tiles; kt++) {{
        uint k_start = kt * TILE;

        // Load A tile: F32 activations → half
        for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {{
            uint row = idx / TILE;
            uint col = idx % TILE;
            uint gr = tile_row + row;
            uint gc = k_start + col;
            float val = (gr < M_DIM && gc < K_DIM) ? A[gr * K_DIM + gc] : 0.0f;
            As[row * PADDED + col] = half(val);
        }}

        // Load W tile: INT8 weights → dequantize → half
        // W is transposed [N, K], so W[n][k] = W[n * K + k].
        // Dequant: w_f32 = (int(char(w_u8)) - zp[n]) * scale[n]
        for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {{
            uint row = idx / TILE;   // k-tile row
            uint col = idx % TILE;   // n-tile col
            uint gk = k_start + row;
            uint gn = tile_col + col;
            half w_half = half(0.0h);
            if (gk < K_DIM && gn < N_DIM) {{
                uchar w_raw = W[gn * K_DIM + gk];
                int w_i8 = int(as_type<char>(w_raw));
                float w_f32 = float(w_i8 - zero_point[gn]) * scale[gn];
                w_half = half(w_f32);
            }}
            Ws[row * PADDED + col] = w_half;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // simdgroup matrix multiply-accumulate
        for (uint kk = 0; kk < TILE; kk += 8) {{
            simdgroup_matrix<half, 8, 8> Bmat;
            simdgroup_load(Bmat, &Ws[kk * PADDED + sg_col_start], PADDED);
            for (uint ri = 0; ri < 4; ri++) {{
                simdgroup_matrix<half, 8, 8> Amat;
                simdgroup_load(Amat, &As[(ri * 8) * PADDED + kk], PADDED);
                simdgroup_multiply_accumulate(acc[ri], Amat, Bmat, acc[ri]);
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Store accumulated results to threadgroup memory
    for (uint ri = 0; ri < 4; ri++) {{
        simdgroup_store(acc[ri], &tile_out[(ri * 8) * PADDED + sg_col_start], PADDED);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write F32 output (with optional bias)
    for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {{
        uint r = idx / TILE;
        uint c = idx % TILE;
        uint gr = tile_row + r;
        uint gc = tile_col + c;
        if (gr < M_DIM && gc < N_DIM) {{
            float val = tile_out[r * PADDED + c];
{bias_add}            C[gr * N_DIM + gc] = val;
        }}
    }}
}}"#
    )
}

/// Number of input buffers for the INT8 GEMM kernel (A + W + scale + zero_point + optional bias).
pub(crate) fn int8_gemm_input_count(has_bias: bool) -> usize {
    if has_bias {
        5
    } else {
        4
    }
}

/// Threadgroup memory bytes for the INT8 GEMM kernel.
///
/// As (half) + Ws (half) + tile_out (float):
/// half:  32 x 33 x 2 = 2,112 bytes each x 2 (As, Ws) = 4,224
/// float: 32 x 33 x 4 = 4,224 bytes (tile_out)
/// Total: 8,448 bytes
pub(crate) fn int8_gemm_threadgroup_bytes() -> u64 {
    2 * 32 * 33 * 2 + 32 * 33 * 4
}

#[cfg(test)]
#[path = "int8_gemm_msl_tests.rs"]
mod tests;
