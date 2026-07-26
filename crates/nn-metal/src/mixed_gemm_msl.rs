// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL generation for hybrid simdgroup GEMM (F32 activations × F16 weights → F32).
//!
//! Generates Metal Shading Language kernels where activations (A) are F32 and
//! weights (B) are F16. Both are loaded as `half` in threadgroup memory —
//! activations are demoted F32→F16 on load, weights are loaded directly.
//! The MAC uses `(float, half, half, float)` simdgroup operations for 2x ALU
//! throughput on Apple Silicon. Output is F32 (F32 intermediates between layers).
//!
//! Supports transpose-B (for Linear: weight stored as `[N, K]`), broadcast-B
//! (for batched matmul with shared weights), and optional bias addition.
//!
//! Part of #3085 (per-op autocast Phase 2), #2981 (F16 pipeline), #3227.

use crate::compiled_model::MixedGemmInfo;

/// Generate MSL source for mixed-precision GEMM with compile-time dimensions.
///
/// Embeds M, N, K as MSL constants (same pattern as inline simdgroup kernels).
/// The kernel name is `simd_gemm_mixed` — `KernelPipeline::from_msl` caches by
/// (MSL source hash, kernel name), so different dimensions get different pipelines.
///
/// Buffer binding layout (matches `KernelPipeline::encode_into`):
/// - `buffer(0)`: A — F32 activations `[batch * M * K]`
/// - `buffer(1)`: B — F16 weights (transposed or row-major)
/// - `buffer(2)`: bias — F16 `[N]` (only when `has_bias`)
/// - `buffer(2 or 3)`: C — F32 output `[batch * M * N]`
pub(crate) fn generate_mixed_gemm_msl(info: &MixedGemmInfo) -> String {
    let MixedGemmInfo {
        m,
        k,
        n,
        batch_count,
        transpose_b,
        broadcast_b,
        has_bias,
        ref activation,
    } = *info;

    let bias_param = if has_bias {
        "    device const half*  bias         [[buffer(2)]],\n"
    } else {
        ""
    };
    let out_buf_idx = if has_bias { 3 } else { 2 };

    let b_load = if transpose_b {
        "(gr < K_DIM && gc < N_DIM) ? B[b_offset + gc * K_DIM + gr] : half(0.0h)".to_string()
    } else {
        "(gr < K_DIM && gc < N_DIM) ? B[b_offset + gr * N_DIM + gc] : half(0.0h)".to_string()
    };

    let b_offset_expr = if broadcast_b {
        "0".to_string()
    } else {
        format!("batch_idx * ({}u)", k * n)
    };

    let bias_add = if has_bias {
        "            val += float(bias[gc]);\n"
    } else {
        ""
    };

    let activation_epilogue = match activation {
        Some(act) => nn_dsl::gemm_activation_msl_var(act, "float", "val", "            "),
        None => String::new(),
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
constant uint BATCH_COUNT = {batch_count}u;

kernel void simd_gemm_mixed(
    device const float* A            [[buffer(0)]],
    device const half*  B            [[buffer(1)]],
{bias_param}    device float*       C            [[buffer({out_buf_idx})]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]]
) {{
    uint batch_idx = tgid.z;
    if (batch_idx >= BATCH_COUNT) return;

    uint a_offset = batch_idx * M_DIM * K_DIM;
    uint b_offset = {b_offset_expr};
    uint c_offset = batch_idx * M_DIM * N_DIM;

    uint tile_row = tgid.y * TILE;
    uint tile_col = tgid.x * TILE;

    threadgroup half  As[TILE * PADDED];
    threadgroup half  Bs[TILE * PADDED];
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

        for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {{
            uint row = idx / TILE;
            uint col = idx % TILE;
            uint gr = tile_row + row;
            uint gc = k_start + col;
            float val = (gr < M_DIM && gc < K_DIM) ? A[a_offset + gr * K_DIM + gc] : 0.0f;
            As[row * PADDED + col] = half(val);
        }}

        for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {{
            uint row = idx / TILE;
            uint col = idx % TILE;
            uint gr = k_start + row;
            uint gc = tile_col + col;
            half bval = {b_load};
            Bs[row * PADDED + col] = bval;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE; kk += 8) {{
            simdgroup_matrix<half, 8, 8> Bmat;
            simdgroup_load(Bmat, &Bs[kk * PADDED + sg_col_start], PADDED);
            for (uint ri = 0; ri < 4; ri++) {{
                simdgroup_matrix<half, 8, 8> Amat;
                simdgroup_load(Amat, &As[(ri * 8) * PADDED + kk], PADDED);
                simdgroup_multiply_accumulate(acc[ri], Amat, Bmat, acc[ri]);
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (uint ri = 0; ri < 4; ri++) {{
        simdgroup_store(acc[ri], &tile_out[(ri * 8) * PADDED + sg_col_start], PADDED);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid_linear; idx < TILE * TILE; idx += 128) {{
        uint r = idx / TILE;
        uint c = idx % TILE;
        uint gr = tile_row + r;
        uint gc = tile_col + c;
        if (gr < M_DIM && gc < N_DIM) {{
            float val = tile_out[r * PADDED + c];
{bias_add}{activation_epilogue}            C[c_offset + gr * N_DIM + gc] = val;
        }}
    }}
}}"#
    )
}

/// Number of input buffers for the mixed GEMM kernel (A + B + optional bias).
pub(crate) fn mixed_gemm_input_count(has_bias: bool) -> usize {
    if has_bias {
        3
    } else {
        2
    }
}

/// Threadgroup memory bytes for the hybrid GEMM kernel.
///
/// As (half) + Bs (half) + tile_out (float):
/// half:  32 × 33 × 2 = 2,112 bytes each × 2 (As, Bs) = 4,224
/// float: 32 × 33 × 4 = 4,224 bytes (tile_out)
/// Total: 8,448 bytes (43% reduction from prior 14,784)
pub(crate) fn mixed_gemm_threadgroup_bytes() -> u64 {
    2 * 32 * 33 * 2 + 32 * 33 * 4
}
