// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL generation for MXFP4 dequantizing GEMM.
//!
//! Generates Metal Shading Language kernels where weights are packed MXFP4
//! (2 values per byte, E1M2 format with per-block shared exponents),
//! activations are F32, and output is F32. Each block of 32 weight elements
//! shares a single 8-bit exponent (E8M0 per OCP Microscaling spec).
//!
//! The kernel unpacks nibbles from U8 packed weights, looks up the E1M2
//! magnitude via a 16-entry LUT, applies the shared block exponent, and
//! accumulates the dequantized weight × activation product in F32.
//!
//! Part of #2242 (MXFP4 block quantization Metal kernel).

/// Parameters for MXFP4 GEMM MSL code generation.
#[derive(Debug, Clone)]
pub(crate) struct Mxfp4GemmInfo {
    /// Rows in the activation matrix (product of leading batch dims).
    pub m: usize,
    /// Contracted dimension (in_features). Must be a multiple of `block_size`.
    pub k: usize,
    /// Output features.
    pub n: usize,
    /// MXFP4 block size (elements per shared exponent). Default: 32.
    pub block_size: usize,
    /// Whether the linear has a bias vector.
    pub has_bias: bool,
}

impl Default for Mxfp4GemmInfo {
    fn default() -> Self {
        Self {
            m: 1,
            k: 32,
            n: 1,
            block_size: 32,
            has_bias: false,
        }
    }
}

/// Generate MSL source for MXFP4 dequantizing GEMM with compile-time dimensions.
///
/// Buffer binding layout:
/// - `buffer(0)`: A — F32 activations `[M * K]`
/// - `buffer(1)`: packed_weights — U8 nibble-packed MXFP4 `[N * K/2]`
///   (2 values per byte, low nibble = even index, high nibble = odd index;
///    stored in transposed layout `packed_w[n][k/2]`)
/// - `buffer(2)`: shared_exponents — U8 `[N * K/block_size]`
///   (one exponent per block of `block_size` elements along K, stored `exp[n][k/block_size]`)
/// - `buffer(3)`: bias — F32 `[N]` (only when `has_bias`)
/// - `buffer(3 or 4)`: C — F32 output `[M * N]`
///
/// Each thread computes one output element `C[row][col]` by iterating over K:
/// - Unpack the nibble for `packed_w[col][k/2]`
/// - Look up `fp4_lut[nibble]` for the E1M2 magnitude (sign-aware)
/// - Multiply by `2^(shared_exp[col][k/block_size] - 127)` (block scale)
/// - Multiply-accumulate with `A[row][k]`
pub(crate) fn generate_mxfp4_gemm_msl(info: &Mxfp4GemmInfo) -> String {
    let Mxfp4GemmInfo {
        m,
        k,
        n,
        block_size,
        has_bias,
    } = *info;

    let blocks_per_row = k / block_size;

    let bias_param = if has_bias {
        "    device const float* bias           [[buffer(3)]],\n"
    } else {
        ""
    };
    let out_buf_idx = if has_bias { 4 } else { 3 };

    let bias_add = if has_bias {
        "        acc += bias[col];\n"
    } else {
        ""
    };

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

// OCP Microscaling E1M2 lookup table (16 entries, sign-aware).
// Index = 4-bit code: bit 3 = sign, bits 2..0 = magnitude index.
// Magnitudes (E=0 subnormal): 0.0, 0.5, 1.0, 1.5
// Magnitudes (E=1 normal):    2.0, 3.0, 4.0, 6.0
constant float fp4_lut[16] = {{
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,  // positive
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f   // negative
}};

constant uint M_DIM = {m}u;
constant uint K_DIM = {k}u;
constant uint N_DIM = {n}u;
constant uint BLOCK_SIZE = {block_size}u;
constant uint BLOCKS_PER_ROW = {blocks_per_row}u;
constant uint PACKED_K = K_DIM / 2u;
constant int SHARED_EXP_BIAS = 127;

kernel void mxfp4_matmul_dequant(
    device const float* A              [[buffer(0)]],
    device const uchar* packed_w       [[buffer(1)]],
    device const uchar* shared_exp     [[buffer(2)]],
{bias_param}    device float*        C              [[buffer({out_buf_idx})]],
    uint2 gid [[thread_position_in_grid]]
) {{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M_DIM || col >= N_DIM) return;

    float acc = 0.0f;

    // Base offsets for this output column's packed weights and exponents.
    uint w_base = col * PACKED_K;     // packed_w[col][0]
    uint e_base = col * BLOCKS_PER_ROW; // shared_exp[col][0]

    for (uint blk = 0; blk < BLOCKS_PER_ROW; blk++) {{
        // Compute block scale: 2^(shared_exp - 127)
        int exp_val = int(shared_exp[e_base + blk]) - SHARED_EXP_BIAS;
        float block_scale = exp2(float(exp_val));

        uint k_start = blk * BLOCK_SIZE;

        // Process BLOCK_SIZE elements (2 per packed byte).
        for (uint j = 0; j < BLOCK_SIZE / 2u; j++) {{
            uchar packed_byte = packed_w[w_base + k_start / 2u + j];

            // Low nibble: even element within the block
            uint nibble_lo = uint(packed_byte & 0x0Fu);
            float w_lo = fp4_lut[nibble_lo] * block_scale;
            float a_lo = A[row * K_DIM + k_start + j * 2u];
            acc += w_lo * a_lo;

            // High nibble: odd element within the block
            uint nibble_hi = uint(packed_byte >> 4u);
            float w_hi = fp4_lut[nibble_hi] * block_scale;
            float a_hi = A[row * K_DIM + k_start + j * 2u + 1u];
            acc += w_hi * a_hi;
        }}
    }}

{bias_add}    C[row * N_DIM + col] = acc;
}}"#
    )
}

/// Number of input buffers for the MXFP4 GEMM kernel.
///
/// Without bias: A + packed_weights + shared_exponents = 3.
/// With bias: A + packed_weights + shared_exponents + bias = 4.
pub(crate) fn mxfp4_gemm_input_count(has_bias: bool) -> usize {
    if has_bias {
        4
    } else {
        3
    }
}

/// Threadgroup memory bytes for the MXFP4 GEMM kernel.
///
/// The naive per-element kernel does not use threadgroup memory.
/// Returns 0. A future tiled/simdgroup version would allocate tiles here.
pub(crate) fn mxfp4_gemm_threadgroup_bytes() -> u64 {
    0
}

#[cfg(test)]
#[path = "mxfp4_gemm_msl_tests.rs"]
mod tests;
