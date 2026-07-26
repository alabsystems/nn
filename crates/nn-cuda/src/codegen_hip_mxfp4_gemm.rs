// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MXFP4 GEMM kernel emitter for HIP — AMD x GPU MODE competition qualifier.
//!
//! Generates a fused dequantize + rocWMMA tiled GEMM kernel that operates
//! directly on packed MXFP4 inputs. The dequantization happens in shared
//! memory during the cooperative tile load phase, avoiding a separate
//! dequantization pass over global memory.
//!
//! # Kernel signature
//!
//! ```c
//! __global__ void mxfp4_gemm(
//!     const unsigned char* __restrict__ A_packed,   // [M, K/2] packed E2M1
//!     const unsigned char* __restrict__ A_scales,   // [M, K/32] E8M0 scales
//!     const unsigned char* __restrict__ B_packed,   // [K, N/2] packed E2M1
//!     const unsigned char* __restrict__ B_scales,   // [K, N/32] E8M0 scales
//!     float* __restrict__ C,                        // [M, N] output
//!     const unsigned int M,
//!     const unsigned int K,
//!     const unsigned int N
//! );
//! ```
//!
//! # Layout
//!
//! - A is row-major: element A[i,j] packed at byte `A_packed[i * (K/2) + j/2]`
//! - B is row-major: element B[i,j] packed at byte `B_packed[i * (N/2) + j/2]`
//! - Scales: one E8M0 byte per 32 consecutive elements in the fast (column) dim
//! - Output C is row-major f32
//!
//! Part of #2543 (AMD x GPU MODE competition) and #2242 (MXFP4 dtype).

use crate::codegen_hip::{safe_hip_uint, HIP_PRELUDE};
use crate::codegen_hip_mxfp4::mxfp4_preamble_hip;
use crate::codegen_hip_tensor_emit_gemm::ROCWMMA_INCLUDE;
use crate::HipCodegenError;

/// Output tile size (matches standard rocWMMA GEMM tile).
const TILE: usize = 32;

/// Padded stride for shared memory bank conflict avoidance.
const PADDED: usize = TILE + 1;

/// Thread block size — 4 wavefronts of 64 (CDNA) or 8 of 32 (RDNA3).
const BLOCK_SIZE: usize = 256;

/// rocWMMA fragment dimension.
const WMMA_TILE: usize = 16;

/// MXFP4 block size (elements per shared scale).
const MX_BLOCK: usize = 32;

/// Emit an MXFP4 GEMM kernel: C[M,N] = dequant(A_mxfp4)[M,K] @ dequant(B_mxfp4)[K,N].
///
/// The kernel fuses dequantization with the tiled GEMM:
/// 1. Cooperative load of packed MXFP4 tiles into shared memory
/// 2. On-the-fly dequantize using E2M1 LUT + E8M0 scale
/// 3. rocWMMA fragment MMA from dequantized shared memory tiles
/// 4. Write f32 output to global memory
///
/// # Arguments
///
/// * `name` — Kernel function name (e.g., `"mxfp4_gemm_4096"`)
/// * `m`, `k`, `n` — Matrix dimensions (must be multiples of 32 for rocWMMA alignment)
/// * `batch_count` — Number of independent GEMMs (batch dim, grid.z)
///
/// # Errors
///
/// Returns `HipCodegenError` if dimensions are not multiples of 32 or overflow u32.
#[allow(clippy::too_many_arguments)]
pub fn emit_mxfp4_gemm_kernel(
    name: &str,
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
) -> Result<String, HipCodegenError> {
    // Validate alignment: rocWMMA needs 16-aligned, MXFP4 scale needs 32-aligned.
    if !m.is_multiple_of(TILE) || !k.is_multiple_of(TILE) || !n.is_multiple_of(TILE) {
        return Err(HipCodegenError::InvalidParameter(format!(
            "MXFP4 GEMM requires M, K, N multiples of {TILE}, got M={m}, K={k}, N={n}"
        )));
    }

    let m_val = safe_hip_uint(m)?;
    let k_val = safe_hip_uint(k)?;
    let n_val = safe_hip_uint(n)?;
    let batch_val = safe_hip_uint(batch_count)?;
    let mk_packed = safe_hip_uint(m * k / 2)?;
    let kn_packed = safe_hip_uint(k * n / 2)?;
    let mk_scales = safe_hip_uint(m * k / MX_BLOCK)?;
    let kn_scales = safe_hip_uint(k * n / MX_BLOCK)?;
    let mn = safe_hip_uint(m * n)?;
    let k_half = safe_hip_uint(k / 2)?;
    let n_half = safe_hip_uint(n / 2)?;
    let k_scale_stride = safe_hip_uint(k / MX_BLOCK)?;
    let n_scale_stride = safe_hip_uint(n / MX_BLOCK)?;

    let preamble = mxfp4_preamble_hip();

    Ok(format!(
        r#"{HIP_PRELUDE}{ROCWMMA_INCLUDE}
{preamble}
extern "C" __global__ void {name}(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ A_scales,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scales,
    float* __restrict__ C,
    const unsigned int total
) {{
    const unsigned int TILE_DIM = {TILE};
    const unsigned int PAD = {PADDED};
    const unsigned int WMMA_DIM = {WMMA_TILE};
    const unsigned int M_DIM = {m_val};
    const unsigned int K_DIM = {k_val};
    const unsigned int N_DIM = {n_val};
    const unsigned int BATCH_COUNT = {batch_val};
    const unsigned int MX_BLK = {MX_BLOCK};

    unsigned int batch_idx = blockIdx.z;
    if (batch_idx >= BATCH_COUNT) return;

    // Per-batch offsets into packed data and scales.
    unsigned int a_pack_off = batch_idx * ({mk_packed});
    unsigned int b_pack_off = batch_idx * ({kn_packed});
    unsigned int a_scale_off = batch_idx * ({mk_scales});
    unsigned int b_scale_off = batch_idx * ({kn_scales});
    unsigned int c_offset = batch_idx * ({mn});

    unsigned int tile_row = blockIdx.y * TILE_DIM;
    unsigned int tile_col = blockIdx.x * TILE_DIM;

    // Warp identification.
    unsigned int warp_id = threadIdx.x / warpSize;
    if (warp_id >= 4u) return;
    unsigned int warp_row = warp_id / 2u;
    unsigned int warp_col = warp_id % 2u;

    // Shared memory for dequantized tiles.
    __shared__ float As[TILE_DIM * PAD];
    __shared__ float Bs[TILE_DIM * PAD];
    __shared__ float tile_out[TILE_DIM * PAD];

    // Accumulator fragment.
    rocwmma::fragment<rocwmma::accumulator, WMMA_DIM, WMMA_DIM, WMMA_DIM, float> acc;
    rocwmma::fill_fragment(acc, 0.0f);

    unsigned int num_k_tiles = K_DIM / TILE_DIM;

    for (unsigned int kt = 0u; kt < num_k_tiles; kt++) {{
        unsigned int k_start = kt * TILE_DIM;

        // === Cooperative dequantize-load A tile [TILE x TILE] ===
        // A is row-major: A[row, col] at packed byte A_packed[row * K/2 + col/2]
        for (unsigned int idx = threadIdx.x; idx < TILE_DIM * TILE_DIM; idx += {BLOCK_SIZE}u) {{
            unsigned int row = idx / TILE_DIM;
            unsigned int col = idx % TILE_DIM;
            unsigned int gr = tile_row + row;  // global row in A
            unsigned int gc = k_start + col;   // global col in A (K dimension)

            if (gr < M_DIM && gc < K_DIM) {{
                unsigned int byte_idx = a_pack_off + gr * ({k_half}) + gc / 2u;
                unsigned int sub = gc & 1u;
                unsigned char packed = A_packed[byte_idx];

                // Scale: one E8M0 per 32 elements along K.
                unsigned int scale_idx = a_scale_off + gr * ({k_scale_stride}) + gc / MX_BLK;
                unsigned char scale = A_scales[scale_idx];

                As[row * PAD + col] = mxfp4_dequant(packed, sub, scale);
            }} else {{
                As[row * PAD + col] = 0.0f;
            }}
        }}

        // === Cooperative dequantize-load B tile [TILE x TILE] ===
        // B is row-major: B[row, col] at packed byte B_packed[row * N/2 + col/2]
        for (unsigned int idx = threadIdx.x; idx < TILE_DIM * TILE_DIM; idx += {BLOCK_SIZE}u) {{
            unsigned int row = idx / TILE_DIM;
            unsigned int col = idx % TILE_DIM;
            unsigned int gr = k_start + row;   // global row in B (K dimension)
            unsigned int gc = tile_col + col;   // global col in B

            if (gr < K_DIM && gc < N_DIM) {{
                unsigned int byte_idx = b_pack_off + gr * ({n_half}) + gc / 2u;
                unsigned int sub = gc & 1u;
                unsigned char packed = B_packed[byte_idx];

                unsigned int scale_idx = b_scale_off + gr * ({n_scale_stride}) + gc / MX_BLK;
                unsigned char scale = B_scales[scale_idx];

                Bs[row * PAD + col] = mxfp4_dequant(packed, sub, scale);
            }} else {{
                Bs[row * PAD + col] = 0.0f;
            }}
        }}

        __syncthreads();

        // === rocWMMA MMA over the dequantized shared-memory tile ===
        for (unsigned int kk = 0u; kk < TILE_DIM; kk += WMMA_DIM) {{
            rocwmma::fragment<rocwmma::matrix_a, WMMA_DIM, WMMA_DIM, WMMA_DIM,
                              float, rocwmma::row_major> a_frag;
            rocwmma::fragment<rocwmma::matrix_b, WMMA_DIM, WMMA_DIM, WMMA_DIM,
                              float, rocwmma::row_major> b_frag;

            rocwmma::load_matrix_sync(a_frag,
                &As[(warp_row * WMMA_DIM) * PAD + kk], PAD);
            rocwmma::load_matrix_sync(b_frag,
                &Bs[kk * PAD + (warp_col * WMMA_DIM)], PAD);

            rocwmma::mma_sync(acc, a_frag, b_frag, acc);
        }}

        __syncthreads();
    }}

    // Store accumulator to shared tile_out via rocWMMA.
    rocwmma::store_matrix_sync(
        &tile_out[(warp_row * WMMA_DIM) * PAD + (warp_col * WMMA_DIM)],
        acc, PAD, rocwmma::mem_row_major);
    __syncthreads();

    // Cooperative write from shared to global memory.
    for (unsigned int idx = threadIdx.x; idx < TILE_DIM * TILE_DIM; idx += {BLOCK_SIZE}u) {{
        unsigned int r = idx / TILE_DIM;
        unsigned int c = idx % TILE_DIM;
        unsigned int gr = tile_row + r;
        unsigned int gc = tile_col + c;
        if (gr < M_DIM && gc < N_DIM) {{
            C[c_offset + gr * N_DIM + gc] = tile_out[r * PAD + c];
        }}
    }}
}}"#,
    ))
}

/// Compute the [`LaunchConfig`](crate::hip_ffi::LaunchConfig) for an MXFP4 GEMM kernel.
///
/// Same grid/block layout as standard rocWMMA GEMM:
/// - Grid: `(ceil(N/32), ceil(M/32), batch_count)`
/// - Block: `(256, 1, 1)`
#[must_use]
pub fn mxfp4_gemm_launch_config(
    m: usize,
    n: usize,
    batch_count: usize,
) -> crate::hip_ffi::LaunchConfig {
    crate::hip_ffi::LaunchConfig::for_rocwmma(m, n, batch_count)
}

/// Generate a complete, standalone MXFP4 GEMM source file for the competition.
///
/// Wraps `emit_mxfp4_gemm_kernel` with includes and preamble to produce a
/// self-contained `.hip.cpp` file ready for `hipcc --genco`.
pub fn emit_mxfp4_gemm_standalone(
    name: &str,
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
) -> Result<String, HipCodegenError> {
    emit_mxfp4_gemm_kernel(name, m, k, n, batch_count)
}

#[cfg(test)]
#[path = "codegen_hip_mxfp4_gemm_tests.rs"]
mod tests;
