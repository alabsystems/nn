// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP C++ emission for rocWMMA tiled GEMM kernels.
//!
//! Parallel to `nn-dsl::codegen_msl_tensor_emit_simdgroup` — uses AMD's
//! rocWMMA library for wavefront-level matrix multiply-accumulate on CDNA/RDNA3.
//! 32x32 output tiles with 16x16 WMMA fragments, shared-memory tiling along K.
//!
//! Part of #2241 (HIP codegen — simdgroup_matrix equivalent).

use crate::codegen_hip::{hip_type, safe_hip_uint};
use crate::HipCodegenError;
use nn_dsl::ScalarType;

/// rocWMMA fragment dimensions (16x16x16 is the standard size for CDNA/RDNA3).
const WMMA_M: usize = 16;
const WMMA_N: usize = 16;
const WMMA_K: usize = 16;

/// Output tile size (matches MSL simdgroup TILE=32).
const TILE: usize = 32;

/// Padded stride for shared memory (TILE + 1 avoids bank conflicts).
const PADDED: usize = TILE + 1;

/// Thread block size — 4 wavefronts of 64 (CDNA) or 8 of 32 (RDNA3).
const BLOCK_SIZE: usize = 256;

/// rocWMMA include directive (added to kernel source when rocWMMA is used).
pub const ROCWMMA_INCLUDE: &str = "#include <rocwmma/rocwmma.hpp>\n";

/// Decide whether to use rocWMMA tiled GEMM vs naive per-element matmul.
///
/// Mirrors `should_use_simdgroup(m, k, n)` from the Metal backend.
/// rocWMMA requires 16-aligned dimensions; the compute threshold ensures
/// the overhead of shared memory tiling is worthwhile.
#[must_use]
pub fn should_use_rocwmma(m: usize, k: usize, n: usize) -> bool {
    m.is_multiple_of(WMMA_M)
        && k.is_multiple_of(WMMA_K)
        && n.is_multiple_of(WMMA_N)
        && m * n >= 16_384
        && k >= 128
}

/// Map `ScalarType` to rocWMMA fragment element type name.
fn rocwmma_frag_type(dtype: ScalarType) -> Result<&'static str, HipCodegenError> {
    match dtype {
        ScalarType::F32 => Ok("float"),
        ScalarType::F16 => Ok("rocwmma::float16_t"),
        ScalarType::BF16 => Ok("rocwmma::bfloat16_t"),
        _ => Err(HipCodegenError::UnsupportedIRVariant {
            variant_desc: "ScalarType (rocWMMA)",
        }),
    }
}

/// Emit a rocWMMA-tiled MatMul kernel (no bias).
pub fn emit_rocwmma_matmul_kernel(
    name: &str,
    dtype: ScalarType,
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
    transpose_right: bool,
    broadcast_right: bool,
    scale: Option<f32>,
) -> Result<String, HipCodegenError> {
    emit_rocwmma_gemm_kernel(
        name,
        dtype,
        m,
        k,
        n,
        batch_count,
        transpose_right,
        broadcast_right,
        false,
        scale,
    )
}

/// Emit a rocWMMA-tiled Linear kernel (matmul + optional bias).
pub fn emit_rocwmma_linear_kernel(
    name: &str,
    dtype: ScalarType,
    in_features: usize,
    out_features: usize,
    batch_size: usize,
    has_bias: bool,
) -> Result<String, HipCodegenError> {
    // Linear: A=[batch, in_feat], B=[out_feat, in_feat]^T => C=[batch, out_feat]
    emit_rocwmma_gemm_kernel(
        name,
        dtype,
        batch_size,
        in_features,
        out_features,
        1,
        true,
        false,
        has_bias,
        None,
    )
}

/// Core rocWMMA GEMM emitter, parameterized for both Linear and MatMul.
///
/// Generates a complete HIP kernel using `rocwmma::fragment` with 32x32 output
/// tiles composed of 2x2 = 4 WMMA 16x16 fragments. Shared memory tiling along K.
///
/// Grid: `(ceil(N/32), ceil(M/32), batch_count)`.
/// Block: `(256, 1, 1)`.
#[allow(clippy::too_many_arguments)]
fn emit_rocwmma_gemm_kernel(
    name: &str,
    dtype: ScalarType,
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
    transpose_b: bool,
    broadcast_b: bool,
    has_bias: bool,
    scale: Option<f32>,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let frag_t = rocwmma_frag_type(dtype)?;
    let needs_cast = t != "float";

    let m_val = safe_hip_uint(m)?;
    let k_val = safe_hip_uint(k)?;
    let n_val = safe_hip_uint(n)?;
    let batch_val = safe_hip_uint(batch_count)?;
    let mk = safe_hip_uint(m * k)?;
    let mn = safe_hip_uint(m * n)?;
    let kn = safe_hip_uint(k * n)?;

    let zero = if needs_cast {
        format!("({t})0")
    } else {
        "0.0f".to_string()
    };

    // Bias parameter line (inserted into kernel signature when present).
    let bias_param = if has_bias {
        format!("    const {t}* __restrict__ bias,\n")
    } else {
        String::new()
    };

    // B offset: broadcast ignores batch, else batch * K * N.
    let b_offset_expr = if broadcast_b {
        "0".to_string()
    } else {
        format!("batch_idx * ({kn})")
    };

    // B shared-memory load expression (normal or transposed).
    let b_load = if transpose_b {
        format!("(gr < {k_val}u && gc < {n_val}u) ? B[b_offset + gc * {k_val}u + gr] : {zero}")
    } else {
        format!("(gr < {k_val}u && gc < {n_val}u) ? B[b_offset + gr * {n_val}u + gc] : {zero}")
    };

    // Post-accumulation: bias add + scale multiply + cast for store.
    let bias_add = if has_bias {
        "            val += bias[gc];\n".to_string()
    } else {
        String::new()
    };
    let scale_mul = match scale {
        Some(s) => format!("            val *= {s:.8}f;\n"),
        None => String::new(),
    };
    let store_expr = if needs_cast {
        format!("({t})(val)")
    } else {
        "val".to_string()
    };

    // Shared memory load for A (always row-major).
    let a_load =
        format!("(gr < {m_val}u && gc < {k_val}u) ? A[a_offset + gr * {k_val}u + gc] : {zero}");

    // rocWMMA layout for B fragments.
    let b_frag_layout = if transpose_b {
        // After loading transposed B into shared in row-major order,
        // the fragment layout is still row_major for the shared tile.
        "rocwmma::row_major"
    } else {
        "rocwmma::row_major"
    };

    Ok(format!(
        r#"{ROCWMMA_INCLUDE}
extern "C" __global__ void {name}(
    const {t}* __restrict__ A,
    const {t}* __restrict__ B,
{bias_param}    {t}* __restrict__ C,
    const unsigned int total
) {{
    const unsigned int TILE = {TILE};
    const unsigned int PADDED = {PADDED};
    const unsigned int WMMA_TILE = {WMMA_M};
    const unsigned int M_DIM = {m_val};
    const unsigned int K_DIM = {k_val};
    const unsigned int N_DIM = {n_val};
    const unsigned int BATCH_COUNT = {batch_val};

    unsigned int batch_idx = blockIdx.z;
    if (batch_idx >= BATCH_COUNT) return;

    unsigned int a_offset = batch_idx * ({mk});
    unsigned int b_offset = {b_offset_expr};
    unsigned int c_offset = batch_idx * ({mn});

    unsigned int tile_row = blockIdx.y * TILE;
    unsigned int tile_col = blockIdx.x * TILE;

    // Warp identification (works for both 64-wide CDNA and 32-wide RDNA3).
    unsigned int warp_id = threadIdx.x / warpSize;
    if (warp_id >= 4u) return;  // Only 4 warps needed for 2x2 fragment layout.
    unsigned int warp_row = warp_id / 2u;
    unsigned int warp_col = warp_id % 2u;

    __shared__ {t} As[TILE * PADDED];
    __shared__ {t} Bs[TILE * PADDED];
    __shared__ float tile_out[TILE * PADDED];

    // Initialize accumulator fragment.
    rocwmma::fragment<rocwmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> acc;
    rocwmma::fill_fragment(acc, 0.0f);

    unsigned int num_k_tiles = (K_DIM + TILE - 1u) / TILE;

    for (unsigned int kt = 0u; kt < num_k_tiles; kt++) {{
        unsigned int k_start = kt * TILE;

        // Cooperative load A tile into shared memory.
        for (unsigned int idx = threadIdx.x; idx < TILE * TILE; idx += {BLOCK_SIZE}u) {{
            unsigned int row = idx / TILE;
            unsigned int col = idx % TILE;
            unsigned int gr = tile_row + row;
            unsigned int gc = k_start + col;
            As[row * PADDED + col] = {a_load};
        }}

        // Cooperative load B tile into shared memory.
        for (unsigned int idx = threadIdx.x; idx < TILE * TILE; idx += {BLOCK_SIZE}u) {{
            unsigned int row = idx / TILE;
            unsigned int col = idx % TILE;
            unsigned int gr = k_start + row;
            unsigned int gc = tile_col + col;
            Bs[row * PADDED + col] = {b_load};
        }}

        __syncthreads();

        // WMMA multiply-accumulate over K in WMMA_TILE-wide steps.
        for (unsigned int kk = 0u; kk < TILE; kk += WMMA_TILE) {{
            rocwmma::fragment<rocwmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE,
                              {frag_t}, rocwmma::row_major> a_frag;
            rocwmma::fragment<rocwmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE,
                              {frag_t}, {b_frag_layout}> b_frag;

            rocwmma::load_matrix_sync(a_frag,
                &As[(warp_row * WMMA_TILE) * PADDED + kk], PADDED);
            rocwmma::load_matrix_sync(b_frag,
                &Bs[kk * PADDED + (warp_col * WMMA_TILE)], PADDED);

            rocwmma::mma_sync(acc, a_frag, b_frag, acc);
        }}

        __syncthreads();
    }}

    // Store accumulator to shared tile_out via rocWMMA store.
    rocwmma::store_matrix_sync(
        &tile_out[(warp_row * WMMA_TILE) * PADDED + (warp_col * WMMA_TILE)],
        acc, PADDED, rocwmma::mem_row_major);
    __syncthreads();

    // Cooperative write from shared tile_out to global memory with bounds check.
    for (unsigned int idx = threadIdx.x; idx < TILE * TILE; idx += {BLOCK_SIZE}u) {{
        unsigned int r = idx / TILE;
        unsigned int c = idx % TILE;
        unsigned int gr = tile_row + r;
        unsigned int gc = tile_col + c;
        if (gr < M_DIM && gc < N_DIM) {{
            float val = tile_out[r * PADDED + c];
{bias_add}{scale_mul}            C[c_offset + gr * N_DIM + gc] = {store_expr};
        }}
    }}
}}"#,
    ))
}

#[cfg(test)]
#[path = "codegen_hip_tensor_tests_gemm.rs"]
mod tests;
