// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MoE (Mixture of Experts) grouped GEMM kernel emitter for HIP.
//!
//! Generates kernels for the AMD x GPU MODE competition MoE problem
//! (DeepSeek-V3 style). The key optimization is **grouped GEMM**: a single
//! kernel dispatch handles multiple expert GEMMs with variable token counts
//! per expert, indexed by prefix-sum offsets.
//!
//! # MoE Forward Pipeline
//!
//! 1. **Gate + top-k:** `scores = softmax(input @ W_g^T)`, select top-k experts
//! 2. **Permute:** Sort tokens by expert assignment, build offset table
//! 3. **Grouped GEMM:** Each expert processes its slice of permuted tokens
//!    - gate_proj: `[tokens_e, d_hidden] x [d_hidden, d_expert] → [tokens_e, d_expert]`
//!    - up_proj:   same dimensions
//!    - SwiGLU:    `silu(gate) * up`
//!    - down_proj: `[tokens_e, d_expert] x [d_expert, d_hidden] → [tokens_e, d_hidden]`
//! 4. **Un-permute:** Scatter weighted expert outputs back to original positions
//!
//! # Competition dimensions (DeepSeek-V3 subset)
//!
//! - d_hidden: 7168, d_expert: 2048
//! - Routed experts: 4–32, top-k: 4, float16 precision
//! - Benchmark: 32 experts, seq_len up to 8192
//!
//! Part of #2543 (AMD x GPU MODE competition) and #2244 (MoE GEMM).

use crate::codegen_hip::{safe_hip_uint, HIP_PRELUDE};
use crate::HipCodegenError;

/// Thread block size for MoE kernels.
const MOE_BLOCK_SIZE: usize = 256;

/// Tile dimension for grouped GEMM (aligned with rocWMMA 16×16 fragments).
const GEMM_TILE: usize = 32;

/// Padded stride for shared memory bank conflict avoidance.
const GEMM_PADDED: usize = GEMM_TILE + 1;

/// Emit a grouped GEMM kernel for MoE expert computation.
///
/// Each expert `e` processes tokens in `[offsets[e], offsets[e+1])` from the
/// permuted input buffer. All experts share the same `(in_dim, out_dim)` but
/// have variable token counts (variable M dimension).
///
/// # Kernel signature
///
/// ```c
/// __global__ void grouped_expert_gemm(
///     const half* __restrict__ input,      // [total_tokens, in_dim] permuted
///     const half* __restrict__ weights,     // [n_experts, out_dim, in_dim] packed
///     half* __restrict__ output,            // [total_tokens, out_dim]
///     const unsigned int* __restrict__ expert_offsets, // [n_experts + 1]
///     const unsigned int total
/// );
/// ```
///
/// # Grid layout
///
/// - grid.x: ceil(out_dim / TILE)  — tile columns
/// - grid.y: total tile rows across all experts (dynamically indexed)
/// - grid.z: 1
///
/// Each block checks `expert_offsets` to find which expert owns its tile row,
/// then indexes into the correct weight matrix.
///
/// # Arguments
///
/// * `name` — Kernel function name
/// * `n_experts` — Number of experts (each with its own weight matrix)
/// * `in_dim` — Input feature dimension (e.g., 7168 for gate/up, 2048 for down)
/// * `out_dim` — Output feature dimension
/// * `max_total_tokens` — Upper bound on total permuted tokens (n_tokens × k)
#[allow(clippy::too_many_arguments)]
pub fn emit_grouped_gemm_kernel(
    name: &str,
    n_experts: usize,
    in_dim: usize,
    out_dim: usize,
    max_total_tokens: usize,
) -> Result<String, HipCodegenError> {
    if !in_dim.is_multiple_of(GEMM_TILE) || !out_dim.is_multiple_of(GEMM_TILE) {
        return Err(HipCodegenError::InvalidParameter(format!(
            "MoE grouped GEMM requires in_dim and out_dim multiples of {GEMM_TILE}, \
             got in_dim={in_dim}, out_dim={out_dim}"
        )));
    }

    let n_exp_val = safe_hip_uint(n_experts)?;
    let in_val = safe_hip_uint(in_dim)?;
    let out_val = safe_hip_uint(out_dim)?;
    let out_in = safe_hip_uint(out_dim * in_dim)?;
    let _max_tok = safe_hip_uint(max_total_tokens)?;

    Ok(format!(
        r#"{HIP_PRELUDE}
extern "C" __global__ void {name}(
    const half* __restrict__ input,
    const half* __restrict__ weights,
    half* __restrict__ output,
    const unsigned int* __restrict__ expert_offsets,
    const unsigned int total
) {{
    const unsigned int TILE = {GEMM_TILE}u;
    const unsigned int PAD = {GEMM_PADDED}u;
    const unsigned int N_EXPERTS = {n_exp_val};
    const unsigned int IN_DIM = {in_val};
    const unsigned int OUT_DIM = {out_val};
    const unsigned int OUT_IN = {out_in};

    // Each block computes a TILE x TILE output tile.
    // grid.x indexes output columns, grid.y indexes rows across all experts.
    unsigned int tile_col = blockIdx.x * TILE;
    unsigned int tile_row_global = blockIdx.y * TILE;

    if (tile_col >= OUT_DIM) return;

    // Find which expert owns this tile row via linear scan on offsets.
    unsigned int expert_id = 0u;
    unsigned int expert_start = 0u;
    for (unsigned int e = 0u; e < N_EXPERTS; e++) {{
        if (tile_row_global >= expert_offsets[e] && tile_row_global < expert_offsets[e + 1u]) {{
            expert_id = e;
            expert_start = expert_offsets[e];
            break;
        }}
    }}

    unsigned int expert_end = expert_offsets[expert_id + 1u];

    // Bounds check: skip if tile row is past last expert's tokens.
    if (tile_row_global >= expert_offsets[N_EXPERTS]) return;

    // Weight matrix for this expert: weights[expert_id * OUT_DIM * IN_DIM ...]
    const half* w_ptr = weights + expert_id * OUT_IN;

    // Shared memory for cooperative tile GEMM.
    __shared__ float As[TILE * PAD];  // input tile
    __shared__ float Bs[TILE * PAD];  // weight tile (transposed load)

    // Thread-to-output mapping: 256 threads cover 32×32 = 1024 elements.
    // Each thread owns one row and 4 contiguous columns (no redundant work).
    const unsigned int THREADS_PER_ROW = {MOE_BLOCK_SIZE}u / TILE; // 256/32 = 8
    const unsigned int COLS_PER_THREAD = TILE / THREADS_PER_ROW; // 32/8 = 4

    unsigned int row_in_tile = threadIdx.x / THREADS_PER_ROW;
    unsigned int col_group = threadIdx.x % THREADS_PER_ROW;
    unsigned int col_start = col_group * COLS_PER_THREAD;

    // Accumulator: each thread accumulates only its 4 columns.
    float acc[COLS_PER_THREAD];
    for (unsigned int i = 0u; i < COLS_PER_THREAD; i++) acc[i] = 0.0f;

    unsigned int num_k_tiles = IN_DIM / TILE;

    for (unsigned int kt = 0u; kt < num_k_tiles; kt++) {{
        unsigned int k_start = kt * TILE;

        // Load input tile: input[tile_row_global + r, k_start + c]
        for (unsigned int idx = threadIdx.x; idx < TILE * TILE; idx += {MOE_BLOCK_SIZE}u) {{
            unsigned int r = idx / TILE;
            unsigned int c = idx % TILE;
            unsigned int gr = tile_row_global + r;
            unsigned int gc = k_start + c;

            float val = 0.0f;
            if (gr < expert_end && gc < IN_DIM) {{
                val = __half2float(input[gr * IN_DIM + gc]);
            }}
            As[r * PAD + c] = val;
        }}

        // Load weight tile: w[tile_col + r, k_start + c] (weights are [OUT_DIM, IN_DIM])
        // We load transposed so Bs[c][r] = w[tile_col + r][k_start + c]
        for (unsigned int idx = threadIdx.x; idx < TILE * TILE; idx += {MOE_BLOCK_SIZE}u) {{
            unsigned int r = idx / TILE;
            unsigned int c = idx % TILE;
            unsigned int gr = tile_col + r;   // output dim
            unsigned int gc = k_start + c;    // input dim

            float val = 0.0f;
            if (gr < OUT_DIM && gc < IN_DIM) {{
                val = __half2float(w_ptr[gr * IN_DIM + gc]);
            }}
            // Store transposed: Bs[c][r] so inner loop accesses Bs[k][out]
            Bs[c * PAD + r] = val;
        }}

        __syncthreads();

        // Compute partial products: each thread handles one row, 4 columns.
        if ((tile_row_global + row_in_tile) < expert_end) {{
            for (unsigned int k = 0u; k < TILE; k++) {{
                float a_val = As[row_in_tile * PAD + k];
                for (unsigned int j = 0u; j < COLS_PER_THREAD; j++) {{
                    acc[j] += a_val * Bs[k * PAD + col_start + j];
                }}
            }}
        }}

        __syncthreads();
    }}

    // Write output: each thread writes its unique 4 elements.
    unsigned int gr = tile_row_global + row_in_tile;
    if (gr < expert_end) {{
        for (unsigned int j = 0u; j < COLS_PER_THREAD; j++) {{
            unsigned int gc = tile_col + col_start + j;
            if (gc < OUT_DIM) {{
                output[gr * OUT_DIM + gc] = __float2half(acc[j]);
            }}
        }}
    }}
}}"#,
    ))
}

/// Emit a fused SwiGLU expert kernel for MoE.
///
/// Computes `output = silu(input @ W_gate^T) * (input @ W_up^T)` for each
/// expert's tokens. Gate and up projections share the same input and are
/// computed together to halve input memory reads.
///
/// # Kernel signature
///
/// ```c
/// __global__ void moe_swiglu(
///     const half* __restrict__ input,       // [total_tokens, d_hidden]
///     const half* __restrict__ gate_weights, // [n_experts, d_expert, d_hidden]
///     const half* __restrict__ up_weights,   // [n_experts, d_expert, d_hidden]
///     half* __restrict__ output,             // [total_tokens, d_expert]
///     const unsigned int* __restrict__ expert_offsets, // [n_experts + 1]
///     const unsigned int total
/// );
/// ```
///
/// # Arguments
///
/// * `name` — Kernel function name
/// * `n_experts` — Number of routed experts
/// * `d_hidden` — Input dimension (7168 for DeepSeek-V3)
/// * `d_expert` — Expert intermediate dimension (2048)
#[allow(clippy::too_many_arguments)]
pub fn emit_moe_swiglu_kernel(
    name: &str,
    n_experts: usize,
    d_hidden: usize,
    d_expert: usize,
) -> Result<String, HipCodegenError> {
    let n_exp_val = safe_hip_uint(n_experts)?;
    let d_hid_val = safe_hip_uint(d_hidden)?;
    let d_exp_val = safe_hip_uint(d_expert)?;
    let exp_hid = safe_hip_uint(d_expert * d_hidden)?;

    Ok(format!(
        r#"{HIP_PRELUDE}
extern "C" __global__ void {name}(
    const half* __restrict__ input,
    const half* __restrict__ gate_weights,
    const half* __restrict__ up_weights,
    half* __restrict__ output,
    const unsigned int* __restrict__ expert_offsets,
    const unsigned int total
) {{
    const unsigned int N_EXPERTS = {n_exp_val};
    const unsigned int D_HIDDEN = {d_hid_val};
    const unsigned int D_EXPERT = {d_exp_val};
    const unsigned int EXP_HID = {exp_hid};

    // Each block handles one (token, output_chunk) pair.
    // grid.x = total_tokens, grid.y = ceil(d_expert / blockDim.x)
    unsigned int token_idx = blockIdx.x;
    unsigned int out_base = blockIdx.y * {MOE_BLOCK_SIZE}u;
    unsigned int tid = threadIdx.x;
    unsigned int out_j = out_base + tid;

    // Find expert for this token via offset scan.
    unsigned int expert_id = 0u;
    for (unsigned int e = 0u; e < N_EXPERTS; e++) {{
        if (token_idx >= expert_offsets[e] && token_idx < expert_offsets[e + 1u]) {{
            expert_id = e;
            break;
        }}
    }}

    unsigned int expert_end = expert_offsets[expert_id + 1u];
    if (token_idx >= expert_end) return;
    if (out_j >= D_EXPERT) return;

    const half* x = input + token_idx * D_HIDDEN;
    const half* w_gate = gate_weights + expert_id * EXP_HID + out_j * D_HIDDEN;
    const half* w_up = up_weights + expert_id * EXP_HID + out_j * D_HIDDEN;

    // Compute gate_val = dot(x, w_gate[out_j]) and up_val = dot(x, w_up[out_j])
    float gate_val = 0.0f;
    float up_val = 0.0f;
    for (unsigned int k = 0u; k < D_HIDDEN; k++) {{
        float xk = __half2float(x[k]);
        gate_val += xk * __half2float(w_gate[k]);
        up_val += xk * __half2float(w_up[k]);
    }}

    // SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
    float swiglu = gate_val * sigmoid_gate * up_val;

    output[token_idx * D_EXPERT + out_j] = __float2half(swiglu);
}}"#,
    ))
}

/// Emit a token permutation kernel for MoE routing.
///
/// Copies tokens from original order to expert-contiguous layout using
/// pre-computed `source_token_ids` (host builds via counting sort).
pub fn emit_moe_permute_kernel(name: &str, d_hidden: usize) -> Result<String, HipCodegenError> {
    let d_hid_val = safe_hip_uint(d_hidden)?;

    Ok(format!(
        r#"{HIP_PRELUDE}
extern "C" __global__ void {name}(
    const half* __restrict__ input,
    half* __restrict__ permuted,
    const unsigned int* __restrict__ sorted_token_ids,
    const unsigned int* __restrict__ source_token_ids,
    const unsigned int n_total,
    const unsigned int total
) {{
    const unsigned int D_HIDDEN = {d_hid_val};

    // Each thread copies one element: (permuted_row, col).
    unsigned int gid = blockIdx.x * {MOE_BLOCK_SIZE}u + threadIdx.x;
    if (gid >= n_total * D_HIDDEN) return;

    unsigned int perm_row = gid / D_HIDDEN;
    unsigned int col = gid % D_HIDDEN;

    if (perm_row >= n_total) return;

    unsigned int src_token = source_token_ids[perm_row];
    permuted[perm_row * D_HIDDEN + col] = input[src_token * D_HIDDEN + col];
}}"#,
    ))
}

/// Emit a weighted un-permute + accumulate kernel for MoE.
///
/// Scatters expert outputs back to original token positions, weighted by
/// routing scores, and accumulates across the k selected experts. Output
/// buffer is `float*` (caller zero-initializes and converts to half after).
pub fn emit_moe_unpermute_kernel(
    name: &str,
    d_hidden: usize,
    experts_per_tok: usize,
) -> Result<String, HipCodegenError> {
    let d_hid_val = safe_hip_uint(d_hidden)?;
    let k_val = safe_hip_uint(experts_per_tok)?;

    Ok(format!(
        r#"{HIP_PRELUDE}
extern "C" __global__ void {name}(
    const half* __restrict__ expert_output,
    float* __restrict__ output,
    const float* __restrict__ routing_weights,
    const unsigned int* __restrict__ dest_token_ids,
    const unsigned int* __restrict__ topk_positions,
    const unsigned int n_total,
    const unsigned int n_tokens,
    const unsigned int total
) {{
    const unsigned int D_HIDDEN = {d_hid_val};
    const unsigned int K = {k_val};

    unsigned int gid = blockIdx.x * {MOE_BLOCK_SIZE}u + threadIdx.x;
    if (gid >= n_total * D_HIDDEN) return;

    unsigned int perm_row = gid / D_HIDDEN;
    unsigned int col = gid % D_HIDDEN;

    if (perm_row >= n_total) return;

    unsigned int dst_token = dest_token_ids[perm_row];
    unsigned int k_idx = topk_positions[perm_row];

    float weight = routing_weights[dst_token * K + k_idx];
    float val = __half2float(expert_output[perm_row * D_HIDDEN + col]) * weight;

    // Atomic add to float output — accumulates across k experts per token.
    atomicAdd(&output[dst_token * D_HIDDEN + col], val);
}}"#,
    ))
}

#[path = "codegen_hip_moe_launch.rs"]
mod launch;
pub use launch::*;

#[cfg(test)]
#[path = "codegen_hip_moe_tests.rs"]
mod tests;
