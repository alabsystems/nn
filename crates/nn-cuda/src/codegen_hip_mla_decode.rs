// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MLA (Multi-head Latent Attention) decode kernel emitter for HIP.
//!
//! Generates a fused MLA decode kernel for single-token autoregressive decoding
//! with compressed KV cache. Uses the "absorbed key" trick from DeepSeek-V2:
//! instead of expanding latents to full K,V, absorb the up-projection into Q.
//!
//! # Algorithm (absorbed key)
//!
//! For each head `h`:
//! 1. `q_absorbed = Q[h] @ W_uk[h]` — absorb K projection into Q (`[head_dim] x [head_dim, d_c] → [d_c]`)
//! 2. For each cached position `s`: `score[s] = q_absorbed · c_kv[s]` — d_c dot product
//! 3. Online softmax over scores (numerically stable, no second pass)
//! 4. `v_weighted = Σ_s softmax[s] * c_kv[s]` — weighted sum in d_c space
//! 5. `out[h] = W_uv[h] @ v_weighted` — expand to head_dim (`[head_dim, d_c] x [d_c] → [head_dim]`)
//!
//! # Advantages over naive expand-then-attend
//!
//! - Never materializes full K,V matrices in global memory
//! - Main loop (steps 2-4) works in d_c space, not head_dim space
//! - Memory reads from KV cache: O(S_kv × d_c) instead of O(S_kv × 2 × head_dim × n_kv_heads)
//! - For DeepSeek-V3: d_c=512 vs n_heads*head_dim=4096 → 8× less memory bandwidth
//!
//! Part of #2543 (AMD x GPU MODE competition) and #2243 (MLA attention).

use crate::codegen_hip::{safe_hip_uint, HIP_PRELUDE};
use crate::HipCodegenError;

/// Thread block size for the MLA decode kernel.
///
/// Each block handles one (batch, head) pair. 256 threads cooperate on the
/// sequential scan over KV cache positions.
const MLA_BLOCK_SIZE: usize = 256;

/// Emit an MLA decode kernel with absorbed K projection.
///
/// # Kernel signature
///
/// ```c
/// __global__ void mla_decode(
///     const float* __restrict__ Q,      // [B, n_heads, head_dim]
///     const float* __restrict__ C_kv,   // [B, S_kv, d_c]
///     const float* __restrict__ W_uk,   // [n_heads, head_dim, d_c]
///     const float* __restrict__ W_uv,   // [n_heads, head_dim, d_c]
///     float* __restrict__ O,            // [B, n_heads, head_dim]
///     const unsigned int total
/// );
/// ```
///
/// # Arguments
///
/// * `name` — Kernel function name
/// * `n_heads` — Number of attention heads
/// * `head_dim` — Dimension per head (Q/K/V)
/// * `d_c` — Compressed latent dimension (KV cache size per position)
/// * `s_kv` — KV cache sequence length (number of cached positions)
/// * `batch_size` — Batch size
/// * `scale` — Attention scale factor (typically `1/sqrt(head_dim)`)
///
/// # Errors
///
/// Returns error if parameters overflow u32 or d_c exceeds shared memory limits.
#[allow(clippy::too_many_arguments)]
pub fn emit_mla_decode_kernel(
    name: &str,
    n_heads: usize,
    head_dim: usize,
    d_c: usize,
    s_kv: usize,
    batch_size: usize,
    scale: f32,
) -> Result<String, HipCodegenError> {
    // Validate d_c fits in shared memory (two arrays of d_c floats).
    // Typical GPU shared memory: 64 KB. Two d_c arrays = 8*d_c bytes.
    // d_c=2048 → 16 KB, safe. d_c=8192 → 64 KB, borderline.
    if d_c * 8 > 65536 {
        return Err(HipCodegenError::InvalidParameter(format!(
            "MLA d_c={d_c} exceeds shared memory for q_absorbed + v_weighted (need {need} bytes, max 65536)",
            need = d_c * 8
        )));
    }

    let n_heads_val = safe_hip_uint(n_heads)?;
    let head_dim_val = safe_hip_uint(head_dim)?;
    let d_c_val = safe_hip_uint(d_c)?;
    let s_kv_val = safe_hip_uint(s_kv)?;
    let batch_val = safe_hip_uint(batch_size)?;
    let head_dim_dc = safe_hip_uint(head_dim * d_c)?;
    let n_heads_hd = safe_hip_uint(n_heads * head_dim)?;
    let s_kv_dc = safe_hip_uint(s_kv * d_c)?;

    Ok(format!(
        r#"{HIP_PRELUDE}
extern "C" __global__ void {name}(
    const float* __restrict__ Q,
    const float* __restrict__ C_kv,
    const float* __restrict__ W_uk,
    const float* __restrict__ W_uv,
    float* __restrict__ O,
    const unsigned int total
) {{
    const unsigned int N_HEADS = {n_heads_val};
    const unsigned int HEAD_DIM = {head_dim_val};
    const unsigned int D_C = {d_c_val};
    const unsigned int S_KV = {s_kv_val};
    const unsigned int BATCH_SIZE = {batch_val};
    const float SCALE = {scale:.8}f;

    // Each block handles one (batch, head) pair.
    unsigned int bh_idx = blockIdx.x;
    unsigned int batch_idx = bh_idx / N_HEADS;
    unsigned int head_idx = bh_idx % N_HEADS;
    if (batch_idx >= BATCH_SIZE) return;

    unsigned int tid = threadIdx.x;

    // Shared memory layout:
    //   q_absorbed[D_C]  — absorbed query in latent space
    //   v_weighted[D_C]  — softmax-weighted sum of cached latents
    //   softmax_meta[3]  — [running_max, running_sum, correction] for online softmax
    extern __shared__ float shmem[];
    float* q_absorbed = shmem;
    float* v_weighted = shmem + D_C;
    float* softmax_meta = shmem + 2u * D_C;

    // Pointers into global arrays for this (batch, head) pair.
    const float* q_ptr = Q + batch_idx * ({n_heads_hd}) + head_idx * HEAD_DIM;
    const float* c_kv_ptr = C_kv + batch_idx * ({s_kv_dc});
    const float* w_uk_ptr = W_uk + head_idx * ({head_dim_dc});
    const float* w_uv_ptr = W_uv + head_idx * ({head_dim_dc});
    float* o_ptr = O + batch_idx * ({n_heads_hd}) + head_idx * HEAD_DIM;

    // === Step 1: Compute q_absorbed = Q[h] @ W_uk[h] ===
    // q_absorbed[j] = sum_i Q[h][i] * W_uk[h][i][j]  for j in [0, d_c)
    // W_uk layout: [head_dim, d_c] row-major
    for (unsigned int j = tid; j < D_C; j += {MLA_BLOCK_SIZE}u) {{
        float acc = 0.0f;
        for (unsigned int i = 0u; i < HEAD_DIM; i++) {{
            acc += q_ptr[i] * w_uk_ptr[i * D_C + j];
        }}
        q_absorbed[j] = acc * SCALE;  // Pre-scale Q for attention
    }}
    __syncthreads();

    // Initialize v_weighted to zero.
    for (unsigned int j = tid; j < D_C; j += {MLA_BLOCK_SIZE}u) {{
        v_weighted[j] = 0.0f;
    }}
    if (tid == 0u) {{
        softmax_meta[0] = -HUGE_VALF;  // running_max
        softmax_meta[1] = 0.0f;        // running_sum
    }}
    __syncthreads();

    // === Steps 2-4: Online softmax scan over KV cache ===
    // Process positions in chunks of blockDim.x for cooperative reduction.
    for (unsigned int s = 0u; s < S_KV; s++) {{
        const float* c_s = c_kv_ptr + s * D_C;

        // Step 2: score = q_absorbed · c_kv[s] (cooperative dot product)
        __shared__ float partial_scores[{MLA_BLOCK_SIZE}];
        float local_sum = 0.0f;
        for (unsigned int j = tid; j < D_C; j += {MLA_BLOCK_SIZE}u) {{
            local_sum += q_absorbed[j] * c_s[j];
        }}
        partial_scores[tid] = local_sum;
        __syncthreads();

        // Tree reduction for dot product.
        for (unsigned int stride = {MLA_BLOCK_SIZE}u / 2u; stride > 0u; stride >>= 1u) {{
            if (tid < stride) {{
                partial_scores[tid] += partial_scores[tid + stride];
            }}
            __syncthreads();
        }}
        float score = partial_scores[0];

        // Step 3: Online softmax update.
        // Thread 0 computes new max, correction factor, and running sum.
        // The correction factor is broadcast to all threads via shared memory.
        if (tid == 0u) {{
            float old_max = softmax_meta[0];
            float new_max = fmaxf(old_max, score);
            float correction = expf(old_max - new_max);
            softmax_meta[1] = softmax_meta[1] * correction + expf(score - new_max);
            softmax_meta[0] = new_max;
            softmax_meta[2] = correction;  // Broadcast correction to all threads.
        }}
        __syncthreads();

        float cur_max = softmax_meta[0];
        float correction = softmax_meta[2];
        float exp_score = expf(score - cur_max);

        // Step 4: Accumulate v_weighted with online softmax correction.
        // When the max changes, v_weighted must be rescaled by the correction
        // factor exp(old_max - new_max) before adding the new weighted latent.
        // Invariant: v_weighted[j] = sum over past positions of
        //   exp(score[pos] - current_max) * c_kv[pos][j]
        for (unsigned int j = tid; j < D_C; j += {MLA_BLOCK_SIZE}u) {{
            v_weighted[j] = v_weighted[j] * correction + exp_score * c_s[j];
        }}
        __syncthreads();
    }}

    // Normalize v_weighted by softmax denominator.
    float denom = softmax_meta[1];
    if (denom > 0.0f) {{
        float inv_denom = 1.0f / denom;
        for (unsigned int j = tid; j < D_C; j += {MLA_BLOCK_SIZE}u) {{
            v_weighted[j] *= inv_denom;
        }}
    }}
    __syncthreads();

    // === Step 5: Expand output: out[h] = W_uv[h] @ v_weighted ===
    // out[i] = sum_j W_uv[h][i][j] * v_weighted[j]  for i in [0, head_dim)
    for (unsigned int i = tid; i < HEAD_DIM; i += {MLA_BLOCK_SIZE}u) {{
        float acc = 0.0f;
        for (unsigned int j = 0u; j < D_C; j++) {{
            acc += w_uv_ptr[i * D_C + j] * v_weighted[j];
        }}
        o_ptr[i] = acc;
    }}
}}"#,
    ))
}

/// Compute the [`LaunchConfig`](crate::hip_ffi::LaunchConfig) for an MLA decode kernel.
///
/// Grid: `(B * n_heads, 1, 1)` — one block per (batch, head) pair.
/// Block: `(256, 1, 1)` — cooperative threads for dot product and softmax.
/// Shared memory: `2 * d_c + 3 + 256` floats (q_absorbed + v_weighted + meta + partial_scores).
#[must_use]
pub fn mla_decode_launch_config(
    batch_size: usize,
    n_heads: usize,
    d_c: usize,
) -> crate::hip_ffi::LaunchConfig {
    let grid_x = (batch_size * n_heads).min(u32::MAX as usize) as u32;
    let shared = ((2 * d_c + 3 + MLA_BLOCK_SIZE) * 4) as u32; // bytes
    crate::hip_ffi::LaunchConfig {
        grid: crate::hip_ffi::Dim3::d1(grid_x),
        block: crate::hip_ffi::Dim3::d1(MLA_BLOCK_SIZE as u32),
        shared_mem_bytes: shared,
    }
}

#[cfg(test)]
#[path = "codegen_hip_mla_decode_tests.rs"]
mod tests;
