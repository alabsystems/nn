// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused multi-head attention CUDA C++ kernel generation.
//!
//! Generates a single CUDA C++ `__global__` kernel implementing fused
//! QKV projection + scaled dot-product attention + output projection.
//! This is a higher-level kernel than [`super::ptx_attention`], which
//! expects pre-projected Q/K/V. Here the kernel takes raw input `X`
//! plus weight matrices `W_Q`, `W_K`, `W_V`, `W_O` and performs:
//!
//! ```text
//! Q = X @ W_Q     (per-head projection)
//! K = X_kv @ W_K  (per-head projection)
//! V = X_kv @ W_V  (per-head projection)
//! attn = softmax(Q @ K^T / sqrt(head_dim)) @ V   (per-head SDPA)
//! output = concat(attn_heads) @ W_O               (output projection)
//! ```
//!
//! ## Thread block configuration
//!
//! Grid: `(batch_size, num_heads, seq_len)` — one block per
//! (batch, head, query_position) tuple.
//! Block: `(block_size, 1, 1)` — threads cooperate on score/softmax/value.
//!
//! ## Shared memory
//!
//! - `q_local[head_dim]` — projected query vector for this (head, q_pos)
//! - `scores[kv_seq_len]` — attention scores after softmax
//! - `reduce_buf[block_size]` — warp reduction scratch

use crate::codegen_ptx::{PtxCodegenError, DEFAULT_SM_TARGET};
use crate::cuda_ffi::CudaLaunchConfig;
use crate::ptx_emit::CUDA_PRELUDE;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// NVIDIA warp size (32 threads).
const WARP_SIZE: usize = 32;

/// Maximum block size (8 warps = 256 threads).
const MAX_BLOCK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for fused multi-head attention PTX generation.
///
/// Unlike [`super::PtxAttentionConfig`] which takes pre-projected Q/K/V,
/// this config describes a fused kernel that includes QKV projection,
/// scaled dot-product attention, and output projection.
#[derive(Debug, Clone)]
pub struct PtxMultiHeadAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head embedding dimension (`d_model / num_heads`).
    pub head_dim: usize,
    /// Query sequence length.
    pub seq_len: usize,
    /// Key/value sequence length. Equals `seq_len` for self-attention.
    pub kv_seq_len: usize,
    /// Whether to apply a causal mask (future positions set to `-inf`).
    pub causal: bool,
    /// SM target for compilation (e.g., `"sm_80"`).
    pub sm_target: String,
}

impl PtxMultiHeadAttentionConfig {
    /// Create a config with sensible defaults.
    ///
    /// `kv_seq_len` defaults to `seq_len` (self-attention).
    /// `causal` defaults to `false`. SM target defaults to `"sm_80"`.
    pub fn new(num_heads: usize, head_dim: usize, seq_len: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            seq_len,
            kv_seq_len: seq_len,
            causal: false,
            sm_target: DEFAULT_SM_TARGET.to_string(),
        }
    }

    /// Set key/value sequence length (for cross-attention).
    #[must_use]
    pub fn with_kv_seq_len(mut self, kv_seq_len: usize) -> Self {
        self.kv_seq_len = kv_seq_len;
        self
    }

    /// Set causal masking.
    #[must_use]
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set SM target (e.g., `"sm_70"`, `"sm_80"`, `"sm_90"`).
    #[must_use]
    pub fn with_sm_target(mut self, target: &str) -> Self {
        self.sm_target = target.to_string();
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PtxCodegenError> {
        if self.num_heads == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "num_heads must be > 0".into(),
            ));
        }
        if self.head_dim == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "head_dim must be > 0".into(),
            ));
        }
        if self.seq_len == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "seq_len must be > 0".into(),
            ));
        }
        if self.kv_seq_len == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "kv_seq_len must be > 0".into(),
            ));
        }
        if !self.sm_target.starts_with("sm_") {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "sm_target must start with \"sm_\", got \"{}\"",
                self.sm_target
            )));
        }
        // Parse the numeric part of the SM target.
        let sm_num_str = &self.sm_target[3..];
        if sm_num_str.parse::<u32>().is_err() {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "sm_target has invalid numeric suffix: \"{}\"",
                self.sm_target
            )));
        }
        Ok(())
    }

    /// The full model dimension (`num_heads * head_dim`).
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Attention scale factor: `1.0 / sqrt(head_dim)`.
    #[must_use]
    pub fn scale(&self) -> f32 {
        if self.head_dim == 0 {
            return 1.0;
        }
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Compute block size for the attention kernel.
    ///
    /// Round `kv_seq_len` up to next multiple of `WARP_SIZE`, cap at
    /// `MAX_BLOCK_SIZE`.
    #[must_use]
    pub fn block_size(&self) -> usize {
        compute_block_size(self.kv_seq_len)
    }

    /// Shared memory bytes needed.
    ///
    /// `head_dim` floats for projected Q + `kv_seq_len` floats for scores +
    /// `block_size` floats for reduction scratch.
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        let q_bytes = self.head_dim * 4;
        let score_bytes = self.kv_seq_len * 4;
        let reduce_bytes = self.block_size() * 4;
        q_bytes + score_bytes + reduce_bytes
    }
}

/// Compute block size: round up to warp boundary, cap at max.
fn compute_block_size(kv_seq_len: usize) -> usize {
    if kv_seq_len == 0 {
        return WARP_SIZE;
    }
    let rounded = kv_seq_len.div_ceil(WARP_SIZE) * WARP_SIZE;
    rounded.min(MAX_BLOCK_SIZE)
}

// ---------------------------------------------------------------------------
// CUDA C++ generation
// ---------------------------------------------------------------------------

/// Generate CUDA C++ source for fused multi-head attention.
///
/// Produces a `__global__` kernel implementing fused QKV projection +
/// scaled dot-product attention + output projection.
///
/// ## Tensor layouts (all f32)
///
/// - `X`: `[batch, seq_len, d_model]` — input embeddings
/// - `X_kv`: `[batch, kv_seq_len, d_model]` — KV input (same as X for
///   self-attention)
/// - `W_Q`: `[d_model, d_model]` — query projection weights
/// - `W_K`: `[d_model, d_model]` — key projection weights
/// - `W_V`: `[d_model, d_model]` — value projection weights
/// - `W_O`: `[d_model, d_model]` — output projection weights
/// - `output`: `[batch, seq_len, d_model]`
///
/// ## Parameters passed to kernel
///
/// `X`, `X_kv`, `W_Q`, `W_K`, `W_V`, `W_O`, `output`, `batch_size`
///
/// # Errors
///
/// Returns `PtxCodegenError::InvalidParameter` if config validation fails.
pub fn generate_multihead_attention_ptx(
    config: &PtxMultiHeadAttentionConfig,
) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let d_model = config.d_model();
    let seq_len = config.seq_len;
    let kv_seq_len = config.kv_seq_len;
    let scale = config.scale();
    let causal = config.causal;
    let block_size = config.block_size();

    let mut src = String::with_capacity(12288);

    // -- CUDA C++ prelude --
    src.push_str(CUDA_PRELUDE);
    src.push_str("#include <float.h>\n\n");

    // -- Comment header --
    src.push_str(&format!(
        "// Fused multi-head attention (QKV proj + SDPA + output proj)\n\
         // num_heads={num_heads}, head_dim={head_dim}, d_model={d_model}\n\
         // seq_len={seq_len}, kv_seq_len={kv_seq_len}\n\
         // causal={causal}, scale={scale}\n\
         // block_size={block_size}\n\n"
    ));

    // -- Kernel signature --
    src.push_str("__global__ void fused_multihead_attention(\n\
         \x20   const float* __restrict__ X,\n\
         \x20   const float* __restrict__ X_kv,\n\
         \x20   const float* __restrict__ W_Q,\n\
         \x20   const float* __restrict__ W_K,\n\
         \x20   const float* __restrict__ W_V,\n\
         \x20   const float* __restrict__ W_O,\n\
         \x20   float* __restrict__ output,\n\
         \x20   const unsigned int batch_size\n\
         ) {\n");

    // -- Shared memory --
    src.push_str(&format!(
        "\x20   // Shared memory: q_local[{head_dim}] + scores[{kv_seq_len}] \
         + reduce_buf[{block_size}]\n\
         \x20   __shared__ float q_local[{head_dim}];\n\
         \x20   __shared__ float scores[{kv_seq_len}];\n\
         \x20   __shared__ float reduce_buf[{block_size}];\n\n"
    ));

    // -- Thread/block index computation --
    src.push_str(
        "\x20   // Block indices: one block per (batch, head, query_pos)\n\
         \x20   const unsigned int batch_idx = blockIdx.x;\n\
         \x20   const unsigned int head_idx  = blockIdx.y;\n\
         \x20   const unsigned int q_pos     = blockIdx.z;\n\
         \x20   const unsigned int tid       = threadIdx.x;\n\n",
    );

    // -- Bounds check --
    src.push_str("\x20   if (batch_idx >= batch_size) return;\n\n");

    // =====================================================================
    // Phase 1: Compute Q projection for this (head, q_pos) into q_local
    // =====================================================================
    // Q[h, q_pos, d] = sum_{k} X[batch, q_pos, k] * W_Q[k, h * head_dim + d]
    src.push_str(&format!(
        "\x20   // ---- Phase 1: Q projection ----\n\
         \x20   const unsigned int x_offset = (batch_idx * {seq_len}u + q_pos) * {d_model}u;\n\
         \x20   const unsigned int wq_col_base = head_idx * {head_dim}u;\n\
         \x20   for (unsigned int d = tid; d < {head_dim}u; d += {block_size}u) {{\n\
         \x20       float acc = 0.0f;\n\
         \x20       for (unsigned int k = 0; k < {d_model}u; k++) {{\n\
         \x20           acc += X[x_offset + k] * W_Q[k * {d_model}u + wq_col_base + d];\n\
         \x20       }}\n\
         \x20       q_local[d] = acc;\n\
         \x20   }}\n\
         \x20   __syncthreads();\n\n"
    ));

    // =====================================================================
    // Phase 2: Score computation (Q @ K^T * scale) with K projected inline
    // =====================================================================
    // score[j] = dot(q_local, K_proj[j]) * scale
    // K_proj[j, d] = sum_{k} X_kv[batch, j, k] * W_K[k, head * head_dim + d]
    src.push_str(&format!(
        "\x20   // ---- Phase 2: score = Q @ K^T * scale (K projected inline) ----\n\
         \x20   const unsigned int wk_col_base = head_idx * {head_dim}u;\n\
         \x20   for (unsigned int j = tid; j < {kv_seq_len}u; j += {block_size}u) {{\n\
         \x20       const unsigned int xkv_offset = (batch_idx * {kv_seq_len}u + j) * {d_model}u;\n\
         \x20       float dot = 0.0f;\n\
         \x20       for (unsigned int d = 0; d < {head_dim}u; d++) {{\n\
         \x20           float k_val = 0.0f;\n\
         \x20           for (unsigned int k = 0; k < {d_model}u; k++) {{\n\
         \x20               k_val += X_kv[xkv_offset + k] * W_K[k * {d_model}u + wk_col_base + d];\n\
         \x20           }}\n\
         \x20           dot += q_local[d] * k_val;\n\
         \x20       }}\n\
         \x20       float score = dot * {scale}f;\n"
    ));

    // -- Causal mask --
    if causal {
        src.push_str(
            "\x20       // Causal mask: zero out future positions\n\
             \x20       if (j > q_pos) {\n\
             \x20           score = -FLT_MAX;\n\
             \x20       }\n",
        );
    }

    src.push_str("\x20       scores[j] = score;\n\
         \x20   }\n\
         \x20   __syncthreads();\n\n");

    // =====================================================================
    // Phase 3: Softmax over scores
    // =====================================================================
    emit_softmax_phases(&mut src, kv_seq_len, block_size);

    // =====================================================================
    // Phase 4: Value aggregation with V projected inline + output projection
    // =====================================================================
    // attn_out[d] = sum_j attn_weight[j] * V_proj[j, d]
    // V_proj[j, d] = sum_k X_kv[batch, j, k] * W_V[k, head * head_dim + d]
    // Then: output[batch, q_pos, out_d] += attn_out[d'] * W_O[head * head_dim + d', out_d]
    src.push_str(&format!(
        "\x20   // ---- Phase 4: value aggregation (V inline) + output projection ----\n\
         \x20   const unsigned int out_base = (batch_idx * {seq_len}u + q_pos) * {d_model}u;\n\
         \x20   const unsigned int wv_col_base = head_idx * {head_dim}u;\n\
         \x20   const unsigned int wo_row_base = head_idx * {head_dim}u;\n\
         \x20   for (unsigned int out_d = tid; out_d < {d_model}u; out_d += {block_size}u) {{\n\
         \x20       float out_val = 0.0f;\n\
         \x20       for (unsigned int d = 0; d < {head_dim}u; d++) {{\n\
         \x20           // Compute attn_out[d] = sum_j attn_weight[j] * V_proj[j, d]\n\
         \x20           float attn_d = 0.0f;\n\
         \x20           for (unsigned int j = 0; j < {kv_seq_len}u; j++) {{\n\
         \x20               float v_val = 0.0f;\n\
         \x20               const unsigned int xkv_off = (batch_idx * {kv_seq_len}u + j) * {d_model}u;\n\
         \x20               for (unsigned int k = 0; k < {d_model}u; k++) {{\n\
         \x20                   v_val += X_kv[xkv_off + k] * W_V[k * {d_model}u + wv_col_base + d];\n\
         \x20               }}\n\
         \x20               attn_d += scores[j] * v_val;\n\
         \x20           }}\n\
         \x20           // Output projection: accumulate attn_out[d] * W_O[wo_row_base + d, out_d]\n\
         \x20           out_val += attn_d * W_O[(wo_row_base + d) * {d_model}u + out_d];\n\
         \x20       }}\n\
         \x20       atomicAdd(&output[out_base + out_d], out_val);\n\
         \x20   }}\n"
    ));

    // -- Kernel close --
    src.push_str("}\n");

    Ok(src)
}

/// Emit the softmax phases (find max, exp+sum, normalize) into `src`.
fn emit_softmax_phases(src: &mut String, kv_seq_len: usize, block_size: usize) {
    let num_warps = block_size / WARP_SIZE;

    // Phase 3a: find max for numerical stability
    src.push_str(&format!(
        "\x20   // ---- Phase 3a: find max score ----\n\
         \x20   float local_max = -FLT_MAX;\n\
         \x20   for (unsigned int j = tid; j < {kv_seq_len}u; j += {block_size}u) {{\n\
         \x20       if (scores[j] > local_max) local_max = scores[j];\n\
         \x20   }}\n\n"
    ));

    // Warp-level max reduction
    src.push_str(
        "\x20   for (int offset = 16; offset > 0; offset >>= 1) {\n\
         \x20       float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);\n\
         \x20       if (other > local_max) local_max = other;\n\
         \x20   }\n\n",
    );

    // Cross-warp max reduction
    if num_warps <= 1 {
        src.push_str("\x20   local_max = __shfl_sync(0xFFFFFFFF, local_max, 0);\n\n");
    } else {
        src.push_str(&format!(
            "\x20   unsigned int warp_id = tid / 32;\n\
             \x20   unsigned int lane_id = tid % 32;\n\
             \x20   if (lane_id == 0) reduce_buf[warp_id] = local_max;\n\
             \x20   __syncthreads();\n\
             \x20   if (tid < {num_warps}u) {{\n\
             \x20       local_max = reduce_buf[tid];\n\
             \x20   }} else {{\n\
             \x20       local_max = -FLT_MAX;\n\
             \x20   }}\n\
             \x20   if (warp_id == 0) {{\n\
             \x20       for (int offset = 16; offset > 0; offset >>= 1) {{\n\
             \x20           float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);\n\
             \x20           if (other > local_max) local_max = other;\n\
             \x20       }}\n\
             \x20   }}\n\
             \x20   if (tid == 0) reduce_buf[0] = local_max;\n\
             \x20   __syncthreads();\n\
             \x20   local_max = reduce_buf[0];\n\n"
        ));
    }

    // Phase 3b: exp and sum
    src.push_str(&format!(
        "\x20   // ---- Phase 3b: exp(score - max) and sum ----\n\
         \x20   float local_sum = 0.0f;\n\
         \x20   for (unsigned int j = tid; j < {kv_seq_len}u; j += {block_size}u) {{\n\
         \x20       float e = expf(scores[j] - local_max);\n\
         \x20       scores[j] = e;\n\
         \x20       local_sum += e;\n\
         \x20   }}\n\n"
    ));

    // Warp-level sum reduction
    src.push_str(
        "\x20   for (int offset = 16; offset > 0; offset >>= 1) {\n\
         \x20       local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);\n\
         \x20   }\n\n",
    );

    // Cross-warp sum reduction
    if num_warps <= 1 {
        src.push_str("\x20   local_sum = __shfl_sync(0xFFFFFFFF, local_sum, 0);\n\n");
    } else {
        src.push_str(&format!(
            "\x20   if (lane_id == 0) reduce_buf[warp_id] = local_sum;\n\
             \x20   __syncthreads();\n\
             \x20   if (tid < {num_warps}u) {{\n\
             \x20       local_sum = reduce_buf[tid];\n\
             \x20   }} else {{\n\
             \x20       local_sum = 0.0f;\n\
             \x20   }}\n\
             \x20   if (warp_id == 0) {{\n\
             \x20       for (int offset = 16; offset > 0; offset >>= 1) {{\n\
             \x20           local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);\n\
             \x20       }}\n\
             \x20   }}\n\
             \x20   if (tid == 0) reduce_buf[0] = local_sum;\n\
             \x20   __syncthreads();\n\
             \x20   local_sum = reduce_buf[0];\n\n"
        ));
    }

    // Phase 3c: normalize
    src.push_str(&format!(
        "\x20   // ---- Phase 3c: normalize ----\n\
         \x20   float inv_sum = (local_sum > 0.0f) ? (1.0f / local_sum) : 0.0f;\n\
         \x20   for (unsigned int j = tid; j < {kv_seq_len}u; j += {block_size}u) {{\n\
         \x20       scores[j] *= inv_sum;\n\
         \x20   }}\n\
         \x20   __syncthreads();\n\n"
    ));
}

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation of multi-head attention.
///
/// Computes: `output = softmax(Q @ K^T / sqrt(head_dim)) @ V` where
/// Q, K, V are pre-projected per-head tensors.
///
/// ## Arguments
///
/// * `q` — `[batch, num_heads, seq_len, head_dim]` flattened row-major
/// * `k` — `[batch, num_heads, kv_seq_len, head_dim]` flattened row-major
/// * `v` — `[batch, num_heads, kv_seq_len, head_dim]` flattened row-major
/// * `batch_size` — number of batch elements
/// * `num_heads` — number of attention heads
/// * `seq_len` — query sequence length
/// * `kv_seq_len` — key/value sequence length
/// * `head_dim` — per-head embedding dimension
/// * `causal` — whether to apply causal mask
///
/// ## Returns
///
/// Output tensor `[batch, num_heads, seq_len, head_dim]` flattened row-major.
#[must_use]
pub fn attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let q_stride_b = num_heads * seq_len * head_dim;
    let q_stride_h = seq_len * head_dim;
    let k_stride_b = num_heads * kv_seq_len * head_dim;
    let k_stride_h = kv_seq_len * head_dim;
    let v_stride_b = k_stride_b;
    let v_stride_h = k_stride_h;

    let scale = if head_dim > 0 {
        1.0 / (head_dim as f32).sqrt()
    } else {
        1.0
    };

    let out_len = batch_size * num_heads * seq_len * head_dim;
    let mut output = vec![0.0f32; out_len];

    for b in 0..batch_size {
        for h in 0..num_heads {
            for i in 0..seq_len {
                // Compute scores: score[j] = dot(Q[i], K[j]) * scale
                let mut scores = vec![0.0f32; kv_seq_len];
                let q_base = b * q_stride_b + h * q_stride_h + i * head_dim;
                for j in 0..kv_seq_len {
                    let k_base = b * k_stride_b + h * k_stride_h + j * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[k_base + d];
                    }
                    scores[j] = dot * scale;
                }

                // Causal mask
                if causal {
                    for j in 0..kv_seq_len {
                        if j > i {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }
                }

                // Softmax
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    let inv = 1.0 / sum;
                    for s in &mut scores {
                        *s *= inv;
                    }
                }

                // Value aggregation
                let out_base = b * q_stride_b + h * q_stride_h + i * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..kv_seq_len {
                        let v_base = b * v_stride_b + h * v_stride_h + j * head_dim;
                        acc += scores[j] * v[v_base + d];
                    }
                    output[out_base + d] = acc;
                }
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

/// Compute the CUDA launch configuration for fused multi-head attention.
///
/// Grid: `(batch_size, num_heads, seq_len)`.
/// Block: `(block_size, 1, 1)`.
///
/// # Arguments
///
/// * `config` — Multi-head attention configuration.
/// * `batch_size` — Number of batch elements.
#[must_use]
pub fn multihead_attention_launch_config(
    config: &PtxMultiHeadAttentionConfig,
    batch_size: usize,
) -> CudaLaunchConfig {
    use crate::cuda_ffi::CudaDim3;

    let block_size = config.block_size() as u32;
    let grid = CudaDim3::new(
        batch_size.min(u32::MAX as usize) as u32,
        config.num_heads.min(u32::MAX as usize) as u32,
        config.seq_len.min(u32::MAX as usize) as u32,
    );
    let block = CudaDim3::d1(block_size);

    CudaLaunchConfig {
        grid,
        block,
        shared_mem_bytes: config.shared_memory_bytes() as u32,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_attention_multihead_tests.rs"]
mod tests;
