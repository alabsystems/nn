// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CUDA C++ kernel generation for fused scaled dot-product attention.
//!
//! Generates CUDA C++ source implementing the complete attention pipeline:
//! `Attention(Q, K, V) = softmax(Q @ K^T * scale) @ V`. Compiled to PTX
//! via `nvcc`. Supports grouped-query attention (GQA) where `num_kv_heads`
//! can be less than `num_heads`, cross-attention (where `kv_seq_len` differs
//! from `seq_len`), and optional causal masking.
//!
//! ## Algorithm
//!
//! For each (batch, head, query_position) tuple:
//! 1. **Score computation:** `score[j] = dot(Q[head, q_pos, :], K[kv_head, j, :]) * scale`
//!    for all key positions `j` in `[0, kv_seq_len)`.
//! 2. **Causal mask:** If enabled, set `score[j] = -inf` for `j > q_pos`.
//! 3. **Softmax:** Numerically stable softmax over the score vector.
//! 4. **Value aggregation:** `out[:] = sum_j(attn_weight[j] * V[kv_head, j, :])`
//!
//! ## GQA support
//!
//! Grouped-query attention maps multiple Q heads to fewer K/V heads:
//! `kv_head_idx = head_idx / (num_heads / num_kv_heads)`. When
//! `num_kv_heads == num_heads`, this degenerates to standard multi-head
//! attention. When `num_kv_heads == 1`, this is multi-query attention.
//!
//! ## Cross-attention support
//!
//! When `kv_seq_len != seq_len`, the kernel supports cross-attention where
//! the query sequence length differs from the key/value sequence length.
//! The score vector has `kv_seq_len` elements, and value aggregation sums
//! over `kv_seq_len` positions.
//!
//! ## Thread block configuration
//!
//! One block per (batch, head, query_position) tuple.
//! Block size: configurable, defaults to `min(round_up(kv_seq_len, 32), 256)`.
//! Each thread handles a strided subset of key positions for the score
//! computation and softmax, then contributes to the output via
//! shared memory reduction.
//!
//! ## Shared memory usage
//!
//! - `scores[kv_seq_len]` — attention scores after softmax (for value aggregation)
//! - `reduce_buf[block_size]` — warp-level reduction scratch for softmax max/sum
//!
//! Parallel to Metal attention in `dyn_tensor_metal_ops.rs`.

use crate::codegen_ptx::{PtxCodegenError, DEFAULT_SM_TARGET};
use crate::cuda_ffi::CudaLaunchConfig;
use crate::ptx_emit::CUDA_PRELUDE;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// NVIDIA warp size (32 threads).
const WARP_SIZE: usize = 32;

/// Maximum block size for attention (8 warps = 256 threads).
const MAX_BLOCK_SIZE: usize = 256;

/// Public attention block size constant (256 threads = 8 warps).
///
/// Matches the maximum block size used by the attention kernel.
/// Useful for external launch configuration calculations.
pub const ATTENTION_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for CUDA attention kernel generation.
#[derive(Debug, Clone)]
pub struct PtxAttentionConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Per-head embedding dimension.
    pub head_dim: usize,
    /// Query sequence length.
    pub seq_len: usize,
    /// Key/value sequence length. For self-attention this equals `seq_len`;
    /// for cross-attention it can differ.
    pub kv_seq_len: usize,
    /// Attention score scaling factor (typically `1.0 / sqrt(head_dim)`).
    pub scale: f32,
    /// Whether to apply a causal mask (future positions set to `-inf`).
    pub causal: bool,
    /// Thread block size. Defaults to `min(round_up(kv_seq_len, 32), 256)`.
    pub block_size: usize,
    /// Number of key/value heads (for GQA: `num_kv_heads < num_heads`).
    pub num_kv_heads: usize,
    /// CUDA C++ data type for Q/K/V elements (`"float"` or `"half"`).
    pub dtype: &'static str,
    /// Kernel function name.
    pub kernel_name: String,
    /// SM target for compilation (e.g., `"sm_80"`).
    pub sm_target: String,
}

impl PtxAttentionConfig {
    /// Create a config with sensible defaults.
    ///
    /// Scale defaults to `1.0 / sqrt(head_dim)`. SM target defaults to `sm_80`.
    /// Dtype defaults to `"float"`. `num_kv_heads` defaults to `num_heads` (MHA).
    /// `block_size` defaults to `min(round_up(kv_seq_len, 32), 256)`.
    /// `causal` defaults to `false`.
    pub fn new(
        kernel_name: &str,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
    ) -> Self {
        let scale = if head_dim > 0 {
            1.0 / (head_dim as f32).sqrt()
        } else {
            1.0
        };
        let block_size = compute_default_block_size(kv_seq_len);
        Self {
            num_heads,
            head_dim,
            seq_len,
            kv_seq_len,
            scale,
            causal: false,
            block_size,
            num_kv_heads: num_heads,
            dtype: "float",
            kernel_name: kernel_name.to_string(),
            sm_target: DEFAULT_SM_TARGET.to_string(),
        }
    }

    /// Set GQA key/value head count.
    #[must_use]
    pub fn with_num_kv_heads(mut self, num_kv_heads: usize) -> Self {
        self.num_kv_heads = num_kv_heads;
        self
    }

    /// Set causal masking.
    #[must_use]
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set data type (`"float"` or `"half"`).
    #[must_use]
    pub fn with_dtype(mut self, dtype: &'static str) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set attention scale factor.
    #[must_use]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set block size (number of threads per block).
    #[must_use]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
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
        if self.head_dim == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "head_dim must be > 0".into(),
            ));
        }
        if self.num_heads == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "num_heads must be > 0".into(),
            ));
        }
        if self.num_kv_heads == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "num_kv_heads must be > 0".into(),
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
        if self.block_size == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "block_size must be > 0".into(),
            ));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }
        if self.kernel_name.is_empty() {
            return Err(PtxCodegenError::InvalidParameter(
                "kernel_name must not be empty".into(),
            ));
        }
        if self.dtype != "float" && self.dtype != "half" {
            return Err(PtxCodegenError::InvalidParameter(format!(
                "dtype must be \"float\" or \"half\", got \"{}\"",
                self.dtype
            )));
        }
        if !self.scale.is_finite() {
            return Err(PtxCodegenError::InvalidParameter(
                "scale must be finite".into(),
            ));
        }
        Ok(())
    }

    /// Number of Q heads that share each K/V head.
    #[must_use]
    pub fn heads_per_kv_group(&self) -> usize {
        if self.num_kv_heads == 0 {
            return 0;
        }
        self.num_heads / self.num_kv_heads
    }

    /// Whether this is standard MHA (num_kv_heads == num_heads).
    #[must_use]
    pub fn is_mha(&self) -> bool {
        self.num_kv_heads == self.num_heads
    }

    /// Whether this is multi-query attention (num_kv_heads == 1).
    #[must_use]
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }

    /// Whether this is cross-attention (kv_seq_len != seq_len).
    #[must_use]
    pub fn is_cross_attention(&self) -> bool {
        self.kv_seq_len != self.seq_len
    }

    /// Shared memory bytes needed.
    ///
    /// `kv_seq_len` floats for attention scores + `block_size` floats for
    /// reduction scratch.
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        let score_bytes = self.kv_seq_len * 4; // f32 scores
        let reduce_bytes = self.block_size * 4; // reduction scratch
        score_bytes + reduce_bytes
    }
}

/// Compute the default block size for a given kv_seq_len.
///
/// Round `kv_seq_len` up to next multiple of `WARP_SIZE`, cap at `MAX_BLOCK_SIZE`.
fn compute_default_block_size(kv_seq_len: usize) -> usize {
    if kv_seq_len == 0 {
        return WARP_SIZE;
    }
    let rounded = kv_seq_len.div_ceil(WARP_SIZE) * WARP_SIZE;
    rounded.min(MAX_BLOCK_SIZE)
}

// ---------------------------------------------------------------------------
// CUDA C++ generation
// ---------------------------------------------------------------------------

/// Generate CUDA C++ source for scaled dot-product attention.
///
/// Generates a `__global__` kernel implementing the full attention pipeline:
/// score computation, optional causal masking, softmax, and value aggregation.
///
/// ## Tensor layouts
///
/// - Q: `[batch, num_heads, seq_len, head_dim]`
/// - K: `[batch, num_kv_heads, kv_seq_len, head_dim]`
/// - V: `[batch, num_kv_heads, kv_seq_len, head_dim]`
/// - Output: `[batch, num_heads, seq_len, head_dim]`
///
/// ## Parameters passed to kernel
///
/// - `Q`, `K`, `V`: device pointers to input tensors
/// - `output`: device pointer to output tensor
/// - `batch_size`: number of batches
///
/// # Errors
///
/// Returns `PtxCodegenError::InvalidParameter` if config validation fails.
pub fn emit_ptx_attention(config: &PtxAttentionConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let name = &config.kernel_name;
    let head_dim = config.head_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let seq_len = config.seq_len;
    let kv_seq_len = config.kv_seq_len;
    let dtype = config.dtype;
    let causal = config.causal;
    let scale = config.scale;
    let block_size = config.block_size;
    let heads_per_group = config.heads_per_kv_group();

    let mut src = String::with_capacity(8192);

    // -- CUDA C++ prelude --
    src.push_str(CUDA_PRELUDE);
    src.push_str("#include <float.h>\n\n");

    // -- Comment header --
    src.push_str(&format!(
        "// Fused scaled dot-product attention\n\
         // head_dim={head_dim}, num_heads={num_heads}, num_kv_heads={num_kv_heads}\n\
         // seq_len={seq_len}, kv_seq_len={kv_seq_len}, dtype={dtype}\n\
         // causal={causal}, scale={scale}\n\
         // block_size={block_size}, GQA group_size={heads_per_group}\n\n"
    ));

    // -- Kernel signature --
    src.push_str(&format!(
        "__global__ void {name}(\n\
         \x20   const {dtype}* __restrict__ Q,\n\
         \x20   const {dtype}* __restrict__ K,\n\
         \x20   const {dtype}* __restrict__ V,\n\
         \x20   {dtype}* __restrict__ output,\n\
         \x20   const unsigned int batch_size\n\
         ) {{\n"
    ));

    // -- Shared memory declarations --
    src.push_str(&format!(
        "\x20   // Shared memory: scores[{kv_seq_len}] + reduce_buf[{block_size}]\n\
         \x20   __shared__ float scores[{kv_seq_len}];\n\
         \x20   __shared__ float reduce_buf[{block_size}];\n\n"
    ));

    // -- Thread/block index computation --
    // Grid: (batch_size, num_heads, seq_len) via blockIdx.x/y/z
    // Each block handles one (batch, head, query_pos) tuple
    src.push_str(
        "\x20   // Block indices: one block per (batch, head, query_pos)\n\
         \x20   const unsigned int batch_idx = blockIdx.x;\n\
         \x20   const unsigned int head_idx  = blockIdx.y;\n\
         \x20   const unsigned int q_pos     = blockIdx.z;\n\
         \x20   const unsigned int tid       = threadIdx.x;\n\n",
    );

    // -- Bounds check --
    src.push_str(
        "\x20   // Early exit if batch out of range\n\
         \x20   if (batch_idx >= batch_size) return;\n\n",
    );

    // -- GQA: compute K/V head index --
    if heads_per_group > 1 {
        src.push_str(&format!(
            "\x20   // GQA: map Q head to K/V head\n\
             \x20   const unsigned int kv_head_idx = head_idx / {heads_per_group}u;\n\n"
        ));
    } else {
        src.push_str(
            "\x20   // MHA: Q head maps 1:1 to K/V head\n\
             \x20   const unsigned int kv_head_idx = head_idx;\n\n",
        );
    }

    // -- Pointer offsets --
    // Q layout: [batch, num_heads, seq_len, head_dim]
    // K/V layout: [batch, num_kv_heads, kv_seq_len, head_dim]
    src.push_str(&format!(
        "\x20   // Pointer offsets into Q, K, V, output\n\
         \x20   const unsigned int q_offset = ((batch_idx * {num_heads}u + head_idx) * {seq_len}u + q_pos) * {head_dim}u;\n\
         \x20   const unsigned int kv_batch_offset = batch_idx * {num_kv_heads}u * {kv_seq_len}u * {head_dim}u;\n\
         \x20   const unsigned int kv_head_offset = kv_head_idx * {kv_seq_len}u * {head_dim}u;\n\
         \x20   const unsigned int kv_base = kv_batch_offset + kv_head_offset;\n\n"
    ));

    // =====================================================================
    // Phase 1: Compute attention scores (Q @ K^T * scale)
    // =====================================================================
    src.push_str(&format!(
        "\x20   // ---- Phase 1: score = Q @ K^T * scale ----\n\
         \x20   for (unsigned int j = tid; j < {kv_seq_len}u; j += {block_size}u) {{\n\
         \x20       float dot = 0.0f;\n\
         \x20       const unsigned int k_offset = kv_base + j * {head_dim}u;\n\
         \x20       for (unsigned int d = 0; d < {head_dim}u; d++) {{\n\
         \x20           dot += (float)Q[q_offset + d] * (float)K[k_offset + d];\n\
         \x20       }}\n\
         \x20       float score = dot * {scale}f;\n"
    ));

    // -- Causal mask: set score to -inf for future positions --
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
    // Phase 2: Softmax over scores
    // =====================================================================
    // Phase 2a: find max for numerical stability
    src.push_str(&format!(
        "\x20   // ---- Phase 2a: find max score (numerically stable softmax) ----\n\
         \x20   float local_max = -FLT_MAX;\n\
         \x20   for (unsigned int j = tid; j < {kv_seq_len}u; j += {block_size}u) {{\n\
         \x20       if (scores[j] > local_max) local_max = scores[j];\n\
         \x20   }}\n\n"
    ));

    // Warp-level max reduction
    src.push_str(
        "\x20   // Warp-level max reduction\n\
         \x20   for (int offset = 16; offset > 0; offset >>= 1) {\n\
         \x20       float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);\n\
         \x20       if (other > local_max) local_max = other;\n\
         \x20   }\n\n",
    );

    // Cross-warp max reduction via shared memory
    emit_cross_warp_reduce_max(&mut src, block_size);

    // Phase 2b: compute exp and sum
    src.push_str(&format!(
        "\x20   // ---- Phase 2b: exp(score - max) and sum ----\n\
         \x20   float local_sum = 0.0f;\n\
         \x20   for (unsigned int j = tid; j < {kv_seq_len}u; j += {block_size}u) {{\n\
         \x20       float e = expf(scores[j] - local_max);\n\
         \x20       scores[j] = e;\n\
         \x20       local_sum += e;\n\
         \x20   }}\n\n"
    ));

    // Warp-level sum reduction
    src.push_str(
        "\x20   // Warp-level sum reduction\n\
         \x20   for (int offset = 16; offset > 0; offset >>= 1) {\n\
         \x20       local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);\n\
         \x20   }\n\n",
    );

    // Cross-warp sum reduction via shared memory
    emit_cross_warp_reduce_sum(&mut src, block_size);

    // Phase 2c: normalize scores
    src.push_str(&format!(
        "\x20   // ---- Phase 2c: normalize (divide by sum) ----\n\
         \x20   float inv_sum = (local_sum > 0.0f) ? (1.0f / local_sum) : 0.0f;\n\
         \x20   for (unsigned int j = tid; j < {kv_seq_len}u; j += {block_size}u) {{\n\
         \x20       scores[j] *= inv_sum;\n\
         \x20   }}\n\
         \x20   __syncthreads();\n\n"
    ));

    // =====================================================================
    // Phase 3: Value aggregation (attn_weights @ V)
    // =====================================================================
    src.push_str(&format!(
        "\x20   // ---- Phase 3: output = attn_weights @ V ----\n\
         \x20   const unsigned int out_offset = ((batch_idx * {num_heads}u + head_idx) * {seq_len}u + q_pos) * {head_dim}u;\n\
         \x20   for (unsigned int d = tid; d < {head_dim}u; d += {block_size}u) {{\n\
         \x20       float acc = 0.0f;\n\
         \x20       for (unsigned int j = 0; j < {kv_seq_len}u; j++) {{\n\
         \x20           acc += scores[j] * (float)V[kv_base + j * {head_dim}u + d];\n\
         \x20       }}\n\
         \x20       output[out_offset + d] = ({dtype})acc;\n\
         \x20   }}\n"
    ));

    // -- Kernel close --
    src.push_str("}\n");

    Ok(src)
}

/// Emit CUDA C++ cross-warp max reduction via shared memory `reduce_buf`.
fn emit_cross_warp_reduce_max(src: &mut String, block_size: usize) {
    let num_warps = block_size / WARP_SIZE;
    if num_warps <= 1 {
        // Single warp: broadcast lane 0's result to all lanes
        src.push_str(
            "\x20   // Broadcast max from lane 0 to all threads in warp\n\
             \x20   local_max = __shfl_sync(0xFFFFFFFF, local_max, 0);\n\n",
        );
        return;
    }

    src.push_str(&format!(
        "\x20   // Cross-warp max reduction (shared memory)\n\
         \x20   unsigned int warp_id = tid / 32;\n\
         \x20   unsigned int lane_id = tid % 32;\n\
         \x20   if (lane_id == 0) reduce_buf[warp_id] = local_max;\n\
         \x20   __syncthreads();\n\
         \x20   // First warp reduces all warp-level maxima\n\
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
         \x20   // Broadcast result to all threads\n\
         \x20   if (tid == 0) reduce_buf[0] = local_max;\n\
         \x20   __syncthreads();\n\
         \x20   local_max = reduce_buf[0];\n\n"
    ));
}

/// Emit CUDA C++ cross-warp sum reduction via shared memory `reduce_buf`.
fn emit_cross_warp_reduce_sum(src: &mut String, block_size: usize) {
    let num_warps = block_size / WARP_SIZE;
    if num_warps <= 1 {
        // Single warp: broadcast lane 0's result to all lanes
        src.push_str(
            "\x20   // Broadcast sum from lane 0 to all threads in warp\n\
             \x20   local_sum = __shfl_sync(0xFFFFFFFF, local_sum, 0);\n\n",
        );
        return;
    }

    src.push_str(&format!(
        "\x20   // Cross-warp sum reduction (shared memory)\n\
         \x20   if (lane_id == 0) reduce_buf[warp_id] = local_sum;\n\
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

/// Generate CUDA C++ attention kernel with default configuration.
///
/// Uses `"float"` dtype, causal=true, MHA (num_kv_heads == num_heads),
/// scale = `1/sqrt(head_dim)`, block_size=256, sm_80 target.
pub fn emit_ptx_attention_default(
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_seq_len: usize,
) -> Result<String, PtxCodegenError> {
    let config =
        PtxAttentionConfig::new("sdpa_attention", num_heads, head_dim, seq_len, kv_seq_len)
            .with_causal(true);
    emit_ptx_attention(&config)
}

/// Compute the CUDA launch configuration for the attention kernel.
///
/// Grid: `(batch_size, num_heads, seq_len)` — one block per query position
/// per head per batch element.
/// Block: `(block_size, 1, 1)` — threads cooperate on scores/softmax.
///
/// # Arguments
///
/// * `config` — Attention configuration.
/// * `batch_size` — Number of batch elements.
///
/// # Returns
///
/// A `CudaLaunchConfig` with grid, block, and shared memory sizes.
#[must_use]
pub fn ptx_attention_launch_config(
    config: &PtxAttentionConfig,
    batch_size: usize,
) -> CudaLaunchConfig {
    use crate::cuda_ffi::CudaDim3;

    let block_size = config.block_size as u32;
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
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Generate CUDA C++ for scaled dot-product attention (non-causal).
///
/// Produces a single-head SDPA kernel: `softmax(Q @ K^T / sqrt(head_dim)) @ V`.
/// Self-attention layout with `seq_len == kv_seq_len`.
///
/// # Arguments
///
/// * `seq_len` — Query and key/value sequence length.
/// * `head_dim` — Per-head embedding dimension.
pub fn generate_sdpa_ptx(seq_len: u32, head_dim: u32) -> String {
    let config = PtxAttentionConfig::new(
        "sdpa_f32",
        1, // single head
        head_dim as usize,
        seq_len as usize,
        seq_len as usize, // self-attention
    );
    emit_ptx_attention(&config).expect("SDPA generation failed")
}

/// Generate CUDA C++ for scaled dot-product attention with causal mask.
///
/// Produces a single-head SDPA kernel with causal masking: future positions
/// in K are set to `-FLT_MAX` before softmax. Self-attention layout with
/// `seq_len == kv_seq_len`.
///
/// # Arguments
///
/// * `seq_len` — Query and key/value sequence length.
/// * `head_dim` — Per-head embedding dimension.
pub fn generate_sdpa_causal_ptx(seq_len: u32, head_dim: u32) -> String {
    let config = PtxAttentionConfig::new(
        "sdpa_causal_f32",
        1, // single head
        head_dim as usize,
        seq_len as usize,
        seq_len as usize, // self-attention
    )
    .with_causal(true);
    emit_ptx_attention(&config).expect("SDPA causal generation failed")
}

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation for scaled dot-product attention.
///
/// Computes `softmax(Q @ K^T / sqrt(head_dim)) @ V` on CPU for a single head.
/// Used for differential testing against the GPU kernel.
///
/// ## Tensor layouts (flat, row-major)
///
/// - Q: `[seq_len, head_dim]`
/// - K: `[kv_seq_len, head_dim]`
/// - V: `[kv_seq_len, head_dim]`
/// - Output: `[seq_len, head_dim]`
///
/// # Arguments
///
/// * `q` — Query tensor, shape `[seq_len, head_dim]`, flattened.
/// * `k` — Key tensor, shape `[kv_seq_len, head_dim]`, flattened.
/// * `v` — Value tensor, shape `[kv_seq_len, head_dim]`, flattened.
/// * `head_dim` — Per-head embedding dimension.
///
/// # Panics
///
/// Panics if `head_dim == 0` or tensor lengths are not multiples of `head_dim`.
pub fn sdpa_reference(q: &[f32], k: &[f32], v: &[f32], head_dim: usize) -> Vec<f32> {
    assert!(head_dim > 0, "head_dim must be > 0");
    assert!(
        q.len().is_multiple_of(head_dim),
        "q length must be a multiple of head_dim"
    );
    assert!(
        k.len().is_multiple_of(head_dim),
        "k length must be a multiple of head_dim"
    );
    assert_eq!(k.len(), v.len(), "k and v must have the same length");

    let seq_len = q.len() / head_dim;
    let kv_seq_len = k.len() / head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0f32; seq_len * head_dim];

    for i in 0..seq_len {
        // Phase 1: Compute scores = Q[i] @ K^T * scale
        let mut scores = vec![0.0f32; kv_seq_len];
        let q_base = i * head_dim;
        for j in 0..kv_seq_len {
            let k_base = j * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_base + d] * k[k_base + d];
            }
            scores[j] = dot * scale;
        }

        // Phase 2: Softmax (numerically stable)
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

        // Phase 3: Value aggregation = attn_weights @ V
        let out_base = i * head_dim;
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for j in 0..kv_seq_len {
                acc += scores[j] * v[j * head_dim + d];
            }
            output[out_base + d] = acc;
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_attention_tests.rs"]
mod tests;
