// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal-optimized dispatch for gpt-oss-20b inference on Apple Silicon.
//!
//! Provides optimized execution planning for M4 Max / M-series chips:
//! - Activation memory estimation for Metal buffer pre-allocation
//! - Chunked prefill to fit within Metal memory budgets
//! - Pre-computed dispatch plans with latency estimation
//! - Contiguous buffer layout for zero-copy weight access
//!
//! This module complements [`gpu_dispatch`](crate::gpu_dispatch) (threadgroup
//! sizes and grid computation) and [`moe_dispatch`](crate::moe_dispatch)
//! (fused MoE execution) by adding higher-level inference planning:
//! memory budgeting, sequence chunking, and dispatch count estimation.
//!
//! # Architecture
//!
//! The dispatch planner does NOT generate MSL or launch Metal kernels.
//! It produces [`MetalDispatchPlan`] and [`MetalBufferLayout`] structs
//! consumed by the Metal backend at runtime.

use nn_core::DType;

use crate::config::GptOssConfig;

// ---------------------------------------------------------------------------
// Metal inference configuration
// ---------------------------------------------------------------------------

/// Configuration for Metal-optimized gpt-oss inference.
///
/// Controls memory budgets, sequence chunking, and dtype selection for
/// Apple Silicon. Use [`MetalInferenceConfig::m4_max()`] for M4 Max
/// defaults or construct manually for other devices.
#[derive(Clone, Debug)]
pub(crate) struct MetalInferenceConfig {
    /// Maximum batch size for inference.
    pub(crate) max_batch_size: usize,
    /// Maximum sequence length (tokens) the model supports.
    pub(crate) max_seq_len: usize,
    /// Use BF16 for weights and activations (M4 Max has native BF16).
    pub(crate) use_bf16: bool,
    /// Activation memory budget in bytes for Metal buffer allocation.
    ///
    /// Chunked prefill splits long sequences so peak activation memory
    /// stays within this budget. Default 4 GB for M4 Max (128 GB unified).
    pub(crate) activation_memory_budget: usize,
    /// Maximum tokens per prefill chunk.
    ///
    /// Long prompts are split into chunks of this size for incremental
    /// processing, reducing peak activation memory. Default 2048.
    pub(crate) prefill_chunk_size: usize,
}

impl MetalInferenceConfig {
    /// Default configuration for M4 Max (128 EU, 128 GB unified memory).
    ///
    /// - BF16 enabled (native hardware support, halves memory vs F32)
    /// - 4 GB activation budget (leaves headroom for weights + KV cache)
    /// - 2048-token prefill chunks (balances throughput vs memory)
    #[must_use]
    pub(crate) fn m4_max() -> Self {
        Self {
            max_batch_size: 1,
            max_seq_len: 131_072,
            use_bf16: true,
            activation_memory_budget: 4 * 1024 * 1024 * 1024, // 4 GB
            prefill_chunk_size: 2048,
        }
    }

    /// Configuration for smaller Apple Silicon (M1/M2 base, 8-16 GB).
    ///
    /// Tighter memory budget and smaller prefill chunks to avoid OOM.
    #[must_use]
    pub(crate) fn apple_silicon_base() -> Self {
        Self {
            max_batch_size: 1,
            max_seq_len: 32_768,
            use_bf16: true,
            activation_memory_budget: 1024 * 1024 * 1024, // 1 GB
            prefill_chunk_size: 512,
        }
    }

    /// Bytes per element for the configured dtype.
    #[must_use]
    fn bytes_per_element(&self) -> usize {
        if self.use_bf16 {
            2
        } else {
            4
        }
    }
}

impl Default for MetalInferenceConfig {
    fn default() -> Self {
        Self::m4_max()
    }
}

// ---------------------------------------------------------------------------
// Activation memory estimation
// ---------------------------------------------------------------------------

/// Estimate peak activation memory (bytes) for a single forward pass.
///
/// Accounts for the three major memory consumers per decoder layer:
///
/// 1. **Attention activations**: Q, K, V projections + attention score matrix
///    + attention output. The score matrix is quadratic in `seq_len` for
///    full-attention layers.
///
/// 2. **KV cache growth**: K and V tensors appended per layer. Sliding
///    attention layers are capped at `min(seq_len, sliding_window)`.
///
/// 3. **MoE intermediate buffers**: Per-expert gate_up activations (top_k
///    experts active), scatter-add accumulation buffer.
///
/// Returns the byte count for the **peak** single-layer activation footprint
/// multiplied by 2 (double-buffering: current layer + residual). This is
/// the minimum Metal buffer arena size needed to avoid mid-pass allocation.
///
/// Uses `checked_mul`/`checked_add` throughout; returns `None` on overflow.
#[must_use]
pub(crate) fn estimate_activation_memory(
    cfg: &GptOssConfig,
    batch: usize,
    seq_len: usize,
    use_bf16: bool,
) -> Option<usize> {
    let bpe: usize = if use_bf16 { 2 } else { 4 };
    let h = cfg.hidden_size;
    let ad = cfg.attn_dim();
    let kvd = cfg.kv_dim();
    let ne = cfg.num_local_experts;
    let inter = cfg.intermediate_size;
    let top_k = cfg.experts_per_token;
    let n_heads = cfg.num_attention_heads;

    let tokens = batch.checked_mul(seq_len)?;

    // -- Attention activations --
    // Q: [tokens, attn_dim]
    let q_elems = tokens.checked_mul(ad)?;
    // K: [tokens, kv_dim]
    let k_elems = tokens.checked_mul(kvd)?;
    // V: [tokens, kv_dim]
    let v_elems = tokens.checked_mul(kvd)?;
    // Attention scores: [batch, n_heads, seq_len, seq_len] (quadratic)
    let score_elems = batch
        .checked_mul(n_heads)?
        .checked_mul(seq_len)?
        .checked_mul(seq_len)?;
    // Attention output: [tokens, attn_dim]
    let attn_out_elems = tokens.checked_mul(ad)?;

    let attn_elems = q_elems
        .checked_add(k_elems)?
        .checked_add(v_elems)?
        .checked_add(score_elems)?
        .checked_add(attn_out_elems)?;

    // -- MoE activations --
    // Router logits: [tokens, num_experts]
    let router_elems = tokens.checked_mul(ne)?;
    // Per-expert intermediate: top_k experts * tokens * 2 * inter (gate + up)
    let expert_tokens = tokens.checked_mul(top_k)?;
    let fused_dim = 2_usize.checked_mul(inter)?;
    let expert_inter_elems = expert_tokens.checked_mul(fused_dim)?;
    // Down projection output: [expert_tokens, hidden]
    let expert_down_elems = expert_tokens.checked_mul(h)?;
    // Scatter-add accumulation: [tokens, hidden]
    let scatter_elems = tokens.checked_mul(h)?;

    let moe_elems = router_elems
        .checked_add(expert_inter_elems)?
        .checked_add(expert_down_elems)?
        .checked_add(scatter_elems)?;

    // Peak per-layer = attention + MoE
    let per_layer_elems = attn_elems.checked_add(moe_elems)?;
    let per_layer_bytes = per_layer_elems.checked_mul(bpe)?;

    // Double-buffer: current layer activations + residual stream
    per_layer_bytes.checked_mul(2)
}

// ---------------------------------------------------------------------------
// Chunked prefill
// ---------------------------------------------------------------------------

/// Split a long sequence into chunks that fit within a Metal memory budget.
///
/// Returns a list of `(start, len)` pairs covering positions `[0, seq_len)`.
/// Each chunk's estimated activation memory is at most `memory_budget` bytes.
///
/// The algorithm binary-searches for the largest chunk size whose activation
/// memory fits the budget, then tiles the sequence with that chunk size.
/// The last chunk may be shorter.
///
/// # Arguments
/// - `cfg`: Model configuration (determines memory per token).
/// - `seq_len`: Total prompt length in tokens.
/// - `memory_budget`: Maximum activation memory per chunk (bytes).
///
/// Returns an empty vec if `seq_len == 0`.
#[must_use]
pub(crate) fn optimal_prefill_chunks(
    cfg: &GptOssConfig,
    seq_len: usize,
    memory_budget: usize,
) -> Vec<(usize, usize)> {
    if seq_len == 0 || memory_budget == 0 {
        return Vec::new();
    }

    // Find the largest chunk_size whose activation memory fits the budget.
    // Binary search on chunk_size in [1, seq_len].
    let mut lo: usize = 1;
    let mut hi: usize = seq_len;
    let mut best: usize = 1;

    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let mem = estimate_activation_memory(cfg, 1, mid, false);
        match mem {
            Some(m) if m <= memory_budget => {
                best = mid;
                if mid == hi {
                    break;
                }
                lo = mid + 1;
            }
            _ => {
                if mid == 1 {
                    break;
                }
                hi = mid - 1;
            }
        }
    }

    // Tile the sequence with the best chunk size.
    let mut chunks = Vec::new();
    let mut pos = 0;
    while pos < seq_len {
        let len = best.min(seq_len - pos);
        chunks.push((pos, len));
        pos += len;
    }
    chunks
}

// ---------------------------------------------------------------------------
// Dispatch plan
// ---------------------------------------------------------------------------

/// Pre-computed Metal dispatch plan for a specific sequence length.
///
/// Provides dispatch counts and estimated latency for a full forward pass
/// (all decoder layers). Used by the Metal backend to pre-allocate command
/// buffer capacity and estimate completion time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MetalDispatchPlan {
    /// Number of attention-related GPU dispatches (Q@K, softmax, attn@V).
    pub(crate) num_attention_dispatches: usize,
    /// Number of MoE-related GPU dispatches (router, expert FFN, scatter).
    pub(crate) num_moe_dispatches: usize,
    /// Total GPU dispatches for the full forward pass.
    pub(crate) total_dispatches: usize,
    /// Estimated latency in microseconds (based on M4 Max dispatch overhead).
    ///
    /// This is a rough estimate using 5 us per dispatch (Metal command encoder
    /// overhead) plus a bandwidth term for large matmuls. Useful for scheduling
    /// and timeout estimation, not for precise benchmarking.
    pub(crate) estimated_latency_us: u64,
}

/// Per-dispatch overhead in microseconds on M4 Max.
///
/// Metal command encoder overhead for a single dispatch is ~3-8 us on M4 Max.
/// We use 5 us as a conservative average.
const DISPATCH_OVERHEAD_US: u64 = 5;

/// Compute the Metal dispatch plan for a forward pass.
///
/// # Dispatch breakdown per decoder layer
///
/// **Attention (prefill, seq_len > 1):**
/// - RMSNorm (1 dispatch)
/// - Q/K/V projections (3 matmul dispatches)
/// - RoPE application (1 dispatch)
/// - Q @ K^T score matmul (1 dispatch)
/// - Softmax (1 dispatch)
/// - Attn @ V matmul (1 dispatch)
/// - O projection (1 dispatch)
/// - Residual add (1 dispatch)
///
/// Total: 10 dispatches per layer
///
/// **Attention (decode, seq_len == 1):**
/// Same ops but attention scores are vector-matrix (1 x cached_len),
/// so dispatch count is identical but each is much faster.
/// Total: 10 dispatches per layer
///
/// **MoE (fused dispatch):**
/// - Post-attention RMSNorm (1 dispatch)
/// - Router matmul + softmax + top-k (1 dispatch batch)
/// - Per-active-expert gate_up + SwiGLU + down (3 dispatches each, top_k experts)
/// - Scatter-add (1 dispatch)
/// - Residual add (1 dispatch)
///
/// Total: 4 + 3 * top_k dispatches per layer
///
/// **Global (once per pass):**
/// - Final RMSNorm (1 dispatch)
/// - lm_head matmul (1 dispatch)
///
/// Total: 2 dispatches
#[must_use]
pub(crate) fn plan_dispatches(
    cfg: &GptOssConfig,
    seq_len: usize,
    cached_len: usize,
) -> MetalDispatchPlan {
    let num_layers = cfg.num_hidden_layers;
    let top_k = cfg.experts_per_token;

    // Per-layer attention dispatches: RMSNorm + Q/K/V + RoPE + QK^T +
    // softmax + attn@V + O_proj + residual = 10
    let attn_per_layer: usize = 10;

    // Per-layer MoE dispatches (fused): post-norm + router_batch +
    // 3*top_k (gate_up + silu_clamp + down per expert) + scatter + residual
    let moe_per_layer: usize = 4 + 3 * top_k;

    let total_attn = attn_per_layer.saturating_mul(num_layers);
    let total_moe = moe_per_layer.saturating_mul(num_layers);

    // Global dispatches: final RMSNorm + lm_head
    let global_dispatches: usize = 2;

    let total = total_attn
        .saturating_add(total_moe)
        .saturating_add(global_dispatches);

    // Latency estimation: dispatch overhead + bandwidth term
    let dispatch_us = (total as u64).saturating_mul(DISPATCH_OVERHEAD_US);

    // Bandwidth term: rough estimate for attention score computation.
    // Prefill is compute-bound (quadratic), decode is bandwidth-bound.
    let bandwidth_us = if seq_len > 1 {
        // Prefill: O(seq^2) attention per layer, ~0.01 us per element
        let score_elements = (seq_len as u64)
            .saturating_mul(seq_len.saturating_add(cached_len) as u64)
            .saturating_mul(cfg.num_attention_heads as u64);
        let per_layer_us = score_elements / 100; // ~100 elements per us on M4 Max
        per_layer_us.saturating_mul(num_layers as u64)
    } else {
        // Decode: bandwidth-bound, ~1 us per layer for vector-matrix attention
        let total_seq = cached_len.saturating_add(1) as u64;
        let per_layer_us = total_seq
            .saturating_mul(cfg.num_attention_heads as u64)
            .saturating_mul(cfg.head_dim as u64)
            / 1_000_000; // bytes / bandwidth (~1 TB/s on M4 Max)
        per_layer_us.saturating_mul(num_layers as u64)
    };

    let estimated_latency_us = dispatch_us.saturating_add(bandwidth_us);

    MetalDispatchPlan {
        num_attention_dispatches: total_attn,
        num_moe_dispatches: total_moe,
        total_dispatches: total,
        estimated_latency_us,
    }
}

// ---------------------------------------------------------------------------
// Buffer layout
// ---------------------------------------------------------------------------

/// Pre-computed buffer offsets for zero-copy weight access from a single
/// contiguous Metal buffer.
///
/// All model weights are laid out in a single mmap-backed Metal buffer.
/// Per-layer offsets enable direct `setBufferOffset:` calls without
/// creating sub-buffers, reducing Metal API overhead.
#[derive(Clone, Debug)]
pub(crate) struct MetalBufferLayout {
    /// Byte offset to each layer's attention weights (Q/K/V/O + norms + sinks).
    pub(crate) attention_offsets: Vec<usize>,
    /// Byte offset to each layer's MoE weights (router + experts).
    pub(crate) moe_offsets: Vec<usize>,
    /// Byte offset to the final norm weight.
    pub(crate) final_norm_offset: usize,
    /// Byte offset to the lm_head weight.
    pub(crate) lm_head_offset: usize,
    /// Total weight buffer size in bytes.
    pub(crate) total_weight_bytes: usize,
}

/// Metal buffer alignment (16 bytes on Apple Silicon).
const BUFFER_ALIGNMENT: usize = 16;

/// Align a byte offset up to 16-byte boundary.
#[must_use]
fn align_up(offset: usize) -> usize {
    (offset + BUFFER_ALIGNMENT - 1) & !(BUFFER_ALIGNMENT - 1)
}

/// Compute the contiguous buffer layout for all model weights.
///
/// Layout order: embedding | per-layer (attention, MoE) | final_norm | lm_head.
/// Each section is 16-byte aligned for Metal buffer offset requirements.
///
/// Returns `None` if size computation overflows `usize`.
#[must_use]
pub(crate) fn compute_buffer_layout(cfg: &GptOssConfig, dtype: DType) -> Option<MetalBufferLayout> {
    let bpe = dtype.size_bytes();
    let h = cfg.hidden_size;
    let ad = cfg.attn_dim();
    let kvd = cfg.kv_dim();
    let ne = cfg.num_local_experts;
    let inter = cfg.intermediate_size;
    let fused_dim = 2_usize.checked_mul(inter)?;
    let num_layers = cfg.num_hidden_layers;

    // Embedding: [vocab_size, hidden_size]
    let embed_bytes = cfg.vocab_size.checked_mul(h)?.checked_mul(bpe)?;
    let mut offset = align_up(embed_bytes);

    let mut attention_offsets = Vec::with_capacity(num_layers);
    let mut moe_offsets = Vec::with_capacity(num_layers);

    for _ in 0..num_layers {
        // -- Attention weights --
        attention_offsets.push(offset);
        // Q: [ad, h] + bias [ad]
        let q_bytes = ad.checked_mul(h)?.checked_add(ad)?.checked_mul(bpe)?;
        // K: [kvd, h] + bias [kvd]
        let k_bytes = kvd.checked_mul(h)?.checked_add(kvd)?.checked_mul(bpe)?;
        // V: [kvd, h] + bias [kvd]
        let v_bytes = kvd.checked_mul(h)?.checked_add(kvd)?.checked_mul(bpe)?;
        // O: [h, ad] + bias [h]
        let o_bytes = h.checked_mul(ad)?.checked_add(h)?.checked_mul(bpe)?;
        // Sinks: [head_dim]
        let sinks_bytes = cfg.head_dim.checked_mul(bpe)?;
        // input_layernorm: [h]
        let in_ln_bytes = h.checked_mul(bpe)?;
        // post_attention_layernorm: [h]
        let post_ln_bytes = h.checked_mul(bpe)?;

        let attn_total = q_bytes
            .checked_add(k_bytes)?
            .checked_add(v_bytes)?
            .checked_add(o_bytes)?
            .checked_add(sinks_bytes)?
            .checked_add(in_ln_bytes)?
            .checked_add(post_ln_bytes)?;

        offset = align_up(offset.checked_add(attn_total)?);

        // -- MoE weights --
        moe_offsets.push(offset);
        // Router: [ne, h] + bias [ne]
        let router_bytes = ne.checked_mul(h)?.checked_add(ne)?.checked_mul(bpe)?;
        // gate_up_proj: [ne, h, fused_dim] + bias [ne, fused_dim]
        let gate_up_w = ne.checked_mul(h)?.checked_mul(fused_dim)?;
        let gate_up_b = ne.checked_mul(fused_dim)?;
        let gate_up_bytes = gate_up_w.checked_add(gate_up_b)?.checked_mul(bpe)?;
        // down_proj: [ne, inter, h] + bias [ne, h]
        let down_w = ne.checked_mul(inter)?.checked_mul(h)?;
        let down_b = ne.checked_mul(h)?;
        let down_bytes = down_w.checked_add(down_b)?.checked_mul(bpe)?;

        let moe_total = router_bytes
            .checked_add(gate_up_bytes)?
            .checked_add(down_bytes)?;

        offset = align_up(offset.checked_add(moe_total)?);
    }

    // Final norm: [h]
    let final_norm_offset = offset;
    let final_norm_bytes = h.checked_mul(bpe)?;
    offset = align_up(offset.checked_add(final_norm_bytes)?);

    // lm_head: [vocab_size, h]
    let lm_head_offset = offset;
    let lm_head_bytes = if cfg.tie_word_embeddings {
        0 // shares embedding buffer
    } else {
        cfg.vocab_size.checked_mul(h)?.checked_mul(bpe)?
    };
    offset = align_up(offset.checked_add(lm_head_bytes)?);

    Some(MetalBufferLayout {
        attention_offsets,
        moe_offsets,
        final_norm_offset,
        lm_head_offset,
        total_weight_bytes: offset,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_20b() -> GptOssConfig {
        GptOssConfig::gptoss_20b()
    }

    fn small_cfg() -> GptOssConfig {
        GptOssConfig::gptoss_20b()
            .with_vocab_size(100)
            .with_num_hidden_layers(2)
            .with_num_local_experts(4)
            .with_experts_per_token(2)
    }

    // -- MetalInferenceConfig --

    #[test]
    fn test_m4_max_config_defaults() {
        let cfg = MetalInferenceConfig::m4_max();
        assert_eq!(cfg.max_batch_size, 1);
        assert_eq!(cfg.max_seq_len, 131_072);
        assert!(cfg.use_bf16);
        assert_eq!(cfg.activation_memory_budget, 4 * 1024 * 1024 * 1024);
        assert_eq!(cfg.prefill_chunk_size, 2048);
    }

    #[test]
    fn test_apple_silicon_base_config() {
        let cfg = MetalInferenceConfig::apple_silicon_base();
        assert!(
            cfg.activation_memory_budget < MetalInferenceConfig::m4_max().activation_memory_budget
        );
        assert!(cfg.prefill_chunk_size < MetalInferenceConfig::m4_max().prefill_chunk_size);
    }

    #[test]
    fn test_bytes_per_element_bf16() {
        let cfg = MetalInferenceConfig::m4_max();
        assert_eq!(cfg.bytes_per_element(), 2);
    }

    #[test]
    fn test_bytes_per_element_f32() {
        let mut cfg = MetalInferenceConfig::m4_max();
        cfg.use_bf16 = false;
        assert_eq!(cfg.bytes_per_element(), 4);
    }

    #[test]
    fn test_default_is_m4_max() {
        let def = MetalInferenceConfig::default();
        let m4 = MetalInferenceConfig::m4_max();
        assert_eq!(def.max_batch_size, m4.max_batch_size);
        assert_eq!(def.max_seq_len, m4.max_seq_len);
        assert_eq!(def.use_bf16, m4.use_bf16);
    }

    // -- Activation memory estimation --

    #[test]
    fn test_activation_memory_nonzero() {
        let cfg = cfg_20b();
        let mem = estimate_activation_memory(&cfg, 1, 128, false).expect("should not overflow");
        assert!(mem > 0, "activation memory must be > 0");
    }

    #[test]
    fn test_activation_memory_scales_with_seq_len() {
        let cfg = cfg_20b();
        let mem_short = estimate_activation_memory(&cfg, 1, 64, false).unwrap();
        let mem_long = estimate_activation_memory(&cfg, 1, 256, false).unwrap();
        assert!(
            mem_long > mem_short,
            "longer sequence should need more memory: short={mem_short}, long={mem_long}"
        );
    }

    #[test]
    fn test_activation_memory_scales_with_batch() {
        let cfg = cfg_20b();
        let mem_b1 = estimate_activation_memory(&cfg, 1, 128, false).unwrap();
        let mem_b2 = estimate_activation_memory(&cfg, 2, 128, false).unwrap();
        assert!(mem_b2 > mem_b1);
    }

    #[test]
    fn test_activation_memory_bf16_half_f32() {
        let cfg = cfg_20b();
        let f32_mem = estimate_activation_memory(&cfg, 1, 128, false).unwrap();
        let bf16_mem = estimate_activation_memory(&cfg, 1, 128, true).unwrap();
        assert_eq!(
            bf16_mem * 2,
            f32_mem,
            "BF16 activation memory should be half F32"
        );
    }

    #[test]
    fn test_activation_memory_single_token_decode() {
        let cfg = cfg_20b();
        let mem = estimate_activation_memory(&cfg, 1, 1, false)
            .expect("single-token decode should not overflow");
        // Single token: attention score matrix is 1x1, very small
        assert!(mem > 0);
        let mem_prefill = estimate_activation_memory(&cfg, 1, 128, false).unwrap();
        assert!(
            mem < mem_prefill,
            "decode should need less memory than prefill"
        );
    }

    #[test]
    fn test_activation_memory_zero_seq_len() {
        let cfg = cfg_20b();
        let mem = estimate_activation_memory(&cfg, 1, 0, false).unwrap();
        assert_eq!(mem, 0, "zero seq_len should produce zero memory");
    }

    // -- Chunked prefill --

    #[test]
    fn test_prefill_chunks_cover_full_sequence() {
        let cfg = small_cfg();
        let seq_len = 1000;
        let budget = 10 * 1024 * 1024; // 10 MB (deliberately small)
        let chunks = optimal_prefill_chunks(&cfg, seq_len, budget);

        assert!(!chunks.is_empty(), "must produce at least one chunk");

        // Verify coverage: chunks must tile [0, seq_len) without gaps
        let mut covered = 0;
        for &(start, len) in &chunks {
            assert_eq!(start, covered, "chunk must start where previous ended");
            assert!(len > 0, "chunk length must be > 0");
            covered += len;
        }
        assert_eq!(covered, seq_len, "chunks must cover entire sequence");
    }

    #[test]
    fn test_prefill_chunks_single_chunk() {
        let cfg = small_cfg();
        let seq_len = 10;
        let budget = 1024 * 1024 * 1024; // 1 GB (huge)
        let chunks = optimal_prefill_chunks(&cfg, seq_len, budget);
        assert_eq!(chunks.len(), 1, "small sequence should fit in one chunk");
        assert_eq!(chunks[0], (0, 10));
    }

    #[test]
    fn test_prefill_chunks_zero_seq_len() {
        let cfg = small_cfg();
        let chunks = optimal_prefill_chunks(&cfg, 0, 1024 * 1024);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_prefill_chunks_zero_budget() {
        let cfg = small_cfg();
        let chunks = optimal_prefill_chunks(&cfg, 100, 0);
        assert!(chunks.is_empty());
    }

    // -- Dispatch plan --

    #[test]
    fn test_dispatch_plan_decode() {
        let cfg = cfg_20b();
        let plan = plan_dispatches(&cfg, 1, 100);
        assert!(plan.total_dispatches > 0);
        assert_eq!(
            plan.total_dispatches,
            plan.num_attention_dispatches + plan.num_moe_dispatches + 2
        );
        assert!(plan.estimated_latency_us > 0);
    }

    #[test]
    fn test_dispatch_plan_prefill() {
        let cfg = cfg_20b();
        let plan = plan_dispatches(&cfg, 128, 0);
        assert!(plan.total_dispatches > 0);
        // Prefill should have higher latency estimate than decode
        let decode_plan = plan_dispatches(&cfg, 1, 128);
        assert!(
            plan.estimated_latency_us >= decode_plan.estimated_latency_us,
            "prefill latency should be >= decode latency"
        );
    }

    #[test]
    fn test_dispatch_plan_attention_count() {
        let cfg = cfg_20b();
        let plan = plan_dispatches(&cfg, 1, 0);
        // 10 attention dispatches per layer * 24 layers = 240
        assert_eq!(plan.num_attention_dispatches, 10 * 24);
    }

    #[test]
    fn test_dispatch_plan_moe_count() {
        let cfg = cfg_20b();
        let plan = plan_dispatches(&cfg, 1, 0);
        // (4 + 3*4) per layer * 24 layers = 16 * 24 = 384
        assert_eq!(plan.num_moe_dispatches, (4 + 3 * 4) * 24);
    }

    #[test]
    fn test_dispatch_plan_total_with_global() {
        let cfg = cfg_20b();
        let plan = plan_dispatches(&cfg, 1, 0);
        // total = attention + moe + 2 global
        let expected = 10 * 24 + (4 + 3 * 4) * 24 + 2;
        assert_eq!(plan.total_dispatches, expected);
    }

    // -- Buffer layout --

    #[test]
    fn test_buffer_layout_nonzero() {
        let cfg = cfg_20b();
        let layout = compute_buffer_layout(&cfg, DType::F32).expect("should not overflow");
        assert!(layout.total_weight_bytes > 0);
    }

    #[test]
    fn test_buffer_layout_layer_count() {
        let cfg = cfg_20b();
        let layout = compute_buffer_layout(&cfg, DType::F32).unwrap();
        assert_eq!(layout.attention_offsets.len(), 24);
        assert_eq!(layout.moe_offsets.len(), 24);
    }

    #[test]
    fn test_buffer_layout_offsets_increasing() {
        let cfg = cfg_20b();
        let layout = compute_buffer_layout(&cfg, DType::F32).unwrap();
        for i in 1..layout.attention_offsets.len() {
            assert!(
                layout.attention_offsets[i] > layout.attention_offsets[i - 1],
                "attention offsets must be strictly increasing"
            );
        }
        for i in 1..layout.moe_offsets.len() {
            assert!(
                layout.moe_offsets[i] > layout.moe_offsets[i - 1],
                "MoE offsets must be strictly increasing"
            );
        }
    }

    #[test]
    fn test_buffer_layout_no_overlap() {
        let cfg = cfg_20b();
        let layout = compute_buffer_layout(&cfg, DType::F32).unwrap();
        // Attention offset < MoE offset within each layer
        for i in 0..cfg.num_hidden_layers {
            assert!(
                layout.attention_offsets[i] < layout.moe_offsets[i],
                "layer {i}: attention offset must precede MoE offset"
            );
        }
        // MoE offset of layer i < attention offset of layer i+1
        for i in 0..cfg.num_hidden_layers - 1 {
            assert!(
                layout.moe_offsets[i] < layout.attention_offsets[i + 1],
                "layer {i}: MoE offset must precede next layer's attention offset"
            );
        }
        // Last MoE offset < final_norm < lm_head < total
        let last = cfg.num_hidden_layers - 1;
        assert!(layout.moe_offsets[last] < layout.final_norm_offset);
        assert!(layout.final_norm_offset < layout.lm_head_offset);
        assert!(layout.lm_head_offset < layout.total_weight_bytes);
    }

    #[test]
    fn test_buffer_layout_16_byte_aligned() {
        let cfg = cfg_20b();
        let layout = compute_buffer_layout(&cfg, DType::F32).unwrap();
        for &off in &layout.attention_offsets {
            assert_eq!(off % 16, 0, "attention offset must be 16-byte aligned");
        }
        for &off in &layout.moe_offsets {
            assert_eq!(off % 16, 0, "MoE offset must be 16-byte aligned");
        }
        assert_eq!(layout.final_norm_offset % 16, 0);
        assert_eq!(layout.lm_head_offset % 16, 0);
        assert_eq!(layout.total_weight_bytes % 16, 0);
    }

    #[test]
    fn test_buffer_layout_bf16_half_f32() {
        let cfg = cfg_20b();
        let f32_layout = compute_buffer_layout(&cfg, DType::F32).unwrap();
        let bf16_layout = compute_buffer_layout(&cfg, DType::BF16).unwrap();
        // BF16 total should be approximately half of F32 (not exact due to alignment)
        let ratio = f32_layout.total_weight_bytes as f64 / bf16_layout.total_weight_bytes as f64;
        assert!(
            ratio > 1.9 && ratio < 2.1,
            "F32/BF16 ratio should be ~2.0, got {ratio:.3}"
        );
    }

    #[test]
    fn test_buffer_layout_small_config() {
        let cfg = small_cfg();
        let layout = compute_buffer_layout(&cfg, DType::F32).unwrap();
        assert_eq!(layout.attention_offsets.len(), 2);
        assert_eq!(layout.moe_offsets.len(), 2);
        assert!(layout.total_weight_bytes > 0);
    }

    #[test]
    fn test_align_up_basic() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 16);
        assert_eq!(align_up(15), 16);
        assert_eq!(align_up(16), 16);
        assert_eq!(align_up(17), 32);
    }
}
