// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Benchmarking and profiling infrastructure for gpt-oss-20b inference.
//!
//! Provides memory estimation functions and benchmark result types for
//! measuring inference performance (tokens/sec, latency, memory) on
//! M4 Max and other hardware targets.
//!
//! Memory estimates use `checked_mul` to prevent silent overflow on large
//! model configurations.

use crate::config::GptOssConfig;
use nn_core::DType;

/// Configuration for running an inference benchmark.
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    /// Prompt sequence length (number of input tokens).
    pub seq_len: usize,
    /// Number of decode iterations to measure (excludes warmup).
    pub num_iterations: usize,
    /// Warmup iterations (not included in timing).
    pub warmup_iterations: usize,
    /// Weight dtype for memory estimation.
    pub dtype: DType,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            seq_len: 128,
            num_iterations: 100,
            warmup_iterations: 10,
            dtype: DType::BF16,
        }
    }
}

/// Results from an inference benchmark run.
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    /// Tokens generated per second (decode phase).
    pub tokens_per_second: f64,
    /// Time to first token in milliseconds (prompt processing).
    pub time_to_first_token_ms: f64,
    /// Mean per-token latency in milliseconds.
    pub mean_token_latency_ms: f64,
    /// 99th percentile per-token latency in milliseconds.
    pub p99_latency_ms: f64,
    /// Estimated peak memory usage in bytes.
    pub memory_bytes: usize,
}

impl BenchmarkResult {
    /// Compute tokens per second from total tokens and elapsed seconds.
    ///
    /// Returns 0.0 if elapsed is not positive and finite.
    #[must_use]
    pub fn compute_tps(num_tokens: usize, elapsed_secs: f64) -> f64 {
        if !elapsed_secs.is_finite() || elapsed_secs <= 0.0 {
            return 0.0;
        }
        num_tokens as f64 / elapsed_secs
    }
}

// -- Memory estimation --------------------------------------------------------

/// Estimate total model weight memory for a given config and dtype.
///
/// Accounts for:
/// - Embedding: `vocab_size * hidden_size` elements
/// - Per-layer attention: Q/K/V/O projections + biases + layernorms + sinks
/// - Per-layer MoE: router + expert gate_up_proj + down_proj + biases
/// - Final norm + lm_head
///
/// Returns `None` if arithmetic overflows.
#[must_use]
pub fn estimate_model_memory(cfg: &GptOssConfig, dtype: DType) -> Option<usize> {
    let bpe = dtype.size_bytes();
    let h = cfg.hidden_size;
    let attn_dim = cfg.attn_dim(); // num_attention_heads * head_dim
    let kv_dim = cfg.kv_dim(); // num_key_value_heads * head_dim
    let ne = cfg.num_local_experts;
    let inter = cfg.intermediate_size;
    let fused_dim = 2_usize.checked_mul(inter)?;

    // Embedding: vocab_size * hidden_size
    let embed_elems = cfg.vocab_size.checked_mul(h)?;

    // Per-layer attention weights:
    //   Q: [attn_dim, h] + bias [attn_dim]
    //   K: [kv_dim, h] + bias [kv_dim]
    //   V: [kv_dim, h] + bias [kv_dim]
    //   O: [h, attn_dim] + bias [h]
    //   sinks: [head_dim]
    //   input_layernorm: [h]
    //   post_attention_layernorm: [h]
    let q_elems = attn_dim.checked_mul(h)?.checked_add(attn_dim)?;
    let k_elems = kv_dim.checked_mul(h)?.checked_add(kv_dim)?;
    let v_elems = kv_dim.checked_mul(h)?.checked_add(kv_dim)?;
    let o_elems = h.checked_mul(attn_dim)?.checked_add(h)?;
    let attn_elems = q_elems
        .checked_add(k_elems)?
        .checked_add(v_elems)?
        .checked_add(o_elems)?
        .checked_add(cfg.head_dim)? // sinks
        .checked_add(h)? // input_layernorm
        .checked_add(h)?; // post_attention_layernorm

    // Per-layer MoE weights:
    //   router: [ne, h] + bias [ne]
    //   gate_up_proj: [ne, h, fused_dim] + bias [ne, fused_dim]
    //   down_proj: [ne, h, h] + bias [ne, h]
    let router_elems = ne.checked_mul(h)?.checked_add(ne)?;
    let gate_up_elems = ne
        .checked_mul(h)?
        .checked_mul(fused_dim)?
        .checked_add(ne.checked_mul(fused_dim)?)?;
    let down_elems = ne
        .checked_mul(h)?
        .checked_mul(h)?
        .checked_add(ne.checked_mul(h)?)?;
    let moe_elems = router_elems
        .checked_add(gate_up_elems)?
        .checked_add(down_elems)?;

    let per_layer_elems = attn_elems.checked_add(moe_elems)?;
    let all_layers_elems = per_layer_elems.checked_mul(cfg.num_hidden_layers)?;

    // Final norm [h] + lm_head [vocab_size, h]
    let final_norm_elems = h;
    let lm_head_elems = if cfg.tie_word_embeddings {
        0 // shares embedding weight
    } else {
        cfg.vocab_size.checked_mul(h)?
    };

    let total_elems = embed_elems
        .checked_add(all_layers_elems)?
        .checked_add(final_norm_elems)?
        .checked_add(lm_head_elems)?;

    total_elems.checked_mul(bpe)
}

/// Estimate KV cache memory for a given sequence length (bytes).
///
/// Each layer stores K and V tensors:
///   K: `[1, num_kv_heads, seq_len, head_dim]`
///   V: `[1, num_kv_heads, seq_len, head_dim]`
///
/// Sliding attention layers are capped at `min(seq_len, sliding_window)`.
/// Assumes F32 storage (4 bytes per element).
///
/// Returns `None` if arithmetic overflows.
#[must_use]
pub fn estimate_kv_cache_memory(cfg: &GptOssConfig, seq_len: usize) -> Option<usize> {
    let bpe = 4_usize; // F32 KV cache storage
    let kv_dim = cfg.kv_dim();

    let mut total: usize = 0;
    for lt in &cfg.layer_types {
        let effective_seq = match lt {
            crate::config::LayerType::SlidingAttention => seq_len.min(cfg.sliding_window),
            crate::config::LayerType::FullAttention => seq_len,
        };
        // K + V per layer: 2 * kv_dim * effective_seq * bpe
        let layer_bytes = 2_usize
            .checked_mul(kv_dim)?
            .checked_mul(effective_seq)?
            .checked_mul(bpe)?;
        total = total.checked_add(layer_bytes)?;
    }
    Some(total)
}

/// Estimate memory when MoE expert weights are MXFP4-quantized (bytes).
///
/// MXFP4 stores 32 values in 17 bytes (16 bytes packed FP4 + 1 byte E8M0 scale).
/// Non-expert weights (attention, norms, embeddings) stay at the given dtype.
///
/// Returns `None` if arithmetic overflows.
#[must_use]
pub fn estimate_mxfp4_memory(cfg: &GptOssConfig) -> Option<usize> {
    let h = cfg.hidden_size;
    let ne = cfg.num_local_experts;
    let inter = cfg.intermediate_size;
    let fused_dim = 2_usize.checked_mul(inter)?;

    // Expert weight elements per layer:
    //   gate_up_proj: ne * h * fused_dim
    //   down_proj: ne * h * h
    let gate_up_numel = ne.checked_mul(h)?.checked_mul(fused_dim)?;
    let down_numel = ne.checked_mul(h)?.checked_mul(h)?;
    let expert_numel_per_layer = gate_up_numel.checked_add(down_numel)?;
    let total_expert_numel = expert_numel_per_layer.checked_mul(cfg.num_hidden_layers)?;

    // MXFP4: 17 bytes per 32 elements
    let num_blocks = total_expert_numel.div_ceil(32);
    let mxfp4_bytes = num_blocks.checked_mul(17)?;

    // Non-expert weights: compute total elements, then multiply by BF16 (2 bytes)
    // as the typical non-quantized dtype for a quantized model.
    let non_expert_bpe = 2_usize; // BF16

    // Embedding
    let embed_elems = cfg.vocab_size.checked_mul(h)?;
    // lm_head (not tied in gpt-oss-20b)
    let lm_head_elems = if cfg.tie_word_embeddings {
        0
    } else {
        cfg.vocab_size.checked_mul(h)?
    };
    // Final norm
    let final_norm_elems = h;

    // Per-layer non-expert weights
    let attn_dim = cfg.attn_dim();
    let kv_dim = cfg.kv_dim();
    let q_elems = attn_dim.checked_mul(h)?.checked_add(attn_dim)?;
    let k_elems = kv_dim.checked_mul(h)?.checked_add(kv_dim)?;
    let v_elems = kv_dim.checked_mul(h)?.checked_add(kv_dim)?;
    let o_elems = h.checked_mul(attn_dim)?.checked_add(h)?;
    let attn_elems = q_elems
        .checked_add(k_elems)?
        .checked_add(v_elems)?
        .checked_add(o_elems)?
        .checked_add(cfg.head_dim)?
        .checked_add(h)?
        .checked_add(h)?;
    // Router: [ne, h] + [ne]
    let router_elems = ne.checked_mul(h)?.checked_add(ne)?;
    // Expert biases stay at full precision:
    //   gate_up_bias: [ne, fused_dim]
    //   down_bias: [ne, h]
    let expert_bias_elems = ne.checked_mul(fused_dim)?.checked_add(ne.checked_mul(h)?)?;
    let per_layer_non_expert = attn_elems
        .checked_add(router_elems)?
        .checked_add(expert_bias_elems)?;
    let all_layers_non_expert = per_layer_non_expert.checked_mul(cfg.num_hidden_layers)?;

    let total_non_expert_elems = embed_elems
        .checked_add(lm_head_elems)?
        .checked_add(final_norm_elems)?
        .checked_add(all_layers_non_expert)?;
    let non_expert_bytes = total_non_expert_elems.checked_mul(non_expert_bpe)?;

    mxfp4_bytes.checked_add(non_expert_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_20b() -> GptOssConfig {
        GptOssConfig::gptoss_20b()
    }

    #[test]
    fn test_benchmark_config_default() {
        let bc = BenchmarkConfig::default();
        assert_eq!(bc.seq_len, 128);
        assert_eq!(bc.num_iterations, 100);
        assert_eq!(bc.warmup_iterations, 10);
        assert_eq!(bc.dtype, DType::BF16);
    }

    #[test]
    fn test_compute_tps_basic() {
        let tps = BenchmarkResult::compute_tps(100, 2.0);
        assert!((tps - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_tps_zero_time() {
        assert_eq!(BenchmarkResult::compute_tps(100, 0.0), 0.0);
    }

    #[test]
    fn test_compute_tps_nan_time() {
        assert_eq!(BenchmarkResult::compute_tps(100, f64::NAN), 0.0);
    }

    #[test]
    fn test_compute_tps_negative_time() {
        assert_eq!(BenchmarkResult::compute_tps(100, -1.0), 0.0);
    }

    #[test]
    fn test_estimate_model_memory_nonzero() {
        let cfg = cfg_20b();
        let mem = estimate_model_memory(&cfg, DType::F32).expect("should not overflow");
        assert!(mem > 0, "model memory must be > 0");
        // 20B params * 4 bytes ~ 80GB; should be in the right ballpark
        let gb = mem as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(gb > 30.0, "expected >30GB for F32, got {gb:.1}GB");
        assert!(gb < 200.0, "expected <200GB for F32, got {gb:.1}GB");
    }

    #[test]
    fn test_estimate_model_memory_bf16_half_f32() {
        let cfg = cfg_20b();
        let f32_mem = estimate_model_memory(&cfg, DType::F32).unwrap();
        let bf16_mem = estimate_model_memory(&cfg, DType::BF16).unwrap();
        assert_eq!(bf16_mem * 2, f32_mem, "BF16 should be exactly half F32");
    }

    #[test]
    fn test_estimate_kv_cache_memory_nonzero() {
        let cfg = cfg_20b();
        let mem = estimate_kv_cache_memory(&cfg, 1024).expect("should not overflow");
        assert!(mem > 0);
    }

    #[test]
    fn test_estimate_kv_cache_memory_zero_seq() {
        let cfg = cfg_20b();
        let mem = estimate_kv_cache_memory(&cfg, 0).expect("should not overflow");
        assert_eq!(mem, 0);
    }

    #[test]
    fn test_estimate_kv_cache_linear_in_seq() {
        let cfg = cfg_20b();
        // At large seq_len (well above sliding_window=128), full attention
        // layers scale linearly while sliding layers saturate.
        let mem_1k = estimate_kv_cache_memory(&cfg, 1024).unwrap();
        let mem_2k = estimate_kv_cache_memory(&cfg, 2048).unwrap();
        // mem_2k should be greater than mem_1k (full attention layers double)
        assert!(mem_2k > mem_1k);
        // But not exactly 2x because sliding layers are capped at 128
        assert!(mem_2k < mem_1k * 2);
    }

    #[test]
    fn test_estimate_mxfp4_less_than_bf16() {
        let cfg = cfg_20b();
        let bf16_mem = estimate_model_memory(&cfg, DType::BF16).unwrap();
        let mxfp4_mem = estimate_mxfp4_memory(&cfg).unwrap();
        assert!(
            mxfp4_mem < bf16_mem,
            "MXFP4 ({mxfp4_mem}) should be less than BF16 ({bf16_mem})"
        );
    }

    #[test]
    fn test_estimate_mxfp4_compression_ratio() {
        let cfg = cfg_20b();
        let bf16_mem = estimate_model_memory(&cfg, DType::BF16).unwrap();
        let mxfp4_mem = estimate_mxfp4_memory(&cfg).unwrap();
        let ratio = bf16_mem as f64 / mxfp4_mem as f64;
        // Expert weights are ~80% of params; MXFP4 is ~3.7x vs BF16 on those.
        // Overall ratio should be between 1.5 and 3.5.
        assert!(
            ratio > 1.5 && ratio < 3.5,
            "compression ratio {ratio:.2}x outside expected range"
        );
    }

    #[test]
    fn test_estimate_model_memory_small_config() {
        let cfg = GptOssConfig::gptoss_20b()
            .with_vocab_size(100)
            .with_num_hidden_layers(2)
            .with_num_local_experts(4)
            .with_experts_per_token(2);
        let mem = estimate_model_memory(&cfg, DType::F32).unwrap();
        assert!(mem > 0);
        // 2 layers with 4 experts at hidden=2880: MoE weights dominate
        // (~800MB for expert weights alone). Full 20b config is ~80GB.
        // This reduced config should be well under 5GB.
        let gb = mem as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(gb < 5.0, "small config should be <5GB, got {gb:.2}GB");
    }
}
