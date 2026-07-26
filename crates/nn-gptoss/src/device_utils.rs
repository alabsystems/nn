// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Device management utilities for gpt-oss model loading and memory reporting.
//!
//! Provides helpers to load safetensors weights onto a specific device/dtype
//! and to report per-component memory usage of a loaded model.

use std::path::Path;

use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device, Result};

use crate::{GptOssConfig, GptOssModel};

/// Load model weights from a safetensors file to a specific device and dtype.
///
/// This is the GPU-aware variant of [`GptOssModel::load_safetensors`], which
/// always loads to CPU with F32. Use this to place weights directly on a Metal
/// (or other) device at the desired precision.
///
/// # Errors
///
/// Returns an error if the safetensors file cannot be read, weights are missing
/// or have unexpected shapes, or the configuration is invalid.
pub(crate) fn load_safetensors_to_device(
    path: impl AsRef<Path>,
    cfg: GptOssConfig,
    device: &Device,
    dtype: DType,
) -> Result<GptOssModel> {
    let tensors = nn_core::load_safetensors(path)?;
    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    GptOssModel::load(&vb, cfg)
}

/// Per-component memory breakdown of a loaded gpt-oss model.
#[derive(Debug, Clone)]
pub(crate) struct MemoryReport {
    /// Total number of scalar parameters across all components.
    pub(crate) total_params: usize,
    /// Total bytes consumed by all parameters (params * bytes_per_element).
    pub(crate) total_bytes: usize,
    /// Bytes consumed by attention weights (Q/K/V/O projections + sinks).
    pub(crate) attention_bytes: usize,
    /// Bytes consumed by MoE weights (router + fused expert projections).
    pub(crate) moe_bytes: usize,
    /// Bytes consumed by embedding and lm_head weights.
    pub(crate) embedding_bytes: usize,
}

impl std::fmt::Display for MemoryReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        write!(
            f,
            "MemoryReport {{ total: {:.1} MB ({} params), \
             attention: {:.1} MB, moe: {:.1} MB, embedding: {:.1} MB }}",
            mb(self.total_bytes),
            self.total_params,
            mb(self.attention_bytes),
            mb(self.moe_bytes),
            mb(self.embedding_bytes),
        )
    }
}

/// Compute a memory report for a loaded gpt-oss model based on its config.
///
/// This is an analytical estimate from the config dimensions; it does not
/// inspect actual tensor allocations (which would require device-specific
/// introspection).
pub(crate) fn model_memory_report(cfg: &GptOssConfig) -> MemoryReport {
    let bpe = 4_usize; // F32 = 4 bytes per element (DynTensor float storage is f32)
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let attn_dim = nh * hd;
    let kv_dim = nkv * hd;
    let ne = cfg.num_local_experts;
    let inter = cfg.intermediate_size;
    let nl = cfg.num_hidden_layers;
    let v = cfg.vocab_size;

    // Per-layer attention: Q [attn_dim, h] + bias [attn_dim]
    //                      K [kv_dim, h] + bias [kv_dim]
    //                      V [kv_dim, h] + bias [kv_dim]
    //                      O [h, attn_dim] + bias [h]
    //                      sinks [hd]
    //                      input_layernorm [h]
    let attn_params_per_layer = attn_dim * h
        + attn_dim
        + kv_dim * h
        + kv_dim
        + kv_dim * h
        + kv_dim
        + h * attn_dim
        + h
        + hd
        + h;

    // Per-layer MoE: router [ne, h] + bias [ne]
    //                gate_up_proj [ne, h, 2*inter] + bias [ne, 2*inter]
    //                down_proj [ne, h, h] + bias [ne, h]
    //                post_attention_layernorm [h]
    let moe_params_per_layer =
        ne * h + ne + ne * h * 2 * inter + ne * 2 * inter + ne * h * h + ne * h + h;

    // Embedding: embed_tokens [v, h] + lm_head [v, h] (or tied) + final norm [h]
    let embedding_params = if cfg.tie_word_embeddings {
        v * h + h
    } else {
        v * h + v * h + h
    };

    let attn_total = attn_params_per_layer * nl;
    let moe_total = moe_params_per_layer * nl;
    let total = attn_total + moe_total + embedding_params;

    MemoryReport {
        total_params: total,
        total_bytes: total * bpe,
        attention_bytes: attn_total * bpe,
        moe_bytes: moe_total * bpe,
        embedding_bytes: embedding_params * bpe,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_report_nonzero() {
        let cfg = GptOssConfig::gptoss_20b();
        let report = model_memory_report(&cfg);
        assert!(report.total_params > 0);
        assert!(report.total_bytes > 0);
        assert!(report.attention_bytes > 0);
        assert!(report.moe_bytes > 0);
        assert!(report.embedding_bytes > 0);
        // Total should be the sum of components
        assert_eq!(
            report.total_bytes,
            report.attention_bytes + report.moe_bytes + report.embedding_bytes
        );
    }

    #[test]
    fn test_memory_report_moe_dominates() {
        let cfg = GptOssConfig::gptoss_20b();
        let report = model_memory_report(&cfg);
        // MoE (32 experts per layer) should consume more memory than attention
        assert!(
            report.moe_bytes > report.attention_bytes,
            "MoE should dominate: moe={} vs attn={}",
            report.moe_bytes,
            report.attention_bytes
        );
    }

    #[test]
    fn test_memory_report_display() {
        let cfg = GptOssConfig::gptoss_20b();
        let report = model_memory_report(&cfg);
        let s = format!("{report}");
        assert!(s.contains("MemoryReport"));
        assert!(s.contains("MB"));
    }

    #[test]
    fn test_memory_report_tied_embeddings() {
        let cfg = GptOssConfig::gptoss_20b();
        let report_untied = model_memory_report(&cfg);

        let cfg_tied = GptOssConfig::gptoss_20b();
        // Default is untied; modify to tied
        let mut cfg_tied = cfg_tied;
        cfg_tied.tie_word_embeddings = true;
        let report_tied = model_memory_report(&cfg_tied);

        // Tied should use less embedding memory
        assert!(
            report_tied.embedding_bytes < report_untied.embedding_bytes,
            "tied={} should be less than untied={}",
            report_tied.embedding_bytes,
            report_untied.embedding_bytes,
        );
    }
}
