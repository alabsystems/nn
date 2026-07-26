// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GGUF weight loading for gpt-oss-20b.
//!
//! Maps GGUF tensor names (llama.cpp convention) to the HuggingFace-style
//! names used by [`GptOssModel::load`] and [`GptOssQuantizedModel`].
//!
//! GGUF naming convention (llama.cpp):
//! ```text
//! token_embd.weight                          -> model.embed_tokens.weight
//! blk.{i}.attn_norm.weight                   -> model.layers.{i}.input_layernorm.weight
//! blk.{i}.attn_q.weight / .bias              -> model.layers.{i}.self_attn.q_proj.weight / .bias
//! blk.{i}.attn_k.weight / .bias              -> model.layers.{i}.self_attn.k_proj.weight / .bias
//! blk.{i}.attn_v.weight / .bias              -> model.layers.{i}.self_attn.v_proj.weight / .bias
//! blk.{i}.attn_output.weight / .bias         -> model.layers.{i}.self_attn.o_proj.weight / .bias
//! blk.{i}.attn_sinks                         -> model.layers.{i}.self_attn.sinks
//! blk.{i}.ffn_norm.weight                    -> model.layers.{i}.post_attention_layernorm.weight
//! blk.{i}.ffn_router.weight / .bias          -> model.layers.{i}.mlp.router.weight / .bias
//! blk.{i}.ffn_experts_gate_up                -> model.layers.{i}.mlp.experts.gate_up_proj
//! blk.{i}.ffn_experts_gate_up_bias           -> model.layers.{i}.mlp.experts.gate_up_proj_bias
//! blk.{i}.ffn_experts_down                   -> model.layers.{i}.mlp.experts.down_proj
//! blk.{i}.ffn_experts_down_bias              -> model.layers.{i}.mlp.experts.down_proj_bias
//! output_norm.weight                         -> model.norm.weight
//! output.weight                              -> lm_head.weight
//! ```

use crate::config::{GptOssConfig, LayerType};
use crate::GptOssError;
use nn_core::Result;

/// Weight name mapping entry: GGUF tensor name prefix → HuggingFace-style
/// name prefix. The `{i}` placeholder in both sides is resolved at runtime.
const LAYER_MAPPINGS: &[(&str, &str)] = &[
    ("attn_norm.weight", "input_layernorm.weight"),
    ("attn_q.weight", "self_attn.q_proj.weight"),
    ("attn_q.bias", "self_attn.q_proj.bias"),
    ("attn_k.weight", "self_attn.k_proj.weight"),
    ("attn_k.bias", "self_attn.k_proj.bias"),
    ("attn_v.weight", "self_attn.v_proj.weight"),
    ("attn_v.bias", "self_attn.v_proj.bias"),
    ("attn_output.weight", "self_attn.o_proj.weight"),
    ("attn_output.bias", "self_attn.o_proj.bias"),
    ("attn_sinks", "self_attn.sinks"),
    ("ffn_norm.weight", "post_attention_layernorm.weight"),
    ("ffn_router.weight", "mlp.router.weight"),
    ("ffn_router.bias", "mlp.router.bias"),
    ("ffn_experts_gate_up", "mlp.experts.gate_up_proj"),
    ("ffn_experts_gate_up_bias", "mlp.experts.gate_up_proj_bias"),
    ("ffn_experts_down", "mlp.experts.down_proj"),
    ("ffn_experts_down_bias", "mlp.experts.down_proj_bias"),
];

/// Global (non-layer) name mappings.
const GLOBAL_MAPPINGS: &[(&str, &str)] = &[
    ("token_embd.weight", "model.embed_tokens.weight"),
    ("output_norm.weight", "model.norm.weight"),
    ("output.weight", "lm_head.weight"),
];

/// Maps a GGUF tensor name to the HuggingFace-style name used by
/// `GptOssModel::load` / VarBuilder.
///
/// Returns `None` if the name does not match any known pattern.
pub(crate) fn map_tensor_name(gguf_name: &str) -> Option<String> {
    // Try global mappings first.
    for &(gguf_prefix, hf_name) in GLOBAL_MAPPINGS {
        if gguf_name == gguf_prefix {
            return Some(hf_name.to_string());
        }
    }

    // Try layer-scoped mappings: "blk.{i}.suffix" -> "model.layers.{i}.suffix"
    let rest = gguf_name.strip_prefix("blk.")?;
    let dot_pos = rest.find('.')?;
    let layer_str = &rest[..dot_pos];
    let suffix = &rest[dot_pos + 1..];

    // Validate that layer_str is a valid non-negative integer.
    let _layer_idx: usize = layer_str.parse().ok()?;

    for &(gguf_suffix, hf_suffix) in LAYER_MAPPINGS {
        if suffix == gguf_suffix {
            return Some(format!("model.layers.{layer_str}.{hf_suffix}"));
        }
    }

    None
}

/// Extracts the layer index from a GGUF tensor name of the form `blk.{i}.xxx`.
///
/// Returns `None` for global tensors or invalid names.
pub(crate) fn extract_layer_index(gguf_name: &str) -> Option<usize> {
    let rest = gguf_name.strip_prefix("blk.")?;
    let dot_pos = rest.find('.')?;
    let layer_str = &rest[..dot_pos];
    layer_str.parse().ok()
}

/// Extracts `GptOssConfig` from GGUF metadata.
///
/// Reads gpt-oss-specific metadata keys (matching the `gptoss.*` namespace
/// convention, with fallbacks to standard `llama.*` keys that llama.cpp
/// uses for compatible architectures).
///
/// # Errors
///
/// Returns an error if required metadata keys are missing or have
/// unexpected types.
pub(crate) fn config_from_gguf_metadata(
    metadata: &dyn GgufMetadataEntries,
) -> Result<GptOssConfig> {
    // Try gptoss.* namespace first, fall back to llama.* for compatibility.
    let hidden_size = require_u32(metadata, "gptoss.embedding_length")
        .or_else(|_| require_u32(metadata, "llama.embedding_length"))
        .map(|v| v as usize)?;

    let num_hidden_layers = require_u32(metadata, "gptoss.block_count")
        .or_else(|_| require_u32(metadata, "llama.block_count"))
        .map(|v| v as usize)?;

    let num_attention_heads = require_u32(metadata, "gptoss.attention.head_count")
        .or_else(|_| require_u32(metadata, "llama.attention.head_count"))
        .map(|v| v as usize)?;

    let num_key_value_heads = get_u32(metadata, "gptoss.attention.head_count_kv")
        .or_else(|| get_u32(metadata, "llama.attention.head_count_kv"))
        .map(|v| v as usize)
        .unwrap_or(num_attention_heads);

    let head_dim = get_u32(metadata, "gptoss.attention.head_dim")
        .map(|v| v as usize)
        .unwrap_or(64);

    let vocab_size = get_u32(metadata, "gptoss.vocab_size")
        .or_else(|| get_u32(metadata, "llama.vocab_size"))
        .map(|v| v as usize)
        .unwrap_or(201_088);

    let intermediate_size = get_u32(metadata, "gptoss.feed_forward_length")
        .or_else(|| get_u32(metadata, "llama.feed_forward_length"))
        .map(|v| v as usize)
        .unwrap_or(hidden_size);

    let rms_norm_eps = get_f32(metadata, "gptoss.attention.layer_norm_rms_epsilon")
        .or_else(|| get_f32(metadata, "llama.attention.layer_norm_rms_epsilon"))
        .map(f64::from)
        .unwrap_or(1e-5);

    let rope_theta = get_f32(metadata, "gptoss.rope.freq_base")
        .or_else(|| get_f32(metadata, "llama.rope.freq_base"))
        .map(f64::from)
        .unwrap_or(150_000.0);

    let max_position_embeddings = get_u32(metadata, "gptoss.context_length")
        .or_else(|| get_u32(metadata, "llama.context_length"))
        .map(|v| v as usize)
        .unwrap_or(131_072);

    let num_local_experts = get_u32(metadata, "gptoss.expert_count")
        .or_else(|| get_u32(metadata, "llama.expert_count"))
        .map(|v| v as usize)
        .unwrap_or(32);

    let experts_per_token = get_u32(metadata, "gptoss.expert_used_count")
        .or_else(|| get_u32(metadata, "llama.expert_used_count"))
        .map(|v| v as usize)
        .unwrap_or(4);

    let sliding_window = get_u32(metadata, "gptoss.attention.sliding_window")
        .map(|v| v as usize)
        .unwrap_or(128);

    let eos_token_id = get_u32(metadata, "gptoss.eos_token_id")
        .map(|v| v as usize)
        .unwrap_or(200_002);

    let swiglu_limit = get_f32(metadata, "gptoss.swiglu_limit")
        .map(f64::from)
        .unwrap_or(7.0);

    let attention_bias = get_bool(metadata, "gptoss.attention.bias").unwrap_or(true);

    let tie_word_embeddings = get_bool(metadata, "gptoss.tie_word_embeddings").unwrap_or(false);

    let layer_types: Vec<LayerType> = (0..num_hidden_layers)
        .map(|i| {
            if i % 2 == 0 {
                LayerType::SlidingAttention
            } else {
                LayerType::FullAttention
            }
        })
        .collect();

    let cfg = GptOssConfig::new(
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        vocab_size,
        rms_norm_eps,
        rope_theta,
        max_position_embeddings,
        tie_word_embeddings,
        None, // rope_scaling: use default YaRN from gptoss_20b for now
        attention_bias,
        num_local_experts,
        experts_per_token,
        swiglu_limit,
        layer_types,
        sliding_window,
        eos_token_id,
    );

    cfg.validate()?;
    Ok(cfg)
}

/// Struct that maps GGUF tensor names to gpt-oss weight names.
///
/// Provides the full tensor name mapping for a gpt-oss model and utilities
/// for validating that a GGUF file contains the expected set of tensors.
pub(crate) struct GgufWeightMapper {
    num_layers: usize,
}

impl GgufWeightMapper {
    /// Create a mapper for a model with the given number of layers.
    pub(crate) fn new(num_layers: usize) -> Self {
        Self { num_layers }
    }

    /// Returns the expected number of weight tensors for this model.
    ///
    /// Per layer: 17 tensors (attn_norm, q/k/v/o weight+bias, sinks,
    /// ffn_norm, router weight+bias, gate_up+bias, down+bias).
    /// Global: 3 tensors (token_embd, output_norm, output).
    pub(crate) fn expected_weight_count(&self) -> usize {
        let per_layer = LAYER_MAPPINGS.len(); // 17
        let global = GLOBAL_MAPPINGS.len(); // 3
        global + per_layer * self.num_layers
    }

    /// Returns all expected GGUF tensor names for this model.
    pub(crate) fn expected_gguf_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.expected_weight_count());
        for &(gguf_name, _) in GLOBAL_MAPPINGS {
            names.push(gguf_name.to_string());
        }
        for layer in 0..self.num_layers {
            for &(gguf_suffix, _) in LAYER_MAPPINGS {
                names.push(format!("blk.{layer}.{gguf_suffix}"));
            }
        }
        names
    }

    /// Check which expected tensors are missing from a set of GGUF tensor names.
    pub(crate) fn find_missing(&self, available: &[&str]) -> Vec<String> {
        let expected = self.expected_gguf_names();
        expected
            .into_iter()
            .filter(|name| !available.contains(&name.as_str()))
            .collect()
    }
}

// -- Metadata helpers ---------------------------------------------------------

/// Trait abstracting over GGUF metadata access.
///
/// This allows testing without depending on nn-gguf types directly.
pub(crate) trait GgufMetadataEntries {
    fn get_u32(&self, key: &str) -> Option<u32>;
    fn get_f32(&self, key: &str) -> Option<f32>;
    fn get_bool(&self, key: &str) -> Option<bool>;
    fn get_str(&self, key: &str) -> Option<&str>;
}

/// Simple in-memory metadata store for testing.
#[cfg(test)]
pub(crate) struct TestMetadata {
    pub(crate) u32s: std::collections::HashMap<String, u32>,
    pub(crate) f32s: std::collections::HashMap<String, f32>,
    pub(crate) bools: std::collections::HashMap<String, bool>,
}

#[cfg(test)]
impl TestMetadata {
    pub(crate) fn new() -> Self {
        Self {
            u32s: std::collections::HashMap::new(),
            f32s: std::collections::HashMap::new(),
            bools: std::collections::HashMap::new(),
        }
    }

    pub(crate) fn gptoss_20b_defaults() -> Self {
        let mut m = Self::new();
        m.u32s.insert("gptoss.embedding_length".into(), 2880);
        m.u32s.insert("gptoss.block_count".into(), 24);
        m.u32s.insert("gptoss.attention.head_count".into(), 64);
        m.u32s.insert("gptoss.attention.head_count_kv".into(), 8);
        m.u32s.insert("gptoss.attention.head_dim".into(), 64);
        m.u32s.insert("gptoss.vocab_size".into(), 201_088);
        m.u32s.insert("gptoss.feed_forward_length".into(), 2880);
        m.u32s.insert("gptoss.context_length".into(), 131_072);
        m.u32s.insert("gptoss.expert_count".into(), 32);
        m.u32s.insert("gptoss.expert_used_count".into(), 4);
        m.u32s.insert("gptoss.attention.sliding_window".into(), 128);
        m.u32s.insert("gptoss.eos_token_id".into(), 200_002);
        m.f32s.insert("gptoss.rope.freq_base".into(), 150_000.0);
        m.f32s
            .insert("gptoss.attention.layer_norm_rms_epsilon".into(), 1e-5);
        m.f32s.insert("gptoss.swiglu_limit".into(), 7.0);
        m.bools.insert("gptoss.attention.bias".into(), true);
        m.bools.insert("gptoss.tie_word_embeddings".into(), false);
        m
    }
}

#[cfg(test)]
impl GgufMetadataEntries for TestMetadata {
    fn get_u32(&self, key: &str) -> Option<u32> {
        self.u32s.get(key).copied()
    }
    fn get_f32(&self, key: &str) -> Option<f32> {
        self.f32s.get(key).copied()
    }
    fn get_bool(&self, key: &str) -> Option<bool> {
        self.bools.get(key).copied()
    }
    fn get_str(&self, _key: &str) -> Option<&str> {
        None
    }
}

fn require_u32(metadata: &dyn GgufMetadataEntries, key: &str) -> Result<u32> {
    metadata.get_u32(key).ok_or_else(|| {
        GptOssError::WeightLoad {
            reason: format!("missing required GGUF metadata key: {key}"),
        }
        .into()
    })
}

fn get_u32(metadata: &dyn GgufMetadataEntries, key: &str) -> Option<u32> {
    metadata.get_u32(key)
}

fn get_f32(metadata: &dyn GgufMetadataEntries, key: &str) -> Option<f32> {
    metadata.get_f32(key)
}

fn get_bool(metadata: &dyn GgufMetadataEntries, key: &str) -> Option<bool> {
    metadata.get_bool(key)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Name mapping tests ----

    #[test]
    fn test_map_global_token_embd() {
        assert_eq!(
            map_tensor_name("token_embd.weight"),
            Some("model.embed_tokens.weight".to_string())
        );
    }

    #[test]
    fn test_map_global_output_norm() {
        assert_eq!(
            map_tensor_name("output_norm.weight"),
            Some("model.norm.weight".to_string())
        );
    }

    #[test]
    fn test_map_global_output_weight() {
        assert_eq!(
            map_tensor_name("output.weight"),
            Some("lm_head.weight".to_string())
        );
    }

    #[test]
    fn test_map_layer_attn_q_weight() {
        assert_eq!(
            map_tensor_name("blk.0.attn_q.weight"),
            Some("model.layers.0.self_attn.q_proj.weight".to_string())
        );
    }

    #[test]
    fn test_map_layer_attn_k_bias() {
        assert_eq!(
            map_tensor_name("blk.12.attn_k.bias"),
            Some("model.layers.12.self_attn.k_proj.bias".to_string())
        );
    }

    #[test]
    fn test_map_layer_ffn_norm() {
        assert_eq!(
            map_tensor_name("blk.23.ffn_norm.weight"),
            Some("model.layers.23.post_attention_layernorm.weight".to_string())
        );
    }

    #[test]
    fn test_map_layer_ffn_router() {
        assert_eq!(
            map_tensor_name("blk.5.ffn_router.weight"),
            Some("model.layers.5.mlp.router.weight".to_string())
        );
    }

    #[test]
    fn test_map_layer_experts_gate_up() {
        assert_eq!(
            map_tensor_name("blk.3.ffn_experts_gate_up"),
            Some("model.layers.3.mlp.experts.gate_up_proj".to_string())
        );
    }

    #[test]
    fn test_map_layer_experts_down_bias() {
        assert_eq!(
            map_tensor_name("blk.10.ffn_experts_down_bias"),
            Some("model.layers.10.mlp.experts.down_proj_bias".to_string())
        );
    }

    #[test]
    fn test_map_layer_attn_sinks() {
        assert_eq!(
            map_tensor_name("blk.7.attn_sinks"),
            Some("model.layers.7.self_attn.sinks".to_string())
        );
    }

    #[test]
    fn test_map_unknown_returns_none() {
        assert_eq!(map_tensor_name("blk.0.unknown_layer"), None);
    }

    #[test]
    fn test_map_invalid_format_returns_none() {
        assert_eq!(map_tensor_name("no_dot_prefix"), None);
    }

    #[test]
    fn test_map_invalid_layer_index() {
        assert_eq!(map_tensor_name("blk.abc.attn_q.weight"), None);
    }

    // ---- Layer index extraction ----

    #[test]
    fn test_extract_layer_index_valid() {
        assert_eq!(extract_layer_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(extract_layer_index("blk.23.ffn_norm.weight"), Some(23));
    }

    #[test]
    fn test_extract_layer_index_global() {
        assert_eq!(extract_layer_index("token_embd.weight"), None);
        assert_eq!(extract_layer_index("output.weight"), None);
    }

    // ---- Name mapping preserves layer index ----

    #[test]
    fn test_mapping_preserves_layer_index() {
        for layer in [0, 5, 12, 23] {
            let gguf_name = format!("blk.{layer}.attn_q.weight");
            let hf_name = map_tensor_name(&gguf_name).unwrap();
            // Verify the mapped name contains the same layer index
            let expected_prefix = format!("model.layers.{layer}.");
            assert!(
                hf_name.starts_with(&expected_prefix),
                "mapped name '{hf_name}' should start with '{expected_prefix}'"
            );
        }
    }

    // ---- Weight mapper ----

    #[test]
    fn test_mapper_expected_weight_count() {
        let mapper = GgufWeightMapper::new(24);
        // 3 global + 17 per-layer * 24 layers = 3 + 408 = 411
        assert_eq!(mapper.expected_weight_count(), 3 + 17 * 24);
    }

    #[test]
    fn test_mapper_expected_weight_count_single_layer() {
        let mapper = GgufWeightMapper::new(1);
        assert_eq!(mapper.expected_weight_count(), 3 + 17);
    }

    #[test]
    fn test_mapper_find_missing_all() {
        let mapper = GgufWeightMapper::new(1);
        let missing = mapper.find_missing(&[]);
        assert_eq!(missing.len(), mapper.expected_weight_count());
    }

    #[test]
    fn test_mapper_find_missing_partial() {
        let mapper = GgufWeightMapper::new(1);
        let available = vec!["token_embd.weight", "output_norm.weight", "output.weight"];
        let missing = mapper.find_missing(&available);
        // Should be missing all 17 layer tensors
        assert_eq!(missing.len(), 17);
        assert!(missing.iter().all(|n| n.starts_with("blk.0.")));
    }

    #[test]
    fn test_mapper_find_missing_complete() {
        let mapper = GgufWeightMapper::new(1);
        let all_names = mapper.expected_gguf_names();
        let all_refs: Vec<&str> = all_names.iter().map(String::as_str).collect();
        let missing = mapper.find_missing(&all_refs);
        assert!(missing.is_empty(), "no tensors should be missing");
    }

    // ---- Config extraction ----

    #[test]
    fn test_config_from_metadata_defaults() {
        let metadata = TestMetadata::gptoss_20b_defaults();
        let cfg = config_from_gguf_metadata(&metadata).unwrap();
        assert_eq!(cfg.hidden_size, 2880);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.vocab_size, 201_088);
        assert_eq!(cfg.num_local_experts, 32);
        assert_eq!(cfg.experts_per_token, 4);
        assert!(cfg.attention_bias);
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn test_config_from_metadata_validates() {
        let metadata = TestMetadata::gptoss_20b_defaults();
        let cfg = config_from_gguf_metadata(&metadata).unwrap();
        cfg.validate().expect("extracted config should validate");
    }

    #[test]
    fn test_config_from_metadata_missing_required() {
        let metadata = TestMetadata::new();
        let result = config_from_gguf_metadata(&metadata);
        assert!(result.is_err(), "should fail with missing required keys");
    }

    #[test]
    fn test_config_from_metadata_llama_fallback() {
        let mut metadata = TestMetadata::new();
        // Use llama.* keys instead of gptoss.*
        metadata.u32s.insert("llama.embedding_length".into(), 2880);
        metadata.u32s.insert("llama.block_count".into(), 24);
        metadata
            .u32s
            .insert("llama.attention.head_count".into(), 64);
        metadata
            .u32s
            .insert("llama.attention.head_count_kv".into(), 8);
        metadata.u32s.insert("llama.vocab_size".into(), 201_088);
        metadata
            .u32s
            .insert("llama.feed_forward_length".into(), 2880);
        metadata.u32s.insert("llama.expert_count".into(), 32);
        metadata.u32s.insert("llama.expert_used_count".into(), 4);
        let cfg = config_from_gguf_metadata(&metadata).unwrap();
        assert_eq!(cfg.hidden_size, 2880);
        assert_eq!(cfg.num_hidden_layers, 24);
    }

    // ---- All layer mappings round-trip ----

    #[test]
    fn test_all_layer_mappings_resolve() {
        for &(gguf_suffix, _hf_suffix) in LAYER_MAPPINGS {
            let gguf_name = format!("blk.0.{gguf_suffix}");
            let mapped = map_tensor_name(&gguf_name);
            assert!(
                mapped.is_some(),
                "layer mapping for '{gguf_suffix}' should resolve"
            );
        }
    }

    #[test]
    fn test_all_global_mappings_resolve() {
        for &(gguf_name, expected_hf) in GLOBAL_MAPPINGS {
            let mapped = map_tensor_name(gguf_name);
            assert_eq!(
                mapped,
                Some(expected_hf.to_string()),
                "global mapping for '{gguf_name}' should resolve to '{expected_hf}'"
            );
        }
    }
}
