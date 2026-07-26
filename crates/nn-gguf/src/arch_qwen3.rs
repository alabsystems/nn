// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Qwen3 architecture builder for GGUF models.
//!
//! Maps GGUF tensor names (llama.cpp convention) to HuggingFace Qwen3 weight
//! names and builds a `HashMap<String, DynTensor>` suitable for
//! `VarBuilder::from_tensors()`.
//!
//! GGUF tensor name → HuggingFace name mapping:
//!
//! | GGUF name                        | HuggingFace name                                    |
//! |----------------------------------|-----------------------------------------------------|
//! | `token_embd.weight`              | `model.embed_tokens.weight`                         |
//! | `blk.{i}.attn_norm.weight`       | `model.layers.{i}.input_layernorm.weight`           |
//! | `blk.{i}.attn_q.weight`          | `model.layers.{i}.self_attn.q_proj.weight`          |
//! | `blk.{i}.attn_k.weight`          | `model.layers.{i}.self_attn.k_proj.weight`          |
//! | `blk.{i}.attn_v.weight`          | `model.layers.{i}.self_attn.v_proj.weight`          |
//! | `blk.{i}.attn_output.weight`     | `model.layers.{i}.self_attn.o_proj.weight`          |
//! | `blk.{i}.attn_q_norm.weight`     | `model.layers.{i}.self_attn.q_norm.weight`          |
//! | `blk.{i}.attn_k_norm.weight`     | `model.layers.{i}.self_attn.k_norm.weight`          |
//! | `blk.{i}.ffn_norm.weight`        | `model.layers.{i}.post_attention_layernorm.weight`  |
//! | `blk.{i}.ffn_gate.weight`        | `model.layers.{i}.mlp.gate_proj.weight`             |
//! | `blk.{i}.ffn_up.weight`          | `model.layers.{i}.mlp.up_proj.weight`               |
//! | `blk.{i}.ffn_down.weight`        | `model.layers.{i}.mlp.down_proj.weight`             |
//! | `output_norm.weight`             | `model.norm.weight`                                 |
//! | `output.weight`                  | `lm_head.weight`                                    |

use std::collections::HashMap;
use std::io::{Read, Seek};

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::error::GgufError;
use crate::reader::GgufFile;

/// Qwen3 model configuration extracted from GGUF metadata.
///
/// GGUF files for Qwen3 use the `qwen3` or `qwen2` architecture tag
/// (llama.cpp treats Qwen3 as a Qwen2 variant with QK-norm).
#[derive(Debug, Clone)]
pub struct Qwen3GgufConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Qwen3GgufConfig {
    /// Extract Qwen3 config from GGUF metadata.
    ///
    /// Reads standard `qwen2.*` / `qwen3.*` metadata keys. Falls back to
    /// deriving `vocab_size` from the `token_embd.weight` tensor shape when
    /// the metadata key is absent.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, GgufError> {
        // Qwen3 GGUF files may use "qwen2" or "qwen3" as architecture.
        if let Some(arch) = file.architecture() {
            if arch != "qwen2" && arch != "qwen3" {
                return Err(GgufError::ArchitectureMismatch {
                    expected: "qwen2 or qwen3".to_string(),
                    found: arch.to_string(),
                });
            }
        }

        // Try qwen2.* keys first (llama.cpp standard), then qwen3.*.
        let prefix = if file.metadata.get_u32("qwen2.embedding_length").is_some() {
            "qwen2"
        } else {
            "qwen3"
        };

        let hidden_size =
            require_u32_meta(&file.metadata, &format!("{prefix}.embedding_length"))? as usize;
        let num_hidden_layers =
            require_u32_meta(&file.metadata, &format!("{prefix}.block_count"))? as usize;
        let num_attention_heads =
            require_u32_meta(&file.metadata, &format!("{prefix}.attention.head_count"))? as usize;
        let num_key_value_heads = file
            .metadata
            .get_u32(&format!("{prefix}.attention.head_count_kv"))
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);

        let head_dim = hidden_size / num_attention_heads;

        let intermediate_size =
            require_u32_meta(&file.metadata, &format!("{prefix}.feed_forward_length"))? as usize;

        let rms_norm_eps = file
            .metadata
            .get(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .and_then(super::metadata::GgufMetadataValue::as_f32)
            .unwrap_or(1e-6);

        let rope_theta = file
            .metadata
            .get(&format!("{prefix}.rope.freq_base"))
            .and_then(super::metadata::GgufMetadataValue::as_f32)
            .unwrap_or(1_000_000.0);

        let max_position_embeddings = file
            .metadata
            .get_u32(&format!("{prefix}.context_length"))
            .map(|v| v as usize)
            .unwrap_or(32768);

        let vocab_size = file
            .metadata
            .get_u32(&format!("{prefix}.vocab_size"))
            .map(|v| v as usize)
            .or_else(|| {
                file.tensors
                    .get("token_embd.weight")
                    .map(|t| t.shape[0] as usize)
            })
            .ok_or_else(|| GgufError::MissingMetadata {
                key: format!("{prefix}.vocab_size (and no token_embd.weight tensor)"),
            })?;

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
        })
    }
}

/// Map a single GGUF tensor name to the corresponding HuggingFace Qwen3 name.
///
/// Returns `None` for unrecognized tensor names (e.g., tokenizer data).
pub fn gguf_to_hf_name(gguf_name: &str) -> Option<String> {
    // Global tensors
    if gguf_name == "token_embd.weight" {
        return Some("model.embed_tokens.weight".to_string());
    }
    if gguf_name == "output_norm.weight" {
        return Some("model.norm.weight".to_string());
    }
    if gguf_name == "output.weight" {
        return Some("lm_head.weight".to_string());
    }

    // Per-layer tensors: blk.{i}.xxx.weight
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        let dot_pos = rest.find('.')?;
        let layer_str = &rest[..dot_pos];
        let suffix = &rest[dot_pos + 1..];

        let hf_suffix = match suffix {
            "attn_norm.weight" => "input_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            _ => return None,
        };

        return Some(format!("model.layers.{layer_str}.{hf_suffix}"));
    }

    None
}

/// Load all GGUF tensors, dequantize to f32, and return as a
/// `HashMap<String, DynTensor>` using HuggingFace Qwen3 weight names.
///
/// The returned map is suitable for `VarBuilder::from_tensors()` and can
/// be passed directly to `Qwen3Model::load()`.
///
/// Tensors whose GGUF names do not map to Qwen3 weight names (e.g.,
/// tokenizer data) are silently skipped.
///
/// When `tie_word_embeddings` is true and `output.weight` is absent in the
/// GGUF file, the `lm_head.weight` entry is created as a clone of
/// `model.embed_tokens.weight`.
pub fn load_qwen3_tensors<R: Read + Seek>(
    gguf: &GgufFile,
    reader: &mut R,
    tie_word_embeddings: bool,
) -> Result<HashMap<String, DynTensor>, GgufError> {
    let device = Device::Cpu;
    let mut tensors = HashMap::new();

    for name in gguf.tensors.keys() {
        if let Some(hf_name) = gguf_to_hf_name(name) {
            let (data, shape) = gguf.read_tensor_f32(reader, name)?;

            // Validate dequantized data is finite.
            let non_finite = data.iter().filter(|v| !v.is_finite()).count();
            if non_finite > 0 {
                return Err(GgufError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "tensor {name} ({hf_name}): {non_finite} non-finite values after dequantization"
                    ),
                )));
            }

            let tensor = DynTensor::from_vec(data, shape.as_slice(), &device).map_err(|e| {
                GgufError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("tensor {name} ({hf_name}): {e}"),
                ))
            })?;
            tensors.insert(hf_name, tensor);
        }
    }

    // Handle tied embeddings: if output.weight is absent, alias embed for lm_head.
    if tie_word_embeddings && !tensors.contains_key("lm_head.weight") {
        if let Some(embed) = tensors.get("model.embed_tokens.weight") {
            tensors.insert("lm_head.weight".to_string(), embed.clone());
        }
    }

    Ok(tensors)
}

/// Read a required `u32` metadata key, returning a descriptive error if absent.
fn require_u32_meta(metadata: &crate::metadata::GgufMetadata, key: &str) -> Result<u32, GgufError> {
    metadata
        .get_u32(key)
        .ok_or_else(|| GgufError::MissingMetadata {
            key: key.to_string(),
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_tensor_name_mapping() {
        assert_eq!(
            gguf_to_hf_name("token_embd.weight"),
            Some("model.embed_tokens.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("output_norm.weight"),
            Some("model.norm.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("output.weight"),
            Some("lm_head.weight".to_string())
        );
    }

    #[test]
    fn test_per_layer_tensor_name_mapping() {
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_norm.weight"),
            Some("model.layers.0.input_layernorm.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_q.weight"),
            Some("model.layers.0.self_attn.q_proj.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_k.weight"),
            Some("model.layers.0.self_attn.k_proj.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_v.weight"),
            Some("model.layers.0.self_attn.v_proj.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_output.weight"),
            Some("model.layers.0.self_attn.o_proj.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_q_norm.weight"),
            Some("model.layers.0.self_attn.q_norm.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.attn_k_norm.weight"),
            Some("model.layers.0.self_attn.k_norm.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_norm.weight"),
            Some("model.layers.0.post_attention_layernorm.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_gate.weight"),
            Some("model.layers.0.mlp.gate_proj.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_up.weight"),
            Some("model.layers.0.mlp.up_proj.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("blk.0.ffn_down.weight"),
            Some("model.layers.0.mlp.down_proj.weight".to_string())
        );
    }

    #[test]
    fn test_multi_layer_mapping() {
        assert_eq!(
            gguf_to_hf_name("blk.27.attn_q.weight"),
            Some("model.layers.27.self_attn.q_proj.weight".to_string())
        );
        assert_eq!(
            gguf_to_hf_name("blk.100.ffn_down.weight"),
            Some("model.layers.100.mlp.down_proj.weight".to_string())
        );
    }

    #[test]
    fn test_unknown_tensor_names_return_none() {
        assert_eq!(gguf_to_hf_name("tokenizer.ggml.tokens"), None);
        assert_eq!(gguf_to_hf_name("blk.0.unknown_thing.weight"), None);
        assert_eq!(gguf_to_hf_name(""), None);
        assert_eq!(gguf_to_hf_name("random_garbage"), None);
    }

    #[test]
    fn test_all_qwen3_layer_weights_mapped() {
        // Verify that a complete Qwen3 layer's GGUF tensors all map correctly.
        let layer_suffixes = [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ];

        for suffix in &layer_suffixes {
            let gguf_name = format!("blk.0.{suffix}");
            assert!(
                gguf_to_hf_name(&gguf_name).is_some(),
                "missing mapping for {gguf_name}"
            );
        }
    }

    #[test]
    fn test_expected_tensor_count() {
        // For a Qwen3 model with N layers, the expected number of mapped tensors is:
        // 3 global (embed, output_norm, lm_head) + 11 per layer
        let num_layers = 28;
        let global_tensors = 3;
        let per_layer_tensors = 11;
        let expected = global_tensors + num_layers * per_layer_tensors;
        assert_eq!(expected, 311);
    }
}
