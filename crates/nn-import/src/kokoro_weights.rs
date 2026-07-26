// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro weight name mapping: PyTorch state_dict keys -> nn VarBuilder paths.
//!
//! The nn KokoroModel was written to match PyTorch's `kokoro_v0_19.pth`
//! state_dict key naming convention. For most keys, the mapping is identity.
//! This module provides validation and any known exception mappings.
//!
//! # Key Layout
//!
//! ```text
//! plbert.embeddings.word_embeddings.weight
//! plbert.embeddings.position_embeddings.weight
//! plbert.embeddings.token_type_embeddings.weight
//! plbert.embeddings.LayerNorm.{weight,bias}
//! plbert.encoder.embedding_hidden_mapping_in.{weight,bias}
//! plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.{query,key,value,dense}.{weight,bias}
//! plbert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.{weight,bias}
//! plbert.encoder.albert_layer_groups.0.albert_layers.0.{ffn,ffn_output}.{weight,bias}
//! plbert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.{weight,bias}
//! bert_encoder.{weight,bias}
//! text_encoder.lstm.{weight_ih_l0,weight_hh_l0,bias_ih_l0,bias_hh_l0}
//! text_encoder.lstm.{weight_ih_l0_reverse,weight_hh_l0_reverse,bias_ih_l0_reverse,bias_hh_l0_reverse}
//! text_encoder.lstm.linear.{weight,bias}
//! prosody_predictor.shared.{i}.conv.{weight,bias}
//! prosody_predictor.shared.{i}.lstm.{weight_ih_l0,weight_hh_l0,bias_ih_l0,bias_hh_l0}
//! prosody_predictor.shared.{i}.lstm.linear.{weight,bias}
//! prosody_predictor.norms.{i}.norm.{weight,bias}
//! prosody_predictor.norms.{i}.fc.{weight,bias}
//! prosody_predictor.duration_proj.{weight,bias}
//! predictor.shared.{weight_ih_l0,weight_hh_l0,bias_ih_l0,bias_hh_l0}
//! predictor.shared.{weight_ih_l0_reverse,weight_hh_l0_reverse,bias_ih_l0_reverse,bias_hh_l0_reverse}
//! predictor.F0.{i}.{n1,n2}.fc.{weight,bias}
//! predictor.F0.{i}.{c1,c2}.{weight,bias}
//! predictor.F0.{i}.{skip,pool}.{weight,bias}       (only for downsampling blocks)
//! predictor.F0_proj.{weight,bias}
//! predictor.N.{i}.{n1,n2}.fc.{weight,bias}
//! predictor.N.{i}.{c1,c2}.{weight,bias}
//! predictor.N.{i}.{skip,pool}.{weight,bias}
//! predictor.N_proj.{weight,bias}
//! decoder.conv_pre.{weight,bias}
//! decoder.ups.{i}.{weight,bias}
//! decoder.noise_convs.{i}.{weight,bias}
//! decoder.noise_res.{i}.convs{1,2}.{j}.{weight,bias}
//! decoder.noise_res.{i}.adain{1,2}.{j}.fc.{weight,bias}
//! decoder.noise_res.{i}.alpha{1,2}.{j}
//! decoder.resblocks.{k}.convs{1,2}.{d}.{weight,bias}
//! decoder.resblocks.{k}.adain{1,2}.{d}.fc.{weight,bias}
//! decoder.resblocks.{k}.alpha{1,2}.{d}
//! decoder.conv_post.{weight,bias}
//! ```
//!
//! Part of #2465, #2218.

use crate::ImportError;

/// Known top-level prefixes in PyTorch kokoro_v0_19 state_dict.
const EXPECTED_PREFIXES: &[&str] = &[
    "plbert.",
    "bert_encoder.",
    "text_encoder.",
    "prosody_predictor.",
    "predictor.",
    "decoder.",
];

/// Map a PyTorch state_dict key to the corresponding nn VarBuilder path.
///
/// The nn KokoroModel was written to match PyTorch naming conventions, so
/// this function is identity for standard kokoro_v0_19 keys. Exceptions are
/// documented and mapped below.
///
/// Returns `None` if the key is unrecognized (not matching any expected prefix).
pub fn map_pytorch_key(key: &str) -> Option<String> {
    // Verify the key matches an expected prefix.
    if !EXPECTED_PREFIXES.iter().any(|p| key.starts_with(p)) {
        return None;
    }
    // Identity mapping — nn was written to match PyTorch naming.
    Some(key.to_string())
}

/// Validate that a set of safetensors keys contains the required Kokoro weights.
///
/// Returns a list of missing required prefixes. An empty list means all
/// required weight groups are present.
pub fn validate_kokoro_keys(keys: &[&str]) -> Vec<&'static str> {
    let mut missing = Vec::new();
    for prefix in EXPECTED_PREFIXES {
        if !keys.iter().any(|k| k.starts_with(prefix)) {
            missing.push(*prefix);
        }
    }
    missing
}

/// Validate and report on safetensors key coverage for Kokoro weights.
///
/// Returns `Ok(mapped_count)` if all required prefixes are present.
/// Returns `Err` with details about missing weight groups.
pub fn validate_kokoro_safetensors(keys: &[String]) -> Result<usize, ImportError> {
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    let missing = validate_kokoro_keys(&key_refs);

    if !missing.is_empty() {
        return Err(ImportError::MissingWeightGroups {
            missing_prefixes: missing.join(", "),
        });
    }

    // Count keys that map successfully.
    let mapped = key_refs
        .iter()
        .filter(|k| map_pytorch_key(k).is_some())
        .count();

    Ok(mapped)
}

/// Create a VarBuilder name mapping closure for Kokoro weights.
///
/// Returns a closure suitable for `VarBuilder::with_name_mapping()` that
/// transforms PyTorch state_dict keys to nn VarBuilder paths.
pub fn kokoro_name_mapping() -> impl Fn(&str) -> String + Send + Sync + 'static {
    move |key: &str| map_pytorch_key(key).unwrap_or_else(|| key.to_string())
}

#[cfg(test)]
#[path = "kokoro_weights_tests.rs"]
mod tests;
