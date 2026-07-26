// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pluggable weight name mapping for HuggingFace-to-NN import.
//!
//! [`WeightNameMapper`] is a trait for translating weight names between
//! checkpoint naming conventions and NN model naming conventions. The primary
//! implementation, [`HfToNnMapper`], handles common HuggingFace patterns.
//!
//! # Architecture
//!
//! Weight names in ML checkpoints are dot-separated hierarchical keys like
//! `model.layers.0.self_attn.q_proj.weight`. Different frameworks and models
//! use different naming conventions for the same logical weight:
//!
//! - HuggingFace: `model.layers.0.self_attn.q_proj.weight`
//! - NN model:   `encoder.layer.0.attention.q.weight`
//!
//! The mapper translates between these conventions using segment-level rules:
//! prefix replacement, segment renaming, and suffix stripping.
//!
//! # Usage
//!
//! ```no_run
//! use nn_core::var_builder::{VarBuilder, HfToNnMapper, WeightNameMapper};
//! use nn_core::{DType, Device};
//!
//! let mapper = HfToNnMapper::new()
//!     .with_prefix_rule("model.layers", "encoder.layer")
//!     .with_segment_rule("self_attn", "attention")
//!     .with_segment_rule("q_proj", "q")
//!     .with_segment_rule("k_proj", "k")
//!     .with_segment_rule("v_proj", "v");
//!
//! let vb = VarBuilder::zeros(DType::F32, &Device::Cpu)
//!     .with_weight_name_mapper(mapper);
//! ```

use std::collections::HashMap;

/// Trait for pluggable weight name translation.
///
/// Implementations translate a fully-resolved NN weight key (after `pp()` prefix
/// + tensor name concatenation) to the corresponding checkpoint key. This is the
///   same position in the pipeline as [`VarBuilder::with_name_mapping`], but
///   structured as a composable trait rather than a bare closure.
///
/// [`VarBuilder::with_name_mapping`]: super::VarBuilder::with_name_mapping
pub trait WeightNameMapper: Send + Sync {
    /// Translate an NN model key to the corresponding checkpoint key.
    ///
    /// The input is the fully-resolved key (e.g., `"encoder.layer.0.attention.q.weight"`).
    /// The output is the checkpoint key to look up in the backend
    /// (e.g., `"model.layers.0.self_attn.q_proj.weight"`).
    ///
    /// Return the input unchanged if no mapping applies.
    fn map_name(&self, nn_name: &str) -> String;

    /// Human-readable description of this mapper (for debug/diagnostics).
    #[allow(clippy::unnecessary_literal_bound)]
    fn description(&self) -> &str {
        "WeightNameMapper"
    }
}

/// A single mapping rule for [`HfToNnMapper`].
#[derive(Debug, Clone)]
pub(crate) enum MappingRule {
    /// Replace a dot-separated prefix.
    /// `("model.layers", "encoder.layer")` maps
    /// `"encoder.layer.0.weight"` -> `"model.layers.0.weight"`.
    Prefix { from: String, to: String },
    /// Replace a segment name anywhere in the key.
    /// `("self_attn", "attention")` maps
    /// `"encoder.layer.0.attention.q.weight"` -> `"encoder.layer.0.self_attn.q.weight"`.
    ///
    /// Also used internally by `with_suffix_rule()` to expand base segments
    /// (e.g., `"q"` -> `"q_proj"`).
    Segment { from: String, to: String },
}

/// Weight name mapper with composable rules for HuggingFace-to-NN translation.
///
/// Rules are applied in order. The mapper translates **from NN names to HF names**
/// (i.e., the NN model code requests `"encoder.layer.0.attention.q.weight"` and
/// the mapper produces `"model.layers.0.self_attn.q_proj.weight"` for backend lookup).
///
/// This direction matches [`VarBuilder::with_name_mapping`]: the function receives
/// the NN name and returns the checkpoint name.
///
/// # Rule Types
///
/// - **Prefix rules**: Replace a dot-separated prefix. Applied first, at most one matches.
/// - **Segment rules**: Replace individual dot-separated segments. All matching rules apply.
/// - **Suffix strip rules**: Append a suffix to matching segments (inverse of stripping).
///
/// # Example
///
/// ```
/// use nn_core::var_builder::{HfToNnMapper, WeightNameMapper};
///
/// let mapper = HfToNnMapper::new()
///     .with_prefix_rule("model.layers", "encoder.layer")
///     .with_segment_rule("self_attn", "attention")
///     .with_segment_rule("q_proj", "q");
///
/// // NN model requests "encoder.layer.0.attention.q.weight"
/// // Mapper produces "model.layers.0.self_attn.q_proj.weight" for checkpoint lookup
/// assert_eq!(
///     mapper.map_name("encoder.layer.0.attention.q.weight"),
///     "model.layers.0.self_attn.q_proj.weight"
/// );
/// ```
#[derive(Debug, Clone)]
pub struct HfToNnMapper {
    rules: Vec<MappingRule>,
    /// Optional exact overrides. Keys are NN names, values are HF checkpoint names.
    /// Applied before rule-based mapping — if an exact match exists, rules are skipped.
    exact_overrides: HashMap<String, String>,
    description: String,
}

impl HfToNnMapper {
    /// Create an empty mapper with no rules.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            exact_overrides: HashMap::new(),
            description: "HfToNnMapper".to_string(),
        }
    }

    /// Add a prefix replacement rule.
    ///
    /// When the NN key starts with `nn_prefix`, replace it with `hf_prefix`
    /// in the output. Only the first matching prefix rule applies.
    ///
    /// ```
    /// # use nn_core::var_builder::{HfToNnMapper, WeightNameMapper};
    /// let mapper = HfToNnMapper::new()
    ///     .with_prefix_rule("model.layers", "encoder.layer");
    /// assert_eq!(
    ///     mapper.map_name("encoder.layer.0.weight"),
    ///     "model.layers.0.weight"
    /// );
    /// ```
    #[must_use]
    pub fn with_prefix_rule(mut self, hf_prefix: &str, nn_prefix: &str) -> Self {
        self.rules.push(MappingRule::Prefix {
            from: nn_prefix.to_string(),
            to: hf_prefix.to_string(),
        });
        self
    }

    /// Add a segment replacement rule.
    ///
    /// Replaces any dot-separated segment matching `nn_segment` with `hf_segment`.
    /// Applied to all matching segments in the key.
    ///
    /// ```
    /// # use nn_core::var_builder::{HfToNnMapper, WeightNameMapper};
    /// let mapper = HfToNnMapper::new()
    ///     .with_segment_rule("self_attn", "attention");
    /// assert_eq!(
    ///     mapper.map_name("layer.0.attention.weight"),
    ///     "layer.0.self_attn.weight"
    /// );
    /// ```
    #[must_use]
    pub fn with_segment_rule(mut self, hf_segment: &str, nn_segment: &str) -> Self {
        self.rules.push(MappingRule::Segment {
            from: nn_segment.to_string(),
            to: hf_segment.to_string(),
        });
        self
    }

    /// Add a suffix restoration rule.
    ///
    /// For segments that match a known pattern (determined by other segment rules
    /// or by checking the base name), appends `suffix` to produce the HF name.
    ///
    /// This is the inverse of "stripping `_proj` from `q_proj` to get `q`":
    /// the NN model uses `q`, and this rule restores it to `q_proj` for HF lookup.
    ///
    /// ```
    /// # use nn_core::var_builder::{HfToNnMapper, WeightNameMapper};
    /// let mapper = HfToNnMapper::new()
    ///     .with_suffix_rule("_proj", &["q", "k", "v", "o"]);
    /// assert_eq!(mapper.map_name("layer.q.weight"), "layer.q_proj.weight");
    /// assert_eq!(mapper.map_name("layer.bias"), "layer.bias"); // no match
    /// ```
    #[must_use]
    pub fn with_suffix_rule(mut self, suffix: &str, base_segments: &[&str]) -> Self {
        for base in base_segments {
            // Store as a segment rule: "q" -> "q_proj"
            self.rules.push(MappingRule::Segment {
                from: (*base).to_string(),
                to: format!("{base}{suffix}"),
            });
        }
        self
    }

    /// Add exact override mappings. These take precedence over rules.
    ///
    /// Keys are NN names, values are HF checkpoint names.
    #[must_use]
    pub fn with_exact_overrides(mut self, overrides: HashMap<String, String>) -> Self {
        self.exact_overrides.extend(overrides);
        self
    }

    /// Set a human-readable description for this mapper.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Apply rules to translate an NN name to an HF checkpoint name.
    fn apply_rules(&self, nn_name: &str) -> String {
        let mut result = nn_name.to_string();

        // Phase 1: Apply prefix rules (first match wins).
        for rule in &self.rules {
            if let MappingRule::Prefix { from, to } = rule {
                if from.is_empty() {
                    // Empty nn_prefix: prepend HF prefix to the name.
                    // "".with_prefix_rule("model.vision_model", "") means
                    // NN has no prefix, HF has "model.vision_model." prefix.
                    if result.is_empty() {
                        result = to.clone();
                    } else {
                        result = format!("{to}.{result}");
                    }
                    break;
                }
                if let Some(rest) = result.strip_prefix(from.as_str()) {
                    // Ensure we match at a segment boundary (dot or end of string).
                    if rest.is_empty() || rest.starts_with('.') {
                        if to.is_empty() {
                            // Empty HF prefix: strip the NN prefix entirely.
                            result = rest.strip_prefix('.').unwrap_or(rest).to_string();
                        } else {
                            result = format!("{to}{rest}");
                        }
                        break;
                    }
                }
            }
        }

        // Phase 2: Apply segment rules.
        // Split into segments, replace matching ones, rejoin.
        let segments: Vec<&str> = result.split('.').collect();
        let mapped_segments: Vec<String> = segments
            .iter()
            .map(|seg| {
                let mut s = (*seg).to_string();
                for rule in &self.rules {
                    if let MappingRule::Segment { from, to } = rule {
                        if s == *from {
                            s = to.clone();
                            break; // First matching segment rule wins per segment
                        }
                    }
                }
                s
            })
            .collect();
        result = mapped_segments.join(".");

        result
    }
}

impl Default for HfToNnMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightNameMapper for HfToNnMapper {
    fn map_name(&self, nn_name: &str) -> String {
        // Exact overrides take precedence.
        if let Some(hf_name) = self.exact_overrides.get(nn_name) {
            return hf_name.clone();
        }
        self.apply_rules(nn_name)
    }

    fn description(&self) -> &str {
        &self.description
    }
}

// -- Pre-built mappers for common HF model families ----------------------------

impl HfToNnMapper {
    /// Pre-built mapper for Qwen3-family models.
    ///
    /// HuggingFace Qwen3 uses names like:
    /// - `model.layers.{i}.self_attn.q_proj.weight`
    /// - `model.layers.{i}.mlp.gate_proj.weight`
    /// - `model.embed_tokens.weight`
    /// - `model.norm.weight`
    /// - `lm_head.weight`
    ///
    /// This mapper handles the common case where the NN model uses the
    /// same naming convention (Qwen3 in nn-qwen3 already matches HF names).
    /// Useful as a starting point for models that rename a few layers.
    #[must_use]
    pub fn qwen3() -> Self {
        Self::new().with_description("HfToNnMapper::qwen3")
        // Qwen3 NN model already uses HF naming, so identity mapping.
        // Users can chain additional rules for custom modifications.
    }

    /// Pre-built mapper for SigLIP2 vision encoder (Granite-Docling pattern).
    ///
    /// HuggingFace Granite-Docling uses:
    /// - `model.vision_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}`
    /// - `model.vision_model.encoder.layers.{i}.layer_norm{1,2}.{weight,bias}`
    /// - `model.vision_model.encoder.layers.{i}.mlp.fc{1,2}.{weight,bias}`
    /// - `model.vision_model.embeddings.patch_embedding.{weight,bias}`
    /// - `model.vision_model.embeddings.position_embedding.weight`
    /// - `model.vision_model.post_layernorm.{weight,bias}`
    ///
    /// NN SigLIP2 uses (without the `model.vision_model.` prefix):
    /// - `encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}`
    /// - `embeddings.patch_embedding.{weight,bias}`
    /// - etc.
    #[must_use]
    pub fn siglip2_granite_docling() -> Self {
        Self::new()
            .with_description("HfToNnMapper::siglip2_granite_docling")
            .with_prefix_rule("model.vision_model", "")
    }

    /// Pre-built mapper for decoder-only transformers with common HF naming.
    ///
    /// Handles the common pattern where HF uses `model.layers.{i}.self_attn.*_proj`
    /// and the NN model uses shorter names like `layers.{i}.attn.{q,k,v,o}`.
    #[must_use]
    pub fn decoder_transformer() -> Self {
        Self::new()
            .with_description("HfToNnMapper::decoder_transformer")
            .with_prefix_rule("model.layers", "layers")
            .with_segment_rule("self_attn", "attn")
            .with_segment_rule("q_proj", "q")
            .with_segment_rule("k_proj", "k")
            .with_segment_rule("v_proj", "v")
            .with_segment_rule("o_proj", "o")
            .with_segment_rule("gate_proj", "gate")
            .with_segment_rule("up_proj", "up")
            .with_segment_rule("down_proj", "down")
            .with_segment_rule("input_layernorm", "ln1")
            .with_segment_rule("post_attention_layernorm", "ln2")
    }
}

/// Verify that a mapper can successfully resolve all expected NN weight names.
///
/// Given a list of NN model weight names and a mapper, applies the mapper to
/// each name and checks whether the result exists in `checkpoint_names`. Returns
/// a list of NN names that failed to map to any checkpoint name.
///
/// This is a diagnostic tool for model import: call it after building a mapper
/// to verify all model weights can be loaded before starting inference.
///
/// # Example
///
/// ```
/// use nn_core::var_builder::{HfToNnMapper, WeightNameMapper, verify_mapper_coverage};
///
/// let mapper = HfToNnMapper::new()
///     .with_prefix_rule("model", "m");
///
/// let checkpoint_names = vec!["model.weight".to_string(), "model.bias".to_string()];
/// let nn_names = vec!["m.weight".to_string(), "m.bias".to_string(), "m.extra".to_string()];
///
/// let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
/// assert_eq!(missing, vec!["m.extra"]); // maps to "model.extra" which is not in checkpoint
/// ```
pub fn verify_mapper_coverage(
    nn_names: &[String],
    checkpoint_names: &[String],
    mapper: &dyn WeightNameMapper,
) -> Vec<String> {
    let checkpoint_set: std::collections::HashSet<&str> =
        checkpoint_names.iter().map(String::as_str).collect();
    nn_names
        .iter()
        .filter(|name| {
            let mapped = mapper.map_name(name);
            !checkpoint_set.contains(mapped.as_str())
        })
        .cloned()
        .collect()
}

#[cfg(test)]
#[path = "weight_name_mapper_tests.rs"]
mod tests;
