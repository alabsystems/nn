// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Model architecture metadata extraction from GGUF files.
//!
//! GGUF files store model architecture parameters under standardized keys:
//! `general.architecture` identifies the model family (e.g., "llama", "qwen2"),
//! and `{arch}.context_length`, `{arch}.embedding_length`, etc. store the
//! hyperparameters. This module provides [`ModelArchitecture`] to extract
//! these into a single struct.

use std::fmt;

use crate::metadata::GgufMetadata;

/// Model architecture metadata extracted from a GGUF file header.
///
/// Fields follow the GGUF metadata key convention where architecture-specific
/// keys are prefixed with the architecture name (e.g., `llama.context_length`,
/// `qwen2.embedding_length`). The architecture name itself comes from
/// `general.architecture`.
///
/// All fields except `architecture` are optional because different model
/// families expose different subsets of metadata, and some GGUF producers
/// omit optional keys.
///
/// # Example
///
/// ```rust,ignore
/// use nn_gguf::{GgufFile, ModelArchitecture};
///
/// let gguf = GgufFile::open("model.gguf")?;
/// let arch = ModelArchitecture::from_metadata(&gguf.metadata);
/// println!("{arch}");
/// // Architecture: llama
/// //   Context length:    4096
/// //   Embedding dim:     4096
/// //   Block count:       32
/// //   Head count:        32
/// //   Head count (KV):   8
/// //   Vocab size:        32000
/// //   RoPE freq base:    10000
/// ```
#[derive(Debug, Clone)]
pub struct ModelArchitecture {
    /// Model architecture family (e.g., "llama", "qwen2", "gpt2", "phi").
    /// Extracted from `general.architecture`. Defaults to `"unknown"` when
    /// the key is absent.
    pub architecture: String,

    /// Maximum context (sequence) length the model was trained for.
    /// Key: `{arch}.context_length`.
    pub context_length: Option<u64>,

    /// Hidden dimension / embedding size.
    /// Key: `{arch}.embedding_length`.
    pub embedding_length: Option<u64>,

    /// Number of transformer blocks (layers).
    /// Key: `{arch}.block_count`.
    pub block_count: Option<u64>,

    /// Number of attention heads (query heads).
    /// Key: `{arch}.attention.head_count`.
    pub head_count: Option<u64>,

    /// Number of key/value heads for grouped-query attention (GQA).
    /// Key: `{arch}.attention.head_count_kv`.
    /// When equal to `head_count`, the model uses standard multi-head attention.
    pub head_count_kv: Option<u64>,

    /// Vocabulary size (number of token embeddings).
    /// Key: `{arch}.vocab_size`.
    pub vocab_size: Option<u64>,

    /// Base frequency for rotary position embeddings (RoPE).
    /// Key: `{arch}.rope.freq_base`.
    pub rope_freq_base: Option<f64>,

    /// GGUF quantization version used for the file.
    /// Key: `general.quantization_version`.
    pub quantization_version: Option<u32>,
}

impl ModelArchitecture {
    /// Extract model architecture metadata from GGUF metadata entries.
    ///
    /// Reads `general.architecture` first to determine the key prefix, then
    /// looks up `{arch}.context_length`, `{arch}.embedding_length`, etc.
    ///
    /// Missing keys result in `None` for the corresponding field (no error).
    /// If `general.architecture` itself is missing, `architecture` is set to
    /// `"unknown"` and all arch-prefixed lookups use `"unknown"` as the prefix
    /// (which will typically find nothing).
    pub fn from_metadata(metadata: &GgufMetadata) -> Self {
        let architecture = metadata
            .get_str("general.architecture")
            .unwrap_or("unknown")
            .to_string();

        let arch = &architecture;

        Self {
            context_length: metadata.get_u64(&format!("{arch}.context_length")),
            embedding_length: metadata.get_u64(&format!("{arch}.embedding_length")),
            block_count: metadata.get_u64(&format!("{arch}.block_count")),
            head_count: metadata.get_u64(&format!("{arch}.attention.head_count")),
            head_count_kv: metadata.get_u64(&format!("{arch}.attention.head_count_kv")),
            vocab_size: metadata.get_u64(&format!("{arch}.vocab_size")),
            rope_freq_base: metadata.get_f64(&format!("{arch}.rope.freq_base")),
            quantization_version: metadata.get_u32("general.quantization_version"),
            architecture,
        }
    }
}

impl fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Architecture: {}", self.architecture)?;

        if let Some(v) = self.context_length {
            writeln!(f, "  Context length:    {v}")?;
        }
        if let Some(v) = self.embedding_length {
            writeln!(f, "  Embedding dim:     {v}")?;
        }
        if let Some(v) = self.block_count {
            writeln!(f, "  Block count:       {v}")?;
        }
        if let Some(v) = self.head_count {
            writeln!(f, "  Head count:        {v}")?;
        }
        if let Some(v) = self.head_count_kv {
            writeln!(f, "  Head count (KV):   {v}")?;
        }
        if let Some(v) = self.vocab_size {
            writeln!(f, "  Vocab size:        {v}")?;
        }
        if let Some(v) = self.rope_freq_base {
            writeln!(f, "  RoPE freq base:    {v}")?;
        }
        if let Some(v) = self.quantization_version {
            writeln!(f, "  Quantization ver:  {v}")?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::metadata::GgufMetadataValue;

    /// Build a GgufMetadata from a list of (key, value) pairs.
    fn build_metadata(entries: Vec<(&str, GgufMetadataValue)>) -> GgufMetadata {
        let map: HashMap<String, GgufMetadataValue> = entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        GgufMetadata { entries: map }
    }

    #[test]
    fn test_full_llama_metadata() {
        let meta = build_metadata(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("llama".into()),
            ),
            ("llama.context_length", GgufMetadataValue::U32(4096)),
            ("llama.embedding_length", GgufMetadataValue::U32(4096)),
            ("llama.block_count", GgufMetadataValue::U32(32)),
            ("llama.attention.head_count", GgufMetadataValue::U32(32)),
            ("llama.attention.head_count_kv", GgufMetadataValue::U32(8)),
            ("llama.vocab_size", GgufMetadataValue::U32(32000)),
            ("llama.rope.freq_base", GgufMetadataValue::F32(10000.0)),
            ("general.quantization_version", GgufMetadataValue::U32(2)),
        ]);

        let arch = ModelArchitecture::from_metadata(&meta);
        assert_eq!(arch.architecture, "llama");
        assert_eq!(arch.context_length, Some(4096));
        assert_eq!(arch.embedding_length, Some(4096));
        assert_eq!(arch.block_count, Some(32));
        assert_eq!(arch.head_count, Some(32));
        assert_eq!(arch.head_count_kv, Some(8));
        assert_eq!(arch.vocab_size, Some(32000));
        assert!((arch.rope_freq_base.unwrap() - 10000.0).abs() < 1e-6);
        assert_eq!(arch.quantization_version, Some(2));
    }

    #[test]
    fn test_missing_optional_fields() {
        let meta = build_metadata(vec![(
            "general.architecture",
            GgufMetadataValue::String("gpt2".into()),
        )]);

        let arch = ModelArchitecture::from_metadata(&meta);
        assert_eq!(arch.architecture, "gpt2");
        assert_eq!(arch.context_length, None);
        assert_eq!(arch.embedding_length, None);
        assert_eq!(arch.block_count, None);
        assert_eq!(arch.head_count, None);
        assert_eq!(arch.head_count_kv, None);
        assert_eq!(arch.vocab_size, None);
        assert_eq!(arch.rope_freq_base, None);
        assert_eq!(arch.quantization_version, None);
    }

    #[test]
    fn test_missing_architecture_defaults_to_unknown() {
        let meta = build_metadata(vec![]);

        let arch = ModelArchitecture::from_metadata(&meta);
        assert_eq!(arch.architecture, "unknown");
        assert_eq!(arch.context_length, None);
    }

    #[test]
    fn test_qwen2_architecture_keys() {
        let meta = build_metadata(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("qwen2".into()),
            ),
            ("qwen2.context_length", GgufMetadataValue::U32(32768)),
            ("qwen2.embedding_length", GgufMetadataValue::U32(3584)),
            ("qwen2.block_count", GgufMetadataValue::U32(28)),
            ("qwen2.attention.head_count", GgufMetadataValue::U32(28)),
            ("qwen2.attention.head_count_kv", GgufMetadataValue::U32(4)),
            ("qwen2.vocab_size", GgufMetadataValue::U32(152064)),
            ("qwen2.rope.freq_base", GgufMetadataValue::F32(1000000.0)),
        ]);

        let arch = ModelArchitecture::from_metadata(&meta);
        assert_eq!(arch.architecture, "qwen2");
        assert_eq!(arch.context_length, Some(32768));
        assert_eq!(arch.embedding_length, Some(3584));
        assert_eq!(arch.block_count, Some(28));
        assert_eq!(arch.head_count, Some(28));
        assert_eq!(arch.head_count_kv, Some(4));
        assert_eq!(arch.vocab_size, Some(152064));
        assert!((arch.rope_freq_base.unwrap() - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_u64_metadata_values() {
        // Some GGUF producers store values as u64 rather than u32.
        let meta = build_metadata(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("llama".into()),
            ),
            ("llama.context_length", GgufMetadataValue::U64(131072)),
            ("llama.vocab_size", GgufMetadataValue::U64(128256)),
        ]);

        let arch = ModelArchitecture::from_metadata(&meta);
        assert_eq!(arch.context_length, Some(131072));
        assert_eq!(arch.vocab_size, Some(128256));
    }

    #[test]
    fn test_f64_rope_freq_base() {
        // Some GGUF producers store rope.freq_base as f64.
        let meta = build_metadata(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("phi".into()),
            ),
            ("phi.rope.freq_base", GgufMetadataValue::F64(250000.0)),
        ]);

        let arch = ModelArchitecture::from_metadata(&meta);
        assert!((arch.rope_freq_base.unwrap() - 250000.0).abs() < 1e-6);
    }

    #[test]
    fn test_partial_metadata() {
        // Only some fields present -- common for smaller models.
        let meta = build_metadata(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("llama".into()),
            ),
            ("llama.embedding_length", GgufMetadataValue::U32(2048)),
            ("llama.block_count", GgufMetadataValue::U32(22)),
        ]);

        let arch = ModelArchitecture::from_metadata(&meta);
        assert_eq!(arch.architecture, "llama");
        assert_eq!(arch.embedding_length, Some(2048));
        assert_eq!(arch.block_count, Some(22));
        assert_eq!(arch.context_length, None);
        assert_eq!(arch.head_count, None);
        assert_eq!(arch.head_count_kv, None);
        assert_eq!(arch.vocab_size, None);
        assert_eq!(arch.rope_freq_base, None);
        assert_eq!(arch.quantization_version, None);
    }

    #[test]
    fn test_wrong_type_returns_none() {
        // If a key exists but with the wrong type, it should yield None.
        let meta = build_metadata(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("llama".into()),
            ),
            // context_length as a string instead of u32/u64 -- should be None.
            (
                "llama.context_length",
                GgufMetadataValue::String("4096".into()),
            ),
        ]);

        let arch = ModelArchitecture::from_metadata(&meta);
        assert_eq!(arch.context_length, None);
    }

    #[test]
    fn test_display_full() {
        let meta = build_metadata(vec![
            (
                "general.architecture",
                GgufMetadataValue::String("llama".into()),
            ),
            ("llama.context_length", GgufMetadataValue::U32(4096)),
            ("llama.embedding_length", GgufMetadataValue::U32(4096)),
            ("llama.block_count", GgufMetadataValue::U32(32)),
            ("llama.attention.head_count", GgufMetadataValue::U32(32)),
            ("llama.attention.head_count_kv", GgufMetadataValue::U32(8)),
            ("llama.vocab_size", GgufMetadataValue::U32(32000)),
            ("llama.rope.freq_base", GgufMetadataValue::F32(10000.0)),
            ("general.quantization_version", GgufMetadataValue::U32(2)),
        ]);

        let arch = ModelArchitecture::from_metadata(&meta);
        let display = format!("{arch}");

        assert!(display.contains("Architecture: llama"));
        assert!(display.contains("Context length:    4096"));
        assert!(display.contains("Embedding dim:     4096"));
        assert!(display.contains("Block count:       32"));
        assert!(display.contains("Head count:        32"));
        assert!(display.contains("Head count (KV):   8"));
        assert!(display.contains("Vocab size:        32000"));
        assert!(display.contains("RoPE freq base:    10000"));
        assert!(display.contains("Quantization ver:  2"));
    }

    #[test]
    fn test_display_minimal() {
        let meta = build_metadata(vec![(
            "general.architecture",
            GgufMetadataValue::String("gpt2".into()),
        )]);

        let arch = ModelArchitecture::from_metadata(&meta);
        let display = format!("{arch}");

        assert!(display.contains("Architecture: gpt2"));
        // No optional fields should appear.
        assert!(!display.contains("Context length"));
        assert!(!display.contains("Embedding dim"));
    }
}
