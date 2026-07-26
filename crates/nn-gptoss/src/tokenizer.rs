// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer for gpt-oss / Chroma Context-1 models.
//!
//! Wraps the HuggingFace `tokenizers` crate for BPE encoding/decoding.
//! Gated behind the `tokenizer` feature flag.

use std::path::Path;

use crate::GptOssError;
use nn_core::Result;

/// BPE tokenizer for Context-1 / gpt-oss models.
///
/// Loads from a HuggingFace `tokenizer.json` file (201,088-token BPE vocabulary).
/// Provides encode (text -> token IDs) and decode (token IDs -> text) operations.
pub struct GptOssTokenizer {
    inner: tokenizers::Tokenizer,
    eos_token_id: u32,
}

impl GptOssTokenizer {
    /// Load tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref()).map_err(|e| {
            GptOssError::WeightLoad {
                reason: format!("tokenizer load failed: {e}"),
            }
        })?;
        Ok(Self {
            inner,
            eos_token_id: 200_002,
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| GptOssError::WeightLoad {
                reason: format!("tokenizer encode failed: {e}"),
            })?;
        Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[usize]) -> Result<String> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        Ok(self
            .inner
            .decode(&u32_ids, true)
            .map_err(|e| GptOssError::WeightLoad {
                reason: format!("tokenizer decode failed: {e}"),
            })?)
    }

    /// Count tokens in text.
    pub fn token_count(&self, text: &str) -> Result<usize> {
        Ok(self.encode(text)?.len())
    }

    /// EOS token ID (200002 for Context-1).
    pub const fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
