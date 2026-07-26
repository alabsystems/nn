// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! FireRed-OCR model builder wrapping Qwen3-VL for document OCR.
//!
//! FireRed-OCR (2B) is a document OCR model built on the Qwen3-VL-2B
//! vision-language architecture. It accepts a document page image and
//! produces token IDs that decode to OCR text.
//!
//! Architecture:
//! - **Base:** Qwen3-VL-2B (vision encoder + decoder-only transformer)
//! - **OCR modes:** FullPage, RegionCrop, LineLevel
//! - **Output:** Autoregressive OCR token generation
//!
//! Reference: `yuyq96/FireRed-OCR-Qwen3-VL-2B` (HuggingFace).
//!
//! # Integration with dpdf
//!
//! FireRed-OCR can serve as an OCR backend in the dpdf document pipeline.
//! To integrate, add a `FireRedOcr` variant to the pipeline's OCR backend
//! selection (see `dpdf_pipeline.rs`). The model accepts pre-cropped region
//! images or full page images depending on [`OcrMode`].

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{check_output_finite, KvCache};
use nn_core::var_builder::VarBuilder;
use nn_core::{Result, TensorError};

use crate::qwen3_vl::{Qwen3VL, Qwen3VLConfig};

// ---------------------------------------------------------------------------
// OCR mode
// ---------------------------------------------------------------------------

/// OCR operating mode controlling input preprocessing and output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[derive(Default)]
pub enum OcrMode {
    /// Full-page OCR: process the entire document page at once.
    /// Best for clean single-column layouts.
    #[default]
    FullPage,
    /// Region-crop OCR: process a cropped bounding-box region.
    /// Best for targeted extraction from complex layouts.
    RegionCrop,
    /// Line-level OCR: process individual text lines.
    /// Best for structured forms and tables.
    LineLevel,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default maximum output tokens for OCR generation.
const DEFAULT_MAX_OUTPUT_TOKENS: usize = 4096;

/// Configuration for FireRed-OCR models.
///
/// Wraps [`Qwen3VLConfig`] with OCR-specific parameters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FireRedOcrConfig {
    /// Base Qwen3-VL model configuration.
    pub base_config: Qwen3VLConfig,
    /// Maximum number of output tokens for OCR generation.
    pub max_output_tokens: usize,
    /// OCR operating mode.
    pub ocr_mode: OcrMode,
}

impl FireRedOcrConfig {
    /// Create the 2B preset configuration (FireRed-OCR-Qwen3-VL-2B).
    ///
    /// Uses the Qwen3-VL-2B base with FireRed-specific vocab size (151936)
    /// and OCR-tuned defaults.
    #[must_use]
    pub fn preset_2b() -> Self {
        let mut base = Qwen3VLConfig::preset_2b();
        // FireRed-OCR uses a slightly different vocab size than base Qwen3-VL
        base.vocab_size = 151936;
        Self {
            base_config: base,
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            ocr_mode: OcrMode::default(),
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        self.base_config.validate()?;
        if self.max_output_tokens == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "FireRedOcrConfig: max_output_tokens must be > 0",
            });
        }
        Ok(())
    }

    /// Access the hidden size from the base config.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.base_config.hidden_size
    }

    /// Access the number of decoder layers from the base config.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.base_config.num_layers
    }

    /// Access the vocabulary size from the base config.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.base_config.vocab_size
    }

    /// Decoder head dimension (delegated to base config).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.base_config.head_dim()
    }

    /// GQA group ratio (delegated to base config).
    #[must_use]
    pub fn gqa_ratio(&self) -> usize {
        self.base_config.gqa_ratio()
    }
}

// ---------------------------------------------------------------------------
// OCR output
// ---------------------------------------------------------------------------

/// Output from FireRed-OCR forward pass.
#[derive(Debug, Clone)]
pub struct FireRedOcrOutput {
    /// Next-token logits: `[B, S, vocab_size]`.
    pub logits: DynTensor,
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// FireRed-OCR: Qwen3-VL-based document OCR model.
///
/// Wraps a [`Qwen3VL`] model with OCR-specific forward and decode methods.
#[derive(Clone)]
pub struct FireRedOcr {
    inner: Qwen3VL,
    config: FireRedOcrConfig,
}

impl std::fmt::Debug for FireRedOcr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FireRedOcr")
            .field("inner", &self.inner)
            .field("ocr_mode", &self.config.ocr_mode)
            .field("max_output_tokens", &self.config.max_output_tokens)
            .finish_non_exhaustive()
    }
}

impl FireRedOcr {
    /// Load the full model from a VarBuilder.
    ///
    /// Weight names follow the Qwen3-VL convention since FireRed-OCR
    /// is fine-tuned from Qwen3-VL-2B.
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: FireRedOcrConfig) -> Result<Self> {
        cfg.validate()?;
        let inner = Qwen3VL::load(vb, cfg.base_config.clone())?;
        Ok(Self { inner, config: cfg })
    }

    /// Forward pass: vision features + text token IDs -> OCR logits.
    ///
    /// - `vision_features`: `[B, N_vis, vision_hidden]` pre-encoded vision
    ///   tokens from the document image. Pass `None` for text-only.
    /// - `input_ids`: token IDs for the OCR prompt/context.
    ///
    /// Returns: [`FireRedOcrOutput`] with logits `[B, total_seq_len, vocab_size]`.
    pub fn forward(
        &self,
        vision_features: Option<&DynTensor>,
        input_ids: &[usize],
    ) -> Result<FireRedOcrOutput> {
        let logits = self.inner.forward(vision_features, input_ids)?;
        check_output_finite(&logits, "FireRedOcr")?;
        Ok(FireRedOcrOutput { logits })
    }

    /// Forward pass with KV cache for autoregressive OCR decoding.
    ///
    /// Returns logits for the last token position: `[B, 1, vocab_size]`.
    pub fn forward_cached(
        &self,
        vision_features: Option<&DynTensor>,
        input_ids: &[usize],
        cache: &mut KvCache,
    ) -> Result<FireRedOcrOutput> {
        let logits = self
            .inner
            .forward_cached(vision_features, input_ids, cache)?;
        check_output_finite(&logits, "FireRedOcr.cached")?;
        Ok(FireRedOcrOutput { logits })
    }

    /// Decode OCR token IDs to text.
    ///
    /// Converts a sequence of generated token IDs into a UTF-8 string by
    /// looking up each token in the vocabulary. Stops at the first EOS
    /// token if `eos_token_id` is provided.
    ///
    /// This is a placeholder that returns the token IDs as a
    /// comma-separated string. A real implementation would use the
    /// model's tokenizer vocabulary.
    #[must_use]
    pub fn decode_ocr_tokens(token_ids: &[usize], eos_token_id: Option<usize>) -> String {
        let ids: Vec<usize> = if let Some(eos) = eos_token_id {
            token_ids
                .iter()
                .copied()
                .take_while(|&id| id != eos)
                .collect()
        } else {
            token_ids.to_vec()
        };
        ids.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Create a fresh [`KvCache`] sized for this model.
    #[must_use]
    pub fn create_cache(&self) -> KvCache {
        self.inner.create_cache()
    }

    /// Access the model configuration.
    #[must_use]
    pub fn config(&self) -> &FireRedOcrConfig {
        &self.config
    }

    /// Access the underlying Qwen3-VL model.
    #[must_use]
    pub fn inner(&self) -> &Qwen3VL {
        &self.inner
    }

    /// Access the OCR mode.
    #[must_use]
    pub fn ocr_mode(&self) -> OcrMode {
        self.config.ocr_mode
    }

    /// Access the maximum output token count.
    #[must_use]
    pub fn max_output_tokens(&self) -> usize {
        self.config.max_output_tokens
    }
}

#[cfg(test)]
#[path = "firered_ocr_tests.rs"]
mod tests;
