// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Neural codec token algebra — verified operations in latent audio space.
//!
//! Neural audio codecs (Encodec, DAC, SpeechTokenizer, Qwen3's 12Hz codec) map
//! audio to discrete token sequences via residual vector quantization (RVQ).
//! Each quantizer level has a codebook of continuous embedding vectors.
//!
//! This module provides arithmetic operations in the continuous embedding space:
//! - **Analogy**: `a - b + c` (voice conversion, emotion transfer)
//! - **Interpolation**: `lerp(a, b, α)` (voice morphing, style blending)
//! - **Centroid computation**: mean embedding across utterances (speaker/emotion identity)
//! - **Quantization**: nearest-neighbor decode back to discrete tokens
//!
//! # Example
//!
//! ```text
//! let space = CodecEmbeddingSpace::from_var_builder(&vb, 8)?;
//! let emb_a = space.embed(&tokens_a)?;
//! let emb_b = space.embed(&tokens_b)?;
//! let blended = space.interpolate(&emb_a, &emb_b, 0.5)?;
//! let tokens_out = space.quantize(&blended)?;
//! ```

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};

use crate::error::{CodecAlgebraKind, TtsVerifyError};

/// Continuous embedding space for residual VQ codec tokens.
///
/// Each quantizer level has a codebook matrix `[vocab_size, embed_dim]`.
/// Token sequences are mapped to continuous embeddings by summing across
/// all RVQ levels (standard residual VQ decoding).
#[derive(Debug, Clone)]
pub struct CodecEmbeddingSpace {
    /// Codebook weight matrices, one per quantizer level.
    /// Each has shape `[vocab_size, embed_dim]`.
    codebooks: Vec<DynTensor>,
    /// Number of residual VQ levels.
    n_levels: usize,
    /// Embedding dimension (columns in each codebook).
    embed_dim: usize,
    /// Vocabulary size per level (rows in each codebook).
    vocab_size: usize,
}

impl CodecEmbeddingSpace {
    /// Load codebook weights from a VarBuilder.
    ///
    /// Expects weights at `quantizer.{level}.codebook.weight` with shape
    /// `[vocab_size, embed_dim]` for each level in `0..n_levels`.
    pub fn from_var_builder(
        vb: impl AsRef<VarBuilder>,
        n_levels: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<Self, TtsVerifyError> {
        let vb = vb.as_ref();
        if n_levels == 0 {
            return Err(TtsVerifyError::CodecAlgebra(
                CodecAlgebraKind::InvalidParam { param: "n_levels" },
            ));
        }
        if vocab_size == 0 || embed_dim == 0 {
            return Err(TtsVerifyError::CodecAlgebra(
                CodecAlgebraKind::InvalidParam {
                    param: "vocab_size/embed_dim",
                },
            ));
        }

        let mut codebooks = Vec::with_capacity(n_levels);
        for level in 0..n_levels {
            let cb_vb = vb.pp(format!("quantizer.{level}.codebook"));
            let weight = cb_vb.get(&[vocab_size, embed_dim], "weight")?;
            codebooks.push(weight);
        }

        Ok(Self {
            codebooks,
            n_levels,
            embed_dim,
            vocab_size,
        })
    }

    /// Create from pre-loaded codebook tensors.
    ///
    /// Each tensor must have shape `[vocab_size, embed_dim]`.
    pub fn from_codebooks(codebooks: Vec<DynTensor>) -> Result<Self, TtsVerifyError> {
        if codebooks.is_empty() {
            return Err(TtsVerifyError::CodecAlgebra(CodecAlgebraKind::EmptyInput {
                what: "at least one codebook required",
            }));
        }

        let first_dims = codebooks[0].dims();
        if first_dims.len() != 2 {
            return Err(TtsVerifyError::CodecAlgebra(
                CodecAlgebraKind::RankMismatch {
                    rank: first_dims.len(),
                },
            ));
        }
        let vocab_size = first_dims[0];
        let embed_dim = first_dims[1];

        for (i, cb) in codebooks.iter().enumerate() {
            let dims = cb.dims();
            if dims != first_dims {
                return Err(TtsVerifyError::CodecAlgebra(
                    CodecAlgebraKind::CodebookShapeMismatch { index: i },
                ));
            }
        }

        let n_levels = codebooks.len();
        Ok(Self {
            codebooks,
            n_levels,
            embed_dim,
            vocab_size,
        })
    }

    /// Number of RVQ levels.
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Embedding dimension per codebook entry.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Vocabulary size per quantizer level.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Look up continuous embedding for a token sequence.
    ///
    /// `tokens` has shape `[n_levels, seq_len]` where each element is a
    /// codebook index in `0..vocab_size`. The embedding is the sum of
    /// per-level codebook lookups (standard residual VQ decoding).
    ///
    /// Returns tensor with shape `[seq_len, embed_dim]`.
    pub fn embed(&self, tokens: &[Vec<u32>]) -> Result<DynTensor, TtsVerifyError> {
        if tokens.len() != self.n_levels {
            return Err(TtsVerifyError::CodecAlgebra(CodecAlgebraKind::LevelCount {
                expected: self.n_levels,
                got: tokens.len(),
            }));
        }

        if tokens[0].is_empty() {
            return Err(TtsVerifyError::CodecAlgebra(CodecAlgebraKind::EmptyInput {
                what: "token sequence is empty",
            }));
        }

        let seq_len = tokens[0].len();
        for (i, level_tokens) in tokens.iter().enumerate() {
            if level_tokens.len() != seq_len {
                return Err(TtsVerifyError::CodecAlgebra(
                    CodecAlgebraKind::SequenceLengthMismatch {
                        level: i,
                        expected: seq_len,
                        got: level_tokens.len(),
                    },
                ));
            }
            for &tok in level_tokens {
                if tok as usize >= self.vocab_size {
                    return Err(TtsVerifyError::CodecAlgebra(
                        CodecAlgebraKind::TokenOutOfRange {
                            token: tok,
                            vocab_size: self.vocab_size,
                        },
                    ));
                }
            }
        }

        let device = self.codebooks[0].device();

        // Sum embeddings across all RVQ levels
        let mut result: Option<DynTensor> = None;
        for (level, level_tokens) in tokens.iter().enumerate() {
            let ids = DynTensor::from_vec_u32(level_tokens.clone(), &[seq_len], &device)?;
            let level_emb = self.codebooks[level].index_select(&ids, 0)?;

            result = Some(match result {
                None => level_emb,
                Some(acc) => acc.add(&level_emb)?,
            });
        }

        result.ok_or_else(|| {
            TtsVerifyError::CodecAlgebra(CodecAlgebraKind::EmptyInput {
                what: "no RVQ levels (should be unreachable)",
            })
        })
    }

    /// Analogy operation: `a - b + c` in embedding space.
    ///
    /// Classic vector arithmetic for attribute transfer:
    /// - Voice conversion: `utterance_A - speaker_A_centroid + speaker_B_centroid`
    /// - Emotion transfer: `neutral_speech - neutral_centroid + happy_centroid`
    pub fn analogy(
        &self,
        a: &DynTensor,
        b: &DynTensor,
        c: &DynTensor,
    ) -> Result<DynTensor, TtsVerifyError> {
        let diff = a.sub(b)?;
        let result = diff.add(c)?;
        Ok(result)
    }

    /// Linear interpolation: `a * (1 - alpha) + b * alpha`.
    ///
    /// Useful for voice morphing and style blending.
    /// `alpha` must be in `[0.0, 1.0]`.
    pub fn interpolate(
        &self,
        a: &DynTensor,
        b: &DynTensor,
        alpha: f32,
    ) -> Result<DynTensor, TtsVerifyError> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(TtsVerifyError::CodecAlgebra(
                CodecAlgebraKind::AlphaOutOfRange { alpha },
            ));
        }
        let a_scaled = a.mul_scalar(f64::from(1.0 - alpha))?;
        let b_scaled = b.mul_scalar(f64::from(alpha))?;
        let result = a_scaled.add(&b_scaled)?;
        Ok(result)
    }

    /// Nearest-neighbor quantize continuous embedding back to discrete tokens.
    ///
    /// For each position in the embedding sequence, finds the closest codebook
    /// entry at each RVQ level using greedy residual quantization.
    ///
    /// Input shape: `[seq_len, embed_dim]`.
    /// Returns `[n_levels][seq_len]` token indices.
    pub fn quantize(&self, embedding: &DynTensor) -> Result<Vec<Vec<u32>>, TtsVerifyError> {
        let dims = embedding.dims();
        if dims.len() != 2 || dims[1] != self.embed_dim {
            return Err(TtsVerifyError::CodecAlgebra(
                CodecAlgebraKind::EmbeddingShape {
                    expected_dim: self.embed_dim,
                },
            ));
        }

        let seq_len = dims[0];
        let mut residual = embedding.clone();
        let mut all_tokens = Vec::with_capacity(self.n_levels);

        for level in 0..self.n_levels {
            // Compute distances: ||residual - codebook||^2 for each entry
            // = ||r||^2 - 2*r@cb^T + ||cb||^2
            // We only need argmin, so we can use -2*r@cb^T + ||cb||^2
            let cb = &self.codebooks[level];

            // cb_t: [embed_dim, vocab_size]
            let cb_t = cb.transpose(0, 1)?;

            // sim: [seq_len, vocab_size] = residual @ cb^T
            let sim = residual.matmul(&cb_t)?;

            // cb_norm_sq: [vocab_size] = sum(cb^2, dim=1)
            let cb_sq = cb.sqr()?;
            let cb_norm_sq = cb_sq.sum_keepdim(1)?;
            // Squeeze to [vocab_size]
            let cb_norm_sq = cb_norm_sq.squeeze(1)?;

            // dist: [seq_len, vocab_size] = -2*sim + cb_norm_sq (broadcast)
            let neg_two_sim = sim.mul_scalar(-2.0)?;
            let dist = neg_two_sim.add(&cb_norm_sq)?;

            // argmin along vocab dimension (dim=1) → nearest codebook entry
            // Use topk with k=1 on negated distances (topk returns largest)
            let neg_dist = dist.neg()?;
            let (_, indices) = neg_dist.topk(1, 1)?;

            // indices: [seq_len, 1] → flatten to [seq_len]
            let indices_flat = indices.squeeze(1)?;

            // Convert to u32 token IDs
            let token_ids = indices_flat.to_dtype(DType::U32)?;
            let token_vec = token_ids.to_flat_vec::<u32>()?;
            all_tokens.push(token_vec);

            // Update residual: residual -= codebook[chosen_idx]
            let chosen_emb = cb.index_select(&token_ids, 0)?;
            residual = residual.sub(&chosen_emb)?;
        }

        // Verify output lengths
        for (i, tokens) in all_tokens.iter().enumerate() {
            if tokens.len() != seq_len {
                return Err(TtsVerifyError::CodecAlgebra(
                    CodecAlgebraKind::SequenceLengthMismatch {
                        level: i,
                        expected: seq_len,
                        got: tokens.len(),
                    },
                ));
            }
        }

        Ok(all_tokens)
    }
}

/// Compute centroid (mean) embedding for a set of utterances.
///
/// Each utterance is a token matrix `[n_levels, seq_len]`. Embeddings are
/// computed per-utterance and averaged across all positions and utterances.
///
/// Returns a single embedding vector with shape `[1, embed_dim]`.
pub fn utterance_centroid(
    space: &CodecEmbeddingSpace,
    utterance_tokens: &[Vec<Vec<u32>>],
) -> Result<DynTensor, TtsVerifyError> {
    if utterance_tokens.is_empty() {
        return Err(TtsVerifyError::CodecAlgebra(CodecAlgebraKind::EmptyInput {
            what: "at least one utterance required for centroid",
        }));
    }

    let device = Device::Cpu;
    let mut sum = DynTensor::zeros(&[1, space.embed_dim()], DType::F32, &device)?;
    let mut total_frames: usize = 0;

    for utterance in utterance_tokens {
        let emb = space.embed(utterance)?;
        // emb: [seq_len, embed_dim]
        let seq_len = emb.dims()[0];
        // Sum across time positions
        let utt_sum = emb.sum_keepdim(0)?;
        // utt_sum: [1, embed_dim]
        sum = sum.add(&utt_sum)?;
        total_frames += seq_len;
    }

    if total_frames == 0 {
        return Err(TtsVerifyError::CodecAlgebra(CodecAlgebraKind::EmptyInput {
            what: "zero total frames across all utterances",
        }));
    }

    let centroid = sum.div_scalar(total_frames as f64)?;
    Ok(centroid)
}

/// Compute speaker centroid from a set of utterances by one speaker.
///
/// Alias for [`utterance_centroid`] with speaker-specific semantics.
pub fn speaker_centroid(
    space: &CodecEmbeddingSpace,
    utterance_tokens: &[Vec<Vec<u32>>],
) -> Result<DynTensor, TtsVerifyError> {
    utterance_centroid(space, utterance_tokens)
}

/// Compute emotion centroid from labeled utterances with the same emotion.
///
/// Alias for [`utterance_centroid`] with emotion-specific semantics.
pub fn emotion_centroid(
    space: &CodecEmbeddingSpace,
    utterance_tokens: &[Vec<Vec<u32>>],
) -> Result<DynTensor, TtsVerifyError> {
    utterance_centroid(space, utterance_tokens)
}

#[cfg(test)]
#[path = "codec_algebra_tests.rs"]
mod tests;
