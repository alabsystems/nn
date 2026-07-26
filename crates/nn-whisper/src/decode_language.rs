// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Language detection from Whisper decoder logits.
//!
//! Implements the single-step language identification used by AI Provider Whisper:
//! feed `[SOT]` token, extract logits, argmax over language token range.

use crate::tokenizer::{LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN, SOT_TOKEN};
use crate::WhisperError;
use crate::WhisperModel;

use super::helpers::{argmax_f32, check_logit_finiteness};

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, D};

/// Result of language detection.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LanguageDetectionResult {
    /// Detected language token ID (in the range `LANGUAGE_TOKEN_START..=LANGUAGE_TOKEN_END`).
    pub language_token: usize,
    /// Probability of the detected language (softmax over language tokens).
    pub probability: f64,
    /// Probability of no speech at the SOT position.
    pub no_speech_prob: f64,
}

/// Detect the spoken language from a single SOT decode step.
///
/// Feeds `[SOT]` as the initial token, extracts logits at that position,
/// and returns the argmax over the language token range (50259-50358).
///
/// This matches AI Provider Whisper's `detect_language()` which uses the same
/// single-step logit extraction approach.
pub fn detect_language(
    model: &mut WhisperModel,
    encoder_output: &DynTensor,
) -> Result<LanguageDetectionResult> {
    let device = encoder_output.device();
    model.reset_kv_cache();

    // Feed [SOT] token to the decoder.
    let sot_u32 = vec![SOT_TOKEN as u32];
    let sot_tensor = DynTensor::from_vec_u32(sot_u32, &[1, 1], &device)?;
    let logits = model.decode(&sot_tensor, encoder_output, true, 0)?;
    check_logit_finiteness(&logits, 0)?;

    let vocab_size = logits.dim(D::Minus1)?;
    let logits_view = logits.to_f32_array()?;
    let logits_contiguous = logits_view.as_standard_layout();
    let flat = logits_contiguous.as_slice().ok_or_else(|| {
        nn_core::TensorError::InvalidShape("logits not contiguous after as_standard_layout".into())
    })?;
    let offset = flat.len().checked_sub(vocab_size).ok_or_else(|| {
        nn_core::TensorError::from(WhisperError::LogitTooSmall {
            logit_len: flat.len(),
            vocab_size,
        })
    })?;
    let last_logits = &flat[offset..];

    // Extract language logits and compute softmax over language tokens.
    let lang_end = LANGUAGE_TOKEN_END + 1; // exclusive end
    if lang_end > vocab_size || LANGUAGE_TOKEN_START >= vocab_size {
        return Err(WhisperError::LanguageTokenRange {
            start: LANGUAGE_TOKEN_START,
            end: lang_end,
            vocab_size,
        }
        .into());
    }

    let lang_logits = &last_logits[LANGUAGE_TOKEN_START..lang_end];
    let max_val = lang_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = lang_logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp.iter().sum();

    let (best_idx, best_prob) = if sum.is_finite() && sum > 0.0 {
        let probs: Vec<f32> = exp.iter().map(|&v| v / sum).collect();
        let idx = argmax_f32(&probs);
        (idx, f64::from(probs[idx]))
    } else {
        (0, 0.0)
    };

    let no_speech = compute_no_speech_prob(last_logits);

    Ok(LanguageDetectionResult {
        language_token: LANGUAGE_TOKEN_START + best_idx,
        probability: best_prob,
        no_speech_prob: no_speech,
    })
}

/// Compute the no-speech probability from a logit slice.
///
/// Returns `softmax(logits)[NO_SPEECH_TOKEN]`. If `NO_SPEECH_TOKEN` is beyond
/// the vocabulary or the softmax sum is not finite, returns 0.0.
pub(super) fn compute_no_speech_prob(logits: &[f32]) -> f64 {
    if NO_SPEECH_TOKEN >= logits.len() {
        return 0.0;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        return 0.0;
    }
    let exp: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if !sum.is_finite() || sum == 0.0 {
        return 0.0;
    }
    f64::from(exp[NO_SPEECH_TOKEN] / sum)
}
