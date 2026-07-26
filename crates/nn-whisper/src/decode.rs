// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper decode loop: greedy + temperature fallback.
//!
//! Implements the autoregressive decode strategy used by dvoice-stt:
//! greedy argmax with temperature fallback, token suppression, and quality checks.

#[path = "decode_helpers.rs"]
mod helpers;

#[path = "decode_language.rs"]
mod language;
pub use language::{detect_language, LanguageDetectionResult};

use crate::tokenizer::{WhisperTokenizer, EOT_TOKEN};
use crate::WhisperModel;
// Re-export helpers at pub(crate) so test submodules can access via `use crate::decode::*`.
pub(crate) use helpers::{apply_suppression_inplace, check_logit_finiteness, sample_token};
// argmax_f32 and compute_log_prob are used by test and kani submodules only.
#[cfg(any(test, kani))]
pub(crate) use helpers::{argmax_f32, compute_log_prob};

use crate::WhisperError;
use language::compute_no_speech_prob;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, TensorError, D};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Maximum decode length (tokens).
pub const MAX_DECODE_LENGTH: usize = 224;

/// Default compression ratio threshold.
pub const DEFAULT_COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

/// Default average log-probability threshold.
pub const DEFAULT_AVG_LOGPROB_THRESHOLD: f64 = -1.0;

/// Default temperature fallback sequence.
pub const DEFAULT_TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

/// Result of a decode pass.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodingResult {
    /// Decoded token IDs (excluding initial prompt tokens).
    pub tokens: Vec<usize>,
    /// Average log-probability of the decoded tokens.
    pub avg_logprob: f64,
    /// Compression ratio of the decoded text (estimated from token repetition).
    pub compression_ratio: f64,
    /// Whether the EOT token was reached.
    pub reached_eot: bool,
    /// Temperature used for this result.
    pub temperature: f64,
    /// Probability of no speech at the SOT position.
    ///
    /// Computed as `softmax(logits)[NO_SPEECH_TOKEN]` at the first decode step.
    /// When this exceeds a threshold (typically 0.6), the segment likely contains
    /// no speech and can be skipped.
    pub no_speech_prob: f64,
}

impl DecodingResult {
    /// Create a new decoding result.
    #[must_use]
    pub fn new(
        tokens: Vec<usize>,
        avg_logprob: f64,
        compression_ratio: f64,
        reached_eot: bool,
        temperature: f64,
        no_speech_prob: f64,
    ) -> Self {
        Self {
            tokens,
            avg_logprob,
            compression_ratio,
            reached_eot,
            temperature,
            no_speech_prob,
        }
    }
}

/// Configuration for the decode loop.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodeConfig {
    /// Maximum number of tokens to generate.
    pub max_length: usize,
    /// Compression ratio threshold for quality check.
    pub compression_ratio_threshold: f64,
    /// Average log-probability threshold for quality check.
    pub avg_logprob_threshold: f64,
    /// Token IDs to suppress during generation.
    pub suppress_tokens: Vec<usize>,
    /// Initial prompt token IDs (e.g., `[50258, 50259, 50360, 50364]` for English transcribe).
    pub initial_tokens: Vec<usize>,
    /// Random seed for temperature sampling. When `Some`, tokens are sampled
    /// from the categorical distribution at temperature > 0. When `None`,
    /// argmax is used regardless of temperature (greedy behavior).
    pub seed: Option<u64>,
}

// DecodeConfig Default, builder methods, and validation extracted to decode_config.rs.
#[path = "decode_config.rs"]
mod decode_config;

/// Check whether a decoding result passes quality thresholds.
#[must_use]
pub fn passes_quality_check(result: &DecodingResult, config: &DecodeConfig) -> bool {
    result.compression_ratio <= config.compression_ratio_threshold
        && result.avg_logprob >= config.avg_logprob_threshold
}

/// Estimate compression ratio from token sequence.
///
/// Uses a simple bigram-based repetition metric: the ratio of total tokens
/// to unique consecutive bigrams. Higher ratio = more repetitive.
#[must_use]
pub fn compression_ratio(tokens: &[usize]) -> f64 {
    if tokens.len() < 2 {
        return 1.0;
    }
    let mut bigrams = std::collections::HashSet::new();
    for pair in tokens.windows(2) {
        bigrams.insert((pair[0], pair[1]));
    }
    // Ratio: total bigram slots / unique bigrams. More repetitive = higher ratio.
    (tokens.len() - 1) as f64 / bigrams.len().max(1) as f64
}

/// Greedy decode: pure argmax at each step.
///
/// Resets the KV cache before decoding so consecutive calls on different
/// audio segments produce independent results.
///
/// Returns the decoded tokens and quality metrics.
pub fn greedy_decode(
    model: &mut WhisperModel,
    encoder_output: &DynTensor,
    config: &DecodeConfig,
) -> Result<DecodingResult> {
    model.reset_kv_cache();
    decode_with_temperature(model, encoder_output, config, 0.0)
}

/// Decode with a specific temperature.
///
/// Temperature must be finite and non-negative. At temperature 0.0 (or very
/// small), uses greedy argmax. At positive temperature with a configured seed,
/// samples from the categorical distribution.
pub fn decode_with_temperature(
    model: &mut WhisperModel,
    encoder_output: &DynTensor,
    config: &DecodeConfig,
    temperature: f64,
) -> Result<DecodingResult> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(WhisperError::InvalidTemperature { temperature }.into());
    }
    config.validate()?;
    let device = encoder_output.device();
    let mut rng = config.seed.map(StdRng::seed_from_u64);
    let mut all_tokens = config.initial_tokens.clone();
    let mut decoded_tokens = Vec::new();
    let mut sum_log_prob = 0.0_f64;
    let mut reached_eot = false;

    // First step: feed all initial tokens at once.
    // Use U32 dtype for token IDs — f32 loses precision for IDs > 2^24.
    // Validate all token IDs fit in u32 (they come from user-provided config).
    if let Some(&t) = all_tokens.iter().find(|&&t| t > u32::MAX as usize) {
        return Err(WhisperError::TokenIdOverflow { token_id: t }.into());
    }
    let initial_u32: Vec<u32> = all_tokens.iter().map(|&t| t as u32).collect();
    let seq_len = initial_u32.len();
    let tokens_tensor = DynTensor::from_vec_u32(initial_u32, &[1, seq_len], &device)?;
    let logits = model.decode(&tokens_tensor, encoder_output, true, 0)?;
    check_logit_finiteness(&logits, 0)?;

    // Extract logits to CPU once, apply suppression in-place, then sample.
    // This avoids the CPU→GPU→CPU round-trip that would occur if we
    // reconstructed a DynTensor after suppression just to extract it again.
    let vocab_size = logits.dim(D::Minus1)?;
    let logits_view = logits.to_f32_array()?;
    let logits_contiguous = logits_view.as_standard_layout();
    let flat = logits_contiguous.as_slice().ok_or_else(|| {
        TensorError::InvalidShape("logits not contiguous after as_standard_layout".into())
    })?;
    let offset = flat.len().checked_sub(vocab_size).ok_or_else(|| {
        TensorError::from(WhisperError::LogitTooSmall {
            logit_len: flat.len(),
            vocab_size,
        })
    })?;

    // Compute no-speech probability before suppression modifies the logits.
    let no_speech_prob = compute_no_speech_prob(&flat[offset..]);

    let mut suppressed = flat[offset..].to_vec();
    apply_suppression_inplace(&mut suppressed, &config.suppress_tokens);
    let (next_token, log_prob) = sample_token(&suppressed, temperature, rng.as_mut());

    if next_token == EOT_TOKEN {
        reached_eot = true;
    } else {
        decoded_tokens.push(next_token);
        sum_log_prob += f64::from(log_prob);
        all_tokens.push(next_token);
    }

    // Autoregressive loop: feed one token at a time.
    let mut step = 1;
    while !reached_eot && step < config.max_length {
        let last_tok = *all_tokens.last().unwrap_or(&0);
        let token_u32 = [u32::try_from(last_tok).map_err(|_| {
            TensorError::from(WhisperError::TokenIdOverflow { token_id: last_tok })
        })?];
        let token_tensor = DynTensor::from_vec_u32(token_u32.to_vec(), &[1, 1], &device)?;
        let position_offset = all_tokens.len() - 1;
        let logits = model.decode(&token_tensor, encoder_output, false, position_offset)?;
        check_logit_finiteness(&logits, step)?;

        let vocab_size = logits.dim(D::Minus1)?;
        let logits_view = logits.to_f32_array()?;
        let logits_contiguous = logits_view.as_standard_layout();
        let flat = logits_contiguous.as_slice().ok_or_else(|| {
            TensorError::InvalidShape("logits not contiguous after as_standard_layout".into())
        })?;
        let offset = flat.len().checked_sub(vocab_size).ok_or_else(|| {
            TensorError::from(WhisperError::LogitTooSmall {
                logit_len: flat.len(),
                vocab_size,
            })
        })?;
        let mut suppressed = flat[offset..].to_vec();
        apply_suppression_inplace(&mut suppressed, &config.suppress_tokens);
        let (next_token, log_prob) = sample_token(&suppressed, temperature, rng.as_mut());

        if next_token == EOT_TOKEN {
            reached_eot = true;
        } else {
            decoded_tokens.push(next_token);
            sum_log_prob += f64::from(log_prob);
            all_tokens.push(next_token);
        }
        step += 1;
    }

    let avg_logprob = if decoded_tokens.is_empty() {
        0.0
    } else {
        sum_log_prob / decoded_tokens.len() as f64
    };

    let cr = compression_ratio(&decoded_tokens);

    Ok(DecodingResult {
        tokens: decoded_tokens,
        avg_logprob,
        compression_ratio: cr,
        reached_eot,
        temperature,
        no_speech_prob,
    })
}

/// Temperature fallback decode.
///
/// Tries each temperature in sequence. Returns the first result that passes
/// quality checks, or the last result if none pass.
///
/// Returns an error if `temperatures` is empty.
pub fn temperature_fallback_decode(
    model: &mut WhisperModel,
    encoder_output: &DynTensor,
    config: &DecodeConfig,
    temperatures: &[f64],
) -> Result<DecodingResult> {
    if temperatures.is_empty() {
        return Err(WhisperError::EmptyDecodeResult {
            reason: "temperatures must not be empty",
        }
        .into());
    }
    let mut best_result = None;

    for &temp in temperatures {
        model.reset_kv_cache();
        let result = decode_with_temperature(model, encoder_output, config, temp)?;

        if passes_quality_check(&result, config) {
            return Ok(result);
        }

        best_result = Some(result);
    }

    // Return the last result if no temperature passed quality checks.
    // `best_result` is always `Some` here because `temperatures` is non-empty,
    // but we use `ok_or` instead of `.expect()` to avoid panicking in production.
    best_result.ok_or_else(|| {
        WhisperError::EmptyDecodeResult {
            reason: "no temperature produced a result",
        }
        .into()
    })
}

/// Result of a transcription (decode + detokenize).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TranscriptionResult {
    /// The transcribed text.
    pub text: String,
    /// The underlying decode result with token-level details.
    pub decode_result: DecodingResult,
    /// Probability of no speech at the SOT position (shortcut from decode_result).
    pub no_speech_prob: f64,
}

/// Transcribe: decode tokens from encoder output and convert to text.
///
/// Combines `greedy_decode()` with tokenizer text conversion. This is the
/// primary convenience function for speech-to-text inference.
///
/// # Arguments
///
/// * `model` - The Whisper model (encoder output is provided separately).
/// * `encoder_output` - Output from `model.encode(mel)`.
/// * `config` - Decode configuration (max length, suppression, etc.).
/// * `tokenizer` - Tokenizer for converting token IDs to text.
pub fn transcribe(
    model: &mut WhisperModel,
    encoder_output: &DynTensor,
    config: &DecodeConfig,
    tokenizer: &WhisperTokenizer,
) -> Result<TranscriptionResult> {
    let decode_result = greedy_decode(model, encoder_output, config)?;
    let text = tokenizer.decode(&decode_result.tokens)?;
    let no_speech_prob = decode_result.no_speech_prob;

    Ok(TranscriptionResult {
        text,
        decode_result,
        no_speech_prob,
    })
}

/// Transcribe with temperature fallback and text conversion.
///
/// Combines `temperature_fallback_decode()` with tokenizer text conversion.
/// Tries each temperature in the default sequence, returning the first result
/// that passes quality checks.
pub fn transcribe_with_fallback(
    model: &mut WhisperModel,
    encoder_output: &DynTensor,
    config: &DecodeConfig,
    tokenizer: &WhisperTokenizer,
    temperatures: &[f64],
) -> Result<TranscriptionResult> {
    let decode_result = temperature_fallback_decode(model, encoder_output, config, temperatures)?;
    let text = tokenizer.decode(&decode_result.tokens)?;
    let no_speech_prob = decode_result.no_speech_prob;

    Ok(TranscriptionResult {
        text,
        decode_result,
        no_speech_prob,
    })
}

#[path = "decode_beam.rs"]
mod beam;
pub use beam::{beam_search_decode, WhisperBeamConfig};

#[path = "decode_long.rs"]
mod long;
pub use long::{transcribe_long, LongFormConfig, LongFormResult, LongFormSegment};

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kani_decode_proofs.rs"]
mod kani_decode_proofs;

#[cfg(kani)]
#[path = "kani_decode_beam_proofs.rs"]
mod kani_decode_beam_proofs;

#[cfg(kani)]
#[path = "kani_decode_proofs_ext.rs"]
mod kani_decode_proofs_ext;

#[cfg(kani)]
#[path = "kani_decode_beam_proofs_ext.rs"]
mod kani_decode_beam_proofs_ext;
