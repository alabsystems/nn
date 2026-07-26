// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper-specific error types.
//!
//! Structured errors for configuration validation, audio preprocessing,
//! tokenization, decoding, weight loading, and attention. All variants
//! convert to `TensorError` via `From` for backward compatibility.

use nn_core::{BackendDomain, BackendErrorKind, TensorError};
use thiserror::Error;

/// Whisper-specific error type with structured variants.
///
/// Each domain category has typed sub-variants for programmatic matching.
/// The original `{ reason: String }` variants are retained for edge cases
/// not covered by structured variants.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WhisperError {
    // -- Config validation ------------------------------------------------
    /// A required config field is zero.
    #[error("invalid config: {field} must be > 0")]
    ZeroConfigField { field: &'static str },

    /// Two config fields fail a divisibility requirement.
    #[error("invalid config: {a_name} ({a_val}) must be divisible by {b_name} ({b_val})")]
    ConfigNotDivisible {
        a_name: &'static str,
        a_val: usize,
        b_name: &'static str,
        b_val: usize,
    },

    /// A config field that must be finite is NaN or Inf.
    #[error("invalid config: {field} must be finite, got {value}")]
    NonFiniteConfigField { field: &'static str, value: f64 },

    /// A config field that must be non-empty is empty.
    #[error("invalid config: {field} must not be empty")]
    EmptyConfigField { field: &'static str },

    /// A config field exceeds a maximum limit.
    #[error("invalid config: {field} ({value}) exceeds limit ({limit})")]
    ConfigExceedsLimit {
        field: &'static str,
        value: usize,
        limit: usize,
    },

    /// Catch-all for configuration errors not covered by structured variants.
    #[error("invalid config: {reason}")]
    InvalidConfig { reason: String },

    // -- Audio preprocessing ----------------------------------------------
    /// Input audio is empty.
    #[error("audio format: {stage}: audio is empty")]
    EmptyAudio { stage: &'static str },

    /// Catch-all for audio format errors.
    #[error("audio format: {reason}")]
    AudioFormat { reason: String },

    // -- Tokenizer --------------------------------------------------------
    /// Vocabulary JSON parsing failed.
    #[error("tokenizer: vocab JSON parse error: {detail}")]
    VocabParseError { detail: String },

    /// Token ID is out of vocabulary range.
    #[error("tokenizer: token ID {id} out of vocabulary range (vocab_size={vocab_size})")]
    TokenOutOfRange { id: usize, vocab_size: usize },

    /// UTF-8 decoding of token bytes failed.
    #[error("tokenizer: UTF-8 decode error: {detail}")]
    Utf8DecodeError { detail: String },

    /// BPE merge file has a malformed line.
    #[error("tokenizer: merges.txt line {line}: {detail}")]
    MergeParseError { line: usize, detail: &'static str },

    /// BPE encoding requires merges but none were loaded.
    #[error("tokenizer: encode requires BPE merges (use from_vocab_and_merges)")]
    MissingMerges,

    /// A BPE token is not in the vocabulary.
    #[error("tokenizer: BPE token {token:?} not in vocabulary")]
    TokenNotInVocab { token: String },

    /// Catch-all for tokenizer errors.
    #[error("tokenizer: {reason}")]
    Tokenizer { reason: String },

    // -- Decoding ---------------------------------------------------------
    /// Token ID exceeds `u32::MAX` (required for tensor representation).
    #[error("decode: token ID {token_id} exceeds u32::MAX")]
    TokenIdOverflow { token_id: usize },

    /// Logit tensor is too small for the vocabulary.
    #[error("decode: logit length {logit_len} < vocab_size {vocab_size}")]
    LogitTooSmall { logit_len: usize, vocab_size: usize },

    /// Temperature is non-finite or negative.
    #[error("decode: temperature must be finite and non-negative, got {temperature}")]
    InvalidTemperature { temperature: f64 },

    /// Decode or beam search produced no results.
    #[error("decode: {reason}")]
    EmptyDecodeResult { reason: &'static str },

    /// Position offset + sequence length overflows usize.
    #[error("decode: position_offset ({offset}) + seq_len ({seq_len}) overflows usize")]
    PositionOverflow { offset: usize, seq_len: usize },

    /// Language token range exceeds vocabulary.
    #[error("decode: language token range [{start}..{end}) exceeds vocab_size {vocab_size}")]
    LanguageTokenRange {
        start: usize,
        end: usize,
        vocab_size: usize,
    },

    /// Catch-all for decode errors.
    #[error("decode: {reason}")]
    Decode { reason: String },

    // -- Weight loading ---------------------------------------------------
    /// Tensor byte data is not aligned to the required element size.
    #[error(
        "weight load: tensor '{tensor_name}': byte length {byte_len} not aligned to {alignment}"
    )]
    ByteAlignment {
        tensor_name: String,
        byte_len: usize,
        alignment: usize,
    },

    /// Safetensors file parsing failed.
    #[error("weight load: safetensors parse: {detail}")]
    SafetensorsParseError { detail: String },

    /// Weight tensor contains NaN or Inf values.
    #[error("weight load: tensor '{tensor_name}': {count} non-finite values (NaN/Inf)")]
    NonFiniteWeight { tensor_name: String, count: usize },

    /// Catch-all for weight load errors.
    #[error("weight load: {reason}")]
    WeightLoad { reason: String },

    // -- Attention --------------------------------------------------------
    /// Encoder and decoder batch sizes don't match in cross-attention.
    #[error(
        "attention: encoder batch size ({encoder_batch}) != decoder batch size ({decoder_batch})"
    )]
    BatchMismatch {
        encoder_batch: usize,
        decoder_batch: usize,
    },

    /// Cached KV sequence length doesn't match encoder output.
    #[error("attention: cross-attention KV cache seq_len ({cached_seq}) != encoder_output seq_len ({encoder_seq})")]
    CacheSeqMismatch {
        cached_seq: usize,
        encoder_seq: usize,
    },

    /// Catch-all for attention errors.
    #[error("attention: {reason}")]
    Attention { reason: String },

    // -- Passthrough ------------------------------------------------------
    /// Passthrough for underlying tensor errors.
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

impl From<WhisperError> for TensorError {
    fn from(e: WhisperError) -> Self {
        match e {
            WhisperError::Tensor(te) => te,
            other => {
                let msg = other.to_string();
                Self::backend_failure_with_source(
                    BackendDomain::Whisper,
                    BackendErrorKind::Other,
                    msg,
                    other,
                )
            }
        }
    }
}
