// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `DecodeConfig` builder methods and validation.
//!
//! Extracted from `decode.rs` for the 500-line file limit.

use crate::WhisperError;
use nn_core::Result;

use super::{
    DecodeConfig, DEFAULT_AVG_LOGPROB_THRESHOLD, DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    MAX_DECODE_LENGTH,
};

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            max_length: MAX_DECODE_LENGTH,
            compression_ratio_threshold: DEFAULT_COMPRESSION_RATIO_THRESHOLD,
            avg_logprob_threshold: DEFAULT_AVG_LOGPROB_THRESHOLD,
            suppress_tokens: Vec::new(),
            initial_tokens: vec![50258, 50259, 50360, 50364],
            seed: None,
        }
    }
}

impl DecodeConfig {
    /// Set the maximum number of tokens to generate.
    #[must_use]
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set the initial prompt token IDs.
    #[must_use]
    pub fn with_initial_tokens(mut self, tokens: Vec<usize>) -> Self {
        self.initial_tokens = tokens;
        self
    }

    /// Set the token IDs to suppress during generation.
    #[must_use]
    pub fn with_suppress_tokens(mut self, tokens: Vec<usize>) -> Self {
        self.suppress_tokens = tokens;
        self
    }

    /// Set the random seed for temperature sampling.
    #[must_use]
    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }

    /// Set the compression ratio threshold.
    #[must_use]
    pub fn with_compression_ratio_threshold(mut self, threshold: f64) -> Self {
        self.compression_ratio_threshold = threshold;
        self
    }

    /// Set the average log-probability threshold.
    #[must_use]
    pub fn with_avg_logprob_threshold(mut self, threshold: f64) -> Self {
        self.avg_logprob_threshold = threshold;
        self
    }

    /// Validate configuration parameters.
    ///
    /// Rejects NaN/Inf thresholds and empty initial tokens. Called by
    /// `decode_with_temperature()` at entry. Matches `WhisperBeamConfig::validate()`
    /// and `WhisperConfig::validate()` patterns.
    pub fn validate(&self) -> Result<()> {
        if self.max_length == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "max_length",
            }
            .into());
        }
        if self.max_length > MAX_DECODE_LENGTH {
            return Err(WhisperError::ConfigExceedsLimit {
                field: "max_length",
                value: self.max_length,
                limit: MAX_DECODE_LENGTH,
            }
            .into());
        }
        if !self.compression_ratio_threshold.is_finite() {
            return Err(WhisperError::NonFiniteConfigField {
                field: "compression_ratio_threshold",
                value: self.compression_ratio_threshold,
            }
            .into());
        }
        if !self.avg_logprob_threshold.is_finite() {
            return Err(WhisperError::NonFiniteConfigField {
                field: "avg_logprob_threshold",
                value: self.avg_logprob_threshold,
            }
            .into());
        }
        if self.initial_tokens.is_empty() {
            return Err(WhisperError::EmptyConfigField {
                field: "initial_tokens",
            }
            .into());
        }
        Ok(())
    }
}
