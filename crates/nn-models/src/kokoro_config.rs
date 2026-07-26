// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`KokoroConfig`] — configuration for Kokoro-82M TTS model.
//!
//! Extracted from `kokoro_tts.rs` for 500-line compliance (#1342).

use crate::kokoro_error::KokoroError;
use crate::plbert::PlbertConfig;

/// Configuration for Kokoro-82M TTS model.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct KokoroConfig {
    /// Encoder dimension (default: 512).
    pub d_en: usize,
    /// Number of prosody predictor blocks (default: 3).
    pub n_prosody_layers: usize,
    /// Style embedding dimension (default: 128, split from 256 voice embedding).
    pub style_dim: usize,
    /// Generator upsample rates (default: [10, 6]).
    pub upsample_rates: Vec<usize>,
    /// Generator upsample kernel sizes (default: [20, 12]).
    pub upsample_kernel_sizes: Vec<usize>,
    /// Generator ResBlock kernel sizes (default: [3, 7, 11]).
    pub resblock_kernel_sizes: Vec<usize>,
    /// Generator ResBlock dilations (default: [[1,3,5], [1,3,5], [1,3,5]]).
    pub resblock_dilations: Vec<Vec<usize>>,
    /// Generator initial channels (default: 512).
    pub gen_initial_channels: usize,
    /// FFT size for iSTFT output (default: 20).
    pub n_fft: usize,
    /// F0/energy predictor BiLSTM hidden size per direction (default: 256).
    pub f0_bilstm_hidden: usize,
    /// Maximum duration bins per phoneme (default: 50).
    ///
    /// Duration is computed as `sigmoid(logits).sum(dim=-1)` over this many
    /// independent Bernoulli bins, producing values in `[0, max_dur]`.
    /// Matches dvoice v0.19's `duration_proj: Linear(d_model, 50)`.
    pub max_dur: usize,
    /// PlBert encoder configuration.
    pub plbert: PlbertConfig,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            d_en: 512,
            n_prosody_layers: 3,
            style_dim: 128,
            upsample_rates: vec![10, 6],
            upsample_kernel_sizes: vec![20, 12],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            gen_initial_channels: 512,
            n_fft: 20,
            f0_bilstm_hidden: 256,
            max_dur: 50,
            plbert: PlbertConfig::default(),
        }
    }
}

impl KokoroConfig {
    /// Create a `KokoroConfig` with default values (Kokoro-82M).
    ///
    /// Required because `#[non_exhaustive]` prevents struct literal
    /// construction outside this crate.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate configuration invariants.
    ///
    /// Checks that all fields satisfy the invariants assumed by the compiled
    /// pipeline. Call at construction time to catch invalid configs early
    /// instead of failing deep inside GPU dispatch. Part of #3004.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.d_en == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "d_en",
                reason: "must be > 0".into(),
            });
        }
        if self.style_dim == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "style_dim",
                reason: "must be > 0".into(),
            });
        }
        if self.max_dur == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "max_dur",
                reason: "must be > 0".into(),
            });
        }
        if self.n_fft == 0 || !self.n_fft.is_multiple_of(4) {
            return Err(KokoroError::InvalidConfig {
                field: "n_fft",
                reason: format!("must be > 0 and divisible by 4, got {}", self.n_fft),
            });
        }
        if self.upsample_rates.is_empty() {
            return Err(KokoroError::InvalidConfig {
                field: "upsample_rates",
                reason: "must be non-empty".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "kokoro_config_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "kokoro_config_arch_tests.rs"]
mod arch_tests;
