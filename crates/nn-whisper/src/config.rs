// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper model configuration.
//!
//! Matches AI Provider Whisper / candle-transformers config fields.

/// Audio constants matching AI Provider Whisper.
pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;
pub const N_SAMPLES: usize = SAMPLE_RATE * CHUNK_LENGTH; // 480_000
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3_000
/// Default mel bins for Whisper large-v3 / large-v3-turbo.
pub const NUM_MEL_BINS: usize = 128;

/// Whisper model configuration.
///
/// Fields match `AI Provider/whisper-large-v3-turbo`'s `config.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WhisperConfig {
    /// Number of mel frequency bins. Default: 128.
    pub num_mel_bins: usize,
    /// Maximum source (audio) positions after Conv1d stride-2 downsampling. Default: 1500.
    pub max_source_positions: usize,
    /// Model hidden dimension. Default: 1280.
    pub d_model: usize,
    /// Number of encoder attention heads. Default: 20.
    pub encoder_attention_heads: usize,
    /// Number of encoder layers. Default: 32 (full, not distilled in turbo).
    pub encoder_layers: usize,
    /// Encoder FFN intermediate dimension. Default: 5120 (4x d_model).
    pub encoder_ffn_dim: usize,
    /// Vocabulary size. Default: 51866.
    pub vocab_size: usize,
    /// Maximum target (token) positions. Default: 448.
    pub max_target_positions: usize,
    /// Number of decoder attention heads. Default: 20.
    pub decoder_attention_heads: usize,
    /// Number of decoder layers. Default: 4 (distilled in turbo).
    pub decoder_layers: usize,
    /// Decoder FFN intermediate dimension. Default: 5120.
    pub decoder_ffn_dim: usize,
}

impl WhisperConfig {
    /// Validate configuration invariants.
    ///
    /// Prevents division-by-zero panics in `encoder_head_dim()` and
    /// `decoder_head_dim()`, and rejects nonsensical zero-size configs.
    ///
    /// Called automatically by [`WhisperModel::load()`](crate::WhisperModel::load).
    pub fn validate(&self) -> nn_core::Result<()> {
        use crate::WhisperError;

        if self.d_model == 0 {
            return Err(WhisperError::ZeroConfigField { field: "d_model" }.into());
        }
        if self.encoder_attention_heads == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "encoder_attention_heads",
            }
            .into());
        }
        if self.decoder_attention_heads == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "decoder_attention_heads",
            }
            .into());
        }
        if !self.d_model.is_multiple_of(self.encoder_attention_heads) {
            return Err(WhisperError::ConfigNotDivisible {
                a_name: "d_model",
                a_val: self.d_model,
                b_name: "encoder_attention_heads",
                b_val: self.encoder_attention_heads,
            }
            .into());
        }
        if !self.d_model.is_multiple_of(self.decoder_attention_heads) {
            return Err(WhisperError::ConfigNotDivisible {
                a_name: "d_model",
                a_val: self.d_model,
                b_name: "decoder_attention_heads",
                b_val: self.decoder_attention_heads,
            }
            .into());
        }
        if self.vocab_size == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "vocab_size",
            }
            .into());
        }
        if self.num_mel_bins == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "num_mel_bins",
            }
            .into());
        }
        if self.encoder_ffn_dim == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "encoder_ffn_dim",
            }
            .into());
        }
        if self.decoder_ffn_dim == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "decoder_ffn_dim",
            }
            .into());
        }
        if self.max_source_positions == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "max_source_positions",
            }
            .into());
        }
        if self.max_target_positions == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "max_target_positions",
            }
            .into());
        }
        Ok(())
    }

    /// Configuration for `whisper-large-v3-turbo` (4-layer distilled decoder).
    #[must_use]
    pub fn large_v3_turbo() -> Self {
        Self {
            num_mel_bins: 128,
            max_source_positions: 1500,
            d_model: 1280,
            encoder_attention_heads: 20,
            encoder_layers: 32,
            encoder_ffn_dim: 5120,
            vocab_size: 51866,
            max_target_positions: 448,
            decoder_attention_heads: 20,
            decoder_layers: 4,
            decoder_ffn_dim: 5120,
        }
    }

    /// Configuration for `whisper-tiny` (39M params, 4+4 layers).
    ///
    /// Matches HuggingFace `AI Provider/whisper-tiny` config.json.
    /// Used for CI-friendly real-weight parity testing (~150 MB weights).
    #[must_use]
    pub fn whisper_tiny() -> Self {
        Self {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 384,
            encoder_attention_heads: 6,
            encoder_layers: 4,
            encoder_ffn_dim: 1536,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 6,
            decoder_layers: 4,
            decoder_ffn_dim: 1536,
        }
    }

    /// Configuration for `whisper-base` (74M params, 6+6 layers).
    #[must_use]
    pub fn whisper_base() -> Self {
        Self {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 512,
            encoder_attention_heads: 8,
            encoder_layers: 6,
            encoder_ffn_dim: 2048,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 8,
            decoder_layers: 6,
            decoder_ffn_dim: 2048,
        }
    }

    /// Configuration for `whisper-small` (244M params, 12+12 layers).
    #[must_use]
    pub fn whisper_small() -> Self {
        Self {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 768,
            encoder_attention_heads: 12,
            encoder_layers: 12,
            encoder_ffn_dim: 3072,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 12,
            decoder_layers: 12,
            decoder_ffn_dim: 3072,
        }
    }

    /// Configuration for `whisper-medium` (769M params, 24+24 layers).
    #[must_use]
    pub fn whisper_medium() -> Self {
        Self {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 1024,
            encoder_attention_heads: 16,
            encoder_layers: 24,
            encoder_ffn_dim: 4096,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 16,
            decoder_layers: 24,
            decoder_ffn_dim: 4096,
        }
    }

    /// Configuration for `whisper-large-v2` (1550M params, 32+32 layers).
    #[must_use]
    pub fn whisper_large_v2() -> Self {
        Self {
            num_mel_bins: 128,
            max_source_positions: 1500,
            d_model: 1280,
            encoder_attention_heads: 20,
            encoder_layers: 32,
            encoder_ffn_dim: 5120,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 20,
            decoder_layers: 32,
            decoder_ffn_dim: 5120,
        }
    }

    // -- Builder-style `with_*` methods for non_exhaustive compat --

    /// Set `num_mel_bins`. Chainable.
    #[must_use]
    pub fn with_num_mel_bins(mut self, v: usize) -> Self {
        self.num_mel_bins = v;
        self
    }

    /// Set `max_source_positions`. Chainable.
    #[must_use]
    pub fn with_max_source_positions(mut self, v: usize) -> Self {
        self.max_source_positions = v;
        self
    }

    /// Set `d_model`. Chainable.
    #[must_use]
    pub fn with_d_model(mut self, v: usize) -> Self {
        self.d_model = v;
        self
    }

    /// Set `encoder_attention_heads`. Chainable.
    #[must_use]
    pub fn with_encoder_attention_heads(mut self, v: usize) -> Self {
        self.encoder_attention_heads = v;
        self
    }

    /// Set `encoder_layers`. Chainable.
    #[must_use]
    pub fn with_encoder_layers(mut self, v: usize) -> Self {
        self.encoder_layers = v;
        self
    }

    /// Set `encoder_ffn_dim`. Chainable.
    #[must_use]
    pub fn with_encoder_ffn_dim(mut self, v: usize) -> Self {
        self.encoder_ffn_dim = v;
        self
    }

    /// Set `vocab_size`. Chainable.
    #[must_use]
    pub fn with_vocab_size(mut self, v: usize) -> Self {
        self.vocab_size = v;
        self
    }

    /// Set `max_target_positions`. Chainable.
    #[must_use]
    pub fn with_max_target_positions(mut self, v: usize) -> Self {
        self.max_target_positions = v;
        self
    }

    /// Set `decoder_attention_heads`. Chainable.
    #[must_use]
    pub fn with_decoder_attention_heads(mut self, v: usize) -> Self {
        self.decoder_attention_heads = v;
        self
    }

    /// Set `decoder_layers`. Chainable.
    #[must_use]
    pub fn with_decoder_layers(mut self, v: usize) -> Self {
        self.decoder_layers = v;
        self
    }

    /// Set `decoder_ffn_dim`. Chainable.
    #[must_use]
    pub fn with_decoder_ffn_dim(mut self, v: usize) -> Self {
        self.decoder_ffn_dim = v;
        self
    }

    /// Head dimension for encoder attention.
    #[must_use]
    pub fn encoder_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Head dimension for decoder attention.
    #[must_use]
    pub fn decoder_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self::large_v3_turbo()
    }
}

#[cfg(kani)]
#[path = "kani_config_proofs.rs"]
mod kani_config_proofs;

#[cfg(kani)]
#[path = "kani_config_proofs_ext.rs"]
mod kani_config_proofs_ext;
