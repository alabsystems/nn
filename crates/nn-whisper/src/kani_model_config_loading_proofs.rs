// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Whisper model-configuration consistency.
//!
//! Covers:
//! - `validate()` accepts configs where `d_model` is divisible by both head counts
//! - `validate()` rejects encoder head splits that do not divide `d_model`
//! - `validate()` rejects decoder head splits that do not divide `d_model`
//! - Preset source position limits match the encoder's stride-2 audio reduction
//!
//! Issue: #3724

#[cfg(kani)]
mod proofs {
    use crate::config::N_FRAMES;
    use crate::WhisperConfig;

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn validate_accepts_consistent_attention_splits() {
        let d_model: usize = kani::any();
        let encoder_heads: usize = kani::any();
        let decoder_heads: usize = kani::any();

        kani::assume(d_model >= 1 && d_model <= 96);
        kani::assume(encoder_heads >= 1 && encoder_heads <= 12);
        kani::assume(decoder_heads >= 1 && decoder_heads <= 12);
        kani::assume(d_model.is_multiple_of(encoder_heads));
        kani::assume(d_model.is_multiple_of(decoder_heads));

        let config = WhisperConfig::whisper_tiny()
            .with_d_model(d_model)
            .with_encoder_attention_heads(encoder_heads)
            .with_decoder_attention_heads(decoder_heads);

        assert!(
            config.validate().is_ok(),
            "load-time config validation must accept exact attention splits"
        );
        assert_eq!(
            config.encoder_head_dim() * config.encoder_attention_heads,
            d_model,
            "encoder head_dim * n_head must reconstruct d_model"
        );
        assert_eq!(
            config.decoder_head_dim() * config.decoder_attention_heads,
            d_model,
            "decoder head_dim * n_head must reconstruct d_model"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn validate_rejects_encoder_split_mismatch() {
        let d_model: usize = kani::any();
        let encoder_heads: usize = kani::any();

        kani::assume(d_model >= 1 && d_model <= 64);
        kani::assume(encoder_heads >= 2 && encoder_heads <= 16);
        kani::assume(!d_model.is_multiple_of(encoder_heads));

        let config = WhisperConfig::whisper_tiny()
            .with_d_model(d_model)
            .with_encoder_attention_heads(encoder_heads)
            .with_decoder_attention_heads(1);

        assert!(
            config.validate().is_err(),
            "encoder_attention_heads must divide d_model"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn validate_rejects_decoder_split_mismatch() {
        let d_model: usize = kani::any();
        let decoder_heads: usize = kani::any();

        kani::assume(d_model >= 1 && d_model <= 64);
        kani::assume(decoder_heads >= 2 && decoder_heads <= 16);
        kani::assume(!d_model.is_multiple_of(decoder_heads));

        let config = WhisperConfig::whisper_tiny()
            .with_d_model(d_model)
            .with_encoder_attention_heads(1)
            .with_decoder_attention_heads(decoder_heads);

        assert!(
            config.validate().is_err(),
            "decoder_attention_heads must divide d_model"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn preset_source_positions_match_audio_stride() {
        let preset: u8 = kani::any();
        kani::assume(preset < 6);

        let config = match preset {
            0 => WhisperConfig::whisper_tiny(),
            1 => WhisperConfig::whisper_base(),
            2 => WhisperConfig::whisper_small(),
            3 => WhisperConfig::whisper_medium(),
            4 => WhisperConfig::whisper_large_v2(),
            _ => WhisperConfig::large_v3_turbo(),
        };

        assert_eq!(
            config.max_source_positions * 2,
            N_FRAMES,
            "config max_source_positions must match stride-2 audio downsampling"
        );
        assert!(
            config.num_mel_bins == 80 || config.num_mel_bins == 128,
            "Whisper presets use either 80 or 128 mel bins"
        );
    }
}
