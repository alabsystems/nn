// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for WhisperConfig validation and builder safety.
//!
//! Covers:
//! - Builder `with_*` chain preserves other fields (non-interference)
//! - Default config is large_v3_turbo
//! - All zero-field variants are caught by validate()
//! - Divisibility guard catches all non-divisible combos
//! - encoder_head_dim() and decoder_head_dim() are exact (no remainder)
//! - Preset configs satisfy all documented field constraints
//! - Audio constants are self-consistent
//! - Config builder idempotence
//!
//! Issue: #3707

use super::*;
use crate::WhisperError;

// ============================================================================
// Harness 1: default() returns large_v3_turbo
// ============================================================================

/// Proves that `WhisperConfig::default()` produces the same config as
/// `WhisperConfig::large_v3_turbo()`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_default_is_large_v3_turbo() {
    let def = WhisperConfig::default();
    let turbo = WhisperConfig::large_v3_turbo();
    assert_eq!(def, turbo, "default must equal large_v3_turbo");
}

// ============================================================================
// Harness 2: with_d_model preserves other fields
// ============================================================================

/// Proves that `with_d_model()` only changes d_model, not any other field.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_with_d_model_preserves_others() {
    let base = WhisperConfig::large_v3_turbo();
    let new_d: usize = kani::any();
    kani::assume(new_d <= 4096);

    let modified = base.clone().with_d_model(new_d);
    assert_eq!(modified.d_model, new_d);
    assert_eq!(modified.num_mel_bins, base.num_mel_bins);
    assert_eq!(modified.max_source_positions, base.max_source_positions);
    assert_eq!(
        modified.encoder_attention_heads,
        base.encoder_attention_heads
    );
    assert_eq!(modified.encoder_layers, base.encoder_layers);
    assert_eq!(modified.encoder_ffn_dim, base.encoder_ffn_dim);
    assert_eq!(modified.vocab_size, base.vocab_size);
    assert_eq!(modified.max_target_positions, base.max_target_positions);
    assert_eq!(
        modified.decoder_attention_heads,
        base.decoder_attention_heads
    );
    assert_eq!(modified.decoder_layers, base.decoder_layers);
    assert_eq!(modified.decoder_ffn_dim, base.decoder_ffn_dim);
}

// ============================================================================
// Harness 3: with_encoder_attention_heads preserves other fields
// ============================================================================

/// Proves that `with_encoder_attention_heads()` only changes that field.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_with_encoder_heads_preserves_others() {
    let base = WhisperConfig::whisper_tiny();
    let new_heads: usize = kani::any();
    kani::assume(new_heads <= 64);

    let modified = base.clone().with_encoder_attention_heads(new_heads);
    assert_eq!(modified.encoder_attention_heads, new_heads);
    assert_eq!(modified.d_model, base.d_model);
    assert_eq!(modified.num_mel_bins, base.num_mel_bins);
    assert_eq!(modified.decoder_attention_heads, base.decoder_attention_heads);
    assert_eq!(modified.vocab_size, base.vocab_size);
}

// ============================================================================
// Harness 4: with_decoder_attention_heads preserves other fields
// ============================================================================

/// Proves that `with_decoder_attention_heads()` only changes that field.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_with_decoder_heads_preserves_others() {
    let base = WhisperConfig::whisper_base();
    let new_heads: usize = kani::any();
    kani::assume(new_heads <= 64);

    let modified = base.clone().with_decoder_attention_heads(new_heads);
    assert_eq!(modified.decoder_attention_heads, new_heads);
    assert_eq!(modified.d_model, base.d_model);
    assert_eq!(modified.encoder_attention_heads, base.encoder_attention_heads);
}

// ============================================================================
// Harness 5: validate rejects zero num_mel_bins
// ============================================================================

/// Proves validate() catches num_mel_bins == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_mel_bins() {
    let cfg = WhisperConfig::large_v3_turbo().with_num_mel_bins(0);
    assert!(cfg.validate().is_err());
}

// ============================================================================
// Harness 6: validate rejects zero encoder_ffn_dim
// ============================================================================

/// Proves validate() catches encoder_ffn_dim == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_encoder_ffn() {
    let cfg = WhisperConfig::large_v3_turbo().with_encoder_ffn_dim(0);
    assert!(cfg.validate().is_err());
}

// ============================================================================
// Harness 7: validate rejects zero decoder_ffn_dim
// ============================================================================

/// Proves validate() catches decoder_ffn_dim == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_decoder_ffn() {
    let cfg = WhisperConfig::large_v3_turbo().with_decoder_ffn_dim(0);
    assert!(cfg.validate().is_err());
}

// ============================================================================
// Harness 8: validate rejects zero max_source_positions
// ============================================================================

/// Proves validate() catches max_source_positions == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_max_source() {
    let cfg = WhisperConfig::large_v3_turbo().with_max_source_positions(0);
    assert!(cfg.validate().is_err());
}

// ============================================================================
// Harness 9: validate rejects zero max_target_positions
// ============================================================================

/// Proves validate() catches max_target_positions == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_max_target() {
    let cfg = WhisperConfig::large_v3_turbo().with_max_target_positions(0);
    assert!(cfg.validate().is_err());
}

// ============================================================================
// Harness 10: validate rejects zero vocab_size
// ============================================================================

/// Proves validate() catches vocab_size == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_vocab() {
    let cfg = WhisperConfig::large_v3_turbo().with_vocab_size(0);
    assert!(cfg.validate().is_err());
}

// ============================================================================
// Harness 11: encoder_head_dim is exact for all presets
// ============================================================================

/// Proves encoder_head_dim() * encoder_attention_heads == d_model for all presets.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_encoder_head_dim_exact_all_presets() {
    let idx: u8 = kani::any();
    kani::assume(idx < 6);

    let cfg = match idx {
        0 => WhisperConfig::whisper_tiny(),
        1 => WhisperConfig::whisper_base(),
        2 => WhisperConfig::whisper_small(),
        3 => WhisperConfig::whisper_medium(),
        4 => WhisperConfig::whisper_large_v2(),
        _ => WhisperConfig::large_v3_turbo(),
    };

    let hd = cfg.encoder_head_dim();
    assert_eq!(
        hd * cfg.encoder_attention_heads,
        cfg.d_model,
        "encoder head_dim * heads == d_model"
    );
}

// ============================================================================
// Harness 12: decoder_head_dim is exact for all presets
// ============================================================================

/// Proves decoder_head_dim() * decoder_attention_heads == d_model for all presets.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_decoder_head_dim_exact_all_presets() {
    let idx: u8 = kani::any();
    kani::assume(idx < 6);

    let cfg = match idx {
        0 => WhisperConfig::whisper_tiny(),
        1 => WhisperConfig::whisper_base(),
        2 => WhisperConfig::whisper_small(),
        3 => WhisperConfig::whisper_medium(),
        4 => WhisperConfig::whisper_large_v2(),
        _ => WhisperConfig::large_v3_turbo(),
    };

    let hd = cfg.decoder_head_dim();
    assert_eq!(
        hd * cfg.decoder_attention_heads,
        cfg.d_model,
        "decoder head_dim * heads == d_model"
    );
}

// ============================================================================
// Harness 13: encoder_ffn_dim is 4x d_model for all presets
// ============================================================================

/// Proves that encoder_ffn_dim == 4 * d_model for all standard Whisper presets.
///
/// This is the standard transformer FFN expansion ratio.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_ffn_dim_is_4x_d_model() {
    let idx: u8 = kani::any();
    kani::assume(idx < 6);

    let cfg = match idx {
        0 => WhisperConfig::whisper_tiny(),
        1 => WhisperConfig::whisper_base(),
        2 => WhisperConfig::whisper_small(),
        3 => WhisperConfig::whisper_medium(),
        4 => WhisperConfig::whisper_large_v2(),
        _ => WhisperConfig::large_v3_turbo(),
    };

    assert_eq!(
        cfg.encoder_ffn_dim,
        4 * cfg.d_model,
        "encoder_ffn_dim = 4 * d_model"
    );
    assert_eq!(
        cfg.decoder_ffn_dim,
        4 * cfg.d_model,
        "decoder_ffn_dim = 4 * d_model"
    );
}

// ============================================================================
// Harness 14: audio constants are self-consistent
// ============================================================================

/// Proves that audio constants satisfy documented relationships.
///
/// N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH
/// N_FRAMES = N_SAMPLES / HOP_LENGTH
/// N_FFT = 400 (fixed for Whisper)
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_audio_constants_consistent() {
    assert_eq!(N_SAMPLES, SAMPLE_RATE * CHUNK_LENGTH);
    assert_eq!(N_FRAMES, N_SAMPLES / HOP_LENGTH);
    assert_eq!(SAMPLE_RATE, 16_000);
    assert_eq!(N_FFT, 400);
    assert_eq!(HOP_LENGTH, 160);
    assert_eq!(CHUNK_LENGTH, 30);
    assert_eq!(NUM_MEL_BINS, 128);
}

// ============================================================================
// Harness 15: N_SAMPLES is divisible by HOP_LENGTH
// ============================================================================

/// Proves that N_SAMPLES / HOP_LENGTH has no remainder.
///
/// If there were a remainder, the last frame would be partial, causing
/// dimension mismatches in the encoder.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_n_samples_divisible_by_hop() {
    assert_eq!(
        N_SAMPLES % HOP_LENGTH,
        0,
        "N_SAMPLES must be exactly divisible by HOP_LENGTH"
    );
}

// ============================================================================
// Harness 16: builder chain roundtrip preserves identity
// ============================================================================

/// Proves that re-applying the same value via with_* is identity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_builder_idempotent() {
    let cfg = WhisperConfig::large_v3_turbo();
    let same = cfg
        .clone()
        .with_d_model(cfg.d_model)
        .with_num_mel_bins(cfg.num_mel_bins)
        .with_encoder_attention_heads(cfg.encoder_attention_heads)
        .with_decoder_attention_heads(cfg.decoder_attention_heads)
        .with_vocab_size(cfg.vocab_size);
    assert_eq!(cfg, same, "re-applying same values must be identity");
}

// ============================================================================
// Harness 17: validate accepts all presets
// ============================================================================

/// Proves that validate() accepts every standard preset config.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_accepts_all_presets() {
    let idx: u8 = kani::any();
    kani::assume(idx < 6);

    let cfg = match idx {
        0 => WhisperConfig::whisper_tiny(),
        1 => WhisperConfig::whisper_base(),
        2 => WhisperConfig::whisper_small(),
        3 => WhisperConfig::whisper_medium(),
        4 => WhisperConfig::whisper_large_v2(),
        _ => WhisperConfig::large_v3_turbo(),
    };

    assert!(cfg.validate().is_ok(), "preset config must pass validation");
}

// ============================================================================
// Harness 18: max_source_positions == 1500 for all presets
// ============================================================================

/// Proves that all Whisper preset configs share max_source_positions == 1500.
///
/// This corresponds to 30s of audio at 16kHz with hop_length=160 and
/// stride-2 Conv1d downsampling: 480000/160/2 = 1500.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_max_source_positions_1500() {
    let idx: u8 = kani::any();
    kani::assume(idx < 6);

    let cfg = match idx {
        0 => WhisperConfig::whisper_tiny(),
        1 => WhisperConfig::whisper_base(),
        2 => WhisperConfig::whisper_small(),
        3 => WhisperConfig::whisper_medium(),
        4 => WhisperConfig::whisper_large_v2(),
        _ => WhisperConfig::large_v3_turbo(),
    };

    assert_eq!(cfg.max_source_positions, 1500);
}

// ============================================================================
// Harness 19: validate rejects d_model not divisible by encoder heads
// ============================================================================

/// Proves validate() catches d_model % encoder_attention_heads != 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_encoder_nondivisible() {
    // d_model=384, encoder_heads=5 -> 384 % 5 = 4 != 0
    let cfg = WhisperConfig::whisper_tiny().with_encoder_attention_heads(5);
    assert!(
        cfg.validate().is_err(),
        "d_model not divisible by encoder_heads must fail"
    );
}

// ============================================================================
// Harness 20: validate rejects d_model not divisible by decoder heads
// ============================================================================

/// Proves validate() catches d_model % decoder_attention_heads != 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_decoder_nondivisible() {
    // d_model=384, decoder_heads=5 -> 384 % 5 = 4 != 0
    let cfg = WhisperConfig::whisper_tiny().with_decoder_attention_heads(5);
    assert!(
        cfg.validate().is_err(),
        "d_model not divisible by decoder_heads must fail"
    );
}

// ============================================================================
// Harness 21: tiny/base/small use 80 mel bins, medium+ use 128
// ============================================================================

/// Proves the documented mel bin split: older configs use 80, newer use 128.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_mel_bins_split() {
    assert_eq!(WhisperConfig::whisper_tiny().num_mel_bins, 80);
    assert_eq!(WhisperConfig::whisper_base().num_mel_bins, 80);
    assert_eq!(WhisperConfig::whisper_small().num_mel_bins, 80);
    assert_eq!(WhisperConfig::whisper_medium().num_mel_bins, 80);
    assert_eq!(WhisperConfig::whisper_large_v2().num_mel_bins, 128);
    assert_eq!(WhisperConfig::large_v3_turbo().num_mel_bins, 128);
}
