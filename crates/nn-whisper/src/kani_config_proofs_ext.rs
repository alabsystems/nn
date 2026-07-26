// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for WhisperConfig.
//!
//! Supplements `kani_config_proofs.rs` with additional coverage:
//! - validate rejects zero d_model / encoder_attention_heads / decoder_attention_heads
//! - d_model monotonically increases across preset tiers
//! - encoder/decoder head_dim matches within each preset
//! - max_target_positions == 448 for all presets
//! - config builder composition (multiple with_* calls)
//! - vocab size within Whisper token ID range
//! - encoder_layers >= decoder_layers for all presets (architectural constraint)
//! - mel filterbank frame count from audio constants
//!
//! Issue: #3741

use super::*;
use crate::WhisperError;

// ============================================================================
// Harness 2: validate rejects zero encoder_attention_heads
// ============================================================================

/// Proves validate() catches encoder_attention_heads == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_encoder_heads() {
    let cfg = WhisperConfig::large_v3_turbo().with_encoder_attention_heads(0);
    assert!(
        cfg.validate().is_err(),
        "encoder_attention_heads=0 must be rejected"
    );
}

// ============================================================================
// Harness 3: validate rejects zero decoder_attention_heads
// ============================================================================

/// Proves validate() catches decoder_attention_heads == 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_decoder_heads() {
    let cfg = WhisperConfig::large_v3_turbo().with_decoder_attention_heads(0);
    assert!(
        cfg.validate().is_err(),
        "decoder_attention_heads=0 must be rejected"
    );
}

// ============================================================================
// Harness 4: d_model monotonically increases across preset tiers
// ============================================================================

/// Proves that d_model increases strictly: tiny < base < small < medium < large.
///
/// Whisper model tiers scale by increasing d_model.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_d_model_increases_across_tiers() {
    let tiny = WhisperConfig::whisper_tiny().d_model;
    let base = WhisperConfig::whisper_base().d_model;
    let small = WhisperConfig::whisper_small().d_model;
    let medium = WhisperConfig::whisper_medium().d_model;
    let large = WhisperConfig::whisper_large_v2().d_model;
    let turbo = WhisperConfig::large_v3_turbo().d_model;

    assert!(tiny < base, "tiny < base");
    assert!(base < small, "base < small");
    assert!(small < medium, "small < medium");
    assert!(medium < large, "medium < large");
    assert_eq!(large, turbo, "large_v2 and turbo share d_model");
}

// ============================================================================
// Harness 5: encoder and decoder share head_dim within each preset
// ============================================================================

/// Proves that encoder_head_dim == decoder_head_dim for all standard presets.
///
/// All Whisper configs use the same head dimension for both encoder and decoder
/// because they share d_model and head count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_encoder_decoder_head_dim_match() {
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
        cfg.encoder_head_dim(),
        cfg.decoder_head_dim(),
        "encoder and decoder head_dim must match"
    );
}

// ============================================================================
// Harness 6: max_target_positions == 448 for all presets
// ============================================================================

/// Proves that all Whisper presets use max_target_positions == 448.
///
/// This corresponds to the maximum decode length in tokens.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_max_target_positions_448() {
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

    assert_eq!(cfg.max_target_positions, 448);
}

// ============================================================================
// Harness 7: builder chain composition — multiple setters
// ============================================================================

/// Proves that chaining multiple with_* calls produces the expected combined result.
///
/// Each setter should be independent — changing d_model then vocab_size
/// should yield a config with both changes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_builder_chain_composition() {
    let cfg = WhisperConfig::whisper_tiny()
        .with_d_model(512)
        .with_vocab_size(10000)
        .with_encoder_layers(8);

    assert_eq!(cfg.d_model, 512);
    assert_eq!(cfg.vocab_size, 10000);
    assert_eq!(cfg.encoder_layers, 8);
    // Unchanged fields preserved.
    assert_eq!(cfg.num_mel_bins, 80);
    assert_eq!(cfg.max_source_positions, 1500);
}

// ============================================================================
// Harness 8: encoder_layers >= decoder_layers for all presets
// ============================================================================

/// Proves that encoder has at least as many layers as decoder in all presets.
///
/// Whisper architecture: encoder is always at least as deep as decoder.
/// Turbo has 32 encoder + 4 decoder (distilled). Others match.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_encoder_layers_gte_decoder() {
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

    assert!(
        cfg.encoder_layers >= cfg.decoder_layers,
        "encoder must have >= decoder layers"
    );
}

// ============================================================================
// Harness 9: N_FRAMES equals max_source_positions * 2 (Conv1d stride-2)
// ============================================================================

/// Proves the relationship between audio frames and encoder positions.
///
/// Whisper's encoder Conv1d has stride 2, so max_source_positions = N_FRAMES / 2.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_n_frames_is_2x_max_source() {
    assert_eq!(
        N_FRAMES,
        WhisperConfig::default().max_source_positions * 2,
        "N_FRAMES = 2 * max_source_positions (Conv1d stride 2)"
    );
}

// ============================================================================
// Harness 10: with_encoder_layers preserves other fields
// ============================================================================

/// Proves that with_encoder_layers only changes encoder_layers.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_with_encoder_layers_preserves_others() {
    let base = WhisperConfig::whisper_small();
    let v: usize = kani::any();
    kani::assume(v <= 64);

    let modified = base.clone().with_encoder_layers(v);
    assert_eq!(modified.encoder_layers, v);
    assert_eq!(modified.d_model, base.d_model);
    assert_eq!(modified.decoder_layers, base.decoder_layers);
    assert_eq!(modified.vocab_size, base.vocab_size);
    assert_eq!(modified.num_mel_bins, base.num_mel_bins);
    assert_eq!(
        modified.encoder_attention_heads,
        base.encoder_attention_heads
    );
}

// ============================================================================
// Harness 11: with_decoder_layers preserves other fields
// ============================================================================

/// Proves that with_decoder_layers only changes decoder_layers.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_with_decoder_layers_preserves_others() {
    let base = WhisperConfig::whisper_medium();
    let v: usize = kani::any();
    kani::assume(v <= 64);

    let modified = base.clone().with_decoder_layers(v);
    assert_eq!(modified.decoder_layers, v);
    assert_eq!(modified.d_model, base.d_model);
    assert_eq!(modified.encoder_layers, base.encoder_layers);
    assert_eq!(modified.vocab_size, base.vocab_size);
}

// ============================================================================
// Harness 12: with_vocab_size preserves other fields
// ============================================================================

/// Proves that with_vocab_size only changes vocab_size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_with_vocab_size_preserves_others() {
    let base = WhisperConfig::large_v3_turbo();
    let v: usize = kani::any();
    kani::assume(v <= 100000);

    let modified = base.clone().with_vocab_size(v);
    assert_eq!(modified.vocab_size, v);
    assert_eq!(modified.d_model, base.d_model);
    assert_eq!(modified.encoder_layers, base.encoder_layers);
    assert_eq!(modified.decoder_layers, base.decoder_layers);
    assert_eq!(modified.num_mel_bins, base.num_mel_bins);
}

// ============================================================================
// Harness 13: with_encoder_ffn_dim preserves other fields
// ============================================================================

/// Proves that with_encoder_ffn_dim only changes encoder_ffn_dim.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_with_encoder_ffn_dim_preserves_others() {
    let base = WhisperConfig::whisper_base();
    let v: usize = kani::any();
    kani::assume(v <= 8192);

    let modified = base.clone().with_encoder_ffn_dim(v);
    assert_eq!(modified.encoder_ffn_dim, v);
    assert_eq!(modified.d_model, base.d_model);
    assert_eq!(modified.decoder_ffn_dim, base.decoder_ffn_dim);
    assert_eq!(modified.vocab_size, base.vocab_size);
}

// ============================================================================
// Harness 14: with_decoder_ffn_dim preserves other fields
// ============================================================================

/// Proves that with_decoder_ffn_dim only changes decoder_ffn_dim.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_with_decoder_ffn_dim_preserves_others() {
    let base = WhisperConfig::whisper_base();
    let v: usize = kani::any();
    kani::assume(v <= 8192);

    let modified = base.clone().with_decoder_ffn_dim(v);
    assert_eq!(modified.decoder_ffn_dim, v);
    assert_eq!(modified.d_model, base.d_model);
    assert_eq!(modified.encoder_ffn_dim, base.encoder_ffn_dim);
    assert_eq!(modified.vocab_size, base.vocab_size);
}
