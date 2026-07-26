// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the unified HTDemucs model struct.
//!
//! Covers: construction, weight validation, forward pass shape correctness,
//! normalization/denormalization, and error handling.
//!
//! Part of #779 — Milestone 1 of BUILDING→USABLE gate.

use super::*;
use crate::demucs_spectral_decoder::DemucsSpectralDecoderWeights;
use crate::demucs_spectral_encoder::DemucsSpectralEncoderWeights;
use crate::demucs_test_common::make_htdemucs_weights;
use crate::test_common::make_cache;

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_construction_accepts_valid_weights() {
    let weights = make_htdemucs_weights();
    let result = HTDemucs::new(weights, 256);
    assert!(
        result.is_ok(),
        "valid weights should be accepted: {result:?}"
    );
}

#[test]
fn test_construction_stores_audio_t() {
    let weights = make_htdemucs_weights();
    let model = HTDemucs::new(weights, 512).expect("valid weights");
    assert_eq!(model.audio_t(), 512);
}

#[test]
fn test_construction_invalid_encoder_weights() {
    let mut weights = make_htdemucs_weights();
    weights.encoder.blocks.pop(); // wrong block count
    let err = HTDemucs::new(weights, 256).unwrap_err();
    assert!(
        matches!(err, HTDemucsError::Encoder(_)),
        "expected encoder error: {err}"
    );
}

#[test]
fn test_construction_invalid_transformer_weights() {
    let mut weights = make_htdemucs_weights();
    weights.transformer.temporal_layers.pop(); // wrong layer count
    let err = HTDemucs::new(weights, 256).unwrap_err();
    assert!(
        matches!(err, HTDemucsError::Transformer(_)),
        "expected transformer error: {err}"
    );
}

#[test]
fn test_construction_invalid_decoder_weights() {
    let mut weights = make_htdemucs_weights();
    weights.decoder.blocks.pop(); // wrong block count
    let err = HTDemucs::new(weights, 256).unwrap_err();
    assert!(
        matches!(err, HTDemucsError::Decoder(_)),
        "expected decoder error: {err}"
    );
}

// ---------------------------------------------------------------------------
// Input validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_forward_wrong_audio_length() {
    let weights = make_htdemucs_weights();
    let model = HTDemucs::new(weights, 256).expect("valid weights");
    let cache = match make_cache() {
        Some(c) => c,
        None => return, // skip on non-Metal
    };
    let bad_audio = vec![0.0f32; 100]; // not 2*256
    let err = model.forward(&cache, &bad_audio).unwrap_err();
    assert!(
        matches!(err, HTDemucsError::AudioLength { .. }),
        "expected AudioLength error: {err}"
    );
}

#[test]
fn test_forward_nan_input() {
    let weights = make_htdemucs_weights();
    let model = HTDemucs::new(weights, 256).expect("valid weights");
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let mut audio = vec![0.1f32; 2 * 256];
    audio[100] = f32::NAN;
    let err = model.forward(&cache, &audio).unwrap_err();
    assert!(
        matches!(err, HTDemucsError::NonFiniteInput { .. }),
        "expected NonFiniteInput error: {err}"
    );
}

// Normalization tests extracted to htdemucs_normalization_tests.rs (#922).

// ---------------------------------------------------------------------------
// Intermediate NaN/Inf check tests (#941)
// ---------------------------------------------------------------------------

/// AC7: Inject NaN into encoder conv weights → forward should return
/// NonFiniteIntermediate at the "encoder" stage (not silently propagate).
#[test]
fn test_forward_nan_encoder_weight_caught_at_encoder_stage() {
    let mut weights = make_htdemucs_weights();
    // Inject NaN into first encoder block's conv weight.
    weights.encoder.blocks[0].conv_weight[0] = f32::NAN;
    let model = HTDemucs::new(weights, 256).expect("valid weights");
    let cache = match make_cache() {
        Some(c) => c,
        None => return, // skip on non-Metal
    };
    let audio = vec![0.1f32; 2 * 256];
    let err = model.forward(&cache, &audio).unwrap_err();
    let err_str = err.to_string();
    // The NaN should be caught either by the encoder's internal check
    // or by the HTDemucs intermediate check at the "encoder" stage.
    assert!(
        matches!(
            err,
            HTDemucsError::Encoder(_)
                | HTDemucsError::NonFiniteIntermediate {
                    stage: "encoder",
                    ..
                }
        ),
        "expected encoder-stage NaN detection, got: {err_str}"
    );
}

/// AC7: Verify the NonFiniteIntermediate error variant includes stage name.
#[test]
fn test_non_finite_intermediate_error_format() {
    let err = HTDemucsError::NonFiniteIntermediate {
        stage: "transformer",
        count: 42,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("transformer"),
        "error should name the stage: {msg}"
    );
    assert!(msg.contains("42"), "error should contain count: {msg}");
}

/// AC7: Verify NonFiniteOutput is still returned for denormalization-stage NaN.
#[test]
fn test_non_finite_output_error_format() {
    let err = HTDemucsError::NonFiniteOutput { count: 7 };
    let msg = err.to_string();
    assert!(msg.contains("7"), "error should contain count: {msg}");
    assert!(
        msg.contains("non-finite output"),
        "error should describe output: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Shape computation tests
// ---------------------------------------------------------------------------

#[test]
fn test_bottleneck_t_computation() {
    // T=256: stride-pad each depth, then Conv1d(k=8,s=4,p=2)
    // Depth 0: 256 → (256+4-8)/4+1 = 63
    // Depth 1: 64 (padded from 63) → (64+4-8)/4+1 = 16
    // Depth 2: 16 → (16+4-8)/4+1 = 4
    // Depth 3: 4 → (4+4-8)/4+1 = 1
    let bt = compute_bottleneck_t(256);
    assert!(bt > 0, "bottleneck T should be positive, got {bt}");
}

#[test]
fn test_encoder_input_lengths_computation() {
    let lengths = compute_encoder_input_lengths(256);
    assert_eq!(lengths.len(), 4);
    assert_eq!(lengths[0], 256);
    // Each subsequent length should be smaller.
    for i in 1..4 {
        assert!(
            lengths[i] < lengths[i - 1],
            "lengths should decrease: {lengths:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Debug display test
// ---------------------------------------------------------------------------

#[test]
fn test_debug_display() {
    let weights = make_htdemucs_weights();
    let model = HTDemucs::new(weights, 256).expect("valid weights");
    let debug = format!("{model:?}");
    assert!(
        debug.contains("HTDemucs"),
        "debug should contain struct name"
    );
    assert!(debug.contains("audio_t"), "debug should contain audio_t");
}

// ---------------------------------------------------------------------------
// forward_gpu NaN/Inf tests (#964)
// ---------------------------------------------------------------------------

/// AC1 (#964): forward_gpu with NaN encoder weights → caught before returning.
///
/// With NanCheckPolicy::Skip (#1915), per-stage checks inside forward_inner are
/// no-ops for GPU performance. NaN from encoder weights propagates to the
/// model-boundary output check (NonFiniteOutput). The important invariant is
/// that NaN is caught before returning to the caller (AC2 of #1958).
#[test]
fn test_forward_gpu_nan_encoder_weight_caught() {
    let mut weights = make_htdemucs_weights();
    weights.encoder.blocks[0].conv_weight[0] = f32::NAN;
    let model = HTDemucs::new(weights, 256).expect("valid weights");
    let cache = match make_cache() {
        Some(c) => c,
        None => return, // skip on non-Metal
    };
    let audio = vec![0.1f32; 2 * 256];
    let err = model.forward_gpu(&cache, &audio).unwrap_err();
    let err_str = err.to_string();
    // Under NanCheckPolicy::Skip, NaN propagates past per-stage checks and is
    // caught at the output boundary. Accept any NaN-related error variant.
    assert!(
        matches!(
            err,
            HTDemucsError::Encoder(_)
                | HTDemucsError::NonFiniteIntermediate { .. }
                | HTDemucsError::NonFiniteOutput { .. }
        ),
        "forward_gpu should catch NaN before returning, got: {err_str}"
    );
}

/// AC1 (#964): forward_gpu with NaN input audio → caught at input validation.
#[test]
fn test_forward_gpu_nan_input_rejected() {
    let weights = make_htdemucs_weights();
    let model = HTDemucs::new(weights, 256).expect("valid weights");
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let mut audio = vec![0.1f32; 2 * 256];
    audio[50] = f32::NAN;
    let err = model.forward_gpu(&cache, &audio).unwrap_err();
    assert!(
        matches!(err, HTDemucsError::NonFiniteInput { .. }),
        "forward_gpu should reject NaN input, got: {err}"
    );
}

/// AC2 (#964): forward_gpu wrong audio length → AudioLength error.
#[test]
fn test_forward_gpu_wrong_audio_length() {
    let weights = make_htdemucs_weights();
    let model = HTDemucs::new(weights, 256).expect("valid weights");
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let bad_audio = vec![0.0f32; 100]; // not 2*256
    let err = model.forward_gpu(&cache, &bad_audio).unwrap_err();
    assert!(
        matches!(err, HTDemucsError::AudioLength { .. }),
        "forward_gpu should reject wrong audio length, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// One-sided spectral branch rejection tests (#1368 AC2, AC3)
// ---------------------------------------------------------------------------

#[test]
fn test_one_sided_spectral_encoder_only_rejected() {
    let mut weights = make_htdemucs_weights();
    // Provide spectral encoder but not decoder — should error.
    weights.spectral_encoder = Some(DemucsSpectralEncoderWeights {
        blocks: vec![],
        freq_emb_weight: None,
    });
    assert!(weights.spectral_decoder.is_none());
    let err = HTDemucs::new(weights, 256).unwrap_err();
    assert!(
        matches!(
            err,
            HTDemucsError::OneSidedSpectral {
                provided: "encoder"
            }
        ),
        "expected OneSidedSpectral(encoder), got: {err}"
    );
}

#[test]
fn test_one_sided_spectral_decoder_only_rejected() {
    let mut weights = make_htdemucs_weights();
    // Provide spectral decoder but not encoder — should error.
    weights.spectral_decoder = Some(DemucsSpectralDecoderWeights { blocks: vec![] });
    assert!(weights.spectral_encoder.is_none());
    let err = HTDemucs::new(weights, 256).unwrap_err();
    assert!(
        matches!(
            err,
            HTDemucsError::OneSidedSpectral {
                provided: "decoder"
            }
        ),
        "expected OneSidedSpectral(decoder), got: {err}"
    );
}
