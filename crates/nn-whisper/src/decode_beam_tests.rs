// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Whisper beam search decode.

use crate::decode::*;
use crate::test_utils::tiny_config;
use crate::WhisperModel;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

fn tiny_model() -> WhisperModel {
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    WhisperModel::load(&vb, tiny_config()).expect("invariant: zero-weight model loads")
}

fn tiny_encoder_output() -> DynTensor {
    let config = tiny_config();
    DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).expect("invariant: zeros tensor")
}

// -- WhisperBeamConfig validation --

#[test]
fn test_beam_config_default() {
    let config = WhisperBeamConfig::default();
    assert_eq!(config.beam_width, 5);
    assert!((config.length_penalty - 1.0).abs() < f64::EPSILON);
    assert!(config.validate().is_ok());
}

#[test]
fn test_beam_config_zero_width_rejected() {
    let config = WhisperBeamConfig {
        beam_width: 0,
        length_penalty: 1.0,
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_beam_config_nan_penalty_rejected() {
    let config = WhisperBeamConfig {
        beam_width: 5,
        length_penalty: f64::NAN,
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_beam_config_inf_penalty_rejected() {
    let config = WhisperBeamConfig {
        beam_width: 5,
        length_penalty: f64::INFINITY,
    };
    assert!(config.validate().is_err());
}

// -- Beam search decode with zeros model --

#[test]
fn test_beam_search_decode_respects_max_length() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 5,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let beam_config = WhisperBeamConfig {
        beam_width: 2,
        length_penalty: 0.0,
    };

    let result = beam_search_decode(&mut model, &encoder_output, &config, &beam_config).unwrap();
    assert!(
        result.tokens.len() <= 5,
        "should respect max_length: got {}",
        result.tokens.len()
    );
}

#[test]
fn test_beam_search_decode_returns_valid_result() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let beam_config = WhisperBeamConfig {
        beam_width: 3,
        length_penalty: 1.0,
    };

    let result = beam_search_decode(&mut model, &encoder_output, &config, &beam_config).unwrap();
    assert!(result.avg_logprob.is_finite());
    assert!(result.compression_ratio >= 1.0);
    assert!((result.temperature - 0.0).abs() < f64::EPSILON);
    assert!(result.no_speech_prob >= 0.0 && result.no_speech_prob <= 1.0);
}

#[test]
fn test_beam_search_width_1_matches_greedy() {
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 5,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };

    // beam_width=1 should produce the same tokens as greedy.
    let mut model1 = tiny_model();
    let greedy = greedy_decode(&mut model1, &encoder_output, &config).unwrap();

    let mut model2 = tiny_model();
    let beam_config = WhisperBeamConfig {
        beam_width: 1,
        length_penalty: 0.0,
    };
    let beam = beam_search_decode(&mut model2, &encoder_output, &config, &beam_config).unwrap();

    assert_eq!(
        greedy.tokens, beam.tokens,
        "beam_width=1 should match greedy: greedy={:?}, beam={:?}",
        greedy.tokens, beam.tokens
    );
}

#[test]
fn test_beam_search_with_suppression() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: vec![0],
        ..Default::default()
    };
    let beam_config = WhisperBeamConfig {
        beam_width: 2,
        length_penalty: 0.0,
    };

    let result = beam_search_decode(&mut model, &encoder_output, &config, &beam_config).unwrap();
    for &t in &result.tokens {
        assert_ne!(t, 0, "suppressed token should not appear in output");
    }
}

// -- Finished beam carry-forward (#1636) --

#[test]
fn test_beam_search_wider_beam_at_least_as_good() {
    // A wider beam search should produce results at least as good as a narrower one
    // in terms of avg_logprob. This exercises the carry-forward logic: if a good
    // hypothesis finishes early, a wider beam must preserve it for final selection.
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 8,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };

    let mut model_narrow = tiny_model();
    let beam_narrow = WhisperBeamConfig {
        beam_width: 1,
        length_penalty: 0.0,
    };
    let r_narrow =
        beam_search_decode(&mut model_narrow, &encoder_output, &config, &beam_narrow).unwrap();

    let mut model_wide = tiny_model();
    let beam_wide = WhisperBeamConfig {
        beam_width: 3,
        length_penalty: 0.0,
    };
    let r_wide = beam_search_decode(&mut model_wide, &encoder_output, &config, &beam_wide).unwrap();

    // Wider beam should produce a score at least as good (higher or equal avg_logprob).
    assert!(
        r_wide.avg_logprob >= r_narrow.avg_logprob - 1e-6,
        "wider beam ({}) should not be worse than narrow ({})",
        r_wide.avg_logprob,
        r_narrow.avg_logprob
    );
}

#[test]
fn test_beam_search_deterministic() {
    // Beam search with the same model should produce identical results.
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 5,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let beam_config = WhisperBeamConfig {
        beam_width: 3,
        length_penalty: 1.0,
    };

    let mut model1 = tiny_model();
    let r1 = beam_search_decode(&mut model1, &encoder_output, &config, &beam_config).unwrap();
    let mut model2 = tiny_model();
    let r2 = beam_search_decode(&mut model2, &encoder_output, &config, &beam_config).unwrap();

    assert_eq!(r1.tokens, r2.tokens, "beam search should be deterministic");
    assert!(
        (r1.avg_logprob - r2.avg_logprob).abs() < 1e-10,
        "avg_logprob should match: {} vs {}",
        r1.avg_logprob,
        r2.avg_logprob
    );
}

#[test]
fn test_beam_search_no_speech_prob_range() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let beam_config = WhisperBeamConfig::default();

    let result = beam_search_decode(&mut model, &encoder_output, &config, &beam_config).unwrap();
    assert!(
        result.no_speech_prob >= 0.0 && result.no_speech_prob <= 1.0,
        "no_speech_prob should be in [0, 1], got {}",
        result.no_speech_prob,
    );
}

// -- DecodeConfig validation in beam_search_decode (#1639) --

#[test]
fn test_beam_search_rejects_nan_threshold() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        compression_ratio_threshold: f64::NAN,
        ..Default::default()
    };
    let beam_config = WhisperBeamConfig::default();

    let err = beam_search_decode(&mut model, &encoder_output, &config, &beam_config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("compression_ratio_threshold"),
        "should reject NaN threshold, got: {msg}"
    );
}

#[test]
fn test_beam_search_rejects_empty_initial_tokens() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: Vec::new(),
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let beam_config = WhisperBeamConfig::default();

    let err = beam_search_decode(&mut model, &encoder_output, &config, &beam_config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("initial_tokens"),
        "should reject empty initial_tokens, got: {msg}"
    );
}

#[test]
fn test_beam_search_rejects_max_length_zero() {
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();

    let config = DecodeConfig {
        max_length: 0,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let beam_config = WhisperBeamConfig::default();

    let err = beam_search_decode(&mut model, &encoder_output, &config, &beam_config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("max_length"),
        "should reject max_length=0, got: {msg}"
    );
}
