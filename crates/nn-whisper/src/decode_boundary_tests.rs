// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Boundary and finiteness validation tests for Whisper decode.
//!
//! Extracted from `decode_tests.rs` for file-size compliance (#1420).

use crate::decode::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::error::TensorError;
use nn_core::test_utils::cpu;

// -- NaN/Inf finiteness validation tests (AC1, AC2, AC3) --

#[test]
fn test_check_logit_finiteness_nan_returns_error() {
    let logits = DynTensor::new(&[1.0, f32::NAN, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();
    let err = check_logit_finiteness(&logits, 0).unwrap_err();
    match err {
        TensorError::NonFiniteData { ref name, count } => {
            assert!(name.contains("decode_logits_step_0"), "name={name}");
            assert_eq!(count, 1);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_check_logit_finiteness_pos_inf_returns_error() {
    let logits = DynTensor::new(&[1.0, f32::INFINITY, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();
    let err = check_logit_finiteness(&logits, 5).unwrap_err();
    match err {
        TensorError::NonFiniteData { ref name, count } => {
            assert!(name.contains("step_5"), "name={name}");
            assert_eq!(count, 1);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_check_logit_finiteness_neg_inf_rejected() {
    // check_logit_finiteness now uses check_output_finite which rejects ALL
    // non-finite values including NEG_INFINITY. The function is called BEFORE
    // token suppression, so no legitimate -Inf should be present in the logits.
    let logits = DynTensor::new(&[1.0, f32::NEG_INFINITY, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();
    let err = check_logit_finiteness(&logits, 0).unwrap_err();
    match err {
        TensorError::NonFiniteData { ref name, count } => {
            assert!(name.contains("step_0"), "name={name}");
            assert_eq!(count, 1);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_check_logit_finiteness_all_finite_ok() {
    let logits = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();
    check_logit_finiteness(&logits, 0).expect("all finite should pass");
}

#[test]
fn test_check_logit_finiteness_multiple_nan() {
    let logits = DynTensor::new(
        &[f32::NAN, f32::NAN, f32::INFINITY, 4.0],
        &[1, 1, 4],
        &cpu(),
    )
    .unwrap();
    let err = check_logit_finiteness(&logits, 2).unwrap_err();
    match err {
        TensorError::NonFiniteData { count, .. } => {
            assert_eq!(count, 3, "2 NaN + 1 +Inf = 3 non-finite");
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_argmax_f32_total_cmp_deterministic() {
    // With total_cmp, NaN sorts above all other values. This verifies
    // the argmax picks the NaN index deterministically (rather than
    // random behavior from partial_cmp unwrap_or Equal).
    let values = [1.0_f32, 5.0, f32::NAN, 3.0];
    let idx = argmax_f32(&values);
    // NaN > everything under total_cmp, so index 2 is picked.
    assert_eq!(idx, 2, "total_cmp should pick NaN as max (index 2)");
}

#[test]
fn test_argmax_f32_no_nan() {
    let values = [1.0_f32, 5.0, 2.0, 3.0];
    assert_eq!(argmax_f32(&values), 1);
}

// -- compute_log_prob boundary tests (algorithm_audit) --

#[test]
fn test_compute_log_prob_all_neg_infinity_returns_neg_infinity() {
    // When all logits are NEG_INFINITY (reachable after apply_suppression_inplace
    // suppresses every token), compute_log_prob returns NEG_INFINITY because
    // the max_val guard short-circuits before the indeterminate form.
    let logits = vec![f32::NEG_INFINITY; 5];
    let lp = compute_log_prob(&logits, 0);
    assert!(
        lp == f32::NEG_INFINITY,
        "all-NEG_INFINITY logits should produce NEG_INFINITY log-prob, got {lp}"
    );
}

#[test]
fn test_sample_token_all_neg_infinity_does_not_panic() {
    // When all logits are NEG_INFINITY (after full token suppression),
    // sample_token should not panic. It falls back to argmax on
    // the original logits (idx 0) and compute_log_prob returns NEG_INFINITY.
    let logits = vec![f32::NEG_INFINITY; 10];
    let (idx, log_prob) = sample_token(&logits, 0.0, None);
    // argmax_f32 on all-equal-NEG_INFINITY returns 0 (first max_by winner).
    assert!(idx < 10, "index must be valid");
    assert!(
        log_prob == f32::NEG_INFINITY,
        "log_prob should be NEG_INFINITY for all-NEG_INFINITY input, got {log_prob}"
    );
}

#[test]
fn test_sample_token_all_neg_infinity_with_temperature() {
    // With positive temperature, all-NEG_INFINITY logits hit the
    // `!sum.is_finite()` guard in sample_token (sum is NaN after scaling).
    let logits = vec![f32::NEG_INFINITY; 10];
    let (idx, log_prob) = sample_token(&logits, 1.0, None);
    assert!(idx < 10, "index must be valid");
    // Fallback path calls compute_log_prob on all-NEG_INFINITY → NEG_INFINITY.
    assert!(
        log_prob == f32::NEG_INFINITY,
        "log_prob should be NEG_INFINITY for all-NEG_INFINITY input, got {log_prob}"
    );
}

// -- DecodeConfig::validate() max_length guard (#1639) --

#[test]
fn test_decode_config_max_length_zero_rejected() {
    let config = DecodeConfig {
        max_length: 0,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("max_length"),
        "should reject max_length=0, got: {msg}"
    );
}

#[test]
fn test_decode_config_max_length_one_accepted() {
    let config = DecodeConfig {
        max_length: 1,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    assert!(config.validate().is_ok(), "max_length=1 should be valid");
}

// -- max_length upper bound test (#1645 AC3) --

#[test]
fn test_decode_config_max_length_exceeds_limit_rejected() {
    let config = DecodeConfig {
        max_length: 225, // > MAX_DECODE_LENGTH (224)
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("exceeds limit"),
        "should reject max_length > 224, got: {msg}"
    );
}

#[test]
fn test_decode_config_max_length_at_limit_accepted() {
    let config = DecodeConfig {
        max_length: 224, // == MAX_DECODE_LENGTH
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    assert!(config.validate().is_ok(), "max_length=224 should be valid");
}

// -- argmax_f32 boundary tests --

#[test]
fn test_argmax_f32_empty_returns_zero() {
    // Empty slice should return 0 via unwrap_or(0).
    let values: &[f32] = &[];
    assert_eq!(argmax_f32(values), 0);
}

#[test]
fn test_argmax_f32_single_element() {
    assert_eq!(argmax_f32(&[42.0]), 0);
}

// -- compute_log_prob boundary tests --

#[test]
fn test_compute_log_prob_empty_slice_returns_neg_infinity() {
    let logits: &[f32] = &[];
    let lp = compute_log_prob(logits, 0);
    assert_eq!(lp, f32::NEG_INFINITY);
}

#[test]
fn test_compute_log_prob_out_of_bounds_idx_returns_neg_infinity() {
    let logits = [1.0_f32, 2.0, 3.0];
    let lp = compute_log_prob(&logits, 10);
    assert_eq!(lp, f32::NEG_INFINITY);
}

// -- compute_no_speech_prob guard path tests --

#[test]
fn test_compute_no_speech_prob_token_beyond_vocab() {
    use super::super::language::compute_no_speech_prob;
    // NO_SPEECH_TOKEN = 50363. A logit slice shorter than that should return 0.0.
    let logits = vec![1.0_f32; 100]; // length 100 < 50363
    assert_eq!(compute_no_speech_prob(&logits), 0.0);
}

#[test]
fn test_compute_no_speech_prob_all_neg_infinity() {
    use super::super::language::compute_no_speech_prob;
    // All logits are NEG_INFINITY → max_val == NEG_INFINITY → returns 0.0.
    let logits = vec![f32::NEG_INFINITY; 51000];
    assert_eq!(compute_no_speech_prob(&logits), 0.0);
}

#[test]
fn test_compute_no_speech_prob_normal_logits() {
    use super::super::language::compute_no_speech_prob;
    // Build logits where NO_SPEECH_TOKEN (50363) has highest value.
    let mut logits = vec![0.0_f32; 51000];
    logits[50363] = 20.0; // Make no-speech token dominant (e^20 >> 51000).
    let prob = compute_no_speech_prob(&logits);
    assert!(prob > 0.5, "no-speech token should dominate, got {prob}");
    assert!(prob <= 1.0);
}

#[test]
fn test_compute_no_speech_prob_uniform_logits() {
    use super::super::language::compute_no_speech_prob;
    // All logits equal → prob = 1/vocab_size.
    let logits = vec![0.0_f32; 51000];
    let prob = compute_no_speech_prob(&logits);
    let expected = 1.0 / 51000.0;
    assert!(
        (prob - expected).abs() < 1e-4,
        "uniform logits should give ~1/vocab_size, got {prob}"
    );
}

// -- sample_token with all-NEG_INFINITY and RNG present --

#[test]
fn test_sample_token_all_neg_infinity_with_rng_fallback() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    // With positive temperature and an RNG, all-NEG_INFINITY logits hit the
    // `!sum.is_finite()` guard and fall back to argmax (not categorical_sample).
    let logits = vec![f32::NEG_INFINITY; 10];
    let mut rng = StdRng::seed_from_u64(42);
    let (idx, log_prob) = sample_token(&logits, 1.0, Some(&mut rng));
    assert!(idx < 10, "index must be valid");
    assert_eq!(
        log_prob,
        f32::NEG_INFINITY,
        "fallback path should produce NEG_INFINITY log-prob"
    );
}

// -- Temperature validation tests (AC3 of #1640) --

#[test]
fn test_decode_with_temperature_negative_rejected() {
    use crate::test_utils::{tiny_encoder_output, tiny_model};
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();
    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let err = decode_with_temperature(&mut model, &encoder_output, &config, -0.5).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("temperature"),
        "should reject negative temperature, got: {msg}"
    );
}

#[test]
fn test_decode_with_temperature_nan_rejected() {
    use crate::test_utils::{tiny_encoder_output, tiny_model};
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();
    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let err = decode_with_temperature(&mut model, &encoder_output, &config, f64::NAN).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("temperature"),
        "should reject NaN temperature, got: {msg}"
    );
}

#[test]
fn test_decode_with_temperature_inf_rejected() {
    use crate::test_utils::{tiny_encoder_output, tiny_model};
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();
    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let err =
        decode_with_temperature(&mut model, &encoder_output, &config, f64::INFINITY).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("temperature"),
        "should reject Inf temperature, got: {msg}"
    );
}

#[test]
fn test_decode_with_temperature_neg_inf_rejected() {
    use crate::test_utils::{tiny_encoder_output, tiny_model};
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();
    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    let err = decode_with_temperature(&mut model, &encoder_output, &config, f64::NEG_INFINITY)
        .unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("temperature"),
        "should reject -Inf temperature, got: {msg}"
    );
}

#[test]
fn test_decode_with_temperature_zero_accepted() {
    use crate::test_utils::{tiny_encoder_output, tiny_model};
    let mut model = tiny_model();
    let encoder_output = tiny_encoder_output();
    let config = DecodeConfig {
        max_length: 3,
        initial_tokens: vec![0],
        suppress_tokens: Vec::new(),
        ..Default::default()
    };
    // Temperature 0.0 is valid (greedy decode).
    let result = decode_with_temperature(&mut model, &encoder_output, &config, 0.0);
    assert!(result.is_ok(), "temperature 0.0 should be accepted");
}

// -- sample_token f64→f32 overflow guard tests (W3-91 / #1648 AC2) --

#[test]
fn test_sample_token_large_f64_temperature_falls_back_to_greedy() {
    // A finite f64 temperature > f32::MAX overflows to f32::INFINITY on cast.
    // The guard in sample_token (decode_helpers.rs:54-58) catches this and
    // falls back to greedy argmax.
    let logits = [1.0_f32, 5.0, 2.0, 3.0];
    let (idx, log_prob) = sample_token(&logits, 1e39, None);
    // Should pick index 1 (highest logit), same as greedy.
    assert_eq!(idx, 1, "overflow temperature should fall back to greedy");
    assert!(log_prob.is_finite(), "log_prob should be finite");
    assert!(log_prob <= 0.0, "log_prob should be non-positive");
}

#[test]
fn test_sample_token_subnormal_temperature_falls_back_to_greedy() {
    // A subnormal f64 temperature rounds to 0.0 as f32.
    // The guard catches temp_f32 == 0.0 and falls back to greedy.
    let logits = [1.0_f32, 5.0, 2.0, 3.0];
    let subnormal_temp = 1e-46_f64; // f64 subnormal, rounds to 0.0 as f32
    let (idx, log_prob) = sample_token(&logits, subnormal_temp, None);
    assert_eq!(idx, 1, "subnormal temperature should fall back to greedy");
    assert!(log_prob.is_finite(), "log_prob should be finite");
}
