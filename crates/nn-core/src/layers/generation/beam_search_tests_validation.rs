// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Validation and NaN guard tests for beam search.
//!
//! Extracted from `beam_search_tests.rs` for file size compliance.

use crate::dyn_tensor::DynTensor;
use crate::layers::kv_cache::KvCache;
use crate::Device;

use super::super::{beam_search, BeamSearchConfig};

/// Re-use the constant_model from the parent test module.
fn constant_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    let mut logits = vec![0.0f32; 5];
    logits[0] = 10.0;
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

#[test]
fn test_beam_search_config_validate_default() {
    let config = BeamSearchConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_beam_search_config_validate_nan_length_penalty() {
    let config = BeamSearchConfig {
        length_penalty: f64::NAN,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("length_penalty") && msg.contains("finite"),
        "expected error about non-finite length_penalty, got: {msg}"
    );
}

#[test]
fn test_beam_search_config_validate_inf_length_penalty() {
    let config = BeamSearchConfig {
        length_penalty: f64::INFINITY,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let config_neg = BeamSearchConfig {
        length_penalty: f64::NEG_INFINITY,
        ..Default::default()
    };
    assert!(config_neg.validate().is_err());
}

#[test]
fn test_beam_search_nan_length_penalty_rejected() {
    let config = BeamSearchConfig {
        length_penalty: f64::NAN,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(constant_model, &[0], &mut cache, &config, &Device::Cpu);
    assert!(result.is_err(), "NaN length_penalty should be rejected");
}

/// Model that returns all NEG_INFINITY logits — exercises the log_softmax NaN guard.
fn all_neg_inf_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    let logits = vec![f32::NEG_INFINITY; 5];
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

#[test]
fn test_beam_search_all_neg_inf_logits_no_nan() {
    // When model returns all -inf logits, log_softmax should produce -inf (not NaN).
    // beam_search should not panic; accumulated log_prob will be -inf.
    let config = BeamSearchConfig {
        beam_width: 2,
        max_new_tokens: 2,
        length_penalty: 0.0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(all_neg_inf_model, &[0], &mut cache, &config, &Device::Cpu);
    // Should not panic — returns Ok with beams having -inf log_prob
    let output = result.expect("should not panic on all-neg-inf logits");
    for beam in &output.beams {
        // log_prob should be -inf (not NaN)
        assert!(
            !beam.log_prob.is_nan(),
            "beam log_prob should not be NaN, got {}",
            beam.log_prob
        );
    }
}

#[test]
fn test_log_softmax_all_neg_inf() {
    // Direct test of the helper function
    use super::super::helpers::log_softmax;
    let result = log_softmax(&[f32::NEG_INFINITY; 5]);
    for &v in &result {
        assert!(
            !v.is_nan(),
            "log_softmax of all -inf should not produce NaN, got {v}"
        );
        assert_eq!(v, f32::NEG_INFINITY);
    }
}

#[test]
fn test_log_softmax_normal_values() {
    // Verify normal case still works correctly
    use super::super::helpers::log_softmax;
    let result = log_softmax(&[1.0, 2.0, 3.0]);
    // Should sum to approximately 1.0 when exponentiated
    let sum_exp: f32 = result.iter().map(|&v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-5,
        "exp(log_softmax) should sum to 1, got {sum_exp}"
    );
    // Largest input should have largest log-probability
    assert!(result[2] > result[1]);
    assert!(result[1] > result[0]);
}

#[test]
fn test_log_softmax_empty() {
    use super::super::helpers::log_softmax;
    let result = log_softmax(&[]);
    assert!(result.is_empty());
}

/// Verify NaN logits are sanitized to -inf before computation.
/// Without sanitization, a single NaN poisons the entire sum through
/// (NaN - max).exp() = NaN, producing NaN for ALL outputs.
#[test]
fn test_log_softmax_nan_sanitization() {
    use super::super::helpers::log_softmax;
    // Token 0: NaN, Token 1: 5.0, Token 2: 3.0
    let result = log_softmax(&[f32::NAN, 5.0, 3.0]);
    // NaN token should get -inf (impossible token)
    assert_eq!(result[0], f32::NEG_INFINITY, "NaN logit should become -inf");
    // Finite tokens should get valid log-probabilities
    assert!(result[1].is_finite(), "finite logit should remain finite");
    assert!(result[2].is_finite(), "finite logit should remain finite");
    // Token 1 (logit 5.0) should have higher probability than token 2 (logit 3.0)
    assert!(
        result[1] > result[2],
        "higher logit should have higher log_prob"
    );
    // Finite outputs should sum to ~1.0 when exponentiated
    let sum_exp: f32 = result[1..].iter().map(|&v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-5,
        "exp(finite log_softmax outputs) should sum to ~1, got {sum_exp}"
    );
}

/// Model that returns identical logits for all tokens (tie-breaking test).
fn uniform_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    let logits = vec![5.0f32; 5];
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

/// Verify beam search produces deterministic output when all tokens have equal probability.
/// After log_softmax of uniform logits, all tokens have log_prob = ln(1/5) ≈ -1.609.
/// Tie-breaking should be stable across runs.
#[test]
fn test_beam_search_uniform_logits_deterministic() {
    let config = BeamSearchConfig {
        beam_width: 3,
        max_new_tokens: 3,
        length_penalty: 0.0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(uniform_model, &[0], &mut cache, &config, &Device::Cpu)
        .expect("uniform logits should not cause panic");

    // All beams should have finite log_prob
    for beam in &result.beams {
        assert!(
            beam.log_prob.is_finite(),
            "uniform logits should produce finite log_prob, got {}",
            beam.log_prob
        );
        // With uniform logits, all beams have the same accumulated probability
        assert!(beam.log_prob < 0.0, "log_prob should be negative");
    }

    // Run again — output should be identical (deterministic tie-breaking)
    let mut cache2 = KvCache::new(1);
    let result2 = beam_search(uniform_model, &[0], &mut cache2, &config, &Device::Cpu)
        .expect("second run should also succeed");
    assert_eq!(
        result.beams.len(),
        result2.beams.len(),
        "beam count should be deterministic"
    );
    for (b1, b2) in result.beams.iter().zip(result2.beams.iter()) {
        assert_eq!(
            b1.token_ids, b2.token_ids,
            "token sequences should be deterministic for uniform logits"
        );
    }
}

/// Verify log_softmax handles a single +Inf input: the +Inf position gets
/// log(1/1) = 0.0, all others get -inf.
#[test]
fn test_log_softmax_single_pos_inf() {
    use super::super::helpers::log_softmax;
    let result = log_softmax(&[f32::INFINITY, 5.0, 3.0]);
    // +Inf position should get log(1/1) = 0.0
    assert!(
        result[0].is_finite(),
        "+Inf position should get finite log-prob, got {}",
        result[0]
    );
    assert!(
        (result[0] - 0.0).abs() < 1e-6,
        "+Inf with count=1 should give log(1) = 0.0, got {}",
        result[0]
    );
    // Finite positions should get -inf (dominated by +inf)
    assert_eq!(result[1], f32::NEG_INFINITY, "finite logit should get -inf");
    assert_eq!(result[2], f32::NEG_INFINITY, "finite logit should get -inf");
}

/// Verify log_softmax distributes probability among multiple +Inf positions.
/// Two +Inf positions → each gets log(1/2) = -ln(2) ≈ -0.693.
#[test]
fn test_log_softmax_multiple_pos_inf() {
    use super::super::helpers::log_softmax;
    let result = log_softmax(&[f32::INFINITY, f32::INFINITY, 3.0]);
    let expected = -(2.0_f32).ln(); // -ln(2) ≈ -0.693
    assert!(
        (result[0] - expected).abs() < 1e-6,
        "+Inf position[0] should get -ln(2), got {}",
        result[0]
    );
    assert!(
        (result[1] - expected).abs() < 1e-6,
        "+Inf position[1] should get -ln(2), got {}",
        result[1]
    );
    assert_eq!(
        result[2],
        f32::NEG_INFINITY,
        "finite logit should get -inf when +inf present"
    );
    // Both +Inf outputs should exponentiate to sum to 1.0
    let sum_exp: f32 = result[0..2].iter().map(|&v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-5,
        "exp of +Inf log_softmax outputs should sum to 1.0, got {sum_exp}"
    );
}

/// Verify all +Inf input: each position gets log(1/N) = -ln(N).
#[test]
fn test_log_softmax_all_pos_inf() {
    use super::super::helpers::log_softmax;
    let result = log_softmax(&[f32::INFINITY; 4]);
    let expected = -(4.0_f32).ln();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-6,
            "all +Inf: position {i} should get -ln(4), got {v}"
        );
    }
}

/// Model that returns NaN for some logit positions.
/// Exercises the total_cmp sorting path — NaN must not cause panic.
fn nan_logit_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    // Token 0: NaN, Token 1: 5.0, Token 2: 3.0, Token 3: -inf, Token 4: 1.0
    let logits = vec![f32::NAN, 5.0, 3.0, f32::NEG_INFINITY, 1.0];
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

/// Verify beam search does not panic when model returns logits containing NaN.
/// The log_softmax of NaN input produces NaN output. The total_cmp sorting
/// should handle NaN without panic — NaN sorts after all other values.
#[test]
fn test_beam_search_nan_logits_no_panic() {
    let config = BeamSearchConfig {
        beam_width: 2,
        max_new_tokens: 2,
        length_penalty: 0.0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    // Should not panic regardless of NaN handling strategy
    let result = beam_search(nan_logit_model, &[0], &mut cache, &config, &Device::Cpu);
    // The key property: no panic from sorting or comparison operations.
    // Additionally verify: if beams are produced, non-NaN tokens should be
    // preferred (token 1 has highest finite logit = 5.0).
    if let Ok(output) = result {
        assert!(!output.beams.is_empty(), "should have at least one beam");
        // Best beam's first token should be token 1 (logit 5.0), not the NaN token 0
        let best = &output.beams[0];
        assert_eq!(
            best.token_ids[0], 1,
            "best beam first token should be token 1 (logit 5.0), got {}",
            best.token_ids[0]
        );
        // log_prob should be finite (NaN tokens should lose to finite ones)
        assert!(
            best.log_prob.is_finite(),
            "best beam log_prob should be finite, got {}",
            best.log_prob
        );
    }
}
