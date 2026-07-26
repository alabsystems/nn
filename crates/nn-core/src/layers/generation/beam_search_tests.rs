#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::kv_cache::KvCache;
use crate::{DType, Device};

use super::{beam_search, BeamSearchConfig};

/// Model that always returns token 0 with highest logit.
/// Makes beam search deterministic: all beams produce [0, 0, 0, ...].
fn constant_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    let mut logits = vec![0.0f32; 5];
    logits[0] = 10.0;
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

/// Model that returns logits where token (step % vocab) has highest logit.
/// Uses the last input token as a step counter.
fn deterministic_model(input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    // ids_to_tensor creates U32 tensors, so convert to f32 first.
    let input_f32 = input.to_dtype(DType::F32)?;
    let flat = input_f32.to_flat_vec::<f32>()?;
    let last_val = flat[flat.len() - 1];
    let next_token = (last_val as usize + 1) % 5;
    let mut logits = vec![0.0f32; 5];
    logits[next_token] = 10.0;
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

/// Model that always returns token 2 (the EOS token) with highest logit.
fn eos_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    let mut logits = vec![0.0f32; 5];
    logits[2] = 10.0;
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

// Static counter for delayed_eos_model (reset before each test).
static DELAYED_EOS_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Model that emits non-EOS on first call (prefill) and EOS on subsequent calls.
/// Call 0 (prefill): tokens 0 (best) and 1 (second best) — both non-EOS.
/// Call 1+ (decode): token 2 (EOS) is dominant.
fn delayed_eos_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    let call = DELAYED_EOS_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut logits = vec![0.0f32; 5];
    if call == 0 {
        logits[0] = 9.0;
        logits[1] = 8.0;
        logits[3] = 4.0;
        logits[2] = 1.0; // EOS distant
    } else {
        logits[2] = 10.0; // EOS
        logits[0] = 2.0;
    }
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

/// Model that returns 3D logits [1, seq_len, vocab].
fn model_3d_logits(input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    let seq_len = input.dim(1)?;
    let mut data = vec![0.0f32; seq_len * 5];
    data[(seq_len - 1) * 5 + 3] = 10.0;
    DynTensor::from_vec(data, &[1, seq_len, 5], &Device::Cpu)
}

// -- BeamSearchConfig tests ---------------------------------------------------

#[test]
fn test_beam_search_config_default() {
    let config = BeamSearchConfig::default();
    assert_eq!(config.beam_width, 4);
    assert_eq!(config.max_new_tokens, 128);
    assert_eq!(config.length_penalty, 1.0);
    assert!(!config.early_stopping);
    assert!(config.eos_token_id.is_none());
}

// -- beam_search() tests ------------------------------------------------------

#[test]
fn test_beam_search_empty_prompt() {
    let config = BeamSearchConfig::default();
    let mut cache = KvCache::new(1);
    let result = beam_search(constant_model, &[], &mut cache, &config, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_beam_search_zero_beam_width() {
    let config = BeamSearchConfig {
        beam_width: 0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(constant_model, &[0], &mut cache, &config, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_beam_search_zero_tokens() {
    let config = BeamSearchConfig {
        max_new_tokens: 0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(constant_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert_eq!(result.beams.len(), 1);
    assert!(result.beams[0].token_ids.is_empty());
}

#[test]
fn test_beam_search_width_1_matches_greedy() {
    // Beam search with width=1 should behave like greedy decoding
    let config = BeamSearchConfig {
        beam_width: 1,
        max_new_tokens: 3,
        length_penalty: 0.0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(deterministic_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert_eq!(result.beams.len(), 1);
    assert_eq!(result.beams[0].token_ids.len(), 3);

    // Compare with greedy generation
    let greedy_config = crate::layers::autoregressive::GenerationConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };
    let mut greedy_cache = KvCache::new(1);
    let greedy_result = crate::layers::autoregressive::generate(
        deterministic_model,
        &[0],
        &mut greedy_cache,
        &greedy_config,
        &Device::Cpu,
    )
    .unwrap();

    assert_eq!(result.beams[0].token_ids, greedy_result.token_ids);
}

#[test]
fn test_beam_search_returns_beam_width_results() {
    let config = BeamSearchConfig {
        beam_width: 3,
        max_new_tokens: 2,
        length_penalty: 0.0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(deterministic_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    // With beam_width=3 and max_new_tokens=2 (deterministic non-EOS model),
    // finalize_tree truncates to beam_width. Expect exactly 3 beams.
    assert_eq!(
        result.beams.len(),
        3,
        "expected exactly beam_width=3 results, got {}",
        result.beams.len()
    );
}

#[test]
fn test_beam_search_results_sorted_by_score() {
    let config = BeamSearchConfig {
        beam_width: 3,
        max_new_tokens: 3,
        length_penalty: 1.0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(constant_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();

    // Scores should be in descending order
    for i in 1..result.beams.len() {
        let score_prev = result.beams[i - 1].log_prob
            / (result.beams[i - 1].token_ids.len() as f64).powf(config.length_penalty);
        let score_curr = result.beams[i].log_prob
            / (result.beams[i].token_ids.len() as f64).powf(config.length_penalty);
        assert!(
            score_prev >= score_curr,
            "beams not sorted: {score_prev} < {score_curr}"
        );
    }
}

#[test]
fn test_beam_search_early_stopping() {
    // EOS model: all beams immediately finish
    let config = BeamSearchConfig {
        beam_width: 3,
        max_new_tokens: 10,
        early_stopping: true,
        eos_token_id: Some(2),
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(eos_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    // All beams should be finished (EOS reached)
    assert!(!result.beams.is_empty());
    for beam in &result.beams {
        assert!(beam.finished);
    }
    // Best beam should have gotten EOS immediately (token 2 is highest logit)
    assert_eq!(result.beams[0].token_ids.len(), 1);
    assert_eq!(result.beams[0].token_ids[0], 2);
}

#[test]
fn test_beam_search_no_early_stopping_continues() {
    // delayed_eos_model: call 0 (prefill) → non-EOS, call 1+ → EOS.
    // With beam_width=2: prefill produces beams [token 0, token 1] (non-EOS).
    // Decode step 1: both beams get EOS → 2 completed >= beam_width=2.
    // early_stopping=true: stops immediately. early_stopping=false: also stops
    // (no active beams remain). Both produce identical finished beams.
    DELAYED_EOS_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    let config_early = BeamSearchConfig {
        beam_width: 2,
        max_new_tokens: 5,
        early_stopping: true,
        eos_token_id: Some(2),
        length_penalty: 0.0,
        ..Default::default()
    };
    let mut cache_early = KvCache::new(1);
    let result_early = beam_search(
        delayed_eos_model,
        &[0],
        &mut cache_early,
        &config_early,
        &Device::Cpu,
    )
    .unwrap();
    let early_calls = DELAYED_EOS_CALLS.load(std::sync::atomic::Ordering::Relaxed);

    DELAYED_EOS_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    let config_no_early = BeamSearchConfig {
        beam_width: 2,
        max_new_tokens: 5,
        early_stopping: false,
        eos_token_id: Some(2),
        length_penalty: 0.0,
        ..Default::default()
    };
    let mut cache_no_early = KvCache::new(1);
    let result_no_early = beam_search(
        delayed_eos_model,
        &[0],
        &mut cache_no_early,
        &config_no_early,
        &Device::Cpu,
    )
    .unwrap();
    let no_early_calls = DELAYED_EOS_CALLS.load(std::sync::atomic::Ordering::Relaxed);

    assert!(!result_early.beams.is_empty());
    assert!(!result_no_early.beams.is_empty());

    // Both should have finished beams.
    assert!(
        result_early.beams.iter().any(|b| b.finished),
        "early stopping should produce finished beams"
    );
    assert!(
        result_no_early.beams.iter().any(|b| b.finished),
        "no early stopping should produce finished beams"
    );

    // Both should use same call count (all beams complete simultaneously).
    assert_eq!(
        early_calls, no_early_calls,
        "both variants should use same call count: early={early_calls}, no_early={no_early_calls}"
    );

    // Output should be identical: same model, same params, same completion point.
    assert_eq!(result_early.beams.len(), result_no_early.beams.len());
    for (i, (e, n)) in result_early
        .beams
        .iter()
        .zip(result_no_early.beams.iter())
        .enumerate()
    {
        assert_eq!(e.token_ids, n.token_ids, "beam {i} token_ids should match");
    }
}

#[test]
fn test_beam_search_eos_model_all_finish() {
    // eos_model: EOS (token 2) is always dominant. With beam_width=2:
    //   Prefill top-2: token 2 (EOS, logit 10) and token 0 (logit 0).
    //   Token 2 finishes at step 1. Token 0 stays active.
    //   Decode: token 0 → EOS at step 2. All beams finished.
    let config = BeamSearchConfig {
        beam_width: 2,
        max_new_tokens: 3,
        early_stopping: false,
        eos_token_id: Some(2),
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(eos_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert!(!result.beams.is_empty());
    for beam in &result.beams {
        assert!(beam.finished, "all beams should finish with eos_model");
    }
    // Best beam should start with EOS (shortest path, highest score).
    assert_eq!(result.beams[0].token_ids[0], 2);
}

#[test]
fn test_beam_search_length_penalty_zero() {
    // No length normalization: raw log-prob ordering.
    // Verify penalty=0 means scores ARE the raw log_prob (no normalization).
    let config = BeamSearchConfig {
        beam_width: 2,
        max_new_tokens: 3,
        length_penalty: 0.0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(constant_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert!(!result.beams.is_empty());
    for beam in &result.beams {
        assert!(beam.log_prob <= 0.0, "log_prob should be <= 0");
        // With penalty=0, len^0 == 1, so score == raw log_prob.
        // Compare via the score() method indirectly: beams are sorted by
        // score(length_penalty), so with penalty=0 they must be sorted by
        // raw log_prob descending.
    }
    // Verify descending raw log_prob ordering (penalty=0 means no normalization).
    for i in 1..result.beams.len() {
        assert!(
            result.beams[i - 1].log_prob >= result.beams[i].log_prob,
            "with penalty=0, beams should be sorted by raw log_prob: {} < {}",
            result.beams[i - 1].log_prob,
            result.beams[i].log_prob,
        );
    }
}

#[test]
fn test_beam_search_3d_logits() {
    // 3D logits should work (extracts last position)
    let config = BeamSearchConfig {
        beam_width: 2,
        max_new_tokens: 2,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(model_3d_logits, &[0, 1], &mut cache, &config, &Device::Cpu).unwrap();
    assert!(!result.beams.is_empty());
    // Best beam should start with token 3 (highest logit in model_3d_logits)
    assert_eq!(result.beams[0].token_ids[0], 3);
}

// -- Edge case tests ----------------------------------------------------------

#[test]
fn test_beam_search_width_1_with_eos() {
    // Width-1 (greedy) + EOS token: beam should finish after one step.
    let config = BeamSearchConfig {
        beam_width: 1,
        max_new_tokens: 10,
        early_stopping: true,
        eos_token_id: Some(2),
        length_penalty: 0.0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(eos_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert_eq!(result.beams.len(), 1);
    assert!(result.beams[0].finished);
    assert_eq!(result.beams[0].token_ids, vec![2]);
}

/// Model that returns very negative logits (near -inf log_prob accumulation).
fn extreme_negative_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    // All logits are large negative values. log_softmax of these
    // will be close to log(1/5) ≈ -1.6 per step when logits are equal,
    // but with one token slightly higher, cumulative log_prob goes deeply negative.
    let logits = vec![-100.0, -100.1, -100.2, -100.3, -100.4];
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

#[test]
fn test_beam_search_cumulative_log_prob_deeply_negative() {
    // Test that beam search handles deeply negative cumulative log_prob
    // without NaN or panic. After many steps of log_softmax with similar
    // logits, log_prob accumulates to large negative values.
    let config = BeamSearchConfig {
        beam_width: 2,
        max_new_tokens: 50,
        length_penalty: 0.0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(
        extreme_negative_model,
        &[0],
        &mut cache,
        &config,
        &Device::Cpu,
    )
    .unwrap();
    assert!(!result.beams.is_empty());
    for beam in &result.beams {
        assert_eq!(beam.token_ids.len(), 50);
        // log_prob should be finite (deeply negative but not -inf or NaN).
        assert!(
            beam.log_prob.is_finite(),
            "log_prob is not finite: {}",
            beam.log_prob
        );
        assert!(beam.log_prob < 0.0, "log_prob should be negative");
    }
}

// Validation and NaN guard tests extracted to beam_search_tests_validation.rs.
#[path = "beam_search_tests_validation.rs"]
mod validation;

// Extended tests (hypothesis scoring, error propagation, performance proofs)
// extracted to beam_search_tests_extended.rs.
#[path = "beam_search_tests_extended.rs"]
mod extended;
