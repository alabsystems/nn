#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended beam search tests: hypothesis scoring, error propagation,
//! edge cases, beam-vs-greedy comparison.
//!
//! Performance proofs extracted to `beam_search_tests_perf.rs`.

use crate::dyn_tensor::DynTensor;
use crate::layers::kv_cache::KvCache;
use crate::Device;

use super::super::{beam_search, BeamSearchConfig};

// -- BeamHypothesis scoring tests ---------------------------------------------

#[test]
fn test_beam_search_hypothesis_score() {
    use super::super::BeamHypothesis;

    let hyp = BeamHypothesis {
        token_ids: vec![1, 2, 3],
        log_prob: -6.0,
        finished: false,
    };

    // length_penalty=0 → raw log_prob
    assert_eq!(hyp.score(0.0), -6.0);

    // length_penalty=1.0 → log_prob / len
    assert!((hyp.score(1.0) - (-2.0)).abs() < 1e-10);

    // length_penalty=2.0 → log_prob / len^2
    assert!((hyp.score(2.0) - (-6.0 / 9.0)).abs() < 1e-10);
}

#[test]
fn test_beam_search_hypothesis_score_empty() {
    use super::super::BeamHypothesis;

    // Use non-zero log_prob so the test is non-degenerate: 0.0/anything == 0.0
    // would trivially pass even if the empty-guard were missing.
    let hyp = BeamHypothesis {
        token_ids: vec![],
        log_prob: -5.0,
        finished: false,
    };

    // Empty tokens: should return raw log_prob regardless of penalty.
    // Without the empty guard, score(1.0) would compute -5.0 / 0^1 = -inf or NaN.
    assert_eq!(hyp.score(0.0), -5.0);
    assert_eq!(hyp.score(1.0), -5.0);
    assert_eq!(hyp.score(2.0), -5.0);
}

// -- Error propagation --------------------------------------------------------

#[test]
fn test_beam_search_model_error_propagates() {
    fn failing_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
        Err(crate::TensorError::InvalidShape("mock error".into()))
    }

    let config = BeamSearchConfig {
        beam_width: 2,
        max_new_tokens: 5,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(failing_model, &[0], &mut cache, &config, &Device::Cpu);
    assert!(result.is_err());
}

// -- Performance proofs (extracted to beam_search_tests_perf.rs) ---------------

#[path = "beam_search_tests_perf.rs"]
mod perf;

// -- Edge case tests ----------------------------------------------------------

/// AC5: beam_width > vocab_size — beam search should not panic and returns
/// at most vocab_size distinct initial beams (top_k capped by vocab).
#[test]
fn test_beam_search_beam_width_exceeds_vocab_size() {
    // Vocab = 3, beam_width = 5: more beams requested than vocab items.
    fn small_vocab_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
        DynTensor::from_vec(vec![5.0f32, 3.0, 1.0], &[1, 3], &Device::Cpu)
    }

    let config = BeamSearchConfig {
        beam_width: 5,
        max_new_tokens: 3,
        ..Default::default()
    };
    let mut cache = KvCache::new(0);
    let result = beam_search(small_vocab_model, &[0], &mut cache, &config, &Device::Cpu);
    let output = result.expect("beam_width > vocab should not panic");
    // Initial top_k returns min(beam_width, vocab) = 3 beams, but
    // subsequent decode steps expand 3 × 3 = 9 candidates truncated to 5.
    // finalize_tree truncates to beam_width.
    assert!(
        output.beams.len() <= 5,
        "should return at most beam_width=5 beams, got {}",
        output.beams.len()
    );
    assert!(!output.beams.is_empty(), "should produce at least one beam");
    // All beams should have the same length (max_new_tokens-1=2 decode steps
    // + 1 initial token = 3 tokens, since no EOS).
    for beam in &output.beams {
        assert_eq!(
            beam.token_ids.len(),
            3,
            "each beam should have max_new_tokens=3 tokens"
        );
    }
}

/// AC6: length_penalty > 1.0 should favor longer sequences over shorter
/// ones at equal raw log_prob.
#[test]
fn test_beam_search_length_penalty_favors_longer() {
    use super::super::BeamHypothesis;

    // Two hypotheses with similar raw log_prob but different lengths.
    let short = BeamHypothesis {
        token_ids: vec![1],
        log_prob: -2.0,
        finished: true,
    };
    let long = BeamHypothesis {
        token_ids: vec![1, 2, 3, 4],
        log_prob: -4.0,
        finished: true,
    };

    // With penalty=0: short wins (raw log_prob: -2 > -4).
    assert!(short.score(0.0) > long.score(0.0));

    // With penalty=2.0: long wins because -4/4^2 = -0.25 > -2/1^2 = -2.
    let short_score = short.score(2.0);
    let long_score = long.score(2.0);
    assert!(
        long_score > short_score,
        "penalty=2.0 should favor longer: long={long_score:.4} vs short={short_score:.4}"
    );
}

/// AC7: Different beams should produce genuinely different token sequences
/// when the model provides distinct logit rankings.
#[test]
fn test_beam_search_divergent_sequences() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

    // Model where the top-2 tokens alternate per decode step,
    // so different beams follow different token paths.
    fn diverging_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
        let call = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        let mut logits = vec![0.0f32; 4];
        // Prefill (call 0): token 0 (best), token 1 (second best).
        // Decode steps: alternate which token pair is strongest.
        if call.is_multiple_of(2) {
            logits[0] = 10.0;
            logits[1] = 9.0;
        } else {
            logits[2] = 10.0;
            logits[3] = 9.0;
        }
        DynTensor::from_vec(logits, &[1, 4], &Device::Cpu)
    }

    CALL_COUNT.store(0, Ordering::Relaxed);

    let config = BeamSearchConfig {
        beam_width: 2,
        max_new_tokens: 4,
        ..Default::default()
    };
    let mut cache = KvCache::new(0);
    let result = beam_search(diverging_model, &[0], &mut cache, &config, &Device::Cpu)
        .expect("beam search should succeed");

    assert_eq!(result.beams.len(), 2, "expected 2 beams");

    // The two beams should have different token sequences.
    assert_ne!(
        result.beams[0].token_ids, result.beams[1].token_ids,
        "beams should produce different token sequences: {:?} vs {:?}",
        result.beams[0].token_ids, result.beams[1].token_ids,
    );
}

/// AC8: max_new_tokens = 1 — beam search should produce exactly one token
/// per beam (the first decode step only).
#[test]
fn test_beam_search_max_new_tokens_one() {
    fn simple_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
        // Token 3 is strongest, token 1 is second.
        DynTensor::from_vec(vec![0.0f32, 5.0, 0.0, 10.0, 0.0], &[1, 5], &Device::Cpu)
    }

    let config = BeamSearchConfig {
        beam_width: 3,
        max_new_tokens: 1,
        ..Default::default()
    };
    let mut cache = KvCache::new(0);
    let result = beam_search(simple_model, &[0], &mut cache, &config, &Device::Cpu)
        .expect("beam search should succeed");

    // With max_new_tokens=1, each beam has exactly 1 token.
    for (i, beam) in result.beams.iter().enumerate() {
        assert_eq!(
            beam.token_ids.len(),
            1,
            "beam {i} should have exactly 1 token, got {:?}",
            beam.token_ids
        );
    }

    // Best beam should be token 3 (highest logit).
    assert_eq!(
        result.beams[0].token_ids[0], 3,
        "best beam should be token 3, got {}",
        result.beams[0].token_ids[0]
    );
}

// -- AC3: beam_search vs greedy on branching logits ---------------------------

/// AC3 (#1619): Beam search finds a globally better path than greedy.
/// Prefill: token 0=5.0 (greedy pick), token 1=4.9. After token 0: poor
/// continuation (token 3=2.0). After token 1: excellent (token 4=10.0).
/// Greedy picks [0,3]. Beam (width=2) discovers [1,4] has higher total log-prob.
#[test]
fn test_beam_search_finds_better_path_than_greedy() {
    use crate::layers::generation::{generate, GenerationConfig};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static BEAM_CALLS: AtomicUsize = AtomicUsize::new(0);
    static GREEDY_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn branching_logits(call: usize, input: &DynTensor) -> crate::Result<DynTensor> {
        let mut logits = vec![0.0f32; 6];
        if call == 0 {
            logits[0] = 5.0; // greedy pick
            logits[1] = 4.9; // second-best
        } else {
            // Input is U32 (token IDs) — convert to get last token.
            let input_f32 = input.to_dtype(crate::DType::F32)?;
            let data = input_f32.to_flat_vec::<f32>()?;
            let last = *data.last().unwrap_or(&0.0) as usize;
            match last {
                0 => logits[3] = 2.0,  // poor continuation
                1 => logits[4] = 10.0, // excellent continuation
                _ => logits[2] = 1.0,
            }
        }
        DynTensor::from_vec(logits, &[1, 6], &Device::Cpu)
    }

    fn model_beam(input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
        branching_logits(BEAM_CALLS.fetch_add(1, Ordering::Relaxed), input)
    }
    fn model_greedy(input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
        branching_logits(GREEDY_CALLS.fetch_add(1, Ordering::Relaxed), input)
    }

    // Greedy decode: argmax at each step → [0, 3]
    GREEDY_CALLS.store(0, Ordering::Relaxed);
    let greedy_cfg = GenerationConfig {
        max_new_tokens: 2,
        ..Default::default()
    };
    let mut gc = KvCache::new(0);
    let greedy = generate(model_greedy, &[99], &mut gc, &greedy_cfg, &Device::Cpu)
        .expect("greedy should succeed");
    assert_eq!(
        greedy.token_ids,
        vec![0, 3],
        "greedy path: {:?}",
        greedy.token_ids
    );

    // Beam search decode: width=2 discovers [1, 4]
    BEAM_CALLS.store(0, Ordering::Relaxed);
    let beam_cfg = BeamSearchConfig {
        beam_width: 2,
        max_new_tokens: 2,
        length_penalty: 0.0,
        ..Default::default()
    };
    let mut bc = KvCache::new(0);
    let beam = beam_search(model_beam, &[99], &mut bc, &beam_cfg, &Device::Cpu)
        .expect("beam search should succeed");

    assert!(!beam.beams.is_empty(), "should return at least one beam");
    let best = &beam.beams[0];
    assert_eq!(
        best.token_ids,
        vec![1, 4],
        "best beam: {:?}",
        best.token_ids
    );

    // If greedy path [0,3] also appears, verify it has lower log-prob.
    if let Some(gb) = beam.beams.iter().find(|b| b.token_ids == vec![0, 3]) {
        assert!(
            best.log_prob > gb.log_prob,
            "[1,4] log_prob={:.4} should beat [0,3] log_prob={:.4}",
            best.log_prob,
            gb.log_prob
        );
    }
}

/// Regression test: beam search must reject empty vocabulary instead of
/// returning zero beams. Matches the autoregressive path guard at
/// autoregressive_token_sampler.rs:81-84.
#[test]
fn test_beam_search_empty_vocabulary_returns_error() {
    fn empty_vocab_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
        // Return logits with vocab_size=0: shape [1, 0].
        DynTensor::from_vec(Vec::<f32>::new(), &[1, 0], &Device::Cpu)
    }

    let config = BeamSearchConfig {
        beam_width: 3,
        max_new_tokens: 5,
        ..BeamSearchConfig::default()
    };
    let mut cache = KvCache::new(1);
    let result = beam_search(empty_vocab_model, &[0], &mut cache, &config, &Device::Cpu);
    assert!(
        result.is_err(),
        "beam_search should reject empty vocabulary"
    );
    let err_str = format!("{}", result.unwrap_err());
    assert!(
        err_str.contains("empty vocabulary"),
        "error should mention empty vocabulary, got: {err_str}"
    );
}
