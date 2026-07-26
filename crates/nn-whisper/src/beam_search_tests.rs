// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for the Whisper beam search module.
//!
//! Covers: config validation, normalize_score, top_k_log_probs, beam state
//! scoring, reconstruct_decoded, would_repeat_ngram, suppress_blank_tokens,
//! apply_ngram_blocking, BeamHypothesis/WhisperBeamOutput construction, and
//! single-beam (greedy) edge cases.
//!
//! All tests run without model weights by exercising internal helper functions.

use super::*;

// ---------------------------------------------------------------------------
// WhisperBeamSearchConfig defaults and validation
// ---------------------------------------------------------------------------

#[test]
fn test_config_default_values() {
    let config = WhisperBeamSearchConfig::default();
    assert_eq!(config.beam_width, 5);
    assert_eq!(config.max_tokens, 448);
    assert!((config.length_penalty - 1.0).abs() < 1e-6);
    assert_eq!(config.no_repeat_ngram_size, 0);
    assert!((config.temperature - 0.0).abs() < 1e-6);
    assert!(config.suppress_blank);
    assert_eq!(config.sot_token, SOT_TOKEN);
    assert_eq!(config.eot_token, EOT_TOKEN);
    assert_eq!(config.no_timestamps_token, NO_TIMESTAMPS_TOKEN);
    assert!(config.suppress_tokens.is_empty());
}

#[test]
fn test_config_validate_default_passes() {
    let config = WhisperBeamSearchConfig::default();
    config.validate().expect("default config should be valid");
}

#[test]
fn test_config_validate_zero_beam_width_fails() {
    let config = WhisperBeamSearchConfig {
        beam_width: 0,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.to_string().contains("beam_width"),
        "error should mention beam_width: {err}"
    );
}

#[test]
fn test_config_validate_zero_max_tokens_fails() {
    let config = WhisperBeamSearchConfig {
        max_tokens: 0,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(
        err.to_string().contains("max_tokens"),
        "error should mention max_tokens: {err}"
    );
}

#[test]
fn test_config_validate_nan_length_penalty_fails() {
    let config = WhisperBeamSearchConfig {
        length_penalty: f32::NAN,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_inf_length_penalty_fails() {
    let config = WhisperBeamSearchConfig {
        length_penalty: f32::INFINITY,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_neg_inf_length_penalty_fails() {
    let config = WhisperBeamSearchConfig {
        length_penalty: f32::NEG_INFINITY,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_nan_temperature_fails() {
    let config = WhisperBeamSearchConfig {
        temperature: f32::NAN,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_negative_temperature_fails() {
    let config = WhisperBeamSearchConfig {
        temperature: -0.1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_inf_temperature_fails() {
    let config = WhisperBeamSearchConfig {
        temperature: f32::INFINITY,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_zero_temperature_passes() {
    let config = WhisperBeamSearchConfig {
        temperature: 0.0,
        ..Default::default()
    };
    config.validate().expect("temperature=0 should be valid");
}

#[test]
fn test_config_validate_positive_temperature_passes() {
    let config = WhisperBeamSearchConfig {
        temperature: 0.8,
        ..Default::default()
    };
    config.validate().expect("temperature=0.8 should be valid");
}

#[test]
fn test_config_validate_single_beam_passes() {
    let config = WhisperBeamSearchConfig {
        beam_width: 1,
        ..Default::default()
    };
    config
        .validate()
        .expect("beam_width=1 (greedy) should be valid");
}

#[test]
fn test_config_validate_negative_length_penalty_passes() {
    let config = WhisperBeamSearchConfig {
        length_penalty: -1.0,
        ..Default::default()
    };
    config
        .validate()
        .expect("negative length_penalty should be valid (penalizes shorter)");
}

// ---------------------------------------------------------------------------
// normalize_score
// ---------------------------------------------------------------------------

#[test]
fn test_normalize_score_no_penalty() {
    let score = normalize_score(-10.0, 5, 0.0);
    assert!(
        (score - (-10.0)).abs() < 1e-6,
        "length_penalty=0 should return raw score"
    );
}

#[test]
fn test_normalize_score_with_penalty_one() {
    let score = normalize_score(-10.0, 4, 1.0);
    assert!(
        (score - (-10.0 / 4.0)).abs() < 1e-6,
        "length_penalty=1 should divide by length"
    );
}

#[test]
fn test_normalize_score_with_penalty_half() {
    let score = normalize_score(-10.0, 4, 0.5);
    let expected = -10.0 / (4.0_f32).powf(0.5);
    assert!(
        (score - expected).abs() < 1e-5,
        "length_penalty=0.5: expected {expected}, got {score}"
    );
}

#[test]
fn test_normalize_score_zero_length() {
    // When length is 0, score should be returned as-is regardless of penalty.
    let score = normalize_score(-5.0, 0, 1.0);
    assert!(
        (score - (-5.0)).abs() < 1e-6,
        "length=0 should return raw score"
    );
}

#[test]
fn test_normalize_score_length_one() {
    let score = normalize_score(-8.0, 1, 1.0);
    // 1^1 = 1, so normalized = -8.0 / 1.0 = -8.0
    assert!(
        (score - (-8.0)).abs() < 1e-6,
        "length=1 should not change score with penalty=1"
    );
}

#[test]
fn test_normalize_score_large_penalty_favors_longer() {
    // Higher length penalty penalizes shorter sequences more.
    let short_score = normalize_score(-6.0, 2, 2.0); // -6.0 / 2^2 = -1.5
    let long_score = normalize_score(-12.0, 4, 2.0); // -12.0 / 4^2 = -0.75
    assert!(
        long_score > short_score,
        "with high penalty, longer sequence (better per-token) should win"
    );
}

// ---------------------------------------------------------------------------
// BeamState::score (internal)
// ---------------------------------------------------------------------------

#[test]
fn test_beam_state_score_no_penalty() {
    let state = BeamState {
        node_idx: None,
        decoded_len: 5,
        sum_log_prob: -10.0,
        finished: false,
    };
    let score = state.score(0.0);
    assert!(
        (score - (-10.0)).abs() < 1e-10,
        "penalty=0 should return raw sum_log_prob"
    );
}

#[test]
fn test_beam_state_score_with_penalty() {
    let state = BeamState {
        node_idx: Some(0),
        decoded_len: 4,
        sum_log_prob: -10.0,
        finished: false,
    };
    let score = state.score(1.0);
    let expected = -10.0 / 4.0;
    assert!(
        (score - expected).abs() < 1e-10,
        "expected {expected}, got {score}"
    );
}

#[test]
fn test_beam_state_score_zero_decoded_len() {
    let state = BeamState {
        node_idx: None,
        decoded_len: 0,
        sum_log_prob: -5.0,
        finished: true,
    };
    // decoded_len == 0 returns raw sum_log_prob regardless of penalty.
    let score = state.score(1.0);
    assert!(
        (score - (-5.0)).abs() < 1e-10,
        "decoded_len=0 should return raw score"
    );
}

#[test]
fn test_beam_state_score_ordering() {
    // Two beams with same total score but different lengths.
    let short_beam = BeamState {
        node_idx: Some(0),
        decoded_len: 2,
        sum_log_prob: -4.0,
        finished: false,
    };
    let long_beam = BeamState {
        node_idx: Some(1),
        decoded_len: 4,
        sum_log_prob: -4.0,
        finished: false,
    };
    // With penalty=1.0: short = -4/2 = -2.0, long = -4/4 = -1.0
    // long_beam has higher normalized score (closer to 0).
    assert!(long_beam.score(1.0) > short_beam.score(1.0));
}

// ---------------------------------------------------------------------------
// reconstruct_decoded (parent-pointer tree)
// ---------------------------------------------------------------------------

#[test]
fn test_reconstruct_decoded_empty() {
    let tree: Vec<(usize, usize)> = vec![];
    let tokens = reconstruct_decoded(None, &tree);
    assert!(tokens.is_empty());
}

#[test]
fn test_reconstruct_decoded_single_token() {
    // Single root node: parent_plus_one=0, token=42
    let tree = vec![(0, 42)];
    let tokens = reconstruct_decoded(Some(0), &tree);
    assert_eq!(tokens, vec![42]);
}

#[test]
fn test_reconstruct_decoded_chain() {
    // Chain: node 0 (root, token=10) -> node 1 (parent=0, token=20) -> node 2 (parent=1, token=30)
    let tree = vec![
        (0, 10),     // idx=0: root, token 10
        (1, 20), // idx=1: parent=0 (0+1=1), token 20
        (1 + 1, 30), // idx=2: parent=1 (1+1=2), token 30
    ];
    let tokens = reconstruct_decoded(Some(2), &tree);
    assert_eq!(tokens, vec![10, 20, 30]);
}

#[test]
fn test_reconstruct_decoded_branching() {
    // Tree with a branch: node 0 -> node 1, node 0 -> node 2
    let tree = vec![
        (0, 10),     // idx=0: root, token 10
        (1, 20), // idx=1: parent=0, token 20
        (1, 30), // idx=2: parent=0, token 30 (branch!)
    ];
    // Reconstruct from node 1: [10, 20]
    assert_eq!(reconstruct_decoded(Some(1), &tree), vec![10, 20]);
    // Reconstruct from node 2: [10, 30]
    assert_eq!(reconstruct_decoded(Some(2), &tree), vec![10, 30]);
}

#[test]
fn test_reconstruct_all_tokens_with_initial() {
    let tree = vec![(0, 100)];
    let initial = vec![1, 2, 3];
    let all = reconstruct_all_tokens(&initial, Some(0), &tree);
    assert_eq!(all, vec![1, 2, 3, 100]);
}

#[test]
fn test_reconstruct_all_tokens_no_decoded() {
    let tree: Vec<(usize, usize)> = vec![];
    let initial = vec![50258, 50259, 50360, 50364];
    let all = reconstruct_all_tokens(&initial, None, &tree);
    assert_eq!(all, initial);
}

// ---------------------------------------------------------------------------
// top_k_log_probs
// ---------------------------------------------------------------------------

#[test]
fn test_top_k_empty_logits() {
    let result = top_k_log_probs(&[], 5, 0.0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, 0);
    assert_eq!(result[0].1, f32::NEG_INFINITY);
}

#[test]
fn test_top_k_single_element() {
    let logits = vec![2.0];
    let result = top_k_log_probs(&logits, 5, 0.0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, 0);
    // log_softmax of single element: log(1) = 0.0
    assert!((result[0].1 - 0.0).abs() < 1e-5);
}

#[test]
fn test_top_k_returns_at_most_k_elements() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = top_k_log_probs(&logits, 3, 0.0);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_top_k_sorted_descending() {
    let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let result = top_k_log_probs(&logits, 3, 0.0);
    // Should return top-3 indices sorted by descending log-prob.
    // Highest logits: index 1 (5.0), index 4 (4.0), index 2 (3.0)
    assert_eq!(result[0].0, 1);
    assert_eq!(result[1].0, 4);
    assert_eq!(result[2].0, 2);
    // Log-probs should be descending.
    assert!(result[0].1 >= result[1].1);
    assert!(result[1].1 >= result[2].1);
}

#[test]
fn test_top_k_log_probs_sum_to_one() {
    // For the full vocabulary (k >= vocab_size), exp(log_probs) should sum to ~1.
    let logits = vec![1.0, 2.0, 3.0];
    let result = top_k_log_probs(&logits, 10, 0.0);
    assert_eq!(result.len(), 3);
    let sum: f64 = result.iter().map(|(_, lp)| f64::from(*lp).exp()).sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "exp(log_probs) should sum to 1.0, got {sum}"
    );
}

#[test]
fn test_top_k_greedy_temperature_zero() {
    // Temperature 0 (greedy): should use raw logits for ordering.
    let logits = vec![0.1, 0.5, 0.3];
    let result = top_k_log_probs(&logits, 1, 0.0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, 1, "greedy should pick the highest logit");
}

#[test]
fn test_top_k_with_temperature() {
    // Temperature > 0 scales logits before softmax.
    let logits = vec![1.0, 2.0, 3.0];
    let result_cold = top_k_log_probs(&logits, 3, 0.1); // Very cold, near-greedy
    let result_hot = top_k_log_probs(&logits, 3, 10.0); // Very hot, near-uniform

    // With hot temperature, log-probs should be closer together (more uniform).
    let spread_cold = result_cold[0].1 - result_cold[2].1;
    let spread_hot = result_hot[0].1 - result_hot[2].1;
    assert!(
        spread_hot < spread_cold,
        "hot temperature should produce more uniform distribution"
    );
}

#[test]
fn test_top_k_all_neg_infinity() {
    let logits = vec![f32::NEG_INFINITY; 5];
    let result = top_k_log_probs(&logits, 3, 0.0);
    // Should return fallback value when max_val == NEG_INFINITY.
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1, f32::NEG_INFINITY);
}

#[test]
fn test_top_k_with_neg_infinity_entries() {
    // Mix of real logits and suppressed (NEG_INFINITY) entries.
    let logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, 2.0, 0.5];
    let result = top_k_log_probs(&logits, 3, 0.0);
    assert_eq!(result.len(), 3);
    // Top-3 by logit value: index 3 (2.0), index 1 (1.0), index 4 (0.5).
    assert_eq!(result[0].0, 3);
    assert_eq!(result[1].0, 1);
    assert_eq!(result[2].0, 4);
}

#[test]
fn test_top_k_k_larger_than_vocab() {
    let logits = vec![1.0, 2.0];
    let result = top_k_log_probs(&logits, 100, 0.0);
    // Should return all elements (2, not 100).
    assert_eq!(result.len(), 2);
}

// ---------------------------------------------------------------------------
// would_repeat_ngram
// ---------------------------------------------------------------------------

#[test]
fn test_would_repeat_ngram_n_zero() {
    // n=0 means disabled, should always return false.
    let tree: Vec<(usize, usize)> = vec![];
    assert!(!would_repeat_ngram(None, 42, 0, &tree, &[]));
}

#[test]
fn test_would_repeat_ngram_too_short() {
    // Sequence too short to form an n-gram of size 3.
    let tree = vec![(0, 10)]; // 1 decoded token
    let initial = vec![1]; // 1 initial token
    // Total: [1, 10, candidate] = 3 tokens, n=3 needs 3 tokens.
    // The candidate n-gram is the last 3: [1, 10, candidate], which can't repeat.
    assert!(!would_repeat_ngram(Some(0), 20, 3, &tree, &initial));
}

#[test]
fn test_would_repeat_bigram_detected() {
    // Sequence: [A, B, A] -- adding B would create repeated bigram [A, B].
    let initial: Vec<usize> = vec![];
    let tree = vec![
        (0, 10),     // idx=0: root, token=10 (A)
        (1, 20), // idx=1: parent=0, token=20 (B)
        (1 + 1, 10), // idx=2: parent=1, token=10 (A)
    ];
    // Current decoded tokens: [10, 20, 10]. Candidate: 20.
    // Full sequence: [10, 20, 10, 20]. Bigram [10, 20] appears at positions 0-1 and 2-3.
    assert!(would_repeat_ngram(Some(2), 20, 2, &tree, &initial));
}

#[test]
fn test_would_repeat_bigram_not_detected() {
    let initial: Vec<usize> = vec![];
    let tree = vec![
        (0, 10),     // A
        (1, 20), // B
        (1 + 1, 10), // A
    ];
    // Adding 30 (C): [10, 20, 10, 30]. Bigram [10, 30] doesn't repeat.
    assert!(!would_repeat_ngram(Some(2), 30, 2, &tree, &initial));
}

#[test]
fn test_would_repeat_trigram_detected() {
    let initial: Vec<usize> = vec![];
    let tree = vec![
        (0, 1),     // idx=0: token 1
        (1, 2), // idx=1: token 2
        (1 + 1, 3), // idx=2: token 3
        (2 + 1, 1), // idx=3: token 1
        (3 + 1, 2), // idx=4: token 2
    ];
    // Decoded: [1, 2, 3, 1, 2]. Candidate: 3.
    // Full: [1, 2, 3, 1, 2, 3]. Trigram [1, 2, 3] repeats.
    assert!(would_repeat_ngram(Some(4), 3, 3, &tree, &initial));
}

#[test]
fn test_would_repeat_with_initial_tokens() {
    // Initial tokens participate in n-gram matching.
    let initial = vec![10, 20];
    let tree = vec![(0, 30)]; // decoded: [30]
    // Full: [10, 20, 30, candidate]. With candidate=10:
    // Full: [10, 20, 30, 10]. No repeated bigrams (bigrams: [10,20], [20,30], [30,10]).
    assert!(!would_repeat_ngram(Some(0), 10, 2, &tree, &initial));

    // But if decoded is [30, 10, 20], candidate=30:
    // Full with initial: [10, 20, 30, 10, 20, 30]. Trigram [10, 20, 30] repeats.
    let tree2 = vec![
        (0, 30),     // idx=0
        (1, 10), // idx=1
        (1 + 1, 20), // idx=2
    ];
    assert!(would_repeat_ngram(Some(2), 30, 3, &tree2, &initial));
}

// ---------------------------------------------------------------------------
// suppress_blank_tokens
// ---------------------------------------------------------------------------

#[test]
fn test_suppress_blank_tokens_sets_neg_infinity() {
    let mut logits = vec![1.0; 51866];
    suppress_blank_tokens(&mut logits, EOT_TOKEN);
    assert_eq!(logits[BLANK_TOKEN], f32::NEG_INFINITY);
    assert_eq!(logits[EOT_TOKEN], f32::NEG_INFINITY);
    // Other positions should be unchanged.
    assert_eq!(logits[0], 1.0);
    assert_eq!(logits[100], 1.0);
}

#[test]
fn test_suppress_blank_tokens_small_vocab() {
    // Vocab smaller than BLANK_TOKEN (220): should not panic.
    let mut logits = vec![1.0; 100];
    suppress_blank_tokens(&mut logits, 50);
    // BLANK_TOKEN (220) is out of bounds, so it's not suppressed.
    // But eot_token (50) IS suppressed.
    assert_eq!(logits[50], f32::NEG_INFINITY);
    // All other values unchanged.
    assert_eq!(logits[0], 1.0);
}

#[test]
fn test_suppress_blank_tokens_eot_at_boundary() {
    let mut logits = vec![1.0; 300];
    suppress_blank_tokens(&mut logits, 299);
    assert_eq!(logits[BLANK_TOKEN], f32::NEG_INFINITY);
    assert_eq!(logits[299], f32::NEG_INFINITY);
}

// ---------------------------------------------------------------------------
// apply_ngram_blocking
// ---------------------------------------------------------------------------

#[test]
fn test_apply_ngram_blocking_suppresses_repeated_token() {
    let initial: Vec<usize> = vec![];
    let tree = vec![
        (0, 10),     // idx=0: token 10
        (1, 20), // idx=1: token 20
        (1 + 1, 10), // idx=2: token 10
    ];
    let mut logits = vec![1.0; 100];
    // Current: [10, 20, 10]. For n=2, token 20 would repeat bigram [10, 20].
    apply_ngram_blocking(&mut logits, Some(2), 2, &tree, &initial);
    assert_eq!(
        logits[20],
        f32::NEG_INFINITY,
        "token 20 should be blocked (repeats bigram [10, 20])"
    );
    // Token 10 with bigram [10, 10]: check if [10, 10] appeared before.
    // Sequence [10, 20, 10, 10] -- bigram [10, 10] not in original, so not blocked.
    // Actually the check: would_repeat_ngram(Some(2), 10, 2, &tree, &[]) for seq [10, 20, 10, 10]
    // Bigrams: [10,20], [20,10], [10,10]. [10,10] doesn't appear earlier. So 10 is NOT blocked.
    assert_ne!(logits[10], f32::NEG_INFINITY);
}

#[test]
fn test_apply_ngram_blocking_n_zero_does_nothing() {
    let tree: Vec<(usize, usize)> = vec![];
    let mut logits = vec![1.0; 10];
    apply_ngram_blocking(&mut logits, None, 0, &tree, &[]);
    // n=0 means disabled, nothing should change.
    for &v in &logits {
        assert_eq!(v, 1.0);
    }
}

// ---------------------------------------------------------------------------
// BeamHypothesis construction
// ---------------------------------------------------------------------------

#[test]
fn test_beam_hypothesis_new() {
    let hyp = BeamHypothesis::new(vec![1, 2, 3], -5.0, -1.67);
    assert_eq!(hyp.tokens, vec![1, 2, 3]);
    assert!((hyp.score - (-5.0)).abs() < 1e-6);
    assert!((hyp.normalized_score - (-1.67)).abs() < 1e-6);
}

#[test]
fn test_beam_hypothesis_empty_tokens() {
    let hyp = BeamHypothesis::new(vec![], 0.0, 0.0);
    assert!(hyp.tokens.is_empty());
}

#[test]
fn test_beam_hypothesis_clone() {
    let hyp = BeamHypothesis::new(vec![10, 20], -3.0, -1.5);
    let hyp2 = hyp.clone();
    assert_eq!(hyp2.tokens, hyp.tokens);
    assert!((hyp2.score - hyp.score).abs() < 1e-6);
    assert!((hyp2.normalized_score - hyp.normalized_score).abs() < 1e-6);
}

#[test]
fn test_beam_hypothesis_debug() {
    let hyp = BeamHypothesis::new(vec![1], -1.0, -1.0);
    let debug = format!("{hyp:?}");
    assert!(debug.contains("BeamHypothesis"));
}

// ---------------------------------------------------------------------------
// WhisperBeamOutput construction
// ---------------------------------------------------------------------------

#[test]
fn test_beam_output_single_hypothesis() {
    let hyp = BeamHypothesis::new(vec![1], -1.0, -1.0);
    let output = WhisperBeamOutput {
        best: hyp.clone(),
        hypotheses: vec![hyp],
    };
    assert_eq!(output.best.tokens, vec![1]);
    assert_eq!(output.hypotheses.len(), 1);
}

#[test]
fn test_beam_output_multiple_hypotheses_best_first() {
    let best = BeamHypothesis::new(vec![1, 2], -2.0, -1.0);
    let second = BeamHypothesis::new(vec![3, 4], -4.0, -2.0);
    let third = BeamHypothesis::new(vec![5, 6, 7], -6.0, -2.0);
    let output = WhisperBeamOutput {
        best: best.clone(),
        hypotheses: vec![best, second, third],
    };
    assert_eq!(output.hypotheses.len(), 3);
    assert_eq!(output.best.tokens, vec![1, 2]);
    // Best should have highest normalized score.
    assert!(output.hypotheses[0].normalized_score >= output.hypotheses[1].normalized_score);
}

#[test]
fn test_beam_output_clone() {
    let hyp = BeamHypothesis::new(vec![1], -1.0, -1.0);
    let output = WhisperBeamOutput {
        best: hyp.clone(),
        hypotheses: vec![hyp],
    };
    let output2 = output.clone();
    assert_eq!(output2.best.tokens, output.best.tokens);
    assert_eq!(output2.hypotheses.len(), output.hypotheses.len());
}

// ---------------------------------------------------------------------------
// Beam score comparison (for pruning)
// ---------------------------------------------------------------------------

#[test]
fn test_beam_score_comparison_finished_vs_active() {
    let finished = BeamState {
        node_idx: Some(0),
        decoded_len: 3,
        sum_log_prob: -3.0,
        finished: true,
    };
    let active = BeamState {
        node_idx: Some(1),
        decoded_len: 5,
        sum_log_prob: -5.0,
        finished: false,
    };
    // Both have same per-token score with penalty=1.0.
    let finished_score = finished.score(1.0);
    let active_score = active.score(1.0);
    assert!(
        (finished_score - active_score).abs() < 1e-10,
        "same per-token score: finished={finished_score}, active={active_score}"
    );
}

#[test]
fn test_beam_score_comparison_different_lengths() {
    // Beam with more tokens but proportionally better total score.
    let short = BeamState {
        node_idx: Some(0),
        decoded_len: 2,
        sum_log_prob: -4.0,
        finished: false,
    };
    let long = BeamState {
        node_idx: Some(1),
        decoded_len: 8,
        sum_log_prob: -8.0,
        finished: false,
    };
    // penalty=1.0: short=-4/2=-2, long=-8/8=-1. Long is better.
    assert!(long.score(1.0) > short.score(1.0));
    // penalty=0.0: short=-4, long=-8. Short is better.
    assert!(short.score(0.0) > long.score(0.0));
}

// ---------------------------------------------------------------------------
// End-of-text detection logic
// ---------------------------------------------------------------------------

#[test]
fn test_eot_detection_at_default_id() {
    let config = WhisperBeamSearchConfig::default();
    assert_eq!(config.eot_token, EOT_TOKEN);
    assert_eq!(config.eot_token, 50257);
}

#[test]
fn test_eot_detection_custom_id() {
    let config = WhisperBeamSearchConfig {
        eot_token: 99,
        ..Default::default()
    };
    config.validate().expect("custom eot_token should be valid");
    assert_eq!(config.eot_token, 99);
}

// ---------------------------------------------------------------------------
// Single beam (greedy) edge case
// ---------------------------------------------------------------------------

#[test]
fn test_single_beam_config_valid() {
    let config = WhisperBeamSearchConfig {
        beam_width: 1,
        ..Default::default()
    };
    config
        .validate()
        .expect("beam_width=1 should be valid for greedy search");
}

#[test]
fn test_top_k_single_beam_returns_one() {
    let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let result = top_k_log_probs(&logits, 1, 0.0);
    assert_eq!(result.len(), 1);
    // Should pick the highest logit (index 1, value 5.0).
    assert_eq!(result[0].0, 1);
}

// ---------------------------------------------------------------------------
// Beam expansion and pruning logic (unit tests on helpers)
// ---------------------------------------------------------------------------

#[test]
fn test_beam_expansion_preserves_parent_pointers() {
    // Simulate beam expansion: two beams, each expanded to 2 candidates.
    let tree: Vec<(usize, usize)> = vec![
        // Beam 0: root token 10
        (0, 10), // idx=0
        // Beam 1: root token 20
        (0, 20), // idx=1
        // Expand beam 0 with tokens 30, 40
        (1, 30), // idx=2: parent=0
        (1, 40), // idx=3: parent=0
        // Expand beam 1 with tokens 50, 60
        (1 + 1, 50), // idx=4: parent=1
        (1 + 1, 60), // idx=5: parent=1
    ];

    // Reconstruct paths.
    assert_eq!(reconstruct_decoded(Some(2), &tree), vec![10, 30]);
    assert_eq!(reconstruct_decoded(Some(3), &tree), vec![10, 40]);
    assert_eq!(reconstruct_decoded(Some(4), &tree), vec![20, 50]);
    assert_eq!(reconstruct_decoded(Some(5), &tree), vec![20, 60]);
}

#[test]
fn test_pruning_by_normalized_score() {
    // Simulate pruning: create 5 beams, sort by normalized score, keep top 3.
    let mut beams = vec![
        BeamState {
            node_idx: Some(0),
            decoded_len: 3,
            sum_log_prob: -3.0,
            finished: false,
        },
        BeamState {
            node_idx: Some(1),
            decoded_len: 3,
            sum_log_prob: -9.0,
            finished: false,
        },
        BeamState {
            node_idx: Some(2),
            decoded_len: 3,
            sum_log_prob: -1.0,
            finished: false,
        },
        BeamState {
            node_idx: Some(3),
            decoded_len: 3,
            sum_log_prob: -6.0,
            finished: false,
        },
        BeamState {
            node_idx: Some(4),
            decoded_len: 3,
            sum_log_prob: -4.0,
            finished: false,
        },
    ];

    let penalty = 1.0_f32;
    beams.sort_by(|a, b| b.score(penalty).total_cmp(&a.score(penalty)));
    beams.truncate(3);

    assert_eq!(beams.len(), 3);
    // Best should be the one with highest score: -1.0/3 > -3.0/3 > -4.0/3
    assert!((beams[0].sum_log_prob - (-1.0)).abs() < 1e-10);
    assert!((beams[1].sum_log_prob - (-3.0)).abs() < 1e-10);
    assert!((beams[2].sum_log_prob - (-4.0)).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Config with suppress_tokens
// ---------------------------------------------------------------------------

#[test]
fn test_config_with_suppress_tokens() {
    let config = WhisperBeamSearchConfig {
        suppress_tokens: vec![0, 1, 50257],
        ..Default::default()
    };
    config
        .validate()
        .expect("config with suppress_tokens should be valid");
    assert_eq!(config.suppress_tokens.len(), 3);
}

#[test]
fn test_config_no_repeat_ngram_size() {
    let config = WhisperBeamSearchConfig {
        no_repeat_ngram_size: 3,
        ..Default::default()
    };
    config
        .validate()
        .expect("config with ngram blocking should be valid");
    assert_eq!(config.no_repeat_ngram_size, 3);
}

// ---------------------------------------------------------------------------
// Length normalization edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_normalize_score_very_long_sequence() {
    // Ensure no overflow with large lengths.
    let score = normalize_score(-1000.0, 10000, 1.0);
    let expected = -1000.0 / 10000.0;
    assert!(
        (score - expected).abs() < 1e-3,
        "expected {expected}, got {score}"
    );
}

#[test]
fn test_normalize_score_penalty_two() {
    let score = normalize_score(-10.0, 4, 2.0);
    let expected = -10.0 / (4.0_f32).powf(2.0); // -10/16 = -0.625
    assert!(
        (score - expected).abs() < 1e-5,
        "expected {expected}, got {score}"
    );
}
