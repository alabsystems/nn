// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::Device;

use super::{ctc_beam_decode, ctc_greedy_decode, CtcConfig};

fn make_logits(data: Vec<f32>, t: usize, vocab: usize) -> DynTensor {
    DynTensor::from_vec(data, &[t, vocab], &Device::Cpu).unwrap()
}

// -- CTC greedy decode tests --------------------------------------------------

#[test]
fn test_ctc_greedy_simple() {
    // 3 timesteps, vocab=4 (blank=0, a=1, b=2, c=3).
    // Logits: step0→a, step1→a, step2→b.
    // Raw: [1, 1, 2], collapsed: [1, 2], blanks removed: [1, 2].
    #[rustfmt::skip]
    let logits = make_logits(vec![
        -10.0, 10.0, -10.0, -10.0,  // step 0: a wins
        -10.0, 10.0, -10.0, -10.0,  // step 1: a wins (repeat)
        -10.0, -10.0, 10.0, -10.0,  // step 2: b wins
    ], 3, 4);
    let config = CtcConfig::default();
    let result = ctc_greedy_decode(&logits, &config).unwrap();
    assert_eq!(result, vec![1, 2]);
}

#[test]
fn test_ctc_greedy_with_blanks() {
    // step0→a, step1→blank, step2→a → collapsed [a, blank, a] → [a, a].
    #[rustfmt::skip]
    let logits = make_logits(vec![
        -10.0, 10.0, -10.0,  // step 0: a
        10.0, -10.0, -10.0,  // step 1: blank
        -10.0, 10.0, -10.0,  // step 2: a
    ], 3, 3);
    let config = CtcConfig::default();
    let result = ctc_greedy_decode(&logits, &config).unwrap();
    assert_eq!(result, vec![1, 1]); // Two separate 'a's separated by blank.
}

#[test]
fn test_ctc_greedy_all_blanks() {
    #[rustfmt::skip]
    let logits = make_logits(vec![
        10.0, -10.0,
        10.0, -10.0,
        10.0, -10.0,
    ], 3, 2);
    let config = CtcConfig::default();
    let result = ctc_greedy_decode(&logits, &config).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_ctc_greedy_empty_input() {
    let logits = DynTensor::from_vec(Vec::<f32>::new(), &[0, 4], &Device::Cpu).unwrap();
    let config = CtcConfig::default();
    let result = ctc_greedy_decode(&logits, &config).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_ctc_greedy_wrong_rank() {
    let logits = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let config = CtcConfig::default();
    assert!(ctc_greedy_decode(&logits, &config).is_err());
}

#[test]
fn test_ctc_greedy_custom_blank() {
    // blank_id=2, step0→a(0), step1→blank(2), step2→b(1).
    #[rustfmt::skip]
    let logits = make_logits(vec![
        10.0, -10.0, -10.0,  // step 0: token 0
        -10.0, -10.0, 10.0,  // step 1: token 2 (blank)
        -10.0, 10.0, -10.0,  // step 2: token 1
    ], 3, 3);
    let config = CtcConfig { blank_id: 2 };
    let result = ctc_greedy_decode(&logits, &config).unwrap();
    assert_eq!(result, vec![0, 1]);
}

// -- CTC beam decode tests ----------------------------------------------------

#[test]
fn test_ctc_beam_simple() {
    // Same as greedy test: beam should agree with greedy for clear logits.
    #[rustfmt::skip]
    let logits = make_logits(vec![
        -10.0, 10.0, -10.0, -10.0,
        -10.0, 10.0, -10.0, -10.0,
        -10.0, -10.0, 10.0, -10.0,
    ], 3, 4);
    let config = CtcConfig::default();
    let results = ctc_beam_decode(&logits, &config, 3).unwrap();
    assert!(!results.is_empty());
    // Best beam should match greedy.
    assert_eq!(results[0].tokens, vec![1, 2]);
}

#[test]
fn test_ctc_beam_width_1_matches_greedy() {
    #[rustfmt::skip]
    let logits = make_logits(vec![
        -10.0, 10.0, -10.0,
        10.0, -10.0, -10.0,
        -10.0, -10.0, 10.0,
    ], 3, 3);
    let config = CtcConfig::default();
    let beam_result = ctc_beam_decode(&logits, &config, 1).unwrap();
    let greedy_result = ctc_greedy_decode(&logits, &config).unwrap();
    assert_eq!(beam_result[0].tokens, greedy_result);
}

#[test]
fn test_ctc_beam_multiple_hypotheses() {
    // Ambiguous logits: multiple valid decodings.
    #[rustfmt::skip]
    let logits = make_logits(vec![
        0.0, 1.0, 0.9,  // step 0: token 1 slightly favored over 2
        0.0, 0.9, 1.0,  // step 1: token 2 slightly favored over 1
    ], 2, 3);
    let config = CtcConfig::default();
    let results = ctc_beam_decode(&logits, &config, 5).unwrap();
    assert!(results.len() > 1, "should have multiple hypotheses");
    // Results should be sorted by log_prob descending.
    for i in 1..results.len() {
        assert!(
            results[i - 1].log_prob >= results[i].log_prob,
            "not sorted: {} < {}",
            results[i - 1].log_prob,
            results[i].log_prob
        );
    }
}

#[test]
fn test_ctc_beam_empty_input() {
    let logits = DynTensor::from_vec(Vec::<f32>::new(), &[0, 4], &Device::Cpu).unwrap();
    let config = CtcConfig::default();
    let results = ctc_beam_decode(&logits, &config, 3).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].tokens.is_empty());
}

#[test]
fn test_ctc_beam_zero_width() {
    let logits = make_logits(vec![1.0, 2.0], 1, 2);
    let config = CtcConfig::default();
    assert!(ctc_beam_decode(&logits, &config, 0).is_err());
}

#[test]
fn test_ctc_beam_wrong_rank() {
    let logits = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let config = CtcConfig::default();
    assert!(ctc_beam_decode(&logits, &config, 3).is_err());
}

#[test]
fn test_ctc_beam_scores_negative() {
    // Log probabilities should be <= 0.
    #[rustfmt::skip]
    let logits = make_logits(vec![
        -10.0, 10.0, -10.0,
        -10.0, -10.0, 10.0,
    ], 2, 3);
    let config = CtcConfig::default();
    let results = ctc_beam_decode(&logits, &config, 3).unwrap();
    for hyp in &results {
        assert!(
            hyp.log_prob <= 0.0,
            "log_prob should be <= 0, got {}",
            hyp.log_prob
        );
    }
}

// -- NaN/Inf edge case tests --------------------------------------------------

#[test]
fn test_ctc_greedy_all_neg_inf_logits_no_nan() {
    // All logits are -inf: the inline log_softmax guard should produce -inf,
    // not NaN. Argmax picks index 0 (deterministic via total_cmp).
    let logits = make_logits(vec![f32::NEG_INFINITY; 6], 2, 3);
    let config = CtcConfig::default();
    let result = ctc_greedy_decode(&logits, &config).unwrap();
    // blank_id=0, so argmax=0 at every step → collapsed to [0] → blank removed → empty.
    assert!(result.is_empty() || result.iter().all(|&t| t != config.blank_id));
}

#[test]
fn test_ctc_greedy_nan_logits_deterministic() {
    // NaN logits should produce deterministic output via total_cmp.
    // total_cmp treats NaN as greater than all finite values, so NaN wins argmax.
    // Place NaN at non-blank indices so deterministic behavior is observable.
    let logits = make_logits(vec![-1.0, f32::NAN, -1.0, -1.0, -1.0, f32::NAN], 2, 3);
    let config = CtcConfig::default();
    let result = ctc_greedy_decode(&logits, &config).unwrap();
    // Step 0: NaN at index 1 wins argmax (NaN > all finite via total_cmp).
    // Step 1: NaN at index 2 wins argmax.
    // Collapsed: [1, 2], blank=0 removed → [1, 2].
    assert_eq!(result, vec![1, 2]);
}

#[test]
fn test_ctc_beam_all_neg_inf_logits_no_nan() {
    // All-neg-inf logits: the log_softmax guard fills with -inf.
    // Beam decode should not produce NaN scores.
    let logits = make_logits(vec![f32::NEG_INFINITY; 6], 2, 3);
    let config = CtcConfig::default();
    let results = ctc_beam_decode(&logits, &config, 2).unwrap();
    for hyp in &results {
        assert!(
            !hyp.log_prob.is_nan(),
            "beam log_prob should not be NaN, got {}",
            hyp.log_prob
        );
    }
}

#[test]
fn test_ctc_beam_single_neg_inf_step() {
    // One step all-neg-inf, one step normal: should not panic or produce NaN.
    #[rustfmt::skip]
    let logits = make_logits(vec![
        f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY,  // step 0: all -inf
        -10.0, 10.0, -10.0,                                       // step 1: token 1 wins
    ], 2, 3);
    let config = CtcConfig::default();
    let results = ctc_beam_decode(&logits, &config, 2).unwrap();
    for hyp in &results {
        assert!(
            !hyp.log_prob.is_nan(),
            "beam log_prob should not be NaN, got {}",
            hyp.log_prob
        );
    }
}
