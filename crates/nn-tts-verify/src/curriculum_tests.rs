// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for verification-guided curriculum selection.

use super::*;
use crate::config::{HardBoundsConfig, RejectionPolicy};
use crate::test_audio_helpers::sine_wave;

/// Build a verifier with Warn policy so analyze_corpus can inspect
/// individual hard bound results rather than getting VerificationRejected
/// for simple sine-wave test audio that fails spectral_coverage.
fn warn_verifier(sample_rate: u32) -> TtsVerifier {
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    TtsVerifier::builder()
        .sample_rate(sample_rate)
        .hard_bounds(hb)
        .build()
        .unwrap()
}

// ---- Test 1: analyze_corpus basic -----------------------------------------

#[test]
fn test_analyze_corpus_basic() {
    let sr = 24000;
    let duration = 0.5; // 0.5 seconds

    // Create 5 utterances: 3 clean sines, 2 problematic.
    let clean = sine_wave(440.0, sr, duration);
    let mut clipped = sine_wave(440.0, sr, duration);
    // Make one utterance clip by scaling to max amplitude.
    for s in &mut clipped {
        *s *= 1.5;
        *s = s.clamp(-1.0, 1.0);
    }
    let silent = vec![0.0001_f32; (f64::from(sr) * duration) as usize];

    let corpus: Vec<(String, Vec<f32>)> = vec![
        ("clean_1".to_string(), clean.clone()),
        ("clean_2".to_string(), clean.clone()),
        ("clean_3".to_string(), clean),
        ("clipped".to_string(), clipped),
        ("near_silent".to_string(), silent),
    ];

    let verifier = warn_verifier(sr);
    let analysis = analyze_corpus(&corpus, None, &verifier).unwrap();

    // All 5 utterances should be analyzed.
    assert_eq!(analysis.utterances.len(), 5);

    // Mean quality should be between 0 and 1.
    assert!(
        analysis.mean_quality >= 0.0 && analysis.mean_quality <= 1.0,
        "mean_quality={} should be in [0, 1]",
        analysis.mean_quality
    );

    // Worst utterances list should be non-empty and sorted.
    assert!(
        !analysis.worst_utterances.is_empty(),
        "should have worst utterances"
    );
    for i in 1..analysis.worst_utterances.len() {
        assert!(
            analysis.worst_utterances[i - 1].quality_score
                <= analysis.worst_utterances[i].quality_score,
            "worst_utterances should be sorted ascending by quality_score"
        );
    }
}

// ---- Test 2: curriculum selects worst -------------------------------------

#[test]
fn test_curriculum_selects_worst() {
    let sr = 24000;
    let duration = 0.5;
    let clean = sine_wave(440.0, sr, duration);
    let silent = vec![0.0001_f32; (f64::from(sr) * duration) as usize];

    // 10 utterances: 8 clean, 2 silent (should fail non-silence check).
    let mut corpus: Vec<(String, Vec<f32>)> = (0..8)
        .map(|i| (format!("clean_{i}"), clean.clone()))
        .collect();
    corpus.push(("silent_1".to_string(), silent.clone()));
    corpus.push(("silent_2".to_string(), silent));

    let verifier = warn_verifier(sr);
    let analysis = analyze_corpus(&corpus, None, &verifier).unwrap();

    // Select bottom 20% = 2 utterances.
    let config = CurriculumConfig {
        bottom_fraction: 0.20,
        quality_threshold: 0.0, // disable threshold
        priority_metrics: Vec::new(),
    };
    let curriculum = select_curriculum(&analysis, &config);

    // Should select at least 2 (20% of 10).
    assert!(
        curriculum.len() >= 2,
        "should select at least 2 utterances, got {}",
        curriculum.len()
    );

    // The selected indices should include the two silent utterances (index 8, 9).
    let has_silent_1 = curriculum.contains(&8);
    let has_silent_2 = curriculum.contains(&9);
    assert!(
        has_silent_1 && has_silent_2,
        "curriculum should include silent utterances: has_8={has_silent_1}, has_9={has_silent_2}, selected={curriculum:?}"
    );
}

// ---- Test 3: curriculum respects threshold --------------------------------

#[test]
fn test_curriculum_respects_threshold() {
    let sr = 24000;
    let duration = 0.5;
    let clean = sine_wave(440.0, sr, duration);
    let silent = vec![0.0001_f32; (f64::from(sr) * duration) as usize];

    // 10 utterances: 6 clean, 4 silent.
    let mut corpus: Vec<(String, Vec<f32>)> = (0..6)
        .map(|i| (format!("clean_{i}"), clean.clone()))
        .collect();
    for i in 0..4 {
        corpus.push((format!("silent_{i}"), silent.clone()));
    }

    let verifier = warn_verifier(sr);
    let analysis = analyze_corpus(&corpus, None, &verifier).unwrap();

    // bottom_fraction = 10% = 1 utterance, but threshold should capture all 4 silent.
    let config = CurriculumConfig {
        bottom_fraction: 0.10,
        quality_threshold: 0.99, // very strict — any failure triggers inclusion
        priority_metrics: Vec::new(),
    };
    let curriculum = select_curriculum(&analysis, &config);

    // Should select at least 4 (the silent ones fail the non-silence check).
    assert!(
        curriculum.len() >= 4,
        "threshold should override fraction: expected >= 4, got {}",
        curriculum.len()
    );
}

// ---- Test 4: metric failure rates sorted ----------------------------------

#[test]
fn test_metric_failure_rates_sorted() {
    let sr = 24000;
    let duration = 0.5;
    let clean = sine_wave(440.0, sr, duration);
    let silent = vec![0.0001_f32; (f64::from(sr) * duration) as usize];

    let mut corpus: Vec<(String, Vec<f32>)> = vec![
        ("clean".to_string(), clean),
        ("silent_1".to_string(), silent.clone()),
        ("silent_2".to_string(), silent),
    ];

    // Add a short utterance that will fail duration check.
    let short = sine_wave(440.0, sr, 0.01); // 10ms — below min duration
    corpus.push(("short".to_string(), short));

    let verifier = warn_verifier(sr);
    let analysis = analyze_corpus(&corpus, None, &verifier).unwrap();

    // metric_failure_rates should be sorted by failure rate descending.
    for i in 1..analysis.metric_failure_rates.len() {
        let (ref name_prev, rate_prev) = analysis.metric_failure_rates[i - 1];
        let (ref name_curr, rate_curr) = analysis.metric_failure_rates[i];
        assert!(
            rate_prev >= rate_curr,
            "failure rates should be sorted descending: {name_prev}={rate_prev} should be >= {name_curr}={rate_curr}"
        );
    }
}

// ---- Test 5: empty corpus returns error -----------------------------------

#[test]
fn test_empty_corpus_returns_error() {
    let verifier = TtsVerifier::builder().build().unwrap();
    let result = analyze_corpus(&[], None, &verifier);
    assert!(result.is_err(), "empty corpus should return error");
}

// ---- Test 6: percentile computation ---------------------------------------

#[test]
fn test_percentile_computation() {
    // p5 of [0.0, 0.25, 0.5, 0.75, 1.0] should be near 0.0.
    let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let p5 = percentile(&data, 5.0);
    assert!(
        p5 < 0.1,
        "5th percentile of [0..1] should be near 0, got {p5}"
    );

    let p50 = percentile(&data, 50.0);
    assert!(
        (p50 - 0.5).abs() < 0.01,
        "50th percentile should be 0.5, got {p50}"
    );

    let p95 = percentile(&data, 95.0);
    assert!(p95 > 0.9, "95th percentile should be near 1.0, got {p95}");
}

// ---- Test 7: default config matches design doc ----------------------------

#[test]
fn test_default_config() {
    let config = CurriculumConfig::default();
    assert!(
        (config.bottom_fraction - 0.10).abs() < f64::EPSILON,
        "default bottom_fraction should be 0.10"
    );
    assert!(
        (config.quality_threshold - 0.5).abs() < f64::EPSILON,
        "default quality_threshold should be 0.5"
    );
    assert!(
        config.priority_metrics.is_empty(),
        "default priority_metrics should be empty"
    );
}
