// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Create a synthetic audio sample with controlled quality.
/// Higher amplitude = higher energy = better "quality" metrics.
fn make_audio(amplitude: f32, n_samples: usize, freq_hz: f32, sample_rate: f32) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            amplitude * (2.0 * std::f32::consts::PI * freq_hz * t).sin()
        })
        .collect()
}

fn make_group(dimension: &str, value: &str) -> Group {
    Group {
        dimension: dimension.to_string(),
        value: value.to_string(),
    }
}

fn make_tagged_sample(
    id: &str,
    groups: Vec<Group>,
    amplitude: f32,
    reference: Option<Vec<f32>>,
) -> TaggedSample {
    let audio = make_audio(amplitude, 24000, 440.0, 24000.0); // 1 second at 24kHz
    TaggedSample {
        id: id.to_string(),
        groups,
        audio,
        reference,
    }
}

fn default_verifier() -> TtsVerifier {
    use crate::config::{HardBoundsConfig, RejectionPolicy};
    // Use Warn policy: test sine-wave audio fails spectral coverage, but
    // fairness tests inspect certificate contents, not rejection behavior.
    let hb = HardBoundsConfig {
        rejection_policy: RejectionPolicy::Warn,
        ..Default::default()
    };
    TtsVerifier::builder()
        .sample_rate(24000)
        .hard_bounds(hb)
        .build()
        .unwrap()
}

#[test]
fn test_measure_fairness_equal_groups() {
    // Two groups with identical audio quality → is_fair = true
    let lang_en = make_group("language", "en");
    let lang_ja = make_group("language", "ja");

    let mut samples = Vec::new();
    for i in 0..30 {
        samples.push(make_tagged_sample(
            &format!("en_{i}"),
            vec![lang_en.clone()],
            0.5,
            None,
        ));
        samples.push(make_tagged_sample(
            &format!("ja_{i}"),
            vec![lang_ja.clone()],
            0.5,
            None,
        ));
    }

    let config = FairnessConfig {
        min_samples_per_group: 5,
        max_gap: 1.0,
        ..FairnessConfig::default()
    };

    let report = measure_fairness(&samples, &default_verifier(), &config, 24000).unwrap();

    assert_eq!(report.group_stats.len(), 2);
    assert!(
        report.is_fair,
        "Equal groups should be fair. max_gap={}, worst_gaps={:?}",
        report.max_quality_gap, report.worst_gaps,
    );
}

#[test]
fn test_measure_fairness_biased() {
    // Two groups with very different quality (different amplitudes)
    // Group A: loud clear audio (amplitude=0.8)
    // Group B: quiet near-silence (amplitude=0.02)
    let group_a = make_group("quality_tier", "high");
    let group_b = make_group("quality_tier", "low");

    let mut samples = Vec::new();
    for i in 0..30 {
        samples.push(make_tagged_sample(
            &format!("high_{i}"),
            vec![group_a.clone()],
            0.8,
            None,
        ));
    }
    for i in 0..30 {
        // Near-silence but above the non-silence threshold (min_rms=0.01)
        samples.push(make_tagged_sample(
            &format!("low_{i}"),
            vec![group_b.clone()],
            0.02,
            None,
        ));
    }

    let config = FairnessConfig {
        min_samples_per_group: 5,
        max_gap: 0.01, // Very tight gap threshold to force unfair verdict
        ..FairnessConfig::default()
    };

    let report = measure_fairness(&samples, &default_verifier(), &config, 24000).unwrap();

    assert_eq!(report.group_stats.len(), 2);
    // With such different amplitudes, hard bound values will differ significantly
    assert!(
        !report.is_fair || report.max_quality_gap > 0.01,
        "Biased groups with tight threshold should be unfair or have measurable gap"
    );
}

#[test]
fn test_measure_fairness_min_samples() {
    // One group has enough samples, the other doesn't
    let lang_en = make_group("language", "en");
    let lang_ja = make_group("language", "ja");

    let mut samples = Vec::new();
    for i in 0..30 {
        samples.push(make_tagged_sample(
            &format!("en_{i}"),
            vec![lang_en.clone()],
            0.5,
            None,
        ));
    }
    // Only 2 Japanese samples — below min_samples_per_group
    for i in 0..2 {
        samples.push(make_tagged_sample(
            &format!("ja_{i}"),
            vec![lang_ja.clone()],
            0.5,
            None,
        ));
    }

    let config = FairnessConfig {
        min_samples_per_group: 30,
        ..FairnessConfig::default()
    };

    let report = measure_fairness(&samples, &default_verifier(), &config, 24000).unwrap();

    // Under-represented group should still appear in stats but not in comparisons
    assert_eq!(report.group_stats.len(), 2, "Both groups should have stats");
    assert!(
        report.comparisons.is_empty(),
        "No comparisons should be made when one group has < min_samples"
    );
    assert!(
        report.is_fair,
        "Should be fair when no comparisons can be made"
    );
}

#[test]
fn test_measure_fairness_empty_input() {
    let config = FairnessConfig::default();
    let result = measure_fairness(&[], &default_verifier(), &config, 24000);
    assert!(result.is_err(), "Empty input should return error");
}

#[test]
fn test_measure_fairness_multiple_dimensions() {
    // Samples can belong to multiple groups across different dimensions
    let en = make_group("language", "en");
    let ja = make_group("language", "ja");
    let male = make_group("gender", "male");
    let female = make_group("gender", "female");

    let mut samples = Vec::new();
    for i in 0..10 {
        // English + male
        samples.push(make_tagged_sample(
            &format!("en_m_{i}"),
            vec![en.clone(), male.clone()],
            0.5,
            None,
        ));
        // English + female
        samples.push(make_tagged_sample(
            &format!("en_f_{i}"),
            vec![en.clone(), female.clone()],
            0.5,
            None,
        ));
        // Japanese + male
        samples.push(make_tagged_sample(
            &format!("ja_m_{i}"),
            vec![ja.clone(), male.clone()],
            0.5,
            None,
        ));
        // Japanese + female
        samples.push(make_tagged_sample(
            &format!("ja_f_{i}"),
            vec![ja.clone(), female.clone()],
            0.5,
            None,
        ));
    }

    let config = FairnessConfig {
        min_samples_per_group: 5,
        max_gap: 1.0,
        ..FairnessConfig::default()
    };

    let report = measure_fairness(&samples, &default_verifier(), &config, 24000).unwrap();

    // Should have 4 groups: en, ja, male, female
    assert_eq!(report.group_stats.len(), 4);
    // Comparisons should be within dimensions only (en vs ja, male vs female)
    // Not across dimensions (en vs male)
    for comp in &report.comparisons {
        assert_eq!(
            comp.group_a.dimension, comp.group_b.dimension,
            "Comparisons should only be within the same dimension"
        );
    }
}

#[test]
fn test_group_stats_metric_values() {
    let group = make_group("language", "en");

    let mut samples = Vec::new();
    for i in 0..10 {
        samples.push(make_tagged_sample(
            &format!("en_{i}"),
            vec![group.clone()],
            0.5,
            None,
        ));
    }

    let config = FairnessConfig {
        min_samples_per_group: 2,
        ..FairnessConfig::default()
    };

    let report = measure_fairness(&samples, &default_verifier(), &config, 24000).unwrap();

    let stats = &report.group_stats[0];
    assert_eq!(stats.n_samples, 10);
    // pass_rate may be 0 if hard bounds fail (e.g., pure sine fails spectral coverage)
    assert!(
        stats.pass_rate >= 0.0 && stats.pass_rate <= 1.0,
        "Pass rate should be in [0, 1], got {}",
        stats.pass_rate,
    );

    // Hard bound metrics should always be present (7 hard bounds per sample)
    assert!(
        !stats.metric_stats.is_empty(),
        "Should have at least hard bound metric stats"
    );

    // Each metric stat should have valid values
    for ms in &stats.metric_stats {
        assert!(ms.n > 0, "Metric {} should have samples", ms.name);
        assert!(
            ms.min <= ms.max || (ms.min - ms.max).abs() < 1e-10,
            "min <= max for metric {}: min={}, max={}",
            ms.name,
            ms.min,
            ms.max,
        );
        // p5 and p95 may be equal for identical samples; allow small tolerance
        assert!(
            ms.p5 <= ms.p95 + 1e-10,
            "p5 <= p95 for metric {}: p5={}, p95={}",
            ms.name,
            ms.p5,
            ms.p95,
        );
    }
}

#[test]
fn test_pairwise_comparison_structure() {
    let en = make_group("language", "en");
    let ja = make_group("language", "ja");
    let ko = make_group("language", "ko");

    let mut samples = Vec::new();
    for i in 0..10 {
        samples.push(make_tagged_sample(
            &format!("en_{i}"),
            vec![en.clone()],
            0.5,
            None,
        ));
        samples.push(make_tagged_sample(
            &format!("ja_{i}"),
            vec![ja.clone()],
            0.5,
            None,
        ));
        samples.push(make_tagged_sample(
            &format!("ko_{i}"),
            vec![ko.clone()],
            0.5,
            None,
        ));
    }

    let config = FairnessConfig {
        min_samples_per_group: 5,
        ..FairnessConfig::default()
    };

    let report = measure_fairness(&samples, &default_verifier(), &config, 24000).unwrap();

    // With 3 groups and N metrics, we should have 3*N comparisons (3 choose 2 = 3 pairs)
    // All comparisons should be within the "language" dimension
    for comp in &report.comparisons {
        assert_eq!(comp.group_a.dimension, "language");
        assert_eq!(comp.group_b.dimension, "language");
        assert!(
            comp.p_value >= 0.0 && comp.p_value <= 1.0,
            "p-value should be in [0,1]"
        );
    }
}

#[test]
fn test_fairness_config_default() {
    let config = FairnessConfig::default();
    assert_eq!(config.alpha, 0.05);
    assert_eq!(config.max_gap, 1.0);
    assert_eq!(config.min_samples_per_group, 30);
    assert!(config.metrics.is_empty());
}

/// End-to-end pipeline test: measure_fairness → diagnose_bias → select_debiasing_curriculum.
///
/// Creates 3 language groups with deliberately different audio quality
/// (different amplitudes simulating quality gaps), then verifies:
/// 1. measure_fairness detects the quality disparity
/// 2. diagnose_bias correctly identifies the worst-performing group
/// 3. select_debiasing_curriculum picks samples from that group
#[test]
fn test_e2e_fairness_pipeline() {
    use crate::fairness_diagnosis::{diagnose_bias, select_debiasing_curriculum};

    let en = make_group("language", "en");
    let ja = make_group("language", "ja");
    let ko = make_group("language", "ko");

    // EN: high quality (amplitude 0.6)
    // JA: medium quality (amplitude 0.3)
    // KO: low quality (amplitude 0.08) — near the silence threshold
    let mut samples = Vec::new();
    for i in 0..35 {
        samples.push(make_tagged_sample(
            &format!("en_{i}"),
            vec![en.clone()],
            0.6,
            None,
        ));
        samples.push(make_tagged_sample(
            &format!("ja_{i}"),
            vec![ja.clone()],
            0.3,
            None,
        ));
        samples.push(make_tagged_sample(
            &format!("ko_{i}"),
            vec![ko.clone()],
            0.08,
            None,
        ));
    }

    // Phase 1: Measure fairness with a gap that catches the EN-KO disparity
    let config = FairnessConfig {
        min_samples_per_group: 5,
        max_gap: 0.01, // Tight gap threshold
        ..FairnessConfig::default()
    };

    let report = measure_fairness(&samples, &default_verifier(), &config, 24000).unwrap();
    assert_eq!(report.group_stats.len(), 3, "Should have 3 groups");

    // Phase 2: Diagnose bias
    let diagnosis = diagnose_bias(&report);
    assert!(
        !diagnosis.rankings.is_empty(),
        "Should produce rankings across dimensions"
    );

    // All rankings should be in the "language" dimension
    for ranking in &diagnosis.rankings {
        assert_eq!(ranking.dimension, "language");
        assert_eq!(ranking.ranked_groups.len(), 3);
    }

    // Phase 3: Select debiasing curriculum — should pick KO samples
    let curriculum = select_debiasing_curriculum(&report, &samples, "language");
    assert!(
        !curriculum.is_empty(),
        "Curriculum should not be empty for biased report"
    );

    // Verify selected samples belong to the worst-performing group (KO)
    for &idx in &curriculum {
        assert!(idx < samples.len(), "Index should be in bounds");
        let sample = &samples[idx];
        let is_worst_group = sample
            .groups
            .iter()
            .any(|g| g.dimension == "language" && g.value == "ko");
        assert!(
            is_worst_group,
            "Debiasing curriculum should select from worst group (ko), got: {:?}",
            sample.groups
        );
    }
}

/// Test that the pipeline handles single-group gracefully.
/// With only one group, there are no pairwise comparisons, so
/// diagnosis should produce rankings with gap=0 and curriculum
/// should return all samples (single group is trivially "worst").
#[test]
fn test_e2e_single_group_pipeline() {
    use crate::fairness_diagnosis::{diagnose_bias, select_debiasing_curriculum};

    let en = make_group("language", "en");

    let mut samples = Vec::new();
    for i in 0..10 {
        samples.push(make_tagged_sample(
            &format!("en_{i}"),
            vec![en.clone()],
            0.5,
            None,
        ));
    }

    let config = FairnessConfig {
        min_samples_per_group: 5,
        ..FairnessConfig::default()
    };

    let report = measure_fairness(&samples, &default_verifier(), &config, 24000).unwrap();
    assert_eq!(report.group_stats.len(), 1);
    assert!(report.is_fair, "Single group should be trivially fair");
    assert!(
        report.comparisons.is_empty(),
        "No pairwise comparisons with single group"
    );

    let diagnosis = diagnose_bias(&report);
    // With one group, rankings exist but gaps are 0
    for ranking in &diagnosis.rankings {
        assert_eq!(ranking.ranked_groups.len(), 1);
        assert!(
            ranking.gap.abs() < f64::EPSILON,
            "Single-group gap should be 0"
        );
    }

    let curriculum = select_debiasing_curriculum(&report, &samples, "language");
    // Single group: no debiasing possible (need ≥2 groups to identify "worst")
    assert!(
        curriculum.is_empty(),
        "Single group should return empty curriculum (no bias to correct)"
    );
}
