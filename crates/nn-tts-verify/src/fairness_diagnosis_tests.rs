// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::fairness::{
    FairnessReport, Group, GroupStats, MetricStat, PairwiseComparison, TaggedSample,
};

fn make_group(dimension: &str, value: &str) -> Group {
    Group {
        dimension: dimension.to_string(),
        value: value.to_string(),
    }
}

fn make_metric_stat(name: &str, mean: f64) -> MetricStat {
    MetricStat {
        name: name.to_string(),
        mean,
        std_dev: 0.1,
        min: mean - 0.2,
        max: mean + 0.2,
        p5: mean - 0.15,
        p95: mean + 0.15,
        n: 30,
    }
}

fn make_group_stats(group: Group, pass_rate: f64, metric_means: &[(&str, f64)]) -> GroupStats {
    GroupStats {
        group,
        n_samples: 30,
        metric_stats: metric_means
            .iter()
            .map(|(name, mean)| make_metric_stat(name, *mean))
            .collect(),
        pass_rate,
    }
}

#[test]
fn test_diagnose_bias_ranking() {
    // 3 language groups with different quality levels
    let en = make_group("language", "en");
    let ja = make_group("language", "ja");
    let ko = make_group("language", "ko");

    let report = FairnessReport {
        group_stats: vec![
            make_group_stats(en.clone(), 0.95, &[("snr", 25.0), ("pesq", 3.8)]),
            make_group_stats(ja, 0.80, &[("snr", 20.0), ("pesq", 3.2)]),
            make_group_stats(ko.clone(), 0.70, &[("snr", 15.0), ("pesq", 2.5)]),
        ],
        comparisons: vec![PairwiseComparison {
            group_a: en.clone(),
            group_b: ko.clone(),
            metric: "snr".to_string(),
            mean_diff: 10.0,
            t_statistic: 5.0,
            p_value: 0.001,
            cohens_d: 2.0,
            significant: true,
        }],
        max_quality_gap: 10.0,
        worst_gaps: vec![(en, ko, "snr".to_string(), 10.0)],
        is_fair: false,
    };

    let diagnosis = diagnose_bias(&report);

    // Rankings should be sorted by gap descending
    assert!(!diagnosis.rankings.is_empty());

    // The SNR ranking (gap=10) should be first
    let snr_ranking = diagnosis
        .rankings
        .iter()
        .find(|r| r.metric == "snr")
        .expect("Should have SNR ranking");

    assert_eq!(snr_ranking.dimension, "language");
    assert_eq!(snr_ranking.ranked_groups.len(), 3);

    // Worst group (lowest SNR) should be Korean
    assert_eq!(snr_ranking.ranked_groups[0].0.value, "ko");
    assert!((snr_ranking.ranked_groups[0].1 - 15.0).abs() < 1e-10);

    // Best group (highest SNR) should be English
    assert_eq!(snr_ranking.ranked_groups[2].0.value, "en");
    assert!((snr_ranking.ranked_groups[2].1 - 25.0).abs() < 1e-10);

    // Gap should be 10.0 (25.0 - 15.0)
    assert!((snr_ranking.gap - 10.0).abs() < 1e-10);

    // Should have recommendations
    assert!(
        !diagnosis.recommendations.is_empty(),
        "Should produce recommendations"
    );

    // First recommendation should mention the worst group
    let first_rec = &diagnosis.recommendations[0];
    assert!(
        first_rec.contains("ko") || first_rec.contains("Korean"),
        "Recommendation should mention worst group, got: {first_rec}",
    );
}

#[test]
fn test_diagnose_bias_no_bias() {
    // All groups have equal quality
    let en = make_group("language", "en");
    let ja = make_group("language", "ja");

    let report = FairnessReport {
        group_stats: vec![
            make_group_stats(en, 0.90, &[("snr", 20.0)]),
            make_group_stats(ja, 0.90, &[("snr", 20.0)]),
        ],
        comparisons: vec![],
        max_quality_gap: 0.0,
        worst_gaps: vec![],
        is_fair: true,
    };

    let diagnosis = diagnose_bias(&report);

    // Rankings should exist but gaps should be zero
    for ranking in &diagnosis.rankings {
        assert!(
            ranking.gap < 1e-10,
            "Gap should be ~0 for equal groups, got {}",
            ranking.gap,
        );
    }

    // Should have "no significant differences" recommendation
    assert!(
        diagnosis
            .recommendations
            .iter()
            .any(|r| r.contains("No statistically significant")),
        "Should note no significant differences"
    );
}

#[test]
fn test_diagnose_bias_multiple_dimensions() {
    // Both language and gender dimensions
    let en = make_group("language", "en");
    let ja = make_group("language", "ja");
    let male = make_group("gender", "male");
    let female = make_group("gender", "female");

    let report = FairnessReport {
        group_stats: vec![
            make_group_stats(en, 0.90, &[("snr", 22.0)]),
            make_group_stats(ja, 0.85, &[("snr", 18.0)]),
            make_group_stats(male, 0.88, &[("snr", 21.0)]),
            make_group_stats(female, 0.87, &[("snr", 20.0)]),
        ],
        comparisons: vec![],
        max_quality_gap: 4.0,
        worst_gaps: vec![],
        is_fair: true,
    };

    let diagnosis = diagnose_bias(&report);

    // Should have rankings for both dimensions
    let lang_rankings: Vec<_> = diagnosis
        .rankings
        .iter()
        .filter(|r| r.dimension == "language")
        .collect();
    let gender_rankings: Vec<_> = diagnosis
        .rankings
        .iter()
        .filter(|r| r.dimension == "gender")
        .collect();

    assert!(!lang_rankings.is_empty(), "Should have language rankings");
    assert!(!gender_rankings.is_empty(), "Should have gender rankings");

    // Language gap (4.0) should be larger than gender gap (1.0)
    let lang_gap = lang_rankings[0].gap;
    let gender_gap = gender_rankings[0].gap;
    assert!(
        lang_gap > gender_gap,
        "Language gap ({lang_gap}) should be larger than gender gap ({gender_gap})",
    );
}

fn make_tagged_samples(groups_and_values: &[(Group, f32, usize)]) -> Vec<TaggedSample> {
    let mut samples = Vec::new();
    for (group, amplitude, count) in groups_and_values {
        for i in 0..*count {
            samples.push(TaggedSample {
                id: format!("{}_{}", group.value, i),
                groups: vec![group.clone()],
                audio: vec![*amplitude; 100],
                reference: None,
            });
        }
    }
    samples
}

#[test]
fn test_debiasing_curriculum_selects_worst_group() {
    let en = make_group("language", "en");
    let ja = make_group("language", "ja");
    let ko = make_group("language", "ko");

    // Korean is worst (lowest pass_rate)
    let report = FairnessReport {
        group_stats: vec![
            make_group_stats(en.clone(), 0.95, &[("snr", 25.0)]),
            make_group_stats(ja.clone(), 0.80, &[("snr", 20.0)]),
            make_group_stats(ko.clone(), 0.60, &[("snr", 15.0)]),
        ],
        comparisons: vec![],
        max_quality_gap: 10.0,
        worst_gaps: vec![],
        is_fair: false,
    };

    // Create samples: indices 0-2 English, 3-5 Japanese, 6-8 Korean
    let samples = make_tagged_samples(&[(en, 0.5, 3), (ja, 0.4, 3), (ko, 0.2, 3)]);

    let curriculum = select_debiasing_curriculum(&report, &samples, "language");

    // Should select Korean samples (indices 6, 7, 8)
    assert_eq!(curriculum.len(), 3, "Should select all Korean samples");
    assert_eq!(curriculum, vec![6, 7, 8]);
}

#[test]
fn test_debiasing_curriculum_unknown_dimension() {
    let en = make_group("language", "en");

    let report = FairnessReport {
        group_stats: vec![make_group_stats(en.clone(), 0.90, &[("snr", 20.0)])],
        comparisons: vec![],
        max_quality_gap: 0.0,
        worst_gaps: vec![],
        is_fair: true,
    };

    let samples = vec![TaggedSample {
        id: "en_0".into(),
        groups: vec![en],
        audio: vec![0.5; 100],
        reference: None,
    }];

    // Nonexistent dimension → empty result
    let curriculum = select_debiasing_curriculum(&report, &samples, "accent");
    assert!(
        curriculum.is_empty(),
        "Unknown dimension should return empty curriculum"
    );
}

#[test]
fn test_debiasing_curriculum_single_group() {
    let en = make_group("language", "en");

    let report = FairnessReport {
        group_stats: vec![make_group_stats(en.clone(), 0.90, &[("snr", 20.0)])],
        comparisons: vec![],
        max_quality_gap: 0.0,
        worst_gaps: vec![],
        is_fair: true,
    };

    let samples = vec![TaggedSample {
        id: "en_0".into(),
        groups: vec![en],
        audio: vec![0.5; 100],
        reference: None,
    }];

    // Single group → no "worst" to identify → empty
    let curriculum = select_debiasing_curriculum(&report, &samples, "language");
    assert!(
        curriculum.is_empty(),
        "Single group in dimension should return empty curriculum"
    );
}

#[test]
fn test_debiasing_curriculum_tiebreak_by_metric() {
    // Two groups with same pass_rate — tiebreak by average metric mean
    let en = make_group("language", "en");
    let ja = make_group("language", "ja");

    let report = FairnessReport {
        group_stats: vec![
            make_group_stats(en.clone(), 0.80, &[("snr", 22.0)]),
            make_group_stats(ja.clone(), 0.80, &[("snr", 18.0)]), // Same pass_rate, lower SNR
        ],
        comparisons: vec![],
        max_quality_gap: 4.0,
        worst_gaps: vec![],
        is_fair: false,
    };

    let samples: Vec<TaggedSample> = vec![
        TaggedSample {
            id: "en_0".into(),
            groups: vec![en],
            audio: vec![0.5; 100],
            reference: None,
        },
        TaggedSample {
            id: "ja_0".into(),
            groups: vec![ja],
            audio: vec![0.4; 100],
            reference: None,
        },
    ];

    let curriculum = select_debiasing_curriculum(&report, &samples, "language");

    // Japanese should be selected (lower average metric mean as tiebreaker)
    assert_eq!(curriculum.len(), 1);
    assert_eq!(curriculum[0], 1, "Should select Japanese sample (index 1)");
}

#[test]
fn test_diagnose_bias_nan_mean_excluded_from_ranking() {
    // NaN metric means should not corrupt ranking order.
    // A group with NaN mean should effectively be skipped or
    // treated consistently (not silently misranked).
    let en = make_group("language", "en");
    let ja = make_group("language", "ja");
    let ko = make_group("language", "ko");

    let report = FairnessReport {
        group_stats: vec![
            make_group_stats(en, 0.95, &[("snr", 25.0)]),
            make_group_stats(ja, 0.80, &[("snr", f64::NAN)]), // NaN mean
            make_group_stats(ko, 0.70, &[("snr", 15.0)]),
        ],
        comparisons: vec![],
        max_quality_gap: 10.0,
        worst_gaps: vec![],
        is_fair: false,
    };

    let diagnosis = diagnose_bias(&report);

    let snr_ranking = diagnosis
        .rankings
        .iter()
        .find(|r| r.metric == "snr")
        .expect("Should have SNR ranking");

    // NaN groups are filtered before sorting — only finite-mean groups remain.
    // The finite groups (en=25, ko=15) give gap=10.
    assert_eq!(snr_ranking.ranked_groups.len(), 2);

    // Gap must be finite since NaN groups were excluded.
    assert!(
        snr_ranking.gap.is_finite(),
        "Gap should be finite after NaN group exclusion, got {}",
        snr_ranking.gap,
    );
    assert!((snr_ranking.gap - 10.0).abs() < 1e-10, "Expected gap=10.0");
}

#[test]
fn test_diagnose_bias_rankings_sorted_by_gap_descending() {
    // Verify the overall sort is gap-descending, not just that gaps exist
    let en = make_group("language", "en");
    let ja = make_group("language", "ja");

    let report = FairnessReport {
        group_stats: vec![
            make_group_stats(en, 0.90, &[("snr", 22.0), ("pesq", 3.8)]),
            make_group_stats(ja, 0.85, &[("snr", 18.0), ("pesq", 3.5)]),
        ],
        comparisons: vec![],
        max_quality_gap: 4.0,
        worst_gaps: vec![],
        is_fair: true,
    };

    let diagnosis = diagnose_bias(&report);

    // Should have 2 rankings (SNR gap=4.0, PESQ gap=0.3)
    assert!(
        diagnosis.rankings.len() >= 2,
        "Should have rankings for both metrics"
    );

    // Verify descending gap order
    for window in diagnosis.rankings.windows(2) {
        assert!(
            window[0].gap >= window[1].gap,
            "Rankings should be sorted by gap descending: {} should be >= {}",
            window[0].gap,
            window[1].gap,
        );
    }
}
