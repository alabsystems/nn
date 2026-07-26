// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bias diagnosis and targeted debiasing curriculum selection.
//!
//! Phase 3 of the two-layer fairness architecture. Uses the empirical
//! [`FairnessReport`] from Phase 1 to:
//! 1. Identify which quality dimensions are biased and for which groups
//! 2. Rank groups from worst to best per metric
//! 3. Select samples from the worst-performing group for targeted fine-tuning
//!
//! Integrates with the verification-guided curriculum (#1726):
//! `select_debiasing_curriculum` selects samples from the worst-performing
//! group as fine-tuning curriculum, using the fairness gap as quality signal.
//!
//! References:
//! - Design doc: `designs/archive/2026-03-10-provably-fair-voice.md` Phase 3

use crate::fairness::{FairnessReport, Group, TaggedSample};

/// Ranking of groups within one fairness dimension on one metric.
#[derive(Debug, Clone)]
pub struct DimensionRanking {
    /// Fairness dimension (e.g., "language", "gender").
    pub dimension: String,
    /// Quality metric name (e.g., "hard_non_silence", "mcd").
    pub metric: String,
    /// Groups sorted by mean metric value (worst first).
    /// Each entry is (group, mean_value).
    pub ranked_groups: Vec<(Group, f64)>,
    /// Absolute gap between worst and best group mean.
    pub gap: f64,
}

/// Diagnosis of which quality dimensions are biased and for which groups.
#[derive(Debug, Clone)]
pub struct BiasDiagnosis {
    /// Per (dimension, metric): groups ranked from worst to best.
    pub rankings: Vec<DimensionRanking>,
    /// Actionable recommendations derived from the rankings.
    pub recommendations: Vec<String>,
}

/// Diagnose bias from a fairness report.
///
/// Produces ranked lists of groups per dimension per metric, sorted by
/// mean metric value (worst first). "Worst" is defined as the lowest mean
/// value — this is correct for metrics where higher is better (PESQ, STOI,
/// SNR, HNR, pass_rate). For metrics where lower is better (MCD, spectral
/// distance), callers should negate the metric values before feeding into
/// the fairness pipeline, or interpret rankings accordingly.
///
/// Also generates actionable recommendations for the worst-performing groups.
pub fn diagnose_bias(report: &FairnessReport) -> BiasDiagnosis {
    let mut rankings = Vec::new();

    // Collect unique dimensions
    let mut dimensions: Vec<String> = report
        .group_stats
        .iter()
        .map(|gs| gs.group.dimension.clone())
        .collect();
    dimensions.sort();
    dimensions.dedup();

    // Collect all metric names across all groups
    let mut all_metrics: Vec<String> = report
        .group_stats
        .iter()
        .flat_map(|gs| gs.metric_stats.iter().map(|ms| ms.name.clone()))
        .collect();
    all_metrics.sort();
    all_metrics.dedup();

    for dimension in &dimensions {
        let groups_in_dim: Vec<&crate::fairness::GroupStats> = report
            .group_stats
            .iter()
            .filter(|gs| &gs.group.dimension == dimension)
            .collect();

        for metric_name in &all_metrics {
            // Collect (group, mean_value) for groups that have this metric
            let mut group_means: Vec<(Group, f64)> = Vec::new();

            for gs in &groups_in_dim {
                if let Some(ms) = gs.metric_stats.iter().find(|ms| &ms.name == metric_name) {
                    group_means.push((gs.group.clone(), ms.mean));
                }
            }

            if group_means.len() < 2 {
                continue; // Need at least 2 groups to rank
            }

            // Filter NaN group means before sorting (NaN corrupts sort ordering).
            group_means.retain(|(_group, mean)| mean.is_finite());
            // Sort by mean value ascending (worst = lowest first)
            group_means.sort_by(|a, b| a.1.total_cmp(&b.1));

            let worst_mean = group_means[0].1;
            let best_mean = group_means[group_means.len() - 1].1;
            let gap = (best_mean - worst_mean).abs();

            rankings.push(DimensionRanking {
                dimension: dimension.clone(),
                metric: metric_name.clone(),
                ranked_groups: group_means,
                gap,
            });
        }
    }

    // Sort rankings by gap descending (largest bias first).
    // total_cmp provides IEEE 754 totalOrder — NaN sorts after +Inf.
    rankings.sort_by(|a, b| b.gap.total_cmp(&a.gap));

    // Generate recommendations
    let recommendations = generate_recommendations(&rankings, report);

    BiasDiagnosis {
        rankings,
        recommendations,
    }
}

/// Generate actionable recommendations from ranked bias findings.
fn generate_recommendations(rankings: &[DimensionRanking], report: &FairnessReport) -> Vec<String> {
    let mut recs = Vec::new();

    // Recommendation for the top 3 worst gaps
    for ranking in rankings.iter().take(3) {
        if ranking.gap < 1e-10 {
            continue; // Skip negligible gaps
        }
        let worst = &ranking.ranked_groups[0];
        let best = &ranking.ranked_groups[ranking.ranked_groups.len() - 1];
        recs.push(format!(
            "Metric '{}' in dimension '{}': group '{}' (mean={:.4}) underperforms group '{}' (mean={:.4}) by gap={:.4}. \
             Consider targeted fine-tuning on '{}' samples.",
            ranking.metric,
            ranking.dimension,
            worst.0.value,
            worst.1,
            best.0.value,
            best.1,
            ranking.gap,
            worst.0.value,
        ));
    }

    // Overall recommendation based on significant comparisons
    let n_significant = report.comparisons.iter().filter(|c| c.significant).count();
    if n_significant > 0 {
        recs.push(format!(
            "{} of {} pairwise comparisons are statistically significant. \
             Focus debiasing on the most affected groups.",
            n_significant,
            report.comparisons.len(),
        ));
    } else {
        recs.push("No statistically significant differences found between groups.".to_string());
    }

    recs
}

/// Average of all metric means for a group. Returns 0.0 if no metrics.
fn avg_metric_mean(metric_stats: &[crate::fairness::MetricStat]) -> f64 {
    if metric_stats.is_empty() {
        return 0.0;
    }
    let sum: f64 = metric_stats.iter().map(|m| m.mean).sum();
    sum / metric_stats.len() as f64
}

/// Select sample indices from the worst-performing group for targeted fine-tuning.
///
/// Given a fairness report and the original samples, identifies the worst
/// group in `target_dimension` (by pass rate, then by average metric mean)
/// and returns indices into `samples` that belong to that group.
///
/// Integrates with the verification-guided curriculum from #1726: these
/// indices can be passed directly to the curriculum selection pipeline
/// as high-priority training samples.
///
/// # Arguments
///
/// * `report` — The fairness report from `measure_fairness()`.
/// * `samples` — The original tagged samples used to generate the report.
/// * `target_dimension` — Which fairness dimension to optimize (e.g., "language").
///
/// # Returns
///
/// Indices into `samples` belonging to the worst-performing group in the
/// target dimension. Returns empty if `target_dimension` has no groups
/// or fewer than 2 groups.
///
/// # Sorting
///
/// Groups are sorted by pass_rate ascending, with ties broken by the average
/// of all metric means (not just the first metric). This avoids dependence
/// on HashMap iteration order.
pub fn select_debiasing_curriculum(
    report: &FairnessReport,
    samples: &[TaggedSample],
    target_dimension: &str,
) -> Vec<usize> {
    // Find groups in the target dimension
    let mut groups_in_dim: Vec<&crate::fairness::GroupStats> = report
        .group_stats
        .iter()
        .filter(|gs| gs.group.dimension == target_dimension)
        .collect();

    if groups_in_dim.len() < 2 {
        return Vec::new(); // Need at least 2 groups to identify "worst"
    }

    // Sort by pass_rate ascending (worst = lowest pass_rate first).
    // Break ties by average metric mean across ALL metrics (lower = worse).
    // Using total_cmp avoids NaN-corrupted ordering and provides deterministic
    // results regardless of HashMap iteration order for metric_stats.
    groups_in_dim.sort_by(|a, b| {
        a.pass_rate.total_cmp(&b.pass_rate).then_with(|| {
            let a_avg = avg_metric_mean(&a.metric_stats);
            let b_avg = avg_metric_mean(&b.metric_stats);
            a_avg.total_cmp(&b_avg)
        })
    });

    let worst_group = &groups_in_dim[0].group;

    // Collect indices of samples belonging to the worst group
    samples
        .iter()
        .enumerate()
        .filter(|(_, sample)| {
            sample
                .groups
                .iter()
                .any(|g| g.dimension == worst_group.dimension && g.value == worst_group.value)
        })
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
#[path = "fairness_diagnosis_tests.rs"]
mod tests;
