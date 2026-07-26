// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Provably fair voice synthesis — group-level quality measurement.
//!
//! Provides automated fairness measurement across demographic/linguistic groups
//! using nn-tts-verify's 12+ quality metrics. Layer 1 of the two-layer
//! fairness architecture (empirical fairness; Layer 2 is CROWN-verified
//! fairness in fairness_crown.rs).
//!
//! References:
//! - Tatman (2017) "Gender and Dialect Bias in YouTube's Automatic Captions"
//! - Koenecke et al. (2020) "Racial Disparities in Automated Speech Recognition"
//! - Meyer et al. (2020) "Artie Bias Corpus"

use std::collections::HashMap;

use crate::error::{validate_finite_positive, InvalidConfigKind, TtsVerifyError};
use crate::stats::{self, percentile};
use crate::TtsVerifier;

/// A demographic or linguistic group for fairness analysis.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Group {
    /// Group dimension (e.g., "language", "gender", "accent").
    pub dimension: String,
    /// Group value (e.g., "ja", "female", "southern_us").
    pub value: String,
}

/// A single sample tagged with its group memberships.
#[derive(Debug, Clone)]
pub struct TaggedSample {
    /// Unique identifier for this sample.
    pub id: String,
    /// Group memberships (a sample can belong to multiple groups
    /// across different dimensions).
    pub groups: Vec<Group>,
    /// Audio samples (PCM f32).
    pub audio: Vec<f32>,
    /// Optional reference audio for paired metrics (MCD, PESQ).
    pub reference: Option<Vec<f32>>,
}

/// Statistics for one metric within one group.
#[derive(Debug, Clone)]
pub struct MetricStat {
    pub name: String,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub p5: f64,
    pub p95: f64,
    pub n: usize,
}

/// Per-group aggregated quality statistics.
#[derive(Debug, Clone)]
pub struct GroupStats {
    pub group: Group,
    pub n_samples: usize,
    /// Per-metric statistics.
    pub metric_stats: Vec<MetricStat>,
    /// Overall pass rate (fraction of samples passing all thresholds).
    pub pass_rate: f64,
}

/// Configuration for fairness measurement.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FairnessConfig {
    /// Significance level for statistical tests (default: 0.05).
    pub alpha: f64,
    /// Maximum acceptable quality gap between any two groups
    /// (in metric-specific units, e.g., PESQ MOS points).
    pub max_gap: f64,
    /// Minimum samples per group for valid comparison (default: 30).
    pub min_samples_per_group: usize,
    /// Which metrics to include in fairness analysis.
    /// Empty means include all available metrics.
    pub metrics: Vec<String>,
}

impl FairnessConfig {
    /// Validate that all f64 fields are finite and positive.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        validate_finite_positive(self.alpha, "alpha")?;
        validate_finite_positive(self.max_gap, "max_gap")?;
        if self.alpha > 1.0 {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::Constraint {
                    what: "alpha must be <= 1.0",
                },
            ));
        }
        Ok(())
    }
}

impl Default for FairnessConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            max_gap: 1.0,
            min_samples_per_group: 30,
            metrics: Vec::new(),
        }
    }
}

/// Result of a pairwise group comparison for one metric.
#[derive(Debug, Clone)]
pub struct PairwiseComparison {
    pub group_a: Group,
    pub group_b: Group,
    pub metric: String,
    /// Difference in means (group_a.mean - group_b.mean).
    pub mean_diff: f64,
    /// Welch's t-test statistic.
    pub t_statistic: f64,
    /// Two-sided p-value (adjusted for multiple comparisons).
    pub p_value: f64,
    /// Effect size (Cohen's d).
    pub cohens_d: f64,
    /// Is the difference statistically significant after correction?
    pub significant: bool,
}

/// Full fairness assessment across all groups and metrics.
#[derive(Debug, Clone)]
pub struct FairnessReport {
    /// Per-group statistics.
    pub group_stats: Vec<GroupStats>,
    /// All pairwise comparisons.
    pub comparisons: Vec<PairwiseComparison>,
    /// Maximum quality gap found (worst-case across all pairs and metrics).
    pub max_quality_gap: f64,
    /// Which (group_a, group_b, metric, gap) pairs have the largest gaps.
    pub worst_gaps: Vec<(Group, Group, String, f64)>,
    /// Overall fairness verdict.
    pub is_fair: bool,
}

/// Measure quality per group and compute fairness statistics.
///
/// Verifies each sample using the provided `TtsVerifier`, groups results by
/// `Group` dimension+value, computes per-group statistics, and runs pairwise
/// Welch's t-tests with Holm-Bonferroni correction.
pub fn measure_fairness(
    samples: &[TaggedSample],
    verifier: &TtsVerifier,
    config: &FairnessConfig,
    sample_rate: u32,
) -> Result<FairnessReport, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }

    // Step 1: Verify each sample and collect per-group metric values.
    // Key: (Group, metric_name) -> Vec<f64>
    let mut group_metric_values: HashMap<(Group, String), Vec<f64>> = HashMap::new();
    let mut group_pass_counts: HashMap<Group, (usize, usize)> = HashMap::new(); // (passed, total)

    // Suppress unused variable warning for sample_rate — it's used for API
    // consistency (verifier already has sample_rate configured).
    let _ = sample_rate;

    for sample in samples {
        let cert = if let Some(ref reference) = sample.reference {
            verifier.verify_with_reference(&sample.audio, reference)?
        } else {
            verifier.verify(&sample.audio)?
        };

        let all_passed = cert.overall_passed;

        // Collect metric values per group
        for group in &sample.groups {
            let (passed, total) = group_pass_counts.entry(group.clone()).or_insert((0, 0));
            *total += 1;
            if all_passed {
                *passed += 1;
            }

            for metric in &cert.quality_metrics {
                let name = metric.name.to_string();
                if !config.metrics.is_empty() && !config.metrics.contains(&name) {
                    continue;
                }
                group_metric_values
                    .entry((group.clone(), name))
                    .or_default()
                    .push(metric.value);
            }

            // Also collect hard bound values as metrics
            for bound in &cert.hard_bounds {
                let name = format!("hard_{}", bound.name);
                if !config.metrics.is_empty() && !config.metrics.contains(&name) {
                    continue;
                }
                group_metric_values
                    .entry((group.clone(), name))
                    .or_default()
                    .push(bound.value);
            }
        }
    }

    // Step 2: Compute per-group statistics.
    let all_groups: Vec<Group> = group_pass_counts.keys().cloned().collect();
    let mut group_stats = Vec::new();

    for group in &all_groups {
        let (passed, total) = group_pass_counts.get(group).copied().unwrap_or((0, 0));
        let pass_rate = if total > 0 {
            passed as f64 / total as f64
        } else {
            0.0
        };

        let mut metric_stats = Vec::new();
        for ((g, metric_name), values) in &group_metric_values {
            if g != group || values.is_empty() {
                continue;
            }
            metric_stats.push(compute_metric_stat(metric_name.clone(), values));
        }

        group_stats.push(GroupStats {
            group: group.clone(),
            n_samples: total,
            metric_stats,
            pass_rate,
        });
    }

    // Step 3: Pairwise comparisons within each dimension.
    let mut raw_comparisons: Vec<(PairwiseComparison, f64)> = Vec::new(); // (comparison, raw_p)

    // Group groups by dimension
    let mut dimension_groups: HashMap<&str, Vec<&Group>> = HashMap::new();
    for group in &all_groups {
        dimension_groups
            .entry(&group.dimension)
            .or_default()
            .push(group);
    }

    for groups_in_dim in dimension_groups.values() {
        for (i, &ga) in groups_in_dim.iter().enumerate() {
            for &gb in groups_in_dim.iter().skip(i + 1) {
                // For each metric present in both groups
                let ga_metrics: Vec<String> = group_metric_values
                    .keys()
                    .filter(|(g, _)| g == ga)
                    .map(|(_, m)| m.clone())
                    .collect();

                for metric_name in &ga_metrics {
                    let vals_a = match group_metric_values.get(&(ga.clone(), metric_name.clone())) {
                        Some(v) if v.len() >= config.min_samples_per_group => v,
                        _ => continue,
                    };
                    let vals_b = match group_metric_values.get(&(gb.clone(), metric_name.clone())) {
                        Some(v) if v.len() >= config.min_samples_per_group => v,
                        _ => continue,
                    };

                    let (t_stat, _df, raw_p) = stats::welch_t_test(vals_a, vals_b)?;
                    let d = stats::cohens_d(vals_a, vals_b)?;
                    let mean_a = vals_a.iter().sum::<f64>() / vals_a.len() as f64;
                    let mean_b = vals_b.iter().sum::<f64>() / vals_b.len() as f64;

                    let comp = PairwiseComparison {
                        group_a: ga.clone(),
                        group_b: gb.clone(),
                        metric: metric_name.clone(),
                        mean_diff: mean_a - mean_b,
                        t_statistic: t_stat,
                        p_value: raw_p, // Will be adjusted below
                        cohens_d: d,
                        significant: false, // Will be set below
                    };
                    raw_comparisons.push((comp, raw_p));
                }
            }
        }
    }

    // Step 4: Apply Holm-Bonferroni correction.
    let raw_ps: Vec<f64> = raw_comparisons.iter().map(|(_, p)| *p).collect();
    let adjusted_ps = stats::holm_bonferroni(&raw_ps)?;

    let mut comparisons = Vec::new();
    for (i, (mut comp, _)) in raw_comparisons.into_iter().enumerate() {
        let adj_p = adjusted_ps.get(i).copied().unwrap_or(1.0);
        comp.p_value = adj_p;
        comp.significant = adj_p < config.alpha;
        comparisons.push(comp);
    }

    // Step 5: Compute max quality gap and worst gaps.
    let mut max_quality_gap = 0.0_f64;
    let mut worst_gaps = Vec::new();

    for comp in &comparisons {
        let gap = comp.mean_diff.abs();
        if gap > max_quality_gap {
            max_quality_gap = gap;
        }
        if comp.significant {
            worst_gaps.push((
                comp.group_a.clone(),
                comp.group_b.clone(),
                comp.metric.clone(),
                gap,
            ));
        }
    }

    // Sort worst gaps by gap size descending
    worst_gaps.sort_by(|a, b| b.3.total_cmp(&a.3));

    let is_fair = max_quality_gap < config.max_gap && !comparisons.iter().any(|c| c.significant);

    Ok(FairnessReport {
        group_stats,
        comparisons,
        max_quality_gap,
        worst_gaps,
        is_fair,
    })
}

/// Compute descriptive statistics for a set of metric values.
fn compute_metric_stat(name: String, values: &[f64]) -> MetricStat {
    let n = values.len();
    if n == 0 {
        return MetricStat {
            name,
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            p5: 0.0,
            p95: 0.0,
            n: 0,
        };
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    let min = sorted[0];
    let max = sorted[n - 1];
    let p5 = percentile(&sorted, 5.0);
    let p95 = percentile(&sorted, 95.0);

    MetricStat {
        name,
        mean,
        std_dev,
        min,
        max,
        p5,
        p95,
        n,
    }
}

#[cfg(test)]
#[path = "fairness_tests.rs"]
mod tests;
