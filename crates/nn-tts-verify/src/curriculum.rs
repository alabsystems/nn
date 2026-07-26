// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verification-guided curriculum selection for TTS fine-tuning.
//!
//! Uses nn-tts-verify quality metrics as an automated evaluator to identify
//! which synthesized utterances need improvement. The worst-quality utterances
//! form a fine-tuning curriculum — no human evaluation required.
//!
//! # Usage
//!
//! ```text
//! let verifier = TtsVerifier::builder().with_quality().build()?;
//! let analysis = analyze_corpus(&synthesized, Some(&references), &verifier)?;
//! let curriculum = select_curriculum(&analysis, &CurriculumConfig::default());
//! // curriculum contains indices of utterances needing fine-tuning
//! ```

use crate::bounds::HardBound;
use crate::error::{validate_finite_positive, DspErrorKind, InvalidConfigKind, TtsVerifyError};
use crate::quality::QualityMetric;
use crate::TtsVerifier;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Analysis of a single synthesized utterance's quality.
#[derive(Debug, Clone)]
pub struct UtteranceAnalysis {
    /// Input text or identifier.
    pub utterance_id: String,
    /// Index into the original corpus.
    pub index: usize,
    /// Per-metric results from [`TtsVerifier`].
    pub metrics: Vec<QualityMetric>,
    /// Hard bound results.
    pub bounds: Vec<HardBound>,
    /// Names of metrics that failed their thresholds.
    pub failures: Vec<String>,
    /// Overall quality score: fraction of metrics that passed, in `[0.0, 1.0]`.
    pub quality_score: f64,
}

/// Corpus-level failure analysis.
#[derive(Debug, Clone)]
pub struct CorpusAnalysis {
    /// Per-utterance analyses, in original corpus order.
    pub utterances: Vec<UtteranceAnalysis>,
    /// Metrics sorted by failure rate (worst first).
    /// Each entry is `(metric_name, failure_rate)` where `failure_rate` is in `[0.0, 1.0]`.
    pub metric_failure_rates: Vec<(String, f64)>,
    /// Bottom-K utterances by quality score (worst first).
    pub worst_utterances: Vec<UtteranceAnalysis>,
    /// Mean quality score across the corpus.
    pub mean_quality: f64,
    /// Standard deviation of quality scores.
    pub std_quality: f64,
    /// 5th percentile quality score.
    pub p5_quality: f64,
}

/// Configuration for verification-guided curriculum selection.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CurriculumConfig {
    /// Fraction of worst utterances to include in fine-tuning set.
    /// Default: 0.10 (bottom 10%).
    pub bottom_fraction: f64,
    /// Quality score threshold: utterances below this are always included.
    /// Overrides `bottom_fraction` — if more utterances fall below the threshold
    /// than `bottom_fraction` would select, all sub-threshold utterances are included.
    /// Default: 0.5 (pass fewer than half of metrics).
    pub quality_threshold: f64,
    /// Which metrics to prioritize for failure selection.
    /// If non-empty, only these metrics contribute to `quality_score`.
    /// Default: empty (all metrics count equally).
    pub priority_metrics: Vec<String>,
}

impl CurriculumConfig {
    /// Validate that all f64 fields are finite and positive.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        validate_finite_positive(self.bottom_fraction, "bottom_fraction")?;
        validate_finite_positive(self.quality_threshold, "quality_threshold")?;
        if self.bottom_fraction > 1.0 {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::Constraint {
                    what: "bottom_fraction must be <= 1.0",
                },
            ));
        }
        Ok(())
    }
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            bottom_fraction: 0.10,
            quality_threshold: 0.5,
            priority_metrics: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/// Analyze a corpus of synthesized utterances using [`TtsVerifier`].
///
/// Each utterance is verified standalone (hard bounds + optional quality metrics).
/// If `references` is provided and lengths match, `verify_with_reference` is used
/// for richer quality metrics (MCD, cosine similarity, SNR, SDR).
///
/// # Errors
///
/// Returns [`TtsVerifyError`] if verification fails on any utterance.
/// Individual utterance failures are recorded in the analysis, not propagated.
pub fn analyze_corpus(
    utterances: &[(String, Vec<f32>)],
    references: Option<&[(String, Vec<f32>)]>,
    verifier: &TtsVerifier,
) -> Result<CorpusAnalysis, TtsVerifyError> {
    if utterances.is_empty() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::EmptyInput {
            what: "analyze_corpus: empty corpus",
        }));
    }

    let mut analyses = Vec::with_capacity(utterances.len());

    for (i, (id, samples)) in utterances.iter().enumerate() {
        // Try to verify — individual failures become low-quality analyses
        // rather than propagating errors.
        let cert_result = if let Some(refs) = references {
            if let Some((_ref_id, ref_samples)) = refs.get(i) {
                if samples.len() == ref_samples.len() {
                    verifier.verify_with_reference(samples, ref_samples)
                } else {
                    verifier.verify(samples)
                }
            } else {
                verifier.verify(samples)
            }
        } else {
            verifier.verify(samples)
        };

        let analysis = match cert_result {
            Ok(cert) => {
                let failures: Vec<String> = cert
                    .hard_bounds
                    .iter()
                    .filter(|b| !b.passed)
                    .map(|b| b.name.to_string())
                    .chain(
                        cert.quality_metrics
                            .iter()
                            .filter(|m| !m.passed)
                            .map(|m| m.name.to_string()),
                    )
                    .collect();

                let total_checks = cert.hard_bounds.len() + cert.quality_metrics.len();
                let passed_checks = cert.hard_bounds.iter().filter(|b| b.passed).count()
                    + cert.quality_metrics.iter().filter(|m| m.passed).count();
                let quality_score = if total_checks > 0 {
                    passed_checks as f64 / total_checks as f64
                } else {
                    1.0 // no checks → vacuously passed
                };

                UtteranceAnalysis {
                    utterance_id: id.clone(),
                    index: i,
                    metrics: cert.quality_metrics,
                    bounds: cert.hard_bounds,
                    failures,
                    quality_score,
                }
            }
            Err(_) => {
                // Verification failed entirely — worst possible quality.
                UtteranceAnalysis {
                    utterance_id: id.clone(),
                    index: i,
                    metrics: Vec::new(),
                    bounds: Vec::new(),
                    failures: vec!["verification_error".to_string()],
                    quality_score: 0.0,
                }
            }
        };

        analyses.push(analysis);
    }

    // Compute metric failure rates.
    let metric_failure_rates = compute_metric_failure_rates(&analyses);

    // Compute quality statistics.
    let scores: Vec<f64> = analyses.iter().map(|a| a.quality_score).collect();
    let mean_quality = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores
        .iter()
        .map(|s| (s - mean_quality) * (s - mean_quality))
        .sum::<f64>()
        / scores.len() as f64;
    let std_quality = variance.sqrt();
    let p5_quality = percentile(&scores, 5.0);

    // Worst utterances (sorted by quality_score ascending).
    let mut worst = analyses.clone();
    worst.sort_by(|a, b| a.quality_score.total_cmp(&b.quality_score));
    let n_worst = (analyses.len() / 10).max(1); // top 10%
    worst.truncate(n_worst);

    Ok(CorpusAnalysis {
        utterances: analyses,
        metric_failure_rates,
        worst_utterances: worst,
        mean_quality,
        std_quality,
        p5_quality,
    })
}

/// Select utterances for fine-tuning based on verification results.
///
/// Returns indices into the original corpus of utterances that should
/// be included in the fine-tuning dataset. Selection criteria:
///
/// 1. All utterances with `quality_score < config.quality_threshold`.
/// 2. If fewer than `bottom_fraction * corpus_size` are selected by (1),
///    add the next-worst utterances to fill the quota.
///
/// Results are sorted by quality score (worst first).
pub fn select_curriculum(analysis: &CorpusAnalysis, config: &CurriculumConfig) -> Vec<usize> {
    let n_total = analysis.utterances.len();
    let min_count = ((config.bottom_fraction * n_total as f64).ceil() as usize).max(1);

    // Rank all utterances by quality score (ascending = worst first).
    let mut ranked: Vec<(usize, f64)> = analysis
        .utterances
        .iter()
        .map(|a| (a.index, a.quality_score))
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));

    // Select: threshold-based + fraction-based (whichever is larger).
    let threshold_count = ranked
        .iter()
        .filter(|(_, score)| *score < config.quality_threshold)
        .count();
    let select_count = threshold_count.max(min_count).min(n_total);

    ranked
        .into_iter()
        .take(select_count)
        .map(|(idx, _)| idx)
        .collect()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute per-metric failure rates across the corpus.
fn compute_metric_failure_rates(analyses: &[UtteranceAnalysis]) -> Vec<(String, f64)> {
    use std::collections::HashMap;

    let n = analyses.len() as f64;
    if n == 0.0 {
        return Vec::new();
    }

    let mut failure_counts: HashMap<String, usize> = HashMap::new();

    for analysis in analyses {
        for bound in &analysis.bounds {
            if !bound.passed {
                *failure_counts.entry(bound.name.to_string()).or_default() += 1;
            }
        }
        for metric in &analysis.metrics {
            if !metric.passed {
                *failure_counts.entry(metric.name.to_string()).or_default() += 1;
            }
        }
    }

    let mut rates: Vec<(String, f64)> = failure_counts
        .into_iter()
        .map(|(name, count)| (name, count as f64 / n))
        .collect();

    // Sort by failure rate descending (worst first).
    rates.sort_by(|a, b| b.1.total_cmp(&a.1));

    rates
}

use crate::stats::percentile;

#[cfg(test)]
#[path = "curriculum_tests.rs"]
mod tests;
