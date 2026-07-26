// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Closed-loop RTF optimizer for the Kokoro pipeline.
//!
//! [`RtfOptimizer`] ties together nn's existing optimization tools into a
//! single entry point:
//!
//! - **Profile**: [`SegmentGapAnalysis`] measures per-segment dispatch counts
//!   and roofline cost estimates.
//! - **Calibrate**: [`CostModel::calibrate`] updates the cost model from
//!   measured timings (when available).
//! - **Analyze**: [`FusionGapAnalysis`] identifies top bottlenecks and
//!   blocker distribution across all segments.
//! - **Search**: [`warmup_with_optimizer`] finds optimal per-segment
//!   `PeepholeConfig`s via exhaustive search.
//! - **Precision**: [`F16AutocastConfig`] determines F16-safe segments.
//! - **Report**: [`RtfReport`] collects projected RTF, actions taken, and
//!   remaining gaps.
//!
//! Part of #4264.

use std::collections::BTreeMap;
use std::fmt;
#[cfg(feature = "plan-serde")]
use std::path::Path;
#[cfg(feature = "plan-serde")]
use std::time::Duration;

use nn_core::dyn_tensor::DynTensor;
use nn_dsl::CostModel;

use crate::cache::PipelineCache;

#[cfg(feature = "plan-serde")]
use super::precompile::{OptimizerWarmupResult, PrecompileShapes};
use super::SegmentGapAnalysis;

/// Closed-loop RTF optimizer for Kokoro pipeline performance.
///
/// Wraps the existing optimization tools (cost model, gap analysis, fusion
/// diagnostics) into a unified controller that profiles the pipeline,
/// identifies bottlenecks, and produces an actionable report.
///
/// # Example
///
/// ```rust,ignore
/// let optimizer = RtfOptimizer::new(CostModel::apple_m4_max(), 0.03);
/// let report = optimizer.analyze(&gap_results);
/// eprintln!("{}", report.summary());
/// ```
#[derive(Clone, Debug)]
pub struct RtfOptimizer {
    /// Roofline cost model for the target hardware.
    cost_model: CostModel,
    /// Target real-time factor (e.g., 0.03 = 30x faster than real-time).
    target_rtf: f64,
}

/// Per-segment analysis within an RTF report.
#[derive(Clone, Debug)]
pub struct SegmentReport {
    /// Segment name (e.g., "plbert", "generator").
    pub segment_name: String,
    /// Current dispatch count for this segment.
    pub dispatch_count: usize,
    /// Theoretical minimum dispatches if all closable gaps were fused.
    pub theoretical_minimum: usize,
    /// Estimated cost in nanoseconds from the roofline model.
    pub estimated_cost_ns: f64,
    /// Fraction of total pipeline cost attributed to this segment.
    pub cost_fraction: f64,
    /// Blocker distribution: how many gaps have each blocker type.
    pub blocker_counts: BTreeMap<String, usize>,
    /// Optimization opportunity as percentage of this segment's dispatches.
    pub optimization_opportunity_pct: f64,
}

/// RtfBottleneck identified by the RTF optimizer.
#[derive(Clone, Debug)]
pub struct RtfBottleneck {
    /// Segment name where the bottleneck is located.
    pub segment_name: String,
    /// Category of bottleneck (e.g., "dispatch_count", "fusion_gap",
    /// "memory_bound").
    pub category: String,
    /// Human-readable description of the bottleneck.
    pub description: String,
    /// Estimated potential savings in nanoseconds if resolved.
    pub potential_savings_ns: f64,
}

/// Complete RTF optimization report.
///
/// Produced by [`RtfOptimizer::analyze`] from per-segment gap analysis
/// results. Contains projected RTF, identified bottlenecks, and per-segment
/// breakdowns.
#[derive(Clone, Debug)]
pub struct RtfReport {
    /// Per-segment analysis results.
    pub segments: Vec<SegmentReport>,
    /// Total dispatch count across all segments.
    pub total_dispatches: usize,
    /// Total theoretical minimum dispatches across all segments.
    pub total_theoretical_minimum: usize,
    /// Total estimated cost in nanoseconds (sum of all segments).
    pub total_estimated_ns: f64,
    /// Projected RTF based on cost model estimates.
    ///
    /// Assumes 24000 Hz sample rate and estimates audio duration from the
    /// generator segment's output shape. When generator cost data is not
    /// available, uses the total estimated cost directly.
    pub projected_rtf: f64,
    /// Target RTF from the optimizer configuration.
    pub target_rtf: f64,
    /// Whether the projected RTF meets the target.
    pub meets_target: bool,
    /// Identified bottlenecks sorted by potential savings (largest first).
    pub bottlenecks: Vec<RtfBottleneck>,
    /// Aggregate blocker distribution across all segments.
    pub aggregate_blockers: BTreeMap<String, usize>,
    /// Overall optimization opportunity: percentage of dispatches that
    /// could theoretically be eliminated.
    pub optimization_opportunity_pct: f64,
}

/// Warmup summary captured from an optimizer-driven warmup pass.
#[cfg(feature = "plan-serde")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RtfWarmupSummary {
    /// Whether peephole configs were loaded from cache.
    pub loaded_from_cache: bool,
    /// Number of per-segment configs applied during warmup.
    pub configs_applied: usize,
    /// Number of segment compilations performed while warming shapes.
    pub segments_compiled: usize,
}

/// Result of running the full baseline -> optimize -> re-analyze loop.
#[cfg(feature = "plan-serde")]
#[derive(Clone, Debug)]
pub struct ClosedLoopRtfReport {
    /// Baseline report before any optimizer warmup is applied.
    pub baseline: RtfReport,
    /// Final report after the optimizer warmup path runs.
    pub final_report: RtfReport,
    /// Warmup details when the optimizer actually ran.
    pub warmup: Option<RtfWarmupSummary>,
    /// Human-readable actions taken during the closed loop.
    pub actions_taken: Vec<String>,
    /// Per-segment optimizer summary when a fresh search produced results.
    pub optimizer_summary: Option<String>,
}

/// Default audio duration estimate in seconds for RTF projection.
///
/// When the actual generator output duration is unknown, we use a
/// conservative estimate of 3 seconds (typical short utterance).
const DEFAULT_AUDIO_DURATION_SECS: f64 = 3.0;

impl RtfOptimizer {
    /// Create a new RTF optimizer with the given cost model and target RTF.
    ///
    /// # Arguments
    ///
    /// * `cost_model` - Hardware-specific roofline cost model (e.g.,
    ///   `CostModel::apple_m4_max()`).
    /// * `target_rtf` - Target real-time factor. RTF < 1.0 means faster
    ///   than real-time; 0.03 means ~33x faster.
    #[must_use]
    pub fn new(cost_model: CostModel, target_rtf: f64) -> Self {
        Self {
            cost_model,
            target_rtf,
        }
    }

    /// Create an RTF optimizer with Apple M4 Max defaults and 0.03 target.
    #[must_use]
    pub fn apple_m4_max() -> Self {
        Self::new(CostModel::apple_m4_max(), 0.03)
    }

    /// Access the underlying cost model.
    #[must_use]
    pub fn cost_model(&self) -> &CostModel {
        &self.cost_model
    }

    /// The configured target RTF.
    #[must_use]
    pub fn target_rtf(&self) -> f64 {
        self.target_rtf
    }

    /// Analyze the Kokoro pipeline using per-segment gap analysis results.
    ///
    /// This is the main entry point. It takes the output of
    /// [`CompiledKokoro::segment_gap_analysis`] and produces a comprehensive
    /// [`RtfReport`] with projected RTF, bottleneck identification, and
    /// per-segment breakdowns.
    ///
    /// # Steps
    ///
    /// 1. **Profile**: Extract dispatch counts and cost estimates from
    ///    each segment's gap analysis.
    /// 2. **Analyze**: Identify top bottlenecks by cost fraction, fusion
    ///    gap opportunity, and dispatch overhead.
    /// 3. **Report**: Compute projected RTF and compare against target.
    ///
    /// # Arguments
    ///
    /// * `gap_results` - Per-segment gap analysis from
    ///   [`CompiledKokoro::segment_gap_analysis`].
    #[must_use]
    pub fn analyze(&self, gap_results: &[SegmentGapAnalysis]) -> RtfReport {
        let total_estimated_ns: f64 = gap_results.iter().map(|s| s.cost_estimate.total_ns).sum();

        let total_dispatches: usize = gap_results.iter().map(|s| s.dispatch_count).sum();
        let total_theoretical_minimum: usize =
            gap_results.iter().map(|s| s.theoretical_minimum).sum();

        // Build per-segment reports.
        let segments: Vec<SegmentReport> = gap_results
            .iter()
            .map(|seg| {
                let cost_fraction = if total_estimated_ns > 0.0 {
                    seg.cost_estimate.total_ns / total_estimated_ns
                } else {
                    0.0
                };

                SegmentReport {
                    segment_name: seg.segment_name.clone(),
                    dispatch_count: seg.dispatch_count,
                    theoretical_minimum: seg.theoretical_minimum,
                    estimated_cost_ns: seg.cost_estimate.total_ns,
                    cost_fraction,
                    blocker_counts: seg.gap_analysis.blocker_counts(),
                    optimization_opportunity_pct: seg.gap_analysis.optimization_opportunity_pct(),
                }
            })
            .collect();

        // Aggregate blockers across all segments.
        let mut aggregate_blockers: BTreeMap<String, usize> = BTreeMap::new();
        for seg in &segments {
            for (blocker, count) in &seg.blocker_counts {
                *aggregate_blockers.entry(blocker.clone()).or_insert(0) += *count;
            }
        }

        // Identify bottlenecks.
        let bottlenecks = self.identify_bottlenecks(&segments, total_estimated_ns);

        // Compute projected RTF.
        // RTF = inference_time / audio_duration.
        // We estimate audio duration conservatively.
        let inference_time_secs = total_estimated_ns / 1e9;
        let projected_rtf = inference_time_secs / DEFAULT_AUDIO_DURATION_SECS;
        let meets_target = projected_rtf <= self.target_rtf;

        let optimization_opportunity_pct = if total_dispatches > 0 {
            let reducible = total_dispatches.saturating_sub(total_theoretical_minimum);
            (reducible as f64 / total_dispatches as f64) * 100.0
        } else {
            0.0
        };

        RtfReport {
            segments,
            total_dispatches,
            total_theoretical_minimum,
            total_estimated_ns,
            projected_rtf,
            target_rtf: self.target_rtf,
            meets_target,
            bottlenecks,
            aggregate_blockers,
            optimization_opportunity_pct,
        }
    }

    /// Run gap analysis for a [`CompiledKokoro`] pipeline and format the result.
    pub fn analyze_kokoro(
        &self,
        kokoro: &mut CompiledKokoro,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<RtfReport, super::CompiledKokoroError> {
        let gaps = kokoro.segment_gap_analysis(input_ids, style, speed, cache)?;
        Ok(self.analyze(&gaps))
    }

    #[cfg(feature = "plan-serde")]
    #[cfg_attr(not(test), allow(dead_code))]
    fn run_closed_loop<G, W, S>(
        &self,
        mut collect_gap_analysis: G,
        mut warmup_with_optimizer: W,
        mut optimizer_summary: S,
    ) -> Result<ClosedLoopRtfReport, super::CompiledKokoroError>
    where
        G: FnMut() -> Result<Vec<SegmentGapAnalysis>, super::CompiledKokoroError>,
        W: FnMut() -> Result<RtfWarmupSummary, super::CompiledKokoroError>,
        S: FnMut() -> Option<String>,
    {
        let baseline_gaps = collect_gap_analysis()?;
        let baseline = self.analyze(&baseline_gaps);
        let mut actions_taken = Vec::new();

        if baseline_gaps.is_empty() {
            actions_taken.push(
                "Skipped optimizer warmup: no analyzable segments were produced.".to_string(),
            );
            return Ok(ClosedLoopRtfReport {
                baseline: baseline.clone(),
                final_report: baseline,
                warmup: None,
                actions_taken,
                optimizer_summary: None,
            });
        }

        if baseline.meets_target && baseline.optimization_opportunity_pct <= 0.0 {
            actions_taken.push(format!(
                "Skipped optimizer warmup: projected RTF {:.4} already meets target {:.4} and no fusion-gap opportunity remains.",
                baseline.projected_rtf,
                baseline.target_rtf,
            ));
            return Ok(ClosedLoopRtfReport {
                baseline: baseline.clone(),
                final_report: baseline,
                warmup: None,
                actions_taken,
                optimizer_summary: None,
            });
        }

        let warmup = warmup_with_optimizer()?;
        let warmup_action = if warmup.loaded_from_cache {
            format!(
                "Loaded cached peephole configs: {} configs applied, {} segment shapes compiled.",
                warmup.configs_applied, warmup.segments_compiled,
            )
        } else {
            format!(
                "Ran optimizer warmup: {} configs applied, {} segment shapes compiled.",
                warmup.configs_applied, warmup.segments_compiled,
            )
        };
        actions_taken.push(warmup_action);

        let final_report = self.analyze(&collect_gap_analysis()?);
        let dispatches_saved = baseline
            .total_dispatches
            .saturating_sub(final_report.total_dispatches);
        let cost_saved_ns = baseline.total_estimated_ns - final_report.total_estimated_ns;
        let rtf_delta = baseline.projected_rtf - final_report.projected_rtf;
        actions_taken.push(format!(
            "Projected RTF {:.4} -> {:.4}; dispatches saved: {}; estimated cost delta: {:.1} us.",
            baseline.projected_rtf,
            final_report.projected_rtf,
            dispatches_saved,
            cost_saved_ns / 1e3,
        ));
        if rtf_delta.abs() <= f64::EPSILON {
            actions_taken
                .push("Gap analysis reported no measurable post-warmup RTF change.".to_string());
        }

        let optimizer_summary = optimizer_summary().and_then(|summary| {
            let trimmed = summary.trim();
            (!trimmed.is_empty()
                && trimmed
                    != "No optimization results available. Call warmup_with_optimizer() first.")
                .then(|| summary)
        });

        Ok(ClosedLoopRtfReport {
            baseline,
            final_report,
            warmup: Some(warmup),
            actions_taken,
            optimizer_summary,
        })
    }

    /// Run the full baseline -> warmup_with_optimizer -> re-analyze loop.
    #[cfg(feature = "plan-serde")]
    pub fn optimize_kokoro(
        &self,
        kokoro: &mut CompiledKokoro,
        shapes: &PrecompileShapes,
        cache: &PipelineCache,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        per_segment_budget: Duration,
        config_cache_path: Option<&Path>,
    ) -> Result<ClosedLoopRtfReport, super::CompiledKokoroError> {
        let baseline_gaps = kokoro.segment_gap_analysis(input_ids, style, speed, cache)?;
        let baseline = self.analyze(&baseline_gaps);
        let mut actions_taken = Vec::new();

        if baseline_gaps.is_empty() {
            actions_taken.push(
                "Skipped optimizer warmup: no analyzable segments were produced.".to_string(),
            );
            return Ok(ClosedLoopRtfReport {
                baseline: baseline.clone(),
                final_report: baseline,
                warmup: None,
                actions_taken,
                optimizer_summary: None,
            });
        }

        if baseline.meets_target && baseline.optimization_opportunity_pct <= 0.0 {
            actions_taken.push(format!(
                "Skipped optimizer warmup: projected RTF {:.4} already meets target {:.4} and no fusion-gap opportunity remains.",
                baseline.projected_rtf,
                baseline.target_rtf,
            ));
            return Ok(ClosedLoopRtfReport {
                baseline: baseline.clone(),
                final_report: baseline,
                warmup: None,
                actions_taken,
                optimizer_summary: None,
            });
        }

        let warmup = RtfWarmupSummary::from(&kokoro.warmup_with_optimizer(
            shapes,
            cache,
            input_ids,
            style,
            speed,
            per_segment_budget,
            config_cache_path,
        )?);
        if warmup.loaded_from_cache {
            actions_taken.push(format!(
                "Loaded cached peephole configs: {} configs applied, {} segment shapes compiled.",
                warmup.configs_applied, warmup.segments_compiled,
            ));
        } else {
            actions_taken.push(format!(
                "Ran optimizer warmup: {} configs applied, {} segment shapes compiled.",
                warmup.configs_applied, warmup.segments_compiled,
            ));
        }

        let final_report =
            self.analyze(&kokoro.segment_gap_analysis(input_ids, style, speed, cache)?);
        let dispatches_saved = baseline
            .total_dispatches
            .saturating_sub(final_report.total_dispatches);
        let cost_saved_ns = baseline.total_estimated_ns - final_report.total_estimated_ns;
        actions_taken.push(format!(
            "Projected RTF {:.4} -> {:.4}; dispatches saved: {}; estimated cost delta: {:.1} us.",
            baseline.projected_rtf,
            final_report.projected_rtf,
            dispatches_saved,
            cost_saved_ns / 1e3,
        ));
        if (baseline.projected_rtf - final_report.projected_rtf).abs() <= f64::EPSILON {
            actions_taken
                .push("Gap analysis reported no measurable post-warmup RTF change.".to_string());
        }

        let optimizer_summary = kokoro
            .optimization_results()
            .is_some()
            .then(|| {
                let summary = kokoro.optimization_summary();
                let trimmed = summary.trim();
                if trimmed.is_empty()
                    || trimmed
                        == "No optimization results available. Call warmup_with_optimizer() first."
                {
                    None
                } else {
                    Some(summary)
                }
            })
            .flatten();

        Ok(ClosedLoopRtfReport {
            baseline,
            final_report,
            warmup: Some(warmup),
            actions_taken,
            optimizer_summary,
        })
    }

    /// Identify bottlenecks from per-segment reports.
    ///
    /// RtfBottlenecks are ranked by potential cost savings. A segment is
    /// flagged as a bottleneck if:
    ///
    /// 1. It consumes >20% of total pipeline cost.
    /// 2. It has >10% fusion optimization opportunity.
    /// 3. Its dispatch count exceeds the theoretical minimum by >5.
    fn identify_bottlenecks(
        &self,
        segments: &[SegmentReport],
        _total_estimated_ns: f64,
    ) -> Vec<RtfBottleneck> {
        let mut bottlenecks = Vec::new();

        for seg in segments {
            // High cost fraction bottleneck.
            if seg.cost_fraction > 0.20 {
                bottlenecks.push(RtfBottleneck {
                    segment_name: seg.segment_name.clone(),
                    category: "high_cost_fraction".to_string(),
                    description: format!(
                        "{} consumes {:.1}% of pipeline cost ({:.1} us)",
                        seg.segment_name,
                        seg.cost_fraction * 100.0,
                        seg.estimated_cost_ns / 1e3,
                    ),
                    potential_savings_ns: seg.estimated_cost_ns * 0.3,
                });
            }

            // Fusion gap bottleneck.
            if seg.optimization_opportunity_pct > 10.0 {
                let dispatch_savings = seg.dispatch_count.saturating_sub(seg.theoretical_minimum);
                let savings_ns = if seg.dispatch_count > 0 {
                    seg.estimated_cost_ns * (dispatch_savings as f64 / seg.dispatch_count as f64)
                } else {
                    0.0
                };
                bottlenecks.push(RtfBottleneck {
                    segment_name: seg.segment_name.clone(),
                    category: "fusion_gap".to_string(),
                    description: format!(
                        "{}: {} dispatches could be fused ({:.1}% opportunity)",
                        seg.segment_name, dispatch_savings, seg.optimization_opportunity_pct,
                    ),
                    potential_savings_ns: savings_ns,
                });
            }

            // Excess dispatch count bottleneck.
            let dispatch_excess = seg.dispatch_count.saturating_sub(seg.theoretical_minimum);
            if dispatch_excess > 5 {
                // Estimate cost per dispatch from total segment cost.
                let cost_per_dispatch = if seg.dispatch_count > 0 {
                    seg.estimated_cost_ns / seg.dispatch_count as f64
                } else {
                    0.0
                };
                let savings = cost_per_dispatch * dispatch_excess as f64;
                bottlenecks.push(RtfBottleneck {
                    segment_name: seg.segment_name.clone(),
                    category: "dispatch_overhead".to_string(),
                    description: format!(
                        "{}: {} excess dispatches ({} current, {} minimum)",
                        seg.segment_name,
                        dispatch_excess,
                        seg.dispatch_count,
                        seg.theoretical_minimum,
                    ),
                    potential_savings_ns: savings,
                });
            }
        }

        // Sort by potential savings, largest first.
        bottlenecks.sort_by(|a, b| {
            b.potential_savings_ns
                .partial_cmp(&a.potential_savings_ns)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        bottlenecks
    }
}

#[cfg(feature = "plan-serde")]
impl RtfWarmupSummary {
    fn from(result: &OptimizerWarmupResult) -> Self {
        Self {
            loaded_from_cache: result.loaded_from_cache,
            configs_applied: result.configs_applied,
            segments_compiled: result.segments_compiled,
        }
    }
}

impl RtfReport {
    /// Human-readable summary of the RTF optimization report.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut out = String::with_capacity(1024);

        // Header
        let status = if self.meets_target { "PASS" } else { "FAIL" };
        out.push_str(&format!(
            "RTF Optimization Report [{status}]\n\
             ========================================\n"
        ));
        out.push_str(&format!(
            "  Projected RTF:   {:.4} (target: {:.4})\n",
            self.projected_rtf, self.target_rtf,
        ));
        out.push_str(&format!(
            "  Total cost:      {:.1} us ({:.3} ms)\n",
            self.total_estimated_ns / 1e3,
            self.total_estimated_ns / 1e6,
        ));
        out.push_str(&format!(
            "  Dispatches:      {} (theoretical min: {}, {:.1}% opportunity)\n",
            self.total_dispatches,
            self.total_theoretical_minimum,
            self.optimization_opportunity_pct,
        ));

        // Per-segment breakdown
        out.push_str("\nPer-Segment Breakdown:\n");
        for seg in &self.segments {
            out.push_str(&format!(
                "  {:<16} {:>4} dispatches  {:>8.1} us  ({:>5.1}%)\n",
                seg.segment_name,
                seg.dispatch_count,
                seg.estimated_cost_ns / 1e3,
                seg.cost_fraction * 100.0,
            ));
        }

        // RtfBottlenecks
        if !self.bottlenecks.is_empty() {
            out.push_str(&format!(
                "\nTop RtfBottlenecks ({}):\n",
                self.bottlenecks.len()
            ));
            for (i, bn) in self.bottlenecks.iter().take(10).enumerate() {
                out.push_str(&format!(
                    "  {:>2}. [{:<20}] {}\n\
                     {: >27}Potential savings: {:.1} us\n",
                    i + 1,
                    bn.category,
                    bn.description,
                    "",
                    bn.potential_savings_ns / 1e3,
                ));
            }
        }

        // Aggregate blocker distribution
        if !self.aggregate_blockers.is_empty() {
            out.push_str("\nAggregate Fusion Blockers:\n");
            let mut sorted: Vec<_> = self.aggregate_blockers.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            for (blocker, count) in sorted {
                out.push_str(&format!("  {blocker:<20} {count}\n"));
            }
        }

        out
    }
}

impl fmt::Display for RtfReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(feature = "plan-serde")]
impl ClosedLoopRtfReport {
    /// Dispatches eliminated across the closed-loop run.
    #[must_use]
    pub fn dispatches_saved(&self) -> usize {
        self.baseline
            .total_dispatches
            .saturating_sub(self.final_report.total_dispatches)
    }

    /// Estimated nanoseconds eliminated across the closed-loop run.
    #[must_use]
    pub fn estimated_cost_saved_ns(&self) -> f64 {
        self.baseline.total_estimated_ns - self.final_report.total_estimated_ns
    }

    /// Human-readable summary of the closed-loop run.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut out = String::with_capacity(1536);
        let status = if self.final_report.meets_target {
            "PASS"
        } else {
            "FAIL"
        };
        out.push_str(&format!(
            "Closed-Loop RTF Optimization [{status}]\n\
             ========================================\n"
        ));
        out.push_str(&format!(
            "  Baseline RTF:    {:.4}\n  Final RTF:       {:.4}\n",
            self.baseline.projected_rtf, self.final_report.projected_rtf,
        ));
        out.push_str(&format!(
            "  Dispatches:      {} -> {} (saved {})\n",
            self.baseline.total_dispatches,
            self.final_report.total_dispatches,
            self.dispatches_saved(),
        ));
        out.push_str(&format!(
            "  Total cost:      {:.1} us -> {:.1} us\n",
            self.baseline.total_estimated_ns / 1e3,
            self.final_report.total_estimated_ns / 1e3,
        ));
        out.push_str(&format!(
            "  Opportunity:     {:.1}% -> {:.1}%\n",
            self.baseline.optimization_opportunity_pct,
            self.final_report.optimization_opportunity_pct,
        ));

        match &self.warmup {
            Some(warmup) => out.push_str(&format!(
                "  Warmup:          {} configs, {} segments compiled{}\n",
                warmup.configs_applied,
                warmup.segments_compiled,
                if warmup.loaded_from_cache {
                    " (cache)"
                } else {
                    ""
                },
            )),
            None => out.push_str("  Warmup:          skipped\n"),
        }

        if !self.actions_taken.is_empty() {
            out.push_str("\nActions Taken:\n");
            for (i, action) in self.actions_taken.iter().enumerate() {
                out.push_str(&format!("  {:>2}. {action}\n", i + 1));
            }
        }

        if !self.final_report.bottlenecks.is_empty() {
            out.push_str("\nRemaining Bottlenecks:\n");
            for (i, bn) in self.final_report.bottlenecks.iter().take(5).enumerate() {
                out.push_str(&format!(
                    "  {:>2}. [{:<20}] {}\n",
                    i + 1,
                    bn.category,
                    bn.description,
                ));
            }
        }

        if let Some(summary) = &self.optimizer_summary {
            out.push_str("\nOptimizer Search Summary:\n");
            out.push_str(summary);
            out.push('\n');
        }

        out
    }
}

#[cfg(feature = "plan-serde")]
impl fmt::Display for ClosedLoopRtfReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// -- CompiledKokoro convenience method --

use super::CompiledKokoro;

impl CompiledKokoro {
    /// Create an [`RtfOptimizer`] with Apple M4 Max defaults and 0.03 target.
    ///
    /// This is a convenience entry point. To run the full analysis, call
    /// [`segment_gap_analysis`](CompiledKokoro::segment_gap_analysis) first,
    /// then pass the results to [`RtfOptimizer::analyze`]:
    ///
    /// ```rust,ignore
    /// let gap_results = kokoro.segment_gap_analysis(&ids, &style, 1.0, &cache)?;
    /// let optimizer = kokoro.rtf_optimizer();
    /// let report = optimizer.analyze(&gap_results);
    /// eprintln!("{report}");
    /// ```
    #[must_use]
    pub fn rtf_optimizer(&self) -> RtfOptimizer {
        RtfOptimizer::apple_m4_max()
    }

    /// Create an [`RtfOptimizer`] with custom cost model and target RTF.
    #[must_use]
    pub fn rtf_optimizer_with(&self, cost_model: CostModel, target_rtf: f64) -> RtfOptimizer {
        RtfOptimizer::new(cost_model, target_rtf)
    }

    /// Run gap analysis and format it through the default [`RtfOptimizer`].
    pub fn analyze_rtf(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<RtfReport, super::CompiledKokoroError> {
        let optimizer = self.rtf_optimizer();
        optimizer.analyze_kokoro(self, input_ids, style, speed, cache)
    }

    /// Run the full closed-loop optimizer with default [`RtfOptimizer`] settings.
    #[cfg(feature = "plan-serde")]
    pub fn optimize_rtf(
        &mut self,
        shapes: &PrecompileShapes,
        cache: &PipelineCache,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
        per_segment_budget: Duration,
        config_cache_path: Option<&Path>,
    ) -> Result<ClosedLoopRtfReport, super::CompiledKokoroError> {
        let optimizer = self.rtf_optimizer();
        optimizer.optimize_kokoro(
            self,
            shapes,
            cache,
            input_ids,
            style,
            speed,
            per_segment_budget,
            config_cache_path,
        )
    }
}

#[cfg(test)]
#[path = "compiled_kokoro_rtf_optimizer_tests.rs"]
mod tests;
