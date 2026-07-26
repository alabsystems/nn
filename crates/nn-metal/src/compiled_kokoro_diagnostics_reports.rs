// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Optimization and performance report generation for [`CompiledKokoro`].
//!
//! Extracted from `compiled_kokoro_diagnostics.rs` for 450-line compliance.
//!
//! Part of #2218, #1815.

use crate::compiled_kokoro::CompiledKokoro;
use crate::dispatch_stats::DispatchStats;

impl CompiledKokoro {
    /// Build a structured [`OptimizationReport`] for the full Kokoro pipeline.
    ///
    /// Generates dispatch-proportion recommendations automatically. If bounds
    /// data is attached (via [`OptimizationReport::with_bounds`]), also
    /// surfaces `NormChainExplosion` warnings from the bound analysis (#2708).
    pub fn optimization_report(
        &self,
        iteration: usize,
    ) -> Result<crate::OptimizationReport, crate::ReportError> {
        let perf = self.performance_report();
        let mut report = crate::OptimizationReport::new(iteration, "kokoro", &perf)?;
        report.generate_dispatch_recommendations();
        report.generate_flush_recommendations();
        report.generate_bounds_recommendations();
        Ok(report)
    }

    /// Build an [`OptimizationReport`] with GPU sync point statistics.
    ///
    /// Like [`optimization_report`](Self::optimization_report) but includes
    /// flush/submit counts from a prior `synthesize_with_stats()` call.
    /// Enables flush budget regression detection in the progressive
    /// tightening loop (#2739).
    pub fn optimization_report_with_stats(
        &self,
        iteration: usize,
        stats: &DispatchStats,
    ) -> Result<crate::OptimizationReport, crate::ReportError> {
        let perf = self.performance_report_with_stats(stats);
        let mut report = crate::OptimizationReport::new(iteration, "kokoro", &perf)?;
        report.generate_dispatch_recommendations();
        report.generate_flush_recommendations();
        report.generate_bounds_recommendations();
        Ok(report)
    }

    /// Build a [`PerformanceReport`] with GPU sync point statistics.
    ///
    /// Enriches the base report with flush/submit counts and actual Metal
    /// dispatch count from a prior `synthesize_with_stats()` call. Enables
    /// regression tracking for #2739 flush budget and #1815 dispatch count.
    #[must_use]
    pub fn performance_report_with_stats(
        &self,
        stats: &DispatchStats,
    ) -> nn_dsl::PerformanceReport {
        self.performance_report().with_gpu_sync_stats(
            stats.flushes,
            stats.submits,
            stats.compute_encodings,
        )
    }

    /// Per-segment dispatch breakdowns for optimization diagnostics (#2780).
    ///
    /// Returns one `(name, ir_counts, native_counts)` triple per compiled
    /// segment. Each element is the output of `CompiledModel::dispatch_breakdown()`
    /// — IR kernel name→count and NativeOp variant name→count.
    ///
    /// Segments not yet compiled are omitted.
    #[must_use]
    pub fn dispatch_breakdowns(&self) -> Vec<(&str, Vec<(String, usize)>, Vec<(String, usize)>)> {
        let segments = [
            ("plbert", &self.seg_plbert),
            ("text", &self.seg_text),
            ("prosody", &self.seg_prosody),
            ("f0_energy", &self.seg_f0),
            ("generator", &self.seg_generator),
            ("regulate", &self.seg_regulate),
            ("sinegen_pre", &self.seg_sinegen_pre),
            ("sinegen_post", &self.seg_sinegen_post),
        ];
        segments
            .into_iter()
            .filter_map(|(name, cache)| {
                cache.most_recent().map(|(_, model)| {
                    let (ir, native) = model.dispatch_breakdown();
                    (name, ir, native)
                })
            })
            .collect()
    }

    /// Build a structured [`PerformanceReport`] for the full Kokoro pipeline.
    #[must_use]
    pub fn performance_report(&self) -> nn_dsl::PerformanceReport {
        let segment_names = [
            ("plbert_encoder", &self.seg_plbert),
            ("text_pipeline", &self.seg_text),
            ("prosody_predictor", &self.seg_prosody),
            ("f0_energy_predictor", &self.seg_f0),
            ("generator", &self.seg_generator),
            ("regulate", &self.seg_regulate),
            ("sinegen_pre", &self.seg_sinegen_pre),
            ("sinegen_post", &self.seg_sinegen_post),
        ];
        let segments: Vec<nn_dsl::SegmentPerformance> = segment_names
            .iter()
            .filter_map(|(name, cache)| {
                cache
                    .most_recent()
                    .map(|(_, model)| model.segment_performance(name))
            })
            .collect();
        nn_dsl::PerformanceReport::from_segments("kokoro", segments)
    }
}
