// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Performance profiling infrastructure for the Kokoro streaming pipeline.
//!
//! [`PipelineProfile`] aggregates per-segment timing, dispatch counts, and
//! memory metrics into a single report that identifies the primary bottleneck
//! kind. This is the main tool for diagnosing why RTF is 0.082 instead of
//! the target 0.03, and where dispatch count 201 needs to shrink to reach 60.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_metal::compiled_kokoro::perf_profile::*;
//!
//! let profile = PipelineProfile::from_diagnostics(
//!     &timing, &gpu_timing, &stats, &dispatch_summary, sample_count,
//! );
//! println!("{}", format_profile_report(&profile));
//! println!("Bottleneck: {:?}", identify_bottleneck(&profile));
//! ```
//!
//! Part of #4264.

use std::fmt;
use std::time::Duration;

/// Timing and dispatch profile for a single pipeline segment (step).
///
/// Captures both CPU encoding time (from [`TimingReport`]) and actual GPU
/// execution time (from [`GpuTimingReport`]) so the profiler can distinguish
/// CPU-bound encoding from GPU-bound execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SegmentProfile {
    /// Segment name (e.g., "encode", "prosody", "generate").
    pub name: String,
    /// CPU-side encoding time (dispatch command preparation).
    pub cpu_time: Duration,
    /// GPU-side execution time (includes GPU wait from per-step flush).
    /// `None` if GPU timing was not collected.
    pub gpu_time: Option<Duration>,
    /// Number of logical dispatches in this segment.
    pub dispatch_count: usize,
    /// Number of estimated Metal kernel launches in this segment.
    pub metal_dispatch_count: usize,
}

impl SegmentProfile {
    /// Constructor for building a segment profile.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        cpu_time: Duration,
        gpu_time: Option<Duration>,
        dispatch_count: usize,
        metal_dispatch_count: usize,
    ) -> Self {
        Self {
            name: name.into(),
            cpu_time,
            gpu_time,
            dispatch_count,
            metal_dispatch_count,
        }
    }

    /// GPU/CPU time ratio. Values > 1.0 indicate GPU-bound; < 1.0 indicate CPU-bound.
    /// Returns `None` if GPU time is unavailable.
    #[must_use]
    pub fn gpu_cpu_ratio(&self) -> Option<f64> {
        let gpu = self.gpu_time?;
        let cpu_secs = self.cpu_time.as_secs_f64();
        if cpu_secs < 1e-12 {
            return None;
        }
        Some(gpu.as_secs_f64() / cpu_secs)
    }

    /// Effective time: whichever is longer of CPU encoding and GPU execution.
    /// Falls back to CPU time if GPU timing is unavailable.
    #[must_use]
    pub fn effective_time(&self) -> Duration {
        match self.gpu_time {
            Some(gpu) => self.cpu_time.max(gpu),
            None => self.cpu_time,
        }
    }
}

/// The dominant bottleneck limiting pipeline throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckKind {
    /// GPU execution time dominates. Focus: kernel fusion, dispatch reduction.
    GpuBound,
    /// CPU encoding time dominates. Focus: ICB replay, segment compilation cache.
    CpuBound,
    /// Memory transfer overhead dominates. Focus: arena sizing, blit elimination.
    MemoryBound,
    /// Too many dispatches create per-dispatch overhead. Focus: fusion, NativeOps.
    DispatchBound,
    /// Insufficient data to determine bottleneck.
    Unknown,
}

impl fmt::Display for BottleneckKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuBound => write!(f, "GPU-bound (kernel execution dominates)"),
            Self::CpuBound => write!(f, "CPU-bound (dispatch encoding dominates)"),
            Self::MemoryBound => write!(f, "Memory-bound (transfer overhead dominates)"),
            Self::DispatchBound => write!(f, "Dispatch-bound (too many kernel launches)"),
            Self::Unknown => write!(f, "Unknown (insufficient profiling data)"),
        }
    }
}

/// Full pipeline performance profile aggregating all segments.
///
/// Combines per-segment profiles with pipeline-level metrics (total RTF,
/// dispatch counts, memory stats) to provide a comprehensive performance
/// picture and bottleneck identification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PipelineProfile {
    /// Per-segment timing and dispatch profiles.
    pub segments: Vec<SegmentProfile>,
    /// Total wall-clock time for the entire pipeline.
    pub total_wall_time: Duration,
    /// Total CPU encoding time (sum of segment CPU times).
    pub total_cpu_time: Duration,
    /// Total GPU execution time (sum of segment GPU times).
    /// `None` if GPU timing was not collected.
    pub total_gpu_time: Option<Duration>,
    /// Total logical dispatches across all segments.
    pub total_dispatches: usize,
    /// Total estimated Metal kernel launches.
    pub total_metal_dispatches: usize,
    /// Number of GPU flushes (commit_and_wait calls).
    pub flush_count: usize,
    /// Number of non-blocking GPU submits.
    pub submit_count: usize,
    /// Number of blit copies.
    pub blit_count: usize,
    /// Number of blits eliminated by optimization.
    pub blits_eliminated: usize,
    /// Number of audio samples produced.
    pub sample_count: usize,
    /// Sample rate in Hz (typically 24000 for Kokoro).
    pub sample_rate: u32,
    /// Number of segment cache misses (recompilations).
    pub cache_misses: usize,
}

impl PipelineProfile {
    /// Construct a profile from existing diagnostic data.
    ///
    /// This is the primary constructor. It maps timing report fields to
    /// named segment profiles using the dispatch summary for per-segment
    /// dispatch counts.
    #[must_use]
    pub fn from_timing_and_stats(
        timing: &super::TimingReport,
        gpu_timing: Option<&super::GpuTimingReport>,
        stats: &crate::dispatch_stats::DispatchStats,
        dispatch_summary: &super::DispatchSummary,
        sample_count: usize,
    ) -> Self {
        let segment_names = [
            "encode",
            "prosody",
            "regulate",
            "f0_energy",
            "harmonic",
            "generate",
            "istft",
            "verify",
        ];

        let cpu_times = [
            timing.encode,
            timing.prosody,
            timing.regulate,
            timing.f0_energy,
            timing.harmonic,
            timing.generate,
            timing.istft,
            timing.verify,
        ];

        let gpu_times: Option<[Duration; 8]> = gpu_timing.map(|gt| {
            [
                gt.encode,
                gt.prosody,
                gt.regulate,
                gt.f0_energy,
                gt.harmonic,
                gt.generate,
                gt.istft,
                gt.verify,
            ]
        });

        // Map dispatch summary to segment order. Encode = plbert + text_encoder,
        // harmonic = sinegen_pre + sinegen_post. Verify has 0 dispatches.
        let dispatch_counts = [
            dispatch_summary.plbert + dispatch_summary.text_encoder, // encode
            dispatch_summary.prosody,
            dispatch_summary.regulate,
            dispatch_summary.f0_energy,
            0, // harmonic (eager path, not in compiled segments)
            dispatch_summary.generator,
            0, // istft (eager path)
            0, // verify (CPU-only)
        ];

        let segments: Vec<SegmentProfile> = segment_names
            .iter()
            .enumerate()
            .map(|(i, &name)| {
                SegmentProfile::new(
                    name,
                    cpu_times[i],
                    gpu_times.map(|gt| gt[i]),
                    dispatch_counts[i],
                    dispatch_counts[i], // approximate: 1:1 for now
                )
            })
            .collect();

        let total_cpu_time: Duration = cpu_times.iter().copied().sum();
        let total_gpu_time: Option<Duration> =
            gpu_times.map(|gt| gt.iter().copied().sum());
        let total_dispatches = dispatch_counts.iter().sum();

        Self {
            segments,
            total_wall_time: timing.total,
            total_cpu_time,
            total_gpu_time,
            total_dispatches,
            total_metal_dispatches: stats.compute_encodings,
            flush_count: stats.flushes,
            submit_count: stats.submits,
            blit_count: stats.blits,
            blits_eliminated: stats.blits_eliminated,
            sample_count,
            sample_rate: 24000,
            cache_misses: timing.cache_misses,
        }
    }

    /// Real-time factor: wall time / audio duration.
    ///
    /// RTF < 1.0 means faster than real-time. Target: < 0.03.
    /// Returns `None` if sample count is zero.
    #[must_use]
    pub fn rtf(&self) -> Option<f64> {
        if self.sample_count == 0 || self.sample_rate == 0 {
            return None;
        }
        let audio_duration = self.sample_count as f64 / f64::from(self.sample_rate);
        Some(self.total_wall_time.as_secs_f64() / audio_duration)
    }

    /// Audio duration in seconds for the synthesized output.
    #[must_use]
    pub fn audio_duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.sample_count as f64 / f64::from(self.sample_rate)
    }

    /// Dispatches per millisecond of wall time.
    ///
    /// High values indicate dispatch overhead is significant (many small
    /// kernels). Target: reduce total dispatches, not increase dispatch rate.
    #[must_use]
    pub fn dispatches_per_ms(&self) -> f64 {
        let ms = self.total_wall_time.as_secs_f64() * 1000.0;
        if ms < 1e-12 {
            return 0.0;
        }
        self.total_dispatches as f64 / ms
    }

    /// Segments sorted by effective time (slowest first).
    #[must_use]
    pub fn slowest_segments(&self) -> Vec<&SegmentProfile> {
        let mut sorted: Vec<&SegmentProfile> = self.segments.iter().collect();
        sorted.sort_by(|a, b| {
            b.effective_time()
                .partial_cmp(&a.effective_time())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Segments sorted by dispatch count (most dispatches first).
    #[must_use]
    pub fn most_dispatches(&self) -> Vec<&SegmentProfile> {
        let mut sorted: Vec<&SegmentProfile> = self.segments.iter().collect();
        sorted.sort_by_key(|s| std::cmp::Reverse(s.dispatch_count));
        sorted
    }

    /// Gap between current total dispatches and the target.
    #[must_use]
    pub fn dispatch_gap(&self, target: usize) -> usize {
        self.total_dispatches.saturating_sub(target)
    }

    /// Gap between current RTF and the target.
    /// Returns `None` if RTF cannot be computed.
    #[must_use]
    pub fn rtf_gap(&self, target: f64) -> Option<f64> {
        self.rtf().map(|r| r - target)
    }
}

/// Identify the primary bottleneck from a pipeline profile.
///
/// Heuristic decision tree:
/// 1. If dispatch count > 150, DispatchBound (too many kernel launches).
/// 2. If GPU timing available and total GPU > 2x total CPU, GpuBound.
/// 3. If GPU timing available and total CPU > 2x total GPU, CpuBound.
/// 4. If blits > 10% of total encodings, MemoryBound.
/// 5. Otherwise, Unknown or balanced.
#[must_use]
pub fn identify_bottleneck(profile: &PipelineProfile) -> BottleneckKind {
    // Dispatch-bound: too many kernel launches create per-launch overhead.
    if profile.total_dispatches > 150 {
        return BottleneckKind::DispatchBound;
    }

    // Memory-bound: excessive blit copies indicate transfer overhead.
    let total_encodings = profile.total_metal_dispatches + profile.blit_count;
    if total_encodings > 0 && profile.blit_count as f64 / total_encodings as f64 > 0.10 {
        return BottleneckKind::MemoryBound;
    }

    // GPU vs CPU bound: compare total times when GPU timing is available.
    if let Some(gpu_total) = profile.total_gpu_time {
        let gpu_secs = gpu_total.as_secs_f64();
        let cpu_secs = profile.total_cpu_time.as_secs_f64();
        if cpu_secs > 1e-12 && gpu_secs / cpu_secs > 2.0 {
            return BottleneckKind::GpuBound;
        }
        if gpu_secs > 1e-12 && cpu_secs / gpu_secs > 2.0 {
            return BottleneckKind::CpuBound;
        }
    }

    BottleneckKind::Unknown
}

/// Format a human-readable performance profile report.
///
/// Produces a table showing per-segment timing, dispatch counts, and
/// the identified bottleneck with actionable recommendations.
#[must_use]
pub fn format_profile_report(profile: &PipelineProfile) -> String {
    fn ms(d: Duration) -> f64 {
        d.as_secs_f64() * 1000.0
    }

    let mut lines = Vec::with_capacity(30);

    lines.push("=== Kokoro Pipeline Performance Profile ===".to_string());
    lines.push(String::new());

    // Header
    if let Some(rtf) = profile.rtf() {
        lines.push(format!(
            "RTF: {rtf:.4}  (target: 0.03, gap: {:.4})",
            rtf - 0.03
        ));
    }
    lines.push(format!(
        "Wall time: {:.2} ms  |  Audio: {:.2} s  |  Samples: {}",
        ms(profile.total_wall_time),
        profile.audio_duration_secs(),
        profile.sample_count,
    ));
    lines.push(format!(
        "Dispatches: {} total  (target: 60, gap: {})  |  Metal: {}  |  Blits: {} ({} eliminated)",
        profile.total_dispatches,
        profile.dispatch_gap(60),
        profile.total_metal_dispatches,
        profile.blit_count,
        profile.blits_eliminated,
    ));
    lines.push(format!(
        "Flushes: {}  |  Submits: {}  |  Cache misses: {}",
        profile.flush_count, profile.submit_count, profile.cache_misses,
    ));
    lines.push(String::new());

    // Per-segment table
    lines.push(format!(
        "{:<12} {:>10} {:>10} {:>10} {:>8} {:>10}",
        "Segment", "CPU (ms)", "GPU (ms)", "Eff (ms)", "Disp", "GPU/CPU"
    ));
    lines.push("-".repeat(70));

    for seg in &profile.segments {
        let gpu_str = seg
            .gpu_time
            .map(|g| format!("{:>10.2}", ms(g)))
            .unwrap_or_else(|| format!("{:>10}", "-"));
        let ratio_str = seg
            .gpu_cpu_ratio()
            .map(|r| format!("{r:>10.2}"))
            .unwrap_or_else(|| format!("{:>10}", "-"));
        lines.push(format!(
            "{:<12} {:>10.2} {} {:>10.2} {:>8} {}",
            seg.name,
            ms(seg.cpu_time),
            gpu_str,
            ms(seg.effective_time()),
            seg.dispatch_count,
            ratio_str,
        ));
    }
    lines.push("-".repeat(70));

    // Totals
    let gpu_total_str = profile
        .total_gpu_time
        .map(|g| format!("{:>10.2}", ms(g)))
        .unwrap_or_else(|| format!("{:>10}", "-"));
    lines.push(format!(
        "{:<12} {:>10.2} {} {:>10.2} {:>8}",
        "TOTAL",
        ms(profile.total_cpu_time),
        gpu_total_str,
        ms(profile.total_wall_time),
        profile.total_dispatches,
    ));
    lines.push(String::new());

    // Top bottleneck segments
    lines.push("Top 3 slowest segments:".to_string());
    for (i, seg) in profile.slowest_segments().iter().take(3).enumerate() {
        let pct = if profile.total_wall_time.as_nanos() > 0 {
            seg.effective_time().as_secs_f64() / profile.total_wall_time.as_secs_f64() * 100.0
        } else {
            0.0
        };
        lines.push(format!(
            "  {}. {:<12} {:.2} ms ({:.1}% of total)",
            i + 1,
            seg.name,
            ms(seg.effective_time()),
            pct,
        ));
    }
    lines.push(String::new());

    // Bottleneck identification
    let bottleneck = identify_bottleneck(profile);
    lines.push(format!("Bottleneck: {bottleneck}"));

    // Actionable recommendations
    lines.push(String::new());
    lines.push("Recommendations:".to_string());
    match bottleneck {
        BottleneckKind::GpuBound => {
            lines.push("  - Focus on kernel fusion to reduce GPU execution time".to_string());
            lines.push("  - Profile individual kernels with Metal GPU profiler".to_string());
            lines.push("  - Consider F16 autocast for 2x ALU throughput".to_string());
        }
        BottleneckKind::CpuBound => {
            lines.push("  - Enable ICB replay to skip CPU dispatch encoding".to_string());
            lines.push(
                "  - Ensure segment caches are warm (cache_misses should be 0)".to_string(),
            );
            lines.push(
                "  - Use two-phase pipeline mode for CPU-GPU overlap".to_string(),
            );
        }
        BottleneckKind::MemoryBound => {
            lines.push("  - Reduce blit copies via planned-buffer optimization".to_string());
            lines.push("  - Pre-size arena to avoid growth events".to_string());
            lines.push(
                "  - Use _production step variants to skip standalone blits".to_string(),
            );
        }
        BottleneckKind::DispatchBound => {
            lines.push(format!(
                "  - Reduce dispatches from {} to target 60 (gap: {})",
                profile.total_dispatches,
                profile.dispatch_gap(60),
            ));
            lines.push("  - Focus fusion on heaviest segments:".to_string());
            for seg in profile.most_dispatches().iter().take(3) {
                if seg.dispatch_count > 0 {
                    lines.push(format!(
                        "    * {}: {} dispatches",
                        seg.name, seg.dispatch_count
                    ));
                }
            }
            lines.push("  - Add NativeOp fused variants for common patterns".to_string());
        }
        BottleneckKind::Unknown => {
            lines.push("  - Collect GPU timing (synthesize_with_gpu_timing) for more detail".to_string());
            lines.push("  - Pipeline may be balanced; focus on overall dispatch reduction".to_string());
        }
    }

    lines.join("\n")
}

#[cfg(test)]
#[path = "compiled_kokoro_perf_profile_tests.rs"]
mod tests;
