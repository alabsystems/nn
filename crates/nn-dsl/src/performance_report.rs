// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structured performance report for compiled model optimization.
//!
//! [`PerformanceReport`] captures dispatch counts, memory metrics, and
//! per-segment breakdown for a compiled model. This is the "performance"
//! section of the optimization report consumed by the progressive
//! tightening loop (Phase 6).
//!
//! Each segment corresponds to a sub-pipeline (e.g., text encoder,
//! prosody predictor) with its own dispatch plan and buffer allocation.

use serde::{Deserialize, Serialize};

/// Performance metrics for a compiled model or pipeline.
///
/// Captures the complete dispatch and memory profile in a structured
/// format suitable for JSON serialization. An LLM or human reads this
/// to decide the next optimization action.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PerformanceReport {
    /// Name of the model (e.g., "kokoro", "whisper").
    pub model_name: String,
    /// Per-segment performance breakdown.
    pub segments: Vec<SegmentPerformance>,
    /// Total GPU dispatch steps (IR + NativeOp) across all segments.
    pub total_dispatches: usize,
    /// Estimated Metal kernel launches after plan expansion (compiled segments only).
    ///
    /// This is a planner estimate — does not include eager paths.
    /// For actual runtime count, see [`actual_metal_dispatches`](Self::actual_metal_dispatches).
    pub total_metal_dispatches: usize,
    /// Total compiled steps across all segments.
    pub total_steps: usize,
    /// Total native (pre-compiled) ops across all segments.
    pub total_native_ops: usize,
    /// Aggregate memory metrics.
    pub memory: MemoryMetrics,
    /// Total `commit_and_wait` calls (GPU flushes) during synthesis.
    ///
    /// 0 means not measured (default from `from_segments()`).
    /// Use [`with_gpu_sync_stats`](Self::with_gpu_sync_stats) to set.
    /// Part of #2739.
    pub total_flushes: usize,
    /// Total non-blocking GPU submit calls during synthesis.
    ///
    /// 0 means not measured (default from `from_segments()`).
    /// Use [`with_gpu_sync_stats`](Self::with_gpu_sync_stats) to set.
    /// Part of #2739.
    pub total_submits: usize,
    /// Actual Metal dispatch encodings measured at runtime (including eager paths).
    ///
    /// Unlike [`total_metal_dispatches`](Self::total_metal_dispatches) which is a
    /// planner estimate for compiled segments only, this counts every GPU dispatch
    /// encoding during synthesis — including eager paths (SineGen, step_regulate,
    /// iSTFT, embeddings, to_device transfers).
    ///
    /// 0 means not measured (default from `from_segments()`).
    /// Use [`with_gpu_sync_stats`](Self::with_gpu_sync_stats) to set.
    /// Part of #1815.
    #[serde(default)]
    pub actual_metal_dispatches: usize,
    /// ISO 8601 timestamp of when the report was generated.
    pub generated_at: String,
}

/// Performance metrics for a single compiled segment/sub-pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SegmentPerformance {
    /// Segment name (e.g., "text_pipeline", "f0_energy_predictor").
    pub name: String,
    /// Number of GPU dispatch steps (IR + NativeOp).
    pub dispatches: usize,
    /// Number of Metal kernel launches after plan expansion.
    pub metal_dispatches: usize,
    /// Total compiled steps in this segment.
    pub steps: usize,
    /// Number of native (pre-compiled) ops.
    pub native_ops: usize,
    /// Number of IR-generated dispatch steps.
    pub ir_dispatches: usize,
    /// Total buffer bytes after reuse optimization.
    pub buffer_bytes: usize,
    /// Total buffer bytes without reuse (naive allocation).
    pub buffer_naive_bytes: usize,
    /// Measured latency in microseconds. `None` if not benchmarked.
    pub latency_us: Option<f64>,
}

impl SegmentPerformance {
    /// Create a new segment performance entry without latency.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dispatches: 0,
            metal_dispatches: 0,
            steps: 0,
            native_ops: 0,
            ir_dispatches: 0,
            buffer_bytes: 0,
            buffer_naive_bytes: 0,
            latency_us: None,
        }
    }

    /// Set measured latency.
    #[must_use]
    pub fn with_latency(mut self, us: f64) -> Self {
        self.latency_us = Some(us);
        self
    }

    /// Buffer reuse ratio: optimized / naive. Lower is better.
    ///
    /// Returns 1.0 if naive bytes is zero (no buffers allocated).
    #[must_use]
    pub fn buffer_reuse_ratio(&self) -> f32 {
        if self.buffer_naive_bytes == 0 {
            return 1.0;
        }
        self.buffer_bytes as f32 / self.buffer_naive_bytes as f32
    }
}

/// Aggregate memory metrics for a compiled model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MemoryMetrics {
    /// Total buffer bytes after reuse optimization.
    pub total_buffer_bytes: usize,
    /// Total buffer bytes without reuse (naive allocation).
    pub naive_buffer_bytes: usize,
    /// Reuse ratio: total / naive. Lower is better (more reuse).
    pub reuse_ratio: f32,
}

impl MemoryMetrics {
    /// Create from totals, computing the ratio.
    #[must_use]
    pub fn new(total_buffer_bytes: usize, naive_buffer_bytes: usize) -> Self {
        let reuse_ratio = if naive_buffer_bytes == 0 {
            1.0
        } else {
            total_buffer_bytes as f32 / naive_buffer_bytes as f32
        };
        Self {
            total_buffer_bytes,
            naive_buffer_bytes,
            reuse_ratio,
        }
    }
}

impl PerformanceReport {
    /// Create a report from a list of segment performances.
    #[must_use]
    pub fn from_segments(model_name: impl Into<String>, segments: Vec<SegmentPerformance>) -> Self {
        let total_dispatches = segments.iter().map(|s| s.dispatches).sum();
        let total_metal_dispatches = segments.iter().map(|s| s.metal_dispatches).sum();
        let total_steps = segments.iter().map(|s| s.steps).sum();
        let total_native_ops = segments.iter().map(|s| s.native_ops).sum();
        let total_buffer_bytes: usize = segments.iter().map(|s| s.buffer_bytes).sum();
        let naive_buffer_bytes: usize = segments.iter().map(|s| s.buffer_naive_bytes).sum();
        let memory = MemoryMetrics::new(total_buffer_bytes, naive_buffer_bytes);

        Self {
            model_name: model_name.into(),
            segments,
            total_dispatches,
            total_metal_dispatches,
            total_steps,
            total_native_ops,
            memory,
            total_flushes: 0,
            total_submits: 0,
            actual_metal_dispatches: 0,
            generated_at: now_iso8601(),
        }
    }

    /// Set GPU sync point statistics (flushes, submits, actual dispatch encodings).
    ///
    /// These are measured by `dispatch_stats::reset_counters()` / `dispatch_stats()`
    /// around a synthesis call. Not available from segment-level data alone.
    ///
    /// `compute_encodings` is compute-only Metal dispatch count (excludes blits) —
    /// use `DispatchStats::compute_encodings` from `synthesize_with_stats()`.
    ///
    /// Part of #2739, #1815.
    #[must_use]
    pub fn with_gpu_sync_stats(mut self, flushes: usize, submits: usize, encodings: usize) -> Self {
        self.total_flushes = flushes;
        self.total_submits = submits;
        self.actual_metal_dispatches = encodings;
        self
    }

    /// Serialize to pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns `Err` if serialization fails (should not happen for valid data).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Current UTC timestamp in ISO 8601 format.
///
/// Uses a simple seconds-since-epoch approach to avoid pulling in `chrono`.
fn now_iso8601() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Format as "2026-03-18T12:00:00Z" (approximate — no full calendar math).
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    // Approximate year/month/day from days since epoch (good enough for timestamps).
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Civil calendar algorithm (Howard Hinnant).
    days += 719_468;
    let era = days / 146_097;
    let doe = days - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_reuse_ratio_no_naive() {
        let seg = SegmentPerformance::new("test");
        assert_eq!(seg.buffer_reuse_ratio(), 1.0);
    }

    #[test]
    fn test_segment_reuse_ratio() {
        let seg = SegmentPerformance {
            buffer_bytes: 500,
            buffer_naive_bytes: 1000,
            ..SegmentPerformance::new("test")
        };
        assert!((seg.buffer_reuse_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_memory_metrics() {
        let m = MemoryMetrics::new(500, 1000);
        assert!((m.reuse_ratio - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_from_segments() {
        let segs = vec![
            SegmentPerformance {
                dispatches: 10,
                metal_dispatches: 50,
                steps: 15,
                native_ops: 2,
                ir_dispatches: 8,
                buffer_bytes: 1000,
                buffer_naive_bytes: 2000,
                ..SegmentPerformance::new("seg_a")
            },
            SegmentPerformance {
                dispatches: 5,
                metal_dispatches: 20,
                steps: 8,
                native_ops: 1,
                ir_dispatches: 4,
                buffer_bytes: 500,
                buffer_naive_bytes: 800,
                ..SegmentPerformance::new("seg_b")
            },
        ];
        let report = PerformanceReport::from_segments("test_model", segs);
        assert_eq!(report.total_dispatches, 15);
        assert_eq!(report.total_metal_dispatches, 70);
        assert_eq!(report.total_steps, 23);
        assert_eq!(report.total_native_ops, 3);
        assert_eq!(report.memory.total_buffer_bytes, 1500);
        assert_eq!(report.memory.naive_buffer_bytes, 2800);
    }

    #[test]
    fn test_json_roundtrip() {
        let report = PerformanceReport::from_segments("test", vec![]);
        let json = report.to_json().expect("serialize");
        let parsed: PerformanceReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.model_name, "test");
        assert_eq!(parsed.total_dispatches, 0);
    }

    #[test]
    fn test_actual_metal_dispatches_default_zero() {
        let report = PerformanceReport::from_segments("test", vec![]);
        assert_eq!(report.actual_metal_dispatches, 0);
    }

    #[test]
    fn test_actual_metal_dispatches_set_via_builder() {
        let report =
            PerformanceReport::from_segments("test", vec![]).with_gpu_sync_stats(3, 0, 424);
        assert_eq!(report.actual_metal_dispatches, 424);
        assert_eq!(report.total_flushes, 3);
        assert_eq!(report.total_submits, 0);
    }

    #[test]
    fn test_actual_metal_dispatches_json_roundtrip() {
        let report =
            PerformanceReport::from_segments("test", vec![]).with_gpu_sync_stats(2, 1, 234);
        let json = report.to_json().expect("serialize");
        let parsed: PerformanceReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.actual_metal_dispatches, 234);
    }

    #[test]
    fn test_actual_metal_dispatches_missing_in_json_defaults_zero() {
        // Simulate old JSON without the field.
        let json = r#"{"model_name":"old","segments":[],"total_dispatches":0,
            "total_metal_dispatches":0,"total_steps":0,"total_native_ops":0,
            "memory":{"total_buffer_bytes":0,"naive_buffer_bytes":0,"reuse_ratio":1.0},
            "total_flushes":0,"total_submits":0,"generated_at":"2026-01-01T00:00:00Z"}"#;
        let parsed: PerformanceReport = serde_json::from_str(json).expect("deserialize old");
        assert_eq!(parsed.actual_metal_dispatches, 0);
    }

    #[test]
    fn test_timestamp_format() {
        let ts = now_iso8601();
        assert!(
            ts.starts_with("20"),
            "timestamp should start with year: {ts}"
        );
        assert!(ts.ends_with('Z'), "timestamp should end with Z: {ts}");
        assert!(ts.contains('T'), "timestamp should contain T: {ts}");
    }
}
