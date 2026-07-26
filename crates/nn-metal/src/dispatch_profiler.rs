// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal GPU dispatch profiler for fine-grained timing and bandwidth analysis.
//!
//! [`DispatchProfiler`] records individual GPU dispatch entries with nanosecond
//! timestamps, byte counts, and dispatch type classification. It produces a
//! [`DispatchProfileReport`] that identifies the slowest dispatches, computes
//! effective memory bandwidth, and flags consecutive ops that could be fused.
//!
//! Part of #4264 (RTF optimization).

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// DispatchType
// ---------------------------------------------------------------------------

/// Classification of a GPU dispatch for profiling attribution.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DispatchType {
    /// A NativeOp dispatch (fused kernel from the compiled registry).
    NativeOp(String),
    /// A fused kernel dispatch (elementwise chain fusion).
    FusedKernel(String),
    /// A standard DynTensor op dispatch (matmul, conv, softmax, etc.).
    StandardOp(String),
}

impl DispatchType {
    /// Human-readable category name for grouping.
    #[must_use]
    pub fn category(&self) -> &str {
        match self {
            Self::NativeOp(_) => "native_op",
            Self::FusedKernel(_) => "fused_kernel",
            Self::StandardOp(_) => "standard_op",
        }
    }

    /// The inner op name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::NativeOp(n) | Self::FusedKernel(n) | Self::StandardOp(n) => n,
        }
    }
}

impl fmt::Display for DispatchType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NativeOp(n) => write!(f, "NativeOp({n})"),
            Self::FusedKernel(n) => write!(f, "FusedKernel({n})"),
            Self::StandardOp(n) => write!(f, "StandardOp({n})"),
        }
    }
}

// ---------------------------------------------------------------------------
// DispatchProfileEntry
// ---------------------------------------------------------------------------

/// A single recorded GPU dispatch with timing and memory metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchProfileEntry {
    /// Index of this dispatch in the execution plan.
    pub step_idx: usize,
    /// Human-readable op name.
    pub op_name: String,
    /// Classification of the dispatch.
    pub dispatch_type: DispatchType,
    /// Start timestamp in nanoseconds (relative to profiler epoch).
    pub start_ns: u64,
    /// End timestamp in nanoseconds (relative to profiler epoch).
    pub end_ns: u64,
    /// Total input bytes read by this dispatch.
    pub input_bytes: usize,
    /// Total output bytes written by this dispatch.
    pub output_bytes: usize,
}

impl DispatchProfileEntry {
    /// Duration of this dispatch in nanoseconds.
    #[must_use]
    pub fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }

    /// Duration of this dispatch in microseconds.
    #[must_use]
    pub fn duration_us(&self) -> f64 {
        self.duration_ns() as f64 / 1000.0
    }

    /// Total bytes transferred (input + output).
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.input_bytes.saturating_add(self.output_bytes)
    }

    /// Effective bandwidth in GB/s for this dispatch.
    ///
    /// Returns 0.0 if the duration is zero.
    #[must_use]
    pub fn bandwidth_gbps(&self) -> f64 {
        let dur = self.duration_ns();
        if dur == 0 {
            return 0.0;
        }
        self.total_bytes() as f64 / dur as f64
    }
}

// ---------------------------------------------------------------------------
// FusionOpportunity
// ---------------------------------------------------------------------------

/// A pair of consecutive dispatches that may benefit from kernel fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionOpportunity {
    /// Index of the first dispatch in `entries`.
    pub first_idx: usize,
    /// Index of the second dispatch in `entries`.
    pub second_idx: usize,
    /// Name of the first op.
    pub first_op: String,
    /// Name of the second op.
    pub second_op: String,
    /// Combined duration of both dispatches in nanoseconds.
    pub combined_ns: u64,
    /// Estimated bytes saved if intermediate buffer is eliminated.
    pub saved_bytes: usize,
}

// ---------------------------------------------------------------------------
// TypeBreakdown
// ---------------------------------------------------------------------------

/// Per-dispatch-type aggregate timing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeBreakdown {
    /// Total nanoseconds spent in this dispatch type.
    pub total_ns: u64,
    /// Number of dispatches of this type.
    pub count: usize,
    /// Total bytes transferred by this type.
    pub total_bytes: usize,
}

// ---------------------------------------------------------------------------
// DispatchProfileReport
// ---------------------------------------------------------------------------

/// Aggregate report produced by [`DispatchProfiler::report`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchProfileReport {
    /// Total dispatch time in nanoseconds.
    pub total_ns: u64,
    /// Total dispatches recorded.
    pub total_dispatches: usize,
    /// Total bytes transferred across all dispatches.
    pub total_bytes: usize,
    /// Effective aggregate memory bandwidth in GB/s.
    pub bandwidth_gbps: f64,
    /// Breakdown by dispatch type category.
    pub by_type: BTreeMap<String, TypeBreakdown>,
    /// Top 10 slowest dispatches (step_idx, op_name, duration_ns).
    pub top_10: Vec<TopEntry>,
    /// Identified fusion opportunities (consecutive element-wise pairs).
    pub fusion_opportunities: Vec<FusionOpportunity>,
}

/// A top-N entry in the report (serializable summary).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopEntry {
    pub step_idx: usize,
    pub op_name: String,
    pub dispatch_type: String,
    pub duration_ns: u64,
    pub total_bytes: usize,
    pub bandwidth_gbps: f64,
}

impl DispatchProfileReport {
    /// Serialize to a JSON string.
    ///
    /// Returns `Err` if serialization fails (should not happen for this type).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl fmt::Display for DispatchProfileReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Dispatch Profile Report ===")?;
        writeln!(
            f,
            "Total: {:.3} ms ({} dispatches, {} transferred, {:.1} GB/s)",
            self.total_ns as f64 / 1_000_000.0,
            self.total_dispatches,
            format_bytes(self.total_bytes),
            self.bandwidth_gbps,
        )?;
        writeln!(f)?;

        writeln!(f, "By dispatch type:")?;
        for (cat, bd) in &self.by_type {
            let pct = if self.total_ns > 0 {
                bd.total_ns as f64 / self.total_ns as f64 * 100.0
            } else {
                0.0
            };
            writeln!(
                f,
                "  {:<16} {:>8.1} us  {:>3} dispatches  {:>8}  ({:>5.1}%)",
                cat,
                bd.total_ns as f64 / 1000.0,
                bd.count,
                format_bytes(bd.total_bytes),
                pct,
            )?;
        }

        writeln!(f)?;
        writeln!(f, "Top 10 slowest dispatches:")?;
        for (rank, entry) in self.top_10.iter().enumerate() {
            writeln!(
                f,
                "  {:>2}. [{:>3}] {:>8.1} us  {:>8}  {:>6.1} GB/s  {} ({})",
                rank + 1,
                entry.step_idx,
                entry.duration_ns as f64 / 1000.0,
                format_bytes(entry.total_bytes),
                entry.bandwidth_gbps,
                entry.op_name,
                entry.dispatch_type,
            )?;
        }

        if !self.fusion_opportunities.is_empty() {
            writeln!(f)?;
            writeln!(
                f,
                "Fusion opportunities ({}):",
                self.fusion_opportunities.len()
            )?;
            for opp in &self.fusion_opportunities {
                writeln!(
                    f,
                    "  [{},{}] {} + {} = {:.1} us, saves {}",
                    opp.first_idx,
                    opp.second_idx,
                    opp.first_op,
                    opp.second_op,
                    opp.combined_ns as f64 / 1000.0,
                    format_bytes(opp.saved_bytes),
                )?;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DispatchProfiler
// ---------------------------------------------------------------------------

/// Records GPU dispatch entries and produces analysis reports.
///
/// Create with [`DispatchProfiler::new()`], which starts disabled. Call
/// [`.enable()`] to start recording, then [`.record()`] for each dispatch.
/// When done, call [`.report()`] for the full analysis or [`.top_n()`] for
/// quick inspection of the slowest dispatches.
///
/// # Example
///
/// ```rust
/// use nn_metal::dispatch_profiler::{DispatchProfiler, DispatchProfileEntry, DispatchType};
///
/// let mut profiler = DispatchProfiler::new();
/// profiler.enable();
/// profiler.record(DispatchProfileEntry {
///     step_idx: 0,
///     op_name: "matmul".into(),
///     dispatch_type: DispatchType::StandardOp("matmul".into()),
///     start_ns: 0,
///     end_ns: 50_000,
///     input_bytes: 4096,
///     output_bytes: 2048,
/// });
/// let report = profiler.report();
/// assert_eq!(report.total_dispatches, 1);
/// ```
pub struct DispatchProfiler {
    entries: Vec<DispatchProfileEntry>,
    enabled: bool,
}

impl DispatchProfiler {
    /// Create a new profiler, disabled by default.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            enabled: false,
        }
    }

    /// Enable dispatch recording.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable dispatch recording.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Whether the profiler is currently recording.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record a dispatch entry. No-op if the profiler is disabled.
    pub fn record(&mut self, entry: DispatchProfileEntry) {
        if self.enabled {
            self.entries.push(entry);
        }
    }

    /// Clear all recorded entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of recorded entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no entries have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrow all recorded entries.
    #[must_use]
    pub fn entries(&self) -> &[DispatchProfileEntry] {
        &self.entries
    }

    /// Total dispatch time in nanoseconds across all entries.
    #[must_use]
    pub fn total_dispatch_ns(&self) -> u64 {
        self.entries.iter().map(DispatchProfileEntry::duration_ns).sum()
    }

    /// Total memory bytes transferred (input + output) across all entries.
    #[must_use]
    pub fn total_memory_bytes(&self) -> usize {
        self.entries.iter().map(DispatchProfileEntry::total_bytes).sum()
    }

    /// Effective aggregate memory bandwidth in GB/s.
    ///
    /// Computed as `total_bytes / total_ns`. Returns 0.0 if total time is zero.
    #[must_use]
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        let total_ns = self.total_dispatch_ns();
        if total_ns == 0 {
            return 0.0;
        }
        self.total_memory_bytes() as f64 / total_ns as f64
    }

    /// Return the N slowest dispatches, sorted by duration descending.
    #[must_use]
    pub fn top_n(&self, n: usize) -> Vec<&DispatchProfileEntry> {
        let mut sorted: Vec<&DispatchProfileEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| {
            b.duration_ns()
                .cmp(&a.duration_ns())
        });
        sorted.truncate(n);
        sorted
    }

    /// Produce a full analysis report.
    #[must_use]
    pub fn report(&self) -> DispatchProfileReport {
        let total_ns = self.total_dispatch_ns();
        let total_bytes = self.total_memory_bytes();
        let bandwidth_gbps = self.memory_bandwidth_gbps();

        // By-type breakdown
        let mut by_type: BTreeMap<String, TypeBreakdown> = BTreeMap::new();
        for entry in &self.entries {
            let cat = entry.dispatch_type.category().to_string();
            let bd = by_type.entry(cat).or_default();
            bd.total_ns += entry.duration_ns();
            bd.count += 1;
            bd.total_bytes += entry.total_bytes();
        }

        // Top 10
        let top_entries = self.top_n(10);
        let top_10: Vec<TopEntry> = top_entries
            .iter()
            .map(|e| TopEntry {
                step_idx: e.step_idx,
                op_name: e.op_name.clone(),
                dispatch_type: e.dispatch_type.category().to_string(),
                duration_ns: e.duration_ns(),
                total_bytes: e.total_bytes(),
                bandwidth_gbps: e.bandwidth_gbps(),
            })
            .collect();

        // Fusion opportunities: consecutive element-wise ops
        let fusion_opportunities = self.find_fusion_opportunities();

        DispatchProfileReport {
            total_ns,
            total_dispatches: self.entries.len(),
            total_bytes,
            bandwidth_gbps,
            by_type,
            top_10,
            fusion_opportunities,
        }
    }

    /// Identify consecutive dispatches that could potentially be fused.
    ///
    /// Heuristic: two consecutive dispatches are a fusion opportunity if they
    /// are both element-wise (same output size) and neither is a matmul/conv.
    fn find_fusion_opportunities(&self) -> Vec<FusionOpportunity> {
        let mut opportunities = Vec::new();
        if self.entries.len() < 2 {
            return opportunities;
        }

        for i in 0..self.entries.len() - 1 {
            let a = &self.entries[i];
            let b = &self.entries[i + 1];

            // Heuristic: both must be element-wise-sized (output == input of next)
            // and output sizes must match.
            if a.output_bytes > 0
                && a.output_bytes == b.input_bytes
                && is_fusable_name(&a.op_name)
                && is_fusable_name(&b.op_name)
            {
                opportunities.push(FusionOpportunity {
                    first_idx: i,
                    second_idx: i + 1,
                    first_op: a.op_name.clone(),
                    second_op: b.op_name.clone(),
                    combined_ns: a.duration_ns().saturating_add(b.duration_ns()),
                    saved_bytes: a.output_bytes, // intermediate buffer eliminated
                });
            }
        }

        opportunities
    }
}

impl Default for DispatchProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for DispatchProfiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DispatchProfiler")
            .field("enabled", &self.enabled)
            .field("entries", &self.entries.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if an op name represents a potentially fusable element-wise operation.
///
/// Excludes reduction/matmul/conv ops that contain fusable substrings
/// (e.g., "matmul" contains "mul" but is not element-wise).
fn is_fusable_name(name: &str) -> bool {
    let lower = name.to_lowercase();

    // Exclude known non-element-wise ops that contain fusable substrings
    if lower.contains("matmul")
        || lower.contains("gemm")
        || lower.contains("conv")
        || lower.contains("lstm")
        || lower.contains("attention")
        || lower.contains("sdpa")
        || lower.contains("softmax")
        || lower.contains("norm")
        || lower.contains("embedding")
        || lower.contains("gather")
        || lower.contains("reduce")
    {
        return false;
    }

    // Element-wise activations and arithmetic are fusable
    lower.contains("snake")
        || lower.contains("relu")
        || lower.contains("gelu")
        || lower.contains("silu")
        || lower.contains("sigmoid")
        || lower.contains("tanh")
        || lower.contains("add")
        || lower.contains("sub")
        || lower.contains("mul")
        || lower.contains("div")
        || lower.contains("elementwise")
        || lower.contains("clamp")
        || lower.contains("exp")
        || lower.contains("neg")
        || lower.contains("abs")
        || lower.contains("pow")
        || lower.contains("sqrt")
        || lower.contains("log")
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
#[path = "dispatch_profiler_tests.rs"]
mod tests;
