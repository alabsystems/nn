// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d dispatch batching optimizer for Kokoro segment fusion.
//!
//! Analyzes consecutive `Conv1dGemm` operations in a segment's compiled
//! dispatch plan and groups compatible conv1ds (same kernel size, stride,
//! dilation, groups) into batches. For each batch of N compatible conv1ds,
//! the im2col + GEMM dispatches can be fused into fewer Metal kernel
//! launches by concatenating the weight matrices and sharing the im2col
//! output.
//!
//! Each individual `Conv1dGemm` with `groups == 1` emits 2-3 Metal
//! dispatches: im2col + simdgroup GEMM + optional bias add. When N
//! consecutive conv1ds share the same `(kernel_size, stride, dilation,
//! groups, input_channels)`, the im2col step is identical across all N
//! and only needs to run once. The N separate GEMMs can be batched into
//! a single larger GEMM with concatenated output channels.
//!
//! **Dispatch savings per batch of N:**
//! - im2col: N dispatches → 1 dispatch (shared unfolding)
//! - GEMM: N dispatches → 1 dispatch (concatenated weight matrix)
//! - bias: N dispatches → 1 dispatch (concatenated bias vector)
//! - Total: N*(2 or 3) → (2 or 3) = saves (N-1)*(2 or 3) dispatches
//!
//! This is a static analysis pass — it does not modify the compiled plan
//! but identifies batching opportunities and computes the potential
//! dispatch reduction. The actual batched execution is a future step
//! that requires coordinating weight concatenation at build time.
//!
//! Part of #4264.

use std::fmt;

use nn_dsl::trace_compile::CompiledStep;
use nn_dsl::NativeOpKind;

/// Describes a group of consecutive Conv1dGemm operations that share
/// compatible convolution parameters and can be batched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv1dBatchGroup {
    /// Step indices in the compiled plan for the conv1ds in this batch.
    pub step_indices: Vec<usize>,
    /// Shared kernel size across all conv1ds in the batch.
    pub kernel_size: usize,
    /// Shared stride across all conv1ds in the batch.
    pub stride: usize,
    /// Shared dilation factor across all conv1ds in the batch.
    pub dilation: usize,
    /// Shared group count across all conv1ds in the batch.
    pub groups: usize,
    /// Shared input channels (C_in) across all conv1ds in the batch.
    pub input_channels: usize,
    /// Per-conv output channel counts (C_out for each conv in the batch).
    pub output_channels: Vec<usize>,
    /// Whether all conv1ds in the batch have bias.
    pub all_have_bias: bool,
    /// Whether any conv1d in the batch has bias.
    pub any_have_bias: bool,
    /// Total output channels across the batch (sum of output_channels).
    pub total_output_channels: usize,
}

impl Conv1dBatchGroup {
    /// Number of conv1d operations in this batch.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.step_indices.len()
    }

    /// Whether this batch uses the K=3 direct sliding-window path.
    ///
    /// K=3, stride=1, dilation=1, groups=1 → direct path (1 conv dispatch).
    /// Other shapes → im2col + GEMM (2 dispatches for the matmul).
    #[must_use]
    fn uses_direct_k3(&self) -> bool {
        self.kernel_size == 3 && self.stride == 1 && self.dilation == 1 && self.groups == 1
    }

    /// Base dispatches per conv1d (excluding bias).
    ///
    /// Direct K=3 path: 1 dispatch. Im2col + GEMM path: 2 dispatches.
    #[must_use]
    fn base_dispatches_per_conv(&self) -> usize {
        if self.uses_direct_k3() { 1 } else { 2 }
    }

    /// Metal dispatches if each conv1d runs independently.
    ///
    /// Direct K=3 path: 1 conv + optional bias = 1 or 2 per conv.
    /// Im2col path: im2col (1) + GEMM (1) + optional bias = 2 or 3 per conv.
    #[must_use]
    pub fn unbatched_dispatches(&self) -> usize {
        let base = self.base_dispatches_per_conv();
        self.step_indices.len() * base
            + if self.any_have_bias { self.step_indices.len() } else { 0 }
    }

    /// Metal dispatches with batched execution.
    ///
    /// Direct K=3 batch: 1 batched conv + optional bias = 1 or 2.
    /// Im2col batch: shared im2col (1) + batched GEMM (1) + optional bias = 2 or 3.
    #[must_use]
    pub fn batched_dispatches(&self) -> usize {
        let base = self.base_dispatches_per_conv();
        base + if self.any_have_bias { 1 } else { 0 }
    }

    /// Dispatch count reduction from batching.
    #[must_use]
    pub fn dispatches_saved(&self) -> usize {
        self.unbatched_dispatches()
            .saturating_sub(self.batched_dispatches())
    }
}

/// Convolution parameters used as a grouping key for batch compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConvBatchKey {
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    input_channels: usize,
    input_length: usize,
    batch_size: usize,
}

/// A candidate Conv1dGemm extracted from a compiled step.
struct Conv1dCandidate {
    step_idx: usize,
    key: ConvBatchKey,
    out_channels: usize,
    has_bias: bool,
}

/// Extract Conv1dGemm parameters from a compiled step, if applicable.
fn extract_conv1d_candidate(step_idx: usize, step: &CompiledStep) -> Option<Conv1dCandidate> {
    match step {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::Conv1dGemm {
                    input_shape,
                    out_channels,
                    kernel_size,
                    stride,
                    padding: _,
                    dilation,
                    groups,
                    has_bias,
                },
            ..
        } => {
            let batch_size = input_shape.first().copied().unwrap_or(1);
            let input_channels = input_shape.get(1).copied().unwrap_or(1);
            let input_length = input_shape.get(2).copied().unwrap_or(0);
            Some(Conv1dCandidate {
                step_idx,
                key: ConvBatchKey {
                    kernel_size: *kernel_size,
                    stride: *stride,
                    dilation: *dilation,
                    groups: *groups,
                    input_channels,
                    input_length,
                    batch_size,
                },
                out_channels: *out_channels,
                has_bias: *has_bias,
            })
        }
        _ => None,
    }
}

/// Result of analyzing a segment's dispatch plan for conv1d batching.
#[derive(Debug, Clone)]
pub struct ConvBatchAnalysis {
    /// Segment name for diagnostic reporting.
    pub segment_name: String,
    /// Identified batch groups of compatible conv1ds.
    pub groups: Vec<Conv1dBatchGroup>,
    /// Total dispatch count in the segment (all step types).
    pub total_dispatches: usize,
    /// Total dispatches from Conv1dGemm steps specifically.
    pub conv1d_dispatches: usize,
    /// Dispatches saved if all identified batches were applied.
    pub total_saved: usize,
    /// Optimized dispatch count after batching.
    pub optimized_dispatches: usize,
}

impl ConvBatchAnalysis {
    /// Whether any batching opportunities were found.
    #[must_use]
    pub fn has_opportunities(&self) -> bool {
        !self.groups.is_empty()
    }

    /// Percentage reduction in Conv1d Metal dispatch count.
    ///
    /// Expressed as `total_saved / conv1d_dispatches * 100`. Uses Conv1d
    /// Metal dispatches as the denominator because savings are measured
    /// in Metal kernel launches, not compiled step count.
    #[must_use]
    pub fn reduction_pct(&self) -> f64 {
        if self.conv1d_dispatches == 0 {
            return 0.0;
        }
        (self.total_saved as f64 / self.conv1d_dispatches as f64) * 100.0
    }
}

impl fmt::Display for ConvBatchAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "ConvBatchAnalysis [{}]: {} dispatches -> {} ({} saved, {:.1}%)",
            self.segment_name,
            self.total_dispatches,
            self.optimized_dispatches,
            self.total_saved,
            self.reduction_pct(),
        )?;
        for (i, group) in self.groups.iter().enumerate() {
            writeln!(
                f,
                "  batch {}: {} conv1ds, K={}, S={}, C_in={}, C_out={:?}, saves {}",
                i,
                group.batch_size(),
                group.kernel_size,
                group.stride,
                group.input_channels,
                group.output_channels,
                group.dispatches_saved(),
            )?;
        }
        Ok(())
    }
}

/// Optimizer that identifies conv1d dispatch batching opportunities.
///
/// Scans a compiled step sequence for consecutive `Conv1dGemm` operations
/// with compatible parameters (same kernel size, stride, dilation, groups,
/// and input shape). Groups of 2+ compatible conv1ds are reported as
/// batching candidates.
///
/// # Compatibility criteria
///
/// Two `Conv1dGemm` steps are compatible if they share:
/// - `kernel_size`, `stride`, `dilation`, `groups`
/// - Input shape `[B, C_in, L_in]` (same batch, channels, and length)
///
/// `out_channels` and `has_bias` may differ — the batched GEMM concatenates
/// weight matrices along the output channel dimension.
///
/// # Consecutive requirement
///
/// Only conv1ds that are adjacent in the compiled step sequence (possibly
/// separated by zero-cost steps like Passthrough, NarrowView, etc.) are
/// considered. Non-adjacent compatible conv1ds cannot be batched because
/// intermediate steps may depend on the conv1d output.
pub struct ConvBatchOptimizer {
    /// Minimum batch size to report (default: 2).
    min_batch_size: usize,
}

impl Default for ConvBatchOptimizer {
    fn default() -> Self {
        Self { min_batch_size: 2 }
    }
}

impl ConvBatchOptimizer {
    /// Create an optimizer with a custom minimum batch size.
    #[must_use]
    pub fn with_min_batch_size(min_batch_size: usize) -> Self {
        Self {
            min_batch_size: min_batch_size.max(2),
        }
    }

    /// Analyze a compiled step sequence for conv1d batching opportunities.
    ///
    /// Returns a [`ConvBatchAnalysis`] with identified batch groups and
    /// dispatch savings metrics.
    ///
    /// # Arguments
    ///
    /// * `segment_name` - Human-readable name for diagnostic reporting.
    /// * `steps` - The compiled step sequence to analyze.
    #[must_use]
    pub fn analyze(&self, segment_name: &str, steps: &[CompiledStep]) -> ConvBatchAnalysis {
        let total_dispatches = count_dispatches(steps);
        let conv1d_dispatches = count_conv1d_dispatches(steps);
        let groups = self.find_batch_groups(steps);
        let total_saved: usize = groups.iter().map(Conv1dBatchGroup::dispatches_saved).sum();

        ConvBatchAnalysis {
            segment_name: segment_name.to_string(),
            groups,
            total_dispatches,
            conv1d_dispatches,
            total_saved,
            optimized_dispatches: total_dispatches.saturating_sub(total_saved),
        }
    }

    /// Analyze all segments from a per-segment step audit.
    ///
    /// Takes the output of `CompiledKokoro::per_segment_step_audit()` and
    /// the corresponding step lists from each segment's compiled model.
    ///
    /// This is a convenience method for analyzing all segments at once.
    #[must_use]
    pub fn analyze_segments(
        &self,
        segment_data: &[(&str, &[CompiledStep])],
    ) -> Vec<ConvBatchAnalysis> {
        segment_data
            .iter()
            .map(|(name, steps)| self.analyze(name, steps))
            .collect()
    }

    /// Find batch groups of consecutive compatible Conv1dGemm steps.
    fn find_batch_groups(&self, steps: &[CompiledStep]) -> Vec<Conv1dBatchGroup> {
        let mut groups = Vec::new();
        let mut candidates: Vec<Conv1dCandidate> = Vec::new();

        // Scan steps, collecting runs of compatible Conv1dGemm operations.
        // Zero-cost steps (Passthrough, NarrowView, etc.) are skipped.
        for (idx, step) in steps.iter().enumerate() {
            if is_zero_cost_step(step) {
                // Zero-cost steps don't break a batch run.
                continue;
            }

            if let Some(candidate) = extract_conv1d_candidate(idx, step) {
                if let Some(last) = candidates.last() {
                    if last.key == candidate.key {
                        // Compatible: extend current run.
                        candidates.push(candidate);
                        continue;
                    }
                    // Incompatible: flush current run.
                    self.flush_candidates(&mut candidates, &mut groups);
                }
                // Start new run.
                candidates.push(candidate);
            } else {
                // Non-conv1d dispatch step: flush current run.
                self.flush_candidates(&mut candidates, &mut groups);
            }
        }

        // Flush any remaining candidates.
        self.flush_candidates(&mut candidates, &mut groups);

        groups
    }

    /// Convert a run of compatible candidates into a batch group if large enough.
    fn flush_candidates(
        &self,
        candidates: &mut Vec<Conv1dCandidate>,
        groups: &mut Vec<Conv1dBatchGroup>,
    ) {
        if candidates.len() >= self.min_batch_size {
            let first = &candidates[0];
            let step_indices: Vec<usize> = candidates.iter().map(|c| c.step_idx).collect();
            let output_channels: Vec<usize> =
                candidates.iter().map(|c| c.out_channels).collect();
            let all_have_bias = candidates.iter().all(|c| c.has_bias);
            let any_have_bias = candidates.iter().any(|c| c.has_bias);
            let total_output_channels: usize = output_channels.iter().sum();

            groups.push(Conv1dBatchGroup {
                step_indices,
                kernel_size: first.key.kernel_size,
                stride: first.key.stride,
                dilation: first.key.dilation,
                groups: first.key.groups,
                input_channels: first.key.input_channels,
                output_channels,
                all_have_bias,
                any_have_bias,
                total_output_channels,
            });
        }
        candidates.clear();
    }
}

/// Count total dispatches in a step sequence (NativeOp + Dispatch + RuntimeOp).
fn count_dispatches(steps: &[CompiledStep]) -> usize {
    steps
        .iter()
        .filter(|s| matches!(s, CompiledStep::NativeOp { .. } | CompiledStep::Dispatch { .. } | CompiledStep::RuntimeOp { .. }))
        .count()
}

/// Count Conv1dGemm-specific dispatches (estimated Metal kernel launches).
fn count_conv1d_dispatches(steps: &[CompiledStep]) -> usize {
    steps
        .iter()
        .filter_map(|s| match s {
            CompiledStep::NativeOp { op, .. } => match op {
                NativeOpKind::Conv1dGemm { .. } => Some(op.estimated_metal_dispatches()),
                _ => None,
            },
            _ => None,
        })
        .sum()
}

/// Check if a step is zero-cost (no GPU dispatch).
fn is_zero_cost_step(step: &CompiledStep) -> bool {
    matches!(
        step,
        CompiledStep::Passthrough { .. }
            | CompiledStep::NarrowView { .. }
            | CompiledStep::InputForward
            | CompiledStep::IdentityPassthrough
            | CompiledStep::ConstantValue { .. }
    )
}

/// Aggregate analysis across all segments.
#[derive(Debug, Clone)]
pub struct PipelineConvBatchSummary {
    /// Per-segment analysis results.
    pub segments: Vec<ConvBatchAnalysis>,
    /// Total dispatches across all segments.
    pub total_dispatches: usize,
    /// Total dispatches saved across all segments.
    pub total_saved: usize,
    /// Total batch groups found across all segments.
    pub total_groups: usize,
}

impl PipelineConvBatchSummary {
    /// Build a summary from a list of per-segment analyses.
    #[must_use]
    pub fn from_analyses(analyses: Vec<ConvBatchAnalysis>) -> Self {
        let total_dispatches: usize = analyses.iter().map(|a| a.total_dispatches).sum();
        let total_saved: usize = analyses.iter().map(|a| a.total_saved).sum();
        let total_groups: usize = analyses.iter().map(|a| a.groups.len()).sum();
        Self {
            segments: analyses,
            total_dispatches,
            total_saved,
            total_groups,
        }
    }

    /// Optimized total dispatch count.
    #[must_use]
    pub fn optimized_dispatches(&self) -> usize {
        self.total_dispatches.saturating_sub(self.total_saved)
    }

    /// Whether any batching opportunities exist.
    #[must_use]
    pub fn has_opportunities(&self) -> bool {
        self.total_groups > 0
    }
}

impl fmt::Display for PipelineConvBatchSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Pipeline Conv1d Batch Summary: {} dispatches -> {} ({} saved, {} groups)",
            self.total_dispatches,
            self.optimized_dispatches(),
            self.total_saved,
            self.total_groups,
        )?;
        for analysis in &self.segments {
            if analysis.has_opportunities() {
                write!(f, "  {analysis}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "compiled_kokoro_conv_batch_tests.rs"]
mod tests;
