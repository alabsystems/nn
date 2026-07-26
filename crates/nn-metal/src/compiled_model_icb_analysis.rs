// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ICB eligibility and concurrent barrier analysis for `CompiledModel`.
//!
//! Build-time analysis utilities: determines which steps can be pre-encoded
//! into an ICB and which need memory barriers for concurrent dispatch.
//!
//! Extracted from `compiled_model_icb.rs` for 450-line compliance.
//! Part of #3258 (Phase 2 selective barriers) and #3206 (ICB replay).

/// Compute which steps need a memory barrier before execution in concurrent mode.
///
/// A step needs a barrier if it reads from a buffer region that a prior
/// concurrent dispatch wrote to. Non-dispatch steps (NativeOp, RuntimeOp,
/// zero-copy) act as implicit full barriers because they pause concurrent
/// GPU execution.
///
/// Returns a `Vec<bool>` parallel to the step list: `true` = insert barrier.
///
/// Part of #3258 (Phase 2 selective barriers).
pub(crate) fn compute_concurrent_barriers(
    edge_map: &[Vec<usize>],
    step_offsets: &[Option<usize>],
    is_gpu_dispatch: &[bool],
) -> Vec<bool> {
    let n = edge_map.len();
    let mut needs_barrier = vec![false; n];
    let mut dirty_offsets: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for i in 0..n {
        if !is_gpu_dispatch[i] {
            dirty_offsets.clear();
            continue;
        }

        let has_dependency = edge_map[i].iter().any(|&src| {
            let offset = resolve_planned_offset(src, step_offsets, edge_map);
            if let Some(off) = offset {
                dirty_offsets.contains(&off)
            } else {
                false
            }
        });

        if has_dependency {
            needs_barrier[i] = true;
            dirty_offsets.clear();
        }

        if let Some(offset) = step_offsets.get(i).and_then(|o| *o) {
            dirty_offsets.insert(offset);
        }
    }

    needs_barrier
}

/// Resolve a step index to its underlying planned buffer offset.
///
/// NarrowView, Passthrough, and IdentityPassthrough steps have
/// `step_offsets[idx] = None` because they alias another step's buffer.
/// Walks the edge_map until a real planned offset is found (max 16 hops).
fn resolve_planned_offset(
    idx: usize,
    step_offsets: &[Option<usize>],
    edge_map: &[Vec<usize>],
) -> Option<usize> {
    let mut current = idx;
    for _ in 0..16 {
        if let Some(offset) = step_offsets.get(current).and_then(|o| *o) {
            return Some(offset);
        }
        if let Some(edges) = edge_map.get(current) {
            if let Some(&parent) = edges.first() {
                if parent == current {
                    return None;
                }
                current = parent;
                continue;
            }
        }
        return None;
    }
    None
}

/// Determine which steps are GPU dispatches for concurrent barrier analysis.
///
/// Broader than ICB eligibility: includes ALL `Dispatch` steps regardless
/// of autocast/mixed-precision mode. Part of #3258.
pub(crate) fn analyze_gpu_dispatch_steps(
    steps: &[nn_dsl::trace_compile::CompiledStep],
) -> Vec<bool> {
    use nn_dsl::trace_compile::CompiledStep;
    steps
        .iter()
        .map(|step| matches!(step, CompiledStep::Dispatch { .. }))
        .collect()
}

/// Summary of barrier analysis for observability logging.
#[derive(Debug, Clone)]
#[allow(dead_code)] // ICB wiring in progress (#3259)
pub(crate) struct BarrierSummary {
    pub(crate) eligible: usize,
    pub(crate) barriers: usize,
    pub(crate) concurrent: usize,
}

/// Summarize barrier analysis for logging at build time.
#[allow(dead_code)] // ICB wiring in progress (#3259)
pub(crate) fn summarize_barriers(icb_eligible: &[bool], needs_barrier: &[bool]) -> BarrierSummary {
    let eligible = icb_eligible.iter().filter(|&&e| e).count();
    let barriers = icb_eligible
        .iter()
        .zip(needs_barrier.iter())
        .filter(|(&e, &b)| e && b)
        .count();
    BarrierSummary {
        eligible,
        barriers,
        concurrent: eligible.saturating_sub(barriers),
    }
}

/// Analyze which compiled steps are eligible for ICB pre-encoding.
///
/// A step is eligible if it is a `Dispatch` step with static shapes and
/// no autocast/mixed-precision boundary casting. Returns a `Vec<bool>`.
///
/// For autocast models (#3426): simulates the runtime `DtypeTracker` at
/// build time to identify Dispatch steps whose inputs all have matching
/// scalar types — meaning no `cast_autocast_inputs()` call is needed.
/// Mixed GEMM steps and RuntimeOp-downstream steps are excluded.
pub(crate) fn analyze_icb_eligibility(
    steps: &[nn_dsl::trace_compile::CompiledStep],
    step_metas: &[super::super::StepMeta],
    mixed_gemm_infos: &[Option<super::super::MixedGemmInfo>],
    autocast_active: bool,
    mixed_precision_active: bool,
) -> Vec<bool> {
    use nn_dsl::ir::ScalarType;
    use nn_dsl::trace_compile::CompiledStep;

    // Mixed precision has runtime dtype mutations too complex for static analysis.
    if mixed_precision_active {
        return vec![false; steps.len()];
    }

    // No autocast: all Dispatch steps eligible (original path).
    if !autocast_active {
        return steps
            .iter()
            .map(|s| matches!(s, CompiledStep::Dispatch { .. }))
            .collect();
    }

    // Autocast: simulate runtime DtypeTracker at build time.
    // Initialize from step scalar types (mirrors DtypeTracker::new).
    let mut sim_dtypes: Vec<ScalarType> = step_metas.iter().map(|m| m.scalar_type).collect();

    // Forward pass: propagate dtype overrides exactly as the execution loop does.
    for (i, step) in steps.iter().enumerate() {
        match step {
            CompiledStep::Dispatch { .. } => {
                // Mixed GEMM always outputs F32 at runtime (execute_mixed_dispatch).
                if mixed_gemm_infos
                    .get(i)
                    .and_then(|info| info.as_ref())
                    .is_some()
                {
                    sim_dtypes[i] = ScalarType::F32;
                }
                // Normal Dispatch outputs step_dt (already in sim_dtypes[i]).
            }
            CompiledStep::RuntimeOp { .. } => {
                // RuntimeOp always produces F32 (hardcoded in execute_runtime).
                sim_dtypes[i] = ScalarType::F32;
            }
            CompiledStep::Passthrough { .. }
            | CompiledStep::IdentityPassthrough
            | CompiledStep::NarrowView { .. } => {
                // Propagate source dtype (mirrors DtypeTracker::propagate_from_source).
                if let Some(&src) = step_metas.get(i).and_then(|m| m.edges.first()) {
                    if let Some(&src_dt) = sim_dtypes.get(src) {
                        sim_dtypes[i] = src_dt;
                    }
                }
            }
            _ => {}
        }
    }

    // Mark Dispatch steps as eligible when no input needs casting.
    steps
        .iter()
        .enumerate()
        .map(|(i, step)| match step {
            CompiledStep::Dispatch { .. } => {
                // Mixed GEMM uses a separate dispatch path, not ICB-compatible.
                if mixed_gemm_infos
                    .get(i)
                    .and_then(|info| info.as_ref())
                    .is_some()
                {
                    return false;
                }
                let step_dt = step_metas[i].scalar_type;
                let needs_cast = step_metas[i].edges.iter().any(|&src| {
                    sim_dtypes.get(src).copied().unwrap_or(ScalarType::F32) != step_dt
                });
                !needs_cast
            }
            _ => false,
        })
        .collect()
}

/// Summary of ICB eligibility analysis.
#[derive(Debug, Clone)]
#[allow(dead_code)] // ICB wiring in progress (#3259)
pub(crate) struct IcbEligibilitySummary {
    pub(crate) total_steps: usize,
    pub(crate) eligible: usize,
    pub(crate) runtime: usize,
    pub(crate) zero_copy: usize,
}

/// A contiguous segment of ICB-eligible steps with pre-compiled metadata.
///
/// At build time, codegen outputs are pre-compiled and cached per step.
/// On first forward pass, an `IndirectCommandBuffer` is lazily encoded
/// from these outputs. Subsequent passes replay the ICB directly.
///
/// Part of #3259 (D1, D2).
pub(crate) struct IcbSegment {
    /// First step index in this segment (inclusive).
    pub(crate) start: usize,
    /// Last step index in this segment (inclusive).
    pub(crate) end: usize,
    /// Pre-compiled codegen outputs for each Dispatch step in [start..=end].
    /// Cached at build time so forward passes skip codegen_for_kernel.
    pub(crate) step_codegen: Vec<std::sync::Arc<crate::msl_codegen_cache::CodegenOutput>>,
    /// External bindings: `(icb_command_idx, buffer_arg_idx, source_step_idx)`.
    /// Activation buffers from outside the segment that need update_buffer at runtime.
    pub(crate) external_bindings: Vec<(usize, usize, usize)>,
    /// Planned buffer bindings: `(icb_command_idx, buffer_arg_idx, byte_offset)`.
    /// Output slots in the planned buffer with static offsets.
    pub(crate) planned_bindings: Vec<(usize, usize, usize)>,
}

impl std::fmt::Debug for IcbSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcbSegment")
            .field("start", &self.start)
            .field("end", &self.end)
            .field("step_codegen_count", &self.step_codegen.len())
            .field("external_bindings", &self.external_bindings.len())
            .field("planned_bindings", &self.planned_bindings.len())
            .finish()
    }
}

/// Detect contiguous runs of ICB-eligible steps.
///
/// Returns `(start, end)` pairs (both inclusive) for each contiguous run
/// of eligible steps with length >= `min_segment_len`. Part of #3259 (D1).
pub(crate) fn detect_icb_segments(
    eligible: &[bool],
    min_segment_len: usize,
) -> Vec<(usize, usize)> {
    let mut segments = Vec::new();
    let mut run_start: Option<usize> = None;

    for (i, &is_eligible) in eligible.iter().enumerate() {
        if is_eligible {
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else if let Some(start) = run_start.take() {
            let end = i - 1;
            if end - start + 1 >= min_segment_len {
                segments.push((start, end));
            }
        }
    }
    if let Some(start) = run_start {
        let end = eligible.len() - 1;
        if end - start + 1 >= min_segment_len {
            segments.push((start, end));
        }
    }

    segments
}

/// Build a `HashMap<usize, usize>` from step index → segment index
/// for O(1) lookup in the execution loop. Part of #3259 (D1).
pub(crate) fn build_segment_starts(
    segments: &[(usize, usize)],
) -> std::collections::HashMap<usize, usize> {
    segments
        .iter()
        .enumerate()
        .map(|(seg_idx, &(start, _end))| (start, seg_idx))
        .collect()
}

/// Summarize ICB eligibility from the per-step boolean vector.
#[allow(dead_code)] // ICB wiring in progress (#3259)
pub(crate) fn summarize_eligibility(
    steps: &[nn_dsl::trace_compile::CompiledStep],
    eligible: &[bool],
) -> IcbEligibilitySummary {
    use nn_dsl::trace_compile::CompiledStep;

    let mut summary = IcbEligibilitySummary {
        total_steps: steps.len(),
        eligible: 0,
        runtime: 0,
        zero_copy: 0,
    };
    for (i, step) in steps.iter().enumerate() {
        if eligible.get(i).copied().unwrap_or(false) {
            summary.eligible += 1;
        } else {
            match step {
                CompiledStep::NativeOp { .. } | CompiledStep::RuntimeOp { .. } => {
                    summary.runtime += 1;
                }
                _ => {
                    summary.zero_copy += 1;
                }
            }
        }
    }
    summary
}

#[cfg(test)]
#[path = "compiled_model_icb_analysis_tests.rs"]
mod tests;
