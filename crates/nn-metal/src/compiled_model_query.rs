// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Query and inspection methods for `CompiledModel`.
//!
//! Extracted from `compiled_model.rs` to keep files under 450 lines.
//! Contains accessors for model metadata (shapes, dtypes, step counts),
//! the precision contract builder, and dispatch count introspection.

use std::sync::Arc;

use nn_core::DType;
use nn_dsl::buffer_planner::BufferPlan;
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::CompiledStep;
use nn_dsl::{build_dispatch_plan, PrecisionContract};

use super::CompiledModel;

impl CompiledModel {
    /// Returns the primary output dtype of the compiled model.
    #[must_use]
    pub fn output_dtype(&self) -> DType {
        self.def.output_metas
            .last()
            .map(|(_, d)| *d)
            .unwrap_or(DType::F32)
    }

    /// Returns the number of compiled steps.
    #[must_use]
    pub fn num_steps(&self) -> usize {
        self.def.steps.len()
    }

    /// Returns the number of GPU dispatch steps (both IR-generated and native).
    ///
    /// Counts `Dispatch` (IR → MSL code-generated kernels) and `NativeOp`
    /// (pre-compiled fused kernels like LSTM sequence). Both are GPU kernel
    /// launches that contribute to dispatch overhead.
    #[must_use]
    pub fn num_dispatches(&self) -> usize {
        self.def.steps
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
                )
            })
            .count()
    }

    /// Returns the number of IR-generated dispatch steps only.
    ///
    /// Excludes `NativeOp` steps. Useful for measuring IR codegen coverage.
    #[must_use]
    pub fn num_ir_dispatches(&self) -> usize {
        self.def.steps
            .iter()
            .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
            .count()
    }

    /// Returns the edge map (input step indices) for a given step.
    ///
    /// Delegates to `step_metas[step_idx].edges`. Part of #1815 StepMeta migration.
    pub(crate) fn edge_map_for(&self, step_idx: usize) -> &[usize] {
        self.def.step_metas
            .get(step_idx)
            .map(|m| m.edges.as_slice())
            .unwrap_or(&[])
    }

    /// Returns whether F16 mixed-precision is active.
    #[must_use]
    pub fn is_mixed_precision(&self) -> bool {
        self.def.mixed_precision_active
    }

    /// Returns whether per-op autocast is active.
    ///
    /// When true, the model was compiled with [`builder().autocast()`].
    /// All intermediate buffers stay F32; the autocast policy controls
    /// per-op weight dtypes (Phase 2). Part of #3085.
    #[must_use]
    pub fn is_autocast(&self) -> bool {
        self.def.autocast_policy.is_some()
    }

    /// Returns the number of steps using mixed-precision simdgroup GEMM.
    ///
    /// When autocast is active, simdgroup-eligible Linear/MatMul steps bypass
    /// the IR dispatch and use `simd_gemm_mixed` (F32 activations × F16 weights).
    /// Returns 0 when autocast is off. Part of #3085.
    #[must_use]
    pub fn num_mixed_gemm_steps(&self) -> usize {
        self.def.mixed_gemm_infos.iter().filter(|o| o.is_some()).count()
    }

    /// Returns the number of steps running in F16 via autocast.
    ///
    /// Counts all steps where autocast set `step_scalar_types` to F16:
    /// GEMM, Conv, Embedding, FlashAttention, NormActivConv1d, etc.
    /// Returns 0 when autocast is off. Part of #2981.
    #[must_use]
    pub fn num_autocast_f16_steps(&self) -> usize {
        if !self.def.autocast_active {
            return 0;
        }
        self.def.step_metas
            .iter()
            .filter(|m| m.scalar_type == ScalarType::F16 || m.scalar_type == ScalarType::BF16)
            .count()
    }

    /// Returns the number of native (pre-compiled) op steps.
    ///
    /// These are operations like LSTM that use pre-compiled Metal kernels
    /// instead of the IR → MSL code-generation path.
    #[must_use]
    pub fn num_native_ops(&self) -> usize {
        self.def.steps
            .iter()
            .filter(|s| matches!(s, CompiledStep::NativeOp { .. }))
            .count()
    }

    /// Returns the actual number of Metal kernel launches after plan expansion.
    ///
    /// Each `Dispatch` step expands via `build_dispatch_plan()` into
    /// `DispatchStep` entries (composite IR ops like norms are decomposed).
    /// `NativeOp` steps use `estimated_metal_dispatches()` which accounts for
    /// internal sub-dispatches (e.g., FusedResBlock → 5-10 launches,
    /// NormActivConv1d → 2). Falls back to 1 per step if plan building fails.
    #[must_use]
    pub fn num_metal_dispatches(&self) -> usize {
        self.def.steps
            .iter()
            .map(|s| match s {
                CompiledStep::Dispatch { kernel, .. } => {
                    build_dispatch_plan(kernel.def(), ScalarType::F32)
                        .map(|(plan, _)| plan.len())
                        .unwrap_or(1)
                }
                CompiledStep::NativeOp { op, .. } => op.estimated_metal_dispatches(),
                _ => 0,
            })
            .sum()
    }

    /// Estimated encoding events (compute dispatches + blit relocations).
    ///
    /// For IR `Dispatch` steps: counts 1 (one `get_or_create_batch()` call,
    /// not `plan.len()`). For `NativeOp` steps: uses `estimated_encoding_events()`
    /// which counts `get_or_create_batch()` calls (not sub-encoders within a batch).
    /// Adds +1 per step with a planned blit relocation (non-zero buffer plan size).
    ///
    /// This metric tracks `TOTAL_ENCODINGS + TOTAL_BLITS` at runtime.
    /// Use this instead of [`num_metal_dispatches()`] for accuracy comparisons
    /// against [`dispatch_stats()`]. See #1815 D5.1, D5.2.
    #[must_use]
    pub fn num_encoding_events(&self) -> usize {
        let mut count = 0;
        for (i, step) in self.def.steps.iter().enumerate() {
            match step {
                CompiledStep::Dispatch { .. } => {
                    count += 1;
                    if self.def.buffer_plan.step_sizes.get(i).copied().unwrap_or(0) > 0 {
                        count += 1; // blit relocation
                    }
                }
                CompiledStep::NativeOp { op, .. } => {
                    count += op.estimated_encoding_events();
                    if self.def.buffer_plan.step_sizes.get(i).copied().unwrap_or(0) > 0 {
                        count += 1; // blit relocation
                    }
                }
                _ => {}
            }
        }
        count
    }

    /// Per-type dispatch breakdown for optimization diagnostics (#2780).
    ///
    /// Returns `(ir_by_name, native_by_kind)` where:
    /// - `ir_by_name`: IR kernel name → Metal dispatches (via `build_dispatch_plan`)
    /// - `native_by_kind`: NativeOp variant name → Metal dispatches (via `estimated_metal_dispatches`)
    ///
    /// Sorted descending by count. Counts reflect actual Metal kernel launches,
    /// not just compiled step count.
    #[must_use]
    pub fn dispatch_breakdown(&self) -> (Vec<(String, usize)>, Vec<(String, usize)>) {
        use std::collections::HashMap;
        let mut ir_map: HashMap<String, usize> = HashMap::new();
        let mut native_map: HashMap<String, usize> = HashMap::new();
        for step in &self.def.steps {
            match step {
                CompiledStep::Dispatch { kernel, .. } => {
                    let count = build_dispatch_plan(kernel.def(), ScalarType::F32)
                        .map(|(plan, _)| plan.len())
                        .unwrap_or(1);
                    *ir_map.entry(kernel.name().to_string()).or_default() += count;
                }
                CompiledStep::NativeOp { op, .. } => {
                    let count = op.estimated_metal_dispatches();
                    *native_map.entry(op.variant_name().to_string()).or_default() += count;
                }
                _ => {}
            }
        }
        let mut ir_vec: Vec<_> = ir_map.into_iter().collect();
        ir_vec.sort_by_key(|x| std::cmp::Reverse(x.1));
        let mut native_vec: Vec<_> = native_map.into_iter().collect();
        native_vec.sort_by_key(|x| std::cmp::Reverse(x.1));
        (ir_vec, native_vec)
    }

    /// Returns the expected number of input tensors.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.def.num_inputs
    }

    /// Returns the expected input shapes and dtypes.
    #[must_use]
    pub fn input_specs(&self) -> &[(Vec<usize>, DType)] {
        &self.def.input_specs
    }

    /// Returns the primary output shape of the compiled model.
    #[must_use]
    pub fn output_shape(&self) -> &[usize] {
        self.def.output_metas
            .last()
            .map(|(s, _)| s.as_slice())
            .unwrap_or(&[])
    }

    /// Returns the number of output nodes.
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        self.def.output_step_indices.len()
    }

    /// Returns the compiled steps (for inspection/debugging).
    #[must_use]
    pub fn steps(&self) -> &[CompiledStep] {
        &self.def.steps
    }

    /// Returns the static buffer allocation plan.
    ///
    /// Inspect `buffer_plan.total_bytes` vs `buffer_plan.naive_total` to
    /// see the memory savings from buffer reuse.
    #[must_use]
    pub fn buffer_plan(&self) -> &BufferPlan {
        &self.def.buffer_plan
    }

    /// Set the precision contract for all dispatch steps in this model.
    ///
    /// Use `PrecisionTier::Strict` for Kahan-compensated reductions in
    /// normalization ops (InstanceNorm, LayerNorm, etc.). Without this,
    /// chained normalization layers accumulate reduction drift.
    #[must_use]
    pub fn with_precision(mut self, contract: PrecisionContract) -> Self {
        // `self` is moved in, so refcount == 1 — `get_mut` always succeeds.
        Arc::get_mut(&mut self.def)
            .expect("with_precision takes owned self, refcount must be 1")
            .precision = Some(contract);
        self
    }

    /// Returns the current precision contract, if set.
    #[must_use]
    pub fn precision(&self) -> Option<&PrecisionContract> {
        self.def.precision.as_ref()
    }

    /// Returns the `DType` for the given step index.
    ///
    /// Converts the per-step `ScalarType` (F32/F16) to `DType`. Used by
    /// `execute_native_*` functions to create `DynTensor` wrappers with
    /// the correct dtype instead of hardcoding `DType::F32`. Part of D5b.
    /// Delegates to `step_metas[step_idx].scalar_type`. Part of #1815.
    pub(crate) fn step_dtype(&self, step_idx: usize) -> DType {
        self.def.step_metas
            .get(step_idx)
            .map(|m| m.scalar_type)
            .unwrap_or(ScalarType::F32)
            .into()
    }

    /// Returns the `ScalarType` for the given step index.
    ///
    /// Used by `execute_native_op_mixed` to decide whether boundary casting
    /// is needed (F32 NativeOps like LSTM need it; F16 NativeOps don't).
    /// Delegates to `step_metas[step_idx].scalar_type`. Part of #1815.
    pub(crate) fn step_scalar_type(&self, step_idx: usize) -> ScalarType {
        self.def.step_metas
            .get(step_idx)
            .map(|m| m.scalar_type)
            .unwrap_or(ScalarType::F32)
    }

    /// Returns the pre-computed element count (numel) for a step.
    ///
    /// Used for F16↔F32 boundary casts because relocated slices (in the planned
    /// buffer) have `buffer.len()` reflecting the entire shared allocation, not
    /// just one step's data.
    /// Delegates to `step_metas[step_idx].numel`. Part of #1815.
    pub(crate) fn step_numel(&self, step_idx: usize) -> usize {
        self.def.step_metas.get(step_idx).map(|m| m.numel).unwrap_or(0)
    }

    /// Returns the effective element count for a step's output buffer.
    ///
    /// For RuntimeOp steps, the output buffer is freshly allocated at runtime
    /// with exact size (not in the planned buffer), so buffer geometry gives
    /// the correct count. For all other steps, relocated slices sit in the
    /// shared planned buffer whose `buffer.len()` reflects the full allocation,
    /// so the pre-computed trace-time `step_numel` is used instead.
    ///
    /// See #3121: `step_numel` returns trace-time shape for RuntimeOp, which
    /// is wrong when the output is data-dependent (e.g., RepeatInterleave).
    pub(crate) fn effective_numel(
        &self,
        step_idx: usize,
        slice: &crate::gpu_slice::GpuSlice,
        dtype: ScalarType,
    ) -> usize {
        if matches!(
            self.def.steps.get(step_idx),
            Some(CompiledStep::RuntimeOp { .. })
        ) {
            let bytes_per_elem = dtype.byte_size();
            let avail = slice.buffer().len().saturating_sub(slice.byte_offset());
            avail / bytes_per_elem
        } else {
            self.step_numel(step_idx)
        }
    }

    /// Returns the number of ICB (Indirect Command Buffer) segments detected.
    ///
    /// ICB segments are contiguous runs of 4+ Dispatch steps that can be
    /// pre-encoded and replayed via a single `executeCommandsInBuffer` call.
    /// Returns 0 when autocast or mixed precision is active.
    #[must_use]
    pub fn num_icb_segments(&self) -> usize {
        self.def.icb_segments.len()
    }

    /// Returns whether an ICB segment starts at the given step index.
    ///
    /// Used by tests to verify ICB segment detection and wiring.
    #[must_use]
    pub fn icb_segment_starts_at(&self, step_idx: usize) -> bool {
        self.def.icb_segment_starts.contains_key(&step_idx)
    }

    /// Build a structured [`SegmentPerformance`] for this compiled model.
    ///
    /// Populates dispatch counts and buffer metrics from the compiled plan.
    /// Latency is `None` — call [`SegmentPerformance::with_latency`] after
    /// benchmarking to fill it in.
    #[must_use]
    pub fn segment_performance(&self, name: &str) -> nn_dsl::SegmentPerformance {
        let bp = &self.def.buffer_plan;
        let mut sp = nn_dsl::SegmentPerformance::new(name);
        sp.dispatches = self.num_dispatches();
        sp.metal_dispatches = self.num_metal_dispatches();
        sp.steps = self.num_steps();
        sp.native_ops = self.num_native_ops();
        sp.ir_dispatches = self.num_ir_dispatches();
        sp.buffer_bytes = bp.total_bytes;
        sp.buffer_naive_bytes = bp.naive_total;
        sp
    }

    /// Generate a memory report for this compiled model.
    ///
    /// Returns a structured [`MemoryReport`](crate::compiled_model_memory_report::MemoryReport)
    /// with per-step breakdown, weight/intermediate totals, and peak memory.
    /// Part of #3828.
    #[must_use]
    pub fn memory_report(&self) -> crate::compiled_model_memory_report::MemoryReport {
        crate::compiled_model_memory_report::generate_memory_report(&self.def)
    }
}
