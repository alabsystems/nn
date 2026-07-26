// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unified step execution loop for `CompiledModel`.
//!
//! Contains `run_steps_inner` — the single implementation of the compiled
//! step execution loop. Both `run_steps` (non-profiled) and
//! `run_steps_profiled` are thin wrappers around this function.
//!
//! Eliminates ~280 lines of duplicated step logic that previously lived
//! in `compiled_model_execute.rs` and `compiled_model_execute_profiled.rs`.
//! The duplication caused 4 parity bugs (see design doc
//! `designs/2026-03-22-unified-execute-core-dedup.md`).
//!
//! Part of #2981.

use std::collections::HashMap;
use std::time::Instant;

use nn_core::{Result, TensorError};
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::CompiledStep;

use crate::cache::PipelineCache;
use crate::compiled_model::profile::{is_gpu_dispatch, step_name, ExecutionProfile, StepProfile};
use crate::gpu_slice::GpuSlice;

use super::super::dtype_tracker::DtypeTracker;
use super::{helpers, CompiledModel, CompiledModelError};

impl CompiledModel {
    /// Unified step execution loop with optional per-step profiling.
    ///
    /// When `profile` is true, flushes GPU after each dispatch step and
    /// records wall-clock timing. The `if profile` branch is one prediction
    /// per step (~10-50 steps) — negligible vs GPU dispatch latency.
    ///
    /// Returns `(buffers, buffer_dtypes, Option<ExecutionProfile>)`.
    /// Non-profiled callers discard the `None` profile.
    pub(super) fn run_steps_inner(
        &self,
        cache: &PipelineCache,
        inputs: &[GpuSlice],
        profile: bool,
    ) -> Result<(
        Vec<Option<GpuSlice>>,
        Vec<ScalarType>,
        Option<ExecutionProfile>,
    )> {
        // Clear stashed batched projection intermediates from prior forward pass (#3269).
        super::native_ops::clear_projection_temps();
        let mut buffers: Vec<Option<GpuSlice>> = (0..self.def.steps.len()).map(|_| None).collect();
        let needs_dtype_tracking = self.def.mixed_precision_active || self.def.autocast_active;
        let mut dtypes = DtypeTracker::new(&self.def.step_metas, needs_dtype_tracking);

        // Use pre-computed release_at from construction time. (#2944)
        let release_at = &self.def.release_at;

        // Scratch HashMap reused across dispatch steps (#2501).
        let mut dispatch_scratch: HashMap<&str, GpuSlice> = HashMap::new();

        // Contiguous buffer for BufferPlan sub-allocation (#2913).
        // Cached on self and reused across forward passes. Output data is
        // blit-copied out by normalize_output_to_offset_zero before run_steps
        // returns, so reusing the same buffer is safe.
        let planned_buf = if self.def.buffer_plan.total_bytes > 0 {
            let needs_alloc = self.cached_planned_buf.borrow().is_none();
            if needs_alloc {
                let buf = cache
                    .context()
                    .create_buffer_zeroed(self.def.buffer_plan.total_bytes)
                    .map_err(|e| {
                        TensorError::from(CompiledModelError::DispatchFailed {
                            step_idx: 0,
                            reason: format!("planned buffer alloc: {e}"),
                        })
                    })?;
                *self.cached_planned_buf.borrow_mut() = Some(buf);
            }
            let cached = self.cached_planned_buf.borrow();
            Some(cached.as_ref().expect("just allocated").alias())
        } else {
            None
        };
        let step_offsets = &self.def.buffer_plan.step_offsets;
        let step_sizes = &self.def.buffer_plan.step_sizes;

        let mut step_profiles: Vec<StepProfile> = if profile {
            Vec::with_capacity(self.def.steps.len())
        } else {
            Vec::new()
        };

        // ICB replay: skip steps covered by a pre-encoded segment (#3259).
        let mut icb_skip_until: Option<usize> = None;

        let mut input_idx = 0;
        for (step_idx, step) in self.def.steps.iter().enumerate() {
            // Skip steps inside an ICB segment that was already replayed.
            if let Some(end) = icb_skip_until {
                if step_idx <= end {
                    continue;
                }
                icb_skip_until = None;
            }

            // ICB segment replay (#3259 D4): attempt replay, fall back to normal.
            if let Some(&seg_idx) = self.def.icb_segment_starts.get(&step_idx) {
                if let Some(seg) = self.def.icb_segments.get(seg_idx) {
                    if self.try_replay_icb(
                        cache,
                        seg_idx,
                        seg,
                        &mut buffers,
                        planned_buf.as_ref(),
                    )? {
                        icb_skip_until = Some(seg.end);
                        continue;
                    }
                }
            }

            // Only timestamp when profiling — avoids syscall per step on hot path.
            let start = profile.then(Instant::now);

            // Save arena checkpoint for steps with planned offsets (#2913).
            // Generation-guarded: if GPU auto-flush resets the arena between
            // checkpoint and restore, restore_default_arena detects the
            // generation mismatch and skips (no panic). (#3133)
            let planned_offset = step_offsets.get(step_idx).and_then(|o| *o);
            let arena_cp = if planned_offset.is_some() && planned_buf.is_some() {
                crate::arena::checkpoint_default_arena()
            } else {
                None
            };

            match step {
                CompiledStep::InputForward => {
                    if input_idx >= inputs.len() {
                        return Err(CompiledModelError::InputCountMismatch {
                            expected: self.def.num_inputs,
                            got: inputs.len(),
                        }
                        .into());
                    }
                    if self.def.mixed_precision_active {
                        buffers[step_idx] = Some(helpers::cast_input_f32_to_f16(
                            cache,
                            &inputs[input_idx],
                            self.step_numel(step_idx),
                        )?);
                    } else {
                        buffers[step_idx] = Some(inputs[input_idx].alias());
                    }
                    input_idx += 1;
                }
                CompiledStep::IdentityPassthrough => {
                    let src = self.resolve_input_slice(step_idx, 0, &buffers)?;
                    buffers[step_idx] = Some(src);
                    dtypes.propagate_from_source(step_idx, &self.def.step_metas);
                }
                CompiledStep::Passthrough { .. } => {
                    let src = self.resolve_input_slice(step_idx, 0, &buffers)?;
                    buffers[step_idx] = Some(src);
                    dtypes.propagate_from_source(step_idx, &self.def.step_metas);
                }
                CompiledStep::NarrowView { byte_offset, .. } => {
                    let src = self.resolve_input_slice(step_idx, 0, &buffers)?;
                    // Zero-copy: alias the input buffer at the pre-computed byte offset.
                    // No GPU dispatch, no memcpy (#2780).
                    // byte_offset is pre-computed assuming F32 (4 bytes) in nn-dsl
                    // (trace_compile_ops.rs). In mixed-precision/autocast mode,
                    // scale to actual element size. (#2981)
                    let effective_offset =
                        dtypes.narrow_byte_offset(step_idx, &self.def.step_metas, *byte_offset);
                    let new_offset =
                        src.byte_offset()
                            .checked_add(effective_offset)
                            .ok_or_else(|| {
                                TensorError::from(CompiledModelError::DispatchFailed {
                                    step_idx,
                                    reason: format!(
                                        "NarrowView byte_offset overflow: {} + {}",
                                        src.byte_offset(),
                                        effective_offset,
                                    ),
                                })
                            })?;
                    // Upper-bound validation: ensure the narrowed view fits
                    // within the source buffer. (#3266)
                    let numel = self.step_numel(step_idx);
                    if numel > 0 {
                        let elem_bytes = dtypes.source_byte_size(step_idx, &self.def.step_metas);
                        let data_bytes = numel.checked_mul(elem_bytes).ok_or_else(|| {
                            TensorError::from(CompiledModelError::DispatchFailed {
                                step_idx,
                                reason: format!(
                                    "NarrowView data size overflow: {numel} * {elem_bytes}",
                                ),
                            })
                        })?;
                        let end = new_offset.checked_add(data_bytes).ok_or_else(|| {
                            TensorError::from(CompiledModelError::DispatchFailed {
                                step_idx,
                                reason: format!(
                                    "NarrowView end overflow: {new_offset} + {data_bytes}",
                                ),
                            })
                        })?;
                        let buf_len = src.buffer().len();
                        if end > buf_len {
                            return Err(CompiledModelError::DispatchFailed {
                                step_idx,
                                reason: format!(
                                    "NarrowView out of bounds: offset={new_offset}, \
                                     data_bytes={data_bytes}, buffer_len={buf_len}",
                                ),
                            }
                            .into());
                        }
                    }
                    buffers[step_idx] = Some(GpuSlice::new(src.buffer().alias(), new_offset));
                    dtypes.propagate_from_source(step_idx, &self.def.step_metas);
                }
                CompiledStep::ConstantValue { .. } => {
                    // Reuse pre-uploaded buffer from construction time (#2338).
                    let buf = self.def.constant_buffers.get(&step_idx).ok_or_else(|| {
                        TensorError::from(CompiledModelError::DispatchFailed {
                            step_idx,
                            reason: "pre-uploaded constant buffer not found".into(),
                        })
                    })?;
                    buffers[step_idx] = Some(GpuSlice::from_ref(buf, 0));
                }
                CompiledStep::Dispatch { kernel, .. } => {
                    if self.def.autocast_active {
                        if let Some(info) =
                            self.def.mixed_gemm_infos.get(step_idx).and_then(|o| o.as_ref())
                        {
                            // Mixed GEMM (#3085 Phase 2): bypass IR dispatch,
                            // use F32 activations × F16 weights → F32 output.
                            // Arm redirect: mixed GEMM returns GpuSlice with offset
                            // (no normalization), so redirect can eliminate blit.
                            let _redirect_guard = helpers::arm_native_op_redirect(
                                planned_offset, &planned_buf, step_sizes, step_idx,
                            );
                            let output =
                                self.execute_mixed_dispatch(cache, info, step_idx, &buffers)?;
                            drop(_redirect_guard);
                            dtypes.set(step_idx, ScalarType::F32);
                            buffers[step_idx] = Some(output);
                        } else {
                            // Non-mixed autocast: cast inputs at F16↔F32 boundaries.
                            let step_dt = self.step_scalar_type(step_idx);
                            let saved = helpers::cast_autocast_inputs(
                                self,
                                cache,
                                step_idx,
                                step_dt,
                                &mut buffers,
                                dtypes.as_slice(),
                            )?;
                            // Arm planned-buffer redirect + skip dispatch
                            // normalization so the output lands directly in the
                            // planned region (#4264).
                            let _redirect_guard = helpers::arm_native_op_redirect(
                                planned_offset, &planned_buf, step_sizes, step_idx,
                            );
                            let _skip_norm = helpers::arm_dispatch_normalization_skip(
                                planned_offset, &planned_buf, step_sizes, step_idx,
                            );
                            let output = self.execute_dispatch(
                                cache,
                                kernel.def(),
                                step_idx,
                                &buffers,
                                &mut dispatch_scratch,
                            )?;
                            drop(_skip_norm);
                            drop(_redirect_guard);
                            helpers::restore_autocast_inputs(&mut buffers, saved);
                            dtypes.set(step_idx, step_dt);
                            buffers[step_idx] = Some(output);
                        }
                    } else {
                        // Arm planned-buffer redirect + skip dispatch
                        // normalization so the output lands directly in the
                        // planned region, eliminating the relocation blit (#4264).
                        let _redirect_guard = helpers::arm_native_op_redirect(
                            planned_offset, &planned_buf, step_sizes, step_idx,
                        );
                        let _skip_norm = helpers::arm_dispatch_normalization_skip(
                            planned_offset, &planned_buf, step_sizes, step_idx,
                        );
                        let output = self.execute_dispatch(
                            cache,
                            kernel.def(),
                            step_idx,
                            &buffers,
                            &mut dispatch_scratch,
                        )?;
                        drop(_skip_norm);
                        drop(_redirect_guard);
                        buffers[step_idx] = Some(output);
                    }
                }
                CompiledStep::NativeOp { op, .. } => {
                    if self.def.mixed_precision_active {
                        helpers::execute_native_op_mixed(
                            self,
                            op,
                            step_idx,
                            &mut buffers,
                            dtypes.as_mut_slice(),
                            cache,
                        )?;
                    } else if self.def.autocast_active {
                        if let Some(info) =
                            self.def.mixed_gemm_infos.get(step_idx).and_then(|o| o.as_ref())
                        {
                            // Mixed GEMM for LinearActivation (#2981): bypass
                            // NativeOp dispatch, use F32×F16→F32+activation kernel.
                            // Arm redirect: mixed GEMM allocates via arena_alloc_or_create.
                            let _redirect_guard = helpers::arm_native_op_redirect(
                                planned_offset, &planned_buf, step_sizes, step_idx,
                            );
                            let output =
                                self.execute_mixed_dispatch(cache, info, step_idx, &buffers)?;
                            drop(_redirect_guard);
                            dtypes.set(step_idx, ScalarType::F32);
                            buffers[step_idx] = Some(output);
                        } else {
                            // Per-op autocast (#3112): cast inputs at F16↔F32
                            // boundaries before NativeOp execution.
                            let step_dt = self.step_scalar_type(step_idx);
                            let saved = helpers::cast_autocast_inputs(
                                self,
                                cache,
                                step_idx,
                                step_dt,
                                &mut buffers,
                                dtypes.as_slice(),
                            )?;
                            // Arm planned-buffer redirect AFTER casts to avoid
                            // the redirect being consumed by a cast allocation
                            // of the same size (#3448). Guard auto-clears on drop.
                            let _redirect_guard = helpers::arm_native_op_redirect(
                                planned_offset, &planned_buf, step_sizes, step_idx,
                            );
                            let output = self.execute_native_op(op, step_idx, &buffers, cache)?;
                            drop(_redirect_guard);
                            helpers::restore_autocast_inputs(&mut buffers, saved);
                            dtypes.set(step_idx, step_dt);
                            buffers[step_idx] = Some(output);
                        }
                    } else {
                        // Arm planned-buffer redirect before NativeOp (#3448).
                        // Guard auto-clears on drop (handles error paths).
                        let _redirect_guard = helpers::arm_native_op_redirect(
                            planned_offset, &planned_buf, step_sizes, step_idx,
                        );
                        let output = self.execute_native_op(op, step_idx, &buffers, cache)?;
                        drop(_redirect_guard);
                        buffers[step_idx] = Some(output);
                    }
                }
                CompiledStep::RuntimeOp { op, .. } => {
                    let output = self.execute_runtime_op(op, step_idx, &buffers)?;
                    buffers[step_idx] = Some(output);
                    // RuntimeOp always produces F32 output (hardcoded DType::F32
                    // in execute_runtime.rs). Override so downstream steps see
                    // the correct dtype. (#3122)
                    dtypes.set(step_idx, ScalarType::F32);
                }
                _ => {
                    return Err(CompiledModelError::DispatchFailed {
                        step_idx,
                        reason: "unsupported compiled step variant".into(),
                    }
                    .into());
                }
            }

            // Profiling: flush GPU after dispatch steps for accurate timing.
            // Flush forces GPU completion so wall time reflects step execution.
            let elapsed_us = if let Some(s) = start {
                if is_gpu_dispatch(step) {
                    crate::gpu_scope::flush()?;
                }
                s.elapsed().as_secs_f64() * 1_000_000.0
            } else {
                0.0
            };

            // Relocate to planned buffer if this step has a planned offset (#2913).
            // Skip the blit when the step wrote directly into the planned buffer
            // (Dispatch via redirect #3448/#4264, NativeOp via redirect #3448) —
            // detected via pointer identity check.
            if let (Some(offset), Some(ref pb)) = (planned_offset, &planned_buf) {
                let size = step_sizes.get(step_idx).copied().unwrap_or(0);
                if size > 0 {
                    if let Some(ref slice) = buffers[step_idx] {
                        let already_in_planned = slice.buffer().is_same_allocation(pb)
                            && slice.byte_offset() == offset;
                        if already_in_planned {
                            // Blit eliminated: step wrote directly into planned
                            // region (#4264). Track for diagnostics.
                            crate::dispatch_stats::TOTAL_BLITS_ELIMINATED.with(|c| {
                                c.set(c.get() + 1);
                            });
                        } else {
                            let relocated = helpers::relocate_to_planned_buffer(
                                pb, slice, offset, size, step_idx,
                            )?;
                            buffers[step_idx] = Some(relocated);
                        }
                    }
                }
            }
            // Restore arena checkpoint to reclaim temporary allocation (#2913).
            crate::arena::restore_default_arena(arena_cp);

            // Eager buffer release: drop intermediates whose last
            // consumer is the current step. Output buffers are excluded
            // during release_at construction (at model build time).
            if let Some(to_release) = release_at.get(step_idx) {
                for &prior in to_release {
                    buffers[prior] = None;
                }
            }

            // Build profiling entry for this step.
            if profile {
                let numel = self.step_numel(step_idx);
                // DtypeTracker always returns the correct dtype: dynamically
                // updated when tracking is on, step_scalar_types when off. (#3020)
                let actual_dtype = dtypes.get(step_idx);
                let elem_bytes = actual_dtype.byte_size();
                step_profiles.push(StepProfile {
                    step_idx,
                    step_name: step_name(step),
                    wall_time_us: elapsed_us,
                    is_gpu_dispatch: is_gpu_dispatch(step),
                    output_bytes: numel * elem_bytes,
                });
            }
        }

        let exec_profile = if profile {
            Some(ExecutionProfile::new(step_profiles))
        } else {
            None
        };
        Ok((buffers, dtypes.into_inner(), exec_profile))
    }
}

// ICB replay methods extracted for 500-line compliance. Part of #3259 (D4).
#[path = "compiled_model_execute_icb_replay.rs"]
mod icb_replay;
