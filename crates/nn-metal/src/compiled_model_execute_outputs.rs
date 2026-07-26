// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-output execution methods and output extraction for `CompiledModel`.
//!
//! Contains `execute_outputs`, `execute_all_outputs`, and the shared
//! `extract_output_buffer` helper used by all 3 output paths (single,
//! profiled, multi-output).

use nn_core::{DType, Result, TensorError};
use nn_dsl::ir::ScalarType;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;

use super::{helpers, CompiledModel, CompiledModelError};

impl CompiledModel {
    /// Execute the compiled plan and return all marked output buffers.
    #[doc(hidden)]
    pub fn execute_outputs(
        &self,
        cache: &PipelineCache,
        inputs: &[&MetalBuffer],
    ) -> Result<Vec<MetalBuffer>> {
        let input_slices: Vec<GpuSlice> = inputs.iter().map(|b| GpuSlice::from_ref(b, 0)).collect();
        self.execute_outputs_from_slices(cache, &input_slices)
    }

    /// Core execution: run steps and collect all marked output buffers.
    fn execute_all_outputs(
        &self,
        cache: &PipelineCache,
        inputs: &[GpuSlice],
    ) -> Result<Vec<MetalBuffer>> {
        let (buffers, buffer_dtypes) = self.run_steps(cache, inputs)?;
        self.def.output_step_indices
            .iter()
            .zip(self.def.output_metas.iter())
            .map(|(&idx, (shape, dtype))| {
                self.extract_output_buffer(cache, &buffers, &buffer_dtypes, idx, shape, *dtype)
            })
            .collect()
    }

    /// Extract a single output buffer from the step buffer table.
    ///
    /// Handles mixed-precision / autocast F16→F32 casting, mixed GEMM
    /// detection (skip cast for F32 accumulator outputs), and offset
    /// normalization. Used by `execute_primary_output`, its profiled variant,
    /// and `execute_all_outputs`.
    ///
    /// Part of D2 (designs/2026-03-22-unified-execute-core-dedup.md).
    pub(super) fn extract_output_buffer(
        &self,
        cache: &PipelineCache,
        buffers: &[Option<GpuSlice>],
        buffer_dtypes: &[ScalarType],
        output_idx: usize,
        output_shape: &[usize],
        output_dtype: DType,
    ) -> Result<MetalBuffer> {
        let slice = buffers
            .get(output_idx)
            .and_then(|opt| opt.as_ref())
            .ok_or_else(|| TensorError::from(CompiledModelError::EmptyPlan))?;

        // Mixed GEMM steps already produce F32 (float accumulator), so
        // skip the cast even though step_scalar_types says F16. (#2981)
        let is_mixed_gemm = self
            .def.mixed_gemm_infos
            .get(output_idx)
            .map_or(false, Option::is_some);

        // Use runtime buffer_dtypes (which propagate through passthrough
        // steps) rather than build-time step_scalar_types. (#2981)
        let runtime_dt = if !buffer_dtypes.is_empty() {
            buffer_dtypes
                .get(output_idx)
                .copied()
                .unwrap_or(ScalarType::F32)
        } else {
            self.step_scalar_type(output_idx)
        };

        let output_is_f16 = self.def.mixed_precision_active
            || (self.def.autocast_active && !is_mixed_gemm && runtime_dt == ScalarType::F16);

        let slice_ref;
        let f32_slice;
        if output_is_f16 {
            // Use effective_numel: RuntimeOp buffers use buffer geometry,
            // planned-buffer steps use pre-computed trace-time numel. (#3121)
            let n = self.effective_numel(output_idx, slice, ScalarType::F16);
            f32_slice =
                helpers::cast_slice_dtype(cache, slice, n, ScalarType::F16, ScalarType::F32)?;
            slice_ref = &f32_slice;
            crate::gpu_scope::get_or_create_batch()?;
        } else {
            slice_ref = slice;
        }

        let numel = crate::metal_backend::checked_dim_product(output_shape).map_err(|e| {
            TensorError::from(CompiledModelError::DispatchFailed {
                step_idx: output_idx,
                reason: format!("output shape overflow: {e}"),
            })
        })?;
        let elem_bytes = output_dtype.size_bytes();
        let out_bytes = numel.checked_mul(elem_bytes).ok_or_else(|| {
            TensorError::from(CompiledModelError::DispatchFailed {
                step_idx: output_idx,
                reason: format!("output byte count overflow for shape {output_shape:?}"),
            })
        })?;
        helpers::normalize_output_to_offset_zero(cache, slice_ref, output_idx, out_bytes)
    }

    /// Execute and return all outputs with a flush fence (#2268).
    pub(in crate::compiled_model) fn execute_outputs_from_slices(
        &self,
        cache: &PipelineCache,
        inputs: &[GpuSlice],
    ) -> Result<Vec<MetalBuffer>> {
        self.validate_slice_inputs(inputs)?;
        crate::gpu_scope::with_gpu_scope(|| self.execute_all_outputs(cache, inputs))
    }

    /// Execute and return all outputs without a flush fence (#2375).
    pub(in crate::compiled_model) fn execute_outputs_from_slices_no_fence(
        &self,
        cache: &PipelineCache,
        inputs: &[GpuSlice],
    ) -> Result<Vec<MetalBuffer>> {
        self.validate_slice_inputs(inputs)?;
        self.execute_all_outputs(cache, inputs)
    }
}
