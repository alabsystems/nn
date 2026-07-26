// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Execution methods for `CompiledModel`.
//!
//! Extracted from `compiled_model.rs` for 450-line compliance.
//! Contains `execute`, `run_steps`, and entry points. `execute_dispatch` and
//! `resolve_input_slice` are in `compiled_model_execute_dispatch.rs`.

use nn_core::Result;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;

use super::{CompiledModel, CompiledModelError};

#[path = "compiled_model_execute_helpers.rs"]
mod helpers;

#[path = "compiled_model_execute_native_encoding.rs"]
pub(super) mod encoding;

#[path = "compiled_model_kernel_spec.rs"]
pub(crate) mod kernel_spec;

#[path = "compiled_model_execute_native.rs"]
mod native_ops;

#[path = "compiled_model_execute_outputs.rs"]
mod outputs;

#[path = "compiled_model_execute_runtime.rs"]
mod runtime_ops;

#[path = "compiled_model_execute_profiled.rs"]
mod profiled;

#[path = "compiled_model_execute_dispatch.rs"]
mod dispatch_impl;

#[path = "compiled_model_execute_mixed.rs"]
mod mixed_dispatch;

#[path = "compiled_model_execute_steps.rs"]
mod steps;

impl CompiledModel {
    /// Execute the compiled plan with raw Metal buffers.
    ///
    /// This is the low-level API for callers that manage Metal buffers
    /// directly. Most consumers should use [`execute_dyn`](Self::execute_dyn)
    /// which accepts and returns `DynTensor` values with shape/dtype
    /// validation and NaN checking.
    #[doc(hidden)]
    pub fn execute(&self, cache: &PipelineCache, inputs: &[&MetalBuffer]) -> Result<MetalBuffer> {
        let input_slices: Vec<GpuSlice> = inputs.iter().map(|b| GpuSlice::from_ref(b, 0)).collect();
        self.execute_from_slices(cache, &input_slices)
    }

    /// Validate inputs before execution.
    fn validate_slice_inputs(&self, inputs: &[GpuSlice]) -> Result<()> {
        if inputs.len() != self.def.num_inputs {
            return Err(CompiledModelError::InputCountMismatch {
                expected: self.def.num_inputs,
                got: inputs.len(),
            }
            .into());
        }
        if self.def.steps.is_empty() {
            return Err(CompiledModelError::EmptyPlan.into());
        }
        Ok(())
    }

    /// Core execution: run steps and extract primary output buffer.
    fn execute_primary_output(
        &self,
        cache: &PipelineCache,
        inputs: &[GpuSlice],
    ) -> Result<MetalBuffer> {
        let (buffers, buffer_dtypes) = self.run_steps(cache, inputs)?;
        let primary_idx = self
            .def
            .output_step_indices
            .last()
            .copied()
            .unwrap_or(self.def.steps.len().saturating_sub(1));
        self.extract_output_buffer(
            cache,
            &buffers,
            &buffer_dtypes,
            primary_idx,
            self.output_shape(),
            self.output_dtype(),
        )
    }

    /// Execute from `GpuSlice` inputs with a flush fence at scope exit (#2268).
    pub(super) fn execute_from_slices(
        &self,
        cache: &PipelineCache,
        inputs: &[GpuSlice],
    ) -> Result<MetalBuffer> {
        self.validate_slice_inputs(inputs)?;
        crate::gpu_scope::with_gpu_scope(|| self.execute_primary_output(cache, inputs))
    }

    /// Execute from `GpuSlice` inputs without a flush fence (#2375).
    pub(super) fn execute_from_slices_no_fence(
        &self,
        cache: &PipelineCache,
        inputs: &[GpuSlice],
    ) -> Result<MetalBuffer> {
        self.validate_slice_inputs(inputs)?;
        self.execute_primary_output(cache, inputs)
    }

    /// Run all compiled steps and return the buffer table + runtime dtypes.
    ///
    /// Delegates to [`run_steps_inner`](Self::run_steps_inner) with profiling
    /// disabled.
    pub(super) fn run_steps(
        &self,
        cache: &PipelineCache,
        inputs: &[GpuSlice],
    ) -> Result<(Vec<Option<GpuSlice>>, Vec<nn_dsl::ir::ScalarType>)> {
        let (bufs, dtypes, _) = self.run_steps_inner(cache, inputs, false)?;
        Ok((bufs, dtypes))
    }
}
