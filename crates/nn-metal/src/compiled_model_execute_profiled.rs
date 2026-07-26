// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Profiled execution for `CompiledModel`.
//!
//! Thin wrappers around `run_steps_inner(profile=true)`. The step execution
//! loop lives in `compiled_model_execute_steps.rs` — this file only adds
//! the profiled entry point and output extraction.
//!
//! Part of #2257, #2981.

use nn_core::Result;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::compiled_model::profile::ExecutionProfile;
use crate::gpu_slice::GpuSlice;

use super::CompiledModel;

impl CompiledModel {
    /// Core profiled execution: runs steps with per-step timing and extracts
    /// the primary output buffer.
    ///
    /// Visible to sibling modules in `compiled_model` for the DynTensor entry
    /// point in `compiled_model_dyn.rs`.
    pub(in crate::compiled_model) fn execute_primary_output_profiled(
        &self,
        cache: &PipelineCache,
        inputs: &[GpuSlice],
    ) -> Result<(MetalBuffer, ExecutionProfile)> {
        let (buffers, buffer_dtypes, profile) = self.run_steps_profiled(cache, inputs)?;
        let primary_idx = self
            .def.output_step_indices
            .last()
            .copied()
            .unwrap_or(self.def.steps.len().saturating_sub(1));
        let out_buf = self.extract_output_buffer(
            cache,
            &buffers,
            &buffer_dtypes,
            primary_idx,
            self.output_shape(),
            self.output_dtype(),
        )?;
        Ok((out_buf, profile))
    }

    /// Run all compiled steps with per-step wall-clock profiling.
    ///
    /// Delegates to [`run_steps_inner`](Self::run_steps_inner) with profiling
    /// enabled. Same step logic as `run_steps` — single source of truth.
    fn run_steps_profiled(
        &self,
        cache: &PipelineCache,
        inputs: &[GpuSlice],
    ) -> Result<(
        Vec<Option<GpuSlice>>,
        Vec<nn_dsl::ir::ScalarType>,
        ExecutionProfile,
    )> {
        let (bufs, dtypes, prof) = self.run_steps_inner(cache, inputs, true)?;
        Ok((bufs, dtypes, prof.expect("profiling was requested")))
    }
}
