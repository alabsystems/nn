// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Non-blocking GPU inference for [`CompiledModel`].
//!
//! Provides [`InFlightInference`] — a handle representing GPU work that has
//! been submitted but not yet waited on. This enables **double-buffered
//! inference**: the CPU can prepare the next inference's inputs while the GPU
//! executes the current one.
//!
//! # Usage Pattern
//!
//! ```rust,no_run
//! use nn_metal::compiled_model::{CompiledModel, InFlightInference};
//! use nn_metal::PipelineCache;
//!
//! let cache = PipelineCache::new();
//! // Submit inference N — GPU starts immediately, CPU continues.
//! let in_flight = compiled.execute_dyn_submit(&cache, &[&input_n])?;
//!
//! // ... CPU prepares input N+1 while GPU runs inference N ...
//!
//! // Collect inference N results (blocks until GPU completes).
//! let output_n = in_flight.wait()?;
//!
//! // Submit inference N+1 — GPU immediately starts on next batch.
//! let in_flight2 = compiled.execute_dyn_submit(&cache, &[&input_n1])?;
//! ```
//!
//! # Performance
//!
//! For sequential single-inference workloads, the overhead is negligible
//! (one extra `submit` call vs. `flush`). For streaming/batch workloads,
//! this eliminates the CPU idle gap between GPU completion and next GPU
//! submission — the GPU is never waiting for the CPU to prepare inputs.
//!
//! Measured benefit depends on the ratio of CPU prep time to GPU execution
//! time. When CPU prep >= GPU execution, this achieves near-2x throughput
//! by fully overlapping the two.
//!
//! Part of #4106.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::check_output_finite;
use nn_core::{DType, Device, Result};

use crate::cache::PipelineCache;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::gpu_fence::GpuFence;

use super::CompiledModel;

/// Handle to GPU inference work that has been submitted but not waited on.
///
/// Created by [`CompiledModel::execute_dyn_submit`]. The GPU is executing
/// the compiled plan asynchronously. Call [`wait`](Self::wait) to block
/// until completion and retrieve the output tensor.
///
/// # Safety
///
/// The output `DynTensor` wraps a GPU buffer that is being written by the
/// submitted command buffer. Reading the tensor's data before `wait()`
/// returns produces undefined results. The `DynTensor` handle itself is
/// safe to hold — it is a metadata wrapper that does not access GPU memory
/// until `to_device(&cpu())` or similar readback is called.
///
/// # Arena interaction
///
/// [`GpuFence::wait`] does NOT reset the activation arena. This is
/// intentional: the caller may hold multiple in-flight inferences sharing
/// arena generations. The arena is reset on the next `flush()` or
/// `sync()` call, or when a new `with_gpu_scope` exits.
pub struct InFlightInference {
    /// The output tensor (GPU-resident, data not yet valid until GPU completes).
    output: DynTensor,
    /// Fence handle for the submitted GPU work. `None` if no GPU work was
    /// pending at submit time (e.g., all ops were metadata-only).
    fence: Option<GpuFence>,
    /// Output shape for validation.
    output_shape: Vec<usize>,
    /// Output dtype for validation.
    output_dtype: DType,
}

impl InFlightInference {
    /// Block until the GPU work completes and return the output tensor.
    ///
    /// After this call, the output tensor's GPU buffer contains valid data.
    /// A NaN/Inf check is performed on the output before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU command buffer failed, timed out, or
    /// if the output contains non-finite values.
    #[must_use = "returns a Result that may contain an error"]
    pub fn wait(self) -> Result<DynTensor> {
        if let Some(fence) = self.fence {
            fence.wait()?;
        }
        check_output_finite(&self.output, "CompiledModel::InFlightInference")?;
        Ok(self.output)
    }

    /// Check if the GPU work has completed without blocking.
    ///
    /// Returns `true` if the GPU has finished executing the submitted
    /// command buffer, or if no GPU work was pending. After this returns
    /// `true`, [`wait`](Self::wait) will return immediately.
    pub fn is_completed(&self) -> bool {
        self.fence.as_ref().map_or(true, GpuFence::is_completed)
    }

    /// Access the output shape that will be returned by [`wait`].
    #[must_use]
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Access the output dtype that will be returned by [`wait`].
    #[must_use]
    pub fn output_dtype(&self) -> DType {
        self.output_dtype
    }
}

impl std::fmt::Debug for InFlightInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InFlightInference")
            .field("output_shape", &self.output_shape)
            .field("output_dtype", &self.output_dtype)
            .field("is_completed", &self.is_completed())
            .finish()
    }
}

/// Multi-output variant of [`InFlightInference`].
///
/// Created by [`CompiledModel::execute_dyn_outputs_submit`]. Holds multiple
/// output tensors from a multi-output model (e.g., encoder-decoder).
pub struct InFlightMultiOutput {
    /// Output tensors (GPU-resident, data not yet valid until GPU completes).
    outputs: Vec<DynTensor>,
    /// Fence handle for the submitted GPU work.
    fence: Option<GpuFence>,
}

impl InFlightMultiOutput {
    /// Block until the GPU work completes and return all output tensors.
    ///
    /// NaN/Inf checks are performed on each output.
    #[must_use = "returns a Result that may contain an error"]
    pub fn wait(self) -> Result<Vec<DynTensor>> {
        if let Some(fence) = self.fence {
            fence.wait()?;
        }
        for (i, output) in self.outputs.iter().enumerate() {
            check_output_finite(output, &format!("CompiledModel::InFlightMultiOutput[{i}]"))?;
        }
        Ok(self.outputs)
    }

    /// Check if the GPU work has completed without blocking.
    pub fn is_completed(&self) -> bool {
        self.fence.as_ref().map_or(true, GpuFence::is_completed)
    }

    /// Number of output tensors.
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }
}

impl std::fmt::Debug for InFlightMultiOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InFlightMultiOutput")
            .field("num_outputs", &self.outputs.len())
            .field("is_completed", &self.is_completed())
            .finish()
    }
}

impl CompiledModel {
    /// Execute the compiled plan and submit GPU work non-blocking.
    ///
    /// Encodes all dispatch steps into the lazy batch, submits the batch
    /// to the GPU via [`GpuFence::submit_current`], and returns an
    /// [`InFlightInference`] handle immediately. The GPU executes
    /// asynchronously while the CPU continues.
    ///
    /// This is the non-blocking counterpart to [`execute_dyn`](Self::execute_dyn).
    /// Use it when the CPU has useful work to do while the GPU runs
    /// (e.g., preparing the next inference's inputs, post-processing
    /// the previous inference's outputs).
    ///
    /// # Error cleanup
    ///
    /// On encoding error, the pending lazy batch is discarded (same as
    /// `execute_dyn_no_fence`). If submission succeeds but `wait()` later
    /// fails, the GPU command buffer error is propagated.
    ///
    /// Part of #4106.
    pub fn execute_dyn_submit(
        &self,
        cache: &PipelineCache,
        inputs: &[&DynTensor],
    ) -> Result<InFlightInference> {
        self.validate_dyn_inputs(inputs)?;
        let input_slices = self.extract_gpu_slices(inputs)?;

        // Encode all dispatch steps into the lazy batch (no flush).
        let out_buf = match self.execute_from_slices_no_fence(cache, &input_slices) {
            Ok(buf) => buf,
            Err(e) => {
                crate::gpu_scope::discard_pending_batch();
                return Err(e);
            }
        };

        // Submit the lazy batch to GPU — non-blocking.
        let fence = match GpuFence::submit_current() {
            Ok(f) => f,
            Err(e) => {
                crate::gpu_scope::discard_pending_batch();
                return Err(e);
            }
        };

        let (shape, dtype) = self.primary_output_meta();
        let storage = MetalTensorData::new(out_buf);
        let output = DynTensor::from_gpu_storage(
            shape.clone(),
            dtype,
            Arc::new(storage),
            Device::metal(),
        )?;

        Ok(InFlightInference {
            output,
            fence,
            output_shape: shape,
            output_dtype: dtype,
        })
    }

    /// Execute multi-output and submit GPU work non-blocking.
    ///
    /// Non-blocking counterpart to [`execute_dyn_outputs`](Self::execute_dyn_outputs).
    /// Returns an [`InFlightMultiOutput`] handle for all marked output tensors.
    ///
    /// Part of #4106.
    pub fn execute_dyn_outputs_submit(
        &self,
        cache: &PipelineCache,
        inputs: &[&DynTensor],
    ) -> Result<InFlightMultiOutput> {
        self.validate_dyn_inputs(inputs)?;
        let input_slices = self.extract_gpu_slices(inputs)?;

        let out_bufs = match self.execute_outputs_from_slices_no_fence(cache, &input_slices) {
            Ok(bufs) => bufs,
            Err(e) => {
                crate::gpu_scope::discard_pending_batch();
                return Err(e);
            }
        };

        let fence = match GpuFence::submit_current() {
            Ok(f) => f,
            Err(e) => {
                crate::gpu_scope::discard_pending_batch();
                return Err(e);
            }
        };

        let outputs: Vec<DynTensor> = out_bufs
            .into_iter()
            .enumerate()
            .map(|(i, buf)| {
                let (shape, dtype) = &self.def.output_metas[i];
                let storage = MetalTensorData::new(buf);
                DynTensor::from_gpu_storage(
                    shape.clone(),
                    *dtype,
                    Arc::new(storage),
                    Device::metal(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(InFlightMultiOutput { outputs, fence })
    }
}
