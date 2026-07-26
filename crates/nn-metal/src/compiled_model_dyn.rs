// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! DynTensor interface for `CompiledModel`.
//!
//! Extracted from `compiled_model.rs` to keep files under 450 lines.
//! Contains `execute_dyn`, `execute_dyn_outputs`, input validation,
//! and DynTensor/MetalBuffer conversion helpers.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::check_output_finite;
use nn_core::{DType, Device, Result, TensorError};

use crate::cache::PipelineCache;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::gpu_slice::GpuSlice;

use super::{CompiledModel, CompiledModelError};

impl CompiledModel {
    /// Validate that DynTensor inputs match the traced graph's expected
    /// shapes and dtypes.
    ///
    /// Under `ShapePolicy::Fixed`, shapes must match exactly.
    /// Under `ShapePolicy::Polymorphic`, structural dimensions must match
    /// exactly while sequence dimensions may be smaller than the compiled
    /// maximum. Part of #3873.
    fn validate_dyn_inputs(&self, inputs: &[&DynTensor]) -> Result<()> {
        if inputs.len() != self.def.input_specs.len() {
            return Err(CompiledModelError::InputCountMismatch {
                expected: self.def.input_specs.len(),
                got: inputs.len(),
            }
            .into());
        }
        for (i, (t, (expected_shape, expected_dtype))) in
            inputs.iter().zip(self.def.input_specs.iter()).enumerate()
        {
            // Shape validation respects the shape policy.
            self.def.shape_policy
                .validate_shape(expected_shape, t.dims(), i)
                .map_err(|_| {
                    TensorError::from(CompiledModelError::ShapeMismatch {
                        index: i,
                        expected: expected_shape.clone(),
                        got: t.dims().to_vec(),
                    })
                })?;
            if t.dtype() != *expected_dtype {
                return Err(CompiledModelError::DtypeMismatch {
                    index: i,
                    expected: *expected_dtype,
                    got: t.dtype(),
                }
                .into());
            }
        }
        Ok(())
    }

    /// Execute the compiled plan using `DynTensor` inputs and output.
    ///
    /// This is the primary execution API for most consumers. It validates
    /// input shapes/dtypes, extracts GPU slices, runs the compiled plan,
    /// and wraps the result as a `DynTensor`. For the low-level Metal
    /// buffer API, see [`execute`](Self::execute).
    ///
    /// Extracts `GpuSlice` handles from each GPU-resident input tensor,
    /// preserving byte offsets from narrow/view tensors. See #2268.
    ///
    /// # Errors
    ///
    /// Returns an error if any input is not a GPU tensor, if input count
    /// is wrong, if shape/dtype mismatches, or if any dispatch step fails.
    pub fn execute_dyn(&self, cache: &PipelineCache, inputs: &[&DynTensor]) -> Result<DynTensor> {
        self.validate_dyn_inputs(inputs)?;
        let input_slices = self.extract_gpu_slices(inputs)?;
        let out_buf = self.execute_from_slices(cache, &input_slices)?;
        // Under polymorphic mode, resolve actual output shape from inputs.
        // Under fixed mode, this returns the compile-time shape unchanged.
        let (shape, dtype) = self.resolve_output_shape(inputs);
        let storage = MetalTensorData::new(out_buf);
        let output = DynTensor::from_gpu_storage(shape, dtype, Arc::new(storage), Device::metal())?;
        check_output_finite(&output, "CompiledModel")?;
        Ok(output)
    }

    /// Execute the compiled plan and return all marked output tensors.
    ///
    /// For multi-output models (encoder-decoder, LSTM hidden+cell), returns
    /// one `DynTensor` per output node in the order they were marked.
    /// For single-output models, returns a `Vec` with one element.
    pub fn execute_dyn_outputs(
        &self,
        cache: &PipelineCache,
        inputs: &[&DynTensor],
    ) -> Result<Vec<DynTensor>> {
        self.validate_dyn_inputs(inputs)?;
        let input_slices = self.extract_gpu_slices(inputs)?;
        let out_bufs = self.execute_outputs_from_slices(cache, &input_slices)?;
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
        for (i, output) in outputs.iter().enumerate() {
            check_output_finite(output, &format!("CompiledModel output {i}"))?;
        }
        Ok(outputs)
    }

    /// Execute without flush fence or NaN check. Caller manages sync.
    ///
    /// Same as [`execute_dyn`](Self::execute_dyn) but skips `with_gpu_scope`
    /// and `check_output_finite`. GPU work is encoded into the lazy batch but
    /// NOT committed. Call `flush()` or `sync()` before reading output data.
    ///
    /// # Error cleanup
    ///
    /// On error, stale GPU commands may remain in the thread-local lazy batch.
    /// Callers MUST call [`gpu_scope::discard_pending_batch()`](crate::gpu_scope::discard_pending_batch)
    /// on the error path to prevent these commands from contaminating the next
    /// dispatch scope on this thread.
    ///
    /// For multi-segment pipelines where CPU-GPU overlap is desired (#2375,
    /// #2619). Use `submit()` between segments for non-blocking pipelining.
    pub fn execute_dyn_no_fence(
        &self,
        cache: &PipelineCache,
        inputs: &[&DynTensor],
    ) -> Result<DynTensor> {
        self.validate_dyn_inputs(inputs)?;
        let input_slices = self.extract_gpu_slices(inputs)?;
        let out_buf = self.execute_from_slices_no_fence(cache, &input_slices)?;
        let (shape, dtype) = self.primary_output_meta();
        let storage = MetalTensorData::new(out_buf);
        DynTensor::from_gpu_storage(shape, dtype, Arc::new(storage), Device::metal())
    }

    /// Execute multi-output without flush fence or NaN check. Caller manages sync.
    ///
    /// Multi-output variant of [`execute_dyn_no_fence`](Self::execute_dyn_no_fence).
    /// Same error cleanup requirement applies — see its documentation.
    pub fn execute_dyn_outputs_no_fence(
        &self,
        cache: &PipelineCache,
        inputs: &[&DynTensor],
    ) -> Result<Vec<DynTensor>> {
        self.validate_dyn_inputs(inputs)?;
        let input_slices = self.extract_gpu_slices(inputs)?;
        let out_bufs = self.execute_outputs_from_slices_no_fence(cache, &input_slices)?;
        out_bufs
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
            .collect()
    }

    /// Execute with per-step profiling.
    ///
    /// Returns `(output, profile)` where `profile` contains wall-clock
    /// timing for each compiled step. GPU work is flushed after each
    /// dispatch step to get accurate per-step timing.
    ///
    /// **Performance note:** Profiling disables lazy GPU batching —
    /// each dispatch step gets its own flush. This adds per-step overhead
    /// (~5-20 us per flush) but is necessary for accurate bottleneck
    /// identification. Do not use in production hot paths.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use nn_metal::compiled_model::profile::ExecutionProfile;
    /// let (output, profile) = compiled.execute_dyn_profiled(&cache, &[&input])?;
    /// println!("{profile}");  // prints top 10 slowest steps
    /// for step in profile.slowest_steps(5) {
    ///     println!("  step {}: {} ({:.1} us)", step.step_idx, step.step_name, step.wall_time_us);
    /// }
    /// ```
    pub fn execute_dyn_profiled(
        &self,
        cache: &PipelineCache,
        inputs: &[&DynTensor],
    ) -> Result<(DynTensor, super::profile::ExecutionProfile)> {
        self.validate_dyn_inputs(inputs)?;
        let input_slices = self.extract_gpu_slices(inputs)?;
        let (out_buf, profile) = crate::gpu_scope::with_gpu_scope(|| {
            self.execute_primary_output_profiled(cache, &input_slices)
        })?;
        let (shape, dtype) = self.primary_output_meta();
        let storage = MetalTensorData::new(out_buf);
        let output = DynTensor::from_gpu_storage(shape, dtype, Arc::new(storage), Device::metal())?;
        check_output_finite(&output, "CompiledModel")?;
        Ok((output, profile))
    }

    /// Extract `GpuSlice` from DynTensor inputs, preserving byte offsets.
    ///
    /// Uses `MetalTensorData::as_gpu_slice()` which includes the byte offset
    /// from narrow/view tensors. Fixes #2268 — the prior `extract_metal_bufs`
    /// returned bare `&MetalBuffer` which dropped the offset.
    fn extract_gpu_slices(&self, inputs: &[&DynTensor]) -> Result<Vec<GpuSlice>> {
        inputs
            .iter()
            .enumerate()
            .map(|(i, t)| {
                // Fast path: tensor already on GPU.
                if let Ok(data) = t.gpu_data::<MetalTensorData>() {
                    return Ok(data.as_gpu_slice());
                }
                // Slow path: CPU tensor — transfer to GPU once (#2567).
                let gpu_t = t
                    .to_device(&Device::metal())
                    .map_err(|_| TensorError::from(CompiledModelError::InputNotGpu { index: i }))?;
                let data = gpu_t
                    .gpu_data::<MetalTensorData>()
                    .map_err(|_| TensorError::from(CompiledModelError::InputNotGpu { index: i }))?;
                Ok(data.as_gpu_slice())
            })
            .collect()
    }

    /// Execute the compiled plan and return a [`GpuFuture`] with the output.
    ///
    /// Encodes all GPU work into the lazy batch, then submits non-blocking via
    /// [`GpuFuture::submit_current`]. The returned [`AsyncGpuResult`] contains
    /// both the future handle and the output `DynTensor`. The tensor's GPU
    /// buffers contain valid data only after `future.wait()` or
    /// `future.is_complete()` returns `true`.
    ///
    /// This is the async counterpart to [`execute_dyn`](Self::execute_dyn).
    /// Use when the CPU has productive work to do while the GPU executes
    /// (streaming, pipelining, concurrent model execution).
    ///
    /// NaN checking is deferred — the caller should run
    /// [`check_output_finite`](nn_core::layers::check_output_finite) after
    /// waiting on the future.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let async_result = compiled.execute_dyn_async(&cache, &[&input])?;
    /// // CPU does other work while GPU executes...
    /// async_result.future.wait()?;
    /// // Now safe to read async_result.value
    /// ```
    pub fn execute_dyn_async(
        &self,
        cache: &PipelineCache,
        inputs: &[&DynTensor],
    ) -> Result<crate::gpu_future::AsyncGpuResult<DynTensor>> {
        self.validate_dyn_inputs(inputs)?;
        let input_slices = self.extract_gpu_slices(inputs)?;
        // Encode without fence — GPU work goes into lazy batch.
        let out_buf = self.execute_from_slices_no_fence(cache, &input_slices)?;
        let (shape, dtype) = self.primary_output_meta();
        let storage = MetalTensorData::new(out_buf);
        let output = DynTensor::from_gpu_storage(shape, dtype, Arc::new(storage), Device::metal())?;

        // Submit the lazy batch non-blocking.
        let future = crate::gpu_future::GpuFuture::submit_current()?
            .ok_or_else(|| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::DispatchFailed,
                    "execute_dyn_async: no lazy batch to submit after encoding".to_string(),
                )
            })?;

        Ok(crate::gpu_future::AsyncGpuResult {
            future,
            value: output,
        })
    }

    /// Primary output shape and dtype (last output, backward compat).
    pub(super) fn primary_output_meta(&self) -> (Vec<usize>, DType) {
        self.def.output_metas
            .last()
            .map(|(s, d)| (s.clone(), *d))
            .unwrap_or_else(|| (Vec::new(), DType::F32))
    }

    /// Resolve the output shape, accounting for polymorphic shape policy.
    ///
    /// Under `ShapePolicy::Fixed`, returns the compile-time output shape.
    ///
    /// Under `ShapePolicy::Polymorphic`, computes the actual output shape
    /// by scaling sequence dimensions proportionally based on the ratio of
    /// actual-to-compiled input sequence dimensions. For example, if the
    /// model was compiled with seq_len=128 and the actual input has
    /// seq_len=64, the output's sequence dimension is halved.
    ///
    /// Falls back to the compile-time shape when inputs are empty or when
    /// the input sequence dimensions match the compiled dimensions exactly.
    ///
    /// Part of #3873.
    fn resolve_output_shape(&self, inputs: &[&DynTensor]) -> (Vec<usize>, DType) {
        let (compiled_shape, dtype) = self.primary_output_meta();

        if self.def.shape_policy.is_fixed() || inputs.is_empty() {
            return (compiled_shape, dtype);
        }

        // Find the first input with a sequence dimension (rank >= 2).
        // Compute the ratio of actual/compiled for the sequence dim.
        let seq_ratio = self.def.input_specs.iter()
            .zip(inputs.iter())
            .find_map(|((expected_shape, _), actual)| {
                if expected_shape.len() >= 2 {
                    let compiled_seq = *expected_shape.last().unwrap_or(&1);
                    let actual_seq = *actual.dims().last().unwrap_or(&1);
                    if compiled_seq > 0 && actual_seq != compiled_seq {
                        Some((actual_seq, compiled_seq))
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

        match seq_ratio {
            Some((actual_seq, compiled_seq)) => {
                // Scale the output shape's sequence dimension(s) proportionally.
                let mut output_shape = compiled_shape;
                if output_shape.len() >= 2 {
                    // Last dim is the sequence dimension.
                    let last = output_shape.len() - 1;
                    let compiled_out_seq = output_shape[last];
                    // Scale: out_seq = compiled_out_seq * actual_seq / compiled_seq
                    // Use integer math to avoid floating-point rounding.
                    output_shape[last] =
                        (compiled_out_seq * actual_seq).div_ceil(compiled_seq);
                }
                (output_shape, dtype)
            }
            None => (compiled_shape, dtype),
        }
    }

    /// Returns the shape policy for this compiled model.
    #[must_use]
    pub fn shape_policy(&self) -> &super::ShapePolicy {
        &self.def.shape_policy
    }
}
