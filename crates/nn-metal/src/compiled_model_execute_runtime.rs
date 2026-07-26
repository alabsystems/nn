// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Runtime operation execution for `CompiledModel`.
//!
//! Extracted from `compiled_model_execute.rs` to keep files under 450 lines.
//! Contains `execute_runtime_op` and per-variant helpers for data-dependent
//! operations whose output shapes cannot be determined at compile time.
//!
//! Part of #2234 (RuntimeOp for data-dependent ops).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result};
use nn_dsl::RuntimeOpKind;

use crate::dyn_tensor_metal::MetalTensorData;
use crate::gpu_slice::GpuSlice;

use super::helpers::native_dispatch_err;
use super::{CompiledModel, CompiledModelError};

impl CompiledModel {
    /// Execute a `RuntimeOp` step by running the operation eagerly.
    ///
    /// Flushes pending GPU work before execution because runtime ops
    /// may need CPU readback of input data (e.g., reading repeat counts).
    pub(super) fn execute_runtime_op(
        &self,
        op: &RuntimeOpKind,
        step_idx: usize,
        buffers: &[Option<GpuSlice>],
    ) -> Result<GpuSlice> {
        // Flush pending GPU work — runtime ops may need CPU readback
        // of input tensors (e.g., repeat counts). Without flush, the
        // input buffers may not contain committed data. See design doc:
        // "flush() before CPU readback commits pending work."
        crate::gpu_scope::flush()?;

        match op {
            RuntimeOpKind::RepeatInterleave {
                dim,
                input_shape,
                counts_shape,
            } => execute_runtime_repeat_interleave(
                self,
                step_idx,
                buffers,
                *dim,
                input_shape,
                counts_shape,
            ),
            _ => Err(CompiledModelError::DispatchFailed {
                step_idx,
                reason: "unsupported RuntimeOp variant".into(),
            }
            .into()),
        }
    }
}

/// Execute a `RuntimeOpKind::RepeatInterleave` step.
///
/// Resolves both inputs (tensor + counts) from the buffer table,
/// wraps them as `DynTensor`, calls `DynTensor::repeat_interleave`,
/// and extracts the output `GpuSlice`.
///
/// The output shape is data-dependent — it depends on the actual
/// repeat count values, which are runtime-variable (e.g., Kokoro
/// duration predictor output).
fn execute_runtime_repeat_interleave(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    dim: usize,
    input_shape: &[usize],
    counts_shape: &[usize],
) -> Result<GpuSlice> {
    // Resolve input 0 (the tensor to repeat) from the edge map.
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    // Resolve input 1 (the repeat counts) from the edge map.
    let counts_slice = model.resolve_input_slice(step_idx, 1, buffers)?;

    // Wrap input GpuSlice as a DynTensor.
    let input_tensor = {
        let storage =
            MetalTensorData::view(input_slice.buffer().alias(), input_slice.byte_offset());
        DynTensor::from_gpu_storage(
            input_shape.to_vec(),
            DType::F32,
            Arc::new(storage),
            Device::metal(),
        )?
    };

    // Wrap counts GpuSlice as a DynTensor.
    let counts_tensor = {
        let storage =
            MetalTensorData::view(counts_slice.buffer().alias(), counts_slice.byte_offset());
        DynTensor::from_gpu_storage(
            counts_shape.to_vec(),
            DType::F32,
            Arc::new(storage),
            Device::metal(),
        )?
    };

    // Execute repeat_interleave eagerly. When both tensors are on GPU,
    // routes to GPU-native Blelloch prefix sum + binary search scatter
    // (dyn_tensor_metal_repeat_interleave_gpu.rs) with only one 4-byte
    // scalar readback. Falls back to CPU for non-F32 or dim_size > 256.
    let output = input_tensor
        .repeat_interleave(dim, &counts_tensor)
        .map_err(|e| native_dispatch_err(step_idx, format!("RuntimeOp RepeatInterleave: {e}")))?;

    // Extract the output MetalBuffer back to a GpuSlice.
    // The output may be on CPU if repeat_interleave fell back to CPU
    // dispatch. Transfer to GPU if needed.
    let output = if !output.device().is_gpu() {
        output.to_device(&Device::metal())?
    } else {
        output
    };

    let out_data = output.gpu_data::<MetalTensorData>().map_err(|_| {
        native_dispatch_err(
            step_idx,
            "RuntimeOp RepeatInterleave: output is not a GPU tensor".into(),
        )
    })?;
    Ok(out_data.as_gpu_slice())
}
