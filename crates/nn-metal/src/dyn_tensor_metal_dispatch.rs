// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU dispatch helpers for [`MetalDynBackend`].
//!
//! Extracted from `dyn_tensor_metal.rs` for file-size compliance (#1863 D1).
//! Contains the thread-local pipeline cache and the two `dispatch_def*` methods
//! that route `TensorKernelDef` dispatch by dtype (F32 vs BF16/F16).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use nn_dsl::ir::ScalarType;

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;
use crate::tensor_dispatch::{
    execute_tensor_dispatch_to_buffer, execute_tensor_dispatch_to_buffer_with_contract,
    DispatchInput,
};

use super::MetalTensorData;

// -- Thread-local pipeline cache ----------------------------------------------

thread_local! {
    pub(crate) static DYN_CACHE: RefCell<Option<PipelineCache>> = const { RefCell::new(None) };
}

/// Access the thread-local pipeline cache, initializing on first use.
///
/// Replaces the 7-line `DYN_CACHE.with(|cell| { init; borrow; … })` boilerplate
/// that appears in every GPU dispatch function. Part of #2218.
pub(crate) fn with_pipeline_cache<F, T>(f: F) -> Result<T>
where
    F: FnOnce(&PipelineCache) -> Result<T>,
{
    DYN_CACHE.with(|cell| {
        if cell.borrow().is_none() {
            let cache = PipelineCache::new_global().map_err(|e| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::KernelCompile,
                    e.to_string(),
                )
            })?;
            *cell.borrow_mut() = Some(cache);
        }
        let guard = cell.borrow();
        let cache = guard.as_ref().ok_or_else(|| {
            TensorError::backend_failure(
                nn_core::BackendDomain::Metal,
                nn_core::BackendErrorKind::Other,
                "pipeline cache not initialized".to_string(),
            )
        })?;
        f(cache)
    })
}

// -- Dispatch helpers ---------------------------------------------------------

impl super::MetalDynBackend {
    /// Dispatch a TensorKernelDef to GPU, returning result as a GPU DynTensor.
    ///
    /// Routes by dtype: F32 dispatches with `f32` Metal buffers (4 bytes/elem),
    /// BF16/F16 dispatch with `half::f16` Metal buffers (2 bytes/elem, MSL `half`).
    /// BF16 is converted to f16 at the Metal boundary since Apple GPUs have no
    /// native bf16 ALU (#1646 D8).
    ///
    /// Non-float dtypes return `DtypeMismatch`.
    pub(super) fn dispatch_def(
        def: &nn_dsl::tensor_ir::TensorKernelDef,
        inputs: &[(&str, GpuSlice)],
        out_shape: &[usize],
        out_dtype: DType,
    ) -> Result<DynTensor> {
        Self::dispatch_def_inner(def, inputs, out_shape, out_dtype, None)
    }

    /// Like [`dispatch_def`](Self::dispatch_def) but with an explicit precision
    /// contract. Use `PrecisionTier::Strict` for Kahan compensated reductions
    /// (#1814).
    pub(super) fn dispatch_def_with_contract(
        def: &nn_dsl::tensor_ir::TensorKernelDef,
        inputs: &[(&str, GpuSlice)],
        out_shape: &[usize],
        out_dtype: DType,
        contract: nn_dsl::PrecisionContract,
    ) -> Result<DynTensor> {
        Self::dispatch_def_inner(def, inputs, out_shape, out_dtype, Some(contract))
    }

    /// Shared implementation for `dispatch_def` and `dispatch_def_with_contract`.
    ///
    /// When `contract` is `None`, dispatches without a precision contract.
    /// When `Some`, uses the contract-aware dispatch path.
    fn dispatch_def_inner(
        def: &nn_dsl::tensor_ir::TensorKernelDef,
        inputs: &[(&str, GpuSlice)],
        out_shape: &[usize],
        out_dtype: DType,
        contract: Option<nn_dsl::PrecisionContract>,
    ) -> Result<DynTensor> {
        with_pipeline_cache(|cache| {
            let result = match out_dtype {
                DType::F32 => {
                    let gpu_inputs: HashMap<&str, DispatchInput<'_, f32>> = inputs
                        .iter()
                        .map(|(k, slice)| (*k, DispatchInput::Gpu(slice.alias())))
                        .collect();
                    dispatch_typed(cache, def, ScalarType::F32, &gpu_inputs, contract)
                }
                DType::BF16 | DType::F16 => {
                    // BF16/F16 tensors use f16 Metal buffers (2 bytes/elem).
                    // MSL codegen emits `half` buffer types with `float` accumulators.
                    // bf16->f16 conversion happens at MetalElement boundary (#1646 D7/D8).
                    let gpu_inputs: HashMap<&str, DispatchInput<'_, half::f16>> = inputs
                        .iter()
                        .map(|(k, slice)| (*k, DispatchInput::Gpu(slice.alias())))
                        .collect();
                    dispatch_typed(cache, def, ScalarType::F16, &gpu_inputs, contract)
                }
                other => {
                    return Err(TensorError::dtype_mismatch(DType::F32, other));
                }
            };
            let slice = result.map_err(|e: crate::tensor_dispatch::TensorDispatchError| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::DispatchFailed,
                    e.to_string(),
                )
            })?;
            let byte_offset = slice.byte_offset();
            let storage = MetalTensorData::from_arena_alloc(slice.into_buffer(), byte_offset);
            DynTensor::from_gpu_storage(
                out_shape.to_vec(),
                out_dtype,
                Arc::new(storage),
                Device::metal(),
            )
        })
    }
}

/// Type-parametric dispatch: returns a GpuSlice with exact output size.
///
/// The returned slice may have a non-zero byte offset when arena-allocated.
/// Callers must preserve the offset (via `GpuSlice::byte_offset()`).
fn dispatch_typed<E: crate::element::MetalElement>(
    cache: &PipelineCache,
    def: &nn_dsl::tensor_ir::TensorKernelDef,
    scalar_type: ScalarType,
    gpu_inputs: &HashMap<&str, DispatchInput<'_, E>>,
    contract: Option<nn_dsl::PrecisionContract>,
) -> std::result::Result<GpuSlice, crate::tensor_dispatch::TensorDispatchError> {
    match contract {
        Some(c) => execute_tensor_dispatch_to_buffer_with_contract::<E>(
            cache,
            def,
            scalar_type,
            gpu_inputs,
            c,
        ),
        None => execute_tensor_dispatch_to_buffer::<E>(cache, def, scalar_type, gpu_inputs),
    }
}
