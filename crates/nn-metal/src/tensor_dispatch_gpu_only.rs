// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-only dispatch path for compiled model execution.
//!
//! Eliminates the `HashMap<&str, DispatchInput<E>>` allocation (Level 1) and
//! the per-call `HashMap<TensorNodeId, MetalBuffer/usize>` allocations (Level 2)
//! that the generic `dispatch_inner_body` requires. Compiled models never have
//! CPU inputs, so the `DispatchInput` wrapping and fresh HashMap creation are
//! pure overhead (~1,470 HashMap alloc/dealloc per forward pass for Kokoro).
//!
//! Level 2 uses thread-local `DispatchContext` to reuse the internal HashMaps
//! across dispatch calls without any caller changes. `HashMap::clear()` retains
//! capacity, so after the first dispatch step, all subsequent steps reuse the
//! same backing memory.
//!
//! Part of #3079 (dispatch transient allocation elimination).

use std::cell::RefCell;
use std::collections::HashMap;

use nn_dsl::ir::ScalarType;
use nn_dsl::{PrecisionContract, TensorKernelDef, TensorNodeId, TensorOpKind};
use objc::rc::autoreleasepool;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;

use super::helpers::checked_product_of_shape;
use super::TensorDispatchError;

/// Element size in bytes for a [`ScalarType`].
fn elem_size_for_scalar_type(dtype: ScalarType) -> usize {
    dtype.byte_size()
}

/// Thread-local reusable internal state for GPU dispatch.
///
/// Eliminates per-call HashMap allocations. `clear()` resets the maps
/// without deallocating (HashMap retains capacity).
struct DispatchContext {
    buffers: HashMap<TensorNodeId, MetalBuffer>,
    offsets: HashMap<TensorNodeId, usize>,
}

impl DispatchContext {
    fn new() -> Self {
        Self {
            buffers: HashMap::with_capacity(16),
            offsets: HashMap::with_capacity(4),
        }
    }

    fn clear(&mut self) {
        self.buffers.clear();
        self.offsets.clear();
    }
}

thread_local! {
    static DISPATCH_CTX: RefCell<DispatchContext> = RefCell::new(DispatchContext::new());
}

/// GPU-only dispatch: accepts `&HashMap<&str, GpuSlice>` directly.
///
/// Compiled models never have CPU inputs. This path:
/// - **Level 1:** Avoids creating `HashMap<&str, DispatchInput<E>>` (~490 allocs/fwd)
/// - **Level 2:** Reuses internal `buffers`/`offsets` HashMaps via thread-local
///   `DispatchContext` (~980 allocs/fwd)
///
/// Total: ~1,470 HashMap alloc/dealloc eliminated per forward pass.
pub(crate) fn execute_tensor_dispatch_to_buffer_gpu_only(
    cache: &PipelineCache,
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    gpu_inputs: &HashMap<&str, GpuSlice>,
    contract_override: Option<PrecisionContract>,
) -> Result<GpuSlice, TensorDispatchError> {
    autoreleasepool(|| {
        DISPATCH_CTX.with(|cell| {
            let mut ctx = cell.borrow_mut();
            ctx.clear();
            dispatch_inner_body_gpu_only(
                cache,
                kernel,
                dtype,
                gpu_inputs,
                contract_override,
                &mut ctx,
            )
        })
    })
}

fn dispatch_inner_body_gpu_only(
    cache: &PipelineCache,
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    gpu_inputs: &HashMap<&str, GpuSlice>,
    contract_override: Option<PrecisionContract>,
    ctx: &mut DispatchContext,
) -> Result<GpuSlice, TensorDispatchError> {
    let elem_size = elem_size_for_scalar_type(dtype);
    let codegen = super::codegen_for_kernel(kernel, dtype, contract_override)?;

    bind_gpu_inputs(
        &codegen.expanded,
        gpu_inputs,
        elem_size,
        &mut ctx.buffers,
        &mut ctx.offsets,
    )?;
    super::dispatch_execute_plan(
        cache,
        &codegen,
        elem_size,
        &mut ctx.buffers,
        &mut ctx.offsets,
    )
}

/// Bind GPU-only inputs into the buffer/offset maps.
fn bind_gpu_inputs(
    expanded: &TensorKernelDef,
    gpu_inputs: &HashMap<&str, GpuSlice>,
    elem_size: usize,
    buffers: &mut HashMap<TensorNodeId, MetalBuffer>,
    offsets: &mut HashMap<TensorNodeId, usize>,
) -> Result<(), TensorDispatchError> {
    for node in &expanded.nodes {
        if let TensorOpKind::Input { name, .. } = &node.kind {
            let slice = gpu_inputs
                .get(name.as_str())
                .ok_or_else(|| TensorDispatchError::MissingInput { name: name.clone() })?;
            let byte_offset = slice.byte_offset();
            let expected_elems = checked_product_of_shape(&node.shape)?;
            let expected_bytes = expected_elems.checked_mul(elem_size).ok_or_else(|| {
                TensorDispatchError::ShapeOverflow {
                    shape: node.shape.clone(),
                }
            })?;
            let required_bytes = byte_offset.checked_add(expected_bytes).ok_or_else(|| {
                TensorDispatchError::ShapeOverflow {
                    shape: node.shape.clone(),
                }
            })?;
            if slice.buffer().len() < required_bytes {
                return Err(TensorDispatchError::BufferSizeMismatch {
                    name: name.clone(),
                    expected: required_bytes,
                    actual: slice.buffer().len(),
                });
            }
            if byte_offset > 0 {
                offsets.insert(node.id, byte_offset);
            }
            buffers.insert(node.id, slice.buffer().alias());
        }
    }
    Ok(())
}
