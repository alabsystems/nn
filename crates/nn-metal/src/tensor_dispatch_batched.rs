// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Batched tensor dispatch for Metal GPU.
//!
//! Extracted from `tensor_dispatch.rs` for file-size compliance (#1863 D3).
//! Contains `execute_tensor_dispatch_batched()` which dispatches the same
//! kernel N times in a single GPU command buffer submission.

use std::collections::HashMap;

use nn_dsl::{
    build_dispatch_plan_full, emit_tensor_msl_with_plan, ir::ScalarType, PrecisionContract,
    PrecisionTier, TensorNodeId, TensorOpKind,
};
use objc::rc::autoreleasepool;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::element::MetalElement;

use super::error_types::TensorDispatchError;
use super::steps;

/// Execute the same tensor kernel N times with different inputs, encoding all
/// dispatches into the thread-local lazy batch.
///
/// This is the batched variant of [`super::execute_tensor_dispatch`]. It builds the
/// dispatch plan and compiles MSL **once**, then encodes N independent dispatch
/// sequences into the lazy batch. `flush()` is called before CPU readback.
///
/// Lazy batch (#2009): all N batch elements encode into the thread-local lazy
/// batch. `flush()` is called before CPU readback.
///
/// # Use Case
///
/// The spectral branch of HTDemucs dispatches the same 1D op per frequency bin
/// or time step. Sequential dispatch creates O(F+T) barriers. This function
/// reduces that to O(1).
///
/// # Parameters
///
/// - `cache`: Pipeline compilation cache.
/// - `kernel`: The tensor-level kernel definition (same for all batch elements).
/// - `dtype`: Scalar type for MSL codegen.
/// - `batch_inputs`: Per-batch-element input maps. Each entry maps input names
///   to data slices for that batch element.
///
/// # Returns
///
/// `Vec<Vec<E>>` — one output vector per batch element, in the same order as
/// `batch_inputs`.
pub fn execute_tensor_dispatch_batched<E: MetalElement, D: AsRef<[E]>>(
    cache: &PipelineCache,
    kernel: &nn_dsl::TensorKernelDef,
    dtype: ScalarType,
    batch_inputs: &[HashMap<&str, D>],
) -> Result<Vec<Vec<E>>, TensorDispatchError> {
    if batch_inputs.is_empty() {
        return Ok(Vec::new());
    }

    // Fast path: single element falls back to non-batched dispatch.
    if batch_inputs.len() == 1 {
        let result = super::execute_tensor_dispatch(cache, kernel, dtype, &batch_inputs[0])?;
        return Ok(vec![result]);
    }

    // Wrapped in `autoreleasepool` as defense-in-depth: Metal buffer creation,
    // pipeline lookup, batch encoding, and commit/wait may create ObjC
    // autoreleased temporaries that leak on background threads (dvoice#1245).
    autoreleasepool(|| dispatch_batched_inner::<E, D>(cache, kernel, dtype, batch_inputs))
}

fn dispatch_batched_inner<E: MetalElement, D: AsRef<[E]>>(
    cache: &PipelineCache,
    kernel: &nn_dsl::TensorKernelDef,
    dtype: ScalarType,
    batch_inputs: &[HashMap<&str, D>],
) -> Result<Vec<Vec<E>>, TensorDispatchError> {
    let ctx = cache.context();
    let elem_size = E::element_size();
    if dtype != E::scalar_type() {
        return Err(TensorDispatchError::DtypeMismatch {
            expected: E::scalar_type(),
            actual: dtype,
        });
    }
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, dtype);

    // Build plan and compile MSL ONCE for all batch elements.
    let (plan, effective_output, expanded) = build_dispatch_plan_full(kernel, dtype)?;
    let tensor_msl = emit_tensor_msl_with_plan(&plan, &expanded, contract)?;
    let combined_msl = tensor_msl;

    // Per-batch-element buffer maps. Each element gets its own set of
    // intermediate buffers so dispatch steps don't overwrite each other.
    let n = batch_inputs.len();
    let mut all_buffers: Vec<HashMap<TensorNodeId, MetalBuffer>> = Vec::with_capacity(n);

    // Bind inputs for each batch element.
    for inputs in batch_inputs {
        let mut buffers: HashMap<TensorNodeId, MetalBuffer> = HashMap::new();
        for node in &expanded.nodes {
            if let TensorOpKind::Input { name, .. } = &node.kind {
                let data = inputs
                    .get(name.as_str())
                    .ok_or_else(|| TensorDispatchError::MissingInput { name: name.clone() })?;
                let buf = E::create_buffer(ctx, data.as_ref())?;
                buffers.insert(node.id, buf);
            }
        }
        all_buffers.push(buffers);
    }

    // Track per-batch-element output byte offsets for correct readback.
    // Arena allocation can place each batch element's output at a different
    // non-zero offset within the arena buffer (#2206).
    let mut output_offsets: Vec<usize> = Vec::with_capacity(n);

    // Lazy batch (#2009): encode all batch elements into the thread-local
    // lazy batch. No commit_and_wait here — flush() before CPU readback.
    crate::gpu_scope::get_or_create_batch().map_err(|e| {
        TensorDispatchError::Metal(crate::error::MetalError::DispatchFailed(e.to_string()))
    })?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        for buffers in &mut all_buffers {
            let mut offsets: HashMap<TensorNodeId, usize> = HashMap::new();
            for step in &plan {
                steps::dispatch_one_step(
                    step,
                    cache,
                    batch,
                    ctx,
                    &combined_msl,
                    &expanded,
                    elem_size,
                    buffers,
                    &mut offsets,
                )?;
            }
            let out_offset = offsets.get(&effective_output).copied().unwrap_or(0);
            output_offsets.push(out_offset);
        }
        Ok(())
    });
    // Unwrap the two Result layers: outer TensorError (batch absent)
    // and inner TensorDispatchError (encoding failure).
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(dispatch_err)) => return Err(dispatch_err),
        Err(tensor_err) => {
            return Err(TensorDispatchError::Metal(
                crate::error::MetalError::DispatchFailed(tensor_err.to_string()),
            ))
        }
    }

    // Lazy batch (#2009): flush before CPU readback.
    crate::gpu_scope::flush().map_err(|e| {
        TensorDispatchError::Metal(crate::error::MetalError::DispatchFailed(e.to_string()))
    })?;

    // Read back outputs for each batch element at the correct arena offset.
    let out_elems = super::helpers::output_elems(kernel, kernel.output)?;
    let mut results = Vec::with_capacity(n);
    for (buffers, &out_offset) in all_buffers.iter().zip(output_offsets.iter()) {
        let out_buf = buffers
            .get(&effective_output)
            .ok_or(TensorDispatchError::MissingBuffer(effective_output))?;
        results.push(E::read_buffer_at_offset(out_buf, out_offset, out_elems)?);
    }
    Ok(results)
}
