// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-step tensor dispatch executor for Metal GPU.
//!
//! Orchestrates a full `Vec<DispatchStep>` pipeline on Metal: compiles each
//! step's MSL into a [`KernelPipeline`], allocates intermediate buffers, and
//! encodes all dispatches into a single [`CommandBatch`] for one
//! `commit_and_wait`. This eliminates per-step GPU synchronization barriers.
//!
//! All intermediate buffers stay on GPU — no CPU readback until the final
//! output is requested.
//!
//! See #315, #815 for tracking.

use std::cell::Cell;
use std::collections::HashMap;

use nn_dsl::{
    build_dispatch_plan_full, emit_tensor_msl_with_plan, ir::ScalarType, PrecisionContract,
    PrecisionTier, TensorKernelDef, TensorNodeId, TensorOpKind,
};
use objc::rc::autoreleasepool;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::element::MetalElement;
use crate::gpu_scope;
use crate::gpu_slice::GpuSlice;

thread_local! {
    /// When `true`, `dispatch_execute_plan` returns the output `GpuSlice` with
    /// its original byte offset instead of blit-copying to a fresh zero-offset
    /// buffer. Set by compiled model execution when the caller handles non-zero
    /// offsets via the planned-buffer redirect mechanism (#4264).
    ///
    /// This is an opt-in flag with minimal blast radius: only the compiled model
    /// execution loop sets it, and it is cleared automatically via an RAII guard.
    /// All other callers (standalone dispatch, readback dispatch) are unaffected.
    static SKIP_DISPATCH_NORMALIZATION: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard that sets `SKIP_DISPATCH_NORMALIZATION` to `true` on creation
/// and restores it to `false` on drop. Ensures cleanup on error paths (#4264).
pub(crate) struct SkipNormalizationGuard(());

impl Drop for SkipNormalizationGuard {
    fn drop(&mut self) {
        SKIP_DISPATCH_NORMALIZATION.with(|c| c.set(false));
    }
}

/// Arm the dispatch normalization skip flag. Returns an RAII guard that
/// clears the flag on drop. Used by compiled model execution to prevent
/// `dispatch_execute_plan` from blit-copying the output to offset 0 when
/// the caller will relocate via the planned buffer (#4264).
pub(crate) fn arm_skip_dispatch_normalization() -> SkipNormalizationGuard {
    SKIP_DISPATCH_NORMALIZATION.with(|c| c.set(true));
    SkipNormalizationGuard(())
}

#[path = "tensor_dispatch_error.rs"]
mod error_types;
pub use error_types::TensorDispatchError;

#[path = "tensor_dispatch_helpers.rs"]
mod helpers;
pub(crate) use helpers::{checked_product_of_shape, output_elems};

#[path = "tensor_dispatch_steps.rs"]
mod steps;

#[path = "tensor_dispatch_packed.rs"]
mod packed;

/// Default threadgroup size for reduction kernels, matching the constant in
/// `nn_dsl::codegen_msl_tensor::REDUCE_THREADGROUP_SIZE`.
const REDUCE_THREADGROUP_SIZE: u32 = 256;

/// Input for buffer-to-buffer dispatch: either CPU data or an existing GPU buffer.
///
/// Models chain multiple `TensorKernelDef` dispatches. With `DispatchInput::Gpu`,
/// the output buffer from one dispatch feeds directly as input to the next without
/// a CPU round-trip. See #895 and `designs/2026-03-03-buffer-to-buffer-dispatch.md`.
#[non_exhaustive]
pub enum DispatchInput<'a, E: MetalElement> {
    /// CPU-resident data that will be uploaded to a Metal buffer.
    Cpu(&'a [E]),
    /// An existing GPU buffer region (buffer + byte offset).
    ///
    /// The [`GpuSlice`] carries both the buffer handle and its byte offset,
    /// preventing the recurring bug pattern where arena offsets are silently
    /// lost at integration boundaries (#2176, #2167, #2009, #2175).
    Gpu(GpuSlice),
}

/// Execute a full tensor kernel dispatch plan on Metal.
///
/// Takes a `TensorKernelDef`, builds its dispatch plan, generates MSL for each
/// step, and encodes them all into a single Metal command buffer. One
/// `commit_and_wait` at the end replaces N per-step synchronizations.
///
/// All intermediate buffers stay on GPU. Returns the final output buffer
/// contents as `Vec<E>`.
///
/// # Type Parameter
///
/// `E: MetalElement` — the element type for input/output data. Existing
/// callers pass `&[f32]` inputs and get `Vec<f32>` back (type inference
/// resolves `E = f32`). For f16 or bf16 inputs, pass the corresponding
/// slice type. bf16 is transparently converted to f16 at the Metal
/// boundary (Apple GPUs have no native bf16 compute).
///
/// # Parameters
///
/// - `cache`: Pipeline compilation cache (avoids redundant Metal compilation).
/// - `kernel`: The tensor-level kernel definition from nn-dsl.
/// - `dtype`: Scalar type for MSL codegen (f32 or f16).
/// - `inputs`: Named input tensors, keyed by the `Input` node name.
///
/// # Errors
///
/// Returns `TensorDispatchError` on codegen failure, missing inputs/buffers,
/// or Metal dispatch errors.
pub fn execute_tensor_dispatch<E: MetalElement, D: AsRef<[E]>>(
    cache: &PipelineCache,
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    inputs: &HashMap<&str, D>,
) -> Result<Vec<E>, TensorDispatchError> {
    // Convert CPU inputs to DispatchInput::Cpu for the shared inner function.
    let dispatch_inputs: HashMap<&str, DispatchInput<'_, E>> = inputs
        .iter()
        .map(|(k, v)| (*k, DispatchInput::Cpu(v.as_ref())))
        .collect();
    let out_slice = dispatch_inner::<E>(cache, kernel, dtype, &dispatch_inputs, None)?;
    gpu_scope::flush().map_err(|e| {
        TensorDispatchError::Metal(crate::error::MetalError::DispatchFailed(e.to_string()))
    })?;
    // Use offset-aware readback for arena-allocated buffers (#2207).
    let elems = output_elems(kernel, kernel.output)?;
    Ok(E::read_buffer_at_offset(
        out_slice.buffer(),
        out_slice.byte_offset(),
        elems,
    )?)
}

/// Like [`execute_tensor_dispatch`] but accepts `DispatchInput` (GPU or CPU).
///
/// Model forward paths chain GPU dispatches and read the final result back
/// to CPU. Uses offset-aware readback for arena-allocated buffers (#2207).
pub fn execute_tensor_dispatch_readback<E: MetalElement>(
    cache: &PipelineCache,
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    inputs: &HashMap<&str, DispatchInput<'_, E>>,
) -> Result<Vec<E>, TensorDispatchError> {
    let out_slice = dispatch_inner::<E>(cache, kernel, dtype, inputs, None)?;
    gpu_scope::flush().map_err(|e| {
        TensorDispatchError::Metal(crate::error::MetalError::DispatchFailed(e.to_string()))
    })?;
    // Use offset-aware readback for arena-allocated buffers (#2207).
    let elems = output_elems(kernel, kernel.output)?;
    Ok(E::read_buffer_at_offset(
        out_slice.buffer(),
        out_slice.byte_offset(),
        elems,
    )?)
}

/// Execute a full tensor kernel dispatch plan, returning the output as a [`GpuSlice`].
///
/// Same as [`execute_tensor_dispatch`] but skips the final CPU readback. The
/// returned [`GpuSlice`] can be used to construct `DispatchInput::Gpu` for a
/// subsequent dispatch, eliminating CPU round-trips between stages.
///
/// # Use Case
///
/// Model forward passes chain 6-48 dispatch calls. With buffer-to-buffer
/// dispatch, intermediate results stay on GPU and only the final output is
/// read back to CPU.
///
/// ```text
/// let enc = execute_tensor_dispatch_to_buffer(cache, &enc_def, dtype, &enc_inputs)?;
/// let lstm_inputs = HashMap::from([("x", DispatchInput::Gpu(enc.alias()))]);
/// let lstm = execute_tensor_dispatch_to_buffer(cache, &lstm_def, dtype, &lstm_inputs)?;
/// ```
pub fn execute_tensor_dispatch_to_buffer<E: MetalElement>(
    cache: &PipelineCache,
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    inputs: &HashMap<&str, DispatchInput<'_, E>>,
) -> Result<GpuSlice, TensorDispatchError> {
    dispatch_inner::<E>(cache, kernel, dtype, inputs, None)
}

/// Like [`execute_tensor_dispatch_to_buffer`] but with an explicit precision
/// contract. Use `PrecisionTier::Strict` for Kahan compensated reductions
/// (#1814).
pub fn execute_tensor_dispatch_to_buffer_with_contract<E: MetalElement>(
    cache: &PipelineCache,
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    inputs: &HashMap<&str, DispatchInput<'_, E>>,
    contract: PrecisionContract,
) -> Result<GpuSlice, TensorDispatchError> {
    dispatch_inner::<E>(cache, kernel, dtype, inputs, Some(contract))
}

/// Shared dispatch implementation that returns the output as a [`GpuSlice`].
///
/// Both [`execute_tensor_dispatch`] (which reads back to CPU) and
/// [`execute_tensor_dispatch_to_buffer`] (which keeps data on GPU) delegate
/// to this function.
///
/// Wrapped in `autoreleasepool` as defense-in-depth: Metal's ObjC runtime may
/// create autoreleased temporary objects during buffer creation, pipeline
/// lookup, encoding, and command buffer lifecycle. Without a pool on background
/// threads, these accumulate indefinitely (dvoice#1245).
fn dispatch_inner<E: MetalElement>(
    cache: &PipelineCache,
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    inputs: &HashMap<&str, DispatchInput<'_, E>>,
    contract_override: Option<PrecisionContract>,
) -> Result<GpuSlice, TensorDispatchError> {
    autoreleasepool(|| dispatch_inner_body(cache, kernel, dtype, inputs, contract_override))
}

fn dispatch_inner_body<E: MetalElement>(
    cache: &PipelineCache,
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    inputs: &HashMap<&str, DispatchInput<'_, E>>,
    contract_override: Option<PrecisionContract>,
) -> Result<GpuSlice, TensorDispatchError> {
    let ctx = cache.context();
    let elem_size = E::element_size();
    if dtype != E::scalar_type() {
        return Err(TensorDispatchError::DtypeMismatch {
            expected: E::scalar_type(),
            actual: dtype,
        });
    }

    // Buffer map: TensorNodeId → MetalBuffer.
    let mut buffers: HashMap<TensorNodeId, MetalBuffer> = HashMap::new();
    let mut offsets: HashMap<TensorNodeId, usize> = HashMap::new();

    // Codegen cache — shared with GPU-only path via dispatch_execute_plan.
    let codegen = codegen_for_kernel(kernel, dtype, contract_override)?;

    // Bind input nodes from the expanded kernel (Input nodes retain their
    // sequential IDs through expansion, but we use the expanded graph to
    // ensure correct ID mapping).
    for node in &codegen.expanded.nodes {
        if let TensorOpKind::Input { name, .. } = &node.kind {
            let input = inputs
                .get(name.as_str())
                .ok_or_else(|| TensorDispatchError::MissingInput { name: name.clone() })?;
            let buf = match input {
                DispatchInput::Cpu(data) => E::create_buffer(ctx, data)?,
                DispatchInput::Gpu(ref slice) => {
                    let byte_offset = slice.byte_offset();
                    let expected_elems = checked_product_of_shape(&node.shape)?;
                    let expected_bytes =
                        expected_elems.checked_mul(elem_size).ok_or_else(|| {
                            TensorDispatchError::ShapeOverflow {
                                shape: node.shape.clone(),
                            }
                        })?;
                    let required_bytes =
                        byte_offset.checked_add(expected_bytes).ok_or_else(|| {
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
                    slice.buffer().alias()
                }
            };
            buffers.insert(node.id, buf);
        }
    }

    dispatch_execute_plan(cache, &codegen, elem_size, &mut buffers, &mut offsets)
}

/// Look up or generate the MSL codegen output for a kernel+dtype+contract.
///
/// Shared between `dispatch_inner_body` (generic) and `dispatch_inner_body_gpu_only`.
pub(super) fn codegen_for_kernel(
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    contract_override: Option<PrecisionContract>,
) -> Result<std::sync::Arc<crate::msl_codegen_cache::CodegenOutput>, TensorDispatchError> {
    let contract = contract_override
        .unwrap_or_else(|| PrecisionContract::bootstrap(PrecisionTier::Normal, dtype));
    crate::msl_codegen_cache::get_or_generate(kernel, dtype, contract, || {
        let (plan, effective_output, expanded) = build_dispatch_plan_full(kernel, dtype)?;
        let msl = emit_tensor_msl_with_plan(&plan, &expanded, contract)?;
        Ok(crate::msl_codegen_cache::CodegenOutput {
            plan,
            effective_output,
            expanded,
            msl,
        })
    })
}

/// Execute a dispatch plan with pre-bound input buffers.
///
/// Contains the batch encoding and output normalization shared between
/// the generic `dispatch_inner_body<E>` and the GPU-only path.
/// Part of #3079 (dispatch transient allocation elimination).
pub(super) fn dispatch_execute_plan(
    cache: &PipelineCache,
    codegen: &crate::msl_codegen_cache::CodegenOutput,
    elem_size: usize,
    buffers: &mut HashMap<TensorNodeId, MetalBuffer>,
    offsets: &mut HashMap<TensorNodeId, usize>,
) -> Result<GpuSlice, TensorDispatchError> {
    let ctx = cache.context();
    let plan = &codegen.plan;
    let effective_output = codegen.effective_output;
    let expanded = &codegen.expanded;
    let combined_msl = &codegen.msl;

    // Lazy batch (#2009): encode all steps into the thread-local lazy batch.
    gpu_scope::get_or_create_batch().map_err(|e| {
        TensorDispatchError::Metal(crate::error::MetalError::DispatchFailed(e.to_string()))
    })?;
    let scope_result = gpu_scope::encode_into_lazy_batch(|batch| {
        for step in plan {
            steps::dispatch_one_step(
                step,
                cache,
                batch,
                ctx,
                combined_msl,
                expanded,
                elem_size,
                buffers,
                offsets,
            )?;
        }
        Ok(())
    });
    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(dispatch_err)) => return Err(dispatch_err),
        Err(tensor_err) => {
            return Err(TensorDispatchError::Metal(
                crate::error::MetalError::DispatchFailed(tensor_err.to_string()),
            ))
        }
    }

    // Output buffer extraction + normalization (#2166).
    let out_buf = buffers
        .remove(&effective_output)
        .ok_or(TensorDispatchError::MissingBuffer(effective_output))?;
    let out_offset = offsets.get(&effective_output).copied().unwrap_or(0);

    let output_node = expanded
        .nodes
        .get(effective_output.index())
        .filter(|n| n.id == effective_output)
        .ok_or(TensorDispatchError::MissingBuffer(effective_output))?;
    let out_elems = checked_product_of_shape(&output_node.shape)?;
    let out_bytes =
        out_elems
            .checked_mul(elem_size)
            .ok_or_else(|| TensorDispatchError::ShapeOverflow {
                shape: output_node.shape.clone(),
            })?;

    if out_offset == 0 && out_buf.len() == out_bytes {
        return Ok(GpuSlice::zero_offset(out_buf));
    }

    // Skip normalization when the compiled model execution loop has armed
    // the flag (#4264). The caller handles non-zero offsets via the planned-
    // buffer redirect — normalizing here would destroy the planned-buffer
    // identity and cause a DOUBLE blit (normalize + relocate).
    if SKIP_DISPATCH_NORMALIZATION.with(Cell::get) {
        // Track the skipped normalization blit for diagnostics (#4264).
        // This blit was previously uncounted (Source 2 in blit_copy_analysis.rs)
        // but is real GPU work eliminated by the optimization.
        crate::dispatch_stats::TOTAL_BLITS_ELIMINATED.with(|c| {
            c.set(c.get() + 1);
        });
        return Ok(GpuSlice::new(out_buf, out_offset));
    }

    let fresh_buf = ctx
        .create_buffer_zeroed(out_bytes)
        .map_err(TensorDispatchError::Metal)?;
    gpu_scope::encode_into_lazy_batch(|batch| -> Result<(), crate::error::MetalError> {
        batch.blit_copy(&out_buf, out_offset, &fresh_buf, 0, out_bytes)?;
        Ok(())
    })
    .map_err(|e| {
        TensorDispatchError::Metal(crate::error::MetalError::DispatchFailed(e.to_string()))
    })?
    .map_err(TensorDispatchError::Metal)?;
    Ok(GpuSlice::zero_offset(fresh_buf))
}

#[path = "tensor_dispatch_gpu_only.rs"]
mod gpu_only;
pub(crate) use gpu_only::execute_tensor_dispatch_to_buffer_gpu_only;

#[path = "tensor_dispatch_batched.rs"]
mod batched;
pub use batched::execute_tensor_dispatch_batched;

#[cfg(test)]
#[path = "tensor_dispatch_tests.rs"]
mod tests;
