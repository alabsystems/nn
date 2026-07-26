// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! NativeOp encoding plan for direct Metal dispatch.
//!
//! Replaces the DynTensor bridge pattern (`GpuSlice → DynTensor → gpu_*()
//! → DynTensor → GpuSlice`) with a structured encoding plan that can be
//! dispatched via [`dispatch_native_encoding`]. Created by `plan_encoding()`
//! at model build time; executed at runtime without per-variant code.
//!
//! Part of #3472 (NativeOp DynTensor bridge elimination).

use nn_core::Result;

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;
use crate::kernel_dispatch::KernelPipeline;

use super::helpers::native_dispatch_err;
use super::CompiledModel;

/// Whether to use `dispatch_threads` or `dispatch_thread_groups`.
///
/// Elementwise kernels (InstanceNorm, MaxPool1d) use `Threads` — Metal
/// auto-computes threadgroup count. Reduction/GEMM kernels (AddNormLinear,
/// FusedResBlock) use `Threadgroups` — caller specifies threadgroup count.
#[derive(Debug, Clone, Copy)]
pub(crate) enum NativeDispatchMode {
    /// `dispatch_threads(grid, threadgroup)` — grid is total thread count.
    Threads,
    /// `dispatch_thread_groups(threadgroups, threads_per_group)` — grid is
    /// threadgroup count.
    Threadgroups,
}

/// Encoding plan for a NativeOp dispatched via direct Metal commands.
///
/// Captures everything needed to encode a single compute dispatch:
/// pipeline, grid/threadgroup dimensions, threadgroup memory, and buffer
/// bindings. At execution time, [`dispatch_native_encoding`] resolves
/// bindings and encodes Metal commands — no per-variant execution code.
///
/// For multi-dispatch sequences (e.g., FusedResBlock), a `Vec<NativeEncoding>`
/// is dispatched via [`dispatch_native_encoding_sequence`]. Intermediate
/// buffers flow between dispatches via [`NativeBindingSource::Intermediate`].
pub(crate) struct NativeEncoding {
    /// Compiled Metal pipeline.
    pub pipeline: KernelPipeline,
    /// Grid size (threads for `Threads` mode, threadgroups for `Threadgroups` mode).
    pub grid: [u32; 3],
    /// Threadgroup size (threads per threadgroup per dimension).
    pub threadgroup: [u32; 3],
    /// Dispatch mode: threads vs threadgroups.
    pub dispatch_mode: NativeDispatchMode,
    /// Threadgroup memory in bytes (0 = none).
    pub threadgroup_memory_bytes: u64,
    /// Total output buffer size in bytes.
    pub output_bytes: usize,
    /// Buffer bindings in dispatch order: (buffer_index, source).
    pub bindings: Vec<(usize, NativeBindingSource)>,
    /// Auxiliary buffer allocations needed by this dispatch.
    /// Allocated before the compute dispatch, each via a separate arena alloc.
    /// Part of #3472 D3 (FusedResBlock multi-dispatch infrastructure).
    pub auxiliary_allocs: Vec<AuxiliaryAlloc>,
}

/// An auxiliary buffer allocation for a NativeOp dispatch.
///
/// Used when a dispatch requires additional buffers beyond the single output
/// (e.g., FusedResBlock dispatch 2 needs counter + partials + next-phase stats).
/// Part of #3472 D3.
pub(crate) struct AuxiliaryAlloc {
    /// Buffer size in bytes.
    pub bytes: usize,
    /// Metal buffer binding index for this allocation.
    pub binding_index: usize,
    /// If true, blit-fill with zeros before the compute dispatch.
    pub zero_fill: bool,
    /// If true, this buffer is exposed for subsequent encodings via
    /// `IntermediateAuxiliary` binding source.
    pub expose_as_intermediate: bool,
}

/// What a buffer binding slot resolves to at execution time.
pub(crate) enum NativeBindingSource {
    /// Input from the step's edge map. Value is the edge index
    /// (0 = first input, 1 = second input, etc.).
    Edge(usize),
    /// Pre-uploaded weight buffer by name.
    Weight(String),
    /// Output buffer (allocated by the dispatch function).
    Output,
    /// Inline constant bytes (encoded via Metal `setBytes`).
    Constant(Vec<u8>),
    /// Output buffer from encoding[i] in a multi-dispatch sequence.
    /// Used by [`dispatch_native_encoding_sequence`] to pass data between
    /// dispatches. Part of #3472 D3.
    Intermediate(usize),
    /// Auxiliary buffer from a prior encoding in a multi-dispatch sequence.
    /// `encoding_idx` indexes the encoding Vec; `auxiliary_idx` indexes
    /// that encoding's `auxiliary_allocs` (only those with
    /// `expose_as_intermediate = true`). Part of #3472 D3.
    IntermediateAuxiliary {
        encoding_idx: usize,
        auxiliary_idx: usize,
    },
    /// Pre-resolved `GpuSlice` passed by the caller.
    ///
    /// Used by executors that resolve buffers before building the encoding
    /// plan (e.g., FusedResBlock gamma/beta from narrow+reshape).
    /// Indexes into the `pre_resolved` parameter of
    /// [`dispatch_native_encoding_sequence`]. Part of #3472 D3 S3.
    PreResolved(usize),
}

impl NativeBindingSource {
    /// Create a constant binding from a `u32` value.
    pub(crate) fn constant_u32(val: u32) -> Self {
        Self::Constant(bytemuck::bytes_of(&val).to_vec())
    }

    /// Create a constant binding from an `f32` value.
    pub(crate) fn constant_f32(val: f32) -> Self {
        Self::Constant(bytemuck::bytes_of(&val).to_vec())
    }
}

/// Execute a [`NativeEncoding`] plan via direct Metal dispatch.
///
/// Resolves buffer bindings, allocates the output buffer via the arena,
/// and encodes a single compute dispatch into the thread-local lazy batch.
///
/// This is the common dispatch function that replaces per-variant executor
/// boilerplate. Each NativeOp variant only needs to produce a `NativeEncoding`;
/// this function handles the actual Metal encoding.
pub(super) fn dispatch_native_encoding(
    model: &CompiledModel,
    encoding: &NativeEncoding,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    cache: &PipelineCache,
) -> Result<GpuSlice> {
    let step_weights = &model.def.weight_buffers[step_idx];

    // Pre-resolve edge inputs outside the encoding closure.
    // `resolve_input_slice` requires `&self` borrows that can't enter the
    // `encode_into_lazy_batch` closure.
    let mut edge_slices: Vec<Option<GpuSlice>> = Vec::new();
    for (_, source) in &encoding.bindings {
        if let NativeBindingSource::Edge(edge_idx) = source {
            let needed = *edge_idx + 1;
            if edge_slices.len() < needed {
                edge_slices.resize_with(needed, || None);
            }
            if edge_slices[*edge_idx].is_none() {
                edge_slices[*edge_idx] =
                    Some(model.resolve_input_slice(step_idx, *edge_idx, buffers)?);
            }
        }
    }

    // Allocate output buffer via arena.
    let ctx = cache.context();
    let (out_buf, out_offset) =
        crate::arena::arena_alloc_or_create(ctx, encoding.output_bytes)
            .map_err(|e| native_dispatch_err(step_idx, format!("NativeEncoding alloc: {e}")))?;

    // Encode and dispatch.
    crate::gpu_scope::get_or_create_batch()?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| {
        let enc = batch.new_encoder()?;

        for (idx, source) in &encoding.bindings {
            match source {
                NativeBindingSource::Edge(edge_idx) => {
                    let slice = edge_slices[*edge_idx].as_ref().ok_or_else(|| {
                        crate::error::MetalError::DispatchFailed(format!(
                            "missing edge slice at index {edge_idx}"
                        ))
                    })?;
                    enc.set_buffer_with_offset(*idx, slice.buffer(), slice.byte_offset());
                }
                NativeBindingSource::Weight(name) => {
                    let buf = step_weights.get(name.as_str()).ok_or_else(|| {
                        crate::error::MetalError::DispatchFailed(format!(
                            "NativeEncoding step {step_idx}: missing weight '{name}'"
                        ))
                    })?;
                    enc.set_buffer_with_offset(*idx, buf, 0);
                }
                NativeBindingSource::Output => {
                    enc.set_buffer_with_offset(*idx, &out_buf, out_offset);
                }
                NativeBindingSource::Constant(bytes) => {
                    enc.set_bytes_raw(*idx, bytes);
                }
                NativeBindingSource::Intermediate(_)
                | NativeBindingSource::IntermediateAuxiliary { .. }
                | NativeBindingSource::PreResolved(_) => {
                    // These bindings are only valid within
                    // dispatch_native_encoding_sequence — not in single-
                    // encoding dispatch. Reaching here means a malformed plan.
                    return Err(crate::error::MetalError::DispatchFailed(format!(
                        "NativeEncoding step {step_idx}: Intermediate/PreResolved \
                         binding used outside multi-dispatch sequence"
                    )));
                }
            }
        }

        if encoding.threadgroup_memory_bytes > 0 {
            enc.set_threadgroup_memory_length(0, encoding.threadgroup_memory_bytes);
        }

        match encoding.dispatch_mode {
            NativeDispatchMode::Threads => {
                enc.encode(
                    encoding.pipeline.pipeline(),
                    encoding.grid,
                    encoding.threadgroup,
                )?;
            }
            NativeDispatchMode::Threadgroups => {
                enc.encode_threadgroups(
                    encoding.pipeline.pipeline(),
                    encoding.grid,
                    encoding.threadgroup,
                )?;
            }
        }
        enc.end_encoding();
        Ok::<(), crate::error::MetalError>(())
    });

    match scope_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(native_dispatch_err(
                step_idx,
                format!("NativeEncoding encode: {e}"),
            ))
        }
        Err(e) => return Err(e),
    }

    Ok(GpuSlice::from_ref(&out_buf, out_offset))
}

/// Auxiliary buffer allocated during a multi-dispatch sequence.
struct AuxSlice {
    slice: GpuSlice,
    exposed: bool,
}

/// Execute a sequence of [`NativeEncoding`] plans via direct Metal dispatch.
///
/// Returns the output `GpuSlice` of the **last** encoding. Intermediate outputs
/// from prior encodings are available via [`NativeBindingSource::Intermediate`],
/// and auxiliary buffers via [`NativeBindingSource::IntermediateAuxiliary`].
///
/// Each encoding gets its own compute encoder — Metal does not guarantee write
/// ordering within a single encoder across different threadgroups, so dependent
/// dispatches (stats → conv) require separate encoders.
///
/// Part of #3472 D3 (FusedResBlock multi-dispatch infrastructure).
pub(super) fn dispatch_native_encoding_sequence(
    model: &CompiledModel,
    encodings: &[NativeEncoding],
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    cache: &PipelineCache,
    pre_resolved: &[GpuSlice],
) -> Result<GpuSlice> {
    if encodings.is_empty() {
        return Err(native_dispatch_err(step_idx, "empty encoding sequence".into()));
    }

    let step_weights = &model.def.weight_buffers[step_idx];
    let ctx = cache.context();

    // Pre-resolve edge inputs (same as single dispatch).
    let mut edge_slices: Vec<Option<GpuSlice>> = Vec::new();
    for encoding in encodings {
        for (_, source) in &encoding.bindings {
            if let NativeBindingSource::Edge(edge_idx) = source {
                let needed = *edge_idx + 1;
                if edge_slices.len() < needed {
                    edge_slices.resize_with(needed, || None);
                }
                if edge_slices[*edge_idx].is_none() {
                    edge_slices[*edge_idx] =
                        Some(model.resolve_input_slice(step_idx, *edge_idx, buffers)?);
                }
            }
        }
    }

    // Track intermediate outputs and auxiliary buffers.
    let mut intermediate_outputs: Vec<GpuSlice> = Vec::with_capacity(encodings.len());
    let mut intermediate_auxiliaries: Vec<Vec<AuxSlice>> = Vec::with_capacity(encodings.len());

    for (enc_idx, encoding) in encodings.iter().enumerate() {
        // Allocate output buffer.
        let (out_buf, out_offset) =
            crate::arena::arena_alloc_or_create(ctx, encoding.output_bytes)
                .map_err(|e| native_dispatch_err(step_idx,
                    format!("sequence[{enc_idx}] output alloc: {e}")))?;

        // Allocate auxiliary buffers.
        let mut aux_slices: Vec<AuxSlice> = Vec::with_capacity(encoding.auxiliary_allocs.len());
        for (aux_idx, aux) in encoding.auxiliary_allocs.iter().enumerate() {
            let (aux_buf, aux_offset) =
                crate::arena::arena_alloc_or_create(ctx, aux.bytes)
                    .map_err(|e| native_dispatch_err(step_idx,
                        format!("sequence[{enc_idx}] aux[{aux_idx}] alloc: {e}")))?;
            aux_slices.push(AuxSlice {
                slice: GpuSlice::from_ref(&aux_buf, aux_offset),
                exposed: aux.expose_as_intermediate,
            });
        }

        // Blit-fill zero-fill auxiliaries, then encode compute dispatch.
        crate::gpu_scope::get_or_create_batch()?;
        let out_buf = out_buf.alias();
        let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| {
            // Zero-fill auxiliary buffers that need it.
            for (aux_idx, aux) in encoding.auxiliary_allocs.iter().enumerate() {
                if aux.zero_fill {
                    let s = &aux_slices[aux_idx].slice;
                    batch.blit_fill(s.buffer(), s.byte_offset(), aux.bytes, 0)?;
                }
            }

            // Encode compute dispatch.
            let enc = batch.new_encoder()?;

            for (idx, source) in &encoding.bindings {
                match source {
                    NativeBindingSource::Edge(edge_idx) => {
                        let slice = edge_slices[*edge_idx].as_ref().ok_or_else(|| {
                        crate::error::MetalError::DispatchFailed(format!(
                            "missing edge slice at index {edge_idx}"
                        ))
                    })?;
                        enc.set_buffer_with_offset(*idx, slice.buffer(), slice.byte_offset());
                    }
                    NativeBindingSource::Weight(name) => {
                        let buf = step_weights.get(name.as_str()).ok_or_else(|| {
                            crate::error::MetalError::DispatchFailed(format!(
                                "sequence[{enc_idx}] step {step_idx}: missing weight '{name}'"
                            ))
                        })?;
                        enc.set_buffer_with_offset(*idx, buf, 0);
                    }
                    NativeBindingSource::Output => {
                        enc.set_buffer_with_offset(*idx, &out_buf, out_offset);
                    }
                    NativeBindingSource::Constant(bytes) => {
                        enc.set_bytes_raw(*idx, bytes);
                    }
                    NativeBindingSource::Intermediate(src_idx) => {
                        let slice = &intermediate_outputs[*src_idx];
                        enc.set_buffer_with_offset(*idx, slice.buffer(), slice.byte_offset());
                    }
                    NativeBindingSource::IntermediateAuxiliary { encoding_idx, auxiliary_idx } => {
                        // Find the Nth exposed auxiliary from the referenced encoding.
                        let enc_auxes = &intermediate_auxiliaries[*encoding_idx];
                        let mut exposed_count = 0;
                        let mut found = false;
                        for aux in enc_auxes {
                            if aux.exposed {
                                if exposed_count == *auxiliary_idx {
                                    enc.set_buffer_with_offset(
                                        *idx,
                                        aux.slice.buffer(),
                                        aux.slice.byte_offset(),
                                    );
                                    found = true;
                                    break;
                                }
                                exposed_count += 1;
                            }
                        }
                        if !found {
                            return Err(crate::error::MetalError::DispatchFailed(format!(
                                "sequence[{enc_idx}] step {step_idx}: IntermediateAuxiliary \
                                 [{encoding_idx}][{auxiliary_idx}] not found"
                            )));
                        }
                    }
                    NativeBindingSource::PreResolved(pr_idx) => {
                        let slice = pre_resolved.get(*pr_idx).ok_or_else(|| {
                            crate::error::MetalError::DispatchFailed(format!(
                                "sequence[{enc_idx}] step {step_idx}: PreResolved[{pr_idx}] \
                                 out of range (have {})",
                                pre_resolved.len()
                            ))
                        })?;
                        enc.set_buffer_with_offset(*idx, slice.buffer(), slice.byte_offset());
                    }
                }
            }

            // Auxiliary buffer bindings (direct index, not via NativeBindingSource).
            for (aux_idx, aux) in encoding.auxiliary_allocs.iter().enumerate() {
                let s = &aux_slices[aux_idx].slice;
                enc.set_buffer_with_offset(aux.binding_index, s.buffer(), s.byte_offset());
            }

            if encoding.threadgroup_memory_bytes > 0 {
                enc.set_threadgroup_memory_length(0, encoding.threadgroup_memory_bytes);
            }

            match encoding.dispatch_mode {
                NativeDispatchMode::Threads => {
                    enc.encode(
                        encoding.pipeline.pipeline(),
                        encoding.grid,
                        encoding.threadgroup,
                    )?;
                }
                NativeDispatchMode::Threadgroups => {
                    enc.encode_threadgroups(
                        encoding.pipeline.pipeline(),
                        encoding.grid,
                        encoding.threadgroup,
                    )?;
                }
            }
            enc.end_encoding();
            Ok::<(), crate::error::MetalError>(())
        });

        match scope_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(native_dispatch_err(
                    step_idx,
                    format!("sequence[{enc_idx}] encode: {e}"),
                ))
            }
            Err(e) => return Err(e),
        }

        intermediate_outputs.push(GpuSlice::from_ref(&out_buf, out_offset));
        intermediate_auxiliaries.push(aux_slices);
    }

    // Return the last encoding's output.
    intermediate_outputs.pop().ok_or_else(||
        native_dispatch_err(step_idx, "empty encoding sequence (unreachable)".into()))
}
