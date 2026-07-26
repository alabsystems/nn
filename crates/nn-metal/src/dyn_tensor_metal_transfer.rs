// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU ↔ CPU transfer for Metal DynTensor backend.
//!
//! Extracted from `dyn_tensor_metal_helpers.rs` to keep it under 450 lines (#3243).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use super::super::MetalTensorData;

/// Compute a validated element range from byte offset + element count.
///
/// Combines alignment check, offset conversion, overflow-safe end computation,
/// and bounds validation into a single call. Returns `(start, end)` indices
/// for slicing the buffer.
///
/// Defense-in-depth: `byte_offset / elem_size` with integer division silently
/// truncates misaligned offsets, producing wrong element indices. Construction
/// guarantees alignment (only `gpu_narrow_contiguous_view` creates non-zero offsets,
/// and it multiplies `start * stride_d * elem_bytes`), but this guard catches
/// any future misalignment bugs before they corrupt data reads.
pub(super) fn validated_elem_range(
    byte_offset: usize,
    elem_size: usize,
    numel: usize,
    buf_len: usize,
) -> Result<(usize, usize)> {
    if !byte_offset.is_multiple_of(elem_size) {
        return Err(TensorError::ValueOutOfRange {
            description: "byte_offset is not aligned to element size",
        });
    }
    let start = byte_offset / elem_size;
    let end = start
        .checked_add(numel)
        .ok_or(TensorError::DimensionOverflow {
            dims: vec![start, numel],
        })?;
    if end > buf_len {
        return Err(TensorError::DataLengthMismatch {
            expected: end,
            actual: buf_len,
        });
    }
    Ok((start, end))
}

impl super::super::MetalDynBackend {
    /// Transfer a GPU DynTensor to CPU by reading its MetalBuffer.
    ///
    /// F32 buffers are 4 bytes/element, read as f32 directly.
    /// BF16/F16 buffers are 2 bytes/element (f16 encoding), converted back
    /// to native half-precision CPU storage (#1646 D7).
    /// U8 tensors are stored as f32 on GPU; this reconstructs the U8 data.
    ///
    /// Byte-offset aware: zero-copy views (#1945) share the parent buffer
    /// with `byte_offset > 0`. This function reads from the correct position
    /// by computing `elem_offset = byte_offset / elem_size` and slicing
    /// `[elem_offset..elem_offset + numel]`.
    pub(in super::super) fn gpu_to_cpu(x: &DynTensor) -> Result<DynTensor> {
        // Lazy batch (#2009): flush pending GPU work before CPU readback.
        // Note: flush() resets the default arena (gen++), so the arena
        // generation check below compares against the post-flush state.
        crate::gpu_scope::flush()?;
        let data = x.gpu_data::<MetalTensorData>()?;

        // Stale arena read detection (#2328): if this tensor was allocated
        // from the arena, verify its generation is still recent. After flush(),
        // the arena gen increments by 1 — a tensor from the just-committed
        // batch (gen N, arena now N+1) is safe. A tensor from gen N where
        // arena is at N+2 or higher means at least one full generation of
        // new allocations has occurred, potentially overwriting this memory.
        //
        // Decode scope exception (#3359): autoregressive decode loops advance
        // the arena generation on each flush, but tensors from earlier decode
        // steps are still valid (ObjC ARC keeps buffers alive). If a decode
        // scope is active and the tensor was allocated within it, skip the
        // stale check.
        if let Some(alloc_gen) = data.arena_generation() {
            // Decode scope: tensors allocated at or after the scope's start
            // generation are considered non-stale for the scope's duration.
            let in_decode_scope = crate::arena::decode_scope_generation()
                .is_some_and(|scope_gen| alloc_gen >= scope_gen);

            if !in_decode_scope {
                if let Some(current_gen) = crate::arena::default_arena_generation() {
                    let stale = if current_gen > alloc_gen + 1 {
                        // Multiple generations passed — definitely stale.
                        true
                    } else if current_gen == alloc_gen + 1 {
                        // One generation passed. Safe only if no new allocs
                        // have overwritten the region (used_bytes == 0).
                        crate::arena::default_arena_used_bytes().unwrap_or(0) > 0
                    } else {
                        false
                    };
                    if stale {
                        return Err(crate::error::MetalError::StaleArenaRead {
                            alloc_gen,
                            current_gen,
                        }
                        .into());
                    }
                }
            }
        }
        let numel = x.checked_numel()?;
        let byte_offset = data.byte_offset;
        // U32 tensors (e.g., argmax/argmin output): read as u32 directly.
        if x.dtype() == DType::U32 {
            let u32s: &[u32] = data.buffer.contents::<u32>().map_err(|e| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::DispatchFailed,
                    e.to_string(),
                )
            })?;
            let (start, end) = validated_elem_range(byte_offset, 4, numel, u32s.len())?;
            return DynTensor::from_vec_u32(u32s[start..end].to_vec(), x.dims(), &Device::Cpu);
        }
        match x.dtype() {
            // BF16/F16: read 2-byte f16 Metal buffer, convert to native storage (#1646 D7).
            DType::BF16 | DType::F16 => {
                let u16s: &[u16] = data.buffer.contents::<u16>().map_err(|e| {
                    TensorError::backend_failure(
                        nn_core::BackendDomain::Metal,
                        nn_core::BackendErrorKind::DispatchFailed,
                        e.to_string(),
                    )
                })?;
                let (start, end) = validated_elem_range(byte_offset, 2, numel, u16s.len())?;
                // Convert f16 bits → f32 → native dtype storage.
                let f32_data: Vec<f32> = u16s[start..end]
                    .iter()
                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                    .collect();
                let f32_tensor = DynTensor::from_vec(f32_data, x.dims(), &Device::Cpu)?;
                f32_tensor.to_dtype(x.dtype())
            }
            // F32: read 4-byte f32 Metal buffer directly.
            DType::F32 => {
                let floats: &[f32] = data.buffer.contents::<f32>().map_err(|e| {
                    TensorError::backend_failure(
                        nn_core::BackendDomain::Metal,
                        nn_core::BackendErrorKind::DispatchFailed,
                        e.to_string(),
                    )
                })?;
                let (start, end) = validated_elem_range(byte_offset, 4, numel, floats.len())?;
                DynTensor::from_vec(floats[start..end].to_vec(), x.dims(), &Device::Cpu)
            }
            // U8 stored as f32 on GPU: read f32 buffer, reconstruct U8 data.
            DType::U8 => {
                let floats: &[f32] = data.buffer.contents::<f32>().map_err(|e| {
                    TensorError::backend_failure(
                        nn_core::BackendDomain::Metal,
                        nn_core::BackendErrorKind::DispatchFailed,
                        e.to_string(),
                    )
                })?;
                let (start, end) = validated_elem_range(byte_offset, 4, numel, floats.len())?;
                let u8_data: Vec<u8> = floats[start..end].iter().map(|&v| v as u8).collect();
                DynTensor::from_vec_u8(u8_data, x.dims(), &Device::Cpu)
            }
            // I64, F64, I32, Bool, and future variants: not supported on Metal
            // GPU. Reject with a clear error instead of silently reinterpreting
            // the buffer as f32 (#1697 F1).
            other => Err(TensorError::Unsupported(format!(
                "gpu_to_cpu for dtype {other:?}",
            ))),
        }
    }

    /// Transfer a CPU DynTensor to Metal by creating a MetalBuffer.
    ///
    /// F32 tensors are stored as f32 (4 bytes/element) in Metal buffers.
    /// BF16/F16 tensors are stored as f16 (2 bytes/element) in Metal buffers
    /// (#1646 D7). MSL kernels use `half` buffer types with `float` accumulators.
    /// U8 tensors are promoted to f32 for GPU storage.
    pub(in super::super) fn cpu_to_gpu(x: &DynTensor) -> Result<DynTensor> {
        // Zero-element tensors: Metal cannot create zero-size buffers.
        // Create a 4-byte sentinel GPU buffer so the result is device-correct
        // (callers like compiled_model_execute_runtime.rs:131 expect GPU tensors).
        // The shape records zero elements, so no kernel will read from the buffer.
        if x.elem_count() == 0 {
            let ctx = Self::ctx()?;
            let sentinel = ctx.create_buffer_zeroed(4).map_err(|e| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::DispatchFailed,
                    e.to_string(),
                )
            })?;
            let storage = MetalTensorData::new(sentinel);
            return DynTensor::from_gpu_storage(
                x.dims().to_vec(),
                x.dtype(),
                Arc::new(storage),
                Device::metal(),
            );
        }
        let ctx = Self::ctx()?;
        // U32 tensors: store as native u32 in Metal buffer.
        // gpu_to_cpu reads U32 buffers with contents::<u32>(), so the buffer
        // must contain raw u32 values (not f32 reinterpretations).
        if x.dtype() == DType::U32 {
            // Extract u32 values natively — no F32 intermediate, so values
            // >= 2^24 are preserved exactly (unlike U32→F32→u32 which loses
            // precision due to F32's 24-bit mantissa).
            let u32_native: Vec<u32> = x.to_flat_vec::<u32>()?;
            let buffer = ctx.create_buffer(&u32_native).map_err(|e| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::DispatchFailed,
                    e.to_string(),
                )
            })?;
            let storage = MetalTensorData::new(buffer);
            return DynTensor::from_gpu_storage(
                x.dims().to_vec(),
                DType::U32,
                Arc::new(storage),
                Device::metal(),
            );
        }
        match x.dtype() {
            // BF16/F16: convert to f16 encoding, store as 2 bytes/element (#1646 D7).
            // bf16→f16 conversion at Metal boundary: Apple GPUs have no native
            // bf16 ALU, so both bf16 and f16 map to MSL `half` (#1646 D8).
            DType::BF16 | DType::F16 => {
                let f32_arr = x.to_f32_array()?;
                let cow = f32_arr.as_standard_layout();
                let f32_data = cow.as_slice().ok_or_else(|| {
                    TensorError::InvalidShape(
                        "tensor data not contiguous after as_standard_layout".into(),
                    )
                })?;
                // Convert f32 → f16 bits (u16) for 2-byte Metal buffer.
                let f16_data: Vec<half::f16> =
                    f32_data.iter().map(|&v| half::f16::from_f32(v)).collect();
                let buffer = ctx.create_buffer(&f16_data).map_err(|e| {
                    TensorError::backend_failure(
                        nn_core::BackendDomain::Metal,
                        nn_core::BackendErrorKind::DispatchFailed,
                        e.to_string(),
                    )
                })?;
                let storage = MetalTensorData::new(buffer);
                DynTensor::from_gpu_storage(
                    x.dims().to_vec(),
                    x.dtype(),
                    Arc::new(storage),
                    Device::metal(),
                )
            }
            // F32: store as 4 bytes/element directly.
            DType::F32 => {
                let f32_arr = x.to_f32_array()?;
                let cow = f32_arr.as_standard_layout();
                let data = cow.as_slice().ok_or_else(|| {
                    TensorError::InvalidShape(
                        "tensor data not contiguous after as_standard_layout".into(),
                    )
                })?;
                let buffer = ctx.create_buffer(data).map_err(|e| {
                    TensorError::backend_failure(
                        nn_core::BackendDomain::Metal,
                        nn_core::BackendErrorKind::DispatchFailed,
                        e.to_string(),
                    )
                })?;
                let storage = MetalTensorData::new(buffer);
                DynTensor::from_gpu_storage(
                    x.dims().to_vec(),
                    DType::F32,
                    Arc::new(storage),
                    Device::metal(),
                )
            }
            // U8: promote to f32 for GPU storage.
            DType::U8 => {
                let f32_arr = x.to_dtype(DType::F32)?.to_f32_array()?;
                let cow = f32_arr.as_standard_layout();
                let data = cow.as_slice().ok_or_else(|| {
                    TensorError::InvalidShape(
                        "tensor data not contiguous after as_standard_layout".into(),
                    )
                })?;
                let buffer = ctx.create_buffer(data).map_err(|e| {
                    TensorError::backend_failure(
                        nn_core::BackendDomain::Metal,
                        nn_core::BackendErrorKind::DispatchFailed,
                        e.to_string(),
                    )
                })?;
                let storage = MetalTensorData::new(buffer);
                DynTensor::from_gpu_storage(
                    x.dims().to_vec(),
                    DType::U8,
                    Arc::new(storage),
                    Device::metal(),
                )
            }
            // I64, F64, I32, Bool, and future variants: not supported for
            // Metal GPU transfer. Reject explicitly (#1697 F1).
            other => Err(TensorError::Unsupported(format!(
                "cpu_to_gpu for dtype {other:?}",
            ))),
        }
    }
}
