// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native narrow (slice) extraction from `dyn_tensor_metal_shape_ops.rs`.
//!
//! Contains `gpu_narrow()` with zero-copy contiguous views (#1945, #2007).
//! Part of #1863 preemptive extraction.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{check_dim, Device, Result, TensorError};

use crate::metal_backend::checked_dim_product;

use nn_dsl::TensorBlockBuilder;

use super::super::MetalTensorData;

impl super::super::MetalDynBackend {
    /// GPU-native narrow (slice) along a dimension.
    ///
    /// Returns a zero-copy buffer view when the narrow produces a contiguous byte
    /// range in the row-major buffer. This applies when:
    /// - dim == 0 (always contiguous), OR
    /// - all dimensions before dim are 1 (single stripe of data)
    ///
    /// Zero-copy views use `MetalBuffer::alias()` with an adjusted byte offset —
    /// no kernel dispatch, no buffer allocation, no memcpy. O(1) metadata-only.
    /// Metal's `setBuffer(_:offset:atIndex:)` natively supports byte offsets for
    /// buffer binding, so downstream dispatch uses the view's offset directly.
    ///
    /// For non-contiguous narrows (e.g., dim-1 of `[B, C, T]` with B>1), falls
    /// back to GPU kernel dispatch via TensorBlockBuilder.
    pub(crate) fn gpu_narrow(
        x: &DynTensor,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_narrow")?;
        let shape = x.dims();

        // Defense-in-depth: validate dimension bounds before indexing.
        check_dim(dim, shape.len())?;
        let end = start
            .checked_add(len)
            .ok_or(TensorError::DimensionOverflow {
                dims: vec![start, len],
            })?;
        if end > shape[dim] {
            return Err(TensorError::ValueOutOfRange {
                description: "gpu_narrow: start + len exceeds dimension size",
            });
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;

        let mut out_shape: Vec<usize> = shape.to_vec();
        out_shape[dim] = len;

        // Fast path: zero-copy view when narrow produces a contiguous byte range.
        // In row-major layout, narrow along dim d is contiguous when all leading
        // dimensions (0..d) are 1 — there is only one "stripe" of data, so the
        // narrowed elements form a single contiguous region starting at
        // parent_offset + start * stride_d * elem_bytes.
        //
        // This covers:
        //   dim==0: always (the standard case)
        //   dim==1 when shape[0]==1: e.g., [1, N, D] → [1, len, D]
        //   dim==2 when shape[0]==shape[1]==1: e.g., [1, 1, T, D] → [1, 1, len, D]
        // Skip for empty narrow (len==0) and scalar tensors (no dims to narrow).
        if len > 0 && !shape.is_empty() && Self::is_narrow_contiguous(shape, dim) {
            return Self::gpu_narrow_contiguous_view(x, x_data, shape, dim, start, &out_shape);
        }

        let def = crate::kernel_def_cache::get_or_build(
            "narrow",
            &[shape],
            &[dim as u64, start as u64, len as u64],
            x.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_narrow");
                let input = b.add_input("data", shape);
                let out = b.add_narrow(input, dim, start, len, &out_shape);
                crate::build_kernel(b, out)
            },
        )?;

        Self::dispatch_def(
            &def,
            &[("data", x_data.as_gpu_slice())],
            &out_shape,
            x.dtype(),
        )
    }

    /// Check if narrow along `dim` produces a contiguous byte range.
    ///
    /// In row-major layout, narrow along dim `d` is contiguous when the product
    /// of all leading dimensions `shape[..d]` equals 1 (i.e., there is exactly
    /// one "stripe" of data). This is equivalent to all leading dims being 1
    /// for positive shapes.
    ///
    /// This covers:
    ///   - dim==0: always (empty product = 1)
    ///   - dim==1 when shape[0]==1: e.g., [1, N, D] -> [1, len, D]
    ///   - dim==2 when shape[0]==shape[1]==1: e.g., [1, 1, T, D]
    ///
    /// **Why last-dim narrow on multi-row tensors is NOT contiguous (#4319):**
    /// For shape [1, 1024, 2304] narrow dim=2, each of the 1024 rows contributes
    /// `len` elements, but they are separated by `(2304 - len)` element gaps.
    /// Row 0 occupies bytes [0, len*4), row 1's narrowed data starts at byte
    /// 2304*4 — not at len*4. A byte-offset view would misinterpret the data
    /// because it assumes dense row-major layout. Eliminating these GPU dispatches
    /// requires strided view support in MetalTensorData and the dispatch pipeline.
    fn is_narrow_contiguous(shape: &[usize], dim: usize) -> bool {
        shape[..dim].iter().all(|&s| s == 1)
    }

    /// Zero-copy narrow via buffer view with byte offset.
    ///
    /// For a row-major tensor, narrow along dim `d` selects a contiguous byte
    /// range when all dims before `d` are 1 (verified by `is_narrow_contiguous`).
    /// The byte offset is `parent_offset + start * stride_d * elem_bytes` where
    /// `stride_d = product(shape[d+1..])`.
    ///
    /// Returns a `MetalTensorData` view sharing the parent buffer (Arc bump)
    /// with adjusted byte offset. No GPU kernel, no command buffer, no memcpy.
    /// Propagates parent's arena generation for stale-read detection (#2328).
    ///
    /// Chained views compose offsets: `narrow(d, 2, 3).narrow(d, 1, 1)` produces
    /// a view at `parent_offset + (2+1) * stride_d * elem_bytes`.
    ///
    /// GpuScope-compatible: no buffer reads, only metadata computation.
    fn gpu_narrow_contiguous_view(
        x: &DynTensor,
        x_data: &MetalTensorData,
        shape: &[usize],
        dim: usize,
        start: usize,
        out_shape: &[usize],
    ) -> Result<DynTensor> {
        let elem_bytes = x.dtype().size_bytes();

        // stride_d = product of all dims after dim d = elements per dim-d slice.
        let stride_d: usize = checked_dim_product(&shape[dim + 1..])?;
        let start_byte_offset = start
            .checked_mul(stride_d)
            .and_then(|v| v.checked_mul(elem_bytes))
            .ok_or(TensorError::DimensionOverflow {
                dims: vec![start, stride_d, elem_bytes],
            })?;

        // Compose with parent's byte offset (supports chained views).
        let new_offset = x_data.byte_offset.checked_add(start_byte_offset).ok_or(
            TensorError::DimensionOverflow {
                dims: vec![x_data.byte_offset, start_byte_offset],
            },
        )?;

        // Validate the view's byte range fits within the buffer.
        let view_byte_len = checked_dim_product(out_shape)?
            .checked_mul(elem_bytes)
            .ok_or(TensorError::DimensionOverflow {
                dims: out_shape.to_vec(),
            })?;
        let view_end =
            new_offset
                .checked_add(view_byte_len)
                .ok_or(TensorError::DimensionOverflow {
                    dims: vec![new_offset, view_byte_len],
                })?;
        if view_end > x_data.buffer.len() {
            return Err(TensorError::ValueOutOfRange {
                description: "gpu_narrow_contiguous_view: view byte range exceeds buffer",
            });
        }

        // Zero-copy: alias the parent buffer (Arc bump) with adjusted offset.
        // Propagate arena generation from the parent tensor so stale-read
        // detection survives through narrow views (#2328 defense-in-depth).
        let storage = match x_data.arena_generation() {
            Some(g) => MetalTensorData::view_arena(x_data.buffer.alias(), new_offset, g),
            None => MetalTensorData::view(x_data.buffer.alias(), new_offset),
        };
        DynTensor::from_gpu_storage(
            out_shape.to_vec(),
            x.dtype(),
            Arc::new(storage),
            Device::metal(),
        )
    }
}
