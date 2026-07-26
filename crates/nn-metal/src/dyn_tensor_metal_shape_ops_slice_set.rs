// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native slice_set extraction from `dyn_tensor_metal_shape_ops.rs`.
//!
//! Contains `gpu_slice_set()` which writes `src` into a region of `dst`.
//! Part of #1863 preemptive extraction.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Device, Result, TensorError};

use super::super::MetalTensorData;

impl super::super::MetalDynBackend {
    /// GPU-native slice_set: write `src` into a region of `dst` along `dim`.
    ///
    /// Clones the destination buffer and copies source data into the correct
    /// offset region. Uses shared-memory byte-level writes (no GPU kernel
    /// dispatch) since Metal buffers use `StorageModeShared`.
    ///
    /// Supports F32, BF16, and F16 via byte-level copies — element byte width
    /// is derived from `DType::size_bytes()`. Both tensors must share the same
    /// dtype (#1711).
    pub(crate) fn gpu_slice_set(
        dst: &DynTensor,
        dim: usize,
        offset: usize,
        src: &DynTensor,
    ) -> Result<DynTensor> {
        // Lazy batch (#2009): flush pending GPU work before CPU readback.
        crate::gpu_scope::flush()?;
        Self::validate_same_float_dtype(dst, src, "gpu_slice_set")?;

        let elem_bytes = dst.dtype().size_bytes();

        let dst_data = dst.gpu_data::<MetalTensorData>()?;
        let src_data = src.gpu_data::<MetalTensorData>()?;

        let dst_numel = crate::metal_backend::checked_dim_product(dst.dims())?;
        let dst_logical_bytes =
            dst_numel
                .checked_mul(elem_bytes)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dst.dims().to_vec(),
                })?;

        let ctx = Self::ctx()?;
        // Clone only the logical region of dst, respecting byte_offset from
        // narrow views (#1969). The output buffer starts at byte 0.
        let mut out_buf = ctx
            .clone_buffer_range(&dst_data.buffer, dst_data.byte_offset, dst_logical_bytes)
            .map_err(|e| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::OutOfMemory,
                    format!("gpu_slice_set clone: {e}"),
                )
            })?;

        let dst_shape = dst.dims();
        let src_shape = src.dims();
        let rank = dst_shape.len();

        let overflow = || TensorError::DimensionOverflow {
            dims: dst_shape.to_vec(),
        };
        let checked = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);

        // Row-major strides: stride[i] = product of dims after i.
        let mut strides: Vec<usize> = vec![1; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            strides[i] = checked(strides[i + 1], dst_shape[i + 1])?;
        }

        // inner_count = contiguous elements per position along dim.
        let inner_count = strides[dim];
        let src_dim_len = src_shape[dim];
        // outer_count = product of all dims before dim.
        let outer_count: usize = if dim == 0 {
            1
        } else {
            crate::metal_backend::checked_dim_product(&dst_shape[..dim])?
        };
        // Elements per outer iteration in dst and src.
        let dst_outer_stride = checked(dst_shape[dim], inner_count)?;
        let src_outer_stride = checked(src_dim_len, inner_count)?;

        let src_full_bytes: &[u8] = src_data.buffer.contents::<u8>().map_err(|e| {
            TensorError::backend_failure(
                nn_core::BackendDomain::Metal,
                nn_core::BackendErrorKind::DispatchFailed,
                format!("gpu_slice_set src read: {e}"),
            )
        })?;
        // Slice past src byte_offset to read from the logical start of the
        // source tensor, not the underlying buffer start (#1969).
        let src_bytes = &src_full_bytes[src_data.byte_offset..];
        // SAFETY: out_buf was just cloned — we have exclusive ownership. No other
        // references exist, no GPU work is pending on this buffer. contents_mut()
        // requires &mut which proves exclusive access at the Rust level.
        let out_bytes: &mut [u8] = unsafe {
            out_buf.contents_mut::<u8>().map_err(|e| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::DispatchFailed,
                    format!("gpu_slice_set dst write: {e}"),
                )
            })?
        };

        let copy_len = checked(src_dim_len, inner_count)?;
        let byte_len = checked(copy_len, elem_bytes)?;
        let offset_elems = checked(offset, inner_count)?;

        for outer in 0..outer_count {
            let dst_base = checked(
                outer
                    .checked_mul(dst_outer_stride)
                    .and_then(|v| v.checked_add(offset_elems))
                    .ok_or_else(overflow)?,
                elem_bytes,
            )?;
            let src_base = checked(checked(outer, src_outer_stride)?, elem_bytes)?;

            // Bounds are validated by the caller's validate_slice_set_args.
            out_bytes[dst_base..dst_base + byte_len]
                .copy_from_slice(&src_bytes[src_base..src_base + byte_len]);
        }

        let storage = MetalTensorData::new(out_buf);
        DynTensor::from_gpu_storage(
            dst_shape.to_vec(),
            dst.dtype(),
            Arc::new(storage),
            Device::metal(),
        )
    }
}
