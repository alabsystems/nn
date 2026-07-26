// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native concatenation for [`MetalDynBackend`].
//!
//! Extracted from `dyn_tensor_metal_shape_ops.rs` to keep files under
//! the 400-line maintenance threshold.
//!
//! Part of #1084 (GPU shape ops).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{check_dim, Result, TensorError};

use nn_dsl::TensorBlockBuilder;

use super::MetalTensorData;

impl super::MetalDynBackend {
    /// GPU-native concatenation along a dimension.
    ///
    /// All input tensors must be on GPU. Creates named inputs ("t0", "t1", ...)
    /// and dispatches a single concat kernel.
    ///
    /// When `dim == 0`, the builder shapes are prepended with a leading dim of 1
    /// and the concat axis is shifted to 1, because axis 0 is reserved for
    /// NY multi-variable stacking in the tensor verification path.
    /// The actual GPU buffer layout is unchanged — only the builder's shape
    /// representation differs.
    pub(super) fn gpu_cat(tensors: &[&DynTensor], dim: usize) -> Result<DynTensor> {
        if tensors.is_empty() {
            return Err(TensorError::InvalidShape(
                "gpu_cat: requires at least one tensor".into(),
            ));
        }
        let first = tensors[0];
        Self::validate_f32(first, "gpu_cat")?;
        let rank = first.dims().len();

        check_dim(dim, rank)?;

        // Single tensor: return as-is (matches PyTorch and CPU semantics).
        if tensors.len() == 1 {
            return first.contiguous();
        }

        for t in tensors.iter().skip(1) {
            Self::validate_f32(t, "gpu_cat")?;
            // All tensors must share the same dtype. dispatch_def uses a single
            // dtype for all input buffers; mixed F32/BF16 would read 2-byte
            // buffers as 4-byte floats, causing silent data corruption.
            if t.dtype() != first.dtype() {
                return Err(TensorError::dtype_mismatch(first.dtype(), t.dtype()));
            }
            if t.dims().len() != rank {
                return Err(TensorError::RankMismatch {
                    expected: rank,
                    actual: t.dims().len(),
                });
            }
            for d in 0..rank {
                if d != dim && t.dims()[d] != first.dims()[d] {
                    return Err(TensorError::shape_mismatch(
                        first.dims().to_vec(),
                        t.dims().to_vec(),
                    ));
                }
            }
        }

        // Compute output shape: sum along concat dim (checked for overflow).
        let mut out_shape: Vec<usize> = first.dims().to_vec();
        let mut total_dim = first.dims()[dim];
        for t in tensors.iter().skip(1) {
            total_dim = total_dim.checked_add(t.dims()[dim]).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: vec![total_dim, t.dims()[dim]],
                }
            })?;
        }
        out_shape[dim] = total_dim;

        // Axis 0 is reserved in the tensor IR validation for NY
        // multi-variable stacking. For pure DynTensor cat, shift shapes and
        // axis by prepending a leading dim of 1 when dim == 0.
        let shift = if dim == 0 { 1 } else { 0 };
        let builder_axis = dim + shift;

        let prepend_dim = |shape: &[usize]| -> Vec<usize> {
            if shift == 0 {
                return shape.to_vec();
            }
            let mut s = Vec::with_capacity(shape.len() + 1);
            s.push(1);
            s.extend_from_slice(shape);
            s
        };

        let builder_out_shape = prepend_dim(&out_shape);

        // Build cache key from all input shapes + dim.
        let shapes: Vec<&[usize]> = tensors.iter().map(|t| t.dims()).collect();
        // Collect GPU data references before the closure (can't borrow tensors inside).
        let mut input_buffers: Vec<(String, &MetalTensorData)> = Vec::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            input_buffers.push((format!("t{i}"), t.gpu_data::<MetalTensorData>()?));
        }

        let def = crate::kernel_def_cache::get_or_build(
            "cat",
            &shapes,
            &[dim as u64, tensors.len() as u64],
            first.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_cat");
                let mut input_nodes = Vec::with_capacity(tensors.len());

                for (i, t) in tensors.iter().enumerate() {
                    let name = format!("t{i}");
                    let builder_shape = prepend_dim(t.dims());
                    let node = b.add_input(&name, &builder_shape);
                    input_nodes.push(node);
                }

                let out = b.add_concat(&input_nodes, builder_axis, &builder_out_shape);
                crate::build_kernel(b, out)
            },
        )?;

        let dispatch_inputs: Vec<(&str, crate::gpu_slice::GpuSlice)> = input_buffers
            .iter()
            .map(|(name, data)| (name.as_str(), data.as_gpu_slice()))
            .collect();

        Self::dispatch_def(&def, &dispatch_inputs, &out_shape, first.dtype())
    }
}
