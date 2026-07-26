// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sorting and selection operations for [`DynTensor`] — topk, arg_sort.
//!
//! Extracted from `ops_ext/mod.rs` for file-size compliance.

use super::{to_cpu, to_orig};
use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{gpu_backend, gpu_backend_dispatch_pair, Dim, DynTensor};
use crate::{Result, TensorError};

impl DynTensor {
    /// Select the top-k values and their indices along a dimension.
    ///
    /// Returns `(values, indices)` where both have the same shape as input
    /// except `dim` is replaced by `k`. `indices` is U32 dtype.
    /// Results are sorted descending by value within each slice.
    ///
    /// Used by MoE routing to select top-k experts per token.
    pub fn topk(&self, dim: impl Dim, k: usize) -> Result<(Self, Self)> {
        let dim = dim.to_index(self.rank())?;
        let dim_size = self.dim(dim)?;
        if k == 0 || k > dim_size {
            return Err(TensorError::InvalidShape(format!(
                "topk k={k} out of range for dim {dim} (size {dim_size})"
            )));
        }
        if dim_size > u32::MAX as usize {
            return Err(TensorError::InvalidShape(format!(
                "topk dim {dim} size {dim_size} exceeds u32::MAX"
            )));
        }
        let (mut values, indices) = self.topk_dispatch(dim, k, dim_size)?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Topk { k, dim },
                &input_ids,
                values.dims(),
                values.dtype(),
            ) {
                values.set_trace_id(id);
            }
        }
        Ok((values, indices))
    }

    /// Dispatch topk to GPU or CPU. Separated so trace recording wraps all paths.
    fn topk_dispatch(&self, dim: usize, k: usize, dim_size: usize) -> Result<(Self, Self)> {
        if self.device().is_gpu() {
            if let Ok(backend) = gpu_backend() {
                if let Some(result) = backend.topk(self, dim, k) {
                    return result;
                }
            }
        }
        self.topk_cpu(dim, k, dim_size)
    }

    /// CPU implementation of topk using partial sort.
    fn topk_cpu(&self, dim: usize, k: usize, dim_size: usize) -> Result<(Self, Self)> {
        let input_dtype = self.dtype;
        let (cpu_self, device) = to_cpu(self)?;
        let arr = cpu_self.to_f32_array()?;
        let nan_count = arr.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            return Err(TensorError::NonFiniteData {
                name: "topk input".into(),
                count: nan_count,
            });
        }
        let axis = ndarray::Axis(dim);
        let mut out_shape = self.dims().to_vec();
        out_shape[dim] = k;
        let mut val_arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&out_shape));
        let mut idx_arr = ndarray::ArrayD::<u32>::zeros(ndarray::IxDyn(&out_shape));
        // Partial sort O(D + k log k) instead of full sort O(D log D).
        let mut indexed: Vec<(usize, f32)> = Vec::with_capacity(dim_size);
        for (lane_in, (mut lane_val, mut lane_idx)) in arr.lanes(axis).into_iter().zip(
            val_arr
                .lanes_mut(axis)
                .into_iter()
                .zip(idx_arr.lanes_mut(axis)),
        ) {
            indexed.clear();
            indexed.extend(lane_in.iter().copied().enumerate());
            if k < indexed.len() {
                indexed.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
                indexed[..k].sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            } else {
                indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            }
            for (j, &(i, v)) in indexed.iter().take(k).enumerate() {
                lane_val[j] = v;
                lane_idx[j] = i as u32;
            }
        }
        let values = Self::from_f32_result(val_arr, input_dtype)?;
        let indices = Self::from_cpu_u32(idx_arr)?;
        Ok((to_orig(values, &device)?, to_orig(indices, &device)?))
    }

    /// Return the indices that would sort along a dimension.
    ///
    /// Returns a U32 tensor with the same shape, where each lane along `dim`
    /// contains the permutation indices that sort that lane.
    ///
    /// `ascending = false` gives descending order (largest first), matching
    /// the dvoice sampling pattern: `logits.arg_sort(D::Minus1, false)`.
    pub fn arg_sort(&self, dim: impl Dim, ascending: bool) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        let rank = self.rank();
        if rank == 0 {
            return Err(TensorError::InvalidShape(
                "arg_sort requires rank >= 1".into(),
            ));
        }
        let (cpu_self, device) = to_cpu(self)?;
        let arr = cpu_self.to_f32_array()?;
        let nan_count = arr.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            return Err(TensorError::NonFiniteData {
                name: "arg_sort input".into(),
                count: nan_count,
            });
        }
        let dim_size = self.dim(dim)?;
        // Indices are stored as U32; validate dimension fits.
        if dim_size > u32::MAX as usize {
            return Err(TensorError::InvalidShape(format!(
                "arg_sort dim {dim} size {dim_size} exceeds u32::MAX"
            )));
        }
        if dim_size == 0 {
            // Empty dimension — return empty U32 tensor with same shape.
            return Self::from_cpu_u32(ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(self.dims()),
                vec![],
            )?);
        }
        let axis = ndarray::Axis(dim);
        let mut idx_arr = ndarray::ArrayD::<u32>::zeros(ndarray::IxDyn(self.dims()));
        // Pre-allocate index buffer once outside lane loop.
        let mut indices: Vec<u32> = (0..dim_size as u32).collect();
        for (lane_in, mut lane_out) in arr.lanes(axis).into_iter().zip(idx_arr.lanes_mut(axis)) {
            // Reset indices to identity permutation (reuse allocation).
            for (i, idx) in indices.iter_mut().enumerate() {
                *idx = i as u32;
            }
            let lane_slice: Vec<f32> = lane_in.iter().copied().collect();
            if ascending {
                indices.sort_unstable_by(|&a, &b| {
                    lane_slice[a as usize].total_cmp(&lane_slice[b as usize])
                });
            } else {
                indices.sort_unstable_by(|&a, &b| {
                    lane_slice[b as usize].total_cmp(&lane_slice[a as usize])
                });
            }
            for (j, &idx) in indices.iter().enumerate() {
                lane_out[j] = idx;
            }
        }
        let result = Self::from_cpu_u32(idx_arr)?;
        let mut result = to_orig(result, &device)?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::ArgSort {
                    dim,
                    descending: !ascending,
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Sort values and their indices along a dimension.
    ///
    /// Returns `(values, indices)` where both have the same shape as input.
    /// `indices` is U32 dtype. When `descending` is true, largest values come first.
    ///
    /// Matches PyTorch `torch.sort(input, dim, descending)`.
    pub fn sort(&self, dim: impl Dim, descending: bool) -> Result<(Self, Self)> {
        let dim = dim.to_index(self.rank())?;
        let rank = self.rank();
        if rank == 0 {
            return Err(TensorError::InvalidShape("sort requires rank >= 1".into()));
        }
        let dim_size = self.dim(dim)?;
        if dim_size > u32::MAX as usize {
            return Err(TensorError::InvalidShape(format!(
                "sort dim {dim} size {dim_size} exceeds u32::MAX"
            )));
        }
        // Try GPU-native sort if tensor is on GPU.
        if self.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch_pair(|b| b.sort(self, dim, descending)) {
                let (mut values, indices) = result?;
                if trace::is_tracing() {
                    let input_ids = Self::trace_input_ids(&[self])?;
                    if let Some(id) = trace::record_op(
                        TraceOp::Sort { dim, descending },
                        &input_ids,
                        values.dims(),
                        values.dtype(),
                    ) {
                        values.set_trace_id(id);
                    }
                }
                return Ok((values, indices));
            }
        }
        let (cpu_self, device) = to_cpu(self)?;
        let arr = cpu_self.to_f32_array()?;
        let nan_count = arr.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            return Err(TensorError::NonFiniteData {
                name: "sort input".into(),
                count: nan_count,
            });
        }
        if dim_size == 0 {
            let vals = Self::from_f32_result(arr, self.dtype)?;
            let idxs = Self::from_cpu_u32(ndarray::ArrayD::<u32>::from_shape_vec(
                ndarray::IxDyn(self.dims()),
                vec![],
            )?)?;
            return Ok((to_orig(vals, &device)?, to_orig(idxs, &device)?));
        }
        let axis = ndarray::Axis(dim);
        let mut val_arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(self.dims()));
        let mut idx_arr = ndarray::ArrayD::<u32>::zeros(ndarray::IxDyn(self.dims()));
        let mut indexed: Vec<(usize, f32)> = Vec::with_capacity(dim_size);
        for (lane_in, (mut lane_val, mut lane_idx)) in arr.lanes(axis).into_iter().zip(
            val_arr
                .lanes_mut(axis)
                .into_iter()
                .zip(idx_arr.lanes_mut(axis)),
        ) {
            indexed.clear();
            indexed.extend(lane_in.iter().copied().enumerate());
            if descending {
                indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            } else {
                indexed.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            }
            for (j, &(i, v)) in indexed.iter().enumerate() {
                lane_val[j] = v;
                lane_idx[j] = i as u32;
            }
        }
        let mut values = Self::from_f32_result(val_arr, self.dtype)?;
        let indices = Self::from_cpu_u32(idx_arr)?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Sort { dim, descending },
                &input_ids,
                values.dims(),
                values.dtype(),
            ) {
                values.set_trace_id(id);
            }
        }
        Ok((to_orig(values, &device)?, to_orig(indices, &device)?))
    }

    /// Return the indices that would sort the last dimension.
    ///
    /// Deprecated: use [`arg_sort`](Self::arg_sort) with `D::Minus1` instead.
    #[deprecated(since = "0.1.0", note = "use arg_sort(D::Minus1, ascending) instead")]
    pub fn arg_sort_last_dim(&self, ascending: bool) -> Result<Self> {
        self.arg_sort(self.rank() - 1, ascending)
    }
}
