// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Top-k selection, argmax, and argmin operations for [`DynTensor`].
//!
//! Provides extended variants with `largest`/`sorted`/`keepdim` parameters
//! for language model sampling (top-k sampling) and loss computation.
//!
//! The core `topk`, `argmax`, and `argmin` methods live in `ops_ext/`.
//! This module adds the full-parameter variants needed for LLM inference.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{Dim, DynTensor};
use crate::{Device, Result, TensorError};

impl DynTensor {
    /// Select the top-k values and their indices along a dimension.
    ///
    /// Returns `(values, indices)` where both have the same shape as input
    /// except `dim` is replaced by `k`. `indices` is U32 dtype.
    ///
    /// # Arguments
    /// * `k` — number of elements to select
    /// * `dim` — dimension to select along (supports negative indexing via `Dim`)
    /// * `largest` — if `true`, select the k largest values; if `false`, k smallest
    /// * `sorted` — if `true`, results are sorted by value (descending for largest,
    ///   ascending for smallest); if `false`, order is unspecified
    ///
    /// # Errors
    /// * `k == 0` or `k > dim_size` — returns `InvalidShape`
    /// * Input contains NaN — returns `NonFiniteData`
    ///
    /// Used by LLM top-k sampling, MoE expert routing, and loss computation.
    pub fn topk_ext(
        &self,
        k: usize,
        dim: impl Dim,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self, Self)> {
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

        // Fast path: largest + sorted matches the existing GPU-accelerated topk.
        if largest && sorted {
            return self.topk(dim, k);
        }

        let (mut values, indices) = self.topk_ext_cpu(dim, k, dim_size, largest, sorted)?;
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

    /// CPU implementation of topk with full parameter support.
    fn topk_ext_cpu(
        &self,
        dim: usize,
        k: usize,
        dim_size: usize,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self, Self)> {
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
        let mut indexed: Vec<(usize, f32)> = Vec::with_capacity(dim_size);

        for (lane_in, (mut lane_val, mut lane_idx)) in arr.lanes(axis).into_iter().zip(
            val_arr
                .lanes_mut(axis)
                .into_iter()
                .zip(idx_arr.lanes_mut(axis)),
        ) {
            indexed.clear();
            indexed.extend(lane_in.iter().copied().enumerate());

            // Comparator for partial sort: largest→descending, smallest→ascending.
            let cmp = if largest {
                |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1)
            } else {
                |a: &(usize, f32), b: &(usize, f32)| a.1.total_cmp(&b.1)
            };

            if k < indexed.len() {
                // Partial sort: O(n + k log k) — only sort the top k.
                indexed.select_nth_unstable_by(k - 1, cmp);
                if sorted {
                    indexed[..k].sort_unstable_by(cmp);
                }
            } else if sorted {
                indexed.sort_unstable_by(cmp);
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

    /// Index of the maximum value along `dim` with `keepdim` control.
    ///
    /// Returns an integer (U32) tensor of indices. When `keepdim` is true,
    /// the reduced dimension is retained with size 1.
    ///
    /// # Arguments
    /// * `dim` — dimension to reduce along (supports negative indexing via `Dim`)
    /// * `keepdim` — if `true`, output has same rank as input with `dim` size 1
    ///
    /// # Errors
    /// * Dimension has zero size — returns `ZeroLengthDimension`
    /// * Input contains NaN — returns `NonFiniteData`
    pub fn argmax_ext(&self, dim: impl Dim, keepdim: bool) -> Result<Self> {
        if keepdim {
            self.argmax_keepdim(dim)
        } else {
            self.argmax(dim)
        }
    }

    /// Index of the minimum value along `dim` with `keepdim` control.
    ///
    /// Returns an integer (U32) tensor of indices. When `keepdim` is true,
    /// the reduced dimension is retained with size 1.
    ///
    /// # Arguments
    /// * `dim` — dimension to reduce along (supports negative indexing via `Dim`)
    /// * `keepdim` — if `true`, output has same rank as input with `dim` size 1
    ///
    /// # Errors
    /// * Dimension has zero size — returns `ZeroLengthDimension`
    /// * Input contains NaN — returns `NonFiniteData`
    pub fn argmin_ext(&self, dim: impl Dim, keepdim: bool) -> Result<Self> {
        if keepdim {
            self.argmin_keepdim(dim)
        } else {
            self.argmin(dim)
        }
    }
}

/// Move a tensor to CPU for computation, returning the original device.
fn to_cpu(t: &DynTensor) -> Result<(DynTensor, Device)> {
    let device = t.device();
    let cpu = if device.is_gpu() {
        t.to_device(&Device::Cpu)?
    } else {
        t.clone()
    };
    Ok((cpu, device))
}

/// Move a CPU result back to the original device if needed.
fn to_orig(t: DynTensor, device: &Device) -> Result<DynTensor> {
    if device.is_gpu() {
        t.to_device(device)
    } else {
        Ok(t)
    }
}
