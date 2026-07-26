// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Selection and indexing operations for [`DynTensor`].
//!
//! Provides `index_select`, `gather`, and `expand`.
//! Conditional selection (`where_cond`) lives in `where_cond.rs`.
//! Accumulation ops (`scatter_add`, `index_add`) live in `accumulate.rs`.
//! Comparison ops (eq/ne/ge/gt/lt/le) live in `compare.rs`.
//! Dtype conversion (`to_dtype`) lives in `dtype_convert.rs`.
//! U32/U8 typed storage lives in `dyn_tensor_typed_storage.rs`.

use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend_dispatch, trace, Dim, DynTensor, FloatStorage, TensorStorage};
use crate::tensor::checked_dim_product;
use crate::{DType, Device, Result, TensorError};
use ndarray::{ArcArray, ArrayD, IxDyn};
use std::any::Any;
use std::sync::Arc;

mod accumulate;
mod compare;
mod dtype_convert;
mod index_put;
mod where_cond;

// dispatch_cpu_typed! macro is defined in parent module (dyn_tensor/mod.rs).

// -- Zero-copy f32 extraction (shared by `_into` accumulation variants) -------

/// Try to extract the underlying f32 array from a `DynTensor` without cloning.
///
/// Succeeds when the tensor holds the only `Arc` reference (refcount == 1)
/// to `FloatStorage` or `ArcArray<f32>`. Returns `Err(tensor)` if the storage
/// is shared, on GPU, or not a recognized float type — giving the caller back
/// ownership so it can fall back to `to_f32_array()`.
fn try_into_f32_array(mut tensor: DynTensor) -> std::result::Result<ArrayD<f32>, DynTensor> {
    if let TensorStorage::Cpu(ref mut arc) = tensor.storage {
        let placeholder: Arc<dyn Any + Send + Sync> = Arc::new(());
        let owned_arc = std::mem::replace(arc, placeholder);
        match owned_arc.downcast::<FloatStorage>() {
            Ok(fs_arc) => match Arc::try_unwrap(fs_arc) {
                Ok(fs) => {
                    let arr = match fs {
                        FloatStorage::F32(a) => a,
                        FloatStorage::F16(a) => a.mapv(half::f16::to_f32),
                        FloatStorage::BF16(a) => a.mapv(half::bf16::to_f32),
                    };
                    return Ok(arr);
                }
                Err(shared) => {
                    *arc = shared;
                    return Err(tensor);
                }
            },
            Err(arc_back) => match arc_back.downcast::<ArcArray<f32, IxDyn>>() {
                Ok(arc_arr) => match Arc::try_unwrap(arc_arr) {
                    Ok(arr) => return Ok(arr.into_owned()),
                    Err(shared) => {
                        *arc = shared;
                        return Err(tensor);
                    }
                },
                Err(unrecognized) => {
                    *arc = unrecognized;
                    return Err(tensor);
                }
            },
        }
    }
    Err(tensor)
}

impl DynTensor {
    // -- Selection Ops --------------------------------------------------------

    /// CPU dispatch for index_select. Separated so `dispatch_cpu_typed!`'s
    /// internal `return` exits this method, not the public `index_select()`.
    fn index_select_dispatch(&self, ids: &Self, dim: usize) -> Result<Self> {
        let dim_size = self.dims[dim];
        let indices: Vec<usize> = match ids.dtype {
            DType::U32 => {
                let idx_arr = ids.as_cpu_u32()?;
                for &idx in idx_arr.iter() {
                    if (idx as usize) >= dim_size {
                        return Err(TensorError::InvalidShape(format!(
                            "index_select: index {idx} out of bounds for dim {dim} (size {dim_size})"
                        )));
                    }
                }
                idx_arr.iter().map(|&i| i as usize).collect()
            }
            DType::I64 => {
                let idx_arr = ids.as_cpu_i64()?;
                for &idx in idx_arr.iter() {
                    if idx < 0 || (idx as usize) >= dim_size {
                        return Err(TensorError::InvalidShape(format!(
                            "index_select: index {idx} out of bounds for dim {dim} (size {dim_size})"
                        )));
                    }
                }
                idx_arr.iter().map(|&i| i as usize).collect()
            }
            other => {
                return Err(TensorError::dtype_mismatch(DType::U32, other));
            }
        };
        dispatch_cpu_typed!(
            self,
            |arr: &ArrayD<_>| -> Result<ArrayD<_>> {
                let selected = arr.select(ndarray::Axis(dim), &indices);
                Ok(selected.as_standard_layout().to_owned())
            },
            "index_select"
        )
    }

    /// Select elements along `dim` using 1-D index tensor (U32).
    ///
    /// Output shape: input shape with `dims[dim]` replaced by `ids.len()`.
    /// Matches candle's `Tensor::index_select`.
    ///
    /// **Note:** Parameter order is `(ids, dim)` — `ids` before `dim` — matching
    /// candle and PyTorch. Most other DynTensor ops take `dim` first.
    ///
    /// # GPU dispatch
    ///
    /// Tries native Metal kernel dispatch via [`GpuBackend::index_select`].
    /// If the backend returns `None` (e.g., non-float dtype), both `self`
    /// and `ids` are transferred to CPU, the operation runs on CPU, and the
    /// result is transferred back to the GPU device.
    pub fn index_select(&self, ids: &Self, dim: impl Dim) -> Result<Self> {
        if ids.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: ids.rank(),
            });
        }
        let dim = dim.to_index(self.rank())?;
        // Try native GPU dispatch; fall back to CPU round-trip.
        let mut result = if self.device().is_gpu() || ids.device().is_gpu() {
            let device = if self.device().is_gpu() {
                self.device()
            } else {
                ids.device()
            };
            let gpu_self = self.to_device(&device)?;
            if let Some(result) = gpu_backend_dispatch(|b| b.index_select(&gpu_self, ids, dim)) {
                result?
            } else {
                let cpu_self = self.to_device(&Device::Cpu)?;
                let cpu_ids = ids.to_device(&Device::Cpu)?;
                cpu_self.index_select(&cpu_ids, dim)?.to_device(&device)?
            }
        } else {
            self.index_select_dispatch(ids, dim)?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, ids])?;
            if let Some(id) = trace::record_op(
                TraceOp::IndexSelect { dim },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Select elements along `dim` without OOB validation (caller guarantees valid indices).
    ///
    /// Same as [`index_select`](Self::index_select) but skips the GPU→CPU readback
    /// used for bounds checking. This eliminates ~8ms flush per call on Metal.
    ///
    /// # Safety contract (not `unsafe`, but caller must guarantee)
    ///
    /// All index values must be `< self.dims()[dim]`. OOB indices are clamped
    /// by the MSL kernel as defense-in-depth, but clamping masks bugs silently.
    ///
    /// # When to use
    ///
    /// Use when indices are computed deterministically (e.g., `arange`, known
    /// vocab lookup) and OOB is structurally impossible.
    pub fn index_select_unchecked(&self, ids: &Self, dim: impl Dim) -> Result<Self> {
        if ids.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: ids.rank(),
            });
        }
        let dim = dim.to_index(self.rank())?;
        let mut result = if self.device().is_gpu() || ids.device().is_gpu() {
            let device = if self.device().is_gpu() {
                self.device()
            } else {
                ids.device()
            };
            let gpu_self = self.to_device(&device)?;
            if let Some(result) =
                gpu_backend_dispatch(|b| b.index_select_unchecked(&gpu_self, ids, dim))
            {
                result?
            } else {
                // No unchecked GPU path available: fall back to checked CPU.
                let cpu_self = self.to_device(&Device::Cpu)?;
                let cpu_ids = ids.to_device(&Device::Cpu)?;
                cpu_self.index_select(&cpu_ids, dim)?.to_device(&device)?
            }
        } else {
            self.index_select_dispatch(ids, dim)?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, ids])?;
            if let Some(id) = trace::record_op(
                TraceOp::IndexSelect { dim },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// CPU dispatch for gather. Separated so `dispatch_cpu_typed!`'s internal
    /// `return` exits this method, not the public `gather()`.
    fn gather_dispatch(&self, ids: &Self, dim: usize) -> Result<Self> {
        let idx_arr = ids.as_cpu_u32()?;
        let dim_size = self.dims[dim];
        let out_shape = ids.dims().to_vec();
        let rank = self.rank();
        dispatch_cpu_typed!(
            self,
            |src_arr: &ArrayD<_>| -> Result<ArrayD<_>> {
                let numel = checked_dim_product(&out_shape)?;
                let mut out_data = Vec::with_capacity(numel);
                let mut coord = vec![0usize; rank];
                let mut src_coord = vec![0usize; rank];
                for flat_idx in 0..numel {
                    let mut rem = flat_idx;
                    for d in (0..rank).rev() {
                        coord[d] = rem % out_shape[d];
                        rem /= out_shape[d];
                    }
                    let gather_idx = idx_arr[IxDyn(&coord)] as usize;
                    if gather_idx >= dim_size {
                        return Err(TensorError::InvalidShape(format!(
                            "gather: index {gather_idx} out of bounds for dim {dim} (size {dim_size})"
                        )));
                    }
                    src_coord.copy_from_slice(&coord);
                    src_coord[dim] = gather_idx;
                    out_data.push(src_arr[IxDyn(&src_coord)]);
                }
                Ok(ArrayD::from_shape_vec(IxDyn(&out_shape), out_data)?)
            },
            "gather"
        )
    }

    /// Gather elements using N-D index tensor (same rank as self, U32 dtype).
    ///
    /// `output[i][j][k] = self[i][ids[i][j][k]][k]` when dim=1.
    /// Output shape = ids shape. Matches candle's `Tensor::gather`.
    ///
    /// **Note:** Parameter order is `(ids, dim)` — `ids` before `dim` — matching
    /// candle and PyTorch. Most other DynTensor ops take `dim` first.
    pub fn gather(&self, ids: &Self, dim: impl Dim) -> Result<Self> {
        if ids.rank() != self.rank() {
            return Err(TensorError::RankMismatch {
                expected: self.rank(),
                actual: ids.rank(),
            });
        }
        let dim = dim.to_index(self.rank())?;
        // Validate non-gather dimensions: ids.dims()[d] <= self.dims()[d] for d != dim.
        for d in 0..self.rank() {
            if d != dim && ids.dims()[d] > self.dims()[d] {
                return Err(TensorError::InvalidShape(format!(
                    "gather: ids size ({}) exceeds self size ({}) on non-gather dim {d}",
                    ids.dims()[d],
                    self.dims()[d],
                )));
            }
        }
        let mut result = if self.device().is_gpu() || ids.device().is_gpu() {
            let device = if self.device().is_gpu() {
                self.device()
            } else {
                ids.device()
            };
            let gpu_self = self.to_device(&device)?;
            if let Some(result) = gpu_backend_dispatch(|b| b.gather(&gpu_self, ids, dim)) {
                result?
            } else {
                let cpu_self = self.to_device(&Device::Cpu)?;
                let cpu_ids = ids.to_device(&Device::Cpu)?;
                cpu_self.gather(&cpu_ids, dim)?.to_device(&device)?
            }
        } else {
            self.gather_dispatch(ids, dim)?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, ids])?;
            if let Some(id) = trace::record_op(
                TraceOp::Gather { dim },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Expand tensor to a larger size using broadcast semantics.
    ///
    /// Dimensions of size 1 can be expanded to any size. Other dimensions
    /// must match exactly. Like PyTorch `Tensor.expand()`.
    ///
    /// # GPU dispatch
    ///
    /// Tries native Metal dispatch via [`GpuBackend::expand`]. If that
    /// returns `None`, float GPU tensors stay on GPU via `broadcast_add`
    /// decomposition (`zeros + self`). Non-float GPU tensors (U32, I64)
    /// are round-tripped through CPU because `broadcast_add` requires
    /// float ops.
    pub fn expand(&self, new_dims: impl AsRef<[usize]>) -> Result<Self> {
        let new_dims = new_dims.as_ref();
        if new_dims.len() != self.rank() {
            return Err(TensorError::RankMismatch {
                expected: self.rank(),
                actual: new_dims.len(),
            });
        }
        for (i, (&old, &new)) in self.dims.iter().zip(new_dims.iter()).enumerate() {
            if old != 1 && old != new {
                return Err(TensorError::InvalidShape(format!(
                    "expand: dim {i} is {old}, cannot expand to {new} (must be 1 or same)"
                )));
            }
        }
        // GPU path: try native expand, fall back to broadcast_add workaround.
        let mut result = if self.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.expand(self, new_dims)) {
                result?
            } else if !matches!(self.dtype, DType::F32 | DType::BF16 | DType::F16) {
                // Non-float GPU tensors (U32, I64) have no GPU expand dispatch.
                // Round-trip through CPU rather than broadcast_add (which requires
                // float ops). (#1709)
                let cpu = self.to_device(&Device::Cpu)?;
                let expanded = cpu.expand(new_dims)?;
                expanded.to_device(&self.device())?
            } else {
                let zeros = Self::full(new_dims, 0.0, self.dtype, &self.device())?;
                zeros.broadcast_add(self)?
            }
        } else {
            let dims_clone = self.dims.clone();
            // Closure wrapper: dispatch_cpu_typed! uses `return` that would
            // bypass trace recording below. Same fix as transpose().
            (|| {
                dispatch_cpu_typed!(
                    self,
                    |arr: &ArrayD<_>| -> Result<ArrayD<_>> {
                        let broadcasted = arr
                            .broadcast(IxDyn(new_dims))
                            .ok_or_else(|| {
                                TensorError::InvalidShape(format!(
                                    "expand: cannot broadcast {dims_clone:?} to {new_dims:?}"
                                ))
                            })?
                            .as_standard_layout()
                            .to_owned();
                        Ok(broadcasted)
                    },
                    "expand"
                )
            })()?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Expand {
                    target_shape: new_dims.to_vec(),
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
}

#[cfg(test)]
#[path = "tests_dtype.rs"]
mod dtype_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_scatter;

#[cfg(kani)]
#[path = "kani_accumulate_proofs.rs"]
mod kani_accumulate_proofs;
#[cfg(kani)]
#[path = "kani_dpdf_gather_scatter_proofs.rs"]
mod kani_dpdf_gather_scatter_proofs;
#[cfg(kani)]
#[path = "kani_dpdf_vlm_indexing_safety.rs"]
mod kani_dpdf_vlm_indexing_safety;
#[cfg(kani)]
#[path = "kani_indexing_dpdf_extended.rs"]
mod kani_indexing_dpdf_extended;
#[cfg(kani)]
#[path = "kani_selection_proofs.rs"]
mod kani_selection_proofs;
#[cfg(kani)]
#[path = "kani_where_cond_proofs.rs"]
mod kani_where_cond_proofs;

#[cfg(test)]
#[path = "tests_index_put.rs"]
mod tests_index_put;
