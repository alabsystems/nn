// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Accumulation operations for [`DynTensor`]: `scatter_add` and `index_add`.
//!
//! Extracted from `selection/mod.rs` to keep the parent module under 500 lines.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{gpu_backend_dispatch, Dim, DynTensor};
use crate::{DType, Device, Result, TensorError};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

// -- Validation helpers -------------------------------------------------------

/// Validate scatter_add arguments: dtype, rank, shape, and non-scatter dim bounds.
fn validate_scatter_add_args(
    dst: &DynTensor,
    dim: usize,
    index: &DynTensor,
    src: &DynTensor,
) -> Result<()> {
    if index.dtype != DType::U32 {
        return Err(TensorError::dtype_mismatch(DType::U32, index.dtype));
    }
    if index.rank() != src.rank() {
        return Err(TensorError::RankMismatch {
            expected: src.rank(),
            actual: index.rank(),
        });
    }
    if index.dims() != src.dims() {
        return Err(TensorError::shape_mismatch(
            src.dims().to_vec(),
            index.dims().to_vec(),
        ));
    }
    if index.rank() != dst.rank() {
        return Err(TensorError::RankMismatch {
            expected: dst.rank(),
            actual: index.rank(),
        });
    }
    for d in 0..dst.rank() {
        if d != dim && src.dims()[d] > dst.dims()[d] {
            return Err(TensorError::InvalidShape(format!(
                "scatter_add: src size ({}) exceeds self size ({}) on non-scatter dim {d}",
                src.dims()[d],
                dst.dims()[d],
            )));
        }
    }
    Ok(())
}

/// Validate index_add arguments: dtype, rank, shape, and dim-size match.
fn validate_index_add_args(
    dst: &DynTensor,
    dim: usize,
    index: &DynTensor,
    src: &DynTensor,
) -> Result<()> {
    if index.dtype != DType::U32 {
        return Err(TensorError::dtype_mismatch(DType::U32, index.dtype));
    }
    if index.rank() != 1 {
        return Err(TensorError::RankMismatch {
            expected: 1,
            actual: index.rank(),
        });
    }
    if src.rank() != dst.rank() {
        return Err(TensorError::RankMismatch {
            expected: dst.rank(),
            actual: src.rank(),
        });
    }
    if index.dims()[0] != src.dims()[dim] {
        return Err(TensorError::InvalidShape(format!(
            "index_add: index length {} != src dim {dim} size {}",
            index.dims()[0],
            src.dims()[dim]
        )));
    }
    for (d, (&s, &t)) in src.dims().iter().zip(dst.dims().iter()).enumerate() {
        if d != dim && s != t {
            return Err(TensorError::shape_mismatch(
                dst.dims().to_vec(),
                src.dims().to_vec(),
            ));
        }
    }
    Ok(())
}

// -- CPU accumulation loops ----------------------------------------------------

/// Inner loop for scatter (overwrite): `output[...dim=index[coord]...] = src[coord]`.
fn scatter_loop(
    output: &mut ArrayD<f32>,
    idx_arr: &ArrayViewD<'_, u32>,
    src_arr: &ArrayD<f32>,
    dim: usize,
    dim_size: usize,
    rank: usize,
) -> Result<()> {
    let mut dst_coord = vec![0usize; rank];
    for (coord, &val) in src_arr.indexed_iter() {
        let scatter_idx = idx_arr[&coord] as usize;
        if scatter_idx >= dim_size {
            return Err(TensorError::InvalidShape(format!(
                "scatter: index {scatter_idx} out of bounds for dim {dim} (size {dim_size})"
            )));
        }
        for d in 0..rank {
            dst_coord[d] = coord[d];
        }
        dst_coord[dim] = scatter_idx;
        output[IxDyn(&dst_coord)] = val;
    }
    Ok(())
}

/// Inner loop for scatter_add: `output[...dim=index[coord]...] += src[coord]`.
fn scatter_add_loop(
    output: &mut ArrayD<f32>,
    idx_arr: &ArrayViewD<'_, u32>,
    src_arr: &ArrayD<f32>,
    dim: usize,
    dim_size: usize,
    rank: usize,
) -> Result<()> {
    let mut dst_coord = vec![0usize; rank];
    for (coord, &val) in src_arr.indexed_iter() {
        let scatter_idx = idx_arr[&coord] as usize;
        if scatter_idx >= dim_size {
            return Err(TensorError::InvalidShape(format!(
                "scatter_add: index {scatter_idx} out of bounds for dim {dim} (size {dim_size})"
            )));
        }
        for d in 0..rank {
            dst_coord[d] = coord[d];
        }
        dst_coord[dim] = scatter_idx;
        output[IxDyn(&dst_coord)] += val;
    }
    Ok(())
}

/// Inner loop for index_add: `output[...dim=index[coord[dim]]...] += src[coord]`.
fn index_add_loop(
    output: &mut ArrayD<f32>,
    idx_arr: &ArrayViewD<'_, u32>,
    src_arr: &ArrayD<f32>,
    dim: usize,
    dim_size: usize,
    rank: usize,
) -> Result<()> {
    let mut dst_coord = vec![0usize; rank];
    for (coord, &val) in src_arr.indexed_iter() {
        let dst_idx = idx_arr[IxDyn(&[coord[dim]])] as usize;
        if dst_idx >= dim_size {
            return Err(TensorError::InvalidShape(format!(
                "index_add: index {dst_idx} out of bounds for dim {dim} (size {dim_size})"
            )));
        }
        for d in 0..rank {
            dst_coord[d] = coord[d];
        }
        dst_coord[dim] = dst_idx;
        output[IxDyn(&dst_coord)] += val;
    }
    Ok(())
}

// -- GPU dispatch helpers -----------------------------------------------------

/// Find the GPU device from whichever tensor is on GPU.
fn find_gpu_device(a: &DynTensor, b: &DynTensor, c: &DynTensor) -> Device {
    if a.device().is_gpu() {
        a.device()
    } else if b.device().is_gpu() {
        b.device()
    } else {
        c.device()
    }
}

fn scatter_add_gpu(
    dst: &DynTensor,
    dim: usize,
    index: &DynTensor,
    src: &DynTensor,
) -> Result<DynTensor> {
    let device = find_gpu_device(dst, index, src);
    if let Some(result) = gpu_backend_dispatch(|b| b.scatter_add(dst, dim, index, src)) {
        return result;
    }
    let (cpu_self, cpu_index, cpu_src) = (
        dst.to_device(&Device::Cpu)?,
        index.to_device(&Device::Cpu)?,
        src.to_device(&Device::Cpu)?,
    );
    cpu_self
        .scatter_add(dim, &cpu_index, &cpu_src)?
        .to_device(&device)
}

fn scatter_gpu(
    dst: &DynTensor,
    dim: usize,
    index: &DynTensor,
    src: &DynTensor,
) -> Result<DynTensor> {
    let device = find_gpu_device(dst, index, src);
    if let Some(result) = gpu_backend_dispatch(|b| b.scatter(dst, dim, index, src)) {
        return result;
    }
    let (cpu_self, cpu_index, cpu_src) = (
        dst.to_device(&Device::Cpu)?,
        index.to_device(&Device::Cpu)?,
        src.to_device(&Device::Cpu)?,
    );
    cpu_self
        .scatter(dim, &cpu_index, &cpu_src)?
        .to_device(&device)
}

fn index_add_gpu(
    dst: &DynTensor,
    dim: usize,
    index: &DynTensor,
    src: &DynTensor,
) -> Result<DynTensor> {
    let device = find_gpu_device(dst, index, src);
    if let Some(result) = gpu_backend_dispatch(|b| b.index_add(dst, dim, index, src)) {
        return result;
    }
    let (cpu_self, cpu_index, cpu_src) = (
        dst.to_device(&Device::Cpu)?,
        index.to_device(&Device::Cpu)?,
        src.to_device(&Device::Cpu)?,
    );
    cpu_self
        .index_add(dim, &cpu_index, &cpu_src)?
        .to_device(&device)
}

// -- CPU inner functions (shared by borrowed and owned variants) ---------------

/// Run scatter_add on a pre-extracted CPU output array.
fn scatter_add_cpu(
    mut output: ArrayD<f32>,
    index: &DynTensor,
    src: &DynTensor,
    dim: usize,
    dim_size: usize,
    rank: usize,
    input_dtype: DType,
) -> Result<DynTensor> {
    let idx_arr = index.as_cpu_u32()?;
    let src_arr = src.to_f32_array()?;
    scatter_add_loop(&mut output, &idx_arr, &src_arr, dim, dim_size, rank)?;
    DynTensor::from_f32_result(output, input_dtype)
}

/// Run scatter (overwrite) on a pre-extracted CPU output array.
fn scatter_cpu(
    mut output: ArrayD<f32>,
    index: &DynTensor,
    src: &DynTensor,
    dim: usize,
    dim_size: usize,
    rank: usize,
    input_dtype: DType,
) -> Result<DynTensor> {
    let idx_arr = index.as_cpu_u32()?;
    let src_arr = src.to_f32_array()?;
    scatter_loop(&mut output, &idx_arr, &src_arr, dim, dim_size, rank)?;
    DynTensor::from_f32_result(output, input_dtype)
}

/// Run index_add on a pre-extracted CPU output array.
fn index_add_cpu(
    mut output: ArrayD<f32>,
    index: &DynTensor,
    src: &DynTensor,
    dim: usize,
    dim_size: usize,
    rank: usize,
    input_dtype: DType,
) -> Result<DynTensor> {
    let idx_arr = index.as_cpu_u32()?;
    let src_arr = src.to_f32_array()?;
    index_add_loop(&mut output, &idx_arr, &src_arr, dim, dim_size, rank)?;
    DynTensor::from_f32_result(output, input_dtype)
}

/// Record a trace op (ScatterAdd or IndexAdd) on a result tensor.
fn record_accumulate_trace(result: &mut DynTensor, op: TraceOp, input_ids: &[trace::NodeId]) {
    if let Some(id) = trace::record_op(op, input_ids, result.dims(), result.dtype()) {
        result.set_trace_id(id);
    }
}

// -- DynTensor impl -----------------------------------------------------------

impl DynTensor {
    /// Scatter: write values from `src` into `self` at positions given by
    /// `index`, along `dim` (overwrite, not accumulate).
    ///
    /// For dim=1: `output[i][index[i][j][k]][k] = src[i][j][k]`
    ///
    /// `index` must have the same rank and shape as `src` (U32 dtype).
    /// Output has the same shape as `self`. Matches PyTorch `Tensor.scatter_`.
    pub fn scatter(&self, dim: impl Dim, index: &Self, src: &Self) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        validate_scatter_add_args(self, dim, index, src)?;
        let mut result =
            if self.device().is_gpu() || index.device().is_gpu() || src.device().is_gpu() {
                scatter_gpu(self, dim, index, src)?
            } else {
                let output = self.to_f32_array()?;
                scatter_cpu(
                    output,
                    index,
                    src,
                    dim,
                    self.dims[dim],
                    self.rank(),
                    self.dtype,
                )?
            };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, index, src])?;
            record_accumulate_trace(&mut result, TraceOp::Scatter { dim }, &input_ids);
        }
        Ok(result)
    }

    /// Scatter-add: accumulate values from `src` into `self` at positions
    /// given by `index`, along `dim`.
    ///
    /// For dim=1: `output[i][index[i][j][k]][k] += src[i][j][k]`
    ///
    /// `index` must have the same rank and shape as `src` (U32 dtype).
    /// Output has the same shape as `self`. Matches PyTorch's `Tensor.scatter_add_`.
    pub fn scatter_add(&self, dim: impl Dim, index: &Self, src: &Self) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        validate_scatter_add_args(self, dim, index, src)?;
        let mut result =
            if self.device().is_gpu() || index.device().is_gpu() || src.device().is_gpu() {
                scatter_add_gpu(self, dim, index, src)?
            } else {
                let output = self.to_f32_array()?;
                scatter_add_cpu(
                    output,
                    index,
                    src,
                    dim,
                    self.dims[dim],
                    self.rank(),
                    self.dtype,
                )?
            };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, index, src])?;
            record_accumulate_trace(&mut result, TraceOp::ScatterAdd { dim }, &input_ids);
        }
        Ok(result)
    }

    /// Like [`scatter_add`](Self::scatter_add), but consumes `self` to avoid
    /// cloning the destination array when the `Arc` has refcount 1.
    ///
    /// Use in backward passes where the destination is freshly created:
    /// `DynTensor::zeros(...).scatter_add_into(...)`.
    pub fn scatter_add_into(
        self,
        dim: impl Dim,
        index: &Self,
        src: &Self,
    ) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        validate_scatter_add_args(&self, dim, index, src)?;
        let trace_inputs = if trace::is_tracing() {
            Some(Self::trace_input_ids(&[&self, index, src])?)
        } else {
            None
        };
        let mut result =
            if self.device().is_gpu() || index.device().is_gpu() || src.device().is_gpu() {
                scatter_add_gpu(&self, dim, index, src)?
            } else {
                let (dim_size, rank, dtype) = (self.dims[dim], self.rank(), self.dtype);
                let output = match super::try_into_f32_array(self) {
                    Ok(arr) => arr,
                    Err(this) => this.to_f32_array()?,
                };
                scatter_add_cpu(output, index, src, dim, dim_size, rank, dtype)?
            };
        if let Some(input_ids) = trace_inputs {
            record_accumulate_trace(&mut result, TraceOp::ScatterAdd { dim }, &input_ids);
        }
        Ok(result)
    }

    /// Accumulate values from `src` into `self` at positions along `dim`
    /// given by 1-D `index` (U32 dtype).
    ///
    /// For dim=0: `output[index[i], j, ...] += src[i, j, ...]`
    /// For dim=1: `output[i, index[j], ...] += src[i, j, ...]`
    ///
    /// `index` must be 1-D with length equal to `src.dims()[dim]`.
    /// `src` must have the same rank as `self`, and all dims except `dim`
    /// must match exactly. Matches candle's `Tensor::index_add`.
    pub fn index_add(&self, dim: impl Dim, index: &Self, src: &Self) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        validate_index_add_args(self, dim, index, src)?;
        let mut result =
            if self.device().is_gpu() || src.device().is_gpu() || index.device().is_gpu() {
                index_add_gpu(self, dim, index, src)?
            } else {
                let output = self.to_f32_array()?;
                index_add_cpu(
                    output,
                    index,
                    src,
                    dim,
                    self.dims[dim],
                    self.rank(),
                    self.dtype,
                )?
            };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, index, src])?;
            record_accumulate_trace(&mut result, TraceOp::IndexAdd { dim }, &input_ids);
        }
        Ok(result)
    }

    /// Like [`index_add`](Self::index_add), but consumes `self` to avoid
    /// cloning the destination array when the `Arc` has refcount 1.
    ///
    /// Use in backward passes where the destination is freshly created:
    /// `DynTensor::zeros(...).index_add_into(...)`.
    pub fn index_add_into(self, dim: impl Dim, index: &Self, src: &Self) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        validate_index_add_args(&self, dim, index, src)?;
        let trace_inputs = if trace::is_tracing() {
            Some(Self::trace_input_ids(&[&self, index, src])?)
        } else {
            None
        };
        let mut result =
            if self.device().is_gpu() || src.device().is_gpu() || index.device().is_gpu() {
                index_add_gpu(&self, dim, index, src)?
            } else {
                let (dim_size, rank, dtype) = (self.dims[dim], self.rank(), self.dtype);
                let output = match super::try_into_f32_array(self) {
                    Ok(arr) => arr,
                    Err(this) => this.to_f32_array()?,
                };
                index_add_cpu(output, index, src, dim, dim_size, rank, dtype)?
            };
        if let Some(input_ids) = trace_inputs {
            record_accumulate_trace(&mut result, TraceOp::IndexAdd { dim }, &input_ids);
        }
        Ok(result)
    }
}
