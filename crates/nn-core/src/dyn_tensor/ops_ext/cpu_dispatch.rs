// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CPU dtype-dispatch helpers for extended DynTensor operations.
//!
//! Contains `cumsum_cpu` and `repeat_interleave_cpu` with multi-dtype support.
//! Extracted from `ops_ext/mod.rs` for 500-line compliance (#1280 Direction 4).

use crate::dyn_tensor::DynTensor;
use crate::{DType, Device, Result};

/// Move a tensor to CPU for computation, returning the original device.
/// If already on CPU, returns a clone with no allocation.
pub(super) fn to_cpu(t: &DynTensor) -> Result<(DynTensor, Device)> {
    let device = t.device();
    let cpu = if device.is_gpu() {
        t.to_device(&Device::Cpu)?
    } else {
        t.clone()
    };
    Ok((cpu, device))
}

/// Move a CPU result back to the original device if needed.
pub(super) fn to_orig(t: DynTensor, device: &Device) -> Result<DynTensor> {
    if device.is_gpu() {
        t.to_device(device)
    } else {
        Ok(t)
    }
}

/// CPU cumsum with dtype dispatch.
pub(super) fn cumsum_cpu(tensor: &DynTensor, dim: usize) -> Result<DynTensor> {
    fn typed<T: Copy + std::ops::AddAssign + 'static>(
        tensor: &DynTensor,
        dim: usize,
        extract: fn(&DynTensor) -> Result<ndarray::ArrayViewD<'_, T>>,
        wrap: fn(ndarray::ArrayD<T>) -> Result<DynTensor>,
    ) -> Result<DynTensor> {
        let arr = extract(tensor)?;
        let axis = ndarray::Axis(dim);
        let mut result = arr.to_owned();
        result.accumulate_axis_inplace(axis, |&prev, curr| *curr += prev);
        wrap(result)
    }

    match tensor.dtype() {
        DType::U32 => typed(tensor, dim, DynTensor::as_cpu_u32, DynTensor::from_cpu_u32),
        DType::U8 => typed(tensor, dim, DynTensor::as_cpu_u8, DynTensor::from_cpu_u8),
        DType::I64 => typed(tensor, dim, DynTensor::as_cpu_i64, DynTensor::from_cpu_i64),
        // Float dtypes: accumulate in f64 matching PyTorch's torch.cumsum
        // CPU behavior (#2691). Sequential f32 accumulation drifts ~9.5e-6
        // over 126 frames; amplified by 2pi*300 this becomes ~18e-3 rad phase
        // error in SineGen, causing STFT 2pi wraps.
        DType::F32 | DType::F16 | DType::BF16 | DType::F64 => {
            let input_dtype = tensor.dtype();
            let arr = tensor.to_f32_array()?;
            let axis = ndarray::Axis(dim);
            let mut f64_arr = arr.mapv(f64::from);
            f64_arr.accumulate_axis_inplace(axis, |&prev, curr| *curr += prev);
            let result = f64_arr.mapv(|x| x as f32);
            DynTensor::from_f32_result(result, input_dtype)
        }
        DType::I32 | DType::Bool => Err(crate::TensorError::Unsupported(format!(
            "cumsum: dtype {} not supported",
            tensor.dtype()
        ))),
    }
}

/// CPU f64-accumulation cumsum for any tensor shape and dimension.
///
/// Used by `cumsum_kahan` CPU fallback: f64 accumulation is higher precision
/// than Kahan f32 compensation, so it's the preferred CPU path.
pub(super) fn cumsum_f64_cpu_generic(x: &DynTensor, dim: usize) -> Result<DynTensor> {
    let shape = x.dims();
    let data = x.to_flat_vec::<f32>()?;
    let mut out = data;

    let axis_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();

    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0f64;
            for a in 0..axis_size {
                let idx = o * (axis_size * inner) + a * inner + i;
                acc += f64::from(out[idx]);
                out[idx] = acc as f32;
            }
        }
    }
    DynTensor::from_vec(out, shape, &Device::Cpu)
}

/// CPU repeat_interleave with dtype dispatch.
pub(super) fn repeat_interleave_cpu(
    tensor: &DynTensor,
    dim: usize,
    counts: &[usize],
) -> Result<DynTensor> {
    fn typed<T: Clone + num_traits::Zero + 'static>(
        tensor: &DynTensor,
        dim: usize,
        counts: &[usize],
        extract: fn(&DynTensor) -> Result<ndarray::ArrayViewD<'_, T>>,
        wrap: fn(ndarray::ArrayD<T>) -> Result<DynTensor>,
    ) -> Result<DynTensor> {
        let arr = extract(tensor)?;
        let axis = ndarray::Axis(dim);
        let total: usize = counts.iter().sum();
        if total == 0 {
            let mut new_dims = tensor.dims().to_vec();
            new_dims[dim] = 0;
            return DynTensor::from_vec(vec![], &new_dims, &tensor.device());
        }
        // Pre-allocate output array and write slices directly (no intermediate arrays).
        let mut out_shape = arr.shape().to_vec();
        out_shape[dim] = total;
        let mut result = ndarray::ArrayD::<T>::zeros(ndarray::IxDyn(&out_shape));
        let mut write_idx = 0;
        for (i, &count) in counts.iter().enumerate() {
            let src = arr.index_axis(axis, i);
            for _ in 0..count {
                let mut dst = result.index_axis_mut(axis, write_idx);
                dst.assign(&src);
                write_idx += 1;
            }
        }
        wrap(result)
    }

    match tensor.dtype() {
        DType::U32 => typed(
            tensor,
            dim,
            counts,
            DynTensor::as_cpu_u32,
            DynTensor::from_cpu_u32,
        ),
        DType::U8 => typed(
            tensor,
            dim,
            counts,
            DynTensor::as_cpu_u8,
            DynTensor::from_cpu_u8,
        ),
        DType::I64 => typed(
            tensor,
            dim,
            counts,
            DynTensor::as_cpu_i64,
            DynTensor::from_cpu_i64,
        ),
        // Float dtypes: promote to f32, compute, demote back to original dtype.
        DType::F32 | DType::F16 | DType::BF16 | DType::F64 => {
            let input_dtype = tensor.dtype();
            let arr = tensor.to_f32_array()?;
            let axis = ndarray::Axis(dim);
            let total: usize = counts.iter().sum();
            if total == 0 {
                let mut new_dims = tensor.dims().to_vec();
                new_dims[dim] = 0;
                return DynTensor::from_vec(vec![], &new_dims, &tensor.device());
            }
            let mut out_shape = arr.shape().to_vec();
            out_shape[dim] = total;
            let mut result = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&out_shape));
            let mut write_idx = 0;
            for (i, &count) in counts.iter().enumerate() {
                let src = arr.index_axis(axis, i);
                for _ in 0..count {
                    let mut dst = result.index_axis_mut(axis, write_idx);
                    dst.assign(&src);
                    write_idx += 1;
                }
            }
            DynTensor::from_f32_result(result, input_dtype)
        }
        DType::I32 | DType::Bool => Err(crate::TensorError::Unsupported(format!(
            "repeat_interleave: dtype {} not supported",
            tensor.dtype()
        ))),
    }
}
