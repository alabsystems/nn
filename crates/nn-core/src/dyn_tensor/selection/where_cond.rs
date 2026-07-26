// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conditional selection operation for [`DynTensor`].
//!
//! Extracted from `selection/mod.rs` to keep the parent module under 500 lines.

use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend_dispatch, trace, DynTensor};
use crate::{DType, Device, Result, TensorError};
use ndarray::ArrayD;

impl DynTensor {
    // -- Conditional Select ---------------------------------------------------

    /// Element-wise conditional: `if self[i] != 0 { on_true[i] } else { on_false[i] }`.
    ///
    /// Mask must be U8 (standard) or F32 with 0.0/1.0 values (GPU fast path,
    /// returned by `gpu_compare` to avoid round-trips — see #1323).
    /// All three tensors must have same shape. Matches candle's `Tensor::where_cond`.
    ///
    /// # GPU dispatch
    ///
    /// Native Metal dispatch via [`GpuBackend::where_cond`] for float tensors.
    /// If the backend returns `None`, all three tensors (mask, on_true,
    /// on_false) are transferred to CPU and the result transferred back.
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        // Accept U8 (standard boolean mask) or F32 (GPU comparison result, 0.0/1.0).
        if self.dtype != DType::U8 && self.dtype != DType::F32 {
            return Err(TensorError::Unsupported(format!(
                "where_cond: mask must be U8 or F32, got {:?}",
                self.dtype
            )));
        }
        // Broadcast all three tensors to a common shape if needed.
        if self.dims != on_true.dims || self.dims != on_false.dims {
            let out_shape = crate::dyn_tensor::ops::broadcast_output_shape(
                &crate::dyn_tensor::ops::broadcast_output_shape(self.dims(), on_true.dims())?,
                on_false.dims(),
            )?;
            let mask_b = self.expand(&out_shape)?;
            let true_b = on_true.expand(&out_shape)?;
            let false_b = on_false.expand(&out_shape)?;
            return mask_b.where_cond(&true_b, &false_b);
        }
        let mut result = if self.device().is_gpu()
            || on_true.device().is_gpu()
            || on_false.device().is_gpu()
        {
            let device = if self.device().is_gpu() {
                self.device()
            } else if on_true.device().is_gpu() {
                on_true.device()
            } else {
                on_false.device()
            };
            // Try native GPU dispatch first.
            if let Some(result) = gpu_backend_dispatch(|b| b.where_cond(self, on_true, on_false)) {
                result?
            } else {
                // Fallback: CPU round-trip.
                let cpu_mask = self.to_device(&Device::Cpu)?;
                let cpu_true = on_true.to_device(&Device::Cpu)?;
                let cpu_false = on_false.to_device(&Device::Cpu)?;
                let r = cpu_mask.where_cond(&cpu_true, &cpu_false)?;
                r.to_device(&device)?
            }
        } else {
            if on_true.dtype != on_false.dtype {
                return Err(TensorError::dtype_mismatch(on_true.dtype, on_false.dtype));
            }
            // F32 mask (0.0/1.0): convert to U8 for CPU dispatch.
            let mask = if self.dtype == DType::F32
                || self.dtype == DType::BF16
                || self.dtype == DType::F16
            {
                let f32_arr = self.to_f32_array()?;
                f32_arr.mapv(|v| u8::from(v != 0.0))
            } else {
                self.as_cpu_u8()?.to_owned()
            };
            where_cond_cpu(&mask.view(), on_true, on_false)?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, on_true, on_false])?;
            if let Some(id) = trace::record_op(
                TraceOp::WhereCond,
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

/// CPU `where_cond` with dtype dispatch.
fn where_cond_cpu(
    mask: &ndarray::ArrayViewD<'_, u8>,
    on_true: &DynTensor,
    on_false: &DynTensor,
) -> Result<DynTensor> {
    fn typed<T: Clone + 'static>(
        mask: &ndarray::ArrayViewD<'_, u8>,
        on_true: &DynTensor,
        on_false: &DynTensor,
        extract: fn(&DynTensor) -> Result<ndarray::ArrayViewD<'_, T>>,
        wrap: fn(ArrayD<T>) -> Result<DynTensor>,
    ) -> Result<DynTensor> {
        let t_arr = extract(on_true)?;
        let f_arr = extract(on_false)?;
        let result = ndarray::Zip::from(mask)
            .and(&t_arr)
            .and(&f_arr)
            .map_collect(|&m, t, f| if m != 0 { t.clone() } else { f.clone() });
        wrap(result)
    }

    match on_true.dtype {
        DType::U32 => typed(
            mask,
            on_true,
            on_false,
            DynTensor::as_cpu_u32,
            DynTensor::from_cpu_u32,
        ),
        DType::U8 => typed(
            mask,
            on_true,
            on_false,
            DynTensor::as_cpu_u8,
            DynTensor::from_cpu_u8,
        ),
        DType::I64 => typed(
            mask,
            on_true,
            on_false,
            DynTensor::as_cpu_i64,
            DynTensor::from_cpu_i64,
        ),
        // Float dtypes: promote to f32 via to_f32_array(), compute, preserve dtype.
        DType::F32 | DType::F16 | DType::BF16 | DType::F64 => {
            let input_dtype = on_true.dtype;
            let t_arr = on_true.to_f32_array()?;
            let f_arr = on_false.to_f32_array()?;
            let result = ndarray::Zip::from(mask)
                .and(&t_arr)
                .and(&f_arr)
                .map_collect(|&m, &t, &f| if m != 0 { t } else { f });
            DynTensor::from_f32_result(result, input_dtype)
        }
        DType::I32 | DType::Bool => Err(TensorError::Unsupported(format!(
            "where_cond: dtype {} not supported",
            on_true.dtype
        ))),
    }
}
