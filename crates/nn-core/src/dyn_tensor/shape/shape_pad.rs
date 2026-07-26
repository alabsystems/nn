// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Constant padding for [`DynTensor`].
//!
//! Implements `pad()` matching PyTorch's `F.pad()` convention:
//! padding is specified as `[left_last, right_last, left_2nd_last, ...]`.

use crate::dyn_tensor::gpu_backend_dispatch;
use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};
use ndarray::{ArrayD, IxDyn, SliceInfoElem};

impl DynTensor {
    /// Pad tensor with a constant value.
    ///
    /// `padding` follows PyTorch's `F.pad()` convention: pairs of
    /// `[left_last, right_last, left_2nd_last, right_2nd_last, ...]`
    /// specifying padding for dimensions from the last to the first.
    ///
    /// Only the last `padding.len() / 2` dimensions are padded; earlier
    /// dimensions are unchanged.
    ///
    /// # Errors
    ///
    /// - `padding.len()` must be even.
    /// - `padding.len() / 2` must not exceed tensor rank.
    pub fn pad(&self, padding: &[usize], value: f64) -> Result<Self> {
        if !padding.len().is_multiple_of(2) {
            return Err(TensorError::InvalidShape(
                "pad: padding length must be even".into(),
            ));
        }
        let n_pad_dims = padding.len() / 2;
        if n_pad_dims > self.rank() {
            return Err(TensorError::InvalidShape(format!(
                "pad: {} padding pairs exceeds rank {}",
                n_pad_dims,
                self.rank()
            )));
        }
        // No-op if all zeros
        if padding.iter().all(|&p| p == 0) {
            return Ok(self.clone());
        }

        // Compute output shape and per-dim (left, right) padding
        let rank = self.rank();
        let mut pad_left = vec![0usize; rank];
        let mut pad_right = vec![0usize; rank];
        for i in 0..n_pad_dims {
            let dim = rank - 1 - i;
            pad_left[dim] = padding[2 * i];
            pad_right[dim] = padding[2 * i + 1];
        }
        let out_dims: Vec<usize> = self
            .dims()
            .iter()
            .enumerate()
            .map(|(d, &s)| s + pad_left[d] + pad_right[d])
            .collect();

        // Try GPU-native pad if tensor is on GPU.
        if self.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.pad(self, padding, value)) {
                let mut result = result?;
                if trace::is_tracing() {
                    let input_ids = Self::trace_input_ids(&[self])?;
                    if let Some(id) = trace::record_op(
                        TraceOp::ConstantPadNd {
                            padding: padding.to_vec(),
                            value,
                        },
                        &input_ids,
                        result.dims(),
                        result.dtype(),
                    ) {
                        result.set_trace_id(id);
                    }
                }
                return Ok(result);
            }
        }

        // CPU implementation (GPU falls back to CPU round-trip)
        let (cpu_self, on_gpu) = if self.device().is_gpu() {
            (self.to_device(&Device::Cpu)?, true)
        } else {
            (self.clone(), false)
        };

        let src_arr = cpu_self.to_f32_array()?;
        let val = value as f32;

        // Create output filled with pad value
        let mut out = ArrayD::from_elem(IxDyn(&out_dims), val);

        // Build slice info for the region where source data goes
        let slice_info: Vec<SliceInfoElem> = (0..rank)
            .map(|d| SliceInfoElem::Slice {
                start: pad_left[d] as isize,
                end: Some((pad_left[d] + self.dims()[d]) as isize),
                step: 1,
            })
            .collect();

        // Copy source into the padded output
        let mut dst_view = out.slice_mut(slice_info.as_slice());
        dst_view.assign(&src_arr);

        let mut result = Self::from_f32_result(out, self.dtype)?;

        if on_gpu {
            result = result.to_device(&self.device())?;
        }

        // Record trace op
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::ConstantPadNd {
                    padding: padding.to_vec(),
                    value,
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
