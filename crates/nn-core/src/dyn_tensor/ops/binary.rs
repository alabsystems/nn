// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CPU binary arithmetic implementations for [`DynTensor`].
//!
//! Extracted from `ops/mod.rs` (#1665) to keep the parent under 300 lines.
//! Contains same-shape and broadcast binary ops, plus helpers for device
//! checking, division finiteness, and broadcast shape computation.

use super::super::{gpu_backend, trace, BinaryOp, DynTensor, TensorStorage};
use crate::dyn_tensor::trace::TraceOp;
use crate::{Result, TensorError};
use ndarray::ArrayD;

/// Convert a BinaryOp to its corresponding TraceOp.
fn binary_op_to_trace_op(op: BinaryOp) -> TraceOp {
    match op {
        BinaryOp::Add => TraceOp::Add,
        BinaryOp::Sub => TraceOp::Sub,
        BinaryOp::Mul => TraceOp::Mul,
        BinaryOp::Div => TraceOp::Div,
        BinaryOp::Maximum => TraceOp::Maximum,
        BinaryOp::Minimum => TraceOp::Minimum,
        BinaryOp::Atan2 => TraceOp::Atan2,
    }
}

// -- Helper: ensure both tensors on same device -------------------------------

pub(crate) fn check_same_device(a: &DynTensor, b: &DynTensor) -> Result<()> {
    if a.device() != b.device() {
        Err(TensorError::Unsupported(format!(
            "mixed-device op: {} vs {}",
            a.device(),
            b.device()
        )))
    } else {
        Ok(())
    }
}

/// Check for non-finite values in a CPU division result.
///
/// Division by zero produces Inf/NaN which silently corrupts downstream
/// computation. This guard returns a descriptive error instead of propagating
/// non-finite values. Applied only to division output, not other ops where
/// Inf can be a legitimate intermediate (e.g., log(0) = -Inf in softmax).
fn check_div_result_finite(result: &ArrayD<f32>) -> Result<()> {
    let non_finite = result.iter().filter(|v| !v.is_finite()).count();
    if non_finite > 0 {
        return Err(TensorError::Unsupported(format!(
            "division produced {non_finite} non-finite value(s) (Inf/NaN from zero or near-zero divisor)"
        )));
    }
    Ok(())
}

// GPU division finiteness check REMOVED (#1147):
// Reading the entire GPU result back to CPU on every division was a severe
// performance bottleneck. Model-level NaN guards (#941, #958) catch non-finite
// values at stage boundaries. The CPU check (check_div_result_finite) is free
// and remains. GPU division follows IEEE 754 semantics (0/0 = NaN, x/0 = Inf).

/// Compute NumPy-style broadcast output shape for two dynamic shapes.
///
/// Right-aligns dimensions and expands size-1 dims. Returns an error if
/// shapes are not broadcast-compatible (mismatched non-1 dimensions).
pub(crate) fn broadcast_output_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let max_ndim = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        let l = if i < max_ndim - lhs.len() {
            1
        } else {
            lhs[i - (max_ndim - lhs.len())]
        };
        let r = if i < max_ndim - rhs.len() {
            1
        } else {
            rhs[i - (max_ndim - rhs.len())]
        };
        if l == r {
            out.push(l);
        } else if l == 1 {
            out.push(r);
        } else if r == 1 {
            out.push(l);
        } else {
            return Err(TensorError::shape_mismatch(lhs.to_vec(), rhs.to_vec()));
        }
    }
    Ok(out)
}

// -- CPU binary implementations -----------------------------------------------

impl DynTensor {
    pub(crate) fn binary_op_impl(&self, op: BinaryOp, rhs: &Self) -> Result<Self> {
        // Auto-dequantize quantized operands before dispatch.
        if self.is_quantized() || rhs.is_quantized() {
            let lhs_deq = self.auto_dequantize()?;
            let rhs_deq = rhs.auto_dequantize()?;
            return lhs_deq.binary_op_impl(op, &rhs_deq);
        }
        check_same_device(self, rhs)?;
        let mut result = match (&self.storage, &rhs.storage) {
            (TensorStorage::Cpu(_), TensorStorage::Cpu(_)) => self.cpu_binary_same_shape(op, rhs),
            (TensorStorage::Gpu { .. }, TensorStorage::Gpu { .. }) => {
                let backend = gpu_backend()?;
                backend.binary_op(op, self, rhs)
            }
            _ => Err(TensorError::Unsupported("mixed CPU/GPU storage".into())),
        }?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, rhs])?;
            let trace_op = binary_op_to_trace_op(op);
            if let Some(id) = trace::record_op(trace_op, &input_ids, result.dims(), result.dtype())
            {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    pub(crate) fn broadcast_binary_op(&self, op: BinaryOp, rhs: &Self) -> Result<Self> {
        // Auto-dequantize quantized operands before dispatch.
        if self.is_quantized() || rhs.is_quantized() {
            let lhs_deq = self.auto_dequantize()?;
            let rhs_deq = rhs.auto_dequantize()?;
            return lhs_deq.broadcast_binary_op(op, &rhs_deq);
        }
        check_same_device(self, rhs)?;
        let mut result = match (&self.storage, &rhs.storage) {
            (TensorStorage::Cpu(_), TensorStorage::Cpu(_)) => self.cpu_broadcast_binary(op, rhs),
            (TensorStorage::Gpu { .. }, TensorStorage::Gpu { .. }) => {
                let backend = gpu_backend()?;
                backend.binary_op(op, self, rhs)
            }
            _ => Err(TensorError::Unsupported("mixed CPU/GPU storage".into())),
        }?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, rhs])?;
            let trace_op = binary_op_to_trace_op(op);
            if let Some(id) = trace::record_op(trace_op, &input_ids, result.dims(), result.dtype())
            {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    fn cpu_binary_same_shape(&self, op: BinaryOp, rhs: &Self) -> Result<Self> {
        // Promote bf16/f16 to f32, compute, convert back (#1646 D3).
        let lhs_arr = self.to_f32_array()?;
        let rhs_arr = rhs.to_f32_array()?;
        if lhs_arr.shape() != rhs_arr.shape() {
            return Err(TensorError::shape_mismatch(
                self.dims().to_vec(),
                rhs.dims().to_vec(),
            ));
        }
        let result = match op {
            BinaryOp::Add => &lhs_arr + &rhs_arr,
            BinaryOp::Sub => &lhs_arr - &rhs_arr,
            BinaryOp::Mul => &lhs_arr * &rhs_arr,
            BinaryOp::Div => {
                let r = &lhs_arr / &rhs_arr;
                check_div_result_finite(&r)?;
                r
            }
            BinaryOp::Maximum => ndarray::Zip::from(&lhs_arr)
                .and(&rhs_arr)
                .map_collect(|&a, &b| a.max(b)),
            BinaryOp::Minimum => ndarray::Zip::from(&lhs_arr)
                .and(&rhs_arr)
                .map_collect(|&a, &b| a.min(b)),
            BinaryOp::Atan2 => ndarray::Zip::from(&lhs_arr)
                .and(&rhs_arr)
                .map_collect(|&a, &b| a.atan2(b)),
        };
        // Result dtype follows lhs (matching PyTorch convention).
        Self::from_f32_result(result, self.dtype)
    }

    fn cpu_broadcast_binary(&self, op: BinaryOp, rhs: &Self) -> Result<Self> {
        // Promote bf16/f16 to f32, compute, convert back (#1646 D3).
        let lhs_arr = self.to_f32_array()?;
        let rhs_arr = rhs.to_f32_array()?;
        // Validate broadcast compatibility upfront. ndarray's arithmetic ops
        // panic on incompatible shapes instead of returning Result.
        // Compute once and reuse for Maximum/Minimum/Atan2 which need explicit broadcast.
        let out_shape = broadcast_output_shape(lhs_arr.shape(), rhs_arr.shape())?;
        // ndarray handles NumPy-style right-aligned broadcasting natively
        let result = match op {
            BinaryOp::Add => (&lhs_arr + &rhs_arr).to_owned(),
            BinaryOp::Sub => (&lhs_arr - &rhs_arr).to_owned(),
            BinaryOp::Mul => (&lhs_arr * &rhs_arr).to_owned(),
            BinaryOp::Div => {
                let r = (&lhs_arr / &rhs_arr).to_owned();
                check_div_result_finite(&r)?;
                r
            }
            BinaryOp::Maximum => {
                let lhs_b = lhs_arr
                    .broadcast(ndarray::IxDyn(&out_shape))
                    .ok_or_else(|| {
                        TensorError::shape_mismatch(self.dims().to_vec(), rhs.dims().to_vec())
                    })?;
                let rhs_b = rhs_arr
                    .broadcast(ndarray::IxDyn(&out_shape))
                    .ok_or_else(|| {
                        TensorError::shape_mismatch(self.dims().to_vec(), rhs.dims().to_vec())
                    })?;
                ndarray::Zip::from(&lhs_b)
                    .and(&rhs_b)
                    .map_collect(|&a, &b| a.max(b))
            }
            BinaryOp::Minimum => {
                let lhs_b = lhs_arr
                    .broadcast(ndarray::IxDyn(&out_shape))
                    .ok_or_else(|| {
                        TensorError::shape_mismatch(self.dims().to_vec(), rhs.dims().to_vec())
                    })?;
                let rhs_b = rhs_arr
                    .broadcast(ndarray::IxDyn(&out_shape))
                    .ok_or_else(|| {
                        TensorError::shape_mismatch(self.dims().to_vec(), rhs.dims().to_vec())
                    })?;
                ndarray::Zip::from(&lhs_b)
                    .and(&rhs_b)
                    .map_collect(|&a, &b| a.min(b))
            }
            BinaryOp::Atan2 => {
                let lhs_b = lhs_arr
                    .broadcast(ndarray::IxDyn(&out_shape))
                    .ok_or_else(|| {
                        TensorError::shape_mismatch(self.dims().to_vec(), rhs.dims().to_vec())
                    })?;
                let rhs_b = rhs_arr
                    .broadcast(ndarray::IxDyn(&out_shape))
                    .ok_or_else(|| {
                        TensorError::shape_mismatch(self.dims().to_vec(), rhs.dims().to_vec())
                    })?;
                ndarray::Zip::from(&lhs_b)
                    .and(&rhs_b)
                    .map_collect(|&a, &b| a.atan2(b))
            }
        };
        // Result dtype follows lhs (matching PyTorch convention).
        Self::from_f32_result(result, self.dtype)
    }
}
