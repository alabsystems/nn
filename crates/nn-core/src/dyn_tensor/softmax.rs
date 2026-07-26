// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Softmax and log-softmax for [`DynTensor`].
//!
//! Both are lenient: they reject NaN but allow `-Inf` for attention masks.

use super::{gpu_backend_dispatch, trace, Dim, DynTensor};
use crate::dyn_tensor::trace::TraceOp;
use crate::{DType, Result, TensorError};
use ndarray::Zip;

impl DynTensor {
    /// Softmax along an arbitrary dimension (lenient — allows `-Inf` for masks).
    ///
    /// Rejects NaN inputs but **allows** `-Inf` values, producing 0.0 for
    /// masked positions. Use this for attention scores with causal/padding masks.
    ///
    /// **Edge case: all-negative-infinity lanes.** When every element in a
    /// lane is `−∞` (e.g., an attention row where all positions are masked),
    /// the max-subtract trick produces `NaN` under IEEE 754. This function
    /// zeros such lanes instead of propagating NaN. Issue: #1310.
    ///
    /// GPU tensors stay on GPU: tries native kernel dispatch, then decomposes
    /// into GPU primitives (max→sub→exp→sum→div). No GPU→CPU transfer.
    ///
    /// Returns an error if input contains NaN (CPU only) or if the specified
    /// dimension is empty.
    pub fn softmax(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        let dim_size = self.dim(dim)?;
        if dim_size == 0 {
            return Err(TensorError::ZeroLengthDimension {
                axis: dim,
                operation: "softmax",
            });
        }
        // traced_forward suppresses decomposed GPU ops (max, sub, exp, sum, div)
        // so only the composite Softmax node appears in the trace graph.
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::Softmax { dim }),
            || self.softmax_dispatch(dim),
        )
    }

    /// Internal dispatch for softmax — separated so trace recording wraps all paths.
    fn softmax_dispatch(&self, dim: usize) -> Result<Self> {
        // Auto-upcast BF16/F16 to F32 for numerical stability (#1813).
        // GPU decomposed softmax uses float accumulators in exp/reduction
        // kernels (D1 of #2981), so skip the upcast+downcast round-trip.
        if matches!(self.dtype(), DType::BF16 | DType::F16) && self.device().is_cpu() {
            let f32_self = self.to_dtype(DType::F32)?;
            let result = f32_self.softmax_dispatch(dim)?;
            return result.to_dtype(self.dtype());
        }
        if self.device().is_gpu() {
            // GPU path: skip NaN pre-check to avoid GPU->CPU copy (#1138).
            // GPU softmax propagates NaN (NaN in -> NaN out). Post-dispatch
            // validation in model forward paths catches NaN if needed (#941).
            if let Some(result) = gpu_backend_dispatch(|b| b.softmax(self, dim)) {
                return result;
            }
            // Fallback: decompose into GPU primitives (#1326 NaN guard).
            return super::ops::softmax::gpu_softmax_decomposed(self, dim);
        }
        // CPU path: promote to f32 for numerically stable computation (#1646).
        // Reject NaN inputs (defense-in-depth, #941 pattern).
        // Inf/-Inf are allowed: attention masks use -inf for masked positions.
        let input_dtype = self.dtype;
        let arr = self.to_f32_array()?;
        let nan_count = arr.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            return Err(TensorError::NonFiniteData {
                name: "softmax input".into(),
                count: nan_count,
            });
        }
        let axis = ndarray::Axis(dim);
        // Numerically stable: subtract max per lane, then exp and normalize
        let max_vals = arr.map_axis(axis, |lane| {
            lane.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        });
        let mut result = arr.to_owned();
        Zip::from(result.lanes_mut(axis))
            .and(&max_vals)
            .for_each(|mut lane, &max_val| {
                // Guard: all-neg-inf lane → zero output (#1310).
                if max_val == f32::NEG_INFINITY {
                    lane.fill(0.0);
                    return;
                }
                // Guard: +inf in lane → uniform over +inf positions, 0 elsewhere.
                // IEEE 754: inf - inf = NaN, so max-subtract trick fails. The
                // mathematically correct limit: exp(+inf) dominates, so +inf
                // positions share probability 1/count, all others get 0.
                if max_val == f32::INFINITY {
                    let inf_count = lane.iter().filter(|&&x| x == f32::INFINITY).count();
                    let prob = 1.0 / inf_count as f32;
                    lane.mapv_inplace(|x| if x == f32::INFINITY { prob } else { 0.0 });
                    return;
                }
                lane.mapv_inplace(|x| (x - max_val).exp());
                let sum: f32 = lane.iter().sum();
                lane.mapv_inplace(|x| x / sum);
            });
        Self::from_f32_result(result, input_dtype)
    }

    /// Log-softmax along a dimension: `log(softmax(x, dim))` (lenient — allows `-Inf`).
    ///
    /// GPU tensors stay on GPU via decomposed primitives (max→sub→exp→sum→log→sub).
    /// No GPU→CPU transfer.
    ///
    /// Like [`softmax`](Self::softmax), rejects NaN (CPU only) but allows `-Inf`
    /// for masks. All-neg-inf lanes produce `-Inf` (log of zero) instead of
    /// NaN. Issue: #1310.
    ///
    /// Returns an error if input contains NaN or if the specified
    /// dimension is empty.
    pub fn log_softmax(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        let dim_size = self.dim(dim)?;
        if dim_size == 0 {
            return Err(TensorError::ZeroLengthDimension {
                axis: dim,
                operation: "log_softmax",
            });
        }
        // traced_forward suppresses decomposed GPU ops (max, sub, exp, sum, log, sub)
        // so only the composite LogSoftmax node appears in the trace graph.
        trace::traced_forward(
            &[self],
            || Ok(TraceOp::LogSoftmax { dim }),
            || self.log_softmax_dispatch(dim),
        )
    }

    /// Internal dispatch for log_softmax — separated so trace recording wraps all paths.
    fn log_softmax_dispatch(&self, dim: usize) -> Result<Self> {
        // Auto-upcast BF16/F16 to F32 for numerical stability (#1813).
        // GPU decomposed log_softmax uses float accumulators in exp/reduction
        // kernels (D1 of #2981), so skip the upcast+downcast round-trip.
        if matches!(self.dtype(), DType::BF16 | DType::F16) && self.device().is_cpu() {
            let f32_self = self.to_dtype(DType::F32)?;
            let result = f32_self.log_softmax_dispatch(dim)?;
            return result.to_dtype(self.dtype());
        }
        if self.device().is_gpu() {
            // GPU path: try native kernel dispatch first (single kernel launch).
            if let Some(result) = gpu_backend_dispatch(|b| b.log_softmax(self, dim)) {
                return result;
            }
            // Fallback: decompose into GPU primitives (#1326 NaN guard).
            return super::ops::softmax::gpu_log_softmax_decomposed(self, dim);
        }
        // CPU path: promote to f32 for numerically stable computation (#1646).
        // Reject NaN inputs (defense-in-depth, #941 pattern).
        let input_dtype = self.dtype;
        let arr = self.to_f32_array()?;
        let nan_count = arr.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            return Err(TensorError::NonFiniteData {
                name: "log_softmax input".into(),
                count: nan_count,
            });
        }
        let axis = ndarray::Axis(dim);
        let max_vals = arr.map_axis(axis, |lane| {
            lane.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        });
        let mut result = arr.to_owned();
        Zip::from(result.lanes_mut(axis))
            .and(&max_vals)
            .for_each(|mut lane, &max_val| {
                // Guard: all-neg-inf lane → -inf output (#1310).
                // log(softmax([-inf, -inf, ...])) = log([0, 0, ...]) = [-inf, ...]
                if max_val == f32::NEG_INFINITY {
                    lane.fill(f32::NEG_INFINITY);
                    return;
                }
                // Guard: +inf in lane → log of softmax result.
                // softmax gives 1/count for +inf positions, 0 for others.
                // log(1/count) = -ln(count) for +inf, log(0) = -inf for others.
                if max_val == f32::INFINITY {
                    let inf_count = lane.iter().filter(|&&x| x == f32::INFINITY).count();
                    let log_prob = -(inf_count as f32).ln();
                    lane.mapv_inplace(|x| {
                        if x == f32::INFINITY {
                            log_prob
                        } else {
                            f32::NEG_INFINITY
                        }
                    });
                    return;
                }
                // log_sum_exp = max + log(sum(exp(x - max)))
                let sum_exp: f32 = lane.iter().map(|&x| (x - max_val).exp()).sum();
                let log_sum_exp = max_val + sum_exp.ln();
                lane.mapv_inplace(|x| x - log_sum_exp);
            });
        Self::from_f32_result(result, input_dtype)
    }
}
