// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended operations for [`DynTensor`] — non-keepdim reductions,
//! argmax/argmin, powf, clamp variants, and flatten.
//!
//! These fill API gaps needed for candle → nn model migration. Each op
//! matches candle's `Tensor` API naming convention.
//! CPU dtype-dispatch helpers live in `cpu_dispatch.rs`.

use super::{gpu_backend_dispatch, trace, Dim, DynTensor};
use crate::dyn_tensor::trace::TraceOp;
use crate::{Device, Result, TensorError};

mod cpu_dispatch;
use cpu_dispatch::{cumsum_cpu, cumsum_f64_cpu_generic, repeat_interleave_cpu, to_cpu, to_orig};

// -- Non-keepdim reductions ---------------------------------------------------

impl DynTensor {
    /// Sum along a dimension, removing the reduced dimension from output shape.
    ///
    /// Matches candle's `Tensor::sum(dim)`. For `[B, C, T].sum(1)` → `[B, T]`.
    pub fn sum(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.sum_keepdim(dim)?.squeeze(dim)
    }

    /// Mean along a dimension, removing the reduced dimension from output shape.
    pub fn mean(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.mean_keepdim(dim)?.squeeze(dim)
    }

    /// Max along a dimension, removing the reduced dimension from output shape.
    ///
    /// Returns only the max values, not indices. Use `argmax` for indices.
    pub fn max(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.max_keepdim(dim)?.squeeze(dim)
    }

    /// Min along a dimension, removing the reduced dimension from output shape.
    pub fn min(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.min_keepdim(dim)?.squeeze(dim)
    }

    /// Variance along a dimension, removing the reduced dimension from output shape.
    ///
    /// Computes population variance: `mean((x - mean(x))^2)`.
    /// Matches the pattern of `sum`/`mean`/`max`/`min` non-keepdim variants.
    pub fn var(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.var_keepdim(dim)?.squeeze(dim)
    }
}

// -- Argmax / Argmin ----------------------------------------------------------
mod argreduce;

// -- Clamp variants -----------------------------------------------------------

impl DynTensor {
    /// Clamp values to be at least `min_val` (no upper bound).
    ///
    /// GPU tensors stay on GPU via relu decomposition:
    /// `clamp_min(x, lo) = relu(x - lo) + lo`.
    /// Uses `traced_forward` so decomposed GPU ops don't create redundant
    /// trace nodes — only the composite Clamp op is recorded.
    pub fn clamp_min(&self, min_val: f64) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || {
                Ok(TraceOp::Clamp {
                    min: Some(min_val),
                    max: None,
                })
            },
            || {
                if self.device().is_gpu() {
                    // Fused GPU path: single dispatch (#1815 D2a)
                    if let Some(result) = gpu_backend_dispatch(|b| b.clamp_min(self, min_val)) {
                        return result;
                    }
                    // Fallback: relu decomposition (3 encodings)
                    self.sub_scalar(min_val)?.relu()?.add_scalar(min_val)
                } else {
                    let input_dtype = self.dtype;
                    let arr = self.to_f32_array()?;
                    let lo = crate::dyn_tensor::checked_f64_to_f32(min_val, "clamp_min()")?;
                    let result = arr.mapv(|x| x.max(lo));
                    Self::from_f32_result(result, input_dtype)
                }
            },
        )
    }

    /// Clamp values to be at most `max_val` (no lower bound).
    ///
    /// GPU tensors stay on GPU via relu decomposition:
    /// `clamp_max(x, hi) = hi - relu(hi - x)`.
    /// Uses `traced_forward` so decomposed GPU ops don't create redundant
    /// trace nodes — only the composite Clamp op is recorded.
    pub fn clamp_max(&self, max_val: f64) -> Result<Self> {
        trace::traced_forward(
            &[self],
            || {
                Ok(TraceOp::Clamp {
                    min: None,
                    max: Some(max_val),
                })
            },
            || {
                if self.device().is_gpu() {
                    // Fused GPU path: single dispatch (#1815 D2a)
                    if let Some(result) = gpu_backend_dispatch(|b| b.clamp_max(self, max_val)) {
                        return result;
                    }
                    // Fallback: relu decomposition (5 encodings)
                    let diff = self.neg()?.add_scalar(max_val)?;
                    diff.relu()?.neg()?.add_scalar(max_val)
                } else {
                    let input_dtype = self.dtype;
                    let arr = self.to_f32_array()?;
                    let hi = crate::dyn_tensor::checked_f64_to_f32(max_val, "clamp_max()")?;
                    let result = arr.mapv(|x| x.min(hi));
                    Self::from_f32_result(result, input_dtype)
                }
            },
        )
    }
}

// -- Partial flatten ----------------------------------------------------------

impl DynTensor {
    /// Flatten dimensions from `start_dim` to `end_dim` (inclusive) into one.
    ///
    /// Matches candle's `Tensor::flatten(start_dim, end_dim)`.
    /// Negative indexing is not supported; use `rank() - 1` for last dim.
    pub fn flatten(&self, start_dim: impl Dim, end_dim: impl Dim) -> Result<Self> {
        let rank = self.rank();
        let start_dim = start_dim.to_index(rank)?;
        let end_dim = end_dim.to_index(rank)?;
        if start_dim > end_dim {
            return Err(TensorError::InvalidShape(format!(
                "flatten({start_dim}, {end_dim}) invalid for rank {rank}"
            )));
        }
        if start_dim == end_dim {
            return Ok(self.clone());
        }
        let dims = self.dims();
        let flat_size = crate::tensor::checked_dim_product(&dims[start_dim..=end_dim])?;
        let mut new_dims = Vec::with_capacity(rank - (end_dim - start_dim));
        new_dims.extend_from_slice(&dims[..start_dim]);
        new_dims.push(flat_size);
        new_dims.extend_from_slice(&dims[end_dim + 1..]);
        self.reshape(&new_dims)
    }
}

// -- Cumulative sum -----------------------------------------------------------

impl DynTensor {
    /// Cumulative sum along a dimension.
    ///
    /// For input `[a, b, c]` with `dim=0`: returns `[a, a+b, a+b+c]`.
    /// Used in Kokoro TTS for duration boundary computation in `length_regulate`.
    ///
    /// # GPU dispatch
    ///
    /// Tries native Metal kernel dispatch via [`GpuBackend::cumsum`]. Metal
    /// backend uses a Blelloch parallel prefix sum kernel (single-pass for
    /// axis ≤ 256, three-pass for axis ≤ 65536). Axis sizes > 65536 fall
    /// back to CPU round-trip.
    pub fn cumsum(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        // Try GPU-native dispatch first.
        let mut result = if self.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.cumsum(self, dim)) {
                result?
            } else {
                let (cpu_self, device) = to_cpu(self)?;
                let r = cumsum_cpu(&cpu_self, dim)?;
                to_orig(r, &device)?
            }
        } else {
            let (cpu_self, device) = to_cpu(self)?;
            let r = cumsum_cpu(&cpu_self, dim)?;
            to_orig(r, &device)?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Cumsum { dim },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Kahan-compensated cumulative sum along `dim` (#2909).
    ///
    /// Error bound: O(nε) vs O(n²ε) for naive f32 accumulation. Sufficient for
    /// SineGen phase precision (worst-case ~0.014 rad vs ~2.3 rad naive).
    /// GPU dispatch uses sequential scan with Kahan compensation (one thread per
    /// slice). CPU fallback uses f64 accumulation.
    pub fn cumsum_kahan(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        if self.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.cumsum_kahan(self, dim)) {
                return result;
            }
        }
        // CPU fallback: use f64 accumulation (higher precision than Kahan f32).
        let (cpu_self, device) = to_cpu(self)?;
        let r = cumsum_f64_cpu_generic(&cpu_self, dim)?;
        to_orig(r, &device)
    }
}

// -- Repeat interleave --------------------------------------------------------

impl DynTensor {
    /// Repeat each element along `dim` by the corresponding count in `repeats`.
    ///
    /// `repeats` must be a 1-D tensor with length equal to the size of `dim`.
    /// Each element `i` along `dim` is repeated `repeats[i]` times.
    ///
    /// Used in Kokoro TTS for `length_regulate` to expand features by predicted durations.
    ///
    /// Example: `[a, b, c].repeat_interleave(0, [2, 1, 3])` → `[a, a, b, c, c, c]`.
    ///
    /// # GPU dispatch
    ///
    /// When both `self` and `repeats` are on GPU, tries
    /// [`GpuSelectionOps::repeat_interleave_from_gpu`] first — this computes the
    /// prefix-sum offsets on GPU with only one scalar readback for the output
    /// buffer size, eliminating the full CPU sync. Falls back to
    /// [`GpuSelectionOps::repeat_interleave`] (CPU-side counts) if the
    /// GPU-native path returns `None`.
    pub fn repeat_interleave(&self, dim: impl Dim, repeats: &Self) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;

        // Fast path: GPU-native dispatch keeps counts on GPU — avoids full
        // CPU sync for the counts tensor. Only one scalar readback (total).
        // Requires both tensors on the same GPU device. (#2616)
        if self.device().is_gpu() && repeats.device().is_gpu() {
            if let Some(result) =
                gpu_backend_dispatch(|b| b.repeat_interleave_from_gpu(self, dim, repeats))
            {
                return record_repeat_interleave_trace(result?, self, repeats, dim);
            }
        }

        let counts = repeat_interleave_validate_counts(repeats, self.dims()[dim])?;
        let total: usize = counts.iter().sum();
        if total == 0 {
            let mut new_dims = self.dims().to_vec();
            new_dims[dim] = 0;
            return Self::from_vec(vec![], &new_dims, &self.device());
        }
        // Try GPU dispatch with CPU-side counts.
        let result = if self.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.repeat_interleave(self, dim, &counts))
            {
                result?
            } else {
                let (cpu_self, device) = to_cpu(self)?;
                to_orig(repeat_interleave_cpu(&cpu_self, dim, &counts)?, &device)?
            }
        } else {
            let (cpu_self, device) = to_cpu(self)?;
            to_orig(repeat_interleave_cpu(&cpu_self, dim, &counts)?, &device)?
        };
        record_repeat_interleave_trace(result, self, repeats, dim)
    }

    // -- Element-wise power ---------------------------------------------------

    /// Element-wise power: `x^exponent` for each element.
    ///
    /// Used by Kokoro TTS F0 source generation (`f0.powf(2.0)`).
    ///
    /// GPU path uses `exp(exponent * log(abs(x)))` for magnitude with sign
    /// correction for integer exponents. Non-integer exponents with negative
    /// inputs produce NaN, matching IEEE 754 `f32::powf` semantics.
    pub fn powf(&self, exponent: f64) -> Result<Self> {
        if self.device().is_gpu() {
            // GPU decomposes to abs/log/exp — those ops are already traced.
            return self.gpu_powf(exponent);
        }
        let input_dtype = self.dtype;
        let arr = self.to_f32_array()?;
        let e = crate::dyn_tensor::checked_f64_to_f32(exponent, "powf() exponent")?;
        let result = arr.mapv(|x| x.powf(e));
        let mut result = Self::from_f32_result(result, input_dtype)?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Powf { exponent },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// GPU-native powf using abs-based decomposition.
    ///
    /// `|x|^e` via `exp(e * log(|x|))`, then:
    /// - Even integer e: result is always positive (no sign correction)
    /// - Odd integer e: negate where x < 0
    /// - Non-integer e: NaN where x < 0 (IEEE 754 semantics)
    fn gpu_powf(&self, exponent: f64) -> Result<Self> {
        let e = crate::dyn_tensor::checked_f64_to_f32(exponent, "powf() exponent")?;
        // Special case: x^0 = 1 for all x (including 0^0 = 1), matching f32::powf.
        if e == 0.0 {
            return Self::full(self.dims(), 1.0, self.dtype(), &self.device());
        }
        // Magnitude: exp(e * log(|x|)) — handles positive x, zero, and |negative x|.
        let abs_pow = self.abs()?.log()?.mul_scalar(exponent)?.exp()?;
        let is_integer = e == e.floor() && e.is_finite();
        if is_integer {
            // f32 cannot represent individual integers beyond 2^24 (16,777,216).
            // For |e| > 2^24, consecutive integers are indistinguishable, so
            // the even/odd classification is meaningless. Treat as even (safe:
            // always-positive result). Also guards against `e as i64` saturation
            // for |e| > i64::MAX (~9.2e18).
            let can_determine_parity = e.abs() <= (1i64 << 24) as f32;
            let is_even = !can_determine_parity || (e as i64) % 2 == 0;
            if is_even {
                // Even integer exponent: (-x)^2n = x^2n, always positive.
                return Ok(abs_pow);
            }
            // Odd integer exponent: negate where x < 0.
            let neg_mask = self.lt(0.0)?;
            let neg_result = abs_pow.neg()?;
            return neg_mask.where_cond(&neg_result, &abs_pow);
        }
        // Non-integer exponent: negative bases produce NaN per IEEE 754.
        // Use where_cond to replace negative-base results with NaN.
        let neg_mask = self.lt(0.0)?;
        let nan_fill = Self::full(self.dims(), f64::NAN, self.dtype(), &self.device())?;
        neg_mask.where_cond(&nan_fill, &abs_pow)
    }
}

// -- Masked fill --------------------------------------------------------------

impl DynTensor {
    /// Replace elements where `mask` is non-zero with `value`.
    ///
    /// Equivalent to PyTorch's `Tensor.masked_fill_(mask, value)` (non-mutating).
    /// Sugar for `mask.where_cond(&full_like(self, value), self)`.
    ///
    /// The mask must be U8 or F32 (0.0/1.0). Broadcasting between `self` and
    /// `mask` is supported via `where_cond`.
    pub fn masked_fill(&self, mask: &Self, value: f64) -> Result<Self> {
        let fill = Self::full(self.dims(), value, self.dtype, &self.device())?;
        mask.where_cond(&fill, self)
    }
}

/// Validate and convert `repeats` tensor to `Vec<usize>` counts.
fn repeat_interleave_validate_counts(repeats: &DynTensor, dim_size: usize) -> Result<Vec<usize>> {
    let repeats_cpu = repeats.to_device(&Device::Cpu)?;
    let repeats_arr = repeats_cpu.to_f32_array()?;
    let repeats_flat: Vec<f32> = repeats_arr.iter().copied().collect();
    if repeats_flat.len() != dim_size {
        return Err(TensorError::InvalidShape(format!(
            "repeat_interleave: repeats length {} != dim size {dim_size}",
            repeats_flat.len()
        )));
    }
    repeats_flat
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if !v.is_finite() || v < 0.0 || v != v.trunc() {
                Err(TensorError::InvalidShape(format!(
                    "repeat_interleave: repeat[{i}] = {v} must be a non-negative integer"
                )))
            } else if f64::from(v) > usize::MAX as f64 {
                Err(TensorError::InvalidShape(format!(
                    "repeat_interleave: repeat[{i}] = {v} exceeds usize::MAX"
                )))
            } else {
                Ok(v as usize)
            }
        })
        .collect::<Result<_>>()
}

/// Record trace op for repeat_interleave and return the result.
fn record_repeat_interleave_trace(
    mut result: DynTensor,
    x: &DynTensor,
    repeats: &DynTensor,
    dim: usize,
) -> Result<DynTensor> {
    if trace::is_tracing() {
        let input_ids = DynTensor::trace_input_ids(&[x, repeats])?;
        if let Some(id) = trace::record_op(
            TraceOp::RepeatInterleave { dim },
            &input_ids,
            result.dims(),
            result.dtype(),
        ) {
            result.set_trace_id(id);
        }
    }
    Ok(result)
}

// -- Sorting and selection (topk, arg_sort) -----------------------------------
mod sorting;

// -- Nearest-neighbor upsample + triangular masks + grid sample ----------------
mod spatial;
pub use spatial::grid_sample::GridSamplePaddingMode;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_extra;

#[cfg(test)]
mod tests_topk_boundary;

#[cfg(test)]
mod tests_triu_tril;

#[cfg(test)]
mod tests_sort_roll;

#[cfg(test)]
mod tests_resize_bilinear;

#[cfg(kani)]
#[path = "kani_ops_ext_proofs.rs"]
mod kani_ops_ext_proofs;

#[cfg(kani)]
#[path = "kani_ops_ext_extended_proofs.rs"]
mod kani_ops_ext_extended_proofs;

#[cfg(kani)]
#[path = "kani_dpdf_grid_sample_proofs.rs"]
mod kani_dpdf_grid_sample_proofs;

#[cfg(kani)]
#[path = "kani_dpdf_topk_sort_proofs.rs"]
mod kani_dpdf_topk_sort_proofs;

#[cfg(kani)]
#[path = "kani_dpdf_roll_proofs.rs"]
mod kani_dpdf_roll_proofs;

#[cfg(kani)]
#[path = "kani_dpdf_pixel_shuffle_upsample_proofs.rs"]
mod kani_dpdf_pixel_shuffle_upsample_proofs;
