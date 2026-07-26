// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Softmax operation for [`DynTensor`].
//!
//! Extracted from `ops/mod.rs` for file-size compliance.

use super::super::{gpu_backend_dispatch, DynTensor};
use crate::{DType, Result, TensorError};
use ndarray::Zip;

/// Return dtype-appropriate clamping constants for softmax decomposition.
///
/// Returns `(max_finite, min_finite, min_positive)` as f64 values suitable
/// for `DynTensor::full()` with the given dtype. BF16/F16 use their native
/// range limits; F32/F64 use f32 limits (F64 demotes to F32 in DynTensor).
///
/// Fixes #1691: hardcoded f32 constants overflow bf16/f16 in
/// `FloatStorage::full()`, causing the GPU decomposed softmax to error.
pub(super) fn softmax_clamp_constants(dtype: DType) -> (f64, f64, f64) {
    match dtype {
        DType::BF16 => (
            f64::from(half::bf16::MAX),          // ~3.39e38
            f64::from(half::bf16::MIN),          // ~-3.39e38
            f64::from(half::bf16::MIN_POSITIVE), // ~1.18e-38
        ),
        DType::F16 => (
            f64::from(half::f16::MAX),          // 65504
            f64::from(half::f16::MIN),          // -65504
            f64::from(half::f16::MIN_POSITIVE), // ~6.1e-5
        ),
        // F32/F64 use f32 limits (F64 demotes to f32 in DynTensor storage).
        // Integer/bool dtypes should not reach softmax, but use f32 limits
        // as conservative fallback rather than silent `_ =>` catch-all (#1409).
        DType::F32
        | DType::F64
        | DType::I32
        | DType::I64
        | DType::U32
        | DType::U8
        | DType::Bool => (
            f64::from(f32::MAX),
            f64::from(f32::MIN),
            f64::from(f32::MIN_POSITIVE),
        ),
    }
}

/// GPU decomposed softmax: max→clamp→sub→exp→sum→clamp→div.
///
/// Guards against all-neg-inf lanes via max/sum clamping (#1326).
/// Guards against +inf lanes via input clamping (#1339): replacing +inf
/// with dtype MAX makes the max-subtract trick produce the correct limit
/// (uniform over +inf positions, 0 elsewhere) without NaN.
///
/// Scalar constants use the input tensor's dtype so bf16/f16 tensors get
/// dtype-appropriate clamping values (#1691).
pub(crate) fn gpu_softmax_decomposed(t: &DynTensor, dim: usize) -> Result<DynTensor> {
    let dt = t.dtype();
    let (max_val, min_val, min_pos) = softmax_clamp_constants(dt);
    // Clamp +inf to MAX so max-subtract trick works (#1339).
    let max_finite = DynTensor::full(&[], max_val, dt, &t.device())?;
    let t_clamped = t.minimum(&max_finite)?;
    let max_vals = t_clamped.max_keepdim(dim)?;
    let min_clamp = DynTensor::full(&[], min_val, dt, &t.device())?;
    let max_vals = max_vals.maximum(&min_clamp)?;
    let shifted = t_clamped.broadcast_sub(&max_vals)?;
    let exp_vals = shifted.exp()?;
    let sum_vals = exp_vals.sum_keepdim(dim)?;
    let sum_clamp = DynTensor::full(&[], min_pos, dt, &t.device())?;
    let sum_vals = sum_vals.maximum(&sum_clamp)?;
    exp_vals.broadcast_div(&sum_vals)
}

/// GPU decomposed log_softmax: max→clamp→sub→exp→sum→clamp→log→sub.
///
/// Guards against all-neg-inf lanes via max/sum clamping (#1326).
/// Guards against +inf lanes via input clamping (#1339): same approach
/// as `gpu_softmax_decomposed`.
/// When all inputs are -inf: shifted = -inf, exp = 0, sum clamped to eps,
/// log(eps) ≈ -87, result = -inf - (-87) = -inf (correct: log(0) = -inf).
///
/// Scalar constants use the input tensor's dtype (#1691).
pub(crate) fn gpu_log_softmax_decomposed(t: &DynTensor, dim: usize) -> Result<DynTensor> {
    let dt = t.dtype();
    let (max_val, min_val, min_pos) = softmax_clamp_constants(dt);
    // Clamp +inf to MAX so max-subtract trick works (#1339).
    let max_finite = DynTensor::full(&[], max_val, dt, &t.device())?;
    let t_clamped = t.minimum(&max_finite)?;
    let max_vals = t_clamped.max_keepdim(dim)?;
    let min_clamp = DynTensor::full(&[], min_val, dt, &t.device())?;
    let max_vals = max_vals.maximum(&min_clamp)?;
    let shifted = t_clamped.broadcast_sub(&max_vals)?;
    let exp_vals = shifted.exp()?;
    let sum_vals = exp_vals.sum_keepdim(dim)?;
    let sum_clamp = DynTensor::full(&[], min_pos, dt, &t.device())?;
    let sum_vals = sum_vals.maximum(&sum_clamp)?;
    let log_sum = sum_vals.log()?;
    shifted.broadcast_sub(&log_sum)
}

/// Softmax over the last dimension.
///
/// Returns an error if any input element is NaN, or if the last dimension
/// has zero length.
///
/// **Edge case: all-negative-infinity lanes.** When every element in a lane
/// is `−∞` (e.g., an attention row where all positions are masked), the
/// standard max-subtract trick produces `−∞ − (−∞) = NaN` under IEEE 754.
/// Instead of propagating NaN, this function zeros the entire lane, which
/// is the mathematically consistent limit (uniform distribution scaled to
/// zero total mass). Issue: #1310.
#[deprecated(note = "use t.softmax(D::Minus1) instead")]
pub fn softmax_last_dim(t: &DynTensor) -> Result<DynTensor> {
    let rank = t.rank();
    if rank == 0 {
        return Err(TensorError::InvalidShape(
            "softmax requires rank >= 1".into(),
        ));
    }
    let last_dim = t.dim(rank - 1)?;
    if last_dim == 0 {
        return Err(TensorError::ZeroLengthDimension {
            axis: rank - 1,
            operation: "softmax_last_dim",
        });
    }
    // Auto-upcast BF16/F16 to F32 for numerical stability (#1813).
    // PyTorch does this unconditionally for half-precision softmax.
    let input_dtype = t.dtype();
    if matches!(input_dtype, DType::BF16 | DType::F16) {
        let t_f32 = t.to_dtype(DType::F32)?;
        let result = softmax_last_dim(&t_f32)?;
        return result.to_dtype(input_dtype);
    }
    if t.device().is_gpu() {
        // GPU path: skip NaN pre-check to avoid GPU->CPU copy (#1138).
        // Try native GPU softmax kernel first.
        if let Some(result) = gpu_backend_dispatch(|b| b.softmax(t, rank - 1)) {
            return result;
        }
        // Fallback: decompose into GPU primitives (#1326 NaN guard).
        return gpu_softmax_decomposed(t, rank - 1);
    }
    // CPU path: reject NaN inputs (defense-in-depth, #941 pattern).
    // Inf/-Inf are allowed: attention masks use -inf for masked positions.
    // Promote bf16/f16 to f32 for computation (#1646 D3).
    let input_dtype = t.dtype();
    let arr = t.to_f32_array()?;
    let nan_count = arr.iter().filter(|v| v.is_nan()).count();
    if nan_count > 0 {
        return Err(TensorError::NonFiniteData {
            name: "softmax_last_dim input".into(),
            count: nan_count,
        });
    }
    let last_axis = ndarray::Axis(rank - 1);
    // Numerically stable softmax: subtract max per lane
    let max_vals = arr.map_axis(last_axis, |lane| {
        lane.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    });
    let mut result = arr.to_owned();
    // Subtract max and exponentiate
    Zip::from(result.lanes_mut(last_axis))
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
    // Preserve input dtype: bf16/f16 inputs return bf16/f16 probabilities (#1691).
    DynTensor::from_f32_result(result, input_dtype)
}

#[cfg(test)]
mod clamp_constants_tests {
    use super::softmax_clamp_constants;
    use crate::DType;

    /// BF16 MAX is ~3.39e38 (8 exponent bits), NOT ~65504 (that's F16).
    #[test]
    fn test_bf16_max_is_not_65504() {
        let (max, min, min_pos) = softmax_clamp_constants(DType::BF16);
        // bf16 has 8 exponent bits: MAX ≈ 3.39e38
        assert!(max > 1e37, "bf16 MAX should be ~3.39e38, got {max}");
        assert!(max < 4e38, "bf16 MAX should be ~3.39e38, got {max}");
        assert!(min < -1e37, "bf16 MIN should be ~-3.39e38, got {min}");
        assert!(min > -4e38, "bf16 MIN should be ~-3.39e38, got {min}");
        assert!(min_pos > 0.0, "bf16 MIN_POSITIVE must be positive");
        assert!(
            min_pos < 1e-37,
            "bf16 MIN_POSITIVE should be ~1.18e-38, got {min_pos}"
        );
    }

    /// F16 MAX is exactly 65504 (5 exponent bits, 10 mantissa).
    #[test]
    fn test_f16_max_is_65504() {
        let (max, min, min_pos) = softmax_clamp_constants(DType::F16);
        assert!(
            (max - 65504.0).abs() < 1.0,
            "f16 MAX should be 65504, got {max}"
        );
        assert!(
            (min + 65504.0).abs() < 1.0,
            "f16 MIN should be -65504, got {min}"
        );
        assert!(min_pos > 0.0, "f16 MIN_POSITIVE must be positive");
        assert!(
            min_pos < 1e-4,
            "f16 MIN_POSITIVE should be ~6.1e-5, got {min_pos}"
        );
    }

    /// F32 constants match the standard f32 range.
    #[test]
    fn test_f32_constants() {
        let (max, min, min_pos) = softmax_clamp_constants(DType::F32);
        assert_eq!(max, f64::from(f32::MAX));
        assert_eq!(min, f64::from(f32::MIN));
        assert_eq!(min_pos, f64::from(f32::MIN_POSITIVE));
    }

    /// BF16 and F16 have different MAX values (common mistake: confuse them).
    #[test]
    fn test_bf16_f16_max_differ() {
        let (bf16_max, _, _) = softmax_clamp_constants(DType::BF16);
        let (f16_max, _, _) = softmax_clamp_constants(DType::F16);
        // bf16 MAX (~3.39e38) is vastly larger than f16 MAX (65504)
        assert!(
            bf16_max > f16_max * 1e30,
            "bf16 MAX ({bf16_max}) should be >> f16 MAX ({f16_max})"
        );
    }

    /// Integer dtypes fall through to f32 limits (conservative fallback).
    #[test]
    fn test_integer_dtypes_use_f32_limits() {
        for dt in [DType::U32, DType::U8, DType::I32, DType::I64, DType::Bool] {
            let (max, min, min_pos) = softmax_clamp_constants(dt);
            assert_eq!(
                max,
                f64::from(f32::MAX),
                "integer dtype {dt:?} should use f32 MAX"
            );
            assert_eq!(
                min,
                f64::from(f32::MIN),
                "integer dtype {dt:?} should use f32 MIN"
            );
            assert_eq!(
                min_pos,
                f64::from(f32::MIN_POSITIVE),
                "integer dtype {dt:?} should use f32 MIN_POSITIVE"
            );
        }
    }
}
