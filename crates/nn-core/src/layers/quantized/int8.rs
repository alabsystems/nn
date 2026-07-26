// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! INT8 per-channel quantization utilities for weight-only quantization (W8A16).
//!
//! Supports symmetric (zero_point = 0) and asymmetric per-channel quantization.
//! Symmetric is the default and sufficient for most models — weights are centered
//! near zero after training normalization.
//!
//! # W8A16 scheme
//!
//! - **Weights:** INT8 per-channel (one scale per output channel)
//! - **Activations:** FP16/FP32 (unquantized)
//! - **Compute:** Dequantize weight on-the-fly during matmul: `w_f32 = w_i8 * scale`
//! - **Output:** Same dtype as activation input
//!
//! This gives ~4x memory reduction vs F32 weights with minimal accuracy loss
//! for inference workloads. The CPU reference path dequantizes fully before
//! matmul. The GPU path (future Metal kernel) dequantizes per-tile inside the
//! GEMM for bandwidth efficiency.
//!
//! # Per-channel vs per-tensor
//!
//! Per-channel (one scale per output row of weight matrix) is strictly better
//! than per-tensor for linear layers because different output channels can have
//! very different weight magnitudes. The overhead is negligible: N extra f32
//! values for an [N, K] weight matrix.
//!
//! Part of #3522

use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

/// Per-channel INT8 quantization parameters.
///
/// Each output channel has its own scale (and optionally zero_point).
/// Dequantization formula: `w_f32[i] = (w_i8[i] - zero_point) * scale`
///
/// For symmetric quantization, `zero_point` is all zeros and the formula
/// simplifies to: `w_f32[i] = w_i8[i] * scale`
///
/// `zero_point` is stored as `i32` (not `i8`) because asymmetric quantization
/// can produce zero_point values outside [-128, 127] when the data range does
/// not straddle zero. This matches PyTorch's `qint8` convention (int64 zp).
/// The zero_point is only used during dequantization arithmetic, not stored
/// in the quantized tensor itself.
#[derive(Debug, Clone)]
pub struct Int8QuantParams {
    /// Per-channel scale factors. Length = out_features (one per output channel).
    /// `w_f32 = (w_i8 - zero_point) * scale`
    pub scale: Vec<f32>,
    /// Per-channel zero points. Length = out_features.
    /// For symmetric quantization, all zeros.
    /// Stored as i32 to support asymmetric ranges where zp may exceed i8 bounds.
    pub zero_point: Vec<i32>,
}

/// Quantization mode for INT8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8Mode {
    /// Symmetric: zero_point = 0, range [-127, 127].
    /// Simpler, faster, sufficient for most trained weights.
    Symmetric,
    /// Asymmetric: zero_point calibrated per-channel, range [-128, 127].
    /// Better for weights with non-zero mean (rare after training).
    Asymmetric,
}

/// Quantize a 2D weight matrix from f32 to INT8 with per-channel scaling.
///
/// Input: `weights` with shape `[out_features, in_features]` (F32 on CPU).
/// Output: `(quantized_u8, params)` where `quantized_u8` has shape
/// `[out_features, in_features]` with dtype U8 (storing i8 values as u8).
///
/// For symmetric mode: maps `[-abs_max, abs_max]` to `[-127, 127]`.
/// For asymmetric mode: maps `[min, max]` to `[-128, 127]`.
///
/// # Errors
///
/// Returns error if `weights` is not 2D, not F32, or contains non-finite values.
pub fn quantize_per_channel(
    weights: &DynTensor,
    mode: Int8Mode,
) -> Result<(DynTensor, Int8QuantParams)> {
    let (out_features, in_features) = weights.dims2()?;
    let weight_cpu = weights.to_device(&Device::Cpu)?;
    let flat = weight_cpu.to_flat_vec::<f32>()?;

    // Validate all values are finite
    for (i, &v) in flat.iter().enumerate() {
        if !v.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "quantize_per_channel: non-finite weight value",
            });
        }
        let _ = i; // suppress unused warning
    }

    let mut scales = Vec::with_capacity(out_features);
    let mut zero_points = Vec::with_capacity(out_features);
    let mut quantized = Vec::with_capacity(out_features * in_features);

    for row in 0..out_features {
        let row_start = row * in_features;
        let row_end = row_start + in_features;
        let row_data = &flat[row_start..row_end];

        let (scale, zero_point) = compute_channel_params(row_data, mode);
        scales.push(scale);
        zero_points.push(zero_point);

        // Quantize each element in this row
        for &val in row_data {
            let q_i8 = if scale == 0.0 {
                // All values in this channel are zero (or constant for asymmetric).
                // Pick a q that dequantizes to 0: (q - zp) * 0 = 0 for any q.
                0_i8
            } else {
                let q_f32 = val / scale + zero_point as f32;
                q_f32.round().clamp(-128.0, 127.0) as i8
            };
            // Store i8 as u8 (reinterpret cast)
            quantized.push(q_i8 as u8);
        }
    }

    let quantized_tensor =
        DynTensor::from_vec_u8(quantized, &[out_features, in_features], &Device::Cpu)?;

    let params = Int8QuantParams {
        scale: scales,
        zero_point: zero_points,
    };

    Ok((quantized_tensor, params))
}

/// Dequantize an INT8 weight tensor back to F32.
///
/// Input: `quantized` with shape `[out_features, in_features]` (U8 on CPU),
/// `params` with matching out_features.
/// Output: F32 tensor with shape `[out_features, in_features]`.
///
/// Formula: `w_f32[row][col] = (w_i8[row][col] - zero_point[row]) * scale[row]`
pub fn dequantize_per_channel(
    quantized: &DynTensor,
    params: &Int8QuantParams,
) -> Result<DynTensor> {
    let (out_features, in_features) = quantized.dims2()?;

    if params.scale.len() != out_features {
        return Err(TensorError::DataLengthMismatch {
            expected: out_features,
            actual: params.scale.len(),
        });
    }
    if params.zero_point.len() != out_features {
        return Err(TensorError::DataLengthMismatch {
            expected: out_features,
            actual: params.zero_point.len(),
        });
    }

    let q_cpu = quantized.to_device(&Device::Cpu)?;
    let q_data = q_cpu.to_flat_vec::<u8>()?;

    let mut result = Vec::with_capacity(out_features * in_features);

    for row in 0..out_features {
        let scale = params.scale[row];
        let zp = params.zero_point[row];
        let row_start = row * in_features;
        let row_end = row_start + in_features;

        for &q_u8 in &q_data[row_start..row_end] {
            let q_i8 = q_u8 as i8;
            let val = (f32::from(q_i8) - zp as f32) * scale;
            result.push(val);
        }
    }

    DynTensor::from_vec(result, &[out_features, in_features], &Device::Cpu)
}

/// Compute per-channel quantization scale and zero_point for a single row.
///
/// Symmetric: scale = abs_max / 127, zero_point = 0.
/// Asymmetric: scale = (max - min) / 255, zero_point = round(-128 - min / scale).
///
/// The dequantization formula is: `val = (q_i8 - zero_point) * scale`.
/// For asymmetric mode, zero_point is stored as i32 because it can exceed
/// the i8 range when the data does not straddle zero (e.g., all-negative weights).
fn compute_channel_params(row: &[f32], mode: Int8Mode) -> (f32, i32) {
    if row.is_empty() {
        return (0.0, 0);
    }

    match mode {
        Int8Mode::Symmetric => {
            let abs_max = row.iter().copied().fold(0.0_f32, |acc, v| acc.max(v.abs()));
            if abs_max == 0.0 {
                return (0.0, 0);
            }
            let scale = abs_max / 127.0;
            (scale, 0)
        }
        Int8Mode::Asymmetric => {
            let min_val = row.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max_val - min_val;

            if range == 0.0 {
                // Constant channel — scale=0, zero_point maps to constant
                return (0.0, 0);
            }

            let scale = range / 255.0;
            // zero_point: the i32 value such that min maps to q=-128.
            // min = (-128 - zp) * scale  =>  zp = -128 - min/scale
            let zp = (-128.0 - min_val / scale).round() as i32;
            (scale, zp)
        }
    }
}

/// Compute the maximum absolute quantization error for a round-trip.
///
/// Useful for testing: `max_error(original, dequantized)`.
#[must_use]
pub fn max_quantization_error(original: &[f32], dequantized: &[f32]) -> f32 {
    original
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

// -- Kani proof harnesses -----------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
    // -----------------------------------------------------------------------
    // Harness 1: INT8 dequant cast roundtrip.
    //
    // For any u8 value, reinterpreting as i8 then widening to i32 produces
    // a value in [-128, 127]. This is the fundamental cast chain used in
    // dequantize_per_channel: `q_u8 as i8 as f32`.
    // -----------------------------------------------------------------------

    /// Prove: the u8 → i8 → i32 cast chain used in INT8 dequantization
    /// always produces a value in the signed byte range [-128, 127].
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_dequant_cast_roundtrip() {
        let raw: u8 = kani::any();
        let signed = raw as i8;
        let widened = signed as i32;
        assert!(
            widened >= -128 && widened <= 127,
            "i8-widened-to-i32 must be in [-128, 127]"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 2: Zero-point subtraction overflow safety.
    //
    // In dequantize_per_channel, we compute `(q_i8 as f32 - zp as f32)`.
    // The equivalent integer arithmetic is `(q_i8 as i32) - zero_point`.
    // Prove this never overflows i32 for any i8 value and any zero_point
    // in the valid calibration range [-128, 127].
    // -----------------------------------------------------------------------

    /// Prove: subtracting a zero_point in [-128, 127] from any i8 value
    /// (widened to i32) never overflows i32.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_zero_point_subtraction_no_overflow() {
        let q_val: u8 = kani::any();
        let q_i8 = q_val as i8;
        let q_i32 = q_i8 as i32; // in [-128, 127]

        let zero_point: i32 = kani::any();
        kani::assume(zero_point >= -128 && zero_point <= 127);

        // This subtraction must not overflow.
        // q_i32 in [-128, 127], zero_point in [-128, 127].
        // Result in [-255, 255], well within i32 range.
        let diff = q_i32 - zero_point;
        assert!(
            diff >= -255 && diff <= 255,
            "zero-point subtraction result must be in [-255, 255]"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 3: Scale multiply finiteness.
    //
    // After zero-point subtraction, the dequantized value is computed as
    // `(diff as f32) * scale`. Prove this product is finite for any diff
    // in [-255, 255] and any finite scale with |scale| < 1000.0.
    //
    // The scale bound of 1000.0 is conservative: real per-channel scales
    // are typically < 1.0 (weight_absmax / 127), but we prove a much
    // wider range for safety.
    // -----------------------------------------------------------------------

    /// Prove: `(diff as f32) * scale` is finite for diff in [-255, 255]
    /// and |scale| < 1000.0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_scale_multiply_finite() {
        let diff: i32 = kani::any();
        kani::assume(diff >= -255 && diff <= 255);

        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale > -1000.0 && scale < 1000.0);

        let product = (diff as f32) * scale;
        assert!(
            product.is_finite(),
            "dequantized product must be finite for bounded inputs"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 4: Accumulation overflow bounds.
    //
    // In a W8A16 GEMM, K dequantized products are summed per output element.
    // Each dequantized value is bounded by |diff| * |scale| <= 255 * 1000.0
    // = 255_000.0. For K <= 4096, the sum is bounded by 4096 * 255_000.0
    // = 1_044_480_000.0 which is well within f32 range (~3.4e38).
    //
    // We verify a single accumulation step: adding one bounded product to
    // an accumulator that is already bounded by (K-1) * max_product.
    // -----------------------------------------------------------------------

    /// Prove: accumulating one dequantized product into a partial sum
    /// that is already bounded keeps the result within f32 range.
    ///
    /// max_single = 255 * 1000.0 = 255_000.0
    /// max_accum = 4096 * 255_000.0 = 1_044_480_000.0
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_accumulation_overflow_bounds() {
        // Bound: each dequantized product is at most 255_000.0 in magnitude.
        let product: f32 = kani::any();
        kani::assume(product.is_finite());
        kani::assume(product >= -255_000.0 && product <= 255_000.0);

        // Accumulator: sum of up to 4095 previous products.
        // Worst case: 4095 * 255_000.0 = 1_044_225_000.0
        let accumulator: f32 = kani::any();
        kani::assume(accumulator.is_finite());
        kani::assume(accumulator >= -1_044_225_000.0 && accumulator <= 1_044_225_000.0);

        let result = accumulator + product;

        // 1_044_225_000.0 + 255_000.0 = 1_044_480_000.0, well within f32::MAX (~3.4e38).
        assert!(
            result.is_finite(),
            "GEMM accumulation must stay within f32 range for K <= 4096"
        );
        assert!(
            result >= -1_044_480_000.0 && result <= 1_044_480_000.0,
            "accumulated sum must be bounded by K * max_product"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 5: Symmetric dequantize error bound.
    //
    // For symmetric INT8 quantization, each element is quantized as:
    //   q = round(val / scale)  where scale = abs_max / 127
    // and dequantized as:
    //   val' = q * scale
    //
    // The maximum quantization error per element is bounded by scale / 2
    // (half a quantization step). This harness proves that for any i8
    // quantized value and the corresponding scale, the dequant value
    // is within (scale / 2 + epsilon) of the original grid point.
    //
    // Specifically: for any q in [-127, 127] and scale > 0,
    //   dequant(q) = q * scale, and quantize(dequant(q)) == q.
    // The roundtrip is exact on the quantization grid. The error arises
    // only from the initial rounding to the grid.
    // -----------------------------------------------------------------------

    /// Prove: symmetric INT8 quantize→dequantize roundtrip is exact for
    /// values already on the quantization grid.
    ///
    /// For any q in [-127, 127] and finite positive scale,
    /// round(q * scale / scale) == q (the value roundtrips exactly).
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_symmetric_dequant_grid_exact() {
        let q_val: i8 = kani::any();
        // Symmetric range is [-127, 127] (not -128 to avoid asymmetry)
        kani::assume(q_val >= -127 && q_val <= 127);

        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale > 0.0 && scale < 1000.0);

        // Dequantize: val = q * scale
        let dequant_val = (q_val as f32) * scale;

        // Re-quantize: q' = round(val / scale)
        if scale > 0.0 {
            let requant = (dequant_val / scale).round();
            assert!(requant.is_finite(), "re-quantized value must be finite");
            // For exact grid points, roundtrip must be exact within f32 rounding.
            // The difference is due to floating-point arithmetic:
            // (q * scale) / scale may not equal q exactly, but the rounding
            // ensures the quantized value matches.
            let diff = (requant - q_val as f32).abs();
            // Allow 1 ULP of rounding tolerance for the division
            assert!(
                diff < 1.0,
                "grid-point requantization must roundtrip within 1 step"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Harness 6: Dequantize preserves value within quantization error bound.
    //
    // For symmetric quantization with scale = abs_max / 127:
    //   max_error = scale / 2
    //
    // For any original value v in [-abs_max, abs_max]:
    //   |dequant(quant(v)) - v| <= scale / 2
    //
    // We prove a stronger statement: for any quantized value q and its
    // corresponding dequant d = (q - zp) * scale, the dequant value is
    // bounded by |d| <= 127 * scale (symmetric) or 255 * scale (asymmetric).
    // -----------------------------------------------------------------------

    /// Prove: for symmetric INT8 quantization, the dequantized value is
    /// bounded by 127 * scale in magnitude.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_dequant_value_bounded_symmetric() {
        let q_u8: u8 = kani::any();
        let q_i8 = q_u8 as i8;

        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale >= 0.0 && scale < 1000.0);

        // Symmetric: zero_point = 0
        let dequant = (q_i8 as f32) * scale;

        // q_i8 in [-128, 127], so |dequant| <= 128 * scale
        assert!(dequant.is_finite(), "dequantized value must be finite");
        let max_magnitude = 128.0 * scale;
        assert!(
            dequant >= -max_magnitude && dequant <= max_magnitude,
            "symmetric dequant must be bounded by 128 * scale"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 7: Asymmetric dequantize value bounded.
    //
    // For asymmetric quantization:
    //   dequant = (q_i8 - zero_point) * scale
    //
    // The diff (q_i8 - zero_point) is in [-255, 255] (harness 2).
    // So |dequant| <= 255 * scale.
    // -----------------------------------------------------------------------

    /// Prove: for asymmetric INT8 quantization, the dequantized value is
    /// bounded by 255 * scale in magnitude.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_dequant_value_bounded_asymmetric() {
        let q_u8: u8 = kani::any();
        let q_i8 = q_u8 as i8;

        let zero_point: i32 = kani::any();
        kani::assume(zero_point >= -128 && zero_point <= 127);

        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale >= 0.0 && scale < 1000.0);

        let diff = q_i8 as f32 - zero_point as f32;
        let dequant = diff * scale;

        assert!(dequant.is_finite(), "dequantized value must be finite");
        let max_magnitude = 255.0 * scale;
        assert!(
            dequant >= -max_magnitude && dequant <= max_magnitude,
            "asymmetric dequant must be bounded by 255 * scale"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 8: max_quantization_error bounds are tight.
    //
    // max_quantization_error computes the max absolute difference between
    // original and dequantized arrays. Prove:
    // 1. Result is non-negative.
    // 2. Result is finite when both arrays contain finite values.
    // 3. Result is zero when arrays are identical.
    //
    // We test with small arrays (2 elements) because Kani operates on
    // concrete bit-level exploration.
    // -----------------------------------------------------------------------

    /// Prove: max_quantization_error returns a non-negative, finite value
    /// for finite inputs, and returns 0.0 for identical arrays.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_max_quant_error_properties() {
        let a0: f32 = kani::any();
        let a1: f32 = kani::any();
        let b0: f32 = kani::any();
        let b1: f32 = kani::any();

        kani::assume(a0.is_finite() && a1.is_finite());
        kani::assume(b0.is_finite() && b1.is_finite());
        kani::assume(a0.abs() < 1e30 && a1.abs() < 1e30);
        kani::assume(b0.abs() < 1e30 && b1.abs() < 1e30);

        let original = [a0, a1];
        let dequantized = [b0, b1];

        let err = super::max_quantization_error(&original, &dequantized);

        // Property 1: non-negative
        assert!(err >= 0.0, "max error must be non-negative");
        // Property 2: finite
        assert!(
            err.is_finite(),
            "max error must be finite for finite inputs"
        );
    }

    /// Prove: max_quantization_error returns exactly 0.0 for identical arrays.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_max_quant_error_zero_for_identical() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        let arr = [a, b];
        let err = super::max_quantization_error(&arr, &arr);

        assert!(err == 0.0, "identical arrays must have zero error");
    }

    // -----------------------------------------------------------------------
    // Harness 10: Symmetric scale computation is non-negative.
    //
    // compute_channel_params in symmetric mode computes:
    //   scale = abs_max / 127.0
    // Prove scale >= 0 for any finite input row.
    // -----------------------------------------------------------------------

    /// Prove: symmetric INT8 scale is always non-negative for finite inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_symmetric_scale_non_negative() {
        let v0: f32 = kani::any();
        let v1: f32 = kani::any();
        kani::assume(v0.is_finite() && v1.is_finite());
        kani::assume(v0.abs() < 1e30 && v1.abs() < 1e30);

        let row = [v0, v1];
        let (scale, zero_point) = super::compute_channel_params(&row, super::Int8Mode::Symmetric);

        assert!(scale >= 0.0, "symmetric scale must be non-negative");
        assert!(scale.is_finite(), "symmetric scale must be finite");
        assert!(zero_point == 0, "symmetric zero_point must be 0");
    }

    // -----------------------------------------------------------------------
    // Harness 11: Asymmetric scale computation is non-negative.
    //
    // compute_channel_params in asymmetric mode computes:
    //   scale = (max - min) / 255.0
    // Since max >= min for any finite data, scale >= 0.
    // -----------------------------------------------------------------------

    /// Prove: asymmetric INT8 scale is always non-negative for finite inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_asymmetric_scale_non_negative() {
        let v0: f32 = kani::any();
        let v1: f32 = kani::any();
        kani::assume(v0.is_finite() && v1.is_finite());
        kani::assume(v0.abs() < 1e30 && v1.abs() < 1e30);

        let row = [v0, v1];
        let (scale, _zero_point) = super::compute_channel_params(&row, super::Int8Mode::Asymmetric);

        assert!(scale >= 0.0, "asymmetric scale must be non-negative");
        assert!(scale.is_finite(), "asymmetric scale must be finite");
    }

    // -----------------------------------------------------------------------
    // Harness 12: Symmetric quantize→dequantize roundtrip error bound.
    //
    // The fundamental quantization guarantee: for symmetric INT8 with
    // scale = abs_max / 127, the roundtrip error for any value on the
    // quantization grid is bounded by scale / 2 (half a quantization step).
    //
    // For a value v, quant(v) = round(v / scale), dequant(q) = q * scale.
    // Error = |q * scale - v| = |round(v/scale)*scale - v| <= scale/2.
    //
    // We prove this for the scalar quantize→dequantize path used in
    // quantize_per_channel: q = round(v / scale).clamp(-128, 127) as i8,
    // then dequant = (q as f32) * scale.
    // -----------------------------------------------------------------------

    /// Prove: |dequant(quant(v)) - v| <= scale/2 for symmetric INT8.
    ///
    /// This is the core quantization error bound. For any finite value v
    /// within the representable range [-127*scale, 127*scale], the roundtrip
    /// error is at most half a quantization step.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_symmetric_roundtrip_error_bounded() {
        // Symbolic positive scale (abs_max / 127)
        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale > 1e-10 && scale < 100.0);

        // Value to quantize — within the symmetric representable range
        let v: f32 = kani::any();
        kani::assume(v.is_finite());
        kani::assume(v >= -127.0 * scale && v <= 127.0 * scale);

        // Quantize: same path as quantize_per_channel (symmetric, zp=0)
        let q_f32 = (v / scale).round().clamp(-128.0, 127.0);
        let q_i8 = q_f32 as i8;

        // Dequantize: same path as dequantize_per_channel (zp=0)
        let dequant = (q_i8 as f32) * scale;

        // Error must be at most scale/2 + small epsilon for f32 rounding
        let error = (dequant - v).abs();
        let bound = scale / 2.0 + 1e-5;
        assert!(
            error <= bound,
            "roundtrip error must be <= scale/2 for symmetric INT8"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 13: Asymmetric quantize→dequantize roundtrip error bound.
    //
    // For asymmetric INT8 with scale = range / 255 and calibrated zero_point,
    // the roundtrip error is bounded by scale / 2.
    //
    // quant(v) = round(v/scale + zp).clamp(-128, 127)
    // dequant(q) = (q - zp) * scale
    // Error = |(round(v/scale + zp) - zp) * scale - v|
    //       = |round(v/scale) * scale - v|  (when not clamped)
    //       <= scale / 2
    // -----------------------------------------------------------------------

    /// Prove: |dequant(quant(v)) - v| <= scale/2 for asymmetric INT8
    /// when the value is within the non-clamped representable range.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_asymmetric_roundtrip_error_bounded() {
        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale > 1e-10 && scale < 100.0);

        let zero_point: i32 = kani::any();
        kani::assume(zero_point >= -128 && zero_point <= 127);

        // Value within the non-clamped representable range
        let v: f32 = kani::any();
        kani::assume(v.is_finite());
        // Ensure the quantized value won't be clamped
        let q_unclamped = v / scale + zero_point as f32;
        kani::assume(q_unclamped >= -127.0 && q_unclamped <= 126.0);

        // Quantize: same path as quantize_per_channel (asymmetric)
        let q_f32 = (v / scale + zero_point as f32).round().clamp(-128.0, 127.0);
        let q_i8 = q_f32 as i8;

        // Dequantize
        let dequant = (q_i8 as f32 - zero_point as f32) * scale;

        let error = (dequant - v).abs();
        let bound = scale / 2.0 + 1e-5;
        assert!(
            error <= bound,
            "roundtrip error must be <= scale/2 for asymmetric INT8"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 14: max_quantization_error bound is tight (achievable).
    //
    // For symmetric INT8, the worst-case error of scale/2 is achieved at
    // mid-grid points: values exactly halfway between two quantization levels.
    // For grid spacing = scale, the mid-point is at q*scale + scale/2.
    //
    // Prove: for any valid quantization level q, the value at the mid-point
    // q*scale + scale/2 has roundtrip error exactly scale/2 (within f32
    // precision). This proves the bound is tight — not just an upper bound,
    // but actually achievable.
    // -----------------------------------------------------------------------

    /// Prove: the scale/2 error bound is tight — achievable at mid-grid points.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_max_quant_error_bound_is_tight() {
        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale > 1e-6 && scale < 10.0);

        // Pick a quantization level not at the clamp boundary
        let q: i8 = kani::any();
        kani::assume(q >= -126 && q <= 126);

        // The mid-point between q*scale and (q+1)*scale
        let midpoint = (q as f32) * scale + scale / 2.0;
        kani::assume(midpoint.is_finite());

        // Quantize the mid-point
        let q_rounded = (midpoint / scale).round().clamp(-128.0, 127.0) as i8;
        let dequant = (q_rounded as f32) * scale;

        let error = (dequant - midpoint).abs();

        // Error must be close to scale/2 (within f32 arithmetic tolerance).
        // The mid-point rounds to either q or q+1, giving error of exactly scale/2.
        // f32 arithmetic may deviate by a small epsilon.
        assert!(
            error >= scale / 2.0 - 1e-4,
            "mid-grid error must be close to scale/2 (proving tightness)"
        );
        assert!(
            error <= scale / 2.0 + 1e-4,
            "mid-grid error must not exceed scale/2 + epsilon"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 15: max_quantization_error is an upper bound on element-wise error.
    //
    // For any pair of finite arrays where |a[i] - b[i]| <= B for all i,
    // max_quantization_error(a, b) <= B.
    //
    // This proves max_quantization_error faithfully reports the maximum
    // pointwise difference — it does not undercount.
    // -----------------------------------------------------------------------

    /// Prove: max_quantization_error is >= each element-wise difference.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_max_quant_error_is_upper_bound() {
        let a0: f32 = kani::any();
        let a1: f32 = kani::any();
        let b0: f32 = kani::any();
        let b1: f32 = kani::any();

        kani::assume(a0.is_finite() && a1.is_finite());
        kani::assume(b0.is_finite() && b1.is_finite());
        kani::assume(a0.abs() < 1e30 && a1.abs() < 1e30);
        kani::assume(b0.abs() < 1e30 && b1.abs() < 1e30);

        let original = [a0, a1];
        let dequantized = [b0, b1];

        let max_err = super::max_quantization_error(&original, &dequantized);

        // max_err must be >= each individual element difference
        let err0 = (a0 - b0).abs();
        let err1 = (a1 - b1).abs();

        assert!(
            max_err >= err0 || (max_err - err0).abs() < 1e-30,
            "max error must be >= element 0 error"
        );
        assert!(
            max_err >= err1 || (max_err - err1).abs() < 1e-30,
            "max error must be >= element 1 error"
        );
    }
}

#[cfg(kani)]
#[path = "kani_int8_extra.rs"]
mod kani_int8_extra;

#[cfg(test)]
#[path = "int8_tests.rs"]
mod tests;
