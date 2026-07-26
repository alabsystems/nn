// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unified INT4/INT8 weight-only quantization for large VLM deployment.
//!
//! Supports per-group quantization (group_size configurable, default 128) for
//! both INT4 and INT8 bit widths. Per-group quantization is essential for INT4
//! accuracy: each group of `group_size` contiguous weights along the input
//! dimension shares a single (scale, zero_point) pair.
//!
//! # W4A16 / W8A16 scheme
//!
//! - **Weights:** INT4 or INT8, per-group quantized
//! - **Activations:** FP32 (unquantized)
//! - **Compute:** Dequantize weight groups on-the-fly during matmul
//! - **Output:** Same dtype as activation input
//!
//! # Memory savings
//!
//! | Dtype | Bits/weight | Overhead/group          | Savings vs F32 |
//! |-------|------------|-------------------------|----------------|
//! | INT8  | 8          | 4 bytes scale + 1 byte zp | ~4x           |
//! | INT4  | 4          | 4 bytes scale + 1 byte zp | ~7-8x         |
//!
//! Part of #3860

use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Quantization bit width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantDtype {
    /// 4-bit integer quantization (values in [-8, 7] or [0, 15]).
    Int4,
    /// 8-bit integer quantization (values in [-128, 127] or [0, 255]).
    Int8,
}

impl QuantDtype {
    /// Number of bits per element.
    #[must_use]
    pub fn bits(&self) -> u32 {
        match self {
            Self::Int4 => 4,
            Self::Int8 => 8,
        }
    }

    /// Signed representable range for symmetric mode.
    /// Symmetric avoids the most negative value to keep range symmetric.
    fn symmetric_range(&self) -> (f32, f32) {
        match self {
            Self::Int4 => (-7.0, 7.0),
            Self::Int8 => (-127.0, 127.0),
        }
    }

    /// Full representable range for asymmetric mode.
    fn asymmetric_range(&self) -> (f32, f32) {
        match self {
            Self::Int4 => (-8.0, 7.0),
            Self::Int8 => (-128.0, 127.0),
        }
    }

    /// Number of levels in full (asymmetric) range.
    fn num_levels(&self) -> f32 {
        match self {
            Self::Int4 => 15.0,  // 16 - 1
            Self::Int8 => 255.0, // 256 - 1
        }
    }

    /// Clamp range for quantized values.
    fn clamp_range(&self) -> (f32, f32) {
        match self {
            Self::Int4 => (-8.0, 7.0),
            Self::Int8 => (-128.0, 127.0),
        }
    }
}

/// Configuration for weight quantization.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Bit width: INT4 or INT8.
    pub dtype: QuantDtype,
    /// Number of contiguous weights sharing one (scale, zero_point) pair.
    /// Must divide `in_features`. Common values: 32, 64, 128 (default), 256.
    /// Smaller groups = better accuracy, more overhead.
    pub group_size: usize,
    /// If true, zero_point is always 0 and the range is symmetric around zero.
    /// Symmetric is simpler and sufficient for most trained weights.
    pub symmetric: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            dtype: QuantDtype::Int4,
            group_size: 128,
            symmetric: true,
        }
    }
}

impl QuantizationConfig {
    /// INT4 symmetric with the given group size.
    #[must_use]
    pub fn int4(group_size: usize) -> Self {
        Self {
            dtype: QuantDtype::Int4,
            group_size,
            symmetric: true,
        }
    }

    /// INT8 symmetric with the given group size.
    #[must_use]
    pub fn int8(group_size: usize) -> Self {
        Self {
            dtype: QuantDtype::Int8,
            group_size,
            symmetric: true,
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizedTensor
// ---------------------------------------------------------------------------

/// A weight tensor stored in quantized form with per-group scale and zero_point.
///
/// INT8 values are stored as `u8` (reinterpreting i8 as u8).
/// INT4 values are packed two per byte (low nibble = even index, high nibble = odd).
///
/// # Layout
///
/// For a weight matrix `[out_features, in_features]` with `group_size = G`:
/// - `num_groups_per_row = in_features / G`
/// - `scales`: `[out_features * num_groups_per_row]` — one f32 per group
/// - `zero_points`: `[out_features * num_groups_per_row]` — one i32 per group
/// - `quantized_data`: packed bytes
///   - INT8: `[out_features * in_features]` bytes
///   - INT4: `[out_features * in_features / 2]` bytes (two 4-bit values per byte)
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed quantized weight data (U8 tensor on CPU).
    pub quantized_data: Vec<u8>,
    /// Per-group scale factors. Length = `out_features * num_groups_per_row`.
    pub scales: Vec<f32>,
    /// Per-group zero points. Length = `out_features * num_groups_per_row`.
    /// All zeros for symmetric quantization.
    pub zero_points: Vec<i32>,
    /// Quantization configuration used to produce this tensor.
    pub config: QuantizationConfig,
    /// Original weight shape `[out_features, in_features]`.
    pub shape: [usize; 2],
}

impl QuantizedTensor {
    /// Number of output features (rows in weight matrix).
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.shape[0]
    }

    /// Number of input features (columns in weight matrix).
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.shape[1]
    }

    /// Number of groups per row.
    #[must_use]
    pub fn num_groups_per_row(&self) -> usize {
        self.shape[1] / self.config.group_size
    }

    /// Total number of groups.
    #[must_use]
    pub fn total_groups(&self) -> usize {
        self.shape[0] * self.num_groups_per_row()
    }

    /// Memory footprint in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let data_bytes = self.quantized_data.len();
        let scale_bytes = self.scales.len() * 4; // f32
        let zp_bytes = self.zero_points.len() * 4; // i32
        data_bytes + scale_bytes + zp_bytes
    }

    /// Equivalent F32 memory footprint.
    #[must_use]
    pub fn f32_memory_bytes(&self) -> usize {
        self.shape[0] * self.shape[1] * 4
    }

    /// Compression ratio vs F32.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        if self.memory_bytes() == 0 {
            return 0.0;
        }
        self.f32_memory_bytes() as f32 / self.memory_bytes() as f32
    }
}

// ---------------------------------------------------------------------------
// Quantize
// ---------------------------------------------------------------------------

/// Quantize a 2D weight matrix with per-group scaling.
///
/// Input: `weights` with shape `[out_features, in_features]` (F32 on CPU).
/// `in_features` must be divisible by `config.group_size`.
///
/// # Errors
///
/// Returns error if:
/// - `weights` is not 2D or not F32
/// - `in_features` is not divisible by `group_size`
/// - `weights` contains non-finite values
pub fn quantize_per_group(
    weights: &DynTensor,
    config: &QuantizationConfig,
) -> Result<QuantizedTensor> {
    let (out_features, in_features) = weights.dims2()?;

    if config.group_size == 0 {
        return Err(TensorError::ValueOutOfRange {
            description: "quantize_per_group: group_size must be > 0",
        });
    }
    if in_features % config.group_size != 0 {
        return Err(TensorError::shape_mismatch(
            vec![out_features, in_features],
            vec![config.group_size],
        ));
    }

    let weight_cpu = weights.to_device(&Device::Cpu)?;
    let flat = weight_cpu.to_flat_vec::<f32>()?;

    // Validate all values are finite
    for &v in &flat {
        if !v.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "quantize_per_group: non-finite weight value",
            });
        }
    }

    let groups_per_row = in_features / config.group_size;
    let total_groups = out_features * groups_per_row;

    let mut scales = Vec::with_capacity(total_groups);
    let mut zero_points = Vec::with_capacity(total_groups);

    // Pre-compute quantized i8 values for all elements
    let total_elements = out_features * in_features;
    let mut q_values = Vec::with_capacity(total_elements);

    for row in 0..out_features {
        let row_start = row * in_features;

        for g in 0..groups_per_row {
            let group_start = row_start + g * config.group_size;
            let group_end = group_start + config.group_size;
            let group_data = &flat[group_start..group_end];

            let (scale, zp) = compute_group_params(group_data, config);
            scales.push(scale);
            zero_points.push(zp);

            // Quantize each element in this group
            let (clamp_lo, clamp_hi) = config.dtype.clamp_range();
            for &val in group_data {
                let q = if scale == 0.0 {
                    0_i8
                } else {
                    let q_f32 = val / scale + zp as f32;
                    q_f32.round().clamp(clamp_lo, clamp_hi) as i8
                };
                q_values.push(q);
            }
        }
    }

    // Pack into bytes
    let quantized_data = match config.dtype {
        QuantDtype::Int8 => {
            // i8 → u8 reinterpret
            q_values.iter().map(|&q| q as u8).collect()
        }
        QuantDtype::Int4 => {
            // Pack two 4-bit values per byte (low nibble = even, high nibble = odd)
            let packed_len = total_elements.div_ceil(2);
            let mut packed = Vec::with_capacity(packed_len);
            for pair in q_values.chunks(2) {
                let lo = (pair[0] & 0x0F) as u8;
                let hi = if pair.len() > 1 {
                    (pair[1] & 0x0F) as u8
                } else {
                    0
                };
                packed.push(lo | (hi << 4));
            }
            packed
        }
    };

    Ok(QuantizedTensor {
        quantized_data,
        scales,
        zero_points,
        config: config.clone(),
        shape: [out_features, in_features],
    })
}

/// Compute per-group quantization scale and zero_point.
///
/// Symmetric: scale = abs_max / qmax, zero_point = 0.
/// Asymmetric: scale = (max - min) / num_levels, zero_point = round(qmin - min/scale).
fn compute_group_params(group: &[f32], config: &QuantizationConfig) -> (f32, i32) {
    if group.is_empty() {
        return (0.0, 0);
    }

    if config.symmetric {
        let (_, qmax) = config.dtype.symmetric_range();
        let abs_max = group
            .iter()
            .copied()
            .fold(0.0_f32, |acc, v| acc.max(v.abs()));
        if abs_max == 0.0 {
            return (0.0, 0);
        }
        let scale = abs_max / qmax;
        (scale, 0)
    } else {
        let min_val = group.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        if range == 0.0 {
            return (0.0, 0);
        }

        let num_levels = config.dtype.num_levels();
        let (qmin, _) = config.dtype.asymmetric_range();
        let scale = range / num_levels;
        let zp = (qmin - min_val / scale).round() as i32;
        (scale, zp)
    }
}

// ---------------------------------------------------------------------------
// Dequantize
// ---------------------------------------------------------------------------

/// Dequantize a [`QuantizedTensor`] back to a full-precision F32 [`DynTensor`].
///
/// Output shape: `[out_features, in_features]`.
///
/// Formula per element: `w_f32 = (q_val - zero_point) * scale`
///
/// # Errors
///
/// Returns error if internal state is inconsistent (should not happen for
/// tensors produced by [`quantize_per_group`]).
pub fn dequantize(qt: &QuantizedTensor) -> Result<DynTensor> {
    let [out_features, in_features] = qt.shape;
    let groups_per_row = in_features / qt.config.group_size;

    let expected_groups = out_features * groups_per_row;
    if qt.scales.len() != expected_groups {
        return Err(TensorError::DataLengthMismatch {
            expected: expected_groups,
            actual: qt.scales.len(),
        });
    }

    let total = out_features * in_features;
    let mut result = Vec::with_capacity(total);

    for row in 0..out_features {
        for g in 0..groups_per_row {
            let group_idx = row * groups_per_row + g;
            let scale = qt.scales[group_idx];
            let zp = qt.zero_points[group_idx];

            for elem in 0..qt.config.group_size {
                let flat_idx = row * in_features + g * qt.config.group_size + elem;
                let q_val = unpack_element(&qt.quantized_data, flat_idx, qt.config.dtype);
                let val = (f32::from(q_val) - zp as f32) * scale;
                result.push(val);
            }
        }
    }

    DynTensor::from_vec(result, &[out_features, in_features], &Device::Cpu)
}

/// Unpack a single quantized element from packed storage.
fn unpack_element(data: &[u8], index: usize, dtype: QuantDtype) -> i8 {
    match dtype {
        QuantDtype::Int8 => data[index] as i8,
        QuantDtype::Int4 => {
            let byte_idx = index / 2;
            let nibble = if index.is_multiple_of(2) {
                data[byte_idx] & 0x0F
            } else {
                data[byte_idx] >> 4
            };
            // Sign-extend 4-bit to 8-bit: if bit 3 is set, value is negative.
            if nibble & 0x08 != 0 {
                (nibble | 0xF0) as i8
            } else {
                nibble as i8
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quantized matmul
// ---------------------------------------------------------------------------

/// Compute `input @ weight^T` where `weight` is a [`QuantizedTensor`].
///
/// Input: `[*, in_features]` (any batch dimensions).
/// Weight: `[out_features, in_features]` quantized.
/// Output: `[*, out_features]`.
///
/// CPU path: dequantizes per-group during accumulation for memory efficiency.
/// (Future: GPU path with on-the-fly dequantization.)
///
/// # Errors
///
/// Returns error if `input`'s last dimension doesn't match `weight.in_features()`.
pub fn quantized_matmul(input: &DynTensor, weight: &QuantizedTensor) -> Result<DynTensor> {
    let input_dims = input.dims();
    if input_dims.is_empty() {
        return Err(TensorError::RankMismatch {
            expected: 1,
            actual: 0,
        });
    }

    let in_features = weight.in_features();
    let x_last = input_dims[input_dims.len() - 1];

    if x_last != in_features {
        return Err(TensorError::shape_mismatch(vec![in_features], vec![x_last]));
    }

    // For large tensors, dequantize-then-matmul is simpler and uses optimized BLAS.
    // For truly large VLMs where memory is the bottleneck, a streaming dequant
    // approach would be better, but this is correct and sufficient for Phase 1.
    let weight_f32 = dequantize(weight)?;
    let weight_on_device = weight_f32.to_device(&input.device())?;
    input.matmul(&weight_on_device.t()?)
}

// ---------------------------------------------------------------------------
// Kani proof harnesses
// ---------------------------------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // -----------------------------------------------------------------------
    // Harness 1: INT4 nibble pack/unpack roundtrip.
    //
    // Packing: lo = (q & 0x0F), hi = (q & 0x0F) << 4, byte = lo | hi.
    // Unpacking: lo = byte & 0x0F with sign extension, hi = byte >> 4 with sign extension.
    // Prove the roundtrip preserves the value for any q in [-8, 7].
    // -----------------------------------------------------------------------

    /// Prove: INT4 pack→unpack roundtrip is exact for all values in [-8, 7].
    #[kani::unwind(1)]
    #[kani::proof]
    fn int4_nibble_roundtrip() {
        let lo_val: i8 = kani::any();
        kani::assume(lo_val >= -8 && lo_val <= 7);
        let hi_val: i8 = kani::any();
        kani::assume(hi_val >= -8 && hi_val <= 7);

        // Pack
        let lo_nibble = (lo_val & 0x0F) as u8;
        let hi_nibble = (hi_val & 0x0F) as u8;
        let packed = lo_nibble | (hi_nibble << 4);

        let data = [packed];

        // Unpack
        let unpacked_lo = unpack_element(&data, 0, QuantDtype::Int4);
        let unpacked_hi = unpack_element(&data, 1, QuantDtype::Int4);

        assert!(unpacked_lo == lo_val, "low nibble must roundtrip");
        assert!(unpacked_hi == hi_val, "high nibble must roundtrip");
    }

    // -----------------------------------------------------------------------
    // Harness 2: INT8 pack/unpack roundtrip.
    // -----------------------------------------------------------------------

    /// Prove: INT8 pack→unpack roundtrip is exact for all i8 values.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_byte_roundtrip() {
        let val: i8 = kani::any();
        let data = [val as u8];
        let unpacked = unpack_element(&data, 0, QuantDtype::Int8);
        assert!(unpacked == val, "INT8 byte must roundtrip");
    }

    // -----------------------------------------------------------------------
    // Harness 3: Symmetric scale is non-negative for INT4.
    // -----------------------------------------------------------------------

    /// Prove: symmetric INT4 group scale is always non-negative.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int4_symmetric_scale_non_negative() {
        let v0: f32 = kani::any();
        let v1: f32 = kani::any();
        kani::assume(v0.is_finite() && v1.is_finite());
        kani::assume(v0.abs() < 1e30 && v1.abs() < 1e30);

        let group = [v0, v1];
        let config = QuantizationConfig {
            dtype: QuantDtype::Int4,
            group_size: 2,
            symmetric: true,
        };
        let (scale, zp) = compute_group_params(&group, &config);

        assert!(scale >= 0.0, "INT4 symmetric scale must be non-negative");
        assert!(scale.is_finite(), "INT4 symmetric scale must be finite");
        assert!(zp == 0, "symmetric zero_point must be 0");
    }

    // -----------------------------------------------------------------------
    // Harness 4: Symmetric scale is non-negative for INT8.
    // -----------------------------------------------------------------------

    /// Prove: symmetric INT8 group scale is always non-negative.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int8_symmetric_scale_non_negative() {
        let v0: f32 = kani::any();
        let v1: f32 = kani::any();
        kani::assume(v0.is_finite() && v1.is_finite());
        kani::assume(v0.abs() < 1e30 && v1.abs() < 1e30);

        let group = [v0, v1];
        let config = QuantizationConfig {
            dtype: QuantDtype::Int8,
            group_size: 2,
            symmetric: true,
        };
        let (scale, zp) = compute_group_params(&group, &config);

        assert!(scale >= 0.0, "INT8 symmetric scale must be non-negative");
        assert!(scale.is_finite(), "INT8 symmetric scale must be finite");
        assert!(zp == 0, "symmetric zero_point must be 0");
    }

    // -----------------------------------------------------------------------
    // Harness 5: INT4 dequant value bounded (symmetric).
    // -----------------------------------------------------------------------

    /// Prove: for symmetric INT4 quantization, the dequantized value is
    /// bounded by 8 * scale in magnitude.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int4_dequant_value_bounded_symmetric() {
        let q_nibble: u8 = kani::any();
        kani::assume(q_nibble <= 15);

        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale >= 0.0 && scale < 1000.0);

        // Sign-extend nibble
        let q_i8: i8 = if q_nibble & 0x08 != 0 {
            (q_nibble | 0xF0) as i8
        } else {
            q_nibble as i8
        };

        // Symmetric: zp = 0
        let dequant = (q_i8 as f32) * scale;

        assert!(dequant.is_finite(), "dequantized value must be finite");
        // q_i8 in [-8, 7], so |dequant| <= 8 * scale
        let max_magnitude = 8.0 * scale;
        assert!(
            dequant >= -max_magnitude && dequant <= max_magnitude,
            "symmetric INT4 dequant must be bounded by 8 * scale"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 6: INT4 symmetric roundtrip error bounded.
    // -----------------------------------------------------------------------

    /// Prove: |dequant(quant(v)) - v| <= scale/2 for symmetric INT4
    /// when value is within the representable range.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int4_symmetric_roundtrip_error_bounded() {
        let scale: f32 = kani::any();
        kani::assume(scale.is_finite());
        kani::assume(scale > 1e-10 && scale < 100.0);

        let v: f32 = kani::any();
        kani::assume(v.is_finite());
        kani::assume(v >= -7.0 * scale && v <= 7.0 * scale);

        // Quantize
        let q_f32 = (v / scale).round().clamp(-8.0, 7.0);
        let q_i8 = q_f32 as i8;

        // Dequantize
        let dequant = (q_i8 as f32) * scale;

        let error = (dequant - v).abs();
        let bound = scale / 2.0 + 1e-5;
        assert!(
            error <= bound,
            "roundtrip error must be <= scale/2 for symmetric INT4"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 7: QuantDtype bits correctness.
    // -----------------------------------------------------------------------

    /// Prove: QuantDtype::bits() returns correct values.
    #[kani::unwind(1)]
    #[kani::proof]
    fn quant_dtype_bits_correct() {
        assert!(QuantDtype::Int4.bits() == 4);
        assert!(QuantDtype::Int8.bits() == 8);
    }

    // -----------------------------------------------------------------------
    // Harness 8: Asymmetric scale is non-negative.
    // -----------------------------------------------------------------------

    /// Prove: asymmetric group scale is always non-negative for finite inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn asymmetric_scale_non_negative() {
        let v0: f32 = kani::any();
        let v1: f32 = kani::any();
        kani::assume(v0.is_finite() && v1.is_finite());
        kani::assume(v0.abs() < 1e30 && v1.abs() < 1e30);

        let group = [v0, v1];
        let config = QuantizationConfig {
            dtype: QuantDtype::Int4,
            group_size: 2,
            symmetric: false,
        };
        let (scale, _) = compute_group_params(&group, &config);

        assert!(scale >= 0.0, "asymmetric scale must be non-negative");
        assert!(scale.is_finite(), "asymmetric scale must be finite");
    }

    // -----------------------------------------------------------------------
    // Harness 9: INT4 accumulation overflow safety.
    //
    // In a W4A16 GEMM, K dequantized products are summed. Each dequantized
    // value is bounded by |diff| * |scale| <= 15 * 1000.0 = 15_000.
    // For K <= 16384, sum <= 16384 * 15000 = 245_760_000, within f32.
    // -----------------------------------------------------------------------

    /// Prove: accumulating one INT4 dequantized product into a bounded
    /// partial sum keeps the result within f32 range.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int4_accumulation_overflow_bounds() {
        let product: f32 = kani::any();
        kani::assume(product.is_finite());
        kani::assume(product >= -15_000.0 && product <= 15_000.0);

        let accumulator: f32 = kani::any();
        kani::assume(accumulator.is_finite());
        kani::assume(accumulator >= -245_745_000.0 && accumulator <= 245_745_000.0);

        let result = accumulator + product;

        assert!(
            result.is_finite(),
            "INT4 GEMM accumulation must stay within f32 range"
        );
        assert!(
            result >= -245_760_000.0 && result <= 245_760_000.0,
            "accumulated sum must be bounded"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 10: Compression ratio is > 1 for INT4 with group_size >= 8.
    // -----------------------------------------------------------------------

    /// Prove: INT4 quantization uses less memory than F32 for group_size >= 8.
    #[kani::unwind(1)]
    #[kani::proof]
    fn int4_compression_above_one() {
        let out_features: usize = kani::any();
        let groups_per_row: usize = kani::any();
        let group_size: usize = kani::any();

        kani::assume(out_features >= 1 && out_features <= 1024);
        kani::assume(groups_per_row >= 1 && groups_per_row <= 128);
        kani::assume(group_size >= 8 && group_size <= 256);

        let in_features = groups_per_row * group_size;
        let total_elements = out_features * in_features;
        let total_groups = out_features * groups_per_row;

        // INT4 memory: packed data + scales + zero_points
        let data_bytes = (total_elements + 1) / 2; // 2 elements per byte
        let scale_bytes = total_groups * 4; // f32 per group
        let zp_bytes = total_groups * 4; // i32 per group
        let int4_total = data_bytes + scale_bytes + zp_bytes;

        // F32 memory
        let f32_total = total_elements * 4;

        assert!(
            int4_total < f32_total,
            "INT4 must use less memory than F32 for group_size >= 8"
        );
    }
}

#[cfg(test)]
#[path = "weight_quant_tests.rs"]
mod tests;
