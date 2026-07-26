// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MXFP4 (OCP Microscaling FP4) quantization for gpt-oss MoE expert weights.
//!
//! Format per block of 32 values:
//! - 1 shared E8M0 scale (8 bits, exponent only, no mantissa)
//! - 32 FP4 values (4 bits each = 16 bytes packed 2-per-byte)
//! - Total: 17 bytes per 32 values (4.25 bits/value)
//!
//! FP4 encoding (4 bits): 1 sign, 2 exponent, 1 mantissa.
//! Representable values: +/-{0, 0.5, 1, 1.5, 2, 3, 4, 6}.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Device, Result};

/// MXFP4 block size: 32 elements share one E8M0 scale factor.
pub const MXFP4_BLOCK_SIZE: usize = 32;

/// FP4 lookup table: maps 4-bit code -> f32 value.
///
/// Encoding: bit[3]=sign, bits[2:1]=exponent, bit[0]=mantissa.
/// Positive values (codes 0-7): {0, 0.5, 1, 1.5, 2, 3, 4, 6}
/// Negative values (codes 8-15): negated positive values.
///
/// Reference: OCP Microscaling Formats (MX) Specification, Table 5.
const FP4_LUT: [f32; 16] = [
    0.0,  // 0b0000: +0
    0.5,  // 0b0001: +0.5 (subnormal: 0.1 * 2^0)
    1.0,  // 0b0010: +1.0 (1.0 * 2^0)
    1.5,  // 0b0011: +1.5 (1.1 * 2^0)
    2.0,  // 0b0100: +2.0 (1.0 * 2^1)
    3.0,  // 0b0101: +3.0 (1.1 * 2^1)
    4.0,  // 0b0110: +4.0 (1.0 * 2^2)
    6.0,  // 0b0111: +6.0 (1.1 * 2^2)
    -0.0, // 0b1000: -0
    -0.5, // 0b1001: -0.5
    -1.0, // 0b1010: -1.0
    -1.5, // 0b1011: -1.5
    -2.0, // 0b1100: -2.0
    -3.0, // 0b1101: -3.0
    -4.0, // 0b1110: -4.0
    -6.0, // 0b1111: -6.0
];

/// Absolute values of the 8 positive FP4 representable magnitudes.
const FP4_ABS_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Quantized tensor storage using MXFP4 (Microscaling FP4).
///
/// Stores model weights at ~4.25 bits per value, achieving ~3.7x compression
/// over BF16 and ~7.5x over F32.
#[derive(Debug, Clone)]
pub struct Mxfp4Tensor {
    /// FP4 values packed 2 per byte (high nibble = even index, low nibble = odd).
    data: Vec<u8>,
    /// E8M0 shared scales, one per block of 32 elements.
    scales: Vec<u8>,
    /// Original tensor shape.
    shape: Vec<usize>,
}

/// Compute E8M0 scale exponent for a block.
///
/// E8M0 is exponent-only: value = 2^(exponent - 127).
/// We choose scale so max_abs / scale fits in FP4 range [0, 6].
fn compute_e8m0_scale(max_abs: f32) -> u8 {
    if !max_abs.is_finite() || max_abs <= 0.0 {
        return 0;
    }
    let bits = max_abs.to_bits();
    let biased_exp = ((bits >> 23) & 0xFF) as u8;
    if biased_exp == 0 {
        return 0;
    }
    // Scale = 2^(biased_exp - 2 - 127) so max_abs/scale ~ mantissa * 4
    // which fits in FP4 range [0, 6].
    let scale_exp = biased_exp.saturating_sub(2);
    scale_exp.min(254)
}

/// Decode an E8M0 byte to its f32 scale value: `2^(exponent - 127)`.
fn decode_e8m0_scale(e8m0: u8) -> f32 {
    if e8m0 == 0 {
        return f32::from_bits(0x0080_0000); // 2^(-126), smallest normal
    }
    let bits: u32 = u32::from(e8m0) << 23;
    f32::from_bits(bits)
}

/// Quantize a single f32 value to a 4-bit FP4 code given a scale.
fn quantize_to_fp4(val: f32, scale: f32) -> u8 {
    if !val.is_finite() || scale <= 0.0 {
        return 0;
    }
    let sign = val < 0.0;
    let abs_scaled = val.abs() / scale;

    let mut best_idx: usize = 0;
    let mut best_dist = f32::MAX;
    for (i, &fp4_val) in FP4_ABS_VALUES.iter().enumerate() {
        let dist = (abs_scaled - fp4_val).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    let code = best_idx as u8;
    if sign {
        code | 0x08
    } else {
        code
    }
}

impl Mxfp4Tensor {
    /// Quantize an f32 slice to MXFP4 format.
    ///
    /// Pads to a multiple of `MXFP4_BLOCK_SIZE` (32) with zeros if needed.
    pub fn quantize(data: &[f32], shape: &[usize]) -> Self {
        let n = data.len();
        let num_blocks = n.div_ceil(MXFP4_BLOCK_SIZE);
        let padded_len = num_blocks * MXFP4_BLOCK_SIZE;

        let mut scales = Vec::with_capacity(num_blocks);
        let mut packed = Vec::with_capacity(padded_len / 2);

        for block_idx in 0..num_blocks {
            let start = block_idx * MXFP4_BLOCK_SIZE;

            let mut max_abs: f32 = 0.0;
            for i in 0..MXFP4_BLOCK_SIZE {
                let idx = start + i;
                let val = if idx < n { data[idx] } else { 0.0 };
                let abs_val = val.abs();
                if abs_val.is_finite() && abs_val > max_abs {
                    max_abs = abs_val;
                }
            }

            let scale_byte = compute_e8m0_scale(max_abs);
            scales.push(scale_byte);
            let scale = decode_e8m0_scale(scale_byte);

            for pair in 0..(MXFP4_BLOCK_SIZE / 2) {
                let idx_hi = start + pair * 2;
                let idx_lo = start + pair * 2 + 1;
                let val_hi = if idx_hi < n { data[idx_hi] } else { 0.0 };
                let val_lo = if idx_lo < n { data[idx_lo] } else { 0.0 };
                let code_hi = quantize_to_fp4(val_hi, scale);
                let code_lo = quantize_to_fp4(val_lo, scale);
                packed.push((code_hi << 4) | code_lo);
            }
        }

        Self {
            data: packed,
            scales,
            shape: shape.to_vec(),
        }
    }

    /// Dequantize to an f32 Vec of the original size.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let total: usize = self.shape.iter().product();
        let num_blocks = self.scales.len();
        let padded_len = num_blocks * MXFP4_BLOCK_SIZE;
        let mut result = Vec::with_capacity(padded_len);

        for (block_idx, &scale_byte) in self.scales.iter().enumerate() {
            let scale = decode_e8m0_scale(scale_byte);
            let byte_start = block_idx * (MXFP4_BLOCK_SIZE / 2);

            for pair in 0..(MXFP4_BLOCK_SIZE / 2) {
                let packed_byte = self.data[byte_start + pair];
                let code_hi = (packed_byte >> 4) & 0x0F;
                let code_lo = packed_byte & 0x0F;
                result.push(FP4_LUT[code_hi as usize] * scale);
                result.push(FP4_LUT[code_lo as usize] * scale);
            }
        }

        result.truncate(total);
        result
    }

    /// Dequantize to a [`DynTensor`] on the given device.
    pub fn to_dyn_tensor(&self, device: &Device) -> Result<DynTensor> {
        let data = self.dequantize();
        let shape: Vec<usize> = self.shape.clone();
        DynTensor::from_vec(data, shape, device)
    }

    /// Memory usage in bytes (packed data + scales).
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.len() + self.scales.len()
    }

    /// Number of elements in the original tensor.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Original tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Reference to the FP4 lookup table (for verification).
    #[must_use]
    pub fn fp4_lut() -> &'static [f32; 16] {
        &FP4_LUT
    }
}

/// Memory savings report for quantization.
#[derive(Debug, Clone)]
pub struct QuantizationReport {
    /// Total original size in bytes (f32).
    pub original_f32_bytes: usize,
    /// Total original size in bytes (bf16).
    pub original_bf16_bytes: usize,
    /// Total quantized size in bytes.
    pub quantized_bytes: usize,
    /// Number of quantized tensors.
    pub num_quantized_tensors: usize,
    /// Number of full-precision tensors (not quantized).
    pub num_full_precision_tensors: usize,
}

impl QuantizationReport {
    /// Compression ratio vs f32.
    #[must_use]
    pub fn compression_ratio_f32(&self) -> f64 {
        if self.quantized_bytes == 0 {
            return 0.0;
        }
        self.original_f32_bytes as f64 / self.quantized_bytes as f64
    }

    /// Compression ratio vs bf16.
    #[must_use]
    pub fn compression_ratio_bf16(&self) -> f64 {
        if self.quantized_bytes == 0 {
            return 0.0;
        }
        self.original_bf16_bytes as f64 / self.quantized_bytes as f64
    }
}

/// Maximum quantization error for a single value given FP4 rounding.
#[must_use]
pub fn max_fp4_error_for_scale(e8m0_byte: u8) -> f32 {
    let scale = decode_e8m0_scale(e8m0_byte);
    // Largest gap between adjacent FP4 values is 4.0->6.0 (gap=2.0).
    // Max rounding error = gap/2 = 1.0 * scale.
    scale * 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp4_lut_all_16_values() {
        let lut = Mxfp4Tensor::fp4_lut();
        assert_eq!(lut[0], 0.0);
        assert_eq!(lut[1], 0.5);
        assert_eq!(lut[2], 1.0);
        assert_eq!(lut[3], 1.5);
        assert_eq!(lut[4], 2.0);
        assert_eq!(lut[5], 3.0);
        assert_eq!(lut[6], 4.0);
        assert_eq!(lut[7], 6.0);
        assert_eq!(lut[8], -0.0);
        assert_eq!(lut[9], -0.5);
        assert_eq!(lut[10], -1.0);
        assert_eq!(lut[11], -1.5);
        assert_eq!(lut[12], -2.0);
        assert_eq!(lut[13], -3.0);
        assert_eq!(lut[14], -4.0);
        assert_eq!(lut[15], -6.0);
    }

    #[test]
    fn test_roundtrip_known_values() {
        let data = vec![0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0];
        let qt = Mxfp4Tensor::quantize(&data, &[8]);
        let recovered = qt.dequantize();
        assert_eq!(recovered.len(), 8);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 1e-6,
                "Roundtrip mismatch: {orig} -> {rec}"
            );
        }
    }

    #[test]
    fn test_roundtrip_bounded_error() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let qt = Mxfp4Tensor::quantize(&data, &[64]);
        let recovered = qt.dequantize();
        for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let max_err = max_fp4_error_for_scale(qt.scales[i / MXFP4_BLOCK_SIZE]);
            assert!(
                (orig - rec).abs() <= max_err + 1e-6,
                "Error too large at {i}: |{orig} - {rec}| > {max_err}"
            );
        }
    }

    #[test]
    fn test_compression_ratio_vs_f32() {
        let n = 1024;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let qt = Mxfp4Tensor::quantize(&data, &[n]);
        let ratio = (n * 4) as f64 / qt.size_bytes() as f64;
        assert!(
            ratio > 7.0,
            "Compression vs f32 should be >7x, got {ratio:.2}x"
        );
    }

    #[test]
    fn test_compression_ratio_vs_bf16() {
        let n = 1024;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let qt = Mxfp4Tensor::quantize(&data, &[n]);
        let ratio = (n * 2) as f64 / qt.size_bytes() as f64;
        assert!(
            ratio > 3.5,
            "Compression vs bf16 should be >3.5x, got {ratio:.2}x"
        );
    }

    #[test]
    fn test_zeros() {
        let data = vec![0.0f32; 64];
        let qt = Mxfp4Tensor::quantize(&data, &[64]);
        for val in qt.dequantize() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_large_values() {
        let data = vec![1000.0f32, -500.0, 250.0, -125.0];
        let qt = Mxfp4Tensor::quantize(&data, &[4]);
        let recovered = qt.dequantize();
        assert_eq!(recovered.len(), 4);
        for val in &recovered {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_non_multiple_of_block_size() {
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let qt = Mxfp4Tensor::quantize(&data, &[50]);
        let recovered = qt.dequantize();
        assert_eq!(recovered.len(), 50);
        assert_eq!(qt.scales.len(), 2); // ceil(50/32) = 2
    }

    #[test]
    fn test_e8m0_decode_known() {
        assert_eq!(decode_e8m0_scale(127), 1.0);
        assert_eq!(decode_e8m0_scale(128), 2.0);
        assert_eq!(decode_e8m0_scale(126), 0.5);
    }

    #[test]
    fn test_to_dyn_tensor() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let qt = Mxfp4Tensor::quantize(&data, &[2, 2]);
        let tensor = qt
            .to_dyn_tensor(&Device::Cpu)
            .expect("should create DynTensor");
        assert_eq!(tensor.dims(), &[2, 2]);
    }

    #[test]
    fn test_size_bytes_correct() {
        let qt = Mxfp4Tensor::quantize(&vec![1.0f32; 128], &[128]);
        assert_eq!(qt.scales.len(), 4);
        assert_eq!(qt.data.len(), 64);
        assert_eq!(qt.size_bytes(), 68);
    }

    #[test]
    fn test_subnormal_input() {
        let data = vec![f32::MIN_POSITIVE / 100.0; 32];
        let qt = Mxfp4Tensor::quantize(&data, &[32]);
        for val in qt.dequantize() {
            assert!(val.abs() < 1e-30);
        }
    }
}
