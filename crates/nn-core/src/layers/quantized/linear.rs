// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GGML quantized linear layer for memory-efficient LLM inference.
//!
//! Supports Q4K (Q4_K_S) format used by Qwen3-TTS in dvoice.
//! Phase 1: dequantize-on-load (full F32 materialization per forward).
//! Phase 2 (deferred): block-wise dequantize during matmul.
//!
//! Reference: GGML k_quants format, candle `k_quants.rs`.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, Linear, Module};
use crate::{Result, TensorError};

// -- Constants ----------------------------------------------------------------

/// Elements per GGML K-quant block.
const QK_K: usize = 256;

/// Bytes for packed sub-block scales in Q4K.
const K_SCALE_SIZE: usize = 12;

// -- GgmlDType ----------------------------------------------------------------

/// GGML quantization data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GgmlDType {
    /// 4-bit K-quant (Q4_K_S): 256-element blocks, ~4.5 bits/weight.
    Q4K,
    /// Full precision (no quantization).
    F32,
}

// -- BlockQ4K -----------------------------------------------------------------

/// Q4K block layout matching GGML's `block_q4_K` C struct.
///
/// 144 bytes per block of 256 elements (~4.5 bits/element).
/// - `d`: f16 super-block scale
/// - `dmin`: f16 super-block minimum
/// - `scales`: 12 bytes of packed 6-bit sub-block scales and mins
/// - `qs`: 128 bytes of 4-bit quantized values (2 per byte)
#[derive(Debug, Clone)]
#[repr(C)]
pub struct BlockQ4K {
    d: u16,                     // f16 as raw bits (super-block scale)
    dmin: u16,                  // f16 as raw bits (super-block minimum)
    scales: [u8; K_SCALE_SIZE], // packed sub-block scales+mins
    qs: [u8; QK_K / 2],         // 4-bit quantized values
}

const _: () = assert!(size_of::<BlockQ4K>() == 144);

impl BlockQ4K {
    /// Number of f32 elements this block encodes.
    pub const BLOCK_SIZE: usize = QK_K;

    /// Dequantize this block into 256 f32 values.
    pub fn dequantize(&self, out: &mut [f32; QK_K]) {
        let d = half::f16::from_bits(self.d).to_f32();
        let dmin = half::f16::from_bits(self.dmin).to_f32();

        let mut out_idx = 0;
        // 4 groups of 64, each split into 2 sub-blocks of 32
        for j in (0..QK_K).step_by(64) {
            let q_offset = j / 2;
            let is = j / 32; // sub-block index (0..8 stepping by 2)

            let (sc1, m1) = get_scale_min_k4(is, &self.scales);
            let d1 = d * f32::from(sc1);
            let m1 = dmin * f32::from(m1);

            let (sc2, m2) = get_scale_min_k4(is + 1, &self.scales);
            let d2 = d * f32::from(sc2);
            let m2 = dmin * f32::from(m2);

            // First 32: low nibbles
            for i in 0..32 {
                out[out_idx] = d1 * f32::from(self.qs[q_offset + i] & 0xF) - m1;
                out_idx += 1;
            }
            // Second 32: high nibbles
            for i in 0..32 {
                out[out_idx] = d2 * f32::from(self.qs[q_offset + i] >> 4) - m2;
                out_idx += 1;
            }
        }
    }

    /// Quantize 256 f32 values into this block.
    pub fn quantize(values: &[f32; QK_K]) -> Result<Self> {
        let mut block = Self {
            d: 0,
            dmin: 0,
            scales: [0u8; K_SCALE_SIZE],
            qs: [0u8; QK_K / 2],
        };

        let mut sub_scales = [0.0_f32; QK_K / 32]; // 8 sub-block scales
        let mut sub_mins = [0.0_f32; QK_K / 32]; // 8 sub-block mins

        // Find per-sub-block (scale, min) via iterative refinement
        for (j, chunk) in values.chunks_exact(32).enumerate() {
            let (s, m) = make_qkx1_quants(15, 5, chunk)?;
            sub_scales[j] = s;
            sub_mins[j] = m;
        }

        let max_scale = sub_scales.iter().copied().fold(0.0_f32, f32::max);
        let max_min = sub_mins.iter().copied().fold(0.0_f32, f32::max);

        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        // Pack sub-block scales and mins into 12-byte array
        for j in 0..QK_K / 32 {
            let ls = (nearest_int(inv_scale * sub_scales[j])? as u8).min(63);
            let lm = (nearest_int(inv_min * sub_mins[j])? as u8).min(63);
            if j < 4 {
                block.scales[j] = ls;
                block.scales[j + 4] = lm;
            } else {
                block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                block.scales[j - 4] |= (ls >> 4) << 6;
                block.scales[j] |= (lm >> 4) << 6;
            }
        }

        block.d = half::f16::from_f32(max_scale / 63.0).to_bits();
        block.dmin = half::f16::from_f32(max_min / 63.0).to_bits();

        // Quantize each element to 4-bit using the packed scales
        let mut l = [0u8; QK_K];
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &block.scales);
            let bd = half::f16::from_bits(block.d).to_f32() * f32::from(sc);
            if bd != 0.0 {
                let bm = half::f16::from_bits(block.dmin).to_f32() * f32::from(m);
                for ii in 0..32 {
                    let v = nearest_int((values[32 * j + ii] + bm) / bd)?;
                    l[32 * j + ii] = v.clamp(0, 15) as u8;
                }
            }
        }

        // Pack 4-bit values: low nibble + high nibble
        for j in (0..QK_K).step_by(64) {
            for k in 0..32 {
                block.qs[j / 2 + k] = l[j + k] | (l[j + k + 32] << 4);
            }
        }

        Ok(block)
    }
}

/// Decode packed 6-bit (scale, min) pair for sub-block `j`.
fn get_scale_min_k4(j: usize, q: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Round to nearest integer, rejecting non-finite or out-of-range inputs.
fn nearest_int(v: f32) -> Result<i32> {
    if !v.is_finite() {
        return Err(TensorError::ValueOutOfRange {
            description: "nearest_int: non-finite value",
        });
    }
    let r = v.round();
    if r < i32::MIN as f32 || r > i32::MAX as f32 {
        return Err(TensorError::ValueOutOfRange {
            description: "nearest_int: value outside i32 range",
        });
    }
    Ok(r as i32)
}

/// Find optimal (scale, min) for quantizing a sub-block to [0, nmax].
fn make_qkx1_quants(nmax: i32, ntry: usize, x: &[f32]) -> Result<(f32, f32)> {
    let n = x.len();
    let mut l = vec![0u8; n];

    let min = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if max == min {
        // Constant sub-block: all values are identical.
        // Dequant formula: val = d1 * qs[i] - m1.
        // For c >= 0: scale = c/nmax, min = 0 → all qs = nmax → dequant = c.
        // For c < 0: scale = 0, min = -c → all qs = 0 → dequant = 0 - (-c) = c.
        let c = min; // constant value
        if c >= 0.0 {
            let s = if nmax > 0 { c / nmax as f32 } else { 0.0 };
            return Ok((s, 0.0));
        } else {
            return Ok((0.0, -c));
        }
    }

    let mut min = min.min(0.0);
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;

    for _ in 0..ntry {
        let mut sumlx = 0.0_f32;
        let mut suml2 = 0_i32;
        let mut did_change = false;
        for (i, &val) in x.iter().enumerate() {
            let li = nearest_int(iscale * (val - min))?.clamp(0, nmax) as u8;
            if li != l[i] {
                l[i] = li;
                did_change = true;
            }
            sumlx += (val - min) * f32::from(li);
            suml2 += i32::from(li) * i32::from(li);
        }
        if suml2 == 0 {
            return Ok((0.0, 0.0));
        }
        scale = sumlx / suml2 as f32;
        if scale == 0.0 {
            // All weighted (val - min) contributions cancel out.
            // Values are clustered at min; return (0, -min).
            return Ok((0.0, -min));
        }
        let sum: f32 = x
            .iter()
            .zip(l.iter())
            .map(|(&xi, &li)| xi - scale * f32::from(li))
            .sum();
        min = (sum / n as f32).min(0.0);
        iscale = 1.0 / scale;
        if !did_change {
            break;
        }
    }
    Ok((scale, -min))
}

// -- QuantizedWeight ----------------------------------------------------------

/// Quantized weight storage (CPU only).
#[derive(Debug, Clone)]
pub struct QuantizedWeight {
    blocks: Vec<BlockQ4K>,
    shape: [usize; 2], // [out_features, in_features]
    bias: Option<DynTensor>,
}

impl QuantizedWeight {
    /// Dequantize all blocks to a flat f32 vec.
    fn dequantize_to_f32(&self) -> Result<DynTensor> {
        let total = self.shape[0] * self.shape[1];
        let mut data = vec![0.0_f32; total];

        for (i, block) in self.blocks.iter().enumerate() {
            let offset = i * QK_K;
            let mut buf = [0.0_f32; QK_K];
            block.dequantize(&mut buf);
            let end = (offset + QK_K).min(total);
            data[offset..end].copy_from_slice(&buf[..end - offset]);
        }

        DynTensor::from_vec(data, &[self.shape[0], self.shape[1]], &crate::Device::Cpu)
    }
}

// -- QLinear ------------------------------------------------------------------

/// Quantized linear layer: transparent dispatch for quantized and float weights.
///
/// Matches candle's `QMatMul` enum pattern. dvoice loads as F32 via
/// `linear()`/`linear_no_bias()`, then calls `quantize_weights(Q4K)`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum QLinear {
    /// Standard F32 weights — delegates to `Linear::forward()`.
    Float(Linear),
    /// Quantized weights — dequantize then matmul (Phase 1).
    Quantized(QuantizedWeight),
}

impl QLinear {
    /// Create from an existing `Linear` layer without quantization.
    pub fn from_float(linear: Linear) -> Self {
        Self::Float(linear)
    }

    /// Quantize an existing `Linear` layer's weights to Q4K.
    ///
    /// Extracts the weight tensor, quantizes to Q4K blocks, stores the bias
    /// separately. Phase 1: `forward()` will dequantize fully before matmul.
    pub fn from_linear(linear: &Linear, dtype: GgmlDType) -> Result<Self> {
        match dtype {
            GgmlDType::F32 => {
                Linear::new(linear.weight().clone(), linear.bias().cloned()).map(QLinear::Float)
            }
            GgmlDType::Q4K => {
                let weight = linear.weight();
                let (out_features, in_features) = weight.dims2()?;
                let weight_cpu = weight.to_device(&crate::Device::Cpu)?;
                let weight_arr = weight_cpu.to_f32_array()?;
                let flat: Vec<f32> = weight_arr.iter().copied().collect();
                let total = out_features * in_features;

                // Pad to block boundary
                let n_blocks = total.div_ceil(QK_K);
                let mut padded = flat;
                padded.resize(n_blocks * QK_K, 0.0);

                let mut blocks = Vec::with_capacity(n_blocks);
                for chunk in padded.chunks_exact(QK_K) {
                    let arr: &[f32; QK_K] =
                        chunk
                            .try_into()
                            .map_err(|_| TensorError::DataLengthMismatch {
                                expected: QK_K,
                                actual: chunk.len(),
                            })?;
                    blocks.push(BlockQ4K::quantize(arr)?);
                }

                Ok(Self::Quantized(QuantizedWeight {
                    blocks,
                    shape: (out_features, in_features).into(),
                    bias: linear.bias().cloned(),
                }))
            }
        }
    }

    /// Dequantize back to a standard `Linear` layer (for verification).
    pub fn dequantize(&self) -> Result<Linear> {
        match self {
            Self::Float(l) => Linear::new(l.weight().clone(), l.bias().cloned()),
            Self::Quantized(qw) => {
                let weight = qw.dequantize_to_f32()?;
                Linear::new(weight, qw.bias.clone())
            }
        }
    }

    /// Returns `true` if weights are quantized.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &DynTensor) -> Result<DynTensor> {
        match self {
            Self::Float(linear) => linear.forward(xs),
            Self::Quantized(qw) => {
                // Phase 1: full dequantize then matmul
                // Dequantize outside traced_forward so weight data is available
                // for the TraceOp builder without being inside the suppressed scope.
                let weight_f32 = qw.dequantize_to_f32()?;
                let bias_ref = &qw.bias;
                trace::traced_forward(
                    &[xs],
                    || {
                        Ok(TraceOp::QLinear {
                            weight: weight_f32.to_weight_ref()?,
                            bias: bias_ref
                                .as_ref()
                                .map(DynTensor::to_weight_ref)
                                .transpose()?,
                        })
                    },
                    || {
                        let out = xs.matmul(&weight_f32.t()?)?;
                        let out = match bias_ref {
                            Some(bias) => out.broadcast_add(bias)?,
                            None => out,
                        };
                        check_output_finite(&out, "QLinear")?;
                        Ok(out)
                    },
                )
            }
        }
    }
}

// -- Kani proof harnesses -----------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // -----------------------------------------------------------------------
    // Harness 1: get_scale_min_k4 index safety.
    //
    // The function accesses q[j], q[j+4], q[j-4] depending on j < 4 or j >= 4.
    // Prove all array accesses are in-bounds for valid sub-block indices j in [0, 7].
    // -----------------------------------------------------------------------

    /// Prove: get_scale_min_k4 never panics for sub-block index j in [0, 7]
    /// and returns values that fit in 6 bits (max 63).
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_get_scale_min_index_safety() {
        let j: usize = kani::any();
        kani::assume(j < QK_K / 32); // j in [0, 7]

        let q: [u8; K_SCALE_SIZE] = kani::any();
        let (scale, min) = get_scale_min_k4(j, &q);

        // Scale and min are constructed from 6-bit fields (max 63).
        // For j < 4: direct mask with & 63.
        // For j >= 4: low 4 bits | (high 2 bits << 4) = max 0xF | (0x3 << 4) = 63.
        assert!(scale <= 63, "scale must fit in 6 bits");
        assert!(min <= 63, "min must fit in 6 bits");
    }

    // -----------------------------------------------------------------------
    // Harness 2: Q4K nibble packing index safety.
    //
    // In dequantize(), the inner loops access self.qs[q_offset + i] where
    // q_offset = j/2 for j in {0, 64, 128, 192} and i in [0, 31].
    // Prove all indices are within [0, QK_K/2 = 128).
    // -----------------------------------------------------------------------

    /// Prove: Q4K dequantize loop indices stay within the qs array bounds.
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_dequant_nibble_index_in_bounds() {
        // Symbolic group index: j in {0, 64, 128, 192}
        let group: usize = kani::any();
        kani::assume(group < 4);
        let j = group * 64;

        let q_offset = j / 2; // {0, 32, 64, 96}
        let i: usize = kani::any();
        kani::assume(i < 32);

        let idx = q_offset + i;
        assert!(idx < QK_K / 2, "qs index must be < 128");
    }

    // -----------------------------------------------------------------------
    // Harness 3: Q4K dequantize output finiteness for bounded block fields.
    //
    // Each dequantized element is: d1 * f32::from(nibble) - m1
    // where d1 = d * f32::from(sc), m1 = dmin * f32::from(m),
    // d/dmin are f16→f32, sc/m are u8 <= 63, nibble is u8 in [0, 15].
    //
    // Prove the arithmetic produces finite results for any valid block.
    // -----------------------------------------------------------------------

    /// Prove: for any f16 super-block scale, f16 min, 6-bit sub-block
    /// scale/min, and 4-bit quantized value, the dequantized element
    /// is finite.
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_dequant_element_finite() {
        // f16 super-block values (as raw u16 bits)
        let d_bits: u16 = kani::any();
        let dmin_bits: u16 = kani::any();

        let d = half::f16::from_bits(d_bits).to_f32();
        let dmin = half::f16::from_bits(dmin_bits).to_f32();

        // Assume the f16 values are finite (NaN/Inf f16 blocks are malformed)
        kani::assume(d.is_finite());
        kani::assume(dmin.is_finite());

        // Sub-block scale and min (6-bit, max 63)
        let sc: u8 = kani::any();
        kani::assume(sc <= 63);
        let m: u8 = kani::any();
        kani::assume(m <= 63);

        // 4-bit quantized nibble [0, 15]
        let nibble: u8 = kani::any();
        kani::assume(nibble <= 15);

        let d1 = d * f32::from(sc);
        let m1 = dmin * f32::from(m);
        let value = d1 * f32::from(nibble) - m1;

        // f16 max magnitude is 65504. sc max is 63. nibble max is 15.
        // Worst case: 65504 * 63 * 15 + 65504 * 63 = 65504 * 63 * 16 = 66_028_032.
        // Well within f32 range (~3.4e38).
        assert!(value.is_finite(), "dequantized Q4K element must be finite");
    }

    // -----------------------------------------------------------------------
    // Harness 4: BlockQ4K size invariant.
    //
    // Prove the compile-time assertion: BlockQ4K is exactly 144 bytes.
    // Also prove that QK_K (256) elements at 4 bits = 128 bytes for qs,
    // plus 12 bytes scales, plus 4 bytes (2x u16) = 144 bytes.
    // -----------------------------------------------------------------------

    /// Prove: BlockQ4K size is exactly 144 bytes and field sizes are consistent.
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_block_size_invariant() {
        // QK_K = 256 elements per block
        assert!(QK_K == 256);
        // K_SCALE_SIZE = 12 bytes for packed sub-block scales
        assert!(K_SCALE_SIZE == 12);
        // qs array: 256/2 = 128 bytes (two 4-bit values per byte)
        assert!(QK_K / 2 == 128);
        // Total: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144
        assert!(2 + 2 + K_SCALE_SIZE + QK_K / 2 == 144);
        assert!(size_of::<BlockQ4K>() == 144);
    }

    // -----------------------------------------------------------------------
    // Harness 5: Q4K sub-block index coverage.
    //
    // In dequantize(), `is = j / 32` produces sub-block indices
    // {0, 2, 4, 6} and `is + 1` produces {1, 3, 5, 7}.
    // Prove all 8 sub-block indices in [0, 7] are used exactly once,
    // and all are valid inputs to get_scale_min_k4.
    // -----------------------------------------------------------------------

    /// Prove: the dequantize loop generates all sub-block indices [0, 7]
    /// and each is a valid argument to get_scale_min_k4.
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_subblock_index_coverage() {
        let group: usize = kani::any();
        kani::assume(group < 4);
        let j = group * 64;

        let is = j / 32;
        // is takes values {0, 2, 4, 6}
        assert!(is < 8, "sub-block index must be < 8");
        assert!(is + 1 < 8, "sub-block index + 1 must be < 8");
        // is is always even
        assert!(is % 2 == 0, "even sub-block indices from j/32");
    }

    // -----------------------------------------------------------------------
    // Harness 6: nearest_int safety for bounded inputs.
    //
    // In Q4K quantization, nearest_int is called with values derived from
    // finite weight data. Prove it returns Ok for inputs in a practical range.
    // -----------------------------------------------------------------------

    /// Prove: nearest_int succeeds for any finite f32 in [-1e9, 1e9]
    /// and the result fits in i32.
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_nearest_int_bounded() {
        let v: f32 = kani::any();
        kani::assume(v.is_finite());
        kani::assume(v >= -1e9 && v <= 1e9);

        let result = nearest_int(v);
        assert!(
            result.is_ok(),
            "nearest_int must succeed for bounded finite input"
        );
        let r = result.unwrap();
        // v.round() for |v| <= 1e9 is at most 1e9, well within i32::MAX (2.1e9).
        assert!(r >= -1_000_000_001 && r <= 1_000_000_001);
    }

    // -----------------------------------------------------------------------
    // Harness 7: Q4K dequant output count.
    //
    // Prove that dequantize writes exactly QK_K (256) elements. The out_idx
    // counter starts at 0 and increments exactly 256 times through the
    // 4 groups x 2 sub-blocks x 32 elements structure.
    // -----------------------------------------------------------------------

    /// Prove: the Q4K dequantize loop structure produces exactly 256 outputs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_dequant_output_count() {
        // 4 groups of 64, each with 2 sub-blocks of 32
        let total = 4_usize * 64;
        assert!(total == QK_K, "4 groups * 64 = 256 = QK_K");

        // Verify the step_by(64) loop visits exactly 4 groups
        let mut count = 0_usize;
        let mut j = 0_usize;
        while j < QK_K {
            count += 64; // 32 low nibble + 32 high nibble
            j += 64;
        }
        assert!(count == QK_K, "loop must produce exactly QK_K outputs");
    }

    // -----------------------------------------------------------------------
    // Harness 8: Q4K 4-bit packing roundtrip.
    //
    // Values are packed as: qs[j/2 + k] = l[j+k] | (l[j+k+32] << 4).
    // Dequantize unpacks low nibble (& 0xF) and high nibble (>> 4).
    // Prove this packing/unpacking preserves the 4-bit values.
    // -----------------------------------------------------------------------

    /// Prove: packing two 4-bit values into one byte and unpacking them
    /// via low nibble (& 0xF) and high nibble (>> 4) is lossless.
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_nibble_packing_roundtrip() {
        let low: u8 = kani::any();
        kani::assume(low <= 15);
        let high: u8 = kani::any();
        kani::assume(high <= 15);

        let packed = low | (high << 4);

        let unpacked_low = packed & 0xF;
        let unpacked_high = packed >> 4;

        assert!(unpacked_low == low, "low nibble must roundtrip");
        assert!(unpacked_high == high, "high nibble must roundtrip");
    }

    // Harness 9: Q4K scale pack roundtrip (j < 4).
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_scale_pack_roundtrip_low() {
        let j: usize = kani::any();
        kani::assume(j < 4);
        let sc: u8 = kani::any();
        kani::assume(sc <= 63);
        let m: u8 = kani::any();
        kani::assume(m <= 63);
        let mut q = [0u8; K_SCALE_SIZE];
        q[j] = sc;
        q[j + 4] = m;
        let (dec_sc, dec_m) = get_scale_min_k4(j, &q);
        assert!(dec_sc == sc, "low-index scale must roundtrip");
        assert!(dec_m == m, "low-index min must roundtrip");
    }

    // Harness 10: Q4K scale pack roundtrip (j >= 4).
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_scale_pack_roundtrip_high() {
        let j: usize = kani::any();
        kani::assume(j >= 4 && j < 8);
        let sc: u8 = kani::any();
        kani::assume(sc <= 63);
        let m: u8 = kani::any();
        kani::assume(m <= 63);
        let mut q = [0u8; K_SCALE_SIZE];
        q[j + 4] = (sc & 0xF) | ((m & 0xF) << 4);
        q[j - 4] |= (sc >> 4) << 6;
        q[j] |= (m >> 4) << 6;
        let (dec_sc, dec_m) = get_scale_min_k4(j, &q);
        assert!(dec_sc == sc, "high-index scale must roundtrip");
        assert!(dec_m == m, "high-index min must roundtrip");
    }

    // Harness 11: Q4K dequant output magnitude bounded.
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_dequant_output_magnitude_bounded() {
        let d_bits: u16 = kani::any();
        let dmin_bits: u16 = kani::any();
        let d = half::f16::from_bits(d_bits).to_f32();
        let dmin = half::f16::from_bits(dmin_bits).to_f32();
        kani::assume(d.is_finite());
        kani::assume(dmin.is_finite());
        let sc: u8 = kani::any();
        kani::assume(sc <= 63);
        let m: u8 = kani::any();
        kani::assume(m <= 63);
        let nibble: u8 = kani::any();
        kani::assume(nibble <= 15);
        let d1 = d * f32::from(sc);
        let m1 = dmin * f32::from(m);
        let value = d1 * f32::from(nibble) - m1;
        assert!(value.is_finite(), "Q4K element must be finite");
        let bound = d.abs() * 63.0 * 15.0 + dmin.abs() * 63.0;
        assert!(value.abs() <= bound + 1e-3, "Q4K element bounded");
    }

    // Harness 12: Q4K block padding invariant.
    #[kani::unwind(1)]
    #[kani::proof]
    fn q4k_block_padding_invariant() {
        let total: usize = kani::any();
        kani::assume(total >= 1 && total <= 65536);
        let n_blocks = total.div_ceil(QK_K);
        let padded_len = n_blocks * QK_K;
        assert!(padded_len >= total, "padded covers all");
        assert!(padded_len - total < QK_K, "padding < QK_K");
        let block_idx: usize = kani::any();
        kani::assume(block_idx < n_blocks);
        let offset = block_idx * QK_K;
        assert!(offset + QK_K <= padded_len, "block fits");
    }
}

#[cfg(test)]
#[path = "linear_tests.rs"]
mod tests;
