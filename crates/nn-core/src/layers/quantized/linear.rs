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

#[cfg(test)]
#[path = "linear_tests.rs"]
mod tests;
