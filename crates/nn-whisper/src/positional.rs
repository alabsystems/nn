// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Positional encoding utilities for Whisper.
//!
//! - Sinusoidal embeddings (encoder, fixed, not learned)
//! - Causal mask (decoder self-attention)

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result};

/// Generate sinusoidal positional embeddings matching AI Provider Whisper layout.
///
/// Returns `[length, channels]` tensor on the given device.
///
/// Layout is **concatenated**: `[sin_0, sin_1, ..., sin_{d/2-1}, cos_0, cos_1, ..., cos_{d/2-1}]`.
/// Frequency formula matches candle-whisper `sinusoids()`:
/// `inv_timescale_i = exp(-i * ln(10000) / (d/2 - 1))`
///
/// Used by the Whisper encoder. Precomputed at model load, not learned.
pub fn sinusoidal_embedding(
    length: usize,
    channels: usize,
    dtype: DType,
    device: &Device,
) -> Result<DynTensor> {
    let half_dim = channels / 2;
    let mut data = vec![0.0f32; length * channels];

    // Match AI Provider Whisper / candle-whisper frequency formula.
    // Denominator is (half_dim - 1), not channels.
    let log_timescale_increment = 10_000.0f32.ln() / (half_dim as f32 - 1.0).max(1.0);

    for pos in 0..length {
        for i in 0..half_dim {
            let inv_timescale = (-(i as f32) * log_timescale_increment).exp();
            let angle = pos as f32 * inv_timescale;
            // Concatenated layout: sin in first half, cos in second half.
            data[pos * channels + i] = angle.sin();
            data[pos * channels + half_dim + i] = angle.cos();
        }
    }

    let t = DynTensor::from_vec(data, &[length, channels], device)?;
    // Convert to model dtype (e.g., BF16) so positional embeddings match
    // model weight dtype in GPU binary ops (#1710).
    if dtype != DType::F32 {
        t.to_dtype(dtype)
    } else {
        Ok(t)
    }
}

/// Pre-compute a causal attention mask: lower-triangular with `NEG_INFINITY`
/// above the diagonal.
///
/// Returns `[max_positions, max_positions]` tensor. The decoder slices this
/// to `[seq_len, seq_len]` per forward call.
///
/// `mask[i][j] = 0.0` if `j <= i` (attend), `NEG_INFINITY` if `j > i` (block).
pub fn causal_mask(max_positions: usize, dtype: DType, device: &Device) -> Result<DynTensor> {
    let mut data = vec![0.0f32; max_positions * max_positions];
    for i in 0..max_positions {
        for j in (i + 1)..max_positions {
            data[i * max_positions + j] = f32::NEG_INFINITY;
        }
    }
    let t = DynTensor::from_vec(data, &[max_positions, max_positions], device)?;
    // Convert to model dtype (e.g., BF16) so mask matches attention weight
    // dtype in GPU binary ops (#1710).
    if dtype != DType::F32 {
        t.to_dtype(dtype)
    } else {
        Ok(t)
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_embedding_shape() {
        let emb = sinusoidal_embedding(10, 64, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(emb.dims(), &[10, 64]);
    }

    #[test]
    fn test_sinusoidal_embedding_first_position_zero() {
        // At position 0, all sin values should be 0, all cos values should be 1.
        // Concatenated layout: [sin_0..sin_{d/2-1}, cos_0..cos_{d/2-1}]
        let channels = 8;
        let half = channels / 2;
        let emb = sinusoidal_embedding(4, channels, DType::F32, &Device::Cpu).unwrap();
        let flat = emb.to_flat_vec::<f32>().unwrap();
        // Row 0: first half is sin(0)=0, second half is cos(0)=1.
        for i in 0..half {
            assert!((flat[i] - 0.0).abs() < 1e-6, "sin at pos 0 should be 0");
            assert!(
                (flat[half + i] - 1.0).abs() < 1e-6,
                "cos at pos 0 should be 1"
            );
        }
    }

    #[test]
    fn test_sinusoidal_embedding_finite() {
        let emb = sinusoidal_embedding(1500, 1280, DType::F32, &Device::Cpu).unwrap();
        let flat = emb.to_flat_vec::<f32>().unwrap();
        for &v in &flat {
            assert!(v.is_finite(), "sinusoidal embedding must be finite");
            assert!((-1.0..=1.0).contains(&v), "sin/cos values in [-1, 1]");
        }
    }

    #[test]
    fn test_causal_mask_shape() {
        let mask = causal_mask(5, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[5, 5]);
    }

    #[test]
    fn test_causal_mask_lower_triangular() {
        let mask = causal_mask(4, DType::F32, &Device::Cpu).unwrap();
        let flat = mask.to_flat_vec::<f32>().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let val = flat[i * 4 + j];
                if j <= i {
                    assert_eq!(val, 0.0, "mask[{i}][{j}] should be 0 (attend)");
                } else {
                    assert_eq!(
                        val,
                        f32::NEG_INFINITY,
                        "mask[{i}][{j}] should be -inf (block)"
                    );
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_size_one() {
        let mask = causal_mask(1, DType::F32, &Device::Cpu).unwrap();
        let flat = mask.to_flat_vec::<f32>().unwrap();
        assert_eq!(flat, vec![0.0]);
    }

    #[test]
    fn test_sinusoidal_odd_channels() {
        // Odd channels: half_dim = 2, so only indices 0..2 get sin, 2..4 get cos.
        // Index 4 (the odd leftover) is left as zero.
        let emb = sinusoidal_embedding(3, 5, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(emb.dims(), &[3, 5]);
        let flat = emb.to_flat_vec::<f32>().unwrap();
        for &v in &flat {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sinusoidal_concatenated_layout() {
        // Verify concatenated layout: first half is sin, second half is cos.
        // Use channels=4, so half_dim=2. At pos=1:
        //   freq_0: inv_timescale = exp(0) = 1.0, angle = 1.0
        //   freq_1: inv_timescale = exp(-ln(10000)/max(1,1)) = 1/10000, angle = 0.0001
        // Row 1 should be: [sin(1.0), sin(0.0001), cos(1.0), cos(0.0001)]
        let channels = 4;
        let half = channels / 2;
        let emb = sinusoidal_embedding(2, channels, DType::F32, &Device::Cpu).unwrap();
        let flat = emb.to_flat_vec::<f32>().unwrap();
        // Row 1 starts at index 4.
        let row1 = &flat[channels..2 * channels];
        // First half: sin values.
        assert!((row1[0] - 1.0f32.sin()).abs() < 1e-5, "sin(1.0) at index 0");
        // Second half: cos values.
        assert!(
            (row1[half] - 1.0f32.cos()).abs() < 1e-5,
            "cos(1.0) at index half_dim"
        );
    }

    #[test]
    fn test_sinusoidal_frequency_formula_matches_candle() {
        // Verify that inv_timescale formula matches candle-whisper:
        // inv_timescale_i = exp(-i * ln(10000) / (half_dim - 1))
        // For channels=8, half_dim=4, denominator = 3.
        let channels = 8;
        let half_dim = channels / 2;
        let emb = sinusoidal_embedding(3, channels, DType::F32, &Device::Cpu).unwrap();
        let flat = emb.to_flat_vec::<f32>().unwrap();

        let log_increment = 10_000.0f32.ln() / (half_dim as f32 - 1.0);
        for pos in 0..3 {
            for i in 0..half_dim {
                let inv_ts = (-(i as f32) * log_increment).exp();
                let angle = pos as f32 * inv_ts;
                let sin_idx = pos * channels + i;
                let cos_idx = pos * channels + half_dim + i;
                assert!(
                    (flat[sin_idx] - angle.sin()).abs() < 1e-6,
                    "sin mismatch at pos={pos}, i={i}"
                );
                assert!(
                    (flat[cos_idx] - angle.cos()).abs() < 1e-6,
                    "cos mismatch at pos={pos}, i={i}"
                );
            }
        }
    }
}

#[cfg(kani)]
#[path = "kani_positional_proofs.rs"]
mod kani_positional_proofs;
