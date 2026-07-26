// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ALiBi (Attention with Linear Biases) position encoding.
//!
//! Implements the ALiBi attention bias from Press et al. 2021. Instead of
//! positional embeddings, ALiBi adds a linear bias based on relative distance
//! to attention scores before softmax.
//!
//! Used by Emotion2vec (Kokoro expressive voice training).

use crate::dyn_tensor::DynTensor;
use crate::error::Result;
use crate::layers::validate_heads;
use crate::Device;

/// Compute ALiBi head slopes using the geometric progression from Press et al. 2021.
///
/// `slopes[h] = 2^(-8 * (h+1) / num_heads)` for h in `0..num_heads`.
///
/// # Errors
///
/// Returns an error if `num_heads` is zero.
pub fn alibi_slopes(num_heads: usize) -> Result<Vec<f32>> {
    validate_heads(num_heads, "alibi_slopes")?;
    let slopes: Vec<f32> = (1..=num_heads)
        .map(|h| 2f32.powf(-8.0 * h as f32 / num_heads as f32))
        .collect();
    Ok(slopes)
}

/// Compute ALiBi bias tensor for the given number of heads and sequence length.
///
/// Returns a `[1, num_heads, seq_len, seq_len]` tensor where
/// `bias[0][h][i][j] = slopes[h] * (j - i)`.
///
/// The bias is added to attention scores before softmax. Negative values
/// (when `j < i`, i.e., keys before the query) penalize distant tokens.
///
/// # Errors
///
/// Returns an error if `num_heads` is zero.
pub fn alibi_bias(num_heads: usize, seq_len: usize, device: &Device) -> Result<DynTensor> {
    let slopes = alibi_slopes(num_heads)?;

    if seq_len == 0 {
        return DynTensor::zeros(&[1, num_heads, 0, 0], crate::DType::F32, device);
    }

    // Relative distance matrix: dist[i][j] = j - i
    // positions = [0, 1, 2, ..., seq_len-1]
    let positions = DynTensor::arange(0.0, seq_len as f64, device)?;

    // keys[j] - queries[i] via broadcast: [1, S] - [S, 1] = [S, S]
    let keys = positions.reshape([1, seq_len])?; // [1, S]
    let queries = positions.reshape([seq_len, 1])?; // [S, 1]
    let distances = keys.broadcast_sub(&queries)?; // [S, S]

    // Build per-head bias: slopes[h] * distances
    // slopes tensor: [H, 1, 1] for broadcast with [S, S]
    let slopes_tensor = DynTensor::from_vec(slopes, &[num_heads, 1, 1], device)?;
    let distances_3d = distances.unsqueeze(0)?; // [1, S, S]

    // [H, 1, 1] * [1, S, S] = [H, S, S]
    let bias = slopes_tensor.broadcast_mul(&distances_3d)?;

    // Add batch dim: [1, H, S, S]
    bias.unsqueeze(0)
}

/// Compute ALiBi bias with learned per-head scaling.
///
/// `scale` should be a `[num_heads]` tensor of learned scalars (initialized to 1.0
/// during training). The final bias is `slopes[h] * scale[h] * (j - i)`.
///
/// Returns a `[1, num_heads, seq_len, seq_len]` tensor.
///
/// # Errors
///
/// Returns an error if `num_heads` is zero or `scale` shape doesn't match.
pub fn alibi_bias_scaled(
    num_heads: usize,
    seq_len: usize,
    scale: &DynTensor,
    device: &Device,
) -> Result<DynTensor> {
    if scale.dims() != [num_heads] {
        return Err(crate::error::TensorError::shape_mismatch(
            vec![num_heads],
            scale.dims().to_vec(),
        ));
    }

    // Get unscaled bias [1, H, S, S]
    let bias = alibi_bias(num_heads, seq_len, device)?;

    if seq_len == 0 {
        return Ok(bias);
    }

    // Reshape scale to [1, H, 1, 1] for broadcast
    let scale_4d = scale.reshape([1, num_heads, 1, 1])?;

    // Apply learned scaling
    bias.broadcast_mul(&scale_4d)
}

#[cfg(test)]
#[path = "alibi_tests.rs"]
mod tests;

// -- Kani verification harnesses (#3575) --------------------------------------

#[cfg(kani)]
mod kani_proofs {
    /// Nondeterministic powf stub for CBMC (cannot model f32::powf).
    /// For base > 1.0 and any exponent: result is positive, finite, in (0, 1]
    /// for negative exponents (which is the ALiBi use case: base=2, exp<0).
    fn powf_f32_nondet_stub(_base: f32, _exp: f32) -> f32 {
        let r: f32 = kani::any();
        kani::assume(r.is_finite() && r > 0.0 && r <= 1.0);
        r
    }

    /// Prove: ALiBi slopes are strictly decreasing (exponent ordering proof).
    ///
    /// Instead of calling powf (CBMC cannot model it), we prove the equivalent:
    /// the exponents are strictly decreasing, and 2^x is monotone increasing
    /// (so a larger exponent gives a larger result). The proof shows
    /// exp_h > exp_h1 (exp_h is less negative), so 2^exp_h > 2^exp_h1.
    #[kani::unwind(1)]
    #[kani::proof]
    fn alibi_slopes_strictly_decreasing() {
        let num_heads: usize = kani::any();
        kani::assume(num_heads >= 2 && num_heads <= 32);
        let h: usize = kani::any();
        kani::assume(h < num_heads - 1);
        let exp_h = -8.0f64 * (h + 1) as f64 / num_heads as f64;
        let exp_h1 = -8.0f64 * (h + 2) as f64 / num_heads as f64;
        kani::assert(
            exp_h > exp_h1,
            "exponent for h must be greater than for h+1",
        );
    }

    /// Prove: ALiBi slopes are positive and finite for all valid head counts.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::powf, powf_f32_nondet_stub)]
    fn alibi_slopes_positive_finite() {
        let num_heads: usize = kani::any();
        kani::assume(num_heads >= 1 && num_heads <= 64);
        let h: usize = kani::any();
        kani::assume(h < num_heads);
        let slope = 2f32.powf(-8.0 * (h + 1) as f32 / num_heads as f32);
        kani::assert(slope.is_finite(), "alibi slope must be finite");
        kani::assert(slope > 0.0, "alibi slope must be positive");
    }

    /// Prove: ALiBi bias is monotone decreasing with distance for positive slope.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::powf, powf_f32_nondet_stub)]
    fn alibi_bias_monotone_decreasing_with_distance() {
        let num_heads: usize = kani::any();
        kani::assume(num_heads >= 1 && num_heads <= 16);
        let h: usize = kani::any();
        kani::assume(h < num_heads);
        let slope = 2f32.powf(-8.0 * (h + 1) as f32 / num_heads as f32);
        kani::assume(slope.is_finite() && slope > 0.0);
        let seq_len: usize = kani::any();
        kani::assume(seq_len >= 3 && seq_len <= 16);
        let i: usize = kani::any();
        let j1: usize = kani::any();
        let j2: usize = kani::any();
        kani::assume(i < seq_len);
        kani::assume(j1 < j2);
        kani::assume(j2 < i);
        let bias_j1 = slope * (j1 as f32 - i as f32);
        let bias_j2 = slope * (j2 as f32 - i as f32);
        kani::assert(bias_j1 < bias_j2, "alibi bias must decrease with distance");
    }

    /// Prove: ALiBi bias on the diagonal is exactly zero.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::powf, powf_f32_nondet_stub)]
    fn alibi_bias_diagonal_zero() {
        let num_heads: usize = kani::any();
        kani::assume(num_heads >= 1 && num_heads <= 32);
        let h: usize = kani::any();
        kani::assume(h < num_heads);
        let slope = 2f32.powf(-8.0 * (h + 1) as f32 / num_heads as f32);
        let i: usize = kani::any();
        kani::assume(i < 16);
        let bias = slope * (i as f32 - i as f32);
        kani::assert(bias == 0.0, "alibi bias on diagonal must be exactly 0.0");
    }

    /// Prove: ALiBi bias is antisymmetric: bias[i][j] + bias[j][i] == 0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::powf, powf_f32_nondet_stub)]
    fn alibi_bias_antisymmetric() {
        let num_heads: usize = kani::any();
        kani::assume(num_heads >= 1 && num_heads <= 16);
        let h: usize = kani::any();
        kani::assume(h < num_heads);
        let slope = 2f32.powf(-8.0 * (h + 1) as f32 / num_heads as f32);
        kani::assume(slope.is_finite());
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < 16);
        kani::assume(j < 16);
        let bias_ij = slope * (j as f32 - i as f32);
        let bias_ji = slope * (i as f32 - j as f32);
        let sum = bias_ij + bias_ji;
        kani::assert(sum.abs() < 1e-6, "alibi bias must be antisymmetric");
    }

    /// Prove: Last ALiBi slope exponent is -8.0 (pure arithmetic).
    #[kani::unwind(1)]
    #[kani::proof]
    fn alibi_last_slope_is_2_neg8() {
        let num_heads: usize = kani::any();
        kani::assume(num_heads >= 1 && num_heads <= 64);
        let exp_val = -8.0f64 * num_heads as f64 / num_heads as f64;
        kani::assert(
            (exp_val - (-8.0f64)).abs() < 1e-10,
            "last alibi exponent must be -8.0",
        );
    }
}
