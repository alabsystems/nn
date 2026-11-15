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
