// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 2D Rotary Position Embedding and sinusoidal 2D positional encoding.
//!
//! [`RotaryEmbedding2d`] applies RoPE independently to height and width
//! frequency bands, splitting the embedding dimension in half. Required by
//! Qwen2-VL and modern vision transformers that operate on 2D spatial grids.
//!
//! [`sinusoidal_2d`] generates fixed sin/cos position encodings on a 2D spatial
//! grid, matching the positional encoding used in DETR and TableFormer.

use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

/// 2D Rotary Position Embedding for vision transformers.
///
/// Splits the head dimension in half: the first half encodes height positions,
/// the second half encodes width positions. Each half uses standard 1D RoPE
/// frequencies.
///
/// Input shape: `[batch, seq_len, head_dim]` where `seq_len = H * W`.
/// Apply with `(h_positions, w_positions)` arrays of length `seq_len`.
///
/// Reference: Qwen2-VL (Wang et al., 2024) uses this for spatial attention.
#[derive(Clone)]
pub struct RotaryEmbedding2d {
    /// Precomputed cos values: `[max_pos, half_dim/2]`
    cos_cache: DynTensor,
    /// Precomputed sin values: `[max_pos, half_dim/2]`
    sin_cache: DynTensor,
    /// Full head dimension (must be divisible by 4).
    head_dim: usize,
    /// Maximum position value (applies to both H and W).
    max_position: usize,
}

impl std::fmt::Debug for RotaryEmbedding2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotaryEmbedding2d")
            .field("head_dim", &self.head_dim)
            .field("max_position", &self.max_position)
            .finish_non_exhaustive()
    }
}

impl RotaryEmbedding2d {
    /// Create a new 2D rotary embedding.
    ///
    /// - `head_dim`: dimension per attention head (must be divisible by 4 —
    ///   half for height, half for width, each half further split into pairs)
    /// - `max_position`: maximum spatial position in either dimension
    /// - `base`: frequency base (typically 10000.0)
    /// - `device`: target device for the cached sin/cos tables
    pub fn new(head_dim: usize, max_position: usize, base: f64, device: &Device) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(4) {
            return Err(TensorError::ValueOutOfRange {
                description: "RotaryEmbedding2d: head_dim must be a positive multiple of 4",
            });
        }
        if max_position == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "RotaryEmbedding2d: max_position must be > 0",
            });
        }
        // Position IDs are stored as U32 for index_select; validate they fit.
        if max_position > u32::MAX as usize {
            return Err(TensorError::ValueOutOfRange {
                description: "RotaryEmbedding2d: max_position exceeds u32::MAX",
            });
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(TensorError::ValueOutOfRange {
                description: "RotaryEmbedding2d: base must be positive finite",
            });
        }

        // Each spatial dimension gets half_dim / 2 frequency pairs.
        let half_dim = head_dim / 2;
        let quarter_dim = half_dim / 2;

        // inv_freq[i] = 1 / (base^(2i / half_dim)) for i = 0..quarter_dim
        let inv_freq: Vec<f32> = (0..quarter_dim)
            .map(|i| {
                let exponent = (2 * i) as f64 / half_dim as f64;
                (1.0 / base.powf(exponent)) as f32
            })
            .collect();

        let mut cos_data = Vec::with_capacity(max_position * quarter_dim);
        let mut sin_data = Vec::with_capacity(max_position * quarter_dim);

        for pos in 0..max_position {
            for &freq in &inv_freq {
                let angle = (pos as f64 * f64::from(freq)) as f32;
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }

        let cos_cache = DynTensor::from_vec(cos_data, &[max_position, quarter_dim], &Device::Cpu)?
            .to_device(device)?;
        let sin_cache = DynTensor::from_vec(sin_data, &[max_position, quarter_dim], &Device::Cpu)?
            .to_device(device)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
            max_position,
        })
    }

    /// Apply 2D RoPE to a tensor at given height and width positions.
    ///
    /// - `x`: input tensor `[..., seq_len, head_dim]`
    /// - `h_positions`: height position for each token (length = seq_len)
    /// - `w_positions`: width position for each token (length = seq_len)
    ///
    /// Returns tensor with same shape as input.
    pub fn apply(
        &self,
        x: &DynTensor,
        h_positions: &[usize],
        w_positions: &[usize],
    ) -> Result<DynTensor> {
        let rank = x.rank();
        if rank < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: rank,
            });
        }
        let dims = x.dims();
        let last_dim = dims[rank - 1];
        let seq_len = dims[rank - 2];
        if last_dim != self.head_dim {
            return Err(TensorError::shape_mismatch(
                vec![seq_len, self.head_dim],
                vec![seq_len, last_dim],
            ));
        }
        if h_positions.len() != seq_len {
            return Err(TensorError::DataLengthMismatch {
                expected: seq_len,
                actual: h_positions.len(),
            });
        }
        if w_positions.len() != seq_len {
            return Err(TensorError::DataLengthMismatch {
                expected: seq_len,
                actual: w_positions.len(),
            });
        }

        // Validate all positions.
        for &p in h_positions.iter().chain(w_positions.iter()) {
            if p >= self.max_position {
                return Err(TensorError::ValueOutOfRange {
                    description: "RotaryEmbedding2d: position exceeds max_position",
                });
            }
        }

        let half_dim = self.head_dim / 2;
        let quarter_dim = half_dim / 2;

        // Gather cos/sin for height and width positions.
        let h_pos_u32: Vec<u32> = h_positions.iter().map(|&p| p as u32).collect();
        let w_pos_u32: Vec<u32> = w_positions.iter().map(|&p| p as u32).collect();
        let h_ids = DynTensor::from_vec_u32(h_pos_u32, &[seq_len], &self.cos_cache.device())?;
        let w_ids = DynTensor::from_vec_u32(w_pos_u32, &[seq_len], &self.cos_cache.device())?;

        // cos/sin tables: [seq_len, quarter_dim]
        let h_cos = self.cos_cache.index_select(&h_ids, 0)?;
        let h_sin = self.sin_cache.index_select(&h_ids, 0)?;
        let w_cos = self.cos_cache.index_select(&w_ids, 0)?;
        let w_sin = self.sin_cache.index_select(&w_ids, 0)?;

        // Split input into height-half and width-half.
        let x_h = x.narrow(rank - 1, 0, half_dim)?;
        let x_w = x.narrow(rank - 1, half_dim, half_dim)?;

        // Apply 1D RoPE to each half independently.
        let y_h = apply_rope_1d(&x_h, &h_cos, &h_sin, rank, seq_len, quarter_dim)?;
        let y_w = apply_rope_1d(&x_w, &w_cos, &w_sin, rank, seq_len, quarter_dim)?;

        // Concatenate halves back together.
        DynTensor::cat(&[&y_h, &y_w], rank - 1)
    }

    /// Head dimension this embedding was created for.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Maximum spatial position supported.
    #[must_use]
    pub fn max_position(&self) -> usize {
        self.max_position
    }
}

/// Apply standard 1D RoPE rotation to a tensor using precomputed cos/sin.
///
/// - `x`: `[..., seq_len, dim]` where `dim = 2 * quarter_dim`
/// - `cos`, `sin`: `[seq_len, quarter_dim]`
fn apply_rope_1d(
    x: &DynTensor,
    cos: &DynTensor,
    sin: &DynTensor,
    rank: usize,
    seq_len: usize,
    quarter_dim: usize,
) -> Result<DynTensor> {
    let dims = x.dims();
    // Reshape last dim into pairs: [..., seq_len, quarter_dim, 2]
    let mut pairs_shape: Vec<usize> = dims[..rank - 1].to_vec();
    pairs_shape.push(quarter_dim);
    pairs_shape.push(2);

    let x_pairs = x.reshape(&pairs_shape)?;
    let x_even = x_pairs.narrow(rank, 0, 1)?.squeeze(rank)?;
    let x_odd = x_pairs.narrow(rank, 1, 1)?.squeeze(rank)?;

    // Broadcast cos/sin to match batch dimensions.
    let mut broadcast_shape = vec![1usize; rank - 2];
    broadcast_shape.push(seq_len);
    broadcast_shape.push(quarter_dim);
    let cos_bc = cos.reshape(&broadcast_shape)?;
    let sin_bc = sin.reshape(&broadcast_shape)?;

    // Standard rotation:
    // y_even = x_even * cos - x_odd * sin
    // y_odd  = x_even * sin + x_odd * cos
    let y_even = x_even
        .broadcast_mul(&cos_bc)?
        .broadcast_sub(&x_odd.broadcast_mul(&sin_bc)?)?;
    let y_odd = x_even
        .broadcast_mul(&sin_bc)?
        .broadcast_add(&x_odd.broadcast_mul(&cos_bc)?)?;

    // Interleave back: [..., quarter_dim, 2] → [..., 2*quarter_dim]
    let y_even_expanded = y_even.unsqueeze(rank)?;
    let y_odd_expanded = y_odd.unsqueeze(rank)?;
    let y_pairs = DynTensor::cat(&[&y_even_expanded, &y_odd_expanded], rank)?;
    y_pairs.reshape(dims)
}

/// Generate fixed sinusoidal 2D positional encoding.
///
/// Produces a `[height * width, dim]` tensor where each row is the positional
/// encoding for position `(h, w)` in raster scan order (row-major).
///
/// The encoding splits `dim` into 4 equal parts:
/// - sin(h * freq_i) for i = 0..dim/4
/// - cos(h * freq_i)
/// - sin(w * freq_i)
/// - cos(w * freq_i)
///
/// where `freq_i = 1 / (temperature^(2i / (dim/2)))`.
///
/// This matches the DETR / TableFormer positional encoding.
///
/// # Arguments
///
/// - `height`: spatial height
/// - `width`: spatial width
/// - `dim`: embedding dimension (must be divisible by 4)
/// - `temperature`: frequency base (typically 10000.0)
/// - `device`: target device
pub fn sinusoidal_2d(
    height: usize,
    width: usize,
    dim: usize,
    temperature: f64,
    device: &Device,
) -> Result<DynTensor> {
    if dim == 0 || !dim.is_multiple_of(4) {
        return Err(TensorError::ValueOutOfRange {
            description: "sinusoidal_2d: dim must be a positive multiple of 4",
        });
    }
    if height == 0 || width == 0 {
        return Err(TensorError::ValueOutOfRange {
            description: "sinusoidal_2d: height and width must be > 0",
        });
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(TensorError::ValueOutOfRange {
            description: "sinusoidal_2d: temperature must be positive finite",
        });
    }

    let quarter_dim = dim / 4;
    let seq_len = height * width;

    // inv_freq[i] = 1 / (temperature^(2i / (dim/2)))
    // Note: dim/2 is the per-axis dimension (height or width each get dim/2).
    let half_dim = dim / 2;
    let inv_freq: Vec<f64> = (0..quarter_dim)
        .map(|i| {
            let exponent = (2 * i) as f64 / half_dim as f64;
            1.0 / temperature.powf(exponent)
        })
        .collect();

    let mut data = vec![0.0f32; seq_len * dim];

    for h in 0..height {
        for w in 0..width {
            let row = h * width + w;
            let offset = row * dim;
            for (i, &freq) in inv_freq.iter().enumerate() {
                let h_angle = (h as f64 * freq) as f32;
                let w_angle = (w as f64 * freq) as f32;
                // [sin_h | cos_h | sin_w | cos_w]
                data[offset + i] = h_angle.sin();
                data[offset + quarter_dim + i] = h_angle.cos();
                data[offset + 2 * quarter_dim + i] = w_angle.sin();
                data[offset + 3 * quarter_dim + i] = w_angle.cos();
            }
        }
    }

    DynTensor::from_vec(data, &[seq_len, dim], &Device::Cpu)?.to_device(device)
}

#[cfg(test)]
#[path = "rope_2d_tests.rs"]
mod tests;
