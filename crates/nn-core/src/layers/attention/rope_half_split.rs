// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Half-split RoPE rotation (HuggingFace `rotate_half` convention).
//!
//! Extracted from `rope.rs` to keep that file under the 450-line limit.
//! Uses `(i, i + head_dim/2)` pairing instead of interleaved `(2i, 2i+1)`.

use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};

use super::RotaryEmbedding;

impl RotaryEmbedding {
    /// Apply RoPE rotation using the half-split (rotate_half) convention.
    ///
    /// Unlike [`apply_pair`] which uses interleaved pairing `(2i, 2i+1)`,
    /// this method uses half-split pairing `(i, i + head_dim/2)`:
    ///
    /// ```text
    /// x1 = x[..., :half], x2 = x[..., half:]
    /// y1 = x1 * cos - x2 * sin
    /// y2 = x1 * sin + x2 * cos
    /// result = cat([y1, y2], dim=-1)
    /// ```
    ///
    /// This matches HuggingFace `rotate_half` used by Qwen3, LLaMA, and most
    /// modern transformer models. Use this instead of [`apply_pair`] when
    /// loading weights directly from HuggingFace safetensors without
    /// permuting Q/K projection columns.
    pub fn apply_pair_half_split(
        &self,
        q: &DynTensor,
        k: &DynTensor,
        positions: &[usize],
    ) -> Result<(DynTensor, DynTensor)> {
        let q_rot = self.apply_half_split_at_positions(q, positions)?;
        let k_rot = self.apply_half_split_at_positions(k, positions)?;
        Ok((q_rot, k_rot))
    }

    /// Apply half-split RoPE at arbitrary (non-contiguous) positions.
    fn apply_half_split_at_positions(
        &self,
        x: &DynTensor,
        positions: &[usize],
    ) -> Result<DynTensor> {
        let rank = x.rank();
        if rank < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: rank,
            });
        }
        let dims = x.dims();
        let seq_len = dims[rank - 2];
        let last_dim = dims[rank - 1];
        if last_dim != self.head_dim {
            return Err(TensorError::shape_mismatch(
                vec![seq_len, self.head_dim],
                vec![seq_len, last_dim],
            ));
        }
        if positions.len() != seq_len {
            return Err(TensorError::DataLengthMismatch {
                expected: seq_len,
                actual: positions.len(),
            });
        }
        let half_dim = self.head_dim / 2;
        for &p in positions {
            if p >= self.max_seq_len {
                return Err(TensorError::ValueOutOfRange {
                    description: "RoPE position exceeds max_seq_len",
                });
            }
        }
        // Gather cos/sin for explicit positions via index_select.
        let pos_u32: Vec<u32> = positions
            .iter()
            .map(|&p| {
                u32::try_from(p).map_err(|_| TensorError::ValueOutOfRange {
                    description: "RoPE position exceeds u32::MAX",
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let pos_ids = DynTensor::from_vec_u32(pos_u32, &[seq_len], &self.cos_cache.device())?;
        let cos = self.cos_cache.index_select(&pos_ids, 0)?;
        let sin = self.sin_cache.index_select(&pos_ids, 0)?;
        self.apply_half_split_with_cos_sin(x, &cos, &sin, rank, seq_len, half_dim)
    }

    /// Core half-split rotation logic.
    ///
    /// Splits input along the last dimension into first/second halves and
    /// cross-rotates them, matching HuggingFace `rotate_half`.
    fn apply_half_split_with_cos_sin(
        &self,
        x: &DynTensor,
        cos: &DynTensor,
        sin: &DynTensor,
        rank: usize,
        seq_len: usize,
        half_dim: usize,
    ) -> Result<DynTensor> {
        let (cos, sin) = if x.dtype() != cos.dtype() {
            (cos.to_dtype(x.dtype())?, sin.to_dtype(x.dtype())?)
        } else {
            (cos.clone(), sin.clone())
        };

        // Split input into first and second halves along last dimension.
        let x1 = x.narrow(rank - 1, 0, half_dim)?;
        let x2 = x.narrow(rank - 1, half_dim, half_dim)?;

        // Broadcast cos/sin: [seq_len, half_dim] → [1, ..., seq_len, half_dim]
        let mut broadcast_shape = vec![1usize; rank - 2];
        broadcast_shape.push(seq_len);
        broadcast_shape.push(half_dim);
        let cos_bc = cos.reshape(&broadcast_shape)?;
        let sin_bc = sin.reshape(&broadcast_shape)?;

        // y1 = x1 * cos - x2 * sin
        let y1 = x1
            .broadcast_mul(&cos_bc)?
            .broadcast_sub(&x2.broadcast_mul(&sin_bc)?)?;
        // y2 = x1 * sin + x2 * cos
        let y2 = x1
            .broadcast_mul(&sin_bc)?
            .broadcast_add(&x2.broadcast_mul(&cos_bc)?)?;

        DynTensor::cat(&[&y1, &y2], rank - 1)
    }
}
