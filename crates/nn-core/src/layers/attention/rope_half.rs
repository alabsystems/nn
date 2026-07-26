// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Half-RoPE and candle-compatible free-function RoPE.
//!
//! [`HalfRotaryEmbedding`] applies RoPE only to the first half of head
//! dimensions; the second half passes through unchanged. Used by Qwen3,
//! Irodori-TTS, and other models with partial rotation.
//!
//! [`rope()`] is a candle-nn compatible free function applying half-dim
//! cos/sin tensors in the GPT-NeoX style.

use super::RotaryEmbedding;
use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};

/// Half-RoPE: rotary position embeddings applied only to the first half
/// of the head dimensions. The second half passes through unchanged.
///
/// Used by Qwen3, Irodori-TTS, and other models where partial rotation
/// is preferred over full-head rotation.
///
/// # Usage
///
/// ```ignore
/// // NOTE: ignore — requires undefined q tensor and offset variable
/// // head_dim=128: first 64 dims rotated, last 64 dims pass-through
/// let half_rope = HalfRotaryEmbedding::new(128, 4096, 1000000.0, &device)?;
/// let q_rotated = half_rope.apply(&q, offset)?;
/// ```
#[derive(Debug, Clone)]
pub struct HalfRotaryEmbedding {
    /// Inner RoPE applied to the first half of head dimensions.
    inner: RotaryEmbedding,
    /// Full head dimension (input tensor last dim).
    full_head_dim: usize,
}

impl HalfRotaryEmbedding {
    /// Create a new Half-RoPE embedding.
    ///
    /// - `head_dim`: full head dimension (must be divisible by 4, since the
    ///   rotated half must be even for RoPE pairing)
    /// - `max_seq_len`: maximum sequence length to precompute for
    /// - `base`: RoPE base frequency (typically 1000000.0 for Qwen3)
    /// - `device`: target device
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        base: f64,
        device: &crate::Device,
    ) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(4) {
            return Err(TensorError::ValueOutOfRange {
                description: "HalfRotaryEmbedding: head_dim must be a positive multiple of 4",
            });
        }
        let rope_dim = head_dim / 2;
        let inner = RotaryEmbedding::new(rope_dim, max_seq_len, base, device)?;
        Ok(Self {
            inner,
            full_head_dim: head_dim,
        })
    }

    /// Apply Half-RoPE rotation to a tensor.
    ///
    /// The last dimension must equal `head_dim`. The first `head_dim/2` elements
    /// are rotated via RoPE; the second `head_dim/2` elements pass through unchanged.
    ///
    /// Input shape: `[..., seq_len, head_dim]`
    /// Output shape: same as input.
    pub fn apply(&self, x: &DynTensor, offset: usize) -> Result<DynTensor> {
        let rank = x.rank();
        if rank < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: rank,
            });
        }
        let last_dim = x.dims()[rank - 1];
        if last_dim != self.full_head_dim {
            return Err(TensorError::shape_mismatch(
                vec![last_dim, self.full_head_dim],
                vec![last_dim, last_dim],
            ));
        }
        let rope_dim = self.full_head_dim / 2;
        let x_rope = x.narrow(rank - 1, 0, rope_dim)?;
        let x_pass = x.narrow(rank - 1, rope_dim, rope_dim)?;
        let x_rotated = self.inner.apply(&x_rope, offset)?;
        DynTensor::cat(&[&x_rotated, &x_pass], rank - 1)
    }

    /// Apply Half-RoPE rotation to query and key tensors at explicit positions.
    ///
    /// Returns `(q_rotated, k_rotated)` with the same shapes.
    pub fn apply_pair(
        &self,
        q: &DynTensor,
        k: &DynTensor,
        positions: &[usize],
    ) -> Result<(DynTensor, DynTensor)> {
        let q_rot = self.apply_at_positions(q, positions)?;
        let k_rot = self.apply_at_positions(k, positions)?;
        Ok((q_rot, k_rot))
    }

    /// Apply Half-RoPE at arbitrary (non-contiguous) positions.
    fn apply_at_positions(&self, x: &DynTensor, positions: &[usize]) -> Result<DynTensor> {
        let rank = x.rank();
        if rank < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: rank,
            });
        }
        let last_dim = x.dims()[rank - 1];
        if last_dim != self.full_head_dim {
            return Err(TensorError::shape_mismatch(
                vec![last_dim, self.full_head_dim],
                vec![last_dim, last_dim],
            ));
        }
        let rope_dim = self.full_head_dim / 2;
        let x_rope = x.narrow(rank - 1, 0, rope_dim)?;
        let x_pass = x.narrow(rank - 1, rope_dim, rope_dim)?;
        let x_rotated = self.inner.apply_at_positions(&x_rope, positions)?;
        DynTensor::cat(&[&x_rotated, &x_pass], rank - 1)
    }

    /// Full head dimension this embedding was created for.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.full_head_dim
    }

    /// The dimension of the rotated portion (head_dim / 2).
    #[must_use]
    pub fn rope_dim(&self) -> usize {
        self.full_head_dim / 2
    }

    /// Maximum sequence length supported.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len()
    }
}

// ---------------------------------------------------------------------------
// Free function: candle-nn compatible half-dim RoPE
// ---------------------------------------------------------------------------

/// Apply rotary position embeddings using pre-computed half-dim cos/sin tensors.
///
/// Matches `candle_nn::rotary_emb::rope(&t, &cos, &sin)` convention:
/// cos/sin have last dim = head_dim / 2 (half-dim, GPT-NeoX style).
///
/// # Shape requirements
/// - `t`: `[..., seq_len, head_dim]` where `head_dim` is even
/// - `cos`: last dim = head_dim/2, broadcastable to `[..., seq_len, half]`
/// - `sin`: last dim = head_dim/2, broadcastable to `[..., seq_len, half]`
///
/// # Rotation (half-dim cos/sin, candle convention)
/// ```text
/// x1 = t[..., :half], x2 = t[..., half:]
/// y1 = x1 * cos - x2 * sin
/// y2 = x1 * sin + x2 * cos
/// result = cat([y1, y2], dim=-1)
/// ```
pub fn rope(t: &DynTensor, cos: &DynTensor, sin: &DynTensor) -> Result<DynTensor> {
    let rank = t.rank();
    if rank < 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: rank,
        });
    }
    let head_dim = t.dims()[rank - 1];
    if !head_dim.is_multiple_of(2) {
        return Err(TensorError::ValueOutOfRange {
            description: "rope requires even head_dim",
        });
    }
    let half = head_dim / 2;

    // Validate cos/sin last dim is half of head_dim (candle convention)
    let cos_last = cos.dims()[cos.rank() - 1];
    let sin_last = sin.dims()[sin.rank() - 1];
    if cos_last != half || sin_last != half {
        return Err(TensorError::shape_mismatch(vec![half], vec![cos_last]));
    }

    // Split input into first and second halves
    let x1 = t.narrow(rank - 1, 0, half)?;
    let x2 = t.narrow(rank - 1, half, half)?;

    // y1 = x1 * cos - x2 * sin
    let y1 = x1
        .broadcast_mul(cos)?
        .broadcast_sub(&x2.broadcast_mul(sin)?)?;
    // y2 = x1 * sin + x2 * cos
    let y2 = x1
        .broadcast_mul(sin)?
        .broadcast_add(&x2.broadcast_mul(cos)?)?;

    DynTensor::cat(&[&y1, &y2], rank - 1)
}
