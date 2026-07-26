// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Rotary Position Embedding (RoPE) for transformer attention.
//!
//! Provides [`RotaryEmbedding`] which precomputes sin/cos frequency tables
//! and applies RoPE rotation to query/key tensors in attention layers.
//!
//! # Usage
//!
//! ```ignore
//! // NOTE: ignore — requires undefined q/k tensors and offset variable
//! let rope = RotaryEmbedding::new(head_dim, max_seq_len, 10000.0, &device)?;
//! let q_rotated = rope.apply(&q, offset)?; // q: [..., seq_len, head_dim]
//! let k_rotated = rope.apply(&k, offset)?;
//! ```
//!
//! The rotation applies to pairs of elements along the last dimension:
//!
//! ```text
//! x_out[..., 2i]   = x[..., 2i] * cos(pos * θ_i) - x[..., 2i+1] * sin(pos * θ_i)
//! x_out[..., 2i+1] = x[..., 2i] * sin(pos * θ_i) + x[..., 2i+1] * cos(pos * θ_i)
//! ```
//!
//! where `θ_i = base^(-2i / head_dim)`.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

#[path = "rope_yarn.rs"]
mod rope_yarn_impl;
pub use rope_yarn_impl::YarnScaling;

/// Rotary Position Embedding for transformer autoregressive decoding.
///
/// Precomputes sin and cos frequency tables once, then efficiently applies
/// rotation to Q/K tensors at each decoding step.
///
/// Supports any input tensor with `head_dim` as the last dimension.
/// Common shapes: `[batch, seq, heads, head_dim]` or `[batch, heads, seq, head_dim]`.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Precomputed cos values: `[max_seq_len, head_dim/2]`
    cos_cache: DynTensor,
    /// Precomputed sin values: `[max_seq_len, head_dim/2]`
    sin_cache: DynTensor,
    head_dim: usize,
    max_seq_len: usize,
}

impl RotaryEmbedding {
    /// Create a new RotaryEmbedding with precomputed frequency tables.
    ///
    /// - `head_dim`: dimension of each attention head (must be even)
    /// - `max_seq_len`: maximum sequence length to precompute for
    /// - `base`: RoPE base frequency (typically 10000.0 for LLaMA, 1000000.0 for Qwen3)
    /// - `device`: target device (currently all computation is CPU f32)
    ///
    /// The frequency for each dimension pair `i` is `base^(-2i / head_dim)`.
    pub fn new(head_dim: usize, max_seq_len: usize, base: f64, device: &Device) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(TensorError::ValueOutOfRange {
                description: "RotaryEmbedding: head_dim must be a positive even number",
            });
        }
        if max_seq_len == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "RotaryEmbedding: max_seq_len must be > 0",
            });
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(TensorError::ValueOutOfRange {
                description: "RotaryEmbedding: base must be a positive finite number",
            });
        }

        let half_dim = head_dim / 2;

        // inv_freq[i] = 1 / (base^(2i / head_dim)) for i = 0..half_dim
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exponent = (2 * i) as f64 / head_dim as f64;
                (1.0 / base.powf(exponent)) as f32
            })
            .collect();

        // positions = [0, 1, 2, ..., max_seq_len - 1]
        // freqs[pos][i] = pos * inv_freq[i]
        let cache_len =
            max_seq_len
                .checked_mul(half_dim)
                .ok_or(TensorError::DimensionOverflow {
                    dims: vec![max_seq_len, half_dim],
                })?;
        let mut cos_data = Vec::with_capacity(cache_len);
        let mut sin_data = Vec::with_capacity(cache_len);

        for pos in 0..max_seq_len {
            for &freq in &inv_freq {
                let angle = (pos as f64 * f64::from(freq)) as f32;
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }

        // Build caches on CPU first (for f32 math), then transfer to target device.
        let cos_cache = DynTensor::from_vec(cos_data, &[max_seq_len, half_dim], &Device::Cpu)?
            .to_device(device)?;
        let sin_cache = DynTensor::from_vec(sin_data, &[max_seq_len, half_dim], &Device::Cpu)?
            .to_device(device)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
            max_seq_len,
        })
    }

    /// Apply RoPE rotation to a tensor.
    ///
    /// The last dimension must equal `head_dim`. The second-to-last dimension
    /// is treated as the sequence dimension. All leading dimensions are batch dims.
    ///
    /// `offset` is the starting position (0 for prefill, `cache_len` for decoding).
    ///
    /// Input shapes: `[..., seq_len, head_dim]`
    /// Output shape: same as input.
    pub fn apply(&self, x: &DynTensor, offset: usize) -> Result<DynTensor> {
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
        let end_pos = offset + seq_len;
        if end_pos > self.max_seq_len {
            return Err(TensorError::ValueOutOfRange {
                description: "RotaryEmbedding: offset + seq_len exceeds max_seq_len",
            });
        }
        let tracing = trace::is_tracing();
        let head_dim = self.head_dim;
        let half_dim = head_dim / 2;

        // Narrow cos/sin caches under trace suppression — the caches were
        // created before the trace scope so they have no trace_node_id.
        let (cos, sin) = trace::with_trace_suppressed(|| -> Result<(DynTensor, DynTensor)> {
            let c = self.cos_cache.narrow(0, offset, seq_len)?;
            let s = self.sin_cache.narrow(0, offset, seq_len)?;
            Ok((c, s))
        })?;

        let mut result = if tracing {
            trace::with_trace_suppressed(|| {
                self.apply_with_cos_sin(x, &cos, &sin, dims, rank, seq_len, half_dim)
            })?
        } else {
            self.apply_with_cos_sin(x, &cos, &sin, dims, rank, seq_len, half_dim)?
        };

        if tracing {
            // Capture narrowed cos/sin as WeightRef for NY RoPE layer.
            let cos_weight = cos.to_weight_ref()?;
            let sin_weight = sin.to_weight_ref()?;

            let input_ids = DynTensor::trace_input_ids(&[x])?;
            if let Some(id) = trace::record_op(
                TraceOp::RotaryEmbedding {
                    head_dim: self.head_dim,
                    offset,
                    cos_cache: cos_weight,
                    sin_cache: sin_weight,
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Apply RoPE rotation to query and key tensors at explicit positions.
    ///
    /// - `q`: `[batch, num_heads, seq_len, head_dim]` (or any rank >= 2)
    /// - `k`: `[batch, num_kv_heads, seq_len, head_dim]`
    /// - `positions`: position index for each token (`positions.len() == seq_len`)
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

    /// Apply RoPE at arbitrary (non-contiguous) positions.
    fn apply_at_positions(&self, x: &DynTensor, positions: &[usize]) -> Result<DynTensor> {
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
        // Validate all positions before gathering.
        for &p in positions {
            if p >= self.max_seq_len {
                return Err(TensorError::ValueOutOfRange {
                    description: "RoPE position exceeds max_seq_len",
                });
            }
        }
        // Gather cos/sin for explicit positions via index_select.
        // This stays on GPU when the cache is on GPU, avoiding a full-cache
        // CPU round-trip that would transfer 2×(max_seq_len × half_dim × 4) bytes.
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
        self.apply_with_cos_sin(x, &cos, &sin, dims, rank, seq_len, half_dim)
    }

    /// Core rotation logic shared between offset and positions APIs.
    ///
    /// GPU fast-path (#1363): when `x` is on GPU and a fused RoPE kernel is
    /// available, delegates to `GpuBackend::rope()` for a single dispatch.
    /// Falls back to the decomposed 11-op path on CPU or when no fused kernel
    /// is registered.
    fn apply_with_cos_sin(
        &self,
        x: &DynTensor,
        cos: &DynTensor,
        sin: &DynTensor,
        dims: &[usize],
        rank: usize,
        seq_len: usize,
        half_dim: usize,
    ) -> Result<DynTensor> {
        // Convert cos/sin to match input dtype so GPU binary ops don't get
        // a DTypeMismatch (e.g., BF16 input × F32 cos). Cos/sin are computed
        // in F32 for precision; conversion happens here at apply time (#1710).
        let (cos, sin) = if x.dtype() != cos.dtype() {
            (cos.to_dtype(x.dtype())?, sin.to_dtype(x.dtype())?)
        } else {
            (cos.clone(), sin.clone())
        };

        // Fused GPU path: single dispatch replaces 11 separate kernel launches.
        if x.device().is_gpu() {
            if let Some(result) = crate::dyn_tensor::gpu_backend_dispatch(|b| b.rope(x, &cos, &sin))
            {
                return result;
            }
        }

        // Decomposed CPU/fallback path.
        let mut pairs_shape: Vec<usize> = dims[..rank - 1].to_vec();
        pairs_shape.push(half_dim);
        pairs_shape.push(2);
        let x_pairs = x.reshape(&pairs_shape)?;
        let x_even = x_pairs.narrow(rank, 0, 1)?.squeeze(rank)?;
        let x_odd = x_pairs.narrow(rank, 1, 1)?.squeeze(rank)?;
        let mut broadcast_shape = vec![1usize; rank - 2];
        broadcast_shape.push(seq_len);
        broadcast_shape.push(half_dim);
        let cos_bc = cos.reshape(&broadcast_shape)?;
        let sin_bc = sin.reshape(&broadcast_shape)?;
        let y_even = x_even
            .broadcast_mul(&cos_bc)?
            .broadcast_sub(&x_odd.broadcast_mul(&sin_bc)?)?;
        let y_odd = x_even
            .broadcast_mul(&sin_bc)?
            .broadcast_add(&x_odd.broadcast_mul(&cos_bc)?)?;
        let y_even_expanded = y_even.unsqueeze(rank)?;
        let y_odd_expanded = y_odd.unsqueeze(rank)?;
        let y_pairs = DynTensor::cat(&[&y_even_expanded, &y_odd_expanded], rank)?;
        y_pairs.reshape(dims)
    }

    /// Head dimension this embedding was created for.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Maximum sequence length supported.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

// -- Half-split RoPE (HuggingFace rotate_half convention, extracted to rope_half_split.rs) --

#[path = "rope_half_split.rs"]
mod rope_half_split_impl;

// -- Half-RoPE and candle-compat free function (extracted to rope_half.rs) ---

#[path = "rope_half.rs"]
mod rope_half_impl;
pub use rope_half_impl::{rope, HalfRotaryEmbedding};

#[cfg(test)]
#[path = "rope_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "rope_tests_extended.rs"]
mod tests_extended;

#[cfg(test)]
#[path = "half_rope_tests.rs"]
mod half_rope_tests;

#[cfg(test)]
#[path = "rope_yarn_tests.rs"]
mod yarn_tests;
