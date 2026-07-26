// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multimodal Rotary Position Embedding (M-ROPE) for vision-language models.
//!
//! [`MultimodalRoPE`] assigns the head dimension to temporal, height, and width
//! rotary sections using the Hugging Face / Qwen layout
//! `[T1 | H1 | W1 | T2 | H2 | W2]`, where the first and second global halves
//! are paired by `rotate_half`. Required by Qwen2-VL, Qwen2.5-VL, and
//! PaddleOCR-VL for encoding multimodal (text + image) position information.
//!
//! For text tokens, all 3 position IDs are identical (sequential positions).
//! For image tokens, temporal is fixed per image, height and width vary per patch.
//!
//! Reference: Qwen2-VL (Wang et al., 2024), §3.2 M-ROPE.

use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

/// Multimodal Rotary Position Embedding (M-ROPE).
///
/// Splits head_dim into temporal, height, and width position sections using
/// the Hugging Face / Qwen six-block layout `[T1 | H1 | W1 | T2 | H2 | W2]`.
/// Each axis uses standard 1D RoPE frequencies independently.
///
/// Input: `[..., seq_len, head_dim]` where `head_dim = 2 * (t_pairs + h_pairs + w_pairs)`.
/// Position IDs: 3 arrays of length `seq_len` (temporal, height, width).
#[derive(Clone)]
pub struct MultimodalRoPE {
    /// Precomputed cos values: `[max_position, section_half_dim]` per section.
    cos_caches: [DynTensor; 3],
    /// Precomputed sin values: `[max_position, section_half_dim]` per section.
    sin_caches: [DynTensor; 3],
    /// Dimensions for each section: [temporal, height, width].
    section_dims: [usize; 3],
    /// Full head dimension (sum of section_dims).
    head_dim: usize,
    /// Maximum position value.
    max_position: usize,
}

impl std::fmt::Debug for MultimodalRoPE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultimodalRoPE")
            .field("head_dim", &self.head_dim)
            .field("section_dims", &self.section_dims)
            .field("max_position", &self.max_position)
            .finish_non_exhaustive()
    }
}

impl MultimodalRoPE {
    /// Create a new multimodal rotary embedding.
    ///
    /// - `head_dim`: total dimension per attention head (must be even)
    /// - `mrope_section_sizes`: 3-element array specifying how many channels
    ///   in each global half belong to `[temporal, height, width]`.
    ///   Sum must equal `head_dim / 2`, and each axis receives twice that many
    ///   total channels across the full `[T1 | H1 | W1 | T2 | H2 | W2]` layout.
    /// - `max_position`: maximum position value for any axis
    /// - `base`: frequency base (typically 1000000.0 for Qwen2.5-VL)
    /// - `device`: target device for cached sin/cos tables
    pub fn new(
        head_dim: usize,
        mrope_section_sizes: [usize; 3],
        max_position: usize,
        base: f64,
        device: &Device,
    ) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(TensorError::ValueOutOfRange {
                description: "MultimodalRoPE: head_dim must be a positive even number",
            });
        }
        let half_dim = head_dim / 2;
        let total_pairs: usize = mrope_section_sizes.iter().sum();
        if total_pairs != half_dim {
            return Err(TensorError::ValueOutOfRange {
                description: "MultimodalRoPE: sum of mrope_section_sizes must equal head_dim / 2",
            });
        }
        for &s in &mrope_section_sizes {
            if s == 0 {
                return Err(TensorError::ValueOutOfRange {
                    description: "MultimodalRoPE: each section size must be > 0",
                });
            }
        }
        if max_position == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MultimodalRoPE: max_position must be > 0",
            });
        }
        if max_position > u32::MAX as usize {
            return Err(TensorError::ValueOutOfRange {
                description: "MultimodalRoPE: max_position exceeds u32::MAX",
            });
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MultimodalRoPE: base must be positive finite",
            });
        }

        let section_dims = [
            mrope_section_sizes[0] * 2,
            mrope_section_sizes[1] * 2,
            mrope_section_sizes[2] * 2,
        ];

        // Build per-section frequency caches.
        // Each section has its own inv_freq computed over the FULL head_dim
        // frequency space but offset to the section's range.
        let mut cos_caches = Vec::with_capacity(3);
        let mut sin_caches = Vec::with_capacity(3);
        let mut freq_offset = 0usize;
        for &section_pairs in &mrope_section_sizes {
            let mut cos_data = Vec::with_capacity(max_position * section_pairs);
            let mut sin_data = Vec::with_capacity(max_position * section_pairs);

            for pos in 0..max_position {
                for i in 0..section_pairs {
                    let global_i = freq_offset + i;
                    let exponent = (2 * global_i) as f64 / head_dim as f64;
                    let inv_freq = (1.0 / base.powf(exponent)) as f32;
                    let angle = (pos as f64 * f64::from(inv_freq)) as f32;
                    cos_data.push(angle.cos());
                    sin_data.push(angle.sin());
                }
            }

            let cos = DynTensor::from_vec(cos_data, &[max_position, section_pairs], &Device::Cpu)?
                .to_device(device)?;
            let sin = DynTensor::from_vec(sin_data, &[max_position, section_pairs], &Device::Cpu)?
                .to_device(device)?;

            cos_caches.push(cos);
            sin_caches.push(sin);
            freq_offset += section_pairs;
        }

        Ok(Self {
            cos_caches: [
                cos_caches.remove(0),
                cos_caches.remove(0),
                cos_caches.remove(0),
            ],
            sin_caches: [
                sin_caches.remove(0),
                sin_caches.remove(0),
                sin_caches.remove(0),
            ],
            section_dims,
            head_dim,
            max_position,
        })
    }

    /// Apply M-ROPE to query and key tensors using the Hugging Face / Qwen
    /// global half-split convention.
    ///
    /// - `q`: `[batch, num_heads, seq_len, head_dim]`
    /// - `k`: `[batch, num_kv_heads, seq_len, head_dim]`
    /// - `t_positions`: temporal position per token (length = seq_len)
    /// - `h_positions`: height position per token (length = seq_len)
    /// - `w_positions`: width position per token (length = seq_len)
    ///
    /// Returns `(q_rotated, k_rotated)`.
    pub fn apply_pair(
        &self,
        q: &DynTensor,
        k: &DynTensor,
        t_positions: &[usize],
        h_positions: &[usize],
        w_positions: &[usize],
    ) -> Result<(DynTensor, DynTensor)> {
        let q_rot = self.apply(q, t_positions, h_positions, w_positions)?;
        let k_rot = self.apply(k, t_positions, h_positions, w_positions)?;
        Ok((q_rot, k_rot))
    }

    /// Apply M-ROPE to a single tensor using the Hugging Face / Qwen
    /// global half-split convention.
    ///
    /// Input: `[..., seq_len, head_dim]`
    pub fn apply(
        &self,
        x: &DynTensor,
        t_positions: &[usize],
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
        let all_positions = [t_positions, h_positions, w_positions];
        for (i, positions) in all_positions.iter().enumerate() {
            if positions.len() != seq_len {
                return Err(TensorError::DataLengthMismatch {
                    expected: seq_len,
                    actual: positions.len(),
                });
            }
            for &p in positions.iter() {
                if p >= self.max_position {
                    let descs = [
                        "MultimodalRoPE: temporal position exceeds max_position",
                        "MultimodalRoPE: height position exceeds max_position",
                        "MultimodalRoPE: width position exceeds max_position",
                    ];
                    return Err(TensorError::ValueOutOfRange {
                        description: descs[i],
                    });
                }
            }
        }

        // Hugging Face / Qwen use a global rotate_half split over the full head
        // dimension, with modality chunks packed as [T1 | H1 | W1 | T2 | H2 | W2].
        // We therefore rotate matching chunks from the first and second halves
        // together instead of doing a local half-split inside each modality block.
        let half_dim = self.head_dim / 2;
        let x_first_half = x.narrow(rank - 1, 0, half_dim)?;
        let x_second_half = x.narrow(rank - 1, half_dim, half_dim)?;

        let mut offset = 0;
        let mut rotated_first_half = Vec::with_capacity(3);
        let mut rotated_second_half = Vec::with_capacity(3);

        for (section_idx, &section_dim) in self.section_dims.iter().enumerate() {
            let section_pairs = section_dim / 2;
            let x1_section = x_first_half.narrow(rank - 1, offset, section_pairs)?;
            let x2_section = x_second_half.narrow(rank - 1, offset, section_pairs)?;
            let positions = all_positions[section_idx];

            // Gather cos/sin for this section's positions.
            let pos_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
            let pos_ids = DynTensor::from_vec_u32(
                pos_u32,
                &[seq_len],
                &self.cos_caches[section_idx].device(),
            )?;
            let cos = self.cos_caches[section_idx].index_select(&pos_ids, 0)?;
            let sin = self.sin_caches[section_idx].index_select(&pos_ids, 0)?;

            let (cos, sin) = if x.dtype() != cos.dtype() {
                (cos.to_dtype(x.dtype())?, sin.to_dtype(x.dtype())?)
            } else {
                (cos, sin)
            };

            let mut broadcast_shape = vec![1usize; rank - 2];
            broadcast_shape.push(seq_len);
            broadcast_shape.push(section_pairs);
            let cos_bc = cos.reshape(&broadcast_shape)?;
            let sin_bc = sin.reshape(&broadcast_shape)?;

            let y1 = x1_section
                .broadcast_mul(&cos_bc)?
                .broadcast_sub(&x2_section.broadcast_mul(&sin_bc)?)?;
            let y2 = x1_section
                .broadcast_mul(&sin_bc)?
                .broadcast_add(&x2_section.broadcast_mul(&cos_bc)?)?;

            rotated_first_half.push(y1);
            rotated_second_half.push(y2);
            offset += section_pairs;
        }

        // Reassemble the HF / Qwen layout: [T1 | H1 | W1 | T2 | H2 | W2].
        let refs: Vec<&DynTensor> = rotated_first_half
            .iter()
            .chain(rotated_second_half.iter())
            .collect();
        DynTensor::cat(&refs, rank - 1)
    }

    /// Head dimension this embedding was created for.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Section dimensions: `[temporal, height, width]`.
    #[must_use]
    pub fn section_dims(&self) -> &[usize; 3] {
        &self.section_dims
    }

    /// Maximum position value supported.
    #[must_use]
    pub fn max_position(&self) -> usize {
        self.max_position
    }
}

#[cfg(test)]
#[path = "rope_multimodal_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "rope_multimodal_integration_tests.rs"]
mod integration_tests;
