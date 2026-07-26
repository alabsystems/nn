// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Interleaved Multimodal Rotary Position Embedding (Interleaved M-ROPE).
//!
//! [`InterleavedMRoPE`] assigns modality sections to head-dimension pairs
//! in an **interleaved** pattern: pair index `i` maps to section `i % 3`,
//! cycling `[temporal, height, width, temporal, height, width, ...]`.
//!
//! This contrasts with standard (concatenated) M-ROPE
//! ([`super::MultimodalRoPE`]) which follows the Hugging Face / Qwen
//! six-block layout `[T1|H1|W1|T2|H2|W2]`.
//!
//! Required by Qwen3-VL which interleaves modality-specific frequencies
//! across the head dimension for more uniform position information mixing.
//!
//! Reference: Qwen3-VL architecture (post Qwen2.5-VL, interleaved variant).

use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

/// Configuration for interleaved multimodal RoPE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterleavedMRoPEConfig {
    /// Dimension per attention head (must be divisible by 6 —
    /// 3 sections x 2 elements per pair, with equal allocation per section).
    pub head_dim: usize,
    /// Maximum position value for any axis.
    pub max_position: usize,
    /// Frequency base (typically 1000000.0 for Qwen-VL family).
    pub base: f64,
}

/// Interleaved Multimodal Rotary Position Embedding.
///
/// For head_dim pairs indexed 0..head_dim/2, pair `i` is assigned to section
/// `i % 3` (0=temporal, 1=height, 2=width). Each pair uses the position ID
/// of its assigned section and a frequency `theta_i = base^(-2i / head_dim)`.
///
/// This interleaving distributes modality information uniformly across the
/// head dimension, rather than keeping it in the standard six-block layout.
///
/// Input shape: `[..., seq_len, head_dim]`
/// Position IDs: 3 arrays of length `seq_len` (temporal, height, width).
#[derive(Clone)]
pub struct InterleavedMRoPE {
    /// Precomputed cos values: `[max_position, pairs_per_section]` per section.
    cos_caches: [DynTensor; 3],
    /// Precomputed sin values: `[max_position, pairs_per_section]` per section.
    sin_caches: [DynTensor; 3],
    /// Full head dimension.
    head_dim: usize,
    /// Number of pairs per section (head_dim / 6 when evenly divisible).
    pairs_per_section: usize,
    /// Maximum position value.
    max_position: usize,
}

impl std::fmt::Debug for InterleavedMRoPE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InterleavedMRoPE")
            .field("head_dim", &self.head_dim)
            .field("pairs_per_section", &self.pairs_per_section)
            .field("max_position", &self.max_position)
            .finish_non_exhaustive()
    }
}

impl InterleavedMRoPE {
    /// Create a new interleaved multimodal rotary embedding.
    ///
    /// - `config`: head_dim, max_position, and frequency base
    /// - `device`: target device for cached sin/cos tables
    ///
    /// `head_dim` must be divisible by 6 (3 sections x 2 elements per pair).
    pub fn new(config: InterleavedMRoPEConfig, device: &Device) -> Result<Self> {
        let InterleavedMRoPEConfig {
            head_dim,
            max_position,
            base,
        } = config;

        if head_dim == 0 || !head_dim.is_multiple_of(6) {
            return Err(TensorError::ValueOutOfRange {
                description: "InterleavedMRoPE: head_dim must be a positive multiple of 6",
            });
        }
        if max_position == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "InterleavedMRoPE: max_position must be > 0",
            });
        }
        if max_position > u32::MAX as usize {
            return Err(TensorError::ValueOutOfRange {
                description: "InterleavedMRoPE: max_position exceeds u32::MAX",
            });
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(TensorError::ValueOutOfRange {
                description: "InterleavedMRoPE: base must be positive finite",
            });
        }

        let half_dim = head_dim / 2;
        let pairs_per_section = half_dim / 3;

        // Build per-section frequency caches.
        // For the interleaved layout, pair index `i` in the full head_dim
        // has frequency theta_i = base^(-2i / head_dim).
        // Section s owns pair indices where i % 3 == s.
        // Within section s, the j-th pair has global index i = 3*j + s.
        let mut cos_caches = Vec::with_capacity(3);
        let mut sin_caches = Vec::with_capacity(3);

        for section in 0..3usize {
            let mut cos_data = Vec::with_capacity(max_position * pairs_per_section);
            let mut sin_data = Vec::with_capacity(max_position * pairs_per_section);

            for pos in 0..max_position {
                for j in 0..pairs_per_section {
                    let global_pair_idx = 3 * j + section;
                    let exponent = (2 * global_pair_idx) as f64 / head_dim as f64;
                    let inv_freq = (1.0 / base.powf(exponent)) as f32;
                    let angle = (pos as f64 * f64::from(inv_freq)) as f32;
                    cos_data.push(angle.cos());
                    sin_data.push(angle.sin());
                }
            }

            let cos =
                DynTensor::from_vec(cos_data, &[max_position, pairs_per_section], &Device::Cpu)?
                    .to_device(device)?;
            let sin =
                DynTensor::from_vec(sin_data, &[max_position, pairs_per_section], &Device::Cpu)?
                    .to_device(device)?;

            cos_caches.push(cos);
            sin_caches.push(sin);
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
            head_dim,
            pairs_per_section,
            max_position,
        })
    }

    /// Apply interleaved M-ROPE to query and key tensors.
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

    /// Apply interleaved M-ROPE to a single tensor.
    ///
    /// Input: `[..., seq_len, head_dim]`
    ///
    /// The rotation decomposes the head_dim into pairs, assigns each pair to
    /// a section via `pair_index % 3`, gathers per-section cos/sin at the
    /// section's position IDs, applies the rotation, then reassembles.
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
                        "InterleavedMRoPE: temporal position exceeds max_position",
                        "InterleavedMRoPE: height position exceeds max_position",
                        "InterleavedMRoPE: width position exceeds max_position",
                    ];
                    return Err(TensorError::ValueOutOfRange {
                        description: descs[i],
                    });
                }
            }
        }

        // Step 1: Reshape input into pairs: [..., seq_len, half_dim, 2]
        let half_dim = self.head_dim / 2;
        let mut pairs_shape: Vec<usize> = dims[..rank - 1].to_vec();
        pairs_shape.push(half_dim);
        pairs_shape.push(2);
        let x_pairs = x.reshape(&pairs_shape)?;

        // Step 2: Extract even/odd elements: [..., seq_len, half_dim]
        let x_even = x_pairs.narrow(rank, 0, 1)?.squeeze(rank)?;
        let x_odd = x_pairs.narrow(rank, 1, 1)?.squeeze(rank)?;

        // Step 3: De-interleave into per-section even/odd tensors.
        // Section s owns pair indices where i % 3 == s, i.e., i = 3*j + s.
        // We extract these via narrow on the half_dim axis at stride 3.
        // Since DynTensor doesn't have stride-select, we use gather-by-index.
        let pps = self.pairs_per_section;

        // Build index tensors for each section.
        let section_even = self.extract_section_pairs(&x_even, rank)?;
        let section_odd = self.extract_section_pairs(&x_odd, rank)?;

        // Step 4: Gather cos/sin per section and apply rotation.
        let mut rotated_even_sections = Vec::with_capacity(3);
        let mut rotated_odd_sections = Vec::with_capacity(3);

        for (section_idx, positions) in all_positions.iter().enumerate() {
            let pos_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
            let pos_ids = DynTensor::from_vec_u32(
                pos_u32,
                &[seq_len],
                &self.cos_caches[section_idx].device(),
            )?;
            let cos = self.cos_caches[section_idx].index_select(&pos_ids, 0)?;
            let sin = self.sin_caches[section_idx].index_select(&pos_ids, 0)?;

            // Convert cos/sin dtype to match input.
            let (cos, sin) = if x.dtype() != cos.dtype() {
                (cos.to_dtype(x.dtype())?, sin.to_dtype(x.dtype())?)
            } else {
                (cos.clone(), sin.clone())
            };

            // Broadcast cos/sin: [seq_len, pps] -> [1..., seq_len, pps]
            let mut broadcast_shape = vec![1usize; rank - 2];
            broadcast_shape.push(seq_len);
            broadcast_shape.push(pps);
            let cos_bc = cos.reshape(&broadcast_shape)?;
            let sin_bc = sin.reshape(&broadcast_shape)?;

            let se = &section_even[section_idx];
            let so = &section_odd[section_idx];

            // Standard rotation per pair:
            // y_even = x_even * cos - x_odd * sin
            // y_odd  = x_even * sin + x_odd * cos
            let y_even = se
                .broadcast_mul(&cos_bc)?
                .broadcast_sub(&so.broadcast_mul(&sin_bc)?)?;
            let y_odd = se
                .broadcast_mul(&sin_bc)?
                .broadcast_add(&so.broadcast_mul(&cos_bc)?)?;

            rotated_even_sections.push(y_even);
            rotated_odd_sections.push(y_odd);
        }

        // Step 5: Re-interleave sections back into the original pair order.
        // Pair index i -> section i % 3, within-section index i / 3.
        // We need to reconstruct [..., seq_len, half_dim] from 3 sections of
        // [..., seq_len, pps] each.
        let full_even = self.reinterleave_sections(&rotated_even_sections, rank, dims)?;
        let full_odd = self.reinterleave_sections(&rotated_odd_sections, rank, dims)?;

        // Step 6: Combine even/odd back into pairs and reshape to original.
        let even_expanded = full_even.unsqueeze(rank)?;
        let odd_expanded = full_odd.unsqueeze(rank)?;
        let y_pairs = DynTensor::cat(&[&even_expanded, &odd_expanded], rank)?;
        y_pairs.reshape(dims)
    }

    /// Extract per-section pair slices from a [..., seq_len, half_dim] tensor.
    ///
    /// Returns 3 tensors, each [..., seq_len, pairs_per_section], containing
    /// the pairs at indices i where i % 3 == section.
    fn extract_section_pairs(&self, x: &DynTensor, rank: usize) -> Result<[DynTensor; 3]> {
        let pps = self.pairs_per_section;
        let device = x.device();

        // Build section index tensors: for section s, indices are [s, s+3, s+6, ...]
        let mut sections = Vec::with_capacity(3);
        for section in 0..3usize {
            let indices: Vec<u32> = (0..pps).map(|j| (3 * j + section) as u32).collect();
            let idx = DynTensor::from_vec_u32(indices, &[pps], &device)?;
            // index_select along the last dimension (rank - 1 of x, which is the
            // half_dim axis since x has shape [..., seq_len, half_dim]).
            let selected = x.index_select(&idx, rank - 1)?;
            sections.push(selected);
        }

        Ok([sections.remove(0), sections.remove(0), sections.remove(0)])
    }

    /// Re-interleave 3 section tensors back to [..., seq_len, half_dim].
    ///
    /// Inverse of `extract_section_pairs`: pair index i gets data from
    /// section i % 3 at within-section index i / 3.
    fn reinterleave_sections(
        &self,
        sections: &[DynTensor],
        rank: usize,
        _original_dims: &[usize],
    ) -> Result<DynTensor> {
        let half_dim = self.head_dim / 2;
        let pps = self.pairs_per_section;

        // Build a scatter-order by concatenating sections in interleaved order.
        // We cat along the last axis as [section0, section1, section2] giving
        // [..., seq_len, 3*pps], then use index_select to reorder.
        let refs: Vec<&DynTensor> = sections.iter().collect();
        let concatenated = DynTensor::cat(&refs, rank - 1)?;

        // Build the reorder index: for output pair index i, the data is at
        // position (i % 3) * pps + (i / 3) in the concatenated tensor.
        let reorder_indices: Vec<u32> = (0..half_dim)
            .map(|i| {
                let section = i % 3;
                let within = i / 3;
                (section * pps + within) as u32
            })
            .collect();
        let idx = DynTensor::from_vec_u32(reorder_indices, &[half_dim], &concatenated.device())?;
        concatenated.index_select(&idx, rank - 1)
    }

    /// Head dimension this embedding was created for.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Number of pairs per section (head_dim / 6).
    #[must_use]
    pub fn pairs_per_section(&self) -> usize {
        self.pairs_per_section
    }

    /// Maximum position value supported.
    #[must_use]
    pub fn max_position(&self) -> usize {
        self.max_position
    }
}

#[cfg(test)]
#[path = "interleaved_mrope_tests.rs"]
mod tests;
