// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized embedding lookup.
//!
//! Given a weight table `[vocab_size, embed_dim]` and indices `[batch]`,
//! produces output `[batch, embed_dim]` by gathering the corresponding rows.
//!
//! The core operation is a strided memory copy (gather), which benefits from
//! wide SIMD loads/stores on large embedding dimensions (256, 512, 768, 1024).
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

use std::fmt;

/// Block size for embedding lookup tiling. Rows are processed in blocks
/// of this size to improve cache locality on large batch lookups.
pub const EMBEDDING_BLOCK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during embedding lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingError {
    /// An index was out of bounds for the vocabulary.
    IndexOutOfBounds {
        index: usize,
        vocab_size: usize,
        batch_position: usize,
    },
    /// The weight table length is not a multiple of `embed_dim`.
    InvalidWeightShape { weight_len: usize, embed_dim: usize },
    /// The output buffer has the wrong length.
    OutputLengthMismatch { expected: usize, actual: usize },
    /// Embedding dimension is zero.
    ZeroEmbedDim,
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfBounds {
                index,
                vocab_size,
                batch_position,
            } => write!(
                f,
                "embedding index {index} out of bounds for vocab_size {vocab_size} \
                 at batch position {batch_position}"
            ),
            Self::InvalidWeightShape {
                weight_len,
                embed_dim,
            } => write!(
                f,
                "weight table length {weight_len} is not a multiple of embed_dim {embed_dim}"
            ),
            Self::OutputLengthMismatch { expected, actual } => write!(
                f,
                "output buffer length {actual} does not match expected {expected}"
            ),
            Self::ZeroEmbedDim => write!(f, "embed_dim must be > 0"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

// ---------------------------------------------------------------------------
// Input validation (shared across all paths)
// ---------------------------------------------------------------------------

/// Validate inputs and return `vocab_size`.
fn validate_inputs(
    weights: &[f32],
    indices: &[u32],
    output: &[f32],
    embed_dim: usize,
) -> Result<usize, EmbeddingError> {
    if embed_dim == 0 {
        return Err(EmbeddingError::ZeroEmbedDim);
    }
    if !weights.len().is_multiple_of(embed_dim) {
        return Err(EmbeddingError::InvalidWeightShape {
            weight_len: weights.len(),
            embed_dim,
        });
    }
    let vocab_size = weights.len() / embed_dim;
    let expected_out = indices.len() * embed_dim;
    if output.len() != expected_out {
        return Err(EmbeddingError::OutputLengthMismatch {
            expected: expected_out,
            actual: output.len(),
        });
    }
    // Validate all indices before any work (fail-fast).
    for (batch_pos, &idx) in indices.iter().enumerate() {
        let idx = idx as usize;
        if idx >= vocab_size {
            return Err(EmbeddingError::IndexOutOfBounds {
                index: idx,
                vocab_size,
                batch_position: batch_pos,
            });
        }
    }
    Ok(vocab_size)
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Scalar embedding lookup: copies rows from `weights` into `output`.
///
/// `weights`: `[vocab_size, embed_dim]` row-major weight table.
/// `indices`: `[batch]` array of vocabulary indices (u32).
/// `output`:  `[batch, embed_dim]` pre-allocated output buffer.
/// `embed_dim`: dimensionality of each embedding vector.
///
/// Returns `Ok(())` on success, or an error if any index is out of bounds
/// or the buffer sizes are inconsistent.
pub fn embedding_scalar(
    weights: &[f32],
    indices: &[u32],
    output: &mut [f32],
    embed_dim: usize,
) -> Result<(), EmbeddingError> {
    validate_inputs(weights, indices, output, embed_dim)?;

    for (batch_pos, &idx) in indices.iter().enumerate() {
        let src_start = (idx as usize) * embed_dim;
        let dst_start = batch_pos * embed_dim;
        output[dst_start..dst_start + embed_dim]
            .copy_from_slice(&weights[src_start..src_start + embed_dim]);
    }

    Ok(())
}

/// CPU reference implementation for verification.
///
/// Returns a newly-allocated `Vec<f32>` of shape `[batch, embed_dim]`.
/// Intended as the ground-truth for differential testing against SIMD
/// and GPU paths.
pub fn embedding_reference(
    weights: &[f32],
    indices: &[u32],
    embed_dim: usize,
) -> Result<Vec<f32>, EmbeddingError> {
    let mut output = vec![0.0f32; indices.len() * embed_dim];
    embedding_scalar(weights, indices, &mut output, embed_dim)?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// NEON-accelerated embedding lookup using 128-bit wide loads/stores.
    ///
    /// Copies 4 floats per iteration for the aligned portion, with a scalar
    /// tail for the remainder.
    pub(super) fn embedding_neon(
        weights: &[f32],
        indices: &[u32],
        output: &mut [f32],
        embed_dim: usize,
    ) -> Result<(), EmbeddingError> {
        validate_inputs(weights, indices, output, embed_dim)?;

        let chunks = embed_dim / 4;
        let remainder = embed_dim % 4;
        let tail_start = chunks * 4;

        for (batch_pos, &idx) in indices.iter().enumerate() {
            let src_start = (idx as usize) * embed_dim;
            let dst_start = batch_pos * embed_dim;
            let src = &weights[src_start..];
            let dst = &mut output[dst_start..];

            // SAFETY: aarch64 NEON is always available. Bounded loads/stores
            // within the validated slice regions.
            unsafe {
                for c in 0..chunks {
                    let offset = c * 4;
                    let v = vld1q_f32(src.as_ptr().add(offset));
                    vst1q_f32(dst.as_mut_ptr().add(offset), v);
                }
                // Scalar tail.
                dst[tail_start..tail_start + remainder]
                    .copy_from_slice(&src[tail_start..tail_start + remainder]);
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// AVX2-accelerated embedding lookup using 256-bit wide loads/stores.
    ///
    /// Copies 8 floats per iteration for the aligned portion, with a scalar
    /// tail for the remainder.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available (`is_x86_feature_detected!("avx2")`).
    #[target_feature(enable = "avx2")]
    pub unsafe fn embedding_avx2(
        weights: &[f32],
        indices: &[u32],
        output: &mut [f32],
        embed_dim: usize,
    ) -> Result<(), EmbeddingError> {
        validate_inputs(weights, indices, output, embed_dim)?;

        let chunks = embed_dim / 8;
        let remainder = embed_dim % 8;
        let tail_start = chunks * 8;

        for (batch_pos, &idx) in indices.iter().enumerate() {
            let src_start = (idx as usize) * embed_dim;
            let dst_start = batch_pos * embed_dim;
            let src = &weights[src_start..];
            let dst = &mut output[dst_start..];

            for c in 0..chunks {
                let offset = c * 8;
                let v = _mm256_loadu_ps(src.as_ptr().add(offset));
                _mm256_storeu_ps(dst.as_mut_ptr().add(offset), v);
            }
            // Scalar tail.
            for i in 0..remainder {
                dst[tail_start + i] = src[tail_start + i];
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// Embedding lookup with automatic SIMD dispatch.
///
/// Given a weight table `[vocab_size, embed_dim]` and indices `[batch]`,
/// writes the gathered rows into `output` `[batch, embed_dim]`.
///
/// Auto-dispatches to NEON (aarch64), AVX2 (x86_64), or scalar fallback.
///
/// # Arguments
/// * `weights` — flattened weight table; length must be `vocab_size * embed_dim`
/// * `indices` — batch of vocabulary indices (u32)
/// * `output` — pre-allocated output buffer; length must be `batch * embed_dim`
/// * `embed_dim` — dimensionality of each embedding vector
///
/// # Errors
/// Returns `EmbeddingError` if any index is out of bounds, buffer sizes are
/// inconsistent, or `embed_dim` is zero.
pub fn embedding(
    weights: &[f32],
    indices: &[u32],
    output: &mut [f32],
    embed_dim: usize,
) -> Result<(), EmbeddingError> {
    #[cfg(target_arch = "aarch64")]
    {
        return neon::embedding_neon(weights, indices, output, embed_dim);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            return unsafe { avx2::embedding_avx2(weights, indices, output, embed_dim) };
        }
    }

    #[allow(unreachable_code)]
    embedding_scalar(weights, indices, output, embed_dim)
}

/// Embedding lookup returning a newly-allocated output vector.
///
/// Convenience wrapper over [`embedding`] that allocates the output buffer.
///
/// # Arguments
/// * `weights` — flattened weight table `[vocab_size, embed_dim]`
/// * `indices` — batch of vocabulary indices (u32)
/// * `embed_dim` — dimensionality of each embedding vector
///
/// # Errors
/// Returns `EmbeddingError` on invalid inputs (see [`embedding`]).
pub fn embedding_lookup(
    weights: &[f32],
    indices: &[u32],
    embed_dim: usize,
) -> Result<Vec<f32>, EmbeddingError> {
    if embed_dim == 0 {
        return Err(EmbeddingError::ZeroEmbedDim);
    }
    let mut output = vec![0.0f32; indices.len() * embed_dim];
    embedding(weights, indices, &mut output, embed_dim)?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_embedding_tests.rs"]
mod simd_embedding_tests;
