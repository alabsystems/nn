// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized gather and scatter-add operations.
//!
//! `gather_1d` gathers elements from `input` by index:
//!   `output[i] = input[indices[i]]`
//!
//! `scatter_add_1d` accumulates elements into bins:
//!   `output[indices[i]] += input[i]`
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.
//! The SIMD paths process indices in chunks, using wide loads/stores
//! for the gathered values where the index pattern permits.

use std::fmt;

/// Block size for gather/scatter tiling. Indices are processed in blocks
/// of this size to improve cache locality on large index arrays.
pub const GATHER_BLOCK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during gather/scatter operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatherError {
    /// An index was out of bounds for the input.
    IndexOutOfBounds {
        index: usize,
        input_len: usize,
        position: usize,
    },
    /// The output buffer has the wrong length.
    OutputLengthMismatch { expected: usize, actual: usize },
}

impl fmt::Display for GatherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfBounds {
                index,
                input_len,
                position,
            } => write!(
                f,
                "gather index {index} out of bounds for input length {input_len} \
                 at position {position}"
            ),
            Self::OutputLengthMismatch { expected, actual } => write!(
                f,
                "output buffer length {actual} does not match expected {expected}"
            ),
        }
    }
}

impl std::error::Error for GatherError {}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

/// Validate gather inputs: all indices must be in bounds, output must match
/// indices length.
fn validate_gather_inputs(
    input: &[f32],
    indices: &[u32],
    output: &[f32],
) -> Result<(), GatherError> {
    if output.len() != indices.len() {
        return Err(GatherError::OutputLengthMismatch {
            expected: indices.len(),
            actual: output.len(),
        });
    }
    for (pos, &idx) in indices.iter().enumerate() {
        let idx = idx as usize;
        if idx >= input.len() {
            return Err(GatherError::IndexOutOfBounds {
                index: idx,
                input_len: input.len(),
                position: pos,
            });
        }
    }
    Ok(())
}

/// Validate scatter-add inputs: all indices must be within `dim_size`,
/// output must have length `dim_size`.
fn validate_scatter_inputs(
    input: &[f32],
    indices: &[u32],
    dim_size: usize,
    output: &[f32],
) -> Result<(), GatherError> {
    if input.len() != indices.len() {
        return Err(GatherError::OutputLengthMismatch {
            expected: indices.len(),
            actual: input.len(),
        });
    }
    if output.len() != dim_size {
        return Err(GatherError::OutputLengthMismatch {
            expected: dim_size,
            actual: output.len(),
        });
    }
    for (pos, &idx) in indices.iter().enumerate() {
        let idx = idx as usize;
        if idx >= dim_size {
            return Err(GatherError::IndexOutOfBounds {
                index: idx,
                input_len: dim_size,
                position: pos,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar fallback — gather_1d
// ---------------------------------------------------------------------------

/// Scalar gather: `output[i] = input[indices[i]]`.
///
/// `input`: source data.
/// `indices`: array of indices into `input` (u32).
/// `output`: pre-allocated output buffer, same length as `indices`.
///
/// Returns `Ok(())` on success, or an error if any index is out of bounds
/// or the output length does not match.
pub fn gather_1d_scalar(
    input: &[f32],
    indices: &[u32],
    output: &mut [f32],
) -> Result<(), GatherError> {
    validate_gather_inputs(input, indices, output)?;

    for (i, &idx) in indices.iter().enumerate() {
        output[i] = input[idx as usize];
    }

    Ok(())
}

/// Scalar reference implementation for differential testing.
///
/// Returns a newly-allocated output vector.
pub fn gather_1d_reference(input: &[f32], indices: &[u32]) -> Result<Vec<f32>, GatherError> {
    let mut output = vec![0.0f32; indices.len()];
    gather_1d_scalar(input, indices, &mut output)?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — gather_1d
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon_gather {
    use super::*;
    use std::arch::aarch64::*;

    /// NEON-accelerated gather using 4-wide vectorized stores.
    ///
    /// Since gather is inherently index-dependent (non-contiguous reads),
    /// we scalar-load each element but store results in 4-wide NEON chunks.
    pub(super) fn gather_1d_neon(
        input: &[f32],
        indices: &[u32],
        output: &mut [f32],
    ) -> Result<(), GatherError> {
        validate_gather_inputs(input, indices, output)?;

        let len = indices.len();
        let chunks = len / 4;
        let remainder = len % 4;
        let tail_start = chunks * 4;

        // SAFETY: aarch64 NEON is always available. Bounded stores within
        // the validated output slice. Input reads validated above.
        unsafe {
            for c in 0..chunks {
                let base = c * 4;
                let v0 = input[indices[base] as usize];
                let v1 = input[indices[base + 1] as usize];
                let v2 = input[indices[base + 2] as usize];
                let v3 = input[indices[base + 3] as usize];
                let mut vec = vdupq_n_f32(0.0);
                vec = vsetq_lane_f32::<0>(v0, vec);
                vec = vsetq_lane_f32::<1>(v1, vec);
                vec = vsetq_lane_f32::<2>(v2, vec);
                vec = vsetq_lane_f32::<3>(v3, vec);
                vst1q_f32(output.as_mut_ptr().add(base), vec);
            }
            for i in 0..remainder {
                output[tail_start + i] = input[indices[tail_start + i] as usize];
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — gather_1d
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2_gather {
    use super::*;
    use std::arch::x86_64::*;

    /// AVX2-accelerated gather using 8-wide vectorized stores.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn gather_1d_avx2(
        input: &[f32],
        indices: &[u32],
        output: &mut [f32],
    ) -> Result<(), GatherError> {
        validate_gather_inputs(input, indices, output)?;

        let len = indices.len();
        let chunks = len / 8;
        let remainder = len % 8;
        let tail_start = chunks * 8;

        for c in 0..chunks {
            let base = c * 8;
            // Scalar gather into a temp buffer, then SIMD store.
            let buf: [f32; 8] = [
                input[indices[base] as usize],
                input[indices[base + 1] as usize],
                input[indices[base + 2] as usize],
                input[indices[base + 3] as usize],
                input[indices[base + 4] as usize],
                input[indices[base + 5] as usize],
                input[indices[base + 6] as usize],
                input[indices[base + 7] as usize],
            ];
            let v = _mm256_loadu_ps(buf.as_ptr());
            _mm256_storeu_ps(output.as_mut_ptr().add(base), v);
        }
        for i in 0..remainder {
            output[tail_start + i] = input[indices[tail_start + i] as usize];
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public dispatch — gather_1d
// ---------------------------------------------------------------------------

/// Gather elements by index with automatic SIMD dispatch.
///
/// `output[i] = input[indices[i]]` for all `i`.
///
/// Auto-dispatches to NEON (aarch64), AVX2 (x86_64), or scalar fallback.
///
/// # Arguments
/// * `input` — source data array
/// * `indices` — array of indices into `input` (u32)
/// * `output` — pre-allocated output buffer; length must equal `indices.len()`
///
/// # Errors
/// Returns `GatherError` if any index is out of bounds or output length
/// does not match.
pub fn gather_1d(input: &[f32], indices: &[u32], output: &mut [f32]) -> Result<(), GatherError> {
    #[cfg(target_arch = "aarch64")]
    {
        return neon_gather::gather_1d_neon(input, indices, output);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            return unsafe { avx2_gather::gather_1d_avx2(input, indices, output) };
        }
    }

    #[allow(unreachable_code)]
    gather_1d_scalar(input, indices, output)
}

// ---------------------------------------------------------------------------
// Scalar fallback — scatter_add_1d
// ---------------------------------------------------------------------------

/// Scalar scatter-add: `output[indices[i]] += input[i]`.
///
/// `input`: values to scatter.
/// `indices`: target indices (u32), one per input element.
/// `dim_size`: size of the output dimension.
/// `output`: pre-allocated output buffer of length `dim_size`, typically
///           zero-initialized by the caller.
///
/// Returns `Ok(())` on success, or an error if any index is out of bounds
/// or buffer sizes are inconsistent.
pub fn scatter_add_1d_scalar(
    input: &[f32],
    indices: &[u32],
    dim_size: usize,
    output: &mut [f32],
) -> Result<(), GatherError> {
    validate_scatter_inputs(input, indices, dim_size, output)?;

    for (i, &idx) in indices.iter().enumerate() {
        output[idx as usize] += input[i];
    }

    Ok(())
}

/// Scalar reference implementation for scatter-add differential testing.
///
/// Returns a newly-allocated zero-initialized output vector of length
/// `dim_size` with values scattered in.
pub fn scatter_add_1d_reference(
    input: &[f32],
    indices: &[u32],
    dim_size: usize,
) -> Result<Vec<f32>, GatherError> {
    let mut output = vec![0.0f32; dim_size];
    scatter_add_1d_scalar(input, indices, dim_size, &mut output)?;
    Ok(output)
}

// ---------------------------------------------------------------------------
// Public dispatch — scatter_add_1d
// ---------------------------------------------------------------------------

/// Scatter-add with automatic dispatch.
///
/// `output[indices[i]] += input[i]` for all `i`.
///
/// Scatter-add is inherently serial (multiple indices may target the same
/// output bin), so SIMD acceleration is limited. The primary path is scalar.
///
/// # Arguments
/// * `input` — values to scatter; length must equal `indices.len()`
/// * `indices` — target indices (u32), one per input element
/// * `dim_size` — size of the output dimension
/// * `output` — pre-allocated output buffer of length `dim_size`
///
/// # Errors
/// Returns `GatherError` if any index is out of bounds or buffer sizes
/// are inconsistent.
pub fn scatter_add_1d(
    input: &[f32],
    indices: &[u32],
    dim_size: usize,
    output: &mut [f32],
) -> Result<(), GatherError> {
    // Scatter-add has data-dependent writes (potential aliasing), so SIMD
    // does not help. Use the scalar path for correctness.
    scatter_add_1d_scalar(input, indices, dim_size, output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// The test module for this file is included once at the crate root in
// `lib.rs` (`mod simd_gather_tests`), matching the pattern used for the other
// `*_tests.rs` files. It is intentionally not re-declared here to avoid
// compiling/running the same tests twice.
