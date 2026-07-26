// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CPU-side helper functions for the Demucs transformer forward pass.
//!
//! Transpose and sinusoidal positional embedding utilities. These are pure
//! math with zero GPU backend dependency — usable from any backend or
//! from nn-verify composition tests.
//!
//! Extracted from nn-metal as part of #860.

/// Transpose a flattened `[C, T]` tensor to `[T, C]` (row-major).
pub fn transpose_ct_to_tc(data: &[f32], channels: usize, seq_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * seq_len];
    for c in 0..channels {
        for t in 0..seq_len {
            out[t * channels + c] = data[c * seq_len + t];
        }
    }
    out
}

/// Transpose a flattened `[T, C]` tensor to `[C, T]` (row-major).
pub fn transpose_tc_to_ct(data: &[f32], seq_len: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * seq_len];
    for t in 0..seq_len {
        for c in 0..channels {
            out[c * seq_len + t] = data[t * channels + c];
        }
    }
    out
}

/// Add 1D sinusoidal positional embedding in-place to a `[T, D]` tensor.
///
/// Uses the same encoding as HTDemucs: first half of D is cos, second half is sin,
/// with log-space frequencies up to max_period=10000.
pub fn add_sinusoidal_1d(data: &mut [f32], seq_len: usize, dim: usize) {
    let half = dim / 2;
    let max_period: f32 = 10000.0;

    for pos in 0..seq_len {
        let row_offset = pos * dim;
        for i in 0..half {
            let freq = (-(i as f64) * f64::from(max_period).ln() / half as f64).exp();
            let angle = (pos as f64 * freq) as f32;
            // First half: cos, second half: sin (matching Python HTDemucs convention).
            data[row_offset + i] += angle.cos();
            data[row_offset + half + i] += angle.sin();
        }
    }
}

/// Build a sinusoidal positional embedding table of shape `[seq_len, dim]`.
///
/// Same encoding as [`add_sinusoidal_1d`] but returns the table itself
/// (not added in-place). Used by GPU-resident dispatch (#1372) where the
/// table is stored as a weight tensor for element-wise GPU addition.
pub fn build_sinusoidal_table(seq_len: usize, dim: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; seq_len * dim];
    add_sinusoidal_1d(&mut table, seq_len, dim);
    table
}

#[cfg(test)]
#[path = "demucs_transformer_helpers_tests.rs"]
mod tests;
