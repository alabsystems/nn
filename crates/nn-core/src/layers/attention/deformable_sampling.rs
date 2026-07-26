// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bilinear sampling helpers for deformable attention.
//!
//! Extracted from `deformable.rs` (#1442).

/// Configuration for deformable attention.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct DeformableAttentionConfig {
    /// Model dimension (input/output size).
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of sampling points per head per reference point.
    pub num_points: usize,
    /// Number of feature levels (1 for single-scale, >1 for multi-scale).
    pub num_levels: usize,
}

impl DeformableAttentionConfig {
    /// Create a single-scale deformable attention config.
    pub fn single_scale(d_model: usize, num_heads: usize, num_points: usize) -> Self {
        Self {
            d_model,
            num_heads,
            num_points,
            num_levels: 1,
        }
    }

    /// Create a multi-scale deformable attention config.
    pub fn multi_scale(
        d_model: usize,
        num_heads: usize,
        num_points: usize,
        num_levels: usize,
    ) -> Self {
        Self {
            d_model,
            num_heads,
            num_points,
            num_levels,
        }
    }
}

/// Safely read a value from the projected value tensor.
///
/// Value layout: `[B, H*W, num_heads, head_dim]` flattened.
/// Returns 0.0 for out-of-bounds spatial coordinates (zero-padding).
pub(super) fn safe_value(
    val_flat: &[f32],
    b: usize,
    y: i64,
    x: i64,
    head: usize,
    d: usize,
    hw: usize,
    num_heads: usize,
    head_dim: usize,
    height: usize,
    width: usize,
) -> f32 {
    if y < 0 || x < 0 || y >= height as i64 || x >= width as i64 {
        return 0.0;
    }
    let spatial_idx = y as usize * width + x as usize;
    let idx = ((b * hw + spatial_idx) * num_heads + head) * head_dim + d;
    if idx < val_flat.len() {
        val_flat[idx]
    } else {
        0.0
    }
}
