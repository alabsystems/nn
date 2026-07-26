// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HTDemucs transformer architecture constants (htdemucs_ft defaults).
//!
//! These constants define the transformer bottleneck dimensions for the
//! standard HTDemucs model. Backend-agnostic — used by both builders
//! (TensorKernelDef construction) and weight validation.
//!
//! Extracted from nn-metal as part of #860.

/// Number of transformer layers per branch.
pub const NUM_LAYERS: usize = 5;

/// Number of attention heads.
pub const NUM_HEADS: usize = 8;

/// Transformer internal dimension.
pub const TRANSFORMER_DIM: usize = 512;

/// Bottleneck channel dimension (channels_at_depth(3) = 48 * 2^3).
pub const BOTTLENECK_DIM: usize = 384;

/// FFN hidden dimension multiplier.
pub const FFN_HIDDEN_SCALE: f64 = 4.0;

/// FFN hidden dimension.
pub const FFN_HIDDEN_DIM: usize = (TRANSFORMER_DIM as f64 * FFN_HIDDEN_SCALE) as usize; // 2048

/// LayerNorm epsilon.
pub const LAYER_NORM_EPS: f32 = 1e-5;

#[cfg(test)]
#[path = "demucs_transformer_constants_tests.rs"]
mod tests;
