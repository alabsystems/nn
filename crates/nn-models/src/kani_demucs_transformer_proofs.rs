// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for demucs_transformer_constants relationships.
//!
//! Proves that:
//! 1. FFN_HIDDEN_DIM == TRANSFORMER_DIM * FFN_HIDDEN_SCALE.
//! 2. TRANSFORMER_DIM is divisible by NUM_HEADS.
//! 3. Head dimension (TRANSFORMER_DIM / NUM_HEADS) is a power of 2.
//! 4. BOTTLENECK_DIM matches channels_at_depth(3).
//! 5. LAYER_NORM_EPS is positive and finite.
//!
//! Part of #3793, #3351.

use crate::demucs_transformer_constants::*;

/// Proof 1: FFN_HIDDEN_DIM == TRANSFORMER_DIM * FFN_HIDDEN_SCALE.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_ffn_hidden_dim_matches_scale() {
    let expected = (TRANSFORMER_DIM as f64 * FFN_HIDDEN_SCALE) as usize;
    assert_eq!(FFN_HIDDEN_DIM, expected);
    // Verify concrete value: 512 * 4.0 = 2048
    assert_eq!(FFN_HIDDEN_DIM, 2048);
}

/// Proof 2: TRANSFORMER_DIM is divisible by NUM_HEADS.
///
/// Multi-head attention requires even splitting of the hidden dimension
/// across heads.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_transformer_dim_divisible_by_heads() {
    assert!(NUM_HEADS > 0, "must have at least one attention head");
    assert_eq!(
        TRANSFORMER_DIM % NUM_HEADS,
        0,
        "TRANSFORMER_DIM must be divisible by NUM_HEADS"
    );
}

/// Proof 3: Head dimension is a power of 2.
///
/// Power-of-2 head dims enable efficient SIMD and tensor core operations.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_head_dim_is_power_of_two() {
    let head_dim = TRANSFORMER_DIM / NUM_HEADS;
    assert!(head_dim > 0);
    assert!(
        head_dim.is_power_of_two(),
        "head_dim={} must be a power of 2",
        head_dim
    );
    // Verify concrete value: 512 / 8 = 64
    assert_eq!(head_dim, 64);
}

/// Proof 4: BOTTLENECK_DIM matches the HTDemucs architecture formula.
///
/// channels_at_depth(3) = 48 * 2^3 = 384 = BOTTLENECK_DIM.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_bottleneck_dim_value() {
    assert_eq!(BOTTLENECK_DIM, 384);
    // 48 * 8 = 384
    assert_eq!(crate::demucs_shared::BASE_CHANNELS * 8, BOTTLENECK_DIM);
}

/// Proof 5: LAYER_NORM_EPS is positive and finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_layer_norm_eps_valid() {
    assert!(LAYER_NORM_EPS > 0.0, "eps must be positive");
    assert!(LAYER_NORM_EPS.is_finite(), "eps must be finite");
    assert!(LAYER_NORM_EPS < 1.0, "eps must be small");
}
