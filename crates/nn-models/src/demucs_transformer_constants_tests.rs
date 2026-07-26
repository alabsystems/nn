// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::demucs_transformer_constants`].

use super::*;

#[test]
fn test_ffn_hidden_dim_matches_scale() {
    assert_eq!(
        FFN_HIDDEN_DIM,
        (TRANSFORMER_DIM as f64 * FFN_HIDDEN_SCALE) as usize
    );
    assert_eq!(FFN_HIDDEN_DIM, 2048);
}

#[test]
fn test_head_dim_divides_evenly() {
    assert_eq!(
        TRANSFORMER_DIM % NUM_HEADS,
        0,
        "TRANSFORMER_DIM must be divisible by NUM_HEADS"
    );
    let head_dim = TRANSFORMER_DIM / NUM_HEADS;
    assert_eq!(head_dim, 64);
}

#[test]
fn test_bottleneck_matches_depth3() {
    use crate::demucs_shared::channels_at_depth;
    assert_eq!(BOTTLENECK_DIM, channels_at_depth(3));
}

#[test]
fn test_layer_norm_eps_positive() {
    // Use runtime value to avoid clippy::assertions_on_constants
    let eps = LAYER_NORM_EPS;
    assert!(eps > 0.0);
    assert!(eps < 1.0);
}
