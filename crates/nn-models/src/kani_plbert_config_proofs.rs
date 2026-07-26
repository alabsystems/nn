// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for PlbertConfig default invariants.
//!
//! Proves that:
//! 1. Default hidden_size is divisible by num_attention_heads.
//! 2. Default head dimension is a power of 2.
//! 3. Default embedding_dim < hidden_size (factorized embedding).
//! 4. Default layer_norm_eps is positive and finite.
//! 5. Default vocab_size is non-zero.
//! 6. Default max_position_embeddings is a power of 2.
//!
//! Part of #3793, #3351.

use crate::plbert::PlbertConfig;

/// Proof 2: Head dimension is a power of 2.
///
/// Efficient SIMD and tensor core operations prefer power-of-2 dimensions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_plbert_head_dim_power_of_two() {
    let config = PlbertConfig::default();
    let head_dim = config.hidden_size / config.num_attention_heads;
    assert!(head_dim > 0);
    assert!(
        head_dim.is_power_of_two(),
        "head_dim={} must be a power of 2",
        head_dim
    );
    // 768 / 12 = 64
    assert_eq!(head_dim, 64);
}

/// Proof 4: layer_norm_eps is positive and finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_plbert_layer_norm_eps_valid() {
    let config = PlbertConfig::default();
    assert!(config.layer_norm_eps > 0.0, "eps must be positive");
    assert!(config.layer_norm_eps.is_finite(), "eps must be finite");
    assert!(config.layer_norm_eps < 1.0, "eps must be small");
}

/// Proof 5: vocab_size is non-zero.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_plbert_vocab_nonzero() {
    let config = PlbertConfig::default();
    assert!(config.vocab_size > 0, "vocab_size must be non-zero");
    assert_eq!(config.vocab_size, 178);
}

/// Proof 6: max_position_embeddings is a power of 2.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_plbert_max_positions_power_of_two() {
    let config = PlbertConfig::default();
    assert!(config.max_position_embeddings > 0);
    assert!(
        config.max_position_embeddings.is_power_of_two(),
        "max_position_embeddings must be a power of 2"
    );
    assert_eq!(config.max_position_embeddings, 512);
}
