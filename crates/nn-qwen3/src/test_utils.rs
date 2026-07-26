// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities for nn-qwen3.
//!
//! Provides a minimal Qwen3 configuration for unit and integration tests.

use crate::Qwen3Config;

/// Minimal Qwen3 config for tests: 2 layers, 2 heads, 2 kv heads, small dims.
///
/// head_dim is always 128 (Qwen3 constant), so hidden_size = num_heads * head_dim = 256.
#[must_use]
pub fn tiny_config() -> Qwen3Config {
    Qwen3Config {
        hidden_size: 256,
        intermediate_size: 512,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        vocab_size: 100,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        max_position_embeddings: 64,
        tie_word_embeddings: true,
        rope_scaling: None,
    }
}
