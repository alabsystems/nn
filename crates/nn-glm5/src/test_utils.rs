// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities for nn-glm5.

use crate::Glm5Config;

/// Minimal GLM-5 config for tests: 2 layers, 4 heads, 2 kv groups, small dims.
///
/// head_dim (kv_channels) = 64, so hidden_size = num_heads * head_dim = 256.
/// ffn_hidden_size = 512 (SwiGLU: dense_h_to_4h outputs 1024 = 512 * 2).
#[must_use]
pub fn tiny_config() -> Glm5Config {
    Glm5Config {
        hidden_size: 256,
        ffn_hidden_size: 512,
        num_layers: 2,
        num_attention_heads: 4,
        multi_query_group_num: 2,
        padded_vocab_size: 100,
        kv_channels: 64,
        layernorm_epsilon: 1e-5,
        seq_length: 64,
        rmsnorm: true,
        add_qkv_bias: true,
        add_bias_linear: false,
        rope_theta: 10_000.0,
    }
}
