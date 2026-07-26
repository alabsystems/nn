// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated transformer/attention composition tests.
//!
//! Combines 9 transformer and attention test files into a single test binary
//! to reduce compilation overhead (9 NY link steps → 1).
//!
//! - `transformer_block`: Transformer block (MHA + FFN) composition
//! - `transformer_layer`: Transformer layer composition
//! - `multi_head_attention`: MHA composite builder (IBP + CROWN)
//! - `multi_head_causal`: Multi-head causal attention composition
//! - `qwen3_decoder`: Qwen3 decoder composition verification
//! - `qwen3_rope`: Qwen3 RoPE (Rotary Position Embedding) composition
//! - `qwen3_gqa`: Qwen3 GQA (Grouped Query Attention) composition
//! - `sliding_window_attention`: Sliding window attention (Mistral/LongNet) composition
//! - `rotary_embedding_2d`: 2D rotary position embedding (Qwen2-VL) composition
//!
//! Part of #1982, #3560, #3563.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_transformer_block.rs"]
mod transformer_block;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_transformer_layer.rs"]
mod transformer_layer;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_multi_head_attention.rs"]
mod multi_head_attention;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_multi_head_causal_attention.rs"]
mod multi_head_causal;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_qwen3_decoder.rs"]
mod qwen3_decoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_qwen3_rope.rs"]
mod qwen3_rope;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_qwen3_gqa.rs"]
mod qwen3_gqa;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_sliding_window_attention.rs"]
mod sliding_window_attention;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_rotary_embedding_2d.rs"]
mod rotary_embedding_2d;
