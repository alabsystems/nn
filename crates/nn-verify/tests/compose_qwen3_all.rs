// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Qwen3 composition tests.
//!
//! Combines Qwen3 sub-component verification into a single test binary
//! to reduce link-time overhead from redundant NY linkage.
//!
//! - `qwen3_attention`: RoPE, GQA, combined attention, SwiGLU MLP
//! - `qwen3_decoder_pipeline`: Decoder block decomposition (RMSNorm, self-attn,
//!   MLP, single block, post-norm, 2-block stack, residual analysis)
//! - `qwen3_decoder_text`: TEXT-ONLY decoder (10 tests): RMSNorm, SwiGLU, GQA,
//!   RoPE, decoder layer, 2-layer stack, LM head, token generation (softmax
//!   bounded [0,1]), KV-cache attention, full pipeline (embedding -> layers -> LM head)
//!
//! Note: `qwen3_decoder`, `qwen3_rope`, `qwen3_gqa` are included in
//! `compose_transformer_all.rs` to consolidate transformer-family tests.
//!
//! - `qwen3_depth`: Deepened verification (embedding+RoPE, RMSNorm+SwiGLU
//!   Conservative, 3-layer stack, MoE forward, QK-Norm attention, decoder-to-logit)
//!
//! - `qwen3_qk_norm`: QK-Norm attention verification infrastructure (#2951).
//!   Per-head RMSNorm on Q/K with structurally accurate reshape-norm-reshape
//!   pattern, CROWN gap documentation (blocked on NY#3172), and
//!   bounds tightening comparison with/without QK-Norm.
//!
//! Part of #3560: Qwen3 RoPE + GQA NY compose verification.
//! Part of #3588: Compose verification for Qwen3 decoder block.
//! Part of #3942: Qwen3 decoder compose verification tests.
//! Part of #4280: Deepen Qwen3 NY compose verification.
//! Part of #2951: QK-Norm attention verification for Qwen3.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_qwen3_attention.rs"]
mod qwen3_attention;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_qwen3_decoder_pipeline.rs"]
mod qwen3_decoder_pipeline;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_qwen3_decoder_text.rs"]
mod qwen3_decoder_text;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_qwen3_deep.rs"]
mod qwen3_deep;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_qwen3_depth.rs"]
mod qwen3_depth;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_qwen3_qk_norm.rs"]
mod qwen3_qk_norm;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_qwen3_mlp_bounds.rs"]
mod compose_qwen3_mlp_bounds;
