// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated DETR composition verification tests.
//!
//! Verifies bounds propagation through the full DETR object detection
//! architecture:
//!
//! - **Encoder block:** Self-attention + FFN with ReLU (applied to CNN features)
//! - **Learned positional encoding:** Constant additive position embeddings
//! - **Decoder self-attention:** Object queries attend to each other
//! - **Decoder cross-attention:** Object queries attend to encoder features
//! - **Decoder block:** Self-attn + Cross-attn + FFN composed
//! - **Detection head:** Class logits (linear) + bbox regression (MLP + sigmoid)
//!
//! Architecture (Carion et al. 2020):
//! - Cross-attention: Q from decoder queries, K/V from encoder features
//! - DETR decoder block: Self-attention + Cross-attention + FFN
//! - Two configurations: small (d=64, heads=4) and medium (d=256, heads=8)
//!
//! Part of #3534: DETR cross-attention compose verification tests.
//! Part of #3548: DETR decoder full block compose verification.
//! Part of #3556: DETR object detection compose verification tests.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_detr_cross_attention.rs"]
mod detr_cross_attention;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_detr_decoder_block.rs"]
mod detr_decoder_block;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_detr_encoder_block.rs"]
mod detr_encoder_block;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_detr_decoder_self_attention.rs"]
mod detr_decoder_self_attention;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_detr_learned_positional_encoding.rs"]
mod detr_learned_positional_encoding;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_detr_detection_head.rs"]
mod detr_detection_head;
