// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated ViT (Vision Transformer) encoder composition tests.
//!
//! Combines patch embedding, self-attention, FFN, and encoder block compose
//! tests into a single test binary to reduce link-time overhead from redundant
//! NY linkage.
//!
//! Architecture verified (Dosovitskiy et al. 2020):
//! - Patch embedding: Linear projection of image patches
//! - Self-attention: Q/K/V projection -> attention core -> output projection
//! - FFN sub-block: Linear -> GELU -> Linear
//! - Encoder block: LayerNorm -> MHA -> residual -> LayerNorm -> FFN -> residual
//! - 2-block encoder stack: composition of multiple encoder layers
//!
//! Part of #3527: ViT encoder NY compose verification tests.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_vit_patch_embedding.rs"]
mod vit_patch_embedding;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_vit_ffn.rs"]
mod vit_ffn;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_vit_encoder_block.rs"]
mod vit_encoder_block;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_vit_self_attention.rs"]
mod vit_self_attention;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_vit_full_encoder.rs"]
mod vit_full_encoder;
