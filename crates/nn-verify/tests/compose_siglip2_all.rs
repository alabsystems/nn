// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated SigLIP2 vision encoder composition tests.
//!
//! Combines sub-block and end-to-end tests into a single test binary to
//! reduce link-time overhead from redundant NY linkage.
//!
//! Sub-blocks (compose_siglip2.rs):
//! - Patch embedding: Conv2d(kernel=stride=P) -> reshape -> transpose
//! - Position embedding: variable + constant addition
//! - SiGLU FFN: Linear -> SiLU gate * Linear up -> Linear down
//! - Full transformer block: LayerNorm -> MHA -> residual -> LayerNorm -> SiGLU FFN -> residual
//!
//! End-to-end (compose_siglip2_e2e.rs):
//! - patch_embed + pos_embed -> N x encoder_block -> head_proj
//!
//! SigLIP2 architecture (Zhai et al. 2023):
//! - Uses SiGLU (SiLU-gated) FFN instead of GELU
//! - Standard bidirectional self-attention (vision encoder, not causal)
//! - Pre-norm transformer with LayerNorm
//!
//! Pipeline (compose_siglip2_pipeline.rs — 25 tests):
//! - Linear patch projection, post-LayerNorm, mean pooling
//! - Multi-block stacking (3 blocks), attention sub-block isolation
//! - Narrow-bounds SiGLU FFN, pos_embed + block composition
//! - Full pipeline with post-norm + mean pool, CROWN tightness analysis
//! - Head projection sub-block, SiGLU FFN with residual (LN + SiGLU + skip)
//! - Two-block widening analysis, CROWN for attention/SiGLU/mean_pool/full pipeline
//! - ViT compose blocks (hidden=8, seq=4, heads=2): patch_embed, single_vit_block,
//!   multi_block_stack with IBP/CROWN and widening analysis
//!
//! Part of #3540, #3583: SigLIP2 NY compose bounds.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_siglip2.rs"]
mod siglip2;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_siglip2_e2e.rs"]
mod siglip2_e2e;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_siglip2_pipeline.rs"]
mod siglip2_pipeline;
