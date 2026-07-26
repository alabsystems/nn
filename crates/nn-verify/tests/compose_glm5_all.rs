// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated GLM-5 (nn-glm5 crate) composition tests.
//!
//! Combines GLM-5 decoder sub-component verification into a single test binary
//! to reduce link-time overhead from redundant NY linkage.
//!
//! - `glm5_decoder`: Self-attention (QKV+bias), fused SwiGLU FFN, decoder block,
//!   2-block stack, RMSNorm, and full embedding-to-logits pipeline.
//!
//! - `glm5_deep`: Deep composition tests with Conservative NormBoundsMode
//!   (targeting Sound soundness), residual widening analysis, tight-input
//!   CROWN precision, post-norm + LM head, and full pipeline with softmax.
//!
//! Part of verification dashboard completeness for the nn-glm5 model crate.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_glm5_decoder.rs"]
mod glm5_decoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_glm5_deep.rs"]
mod glm5_deep;
