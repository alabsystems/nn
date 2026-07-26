// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated GLM-4/5 composition tests.
//!
//! Combines GLM decoder sub-component verification into a single test binary
//! to reduce link-time overhead from redundant NY linkage.
//!
//! - `glm_decoder`: Self-attention (QKV+bias), fused SwiGLU FFN, decoder block, 2-block stack
//!
//! Part of #3569: GLM decoder block NY compose verification.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_glm_decoder.rs"]
mod glm_decoder;
