// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Umbrella binary for cross-attention verification tests.
//!
//! Consolidates 4 former standalone binaries (31 tests total):
//! - compose_cross_attention_asymmetric (9 tests)
//! - compose_cross_attention_block (5 tests)
//! - compose_cross_attention_causal (11 tests)
//! - compose_cross_attention_monotonicity (6 tests)
//!
//! Part of #1982: test binary consolidation.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[path = "helpers/asymmetric_attention.rs"]
mod asym;

#[path = "helpers/causal_attention.rs"]
mod causal;

#[path = "helpers/cross_attention_asymmetric_tests.rs"]
mod asymmetric_tests;

#[path = "helpers/cross_attention_block_tests.rs"]
mod block_tests;

#[path = "helpers/cross_attention_causal_tests.rs"]
mod causal_tests;

#[path = "helpers/cross_attention_monotonicity_tests.rs"]
mod monotonicity_tests;
