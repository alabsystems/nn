// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Test index for trace_compile — consolidates 11 test modules.
//! See designs/2026-03-22-code-structure-wave16-test-index-modules.md.

#[allow(unused_imports)]
pub(crate) use super::*;

// Re-export private submodule for attention_tests and sdpa_causal_tests.
#[allow(unused_imports)]
pub(crate) use super::trace_compile_attention;

#[path = "trace_compile_tests.rs"]
mod basic;

#[path = "trace_compile_tests_ops2.rs"]
mod ops2;

#[path = "trace_compile_prover_tests.rs"]
mod prover;

#[path = "trace_compile_spatial_tests.rs"]
mod spatial;

#[path = "trace_compile_identity_tests.rs"]
mod identity;

#[path = "trace_compile_constant_tests.rs"]
mod constant;

#[path = "trace_compile_tests_reduce.rs"]
mod reduce;

#[path = "trace_compile_tests_conv.rs"]
mod conv;

#[path = "trace_compile_attention_tests.rs"]
mod attention;

#[path = "trace_compile_sdpa_causal_tests.rs"]
mod sdpa_causal;

#[path = "trace_compile_tests_adain.rs"]
mod adain;
