#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated nn-core integration tests (3 → 1 binary).

#[allow(dead_code, unreachable_pub)]
#[path = "core/perf_proof_tests.rs"]
mod perf_proof_tests;

#[allow(dead_code, unreachable_pub)]
#[path = "core/perf_proof_norm_optimizer.rs"]
mod perf_proof_norm_optimizer;

#[allow(dead_code, unreachable_pub)]
#[path = "core/trace_suppression_tests.rs"]
mod trace_suppression_tests;

#[allow(dead_code, unreachable_pub)]
#[path = "core/trace_dtype_nn_tests.rs"]
mod trace_dtype_nn_tests;
