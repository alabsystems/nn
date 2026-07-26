// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated verification pipeline and infrastructure tests.
//!
//! Combines 6 verification/infrastructure test files into a single test binary
//! to reduce compilation overhead (6 NY link steps → 1).
//!
//! - `snake_verify`: Snake kernel verification and status recording
//! - `kernel_pipeline_verify`: Kernel pipeline verification
//! - `structural_contracts`: Structural contract verification
//! - `bound_widening_measurement`: Bounds widening measurement
//! - `zonotope_parallel_mha`: Zonotope parallel multi-head attention
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/snake_verify.rs"]
mod snake_verify;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/kernel_pipeline_verify.rs"]
mod kernel_pipeline_verify;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/structural_contracts.rs"]
mod structural_contracts;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/bound_widening_measurement.rs"]
mod bound_widening_measurement;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/zonotope_parallel_mha.rs"]
mod zonotope_parallel_mha;
