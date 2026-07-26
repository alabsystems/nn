// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Metal IR parity tests.
//!
//! Combines 2 Metal IR parity test files into a single test binary
//! to reduce compilation overhead (2 NY link steps → 1).
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/metal_ir_parity.rs"]
mod metal_ir_parity;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/metal_ir_parity_reduction.rs"]
mod metal_ir_parity_reduction;
