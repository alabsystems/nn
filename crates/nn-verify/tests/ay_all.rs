// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated ay SMT verification tests.
//!
//! Combines 2 ay test files into a single test binary
//! to reduce compilation overhead (2 NY link steps → 1).
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_api_compat.rs"]
mod ay_api_compat;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_smt_verify.rs"]
mod ay_smt_verify;
