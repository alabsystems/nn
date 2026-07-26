// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated regression sentinel tests.
//!
//! Combines 2 regression sentinel test files into a single test binary
//! to reduce compilation overhead (2 NY link steps → 1).
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/regression_sentinels.rs"]
mod regression_sentinels;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/regression_sentinels_ay.rs"]
mod regression_sentinels_ay;
