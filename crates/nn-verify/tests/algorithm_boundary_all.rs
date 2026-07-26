// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated algorithm boundary tests.
//!
//! Combines 2 algorithm boundary test files into a single test binary
//! to reduce compilation overhead (2 NY link steps → 1).
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/algorithm_boundary.rs"]
mod algorithm_boundary;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/algorithm_boundary_edge.rs"]
mod algorithm_boundary_edge;
