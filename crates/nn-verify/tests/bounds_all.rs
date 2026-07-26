// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated bounds test binary.
//!
//! Combines bridge, contract, and boundary tests into a single test binary
//! to reduce compilation overhead. Part of #1982 test consolidation.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[path = "helpers/bounds_bridge_tests.rs"]
mod bounds_bridge;

#[path = "helpers/bounds_contract_tests.rs"]
mod bounds_contract;

#[path = "helpers/bounds_contract_boundary_tests.rs"]
mod bounds_contract_boundary;
