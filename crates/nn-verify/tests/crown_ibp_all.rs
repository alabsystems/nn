// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated CROWN/IBP identity and verification tests.
//!
//! Single-binary aggregator for CROWN vs IBP comparison tests.
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/crown_ibp_identity.rs"]
mod crown_ibp_identity;
