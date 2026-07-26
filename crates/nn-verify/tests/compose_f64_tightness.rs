// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! f64 tightness compose tests: measure precision gap between f32 IBP/CROWN
//! bounds and f64 concrete evaluation for Linear+ReLU sequential networks.
//!
//! Part of #4316: f64 evaluation for bound tightness.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/f64_tightness_linear_relu.rs"]
mod f64_tightness_linear_relu;
