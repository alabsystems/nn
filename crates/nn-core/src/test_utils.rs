// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-crate test helpers for DynTensor construction and comparison.
//!
//! Re-exports from [`crate::dyn_tensor::test_helpers`] so external crates
//! can write `use nn_core::test_utils::cpu;` instead of duplicating
//! `fn cpu() -> Device { Device::Cpu }` in every test file.
//!
//! See also [`crate::test_prng`] for deterministic random data generation.

pub use crate::dyn_tensor::test_helpers::{
    approx_eq, assert_close, assert_close_scalar_f64, assert_close_with_label, cpu, make_linear,
    make_linear_seeded, make_linear_seeded_with_bias, make_linear_with_bias, t1d, t2d, tnd,
};
