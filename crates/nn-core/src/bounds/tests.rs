// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended proof-coverage tests for IntervalBounds.
//!
//! Split into submodules to stay under 500 lines per file:
//! - `construction`: new, from_epsilon, concrete, validation, max_width, accessors
//! - `rounding`: round_for_soundness, next_down_f32/next_up_f32, NaN repair
//!
//! IBP arithmetic tests (add, mul, scale, shift) removed in #2005 —
//! arithmetic is provided by ny_tensor::BoundedTensor.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use ndarray::arr1;

#[path = "tests_construction.rs"]
mod construction;

#[path = "tests_rounding.rs"]
mod rounding;

#[path = "memory_tests.rs"]
mod memory;

#[path = "tests_ieee754_edge.rs"]
mod ieee754_edge;

#[path = "tests_ieee754_subnormal.rs"]
mod ieee754_subnormal;
