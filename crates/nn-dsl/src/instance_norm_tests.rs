// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for InstanceNorm K2 tensor kernel (Phase E of #31).
//!
//! Split into two modules for the 500-line limit (#462):
//! - core: IR validation, shapes, pretty-print, decomposed codegen, ref basic tests
//! - edge: numerical stability, error handling, precision boundary, non-finite input

use super::*;

#[path = "instance_norm_tests_core.rs"]
mod core;

#[path = "instance_norm_tests_edge.rs"]
mod edge;
