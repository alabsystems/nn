// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Test index for dyn_tensor/trace — consolidates 10 test modules.
//! See designs/2026-03-22-code-structure-wave16-test-index-modules.md.

#[allow(unused_imports)]
pub(crate) use super::*;

#[path = "trace_tests.rs"]
mod basic;

#[path = "trace_tests_nn.rs"]
mod layers;

#[path = "trace_tests_shape_ops.rs"]
mod shape_ops;

#[path = "trace_tests_attention.rs"]
mod attention;

#[path = "trace_tests_compound.rs"]
mod compound;

#[path = "trace_tests_lstm.rs"]
mod lstm;

#[path = "trace_op_class_tests.rs"]
mod op_class;

#[path = "trace_selection_medium_tests.rs"]
mod selection_medium;

#[path = "trace_structural_tests.rs"]
mod structural;

#[path = "traced_forward_tests.rs"]
mod traced_forward;

#[path = "trace_shape_override_tests.rs"]
mod shape_override;
