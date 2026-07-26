// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for status module tests.
//!
//! Extracted from `status_tests.rs` and `status_tests_smt.rs` to
//! deduplicate `single_input_bounds` and `scalar_output_bounds`.

use super::{InputBoundsRecord, OutputBoundsRecord, ParamInputRecord};

pub(super) fn single_input_bounds(
    lower: f32,
    upper: f32,
    constant_params: Vec<f32>,
) -> InputBoundsRecord {
    InputBoundsRecord {
        variable_inputs: vec![ParamInputRecord {
            param_index: 0,
            lower,
            upper,
        }],
        constant_params,
        input_shape: Some(vec![1]),
        input_range: Some((lower, upper)),
    }
}

pub(super) fn scalar_output_bounds(lower: f32, upper: f32) -> OutputBoundsRecord {
    OutputBoundsRecord {
        lower,
        upper,
        tensor_lower: None,
        tensor_upper: None,
        shape: None,
        is_infeasible: false,
    }
}
