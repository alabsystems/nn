// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verification status record types: input/output bounds records.
//!
//! Extracted from `status.rs` to keep it under 450 lines (#2575).

use serde::{Deserialize, Serialize};

/// Per-variable parameter input bounds record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ParamInputRecord {
    /// Index of the parameter in the multi-input model.
    /// Defaults to 0 for legacy single-input entries.
    #[serde(default)]
    pub param_index: usize,
    pub lower: f32,
    pub upper: f32,
}

impl ParamInputRecord {
    /// Create a new parameter input bounds record.
    #[must_use]
    pub fn new(param_index: usize, lower: f32, upper: f32) -> Self {
        Self {
            param_index,
            lower,
            upper,
        }
    }
}

/// Record of input bounds used for verification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct InputBoundsRecord {
    /// Per-variable parameter bounds.
    #[serde(default)]
    pub variable_inputs: Vec<ParamInputRecord>,
    /// Constant parameter values used for this verification run.
    #[serde(default)]
    pub constant_params: Vec<f32>,
    /// Optional shape metadata for variable input tensor(s).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub input_shape: Option<Vec<usize>>,
    /// Legacy single-variable bridge field for old status JSON files.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub input_range: Option<(f32, f32)>,
}

impl InputBoundsRecord {
    /// Construct from variable inputs and constant parameters.
    ///
    /// Sets `input_shape` from variable count and populates `input_range`
    /// for single-variable legacy compatibility.
    #[must_use]
    pub fn new(variable_inputs: &[ParamInputRecord], constant_params: &[f32]) -> Self {
        Self::from_variable_inputs(variable_inputs, constant_params, None)
    }
}

/// Computed output bounds. `lower`/`upper` are global min/max; optional tensor
/// fields store per-element bounds. `0.0` sentinels for failed/non-finite runs.
///
/// When `is_infeasible` is `true`, the `(lower, upper)` values are `(0.0, 0.0)`
/// sentinels that replaced the original infeasible bounds (`+Inf, -Inf`) for
/// JSON serialization safety. Consumers must check `is_infeasible` before
/// interpreting `(0.0, 0.0)` as a verified tight bound (#1692 F3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OutputBoundsRecord {
    /// Scalar lower bound. Defaults to 0.0 for legacy entries missing this field.
    #[serde(default)]
    pub lower: f32,
    /// Scalar upper bound. Defaults to 0.0 for legacy entries missing this field.
    #[serde(default)]
    pub upper: f32,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tensor_lower: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tensor_upper: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub shape: Option<Vec<usize>>,
    /// `true` when the original bounds were infeasible (e.g., `lower=+Inf,
    /// upper=-Inf` from `mark_infeasible_all()`). The scalar `lower`/`upper`
    /// fields are `0.0` sentinels in this case — not verified bounds.
    /// Legacy JSON files without this field default to `false`.
    #[serde(default)]
    pub is_infeasible: bool,
}

impl OutputBoundsRecord {
    /// Construct scalar output bounds (no per-element tensor data).
    #[must_use]
    pub fn new(lower: f32, upper: f32) -> Self {
        Self {
            lower,
            upper,
            tensor_lower: None,
            tensor_upper: None,
            shape: None,
            is_infeasible: false,
        }
    }

    /// Construct scalar output bounds with shape metadata.
    #[must_use]
    pub fn with_shape(lower: f32, upper: f32, shape: Vec<usize>) -> Self {
        Self {
            lower,
            upper,
            tensor_lower: None,
            tensor_upper: None,
            shape: Some(shape),
            is_infeasible: false,
        }
    }

    /// Construct zero-valued output bounds (for failure/degenerate cases).
    #[must_use]
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}
