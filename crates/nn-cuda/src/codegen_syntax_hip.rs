// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP implementation of [`CodegenSyntax`].
//!
//! Part of #3338 D3.

use crate::codegen_hip::{hip_accumulator_type, hip_type, safe_hip_uint};
use crate::HipCodegenError;
use nn_dsl::codegen_syntax::CodegenSyntax;
use nn_dsl::ScalarType;

/// HIP C++ syntax implementation.
pub struct HipSyntax;

impl CodegenSyntax for HipSyntax {
    type Error = HipCodegenError;

    fn uint_keyword(&self) -> &str {
        "unsigned int"
    }

    fn type_name(&self, dtype: ScalarType) -> Result<&'static str, Self::Error> {
        hip_type(dtype)
    }

    fn accum_type(&self, dtype: ScalarType) -> &'static str {
        hip_accumulator_type(dtype)
    }

    fn safe_uint(&self, val: usize) -> Result<String, Self::Error> {
        safe_hip_uint(val)
    }

    fn cast_expr(&self, target_type: &str, expr: &str) -> String {
        format!("({target_type}){expr}")
    }

    fn invalid_parameter_error(&self, msg: String) -> Self::Error {
        HipCodegenError::InvalidParameter(msg)
    }

    fn backend_name(&self) -> &str {
        "HIP"
    }
}
