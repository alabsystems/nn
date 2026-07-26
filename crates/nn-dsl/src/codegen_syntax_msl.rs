// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL implementation of [`CodegenSyntax`].
//!
//! Part of #3338 D3.

use crate::codegen_msl_structural::safe_msl_uint;
use crate::codegen_msl_tensor::TensorMSLCodegenError;
use crate::codegen_syntax::CodegenSyntax;
use crate::ir::ScalarType;

/// Metal Shading Language syntax implementation.
pub struct MslSyntax;

impl CodegenSyntax for MslSyntax {
    type Error = TensorMSLCodegenError;

    fn uint_keyword(&self) -> &str {
        "uint"
    }

    fn type_name(&self, dtype: ScalarType) -> Result<&'static str, Self::Error> {
        Ok(dtype.msl_str())
    }

    fn accum_type(&self, dtype: ScalarType) -> &'static str {
        dtype.msl_accumulator_str()
    }

    fn safe_uint(&self, val: usize) -> Result<String, Self::Error> {
        safe_msl_uint(val)
    }

    fn cast_expr(&self, target_type: &str, expr: &str) -> String {
        format!("{target_type}({expr})")
    }

    fn invalid_parameter_error(&self, msg: String) -> Self::Error {
        TensorMSLCodegenError::InvalidParameter(msg)
    }

    fn backend_name(&self) -> &str {
        "MSL"
    }
}
