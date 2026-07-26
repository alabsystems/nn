// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `CodegenSyntax` trait: backend-agnostic GPU syntax primitives.
//!
//! Shared algorithmic codegen (conv loops, structural ops) calls this trait;
//! backends provide MSL- or HIP-specific syntax. This eliminates ~1,200 lines
//! of structurally identical codegen duplicated between nn-dsl and nn-cuda.
//!
//! Part of #3338 D3.

use crate::ir::ScalarType;
use std::fmt;

/// GPU syntax primitives that differ between Metal (MSL) and HIP backends.
///
/// Shared codegen functions (e.g., `emit_conv1d_kernel_generic`) are generic
/// over this trait. Each backend implements it once; the loop bodies and
/// indexing math are written once in terms of these primitives.
pub trait CodegenSyntax {
    /// Backend-specific error type for codegen failures.
    type Error: fmt::Display;

    /// Integer type keyword for indexing: `"uint"` (MSL) or `"unsigned int"` (HIP).
    fn uint_keyword(&self) -> &str;

    /// Map a `ScalarType` to the backend's type name.
    ///
    /// MSL: `"float"`, `"half"`, `"bfloat"`.
    /// HIP: `"float"`, `"half"`, `"hip_bfloat16"`.
    fn type_name(&self, dtype: ScalarType) -> Result<&'static str, Self::Error>;

    /// Accumulator type for mixed-precision (f16/bf16 → f32 accumulation).
    fn accum_type(&self, dtype: ScalarType) -> &'static str;

    /// Validate that a `usize` fits in a 32-bit uint and return as a string.
    fn safe_uint(&self, val: usize) -> Result<String, Self::Error>;

    /// Emit a type cast expression.
    ///
    /// MSL constructor syntax: `"float(expr)"`.
    /// HIP C-style cast: `"(float)expr"`.
    fn cast_expr(&self, target_type: &str, expr: &str) -> String;

    /// Emit a const integer declaration.
    ///
    /// MSL: `"    const uint NAME = VALUE;"`
    /// HIP: `"    const unsigned int NAME = VALUE;"`
    fn const_uint_decl(&self, name: &str, value: &str) -> String {
        format!("    const {} {} = {};", self.uint_keyword(), name, value)
    }

    /// Emit a local variable declaration.
    ///
    /// MSL: `"    uint name = expr;"`
    /// HIP: `"    unsigned int name = expr;"`
    fn uint_var_decl(&self, name: &str, expr: &str) -> String {
        format!("    {} {} = {};", self.uint_keyword(), name, expr)
    }

    /// Emit a for-loop header.
    ///
    /// MSL: `"    for (uint var = 0; var < limit; var++) {"`
    /// HIP: `"    for (unsigned int var = 0; var < limit; var++) {"`
    fn for_loop_header(&self, var: &str, limit: &str) -> String {
        format!(
            "    for ({} {} = 0; {} < {}; {}++) {{",
            self.uint_keyword(),
            var,
            var,
            limit,
            var
        )
    }

    /// Create an "invalid parameter" error with a message.
    fn invalid_parameter_error(&self, msg: String) -> Self::Error;

    /// Backend name for diagnostics: `"MSL"` or `"HIP"`.
    fn backend_name(&self) -> &str;
}
