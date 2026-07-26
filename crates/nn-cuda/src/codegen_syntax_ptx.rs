// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX implementation of [`CodegenSyntax`] for NVIDIA CUDA C++ emission.
//!
//! While PTX itself is an assembly-like ISA, kernel dispatch wrappers and
//! high-level codegen use CUDA C++ syntax (compiled via `nvcc`). This
//! implementation provides CUDA C++ syntax for the shared codegen pipeline.
//!
//! Parallel to [`codegen_syntax_hip`](super::codegen_syntax_hip).

use crate::codegen_ptx::{cuda_type, ptx_accumulator_type, safe_ptx_uint, PtxCodegenError};
use nn_dsl::codegen_syntax::CodegenSyntax;
use nn_dsl::ScalarType;

/// CUDA C++ syntax implementation for shared codegen.
///
/// CUDA and HIP share nearly identical C++ syntax (HIP was designed as a
/// source-compatible layer). The key differences are type names for bf16
/// and include paths.
pub struct CudaSyntax;

impl CodegenSyntax for CudaSyntax {
    type Error = PtxCodegenError;

    fn uint_keyword(&self) -> &'static str {
        "unsigned int"
    }

    fn type_name(&self, dtype: ScalarType) -> Result<&'static str, Self::Error> {
        cuda_type(dtype)
    }

    fn accum_type(&self, dtype: ScalarType) -> &'static str {
        // PTX accumulator is ".f32" but for CUDA C++ we return "float"
        let _ = ptx_accumulator_type(dtype);
        "float"
    }

    fn safe_uint(&self, val: usize) -> Result<String, Self::Error> {
        safe_ptx_uint(val)
    }

    fn cast_expr(&self, target_type: &str, expr: &str) -> String {
        // CUDA uses C-style casts like HIP
        format!("({target_type}){expr}")
    }

    fn invalid_parameter_error(&self, msg: String) -> Self::Error {
        PtxCodegenError::InvalidParameter(msg)
    }

    fn backend_name(&self) -> &'static str {
        "CUDA"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_syntax_uint_keyword() {
        let s = CudaSyntax;
        assert_eq!(s.uint_keyword(), "unsigned int");
    }

    #[test]
    fn test_cuda_syntax_type_name() {
        let s = CudaSyntax;
        assert_eq!(s.type_name(ScalarType::F32).unwrap(), "float");
        assert_eq!(s.type_name(ScalarType::F16).unwrap(), "__half");
        assert_eq!(s.type_name(ScalarType::BF16).unwrap(), "__nv_bfloat16");
    }

    #[test]
    fn test_cuda_syntax_accum_type() {
        let s = CudaSyntax;
        assert_eq!(s.accum_type(ScalarType::F32), "float");
        assert_eq!(s.accum_type(ScalarType::F16), "float");
        assert_eq!(s.accum_type(ScalarType::BF16), "float");
    }

    #[test]
    fn test_cuda_syntax_cast_expr() {
        let s = CudaSyntax;
        assert_eq!(s.cast_expr("float", "x"), "(float)x");
    }

    #[test]
    fn test_cuda_syntax_backend_name() {
        let s = CudaSyntax;
        assert_eq!(s.backend_name(), "CUDA");
    }

    #[test]
    fn test_cuda_syntax_const_uint_decl() {
        let s = CudaSyntax;
        let decl = s.const_uint_decl("N", "1024");
        assert_eq!(decl, "    const unsigned int N = 1024;");
    }

    #[test]
    fn test_cuda_syntax_for_loop_header() {
        let s = CudaSyntax;
        let header = s.for_loop_header("i", "N");
        assert!(header.contains("unsigned int i = 0"));
        assert!(header.contains("i < N"));
    }
}
