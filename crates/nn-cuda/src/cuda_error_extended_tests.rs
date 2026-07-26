// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for CUDA error types: validation error coverage, runtime error
//! coverage, codegen error coverage, compile error coverage, and Display impls.
//!
//! Ensures all error variants construct correctly, have meaningful Display output,
//! and follow the thiserror pattern used across nn backends.

use crate::codegen_ptx::PtxCodegenError;
use crate::compile_ptx::PtxCompileError;
use crate::cuda_runtime::CudaRuntimeError;
use crate::cuda_validation::CudaValidationError;
use crate::error::HipCodegenError;

// =========================================================================
// 1. CudaValidationError coverage
// =========================================================================

mod validation_errors {
    use super::*;

    #[test]
    fn test_empty_ptx_error_display() {
        let err = CudaValidationError::EmptyPtx {
            kernel_name: "softmax".into(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("softmax"),
            "error should name the kernel: {msg}"
        );
        assert!(
            msg.contains("empty"),
            "error should mention emptiness: {msg}"
        );
    }

    #[test]
    fn test_structural_failure_display() {
        let err = CudaValidationError::StructuralFailure {
            kernel_name: "matmul".into(),
            reason: "missing .entry directive".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("matmul"));
        assert!(msg.contains("missing .entry"));
    }

    #[test]
    fn test_length_mismatch_display() {
        let err = CudaValidationError::LengthMismatch {
            expected: 1024,
            actual: 512,
        };
        let msg = err.to_string();
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn test_numerical_failure_display() {
        let err = CudaValidationError::NumericalFailure {
            kernel_name: "layernorm".into(),
            max_abs_error: 0.5,
            tolerance: 0.001,
        };
        let msg = err.to_string();
        assert!(msg.contains("layernorm"));
        assert!(msg.contains("tolerance"));
    }

    #[test]
    fn test_generation_failed_display() {
        let err = CudaValidationError::GenerationFailed {
            kernel_name: "conv2d".into(),
            reason: "shape overflow".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("conv2d"));
        assert!(msg.contains("shape overflow"));
    }
}

// =========================================================================
// 2. PtxCodegenError coverage
// =========================================================================

mod codegen_errors {
    use super::*;

    #[test]
    fn test_unsupported_type_display() {
        let err = PtxCodegenError::UnsupportedType {
            type_desc: "non-float ScalarType",
        };
        let msg = err.to_string();
        assert!(msg.contains("non-float"));
    }

    #[test]
    fn test_value_exceeds_u32_display() {
        let err = PtxCodegenError::ValueExceedsU32 {
            value: 5_000_000_000,
            max: u32::MAX,
        };
        let msg = err.to_string();
        assert!(msg.contains("u32::MAX"));
    }

    #[test]
    fn test_shape_product_overflow_display() {
        let err = PtxCodegenError::ShapeProductOverflow {
            shape: vec![usize::MAX, 2],
        };
        let msg = err.to_string();
        assert!(msg.contains("overflow"));
    }

    #[test]
    fn test_unsupported_step_display() {
        let err = PtxCodegenError::UnsupportedStep {
            step_name: "custom_op",
        };
        let msg = err.to_string();
        assert!(msg.contains("custom_op"));
    }

    #[test]
    fn test_invalid_parameter_display() {
        let err = PtxCodegenError::InvalidParameter("negative stride".into());
        let msg = err.to_string();
        assert!(msg.contains("negative stride"));
    }

    #[test]
    fn test_axis_out_of_bounds_display() {
        let err = PtxCodegenError::AxisOutOfBounds { axis: 5, rank: 3 };
        let msg = err.to_string();
        assert!(msg.contains("5"));
        assert!(msg.contains("3"));
    }
}

// =========================================================================
// 3. PtxCompileError coverage
// =========================================================================

mod compile_errors {
    use super::*;

    #[test]
    fn test_nvcc_not_found_display() {
        let err = PtxCompileError::NvccNotFound;
        let msg = err.to_string();
        assert!(msg.contains("nvcc"));
        assert!(msg.contains("CUDA Toolkit"));
    }

    #[test]
    fn test_ptxas_not_found_display() {
        let err = PtxCompileError::PtxasNotFound;
        let msg = err.to_string();
        assert!(msg.contains("ptxas"));
    }

    #[test]
    fn test_compilation_failed_display() {
        let err = PtxCompileError::CompilationFailed {
            exit_code: Some(1),
            stderr: "error: unrecognized type".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("unrecognized type"));
    }

    #[test]
    fn test_compilation_failed_no_exit_code() {
        let err = PtxCompileError::CompilationFailed {
            exit_code: None,
            stderr: "signal 9".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("signal 9"));
    }

    #[test]
    fn test_assembly_failed_display() {
        let err = PtxCompileError::AssemblyFailed {
            exit_code: Some(2),
            stderr: "ptxas fatal error".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("ptxas"));
        assert!(msg.contains("fatal error"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: PtxCompileError = io_err.into();
        let msg = err.to_string();
        assert!(msg.contains("file not found"));
    }
}

// =========================================================================
// 4. CudaRuntimeError coverage
// =========================================================================

mod runtime_errors {
    use super::*;

    #[test]
    fn test_not_available_display() {
        let err = CudaRuntimeError::NotAvailable;
        let msg = err.to_string();
        assert!(msg.contains("CUDA"));
        assert!(msg.contains("NVIDIA"));
    }

    #[test]
    fn test_no_devices_display() {
        let err = CudaRuntimeError::NoDevices;
        let msg = err.to_string();
        assert!(msg.contains("no NVIDIA GPU"));
    }

    #[test]
    fn test_api_error_display() {
        let err = CudaRuntimeError::ApiError {
            function: "cudaMalloc",
            code: 2,
        };
        let msg = err.to_string();
        assert!(msg.contains("cudaMalloc"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn test_out_of_memory_display() {
        let err = CudaRuntimeError::OutOfMemory {
            requested: 8_589_934_592, // 8GB
        };
        let msg = err.to_string();
        assert!(msg.contains("8589934592"));
    }

    #[test]
    fn test_kernel_not_found_display() {
        let err = CudaRuntimeError::KernelNotFound {
            name: "softmax_f32".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("softmax_f32"));
    }

    #[test]
    fn test_module_load_failed_display() {
        let err = CudaRuntimeError::ModuleLoadFailed {
            path: "/tmp/kernel.ptx".into(),
            reason: "invalid PTX".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("/tmp/kernel.ptx"));
        assert!(msg.contains("invalid PTX"));
    }

    #[test]
    fn test_invalid_launch_config_display() {
        let err = CudaRuntimeError::InvalidLaunchConfig {
            reason: "threads per block (2048) exceeds max (1024)".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("2048"));
        assert!(msg.contains("1024"));
    }

    #[test]
    fn test_buffer_size_mismatch_display() {
        let err = CudaRuntimeError::BufferSizeMismatch {
            expected: 4096,
            actual: 8192,
        };
        let msg = err.to_string();
        assert!(msg.contains("4096"));
        assert!(msg.contains("8192"));
    }
}

// =========================================================================
// 5. HipCodegenError coverage
// =========================================================================

mod hip_errors {
    use super::*;

    #[test]
    fn test_shape_product_overflow_display() {
        let err = HipCodegenError::ShapeProductOverflow {
            shape: vec![usize::MAX, 2],
        };
        let msg = err.to_string();
        assert!(msg.contains("overflow"));
    }

    #[test]
    fn test_unsupported_step_display() {
        let err = HipCodegenError::UnsupportedStep {
            step_name: "custom_op",
        };
        let msg = err.to_string();
        assert!(msg.contains("custom_op"));
    }

    #[test]
    fn test_stride_exceeds_u32_display() {
        let err = HipCodegenError::StrideExceedsU32 {
            value: 5_000_000_000,
            max: u32::MAX,
        };
        let msg = err.to_string();
        assert!(msg.contains("u32::MAX"));
    }

    #[test]
    fn test_invalid_parameter_display() {
        let err = HipCodegenError::InvalidParameter("negative padding".into());
        let msg = err.to_string();
        assert!(msg.contains("negative padding"));
    }

    #[test]
    fn test_axis_out_of_bounds_display() {
        let err = HipCodegenError::AxisOutOfBounds { axis: 4, rank: 2 };
        let msg = err.to_string();
        assert!(msg.contains("4"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn test_empty_stack_display() {
        let err = HipCodegenError::EmptyStack;
        let msg = err.to_string();
        assert!(msg.contains("n_inputs=0"));
    }

    #[test]
    fn test_unsupported_ir_variant_display() {
        let err = HipCodegenError::UnsupportedIRVariant {
            variant_desc: "FusedMHA",
        };
        let msg = err.to_string();
        assert!(msg.contains("FusedMHA"));
    }
}

// =========================================================================
// 6. CUDA error code constants
// =========================================================================

mod error_codes {
    use crate::cuda_ffi::error_code;

    #[test]
    fn test_success_is_zero() {
        assert_eq!(error_code::CUDA_SUCCESS, 0);
    }

    #[test]
    fn test_error_codes_are_distinct() {
        let codes = [
            error_code::CUDA_SUCCESS,
            error_code::CUDA_ERROR_INVALID_VALUE,
            error_code::CUDA_ERROR_OUT_OF_MEMORY,
            error_code::CUDA_ERROR_NOT_INITIALIZED,
            error_code::CUDA_ERROR_INVALID_DEVICE,
            error_code::CUDA_ERROR_FILE_NOT_FOUND,
            error_code::CUDA_ERROR_NOT_FOUND,
            error_code::CUDA_ERROR_LAUNCH_FAILED,
            error_code::CUDA_ERROR_NO_DEVICE,
        ];
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                assert_ne!(
                    codes[i], codes[j],
                    "error codes at positions {i} and {j} must be distinct"
                );
            }
        }
    }

    #[test]
    fn test_oom_code_is_2() {
        // Matches cudaError_t enumeration
        assert_eq!(error_code::CUDA_ERROR_OUT_OF_MEMORY, 2);
    }

    #[test]
    fn test_no_device_code_is_100() {
        assert_eq!(error_code::CUDA_ERROR_NO_DEVICE, 100);
    }
}
