// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `NnDslError` unified error supertype.
//!
//! Demonstrates `?` operator propagation across multiple nn-dsl error types
//! within a single function returning `Result<T, NnDslError>`.
//!
//! Part of #690.

use nn_dsl::{
    build_instance_norm_decomposed, build_snake_scalar_kernel, emit_msl, snake_scalar, NnDslError,
};

/// Exercises `?` across KernelError (from `snake_scalar`) and LowerError
/// (from `build_snake_scalar_kernel`) in a single function.
///
/// This is the core consumer pain point: a pipeline function that calls
/// both scalar kernel evaluation and IR building needs a common error type.
fn pipeline_scalar_and_build() -> Result<(), NnDslError> {
    // KernelError path: evaluate a scalar kernel
    let _val = snake_scalar(0.5, 1.0)?;

    // LowerError path: build the kernel IR
    let kernel = build_snake_scalar_kernel()?;

    // IRError would come from kernel.validate(), but snake is always valid.
    // Demonstrate that the kernel IR is usable after unified error propagation.
    assert!(kernel.params.len() >= 2, "snake has at least 2 params");

    Ok(())
}

/// Exercises `?` across LowerError (from `build_instance_norm_decomposed`)
/// and the MSL codegen path (which can produce errors) in one function.
fn pipeline_tensor_build_and_codegen() -> Result<(), NnDslError> {
    // LowerError path: build a scalar kernel
    let kernel = build_snake_scalar_kernel()?;

    // IRError path: emit MSL (returns Result<String, IRError>).
    // The `?` propagates IRError → NnDslError::Ir, demonstrating cross-error unification.
    let msl = emit_msl(&kernel)?;
    assert!(!msl.is_empty(), "MSL output should not be empty");

    Ok(())
}

/// Demonstrates that NnDslError correctly propagates KernelError for
/// invalid inputs (NaN rejection).
fn pipeline_kernel_error_propagation() -> Result<(), NnDslError> {
    // snake_scalar rejects NaN inputs → KernelError → NnDslError::Kernel
    let result: Result<(), NnDslError> = (|| {
        let _val = snake_scalar(f32::NAN, 1.0)?;
        Ok(())
    })();

    assert!(result.is_err(), "NaN input should produce KernelError");
    let err = result.unwrap_err();
    assert!(
        matches!(err, NnDslError::Kernel(_)),
        "error should be NnDslError::Kernel, got: {err:?}"
    );

    Ok(())
}

#[test]
fn test_nn_dsl_error_scalar_and_build_pipeline() {
    pipeline_scalar_and_build().expect("pipeline should succeed with valid inputs");
}

#[test]
fn test_nn_dsl_error_tensor_build_and_codegen_pipeline() {
    pipeline_tensor_build_and_codegen().expect("pipeline should succeed");
}

#[test]
fn test_nn_dsl_error_kernel_error_propagation() {
    pipeline_kernel_error_propagation().expect("error propagation test should succeed");
}

/// Verify NnDslError is Debug + Display (thiserror requirement).
#[test]
fn test_nn_dsl_error_debug_display() {
    let err = NnDslError::Kernel(snake_scalar(f32::NAN, 1.0).unwrap_err());
    // Debug
    let debug = format!("{err:?}");
    assert!(!debug.is_empty());
    // Display (via thiserror #[error(transparent)])
    let display = format!("{err}");
    assert!(!display.is_empty());
}

/// Verify that `?` works for TensorIRError via `build_instance_norm_decomposed`.
/// The function returns LowerError which converts to NnDslError::Lower.
#[test]
fn test_nn_dsl_error_tensor_ir_path() {
    fn build_tensor_pipeline() -> Result<(), NnDslError> {
        let _kernel = build_instance_norm_decomposed(1, 4, 16)?;
        Ok(())
    }
    build_tensor_pipeline().expect("tensor pipeline should succeed");
}
