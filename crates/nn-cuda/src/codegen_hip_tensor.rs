// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP C++ code generation from TensorOpKind IR.
//!
//! Entry point: `emit_tensor_hip()` takes a `TensorKernelDef` and produces
//! complete HIP C++ source with all kernel functions. Uses the same
//! backend-agnostic `build_dispatch_plan()` as MSL codegen — only the
//! emission stage differs.
//!
//! # Architecture
//!
//! ```text
//! TensorKernelDef
//!   → build_dispatch_plan()   (nn-dsl, backend-agnostic)
//!   → Vec<DispatchStep>
//!   → emit_step_hip()         (this crate, HIP-specific)
//!   → HIP C++ source string
//! ```

use crate::codegen_hip::HIP_PRELUDE;
use crate::codegen_hip_tensor_emit_step::emit_step_hip;
use crate::HipCodegenError;
use nn_dsl::{build_dispatch_plan_full, ScalarType, TensorKernelDef};

/// Generate HIP C++ source for all supported ops in a tensor kernel.
///
/// Returns the concatenated HIP C++ source with the HIP prelude and all
/// kernel functions. Returns an error if any dispatch step is unsupported.
pub fn emit_tensor_hip(
    kernel: &TensorKernelDef,
    dtype: ScalarType,
) -> Result<String, HipCodegenError> {
    let (plan, _output_id, expanded) = build_dispatch_plan_full(kernel, dtype)
        .map_err(|e| HipCodegenError::InvalidParameter(format!("dispatch plan failed: {e}")))?;
    emit_tensor_hip_with_plan(&plan, &expanded)
}

/// Generate HIP C++ source from a pre-computed dispatch plan.
///
/// This is the plan-based entry point, parallel to
/// `nn-dsl::emit_tensor_msl_with_plan`.
pub fn emit_tensor_hip_with_plan(
    plan: &[nn_dsl::DispatchStep],
    expanded_kernel: &TensorKernelDef,
) -> Result<String, HipCodegenError> {
    let mut sources = vec![HIP_PRELUDE.to_string()];

    for step in plan {
        match emit_step_hip(step, expanded_kernel) {
            Ok(Some(hip_src)) => sources.push(hip_src),
            Ok(None) => { /* no-op step (Reshape) */ }
            Err(e) => return Err(e),
        }
    }

    Ok(sources.join("\n"))
}

/// Generate HIP C++ source for a single matmul operation.
///
/// Convenience function for the GEMM PoC — creates a minimal tensor IR graph
/// for `C = A @ B` and generates the corresponding HIP kernel.
pub fn emit_gemm_hip(
    name: &str,
    dtype: ScalarType,
    m: usize,
    k: usize,
    n: usize,
) -> Result<String, HipCodegenError> {
    let matmul_src = crate::codegen_hip_tensor_emit_complex::emit_matmul_kernel(
        name, dtype, m, k, n, false, false, None,
    )?;
    Ok(format!("{HIP_PRELUDE}{matmul_src}\n"))
}

#[cfg(test)]
#[path = "codegen_hip_tensor_tests.rs"]
mod tests;
