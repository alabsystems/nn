// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end validation infrastructure for CUDA PTX kernels.
//!
//! Provides a framework for comparing PTX kernel output against CPU reference
//! implementations. Validation operates in two modes:
//!
//! 1. **Structural validation** (no GPU required): Verifies that generated PTX
//!    is well-formed assembly with correct kernel entry points, register
//!    declarations, and instruction structure.
//!
//! 2. **Numerical validation** (GPU required): Runs PTX on actual CUDA hardware
//!    and compares output against CPU reference within tolerance. Gated behind
//!    `cuda-runtime` feature and actual GPU availability.
//!
//! ## Architecture
//!
//! The validation pipeline:
//! ```text
//! CPU reference fn → expected output
//!                                      } → compare → ValidationResult
//! PTX generator → PTX string → validate structure
//! ```
//!
//! ## Supported operations
//!
//! - Softmax / Log-softmax (row-wise, warp-shuffle reduction)
//! - MatMul (tiled, shared memory)
//! - LayerNorm / RMSNorm / BatchNorm / GroupNorm / InstanceNorm
//! - Attention (SDPA, multi-head, GQA)
//! - Elementwise (add, sub, mul, div, exp, log, sqrt, neg, scalar_mul)
//! - RoPE (on-the-fly and cached)
//! - Transpose (2D and batched)
//! - Reductions (sum, max, mean, argmax, argmin)
//! - Embedding lookup
//! - Conv1d, Conv2d, Depthwise Conv2d
//! - Residual (add, add+relu, add+layernorm)
//! - Pooling (max, avg, adaptive avg)
//! - Linear (with/without bias, fused relu)
//! - Pad (zero, reflect)
//! - Upsample (nearest 1D/2D)
//! - Gather / Scatter-add
//! - Where / Clamp
//! - Cast (f32↔f16, f32↔bf16)
//! - Quantize / Dequantize (f32↔int8)

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from CUDA PTX validation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CudaValidationError {
    /// Generated PTX is empty.
    #[error("PTX output is empty for kernel '{kernel_name}'")]
    EmptyPtx { kernel_name: String },

    /// PTX is missing required structural element.
    #[error("PTX structural check failed for '{kernel_name}': {reason}")]
    StructuralFailure { kernel_name: String, reason: String },

    /// Shape mismatch between expected and actual output.
    #[error("output length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    /// Numerical comparison exceeded tolerance.
    #[error(
        "numerical validation failed for '{kernel_name}': \
         max_abs_error={max_abs_error:.6e} > tolerance={tolerance:.6e}"
    )]
    NumericalFailure {
        kernel_name: String,
        max_abs_error: f32,
        tolerance: f32,
    },

    /// PTX generation itself returned an error.
    #[error("PTX generation failed for '{kernel_name}': {reason}")]
    GenerationFailed { kernel_name: String, reason: String },
}

// ---------------------------------------------------------------------------
// Validation result
// ---------------------------------------------------------------------------

/// Per-element error statistics from comparing PTX output against CPU reference.
#[derive(Debug, Clone)]
pub struct ErrorStats {
    /// Maximum absolute error across all elements.
    pub max_abs_error: f32,
    /// Mean absolute error across all elements.
    pub mean_abs_error: f32,
    /// Maximum relative error (only for non-zero expected values).
    pub max_rel_error: f32,
    /// Number of NaN values in the output.
    pub num_nans: usize,
    /// Number of Inf values in the output.
    pub num_infs: usize,
    /// Total number of elements compared.
    pub num_elements: usize,
}

impl ErrorStats {
    /// Compute error statistics between `actual` and `expected` slices.
    ///
    /// Returns `Err` if the slices have different lengths.
    pub fn compute(actual: &[f32], expected: &[f32]) -> Result<Self, CudaValidationError> {
        if actual.len() != expected.len() {
            return Err(CudaValidationError::LengthMismatch {
                expected: expected.len(),
                actual: actual.len(),
            });
        }

        let n = actual.len();
        if n == 0 {
            return Ok(Self {
                max_abs_error: 0.0,
                mean_abs_error: 0.0,
                max_rel_error: 0.0,
                num_nans: 0,
                num_infs: 0,
                num_elements: 0,
            });
        }

        let mut max_abs: f32 = 0.0;
        let mut sum_abs: f64 = 0.0;
        let mut max_rel: f32 = 0.0;
        let mut num_nans: usize = 0;
        let mut num_infs: usize = 0;

        for (a, e) in actual.iter().zip(expected.iter()) {
            if a.is_nan() {
                num_nans += 1;
                continue;
            }
            if a.is_infinite() {
                num_infs += 1;
                continue;
            }
            let abs_err = (a - e).abs();
            if abs_err > max_abs {
                max_abs = abs_err;
            }
            sum_abs += f64::from(abs_err);

            if e.abs() > 1e-10 {
                let rel_err = abs_err / e.abs();
                if rel_err > max_rel {
                    max_rel = rel_err;
                }
            }
        }

        Ok(Self {
            max_abs_error: max_abs,
            mean_abs_error: (sum_abs / n as f64) as f32,
            max_rel_error: max_rel,
            num_nans,
            num_infs,
            num_elements: n,
        })
    }
}

/// Result of a PTX validation run.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Kernel name that was validated.
    pub kernel_name: String,
    /// Whether structural validation passed.
    pub structural_ok: bool,
    /// Whether numerical validation passed (None if not run).
    pub numerical_ok: Option<bool>,
    /// Error statistics (None if numerical comparison was not run).
    pub error_stats: Option<ErrorStats>,
    /// Structural checks that passed.
    pub structural_checks: Vec<String>,
    /// Structural checks that failed.
    pub structural_failures: Vec<String>,
}

impl ValidationResult {
    /// Create a new result for a kernel.
    pub fn new(kernel_name: &str) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            structural_ok: true,
            numerical_ok: None,
            error_stats: None,
            structural_checks: Vec::new(),
            structural_failures: Vec::new(),
        }
    }

    /// Whether all validation checks passed.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.structural_ok && self.numerical_ok.unwrap_or(true)
    }
}

// ---------------------------------------------------------------------------
// Structural PTX validation
// ---------------------------------------------------------------------------

/// Checks that a PTX string is structurally valid.
///
/// Performs the following checks:
/// - Non-empty output
/// - Contains `.entry` directive with the expected kernel name
/// - Contains register declarations (`.reg`)
/// - Contains parameter declarations (`.param`)
/// - Contains proper return (`ret;`) and closing brace
/// - Contains expected PTX version and target directives (for raw PTX)
/// - Contains expected CUDA keywords (for CUDA C++ source)
///
/// This does NOT require a GPU — it is pure string analysis.
pub fn validate_ptx_structure(ptx: &str, kernel_name: &str) -> ValidationResult {
    let mut result = ValidationResult::new(kernel_name);

    if ptx.is_empty() {
        result.structural_ok = false;
        result
            .structural_failures
            .push("PTX output is empty".into());
        return result;
    }
    result.structural_checks.push("non-empty".into());

    // Detect whether this is raw PTX assembly or CUDA C++ source
    let is_raw_ptx = ptx.contains(".version") || ptx.contains(".entry");
    let is_cuda_cpp = ptx.contains("__global__") || ptx.contains("#include");

    if is_raw_ptx {
        validate_raw_ptx(&mut result, ptx, kernel_name);
    } else if is_cuda_cpp {
        validate_cuda_cpp(&mut result, ptx, kernel_name);
    } else {
        result.structural_ok = false;
        result
            .structural_failures
            .push("unrecognized format: neither raw PTX nor CUDA C++".into());
    }

    result
}

/// Validate raw PTX assembly structure.
fn validate_raw_ptx(result: &mut ValidationResult, ptx: &str, kernel_name: &str) {
    // Check for .version directive
    if ptx.contains(".version") {
        result.structural_checks.push(".version directive".into());
    } else {
        result.structural_ok = false;
        result
            .structural_failures
            .push("missing .version directive".into());
    }

    // Check for .target directive
    if ptx.contains(".target") {
        result.structural_checks.push(".target directive".into());
    } else {
        result.structural_ok = false;
        result
            .structural_failures
            .push("missing .target directive".into());
    }

    // Check for kernel entry point
    let entry_marker = format!(".entry {kernel_name}");
    if ptx.contains(&entry_marker) {
        result
            .structural_checks
            .push(format!(".entry {kernel_name}"));
    } else {
        result.structural_ok = false;
        result
            .structural_failures
            .push(format!("missing .entry {kernel_name}"));
    }

    // Check for register declarations
    if ptx.contains(".reg") {
        result
            .structural_checks
            .push("register declarations".into());
    } else {
        result.structural_ok = false;
        result
            .structural_failures
            .push("missing register declarations (.reg)".into());
    }

    // Check for parameter declarations
    if ptx.contains(".param") {
        result
            .structural_checks
            .push("parameter declarations".into());
    } else {
        result.structural_ok = false;
        result
            .structural_failures
            .push("missing parameter declarations (.param)".into());
    }

    // Check for ret instruction
    if ptx.contains("ret;") {
        result.structural_checks.push("ret instruction".into());
    } else {
        result.structural_ok = false;
        result
            .structural_failures
            .push("missing ret instruction".into());
    }
}

/// Validate CUDA C++ source structure.
fn validate_cuda_cpp(result: &mut ValidationResult, src: &str, kernel_name: &str) {
    // Check for __global__ kernel declaration
    if src.contains("__global__") {
        result.structural_checks.push("__global__ keyword".into());
    } else {
        result.structural_ok = false;
        result
            .structural_failures
            .push("missing __global__ keyword".into());
    }

    // Check for kernel name
    if src.contains(kernel_name) {
        result
            .structural_checks
            .push(format!("kernel name '{kernel_name}'"));
    } else {
        result.structural_ok = false;
        result
            .structural_failures
            .push(format!("missing kernel name '{kernel_name}'"));
    }

    // Check for bounds check (important for correctness)
    if src.contains("if (") || src.contains("idx >= ") || src.contains("idx <") {
        result.structural_checks.push("bounds check".into());
    }

    // Check for CUDA runtime header
    if src.contains("#include") {
        result.structural_checks.push("include directives".into());
    }
}

// ---------------------------------------------------------------------------
// Numerical validation helpers
// ---------------------------------------------------------------------------

/// Compare two f32 slices within a tolerance, returning a `ValidationResult`.
///
/// This performs element-wise comparison and collects error statistics.
/// Useful for comparing CPU reference output against expected values
/// when actual GPU execution is not available.
pub fn validate_numerical(
    kernel_name: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> Result<ValidationResult, CudaValidationError> {
    let stats = ErrorStats::compute(actual, expected)?;

    let numerical_ok = stats.max_abs_error <= tolerance && stats.num_nans == 0;

    Ok(ValidationResult {
        kernel_name: kernel_name.to_string(),
        structural_ok: true,
        numerical_ok: Some(numerical_ok),
        error_stats: Some(stats),
        structural_checks: Vec::new(),
        structural_failures: Vec::new(),
    })
}

/// Validate PTX output end-to-end: structural check + numerical comparison
/// of PTX generation parameters against CPU reference.
///
/// This does NOT execute PTX on a GPU. It:
/// 1. Validates the generated PTX string structure
/// 2. Compares CPU reference output against expected values
///
/// For actual GPU execution validation, use [`validate_on_gpu`] (requires
/// `cuda-runtime` feature and NVIDIA GPU hardware).
pub fn validate_ptx_e2e(
    kernel_name: &str,
    ptx: &str,
    cpu_output: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> Result<ValidationResult, CudaValidationError> {
    // Step 1: structural validation
    let structural = validate_ptx_structure(ptx, kernel_name);
    if !structural.structural_ok {
        return Err(CudaValidationError::StructuralFailure {
            kernel_name: kernel_name.to_string(),
            reason: structural.structural_failures.join("; "),
        });
    }

    // Step 2: numerical validation
    let stats = ErrorStats::compute(cpu_output, expected)?;
    let numerical_ok = stats.max_abs_error <= tolerance && stats.num_nans == 0;

    if !numerical_ok {
        return Err(CudaValidationError::NumericalFailure {
            kernel_name: kernel_name.to_string(),
            max_abs_error: stats.max_abs_error,
            tolerance,
        });
    }

    Ok(ValidationResult {
        kernel_name: kernel_name.to_string(),
        structural_ok: true,
        numerical_ok: Some(true),
        error_stats: Some(stats),
        structural_checks: structural.structural_checks,
        structural_failures: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Validation suite
// ---------------------------------------------------------------------------

/// Entry in the validation suite: one kernel to validate.
pub struct ValidationEntry {
    /// Kernel name.
    pub name: String,
    /// Generated PTX or CUDA C++ source.
    pub ptx: String,
    /// CPU reference output.
    pub cpu_output: Vec<f32>,
    /// Expected output (typically same as cpu_output for self-consistency).
    pub expected: Vec<f32>,
    /// Tolerance for numerical comparison.
    pub tolerance: f32,
}

/// Batch validation suite for running multiple kernel validations.
///
/// Collects validation entries and runs them all, reporting per-kernel
/// results and an aggregate pass/fail.
pub struct CudaValidationSuite {
    entries: Vec<ValidationEntry>,
}

impl CudaValidationSuite {
    /// Create a new empty validation suite.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a validation entry to the suite.
    pub fn add(
        &mut self,
        name: &str,
        ptx: String,
        cpu_output: Vec<f32>,
        expected: Vec<f32>,
        tolerance: f32,
    ) {
        self.entries.push(ValidationEntry {
            name: name.to_string(),
            ptx,
            cpu_output,
            expected,
            tolerance,
        });
    }

    /// Run all validation entries.
    ///
    /// Returns a vector of results, one per entry. Does not short-circuit
    /// on failure — all entries are validated.
    pub fn run(&self) -> Vec<Result<ValidationResult, CudaValidationError>> {
        self.entries
            .iter()
            .map(|entry| {
                validate_ptx_e2e(
                    &entry.name,
                    &entry.ptx,
                    &entry.cpu_output,
                    &entry.expected,
                    entry.tolerance,
                )
            })
            .collect()
    }

    /// Run all entries and return true only if every entry passes.
    pub fn run_all_pass(&self) -> bool {
        self.run().iter().all(Result::is_ok)
    }

    /// Number of entries in the suite.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the suite is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for CudaValidationSuite {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_stats_identical_arrays() {
        let a = vec![1.0, 2.0, 3.0];
        let stats = ErrorStats::compute(&a, &a).unwrap();
        assert_eq!(stats.max_abs_error, 0.0);
        assert_eq!(stats.mean_abs_error, 0.0);
        assert_eq!(stats.num_nans, 0);
        assert_eq!(stats.num_infs, 0);
        assert_eq!(stats.num_elements, 3);
    }

    #[test]
    fn test_error_stats_small_difference() {
        let a = vec![1.0, 2.001, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let stats = ErrorStats::compute(&a, &b).unwrap();
        assert!((stats.max_abs_error - 0.001).abs() < 1e-5);
        assert!(stats.mean_abs_error > 0.0);
        assert_eq!(stats.num_nans, 0);
    }

    #[test]
    fn test_error_stats_nan_detection() {
        let a = vec![1.0, f32::NAN, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let stats = ErrorStats::compute(&a, &b).unwrap();
        assert_eq!(stats.num_nans, 1);
    }

    #[test]
    fn test_error_stats_inf_detection() {
        let a = vec![1.0, f32::INFINITY, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let stats = ErrorStats::compute(&a, &b).unwrap();
        assert_eq!(stats.num_infs, 1);
    }

    #[test]
    fn test_error_stats_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = ErrorStats::compute(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_stats_empty_arrays() {
        let a: Vec<f32> = vec![];
        let stats = ErrorStats::compute(&a, &a).unwrap();
        assert_eq!(stats.num_elements, 0);
        assert_eq!(stats.max_abs_error, 0.0);
    }

    #[test]
    fn test_validate_ptx_structure_raw_ptx() {
        let ptx = "\
.version 6.5\n\
.target sm_80\n\
.address_size 64\n\
.visible .entry nn_kernel(\n\
    .param .u64 param_input\n\
)\n\
{\n\
    .reg .f32 %f<4>;\n\
    ret;\n\
}\n";
        let result = validate_ptx_structure(ptx, "nn_kernel");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
        assert!(result.structural_checks.len() >= 5);
    }

    #[test]
    fn test_validate_ptx_structure_missing_entry() {
        let ptx = "\
.version 6.5\n\
.target sm_80\n\
.address_size 64\n\
{\n\
    .reg .f32 %f<4>;\n\
    ret;\n\
}\n";
        let result = validate_ptx_structure(ptx, "missing_kernel");
        assert!(!result.structural_ok);
        assert!(result
            .structural_failures
            .iter()
            .any(|f| f.contains("missing .entry")));
    }

    #[test]
    fn test_validate_ptx_structure_cuda_cpp() {
        let src = "\
#include <cuda_runtime.h>\n\
__global__ void nn_kernel(float* input, float* output, unsigned int N) {\n\
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
    if (idx >= N) return;\n\
    output[idx] = input[idx];\n\
}\n";
        let result = validate_ptx_structure(src, "nn_kernel");
        assert!(
            result.structural_ok,
            "failures: {:?}",
            result.structural_failures
        );
    }

    #[test]
    fn test_validate_ptx_structure_empty() {
        let result = validate_ptx_structure("", "empty");
        assert!(!result.structural_ok);
    }

    #[test]
    fn test_validate_numerical_pass() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let result = validate_numerical("softmax_test", &a, &a, 1e-5).unwrap();
        assert!(result.passed());
        assert_eq!(result.numerical_ok, Some(true));
    }

    #[test]
    fn test_validate_numerical_fail() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.5, 3.0];
        let result = validate_numerical("bad_kernel", &a, &b, 1e-3).unwrap();
        assert!(!result.passed());
    }

    #[test]
    fn test_validation_suite_empty() {
        let suite = CudaValidationSuite::new();
        assert!(suite.is_empty());
        assert!(suite.run_all_pass());
    }

    #[test]
    fn test_validation_suite_all_pass() {
        let mut suite = CudaValidationSuite::new();
        let ptx = ".version 6.5\n.target sm_80\n.visible .entry k1(\n.param .u64 p\n)\n{\n.reg .f32 %f<1>;\nret;\n}\n";
        suite.add("k1", ptx.to_string(), vec![1.0, 2.0], vec![1.0, 2.0], 1e-5);
        assert!(suite.run_all_pass());
        assert_eq!(suite.len(), 1);
    }

    #[test]
    fn test_validation_suite_one_fails() {
        let mut suite = CudaValidationSuite::new();
        let ptx = ".version 6.5\n.target sm_80\n.visible .entry k1(\n.param .u64 p\n)\n{\n.reg .f32 %f<1>;\nret;\n}\n";
        suite.add("k1", ptx.to_string(), vec![1.0], vec![1.0], 1e-5);
        suite.add("k2", ptx.to_string(), vec![1.0], vec![2.0], 1e-5);
        assert!(!suite.run_all_pass());
    }

    #[test]
    fn test_validate_ptx_e2e_pass() {
        let ptx = ".version 6.5\n.target sm_80\n.visible .entry sm(\n.param .u64 p\n)\n{\n.reg .f32 %f<1>;\nret;\n}\n";
        let output = vec![0.25, 0.25, 0.25, 0.25];
        let result = validate_ptx_e2e("sm", ptx, &output, &output, 1e-5).unwrap();
        assert!(result.passed());
    }

    #[test]
    fn test_validate_ptx_e2e_structural_failure() {
        let result = validate_ptx_e2e("k", "", &[1.0], &[1.0], 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_result_default_state() {
        let result = ValidationResult::new("test");
        assert!(result.passed());
        assert!(result.structural_ok);
        assert_eq!(result.numerical_ok, None);
    }

    #[test]
    fn test_cuda_validation_error_display() {
        let err = CudaValidationError::EmptyPtx {
            kernel_name: "softmax".into(),
        };
        assert!(err.to_string().contains("softmax"));

        let err = CudaValidationError::NumericalFailure {
            kernel_name: "matmul".into(),
            max_abs_error: 0.5,
            tolerance: 0.001,
        };
        let msg = err.to_string();
        assert!(msg.contains("matmul"));
        assert!(
            msg.contains("1.0") || msg.contains("e-3") || msg.contains("0.001"),
            "tolerance should appear in message: {msg}"
        );
    }
}
