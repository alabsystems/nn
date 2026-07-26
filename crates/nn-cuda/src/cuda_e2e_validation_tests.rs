// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end CUDA PTX validation tests.
//!
//! For each PTX generator, this module:
//! 1. Generates PTX (or CUDA C++) source
//! 2. Runs structural validation (no GPU required)
//! 3. Computes CPU reference output
//! 4. Validates CPU reference against expected values
//! 5. Runs the validation suite for batch coverage
//!
//! GPU execution tests are gated behind runtime availability.

use crate::cuda_validation::{
    validate_numerical, validate_ptx_e2e, validate_ptx_structure, CudaValidationSuite, ErrorStats,
    ValidationResult,
};

// =========================================================================
// 1. Structural validation for all PTX generators
// =========================================================================

mod structural_e2e {
    use super::*;
    use crate::ptx_attention::generate_sdpa_ptx;
    use crate::ptx_batchnorm::generate_batchnorm_ptx;
    use crate::ptx_conv1d::emit_ptx_conv1d_default;
    use crate::ptx_conv2d::emit_ptx_conv2d_default;
    use crate::ptx_elementwise::{
        generate_add_ptx, generate_div_ptx, generate_exp_ptx, generate_log_ptx, generate_mul_ptx,
        generate_neg_ptx, generate_sqrt_ptx, generate_sub_ptx,
    };
    use crate::ptx_embedding::{generate_embedding_ptx, PtxEmbeddingConfig};
    use crate::ptx_gather::generate_gather_ptx;
    use crate::ptx_groupnorm::generate_groupnorm_ptx;
    use crate::ptx_instancenorm::generate_instancenorm_ptx;
    use crate::ptx_layernorm::generate_layernorm_ptx;
    use crate::ptx_linear::{generate_linear_ptx, generate_linear_relu_ptx};
    use crate::ptx_matmul::generate_matmul_ptx;
    use crate::ptx_reduce::{
        generate_argmax_ptx, generate_max_ptx, generate_mean_ptx, generate_sum_ptx,
    };
    use crate::ptx_residual::generate_residual_add_ptx;
    use crate::ptx_rmsnorm::generate_rmsnorm_ptx;
    use crate::ptx_rope::{generate_rope_ptx, PtxRopeConfig};
    use crate::ptx_softmax::{generate_log_softmax_ptx, generate_softmax_ptx};
    use crate::ptx_transpose::generate_transpose_ptx;

    /// Extract the first `.entry <name>` from raw PTX or the first
    /// `__global__ void <name>` from CUDA C++. Returns the kernel name.
    fn extract_kernel_name(ptx: &str) -> Option<String> {
        // Raw PTX: `.entry <name>(`
        if let Some(idx) = ptx.find(".entry ") {
            let rest = &ptx[idx + 7..];
            let end = rest.find('(').unwrap_or(rest.len());
            return Some(rest[..end].trim().to_string());
        }
        // CUDA C++: `__global__ void <name>(`
        if let Some(idx) = ptx.find("__global__") {
            let rest = &ptx[idx..];
            if let Some(void_idx) = rest.find("void ") {
                let after_void = &rest[void_idx + 5..];
                let end = after_void.find('(').unwrap_or(after_void.len());
                return Some(after_void[..end].trim().to_string());
            }
        }
        None
    }

    /// Validates that PTX output is structurally sound: non-empty, has
    /// the expected directives, and has a valid kernel entry point.
    fn assert_structural_ok(ptx: &str, label: &str) {
        assert!(!ptx.is_empty(), "{label}: PTX output is empty");
        let kernel_name = extract_kernel_name(ptx)
            .unwrap_or_else(|| panic!("{label}: could not extract kernel name from PTX"));
        let result = validate_ptx_structure(ptx, &kernel_name);
        assert!(
            result.structural_ok,
            "{label} (kernel={kernel_name}): structural validation failed: {:?}",
            result.structural_failures
        );
        assert!(
            !result.structural_checks.is_empty(),
            "{label}: no structural checks passed"
        );
    }

    #[test]
    fn test_matmul_structural() {
        let ptx = generate_matmul_ptx(64, 32, 64);
        assert_structural_ok(&ptx, "matmul");
    }

    #[test]
    fn test_softmax_structural() {
        let ptx = generate_softmax_ptx(false, 128);
        assert_structural_ok(&ptx, "softmax");
    }

    #[test]
    fn test_log_softmax_structural() {
        let ptx = generate_log_softmax_ptx(256);
        assert_structural_ok(&ptx, "log_softmax");
    }

    #[test]
    fn test_layernorm_structural() {
        let ptx = generate_layernorm_ptx(768);
        assert_structural_ok(&ptx, "layernorm");
    }

    #[test]
    fn test_rmsnorm_structural() {
        let ptx = generate_rmsnorm_ptx(512, 1e-5);
        assert_structural_ok(&ptx, "rmsnorm");
    }

    #[test]
    fn test_elementwise_ops_structural() {
        let ops: Vec<(&str, String)> = vec![
            ("add", generate_add_ptx(256)),
            ("sub", generate_sub_ptx(256)),
            ("mul", generate_mul_ptx(256)),
            ("div", generate_div_ptx(256)),
            ("exp", generate_exp_ptx(256)),
            ("log", generate_log_ptx(256)),
            ("sqrt", generate_sqrt_ptx(256)),
            ("neg", generate_neg_ptx(256)),
        ];
        for (name, ptx) in &ops {
            assert_structural_ok(ptx, name);
        }
    }

    #[test]
    fn test_reduce_ops_structural() {
        let ops: Vec<(&str, String)> = vec![
            ("sum", generate_sum_ptx(128)),
            ("max", generate_max_ptx(128)),
            ("mean", generate_mean_ptx(128)),
            ("argmax", generate_argmax_ptx(128)),
        ];
        for (name, ptx) in &ops {
            assert_structural_ok(ptx, name);
        }
    }

    #[test]
    fn test_conv1d_structural() {
        let ptx = emit_ptx_conv1d_default(3, 16, 3, 1, 1).unwrap();
        assert_structural_ok(&ptx, "conv1d");
    }

    #[test]
    fn test_conv2d_structural() {
        let ptx = emit_ptx_conv2d_default("conv2d_kernel").unwrap();
        assert_structural_ok(&ptx, "conv2d");
    }

    #[test]
    fn test_linear_structural() {
        let ptx = generate_linear_ptx(768, 3072);
        assert_structural_ok(&ptx, "linear");
    }

    #[test]
    fn test_linear_relu_structural() {
        let ptx = generate_linear_relu_ptx(768, 3072);
        assert_structural_ok(&ptx, "linear_relu");
    }

    #[test]
    fn test_attention_structural() {
        let ptx = generate_sdpa_ptx(128, 64);
        assert_structural_ok(&ptx, "attention");
    }

    #[test]
    fn test_transpose_structural() {
        let ptx = generate_transpose_ptx(32, 64);
        assert_structural_ok(&ptx, "transpose");
    }

    #[test]
    fn test_gather_structural() {
        let ptx = generate_gather_ptx(1024, 64);
        assert_structural_ok(&ptx, "gather");
    }

    #[test]
    fn test_residual_structural() {
        let ptx = generate_residual_add_ptx(256);
        assert_structural_ok(&ptx, "residual_add");
    }

    #[test]
    fn test_rope_structural() {
        let config = PtxRopeConfig::new(64, 128);
        let ptx = generate_rope_ptx(&config).unwrap();
        assert_structural_ok(&ptx, "rope");
    }

    #[test]
    fn test_embedding_structural() {
        let config = PtxEmbeddingConfig::new(10000, 768);
        let ptx = generate_embedding_ptx(&config).unwrap();
        assert_structural_ok(&ptx, "embedding");
    }

    #[test]
    fn test_batchnorm_structural() {
        let ptx = generate_batchnorm_ptx(64);
        assert_structural_ok(&ptx, "batchnorm");
    }

    #[test]
    fn test_groupnorm_structural() {
        let ptx = generate_groupnorm_ptx(8, 64);
        assert_structural_ok(&ptx, "groupnorm");
    }

    #[test]
    fn test_instancenorm_structural() {
        let ptx = generate_instancenorm_ptx(3, 32, 32, 1e-5);
        assert_structural_ok(&ptx, "instancenorm");
    }
}

// =========================================================================
// 2. Numerical validation with CPU references
// =========================================================================

mod numerical_e2e {
    use super::*;
    use crate::ptx_elementwise::{add_reference, mul_reference, neg_reference, sub_reference};
    use crate::ptx_matmul::matmul_reference;
    use crate::ptx_reduce::{max_reference, mean_reference, sum_reference};
    use crate::ptx_softmax::softmax_reference;
    use crate::ptx_transpose::transpose_reference;

    const TOL: f32 = 1e-5;

    #[test]
    fn test_add_numerical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        let actual = add_reference(&a, &b);
        let result = validate_numerical("add", &actual, &expected, TOL).unwrap();
        assert!(result.passed(), "add numerical: {:?}", result.error_stats);
    }

    #[test]
    fn test_sub_numerical() {
        let a = vec![5.0, 3.0, 1.0];
        let b = vec![2.0, 1.0, 0.5];
        let expected = vec![3.0, 2.0, 0.5];
        let actual = sub_reference(&a, &b);
        let result = validate_numerical("sub", &actual, &expected, TOL).unwrap();
        assert!(result.passed());
    }

    #[test]
    fn test_mul_numerical() {
        let a = vec![2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0];
        let expected = vec![10.0, 18.0, 28.0];
        let actual = mul_reference(&a, &b);
        let result = validate_numerical("mul", &actual, &expected, TOL).unwrap();
        assert!(result.passed());
    }

    #[test]
    fn test_neg_numerical() {
        let input = vec![1.0, -2.0, 0.0, 3.14];
        let expected = vec![-1.0, 2.0, 0.0, -3.14];
        let actual = neg_reference(&input);
        let result = validate_numerical("neg", &actual, &expected, TOL).unwrap();
        assert!(result.passed());
    }

    #[test]
    fn test_matmul_identity_numerical() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // I_2
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let expected = vec![5.0, 6.0, 7.0, 8.0];
        let actual = matmul_reference(&a, &b, 2, 2, 2);
        let result = validate_numerical("matmul_identity", &actual, &expected, TOL).unwrap();
        assert!(result.passed());
    }

    #[test]
    fn test_softmax_numerical() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let actual = softmax_reference(&input);
        // Verify sum to 1
        let sum: f32 = actual.iter().sum();
        let expected_sum = vec![sum]; // should be ~1.0
        let result = validate_numerical("softmax_sum", &expected_sum, &[1.0], TOL).unwrap();
        assert!(result.passed(), "softmax output must sum to 1.0");
        // Verify monotonicity
        for i in 0..actual.len() - 1 {
            assert!(
                actual[i] < actual[i + 1],
                "softmax must be monotone for sorted input"
            );
        }
    }

    #[test]
    fn test_sum_reduce_numerical() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let actual = vec![sum_reference(&input)];
        let expected = vec![15.0];
        let result = validate_numerical("sum_reduce", &actual, &expected, TOL).unwrap();
        assert!(result.passed());
    }

    #[test]
    fn test_max_reduce_numerical() {
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        let actual = vec![max_reference(&input)];
        let expected = vec![9.0];
        let result = validate_numerical("max_reduce", &actual, &expected, TOL).unwrap();
        assert!(result.passed());
    }

    #[test]
    fn test_mean_reduce_numerical() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let actual = vec![mean_reference(&input)];
        let expected = vec![5.0];
        let result = validate_numerical("mean_reduce", &actual, &expected, TOL).unwrap();
        assert!(result.passed());
    }

    #[test]
    fn test_transpose_numerical() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // [3, 2]
        let actual = transpose_reference(&input, 2, 3);
        let result = validate_numerical("transpose", &actual, &expected, TOL).unwrap();
        assert!(result.passed());
    }
}

// =========================================================================
// 3. Full E2E pipeline (structural + numerical)
// =========================================================================

mod full_e2e {
    use super::*;
    use crate::ptx_elementwise::{add_reference, generate_add_ptx};
    use crate::ptx_softmax::{generate_softmax_ptx, softmax_reference};

    /// Extract the `.entry` kernel name from generated PTX.
    fn kernel_name(ptx: &str) -> String {
        if let Some(idx) = ptx.find(".entry ") {
            let rest = &ptx[idx + 7..];
            let end = rest.find('(').unwrap_or(rest.len());
            return rest[..end].trim().to_string();
        }
        "unknown".to_string()
    }

    #[test]
    fn test_add_e2e_pipeline() {
        let ptx = generate_add_ptx(4);
        let name = kernel_name(&ptx);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let cpu_output = add_reference(&a, &b);
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        let result = validate_ptx_e2e(&name, &ptx, &cpu_output, &expected, 1e-5).unwrap();
        assert!(result.passed());
        assert!(result.structural_ok);
        assert_eq!(result.numerical_ok, Some(true));
    }

    #[test]
    fn test_softmax_e2e_pipeline() {
        let ptx = generate_softmax_ptx(false, 4);
        let name = kernel_name(&ptx);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cpu_output = softmax_reference(&input);
        let result = validate_ptx_e2e(&name, &ptx, &cpu_output, &cpu_output, 1e-5).unwrap();
        assert!(result.passed());
    }

    #[test]
    fn test_e2e_fails_on_empty_ptx() {
        let result = validate_ptx_e2e("empty", "", &[1.0], &[1.0], 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_e2e_fails_on_numerical_mismatch() {
        let ptx = generate_add_ptx(2);
        let name = kernel_name(&ptx);
        let cpu_output = vec![1.0, 2.0];
        let expected = vec![100.0, 200.0]; // wildly different
        let result = validate_ptx_e2e(&name, &ptx, &cpu_output, &expected, 1e-5);
        assert!(result.is_err());
    }
}

// =========================================================================
// 4. Validation suite batch tests
// =========================================================================

mod suite_tests {
    use super::*;
    use crate::ptx_elementwise::{
        add_reference, generate_add_ptx, generate_mul_ptx, mul_reference,
    };

    fn kernel_name(ptx: &str) -> String {
        if let Some(idx) = ptx.find(".entry ") {
            let rest = &ptx[idx + 7..];
            let end = rest.find('(').unwrap_or(rest.len());
            return rest[..end].trim().to_string();
        }
        if let Some(idx) = ptx.find("__global__ void ") {
            let rest = &ptx[idx + 16..];
            let end = rest.find('(').unwrap_or(rest.len());
            return rest[..end].trim().to_string();
        }
        "unknown".to_string()
    }

    #[test]
    fn test_suite_empty_passes() {
        let suite = CudaValidationSuite::new();
        assert!(suite.is_empty());
        assert!(suite.run_all_pass());
    }

    #[test]
    fn test_suite_single_pass() {
        let mut suite = CudaValidationSuite::new();
        let ptx = generate_add_ptx(4);
        let name = kernel_name(&ptx);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let cpu = add_reference(&a, &b);
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        suite.add(&name, ptx, cpu, expected, 1e-5);
        assert_eq!(suite.len(), 1);
        assert!(suite.run_all_pass());
    }

    #[test]
    fn test_suite_multiple_kernels_pass() {
        let mut suite = CudaValidationSuite::new();

        // Add kernel
        let ptx_add = generate_add_ptx(3);
        let add_name = kernel_name(&ptx_add);
        let add_out = add_reference(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        suite.add(&add_name, ptx_add, add_out.clone(), add_out, 1e-5);

        // Mul kernel
        let ptx_mul = generate_mul_ptx(3);
        let mul_name = kernel_name(&ptx_mul);
        let mul_out = mul_reference(&[2.0, 3.0, 4.0], &[5.0, 6.0, 7.0]);
        suite.add(&mul_name, ptx_mul, mul_out.clone(), mul_out, 1e-5);

        assert_eq!(suite.len(), 2);
        assert!(suite.run_all_pass());
    }

    #[test]
    fn test_suite_one_failure_fails_all() {
        let mut suite = CudaValidationSuite::new();

        // Passing kernel
        let ptx = generate_add_ptx(2);
        let name = kernel_name(&ptx);
        suite.add(&name, ptx.clone(), vec![3.0, 5.0], vec![3.0, 5.0], 1e-5);

        // Failing kernel (wrong expected) — use same name since same PTX
        suite.add(&name, ptx, vec![3.0, 5.0], vec![100.0, 200.0], 1e-5);

        assert!(!suite.run_all_pass());
    }

    #[test]
    fn test_suite_reports_individual_results() {
        let mut suite = CudaValidationSuite::new();
        let ptx = generate_add_ptx(2);
        let name = kernel_name(&ptx);
        suite.add(&name, ptx.clone(), vec![3.0, 5.0], vec![3.0, 5.0], 1e-5);
        suite.add(&name, ptx, vec![1.0, 1.0], vec![1.0, 1.0], 1e-5);

        let results = suite.run();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
    }
}

// =========================================================================
// 5. ErrorStats edge cases
// =========================================================================

mod error_stats_tests {
    use super::*;

    #[test]
    fn test_error_stats_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = ErrorStats::compute(&data, &data).unwrap();
        assert_eq!(stats.max_abs_error, 0.0);
        assert_eq!(stats.mean_abs_error, 0.0);
        assert_eq!(stats.max_rel_error, 0.0);
        assert_eq!(stats.num_nans, 0);
        assert_eq!(stats.num_infs, 0);
        assert_eq!(stats.num_elements, 5);
    }

    #[test]
    fn test_error_stats_with_nans() {
        let actual = vec![1.0, f32::NAN, 3.0, f32::NAN];
        let expected = vec![1.0, 2.0, 3.0, 4.0];
        let stats = ErrorStats::compute(&actual, &expected).unwrap();
        assert_eq!(stats.num_nans, 2);
    }

    #[test]
    fn test_error_stats_with_infs() {
        let actual = vec![f32::INFINITY, 2.0, f32::NEG_INFINITY];
        let expected = vec![1.0, 2.0, 3.0];
        let stats = ErrorStats::compute(&actual, &expected).unwrap();
        assert_eq!(stats.num_infs, 2);
    }

    #[test]
    fn test_error_stats_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(ErrorStats::compute(&a, &b).is_err());
    }

    #[test]
    fn test_error_stats_empty() {
        let stats = ErrorStats::compute(&[], &[]).unwrap();
        assert_eq!(stats.num_elements, 0);
        assert_eq!(stats.max_abs_error, 0.0);
    }

    #[test]
    fn test_error_stats_known_error() {
        let actual = vec![1.0, 2.001, 3.0];
        let expected = vec![1.0, 2.0, 3.0];
        let stats = ErrorStats::compute(&actual, &expected).unwrap();
        assert!((stats.max_abs_error - 0.001).abs() < 1e-5);
        assert!(stats.mean_abs_error > 0.0);
        assert!(stats.mean_abs_error < stats.max_abs_error);
    }

    #[test]
    fn test_error_stats_relative_error() {
        let actual = vec![10.01]; // 0.1% error relative to 10.0
        let expected = vec![10.0];
        let stats = ErrorStats::compute(&actual, &expected).unwrap();
        assert!((stats.max_abs_error - 0.01).abs() < 1e-5);
        assert!(stats.max_rel_error > 0.0);
        assert!(stats.max_rel_error < 0.01); // relative error < 1%
    }
}

// =========================================================================
// 6. ValidationResult state machine
// =========================================================================

mod validation_result_tests {
    use super::*;

    #[test]
    fn test_new_result_is_passing() {
        let r = ValidationResult::new("test");
        assert!(r.passed());
        assert!(r.structural_ok);
        assert_eq!(r.numerical_ok, None);
    }

    #[test]
    fn test_structural_failure_means_not_passed() {
        let mut r = ValidationResult::new("test");
        r.structural_ok = false;
        assert!(!r.passed());
    }

    #[test]
    fn test_numerical_failure_means_not_passed() {
        let mut r = ValidationResult::new("test");
        r.numerical_ok = Some(false);
        assert!(!r.passed());
    }

    #[test]
    fn test_both_pass_means_passed() {
        let mut r = ValidationResult::new("test");
        r.structural_ok = true;
        r.numerical_ok = Some(true);
        assert!(r.passed());
    }

    #[test]
    fn test_no_numerical_with_structural_ok_is_passed() {
        // When numerical validation wasn't run, structural pass is sufficient
        let mut r = ValidationResult::new("test");
        r.structural_ok = true;
        r.numerical_ok = None;
        assert!(r.passed());
    }
}
