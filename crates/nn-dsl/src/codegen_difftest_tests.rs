// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for differential test codegen (`codegen_difftest.rs`).

use super::*;
use crate::precision::InputBound;
use crate::test_kernels::parse_kernel as lower;

#[test]
fn test_difftest_contains_key_elements() {
    let kernel = lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
                x + (1.0 / alpha) * (alpha * x).sin().powi(2)
            }",
    );
    let test = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    assert!(
        test.contains("#[cfg(target_os = \"macos\")]"),
        "test:\n{test}"
    );
    assert!(test.contains("#[test]"), "test:\n{test}");
    assert!(test.contains("test_snake_differential"), "test:\n{test}");
    assert!(test.contains("SNAKE_DESCRIPTOR"), "test:\n{test}");
    assert!(
        test.contains("KernelPipeline::from_descriptor"),
        "test:\n{test}"
    );
    assert!(test.contains("dispatch_elementwise"), "test:\n{test}");
    assert!(test.contains("within_differential_budget"), "test:\n{test}");
}

#[test]
fn test_difftest_single_param() {
    let kernel = lower("fn relu(x: f32) -> f32 { x.max(0.0) }");
    let test = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    assert!(test.contains("test_relu_differential"), "test:\n{test}");
    assert!(test.contains("RELU_DESCRIPTOR"), "test:\n{test}");
    assert!(test.contains("x_data.as_slice()"), "test:\n{test}");
    assert!(
        !test.contains("alpha_data"),
        "single-param kernel should not reference alpha, test:\n{test}"
    );
}

#[test]
fn test_difftest_three_params() {
    let kernel = lower("fn add3(a: f32, b: f32, c: f32) -> f32 { a + b + c }");
    let test = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    assert!(test.contains("a_data"), "test:\n{test}");
    assert!(test.contains("b_data"), "test:\n{test}");
    assert!(test.contains("c_data"), "test:\n{test}");
    assert!(
        test.contains("&[a_data.as_slice(), b_data.as_slice(), c_data.as_slice()]"),
        "test:\n{test}"
    );
}

#[test]
fn test_difftest_has_random_samples() {
    let kernel = lower("fn relu(x: f32) -> f32 { x.max(0.0) }");
    let test = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    assert!(
        test.contains(&format!("let n: usize = {RANDOM_SAMPLE_COUNT}")),
        "should use {RANDOM_SAMPLE_COUNT} samples, test:\n{test}"
    );
}

#[test]
fn test_difftest_has_edge_cases() {
    let kernel = lower("fn relu(x: f32) -> f32 { x.max(0.0) }");
    let test = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    assert!(test.contains("single element"), "test:\n{test}");
    assert!(test.contains("threadgroup boundary"), "test:\n{test}");
    assert!(test.contains("denormal inputs"), "test:\n{test}");
    assert!(test.contains("bound edge values"), "test:\n{test}");
    assert!(test.contains("zero-length input"), "test:\n{test}");
}

#[test]
fn test_difftest_with_explicit_bounds() {
    let kernel = lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
                x + (1.0 / alpha) * (alpha * x).sin().powi(2)
            }",
    );
    let mut bounds = InputBounds::new();
    bounds.insert("x", InputBound::new(-1e4, 1e4).expect("valid bound"));
    bounds.insert("alpha", InputBound::new(1e-8, 1e3).expect("valid bound"));
    let test =
        emit_differential_test_with_bounds(&kernel, PrecisionTier::Normal, &bounds).expect("emit");
    assert!(test.contains("-10000"), "test:\n{test}");
    assert!(test.contains("10000"), "test:\n{test}");
}

#[test]
fn test_difftest_diagnostic_output() {
    let kernel = lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
                x + (1.0 / alpha) * (alpha * x).sin().powi(2)
            }",
    );
    let test = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    assert!(test.contains("x="), "test:\n{test}");
    assert!(test.contains("alpha="), "test:\n{test}");
    assert!(test.contains("rust="), "test:\n{test}");
    assert!(test.contains("metal="), "test:\n{test}");
    assert!(test.contains("delta="), "test:\n{test}");
}

// --- Syntax validation tests (syn-parse) ---

/// Parse generated code as valid Rust syntax — catches broken braces,
/// missing semicolons, and other syntax errors that substring checks miss.
fn assert_valid_rust_syntax(code: &str) {
    syn::parse_str::<syn::File>(code)
        .unwrap_or_else(|e| panic!("generated test is not valid Rust syntax: {e}\n\n{code}"));
}

#[test]
fn test_difftest_snake_valid_rust_syntax() {
    let kernel = lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
                x + (1.0 / alpha) * (alpha * x).sin().powi(2)
            }",
    );
    let test = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    assert_valid_rust_syntax(&test);
}

#[test]
fn test_difftest_single_param_valid_rust_syntax() {
    let kernel = lower("fn relu(x: f32) -> f32 { x.max(0.0) }");
    let test = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    assert_valid_rust_syntax(&test);
}

#[test]
fn test_difftest_explicit_bounds_valid_rust_syntax() {
    let kernel = lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
                x + (1.0 / alpha) * (alpha * x).sin().powi(2)
            }",
    );
    let mut bounds = InputBounds::new();
    bounds.insert("x", InputBound::new(-1e4, 1e4).expect("valid bound"));
    bounds.insert("alpha", InputBound::new(1e-8, 1e3).expect("valid bound"));
    let test =
        emit_differential_test_with_bounds(&kernel, PrecisionTier::Normal, &bounds).expect("emit");
    assert_valid_rust_syntax(&test);
}

#[test]
fn test_difftest_three_params_valid_rust_syntax() {
    let kernel = lower("fn add3(a: f32, b: f32, c: f32) -> f32 { a + b + c }");
    let test = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    assert_valid_rust_syntax(&test);
}

// --- Denormal bounds-awareness tests ---

#[test]
fn test_denormals_filtered_by_positive_only_bounds() {
    // Alpha is bounded to [0.1, 1e6], so all f32 denormal candidates
    // (1e-38, -1e-38, 5e-39) are below alpha's lower bound.  x has default
    // bounds [-1e6, 1e6] which include all candidates, but min_len = min(3, 0)
    // = 0, so the codegen falls back to a near-lo value (1 element per param).
    //
    // Uses `x * alpha` (Mul only) instead of snake (which has `1.0/alpha` = Div).
    // Div triggers has_ftz_sensitive_op() which skips the denormal section entirely,
    // masking the bounds-filtering fallback path we want to test here.
    let kernel = lower("fn scale(x: f32, alpha: f32) -> f32 { x * alpha }");
    let mut bounds = InputBounds::new();
    bounds.insert("alpha", InputBound::new(0.1, 1e6).expect("valid"));
    let test =
        emit_differential_test_with_bounds(&kernel, PrecisionTier::Normal, &bounds).expect("emit");
    // The denormal section should exist
    assert!(test.contains("denormal inputs"), "test:\n{test}");
    // With alpha excluding all denormals, min_len==0 triggers the fallback path.
    // Each param gets exactly 1 near-lo value, so each vec has 0 commas.
    let denorm_section = test
        .split("denormal inputs")
        .nth(1)
        .expect("denormal section should exist");
    // Find the alpha vec line (second vec in the section)
    let vec_lines: Vec<&str> = denorm_section
        .lines()
        .filter(|l| l.contains("vec![") && l.contains("_f32"))
        .collect();
    assert!(
        vec_lines.len() >= 2,
        "should have vec lines for x and alpha in denormal section"
    );
    // Both params should have exactly 1 element (the near-lo fallback)
    for line in &vec_lines {
        let commas = line.matches(',').count();
        assert_eq!(
            commas, 0,
            "fallback path should produce single-element vecs, line: {line}"
        );
    }
}

#[test]
fn test_denormals_included_with_default_bounds() {
    // Default bounds are [-1e6, 1e6], which include all f32 denormals.
    let kernel = lower("fn relu(x: f32) -> f32 { x.max(0.0) }");
    let test = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    assert!(test.contains("denormal inputs"), "test:\n{test}");
    // Default bounds include denormals, so multiple vec elements should appear
    // (the format uses Rust's Display trait, so 1e-38 renders as a long decimal).
    // Verify the denormal section has 3 elements (all 3 candidates are in-range).
    let denorm_section = test
        .split("denormal inputs")
        .nth(1)
        .expect("denormal section should exist");
    let first_vec_line = denorm_section
        .lines()
        .find(|l| l.contains("vec!["))
        .expect("should have a vec literal in denormal section");
    // Count comma-separated elements: 3 candidates means 2 commas
    let comma_count = first_vec_line.matches(',').count();
    assert_eq!(
        comma_count, 2,
        "default bounds should include all 3 denormal candidates, line: {first_vec_line}"
    );
}

#[test]
fn test_denormals_mixed_params_fallback() {
    // If one param excludes denormals but the other includes them,
    // the min_len logic should trigger fallback (min_len==0).
    let kernel = lower("fn add_scaled(x: f32, scale: f32) -> f32 { x * scale }");
    let mut bounds = InputBounds::new();
    // x: default [-1e6, 1e6] includes denormals
    // scale: [1.0, 10.0] excludes all denormals
    bounds.insert("scale", InputBound::new(1.0, 10.0).expect("valid"));
    let test =
        emit_differential_test_with_bounds(&kernel, PrecisionTier::Normal, &bounds).expect("emit");
    assert!(test.contains("denormal inputs"), "test:\n{test}");
    // Since scale has 0 valid denormals, min_len==0 triggers fallback.
    // All params get single-element near-lo vecs.
    let denorm_section = test
        .split("denormal inputs")
        .nth(1)
        .expect("denormal section should exist");
    let vec_lines: Vec<&str> = denorm_section
        .lines()
        .filter(|l| l.contains("vec![") && l.contains("_f32"))
        .collect();
    assert!(
        vec_lines.len() >= 2,
        "should have vec lines for x and scale"
    );
    for line in &vec_lines {
        let commas = line.matches(',').count();
        assert_eq!(
            commas, 0,
            "fallback should produce single-element vecs, line: {line}"
        );
    }
}

// --- Per-kernel codegen validation: verify emit_differential_test produces
// valid output for each of the 8 dvoice kernel builders (issue #22 AC1). ---

/// Helper: assert generated test code is valid Rust, contains the expected
/// descriptor name, and includes all 5 edge-case phases.
fn assert_kernel_difftest_codegen(kernel: &KernelDef, descriptor_name: &str) {
    let test_code = emit_differential_test(kernel, PrecisionTier::Normal)
        .expect("emit_differential_test should succeed");
    assert_valid_rust_syntax(&test_code);
    assert!(
        test_code.contains(descriptor_name),
        "generated code should reference {descriptor_name}, got:\n{test_code}"
    );
    assert!(
        test_code.contains("#[test]"),
        "missing #[test]:\n{test_code}"
    );
    assert!(
        test_code.contains("within_differential_budget"),
        "missing precision comparison:\n{test_code}"
    );
    // Edge cases
    assert!(
        test_code.contains("single element"),
        "missing single element edge case"
    );
    assert!(
        test_code.contains("threadgroup boundary"),
        "missing threadgroup boundary edge case"
    );
    assert!(
        test_code.contains("bound edge values"),
        "missing bound edge values"
    );
    assert!(
        test_code.contains("zero-length input"),
        "missing zero-length edge case"
    );
}

#[test]
fn test_difftest_codegen_k1_snake() {
    let kernel = crate::adain::build_snake_scalar_kernel().expect("build");
    assert_kernel_difftest_codegen(&kernel, "SNAKE_DESCRIPTOR");
}

#[test]
fn test_difftest_codegen_k2_instance_norm() {
    let kernel = crate::instance_norm::build_instance_norm_scalar_kernel().expect("build");
    assert_kernel_difftest_codegen(&kernel, "INSTANCE_NORM_SCALAR_DESCRIPTOR");
}

#[test]
fn test_difftest_codegen_k3_adain() {
    let kernel = crate::adain::build_adain_scalar_kernel().expect("build");
    assert_kernel_difftest_codegen(&kernel, "ADAIN_DESCRIPTOR");
}

#[test]
fn test_difftest_codegen_k4_adain_snake_fused() {
    let kernel = crate::adain::build_adain_snake_fused_kernel().expect("build");
    assert_kernel_difftest_codegen(&kernel, "ADAIN_SNAKE_DESCRIPTOR");
}

#[test]
fn test_difftest_codegen_k5_rms_norm() {
    let kernel = crate::rms_norm::build_rms_norm_scalar_kernel().expect("build");
    assert_kernel_difftest_codegen(&kernel, "RMS_NORM_SCALAR_DESCRIPTOR");
}

#[test]
fn test_difftest_codegen_k6_rope_cos() {
    let kernel = crate::rope::build_rope_cos_kernel().expect("build");
    assert_kernel_difftest_codegen(&kernel, "ROPE_COS_DESCRIPTOR");
}

#[test]
fn test_difftest_codegen_k6_rope_sin() {
    let kernel = crate::rope::build_rope_sin_kernel().expect("build");
    assert_kernel_difftest_codegen(&kernel, "ROPE_SIN_DESCRIPTOR");
}

#[test]
fn test_difftest_codegen_k7_layer_norm() {
    let kernel = crate::layer_norm::build_layer_norm_scalar_kernel().expect("build");
    assert_kernel_difftest_codegen(&kernel, "LAYER_NORM_SCALAR_DESCRIPTOR");
}

#[test]
fn test_difftest_codegen_k8_silu_mul() {
    let kernel = crate::silu_mul::build_silu_mul_kernel().expect("build");
    assert_kernel_difftest_codegen(&kernel, "SILU_MUL_DESCRIPTOR");
}

// --- Verify multi-parameter codegen produces correct input buffer refs ---

#[test]
fn test_difftest_codegen_adain_6_params() {
    // AdaIN has 6 parameters — verify all 6 get input vectors and dispatch buffers
    let kernel = crate::adain::build_adain_scalar_kernel().expect("build");
    let test_code = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    for param in ["x", "mu", "var_val", "gamma", "beta", "eps"] {
        assert!(
            test_code.contains(&format!("{param}_data")),
            "missing input vector for param '{param}':\n{test_code}"
        );
    }
}

#[test]
fn test_difftest_codegen_fused_7_params() {
    // AdaIN+Snake fused has 7 parameters — the widest kernel
    let kernel = crate::adain::build_adain_snake_fused_kernel().expect("build");
    let test_code = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    for param in ["x", "mu", "var_val", "gamma", "beta", "alpha", "eps"] {
        assert!(
            test_code.contains(&format!("{param}_data")),
            "missing input vector for param '{param}':\n{test_code}"
        );
    }
}

#[test]
fn test_difftest_codegen_instance_norm_4_params() {
    let kernel = crate::instance_norm::build_instance_norm_scalar_kernel().expect("build");
    let test_code = emit_differential_test(&kernel, PrecisionTier::Normal).expect("emit");
    for param in ["x", "mean", "var_val", "eps"] {
        assert!(
            test_code.contains(&format!("{param}_data")),
            "missing input vector for param '{param}':\n{test_code}"
        );
    }
}
