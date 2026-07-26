// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harness generation from KernelIR.
//!
//! Generates a `#[kani::proof]` harness that:
//! 1. Creates symbolic inputs via `kani::any()`
//! 2. Assumes inputs are finite and bounded
//! 3. Calls the original kernel function
//! 4. Asserts the result is finite (no NaN, no Inf)

use crate::ir::{IRError, KernelDef, ScalarType};

/// Emit a Kani proof harness as Rust source text.
///
/// The harness calls the original function (not a reconstructed one), so it
/// verifies the actual Rust implementation. The default input bound is `1e6`
/// which covers the practical domain for ML kernel inputs.
///
/// # Errors
///
/// Returns `IRError` if the kernel IR is malformed.
#[must_use = "Kani harness string is computed but not used"]
pub fn emit_kani_harness(kernel: &KernelDef) -> Result<String, IRError> {
    emit_kani_harness_with_bound(kernel, 1e6)
}

/// Emit a Kani proof harness with a custom input magnitude bound.
///
/// # Errors
///
/// Returns `IRError` if the kernel IR is malformed.
#[must_use = "Kani harness string is computed but not used"]
pub(crate) fn emit_kani_harness_with_bound(
    kernel: &KernelDef,
    bound: f64,
) -> Result<String, IRError> {
    kernel.validate()?;
    let mut lines = Vec::new();

    lines.push("#[cfg(kani)]".to_string());
    lines.push("#[kani::proof]".to_string());
    lines.push(format!("fn kani_verify_{}() {{", kernel.name));

    for param in &kernel.params {
        match param.ty {
            ScalarType::F32 => {
                lines.push(format!("    let {}: f32 = kani::any();", param.name));
                lines.push(format!(
                    "    kani::assume({name}.is_finite() && {name}.abs() < {bound}_f32);",
                    name = param.name,
                ));
            }
            ScalarType::F16 => {
                let sym = format!("{}_f32", param.name);
                lines.push(format!("    let {sym}: f32 = kani::any();"));
                lines.push(format!(
                    "    kani::assume({sym}.is_finite() && {sym}.abs() < {bound}_f32);"
                ));
                lines.push(format!(
                    "    let {}: half::f16 = half::f16::from_f32({sym});",
                    param.name
                ));
            }
            ScalarType::BF16 => {
                let sym = format!("{}_f32", param.name);
                lines.push(format!("    let {sym}: f32 = kani::any();"));
                lines.push(format!(
                    "    kani::assume({sym}.is_finite() && {sym}.abs() < {bound}_f32);"
                ));
                lines.push(format!(
                    "    let {}: half::bf16 = half::bf16::from_f32({sym});",
                    param.name
                ));
            }
        }
    }

    let call_args: Vec<&str> = kernel.params.iter().map(|p| p.name.as_str()).collect();
    lines.push(format!(
        "    let result = {}({});",
        kernel.name,
        call_args.join(", ")
    ));
    lines.push("    assert!(result.is_finite(), \"kernel output must be finite\");".to_string());
    lines.push("}".to_string());

    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_kernels::parse_kernel as lower;

    /// Parse a generated harness as valid Rust syntax. This catches broken
    /// braces, missing semicolons, and other syntax errors that substring
    /// checks would miss.
    fn assert_valid_rust_syntax(harness: &str) {
        syn::parse_str::<syn::File>(harness).unwrap_or_else(|e| {
            panic!("generated harness is not valid Rust syntax: {e}\n\n{harness}")
        });
    }

    #[test]
    fn test_snake_kani_harness() {
        let kernel = lower(
            "fn snake(x: f32, alpha: f32) -> f32 {
                x + (1.0 / alpha) * (alpha * x).sin().powi(2)
            }",
        );
        let harness = emit_kani_harness(&kernel).expect("emit");
        assert_valid_rust_syntax(&harness);
        assert!(harness.contains("#[cfg(kani)]"), "harness:\n{harness}");
        assert!(harness.contains("#[kani::proof]"), "harness:\n{harness}");
        assert!(harness.contains("kani_verify_snake"), "harness:\n{harness}");
        assert!(harness.contains("kani::any()"), "harness:\n{harness}");
        assert!(harness.contains("kani::assume("), "harness:\n{harness}");
        assert!(
            harness.contains("result.is_finite()"),
            "harness:\n{harness}"
        );
        assert!(harness.contains("snake(x, alpha)"), "harness:\n{harness}");
    }

    #[test]
    fn test_single_param_harness() {
        let kernel = lower("fn relu(x: f32) -> f32 { x.max(0.0) }");
        let harness = emit_kani_harness(&kernel).expect("emit");
        assert_valid_rust_syntax(&harness);
        assert!(harness.contains("relu(x)"), "harness:\n{harness}");
        // Should not reference 'alpha'
        assert!(!harness.contains("alpha"), "harness:\n{harness}");
    }

    #[test]
    fn test_custom_bound() {
        let kernel = lower("fn small(x: f32) -> f32 { x }");
        let harness = emit_kani_harness_with_bound(&kernel, 100.0).expect("emit");
        assert_valid_rust_syntax(&harness);
        // Check bound appears in the kani::assume line specifically, not just anywhere
        assert!(harness.contains("100_f32"), "harness:\n{harness}");
    }

    #[test]
    fn test_f16_inputs_are_symbolic_f32_then_converted() {
        let kernel = lower("fn half_add(x: f16, y: f16) -> f16 { x + y }");
        let harness = emit_kani_harness(&kernel).expect("emit");
        assert_valid_rust_syntax(&harness);
        assert!(
            harness.contains("let x_f32: f32 = kani::any();"),
            "harness:\n{harness}"
        );
        assert!(
            harness.contains("let y_f32: f32 = kani::any();"),
            "harness:\n{harness}"
        );
        assert!(
            harness.contains("let x: half::f16 = half::f16::from_f32(x_f32);"),
            "harness:\n{harness}"
        );
        assert!(
            harness.contains("let y: half::f16 = half::f16::from_f32(y_f32);"),
            "harness:\n{harness}"
        );
        assert!(harness.contains("half_add(x, y)"), "harness:\n{harness}");
    }

    #[test]
    fn test_zero_param_kernel_harness() {
        let kernel = KernelDef {
            name: "literal_42".to_string(),
            params: vec![],
            return_type: ScalarType::F32,
            nodes: vec![crate::ir::IRNode {
                id: crate::ir::NodeId::new(0),
                kind: crate::ir::IRNodeKind::Literal(42.0),
            }],
            output: crate::ir::NodeId::new(0),
        };
        let harness = emit_kani_harness(&kernel).expect("emit");
        assert_valid_rust_syntax(&harness);
        assert!(harness.contains("literal_42()"), "harness:\n{harness}");
        // No kani::any() calls for zero-param kernel
        assert!(!harness.contains("kani::any()"), "harness:\n{harness}");
    }

    #[test]
    fn test_mixed_f32_f16_param_harness() {
        let kernel = KernelDef {
            name: "mixed".to_string(),
            params: vec![
                crate::ir::Param {
                    name: "x".to_string(),
                    ty: ScalarType::F32,
                },
                crate::ir::Param {
                    name: "y".to_string(),
                    ty: ScalarType::F16,
                },
            ],
            return_type: ScalarType::F32,
            nodes: vec![
                crate::ir::IRNode {
                    id: crate::ir::NodeId::new(0),
                    kind: crate::ir::IRNodeKind::Param(0),
                },
                crate::ir::IRNode {
                    id: crate::ir::NodeId::new(1),
                    kind: crate::ir::IRNodeKind::Param(1),
                },
            ],
            output: crate::ir::NodeId::new(0),
        };
        let harness = emit_kani_harness(&kernel).expect("emit");
        assert_valid_rust_syntax(&harness);
        // x is direct f32
        assert!(
            harness.contains("let x: f32 = kani::any();"),
            "harness:\n{harness}"
        );
        // y goes through f32 intermediate
        assert!(
            harness.contains("let y_f32: f32 = kani::any();"),
            "harness:\n{harness}"
        );
        assert!(
            harness.contains("let y: half::f16 = half::f16::from_f32(y_f32);"),
            "harness:\n{harness}"
        );
        assert!(harness.contains("mixed(x, y)"), "harness:\n{harness}");
    }
}
