// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::ir::{BinOpKind, IRNodeKind, MinMaxKind, NodeId, UnaryFnKind};
use crate::snake::{snake_scalar, SNAKE_MIN_ALPHA};

fn is_snake_min_alpha_literal(kind: &IRNodeKind) -> bool {
    matches!(
        kind,
        IRNodeKind::Literal(v) if (*v as f32).to_bits() == SNAKE_MIN_ALPHA.to_bits()
    )
}

fn fused_kernel_alpha_clamp_node(kernel: &KernelDef) -> Option<NodeId> {
    let alpha_param_idx = kernel.params.iter().position(|param| param.name == "alpha");
    let alpha_param_idx = alpha_param_idx?;

    kernel.nodes.iter().find_map(|node| {
        let IRNodeKind::MinMax {
            op: MinMaxKind::Max,
            lhs,
            rhs,
        } = &node.kind
        else {
            return None;
        };

        let lhs_node = &kernel.nodes[lhs.index()].kind;
        let rhs_node = &kernel.nodes[rhs.index()].kind;

        let lhs_is_alpha = matches!(lhs_node, IRNodeKind::Param(idx) if *idx == alpha_param_idx);
        let rhs_is_alpha = matches!(rhs_node, IRNodeKind::Param(idx) if *idx == alpha_param_idx);
        let lhs_is_min_alpha = is_snake_min_alpha_literal(lhs_node);
        let rhs_is_min_alpha = is_snake_min_alpha_literal(rhs_node);

        if (lhs_is_alpha && rhs_is_min_alpha) || (rhs_is_alpha && lhs_is_min_alpha) {
            Some(node.id)
        } else {
            None
        }
    })
}

#[test]
fn test_adain_kernel_builds() {
    let kernel = build_adain_scalar_kernel().expect("adain should build");
    assert_eq!(kernel.params.len(), 6);
    assert_eq!(kernel.name, "adain");
}

#[test]
fn test_snake_kernel_builds() {
    let kernel = build_snake_scalar_kernel().expect("snake should build");
    assert_eq!(kernel.params.len(), 2);
    assert_eq!(kernel.name, "snake");
}

#[test]
fn test_snake_kernel_ir_has_alpha_clamp() {
    let kernel = build_snake_scalar_kernel().expect("snake should build");
    kernel.validate().expect("snake kernel IR should validate");
    fused_kernel_alpha_clamp_node(&kernel)
        .expect("snake kernel should clamp alpha via max(alpha, SNAKE_MIN_ALPHA)");
}

#[test]
fn test_fused_kernel_builds() {
    let kernel = build_adain_snake_fused_kernel().expect("fused should build");
    assert_eq!(kernel.params.len(), 7);
    assert_eq!(kernel.name, "adain_snake");
}

#[test]
fn test_adain_kernel_ir_has_rsqrt_and_add_output() {
    let kernel = build_adain_scalar_kernel().expect("adain should build");
    kernel.validate().expect("adain kernel IR should validate");

    let rsqrt_count = kernel
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.kind,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Rsqrt,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        rsqrt_count, 1,
        "adain kernel should contain exactly one rsqrt op"
    );

    let output = &kernel.nodes[kernel.output.index()];
    assert!(
        matches!(
            output.kind,
            IRNodeKind::BinOp {
                op: BinOpKind::Add,
                ..
            }
        ),
        "adain output should be an add node"
    );
}

#[test]
fn test_fused_kernel_ir_has_sin_and_powi() {
    let kernel = build_adain_snake_fused_kernel().expect("fused should build");
    kernel.validate().expect("fused kernel IR should validate");

    let has_sin = kernel.nodes.iter().any(|node| {
        matches!(
            node.kind,
            IRNodeKind::UnaryFn {
                op: UnaryFnKind::Sin,
                ..
            }
        )
    });
    assert!(has_sin, "fused kernel should contain a sin op");

    let has_powi2 = kernel
        .nodes
        .iter()
        .any(|node| matches!(node.kind, IRNodeKind::Powi { exp: 2, .. }));
    assert!(has_powi2, "fused kernel should contain powi(..., 2)");

    let alpha_clamp = fused_kernel_alpha_clamp_node(&kernel)
        .expect("fused kernel should clamp alpha via max(alpha, SNAKE_MIN_ALPHA)");
    assert_eq!(kernel.nodes[alpha_clamp.index()].id, alpha_clamp);
}

#[test]
fn test_fused_matches_sequential() {
    // Verify the reference implementations are consistent:
    // fused(x, mu, var, gamma, beta, alpha, eps) == snake(adain(x, mu, var, gamma, beta, eps), alpha)
    //
    // Bit-exact for valid alpha domain (alpha >= 0): both paths
    // execute the same IEEE 754 operations in the same order.
    let cases = [
        (
            1.5_f32, 0.5_f32, 0.25_f32, 1.2_f32, 0.1_f32, 2.0_f32, 1e-5_f32,
        ),
        (
            -0.8_f32, -0.5_f32, 0.9_f32, -1.0_f32, 0.3_f32, 0.75_f32, 1e-6_f32,
        ),
        (
            3.0_f32,
            1.0_f32,
            2.0_f32,
            0.4_f32,
            -0.2_f32,
            SNAKE_MIN_ALPHA,
            1e-4_f32,
        ),
        (
            -4.2_f32, 2.1_f32, 4.0_f32, 2.5_f32, -1.5_f32, 5.0_f32, 1e-3_f32,
        ),
        (
            2.0_f32, 0.0_f32, 1.0_f32, 1.0_f32, 0.0_f32, 0.0_f32, 1e-5_f32,
        ),
        (
            500.0_f32, 0.0_f32, 1.0_f32, 1.0_f32, 0.0_f32, 1e-10_f32, 1e-5_f32,
        ),
        (
            -2.000002_f32,
            0.0_f32,
            1.0_f32,
            1.0_f32,
            0.0_f32,
            9.192355_f32,
            1e-5_f32,
        ),
    ];

    for (case_index, (x, mu, var_val, gamma, beta, alpha, eps)) in cases.into_iter().enumerate() {
        let y = adain_scalar(x, mu, var_val, gamma, beta, eps)
            .expect("adain_scalar should succeed for valid inputs");
        let sequential =
            snake_scalar(y, alpha).expect("snake_scalar should succeed for valid inputs");
        let fused = adain_snake_fused_scalar(x, mu, var_val, gamma, beta, alpha, eps)
            .expect("adain_snake_fused_scalar should succeed for valid inputs");

        assert_eq!(
            fused.to_bits(),
            sequential.to_bits(),
            "case {case_index}: fused ({fused}) and sequential ({sequential}) must be bit-exact",
        );
    }
}

#[test]
fn test_adain_ref_known_values() {
    // x=1, mu=0, var=1, gamma=1, beta=0, eps=0 -> 1/sqrt(1) = 1.0
    let result = adain_scalar(1.0, 0.0, 1.0, 1.0, 0.0, 0.0).expect("var_val + eps = 1.0 > 0");
    assert!((result - 1.0).abs() < 1e-6, "got {result}");
}

/// Non-degenerate known-value test: all parameters nonzero, hand-computed.
///
/// Formula: gamma * (x - mu) * rsqrt(var_val + eps) + beta
///   = 2.0 * (3.0 - 1.0) * (1.0 / sqrt(4.01)) + 0.5
///   = 2.0 * 2.0 * 0.49937... + 0.5
///   ≈ 2.4975
///
/// A sign error in `(x - mu)` or a missing `gamma` would produce a wildly
/// different result. The previous test used mostly-zero params where such
/// errors would be invisible.
#[test]
fn test_adain_ref_known_values_nondegenerate() {
    let result = adain_scalar(3.0, 1.0, 4.0, 2.0, 0.5, 0.01)
        .expect("all params finite and var_val + eps > 0");
    // f64 reference: 2.0 * 2.0 / sqrt(4.01) + 0.5 ≈ 2.497505
    assert!(
        (result - 2.4975).abs() < 1e-3,
        "expected ≈ 2.4975, got {result}"
    );
    // Also test with negative gamma (should flip the sign of the (x-mu) contribution)
    let result_neg_gamma =
        adain_scalar(3.0, 1.0, 4.0, -2.0, 0.5, 0.01).expect("negative gamma is valid");
    // -2.0 * 2.0 / sqrt(4.01) + 0.5 ≈ -1.4975
    assert!(
        (result_neg_gamma - (-1.4975)).abs() < 1e-3,
        "expected ≈ -1.4975 with negative gamma, got {result_neg_gamma}"
    );
}

/// Regression: fused ref and sequential ref must remain bit-exact at alpha=0.
///
/// Both paths clamp `alpha` to `SNAKE_MIN_ALPHA`, so division-by-zero is avoided
/// and fusion equivalence holds even when the caller passes 0.
#[test]
fn test_fused_matches_sequential_at_zero_alpha() {
    let x = 2.0_f32;
    let mu = 0.0_f32;
    let var_val = 1.0_f32;
    let gamma = 1.0_f32;
    let beta = 0.0_f32;
    let eps = 1e-5_f32;
    let alpha = 0.0_f32;

    let y = adain_scalar(x, mu, var_val, gamma, beta, eps).expect("valid inputs");

    // Sequential path clamps alpha -> SNAKE_MIN_ALPHA.
    let sequential = snake_scalar(y, alpha).expect("finite inputs");
    assert!(
        sequential.is_finite(),
        "sequential (clamped) must be finite, got {sequential}"
    );

    // Fused path must also clamp and remain finite.
    let fused =
        adain_snake_fused_scalar(x, mu, var_val, gamma, beta, alpha, eps).expect("valid inputs");
    assert!(
        fused.is_finite(),
        "fused with alpha=0 must be finite after clamping, got {fused}"
    );
    assert_eq!(
        fused.to_bits(),
        sequential.to_bits(),
        "fused and sequential must be bit-exact at alpha=0 after clamping"
    );
}

/// Regression: fused and sequential remain bit-exact for small positive alpha
/// below `SNAKE_MIN_ALPHA` because both paths clamp before computing Snake.
#[test]
fn test_fused_matches_sequential_small_alpha_large_y() {
    // Use large x so the snake correction term (≈ alpha * y²) is measurable in f32.
    let x = 500.0_f32;
    let mu = 0.0_f32;
    let var_val = 1.0_f32;
    let gamma = 1.0_f32;
    let beta = 0.0_f32;
    let eps = 1e-5_f32;
    let alpha = 1e-10_f32;
    assert!(alpha < SNAKE_MIN_ALPHA);

    let y = adain_scalar(x, mu, var_val, gamma, beta, eps).expect("valid inputs");

    let sequential = snake_scalar(y, alpha).expect("finite inputs");
    let fused =
        adain_snake_fused_scalar(x, mu, var_val, gamma, beta, alpha, eps).expect("valid inputs");

    // Both finite and bit-exact because both paths clamp alpha to 1e-8.
    assert!(sequential.is_finite());
    assert!(fused.is_finite());
    assert_eq!(
        fused.to_bits(),
        sequential.to_bits(),
        "fused and sequential must be bit-exact for alpha={alpha} < SNAKE_MIN_ALPHA \
         after clamping: fused={fused}, sequential={sequential}"
    );
}

// Input validation tests (NaN/Inf rejection, negative denominator, extreme magnitudes)
// extracted to adain_tests_validation.rs (#1420).
#[path = "adain_tests_validation.rs"]
mod validation;

// --- MSL codegen tests ---

#[test]
fn test_adain_kernel_msl_codegen() {
    let kernel = build_adain_scalar_kernel().expect("build must succeed");
    let msl = crate::emit_msl(&kernel).expect("MSL codegen");
    assert!(msl.contains("adain"), "MSL should contain kernel name");
    assert!(
        msl.contains("rsqrt("),
        "MSL should use rsqrt for normalization"
    );
    assert!(
        msl.contains("[[kernel]]"),
        "MSL should have kernel attribute",
    );
}

#[test]
fn test_adain_snake_fused_kernel_msl_codegen() {
    let kernel = build_adain_snake_fused_kernel().expect("build must succeed");
    let msl = crate::emit_msl(&kernel).expect("MSL codegen");
    assert!(
        msl.contains("adain_snake"),
        "MSL should contain kernel name"
    );
    assert!(
        msl.contains("sin("),
        "MSL should use sin for snake activation"
    );
    assert!(
        msl.contains("[[kernel]]"),
        "MSL should have kernel attribute",
    );
}

// --- Tensor-level builder tests (#733) ---
// Extracted to `adain_tests_tensor.rs` to keep this file under 500 lines.

#[path = "adain_tests_tensor.rs"]
mod tensor;
