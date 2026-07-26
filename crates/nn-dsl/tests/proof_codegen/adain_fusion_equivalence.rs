// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for AdaIN + Snake fused equivalence semantics.

use nn_dsl::ir::MinMaxKind;
use nn_dsl::{
    adain_scalar, adain_snake_fused_scalar, build_adain_scalar_kernel,
    build_adain_snake_fused_kernel, snake_scalar, BinOpKind, IRNodeKind, NodeId, UnaryFnKind,
    SNAKE_MIN_ALPHA,
};

fn is_snake_min_alpha_literal(kind: &IRNodeKind) -> bool {
    matches!(
        kind,
        IRNodeKind::Literal(v) if (*v as f32).to_bits() == SNAKE_MIN_ALPHA.to_bits()
    )
}

fn fused_kernel_alpha_clamp_node(kernel: &nn_dsl::KernelDef) -> Option<NodeId> {
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

    let div_uses_clamp = kernel.nodes.iter().any(|node| {
        matches!(
            node.kind,
            IRNodeKind::BinOp {
                op: BinOpKind::Div,
                rhs,
                ..
            } if rhs == alpha_clamp
        )
    });
    assert!(
        div_uses_clamp,
        "fused kernel must divide by clamped alpha, not raw alpha"
    );

    let mul_with_clamp = kernel.nodes.iter().find_map(|node| match node.kind {
        IRNodeKind::BinOp {
            op: BinOpKind::Mul,
            lhs,
            rhs,
        } if lhs == alpha_clamp || rhs == alpha_clamp => Some(node.id),
        _ => None,
    });
    let mul_with_clamp =
        mul_with_clamp.expect("fused kernel must multiply y by clamped alpha before sin");

    let sin_uses_clamped_mul = kernel.nodes.iter().any(|node| {
        matches!(
            node.kind,
            IRNodeKind::UnaryFn {
                op: UnaryFnKind::Sin,
                input
            } if input == mul_with_clamp
        )
    });
    assert!(
        sin_uses_clamped_mul,
        "fused kernel sin input must use the clamped-alpha multiply node"
    );
}

#[test]
fn test_fused_matches_sequential_bit_exact() {
    // Bit-exact across representative alpha values, including edge-domain
    // values that trigger alpha clamping.
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
    ];

    for (case_index, (x, mu, var_val, gamma, beta, alpha, eps)) in cases.into_iter().enumerate() {
        let y = adain_scalar(x, mu, var_val, gamma, beta, eps).expect("valid test inputs");
        let sequential = snake_scalar(y, alpha).expect("finite test inputs");
        let fused = adain_snake_fused_scalar(x, mu, var_val, gamma, beta, alpha, eps)
            .expect("valid test inputs");

        assert_eq!(
            fused.to_bits(),
            sequential.to_bits(),
            "case {case_index}: fused ({fused}) and sequential ({sequential}) must be bit-exact",
        );
    }
}

#[test]
fn test_fused_matches_sequential_bit_exact_grid() {
    // Broad deterministic grid over the valid domain:
    // - var and eps keep denominator positive
    // - alpha stays within the proven Snake domain (>= SNAKE_MIN_ALPHA)
    let x_vals = [-10.0_f32, -1.25_f32, 0.0_f32, 1.5_f32, 7.75_f32];
    let mu_vals = [-2.0_f32, 0.0_f32, 2.0_f32];
    let var_vals = [1e-6_f32, 0.1_f32, 1.0_f32, 10.0_f32];
    let gamma_vals = [-3.0_f32, -0.5_f32, 0.5_f32, 3.0_f32];
    let beta_vals = [-1.0_f32, 0.0_f32, 1.0_f32];
    let alpha_vals = [SNAKE_MIN_ALPHA, 0.1_f32, 0.75_f32, 2.0_f32, 10.0_f32];
    let eps_vals = [0.0_f32, 1e-7_f32, 1e-5_f32, 1e-3_f32];

    let mut checked = 0usize;
    for x in x_vals {
        for mu in mu_vals {
            for var_val in var_vals {
                for gamma in gamma_vals {
                    for beta in beta_vals {
                        for alpha in alpha_vals {
                            for eps in eps_vals {
                                let y = adain_scalar(x, mu, var_val, gamma, beta, eps)
                                    .expect("grid inputs have var_val > 0");
                                let sequential =
                                    snake_scalar(y, alpha).expect("finite grid inputs");
                                let fused = adain_snake_fused_scalar(
                                    x, mu, var_val, gamma, beta, alpha, eps,
                                )
                                .expect("grid inputs have var_val > 0");
                                assert_eq!(
                                    fused.to_bits(),
                                    sequential.to_bits(),
                                    "bit mismatch for x={x}, mu={mu}, var={var_val}, gamma={gamma}, beta={beta}, alpha={alpha}, eps={eps}",
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    assert_eq!(
        checked, 14_400,
        "grid cardinality changed; update expected count if dimensions intentionally change"
    );
}
