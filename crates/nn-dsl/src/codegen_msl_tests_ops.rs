// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for MSL code generation of operator-specific patterns:
//! powi binary exponentiation, Compare, and Select.

use super::*;
use crate::ir::{
    BinaryFnKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType,
};
use crate::test_kernels::parse_kernel as lower;

// ======================== powi binary exponentiation ========================

#[test]
fn test_powi_2_inlines_as_multiply() {
    let kernel = lower("fn sq(x: f32) -> f32 { x.powi(2) }");
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    // powi(2) stays inline: x * x
    assert!(
        scalar.contains("x * x"),
        "powi(2) should inline as x * x, scalar:\n{scalar}"
    );
}

#[test]
fn test_powi_8_uses_binary_exponentiation() {
    // Build a kernel with powi(8) manually to test binary exponentiation.
    let kernel = KernelDef::new(
        "pow8",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 8,
                },
            ),
        ],
        NodeId::new(1),
    );
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    // powi(8) = (((x^2)^2)^2): three squarings, not seven multiplications.
    // Should have intermediate temporaries and NOT an 8-way multiplication chain.
    assert!(
        !scalar.contains("x * x * x * x * x * x * x * x"),
        "powi(8) should use binary exponentiation, not O(n) chain, scalar:\n{scalar}"
    );
    // Should have squaring temporaries
    assert!(
        scalar.contains("_p2") && scalar.contains("_p4"),
        "powi(8) should generate squaring temporaries, scalar:\n{scalar}"
    );
}

#[test]
fn test_powi_7_uses_binary_exponentiation() {
    // 7 = 0b111 = base * base^2 * base^4: tests non-power-of-2 exponent
    let kernel = KernelDef::new(
        "pow7",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 7,
                },
            ),
        ],
        NodeId::new(1),
    );
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    // 7 = 1 + 2 + 4: should have base, p2, and p4 temporaries multiplied
    assert!(
        scalar.contains("_base") && scalar.contains("_p2") && scalar.contains("_p4"),
        "powi(7) should decompose to base * p2 * p4, scalar:\n{scalar}"
    );
}

#[test]
fn test_powi_neg8_uses_binary_exponentiation_with_reciprocal() {
    let kernel = KernelDef::new(
        "inv_pow8",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: -8,
                },
            ),
        ],
        NodeId::new(1),
    );
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    // Should use binary exponentiation wrapped in reciprocal
    assert!(
        scalar.contains("float(1) /"),
        "powi(-8) should emit reciprocal, scalar:\n{scalar}"
    );
    assert!(
        scalar.contains("_p2"),
        "powi(-8) should use binary exponentiation, scalar:\n{scalar}"
    );
}

#[test]
fn test_powi_65_rejected_by_validation() {
    use crate::ir::POWI_MAX_EXPONENT;
    let kernel = KernelDef::new(
        "too_large",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: (POWI_MAX_EXPONENT as i32) + 1,
                },
            ),
        ],
        NodeId::new(1),
    );
    let err = emit_msl(&kernel).unwrap_err();
    assert!(
        err.to_string().contains("exceeds maximum"),
        "expected exponent-too-large error, got: {err}"
    );
}

#[test]
fn test_powi_neg65_rejected_by_validation() {
    use crate::ir::POWI_MAX_EXPONENT;
    let kernel = KernelDef::new(
        "too_large_neg",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: -((POWI_MAX_EXPONENT as i32) + 1),
                },
            ),
        ],
        NodeId::new(1),
    );
    let err = emit_msl(&kernel).unwrap_err();
    assert!(
        err.to_string().contains("exceeds maximum"),
        "expected exponent-too-large error for negative exp, got: {err}"
    );
}

#[test]
fn test_powi_64_accepted_by_validation() {
    use crate::ir::POWI_MAX_EXPONENT;
    let kernel = KernelDef::new(
        "at_limit",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: POWI_MAX_EXPONENT as i32,
                },
            ),
        ],
        NodeId::new(1),
    );
    // Exactly at the limit should succeed
    assert!(
        emit_msl(&kernel).is_ok(),
        "powi({POWI_MAX_EXPONENT}) should be accepted"
    );
}

#[test]
fn test_f16_snake_uses_float_accumulator() {
    use crate::adain::build_snake_scalar_kernel;
    let kernel = build_snake_scalar_kernel().expect("build");
    let mut f16_kernel = kernel;
    f16_kernel.return_type = ScalarType::F16;
    for p in &mut f16_kernel.params {
        p.ty = ScalarType::F16;
    }
    let msl = emit_msl(&f16_kernel).expect("emit f16 MSL");
    // Float-accumulator mode: F16 scalar functions compute in float internally.
    // The 1e-8 literal is representable in float (no F16 clamping needed).
    assert!(
        msl.contains("0.00000001"),
        "float-accumulator F16 MSL should preserve the 1e-8 literal, MSL:\n{msl}"
    );
    // Function signature stays half (buffer types unchanged).
    assert!(
        msl.contains("half _nn_snake(half y, half alpha)"),
        "F16 function signature should use half, MSL:\n{msl}"
    );
    // Internal computation uses float.
    assert!(
        msl.contains("float y_f = float(y);"),
        "F16 should promote params to float, MSL:\n{msl}"
    );
    assert!(
        msl.contains("return half("),
        "F16 should demote result to half, MSL:\n{msl}"
    );
}

/// Verify that f32 snake MSL codegen embeds the SNAKE_MIN_ALPHA clamp.
/// Regression test for #325 AC1: the production codegen path
/// (`build_snake_scalar_kernel` → `emit_msl`) must include `max(alpha, 1e-8)`
/// so that any consumer of the generated MSL enforces the proof precondition.
#[test]
fn test_f32_snake_clamp_in_generated_msl() {
    use crate::adain::build_snake_scalar_kernel;
    let kernel = build_snake_scalar_kernel().expect("build");
    let msl = emit_msl(&kernel).expect("emit f32 MSL");
    // The generated MSL must contain `max()` (the clamp call) AND the
    // SNAKE_MIN_ALPHA literal `0.00000001` (format_float output for 1e-8).
    // Checking both prevents false passes from unrelated max() calls.
    assert!(
        msl.contains("max("),
        "f32 snake MSL missing max() clamp for SNAKE_MIN_ALPHA:\n{msl}"
    );
    assert!(
        msl.contains("0.00000001"),
        "f32 snake MSL missing SNAKE_MIN_ALPHA literal (1e-8 = 0.00000001):\n{msl}"
    );
}

// ======================== Compare MSL codegen ========================

#[test]
fn test_compare_gt_msl_emits_bool_with_operator() {
    // Compare(Gt, x, 0.0) should emit: bool t2 = x > 0.0;
    let kernel = KernelDef::new(
        "cmp_gt",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(3),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(4),
    );
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    // Literal(0.0) gets temp t1, so Compare emits: bool t2 = x > t1;
    assert!(
        scalar.contains("bool t2 = x > t1;"),
        "Compare(Gt) should emit bool with > operator, scalar:\n{scalar}"
    );
}

#[test]
fn test_compare_all_ops_emit_correct_operators() {
    use CompareOpKind::*;
    let ops_and_symbols: &[(CompareOpKind, &str)] = &[
        (Gt, ">"),
        (Ge, ">="),
        (Lt, "<"),
        (Le, "<="),
        (Eq, "=="),
        (Ne, "!="),
    ];
    for (op, expected_sym) in ops_and_symbols {
        // Wrap Compare in Select to produce F32 output (type-checks correctly).
        let kernel = KernelDef::new(
            format!("cmp_{op:?}"),
            vec![Param::new("x", ScalarType::F32)],
            ScalarType::F32,
            vec![
                IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
                IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.0)),
                IRNode::new(
                    NodeId::new(2),
                    IRNodeKind::Compare {
                        op: *op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
                IRNode::new(NodeId::new(3), IRNodeKind::Literal(0.0)),
                IRNode::new(
                    NodeId::new(4),
                    IRNodeKind::Select {
                        cond: NodeId::new(2),
                        then_val: NodeId::new(1),
                        else_val: NodeId::new(3),
                    },
                ),
            ],
            NodeId::new(4),
        );
        let scalar = emit_scalar_fn(&kernel).expect("emit");
        // Literal(1.0) gets temp t1, so Compare emits: bool t2 = x <op> t1;
        let expected_pattern = format!("bool t2 = x {expected_sym} t1;");
        assert!(
            scalar.contains(&expected_pattern),
            "Compare({op:?}) should emit `{expected_pattern}`, scalar:\n{scalar}"
        );
    }
}

// ======================== Select MSL codegen ========================

#[test]
fn test_select_msl_emits_ternary() {
    // Select(cond, then, else) should emit: float t4 = (t2 ? 1.0 : 0.0);
    let kernel = KernelDef::new(
        "sel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(3),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(4),
    );
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    // t1=0.0, t2=compare, t3=1.0 → Select emits: (t2 ? t3 : t1)
    assert!(
        scalar.contains("(t2 ? t3 : t1)"),
        "Select should emit ternary with temp refs, scalar:\n{scalar}"
    );
}

#[test]
fn test_leaky_relu_msl_emits_compare_and_select() {
    // LeakyReLU: if x > 0 { x } else { 0.01 * x }
    let kernel = lower("fn leaky_relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.01 * x } }");
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    assert!(
        scalar.contains("bool"),
        "LeakyReLU should have a bool comparison, scalar:\n{scalar}"
    );
    assert!(
        scalar.contains("?"),
        "LeakyReLU should have a ternary select, scalar:\n{scalar}"
    );
}

// ======================== BinaryFn (atan2) ========================

#[test]
fn test_atan2_msl_emits_native_intrinsic() {
    // Build IR directly: fn atan2(y: f32, x: f32) -> f32 { atan2(y, x) }
    let kernel = KernelDef::new(
        "atan2",
        vec![
            Param::new("y", ScalarType::F32),
            Param::new("x", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinaryFn {
                    op: BinaryFnKind::Atan2,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let scalar = emit_scalar_fn(&kernel).expect("emit atan2");
    assert!(
        scalar.contains("atan2("),
        "atan2 should emit native MSL atan2() call, scalar:\n{scalar}"
    );
}
