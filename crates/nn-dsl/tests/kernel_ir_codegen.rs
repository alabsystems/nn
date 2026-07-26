// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for KernelIR construction, code generation, and
//! verification metadata.
//!
//! Covers:
//! - KernelIR construction (all node kinds)
//! - MSL code generation (scalar + wrapper)
//! - Kani harness template generation
//! - Pretty-print round-trip consistency
//! - FTZ sensitivity detection
//! - Serde round-trip for KernelDef
//! - Verifiability classification
//! - Fusion detection (is_fusible_elementwise)
//! - ScalarType / UnaryFnKind lookup tables

use nn_dsl::{
    BinOpKind, CompareOpKind, IRError, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param,
    ScalarType, UnaryFnKind, POWI_MAX_EXPONENT,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn n(id: usize, kind: IRNodeKind) -> IRNode {
    IRNode::new(NodeId::new(id), kind)
}

fn f32_param(name: &str) -> Param {
    Param::new(name.to_string(), ScalarType::F32)
}

// ---------------------------------------------------------------------------
// A. KernelIR construction: every node kind
// ---------------------------------------------------------------------------

/// Build a kernel exercising every IRNodeKind variant and validate it.
#[test]
fn test_all_node_kinds_construction_and_validation() {
    // kernel all_ops(x: f32, y: f32) -> f32
    // Exercises: Param, Literal, BinOp, Compare, UnaryFn, Powi, Clamp,
    //            MinMax, Select, SumReduce, BinaryFn
    let kernel = KernelDef::new(
        "all_ops",
        vec![f32_param("x"), f32_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),      // x
            n(1, IRNodeKind::Param(1)),      // y
            n(2, IRNodeKind::Literal(1.0)),  // 1.0
            n(3, IRNodeKind::Literal(0.0)),  // 0.0
            n(4, IRNodeKind::Literal(-1.0)), // -1.0
            n(
                5,
                IRNodeKind::BinOp {
                    // x + y
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            n(
                6,
                IRNodeKind::BinOp {
                    // (x+y) * 1.0
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(5),
                    rhs: NodeId::new(2),
                },
            ),
            n(
                7,
                IRNodeKind::UnaryFn {
                    // sin(x+y)
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(5),
                },
            ),
            n(
                8,
                IRNodeKind::Powi {
                    // sin(x+y)^2
                    base: NodeId::new(7),
                    exp: 2,
                },
            ),
            n(
                9,
                IRNodeKind::Compare {
                    // x > 0
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(3),
                },
            ),
            n(
                10,
                IRNodeKind::Select {
                    // x>0 ? sin^2 : 0
                    cond: NodeId::new(9),
                    then_val: NodeId::new(8),
                    else_val: NodeId::new(3),
                },
            ),
            n(
                11,
                IRNodeKind::Clamp {
                    // clamp(select, -1, 1)
                    input: NodeId::new(10),
                    min: NodeId::new(4),
                    max: NodeId::new(2),
                },
            ),
            n(
                12,
                IRNodeKind::MinMax {
                    // max(clamp, 0)
                    op: MinMaxKind::Max,
                    lhs: NodeId::new(11),
                    rhs: NodeId::new(3),
                },
            ),
            n(
                13,
                IRNodeKind::SumReduce {
                    // sum(mul, max)
                    inputs: vec![NodeId::new(6), NodeId::new(12)],
                },
            ),
            n(
                14,
                IRNodeKind::BinaryFn {
                    // atan2(sum, y)
                    op: nn_dsl::ir::BinaryFnKind::Atan2,
                    lhs: NodeId::new(13),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(14),
    );
    kernel
        .validate()
        .expect("kernel with all node kinds should validate");
}

/// Verify that NodeId round-trips through index().
#[test]
fn test_node_id_index_roundtrip() {
    for i in [0, 1, 42, 999, usize::MAX] {
        let id = NodeId::new(i);
        assert_eq!(id.index(), i);
    }
}

/// BinOp: all four arithmetic ops in isolation.
#[test]
fn test_binop_all_variants_validate() {
    for op in [
        BinOpKind::Add,
        BinOpKind::Sub,
        BinOpKind::Mul,
        BinOpKind::Div,
    ] {
        let kernel = KernelDef::new(
            format!("binop_{op:?}"),
            vec![f32_param("a"), f32_param("b")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Param(1)),
                n(
                    2,
                    IRNodeKind::BinOp {
                        op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );
        kernel
            .validate()
            .unwrap_or_else(|e| panic!("BinOp {op:?} failed: {e}"));
    }
}

/// Compare: all six comparison ops.
#[test]
fn test_compare_all_variants_validate() {
    for op in [
        CompareOpKind::Eq,
        CompareOpKind::Ne,
        CompareOpKind::Lt,
        CompareOpKind::Le,
        CompareOpKind::Gt,
        CompareOpKind::Ge,
    ] {
        let kernel = KernelDef::new(
            format!("cmp_{op:?}"),
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Literal(0.0)),
                n(
                    2,
                    IRNodeKind::Compare {
                        op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
                n(3, IRNodeKind::Literal(1.0)),
                n(
                    4,
                    IRNodeKind::Select {
                        cond: NodeId::new(2),
                        then_val: NodeId::new(3),
                        else_val: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(4),
        );
        kernel
            .validate()
            .unwrap_or_else(|e| panic!("Compare {op:?} failed: {e}"));
    }
}

/// UnaryFn: all 13 variants.
#[test]
fn test_unary_fn_all_variants_validate() {
    let all_ops = [
        UnaryFnKind::Sin,
        UnaryFnKind::Cos,
        UnaryFnKind::Sqrt,
        UnaryFnKind::Rsqrt,
        UnaryFnKind::Exp,
        UnaryFnKind::Abs,
        UnaryFnKind::Recip,
        UnaryFnKind::Tanh,
        UnaryFnKind::Log,
        UnaryFnKind::Floor,
        UnaryFnKind::Round,
        UnaryFnKind::Fract,
        UnaryFnKind::Neg,
    ];
    for op in all_ops {
        let kernel = KernelDef::new(
            format!("unary_{}", op.method_name()),
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(
                    1,
                    IRNodeKind::UnaryFn {
                        op,
                        input: NodeId::new(0),
                    },
                ),
            ],
            NodeId::new(1),
        );
        kernel
            .validate()
            .unwrap_or_else(|e| panic!("UnaryFn {op:?} failed: {e}"));
    }
}

/// Powi: positive, negative, and zero exponents.
#[test]
fn test_powi_various_exponents_validate() {
    for exp in [-3_i32, -2, -1, 0, 1, 2, 3, 10, 64] {
        let name = if exp < 0 {
            format!("powi_neg{}", exp.unsigned_abs())
        } else {
            format!("powi_{exp}")
        };
        let kernel = KernelDef::new(
            &name,
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(
                    1,
                    IRNodeKind::Powi {
                        base: NodeId::new(0),
                        exp,
                    },
                ),
            ],
            NodeId::new(1),
        );
        kernel
            .validate()
            .unwrap_or_else(|e| panic!("Powi exp={exp} failed: {e}"));
    }
}

/// Powi: exponent exceeding POWI_MAX_EXPONENT is rejected.
#[test]
fn test_powi_excessive_exponent_rejected() {
    let exp = (POWI_MAX_EXPONENT as i32) + 1;
    let kernel = KernelDef::new(
        "powi_too_large",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp,
                },
            ),
        ],
        NodeId::new(1),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(err, IRError::PowiExponentTooLarge { .. }),
        "expected PowiExponentTooLarge, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// B. MSL code generation
// ---------------------------------------------------------------------------

/// Snake kernel MSL: contains sin, powi expansion, metal prelude.
#[test]
fn test_msl_snake_kernel_structure() {
    let kernel = nn_dsl::test_kernels::parse_kernel(
        "fn snake(x: f32, alpha: f32) -> f32 {
            x + (1.0 / alpha) * (alpha * x).sin().powi(2)
        }",
    );
    let msl = nn_dsl::emit_msl(&kernel).expect("emit_msl");

    // Metal prelude
    assert!(msl.contains("#include <metal_stdlib>"), "MSL:\n{msl}");
    assert!(msl.contains("using namespace metal;"), "MSL:\n{msl}");

    // Scalar helper (prefixed with _nn_ to avoid MSL builtin collision)
    assert!(msl.contains("_nn_snake"), "MSL:\n{msl}");

    // Kernel entry point
    assert!(msl.contains("[[kernel]]"), "MSL:\n{msl}");
    assert!(msl.contains("snake_kernel"), "MSL:\n{msl}");

    // sin should use precise intrinsic by default (Normal tier)
    assert!(msl.contains("metal::precise::sin"), "MSL:\n{msl}");

    // powi(2) should expand to multiplication, not metal::pow
    assert!(!msl.contains("metal::pow"), "MSL:\n{msl}");
}

/// Compare + Select produces MSL bool and ternary operator.
#[test]
fn test_msl_compare_select_codegen() {
    let kernel = KernelDef::new(
        "relu_cond",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Literal(0.0)),
            n(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            n(
                3,
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(3),
    );
    let scalar = nn_dsl::emit_scalar_fn(&kernel).expect("emit");
    // Compare should produce a bool variable
    assert!(
        scalar.contains("bool t2"),
        "expected bool for compare, scalar:\n{scalar}"
    );
    // Select should produce a ternary
    assert!(
        scalar.contains("?"),
        "expected ternary for select, scalar:\n{scalar}"
    );
}

/// BinaryFn::Atan2 generates MSL atan2() call.
#[test]
fn test_msl_atan2_codegen() {
    let kernel = KernelDef::new(
        "atan2_test",
        vec![f32_param("y"), f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::BinaryFn {
                    op: nn_dsl::ir::BinaryFnKind::Atan2,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let scalar = nn_dsl::emit_scalar_fn(&kernel).expect("emit");
    assert!(
        scalar.contains("atan2("),
        "expected atan2 call, scalar:\n{scalar}"
    );
}

/// SumReduce with 3 inputs generates an addition chain.
#[test]
fn test_msl_sum_reduce_three_inputs() {
    let kernel = KernelDef::new(
        "sum3",
        vec![f32_param("a"), f32_param("b"), f32_param("c")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(2, IRNodeKind::Param(2)),
            n(
                3,
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
                },
            ),
        ],
        NodeId::new(3),
    );
    let scalar = nn_dsl::emit_scalar_fn(&kernel).expect("emit");
    assert!(
        scalar.contains("a + b + c"),
        "expected explicit add chain, scalar:\n{scalar}"
    );
}

/// Clamp emits MSL clamp() intrinsic.
#[test]
fn test_msl_clamp_codegen() {
    let kernel = KernelDef::new(
        "clamp_test",
        vec![f32_param("x"), f32_param("lo"), f32_param("hi")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(2, IRNodeKind::Param(2)),
            n(
                3,
                IRNodeKind::Clamp {
                    input: NodeId::new(0),
                    min: NodeId::new(1),
                    max: NodeId::new(2),
                },
            ),
        ],
        NodeId::new(3),
    );
    let scalar = nn_dsl::emit_scalar_fn(&kernel).expect("emit");
    assert!(
        scalar.contains("clamp("),
        "expected clamp() call, scalar:\n{scalar}"
    );
}

/// MinMax::Min and MinMax::Max emit MSL min()/max().
#[test]
fn test_msl_minmax_codegen() {
    for (op, expected_fn) in [(MinMaxKind::Min, "min("), (MinMaxKind::Max, "max(")] {
        let kernel = KernelDef::new(
            format!("minmax_{op:?}"),
            vec![f32_param("a"), f32_param("b")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Param(1)),
                n(
                    2,
                    IRNodeKind::MinMax {
                        op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );
        let scalar = nn_dsl::emit_scalar_fn(&kernel).expect("emit");
        assert!(
            scalar.contains(expected_fn),
            "{op:?} should emit {expected_fn}, scalar:\n{scalar}"
        );
    }
}

/// F16 kernel uses half types with float accumulator.
#[test]
fn test_msl_f16_accumulator_mode() {
    let kernel = KernelDef::new(
        "half_op",
        vec![
            Param::new("x", ScalarType::F16),
            Param::new("y", ScalarType::F16),
        ],
        ScalarType::F16,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let scalar = nn_dsl::emit_scalar_fn(&kernel).expect("emit");
    // Signature uses half
    assert!(
        scalar.contains("half half_op(half x, half y)"),
        "scalar:\n{scalar}"
    );
    // Intermediates promoted to float
    assert!(scalar.contains("float x_f = float(x)"), "scalar:\n{scalar}");
    assert!(scalar.contains("float y_f = float(y)"), "scalar:\n{scalar}");
    // Result demoted back
    assert!(scalar.contains("return half("), "scalar:\n{scalar}");
}

// ---------------------------------------------------------------------------
// C. Kani harness generation
// ---------------------------------------------------------------------------

/// Kani harness contains all required elements.
#[test]
fn test_kani_harness_structure() {
    let kernel = nn_dsl::test_kernels::square_kernel();
    let harness = nn_dsl::emit_kani_harness(&kernel).expect("emit");

    assert!(harness.contains("#[cfg(kani)]"), "harness:\n{harness}");
    assert!(harness.contains("#[kani::proof]"), "harness:\n{harness}");
    assert!(
        harness.contains("kani_verify_square"),
        "harness:\n{harness}"
    );
    assert!(harness.contains("kani::any()"), "harness:\n{harness}");
    assert!(harness.contains("kani::assume("), "harness:\n{harness}");
    assert!(harness.contains("is_finite()"), "harness:\n{harness}");
    assert!(harness.contains("square(x)"), "harness:\n{harness}");
}

/// Kani harness for BF16 inputs: symbolic f32 then converted.
#[test]
fn test_kani_harness_bf16_conversion() {
    let kernel = KernelDef::new(
        "bf16_fn",
        vec![Param::new("x", ScalarType::BF16)],
        ScalarType::BF16,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    let harness = nn_dsl::emit_kani_harness(&kernel).expect("emit");
    assert!(
        harness.contains("x_f32: f32 = kani::any()"),
        "harness:\n{harness}"
    );
    assert!(
        harness.contains("half::bf16::from_f32"),
        "harness:\n{harness}"
    );
}

/// Kani harness is valid Rust syntax.
#[test]
fn test_kani_harness_valid_syntax() {
    let kernel = nn_dsl::test_kernels::snake_kernel();
    let harness = nn_dsl::emit_kani_harness(&kernel).expect("emit");
    syn::parse_str::<syn::File>(&harness)
        .unwrap_or_else(|e| panic!("generated harness is not valid Rust syntax: {e}\n\n{harness}"));
}

// ---------------------------------------------------------------------------
// D. Pretty-print
// ---------------------------------------------------------------------------

/// Pretty-print output contains kernel name, params, return type, and all node labels.
#[test]
fn test_pretty_print_structure() {
    let kernel = KernelDef::new(
        "nn_fn",
        vec![f32_param("x"), f32_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(2, IRNodeKind::Literal(42.0)),
            n(
                3,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            n(
                4,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Exp,
                    input: NodeId::new(3),
                },
            ),
        ],
        NodeId::new(4),
    );
    let pp = nn_dsl::ir_pretty_print(&kernel);

    assert!(pp.contains("kernel nn_fn("), "pp:\n{pp}");
    assert!(pp.contains("x: f32"), "pp:\n{pp}");
    assert!(pp.contains("y: f32"), "pp:\n{pp}");
    assert!(pp.contains("-> f32"), "pp:\n{pp}");
    assert!(pp.contains("%0 = param(x)"), "pp:\n{pp}");
    assert!(pp.contains("%1 = param(y)"), "pp:\n{pp}");
    assert!(pp.contains("%2 = const(42.0)"), "pp:\n{pp}");
    assert!(pp.contains("%3 = add(%0, %1)"), "pp:\n{pp}");
    assert!(pp.contains("return %4"), "pp:\n{pp}");
}

// ---------------------------------------------------------------------------
// E. FTZ sensitivity detection
// ---------------------------------------------------------------------------
// NOTE: has_ftz_sensitive_op() is pub(crate), so direct testing is done
// in unit tests within the crate. Here we verify FTZ-sensitive ops produce
// correct MSL codegen (observable behavior from integration tests).

/// Kernels containing rsqrt should produce MSL with rsqrt intrinsic.
#[test]
fn test_ftz_sensitive_ops_in_msl_output() {
    let kernel = KernelDef::new(
        "ftz_rsqrt",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Rsqrt,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let msl = nn_dsl::emit_msl(&kernel).expect("MSL generation should succeed");
    assert!(
        msl.contains("rsqrt"),
        "MSL for rsqrt kernel should contain rsqrt intrinsic"
    );

    // Div should produce division operator in MSL
    let div_kernel = KernelDef::new(
        "ftz_div",
        vec![f32_param("x"), f32_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::BinOp {
                    op: BinOpKind::Div,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let msl = nn_dsl::emit_msl(&div_kernel).expect("MSL generation should succeed");
    assert!(
        msl.contains('/'),
        "MSL for div kernel should contain division operator"
    );
}

// ---------------------------------------------------------------------------
// F. Serde round-trip
// ---------------------------------------------------------------------------

/// KernelDef serializes to JSON and deserializes back with identical structure.
#[test]
fn test_kernel_def_serde_roundtrip() {
    let kernel = KernelDef::new(
        "serde_test",
        vec![f32_param("x"), Param::new("y", ScalarType::F16)],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(2, IRNodeKind::Literal(3.14)),
            n(
                3,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(2),
                },
            ),
            n(
                4,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(3),
                },
            ),
            n(
                5,
                IRNodeKind::Powi {
                    base: NodeId::new(4),
                    exp: -2,
                },
            ),
        ],
        NodeId::new(5),
    );

    let json = serde_json::to_string(&kernel).expect("serialize");
    let deserialized: KernelDef = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.name, kernel.name);
    assert_eq!(deserialized.params.len(), kernel.params.len());
    assert_eq!(deserialized.nodes.len(), kernel.nodes.len());
    assert_eq!(deserialized.output.index(), kernel.output.index());

    // Deserialized kernel must also pass validation
    deserialized
        .validate()
        .expect("deserialized kernel should validate");
}

// ---------------------------------------------------------------------------
// G. ScalarType lookup tables
// ---------------------------------------------------------------------------

/// ScalarType::from_type_name round-trips for all variants.
#[test]
fn test_scalar_type_name_roundtrip() {
    for st in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        let name = st.type_name();
        let roundtrip = ScalarType::from_type_name(name)
            .unwrap_or_else(|| panic!("from_type_name({name}) should succeed"));
        assert_eq!(roundtrip, st);
    }
}

/// ScalarType::from_type_name returns None for unknown names.
#[test]
fn test_scalar_type_from_unknown_name_returns_none() {
    assert!(ScalarType::from_type_name("f64").is_none());
    assert!(ScalarType::from_type_name("i32").is_none());
    assert!(ScalarType::from_type_name("").is_none());
}

/// UnaryFnKind::from_method_name round-trips for all variants.
#[test]
fn test_unary_fn_kind_method_name_roundtrip() {
    let all_ops = [
        UnaryFnKind::Sin,
        UnaryFnKind::Cos,
        UnaryFnKind::Sqrt,
        UnaryFnKind::Rsqrt,
        UnaryFnKind::Exp,
        UnaryFnKind::Abs,
        UnaryFnKind::Recip,
        UnaryFnKind::Tanh,
        UnaryFnKind::Log,
        UnaryFnKind::Floor,
        UnaryFnKind::Round,
        UnaryFnKind::Fract,
        UnaryFnKind::Neg,
    ];
    for op in all_ops {
        let name = op.method_name();
        let roundtrip = UnaryFnKind::from_method_name(name)
            .unwrap_or_else(|| panic!("from_method_name({name}) should succeed"));
        assert_eq!(roundtrip, op);
    }
}

/// UnaryFnKind::from_method_name returns None for unknown names.
#[test]
fn test_unary_fn_kind_unknown_method_returns_none() {
    assert!(UnaryFnKind::from_method_name("softmax").is_none());
    assert!(UnaryFnKind::from_method_name("").is_none());
}

/// ScalarType byte sizes are correct.
#[test]
fn test_scalar_type_byte_sizes() {
    assert_eq!(ScalarType::F32.byte_size(), 4);
    assert_eq!(ScalarType::F16.byte_size(), 2);
    assert_eq!(ScalarType::BF16.byte_size(), 2);
}

/// ScalarType MSL names map correctly.
#[test]
fn test_scalar_type_msl_names() {
    assert_eq!(ScalarType::F32.msl_str(), "float");
    assert_eq!(ScalarType::F16.msl_str(), "half");
    assert_eq!(ScalarType::BF16.msl_str(), "half"); // BF16 maps to half on Apple GPUs
}

/// ScalarType accumulator types: all accumulate in float.
#[test]
fn test_scalar_type_accumulator() {
    for st in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        assert_eq!(st.msl_accumulator_str(), "float", "{st:?} accumulator");
    }
}

// ---------------------------------------------------------------------------
// H. Verifiability classification
// ---------------------------------------------------------------------------

/// VerifiabilityClass method behavior.
#[test]
fn test_verifiability_class_allows_compilation() {
    use nn_dsl::VerifiabilityClass;

    assert!(VerifiabilityClass::Verifiable.allows_compilation());
    assert!(VerifiabilityClass::ShapeOnly.allows_compilation());
    assert!(VerifiabilityClass::Passthrough.allows_compilation());
    assert!(VerifiabilityClass::UnverifiableSafe.allows_compilation());
    assert!(!VerifiabilityClass::UnverifiableLearned.allows_compilation());

    let bounded = VerifiabilityClass::VerifiableBounded { max_dim: 512 };
    assert!(bounded.allows_compilation());
    assert!(!bounded.needs_decomposition(256));
    assert!(bounded.needs_decomposition(1024));
}

/// classify_callee_name covers common operations.
#[test]
fn test_classify_callee_name_coverage() {
    use nn_dsl::{classify_callee_name, VerifiabilityClass};

    // Verifiable activations
    for name in [
        "relu", "gelu", "sigmoid", "tanh", "silu", "exp", "sin", "cos",
    ] {
        assert!(
            matches!(classify_callee_name(name), VerifiabilityClass::Verifiable),
            "{name} should be Verifiable"
        );
    }

    // Verifiable binary ops
    for name in ["add", "sub", "mul", "div"] {
        assert!(
            matches!(classify_callee_name(name), VerifiabilityClass::Verifiable),
            "{name} should be Verifiable"
        );
    }

    // Shape-only
    for name in ["reshape", "transpose", "narrow", "squeeze"] {
        assert!(
            matches!(classify_callee_name(name), VerifiabilityClass::ShapeOnly),
            "{name} should be ShapeOnly"
        );
    }

    // Passthrough
    assert!(matches!(
        classify_callee_name("dropout"),
        VerifiabilityClass::Passthrough
    ));

    // Unknown -> UnverifiableLearned
    assert!(matches!(
        classify_callee_name("some_custom_op"),
        VerifiabilityClass::UnverifiableLearned
    ));
}

// ---------------------------------------------------------------------------
// I. Lowering: Rust source -> KernelDef
// ---------------------------------------------------------------------------

/// Lowering a simple identity function produces correct IR.
#[test]
fn test_lower_identity_function() {
    let kernel = nn_dsl::test_kernels::parse_kernel("fn id(x: f32) -> f32 { x }");
    assert_eq!(kernel.name, "id");
    assert_eq!(kernel.params.len(), 1);
    assert_eq!(kernel.params[0].name, "x");
    assert_eq!(kernel.params[0].ty, ScalarType::F32);
    assert_eq!(kernel.return_type, ScalarType::F32);
    kernel.validate().expect("identity kernel should validate");
}

/// Lowering a multi-step kernel preserves structure.
#[test]
fn test_lower_multi_step_kernel() {
    let kernel = nn_dsl::test_kernels::parse_kernel(
        "fn multi(x: f32, y: f32) -> f32 { (x + y).sin().abs() }",
    );
    assert_eq!(kernel.params.len(), 2);
    kernel
        .validate()
        .expect("multi-step kernel should validate");

    // The output node should be the result of abs(sin(x+y))
    let output_node = &kernel.nodes[kernel.output.index()];
    assert!(
        matches!(
            output_node.kind,
            IRNodeKind::UnaryFn {
                op: UnaryFnKind::Abs,
                ..
            }
        ),
        "output should be abs(), got: {:?}",
        output_node.kind
    );
}

/// Lowered kernel produces valid MSL and Kani harness.
#[test]
fn test_lower_then_codegen_roundtrip() {
    let kernel = nn_dsl::test_kernels::parse_kernel(
        "fn roundtrip(x: f32, alpha: f32) -> f32 { x + (1.0 / alpha) * x.exp() }",
    );

    // MSL generation
    let msl = nn_dsl::emit_msl(&kernel).expect("MSL should generate");
    assert!(msl.contains("[[kernel]]"), "MSL missing kernel entry point");
    assert!(msl.contains("metal::precise::exp"), "MSL missing exp call");

    // Kani harness generation
    let harness = nn_dsl::emit_kani_harness(&kernel).expect("Kani should generate");
    assert!(
        harness.contains("#[kani::proof]"),
        "Kani missing proof attribute"
    );
    assert!(
        harness.contains("roundtrip(x, alpha)"),
        "Kani missing function call"
    );
}

// ---------------------------------------------------------------------------
// J. Validation error specificity
// ---------------------------------------------------------------------------

/// Invalid parameter reference reports the correct indices.
#[test]
fn test_invalid_param_ref_error() {
    let kernel = KernelDef::new(
        "bad_param",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(5)), // only 1 param, index 5 is out of bounds
        ],
        NodeId::new(0),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(err, IRError::InvalidParamRef(5, 1)),
        "expected InvalidParamRef(5, 1), got: {err}"
    );
}

/// Out-of-bounds output reference is caught.
#[test]
fn test_output_out_of_bounds_error() {
    let kernel = KernelDef::new(
        "bad_output",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(99), // out of bounds
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(err, IRError::InvalidNodeRef(_)),
        "expected InvalidNodeRef, got: {err}"
    );
}
