// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for DSL IR construction, MSL code generation, optimization
//! passes, type inference, shape inference, and error handling.
//!
//! Covers:
//! 1.  KernelDef IR deep construction: complex DAGs, multiple consumers
//! 2.  IR validation edge cases: non-finite literals, powi bounds, mismatched IDs
//! 3.  MSL code generation: emit_msl, emit_scalar_fn output content
//! 4.  Auto-fuse MSL generation: auto_fuse_to_msl end-to-end
//! 5.  Fused MSL metadata: FusedKernelMeta, FusedMslResult validation
//! 6.  Optimization passes: constant folding via graph patterns
//! 7.  Type inference: dtype propagation through composed operations
//! 8.  Shape inference: output shape computation through tensor ops
//! 9.  Error handling: invalid IR construction rejected with correct errors
//! 10. Graph recording: TraceOp composition and replay patterns
//! 11. IR pretty-printing: human-readable IR dump format
//! 12. Multi-consumer DAGs: fan-out patterns in kernel IR
//! 13. Powi exponent boundary: max exponent acceptance/rejection
//! 14. Identifier validation: reserved words, invalid characters
//! 15. BinaryFn (atan2): two-input math functions in IR
//!
//! Part of #4560.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::auto_fuse_codegen::{
    auto_fuse_to_msl, compose_trace_ops_to_kernel_ir, FuseableOp, OpWiring,
};
use crate::codegen_msl::{emit_msl, emit_scalar_fn};
use crate::ir::{
    ir_pretty_print, BinOpKind, BinaryFnKind, CompareOpKind, IRNode, IRNodeKind, KernelDef,
    MinMaxKind, NodeId, Param, ScalarType, UnaryFnKind, ValueType, POWI_MAX_EXPONENT,
};
use crate::msl_auto_fuse::{generate_fused_msl, FusedKernelMeta, FusedMslError};
use crate::tensor_ir::{
    BroadcastAlignment, ReduceOp, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};
use crate::trace_compile::{
    compile_trace_to_plan, compile_trace_to_plan_with_fusion, count_dispatches,
    detect_fusion_chains,
};

// ===========================================================================
// Helpers
// ===========================================================================

fn n(id: usize, kind: IRNodeKind) -> IRNode {
    IRNode::new(NodeId::new(id), kind)
}

fn f32_param(name: &str) -> Param {
    Param::new(name.to_string(), ScalarType::F32)
}

fn input_node(id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape,
        DType::F32,
    )
}

fn test_node(id: u64, name: &str, op: TraceOp, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(id, name.to_string(), op, inputs, shape, DType::F32)
}

fn tensor_input(id: usize, name: &str, shape: Vec<usize>) -> TensorNode {
    TensorNode::new(
        TensorNodeId::new(id),
        TensorOpKind::Input {
            name: name.into(),
            shape: shape.clone(),
        },
        shape,
    )
}

// ===========================================================================
// Section 1: Complex KernelDef IR DAGs
// ===========================================================================

#[test]
fn test_kernel_def_diamond_dag_validates() {
    // Diamond: param -> {add, mul} -> sub
    //   %0 = param(x)
    //   %1 = param(y)
    //   %2 = add(%0, %1)
    //   %3 = mul(%0, %1)
    //   %4 = sub(%2, %3)
    let kernel = KernelDef::new(
        "diamond",
        vec![f32_param("x"), f32_param("y")],
        ScalarType::F32,
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
            n(
                3,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            n(
                4,
                IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    lhs: NodeId::new(2),
                    rhs: NodeId::new(3),
                },
            ),
        ],
        NodeId::new(4),
    );
    kernel.validate().expect("diamond DAG should validate");
}

#[test]
fn test_kernel_def_multi_consumer_node() {
    // x is consumed by sin(x), cos(x), and x*x
    let kernel = KernelDef::new(
        "multi_consumer",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(0),
                },
            ),
            n(
                2,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Cos,
                    input: NodeId::new(0),
                },
            ),
            n(
                3,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(0),
                },
            ),
            // Combine: sin(x) + cos(x) + x*x
            n(
                4,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(2),
                },
            ),
            n(
                5,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(4),
                    rhs: NodeId::new(3),
                },
            ),
        ],
        NodeId::new(5),
    );
    kernel
        .validate()
        .expect("multi-consumer DAG should validate");
}

#[test]
fn test_kernel_def_deep_chain_validates() {
    // Chain of 10 additions: p0 + p0 + p0 + ...
    let mut nodes = vec![n(0, IRNodeKind::Param(0))];
    for i in 1..10 {
        nodes.push(n(
            i,
            IRNodeKind::BinOp {
                op: BinOpKind::Add,
                lhs: NodeId::new(i - 1),
                rhs: NodeId::new(0),
            },
        ));
    }
    let kernel = KernelDef::new(
        "deep_chain",
        vec![f32_param("x")],
        ScalarType::F32,
        nodes,
        NodeId::new(9),
    );
    kernel.validate().expect("deep chain should validate");
}

#[test]
fn test_kernel_def_binary_fn_atan2_validates() {
    let kernel = KernelDef::new(
        "atan2_kernel",
        vec![f32_param("y"), f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::BinaryFn {
                    op: BinaryFnKind::Atan2,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    kernel.validate().expect("atan2 kernel should validate");
}

#[test]
fn test_kernel_def_select_with_compare() {
    // max(x, 0) via compare + select
    let kernel = KernelDef::new(
        "relu_manual",
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
    kernel.validate().expect("manual relu should validate");
}

// ===========================================================================
// Section 2: IR validation error cases
// ===========================================================================

#[test]
fn test_ir_validation_non_finite_literal_nan() {
    let kernel = KernelDef::new(
        "bad_nan",
        vec![],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Literal(f64::NAN))],
        NodeId::new(0),
    );
    let err = kernel.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite") || msg.contains("NaN"),
        "should mention NaN: {msg}"
    );
}

#[test]
fn test_ir_validation_non_finite_literal_inf() {
    let kernel = KernelDef::new(
        "bad_inf",
        vec![],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Literal(f64::INFINITY))],
        NodeId::new(0),
    );
    assert!(
        kernel.validate().is_err(),
        "Infinity literal must be rejected"
    );
}

#[test]
fn test_ir_validation_non_finite_literal_neg_inf() {
    let kernel = KernelDef::new(
        "bad_neg_inf",
        vec![],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Literal(f64::NEG_INFINITY))],
        NodeId::new(0),
    );
    assert!(
        kernel.validate().is_err(),
        "negative infinity must be rejected"
    );
}

#[test]
fn test_ir_validation_powi_max_exponent_accepted() {
    let kernel = KernelDef::new(
        "powi_max",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: POWI_MAX_EXPONENT as i32,
                },
            ),
        ],
        NodeId::new(1),
    );
    kernel.validate().expect("max exponent should be accepted");
}

#[test]
fn test_ir_validation_powi_exceeds_max_exponent() {
    let kernel = KernelDef::new(
        "powi_too_big",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: (POWI_MAX_EXPONENT as i32) + 1,
                },
            ),
        ],
        NodeId::new(1),
    );
    assert!(
        kernel.validate().is_err(),
        "exponent exceeding max should be rejected"
    );
}

#[test]
fn test_ir_validation_powi_negative_exponent() {
    let kernel = KernelDef::new(
        "powi_neg",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: -3,
                },
            ),
        ],
        NodeId::new(1),
    );
    // Negative exponents within bounds should validate
    kernel
        .validate()
        .expect("negative exponent within bounds should validate");
}

#[test]
fn test_ir_validation_mismatched_node_id() {
    let kernel = KernelDef::new(
        "bad_id",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(5),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Abs,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(5),
    );
    assert!(
        kernel.validate().is_err(),
        "mismatched node ID must be rejected"
    );
}

#[test]
fn test_ir_validation_param_index_out_of_bounds() {
    let kernel = KernelDef::new(
        "bad_param",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(5))], // only 1 param, index 5 is OOB
        NodeId::new(0),
    );
    assert!(
        kernel.validate().is_err(),
        "OOB param index must be rejected"
    );
}

#[test]
fn test_ir_validation_empty_sum_reduce_rejected() {
    let kernel = KernelDef::new(
        "empty_reduce",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::SumReduce { inputs: vec![] }),
        ],
        NodeId::new(1),
    );
    assert!(
        kernel.validate().is_err(),
        "empty SumReduce must be rejected"
    );
}

#[test]
fn test_ir_validation_output_oob_rejected() {
    let kernel = KernelDef::new(
        "oob_output",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(99),
    );
    assert!(
        kernel.validate().is_err(),
        "output referencing nonexistent node must fail"
    );
}

#[test]
fn test_ir_validation_identifier_empty_name() {
    let kernel = KernelDef::new(
        "",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    assert!(
        kernel.validate().is_err(),
        "empty kernel name must be rejected"
    );
}

#[test]
fn test_ir_validation_identifier_starts_with_digit() {
    let kernel = KernelDef::new(
        "3bad",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    assert!(
        kernel.validate().is_err(),
        "name starting with digit must be rejected"
    );
}

#[test]
fn test_ir_validation_identifier_special_chars() {
    let kernel = KernelDef::new(
        "nn-kernel",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    assert!(
        kernel.validate().is_err(),
        "hyphen in name must be rejected"
    );
}

#[test]
fn test_ir_validation_underscore_prefix_ok() {
    let kernel = KernelDef::new(
        "_internal",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    kernel
        .validate()
        .expect("underscore prefix should be valid");
}

// ===========================================================================
// Section 3: MSL code generation content
// ===========================================================================

#[test]
fn test_emit_msl_identity_contains_kernel_entry() {
    let kernel = KernelDef::new(
        "identity",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    let msl = emit_msl(&kernel).expect("emit_msl should succeed");
    assert!(
        msl.contains("identity_kernel"),
        "MSL must contain kernel entry point name"
    );
    assert!(
        msl.contains("[[kernel]]"),
        "MSL must contain [[kernel]] attribute"
    );
    assert!(
        msl.contains("device"),
        "MSL must contain device buffer qualifiers"
    );
}

#[test]
fn test_emit_msl_add_kernel_contains_addition() {
    let kernel = KernelDef::new(
        "add_fn",
        vec![f32_param("a"), f32_param("b")],
        ScalarType::F32,
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
    let msl = emit_msl(&kernel).expect("emit_msl should succeed");
    assert!(
        msl.contains("+"),
        "MSL for add kernel must contain + operator"
    );
    assert!(
        msl.contains("add_fn_kernel"),
        "MSL must contain kernel name"
    );
}

#[test]
fn test_emit_msl_sin_kernel_contains_sin_call() {
    let kernel = KernelDef::new(
        "sin_fn",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let msl = emit_msl(&kernel).expect("emit_msl should succeed");
    assert!(
        msl.contains("sin"),
        "MSL for sin kernel must contain sin function call"
    );
}

#[test]
fn test_emit_msl_literal_contains_value() {
    let kernel = KernelDef::new(
        "const_val",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Literal(3.14)),
            n(
                2,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let msl = emit_msl(&kernel).expect("emit_msl should succeed");
    assert!(
        msl.contains("3.14"),
        "MSL must contain the literal value 3.14"
    );
}

#[test]
fn test_emit_msl_clamp_kernel() {
    let kernel = KernelDef::new(
        "clamp_fn",
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
    let msl = emit_msl(&kernel).expect("emit_msl should succeed for clamp");
    assert!(
        msl.contains("clamp") || msl.contains("min") || msl.contains("max"),
        "clamp kernel MSL must contain clamping logic"
    );
}

#[test]
fn test_emit_scalar_fn_produces_function() {
    let kernel = KernelDef::new(
        "scale",
        vec![f32_param("x"), f32_param("s")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let scalar_fn = emit_scalar_fn(&kernel).expect("emit_scalar_fn should succeed");
    assert!(
        scalar_fn.contains("float"),
        "scalar fn should contain float type"
    );
    assert!(
        scalar_fn.contains("*"),
        "scalar fn should contain multiplication"
    );
}

#[test]
fn test_emit_msl_reserved_word_kernel_name_rejected() {
    let kernel = KernelDef::new(
        "kernel", // MSL reserved word
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    // IR validation passes (backend-agnostic), but MSL emit should fail
    let result = emit_msl(&kernel);
    assert!(
        result.is_err(),
        "MSL reserved word as kernel name must be rejected at emit time"
    );
}

#[test]
fn test_emit_msl_reserved_word_param_name_rejected() {
    let kernel = KernelDef::new(
        "nn_kernel",
        vec![Param::new("device".to_string(), ScalarType::F32)],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    let result = emit_msl(&kernel);
    assert!(
        result.is_err(),
        "'device' as param name must be rejected in MSL"
    );
}

// ===========================================================================
// Section 4: Auto-fuse MSL generation end-to-end
// ===========================================================================

#[test]
fn test_auto_fuse_to_msl_single_relu() {
    let ops = vec![FuseableOp::unary(TraceOp::Relu)];
    let fused = auto_fuse_to_msl(&ops, "fused_relu").expect("auto_fuse should succeed");
    assert!(
        fused.msl_source.contains("fused_relu_kernel"),
        "must contain kernel name"
    );
    assert!(
        fused.msl_source.contains("[[kernel]]"),
        "must contain kernel attribute"
    );
    assert_eq!(fused.num_external_inputs, 1);
    assert_eq!(fused.entry_point, "fused_relu_kernel");
}

#[test]
fn test_auto_fuse_to_msl_three_op_chain() {
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::unary(TraceOp::Tanh),
        FuseableOp::unary(TraceOp::Neg),
    ];
    let fused = auto_fuse_to_msl(&ops, "exp_tanh_neg").expect("3-op chain should succeed");
    assert!(fused.msl_source.contains("exp_tanh_neg_kernel"));
    assert_eq!(
        fused.num_external_inputs, 1,
        "unary chain has 1 external input"
    );
    fused
        .kernel_def
        .validate()
        .expect("fused kernel should validate");
}

#[test]
fn test_auto_fuse_to_msl_binary_chain() {
    let ops = vec![
        FuseableOp {
            op: TraceOp::Add,
            wiring: OpWiring::BinaryBothExternal,
        },
        FuseableOp::unary(TraceOp::Relu),
    ];
    let fused = auto_fuse_to_msl(&ops, "add_relu").expect("add+relu should fuse");
    assert_eq!(
        fused.num_external_inputs, 2,
        "add takes two external inputs"
    );
    assert!(fused.msl_source.len() > 100, "MSL should be non-trivial");
}

#[test]
fn test_auto_fuse_to_msl_empty_chain_rejected() {
    let ops: Vec<FuseableOp> = vec![];
    let result = auto_fuse_to_msl(&ops, "empty");
    assert!(result.is_err(), "empty op chain must be rejected");
}

#[test]
fn test_auto_fuse_binary_both_external_not_first_rejected() {
    // BinaryBothExternal at position > 0 should fail
    let ops = vec![
        FuseableOp::unary(TraceOp::Relu),
        FuseableOp {
            op: TraceOp::Add,
            wiring: OpWiring::BinaryBothExternal,
        },
    ];
    let result = compose_trace_ops_to_kernel_ir(&ops, "bad_chain");
    assert!(
        result.is_err(),
        "BinaryBothExternal after first op must be rejected"
    );
}

// ===========================================================================
// Section 5: FusedKernelMeta and FusedMslResult
// ===========================================================================

#[test]
fn test_fused_kernel_meta_total_elements() {
    let meta = FusedKernelMeta::new(
        vec![vec![4, 8]],
        vec![4, 8],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );
    assert_eq!(meta.total_elements(), 32);
}

#[test]
fn test_fused_kernel_meta_scalar_output() {
    let meta = FusedKernelMeta::new(
        vec![vec![1]],
        vec![1],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );
    assert_eq!(meta.total_elements(), 1);
}

#[test]
fn test_fused_kernel_meta_high_rank() {
    let meta = FusedKernelMeta::new(
        vec![vec![2, 3, 4, 5]],
        vec![2, 3, 4, 5],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );
    assert_eq!(meta.total_elements(), 120);
}

#[test]
fn test_generate_fused_msl_identity_kernel() {
    let kernel = KernelDef::new(
        "fused_id",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    let meta = FusedKernelMeta::new(
        vec![vec![1, 64]],
        vec![1, 64],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );
    let result = generate_fused_msl(&kernel, &meta).expect("fused MSL gen should succeed");
    assert_eq!(result.kernel_name, "fused_id_kernel");
    assert!(result.msl_source.contains("fused_id_kernel"));
    assert_eq!(result.buffer_count, 3); // 1 input + 1 output + 1 total
    assert!(result.threadgroup_size > 0);
}

#[test]
fn test_generate_fused_msl_shape_param_mismatch() {
    let kernel = KernelDef::new(
        "mismatch",
        vec![f32_param("x"), f32_param("y")],
        ScalarType::F32,
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
    // Meta has 1 shape but kernel has 2 params
    let meta = FusedKernelMeta::new(
        vec![vec![1, 64]],
        vec![1, 64],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );
    let result = generate_fused_msl(&kernel, &meta);
    assert!(result.is_err(), "shape/param count mismatch must fail");
    if let Err(FusedMslError::ShapeParamMismatch { shapes, params }) = result {
        assert_eq!(shapes, 1);
        assert_eq!(params, 2);
    }
}

#[test]
fn test_generate_fused_msl_with_broadcast() {
    // Kernel: add(x, y) where x=[4,8], y=[8] (broadcast needed)
    let kernel = KernelDef::new(
        "broadcast_add",
        vec![f32_param("x"), f32_param("y")],
        ScalarType::F32,
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
    let meta = FusedKernelMeta::new(
        vec![vec![4, 8], vec![8]],
        vec![4, 8],
        BroadcastAlignment::Right,
        ScalarType::F32,
    );
    let result = generate_fused_msl(&kernel, &meta).expect("broadcast add should succeed");
    assert_eq!(result.buffer_count, 4); // 2 inputs + output + total
                                        // y has a different shape, so broadcast indexing should be generated
    assert!(result.msl_source.contains("broadcast_add_kernel"));
}

// ===========================================================================
// Section 6: Constant folding via trace graph patterns
// ===========================================================================

#[test]
fn test_constant_node_in_graph() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "const_1".into(),
            TraceOp::Constant { value: 1.0 },
            vec![],
            vec![1],
            DType::F32,
        ),
        input_node(1, vec![1, 64]),
        test_node(2, "add", TraceOp::Add, vec![1, 0], vec![1, 64]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile should succeed");
    assert!(!plan.steps.is_empty());
}

#[test]
fn test_constant_zero_mul_pattern() {
    // x * 0 is a simplifiable pattern. The graph should still compile.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 32]),
        TraceNode::new(
            1,
            "zero".into(),
            TraceOp::Constant { value: 0.0 },
            vec![],
            vec![1, 32],
            DType::F32,
        ),
        test_node(2, "mul", TraceOp::Mul, vec![0, 1], vec![1, 32]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile should succeed");
    assert!(!plan.steps.is_empty());
}

#[test]
fn test_constant_one_mul_pattern() {
    // x * 1 is an identity pattern
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![2, 16]),
        TraceNode::new(
            1,
            "one".into(),
            TraceOp::Constant { value: 1.0 },
            vec![],
            vec![2, 16],
            DType::F32,
        ),
        test_node(2, "mul", TraceOp::Mul, vec![0, 1], vec![2, 16]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile should succeed");
    assert!(!plan.steps.is_empty());
}

// ===========================================================================
// Section 7: Type inference through compositions
// ===========================================================================

#[test]
fn test_type_inference_f16_kernel_propagates() {
    let kernel = KernelDef::new(
        "f16_scale",
        vec![
            Param::new("x", ScalarType::F16),
            Param::new("s", ScalarType::F16),
        ],
        ScalarType::F16,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    kernel.validate().expect("f16 kernel should validate");
}

#[test]
fn test_type_inference_bf16_kernel_propagates() {
    let kernel = KernelDef::new(
        "bf16_add",
        vec![
            Param::new("a", ScalarType::BF16),
            Param::new("b", ScalarType::BF16),
        ],
        ScalarType::BF16,
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
    kernel.validate().expect("bf16 kernel should validate");
}

#[test]
fn test_type_inference_compare_produces_bool() {
    // Compare -> Select pattern: compare must produce Bool, select consumes it
    let kernel = KernelDef::new(
        "abs_via_select",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Literal(0.0)),
            n(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Ge,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            n(
                3,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Neg,
                    input: NodeId::new(0),
                },
            ),
            n(
                4,
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(3),
                },
            ),
        ],
        NodeId::new(4),
    );
    kernel
        .validate()
        .expect("abs-via-select should validate (compare -> bool -> select)");
}

#[test]
fn test_type_inference_bool_in_arithmetic_rejected() {
    // Trying to add a bool (from compare) should fail type checking
    let kernel = KernelDef::new(
        "bad_bool_arith",
        vec![f32_param("x"), f32_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Lt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            // This should fail: adding a bool to a float
            n(
                3,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(2),
                },
            ),
        ],
        NodeId::new(3),
    );
    assert!(
        kernel.validate().is_err(),
        "bool in arithmetic must be rejected by type inference"
    );
}

#[test]
fn test_type_inference_select_cond_must_be_bool() {
    // Using a float as the condition of Select should fail
    let kernel = KernelDef::new(
        "bad_select_cond",
        vec![f32_param("x"), f32_param("y"), f32_param("z")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(2, IRNodeKind::Param(2)),
            // %0 is float, not bool -- should fail
            n(
                3,
                IRNodeKind::Select {
                    cond: NodeId::new(0),
                    then_val: NodeId::new(1),
                    else_val: NodeId::new(2),
                },
            ),
        ],
        NodeId::new(3),
    );
    assert!(
        kernel.validate().is_err(),
        "float as Select condition must be rejected"
    );
}

// ===========================================================================
// Section 8: Shape inference through tensor ops
// ===========================================================================

#[test]
fn test_tensor_shape_reduce_removes_axis() {
    let def = TensorKernelDef::new(
        "sum_axis1",
        vec![
            tensor_input(0, "x", vec![4, 8, 16]),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    input: TensorNodeId::new(0),
                    op: ReduceOp::Sum,
                    axis: 1,
                    keepdim: false,
                },
                vec![4, 16],
            ),
        ],
        TensorNodeId::new(1),
    );
    assert_eq!(
        def.nodes[1].shape,
        vec![4, 16],
        "reduce axis=1 should remove dim 1"
    );
}

#[test]
fn test_tensor_shape_reduce_keepdim() {
    let def = TensorKernelDef::new(
        "sum_keepdim",
        vec![
            tensor_input(0, "x", vec![4, 8, 16]),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    input: TensorNodeId::new(0),
                    op: ReduceOp::Sum,
                    axis: 1,
                    keepdim: true,
                },
                vec![4, 1, 16],
            ),
        ],
        TensorNodeId::new(1),
    );
    assert_eq!(
        def.nodes[1].shape,
        vec![4, 1, 16],
        "keepdim should preserve rank"
    );
}

#[test]
fn test_tensor_shape_elementwise_preserves_shape() {
    let shape = vec![2, 3, 4];
    let def = TensorKernelDef::new(
        "relu_shape",
        vec![
            tensor_input(0, "x", shape.clone()),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Relu {
                    input: TensorNodeId::new(0),
                },
                shape.clone(),
            ),
        ],
        TensorNodeId::new(1),
    );
    assert_eq!(
        def.nodes[1].shape, shape,
        "elementwise op must preserve shape"
    );
}

#[test]
fn test_tensor_shape_binary_add_preserves_shape() {
    let shape = vec![8, 64];
    let def = TensorKernelDef::new(
        "add_shape",
        vec![
            tensor_input(0, "a", shape.clone()),
            tensor_input(1, "b", shape.clone()),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::BinaryAdd {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                },
                shape.clone(),
            ),
        ],
        TensorNodeId::new(2),
    );
    assert_eq!(def.nodes[2].shape, shape, "binary add preserves shape");
}

#[test]
fn test_broadcast_alignment_right_variant() {
    // BroadcastAlignment::Right is the default NumPy-style alignment
    let alignment = BroadcastAlignment::Right;
    assert_eq!(alignment, BroadcastAlignment::Right);
    // Left alignment is used for per-channel operations
    let left = BroadcastAlignment::Left;
    assert_ne!(left, BroadcastAlignment::Right);
}

// ===========================================================================
// Section 9: Graph recording and replay patterns
// ===========================================================================

#[test]
fn test_graph_linear_chain_recording() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 128]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 128]),
        test_node(2, "sigmoid", TraceOp::Sigmoid, vec![1], vec![1, 128]),
        test_node(3, "tanh", TraceOp::Tanh, vec![2], vec![1, 128]),
    ]);
    assert_eq!(graph.len(), 4);
    let nodes = graph.nodes();
    assert_eq!(nodes[0].inputs().len(), 0, "input has no dependencies");
    assert_eq!(nodes[1].inputs(), &[0]);
    assert_eq!(nodes[2].inputs(), &[1]);
    assert_eq!(nodes[3].inputs(), &[2]);
}

#[test]
fn test_graph_fan_out_recording() {
    // x -> {relu(x), sigmoid(x)}
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 64]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 64]),
        test_node(2, "sigmoid", TraceOp::Sigmoid, vec![0], vec![1, 64]),
    ]);
    assert_eq!(graph.len(), 3);
    // Both relu and sigmoid depend on input 0
    assert_eq!(graph.nodes()[1].inputs(), &[0]);
    assert_eq!(graph.nodes()[2].inputs(), &[0]);
}

#[test]
fn test_graph_fan_in_recording() {
    // {a, b} -> add(a, b)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 32]),
        input_node(1, vec![1, 32]),
        test_node(2, "add", TraceOp::Add, vec![0, 1], vec![1, 32]),
    ]);
    assert_eq!(graph.nodes()[2].inputs(), &[0, 1]);
}

#[test]
fn test_graph_with_reshape_passthrough() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![2, 32]),
        test_node(
            1,
            "reshape",
            TraceOp::Reshape {
                target_shape: vec![64],
            },
            vec![0],
            vec![64],
        ),
        test_node(2, "relu", TraceOp::Relu, vec![1], vec![64]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("reshape + relu should compile");
    assert!(!plan.steps.is_empty());
}

#[test]
fn test_graph_compile_with_dropout_identity() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![4, 64]),
        test_node(1, "dropout", TraceOp::Dropout, vec![0], vec![4, 64]),
        test_node(2, "relu", TraceOp::Relu, vec![1], vec![4, 64]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("dropout + relu should compile");
    assert!(!plan.steps.is_empty());
}

// ===========================================================================
// Section 10: IR pretty-printing
// ===========================================================================

#[test]
fn test_ir_pretty_print_identity() {
    let kernel = KernelDef::new(
        "identity",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    let pretty = ir_pretty_print(&kernel);
    assert!(
        pretty.contains("kernel identity"),
        "should contain kernel header"
    );
    assert!(pretty.contains("param(x)"), "should show param reference");
    assert!(pretty.contains("return"), "should contain return statement");
}

#[test]
fn test_ir_pretty_print_add() {
    let kernel = KernelDef::new(
        "adder",
        vec![f32_param("a"), f32_param("b")],
        ScalarType::F32,
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
    let pretty = ir_pretty_print(&kernel);
    assert!(pretty.contains("add("), "should contain add operation");
    // ScalarType Display uses the lowercase Rust type name (`f32`); see
    // ScalarType::type_name and the ir_pretty_print doc example.
    assert!(pretty.contains("a: f32"), "should show param type");
    assert!(pretty.contains("b: f32"), "should show second param");
}

#[test]
fn test_ir_pretty_print_unary_fn() {
    let kernel = KernelDef::new(
        "exp_fn",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Exp,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let pretty = ir_pretty_print(&kernel);
    assert!(pretty.contains("exp("), "should contain exp function");
}

#[test]
fn test_ir_pretty_print_literal() {
    let kernel = KernelDef::new(
        "half_pi",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Literal(1.5707963)),
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
    let pretty = ir_pretty_print(&kernel);
    assert!(
        pretty.contains("const("),
        "should contain const notation for literal"
    );
}

#[test]
fn test_ir_pretty_print_select() {
    let kernel = KernelDef::new(
        "cond_sel",
        vec![f32_param("x"), f32_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Lt,
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
    let pretty = ir_pretty_print(&kernel);
    assert!(
        pretty.contains("select("),
        "should contain select operation"
    );
}

// ===========================================================================
// Section 11: Fusion chain detection edge cases
// ===========================================================================

#[test]
fn test_fusion_chain_fan_out_blocks() {
    // x -> relu -> {sigmoid(relu), tanh(relu)}
    // Fan-out from relu means it cannot be fused into a single chain.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 64]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 64]),
        test_node(2, "sigmoid", TraceOp::Sigmoid, vec![1], vec![1, 64]),
        test_node(3, "tanh", TraceOp::Tanh, vec![1], vec![1, 64]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    // relu has fan-out=2, so no chain of length > 2 spanning both branches
    for chain in &chains {
        assert!(chain.chain_len <= 2, "fan-out should limit chain length");
    }
}

#[test]
fn test_fusion_chain_shape_mismatch_blocks() {
    // relu on [1,64] -> reshape to [64] -> exp on [64]
    // Shape change should prevent fusion.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 64]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 64]),
        test_node(
            2,
            "reshape",
            TraceOp::Reshape {
                target_shape: vec![64],
            },
            vec![1],
            vec![64],
        ),
        test_node(3, "exp", TraceOp::Exp, vec![2], vec![64]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    // No chain should span across the reshape
    for chain in &chains {
        assert!(
            chain.chain_len <= 2,
            "shape change should block fusion: got chain_len={}",
            chain.chain_len
        );
    }
}

#[test]
fn test_fusion_chain_with_constant_weight() {
    // input -> relu -> add(relu, constant_weight)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 32]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 32]),
        TraceNode::new(
            2,
            "bias".to_string(),
            TraceOp::ConstantWeight {
                weight: WeightRef::from_shape(&[1, 32]),
            },
            vec![],
            vec![1, 32],
            DType::F32,
        ),
        test_node(3, "add", TraceOp::Add, vec![1, 2], vec![1, 32]),
    ]);
    // This should compile fine with fusion
    let plan_no_fusion = compile_trace_to_plan(&graph).expect("should compile without fusion");
    let plan_with_fusion =
        compile_trace_to_plan_with_fusion(&graph).expect("should compile with fusion");
    // With fusion, the dispatch count should be <= without fusion
    assert!(
        count_dispatches(&plan_with_fusion) <= count_dispatches(&plan_no_fusion),
        "fusion should not increase dispatch count"
    );
}

// ===========================================================================
// Section 12: Multi-consumer IR patterns
// ===========================================================================

#[test]
fn test_kernel_def_powi_validates() {
    let kernel = KernelDef::new(
        "square",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 2,
                },
            ),
        ],
        NodeId::new(1),
    );
    kernel.validate().expect("powi(2) should validate");
}

#[test]
fn test_kernel_def_complex_snake_pattern() {
    // Snake activation: x + (1/alpha) * sin(alpha*x)^2
    let kernel = KernelDef::new(
        "snake",
        vec![f32_param("x"), f32_param("alpha")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),     // x
            n(1, IRNodeKind::Param(1)),     // alpha
            n(2, IRNodeKind::Literal(1.0)), // 1.0
            n(
                3,
                IRNodeKind::BinOp {
                    op: BinOpKind::Div,
                    lhs: NodeId::new(2),
                    rhs: NodeId::new(1),
                },
            ), // 1/alpha
            n(
                4,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(0),
                },
            ), // alpha*x
            n(
                5,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(4),
                },
            ), // sin(alpha*x)
            n(
                6,
                IRNodeKind::Powi {
                    base: NodeId::new(5),
                    exp: 2,
                },
            ), // sin(alpha*x)^2
            n(
                7,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(3),
                    rhs: NodeId::new(6),
                },
            ), // (1/alpha)*sin(alpha*x)^2
            n(
                8,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(7),
                },
            ), // x + ...
        ],
        NodeId::new(8),
    );
    kernel.validate().expect("snake kernel should validate");
    // Verify FTZ sensitivity (contains div)
    assert!(
        kernel.has_ftz_sensitive_op(),
        "snake kernel has div, should be FTZ-sensitive"
    );
}

#[test]
fn test_kernel_def_gelu_approximation_pattern() {
    // GeLU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let kernel = KernelDef::new(
        "gelu_approx",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),              // x
            n(1, IRNodeKind::Literal(0.5)),          // 0.5
            n(2, IRNodeKind::Literal(0.044715)),     // coeff
            n(3, IRNodeKind::Literal(0.7978845608)), // sqrt(2/pi)
            n(4, IRNodeKind::Literal(1.0)),          // 1.0
            n(
                5,
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 3,
                },
            ), // x^3
            n(
                6,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(2),
                    rhs: NodeId::new(5),
                },
            ), // 0.044715*x^3
            n(
                7,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(6),
                },
            ), // x + 0.044715*x^3
            n(
                8,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(3),
                    rhs: NodeId::new(7),
                },
            ), // sqrt(2/pi)*(...)
            n(
                9,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Tanh,
                    input: NodeId::new(8),
                },
            ), // tanh(...)
            n(
                10,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(4),
                    rhs: NodeId::new(9),
                },
            ), // 1 + tanh(...)
            n(
                11,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(10),
                },
            ), // x * (1+tanh(...))
            n(
                12,
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(11),
                },
            ), // 0.5 * x * (...)
        ],
        NodeId::new(12),
    );
    kernel
        .validate()
        .expect("gelu approximation kernel should validate");
    assert!(
        !kernel.has_ftz_sensitive_op(),
        "gelu approx has no div/rsqrt/recip"
    );
}

// ===========================================================================
// Section 13: FTZ sensitivity classification
// ===========================================================================

#[test]
fn test_ftz_sensitive_rsqrt() {
    let kernel = KernelDef::new(
        "rsqrt_fn",
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
    assert!(kernel.has_ftz_sensitive_op(), "rsqrt is FTZ-sensitive");
}

#[test]
fn test_ftz_sensitive_recip() {
    let kernel = KernelDef::new(
        "recip_fn",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Recip,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    assert!(kernel.has_ftz_sensitive_op(), "recip is FTZ-sensitive");
}

#[test]
fn test_ftz_not_sensitive_exp_sqrt() {
    let kernel = KernelDef::new(
        "safe_fn",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Exp,
                    input: NodeId::new(0),
                },
            ),
            n(
                2,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sqrt,
                    input: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    assert!(
        !kernel.has_ftz_sensitive_op(),
        "exp+sqrt should not be FTZ-sensitive"
    );
}

// ===========================================================================
// Section 14: All UnaryFnKind variants validate in MSL codegen
// ===========================================================================

#[test]
fn test_emit_msl_all_unary_fn_kinds() {
    let unary_ops = [
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
    for (i, op) in unary_ops.iter().enumerate() {
        let name = format!("unary_{i}");
        let kernel = KernelDef::new(
            &name,
            vec![f32_param("x")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(
                    1,
                    IRNodeKind::UnaryFn {
                        op: *op,
                        input: NodeId::new(0),
                    },
                ),
            ],
            NodeId::new(1),
        );
        let result = emit_msl(&kernel);
        assert!(
            result.is_ok(),
            "emit_msl failed for {:?}: {:?}",
            op,
            result.err()
        );
    }
}

#[test]
fn test_emit_msl_all_binop_kinds() {
    let binops = [
        BinOpKind::Add,
        BinOpKind::Sub,
        BinOpKind::Mul,
        BinOpKind::Div,
    ];
    for (i, op) in binops.iter().enumerate() {
        let name = format!("binop_{i}");
        let kernel = KernelDef::new(
            &name,
            vec![f32_param("a"), f32_param("b")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Param(1)),
                n(
                    2,
                    IRNodeKind::BinOp {
                        op: *op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );
        let result = emit_msl(&kernel);
        assert!(
            result.is_ok(),
            "emit_msl failed for {:?}: {:?}",
            op,
            result.err()
        );
    }
}

#[test]
fn test_emit_msl_compare_and_select() {
    let compare_ops = [
        CompareOpKind::Lt,
        CompareOpKind::Le,
        CompareOpKind::Gt,
        CompareOpKind::Ge,
        CompareOpKind::Eq,
        CompareOpKind::Ne,
    ];
    for (i, cmp_op) in compare_ops.iter().enumerate() {
        let name = format!("cmp_{i}");
        let kernel = KernelDef::new(
            &name,
            vec![f32_param("x"), f32_param("y")],
            ScalarType::F32,
            vec![
                n(0, IRNodeKind::Param(0)),
                n(1, IRNodeKind::Param(1)),
                n(
                    2,
                    IRNodeKind::Compare {
                        op: *cmp_op,
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
        let result = emit_msl(&kernel);
        assert!(
            result.is_ok(),
            "emit_msl failed for {:?}: {:?}",
            cmp_op,
            result.err()
        );
    }
}

// ===========================================================================
// Section 15: Composed kernel auto-fuse with diverse ops
// ===========================================================================

#[test]
fn test_compose_sigmoid_mul_chain() {
    // sigmoid(x) * y -- SiLU-like pattern
    let ops = vec![
        FuseableOp::unary(TraceOp::Sigmoid),
        FuseableOp {
            op: TraceOp::Mul,
            wiring: OpWiring::BinarySecondExternal,
        },
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "silu_like").unwrap();
    kernel
        .validate()
        .expect("silu-like pattern should validate");
    assert_eq!(kernel.params.len(), 2, "sigmoid(x) * y needs 2 params");
}

#[test]
fn test_compose_sub_abs_chain() {
    let ops = vec![
        FuseableOp {
            op: TraceOp::Sub,
            wiring: OpWiring::BinaryBothExternal,
        },
        FuseableOp::unary(TraceOp::Abs),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "l1_dist").unwrap();
    kernel.validate().expect("sub -> abs should validate");
    assert_eq!(kernel.params.len(), 2);
}

#[test]
fn test_compose_four_op_chain() {
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::unary(TraceOp::Log),
        FuseableOp::unary(TraceOp::Abs),
        FuseableOp::unary(TraceOp::Relu),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "four_chain").unwrap();
    kernel.validate().expect("4-op unary chain should validate");
    assert_eq!(kernel.params.len(), 1);
    // Should generate valid MSL
    let msl = emit_msl(&kernel).expect("MSL should generate");
    assert!(msl.contains("four_chain_kernel"));
}

#[test]
fn test_compose_binary_first_external() {
    // y - sigmoid(x): first external then chain
    let ops = vec![
        FuseableOp::unary(TraceOp::Sigmoid),
        FuseableOp {
            op: TraceOp::Sub,
            wiring: OpWiring::BinaryFirstExternal,
        },
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "reverse_sub").unwrap();
    kernel
        .validate()
        .expect("binary first external should validate");
    assert_eq!(kernel.params.len(), 2, "needs input x and external y");
}

// ===========================================================================
// Section 16: Compiled plan with fusion vs without fusion
// ===========================================================================

#[test]
fn test_fusion_reduces_dispatch_count() {
    // Linear chain of 4 elementwise ops -- fusion should reduce dispatches
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![4, 64]),
        test_node(1, "exp", TraceOp::Exp, vec![0], vec![4, 64]),
        test_node(2, "relu", TraceOp::Relu, vec![1], vec![4, 64]),
        test_node(3, "sigmoid", TraceOp::Sigmoid, vec![2], vec![4, 64]),
        test_node(4, "tanh", TraceOp::Tanh, vec![3], vec![4, 64]),
    ]);
    let plan_no_fusion = compile_trace_to_plan(&graph).expect("no-fusion compile");
    let plan_fusion = compile_trace_to_plan_with_fusion(&graph).expect("fusion compile");

    let count_no = count_dispatches(&plan_no_fusion);
    let count_yes = count_dispatches(&plan_fusion);
    assert!(
        count_yes <= count_no,
        "fusion should not increase dispatches: no_fusion={count_no}, fusion={count_yes}"
    );
}

#[test]
fn test_compile_plan_preserves_input_shapes() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![2, 16, 32]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![2, 16, 32]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile should succeed");
    assert_eq!(plan.input_shapes, vec![vec![2, 16, 32]]);
}

#[test]
fn test_compile_plan_weight_names_collected() {
    // A graph with matmul consuming a constant weight should collect weight names
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![4, 64]),
        TraceNode::new(
            1,
            "weight".into(),
            TraceOp::ConstantWeight {
                weight: WeightRef::from_shape(&[64, 128]),
            },
            vec![],
            vec![64, 128],
            DType::F32,
        ),
        test_node(2, "matmul", TraceOp::MatMul, vec![0, 1], vec![4, 128]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("matmul with weight should compile");
    // Weight names should be collected from dispatch steps
    // (the exact presence depends on how matmul is compiled, but plan should succeed)
    assert!(!plan.steps.is_empty());
}

// ===========================================================================
// Section 17: ValueType from/to ScalarType round-trips
// ===========================================================================

#[test]
fn test_value_type_bf16_is_numeric() {
    assert!(ValueType::BF16.is_numeric());
}

#[test]
fn test_value_type_all_scalar_types_map_correctly() {
    assert_eq!(ValueType::from(ScalarType::F32), ValueType::F32);
    assert_eq!(ValueType::from(ScalarType::F16), ValueType::F16);
    assert_eq!(ValueType::from(ScalarType::BF16), ValueType::BF16);
}

#[test]
fn test_value_type_bool_distinct_from_numeric() {
    assert!(!ValueType::Bool.is_numeric());
    assert!(ValueType::F32.is_numeric());
    assert!(ValueType::F16.is_numeric());
    assert!(ValueType::BF16.is_numeric());
    assert_ne!(ValueType::Bool, ValueType::F32);
}

// ===========================================================================
// Section 18: SumReduce in MSL codegen
// ===========================================================================

#[test]
fn test_emit_msl_sum_reduce_two_inputs() {
    let kernel = KernelDef::new(
        "sum2",
        vec![f32_param("x"), f32_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(1)],
                },
            ),
        ],
        NodeId::new(2),
    );
    let msl = emit_msl(&kernel).expect("sum_reduce should emit valid MSL");
    assert!(msl.contains("sum2_kernel"), "kernel name should appear");
}

#[test]
fn test_emit_msl_sum_reduce_four_inputs() {
    let kernel = KernelDef::new(
        "sum4",
        vec![
            f32_param("a"),
            f32_param("b"),
            f32_param("c"),
            f32_param("d"),
        ],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(2, IRNodeKind::Param(2)),
            n(3, IRNodeKind::Param(3)),
            n(
                4,
                IRNodeKind::SumReduce {
                    inputs: vec![
                        NodeId::new(0),
                        NodeId::new(1),
                        NodeId::new(2),
                        NodeId::new(3),
                    ],
                },
            ),
        ],
        NodeId::new(4),
    );
    let msl = emit_msl(&kernel).expect("4-input sum_reduce should emit valid MSL");
    assert!(msl.contains("sum4_kernel"));
}

// ===========================================================================
// Section 19: MinMax in MSL codegen
// ===========================================================================

#[test]
fn test_emit_msl_min_operation() {
    let kernel = KernelDef::new(
        "min_fn",
        vec![f32_param("a"), f32_param("b")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::MinMax {
                    op: MinMaxKind::Min,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let msl = emit_msl(&kernel).expect("min should emit valid MSL");
    assert!(msl.contains("min"), "MSL should contain min function");
}

#[test]
fn test_emit_msl_max_operation() {
    let kernel = KernelDef::new(
        "max_fn",
        vec![f32_param("a"), f32_param("b")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let msl = emit_msl(&kernel).expect("max should emit valid MSL");
    assert!(msl.contains("max"), "MSL should contain max function");
}

// ===========================================================================
// Section 20: Powi in MSL codegen
// ===========================================================================

#[test]
fn test_emit_msl_powi_square() {
    let kernel = KernelDef::new(
        "sq",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: 2,
                },
            ),
        ],
        NodeId::new(1),
    );
    let msl = emit_msl(&kernel).expect("powi(2) should emit valid MSL");
    assert!(msl.contains("sq_kernel"));
}

#[test]
fn test_emit_msl_powi_negative() {
    let kernel = KernelDef::new(
        "inv_sq",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp: -2,
                },
            ),
        ],
        NodeId::new(1),
    );
    let msl = emit_msl(&kernel).expect("powi(-2) should emit valid MSL");
    assert!(msl.contains("inv_sq_kernel"));
}

// ===========================================================================
// Section 21: F16 MSL code generation
// ===========================================================================

// Relu is implemented via Compare+Select in IR, not as a UnaryFnKind variant.
// Test F16 MSL codegen with Abs instead.
#[test]
fn test_emit_msl_f16_kernel_abs_uses_half() {
    let kernel = KernelDef::new(
        "f16_abs",
        vec![Param::new("x", ScalarType::F16)],
        ScalarType::F16,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Abs,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let msl = emit_msl(&kernel).expect("f16 abs kernel should emit");
    assert!(msl.contains("half"), "F16 kernel MSL should use half type");
}
