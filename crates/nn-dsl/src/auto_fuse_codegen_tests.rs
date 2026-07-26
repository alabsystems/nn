// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for auto-fuse codegen: TraceOp chain → KernelDef → MSL.

use nn_core::dyn_tensor::trace::TraceOp;

use super::{auto_fuse_to_msl, compose_trace_ops_to_kernel_ir, FuseableOp};
use crate::ir::{IRNodeKind, MinMaxKind, UnaryFnKind};

#[test]
fn test_single_unary_op() {
    let ops = vec![FuseableOp::unary(TraceOp::Exp)];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "test_exp").unwrap();

    assert_eq!(kernel.params.len(), 1, "single unary has 1 param");
    assert_eq!(kernel.name, "test_exp");
    // Param + UnaryFn = 2 nodes
    assert_eq!(kernel.nodes.len(), 2);
    assert!(matches!(kernel.nodes[0].kind, IRNodeKind::Param(0)));
    assert!(matches!(
        kernel.nodes[1].kind,
        IRNodeKind::UnaryFn {
            op: UnaryFnKind::Exp,
            ..
        }
    ));
}

#[test]
fn test_two_unary_chain() {
    // exp → relu
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::unary(TraceOp::Relu),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "test_exp_relu").unwrap();

    assert_eq!(kernel.params.len(), 1, "exp→relu has 1 external input");
    // Param(0), UnaryFn(Exp), Literal(0.0), MinMax(Max) = 4 nodes
    assert!(kernel.nodes.len() >= 4);
}

#[test]
fn test_three_unary_chain() {
    // exp → sqrt → tanh
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::unary(TraceOp::Sqrt),
        FuseableOp::unary(TraceOp::Tanh),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "test_3chain").unwrap();

    assert_eq!(kernel.params.len(), 1);
    // Param + Exp + Sqrt + Tanh = 4 nodes
    assert_eq!(kernel.nodes.len(), 4);
}

#[test]
fn test_binary_both_external_then_unary() {
    // add(x, y) → relu
    let ops = vec![
        FuseableOp::binary_both_external(TraceOp::Add),
        FuseableOp::unary(TraceOp::Relu),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "test_add_relu").unwrap();

    assert_eq!(kernel.params.len(), 2, "add(x,y)→relu has 2 params");
    // Param(0), Param(1), BinOp(Add), Literal(0), MinMax(Max) = 5
    assert!(kernel.nodes.len() >= 5);
}

#[test]
fn test_binary_second_external() {
    // exp(x) → add(_, y) → relu
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::binary_second_external(TraceOp::Add),
        FuseableOp::unary(TraceOp::Relu),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "test_exp_add_relu").unwrap();

    assert_eq!(
        kernel.params.len(),
        2,
        "exp→add(_,y)→relu has 2 external inputs"
    );
}

#[test]
fn test_binary_first_external() {
    // exp(x) → sub(y, _) where y is new external, _ is chain output
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::binary_first_external(TraceOp::Sub),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "test_sub_ext_first").unwrap();

    assert_eq!(kernel.params.len(), 2);
}

#[test]
fn test_msl_generation_single_op() {
    let ops = vec![FuseableOp::unary(TraceOp::Exp)];
    let fused = auto_fuse_to_msl(&ops, "nn_exp").unwrap();

    assert_eq!(fused.num_external_inputs, 1);
    assert_eq!(fused.entry_point, "nn_exp_kernel");
    assert!(fused.msl_source.contains("nn_exp_kernel"));
    assert!(fused.msl_source.contains("exp("));
    assert!(fused.msl_source.contains("device const float*"));
    assert!(fused.msl_source.contains("[[kernel]]"));
}

#[test]
fn test_msl_generation_chain() {
    // exp → relu → add(_, y)
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::unary(TraceOp::Relu),
        FuseableOp::binary_second_external(TraceOp::Add),
    ];
    let fused = auto_fuse_to_msl(&ops, "fused_exp_relu_add").unwrap();

    assert_eq!(fused.num_external_inputs, 2);
    assert_eq!(fused.entry_point, "fused_exp_relu_add_kernel");
    // MSL should have 2 input buffers + 1 output + 1 total count
    assert!(fused.msl_source.contains("buffer(0)"));
    assert!(fused.msl_source.contains("buffer(1)"));
    // Output buffer
    assert!(fused.msl_source.contains("buffer(2)"));
}

#[test]
fn test_msl_generation_sigmoid_silu() {
    let ops = vec![
        FuseableOp::unary(TraceOp::Sigmoid),
        FuseableOp::unary(TraceOp::Silu),
    ];
    let fused = auto_fuse_to_msl(&ops, "sig_silu").unwrap();

    assert_eq!(fused.num_external_inputs, 1);
    assert!(fused.msl_source.contains("exp("));
    // Sigmoid and silu both use exp
}

#[test]
fn test_empty_chain_errors() {
    let ops: Vec<FuseableOp> = vec![];
    let result = compose_trace_ops_to_kernel_ir(&ops, "empty");
    assert!(result.is_err());
}

#[test]
fn test_unsupported_op_errors() {
    // MatMul is not elementwise
    let ops = vec![FuseableOp::unary(TraceOp::MatMul)];
    let result = compose_trace_ops_to_kernel_ir(&ops, "bad_op");
    assert!(result.is_err());
}

#[test]
fn test_binary_both_external_not_first_errors() {
    // BinaryBothExternal is only valid for the first op
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::binary_both_external(TraceOp::Add),
    ];
    let result = compose_trace_ops_to_kernel_ir(&ops, "bad_wiring");
    assert!(result.is_err());
}

#[test]
fn test_all_unary_ops_produce_valid_msl() {
    let unary_ops = vec![
        TraceOp::Exp,
        TraceOp::Log,
        TraceOp::Sqrt,
        TraceOp::Sqr,
        TraceOp::Abs,
        TraceOp::Neg,
        TraceOp::Recip,
        TraceOp::Sin,
        TraceOp::Cos,
        TraceOp::Floor,
        TraceOp::Round,
        TraceOp::Fract,
        TraceOp::Tanh,
        TraceOp::Relu,
        TraceOp::Gelu,
        TraceOp::GeluErf,
        TraceOp::Sigmoid,
        TraceOp::Silu,
        TraceOp::Softplus,
        TraceOp::LeakyRelu { slope: 0.01 },
        TraceOp::Elu { alpha: 1.0 },
        TraceOp::Clamp {
            min: Some(-1.0),
            max: Some(1.0),
        },
        TraceOp::Powf { exponent: 2.0 },
    ];

    for (i, op) in unary_ops.into_iter().enumerate() {
        let ops = vec![FuseableOp::unary(op)];
        let result = auto_fuse_to_msl(&ops, &format!("test_op_{i}"));
        assert!(
            result.is_ok(),
            "MSL generation failed for unary op {i}: {:?}",
            result.err()
        );
        let fused = result.unwrap();
        assert_eq!(fused.num_external_inputs, 1);
        assert!(fused.msl_source.contains("[[kernel]]"));
    }
}

#[test]
fn test_all_binary_ops_produce_valid_msl() {
    let binary_ops = vec![
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
        TraceOp::Minimum,
        TraceOp::Atan2,
    ];

    for (i, op) in binary_ops.into_iter().enumerate() {
        let ops = vec![FuseableOp::binary_both_external(op)];
        let result = auto_fuse_to_msl(&ops, &format!("test_binop_{i}"));
        assert!(
            result.is_ok(),
            "MSL generation failed for binary op {i}: {:?}",
            result.err()
        );
        let fused = result.unwrap();
        assert_eq!(fused.num_external_inputs, 2);
    }
}

#[test]
fn test_long_chain_five_ops() {
    // exp → relu → abs → neg → tanh
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::unary(TraceOp::Relu),
        FuseableOp::unary(TraceOp::Abs),
        FuseableOp::unary(TraceOp::Neg),
        FuseableOp::unary(TraceOp::Tanh),
    ];
    let fused = auto_fuse_to_msl(&ops, "chain5").unwrap();

    assert_eq!(fused.num_external_inputs, 1);
    assert!(fused.msl_source.contains("chain5_kernel"));
}

#[test]
fn test_mixed_chain_with_multiple_external_inputs() {
    // add(x, y) → mul(_, z) → relu
    let ops = vec![
        FuseableOp::binary_both_external(TraceOp::Add),
        FuseableOp::binary_second_external(TraceOp::Mul),
        FuseableOp::unary(TraceOp::Relu),
    ];
    let fused = auto_fuse_to_msl(&ops, "mixed_3ext").unwrap();

    assert_eq!(
        fused.num_external_inputs, 3,
        "add(x,y)→mul(_,z)→relu = 3 inputs"
    );
    assert!(fused.msl_source.contains("buffer(0)"));
    assert!(fused.msl_source.contains("buffer(1)"));
    assert!(fused.msl_source.contains("buffer(2)"));
}

#[test]
fn test_kernel_def_validates() {
    // Verify that all composed kernels pass KernelDef::validate().
    let ops = vec![
        FuseableOp::unary(TraceOp::GeluErf),
        FuseableOp::binary_second_external(TraceOp::Mul),
        FuseableOp::unary(TraceOp::Sigmoid),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "validate_test").unwrap();

    // validate() is called internally, but double-check
    assert!(kernel.validate().is_ok());
}

#[test]
fn test_powf_even_integer() {
    let ops = vec![FuseableOp::unary(TraceOp::Powf { exponent: 4.0 })];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "pow4").unwrap();
    assert_eq!(kernel.params.len(), 1);
    // Even integer: uses |x|^n directly, no sign select
    let has_select = kernel
        .nodes
        .iter()
        .any(|n| matches!(n.kind, IRNodeKind::Select { .. }));
    assert!(!has_select, "even powf should not need Select");
}

#[test]
fn test_powf_odd_integer() {
    let ops = vec![FuseableOp::unary(TraceOp::Powf { exponent: 3.0 })];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "pow3").unwrap();
    // Odd integer: needs sign correction via Select
    let has_select = kernel
        .nodes
        .iter()
        .any(|n| matches!(n.kind, IRNodeKind::Select { .. }));
    assert!(has_select, "odd powf needs Select for sign correction");
}

#[test]
fn test_clamp_min_only() {
    let ops = vec![FuseableOp::unary(TraceOp::Clamp {
        min: Some(0.0),
        max: None,
    })];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "clamp_min").unwrap();
    // Clamp with min only = max(x, min_val)
    let has_max = kernel.nodes.iter().any(|n| {
        matches!(
            n.kind,
            IRNodeKind::MinMax {
                op: MinMaxKind::Max,
                ..
            }
        )
    });
    assert!(has_max);
}
