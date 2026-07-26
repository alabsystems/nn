// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Validation tests for MSL code generation: invalid kernels, buffer limits,
//! and reserved word rejection.
//!
//! Extracted from codegen_msl_tests.rs in #1565.

use super::*;
use crate::ir::{BinOpKind, IRNode, Param};

// ======================== invalid KernelDef validation ========================

/// Build a KernelDef with a forward reference (node 0 references node 1).
fn invalid_kernel_forward_ref() -> KernelDef {
    KernelDef::new(
        "bad",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(
                NodeId::new(0),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(1), // forward reference — invalid
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(0)),
        ],
        NodeId::new(0),
    )
}

#[test]
fn test_emit_msl_rejects_invalid_kernel() {
    let kernel = invalid_kernel_forward_ref();
    let err = emit_msl(&kernel).expect_err("forward-ref kernel should be rejected");
    assert!(
        err.to_string().contains("forward or self reference"),
        "expected forward-ref error, got: {err}"
    );
}

#[test]
fn test_emit_scalar_fn_rejects_invalid_kernel() {
    let kernel = invalid_kernel_forward_ref();
    let err = emit_scalar_fn(&kernel).expect_err("forward-ref kernel should be rejected");
    assert!(
        err.to_string().contains("forward or self reference"),
        "expected forward-ref error, got: {err}"
    );
}

#[test]
fn test_emit_kani_harness_rejects_invalid_kernel() {
    use crate::codegen_kani::emit_kani_harness;
    let kernel = invalid_kernel_forward_ref();
    let err = emit_kani_harness(&kernel).expect_err("forward-ref kernel should be rejected");
    assert!(
        err.to_string().contains("forward or self reference"),
        "expected forward-ref error, got: {err}"
    );
}

#[test]
fn test_emit_differential_test_rejects_invalid_kernel() {
    use crate::codegen_difftest::emit_differential_test;
    let kernel = invalid_kernel_forward_ref();
    let err = emit_differential_test(&kernel, PrecisionTier::Normal)
        .expect_err("forward-ref kernel should be rejected");
    assert!(
        err.to_string().contains("forward or self reference"),
        "expected forward-ref error, got: {err}"
    );
}

// ======================== Metal buffer limit validation (#290) ========================

/// Build a kernel with `n` f32 parameters that adds them all.
fn many_param_kernel(n: usize) -> KernelDef {
    let params: Vec<Param> = (0..n)
        .map(|i| Param::new(format!("p{i}"), ScalarType::F32))
        .collect();
    let mut nodes: Vec<IRNode> = (0..n)
        .map(|i| IRNode::new(NodeId::new(i), IRNodeKind::Param(i)))
        .collect();
    // SumReduce over all params
    let sum_id = NodeId::new(n);
    let input_ids: Vec<NodeId> = (0..n).map(NodeId::new).collect();
    nodes.push(IRNode::new(
        sum_id,
        IRNodeKind::SumReduce { inputs: input_ids },
    ));
    KernelDef::new("many_params", params, ScalarType::F32, nodes, sum_id)
}

#[test]
fn test_emit_msl_29_params_accepted() {
    // 29 params → 31 buffers (29 + out + total), highest index = 30 = MAX.
    let kernel = many_param_kernel(29);
    let result = emit_msl(&kernel);
    assert!(
        result.is_ok(),
        "29-param kernel should be accepted (31 buffers, max index 30), got: {result:?}"
    );
}

#[test]
fn test_emit_msl_30_params_rejected() {
    // 30 params → 32 buffers (30 + out + total), highest index = 31 > 30.
    let kernel = many_param_kernel(30);
    let err = emit_msl(&kernel).expect_err("30-param kernel should be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("buffer limit exceeded"),
        "error should mention buffer limit, got: {msg}"
    );
    assert!(
        msg.contains("32"),
        "error should mention required count (32), got: {msg}"
    );
    assert!(
        msg.contains("31"),
        "error should mention Metal limit (31), got: {msg}"
    );
}

/// MSL reserved word as kernel name: passes IR validation, rejected at codegen.
/// Part of #586 — backend-specific reserved words checked at emit time.
#[test]
fn test_emit_msl_rejects_reserved_kernel_name() {
    let kernel = KernelDef::new(
        "kernel",
        vec![Param::new("x".to_string(), ScalarType::F32)],
        ScalarType::F32,
        vec![IRNode::new(NodeId::new(0), IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    // IR validation passes (structural-only)
    kernel
        .validate()
        .expect("reserved words should pass IR validation");
    // MSL codegen rejects
    let err = emit_msl(&kernel).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("reserved"),
        "expected MSL reserved word error, got: {msg}"
    );
}

/// MSL reserved word as parameter name: passes IR validation, rejected at codegen.
/// Part of #586.
#[test]
fn test_emit_msl_rejects_reserved_param_name() {
    let kernel = KernelDef::new(
        "nn_kernel",
        vec![Param::new("thread".to_string(), ScalarType::F32)],
        ScalarType::F32,
        vec![IRNode::new(NodeId::new(0), IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    kernel
        .validate()
        .expect("reserved param names should pass IR validation");
    let err = emit_msl(&kernel).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("reserved") && msg.contains("parameter"),
        "expected param reserved word error, got: {msg}"
    );
}

/// emit_scalar_fn also rejects MSL reserved words.
/// Part of #586.
#[test]
fn test_emit_scalar_fn_rejects_reserved_kernel_name() {
    let kernel = KernelDef::new(
        "constant",
        vec![Param::new("x".to_string(), ScalarType::F32)],
        ScalarType::F32,
        vec![IRNode::new(NodeId::new(0), IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    let err = emit_scalar_fn(&kernel).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("reserved"),
        "expected reserved word error from scalar fn, got: {msg}"
    );
}
