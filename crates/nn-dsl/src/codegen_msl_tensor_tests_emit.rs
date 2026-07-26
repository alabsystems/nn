// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor MSL codegen tests: broadcast index correctness, emit validation,
//! and precision contract threading.

use crate::codegen_msl_tensor::{build_dispatch_plan, TensorMSLCodegenError};
use crate::codegen_msl_tensor_emit::{
    emit_broadcast_kernel, emit_reduce_kernel, emit_tensor_msl, emit_tensor_msl_with_contract,
};
use crate::ir::ScalarType;
use crate::precision::{PrecisionContract, PrecisionTier};
use crate::tensor_ir::{
    BroadcastAlignment, ReduceOp, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

/// Build an input→reduce TensorKernelDef for simple reduce tests.
fn simple_reduce_def(name: &str, shape: Vec<usize>, op: ReduceOp, axis: usize) -> TensorKernelDef {
    let mut out_shape = shape.clone();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1); // scalar represented as [1], matching compute_output_shape
    }
    TensorKernelDef::new(
        name.to_string(),
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: shape.clone(),
                },
                shape,
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op,
                    input: TensorNodeId::new(0),
                    axis,
                    keepdim: false,
                },
                out_shape,
            ),
        ],
        TensorNodeId::new(1),
    )
}

// --- broadcast index correctness tests ---

/// Left-aligned broadcast [4,32] -> [4,32,128] must include batch coord_0.
#[test]
fn test_broadcast_left_aligned_uses_batch_dim() {
    let msl = emit_broadcast_kernel(
        "bcast_left",
        ScalarType::F32,
        &[4, 32],
        &[4, 32, 128],
        BroadcastAlignment::Left,
    )
    .expect("small shape should not overflow");
    assert!(
        msl.contains("in_idx += coord_0"),
        "must use batch dim:\n{msl}"
    );
}

/// Right-aligned broadcast [128] -> [4,32,128] must use only coord_2.
#[test]
fn test_broadcast_right_aligned_skips_leading_dims() {
    let msl = emit_broadcast_kernel(
        "bcast_right",
        ScalarType::F32,
        &[128],
        &[4, 32, 128],
        BroadcastAlignment::Right,
    )
    .expect("small shape should not overflow");
    assert!(
        msl.contains("in_idx += coord_2"),
        "must use last dim:\n{msl}"
    );
    assert!(
        !msl.contains("in_idx += coord_0"),
        "must skip batch:\n{msl}"
    );
    assert!(
        !msl.contains("in_idx += coord_1"),
        "must skip channel:\n{msl}"
    );
}

/// Same-rank broadcast [4,32,1] -> [4,32,128]: uses coord_0/coord_1, skips dim-1 coord_2.
#[test]
fn test_broadcast_same_rank_dim1_skips_reduced_axis() {
    let msl = emit_broadcast_kernel(
        "bcast_same",
        ScalarType::F32,
        &[4, 32, 1],
        &[4, 32, 128],
        BroadcastAlignment::Left,
    )
    .expect("small shape should not overflow");
    assert!(
        msl.contains("in_idx += coord_0"),
        "must use batch dim:\n{msl}"
    );
    assert!(
        msl.contains("in_idx += coord_1"),
        "must use channel dim:\n{msl}"
    );
    assert!(
        !msl.contains("in_idx += coord_2"),
        "must skip dim-1:\n{msl}"
    );
}

/// `emit_tensor_msl` rejects non-last-axis reductions and surfaces typed errors.
#[test]
fn test_emit_tensor_msl_rejects_nonlast_axis_reduce() {
    let def = simple_reduce_def("axis0", vec![8, 4], ReduceOp::Sum, 0);
    let err = emit_tensor_msl(&def, ScalarType::F32)
        .expect_err("non-last-axis reduce should be rejected");
    assert!(
        matches!(
            &err,
            TensorMSLCodegenError::NonLastAxisReduce {
                axis,
                shape,
                ..
            } if *axis == 0 && shape.as_slice() == [8, 4]
        ),
        "unexpected error: {err:?}"
    );
}

// --- emit_tensor_msl tests ---

#[test]
fn test_emit_tensor_msl_contains_prelude() {
    let def = simple_reduce_def("pt", vec![8, 64], ReduceOp::Sum, 1);
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("tensor MSL should emit");
    assert!(msl.starts_with("#include <metal_stdlib>"));
    assert!(msl.contains("[[kernel]]"));
}

#[test]
fn test_emit_tensor_msl_reduce_and_broadcast() {
    let def = TensorKernelDef::new(
        "rb_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 4, 8],
                },
                vec![2, 4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(0),
                    axis: 2,
                    keepdim: false,
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(1),
                    target_shape: vec![2, 4, 8],
                    alignment: BroadcastAlignment::Left,
                },
                vec![2, 4, 8],
            ),
        ],
        TensorNodeId::new(2),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("tensor MSL should emit");
    assert!(msl.contains("rb_test_reduce_mean_n1"), "reduce kernel");
    assert!(msl.contains("rb_test_broadcast_n2"), "broadcast kernel");
}

// --- validation gate tests ---

/// `build_dispatch_plan` must validate IR before indexing into nodes.
/// Without the validation gate, this would panic with an index out-of-bounds.
#[test]
fn test_dispatch_plan_rejects_invalid_node_ref() {
    // Node 1 references TensorNodeId::new(99) which is out of bounds.
    let def = TensorKernelDef::new(
        "bad_ref",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 32],
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(99),
                    axis: 1,
                    keepdim: false,
                },
                vec![4],
            ),
        ],
        TensorNodeId::new(1),
    );
    let err = build_dispatch_plan(&def, ScalarType::F32)
        .expect_err("invalid node ref should be caught by validation");
    assert!(
        matches!(&err, TensorMSLCodegenError::TensorIrValidation(_)),
        "expected TensorIrValidation, got: {err:?}"
    );
}

/// `emit_tensor_msl` inherits the validation gate from `build_dispatch_plan`.
#[test]
fn test_emit_tensor_msl_rejects_empty_graph() {
    let def = TensorKernelDef::new("empty", vec![], TensorNodeId::new(0));
    let err = emit_tensor_msl(&def, ScalarType::F32)
        .expect_err("empty graph should be caught by validation");
    assert!(
        matches!(&err, TensorMSLCodegenError::TensorIrValidation(_)),
        "expected TensorIrValidation, got: {err:?}"
    );
}

/// Regression: broadcast stride overflow must return ShapeProductOverflow, not
/// silently wrap in release mode. Guards added in 7ddbc2c.
#[test]
fn test_broadcast_stride_overflow_returns_error() {
    // Shape [2, usize::MAX, 2]: stride[2]=1, stride[1]=2, stride[0]=2*MAX → overflow.
    let err = emit_broadcast_kernel(
        "bcast_overflow",
        ScalarType::F32,
        &[1],
        &[2, usize::MAX, 2],
        BroadcastAlignment::Right,
    )
    .expect_err("overflow shape should be caught by checked_mul");
    assert!(
        matches!(&err, TensorMSLCodegenError::ShapeProductOverflow { .. }),
        "expected ShapeProductOverflow, got: {err:?}"
    );
}

// --- precision contract tests ---

/// Strict mode uses Kahan compensated summation for near-f64 precision (#1814).
#[test]
fn test_reduce_strict_uses_kahan_compensation() {
    let strict = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    let msl = emit_reduce_kernel("strict_sum", ReduceOp::Sum, ScalarType::F32, strict);
    // Phase 1: Kahan compensated accumulation with `comp` error tracking
    assert!(
        msl.contains("- comp;"),
        "strict must use Kahan compensation variable:\n{msl}"
    );
    assert!(
        msl.contains("comp = (t - partial) - y;"),
        "strict must compute Kahan error term:\n{msl}"
    );
    assert!(
        msl.contains("partial = t;"),
        "strict must update partial from compensated sum:\n{msl}"
    );
    // Phase 2: Kahan-compensated tree reduction with shared_comp array
    assert!(
        msl.contains("shared_comp[lid]"),
        "strict must use shared compensation array:\n{msl}"
    );
    assert!(
        msl.contains("float a = shared[lid];"),
        "strict must use named a in tree reduction:\n{msl}"
    );
    assert!(
        msl.contains("// precision: strict"),
        "strict must emit precision comment:\n{msl}"
    );
}

/// Relaxed mode uses raw `+=` for maximum performance.
#[test]
fn test_reduce_relaxed_uses_raw_accum() {
    let relaxed = PrecisionContract::bootstrap(PrecisionTier::Relaxed, ScalarType::F32);
    let msl = emit_reduce_kernel("relaxed_sum", ReduceOp::Sum, ScalarType::F32, relaxed);
    assert!(
        msl.contains("partial += input["),
        "relaxed must use raw +=:\n{msl}"
    );
    assert!(
        msl.contains("shared[lid] += shared[lid + stride];"),
        "relaxed must use raw shared +=:\n{msl}"
    );
    assert!(
        msl.contains("// precision: relaxed"),
        "relaxed must emit precision comment:\n{msl}"
    );
}

/// `emit_tensor_msl_with_contract` threads the contract through to reduction kernels.
#[test]
fn test_emit_tensor_msl_with_contract_strict_reduction() {
    let def = simple_reduce_def("strict_t", vec![4, 32, 128], ReduceOp::Sum, 2);
    let strict = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    let msl = emit_tensor_msl_with_contract(&def, ScalarType::F32, strict)
        .expect("strict tensor MSL should emit");
    assert!(
        msl.contains("// precision: strict"),
        "strict contract must reach reduction kernel:\n{msl}"
    );
    assert!(
        msl.contains("comp = (t - partial) - y;"),
        "strict must use Kahan compensation in tensor MSL:\n{msl}"
    );
}

/// Default `emit_tensor_msl` uses Normal precision (named intermediates).
#[test]
fn test_emit_tensor_msl_default_is_normal() {
    let def = simple_reduce_def("normal_t", vec![8, 64], ReduceOp::Sum, 1);
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("normal tensor MSL should emit");
    assert!(
        msl.contains("// precision: normal"),
        "default must be normal:\n{msl}"
    );
}

// Op-specific emission tests (Conv1d, BinaryAdd, Linear) extracted to stay
// under the 500-line limit.
#[path = "codegen_msl_tensor_tests_emit_ops.rs"]
mod ops;

// Softmax emission tests extracted to stay under 500-line limit.
#[path = "codegen_msl_tensor_tests_emit_softmax.rs"]
mod softmax;
