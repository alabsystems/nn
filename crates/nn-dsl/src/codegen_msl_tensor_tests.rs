// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for tensor-level MSL code generation: dispatch plans, reductions, broadcasts.
//!
//! Conv1d/ConvTranspose1d dispatch tests: codegen_msl_tensor_tests_conv.rs
//! Norm dispatch and error tests: codegen_msl_tensor_tests_norm.rs
//! Broadcast index correctness, emit validation, and precision contract tests:
//! codegen_msl_tensor_tests_emit.rs

use crate::codegen_msl_tensor::{
    build_dispatch_plan, DispatchStep, TensorMSLCodegenError, REDUCE_THREADGROUP_SIZE,
};
use crate::codegen_msl_tensor_emit::{emit_broadcast_kernel, emit_reduce_kernel};
use crate::ir::ScalarType;
use crate::precision::{PrecisionContract, PrecisionTier};
use crate::tensor_ir::{
    BroadcastAlignment, ReduceOp, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};
use crate::test_kernels::{square_kernel, sub_kernel};

/// Normal-precision contract for f32, used by existing tests.
fn normal_f32() -> PrecisionContract {
    PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32)
}

/// Normal-precision contract for f16, used by existing tests.
fn normal_f16() -> PrecisionContract {
    PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F16)
}

/// Build an input→reduce TensorKernelDef with explicit keepdim control.
fn reduce_def_keepdim(
    name: &str,
    shape: Vec<usize>,
    op: ReduceOp,
    axis: usize,
    keepdim: bool,
) -> TensorKernelDef {
    let out_shape = if keepdim {
        let mut s = shape.clone();
        s[axis] = 1;
        s
    } else {
        let mut s = shape.clone();
        s.remove(axis);
        if s.is_empty() {
            s.push(1);
        }
        s
    };
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
                    keepdim,
                },
                out_shape,
            ),
        ],
        TensorNodeId::new(1),
    )
}

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

// --- emit_reduce_kernel tests ---

#[test]
fn test_reduce_sum_f32_signature_and_buffers() {
    let msl = emit_reduce_kernel(
        "test_reduce_sum",
        ReduceOp::Sum,
        ScalarType::F32,
        normal_f32(),
    );
    assert!(msl.contains("[[kernel]] void test_reduce_sum("));
    assert!(msl.contains("device const float* input"));
    assert!(msl.contains("device float* output"));
}

#[test]
fn test_reduce_mean_f32_includes_divisor() {
    let msl = emit_reduce_kernel("mean_k", ReduceOp::Mean, ScalarType::F32, normal_f32());
    assert!(
        msl.contains("/ float(reduce_dim)"),
        "mean must divide; got:\n{msl}"
    );
}

#[test]
fn test_reduce_sum_f32_no_divisor() {
    let msl = emit_reduce_kernel("sum_k", ReduceOp::Sum, ScalarType::F32, normal_f32());
    assert!(!msl.contains("/ float(reduce_dim)"), "sum must not divide");
}

#[test]
fn test_reduce_f16_uses_half_type() {
    let msl = emit_reduce_kernel("f16_reduce", ReduceOp::Mean, ScalarType::F16, normal_f16());
    // Input/output buffers use half (storage type).
    assert!(msl.contains("device const half* input"));
    // Shared memory and accumulation use float (accumulator type) to avoid
    // catastrophic precision loss in half-precision reductions (#1352).
    assert!(
        msl.contains("threadgroup float shared"),
        "Expected float accumulator for f16 reduce, got:\n{msl}"
    );
    // Mean divisor uses accumulator type (float), final store casts back to half.
    assert!(
        msl.contains("/ float(reduce_dim)"),
        "Expected float divisor for f16 mean, got:\n{msl}"
    );
}

#[test]
fn test_reduce_kernel_structure() {
    let msl = emit_reduce_kernel("k", ReduceOp::Sum, ScalarType::F32, normal_f32());
    let barriers = msl
        .matches("threadgroup_barrier(mem_flags::mem_threadgroup)")
        .count();
    assert!(barriers >= 2, "need >=2 barriers; found {barriers}");
    let expected_shared = format!("threadgroup float shared[{REDUCE_THREADGROUP_SIZE}]");
    assert!(msl.contains(&expected_shared));
    assert!(msl.contains("[[threadgroup_position_in_grid]]"));
    assert!(msl.contains("[[thread_position_in_threadgroup]]"));
    assert!(msl.contains("[[threads_per_threadgroup]]"));
    assert!(msl.contains("if (gid >= outer_size) return;"));
}

// --- emit_broadcast_kernel tests ---

#[test]
fn test_broadcast_scalar_to_3d() {
    let msl = emit_broadcast_kernel(
        "bcast_test",
        ScalarType::F32,
        &[1],
        &[4, 32, 128],
        BroadcastAlignment::Left,
    )
    .expect("small shape should not overflow");
    assert!(msl.contains("[[kernel]] void bcast_test("));
    assert!(msl.contains("device const float* input"));
    assert!(msl.contains("if (tid >= total) return;"));
    assert!(msl.contains("output[tid] = input[in_idx];"));
}

#[test]
fn test_broadcast_2d_to_3d() {
    let msl = emit_broadcast_kernel(
        "bcast_2d",
        ScalarType::F32,
        &[4, 32],
        &[4, 32, 128],
        BroadcastAlignment::Left,
    )
    .expect("small shape should not overflow");
    assert!(msl.contains("output[tid] = input[in_idx];"));
}

#[test]
fn test_broadcast_f16_type() {
    let msl = emit_broadcast_kernel(
        "bcast_f16",
        ScalarType::F16,
        &[1],
        &[8, 16],
        BroadcastAlignment::Left,
    )
    .expect("small shape should not overflow");
    assert!(msl.contains("device const half* input"));
}

// --- build_dispatch_plan tests ---

#[test]
fn test_dispatch_plan_simple_reduce() {
    let def = simple_reduce_def("sr", vec![4, 32, 128], ReduceOp::Mean, 2);
    let (plan, _) =
        build_dispatch_plan(&def, ScalarType::F32).expect("last-axis reduce should plan");
    assert_eq!(plan.len(), 1);
    assert!(
        matches!(&plan[0], DispatchStep::Reduce { op, reduce_dim, outer_size, .. }
        if *op == ReduceOp::Mean && *reduce_dim == 128 && *outer_size == 128)
    );
}

#[test]
fn test_dispatch_plan_reduce_axis_0() {
    let def = simple_reduce_def("a0", vec![8, 4], ReduceOp::Sum, 0);
    let err = build_dispatch_plan(&def, ScalarType::F32)
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

#[test]
fn test_dispatch_plan_1d_reduce_to_scalar() {
    let def = simple_reduce_def("sc", vec![64], ReduceOp::Mean, 0);
    let (plan, _) =
        build_dispatch_plan(&def, ScalarType::F32).expect("1-D reduce to scalar should plan");
    assert_eq!(plan.len(), 1);
    assert!(
        matches!(&plan[0], DispatchStep::Reduce { reduce_dim, outer_size, .. }
        if *reduce_dim == 64 && *outer_size == 1)
    );
}

#[test]
fn test_dispatch_plan_variance_composition() {
    // var(x) = mean(x^2) - mean(x)^2 — canonical decomposition from design doc
    let s = vec![4, 32, 128];
    let s_reduced = vec![4, 32];
    let def = TensorKernelDef::new(
        "var",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: s.clone(),
                },
                s.clone(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Elementwise {
                    kernel: square_kernel(),
                    inputs: vec![TensorNodeId::new(0)],
                },
                s,
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(0),
                    axis: 2,
                    keepdim: false,
                },
                s_reduced.clone(),
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(1),
                    axis: 2,
                    keepdim: false,
                },
                s_reduced.clone(),
            ),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::Elementwise {
                    kernel: square_kernel(),
                    inputs: vec![TensorNodeId::new(2)],
                },
                s_reduced.clone(),
            ),
            TensorNode::new(
                TensorNodeId::new(5),
                TensorOpKind::Elementwise {
                    kernel: sub_kernel(),
                    inputs: vec![TensorNodeId::new(3), TensorNodeId::new(4)],
                },
                s_reduced,
            ),
        ],
        TensorNodeId::new(5),
    );
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("variance plan should build");
    assert_eq!(plan.len(), 5, "ew + reduce + reduce + ew + ew");
    assert!(matches!(plan[0], DispatchStep::Elementwise { .. }));
    assert!(matches!(plan[1], DispatchStep::Reduce { .. }));
    assert!(matches!(plan[2], DispatchStep::Reduce { .. }));
    assert!(matches!(plan[3], DispatchStep::Elementwise { .. }));
    assert!(matches!(plan[4], DispatchStep::Elementwise { .. }));
}

#[test]
fn test_dispatch_plan_with_broadcast() {
    let def = TensorKernelDef::new(
        "rb",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 32, 128],
                },
                vec![4, 32, 128],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(0),
                    axis: 2,
                    keepdim: false,
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(1),
                    target_shape: vec![4, 32, 128],
                    alignment: BroadcastAlignment::Left,
                },
                vec![4, 32, 128],
            ),
        ],
        TensorNodeId::new(2),
    );
    let (plan, _) =
        build_dispatch_plan(&def, ScalarType::F32).expect("reduce+broadcast plan should build");
    assert_eq!(plan.len(), 2);
    assert!(matches!(
        &plan[0],
        DispatchStep::Reduce {
            reduce_dim: 128,
            outer_size: 128,
            ..
        }
    ));
    assert!(
        matches!(&plan[1], DispatchStep::Broadcast { total_elements, .. } if *total_elements == 4 * 32 * 128)
    );
}

#[test]
fn test_reduce_kernel_names_are_unique() {
    let def = TensorKernelDef::new(
        "dual",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 32, 128],
                },
                vec![4, 32, 128],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(0),
                    axis: 2,
                    keepdim: false,
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Reduce {
                    op: ReduceOp::Sum,
                    input: TensorNodeId::new(0),
                    axis: 2,
                    keepdim: false,
                },
                vec![4, 32],
            ),
        ],
        TensorNodeId::new(2),
    );
    let (plan, _) =
        build_dispatch_plan(&def, ScalarType::F32).expect("dual reduce plan should build");
    assert_eq!(plan.len(), 2);
    let names: Vec<&str> = plan
        .iter()
        .filter_map(|s| match s {
            DispatchStep::Reduce { kernel_name, .. } => Some(kernel_name.as_str()),
            _ => None,
        })
        .collect();
    assert_ne!(names[0], names[1], "kernel names must be unique");
    assert!(names[0].contains("n1"));
    assert!(names[1].contains("n2"));
}

#[test]
fn test_dispatch_plan_reduce_keepdim_true() {
    let def = reduce_def_keepdim("kd", vec![4, 32, 128], ReduceOp::Sum, 2, true);
    let (plan, _) =
        build_dispatch_plan(&def, ScalarType::F32).expect("keepdim=true reduce should plan");
    assert_eq!(plan.len(), 1);
    assert!(
        matches!(&plan[0], DispatchStep::Reduce { keepdim, reduce_dim, outer_size, .. }
        if *keepdim && *reduce_dim == 128 && *outer_size == 128)
    );
}

#[test]
fn test_dispatch_plan_reduce_keepdim_false() {
    let def = reduce_def_keepdim("nkd", vec![4, 32, 128], ReduceOp::Mean, 2, false);
    let (plan, _) =
        build_dispatch_plan(&def, ScalarType::F32).expect("keepdim=false reduce should plan");
    assert_eq!(plan.len(), 1);
    assert!(
        matches!(&plan[0], DispatchStep::Reduce { keepdim, reduce_dim, outer_size, .. }
        if !*keepdim && *reduce_dim == 128 && *outer_size == 128)
    );
}
