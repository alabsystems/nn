// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Norm op dispatch plan tests and error variant tests.
//!
//! Extracted from codegen_msl_tensor_tests.rs per 500-line file limit.
//! Covers: native norm auto-decomposition (#667), GroupNorm (#697),
//! UnexpandedNormOp error variant (#731).

use crate::adain::build_adain1d;
use crate::codegen_msl_tensor::{build_dispatch_plan, DispatchStep, TensorMSLCodegenError};
use crate::instance_norm::build_instance_norm_affine;
use crate::instance_norm::{build_instance_norm, build_instance_norm_decomposed};
use crate::ir::ScalarType;
use crate::rms_norm::{build_rms_norm, build_rms_norm_decomposed};
use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};

// --- Native norm op auto-decomposition tests (#667) ---

#[test]
fn test_dispatch_plan_native_instance_norm_succeeds() {
    // Native InstanceNorm1d auto-decomposes via expand_norm_ops. Verify it
    // produces the same number of dispatch steps as the explicit decomposed builder.
    let native = build_instance_norm(1, 4, 32).expect("build native");
    let decomposed = build_instance_norm_decomposed(1, 4, 32).expect("build decomposed");

    let (native_plan, _) =
        build_dispatch_plan(&native, ScalarType::F32).expect("native norm should plan");
    let (decomposed_plan, _) =
        build_dispatch_plan(&decomposed, ScalarType::F32).expect("decomposed should plan");

    assert_eq!(
        native_plan.len(),
        decomposed_plan.len(),
        "native and decomposed should produce same step count"
    );

    // Both should contain Reduce + Broadcast + Elementwise steps
    let reduce_count = native_plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Reduce { .. }))
        .count();
    assert_eq!(
        reduce_count, 2,
        "InstanceNorm needs 2 reductions (mean, variance)"
    );
}

#[test]
fn test_dispatch_plan_native_instance_norm_affine_succeeds() {
    let native = build_instance_norm_affine(1, 8, 64).expect("build affine native");
    let (plan, _) =
        build_dispatch_plan(&native, ScalarType::F32).expect("affine InstanceNorm should plan");

    // Affine adds reshape + broadcast + mul + reshape + broadcast + add (6 extra steps)
    let reduce_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Reduce { .. }))
        .count();
    assert_eq!(reduce_count, 2, "affine InstanceNorm needs 2 reductions");

    // Should have reshape steps for gamma/beta [C] -> [1,C,1]
    let reshape_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Reshape { .. }))
        .count();
    assert!(
        reshape_count >= 2,
        "affine needs reshapes for gamma/beta; got {reshape_count}"
    );
}

#[test]
fn test_dispatch_plan_native_rms_norm_succeeds() {
    let native = build_rms_norm(4, 32).expect("build native RmsNorm");
    let decomposed = build_rms_norm_decomposed(4, 32).expect("build decomposed RmsNorm");

    let (native_plan, _) =
        build_dispatch_plan(&native, ScalarType::F32).expect("native RmsNorm should plan");
    let (decomposed_plan, _) =
        build_dispatch_plan(&decomposed, ScalarType::F32).expect("decomposed should plan");

    assert_eq!(
        native_plan.len(),
        decomposed_plan.len(),
        "native and decomposed RmsNorm should produce same step count"
    );

    // RmsNorm: 1 reduction (mean of x^2)
    let reduce_count = native_plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Reduce { .. }))
        .count();
    assert_eq!(
        reduce_count, 1,
        "RmsNorm needs 1 reduction (mean of squares)"
    );
}

#[test]
fn test_dispatch_plan_native_adain1d_succeeds() {
    let native = build_adain1d(8, 64).expect("build native AdaIN1d");
    let (plan, _) =
        build_dispatch_plan(&native, ScalarType::F32).expect("native AdaIN1d should plan");

    // AdaIN1d = InstanceNorm (10 steps) + style scale/shift (6 steps)
    // Should have at least 2 reductions from InstanceNorm
    let reduce_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Reduce { .. }))
        .count();
    assert_eq!(
        reduce_count, 2,
        "AdaIN1d needs 2 reductions from InstanceNorm"
    );

    // Should have reshape steps for style_gamma/beta
    let reshape_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Reshape { .. }))
        .count();
    assert!(
        reshape_count >= 2,
        "AdaIN1d needs reshapes for style params; got {reshape_count}"
    );
}

#[test]
fn test_dispatch_plan_native_norm_step_types_are_primitive() {
    // Verify that after expansion, no native norm DispatchStep types appear.
    // All steps should be Reduce, Broadcast, Elementwise, or Reshape.
    let native = build_instance_norm(1, 4, 32).expect("build");
    let (plan, _) = build_dispatch_plan(&native, ScalarType::F32).expect("plan");

    for step in &plan {
        match step {
            DispatchStep::Reduce { .. }
            | DispatchStep::Broadcast { .. }
            | DispatchStep::Elementwise { .. }
            | DispatchStep::Reshape { .. } => {}
            other => panic!("unexpected step type after norm expansion: {other:?}"),
        }
    }
}

// --- GroupNorm dispatch plan tests (#697) ---

#[test]
fn test_dispatch_plan_group_norm_g1_no_affine() {
    // GroupNorm(groups=1) should produce a dispatch plan with only primitive steps.
    // add_group_norm_g1 creates InstanceNorm1d nodes which expand_norm_ops
    // decomposes into primitives before dispatch planning.
    use crate::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new("gn1_dispatch");
    let x = b.add_input("x", &[4, 32]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_group_norm_g1(x, eps, None, None, 4, 32);
    let def = b.build(out).expect("valid graph");

    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32)
        .expect("GroupNorm g1 dispatch plan should succeed (AC2)");

    // Should contain 2 reductions (mean, variance)
    let reduce_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Reduce { .. }))
        .count();
    assert_eq!(reduce_count, 2, "GroupNorm needs 2 reductions");

    // All steps should be primitive (no InstanceNorm1d-derived steps)
    for step in &plan {
        match step {
            DispatchStep::Reduce { .. }
            | DispatchStep::Broadcast { .. }
            | DispatchStep::Elementwise { .. }
            | DispatchStep::Reshape { .. } => {}
            other => panic!("unexpected step in GroupNorm dispatch: {other:?}"),
        }
    }
}

#[test]
fn test_dispatch_plan_group_norm_g1_affine() {
    // GroupNorm(groups=1) with affine parameters should also dispatch correctly.
    use crate::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new("gn1_affine_dispatch");
    let x = b.add_input("x", &[8, 64]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[8]);
    let beta = b.add_input("beta", &[8]);
    let out = b.add_group_norm_g1(x, eps, Some(gamma), Some(beta), 8, 64);
    let def = b.build(out).expect("valid graph");

    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32)
        .expect("Affine GroupNorm g1 dispatch should succeed (AC3)");

    // Should contain 2 reductions from norm + broadcast/mul/add for affine
    let reduce_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Reduce { .. }))
        .count();
    assert_eq!(reduce_count, 2, "affine GroupNorm needs 2 reductions");

    // Must have BinaryMul (gamma scale) and BinaryAdd (beta shift)
    let mul_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::BinaryMul { .. }))
        .count();
    let add_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::BinaryAdd { .. }))
        .count();
    assert!(mul_count >= 1, "affine needs gamma mul; got {mul_count}");
    assert!(add_count >= 1, "affine needs beta add; got {add_count}");
}

// --- UnexpandedNormOp error variant tests (#731) ---

#[test]
fn test_unexpanded_norm_op_error_display() {
    // Verify the error variant produces an informative message.
    let err = TensorMSLCodegenError::UnexpandedNormOp {
        node_id: TensorNodeId::new(7),
        op_name: "InstanceNorm1d",
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("InstanceNorm1d"),
        "error should name the op: {msg}"
    );
    assert!(
        msg.contains("expand_norm_ops"),
        "error should reference expand_norm_ops: {msg}"
    );
}

#[test]
fn test_unexpanded_norm_op_error_is_not_panic() {
    // The UnexpandedNormOp variant must be a typed error, not a panic.
    // This test ensures the variant exists and implements Error + Display.
    let err: Box<dyn std::error::Error> = Box::new(TensorMSLCodegenError::UnexpandedNormOp {
        node_id: TensorNodeId::new(0),
        op_name: "RmsNorm",
    });
    assert!(err.to_string().contains("RmsNorm"));
}

// --- Broadcast+Binary fusion peephole tests (#1815 Tier 4 D2) ---

/// Verifies that Broadcast + BinaryMul/BinaryAdd pairs are fused into
/// broadcast-aware binary ops. Uses a manually constructed kernel with
/// Broadcast→BinaryMul and Broadcast→BinaryAdd (matching production trace
/// compiler output, not norm expansion which uses Elementwise).
#[test]
fn test_broadcast_binary_fusion_reduces_broadcast_count() {
    use crate::tensor_ir::BroadcastAlignment;

    // Build a kernel: x[1,4,8] * broadcast(reshape(gamma)) + broadcast(reshape(beta))
    // Reshape [4]->[1,4,1] then broadcast [1,4,1]->[1,4,8] matches production patterns.
    let def = TensorKernelDef::new(
        "bcast_fuse",
        vec![
            // 0: input [1, 4, 8]
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![1, 4, 8],
                },
                vec![1, 4, 8],
            ),
            // 1: gamma [4]
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "gamma".into(),
                    shape: vec![4],
                },
                vec![4],
            ),
            // 2: beta [4]
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Input {
                    name: "beta".into(),
                    shape: vec![4],
                },
                vec![4],
            ),
            // 3: reshape gamma [4] -> [1, 4, 1]
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(1),
                    target_shape: vec![1, 4, 1],
                },
                vec![1, 4, 1],
            ),
            // 4: broadcast gamma [1,4,1] -> [1, 4, 8]
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(3),
                    target_shape: vec![1, 4, 8],
                    alignment: BroadcastAlignment::Left,
                },
                vec![1, 4, 8],
            ),
            // 5: x * broadcast_gamma
            TensorNode::new(
                TensorNodeId::new(5),
                TensorOpKind::BinaryMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(4),
                },
                vec![1, 4, 8],
            ),
            // 6: reshape beta [4] -> [1, 4, 1]
            TensorNode::new(
                TensorNodeId::new(6),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(2),
                    target_shape: vec![1, 4, 1],
                },
                vec![1, 4, 1],
            ),
            // 7: broadcast beta [1,4,1] -> [1, 4, 8]
            TensorNode::new(
                TensorNodeId::new(7),
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(6),
                    target_shape: vec![1, 4, 8],
                    alignment: BroadcastAlignment::Left,
                },
                vec![1, 4, 8],
            ),
            // 8: scaled + broadcast_beta
            TensorNode::new(
                TensorNodeId::new(8),
                TensorOpKind::BinaryAdd {
                    left: TensorNodeId::new(5),
                    right: TensorNodeId::new(7),
                },
                vec![1, 4, 8],
            ),
        ],
        TensorNodeId::new(8),
    );

    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("plan");

    // After fusion: both Broadcast steps should be replaced with Reshape.
    let broadcast_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Broadcast { .. }))
        .count();
    assert_eq!(broadcast_count, 0, "all broadcasts should be fused");

    // Both binary ops should have broadcast info set.
    let fused_count = plan
        .iter()
        .filter(|s| match s {
            DispatchStep::BinaryAdd { broadcast, .. }
            | DispatchStep::BinaryMul { broadcast, .. } => broadcast.is_some(),
            _ => false,
        })
        .count();
    assert_eq!(
        fused_count, 2,
        "both BinaryMul and BinaryAdd should have broadcast info"
    );

    // Plan: 4 Reshape (2 explicit + 2 fused broadcasts) + 1 BinaryMul + 1 BinaryAdd
    // (3 Input nodes produce no steps)
    let non_reshape = plan
        .iter()
        .filter(|s| !matches!(s, DispatchStep::Reshape { .. }))
        .count();
    assert_eq!(
        non_reshape, 2,
        "only 2 dispatch steps (BinaryMul + BinaryAdd)"
    );
}

/// Verifies the MSL string content of broadcast-aware binary ops.
/// Checks that modular indexing (`in_idx`) is emitted correctly and that
/// the broadcast side (left vs right operand) uses the correct index.
#[test]
fn test_broadcast_binary_fusion_msl_content() {
    use crate::tensor_ir::BroadcastAlignment;

    // gamma[1,4,1] broadcast to [1,4,8], then BinaryMul with x[1,4,8].
    // Broadcast is on the RIGHT operand.
    let def = TensorKernelDef::new(
        "bcast_msl",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![1, 4, 8],
                },
                vec![1, 4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "gamma".into(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(1),
                    target_shape: vec![1, 4, 1],
                },
                vec![1, 4, 1],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(2),
                    target_shape: vec![1, 4, 8],
                    alignment: BroadcastAlignment::Left,
                },
                vec![1, 4, 8],
            ),
            // x * broadcast_gamma: broadcast is on the right
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::BinaryMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(3),
                },
                vec![1, 4, 8],
            ),
        ],
        TensorNodeId::new(4),
    );

    let msl = crate::emit_tensor_msl(&def, ScalarType::F32).expect("emit MSL");

    // The broadcast side is Right (gamma is smaller, on the right operand).
    // So left should use flat tid, right should use in_idx.
    assert!(
        msl.contains("left[tid]"),
        "flat operand should use tid: {msl}"
    );
    assert!(
        msl.contains("right[in_idx]"),
        "broadcast operand should use in_idx: {msl}"
    );

    // Verify modular indexing is present: remainder / stride, remainder % stride.
    assert!(
        msl.contains("in_idx"),
        "broadcast index body should declare in_idx: {msl}"
    );
    assert!(
        msl.contains("remainder"),
        "broadcast index body should use remainder: {msl}"
    );

    // For [1,4,1] -> [1,4,8], out_strides = [32, 8, 1].
    // Only dim 1 contributes to in_idx (input dims 0 and 2 are size 1).
    // in_strides for [1,4,1] = [4, 1, 1], so dim 1 contributes coord_1 * 1.
    assert!(
        msl.contains("coord_1"),
        "should decompose output coords: {msl}"
    );
    assert!(
        msl.contains("in_idx += coord_1 * 1;"),
        "dim 1 should contribute to in_idx: {msl}"
    );

    // Dims 0 and 2 are size 1 in input — they should NOT contribute to in_idx.
    assert!(
        !msl.contains("in_idx += coord_0"),
        "dim 0 (size 1) should be skipped: {msl}"
    );
    assert!(
        !msl.contains("in_idx += coord_2"),
        "dim 2 (size 1) should be skipped: {msl}"
    );

    // Output uses fused broadcast multiplication.
    assert!(
        msl.contains("left[tid] * right[in_idx]"),
        "should emit fused broadcast mul: {msl}"
    );
}

/// Verifies broadcast-aware binary ops with Right alignment (NumPy-style).
/// Input [4] broadcast to [2, 4] with Right alignment.
#[test]
fn test_broadcast_binary_fusion_right_alignment_msl() {
    use crate::tensor_ir::BroadcastAlignment;

    let def = TensorKernelDef::new(
        "bcast_right",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "bias".into(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(1),
                    target_shape: vec![2, 4],
                    alignment: BroadcastAlignment::Right,
                },
                vec![2, 4],
            ),
            // x + broadcast_bias: broadcast is on the right
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::BinaryAdd {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(2),
                },
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(3),
    );

    let msl = crate::emit_tensor_msl(&def, ScalarType::F32).expect("emit MSL");

    // Right alignment: offset = 2 - 1 = 1, so dim 0 is skipped (before offset).
    // Only dim 1 maps to input dim 0 (size 4), contributes coord_1 * 1.
    assert!(
        msl.contains("left[tid]"),
        "flat operand should use tid: {msl}"
    );
    assert!(
        msl.contains("right[in_idx]"),
        "broadcast operand should use in_idx: {msl}"
    );
    assert!(
        msl.contains("in_idx += coord_1 * 1;"),
        "dim 1 maps to input dim 0: {msl}"
    );
    assert!(
        !msl.contains("in_idx += coord_0"),
        "dim 0 skipped (before offset): {msl}"
    );
    assert!(
        msl.contains("left[tid] + right[in_idx]"),
        "should emit fused broadcast add: {msl}"
    );
}

/// Verifies broadcast on the LEFT operand emits correct index assignment.
#[test]
fn test_broadcast_binary_fusion_left_operand() {
    use crate::tensor_ir::BroadcastAlignment;

    let def = TensorKernelDef::new(
        "bcast_left",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "scale".into(),
                    shape: vec![3],
                },
                vec![3],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![2, 3],
                },
                vec![2, 3],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(0),
                    target_shape: vec![2, 3],
                    alignment: BroadcastAlignment::Right,
                },
                vec![2, 3],
            ),
            // broadcast_scale * x: broadcast is on the LEFT
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::BinaryMul {
                    left: TensorNodeId::new(2),
                    right: TensorNodeId::new(1),
                },
                vec![2, 3],
            ),
        ],
        TensorNodeId::new(3),
    );

    let msl = crate::emit_tensor_msl(&def, ScalarType::F32).expect("emit MSL");

    // Broadcast is on the LEFT operand, so left uses in_idx, right uses tid.
    assert!(
        msl.contains("left[in_idx]"),
        "broadcast on left should use in_idx: {msl}"
    );
    assert!(
        msl.contains("right[tid]"),
        "flat operand on right should use tid: {msl}"
    );
}
