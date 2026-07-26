// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for tensor codegen behavior on structural tensor ops.

use nn_dsl::ir::ScalarType;
use nn_dsl::{
    build_dispatch_plan, emit_tensor_msl, DispatchStep, TensorKernelDef, TensorNode, TensorNodeId,
    TensorOpKind,
};

#[test]
fn test_dispatch_plan_reshape_is_zero_copy_alias() {
    let def = TensorKernelDef::new(
        "reshape_only",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(0),
                    target_shape: vec![1, 8],
                },
                vec![1, 8],
            ),
        ],
        TensorNodeId::new(1),
    );

    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32)
        .expect("reshape should succeed as zero-copy alias");

    // Reshape emits a Reshape step (no MSL kernel).
    assert_eq!(plan.len(), 1);
    assert!(
        matches!(
            &plan[0],
            DispatchStep::Reshape {
                input,
                output,
            } if *input == TensorNodeId::new(0) && *output == TensorNodeId::new(1)
        ),
        "expected Reshape step, got: {:?}",
        plan[0]
    );
}

#[test]
fn test_dispatch_plan_rope_full_pipeline() {
    // Build the K6 RoPE kernel and verify the dispatch plan succeeds.
    let kernel =
        nn_dsl::build_rope_rotate_kernel(2, 4, 8).expect("RoPE kernel build must succeed");

    let (plan, _) =
        build_dispatch_plan(&kernel, ScalarType::F32).expect("RoPE dispatch plan must succeed");

    // Expected steps:
    // Node 0: Input (no step)
    // Node 1: Input (no step)
    // Node 2: Reshape (zero-copy)
    // Node 3: AxisSelect (even)
    // Node 4: AxisSelect (odd)
    // Node 5: Broadcast
    // Node 6: Elementwise (rope_cos)
    // Node 7: Elementwise (rope_sin)
    // Node 8: Stack
    // Node 9: Reshape (zero-copy)

    // Count step types.
    let reshapes = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Reshape { .. }))
        .count();
    let selects = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::AxisSelect { .. }))
        .count();
    let stacks = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Stack { .. }))
        .count();
    let broadcasts = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Broadcast { .. }))
        .count();
    let elementwise = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Elementwise { .. }))
        .count();

    assert_eq!(reshapes, 2, "two reshapes: to pairs and back");
    assert_eq!(selects, 2, "two axis selects: even and odd");
    assert_eq!(stacks, 1, "one stack: reassemble pairs");
    assert_eq!(broadcasts, 1, "one broadcast: freqs");
    assert_eq!(elementwise, 2, "two elementwise: rope_cos and rope_sin");
    assert_eq!(plan.len(), 8, "total steps: 2+2+1+1+2");
}

#[test]
fn test_emit_tensor_msl_rope_produces_valid_msl() {
    let kernel =
        nn_dsl::build_rope_rotate_kernel(2, 4, 8).expect("RoPE kernel build must succeed");

    let msl = emit_tensor_msl(&kernel, ScalarType::F32).expect("RoPE MSL emission must succeed");

    // Should contain axis_select and stack kernels.
    assert!(
        msl.contains("axis_select"),
        "MSL must contain axis_select kernel"
    );
    assert!(msl.contains("stack"), "MSL must contain stack kernel");
    assert!(
        msl.contains("[[kernel]]"),
        "MSL must contain kernel attributes"
    );
    // Should contain broadcast kernel for freqs.
    assert!(
        msl.contains("broadcast"),
        "MSL must contain broadcast kernel for freqs"
    );
}
