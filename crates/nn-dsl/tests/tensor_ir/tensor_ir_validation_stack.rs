// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for tensor-level IR validation of Stack, Reshape,
//! and the RoPE reshape+axis_select composition pattern.

use nn_dsl::{TensorIRError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};

#[test]
fn test_stack_axis_zero_accepted() {
    // axis=0 is allowed for Stack (W1-700, 413ed054) to support LSTM dual
    // builder stacking. NY UnsqueezeLayer + ConcatLayer accept
    // axis 0, and graph_tensor_structural axis_offset handles it correctly.
    let def = TensorKernelDef::new(
        "stack_axis0",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Stack {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 0,
                },
                vec![2, 4, 6],
            ),
        ],
        TensorNodeId::new(2),
    );

    def.validate()
        .expect("stack axis=0 is allowed for LSTM dual builder support");

    // Verify output shape: axis=0 inserts num_inputs dimension at front.
    assert_eq!(def.nodes[2].shape, vec![2, 4, 6]);
}

#[test]
fn test_stack_inserts_axis_and_increases_rank() {
    let def = TensorKernelDef::new(
        "stack_rank",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Stack {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 2,
                },
                vec![4, 6, 2],
            ),
        ],
        TensorNodeId::new(2),
    );

    def.validate()
        .expect("stack should insert a new axis and increase rank by one");
}

#[test]
fn test_stack_rejects_shape_mismatch() {
    let def = TensorKernelDef::new(
        "stack_mismatch",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![4, 7],
                },
                vec![4, 7],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Stack {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 1,
                },
                vec![4, 2, 6],
            ),
        ],
        TensorNodeId::new(2),
    );

    let err = def
        .validate()
        .expect_err("stack should reject differing input shapes");
    assert!(
        matches!(err, TensorIRError::StackShapeMismatch { .. }),
        "expected StackShapeMismatch, got: {err:?}"
    );
}

#[test]
fn test_stack_rejects_axis_out_of_bounds() {
    let def = TensorKernelDef::new(
        "stack_axis_oob",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Stack {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 3,
                },
                vec![4, 6, 2],
            ),
        ],
        TensorNodeId::new(2),
    );

    let err = def
        .validate()
        .expect_err("stack axis > rank should be rejected");
    assert!(
        matches!(
            err,
            TensorIRError::StackAxisOutOfBounds { axis: 3, rank: 2 }
        ),
        "expected StackAxisOutOfBounds(axis=3, rank=2), got: {err:?}"
    );
}

#[test]
fn test_stack_empty_inputs_rejected() {
    let def = TensorKernelDef::new(
        "stack_empty",
        vec![TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Stack {
                inputs: vec![],
                axis: 1,
            },
            vec![],
        )],
        TensorNodeId::new(0),
    );

    let err = def
        .validate()
        .expect_err("stack with empty inputs must be rejected");
    assert!(
        matches!(err, TensorIRError::EmptyStack),
        "expected EmptyStack, got: {err:?}"
    );
}

#[test]
fn test_stack_axis_zero_single_input() {
    // axis=0 with a single input is also valid (unsqueeze-equivalent).
    let def = TensorKernelDef::new(
        "stack_axis0_single",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4, 6],
                },
                vec![4, 6],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Stack {
                    inputs: vec![TensorNodeId::new(0)],
                    axis: 0,
                },
                vec![1, 4, 6],
            ),
        ],
        TensorNodeId::new(1),
    );

    def.validate()
        .expect("stack axis=0 with single input is valid");

    // Verify output shape: single-input axis=0 is unsqueeze-equivalent.
    assert_eq!(def.nodes[1].shape, vec![1, 4, 6]);
}

#[test]
fn test_stack_axis_zero_three_inputs_accepted() {
    // axis=0 with three inputs is also valid after the policy change.
    let def = TensorKernelDef::new(
        "stack_axis0_three",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Input {
                    name: "c".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Stack {
                    inputs: vec![
                        TensorNodeId::new(0),
                        TensorNodeId::new(1),
                        TensorNodeId::new(2),
                    ],
                    axis: 0,
                },
                vec![3, 4],
            ),
        ],
        TensorNodeId::new(3),
    );

    def.validate()
        .expect("stack axis=0 with three inputs is valid");

    // Verify output shape: 3 rank-1 inputs stacked at axis=0 → [3, 4].
    assert_eq!(def.nodes[3].shape, vec![3, 4]);
}

#[test]
fn test_stack_axis_one_on_rank1_inputs_validates() {
    // Minimum valid axis for rank-1 inputs: axis=1. [4] x 2 -> [4, 2]
    let def = TensorKernelDef::new(
        "stack_rank1_axis1",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".to_string(),
                    shape: vec![4],
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Stack {
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                    axis: 1,
                },
                vec![4, 2],
            ),
        ],
        TensorNodeId::new(2),
    );

    def.validate()
        .expect("stack axis=1 on rank-1 inputs is the minimum valid axis");
}

// --- Reshape tests ---

#[test]
fn test_reshape_rejects_product_mismatch() {
    let def = TensorKernelDef::new(
        "reshape_bad_product",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 3, 4],
                },
                vec![2, 3, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(0),
                    target_shape: vec![2, 6],
                },
                vec![2, 6],
            ),
        ],
        TensorNodeId::new(1),
    );

    let err = def
        .validate()
        .expect_err("reshape should reject target shapes with mismatched element count");
    assert!(
        matches!(
            err,
            TensorIRError::ReshapeProductMismatch {
                input_product: 24,
                target_product: 12
            }
        ),
        "expected ReshapeProductMismatch(24 != 12), got: {err:?}"
    );
}

// --- RoPE pattern test ---

#[test]
fn test_reshape_then_axis_select_rope_pattern() {
    // The RoPE design pattern from the #105 design doc:
    // [B, S, D] -> reshape [B, S, D/2, 2] -> axis_select(axis=3, index=0) -> [B, S, D/2]
    let def = TensorKernelDef::new(
        "rope_decompose",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2, 8, 64],
                },
                vec![2, 8, 64],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reshape {
                    input: TensorNodeId::new(0),
                    target_shape: vec![2, 8, 32, 2],
                },
                vec![2, 8, 32, 2],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::AxisSelect {
                    input: TensorNodeId::new(1),
                    axis: 3,
                    index: 0,
                },
                vec![2, 8, 32],
            ),
        ],
        TensorNodeId::new(2),
    );

    def.validate()
        .expect("RoPE decompose pattern (reshape + axis_select) should validate");
}
