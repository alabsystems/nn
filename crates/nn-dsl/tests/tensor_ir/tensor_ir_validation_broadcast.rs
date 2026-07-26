// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for tensor-level IR broadcast validation and
//! broadcast alignment inference.

use nn_dsl::{
    infer_broadcast_alignment, BroadcastAlignment, ReduceOp, TensorIRError, TensorKernelDef,
    TensorNode, TensorNodeId, TensorOpKind,
};

#[test]
fn test_broadcast_validates() {
    let def = TensorKernelDef::new(
        "bcast",
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

    def.validate().expect("broadcast should validate");
}

#[test]
fn test_incompatible_broadcast_rejected() {
    let def = TensorKernelDef::new(
        "bad_bcast",
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
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(0),
                    target_shape: vec![3, 16],
                    alignment: BroadcastAlignment::Left,
                },
                vec![3, 16],
            ),
        ],
        TensorNodeId::new(1),
    );

    let err = def
        .validate()
        .expect_err("incompatible broadcast should fail");
    assert!(
        matches!(err, TensorIRError::IncompatibleBroadcast { .. }),
        "expected IncompatibleBroadcast, got: {err:?}"
    );
}

#[test]
fn test_broadcast_left_aligned() {
    // [4, 32] -> [4, 32, 128]: reduce->broadcast pattern (left-aligned)
    let def = TensorKernelDef::new(
        "left_bcast",
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
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(0),
                    target_shape: vec![4, 32, 128],
                    alignment: BroadcastAlignment::Left,
                },
                vec![4, 32, 128],
            ),
        ],
        TensorNodeId::new(1),
    );

    def.validate()
        .expect("left-aligned broadcast should validate");
}

#[test]
fn test_broadcast_right_aligned() {
    // [128] -> [4, 32, 128]: weight broadcast (NumPy-style right-aligned)
    let def = TensorKernelDef::new(
        "right_bcast",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "w".to_string(),
                    shape: vec![128],
                },
                vec![128],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(0),
                    target_shape: vec![4, 32, 128],
                    alignment: BroadcastAlignment::Right,
                },
                vec![4, 32, 128],
            ),
        ],
        TensorNodeId::new(1),
    );

    def.validate()
        .expect("right-aligned broadcast should validate");
}

#[test]
fn test_broadcast_scalar() {
    // [1] -> [4, 32, 128]: scalar broadcast
    let def = TensorKernelDef::new(
        "scalar_bcast",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "eps".to_string(),
                    shape: vec![1],
                },
                vec![1],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(0),
                    target_shape: vec![4, 32, 128],
                    alignment: BroadcastAlignment::Left,
                },
                vec![4, 32, 128],
            ),
        ],
        TensorNodeId::new(1),
    );

    def.validate().expect("scalar broadcast should validate");
}

#[test]
fn test_broadcast_cannot_shrink() {
    // [4, 32, 128] -> [4, 32]: cannot shrink dimensions
    let def = TensorKernelDef::new(
        "shrink",
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
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(0),
                    target_shape: vec![4, 32],
                    alignment: BroadcastAlignment::Left,
                },
                vec![4, 32],
            ),
        ],
        TensorNodeId::new(1),
    );

    let err = def.validate().expect_err("shrink broadcast should fail");
    assert!(
        matches!(err, TensorIRError::IncompatibleBroadcast { .. }),
        "expected IncompatibleBroadcast, got: {err:?}"
    );
}

#[test]
fn test_infer_broadcast_alignment_unique_cases() {
    assert_eq!(
        infer_broadcast_alignment(&[4, 32], &[4, 32, 128]).expect("left-aligned case"),
        BroadcastAlignment::Left
    );
    assert_eq!(
        infer_broadcast_alignment(&[128], &[4, 32, 128]).expect("right-aligned case"),
        BroadcastAlignment::Right
    );
}

#[test]
fn test_infer_broadcast_alignment_rejects_ambiguous_nontrivial_case() {
    let err = infer_broadcast_alignment(&[2], &[2, 2])
        .expect_err("[2] -> [2,2] must be rejected as ambiguous");
    assert!(
        matches!(
            err,
            TensorIRError::AmbiguousBroadcast {
                ref input,
                ref target
            } if *input == vec![2] && *target == vec![2, 2]
        ),
        "expected AmbiguousBroadcast([2] -> [2,2]), got: {err:?}"
    );
}

// --- Broadcast alignment inference tests ---

#[test]
fn test_infer_broadcast_alignment_left_only() {
    // [4, 32] -> [4, 32, 128]: only left-aligned works
    let result = infer_broadcast_alignment(&[4, 32], &[4, 32, 128]);
    assert_eq!(result.unwrap(), BroadcastAlignment::Left);
}

#[test]
fn test_infer_broadcast_alignment_right_only() {
    // [128] -> [4, 32, 128]: only right-aligned works
    let result = infer_broadcast_alignment(&[128], &[4, 32, 128]);
    assert_eq!(result.unwrap(), BroadcastAlignment::Right);
}

#[test]
fn test_infer_broadcast_alignment_same_rank() {
    // Same rank -> always Left (offset is 0 either way)
    let result = infer_broadcast_alignment(&[4, 1, 128], &[4, 32, 128]);
    assert_eq!(result.unwrap(), BroadcastAlignment::Left);
}

#[test]
fn test_infer_broadcast_alignment_ambiguous() {
    // [2] -> [2, 2]: both left and right match with non-1 dim -> ambiguous
    let err = infer_broadcast_alignment(&[2], &[2, 2]).expect_err("should be ambiguous");
    assert!(
        matches!(err, TensorIRError::AmbiguousBroadcast { .. }),
        "expected AmbiguousBroadcast, got: {err:?}"
    );
}

#[test]
fn test_infer_broadcast_alignment_all_ones_not_ambiguous() {
    // [1] -> [4, 32]: all-ones input -> both alignments produce identical mapping
    let result = infer_broadcast_alignment(&[1], &[4, 32]);
    assert_eq!(result.unwrap(), BroadcastAlignment::Left);
}

#[test]
fn test_infer_broadcast_alignment_incompatible() {
    // [3] -> [4, 2]: neither alignment matches
    let err = infer_broadcast_alignment(&[3], &[4, 2]).expect_err("should be incompatible");
    assert!(
        matches!(err, TensorIRError::IncompatibleBroadcast { .. }),
        "expected IncompatibleBroadcast, got: {err:?}"
    );
}

#[test]
fn test_infer_broadcast_alignment_input_larger_rank() {
    // [4, 32, 128] -> [4, 32]: input rank > target rank
    let err =
        infer_broadcast_alignment(&[4, 32, 128], &[4, 32]).expect_err("should be incompatible");
    assert!(
        matches!(err, TensorIRError::IncompatibleBroadcast { .. }),
        "expected IncompatibleBroadcast, got: {err:?}"
    );
}

#[test]
fn test_ambiguous_broadcast_explicit_alignment_validates() {
    // [2] -> [2, 2] is ambiguous for inference, but explicit Left alignment validates.
    let def = TensorKernelDef::new(
        "explicit_ambiguous",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![2],
                },
                vec![2],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(0),
                    target_shape: vec![2, 2],
                    alignment: BroadcastAlignment::Left,
                },
                vec![2, 2],
            ),
        ],
        TensorNodeId::new(1),
    );

    // Explicit Left is valid (even though both alignments would work)
    def.validate()
        .expect("explicit Left alignment for [2]->[2,2] should validate");

    // But inference flags this as ambiguous
    let err =
        infer_broadcast_alignment(&[2], &[2, 2]).expect_err("inference should flag ambiguity");
    assert!(
        matches!(err, TensorIRError::AmbiguousBroadcast { .. }),
        "expected AmbiguousBroadcast from inference, got: {err:?}"
    );
}
