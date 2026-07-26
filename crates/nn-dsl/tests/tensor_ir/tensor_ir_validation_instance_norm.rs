// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for tensor-level IR validation of InstanceNorm1d.

use nn_dsl::{
    TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

/// Helper: build an InstanceNorm1d kernel def with the given input shape,
/// eps shape, axis, and output shape. Callers control shapes to test
/// both valid and invalid configurations.
fn instance_norm_def(
    input_shape: Vec<usize>,
    eps_shape: Vec<usize>,
    axis: usize,
    output_shape: Vec<usize>,
) -> TensorKernelDef {
    TensorKernelDef::new(
        "instance_norm_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: input_shape.clone(),
                },
                input_shape,
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "eps".to_string(),
                    shape: eps_shape.clone(),
                },
                eps_shape,
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis,
                    gamma: None,
                    beta: None,
                },
                output_shape,
            ),
        ],
        TensorNodeId::new(2),
    )
}

#[test]
fn test_instance_norm_valid_3d_validates() {
    // AC1: Valid InstanceNorm1d with 3D input [B, C, T] passes validation.
    let def = instance_norm_def(vec![4, 32, 128], vec![1], 2, vec![4, 32, 128]);
    def.validate()
        .expect("valid 3D InstanceNorm1d should validate");
}

#[test]
fn test_instance_norm_valid_2d_validates() {
    // AC1: Valid InstanceNorm1d with 2D input [C, T] passes validation.
    let def = instance_norm_def(vec![32, 128], vec![1], 1, vec![32, 128]);
    def.validate()
        .expect("valid 2D InstanceNorm1d should validate");
}

#[test]
fn test_instance_norm_1d_input_rejected() {
    // AC2: 1D input (rank < 2) is rejected with InstanceNormRankTooLow.
    let def = instance_norm_def(vec![128], vec![1], 0, vec![128]);
    let err = def
        .validate()
        .expect_err("1D input should fail InstanceNorm1d validation");
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::InstanceNormRankTooLow { rank: 1 })
        ),
        "expected InstanceNormRankTooLow with rank=1, got: {err:?}"
    );
}

#[test]
fn test_instance_norm_eps_not_scalar_rejected() {
    // AC3: Non-scalar eps shape is rejected with InstanceNormEpsNotScalar.
    // (The Prover's AC3 mentioned "mismatched weight shape" but InstanceNorm1d
    // has no weight node -- eps shape validation is the analogous check.)
    let def = instance_norm_def(vec![4, 32, 128], vec![32], 2, vec![4, 32, 128]);
    let err = def
        .validate()
        .expect_err("non-scalar eps should fail validation");
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::InstanceNormEpsNotScalar { .. })
        ),
        "expected InstanceNormEpsNotScalar, got: {err:?}"
    );
}

#[test]
fn test_instance_norm_eps_multi_dim_rejected() {
    // AC3 variant: multi-dimensional eps shape [2, 3] is rejected.
    let def = instance_norm_def(vec![4, 32, 128], vec![2, 3], 2, vec![4, 32, 128]);
    let err = def
        .validate()
        .expect_err("multi-dim eps should fail validation");
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::InstanceNormEpsNotScalar { .. })
        ),
        "expected InstanceNormEpsNotScalar, got: {err:?}"
    );
}

#[test]
fn test_instance_norm_axis_out_of_bounds_rejected() {
    // AC2 variant: axis >= rank is rejected (reuses ReduceAxisOutOfBounds).
    let def = instance_norm_def(vec![4, 32, 128], vec![1], 5, vec![4, 32, 128]);
    let err = def.validate().expect_err("axis 5 on 3D shape should fail");
    assert!(
        matches!(err, TensorIRError::ReduceAxisOutOfBounds { axis: 5, .. }),
        "expected ReduceAxisOutOfBounds with axis=5, got: {err:?}"
    );
}

#[test]
fn test_instance_norm_output_shape_matches_input() {
    // AC5: Output shape must match input shape (InstanceNorm preserves shape).
    // When the output shape on the node disagrees with what the op produces,
    // validation detects the mismatch.
    let def = instance_norm_def(vec![4, 32, 128], vec![1], 2, vec![4, 32]);
    let err = def.validate().expect_err("wrong output shape should fail");
    assert!(
        matches!(err, TensorIRError::IncompatibleBroadcast { .. }),
        "expected shape mismatch error, got: {err:?}"
    );
}

#[test]
fn test_instance_norm_4d_input_validates() {
    // AC2 supplement: 4D input is valid (rank >= 2).
    let def = instance_norm_def(vec![2, 4, 32, 128], vec![1], 3, vec![2, 4, 32, 128]);
    def.validate()
        .expect("4D InstanceNorm1d should validate (rank >= 2)");
}

// --- Affine InstanceNorm1d validation tests (#302) ---

/// Helper: build an InstanceNorm1d with affine params (gamma and beta inputs).
fn instance_norm_affine_def(
    input_shape: Vec<usize>,
    eps_shape: Vec<usize>,
    gamma_shape: Vec<usize>,
    beta_shape: Vec<usize>,
    axis: usize,
    output_shape: Vec<usize>,
) -> TensorKernelDef {
    TensorKernelDef::new(
        "instance_norm_affine_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: input_shape.clone(),
                },
                input_shape,
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "eps".to_string(),
                    shape: eps_shape.clone(),
                },
                eps_shape,
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Input {
                    name: "gamma".to_string(),
                    shape: gamma_shape.clone(),
                },
                gamma_shape,
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Input {
                    name: "beta".to_string(),
                    shape: beta_shape.clone(),
                },
                beta_shape,
            ),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis,
                    gamma: Some(TensorNodeId::new(2)),
                    beta: Some(TensorNodeId::new(3)),
                },
                output_shape,
            ),
        ],
        TensorNodeId::new(4),
    )
}

#[test]
fn test_instance_norm_affine_valid_3d_validates() {
    // Affine InstanceNorm with correct gamma [C] and beta [C] passes.
    let def = instance_norm_affine_def(
        vec![4, 32, 128],
        vec![1],
        vec![32],
        vec![32],
        2,
        vec![4, 32, 128],
    );
    def.validate()
        .expect("valid affine InstanceNorm1d should validate");
}

#[test]
fn test_instance_norm_affine_gamma_wrong_channels_rejected() {
    // gamma shape [16] when input [B, 32, T] expects [32].
    let def = instance_norm_affine_def(
        vec![4, 32, 128],
        vec![1],
        vec![16], // wrong: should be [32]
        vec![32],
        2,
        vec![4, 32, 128],
    );
    let err = def
        .validate()
        .expect_err("gamma with wrong channel count should fail");
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::InstanceNormAffineShapeMismatch { .. })
        ),
        "expected InstanceNormAffineShapeMismatch, got: {err:?}"
    );
}

#[test]
fn test_instance_norm_affine_beta_wrong_channels_rejected() {
    // beta shape [16] when input [B, 32, T] expects [32].
    let def = instance_norm_affine_def(
        vec![4, 32, 128],
        vec![1],
        vec![32],
        vec![16], // wrong: should be [32]
        2,
        vec![4, 32, 128],
    );
    let err = def
        .validate()
        .expect_err("beta with wrong channel count should fail");
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::InstanceNormAffineShapeMismatch { .. })
        ),
        "expected InstanceNormAffineShapeMismatch, got: {err:?}"
    );
}

#[test]
fn test_instance_norm_gamma_only_rejected() {
    // gamma without beta must be rejected.
    let def = TensorKernelDef::new(
        "instance_norm_gamma_only",
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
                TensorOpKind::Input {
                    name: "eps".to_string(),
                    shape: vec![1],
                },
                vec![1],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Input {
                    name: "gamma".to_string(),
                    shape: vec![32],
                },
                vec![32],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 2,
                    gamma: Some(TensorNodeId::new(2)),
                    beta: None, // gamma without beta
                },
                vec![4, 32, 128],
            ),
        ],
        TensorNodeId::new(3),
    );
    let err = def.validate().expect_err("gamma without beta should fail");
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::InstanceNormAffineMismatch)
        ),
        "expected InstanceNormAffineMismatch, got: {err:?}"
    );
}

#[test]
fn test_instance_norm_beta_only_rejected() {
    // beta without gamma must be rejected.
    let def = TensorKernelDef::new(
        "instance_norm_beta_only",
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
                TensorOpKind::Input {
                    name: "eps".to_string(),
                    shape: vec![1],
                },
                vec![1],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Input {
                    name: "beta".to_string(),
                    shape: vec![32],
                },
                vec![32],
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 2,
                    gamma: None,
                    beta: Some(TensorNodeId::new(2)), // beta without gamma
                },
                vec![4, 32, 128],
            ),
        ],
        TensorNodeId::new(3),
    );
    let err = def.validate().expect_err("beta without gamma should fail");
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::InstanceNormAffineMismatch)
        ),
        "expected InstanceNormAffineMismatch, got: {err:?}"
    );
}
