// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error-path tests for layer-specific tensor IR validators.
//!
//! Part of #735 AC2: each of validate_rms_norm, validate_adain1d,
//! validate_conv_transpose_1d, and validate_linear has ≥1 error-path test.
//!
//! Tests exercise the validators through the public `validate()` method by
//! constructing TensorKernelDefs with intentionally invalid op configurations.
//!
//! ConvTranspose1d and Linear tests extracted to `tensor_ir_validate_tests_layers_ops.rs`
//! in #833 to keep files under the 500-line limit.

use super::*;
use crate::tensor_ir::TensorIRLayerError;

#[path = "tensor_ir_validate_tests_layers_ops.rs"]
mod ops;

// ---------------------------------------------------------------------------
// Helper: build a TensorKernelDef with given nodes, output at last node
// ---------------------------------------------------------------------------

fn make_def(nodes: Vec<TensorNode>) -> TensorKernelDef {
    let output = TensorNodeId::new(nodes.len() - 1);
    TensorKernelDef {
        name: "test".into(),
        nodes,
        output,
    }
}

// Reuse the local input_node helper from the parent test module via `super::*`.

// ===========================================================================
// validate_rms_norm error paths
// ===========================================================================

/// RmsNorm with 1-D input (rank < 2) must fail.
#[test]
fn test_rms_norm_rank_too_low() {
    let def = make_def(vec![
        input_node(0, "x", vec![128]),
        input_node(1, "eps", vec![1]),
        input_node(2, "weight", vec![128]),
        TensorNode::new(
            TensorNodeId::new(3),
            TensorOpKind::RmsNorm {
                input: TensorNodeId::new(0),
                eps: TensorNodeId::new(1),
                axis: 0,
                weight: TensorNodeId::new(2),
            },
            vec![128],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::RmsNormRankTooLow { rank: 1 })
        ),
        "expected RmsNormRankTooLow, got: {err}"
    );
}

/// RmsNorm with eps shape [2] instead of [1].
#[test]
fn test_rms_norm_eps_not_scalar() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "eps", vec![2]),
        input_node(2, "weight", vec![128]),
        TensorNode::new(
            TensorNodeId::new(3),
            TensorOpKind::RmsNorm {
                input: TensorNodeId::new(0),
                eps: TensorNodeId::new(1),
                axis: 1,
                weight: TensorNodeId::new(2),
            },
            vec![4, 128],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::RmsNormEpsNotScalar { .. })
        ),
        "expected RmsNormEpsNotScalar, got: {err}"
    );
}

/// RmsNorm with axis=0 for [4, 128] — must be last axis (1).
#[test]
fn test_rms_norm_axis_not_last() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "eps", vec![1]),
        input_node(2, "weight", vec![4]),
        TensorNode::new(
            TensorNodeId::new(3),
            TensorOpKind::RmsNorm {
                input: TensorNodeId::new(0),
                eps: TensorNodeId::new(1),
                axis: 0,
                weight: TensorNodeId::new(2),
            },
            vec![4, 128],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::RmsNormAxisNotLast { axis: 0, rank: 2 })
        ),
        "expected RmsNormAxisNotLast, got: {err}"
    );
}

/// RmsNorm weight shape [64] mismatches hidden dim 128.
#[test]
fn test_rms_norm_weight_shape_mismatch() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "eps", vec![1]),
        input_node(2, "weight", vec![64]),
        TensorNode::new(
            TensorNodeId::new(3),
            TensorOpKind::RmsNorm {
                input: TensorNodeId::new(0),
                eps: TensorNodeId::new(1),
                axis: 1,
                weight: TensorNodeId::new(2),
            },
            vec![4, 128],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::RmsNormWeightShape {
                expected_hidden: 128,
                ..
            })
        ),
        "expected RmsNormWeightShape, got: {err}"
    );
}

// ===========================================================================
// validate_adain1d error paths
// ===========================================================================

/// AdaIN1d with 1-D input (rank < 2) must fail.
#[test]
fn test_adain1d_rank_too_low() {
    let def = make_def(vec![
        input_node(0, "x", vec![128]),
        input_node(1, "eps", vec![1]),
        input_node(2, "sg", vec![1]),
        input_node(3, "sb", vec![1]),
        TensorNode::new(
            TensorNodeId::new(4),
            TensorOpKind::AdaIN1d {
                input: TensorNodeId::new(0),
                eps: TensorNodeId::new(1),
                axis: 0,
                style_gamma: TensorNodeId::new(2),
                style_beta: TensorNodeId::new(3),
            },
            vec![128],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::AdaIN1dRankTooLow { rank: 1 })
        ),
        "expected AdaIN1dRankTooLow, got: {err}"
    );
}

/// AdaIN1d style_gamma shape [8] mismatches num_channels 4.
#[test]
fn test_adain1d_style_gamma_shape_mismatch() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "eps", vec![1]),
        input_node(2, "sg", vec![8]),
        input_node(3, "sb", vec![4]),
        TensorNode::new(
            TensorNodeId::new(4),
            TensorOpKind::AdaIN1d {
                input: TensorNodeId::new(0),
                eps: TensorNodeId::new(1),
                axis: 1,
                style_gamma: TensorNodeId::new(2),
                style_beta: TensorNodeId::new(3),
            },
            vec![4, 128],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::AdaIN1dStyleShapeMismatch {
                param: "style_gamma",
                expected_channels: 4,
                ..
            })
        ),
        "expected AdaIN1dStyleShapeMismatch for gamma, got: {err}"
    );
}

/// AdaIN1d style_beta shape [16] mismatches num_channels 4.
#[test]
fn test_adain1d_style_beta_shape_mismatch() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "eps", vec![1]),
        input_node(2, "sg", vec![4]),
        input_node(3, "sb", vec![16]),
        TensorNode::new(
            TensorNodeId::new(4),
            TensorOpKind::AdaIN1d {
                input: TensorNodeId::new(0),
                eps: TensorNodeId::new(1),
                axis: 1,
                style_gamma: TensorNodeId::new(2),
                style_beta: TensorNodeId::new(3),
            },
            vec![4, 128],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::AdaIN1dStyleShapeMismatch {
                param: "style_beta",
                expected_channels: 4,
                ..
            })
        ),
        "expected AdaIN1dStyleShapeMismatch for beta, got: {err}"
    );
}

/// AdaIN1d with axis=0 for [4, 128] — must be 1 (last).
#[test]
fn test_adain1d_axis_not_last() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "eps", vec![1]),
        input_node(2, "sg", vec![4]),
        input_node(3, "sb", vec![4]),
        TensorNode::new(
            TensorNodeId::new(4),
            TensorOpKind::AdaIN1d {
                input: TensorNodeId::new(0),
                eps: TensorNodeId::new(1),
                axis: 0,
                style_gamma: TensorNodeId::new(2),
                style_beta: TensorNodeId::new(3),
            },
            vec![4, 128],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::AdaIN1dAxisNotLast { axis: 0, rank: 2 })
        ),
        "expected AdaIN1dAxisNotLast, got: {err}"
    );
}
