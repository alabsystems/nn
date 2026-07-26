// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error-path tests for ConvTranspose1d and Linear tensor IR validators.
//!
//! Extracted from `tensor_ir_validate_tests_layers.rs` (#833) to keep
//! files under the 500-line limit.

use super::*;
use crate::tensor_ir::{TensorIRConvError, TensorIRLayerError};

// ===========================================================================
// validate_conv_transpose_1d error paths
// ===========================================================================

/// ConvTranspose1d with 1-D input (rank < 2) must fail.
#[test]
fn test_conv_transpose_1d_input_rank_too_low() {
    let def = make_def(vec![
        input_node(0, "x", vec![128]),
        input_node(1, "w", vec![4, 2, 3]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::ConvTranspose1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride: 2,
                padding: 0,
                dilation: 1,
                groups: 1,
                output_padding: 0,
            },
            vec![2, 256],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::ConvTranspose1dInputRankTooLow { rank: 1 }
            ))
        ),
        "expected ConvTranspose1dInputRankTooLow, got: {err}"
    );
}

/// ConvTranspose1d with 2-D weight (must be 3-D).
#[test]
fn test_conv_transpose_1d_weight_not_3d() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "w", vec![4, 2]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::ConvTranspose1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride: 2,
                padding: 0,
                dilation: 1,
                groups: 1,
                output_padding: 0,
            },
            vec![2, 256],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::ConvTranspose1dWeightShape { .. }
            ))
        ),
        "expected ConvTranspose1dWeightShape, got: {err}"
    );
}

/// ConvTranspose1d with stride=0 must fail.
#[test]
fn test_conv_transpose_1d_zero_stride() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "w", vec![4, 2, 3]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::ConvTranspose1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride: 0,
                padding: 0,
                dilation: 1,
                groups: 1,
                output_padding: 0,
            },
            vec![2, 256],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::ConvTranspose1dZeroStride
            ))
        ),
        "expected ConvTranspose1dZeroStride, got: {err}"
    );
}

/// ConvTranspose1d in_channels=4 but weight[0]=8 — channel mismatch.
#[test]
fn test_conv_transpose_1d_channel_mismatch() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "w", vec![8, 2, 3]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::ConvTranspose1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride: 2,
                padding: 0,
                dilation: 1,
                groups: 1,
                output_padding: 0,
            },
            vec![2, 256],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::ConvTranspose1dChannelMismatch {
                    expected: 8,
                    got: 4
                }
            ))
        ),
        "expected ConvTranspose1dChannelMismatch, got: {err}"
    );
}

/// ConvTranspose1d bias shape [8] mismatches out_channels 2.
#[test]
fn test_conv_transpose_1d_bias_shape_mismatch() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "w", vec![4, 2, 3]),
        input_node(2, "bias", vec![8]),
        TensorNode::new(
            TensorNodeId::new(3),
            TensorOpKind::ConvTranspose1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: Some(TensorNodeId::new(2)),
                stride: 2,
                padding: 0,
                dilation: 1,
                groups: 1,
                output_padding: 0,
            },
            vec![2, 256],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::Conv(
                TensorIRConvError::ConvTranspose1dBiasShape { expected: 2, .. }
            ))
        ),
        "expected ConvTranspose1dBiasShape, got: {err}"
    );
}

// ===========================================================================
// validate_linear error paths
// ===========================================================================

/// Linear weight [128] (1-D) instead of 2-D [out, in].
#[test]
fn test_linear_weight_not_matrix() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "w", vec![128]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Linear {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
            },
            vec![4, 128],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::LinearWeightNotMatrix { .. })
        ),
        "expected LinearWeightNotMatrix, got: {err}"
    );
}

/// Linear input last dim 128, weight [64, 256] → in_features=256 ≠ 128.
#[test]
fn test_linear_feature_mismatch() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "w", vec![64, 256]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Linear {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
            },
            vec![4, 64],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::LinearFeatureMismatch {
                input_features: 128,
                weight_in: 256,
            })
        ),
        "expected LinearFeatureMismatch, got: {err}"
    );
}

/// Linear bias [32] mismatches out_features=64.
#[test]
fn test_linear_bias_shape_mismatch() {
    let def = make_def(vec![
        input_node(0, "x", vec![4, 128]),
        input_node(1, "w", vec![64, 128]),
        input_node(2, "bias", vec![32]),
        TensorNode::new(
            TensorNodeId::new(3),
            TensorOpKind::Linear {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: Some(TensorNodeId::new(2)),
            },
            vec![4, 64],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::LinearBiasShape { expected: 64, .. })
        ),
        "expected LinearBiasShape, got: {err}"
    );
}

// ===========================================================================
// LeakyRelu negative_slope finiteness (Part of #2315)
// ===========================================================================

/// LeakyRelu with NaN negative_slope must fail validation.
#[test]
fn test_leaky_relu_nan_slope_rejected() {
    let def = make_def(vec![
        input_node(0, "x", vec![2, 4]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::LeakyRelu {
                input: TensorNodeId::new(0),
                negative_slope: f32::NAN,
            },
            vec![2, 4],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::LeakyReluSlopeInvalid { .. })
        ),
        "expected LeakyReluSlopeInvalid, got: {err}"
    );
}

/// LeakyRelu with infinite negative_slope must fail validation.
#[test]
fn test_leaky_relu_inf_slope_rejected() {
    let def = make_def(vec![
        input_node(0, "x", vec![2, 4]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::LeakyRelu {
                input: TensorNodeId::new(0),
                negative_slope: f32::INFINITY,
            },
            vec![2, 4],
        ),
    ]);
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::LeakyReluSlopeInvalid { .. })
        ),
        "expected LeakyReluSlopeInvalid, got: {err}"
    );
}

/// LeakyRelu with finite negative_slope passes validation.
#[test]
fn test_leaky_relu_valid_slope_passes() {
    let def = make_def(vec![
        input_node(0, "x", vec![2, 4]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::LeakyRelu {
                input: TensorNodeId::new(0),
                negative_slope: 0.01,
            },
            vec![2, 4],
        ),
    ]);
    def.validate().expect("valid LeakyRelu should pass");
}
