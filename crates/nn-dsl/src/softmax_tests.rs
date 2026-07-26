// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `build_softmax()` builder and `softmax_ref()` reference implementation.

use crate::softmax::{build_softmax, resolve_softmax_axis, softmax_ref};
use crate::tensor_ir::tensor_ir_pretty_print;
use crate::tensor_ir::{TensorIRError, TensorIRLayerError, TensorOpKind};

#[test]
fn test_build_softmax_2d_last_axis() {
    let def = build_softmax("softmax_attn", &[8, 64], -1).unwrap();
    def.validate().unwrap();
    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.nodes[1].shape, vec![8, 64]);
    match &def.nodes[1].kind {
        TensorOpKind::Softmax { axis, .. } => assert_eq!(*axis, -1),
        other => panic!("expected Softmax, got {other:?}"),
    }
}

#[test]
fn test_build_softmax_3d_axis_minus_1() {
    let def = build_softmax("sm", &[4, 8, 64], -1).unwrap();
    def.validate().unwrap();
    assert_eq!(def.nodes[1].shape, vec![4, 8, 64]);
}

#[test]
fn test_build_softmax_positive_axis() {
    let def = build_softmax("sm", &[4, 8, 64], 2).unwrap();
    def.validate().unwrap();
    assert_eq!(def.nodes[1].shape, vec![4, 8, 64]);
    match &def.nodes[1].kind {
        TensorOpKind::Softmax { axis, .. } => assert_eq!(*axis, 2),
        other => panic!("expected Softmax, got {other:?}"),
    }
}

#[test]
fn test_build_softmax_axis_0() {
    // axis=0 is valid for Softmax (unlike AxisSelect/Stack which reserve axis 0).
    let def = build_softmax("sm", &[4, 8], 0).unwrap();
    def.validate().unwrap();
    match &def.nodes[1].kind {
        TensorOpKind::Softmax { axis, .. } => assert_eq!(*axis, 0),
        other => panic!("expected Softmax, got {other:?}"),
    }
}

#[test]
fn test_build_softmax_1d() {
    let def = build_softmax("sm", &[10], 0).unwrap();
    def.validate().unwrap();
    assert_eq!(def.nodes[1].shape, vec![10]);
}

#[test]
fn test_build_softmax_axis_out_of_bounds_positive() {
    let err = build_softmax("sm", &[4, 8], 2).unwrap_err();
    assert!(matches!(
        err,
        TensorIRError::Layer(TensorIRLayerError::SoftmaxAxisOutOfBounds {
            axis: 2,
            rank: 2,
            ..
        })
    ));
}

#[test]
fn test_build_softmax_axis_out_of_bounds_negative() {
    let err = build_softmax("sm", &[4, 8], -3).unwrap_err();
    assert!(matches!(
        err,
        TensorIRError::Layer(TensorIRLayerError::SoftmaxAxisOutOfBounds {
            axis: -3,
            rank: 2,
            ..
        })
    ));
}

#[test]
fn test_build_softmax_empty_shape() {
    let err = build_softmax("sm", &[], 0).unwrap_err();
    assert!(matches!(
        err,
        TensorIRError::Layer(TensorIRLayerError::SoftmaxInputScalar)
    ));
}

#[test]
fn test_build_softmax_zero_dimension() {
    let err = build_softmax("sm", &[4, 0, 8], 0).unwrap_err();
    assert!(matches!(err, TensorIRError::EmptyDimension(_)));
}

#[test]
fn test_build_softmax_pretty_print() {
    let def = build_softmax("attn_softmax", &[4, 8, 64], -1).unwrap();
    let pp = tensor_ir_pretty_print(&def);
    assert!(pp.contains("softmax(%0, axis=-1)"));
    assert!(pp.contains("[4, 8, 64]"));
}

#[test]
fn test_build_softmax_output_node() {
    let def = build_softmax("sm", &[4, 8], -1).unwrap();
    assert_eq!(def.output.index(), 1);
}

#[test]
fn test_resolve_softmax_axis_positive() {
    assert_eq!(resolve_softmax_axis(2, 3), 2);
    assert_eq!(resolve_softmax_axis(0, 3), 0);
}

#[test]
fn test_resolve_softmax_axis_negative() {
    assert_eq!(resolve_softmax_axis(-1, 3), 2);
    assert_eq!(resolve_softmax_axis(-2, 3), 1);
    assert_eq!(resolve_softmax_axis(-3, 3), 0);
}

/// Dvoice-representative Softmax: attention weights for Qwen3-TTS
/// with typical dimensions (num_heads=8, seq_len=64).
#[test]
fn test_build_softmax_dvoice_attention() {
    // Attention: softmax(Q @ K^T / sqrt(d), dim=-1) with shape [heads, seq, seq]
    let def = build_softmax("attn_softmax", &[8, 64, 64], -1).unwrap();
    def.validate().unwrap();
    assert_eq!(def.nodes[1].shape, vec![8, 64, 64]);
}

// ---- softmax_ref numerical correctness tests ----

#[test]
fn test_softmax_ref_uniform_input() {
    // softmax([1, 1, 1, 1]) = [0.25, 0.25, 0.25, 0.25]
    let result = softmax_ref(&[1.0, 1.0, 1.0, 1.0]).unwrap();
    assert_eq!(result.len(), 4);
    for &v in &result {
        assert!((v - 0.25).abs() < 1e-6, "expected 0.25, got {v}");
    }
}

#[test]
fn test_softmax_ref_single_element() {
    // softmax([x]) = [1.0] for any finite x.
    let result = softmax_ref(&[42.0]).unwrap();
    assert_eq!(result.len(), 1);
    assert!(
        (result[0] - 1.0).abs() < 1e-6,
        "single-element softmax must be 1.0"
    );
}

#[test]
fn test_softmax_ref_sum_to_one() {
    let result = softmax_ref(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let sum: f32 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax output must sum to 1.0, got {sum}"
    );
}

#[test]
fn test_softmax_ref_output_in_unit_interval() {
    let result = softmax_ref(&[-10.0, 0.0, 10.0]).unwrap();
    for (i, &v) in result.iter().enumerate() {
        assert!(v >= 0.0, "softmax[{i}] = {v} must be >= 0");
        assert!(v <= 1.0, "softmax[{i}] = {v} must be <= 1");
    }
}

#[test]
fn test_softmax_ref_known_values() {
    // softmax([0, 0]) = [0.5, 0.5]
    let result = softmax_ref(&[0.0, 0.0]).unwrap();
    assert!((result[0] - 0.5).abs() < 1e-6);
    assert!((result[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_softmax_ref_large_positive() {
    // Numerical stability test: large values should not overflow.
    let result = softmax_ref(&[1000.0, 1000.0, 1000.0]).unwrap();
    for &v in &result {
        assert!(v.is_finite(), "softmax must be finite for large inputs");
        assert!((v - 1.0 / 3.0).abs() < 1e-5, "expected ~0.333, got {v}");
    }
}

#[test]
fn test_softmax_ref_large_negative() {
    // Numerical stability test: large negative values.
    let result = softmax_ref(&[-1000.0, -1000.0]).unwrap();
    for &v in &result {
        assert!(
            v.is_finite(),
            "softmax must be finite for large negative inputs"
        );
        assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
    }
}

#[test]
fn test_softmax_ref_monotonicity() {
    // Larger input → larger softmax output (when all else equal).
    let result = softmax_ref(&[1.0, 2.0, 3.0]).unwrap();
    assert!(result[0] < result[1], "softmax should be monotone");
    assert!(result[1] < result[2], "softmax should be monotone");
}

#[test]
fn test_softmax_ref_empty_rejected() {
    assert!(softmax_ref(&[]).is_err());
}

#[test]
fn test_softmax_ref_nan_rejected() {
    assert!(softmax_ref(&[1.0, f32::NAN, 3.0]).is_err());
}

#[test]
fn test_softmax_ref_inf_rejected() {
    assert!(softmax_ref(&[1.0, f32::INFINITY]).is_err());
    assert!(softmax_ref(&[f32::NEG_INFINITY, 1.0]).is_err());
}
