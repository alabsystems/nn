// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for causal Conv1d builder.

use super::build_causal_conv1d;
use crate::tensor_ir::TensorOpKind;

#[test]
fn test_causal_conv1d_basic() {
    // dvoice pattern: kernel=3, dilation=1, stride=1
    // pad_left = (3-1)*1 = 2, padded = 64+2 = 66, eff_kernel = 3
    // out_length = (66 - 3) / 1 + 1 = 64 (same as input — causal preserves length)
    let def = build_causal_conv1d("causal_test", 8, 16, 3, 64, 1, 1, 1, false).unwrap();
    assert_eq!(def.nodes.len(), 4); // data, weight, ZeroPad1d, Conv1d
    let output_node = &def.nodes[def.nodes.len() - 1];
    assert_eq!(output_node.shape, vec![16, 64]);

    // Check ZeroPad1d node
    match &def.nodes[2].kind {
        TensorOpKind::ZeroPad1d {
            pad_left,
            pad_right,
            ..
        } => {
            assert_eq!(*pad_left, 2);
            assert_eq!(*pad_right, 0);
        }
        other => panic!("Expected ZeroPad1d, got {other:?}"),
    }

    // Check Conv1d has padding=0
    match &def.nodes[3].kind {
        TensorOpKind::Conv1d { padding, .. } => assert_eq!(*padding, 0),
        other => panic!("Expected Conv1d, got {other:?}"),
    }
}

#[test]
fn test_causal_conv1d_dilated() {
    // dvoice pattern: kernel=3, dilation=3, stride=1
    // pad_left = (3-1)*3 = 6, padded = 64+6 = 70
    // eff_kernel = 6+1 = 7, out = (70-7)/1+1 = 64 (same length)
    let def = build_causal_conv1d("causal_dil", 8, 16, 3, 64, 1, 3, 1, false).unwrap();
    let output_node = &def.nodes[def.nodes.len() - 1];
    assert_eq!(output_node.shape, vec![16, 64]);

    match &def.nodes[2].kind {
        TensorOpKind::ZeroPad1d { pad_left, .. } => assert_eq!(*pad_left, 6),
        other => panic!("Expected ZeroPad1d, got {other:?}"),
    }
}

#[test]
fn test_causal_conv1d_with_bias() {
    let def = build_causal_conv1d("causal_bias", 8, 16, 3, 64, 1, 1, 1, true).unwrap();
    assert_eq!(def.nodes.len(), 5); // data, weight, bias, ZeroPad1d, Conv1d
    assert_eq!(def.nodes[2].shape, vec![16]); // bias shape

    // ZeroPad1d is node 3
    match &def.nodes[3].kind {
        TensorOpKind::ZeroPad1d { .. } => {}
        other => panic!("Expected ZeroPad1d, got {other:?}"),
    }
}

#[test]
fn test_causal_conv1d_stride2() {
    // kernel=3, dilation=1, stride=2
    // pad_left = 2, padded = 66, eff_kernel = 3
    // out = (66 - 3) / 2 + 1 = 32
    let def = build_causal_conv1d("causal_s2", 8, 16, 3, 64, 2, 1, 1, false).unwrap();
    let output_node = &def.nodes[def.nodes.len() - 1];
    assert_eq!(output_node.shape, vec![16, 32]);
}

#[test]
fn test_causal_conv1d_zero_stride() {
    let err = build_causal_conv1d("bad", 8, 16, 3, 64, 0, 1, 1, false).unwrap_err();
    assert!(format!("{err}").contains("stride must be >= 1"));
}

#[test]
fn test_causal_conv1d_zero_kernel() {
    let err = build_causal_conv1d("bad", 8, 16, 0, 64, 1, 1, 1, false).unwrap_err();
    assert!(format!("{err}").contains("kernel_size must be >= 1"));
}

#[test]
fn test_causal_conv1d_validates() {
    // Validation should pass for the constructed graph
    let def = build_causal_conv1d("validate_test", 8, 16, 3, 64, 1, 1, 1, false).unwrap();
    assert!(def.validate().is_ok());
}
