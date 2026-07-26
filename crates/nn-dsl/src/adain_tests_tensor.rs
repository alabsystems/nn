// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor-level builder tests for AdaIN/Snake kernels.
//!
//! Extracted from `adain_tests.rs` to stay within the 500-line file limit.
//! Tests the `build_snake_tensor` and `build_adain_snake_tensor` builders.

use super::super::*;

#[test]
fn test_snake_tensor_builds_and_validates() {
    let k1 = build_snake_tensor(8, 64).expect("build must succeed");
    k1.validate().expect("Snake K1 tensor IR must validate");
}

#[test]
fn test_snake_tensor_node_count() {
    let k1 = build_snake_tensor(8, 64).expect("build");
    // 4 nodes: x input, alpha input, broadcast, elementwise
    assert_eq!(k1.nodes.len(), 4, "Snake K1 should have 4 nodes");
}

#[test]
fn test_snake_tensor_output_shape() {
    let k1 = build_snake_tensor(8, 64).expect("build");
    let output = &k1.nodes[k1.output.index()];
    assert_eq!(output.shape, vec![8, 64], "output shape must be [C, T]");
}

#[test]
fn test_snake_tensor_zero_dim_returns_err() {
    assert!(build_snake_tensor(0, 64).is_err(), "zero channels");
    assert!(build_snake_tensor(8, 0).is_err(), "zero time");
}

#[test]
fn test_adain_snake_tensor_builds_and_validates() {
    let k4 = build_adain_snake_tensor(8, 64).expect("build must succeed");
    k4.validate()
        .expect("AdaIN+Snake K4 tensor IR must validate");
}

#[test]
fn test_adain_snake_tensor_node_count() {
    let k4 = build_adain_snake_tensor(8, 64).expect("build");
    // 8 nodes: x, eps, style_gamma, style_beta, alpha, AdaIN1d, broadcast, elementwise
    assert_eq!(k4.nodes.len(), 8, "AdaIN+Snake K4 should have 8 nodes");
}

#[test]
fn test_adain_snake_tensor_output_shape() {
    let k4 = build_adain_snake_tensor(8, 64).expect("build");
    let output = &k4.nodes[k4.output.index()];
    assert_eq!(output.shape, vec![8, 64], "output shape must be [C, T]");
}

#[test]
fn test_adain_snake_tensor_zero_dim_returns_err() {
    assert!(build_adain_snake_tensor(0, 64).is_err(), "zero channels");
    assert!(build_adain_snake_tensor(8, 0).is_err(), "zero time");
}

#[test]
fn test_adain_snake_tensor_has_adain1d_op() {
    use crate::tensor_ir::TensorOpKind;
    let k4 = build_adain_snake_tensor(8, 64).expect("build");
    let has_adain = k4
        .nodes
        .iter()
        .any(|n| matches!(n.kind, TensorOpKind::AdaIN1d { .. }));
    assert!(has_adain, "K4 must contain an AdaIN1d native op");
}
