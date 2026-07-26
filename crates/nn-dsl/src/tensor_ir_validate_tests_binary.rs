// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Binary op (BinaryAdd, BinaryMul) validation tests.
//!
//! Extracted from `tensor_ir_validate_tests_structural.rs` per 500-line limit.

use super::*;
use crate::tensor_ir::TensorIRLayerError;

fn input_node(id: usize, name: &str, shape: Vec<usize>) -> TensorNode {
    TensorNode::new(
        TensorNodeId::new(id),
        TensorOpKind::Input {
            name: name.to_string(),
            shape: shape.clone(),
        },
        shape,
    )
}

// ===========================================================================
// BinaryAdd validation tests (#640)
// ===========================================================================

#[test]
fn test_binary_add_validates_same_shape() {
    let nodes = vec![
        input_node(0, "a", vec![4, 8]),
        input_node(1, "b", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::BinaryAdd {
                left: TensorNodeId::new(0),
                right: TensorNodeId::new(1),
            },
            vec![4, 8],
        ),
    ];
    let def = TensorKernelDef::new("add_ok", nodes, TensorNodeId::new(2));
    assert!(
        def.validate().is_ok(),
        "BinaryAdd with same shapes should pass"
    );
}

#[test]
fn test_binary_add_rejects_shape_mismatch() {
    let nodes = vec![
        input_node(0, "a", vec![4, 8]),
        input_node(1, "b", vec![3, 8]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::BinaryAdd {
                left: TensorNodeId::new(0),
                right: TensorNodeId::new(1),
            },
            vec![4, 8],
        ),
    ];
    let def = TensorKernelDef::new("add_bad", nodes, TensorNodeId::new(2));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::BinaryAddShapeMismatch { .. })
        ),
        "expected BinaryAddShapeMismatch, got: {err}"
    );
}

#[test]
fn test_binary_add_output_shape_is_left() {
    let nodes = vec![
        input_node(0, "a", vec![2, 3, 4]),
        input_node(1, "b", vec![2, 3, 4]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::BinaryAdd {
                left: TensorNodeId::new(0),
                right: TensorNodeId::new(1),
            },
            vec![2, 3, 4],
        ),
    ];
    let def = TensorKernelDef::new("add_shape", nodes, TensorNodeId::new(2));
    assert!(def.validate().is_ok());
    assert_eq!(def.nodes[2].shape, vec![2, 3, 4]);
}

#[test]
fn test_binary_add_rejects_forward_ref() {
    let nodes = vec![
        input_node(0, "a", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::BinaryAdd {
                left: TensorNodeId::new(0),
                right: TensorNodeId::new(2), // forward ref
            },
            vec![4, 8],
        ),
        input_node(2, "b", vec![4, 8]),
    ];
    let def = TensorKernelDef::new("add_fwd", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::ForwardRef(..)),
        "expected ForwardRef, got: {err}"
    );
}

// ===========================================================================
// BinaryMul validation tests (#643)
// ===========================================================================

#[test]
fn test_binary_mul_validates_same_shape() {
    let nodes = vec![
        input_node(0, "a", vec![4, 8]),
        input_node(1, "b", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::BinaryMul {
                left: TensorNodeId::new(0),
                right: TensorNodeId::new(1),
            },
            vec![4, 8],
        ),
    ];
    let def = TensorKernelDef::new("mul_ok", nodes, TensorNodeId::new(2));
    assert!(
        def.validate().is_ok(),
        "BinaryMul with same shapes should pass"
    );
}

#[test]
fn test_binary_mul_rejects_shape_mismatch() {
    let nodes = vec![
        input_node(0, "a", vec![4, 8]),
        input_node(1, "b", vec![3, 8]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::BinaryMul {
                left: TensorNodeId::new(0),
                right: TensorNodeId::new(1),
            },
            vec![4, 8],
        ),
    ];
    let def = TensorKernelDef::new("mul_bad", nodes, TensorNodeId::new(2));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::BinaryMulShapeMismatch { .. })
        ),
        "expected BinaryMulShapeMismatch, got: {err}"
    );
}

#[test]
fn test_binary_mul_output_shape_is_left() {
    let nodes = vec![
        input_node(0, "a", vec![2, 3, 4]),
        input_node(1, "b", vec![2, 3, 4]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::BinaryMul {
                left: TensorNodeId::new(0),
                right: TensorNodeId::new(1),
            },
            vec![2, 3, 4],
        ),
    ];
    let def = TensorKernelDef::new("mul_shape", nodes, TensorNodeId::new(2));
    assert!(def.validate().is_ok());
    assert_eq!(def.nodes[2].shape, vec![2, 3, 4]);
}

#[test]
fn test_binary_mul_rejects_forward_ref() {
    let nodes = vec![
        input_node(0, "a", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::BinaryMul {
                left: TensorNodeId::new(0),
                right: TensorNodeId::new(2), // forward ref
            },
            vec![4, 8],
        ),
        input_node(2, "b", vec![4, 8]),
    ];
    let def = TensorKernelDef::new("mul_fwd", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::ForwardRef(..)),
        "expected ForwardRef, got: {err}"
    );
}
