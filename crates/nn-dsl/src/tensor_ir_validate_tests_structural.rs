// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural operation validation tests: elementwise, reshape, axis_select, stack.
//!
//! Split from `tensor_ir_validate_tests.rs` per #631.

use super::*;

// ===========================================================================
// Elementwise validation tests
// ===========================================================================

#[test]
fn test_output_shape_elementwise_preserves_shape() {
    // identity kernel on [4, 8] → output [4, 8]
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Elementwise {
                kernel: identity_kernel(),
                inputs: vec![TensorNodeId::new(0)],
            },
            vec![4, 8],
        ),
    ];
    let def = TensorKernelDef::new("ew_shape", nodes, TensorNodeId::new(1));
    assert!(def.validate().is_ok(), "elementwise should preserve shape");
}

#[test]
fn test_elementwise_shape_mismatch() {
    // add kernel: inputs [4, 8] and [4, 16] → shape mismatch
    let nodes = vec![
        input_node(0, "a", vec![4, 8]),
        input_node(1, "b", vec![4, 16]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Elementwise {
                kernel: add_kernel(),
                inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
            },
            vec![4, 8],
        ),
    ];
    let def = TensorKernelDef::new("ew_mismatch", nodes, TensorNodeId::new(2));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::ElementwiseShapeMismatch { index: 1, .. }
        ),
        "expected ElementwiseShapeMismatch, got: {err}"
    );
}

#[test]
fn test_elementwise_param_mismatch() {
    // identity kernel (1 param) with 2 inputs
    let nodes = vec![
        input_node(0, "a", vec![4]),
        input_node(1, "b", vec![4]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Elementwise {
                kernel: identity_kernel(),
                inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
            },
            vec![4],
        ),
    ];
    let def = TensorKernelDef::new("ew_param", nodes, TensorNodeId::new(2));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::ElementwiseParamMismatch {
                expected: 1,
                got: 2
            }
        ),
        "expected ElementwiseParamMismatch, got: {err}"
    );
}

// ===========================================================================
// Reshape validation tests
// ===========================================================================

#[test]
fn test_output_shape_reshape() {
    // [4, 8] reshape to [32]
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reshape {
                input: TensorNodeId::new(0),
                target_shape: vec![32],
            },
            vec![32],
        ),
    ];
    let def = TensorKernelDef::new("reshape_ok", nodes, TensorNodeId::new(1));
    assert!(def.validate().is_ok(), "reshape [4,8] -> [32] should pass");
}

#[test]
fn test_reshape_product_mismatch() {
    // [4, 8]=32 elements reshape to [5, 5]=25 elements — mismatch
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Reshape {
                input: TensorNodeId::new(0),
                target_shape: vec![5, 5],
            },
            vec![5, 5],
        ),
    ];
    let def = TensorKernelDef::new("reshape_bad", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::ReshapeProductMismatch {
                input_product: 32,
                target_product: 25
            }
        ),
        "expected ReshapeProductMismatch, got: {err}"
    );
}

// ===========================================================================
// AxisSelect validation tests
// ===========================================================================

#[test]
fn test_output_shape_axis_select() {
    // input [4, 3, 8], axis_select axis=1 index=0 → [4, 8]
    let nodes = vec![
        input_node(0, "x", vec![4, 3, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::AxisSelect {
                input: TensorNodeId::new(0),
                axis: 1,
                index: 0,
            },
            vec![4, 8],
        ),
    ];
    let def = TensorKernelDef::new("axis_sel", nodes, TensorNodeId::new(1));
    assert!(def.validate().is_ok(), "axis_select should remove axis dim");
}

#[test]
fn test_axis_select_axis_zero_reserved() {
    // axis 0 is reserved in tensor verification paths.
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::AxisSelect {
                input: TensorNodeId::new(0),
                axis: 0,
                index: 0,
            },
            vec![8],
        ),
    ];
    let def = TensorKernelDef::new("axis0", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::AxisZeroReserved { op: "axis_select" }),
        "expected AxisZeroReserved, got: {err}"
    );
}

#[test]
fn test_axis_select_index_out_of_bounds() {
    // input [4, 3], axis=1, index=5 — index >= dim size (3)
    let nodes = vec![
        input_node(0, "x", vec![4, 3]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::AxisSelect {
                input: TensorNodeId::new(0),
                axis: 1,
                index: 5,
            },
            vec![4],
        ),
    ];
    let def = TensorKernelDef::new("axis_sel_oob", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::AxisSelectIndexOutOfBounds {
                index: 5,
                dim: 3,
                axis: 1
            }
        ),
        "expected AxisSelectIndexOutOfBounds, got: {err}"
    );
}

// ===========================================================================
// Stack validation tests
// ===========================================================================

#[test]
fn test_output_shape_stack() {
    // Stack two [4, 8] tensors at axis=1 → [4, 2, 8]
    let nodes = vec![
        input_node(0, "a", vec![4, 8]),
        input_node(1, "b", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Stack {
                inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                axis: 1,
            },
            vec![4, 2, 8],
        ),
    ];
    let def = TensorKernelDef::new("stack_ok", nodes, TensorNodeId::new(2));
    assert!(
        def.validate().is_ok(),
        "stack at axis=1 should produce [4,2,8]"
    );
}

#[test]
fn test_stack_shape_mismatch() {
    // Stack [4, 8] and [4, 16] — different shapes
    let nodes = vec![
        input_node(0, "a", vec![4, 8]),
        input_node(1, "b", vec![4, 16]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Stack {
                inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                axis: 1,
            },
            vec![4, 2, 8],
        ),
    ];
    let def = TensorKernelDef::new("stack_mismatch", nodes, TensorNodeId::new(2));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::StackShapeMismatch { .. }),
        "expected StackShapeMismatch, got: {err}"
    );
}

#[test]
fn test_stack_axis_zero_allowed() {
    // Stack axis 0 is allowed: NY UnsqueezeLayer and ConcatLayer
    // both accept axis 0, and axis_offset handles multi-variable correctly.
    let nodes = vec![
        input_node(0, "a", vec![4]),
        input_node(1, "b", vec![4]),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Stack {
                inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                axis: 0,
            },
            vec![2, 4],
        ),
    ];
    let def = TensorKernelDef::new("stack_axis0", nodes, TensorNodeId::new(2));
    assert!(
        def.validate().is_ok(),
        "axis-0 stack should validate: {:?}",
        def.validate()
    );
}

#[test]
fn test_stack_empty_inputs() {
    let nodes = vec![
        input_node(0, "dummy", vec![4]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Stack {
                inputs: vec![],
                axis: 1,
            },
            vec![4],
        ),
    ];
    let def = TensorKernelDef::new("stack_empty", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::EmptyStack),
        "expected EmptyStack, got: {err}"
    );
}

// ===========================================================================
// Shape consistency test
// ===========================================================================

#[test]
fn test_node_shape_consistency_enforced() {
    // Node declares shape [4, 99] but compute_output_shape would produce [4, 8].
    // The validator should catch the mismatch.
    let nodes = vec![
        input_node(0, "x", vec![4, 8]),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Elementwise {
                kernel: identity_kernel(),
                inputs: vec![TensorNodeId::new(0)],
            },
            vec![4, 99], // wrong: should be [4, 8]
        ),
    ];
    let def = TensorKernelDef::new("shape_mismatch", nodes, TensorNodeId::new(1));
    let err = def.validate().unwrap_err();
    assert!(
        matches!(err, TensorIRError::IncompatibleBroadcast { .. }),
        "expected shape consistency error, got: {err}"
    );
}

// Binary op tests (BinaryAdd, BinaryMul) extracted to tensor_ir_validate_tests_binary.rs
// to stay under the 500-line limit.
#[path = "tensor_ir_validate_tests_binary.rs"]
mod binary;
