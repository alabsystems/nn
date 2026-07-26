// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `TensorOpKind::remap_ids`.
//!
//! Part of #735 AC1: ≥4 unit tests covering identity, sequential, gap, and
//! topology-preserving remaps.

use super::*;
use std::collections::HashMap;

/// Identity remap: {0→0, 1→1, 2→2} leaves all node refs unchanged.
#[test]
fn test_remap_identity() {
    let op = TensorOpKind::Elementwise {
        kernel: crate::test_kernels::identity_kernel(),
        inputs: vec![TensorNodeId::new(0)],
    };
    let map: HashMap<usize, usize> = [(0, 0), (1, 1), (2, 2)].into_iter().collect();
    let remapped = op.remap_ids(&map);
    match remapped {
        TensorOpKind::Elementwise { inputs, .. } => {
            assert_eq!(inputs, vec![TensorNodeId::new(0)]);
        }
        other => panic!("expected Elementwise, got {other:?}"),
    }
}

/// Sequential offset remap: shift all ids by +10.
#[test]
fn test_remap_sequential_offset() {
    let op = TensorOpKind::BinaryAdd {
        left: TensorNodeId::new(0),
        right: TensorNodeId::new(1),
    };
    let map: HashMap<usize, usize> = [(0, 10), (1, 11)].into_iter().collect();
    let remapped = op.remap_ids(&map);
    match remapped {
        TensorOpKind::BinaryAdd { left, right } => {
            assert_eq!(left, TensorNodeId::new(10));
            assert_eq!(right, TensorNodeId::new(11));
        }
        other => panic!("expected BinaryAdd, got {other:?}"),
    }
}

/// Gap remap: non-contiguous mapping {0→5, 3→8, 7→12}.
#[test]
fn test_remap_with_gaps() {
    let op = TensorOpKind::Reduce {
        op: ReduceOp::Sum,
        input: TensorNodeId::new(3),
        axis: 1,
        keepdim: false,
    };
    let map: HashMap<usize, usize> = [(0, 5), (3, 8), (7, 12)].into_iter().collect();
    let remapped = op.remap_ids(&map);
    match remapped {
        TensorOpKind::Reduce { input, axis, .. } => {
            assert_eq!(input, TensorNodeId::new(8));
            assert_eq!(axis, 1, "axis must be preserved, not remapped");
        }
        other => panic!("expected Reduce, got {other:?}"),
    }
}

/// Topology preservation: Conv1d with input, weight, and optional bias all remap
/// correctly while scalar fields (stride, padding, dilation, groups) are preserved.
#[test]
fn test_remap_preserves_conv1d_topology() {
    let op = TensorOpKind::Conv1d {
        input: TensorNodeId::new(0),
        weight: TensorNodeId::new(1),
        bias: Some(TensorNodeId::new(2)),
        stride: 4,
        padding: 2,
        dilation: 1,
        groups: 1,
    };
    let map: HashMap<usize, usize> = [(0, 100), (1, 101), (2, 102)].into_iter().collect();
    let remapped = op.remap_ids(&map);
    match remapped {
        TensorOpKind::Conv1d {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        } => {
            assert_eq!(input, TensorNodeId::new(100));
            assert_eq!(weight, TensorNodeId::new(101));
            assert_eq!(bias, Some(TensorNodeId::new(102)));
            assert_eq!(stride, 4, "stride must be preserved");
            assert_eq!(padding, 2, "padding must be preserved");
            assert_eq!(dilation, 1, "dilation must be preserved");
            assert_eq!(groups, 1, "groups must be preserved");
        }
        other => panic!("expected Conv1d, got {other:?}"),
    }
}

/// Input nodes have no node-id references to remap — only name and shape.
#[test]
fn test_remap_input_preserves_name_and_shape() {
    let op = TensorOpKind::Input {
        name: "x".to_string(),
        shape: vec![4, 128],
    };
    let map: HashMap<usize, usize> = [(0, 99)].into_iter().collect();
    let remapped = op.remap_ids(&map);
    match remapped {
        TensorOpKind::Input { name, shape } => {
            assert_eq!(name, "x");
            assert_eq!(shape, vec![4, 128]);
        }
        other => panic!("expected Input, got {other:?}"),
    }
}

/// Optional bias=None in Conv1d stays None after remap.
#[test]
fn test_remap_conv1d_no_bias_stays_none() {
    let op = TensorOpKind::Conv1d {
        input: TensorNodeId::new(0),
        weight: TensorNodeId::new(1),
        bias: None,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    };
    let map: HashMap<usize, usize> = [(0, 10), (1, 11)].into_iter().collect();
    let remapped = op.remap_ids(&map);
    match remapped {
        TensorOpKind::Conv1d { bias, .. } => {
            assert_eq!(bias, None, "bias=None must stay None after remap");
        }
        other => panic!("expected Conv1d, got {other:?}"),
    }
}

/// Stack remap: all inputs in the Vec are remapped.
#[test]
fn test_remap_stack_all_inputs() {
    let op = TensorOpKind::Stack {
        inputs: vec![
            TensorNodeId::new(0),
            TensorNodeId::new(1),
            TensorNodeId::new(2),
        ],
        axis: 0,
    };
    let map: HashMap<usize, usize> = [(0, 20), (1, 21), (2, 22)].into_iter().collect();
    let remapped = op.remap_ids(&map);
    match remapped {
        TensorOpKind::Stack { inputs, axis } => {
            assert_eq!(
                inputs,
                vec![
                    TensorNodeId::new(20),
                    TensorNodeId::new(21),
                    TensorNodeId::new(22)
                ]
            );
            assert_eq!(axis, 0, "axis must be preserved");
        }
        other => panic!("expected Stack, got {other:?}"),
    }
}
