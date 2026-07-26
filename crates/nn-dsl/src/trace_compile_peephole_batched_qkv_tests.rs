// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for pass 12: batched Linear projection (QKV batching).
//!
//! Part of #3269.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::WeightRef;

use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use crate::trace_compile::{CompiledKernel, CompiledStep, NativeOpKind};

/// Build a fake Dispatch{linear} step with given weight dimensions.
fn make_linear_dispatch(
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    input_shape: Vec<usize>,
) -> CompiledStep {
    let mut output_shape = input_shape.clone();
    if let Some(last) = output_shape.last_mut() {
        *last = out_features;
    }

    let input_node = TensorNode::new(
        TensorNodeId::new(0),
        TensorOpKind::Input {
            name: "input".to_string(),
            shape: input_shape.clone(),
        },
        input_shape,
    );

    let weight_node = TensorNode::new(
        TensorNodeId::new(1),
        TensorOpKind::Input {
            name: "weight".to_string(),
            shape: vec![out_features, in_features],
        },
        vec![out_features, in_features],
    );

    let mut nodes = vec![input_node, weight_node];
    let mut bias_id = None;

    if has_bias {
        let bias_node = TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Input {
                name: "bias".to_string(),
                shape: vec![out_features],
            },
            vec![out_features],
        );
        nodes.push(bias_node);
        bias_id = Some(TensorNodeId::new(2));
    }

    let linear_id = TensorNodeId::new(nodes.len());
    let linear_node = TensorNode::new(
        linear_id,
        TensorOpKind::Linear {
            input: TensorNodeId::new(0),
            weight: TensorNodeId::new(1),
            bias: bias_id,
        },
        output_shape,
    );
    nodes.push(linear_node);

    let def = TensorKernelDef::new("linear", nodes, linear_id);
    let kernel = CompiledKernel::new(def);

    let mut weight_data = HashMap::new();
    let w = WeightRef::new(
        vec![0.0f32; out_features * in_features],
        vec![out_features, in_features],
    )
    .unwrap();
    weight_data.insert("weight".to_string(), w);
    if has_bias {
        let b = WeightRef::new(vec![0.0f32; out_features], vec![out_features]).unwrap();
        weight_data.insert("bias".to_string(), b);
    }

    CompiledStep::Dispatch {
        kernel,
        weight_data,
        external_node_ids: None,
    }
}

#[test]
fn test_batched_qkv_three_projections() {
    // Step 0: source, Step 1-3: Q/K/V Linears all consuming step 0.
    let mut steps = vec![
        CompiledStep::ConstantValue {
            value: 1.0,
            shape: vec![2, 4, 768],
        },
        make_linear_dispatch(768, 768, true, vec![2, 4, 768]),
        make_linear_dispatch(768, 768, true, vec![2, 4, 768]),
        make_linear_dispatch(768, 768, true, vec![2, 4, 768]),
    ];
    // Edge map: steps 1,2,3 all have step 0 as their primary input.
    let edge_map = vec![
        vec![],  // step 0: no inputs
        vec![0], // step 1: consumes step 0
        vec![0], // step 2: consumes step 0
        vec![0], // step 3: consumes step 0
    ];

    super::batch_with_edge_map(&mut steps, &edge_map);

    // Step 1 should be BatchedLinearProjection.
    match &steps[1] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::BatchedLinearProjection {
                    in_features,
                    total_out_features,
                    projection_sizes,
                    has_bias,
                    input_shape,
                },
            weight_data,
        } => {
            assert_eq!(*in_features, 768);
            assert_eq!(*total_out_features, 768 * 3);
            assert_eq!(projection_sizes, &[768, 768, 768]);
            assert!(*has_bias);
            assert_eq!(input_shape, &[2, 4, 768]);
            assert!(weight_data.contains_key("weight_t"));
            assert!(weight_data.contains_key("bias"));
            let wt = weight_data.get("weight_t").unwrap();
            assert_eq!(wt.shape(), &[768, 768 * 3]);
        }
        other => panic!("Expected BatchedLinearProjection, got: {other:?}"),
    }

    // Steps 2 and 3 should be ProjectionSlice.
    match &steps[2] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::ProjectionSlice {
                    source_step,
                    dim,
                    start,
                    length,
                    output_shape,
                },
            ..
        } => {
            assert_eq!(*source_step, 1);
            assert_eq!(*dim, 2);
            assert_eq!(*start, 768);
            assert_eq!(*length, 768);
            assert_eq!(output_shape, &[2, 4, 768]);
        }
        other => panic!("Expected ProjectionSlice at step 2, got: {other:?}"),
    }

    match &steps[3] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::ProjectionSlice {
                    source_step,
                    dim,
                    start,
                    length,
                    output_shape,
                },
            ..
        } => {
            assert_eq!(*source_step, 1);
            assert_eq!(*dim, 2);
            assert_eq!(*start, 768 * 2);
            assert_eq!(*length, 768);
            assert_eq!(output_shape, &[2, 4, 768]);
        }
        other => panic!("Expected ProjectionSlice at step 3, got: {other:?}"),
    }
}

#[test]
fn test_batched_qkv_different_out_features() {
    // Q: 768->768, K: 768->256, V: 768->256 (GQA pattern).
    let mut steps = vec![
        CompiledStep::ConstantValue {
            value: 1.0,
            shape: vec![2, 4, 768],
        },
        make_linear_dispatch(768, 768, false, vec![2, 4, 768]),
        make_linear_dispatch(768, 256, false, vec![2, 4, 768]),
        make_linear_dispatch(768, 256, false, vec![2, 4, 768]),
    ];
    let edge_map = vec![vec![], vec![0], vec![0], vec![0]];

    super::batch_with_edge_map(&mut steps, &edge_map);

    match &steps[1] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::BatchedLinearProjection {
                    total_out_features,
                    projection_sizes,
                    has_bias,
                    ..
                },
            ..
        } => {
            assert_eq!(*total_out_features, 1280);
            assert_eq!(projection_sizes, &[768, 256, 256]);
            assert!(!*has_bias);
        }
        other => panic!("Expected BatchedLinearProjection, got: {other:?}"),
    }

    match &steps[2] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::ProjectionSlice {
                    start,
                    length,
                    output_shape,
                    ..
                },
            ..
        } => {
            assert_eq!(*start, 768);
            assert_eq!(*length, 256);
            assert_eq!(output_shape, &[2, 4, 256]);
        }
        other => panic!("Expected ProjectionSlice at step 2, got: {other:?}"),
    }

    match &steps[3] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::ProjectionSlice {
                    start,
                    length,
                    output_shape,
                    ..
                },
            ..
        } => {
            assert_eq!(*start, 1024);
            assert_eq!(*length, 256);
            assert_eq!(output_shape, &[2, 4, 256]);
        }
        other => panic!("Expected ProjectionSlice at step 3, got: {other:?}"),
    }
}

#[test]
fn test_batched_qkv_mismatched_in_features_skipped() {
    let mut steps = vec![
        CompiledStep::ConstantValue {
            value: 1.0,
            shape: vec![2, 4, 768],
        },
        make_linear_dispatch(768, 768, false, vec![2, 4, 768]),
        make_linear_dispatch(512, 768, false, vec![2, 4, 512]),
    ];
    // Both claim step 0 as source, but in_features differ.
    let edge_map = vec![vec![], vec![0], vec![0]];

    super::batch_with_edge_map(&mut steps, &edge_map);

    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
    assert!(matches!(&steps[2], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_batched_qkv_single_linear_not_batched() {
    let mut steps = vec![
        CompiledStep::ConstantValue {
            value: 1.0,
            shape: vec![2, 4, 768],
        },
        make_linear_dispatch(768, 768, false, vec![2, 4, 768]),
    ];
    let edge_map = vec![vec![], vec![0]];

    super::batch_with_edge_map(&mut steps, &edge_map);

    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_batched_qkv_weight_transposition() {
    let mut steps = vec![
        CompiledStep::ConstantValue {
            value: 1.0,
            shape: vec![1, 4],
        },
        make_linear_dispatch(4, 3, false, vec![1, 4]),
        make_linear_dispatch(4, 2, false, vec![1, 4]),
    ];
    let edge_map = vec![vec![], vec![0], vec![0]];

    // Set distinguishable weight values.
    if let CompiledStep::Dispatch { weight_data, .. } = &mut steps[1] {
        let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        *weight_data.get_mut("weight").unwrap() = WeightRef::new(data, vec![3, 4]).unwrap();
    }
    if let CompiledStep::Dispatch { weight_data, .. } = &mut steps[2] {
        let data: Vec<f32> = (13..=20).map(|i| i as f32).collect();
        *weight_data.get_mut("weight").unwrap() = WeightRef::new(data, vec![2, 4]).unwrap();
    }

    super::batch_with_edge_map(&mut steps, &edge_map);

    match &steps[1] {
        CompiledStep::NativeOp { weight_data, .. } => {
            let wt = weight_data.get("weight_t").unwrap();
            assert_eq!(wt.shape(), &[4, 5]);
            let data = wt.data();
            // Concatenated [3,4]+[2,4] = [5,4]. Transposed to [4,5].
            // Row 0 of transposed = col 0 of original = [1, 5, 9, 13, 17]
            assert_eq!(data[0], 1.0);
            assert_eq!(data[1], 5.0);
            assert_eq!(data[2], 9.0);
            assert_eq!(data[3], 13.0);
            assert_eq!(data[4], 17.0);
        }
        _ => panic!("Expected NativeOp"),
    }
}
