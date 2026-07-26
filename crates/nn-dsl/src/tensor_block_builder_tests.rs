// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `TensorBlockBuilder` — extracted from inline module (#175 pattern).

use super::*;
use crate::adain::build_snake_scalar_kernel;

#[test]
fn test_builder_demucs_block() {
    let snake = build_snake_scalar_kernel().expect("snake kernel");
    let mut b = TensorBlockBuilder::new("demucs_enc");

    let data = b.add_input("data", &[1, 64]);
    let weight = b.add_input("weight", &[48, 1, 8]);
    let alpha = b.add_input("alpha", &[1]);
    let eps = b.add_input("eps", &[1]);

    let conv = b.add_conv1d(data, weight, None, 4, 2, &[48, 16]);
    let alpha_bc = b.add_broadcast(alpha, &[48, 16]);
    let act = b.add_elementwise(snake, &[conv, alpha_bc], &[48, 16]);
    let norm = b.add_instance_norm(act, eps, 1, None, None, &[48, 16]);

    let def = b.build(norm).expect("valid graph");
    assert_eq!(def.name, "demucs_enc");
    assert_eq!(def.nodes.len(), 8);
    assert_eq!(def.output, TensorNodeId::new(7));
    assert_eq!(def.nodes.last().unwrap().shape, vec![48, 16]);
}

#[test]
fn test_builder_conv1d_with_bias() {
    let mut b = TensorBlockBuilder::new("conv_bias");
    let data = b.add_input("data", &[2, 16]);
    let weight = b.add_input("weight", &[4, 2, 3]);
    let bias = b.add_input("bias", &[4]);

    let conv = b.add_conv1d(data, weight, Some(bias), 1, 0, &[4, 14]);
    let def = b.build(conv).expect("valid graph");

    assert_eq!(def.nodes.len(), 4);
    assert_eq!(def.output, TensorNodeId::new(3));
}

#[test]
fn test_builder_conv1d_dilated() {
    let mut b = TensorBlockBuilder::new("dilated");
    let data = b.add_input("data", &[1, 32]);
    let weight = b.add_input("weight", &[8, 1, 3]);

    let conv = b.add_conv1d_full(data, weight, None, 1, 0, 2, 1, &[8, 28]);
    let def = b.build(conv).expect("valid graph");

    assert_eq!(def.nodes.len(), 3);
    match &def.nodes[2].kind {
        TensorOpKind::Conv1d { dilation, .. } => assert_eq!(dilation, &2),
        _ => panic!("expected Conv1d"),
    }
}

#[test]
fn test_builder_conv_transpose_1d() {
    // ConvTranspose1d: upsampling with stride=4, padding=2.
    // Input [48, 16], weight [48, 1, 8] → output [1, 64] (Demucs decoder pattern).
    let mut b = TensorBlockBuilder::new("demucs_dec");
    let data = b.add_input("data", &[48, 16]);
    let weight = b.add_input("weight", &[48, 1, 8]);

    let deconv = b.add_conv_transpose_1d(data, weight, None, 4, 2, 1, 1, 0, &[1, 64]);
    let def = b.build(deconv).expect("valid graph");

    assert_eq!(def.nodes.len(), 3);
    assert_eq!(def.output, TensorNodeId::new(2));
    assert_eq!(def.nodes.last().unwrap().shape, vec![1, 64]);
    match &def.nodes[2].kind {
        TensorOpKind::ConvTranspose1d {
            stride, padding, ..
        } => {
            assert_eq!(*stride, 4);
            assert_eq!(*padding, 2);
        }
        other => panic!("expected ConvTranspose1d, got {other:?}"),
    }
}

#[test]
fn test_builder_conv_transpose_1d_with_bias() {
    let mut b = TensorBlockBuilder::new("deconv_bias");
    let data = b.add_input("data", &[4, 8]);
    let weight = b.add_input("weight", &[4, 2, 3]);
    let bias = b.add_input("bias", &[2]);

    let deconv = b.add_conv_transpose_1d(data, weight, Some(bias), 2, 0, 1, 1, 0, &[2, 17]);
    let def = b.build(deconv).expect("valid graph");

    assert_eq!(def.nodes.len(), 4);
    assert_eq!(def.output, TensorNodeId::new(3));
    match &def.nodes[3].kind {
        TensorOpKind::ConvTranspose1d { bias, .. } => {
            assert!(bias.is_some(), "bias should be present");
        }
        other => panic!("expected ConvTranspose1d, got {other:?}"),
    }
}

#[test]
fn test_builder_glu_decomposition() {
    // GLU: input [C=8, T=16] with axis=0 → output [C/2=4, T=16]
    // Decomposes into: narrow(data) + narrow(gate) + sigmoid(gate) + binary_mul
    let mut b = TensorBlockBuilder::new("glu_test");
    let x = b.add_input("x", &[8, 16]);
    let glu = b.add_glu(x, 0, &[8, 16]).expect("even dim");
    let def = b.build(glu).expect("valid graph");

    assert_eq!(def.name, "glu_test");
    // 1 input + 2 narrow + 1 sigmoid + 1 binary_mul = 5 nodes
    assert_eq!(def.nodes.len(), 5);
    // Output shape should be [4, 16] (halved along axis 0)
    assert_eq!(def.nodes.last().unwrap().shape, vec![4, 16]);

    // Verify node types: narrow, narrow, sigmoid, binary_mul
    assert!(matches!(
        &def.nodes[1].kind,
        TensorOpKind::Narrow {
            start: 0,
            length: 4,
            ..
        }
    ));
    assert!(matches!(
        &def.nodes[2].kind,
        TensorOpKind::Narrow {
            start: 4,
            length: 4,
            ..
        }
    ));
    assert!(matches!(&def.nodes[3].kind, TensorOpKind::Sigmoid { .. }));
    assert!(matches!(&def.nodes[4].kind, TensorOpKind::BinaryMul { .. }));
}

#[test]
fn test_builder_group_norm_g1_no_affine() {
    // GroupNorm(groups=1): decomposed into Reshape + Reduce/Broadcast/Elementwise + Reshape.
    // No InstanceNorm1d nodes — all primitives are directly dispatchable.
    let mut b = TensorBlockBuilder::new("gn1_test");
    let x = b.add_input("x", &[4, 8]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_group_norm_g1(x, eps, None, None, 4, 8);
    let def = b.build(out).expect("valid graph");

    // 2 inputs + reshape + 10 norm primitives + reshape = 14 nodes
    assert_eq!(def.nodes.len(), 14);
    assert_eq!(def.nodes.last().unwrap().shape, vec![4, 8]);

    // First internal node: reshape [4,8] → [1,32]
    assert!(matches!(
        &def.nodes[2].kind,
        TensorOpKind::Reshape { target_shape, .. } if target_shape == &[1, 32]
    ));
    // No InstanceNorm1d in the graph (AC1)
    assert!(
        !def.nodes
            .iter()
            .any(|n| matches!(n.kind, TensorOpKind::InstanceNorm1d { .. })),
        "GroupNorm g1 must not contain InstanceNorm1d nodes"
    );
    // Must contain Reduce ops (mean, variance)
    let reduce_count = def
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::Reduce { .. }))
        .count();
    assert_eq!(
        reduce_count, 2,
        "InstanceNorm decomposition needs 2 reductions"
    );
    // Last node: reshape [1,32] → [4,8]
    assert!(matches!(
        &def.nodes[13].kind,
        TensorOpKind::Reshape { target_shape, .. } if target_shape == &[4, 8]
    ));
}

#[test]
fn test_builder_group_norm_g1_affine() {
    // GroupNorm(1) with affine: decomposed norm + broadcast+mul + broadcast+add
    let mut b = TensorBlockBuilder::new("gn1_affine");
    let x = b.add_input("x", &[4, 8]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[4]);
    let beta = b.add_input("beta", &[4]);
    let out = b.add_group_norm_g1(x, eps, Some(gamma), Some(beta), 4, 8);
    let def = b.build(out).expect("valid graph");

    // 4 inputs + reshape + 10 norm + reshape + bc_gamma + mul + bc_beta + add = 20
    assert_eq!(def.nodes.len(), 20);
    assert_eq!(def.nodes.last().unwrap().shape, vec![4, 8]);

    // No InstanceNorm1d in the graph
    assert!(
        !def.nodes
            .iter()
            .any(|n| matches!(n.kind, TensorOpKind::InstanceNorm1d { .. })),
        "Affine GroupNorm must not contain InstanceNorm1d nodes"
    );
    // Final node should be BinaryAdd (beta addition)
    assert!(matches!(
        &def.nodes[19].kind,
        TensorOpKind::BinaryAdd { .. }
    ));
    // Preceding node should be Broadcast (beta broadcast)
    assert!(matches!(
        &def.nodes[18].kind,
        TensorOpKind::Broadcast { target_shape, .. } if target_shape == &[4, 8]
    ));
}

#[test]
fn test_builder_layer_scale() {
    // LayerScale: x * broadcast(scale), where scale is per-channel [C].
    // Used in DConv blocks (Demucs decoder).
    let mut b = TensorBlockBuilder::new("layer_scale");
    let x = b.add_input("x", &[48, 16]);
    let scale = b.add_input("scale", &[48]);

    let out = b.add_layer_scale(x, scale, &[48, 16]);
    let def = b.build(out).expect("valid graph");

    // 2 inputs + 1 broadcast + 1 binary_mul = 4 nodes
    assert_eq!(def.nodes.len(), 4);
    assert_eq!(def.nodes.last().unwrap().shape, vec![48, 16]);

    // Node 2: left-aligned broadcast [48] → [48, 16]
    assert!(matches!(
        &def.nodes[2].kind,
        TensorOpKind::Broadcast { target_shape, alignment, .. }
            if target_shape == &[48, 16] && *alignment == BroadcastAlignment::Left
    ));
    // Node 3: binary_mul(x, broadcast(scale))
    assert!(matches!(
        &def.nodes[3].kind,
        TensorOpKind::BinaryMul { left, right }
            if *left == TensorNodeId::new(0) && *right == TensorNodeId::new(2)
    ));
}

// ===========================================================================
// add_rms_norm tests (#740 AC1)
// ===========================================================================

#[test]
fn test_builder_rms_norm_basic() {
    let mut b = TensorBlockBuilder::new("rms_norm");
    let x = b.add_input("x", &[4, 128]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[128]);

    let norm = b.add_rms_norm(x, eps, 1, weight, &[4, 128]);
    let def = b.build(norm).expect("valid graph");

    // 3 inputs + 1 rms_norm = 4 nodes
    assert_eq!(def.nodes.len(), 4);
    assert_eq!(def.output, TensorNodeId::new(3));
    assert_eq!(def.nodes.last().unwrap().shape, vec![4, 128]);

    match &def.nodes[3].kind {
        TensorOpKind::RmsNorm {
            input,
            eps: eps_id,
            axis,
            weight: weight_id,
        } => {
            assert_eq!(*input, TensorNodeId::new(0));
            assert_eq!(*eps_id, TensorNodeId::new(1));
            assert_eq!(*axis, 1);
            assert_eq!(*weight_id, TensorNodeId::new(2));
        }
        other => panic!("expected RmsNorm, got {other:?}"),
    }
}

#[test]
fn test_builder_rms_norm_validates() {
    // RmsNorm followed by Linear — validates correctly as a composite graph.
    let mut b = TensorBlockBuilder::new("rms_linear");
    let x = b.add_input("x", &[4, 128]);
    let eps = b.add_input("eps", &[1]);
    let weight_norm = b.add_input("weight_norm", &[128]);
    let weight_linear = b.add_input("weight_linear", &[64, 128]);

    let norm = b.add_rms_norm(x, eps, 1, weight_norm, &[4, 128]);
    let linear = b.add_linear(norm, weight_linear, None, &[4, 64]);
    let def = b.build(linear).expect("valid graph");

    assert_eq!(def.nodes.len(), 6);
    assert!(def.validate().is_ok());
}

// ===========================================================================
// add_stack tests (#740 AC2)
// ===========================================================================

#[test]
fn test_builder_stack_two_inputs() {
    // Stack two [2, 3] tensors at axis=2 (new trailing dim) → [2, 3, 2]
    let mut b = TensorBlockBuilder::new("stack_test");
    let a = b.add_input("a", &[2, 3]);
    let c = b.add_input("b", &[2, 3]);

    let stacked = b.add_stack(&[a, c], 2, &[2, 3, 2]);
    let def = b.build(stacked).expect("valid graph");

    // 2 inputs + 1 stack = 3 nodes
    assert_eq!(def.nodes.len(), 3);
    assert_eq!(def.nodes.last().unwrap().shape, vec![2, 3, 2]);

    match &def.nodes[2].kind {
        TensorOpKind::Stack { inputs, axis } => {
            assert_eq!(inputs.len(), 2);
            assert_eq!(*axis, 2);
        }
        other => panic!("expected Stack, got {other:?}"),
    }
}

#[test]
fn test_builder_stack_rope_pattern() {
    // RoPE: stack 2 inputs of [BH, S, D/2] at axis=2 → [BH, S, D/2, 2]
    let mut b = TensorBlockBuilder::new("rope_stack");
    let cos_part = b.add_input("cos", &[2, 4, 3]);
    let sin_part = b.add_input("sin", &[2, 4, 3]);

    let stacked = b.add_stack(&[cos_part, sin_part], 3, &[2, 4, 3, 2]);
    let def = b.build(stacked).expect("valid graph");

    assert_eq!(def.nodes.len(), 3);
    assert_eq!(def.nodes.last().unwrap().shape, vec![2, 4, 3, 2]);

    match &def.nodes[2].kind {
        TensorOpKind::Stack { inputs, axis } => {
            assert_eq!(inputs.len(), 2);
            assert_eq!(*axis, 3, "stack at last position for RoPE pair");
        }
        other => panic!("expected Stack, got {other:?}"),
    }
}

// ===========================================================================
// add_axis_select tests (#740 AC3)
// ===========================================================================

#[test]
fn test_builder_axis_select_basic() {
    let mut b = TensorBlockBuilder::new("select_test");
    let x = b.add_input("x", &[2, 3, 4]);

    let selected = b.add_axis_select(x, 1, 0, &[2, 4]);
    let def = b.build(selected).expect("valid graph");

    // 1 input + 1 axis_select = 2 nodes
    assert_eq!(def.nodes.len(), 2);
    assert_eq!(def.nodes.last().unwrap().shape, vec![2, 4]);

    match &def.nodes[1].kind {
        TensorOpKind::AxisSelect { input, axis, index } => {
            assert_eq!(*input, TensorNodeId::new(0));
            assert_eq!(*axis, 1);
            assert_eq!(*index, 0);
        }
        other => panic!("expected AxisSelect, got {other:?}"),
    }
}

#[test]
fn test_builder_axis_select_rope_split() {
    // RoPE: select from [BH, S, D/2, 2] at axis=3 to get even/odd parts
    let mut b = TensorBlockBuilder::new("rope_split");
    let paired = b.add_input("paired", &[2, 4, 3, 2]);

    let even = b.add_axis_select(paired, 3, 0, &[2, 4, 3]);
    let odd = b.add_axis_select(paired, 3, 1, &[2, 4, 3]);
    let def = b.build(odd).expect("valid graph");

    // 1 input + 2 axis_select = 3 nodes
    assert_eq!(def.nodes.len(), 3);
    assert_eq!(def.nodes[1].shape, vec![2, 4, 3]);
    assert_eq!(def.nodes[2].shape, vec![2, 4, 3]);

    // even selects index 0
    match &def.nodes[1].kind {
        TensorOpKind::AxisSelect { index: 0, .. } => {}
        other => panic!("expected AxisSelect index=0, got {other:?}"),
    }
    // odd selects index 1
    match &def.nodes[2].kind {
        TensorOpKind::AxisSelect { index: 1, .. } => {}
        other => panic!("expected AxisSelect index=1, got {other:?}"),
    }
    let _ = even; // used in the graph via wiring, output is odd
}

/// AC3: `build()` rejects an invalid graph (dangling output node reference)
/// in all build profiles, not just debug. (#792)
#[test]
fn test_build_rejects_dangling_output_reference() {
    let mut b = TensorBlockBuilder::new("invalid");
    let _input = b.add_input("x", &[4]);
    // TensorNodeId::new(99) doesn't exist in the graph — build must return Err.
    let dangling = TensorNodeId::new(99);
    let result = b.build(dangling);
    assert!(result.is_err(), "build() must reject dangling output node");
}
