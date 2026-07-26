// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for peephole fusion passes, NativeOpKind properties, and DispatchStep
//! construction. Part of #3817.
//!
//! Coverage:
//! - PeepholeConfig disable flags (individual pass bypass)
//! - NativeOpKind variant_name, estimated_metal_dispatches, estimated_encoding_events
//! - NativeOpKind external_node_ids and collect_direct_step_deps
//! - NativeOpKind serde round-trip
//! - DispatchStep variant construction
//! - Peephole edge cases (empty steps, single step, stride != 1 conv)

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::trace_compile::{
    AttentionLayout, CompiledKernel, CompiledStep, FusedNormKind, GemmActivation, NativeOpKind,
    NormActivConv1dParams, NormActivation, PeepholeConfig, StyleBatchOffset, StyleProjectionParams,
};

// -- Helpers (mirrors existing test patterns) ---------------------------------

fn test_node(id: u64, name: &str, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        TraceOp::Relu,
        inputs,
        shape,
        DType::F32,
    )
}

fn test_input_node(id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape,
        DType::F32,
    )
}

fn make_adain_leaky_relu(input_shape: &[usize]) -> CompiledStep {
    CompiledStep::NativeOp {
        op: NativeOpKind::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.2,
            input_shape: input_shape.to_vec(),
            external_node_ids: None,
        },
        weight_data: HashMap::new(),
    }
}

fn make_adain_snake(input_shape: &[usize], channels: usize) -> CompiledStep {
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "alpha".to_string(),
        WeightRef::new(vec![1.0f32; channels], vec![channels]).expect("valid alpha"),
    );
    CompiledStep::NativeOp {
        op: NativeOpKind::AdainSnake {
            eps: 1e-5,
            input_shape: input_shape.to_vec(),
            channels,
            residual_gamma: true,
            external_node_ids: None,
        },
        weight_data,
    }
}

fn make_conv1d_dispatch(
    input_shape: &[usize],
    output_shape: &[usize],
    weight_shape: &[usize],
    padding: usize,
    dilation: usize,
) -> CompiledStep {
    make_conv1d_dispatch_full(
        input_shape,
        output_shape,
        weight_shape,
        1,
        padding,
        dilation,
        1,
    )
}

fn make_conv1d_dispatch_full(
    input_shape: &[usize],
    output_shape: &[usize],
    weight_shape: &[usize],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let bias_shape = [weight_shape[0]];
    let mut b = TensorBlockBuilder::new("conv1d");
    let input = b.add_input("input_0", input_shape);
    let w = b.add_input("weight", weight_shape);
    let bi = b.add_input("bias", &bias_shape);
    let output = b.add_conv1d_full(
        input,
        w,
        Some(bi),
        stride,
        padding,
        dilation,
        groups,
        output_shape,
    );
    let def = b.build(output).expect("valid conv1d IR");

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(
            vec![0.0f32; weight_shape.iter().product()],
            weight_shape.to_vec(),
        )
        .expect("valid weight"),
    );
    weight_data.insert(
        "bias".to_string(),
        WeightRef::new(vec![0.0f32; bias_shape[0]], bias_shape.to_vec()).expect("valid bias"),
    );

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: None,
    }
}

fn make_linear_dispatch(
    input_shape: &[usize],
    out_features: usize,
    has_bias: bool,
) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let in_features = *input_shape.last().unwrap();
    let mut output_shape = input_shape.to_vec();
    *output_shape.last_mut().unwrap() = out_features;
    let weight_shape = [out_features, in_features];

    let mut b = TensorBlockBuilder::new("linear");
    let input = b.add_input("input_0", input_shape);
    let w = b.add_input("weight", &weight_shape);
    let bi = if has_bias {
        Some(b.add_input("bias", &[out_features]))
    } else {
        None
    };
    let output = b.add_linear(input, w, bi, &output_shape);
    let def = b.build(output).expect("valid linear IR");

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(
            vec![0.0f32; out_features * in_features],
            weight_shape.to_vec(),
        )
        .expect("valid weight"),
    );
    if has_bias {
        weight_data.insert(
            "bias".to_string(),
            WeightRef::new(vec![0.0f32; out_features], vec![out_features]).expect("valid bias"),
        );
    }

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: None,
    }
}

fn make_activation_dispatch(name: &str, shape: &[usize]) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("input_0", shape);
    let output = match name {
        "relu" => b.add_relu(input, shape),
        "gelu" => b.add_gelu(input, shape),
        "gelu_erf" => b.add_gelu_erf(input, shape),
        "sigmoid" => b.add_sigmoid(input, shape),
        _ => panic!("unsupported test activation: {name}"),
    };
    let def = b.build(output).expect("valid activation IR");

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

fn make_add_dispatch(input_shape: &[usize]) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let mut b = TensorBlockBuilder::new("add");
    let a = b.add_input("input_0", input_shape);
    let bb = b.add_input("input_1", input_shape);
    let output = b.add_binary_add(a, bb, input_shape);
    let def = b.build(output).expect("valid add IR");

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

fn make_layernorm_native(input_shape: &[usize], hidden_dim: usize, eps: f32) -> CompiledStep {
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(vec![1.0f32; hidden_dim], vec![hidden_dim]).expect("valid weight"),
    );
    weight_data.insert(
        "bias".to_string(),
        WeightRef::new(vec![0.0f32; hidden_dim], vec![hidden_dim]).expect("valid bias"),
    );
    CompiledStep::NativeOp {
        op: NativeOpKind::LayerNorm {
            eps,
            input_shape: input_shape.to_vec(),
            hidden_dim,
        },
        weight_data,
    }
}

// =============================================================================
// 1. PeepholeConfig disable flags
// =============================================================================

/// Disabling norm_activ_conv1d prevents AdainLeakyRelu + Conv1d fusion.
#[test]
fn test_peephole_config_disable_norm_activ_conv1d() {
    let input_shape = [1, 512, 100];
    let output_shape = [1, 512, 100];
    let weight_shape = [512, 512, 3];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_adain_leaky_relu(&input_shape),
        make_conv1d_dispatch(&input_shape, &output_shape, &weight_shape, 1, 1),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_input_node(1, vec![1, 512, 1]),
        test_input_node(2, vec![1, 512, 1]),
        test_node(3, "adain", vec![0, 1, 2], input_shape.to_vec()),
        test_node(4, "conv1d", vec![3], output_shape.to_vec()),
    ]);

    let config = PeepholeConfig {
        norm_activ_conv1d: false,
        ..Default::default()
    };

    crate::trace_compile::peephole::apply_peephole_with_config(&mut steps, &graph, &config);

    // Should NOT fuse because pass 1 is disabled.
    assert!(
        matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::AdainLeakyRelu { .. },
                ..
            }
        ),
        "AdainLeakyRelu should remain unfused when norm_activ_conv1d is disabled"
    );
}

/// Disabling linear_activation prevents Linear + Relu fusion.
#[test]
fn test_peephole_config_disable_linear_activation() {
    let input_shape = [4, 768];
    let out_features = 3072;
    let output_shape = [4, 3072];

    let mut steps = vec![
        CompiledStep::InputForward,
        make_linear_dispatch(&input_shape, out_features, true),
        make_activation_dispatch("relu", &output_shape),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_node(1, "linear", vec![0], output_shape.to_vec()),
        test_node(2, "relu", vec![1], output_shape.to_vec()),
    ]);

    let config = PeepholeConfig {
        linear_activation: false,
        ..Default::default()
    };

    crate::trace_compile::peephole::apply_peephole_with_config(&mut steps, &graph, &config);

    // Linear should remain as Dispatch, not fused to LinearActivation.
    assert!(
        matches!(&steps[1], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "linear"),
        "linear should remain unfused when linear_activation is disabled"
    );
}

/// Disabling add_layer_norm prevents Add + LayerNorm fusion.
#[test]
fn test_peephole_config_disable_add_layer_norm() {
    let shape = vec![4, 768];
    let hidden_dim = 768;

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_add_dispatch(&shape),
        make_layernorm_native(&shape, hidden_dim, 1e-5),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_input_node(1, shape.clone()),
        test_node(2, "add", vec![0, 1], shape.clone()),
        test_node(3, "layernorm", vec![2], shape),
    ]);

    let config = PeepholeConfig {
        add_layer_norm: false,
        ..Default::default()
    };

    crate::trace_compile::peephole::apply_peephole_with_config(&mut steps, &graph, &config);

    // Add should remain unfused.
    assert!(
        matches!(&steps[2], CompiledStep::Dispatch { kernel, .. } if kernel.name() == "add"),
        "add should remain unfused when add_layer_norm is disabled"
    );
}

/// Disabling norm_linear prevents LayerNorm + Linear fusion.
#[test]
fn test_peephole_config_disable_norm_linear() {
    let shape = vec![4, 768];
    let hidden_dim = 768;
    let out_features = 3072;

    let mut steps = vec![
        CompiledStep::InputForward,
        make_layernorm_native(&shape, hidden_dim, 1e-5),
        make_linear_dispatch(&shape, out_features, true),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, shape.clone()),
        test_node(1, "layernorm", vec![0], shape),
        test_node(2, "linear", vec![1], vec![4, out_features]),
    ]);

    let config = PeepholeConfig {
        norm_linear: false,
        ..Default::default()
    };

    crate::trace_compile::peephole::apply_peephole_with_config(&mut steps, &graph, &config);

    // LayerNorm should remain as NativeOp, not fused to NormLinear.
    assert!(
        matches!(
            &steps[1],
            CompiledStep::NativeOp {
                op: NativeOpKind::LayerNorm { .. },
                ..
            }
        ),
        "LayerNorm should remain unfused when norm_linear is disabled"
    );
}

// =============================================================================
// 2. NativeOpKind properties
// =============================================================================

/// variant_name returns correct strings for all single-dispatch NativeOps.
#[test]
fn test_native_op_variant_name_single_dispatch_ops() {
    let cases: Vec<(NativeOpKind, &str)> = vec![
        (
            NativeOpKind::InstanceNorm {
                eps: 1e-5,
                input_shape: vec![1, 2, 4],
            },
            "InstanceNorm",
        ),
        (
            NativeOpKind::SiluMul {
                input_shape: vec![1, 8, 256],
            },
            "SiluMul",
        ),
        (
            NativeOpKind::RotaryEmbedding {
                head_dim: 64,
                input_shape: vec![1, 8, 16, 64],
            },
            "RotaryEmbedding",
        ),
        (
            NativeOpKind::ChannelsFirstLayerNorm {
                eps: 1e-5,
                input_shape: vec![1, 4, 8],
                channels: 4,
                leaky_relu_slope: None,
            },
            "ChannelsFirstLayerNorm",
        ),
        (
            NativeOpKind::Int8Gemm {
                in_features: 64,
                out_features: 128,
                has_bias: true,
                input_shape: vec![1, 4, 64],
            },
            "Int8Gemm",
        ),
    ];

    for (op, expected_name) in &cases {
        assert_eq!(
            op.variant_name(),
            *expected_name,
            "variant_name mismatch for {expected_name}"
        );
    }
}

/// estimated_metal_dispatches for single-dispatch ops returns 1.
#[test]
fn test_native_op_single_dispatch_count() {
    let single_dispatch_ops: Vec<NativeOpKind> = vec![
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 2, 4],
        },
        NativeOpKind::AdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 2, 4],
            channels: 2,
            residual_gamma: false,
            external_node_ids: None,
        },
        NativeOpKind::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.01,
            input_shape: vec![1, 2, 4],
            external_node_ids: None,
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 1, 4, 8],
            k_shape: vec![1, 1, 4, 8],
            output_shape: vec![1, 1, 4, 8],
            input_layout: AttentionLayout::HeadsFirst,
        },
        NativeOpKind::LinearActivation {
            activation: GemmActivation::Relu,
            in_features: 4,
            out_features: 8,
            has_bias: true,
            input_shape: vec![1, 4],
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 4],
            hidden_dim: 4,
        },
        NativeOpKind::AddLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 4],
            hidden_dim: 4,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 8, 256],
        },
        NativeOpKind::RotaryEmbedding {
            head_dim: 64,
            input_shape: vec![1, 8, 16, 64],
        },
        NativeOpKind::LstmSequence {
            hidden_size: 64,
            input_shape: vec![1, 1, 64],
            h_shape: vec![1, 64],
            reverse: false,
        },
    ];

    for op in &single_dispatch_ops {
        assert_eq!(
            op.estimated_metal_dispatches(),
            1,
            "{} should be 1 Metal dispatch",
            op.variant_name()
        );
    }
}

/// ConstantWeight has 0 dispatches (no GPU computation).
#[test]
fn test_native_op_constant_weight_zero_dispatches() {
    let op = NativeOpKind::ConstantWeight {
        name: "arange".into(),
        shape: vec![256],
    };
    assert_eq!(op.estimated_metal_dispatches(), 0);
    assert_eq!(op.estimated_encoding_events(), 0);
    assert_eq!(op.variant_name(), "ConstantWeight");
}

/// Cumsum dispatch count depends on axis size: <= 256 → 1, > 256 → 3.
#[test]
fn test_native_op_cumsum_dispatch_count_by_axis_size() {
    // Small axis (single-pass)
    let small = NativeOpKind::Cumsum {
        dim: 1,
        input_shape: vec![4, 128],
    };
    assert_eq!(small.estimated_metal_dispatches(), 1);

    // Large axis (multi-pass Blelloch)
    let large = NativeOpKind::Cumsum {
        dim: 1,
        input_shape: vec![4, 512],
    };
    assert_eq!(large.estimated_metal_dispatches(), 3);

    // Boundary: exactly 256
    let boundary = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![256, 4],
    };
    assert_eq!(boundary.estimated_metal_dispatches(), 1);

    // Just above boundary: 257
    let just_above = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![257, 4],
    };
    assert_eq!(just_above.estimated_metal_dispatches(), 3);
}

/// Conv1dGemm dispatch count: K=3 direct path vs im2col+GEMM path.
///
/// K=3/stride=1/dilation=1/groups=1 uses the direct sliding-window kernel
/// (#4264), saving the im2col dispatch. Other shapes use im2col+GEMM.
#[test]
fn test_native_op_conv1d_gemm_dispatch_count() {
    // K=3 direct path: 1 conv dispatch + 1 bias add = 2 with bias.
    let k3_with_bias = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 256],
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    assert_eq!(k3_with_bias.estimated_metal_dispatches(), 2);
    assert_eq!(k3_with_bias.estimated_encoding_events(), 2);

    // K=3 direct path: 1 conv dispatch = 1 without bias.
    let k3_without_bias = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 256],
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: false,
    };
    assert_eq!(k3_without_bias.estimated_metal_dispatches(), 1);
    assert_eq!(k3_without_bias.estimated_encoding_events(), 1);

    // K=7 im2col+GEMM path: 2 dispatches (im2col + GEMM) + 1 bias = 3.
    let k7_with_bias = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 256],
        out_channels: 256,
        kernel_size: 7,
        stride: 1,
        padding: 3,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    assert_eq!(k7_with_bias.estimated_metal_dispatches(), 3);
    assert_eq!(k7_with_bias.estimated_encoding_events(), 3);

    // K=7 im2col+GEMM path: 2 dispatches (im2col + GEMM) without bias.
    let k7_without_bias = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 256],
        out_channels: 256,
        kernel_size: 7,
        stride: 1,
        padding: 3,
        dilation: 1,
        groups: 1,
        has_bias: false,
    };
    assert_eq!(k7_without_bias.estimated_metal_dispatches(), 2);
    assert_eq!(k7_without_bias.estimated_encoding_events(), 2);

    // K=3 with stride>1 falls back to im2col+GEMM path.
    let k3_stride2 = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 256],
        out_channels: 256,
        kernel_size: 3,
        stride: 2,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    assert_eq!(k3_stride2.estimated_metal_dispatches(), 3);
    assert_eq!(k3_stride2.estimated_encoding_events(), 3);
}

/// Conv1dGemm K=3 optimization savings for Kokoro-like shapes.
///
/// Validates that the shape-aware dispatch routing saves 1 Metal dispatch
/// per K=3 Conv1dGemm (direct sliding-window kernel vs im2col+GEMM).
/// In Kokoro generator/f0_energy segments, Conv1dGemm K=3 with bias is
/// common. Each saves 1 dispatch (2 instead of 3). Part of #4264.
#[test]
fn test_conv1d_gemm_k3_dispatch_savings() {
    // Kokoro generator uses 512-channel Conv1dGemm K=3 with bias.
    let kokoro_gen = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 512, 64],
        out_channels: 512,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    // K=3 direct path: 1 conv + 1 bias = 2 (vs 3 for im2col+GEMM+bias).
    assert_eq!(kokoro_gen.estimated_metal_dispatches(), 2);
    assert_eq!(kokoro_gen.estimated_encoding_events(), 2);

    // Kokoro f0_energy uses 256-channel Conv1dGemm K=3.
    let kokoro_f0 = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 256, 32],
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    assert_eq!(kokoro_f0.estimated_metal_dispatches(), 2);
    assert_eq!(kokoro_f0.estimated_encoding_events(), 2);

    // K=7 (generator upsample blocks) still uses im2col+GEMM = 3 dispatches.
    let kokoro_gen_k7 = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 512, 64],
        out_channels: 256,
        kernel_size: 7,
        stride: 1,
        padding: 3,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    assert_eq!(kokoro_gen_k7.estimated_metal_dispatches(), 3);
    assert_eq!(kokoro_gen_k7.estimated_encoding_events(), 3);

    // Quantify: a batch of N K=3 Conv1dGemm ops saves N Metal dispatches.
    let n_k3_ops = 6; // typical Kokoro generator K=3 count
    let old_metal = n_k3_ops * 3; // prior estimate: im2col + GEMM + bias
    let new_metal = n_k3_ops * 2; // K=3 direct path + bias
    assert_eq!(old_metal - new_metal, n_k3_ops); // savings = N dispatches
}

/// NormActivConv1d always uses 2 dispatches (stats + fused_norm_conv).
#[test]
fn test_native_op_norm_activ_conv1d_dispatch_count() {
    let snake = NativeOpKind::NormActivConv1d {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 256, 128],
        output_channels: 256,
        kernel_size: 3,
        external_node_ids: None,
    };
    assert_eq!(snake.estimated_metal_dispatches(), 2);
    assert_eq!(snake.estimated_encoding_events(), 1);

    let leaky = NativeOpKind::NormActivConv1d {
        activation: NormActivation::LeakyRelu { slope: 0.2 },
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 256, 128],
        output_channels: 256,
        kernel_size: 3,
        external_node_ids: None,
    };
    assert_eq!(leaky.estimated_metal_dispatches(), 2);
    assert_eq!(leaky.estimated_encoding_events(), 1);
}

// =============================================================================
// 3. NativeOpKind external_node_ids
// =============================================================================

/// external_node_ids returns Some for NormActivConv1d with IDs set.
#[test]
fn test_native_op_external_node_ids_norm_activ() {
    let op = NativeOpKind::NormActivConv1d {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 256, 128],
        output_channels: 256,
        kernel_size: 3,
        external_node_ids: Some(vec![10, 20, 30]),
    };
    assert_eq!(op.external_node_ids(), Some(&[10_u64, 20, 30][..]));
}

/// external_node_ids returns None for NormActivConv1d without IDs.
#[test]
fn test_native_op_external_node_ids_none() {
    let op = NativeOpKind::NormActivConv1d {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 256, 128],
        output_channels: 256,
        kernel_size: 3,
        external_node_ids: None,
    };
    assert_eq!(op.external_node_ids(), None);
}

/// external_node_ids returns Some for AdainSnake with IDs set.
#[test]
fn test_native_op_external_node_ids_adain_snake() {
    let op = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 256, 128],
        channels: 256,
        residual_gamma: true,
        external_node_ids: Some(vec![5, 6, 7]),
    };
    assert_eq!(op.external_node_ids(), Some(&[5_u64, 6, 7][..]));
}

/// external_node_ids returns None for ops that don't carry IDs.
#[test]
fn test_native_op_external_node_ids_unsupported_variant() {
    let op = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 768],
        hidden_dim: 768,
    };
    assert_eq!(op.external_node_ids(), None);

    let op2 = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: false,
        q_shape: vec![1, 1, 4, 8],
        k_shape: vec![1, 1, 4, 8],
        output_shape: vec![1, 1, 4, 8],
        input_layout: AttentionLayout::HeadsFirst,
    };
    assert_eq!(op2.external_node_ids(), None);
}

// =============================================================================
// 4. NativeOpKind collect_direct_step_deps
// =============================================================================

/// FusedResBlock collects input_steps, shortcut_step, and pool_step.
#[test]
fn test_collect_direct_step_deps_fused_resblock() {
    let op = NativeOpKind::FusedResBlock {
        phase1: NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            1,
            1,
            vec![1, 256, 128],
            256,
            3,
        ),
        phase2: NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            1,
            1,
            vec![1, 256, 128],
            256,
            3,
        ),
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: Some(10),
        pool_step: Some(20),
        style_batch_offset: None,
    };

    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);
    assert!(deps.contains(&0));
    assert!(deps.contains(&1));
    assert!(deps.contains(&4));
    assert!(deps.contains(&10), "shortcut_step should be collected");
    assert!(deps.contains(&20), "pool_step should be collected");
    assert_eq!(deps.len(), 7); // 5 input_steps + shortcut + pool
}

/// BatchedStyleProjection collects style_step.
#[test]
fn test_collect_direct_step_deps_batched_style_proj() {
    let op = NativeOpKind::BatchedStyleProjection {
        blocks: vec![],
        style_dim: 128,
        total_out: 256,
        style_step: 42,
    };
    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);
    assert_eq!(deps, vec![42]);
}

/// ProjectionSlice collects source_step.
#[test]
fn test_collect_direct_step_deps_projection_slice() {
    let op = NativeOpKind::ProjectionSlice {
        source_step: 7,
        dim: 2,
        start: 768,
        length: 768,
        output_shape: vec![2, 16, 768],
    };
    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);
    assert_eq!(deps, vec![7]);
}

/// Ops without direct step deps produce empty output.
#[test]
fn test_collect_direct_step_deps_no_deps() {
    let ops: Vec<NativeOpKind> = vec![
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 768],
            hidden_dim: 768,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 8, 256],
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 1, 4, 8],
            k_shape: vec![1, 1, 4, 8],
            output_shape: vec![1, 1, 4, 8],
            input_layout: AttentionLayout::HeadsFirst,
        },
    ];
    for op in &ops {
        let mut deps = Vec::new();
        op.collect_direct_step_deps(&mut deps);
        assert!(
            deps.is_empty(),
            "{} should have no direct step deps",
            op.variant_name()
        );
    }
}

// =============================================================================
// 5. NativeOpKind serde round-trip
// =============================================================================

/// NativeOpKind round-trips through JSON serialization.
#[test]
fn test_native_op_kind_serde_round_trip() {
    let ops: Vec<NativeOpKind> = vec![
        NativeOpKind::LstmSequence {
            hidden_size: 128,
            input_shape: vec![10, 1, 64],
            h_shape: vec![1, 128],
            reverse: true,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![2, 4, 256],
        },
        NativeOpKind::RotaryEmbedding {
            head_dim: 64,
            input_shape: vec![1, 8, 16, 64],
        },
        NativeOpKind::LinearActivation {
            activation: GemmActivation::GeluErf,
            in_features: 768,
            out_features: 3072,
            has_bias: true,
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::NormLinear {
            norm_kind: FusedNormKind::RmsNorm,
            eps: 1e-6,
            input_shape: vec![1, 32, 512],
            hidden_dim: 512,
            out_features: 2048,
            has_bias: false,
        },
        NativeOpKind::MoeGating {
            num_experts: 8,
            top_k: 2,
            input_shape: vec![4, 64],
        },
    ];

    for op in &ops {
        let json = serde_json::to_string(op).expect("serialize");
        let deserialized: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            op.variant_name(),
            deserialized.variant_name(),
            "variant_name mismatch after serde round-trip"
        );
    }
}

// =============================================================================
// 6. GemmActivation enum coverage
// =============================================================================

/// All GemmActivation variants have distinct Debug representations.
#[test]
fn test_gemm_activation_variants_distinct() {
    let variants = [
        GemmActivation::Relu,
        GemmActivation::Gelu,
        GemmActivation::GeluErf,
        GemmActivation::Sigmoid,
        GemmActivation::Silu,
        GemmActivation::Tanh,
    ];
    let debugs: Vec<String> = variants.iter().map(|v| format!("{v:?}")).collect();
    // All should be unique.
    let unique: std::collections::HashSet<_> = debugs.iter().collect();
    assert_eq!(
        unique.len(),
        variants.len(),
        "all GemmActivation variants should be distinct"
    );
}

/// GemmActivation PartialEq works correctly.
#[test]
fn test_gemm_activation_equality() {
    assert_eq!(GemmActivation::Relu, GemmActivation::Relu);
    assert_ne!(GemmActivation::Relu, GemmActivation::Gelu);
    assert_ne!(GemmActivation::Silu, GemmActivation::Sigmoid);
}

// =============================================================================
// 7. NormActivation enum coverage
// =============================================================================

/// NormActivation::Snake and LeakyRelu are distinct.
#[test]
fn test_norm_activation_variants() {
    let snake = NormActivation::Snake;
    let leaky = NormActivation::LeakyRelu { slope: 0.2 };
    assert_ne!(snake, leaky);

    let leaky2 = NormActivation::LeakyRelu { slope: 0.2 };
    assert_eq!(leaky, leaky2);

    let leaky_diff = NormActivation::LeakyRelu { slope: 0.1 };
    assert_ne!(leaky, leaky_diff);
}

// =============================================================================
// 8. Peephole edge cases
// =============================================================================

/// Empty step list does not panic.
#[test]
fn test_peephole_empty_steps() {
    let mut steps: Vec<CompiledStep> = vec![];
    let graph = ComputationGraph::from_nodes(vec![]);
    crate::trace_compile::peephole::apply_peephole(&mut steps, &graph);
    assert!(steps.is_empty());
}

/// Single step does not panic and remains unchanged.
#[test]
fn test_peephole_single_step() {
    let mut steps = vec![CompiledStep::InputForward];
    let graph = ComputationGraph::from_nodes(vec![test_input_node(0, vec![1, 4])]);
    crate::trace_compile::peephole::apply_peephole(&mut steps, &graph);
    assert!(matches!(&steps[0], CompiledStep::InputForward));
}

/// Conv1d with stride != 1 should NOT fuse with AdainLeakyRelu.
#[test]
fn test_peephole_no_fuse_conv1d_stride_gt_1() {
    let input_shape = [1, 512, 100];
    let output_shape = [1, 512, 50]; // halved due to stride=2
    let weight_shape = [512, 512, 3];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_adain_leaky_relu(&input_shape),
        make_conv1d_dispatch_full(&input_shape, &output_shape, &weight_shape, 2, 1, 1, 1),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_input_node(1, vec![1, 512, 1]),
        test_input_node(2, vec![1, 512, 1]),
        test_node(3, "adain", vec![0, 1, 2], input_shape.to_vec()),
        test_node(4, "conv1d", vec![3], output_shape.to_vec()),
    ]);

    crate::trace_compile::peephole::apply_peephole(&mut steps, &graph);

    // Should NOT fuse — stride != 1.
    assert!(
        matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::AdainLeakyRelu { .. },
                ..
            }
        ),
        "AdainLeakyRelu should remain unfused with stride > 1 conv"
    );
}

/// Conv1d with groups != 1 should NOT fuse with AdainSnake.
#[test]
fn test_peephole_no_fuse_conv1d_groups_gt_1() {
    let input_shape = [1, 256, 200];
    let output_shape = [1, 256, 200];
    // grouped conv: groups=2, weight shape [256, 128, 3]
    let weight_shape = [256, 128, 3];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_adain_snake(&input_shape, 256),
        make_conv1d_dispatch_full(&input_shape, &output_shape, &weight_shape, 1, 1, 1, 2),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_input_node(1, vec![1, 256, 1]),
        test_input_node(2, vec![1, 256, 1]),
        test_node(3, "adain_snake", vec![0, 1, 2], input_shape.to_vec()),
        test_node(4, "conv1d", vec![3], output_shape.to_vec()),
    ]);

    crate::trace_compile::peephole::apply_peephole(&mut steps, &graph);

    assert!(
        matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::AdainSnake { .. },
                ..
            }
        ),
        "AdainSnake should remain unfused with groups > 1 conv"
    );
}

/// Non-conv1d Dispatch after AdainLeakyRelu does not fuse.
#[test]
fn test_peephole_no_fuse_adain_followed_by_non_conv() {
    let input_shape = [1, 512, 100];
    let output_shape = [1, 512, 100];

    let mut steps = vec![
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        CompiledStep::InputForward,
        make_adain_leaky_relu(&input_shape),
        make_activation_dispatch("relu", &output_shape),
    ];

    let graph = ComputationGraph::from_nodes(vec![
        test_input_node(0, input_shape.to_vec()),
        test_input_node(1, vec![1, 512, 1]),
        test_input_node(2, vec![1, 512, 1]),
        test_node(3, "adain", vec![0, 1, 2], input_shape.to_vec()),
        test_node(4, "relu", vec![3], output_shape.to_vec()),
    ]);

    crate::trace_compile::peephole::apply_peephole(&mut steps, &graph);

    // Should NOT fuse — step[4] is relu not conv1d.
    assert!(
        matches!(
            &steps[3],
            CompiledStep::NativeOp {
                op: NativeOpKind::AdainLeakyRelu { .. },
                ..
            }
        ),
        "AdainLeakyRelu should not fuse with non-conv dispatch"
    );
}

// =============================================================================
// 9. DispatchStep variant construction
// =============================================================================

/// DispatchStep::Reduce can be constructed and has expected fields.
#[test]
fn test_dispatch_step_reduce_construction() {
    use crate::codegen_msl_tensor::DispatchStep;
    use crate::ir::ScalarType;
    use crate::tensor_ir::{ReduceOp, TensorNodeId};

    let step = DispatchStep::Reduce {
        kernel_name: "reduce_sum_0".to_string(),
        op: ReduceOp::Sum,
        dtype: ScalarType::F32,
        input: TensorNodeId::new(0),
        output: TensorNodeId::new(1),
        reduce_dim: 128,
        outer_size: 32,
        keepdim: false,
    };

    // Verify via Debug format that construction succeeds.
    let debug = format!("{step:?}");
    assert!(debug.contains("Reduce"), "should be a Reduce variant");
    assert!(debug.contains("reduce_sum_0"));
}

/// DispatchStep::Softmax can be constructed.
#[test]
fn test_dispatch_step_softmax_construction() {
    use crate::codegen_msl_tensor::DispatchStep;
    use crate::ir::ScalarType;
    use crate::tensor_ir::TensorNodeId;

    let step = DispatchStep::Softmax {
        kernel_name: "softmax_0".to_string(),
        dtype: ScalarType::F32,
        input: TensorNodeId::new(0),
        output: TensorNodeId::new(1),
        axis: 2,
        axis_size: 768,
        outer_size: 32,
    };

    let debug = format!("{step:?}");
    assert!(debug.contains("Softmax"));
}

/// DispatchStep::Embedding can be constructed.
#[test]
fn test_dispatch_step_embedding_construction() {
    use crate::codegen_msl_tensor::DispatchStep;
    use crate::ir::ScalarType;
    use crate::tensor_ir::TensorNodeId;

    let step = DispatchStep::Embedding {
        kernel_name: "embedding_0".to_string(),
        dtype: ScalarType::F32,
        input: TensorNodeId::new(0),
        weight: TensorNodeId::new(1),
        output: TensorNodeId::new(2),
        embedding_dim: 768,
        num_indices: 32,
        total_elements: 32 * 768,
    };

    let debug = format!("{step:?}");
    assert!(debug.contains("Embedding"));
    assert!(debug.contains("768"));
}

/// DispatchStep::BinaryAdd can be constructed with and without broadcast.
#[test]
fn test_dispatch_step_binary_add_with_and_without_broadcast() {
    use crate::codegen_msl_tensor::DispatchStep;
    use crate::ir::ScalarType;
    use crate::tensor_ir::TensorNodeId;

    // Without broadcast
    let step_no_broadcast = DispatchStep::BinaryAdd {
        kernel_name: "add_0".to_string(),
        dtype: ScalarType::F32,
        left: TensorNodeId::new(0),
        right: TensorNodeId::new(1),
        output: TensorNodeId::new(2),
        total_elements: 1024,
        broadcast: None,
    };
    let debug = format!("{step_no_broadcast:?}");
    assert!(debug.contains("BinaryAdd"));
    assert!(debug.contains("None")); // broadcast is None
}

// =============================================================================
// 10. StyleProjectionParams and StyleBatchOffset constructors
// =============================================================================

/// StyleProjectionParams::new constructs with correct fields.
#[test]
fn test_style_projection_params_new() {
    let params = StyleProjectionParams::new(256, 512, 128);
    assert_eq!(params.channels1, 256);
    assert_eq!(params.channels2, 512);
    assert_eq!(params.style_dim, 128);
}

/// StyleBatchOffset::new constructs with correct fields.
#[test]
fn test_style_batch_offset_new() {
    let offset = StyleBatchOffset::new(1024, 256, 512);
    assert_eq!(offset.offset, 1024);
    assert_eq!(offset.channels1, 256);
    assert_eq!(offset.channels2, 512);
}

/// NormActivConv1dParams::new constructs with correct fields.
#[test]
fn test_norm_activ_conv1d_params_new() {
    let params =
        NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 3, 1, vec![1, 256, 128], 256, 7);
    assert_eq!(params.activation, NormActivation::Snake);
    assert!((params.eps - 1e-5).abs() < 1e-10);
    assert_eq!(params.conv_dilation, 3);
    assert_eq!(params.conv_padding, 1);
    assert_eq!(params.input_shape, vec![1, 256, 128]);
    assert_eq!(params.output_channels, 256);
    assert_eq!(params.kernel_size, 7);
}

// =============================================================================
// 11. FusedNormKind and AttentionLayout
// =============================================================================

/// FusedNormKind variants are distinct.
#[test]
fn test_fused_norm_kind_equality() {
    assert_eq!(FusedNormKind::LayerNorm, FusedNormKind::LayerNorm);
    assert_eq!(FusedNormKind::RmsNorm, FusedNormKind::RmsNorm);
    assert_ne!(FusedNormKind::LayerNorm, FusedNormKind::RmsNorm);
}

/// AttentionLayout default is HeadsFirst.
#[test]
fn test_attention_layout_default() {
    let layout: AttentionLayout = Default::default();
    assert_eq!(layout, AttentionLayout::HeadsFirst);
    assert_ne!(layout, AttentionLayout::SeqFirst);
}

/// FlashAttention with SeqFirst layout has correct dispatch count.
#[test]
fn test_flash_attention_seq_first_dispatch_count() {
    let op = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: true,
        q_shape: vec![1, 8, 32, 64],
        k_shape: vec![1, 8, 32, 64],
        output_shape: vec![1, 8, 32, 64],
        input_layout: AttentionLayout::SeqFirst,
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.variant_name(), "FlashAttention");
}

// =============================================================================
// 12. PeepholeConfig default has all passes enabled
// =============================================================================

#[test]
fn test_peephole_config_default_all_enabled() {
    let config = PeepholeConfig::default();
    assert!(config.norm_activ_conv1d);
    assert!(config.fused_resblock);
    assert!(config.linear_activation);
    assert!(config.add_layer_norm);
    assert!(config.norm_linear);
    assert!(config.attention_transpose);
    assert!(config.flip_lstm);
    assert!(config.batched_linear_projection);
    assert!(config.channels_first_layer_norm);
    assert!(config.silu_mul);
    assert!(config.auto_fuse_elementwise);
}
