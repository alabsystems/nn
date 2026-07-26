// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for ICB NativeOp encoding classification.
//!
//! Tests the pure logic in `compiled_model_icb_native.rs`:
//! NativeOp ICB eligibility, dispatch geometry computation,
//! and buffer binding counts.
//!
//! Part of #3458.

use std::collections::HashMap;

use nn_dsl::NativeOpKind;

use super::{
    compute_native_dispatch_geometry, count_icb_eligible_native_ops, is_native_op_icb_eligible,
    try_encode_native_op_icb, IcbNativeDispatchKind,
};

// ── SiluMul ────────────────────────────────────────────────────────────

#[test]
fn silu_mul_is_icb_eligible() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![1, 4096, 3584],
    };
    assert!(is_native_op_icb_eligible(&op));
}

#[test]
fn silu_mul_command_metadata() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![1, 4096, 3584],
    };
    let cmd = try_encode_native_op_icb(&op, 42).expect("SiluMul should be ICB-eligible");

    assert_eq!(cmd.buffer_binding_count, 3); // gate + up + output
    assert_eq!(cmd.total_output_elements, 1 * 4096 * 3584);
    assert_eq!(cmd.op_tag, "SiluMul");
    assert!(matches!(cmd.dispatch_kind, IcbNativeDispatchKind::Elementwise));
}

#[test]
fn silu_mul_dispatch_geometry() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![1, 256],
    };
    let cmd = try_encode_native_op_icb(&op, 0).unwrap();
    let (grid, threads) = compute_native_dispatch_geometry(&cmd).unwrap();

    // 256 elements / 256 threads per TG = 1 threadgroup
    assert_eq!(grid, [1, 1, 1]);
    assert_eq!(threads, [256, 1, 1]);
}

#[test]
fn silu_mul_dispatch_geometry_large() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![2, 1024],
    };
    let cmd = try_encode_native_op_icb(&op, 0).unwrap();
    let (grid, threads) = compute_native_dispatch_geometry(&cmd).unwrap();

    // 2048 elements / 256 = 8 threadgroups
    assert_eq!(grid, [8, 1, 1]);
    assert_eq!(threads, [256, 1, 1]);
}

#[test]
fn silu_mul_empty_shape_not_eligible() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![0, 256],
    };
    assert!(!is_native_op_icb_eligible(&op));
}

// ── RotaryEmbedding ────────────────────────────────────────────────────

#[test]
fn rotary_embedding_is_icb_eligible() {
    let op = NativeOpKind::RotaryEmbedding {
        head_dim: 128,
        input_shape: vec![1, 32, 512, 128],
    };
    assert!(is_native_op_icb_eligible(&op));
}

#[test]
fn rotary_embedding_command_metadata() {
    let op = NativeOpKind::RotaryEmbedding {
        head_dim: 128,
        input_shape: vec![1, 32, 512, 128],
    };
    let cmd = try_encode_native_op_icb(&op, 10).unwrap();

    assert_eq!(cmd.buffer_binding_count, 4); // input + cos + sin + output
    assert_eq!(cmd.total_output_elements, 1 * 32 * 512 * 128);
    assert_eq!(cmd.op_tag, "RotaryEmbedding");
    assert!(matches!(cmd.dispatch_kind, IcbNativeDispatchKind::Elementwise));
}

#[test]
fn rotary_embedding_zero_head_dim_not_eligible() {
    let op = NativeOpKind::RotaryEmbedding {
        head_dim: 0,
        input_shape: vec![1, 32, 512, 128],
    };
    assert!(!is_native_op_icb_eligible(&op));
}

// ── MaxPool1d ──────────────────────────────────────────────────────────

#[test]
fn max_pool1d_is_icb_eligible() {
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 2,
        padding: 1,
        input_shape: vec![4, 64, 100],
    };
    assert!(is_native_op_icb_eligible(&op));
}

#[test]
fn max_pool1d_command_metadata() {
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 2,
        padding: 1,
        input_shape: vec![4, 64, 100],
    };
    let cmd = try_encode_native_op_icb(&op, 5).unwrap();

    // l_out = (100 + 2*1 - 3) / 2 + 1 = 99 / 2 + 1 = 50
    assert_eq!(cmd.buffer_binding_count, 2); // input + output
    assert_eq!(cmd.total_output_elements, 4 * 64 * 50);
    assert_eq!(cmd.op_tag, "MaxPool1d");
    assert!(matches!(cmd.dispatch_kind, IcbNativeDispatchKind::Elementwise));
}

#[test]
fn max_pool1d_short_input_not_eligible() {
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 10,
        stride: 1,
        padding: 0,
        input_shape: vec![1, 1, 5], // l_in=5 < kernel_size=10
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn max_pool1d_rank_too_low_not_eligible() {
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 1,
        padding: 0,
        input_shape: vec![10, 5], // rank 2 < required 3
    };
    assert!(!is_native_op_icb_eligible(&op));
}

// ── LinearActivation ───────────────────────────────────────────────────

#[test]
fn linear_activation_is_icb_eligible() {
    let op = NativeOpKind::LinearActivation {
        activation: nn_dsl::GemmActivation::Gelu,
        in_features: 768,
        out_features: 3072,
        has_bias: true,
        input_shape: vec![1, 512, 768],
    };
    assert!(is_native_op_icb_eligible(&op));
}

#[test]
fn linear_activation_command_with_bias() {
    let op = NativeOpKind::LinearActivation {
        activation: nn_dsl::GemmActivation::Relu,
        in_features: 256,
        out_features: 1024,
        has_bias: true,
        input_shape: vec![4, 256],
    };
    let cmd = try_encode_native_op_icb(&op, 0).unwrap();

    assert_eq!(cmd.buffer_binding_count, 4); // input + weight + bias + output
    assert_eq!(cmd.total_output_elements, 4 * 1024);
    assert_eq!(cmd.op_tag, "LinearActivation");
    match &cmd.dispatch_kind {
        IcbNativeDispatchKind::TiledGemm { m, k, n } => {
            assert_eq!(*m, 4);
            assert_eq!(*k, 256);
            assert_eq!(*n, 1024);
        }
        other => panic!("expected TiledGemm, got {other:?}"),
    }
}

#[test]
fn linear_activation_command_without_bias() {
    let op = NativeOpKind::LinearActivation {
        activation: nn_dsl::GemmActivation::Silu,
        in_features: 128,
        out_features: 512,
        has_bias: false,
        input_shape: vec![8, 128],
    };
    let cmd = try_encode_native_op_icb(&op, 0).unwrap();

    assert_eq!(cmd.buffer_binding_count, 3); // input + weight + output (no bias)
}

#[test]
fn linear_activation_dispatch_geometry() {
    let op = NativeOpKind::LinearActivation {
        activation: nn_dsl::GemmActivation::Gelu,
        in_features: 768,
        out_features: 3072,
        has_bias: true,
        input_shape: vec![1, 64, 768],
    };
    let cmd = try_encode_native_op_icb(&op, 0).unwrap();
    let (grid, threads) = compute_native_dispatch_geometry(&cmd).unwrap();

    // m = 1*64 = 64, n = 3072
    // grid = [3072/32, 64/32, 1] = [96, 2, 1]
    assert_eq!(grid, [96, 2, 1]);
    assert_eq!(threads, [32, 4, 1]);
}

#[test]
fn linear_activation_zero_features_not_eligible() {
    let op = NativeOpKind::LinearActivation {
        activation: nn_dsl::GemmActivation::Relu,
        in_features: 0,
        out_features: 256,
        has_bias: false,
        input_shape: vec![1, 0],
    };
    assert!(!is_native_op_icb_eligible(&op));
}

// ── Int8Gemm ───────────────────────────────────────────────────────────

#[test]
fn int8_gemm_is_icb_eligible() {
    let op = NativeOpKind::Int8Gemm {
        in_features: 4096,
        out_features: 4096,
        has_bias: false,
        input_shape: vec![1, 128, 4096],
    };
    assert!(is_native_op_icb_eligible(&op));
}

#[test]
fn int8_gemm_command_with_bias() {
    let op = NativeOpKind::Int8Gemm {
        in_features: 256,
        out_features: 512,
        has_bias: true,
        input_shape: vec![2, 256],
    };
    let cmd = try_encode_native_op_icb(&op, 0).unwrap();

    // input + weight_int8 + scale + zero_point + bias + output = 6
    assert_eq!(cmd.buffer_binding_count, 6);
    assert_eq!(cmd.total_output_elements, 2 * 512);
    assert_eq!(cmd.op_tag, "Int8Gemm");
}

#[test]
fn int8_gemm_command_without_bias() {
    let op = NativeOpKind::Int8Gemm {
        in_features: 256,
        out_features: 512,
        has_bias: false,
        input_shape: vec![2, 256],
    };
    let cmd = try_encode_native_op_icb(&op, 0).unwrap();

    // input + weight_int8 + scale + zero_point + output = 5
    assert_eq!(cmd.buffer_binding_count, 5);
}

// ── Multi-dispatch ops: NOT ICB-eligible ───────────────────────────────

#[test]
fn lstm_not_icb_eligible() {
    let op = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![8, 1, 640],
        h_shape: vec![1, 256],
        reverse: false,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn cumsum_not_icb_eligible() {
    let op = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![100],
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn instance_norm_not_icb_eligible() {
    let op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 64, 256],
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn layer_norm_not_icb_eligible() {
    let op = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 128, 768],
        hidden_dim: 768,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn add_layer_norm_not_icb_eligible() {
    let op = NativeOpKind::AddLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 128, 768],
        hidden_dim: 768,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn channels_first_layer_norm_not_icb_eligible() {
    let op = NativeOpKind::ChannelsFirstLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 64, 256],
        channels: 64,
        leaky_relu_slope: None,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn adain_snake_not_icb_eligible() {
    let op = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 256, 100],
        channels: 256,
        residual_gamma: true,
        external_node_ids: None,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn adain_leaky_relu_not_icb_eligible() {
    let op = NativeOpKind::AdainLeakyRelu {
        eps: 1e-5,
        slope: 0.2,
        input_shape: vec![1, 256, 100],
        external_node_ids: None,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn ada_layer_norm_not_icb_eligible() {
    let op = NativeOpKind::AdaLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 128, 256],
        hidden_dim: 256,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn flash_attention_not_icb_eligible() {
    let op = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: false,
        q_shape: vec![1, 8, 128, 64],
        k_shape: vec![1, 8, 128, 64],
        output_shape: vec![1, 8, 128, 64],
        input_layout: nn_dsl::AttentionLayout::default(),
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn fused_resblock_not_icb_eligible() {
    let op = NativeOpKind::FusedResBlock {
        phase1: nn_dsl::NormActivConv1dParams::new(
            nn_dsl::NormActivation::Snake,
            1e-5,
            1,
            1,
            vec![1, 256, 100],
            256,
            3,
        ),
        phase2: nn_dsl::NormActivConv1dParams::new(
            nn_dsl::NormActivation::Snake,
            1e-5,
            1,
            1,
            vec![1, 256, 100],
            256,
            3,
        ),
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn norm_activ_conv1d_not_icb_eligible() {
    let op = NativeOpKind::NormActivConv1d {
        activation: nn_dsl::NormActivation::LeakyRelu { slope: 0.2 },
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 256, 100],
        output_channels: 256,
        kernel_size: 3,
        external_node_ids: None,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn norm_linear_not_icb_eligible() {
    let op = NativeOpKind::NormLinear {
        norm_kind: nn_dsl::FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![1, 128, 768],
        hidden_dim: 768,
        out_features: 3072,
        has_bias: true,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn batched_linear_projection_not_icb_eligible() {
    let op = NativeOpKind::BatchedLinearProjection {
        in_features: 768,
        total_out_features: 2304,
        projection_sizes: vec![768, 768, 768],
        has_bias: false,
        input_shape: vec![1, 128, 768],
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn projection_slice_not_icb_eligible() {
    let op = NativeOpKind::ProjectionSlice {
        source_step: 10,
        dim: 2,
        start: 768,
        length: 768,
        output_shape: vec![1, 128, 768],
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn batched_style_projection_not_icb_eligible() {
    let op = NativeOpKind::BatchedStyleProjection {
        blocks: vec![],
        style_dim: 128,
        total_out: 1024,
        style_step: 0,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn constant_weight_not_icb_eligible() {
    let op = NativeOpKind::ConstantWeight {
        name: "arange".into(),
        shape: vec![100],
    };
    assert!(!is_native_op_icb_eligible(&op));
}

#[test]
fn conv1d_gemm_not_icb_eligible() {
    // Conv1dGemm uses im2col + GEMM internally (2 dispatches).
    let op = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 256],
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    assert!(!is_native_op_icb_eligible(&op));
}

// ── count_icb_eligible_native_ops ──────────────────────────────────────

#[test]
fn count_eligible_mixed_steps() {
    use nn_dsl::CompiledStep;

    let steps = vec![
        CompiledStep::NativeOp {
            op: NativeOpKind::SiluMul {
                input_shape: vec![1, 256],
            },
            weight_data: HashMap::new(),
        },
        CompiledStep::NativeOp {
            op: NativeOpKind::LstmSequence {
                hidden_size: 256,
                input_shape: vec![8, 1, 640],
                h_shape: vec![1, 256],
                reverse: false,
            },
            weight_data: HashMap::new(),
        },
        CompiledStep::NativeOp {
            op: NativeOpKind::RotaryEmbedding {
                head_dim: 128,
                input_shape: vec![1, 32, 128, 128],
            },
            weight_data: HashMap::new(),
        },
        // Dispatch step (not a NativeOp) should not be counted.
        make_dispatch_step(),
    ];

    let (eligible, total) = count_icb_eligible_native_ops(&steps);
    assert_eq!(eligible, 2); // SiluMul + RotaryEmbedding
    assert_eq!(total, 3); // 3 NativeOp steps (Dispatch not counted)
}

#[test]
fn count_eligible_empty() {
    let (eligible, total) = count_icb_eligible_native_ops(&[]);
    assert_eq!(eligible, 0);
    assert_eq!(total, 0);
}

#[test]
fn count_eligible_no_native_ops() {
    let steps = vec![make_dispatch_step(), make_dispatch_step()];
    let (eligible, total) = count_icb_eligible_native_ops(&steps);
    assert_eq!(eligible, 0);
    assert_eq!(total, 0);
}

// ── compute_native_dispatch_geometry ───────────────────────────────────

#[test]
fn geometry_elementwise_exact_multiple() {
    let cmd = super::IcbNativeCommand {
        buffer_binding_count: 3,
        total_output_elements: 1024,
        op_tag: "test",
        dispatch_kind: IcbNativeDispatchKind::Elementwise,
    };
    let (grid, threads) = compute_native_dispatch_geometry(&cmd).unwrap();
    assert_eq!(grid, [4, 1, 1]); // 1024 / 256 = 4
    assert_eq!(threads, [256, 1, 1]);
}

#[test]
fn geometry_elementwise_non_multiple() {
    let cmd = super::IcbNativeCommand {
        buffer_binding_count: 3,
        total_output_elements: 300,
        op_tag: "test",
        dispatch_kind: IcbNativeDispatchKind::Elementwise,
    };
    let (grid, threads) = compute_native_dispatch_geometry(&cmd).unwrap();
    assert_eq!(grid, [2, 1, 1]); // ceil(300 / 256) = 2
    assert_eq!(threads, [256, 1, 1]);
}

#[test]
fn geometry_tiled_gemm() {
    let cmd = super::IcbNativeCommand {
        buffer_binding_count: 4,
        total_output_elements: 64 * 256,
        op_tag: "test",
        dispatch_kind: IcbNativeDispatchKind::TiledGemm {
            m: 64,
            k: 128,
            n: 256,
        },
    };
    let (grid, threads) = compute_native_dispatch_geometry(&cmd).unwrap();
    assert_eq!(grid, [8, 2, 1]); // ceil(256/32)=8, ceil(64/32)=2
    assert_eq!(threads, [32, 4, 1]);
}

#[test]
fn geometry_grid3d_passthrough() {
    let cmd = super::IcbNativeCommand {
        buffer_binding_count: 2,
        total_output_elements: 100,
        op_tag: "test",
        dispatch_kind: IcbNativeDispatchKind::Grid3D {
            grid: [5, 10, 1],
            threads: [32, 8, 1],
        },
    };
    let (grid, threads) = compute_native_dispatch_geometry(&cmd).unwrap();
    assert_eq!(grid, [5, 10, 1]);
    assert_eq!(threads, [32, 8, 1]);
}

// ── Helpers ────────────────────────────────────────────────────────────

fn make_dispatch_step() -> nn_dsl::CompiledStep {
    use nn_dsl::{
        CompiledKernel, CompiledStep, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
    };

    let node = TensorNode::new(
        TensorNodeId::new(0),
        TensorOpKind::Input {
            name: "x".into(),
            shape: vec![1],
        },
        vec![1],
    );
    let def = TensorKernelDef::new("test_kernel", vec![node], TensorNodeId::new(0));
    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}
