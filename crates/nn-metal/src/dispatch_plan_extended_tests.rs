// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for dispatch plan, kernel execution infrastructure, and
//! compiled model support types.
//!
//! Tests cover:
//! 1. DispatchStep variant structure consistency
//! 2. Dispatch plan analysis (dispatch count, memory estimation)
//! 3. NativeOpKind registry — all 31 variants covered
//! 4. Kernel fusion opportunity detection between adjacent steps
//! 5. F16 autocast config for compute-heavy ops
//! 6. Activation arena allocation and buffer reuse tracking
//! 7. Memory barrier planning (command batch barrier strategy)
//! 8. Workgroup configuration (optimal workgroup size selection)
//!
//! All tests are structure/config tests only — no GPU required.
//! Part of #4186.

use std::collections::HashSet;

use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_dsl::ir::{NodeId, ScalarType};
use nn_dsl::{DispatchStep, NativeOpKind, TensorNodeId};

use crate::dispatch_plan::{
    plan_elementwise, plan_grid_3d, plan_reduction, threadgroup_width_1d,
    DispatchMode,
};
use crate::dispatch_profiler::{DispatchProfileEntry, DispatchType, FusionOpportunity};
use crate::simdgroup_tile_select::{
    select_gemm_tiles, TileConfig, SIMDGROUP_ALIGN, SMALL_K_THRESHOLD, STANDARD_MN_THRESHOLD,
    TALL_SKINNY_RATIO, TINY_THRESHOLD, WIDE_RATIO,
};
use crate::F16AutocastConfig;

// ═══════════════════════════════════════════════════════════════════════
// 1. DispatchStep variants — structural consistency
// ═══════════════════════════════════════════════════════════════════════

/// Every DispatchStep variant that carries a kernel_name should have a
/// non-empty name string. Tests construction of each variant with consistent
/// node IDs and verifies Debug output includes the variant name.
#[test]
fn dispatch_step_all_variants_have_debug_output() {
    let node0 = TensorNodeId::new(0);
    let node1 = TensorNodeId::new(1);
    let node2 = TensorNodeId::new(2);

    let steps: Vec<(&str, DispatchStep)> = vec![
        (
            "Reduce",
            DispatchStep::Reduce {
                kernel_name: "reduce_sum".into(),
                op: nn_dsl::ReduceOp::Sum,
                dtype: ScalarType::F32,
                input: node0,
                output: node1,
                reduce_dim: 64,
                outer_size: 32,
                keepdim: false,
            },
        ),
        (
            "Elementwise",
            DispatchStep::Elementwise {
                kernel_name: "ew_add".into(),
                scalar_kernel: nn_dsl::ir::KernelDef::new(
                    "add",
                    vec![],
                    ScalarType::F32,
                    vec![],
                    NodeId::new(0),
                ),
                inputs: vec![node0],
                output: node1,
                total_elements: 1024,
            },
        ),
        (
            "BinaryAdd",
            DispatchStep::BinaryAdd {
                kernel_name: "bin_add".into(),
                dtype: ScalarType::F32,
                left: node0,
                right: node1,
                output: node2,
                total_elements: 512,
                broadcast: None,
            },
        ),
        (
            "BinaryMul",
            DispatchStep::BinaryMul {
                kernel_name: "bin_mul".into(),
                dtype: ScalarType::F32,
                left: node0,
                right: node1,
                output: node2,
                total_elements: 512,
                broadcast: None,
            },
        ),
        (
            "Sigmoid",
            DispatchStep::Sigmoid {
                kernel_name: "sigmoid".into(),
                dtype: ScalarType::F32,
                input: node0,
                output: node1,
                total_elements: 256,
            },
        ),
        (
            "Gelu",
            DispatchStep::Gelu {
                kernel_name: "gelu".into(),
                dtype: ScalarType::F32,
                input: node0,
                output: node1,
                total_elements: 256,
            },
        ),
        (
            "GeluErf",
            DispatchStep::GeluErf {
                kernel_name: "gelu_erf".into(),
                dtype: ScalarType::F32,
                input: node0,
                output: node1,
                total_elements: 256,
            },
        ),
        (
            "Relu",
            DispatchStep::Relu {
                kernel_name: "relu".into(),
                dtype: ScalarType::F32,
                input: node0,
                output: node1,
                total_elements: 256,
            },
        ),
        (
            "Tanh",
            DispatchStep::Tanh {
                kernel_name: "tanh".into(),
                dtype: ScalarType::F32,
                input: node0,
                output: node1,
                total_elements: 256,
            },
        ),
        (
            "Reshape",
            DispatchStep::Reshape {
                input: node0,
                output: node1,
            },
        ),
        (
            "Softmax",
            DispatchStep::Softmax {
                kernel_name: "softmax".into(),
                dtype: ScalarType::F32,
                input: node0,
                output: node1,
                axis: 1,
                axis_size: 768,
                outer_size: 32,
            },
        ),
        (
            "Transpose",
            DispatchStep::Transpose {
                kernel_name: "transpose".into(),
                dtype: ScalarType::F32,
                input: node0,
                output: node1,
                input_shape: vec![4, 8, 16],
                axes: vec![0, 2, 1],
                total_elements: 512,
            },
        ),
        (
            "Embedding",
            DispatchStep::Embedding {
                kernel_name: "embed".into(),
                dtype: ScalarType::F32,
                input: node0,
                weight: node1,
                output: node2,
                embedding_dim: 768,
                num_indices: 128,
                total_elements: 128 * 768,
            },
        ),
    ];

    for (name, step) in &steps {
        let dbg = format!("{step:?}");
        assert!(
            dbg.contains(name),
            "Debug output for {name} should include variant name, got: {dbg}"
        );
    }
}

/// DispatchStep::Reshape has no kernel_name — it is zero-copy.
#[test]
fn dispatch_step_reshape_is_zero_copy() {
    let step = DispatchStep::Reshape {
        input: TensorNodeId::new(0),
        output: TensorNodeId::new(1),
    };
    let dbg = format!("{step:?}");
    assert!(
        !dbg.contains("kernel_name"),
        "Reshape should not have a kernel_name: {dbg}"
    );
}

/// DispatchStep::uses_input correctly identifies inputs for binary ops.
#[test]
fn dispatch_step_uses_input_binary_ops() {
    let left = TensorNodeId::new(10);
    let right = TensorNodeId::new(20);
    let output = TensorNodeId::new(30);
    let other = TensorNodeId::new(99);

    let step = DispatchStep::BinaryAdd {
        kernel_name: "add".into(),
        dtype: ScalarType::F32,
        left,
        right,
        output,
        total_elements: 100,
        broadcast: None,
    };

    assert!(step.uses_input(left), "left should be an input");
    assert!(step.uses_input(right), "right should be an input");
    assert!(!step.uses_input(output), "output should not be an input");
    assert!(!step.uses_input(other), "unrelated node should not be input");
}

/// DispatchStep::uses_input for reduction ops.
#[test]
fn dispatch_step_uses_input_reduce() {
    let input = TensorNodeId::new(5);
    let output = TensorNodeId::new(6);

    let step = DispatchStep::Reduce {
        kernel_name: "reduce".into(),
        op: nn_dsl::ReduceOp::Sum,
        dtype: ScalarType::F32,
        input,
        output,
        reduce_dim: 64,
        outer_size: 16,
        keepdim: false,
    };

    assert!(step.uses_input(input));
    assert!(!step.uses_input(output));
}

/// DispatchStep::Linear uses_input correctly identifies input, weight, bias.
#[test]
fn dispatch_step_uses_input_linear() {
    let input = TensorNodeId::new(0);
    let weight = TensorNodeId::new(1);
    let bias = TensorNodeId::new(2);
    let output = TensorNodeId::new(3);

    let step = DispatchStep::Linear {
        kernel_name: "linear".into(),
        dtype: ScalarType::F32,
        input,
        weight,
        bias: Some(bias),
        output,
        in_features: 768,
        out_features: 3072,
        batch_size: 1,
        total_elements: 3072,
    };

    assert!(step.uses_input(input));
    assert!(step.uses_input(weight));
    assert!(step.uses_input(bias));
    assert!(!step.uses_input(output));
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Dispatch plan analysis — dispatch count & memory estimation
// ═══════════════════════════════════════════════════════════════════════

/// Memory estimation via output_elems * element_size for F32.
#[test]
fn dispatch_plan_memory_estimation_elementwise() {
    let plan = plan_elementwise(1024).unwrap();
    let element_size_f32: usize = 4;
    let estimated_bytes = plan.output_elems() * element_size_f32;
    assert_eq!(estimated_bytes, 4096, "1024 F32 elements = 4096 bytes");
}

/// Memory estimation for Grid3D plan — product of grid dims * element size.
#[test]
fn dispatch_plan_memory_estimation_grid3d() {
    let plan = plan_grid_3d([8, 16, 32], [4, 4, 4]).unwrap();
    let element_size_f32: usize = 4;
    let estimated_bytes = plan.output_elems() * element_size_f32;
    assert_eq!(
        estimated_bytes,
        8 * 16 * 32 * 4,
        "8*16*32 F32 elements"
    );
}

/// Memory estimation for reduction — one output per outer slice.
#[test]
fn dispatch_plan_memory_estimation_reduction() {
    let plan = plan_reduction(256, 768, 128, 2048).unwrap();
    let element_size_f32: usize = 4;
    let estimated_bytes = plan.output_elems() * element_size_f32;
    assert_eq!(
        estimated_bytes,
        256 * 4,
        "256 outer slices, each producing 1 F32 output"
    );
}

/// Multi-plan dispatch counting: count how many plans use threadgroups (reduction).
#[test]
fn dispatch_plan_count_threadgroup_vs_threads() {
    let plans = [
        DispatchMode::Elementwise { total: 512 }.plan().unwrap(),
        DispatchMode::Grid2D {
            grid: [32, 16],
            threads: [8, 4],
        }
        .plan()
        .unwrap(),
        DispatchMode::PerSliceReduction {
            outer: 64,
            reduce: 256,
            threads: 128,
            shared_bytes: 1024,
        }
        .plan()
        .unwrap(),
        DispatchMode::Elementwise { total: 256 }.plan().unwrap(),
        DispatchMode::PerSliceReduction {
            outer: 32,
            reduce: 512,
            threads: 64,
            shared_bytes: 2048,
        }
        .plan()
        .unwrap(),
    ];

    let threadgroup_count = plans.iter().filter(|p| p.use_threadgroups()).count();
    let threads_count = plans.iter().filter(|p| !p.use_threadgroups()).count();
    assert_eq!(threadgroup_count, 2, "2 reductions use threadgroups");
    assert_eq!(threads_count, 3, "3 non-reductions use dispatch_threads");
}

// ═══════════════════════════════════════════════════════════════════════
// 3. NativeOpKind registry — all 31 variants covered
// ═══════════════════════════════════════════════════════════════════════

/// Build one instance of every NativeOpKind variant and verify variant_name().
/// This test ensures all 31+ variants are constructable and named.
#[test]
fn native_op_kind_all_variants_named() {
    let all_ops: Vec<NativeOpKind> = vec![
        NativeOpKind::LstmSequence {
            hidden_size: 256,
            input_shape: vec![100, 1, 512],
            h_shape: vec![1, 256],
            reverse: false,
        },
        NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![1, 10],
        },
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 64, 256],
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 128, 768],
            hidden_dim: 768,
        },
        NativeOpKind::AddLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 128, 768],
            hidden_dim: 768,
        },
        NativeOpKind::AdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 64, 256],
            channels: 64,
            residual_gamma: true,
            external_node_ids: None,
        },
        NativeOpKind::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.2,
            input_shape: vec![1, 64, 256],
            external_node_ids: None,
        },
        NativeOpKind::AdaLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 128, 256],
            hidden_dim: 256,
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 8, 64, 64],
            k_shape: vec![1, 8, 64, 64],
            output_shape: vec![1, 8, 64, 64],
            input_layout: nn_dsl::AttentionLayout::default(),
        },
        NativeOpKind::MaxPool1d {
            kernel_size: 3,
            stride: 2,
            padding: 1,
            input_shape: vec![1, 64, 256],
        },
        NativeOpKind::ConstantWeight {
            name: "arange".into(),
            shape: vec![256],
        },
        NativeOpKind::FusedResBlock {
            phase1: nn_dsl::NormActivConv1dParams::new(
                nn_dsl::NormActivation::Snake,
                1e-5,
                1, // conv_dilation
                1, // conv_padding
                vec![1, 64, 256], // input_shape
                64, // output_channels
                3,  // kernel_size
            ),
            phase2: nn_dsl::NormActivConv1dParams::new(
                nn_dsl::NormActivation::Snake,
                1e-5,
                1,
                1,
                vec![1, 64, 256],
                64,
                3,
            ),
            input_steps: vec![0, 1, 2, 3, 4],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        },
        NativeOpKind::NormActivConv1d {
            activation: nn_dsl::NormActivation::Snake,
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 1,
            input_shape: vec![1, 64, 256],
            output_channels: 64,
            kernel_size: 3,
            external_node_ids: None,
        },
        NativeOpKind::LinearActivation {
            activation: nn_dsl::GemmActivation::Relu,
            in_features: 768,
            out_features: 3072,
            has_bias: true,
            input_shape: vec![1, 128, 768],
        },
        NativeOpKind::BatchedLinearProjection {
            in_features: 768,
            total_out_features: 2304,
            projection_sizes: vec![768, 768, 768],
            has_bias: true,
            input_shape: vec![1, 128, 768],
        },
        NativeOpKind::ProjectionSlice {
            source_step: 5,
            dim: 2,
            start: 768,
            length: 768,
            output_shape: vec![1, 128, 768],
        },
        NativeOpKind::NormLinear {
            norm_kind: nn_dsl::FusedNormKind::LayerNorm,
            eps: 1e-5,
            input_shape: vec![1, 128, 768],
            hidden_dim: 768,
            out_features: 3072,
            has_bias: true,
        },
        NativeOpKind::BatchedStyleProjection {
            blocks: vec![],
            style_dim: 128,
            total_out: 512,
            style_step: 0,
        },
        NativeOpKind::ChannelsFirstLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 64, 256],
            channels: 64,
            leaky_relu_slope: None,
        },
        NativeOpKind::Int8Gemm {
            in_features: 768,
            out_features: 3072,
            has_bias: true,
            input_shape: vec![1, 128, 768],
        },
        NativeOpKind::Conv1dGemm {
            input_shape: vec![1, 128, 512],
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 128, 3072],
        },
        NativeOpKind::RotaryEmbedding {
            head_dim: 64,
            input_shape: vec![1, 32, 128, 64],
        },
        NativeOpKind::AddNormLinear {
            eps: 1e-5,
            input_shape: vec![1, 128, 768],
            hidden_dim: 768,
            out_features: 3072,
            has_bias: true,
        },
        NativeOpKind::MoeGating {
            num_experts: 8,
            top_k: 2,
            input_shape: vec![1, 128, 768],
        },
        NativeOpKind::FusedAdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 64, 256],
            channels: 64,
            external_node_ids: None,
        },
        NativeOpKind::FusedUpsampleConv1d {
            upsample_factor: 2,
            in_channels: 64,
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_shape: vec![1, 64, 128],
        },
        NativeOpKind::BiLstmCat {
            hidden_size: 256,
            input_shape: vec![100, 1, 512],
            h_shape: vec![1, 256],
            fwd_lstm_step: 0,
            rev_lstm_step: 1,
        },
        NativeOpKind::FusedMulAdd {
            input_shape: vec![1, 128, 768],
        },
        NativeOpKind::FusedSiGLU {
            input_shape: vec![1, 128, 768],
        },
        NativeOpKind::FusedGeGLU {
            input_shape: vec![1, 128, 3072],
        },
        NativeOpKind::FusedLayerNormLinear {
            eps: 1e-5,
            input_shape: vec![1, 128, 768],
            hidden_dim: 768,
            out_features: 3072,
            has_bias: true,
        },
        NativeOpKind::FusedInstanceNormMulAdd {
            eps: 1e-5,
            input_shape: vec![1, 64, 256],
            channels: 64,
            external_node_ids: None,
        },
    ];

    // Collect all variant names into a set.
    let names: HashSet<&str> = all_ops.iter().map(NativeOpKind::variant_name).collect();

    // Verify all expected variant names are present.
    let expected_names = [
        "LstmSequence",
        "Cumsum",
        "InstanceNorm",
        "LayerNorm",
        "AddLayerNorm",
        "AdainSnake",
        "AdainLeakyRelu",
        "AdaLayerNorm",
        "FlashAttention",
        "MaxPool1d",
        "ConstantWeight",
        "FusedResBlock",
        "NormActivConv1d",
        "LinearActivation",
        "BatchedLinearProjection",
        "ProjectionSlice",
        "NormLinear",
        "BatchedStyleProjection",
        "ChannelsFirstLayerNorm",
        "Int8Gemm",
        "Conv1dGemm",
        "SiluMul",
        "RotaryEmbedding",
        "AddNormLinear",
        "MoeGating",
        "FusedAdainSnake",
        "FusedUpsampleConv1d",
        "BiLstmCat",
        "FusedMulAdd",
        "FusedSiGLU",
        "FusedGeGLU",
        "FusedLayerNormLinear",
        "FusedInstanceNormMulAdd",
    ];

    for name in &expected_names {
        assert!(
            names.contains(name),
            "Missing NativeOpKind variant: {name}"
        );
    }

    // Verify total variant count is at least 31 (the documented count).
    assert!(
        all_ops.len() >= 31,
        "Expected at least 31 NativeOpKind variants, got {}",
        all_ops.len()
    );
    assert_eq!(
        names.len(),
        expected_names.len(),
        "Variant name count should match expected count"
    );
}

/// NativeOpKind::variant_name() returns distinct names for each variant.
#[test]
fn native_op_kind_variant_names_are_unique() {
    let ops = [
        NativeOpKind::LstmSequence {
            hidden_size: 256,
            input_shape: vec![10, 1, 512],
            h_shape: vec![1, 256],
            reverse: false,
        },
        NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![1, 10],
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 8, 64, 64],
            k_shape: vec![1, 8, 64, 64],
            output_shape: vec![1, 8, 64, 64],
            input_layout: nn_dsl::AttentionLayout::default(),
        },
    ];

    let names: Vec<&str> = ops.iter().map(NativeOpKind::variant_name).collect();
    let unique: HashSet<&&str> = names.iter().collect();
    assert_eq!(
        names.len(),
        unique.len(),
        "All variant names should be unique"
    );
}

/// NativeOpKind is Clone and Debug.
#[test]
fn native_op_kind_clone_debug() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![1, 128, 3072],
    };
    let cloned = op;
    let dbg = format!("{cloned:?}");
    assert!(dbg.contains("SiluMul"), "Debug should include variant name");
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Kernel fusion opportunities — detection between adjacent steps
// ═══════════════════════════════════════════════════════════════════════

/// FusionOpportunity struct holds consistent field values.
#[test]
fn fusion_opportunity_struct_consistency() {
    let opp = FusionOpportunity {
        first_idx: 0,
        second_idx: 1,
        first_op: "sigmoid".into(),
        second_op: "mul".into(),
        combined_ns: 5000,
        saved_bytes: 4096,
    };

    assert_eq!(opp.first_idx, 0);
    assert_eq!(opp.second_idx, 1);
    assert_eq!(opp.first_op, "sigmoid");
    assert_eq!(opp.second_op, "mul");
    assert_eq!(opp.combined_ns, 5000);
    assert_eq!(opp.saved_bytes, 4096);
}

/// Adjacent elementwise dispatches are fusion candidates (sigmoid -> mul = SiGLU).
#[test]
fn fusion_opportunity_sigmoid_mul_is_siglu_pattern() {
    // Simulate profiled entries for sigmoid -> mul pattern.
    let entries = [
        DispatchProfileEntry {
            step_idx: 0,
            op_name: "sigmoid".into(),
            dispatch_type: DispatchType::StandardOp("sigmoid".into()),
            start_ns: 0,
            end_ns: 1000,
            input_bytes: 4096,
            output_bytes: 4096,
        },
        DispatchProfileEntry {
            step_idx: 1,
            op_name: "mul".into(),
            dispatch_type: DispatchType::StandardOp("mul".into()),
            start_ns: 1000,
            end_ns: 2000,
            input_bytes: 8192,
            output_bytes: 4096,
        },
    ];

    // The fusion opportunity eliminates the intermediate buffer.
    let opp = FusionOpportunity {
        first_idx: 0,
        second_idx: 1,
        first_op: entries[0].op_name.clone(),
        second_op: entries[1].op_name.clone(),
        combined_ns: entries[0].duration_ns() + entries[1].duration_ns(),
        saved_bytes: entries[0].output_bytes,
    };

    assert_eq!(opp.combined_ns, 2000);
    assert_eq!(opp.saved_bytes, 4096, "intermediate sigmoid output eliminated");
}

/// DispatchProfileEntry duration and bandwidth calculations.
#[test]
fn dispatch_profile_entry_duration_bandwidth() {
    let entry = DispatchProfileEntry {
        step_idx: 0,
        op_name: "matmul".into(),
        dispatch_type: DispatchType::StandardOp("matmul".into()),
        start_ns: 100,
        end_ns: 1100,
        input_bytes: 1024,
        output_bytes: 512,
    };

    assert_eq!(entry.duration_ns(), 1000);
    assert!((entry.duration_us() - 1.0).abs() < f64::EPSILON);
    assert_eq!(entry.total_bytes(), 1536);
    // bandwidth = 1536 / 1000 = 1.536 GB/s
    assert!((entry.bandwidth_gbps() - 1.536).abs() < 0.001);
}

/// Zero-duration entry has zero bandwidth (no division by zero).
#[test]
fn dispatch_profile_entry_zero_duration_no_panic() {
    let entry = DispatchProfileEntry {
        step_idx: 0,
        op_name: "noop".into(),
        dispatch_type: DispatchType::StandardOp("noop".into()),
        start_ns: 100,
        end_ns: 100,
        input_bytes: 1024,
        output_bytes: 512,
    };

    assert_eq!(entry.duration_ns(), 0);
    assert_eq!(entry.bandwidth_gbps(), 0.0);
}

/// DispatchType category and name accessors.
#[test]
fn dispatch_type_category_and_name() {
    let native = DispatchType::NativeOp("FlashAttention".into());
    assert_eq!(native.category(), "native_op");
    assert_eq!(native.name(), "FlashAttention");

    let fused = DispatchType::FusedKernel("sigmoid_mul".into());
    assert_eq!(fused.category(), "fused_kernel");
    assert_eq!(fused.name(), "sigmoid_mul");

    let standard = DispatchType::StandardOp("matmul".into());
    assert_eq!(standard.category(), "standard_op");
    assert_eq!(standard.name(), "matmul");
}

/// DispatchType Display output.
#[test]
fn dispatch_type_display() {
    let native = DispatchType::NativeOp("AdainSnake".into());
    assert_eq!(format!("{native}"), "NativeOp(AdainSnake)");

    let standard = DispatchType::StandardOp("conv1d".into());
    assert_eq!(format!("{standard}"), "StandardOp(conv1d)");
}

// ═══════════════════════════════════════════════════════════════════════
// 5. F16 autocast — compute-heavy op identification
// ═══════════════════════════════════════════════════════════════════════

/// Recommended config enables all compute-heavy segments.
#[test]
fn f16_autocast_recommended_enables_compute_heavy() {
    let config = F16AutocastConfig::recommended(MixedPrecisionPolicy::apple_silicon_default());

    // Compute-heavy segments that benefit from F16 bandwidth reduction.
    let compute_heavy = ["plbert", "text", "prosody", "f0", "generator", "sinegen_post"];
    for seg in &compute_heavy {
        assert!(
            config.policy_for_segment(seg).is_some(),
            "{seg} should be enabled in recommended config"
        );
    }

    // Lightweight elementwise segments — no significant bandwidth benefit.
    let lightweight = ["regulate", "sinegen_pre"];
    for seg in &lightweight {
        assert!(
            config.policy_for_segment(seg).is_none(),
            "{seg} should be disabled in recommended config"
        );
    }
}

/// Recommended config has exactly 6 segments enabled.
#[test]
fn f16_autocast_recommended_count() {
    let config = F16AutocastConfig::recommended(MixedPrecisionPolicy::apple_silicon_default());
    assert_eq!(config.enabled_count(), 6);
}

/// Generator-only config enables exactly 1 segment.
#[test]
fn f16_autocast_generator_only_count() {
    let config = F16AutocastConfig::generator_only(MixedPrecisionPolicy::apple_silicon_default());
    assert_eq!(config.enabled_count(), 1);
    assert!(config.generator);
    assert!(!config.plbert);
}

/// F16AutocastConfig builder chain produces expected result.
#[test]
fn f16_autocast_builder_chain() {
    let config = F16AutocastConfig::none(MixedPrecisionPolicy::apple_silicon_default())
        .with_generator(true)
        .with_plbert(true)
        .with_text(true);
    assert_eq!(config.enabled_count(), 3);
    assert!(config.policy_for_segment("generator").is_some());
    assert!(config.policy_for_segment("plbert").is_some());
    assert!(config.policy_for_segment("text").is_some());
    assert!(config.policy_for_segment("prosody").is_none());
}

/// All 8 segment names are recognized by policy_for_segment.
#[test]
fn f16_autocast_all_segment_names_recognized() {
    let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default());
    let all_segments = [
        "plbert",
        "text",
        "prosody",
        "f0",
        "generator",
        "regulate",
        "sinegen_pre",
        "sinegen_post",
    ];
    for seg in &all_segments {
        assert!(
            config.policy_for_segment(seg).is_some(),
            "Segment '{seg}' should be recognized when all() is used"
        );
    }
}

/// Unknown segment names always return None.
#[test]
fn f16_autocast_unknown_segment_returns_none() {
    let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default());
    assert!(config.policy_for_segment("unknown").is_none());
    assert!(config.policy_for_segment("").is_none());
    assert!(config.policy_for_segment("PLBERT").is_none()); // case sensitive
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Activation arena — allocation and buffer reuse tracking
// ═══════════════════════════════════════════════════════════════════════

/// ArenaEstimate from known step sizes computes correct peak and total.
#[test]
fn arena_estimate_peak_and_total() {
    use crate::arena::estimate_arena_peak_bytes;

    let steps = vec![1024, 2048, 512, 4096];
    let estimate = estimate_arena_peak_bytes(steps);

    // All steps are cumulative (bump allocator), so peak = sum of all
    // aligned allocations. Each allocation aligns to 256 bytes.
    assert!(
        estimate.peak_bytes > 0,
        "peak_bytes should be positive: {}",
        estimate.peak_bytes
    );
    assert!(
        estimate.total_bytes > 0,
        "total_bytes should be positive: {}",
        estimate.total_bytes
    );
    assert_eq!(estimate.step_count, 4, "4 allocation steps");
    assert!(
        estimate.peak_bytes >= 1024 + 2048 + 512 + 4096,
        "peak should be at least the sum of raw sizes (alignment adds padding)"
    );
}

/// ArenaEstimate with zero-size steps are skipped.
#[test]
fn arena_estimate_skips_zero_size_steps() {
    use crate::arena::estimate_arena_peak_bytes;

    let steps = vec![0, 1024, 0, 2048, 0];
    let estimate = estimate_arena_peak_bytes(steps);
    assert_eq!(
        estimate.step_count, 2,
        "zero-size steps should be skipped"
    );
}

/// ArenaEstimate from shapes produces consistent results.
#[test]
fn arena_estimate_from_shapes() {
    use crate::arena::estimate_arena_peak_from_shapes;

    let entries: Vec<(&str, &[usize], usize)> = vec![
        ("conv1d_out", &[1, 64, 256], 4),  // 64*256*4 = 65536
        ("relu_out", &[1, 64, 256], 4),     // same shape
        ("pool_out", &[1, 64, 128], 4),     // 64*128*4 = 32768
    ];

    let estimate = estimate_arena_peak_from_shapes(&entries);
    assert_eq!(estimate.step_count, 3);
    assert!(
        estimate.total_bytes >= 65536 + 65536 + 32768,
        "total should be at least raw sum: {}",
        estimate.total_bytes
    );
}

/// ArenaEstimate with empty input returns zero.
#[test]
fn arena_estimate_empty_returns_zero() {
    use crate::arena::estimate_arena_peak_bytes;

    let estimate = estimate_arena_peak_bytes(std::iter::empty());
    assert_eq!(estimate.peak_bytes, 0);
    assert_eq!(estimate.total_bytes, 0);
    assert_eq!(estimate.step_count, 0);
}

/// ArenaStats hit_rate calculation edge cases (via arena_stats() function).
#[test]
fn arena_stats_hit_rate_edge_cases() {
    use crate::arena::{arena_stats, reset_arena_stats};

    // Reset stats to a clean baseline.
    reset_arena_stats();
    let stats = arena_stats();
    // With no allocations, hit rate should be 0.0.
    assert_eq!(
        stats.hit_rate(),
        0.0,
        "zero allocations should give 0.0 hit rate"
    );
}

/// ArenaStats fresh_allocs is misses - pool.hits (tested via accessor).
#[test]
fn arena_stats_fresh_allocs_accessor() {
    use crate::arena::{arena_stats, reset_arena_stats};

    reset_arena_stats();
    let stats = arena_stats();
    // After reset, fresh_allocs = 0 - 0 = 0.
    assert_eq!(
        stats.fresh_allocs(),
        0,
        "fresh_allocs should be 0 after reset"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 7. Memory barrier planning — command batch barrier strategy
// ═══════════════════════════════════════════════════════════════════════

/// DispatchMode variants that use threadgroups (PerSliceReduction) indicate
/// barrier-requiring dispatches (separate command batch phase).
#[test]
fn barrier_strategy_reduction_requires_threadgroups() {
    let reduction = DispatchMode::PerSliceReduction {
        outer: 128,
        reduce: 768,
        threads: 256,
        shared_bytes: 1024,
    }
    .plan()
    .unwrap();

    assert!(
        reduction.use_threadgroups(),
        "Reduction dispatches use dispatch_threadgroups which requires barrier awareness"
    );
    assert!(
        reduction.threadgroup_memory_bytes().is_some(),
        "Reduction plans should declare shared memory"
    );
}

/// Elementwise dispatches do NOT require threadgroup barriers.
#[test]
fn barrier_strategy_elementwise_no_threadgroup_barrier() {
    let ew = DispatchMode::Elementwise { total: 1024 }
        .plan()
        .unwrap();

    assert!(
        !ew.use_threadgroups(),
        "Elementwise dispatches use dispatch_threads, no threadgroup barrier"
    );
    assert!(
        ew.threadgroup_memory_bytes().is_none(),
        "Elementwise plans should not declare shared memory"
    );
}

/// Grid2D dispatches do NOT require threadgroup barriers.
#[test]
fn barrier_strategy_grid2d_no_threadgroup_barrier() {
    let grid = DispatchMode::Grid2D {
        grid: [32, 16],
        threads: [8, 4],
    }
    .plan()
    .unwrap();

    assert!(!grid.use_threadgroups());
    assert!(grid.threadgroup_memory_bytes().is_none());
}

/// Mixed dispatch plan: count barrier and non-barrier steps.
#[test]
fn barrier_strategy_mixed_plan_counting() {
    let modes = [
        DispatchMode::Elementwise { total: 512 },
        DispatchMode::PerSliceReduction {
            outer: 64,
            reduce: 256,
            threads: 128,
            shared_bytes: 1024,
        },
        DispatchMode::Elementwise { total: 256 },
        DispatchMode::Grid2D {
            grid: [16, 8],
            threads: [4, 4],
        },
        DispatchMode::PerSliceReduction {
            outer: 32,
            reduce: 512,
            threads: 64,
            shared_bytes: 2048,
        },
    ];

    let plans: Vec<_> = modes.iter().map(|m| m.plan().unwrap()).collect();
    let barrier_count = plans.iter().filter(|p| p.use_threadgroups()).count();
    let non_barrier_count = plans.iter().filter(|p| !p.use_threadgroups()).count();

    assert_eq!(barrier_count, 2, "2 reductions require threadgroup dispatch");
    assert_eq!(non_barrier_count, 3, "3 non-reductions use dispatch_threads");

    // Total shared memory across barrier steps.
    let total_shared: u64 = plans
        .iter()
        .filter_map(crate::dispatch_plan::DispatchPlan::threadgroup_memory_bytes)
        .sum();
    assert_eq!(total_shared, 1024 + 2048, "total shared memory = 3072");
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Workgroup configuration — optimal workgroup size selection
// ═══════════════════════════════════════════════════════════════════════

/// Tiny matrices (M*N < 1024) fall back to scalar dispatch.
#[test]
fn workgroup_tiny_matrix_scalar_fallback() {
    // 1 x 640 = 640 < TINY_THRESHOLD (1024)
    assert!(
        select_gemm_tiles(1, 640, 256).is_none(),
        "LSTM recurrent step (M=1) should fall back to scalar"
    );
    // 16 x 16 = 256 < 1024
    assert!(
        select_gemm_tiles(16, 32, 16).is_none(),
        "16x16 output should fall back to scalar"
    );
}

/// Large square matrices get the standard 32x32 tile.
#[test]
fn workgroup_large_square_standard_tile() {
    let cfg = select_gemm_tiles(256, 768, 768).unwrap();
    assert_eq!(cfg.tile_m, 32);
    assert_eq!(cfg.tile_n, 32);
    assert_eq!(cfg.tile_k, 32);
    assert_eq!(cfg.threadgroup_size, [32, 4, 1]);
}

/// Tall-skinny matrices (M >> N) get the 64x16 tile.
#[test]
fn workgroup_tall_skinny_tile() {
    // M=256, N=32 -> M/N = 8 >= TALL_SKINNY_RATIO (4)
    // Both aligned to SIMDGROUP_ALIGN (8)
    let cfg = select_gemm_tiles(256, 768, 32).unwrap();
    assert_eq!(cfg.tile_m, 64, "tall-skinny should use 64 M-rows");
    assert_eq!(cfg.tile_n, 16, "tall-skinny should use 16 N-cols");
}

/// Wide matrices (N >> M) get the 16x64 tile.
#[test]
fn workgroup_wide_tile() {
    // M=32, N=256 -> N/M = 8 >= WIDE_RATIO (4)
    let cfg = select_gemm_tiles(32, 768, 256).unwrap();
    assert_eq!(cfg.tile_m, 16, "wide should use 16 M-rows");
    assert_eq!(cfg.tile_n, 64, "wide should use 64 N-cols");
}

/// Small K gets a 32x32 tile with tile_k clamped to K (rounded to alignment).
#[test]
fn workgroup_small_k_tile() {
    // K=16, which is <= SMALL_K_THRESHOLD (32)
    let cfg = select_gemm_tiles(64, 16, 64).unwrap();
    assert_eq!(cfg.tile_m, 32);
    assert_eq!(cfg.tile_n, 32);
    assert_eq!(
        cfg.tile_k, 16,
        "tile_k should be K rounded up to SIMDGROUP_ALIGN"
    );
}

/// TileConfig::output_per_threadgroup is tile_m * tile_n.
#[test]
fn tile_config_output_per_threadgroup() {
    assert_eq!(TileConfig::SQUARE.output_per_threadgroup(), 32 * 32);
    assert_eq!(TileConfig::TALL_SKINNY.output_per_threadgroup(), 64 * 16);
    assert_eq!(TileConfig::WIDE.output_per_threadgroup(), 16 * 64);
    // All configs produce 1024 output elements per TG.
    assert_eq!(TileConfig::SQUARE.output_per_threadgroup(), 1024);
    assert_eq!(TileConfig::TALL_SKINNY.output_per_threadgroup(), 1024);
    assert_eq!(TileConfig::WIDE.output_per_threadgroup(), 1024);
}

/// TileConfig::threadgroup_count for various output sizes.
#[test]
fn tile_config_threadgroup_count() {
    // 256x256 with 32x32 tiles -> 8x8 = 64 threadgroups
    assert_eq!(TileConfig::SQUARE.threadgroup_count(256, 256), 64);
    // 256x64 with 64x16 tiles -> 4x4 = 16 threadgroups
    assert_eq!(TileConfig::TALL_SKINNY.threadgroup_count(256, 64), 16);
    // 64x256 with 16x64 tiles -> 4x4 = 16 threadgroups
    assert_eq!(TileConfig::WIDE.threadgroup_count(64, 256), 16);
}

/// TileConfig::threads_per_threadgroup is always 128 for standard configs.
#[test]
fn tile_config_threads_per_threadgroup() {
    assert_eq!(TileConfig::SQUARE.threads_per_threadgroup(), 128);
    assert_eq!(TileConfig::TALL_SKINNY.threads_per_threadgroup(), 128);
    assert_eq!(TileConfig::WIDE.threads_per_threadgroup(), 128);
}

/// Alignment constants are consistent.
#[test]
fn workgroup_alignment_constants() {
    assert_eq!(SIMDGROUP_ALIGN, 8);
    assert_eq!(TINY_THRESHOLD, 1024);
    assert_eq!(STANDARD_MN_THRESHOLD, 16_384);
    assert_eq!(SMALL_K_THRESHOLD, 32);
    assert_eq!(TALL_SKINNY_RATIO, 4);
    assert_eq!(WIDE_RATIO, 4);
}

/// threadgroup_width_1d correctly caps at 64 for various inputs.
#[test]
fn workgroup_threadgroup_width_1d_coverage() {
    // Below 64: identity
    assert_eq!(threadgroup_width_1d(1), 1);
    assert_eq!(threadgroup_width_1d(32), 32);
    assert_eq!(threadgroup_width_1d(63), 63);
    // At and above 64: capped
    assert_eq!(threadgroup_width_1d(64), 64);
    assert_eq!(threadgroup_width_1d(128), 64);
    assert_eq!(threadgroup_width_1d(1_000_000), 64);
}
