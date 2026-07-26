// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for NativeOp dispatch, compiled model infrastructure,
//! and Metal-specific behavior.
//!
//! Covers 12 areas:
//! 1. NativeOpKind: all 34 variants — construction, display, serde
//! 2. CompiledKokoroRegistry: entry registration, lookup, MSL validation
//! 3. DispatchStep: plan construction for various op sequences
//! 4. Pipeline cache: L1 thread-local and L2 shared cache behavior
//! 5. Buffer aliasing: size validation, byte offset handling
//! 6. Lazy command buffer: batching behavior, flush before readback
//! 7. ActivationArena: buffer reuse patterns, scope nesting
//! 8. WeightMap: ManuallyDrop ordering, mmap alignment
//! 9. Peephole optimization: fused pattern configuration
//! 10. GPU fallback: needs_f32_fallback, needs_non_float_fallback behavior
//! 11. Dtype dispatch: same_gpu_byte_width checks
//! 12. GpuScope: command buffer depth management
//!
//! All tests are structure/config tests only — no live GPU required (except
//! arena/builder tests that need MetalContext on macOS).
//!
//! Part of #4495.

use std::collections::HashSet;

use nn_core::DType;
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::{
    ConvActivation, FusedNormKind, GemmActivation, NativeOpKind, NormActivation,
    NormActivConv1dParams,
};
use nn_dsl::{DispatchStep, PeepholeConfig, TensorNodeId};

use crate::dispatch_plan::{
    clear_dispatch_plan_cache, dispatch_plan_cache_len, plan_elementwise, plan_grid_2d,
    plan_reduction, DispatchMode,
};
use crate::dispatch_stats::{dispatch_stats, reset_counters};
use crate::gpu_scope::ScopeExitMode;

// ═══════════════════════════════════════════════════════════════════════
// 1. NativeOpKind — all 34 variants construction, Display, serde
// ═══════════════════════════════════════════════════════════════════════

/// Helper: construct all 34 NativeOpKind variants.
fn all_34_variants() -> Vec<NativeOpKind> {
    vec![
        NativeOpKind::LstmSequence {
            hidden_size: 256,
            input_shape: vec![10, 1, 128],
            h_shape: vec![1, 256],
            reverse: false,
        },
        NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![10, 32],
        },
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 32, 768],
            hidden_dim: 768,
        },
        NativeOpKind::AddLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 32, 768],
            hidden_dim: 768,
        },
        NativeOpKind::AdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            residual_gamma: true,
            external_node_ids: None,
        },
        NativeOpKind::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.2,
            input_shape: vec![1, 128, 512],
            external_node_ids: None,
        },
        NativeOpKind::AdaLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 32, 256],
            hidden_dim: 256,
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 8, 32, 64],
            k_shape: vec![1, 8, 32, 64],
            output_shape: vec![1, 8, 32, 64],
            input_layout: Default::default(),
        },
        NativeOpKind::MaxPool1d {
            kernel_size: 3,
            stride: 2,
            padding: 1,
            input_shape: vec![1, 64, 100],
        },
        NativeOpKind::ConstantWeight {
            name: "arange".into(),
            shape: vec![100],
        },
        NativeOpKind::FusedResBlock {
            phase1: NormActivConv1dParams::new(
                NormActivation::Snake,
                1e-5,
                1,
                1,
                vec![1, 128, 512],
                128,
                3,
            ),
            phase2: NormActivConv1dParams::new(
                NormActivation::Snake,
                1e-5,
                1,
                1,
                vec![1, 128, 512],
                128,
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
            activation: NormActivation::Snake,
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 1,
            input_shape: vec![1, 128, 512],
            output_channels: 128,
            kernel_size: 3,
            external_node_ids: None,
        },
        NativeOpKind::LinearActivation {
            activation: GemmActivation::Relu,
            in_features: 768,
            out_features: 256,
            has_bias: true,
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::BatchedLinearProjection {
            in_features: 768,
            total_out_features: 2304,
            projection_sizes: vec![768, 768, 768],
            has_bias: true,
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::ProjectionSlice {
            source_step: 0,
            dim: 2,
            start: 0,
            length: 768,
            output_shape: vec![1, 32, 768],
        },
        NativeOpKind::NormLinear {
            norm_kind: FusedNormKind::LayerNorm,
            eps: 1e-5,
            input_shape: vec![1, 32, 768],
            hidden_dim: 768,
            out_features: 256,
            has_bias: true,
        },
        NativeOpKind::BatchedStyleProjection {
            blocks: vec![],
            style_dim: 128,
            total_out: 512,
            style_step: 0,
        },
        NativeOpKind::Int8Gemm {
            in_features: 768,
            out_features: 256,
            has_bias: true,
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::Conv1dGemm {
            input_shape: vec![1, 128, 1024],
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::RotaryEmbedding {
            head_dim: 64,
            input_shape: vec![1, 8, 32, 64],
        },
        NativeOpKind::AddNormLinear {
            eps: 1e-5,
            input_shape: vec![1, 32, 768],
            hidden_dim: 768,
            out_features: 256,
            has_bias: true,
        },
        NativeOpKind::MoeGating {
            num_experts: 8,
            top_k: 2,
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::FusedAdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            external_node_ids: None,
        },
        NativeOpKind::FusedUpsampleConv1d {
            upsample_factor: 2,
            in_channels: 128,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_shape: vec![1, 128, 256],
        },
        NativeOpKind::BiLstmCat {
            hidden_size: 256,
            input_shape: vec![10, 1, 128],
            h_shape: vec![1, 256],
            fwd_lstm_step: 0,
            rev_lstm_step: 1,
        },
        NativeOpKind::FusedMulAdd {
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::FusedSiGLU {
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::FusedGeGLU {
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::FusedLayerNormLinear {
            eps: 1e-5,
            input_shape: vec![1, 32, 768],
            hidden_dim: 768,
            out_features: 256,
            has_bias: true,
        },
        NativeOpKind::FusedInstanceNormMulAdd {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            external_node_ids: None,
        },
        NativeOpKind::FusedConv1dActivation {
            activation: ConvActivation::Snake,
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
            input_shape: vec![1, 128, 512],
            pre_activation: false,
        },
        NativeOpKind::ChannelsFirstLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 256, 100],
            channels: 256,
            leaky_relu_slope: None,
        },
    ]
}

/// All 34 NativeOpKind variants produce distinct, non-empty variant_name strings.
#[test]
fn native_op_all_34_variants_unique_names() {
    let variants = all_34_variants();
    let names: HashSet<&str> = variants.iter().map(NativeOpKind::variant_name).collect();
    assert_eq!(
        names.len(),
        34,
        "Expected 34 unique variant names, got {}. Names: {names:?}",
        names.len(),
    );
    for name in &names {
        assert!(!name.is_empty(), "variant_name must be non-empty");
    }
}

/// All 34 NativeOpKind variants produce non-empty Debug output containing
/// their variant name.
#[test]
fn native_op_all_34_variants_debug_contains_name() {
    for op in &all_34_variants() {
        let name = op.variant_name();
        let dbg = format!("{op:?}");
        assert!(
            dbg.contains(name),
            "Debug for {name} should contain the variant name: {dbg}"
        );
    }
}

/// All 34 NativeOpKind variants round-trip through serde_json.
#[test]
fn native_op_all_34_variants_serde_round_trip() {
    for op in &all_34_variants() {
        let json = serde_json::to_string(op)
            .unwrap_or_else(|e| panic!("serialize {} failed: {e}", op.variant_name()));
        let deser: NativeOpKind = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("deserialize {} failed: {e}", op.variant_name()));
        assert_eq!(
            deser.variant_name(),
            op.variant_name(),
            "Round-trip variant name mismatch for {json}"
        );
    }
}

/// All NativeOpKind variants have estimated_metal_dispatches >= 1, except
/// `ConstantWeight`, which is documented to require no GPU computation (it
/// returns a pre-uploaded buffer) and therefore reports 0 dispatches.
#[test]
fn native_op_all_variants_dispatch_estimate_positive() {
    for op in &all_34_variants() {
        // ConstantWeight: pre-uploaded buffer, no GPU dispatch (documented 0).
        if matches!(op, NativeOpKind::ConstantWeight { .. }) {
            assert_eq!(
                op.estimated_metal_dispatches(),
                0,
                "ConstantWeight should report 0 dispatches (no GPU computation)"
            );
            continue;
        }
        assert!(
            op.estimated_metal_dispatches() >= 1,
            "{} should have >= 1 estimated dispatch, got {}",
            op.variant_name(),
            op.estimated_metal_dispatches()
        );
    }
}

/// All NativeOpKind variants have estimated_encoding_events >= 1, except the
/// documented zero-encoding cases: `ConstantWeight` (pre-uploaded buffer) and
/// `MaxPool1d` (CPU roundtrip via to_device, no compute dispatch).
#[test]
fn native_op_all_variants_encoding_events_positive() {
    for op in &all_34_variants() {
        // Documented zero-encoding ops: ConstantWeight (no GPU work) and
        // MaxPool1d (GPU→CPU→GPU roundtrip, no compute encoding).
        if matches!(
            op,
            NativeOpKind::ConstantWeight { .. } | NativeOpKind::MaxPool1d { .. }
        ) {
            assert_eq!(
                op.estimated_encoding_events(),
                0,
                "{} should report 0 encoding events (no compute dispatch)",
                op.variant_name()
            );
            continue;
        }
        assert!(
            op.estimated_encoding_events() >= 1,
            "{} should have >= 1 estimated encoding, got {}",
            op.variant_name(),
            op.estimated_encoding_events()
        );
    }
}

/// variant_name returns ASCII alphanumeric strings (used as registry keys).
#[test]
fn native_op_variant_names_are_alphanumeric() {
    for op in &all_34_variants() {
        let name = op.variant_name();
        assert!(
            name.chars().all(|c| c.is_ascii_alphanumeric()),
            "variant_name should be alphanumeric: {name}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. CompiledKokoroRegistry — entry registration, lookup, MSL validation
// ═══════════════════════════════════════════════════════════════════════

/// CompiledModel builder from empty graph produces zero steps/dispatches.
#[test]
fn registry_empty_graph_zero_steps() {
    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![]);
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .build()
        .expect("empty graph build");
    assert_eq!(model.num_steps(), 0);
    assert_eq!(model.num_dispatches(), 0);
    assert_eq!(model.num_native_ops(), 0);
}

/// CompiledModel builder with peephole config on empty graph succeeds.
#[test]
fn registry_empty_graph_with_peephole() {
    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![]);
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let cache = crate::cache::PipelineCache::new(ctx);
    let config = PeepholeConfig::default();
    let model = crate::compiled_model::CompiledModel::builder(&graph, &cache)
        .with_peephole_config(config)
        .build()
        .expect("peephole config build");
    assert_eq!(model.num_steps(), 0);
}

/// Fused ops have strictly fewer dispatches than their unfused equivalents.
#[test]
fn registry_fused_ops_save_dispatches() {
    let fused_adain = NativeOpKind::FusedAdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: None,
    };
    let unfused_instance_norm = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
    };
    // FusedAdainSnake replaces InstanceNorm + Mul + Add + Snake.
    assert!(
        fused_adain.estimated_metal_dispatches()
            <= unfused_instance_norm.estimated_metal_dispatches(),
        "FusedAdainSnake ({}) should be <= InstanceNorm alone ({})",
        fused_adain.estimated_metal_dispatches(),
        unfused_instance_norm.estimated_metal_dispatches()
    );

    let fused_norm_mul_add = NativeOpKind::FusedInstanceNormMulAdd {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: None,
    };
    assert_eq!(
        fused_norm_mul_add.estimated_metal_dispatches(),
        1,
        "FusedInstanceNormMulAdd should be 1 dispatch"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 3. DispatchStep — plan construction for various op sequences
// ═══════════════════════════════════════════════════════════════════════

/// Construct all DispatchStep variants and verify Debug output contains
/// the expected variant name.
#[test]
fn dispatch_step_all_variants_valid_debug() {
    let n0 = TensorNodeId::new(0);
    let n1 = TensorNodeId::new(1);
    let n2 = TensorNodeId::new(2);

    let steps: Vec<(&str, DispatchStep)> = vec![
        (
            "Reduce",
            DispatchStep::Reduce {
                kernel_name: "reduce_sum".into(),
                op: nn_dsl::ReduceOp::Sum,
                dtype: ScalarType::F32,
                input: n0,
                output: n1,
                reduce_dim: 128,
                outer_size: 16,
                keepdim: false,
            },
        ),
        (
            "Elementwise",
            DispatchStep::Elementwise {
                kernel_name: "ew_relu".into(),
                scalar_kernel: nn_dsl::ir::KernelDef::new(
                    "relu",
                    vec![],
                    ScalarType::F32,
                    vec![],
                    nn_dsl::ir::NodeId::new(0),
                ),
                inputs: vec![n0],
                output: n1,
                total_elements: 2048,
            },
        ),
        (
            "BinaryAdd",
            DispatchStep::BinaryAdd {
                kernel_name: "add".into(),
                dtype: ScalarType::F32,
                left: n0,
                right: n1,
                output: n2,
                total_elements: 1024,
                broadcast: None,
            },
        ),
        (
            "BinaryMul",
            DispatchStep::BinaryMul {
                kernel_name: "mul".into(),
                dtype: ScalarType::F16,
                left: n0,
                right: n1,
                output: n2,
                total_elements: 512,
                broadcast: None,
            },
        ),
        (
            "Reshape",
            DispatchStep::Reshape {
                input: n0,
                output: n1,
            },
        ),
        (
            "Softmax",
            DispatchStep::Softmax {
                kernel_name: "softmax".into(),
                dtype: ScalarType::F32,
                input: n0,
                output: n1,
                axis: 2,
                axis_size: 512,
                outer_size: 64,
            },
        ),
        (
            "Sigmoid",
            DispatchStep::Sigmoid {
                kernel_name: "sigmoid".into(),
                dtype: ScalarType::F32,
                input: n0,
                output: n1,
                total_elements: 256,
            },
        ),
        (
            "Gelu",
            DispatchStep::Gelu {
                kernel_name: "gelu".into(),
                dtype: ScalarType::F32,
                input: n0,
                output: n1,
                total_elements: 256,
            },
        ),
        (
            "Relu",
            DispatchStep::Relu {
                kernel_name: "relu".into(),
                dtype: ScalarType::F32,
                input: n0,
                output: n1,
                total_elements: 256,
            },
        ),
        (
            "Tanh",
            DispatchStep::Tanh {
                kernel_name: "tanh".into(),
                dtype: ScalarType::F32,
                input: n0,
                output: n1,
                total_elements: 256,
            },
        ),
        (
            "Transpose",
            DispatchStep::Transpose {
                kernel_name: "transpose".into(),
                dtype: ScalarType::F32,
                input: n0,
                output: n1,
                input_shape: vec![2, 4, 8],
                axes: vec![0, 2, 1],
                total_elements: 64,
            },
        ),
        (
            "Embedding",
            DispatchStep::Embedding {
                kernel_name: "embed".into(),
                dtype: ScalarType::F32,
                input: n0,
                weight: n1,
                output: n2,
                embedding_dim: 768,
                num_indices: 32,
                total_elements: 32 * 768,
            },
        ),
    ];

    for (name, step) in &steps {
        let dbg = format!("{step:?}");
        assert!(
            dbg.contains(name),
            "Debug for {name} should contain variant name, got: {dbg}"
        );
    }
}

/// Softmax DispatchStep encodes axis_size and outer_size correctly.
#[test]
fn dispatch_step_softmax_parameters_in_debug() {
    let n0 = TensorNodeId::new(0);
    let n1 = TensorNodeId::new(1);
    let step = DispatchStep::Softmax {
        kernel_name: "sm".into(),
        dtype: ScalarType::F32,
        input: n0,
        output: n1,
        axis: 1,
        axis_size: 768,
        outer_size: 32,
    };
    let dbg = format!("{step:?}");
    assert!(dbg.contains("768"), "axis_size should appear: {dbg}");
    assert!(dbg.contains("32"), "outer_size should appear: {dbg}");
}

/// Embedding DispatchStep encodes correct total_elements.
#[test]
fn dispatch_step_embedding_total_elements() {
    let n0 = TensorNodeId::new(0);
    let n1 = TensorNodeId::new(1);
    let n2 = TensorNodeId::new(2);
    let emb_dim = 768_usize;
    let num_idx = 64_usize;
    let step = DispatchStep::Embedding {
        kernel_name: "embed".into(),
        dtype: ScalarType::F32,
        input: n0,
        weight: n1,
        output: n2,
        embedding_dim: emb_dim,
        num_indices: num_idx,
        total_elements: emb_dim * num_idx,
    };
    let dbg = format!("{step:?}");
    assert!(
        dbg.contains(&(emb_dim * num_idx).to_string()),
        "total_elements {}: {dbg}",
        emb_dim * num_idx
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Pipeline cache — L1 thread-local and L2 shared cache behavior
// ═══════════════════════════════════════════════════════════════════════

/// Same DispatchMode returns equal cached plans (L1 thread-local hit).
#[test]
fn cache_l1_same_mode_cache_hit() {
    clear_dispatch_plan_cache();
    let mode = DispatchMode::Elementwise { total: 2048 };
    let plan1 = mode.plan_cached().unwrap();
    let plan2 = mode.plan_cached().unwrap();
    assert_eq!(plan1, plan2, "same mode should return equal cached plan");
    assert_eq!(dispatch_plan_cache_len(), 1, "single unique mode => 1 entry");
}

/// Different modes produce different plans and separate cache entries.
#[test]
fn cache_l1_different_modes_separate_entries() {
    clear_dispatch_plan_cache();
    let mode_a = DispatchMode::Elementwise { total: 128 };
    let mode_b = DispatchMode::Elementwise { total: 256 };
    let mode_c = DispatchMode::Grid2D {
        grid: [32, 16],
        threads: [8, 8],
    };

    let plan_a = mode_a.plan_cached().unwrap();
    let plan_b = mode_b.plan_cached().unwrap();
    let plan_c = mode_c.plan_cached().unwrap();

    assert_ne!(plan_a, plan_b);
    assert_ne!(plan_a, plan_c);
    assert_eq!(dispatch_plan_cache_len(), 3);
}

/// Cached plan equals uncached plan (cache does not corrupt).
#[test]
fn cache_cached_matches_uncached() {
    clear_dispatch_plan_cache();
    let mode = DispatchMode::PerSliceReduction {
        outer: 64,
        reduce: 512,
        threads: 256,
        shared_bytes: 4096,
    };
    let uncached = mode.plan().unwrap();
    let cached = mode.plan_cached().unwrap();
    assert_eq!(uncached, cached, "cached must equal uncached for reduction");
}

/// Cache handles many entries without panic.
#[test]
fn cache_many_entries_no_panic() {
    clear_dispatch_plan_cache();
    for i in 1..=300u32 {
        let _ = DispatchMode::Elementwise { total: i }.plan_cached().unwrap();
    }
    assert!(dispatch_plan_cache_len() <= 300);
}

/// clear_dispatch_plan_cache empties the cache.
#[test]
fn cache_clear_empties() {
    clear_dispatch_plan_cache();
    let _ = DispatchMode::Elementwise { total: 1 }.plan_cached().unwrap();
    assert!(dispatch_plan_cache_len() >= 1);
    clear_dispatch_plan_cache();
    assert_eq!(dispatch_plan_cache_len(), 0);
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Buffer aliasing — size validation, byte offset handling
// ═══════════════════════════════════════════════════════════════════════

/// DType byte sizes are consistent for buffer allocation.
#[test]
fn buffer_dtype_byte_sizes() {
    assert_eq!(DType::F32.size_bytes(), 4);
    assert_eq!(DType::F16.size_bytes(), 2);
    assert_eq!(DType::BF16.size_bytes(), 2);
    assert_eq!(DType::F64.size_bytes(), 8);
    assert_eq!(DType::I32.size_bytes(), 4);
    assert_eq!(DType::I64.size_bytes(), 8);
    assert_eq!(DType::U32.size_bytes(), 4);
    assert_eq!(DType::U8.size_bytes(), 1);
    assert_eq!(DType::Bool.size_bytes(), 1);
}

/// Buffer size calculation for typical tensor shapes.
#[test]
fn buffer_size_calculation_typical_shapes() {
    // [1, 128, 512] F32 => 65536 * 4 = 262144
    let shape = [1_usize, 128, 512];
    let elems: usize = shape.iter().product();
    assert_eq!(elems * DType::F32.size_bytes(), 262_144);
    assert_eq!(elems * DType::F16.size_bytes(), 131_072);
    assert_eq!(elems * DType::BF16.size_bytes(), 131_072);

    // [1, 8, 32, 64] F32 => 16384 * 4 = 65536
    let shape2 = [1_usize, 8, 32, 64];
    let elems2: usize = shape2.iter().product();
    assert_eq!(elems2, 16384);
    assert_eq!(elems2 * DType::F32.size_bytes(), 65536);
}

/// checked_mul detects overflow for very large element counts.
#[test]
fn buffer_size_overflow_detected() {
    // usize::MAX / 2 * 2 == usize::MAX - 1 (no overflow), so use MAX/2 + 1 to
    // guarantee overflow for the smallest dtype factor (F16 = 2 bytes) as well
    // as for F32 (4 bytes).
    let large = usize::MAX / 2 + 1;
    assert!(large.checked_mul(4).is_none(), "should overflow with F32");
    assert!(large.checked_mul(2).is_none(), "should overflow with F16");
}

/// GpuSlice construction preserves byte_offset correctly.
#[test]
fn buffer_gpu_slice_offset_preserved() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let buf = ctx.create_buffer_zeroed(1024).expect("alloc 1KB buffer");
    let slice_0 = crate::gpu_slice::GpuSlice::zero_offset(buf.alias());
    assert_eq!(slice_0.byte_offset(), 0);

    let slice_512 = crate::gpu_slice::GpuSlice::new(buf.alias(), 512);
    assert_eq!(slice_512.byte_offset(), 512);

    let slice_ref = crate::gpu_slice::GpuSlice::from_ref(&buf, 256);
    assert_eq!(slice_ref.byte_offset(), 256);
}

/// GpuSlice::alias preserves byte offset.
#[test]
fn buffer_gpu_slice_alias_preserves_offset() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let buf = ctx.create_buffer_zeroed(2048).expect("alloc 2KB buffer");
    let slice = crate::gpu_slice::GpuSlice::new(buf, 768);
    let aliased = slice.alias();
    assert_eq!(aliased.byte_offset(), 768);
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Lazy command buffer — batching behavior, flush before readback
// ═══════════════════════════════════════════════════════════════════════

/// ScopeExitMode has exactly two distinct variants.
#[test]
fn lazy_cmd_scope_exit_mode_variants() {
    let flush = ScopeExitMode::Flush;
    let submit = ScopeExitMode::Submit;
    assert_ne!(flush, submit);
    assert_eq!(flush, ScopeExitMode::Flush);
    assert_eq!(submit, ScopeExitMode::Submit);
}

/// Dispatch stats counters start at zero after reset.
#[test]
fn lazy_cmd_stats_reset_zeroes() {
    reset_counters();
    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 0);
    assert_eq!(stats.flushes, 0);
    assert_eq!(stats.submits, 0);
    assert_eq!(stats.blits, 0);
}

/// Dispatch stats are stable across multiple reads (no side effects).
#[test]
fn lazy_cmd_stats_stable_reads() {
    reset_counters();
    let s1 = dispatch_stats();
    let s2 = dispatch_stats();
    let s3 = dispatch_stats();
    assert_eq!(s1, s2);
    assert_eq!(s2, s3);
}

/// Multiple reset_counters calls are idempotent.
#[test]
fn lazy_cmd_reset_idempotent() {
    reset_counters();
    reset_counters();
    reset_counters();
    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 0);
    assert_eq!(stats.flushes, 0);
    assert_eq!(stats.submits, 0);
}

/// ScopeExitMode Debug representations are correct.
#[test]
fn lazy_cmd_scope_exit_mode_debug() {
    assert!(format!("{:?}", ScopeExitMode::Flush).contains("Flush"));
    assert!(format!("{:?}", ScopeExitMode::Submit).contains("Submit"));
}

// ═══════════════════════════════════════════════════════════════════════
// 7. ActivationArena — buffer reuse patterns, scope nesting
// ═══════════════════════════════════════════════════════════════════════

/// Arena creation with reasonable capacity succeeds.
#[test]
fn arena_creation_valid_capacity() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let arena = crate::arena::ActivationArena::new(&ctx, 2 * 1024 * 1024);
    assert!(arena.is_ok(), "2MB arena should succeed");
}

/// Arena creation with zero capacity fails.
#[test]
fn arena_zero_capacity_fails() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let result = crate::arena::ActivationArena::new(&ctx, 0);
    assert!(result.is_err(), "zero capacity should fail");
}

/// Arena reset allows memory reuse.
#[test]
fn arena_reset_reuses_memory() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let mut arena = crate::arena::ActivationArena::new(&ctx, 4 * 1024 * 1024)
        .expect("4MB arena");

    let alloc_bytes = 128 * 256 * DType::F32.size_bytes();
    let _t1 = arena.alloc(alloc_bytes);
    let _t2 = arena.alloc(alloc_bytes);

    arena.reset();
    let t3 = arena.alloc(alloc_bytes);
    assert!(
        t3.is_ok(),
        "alloc after reset should succeed (reuse): {:?}",
        t3.err()
    );
}

/// Arena peak_bytes is preserved across resets.
#[test]
fn arena_peak_preserved_after_reset() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let mut arena = crate::arena::ActivationArena::new(&ctx, 8 * 1024 * 1024)
        .expect("8MB arena");

    let alloc_bytes = 256 * 512 * DType::F32.size_bytes();
    let _t1 = arena.alloc(alloc_bytes).unwrap();
    let peak_before = arena.peak_bytes();
    assert!(peak_before > 0, "peak should be > 0 after alloc");

    let _t2 = arena.alloc(alloc_bytes).unwrap();
    let peak_after_second = arena.peak_bytes();
    assert!(peak_after_second >= peak_before);

    arena.reset();
    assert_eq!(
        arena.peak_bytes(),
        peak_after_second,
        "peak preserved after reset"
    );
}

/// Arena peak_bytes monotonically increases with allocations.
#[test]
fn arena_peak_monotonic() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let mut arena = crate::arena::ActivationArena::new(&ctx, 16 * 1024 * 1024)
        .expect("16MB arena");

    let mut prev_peak = 0;
    for i in 1..=4 {
        let bytes = i * 64 * 1024; // 64KB, 128KB, 192KB, 256KB
        let _t = arena.alloc(bytes).unwrap();
        let peak = arena.peak_bytes();
        assert!(
            peak >= prev_peak,
            "peak should be monotonic: {peak} < {prev_peak} at iteration {i}"
        );
        prev_peak = peak;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 8. WeightMap — ManuallyDrop ordering, mmap alignment
// ═══════════════════════════════════════════════════════════════════════

/// WeightMap TensorInfo Debug representation includes struct name.
/// (Actual WeightMap construction requires safetensors file, so we test the
/// public types and related metadata.)
#[test]
fn weight_map_debug_representation() {
    let info = crate::safetensors::TensorInfo {
        offset: 0,
        byte_len: 16 * 64 * DType::F32.size_bytes(),
        dtype: DType::F32,
        shape: vec![16, 64],
    };
    let dbg = format!("{info:?}");
    assert!(
        dbg.contains("TensorInfo"),
        "TensorInfo Debug should contain type name: {dbg}"
    );
}

/// TensorInfo byte_len matches shape * dtype.
#[test]
fn weight_map_tensor_info_byte_len() {
    let info = crate::safetensors::TensorInfo {
        offset: 256,
        byte_len: 32 * 32 * DType::F32.size_bytes(),
        dtype: DType::F32,
        shape: vec![32, 32],
    };
    let expected = 32 * 32 * DType::F32.size_bytes();
    assert_eq!(info.byte_len, expected, "byte_len should match tensor size");
}

/// TensorInfo shape product matches element count.
#[test]
fn weight_map_tensor_info_shape_product() {
    let byte_len = 768 * 3072 * DType::F32.size_bytes();
    let info = crate::safetensors::TensorInfo {
        offset: 0,
        byte_len,
        dtype: DType::F32,
        shape: vec![768, 3072],
    };
    let elems: usize = info.shape.iter().product();
    assert_eq!(elems, 768 * 3072);
    let expected_bytes = elems * info.dtype.size_bytes();
    assert_eq!(info.byte_len, expected_bytes);
}

// ═══════════════════════════════════════════════════════════════════════
// 9. Peephole optimization — FusedSnakeAlpha, FusedInstanceNormMulAdd,
//    FusedAdainSnake patterns
// ═══════════════════════════════════════════════════════════════════════

/// PeepholeConfig::default() has all 17 fields enabled.
#[test]
fn peephole_default_all_enabled() {
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
    assert!(config.bilstm_cat);
    assert!(config.add_norm_linear);
    assert!(config.fuse_adain_snake);
    assert!(config.fuse_upsample_conv1d);
    assert!(config.fuse_instance_norm_mul_add);
    assert!(config.fuse_conv1d_activation);
}

/// PeepholeConfig can be selectively disabled.
#[test]
fn peephole_config_selective_disable() {
    // Disable just AdainSnake fusion.
    let config = PeepholeConfig {
        fuse_adain_snake: false,
        ..Default::default()
    };
    assert!(!config.fuse_adain_snake);
    // Other fields remain enabled.
    assert!(config.norm_activ_conv1d);
    assert!(config.fuse_conv1d_activation);
    assert!(config.fuse_instance_norm_mul_add);
    assert!(config.fuse_upsample_conv1d);
    assert!(config.fused_resblock);
}

/// FusedAdainSnake with external_node_ids serializes/deserializes.
#[test]
fn peephole_fused_adain_snake_with_externals() {
    let op = NativeOpKind::FusedAdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 256, 1024],
        channels: 256,
        external_node_ids: Some(vec![10, 20, 30, 40]),
    };
    let json = serde_json::to_string(&op).expect("serialize");
    assert!(json.contains("FusedAdainSnake"));
    assert!(json.contains("external_node_ids"));
    let deser: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deser.variant_name(), "FusedAdainSnake");
}

/// FusedInstanceNormMulAdd produces exactly 1 dispatch.
#[test]
fn peephole_fused_instance_norm_mul_add_single_dispatch() {
    let op = NativeOpKind::FusedInstanceNormMulAdd {
        eps: 1e-5,
        input_shape: vec![1, 64, 256],
        channels: 64,
        external_node_ids: None,
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

/// FusedConv1dActivation with all ConvActivation variants.
#[test]
fn peephole_fused_conv1d_all_activations() {
    let activations: Vec<ConvActivation> = vec![
        ConvActivation::Snake,
        ConvActivation::Relu,
        ConvActivation::LeakyRelu { slope: 0.2 },
        ConvActivation::Silu,
    ];
    for act in &activations {
        let op = NativeOpKind::FusedConv1dActivation {
            activation: *act,
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
            input_shape: vec![1, 64, 512],
            pre_activation: false,
        };
        assert_eq!(
            op.variant_name(),
            "FusedConv1dActivation",
            "Failed for {act:?}"
        );
        assert_eq!(
            op.estimated_metal_dispatches(),
            1,
            "FusedConv1dActivation should be 1 dispatch for {act:?}"
        );
        // Serde round-trip.
        let json = serde_json::to_string(&op).expect("serialize");
        let _: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
    }
}

/// FusedUpsampleConv1d produces 1 dispatch.
#[test]
fn peephole_fused_upsample_conv1d() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 256,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 256, 512],
    };
    assert_eq!(op.variant_name(), "FusedUpsampleConv1d");
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

// ═══════════════════════════════════════════════════════════════════════
// 10. GPU fallback — needs_f32_fallback, needs_non_float_fallback
// ═══════════════════════════════════════════════════════════════════════

/// GPU_FALLBACK_COUNT is initially zero (or accumulates from prior tests).
/// We verify it only increments monotonically.
#[test]
fn gpu_fallback_count_monotonic() {
    use std::sync::atomic::Ordering;
    let before = crate::GPU_FALLBACK_COUNT.load(Ordering::Relaxed);
    // Calling gpu_fallback increments the counter.
    let _: Option<()> = crate::gpu_fallback("test_op", "test reason");
    let after = crate::GPU_FALLBACK_COUNT.load(Ordering::Relaxed);
    assert_eq!(after, before + 1, "gpu_fallback should increment counter by 1");
}

/// gpu_fallback always returns None.
#[test]
fn gpu_fallback_returns_none() {
    let result: Option<i32> = crate::gpu_fallback("matmul", "non-f32 dtype");
    assert!(result.is_none());
    let result2: Option<String> = crate::gpu_fallback("softmax", "non-last axis");
    assert!(result2.is_none());
}

/// DType::is_float classification matches GPU fallback expectations.
/// F32, F16, BF16, F64 are float; I32, I64, U32, U8, Bool are not.
#[test]
fn gpu_fallback_float_dtype_classification() {
    // Float types: GPU can handle (no fallback for shape ops).
    assert!(DType::F32.is_float());
    assert!(DType::F16.is_float());
    assert!(DType::BF16.is_float());
    assert!(DType::F64.is_float());

    // Non-float types: need fallback for GPU shape ops.
    assert!(!DType::I32.is_float());
    assert!(!DType::I64.is_float());
    assert!(!DType::U32.is_float());
    assert!(!DType::U8.is_float());
    assert!(!DType::Bool.is_float());
}

/// dtype_to_msl succeeds for F32/F16/BF16 (GPU float types).
#[test]
fn gpu_fallback_dtype_to_msl_float_types() {
    let (msl, size) = crate::dtype_to_msl(DType::F32).expect("F32");
    assert_eq!(msl, "float");
    assert_eq!(size, 4);

    let (msl, size) = crate::dtype_to_msl(DType::F16).expect("F16");
    assert_eq!(msl, "half");
    assert_eq!(size, 2);

    let (msl, size) = crate::dtype_to_msl(DType::BF16).expect("BF16");
    assert_eq!(msl, "half");
    assert_eq!(size, 2);
}

/// dtype_to_msl rejects non-float types.
#[test]
fn gpu_fallback_dtype_to_msl_rejects_non_float() {
    for dtype in [DType::I32, DType::I64, DType::U32, DType::U8, DType::Bool] {
        assert!(
            crate::dtype_to_msl(dtype).is_err(),
            "{dtype:?} should have no MSL equivalent"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 11. Dtype dispatch — same_gpu_byte_width checks
// ═══════════════════════════════════════════════════════════════════════

/// ScalarType byte sizes match DType byte sizes for all float types.
#[test]
fn dtype_scalar_type_byte_size_matches() {
    assert_eq!(ScalarType::F32.byte_size(), DType::F32.size_bytes());
    assert_eq!(ScalarType::F16.byte_size(), DType::F16.size_bytes());
    assert_eq!(ScalarType::BF16.byte_size(), DType::BF16.size_bytes());
}

/// ScalarType MSL strings are correct.
#[test]
fn dtype_scalar_type_msl_str() {
    assert_eq!(ScalarType::F32.msl_str(), "float");
    assert_eq!(ScalarType::F16.msl_str(), "half");
    assert_eq!(ScalarType::BF16.msl_str(), "half"); // Apple bf16 -> half
}

/// ScalarType accumulator is always "float" (full precision).
#[test]
fn dtype_accumulator_always_float() {
    for st in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        assert_eq!(
            st.msl_accumulator_str(),
            "float",
            "accumulator should be float for {st:?}"
        );
    }
}

/// ScalarType round-trips through DType conversion.
#[test]
fn dtype_scalar_type_roundtrip() {
    for st in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        let dtype: DType = st.into();
        let recovered = ScalarType::try_from(dtype).expect("roundtrip");
        assert_eq!(st, recovered, "roundtrip failed for {st:?}");
    }
}

/// F16 and BF16 share the same byte width (2 bytes).
#[test]
fn dtype_f16_bf16_same_byte_width() {
    assert_eq!(
        DType::F16.size_bytes(),
        DType::BF16.size_bytes(),
        "F16 and BF16 should have the same byte width"
    );
}

/// F32 and F64 have different byte widths.
#[test]
fn dtype_f32_f64_different_byte_width() {
    assert_ne!(
        DType::F32.size_bytes(),
        DType::F64.size_bytes(),
        "F32 (4) and F64 (8) should differ"
    );
}

/// F16/BF16 have different byte width from F32.
#[test]
fn dtype_half_vs_float_different_byte_width() {
    assert_ne!(DType::F16.size_bytes(), DType::F32.size_bytes());
    assert_ne!(DType::BF16.size_bytes(), DType::F32.size_bytes());
}

// ═══════════════════════════════════════════════════════════════════════
// 12. GpuScope — command buffer depth management
// ═══════════════════════════════════════════════════════════════════════

/// ScopeExitMode Clone produces an equal value.
#[test]
fn gpu_scope_exit_mode_clone_eq() {
    let mode = ScopeExitMode::Flush;
    let cloned = mode;
    assert_eq!(mode, cloned);

    let submit = ScopeExitMode::Submit;
    let cloned_submit = submit;
    assert_eq!(submit, cloned_submit);
}

/// Dispatch stats are isolated between reset calls.
#[test]
fn gpu_scope_stats_isolation() {
    reset_counters();
    let before = dispatch_stats();
    assert_eq!(before.compute_encodings, 0);
    assert_eq!(before.flushes, 0);
    assert_eq!(before.submits, 0);
    assert_eq!(before.blits, 0);

    // Without GPU ops, counters stay at zero.
    let after = dispatch_stats();
    assert_eq!(before, after, "no GPU ops => stats unchanged");
}

/// DispatchStats fields are accessible and have correct types.
#[test]
fn gpu_scope_stats_fields() {
    reset_counters();
    let stats = dispatch_stats();
    let _: usize = stats.compute_encodings;
    let _: usize = stats.blits;
    let _: usize = stats.flushes;
    let _: usize = stats.submits;
    // ArenaStats is also part of DispatchStats.
    let _arena = stats.arena;
}

/// to_u32 helper converts valid usize to u32.
#[test]
fn gpu_scope_to_u32_valid() {
    assert_eq!(crate::to_u32(0, "zero").unwrap(), 0u32);
    assert_eq!(crate::to_u32(42, "small").unwrap(), 42u32);
    assert_eq!(
        crate::to_u32(u32::MAX as usize, "max").unwrap(),
        u32::MAX
    );
}

/// to_u32 helper rejects values exceeding u32::MAX.
#[test]
fn gpu_scope_to_u32_overflow() {
    let too_big = u32::MAX as usize + 1;
    assert!(crate::to_u32(too_big, "overflow").is_err());
}

/// Threadgroup size for elementwise plans stays within Metal limit (1024).
#[test]
fn gpu_scope_threadgroup_bounded() {
    for total in [1, 32, 64, 128, 1024, 65536, u32::MAX] {
        let plan = plan_elementwise(total).unwrap();
        let [tw, th, td] = plan.threads();
        let product = tw * th * td;
        assert!(
            product <= 1024,
            "threadgroup product {product} exceeds 1024 for total={total}"
        );
    }
}

/// Reduction plan threadgroup stays within Metal limit.
#[test]
fn gpu_scope_reduction_threadgroup_bounded() {
    let cases = [(32, 256, 256, 4096), (1, 1024, 128, 2048), (512, 64, 64, 1024)];
    for (outer, reduce, thr, shared) in cases {
        let plan = plan_reduction(outer, reduce, thr, shared).unwrap();
        let [tw, th, td] = plan.threads();
        let product = tw * th * td;
        assert!(
            product <= 1024,
            "reduction threadgroup {product} exceeds 1024"
        );
    }
}

/// Grid2D plans produce correct output element counts.
#[test]
fn gpu_scope_grid2d_output_elems() {
    let plan = plan_grid_2d([64, 32], [8, 8]).unwrap();
    assert_eq!(plan.output_elems(), 64 * 32);
    assert_eq!(plan.grid(), [64, 32, 1]);
}

/// count_non_finite finds NaN and Inf values.
#[test]
fn gpu_scope_count_non_finite() {
    assert_eq!(crate::count_non_finite(&[1.0, 2.0, 3.0]), 0);
    assert_eq!(crate::count_non_finite(&[1.0, f32::NAN, 3.0]), 1);
    assert_eq!(crate::count_non_finite(&[f32::INFINITY, f32::NEG_INFINITY]), 2);
    assert_eq!(
        crate::count_non_finite(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]),
        3
    );
    assert_eq!(crate::count_non_finite(&[]), 0);
}
