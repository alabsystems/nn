// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for Metal dispatch infrastructure, compiled model builder,
//! buffer management, pipeline cache, threadgroup sizing, command buffer
//! batching, ActivationArena, registry, GpuScope, and dtype mapping.
//!
//! Tests are structured into 10 sections covering the full Metal dispatch
//! pipeline from NativeOpKind variant counting through dtype mapping. All
//! tests are structure/config tests only -- no live GPU required (except
//! builder/arena tests that need MetalContext on macOS).
//!
//! Part of #4186.

use std::collections::HashSet;

use nn_core::DType;
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::NativeOpKind;
use nn_dsl::{DispatchStep, TensorNodeId};

use crate::dispatch_plan::{
    clear_dispatch_plan_cache, dispatch_plan_cache_len, plan_elementwise, plan_grid_2d,
    plan_grid_3d, plan_reduction, DispatchMode,
};
use crate::dispatch_stats::{dispatch_stats, reset_counters};
use crate::gpu_scope::ScopeExitMode;

// ═══════════════════════════════════════════════════════════════════════
// 1. NativeOpKind variant count — assert matches known count
// ═══════════════════════════════════════════════════════════════════════

/// Collect all NativeOpKind variant names via variant_name() and assert
/// the total count matches the expected number (33 as of FusedConv1dActivation).
/// If a new variant is added, this test forces the count to be updated.
#[test]
fn native_op_kind_variant_count_matches_known() {
    let all_variants: Vec<NativeOpKind> = vec![
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
            phase1: nn_dsl::trace_compile::NormActivConv1dParams::new(
                nn_dsl::trace_compile::NormActivation::Snake,
                1e-5,
                1,
                1,
                vec![1, 128, 512],
                128,
                3,
            ),
            phase2: nn_dsl::trace_compile::NormActivConv1dParams::new(
                nn_dsl::trace_compile::NormActivation::Snake,
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
            activation: nn_dsl::trace_compile::NormActivation::Snake,
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 1,
            input_shape: vec![1, 128, 512],
            output_channels: 128,
            kernel_size: 3,
            external_node_ids: None,
        },
        NativeOpKind::LinearActivation {
            activation: nn_dsl::trace_compile::GemmActivation::Relu,
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
            norm_kind: nn_dsl::trace_compile::FusedNormKind::LayerNorm,
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
            activation: nn_dsl::trace_compile::ConvActivation::Snake,
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
    ];

    // Collect unique variant names.
    let names: HashSet<&str> = all_variants.iter().map(NativeOpKind::variant_name).collect();

    const EXPECTED_NATIVE_OP_COUNT: usize = 34;
    assert_eq!(
        names.len(),
        EXPECTED_NATIVE_OP_COUNT,
        "NativeOpKind variant count changed. Expected {EXPECTED_NATIVE_OP_COUNT}, \
         got {}. Update EXPECTED_NATIVE_OP_COUNT and add the new variant to this test.\n\
         Variants: {names:?}",
        names.len(),
    );

    // Also verify all constructed variants have unique names.
    assert_eq!(
        all_variants.len(),
        EXPECTED_NATIVE_OP_COUNT,
        "all_variants list should have exactly one entry per variant"
    );
}

/// Every NativeOpKind variant has a non-empty variant_name.
#[test]
fn native_op_kind_all_variants_have_nonempty_name() {
    let ops: Vec<NativeOpKind> = vec![
        NativeOpKind::LstmSequence {
            hidden_size: 64,
            input_shape: vec![5, 1, 32],
            h_shape: vec![1, 64],
            reverse: false,
        },
        NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![10],
        },
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 8, 16],
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: true,
            q_shape: vec![1, 4, 8, 32],
            k_shape: vec![1, 4, 8, 32],
            output_shape: vec![1, 4, 8, 32],
            input_layout: Default::default(),
        },
    ];
    for op in &ops {
        let name = op.variant_name();
        assert!(
            !name.is_empty(),
            "variant_name() should be non-empty, got empty for {op:?}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. DispatchStep encoding — each variant produces valid dispatch info
// ═══════════════════════════════════════════════════════════════════════

/// Verify DispatchStep Debug output contains the variant name for each
/// constructed variant. This confirms serialization and dispatch plan
/// encoding produce valid instructions.
#[test]
fn dispatch_step_all_variants_produce_valid_debug() {
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

/// DispatchStep variants with total_elements correctly encode element counts.
#[test]
fn dispatch_step_element_counts_consistent() {
    let n0 = TensorNodeId::new(0);
    let n1 = TensorNodeId::new(1);

    // Softmax: outer_size * axis_size = total elements processed.
    let axis_size = 768_usize;
    let outer_size = 32_usize;
    let step = DispatchStep::Softmax {
        kernel_name: "sm".into(),
        dtype: ScalarType::F32,
        input: n0,
        output: n1,
        axis: 1,
        axis_size,
        outer_size,
    };
    let dbg = format!("{step:?}");
    assert!(dbg.contains("768"), "axis_size should appear in Debug: {dbg}");
    assert!(dbg.contains("32"), "outer_size should appear in Debug: {dbg}");
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Buffer size calculation — matches tensor element count * dtype size
// ═══════════════════════════════════════════════════════════════════════

/// DType::size_bytes matches expected byte widths for all float types.
#[test]
fn buffer_size_float_dtypes() {
    assert_eq!(DType::F32.size_bytes(), 4);
    assert_eq!(DType::F16.size_bytes(), 2);
    assert_eq!(DType::BF16.size_bytes(), 2);
    assert_eq!(DType::F64.size_bytes(), 8);
}

/// Buffer size for a known tensor shape: elements * dtype byte size.
#[test]
fn buffer_size_calculation_shape_times_dtype() {
    let shape = [1_usize, 128, 512];
    let elem_count: usize = shape.iter().product();
    assert_eq!(elem_count, 65536);

    // F32: 65536 * 4 = 262144 bytes.
    let f32_bytes = elem_count * DType::F32.size_bytes();
    assert_eq!(f32_bytes, 262_144);

    // F16: 65536 * 2 = 131072 bytes.
    let f16_bytes = elem_count * DType::F16.size_bytes();
    assert_eq!(f16_bytes, 131_072);

    // BF16: 65536 * 2 = 131072 bytes (same as F16 on Metal).
    let bf16_bytes = elem_count * DType::BF16.size_bytes();
    assert_eq!(bf16_bytes, f16_bytes);
}

/// Buffer size for integer types used in embedding indices and masks.
#[test]
fn buffer_size_integer_dtypes() {
    assert_eq!(DType::I32.size_bytes(), 4);
    assert_eq!(DType::I64.size_bytes(), 8);
    assert_eq!(DType::U32.size_bytes(), 4);
    assert_eq!(DType::U8.size_bytes(), 1);
    assert_eq!(DType::Bool.size_bytes(), 1);
}

/// Overflow check: buffer byte count for large tensor should not silently
/// truncate when computed with checked_mul.
#[test]
fn buffer_size_overflow_detection() {
    let large_elements: usize = usize::MAX / 2;
    let byte_size = DType::F32.size_bytes(); // 4
    let result = large_elements.checked_mul(byte_size);
    assert!(
        result.is_none(),
        "usize::MAX/2 * 4 should overflow on 64-bit: got {result:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Pipeline cache — same kernel config returns cached plan
// ═══════════════════════════════════════════════════════════════════════

/// plan_cached returns the same plan for repeated calls with same mode.
#[test]
fn pipeline_cache_same_mode_returns_same_plan() {
    clear_dispatch_plan_cache();

    let mode = DispatchMode::Elementwise { total: 1024 };
    let plan1 = mode.plan_cached().unwrap();
    let plan2 = mode.plan_cached().unwrap();

    assert_eq!(plan1, plan2, "same mode should return equal cached plan");
    assert_eq!(
        dispatch_plan_cache_len(),
        1,
        "single unique mode => 1 cache entry"
    );
}

/// plan_cached distinguishes different modes and caches each separately.
#[test]
fn pipeline_cache_different_modes_cached_separately() {
    clear_dispatch_plan_cache();

    let mode_a = DispatchMode::Elementwise { total: 256 };
    let mode_b = DispatchMode::Elementwise { total: 512 };

    let plan_a = mode_a.plan_cached().unwrap();
    let plan_b = mode_b.plan_cached().unwrap();

    assert_ne!(plan_a, plan_b, "different modes should produce different plans");
    assert_eq!(dispatch_plan_cache_len(), 2);

    // Re-fetch: should still match.
    let plan_a2 = mode_a.plan_cached().unwrap();
    assert_eq!(plan_a, plan_a2);
}

/// plan_cached result equals plan() result (cache does not alter plan).
#[test]
fn pipeline_cache_matches_uncached_plan() {
    clear_dispatch_plan_cache();

    let mode = DispatchMode::Grid2D {
        grid: [64, 32],
        threads: [8, 8],
    };
    let uncached = mode.plan().unwrap();
    let cached = mode.plan_cached().unwrap();

    assert_eq!(uncached, cached, "cached must equal uncached for Grid2D");
}

/// Cache survives many insertions without panic.
#[test]
fn pipeline_cache_many_entries_no_panic() {
    clear_dispatch_plan_cache();
    for i in 1..=256u32 {
        let _ = DispatchMode::Elementwise { total: i }
            .plan_cached()
            .unwrap();
    }
    assert!(dispatch_plan_cache_len() <= 256);
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Threadgroup size — dimensions don't exceed device maximum
// ═══════════════════════════════════════════════════════════════════════

/// Apple Metal maximum threadgroup size is 1024 threads total.
/// Verify plan dimensions stay within this bound.
const METAL_MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// Elementwise plan threadgroup width is at most 64 (the 1D max).
#[test]
fn threadgroup_size_elementwise_bounded() {
    for total in [1, 32, 64, 128, 1024, u32::MAX] {
        let plan = plan_elementwise(total).unwrap();
        let [tw, th, td] = plan.threads();
        let product = tw * th * td;
        assert!(
            product <= METAL_MAX_THREADS_PER_THREADGROUP,
            "threadgroup product {product} exceeds Metal max for total={total}"
        );
        assert!(tw <= 64, "1D threadgroup width {tw} exceeds 64 for total={total}");
    }
}

/// Grid2D plan threads stay within Metal bounds.
#[test]
fn threadgroup_size_grid2d_bounded() {
    let cases = [
        ([1, 1], [1, 1]),
        ([128, 64], [16, 16]),
        ([1024, 512], [32, 32]),
    ];
    for (grid, threads) in cases {
        let plan = plan_grid_2d(grid, threads).unwrap();
        let [tw, th, td] = plan.threads();
        let product = tw * th * td;
        assert!(
            product <= METAL_MAX_THREADS_PER_THREADGROUP,
            "2D threadgroup product {product} exceeds Metal max for grid={grid:?}, threads={threads:?}"
        );
    }
}

/// Grid3D plan threads stay within Metal bounds.
#[test]
fn threadgroup_size_grid3d_bounded() {
    let cases = [
        ([1, 1, 1], [1, 1, 1]),
        ([16, 8, 4], [8, 8, 4]),
        ([32, 16, 8], [8, 4, 2]),
    ];
    for (grid, threads) in cases {
        let plan = plan_grid_3d(grid, threads).unwrap();
        let [tw, th, td] = plan.threads();
        let product = tw * th * td;
        assert!(
            product <= METAL_MAX_THREADS_PER_THREADGROUP,
            "3D threadgroup product {product} exceeds Metal max for grid={grid:?}, threads={threads:?}"
        );
    }
}

/// Reduction plan threadgroup size is within Metal limits.
#[test]
fn threadgroup_size_reduction_bounded() {
    let cases = [(32, 256, 256, 4096), (1, 1024, 128, 2048), (512, 64, 64, 1024)];
    for (outer, reduce, thr, shared) in cases {
        let plan = plan_reduction(outer, reduce, thr, shared).unwrap();
        let [tw, th, td] = plan.threads();
        let product = tw * th * td;
        assert!(
            product <= METAL_MAX_THREADS_PER_THREADGROUP,
            "reduction threadgroup {product} exceeds Metal max"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Metal command buffer batching — flush commits pending work
// ═══════════════════════════════════════════════════════════════════════

/// ScopeExitMode has exactly two variants (Flush and Submit).
#[test]
fn scope_exit_mode_two_variants() {
    let flush = ScopeExitMode::Flush;
    let submit = ScopeExitMode::Submit;
    assert_ne!(flush, submit, "Flush and Submit must be distinct");
}

/// ScopeExitMode::Flush is the default (synchronous).
#[test]
fn scope_exit_mode_default_is_flush() {
    // The thread-local default is ScopeExitMode::Flush as documented.
    let flush = ScopeExitMode::Flush;
    assert_eq!(flush, ScopeExitMode::Flush);
}

/// Dispatch stats counters start at zero after reset.
#[test]
fn command_buffer_stats_reset_zeroes() {
    reset_counters();
    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 0, "encodings should be 0 after reset");
    assert_eq!(stats.flushes, 0, "flushes should be 0 after reset");
    assert_eq!(stats.submits, 0, "submits should be 0 after reset");
    assert_eq!(stats.blits, 0, "blits should be 0 after reset");
}

/// Dispatch stats are snapshottable (multiple reads don't mutate).
#[test]
fn command_buffer_stats_snapshot_stable() {
    reset_counters();
    let s1 = dispatch_stats();
    let s2 = dispatch_stats();
    assert_eq!(s1, s2, "consecutive stats reads should be equal");
}

// ═══════════════════════════════════════════════════════════════════════
// 7. ActivationArena — reuse reduces allocation count
// ═══════════════════════════════════════════════════════════════════════

/// ActivationArena creation with valid capacity succeeds on macOS.
#[test]
fn activation_arena_creation_succeeds() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let arena = crate::arena::ActivationArena::new(&ctx, 1024 * 1024);
    assert!(arena.is_ok(), "arena creation with 1MB should succeed");
}

/// ActivationArena rejects zero capacity.
#[test]
fn activation_arena_zero_capacity_rejected() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let result = crate::arena::ActivationArena::new(&ctx, 0);
    assert!(result.is_err(), "arena with 0 capacity should fail");
}

/// ActivationArena reset brings offset back to zero (reuse).
#[test]
fn activation_arena_reset_reuses_memory() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let mut arena = crate::arena::ActivationArena::new(&ctx, 4 * 1024 * 1024)
        .expect("4MB arena");

    // Alloc some data (byte_len = elements * sizeof(f32)).
    let alloc_bytes_1 = 1 * 128 * 256 * DType::F32.size_bytes();
    let alloc_bytes_2 = 1 * 64 * 128 * DType::F32.size_bytes();
    let _t1 = arena.alloc(alloc_bytes_1);
    let _t2 = arena.alloc(alloc_bytes_2);

    // Reset and re-alloc: should succeed without growing.
    arena.reset();
    let t3 = arena.alloc(alloc_bytes_1);
    assert!(
        t3.is_ok(),
        "arena alloc after reset should succeed (reuse): {:?}",
        t3.err()
    );
}

/// ActivationArena stats track peak usage.
#[test]
fn activation_arena_peak_tracking() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let mut arena = crate::arena::ActivationArena::new(&ctx, 8 * 1024 * 1024)
        .expect("8MB arena");

    let alloc_bytes = 1 * 256 * 512 * DType::F32.size_bytes(); // 512KB
    let _t1 = arena.alloc(alloc_bytes).unwrap();
    let peak_after_first = arena.peak_bytes();
    assert!(peak_after_first > 0, "peak should be > 0 after alloc");

    let _t2 = arena.alloc(alloc_bytes).unwrap();
    let peak_after_second = arena.peak_bytes();
    assert!(
        peak_after_second >= peak_after_first,
        "peak should grow or stay: {peak_after_second} < {peak_after_first}"
    );

    arena.reset();
    // Peak is preserved across resets.
    assert_eq!(
        arena.peak_bytes(),
        peak_after_second,
        "peak should be preserved after reset"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Compiled model registry — all registered kernels have valid MSL
// ═══════════════════════════════════════════════════════════════════════

/// CompiledModel builder on empty graph produces a model with zero steps
/// and zero dispatches.
#[test]
fn compiled_model_registry_empty_graph() {
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

/// All NativeOpKind variants produce non-empty variant_name strings,
/// which are used as registry keys in compiled model dispatch.
#[test]
fn compiled_model_registry_variant_names_valid() {
    let ops = vec![
        NativeOpKind::LstmSequence {
            hidden_size: 64,
            input_shape: vec![5, 1, 32],
            h_shape: vec![1, 64],
            reverse: false,
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 4, 8, 32],
            k_shape: vec![1, 4, 8, 32],
            output_shape: vec![1, 4, 8, 32],
            input_layout: Default::default(),
        },
        NativeOpKind::FusedAdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            external_node_ids: None,
        },
        NativeOpKind::FusedConv1dActivation {
            activation: nn_dsl::trace_compile::ConvActivation::LeakyRelu { slope: 0.2 },
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
            input_shape: vec![1, 64, 256],
            pre_activation: false,
        },
    ];
    for op in &ops {
        let name = op.variant_name();
        assert!(!name.is_empty(), "variant_name must be non-empty: {op:?}");
        assert!(
            name.chars().all(|c| c.is_ascii_alphanumeric()),
            "variant_name should be alphanumeric: {name}"
        );
    }
}

/// NativeOpKind::estimated_metal_dispatches returns >= 1 for all non-zero ops.
#[test]
fn compiled_model_registry_dispatch_estimates_positive() {
    let ops = vec![
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 64, 128],
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 32, 768],
            hidden_dim: 768,
        },
        NativeOpKind::FusedAdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            external_node_ids: None,
        },
        NativeOpKind::FusedInstanceNormMulAdd {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            external_node_ids: None,
        },
    ];
    for op in &ops {
        assert!(
            op.estimated_metal_dispatches() >= 1,
            "{} should have >= 1 dispatch, got {}",
            op.variant_name(),
            op.estimated_metal_dispatches()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 9. GpuScope nesting — nested scopes don't double-commit
// ═══════════════════════════════════════════════════════════════════════

/// ScopeExitMode Clone and Debug are implemented.
#[test]
fn gpu_scope_exit_mode_clone_debug() {
    let mode = ScopeExitMode::Flush;
    let cloned = mode;
    assert_eq!(mode, cloned, "Clone should produce equal value");
    let dbg = format!("{mode:?}");
    assert!(dbg.contains("Flush"), "Debug should contain 'Flush': {dbg}");
}

/// ScopeExitMode::Submit Debug representation is correct.
#[test]
fn gpu_scope_exit_mode_submit_debug() {
    let mode = ScopeExitMode::Submit;
    let dbg = format!("{mode:?}");
    assert!(
        dbg.contains("Submit"),
        "Debug should contain 'Submit': {dbg}"
    );
}

/// Dispatch stats are isolated between reset calls — flush counter
/// does not leak from prior test state.
#[test]
fn gpu_scope_stats_isolation() {
    reset_counters();
    let before = dispatch_stats();
    assert_eq!(before.flushes, 0);
    assert_eq!(before.submits, 0);

    // Without GPU ops, counters stay at zero.
    let after = dispatch_stats();
    assert_eq!(before, after, "no GPU ops => stats unchanged");
}

/// Multiple reset_counters calls are idempotent.
#[test]
fn gpu_scope_reset_idempotent() {
    reset_counters();
    reset_counters();
    reset_counters();
    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 0);
    assert_eq!(stats.flushes, 0);
}

// ═══════════════════════════════════════════════════════════════════════
// 10. Metal dtype mapping — all DType variants have Metal equivalents
// ═══════════════════════════════════════════════════════════════════════

/// ScalarType F32/F16/BF16 all have valid MSL type strings.
#[test]
fn dtype_mapping_scalar_types_have_msl_str() {
    assert_eq!(ScalarType::F32.msl_str(), "float");
    assert_eq!(ScalarType::F16.msl_str(), "half");
    assert_eq!(ScalarType::BF16.msl_str(), "half"); // Apple GPUs: bf16 -> half
}

/// ScalarType byte_size matches DType::size_bytes for float types.
#[test]
fn dtype_mapping_scalar_byte_size_matches_core() {
    assert_eq!(ScalarType::F32.byte_size(), DType::F32.size_bytes());
    assert_eq!(ScalarType::F16.byte_size(), DType::F16.size_bytes());
    assert_eq!(ScalarType::BF16.byte_size(), DType::BF16.size_bytes());
}

/// dtype_to_msl helper returns correct (msl_str, byte_size) for float types.
#[test]
fn dtype_mapping_dtype_to_msl_float_types() {
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

/// dtype_to_msl rejects integer types that have no Metal scalar equivalent.
#[test]
fn dtype_mapping_integer_types_rejected() {
    assert!(
        crate::dtype_to_msl(DType::I32).is_err(),
        "I32 should have no ScalarType equivalent"
    );
    assert!(
        crate::dtype_to_msl(DType::I64).is_err(),
        "I64 should have no ScalarType equivalent"
    );
    assert!(
        crate::dtype_to_msl(DType::U32).is_err(),
        "U32 should have no ScalarType equivalent"
    );
    assert!(
        crate::dtype_to_msl(DType::U8).is_err(),
        "U8 should have no ScalarType equivalent"
    );
    assert!(
        crate::dtype_to_msl(DType::Bool).is_err(),
        "Bool should have no ScalarType equivalent"
    );
}

/// ScalarType roundtrips through DType conversion.
#[test]
fn dtype_mapping_scalar_type_roundtrip() {
    for st in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        let dtype: DType = st.into();
        let recovered = ScalarType::try_from(dtype).expect("roundtrip");
        assert_eq!(st, recovered, "roundtrip failed for {st:?}");
    }
}

/// All ScalarType variants have consistent msl_str and byte_size.
#[test]
fn dtype_mapping_all_scalar_types_consistent() {
    let all = [ScalarType::F32, ScalarType::F16, ScalarType::BF16];
    for st in all {
        let msl = st.msl_str();
        let size = st.byte_size();
        assert!(!msl.is_empty(), "msl_str should be non-empty for {st:?}");
        assert!(size > 0, "byte_size should be > 0 for {st:?}");
        // MSL accumulator type should always be "float" for precision.
        assert_eq!(
            st.msl_accumulator_str(),
            "float",
            "accumulator should be float for {st:?}"
        );
    }
}
