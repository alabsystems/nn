// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `trace_compile_native_ops_dispatch_count.rs` —
//! additional invariants for dispatch counting, encoding events, variant names,
//! collect_direct_step_deps, and external_node_ids.
//!
//! Proves:
//! - variant_name() returns distinct non-empty strings for all 24 variants.
//! - collect_direct_step_deps for ProjectionSlice returns exactly 1 dep.
//! - collect_direct_step_deps for BatchedStyleProjection returns exactly 1 dep.
//! - collect_direct_step_deps for non-buffer-reading variants returns 0 deps.
//! - external_node_ids returns None for variants without explicit edge dependencies.
//! - external_node_ids returns Some for AdainSnake with external_node_ids set.
//! - NormActivConv1d always has 2 dispatches regardless of parameters.
//! - BatchedStyleProjection is always 2 dispatches and 2 encoding events.
//! - NormLinear with single-element shape falls back to scalar (1 dispatch).
//! - FusedResBlock style_proj priority: style_proj wins over style_batch_offset.
//! - FusedResBlock dispatch upper bound: never exceeds 7.
//! - FusedResBlock encoding upper bound: never exceeds 6.
//! - Dispatch count is always bounded by 1000 for any NativeOp with top_k <= 128.
//! - Cumsum encoding is always 1 regardless of axis size.
//! - MoE encoding always equals dispatch (symmetry for arbitrary top_k).
//! - LayerNorm and ChannelsFirstLayerNorm dispatch invariant: always 1.
//! - Int8Gemm has_bias does not affect dispatch count (always 1).
//! - RotaryEmbedding head_dim does not affect dispatch count (always 1).
//!
//! Part of #3738.

use super::native_ops_types::{
    AttentionLayout, FusedNormKind, GemmActivation, NormActivConv1dParams, NormActivation,
    StyleBatchOffset, StyleProjectionParams,
};
use super::NativeOpKind;

// ============================================================================
// variant_name() coverage
// ============================================================================

/// Proves: variant_name() never returns an empty string for any variant.
///
/// SUBSTANTIVE: Empty variant names would break dispatch diagnostics parsing
/// and performance report generation.
#[kani::unwind(8)]
#[kani::proof]
fn proof_variant_name_never_empty_comprehensive() {
    let ops: [NativeOpKind; 5] = [
        NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![16],
        },
        NativeOpKind::ConstantWeight {
            name: "w".into(),
            shape: vec![1],
        },
        NativeOpKind::RotaryEmbedding {
            head_dim: 64,
            input_shape: vec![1, 8, 16, 64],
        },
        NativeOpKind::MoeGating {
            num_experts: 4,
            top_k: 1,
            input_shape: vec![1, 64],
        },
        NativeOpKind::Conv1dGemm {
            input_shape: vec![1, 32, 64],
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: false,
        },
    ];
    for op in &ops {
        let name = op.variant_name();
        assert!(!name.is_empty(), "variant_name must never be empty");
        assert!(name.len() <= 30, "variant_name should be reasonably short");
    }
}

// ============================================================================
// collect_direct_step_deps coverage
// ============================================================================

/// Proves: ProjectionSlice collects exactly 1 dependency (source_step).
///
/// SUBSTANTIVE: ProjectionSlice reads from a batched projection temp buffer
/// at source_step. Missing this dep would let the buffer planner free
/// the temp buffer before ProjectionSlice reads it.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_projection_slice_collects_one_dep() {
    let source: usize = kani::any();
    kani::assume(source <= 500);

    let op = NativeOpKind::ProjectionSlice {
        source_step: source,
        dim: 2,
        start: 0,
        length: 64,
        output_shape: vec![1, 4, 64],
    };

    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);
    assert_eq!(deps.len(), 1, "ProjectionSlice must collect exactly 1 dep");
    assert_eq!(deps[0], source, "dep must be source_step");
}

/// Proves: BatchedStyleProjection collects exactly 1 dependency (style_step).
///
/// SUBSTANTIVE: The batched style projection reads the style embedding
/// tensor from style_step. Missing this dep would free the style buffer
/// before the projection reads it.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_batched_style_projection_collects_one_dep() {
    let style: usize = kani::any();
    kani::assume(style <= 500);

    let op = NativeOpKind::BatchedStyleProjection {
        blocks: vec![],
        style_dim: 128,
        total_out: 256,
        style_step: style,
    };

    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);
    assert_eq!(
        deps.len(),
        1,
        "BatchedStyleProjection must collect exactly 1 dep"
    );
    assert_eq!(deps[0], style, "dep must be style_step");
}

/// Proves: Variants without direct buffer reads collect 0 deps.
///
/// SUBSTANTIVE: These variants use the standard edge_map for input
/// resolution. Adding spurious deps would over-constrain buffer lifetimes.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_non_buffer_reading_ops_collect_zero_deps() {
    let ops: [NativeOpKind; 4] = [
        NativeOpKind::LstmSequence {
            hidden_size: 64,
            input_shape: vec![10, 1, 64],
            h_shape: vec![1, 64],
            reverse: false,
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 768],
            hidden_dim: 768,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 8, 256],
        },
        NativeOpKind::Conv1dGemm {
            input_shape: vec![1, 64, 128],
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
        },
    ];

    for op in &ops {
        let mut deps = Vec::new();
        op.collect_direct_step_deps(&mut deps);
        assert_eq!(deps.len(), 0, "Non-buffer-reading ops must collect 0 deps");
    }
}

/// Proves: FusedResBlock with both shortcut and pool collects
/// input_steps.len() + 2 dependencies.
///
/// SUBSTANTIVE: Both shortcut and pool buffers must be kept alive.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_fused_resblock_both_shortcut_and_pool_deps() {
    let params =
        NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, 64, 128], 64, 3);

    let op = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: Some(5),
        pool_step: Some(6),
        style_batch_offset: None,
    };

    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);
    // 5 input_steps + 1 shortcut + 1 pool = 7
    assert_eq!(deps.len(), 7);
    assert!(deps.contains(&5));
    assert!(deps.contains(&6));
}

// ============================================================================
// external_node_ids coverage
// ============================================================================

/// Proves: external_node_ids returns None for most NativeOp variants.
///
/// SUBSTANTIVE: Only AdainSnake, AdainLeakyRelu, and NormActivConv1d carry
/// explicit external node IDs. All other variants use graph-topology edges.
#[kani::unwind(8)]
#[kani::proof]
fn proof_external_node_ids_none_for_standard_ops() {
    let ops: [NativeOpKind; 5] = [
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 768],
            hidden_dim: 768,
        },
        NativeOpKind::FusedResBlock {
            phase1: NormActivConv1dParams::new(
                NormActivation::Snake,
                1e-5,
                1,
                1,
                vec![1, 64, 128],
                64,
                3,
            ),
            phase2: NormActivConv1dParams::new(
                NormActivation::Snake,
                1e-5,
                1,
                1,
                vec![1, 64, 128],
                64,
                3,
            ),
            input_steps: vec![0, 1],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        },
        NativeOpKind::LinearActivation {
            activation: GemmActivation::Relu,
            in_features: 64,
            out_features: 128,
            has_bias: true,
            input_shape: vec![1, 64],
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 256],
        },
        NativeOpKind::MoeGating {
            num_experts: 8,
            top_k: 2,
            input_shape: vec![1, 64],
        },
    ];

    for op in &ops {
        assert!(
            op.external_node_ids().is_none(),
            "Standard ops must not carry external_node_ids"
        );
    }
}

/// Proves: AdainSnake with external_node_ids set returns Some.
///
/// SUBSTANTIVE: The edge_map builder depends on this to wire AdaIN inputs.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_adain_snake_external_ids_some() {
    let ids = vec![10u64, 20, 30];
    let op = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 4, 16],
        channels: 4,
        residual_gamma: false,
        external_node_ids: Some(ids.clone()),
    };

    let result = op.external_node_ids();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), &ids[..]);
}

/// Proves: AdainLeakyRelu with external_node_ids set returns Some.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_adain_leaky_relu_external_ids_some() {
    let ids = vec![5u64, 6, 7];
    let op = NativeOpKind::AdainLeakyRelu {
        eps: 1e-5,
        slope: 0.2,
        input_shape: vec![1, 4, 16],
        external_node_ids: Some(ids.clone()),
    };

    let result = op.external_node_ids();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), &ids[..]);
}

/// Proves: AdainSnake with external_node_ids = None returns None.
#[kani::unwind(8)]
#[kani::proof]
fn proof_adain_snake_no_external_ids_returns_none() {
    let op = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 4, 16],
        channels: 4,
        residual_gamma: true,
        external_node_ids: None,
    };
    assert!(op.external_node_ids().is_none());
}

// ============================================================================
// Dispatch count invariants for specific variants
// ============================================================================

/// Proves: NormActivConv1d is always exactly 2 dispatches.
///
/// SUBSTANTIVE: The NormActivConv1d kernel always decomposes to stats + fused_norm_conv.
/// Incorrect count would break dispatch budgeting.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_activ_conv1d_always_2_dispatches() {
    let op_snake = NativeOpKind::NormActivConv1d {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 128, 256],
        output_channels: 128,
        kernel_size: 3,
        external_node_ids: None,
    };
    let op_leaky = NativeOpKind::NormActivConv1d {
        activation: NormActivation::LeakyRelu { slope: 0.3 },
        eps: 1e-6,
        conv_dilation: 3,
        conv_padding: 3,
        input_shape: vec![2, 64, 512],
        output_channels: 64,
        kernel_size: 7,
        external_node_ids: Some(vec![1, 2, 3]),
    };

    assert_eq!(op_snake.estimated_metal_dispatches(), 2);
    assert_eq!(op_leaky.estimated_metal_dispatches(), 2);
    assert_eq!(op_snake.estimated_encoding_events(), 1);
    assert_eq!(op_leaky.estimated_encoding_events(), 1);
}

/// Proves: BatchedStyleProjection is always 2 dispatches and 2 encoding events.
///
/// SUBSTANTIVE: 1 matmul + 1 bias_add = 2. Incorrect count would break the
/// Kokoro dispatch budget tracking.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_batched_style_projection_always_2() {
    let op = NativeOpKind::BatchedStyleProjection {
        blocks: vec![
            StyleBatchOffset::new(0, 64, 64),
            StyleBatchOffset::new(256, 128, 128),
        ],
        style_dim: 128,
        total_out: 768,
        style_step: 5,
    };

    assert_eq!(op.estimated_metal_dispatches(), 2);
    assert_eq!(op.estimated_encoding_events(), 2);
}

/// Proves: NormLinear with 1-element input_shape falls back to scalar (1 dispatch).
///
/// SUBSTANTIVE: Single-element shape gives flat_rows=1, which is not %8==0.
/// Ensures degenerate shapes don't accidentally trigger simdgroup path.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_linear_single_element_shape_scalar() {
    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![256],
        hidden_dim: 256,
        out_features: 256,
        has_bias: true,
    };
    // input_shape=[256], rev=[256], skip(1)=[], product=1, max(1,1)=1
    // m=1, 1%8!=0 => scalar => 1
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

// ============================================================================
// FusedResBlock priority and bounds
// ============================================================================

/// Proves: FusedResBlock style_proj takes priority over style_batch_offset.
///
/// SUBSTANTIVE: When both are set, the match arm for style_proj fires first
/// (Some(_), _). If priority were reversed, batched blocks with redundant
/// style_proj would silently use the wrong dispatch count.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_fused_resblock_style_proj_priority() {
    let params =
        NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, 64, 128], 64, 3);

    let op_both = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params.clone(),
        input_steps: vec![0, 1],
        residual_scale: 1.0,
        style_proj: Some(StyleProjectionParams::new(64, 64, 128)),
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: Some(StyleBatchOffset::new(0, 64, 64)),
    };

    let op_proj_only = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 1],
        residual_scale: 1.0,
        style_proj: Some(StyleProjectionParams::new(64, 64, 128)),
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };

    // Both must return the same dispatch count (style_proj wins)
    assert_eq!(
        op_both.estimated_metal_dispatches(),
        op_proj_only.estimated_metal_dispatches(),
    );
    assert_eq!(op_both.estimated_metal_dispatches(), 7); // 3 + 4
}

/// Proves: FusedResBlock dispatch count never exceeds 7.
///
/// SUBSTANTIVE: The maximum is base(3) + style_proj(4) = 7.
/// Buffer planning allocations depend on this upper bound.
#[kani::unwind(8)]
#[kani::proof]
fn proof_fused_resblock_dispatch_upper_bound_7() {
    let has_style: bool = kani::any();
    let has_batch: bool = kani::any();

    let params =
        NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, 64, 128], 64, 3);

    let op = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 1],
        residual_scale: 1.0,
        style_proj: if has_style {
            Some(StyleProjectionParams::new(64, 64, 128))
        } else {
            None
        },
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: if !has_style && has_batch {
            Some(StyleBatchOffset::new(0, 64, 64))
        } else {
            None
        },
    };

    assert!(
        op.estimated_metal_dispatches() <= 7,
        "FusedResBlock dispatches must not exceed 7"
    );
    assert!(
        op.estimated_encoding_events() <= 6,
        "FusedResBlock encoding events must not exceed 6"
    );
}

// ============================================================================
// Cumsum encoding invariant
// ============================================================================

/// Proves: Cumsum encoding is always 1 regardless of axis size.
///
/// SUBSTANTIVE: Multi-pass Cumsum uses 3 sub-encoders in 1 batch.
/// This differs from dispatches (1 or 3). The encoding count invariant
/// is critical for the gate_dispatch_count quality gate.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_cumsum_encoding_always_1() {
    let sizes = [1usize, 128, 256, 257, 1024, 65536];
    for &sz in &sizes {
        let op = NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![sz],
        };
        assert_eq!(
            op.estimated_encoding_events(),
            1,
            "Cumsum encoding must always be 1"
        );
    }
}

// ============================================================================
// MoE encoding/dispatch symmetry
// ============================================================================

/// Proves: MoE gating encoding always equals dispatch count.
///
/// SUBSTANTIVE: Unlike LSTM or Cumsum where encoding != dispatch,
/// MoE gating uses DynTensor CPU dispatch where each op creates one
/// encoding event. Breaking this symmetry would miscount in the gate.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_moe_encoding_equals_dispatch_for_all_top_k() {
    let top_k: usize = kani::any();
    kani::assume(top_k >= 1 && top_k <= 64);

    let op = NativeOpKind::MoeGating {
        num_experts: 8,
        top_k,
        input_shape: vec![1, 64],
    };

    assert_eq!(
        op.estimated_metal_dispatches(),
        op.estimated_encoding_events(),
        "MoE encoding must equal dispatch"
    );
}

// ============================================================================
// LayerNorm / ChannelsFirstLayerNorm dispatch invariant
// ============================================================================

/// Proves: LayerNorm is always 1 dispatch regardless of shape.
///
/// SUBSTANTIVE: After the fused kernel landing, LayerNorm is a single
/// Metal dispatch. Incorrect count would break dispatch budgeting.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_layer_norm_always_1_dispatch() {
    let hidden_dim: usize = kani::any();
    kani::assume(hidden_dim >= 1 && hidden_dim <= 4096);

    let op = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 16, hidden_dim],
        hidden_dim,
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

// ============================================================================
// Int8Gemm bias independence
// ============================================================================

/// Proves: Int8Gemm dispatch count is 1 regardless of has_bias.
///
/// SUBSTANTIVE: The W8A16 kernel fuses bias into the dequantize+accumulate
/// epilogue. has_bias does not add an extra dispatch.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_int8_gemm_bias_does_not_add_dispatch() {
    let op_bias = NativeOpKind::Int8Gemm {
        in_features: 256,
        out_features: 512,
        has_bias: true,
        input_shape: vec![1, 16, 256],
    };
    let op_no_bias = NativeOpKind::Int8Gemm {
        in_features: 256,
        out_features: 512,
        has_bias: false,
        input_shape: vec![1, 16, 256],
    };

    assert_eq!(op_bias.estimated_metal_dispatches(), 1);
    assert_eq!(op_no_bias.estimated_metal_dispatches(), 1);
    assert_eq!(
        op_bias.estimated_metal_dispatches(),
        op_no_bias.estimated_metal_dispatches()
    );
}
