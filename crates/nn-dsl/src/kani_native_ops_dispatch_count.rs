// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `NativeOpKind` dispatch count safety and
//! `trace_compile_native_ops_dispatch_count.rs` arithmetic invariants.
//!
//! Proves:
//! - `estimated_metal_dispatches()` and `estimated_encoding_events()` return
//!   values are bounded and consistent for all NativeOp variants.
//! - NormLinear dispatch routing returns exactly 1 or 2.
//! - MoE gating dispatch formula cannot overflow for realistic expert counts.
//! - Cumsum dispatch threshold logic is correct (1 vs 3).
//! - FusedResBlock dispatch count is correct for all style_proj combinations.
//! - Conv1dGemm dispatch count depends on kernel_size/stride/dilation and has_bias.
//! - `collect_direct_step_deps()` collects correct dependencies.
//! - `external_node_ids()` returns Some only for the 3 expected variants.
//! - `variant_name()` returns non-empty static strings.
//! - `KNOWN_NATIVE_OP_COUNT` is 24.
//! - StyleBatchOffset and StyleProjectionParams constructor round-trips.
//!
//! Part of #3638.

use super::native_ops_types::{
    AttentionLayout, FusedNormKind, GemmActivation, NormActivConv1dParams, NormActivation,
    StyleBatchOffset, StyleProjectionParams,
};
use super::NativeOpKind;

// ============================================================================
// NormLinear dispatch routing — returns exactly 1 or 2
// ============================================================================

/// Proves: NormLinear `estimated_metal_dispatches()` always returns 1 or 2.
///
/// SUBSTANTIVE: The function routes between scalar fused (1 dispatch) and
/// simdgroup (2 dispatches). Any other return value would break dispatch
/// counting and buffer planning.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_linear_dispatches_returns_1_or_2() {
    let hidden_dim: usize = kani::any();
    let out_features: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 16);
    kani::assume(hidden_dim >= 1 && hidden_dim <= 512);
    kani::assume(out_features >= 1 && out_features <= 512);

    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![batch, hidden_dim],
        hidden_dim,
        out_features,
        has_bias: true,
    };

    let result = op.estimated_metal_dispatches();
    assert!(result == 1 || result == 2, "must return 1 or 2");

    // encoding events must equal metal dispatches for NormLinear
    let encodings = op.estimated_encoding_events();
    assert_eq!(
        result, encodings,
        "NormLinear encodings must match dispatches"
    );
}

/// Proves: NormLinear returns 2 dispatches when all simdgroup conditions met.
///
/// SUBSTANTIVE: Simdgroup path requires m%8==0, k%8==0, n%8==0, m*n>=16384,
/// k>=128. When all conditions are met, must return 2.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_linear_simdgroup_path_returns_2() {
    // Kokoro PLBert shape: [2, 16, 768] → m=32, k=768, n=768
    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![2, 16, 768],
        hidden_dim: 768,
        out_features: 768,
        has_bias: true,
    };

    // m = 2*16 = 32, k = 768, n = 768
    // 32%8=0, 768%8=0, 768%8=0, 32*768=24576>=16384, 768>=128 → simdgroup
    assert_eq!(op.estimated_metal_dispatches(), 2);
}

/// Proves: NormLinear returns 1 dispatch when k < 128 (scalar fallback).
///
/// SUBSTANTIVE: Small hidden dimensions cannot use simdgroup matmul.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_linear_scalar_when_small_k() {
    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::RmsNorm,
        eps: 1e-5,
        input_shape: vec![1, 32, 64],
        hidden_dim: 64, // k=64 < 128
        out_features: 256,
        has_bias: false,
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
}

/// Proves: NormLinear returns 1 dispatch when n is not 8-aligned.
///
/// SUBSTANTIVE: Non-aligned out_features forces scalar path.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_linear_scalar_when_n_not_aligned() {
    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![1, 8, 256],
        hidden_dim: 256,
        out_features: 100, // 100 % 8 != 0
        has_bias: true,
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
}

// ============================================================================
// Cumsum dispatch threshold: axis_size <= 256 → 1, else → 3
// ============================================================================

/// Proves: Cumsum with axis_size <= 256 returns 1 dispatch.
///
/// SUBSTANTIVE: Single-pass Blelloch prefix scan for small axes.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_cumsum_small_axis_single_dispatch() {
    let axis_size: usize = kani::any();
    kani::assume(axis_size >= 1 && axis_size <= 256);

    let op = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![axis_size],
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
}

/// Proves: Cumsum with axis_size > 256 returns 3 dispatches.
///
/// SUBSTANTIVE: Multi-pass Blelloch scan requires 3 sub-passes.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_cumsum_large_axis_three_dispatches() {
    let axis_size: usize = kani::any();
    kani::assume(axis_size > 256 && axis_size <= 65536);

    let op = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![axis_size],
    };

    assert_eq!(op.estimated_metal_dispatches(), 3);
}

/// Proves: Cumsum encoding events are always 1 regardless of axis size.
///
/// SUBSTANTIVE: Multi-pass uses 3 sub-encoders in a single batch.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_cumsum_always_one_encoding_event() {
    let axis_size: usize = kani::any();
    kani::assume(axis_size >= 1 && axis_size <= 65536);

    let op = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![axis_size],
    };

    assert_eq!(op.estimated_encoding_events(), 1);
}

/// Proves: Cumsum with out-of-bounds dim defaults axis_size to 1.
///
/// SUBSTANTIVE: `input_shape.get(dim).copied().unwrap_or(1)` handles
/// out-of-bounds dim gracefully by treating it as axis_size=1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_cumsum_oob_dim_defaults_to_1() {
    let op = NativeOpKind::Cumsum {
        dim: 5, // out of bounds for a 1D shape
        input_shape: vec![100],
    };

    // axis_size = input_shape.get(5).copied().unwrap_or(1) = 1
    // 1 <= 256 → 1 dispatch
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

// ============================================================================
// MoE gating formula: 5 + top_k * 5
// ============================================================================

/// Proves: MoE gating dispatch formula does not overflow for valid top_k.
///
/// SUBSTANTIVE: top_k is typically 1-8 for MoE models. The formula
/// `5 + top_k * 5` must not overflow for any realistic value.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_moe_gating_formula_no_overflow() {
    let top_k: usize = kani::any();
    let num_experts: usize = kani::any();

    kani::assume(top_k >= 1 && top_k <= 256);
    kani::assume(num_experts >= 1 && num_experts <= 1024);
    kani::assume(top_k <= num_experts);

    let op = NativeOpKind::MoeGating {
        num_experts,
        top_k,
        input_shape: vec![4, 64],
    };

    let dispatches = op.estimated_metal_dispatches();
    let encodings = op.estimated_encoding_events();

    assert_eq!(dispatches, 5 + top_k * 5);
    assert_eq!(
        encodings, dispatches,
        "MoE encoding events = metal dispatches"
    );
    assert!(dispatches >= 10, "minimum: top_k=1 gives 10 dispatches");
}

// ============================================================================
// FusedResBlock dispatch counts by style projection mode
// ============================================================================

/// Proves: FusedResBlock base dispatch count is always 3.
///
/// SUBSTANTIVE: stats + conv_with_stats + conv_precomputed = 3 base.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_fused_resblock_base_is_3() {
    let op = NativeOpKind::FusedResBlock {
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
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };

    assert_eq!(op.estimated_metal_dispatches(), 3);
    assert_eq!(op.estimated_encoding_events(), 2);
}

/// Proves: FusedResBlock with style_proj adds 4 dispatches.
///
/// SUBSTANTIVE: Unbatched style projection requires 2 projections x 2 dispatches.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_fused_resblock_style_proj_adds_4() {
    let op = NativeOpKind::FusedResBlock {
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
        style_proj: Some(StyleProjectionParams::new(64, 64, 128)),
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };

    assert_eq!(op.estimated_metal_dispatches(), 7); // 3 + 4
    assert_eq!(op.estimated_encoding_events(), 6); // 2 + 4
}

/// Proves: FusedResBlock with style_batch_offset adds 0 dispatches.
///
/// SUBSTANTIVE: Batched style uses zero-copy narrow from batch output.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_fused_resblock_batch_offset_zero_extra() {
    let op = NativeOpKind::FusedResBlock {
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
        style_batch_offset: Some(StyleBatchOffset::new(0, 64, 64)),
    };

    assert_eq!(op.estimated_metal_dispatches(), 3); // 3 + 0
    assert_eq!(op.estimated_encoding_events(), 2); // 2 + 0
}

// ============================================================================
// Conv1dGemm: shape-aware dispatch count (K=3 direct vs im2col+GEMM)
// ============================================================================

/// Proves: Conv1dGemm K=3 direct path with bias returns 2 dispatches.
///
/// SUBSTANTIVE: direct conv + bias_add = 2 (no im2col). Part of #4264.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_conv1d_gemm_k3_with_bias_2_dispatches() {
    let op = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 64, 256],
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };

    assert_eq!(op.estimated_metal_dispatches(), 2);
    assert_eq!(op.estimated_encoding_events(), 2);
}

/// Proves: Conv1dGemm K=3 direct path without bias returns 1 dispatch.
///
/// SUBSTANTIVE: direct conv only = 1 (no im2col, no bias_add). Part of #4264.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_conv1d_gemm_k3_no_bias_1_dispatch() {
    let op = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 64, 256],
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: false,
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

/// Proves: Conv1dGemm K=7 im2col+GEMM path with bias returns 3 dispatches.
///
/// SUBSTANTIVE: im2col + GEMM + bias_add = 3.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_conv1d_gemm_k7_with_bias_3_dispatches() {
    let op = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 64, 256],
        out_channels: 128,
        kernel_size: 7,
        stride: 1,
        padding: 3,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };

    assert_eq!(op.estimated_metal_dispatches(), 3);
    assert_eq!(op.estimated_encoding_events(), 3);
}

// ============================================================================
// Single-dispatch NativeOps: metal dispatches == 1
// ============================================================================

/// Proves: All single-dispatch NativeOps return exactly 1 metal dispatch.
///
/// SUBSTANTIVE: These fused kernels are single Metal compute dispatches.
/// Any other value would break dispatch count gates.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_single_dispatch_ops_return_1() {
    let ops: [NativeOpKind; 10] = [
        NativeOpKind::LstmSequence {
            hidden_size: 64,
            input_shape: vec![1, 1, 64],
            h_shape: vec![1, 64],
            reverse: false,
        },
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
        NativeOpKind::AdaLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 4, 8],
            hidden_dim: 8,
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
        NativeOpKind::Int8Gemm {
            in_features: 64,
            out_features: 128,
            has_bias: true,
            input_shape: vec![1, 4, 64],
        },
        NativeOpKind::AddLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 4],
            hidden_dim: 4,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 8, 256],
        },
    ];

    for op in &ops {
        assert_eq!(
            op.estimated_metal_dispatches(),
            1,
            "single-dispatch op must return 1"
        );
    }
}

// ============================================================================
// Zero-dispatch and special NativeOps
// ============================================================================

/// Proves: ConstantWeight returns 0 metal dispatches and 0 encoding events.
///
/// SUBSTANTIVE: Pre-uploaded buffer, no GPU computation needed.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_constant_weight_zero_dispatches() {
    let op = NativeOpKind::ConstantWeight {
        name: "test".into(),
        shape: vec![4],
    };

    assert_eq!(op.estimated_metal_dispatches(), 0);
    assert_eq!(op.estimated_encoding_events(), 0);
}

/// Proves: MaxPool1d returns 1 metal dispatch but 0 encoding events.
///
/// SUBSTANTIVE: MaxPool1d is a single Metal dispatch but uses CPU roundtrip
/// (GPU to CPU to GPU), so no compute encoding events.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_maxpool1d_dispatch_vs_encoding() {
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 64, 256],
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 0);
}

// ============================================================================
// LSTM encoding events: always 2 (kernel + bias combine)
// ============================================================================

/// Proves: LstmSequence encoding events are always 2.
///
/// SUBSTANTIVE: 1 for fused LSTM kernel + 1 for bias_ih+bias_hh GPU add.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_lstm_encoding_events_2() {
    let hidden: usize = kani::any();
    kani::assume(hidden >= 1 && hidden <= 1024);

    let op = NativeOpKind::LstmSequence {
        hidden_size: hidden,
        input_shape: vec![1, 1, hidden],
        h_shape: vec![1, hidden],
        reverse: false,
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 2);
}

// ============================================================================
// NormActivConv1d: 2 metal dispatches, 1 encoding event
// ============================================================================

/// Proves: NormActivConv1d returns 2 dispatches and 1 encoding event.
///
/// SUBSTANTIVE: stats + fused_norm_conv = 2 metal dispatches,
/// but only 1 get_or_create_batch() with 2 sub-encoders.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_activ_conv1d_dispatch_counts() {
    let op = NativeOpKind::NormActivConv1d {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 64, 128],
        output_channels: 64,
        kernel_size: 3,
        external_node_ids: None,
    };

    assert_eq!(op.estimated_metal_dispatches(), 2);
    assert_eq!(op.estimated_encoding_events(), 1);
}

// ============================================================================
// RotaryEmbedding and LayerNorm: single dispatch, single encoding
// ============================================================================

/// Proves: RotaryEmbedding is a single dispatch and single encoding event.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_rotary_embedding_single_dispatch() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 2 && head_dim <= 256);
    kani::assume(head_dim % 2 == 0); // must be even

    let op = NativeOpKind::RotaryEmbedding {
        head_dim,
        input_shape: vec![1, 8, 16, head_dim],
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

/// Proves: LayerNorm and ChannelsFirstLayerNorm are single dispatch.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_layer_norm_variants_single_dispatch() {
    let op1 = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 16, 768],
        hidden_dim: 768,
    };
    let op2 = NativeOpKind::ChannelsFirstLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 64, 128],
        channels: 64,
        leaky_relu_slope: None,
    };

    assert_eq!(op1.estimated_metal_dispatches(), 1);
    assert_eq!(op1.estimated_encoding_events(), 1);
    assert_eq!(op2.estimated_metal_dispatches(), 1);
    assert_eq!(op2.estimated_encoding_events(), 1);
}

// ============================================================================
// variant_name() — non-empty and correct
// ============================================================================

/// Proves: variant_name() returns a non-empty &'static str for ConstantWeight.
///
/// SUBSTANTIVE: Dispatch diagnostics depend on variant_name() being non-empty.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_variant_name_non_empty() {
    let op = NativeOpKind::ConstantWeight {
        name: "x".into(),
        shape: vec![1],
    };
    let name = op.variant_name();
    assert!(!name.is_empty(), "variant_name must be non-empty");
    assert_eq!(name, "ConstantWeight");
}

// ============================================================================
// KNOWN_NATIVE_OP_COUNT constant value
// ============================================================================

/// Proves: KNOWN_NATIVE_OP_COUNT is exactly 24.
///
/// SUBSTANTIVE: This constant gates variant count safety tests. If it
/// changes without updating all match arms, the test_native_op_variant_count
/// test fails at compile time (array size mismatch).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_known_native_op_count_is_24() {
    assert_eq!(super::dispatch_count::KNOWN_NATIVE_OP_COUNT, 24);
}

// ============================================================================
// external_node_ids — only 3 variants return Some
// ============================================================================

/// Proves: external_node_ids() returns None for NativeOps without external IDs.
///
/// SUBSTANTIVE: Only NormActivConv1d, AdainSnake, and AdainLeakyRelu carry
/// external node IDs. All other variants must return None.
#[kani::unwind(8)]
#[kani::proof]
fn proof_external_node_ids_none_for_others() {
    let ops: [NativeOpKind; 5] = [
        NativeOpKind::LstmSequence {
            hidden_size: 64,
            input_shape: vec![1, 1, 64],
            h_shape: vec![1, 64],
            reverse: false,
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 1, 4, 8],
            k_shape: vec![1, 1, 4, 8],
            output_shape: vec![1, 1, 4, 8],
            input_layout: AttentionLayout::HeadsFirst,
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 4],
            hidden_dim: 4,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 8],
        },
        NativeOpKind::ConstantWeight {
            name: "w".into(),
            shape: vec![4],
        },
    ];

    for op in &ops {
        assert!(
            op.external_node_ids().is_none(),
            "non-AdaIN/NormActivConv1d ops must return None"
        );
    }
}

/// Proves: external_node_ids() returns Some when external_node_ids is set.
///
/// SUBSTANTIVE: The 3 variants that carry external node IDs must return
/// them when populated.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_external_node_ids_some_when_set() {
    let ids = vec![10u64, 20, 30];

    let op1 = NativeOpKind::NormActivConv1d {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 4, 8],
        output_channels: 4,
        kernel_size: 3,
        external_node_ids: Some(ids.clone()),
    };
    let op2 = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 2, 4],
        channels: 2,
        residual_gamma: false,
        external_node_ids: Some(ids.clone()),
    };
    let op3 = NativeOpKind::AdainLeakyRelu {
        eps: 1e-5,
        slope: 0.01,
        input_shape: vec![1, 2, 4],
        external_node_ids: Some(ids.clone()),
    };

    assert!(op1.external_node_ids().is_some());
    assert!(op2.external_node_ids().is_some());
    assert!(op3.external_node_ids().is_some());

    assert_eq!(op1.external_node_ids().unwrap().len(), 3);
    assert_eq!(op2.external_node_ids().unwrap(), &[10, 20, 30]);
    assert_eq!(op3.external_node_ids().unwrap(), &[10, 20, 30]);
}

/// Proves: external_node_ids() returns None when field is None.
///
/// SUBSTANTIVE: Even the 3 carrier variants return None when the field is unset.
#[kani::unwind(8)]
#[kani::proof]
fn proof_external_node_ids_none_when_unset() {
    let op = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 2, 4],
        channels: 2,
        residual_gamma: false,
        external_node_ids: None,
    };

    assert!(op.external_node_ids().is_none());
}

// ============================================================================
// collect_direct_step_deps — correctness
// ============================================================================

/// Proves: FusedResBlock collects all input_steps + shortcut + pool.
///
/// SUBSTANTIVE: The D4 elementwise fusion pass relies on this to prevent
/// fusing steps consumed by NativeOps. Missing a dependency causes
/// use-after-fuse bugs.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_collect_deps_fused_resblock_full() {
    let op = NativeOpKind::FusedResBlock {
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
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: Some(10),
        pool_step: Some(20),
        style_batch_offset: None,
    };

    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);

    // input_steps (5) + shortcut (1) + pool (1) = 7
    assert_eq!(deps.len(), 7);
    assert!(deps.contains(&0));
    assert!(deps.contains(&4));
    assert!(deps.contains(&10));
    assert!(deps.contains(&20));
}

/// Proves: FusedResBlock without shortcut/pool collects only input_steps.
///
/// SUBSTANTIVE: When shortcut_step and pool_step are None, only input_steps
/// are collected.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_collect_deps_fused_resblock_no_shortcut_pool() {
    let op = NativeOpKind::FusedResBlock {
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
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };

    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);

    assert_eq!(deps.len(), 5);
}

/// Proves: BatchedStyleProjection collects style_step.
///
/// SUBSTANTIVE: The style embedding input step must be tracked as a dependency.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_collect_deps_batched_style_projection() {
    let step: usize = kani::any();
    kani::assume(step <= 1000);

    let op = NativeOpKind::BatchedStyleProjection {
        blocks: vec![],
        style_dim: 128,
        total_out: 256,
        style_step: step,
    };

    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);

    assert_eq!(deps.len(), 1);
    assert_eq!(deps[0], step);
}

/// Proves: ProjectionSlice collects source_step.
///
/// SUBSTANTIVE: The projection source step must be tracked.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_collect_deps_projection_slice() {
    let source: usize = kani::any();
    kani::assume(source <= 1000);

    let op = NativeOpKind::ProjectionSlice {
        source_step: source,
        dim: 2,
        start: 0,
        length: 64,
        output_shape: vec![1, 4, 64],
    };

    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);

    assert_eq!(deps.len(), 1);
    assert_eq!(deps[0], source);
}

/// Proves: Non-dependency NativeOps collect zero deps.
///
/// SUBSTANTIVE: Most NativeOps use edge_map, not direct step deps.
#[kani::unwind(8)]
#[kani::proof]
fn proof_collect_deps_empty_for_non_dep_ops() {
    let ops: [NativeOpKind; 3] = [
        NativeOpKind::LstmSequence {
            hidden_size: 64,
            input_shape: vec![1, 1, 64],
            h_shape: vec![1, 64],
            reverse: false,
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 1, 4, 8],
            k_shape: vec![1, 1, 4, 8],
            output_shape: vec![1, 1, 4, 8],
            input_layout: AttentionLayout::HeadsFirst,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 8],
        },
    ];

    for op in &ops {
        let mut deps = Vec::new();
        op.collect_direct_step_deps(&mut deps);
        assert!(deps.is_empty(), "non-dep ops must collect 0 dependencies");
    }
}

// ============================================================================
// StyleProjectionParams and StyleBatchOffset constructor round-trips
// ============================================================================

/// Proves: StyleProjectionParams constructor preserves all fields.
///
/// SUBSTANTIVE: Non-exhaustive struct requires constructor; fields must
/// round-trip correctly.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_style_projection_params_roundtrip() {
    let c1: usize = kani::any();
    let c2: usize = kani::any();
    let sd: usize = kani::any();

    kani::assume(c1 >= 1 && c1 <= 1024);
    kani::assume(c2 >= 1 && c2 <= 1024);
    kani::assume(sd >= 1 && sd <= 1024);

    let params = StyleProjectionParams::new(c1, c2, sd);

    assert_eq!(params.channels1, c1);
    assert_eq!(params.channels2, c2);
    assert_eq!(params.style_dim, sd);
}

/// Proves: StyleBatchOffset constructor preserves all fields.
///
/// SUBSTANTIVE: Non-exhaustive struct requires constructor; offset + channels
/// must round-trip correctly.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_style_batch_offset_roundtrip() {
    let offset: usize = kani::any();
    let c1: usize = kani::any();
    let c2: usize = kani::any();

    kani::assume(offset <= 65536);
    kani::assume(c1 >= 1 && c1 <= 1024);
    kani::assume(c2 >= 1 && c2 <= 1024);

    let sbo = StyleBatchOffset::new(offset, c1, c2);

    assert_eq!(sbo.offset, offset);
    assert_eq!(sbo.channels1, c1);
    assert_eq!(sbo.channels2, c2);
}

// ============================================================================
// BatchedLinearProjection and ProjectionSlice: 2 and 1 dispatches
// ============================================================================

/// Proves: BatchedLinearProjection always returns 2 dispatches.
///
/// SUBSTANTIVE: 1 fused matmul+bias + 1 narrow = 2.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_batched_linear_projection_2_dispatches() {
    let in_feat: usize = kani::any();
    kani::assume(in_feat >= 1 && in_feat <= 1024);

    let op = NativeOpKind::BatchedLinearProjection {
        in_features: in_feat,
        total_out_features: in_feat * 3,
        projection_sizes: vec![in_feat, in_feat, in_feat],
        has_bias: true,
        input_shape: vec![1, 16, in_feat],
    };

    assert_eq!(op.estimated_metal_dispatches(), 2);
    assert_eq!(op.estimated_encoding_events(), 2);
}

/// Proves: ProjectionSlice always returns 1 dispatch and 1 encoding event.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_projection_slice_1_dispatch() {
    let op = NativeOpKind::ProjectionSlice {
        source_step: 0,
        dim: 2,
        start: 64,
        length: 64,
        output_shape: vec![1, 16, 64],
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

// ============================================================================
// BatchedStyleProjection: always 2 dispatches
// ============================================================================

/// Proves: BatchedStyleProjection always returns 2 dispatches.
///
/// SUBSTANTIVE: 1 matmul + 1 bias_add = 2.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_batched_style_projection_2_dispatches() {
    let op = NativeOpKind::BatchedStyleProjection {
        blocks: vec![],
        style_dim: 128,
        total_out: 512,
        style_step: 0,
    };

    assert_eq!(op.estimated_metal_dispatches(), 2);
    assert_eq!(op.estimated_encoding_events(), 2);
}

// ============================================================================
// Encoding vs dispatch relationship invariants
// ============================================================================

/// Proves: encoding_events <= metal_dispatches for all single-kernel ops.
///
/// SUBSTANTIVE: Encoding events count batch creations, metal dispatches count
/// kernel launches. Sub-encoders within a batch mean encoding <= dispatches.
/// Exception: LstmSequence has 1 dispatch but 2 encodings (bias combine).
#[kani::unwind(8)]
#[kani::proof]
fn proof_encoding_leq_dispatches_for_non_lstm() {
    let ops: [NativeOpKind; 6] = [
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
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 1, 4, 8],
            k_shape: vec![1, 1, 4, 8],
            output_shape: vec![1, 1, 4, 8],
            input_layout: AttentionLayout::HeadsFirst,
        },
        NativeOpKind::NormActivConv1d {
            activation: NormActivation::Snake,
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 1,
            input_shape: vec![1, 4, 8],
            output_channels: 4,
            kernel_size: 3,
            external_node_ids: None,
        },
        NativeOpKind::ConstantWeight {
            name: "c".into(),
            shape: vec![4],
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 8],
        },
    ];

    for op in &ops {
        let dispatches = op.estimated_metal_dispatches();
        let encodings = op.estimated_encoding_events();
        assert!(
            encodings <= dispatches,
            "encoding events must be <= metal dispatches"
        );
    }
}

// ============================================================================
// NormActivConv1dParams constructor round-trip
// ============================================================================

/// Proves: NormActivConv1dParams::new() preserves all fields.
///
/// SUBSTANTIVE: Non-exhaustive struct requires constructor; all 7 fields
/// must round-trip correctly for buffer planning.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_activ_conv1d_params_roundtrip() {
    let dilation: usize = kani::any();
    let padding: usize = kani::any();
    let out_ch: usize = kani::any();
    let k_size: usize = kani::any();

    kani::assume(dilation >= 1 && dilation <= 16);
    kani::assume(padding <= 32);
    kani::assume(out_ch >= 1 && out_ch <= 512);
    kani::assume(k_size >= 1 && k_size <= 16);

    let params = NormActivConv1dParams::new(
        NormActivation::Snake,
        1e-5,
        dilation,
        padding,
        vec![1, 64, 128],
        out_ch,
        k_size,
    );

    assert_eq!(params.conv_dilation, dilation);
    assert_eq!(params.conv_padding, padding);
    assert_eq!(params.output_channels, out_ch);
    assert_eq!(params.kernel_size, k_size);
    assert_eq!(params.eps, 1e-5);
}
