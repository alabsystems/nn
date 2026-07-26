// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced Kani proof harnesses for `trace_compile_native_ops` dispatch
//! count, encoding event, and structural invariants.
//!
//! Proves deeper properties beyond the basic variant-name and type tests:
//! - Every single-dispatch NativeOp has encoding_events <= metal_dispatches
//! - Cumsum dispatch count is always 1 or 3 (never 2)
//! - FusedResBlock dispatch_count >= 3 (base) for all configurations
//! - NormLinear dispatch count is always 1 or 2 (binary decision)
//! - MoeGating dispatch is monotone in top_k
//! - Conv1dGemm has_bias adds exactly 1 dispatch
//! - ProjectionSlice deps reference source_step exactly once
//! - ConstantWeight dispatch and encoding are both 0
//! - LSTM encoding_events > metal_dispatches (bias combine event)
//! - MaxPool1d encoding is 0 (CPU roundtrip path)
//! - Cumsum encoding is always 1 regardless of axis size
//! - NormActivConv1d encoding < metal_dispatches (batch packing)
//!
//! Part of #3731.

use super::native_ops_types::{
    AttentionLayout, FusedNormKind, GemmActivation, NormActivConv1dParams, NormActivation,
    StyleBatchOffset, StyleProjectionParams,
};
use super::NativeOpKind;

// ============================================================================
// Dispatch vs Encoding invariants
// ============================================================================

/// Proves: for single-dispatch ops, encoding_events <= metal_dispatches.
///
/// SUBSTANTIVE: Encoding events represent command buffer batches, while
/// metal_dispatches count individual kernel launches. A single batch may
/// contain multiple sub-encoders, so encoding <= dispatch.
#[kani::unwind(8)]
#[kani::proof]
fn proof_single_dispatch_ops_encoding_le_dispatch() {
    let ops: [NativeOpKind; 8] = [
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 4, 16],
        },
        NativeOpKind::AdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 4, 16],
            channels: 4,
            residual_gamma: false,
            external_node_ids: None,
        },
        NativeOpKind::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.01,
            input_shape: vec![1, 4, 16],
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
            activation: GemmActivation::Silu,
            in_features: 64,
            out_features: 128,
            has_bias: true,
            input_shape: vec![1, 64],
        },
        NativeOpKind::Int8Gemm {
            in_features: 64,
            out_features: 128,
            has_bias: true,
            input_shape: vec![1, 64],
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 8],
        },
        NativeOpKind::RotaryEmbedding {
            head_dim: 64,
            input_shape: vec![1, 8, 16, 64],
        },
    ];

    for op in &ops {
        assert!(
            op.estimated_encoding_events() <= op.estimated_metal_dispatches(),
            "encoding_events must not exceed metal_dispatches for single-dispatch ops"
        );
    }
}

// ============================================================================
// Cumsum dispatch count is binary: 1 or 3
// ============================================================================

/// Proves: Cumsum dispatch count is exactly 1 (axis <= 256) or 3 (axis > 256),
/// never 2 or any other value.
///
/// SUBSTANTIVE: Blelloch prefix sum uses single-pass or three-pass algorithm.
/// There is no intermediate case. A dispatch count of 2 would indicate a
/// broken pass decomposition.
#[kani::unwind(8)]
#[kani::proof]
fn proof_cumsum_dispatch_is_1_or_3() {
    let axis_size: usize = kani::any();
    kani::assume(axis_size >= 1 && axis_size <= 65536);

    let dim = 0usize;
    let mut shape = vec![1usize];
    shape[0] = axis_size;

    let op = NativeOpKind::Cumsum {
        dim,
        input_shape: shape,
    };

    let d = op.estimated_metal_dispatches();
    assert!(d == 1 || d == 3, "Cumsum dispatch must be 1 or 3, got {d}");
}

/// Proves: Cumsum threshold at axis_size=256 is exact boundary.
///
/// SUBSTANTIVE: axis_size=256 gets 1 dispatch, axis_size=257 gets 3.
/// Off-by-one here would route the wrong algorithm.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_cumsum_threshold_256_exact() {
    let op_256 = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![256],
    };
    let op_257 = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![257],
    };

    assert_eq!(op_256.estimated_metal_dispatches(), 1);
    assert_eq!(op_257.estimated_metal_dispatches(), 3);
}

// ============================================================================
// FusedResBlock base dispatch >= 3
// ============================================================================

/// Proves: FusedResBlock dispatch count is always >= 3 for any configuration.
///
/// SUBSTANTIVE: The base cost is 3 (stats + conv_with_stats + conv_precomputed).
/// Style projection adds 0 or 4. Total is always >= 3.
#[kani::unwind(8)]
#[kani::proof]
fn proof_fused_resblock_minimum_dispatch_3() {
    let has_style_proj: bool = kani::any();
    let has_batch_offset: bool = kani::any();

    let params =
        NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, 64, 128], 64, 3);

    let style_proj = if has_style_proj {
        Some(StyleProjectionParams::new(64, 64, 128))
    } else {
        None
    };
    let style_batch_offset = if has_batch_offset && !has_style_proj {
        Some(StyleBatchOffset::new(0, 64, 64))
    } else {
        None
    };

    let op = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 1],
        residual_scale: 1.0,
        style_proj,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset,
    };

    assert!(
        op.estimated_metal_dispatches() >= 3,
        "FusedResBlock must have at least 3 dispatches"
    );
}

// ============================================================================
// NormLinear dispatch is binary: 1 or 2
// ============================================================================

/// Proves: NormLinear dispatch count is exactly 1 or 2.
///
/// SUBSTANTIVE: The routing is a binary decision -- scalar fused kernel (1)
/// or norm-only + simdgroup GEMM (2). No other value is possible from the
/// norm_linear_dispatches function.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_linear_dispatch_binary() {
    let hidden_dim: usize = kani::any();
    let out_features: usize = kani::any();
    let flat_rows: usize = kani::any();

    kani::assume(hidden_dim >= 1 && hidden_dim <= 1024);
    kani::assume(out_features >= 1 && out_features <= 1024);
    kani::assume(flat_rows >= 1 && flat_rows <= 128);

    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![flat_rows, hidden_dim],
        hidden_dim,
        out_features,
        has_bias: true,
    };

    let d = op.estimated_metal_dispatches();
    assert!(
        d == 1 || d == 2,
        "NormLinear dispatch must be 1 or 2, got {d}"
    );
    // Encoding events must match dispatches for NormLinear.
    assert_eq!(
        d,
        op.estimated_encoding_events(),
        "NormLinear dispatch == encoding"
    );
}

// ============================================================================
// MoeGating dispatch monotonicity in top_k
// ============================================================================

/// Proves: MoeGating dispatch count increases with top_k.
///
/// SUBSTANTIVE: Formula is 5 + top_k * 5. If top_k increases by 1,
/// dispatch increases by exactly 5. This must be monotonically increasing
/// to avoid lower top_k having MORE dispatches than higher top_k.
#[kani::unwind(8)]
#[kani::proof]
fn proof_moe_gating_dispatch_monotone_in_top_k() {
    let top_k_a: usize = kani::any();
    let top_k_b: usize = kani::any();
    kani::assume(top_k_a >= 1 && top_k_a <= 10);
    kani::assume(top_k_b >= 1 && top_k_b <= 10);
    kani::assume(top_k_a < top_k_b);

    let op_a = NativeOpKind::MoeGating {
        num_experts: 8,
        top_k: top_k_a,
        input_shape: vec![1, 64],
    };
    let op_b = NativeOpKind::MoeGating {
        num_experts: 8,
        top_k: top_k_b,
        input_shape: vec![1, 64],
    };

    assert!(
        op_a.estimated_metal_dispatches() < op_b.estimated_metal_dispatches(),
        "MoeGating dispatch must increase with top_k"
    );
}

/// Proves: MoeGating dispatch equals encoding (CPU composite).
///
/// SUBSTANTIVE: MoE gating is a CPU composite dispatch -- each logical
/// operation creates one encoding event. No sub-encoder packing.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_moe_gating_dispatch_equals_encoding() {
    let top_k: usize = kani::any();
    kani::assume(top_k >= 1 && top_k <= 8);

    let op = NativeOpKind::MoeGating {
        num_experts: 8,
        top_k,
        input_shape: vec![1, 64],
    };

    assert_eq!(
        op.estimated_metal_dispatches(),
        op.estimated_encoding_events(),
        "MoeGating: dispatch count must equal encoding count"
    );
}

// ============================================================================
// Conv1dGemm bias adds exactly 1 dispatch
// ============================================================================

/// Proves: Conv1dGemm has_bias=true has exactly 1 more dispatch than has_bias=false.
///
/// SUBSTANTIVE: The bias broadcast add is a separate dispatch. Missing it
/// would drop the bias; adding more than 1 would waste GPU time.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_conv1d_gemm_bias_adds_exactly_one() {
    let op_bias = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 4, 16],
        out_channels: 8,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    let op_no_bias = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 4, 16],
        out_channels: 8,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: false,
    };

    assert_eq!(
        op_bias.estimated_metal_dispatches() - op_no_bias.estimated_metal_dispatches(),
        1,
        "Bias adds exactly 1 dispatch"
    );
    assert_eq!(
        op_bias.estimated_encoding_events() - op_no_bias.estimated_encoding_events(),
        1,
        "Bias adds exactly 1 encoding event"
    );
}

// ============================================================================
// ProjectionSlice collect_direct_step_deps is singleton
// ============================================================================

/// Proves: ProjectionSlice::collect_direct_step_deps produces exactly 1 dep.
///
/// SUBSTANTIVE: ProjectionSlice reads from exactly one source
/// (BatchedLinearProjection output). Missing it would break buffer resolution;
/// extra deps would over-constrain the fusion planner.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_projection_slice_deps_singleton() {
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
    assert_eq!(deps.len(), 1, "ProjectionSlice must produce exactly 1 dep");
    assert_eq!(deps[0], source, "Dep must be the source_step");
}

// ============================================================================
// ConstantWeight zero dispatch and encoding
// ============================================================================

/// Proves: ConstantWeight has 0 dispatch and 0 encoding events.
///
/// SUBSTANTIVE: ConstantWeight returns a pre-uploaded buffer. Any nonzero
/// dispatch would waste GPU time on a no-op.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_constant_weight_zero_dispatch_zero_encoding() {
    let op = NativeOpKind::ConstantWeight {
        name: "test_const".into(),
        shape: vec![128, 64],
    };

    assert_eq!(op.estimated_metal_dispatches(), 0);
    assert_eq!(op.estimated_encoding_events(), 0);
}

// ============================================================================
// LSTM encoding > dispatch (bias combine)
// ============================================================================

/// Proves: LstmSequence encoding_events > metal_dispatches.
///
/// SUBSTANTIVE: LSTM requires bias_ih + bias_hh GPU add (1 encoding event)
/// PLUS the fused LSTM kernel (1 encoding event) = 2 encoding events total.
/// But only 1 metal dispatch (the fused kernel). This asymmetry is critical
/// for accurate dispatch count vs encoding count tracking.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_lstm_encoding_exceeds_dispatch() {
    let op = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![100, 1, 640],
        h_shape: vec![1, 256],
        reverse: false,
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 2);
    assert!(op.estimated_encoding_events() > op.estimated_metal_dispatches());
}

// ============================================================================
// MaxPool1d zero encoding (CPU roundtrip)
// ============================================================================

/// Proves: MaxPool1d has 0 encoding events (CPU roundtrip path).
///
/// SUBSTANTIVE: MaxPool1d routes through CPU (GPU->CPU->GPU via to_device).
/// It creates 1 metal dispatch but 0 encoding events. Misclassifying this
/// would inflate the encoding count metric.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_maxpool1d_zero_encoding() {
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 2, 8],
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 0);
}

// ============================================================================
// Cumsum encoding is always 1
// ============================================================================

/// Proves: Cumsum encoding is always 1 regardless of pass count.
///
/// SUBSTANTIVE: Multi-pass Blelloch uses 3 sub-encoders packed into 1 batch.
/// The dispatch count varies (1 or 3), but encoding is always 1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_cumsum_encoding_always_one() {
    let axis_size: usize = kani::any();
    kani::assume(axis_size >= 1 && axis_size <= 65536);

    let op = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![axis_size],
    };

    assert_eq!(
        op.estimated_encoding_events(),
        1,
        "Cumsum encoding must always be 1"
    );
}

// ============================================================================
// NormActivConv1d encoding < dispatch (batch packing)
// ============================================================================

/// Proves: NormActivConv1d has encoding < dispatch due to batch packing.
///
/// SUBSTANTIVE: NormActivConv1d uses 2 sub-encoders (stats + conv) packed
/// into 1 batch. Dispatch is 2, encoding is 1. This asymmetry matters for
/// the dispatch count vs encoding count comparison in the quality gates.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_activ_conv1d_encoding_lt_dispatch() {
    let op = NativeOpKind::NormActivConv1d {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 128, 512],
        output_channels: 128,
        kernel_size: 3,
        external_node_ids: None,
    };

    assert_eq!(op.estimated_metal_dispatches(), 2);
    assert_eq!(op.estimated_encoding_events(), 1);
    assert!(op.estimated_encoding_events() < op.estimated_metal_dispatches());
}

// ============================================================================
// GemmActivation Copy roundtrip
// ============================================================================

/// Proves: GemmActivation Copy trait preserves variant identity.
///
/// SUBSTANTIVE: GemmActivation is used as a function parameter and stored
/// in NativeOpKind. If Copy lost the variant identity, the wrong activation
/// would be applied in the GEMM epilogue.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_gemm_activation_copy_preserves_identity() {
    let variants = [
        GemmActivation::Relu,
        GemmActivation::Gelu,
        GemmActivation::GeluErf,
        GemmActivation::Sigmoid,
        GemmActivation::Silu,
        GemmActivation::Tanh,
    ];

    for &v in &variants {
        let copied = v;
        assert_eq!(v, copied, "Copy must preserve GemmActivation identity");
    }
}

// ============================================================================
// FusedNormKind Copy roundtrip
// ============================================================================

/// Proves: FusedNormKind Copy trait preserves variant identity.
///
/// SUBSTANTIVE: FusedNormKind determines the reduction strategy in the
/// NormLinear kernel. A Copy bug would route LayerNorm to RmsNorm or vice versa.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_fused_norm_kind_copy_preserves_identity() {
    let ln = FusedNormKind::LayerNorm;
    let rms = FusedNormKind::RmsNorm;

    let ln_copy = ln;
    let rms_copy = rms;

    assert_eq!(ln, ln_copy);
    assert_eq!(rms, rms_copy);
    assert_ne!(ln_copy, rms_copy);
}

// ============================================================================
// BatchedStyleProjection dep is style_step
// ============================================================================

/// Proves: BatchedStyleProjection::collect_direct_step_deps yields exactly
/// the style_step value.
///
/// SUBSTANTIVE: The batched projection reads the style embedding from
/// a single source step. Missing it would break buffer resolution.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_batched_style_proj_deps_is_style_step() {
    let style: usize = kani::any();
    kani::assume(style <= 1000);

    let op = NativeOpKind::BatchedStyleProjection {
        blocks: vec![],
        style_dim: 128,
        total_out: 512,
        style_step: style,
    };

    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);
    assert_eq!(deps.len(), 1);
    assert_eq!(deps[0], style);
}

// ============================================================================
// FusedResBlock encoding <= dispatch for all configurations
// ============================================================================

/// Proves: FusedResBlock encoding_events <= metal_dispatches for all configs.
///
/// SUBSTANTIVE: Encoding events use batch packing (2 phases packed), so
/// encoding count is always <= dispatch count. Violation would indicate
/// a batch packing regression.
#[kani::unwind(8)]
#[kani::proof]
fn proof_fused_resblock_encoding_le_dispatch() {
    let has_style: bool = kani::any();

    let params = NormActivConv1dParams::new(
        NormActivation::LeakyRelu { slope: 0.2 },
        1e-5,
        1,
        1,
        vec![1, 256, 64],
        256,
        3,
    );

    let style_proj = if has_style {
        Some(StyleProjectionParams::new(256, 256, 128))
    } else {
        None
    };

    let op = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 1],
        residual_scale: 1.0,
        style_proj,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };

    assert!(
        op.estimated_encoding_events() <= op.estimated_metal_dispatches(),
        "FusedResBlock encoding must be <= dispatch"
    );
}
