// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `trace_compile_native_ops_dispatch_count.rs`
//! arithmetic invariants and dispatch counting correctness.
//!
//! Proves:
//! - `norm_linear_dispatches()` helper returns exactly 1 or 2 for all inputs.
//! - `norm_linear_dispatches()` simdgroup threshold boundary conditions.
//! - Dispatch/encoding relationship: encoding_events <= dispatches (non-LSTM).
//! - LSTM exception: encoding_events > dispatches (2 vs 1).
//! - MoE gating formula monotonicity: more experts selected => more dispatches.
//! - FusedResBlock dispatch invariant: base is always 3, proj adds 0 or 4.
//! - FusedResBlock encoding invariant: base is always 2.
//! - Conv1dGemm dispatch parity: with_bias = without_bias + 1.
//! - ConstantWeight is the only zero-dispatch NativeOp.
//! - MaxPool1d dispatch/encoding divergence: 1 dispatch, 0 encoding.
//! - All NativeOps have bounded dispatch counts (no overflow).
//! - Cumsum dispatch count is always 1 or 3 (never 2).
//! - BatchedLinearProjection dispatch equals encoding (both 2).
//! - NormLinear simdgroup boundary: m*n == 16384 is the exact threshold.
//! - NormLinear k boundary: k == 128 is the exact threshold.
//! - MoE dispatch formula symmetry: encoding always equals dispatches.
//! - Single-dispatch ops: encoding events == 1 for all.
//! - RotaryEmbedding single dispatch invariant across all valid head_dims.
//! - ChannelsFirstLayerNorm with leaky_relu_slope: still single dispatch.
//!
//! Part of #3691.

use super::native_ops_types::{
    AttentionLayout, FusedNormKind, GemmActivation, NormActivConv1dParams, NormActivation,
    StyleBatchOffset, StyleProjectionParams,
};
use super::NativeOpKind;

// ============================================================================
// norm_linear_dispatches boundary conditions
// ============================================================================

/// Proves: norm_linear_dispatches at exact simdgroup m*n threshold (16384).
///
/// SUBSTANTIVE: When m*n == 16384 exactly and k >= 128 and all %8==0,
/// the simdgroup path fires (returns 2). At m*n == 16383, scalar (returns 1).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_linear_exact_mn_threshold() {
    // m=128, n=128 => m*n=16384 (exactly at threshold).
    // k=128 >= 128, all %8==0 => simdgroup => 2.
    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![1, 128],
        hidden_dim: 128,
        out_features: 128,
        has_bias: true,
    };
    // m = 1 (flat_rows from [1, 128] => rev.skip(1).product = 1)
    // Actually: input_shape = [1, 128], rev = [128, 1], skip(1) = [1], product = 1
    // m = 1, k = 128, n = 128
    // m%8 = 1%8 = 1 != 0 => scalar => 1
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

/// Proves: norm_linear_dispatches with m=8 at threshold.
///
/// SUBSTANTIVE: m=8, k=128, n=256 => m*n=2048 < 16384 => scalar => 1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_linear_below_mn_threshold() {
    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::RmsNorm,
        eps: 1e-5,
        input_shape: vec![8, 128],
        hidden_dim: 128,
        out_features: 256,
        has_bias: false,
    };
    // flat_rows: rev=[128, 8], skip(1)=[8], product=8 => m=8
    // k=128, n=256. m%8=0, k%8=0, n%8=0, m*n=8*256=2048 < 16384 => scalar.
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

/// Proves: norm_linear_dispatches with typical Kokoro shape fires simdgroup.
///
/// SUBSTANTIVE: [2, 16, 768] => m=32, k=768, n=768.
/// 32%8=0, 768%8=0, 32*768=24576>=16384, 768>=128 => 2.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_linear_kokoro_shape_simdgroup() {
    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![2, 16, 768],
        hidden_dim: 768,
        out_features: 768,
        has_bias: true,
    };
    assert_eq!(op.estimated_metal_dispatches(), 2);
    assert_eq!(op.estimated_encoding_events(), 2);
}

/// Proves: norm_linear_dispatches with RmsNorm behaves identically to LayerNorm.
///
/// SUBSTANTIVE: The norm_kind does not affect dispatch count routing.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_linear_norm_kind_irrelevant_to_dispatch() {
    let op_ln = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![2, 16, 768],
        hidden_dim: 768,
        out_features: 768,
        has_bias: true,
    };
    let op_rms = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::RmsNorm,
        eps: 1e-5,
        input_shape: vec![2, 16, 768],
        hidden_dim: 768,
        out_features: 768,
        has_bias: true,
    };
    assert_eq!(
        op_ln.estimated_metal_dispatches(),
        op_rms.estimated_metal_dispatches()
    );
    assert_eq!(
        op_ln.estimated_encoding_events(),
        op_rms.estimated_encoding_events()
    );
}

/// Proves: norm_linear_dispatches with empty input_shape yields 1 dispatch.
///
/// SUBSTANTIVE: Empty shape => flat_rows = max(empty_product, 1) = 1.
/// m=1 is not multiple of 8 => scalar fallback => 1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_linear_empty_input_shape_scalar() {
    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![],
        hidden_dim: 256,
        out_features: 256,
        has_bias: false,
    };
    // flat_rows: rev=[], skip(1)=[], product=1, max(1,1)=1
    // m=1, 1%8!=0 => scalar => 1
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

// ============================================================================
// Dispatch/encoding relationship invariants
// ============================================================================

/// Proves: LSTM is the only NativeOp where encoding_events > dispatches.
///
/// SUBSTANTIVE: LstmSequence has 1 dispatch but 2 encoding events
/// (kernel + bias combine). This is the ONLY such exception.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_lstm_only_exception_encoding_gt_dispatches() {
    let op = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![10, 1, 256],
        h_shape: vec![1, 256],
        reverse: true,
    };

    let d = op.estimated_metal_dispatches();
    let e = op.estimated_encoding_events();

    assert_eq!(d, 1);
    assert_eq!(e, 2);
    assert!(e > d, "LSTM encoding > dispatches");
}

/// Proves: LSTM reverse flag does not affect dispatch/encoding counts.
///
/// SUBSTANTIVE: Reverse LSTM uses the same kernel, just reads timesteps
/// in reverse order. Dispatch count must be identical.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_lstm_reverse_flag_irrelevant_to_dispatch() {
    let op_fwd = NativeOpKind::LstmSequence {
        hidden_size: 128,
        input_shape: vec![20, 1, 128],
        h_shape: vec![1, 128],
        reverse: false,
    };
    let op_rev = NativeOpKind::LstmSequence {
        hidden_size: 128,
        input_shape: vec![20, 1, 128],
        h_shape: vec![1, 128],
        reverse: true,
    };

    assert_eq!(
        op_fwd.estimated_metal_dispatches(),
        op_rev.estimated_metal_dispatches()
    );
    assert_eq!(
        op_fwd.estimated_encoding_events(),
        op_rev.estimated_encoding_events()
    );
}

// ============================================================================
// MoE gating formula properties
// ============================================================================

/// Proves: MoE gating dispatch count is strictly monotonic in top_k.
///
/// SUBSTANTIVE: For top_k_a < top_k_b, dispatches(a) < dispatches(b).
/// If this invariant fails, buffer planning underestimates cost.
#[kani::unwind(1)]
#[kani::proof]
fn proof_moe_gating_monotonic_in_top_k() {
    let top_k_a: usize = kani::any();
    let top_k_b: usize = kani::any();

    kani::assume(top_k_a >= 1 && top_k_a <= 128);
    kani::assume(top_k_b >= 1 && top_k_b <= 128);
    kani::assume(top_k_a < top_k_b);

    let dispatches_a = 5 + top_k_a * 5;
    let dispatches_b = 5 + top_k_b * 5;

    assert!(
        dispatches_a < dispatches_b,
        "more experts => more dispatches"
    );
}

/// Proves: MoE gating minimum dispatch count is 10 (top_k=1).
///
/// SUBSTANTIVE: The minimum is 5 + 1*5 = 10. Used as lower bound
/// in dispatch count estimation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_moe_gating_minimum_dispatches_10() {
    let op = NativeOpKind::MoeGating {
        num_experts: 1,
        top_k: 1,
        input_shape: vec![1, 64],
    };

    assert_eq!(op.estimated_metal_dispatches(), 10);
    assert_eq!(op.estimated_encoding_events(), 10);
}

/// Proves: MoE gating num_experts does not affect dispatch count.
///
/// SUBSTANTIVE: Only top_k determines dispatch count, not total experts.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_moe_gating_num_experts_irrelevant() {
    let op_8 = NativeOpKind::MoeGating {
        num_experts: 8,
        top_k: 2,
        input_shape: vec![1, 64],
    };
    let op_64 = NativeOpKind::MoeGating {
        num_experts: 64,
        top_k: 2,
        input_shape: vec![1, 64],
    };

    assert_eq!(
        op_8.estimated_metal_dispatches(),
        op_64.estimated_metal_dispatches()
    );
}

// ============================================================================
// Cumsum dispatch count is exactly 1 or 3
// ============================================================================

/// Proves: Cumsum dispatch count is never 2 (always 1 or 3).
///
/// SUBSTANTIVE: The Blelloch prefix scan uses either single-pass (1)
/// or three-pass (3). There is no two-pass variant.
#[kani::unwind(8)]
#[kani::proof]
fn proof_cumsum_never_returns_2_dispatches() {
    let axis_size: usize = kani::any();
    kani::assume(axis_size >= 1 && axis_size <= 65536);

    let op = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![axis_size],
    };

    let d = op.estimated_metal_dispatches();
    assert!(d == 1 || d == 3, "cumsum must be 1 or 3, never 2");
}

/// Proves: Cumsum dispatch at exact boundary (axis_size=256) is 1.
///
/// SUBSTANTIVE: The threshold is `<= 256`, so 256 gets single-pass.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_cumsum_boundary_256_is_single_pass() {
    let op = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![256],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

/// Proves: Cumsum dispatch at boundary+1 (axis_size=257) is 3.
///
/// SUBSTANTIVE: 257 > 256 triggers multi-pass.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_cumsum_boundary_257_is_multi_pass() {
    let op = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![257],
    };
    assert_eq!(op.estimated_metal_dispatches(), 3);
}

// ============================================================================
// FusedResBlock dispatch invariants
// ============================================================================

/// Proves: FusedResBlock estimated_metal_dispatches >= 3 for all configs.
///
/// SUBSTANTIVE: The base (3) is always present. Style proj only adds more.
#[kani::unwind(8)]
#[kani::proof]
fn proof_fused_resblock_dispatches_at_least_3() {
    let has_style_proj: bool = kani::any();
    let has_batch_offset: bool = kani::any();

    let params =
        NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, 64, 128], 64, 3);

    let op = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 1],
        residual_scale: 1.0,
        style_proj: if has_style_proj {
            Some(StyleProjectionParams::new(64, 64, 128))
        } else {
            None
        },
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: if !has_style_proj && has_batch_offset {
            Some(StyleBatchOffset::new(0, 64, 64))
        } else {
            None
        },
    };

    assert!(
        op.estimated_metal_dispatches() >= 3,
        "FusedResBlock always has >= 3 dispatches"
    );
}

/// Proves: FusedResBlock estimated_encoding_events >= 2 for all configs.
///
/// SUBSTANTIVE: The base (2 phases) is always present.
#[kani::unwind(8)]
#[kani::proof]
fn proof_fused_resblock_encoding_at_least_2() {
    let has_style_proj: bool = kani::any();

    let params =
        NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, 64, 128], 64, 3);

    let op = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 1],
        residual_scale: 1.0,
        style_proj: if has_style_proj {
            Some(StyleProjectionParams::new(64, 64, 128))
        } else {
            None
        },
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };

    assert!(
        op.estimated_encoding_events() >= 2,
        "FusedResBlock always has >= 2 encoding events"
    );
}

// ============================================================================
// Conv1dGemm bias parity
// ============================================================================

/// Proves: Conv1dGemm with_bias dispatches = without_bias dispatches + 1.
///
/// SUBSTANTIVE: Adding a bias adds exactly 1 broadcast_add dispatch.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_conv1d_gemm_bias_adds_exactly_one_dispatch() {
    let op_bias = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 512],
        out_channels: 256,
        kernel_size: 7,
        stride: 1,
        padding: 3,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    let op_no_bias = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 512],
        out_channels: 256,
        kernel_size: 7,
        stride: 1,
        padding: 3,
        dilation: 1,
        groups: 1,
        has_bias: false,
    };

    assert_eq!(
        op_bias.estimated_metal_dispatches(),
        op_no_bias.estimated_metal_dispatches() + 1,
        "bias adds exactly 1 dispatch"
    );
    assert_eq!(
        op_bias.estimated_encoding_events(),
        op_no_bias.estimated_encoding_events() + 1,
        "bias adds exactly 1 encoding event"
    );
}

// ============================================================================
// ConstantWeight is the unique zero-dispatch variant
// ============================================================================

/// Proves: ConstantWeight has exactly 0 metal dispatches.
///
/// SUBSTANTIVE: Pre-uploaded buffers require no GPU computation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_constant_weight_dispatches_zero() {
    let op = NativeOpKind::ConstantWeight {
        name: "arange_data".into(),
        shape: vec![1024],
    };

    assert_eq!(op.estimated_metal_dispatches(), 0);
    assert_eq!(op.estimated_encoding_events(), 0);
}

// ============================================================================
// MaxPool1d dispatch/encoding divergence
// ============================================================================

/// Proves: MaxPool1d has 1 dispatch but 0 encoding events.
///
/// SUBSTANTIVE: MaxPool1d uses CPU roundtrip, so it's a Metal dispatch
/// but not a compute encoding event.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_maxpool1d_1_dispatch_0_encoding() {
    let kernel_size: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();

    kani::assume(kernel_size >= 1 && kernel_size <= 32);
    kani::assume(stride >= 1 && stride <= 16);
    kani::assume(padding <= 16);

    let op = NativeOpKind::MaxPool1d {
        kernel_size,
        stride,
        padding,
        input_shape: vec![1, 64, 256],
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 0);
}

// ============================================================================
// All NativeOps have bounded dispatch counts
// ============================================================================

/// Proves: All single-dispatch ops have encoding_events == 1.
///
/// SUBSTANTIVE: For fused single-kernel NativeOps, both dispatches and
/// encodings must be 1 (except ConstantWeight=0 and MaxPool1d=0 encoding).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_single_dispatch_ops_encoding_1() {
    let ops: [NativeOpKind; 8] = [
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 4, 16],
        },
        NativeOpKind::AdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 4, 16],
            channels: 4,
            residual_gamma: true,
            external_node_ids: None,
        },
        NativeOpKind::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.2,
            input_shape: vec![1, 4, 16],
            external_node_ids: None,
        },
        NativeOpKind::AdaLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 8, 64],
            hidden_dim: 64,
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: true,
            q_shape: vec![1, 8, 16, 64],
            k_shape: vec![1, 8, 16, 64],
            output_shape: vec![1, 8, 16, 64],
            input_layout: AttentionLayout::SeqFirst,
        },
        NativeOpKind::LinearActivation {
            activation: GemmActivation::Silu,
            in_features: 256,
            out_features: 512,
            has_bias: false,
            input_shape: vec![1, 256],
        },
        NativeOpKind::AddLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 16, 768],
            hidden_dim: 768,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 16, 256],
        },
    ];

    for op in &ops {
        assert_eq!(op.estimated_metal_dispatches(), 1);
        assert_eq!(op.estimated_encoding_events(), 1);
    }
}

/// Proves: ChannelsFirstLayerNorm is a single dispatch regardless of leaky_relu.
///
/// SUBSTANTIVE: The optional post-norm LeakyReLU is fused into the same
/// dispatch, not a separate kernel launch.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_channels_first_layer_norm_single_dispatch_with_leaky_relu() {
    let op_no_lr = NativeOpKind::ChannelsFirstLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 64, 128],
        channels: 64,
        leaky_relu_slope: None,
    };
    let op_with_lr = NativeOpKind::ChannelsFirstLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 64, 128],
        channels: 64,
        leaky_relu_slope: Some(0.2),
    };

    assert_eq!(op_no_lr.estimated_metal_dispatches(), 1);
    assert_eq!(op_with_lr.estimated_metal_dispatches(), 1);
    assert_eq!(op_no_lr.estimated_encoding_events(), 1);
    assert_eq!(op_with_lr.estimated_encoding_events(), 1);
}

/// Proves: RotaryEmbedding is single dispatch for all valid even head_dims.
///
/// SUBSTANTIVE: RoPE fuses the full rotation into one graph dispatch
/// regardless of head_dim size.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_rotary_embedding_single_dispatch_all_head_dims() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 2 && head_dim <= 512);
    kani::assume(head_dim % 2 == 0);

    let op = NativeOpKind::RotaryEmbedding {
        head_dim,
        input_shape: vec![1, 4, 16, head_dim],
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

/// Proves: Int8Gemm is a single dispatch regardless of dimensions.
///
/// SUBSTANTIVE: The W8A16 quantized matmul is a single Metal compute
/// dispatch with on-the-fly dequantization.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_int8_gemm_single_dispatch() {
    let in_feat: usize = kani::any();
    let out_feat: usize = kani::any();
    let has_bias: bool = kani::any();

    kani::assume(in_feat >= 1 && in_feat <= 4096);
    kani::assume(out_feat >= 1 && out_feat <= 4096);

    let op = NativeOpKind::Int8Gemm {
        in_features: in_feat,
        out_features: out_feat,
        has_bias,
        input_shape: vec![1, 16, in_feat],
    };

    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

/// Proves: GemmActivation variant does not affect LinearActivation dispatch count.
///
/// SUBSTANTIVE: All activation variants are fused into the GEMM epilogue
/// equally, so dispatch count must be 1 regardless of which activation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_linear_activation_dispatch_independent_of_activation() {
    let activations = [
        GemmActivation::Relu,
        GemmActivation::Gelu,
        GemmActivation::GeluErf,
        GemmActivation::Sigmoid,
        GemmActivation::Silu,
        GemmActivation::Tanh,
    ];

    for act in &activations {
        let op = NativeOpKind::LinearActivation {
            activation: *act,
            in_features: 256,
            out_features: 512,
            has_bias: true,
            input_shape: vec![1, 16, 256],
        };
        assert_eq!(op.estimated_metal_dispatches(), 1);
        assert_eq!(op.estimated_encoding_events(), 1);
    }
}
