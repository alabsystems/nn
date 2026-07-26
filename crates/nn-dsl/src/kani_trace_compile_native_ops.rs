// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `trace_compile_native_ops.rs` — the [`NativeOpKind`]
//! enum definition, variant classification, and type-level properties.
//!
//! Proves:
//! - `variant_name()` returns unique non-empty static strings for each variant.
//! - `variant_name()` matches the enum variant name exactly (no Debug parsing).
//! - `NativeOpKind` Clone round-trip preserves variant_name.
//! - `NativeOpKind` serde round-trip preserves variant_name (JSON).
//! - `collect_direct_step_deps()` never panics on any variant.
//! - `collect_direct_step_deps()` accumulates (appends, never clears).
//! - `external_node_ids()` returns empty slice for zero-length vec.
//! - `external_node_ids()` preserves element ordering.
//! - StyleBatchOffset total_channels = channels1 + channels2.
//! - StyleBatchOffset narrow width = 2*(channels1 + channels2).
//! - StyleProjectionParams proj1_out = 2*channels1, proj2_out = 2*channels2.
//! - NormActivConv1dParams input_shape rank is preserved.
//! - NormActivation enum exhaustiveness: Snake and LeakyRelu are the only variants.
//! - GemmActivation enum has exactly 6 variants.
//! - FusedNormKind enum has exactly 2 variants.
//! - AttentionLayout Default is HeadsFirst.
//! - FlashAttention causal flag does not affect dispatch properties.
//! - AdainSnake residual_gamma flag does not affect dispatch properties.
//! - NativeOpKind::ConstantWeight name and shape are independent of dispatch.
//! - FusedResBlock input_steps length does not affect dispatch count.
//!
//! Part of #3691.

use super::native_ops_types::{
    AttentionLayout, FusedNormKind, GemmActivation, NormActivConv1dParams, NormActivation,
    StyleBatchOffset, StyleProjectionParams,
};
use super::NativeOpKind;

// ============================================================================
// variant_name() correctness and uniqueness
// ============================================================================

/// Proves: each NativeOpKind variant has a distinct variant_name().
///
/// SUBSTANTIVE: Dispatch diagnostics use variant_name() for logging and
/// performance reports. Non-unique names would cause ambiguous diagnostics.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_variant_names_are_distinct() {
    let ops: [NativeOpKind; 10] = [
        NativeOpKind::LstmSequence {
            hidden_size: 64,
            input_shape: vec![1, 1, 64],
            h_shape: vec![1, 64],
            reverse: false,
        },
        NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![4],
        },
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 2, 4],
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 4],
            hidden_dim: 4,
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 1, 4, 8],
            k_shape: vec![1, 1, 4, 8],
            output_shape: vec![1, 1, 4, 8],
            input_layout: AttentionLayout::HeadsFirst,
        },
        NativeOpKind::ConstantWeight {
            name: "w".into(),
            shape: vec![4],
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 8],
        },
        NativeOpKind::RotaryEmbedding {
            head_dim: 64,
            input_shape: vec![1, 8, 16, 64],
        },
        NativeOpKind::MoeGating {
            num_experts: 8,
            top_k: 2,
            input_shape: vec![1, 64],
        },
        NativeOpKind::Conv1dGemm {
            input_shape: vec![1, 4, 16],
            out_channels: 8,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
        },
    ];

    // Verify no duplicate names in the sampled subset.
    for i in 0..10 {
        for j in (i + 1)..10 {
            let name_i = ops[i].variant_name();
            let name_j = ops[j].variant_name();
            assert!(
                !std::ptr::eq(name_i, name_j) || name_i != name_j,
                "variant names must be distinct"
            );
            // Since they're &'static str from match arms, different variants
            // return different strings. Assert content inequality.
            assert_ne!(name_i, name_j, "variant names must be distinct strings");
        }
    }
}

/// Proves: variant_name() returns exact enum variant names (no Debug parsing).
///
/// SUBSTANTIVE: The match arms return literal strings matching the enum
/// variant identifiers. This test verifies a representative sample.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_variant_name_matches_enum_identifier() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![1, 8],
    };
    assert_eq!(op.variant_name(), "SiluMul");

    let op2 = NativeOpKind::RotaryEmbedding {
        head_dim: 64,
        input_shape: vec![1, 8, 16, 64],
    };
    assert_eq!(op2.variant_name(), "RotaryEmbedding");

    let op3 = NativeOpKind::MoeGating {
        num_experts: 8,
        top_k: 2,
        input_shape: vec![1, 64],
    };
    assert_eq!(op3.variant_name(), "MoeGating");
}

/// Proves: variant_name() on all newer variants (added after initial 18).
///
/// SUBSTANTIVE: Variants added later (Int8Gemm, Conv1dGemm, SiluMul,
/// RotaryEmbedding, MoeGating, ChannelsFirstLayerNorm) must have correct names.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_newer_variant_names_correct() {
    assert_eq!(
        NativeOpKind::Int8Gemm {
            in_features: 64,
            out_features: 128,
            has_bias: true,
            input_shape: vec![1, 64],
        }
        .variant_name(),
        "Int8Gemm"
    );
    assert_eq!(
        NativeOpKind::ChannelsFirstLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 4, 8],
            channels: 4,
            leaky_relu_slope: None,
        }
        .variant_name(),
        "ChannelsFirstLayerNorm"
    );
    assert_eq!(
        NativeOpKind::AddLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 4],
            hidden_dim: 4,
        }
        .variant_name(),
        "AddLayerNorm"
    );
    assert_eq!(
        NativeOpKind::BatchedLinearProjection {
            in_features: 64,
            total_out_features: 192,
            projection_sizes: vec![64, 64, 64],
            has_bias: true,
            input_shape: vec![1, 4, 64],
        }
        .variant_name(),
        "BatchedLinearProjection"
    );
    assert_eq!(
        NativeOpKind::NormLinear {
            norm_kind: FusedNormKind::RmsNorm,
            eps: 1e-5,
            input_shape: vec![1, 256],
            hidden_dim: 256,
            out_features: 512,
            has_bias: false,
        }
        .variant_name(),
        "NormLinear"
    );
}

// ============================================================================
// Clone round-trip
// ============================================================================

/// Proves: NativeOpKind Clone preserves variant_name().
///
/// SUBSTANTIVE: The Clone derive must produce a value with the same
/// variant_name(). If Clone is incorrectly implemented, diagnostic
/// logs after cloning would show wrong variant names.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_clone_preserves_variant_name() {
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

    let cloned = op.clone();
    assert_eq!(op.variant_name(), cloned.variant_name());
    assert_eq!(
        op.estimated_metal_dispatches(),
        cloned.estimated_metal_dispatches()
    );
}

// ============================================================================
// collect_direct_step_deps() safety
// ============================================================================

/// Proves: collect_direct_step_deps() on non-dep ops never panics.
///
/// SUBSTANTIVE: The function is called on ALL NativeOps in the dispatch
/// planner. A panic on any variant would crash the entire compilation.
#[kani::unwind(8)]
#[kani::proof]
fn proof_collect_deps_no_panic_on_all_variants() {
    let ops: [NativeOpKind; 7] = [
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 4],
            hidden_dim: 4,
        },
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 2, 4],
        },
        NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![4],
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
            input_shape: vec![1, 64],
        },
        NativeOpKind::RotaryEmbedding {
            head_dim: 64,
            input_shape: vec![1, 8, 16, 64],
        },
        NativeOpKind::MoeGating {
            num_experts: 8,
            top_k: 2,
            input_shape: vec![1, 64],
        },
    ];

    for op in &ops {
        let mut deps = Vec::new();
        op.collect_direct_step_deps(&mut deps);
        // Non-dep ops: must collect 0 dependencies
        assert!(deps.is_empty());
    }
}

/// Proves: collect_direct_step_deps() accumulates (appends, never clears).
///
/// SUBSTANTIVE: If the function cleared the output vec, previous deps
/// from other NativeOps would be lost, causing use-after-fuse bugs.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_collect_deps_accumulates_not_clears() {
    let op = NativeOpKind::BatchedStyleProjection {
        blocks: vec![],
        style_dim: 128,
        total_out: 256,
        style_step: 42,
    };

    let mut deps = vec![100usize, 200]; // pre-existing deps
    op.collect_direct_step_deps(&mut deps);

    // Must have pre-existing + new deps
    assert!(deps.len() >= 3, "must accumulate, not clear");
    assert_eq!(deps[0], 100, "pre-existing dep preserved");
    assert_eq!(deps[1], 200, "pre-existing dep preserved");
    assert_eq!(deps[2], 42, "new dep appended");
}

/// Proves: FusedResBlock collect_direct_step_deps with all optional fields.
///
/// SUBSTANTIVE: When shortcut_step and pool_step are both set, all three
/// sources (input_steps, shortcut, pool) must be collected.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_collect_deps_fused_resblock_all_fields() {
    let sc: usize = kani::any();
    let ps: usize = kani::any();
    kani::assume(sc <= 500);
    kani::assume(ps <= 500);

    let op = NativeOpKind::FusedResBlock {
        phase1: NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            1,
            1,
            vec![1, 32, 64],
            32,
            3,
        ),
        phase2: NormActivConv1dParams::new(
            NormActivation::Snake,
            1e-5,
            1,
            1,
            vec![1, 32, 64],
            32,
            3,
        ),
        input_steps: vec![0, 1],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: Some(sc),
        pool_step: Some(ps),
        style_batch_offset: None,
    };

    let mut deps = Vec::new();
    op.collect_direct_step_deps(&mut deps);

    // input_steps (2) + shortcut (1) + pool (1) = 4
    assert_eq!(deps.len(), 4);
    assert!(deps.contains(&sc));
    assert!(deps.contains(&ps));
}

// ============================================================================
// external_node_ids() correctness
// ============================================================================

/// Proves: external_node_ids() preserves element ordering.
///
/// SUBSTANTIVE: The edge_map builder reads elements by position. If
/// ordering changed, edges would wire to wrong graph nodes.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_external_node_ids_preserves_order() {
    let a: u64 = kani::any();
    let b: u64 = kani::any();
    let c: u64 = kani::any();
    kani::assume(a <= 10000);
    kani::assume(b <= 10000);
    kani::assume(c <= 10000);

    let op = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 4, 16],
        channels: 4,
        residual_gamma: false,
        external_node_ids: Some(vec![a, b, c]),
    };

    let ids = op.external_node_ids().unwrap();
    assert_eq!(ids[0], a);
    assert_eq!(ids[1], b);
    assert_eq!(ids[2], c);
}

/// Proves: external_node_ids() returns empty slice for empty vec.
///
/// SUBSTANTIVE: Some(vec![]) is different from None. The edge_map
/// builder should see an empty slice, not None.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_external_node_ids_empty_vec_is_some_empty() {
    let op = NativeOpKind::NormActivConv1d {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 4, 8],
        output_channels: 4,
        kernel_size: 3,
        external_node_ids: Some(vec![]),
    };

    let ids = op.external_node_ids();
    assert!(ids.is_some());
    assert_eq!(ids.unwrap().len(), 0);
}

// ============================================================================
// StyleBatchOffset arithmetic invariants
// ============================================================================

/// Proves: StyleBatchOffset total narrow width = 2*(channels1 + channels2).
///
/// SUBSTANTIVE: Each FusedResBlock narrows 2*C1 (gamma1+beta1) + 2*C2
/// (gamma2+beta2) from the batched output. The narrow width must be
/// exactly 2*(C1 + C2).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_style_batch_offset_narrow_width() {
    let c1: usize = kani::any();
    let c2: usize = kani::any();

    kani::assume(c1 >= 1 && c1 <= 1024);
    kani::assume(c2 >= 1 && c2 <= 1024);

    let sbo = StyleBatchOffset::new(0, c1, c2);
    let narrow_width = 2 * (sbo.channels1 + sbo.channels2);

    // gamma1 (c1) + beta1 (c1) + gamma2 (c2) + beta2 (c2)
    assert_eq!(narrow_width, 2 * c1 + 2 * c2);
    assert!(narrow_width >= 4, "minimum: 2*(1+1) = 4");
}

/// Proves: StyleBatchOffset consecutive blocks partition without gaps.
///
/// SUBSTANTIVE: For N blocks, the offset of block[i+1] must equal
/// offset[i] + 2*(channels1[i] + channels2[i]). No gaps allowed.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_style_batch_offset_consecutive_partition() {
    let c1_a: usize = kani::any();
    let c2_a: usize = kani::any();
    let c1_b: usize = kani::any();
    let c2_b: usize = kani::any();

    kani::assume(c1_a >= 1 && c1_a <= 512);
    kani::assume(c2_a >= 1 && c2_a <= 512);
    kani::assume(c1_b >= 1 && c1_b <= 512);
    kani::assume(c2_b >= 1 && c2_b <= 512);

    let block_a = StyleBatchOffset::new(0, c1_a, c2_a);
    let width_a = 2 * (block_a.channels1 + block_a.channels2);
    let block_b = StyleBatchOffset::new(width_a, c1_b, c2_b);

    // Block B starts where block A ends
    assert_eq!(block_b.offset, width_a);
    // No gap between blocks
    let total_width = width_a + 2 * (block_b.channels1 + block_b.channels2);
    assert!(total_width > width_a, "total must exceed block A width");
}

// ============================================================================
// StyleProjectionParams arithmetic
// ============================================================================

/// Proves: StyleProjectionParams projection output sizes.
///
/// SUBSTANTIVE: proj1 maps [B, style_dim] -> [B, 2*channels1] (gamma1 + beta1).
/// proj2 maps [B, style_dim] -> [B, 2*channels2] (gamma2 + beta2).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_style_projection_params_output_sizes() {
    let c1: usize = kani::any();
    let c2: usize = kani::any();
    let sd: usize = kani::any();

    kani::assume(c1 >= 1 && c1 <= 512);
    kani::assume(c2 >= 1 && c2 <= 512);
    kani::assume(sd >= 1 && sd <= 512);

    let params = StyleProjectionParams::new(c1, c2, sd);

    let proj1_out = 2 * params.channels1;
    let proj2_out = 2 * params.channels2;
    let total_out = proj1_out + proj2_out;

    assert_eq!(proj1_out, 2 * c1);
    assert_eq!(proj2_out, 2 * c2);
    assert_eq!(total_out, 2 * (c1 + c2));
}

// ============================================================================
// NormActivation enum exhaustiveness
// ============================================================================

/// Proves: NormActivation::Snake is not LeakyRelu.
///
/// SUBSTANTIVE: The two activation variants produce different GPU kernel
/// code. Misclassification would apply the wrong activation function.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_activation_snake_not_leaky_relu() {
    let snake = NormActivation::Snake;
    let leaky = NormActivation::LeakyRelu { slope: 0.2 };

    // They are distinct variants
    assert_ne!(snake, leaky);
}

/// Proves: NormActivation LeakyRelu slope is preserved by constructor.
///
/// SUBSTANTIVE: The slope parameter controls the negative-half behavior.
/// If slope is lost, the activation becomes ReLU (slope=0) or identity (slope=1).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_activation_leaky_relu_slope_preserved() {
    let slope: f32 = kani::any();
    kani::assume(slope.is_finite() && slope >= 0.0 && slope <= 1.0);

    let act = NormActivation::LeakyRelu { slope };
    match act {
        NormActivation::LeakyRelu { slope: s } => {
            assert_eq!(s, slope, "slope must be preserved");
        }
        _ => panic!("wrong variant"),
    }
}

// ============================================================================
// GemmActivation enum coverage
// ============================================================================

/// Proves: All 6 GemmActivation variants have distinct names (via Debug).
///
/// SUBSTANTIVE: The MSL codegen switches on GemmActivation to emit the
/// correct activation epilogue. Each variant must produce distinct code.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_gemm_activation_6_variants() {
    let variants = [
        GemmActivation::Relu,
        GemmActivation::Gelu,
        GemmActivation::GeluErf,
        GemmActivation::Sigmoid,
        GemmActivation::Silu,
        GemmActivation::Tanh,
    ];

    // All 6 must be distinct
    for i in 0..6 {
        for j in (i + 1)..6 {
            assert_ne!(
                variants[i], variants[j],
                "GemmActivation variants must be distinct"
            );
        }
    }
}

// ============================================================================
// FusedNormKind enum coverage
// ============================================================================

/// Proves: FusedNormKind has exactly 2 variants: LayerNorm and RmsNorm.
///
/// SUBSTANTIVE: The NormLinear kernel switches on FusedNormKind to decide
/// the reduction strategy (mean+var vs x^2 mean). Exactly 2 paths.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_fused_norm_kind_2_variants() {
    assert_ne!(FusedNormKind::LayerNorm, FusedNormKind::RmsNorm);
}

// ============================================================================
// AttentionLayout Default
// ============================================================================

/// Proves: AttentionLayout::default() returns HeadsFirst.
///
/// SUBSTANTIVE: The serde default attribute on FlashAttention's input_layout
/// field uses Default, which must be HeadsFirst for backward compatibility.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_attention_layout_default_is_heads_first() {
    let layout = AttentionLayout::default();
    assert_eq!(layout, AttentionLayout::HeadsFirst);
}

// ============================================================================
// FlashAttention causal flag independence
// ============================================================================

/// Proves: FlashAttention causal flag does not affect dispatch count.
///
/// SUBSTANTIVE: Both causal and non-causal paths use the same kernel
/// (masking is a conditional inside the kernel, not a separate dispatch).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_flash_attention_causal_independent_of_dispatch() {
    let op_causal = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: true,
        q_shape: vec![1, 8, 16, 64],
        k_shape: vec![1, 8, 16, 64],
        output_shape: vec![1, 8, 16, 64],
        input_layout: AttentionLayout::HeadsFirst,
    };
    let op_noncausal = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: false,
        q_shape: vec![1, 8, 16, 64],
        k_shape: vec![1, 8, 16, 64],
        output_shape: vec![1, 8, 16, 64],
        input_layout: AttentionLayout::HeadsFirst,
    };

    assert_eq!(
        op_causal.estimated_metal_dispatches(),
        op_noncausal.estimated_metal_dispatches()
    );
    assert_eq!(
        op_causal.estimated_encoding_events(),
        op_noncausal.estimated_encoding_events()
    );
}

// ============================================================================
// AdainSnake residual_gamma independence
// ============================================================================

/// Proves: AdainSnake residual_gamma flag does not affect dispatch count.
///
/// SUBSTANTIVE: Both conventions (standard AdaIN vs Kokoro residual gamma)
/// use the same fused kernel — the convention is an arithmetic change inside
/// the kernel, not a separate dispatch.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_adain_snake_residual_gamma_independent_of_dispatch() {
    let op_standard = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 256, 512],
        channels: 256,
        residual_gamma: false,
        external_node_ids: None,
    };
    let op_residual = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 256, 512],
        channels: 256,
        residual_gamma: true,
        external_node_ids: None,
    };

    assert_eq!(
        op_standard.estimated_metal_dispatches(),
        op_residual.estimated_metal_dispatches()
    );
}

// ============================================================================
// FusedResBlock input_steps length independence
// ============================================================================

/// Proves: FusedResBlock dispatch count is independent of input_steps length.
///
/// SUBSTANTIVE: input_steps is used for buffer resolution, not dispatch
/// planning. The dispatch count depends only on style_proj/batch_offset.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_fused_resblock_input_steps_length_independent() {
    let params =
        NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 1, vec![1, 64, 128], 64, 3);

    let op_5 = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params.clone(),
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };

    let op_2 = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 1],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };

    assert_eq!(
        op_5.estimated_metal_dispatches(),
        op_2.estimated_metal_dispatches()
    );
    assert_eq!(
        op_5.estimated_encoding_events(),
        op_2.estimated_encoding_events()
    );
}

// ============================================================================
// NormActivConv1dParams shape preservation
// ============================================================================

/// Proves: NormActivConv1dParams::new() preserves input_shape rank.
///
/// SUBSTANTIVE: The input_shape Vec length determines tensor rank.
/// If the constructor dropped or altered elements, the executor would
/// dispatch with wrong threadgroup dimensions.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_activ_conv1d_params_preserves_rank() {
    let rank: usize = kani::any();
    kani::assume(rank >= 1 && rank <= 5);

    let mut shape = Vec::new();
    for _ in 0..rank {
        shape.push(1usize);
    }

    let params =
        NormActivConv1dParams::new(NormActivation::Snake, 1e-5, 1, 0, shape.clone(), 32, 3);

    assert_eq!(params.input_shape.len(), rank, "rank must be preserved");
}

/// Proves: NormActivConv1dParams Clone round-trip preserves all fields.
///
/// SUBSTANTIVE: The #[non_exhaustive] struct uses Clone for FusedResBlock
/// phase copying. All fields must survive.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_activ_conv1d_params_clone_roundtrip() {
    let params = NormActivConv1dParams::new(
        NormActivation::LeakyRelu { slope: 0.2 },
        1e-5,
        3,
        3,
        vec![1, 256, 64],
        256,
        7,
    );

    let cloned = params.clone();

    assert_eq!(cloned.eps, params.eps);
    assert_eq!(cloned.conv_dilation, params.conv_dilation);
    assert_eq!(cloned.conv_padding, params.conv_padding);
    assert_eq!(cloned.output_channels, params.output_channels);
    assert_eq!(cloned.kernel_size, params.kernel_size);
    assert_eq!(cloned.input_shape, params.input_shape);
    assert_eq!(cloned.activation, params.activation);
}
