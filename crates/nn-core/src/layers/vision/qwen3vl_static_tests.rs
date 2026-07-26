// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compile-time static tests for Qwen3-VL-8B (dense) and Qwen3-VL-30B-A3B
//! (MoE) multimodal architectures.
//!
//! Qwen3-VL combines a ViT vision encoder with a decoder-only language model.
//! The 8B variant uses dense SwiGLU MLPs; the 30B-A3B variant replaces the
//! dense MLP with a Mixture-of-Experts layer (64 experts, top-8 routing,
//! ~3B active parameters per token).
//!
//! Both variants use:
//! - Interleaved M-ROPE with 3 sections (temporal, height, width)
//! - Grouped Query Attention (GQA)
//! - Conv3d patch embedding (temporal_patch=2, spatial_patch=14)
//! - DeepStack multi-level ViT feature fusion
//!
//! These tests validate architectural invariants at compile time (const
//! assertions) and test time (f32/derived checks) to catch configuration
//! errors before any weights are loaded or inference is run.
//!
//! References:
//! - Qwen3-VL technical report (arXiv:2502.13923)
//! - `crates/nn-qwen3/src/config.rs` (Qwen3Config)
//! - `crates/nn-qwen3/src/moe.rs` (Qwen3MoeConfig)
//! - `crates/nn-core/src/nn/attention/interleaved_mrope.rs`

// =============================================================================
// Qwen3-VL-8B (dense) configuration constants
// =============================================================================

/// Language model hidden dimension.
const Q3VL_8B_HIDDEN_DIM: usize = 3584;

/// Number of attention heads.
const Q3VL_8B_NUM_HEADS: usize = 28;

/// Number of key-value heads (GQA).
const Q3VL_8B_NUM_KV_HEADS: usize = 4;

/// Number of transformer decoder layers.
const Q3VL_8B_NUM_LAYERS: usize = 28;

/// SwiGLU FFN intermediate size.
const Q3VL_8B_INTERMEDIATE_SIZE: usize = 18944;

/// Vocabulary size (shared across all Qwen3 variants).
const Q3VL_8B_VOCAB_SIZE: usize = 151_936;

/// Head dimension (constant = 128 across all Qwen3 variants).
const Q3VL_8B_HEAD_DIM: usize = Q3VL_8B_HIDDEN_DIM / Q3VL_8B_NUM_HEADS;

/// GQA group ratio (num_heads / num_kv_heads).
const Q3VL_8B_GQA_RATIO: usize = Q3VL_8B_NUM_HEADS / Q3VL_8B_NUM_KV_HEADS;

/// Maximum position embeddings.
const Q3VL_8B_MAX_POSITION: usize = 32_768;

// =============================================================================
// Qwen3-VL-30B-A3B (MoE) configuration constants
// =============================================================================

/// Language model hidden dimension (same as 8B).
const Q3VL_30B_HIDDEN_DIM: usize = 3584;

/// Number of attention heads.
const Q3VL_30B_NUM_HEADS: usize = 28;

/// Number of key-value heads (GQA).
const Q3VL_30B_NUM_KV_HEADS: usize = 4;

/// Number of transformer decoder layers.
const Q3VL_30B_NUM_LAYERS: usize = 48;

/// Per-expert SwiGLU FFN intermediate size.
const Q3VL_30B_EXPERT_INTERMEDIATE_SIZE: usize = 2560;

/// Total number of experts per MoE layer.
const Q3VL_30B_NUM_EXPERTS: usize = 64;

/// Number of active experts per token (top-k).
const Q3VL_30B_TOP_K: usize = 8;

/// Vocabulary size (shared).
const Q3VL_30B_VOCAB_SIZE: usize = 151_936;

/// Head dimension (constant = 128 across all Qwen3 variants).
const Q3VL_30B_HEAD_DIM: usize = Q3VL_30B_HIDDEN_DIM / Q3VL_30B_NUM_HEADS;

/// GQA group ratio.
const Q3VL_30B_GQA_RATIO: usize = Q3VL_30B_NUM_HEADS / Q3VL_30B_NUM_KV_HEADS;

/// Maximum position embeddings.
const Q3VL_30B_MAX_POSITION: usize = 32_768;

// =============================================================================
// Shared vision encoder configuration (Qwen3-VL ViT)
// =============================================================================

/// Vision encoder hidden dimension (shared across both variants).
const VIT_HIDDEN_DIM: usize = 1280;

/// Vision encoder number of attention heads.
const VIT_NUM_HEADS: usize = 16;

/// Vision encoder number of transformer layers.
const VIT_NUM_LAYERS: usize = 32;

/// Vision encoder intermediate (MLP) size.
const VIT_INTERMEDIATE_SIZE: usize = 5120;

/// Spatial patch size in pixels (square patches).
const VIT_SPATIAL_PATCH: usize = 14;

/// Temporal patch size for video frames.
const VIT_TEMPORAL_PATCH: usize = 2;

/// Number of input channels (RGB).
const VIT_NUM_CHANNELS: usize = 3;

/// Vision encoder head dimension.
const VIT_HEAD_DIM: usize = VIT_HIDDEN_DIM / VIT_NUM_HEADS;

// =============================================================================
// Interleaved M-ROPE constants
// =============================================================================

/// M-ROPE has exactly 3 sections: temporal, height, width.
const MROPE_NUM_SECTIONS: usize = 3;

/// Pairs per M-ROPE section (head_dim / 2 / 3 = 128 / 2 / 3 is NOT integer
/// for head_dim=128. Qwen3-VL uses head_dim=128 with interleaved M-ROPE
/// which requires head_dim divisible by 6. 128/6 is not integer, so Qwen3-VL
/// uses a different allocation: the first two sections get ceil(head_dim/6)
/// pairs and the last section gets the remainder.)
///
/// NOTE: For the standard interleaved M-ROPE in nn (InterleavedMRoPE), head_dim
/// must be divisible by 6. Qwen3-VL's actual implementation distributes remainder
/// pairs unevenly. This static test verifies the architectural intent.
const MROPE_PAIRS_TOTAL: usize = Q3VL_8B_HEAD_DIM / 2; // 64 pairs

// =============================================================================
// Conv3d patch embedding constants
// =============================================================================

/// Conv3d kernel: (temporal_patch, spatial_patch, spatial_patch).
/// Input channels = VIT_NUM_CHANNELS * VIT_TEMPORAL_PATCH (merged temporal dim).
const CONV3D_IN_CHANNELS: usize = VIT_NUM_CHANNELS * VIT_TEMPORAL_PATCH;

/// Conv3d output channels = vision hidden dim.
const CONV3D_OUT_CHANNELS: usize = VIT_HIDDEN_DIM;

/// Conv3d total kernel volume: temporal * spatial * spatial.
const CONV3D_KERNEL_VOLUME: usize = VIT_TEMPORAL_PATCH * VIT_SPATIAL_PATCH * VIT_SPATIAL_PATCH;

// =============================================================================
// DeepStack fusion constants
// =============================================================================

/// Number of intermediate ViT layers fused by DeepStack.
/// Qwen3-VL typically fuses 4 layers (shallow + mid + deep + final).
const DEEPSTACK_NUM_LAYERS: usize = 4;

/// DeepStack input dimension = VIT_HIDDEN_DIM per fused layer.
/// Concatenated dimension = DEEPSTACK_NUM_LAYERS * VIT_HIDDEN_DIM.
const DEEPSTACK_CONCAT_DIM: usize = DEEPSTACK_NUM_LAYERS * VIT_HIDDEN_DIM;

// =============================================================================
// Compile-time const assertions: Qwen3-VL-8B (dense)
// =============================================================================

// --- GQA divisibility ---

const _: () = assert!(
    Q3VL_8B_NUM_HEADS.is_multiple_of(Q3VL_8B_NUM_KV_HEADS),
    "Qwen3-VL-8B: num_heads must be divisible by num_kv_heads (GQA)"
);

const _: () = assert!(
    Q3VL_8B_GQA_RATIO == 7,
    "Qwen3-VL-8B: GQA ratio must be 7 (28 heads / 4 kv_heads)"
);

const _: () = assert!(
    Q3VL_8B_NUM_KV_HEADS > 0,
    "Qwen3-VL-8B: num_kv_heads must be > 0"
);

// --- Head dimension ---

const _: () = assert!(
    Q3VL_8B_HIDDEN_DIM.is_multiple_of(Q3VL_8B_NUM_HEADS),
    "Qwen3-VL-8B: hidden_dim must be divisible by num_heads"
);

const _: () = assert!(
    Q3VL_8B_HEAD_DIM == 128,
    "Qwen3-VL-8B: head_dim must be 128 (Qwen3 constant)"
);

// --- Layer count ---

const _: () = assert!(
    Q3VL_8B_NUM_LAYERS > 0,
    "Qwen3-VL-8B: num_layers must be > 0"
);

// --- FFN intermediate size ---

const _: () = assert!(
    Q3VL_8B_INTERMEDIATE_SIZE > Q3VL_8B_HIDDEN_DIM,
    "Qwen3-VL-8B: intermediate_size must exceed hidden_dim (FFN expansion)"
);

const _: () = assert!(
    Q3VL_8B_INTERMEDIATE_SIZE > 0,
    "Qwen3-VL-8B: intermediate_size must be > 0"
);

// --- Vocabulary ---

const _: () = assert!(
    Q3VL_8B_VOCAB_SIZE > 0,
    "Qwen3-VL-8B: vocab_size must be > 0"
);

const _: () = assert!(
    Q3VL_8B_VOCAB_SIZE == 151_936,
    "Qwen3-VL-8B: vocab_size must be 151936 (Qwen3 standard)"
);

// --- Position embeddings ---

const _: () = assert!(
    Q3VL_8B_MAX_POSITION > 0,
    "Qwen3-VL-8B: max_position_embeddings must be > 0"
);

// =============================================================================
// Compile-time const assertions: Qwen3-VL-30B-A3B (MoE)
// =============================================================================

// --- GQA divisibility ---

const _: () = assert!(
    Q3VL_30B_NUM_HEADS.is_multiple_of(Q3VL_30B_NUM_KV_HEADS),
    "Qwen3-VL-30B-A3B: num_heads must be divisible by num_kv_heads (GQA)"
);

const _: () = assert!(
    Q3VL_30B_GQA_RATIO == 7,
    "Qwen3-VL-30B-A3B: GQA ratio must be 7 (28 heads / 4 kv_heads)"
);

const _: () = assert!(
    Q3VL_30B_NUM_KV_HEADS > 0,
    "Qwen3-VL-30B-A3B: num_kv_heads must be > 0"
);

// --- Head dimension ---

const _: () = assert!(
    Q3VL_30B_HIDDEN_DIM.is_multiple_of(Q3VL_30B_NUM_HEADS),
    "Qwen3-VL-30B-A3B: hidden_dim must be divisible by num_heads"
);

const _: () = assert!(
    Q3VL_30B_HEAD_DIM == 128,
    "Qwen3-VL-30B-A3B: head_dim must be 128 (Qwen3 constant)"
);

// --- Layer count ---

const _: () = assert!(
    Q3VL_30B_NUM_LAYERS > 0,
    "Qwen3-VL-30B-A3B: num_layers must be > 0"
);

const _: () = assert!(
    Q3VL_30B_NUM_LAYERS > Q3VL_8B_NUM_LAYERS,
    "Qwen3-VL-30B-A3B: must have more layers than 8B (deeper with MoE)"
);

// --- MoE configuration ---

const _: () = assert!(
    Q3VL_30B_NUM_EXPERTS > 0,
    "Qwen3-VL-30B-A3B: num_experts must be > 0"
);

const _: () = assert!(Q3VL_30B_TOP_K > 0, "Qwen3-VL-30B-A3B: top_k must be > 0");

const _: () = assert!(
    Q3VL_30B_TOP_K <= Q3VL_30B_NUM_EXPERTS,
    "Qwen3-VL-30B-A3B: top_k must be <= num_experts"
);

const _: () = assert!(
    Q3VL_30B_NUM_EXPERTS == 64,
    "Qwen3-VL-30B-A3B: must have exactly 64 experts"
);

const _: () = assert!(
    Q3VL_30B_TOP_K == 8,
    "Qwen3-VL-30B-A3B: must route to 8 experts per token"
);

// --- Expert FFN size ---

const _: () = assert!(
    Q3VL_30B_EXPERT_INTERMEDIATE_SIZE > 0,
    "Qwen3-VL-30B-A3B: expert_intermediate_size must be > 0"
);

// --- Vocabulary ---

const _: () = assert!(
    Q3VL_30B_VOCAB_SIZE > 0,
    "Qwen3-VL-30B-A3B: vocab_size must be > 0"
);

const _: () = assert!(
    Q3VL_30B_VOCAB_SIZE == Q3VL_8B_VOCAB_SIZE,
    "Qwen3-VL: vocab_size must be consistent across 8B and 30B variants"
);

// --- Hidden dim consistency between variants ---

const _: () = assert!(
    Q3VL_30B_HIDDEN_DIM == Q3VL_8B_HIDDEN_DIM,
    "Qwen3-VL: hidden_dim must be consistent across 8B and 30B variants"
);

// --- Position embeddings ---

const _: () = assert!(
    Q3VL_30B_MAX_POSITION > 0,
    "Qwen3-VL-30B-A3B: max_position_embeddings must be > 0"
);

// =============================================================================
// Compile-time const assertions: Vision encoder (ViT)
// =============================================================================

// --- Head dimension ---

const _: () = assert!(
    VIT_HIDDEN_DIM.is_multiple_of(VIT_NUM_HEADS),
    "Qwen3-VL ViT: hidden_dim must be divisible by num_heads"
);

const _: () = assert!(
    VIT_HEAD_DIM == 80,
    "Qwen3-VL ViT: head_dim must be 80 (1280 / 16)"
);

// --- Dimensions positive ---

const _: () = assert!(VIT_HIDDEN_DIM > 0, "Qwen3-VL ViT: hidden_dim must be > 0");
const _: () = assert!(VIT_NUM_HEADS > 0, "Qwen3-VL ViT: num_heads must be > 0");
const _: () = assert!(VIT_NUM_LAYERS > 0, "Qwen3-VL ViT: num_layers must be > 0");
const _: () = assert!(
    VIT_INTERMEDIATE_SIZE > 0,
    "Qwen3-VL ViT: intermediate_size must be > 0"
);

// --- MLP expansion ---

const _: () = assert!(
    VIT_INTERMEDIATE_SIZE > VIT_HIDDEN_DIM,
    "Qwen3-VL ViT: intermediate_size must exceed hidden_dim (MLP expansion)"
);

// --- RGB channels ---

const _: () = assert!(
    VIT_NUM_CHANNELS == 3,
    "Qwen3-VL ViT: num_channels must be 3 (RGB)"
);

// =============================================================================
// Compile-time const assertions: Conv3d patch embedding
// =============================================================================

// --- Patch sizes positive ---

const _: () = assert!(
    VIT_SPATIAL_PATCH > 0,
    "Qwen3-VL: spatial patch size must be > 0"
);

const _: () = assert!(
    VIT_TEMPORAL_PATCH > 0,
    "Qwen3-VL: temporal patch size must be > 0"
);

// --- Conv3d channel computation ---

const _: () = assert!(
    CONV3D_IN_CHANNELS == 6,
    "Qwen3-VL: Conv3d input channels must be 3 * 2 = 6 (RGB * temporal_patch)"
);

const _: () = assert!(
    CONV3D_OUT_CHANNELS == VIT_HIDDEN_DIM,
    "Qwen3-VL: Conv3d output channels must equal ViT hidden_dim"
);

// --- Kernel volume ---

const _: () = assert!(
    CONV3D_KERNEL_VOLUME == 2 * 14 * 14,
    "Qwen3-VL: Conv3d kernel volume must be temporal * spatial^2 = 392"
);

const _: () = assert!(
    CONV3D_KERNEL_VOLUME > 0,
    "Qwen3-VL: Conv3d kernel volume must be > 0"
);

// =============================================================================
// Compile-time const assertions: M-ROPE
// =============================================================================

// --- M-ROPE section count ---

const _: () = assert!(
    MROPE_NUM_SECTIONS == 3,
    "Qwen3-VL M-ROPE: must have exactly 3 sections (temporal, height, width)"
);

// --- Head dim / 2 gives total pairs ---

const _: () = assert!(
    MROPE_PAIRS_TOTAL == 64,
    "Qwen3-VL M-ROPE: total pairs must be head_dim/2 = 64"
);

// --- M-ROPE sections must cover all pairs ---
// 3 sections cover 64 pairs total (not necessarily equal: 22+21+21 or similar).
// Verify that num_sections * floor(pairs/sections) + remainder = total.

const _: () = assert!(
    (MROPE_PAIRS_TOTAL / MROPE_NUM_SECTIONS) * MROPE_NUM_SECTIONS
        + (MROPE_PAIRS_TOTAL % MROPE_NUM_SECTIONS)
        == MROPE_PAIRS_TOTAL,
    "Qwen3-VL M-ROPE: sections must account for all pairs"
);

// =============================================================================
// Compile-time const assertions: DeepStack fusion
// =============================================================================

const _: () = assert!(
    DEEPSTACK_NUM_LAYERS > 0,
    "Qwen3-VL DeepStack: num_fused_layers must be > 0"
);

const _: () = assert!(
    DEEPSTACK_CONCAT_DIM == 5120,
    "Qwen3-VL DeepStack: concatenated dim must be 4 * 1280 = 5120"
);

const _: () = assert!(
    DEEPSTACK_NUM_LAYERS <= VIT_NUM_LAYERS,
    "Qwen3-VL DeepStack: fused layers must not exceed total ViT layers"
);

// =============================================================================
// Runtime tests
// =============================================================================

#[test]
fn test_qwen3vl_8b_ffn_expansion_ratio() {
    // SwiGLU effective expansion: intermediate_size / hidden_dim
    // 18944 / 3584 = 5.29 (approximately 8/3 * hidden_dim for SwiGLU)
    let ratio = Q3VL_8B_INTERMEDIATE_SIZE as f64 / Q3VL_8B_HIDDEN_DIM as f64;
    assert!(
        (4.0..=8.0).contains(&ratio),
        "Qwen3-VL-8B: FFN expansion ratio should be in [4, 8], got {ratio:.2}"
    );
    assert!(
        ratio.is_finite(),
        "Qwen3-VL-8B: FFN expansion ratio must be finite"
    );
}

#[test]
fn test_qwen3vl_30b_moe_active_params_estimate() {
    // Active parameters per token ~ (top_k / num_experts) * total_expert_params
    // Each expert has: gate_proj(H, I) + up_proj(H, I) + down_proj(I, H) = 3*H*I params
    // Total expert params = num_experts * 3 * hidden_dim * expert_intermediate_size
    // Active expert params = top_k * 3 * hidden_dim * expert_intermediate_size
    let active_expert_params =
        Q3VL_30B_TOP_K * 3 * Q3VL_30B_HIDDEN_DIM * Q3VL_30B_EXPERT_INTERMEDIATE_SIZE;
    // ~220M active expert params per layer.
    // With 48 layers + attention + embeddings, total active ~ 3B.
    assert!(
        active_expert_params > 0,
        "Qwen3-VL-30B-A3B: active expert params must be > 0"
    );

    // Sparsity ratio: top_k / num_experts
    let sparsity = Q3VL_30B_TOP_K as f64 / Q3VL_30B_NUM_EXPERTS as f64;
    assert!(
        sparsity > 0.0 && sparsity < 1.0,
        "Qwen3-VL-30B-A3B: sparsity ratio must be in (0, 1), got {sparsity:.3}"
    );
    assert!(
        (sparsity - 0.125).abs() < 0.001,
        "Qwen3-VL-30B-A3B: sparsity ratio should be 8/64 = 0.125, got {sparsity:.4}"
    );
}

#[test]
fn test_qwen3vl_gqa_dimensions() {
    // 8B: KV projection size = num_kv_heads * head_dim
    let kv_proj_8b = Q3VL_8B_NUM_KV_HEADS * Q3VL_8B_HEAD_DIM;
    assert_eq!(kv_proj_8b, 512, "8B KV projection dim: 4 * 128 = 512");

    // 30B: same structure
    let kv_proj_30b = Q3VL_30B_NUM_KV_HEADS * Q3VL_30B_HEAD_DIM;
    assert_eq!(kv_proj_30b, 512, "30B KV projection dim: 4 * 128 = 512");

    // Q projection size = num_heads * head_dim = hidden_dim
    let q_proj_8b = Q3VL_8B_NUM_HEADS * Q3VL_8B_HEAD_DIM;
    assert_eq!(q_proj_8b, Q3VL_8B_HIDDEN_DIM);

    let q_proj_30b = Q3VL_30B_NUM_HEADS * Q3VL_30B_HEAD_DIM;
    assert_eq!(q_proj_30b, Q3VL_30B_HIDDEN_DIM);
}

#[test]
fn test_qwen3vl_conv3d_patch_embedding() {
    // Conv3d: [in_c, out_c, T, H, W] = [6, 1280, 2, 14, 14]
    let kernel_params = CONV3D_IN_CHANNELS * CONV3D_OUT_CHANNELS * CONV3D_KERNEL_VOLUME;
    // 6 * 1280 * 392 = 3,010,560 parameters in the patch embedding conv
    assert_eq!(kernel_params, 6 * 1280 * 392);
    assert!(kernel_params > 0);

    // Spatial downsampling factor = spatial_patch
    let spatial_downsample = VIT_SPATIAL_PATCH;
    assert_eq!(spatial_downsample, 14);

    // Temporal downsampling factor = temporal_patch
    let temporal_downsample = VIT_TEMPORAL_PATCH;
    assert_eq!(temporal_downsample, 2);
}

#[test]
fn test_qwen3vl_vision_lm_interface() {
    // Vision encoder outputs VIT_HIDDEN_DIM (1280) per patch.
    // Language model expects Q3VL_8B_HIDDEN_DIM (3584) as input.
    // These differ, so a projection layer (or DeepStack) bridges the gap.
    assert_ne!(
        VIT_HIDDEN_DIM, Q3VL_8B_HIDDEN_DIM,
        "Vision dim != LM dim: projection is required"
    );

    // DeepStack fusion: 4 layers * 1280 = 5120 -> projected to LM hidden dim.
    // The projection weight shape is [lm_hidden, deepstack_concat].
    assert_eq!(DEEPSTACK_CONCAT_DIM, 5120);
    assert!(DEEPSTACK_CONCAT_DIM > Q3VL_8B_HIDDEN_DIM);
}

#[test]
fn test_qwen3vl_mrope_pair_distribution() {
    // 128-dim head has 64 pairs. 3 sections.
    // 64 / 3 = 21 remainder 1.
    // Distribution: sections get 22, 21, 21 pairs (or similar uneven split).
    let base_pairs = MROPE_PAIRS_TOTAL / MROPE_NUM_SECTIONS;
    let remainder = MROPE_PAIRS_TOTAL % MROPE_NUM_SECTIONS;

    assert_eq!(base_pairs, 21, "Each section gets at least 21 pairs");
    assert_eq!(remainder, 1, "One extra pair distributed to one section");

    // Verify all pairs are accounted for
    let total = base_pairs * MROPE_NUM_SECTIONS + remainder;
    assert_eq!(total, MROPE_PAIRS_TOTAL);
}

#[test]
fn test_qwen3vl_30b_expert_routing_coverage() {
    // With 64 experts and top-8, each token activates 8/64 = 12.5% of experts.
    // For a batch of N tokens, expected load per expert = N * top_k / num_experts.
    let tokens = 1024usize;
    let expected_load = tokens * Q3VL_30B_TOP_K / Q3VL_30B_NUM_EXPERTS;
    assert_eq!(
        expected_load, 128,
        "Expected 128 tokens per expert for 1024-token batch"
    );

    // Router gate weight shape: [hidden_dim, num_experts]
    let gate_params = Q3VL_30B_HIDDEN_DIM * Q3VL_30B_NUM_EXPERTS;
    assert_eq!(gate_params, 3584 * 64);
    assert!(gate_params > 0);
}

#[test]
fn test_qwen3vl_variant_consistency() {
    // Both variants share the same vision encoder, vocab, and head_dim.
    assert_eq!(
        Q3VL_8B_HEAD_DIM, Q3VL_30B_HEAD_DIM,
        "Head dim must match across variants"
    );
    assert_eq!(
        Q3VL_8B_VOCAB_SIZE, Q3VL_30B_VOCAB_SIZE,
        "Vocab size must match"
    );
    assert_eq!(
        Q3VL_8B_HIDDEN_DIM, Q3VL_30B_HIDDEN_DIM,
        "Hidden dim must match"
    );
    assert_eq!(
        Q3VL_8B_NUM_HEADS, Q3VL_30B_NUM_HEADS,
        "Num heads must match"
    );
    assert_eq!(
        Q3VL_8B_NUM_KV_HEADS, Q3VL_30B_NUM_KV_HEADS,
        "Num KV heads must match"
    );
    assert_eq!(
        Q3VL_8B_GQA_RATIO, Q3VL_30B_GQA_RATIO,
        "GQA ratio must match"
    );
}
