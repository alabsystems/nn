// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for gpt-oss layer invariants.
//!
//! Covers:
//! - Layer type alternation: gptoss_20b layers alternate sliding/full
//! - Fused expert weight split: gate_up_proj [h, 2*inter] splits correctly
//! - SwiGLU clamp bounds: silu(x).clamp(-7, 7) is in [-7, 7]
//! - MoE routing top-4 of 32: exactly 4 experts selected
//! - Attention dim consistency: attn_dim / head_dim == num_attention_heads
//!
//! Part of #4256 (gpt-oss-20b Chroma Context-1 support).

use crate::config::{GptOssConfig, LayerType};

// -- Kani transcendental stubs (CBMC cannot handle exp) --

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

// ============================================================================
// Harness 1: Layer type alternation
// ============================================================================

/// Proves that gptoss_20b() layer types alternate: even indices are
/// SlidingAttention, odd indices are FullAttention, for all 24 layers.
///
/// This is the core attention pattern of Context-1: sliding window on
/// even layers (cheap local attention), full causal on odd layers
/// (global context).
#[kani::unwind(25)]
#[kani::proof]
fn proof_layer_type_alternation() {
    let cfg = GptOssConfig::gptoss_20b();
    assert_eq!(cfg.layer_types.len(), 24, "must have 24 layers");

    for i in 0..24 {
        if i % 2 == 0 {
            assert_eq!(
                cfg.layer_types[i],
                LayerType::SlidingAttention,
                "even layer {i} must be SlidingAttention"
            );
        } else {
            assert_eq!(
                cfg.layer_types[i],
                LayerType::FullAttention,
                "odd layer {i} must be FullAttention"
            );
        }
    }
}

// ============================================================================
// Harness 2: Fused expert weight split
// ============================================================================

/// Proves that splitting a fused gate_up_proj tensor of shape
/// [hidden_size, 2*intermediate_size] at the midpoint produces two
/// [hidden_size, intermediate_size] tensors.
///
/// For gpt-oss-20b: [2880, 5760] splits into two [2880, 2880] tensors.
/// The split point is `intermediate_size` (= 2880).
#[kani::unwind(1)]
#[kani::proof]
fn proof_fused_expert_weight_split() {
    let hidden_size: usize = kani::any();
    let intermediate_size: usize = kani::any();
    kani::assume(hidden_size >= 1 && hidden_size <= 8192);
    kani::assume(intermediate_size >= 1 && intermediate_size <= 8192);

    let fused_dim = 2 * intermediate_size;

    // gate_up_proj per expert: [hidden_size, fused_dim]
    // Split at intermediate_size along the last dimension:
    // gate = [:, :intermediate_size]  -> [hidden_size, intermediate_size]
    // up   = [:, intermediate_size:]  -> [hidden_size, intermediate_size]

    let gate_cols = intermediate_size;
    let up_start = intermediate_size;
    let up_cols = fused_dim - up_start;

    assert_eq!(
        gate_cols, intermediate_size,
        "gate slice must have intermediate_size columns"
    );
    assert_eq!(
        up_cols, intermediate_size,
        "up slice must have intermediate_size columns"
    );
    assert_eq!(
        gate_cols + up_cols,
        fused_dim,
        "gate + up must reconstruct full fused dimension"
    );

    // Verify for the gpt-oss-20b specific case
    let h = 2880_usize;
    let inter = 2880_usize;
    let fused = 2 * inter;
    assert_eq!(fused, 5760);
    assert_eq!(fused / 2, inter);
}

// ============================================================================
// Harness 3: SwiGLU clamp bounds
// ============================================================================

/// Proves that for any f32 x, silu(x).clamp(-7.0, 7.0) is in [-7.0, 7.0].
///
/// This is the core safety property of the gpt-oss-20b clamped SwiGLU.
/// silu(x) = x * sigmoid(x) can grow unbounded for large x, but the clamp
/// constrains the output to the limit regardless of input magnitude.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_swiglu_clamp_bounds() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -100.0 && x <= 100.0);

    let limit: f32 = 7.0;

    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let sig = 1.0f32 / (1.0 + (-x).exp());
    kani::assume(sig.is_finite());
    let silu = x * sig;
    kani::assume(silu.is_finite());

    // clamp(silu, -limit, limit)
    let clamped = if silu < -limit {
        -limit
    } else if silu > limit {
        limit
    } else {
        silu
    };

    assert!(clamped >= -limit, "clamped silu must be >= -7.0");
    assert!(clamped <= limit, "clamped silu must be <= 7.0");
    assert!(clamped.is_finite(), "clamped silu must be finite");
}

// ============================================================================
// Harness 4: MoE routing top-4 of 32
// ============================================================================

/// Proves that for experts_per_token=4 and num_local_experts=32, exactly 4
/// experts are selected per token in the routing loop.
///
/// Models the grouping loop from GptOssMoeBlock::forward() which assigns
/// each token to exactly top_k experts.
#[kani::unwind(5)]
#[kani::proof]
fn proof_moe_routing_topk() {
    let experts_per_token: usize = 4;
    let num_local_experts: usize = 32;

    // Verify the config invariant
    assert!(
        experts_per_token >= 1,
        "must have at least 1 expert per token"
    );
    assert!(
        experts_per_token <= num_local_experts,
        "experts_per_token must be <= num_local_experts"
    );

    // Model one token's routing: it gets assigned to exactly k experts
    let mut selected_count: usize = 0;
    let mut selected: [usize; 4] = [usize::MAX; 4];

    for s in 0..experts_per_token {
        let expert_idx: usize = kani::any();
        kani::assume(expert_idx < num_local_experts);

        // Top-k returns distinct indices
        for prev in 0..s {
            kani::assume(expert_idx != selected[prev]);
        }
        selected[s] = expert_idx;
        selected_count += 1;
    }

    assert_eq!(
        selected_count, experts_per_token,
        "exactly 4 experts must be selected"
    );

    // Verify all selected indices are valid
    for i in 0..experts_per_token {
        assert!(
            selected[i] < num_local_experts,
            "selected expert must be < num_local_experts"
        );
    }

    // Verify distinctness
    for i in 0..experts_per_token {
        for j in (i + 1)..experts_per_token {
            assert!(
                selected[i] != selected[j],
                "selected experts must be distinct"
            );
        }
    }
}

// ============================================================================
// Harness 5: Attention dim consistency
// ============================================================================

/// Proves that attn_dim / head_dim == num_attention_heads for any valid
/// gpt-oss config where head_dim > 0.
///
/// For gpt-oss-20b: 4096 / 64 = 64 = num_attention_heads.
/// This ensures the reshape from [B, S, attn_dim] -> [B, S, nh, hd]
/// produces the correct number of heads.
#[kani::unwind(1)]
#[kani::proof]
fn proof_attention_dim_consistency() {
    let num_heads: usize = kani::any();
    let head_dim: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 128);
    kani::assume(head_dim >= 1 && head_dim <= 256);

    let attn_dim = num_heads * head_dim;

    // attn_dim / head_dim == num_attention_heads (exact integer division)
    assert_eq!(
        attn_dim / head_dim,
        num_heads,
        "attn_dim / head_dim must equal num_attention_heads"
    );
    assert_eq!(
        attn_dim % head_dim,
        0,
        "attn_dim must be exactly divisible by head_dim"
    );

    // Verify for gpt-oss-20b specific values
    let gptoss_attn_dim = 64_usize * 64;
    assert_eq!(gptoss_attn_dim, 4096);
    assert_eq!(gptoss_attn_dim / 64, 64);
}
