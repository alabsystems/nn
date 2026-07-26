// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for GptOssConfig validation and computed properties.
//!
//! Covers:
//! - Config validation rejects zero heads
//! - Config validation rejects misaligned GQA groups
//! - attn_dim computation: num_attention_heads * head_dim
//! - kv_dim computation: num_key_value_heads * head_dim
//! - GQA repeat factor bounds: kv_repeat_factor <= num_attention_heads
//! - Preset validation: gptoss_20b().validate() succeeds
//!
//! Part of #4256 (gpt-oss-20b Chroma Context-1 support).

use super::*;

// ============================================================================
// Harness 1: Config validation rejects zero attention heads
// ============================================================================

/// Proves that validate() returns Err when num_attention_heads == 0.
///
/// Zero attention heads would cause division-by-zero in GQA group calculation
/// and produce degenerate attention layers with no Q projections.
#[kani::unwind(1)]
#[kani::proof]
fn proof_config_rejects_zero_heads() {
    let mut cfg = GptOssConfig::gptoss_20b();
    cfg.num_attention_heads = 0;
    assert!(
        cfg.validate().is_err(),
        "validate must reject zero attention heads"
    );
}

// ============================================================================
// Harness 2: Config validation rejects misaligned GQA
// ============================================================================

/// Proves that validate() returns Err when num_attention_heads is not
/// divisible by num_key_value_heads.
///
/// GQA requires heads % kv_heads == 0 so each KV head serves an equal
/// number of Q heads. 7 % 3 != 0 is an invalid grouping.
#[kani::unwind(1)]
#[kani::proof]
fn proof_config_rejects_misaligned_gqa() {
    let mut cfg = GptOssConfig::gptoss_20b();
    // 64 attention heads, set KV heads to 5: 64 % 5 != 0
    cfg.num_key_value_heads = 5;
    assert!(
        cfg.validate().is_err(),
        "validate must reject non-divisible GQA heads (64 % 5 != 0)"
    );
}

// ============================================================================
// Harness 3: attn_dim computation
// ============================================================================

/// Proves that attn_dim() == num_attention_heads * head_dim for any valid
/// config values within bounded ranges.
///
/// For gpt-oss-20b: attn_dim = 64 * 64 = 4096 > hidden_size = 2880.
/// The O projection maps attn_dim back to hidden_size.
#[kani::unwind(1)]
#[kani::proof]
fn proof_config_attn_dim_computation() {
    let num_heads: usize = kani::any();
    let head_dim: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 128);
    kani::assume(head_dim >= 1 && head_dim <= 256);

    let mut cfg = GptOssConfig::gptoss_20b();
    cfg.num_attention_heads = num_heads;
    cfg.head_dim = head_dim;

    assert_eq!(
        cfg.attn_dim(),
        num_heads * head_dim,
        "attn_dim must equal num_attention_heads * head_dim"
    );
}

// ============================================================================
// Harness 4: kv_dim computation
// ============================================================================

/// Proves that kv_dim() == num_key_value_heads * head_dim for any valid
/// config values within bounded ranges.
///
/// For gpt-oss-20b: kv_dim = 8 * 64 = 512.
#[kani::unwind(1)]
#[kani::proof]
fn proof_config_kv_dim_computation() {
    let num_kv_heads: usize = kani::any();
    let head_dim: usize = kani::any();
    kani::assume(num_kv_heads >= 1 && num_kv_heads <= 128);
    kani::assume(head_dim >= 1 && head_dim <= 256);

    let mut cfg = GptOssConfig::gptoss_20b();
    cfg.num_key_value_heads = num_kv_heads;
    cfg.head_dim = head_dim;

    assert_eq!(
        cfg.kv_dim(),
        num_kv_heads * head_dim,
        "kv_dim must equal num_key_value_heads * head_dim"
    );
}

// ============================================================================
// Harness 5: GQA repeat factor bounds
// ============================================================================

/// Proves that kv_repeat_factor() <= num_attention_heads for any valid config
/// where num_key_value_heads > 0 and heads are divisible.
///
/// The repeat factor is heads / kv_heads. Since kv_heads >= 1, the factor
/// is at most heads. For gpt-oss-20b: 64 / 8 = 8 <= 64.
#[kani::unwind(1)]
#[kani::proof]
fn proof_config_gqa_repeat_bounds() {
    let num_heads: usize = kani::any();
    let num_kv_heads: usize = kani::any();
    kani::assume(num_kv_heads >= 1 && num_kv_heads <= 64);
    kani::assume(num_heads >= num_kv_heads && num_heads <= 128);
    kani::assume(num_heads % num_kv_heads == 0);

    let mut cfg = GptOssConfig::gptoss_20b();
    cfg.num_attention_heads = num_heads;
    cfg.num_key_value_heads = num_kv_heads;

    let factor = cfg.kv_repeat_factor().unwrap();
    assert!(
        factor <= num_heads,
        "kv_repeat_factor must be <= num_attention_heads"
    );
    assert!(factor >= 1, "kv_repeat_factor must be >= 1");
    assert_eq!(
        factor * num_kv_heads,
        num_heads,
        "factor * kv_heads must equal num_heads"
    );
}

// ============================================================================
// Harness 6: Preset validation
// ============================================================================

/// Proves that gptoss_20b().validate() succeeds.
///
/// The preset configuration must satisfy all invariants: non-zero fields,
/// divisible GQA groups, valid MoE config, matching layer_types length.
#[kani::unwind(25)]
#[kani::proof]
fn proof_config_preset_valid() {
    let cfg = GptOssConfig::gptoss_20b();
    assert!(
        cfg.validate().is_ok(),
        "gptoss_20b preset must pass validation"
    );

    // Verify key structural properties of the preset
    assert_eq!(cfg.hidden_size, 2880);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.num_attention_heads, 64);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.head_dim, 64);
    assert_eq!(cfg.num_local_experts, 32);
    assert_eq!(cfg.experts_per_token, 4);
    assert_eq!(cfg.layer_types.len(), 24);
    assert_eq!(cfg.attn_dim(), 4096);
    assert_eq!(cfg.kv_dim(), 512);
}
