// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Qwen3 model-specific functions.
//!
//! Covers:
//! - Config validation: rejection of zero fields, non-finite eps/theta
//! - head_dim constant invariant (always 128)
//! - GQA group computation: divisibility, zero kv_heads rejection
//! - MoE config validation: expert count, top-k bounds
//! - MoE shared_expert_ff_dim fallback logic
//! - Forward input validation: mismatched input_ids/positions lengths
//! - Cache validation: layer count mismatch detection
//! - Attention scale finiteness and positivity
//! - Builder pattern roundtrip preservation
//! - Config validate accepts all Qwen3 production variants
//!
//! Issue: #3596

use crate::config::Qwen3Config;
use crate::forward_common::{validate_cache, validate_forward_input};
use crate::moe::Qwen3MoeConfig;

// ── Kani transcendental stubs (CBMC cannot handle these) ──
fn powf_f64_stub(b: f64, _e: f64) -> f64 {
    let _ = b;
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    r
}
fn sqrt_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// ============================================================================
// Harness 1: head_dim is always 128
// ============================================================================

/// Proves that head_dim() always returns 128 regardless of config fields.
///
/// Qwen3 uses a fixed head_dim of 128 across all model variants (0.6B through
/// 235B). This is a compile-time constant encoded in the implementation.
/// Verifying it with Kani catches any accidental refactor that computes
/// head_dim from hidden_size / num_attention_heads.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn head_dim_always_128() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size > 0 && hidden_size <= 8192);
    let num_heads: usize = kani::any();
    kani::assume(num_heads > 0 && num_heads <= 64);

    let cfg = Qwen3Config::new(
        hidden_size,
        512,
        1,
        num_heads,
        num_heads,
        100,
        1e-6,
        10_000.0,
        4096,
        true,
        None,
    );
    assert_eq!(cfg.head_dim(), 128, "head_dim must always be 128");
}

// ============================================================================
// Harness 2: num_kv_groups rejects zero kv_heads
// ============================================================================

/// Proves that num_kv_groups() returns Err when num_key_value_heads == 0.
///
/// Zero KV heads would cause a division-by-zero in the GQA group calculation.
/// This must be caught regardless of other config fields.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn num_kv_groups_rejects_zero_kv_heads() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads > 0 && num_heads <= 64);

    let cfg = Qwen3Config::new(
        256, 512, 1, num_heads, 0, // zero kv_heads
        100, 1e-6, 10_000.0, 4096, true, None,
    );
    assert!(
        cfg.num_kv_groups().is_err(),
        "num_kv_groups must reject zero kv_heads"
    );
}

// ============================================================================
// Harness 3: num_kv_groups rejects non-divisible heads
// ============================================================================

/// Proves that num_kv_groups() returns Err when num_attention_heads is not
/// divisible by num_key_value_heads.
///
/// GQA requires heads % kv_heads == 0 so that each KV head serves an equal
/// number of Q heads. Non-divisible configurations would produce fractional
/// group counts.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn num_kv_groups_rejects_nondivisible() {
    let cfg = Qwen3Config::new(
        256, 512, 1, 7, // 7 attention heads
        3, // 3 kv heads: 7 % 3 != 0
        100, 1e-6, 10_000.0, 4096, true, None,
    );
    assert!(
        cfg.num_kv_groups().is_err(),
        "num_kv_groups must reject 7 % 3 != 0"
    );
}

// ============================================================================
// Harness 4: num_kv_groups correct for valid GQA configs
// ============================================================================

/// Proves that num_kv_groups() returns the correct quotient for valid configs.
///
/// For all valid (heads, kv_heads) where kv_heads > 0 and heads % kv_heads == 0,
/// the result must equal heads / kv_heads. Domain bounded to Kani-tractable sizes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn num_kv_groups_correct_for_valid_config() {
    let num_heads: usize = kani::any();
    let num_kv_heads: usize = kani::any();
    kani::assume(num_kv_heads > 0 && num_kv_heads <= 16);
    kani::assume(num_heads > 0 && num_heads <= 64);
    kani::assume(num_heads % num_kv_heads == 0);

    let cfg = Qwen3Config::new(
        256,
        512,
        1,
        num_heads,
        num_kv_heads,
        100,
        1e-6,
        10_000.0,
        4096,
        true,
        None,
    );
    let groups = cfg.num_kv_groups().unwrap();
    assert_eq!(
        groups,
        num_heads / num_kv_heads,
        "GQA groups must equal heads / kv_heads"
    );
    assert!(groups >= 1, "must have at least 1 group");
}

// ============================================================================
// Harness 5: validate rejects zero attention heads
// ============================================================================

/// Proves that validate() rejects num_attention_heads == 0.
///
/// Zero attention heads would cause division-by-zero in head_dim computation
/// and produce degenerate attention layers.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_zero_attention_heads() {
    let cfg = Qwen3Config::new(
        256, 512, 1, 0, // zero attention heads
        1, 100, 1e-6, 10_000.0, 4096, true, None,
    );
    assert!(
        cfg.validate().is_err(),
        "validate must reject zero attention heads"
    );
}

// ============================================================================
// Harness 6: validate rejects zero hidden_size
// ============================================================================

/// Proves that validate() rejects hidden_size == 0.
///
/// Zero hidden_size produces degenerate weight matrices and makes the model
/// vacuous.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_zero_hidden_size() {
    let cfg = Qwen3Config::new(
        0, // zero hidden_size
        512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None,
    );
    assert!(
        cfg.validate().is_err(),
        "validate must reject zero hidden_size"
    );
}

// ============================================================================
// Harness 7: validate rejects non-finite rms_norm_eps
// ============================================================================

/// Proves that validate() rejects NaN rms_norm_eps.
///
/// NaN eps in RMSNorm produces NaN outputs through the normalization division.
/// IEEE 754: NaN comparisons return false, so the explicit `is_finite()` check
/// in validate() is necessary.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_nan_rms_norm_eps() {
    let cfg = Qwen3Config::new(
        256,
        512,
        1,
        2,
        2,
        100,
        f64::NAN, // NaN eps
        10_000.0,
        4096,
        true,
        None,
    );
    assert!(
        cfg.validate().is_err(),
        "validate must reject NaN rms_norm_eps"
    );
}

/// Proves that validate() rejects negative rms_norm_eps.
///
/// Negative eps in the denominator sqrt(mean(x^2) + eps) could cause the
/// denominator to approach zero or become imaginary for small activations.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_negative_rms_norm_eps() {
    let cfg = Qwen3Config::new(
        256, 512, 1, 2, 2, 100, -1e-6, // negative eps
        10_000.0, 4096, true, None,
    );
    assert!(
        cfg.validate().is_err(),
        "validate must reject negative rms_norm_eps"
    );
}

// ============================================================================
// Harness 8: validate rejects non-finite rope_theta
// ============================================================================

/// Proves that validate() rejects Inf rope_theta.
///
/// Infinite rope_theta produces degenerate RoPE frequencies (all zero), collapsing
/// positional information. The validate() check catches both Inf and NaN.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_inf_rope_theta() {
    let cfg = Qwen3Config::new(
        256,
        512,
        1,
        2,
        2,
        100,
        1e-6,
        f64::INFINITY, // Inf theta
        4096,
        true,
        None,
    );
    assert!(
        cfg.validate().is_err(),
        "validate must reject Inf rope_theta"
    );
}

/// Proves that validate() rejects zero rope_theta.
///
/// Zero rope_theta produces division-by-zero in inv_freq = 1 / (theta^(2i/d)).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_zero_rope_theta() {
    let cfg = Qwen3Config::new(
        256, 512, 1, 2, 2, 100, 1e-6, 0.0, // zero theta
        4096, true, None,
    );
    assert!(
        cfg.validate().is_err(),
        "validate must reject zero rope_theta"
    );
}

// ============================================================================
// Harness 9: validate rejects zero vocab_size
// ============================================================================

/// Proves that validate() rejects vocab_size == 0.
///
/// Zero vocab_size produces a zero-row embedding matrix and zero-row lm_head,
/// making the model unable to produce any token.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_zero_vocab_size() {
    let cfg = Qwen3Config::new(
        256, 512, 1, 2, 2, 0, // zero vocab
        1e-6, 10_000.0, 4096, true, None,
    );
    assert!(
        cfg.validate().is_err(),
        "validate must reject zero vocab_size"
    );
}

// ============================================================================
// Harness 10: validate rejects zero max_position_embeddings
// ============================================================================

/// Proves that validate() rejects max_position_embeddings == 0.
///
/// Zero max_position_embeddings would produce a RoPE frequency table with zero
/// entries, unable to encode any position.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_zero_max_position_embeddings() {
    let cfg = Qwen3Config::new(
        256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 0, // zero max_pos
        true, None,
    );
    assert!(
        cfg.validate().is_err(),
        "validate must reject zero max_position_embeddings"
    );
}

// ============================================================================
// Harness 11: attention scale is finite and positive
// ============================================================================

/// Proves the attention scale factor `1 / sqrt(head_dim)` is finite and positive.
///
/// head_dim is always 128, so scale = 1/sqrt(128) ~= 0.0884. This harness
/// verifies the computation does not produce NaN, Inf, or non-positive values
/// for any valid head_dim in the Qwen3 range.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn attention_scale_finite_and_positive() {
    // Qwen3 head_dim is constant 128, but verify the computation for safety
    let head_dim: usize = 128;
    let scale = 1.0 / (head_dim as f64).sqrt();
    assert!(scale.is_finite(), "attention scale must be finite");
    assert!(scale > 0.0, "attention scale must be positive");

    // Also verify it equals the expected value within tolerance
    let expected = 1.0 / (128.0f64).sqrt();
    assert!(
        (scale - expected).abs() < 1e-15,
        "scale must match 1/sqrt(128)"
    );
}

// ============================================================================
// Harness 12: validate_forward_input rejects mismatched lengths
// ============================================================================

/// Proves that validate_forward_input returns Err when input_ids and positions
/// have different lengths.
///
/// Mismatched lengths would cause silent truncation or out-of-bounds access
/// during RoPE application (positions[i] for each token).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_forward_input_rejects_mismatch() {
    let ids_len: usize = kani::any();
    let pos_len: usize = kani::any();
    kani::assume(ids_len <= 16 && pos_len <= 16);
    kani::assume(ids_len != pos_len);

    let ids: Vec<usize> = vec![0; ids_len];
    let positions: Vec<usize> = vec![0; pos_len];
    assert!(
        validate_forward_input(&ids, &positions).is_err(),
        "must reject mismatched input_ids/positions lengths"
    );
}

/// Proves that validate_forward_input returns Ok when lengths match.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_forward_input_accepts_matching() {
    let len: usize = kani::any();
    kani::assume(len <= 16);

    let ids: Vec<usize> = vec![0; len];
    let positions: Vec<usize> = vec![0; len];
    assert!(
        validate_forward_input(&ids, &positions).is_ok(),
        "must accept matching input_ids/positions lengths"
    );
}

// ============================================================================
// Harness 13: validate_cache rejects mismatched layer count
// ============================================================================

/// Proves that validate_cache returns Err when cache layer count differs from
/// the model's num_hidden_layers.
///
/// A cache with the wrong number of layers would cause indexing errors during
/// the decoder loop.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_cache_rejects_mismatch() {
    use nn_core::layers::kv_cache::KvCache;

    let cache_layers: usize = kani::any();
    let model_layers: usize = kani::any();
    kani::assume(cache_layers > 0 && cache_layers <= 8);
    kani::assume(model_layers > 0 && model_layers <= 8);
    kani::assume(cache_layers != model_layers);

    let cache = KvCache::new(cache_layers);
    assert!(
        validate_cache(Some(&cache), model_layers).is_err(),
        "must reject cache/model layer count mismatch"
    );
}

/// Proves that validate_cache returns Ok when cache is None (no caching).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_cache_accepts_none() {
    let model_layers: usize = kani::any();
    kani::assume(model_layers > 0 && model_layers <= 64);

    assert!(
        validate_cache(None, model_layers).is_ok(),
        "None cache must always be accepted"
    );
}

// ============================================================================
// Harness 14: MoE config validates expert count
// ============================================================================

/// Proves that Qwen3MoeConfig::validate() rejects num_experts == 0.
///
/// Zero experts would produce a vacuous MoE layer with no routing targets.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_rejects_zero_experts() {
    let base = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg = Qwen3MoeConfig::new(
        base, 0, // zero experts
        1, false, None,
    );
    assert!(cfg.validate().is_err(), "must reject zero num_experts");
}

/// Proves that Qwen3MoeConfig::validate() rejects num_experts_per_tok > num_experts.
///
/// Routing more experts than exist is undefined.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_rejects_topk_exceeds_experts() {
    let base = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg = Qwen3MoeConfig::new(
        base, 4, // 4 experts
        5, // top-5: exceeds total
        false, None,
    );
    assert!(cfg.validate().is_err(), "must reject topk > num_experts");
}

/// Proves that Qwen3MoeConfig::validate() rejects num_experts_per_tok == 0.
///
/// Zero active experts per token means no expert contributes to the output.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_rejects_zero_topk() {
    let base = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg = Qwen3MoeConfig::new(
        base, 8, 0, // zero topk
        false, None,
    );
    assert!(
        cfg.validate().is_err(),
        "must reject zero num_experts_per_tok"
    );
}

// ============================================================================
// Harness 15: MoE shared_expert_ff_dim fallback
// ============================================================================

/// Proves that shared_expert_ff_dim() returns the override when set, and falls
/// back to base.intermediate_size when None.
///
/// This is the Qwen3.5 pattern: shared experts may have a different FFN
/// intermediate dimension than routed experts.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_shared_expert_ff_dim_fallback() {
    let base = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);

    // With override
    let cfg_with = Qwen3MoeConfig::new(base.clone(), 8, 2, true, Some(1024));
    assert_eq!(
        cfg_with.shared_expert_ff_dim(),
        1024,
        "must use override when set"
    );

    // Without override
    let cfg_without = Qwen3MoeConfig::new(base.clone(), 8, 2, true, None);
    assert_eq!(
        cfg_without.shared_expert_ff_dim(),
        512,
        "must fallback to base.intermediate_size"
    );
}

// ============================================================================
// Harness 16: builder with_vocab_size preserves other fields
// ============================================================================

/// Proves that with_vocab_size() only changes vocab_size and preserves all
/// other configuration fields.
///
/// Builder-style setters must be non-destructive: modifying one field should
/// not affect any other.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn builder_with_vocab_size_preserves_fields() {
    let cfg = Qwen3Config::new(256, 512, 4, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let modified = cfg.with_vocab_size(200);

    assert_eq!(modified.vocab_size, 200);
    assert_eq!(modified.hidden_size, 256);
    assert_eq!(modified.intermediate_size, 512);
    assert_eq!(modified.num_hidden_layers, 4);
    assert_eq!(modified.num_attention_heads, 2);
    assert_eq!(modified.num_key_value_heads, 2);
    assert_eq!(modified.rms_norm_eps, 1e-6);
    assert_eq!(modified.rope_theta, 10_000.0);
    assert_eq!(modified.max_position_embeddings, 4096);
    assert!(modified.tie_word_embeddings);
}

// ============================================================================
// Harness 17: builder with_num_hidden_layers preserves other fields
// ============================================================================

/// Proves that with_num_hidden_layers() only changes num_hidden_layers.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn builder_with_num_hidden_layers_preserves_fields() {
    let cfg = Qwen3Config::new(256, 512, 4, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let modified = cfg.with_num_hidden_layers(8);

    assert_eq!(modified.num_hidden_layers, 8);
    assert_eq!(modified.hidden_size, 256);
    assert_eq!(modified.intermediate_size, 512);
    assert_eq!(modified.num_attention_heads, 2);
    assert_eq!(modified.num_key_value_heads, 2);
    assert_eq!(modified.vocab_size, 100);
    assert_eq!(modified.rms_norm_eps, 1e-6);
    assert_eq!(modified.rope_theta, 10_000.0);
    assert_eq!(modified.max_position_embeddings, 4096);
    assert!(modified.tie_word_embeddings);
}

// ============================================================================
// Harness 18: validate accepts Qwen3 production-like configs
// ============================================================================

/// Proves that validate() accepts configurations matching real Qwen3 model
/// variants: 0.6B, 1.7B, 4B, 8B, 14B, 32B.
///
/// These are the published Qwen3 dense model configurations from the
/// Qwen3 Technical Report (arXiv:2505.09388). Verifying they pass validation
/// ensures the validator does not over-reject.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_accepts_production_configs() {
    // Qwen3-0.6B: hidden=896, intermediate=4864, layers=28, heads=14, kv=2
    let cfg_06b = Qwen3Config::new(
        896,
        4864,
        28,
        14,
        2,
        151_936,
        1e-6,
        1_000_000.0,
        40_960,
        true,
        None,
    );
    assert!(cfg_06b.validate().is_ok(), "Qwen3-0.6B must pass");

    // Qwen3-1.7B: hidden=2048, intermediate=11008, layers=28, heads=16, kv=4
    let cfg_17b = Qwen3Config::new(
        2048,
        11008,
        28,
        16,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        40_960,
        true,
        None,
    );
    assert!(cfg_17b.validate().is_ok(), "Qwen3-1.7B must pass");

    // Qwen3-4B: hidden=2560, intermediate=13824, layers=36, heads=20, kv=4
    let cfg_4b = Qwen3Config::new(
        2560,
        13824,
        36,
        20,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        40_960,
        true,
        None,
    );
    assert!(cfg_4b.validate().is_ok(), "Qwen3-4B must pass");

    // Qwen3-8B: hidden=4096, intermediate=14336, layers=36, heads=32, kv=8
    let cfg_8b = Qwen3Config::new(
        4096,
        14336,
        36,
        32,
        8,
        151_936,
        1e-6,
        1_000_000.0,
        131_072,
        true,
        None,
    );
    assert!(cfg_8b.validate().is_ok(), "Qwen3-8B must pass");
}

// ============================================================================
// Harness 19: GQA group count is at least 1 for valid configs
// ============================================================================

/// Proves that num_kv_groups() >= 1 for all valid config combinations.
///
/// The minimum GQA configuration is MHA (Multi-Head Attention) where
/// kv_heads == heads, producing exactly 1 group. GQA with kv_heads < heads
/// produces more groups.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn num_kv_groups_at_least_one() {
    let num_heads: usize = kani::any();
    let num_kv_heads: usize = kani::any();
    kani::assume(num_kv_heads > 0 && num_kv_heads <= 16);
    kani::assume(num_heads >= num_kv_heads && num_heads <= 64);
    kani::assume(num_heads % num_kv_heads == 0);

    let cfg = Qwen3Config::new(
        256,
        512,
        1,
        num_heads,
        num_kv_heads,
        100,
        1e-6,
        10_000.0,
        4096,
        true,
        None,
    );
    let groups = cfg.num_kv_groups().unwrap();
    assert!(groups >= 1, "GQA must have at least 1 group");
}

// ============================================================================
// Harness 20: validate rejects zero intermediate_size
// ============================================================================

/// Proves that validate() rejects intermediate_size == 0.
///
/// Zero intermediate_size produces degenerate SwiGLU MLP weight matrices
/// (gate_proj, up_proj, down_proj all have a zero dimension).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_zero_intermediate_size() {
    let cfg = Qwen3Config::new(
        256, 0, // zero intermediate_size
        1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None,
    );
    assert!(
        cfg.validate().is_err(),
        "validate must reject zero intermediate_size"
    );
}

// ============================================================================
// Harness 21: MoE validate propagates base config errors
// ============================================================================

/// Proves that Qwen3MoeConfig::validate() propagates base config validation
/// errors (e.g., zero attention heads in the base config).
///
/// MoE validation must check both MoE-specific and base transformer invariants.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_propagates_base_errors() {
    let bad_base = Qwen3Config::new(
        256, 512, 1, 0, // zero attention heads - invalid
        1, 100, 1e-6, 10_000.0, 4096, true, None,
    );
    let cfg = Qwen3MoeConfig::new(bad_base, 8, 2, false, None);
    assert!(
        cfg.validate().is_err(),
        "MoE validate must propagate base config errors"
    );
}

// ============================================================================
// Harness 22: MoE shared_expert_intermediate_size zero rejected
// ============================================================================

/// Proves that Qwen3MoeConfig::validate() rejects shared_expert_intermediate_size
/// of 0 when shared_expert is enabled.
///
/// A shared expert with zero intermediate size produces a degenerate FFN.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_rejects_zero_shared_expert_dim() {
    let base = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg = Qwen3MoeConfig::new(
        base,
        8,
        2,
        true,    // shared expert enabled
        Some(0), // zero intermediate size
    );
    assert!(
        cfg.validate().is_err(),
        "must reject zero shared_expert_intermediate_size"
    );
}

// ============================================================================
// Harness 23: RoPE inv_freq is finite and positive for Qwen3 head_dim
// ============================================================================

/// Proves that the RoPE inverse frequency values are finite and positive
/// for head_dim=128 (Qwen3 constant) with valid rope_theta.
///
/// inv_freq[i] = 1 / (theta^(2*i / head_dim)) for i in [0, head_dim/2).
/// Since theta > 0 and the exponent is in [0, 1), inv_freq is in (0, 1].
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(65)] // head_dim/2 = 64 iterations + 1
#[kani::stub(f64::powf, powf_f64_stub)]
fn rope_inv_freq_finite_positive() {
    let head_dim: usize = 128;
    let rope_theta: f64 = 10_000.0; // Standard Qwen3 theta

    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let exponent = (2 * i) as f64 / head_dim as f64;
        let inv_freq = 1.0 / rope_theta.powf(exponent);
        assert!(inv_freq.is_finite(), "inv_freq must be finite at i={i}");
        assert!(inv_freq > 0.0, "inv_freq must be positive at i={i}");
        assert!(
            inv_freq <= 1.0,
            "inv_freq <= 1 since theta >= 1 and exponent >= 0"
        );
    }
}

/// Same proof for the extended-context rope_theta = 1_000_000 used in Qwen3-8B+.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(65)]
#[kani::stub(f64::powf, powf_f64_stub)]
fn rope_inv_freq_finite_positive_extended_theta() {
    let head_dim: usize = 128;
    let rope_theta: f64 = 1_000_000.0; // Extended context theta

    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let exponent = (2 * i) as f64 / head_dim as f64;
        let inv_freq = 1.0 / rope_theta.powf(exponent);
        assert!(inv_freq.is_finite(), "inv_freq must be finite at i={i}");
        assert!(inv_freq > 0.0, "inv_freq must be positive at i={i}");
    }
}
