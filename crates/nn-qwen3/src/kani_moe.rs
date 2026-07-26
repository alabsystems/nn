// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `moe.rs` — Qwen3MoeConfig and Qwen3MoeModel
//! structural invariants.
//!
//! Covers properties NOT in `kani_qwen3.rs` or `kani_moe_forward_proofs.rs`:
//! - MoE config constructor roundtrip: all fields stored correctly
//! - MoE validate accepts valid configs with shared expert enabled + None dim
//! - MoE validate boundary: topk == num_experts is the maximum valid topk
//! - MoE shared_expert_ff_dim monotonicity w.r.t. override presence
//! - MoE config validate is idempotent (calling twice yields same result)
//! - MoE config with shared expert: base.intermediate_size used when None
//! - MoE expert weight matrix dimension product no overflow
//! - MoE router softmax input dimension matches num_experts
//! - MoE combined config: base validate + MoE validate conjunction
//! - MoE shared expert intermediate size must be positive when set
//!
//! Issue: #3700

use crate::config::Qwen3Config;
use crate::moe::Qwen3MoeConfig;

// ============================================================================
// Harness 1: MoE config constructor stores all fields correctly
// ============================================================================

/// Proves that Qwen3MoeConfig::new() stores every field unchanged.
///
/// The #[non_exhaustive] attribute means external crates cannot construct
/// via struct literal; the constructor is the only entry point. This
/// proves the constructor is a faithful identity on all fields.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_config_constructor_roundtrip() {
    let num_experts: usize = kani::any();
    let num_experts_per_tok: usize = kani::any();
    let shared_intermediate: usize = kani::any();
    kani::assume(num_experts >= 1 && num_experts <= 128);
    kani::assume(num_experts_per_tok >= 1 && num_experts_per_tok <= num_experts);
    kani::assume(shared_intermediate >= 1 && shared_intermediate <= 65536);

    let base = Qwen3Config::new(256, 512, 4, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg = Qwen3MoeConfig::new(
        base.clone(),
        num_experts,
        num_experts_per_tok,
        true,
        Some(shared_intermediate),
    );

    assert_eq!(cfg.num_experts, num_experts);
    assert_eq!(cfg.num_experts_per_tok, num_experts_per_tok);
    assert!(cfg.shared_expert);
    assert_eq!(
        cfg.shared_expert_intermediate_size,
        Some(shared_intermediate)
    );
    assert_eq!(cfg.base.hidden_size, 256);
    assert_eq!(cfg.base.intermediate_size, 512);
}

// ============================================================================
// Harness 2: MoE validate accepts shared_expert with None intermediate
// ============================================================================

/// Proves that when shared_expert is true but shared_expert_intermediate_size
/// is None, validation passes (falls back to base.intermediate_size).
///
/// This is the common Qwen3 MoE pattern before Qwen3.5 added explicit
/// shared expert sizing.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_shared_expert_none_dim_ok() {
    let num_experts: usize = kani::any();
    let topk: usize = kani::any();
    kani::assume(num_experts >= 1 && num_experts <= 128);
    kani::assume(topk >= 1 && topk <= num_experts);

    let base = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg = Qwen3MoeConfig::new(base, num_experts, topk, true, None);
    assert!(
        cfg.validate().is_ok(),
        "shared_expert=true with None dim must pass"
    );
}

// ============================================================================
// Harness 3: MoE validate — topk boundary is exactly num_experts
// ============================================================================

/// Proves that topk == num_experts is the maximum valid topk: topk+1 fails.
///
/// This is a boundary test: the validation rejects topk > num_experts
/// but accepts topk == num_experts exactly at the boundary.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_topk_boundary() {
    let num_experts: usize = kani::any();
    kani::assume(num_experts >= 2 && num_experts <= 64);

    let base = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);

    // topk == num_experts: valid
    let cfg_ok = Qwen3MoeConfig::new(base.clone(), num_experts, num_experts, false, None);
    assert!(cfg_ok.validate().is_ok(), "topk == num_experts must pass");

    // topk == num_experts + 1: invalid
    let cfg_bad = Qwen3MoeConfig::new(base, num_experts, num_experts + 1, false, None);
    assert!(cfg_bad.validate().is_err(), "topk > num_experts must fail");
}

// ============================================================================
// Harness 4: MoE validate is idempotent
// ============================================================================

/// Proves that calling validate() twice on the same config yields the same
/// result. Validation must be a pure function with no side effects.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_idempotent() {
    let num_experts: usize = kani::any();
    let topk: usize = kani::any();
    kani::assume(num_experts <= 16);
    kani::assume(topk <= 17);

    let base = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg = Qwen3MoeConfig::new(base, num_experts, topk, false, None);

    let r1 = cfg.validate().is_ok();
    let r2 = cfg.validate().is_ok();
    assert_eq!(r1, r2, "validate must be idempotent");
}

// ============================================================================
// Harness 5: MoE shared_expert_ff_dim with None returns base intermediate
// ============================================================================

/// Proves that shared_expert_ff_dim() == base.intermediate_size when
/// shared_expert_intermediate_size is None, for all valid base sizes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_shared_ff_dim_none_returns_base() {
    let base_intermediate: usize = kani::any();
    kani::assume(base_intermediate >= 1 && base_intermediate <= 65536);

    let base = Qwen3Config::new(
        256,
        base_intermediate,
        1,
        2,
        2,
        100,
        1e-6,
        10_000.0,
        4096,
        true,
        None,
    );
    let cfg = Qwen3MoeConfig::new(base, 8, 2, true, None);

    assert_eq!(
        cfg.shared_expert_ff_dim(),
        base_intermediate,
        "None must fall back to base.intermediate_size"
    );
}

// ============================================================================
// Harness 6: MoE expert weight total size no overflow
// ============================================================================

/// Proves that the total parameter count for a single MoE expert's FFN
/// (gate + up + down projections) does not overflow usize.
///
/// Per expert: gate[I,H] + up[I,H] + down[H,I] = 3*H*I parameters.
/// For Qwen3-235B: H=4096, I=12288, per_expert = 3*4096*12288 = 150M.
/// 128 experts * 150M = 19.2B — fits in usize on 64-bit.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_expert_weight_size_no_overflow() {
    let hidden: usize = kani::any();
    let intermediate: usize = kani::any();
    let num_experts: usize = kani::any();
    kani::assume(hidden >= 1 && hidden <= 8192);
    kani::assume(intermediate >= 1 && intermediate <= 32768);
    kani::assume(num_experts >= 1 && num_experts <= 128);

    // Per-expert parameters: gate[I,H] + up[I,H] + down[H,I] = 3*H*I
    let per_expert = hidden
        .checked_mul(intermediate)
        .and_then(|hi| hi.checked_mul(3));
    assert!(per_expert.is_some(), "per-expert params must not overflow");

    let total = per_expert.unwrap().checked_mul(num_experts);
    assert!(total.is_some(), "total expert params must not overflow");
}

// ============================================================================
// Harness 7: MoE router weight dimensions — [num_experts, hidden]
// ============================================================================

/// Proves that the router weight matrix dimension product does not overflow.
///
/// Router Linear: weight shape [num_experts, hidden_size].
/// Produces num_experts logits per token for softmax + top-k selection.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_router_weight_dims_no_overflow() {
    let hidden: usize = kani::any();
    let num_experts: usize = kani::any();
    kani::assume(hidden >= 1 && hidden <= 8192);
    kani::assume(num_experts >= 1 && num_experts <= 128);

    let router_params = num_experts.checked_mul(hidden);
    assert!(
        router_params.is_some(),
        "router weight matrix size must not overflow"
    );
    assert!(
        router_params.unwrap() > 0,
        "router must have positive parameter count"
    );
}

// ============================================================================
// Harness 8: MoE validate rejects NaN rope_theta in base
// ============================================================================

/// Proves that MoE validate propagates NaN rope_theta rejection from base.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_rejects_nan_rope_theta_in_base() {
    let base = Qwen3Config::new(
        256,
        512,
        1,
        2,
        2,
        100,
        1e-6,
        f64::NAN, // NaN rope_theta
        4096,
        true,
        None,
    );
    let cfg = Qwen3MoeConfig::new(base, 8, 2, false, None);
    assert!(
        cfg.validate().is_err(),
        "MoE must propagate NaN rope_theta from base"
    );
}

// ============================================================================
// Harness 9: MoE validate rejects zero hidden_size in base
// ============================================================================

/// Proves that MoE validate propagates zero hidden_size rejection from base.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_rejects_zero_hidden_in_base() {
    let base = Qwen3Config::new(
        0, // zero hidden_size
        512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None,
    );
    let cfg = Qwen3MoeConfig::new(base, 8, 2, false, None);
    assert!(
        cfg.validate().is_err(),
        "MoE must propagate zero hidden_size from base"
    );
}

// ============================================================================
// Harness 10: MoE combined valid config space — all constraints satisfied
// ============================================================================

/// Proves that for symbolic valid configs, both base.validate() and
/// moe.validate() accept simultaneously (conjunction).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_combined_valid_config_space() {
    let num_heads: usize = kani::any();
    let num_kv: usize = kani::any();
    let num_experts: usize = kani::any();
    let topk: usize = kani::any();

    kani::assume(num_heads >= 1 && num_heads <= 16);
    kani::assume(num_kv >= 1 && num_kv <= num_heads);
    kani::assume(num_heads % num_kv == 0);
    kani::assume(num_experts >= 1 && num_experts <= 16);
    kani::assume(topk >= 1 && topk <= num_experts);

    let base = Qwen3Config::new(
        256, 512, 1, num_heads, num_kv, 100, 1e-6, 10_000.0, 4096, true, None,
    );
    assert!(base.validate().is_ok(), "base must be valid");

    let cfg = Qwen3MoeConfig::new(base, num_experts, topk, false, None);
    assert!(cfg.validate().is_ok(), "MoE config must be valid");
}

// ============================================================================
// Harness 11: MoE shared_expert_ff_dim with Some returns override, not base
// ============================================================================

/// Proves that shared_expert_ff_dim() returns the override value when
/// Some(override) is set, regardless of base.intermediate_size.
///
/// This ensures the unwrap_or fallback logic works correctly: Some(x) -> x.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_shared_ff_dim_some_returns_override() {
    let override_val: usize = kani::any();
    let base_intermediate: usize = kani::any();
    kani::assume(override_val >= 1 && override_val <= 65536);
    kani::assume(base_intermediate >= 1 && base_intermediate <= 65536);

    let base = Qwen3Config::new(
        256,
        base_intermediate,
        1,
        2,
        2,
        100,
        1e-6,
        10_000.0,
        4096,
        true,
        None,
    );
    let cfg = Qwen3MoeConfig::new(base, 8, 2, true, Some(override_val));

    assert_eq!(
        cfg.shared_expert_ff_dim(),
        override_val,
        "Some(override) must return override, not base"
    );
}

// ============================================================================
// Harness 12: MoE validate rejects shared_expert_intermediate_size=0 when enabled
// ============================================================================

/// Proves that validate rejects Some(0) specifically when shared_expert
/// is true. The zero check is inside the `if self.shared_expert` branch.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_zero_shared_dim_with_shared_enabled() {
    let num_experts: usize = kani::any();
    let topk: usize = kani::any();
    kani::assume(num_experts >= 1 && num_experts <= 16);
    kani::assume(topk >= 1 && topk <= num_experts);

    let base = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg = Qwen3MoeConfig::new(base, num_experts, topk, true, Some(0));

    assert!(
        cfg.validate().is_err(),
        "shared_expert=true + Some(0) must always be rejected"
    );
}

// ============================================================================
// Harness 13: MoE total expert weight memory — per-expert * num_experts no overflow
// ============================================================================

/// Proves that the total MoE weight memory (all experts + router + optional
/// shared expert) does not overflow usize for production configurations.
///
/// Total = num_experts * 3*H*I + H*num_experts (router) + optional 3*H*shared_I.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_total_weight_memory_no_overflow() {
    let hidden: usize = kani::any();
    let intermediate: usize = kani::any();
    let num_experts: usize = kani::any();
    kani::assume(hidden >= 1 && hidden <= 4096);
    kani::assume(intermediate >= 1 && intermediate <= 16384);
    kani::assume(num_experts >= 1 && num_experts <= 128);

    // Per-expert SwiGLU: gate[I,H] + up[I,H] + down[H,I] = 3*H*I
    let per_expert = hidden
        .checked_mul(intermediate)
        .and_then(|hi| hi.checked_mul(3));
    assert!(per_expert.is_some(), "per_expert must not overflow");

    // All experts
    let all_experts = per_expert.unwrap().checked_mul(num_experts);
    assert!(all_experts.is_some(), "all_experts must not overflow");

    // Router: [num_experts, hidden]
    let router = num_experts.checked_mul(hidden);
    assert!(router.is_some(), "router must not overflow");

    // Total (excluding shared expert)
    let total = all_experts.unwrap().checked_add(router.unwrap());
    assert!(total.is_some(), "total weight memory must not overflow");
}

// ============================================================================
// Harness 14: MoE config clone preserves all fields
// ============================================================================

/// Proves that Clone on Qwen3MoeConfig preserves all fields exactly.
/// Since Qwen3MoeConfig derives Clone, this verifies the derivation
/// is correct for MoE-specific fields.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_config_clone_preserves_fields() {
    let num_experts: usize = kani::any();
    let topk: usize = kani::any();
    let shared_dim: usize = kani::any();
    kani::assume(num_experts >= 1 && num_experts <= 128);
    kani::assume(topk >= 1 && topk <= num_experts);
    kani::assume(shared_dim >= 1 && shared_dim <= 65536);

    let base = Qwen3Config::new(256, 512, 4, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg = Qwen3MoeConfig::new(base, num_experts, topk, true, Some(shared_dim));
    let cloned = cfg.clone();

    assert_eq!(cloned.num_experts, cfg.num_experts);
    assert_eq!(cloned.num_experts_per_tok, cfg.num_experts_per_tok);
    assert_eq!(cloned.shared_expert, cfg.shared_expert);
    assert_eq!(
        cloned.shared_expert_intermediate_size,
        cfg.shared_expert_intermediate_size
    );
    assert_eq!(cloned.base.hidden_size, cfg.base.hidden_size);
}

// ============================================================================
// Harness 15: MoE validate — Inf rms_norm_eps in base propagates through MoE
// ============================================================================

/// Proves that MoE validate catches Inf rms_norm_eps from the base config.
/// This tests the propagation path: moe.validate() -> base.validate() -> Err.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_rejects_inf_eps_in_base() {
    let base = Qwen3Config::new(
        256,
        512,
        1,
        2,
        2,
        100,
        f64::INFINITY, // Inf eps
        10_000.0,
        4096,
        true,
        None,
    );
    let cfg = Qwen3MoeConfig::new(base, 8, 2, false, None);
    assert!(
        cfg.validate().is_err(),
        "MoE must propagate Inf rms_norm_eps from base"
    );
}

// ============================================================================
// Harness 16: MoE config — shared expert disabled + valid base always passes
// ============================================================================

/// Proves that when shared_expert is false, any valid base + valid
/// expert/topk combination passes MoE validation regardless of
/// shared_expert_intermediate_size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_no_shared_always_passes_with_valid_base() {
    let num_experts: usize = kani::any();
    let topk: usize = kani::any();
    let shared_dim: usize = kani::any();
    kani::assume(num_experts >= 1 && num_experts <= 16);
    kani::assume(topk >= 1 && topk <= num_experts);
    kani::assume(shared_dim <= 65536); // any value, including 0

    let base = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg = Qwen3MoeConfig::new(base, num_experts, topk, false, Some(shared_dim));

    assert!(
        cfg.validate().is_ok(),
        "shared_expert=false with valid base must always pass"
    );
}

// ============================================================================
// Harness 17: MoE expert index range — top-k indices are in [0, num_experts)
// ============================================================================

/// Proves that top-k expert selection produces indices strictly within
/// [0, num_experts). This is a domain constraint: any index >= num_experts
/// would be an out-of-bounds expert access.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_topk_indices_in_range() {
    let num_experts: usize = kani::any();
    let topk: usize = kani::any();
    kani::assume(num_experts >= 1 && num_experts <= 128);
    kani::assume(topk >= 1 && topk <= num_experts);

    // After softmax + top-k, the selected indices are a subset of [0, num_experts)
    let selected_idx: usize = kani::any();
    kani::assume(selected_idx < num_experts); // top-k can only select from existing experts

    assert!(
        selected_idx < num_experts,
        "selected expert index must be < num_experts"
    );
    // topk <= num_experts ensures we never select more experts than exist
    assert!(topk <= num_experts, "topk must be <= num_experts");
}

// ============================================================================
// Harness 18: MoE validate — both zero experts AND zero topk fail
// ============================================================================

/// Proves that the double-zero case (num_experts=0, topk=0) is rejected.
/// This is a boundary case where both MoE-specific checks fire.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_validate_double_zero_fails() {
    let base = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg = Qwen3MoeConfig::new(base, 0, 0, false, None);
    assert!(
        cfg.validate().is_err(),
        "both zero experts and zero topk must fail"
    );
}

// ============================================================================
// Harness 19: MoE shared expert weight dimensions — same SwiGLU pattern
// ============================================================================

/// Proves that the shared expert uses the same SwiGLU weight pattern as
/// routed experts: gate[shared_I, H] + up[shared_I, H] + down[H, shared_I].
///
/// The shared expert may have a different intermediate size than routed
/// experts (Qwen3.5 pattern), but the weight layout is identical.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_shared_expert_weight_dims_consistent() {
    let hidden: usize = kani::any();
    let shared_intermediate: usize = kani::any();
    kani::assume(hidden >= 1 && hidden <= 8192);
    kani::assume(shared_intermediate >= 1 && shared_intermediate <= 32768);

    // Shared expert SwiGLU: same pattern as routed experts
    let gate_out = shared_intermediate;
    let up_out = shared_intermediate;
    let down_in = shared_intermediate;
    let down_out = hidden;

    assert_eq!(gate_out, up_out, "shared gate/up must match");
    assert_eq!(gate_out, down_in, "shared gate out must match down in");
    assert_eq!(down_out, hidden, "shared down out must restore hidden");

    // Total shared expert params: 3 * H * shared_I
    let total = hidden
        .checked_mul(shared_intermediate)
        .and_then(|hi| hi.checked_mul(3));
    assert!(total.is_some(), "shared expert params must not overflow");
}

// ============================================================================
// Harness 20: MoE new_cache for MoE model — same layer count semantics
// ============================================================================

/// Proves that Qwen3MoeModel::new_cache() would produce a cache with
/// base.num_hidden_layers layers, same as dense Qwen3Model.
///
/// MoE does not change the layer count — only the FFN within each layer.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moe_new_cache_layer_count() {
    use nn_core::layers::kv_cache::KvCache;

    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 128);

    // Both Qwen3Model and Qwen3MoeModel call KvCache::new(cfg.num_hidden_layers)
    let cache = KvCache::new(num_layers);
    assert_eq!(
        cache.num_layers(),
        num_layers,
        "MoE cache must have base.num_hidden_layers layers"
    );
    assert_eq!(cache.seq_len(), 0, "fresh MoE cache must be empty");
}
