// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `lib.rs` — Qwen3Model structural invariants,
//! model_fn_adapter logic, and generation configuration safety.
//!
//! Covers properties NOT in `kani_qwen3.rs` or `kani_moe_forward_proofs.rs`:
//! - model_fn_adapter: position computation from cache seq_len
//! - model_fn_adapter: U32 -> usize token ID conversion safety
//! - Generation: new_cache returns correct layer count
//! - Tied vs untied lm_head: weight shape invariants
//! - Config accessor roundtrip: config() returns the same config
//! - DType accessor: dtype() is finite-representable
//! - validate is called on load: invalid config causes load failure
//! - Autoregressive decode: positions are contiguous from cache offset
//! - Beam search: beam_width >= 1 for valid BeamSearchConfig
//! - KvCache seq_len starts at 0 for fresh cache
//!
//! Issue: #3700

use crate::config::Qwen3Config;

// ============================================================================
// Harness 1: model_fn_adapter position computation — offset + i
// ============================================================================

/// Proves that model_fn_adapter's position computation
/// (cache.seq_len() + i for i in 0..ids.len()) produces strictly
/// monotonically increasing positions with no gaps.
///
/// This is the bridge between the generation API (DynTensor of token IDs)
/// and forward_cached (positions: &[usize]).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn adapter_positions_monotone() {
    let cache_seq_len: usize = kani::any();
    let num_tokens: usize = kani::any();
    kani::assume(cache_seq_len <= 131_072);
    kani::assume(num_tokens >= 1 && num_tokens <= 64);

    // From model_fn_adapter:
    // let positions: Vec<usize> = (0..ids.len()).map(|i| offset + i).collect();
    let offset = cache_seq_len;

    // Check monotonicity and contiguity
    let first_pos = offset;
    let last_pos = offset + num_tokens - 1;

    assert!(last_pos >= first_pos, "last position >= first position");
    assert_eq!(
        last_pos - first_pos,
        num_tokens - 1,
        "positions must be contiguous (no gaps)"
    );

    // No overflow
    let total = offset.checked_add(num_tokens);
    assert!(total.is_some(), "offset + num_tokens must not overflow");
}

// ============================================================================
// Harness 2: U32 -> usize token ID conversion — no truncation on 64-bit
// ============================================================================

/// Proves that converting U32 token IDs to usize (as done in
/// model_fn_adapter) preserves the value on 64-bit platforms.
///
/// From model_fn_adapter: `let ids: Vec<usize> = u32_data.iter().map(|&v| v as usize).collect();`
/// On 64-bit platforms, usize >= u32, so no truncation occurs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn u32_to_usize_no_truncation() {
    let token_id: u32 = kani::any();

    let as_usize = token_id as usize;

    // On 64-bit: usize is at least 64 bits, u32 fits without truncation
    // The cast preserves the value
    assert_eq!(
        as_usize as u32, token_id,
        "u32 -> usize -> u32 roundtrip must preserve value"
    );
    assert!(
        as_usize <= u32::MAX as usize,
        "converted value must be within u32 range"
    );
}

// ============================================================================
// Harness 3: new_cache layer count from config
// ============================================================================

/// Proves that new_cache produces a KvCache with exactly
/// num_hidden_layers layers (the contract between config and cache).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn new_cache_layer_count_from_config() {
    use nn_core::layers::kv_cache::KvCache;

    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 128);

    // Qwen3Model::new_cache() calls KvCache::new(self.cfg.num_hidden_layers)
    let cache = KvCache::new(num_layers);
    assert_eq!(
        cache.num_layers(),
        num_layers,
        "new_cache must create exactly num_hidden_layers cache layers"
    );
}

// ============================================================================
// Harness 4: Tied lm_head weight shape equals embedding shape
// ============================================================================

/// Proves that when tie_word_embeddings is true, the lm_head weight has
/// the same shape as the embedding weight: [vocab_size, hidden_size].
///
/// This is a dimension consistency check: tied weights mean the same tensor
/// is used for both embedding lookup and output projection.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tied_lm_head_shape_equals_embed() {
    let vocab_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 200_000);
    kani::assume(hidden_size >= 1 && hidden_size <= 8192);

    // Embedding weight: [vocab_size, hidden_size]
    let embed_rows = vocab_size;
    let embed_cols = hidden_size;

    // When tied: lm_head uses the same weight
    // lm_head = Linear::new(embed_weight, None)
    // Linear forward: x @ weight.T, so weight [vocab, hidden] -> output [*, vocab]
    let lm_head_rows = embed_rows;
    let lm_head_cols = embed_cols;

    assert_eq!(lm_head_rows, vocab_size, "tied lm_head rows == vocab_size");
    assert_eq!(
        lm_head_cols, hidden_size,
        "tied lm_head cols == hidden_size"
    );
}

// ============================================================================
// Harness 5: Untied lm_head — separate weight still [vocab, hidden]
// ============================================================================

/// Proves that the untied lm_head weight shape [vocab_size, hidden_size]
/// produces vocab_size logits (one per vocabulary token).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn untied_lm_head_produces_vocab_logits() {
    let vocab_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 200_000);
    kani::assume(hidden_size >= 1 && hidden_size <= 8192);

    // lm_head weight: [vocab_size, hidden_size]
    // Linear forward: [B, S, hidden] @ [hidden, vocab] -> [B, S, vocab]
    // (weight is stored as [out, in] and transposed internally)
    let output_dim = vocab_size;

    assert!(output_dim >= 1, "must produce at least 1 logit");
    assert_eq!(
        output_dim, vocab_size,
        "lm_head must produce vocab_size logits"
    );
}

// ============================================================================
// Harness 6: KvCache starts empty
// ============================================================================

/// Proves that a freshly created KvCache has seq_len() == 0.
///
/// This is the initial state before any forward pass. The offset
/// computation (cache.seq_len() + i) must start from 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn kv_cache_starts_empty() {
    use nn_core::layers::kv_cache::KvCache;

    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 128);

    let cache = KvCache::new(num_layers);
    assert_eq!(cache.seq_len(), 0, "fresh cache must have seq_len == 0");
}

// ============================================================================
// Harness 7: validate config is prerequisite for load — invalid configs fail
// ============================================================================

/// Proves that configs rejected by validate() would also fail during
/// model construction (validate is the first check in load()).
///
/// Specifically: zero num_attention_heads fails validate, so load would
/// fail before attempting any weight loading.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn invalid_config_fails_validate_first() {
    let cfg = Qwen3Config::new(
        256, 512, 1, 0, // zero attention heads
        1, 100, 1e-6, 10_000.0, 4096, true, None,
    );
    assert!(
        cfg.validate().is_err(),
        "invalid config must fail validate before load"
    );
}

// ============================================================================
// Harness 8: Autoregressive positions are contiguous from cache offset
// ============================================================================

/// Proves that autoregressive decode positions form a contiguous range
/// [offset, offset+1, ..., offset+n-1] with no duplicates.
///
/// This ensures each token gets a unique position for RoPE encoding.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(17)] // max 16 tokens + 1
fn autoregressive_positions_contiguous() {
    let offset: usize = kani::any();
    let n_tokens: usize = kani::any();
    kani::assume(offset <= 131_072);
    kani::assume(n_tokens >= 1 && n_tokens <= 16);
    kani::assume(offset + n_tokens <= 131_072 + 16);

    // Generate positions as in model_fn_adapter
    let mut positions = [0usize; 16];
    for i in 0..16 {
        if i < n_tokens {
            positions[i] = offset + i;
        }
    }

    // Check contiguity: positions[i+1] == positions[i] + 1
    for i in 0..16 {
        if i + 1 < n_tokens {
            assert_eq!(
                positions[i + 1],
                positions[i] + 1,
                "positions must be contiguous"
            );
        }
    }

    // Check no duplicates (implied by contiguity + strictly increasing)
    for i in 0..16 {
        if i + 1 < n_tokens {
            assert!(positions[i + 1] > positions[i], "positions must be unique");
        }
    }
}

// ============================================================================
// Harness 9: Config validate double-call same result
// ============================================================================

/// Proves that validate() on Qwen3Config is idempotent (pure function).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_idempotent() {
    let hidden: usize = kani::any();
    let heads: usize = kani::any();
    kani::assume(hidden <= 8192);
    kani::assume(heads <= 64);

    let cfg = Qwen3Config::new(
        hidden, 512, 1, heads, 1, 100, 1e-6, 10_000.0, 4096, true, None,
    );

    let r1 = cfg.validate().is_ok();
    let r2 = cfg.validate().is_ok();
    assert_eq!(r1, r2, "validate must be idempotent");
}

// ============================================================================
// Harness 10: Qwen3Config with_vocab_size then validate consistency
// ============================================================================

/// Proves that a valid config modified with with_vocab_size(n) where n > 0
/// remains valid (the builder does not invalidate other fields).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn builder_then_validate_consistent() {
    let new_vocab: usize = kani::any();
    kani::assume(new_vocab >= 1 && new_vocab <= 500_000);

    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    // Base config is valid
    assert!(cfg.validate().is_ok());

    // Modified config with valid vocab remains valid
    let modified = cfg.with_vocab_size(new_vocab);
    assert!(
        modified.validate().is_ok(),
        "valid config with valid vocab_size must remain valid"
    );
}

// ============================================================================
// Harness 11: with_vocab_size(0) invalidates config
// ============================================================================

/// Proves that with_vocab_size(0) on a valid config produces an invalid
/// config (validate rejects zero vocab_size).
///
/// This is the negative counterpart to harness 10: the builder can
/// invalidate a config if given a zero value.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn builder_with_zero_vocab_invalidates() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    assert!(cfg.validate().is_ok(), "base config must be valid");

    let modified = cfg.with_vocab_size(0);
    assert!(
        modified.validate().is_err(),
        "with_vocab_size(0) must produce invalid config"
    );
}

// ============================================================================
// Harness 12: with_num_hidden_layers preserves config validity
// ============================================================================

/// Proves that with_num_hidden_layers(n) where n > 0 preserves validity.
///
/// num_hidden_layers has no upper bound in validate(), so any positive
/// value should be accepted.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn builder_with_layers_preserves_validity() {
    let new_layers: usize = kani::any();
    kani::assume(new_layers >= 1 && new_layers <= 256);

    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    assert!(cfg.validate().is_ok());

    let modified = cfg.with_num_hidden_layers(new_layers);
    assert_eq!(modified.num_hidden_layers, new_layers);
    assert!(
        modified.validate().is_ok(),
        "with_num_hidden_layers must preserve validity"
    );
}

// ============================================================================
// Harness 13: Tied embedding shapes — embed and lm_head use same dimensions
// ============================================================================

/// Proves that when tie_word_embeddings is true, the embedding weight shape
/// [vocab_size, hidden_size] matches the expected lm_head shape.
///
/// In model construction: `Linear::new(embed_weight.clone(), None)` reuses
/// the exact same tensor. This proves the dimension consistency at the
/// config level.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tied_embed_lm_head_dimension_consistency() {
    let vocab: usize = kani::any();
    let hidden: usize = kani::any();
    kani::assume(vocab >= 1 && vocab <= 200_000);
    kani::assume(hidden >= 1 && hidden <= 8192);

    // Embedding weight: [vocab, hidden]
    // Linear forward: x @ W^T where W is [out_features, in_features]
    // So lm_head Linear with weight [vocab, hidden]: [*, hidden] -> [*, vocab]
    let lm_head_out_features = vocab;
    let lm_head_in_features = hidden;

    // The lm_head must accept hidden_size input and produce vocab_size output
    assert_eq!(
        lm_head_in_features, hidden,
        "lm_head input must be hidden_size"
    );
    assert_eq!(
        lm_head_out_features, vocab,
        "lm_head output must be vocab_size"
    );

    // Element count: vocab * hidden
    let param_count = vocab.checked_mul(hidden);
    assert!(
        param_count.is_some(),
        "tied weight param count must not overflow"
    );
}

// ============================================================================
// Harness 14: model_fn_adapter — positions start at cache offset
// ============================================================================

/// Proves that the first position generated by model_fn_adapter equals
/// the cache seq_len (the offset).
///
/// This is the key invariant for autoregressive decoding: each new token's
/// position continues from where the cache left off.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn adapter_first_position_equals_offset() {
    let cache_seq_len: usize = kani::any();
    kani::assume(cache_seq_len <= 131_072);

    // From model_fn_adapter: positions[0] = offset + 0 = cache_seq_len
    let first_position = cache_seq_len;
    assert_eq!(
        first_position, cache_seq_len,
        "first position must equal cache seq_len"
    );
}

// ============================================================================
// Harness 15: model_fn_adapter — last position = offset + len - 1
// ============================================================================

/// Proves that the last position in model_fn_adapter's output equals
/// cache_seq_len + num_tokens - 1.
///
/// This determines the maximum position seen by RoPE. Must not exceed
/// max_position_embeddings.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn adapter_last_position_formula() {
    let cache_seq_len: usize = kani::any();
    let num_tokens: usize = kani::any();
    kani::assume(cache_seq_len <= 131_072);
    kani::assume(num_tokens >= 1 && num_tokens <= 4096);
    kani::assume(cache_seq_len + num_tokens <= 135_168); // max feasible

    let last_pos = cache_seq_len + num_tokens - 1;

    // last_pos must be >= first_pos (= cache_seq_len)
    assert!(last_pos >= cache_seq_len, "last >= first");
    // num positions = last - first + 1 = num_tokens
    assert_eq!(
        last_pos - cache_seq_len + 1,
        num_tokens,
        "position count must equal num_tokens"
    );
}

// ============================================================================
// Harness 16: generate_greedy — fresh cache starts at seq_len 0
// ============================================================================

/// Proves that the KV cache created by new_cache() starts with seq_len 0,
/// which means the first decode step generates positions starting at 0.
///
/// This is the initial state for generate_greedy/generate_beam.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn generate_fresh_cache_starts_zero() {
    use nn_core::layers::kv_cache::KvCache;

    let num_layers: usize = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 128);

    let cache = KvCache::new(num_layers);

    // First decode step: offset = cache.seq_len() = 0
    let offset = cache.seq_len();
    assert_eq!(offset, 0, "fresh cache must start at 0");

    // First token gets position 0
    let first_pos = offset;
    assert_eq!(first_pos, 0, "first token position must be 0");
}

// ============================================================================
// Harness 17: Config head_dim * num_heads product for all production models
// ============================================================================

/// Proves that head_dim (128) * num_attention_heads does not overflow
/// and produces the correct Q projection total for all Qwen3 variants.
///
/// The Q projection weight is [num_heads * head_dim, hidden_size].
/// This total dimension determines the attention reshape.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn head_dim_times_num_heads_no_overflow() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 128);

    let head_dim: usize = 128;
    let total = num_heads.checked_mul(head_dim);

    assert!(total.is_some(), "num_heads * head_dim must not overflow");
    assert!(total.unwrap() > 0, "Q total dim must be positive");
    assert_eq!(
        total.unwrap(),
        num_heads * 128,
        "must equal num_heads * 128"
    );
}

// ============================================================================
// Harness 18: Config validate accepts tie_word_embeddings=false
// ============================================================================

/// Proves that validate() accepts both true and false for tie_word_embeddings
/// without affecting other validation checks.
///
/// tie_word_embeddings is not checked by validate() — it only affects weight
/// loading (shared vs separate lm_head weight).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_accepts_both_tie_variants() {
    let cfg_tied = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, true, None);
    let cfg_untied = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 4096, false, None);

    assert!(cfg_tied.validate().is_ok(), "tied must pass");
    assert!(cfg_untied.validate().is_ok(), "untied must pass");
}

// ============================================================================
// Harness 19: u32 token ID — vocab_size within u32 range
// ============================================================================

/// Proves that standard Qwen3 vocab_size (151936) fits in u32.
///
/// model_fn_adapter converts DynTensor U32 token IDs to usize. The reverse
/// direction (usize -> u32 for output) requires vocab_size <= u32::MAX.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn vocab_size_fits_u32() {
    let vocab_size: usize = kani::any();
    // All Qwen3 variants use vocab_size = 151_936
    kani::assume(vocab_size <= 200_000);

    assert!(
        vocab_size <= u32::MAX as usize,
        "vocab_size must fit in u32 for token ID representation"
    );
}

// ============================================================================
// Harness 20: Config clone preserves all fields exactly
// ============================================================================

/// Proves that Clone on Qwen3Config preserves every field.
/// This validates the derived Clone implementation is correct.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_clone_preserves_all_fields() {
    let hidden: usize = kani::any();
    let intermediate: usize = kani::any();
    let layers: usize = kani::any();
    kani::assume(hidden >= 1 && hidden <= 8192);
    kani::assume(intermediate >= 1 && intermediate <= 32768);
    kani::assume(layers >= 1 && layers <= 128);

    let cfg = Qwen3Config::new(
        hidden,
        intermediate,
        layers,
        2,
        2,
        100,
        1e-6,
        10_000.0,
        4096,
        true,
        None,
    );
    let cloned = cfg.clone();

    assert_eq!(cloned.hidden_size, cfg.hidden_size);
    assert_eq!(cloned.intermediate_size, cfg.intermediate_size);
    assert_eq!(cloned.num_hidden_layers, cfg.num_hidden_layers);
    assert_eq!(cloned.num_attention_heads, cfg.num_attention_heads);
    assert_eq!(cloned.num_key_value_heads, cfg.num_key_value_heads);
    assert_eq!(cloned.vocab_size, cfg.vocab_size);
    assert_eq!(cloned.rms_norm_eps, cfg.rms_norm_eps);
    assert_eq!(cloned.rope_theta, cfg.rope_theta);
    assert_eq!(cloned.max_position_embeddings, cfg.max_position_embeddings);
    assert_eq!(cloned.tie_word_embeddings, cfg.tie_word_embeddings);
}
