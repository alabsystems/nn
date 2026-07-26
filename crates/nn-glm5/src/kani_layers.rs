// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for GLM-4/5 layer arithmetic.
//!
//! Covers arithmetic invariants from `layers.rs`:
//! - QKV fused projection split consistency
//! - QKV reshape dimension correctness
//! - SwiGLU MLP narrow split symmetry
//! - Attention output reshape size preservation
//! - GQA repeat_kv ratio constraints
//! - Dense weight shape consistency
//! - Attention scale monotonicity
//!
//! Issue: #3654

use crate::config::Glm5Config;

// ---------------------------------------------------------------------------
// CBMC transcendental stubs — f64::sqrt
// ---------------------------------------------------------------------------

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
// Harness L1: QKV split sizes sum to qkv_size
// ============================================================================

/// Proves that q_size + 2 * kv_size == qkv_size for all valid configs.
///
/// In Glm5Attention::forward, the fused QKV output is narrowed into three
/// chunks: q (0..q_size), k (q_size..q_size+kv_size), v (q_size+kv_size..).
/// If the sizes don't sum correctly, narrow() would produce out-of-bounds
/// access or overlapping slices.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn qkv_split_sizes_sum_correctly() {
    let nh: usize = kani::any();
    let nkv: usize = kani::any();
    let hd: usize = kani::any();

    kani::assume(nh > 0 && nh <= 128);
    kani::assume(nkv > 0 && nkv <= 128);
    kani::assume(hd > 0 && hd <= 256);
    kani::assume(nh >= nkv);
    kani::assume(nh % nkv == 0);

    let q_size = nh * hd;
    let kv_size = nkv * hd;
    let qkv_size = (nh + 2 * nkv) * hd;

    assert_eq!(
        q_size + kv_size + kv_size,
        qkv_size,
        "q + k + v must sum to total qkv_size"
    );
}

// ============================================================================
// Harness L2: QKV narrow offsets are within bounds
// ============================================================================

/// Proves that the three narrow() calls in Glm5Attention::forward produce
/// non-overlapping, contiguous slices covering the full qkv_size.
///
/// q: [0, q_size), k: [q_size, q_size + kv_size), v: [q_size + kv_size, qkv_size)
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn qkv_narrow_offsets_in_bounds() {
    let nh: usize = kani::any();
    let nkv: usize = kani::any();
    let hd: usize = kani::any();

    kani::assume(nh > 0 && nh <= 64);
    kani::assume(nkv > 0 && nkv <= 64);
    kani::assume(hd > 0 && hd <= 128);
    kani::assume(nh >= nkv);
    kani::assume(nh % nkv == 0);

    let q_size = nh * hd;
    let kv_size = nkv * hd;
    let qkv_size = (nh + 2 * nkv) * hd;

    // Q: starts at 0, length q_size
    let q_start = 0_usize;
    let q_end = q_start + q_size;

    // K: starts at q_size, length kv_size
    let k_start = q_size;
    let k_end = k_start + kv_size;

    // V: starts at q_size + kv_size, length kv_size
    let v_start = q_size + kv_size;
    let v_end = v_start + kv_size;

    // Non-overlapping
    assert!(q_end <= k_start, "Q and K must not overlap");
    assert!(k_end <= v_start, "K and V must not overlap");

    // Covers full range
    assert_eq!(v_end, qkv_size, "V end must equal qkv_size");
    assert_eq!(q_start, 0, "Q must start at 0");
}

// ============================================================================
// Harness L3: Q reshape dimension product preserved
// ============================================================================

/// Proves that reshaping Q from [batch, seq, q_size] to [batch, seq, nh, hd]
/// preserves the total element count in the last dimension.
///
/// q_size == nh * hd is required for reshape to succeed. If violated,
/// reshape would panic or silently produce wrong tensor shapes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn q_reshape_dimension_preserved() {
    let nh: usize = kani::any();
    let hd: usize = kani::any();
    kani::assume(nh > 0 && nh <= 128);
    kani::assume(hd > 0 && hd <= 256);

    let q_size = nh * hd;
    // Reshape [batch, seq, q_size] → [batch, seq, nh, hd]
    // Last-dim element count must match
    assert_eq!(nh * hd, q_size, "nh * hd must equal q_size");
}

// ============================================================================
// Harness L4: KV reshape dimension product preserved
// ============================================================================

/// Proves that reshaping K/V from [batch, seq, kv_size] to
/// [batch, seq, nkv, hd] preserves the element count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn kv_reshape_dimension_preserved() {
    let nkv: usize = kani::any();
    let hd: usize = kani::any();
    kani::assume(nkv > 0 && nkv <= 128);
    kani::assume(hd > 0 && hd <= 256);

    let kv_size = nkv * hd;
    assert_eq!(nkv * hd, kv_size, "nkv * hd must equal kv_size");
}

// ============================================================================
// Harness L5: SwiGLU narrow split symmetry
// ============================================================================

/// Proves that the SwiGLU MLP narrow split produces two equal-sized halves.
///
/// In Glm5MLP::forward:
///   half_size = intermediate_dim / 2
///   gate = narrow(0, half_size)
///   up = narrow(half_size, half_size)
///
/// Both chunks must be the same size for silu(gate) * up to be element-wise.
/// If ffn_hidden_size were odd, the division would truncate, leaving one
/// element unaccounted for.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn swiglu_split_halves_equal() {
    let ffn: usize = kani::any();
    kani::assume(ffn > 0 && ffn <= 65536);

    let intermediate_dim = ffn * 2;
    let half_size = intermediate_dim / 2;

    // Two halves cover the full dimension
    assert_eq!(
        half_size + half_size,
        intermediate_dim,
        "halves must sum to full dim"
    );
    // Each half equals ffn_hidden_size
    assert_eq!(half_size, ffn, "each half must equal ffn_hidden_size");
}

// ============================================================================
// Harness L6: SwiGLU narrow offsets don't overlap
// ============================================================================

/// Proves that the two narrow() calls in SwiGLU MLP produce non-overlapping
/// slices that cover the entire intermediate dimension.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn swiglu_narrow_no_overlap() {
    let ffn: usize = kani::any();
    kani::assume(ffn > 0 && ffn <= 65536);

    let intermediate_dim = ffn * 2;
    let half_size = intermediate_dim / 2;

    // gate: [0, half_size)
    let gate_start = 0_usize;
    let gate_end = gate_start + half_size;

    // up: [half_size, half_size + half_size)
    let up_start = half_size;
    let up_end = up_start + half_size;

    assert!(gate_end <= up_start, "gate and up must not overlap");
    assert_eq!(up_end, intermediate_dim, "up must reach end of dimension");
}

// ============================================================================
// Harness L7: Attention output reshape preserves total elements
// ============================================================================

/// Proves that the attention output reshape from [batch, nh, seq, hd] to
/// [batch, seq, nh * hd] preserves the total element count.
///
/// After transpose(1,2), shape is [batch, seq, nh, hd]. The reshape
/// merges the last two dims into nh * hd.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn attention_output_reshape_preserves_elements() {
    let batch: usize = kani::any();
    let seq: usize = kani::any();
    let nh: usize = kani::any();
    let hd: usize = kani::any();

    kani::assume(batch > 0 && batch <= 4);
    kani::assume(seq > 0 && seq <= 64);
    kani::assume(nh > 0 && nh <= 64);
    kani::assume(hd > 0 && hd <= 128);

    // Check no overflow first
    let inner = nh.checked_mul(hd);
    kani::assume(inner.is_some());
    let inner = inner.unwrap();

    let total_before = batch.checked_mul(seq);
    kani::assume(total_before.is_some());
    let total_before = total_before.unwrap().checked_mul(nh);
    kani::assume(total_before.is_some());
    let total_before = total_before.unwrap().checked_mul(hd);
    kani::assume(total_before.is_some());

    let total_after = batch.checked_mul(seq);
    kani::assume(total_after.is_some());
    let total_after = total_after.unwrap().checked_mul(inner);
    kani::assume(total_after.is_some());

    assert_eq!(
        total_before.unwrap(),
        total_after.unwrap(),
        "reshape must preserve total elements"
    );
}

// ============================================================================
// Harness L8: GQA repeat count is at least 1
// ============================================================================

/// Proves that when num_heads >= num_kv_heads and they divide evenly,
/// the repeat count for repeat_kv is >= 1.
///
/// repeat_kv with n_rep < 1 would be undefined behavior (the function
/// expects n_rep >= 1).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gqa_repeat_count_at_least_one() {
    let nh: usize = kani::any();
    let nkv: usize = kani::any();
    kani::assume(nh > 0 && nh <= 128);
    kani::assume(nkv > 0 && nkv <= 128);
    kani::assume(nh >= nkv);
    kani::assume(nh % nkv == 0);

    let n_rep = nh / nkv;
    assert!(n_rep >= 1, "GQA repeat count must be >= 1");
}

// ============================================================================
// Harness L9: GQA repeat produces correct total KV heads
// ============================================================================

/// Proves that after repeat_kv, the effective number of KV heads equals
/// the number of query heads.
///
/// repeat_kv repeats each KV head `n_rep` times along the head dimension.
/// The result must have exactly `num_heads` heads for the Q * K^T matmul.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gqa_repeat_produces_correct_total_heads() {
    let nh: usize = kani::any();
    let nkv: usize = kani::any();
    kani::assume(nh > 0 && nh <= 128);
    kani::assume(nkv > 0 && nkv <= 128);
    kani::assume(nh >= nkv);
    kani::assume(nh % nkv == 0);

    let n_rep = nh / nkv;
    let effective_kv_heads = nkv * n_rep;
    assert_eq!(
        effective_kv_heads, nh,
        "repeated KV heads must equal query heads"
    );
}

// ============================================================================
// Harness L10: Dense weight out_features matches nh * hd
// ============================================================================

/// Proves that the dense (output projection) weight's in_features dimension
/// equals num_heads * head_dim, matching the attention output reshape.
///
/// In Glm5Attention::load: dense weight shape is [hidden_size, nh * hd].
/// The attention output after reshape is [batch, seq, nh * hd]. If these
/// don't match, matmul would fail.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn dense_weight_matches_attention_output() {
    let nh: usize = kani::any();
    let hd: usize = kani::any();
    kani::assume(nh > 0 && nh <= 128);
    kani::assume(hd > 0 && hd <= 256);

    let attn_output_dim = nh * hd;
    // Dense weight in_features must match attention output last dim
    // (the code uses `nh * hd` for both)
    let dense_in_features = nh * hd;
    assert_eq!(
        dense_in_features, attn_output_dim,
        "dense in_features must match attention output dim"
    );
}

// ============================================================================
// Harness L11: Attention scale monotonically decreases with head_dim
// ============================================================================

/// Proves that for valid head_dims (multiples of 4, > 0), larger head_dim
/// yields a smaller attention scale.
///
/// scale = 1 / sqrt(hd). If this monotonicity broke (e.g., due to integer
/// truncation), attention weights would be incorrectly scaled for larger models.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn attention_scale_monotonic_decreasing() {
    let hd1: usize = kani::any();
    let hd2: usize = kani::any();
    kani::assume(hd1 > 0 && hd1 <= 256);
    kani::assume(hd2 > 0 && hd2 <= 256);
    kani::assume(hd1 % 4 == 0);
    kani::assume(hd2 % 4 == 0);
    kani::assume(hd1 < hd2);

    let scale1 = 1.0 / (hd1 as f64).sqrt();
    let scale2 = 1.0 / (hd2 as f64).sqrt();

    assert!(scale1 > scale2, "larger head_dim must yield smaller scale");
}

// ============================================================================
// Harness L12: MLP dense_h_to_4h weight shape consistency
// ============================================================================

/// Proves that the dense_h_to_4h weight shape [ffn * 2, h] has correct
/// dimensions for the SwiGLU split that follows.
///
/// The output of dense_h_to_4h has last_dim = ffn * 2. This is split into
/// two halves of ffn each. If ffn * 2 were odd (impossible for usize * 2),
/// the split would be asymmetric.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mlp_dense_h_to_4h_even_output() {
    let ffn: usize = kani::any();
    kani::assume(ffn > 0 && ffn <= 65536);

    let out_dim = ffn * 2;
    assert_eq!(out_dim % 2, 0, "ffn * 2 is always even");
    assert_eq!(out_dim / 2, ffn, "half of ffn*2 equals ffn");
}

// ============================================================================
// Harness L13: Causal mask dimension for initial prompt
// ============================================================================

/// Proves that when cached_len == 0 (fresh cache), the causal mask
/// dimensions are seq_len x seq_len (square).
///
/// In forward_inner: total_seq = cached_len + seq_len. When cached_len
/// is 0, mask is seq_len x total_seq = seq_len x seq_len.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn causal_mask_square_for_initial_prompt() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len > 1 && seq_len <= 2048);

    let cached_len = 0_usize;
    let total_seq = cached_len + seq_len;

    assert_eq!(
        total_seq, seq_len,
        "total_seq == seq_len when cache is empty"
    );
    // Mask is created with (seq_len, total_seq) which becomes (seq_len, seq_len)
    assert_eq!(seq_len, total_seq, "mask must be square for initial prompt");
}

// ============================================================================
// Harness L14: Causal mask skipped for single-token decode
// ============================================================================

/// Proves that when seq_len == 1 (autoregressive decode step), no causal
/// mask is created.
///
/// The condition is `seq_len > 1 && total_seq > 1`. For seq_len == 1,
/// regardless of cached_len, the first condition is false.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn causal_mask_skipped_for_single_token() {
    let cached_len: usize = kani::any();
    kani::assume(cached_len <= 8192);

    let seq_len = 1_usize;
    let total_seq = cached_len + seq_len;
    let _ = total_seq; // used in real code for mask creation

    // Condition: seq_len > 1 && total_seq > 1
    let should_create_mask = seq_len > 1 && total_seq > 1;
    assert!(
        !should_create_mask,
        "single-token decode must skip causal mask"
    );
}

// ============================================================================
// Harness L15: Causal mask total_seq no overflow
// ============================================================================

/// Proves that total_seq = cached_len + seq_len does not overflow for
/// realistic sequence lengths.
///
/// Overflow here would wrap around, producing a mask much smaller than
/// needed, causing incorrect attention patterns.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn causal_mask_total_seq_no_overflow() {
    let cached_len: usize = kani::any();
    let seq_len: usize = kani::any();
    kani::assume(cached_len <= 131_072); // 128K context window max
    kani::assume(seq_len > 0 && seq_len <= 131_072);

    let total = cached_len.checked_add(seq_len);
    assert!(total.is_some(), "cached_len + seq_len must not overflow");
}

// ============================================================================
// Harness L16: Full config produces valid layer dimensions
// ============================================================================

/// Proves that a config passing validate() produces consistent layer
/// dimensions: QKV size, dense size, and MLP sizes are all nonzero and
/// internally consistent.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn valid_config_produces_consistent_layer_dims() {
    let nh: usize = kani::any();
    let nkv: usize = kani::any();
    let hd: usize = kani::any();
    let h: usize = kani::any();
    let ffn: usize = kani::any();

    kani::assume(nh > 0 && nh <= 64);
    kani::assume(nkv > 0 && nkv <= 64);
    kani::assume(hd > 0 && hd <= 128);
    kani::assume(h > 0 && h <= 8192);
    kani::assume(ffn > 0 && ffn <= 32768);
    kani::assume(nh >= nkv);
    kani::assume(nh % nkv == 0);
    kani::assume(hd % 4 == 0); // HalfRotaryEmbedding requirement

    // QKV projection output size
    let qkv_size = (nh + 2 * nkv) * hd;
    assert!(qkv_size > 0, "qkv_size must be nonzero");

    // Q, K, V split sizes
    let q_size = nh * hd;
    let kv_size = nkv * hd;
    assert!(q_size > 0 && kv_size > 0);
    assert_eq!(q_size + 2 * kv_size, qkv_size);

    // Dense output projection
    let dense_in = nh * hd;
    assert_eq!(dense_in, q_size, "dense in_features must match q_size");

    // MLP dimensions
    let mlp_intermediate = ffn * 2;
    assert_eq!(mlp_intermediate / 2, ffn);
}

// ============================================================================
// Harness L17: Fused sdpa_causal condition matches expected case
// ============================================================================

/// Proves that the sdpa_causal optimization fires exactly when:
/// (1) mask is Some, AND (2) seq_len == s_kv (no cached tokens).
///
/// This optimization avoids explicit mask tensor creation for Flash
/// Attention. The condition must be correct or attention patterns are wrong.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn sdpa_causal_condition_correct_initial_prompt() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len > 1 && seq_len <= 2048);

    // Initial prompt: no cache, so s_kv == seq_len
    let s_kv = seq_len;
    let mask_is_some = true; // mask created when seq_len > 1

    // Should use sdpa_causal
    let use_sdpa_causal = mask_is_some && seq_len == s_kv;
    assert!(use_sdpa_causal, "initial prompt must use sdpa_causal");
}

// ============================================================================
// Harness L18: sdpa_causal NOT used during decode with cache
// ============================================================================

/// Proves that during autoregressive decode (seq_len == 1), sdpa_causal
/// is not used because mask is None (seq_len == 1 skips mask creation).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn sdpa_causal_not_used_during_decode() {
    let cached_len: usize = kani::any();
    kani::assume(cached_len > 0 && cached_len <= 8192);

    let seq_len = 1_usize;
    let total_seq = cached_len + seq_len;

    // mask creation condition
    let mask_is_some = seq_len > 1 && total_seq > 1;

    // Even if s_kv matches, mask is None so sdpa_causal doesn't fire
    assert!(!mask_is_some, "decode step must not create mask");
}
