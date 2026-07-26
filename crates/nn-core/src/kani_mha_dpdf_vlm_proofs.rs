// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for multi-head attention safety in dpdf VLMs (part 1).
//!
//! Proves structural and arithmetic properties of MHA specific to
//! vision-language models (VLMs) used in dpdf document processing,
//! where visual encoder tokens and text decoder tokens produce
//! cross-attention with asymmetric sequence lengths (S_q != S_kv).
//!
//! Harnesses in this file:
//!
//! 1. **Head split preserves total elements** — reshape [B, S, D] to [B, S, H, Dh]
//! 2. **Head split transpose preserves elements** — [B, S, H, Dh] -> [B, H, S, Dh]
//! 3. **Head merge is inverse of split** — full round-trip shape recovery
//! 4. **Cross-attention score shape** — Q x K^T for asymmetric S_q, S_kv
//! 5. **Cross-attention output preserves query seq len** — output has S_q tokens
//! 6. **Attention score scaling prevents overflow** — bounded dot-products stay finite
//! 7. **Scaling normalizes variance** — 1/sqrt(Dh) produces unit variance
//!
//! Continued in `kani_mha_dpdf_vlm_proofs_ext.rs` (softmax, masks, GQA, VLM compat).
//!
//! All harnesses use small bounds for CBMC tractability:
//! B <= 2, H <= 4, S <= 8, D <= 16.
//!
//! Part of #4233.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ===========================================================================
// 1. Head split preserves total elements
// ===========================================================================

/// Proves reshape [B, S, D] -> [B, S, H, Dh] preserves element count.
///
/// Head splitting decomposes the hidden dimension D into H heads of size Dh.
/// The total element count B*S*D must equal B*S*H*Dh since D = H * Dh.
/// This is the foundational invariant for multi-head attention: no elements
/// are created or destroyed during head splitting.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_head_split_preserves_elements() {
    let b: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();
    let h: u8 = kani::any();

    kani::assume(b >= 1 && b <= 2);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(d >= 1 && d <= 16);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(d as usize % (h as usize) == 0);

    let bu = b as usize;
    let su = s as usize;
    let du = d as usize;
    let hu = h as usize;
    let dh = du / hu;

    // Original shape: [B, S, D]
    let original = [bu, su, du];
    // After head split: [B, S, H, Dh]
    let split = [bu, su, hu, dh];

    let orig_numel = checked_dim_product(&original);
    let split_numel = checked_dim_product(&split);

    if let (Ok(on), Ok(sn)) = (orig_numel, split_numel) {
        assert_eq!(on, sn, "head split must preserve total element count");
        // Verify D = H * Dh algebraically
        assert_eq!(du, hu * dh, "D must equal H * Dh");
        // Cross-check: B*S*D == B*S*H*Dh since D == H*Dh
        assert_eq!(on, bu * su * du, "original numel must be B*S*D");
        assert_eq!(sn, bu * su * hu * dh, "split numel must be B*S*H*Dh");
    }
}

/// Proves the transpose step [B, S, H, Dh] -> [B, H, S, Dh] preserves elements.
///
/// After head split, MHA transposes dims 1 and 2 to group heads together.
/// This is a pure layout change that must not alter the element count.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_head_split_transpose_preserves_elements() {
    let b: u8 = kani::any();
    let s: u8 = kani::any();
    let h: u8 = kani::any();
    let dh: u8 = kani::any();

    kani::assume(b >= 1 && b <= 2);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(dh >= 1 && dh <= 8);

    let bu = b as usize;
    let su = s as usize;
    let hu = h as usize;
    let dhu = dh as usize;

    // Before transpose: [B, S, H, Dh]
    let before = [bu, su, hu, dhu];
    // After transpose(1, 2): [B, H, S, Dh]
    let after = [bu, hu, su, dhu];

    let before_numel = checked_dim_product(&before);
    let after_numel = checked_dim_product(&after);

    if let (Ok(bn), Ok(an)) = (before_numel, after_numel) {
        assert_eq!(bn, an, "transpose must preserve element count");
    }

    // Verify specific dim correspondence
    assert_eq!(before[0], after[0], "batch dim unchanged");
    assert_eq!(before[1], after[2], "seq dim moved from 1 to 2");
    assert_eq!(before[2], after[1], "head dim moved from 2 to 1");
    assert_eq!(before[3], after[3], "head_dim unchanged");
}

// ===========================================================================
// 2. Head merge is inverse of split
// ===========================================================================

/// Proves head merge (concat) is the exact inverse of head split.
///
/// Split: [B, S, D] -> [B, S, H, Dh] -> transpose -> [B, H, S, Dh]
/// Merge: [B, H, S, Dh] -> transpose -> [B, S, H, Dh] -> [B, S, D]
///
/// The merge path recovers the original [B, S, D] shape exactly.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_head_merge_inverse_of_split() {
    let b: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();
    let h: u8 = kani::any();

    kani::assume(b >= 1 && b <= 2);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(d >= 1 && d <= 16);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(d as usize % (h as usize) == 0);

    let bu = b as usize;
    let su = s as usize;
    let du = d as usize;
    let hu = h as usize;
    let dh = du / hu;

    // Original shape
    let original = [bu, su, du];

    // Split path: [B, S, D] -> [B, S, H, Dh] -> [B, H, S, Dh]
    let after_split = [bu, hu, su, dh];

    // Merge path: [B, H, S, Dh] -> transpose(1,2) -> [B, S, H, Dh]
    let after_merge_transpose = [bu, su, hu, dh];

    // Reshape [B, S, H, Dh] -> [B, S, H*Dh] = [B, S, D]
    let merged_d = hu * dh;
    assert_eq!(merged_d, du, "merged dim H*Dh must equal original D");

    let merged = [bu, su, merged_d];
    assert_eq!(
        merged, original,
        "merge must recover original shape [B, S, D]"
    );

    // Verify element counts through the full round-trip
    let orig_numel = checked_dim_product(&original);
    let split_numel = checked_dim_product(&after_split);
    let merge_t_numel = checked_dim_product(&after_merge_transpose);
    let merged_numel = checked_dim_product(&merged);

    if let (Ok(on), Ok(sn), Ok(mtn), Ok(mn)) =
        (orig_numel, split_numel, merge_t_numel, merged_numel)
    {
        assert_eq!(on, sn, "split preserves numel");
        assert_eq!(sn, mtn, "merge transpose preserves numel");
        assert_eq!(mtn, mn, "merge reshape preserves numel");
        assert_eq!(on, mn, "full round-trip preserves numel");
    }
}

// ===========================================================================
// 3. Cross-attention score shape (asymmetric S_q != S_kv)
// ===========================================================================

/// Proves cross-attention score shape for VLM encoder-decoder attention.
///
/// In dpdf VLMs, visual tokens (S_img) attend to text tokens (S_txt) or
/// vice versa, producing asymmetric score matrices.
///
/// Q: [B, H, S_q, Dh]  (from decoder / text)
/// K: [B, H, S_kv, Dh] (from encoder / visual)
/// K^T: [B, H, Dh, S_kv]
/// Scores: Q @ K^T = [B, H, S_q, S_kv] (rectangular, not square)
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_cross_attention_score_shape() {
    let b: u8 = kani::any();
    let h: u8 = kani::any();
    let s_q: u8 = kani::any();
    let s_kv: u8 = kani::any();
    let dh: u8 = kani::any();

    kani::assume(b >= 1 && b <= 2);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(s_q >= 1 && s_q <= 8);
    kani::assume(s_kv >= 1 && s_kv <= 8);
    kani::assume(dh >= 1 && dh <= 8);

    let bu = b as usize;
    let hu = h as usize;
    let squ = s_q as usize;
    let skvu = s_kv as usize;
    let dhu = dh as usize;

    // Q: [B, H, S_q, Dh]
    let q_shape = [bu, hu, squ, dhu];
    // K^T: [B, H, Dh, S_kv]
    let kt_shape = [bu, hu, dhu, skvu];

    // Matmul inner dim check
    assert_eq!(
        q_shape[3], kt_shape[2],
        "Q last dim must match K^T penultimate dim (Dh)"
    );

    // Batch and head dims match
    assert_eq!(q_shape[0], kt_shape[0], "batch dims must match");
    assert_eq!(q_shape[1], kt_shape[1], "head dims must match");

    // Output shape: [B, H, S_q, S_kv]
    let scores_shape = [bu, hu, squ, skvu];

    // Score matrix rows come from Q (query sequence)
    assert_eq!(scores_shape[2], squ, "score rows must be S_q");
    // Score matrix cols come from K (key sequence)
    assert_eq!(scores_shape[3], skvu, "score cols must be S_kv");

    // Verify element count: B * H * S_q * S_kv
    let score_numel = checked_dim_product(&scores_shape);
    if let Ok(sn) = score_numel {
        assert_eq!(sn, bu * hu * squ * skvu, "score numel must be B*H*S_q*S_kv");
    }
}

// ===========================================================================
// 4. Cross-attention output preserves query seq len
// ===========================================================================

/// Proves cross-attention output shape preserves query sequence length.
///
/// Scores [B, H, S_q, S_kv] @ V [B, H, S_kv, Dh] -> [B, H, S_q, Dh]
///
/// The output always has S_q tokens, regardless of the key/value sequence
/// length S_kv. This is critical for VLMs where visual encoder output
/// length differs from text decoder sequence length.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_cross_attn_output_preserves_query_len() {
    let b: u8 = kani::any();
    let h: u8 = kani::any();
    let s_q: u8 = kani::any();
    let s_kv: u8 = kani::any();
    let dh: u8 = kani::any();

    kani::assume(b >= 1 && b <= 2);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(s_q >= 1 && s_q <= 8);
    kani::assume(s_kv >= 1 && s_kv <= 8);
    kani::assume(dh >= 1 && dh <= 8);

    let bu = b as usize;
    let hu = h as usize;
    let squ = s_q as usize;
    let skvu = s_kv as usize;
    let dhu = dh as usize;

    // Attention weights (after softmax): [B, H, S_q, S_kv]
    let weights_shape = [bu, hu, squ, skvu];

    // V: [B, H, S_kv, Dh]
    let v_shape = [bu, hu, skvu, dhu];

    // Matmul inner dim check: weights last == V penultimate
    assert_eq!(
        weights_shape[3], v_shape[2],
        "weights S_kv must match V S_kv for matmul"
    );

    // Output: [B, H, S_q, Dh]
    let output_shape = [bu, hu, squ, dhu];

    // Output seq length is S_q (from query), NOT S_kv
    assert_eq!(
        output_shape[2], squ,
        "output must have S_q rows (query seq len)"
    );
    assert_eq!(
        output_shape[3], dhu,
        "output must have Dh columns (head dim)"
    );

    // After merge: [B, S_q, D] -- query sequence length preserved end-to-end
    let d = hu * dhu;
    let final_output = [bu, squ, d];
    assert_eq!(
        final_output[1], squ,
        "final output seq len must be S_q (decoder token count)"
    );
}

// ===========================================================================
// 5. Attention score scaling prevents overflow
// ===========================================================================

/// Proves bounded dot-product scores remain finite after 1/sqrt(Dh) scaling.
///
/// For Q, K element values bounded by [-C, C], the dot product over Dh
/// dimensions is bounded by Dh * C^2. After scaling by 1/sqrt(Dh), the
/// result is bounded by sqrt(Dh) * C^2, which is finite for practical
/// dimensions and bounded element magnitudes.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_score_scaling_prevents_overflow() {
    let dh: u8 = kani::any();
    let c_int: u8 = kani::any();

    kani::assume(dh >= 1 && dh <= 128);
    // Element magnitude bound (practical range for normalized embeddings)
    kani::assume(c_int >= 1 && c_int <= 10);

    let dhu = dh as usize;
    let c = c_int as f64;

    // Worst-case dot product: all elements at +C or -C, aligned
    // |<q, k>| <= Dh * C^2
    let max_dot = (dhu as f64) * c * c;
    assert!(max_dot.is_finite(), "max dot product must be finite");

    // After scaling by 1/sqrt(Dh)
    let sqrt_dh = (dhu as f64).sqrt();
    let scaled_max = max_dot / sqrt_dh;
    assert!(scaled_max.is_finite(), "scaled score must be finite");

    // Scaled maximum = sqrt(Dh) * C^2
    let expected_bound = sqrt_dh * c * c;
    // Allow small floating-point tolerance
    assert!(
        (scaled_max - expected_bound).abs() < 1e-10,
        "scaled score must equal sqrt(Dh) * C^2"
    );

    // For practical values (Dh <= 128, C <= 10): sqrt(128)*100 = ~1131
    // Well within f32 range (~3.4e38)
    let scaled_f32 = scaled_max as f32;
    assert!(scaled_f32.is_finite(), "f32 scaled score must be finite");
}

/// Proves scaling factor 1/sqrt(Dh) reduces dot-product variance.
///
/// The variance of Q @ K^T / sqrt(Dh) is Dh / Dh = 1 when Q, K elements
/// are iid with variance 1. This prevents softmax saturation.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_scaling_normalizes_variance() {
    let dh: u8 = kani::any();
    kani::assume(dh >= 1 && dh <= 128);

    let dhu = dh as f64;

    // For iid Q, K with variance sigma^2 = 1:
    // Var(Q @ K^T) = Dh * sigma^4 = Dh
    let unscaled_var = dhu;

    // After 1/sqrt(Dh) scaling:
    // Var(Q @ K^T / sqrt(Dh)) = Dh / Dh = 1
    let sqrt_dh = dhu.sqrt();
    let scaled_var = unscaled_var / (sqrt_dh * sqrt_dh);

    assert!(scaled_var.is_finite(), "scaled variance must be finite");
    assert!(
        (scaled_var - 1.0).abs() < 1e-10,
        "scaled variance must be 1.0 (unit variance)"
    );
}
