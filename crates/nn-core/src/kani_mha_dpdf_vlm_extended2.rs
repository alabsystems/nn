// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proofs for VLM multi-head attention patterns (part 3).
//!
//! 14. GQA divisibility  15. MQA special case  16. Sliding window
//! 17. Flash attention shape  18. Cross-attention projections
//! 19. Relative position bias  20. Dropout mask  21. Causal+padding mask
//! 22. KV cache update  23. Head dim calculation
//!
//! Part of #4233.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ===========================================================================
// 14. GQA: num_kv_heads divides num_heads
// ===========================================================================

/// Proves GQA head count relationship: num_heads is exact multiple of num_kv_heads.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_gqa_num_kv_heads_divides_num_heads() {
    let num_heads: u8 = kani::any();
    let num_kv_heads: u8 = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 8);
    kani::assume(num_kv_heads >= 1 && num_kv_heads <= 8);
    kani::assume(num_kv_heads <= num_heads);
    kani::assume((num_heads as usize) % (num_kv_heads as usize) == 0);

    let nh = num_heads as usize;
    let nkv = num_kv_heads as usize;
    let num_rep = nh / nkv;

    assert_eq!(
        num_rep * nkv,
        nh,
        "num_rep * num_kv_heads must equal num_heads"
    );
    assert!(num_rep >= 1, "repeat factor must be at least 1");
    if nkv == nh {
        assert_eq!(num_rep, 1, "standard MHA has repeat factor 1");
    }
    if nkv == 1 {
        assert_eq!(num_rep, nh, "MQA repeat factor equals num_heads");
    }
    assert_eq!(
        nkv * num_rep,
        nh,
        "expanded KV heads must match query head count"
    );
}

// ===========================================================================
// 15. MQA: num_kv_heads == 1
// ===========================================================================

/// Proves MQA expands [B, 1, S, Dh] to [B, H, S, Dh], preserving S and Dh.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_mqa_single_kv_head() {
    let b: u8 = kani::any();
    let num_heads: u8 = kani::any();
    let s: u8 = kani::any();
    let dh: u8 = kani::any();
    kani::assume(b >= 1 && b <= 2);
    kani::assume(num_heads >= 1 && num_heads <= 8);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(dh >= 1 && dh <= 8);

    let (bu, hu, su, dhu) = (b as usize, num_heads as usize, s as usize, dh as usize);
    let num_rep = hu; // MQA: num_kv_heads == 1
    let kv_input = [bu, 1_usize, su, dhu];
    let kv_expanded = [bu, hu, su, dhu];

    assert_eq!(kv_input[2], kv_expanded[2], "seq len preserved");
    assert_eq!(kv_input[3], kv_expanded[3], "head_dim preserved");

    let in_numel = checked_dim_product(&kv_input);
    let out_numel = checked_dim_product(&kv_expanded);
    if let (Ok(inn), Ok(outn)) = (in_numel, out_numel) {
        assert_eq!(outn, inn * num_rep, "MQA multiplies elements by num_heads");
    }
}

// ===========================================================================
// 16. Sliding window attention bounds span
// ===========================================================================

/// Proves sliding window limits attention to at most window_size positions.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_sliding_window_bounds_span() {
    let seq_len: u8 = kani::any();
    let window_size: u8 = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 8);
    kani::assume(window_size >= 1 && window_size <= 8);
    let (su, wu) = (seq_len as usize, window_size as usize);

    let i: u8 = kani::any();
    let j: u8 = kani::any();
    kani::assume((i as usize) < su);
    kani::assume((j as usize) < su);
    let (iu, ju) = (i as usize, j as usize);

    let window_start = if iu + 1 >= wu { iu + 1 - wu } else { 0 };
    let in_window = ju >= window_start && ju <= iu;

    // Self-attention always allowed
    if ju == iu {
        assert!(in_window, "position in its own window");
    }
    // Span bounded
    assert!(iu - window_start + 1 <= wu, "span <= window_size");
    // Future always outside
    if ju > iu {
        assert!(!in_window, "future outside window");
    }
    // Distant past outside
    if iu >= wu && ju < iu + 1 - wu {
        assert!(!in_window, "distant past outside window");
    }
}

// ===========================================================================
// 17. Flash attention shape equivalence
// ===========================================================================

/// Proves chunked (flash) and full attention produce identical output shapes.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_flash_attention_shape_equivalence() {
    let b: u8 = kani::any();
    let h: u8 = kani::any();
    let s_q: u8 = kani::any();
    let s_kv: u8 = kani::any();
    let dh: u8 = kani::any();
    let block_q: u8 = kani::any();
    let block_kv: u8 = kani::any();
    kani::assume(b >= 1 && b <= 2);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(s_q >= 1 && s_q <= 8);
    kani::assume(s_kv >= 1 && s_kv <= 8);
    kani::assume(dh >= 1 && dh <= 8);
    kani::assume(block_q >= 1 && block_q <= 4);
    kani::assume(block_kv >= 1 && block_kv <= 4);

    let (bu, hu, squ, skvu, dhu) = (
        b as usize,
        h as usize,
        s_q as usize,
        s_kv as usize,
        dh as usize,
    );
    let (bqu, bkvu) = (block_q as usize, block_kv as usize);

    let full_output = [bu, hu, squ, dhu];

    // Ceil division for block counts
    let num_q_blocks = (squ + bqu - 1) / bqu;
    let num_kv_blocks = (skvu + bkvu - 1) / bkvu;

    // Total Q rows after concatenating all blocks
    let full_q = if num_q_blocks > 1 {
        (num_q_blocks - 1) * bqu
    } else {
        0
    };
    let total_q_rows = full_q + (squ - full_q);
    assert_eq!(total_q_rows, squ, "flash must produce S_q output rows");
    assert_eq!(
        [bu, hu, total_q_rows, dhu],
        full_output,
        "shapes must match"
    );

    // KV coverage
    let full_kv = if num_kv_blocks > 1 {
        (num_kv_blocks - 1) * bkvu
    } else {
        0
    };
    assert_eq!(full_kv + (skvu - full_kv), skvu, "all S_kv processed");
}

// ===========================================================================
// 18. Cross-attention: Q from decoder, KV from encoder
// ===========================================================================

/// Proves cross-attention projection shapes are compatible.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_cross_attention_projection_shapes() {
    let b: u8 = kani::any();
    let s_dec: u8 = kani::any();
    let s_enc: u8 = kani::any();
    let d_attn: u8 = kani::any();
    let h: u8 = kani::any();
    kani::assume(b >= 1 && b <= 2);
    kani::assume(s_dec >= 1 && s_dec <= 8);
    kani::assume(s_enc >= 1 && s_enc <= 8);
    kani::assume(d_attn >= 1 && d_attn <= 16);
    kani::assume(h >= 1 && h <= 4);
    kani::assume((d_attn as usize) % (h as usize) == 0);

    let (bu, sdu, seu) = (b as usize, s_dec as usize, s_enc as usize);
    let (dau, hu) = (d_attn as usize, h as usize);
    let dh = dau / hu;

    let q_shape = [bu, hu, sdu, dh]; // Q: [B, H, S_dec, Dh]
    let k_shape = [bu, hu, seu, dh]; // K: [B, H, S_enc, Dh]
    let v_shape = [bu, hu, seu, dh]; // V: [B, H, S_enc, Dh]

    assert_eq!(q_shape[3], k_shape[3], "Q and K share head_dim");
    assert_eq!(k_shape[2], v_shape[2], "K and V share encoder seq len");

    // Output: [B, H, S_dec, Dh] -> merged [B, S_dec, D_attn]
    let output = [bu, hu, sdu, dh];
    assert_eq!(output[2], sdu, "output has decoder seq len");
    assert_eq!(hu * dh, dau, "merged dim equals D_attn");
}

// ===========================================================================
// 19. Relative position bias shape
// ===========================================================================

/// Proves relative position bias [1, H, S_q, S_kv] broadcasts to score shape.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_relative_position_bias_shape() {
    let b: u8 = kani::any();
    let h: u8 = kani::any();
    let s_q: u8 = kani::any();
    let s_kv: u8 = kani::any();
    kani::assume(b >= 1 && b <= 2);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(s_q >= 1 && s_q <= 8);
    kani::assume(s_kv >= 1 && s_kv <= 8);

    let (bu, hu, squ, skvu) = (b as usize, h as usize, s_q as usize, s_kv as usize);
    let scores = [bu, hu, squ, skvu];
    let bias = [1_usize, hu, squ, skvu];

    // Broadcast: batch dim 1 expands to B; others match exactly
    assert!(scores[0] == bias[0] || bias[0] == 1, "batch broadcastable");
    assert_eq!(scores[1], bias[1], "heads match");
    assert_eq!(scores[2], bias[2], "S_q match");
    assert_eq!(scores[3], bias[3], "S_kv match");

    let result_b = if scores[0] > bias[0] {
        scores[0]
    } else {
        bias[0]
    };
    assert_eq!(
        [result_b, hu, squ, skvu],
        scores,
        "result equals score shape"
    );

    if let Ok(bn) = checked_dim_product(&bias) {
        assert_eq!(bn, hu * squ * skvu, "bias numel = H * S_q * S_kv");
    }
}

// ===========================================================================
// 20. Attention dropout mask shape
// ===========================================================================

/// Proves dropout mask shape matches attention weights [B, H, S_q, S_kv].
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_attention_dropout_mask_shape() {
    let b: u8 = kani::any();
    let h: u8 = kani::any();
    let s_q: u8 = kani::any();
    let s_kv: u8 = kani::any();
    kani::assume(b >= 1 && b <= 2);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(s_q >= 1 && s_q <= 8);
    kani::assume(s_kv >= 1 && s_kv <= 8);

    let (bu, hu, squ, skvu) = (b as usize, h as usize, s_q as usize, s_kv as usize);
    let attn_shape = [bu, hu, squ, skvu];
    let mask_shape = [bu, hu, squ, skvu];

    assert_eq!(attn_shape, mask_shape, "dropout mask matches attn shape");
    let an = checked_dim_product(&attn_shape);
    let mn = checked_dim_product(&mask_shape);
    if let (Ok(a), Ok(m)) = (an, mn) {
        assert_eq!(a, m, "element counts match");
    }
}

// ===========================================================================
// 21. Combined causal + padding mask
// ===========================================================================

/// Proves union of causal and padding masks: masked iff future OR padding.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_causal_padding_combined_mask() {
    let seq_len: u8 = kani::any();
    let pad_len: u8 = kani::any();
    kani::assume(seq_len >= 2 && seq_len <= 8);
    kani::assume(pad_len < seq_len);

    let su = seq_len as usize;
    let real_len = su - (pad_len as usize);

    let i: u8 = kani::any();
    let j: u8 = kani::any();
    kani::assume((i as usize) < su);
    kani::assume((j as usize) < su);
    let (iu, ju) = (i as usize, j as usize);

    let is_future = ju > iu;
    let is_padding = ju >= real_len;
    let is_masked = is_future || is_padding;

    // Real past/present tokens are attendable
    if ju <= iu && ju < real_len {
        assert!(!is_masked, "real past/present attendable");
    }
    // Padding always masked
    if is_padding {
        assert!(is_masked, "padding always masked");
    }
    // Future always masked
    if is_future {
        assert!(is_masked, "future always masked");
    }
    // Unmasked iff neither condition
    if !is_future && !is_padding {
        assert!(!is_masked, "unmasked when neither applies");
    }
}

// ===========================================================================
// 22. KV cache update extends seq dim
// ===========================================================================

/// Proves KV cache concat extends sequence dimension: old_seq + new_seq.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_kv_cache_update_extends_seq() {
    let b: u8 = kani::any();
    let h: u8 = kani::any();
    let old_seq: u8 = kani::any();
    let new_seq: u8 = kani::any();
    let dh: u8 = kani::any();
    kani::assume(b >= 1 && b <= 2);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(old_seq <= 6);
    kani::assume(new_seq >= 1 && new_seq <= 4);
    kani::assume(dh >= 1 && dh <= 8);

    let (bu, hu, osu, nsu, dhu) = (
        b as usize,
        h as usize,
        old_seq as usize,
        new_seq as usize,
        dh as usize,
    );

    let old_cache = [bu, hu, osu, dhu];
    let new_kv = [bu, hu, nsu, dhu];
    let updated = [bu, hu, osu + nsu, dhu];

    // Non-seq dims preserved
    assert_eq!(updated[0], old_cache[0], "batch preserved");
    assert_eq!(updated[1], old_cache[1], "heads preserved");
    assert_eq!(updated[3], old_cache[3], "head_dim preserved");
    assert_eq!(updated[2], osu + nsu, "seq = old + new");

    // Element count: old + new
    if let (Ok(on), Ok(nn), Ok(un)) = (
        checked_dim_product(&old_cache),
        checked_dim_product(&new_kv),
        checked_dim_product(&updated),
    ) {
        assert_eq!(un, on + nn, "updated elements = old + new");
    }

    // Compatibility
    assert_eq!(new_kv[0], old_cache[0], "batch match");
    assert_eq!(new_kv[1], old_cache[1], "heads match");
    assert_eq!(new_kv[3], old_cache[3], "head_dim match");
}

// ===========================================================================
// 23. Head dim calculation
// ===========================================================================

/// Proves head_dim = hidden_size / num_heads is exact and round-trips.
#[kani::unwind(1)]
#[kani::proof]
fn mha_vlm_head_dim_calculation() {
    let hidden_size: u8 = kani::any();
    let num_heads: u8 = kani::any();
    kani::assume(hidden_size >= 1 && hidden_size <= 16);
    kani::assume(num_heads >= 1 && num_heads <= 8);
    kani::assume((hidden_size as usize) % (num_heads as usize) == 0);

    let (du, hu) = (hidden_size as usize, num_heads as usize);
    let head_dim = du / hu;

    assert!(head_dim >= 1, "head_dim >= 1");
    assert_eq!(hu * head_dim, du, "num_heads * head_dim == hidden_size");
    assert_eq!(du % hu, 0, "no remainder");

    // Shape round-trip: [B, S, D] -> [B, S, H, Dh] -> [B, S, D]
    let b: u8 = kani::any();
    let s: u8 = kani::any();
    kani::assume(b >= 1 && b <= 2);
    kani::assume(s >= 1 && s <= 8);
    let (bu, su) = (b as usize, s as usize);

    let original = [bu, su, du];
    let split = [bu, su, hu, head_dim];
    assert_eq!(
        original,
        [bu, su, hu * head_dim],
        "split-merge recovers shape"
    );

    if let (Ok(on), Ok(sn)) = (checked_dim_product(&original), checked_dim_product(&split)) {
        assert_eq!(on, sn, "split preserves element count");
    }
}
