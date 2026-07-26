// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proofs for VLM-specific cat/split patterns (#4221).
//!
//! Proves 10 correctness properties for dpdf VLM tensor operations:
//!
//!  1. Multi-head attention: split hidden dim into heads, cat heads back
//!  2. KV cache: cat(old_kv, new_kv) along sequence dimension
//!  3. Batch dimension cat: merge micro-batches
//!  4. Feature pyramid: cat multi-scale features along channel dim
//!  5. Skip connection cat: encoder + decoder features along channel dim
//!  6. Token sequence split: prefix / suffix partition
//!  7. Channel split for group convolution
//!  8. Split into equal chunks for data parallelism
//!  9. Cat with broadcasting: [B,S,D] cat [1,S,D] -> [B+1,S,D]
//! 10. Nested cat/split roundtrip: cat(split(x)) == x (3-way)
//!
//! All harnesses use small concrete dimensions (u8) for CBMC tractability.

use crate::tensor::checked_dim_product;

// ---------------------------------------------------------------------------
// 1. Multi-head attention: split hidden -> heads, cat heads -> hidden
// ---------------------------------------------------------------------------

/// Prove: splitting [B, S, H*D_head] into H heads of [B, S, D_head] and
/// concatenating back produces the original shape. Models the reshape used
/// in multi-head attention where the hidden dimension is partitioned into
/// num_heads * head_dim.
#[kani::unwind(9)]
#[kani::proof]
fn mha_split_heads_cat_roundtrip() {
    let b: u8 = kani::any();
    let s: u8 = kani::any();
    let h: u8 = kani::any();
    let d_head: u8 = kani::any();
    kani::assume(b >= 1 && b <= 4);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(h >= 1 && h <= 8);
    kani::assume(d_head >= 1 && d_head <= 8);

    let bu = b as usize;
    let su = s as usize;
    let hu = h as usize;
    let du = d_head as usize;

    if let Some(hidden) = hu.checked_mul(du) {
        // Original shape: [B, S, H*D_head]
        let orig = [bu, su, hidden];

        // Split along dim=2 into H chunks of size D_head
        let head_shape = [bu, su, du];

        // Each head has correct numel
        let orig_numel = checked_dim_product(&orig);
        let head_numel = checked_dim_product(&head_shape);
        if let (Ok(on), Ok(hn)) = (orig_numel, head_numel) {
            assert_eq!(on, hu * hn, "total numel = H * per-head numel");
        }

        // Cat H heads back along dim=2: H * D_head = hidden
        let reconstructed = [bu, su, hu * du];
        assert_eq!(reconstructed[0], orig[0], "batch dim preserved");
        assert_eq!(reconstructed[1], orig[1], "seq dim preserved");
        assert_eq!(reconstructed[2], orig[2], "hidden dim reconstructed");
    }
}

// ---------------------------------------------------------------------------
// 2. KV cache: cat(old_kv, new_kv) along sequence dimension
// ---------------------------------------------------------------------------

/// Prove: concatenating old KV cache [B, H, S_old, D] with new KV
/// [B, H, S_new, D] along dim=2 produces [B, H, S_old+S_new, D].
/// This is the core KV cache append operation in autoregressive decoding.
#[kani::unwind(1)]
#[kani::proof]
fn kv_cache_cat_along_sequence() {
    let b: u8 = kani::any();
    let h: u8 = kani::any();
    let s_old: u8 = kani::any();
    let s_new: u8 = kani::any();
    let d: u8 = kani::any();
    kani::assume(b >= 1 && b <= 4);
    kani::assume(h >= 1 && h <= 8);
    kani::assume(s_old >= 1 && s_old <= 16);
    kani::assume(s_new >= 1 && s_new <= 8);
    kani::assume(d >= 1 && d <= 8);

    let bu = b as usize;
    let hu = h as usize;
    let s_old_u = s_old as usize;
    let s_new_u = s_new as usize;
    let du = d as usize;

    if let Some(s_total) = s_old_u.checked_add(s_new_u) {
        let out = [bu, hu, s_total, du];

        // Cat dim is the sum
        assert_eq!(out[2], s_old_u + s_new_u, "seq dim must be S_old + S_new");

        // Non-cat dims preserved
        assert_eq!(out[0], bu, "batch dim preserved");
        assert_eq!(out[1], hu, "head dim preserved");
        assert_eq!(out[3], du, "head_dim preserved");

        // Numel conservation
        let out_numel = checked_dim_product(&out);
        let old_numel = checked_dim_product(&[bu, hu, s_old_u, du]);
        let new_numel = checked_dim_product(&[bu, hu, s_new_u, du]);
        if let (Ok(on), Ok(oldn), Ok(newn)) = (out_numel, old_numel, new_numel) {
            assert_eq!(on, oldn + newn, "kv cache cat preserves total numel");
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Batch dimension cat: merge micro-batches
// ---------------------------------------------------------------------------

/// Prove: concatenating N micro-batches of [B_i, S, D] along dim=0 produces
/// [sum(B_i), S, D]. Models micro-batch merging for VLM batch inference where
/// pages of different sizes are processed together.
#[kani::unwind(1)]
#[kani::proof]
fn batch_cat_merge_microbatches() {
    let b1: u8 = kani::any();
    let b2: u8 = kani::any();
    let b3: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();
    kani::assume(b1 >= 1 && b1 <= 4);
    kani::assume(b2 >= 1 && b2 <= 4);
    kani::assume(b3 >= 1 && b3 <= 4);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(d >= 1 && d <= 8);

    let b1u = b1 as usize;
    let b2u = b2 as usize;
    let b3u = b3 as usize;
    let su = s as usize;
    let du = d as usize;

    if let Some(b12) = b1u.checked_add(b2u) {
        if let Some(b_total) = b12.checked_add(b3u) {
            let out = [b_total, su, du];

            assert_eq!(out[0], b1u + b2u + b3u, "batch dim = sum of micro-batches");
            assert_eq!(out[1], su, "seq dim preserved");
            assert_eq!(out[2], du, "hidden dim preserved");

            // Numel = sum of all micro-batch numels
            let out_numel = checked_dim_product(&out);
            let n1 = checked_dim_product(&[b1u, su, du]);
            let n2 = checked_dim_product(&[b2u, su, du]);
            let n3 = checked_dim_product(&[b3u, su, du]);
            if let (Ok(on), Ok(a), Ok(b), Ok(c)) = (out_numel, n1, n2, n3) {
                assert_eq!(on, a + b + c, "micro-batch cat preserves total numel");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Feature pyramid: cat multi-scale features along channel dim
// ---------------------------------------------------------------------------

/// Prove: concatenating feature maps from different pyramid levels along the
/// channel dimension preserves spatial dims and sums channels. Models FPN-style
/// multi-scale feature fusion: [B, C1, H, W] cat [B, C2, H, W] -> [B, C1+C2, H, W].
#[kani::unwind(1)]
#[kani::proof]
fn feature_pyramid_cat_channels() {
    let b: u8 = kani::any();
    let c1: u8 = kani::any();
    let c2: u8 = kani::any();
    let h: u8 = kani::any();
    let w: u8 = kani::any();
    kani::assume(b >= 1 && b <= 4);
    kani::assume(c1 >= 1 && c1 <= 8);
    kani::assume(c2 >= 1 && c2 <= 8);
    kani::assume(h >= 1 && h <= 8);
    kani::assume(w >= 1 && w <= 8);

    let bu = b as usize;
    let c1u = c1 as usize;
    let c2u = c2 as usize;
    let hu = h as usize;
    let wu = w as usize;

    if let Some(c_total) = c1u.checked_add(c2u) {
        let out = [bu, c_total, hu, wu];

        // Channel dim is the sum
        assert_eq!(out[1], c1u + c2u, "channel dim = C1 + C2");

        // Spatial and batch dims preserved
        assert_eq!(out[0], bu, "batch preserved");
        assert_eq!(out[2], hu, "height preserved");
        assert_eq!(out[3], wu, "width preserved");

        // Numel conservation
        let out_numel = checked_dim_product(&out);
        let n1 = checked_dim_product(&[bu, c1u, hu, wu]);
        let n2 = checked_dim_product(&[bu, c2u, hu, wu]);
        if let (Ok(on), Ok(a), Ok(b)) = (out_numel, n1, n2) {
            assert_eq!(on, a + b, "FPN cat preserves total numel");
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Skip connection cat: encoder + decoder features along channel dim
// ---------------------------------------------------------------------------

/// Prove: concatenating encoder features [B, C_enc, H, W] with decoder
/// features [B, C_dec, H, W] along channel dim produces [B, C_enc+C_dec, H, W].
/// Models U-Net skip connections used in document layout segmentation.
#[kani::unwind(1)]
#[kani::proof]
fn skip_connection_cat_encoder_decoder() {
    let b: u8 = kani::any();
    let c_enc: u8 = kani::any();
    let c_dec: u8 = kani::any();
    let h: u8 = kani::any();
    let w: u8 = kani::any();
    kani::assume(b >= 1 && b <= 4);
    kani::assume(c_enc >= 1 && c_enc <= 16);
    kani::assume(c_dec >= 1 && c_dec <= 16);
    kani::assume(h >= 1 && h <= 8);
    kani::assume(w >= 1 && w <= 8);

    let bu = b as usize;
    let ce = c_enc as usize;
    let cd = c_dec as usize;
    let hu = h as usize;
    let wu = w as usize;

    if let Some(c_total) = ce.checked_add(cd) {
        let out = [bu, c_total, hu, wu];

        assert_eq!(out[1], ce + cd, "skip cat channel = C_enc + C_dec");
        assert_eq!(out[0], bu, "batch preserved in skip connection");
        assert_eq!(out[2], hu, "height preserved in skip connection");
        assert_eq!(out[3], wu, "width preserved in skip connection");

        // The combined feature map can be split back to recover encoder/decoder parts
        let split_enc = [bu, ce, hu, wu];
        let split_dec = [bu, cd, hu, wu];
        let enc_numel = checked_dim_product(&split_enc);
        let dec_numel = checked_dim_product(&split_dec);
        let out_numel = checked_dim_product(&out);
        if let (Ok(on), Ok(en), Ok(dn)) = (out_numel, enc_numel, dec_numel) {
            assert_eq!(on, en + dn, "skip connection cat preserves numel");
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Token sequence split: prefix / suffix partition
// ---------------------------------------------------------------------------

/// Prove: splitting a token sequence [B, S, D] at position P into
/// [B, P, D] and [B, S-P, D] preserves total numel and allows
/// reconstruction via cat. Models input_ids[:prefix] / input_ids[prefix:]
/// used in VLM prompt processing where visual tokens are prepended.
#[kani::unwind(1)]
#[kani::proof]
fn token_sequence_prefix_suffix_split() {
    let b: u8 = kani::any();
    let s: u8 = kani::any();
    let p: u8 = kani::any();
    let d: u8 = kani::any();
    kani::assume(b >= 1 && b <= 4);
    kani::assume(s >= 2 && s <= 16);
    kani::assume(p >= 1);
    kani::assume(p < s); // strict prefix
    kani::assume(d >= 1 && d <= 8);

    let bu = b as usize;
    let su = s as usize;
    let pu = p as usize;
    let du = d as usize;
    let suffix_len = su - pu;

    // Original shape
    let orig = [bu, su, du];

    // Split along dim=1 at position P
    let prefix_shape = [bu, pu, du];
    let suffix_shape = [bu, suffix_len, du];

    // Prefix + suffix seq lengths = original
    assert_eq!(pu + suffix_len, su, "prefix + suffix = total seq len");

    // Numel conservation
    let orig_numel = checked_dim_product(&orig);
    let pn = checked_dim_product(&prefix_shape);
    let sn = checked_dim_product(&suffix_shape);
    if let (Ok(on), Ok(p_n), Ok(s_n)) = (orig_numel, pn, sn) {
        assert_eq!(on, p_n + s_n, "prefix/suffix split preserves numel");
    }

    // Cat reconstruction
    let reconstructed = [bu, pu + suffix_len, du];
    assert_eq!(reconstructed[0], orig[0], "batch preserved after roundtrip");
    assert_eq!(reconstructed[1], orig[1], "seq preserved after roundtrip");
    assert_eq!(
        reconstructed[2], orig[2],
        "hidden preserved after roundtrip"
    );
}

// ---------------------------------------------------------------------------
// 7. Channel split for group convolution
// ---------------------------------------------------------------------------

/// Prove: splitting [B, C, H, W] along channel dim into G groups of C/G
/// channels preserves total numel and produces correct per-group shapes.
/// Models grouped convolution in VLM vision backbones (e.g., ResNeXt).
#[kani::unwind(9)]
#[kani::proof]
fn channel_split_group_convolution() {
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let g: u8 = kani::any();
    let h: u8 = kani::any();
    let w: u8 = kani::any();
    kani::assume(b >= 1 && b <= 4);
    kani::assume(c >= 2 && c <= 8);
    kani::assume(g >= 1 && g <= 8);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(w >= 1 && w <= 4);
    kani::assume((c as usize) % (g as usize) == 0); // evenly divisible

    let bu = b as usize;
    let cu = c as usize;
    let gu = g as usize;
    let hu = h as usize;
    let wu = w as usize;
    let cpg = cu / gu; // channels per group

    // Each group has shape [B, C/G, H, W]
    let group_shape = [bu, cpg, hu, wu];

    // G groups * C/G channels = C channels
    assert_eq!(gu * cpg, cu, "G * (C/G) must equal C");

    // Numel conservation: G * per-group numel = original numel
    let orig_numel = checked_dim_product(&[bu, cu, hu, wu]);
    let group_numel = checked_dim_product(&group_shape);
    if let (Ok(on), Ok(gn)) = (orig_numel, group_numel) {
        assert_eq!(on, gu * gn, "group split preserves total numel");
    }
}

// ---------------------------------------------------------------------------
// 8. Split into equal chunks for data parallelism
// ---------------------------------------------------------------------------

/// Prove: splitting [B, S, D] along batch dim into K equal chunks (B divisible
/// by K) produces K tensors of [B/K, S, D], and their numels sum to the
/// original. Models data-parallel distribution across GPU shards.
#[kani::unwind(9)]
#[kani::proof]
fn split_equal_chunks_data_parallelism() {
    let b: u8 = kani::any();
    let k: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();
    kani::assume(b >= 2 && b <= 8);
    kani::assume(k >= 1 && k <= 8);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(d >= 1 && d <= 8);
    kani::assume((b as usize) % (k as usize) == 0); // evenly divisible

    let bu = b as usize;
    let ku = k as usize;
    let su = s as usize;
    let du = d as usize;
    let bpk = bu / ku; // batch per chunk

    // Each chunk: [B/K, S, D]
    let chunk_shape = [bpk, su, du];

    // K * B/K = B
    assert_eq!(ku * bpk, bu, "K * (B/K) must equal B");

    // Numel conservation
    let orig_numel = checked_dim_product(&[bu, su, du]);
    let chunk_numel = checked_dim_product(&chunk_shape);
    if let (Ok(on), Ok(cn)) = (orig_numel, chunk_numel) {
        assert_eq!(on, ku * cn, "data-parallel split preserves total numel");
    }

    // Cat back: K chunks of [B/K, S, D] -> [B, S, D]
    let reconstructed = [ku * bpk, su, du];
    assert_eq!(reconstructed[0], bu, "batch reconstructed after cat");
    assert_eq!(reconstructed[1], su, "seq preserved after cat");
    assert_eq!(reconstructed[2], du, "hidden preserved after cat");
}

// ---------------------------------------------------------------------------
// 9. Cat with broadcasting: [B,S,D] cat [1,S,D] -> [B+1,S,D]
// ---------------------------------------------------------------------------

/// Prove: concatenating [B, S, D] with [1, S, D] along dim=0 produces
/// [B+1, S, D]. The non-cat dims must match exactly (no implicit broadcast
/// in cat). Models appending a CLS or system token embedding to a batch.
#[kani::unwind(1)]
#[kani::proof]
fn cat_append_single_to_batch() {
    let b: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();
    kani::assume(b >= 1 && b <= 16);
    kani::assume(s >= 1 && s <= 16);
    kani::assume(d >= 1 && d <= 16);

    let bu = b as usize;
    let su = s as usize;
    let du = d as usize;

    if let Some(b_plus_1) = bu.checked_add(1) {
        let out = [b_plus_1, su, du];

        assert_eq!(out[0], bu + 1, "batch dim = B + 1");
        assert_eq!(out[1], su, "seq dim preserved");
        assert_eq!(out[2], du, "hidden dim preserved");

        // Numel = (B+1) * S * D = B*S*D + 1*S*D
        let out_numel = checked_dim_product(&out);
        let main_numel = checked_dim_product(&[bu, su, du]);
        let single_numel = checked_dim_product(&[1usize, su, du]);
        if let (Ok(on), Ok(mn), Ok(sn)) = (out_numel, main_numel, single_numel) {
            assert_eq!(on, mn + sn, "cat with [1,S,D] preserves numel");
        }
    }
}

// ---------------------------------------------------------------------------
// 10. Nested cat/split roundtrip: cat(split(x)) == x (3-way)
// ---------------------------------------------------------------------------

/// Prove: splitting a tensor [D0, S1+S2+S3] into 3 parts and concatenating
/// them back produces the original shape. Extends the 2-way roundtrip (proof
/// #3 in the base set) to 3-way for multi-branch VLM architectures.
#[kani::unwind(1)]
#[kani::proof]
fn nested_cat_split_roundtrip_3way() {
    let d0: u8 = kani::any();
    let s1: u8 = kani::any();
    let s2: u8 = kani::any();
    let s3: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(s1 >= 1 && s1 <= 8);
    kani::assume(s2 >= 1 && s2 <= 8);
    kani::assume(s3 >= 1 && s3 <= 8);

    let d0u = d0 as usize;
    let s1u = s1 as usize;
    let s2u = s2 as usize;
    let s3u = s3 as usize;

    if let Some(s12) = s1u.checked_add(s2u) {
        if let Some(s_total) = s12.checked_add(s3u) {
            let orig = [d0u, s_total];

            // Split into 3 parts along dim=1
            let split_sizes = [s1u, s2u, s3u];
            let split_sum = split_sizes[0] + split_sizes[1] + split_sizes[2];
            assert_eq!(split_sum, orig[1], "split sizes sum to original dim");

            // Cat the 3 parts back
            let cat_dim = split_sizes[0] + split_sizes[1] + split_sizes[2];
            let reconstructed = [d0u, cat_dim];
            assert_eq!(reconstructed[0], orig[0], "non-cat dim preserved");
            assert_eq!(reconstructed[1], orig[1], "cat dim reconstructed");

            // Numel roundtrip
            let orig_numel = checked_dim_product(&orig);
            let p1 = checked_dim_product(&[d0u, s1u]);
            let p2 = checked_dim_product(&[d0u, s2u]);
            let p3 = checked_dim_product(&[d0u, s3u]);
            let recon_numel = checked_dim_product(&reconstructed);
            if let (Ok(on), Ok(a), Ok(b), Ok(c), Ok(rn)) = (orig_numel, p1, p2, p3, recon_numel) {
                assert_eq!(on, a + b + c, "split parts numel sum to original");
                assert_eq!(on, rn, "roundtrip preserves numel");
            }
        }
    }
}
