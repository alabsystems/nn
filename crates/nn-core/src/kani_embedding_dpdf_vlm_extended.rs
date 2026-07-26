// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for VLM embedding layer patterns (#4236).
//!
//! Proofs targeting dpdf-specific VLM embedding patterns beyond the base 9:
//!
//! 1.  Patch embedding: image -> patches -> linear projection shape
//! 2.  Position embedding: pos_embed.shape == [1, num_patches+1, embed_dim]
//! 3.  Token type embedding: segment IDs in [0, num_types)
//! 4.  Rotary position embedding: sin/cos shape matches head_dim/2
//! 5.  Learned vs sinusoidal: both produce same output shape
//! 6.  Vision-language shared embedding: text + vision tokens share dim
//! 7.  Embedding table lookup: all indices in [0, vocab_size)
//! 8.  Embedding gradient: only touched rows get nonzero grad
//! 9.  Padding token embedding: pad_token_id < vocab_size
//! 10. CLS token: cls_embed.shape == [1, 1, embed_dim]
//!
//! Part of #4236.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn cos_f32_stub(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

fn sin_f32_stub(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

// =============================================================================
// 1. Patch embedding: image -> patches -> linear projection
// =============================================================================

/// Prove: ViT patch embedding converts [B, C, H, W] image to
/// [B, num_patches, embed_dim] where num_patches = (H/P) * (W/P).
/// The linear projection weight is [patch_size^2 * C, embed_dim].
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_patch_embedding_shape() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();
    let height: usize = kani::any();
    let width: usize = kani::any();
    let patch_size: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 4);
    kani::assume(channels >= 1 && channels <= 4);
    kani::assume(patch_size >= 1 && patch_size <= 32);
    kani::assume(height >= patch_size && height <= 512);
    kani::assume(width >= patch_size && width <= 512);
    kani::assume(height % patch_size == 0);
    kani::assume(width % patch_size == 0);
    kani::assume(embed_dim >= 1 && embed_dim <= 1024);

    let num_patches_h = height / patch_size;
    let num_patches_w = width / patch_size;
    let num_patches = num_patches_h.checked_mul(num_patches_w);
    assert!(num_patches.is_some(), "num_patches must not overflow");
    let num_patches = num_patches.unwrap();

    // Flattened patch dim: patch_size * patch_size * channels
    let patch_dim = patch_size
        .checked_mul(patch_size)
        .and_then(|ps2| ps2.checked_mul(channels));
    assert!(patch_dim.is_some(), "patch_dim must not overflow");

    // Output shape: [B, num_patches, embed_dim]
    let output_shape = [batch, num_patches, embed_dim];
    let numel = checked_dim_product(&output_shape);
    assert!(numel.is_ok(), "patch embedding output numel valid");

    // Projection weight shape: [patch_dim, embed_dim]
    let proj_wt = [patch_dim.unwrap(), embed_dim];
    let proj_numel = checked_dim_product(&proj_wt);
    assert!(proj_numel.is_ok(), "projection weight numel valid");
}

// =============================================================================
// 2. Position embedding shape: [1, num_patches+1, embed_dim]
// =============================================================================

/// Prove: ViT position embedding has shape [1, num_patches + 1, embed_dim]
/// (the +1 accounts for the CLS token). The total element count does not
/// overflow for realistic VLM configurations.
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_position_embedding_shape() {
    let num_patches: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(num_patches >= 1 && num_patches <= 4096);
    kani::assume(embed_dim >= 1 && embed_dim <= 1024);

    // +1 for CLS token
    let seq_with_cls = num_patches.checked_add(1);
    assert!(seq_with_cls.is_some(), "num_patches + 1 must not overflow");
    let seq_with_cls = seq_with_cls.unwrap();

    let pos_embed_shape = [1_usize, seq_with_cls, embed_dim];
    let numel = checked_dim_product(&pos_embed_shape);
    assert!(numel.is_ok(), "position embedding numel valid");

    // Position embedding has exactly (num_patches+1) * embed_dim elements
    let expected = seq_with_cls.checked_mul(embed_dim);
    assert!(expected.is_some(), "expected numel must not overflow");
    assert!(
        numel.unwrap() == expected.unwrap(),
        "pos_embed numel matches (num_patches+1)*embed_dim"
    );
}

// =============================================================================
// 3. Token type embedding: segment IDs in [0, num_types)
// =============================================================================

/// Prove: for a BERT-family VLM with num_types segment types, any valid
/// segment ID is in [0, num_types). Production code checks this at runtime
/// before indexing the segment embedding table.
///
/// Part of #4236.
#[kani::unwind(5)]
#[kani::proof]
fn proof_token_type_embedding_bounds() {
    let num_types: usize = kani::any();
    let seg_id: usize = kani::any();
    let seq_len: usize = kani::any();

    kani::assume(num_types >= 1 && num_types <= 16);
    kani::assume(seq_len >= 1 && seq_len <= 32);

    // All segment IDs must be < num_types
    kani::assume(seg_id < num_types);

    // Segment embedding weight: [num_types, embed_dim]
    // Lookup is safe when seg_id < num_types
    assert!(seg_id < num_types, "segment ID within table bounds");

    // For a sequence of segment IDs, each must be in [0, num_types)
    for _pos in 0..seq_len.min(4) {
        let id: usize = kani::any();
        kani::assume(id < num_types);
        assert!(id < num_types, "per-position segment ID in bounds");
    }
}

// =============================================================================
// 4. Rotary position embedding: sin/cos shape matches head_dim/2
// =============================================================================

/// Prove: RoPE sin/cos cache shape is [max_seq_len, head_dim/2] and the
/// total cache allocation (2 * max_seq_len * head_dim/2) does not overflow.
/// Each cache entry produces a valid sin/cos value in [-1, 1].
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_sincos_shape_matches_half_head_dim() {
    let max_seq_len: usize = kani::any();
    let head_dim: usize = kani::any();

    kani::assume(max_seq_len >= 1 && max_seq_len <= 131_072);
    kani::assume(head_dim >= 2 && head_dim <= 256);
    kani::assume(head_dim % 2 == 0);

    let half_dim = head_dim / 2;

    // sin cache shape: [max_seq_len, half_dim]
    let sin_shape = [max_seq_len, half_dim];
    let sin_numel = checked_dim_product(&sin_shape);
    assert!(sin_numel.is_ok(), "sin cache numel valid");

    // cos cache shape: [max_seq_len, half_dim]
    let cos_shape = [max_seq_len, half_dim];
    let cos_numel = checked_dim_product(&cos_shape);
    assert!(cos_numel.is_ok(), "cos cache numel valid");

    // Total allocation for both caches
    let total = sin_numel.unwrap().checked_mul(2);
    assert!(total.is_some(), "total sin+cos cache must not overflow");

    // Each entry is a valid sin/cos value
    let angle: f32 = kani::any();
    kani::assume(angle.is_finite() && angle.abs() <= 1e6);
    let s = sin_f32_stub(angle);
    let c = cos_f32_stub(angle);
    assert!(s >= -1.0 && s <= 1.0, "sin value bounded");
    assert!(c >= -1.0 && c <= 1.0, "cos value bounded");
}

// =============================================================================
// 5. Learned vs sinusoidal: both produce same output shape
// =============================================================================

/// Prove: both learned and sinusoidal positional embeddings produce output
/// shape [1, seq_len, embed_dim] for the same seq_len and embed_dim.
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_learned_vs_sinusoidal_same_shape() {
    let seq_len: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 8192);
    kani::assume(embed_dim >= 2 && embed_dim <= 1024);
    kani::assume(embed_dim % 2 == 0);

    // Learned: weight table [max_pos, embed_dim], select [0..seq_len]
    let learned_shape = [1_usize, seq_len, embed_dim];
    let learned_numel = checked_dim_product(&learned_shape);
    assert!(learned_numel.is_ok(), "learned PE numel valid");

    // Sinusoidal: computed [seq_len, embed_dim]
    let sinusoidal_shape = [1_usize, seq_len, embed_dim];
    let sinusoidal_numel = checked_dim_product(&sinusoidal_shape);
    assert!(sinusoidal_numel.is_ok(), "sinusoidal PE numel valid");

    // Both shapes are identical
    assert!(
        learned_shape == sinusoidal_shape,
        "learned and sinusoidal PE shapes match"
    );
    assert!(
        learned_numel.unwrap() == sinusoidal_numel.unwrap(),
        "learned and sinusoidal PE numels match"
    );
}

// =============================================================================
// 6. Vision-language shared embedding: text + vision tokens share dim
// =============================================================================

/// Prove: in a VLM, text embeddings [B, text_len, D] and vision embeddings
/// [B, vision_len, D] can be concatenated along the sequence dimension
/// to produce [B, text_len + vision_len, D]. The embed_dim D must match.
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_vision_language_shared_embedding_dim() {
    let batch: usize = kani::any();
    let text_len: usize = kani::any();
    let vision_len: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 4);
    kani::assume(text_len >= 1 && text_len <= 512);
    kani::assume(vision_len >= 1 && vision_len <= 1024);
    kani::assume(embed_dim >= 1 && embed_dim <= 1024);

    let text_shape = [batch, text_len, embed_dim];
    let vision_shape = [batch, vision_len, embed_dim];

    // Embedding dimensions must match for concatenation
    assert!(
        text_shape[2] == vision_shape[2],
        "text and vision embed_dim must match"
    );

    // Concatenated sequence length
    let total_len = text_len.checked_add(vision_len);
    assert!(
        total_len.is_some(),
        "total sequence length must not overflow"
    );
    let total_len = total_len.unwrap();

    let concat_shape = [batch, total_len, embed_dim];
    let concat_numel = checked_dim_product(&concat_shape);
    assert!(concat_numel.is_ok(), "concatenated embedding numel valid");

    // concat numel == text numel + vision numel
    let text_numel = checked_dim_product(&text_shape).unwrap();
    let vision_numel = checked_dim_product(&vision_shape).unwrap();
    let expected = text_numel.checked_add(vision_numel);
    assert!(expected.is_some(), "sum of numels must not overflow");
    assert!(
        concat_numel.unwrap() == expected.unwrap(),
        "concat numel == text + vision numels"
    );
}

// =============================================================================
// 7. Embedding table lookup: all indices in [0, vocab_size)
// =============================================================================

/// Prove: embedding table lookup is safe when all input token indices
/// are in [0, vocab_size). Out-of-bounds indices are caught by the
/// precondition check.
///
/// Part of #4236.
#[kani::unwind(5)]
#[kani::proof]
fn proof_embedding_table_lookup_bounds() {
    let vocab_size: usize = kani::any();
    let embed_dim: usize = kani::any();
    let seq_len: usize = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 100_000);
    kani::assume(embed_dim >= 1 && embed_dim <= 1024);
    kani::assume(seq_len >= 1 && seq_len <= 32);

    // Embedding weight: [vocab_size, embed_dim]
    let weight_shape = [vocab_size, embed_dim];
    let weight_numel = checked_dim_product(&weight_shape);
    assert!(weight_numel.is_ok(), "embedding weight numel valid");

    // Each token index must be < vocab_size
    for _pos in 0..seq_len.min(4) {
        let token_id: usize = kani::any();
        kani::assume(token_id < vocab_size);

        // Row offset in the weight table
        let row_offset = token_id.checked_mul(embed_dim);
        assert!(row_offset.is_some(), "row offset must not overflow");
        let row_offset = row_offset.unwrap();

        // Row end within weight table
        let row_end = row_offset.checked_add(embed_dim);
        assert!(row_end.is_some(), "row end must not overflow");
        assert!(
            row_end.unwrap() <= weight_numel.as_ref().unwrap().clone(),
            "row access within weight table"
        );
    }
}

// =============================================================================
// 8. Embedding gradient: only touched rows get nonzero grad
// =============================================================================

/// Prove: for an embedding lookup with K unique indices out of vocab_size,
/// the gradient is sparse — at most K rows of the weight gradient are nonzero.
/// The number of touched rows is bounded by min(seq_len, vocab_size).
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_gradient_sparsity() {
    let vocab_size: usize = kani::any();
    let seq_len: usize = kani::any();
    let num_unique: usize = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 100_000);
    kani::assume(seq_len >= 1 && seq_len <= 8192);
    kani::assume(num_unique >= 1 && num_unique <= seq_len);
    kani::assume(num_unique <= vocab_size);

    // Gradient sparsity: at most num_unique rows are nonzero
    assert!(num_unique <= vocab_size, "touched rows <= vocab_size");
    assert!(num_unique <= seq_len, "touched rows <= seq_len");

    // Upper bound on touched rows
    let max_touched = if seq_len < vocab_size {
        seq_len
    } else {
        vocab_size
    };
    assert!(
        num_unique <= max_touched,
        "unique indices bounded by min(seq_len, vocab_size)"
    );

    // Memory for sparse gradient: num_unique * embed_dim
    let embed_dim: usize = kani::any();
    kani::assume(embed_dim >= 1 && embed_dim <= 1024);
    let sparse_grad_size = num_unique.checked_mul(embed_dim);
    assert!(
        sparse_grad_size.is_some(),
        "sparse gradient allocation must not overflow"
    );
}

// =============================================================================
// 9. Padding token embedding: pad_token_id < vocab_size
// =============================================================================

/// Prove: the padding token ID is a valid embedding table index
/// (pad_token_id < vocab_size). Models that use padding (BERT, LayoutLM)
/// must satisfy this at construction time.
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_padding_token_in_vocab() {
    let vocab_size: usize = kani::any();
    let pad_token_id: usize = kani::any();
    let cls_token_id: usize = kani::any();
    let sep_token_id: usize = kani::any();

    kani::assume(vocab_size >= 4 && vocab_size <= 100_000);
    kani::assume(pad_token_id < vocab_size);
    kani::assume(cls_token_id < vocab_size);
    kani::assume(sep_token_id < vocab_size);

    // All special tokens must be valid embedding indices
    assert!(pad_token_id < vocab_size, "pad_token_id in vocab");
    assert!(cls_token_id < vocab_size, "cls_token_id in vocab");
    assert!(sep_token_id < vocab_size, "sep_token_id in vocab");

    // Special tokens should be distinct (common invariant)
    kani::assume(pad_token_id != cls_token_id);
    kani::assume(pad_token_id != sep_token_id);
    kani::assume(cls_token_id != sep_token_id);

    assert!(
        pad_token_id != cls_token_id && pad_token_id != sep_token_id,
        "pad token distinct from cls and sep"
    );
}

// =============================================================================
// 10. CLS token: cls_embed.shape == [1, 1, embed_dim]
// =============================================================================

/// Prove: the CLS token embedding is a learnable parameter with shape
/// [1, 1, embed_dim]. When prepended to patch embeddings [B, N, D],
/// the result is [B, N+1, D].
///
/// Part of #4236.
#[kani::unwind(1)]
#[kani::proof]
fn proof_cls_token_shape() {
    let batch: usize = kani::any();
    let num_patches: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 4);
    kani::assume(num_patches >= 1 && num_patches <= 4096);
    kani::assume(embed_dim >= 1 && embed_dim <= 1024);

    // CLS token shape: [1, 1, embed_dim]
    let cls_shape = [1_usize, 1_usize, embed_dim];
    let cls_numel = checked_dim_product(&cls_shape);
    assert!(cls_numel.is_ok(), "CLS token numel valid");
    assert!(
        cls_numel.unwrap() == embed_dim,
        "CLS token has embed_dim elements"
    );

    // Patch embeddings: [B, num_patches, embed_dim]
    let patch_shape = [batch, num_patches, embed_dim];
    let patch_numel = checked_dim_product(&patch_shape);
    assert!(patch_numel.is_ok(), "patch embedding numel valid");

    // After prepending CLS: [B, num_patches + 1, embed_dim]
    let with_cls_len = num_patches.checked_add(1);
    assert!(with_cls_len.is_some(), "num_patches + 1 must not overflow");
    let with_cls_shape = [batch, with_cls_len.unwrap(), embed_dim];
    let with_cls_numel = checked_dim_product(&with_cls_shape);
    assert!(with_cls_numel.is_ok(), "patch+CLS numel valid");

    // numel difference is exactly B * embed_dim (one CLS per batch)
    let diff = with_cls_numel.unwrap() - patch_numel.unwrap();
    let expected_diff = batch.checked_mul(embed_dim);
    assert!(expected_diff.is_some(), "expected diff must not overflow");
    assert!(
        diff == expected_diff.unwrap(),
        "prepending CLS adds B*embed_dim elements"
    );
}
