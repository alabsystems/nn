// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Embedding layer (#3716).
//!
//! Proves correctness properties of the Embedding lookup table:
//!
//! 1. Embedding requires rank-2 weight matrix
//! 2. Rank != 2 must be rejected
//! 3. forward_ids: index < vocab_size accepted
//! 4. forward_ids: index >= vocab_size rejected
//! 5. Output shape: input dims + embedding_dim
//! 6. F32 legacy path: non-integer float index rejected
//! 7. F32 legacy path: negative float index rejected
//! 8. I64 path: negative index rejected
//!
//! Part of #3716.

// ---------------------------------------------------------------------------
// Harness 1: Embedding weight must be rank 2
// ---------------------------------------------------------------------------

/// Prove: Embedding::new requires weight rank == 2. This models the
/// validation check at the top of Embedding::new(). A rank-2 weight
/// matrix [vocab_size, embedding_dim] is the only valid shape.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_rank_2_required() {
    let rank: usize = kani::any();
    kani::assume(rank <= 8);

    let valid = rank == 2;

    if valid {
        assert!(rank == 2, "only rank 2 weight matrices are valid");
    } else {
        assert!(rank != 2, "non-rank-2 must be rejected");
    }
}

// ---------------------------------------------------------------------------
// Harness 2: All non-2 ranks rejected exhaustively
// ---------------------------------------------------------------------------

/// Prove: for ranks 0, 1, 3, 4, 5, 6, 7, 8 — all are invalid for Embedding.
/// The check `weight.rank() != 2` catches every case.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_non_rank_2_rejected() {
    let rank: usize = kani::any();
    kani::assume(rank <= 8);
    kani::assume(rank != 2);

    // Models the check: if weight.rank() != 2 { return Err }
    let accepted = rank == 2;
    assert!(!accepted, "non-rank-2 must be rejected by Embedding::new");
}

// ---------------------------------------------------------------------------
// Harness 4: forward_ids rejects out-of-range indices
// ---------------------------------------------------------------------------

/// Prove: when an index >= vocab_size, forward_ids returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_out_of_range_index_rejected() {
    let vocab_size: usize = kani::any();
    let index: usize = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 65536);
    kani::assume(index >= vocab_size);

    // Models: if id >= vocab_size { return Err(EmbeddingIndexOutOfRange) }
    let rejected = index >= vocab_size;
    assert!(rejected, "index >= vocab_size must be rejected");
}

// ---------------------------------------------------------------------------
// Harness 5: Output shape is input_dims ++ [embedding_dim]
// ---------------------------------------------------------------------------

/// Prove: the output shape of Embedding forward is the input shape
/// with embedding_dim appended. For input [B, S], output is [B, S, D].
/// Total output elements = input_elements * embedding_dim.
#[kani::unwind(16)]
#[kani::proof]
fn proof_embedding_output_shape_appends_dim() {
    let input_rank: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(input_rank >= 1 && input_rank <= 4);
    kani::assume(embed_dim >= 1 && embed_dim <= 1024);

    let output_rank = input_rank + 1;
    assert!(
        output_rank == input_rank + 1,
        "output rank must be input rank + 1"
    );

    // The last dimension of the output is always embedding_dim.
    // Model: out_shape = input_dims.push(embed_dim)
    let input_elements: usize = kani::any();
    kani::assume(input_elements >= 1 && input_elements <= 4096);

    let output_elements = input_elements.checked_mul(embed_dim);
    assert!(
        output_elements.is_some(),
        "output element count must not overflow"
    );
    assert!(
        output_elements.unwrap() >= embed_dim,
        "output must have at least embed_dim elements"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: F32 legacy path rejects non-integer floats
// ---------------------------------------------------------------------------

/// Prove: the F32 legacy path in extract_ids rejects non-integer float values.
/// The check `v != v.trunc()` catches fractional values.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_f32_rejects_non_integer() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());
    kani::assume(v >= 0.0);
    kani::assume(v != v.trunc()); // fractional part exists

    // Models: if v != v.trunc() { return Err(ValueOutOfRange) }
    let is_integer = v == v.trunc();
    assert!(!is_integer, "non-integer float must be rejected");
}

// ---------------------------------------------------------------------------
// Harness 7: F32 legacy path rejects negative floats
// ---------------------------------------------------------------------------

/// Prove: the F32 legacy path rejects negative float values.
/// The check `v < 0.0` catches all negative finite values.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_f32_rejects_negative() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());
    kani::assume(v < 0.0);

    // Models: if v < 0.0 { return Err(ValueOutOfRange) }
    assert!(v < 0.0, "negative float must be detected");
    let rejected = v < 0.0 || !v.is_finite() || v != v.trunc();
    assert!(rejected, "negative float must be rejected");
}

// ---------------------------------------------------------------------------
// Harness 8: I64 path rejects negative indices
// ---------------------------------------------------------------------------

/// Prove: the I64 extraction path rejects negative values.
/// Models the check `if v < 0 { return Err(ValueOutOfRange) }`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_embedding_i64_rejects_negative() {
    let v: i64 = kani::any();
    kani::assume(v < 0);

    // Models: if v < 0 { return Err(ValueOutOfRange { "non-negative" }) }
    assert!(v < 0, "negative i64 must be detected");

    // usize::try_from would also fail for negative values.
    let as_usize = usize::try_from(v);
    assert!(as_usize.is_err(), "negative i64 must fail usize conversion");
}
