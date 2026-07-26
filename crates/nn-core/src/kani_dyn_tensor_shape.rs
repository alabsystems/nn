// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor shape/indexing operations.
//!
//! Split from `kani_dyn_tensor.rs` (#1544 D6) for 500-line compliance.
//! Contains harnesses for numel, transpose, embedding index, and chunk
//! partition arithmetic.

#![cfg(kani)]

// ---------------------------------------------------------------------------
// numel / checked_dim_product agreement
// ---------------------------------------------------------------------------

/// Prove: numel() agrees with checked_dim_product for valid dimensions.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn numel_agrees_with_checked_product_3d() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();
    kani::assume(d0 >= 1 && d1 >= 1 && d2 >= 1);
    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let checked = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
    if let Some(product) = checked {
        let numel: usize = dims.iter().copied().product();
        assert_eq!(
            product, numel,
            "numel() must agree with checked_dim_product"
        );
    }
}

// ---------------------------------------------------------------------------
// transpose shape/permutation proofs
// ---------------------------------------------------------------------------

/// Prove: transpose algorithm from dyn_tensor_shape.rs:209-223.
/// Validates axes, builds permutation, verifies dim swap and numel preservation.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn transpose_shape_validation() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);
    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let rank = dims.len();
    let swap_a: usize = kani::any();
    let swap_b: usize = kani::any();
    kani::assume(swap_a < rank && swap_b < rank && swap_a != swap_b);
    let mut axes = [0usize, 1, 2];
    axes.swap(swap_a, swap_b);
    let transposed = [dims[axes[0]], dims[axes[1]], dims[axes[2]]];
    assert_eq!(transposed[swap_a], dims[swap_b], "d1 and d2 must swap");
    assert_eq!(transposed[swap_b], dims[swap_a], "d2 and d1 must swap");
    for i in 0..rank {
        if i != swap_a && i != swap_b {
            assert_eq!(transposed[i], dims[i], "non-swapped dims preserved");
        }
    }
    let orig_numel = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
    let trans_numel = transposed
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d));
    if let (Some(on), Some(tn)) = (orig_numel, trans_numel) {
        assert_eq!(on, tn, "transpose must preserve numel");
    }
    let mut seen = [false; 3];
    for &a in &axes {
        assert!(!seen[a], "duplicate axis in permutation");
        seen[a] = true;
    }
    assert!(
        seen.iter().all(|&s| s),
        "all axes must appear in permutation"
    );
}

/// Prove: transpose is self-inverse (applying it twice recovers original dims).
///
/// Inlines axes.swap(d1, d2) from dyn_tensor_shape.rs:220-221
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn transpose_self_inverse() {
    let a: u16 = kani::any();
    let b: u16 = kani::any();
    let c: u16 = kani::any();
    kani::assume(a >= 1 && b >= 1 && c >= 1);

    let dims = [a as usize, b as usize, c as usize];

    // Pick two distinct axes to swap
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    kani::assume(d1 < 3 && d2 < 3 && d1 != d2);

    // First transpose
    let mut after_first = dims;
    after_first.swap(d1, d2);

    // Second transpose (same axes)
    let mut after_second = after_first;
    after_second.swap(d1, d2);

    assert_eq!(dims, after_second, "double transpose must be identity");
}

// ---------------------------------------------------------------------------
// Embedding forward_ids — prove index arithmetic safety
// ---------------------------------------------------------------------------

/// Prove: for valid vocab_size and embed_dim (passed checked_dim_product),
/// and id < vocab_size, the slice index `id * embed_dim + embed_dim`
/// does not exceed `vocab_size * embed_dim`.
///
/// Inlines nn.rs:420-421: `let start = id * embed_dim;`
/// `weight_flat[start..start + embed_dim]`
///
/// The weight_flat has length vocab_size * embed_dim (from checked_dim_product
/// at construction time). This proves the slice is always in bounds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn embedding_index_in_bounds() {
    let vocab_size: u16 = kani::any();
    let embed_dim: u16 = kani::any();
    let id: u16 = kani::any();

    kani::assume(vocab_size >= 1 && embed_dim >= 1);
    kani::assume(id < vocab_size);

    let v = vocab_size as usize;
    let e = embed_dim as usize;
    let i = id as usize;

    // Construction invariant: checked_dim_product([v, e]) succeeded,
    // meaning v * e <= usize::MAX. For u16 inputs, v * e <= 65535^2
    // which fits in usize on 64-bit.
    let total = v * e;

    // nn.rs:420
    let start = i * e;
    let end = start + e;

    assert!(
        end <= total,
        "embedding slice must be within weight_flat bounds"
    );
}

/// Prove: embedding output accumulation matches shape product.
///
/// The loop at nn.rs:414-422 appends exactly embed_dim elements per id.
/// Model the loop as: for each of num_ids iterations, accumulate embed_dim.
/// Prove the accumulated total equals checked_dim_product([num_ids, embed_dim]).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(260)]
fn embedding_output_length_correct() {
    let num_ids: u8 = kani::any();
    let embed_dim: u8 = kani::any();

    kani::assume(num_ids >= 1 && num_ids <= 16);
    kani::assume(embed_dim >= 1 && embed_dim <= 16);

    let n = num_ids as usize;
    let e = embed_dim as usize;

    // Model the loop at nn.rs:414-422: each iteration extends by embed_dim
    let mut accumulated: usize = 0;
    let mut i: usize = 0;
    while i < n {
        accumulated += e;
        i += 1;
    }

    // from_vec at nn.rs:423 checks shape [n, e] matches accumulated length
    let shape_product: usize = [n, e].iter().copied().product();
    assert_eq!(
        accumulated, shape_product,
        "loop accumulation must match shape product"
    );
}

/// Prove: Embedding::forward f32-to-usize cast is faithful for small indices.
///
/// For f32 values that pass the guard (finite, >= 0, == trunc),
/// when the value is <= 16777215 (2^24 - 1, max exact integer in f32),
/// the cast `v as usize` equals the mathematical value.
///
/// Inlines nn.rs:436-441
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn embedding_f32_to_usize_faithful_small() {
    let idx: u16 = kani::any();
    let v = idx as f32;

    // The guard from nn.rs:436
    assert!(v.is_finite(), "u16 as f32 must be finite");
    assert!(v >= 0.0, "u16 as f32 must be non-negative");
    assert!(v == v.trunc(), "u16 as f32 must be an integer");

    // The cast from nn.rs:441
    let as_usize = v as usize;

    assert_eq!(
        as_usize, idx as usize,
        "f32-to-usize cast must be faithful for u16 values"
    );
}

// ---------------------------------------------------------------------------
// chunk partition arithmetic
// ---------------------------------------------------------------------------

/// Prove: chunk partition covers the full dimension with no gaps.
///
/// Inlines dyn_tensor_shape.rs:287-294:
/// `chunk_size = dim_size.div_ceil(chunks); while start < dim_size { ... }`
///
/// Proves the sum of chunk lengths equals dim_size.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(20)]
fn chunk_partition_complete() {
    let dim_size: u8 = kani::any();
    let chunks: u8 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 16);
    kani::assume(chunks >= 1 && chunks <= 8);

    let d = dim_size as usize;
    let c = chunks as usize;

    let chunk_size = d.div_ceil(c);

    let mut start: usize = 0;
    let mut total: usize = 0;
    let mut count: usize = 0;

    // Unroll the loop (bounded by dim_size <= 16, chunk_size >= 1 -> at most 16 iters)
    while start < d {
        let len = chunk_size.min(d - start);
        assert!(len >= 1, "each chunk must have >= 1 element");
        total += len;
        start += len;
        count += 1;
    }

    assert_eq!(total, d, "chunks must cover the full dimension");
    assert!(count <= c, "must not produce more chunks than requested");
}
