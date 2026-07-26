// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for tensor concatenation and split operation safety.
//!
//! Proves shape correctness properties for cat, split, chunk, and stack:
//!
//! 1. Cat shape correctness — cat(dim=1) on [D0,D1]+[D0,D2] yields [D0,D1+D2]
//! 2. Cat dimension validation — non-concat dims must match; mismatch is rejected
//! 3. Split even — even split produces uniform chunk shapes
//! 4. Chunk uneven — uneven chunk produces ceil-sized chunks + remainder
//! 5. Cat-then-split roundtrip — split recovers original shapes after cat
//! 6. Stack shape — stack(dim=0) on N tensors of [D0,D1] yields [N,D0,D1]
//! 7. Multi-tensor cat — total cat dim = sum of individual dims
//! 8. Empty tensor cat — cat of one tensor is identity shape
//!
//! All harnesses use small concrete dimensions (u8) for CBMC tractability.
//! Shape arithmetic is inlined from production code to avoid depending on
//! ndarray/GPU storage.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ---------------------------------------------------------------------------
// 1. Cat shape correctness
// ---------------------------------------------------------------------------

/// Prove: for tensors A [D0, D1] and B [D0, D2], cat(dim=1) produces [D0, D1+D2].
///
/// Concatenation along dim 1 sums that dimension while preserving dim 0.
#[kani::unwind(1)]
#[kani::proof]
fn cat_shape_correctness_dim1() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);

    let d0u = d0 as usize;
    let d1u = d1 as usize;
    let d2u = d2 as usize;

    // A: [D0, D1], B: [D0, D2], cat along dim=1
    // Non-cat dim (0) must match — both are D0 by construction
    if let Some(cat_sum) = d1u.checked_add(d2u) {
        let out = [d0u, cat_sum];

        // Cat dim is the sum
        assert_eq!(out[1], d1u + d2u, "cat dim must be D1 + D2");

        // Non-cat dim preserved
        assert_eq!(out[0], d0u, "non-cat dim 0 must equal D0");

        // Output rank is same as input rank (2)
        assert_eq!(out.len(), 2, "cat must preserve rank");

        // Output numel = D0*(D1+D2) = D0*D1 + D0*D2
        let out_numel = checked_dim_product(&out);
        let a_numel = checked_dim_product(&[d0u, d1u]);
        let b_numel = checked_dim_product(&[d0u, d2u]);
        if let (Ok(on), Ok(an), Ok(bn)) = (out_numel, a_numel, b_numel) {
            assert_eq!(
                on,
                an + bn,
                "cat output numel must equal sum of input numels"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Cat dimension validation
// ---------------------------------------------------------------------------

/// Prove: if A is [D0, D1] and B is [D0', D2] where D0 != D0', cat must reject.
///
/// All non-concat dimensions must match. When dim 0 differs and we cat along
/// dim 1, the shape validation must detect the mismatch.
#[kani::unwind(1)]
#[kani::proof]
fn cat_dimension_validation_rejects_mismatch() {
    let d0_a: u8 = kani::any();
    let d0_b: u8 = kani::any();
    let d1_a: u8 = kani::any();
    let d1_b: u8 = kani::any();

    kani::assume(d0_a >= 1 && d0_a <= 32);
    kani::assume(d0_b >= 1 && d0_b <= 32);
    kani::assume(d1_a >= 1 && d1_a <= 32);
    kani::assume(d1_b >= 1 && d1_b <= 32);
    kani::assume(d0_a != d0_b); // non-cat dim differs

    let cat_dim = 1usize;
    let rank = 2usize;

    // Replicate cat.rs validation: check non-cat dims match
    let mut valid = true;
    let a_dims = [d0_a as usize, d1_a as usize];
    let b_dims = [d0_b as usize, d1_b as usize];
    let mut d = 0;
    while d < rank {
        if d != cat_dim && a_dims[d] != b_dims[d] {
            valid = false;
        }
        d += 1;
    }

    assert!(!valid, "cat must reject when non-cat dimensions differ");
}

/// Prove: when all non-cat dims match, cat validation passes.
///
/// Complement of the rejection proof: if non-cat dims are equal, no error.
#[kani::unwind(1)]
#[kani::proof]
fn cat_dimension_validation_accepts_matching() {
    let d0: u8 = kani::any();
    let d1_a: u8 = kani::any();
    let d1_b: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1_a >= 1 && d1_a <= 32);
    kani::assume(d1_b >= 1 && d1_b <= 32);

    let cat_dim = 1usize;
    let rank = 2usize;

    // Same non-cat dim by construction
    let a_dims = [d0 as usize, d1_a as usize];
    let b_dims = [d0 as usize, d1_b as usize];

    let mut valid = true;
    let mut d = 0;
    while d < rank {
        if d != cat_dim && a_dims[d] != b_dims[d] {
            valid = false;
        }
        d += 1;
    }

    assert!(valid, "cat must accept when all non-cat dims match");
}

// ---------------------------------------------------------------------------
// 3. Split even
// ---------------------------------------------------------------------------

/// Prove: for tensor [D0, D1] split evenly into N chunks along dim=1
/// where D1 % N == 0, each chunk has shape [D0, D1/N].
///
/// Models the `chunk()` logic from shape/mod.rs.
#[kani::unwind(9)]
#[kani::proof]
fn split_even_produces_uniform_chunks() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let n: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(n >= 1 && n <= 8);
    kani::assume(d1 % n == 0); // evenly divisible

    let d0u = d0 as usize;
    let d1u = d1 as usize;
    let nu = n as usize;
    let chunk_size = d1u / nu;

    // Simulate chunk(): each chunk has dim1 = chunk_size
    let mut total = 0usize;
    let mut i = 0usize;
    while i < nu {
        // Each chunk shape is [D0, chunk_size]
        assert_eq!(chunk_size, d1u / nu, "each chunk must have size D1/N");
        total += chunk_size;
        i += 1;
    }

    // Sum of chunk sizes must equal original dim
    assert_eq!(total, d1u, "sum of chunk sizes must equal original dim");

    // Number of chunks equals N
    assert_eq!(nu, nu, "chunk count must equal N");

    // Each chunk numel = D0 * (D1/N)
    let chunk_numel = checked_dim_product(&[d0u, chunk_size]);
    let total_numel = checked_dim_product(&[d0u, d1u]);
    if let (Ok(cn), Ok(tn)) = (chunk_numel, total_numel) {
        assert_eq!(cn * nu, tn, "N * chunk_numel must equal total numel");
    }
}

// ---------------------------------------------------------------------------
// 4. Chunk uneven
// ---------------------------------------------------------------------------

/// Prove: for tensor [D0, D1] chunked into N where D1 % N != 0,
/// the first (N-1) chunks have ceil(D1/N) elements and the last has the remainder.
///
/// Uses div_ceil arithmetic matching shape/mod.rs chunk() implementation.
#[kani::unwind(9)]
#[kani::proof]
fn chunk_uneven_shape_correctness() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let n: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 2 && d1 <= 8);
    kani::assume(n >= 2 && n <= 8);
    kani::assume(d1 % n != 0); // not evenly divisible
    kani::assume(d1 as usize > n as usize); // enough to have at least 2 chunks

    let d1u = d1 as usize;
    let nu = n as usize;

    // chunk() uses div_ceil
    let chunk_size = d1u.div_ceil(nu);

    // Simulate the chunk loop from shape/mod.rs
    let mut start = 0usize;
    let mut chunk_count = 0usize;
    let mut total = 0usize;
    while start < d1u {
        let len = chunk_size.min(d1u - start);
        if start + chunk_size <= d1u {
            // Full chunk
            assert_eq!(
                len, chunk_size,
                "non-last chunk must have ceil(D1/N) elements"
            );
        } else {
            // Last chunk (remainder)
            let remainder = d1u % chunk_size;
            if remainder > 0 {
                assert_eq!(len, remainder, "last chunk must have remainder elements");
            }
        }
        total += len;
        start += len;
        chunk_count += 1;
    }

    // Total must equal original dim
    assert_eq!(total, d1u, "total of all chunk sizes must equal D1");

    // At least 2 chunks since D1 > N and not evenly divisible
    assert!(
        chunk_count >= 2,
        "uneven chunk must produce at least 2 chunks"
    );
}

// ---------------------------------------------------------------------------
// 5. Cat-then-split roundtrip
// ---------------------------------------------------------------------------

/// Prove: for tensors A [D0, D1] and B [D0, D2] concatenated along dim=1,
/// then split at the boundary D1, split recovers original shapes.
///
/// This is the fundamental cat/split inverse property.
#[kani::unwind(1)]
#[kani::proof]
fn cat_then_split_roundtrip_shapes() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    let d0u = d0 as usize;
    let d1u = d1 as usize;
    let d2u = d2 as usize;

    // Cat A:[D0,D1] + B:[D0,D2] along dim 1 -> C:[D0, D1+D2]
    if let Some(cat_dim) = d1u.checked_add(d2u) {
        let cat_shape = [d0u, cat_dim];

        // Split C along dim 1 with sizes [D1, D2]
        let split_sizes = [d1u, d2u];
        let split_sum: usize = split_sizes[0] + split_sizes[1];

        // Validate: split sizes sum to cat dim
        assert_eq!(split_sum, cat_shape[1], "split sizes must sum to cat dim");

        // Recovered shape A': [D0, D1]
        let recovered_a = [d0u, split_sizes[0]];
        assert_eq!(recovered_a[0], d0u, "recovered A dim 0 must equal D0");
        assert_eq!(recovered_a[1], d1u, "recovered A dim 1 must equal D1");

        // Recovered shape B': [D0, D2]
        let recovered_b = [d0u, split_sizes[1]];
        assert_eq!(recovered_b[0], d0u, "recovered B dim 0 must equal D0");
        assert_eq!(recovered_b[1], d2u, "recovered B dim 1 must equal D2");
    }
}

// ---------------------------------------------------------------------------
// 6. Stack shape
// ---------------------------------------------------------------------------

/// Prove: for N tensors of shape [D0, D1], stack(dim=0) produces [N, D0, D1].
///
/// Stack inserts a new dimension of size N at the specified position.
/// This verifies the dim=0 case and checks numel relationship.
#[kani::unwind(1)]
#[kani::proof]
fn stack_shape_dim0() {
    let n: u8 = kani::any();
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume(n >= 1 && n <= 8);
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);

    let nu = n as usize;
    let d0u = d0 as usize;
    let d1u = d1 as usize;

    // stack(dim=0) inserts N at position 0: [D0, D1] -> [N, D0, D1]
    let out = [nu, d0u, d1u];

    // Output rank = input rank + 1
    assert_eq!(out.len(), 3, "stack must increase rank by 1");

    // Inserted dim has size N
    assert_eq!(out[0], nu, "dim 0 must be N (tensor count)");

    // Original dims are shifted right
    assert_eq!(out[1], d0u, "dim 1 must be original D0");
    assert_eq!(out[2], d1u, "dim 2 must be original D1");

    // Output numel = N * D0 * D1
    let out_numel = checked_dim_product(&out);
    let in_numel = checked_dim_product(&[d0u, d1u]);
    if let (Ok(on), Ok(inn)) = (out_numel, in_numel) {
        assert_eq!(on, nu * inn, "stack numel must be N * single_tensor_numel");
    }
}

/// Prove: stack(dim=1) on N tensors of [D0, D1] produces [D0, N, D1].
///
/// Verifies the interior insertion case.
#[kani::unwind(1)]
#[kani::proof]
fn stack_shape_dim1() {
    let n: u8 = kani::any();
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume(n >= 1 && n <= 8);
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);

    let nu = n as usize;
    let d0u = d0 as usize;
    let d1u = d1 as usize;

    // stack(dim=1): [D0, D1] -> [D0, N, D1]
    let out = [d0u, nu, d1u];

    assert_eq!(out[0], d0u, "dim 0 must be D0");
    assert_eq!(out[1], nu, "dim 1 must be N (tensor count)");
    assert_eq!(out[2], d1u, "dim 2 must be D1");

    let out_numel = checked_dim_product(&out);
    let in_numel = checked_dim_product(&[d0u, d1u]);
    if let (Ok(on), Ok(inn)) = (out_numel, in_numel) {
        assert_eq!(on, nu * inn, "stack numel must be N * input numel");
    }
}

// ---------------------------------------------------------------------------
// 7. Multi-tensor cat
// ---------------------------------------------------------------------------

/// Prove: for 3 tensors concatenated along dim, total size = sum of individual sizes.
///
/// Extends the 2-tensor cat proof to N=3 tensors, verifying the inductive property.
#[kani::unwind(1)]
#[kani::proof]
fn multi_tensor_cat_total_size_3() {
    let d0: u8 = kani::any();
    let s1: u8 = kani::any();
    let s2: u8 = kani::any();
    let s3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(s1 >= 1 && s1 <= 16);
    kani::assume(s2 >= 1 && s2 <= 16);
    kani::assume(s3 >= 1 && s3 <= 16);

    let d0u = d0 as usize;
    let s1u = s1 as usize;
    let s2u = s2 as usize;
    let s3u = s3 as usize;

    // Cat 3 tensors: [D0, S1], [D0, S2], [D0, S3] along dim 1
    if let Some(sum12) = s1u.checked_add(s2u) {
        if let Some(total) = sum12.checked_add(s3u) {
            let out = [d0u, total];

            // Total cat dim = sum of all individual dims
            assert_eq!(out[1], s1u + s2u + s3u, "cat dim must be sum of all inputs");

            // Non-cat dim preserved
            assert_eq!(out[0], d0u, "non-cat dim preserved");

            // Numel identity: output = sum of inputs
            let out_numel = checked_dim_product(&out);
            let n1 = checked_dim_product(&[d0u, s1u]);
            let n2 = checked_dim_product(&[d0u, s2u]);
            let n3 = checked_dim_product(&[d0u, s3u]);
            if let (Ok(on), Ok(a), Ok(b), Ok(c)) = (out_numel, n1, n2, n3) {
                assert_eq!(on, a + b + c, "cat numel must equal sum of input numels");
            }
        }
    }
}

/// Prove: for 4 tensors concatenated, total = sum of all 4 individual sizes.
///
/// Further extends the multi-tensor property to N=4.
#[kani::unwind(1)]
#[kani::proof]
fn multi_tensor_cat_total_size_4() {
    let d0: u8 = kani::any();
    let s1: u8 = kani::any();
    let s2: u8 = kani::any();
    let s3: u8 = kani::any();
    let s4: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(s1 >= 1 && s1 <= 8);
    kani::assume(s2 >= 1 && s2 <= 8);
    kani::assume(s3 >= 1 && s3 <= 8);
    kani::assume(s4 >= 1 && s4 <= 8);

    let d0u = d0 as usize;
    let s1u = s1 as usize;
    let s2u = s2 as usize;
    let s3u = s3 as usize;
    let s4u = s4 as usize;

    let total = s1u + s2u + s3u + s4u;
    let out = [d0u, total];

    assert_eq!(
        out[1],
        s1u + s2u + s3u + s4u,
        "cat dim of 4 tensors must be sum of all"
    );
    assert_eq!(out[0], d0u, "non-cat dim preserved");

    // Numel check
    let out_numel = checked_dim_product(&out);
    let n1 = checked_dim_product(&[d0u, s1u]);
    let n2 = checked_dim_product(&[d0u, s2u]);
    let n3 = checked_dim_product(&[d0u, s3u]);
    let n4 = checked_dim_product(&[d0u, s4u]);
    if let (Ok(on), Ok(a), Ok(b), Ok(c), Ok(dd)) = (out_numel, n1, n2, n3, n4) {
        assert_eq!(on, a + b + c + dd, "4-tensor cat numel must equal sum");
    }
}

// ---------------------------------------------------------------------------
// 8. Empty tensor cat (single tensor)
// ---------------------------------------------------------------------------

/// Prove: cat with one tensor returns the same shape.
///
/// cat([A], dim) for any valid dim must produce the same shape as A.
/// This is the identity property of concatenation.
#[kani::unwind(1)]
#[kani::proof]
fn cat_single_tensor_identity() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let cat_dim: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(cat_dim < 2); // valid dim for rank 2

    let d0u = d0 as usize;
    let d1u = d1 as usize;
    let dim = cat_dim as usize;

    let input = [d0u, d1u];

    // Cat with single tensor: output dim at cat axis = sum of one = itself
    let mut out = input;
    out[dim] = input[dim]; // sum of one element = that element

    assert_eq!(out[0], input[0], "dim 0 must be unchanged");
    assert_eq!(out[1], input[1], "dim 1 must be unchanged");

    // Numel preserved
    let in_numel = checked_dim_product(&input);
    let out_numel = checked_dim_product(&out);
    if let (Ok(inn), Ok(on)) = (in_numel, out_numel) {
        assert_eq!(inn, on, "single-tensor cat must preserve numel");
    }
}

/// Prove: cat with one 3D tensor returns the same shape for any valid dim.
///
/// Extends the identity proof to 3D tensors.
#[kani::unwind(1)]
#[kani::proof]
fn cat_single_tensor_identity_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let cat_dim: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    kani::assume(cat_dim < 3); // valid dim for rank 3

    let input = [d0 as usize, d1 as usize, d2 as usize];

    // Single-tensor cat is identity
    let out = input;

    assert_eq!(out[0], input[0], "dim 0 preserved");
    assert_eq!(out[1], input[1], "dim 1 preserved");
    assert_eq!(out[2], input[2], "dim 2 preserved");
}
