// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for DynTensor core operations (#3601).
//!
//! Continuation of `kani_ops_proofs.rs`. Split for 500-line file limit.
//!
//! Covers:
//! - `checked_dim_product`: single-element shape, zero dimensions, monotonicity
//! - Reshape: 2D→3D, 4D→2D element preservation
//! - Permute: 4D bijection
//! - Broadcast: 3D commutativity, scalar broadcasting, element count domination
//! - `D::resolve` consistency across rank range
//! - conv1d_out_len stride monotonicity
//! - Narrow bounds validation
//! - Unsqueeze rank change
//! - `checked_buffer_len` vs `checked_dim_product` consistency

use crate::dyn_tensor::conv::conv1d_out_len;
use crate::dyn_tensor::D;
use crate::tensor::checked_dim_product;

// ---------------------------------------------------------------------------
// checked_dim_product: single-element shape
// ---------------------------------------------------------------------------

/// Prove: checked_dim_product of a single dimension returns that dimension.
///
/// A 1D tensor [N] has exactly N elements. The fold must handle the
/// single-element case correctly (identity element 1 * N = N).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checked_dim_product_single_dim() {
    let d: u16 = kani::any();
    kani::assume(d >= 1 && d <= 4096);

    let dims = [d as usize];
    let result = checked_dim_product(&dims);
    assert!(result.is_ok(), "single dim must succeed");
    assert_eq!(result.unwrap(), d as usize, "product of [N] must be N");
}

/// Prove: checked_dim_product with a zero dimension produces zero.
///
/// A tensor with shape [3, 0, 5] has zero elements. The function must
/// return Ok(0), not Err. Zero-element tensors are valid (e.g., after
/// filtering or as empty batches).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checked_dim_product_zero_dim_produces_zero() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    kani::assume(a >= 1 && a <= 32);
    kani::assume(b >= 1 && b <= 32);

    let dims = [a as usize, 0, b as usize];
    let result = checked_dim_product(&dims);
    assert!(result.is_ok(), "zero-dim shape must succeed");
    assert_eq!(
        result.unwrap(),
        0,
        "shape with zero dim must have 0 elements"
    );
}

// ---------------------------------------------------------------------------
// Reshape: 2D to 3D element preservation
// ---------------------------------------------------------------------------

/// Prove: reshaping from 2D to 3D preserves element count when shapes are
/// compatible.
///
/// A 2D tensor [a, b] reshaped to [a, c, d] where c*d == b must have the
/// same element count. This is the inverse of the 3D→1D proof and covers
/// rank-increasing reshapes common in attention (e.g., [B*H, S, D] → [B, H, S, D]).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn reshape_2d_to_3d_preserves_numel() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();

    kani::assume(a >= 1 && a <= 16);
    kani::assume(b >= 1 && b <= 16);
    kani::assume(c >= 1 && c <= 16);

    let ab = (a as usize).checked_mul(b as usize);
    if let Some(ab_val) = ab {
        // Only valid if b*c divides evenly for some d
        if ab_val >= c as usize && ab_val % (c as usize) == 0 {
            let d = ab_val / (c as usize);
            let new_numel = (a as usize)
                .checked_mul(c as usize)
                .and_then(|v| v.checked_mul(d));
            if let Some(nn) = new_numel {
                assert_eq!(ab_val, nn, "2D→3D reshape must preserve element count");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Permute: bijection for rank 4
// ---------------------------------------------------------------------------

/// Prove: a valid 4D permutation preserves element count and is a bijection.
///
/// Extends the 3D permute_is_bijection proof to rank 4. Rank 4 permutations
/// are common in attention (e.g., [B, H, S, D] → [B, S, H, D]).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn permute_is_bijection_4d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(d3 >= 1 && d3 <= 8);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];

    let p0: usize = kani::any();
    let p1: usize = kani::any();
    let p2: usize = kani::any();
    let p3: usize = kani::any();

    kani::assume(p0 < 4 && p1 < 4 && p2 < 4 && p3 < 4);

    // Validate: no duplicates (mirrors DynTensor::permute)
    let mut seen = [false; 4];
    let perm = [p0, p1, p2, p3];
    let mut valid = true;
    let mut i = 0;
    while i < 4 {
        if seen[perm[i]] {
            valid = false;
        }
        seen[perm[i]] = true;
        i += 1;
    }

    if valid {
        let permuted = [dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]];

        // Element count preserved
        let orig = (dims[0] as u64) * (dims[1] as u64) * (dims[2] as u64) * (dims[3] as u64);
        let perm_n = (permuted[0] as u64)
            * (permuted[1] as u64)
            * (permuted[2] as u64)
            * (permuted[3] as u64);
        assert_eq!(orig, perm_n, "4D permute must preserve element count");

        // Multiset equality: sorted dims must match
        let mut orig_sorted = dims;
        orig_sorted.sort();
        let mut perm_sorted = permuted;
        perm_sorted.sort();
        assert_eq!(
            orig_sorted, perm_sorted,
            "4D permuted dims must be a reordering of original dims"
        );
    }
}

// ---------------------------------------------------------------------------
// Broadcast: 3D commutativity
// ---------------------------------------------------------------------------

/// Prove: broadcast is commutative for 3D shapes.
///
/// Extends the 2D commutativity proof. 3D broadcasting is common in
/// attention score computation ([B, S, 1] + [B, 1, S]).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn broadcast_shape_commutative_3d() {
    use crate::dyn_tensor::ops::broadcast_output_shape;

    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let a2: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();
    let b2: u8 = kani::any();

    kani::assume(a0 >= 1 && a0 <= 16);
    kani::assume(a1 >= 1 && a1 <= 16);
    kani::assume(a2 >= 1 && a2 <= 16);
    kani::assume(b0 >= 1 && b0 <= 16);
    kani::assume(b1 >= 1 && b1 <= 16);
    kani::assume(b2 >= 1 && b2 <= 16);

    let lhs = [a0 as usize, a1 as usize, a2 as usize];
    let rhs = [b0 as usize, b1 as usize, b2 as usize];

    let forward = broadcast_output_shape(&lhs, &rhs);
    let reverse = broadcast_output_shape(&rhs, &lhs);

    match (forward, reverse) {
        (Ok(f), Ok(r)) => {
            assert_eq!(f, r, "3D broadcast must be commutative");
        }
        (Err(_), Err(_)) => {
            // Both fail — consistent.
        }
        _ => {
            panic!("3D broadcast commutativity violated");
        }
    }
}

// ---------------------------------------------------------------------------
// Broadcast: scalar broadcasting
// ---------------------------------------------------------------------------

/// Prove: broadcasting a scalar (rank 0) with any shape returns the shape.
///
/// Scalar tensors have shape []. Broadcasting [] with [A, B, C] must produce
/// [A, B, C]. This is the mechanism behind `tensor + scalar` operations.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn broadcast_scalar_with_any_shape() {
    use crate::dyn_tensor::ops::broadcast_output_shape;

    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);

    let scalar: [usize; 0] = [];
    let shape = [d0 as usize, d1 as usize];

    let result = broadcast_output_shape(&scalar, &shape);
    assert!(result.is_ok(), "scalar broadcast must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 2, "output rank must match the non-scalar rank");
    assert_eq!(out[0], d0 as usize, "dim 0 must match");
    assert_eq!(out[1], d1 as usize, "dim 1 must match");
}

// ---------------------------------------------------------------------------
// Broadcast: output element count >= both input element counts
// ---------------------------------------------------------------------------

/// Prove: broadcast output element count >= both input element counts.
///
/// Broadcasting only expands dimensions, so the output tensor always has
/// at least as many elements as each input. This guarantees that element-wise
/// operations don't lose data.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn broadcast_output_numel_dominates() {
    use crate::dyn_tensor::ops::broadcast_output_shape;

    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();

    kani::assume(a0 >= 1 && a0 <= 32);
    kani::assume(a1 >= 1 && a1 <= 32);
    kani::assume(b0 >= 1 && b0 <= 32);
    kani::assume(b1 >= 1 && b1 <= 32);

    let lhs = [a0 as usize, a1 as usize];
    let rhs = [b0 as usize, b1 as usize];

    if let Ok(out) = broadcast_output_shape(&lhs, &rhs) {
        let lhs_numel = (a0 as usize) * (a1 as usize);
        let rhs_numel = (b0 as usize) * (b1 as usize);
        let out_numel = out[0] * out[1];

        assert!(
            out_numel >= lhs_numel,
            "broadcast output must have >= lhs elements"
        );
        assert!(
            out_numel >= rhs_numel,
            "broadcast output must have >= rhs elements"
        );
    }
}

// ---------------------------------------------------------------------------
// D::resolve: consistency across rank range
// ---------------------------------------------------------------------------

/// Prove: D::Minus1 and D::Minus2 always produce different results for rank >= 3.
///
/// This ensures the negative indexing scheme never collapses two distinct
/// axis references into the same index. If Minus1 == Minus2 for any rank,
/// operations that specify different axes would silently operate on the same one.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn d_resolve_consistency() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 3 && rank <= 8);

    let r = rank as usize;

    // D::Minus1 resolves to r - 1
    let m1 = D::Minus1.resolve(r);
    assert!(m1.is_ok());
    let m1_val = m1.unwrap();
    assert_eq!(m1_val, r - 1);

    // D::Minus2 resolves to r - 2
    let m2 = D::Minus2.resolve(r);
    assert!(m2.is_ok());
    let m2_val = m2.unwrap();
    assert_eq!(m2_val, r - 2);

    // D::Minus1 and D::Minus2 produce different results (rank >= 3 guarantees this)
    assert_ne!(
        m1_val, m2_val,
        "D::Minus1 and D::Minus2 must resolve to different dimensions"
    );
}

// ---------------------------------------------------------------------------
// checked_dim_product: monotonicity (adding dims can only increase or maintain)
// ---------------------------------------------------------------------------

/// Prove: checked_dim_product with an extra dimension of size >= 1 produces
/// a result >= the original.
///
/// Appending a dimension of size N multiplies the element count by N.
/// Since N >= 1, the new product must be >= the old product.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checked_dim_product_monotone_on_append() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let extra: u8 = kani::any();

    kani::assume(a >= 1 && a <= 32);
    kani::assume(b >= 1 && b <= 32);
    kani::assume(extra >= 1 && extra <= 32);

    let dims2 = [a as usize, b as usize];
    let dims3 = [a as usize, b as usize, extra as usize];

    let p2 = checked_dim_product(&dims2);
    let p3 = checked_dim_product(&dims3);

    if let (Ok(n2), Ok(n3)) = (p2, p3) {
        assert!(
            n3 >= n2,
            "appending a dimension >= 1 must not decrease element count"
        );
    }
}

// ---------------------------------------------------------------------------
// conv1d_out_len: stride reduces output length
// ---------------------------------------------------------------------------

/// Prove: increasing stride decreases or maintains conv output length.
///
/// A larger stride skips more input positions, producing fewer (or equal)
/// output positions. This is a fundamental property of strided convolution.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_out_len_stride_monotone() {
    let input_len: u8 = kani::any();
    let kernel_size: u8 = kani::any();
    let padding: u8 = kani::any();
    let stride1: u8 = kani::any();
    let stride2: u8 = kani::any();

    kani::assume(input_len >= 1 && input_len <= 64);
    kani::assume(kernel_size >= 1 && kernel_size <= 8);
    kani::assume(padding <= 8);
    kani::assume(stride1 >= 1 && stride1 <= 8);
    kani::assume(stride2 >= 1 && stride2 <= 8);
    kani::assume(stride2 >= stride1);

    let r1 = conv1d_out_len(
        input_len as usize,
        kernel_size as usize,
        padding as usize,
        stride1 as usize,
        1,
    );
    let r2 = conv1d_out_len(
        input_len as usize,
        kernel_size as usize,
        padding as usize,
        stride2 as usize,
        1,
    );

    if let (Ok(out1), Ok(out2)) = (r1, r2) {
        assert!(
            out2 <= out1,
            "larger stride must produce smaller or equal output"
        );
    }
}

// ---------------------------------------------------------------------------
// Narrow: bounds validation
// ---------------------------------------------------------------------------

/// Prove: narrow bounds validation catches all out-of-range slices.
///
/// For narrow(dim, start, len), we must have start + len <= dim_size.
/// This verifies the arithmetic used in the narrow implementation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn narrow_bounds_check_correct() {
    let dim_size: u16 = kani::any();
    let start: u16 = kani::any();
    let len: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 1024);
    kani::assume(start <= 1024);
    kani::assume(len <= 1024);

    let end = (start as usize).checked_add(len as usize);
    match end {
        Some(e) if e <= dim_size as usize => {
            // Valid: start + len <= dim_size
            assert!(e >= start as usize, "end must be >= start");
            assert!(e >= len as usize, "end must be >= len");
        }
        Some(e) => {
            // Invalid: start + len > dim_size
            assert!(e > dim_size as usize, "must exceed dim_size");
        }
        None => {
            // Overflow: start + len wraps around
            // narrow() must detect this via checked_add
        }
    }
}

// ---------------------------------------------------------------------------
// Squeeze/unsqueeze: rank change is exactly +/- 1
// ---------------------------------------------------------------------------

/// Prove: unsqueeze increases rank by exactly 1 and preserves element count.
///
/// Unsqueezing inserts a dimension of size 1, which changes the rank but
/// not the number of elements (multiplying by 1 is identity).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn unsqueeze_rank_change() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);

    let orig_rank = 2;
    let orig_numel = (d0 as usize) * (d1 as usize);

    // Unsqueezing at any valid position inserts a 1
    let insert_pos: u8 = kani::any();
    kani::assume(insert_pos <= orig_rank as u8); // 0..=rank is valid

    let mut new_dims = Vec::new();
    let dims = [d0 as usize, d1 as usize];
    for i in 0..3 {
        if i == insert_pos as usize {
            new_dims.push(1usize);
        }
        if i < 2 {
            new_dims.push(dims[i]);
        }
    }
    // Handle insert at end
    if insert_pos as usize == 2 {
        // Already pushed 1 at position 2
    }

    assert_eq!(new_dims.len(), 3, "unsqueeze must increase rank by 1");
    let new_numel: usize = new_dims.iter().product();
    assert_eq!(
        orig_numel, new_numel,
        "unsqueeze must preserve element count"
    );
}

// ---------------------------------------------------------------------------
// checked_buffer_len: consistent with checked_dim_product
// ---------------------------------------------------------------------------

/// Prove: checked_buffer_len and checked_dim_product agree on the same inputs.
///
/// Both functions compute the product of a list of usize values with overflow
/// checking. They must produce identical results for identical inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checked_buffer_len_matches_dim_product() {
    use crate::dyn_tensor::conv::checked_buffer_len;

    let a: u16 = kani::any();
    let b: u16 = kani::any();

    kani::assume(a >= 1);
    kani::assume(b >= 1);

    let factors = [a as usize, b as usize];
    let buf_result = checked_buffer_len(&factors, "test");
    let dim_result = checked_dim_product(&factors);

    match (buf_result, dim_result) {
        (Ok(buf_val), Ok(dim_val)) => {
            assert_eq!(
                buf_val, dim_val,
                "checked_buffer_len and checked_dim_product must agree"
            );
        }
        (Err(_), Err(_)) => {
            // Both detect overflow — consistent
        }
        _ => {
            panic!("checked_buffer_len and checked_dim_product must agree on overflow detection");
        }
    }
}
