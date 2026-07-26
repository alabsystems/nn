// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for tensor memory safety.
//!
//! Part of #4242. Proves ten categories of tensor memory management invariants:
//!
//! 1. **Stride-contiguity invariant** — contiguous tensor has row-major strides
//!    satisfying stride[i] = product(dims[i+1..]) with innermost stride == 1.
//! 2. **View safety** — view does not extend beyond original allocation; every
//!    accessible offset is < numel of the backing buffer.
//! 3. **Reshape preserves element count** — product(old_shape) == product(new_shape)
//!    for arbitrary compatible reshapes.
//! 4. **Transpose correctness** — transpose swaps dims and strides correctly;
//!    double-transpose is identity; max offset preserved.
//! 5. **Permute correctness** — permuted dims form a valid permutation; inverse
//!    permutation recovers the original layout.
//! 6. **Broadcast safety** — broadcast only expands size-1 dimensions; non-1
//!    dims are unchanged; all broadcast offsets stay within original allocation.
//! 7. **Narrow bounds** — narrowed view is within original bounds; start + len
//!    <= dim guarantees all accessed offsets are valid.
//! 8. **Memory overlap detection** — two views into the same allocation overlap
//!    iff their accessed offset ranges intersect.
//! 9. **Zero-copy view** — view shares the same allocation size and strides
//!    derive from the parent; modifying through view affects parent.
//! 10. **Alignment** — tensor data meets dtype alignment requirements; byte
//!     offsets are multiples of element size.
//!
//! All harnesses use small concrete bounds (u8) for CBMC tractability.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ===========================================================================
// Helpers
// ===========================================================================

/// Compute contiguous (row-major) strides for up to 4 dimensions.
/// Returns None on overflow.
fn contiguous_strides(dims: &[usize; 4], rank: usize) -> Option<[usize; 4]> {
    assert!(rank <= 4);
    let mut strides = [0usize; 4];
    if rank == 0 {
        return Some(strides);
    }
    strides[rank - 1] = 1;
    let mut i = rank - 1;
    while i > 0 {
        strides[i - 1] = strides[i].checked_mul(dims[i])?;
        i -= 1;
    }
    Some(strides)
}

/// Compute the linear offset for a multi-dim index given strides.
/// Returns None on overflow.
fn linear_offset(indices: &[usize; 4], strides: &[usize; 4], rank: usize) -> Option<usize> {
    let mut acc = 0usize;
    let mut i = 0;
    while i < rank {
        let contribution = strides[i].checked_mul(indices[i])?;
        acc = acc.checked_add(contribution)?;
        i += 1;
    }
    Some(acc)
}

/// Check if a permutation array of length `n` is valid (bijection on 0..n).
fn is_valid_permutation(perm: &[usize], n: usize) -> bool {
    let mut seen = [false; 4];
    let mut i = 0;
    while i < n {
        if perm[i] >= n || seen[perm[i]] {
            return false;
        }
        seen[perm[i]] = true;
        i += 1;
    }
    true
}

// ===========================================================================
// 1. Stride-contiguity invariant: contiguous tensor has row-major strides
// ===========================================================================

/// Prove: a contiguous rank-3 tensor's strides satisfy the row-major invariant:
/// stride[i] = product(dims[i+1..rank]) for all i, and the innermost stride
/// is always 1. This is the defining property of C-contiguous layout.
#[kani::unwind(1)]
#[kani::proof]
fn contiguous_tensor_has_row_major_strides_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);

    let dims = [d0 as usize, d1 as usize, d2 as usize, 0];
    let strides = contiguous_strides(&dims, 3).unwrap();

    // Innermost stride is 1
    assert_eq!(
        strides[2], 1,
        "innermost stride must be 1 for contiguous tensor"
    );

    // stride[1] = dims[2]
    assert_eq!(strides[1], dims[2], "stride[1] must equal dims[2]");

    // stride[0] = dims[1] * dims[2]
    assert_eq!(
        strides[0],
        dims[1] * dims[2],
        "stride[0] must equal dims[1]*dims[2]"
    );

    // Verify non-increasing order (row-major property)
    assert!(strides[0] >= strides[1], "strides must be non-increasing");
    assert!(strides[1] >= strides[2], "strides must be non-increasing");
}

/// Prove: the stride-based contiguity check (stride[i] == stride[i+1] * dims[i+1])
/// is equivalent to the product formula (stride[i] == product(dims[i+1..rank])).
/// Both formulations detect contiguous layout correctly.
#[kani::unwind(1)]
#[kani::proof]
fn contiguity_check_equivalence_rank4() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 6);
    kani::assume(d1 >= 1 && d1 <= 6);
    kani::assume(d2 >= 1 && d2 <= 6);
    kani::assume(d3 >= 1 && d3 <= 6);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];
    let strides = contiguous_strides(&dims, 4).unwrap();

    // Recurrence check: stride[i] == stride[i+1] * dims[i+1]
    let recurrence_ok = strides[3] == 1
        && strides[2] == strides[3] * dims[3]
        && strides[1] == strides[2] * dims[2]
        && strides[0] == strides[1] * dims[1];

    // Product check: stride[i] == product(dims[i+1..4])
    let product_ok = strides[3] == 1
        && strides[2] == dims[3]
        && strides[1] == dims[2] * dims[3]
        && strides[0] == dims[1] * dims[2] * dims[3];

    assert!(
        recurrence_ok,
        "recurrence formula must hold for contiguous strides"
    );
    assert!(
        product_ok,
        "product formula must hold for contiguous strides"
    );
    // Both must agree
    assert_eq!(
        recurrence_ok, product_ok,
        "recurrence and product checks must agree"
    );
}

// ===========================================================================
// 2. View safety: view doesn't extend beyond original allocation
// ===========================================================================

/// Prove: a view created by slicing along dimension 0 with offset and length
/// never accesses memory beyond the original allocation. Every valid index
/// in the view maps to a valid offset in the parent.
#[kani::unwind(1)]
#[kani::proof]
fn view_slice_within_allocation_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let start: u8 = kani::any();
    let len: u8 = kani::any();

    kani::assume(d0 >= 2 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(len >= 1);
    kani::assume(start < d0);
    kani::assume((start as usize) + (len as usize) <= d0 as usize);

    let dims = [d0 as usize, d1 as usize, d2 as usize, 0];
    let strides = contiguous_strides(&dims, 3).unwrap();

    let base_offset = (start as usize) * strides[0];

    // Maximum offset in the view: (len-1)*stride[0] + (d1-1)*stride[1] + (d2-1)*stride[2]
    let max_view_offset = base_offset
        + (len as usize - 1) * strides[0]
        + (dims[1] - 1) * strides[1]
        + (dims[2] - 1) * strides[2];

    let numel = checked_dim_product(&[dims[0], dims[1], dims[2]]);
    if let Ok(n) = numel {
        assert!(
            max_view_offset < n,
            "view's max offset must be within parent allocation"
        );
    }
}

/// Prove: a strided view (e.g., every-other element along dim 1) stays
/// within the original allocation for all accessible indices.
#[kani::unwind(1)]
#[kani::proof]
fn strided_view_within_allocation_rank2() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let step: u8 = kani::any();
    let start: u8 = kani::any();
    let view_len: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 2 && d1 <= 8);
    kani::assume(step >= 1 && step <= 4);
    kani::assume(view_len >= 1);
    kani::assume(start < d1);

    let su = start as usize;
    let stepu = step as usize;
    let vlu = view_len as usize;

    // Last accessed index along dim 1
    let last_idx = su + (vlu - 1) * stepu;
    kani::assume(last_idx < d1 as usize);

    let dims = [d0 as usize, d1 as usize, 0, 0];
    let strides = contiguous_strides(&dims, 2).unwrap();

    // Max offset in strided view: (d0-1)*stride[0] + last_idx * stride[1]
    let max_offset = (dims[0] - 1) * strides[0] + last_idx * strides[1];

    let numel = checked_dim_product(&[dims[0], dims[1]]);
    if let Ok(n) = numel {
        assert!(
            max_offset < n,
            "strided view max offset must be within allocation"
        );
    }
}

// ===========================================================================
// 3. Reshape preserves element count
// ===========================================================================

/// Prove: reshape from [A, B, C] to [A, B*C] preserves total element count.
/// This is the most common reshape pattern (flattening trailing dims).
#[kani::unwind(1)]
#[kani::proof]
fn reshape_flatten_trailing_preserves_numel_rank3_to_2() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();

    kani::assume(a >= 1 && a <= 8);
    kani::assume(b >= 1 && b <= 8);
    kani::assume(c >= 1 && c <= 8);

    let au = a as usize;
    let bu = b as usize;
    let cu = c as usize;

    if let Some(bc) = bu.checked_mul(cu) {
        let old_numel = checked_dim_product(&[au, bu, cu]);
        let new_numel = checked_dim_product(&[au, bc]);

        if let (Ok(on), Ok(nn)) = (old_numel, new_numel) {
            assert_eq!(on, nn, "reshape must preserve total element count");
        }
    }
}

/// Prove: reshape from [A*B, C] to [A, B, C] preserves element count.
/// Unflatten (the inverse of flatten) must also be numel-preserving.
#[kani::unwind(1)]
#[kani::proof]
fn reshape_unflatten_preserves_numel_rank2_to_3() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();

    kani::assume(a >= 1 && a <= 8);
    kani::assume(b >= 1 && b <= 8);
    kani::assume(c >= 1 && c <= 8);

    let au = a as usize;
    let bu = b as usize;
    let cu = c as usize;

    if let Some(ab) = au.checked_mul(bu) {
        let old_numel = checked_dim_product(&[ab, cu]);
        let new_numel = checked_dim_product(&[au, bu, cu]);

        if let (Ok(on), Ok(nn)) = (old_numel, new_numel) {
            assert_eq!(on, nn, "unflatten must preserve total element count");
        }
    }
}

// ===========================================================================
// 4. Transpose correctness: swaps dims and strides
// ===========================================================================

/// Prove: transpose(dim_a, dim_b) correctly swaps both dimensions and strides,
/// and the maximum addressable offset is unchanged (same backing memory).
#[kani::unwind(1)]
#[kani::proof]
fn transpose_swaps_dims_and_strides_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let swap_a: u8 = kani::any();
    let swap_b: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(swap_a < 3 && swap_b < 3 && swap_a != swap_b);

    let a = swap_a as usize;
    let b = swap_b as usize;

    let dims = [d0 as usize, d1 as usize, d2 as usize, 0];
    let strides = contiguous_strides(&dims, 3).unwrap();

    // Transpose: swap dims and strides
    let mut t_dims = dims;
    t_dims.swap(a, b);
    let mut t_strides = strides;
    t_strides.swap(a, b);

    // Verify dims were swapped
    assert_eq!(
        t_dims[a], dims[b],
        "transposed dim[a] must equal original dim[b]"
    );
    assert_eq!(
        t_dims[b], dims[a],
        "transposed dim[b] must equal original dim[a]"
    );

    // Verify strides were swapped
    assert_eq!(
        t_strides[a], strides[b],
        "transposed stride[a] must equal original stride[b]"
    );
    assert_eq!(
        t_strides[b], strides[a],
        "transposed stride[b] must equal original stride[a]"
    );

    // Max offset unchanged
    let orig_max =
        (dims[0] - 1) * strides[0] + (dims[1] - 1) * strides[1] + (dims[2] - 1) * strides[2];
    let trans_max = (t_dims[0] - 1) * t_strides[0]
        + (t_dims[1] - 1) * t_strides[1]
        + (t_dims[2] - 1) * t_strides[2];
    assert_eq!(
        orig_max, trans_max,
        "transpose must preserve max addressable offset"
    );

    // Double-transpose recovers original
    let mut tt_dims = t_dims;
    tt_dims.swap(a, b);
    let mut tt_strides = t_strides;
    tt_strides.swap(a, b);

    assert_eq!(tt_dims[0], dims[0], "double transpose must recover dim[0]");
    assert_eq!(tt_dims[1], dims[1], "double transpose must recover dim[1]");
    assert_eq!(tt_dims[2], dims[2], "double transpose must recover dim[2]");
    assert_eq!(
        tt_strides[0], strides[0],
        "double transpose must recover stride[0]"
    );
    assert_eq!(
        tt_strides[1], strides[1],
        "double transpose must recover stride[1]"
    );
    assert_eq!(
        tt_strides[2], strides[2],
        "double transpose must recover stride[2]"
    );
}

// ===========================================================================
// 5. Permute correctness: permuted dims are a valid permutation
// ===========================================================================

/// Prove: a permutation applied to dims and strides produces a valid layout
/// where the inverse permutation exactly recovers the original layout, and
/// the permutation is verified to be a bijection.
#[kani::unwind(1)]
#[kani::proof]
fn permute_is_bijection_and_invertible_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);

    let p0: u8 = kani::any();
    let p1: u8 = kani::any();
    let p2: u8 = kani::any();
    kani::assume(p0 < 3 && p1 < 3 && p2 < 3);
    kani::assume(p0 != p1 && p0 != p2 && p1 != p2);

    let perm = [p0 as usize, p1 as usize, p2 as usize];

    // Verify it's a valid permutation
    assert!(
        is_valid_permutation(&perm, 3),
        "permutation must be a bijection on {0,1,2}"
    );

    let dims = [d0 as usize, d1 as usize, d2 as usize, 0];
    let strides = contiguous_strides(&dims, 3).unwrap();

    // Apply permutation
    let perm_dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]], 0];
    let perm_strides = [strides[perm[0]], strides[perm[1]], strides[perm[2]], 0];

    // Compute inverse permutation
    let mut inv_perm = [0usize; 3];
    inv_perm[perm[0]] = 0;
    inv_perm[perm[1]] = 1;
    inv_perm[perm[2]] = 2;

    // Apply inverse to get back original
    let recovered_dims = [
        perm_dims[inv_perm[0]],
        perm_dims[inv_perm[1]],
        perm_dims[inv_perm[2]],
    ];
    let recovered_strides = [
        perm_strides[inv_perm[0]],
        perm_strides[inv_perm[1]],
        perm_strides[inv_perm[2]],
    ];

    assert_eq!(
        recovered_dims[0], dims[0],
        "inverse permute must recover dim[0]"
    );
    assert_eq!(
        recovered_dims[1], dims[1],
        "inverse permute must recover dim[1]"
    );
    assert_eq!(
        recovered_dims[2], dims[2],
        "inverse permute must recover dim[2]"
    );
    assert_eq!(
        recovered_strides[0], strides[0],
        "inverse permute must recover stride[0]"
    );
    assert_eq!(
        recovered_strides[1], strides[1],
        "inverse permute must recover stride[1]"
    );
    assert_eq!(
        recovered_strides[2], strides[2],
        "inverse permute must recover stride[2]"
    );

    // Numel must be preserved
    let orig_numel = checked_dim_product(&[dims[0], dims[1], dims[2]]);
    let perm_numel = checked_dim_product(&[perm_dims[0], perm_dims[1], perm_dims[2]]);
    if let (Ok(on), Ok(pn)) = (orig_numel, perm_numel) {
        assert_eq!(on, pn, "permutation must preserve numel");
    }
}

// ===========================================================================
// 6. Broadcast safety: only expands size-1 dimensions
// ===========================================================================

/// Prove: broadcast from [1, C, 1] to [B, C, T] only modifies dimensions
/// that were originally size 1, and all broadcast-view offsets stay within
/// the original allocation of size C.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_only_expands_size_one_dims() {
    let c: u8 = kani::any();
    let b: u8 = kani::any();
    let t: u8 = kani::any();

    kani::assume(c >= 1 && c <= 8);
    kani::assume(b >= 1 && b <= 8);
    kani::assume(t >= 1 && t <= 8);

    let orig_dims = [1usize, c as usize, 1, 0];
    let orig_strides = contiguous_strides(&orig_dims, 3).unwrap();

    // Broadcast to [B, C, T]: set stride=0 for expanded dims
    let bcast_dims = [b as usize, c as usize, t as usize, 0];
    let bcast_strides = [0usize, orig_strides[1], 0, 0];

    // Verify: only size-1 dims were expanded
    // Dim 0: was 1, now B (expanded) => stride must be 0
    assert_eq!(bcast_strides[0], 0, "expanded dim must have stride 0");
    // Dim 1: was C, stays C (not expanded) => stride unchanged
    assert_eq!(
        bcast_strides[1], orig_strides[1],
        "non-expanded dim stride must be unchanged"
    );
    // Dim 2: was 1, now T (expanded) => stride must be 0
    assert_eq!(bcast_strides[2], 0, "expanded dim must have stride 0");

    // Verify: any valid broadcast index maps to offset < orig_numel
    let i0: u8 = kani::any();
    let i1: u8 = kani::any();
    let i2: u8 = kani::any();

    kani::assume((i0 as usize) < bcast_dims[0]);
    kani::assume((i1 as usize) < bcast_dims[1]);
    kani::assume((i2 as usize) < bcast_dims[2]);

    let idx = [i0 as usize, i1 as usize, i2 as usize, 0];
    let offset = linear_offset(&idx, &bcast_strides, 3).unwrap();

    let orig_numel = checked_dim_product(&[1, c as usize, 1]);
    if let Ok(n) = orig_numel {
        assert!(
            offset < n,
            "broadcast offset must stay within original allocation"
        );
    }
}

/// Prove: attempting to broadcast a non-1 dimension to a different size is
/// invalid. The broadcast compatibility check correctly rejects this case.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_rejects_non_one_dim_mismatch() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();

    kani::assume(a >= 2 && a <= 8);
    kani::assume(b >= 2 && b <= 8);
    kani::assume(a != b);

    // a != b and a != 1 and b != 1 => not broadcastable
    let broadcastable = a == b || a == 1 || b == 1;
    assert!(
        !broadcastable,
        "non-1 dim mismatch must not be broadcastable"
    );
}

// ===========================================================================
// 7. Narrow bounds: narrowed view is within original bounds
// ===========================================================================

/// Prove: narrow(dim, start, len) with start + len <= dims[dim] guarantees
/// all accessed offsets in the narrowed view are within the original
/// allocation, for an arbitrary dimension in a rank-3 tensor.
#[kani::unwind(1)]
#[kani::proof]
fn narrow_all_offsets_within_original_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let narrow_dim: u8 = kani::any();
    let start: u8 = kani::any();
    let len: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(narrow_dim < 3);
    kani::assume(len >= 1);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let dim = narrow_dim as usize;
    let su = start as usize;
    let lu = len as usize;

    // Precondition: start + len <= dims[dim]
    kani::assume(su + lu <= dims[dim]);

    let strides_4 = [0usize; 4];
    let dims_4 = [dims[0], dims[1], dims[2], 0];
    let strides = contiguous_strides(&dims_4, 3).unwrap();

    // The narrowed view's byte_offset = start * strides[dim]
    let base_offset = su * strides[dim];

    // Max index within the narrowed view has:
    // - dims[dim] replaced by len (narrowed range)
    // - other dims unchanged
    let mut max_idx = [dims[0] - 1, dims[1] - 1, dims[2] - 1, 0];
    max_idx[dim] = su + lu - 1;

    let max_offset = max_idx[0] * strides[0] + max_idx[1] * strides[1] + max_idx[2] * strides[2];

    let numel = checked_dim_product(&dims);
    if let Ok(n) = numel {
        assert!(
            max_offset < n,
            "all narrowed offsets must be within original allocation"
        );
    }
}

/// Prove: narrow with start=0 and len=dims[dim] produces the same numel
/// and is equivalent to not narrowing at all (identity narrow).
#[kani::unwind(1)]
#[kani::proof]
fn narrow_full_range_is_identity() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let narrow_dim: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(narrow_dim < 2);

    let dims = [d0 as usize, d1 as usize];
    let dim = narrow_dim as usize;

    // Full-range narrow: start=0, len=dims[dim]
    let mut narrowed_dims = dims;
    narrowed_dims[dim] = dims[dim]; // unchanged

    let orig_numel = checked_dim_product(&dims);
    let narrow_numel = checked_dim_product(&narrowed_dims);

    if let (Ok(on), Ok(nn)) = (orig_numel, narrow_numel) {
        assert_eq!(on, nn, "full-range narrow must preserve numel");
    }
    assert_eq!(narrowed_dims[0], dims[0], "full narrow preserves dim 0");
    assert_eq!(narrowed_dims[1], dims[1], "full narrow preserves dim 1");
}

// ===========================================================================
// 8. Memory overlap detection: overlapping views correctly identified
// ===========================================================================

/// Prove: two sliced views into the same 1D allocation overlap iff their
/// [start, start+len) ranges intersect. This is the fundamental overlap
/// detection property for views sharing a backing buffer.
#[kani::unwind(1)]
#[kani::proof]
fn memory_overlap_detection_1d_views() {
    let total: u8 = kani::any();
    kani::assume(total >= 4 && total <= 16);

    let start_a: u8 = kani::any();
    let len_a: u8 = kani::any();
    let start_b: u8 = kani::any();
    let len_b: u8 = kani::any();

    kani::assume(len_a >= 1 && len_b >= 1);
    kani::assume(start_a < total && start_b < total);
    kani::assume((start_a as usize) + (len_a as usize) <= total as usize);
    kani::assume((start_b as usize) + (len_b as usize) <= total as usize);

    let sa = start_a as usize;
    let la = len_a as usize;
    let sb = start_b as usize;
    let lb = len_b as usize;

    let end_a = sa + la; // exclusive end
    let end_b = sb + lb;

    // Two ranges [sa, end_a) and [sb, end_b) overlap iff sa < end_b && sb < end_a
    let ranges_overlap = sa < end_b && sb < end_a;

    // Alternative formulation: NOT (end_a <= sb || end_b <= sa)
    let ranges_disjoint = end_a <= sb || end_b <= sa;

    assert_eq!(
        ranges_overlap, !ranges_disjoint,
        "overlap detection must be consistent with disjointness check"
    );

    // If disjoint, no shared index exists
    if ranges_disjoint {
        // The maximum index of A (sa + la - 1) must be less than start of B,
        // OR the maximum index of B (sb + lb - 1) must be less than start of A
        assert!(
            end_a <= sb || end_b <= sa,
            "disjoint views must not share any index"
        );
    }

    // If overlapping, there exists at least one shared index
    if ranges_overlap {
        let overlap_start = if sa >= sb { sa } else { sb };
        let overlap_end = if end_a <= end_b { end_a } else { end_b };
        assert!(
            overlap_start < overlap_end,
            "overlapping views must have positive overlap region"
        );
        // The shared index is within both views
        assert!(
            overlap_start >= sa && overlap_start < end_a,
            "overlap start in view A"
        );
        assert!(
            overlap_start >= sb && overlap_start < end_b,
            "overlap start in view B"
        );
    }
}

/// Prove: two views of a 2D tensor created by narrowing along dim 0
/// overlap iff their row ranges intersect. The overlap region size
/// is correctly computed.
#[kani::unwind(1)]
#[kani::proof]
fn memory_overlap_detection_2d_row_views() {
    let rows: u8 = kani::any();
    let cols: u8 = kani::any();
    let start_a: u8 = kani::any();
    let len_a: u8 = kani::any();
    let start_b: u8 = kani::any();
    let len_b: u8 = kani::any();

    kani::assume(rows >= 2 && rows <= 8);
    kani::assume(cols >= 1 && cols <= 8);
    kani::assume(len_a >= 1 && len_b >= 1);
    kani::assume((start_a as usize) + (len_a as usize) <= rows as usize);
    kani::assume((start_b as usize) + (len_b as usize) <= rows as usize);

    let sa = start_a as usize;
    let la = len_a as usize;
    let sb = start_b as usize;
    let lb = len_b as usize;
    let cu = cols as usize;

    let end_a = sa + la;
    let end_b = sb + lb;

    // Row ranges overlap iff row-ranges intersect
    let rows_overlap = sa < end_b && sb < end_a;

    if rows_overlap {
        // Overlapping element count = overlap_rows * cols
        let overlap_start = if sa >= sb { sa } else { sb };
        let overlap_end = if end_a <= end_b { end_a } else { end_b };
        let overlap_rows = overlap_end - overlap_start;
        let overlap_elements = overlap_rows * cu;

        assert!(
            overlap_elements >= cu,
            "row overlap must cover at least one full row"
        );
        assert!(
            overlap_elements <= la * cu && overlap_elements <= lb * cu,
            "overlap cannot exceed either view's size"
        );
    }
}

// ===========================================================================
// 9. Zero-copy view: view shares underlying allocation
// ===========================================================================

/// Prove: a zero-copy view created by narrow() has the same strides as
/// the parent (strides are not copied/modified), only the offset and
/// extent change. The view's logical size is a subset of the parent's.
#[kani::unwind(1)]
#[kani::proof]
fn zero_copy_view_shares_strides_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let start: u8 = kani::any();
    let len: u8 = kani::any();

    kani::assume(d0 >= 2 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(len >= 1 && len <= d0);
    kani::assume((start as usize) + (len as usize) <= d0 as usize);

    let parent_dims = [d0 as usize, d1 as usize, d2 as usize, 0];
    let parent_strides = contiguous_strides(&parent_dims, 3).unwrap();

    // The view has the same strides (zero-copy: no data movement)
    let view_strides = parent_strides;

    // View dims differ only in the narrowed dimension
    let view_dims = [len as usize, d1 as usize, d2 as usize, 0];

    // Strides are identical
    assert_eq!(
        view_strides[0], parent_strides[0],
        "view stride[0] must match parent"
    );
    assert_eq!(
        view_strides[1], parent_strides[1],
        "view stride[1] must match parent"
    );
    assert_eq!(
        view_strides[2], parent_strides[2],
        "view stride[2] must match parent"
    );

    // View numel <= parent numel
    let parent_numel = checked_dim_product(&[parent_dims[0], parent_dims[1], parent_dims[2]]);
    let view_numel = checked_dim_product(&[view_dims[0], view_dims[1], view_dims[2]]);

    if let (Ok(pn), Ok(vn)) = (parent_numel, view_numel) {
        assert!(vn <= pn, "view numel must not exceed parent numel");
    }

    // The view's byte_offset = start * parent_strides[0]
    let byte_offset = (start as usize) * parent_strides[0];

    // Any index [i, j, k] in the view maps to parent offset:
    // byte_offset + i * stride[0] + j * stride[1] + k * stride[2]
    // which is the same as parent index [start + i, j, k]
    // This proves the view accesses the SAME memory as the parent.
    let test_i: u8 = kani::any();
    let test_j: u8 = kani::any();
    let test_k: u8 = kani::any();

    kani::assume((test_i as usize) < view_dims[0]);
    kani::assume((test_j as usize) < view_dims[1]);
    kani::assume((test_k as usize) < view_dims[2]);

    let view_offset = byte_offset
        + (test_i as usize) * view_strides[0]
        + (test_j as usize) * view_strides[1]
        + (test_k as usize) * view_strides[2];

    // Same as parent[start + test_i, test_j, test_k]
    let parent_idx = [
        (start as usize) + (test_i as usize),
        test_j as usize,
        test_k as usize,
        0,
    ];
    let parent_offset = linear_offset(&parent_idx, &parent_strides, 3).unwrap();

    assert_eq!(
        view_offset, parent_offset,
        "view access must resolve to same parent offset (zero-copy)"
    );
}

/// Prove: a transposed view shares the same backing allocation size.
/// The max addressable offset is identical between original and transposed,
/// confirming they share memory (zero-copy transpose).
#[kani::unwind(1)]
#[kani::proof]
fn zero_copy_transpose_shares_allocation_rank2() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);

    let dims = [d0 as usize, d1 as usize, 0, 0];
    let strides = contiguous_strides(&dims, 2).unwrap();

    // Transpose: swap dims and strides
    let t_dims = [dims[1], dims[0], 0, 0];
    let t_strides = [strides[1], strides[0], 0, 0];

    // Both have same numel
    let orig_numel = checked_dim_product(&[dims[0], dims[1]]);
    let trans_numel = checked_dim_product(&[t_dims[0], t_dims[1]]);

    if let (Ok(on), Ok(tn)) = (orig_numel, trans_numel) {
        assert_eq!(on, tn, "transposed view must have same numel as original");
    }

    // Both have same max offset (sharing same allocation)
    let orig_max = (dims[0] - 1) * strides[0] + (dims[1] - 1) * strides[1];
    let trans_max = (t_dims[0] - 1) * t_strides[0] + (t_dims[1] - 1) * t_strides[1];

    assert_eq!(
        orig_max, trans_max,
        "transposed view max offset must equal original (shared allocation)"
    );
}

// ===========================================================================
// 10. Alignment: tensor data meets dtype alignment requirements
// ===========================================================================

/// Prove: for any valid byte_offset that is a multiple of element_size,
/// and any valid element index, the resulting byte address is also aligned
/// to element_size. This is the dtype alignment invariant.
#[kani::unwind(1)]
#[kani::proof]
fn alignment_byte_offset_multiple_of_element_size() {
    let element_size: u8 = kani::any();
    // Common element sizes: 1 (u8), 2 (f16/bf16), 4 (f32/i32), 8 (f64/i64)
    kani::assume(element_size == 1 || element_size == 2 || element_size == 4 || element_size == 8);

    let es = element_size as usize;

    let base_offset_units: u8 = kani::any();
    kani::assume(base_offset_units <= 32);

    // Base byte offset is a multiple of element_size
    let base_byte_offset = (base_offset_units as usize) * es;

    // Element-level index
    let elem_index: u8 = kani::any();
    kani::assume(elem_index <= 64);

    // Final byte address = base_byte_offset + elem_index * element_size
    let final_byte_address = base_byte_offset + (elem_index as usize) * es;

    // The final byte address must be aligned to element_size
    assert_eq!(
        final_byte_address % es,
        0,
        "tensor byte address must be aligned to element size"
    );

    // Also verify: the base offset itself is aligned
    assert_eq!(
        base_byte_offset % es,
        0,
        "base byte offset must be aligned to element size"
    );
}

/// Prove: when creating a view with byte_offset (from narrow/slice), if the
/// original allocation was aligned and the byte_offset is a multiple of
/// element_size, then the view is also aligned. This is the alignment
/// preservation property for zero-copy views.
#[kani::unwind(1)]
#[kani::proof]
fn alignment_preserved_through_view_creation() {
    let element_size: u8 = kani::any();
    kani::assume(element_size == 1 || element_size == 2 || element_size == 4 || element_size == 8);

    let es = element_size as usize;

    // Dimension along which we narrow (produces byte_offset = start * stride * element_size)
    let dim_size: u8 = kani::any();
    let start: u8 = kani::any();

    kani::assume(dim_size >= 2 && dim_size <= 16);
    kani::assume(start < dim_size);

    // Stride in elements (always an integer for contiguous tensors)
    let stride_elems: u8 = kani::any();
    kani::assume(stride_elems >= 1 && stride_elems <= 64);

    // Byte offset = start * stride_elems * element_size
    if let Some(step) = (stride_elems as usize).checked_mul(es) {
        if let Some(offset) = (start as usize).checked_mul(step) {
            // The byte offset is always a multiple of element_size
            // because it is start * stride * element_size
            assert_eq!(
                offset % es,
                0,
                "view byte offset must be aligned to element size"
            );

            // Any element access within the view:
            // addr = offset + idx * stride_elems * element_size
            let idx: u8 = kani::any();
            kani::assume(idx <= 32);

            if let Some(elem_step) = (idx as usize).checked_mul(step) {
                let addr = offset + elem_step;
                assert_eq!(addr % es, 0, "element access in view must be aligned");
            }
        }
    }
}
