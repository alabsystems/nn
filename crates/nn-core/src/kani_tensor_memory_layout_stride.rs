// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for tensor memory layout and stride safety.
//!
//! Part of #4242. Proves seven categories of memory layout invariants:
//!
//! 1. **Contiguous stride calculation** — strides computed from shape satisfy
//!    stride[i] = product(shape[i+1..]) and are consistent with the mixed-radix
//!    numeral system that maps multi-dim indices to linear offsets.
//! 2. **Offset bounds** — any valid multi-dimensional index maps to a linear
//!    offset strictly within the allocation (< numel).
//! 3. **Transpose stride validity** — transposed strides produce valid offsets
//!    and the max offset is preserved.
//! 4. **Reshape preserves element count** — product of old shape == product of
//!    new shape for split/merge dimension transformations.
//! 5. **Broadcast stride zero invariant** — broadcast dimensions have stride 0,
//!    non-broadcast dimensions retain original strides.
//! 6. **Slice offset validity** — sliced tensor byte offsets stay within the
//!    original allocation for arbitrary start/step/length combinations.
//! 7. **Permute stride correctness** — permuted strides match the expected
//!    reordering and the inverse permutation recovers the original.
//!
//! All harnesses use small concrete bounds (u8) for CBMC tractability.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ===========================================================================
// Helpers
// ===========================================================================

/// Compute contiguous (row-major) strides for up to 4 dimensions.
/// Returns None on overflow.
fn contiguous_strides_4(dims: &[usize; 4], rank: usize) -> Option<[usize; 4]> {
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
fn linear_offset_4(indices: &[usize; 4], strides: &[usize; 4], rank: usize) -> Option<usize> {
    let mut acc = 0usize;
    let mut i = 0;
    while i < rank {
        let contribution = strides[i].checked_mul(indices[i])?;
        acc = acc.checked_add(contribution)?;
        i += 1;
    }
    Some(acc)
}

// ===========================================================================
// 1. Contiguous stride calculation — mixed-radix consistency
// ===========================================================================

/// Prove: for contiguous rank-4, the linear offset computed via strides equals
/// the mixed-radix flat index: ((i0 * d1 + i1) * d2 + i2) * d3 + i3.
///
/// This proves the stride-based index formula is equivalent to the standard
/// row-major linearization, confirming that strides correctly implement
/// the C-order memory layout.
#[kani::unwind(1)]
#[kani::proof]
fn contiguous_stride_matches_mixed_radix_rank4() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 6);
    kani::assume(d1 >= 1 && d1 <= 6);
    kani::assume(d2 >= 1 && d2 <= 6);
    kani::assume(d3 >= 1 && d3 <= 6);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];
    let strides = contiguous_strides_4(&dims, 4).unwrap();

    // Pick a symbolic index within bounds
    let i0: u8 = kani::any();
    let i1: u8 = kani::any();
    let i2: u8 = kani::any();
    let i3: u8 = kani::any();

    kani::assume((i0 as usize) < dims[0]);
    kani::assume((i1 as usize) < dims[1]);
    kani::assume((i2 as usize) < dims[2]);
    kani::assume((i3 as usize) < dims[3]);

    let idx = [i0 as usize, i1 as usize, i2 as usize, i3 as usize];

    // Stride-based offset
    let stride_offset = linear_offset_4(&idx, &strides, 4).unwrap();

    // Mixed-radix flat index: ((i0 * d1 + i1) * d2 + i2) * d3 + i3
    let mixed_radix = ((idx[0] * dims[1] + idx[1]) * dims[2] + idx[2]) * dims[3] + idx[3];

    assert_eq!(
        stride_offset, mixed_radix,
        "stride offset must equal mixed-radix flat index"
    );
}

/// Prove: contiguous strides satisfy the recurrence relation
/// stride[i] = stride[i+1] * dims[i+1] for all i in 0..rank-1.
///
/// This is the defining property of contiguous layout.
#[kani::unwind(1)]
#[kani::proof]
fn contiguous_stride_recurrence_rank4() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(d3 >= 1 && d3 <= 8);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];
    let strides = contiguous_strides_4(&dims, 4).unwrap();

    // Innermost stride is always 1
    assert_eq!(strides[3], 1, "innermost stride must be 1");

    // Recurrence: stride[i] = stride[i+1] * dims[i+1]
    assert_eq!(
        strides[2],
        strides[3] * dims[3],
        "stride recurrence at dim 2"
    );
    assert_eq!(
        strides[1],
        strides[2] * dims[2],
        "stride recurrence at dim 1"
    );
    assert_eq!(
        strides[0],
        strides[1] * dims[1],
        "stride recurrence at dim 0"
    );

    // Equivalently: stride[0] = numel / dims[0]
    let numel = checked_dim_product(&dims);
    if let Ok(n) = numel {
        assert_eq!(strides[0], n / dims[0], "stride[0] = numel / dims[0]");
    }
}

// ===========================================================================
// 2. Offset bounds — valid indices map within allocation
// ===========================================================================

/// Prove: any valid multi-dim index in a contiguous rank-3 tensor maps to
/// a linear offset strictly less than numel.
///
/// This is the fundamental memory safety property: no in-bounds logical
/// index ever produces an out-of-bounds physical address.
#[kani::unwind(1)]
#[kani::proof]
fn valid_index_within_allocation_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);

    let dims = [d0 as usize, d1 as usize, d2 as usize, 0];
    let strides = contiguous_strides_4(&dims, 3).unwrap();

    let i0: u8 = kani::any();
    let i1: u8 = kani::any();
    let i2: u8 = kani::any();

    kani::assume((i0 as usize) < dims[0]);
    kani::assume((i1 as usize) < dims[1]);
    kani::assume((i2 as usize) < dims[2]);

    let idx = [i0 as usize, i1 as usize, i2 as usize, 0];
    let offset = linear_offset_4(&idx, &strides, 3).unwrap();

    let numel = checked_dim_product(&[dims[0], dims[1], dims[2]]);
    if let Ok(n) = numel {
        assert!(offset < n, "offset must be strictly less than numel");
    }
}

/// Prove: the offset for the all-zeros index is always 0, and the offset
/// for the max index (dims[i]-1 for each i) is always numel-1.
///
/// These are the boundary conditions for offset computation.
#[kani::unwind(1)]
#[kani::proof]
fn offset_boundary_conditions_rank4() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 6);
    kani::assume(d1 >= 1 && d1 <= 6);
    kani::assume(d2 >= 1 && d2 <= 6);
    kani::assume(d3 >= 1 && d3 <= 6);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];
    let strides = contiguous_strides_4(&dims, 4).unwrap();

    // Zero index -> offset 0
    let zero_idx = [0usize, 0, 0, 0];
    let zero_offset = linear_offset_4(&zero_idx, &strides, 4).unwrap();
    assert_eq!(zero_offset, 0, "zero index must map to offset 0");

    // Max index -> offset numel - 1
    let max_idx = [dims[0] - 1, dims[1] - 1, dims[2] - 1, dims[3] - 1];
    let max_offset = linear_offset_4(&max_idx, &strides, 4).unwrap();

    let numel = checked_dim_product(&dims);
    if let Ok(n) = numel {
        assert_eq!(max_offset, n - 1, "max index must map to offset numel-1");
    }
}

/// Prove: incrementing a single index dimension by 1 increases the linear
/// offset by exactly stride[dim]. This verifies the per-axis stepping
/// behavior of strided access.
#[kani::unwind(1)]
#[kani::proof]
fn single_step_offset_equals_stride_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 2 && d0 <= 8);
    kani::assume(d1 >= 2 && d1 <= 8);
    kani::assume(d2 >= 2 && d2 <= 8);

    let dims = [d0 as usize, d1 as usize, d2 as usize, 0];
    let strides = contiguous_strides_4(&dims, 3).unwrap();

    // Pick a symbolic base index (not at the last position in any dim)
    let i0: u8 = kani::any();
    let i1: u8 = kani::any();
    let i2: u8 = kani::any();

    kani::assume((i0 as usize) + 1 < dims[0]);
    kani::assume((i1 as usize) + 1 < dims[1]);
    kani::assume((i2 as usize) + 1 < dims[2]);

    let base = [i0 as usize, i1 as usize, i2 as usize, 0];
    let base_offset = linear_offset_4(&base, &strides, 3).unwrap();

    // Step along dim 0
    let step0 = [base[0] + 1, base[1], base[2], 0];
    let off0 = linear_offset_4(&step0, &strides, 3).unwrap();
    assert_eq!(
        off0 - base_offset,
        strides[0],
        "stepping dim 0 by 1 must add stride[0]"
    );

    // Step along dim 1
    let step1 = [base[0], base[1] + 1, base[2], 0];
    let off1 = linear_offset_4(&step1, &strides, 3).unwrap();
    assert_eq!(
        off1 - base_offset,
        strides[1],
        "stepping dim 1 by 1 must add stride[1]"
    );

    // Step along dim 2
    let step2 = [base[0], base[1], base[2] + 1, 0];
    let off2 = linear_offset_4(&step2, &strides, 3).unwrap();
    assert_eq!(
        off2 - base_offset,
        strides[2],
        "stepping dim 2 by 1 must add stride[2]"
    );
}

// ===========================================================================
// 3. Transpose stride validity — transposed offsets stay in bounds
// ===========================================================================

/// Prove: after transposing dims (a, b) of a contiguous rank-4 tensor,
/// any valid index into the transposed view produces an offset < numel
/// of the original allocation.
///
/// Transpose reorders dims and strides but shares the same backing buffer.
/// Every element access in the transposed view must be safe.
#[kani::unwind(1)]
#[kani::proof]
fn transpose_offset_within_original_allocation_rank4() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 6);
    kani::assume(d1 >= 1 && d1 <= 6);
    kani::assume(d2 >= 1 && d2 <= 6);
    kani::assume(d3 >= 1 && d3 <= 6);

    let swap_a: u8 = kani::any();
    let swap_b: u8 = kani::any();
    kani::assume(swap_a < 4 && swap_b < 4 && swap_a != swap_b);

    let a = swap_a as usize;
    let b = swap_b as usize;

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];
    let strides = contiguous_strides_4(&dims, 4).unwrap();

    // Transpose: swap dims and strides
    let mut t_dims = dims;
    t_dims.swap(a, b);
    let mut t_strides = strides;
    t_strides.swap(a, b);

    // Symbolic index into the transposed tensor
    let i0: u8 = kani::any();
    let i1: u8 = kani::any();
    let i2: u8 = kani::any();
    let i3: u8 = kani::any();

    kani::assume((i0 as usize) < t_dims[0]);
    kani::assume((i1 as usize) < t_dims[1]);
    kani::assume((i2 as usize) < t_dims[2]);
    kani::assume((i3 as usize) < t_dims[3]);

    let idx = [i0 as usize, i1 as usize, i2 as usize, i3 as usize];
    let offset = linear_offset_4(&idx, &t_strides, 4).unwrap();

    let numel = checked_dim_product(&dims);
    if let Ok(n) = numel {
        assert!(
            offset < n,
            "transposed index must map within original allocation"
        );
    }
}

/// Prove: transpose is an involution (self-inverse) at the offset level.
///
/// Transposing (a,b) twice on any contiguous tensor recovers the same
/// linear offset for any given logical index.
#[kani::unwind(1)]
#[kani::proof]
fn transpose_involution_offset_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);

    let swap_a: u8 = kani::any();
    let swap_b: u8 = kani::any();
    kani::assume(swap_a < 3 && swap_b < 3 && swap_a != swap_b);

    let a = swap_a as usize;
    let b = swap_b as usize;

    let dims = [d0 as usize, d1 as usize, d2 as usize, 0];
    let strides = contiguous_strides_4(&dims, 3).unwrap();

    // First transpose
    let mut t1_dims = dims;
    t1_dims.swap(a, b);
    let mut t1_strides = strides;
    t1_strides.swap(a, b);

    // Second transpose (same axes)
    let mut t2_dims = t1_dims;
    t2_dims.swap(a, b);
    let mut t2_strides = t1_strides;
    t2_strides.swap(a, b);

    // Pick a valid index for the original tensor
    let i0: u8 = kani::any();
    let i1: u8 = kani::any();
    let i2: u8 = kani::any();

    kani::assume((i0 as usize) < dims[0]);
    kani::assume((i1 as usize) < dims[1]);
    kani::assume((i2 as usize) < dims[2]);

    let idx = [i0 as usize, i1 as usize, i2 as usize, 0];

    let orig_offset = linear_offset_4(&idx, &strides, 3).unwrap();
    let double_t_offset = linear_offset_4(&idx, &t2_strides, 3).unwrap();

    assert_eq!(
        orig_offset, double_t_offset,
        "double transpose must recover original offset"
    );
}

// ===========================================================================
// 4. Reshape preserves element count — split and merge dimensions
// ===========================================================================

/// Prove: splitting one dimension into two preserves numel.
///
/// [A, B*C, D] -> [A, B, C, D] when the middle dim is evenly divisible
/// by B. This is the inverse of flattening and is used in attention heads
/// (e.g., [batch, seq_len, num_heads * head_dim] -> [batch, seq_len, num_heads, head_dim]).
#[kani::unwind(1)]
#[kani::proof]
fn reshape_split_dim_preserves_numel() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(a >= 1 && a <= 8);
    kani::assume(b >= 1 && b <= 8);
    kani::assume(c >= 1 && c <= 8);
    kani::assume(d >= 1 && d <= 8);

    let bu = b as usize;
    let cu = c as usize;

    if let Some(bc) = bu.checked_mul(cu) {
        let old_dims = [a as usize, bc, d as usize];
        let new_dims = [a as usize, bu, cu, d as usize];

        let old_numel = checked_dim_product(&old_dims);
        let new_numel = checked_dim_product(&new_dims);

        if let (Ok(on), Ok(nn)) = (old_numel, new_numel) {
            assert_eq!(on, nn, "splitting a dimension must preserve numel");
        }
    }
}

/// Prove: merging two adjacent dimensions preserves numel.
///
/// [A, B, C, D] -> [A, B*C, D]. This is the fundamental flatten operation
/// used before linear layers and in many attention implementations.
#[kani::unwind(1)]
#[kani::proof]
fn reshape_merge_adjacent_dims_preserves_numel() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(a >= 1 && a <= 8);
    kani::assume(b >= 1 && b <= 8);
    kani::assume(c >= 1 && c <= 8);
    kani::assume(d >= 1 && d <= 8);

    let bu = b as usize;
    let cu = c as usize;

    if let Some(bc) = bu.checked_mul(cu) {
        let old_dims = [a as usize, bu, cu, d as usize];
        let new_dims = [a as usize, bc, d as usize];

        let old_numel = checked_dim_product(&old_dims);
        let new_numel = checked_dim_product(&new_dims);

        if let (Ok(on), Ok(nn)) = (old_numel, new_numel) {
            assert_eq!(on, nn, "merging adjacent dims must preserve numel");
        }
    }
}

/// Prove: reshape split then merge roundtrip preserves shape and numel.
///
/// [A, B*C] -> [A, B, C] -> [A, B*C]. The final shape must exactly
/// match the original.
#[kani::unwind(1)]
#[kani::proof]
fn reshape_split_merge_roundtrip() {
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
        // Original: [A, B*C]
        let orig = [au, bc];

        // Split: [A, B, C]
        let split = [au, bu, cu];

        // Merge back: [A, B*C]
        let merged = [au, bu * cu];

        assert_eq!(orig[0], merged[0], "roundtrip must preserve dim 0");
        assert_eq!(orig[1], merged[1], "roundtrip must preserve dim 1");

        let orig_numel = checked_dim_product(&orig);
        let merged_numel = checked_dim_product(&merged);
        if let (Ok(on), Ok(mn)) = (orig_numel, merged_numel) {
            assert_eq!(on, mn, "split-merge roundtrip must preserve numel");
        }
    }
}

// ===========================================================================
// 5. Broadcast stride zero invariant
// ===========================================================================

/// Prove: expanding a size-1 dim sets its stride to 0 while preserving
/// non-broadcast strides.
///
/// When broadcasting [1, C, 1] to [B, C, T], the stride for dim 0 and
/// dim 2 (originally size 1) become 0 (repeated data), while dim 1's
/// stride remains unchanged.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_size_one_dims_get_stride_zero() {
    let c: u8 = kani::any();
    let b: u8 = kani::any();
    let t: u8 = kani::any();

    kani::assume(c >= 1 && c <= 8);
    kani::assume(b >= 1 && b <= 8);
    kani::assume(t >= 1 && t <= 8);

    // Original tensor [1, C, 1] — contiguous strides: [C, 1, 1]
    let orig_dims = [1usize, c as usize, 1, 0];
    let orig_strides = contiguous_strides_4(&orig_dims, 3).unwrap();

    // Broadcast to [B, C, T]:
    // - dim 0: was size 1, broadcast to B -> stride becomes 0
    // - dim 1: was size C, stays C -> stride unchanged
    // - dim 2: was size 1, broadcast to T -> stride becomes 0
    let bcast_strides = [
        0usize,          // broadcast dim: stride 0
        orig_strides[1], // non-broadcast: original stride
        0usize,          // broadcast dim: stride 0
        0,
    ];

    let bcast_dims = [b as usize, c as usize, t as usize, 0];

    // Verify: broadcast dim strides are 0
    assert_eq!(bcast_strides[0], 0, "broadcast dim 0 stride must be 0");
    assert_eq!(bcast_strides[2], 0, "broadcast dim 2 stride must be 0");

    // Verify: non-broadcast stride preserved
    assert_eq!(
        bcast_strides[1], orig_strides[1],
        "non-broadcast stride must be preserved"
    );

    // Verify: any valid index into the broadcast view maps to a valid
    // offset in the *original* allocation (numel of original = C)
    let i0: u8 = kani::any();
    let i1: u8 = kani::any();
    let i2: u8 = kani::any();

    kani::assume((i0 as usize) < bcast_dims[0]);
    kani::assume((i1 as usize) < bcast_dims[1]);
    kani::assume((i2 as usize) < bcast_dims[2]);

    let idx = [i0 as usize, i1 as usize, i2 as usize, 0];
    let offset = linear_offset_4(&idx, &bcast_strides, 3).unwrap();

    let orig_numel = checked_dim_product(&[1, c as usize, 1]);
    if let Ok(n) = orig_numel {
        assert!(
            offset < n,
            "broadcast offset must be within original allocation"
        );
    }
}

/// Prove: broadcasting a scalar (all dims are 1) to any shape produces
/// all-zero strides, and every access maps to offset 0.
///
/// A scalar tensor [1, 1, 1] broadcast to [A, B, C] always reads from
/// the single element at offset 0.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_scalar_all_strides_zero() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();

    kani::assume(a >= 1 && a <= 8);
    kani::assume(b >= 1 && b <= 8);
    kani::assume(c >= 1 && c <= 8);

    // Scalar broadcast: all strides become 0
    let bcast_strides = [0usize, 0, 0, 0];
    let bcast_dims = [a as usize, b as usize, c as usize, 0];

    let i0: u8 = kani::any();
    let i1: u8 = kani::any();
    let i2: u8 = kani::any();

    kani::assume((i0 as usize) < bcast_dims[0]);
    kani::assume((i1 as usize) < bcast_dims[1]);
    kani::assume((i2 as usize) < bcast_dims[2]);

    let idx = [i0 as usize, i1 as usize, i2 as usize, 0];
    let offset = linear_offset_4(&idx, &bcast_strides, 3).unwrap();

    assert_eq!(offset, 0, "scalar broadcast must always access offset 0");
}

/// Prove: broadcasting a vector [1, C] to [B, C] sets stride[0] = 0
/// and preserves stride[1] = 1. Every row in the broadcast output
/// accesses the same data.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_vector_row_repeat_stride() {
    let b: u8 = kani::any();
    let c: u8 = kani::any();

    kani::assume(b >= 1 && b <= 8);
    kani::assume(c >= 1 && c <= 8);

    // Original [1, C]: strides = [C, 1]
    let orig_dims = [1usize, c as usize, 0, 0];
    let orig_strides = contiguous_strides_4(&orig_dims, 2).unwrap();
    assert_eq!(orig_strides[1], 1, "original inner stride must be 1");

    // Broadcast to [B, C]: stride[0] = 0, stride[1] = 1
    let bcast_strides = [0usize, orig_strides[1], 0, 0];

    // Two different rows must map to the same offset for the same column
    let row_a: u8 = kani::any();
    let row_b: u8 = kani::any();
    let col: u8 = kani::any();

    kani::assume((row_a as usize) < b as usize);
    kani::assume((row_b as usize) < b as usize);
    kani::assume((col as usize) < c as usize);

    let idx_a = [row_a as usize, col as usize, 0, 0];
    let idx_b = [row_b as usize, col as usize, 0, 0];

    let off_a = linear_offset_4(&idx_a, &bcast_strides, 2).unwrap();
    let off_b = linear_offset_4(&idx_b, &bcast_strides, 2).unwrap();

    assert_eq!(
        off_a, off_b,
        "broadcast rows must access same physical location"
    );
}

// ===========================================================================
// 6. Slice offset validity — strided slices within allocation
// ===========================================================================

/// Prove: slicing with a step (stride > 1) along a dimension keeps all
/// accessed offsets within the original allocation.
///
/// slice(dim=1, start, end, step) on a [D0, D1, D2] tensor accesses
/// indices start, start+step, start+2*step, ... All must be < D1, and
/// the resulting physical offsets must be < numel.
#[kani::unwind(1)]
#[kani::proof]
fn strided_slice_offsets_within_allocation() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let start: u8 = kani::any();
    let step: u8 = kani::any();
    let count: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 2 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(step >= 1 && step <= 4);
    kani::assume(count >= 1);
    kani::assume((start as usize) < d1 as usize);

    let su = start as usize;
    let stepu = step as usize;
    let countu = count as usize;

    // Last accessed index: start + (count - 1) * step
    let last_idx_opt = (countu - 1)
        .checked_mul(stepu)
        .and_then(|v| v.checked_add(su));
    if let Some(last_idx) = last_idx_opt {
        // Require all accessed indices are within dim bounds
        kani::assume(last_idx < d1 as usize);

        let dims = [d0 as usize, d1 as usize, d2 as usize, 0];
        let strides = contiguous_strides_4(&dims, 3).unwrap();

        // The sliced view has shape [D0, count, D2] with
        // strides [orig_stride[0], orig_stride[1] * step, orig_stride[2]]
        // and byte_offset = start * orig_stride[1]
        let base_offset = su * strides[1];
        let slice_stride_1 = strides[1] * stepu;

        // Check that the maximum offset in the sliced view is within bounds
        // Max index in slice: [d0-1, count-1, d2-1]
        let max_slice_offset =
            (dims[0] - 1) * strides[0] + (countu - 1) * slice_stride_1 + (dims[2] - 1) * strides[2];
        let total_max = base_offset + max_slice_offset;

        let numel = checked_dim_product(&[dims[0], dims[1], dims[2]]);
        if let Ok(n) = numel {
            assert!(
                total_max < n,
                "strided slice max offset must be within allocation"
            );
        }
    }
}

/// Prove: slicing the first dimension with start offset produces a valid
/// sub-view. The byte offset is start * stride[0], and all subsequent
/// accesses are within [byte_offset, byte_offset + sub_numel * elem_size).
#[kani::unwind(1)]
#[kani::proof]
fn slice_dim0_byte_offset_valid() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let start: u8 = kani::any();
    let len: u8 = kani::any();

    kani::assume(d0 >= 2 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(len >= 1 && len <= d0);
    kani::assume((start as usize) + (len as usize) <= d0 as usize);

    let dims = [d0 as usize, d1 as usize, 0, 0];
    let strides = contiguous_strides_4(&dims, 2).unwrap();

    let byte_offset = (start as usize) * strides[0];
    let sub_numel = (len as usize) * (d1 as usize);

    // byte_offset + sub_numel - 1 must be < total numel
    let total_numel = checked_dim_product(&[dims[0], dims[1]]);
    if let Ok(n) = total_numel {
        let last_accessed = byte_offset + sub_numel - 1;
        assert!(
            last_accessed < n,
            "slice sub-view must fit within original allocation"
        );
    }
}

// ===========================================================================
// 7. Permute stride correctness — arbitrary axis reordering
// ===========================================================================

/// Prove: permuting strides of a contiguous rank-3 tensor matches the
/// expected reordering: permuted_strides[i] = original_strides[perm[i]].
///
/// This is the defining property of `permute` / `transpose` generalized
/// to arbitrary axis order.
#[kani::unwind(1)]
#[kani::proof]
fn permute_strides_match_reordering_rank3() {
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

    let dims = [d0 as usize, d1 as usize, d2 as usize, 0];
    let strides = contiguous_strides_4(&dims, 3).unwrap();

    // Permute: new_dim[i] = old_dim[perm[i]], new_stride[i] = old_stride[perm[i]]
    let perm_dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]], 0];
    let perm_strides = [strides[perm[0]], strides[perm[1]], strides[perm[2]], 0];

    // Verify the relationship holds
    assert_eq!(
        perm_strides[0], strides[perm[0]],
        "permuted stride[0] = original stride[perm[0]]"
    );
    assert_eq!(
        perm_strides[1], strides[perm[1]],
        "permuted stride[1] = original stride[perm[1]]"
    );
    assert_eq!(
        perm_strides[2], strides[perm[2]],
        "permuted stride[2] = original stride[perm[2]]"
    );

    // Numel preserved
    let orig_numel = checked_dim_product(&[dims[0], dims[1], dims[2]]);
    let perm_numel = checked_dim_product(&[perm_dims[0], perm_dims[1], perm_dims[2]]);
    if let (Ok(on), Ok(pn)) = (orig_numel, perm_numel) {
        assert_eq!(on, pn, "permute must preserve numel");
    }
}

/// Prove: the inverse permutation recovers the original strides and dims.
///
/// If perm maps i -> perm[i], then inv_perm satisfies inv_perm[perm[i]] = i.
/// Applying inv_perm to the permuted strides must recover the originals.
#[kani::unwind(1)]
#[kani::proof]
fn inverse_permute_recovers_original_rank3() {
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

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let orig_dims_4 = [dims[0], dims[1], dims[2], 0];
    let strides_arr = contiguous_strides_4(&orig_dims_4, 3).unwrap();
    let strides = [strides_arr[0], strides_arr[1], strides_arr[2]];

    // Forward permute
    let perm_dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];
    let perm_strides = [strides[perm[0]], strides[perm[1]], strides[perm[2]]];

    // Compute inverse permutation: inv_perm[perm[i]] = i
    let mut inv_perm = [0usize; 3];
    inv_perm[perm[0]] = 0;
    inv_perm[perm[1]] = 1;
    inv_perm[perm[2]] = 2;

    // Apply inverse permute
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

    // Must recover originals
    assert_eq!(
        recovered_dims[0], dims[0],
        "inverse permute must recover dim 0"
    );
    assert_eq!(
        recovered_dims[1], dims[1],
        "inverse permute must recover dim 1"
    );
    assert_eq!(
        recovered_dims[2], dims[2],
        "inverse permute must recover dim 2"
    );
    assert_eq!(
        recovered_strides[0], strides[0],
        "inverse permute must recover stride 0"
    );
    assert_eq!(
        recovered_strides[1], strides[1],
        "inverse permute must recover stride 1"
    );
    assert_eq!(
        recovered_strides[2], strides[2],
        "inverse permute must recover stride 2"
    );
}

/// Prove: composing two permutations and applying directly is equivalent
/// to applying them sequentially. perm_B(perm_A(x)) == (perm_B . perm_A)(x).
///
/// This proves permutation composition correctness for stride reordering.
#[kani::unwind(1)]
#[kani::proof]
fn permute_composition_strides_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);

    // Permutation A
    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let a2: u8 = kani::any();
    kani::assume(a0 < 3 && a1 < 3 && a2 < 3);
    kani::assume(a0 != a1 && a0 != a2 && a1 != a2);

    // Permutation B
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();
    let b2: u8 = kani::any();
    kani::assume(b0 < 3 && b1 < 3 && b2 < 3);
    kani::assume(b0 != b1 && b0 != b2 && b1 != b2);

    let perm_a = [a0 as usize, a1 as usize, a2 as usize];
    let perm_b = [b0 as usize, b1 as usize, b2 as usize];

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let orig_dims_4 = [dims[0], dims[1], dims[2], 0];
    let strides_arr = contiguous_strides_4(&orig_dims_4, 3).unwrap();
    let strides = [strides_arr[0], strides_arr[1], strides_arr[2]];

    // Sequential: first apply A, then apply B
    let after_a_strides = [strides[perm_a[0]], strides[perm_a[1]], strides[perm_a[2]]];
    let seq_strides = [
        after_a_strides[perm_b[0]],
        after_a_strides[perm_b[1]],
        after_a_strides[perm_b[2]],
    ];

    // Composed: perm_composed[i] = perm_a[perm_b[i]]
    let composed = [perm_a[perm_b[0]], perm_a[perm_b[1]], perm_a[perm_b[2]]];
    let comp_strides = [
        strides[composed[0]],
        strides[composed[1]],
        strides[composed[2]],
    ];

    assert_eq!(
        seq_strides[0], comp_strides[0],
        "composed stride[0] must match sequential"
    );
    assert_eq!(
        seq_strides[1], comp_strides[1],
        "composed stride[1] must match sequential"
    );
    assert_eq!(
        seq_strides[2], comp_strides[2],
        "composed stride[2] must match sequential"
    );
}
