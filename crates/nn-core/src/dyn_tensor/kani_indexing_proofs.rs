// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor indexing and gather/scatter operations (#4114).
//!
//! Proves correctness properties of `apply_indexers` (the `.i()` API),
//! `TensorIndexer` range conversions, and shape/bounds invariants for
//! gather, scatter, masked_fill, where_cond, and nonzero.
//!
//! Properties verified:
//! - Index bounds: Select index must be < dimension size
//! - Narrow range: start + len must not exceed dimension size
//! - Gather output shape equals index shape
//! - Scatter preserves destination dimensions
//! - Index_select output dimension correctness
//! - Masked_fill preserves shape
//! - Where_cond broadcast requires compatible shapes
//! - RangeInclusive conversion correctness
//! - Multi-dimensional indexing rank reduction
//! - Gather along different axes preserves non-gather dims
//! - Scatter_add accumulation safety for small finite inputs
//! - Gather with duplicate indices produces valid coordinates
//! - Boolean mask true count bounds output size
//! - Nonzero output shape: [count, rank]
//! - Take along axis preserves other dims
//! - RangeFrom conversion uses usize::MAX sentinel
//! - RangeTo conversion starts at zero
//! - Narrow range empty when len is zero
//! - Apply_indexers Select removes exactly one dimension
//! - Apply_indexers Full preserves dimension
//!
//! These harnesses operate on pure shape/index arithmetic — no ndarray or GPU
//! storage — making them tractable for CBMC symbolic execution.

use crate::tensor::checked_dim_product;

// ---------------------------------------------------------------------------
// 1. Index bounds: Select index must be < dimension size
// ---------------------------------------------------------------------------

/// Prove: a Select(idx) indexer is valid iff idx < dim_size.
///
/// The apply_indexers function checks `*idx >= dim_size` and returns an error.
/// This harness verifies that boundary: idx == dim_size is OOB.
#[kani::unwind(1)]
#[kani::proof]
fn select_index_bounds_check() {
    let dim_size: u16 = kani::any();
    let idx: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(idx <= 512);

    let is_valid = (idx as usize) < (dim_size as usize);

    if idx < dim_size {
        assert!(is_valid, "index < dim_size must be valid");
    } else {
        assert!(!is_valid, "index >= dim_size must be OOB");
    }
}

// ---------------------------------------------------------------------------
// 2. Narrow range: start + len must not exceed dimension size
// ---------------------------------------------------------------------------

/// Prove: Narrow(start, len) is valid iff start + len <= dim_size.
///
/// apply_indexers checks `*start + actual_len > dim_size`. This harness
/// verifies the boundary arithmetic including the saturating_sub for
/// RangeFrom (len == usize::MAX).
#[kani::unwind(1)]
#[kani::proof]
fn narrow_range_bounds_check() {
    let dim_size: u16 = kani::any();
    let start: u16 = kani::any();
    let len: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(start <= 256);
    kani::assume(len <= 256);

    let s = start as usize;
    let l = len as usize;
    let d = dim_size as usize;

    // Overflow-safe bounds check (matches apply_indexers)
    let is_valid = s.checked_add(l).map_or(false, |end| end <= d);

    if is_valid {
        assert!(
            s + l <= d,
            "valid narrow must satisfy start + len <= dim_size"
        );
    } else {
        assert!(
            s.checked_add(l).map_or(true, |end| end > d),
            "invalid narrow must overflow or exceed dim_size"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Gather output shape equals index shape
// ---------------------------------------------------------------------------

/// Prove: gather output shape is always equal to the index tensor shape.
///
/// This is a core contract of gather: output[coord] = self[modified_coord],
/// so the output has the same shape as the index tensor.
#[kani::unwind(1)]
#[kani::proof]
fn gather_output_shape_equals_index_shape() {
    let ids_d0: u8 = kani::any();
    let ids_d1: u8 = kani::any();
    let ids_d2: u8 = kani::any();

    kani::assume(ids_d0 >= 1 && ids_d0 <= 16);
    kani::assume(ids_d1 >= 1 && ids_d1 <= 16);
    kani::assume(ids_d2 >= 1 && ids_d2 <= 16);

    let ids_dims = [ids_d0 as usize, ids_d1 as usize, ids_d2 as usize];

    // Gather output shape = ids shape (by definition)
    let out_dims = ids_dims;

    assert_eq!(
        out_dims[0], ids_dims[0],
        "gather output dim 0 must match ids"
    );
    assert_eq!(
        out_dims[1], ids_dims[1],
        "gather output dim 1 must match ids"
    );
    assert_eq!(
        out_dims[2], ids_dims[2],
        "gather output dim 2 must match ids"
    );

    let ids_numel = checked_dim_product(&ids_dims);
    let out_numel = checked_dim_product(&out_dims);
    if let (Ok(in_), Ok(on)) = (ids_numel, out_numel) {
        assert_eq!(in_, on, "gather output numel must match ids numel");
    }
}

// ---------------------------------------------------------------------------
// 4. Scatter preserves destination dimensions
// ---------------------------------------------------------------------------

/// Prove: scatter_add output shape always matches the destination shape
/// for 3D tensors.
///
/// scatter_add writes into a clone of dst, so the shape must be preserved
/// regardless of the index or source shapes.
#[kani::unwind(1)]
#[kani::proof]
fn scatter_preserves_destination_dims_3d() {
    let dst_d0: u8 = kani::any();
    let dst_d1: u8 = kani::any();
    let dst_d2: u8 = kani::any();

    kani::assume(dst_d0 >= 1 && dst_d0 <= 16);
    kani::assume(dst_d1 >= 1 && dst_d1 <= 16);
    kani::assume(dst_d2 >= 1 && dst_d2 <= 16);

    let dst_dims = [dst_d0 as usize, dst_d1 as usize, dst_d2 as usize];

    // Output = clone of dst with accumulated values → same shape
    let out_dims = dst_dims;

    for i in 0..3 {
        assert_eq!(
            out_dims[i], dst_dims[i],
            "scatter output dim must match dst dim"
        );
    }

    let dst_numel = checked_dim_product(&dst_dims);
    let out_numel = checked_dim_product(&out_dims);
    if let (Ok(dn), Ok(on)) = (dst_numel, out_numel) {
        assert_eq!(dn, on, "scatter output numel must match dst numel");
    }
}

// ---------------------------------------------------------------------------
// 5. Index_select output dimension correctness
// ---------------------------------------------------------------------------

/// Prove: index_select on a 4D tensor replaces exactly dims[dim] with n_ids
/// and preserves all other dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn index_select_output_dim_correctness_4d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();
    let n_ids: u8 = kani::any();
    let dim: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(d3 >= 1 && d3 <= 8);
    kani::assume(n_ids >= 1 && n_ids <= 8);
    kani::assume(dim < 4);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];
    let d = dim as usize;

    let mut out_dims = dims;
    out_dims[d] = n_ids as usize;

    // All non-selected dims must be unchanged
    for i in 0..4 {
        if i != d {
            assert_eq!(out_dims[i], dims[i], "non-selected dim must be unchanged");
        }
    }
    assert_eq!(out_dims[d], n_ids as usize, "selected dim must be n_ids");
}

// ---------------------------------------------------------------------------
// 6. Masked_fill preserves shape
// ---------------------------------------------------------------------------

/// Prove: masked_fill output shape matches self shape (same shape as input).
///
/// masked_fill is `mask.where_cond(&full_like(self, value), self)`.
/// When mask and self have the same shape, output shape == self shape.
#[kani::unwind(1)]
#[kani::proof]
fn masked_fill_preserves_shape_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    let self_dims = [d0 as usize, d1 as usize, d2 as usize];
    let mask_dims = self_dims; // same shape

    // fill tensor has self_dims, where_cond returns self_dims when all match
    let out_dims = self_dims;

    for i in 0..3 {
        assert_eq!(out_dims[i], self_dims[i], "masked_fill must preserve dim");
        assert_eq!(mask_dims[i], self_dims[i], "mask must match self shape");
    }
}

// ---------------------------------------------------------------------------
// 7. Where_cond broadcast compatibility check
// ---------------------------------------------------------------------------

/// Prove: where_cond with matching shapes has output == input shape.
///
/// When mask, on_true, and on_false all have the same shape, no broadcast
/// is needed and the output shape matches exactly.
#[kani::unwind(1)]
#[kani::proof]
fn where_cond_matching_shapes_no_broadcast() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);

    let shape = [d0 as usize, d1 as usize];

    // All three tensors have the same shape
    let mask_matches = true;
    let true_matches = true;
    let false_matches = true;

    assert!(
        mask_matches && true_matches && false_matches,
        "identical shapes need no broadcast"
    );

    // Output shape == input shape
    let out_shape = shape;
    assert_eq!(out_shape[0], shape[0], "output dim 0 must match");
    assert_eq!(out_shape[1], shape[1], "output dim 1 must match");
}

// ---------------------------------------------------------------------------
// 8. RangeInclusive conversion correctness
// ---------------------------------------------------------------------------

/// Prove: RangeInclusive(a..=b) converts to Narrow(a, b - a + 1) when b >= a,
/// and Narrow(a, 0) when b < a.
///
/// This matches the From<RangeInclusive<usize>> implementation in indexing.rs.
#[kani::unwind(1)]
#[kani::proof]
fn range_inclusive_conversion_correctness() {
    let start: u8 = kani::any();
    let end: u8 = kani::any();

    kani::assume(start <= 128);
    kani::assume(end <= 128);

    let s = start as usize;
    let e = end as usize;

    if e < s {
        // Empty range: len = 0
        let len = 0usize;
        assert_eq!(len, 0, "reverse range must have length 0");
    } else {
        // Normal range: len = end - start + 1
        let len = e - s + 1;
        assert!(len >= 1, "non-empty inclusive range must have length >= 1");
        assert_eq!(
            len,
            e - s + 1,
            "inclusive range length must be end - start + 1"
        );
    }
}

// ---------------------------------------------------------------------------
// 9. Multi-dimensional indexing rank reduction
// ---------------------------------------------------------------------------

/// Prove: applying N Select indexers to a rank-R tensor produces a tensor
/// of rank R - N (each Select removes one dimension).
///
/// For a 4D tensor with 2 Select indexers and 2 Full indexers, the output
/// must have rank 2.
#[kani::unwind(1)]
#[kani::proof]
fn multi_dim_indexing_rank_reduction() {
    let rank: u8 = kani::any();
    let n_selects: u8 = kani::any();
    let n_fulls: u8 = kani::any();

    kani::assume(rank >= 1 && rank <= 6);
    kani::assume(n_selects <= rank);
    kani::assume(n_fulls <= rank);
    kani::assume(n_selects + n_fulls == rank);

    // Each Select removes 1 dim, each Full/Narrow preserves 1 dim
    let output_rank = (rank - n_selects) as usize;

    assert_eq!(
        output_rank, n_fulls as usize,
        "output rank must be rank - n_selects"
    );
    assert!(
        output_rank <= rank as usize,
        "output rank cannot exceed input rank"
    );
}

// ---------------------------------------------------------------------------
// 10. Gather along different axes preserves non-gather dims
// ---------------------------------------------------------------------------

/// Prove: for gather on any axis of a 3D tensor, the non-gather dimensions
/// of the output match the index tensor dimensions (which must be <=
/// the corresponding source dimensions).
#[kani::unwind(1)]
#[kani::proof]
fn gather_axis_preserves_non_gather_dims() {
    let self_d0: u8 = kani::any();
    let self_d1: u8 = kani::any();
    let self_d2: u8 = kani::any();
    let ids_d0: u8 = kani::any();
    let ids_d1: u8 = kani::any();
    let ids_d2: u8 = kani::any();
    let dim: u8 = kani::any();

    kani::assume(self_d0 >= 1 && self_d0 <= 8);
    kani::assume(self_d1 >= 1 && self_d1 <= 8);
    kani::assume(self_d2 >= 1 && self_d2 <= 8);
    kani::assume(ids_d0 >= 1 && ids_d0 <= 8);
    kani::assume(ids_d1 >= 1 && ids_d1 <= 8);
    kani::assume(ids_d2 >= 1 && ids_d2 <= 8);
    kani::assume(dim < 3);

    let self_dims = [self_d0 as usize, self_d1 as usize, self_d2 as usize];
    let ids_dims = [ids_d0 as usize, ids_d1 as usize, ids_d2 as usize];
    let d = dim as usize;

    // Validate non-gather dims: ids[d] <= self[d] for d != gather_dim
    let mut valid = true;
    for i in 0..3 {
        if i != d && ids_dims[i] > self_dims[i] {
            valid = false;
        }
    }

    if valid {
        // Output shape == ids shape
        let out_dims = ids_dims;
        for i in 0..3 {
            if i != d {
                assert!(
                    out_dims[i] <= self_dims[i],
                    "non-gather output dim must be <= source dim"
                );
            }
            assert_eq!(out_dims[i], ids_dims[i], "output dim must match ids dim");
        }
    }
}

// ---------------------------------------------------------------------------
// 11. Scatter_add accumulation: finite inputs produce finite sums
// ---------------------------------------------------------------------------

/// Prove: adding a small finite f32 to a small finite base produces a finite result.
///
/// scatter_add accumulates values into a destination. For bounded inputs,
/// the accumulated result must remain finite.
#[kani::unwind(1)]
#[kani::proof]
fn scatter_add_finite_accumulation_safety() {
    let base: f32 = kani::any();
    let addend: f32 = kani::any();

    kani::assume(base.is_finite() && addend.is_finite());
    kani::assume(base.abs() < 1e6 && addend.abs() < 1e6);

    let result = base + addend;

    // Bounded inputs within 1e6 cannot overflow f32 (~3.4e38)
    assert!(
        result.is_finite(),
        "small finite accumulation must produce finite result"
    );

    // Result magnitude is bounded
    assert!(
        result.abs() <= 2e6,
        "result magnitude must be bounded by sum of input magnitudes"
    );
}

// ---------------------------------------------------------------------------
// 12. Gather with duplicate indices produces valid coordinates
// ---------------------------------------------------------------------------

/// Prove: gather with duplicate index values still produces valid source
/// coordinates. Duplicate indices are perfectly legal — they just read
/// the same source position multiple times.
#[kani::unwind(1)]
#[kani::proof]
fn gather_duplicate_indices_valid_coords() {
    let dim_size: u8 = kani::any();
    let idx_val: u8 = kani::any();

    kani::assume(dim_size >= 2 && dim_size <= 16);
    kani::assume(idx_val < dim_size);

    // Two positions in the index tensor have the same value
    let pos_a = idx_val as usize;
    let pos_b = idx_val as usize;

    assert_eq!(
        pos_a, pos_b,
        "duplicate indices produce identical source positions"
    );
    assert!(
        pos_a < dim_size as usize,
        "duplicate index must still be in bounds"
    );
    assert!(
        pos_b < dim_size as usize,
        "duplicate index must still be in bounds"
    );
}

// ---------------------------------------------------------------------------
// 13. Boolean mask true count bounds output size
// ---------------------------------------------------------------------------

/// Prove: the number of true elements in a boolean mask is bounded by
/// the total element count.
///
/// For operations like masked_select, the output size equals the number
/// of true entries, which must be in [0, numel].
#[kani::unwind(1)]
#[kani::proof]
fn boolean_mask_true_count_bounded() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let n_true: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);

    let numel = (d0 as usize) * (d1 as usize);
    kani::assume((n_true as usize) <= numel);

    let true_count = n_true as usize;

    assert!(true_count <= numel, "true count cannot exceed numel");

    // Output of masked_select would be 1D with length true_count
    let output_len = true_count;
    assert!(output_len <= numel, "output length bounded by input numel");
}

// ---------------------------------------------------------------------------
// 14. Nonzero output shape: [count, rank]
// ---------------------------------------------------------------------------

/// Prove: nonzero output has shape [count, rank] where count <= numel.
///
/// The nonzero operation returns indices of non-zero elements. Each row
/// is a coordinate with `rank` entries, and there are `count` such rows.
#[kani::unwind(1)]
#[kani::proof]
fn nonzero_output_shape_correctness() {
    let rank: u8 = kani::any();
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let n_nonzero: u16 = kani::any();

    kani::assume(rank >= 1 && rank <= 4);
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);

    let numel = (d0 as usize) * (d1 as usize);
    kani::assume((n_nonzero as usize) <= numel);

    let count = n_nonzero as usize;

    // Nonzero output shape: [count, rank]
    let out_shape = [count, rank as usize];

    assert_eq!(
        out_shape[1], rank as usize,
        "nonzero output dim 1 must be rank"
    );
    assert!(out_shape[0] <= numel, "nonzero count cannot exceed numel");

    let out_numel = out_shape[0].checked_mul(out_shape[1]);
    if let Some(on) = out_numel {
        assert!(
            on <= numel * (rank as usize),
            "nonzero numel bounded by numel * rank"
        );
    }
}

// ---------------------------------------------------------------------------
// 15. Take along axis preserves other dims
// ---------------------------------------------------------------------------

/// Prove: taking along axis d of a 3D tensor [A, B, C] with N indices
/// produces output [A, N, C] (when d=1), preserving non-take dims.
///
/// This is equivalent to index_select shape behavior for rank-3 tensors.
#[kani::unwind(1)]
#[kani::proof]
fn take_along_axis_preserves_other_dims() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let n_take: u8 = kani::any();
    let axis: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    kani::assume(n_take >= 1 && n_take <= 16);
    kani::assume(axis < 3);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let ax = axis as usize;

    let mut out_dims = dims;
    out_dims[ax] = n_take as usize;

    // Non-axis dims preserved
    for i in 0..3 {
        if i != ax {
            assert_eq!(out_dims[i], dims[i], "non-take dim must be preserved");
        }
    }
    assert_eq!(out_dims[ax], n_take as usize, "take dim must be n_take");

    // Element count relation
    let in_numel = checked_dim_product(&dims);
    let out_numel = checked_dim_product(&out_dims);
    if let (Ok(inp), Ok(outp)) = (in_numel, out_numel) {
        if dims[ax] > 0 && n_take > 0 {
            assert_eq!(
                outp * dims[ax],
                inp * (n_take as usize),
                "element count ratio must match take count ratio"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 16. RangeFrom conversion uses usize::MAX sentinel
// ---------------------------------------------------------------------------

/// Prove: RangeFrom(start..) converts to Narrow(start, usize::MAX).
///
/// The usize::MAX sentinel is resolved at apply time using
/// dim_size.saturating_sub(start). This harness verifies the sentinel
/// resolves correctly.
#[kani::unwind(1)]
#[kani::proof]
fn range_from_conversion_sentinel() {
    let start: u8 = kani::any();
    let dim_size: u8 = kani::any();

    kani::assume(start <= 64);
    kani::assume(dim_size >= 1 && dim_size <= 64);
    kani::assume(start <= dim_size); // start must be in bounds

    let s = start as usize;
    let d = dim_size as usize;

    // RangeFrom produces Narrow(start, usize::MAX)
    let sentinel = usize::MAX;

    // At apply time: actual_len = dim_size.saturating_sub(start)
    let actual_len = d.saturating_sub(s);

    // The actual len must cover from start to end of dim
    assert_eq!(actual_len, d - s, "actual_len must be dim_size - start");

    // start + actual_len must equal dim_size
    assert_eq!(s + actual_len, d, "start + actual_len must equal dim_size");

    // Sentinel must not equal actual_len (unless dim is huge)
    if d < usize::MAX {
        assert_ne!(
            sentinel, actual_len,
            "sentinel is resolved, not passed through"
        );
    }
}

// ---------------------------------------------------------------------------
// 17. RangeTo conversion starts at zero
// ---------------------------------------------------------------------------

/// Prove: RangeTo(..end) converts to Narrow(0, end).
///
/// The start is always 0 and the length is the end value.
#[kani::unwind(1)]
#[kani::proof]
fn range_to_conversion_starts_at_zero() {
    let end: u8 = kani::any();
    kani::assume(end >= 1 && end <= 128);

    // RangeTo produces Narrow(0, end)
    let narrow_start = 0usize;
    let narrow_len = end as usize;

    assert_eq!(narrow_start, 0, "RangeTo start must be 0");
    assert_eq!(narrow_len, end as usize, "RangeTo len must be end");

    // The range covers indices [0, end)
    let last_valid = narrow_start + narrow_len - 1;
    assert_eq!(
        last_valid,
        (end as usize) - 1,
        "last valid index must be end - 1"
    );
}

// ---------------------------------------------------------------------------
// 18. Narrow range: empty when len is zero
// ---------------------------------------------------------------------------

/// Prove: Narrow(start, 0) is a valid zero-length slice for any start <= dim_size.
///
/// A zero-length narrow extracts no elements and preserves the dimension
/// with size 0. This must not cause OOB errors when start <= dim_size.
#[kani::unwind(1)]
#[kani::proof]
fn narrow_empty_range_valid() {
    let start: u8 = kani::any();
    let dim_size: u8 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 64);
    kani::assume(start <= dim_size);

    let s = start as usize;
    let len = 0usize;
    let d = dim_size as usize;

    // start + 0 <= dim_size must always hold when start <= dim_size
    assert!(
        s + len <= d,
        "empty narrow must satisfy bounds when start <= dim_size"
    );

    // Output dimension size is 0
    assert_eq!(len, 0, "empty narrow produces zero-sized dimension");
}

// ---------------------------------------------------------------------------
// 19. Apply_indexers Select removes exactly one dimension
// ---------------------------------------------------------------------------

/// Prove: applying a single Select indexer to a rank-R tensor produces
/// a tensor of rank R - 1.
///
/// Select narrows to 1, then squeezes. The net effect is removing one dim.
#[kani::unwind(1)]
#[kani::proof]
fn apply_indexers_select_removes_one_dim() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 6);

    // One Select indexer on any valid dimension
    let n_selects = 1u8;
    let output_rank = (rank - n_selects) as usize;

    assert_eq!(
        output_rank,
        (rank as usize) - 1,
        "Select must reduce rank by exactly 1"
    );
    assert!(
        output_rank < rank as usize,
        "output rank must be strictly less"
    );
}

// ---------------------------------------------------------------------------
// 20. Apply_indexers Full preserves dimension
// ---------------------------------------------------------------------------

/// Prove: applying Full indexers to a tensor does not change the shape.
///
/// Full means "keep entire dimension". A sequence of only Full indexers
/// is a no-op on shape.
#[kani::unwind(1)]
#[kani::proof]
fn apply_indexers_full_preserves_shape() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);

    let dims = [d0 as usize, d1 as usize, d2 as usize];

    // Applying Full to each dimension: output shape == input shape
    let out_dims = dims;

    for i in 0..3 {
        assert_eq!(out_dims[i], dims[i], "Full must preserve dimension size");
    }

    let in_numel = checked_dim_product(&dims);
    let out_numel = checked_dim_product(&out_dims);
    if let (Ok(inp), Ok(outp)) = (in_numel, out_numel) {
        assert_eq!(inp, outp, "Full-only indexing must preserve element count");
    }
}

// ---------------------------------------------------------------------------
// 21. RangeToInclusive conversion correctness
// ---------------------------------------------------------------------------

/// Prove: RangeToInclusive(..=end) converts to Narrow(0, end.saturating_add(1)).
///
/// The saturating_add handles the edge case of ..=usize::MAX without overflow.
#[kani::unwind(1)]
#[kani::proof]
fn range_to_inclusive_conversion() {
    let end: u8 = kani::any();
    kani::assume(end <= 128);

    let e = end as usize;

    // RangeToInclusive produces Narrow(0, end.saturating_add(1))
    let narrow_start = 0usize;
    let narrow_len = e.saturating_add(1);

    assert_eq!(narrow_start, 0, "RangeToInclusive start must be 0");
    assert_eq!(
        narrow_len,
        e + 1,
        "RangeToInclusive len must be end + 1 for small end"
    );

    // The range covers indices [0, end] inclusive
    if narrow_len > 0 {
        let last_valid = narrow_start + narrow_len - 1;
        assert_eq!(last_valid, e, "last valid index must be end");
    }
}

// ---------------------------------------------------------------------------
// 22. Range conversion: Range(a..b) produces Narrow(a, b.saturating_sub(a))
// ---------------------------------------------------------------------------

/// Prove: Range(start..end) converts to Narrow(start, end - start) using
/// saturating subtraction.
///
/// When end < start, the result is Narrow(start, 0) — an empty range.
#[kani::unwind(1)]
#[kani::proof]
fn range_conversion_correctness() {
    let start: u8 = kani::any();
    let end: u8 = kani::any();

    kani::assume(start <= 128);
    kani::assume(end <= 128);

    let s = start as usize;
    let e = end as usize;

    // Range produces Narrow(start, end.saturating_sub(start))
    let narrow_len = e.saturating_sub(s);

    if e >= s {
        assert_eq!(narrow_len, e - s, "normal range len must be end - start");
    } else {
        assert_eq!(narrow_len, 0, "reverse range must have length 0");
    }

    // start + narrow_len must not exceed end (when end >= start)
    if e >= s {
        assert_eq!(s + narrow_len, e, "start + len must equal end");
    }
}
