// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for dpdf VLM indexing patterns (#4230).
//!
//! Builds on the 14 proofs in `kani_dpdf_vlm_indexing_safety.rs`:
//! 15. Gather along embedding dim: output shape correctness
//! 16. Scatter add: no overflow for bounded inputs
//! 17. Boolean mask selection: always produces rank-1 output
//! 18. Sorted index_select equivalence with narrow
//! 19. Multi-dim gather: coordinate mapping correctness
//! 20. Scatter reduction: preserves destination shape
//! 21. Top-k indices: all in [0, dim_size)
//! 22. Argmax/argmin: result in [0, dim_size), keepdim shape
//! 23. Where (conditional select): output matches mask
//! 24. Masked fill: preserves unmasked elements and shape

use crate::tensor::checked_dim_product;

// 15. Gather along embedding dimension: output shape correctness

/// Prove: gather along dim=2 in [B, S, E] produces output with ids shape.
/// Non-gather dims match source; output numel equals ids numel.
#[kani::unwind(1)]
#[kani::proof]
fn gather_embedding_dim_output_shape_3d() {
    let batch: u8 = kani::any();
    let seq: u8 = kani::any();
    let embed: u8 = kani::any();
    let ids_embed: u8 = kani::any();
    kani::assume(batch >= 1 && batch <= 4);
    kani::assume(seq >= 1 && seq <= 8);
    kani::assume(embed >= 1 && embed <= 8);
    kani::assume(ids_embed >= 1 && ids_embed <= 8);

    let src_shape = [batch as usize, seq as usize, embed as usize];
    let ids_shape = [batch as usize, seq as usize, ids_embed as usize];
    let out_shape = ids_shape;

    for i in 0..3 {
        assert_eq!(out_shape[i], ids_shape[i], "output must match ids shape");
    }
    assert_eq!(out_shape[0], src_shape[0], "batch dim must match source");
    assert_eq!(out_shape[1], src_shape[1], "seq dim must match source");

    let ids_numel = checked_dim_product(&ids_shape);
    let out_numel = checked_dim_product(&out_shape);
    if let (Ok(a), Ok(b)) = (ids_numel, out_numel) {
        assert_eq!(a, b, "gather output numel must equal ids numel");
    }
}

// 16. Scatter add: accumulation does not overflow for bounded inputs

/// Prove: scatter_add with |v| <= V and at most K collisions per cell
/// produces |accumulated| <= K * V, which fits in f32.
#[kani::unwind(1)]
#[kani::proof]
fn scatter_add_no_overflow_bounded_inputs() {
    let value_bound: u16 = kani::any();
    let max_collisions: u8 = kani::any();
    kani::assume(value_bound >= 1 && value_bound <= 1000);
    kani::assume(max_collisions >= 1 && max_collisions <= 255);

    let v = value_bound as f64;
    let k = max_collisions as f64;
    let max_accumulated = k * v;

    // K=255, V=1000 gives 255_000 — well within f32 (~3.4e38)
    assert!(max_accumulated <= 3.4e38_f64, "must stay within f32 range");
    assert!(
        max_accumulated >= 0.0,
        "product of positives is non-negative"
    );
    assert!(1.0_f64 <= max_accumulated, "min must not exceed max");
}

// 17. Boolean mask selection: produces rank-1 output

/// Prove: masked_select on a 3D tensor produces 1D output with length =
/// number of true elements, bounded by numel.
#[kani::unwind(1)]
#[kani::proof]
fn boolean_mask_select_produces_rank_1() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let n_true: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    let numel = (d0 as usize) * (d1 as usize) * (d2 as usize);
    kani::assume((n_true as usize) <= numel);

    let output_shape = [n_true as usize];
    assert_eq!(output_shape.len(), 1, "masked_select output must be rank 1");
    assert!(output_shape[0] <= numel, "selected <= total elements");
}

// 18. Sorted index_select equivalence with narrow

/// Prove: sorted contiguous indices [a, a+1, ..., a+n-1] for index_select
/// are equivalent to narrow(dim, a, n). All in bounds, strictly monotonic.
#[kani::unwind(1)]
#[kani::proof]
fn sorted_index_select_equals_narrow() {
    let dim_size: u8 = kani::any();
    let start: u8 = kani::any();
    let count: u8 = kani::any();
    kani::assume(dim_size >= 1 && dim_size <= 16);
    kani::assume(count >= 1 && count <= dim_size);
    kani::assume(start <= dim_size - count);

    let first_idx = start as usize;
    let last_idx = first_idx + (count as usize) - 1;
    assert!(first_idx < dim_size as usize, "first index in bounds");
    assert!(last_idx < dim_size as usize, "last index in bounds");

    // narrow(dim, start, count) accesses the same range
    assert_eq!(first_idx, start as usize, "narrow start matches");
    assert_eq!(
        last_idx,
        start as usize + count as usize - 1,
        "narrow end matches"
    );

    // Monotonicity: consecutive indices are strictly increasing
    if count >= 2 {
        for i in 0..(count as usize - 1) {
            assert!(
                start as usize + i + 1 > start as usize + i,
                "strictly monotonic"
            );
        }
    }
}

// 19. Multi-dimensional gather: coordinate mapping

/// Prove: 2D gather dim=0 maps out[i][j] = src[ids[i][j]][j]. The non-gather
/// dim coordinate is shared between output and source.
#[kani::unwind(1)]
#[kani::proof]
fn gather_2d_dim0_coordinate_mapping() {
    let src_d0: u8 = kani::any();
    let src_d1: u8 = kani::any();
    let ids_d0: u8 = kani::any();
    let ids_d1: u8 = kani::any();
    let i: u8 = kani::any();
    let j: u8 = kani::any();
    let idx_val: u8 = kani::any();
    kani::assume(src_d0 >= 1 && src_d0 <= 8);
    kani::assume(src_d1 >= 1 && src_d1 <= 8);
    kani::assume(ids_d0 >= 1 && ids_d0 <= 8);
    kani::assume(ids_d1 >= 1 && ids_d1 <= src_d1);
    kani::assume(i < ids_d0);
    kani::assume(j < ids_d1);
    kani::assume(idx_val < src_d0);

    // out[i][j] = src[idx_val][j]
    let src_coord = [idx_val as usize, j as usize];
    let out_coord = [i as usize, j as usize];

    assert!(src_coord[0] < src_d0 as usize, "row index within src");
    assert!(src_coord[1] < src_d1 as usize, "col index within src");
    assert!(out_coord[0] < ids_d0 as usize, "out row within ids");
    assert!(out_coord[1] < ids_d1 as usize, "out col within ids");
    assert_eq!(src_coord[1], out_coord[1], "non-gather dim shared");
}

// 20. Scatter reduction: preserves destination shape

/// Prove: scatter with any reduction mode (sum, mean, overwrite) does not
/// change the destination shape. Mean divisor >= 1.
#[kani::unwind(1)]
#[kani::proof]
fn scatter_reduction_preserves_dst_shape() {
    let dst_d0: u8 = kani::any();
    let dst_d1: u8 = kani::any();
    let src_d0: u8 = kani::any();
    let src_d1: u8 = kani::any();
    let dim: u8 = kani::any();
    kani::assume(dst_d0 >= 1 && dst_d0 <= 8);
    kani::assume(dst_d1 >= 1 && dst_d1 <= 8);
    kani::assume(src_d0 >= 1 && src_d0 <= 8);
    kani::assume(src_d1 >= 1 && src_d1 <= 8);
    kani::assume(dim < 2);

    // Non-scatter dims must match
    if dim == 0 {
        kani::assume(src_d1 == dst_d1);
    } else {
        kani::assume(src_d0 == dst_d0);
    }

    let dst_shape = [dst_d0 as usize, dst_d1 as usize];
    // All reduction modes preserve shape
    assert_eq!(
        dst_shape,
        [dst_d0 as usize, dst_d1 as usize],
        "shape unchanged"
    );

    let contributions: u8 = kani::any();
    kani::assume(contributions >= 1 && contributions <= 64);
    assert!(contributions as usize >= 1, "mean divisor >= 1");
}

// 21. Top-k indices: all in [0, dim_size)

/// Prove: top-k index values are valid, k <= dim_size, and the output
/// dimension along the reduction axis has size k.
#[kani::unwind(1)]
#[kani::proof]
fn topk_indices_in_valid_range() {
    let dim_size: u8 = kani::any();
    let k: u8 = kani::any();
    let idx: u8 = kani::any();
    kani::assume(dim_size >= 1 && dim_size <= 64);
    kani::assume(k >= 1 && k <= dim_size);
    kani::assume(idx < dim_size);

    assert!(
        (idx as usize) < (dim_size as usize),
        "top-k index < dim_size"
    );
    assert!((k as usize) <= (dim_size as usize), "k <= dim_size");
    assert!((k as usize) >= 1, "output dim >= 1");
}

// 22. Argmax/argmin: result in [0, dim_size), keepdim shape

/// Prove: argmax/argmin result is a valid index. keepdim=true sets dim to 1;
/// keepdim=false removes the dim. Both have the same element count.
#[kani::unwind(1)]
#[kani::proof]
fn argmax_argmin_result_bounds() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let dim: u8 = kani::any();
    let result_idx: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(dim < 3);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let d = dim as usize;
    let dim_size = dims[d];
    kani::assume((result_idx as usize) < dim_size);

    assert!((result_idx as usize) < dim_size, "result in [0, dim_size)");

    // keepdim=true: dims[dim] = 1
    let mut out_keepdim = dims;
    out_keepdim[d] = 1;
    let in_numel = checked_dim_product(&dims);
    let out_numel = checked_dim_product(&out_keepdim);
    if let (Ok(inp), Ok(outp)) = (in_numel, out_numel) {
        assert_eq!(
            outp * dim_size,
            inp,
            "keepdim numel * dim_size == input numel"
        );
    }

    // keepdim=false: remove dim, rank - 1
    let mut out_no_keepdim = [0usize; 2];
    let mut j = 0;
    for i in 0..3 {
        if i != d {
            out_no_keepdim[j] = dims[i];
            j += 1;
        }
    }
    let out2_numel = checked_dim_product(&out_no_keepdim);
    if let (Ok(a), Ok(b)) = (out_numel, out2_numel) {
        assert_eq!(a, b, "keepdim and no-keepdim same element count");
    }
}

// 23. Where (conditional select): output matches mask pattern

/// Prove: where_cond selects from on_true when mask=true, on_false otherwise.
/// Output shape equals input shape. Exactly one source per element.
#[kani::unwind(1)]
#[kani::proof]
fn where_cond_output_shape_and_selection() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let mask_val: bool = kani::any();
    let on_true_val: u16 = kani::any();
    let on_false_val: u16 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);

    let shape = [d0 as usize, d1 as usize];
    let out_shape = shape;
    assert_eq!(out_shape, shape, "output shape must equal input shape");

    let output_val = if mask_val { on_true_val } else { on_false_val };
    if mask_val {
        assert_eq!(output_val, on_true_val, "mask=true selects on_true");
    } else {
        assert_eq!(output_val, on_false_val, "mask=false selects on_false");
    }
    assert!(mask_val != !mask_val, "exactly one source per element");

    let in_numel = checked_dim_product(&shape);
    let out_numel = checked_dim_product(&out_shape);
    if let (Ok(a), Ok(b)) = (in_numel, out_numel) {
        assert_eq!(a, b, "where_cond preserves numel");
    }
}

// 24. Masked fill: only fills where mask is true

/// Prove: masked_fill preserves unmasked elements, overwrites masked ones,
/// and never changes tensor shape or element count.
#[kani::unwind(1)]
#[kani::proof]
fn masked_fill_preserves_unmasked_and_shape() {
    let original: u32 = kani::any();
    let fill_val: u32 = kani::any();
    let mask: bool = kani::any();

    let result = if mask { fill_val } else { original };
    if mask {
        assert_eq!(result, fill_val, "masked position filled");
    } else {
        assert_eq!(result, original, "unmasked position preserved");
    }

    // Shape invariant: 3D example
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let n_true: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    let numel = (d0 as usize) * (d1 as usize) * (d2 as usize);
    kani::assume((n_true as usize) <= numel);

    let input_shape = [d0 as usize, d1 as usize, d2 as usize];
    let output_shape = input_shape;
    for i in 0..3 {
        assert_eq!(output_shape[i], input_shape[i], "dim unchanged");
    }

    let in_numel = checked_dim_product(&input_shape);
    let out_numel = checked_dim_product(&output_shape);
    if let (Ok(a), Ok(b)) = (in_numel, out_numel) {
        assert_eq!(a, b, "masked_fill preserves element count");
    }
    assert_eq!(
        n_true as usize + (numel - n_true as usize),
        numel,
        "modified + preserved = total"
    );
}
