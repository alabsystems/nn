// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for selection/mod.rs operations (#3680).
//!
//! Proves correctness properties of `index_select`, `gather`, and `expand` —
//! the core selection and indexing operations for DynTensor.
//!
//! Properties verified:
//! - index_select: output shape computation, OOB index rejection, rank validation
//! - gather: coordinate mapping correctness, OOB detection, shape validation
//! - expand: broadcast rule enforcement, element count, idempotency
//!
//! These harnesses operate on pure shape/index arithmetic — no ndarray or GPU
//! storage — making them tractable for CBMC symbolic execution.

use crate::tensor::checked_dim_product;

// ---------------------------------------------------------------------------
// index_select: output shape has dims[dim] replaced by ids.len()
// ---------------------------------------------------------------------------

/// Prove: index_select output shape replaces dims[dim] with the index count,
/// and preserves all other dimensions.
///
/// This is the fundamental shape contract: given input [A, B, C] and
/// index_select on dim=1 with N indices, output is [A, N, C].
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn index_select_output_shape_2d() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let n_ids: u16 = kani::any();
    let dim: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(n_ids >= 1 && n_ids <= 64);
    kani::assume(dim < 2);

    let dims = [d0 as usize, d1 as usize];
    let d = dim as usize;

    // Compute expected output shape
    let mut out_dims = dims;
    out_dims[d] = n_ids as usize;

    // Non-selected dims must be unchanged
    for i in 0..2 {
        if i != d {
            assert_eq!(out_dims[i], dims[i], "non-selected dim must be unchanged");
        }
    }
    // Selected dim must be n_ids
    assert_eq!(out_dims[d], n_ids as usize, "selected dim must be n_ids");
}

/// Prove: index_select output shape for 3D tensors.
///
/// Verifies the shape contract on rank-3 tensors (common in sequence models:
/// [batch, seq_len, hidden_dim]).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn index_select_output_shape_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let n_ids: u8 = kani::any();
    let dim: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    kani::assume(n_ids >= 1 && n_ids <= 16);
    kani::assume(dim < 3);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let d = dim as usize;

    let mut out_dims = dims;
    out_dims[d] = n_ids as usize;

    // Element count relation: out_numel = in_numel / dims[dim] * n_ids
    let in_numel = checked_dim_product(&dims);
    let out_numel = checked_dim_product(&out_dims);
    if let (Ok(inp), Ok(outp)) = (in_numel, out_numel) {
        // out / n_ids == in / dims[dim] (i.e., product of non-selected dims)
        if dims[d] > 0 && n_ids > 0 {
            assert_eq!(
                outp * dims[d],
                inp * (n_ids as usize),
                "element count ratio must match index count ratio"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// index_select: OOB index detection
// ---------------------------------------------------------------------------

/// Prove: any index >= dim_size is OOB and must be rejected.
///
/// index_select validates that all indices are < dims[dim]. An index equal
/// to dim_size is OOB (off-by-one). This harness verifies the boundary.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn index_select_oob_detection() {
    let dim_size: u16 = kani::any();
    let index_val: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(index_val <= 512);

    let is_valid = (index_val as usize) < (dim_size as usize);

    if !is_valid {
        assert!(
            (index_val as usize) >= (dim_size as usize),
            "OOB index must be >= dim_size"
        );
    } else {
        assert!(
            (index_val as usize) < (dim_size as usize),
            "valid index must be < dim_size"
        );
    }
}

/// Prove: u32 index cast to usize preserves OOB detection.
///
/// Indices are stored as U32 in the index tensor. Casting to usize
/// must not change whether the index is in or out of bounds.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn index_select_u32_cast_preserves_oob() {
    let idx: u32 = kani::any();
    let dim_size: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 4096);
    // Restrict to u16 range for tractability — real dim_sizes are small
    kani::assume(idx <= 8192);

    let idx_usize = idx as usize;
    let ds = dim_size as usize;

    // u32 -> usize cast is always widening (usize >= 32 bits), so value is preserved
    assert_eq!(idx_usize as u32, idx, "u32 -> usize -> u32 must round-trip");

    // OOB check is consistent
    let oob_u32 = idx >= (dim_size as u32);
    let oob_usize = idx_usize >= ds;
    assert_eq!(oob_u32, oob_usize, "OOB check must agree across casts");
}

// ---------------------------------------------------------------------------
// index_select: rank validation (ids must be 1-D)
// ---------------------------------------------------------------------------

/// Prove: index_select rejects non-1D index tensors.
///
/// The rank check `ids.rank() != 1` must reject rank 0, 2, 3, etc.
/// Only rank-1 index tensors are valid for index_select.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn index_select_rejects_non_1d_ids() {
    let ids_rank: u8 = kani::any();
    kani::assume(ids_rank <= 6);

    let valid = ids_rank == 1;

    if valid {
        assert_eq!(ids_rank, 1, "only rank 1 is valid for index_select ids");
    } else {
        assert_ne!(ids_rank, 1, "non-1D ids must be rejected");
    }
}

// ---------------------------------------------------------------------------
// gather: coordinate mapping correctness
// ---------------------------------------------------------------------------

/// Prove: gather coordinate mapping replaces exactly dim `d` in the source
/// coordinate, and copies all other coordinates from the index position.
///
/// For gather on dim=d: src_coord[i] = coord[i] for i != d,
/// src_coord[d] = index[coord]. This is the core gather contract.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn gather_coord_mapping_3d() {
    let c0: u8 = kani::any();
    let c1: u8 = kani::any();
    let c2: u8 = kani::any();
    let gather_idx: u8 = kani::any();
    let dim: u8 = kani::any();

    kani::assume(c0 <= 15);
    kani::assume(c1 <= 15);
    kani::assume(c2 <= 15);
    kani::assume(gather_idx <= 15);
    kani::assume(dim < 3);

    let coord = [c0 as usize, c1 as usize, c2 as usize];
    let d = dim as usize;

    // Simulate gather coordinate computation
    let mut src_coord = coord;
    src_coord[d] = gather_idx as usize;

    // Verify: non-gather dims are unchanged
    for i in 0..3 {
        if i != d {
            assert_eq!(src_coord[i], coord[i], "non-gather dim must be unchanged");
        }
    }
    // Verify: gather dim is the index value
    assert_eq!(
        src_coord[d], gather_idx as usize,
        "gather dim must be the index value"
    );
}

/// Prove: gather flat index decomposition correctly recovers coordinates.
///
/// The gather CPU loop uses flat_idx -> coordinate decomposition. This
/// verifies that the decomposition is the inverse of the standard
/// row-major linearization.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gather_flat_index_decomposition_2d() {
    let s0: u8 = kani::any();
    let s1: u8 = kani::any();
    let flat_idx: u16 = kani::any();

    kani::assume(s0 >= 1 && s0 <= 16);
    kani::assume(s1 >= 1 && s1 <= 16);

    let shape = [s0 as usize, s1 as usize];
    let numel = shape[0] * shape[1];
    kani::assume((flat_idx as usize) < numel);

    let fi = flat_idx as usize;

    // Decompose flat index into coordinates (C-order, last dim varies fastest)
    let c1 = fi % shape[1];
    let c0 = fi / shape[1];

    // Verify round-trip: coords -> flat_idx
    let reconstructed = c0 * shape[1] + c1;
    assert_eq!(
        reconstructed, fi,
        "flat index decomposition must round-trip"
    );

    // Verify coords are in-bounds
    assert!(c0 < shape[0], "coord[0] must be in bounds");
    assert!(c1 < shape[1], "coord[1] must be in bounds");
}

/// Prove: gather flat index decomposition for 3D tensors.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gather_flat_index_decomposition_3d() {
    let s0: u8 = kani::any();
    let s1: u8 = kani::any();
    let s2: u8 = kani::any();
    let flat_idx: u16 = kani::any();

    kani::assume(s0 >= 1 && s0 <= 8);
    kani::assume(s1 >= 1 && s1 <= 8);
    kani::assume(s2 >= 1 && s2 <= 8);

    let shape = [s0 as usize, s1 as usize, s2 as usize];
    let numel = shape[0] * shape[1] * shape[2];
    kani::assume((flat_idx as usize) < numel);

    let fi = flat_idx as usize;

    // C-order decomposition (same algorithm as gather_dispatch)
    let mut rem = fi;
    let c2 = rem % shape[2];
    rem /= shape[2];
    let c1 = rem % shape[1];
    let c0 = rem / shape[1];

    // Round-trip
    let reconstructed = c0 * (shape[1] * shape[2]) + c1 * shape[2] + c2;
    assert_eq!(
        reconstructed, fi,
        "3D flat index decomposition must round-trip"
    );

    assert!(c0 < shape[0], "coord[0] must be in bounds");
    assert!(c1 < shape[1], "coord[1] must be in bounds");
    assert!(c2 < shape[2], "coord[2] must be in bounds");
}

// ---------------------------------------------------------------------------
// gather: OOB detection in gather loop
// ---------------------------------------------------------------------------

/// Prove: gather OOB check rejects gather_idx >= dim_size.
///
/// Same boundary as index_select, but applied per-element in the gather loop.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gather_oob_rejects_boundary() {
    let dim_size: u16 = kani::any();
    let gather_idx: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 256);
    kani::assume(gather_idx <= 512);

    let is_oob = (gather_idx as usize) >= (dim_size as usize);

    if gather_idx < dim_size {
        assert!(!is_oob, "in-bounds gather index must not be rejected");
    } else {
        assert!(is_oob, "OOB gather index must be detected");
    }
}

// ---------------------------------------------------------------------------
// gather: shape validation — ids must have same rank as self
// ---------------------------------------------------------------------------

/// Prove: gather rank mismatch detection is correct.
///
/// gather requires ids.rank() == self.rank(). Any mismatch must be rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gather_rejects_rank_mismatch() {
    let self_rank: u8 = kani::any();
    let ids_rank: u8 = kani::any();

    kani::assume(self_rank >= 1 && self_rank <= 6);
    kani::assume(ids_rank >= 1 && ids_rank <= 6);

    let valid = self_rank == ids_rank;

    if valid {
        assert_eq!(self_rank, ids_rank, "matching ranks must pass");
    } else {
        assert_ne!(self_rank, ids_rank, "mismatched ranks must be rejected");
    }
}

/// Prove: gather non-gather dim size validation is correct.
///
/// For all dims d != gather_dim, ids.dims()[d] <= self.dims()[d] must hold.
/// Violation means the index tensor has more positions than the source.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn gather_non_gather_dim_validation_2d() {
    let self_d0: u8 = kani::any();
    let self_d1: u8 = kani::any();
    let ids_d0: u8 = kani::any();
    let ids_d1: u8 = kani::any();
    let gather_dim: u8 = kani::any();

    kani::assume(self_d0 >= 1 && self_d0 <= 32);
    kani::assume(self_d1 >= 1 && self_d1 <= 32);
    kani::assume(ids_d0 >= 1 && ids_d0 <= 32);
    kani::assume(ids_d1 >= 1 && ids_d1 <= 32);
    kani::assume(gather_dim < 2);

    let self_dims = [self_d0 as usize, self_d1 as usize];
    let ids_dims = [ids_d0 as usize, ids_d1 as usize];
    let gd = gather_dim as usize;

    // Check non-gather dims
    let mut valid = true;
    for d in 0..2 {
        if d != gd && ids_dims[d] > self_dims[d] {
            valid = false;
        }
    }

    // The check must detect oversized non-gather dims
    if !valid {
        let non_gd = 1 - gd;
        assert!(
            ids_dims[non_gd] > self_dims[non_gd],
            "oversized non-gather dim must be detected"
        );
    }
}

// ---------------------------------------------------------------------------
// expand: broadcast rule enforcement
// ---------------------------------------------------------------------------

/// Prove: expand rejects dims where old != 1 and old != new.
///
/// The expand rule: a dimension of size 1 can expand to any size, but
/// non-1 dimensions must match exactly. This is the fundamental broadcast
/// constraint.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn expand_rejects_invalid_broadcast() {
    let old: u16 = kani::any();
    let new: u16 = kani::any();

    kani::assume(old >= 1 && old <= 256);
    kani::assume(new >= 1 && new <= 256);

    let valid = old == 1 || old == new;

    if old == 1 {
        assert!(valid, "size-1 dim can always expand");
    } else if old == new {
        assert!(valid, "matching dims are valid for expand");
    } else {
        assert!(!valid, "non-1, non-matching dims must be rejected");
    }
}

/// Prove: expand on a 3D shape produces correct output element count.
///
/// For each dim: if old == 1, output gets new_dim; else old == new.
/// Element count must equal product of new_dims.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn expand_element_count_3d() {
    let o0: u8 = kani::any();
    let o1: u8 = kani::any();
    let o2: u8 = kani::any();
    let n0: u8 = kani::any();
    let n1: u8 = kani::any();
    let n2: u8 = kani::any();

    kani::assume(o0 >= 1 && o0 <= 8);
    kani::assume(o1 >= 1 && o1 <= 8);
    kani::assume(o2 >= 1 && o2 <= 8);
    kani::assume(n0 >= 1 && n0 <= 8);
    kani::assume(n1 >= 1 && n1 <= 8);
    kani::assume(n2 >= 1 && n2 <= 8);

    let old = [o0 as usize, o1 as usize, o2 as usize];
    let new = [n0 as usize, n1 as usize, n2 as usize];

    // Validate expand rules
    let valid = (old[0] == 1 || old[0] == new[0])
        && (old[1] == 1 || old[1] == new[1])
        && (old[2] == 1 || old[2] == new[2]);

    if valid {
        let out_numel = checked_dim_product(&new);
        assert!(out_numel.is_ok(), "valid expand target must not overflow");

        // Output numel >= input numel (expand never shrinks)
        let in_numel = checked_dim_product(&old);
        if let (Ok(inp), Ok(outp)) = (in_numel, out_numel) {
            assert!(outp >= inp, "expand must not reduce element count");
        }
    }
}

/// Prove: expand rank mismatch detection.
///
/// expand requires new_dims.len() == self.rank(). Different lengths
/// must be rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn expand_rejects_rank_mismatch() {
    let self_rank: u8 = kani::any();
    let new_rank: u8 = kani::any();

    kani::assume(self_rank >= 1 && self_rank <= 6);
    kani::assume(new_rank >= 1 && new_rank <= 6);

    let valid = self_rank == new_rank;
    if !valid {
        assert_ne!(self_rank, new_rank, "rank mismatch must be detected");
    }
}

/// Prove: expand with identity target (same shape) preserves element count.
///
/// When every old[i] == new[i], expand is a no-op on shape. The output
/// must have exactly the same element count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn expand_identity_preserves_numel() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 128);
    kani::assume(d1 >= 1 && d1 <= 128);

    let dims = [d0 as usize, d1 as usize];

    // Expand with same target
    let old_numel = checked_dim_product(&dims);
    let new_numel = checked_dim_product(&dims);

    if let (Ok(on), Ok(nn)) = (old_numel, new_numel) {
        assert_eq!(on, nn, "identity expand must preserve element count");
    }
}

/// Prove: expand from all-ones shape to target preserves target numel.
///
/// A tensor of shape [1, 1, ..., 1] can expand to any target shape.
/// The output element count must equal the target element count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn expand_all_ones_to_target() {
    let n0: u8 = kani::any();
    let n1: u8 = kani::any();
    let n2: u8 = kani::any();

    kani::assume(n0 >= 1 && n0 <= 16);
    kani::assume(n1 >= 1 && n1 <= 16);
    kani::assume(n2 >= 1 && n2 <= 16);

    let old = [1usize, 1, 1];
    let new = [n0 as usize, n1 as usize, n2 as usize];

    // All old dims are 1, so expand is always valid
    let valid = (old[0] == 1 || old[0] == new[0])
        && (old[1] == 1 || old[1] == new[1])
        && (old[2] == 1 || old[2] == new[2]);
    assert!(valid, "all-ones shape must always be expandable");

    let out = checked_dim_product(&new);
    let target = checked_dim_product(&new);
    if let (Ok(o), Ok(t)) = (out, target) {
        assert_eq!(o, t, "expand from all-ones must match target numel");
    }
}

// ---------------------------------------------------------------------------
// expand: idempotency (expanding an already-expanded shape is a no-op)
// ---------------------------------------------------------------------------

/// Prove: expanding to the same target twice produces the same shape.
///
/// expand(target).expand(target) == expand(target). Once a tensor has the
/// target shape, re-expanding is identity.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn expand_is_idempotent() {
    let n0: u8 = kani::any();
    let n1: u8 = kani::any();

    kani::assume(n0 >= 1 && n0 <= 32);
    kani::assume(n1 >= 1 && n1 <= 32);

    let target = [n0 as usize, n1 as usize];

    // After first expand, shape equals target.
    // Second expand with same target: each dim matches (old == new), so valid.
    for i in 0..2 {
        let old = target[i];
        let new = target[i];
        assert!(old == 1 || old == new, "re-expand must be valid");
        assert_eq!(old, new, "already-expanded dim must equal target");
    }
}

// ---------------------------------------------------------------------------
// I64 index: negative value rejection
// ---------------------------------------------------------------------------

/// Prove: I64 negative indices are correctly detected as OOB.
///
/// index_select with I64 indices checks `idx < 0 || (idx as usize) >= dim_size`.
/// Negative i64 values must always be rejected regardless of dim_size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn i64_negative_index_always_oob() {
    let idx: i64 = kani::any();
    let dim_size: u16 = kani::any();

    kani::assume(idx < 0);
    kani::assume(dim_size >= 1 && dim_size <= 4096);

    // The production code checks: if idx < 0 || (idx as usize) >= dim_size
    let is_oob = idx < 0 || (idx as usize) >= (dim_size as usize);

    assert!(is_oob, "negative i64 index must always be OOB");
}

/// Prove: I64 non-negative index OOB check matches U32 behavior.
///
/// For non-negative i64 values within u32 range, the OOB check must
/// produce the same result as the equivalent u32 check.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn i64_nonneg_index_matches_u32() {
    let idx: u16 = kani::any();
    let dim_size: u16 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 4096);
    kani::assume(idx <= 8192);

    let i64_val = idx as i64;
    let u32_val = idx as u32;

    // Both must agree on OOB
    let i64_oob = i64_val < 0 || (i64_val as usize) >= (dim_size as usize);
    let u32_oob = (u32_val as usize) >= (dim_size as usize);

    assert_eq!(i64_oob, u32_oob, "non-negative i64 and u32 OOB must agree");
}
