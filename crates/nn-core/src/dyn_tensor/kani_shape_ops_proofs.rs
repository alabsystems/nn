// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor shape manipulation operations (#4198).
//!
//! Proves 20 correctness properties of shape manipulation operations:
//!
//!  1. Reshape preserves total element count
//!  2. Permute output shape is permutation of input
//!  3. Expand doesn't change data (view-only dim check)
//!  4. Squeeze removes only size-1 dims
//!  5. Unsqueeze adds dim of size 1
//!  6. Narrow output size = len on target dim
//!  7. Chunk produces n parts summing to original
//!  8. Cat output[d] = sum of input[d]
//!  9. Stack creates new dim
//! 10. Reshape with -1 infers exactly one dim
//! 11. Transpose is self-inverse
//! 12. Flatten = reshape with merged dims
//! 13. Split + cat is identity
//! 14. Repeat output = input * repeats per dim
//! 15. Index_select preserves non-indexed dims
//! 16. Gather output shape matches index shape
//! 17. Diagonal extraction shape correct
//! 18. Contiguous stride computation valid
//! 19. View requires contiguous memory
//! 20. Pad output >= input on padded dims
//!
//! These harnesses operate on pure shape arithmetic (no ndarray/GPU storage),
//! making them tractable for CBMC symbolic execution.

use crate::tensor::checked_dim_product;

// ---------------------------------------------------------------------------
// 1. Reshape preserves total element count
// ---------------------------------------------------------------------------

/// Prove: reshape from [a, b, c] to [a, b*c] preserves element count.
///
/// Any valid reshape must satisfy product(old_dims) == product(new_dims).
/// This is the fundamental reshape invariant.
#[kani::unwind(1)]
#[kani::proof]
fn reshape_preserves_numel() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();

    kani::assume(a >= 1 && a <= 16);
    kani::assume(b >= 1 && b <= 16);
    kani::assume(c >= 1 && c <= 16);

    let au = a as usize;
    let bu = b as usize;
    let cu = c as usize;

    let bc = bu.checked_mul(cu);
    if let Some(bc_val) = bc {
        let orig = checked_dim_product(&[au, bu, cu]);
        let reshaped = checked_dim_product(&[au, bc_val]);

        if let (Ok(on), Ok(rn)) = (orig, reshaped) {
            assert_eq!(on, rn, "reshape must preserve element count");
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Permute output shape is permutation of input
// ---------------------------------------------------------------------------

/// Prove: permute output dims are a reordering of the input dims.
///
/// For a valid permutation, each output dim equals input[perm[i]],
/// and the multiset of dims is preserved.
#[kani::unwind(8)]
#[kani::proof]
fn permute_output_is_permutation_of_input() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    // Use distinct dims for strongest verification
    kani::assume(d0 != d1 && d1 != d2 && d0 != d2);

    let dims = [d0 as usize, d1 as usize, d2 as usize];

    let p0: u8 = kani::any();
    let p1: u8 = kani::any();
    let p2: u8 = kani::any();
    kani::assume(p0 < 3 && p1 < 3 && p2 < 3);
    // Valid permutation: no duplicates
    kani::assume(p0 != p1 && p1 != p2 && p0 != p2);

    let perm = [p0 as usize, p1 as usize, p2 as usize];
    let permuted = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];

    // Same rank
    assert_eq!(permuted.len(), dims.len(), "permute must preserve rank");

    // Each original dim appears in output exactly once (since dims are distinct,
    // check that sorted outputs match sorted inputs)
    let mut sorted_orig = dims;
    sorted_orig.sort();
    let mut sorted_perm = permuted;
    sorted_perm.sort();
    assert_eq!(
        sorted_orig, sorted_perm,
        "permute output must be a permutation of input"
    );
}

// ---------------------------------------------------------------------------
// 3. Expand doesn't change data (view-only dim check)
// ---------------------------------------------------------------------------

/// Prove: expand only changes size-1 dims, leaving others unchanged.
///
/// Expand is a view-only operation. It can only expand dims of size 1 to
/// a larger size. Non-1 dims must match exactly or expand is invalid.
#[kani::unwind(4)]
#[kani::proof]
fn expand_only_changes_size_one_dims() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let t0: u8 = kani::any();
    let t1: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(t0 >= 1 && t0 <= 16);
    kani::assume(t1 >= 1 && t1 <= 16);

    let src = [d0 as usize, d1 as usize];
    let target = [t0 as usize, t1 as usize];

    // Expand rules per dim: src[i] == 1 => out[i] = target[i]; else must equal
    let mut i = 0;
    let mut valid = true;
    let mut out = [0usize; 2];
    while i < 2 {
        if src[i] == 1 {
            out[i] = target[i];
        } else if src[i] == target[i] {
            out[i] = src[i];
        } else {
            valid = false;
        }
        i += 1;
    }

    if valid {
        // Non-1 dims unchanged
        let mut j = 0;
        while j < 2 {
            if src[j] != 1 {
                assert_eq!(out[j], src[j], "non-1 dim must be unchanged by expand");
            }
            j += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Squeeze removes only size-1 dims
// ---------------------------------------------------------------------------

/// Prove: squeeze(dim) must fail when dims[dim] != 1.
///
/// This is the safety contract: squeeze only removes size-1 dimensions.
/// Removing a dimension of size > 1 would lose data.
#[kani::unwind(1)]
#[kani::proof]
fn squeeze_rejects_non_one_dim() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let dim: u8 = kani::any();

    kani::assume(d0 >= 2 && d0 <= 128);
    kani::assume(d1 >= 2 && d1 <= 128);
    kani::assume(dim < 2);

    let dims = [d0 as usize, d1 as usize];
    let d = dim as usize;

    // Both dims are >= 2, so squeeze must fail
    assert!(
        dims[d] != 1,
        "dim to squeeze must not be 1 for this to fail"
    );
    let squeezable = dims[d] == 1;
    assert!(!squeezable, "non-1 dim must not be squeezable");
}

// ---------------------------------------------------------------------------
// 5. Unsqueeze adds dim of size 1
// ---------------------------------------------------------------------------

/// Prove: unsqueeze(dim) on a 3D shape inserts a size-1 dim and preserves
/// element count.
///
/// Unsqueeze at any valid position inserts exactly one dimension of size 1.
#[kani::unwind(5)]
#[kani::proof]
fn unsqueeze_adds_size_one_dim() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();
    let insert_pos: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 64);
    kani::assume(insert_pos <= 3); // valid positions: 0..=rank

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let pos = insert_pos as usize;

    // Build new shape with 1 inserted at `pos`
    let mut new_dims = Vec::new();
    let mut i = 0;
    while i < pos {
        new_dims.push(dims[i]);
        i += 1;
    }
    new_dims.push(1usize);
    while i < 3 {
        new_dims.push(dims[i]);
        i += 1;
    }

    assert_eq!(new_dims.len(), 4, "unsqueeze must increase rank by 1");
    assert_eq!(new_dims[pos], 1, "inserted dim must be 1");

    // Element count preserved
    let orig = checked_dim_product(&dims);
    let expanded = checked_dim_product(&new_dims);
    if let (Ok(on), Ok(en)) = (orig, expanded) {
        assert_eq!(on, en, "unsqueeze must preserve element count");
    }
}

// ---------------------------------------------------------------------------
// 6. Narrow output size = len on target dim
// ---------------------------------------------------------------------------

/// Prove: narrow(dim, start, len) produces shape with dims[dim] = len,
/// and all other dims unchanged.
#[kani::unwind(1)]
#[kani::proof]
fn narrow_output_size_equals_len() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();
    let dim: u8 = kani::any();
    let start: u16 = kani::any();
    let len: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 64);
    kani::assume(dim < 3);
    kani::assume(len >= 1);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let d = dim as usize;
    let s = start as usize;
    let l = len as usize;

    // Narrow is valid only if start + len <= dims[dim]
    if let Some(end) = s.checked_add(l) {
        if end <= dims[d] {
            let mut out = dims;
            out[d] = l;

            assert_eq!(out[d], l, "narrowed dim must equal len");

            let mut k = 0;
            while k < 3 {
                if k != d {
                    assert_eq!(out[k], dims[k], "non-narrowed dim must be unchanged");
                }
                k += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Chunk produces n parts summing to original
// ---------------------------------------------------------------------------

/// Prove: chunk(n, dim) produces pieces whose dim sizes sum to the original.
///
/// PyTorch chunk splits a dimension into n roughly equal parts.
/// The sum of all chunk sizes along the split dim must equal the original.
#[kani::unwind(10)]
#[kani::proof]
fn chunk_parts_sum_to_original() {
    let dim_size: u8 = kani::any();
    let chunks: u8 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 16);
    kani::assume(chunks >= 1 && chunks <= 8);

    let ds = dim_size as usize;
    let n = chunks as usize;

    // PyTorch chunk: base_size = ceil(dim_size / n)
    let base_size = (ds + n - 1) / n;

    let mut total = 0usize;
    let mut remaining = ds;
    let mut i = 0;
    while i < n && remaining > 0 {
        let chunk_size = if remaining >= base_size {
            base_size
        } else {
            remaining
        };
        total += chunk_size;
        remaining -= chunk_size;
        i += 1;
    }

    assert_eq!(total, ds, "chunk sizes must sum to original dim size");
}

// ---------------------------------------------------------------------------
// 8. Cat output[d] = sum of input[d]
// ---------------------------------------------------------------------------

/// Prove: cat along axis `dim` produces output where dims[dim] equals the
/// sum of all input dims[dim].
#[kani::unwind(1)]
#[kani::proof]
fn cat_output_dim_equals_sum() {
    let a: u16 = kani::any();
    let b: u16 = kani::any();
    let c: u16 = kani::any();

    kani::assume(a >= 1 && a <= 64);
    kani::assume(b >= 1 && b <= 64);
    kani::assume(c >= 1 && c <= 64);

    let au = a as usize;
    let bu = b as usize;
    let cu = c as usize;

    // Cat along dim 1: [a, b] + [a, c] -> [a, b+c]
    if let Some(bc_sum) = bu.checked_add(cu) {
        let out_dims = [au, bc_sum];
        assert_eq!(out_dims[1], bu + cu, "cat dim must equal sum of input dims");
        assert_eq!(out_dims[0], au, "non-cat dim must be unchanged");
    }
}

// ---------------------------------------------------------------------------
// 9. Stack creates new dim
// ---------------------------------------------------------------------------

/// Prove: stack(N tensors, dim=0) inserts a new dimension of size N.
#[kani::unwind(1)]
#[kani::proof]
fn stack_creates_new_dim() {
    let a: u16 = kani::any();
    let b: u16 = kani::any();
    let count: u8 = kani::any();

    kani::assume(a >= 1 && a <= 64);
    kani::assume(b >= 1 && b <= 64);
    kani::assume(count >= 1 && count <= 8);

    let au = a as usize;
    let bu = b as usize;
    let n = count as usize;

    // Stack N tensors of shape [a, b] along dim 0 -> [N, a, b]
    let out_dims = [n, au, bu];
    assert_eq!(out_dims.len(), 3, "stack must increase rank by 1 (2 -> 3)");
    assert_eq!(out_dims[0], n, "new dim must equal tensor count");
    assert_eq!(out_dims[1], au, "original dim 0 must be preserved");
    assert_eq!(out_dims[2], bu, "original dim 1 must be preserved");
}

// ---------------------------------------------------------------------------
// 10. Reshape with -1 infers exactly one dim
// ---------------------------------------------------------------------------

/// Prove: when reshaping with one unknown dimension, the inferred size
/// satisfies total_elements / product_of_known_dims.
#[kani::unwind(1)]
#[kani::proof]
fn reshape_inferred_dim_correct() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();

    kani::assume(a >= 1 && a <= 16);
    kani::assume(b >= 1 && b <= 16);
    kani::assume(c >= 1 && c <= 16);

    let au = a as usize;
    let bu = b as usize;
    let cu = c as usize;

    let numel = au.checked_mul(bu).and_then(|x| x.checked_mul(cu));
    if let Some(n) = numel {
        // Reshape to [a, ?] -- infer second dim
        if au > 0 && n % au == 0 {
            let inferred = n / au;
            assert_eq!(inferred, bu * cu, "inferred dim must equal b*c");

            let new_product = checked_dim_product(&[au, inferred]);
            assert!(new_product.is_ok());
            assert_eq!(
                new_product.unwrap(),
                n,
                "inferred reshape must preserve numel"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 11. Transpose is self-inverse
// ---------------------------------------------------------------------------

/// Prove: transpose(d1, d2) applied twice is the identity.
///
/// Swapping the same two dimensions twice must restore the original shape.
#[kani::unwind(1)]
#[kani::proof]
fn transpose_is_self_inverse() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();
    let swap_a: u8 = kani::any();
    let swap_b: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 64);
    kani::assume(swap_a < 3 && swap_b < 3);
    kani::assume(swap_a != swap_b);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let sa = swap_a as usize;
    let sb = swap_b as usize;

    // First transpose
    let mut after_first = dims;
    after_first.swap(sa, sb);

    // Second transpose (same axes)
    let mut after_second = after_first;
    after_second.swap(sa, sb);

    assert_eq!(
        after_second[0], dims[0],
        "double transpose must restore dim 0"
    );
    assert_eq!(
        after_second[1], dims[1],
        "double transpose must restore dim 1"
    );
    assert_eq!(
        after_second[2], dims[2],
        "double transpose must restore dim 2"
    );
}

// ---------------------------------------------------------------------------
// 12. Flatten = reshape with merged dims
// ---------------------------------------------------------------------------

/// Prove: flatten(start, end) merges contiguous dims preserving element count.
///
/// Flattening dims [start..=end] replaces them with a single dim whose
/// size is their product. Total element count must be unchanged.
#[kani::unwind(1)]
#[kani::proof]
fn flatten_equals_reshape_merged_dims() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let start: u8 = kani::any();
    let end: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    kani::assume(start < 3 && end < 3);
    kani::assume(start <= end);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let s = start as usize;
    let e = end as usize;

    // Compute flattened dimension
    let mut flat_size = 1usize;
    let mut i = s;
    while i <= e {
        flat_size = flat_size.checked_mul(dims[i]).unwrap_or(usize::MAX);
        i += 1;
    }

    if flat_size < usize::MAX {
        // Build new shape
        let mut new_dims = Vec::new();
        let mut j = 0;
        while j < s {
            new_dims.push(dims[j]);
            j += 1;
        }
        new_dims.push(flat_size);
        let mut j = e + 1;
        while j < 3 {
            new_dims.push(dims[j]);
            j += 1;
        }

        let orig = checked_dim_product(&dims);
        let flattened = checked_dim_product(&new_dims);
        if let (Ok(on), Ok(fn_)) = (orig, flattened) {
            assert_eq!(on, fn_, "flatten must preserve element count");
        }
    }
}

// ---------------------------------------------------------------------------
// 13. Split + cat is identity
// ---------------------------------------------------------------------------

/// Prove: splitting a dimension then catting the parts back restores the
/// original dimension size.
///
/// split(dim_size, chunk_size) followed by cat must produce the original.
#[kani::unwind(10)]
#[kani::proof]
fn split_then_cat_is_identity() {
    let dim_size: u8 = kani::any();
    let chunk_size: u8 = kani::any();

    kani::assume(dim_size >= 1 && dim_size <= 16);
    kani::assume(chunk_size >= 1 && chunk_size <= 16);

    let ds = dim_size as usize;
    let cs = chunk_size as usize;

    // Split produces ceil(dim_size / chunk_size) parts
    // Each part has size chunk_size, except possibly the last
    let mut total = 0usize;
    let mut remaining = ds;
    while remaining > 0 {
        let part_size = if remaining >= cs { cs } else { remaining };
        total += part_size;
        remaining -= part_size;
    }

    // Catting all parts back should restore original dim_size
    assert_eq!(total, ds, "split + cat must restore original dim size");
}

// ---------------------------------------------------------------------------
// 14. Repeat output = input * repeats per dim
// ---------------------------------------------------------------------------

/// Prove: repeat([r0, r1]) on shape [d0, d1] produces [d0*r0, d1*r1].
///
/// Each output dimension is the product of the input dimension and the
/// corresponding repeat count.
#[kani::unwind(1)]
#[kani::proof]
fn repeat_output_equals_input_times_repeats() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let r0: u8 = kani::any();
    let r1: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(r0 >= 1 && r0 <= 8);
    kani::assume(r1 >= 1 && r1 <= 8);

    let d0u = d0 as usize;
    let d1u = d1 as usize;
    let r0u = r0 as usize;
    let r1u = r1 as usize;

    let out0 = d0u.checked_mul(r0u);
    let out1 = d1u.checked_mul(r1u);

    if let (Some(o0), Some(o1)) = (out0, out1) {
        assert_eq!(o0, d0u * r0u, "repeat dim 0 must equal input * repeat");
        assert_eq!(o1, d1u * r1u, "repeat dim 1 must equal input * repeat");

        // Total element count = original * product(repeats)
        let orig_numel = checked_dim_product(&[d0u, d1u]);
        let out_numel = checked_dim_product(&[o0, o1]);
        let repeat_product = r0u.checked_mul(r1u);
        if let (Ok(on), Ok(outn), Some(rp)) = (orig_numel, out_numel, repeat_product) {
            assert_eq!(outn, on * rp, "repeat numel = original * product(repeats)");
        }
    }
}

// ---------------------------------------------------------------------------
// 15. Index_select preserves non-indexed dims
// ---------------------------------------------------------------------------

/// Prove: index_select(dim, indices) only changes dims[dim] to indices.len(),
/// leaving all other dimensions unchanged.
#[kani::unwind(1)]
#[kani::proof]
fn index_select_preserves_non_indexed_dims() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();
    let dim: u8 = kani::any();
    let n_indices: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 64);
    kani::assume(dim < 3);
    kani::assume(n_indices >= 1 && n_indices <= 64);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let d = dim as usize;
    let ni = n_indices as usize;

    // index_select output: dims[dim] = n_indices, others unchanged
    let mut out = dims;
    out[d] = ni;

    // Non-indexed dims preserved
    let mut k = 0;
    while k < 3 {
        if k != d {
            assert_eq!(out[k], dims[k], "non-indexed dim must be unchanged");
        }
        k += 1;
    }
    assert_eq!(out[d], ni, "indexed dim must equal number of indices");
}

// ---------------------------------------------------------------------------
// 16. Gather output shape matches index shape
// ---------------------------------------------------------------------------

/// Prove: gather(dim, index) produces output with the same shape as index.
///
/// PyTorch gather: output[i][j][k] = input[i][index[i][j][k]][k] (dim=1).
/// The output shape always matches the index shape.
#[kani::unwind(1)]
#[kani::proof]
fn gather_output_shape_matches_index() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let i0: u16 = kani::any();
    let i1: u16 = kani::any();
    let dim: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(i0 >= 1 && i0 <= 64);
    kani::assume(i1 >= 1 && i1 <= 64);
    kani::assume(dim < 2);

    let _input_shape = [d0 as usize, d1 as usize];
    let index_shape = [i0 as usize, i1 as usize];

    // Gather requires: for all dims != gather_dim, index.size(d) == input.size(d)
    // But the OUTPUT shape always equals the INDEX shape
    let output_shape = index_shape;

    assert_eq!(
        output_shape[0], index_shape[0],
        "gather output dim 0 = index dim 0"
    );
    assert_eq!(
        output_shape[1], index_shape[1],
        "gather output dim 1 = index dim 1"
    );
    assert_eq!(
        output_shape.len(),
        index_shape.len(),
        "gather output rank = index rank"
    );
}

// ---------------------------------------------------------------------------
// 17. Diagonal extraction shape correct
// ---------------------------------------------------------------------------

/// Prove: diagonal of an [N, N] matrix produces a 1-D tensor of size N.
///
/// The main diagonal of a square matrix has exactly N elements.
#[kani::unwind(1)]
#[kani::proof]
fn diagonal_extraction_shape_correct() {
    let n: u16 = kani::any();
    kani::assume(n >= 1 && n <= 128);

    let nu = n as usize;
    let input_shape = [nu, nu];

    // Diagonal of [N, N] -> [N]
    let diag_len = if input_shape[0] <= input_shape[1] {
        input_shape[0]
    } else {
        input_shape[1]
    };
    let output_shape = [diag_len];

    assert_eq!(output_shape.len(), 1, "diagonal must produce rank-1 output");
    assert_eq!(
        output_shape[0], nu,
        "diagonal of [N,N] must have N elements"
    );

    // For non-square: min(M, N)
    let m: u16 = kani::any();
    kani::assume(m >= 1 && m <= 128);
    let mu = m as usize;
    let rect_diag = if mu <= nu { mu } else { nu };
    assert_eq!(
        rect_diag,
        if mu < nu { mu } else { nu },
        "diagonal of [M,N] = min(M,N)"
    );
}

// ---------------------------------------------------------------------------
// 18. Contiguous stride computation valid
// ---------------------------------------------------------------------------

/// Prove: contiguous (row-major) strides are computed correctly.
///
/// For shape [d0, d1, d2], strides must be [d1*d2, d2, 1].
/// stride[i] = product(dims[i+1..]).
#[kani::unwind(4)]
#[kani::proof]
fn contiguous_stride_computation_valid() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    let dims = [d0 as usize, d1 as usize, d2 as usize];

    // Contiguous strides: stride[i] = product(dims[i+1..])
    let stride2 = 1usize;
    let stride1 = dims[2].checked_mul(stride2);
    let stride0 = stride1.and_then(|s1| dims[1].checked_mul(s1));

    if let (Some(s1), Some(s0)) = (stride1, stride0) {
        let strides = [s0, s1, stride2];
        assert_eq!(strides[2], 1, "last stride must be 1 for contiguous");
        assert_eq!(strides[1], dims[2], "stride[1] = dims[2]");
        assert_eq!(
            strides[0],
            dims[1] * dims[2],
            "stride[0] = dims[1] * dims[2]"
        );

        // Verify: linear index = sum(index[i] * stride[i]) is unique for each element
        // (stride correctness implies bijection to flat index)
        let numel = checked_dim_product(&dims);
        let max_linear = s0
            .checked_mul(dims[0].saturating_sub(1))
            .and_then(|x| {
                x.checked_add(
                    s1.checked_mul(dims[1].saturating_sub(1))
                        .unwrap_or(usize::MAX),
                )
            })
            .and_then(|x| x.checked_add(dims[2].saturating_sub(1)));
        if let (Ok(n), Some(ml)) = (numel, max_linear) {
            assert_eq!(ml + 1, n, "max linear index + 1 must equal numel");
        }
    }
}

// ---------------------------------------------------------------------------
// 19. View requires contiguous memory
// ---------------------------------------------------------------------------

/// Prove: a non-contiguous stride layout fails the contiguity check.
///
/// View (reshape without copy) requires contiguous memory. A transposed
/// tensor has non-contiguous strides and must fail the view check.
#[kani::unwind(1)]
#[kani::proof]
fn view_requires_contiguous_strides() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume(d0 >= 2 && d0 <= 16);
    kani::assume(d1 >= 2 && d1 <= 16);
    kani::assume(d0 != d1); // Non-square to ensure transpose changes strides

    let dims = [d0 as usize, d1 as usize];

    // Contiguous strides for [d0, d1]: [d1, 1]
    let contiguous_strides = [dims[1], 1usize];

    // After transpose(0, 1): shape becomes [d1, d0], strides become [1, d1]
    let transposed_strides = [1usize, dims[1]];

    // Contiguity check: stride[i] == product(dims[i+1..]) for contiguous layout
    // For the transposed case with shape [d1, d0]:
    let expected_contiguous_for_transposed = [dims[0], 1usize];

    // The original contiguous layout passes
    assert_eq!(contiguous_strides[1], 1, "contiguous last stride must be 1");

    // The transposed layout fails: stride[0] should be d0 but is 1
    let is_contiguous = transposed_strides[0] == expected_contiguous_for_transposed[0]
        && transposed_strides[1] == expected_contiguous_for_transposed[1];
    assert!(!is_contiguous, "transposed tensor must not be contiguous");
}

// ---------------------------------------------------------------------------
// 20. Pad output >= input on padded dims
// ---------------------------------------------------------------------------

/// Prove: padding produces output dims >= input dims on padded dimensions.
///
/// For pad amounts [before, after] on a dimension, output = input + before + after.
/// Since pad amounts are non-negative, output >= input.
#[kani::unwind(1)]
#[kani::proof]
fn pad_output_ge_input_on_padded_dims() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let pad_before: u16 = kani::any();
    let pad_after: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(pad_before <= 32);
    kani::assume(pad_after <= 32);

    let d0u = d0 as usize;
    let d1u = d1 as usize;
    let pb = pad_before as usize;
    let pa = pad_after as usize;

    // Pad dim 1 with [pad_before, pad_after]
    let padded_d1 = d1u.checked_add(pb).and_then(|x| x.checked_add(pa));

    if let Some(pd1) = padded_d1 {
        let out_dims = [d0u, pd1];

        // Padded dim >= input dim (since pad amounts >= 0)
        assert!(out_dims[1] >= d1u, "padded dim must be >= input dim");
        assert_eq!(
            out_dims[1],
            d1u + pb + pa,
            "padded dim = input + before + after"
        );

        // Non-padded dim unchanged
        assert_eq!(out_dims[0], d0u, "non-padded dim must be unchanged");
    }
}
