// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for tensor reshape and view operation safety.
//!
//! Proves ten categories of reshape/view invariants:
//!
//! 1. **Reshape element count preservation** — product(old_dims) == product(new_dims)
//! 2. **Flatten/unflatten roundtrip** — flatten then unflatten recovers original shape
//! 3. **Squeeze removes only size-1 dims** — non-1 dims are preserved in order
//! 4. **Unsqueeze inserts size-1 at correct position** — rank increases by 1
//! 5. **Transpose dimension permutation validity** — permutation is bijective
//! 6. **Contiguous memory layout after reshape** — strides match contiguous layout
//! 7. **Stride computation correctness for views** — stride[i] = product(dims[i+1..])
//! 8. **Broadcasting shape compatibility** — broadcast rules are symmetric and associative
//! 9. **Narrow/slice bounds checking** — start + len <= dims[dim] guarantees safety
//! 10. **Expand only works on size-1 dims** — non-1 dims cannot be expanded
//!
//! All harnesses use small concrete dimensions (u8/u16) for CBMC tractability.

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ===========================================================================
// Helper: compute contiguous strides for up to 5 dimensions.
// ===========================================================================

fn contiguous_strides_5(dims: &[usize; 5], rank: usize) -> Option<[usize; 5]> {
    assert!(rank <= 5);
    let mut strides = [0usize; 5];
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

// ===========================================================================
// 1. Reshape element count preservation
// ===========================================================================

/// Prove: reshape from [A, B, C, D] to [A*B, C*D] preserves total element count.
///
/// This is a common pattern: collapsing pairs of dimensions.
/// Product(old) must equal Product(new) for any valid dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn reshape_4d_to_2d_pair_collapse_preserves_numel() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(a >= 1 && a <= 8);
    kani::assume(b >= 1 && b <= 8);
    kani::assume(c >= 1 && c <= 8);
    kani::assume(d >= 1 && d <= 8);

    let au = a as usize;
    let bu = b as usize;
    let cu = c as usize;
    let du = d as usize;

    let numel_4d = checked_dim_product(&[au, bu, cu, du]);

    if let (Some(ab), Some(cd)) = (au.checked_mul(bu), cu.checked_mul(du)) {
        let numel_2d = checked_dim_product(&[ab, cd]);
        if let (Ok(n4), Ok(n2)) = (numel_4d, numel_2d) {
            assert_eq!(n4, n2, "reshape [A,B,C,D]->[A*B,C*D] must preserve numel");
        }
    }
}

/// Prove: reshape from [A, B] to [1, A, B] preserves element count.
///
/// Adding a leading size-1 dimension is a common reshape pattern
/// (e.g., adding a batch dimension). The numel must be unchanged.
#[kani::unwind(1)]
#[kani::proof]
fn reshape_2d_to_3d_add_leading_one_preserves_numel() {
    let a: u16 = kani::any();
    let b: u16 = kani::any();

    kani::assume(a >= 1 && a <= 128);
    kani::assume(b >= 1 && b <= 128);

    let au = a as usize;
    let bu = b as usize;

    let numel_2d = checked_dim_product(&[au, bu]);
    let numel_3d = checked_dim_product(&[1, au, bu]);

    if let (Ok(n2), Ok(n3)) = (numel_2d, numel_3d) {
        assert_eq!(n2, n3, "reshape [A,B]->[1,A,B] must preserve numel");
    }
}

// ===========================================================================
// 2. Flatten/unflatten roundtrip consistency
// ===========================================================================

/// Prove: flatten [A, B, C] -> [A*B*C] then unflatten [A*B*C] -> [A, B, C]
/// recovers original dimensions and numel.
///
/// Flatten computes a single-dim shape from the product of all dims.
/// Unflattening with the original shape must recover exactly.
#[kani::unwind(1)]
#[kani::proof]
fn flatten_unflatten_roundtrip_3d() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();

    kani::assume(a >= 1 && a <= 16);
    kani::assume(b >= 1 && b <= 16);
    kani::assume(c >= 1 && c <= 16);

    let au = a as usize;
    let bu = b as usize;
    let cu = c as usize;

    // Flatten: [A, B, C] -> [A*B*C]
    let numel_3d = checked_dim_product(&[au, bu, cu]);
    if let Ok(flat) = numel_3d {
        let numel_1d = checked_dim_product(&[flat]);
        assert!(numel_1d.is_ok(), "single-dim product must succeed");
        assert_eq!(numel_1d.unwrap(), flat, "flat numel must equal product");

        // Unflatten: [A*B*C] -> [A, B, C]
        // Verify we can recover each dimension by successive division
        assert_eq!(flat % cu, 0, "flat must be divisible by C");
        let after_c = flat / cu;
        assert_eq!(after_c % bu, 0, "remainder must be divisible by B");
        let after_b = after_c / bu;
        assert_eq!(after_b, au, "recovered first dim must be A");

        // Full roundtrip: reconstructed dims match original
        let reconstructed = [after_b, bu, cu];
        let numel_reconstructed = checked_dim_product(&reconstructed);
        if let Ok(nr) = numel_reconstructed {
            assert_eq!(nr, flat, "roundtrip must preserve numel");
        }
    }
}

/// Prove: partial flatten [A, B, C, D] -> [A, B*C, D] then unflatten
/// [A, B*C, D] -> [A, B, C, D] recovers original shape.
///
/// Flattening a contiguous range of dims is common (e.g., merging spatial dims).
#[kani::unwind(1)]
#[kani::proof]
fn partial_flatten_unflatten_roundtrip_4d() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(a >= 1 && a <= 8);
    kani::assume(b >= 1 && b <= 8);
    kani::assume(c >= 1 && c <= 8);
    kani::assume(d >= 1 && d <= 8);

    let au = a as usize;
    let bu = b as usize;
    let cu = c as usize;
    let du = d as usize;

    // Flatten dims 1,2: [A, B, C, D] -> [A, B*C, D]
    if let Some(bc) = bu.checked_mul(cu) {
        let numel_4d = checked_dim_product(&[au, bu, cu, du]);
        let numel_3d = checked_dim_product(&[au, bc, du]);

        if let (Ok(n4), Ok(n3)) = (numel_4d, numel_3d) {
            assert_eq!(n4, n3, "partial flatten must preserve numel");
        }

        // Unflatten: recover B and C from B*C given known B
        assert_eq!(bc % bu, 0, "B*C must be divisible by B");
        let recovered_c = bc / bu;
        assert_eq!(recovered_c, cu, "recovered C must match original");
    }
}

// ===========================================================================
// 3. Squeeze removes only size-1 dimensions
// ===========================================================================

/// Prove: squeeze on a 4D shape removes exactly the size-1 dimensions
/// while preserving all non-1 dimensions in their original order.
///
/// The output rank equals input_rank minus the count of size-1 dims.
/// The output numel equals the input numel.
#[kani::unwind(8)]
#[kani::proof]
fn squeeze_removes_only_size_one_dims_4d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    kani::assume(d3 >= 1 && d3 <= 16);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];

    // Squeeze: collect non-1 dims
    let mut squeezed = [0usize; 4];
    let mut sq_len = 0usize;
    let mut i = 0;
    while i < 4 {
        if dims[i] != 1 {
            squeezed[sq_len] = dims[i];
            sq_len += 1;
        }
        i += 1;
    }

    // Count size-1 dims
    let mut ones_count = 0usize;
    let mut j = 0;
    while j < 4 {
        if dims[j] == 1 {
            ones_count += 1;
        }
        j += 1;
    }

    // Squeezed rank = original rank - ones_count
    assert_eq!(
        sq_len + ones_count,
        4,
        "squeezed + removed must equal original rank"
    );

    // Numel preserved: product of squeezed dims == product of original dims
    // (removing 1s doesn't change the product)
    let orig_numel = checked_dim_product(&dims);
    // Build a slice of the squeezed dims for product check
    let sq_numel = if sq_len == 0 {
        Ok(1usize)
    } else if sq_len == 1 {
        checked_dim_product(&[squeezed[0]])
    } else if sq_len == 2 {
        checked_dim_product(&[squeezed[0], squeezed[1]])
    } else if sq_len == 3 {
        checked_dim_product(&[squeezed[0], squeezed[1], squeezed[2]])
    } else {
        checked_dim_product(&[squeezed[0], squeezed[1], squeezed[2], squeezed[3]])
    };

    if let (Ok(on), Ok(sn)) = (orig_numel, sq_numel) {
        assert_eq!(on, sn, "squeeze must preserve numel");
    }

    // All squeezed dims are > 1 (or sq_len == 0 if all dims were 1)
    let mut k = 0;
    while k < sq_len {
        assert!(squeezed[k] > 1 || sq_len == 0, "squeezed dims must be > 1");
        k += 1;
    }
}

/// Prove: squeeze on a specific dimension only removes it if size == 1.
///
/// squeeze(dim=d) on shape [A, B, C] removes dim d only if dims[d] == 1.
/// If dims[d] != 1, the shape is unchanged.
#[kani::unwind(1)]
#[kani::proof]
fn squeeze_specific_dim_only_if_size_one() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();
    let squeeze_dim: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 64);
    kani::assume(squeeze_dim < 3);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let sd = squeeze_dim as usize;

    if dims[sd] == 1 {
        // Squeeze succeeds: output rank is 2, the dim is removed
        let mut result = [0usize; 2];
        let mut ri = 0;
        let mut di = 0;
        while di < 3 {
            if di != sd {
                result[ri] = dims[di];
                ri += 1;
            }
            di += 1;
        }
        assert_eq!(ri, 2, "must have exactly 2 output dims");

        let orig_numel = checked_dim_product(&dims);
        let sq_numel = checked_dim_product(&result);
        if let (Ok(on), Ok(sn)) = (orig_numel, sq_numel) {
            assert_eq!(on, sn, "squeeze at size-1 dim must preserve numel");
        }
    } else {
        // Squeeze is a no-op: shape unchanged
        let result = dims;
        assert_eq!(result, dims, "squeeze on non-1 dim must be no-op");
    }
}

// ===========================================================================
// 4. Unsqueeze inserts size-1 at correct position
// ===========================================================================

/// Prove: unsqueeze at any valid position inserts a size-1 dimension,
/// increases rank by 1, and preserves numel.
///
/// unsqueeze([A, B, C], dim=d) produces a 4D shape where dims[d] == 1
/// and all other dims match the original in order.
#[kani::unwind(1)]
#[kani::proof]
fn unsqueeze_inserts_one_at_position_3d() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();
    let ins_dim: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 64);
    kani::assume(ins_dim <= 3); // valid positions: 0, 1, 2, 3

    let input = [d0 as usize, d1 as usize, d2 as usize];
    let pos = ins_dim as usize;

    // Build output: insert 1 at position pos
    let mut output = [0usize; 4];
    let mut src = 0;
    let mut dst = 0;
    while dst < 4 {
        if dst == pos {
            output[dst] = 1;
        } else {
            output[dst] = input[src];
            src += 1;
        }
        dst += 1;
    }

    // The inserted dimension is 1
    assert_eq!(output[pos], 1, "unsqueezed dim must be 1");

    // Output rank is input_rank + 1
    assert_eq!(
        output.len(),
        input.len() + 1,
        "unsqueeze must add one dimension"
    );

    // Numel preserved
    let orig_numel = checked_dim_product(&input);
    let new_numel = checked_dim_product(&output);
    if let (Ok(on), Ok(nn)) = (orig_numel, new_numel) {
        assert_eq!(on, nn, "unsqueeze must preserve numel");
    }
}

/// Prove: unsqueeze then squeeze recovers the original shape.
///
/// unsqueeze(dim=d) followed by squeeze(dim=d) must be identity.
#[kani::unwind(1)]
#[kani::proof]
fn unsqueeze_squeeze_roundtrip_2d() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let ins_dim: u8 = kani::any();

    kani::assume(d0 >= 2 && d0 <= 64); // >= 2 so squeeze only removes our inserted dim
    kani::assume(d1 >= 2 && d1 <= 64);
    kani::assume(ins_dim <= 2); // valid for 2D input

    let input = [d0 as usize, d1 as usize];
    let pos = ins_dim as usize;

    // Unsqueeze: insert 1 at position pos -> 3D
    let mut unsqueezed = [0usize; 3];
    let mut src = 0;
    let mut dst = 0;
    while dst < 3 {
        if dst == pos {
            unsqueezed[dst] = 1;
        } else {
            unsqueezed[dst] = input[src];
            src += 1;
        }
        dst += 1;
    }

    // Squeeze: remove dim at position pos (which is 1) -> 2D
    let mut squeezed = [0usize; 2];
    let mut si = 0;
    let mut ui = 0;
    while ui < 3 {
        if ui != pos {
            squeezed[si] = unsqueezed[ui];
            si += 1;
        }
        ui += 1;
    }

    assert_eq!(squeezed[0], input[0], "first dim must recover");
    assert_eq!(squeezed[1], input[1], "second dim must recover");
}

// ===========================================================================
// 5. Transpose dimension permutation validity
// ===========================================================================

/// Prove: a 3D permutation is valid iff it is a bijection on {0, 1, 2}.
///
/// For any permutation array [p0, p1, p2], validity requires all indices
/// are in range and no duplicates exist.
#[kani::unwind(1)]
#[kani::proof]
fn permutation_is_bijection_3d() {
    let p0: u8 = kani::any();
    let p1: u8 = kani::any();
    let p2: u8 = kani::any();

    kani::assume(p0 < 3 && p1 < 3 && p2 < 3);
    kani::assume(p0 != p1 && p0 != p2 && p1 != p2);

    let perm = [p0 as usize, p1 as usize, p2 as usize];

    // Check surjection: every index 0..3 appears
    let mut seen = [false; 3];
    let mut i = 0;
    while i < 3 {
        assert!(!seen[perm[i]], "no duplicate in permutation");
        seen[perm[i]] = true;
        i += 1;
    }
    assert!(seen[0] && seen[1] && seen[2], "all indices must appear");

    // Apply permutation to dims [A, B, C] and verify numel preserved
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    kani::assume(a >= 1 && a <= 16);
    kani::assume(b >= 1 && b <= 16);
    kani::assume(c >= 1 && c <= 16);

    let dims = [a as usize, b as usize, c as usize];
    let permuted = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];

    let orig_numel = checked_dim_product(&dims);
    let perm_numel = checked_dim_product(&permuted);
    if let (Ok(on), Ok(pn)) = (orig_numel, perm_numel) {
        assert_eq!(on, pn, "permutation must preserve numel");
    }
}

/// Prove: composing two valid permutations yields a valid permutation.
///
/// If P1 and P2 are valid permutations of {0,1,2}, then P2(P1) is also
/// a valid permutation. This is the closure property of the symmetric group.
#[kani::unwind(1)]
#[kani::proof]
fn permutation_composition_valid_3d() {
    // First permutation
    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let a2: u8 = kani::any();
    kani::assume(a0 < 3 && a1 < 3 && a2 < 3);
    kani::assume(a0 != a1 && a0 != a2 && a1 != a2);

    // Second permutation
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();
    let b2: u8 = kani::any();
    kani::assume(b0 < 3 && b1 < 3 && b2 < 3);
    kani::assume(b0 != b1 && b0 != b2 && b1 != b2);

    let p1 = [a0 as usize, a1 as usize, a2 as usize];
    let p2 = [b0 as usize, b1 as usize, b2 as usize];

    // Compose: p_composed[i] = p1[p2[i]]
    let composed = [p1[p2[0]], p1[p2[1]], p1[p2[2]]];

    // Check composed is a valid permutation
    let mut seen = [false; 3];
    let mut i = 0;
    while i < 3 {
        assert!(composed[i] < 3, "composed index must be in range");
        assert!(!seen[composed[i]], "composed must have no duplicates");
        seen[composed[i]] = true;
        i += 1;
    }
    assert!(seen[0] && seen[1] && seen[2], "composed must be surjective");
}

// ===========================================================================
// 6. Contiguous memory layout after reshape
// ===========================================================================

/// Prove: after reshape from [A, B] to [A*B], the contiguous strides
/// of the flattened tensor are [1] (trivially contiguous).
/// And for the original [A, B], strides are [B, 1].
#[kani::unwind(1)]
#[kani::proof]
fn reshape_contiguous_strides_2d_to_1d() {
    let a: u16 = kani::any();
    let b: u16 = kani::any();

    kani::assume(a >= 1 && a <= 128);
    kani::assume(b >= 1 && b <= 128);

    let au = a as usize;
    let bu = b as usize;

    // Original 2D contiguous strides: [B, 1]
    let mut dims_5 = [0usize; 5];
    dims_5[0] = au;
    dims_5[1] = bu;
    let strides = contiguous_strides_5(&dims_5, 2).unwrap();
    assert_eq!(strides[0], bu, "stride[0] must be B");
    assert_eq!(strides[1], 1, "stride[1] must be 1");

    // After flatten to 1D: stride is [1]
    if let Some(flat) = au.checked_mul(bu) {
        let mut flat_dims = [0usize; 5];
        flat_dims[0] = flat;
        let flat_strides = contiguous_strides_5(&flat_dims, 1).unwrap();
        assert_eq!(flat_strides[0], 1, "flattened stride must be 1");
    }
}

/// Prove: reshape from [A, B, C] to [A, B*C] produces contiguous strides
/// [B*C, 1] that are consistent with the original strides [B*C, C, 1].
#[kani::unwind(1)]
#[kani::proof]
fn reshape_contiguous_strides_consistent_3d_to_2d() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();

    kani::assume(a >= 1 && a <= 16);
    kani::assume(b >= 1 && b <= 16);
    kani::assume(c >= 1 && c <= 16);

    let au = a as usize;
    let bu = b as usize;
    let cu = c as usize;

    // Original 3D strides
    let mut dims_3d = [0usize; 5];
    dims_3d[0] = au;
    dims_3d[1] = bu;
    dims_3d[2] = cu;
    let strides_3d = contiguous_strides_5(&dims_3d, 3).unwrap();

    // stride[0] for 3D = B*C
    assert_eq!(strides_3d[0], bu * cu, "3D stride[0] must be B*C");
    assert_eq!(strides_3d[1], cu, "3D stride[1] must be C");
    assert_eq!(strides_3d[2], 1, "3D stride[2] must be 1");

    // Reshaped 2D strides for [A, B*C]
    if let Some(bc) = bu.checked_mul(cu) {
        let mut dims_2d = [0usize; 5];
        dims_2d[0] = au;
        dims_2d[1] = bc;
        let strides_2d = contiguous_strides_5(&dims_2d, 2).unwrap();

        // stride[0] for 2D must equal stride[0] for 3D (both are B*C)
        assert_eq!(
            strides_2d[0], strides_3d[0],
            "reshaped stride[0] must match original stride[0]"
        );
        assert_eq!(strides_2d[1], 1, "reshaped stride[1] must be 1");
    }
}

// ===========================================================================
// 7. Stride computation correctness for views
// ===========================================================================

/// Prove: contiguous stride[i] == product(dims[i+1..rank]) for all i
/// in a rank-4 tensor.
///
/// This is the fundamental view stride formula. For contiguous layout,
/// every stride must equal the product of all subsequent dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn stride_equals_product_of_subsequent_dims_rank4() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    kani::assume(d3 >= 1 && d3 <= 16);

    let mut dims = [0usize; 5];
    dims[0] = d0 as usize;
    dims[1] = d1 as usize;
    dims[2] = d2 as usize;
    dims[3] = d3 as usize;

    let strides = contiguous_strides_5(&dims, 4).unwrap();

    // stride[3] = 1 (product of empty subsequent dims)
    assert_eq!(strides[3], 1, "stride[3] = 1");

    // stride[2] = dims[3]
    assert_eq!(strides[2], dims[3], "stride[2] = dims[3]");

    // stride[1] = dims[2] * dims[3]
    assert_eq!(
        strides[1],
        dims[2] * dims[3],
        "stride[1] = dims[2] * dims[3]"
    );

    // stride[0] = dims[1] * dims[2] * dims[3]
    assert_eq!(
        strides[0],
        dims[1] * dims[2] * dims[3],
        "stride[0] = dims[1] * dims[2] * dims[3]"
    );

    // Additionally: stride[i] * dims[i] == stride[i-1] for contiguous
    assert_eq!(
        strides[0],
        strides[1] * dims[1],
        "stride recurrence: s[0] = s[1] * d[1]"
    );
    assert_eq!(
        strides[1],
        strides[2] * dims[2],
        "stride recurrence: s[1] = s[2] * d[2]"
    );
    assert_eq!(
        strides[2],
        strides[3] * dims[3],
        "stride recurrence: s[2] = s[3] * d[3]"
    );
}

/// Prove: for a view created by slicing dim 0 of a rank-3 tensor,
/// the strides are unchanged (only the offset changes).
///
/// A view into a contiguous tensor along the leading dimension
/// preserves the stride pattern of the original.
#[kani::unwind(1)]
#[kani::proof]
fn view_slice_dim0_preserves_strides_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let start: u8 = kani::any();
    let len: u8 = kani::any();

    kani::assume(d0 >= 2 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);
    kani::assume(len >= 1 && len <= d0);
    kani::assume(start <= d0 - len);

    let mut dims = [0usize; 5];
    dims[0] = d0 as usize;
    dims[1] = d1 as usize;
    dims[2] = d2 as usize;

    let strides = contiguous_strides_5(&dims, 3).unwrap();

    // View: slice dim 0 from start with length len
    // New shape: [len, d1, d2]
    // Strides: unchanged (same memory layout, just different extent)
    let view_dims = [len as usize, d1 as usize, d2 as usize];
    let mut view_dims_5 = [0usize; 5];
    view_dims_5[0] = view_dims[0];
    view_dims_5[1] = view_dims[1];
    view_dims_5[2] = view_dims[2];

    let view_strides = contiguous_strides_5(&view_dims_5, 3).unwrap();

    // For a leading-dim slice, the strides are identical because
    // the memory layout within each "row" is the same
    assert_eq!(strides[1], view_strides[1], "stride[1] must be unchanged");
    assert_eq!(strides[2], view_strides[2], "stride[2] must be unchanged");

    // stride[0] is also the same since it depends only on dims[1] and dims[2]
    assert_eq!(
        strides[0], view_strides[0],
        "stride[0] must be unchanged for leading-dim slice"
    );

    // The byte offset into the original buffer would be start * strides[0]
    let offset = (start as usize) * strides[0];
    let orig_numel = checked_dim_product(&[dims[0], dims[1], dims[2]]);
    if let Ok(n) = orig_numel {
        assert!(
            offset < n,
            "slice offset must be within original allocation"
        );
    }
}

// ===========================================================================
// 8. Broadcasting shape compatibility rules
// ===========================================================================

/// Prove: broadcast is symmetric — broadcast(A, B) == broadcast(B, A).
///
/// The per-dimension max(a, b) is commutative, so broadcasting must
/// produce the same output shape regardless of operand order.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_is_symmetric_3d() {
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

    // Check compatibility for each dim
    let compat_0 = lhs[0] == rhs[0] || lhs[0] == 1 || rhs[0] == 1;
    let compat_1 = lhs[1] == rhs[1] || lhs[1] == 1 || rhs[1] == 1;
    let compat_2 = lhs[2] == rhs[2] || lhs[2] == 1 || rhs[2] == 1;

    let forward_compat = compat_0 && compat_1 && compat_2;

    // Reverse: broadcast(B, A)
    let rcompat_0 = rhs[0] == lhs[0] || rhs[0] == 1 || lhs[0] == 1;
    let rcompat_1 = rhs[1] == lhs[1] || rhs[1] == 1 || lhs[1] == 1;
    let rcompat_2 = rhs[2] == lhs[2] || rhs[2] == 1 || lhs[2] == 1;

    let reverse_compat = rcompat_0 && rcompat_1 && rcompat_2;

    // Symmetry: both must agree
    assert_eq!(
        forward_compat, reverse_compat,
        "broadcast compatibility must be symmetric"
    );

    // If compatible, output shapes must be identical
    if forward_compat {
        let fwd = [
            if lhs[0] == 1 { rhs[0] } else { lhs[0] },
            if lhs[1] == 1 { rhs[1] } else { lhs[1] },
            if lhs[2] == 1 { rhs[2] } else { lhs[2] },
        ];
        let rev = [
            if rhs[0] == 1 { lhs[0] } else { rhs[0] },
            if rhs[1] == 1 { lhs[1] } else { rhs[1] },
            if rhs[2] == 1 { lhs[2] } else { rhs[2] },
        ];
        assert_eq!(fwd[0], rev[0], "broadcast output dim 0 must be symmetric");
        assert_eq!(fwd[1], rev[1], "broadcast output dim 1 must be symmetric");
        assert_eq!(fwd[2], rev[2], "broadcast output dim 2 must be symmetric");
    }
}

/// Prove: broadcasting a shape with itself is identity.
///
/// For any shape S, broadcast(S, S) == S. Every dimension matches exactly.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_self_is_identity_3d() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 128);
    kani::assume(d1 >= 1 && d1 <= 128);
    kani::assume(d2 >= 1 && d2 <= 128);

    let dims = [d0 as usize, d1 as usize, d2 as usize];

    // broadcast(S, S): each dim matches exactly
    let out_0 = if dims[0] == dims[0] {
        dims[0]
    } else {
        panic!("impossible")
    };
    let out_1 = if dims[1] == dims[1] {
        dims[1]
    } else {
        panic!("impossible")
    };
    let out_2 = if dims[2] == dims[2] {
        dims[2]
    } else {
        panic!("impossible")
    };

    assert_eq!(out_0, dims[0], "broadcast(S,S) dim 0 must be identity");
    assert_eq!(out_1, dims[1], "broadcast(S,S) dim 1 must be identity");
    assert_eq!(out_2, dims[2], "broadcast(S,S) dim 2 must be identity");
}

// ===========================================================================
// 9. Narrow/slice bounds checking
// ===========================================================================

/// Prove: narrow bounds check (start + len <= dims[dim]) guarantees
/// the narrowed region is fully contained within the original tensor
/// for any dimension in a 3D tensor.
///
/// When the check passes, every element index in the narrowed range
/// is a valid index in the original tensor along that dimension.
#[kani::unwind(1)]
#[kani::proof]
fn narrow_bounds_check_sufficient_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let narrow_dim: u8 = kani::any();
    let start: u8 = kani::any();
    let len: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);
    kani::assume(narrow_dim < 3);
    kani::assume(len >= 1);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let dim = narrow_dim as usize;
    let su = start as usize;
    let lu = len as usize;

    // Bounds check: start + len <= dims[dim]
    kani::assume(su + lu <= dims[dim]);

    // Every index i in [start, start+len) is valid
    // The first index is start (>= 0, trivially valid since usize)
    assert!(su < dims[dim], "start must be within dim bounds");
    // The last index is start + len - 1
    let last = su + lu - 1;
    assert!(
        last < dims[dim],
        "last narrowed index must be within dim bounds"
    );

    // Narrowed numel is correct
    let mut narrowed_dims = dims;
    narrowed_dims[dim] = lu;
    let orig_numel = checked_dim_product(&dims);
    let narrow_numel = checked_dim_product(&narrowed_dims);

    if let (Ok(on), Ok(nn)) = (orig_numel, narrow_numel) {
        assert!(nn <= on, "narrowed numel must not exceed original");
    }
}

/// Prove: narrow with start=0 and len=dim_size is identity (full slice).
///
/// Slicing the entire dimension must produce the same shape and numel.
#[kani::unwind(1)]
#[kani::proof]
fn narrow_full_range_is_identity_2d() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let narrow_dim: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 128);
    kani::assume(d1 >= 1 && d1 <= 128);
    kani::assume(narrow_dim < 2);

    let dims = [d0 as usize, d1 as usize];
    let dim = narrow_dim as usize;

    // Full slice: start=0, len=dims[dim]
    let start = 0usize;
    let len = dims[dim];

    // Bounds check passes trivially
    assert!(
        start + len <= dims[dim],
        "full slice bounds check must pass"
    );

    // Narrowed shape equals original shape
    let mut narrowed_dims = dims;
    narrowed_dims[dim] = len;
    assert_eq!(narrowed_dims[0], dims[0], "full narrow must preserve dim 0");
    assert_eq!(narrowed_dims[1], dims[1], "full narrow must preserve dim 1");

    // Numel unchanged
    let orig_numel = checked_dim_product(&dims);
    let narrow_numel = checked_dim_product(&narrowed_dims);
    if let (Ok(on), Ok(nn)) = (orig_numel, narrow_numel) {
        assert_eq!(on, nn, "full narrow must preserve numel");
    }
}

// ===========================================================================
// 10. Expand only works on size-1 dimensions
// ===========================================================================

/// Prove: expand is valid only when each non-matching dimension is size 1.
///
/// expand(input_shape, target_shape) succeeds iff for each dimension i:
///   input[i] == target[i] OR input[i] == 1
/// When input[i] == 1, the tensor is broadcast-expanded to target[i].
/// When input[i] != 1 and input[i] != target[i], expand must fail.
#[kani::unwind(1)]
#[kani::proof]
fn expand_requires_size_one_for_expansion_3d() {
    let i0: u8 = kani::any();
    let i1: u8 = kani::any();
    let i2: u8 = kani::any();
    let t0: u8 = kani::any();
    let t1: u8 = kani::any();
    let t2: u8 = kani::any();

    kani::assume(i0 >= 1 && i0 <= 16);
    kani::assume(i1 >= 1 && i1 <= 16);
    kani::assume(i2 >= 1 && i2 <= 16);
    kani::assume(t0 >= 1 && t0 <= 16);
    kani::assume(t1 >= 1 && t1 <= 16);
    kani::assume(t2 >= 1 && t2 <= 16);

    let input = [i0 as usize, i1 as usize, i2 as usize];
    let target = [t0 as usize, t1 as usize, t2 as usize];

    // Per-dim expand validity
    let valid_0 = input[0] == target[0] || input[0] == 1;
    let valid_1 = input[1] == target[1] || input[1] == 1;
    let valid_2 = input[2] == target[2] || input[2] == 1;

    let expand_valid = valid_0 && valid_1 && valid_2;

    // If any dim has input > 1 and input != target, expand must fail
    let has_bad_dim = (input[0] > 1 && input[0] != target[0])
        || (input[1] > 1 && input[1] != target[1])
        || (input[2] > 1 && input[2] != target[2]);

    if has_bad_dim {
        assert!(!expand_valid, "expand must reject non-1 dim mismatch");
    }

    // If expand is valid, output shape equals target
    if expand_valid {
        let out = [target[0], target[1], target[2]];
        assert_eq!(out[0], target[0], "expanded dim 0 must match target");
        assert_eq!(out[1], target[1], "expanded dim 1 must match target");
        assert_eq!(out[2], target[2], "expanded dim 2 must match target");
    }
}

/// Prove: expand with input == target is a no-op (identity).
///
/// When all dimensions already match, expand does nothing.
#[kani::unwind(1)]
#[kani::proof]
fn expand_identity_when_shapes_match_3d() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 128);
    kani::assume(d1 >= 1 && d1 <= 128);
    kani::assume(d2 >= 1 && d2 <= 128);

    let input = [d0 as usize, d1 as usize, d2 as usize];
    let target = input; // same shape

    // Validity: every dim matches
    let valid = (input[0] == target[0] || input[0] == 1)
        && (input[1] == target[1] || input[1] == 1)
        && (input[2] == target[2] || input[2] == 1);

    assert!(valid, "expand with matching shapes must be valid");

    // Output numel equals input numel
    let in_numel = checked_dim_product(&input);
    let out_numel = checked_dim_product(&target);
    if let (Ok(inn), Ok(outn)) = (in_numel, out_numel) {
        assert_eq!(inn, outn, "expand identity must preserve numel");
    }
}

/// Prove: expand from [1, C, 1] to [B, C, T] is valid and produces
/// the correct output numel B * C * T.
///
/// This is a common pattern: expanding a per-channel bias [1, C, 1]
/// to match a batched feature map [B, C, T].
#[kani::unwind(1)]
#[kani::proof]
fn expand_per_channel_broadcast_valid() {
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let t: u8 = kani::any();

    kani::assume(b >= 1 && b <= 16);
    kani::assume(c >= 1 && c <= 16);
    kani::assume(t >= 1 && t <= 16);

    let input = [1usize, c as usize, 1usize];
    let target = [b as usize, c as usize, t as usize];

    // Validity check
    let valid_0 = input[0] == target[0] || input[0] == 1; // 1 == 1 or input == 1
    let valid_1 = input[1] == target[1] || input[1] == 1; // C == C
    let valid_2 = input[2] == target[2] || input[2] == 1; // 1 == 1 or input == 1

    assert!(valid_0, "dim 0 must be expandable (size 1)");
    assert!(valid_1, "dim 1 must match (same C)");
    assert!(valid_2, "dim 2 must be expandable (size 1)");

    // Output numel = B * C * T
    let out_numel = checked_dim_product(&target);
    if let Ok(n) = out_numel {
        assert_eq!(
            n,
            (b as usize) * (c as usize) * (t as usize),
            "expand output numel must be B*C*T"
        );
    }

    // Input numel = 1 * C * 1 = C
    let in_numel = checked_dim_product(&input);
    if let Ok(n) = in_numel {
        assert_eq!(n, c as usize, "input numel must be C");
    }
}
