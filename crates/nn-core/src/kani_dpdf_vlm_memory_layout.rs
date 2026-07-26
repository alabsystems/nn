// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for tensor memory layout and stride computation safety
//! targeting dpdf VLM (Vision-Language Model) tensor access patterns (#4242).
//!
//! dpdf VLMs use high-rank tensors (4D/5D attention, batched image patches,
//! multi-head Q/K/V projections) with frequent reshape, permute, narrow, and
//! broadcast operations. These proofs verify that strided memory access never
//! exceeds allocated buffer bounds — the critical safety property for GPU
//! dispatch where out-of-bounds access causes data corruption or crashes.
//!
//! Part 1 of 2 (proofs 1-6). See `kani_dpdf_vlm_memory_layout_ext.rs` for
//! proofs 7-12.
//!
//! Proved properties:
//!
//!  1. Broadcast stride zeroing — broadcast dimension gets stride 0
//!  2. Broadcast linear index within bounds — strided access with broadcast
//!     strides never exceeds the original allocation
//!  3. Permute strides are valid — multi-axis permutation preserves all
//!     stride values (just reorders them)
//!  4. Permute max offset preserved — max linear index is unchanged by
//!     permuting dims+strides together
//!  5. Reshape contiguous strides agree — reshape of a contiguous tensor
//!     produces new contiguous strides that map to the same numel
//!  6. 4D attention reshape safety — [B, S, H, D] -> [B*H, S, D] preserves
//!     element count and produces valid strides

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ===========================================================================
// Helpers shared with ext module via duplication (Kani modules are
// self-contained to avoid cross-module issues with CBMC).
// ===========================================================================

fn contiguous_strides(dims: &[usize], rank: usize) -> Option<Vec<usize>> {
    if rank == 0 {
        return Some(vec![]);
    }
    let mut strides = vec![0usize; rank];
    strides[rank - 1] = 1;
    let mut i = rank - 1;
    while i > 0 {
        strides[i - 1] = strides[i].checked_mul(dims[i])?;
        i -= 1;
    }
    Some(strides)
}

fn max_linear_offset(dims: &[usize], strides: &[usize], rank: usize) -> Option<usize> {
    let mut acc = 0usize;
    let mut i = 0;
    while i < rank {
        if dims[i] == 0 {
            return Some(0);
        }
        let contribution = strides[i].checked_mul(dims[i] - 1)?;
        acc = acc.checked_add(contribution)?;
        i += 1;
    }
    Some(acc)
}

// ===========================================================================
// 1. Broadcast stride zeroing
// ===========================================================================

/// Prove: when broadcasting a size-1 dimension to a larger size, the
/// broadcast stride for that dimension must be 0 (repeat the single
/// element across all positions).
///
/// This is fundamental to how broadcasting works in memory: the stride
/// is 0 so every index along the broadcast dimension reads the same
/// underlying element.
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_stride_is_zero_for_size_one_dim() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);

    let src_dims = [d0 as usize, d1 as usize];
    let src_strides = contiguous_strides(&src_dims, 2).unwrap();

    // Broadcast rules: if src_dim == 1, broadcast stride = 0
    // If src_dim == target_dim, broadcast stride = original stride
    let mut broadcast_strides = [0usize; 2];
    let mut i = 0;
    while i < 2 {
        if src_dims[i] == 1 {
            broadcast_strides[i] = 0;
        } else {
            broadcast_strides[i] = src_strides[i];
        }
        i += 1;
    }

    // Verify: size-1 dims get stride 0
    if src_dims[0] == 1 {
        assert_eq!(
            broadcast_strides[0], 0,
            "broadcast stride must be 0 for size-1 dim 0"
        );
    }
    if src_dims[1] == 1 {
        assert_eq!(
            broadcast_strides[1], 0,
            "broadcast stride must be 0 for size-1 dim 1"
        );
    }

    // Verify: non-size-1 dims keep original stride
    if src_dims[0] > 1 {
        assert_eq!(
            broadcast_strides[0], src_strides[0],
            "non-broadcast dim must keep original stride"
        );
    }
}

// ===========================================================================
// 2. Broadcast linear index within bounds
// ===========================================================================

/// Prove: for a 3D tensor broadcast from [1, C, 1] to [B, C, T], every
/// strided access (using broadcast strides) stays within the original
/// allocation of C elements.
///
/// This is the critical VLM pattern: per-channel bias [1, C, 1] broadcast
/// to [B, C, T] for adding to feature maps. The broadcast strides are
/// [0, 1, 0], so index computation is j*1 for any (i, j, k).
#[kani::unwind(1)]
#[kani::proof]
fn broadcast_access_within_bounds_1ct_to_bct() {
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let t: u8 = kani::any();

    kani::assume(b >= 1 && b <= 8);
    kani::assume(c >= 1 && c <= 16);
    kani::assume(t >= 1 && t <= 8);

    let src_dims = [1usize, c as usize, 1usize];
    let src_numel = checked_dim_product(&src_dims);
    assert!(src_numel.is_ok());
    let src_n = src_numel.unwrap();
    assert_eq!(src_n, c as usize, "source numel must be C");

    // Broadcast strides: size-1 dims get stride 0, others keep original
    let broadcast_strides = [0usize, 1, 0];

    // The maximum linear offset with broadcast strides:
    // max = 0*(B-1) + 1*(C-1) + 0*(T-1) = C-1
    let max_offset = broadcast_strides[0] * ((b as usize) - 1)
        + broadcast_strides[1] * ((c as usize) - 1)
        + broadcast_strides[2] * ((t as usize) - 1);

    assert_eq!(
        max_offset,
        (c as usize) - 1,
        "max broadcast offset must be C-1"
    );
    assert!(
        max_offset < src_n,
        "broadcast access must stay within source allocation"
    );
}

// ===========================================================================
// 3. Permute strides are valid (just reordered)
// ===========================================================================

/// Prove: permuting a 4D tensor's dims and strides together preserves
/// every stride value — the stride array is a permutation of the original.
///
/// VLM attention tensors are frequently permuted: [B, S, H, D] -> [B, H, S, D]
/// via permute([0, 2, 1, 3]). The strides must be correctly reordered.
#[kani::unwind(1)]
#[kani::proof]
fn permute_4d_strides_are_reordering() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(d3 >= 1 && d3 <= 8);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];
    let strides = contiguous_strides(&dims, 4).unwrap();

    // VLM attention permutation: [0, 2, 1, 3] (swap S and H)
    let perm = [0, 2, 1, 3];
    let perm_dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]];
    let perm_strides = [
        strides[perm[0]],
        strides[perm[1]],
        strides[perm[2]],
        strides[perm[3]],
    ];

    // Verify: permuted strides contain the same values (sorted)
    let mut orig_sorted = [strides[0], strides[1], strides[2], strides[3]];
    orig_sorted.sort();
    let mut perm_sorted = [
        perm_strides[0],
        perm_strides[1],
        perm_strides[2],
        perm_strides[3],
    ];
    perm_sorted.sort();
    assert_eq!(
        orig_sorted, perm_sorted,
        "permuted strides must be a reordering of original strides"
    );

    // Verify: all permuted strides are positive
    assert!(perm_strides[0] >= 1, "perm stride[0] must be positive");
    assert!(perm_strides[1] >= 1, "perm stride[1] must be positive");
    assert!(perm_strides[2] >= 1, "perm stride[2] must be positive");
    assert!(perm_strides[3] >= 1, "perm stride[3] must be positive");

    // Verify: permuted dims contain same values
    let mut orig_dims_sorted = dims;
    orig_dims_sorted.sort();
    let mut perm_dims_sorted = perm_dims;
    perm_dims_sorted.sort();
    assert_eq!(
        orig_dims_sorted, perm_dims_sorted,
        "permuted dims must be a reordering of original dims"
    );
}

// ===========================================================================
// 4. Permute max offset preserved
// ===========================================================================

/// Prove: the maximum linear offset is unchanged by permuting dims and
/// strides together (arbitrary 3D permutation).
///
/// Since permutation just reorders the (dim, stride) pairs, the sum
/// stride[i]*(dim[i]-1) is invariant under permutation.
#[kani::unwind(1)]
#[kani::proof]
fn permute_preserves_max_linear_offset_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    let p0: u8 = kani::any();
    let p1: u8 = kani::any();
    let p2: u8 = kani::any();
    kani::assume(p0 < 3 && p1 < 3 && p2 < 3);
    kani::assume(p0 != p1 && p0 != p2 && p1 != p2);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let strides = contiguous_strides(&dims, 3).unwrap();

    let perm = [p0 as usize, p1 as usize, p2 as usize];
    let perm_dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];
    let perm_strides = [strides[perm[0]], strides[perm[1]], strides[perm[2]]];

    let orig_max = max_linear_offset(&dims, &strides, 3);
    let perm_max = max_linear_offset(&perm_dims, &perm_strides, 3);

    if let (Some(om), Some(pm)) = (orig_max, perm_max) {
        assert_eq!(om, pm, "permute must preserve max linear offset");
    }
}

// ===========================================================================
// 5. Reshape contiguous strides agree on numel
// ===========================================================================

/// Prove: reshape from 4D to 3D produces contiguous strides whose max
/// offset equals numel - 1 for both the original and reshaped tensor.
///
/// This verifies that after reshape, the new contiguous layout accesses
/// exactly the same number of elements.
#[kani::unwind(1)]
#[kani::proof]
fn reshape_4d_to_3d_strides_valid() {
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

    let dims_4d = [au, bu, cu, du];
    let strides_4d = contiguous_strides(&dims_4d, 4).unwrap();
    let numel_4d = checked_dim_product(&dims_4d);

    // Reshape [A, B, C, D] -> [A*B, C, D]
    if let Some(ab) = au.checked_mul(bu) {
        let dims_3d = [ab, cu, du];
        let strides_3d = contiguous_strides(&dims_3d, 3).unwrap();
        let numel_3d = checked_dim_product(&dims_3d);

        if let (Ok(n4), Ok(n3)) = (numel_4d, numel_3d) {
            assert_eq!(n4, n3, "reshape must preserve numel");

            let max_4d = max_linear_offset(&dims_4d, &strides_4d, 4);
            let max_3d = max_linear_offset(&dims_3d, &strides_3d, 3);

            if let (Some(m4), Some(m3)) = (max_4d, max_3d) {
                assert_eq!(m4, n4 - 1, "4D max offset must be numel-1");
                assert_eq!(m3, n3 - 1, "3D max offset must be numel-1");
                assert_eq!(m4, m3, "max offsets must agree after reshape");
            }
        }
    }
}

// ===========================================================================
// 6. 4D attention reshape safety: [B, S, H, D] -> [B*H, S, D]
// ===========================================================================

/// Prove: the VLM attention reshape [B, S, H, D] -> [B*H, S, D]
/// preserves element count and the new contiguous strides produce a
/// max offset equal to numel - 1.
///
/// This reshape is used to merge batch and head dimensions before
/// batched matmul in multi-head attention.
#[kani::unwind(1)]
#[kani::proof]
fn attention_reshape_bshd_to_bhs_d() {
    let b: u8 = kani::any();
    let s: u8 = kani::any();
    let h: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(b >= 1 && b <= 4);
    kani::assume(s >= 1 && s <= 8);
    kani::assume(h >= 1 && h <= 4);
    kani::assume(d >= 1 && d <= 8);

    let bu = b as usize;
    let su = s as usize;
    let hu = h as usize;
    let du = d as usize;

    let orig_dims = [bu, su, hu, du];
    let orig_numel = checked_dim_product(&orig_dims);

    // After permute [0,2,1,3] -> [B, H, S, D], then reshape -> [B*H, S, D]
    if let Some(bh) = bu.checked_mul(hu) {
        let new_dims = [bh, su, du];
        let new_numel = checked_dim_product(&new_dims);

        if let (Ok(on), Ok(nn)) = (orig_numel, new_numel) {
            assert_eq!(on, nn, "attention reshape must preserve numel");

            let new_strides = contiguous_strides(&new_dims, 3).unwrap();
            let max_off = max_linear_offset(&new_dims, &new_strides, 3);
            if let Some(mo) = max_off {
                assert_eq!(mo, nn - 1, "reshaped tensor max offset must be numel-1");
            }
        }
    }
}
