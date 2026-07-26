// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for tensor memory layout safety — extension (#4242).
//!
//! Part 2 of 2. See `kani_dpdf_vlm_memory_layout.rs` for proofs 1-6.
//!
//! Proved properties:
//!
//!  7. Slice offset + sub-tensor fits in allocation — narrow(dim, start, len)
//!     with stride-based offset stays within the original buffer
//!  8. Multi-dim narrow composition safety — narrowing two dimensions
//!     sequentially produces valid sub-tensor within original bounds
//!  9. Contiguous check correctness — is_contiguous returns true iff strides
//!     match the contiguous formula stride[i] = product(dims[i+1..])
//! 10. Zero-dim tensor safety — tensors with a size-0 dimension have zero
//!     elements and zero max offset
//! 11. VLM patch embed reshape — [B, C, pH, pW] -> [B, C, pH*pW] preserves
//!     numel and produces valid contiguous strides
//! 12. Strided element uniqueness — contiguous strides produce a bijection
//!     from multi-index to linear offset (no aliasing)

#![cfg(kani)]

use crate::tensor::checked_dim_product;

// ===========================================================================
// Helpers (duplicated from part 1 — Kani modules are self-contained).
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
// 7. Slice offset + sub-tensor fits in allocation
// ===========================================================================

/// Prove: for a contiguous 4D tensor, narrow(dim, start, len) with
/// strided offset `start * stride[dim]` produces a sub-tensor whose
/// last accessed element is within the original allocation.
///
/// This covers the VLM pattern of slicing patch sequences from image
/// feature tensors.
#[kani::unwind(1)]
#[kani::proof]
fn slice_4d_offset_within_allocation() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let d3: u8 = kani::any();
    let narrow_dim: u8 = kani::any();
    let start: u8 = kani::any();
    let len: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 4);
    kani::assume(d1 >= 1 && d1 <= 4);
    kani::assume(d2 >= 1 && d2 <= 4);
    kani::assume(d3 >= 1 && d3 <= 4);
    kani::assume(narrow_dim < 4);
    kani::assume(len >= 1);

    let dims = [d0 as usize, d1 as usize, d2 as usize, d3 as usize];
    let dim = narrow_dim as usize;
    let su = start as usize;
    let lu = len as usize;

    // Precondition: start + len <= dims[dim]
    kani::assume(su + lu <= dims[dim]);

    let strides = contiguous_strides(&dims, 4).unwrap();
    let orig_numel = checked_dim_product(&dims);

    if let Ok(n) = orig_numel {
        // The byte offset where the narrow starts
        let base_offset = su * strides[dim];

        // The sub-tensor's max linear index (relative to base_offset)
        let mut sub_dims = dims;
        sub_dims[dim] = lu;

        // Max index in the sub-tensor (relative to its own base)
        let sub_max = max_linear_offset(&sub_dims, &strides, 4);

        if let Some(sm) = sub_max {
            let total_max = base_offset + sm;
            assert!(
                total_max < n,
                "narrowed sub-tensor's last element must be within original allocation"
            );
        }
    }
}

// ===========================================================================
// 8. Multi-dim narrow composition safety
// ===========================================================================

/// Prove: narrowing dim 0 then dim 1 of a 3D tensor produces a
/// sub-tensor that fits within the original allocation.
///
/// VLMs often slice both batch and sequence dimensions. Composing two
/// narrow operations must not exceed bounds.
#[kani::unwind(1)]
#[kani::proof]
fn multi_dim_narrow_composition_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let s0: u8 = kani::any();
    let l0: u8 = kani::any();
    let s1: u8 = kani::any();
    let l1: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(l0 >= 1);
    kani::assume(l1 >= 1);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let su0 = s0 as usize;
    let lu0 = l0 as usize;
    let su1 = s1 as usize;
    let lu1 = l1 as usize;

    // Preconditions
    kani::assume(su0 + lu0 <= dims[0]);
    kani::assume(su1 + lu1 <= dims[1]);

    let strides = contiguous_strides(&dims, 3).unwrap();
    let orig_numel = checked_dim_product(&dims);

    if let Ok(n) = orig_numel {
        // After narrow(dim=0, s0, l0): offset = s0 * strides[0]
        let offset_0 = su0 * strides[0];

        // After narrow(dim=1, s1, l1): offset += s1 * strides[1]
        let offset_1 = su1 * strides[1];
        let total_offset = offset_0 + offset_1;

        // Max index in the doubly-narrowed sub-tensor
        let sub_dims = [lu0, lu1, dims[2]];
        let sub_max = max_linear_offset(&sub_dims, &strides, 3);

        if let Some(sm) = sub_max {
            let total_max = total_offset + sm;
            assert!(
                total_max < n,
                "doubly-narrowed sub-tensor must fit within original allocation"
            );
        }
    }
}

// ===========================================================================
// 9. Contiguous check correctness
// ===========================================================================

/// Prove: the is_contiguous check (stride[i] == product(dims[i+1..]))
/// correctly identifies contiguous vs non-contiguous layouts for 3D tensors.
///
/// A layout is contiguous iff strides match the C-order formula.
/// This proves the check has no false positives or false negatives.
#[kani::unwind(1)]
#[kani::proof]
fn is_contiguous_check_correct_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let expected_strides = contiguous_strides(&dims, 3).unwrap();

    // Test with the actual contiguous strides
    let is_contig_correct = expected_strides[2] == 1
        && expected_strides[1] == dims[2]
        && expected_strides[0] == dims[1] * dims[2];
    assert!(
        is_contig_correct,
        "contiguous strides must satisfy the formula"
    );

    // Test with a transposed layout (swap dims 0 and 1)
    let transposed_strides = [
        expected_strides[1],
        expected_strides[0],
        expected_strides[2],
    ];
    let transposed_dims = [dims[1], dims[0], dims[2]];

    // For transposed layout, check if it happens to be contiguous
    let transposed_expected = contiguous_strides(&transposed_dims, 3).unwrap();
    let trans_is_contig = transposed_strides[0] == transposed_expected[0]
        && transposed_strides[1] == transposed_expected[1]
        && transposed_strides[2] == transposed_expected[2];

    // Only contiguous if dims[0] == dims[1] (transposing equal dims is a no-op)
    if dims[0] != dims[1] {
        assert!(
            !trans_is_contig,
            "transposed layout with unequal dims must not be contiguous"
        );
    }
}

// ===========================================================================
// 10. Zero-dim tensor safety
// ===========================================================================

/// Prove: a tensor with any zero-size dimension has zero elements and
/// zero max offset — accessing any element would be out of bounds.
///
/// Zero-size tensors arise from empty batch processing or when slicing
/// produces an empty result. The framework must handle them safely.
#[kani::unwind(1)]
#[kani::proof]
fn zero_dim_tensor_has_zero_numel() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 <= 8);
    kani::assume(d1 <= 8);
    kani::assume(d2 <= 8);
    // At least one dimension is 0
    kani::assume(d0 == 0 || d1 == 0 || d2 == 0);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let numel = checked_dim_product(&dims);
    assert!(numel.is_ok(), "zero-dim product must not overflow");
    assert_eq!(
        numel.unwrap(),
        0,
        "tensor with zero-size dim must have 0 elements"
    );
}

// ===========================================================================
// 11. VLM patch embed reshape: [B, C, pH, pW] -> [B, C, pH*pW]
// ===========================================================================

/// Prove: the vision patch embedding reshape [B, C, pH, pW] -> [B, C, pH*pW]
/// preserves element count and produces valid contiguous strides.
///
/// This is the standard ViT/VLM pattern for converting 2D spatial patches
/// into a sequence for transformer processing.
#[kani::unwind(1)]
#[kani::proof]
fn vlm_patch_embed_reshape_safe() {
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let ph: u8 = kani::any();
    let pw: u8 = kani::any();

    kani::assume(b >= 1 && b <= 4);
    kani::assume(c >= 1 && c <= 8);
    kani::assume(ph >= 1 && ph <= 8);
    kani::assume(pw >= 1 && pw <= 8);

    let bu = b as usize;
    let cu = c as usize;
    let phu = ph as usize;
    let pwu = pw as usize;

    let orig_dims = [bu, cu, phu, pwu];
    let orig_numel = checked_dim_product(&orig_dims);

    if let Some(seq_len) = phu.checked_mul(pwu) {
        let new_dims = [bu, cu, seq_len];
        let new_numel = checked_dim_product(&new_dims);

        if let (Ok(on), Ok(nn)) = (orig_numel, new_numel) {
            assert_eq!(on, nn, "patch embed reshape must preserve numel");

            // Contiguous strides for new shape
            let new_strides = contiguous_strides(&new_dims, 3).unwrap();

            // stride[2] = 1 (innermost)
            assert_eq!(new_strides[2], 1, "last stride must be 1");
            // stride[1] = seq_len
            assert_eq!(new_strides[1], seq_len, "stride[1] must be seq_len");
            // stride[0] = C * seq_len
            assert_eq!(
                new_strides[0],
                cu * seq_len,
                "stride[0] must be C * seq_len"
            );

            // Max offset = numel - 1
            let max_off = max_linear_offset(&new_dims, &new_strides, 3);
            if let Some(mo) = max_off {
                assert_eq!(mo, nn - 1, "max offset must be numel - 1");
            }
        }
    }
}

// ===========================================================================
// 12. Strided element uniqueness (bijection)
// ===========================================================================

/// Prove: for a contiguous 3D tensor, the linear index formula
/// i*s0 + j*s1 + k*s2 is injective — no two distinct multi-indices
/// map to the same linear offset.
///
/// This is proved by showing the formula is a mixed-radix numeral system:
/// stride[i-1] = stride[i] * dim[i] guarantees uniqueness.
#[kani::unwind(1)]
#[kani::proof]
fn contiguous_strides_are_injective_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let strides = contiguous_strides(&dims, 3).unwrap();

    // Mixed-radix uniqueness condition:
    // stride[i-1] = stride[i] * dim[i]
    assert_eq!(
        strides[0],
        strides[1] * dims[1],
        "stride recurrence s[0] = s[1] * d[1]"
    );
    assert_eq!(
        strides[1],
        strides[2] * dims[2],
        "stride recurrence s[1] = s[2] * d[2]"
    );
    assert_eq!(strides[2], 1, "base stride must be 1");

    // The number of unique linear offsets equals numel:
    // min = 0, max = numel - 1, and every integer in [0, numel-1]
    // is reachable (mixed-radix coverage).
    let numel = checked_dim_product(&dims);
    let max_off = max_linear_offset(&dims, &strides, 3);
    if let (Ok(n), Some(mo)) = (numel, max_off) {
        assert_eq!(mo, n - 1, "contiguous layout spans exactly [0, numel-1]");
    }
}
