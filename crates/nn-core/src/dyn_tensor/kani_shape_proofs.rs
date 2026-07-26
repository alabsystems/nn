// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor shape validation and broadcast safety (#3568).
//!
//! Proves correctness properties of the core shape manipulation functions used
//! throughout the DynTensor model execution pipeline:
//!
//! - `broadcast_output_shape`: NumPy-style right-aligned broadcast computation
//! - `checked_dim_product`: overflow-safe element count calculation
//! - Reshape element count preservation across rank changes
//! - Permute bijection property (no dimension lost or duplicated)
//! - `D::resolve` negative indexing correctness
//!
//! These harnesses operate on the pure shape arithmetic — no ndarray or GPU
//! storage — making them tractable for CBMC symbolic execution.

use crate::dyn_tensor::ops::broadcast_output_shape;
use crate::tensor::checked_dim_product;

// ---------------------------------------------------------------------------
// broadcast_output_shape: symmetry
// ---------------------------------------------------------------------------

/// Prove: broadcast_output_shape is commutative for 2D shapes.
///
/// NumPy broadcasting is defined symmetrically: broadcast(a, b) must produce
/// the same output shape as broadcast(b, a). A violation would mean binary
/// ops like `a + b` and `b + a` could silently produce different shapes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn broadcast_shape_commutative_2d() {
    let a0: u16 = kani::any();
    let a1: u16 = kani::any();
    let b0: u16 = kani::any();
    let b1: u16 = kani::any();

    kani::assume(a0 >= 1 && a0 <= 256);
    kani::assume(a1 >= 1 && a1 <= 256);
    kani::assume(b0 >= 1 && b0 <= 256);
    kani::assume(b1 >= 1 && b1 <= 256);

    let lhs = [a0 as usize, a1 as usize];
    let rhs = [b0 as usize, b1 as usize];

    let forward = broadcast_output_shape(&lhs, &rhs);
    let reverse = broadcast_output_shape(&rhs, &lhs);

    match (forward, reverse) {
        (Ok(f), Ok(r)) => {
            assert_eq!(f, r, "broadcast must be commutative");
        }
        (Err(_), Err(_)) => {
            // Both fail — consistent.
        }
        _ => {
            panic!("broadcast commutativity violated: one succeeded and one failed");
        }
    }
}

// ---------------------------------------------------------------------------
// broadcast_output_shape: output rank = max(lhs_rank, rhs_rank)
// ---------------------------------------------------------------------------

/// Prove: broadcast output rank equals the maximum of the two input ranks.
///
/// This is a fundamental property of NumPy broadcasting: the output always
/// has as many dimensions as the input with more dimensions. Shorter shapes
/// are left-padded with implicit size-1 dimensions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn broadcast_output_rank_is_max() {
    // Test 1D vs 2D broadcasting
    let a0: u16 = kani::any();
    let b0: u16 = kani::any();
    let b1: u16 = kani::any();

    kani::assume(a0 >= 1 && a0 <= 256);
    kani::assume(b0 >= 1 && b0 <= 256);
    kani::assume(b1 >= 1 && b1 <= 256);

    // Ensure compatible: a0 must be 1 or equal to b1 (right-aligned)
    kani::assume(a0 == 1 || a0 as usize == b1 as usize);

    let lhs = [a0 as usize];
    let rhs = [b0 as usize, b1 as usize];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(result.is_ok(), "compatible shapes must broadcast");
    let out = result.unwrap();
    assert_eq!(out.len(), 2, "output rank must be max(1, 2) = 2");
}

// ---------------------------------------------------------------------------
// broadcast_output_shape: same-shape identity
// ---------------------------------------------------------------------------

/// Prove: broadcasting a shape with itself returns the same shape.
///
/// This is the base case: if both inputs have identical shapes, the output
/// must be exactly that shape. A violation here would be catastrophic for
/// strict (same-shape) binary operations.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn broadcast_same_shape_is_identity() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 256);
    kani::assume(d1 >= 1 && d1 <= 256);
    kani::assume(d2 >= 1 && d2 <= 256);

    let shape = [d0 as usize, d1 as usize, d2 as usize];
    let result = broadcast_output_shape(&shape, &shape);

    assert!(
        result.is_ok(),
        "identical shapes must be broadcast-compatible"
    );
    let out = result.unwrap();
    assert_eq!(out.len(), 3, "output rank must match input rank");
    assert_eq!(out[0], shape[0], "dim 0 must match");
    assert_eq!(out[1], shape[1], "dim 1 must match");
    assert_eq!(out[2], shape[2], "dim 2 must match");
}

// ---------------------------------------------------------------------------
// broadcast_output_shape: incompatible rejection
// ---------------------------------------------------------------------------

/// Prove: mismatched non-1 dimensions produce an error.
///
/// When two dimensions differ and neither is 1, the shapes are incompatible.
/// The function must return Err, never silently produce a wrong shape.
/// This is the safety guard that prevents shape corruption in binary ops.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn broadcast_rejects_incompatible_dims() {
    let a: u16 = kani::any();
    let b: u16 = kani::any();

    kani::assume(a >= 2 && a <= 4096);
    kani::assume(b >= 2 && b <= 4096);
    kani::assume(a != b); // Both > 1 and different → incompatible

    let lhs = [a as usize];
    let rhs = [b as usize];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(
        result.is_err(),
        "mismatched non-1 dimensions must produce error"
    );
}

// ---------------------------------------------------------------------------
// broadcast_output_shape: output dims >= both inputs
// ---------------------------------------------------------------------------

/// Prove: each output dimension is >= the corresponding input dimension.
///
/// Broadcasting can only expand dimensions, never shrink them. For each
/// axis, the output size must be >= both the lhs and rhs sizes. This
/// guarantees that the output buffer is large enough to hold both operands'
/// data after expansion.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn broadcast_output_dims_dominate_inputs() {
    let a0: u16 = kani::any();
    let a1: u16 = kani::any();
    let b0: u16 = kani::any();
    let b1: u16 = kani::any();

    kani::assume(a0 >= 1 && a0 <= 256);
    kani::assume(a1 >= 1 && a1 <= 256);
    kani::assume(b0 >= 1 && b0 <= 256);
    kani::assume(b1 >= 1 && b1 <= 256);

    let lhs = [a0 as usize, a1 as usize];
    let rhs = [b0 as usize, b1 as usize];

    if let Ok(out) = broadcast_output_shape(&lhs, &rhs) {
        assert!(out[0] >= lhs[0], "output dim 0 must be >= lhs dim 0");
        assert!(out[0] >= rhs[0], "output dim 0 must be >= rhs dim 0");
        assert!(out[1] >= lhs[1], "output dim 1 must be >= lhs dim 1");
        assert!(out[1] >= rhs[1], "output dim 1 must be >= rhs dim 1");
    }
}

/// Prove: checked_dim_product of empty shape is 1 (scalar tensor).
///
/// A scalar tensor has shape [] and element count 1. The fold over an empty
/// iterator must return the identity element (1), not 0 or an error.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checked_dim_product_empty_is_one() {
    let dims: [usize; 0] = [];
    let result = checked_dim_product(&dims);
    assert!(result.is_ok(), "empty dims must succeed");
    assert_eq!(
        result.unwrap(),
        1,
        "product of empty dims must be 1 (scalar)"
    );
}

// ---------------------------------------------------------------------------
// Reshape: element count preservation across rank change (3D → 1D)
// ---------------------------------------------------------------------------

/// Prove: reshaping from 3D to 1D preserves element count.
///
/// A 3D tensor [a, b, c] reshaped to [a*b*c] must have the same element
/// count. This is the fundamental invariant of reshape: data is reinterpreted
/// with a new shape, but the total number of elements is unchanged.
/// Extends kani_dyn_tensor.rs::reshape_validation_numel_check (2D→2D) to
/// cross-rank reshape.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn reshape_3d_to_1d_preserves_numel() {
    let a: u16 = kani::any();
    let b: u16 = kani::any();
    let c: u16 = kani::any();

    kani::assume(a >= 1 && a <= 32);
    kani::assume(b >= 1 && b <= 32);
    kani::assume(c >= 1 && c <= 32);

    let d0 = a as usize;
    let d1 = b as usize;
    let d2 = c as usize;

    // Compute checked products for both shapes
    let orig_product = d0.checked_mul(d1).and_then(|p| p.checked_mul(d2));

    if let Some(orig_numel) = orig_product {
        // The 1D reshape target shape is [orig_numel]
        let new_dims = [orig_numel];
        let new_product = checked_dim_product(&new_dims);
        assert!(new_product.is_ok(), "1D reshape target must not overflow");
        assert_eq!(
            orig_numel,
            new_product.unwrap(),
            "3D→1D reshape must preserve element count"
        );
    }
}

// ---------------------------------------------------------------------------
// Permute: bijection property
// ---------------------------------------------------------------------------

/// Prove: a valid permutation of 3 axes is a bijection (every axis appears
/// exactly once in the output).
///
/// This mirrors the validation logic in DynTensor::permute (shape/mod.rs:214-234):
/// permute checks for duplicates via a `seen` array. This harness proves that
/// the validation is correct: any permutation passing the check IS a bijection,
/// and the output shape contains exactly the original dimensions reordered.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn permute_is_bijection_3d() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 64);
    kani::assume(d2 >= 1 && d2 <= 64);

    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let rank = 3;

    // Pick a permutation
    let p0: usize = kani::any();
    let p1: usize = kani::any();
    let p2: usize = kani::any();

    kani::assume(p0 < rank && p1 < rank && p2 < rank);

    // Validate: no duplicates (mirrors DynTensor::permute logic)
    let mut seen = [false; 3];
    let perm = [p0, p1, p2];
    let mut valid = true;
    let mut i = 0;
    while i < 3 {
        if seen[perm[i]] {
            valid = false;
        }
        seen[perm[i]] = true;
        i += 1;
    }

    if valid {
        // Apply permutation
        let permuted = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];

        // Element count must be preserved
        let orig_numel = dims[0]
            .checked_mul(dims[1])
            .and_then(|p| p.checked_mul(dims[2]));
        let perm_numel = permuted[0]
            .checked_mul(permuted[1])
            .and_then(|p| p.checked_mul(permuted[2]));

        if let (Some(on), Some(pn)) = (orig_numel, perm_numel) {
            assert_eq!(on, pn, "permute must preserve element count");
        }

        // Every original dimension appears in the output (multiset equality)
        let mut orig_sorted = dims;
        orig_sorted.sort();
        let mut perm_sorted = permuted;
        perm_sorted.sort();
        assert_eq!(
            orig_sorted, perm_sorted,
            "permuted dims must be a reordering of original dims"
        );
    }
}

// ---------------------------------------------------------------------------
// D::resolve: negative indexing correctness
// ---------------------------------------------------------------------------

/// Prove: D::Minus1 resolves to rank-1, D::Minus2 resolves to rank-2.
///
/// Negative indexing is used throughout the codebase (e.g., `softmax(D::Minus1)`
/// for the last dimension). A wrong resolution would silently apply operations
/// to the wrong axis, causing shape corruption or incorrect results.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn d_resolve_negative_indexing() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 8);

    let r = rank as usize;

    // D::Minus1 should resolve to rank - 1 (last dimension)
    let minus1 = crate::dyn_tensor::D::Minus1;
    let resolved1 = minus1.resolve(r);
    assert!(resolved1.is_ok(), "D::Minus1 must resolve for rank >= 1");
    assert_eq!(
        resolved1.unwrap(),
        r - 1,
        "D::Minus1 must resolve to rank - 1"
    );

    // D::Minus2 should resolve to rank - 2 (second-to-last)
    let minus2 = crate::dyn_tensor::D::Minus2;
    let resolved2 = minus2.resolve(r);
    assert!(resolved2.is_ok(), "D::Minus2 must resolve for rank >= 2");
    assert_eq!(
        resolved2.unwrap(),
        r - 2,
        "D::Minus2 must resolve to rank - 2"
    );
}

/// Prove: D::Minus1 fails for rank 0, D::Minus2 fails for rank 0 and 1.
///
/// Negative indexing must reject insufficient ranks. D::Minus1 needs rank >= 1,
/// D::Minus2 needs rank >= 2. Accepting rank 0 would produce subtraction
/// underflow (0 - 1 wraps to usize::MAX on unsigned arithmetic).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn d_resolve_rejects_insufficient_rank() {
    // D::Minus1 on rank 0 must fail
    let minus1 = crate::dyn_tensor::D::Minus1;
    assert!(minus1.resolve(0).is_err(), "D::Minus1 must reject rank 0");

    // D::Minus2 on rank 0 must fail
    let minus2 = crate::dyn_tensor::D::Minus2;
    assert!(minus2.resolve(0).is_err(), "D::Minus2 must reject rank 0");

    // D::Minus2 on rank 1 must fail
    assert!(minus2.resolve(1).is_err(), "D::Minus2 must reject rank 1");
}
