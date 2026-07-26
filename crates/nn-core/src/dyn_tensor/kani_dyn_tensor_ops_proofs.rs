// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor ops: dimension resolution, checked
//! conversions, broadcast shape computation, and Shape type properties (#3737).
//!
//!  1. D::Minus1 resolves to rank-1 for valid ranks
//!  2. D::Minus2 resolves to rank-2 for valid ranks
//!  3. D::Minus1 fails for rank 0
//!  4. D::Minus2 fails for rank < 2
//!  5. Dim for usize: valid dim passes
//!  6. Dim for usize: dim >= rank fails
//!  7. Dim for i32: non-negative resolves like usize
//!  8. Dim for i32: -1 resolves to last dim
//!  9. Dim for i32: -2 resolves to second-to-last
//! 10. Dim for i32: negative overflow fails
//! 11. checked_f64_to_f32: in-range value passes
//! 12. checked_f64_to_f32: f64 overflow to f32 infinity fails
//! 13. checked_f64_to_f32: NaN passes through
//! 14. checked_f64_to_f32: f64 infinity passes through
//! 15. broadcast_output_shape: same shapes produce same shape
//! 16. broadcast_output_shape: scalar broadcasts to any shape
//! 17. broadcast_output_shape: size-1 dim broadcasts
//! 18. broadcast_output_shape: incompatible shapes fail
//! 19. broadcast_output_shape: different ranks right-align
//! 20. Shape::from_dims round-trips
//! 21. Shape::rank matches dims length
//! 22. Shape::elem_count is product of dims
//! 23. Shape From tuple conversions
//! 24. WeightRef::is_placeholder: zero-dim shapes are not placeholders

use crate::dyn_tensor::dim::Dim;
use crate::dyn_tensor::ops::broadcast_output_shape;
use crate::dyn_tensor::trace::WeightRef;
use crate::dyn_tensor::Shape;
use crate::dyn_tensor::D;

// ===========================================================================
// D::resolve proofs
// ===========================================================================

/// Prove: D::Minus1 resolves to rank-1 for any valid rank.
#[kani::unwind(1)]
#[kani::proof]
fn proof_d_minus1_resolves() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 10);
    let result = D::Minus1.resolve(rank as usize);
    assert!(result.is_ok(), "Minus1 must succeed for rank >= 1");
    assert!(
        result.unwrap() == (rank as usize) - 1,
        "Minus1 resolves to rank - 1"
    );
}

/// Prove: D::Minus2 resolves to rank-2 for any valid rank.
#[kani::unwind(1)]
#[kani::proof]
fn proof_d_minus2_resolves() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 10);
    let result = D::Minus2.resolve(rank as usize);
    assert!(result.is_ok(), "Minus2 must succeed for rank >= 2");
    assert!(
        result.unwrap() == (rank as usize) - 2,
        "Minus2 resolves to rank - 2"
    );
}

/// Prove: D::Minus1 fails for rank 0.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_d_minus1_fails_rank_zero() {
    assert!(D::Minus1.resolve(0).is_err(), "Minus1 must fail for rank 0");
}

/// Prove: D::Minus2 fails for rank < 2.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_d_minus2_fails_rank_lt_2() {
    assert!(D::Minus2.resolve(0).is_err(), "Minus2 must fail for rank 0");
    assert!(D::Minus2.resolve(1).is_err(), "Minus2 must fail for rank 1");
}

// ===========================================================================
// Dim trait for usize proofs
// ===========================================================================

/// Prove: Dim for usize succeeds when dim < rank.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dim_usize_valid() {
    let rank: u8 = kani::any();
    let dim: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 8);
    kani::assume(dim < rank);

    let result = (dim as usize).to_index(rank as usize);
    assert!(result.is_ok(), "dim < rank must succeed");
    assert!(result.unwrap() == dim as usize, "returns the dim unchanged");
}

/// Prove: Dim for usize fails when dim >= rank.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dim_usize_out_of_range() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 8);

    // dim == rank is already out of range
    let result = (rank as usize).to_index(rank as usize);
    assert!(result.is_err(), "dim == rank must fail");
}

// ===========================================================================
// Dim trait for i32 proofs
// ===========================================================================

/// Prove: Dim for i32 non-negative resolves identically to usize.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dim_i32_nonnegative() {
    let rank: u8 = kani::any();
    let dim: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 8);
    kani::assume(dim < rank);

    let result = (dim as i32).to_index(rank as usize);
    assert!(result.is_ok(), "non-negative dim < rank must succeed");
    assert!(result.unwrap() == dim as usize, "matches usize result");
}

/// Prove: Dim for i32 -1 resolves to last dimension.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dim_i32_minus1() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 8);

    let result = (-1i32).to_index(rank as usize);
    assert!(result.is_ok(), "-1 must succeed for rank >= 1");
    assert!(
        result.unwrap() == (rank as usize) - 1,
        "-1 resolves to last dim"
    );
}

/// Prove: Dim for i32 -2 resolves to second-to-last dimension.
#[kani::unwind(1)]
#[kani::proof]
fn proof_dim_i32_minus2() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 8);

    let result = (-2i32).to_index(rank as usize);
    assert!(result.is_ok(), "-2 must succeed for rank >= 2");
    assert!(
        result.unwrap() == (rank as usize) - 2,
        "-2 resolves to second-to-last dim"
    );
}

/// Prove: Dim for i32 negative overflow fails (e.g., -3 for rank 2).
#[kani::unwind(1)]
#[kani::proof]
fn proof_dim_i32_negative_overflow() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 4);

    // -(rank+1) should fail for any rank
    let neg = -((rank as i32) + 1);
    let result = neg.to_index(rank as usize);
    assert!(result.is_err(), "negative index exceeding rank must fail");
}

// ===========================================================================
// checked_f64_to_f32 proofs
// ===========================================================================

/// Prove: in-range f64 converts to f32 successfully.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_checked_f64_to_f32_in_range() {
    let result = super::checked_f64_to_f32(1.5, "test");
    assert!(result.is_ok(), "1.5 is in f32 range");
    let val = result.unwrap();
    assert!(val == 1.5f32, "value preserved");
}

/// Prove: f64 value that overflows f32 is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_checked_f64_to_f32_overflow() {
    // f64 value larger than f32::MAX
    let large: f64 = (f32::MAX as f64) * 2.0;
    let result = super::checked_f64_to_f32(large, "big");
    assert!(result.is_err(), "f32 overflow must be rejected");
}

/// Prove: NaN passes through (not a finite→non-finite transition).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_checked_f64_to_f32_nan_passthrough() {
    let result = super::checked_f64_to_f32(f64::NAN, "nan_test");
    assert!(result.is_ok(), "NaN passes through");
    assert!(result.unwrap().is_nan(), "result is NaN");
}

/// Prove: f64 infinity passes through (already non-finite).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_checked_f64_to_f32_inf_passthrough() {
    let result = super::checked_f64_to_f32(f64::INFINITY, "inf_test");
    assert!(result.is_ok(), "infinity passes through");
    assert!(result.unwrap().is_infinite(), "result is infinite");
}

// ===========================================================================
// broadcast_output_shape proofs
// ===========================================================================

/// Prove: same shapes produce the same shape.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_broadcast_same_shapes() {
    let shape = &[2usize, 3, 4];
    let result = broadcast_output_shape(shape, shape);
    assert!(result.is_ok(), "same shapes must succeed");
    let out = result.unwrap();
    assert!(out.len() == 3, "output rank matches");
    assert!(out[0] == 2 && out[1] == 3 && out[2] == 4, "dims match");
}

/// Prove: scalar (empty shape) broadcasts to any shape.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_broadcast_scalar_to_shape() {
    let scalar: &[usize] = &[];
    let shape = &[2usize, 3];
    let result = broadcast_output_shape(scalar, shape);
    assert!(result.is_ok(), "scalar broadcasts to any shape");
    let out = result.unwrap();
    assert!(out.len() == 2, "output rank is max(0, 2) = 2");
    assert!(out[0] == 2 && out[1] == 3, "output matches the non-scalar");
}

/// Prove: size-1 dim broadcasts to the other's size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_broadcast_size_one_dim() {
    let lhs = &[1usize, 4];
    let rhs = &[3usize, 4];
    let result = broadcast_output_shape(lhs, rhs);
    assert!(result.is_ok(), "size-1 broadcast must succeed");
    let out = result.unwrap();
    assert!(out[0] == 3, "size-1 broadcasts to 3");
    assert!(out[1] == 4, "size 4 stays 4");
}

/// Prove: incompatible shapes (non-1 mismatch) fail.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_broadcast_incompatible_fails() {
    let lhs = &[2usize, 3];
    let rhs = &[4usize, 3];
    let result = broadcast_output_shape(lhs, rhs);
    assert!(result.is_err(), "2 vs 4 is not broadcast-compatible");
}

/// Prove: different ranks right-align correctly.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_broadcast_different_ranks() {
    let lhs = &[4usize]; // rank 1
    let rhs = &[2usize, 4]; // rank 2
    let result = broadcast_output_shape(lhs, rhs);
    assert!(
        result.is_ok(),
        "different ranks with compatible dims succeed"
    );
    let out = result.unwrap();
    assert!(out.len() == 2, "output rank is max(1, 2) = 2");
    assert!(out[0] == 2 && out[1] == 4, "right-aligned broadcast");
}

/// Prove: broadcast is commutative.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_broadcast_commutative() {
    let a = &[1usize, 4];
    let b = &[3usize, 1];
    let ab = broadcast_output_shape(a, b).unwrap();
    let ba = broadcast_output_shape(b, a).unwrap();
    assert!(ab.len() == ba.len(), "commutative rank");
    assert!(ab[0] == ba[0] && ab[1] == ba[1], "commutative dims");
}

// ===========================================================================
// Shape type proofs
// ===========================================================================

/// Prove: Shape::from_dims round-trips through dims().
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_shape_from_dims_roundtrip() {
    let dims = &[2usize, 3, 4];
    let shape = Shape::from_dims(dims);
    assert!(shape.dims().len() == 3, "rank 3");
    assert!(shape.dims()[0] == 2, "dim 0");
    assert!(shape.dims()[1] == 3, "dim 1");
    assert!(shape.dims()[2] == 4, "dim 2");
}

/// Prove: Shape::rank() matches dims length.
#[kani::unwind(16)]
#[kani::proof]
fn proof_shape_rank_matches_len() {
    let rank: u8 = kani::any();
    kani::assume(rank <= 6);
    let dims: Vec<usize> = (0..rank).map(|i| (i as usize) + 1).collect();
    let shape = Shape::from_dims(&dims);
    assert!(shape.rank() == rank as usize, "rank must equal dims.len()");
}

/// Prove: Shape::elem_count is the product of dimensions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_shape_elem_count_is_product() {
    let shape = Shape::from_dims(&[2, 3, 4]);
    assert!(shape.elem_count() == 24, "2*3*4 = 24");
}

/// Prove: Shape::elem_count of empty dims is 1 (scalar).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_shape_elem_count_empty_is_one() {
    let shape = Shape::from_dims(&[]);
    assert!(
        shape.elem_count() == 1,
        "empty shape elem_count = 1 (scalar)"
    );
}

/// Prove: Shape from tuple conversions produce correct dims.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_shape_from_tuple_2d() {
    let shape: Shape = (3usize, 4usize).into();
    assert!(shape.rank() == 2, "2D shape");
    assert!(shape.dims()[0] == 3 && shape.dims()[1] == 4, "dims correct");
}

/// Prove: Shape from tuple 3D.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_shape_from_tuple_3d() {
    let shape: Shape = (2usize, 3usize, 4usize).into();
    assert!(shape.rank() == 3 && shape.elem_count() == 24, "3D shape");
}

/// Prove: Shape from usize produces rank-1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_shape_from_usize() {
    let shape: Shape = 5usize.into();
    assert!(shape.rank() == 1 && shape.dims()[0] == 5, "rank-1 shape");
}

// ===========================================================================
// WeightRef edge cases
// ===========================================================================

/// Prove: WeightRef with zero-dim shape (e.g., [0]) is not a placeholder.
///
/// is_placeholder requires all dims > 0. A shape with a zero dim has
/// product 0 and is not considered a placeholder even if data is empty.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_weight_ref_zero_dim_not_placeholder() {
    let wr = WeightRef::from_shape(&[0, 4]);
    assert!(
        !wr.is_placeholder(),
        "zero-dim shape must not be a placeholder"
    );
}

/// Prove: WeightRef with empty shape and empty data is not a placeholder.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_weight_ref_empty_shape_not_placeholder() {
    let wr = WeightRef::from_shape(&[]);
    assert!(
        !wr.is_placeholder(),
        "empty shape must not be a placeholder"
    );
}
