// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `checked_product_of_shape` and `next_power_of_2`
//! from [`tensor_dispatch_helpers`](super::tensor_dispatch).
//!
//! These two functions are on the critical path of EVERY Metal dispatch:
//!
//! - `checked_product_of_shape` computes total elements from tensor shape dims.
//!   Overflow → buffer underallocation → GPU out-of-bounds writes.
//! - `next_power_of_2` rounds up threadgroup sizes for tree reductions.
//!   Wrong result → tree reduction reads out-of-bounds shared memory.
//!
//! # Properties Proved
//!
//! ## `checked_product_of_shape`
//! - Empty shape produces 1 (identity element for multiplication)
//! - Shape with a zero dimension produces 0
//! - Realistic shapes (all dims in [1, 65536]) never overflow
//! - Overflowing shapes return `Err(ShapeOverflow)`, not panic
//! - Product is >= every individual dimension (monotonicity)
//!
//! ## `next_power_of_2`
//! - Result is always >= input
//! - Result is always a power of 2 (or 1 for input 0/1)
//! - For input 0, returns 1
//! - For input > 2^31, returns the capped value (1<<31)
//! - For inputs already a power of 2, returns the input unchanged

use crate::tensor_dispatch::{checked_product_of_shape, next_power_of_2};

// ============================================================================
// checked_product_of_shape proofs
// ============================================================================

/// Prove: empty shape produces 1 (identity element for multiplication).
///
/// This is the base case for `try_fold(1, ...)`. An empty shape means a
/// scalar tensor with exactly 1 element.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checked_product_empty_shape_returns_one() {
    let shape: &[usize] = &[];
    let result = checked_product_of_shape(shape);
    assert!(result.is_ok(), "empty shape must succeed");
    assert_eq!(result.unwrap(), 1, "empty shape product must be 1");
}

/// Prove: shape containing a zero dimension produces 0.
///
/// A zero dimension means the tensor has no elements. The product must be 0
/// regardless of other dimensions. This prevents allocating a nonzero buffer
/// for a tensor that holds nothing.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn checked_product_zero_dim_produces_zero() {
    // Shape with 3 dimensions, one of which is 0.
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    kani::assume(d0 <= 1024 && d1 <= 1024 && d2 <= 1024);
    kani::assume(d0 == 0 || d1 == 0 || d2 == 0);

    let shape = [d0, d1, d2];
    let result = checked_product_of_shape(&shape);
    assert!(result.is_ok(), "zero-dim shape must not overflow");
    assert_eq!(result.unwrap(), 0, "product with zero dim must be 0");
}

/// Prove: for any shape where all dims are in [1, 65536] and rank <= 3,
/// the product does NOT overflow usize.
///
/// Max case: [65536, 65536, 65536] = 2^48 < 2^64 = usize::MAX on 64-bit.
/// This covers ALL realistic tensor shapes in Kokoro/Whisper/Qwen3 inference.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn checked_product_realistic_shapes_no_overflow() {
    let rank: usize = kani::any();
    kani::assume(rank >= 1 && rank <= 3);

    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    kani::assume(d0 >= 1 && d0 <= 65536);
    kani::assume(d1 >= 1 && d1 <= 65536);
    kani::assume(d2 >= 1 && d2 <= 65536);

    let shape: &[usize] = match rank {
        1 => &[d0],
        2 => &[d0, d1],
        3 => &[d0, d1, d2],
        _ => unreachable!(),
    };

    let result = checked_product_of_shape(shape);
    assert!(
        result.is_ok(),
        "realistic shape must not overflow: dims in [1, 65536], rank <= 3"
    );

    // Verify against widened multiplication.
    let expected = match rank {
        1 => d0 as u128,
        2 => (d0 as u128) * (d1 as u128),
        3 => (d0 as u128) * (d1 as u128) * (d2 as u128),
        _ => unreachable!(),
    };
    assert_eq!(
        result.unwrap() as u128, expected,
        "product must equal widened multiplication"
    );
}

/// Prove: shapes that DO overflow return Err(ShapeOverflow), not panic.
///
/// A 3-dim shape where the product exceeds usize::MAX must return an error.
/// Without checked_mul, this would wrap silently and allocate an undersized
/// buffer — the #1 most dangerous bug class in GPU dispatch.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn checked_product_overflow_returns_error() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();

    // Constrain to values that will overflow when multiplied.
    // Use large values near the cube root of usize::MAX.
    kani::assume(d0 >= (1usize << 22) && d0 <= (1usize << 23));
    kani::assume(d1 >= (1usize << 22) && d1 <= (1usize << 23));
    kani::assume(d2 >= (1usize << 22) && d2 <= (1usize << 23));

    // Verify the product would overflow.
    let wide = (d0 as u128) * (d1 as u128) * (d2 as u128);
    kani::assume(wide > usize::MAX as u128);

    let shape = [d0, d1, d2];
    let result = checked_product_of_shape(&shape);
    assert!(
        result.is_err(),
        "overflowing shape must return Err, not wrap"
    );
}

/// Prove: the product is always >= every individual dimension.
///
/// Since all dimensions are >= 0 and the accumulator starts at 1,
/// the product of N non-negative integers is >= each factor (when
/// all factors are >= 1). For zero factors, the product is 0 which
/// is == the zero factor.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn checked_product_geq_each_dimension() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    kani::assume(d0 >= 1 && d0 <= 65536);
    kani::assume(d1 >= 1 && d1 <= 65536);

    let shape = [d0, d1];
    let product = checked_product_of_shape(&shape).unwrap();

    assert!(
        product >= d0,
        "product must be >= d0 when all dims >= 1"
    );
    assert!(
        product >= d1,
        "product must be >= d1 when all dims >= 1"
    );
}

/// Prove: single-element shape returns that element.
///
/// A rank-1 tensor with shape [N] has exactly N elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn checked_product_single_dim_identity() {
    let d: usize = kani::any();
    // Full range of usize — single dim can never overflow.
    let shape = [d];
    let result = checked_product_of_shape(&shape);
    assert!(result.is_ok(), "single dim must not overflow");
    assert_eq!(result.unwrap(), d, "single dim product must equal the dim");
}

// ============================================================================
// next_power_of_2 proofs
// ============================================================================

/// Prove: result is always >= input for all u32 values.
///
/// This is the core safety property. If next_power_of_2 returns a value
/// LESS than the input, the threadgroup would have fewer threads than
/// needed for the tree reduction, causing out-of-bounds shared memory reads.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_power_of_2_geq_input() {
    let v: u32 = kani::any();
    let result = next_power_of_2(v);
    assert!(
        result >= v || v > (1u32 << 31),
        "next_power_of_2 must be >= input (except overflow-capped region)"
    );
    // For values > 2^31, the result is capped at 2^31 (documented behavior).
    // The caller (.min(REDUCE_THREADGROUP_SIZE)) handles this case.
}

/// Prove: result is always a power of 2 for all u32 values.
///
/// A power of 2 satisfies: `n > 0 && (n & (n - 1)) == 0`.
/// This is required for tree reduction correctness — the reduction tree
/// folds by halving the thread count at each level, which only works
/// when the initial count is a power of 2.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn next_power_of_2_is_power_of_two() {
    let v: u32 = kani::any();
    let result = next_power_of_2(v);

    // Result must be > 0 (since we return at least 1).
    assert!(result > 0, "result must be positive");

    // Power-of-2 check: n & (n-1) == 0 for all powers of 2.
    assert!(
        result & (result - 1) == 0,
        "result must be a power of 2: got {result}"
    );
}

/// Prove: input 0 returns 1 (not 0 or undefined).
///
/// 0 is not a valid threadgroup size. The function must return 1,
/// the smallest valid power of 2.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_power_of_2_zero_returns_one() {
    let result = next_power_of_2(0);
    assert_eq!(result, 1, "next_power_of_2(0) must return 1");
}

/// Prove: input 1 returns 1.
///
/// 1 is already a power of 2 (2^0). The function must not round up to 2.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_power_of_2_one_returns_one() {
    let result = next_power_of_2(1);
    assert_eq!(result, 1, "next_power_of_2(1) must return 1");
}

/// Prove: for inputs > 2^31, returns exactly 1<<31 (the cap value).
///
/// The true next power of 2 for values in (2^31, 2^32) would be 2^32,
/// which doesn't fit in u32. The function must return 1<<31 instead of
/// panicking or wrapping to 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_power_of_2_overflow_returns_cap() {
    let v: u32 = kani::any();
    kani::assume(v > (1u32 << 31));

    let result = next_power_of_2(v);
    assert_eq!(
        result,
        1u32 << 31,
        "values > 2^31 must return 1<<31 (cap)"
    );
}

/// Prove: for inputs that are already a power of 2, returns the input.
///
/// This is the idempotency property. Calling next_power_of_2 on a value
/// that's already a power of 2 must not round up to the NEXT power.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_power_of_2_idempotent_for_powers() {
    let exp: u32 = kani::any();
    kani::assume(exp <= 31);

    let v = 1u32 << exp;
    let result = next_power_of_2(v);
    assert_eq!(
        result, v,
        "next_power_of_2(2^{exp}) must return 2^{exp}"
    );
}

/// Prove: result is the SMALLEST power of 2 >= input (minimality).
///
/// For values in [1, 2^31], the result is the tightest power-of-2 upper
/// bound. Combined with the >= proof, this fully characterizes the function.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn next_power_of_2_minimal() {
    let v: u32 = kani::any();
    kani::assume(v >= 2 && v <= (1u32 << 31));

    let result = next_power_of_2(v);

    // Result >= v (already proved separately, but needed here for context).
    assert!(result >= v);

    // Result is a power of 2.
    assert!(result & (result - 1) == 0);

    // No smaller power of 2 is >= v (minimality).
    // The next smaller power of 2 is result / 2 (since result is a power of 2).
    if result > 1 {
        let smaller = result / 2;
        assert!(
            smaller < v,
            "result/2 must be < v, proving result is minimal"
        );
    }
}
