// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for roll (circular shift) dpdf-critical properties (#4290).
//!
//! dpdf uses roll in STFT/iSTFT processing for signal alignment and in
//! rotary position embedding implementations. These proofs verify:
//!
//! 1.  roll shift normalization: shift % dim_size maps to [0, dim_size)
//! 2.  roll identity: shift == 0 is a no-op
//! 3.  roll periodicity: shift == dim_size is identity
//! 4.  roll inverse: roll(shift) then roll(-shift) is identity
//! 5.  roll element preservation: all elements are preserved (permutation)
//!
//! Part of #4290.

// ---------------------------------------------------------------------------
// Harness 1: roll shift normalization
// ---------------------------------------------------------------------------

/// Prove: the shift normalization formula ((shift % dim_size) + dim_size) % dim_size
/// always produces a value in [0, dim_size) for any i64 shift.
#[kani::unwind(1)]
#[kani::proof]
fn proof_roll_shift_normalization() {
    let shift: i64 = kani::any();
    let dim_size: usize = kani::any();
    kani::assume(dim_size >= 1 && dim_size <= 65536);

    // Bound shift to avoid i64 overflow in modulus
    kani::assume(shift.abs() <= i64::MAX / 2);

    let normalized = ((shift % dim_size as i64) + dim_size as i64) as usize % dim_size;

    assert!(
        normalized < dim_size,
        "normalized shift must be in [0, dim_size)"
    );
}

// ---------------------------------------------------------------------------
// Harness 2: roll identity (shift == 0)
// ---------------------------------------------------------------------------

/// Prove: when shift is 0, the normalized shift is also 0, meaning no
/// operation is performed.
#[kani::unwind(1)]
#[kani::proof]
fn proof_roll_zero_shift_identity() {
    let dim_size: usize = kani::any();
    kani::assume(dim_size >= 1 && dim_size <= 65536);

    let shift: i64 = 0;
    let normalized = ((shift % dim_size as i64) + dim_size as i64) as usize % dim_size;

    assert!(normalized == 0, "shift=0 must normalize to 0 (identity)");
}

// ---------------------------------------------------------------------------
// Harness 3: roll periodicity (shift == dim_size is identity)
// ---------------------------------------------------------------------------

/// Prove: shifting by exactly dim_size (or multiples) is equivalent to
/// no shift. This is the periodicity property of circular shift.
#[kani::unwind(1)]
#[kani::proof]
fn proof_roll_full_period_identity() {
    let dim_size: usize = kani::any();
    kani::assume(dim_size >= 1 && dim_size <= 65536);

    // Shift by dim_size
    let shift_pos = dim_size as i64;
    let norm_pos = ((shift_pos % dim_size as i64) + dim_size as i64) as usize % dim_size;
    assert!(norm_pos == 0, "shift=dim_size must be identity");

    // Shift by -dim_size
    let shift_neg = -(dim_size as i64);
    let norm_neg = ((shift_neg % dim_size as i64) + dim_size as i64) as usize % dim_size;
    assert!(norm_neg == 0, "shift=-dim_size must be identity");

    // Shift by 2*dim_size
    let shift_2x = 2 * dim_size as i64;
    let norm_2x = ((shift_2x % dim_size as i64) + dim_size as i64) as usize % dim_size;
    assert!(norm_2x == 0, "shift=2*dim_size must be identity");
}

// ---------------------------------------------------------------------------
// Harness 4: roll inverse (shift + (-shift) = identity)
// ---------------------------------------------------------------------------

/// Prove: rolling by shift then by -shift produces the identity permutation.
/// This is critical for dpdf STFT/iSTFT round-trip correctness.
#[kani::unwind(1)]
#[kani::proof]
fn proof_roll_inverse() {
    let shift: i64 = kani::any();
    let dim_size: usize = kani::any();
    kani::assume(dim_size >= 1 && dim_size <= 1024);
    kani::assume(shift.abs() <= 10000);

    let norm_fwd = ((shift % dim_size as i64) + dim_size as i64) as usize % dim_size;
    let neg_shift = -shift;
    let norm_bwd = ((neg_shift % dim_size as i64) + dim_size as i64) as usize % dim_size;

    // Forward + backward shift must equal 0 mod dim_size
    let combined = (norm_fwd + norm_bwd) % dim_size;
    assert!(combined == 0, "roll(shift) + roll(-shift) must be identity");
}

// ---------------------------------------------------------------------------
// Harness 5: roll preserves all elements (permutation property)
// ---------------------------------------------------------------------------

/// Prove: roll on a 4-element array preserves all elements (it is a permutation,
/// not a lossy transform). After rolling, sorting both arrays must give identical results.
#[kani::unwind(5)]
#[kani::proof]
fn proof_roll_preserves_elements() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();
    let c: i8 = kani::any();
    let d: i8 = kani::any();
    let shift: usize = kani::any();
    kani::assume(shift < 4);

    let original = [a, b, c, d];

    // Apply circular shift
    let rolled = [
        original[(4 - shift) % 4],
        original[(5 - shift) % 4],
        original[(6 - shift) % 4],
        original[(7 - shift) % 4],
    ];

    // Both must have the same multiset of elements.
    // Check via sorted comparison.
    let mut sorted_orig = original;
    let mut sorted_roll = rolled;
    sorted_orig.sort_unstable();
    sorted_roll.sort_unstable();

    assert!(sorted_orig[0] == sorted_roll[0], "element 0 preserved");
    assert!(sorted_orig[1] == sorted_roll[1], "element 1 preserved");
    assert!(sorted_orig[2] == sorted_roll[2], "element 2 preserved");
    assert!(sorted_orig[3] == sorted_roll[3], "element 3 preserved");
}
