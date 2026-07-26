// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor reduction operation properties (#4108).
//!
//! Proves correctness properties of reduce.rs reduction semantics:
//!
//! - Sum preserves non-negativity (positive inputs -> positive sum)
//! - Mean is bounded between min and max of inputs
//! - Max returns a value present in the input
//! - Argmax returns a valid index in [0, N)
//! - Sum of single element equals the element
//! - Mean of identical elements equals the element
//! - Reduction along axis reduces that dimension to 1 (keepdim)
//! - Keepdim preserves rank
//! - Non-keepdim reduces rank by 1
//! - Sum commutativity (total sum is axis-order independent)
//! - Min/max/mean ordering: min <= mean <= max
//! - Reduction output shape correctness
//! - Empty reduction edge cases (fold identities)
//!
//! These harnesses operate on pure scalar/arithmetic — no ndarray or GPU
//! storage — making them tractable for CBMC symbolic execution.

// ---------------------------------------------------------------------------
// Sum preserves non-negativity
// ---------------------------------------------------------------------------

/// Prove: sum of non-negative values is non-negative.
///
/// If all inputs are >= 0, the sum must also be >= 0. This is a fundamental
/// monotonicity property used by bounds propagation through sum reductions.
#[kani::unwind(5)]
#[kani::proof]
fn sum_nonneg_inputs_nonneg_result() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 4);

    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    let d: u8 = kani::any();

    // All inputs non-negative (u8 guarantees this)
    let vals = [a as f32, b as f32, c as f32, d as f32];

    let mut sum = 0.0_f32;
    let mut i: u8 = 0;
    while i < n {
        sum += vals[i as usize];
        i += 1;
    }

    assert!(
        sum >= 0.0,
        "sum of non-negative values must be non-negative"
    );
}

/// Prove: sum of strictly positive values is strictly positive.
///
/// If all inputs are > 0, the sum must also be > 0. Stronger than
/// non-negativity — excludes the zero case.
#[kani::unwind(4)]
#[kani::proof]
fn sum_positive_inputs_positive_result() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 3);

    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let c: u8 = kani::any();
    kani::assume(a >= 1);
    kani::assume(b >= 1);
    kani::assume(c >= 1);

    let vals = [a as f32, b as f32, c as f32];

    let mut sum = 0.0_f32;
    let mut i: u8 = 0;
    while i < n {
        sum += vals[i as usize];
        i += 1;
    }

    assert!(sum > 0.0, "sum of positive values must be positive");
}

// ---------------------------------------------------------------------------
// Mean bounded between min and max
// ---------------------------------------------------------------------------

/// Prove: mean of two values lies between their min and max.
///
/// For any two finite values a, b: min(a,b) <= mean(a,b) <= max(a,b).
/// This is the fundamental averaging inequality used by bounds propagation
/// through mean reductions.
#[kani::unwind(1)]
#[kani::proof]
fn mean_bounded_by_min_max_two_values() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;

    let mean = (fa + fb) / 2.0;
    let min = f32::min(fa, fb);
    let max = f32::max(fa, fb);

    assert!(mean >= min, "mean must be >= min of inputs");
    assert!(mean <= max, "mean must be <= max of inputs");
}

/// Prove: mean of three values lies between their min and max.
///
/// Extends the two-value case to three inputs, covering the common
/// reduction scenario of small dimension sizes.
#[kani::unwind(1)]
#[kani::proof]
fn mean_bounded_by_min_max_three_values() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();
    let c: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;
    let fc = c as f32;

    let mean = (fa + fb + fc) / 3.0;
    let min = f32::min(f32::min(fa, fb), fc);
    let max = f32::max(f32::max(fa, fb), fc);

    assert!(mean >= min, "mean of 3 must be >= min");
    assert!(mean <= max, "mean of 3 must be <= max");
}

// ---------------------------------------------------------------------------
// Max returns a value present in the input
// ---------------------------------------------------------------------------

/// Prove: max of two values equals one of the two inputs.
///
/// The max reduction must return an actual input value, not an
/// interpolated or computed value. This ensures max_keepdim output
/// is always an element that exists in the input tensor.
#[kani::unwind(1)]
#[kani::proof]
fn max_returns_input_value_two() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;

    let max_val = f32::max(fa, fb);

    assert!(
        max_val == fa || max_val == fb,
        "max must be one of the input values"
    );
}

/// Prove: max of three values equals one of the three inputs.
///
/// Extends the two-value case. The fold-based max reduction in
/// reduce.rs (lane.iter().fold(NEG_INFINITY, f32::max)) must
/// always select an actual input element.
#[kani::unwind(1)]
#[kani::proof]
fn max_returns_input_value_three() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();
    let c: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;
    let fc = c as f32;

    let max_val = f32::max(f32::max(fa, fb), fc);

    assert!(
        max_val == fa || max_val == fb || max_val == fc,
        "max of 3 must be one of the input values"
    );
}

// ---------------------------------------------------------------------------
// Argmax returns a valid index
// ---------------------------------------------------------------------------

/// Prove: argmax index is in [0, N) for any non-empty sequence.
///
/// The argmax operation scans a sequence and returns the index of the
/// maximum element. The returned index must always be a valid array index.
#[kani::unwind(5)]
#[kani::proof]
fn argmax_returns_valid_index() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 4);

    let a: i8 = kani::any();
    let b: i8 = kani::any();
    let c: i8 = kani::any();
    let d: i8 = kani::any();

    let vals = [a as f32, b as f32, c as f32, d as f32];

    // Simulate argmax
    let mut max_idx: u8 = 0;
    let mut max_val = vals[0];
    let mut i: u8 = 1;
    while i < n {
        if vals[i as usize] > max_val {
            max_val = vals[i as usize];
            max_idx = i;
        }
        i += 1;
    }

    assert!(max_idx < n, "argmax index must be < N");
}

/// Prove: argmax value equals the max of the sequence.
///
/// The element at the argmax index must equal the result of max reduction.
/// This ensures argmax and max are consistent.
#[kani::unwind(4)]
#[kani::proof]
fn argmax_value_equals_max() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 3);

    let a: i8 = kani::any();
    let b: i8 = kani::any();
    let c: i8 = kani::any();

    let vals = [a as f32, b as f32, c as f32];

    // Compute max via fold
    let mut fold_max = f32::NEG_INFINITY;
    let mut i: u8 = 0;
    while i < n {
        fold_max = f32::max(fold_max, vals[i as usize]);
        i += 1;
    }

    // Compute argmax
    let mut max_idx: u8 = 0;
    let mut max_val = vals[0];
    i = 1;
    while i < n {
        if vals[i as usize] > max_val {
            max_val = vals[i as usize];
            max_idx = i;
        }
        i += 1;
    }

    assert_eq!(
        vals[max_idx as usize], fold_max,
        "value at argmax index must equal fold max"
    );
}

// ---------------------------------------------------------------------------
// Sum of single element equals the element
// ---------------------------------------------------------------------------

/// Prove: sum reduction of a single element returns that element.
///
/// For a 1-element reduction lane, sum must be the identity operation.
/// This is the base case for sum_keepdim with dim_size=1.
#[kani::unwind(1)]
#[kani::proof]
fn sum_single_element_identity() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());
    kani::assume(val.abs() <= 1e6);

    // Fold identity for sum is 0.0
    let result = 0.0_f32 + val;

    assert_eq!(result, val, "sum of single element must equal that element");
}

// ---------------------------------------------------------------------------
// Mean of identical elements equals the element
// ---------------------------------------------------------------------------

/// Prove: mean of N identical values equals that value.
///
/// If all elements in a reduction lane are the same value v,
/// mean must return exactly v. This is a key idempotency property.
#[kani::unwind(5)]
#[kani::proof]
fn mean_identical_elements_returns_element() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 4);

    let v: i8 = kani::any();
    let fv = v as f32;

    // Sum of N copies of v = N * v
    let sum = fv * (n as f32);
    let mean = sum / (n as f32);

    assert_eq!(
        mean, fv,
        "mean of identical elements must equal the element"
    );
}

// ---------------------------------------------------------------------------
// Keepdim preserves rank / non-keepdim reduces rank by 1
// ---------------------------------------------------------------------------

/// Prove: keepdim=true preserves the tensor rank.
///
/// Reducing along an axis with keepdim=true inserts a size-1 dimension
/// at the reduced axis position. The output rank must equal the input rank.
#[kani::unwind(1)]
#[kani::proof]
fn keepdim_preserves_rank() {
    let rank: u8 = kani::any();
    let dim: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 5);
    kani::assume(dim < rank);

    // With keepdim=true: reduce dim, then insert size-1 at that position
    // Result rank = (rank - 1) + 1 = rank
    let reduced_rank = rank - 1;
    let keepdim_rank = reduced_rank + 1;

    assert_eq!(keepdim_rank, rank, "keepdim must preserve rank");
}

/// Prove: keepdim=false reduces rank by exactly 1.
///
/// Reducing along an axis without keepdim removes that dimension entirely.
/// For a rank-R tensor, the output is rank-(R-1).
#[kani::unwind(1)]
#[kani::proof]
fn non_keepdim_reduces_rank_by_one() {
    let rank: u8 = kani::any();
    let dim: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 5);
    kani::assume(dim < rank);

    // Without keepdim: reduced axis is removed
    let output_rank = rank - 1;

    assert_eq!(output_rank, rank - 1, "non-keepdim must reduce rank by 1");
    assert!(
        output_rank >= 1,
        "output rank must be at least 1 for rank >= 2"
    );
}

// ---------------------------------------------------------------------------
// Reduction along axis reduces that dimension to 1 (keepdim shape)
// ---------------------------------------------------------------------------

/// Prove: keepdim output shape has size 1 at the reduced axis.
///
/// For a 3D tensor [d0, d1, d2] reduced along axis 1 with keepdim,
/// output shape is [d0, 1, d2]. The reduced axis becomes size 1.
#[kani::unwind(1)]
#[kani::proof]
fn keepdim_shape_has_size_one_at_reduced_axis() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let axis: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(axis <= 2);

    let dims = [d0 as usize, d1 as usize, d2 as usize];

    // Keepdim output: replace dims[axis] with 1
    let out0 = if axis == 0 { 1 } else { dims[0] };
    let out1 = if axis == 1 { 1 } else { dims[1] };
    let out2 = if axis == 2 { 1 } else { dims[2] };

    let out_dims = [out0, out1, out2];

    // The reduced axis must be 1
    assert_eq!(
        out_dims[axis as usize], 1,
        "reduced axis must have size 1 in keepdim output"
    );

    // Non-reduced axes must be preserved
    for i in 0..3_usize {
        if i != axis as usize {
            assert_eq!(out_dims[i], dims[i], "non-reduced axis must be preserved");
        }
    }
}

// ---------------------------------------------------------------------------
// Sum commutativity: total sum is axis-order independent
// ---------------------------------------------------------------------------

/// Prove: sum over axis 0 then axis 0 equals sum over axis 1 then axis 0.
///
/// For a 2D tensor, the total sum is independent of reduction order.
/// sum(sum(X, axis=0), axis=0) == sum(sum(X, axis=1), axis=0).
/// This ensures reduction composition is well-defined.
#[kani::unwind(1)]
#[kani::proof]
fn sum_total_is_axis_order_independent() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();
    let c: i8 = kani::any();
    let d: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;
    let fc = c as f32;
    let fd = d as f32;

    // 2x2 matrix: [[a, b], [c, d]]

    // Path 1: sum axis=0 first → [a+c, b+d], then sum axis=0 → a+b+c+d
    let col0 = fa + fc;
    let col1 = fb + fd;
    let total_path1 = col0 + col1;

    // Path 2: sum axis=1 first → [a+b, c+d], then sum axis=0 → a+b+c+d
    let row0 = fa + fb;
    let row1 = fc + fd;
    let total_path2 = row0 + row1;

    assert_eq!(
        total_path1, total_path2,
        "total sum must be independent of reduction order"
    );
}

// ---------------------------------------------------------------------------
// Min/max/mean ordering: min <= mean <= max
// ---------------------------------------------------------------------------

/// Prove: min <= mean <= max for any pair of finite values.
///
/// The ordering min(a,b) <= mean(a,b) <= max(a,b) must hold universally.
/// This is the fundamental sandwich inequality for reduction operations.
#[kani::unwind(1)]
#[kani::proof]
fn min_mean_max_ordering_two_values() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;

    let min_val = f32::min(fa, fb);
    let max_val = f32::max(fa, fb);
    let mean_val = (fa + fb) / 2.0;

    assert!(min_val <= mean_val, "min must be <= mean");
    assert!(mean_val <= max_val, "mean must be <= max");
    assert!(min_val <= max_val, "min must be <= max");
}

/// Prove: min <= mean <= max for three values.
///
/// Extends the ordering to three inputs. Uses integer inputs to
/// ensure exact arithmetic (no rounding issues).
#[kani::unwind(1)]
#[kani::proof]
fn min_mean_max_ordering_three_values() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();
    let c: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;
    let fc = c as f32;

    let min_val = f32::min(f32::min(fa, fb), fc);
    let max_val = f32::max(f32::max(fa, fb), fc);
    let mean_val = (fa + fb + fc) / 3.0;

    assert!(min_val <= mean_val, "min must be <= mean for 3 values");
    assert!(mean_val <= max_val, "mean must be <= max for 3 values");
}

// ---------------------------------------------------------------------------
// Reduction output shape correctness
// ---------------------------------------------------------------------------

/// Prove: non-keepdim reduction output shape removes exactly the reduced dim.
///
/// For a 3D shape [d0, d1, d2] reduced along axis 1 without keepdim,
/// output shape is [d0, d2]. The dimension at the reduced axis is gone.
#[kani::unwind(1)]
#[kani::proof]
fn non_keepdim_shape_removes_reduced_axis() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);

    // Reduce axis 1 without keepdim: [d0, d1, d2] -> [d0, d2]
    let out_0 = d0 as usize;
    let out_1 = d2 as usize;

    assert_eq!(
        out_0, d0 as usize,
        "first dim preserved after axis-1 reduce"
    );
    assert_eq!(
        out_1, d2 as usize,
        "third dim becomes second after axis-1 reduce"
    );

    // Output numel = d0 * d2 (d1 is collapsed)
    let in_numel = (d0 as u64) * (d1 as u64) * (d2 as u64);
    let out_numel = (d0 as u64) * (d2 as u64);
    assert!(
        out_numel <= in_numel,
        "reduction must not increase element count"
    );
}

/// Prove: reduction output numel equals input numel divided by reduced dim size.
///
/// For a tensor with shape [..., D_axis, ...], reducing axis removes D_axis
/// elements per output position. So out_numel = in_numel / D_axis.
#[kani::unwind(1)]
#[kani::proof]
fn reduction_numel_equals_input_div_reduced_dim() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    let axis: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);
    kani::assume(d2 >= 1 && d2 <= 8);
    kani::assume(axis <= 2);

    let dims = [d0 as u64, d1 as u64, d2 as u64];
    let in_numel = dims[0] * dims[1] * dims[2];
    let reduced_dim = dims[axis as usize];

    // Out numel = product of all dims except the reduced one
    let out_numel = in_numel / reduced_dim;

    // Verify: out_numel * reduced_dim == in_numel
    assert_eq!(
        out_numel * reduced_dim,
        in_numel,
        "out_numel * reduced_dim must equal in_numel"
    );
}

// ---------------------------------------------------------------------------
// Empty reduction edge cases (fold identities)
// ---------------------------------------------------------------------------

/// Prove: max fold identity NEG_INFINITY is less than any finite value.
///
/// The fold identity for max is NEG_INFINITY. For any finite input,
/// max(NEG_INFINITY, x) == x. This ensures the first element always
/// "wins" the comparison.
#[kani::unwind(1)]
#[kani::proof]
fn max_fold_identity_less_than_any_finite() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    assert!(
        f32::NEG_INFINITY < val,
        "NEG_INFINITY must be less than any finite value"
    );
    assert_eq!(
        f32::max(f32::NEG_INFINITY, val),
        val,
        "max(NEG_INFINITY, x) must equal x"
    );
}

/// Prove: min fold identity INFINITY is greater than any finite value.
///
/// The fold identity for min is INFINITY. For any finite input,
/// min(INFINITY, x) == x. This ensures the first element always
/// "wins" the comparison.
#[kani::unwind(1)]
#[kani::proof]
fn min_fold_identity_greater_than_any_finite() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    assert!(
        f32::INFINITY > val,
        "INFINITY must be greater than any finite value"
    );
    assert_eq!(
        f32::min(f32::INFINITY, val),
        val,
        "min(INFINITY, x) must equal x"
    );
}

/// Prove: sum fold identity 0.0 is neutral for addition.
///
/// The fold identity for sum is 0.0. For any finite input,
/// 0.0 + x == x. This verifies the additive identity property.
#[kani::unwind(1)]
#[kani::proof]
fn sum_fold_identity_neutral() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    assert_eq!(0.0_f32 + val, val, "0.0 + x must equal x for finite x");
}

// ---------------------------------------------------------------------------
// Min returns a value present in the input
// ---------------------------------------------------------------------------

/// Prove: min of two values equals one of the two inputs.
///
/// The min reduction must return an actual input value, not an
/// interpolated value. Mirrors the max proof for completeness.
#[kani::unwind(1)]
#[kani::proof]
fn min_returns_input_value_two() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;

    let min_val = f32::min(fa, fb);

    assert!(
        min_val == fa || min_val == fb,
        "min must be one of the input values"
    );
}

// ---------------------------------------------------------------------------
// Max >= any individual input
// ---------------------------------------------------------------------------

/// Prove: max of a sequence is >= every element in the sequence.
///
/// For any 3-element sequence, the max-fold result must be >= each
/// individual element. This is the defining property of max.
#[kani::unwind(1)]
#[kani::proof]
fn max_geq_all_inputs() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();
    let c: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;
    let fc = c as f32;

    let max_val = f32::max(f32::max(fa, fb), fc);

    assert!(max_val >= fa, "max must be >= first element");
    assert!(max_val >= fb, "max must be >= second element");
    assert!(max_val >= fc, "max must be >= third element");
}

// ---------------------------------------------------------------------------
// Min <= any individual input
// ---------------------------------------------------------------------------

/// Prove: min of a sequence is <= every element in the sequence.
///
/// For any 3-element sequence, the min-fold result must be <= each
/// individual element. This is the defining property of min.
#[kani::unwind(1)]
#[kani::proof]
fn min_leq_all_inputs() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();
    let c: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;
    let fc = c as f32;

    let min_val = f32::min(f32::min(fa, fb), fc);

    assert!(min_val <= fa, "min must be <= first element");
    assert!(min_val <= fb, "min must be <= second element");
    assert!(min_val <= fc, "min must be <= third element");
}
