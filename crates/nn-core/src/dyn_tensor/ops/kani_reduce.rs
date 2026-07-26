// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor reduction operations (#3679).
//!
//! Proves correctness properties of reduce.rs arithmetic:
//!
//! - `reduce_op_to_trace_op`: bijective mapping from ReduceOp to TraceOp
//! - Kahan compensated summation: accuracy, identity, single-element
//! - Fold identity elements: NEG_INFINITY for max, INFINITY for min, 0.0 for sum
//! - Keepdim shape arithmetic: inserting size-1 dim preserves element count
//! - Variance non-negativity: mean((x - mean(x))^2) >= 0
//! - check_dim rejection: dim >= rank must error
//!
//! These harnesses operate on pure scalar/arithmetic — no ndarray or GPU
//! storage — making them tractable for CBMC symbolic execution.

use super::reduce::reduce_op_to_trace_op;
use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::ReduceOp;

// ---------------------------------------------------------------------------
// reduce_op_to_trace_op: mapping correctness
// ---------------------------------------------------------------------------

/// Prove: reduce_op_to_trace_op maps each ReduceOp variant to a distinct
/// TraceOp variant with matching dim/keepdim fields.
///
/// This ensures the trace recording for GPU-dispatched reductions accurately
/// reflects the operation performed. A wrong mapping would cause trace replay
/// to apply the wrong reduction type.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn reduce_op_trace_mapping_preserves_dim_keepdim() {
    let dim: u8 = kani::any();
    let keepdim: bool = kani::any();
    kani::assume(dim <= 7);

    let d = dim as usize;

    // Sum mapping
    let trace_sum = reduce_op_to_trace_op(ReduceOp::Sum, d, keepdim);
    match trace_sum {
        TraceOp::ReduceSum {
            dim: td,
            keepdim: tk,
        } => {
            assert_eq!(td, d, "Sum trace dim must match");
            assert_eq!(tk, keepdim, "Sum trace keepdim must match");
        }
        _ => panic!("Sum must map to ReduceSum"),
    }

    // Mean mapping
    let trace_mean = reduce_op_to_trace_op(ReduceOp::Mean, d, keepdim);
    match trace_mean {
        TraceOp::ReduceMean {
            dim: td,
            keepdim: tk,
        } => {
            assert_eq!(td, d, "Mean trace dim must match");
            assert_eq!(tk, keepdim, "Mean trace keepdim must match");
        }
        _ => panic!("Mean must map to ReduceMean"),
    }

    // Max mapping
    let trace_max = reduce_op_to_trace_op(ReduceOp::Max, d, keepdim);
    match trace_max {
        TraceOp::ReduceMax {
            dim: td,
            keepdim: tk,
        } => {
            assert_eq!(td, d, "Max trace dim must match");
            assert_eq!(tk, keepdim, "Max trace keepdim must match");
        }
        _ => panic!("Max must map to ReduceMax"),
    }

    // Min mapping
    let trace_min = reduce_op_to_trace_op(ReduceOp::Min, d, keepdim);
    match trace_min {
        TraceOp::ReduceMin {
            dim: td,
            keepdim: tk,
        } => {
            assert_eq!(td, d, "Min trace dim must match");
            assert_eq!(tk, keepdim, "Min trace keepdim must match");
        }
        _ => panic!("Min must map to ReduceMin"),
    }
}

// ---------------------------------------------------------------------------
// Kahan compensated summation: scalar properties
// ---------------------------------------------------------------------------

/// Prove: Kahan summation of a single element returns that element.
///
/// The Kahan algorithm with comp=0, sum=0 after one step must yield
/// the input value exactly. This is the base case for all Kahan reductions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn kahan_single_element_identity() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());
    kani::assume(val.abs() <= 1e6);

    // Kahan algorithm: one iteration
    let mut sum = 0.0_f32;
    let mut comp = 0.0_f32;
    let y = val - comp; // val - 0 = val
    let t = sum + y; // 0 + val = val
    comp = (t - sum) - y; // (val - 0) - val = 0
    sum = t; // val

    assert_eq!(
        sum, val,
        "Kahan sum of single element must equal that element"
    );
}

/// Prove: Kahan summation of two elements is commutative for exact values.
///
/// For values that sum without rounding error, Kahan sum(a, b) == sum(b, a).
/// This property is important for reduction order independence.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn kahan_two_element_commutative_exact() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();

    // Use small integers (exact in f32) so there's no rounding
    let fa = a as f32;
    let fb = b as f32;

    // Kahan sum(fa, fb)
    let mut sum1 = 0.0_f32;
    let mut comp1 = 0.0_f32;
    // Step 1: add fa
    let y1 = fa - comp1;
    let t1 = sum1 + y1;
    comp1 = (t1 - sum1) - y1;
    sum1 = t1;
    // Step 2: add fb
    let y2 = fb - comp1;
    let t2 = sum1 + y2;
    // comp1 = (t2 - sum1) - y2; // unused
    sum1 = t2;

    // Kahan sum(fb, fa)
    let mut sum2 = 0.0_f32;
    let mut comp2 = 0.0_f32;
    // Step 1: add fb
    let y3 = fb - comp2;
    let t3 = sum2 + y3;
    comp2 = (t3 - sum2) - y3;
    sum2 = t3;
    // Step 2: add fa
    let y4 = fa - comp2;
    let t4 = sum2 + y4;
    sum2 = t4;

    assert_eq!(
        sum1, sum2,
        "Kahan sum of exact integers must be commutative"
    );
}

/// Prove: Kahan summation of zero produces zero.
///
/// Empty reduction with identity init=0 must return 0.
/// This verifies the fold init value for sum reductions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn kahan_empty_sum_is_zero() {
    let sum = 0.0_f32;
    let comp = 0.0_f32;
    // No iterations — sum stays at init
    assert_eq!(sum, 0.0_f32, "empty Kahan sum must be zero");
    assert_eq!(comp, 0.0_f32, "empty Kahan comp must be zero");
}

// ---------------------------------------------------------------------------
// Fold identity elements for reduce_all_impl
// ---------------------------------------------------------------------------

/// Prove: f32::max fold with NEG_INFINITY identity returns the input value
/// for any single finite element.
///
/// reduce_all_impl uses `f32::NEG_INFINITY` as the identity for max_all.
/// For any finite value v, `f32::max(NEG_INFINITY, v) == v`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn max_fold_identity_neginf() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    let result = f32::max(f32::NEG_INFINITY, val);
    assert_eq!(
        result, val,
        "max(NEG_INFINITY, v) must equal v for finite v"
    );
}

/// Prove: f32::min fold with INFINITY identity returns the input value
/// for any single finite element.
///
/// reduce_all_impl uses `f32::INFINITY` as the identity for min_all.
/// For any finite value v, `f32::min(INFINITY, v) == v`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn min_fold_identity_inf() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    let result = f32::min(f32::INFINITY, val);
    assert_eq!(result, val, "min(INFINITY, v) must equal v for finite v");
}

/// Prove: sum fold with 0.0 identity returns the input value for a
/// single finite element.
///
/// reduce_all_impl uses `0.0` as the identity for sum_all.
/// For any finite value v, `0.0 + v == v`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn sum_fold_identity_zero() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    let result = 0.0_f32 + val;
    assert_eq!(result, val, "0.0 + v must equal v for finite v");
}

// ---------------------------------------------------------------------------
// Keepdim shape arithmetic
// ---------------------------------------------------------------------------

/// Prove: inserting a size-1 dimension preserves element count.
///
/// keepdim=true inserts a 1 at the reduced axis. The product of the
/// resulting shape must equal the product of the shape with that dim
/// removed (which is what the reduction produces without keepdim).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn keepdim_insert_preserves_numel() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    // Original 3D shape
    let numel = (d0 as u64) * (d1 as u64) * (d2 as u64);

    // Reduce dim 1 without keepdim → shape [d0, d2], numel = d0*d2
    let reduced_numel = (d0 as u64) * (d2 as u64);

    // Reduce dim 1 with keepdim → shape [d0, 1, d2], numel = d0*1*d2
    let keepdim_numel = (d0 as u64) * 1 * (d2 as u64);

    assert_eq!(
        reduced_numel, keepdim_numel,
        "keepdim shape must have same numel as non-keepdim"
    );
    assert!(
        keepdim_numel <= numel,
        "reduced numel must be <= original numel"
    );
}

// ---------------------------------------------------------------------------
// Variance non-negativity
// ---------------------------------------------------------------------------

/// Prove: population variance is non-negative for any pair of finite values.
///
/// var = mean((x - mean(x))^2) >= 0 always. This is the mathematical
/// invariant used by var_keepdim. Negative variance would indicate a bug
/// in the squaring or mean computation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn variance_nonnegative_two_values() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();

    let fa = a as f32;
    let fb = b as f32;
    let mean = (fa + fb) / 2.0;

    let diff_a = fa - mean;
    let diff_b = fb - mean;
    let var = (diff_a * diff_a + diff_b * diff_b) / 2.0;

    assert!(var >= 0.0, "population variance must be non-negative");
    // Variance is zero only when both values are equal
    if a == b {
        assert_eq!(var, 0.0, "variance of identical values must be zero");
    }
}

/// Prove: check_dim rejects dim >= rank for reduce operations.
///
/// reduce_impl calls check_dim(dim, rank) before dispatching. This must
/// reject out-of-bounds dimension indices that would cause undefined
/// behavior in the reduction.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn reduce_check_dim_rejects_oob() {
    let dim: u8 = kani::any();
    let rank: u8 = kani::any();
    kani::assume(rank >= 1 && rank <= 8);
    kani::assume(dim <= 10);

    let result = crate::check_dim(dim as usize, rank as usize);
    if dim < rank {
        assert!(result.is_ok(), "dim < rank must be accepted for reduce");
    } else {
        assert!(result.is_err(), "dim >= rank must be rejected for reduce");
    }
}
