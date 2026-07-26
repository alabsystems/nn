// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor binary ops shape broadcasting (#4107).
//!
//! Proves correctness properties of binary operation broadcasting:
//!
//!  1. Same-shape add produces identical output shape
//!  2. Same-shape mul produces identical output shape
//!  3. Scalar-tensor broadcast for add (rank 0 + rank N)
//!  4. Scalar-tensor broadcast for div (rank 0 + rank N)
//!  5. Trailing dimension broadcast: [N,1] + [N,M] -> [N,M]
//!  6. Leading dimension broadcast: [1,M] + [N,M] -> [N,M]
//!  7. Multi-dimension broadcast: [1,N,1] + [M,N,P] -> [M,N,P]
//!  8. Rank mismatch broadcast: [M] + [N,M] -> [N,M]
//!  9. Division by zero: broadcast_output_shape is shape-only (always succeeds for compatible shapes)
//! 10. Output shape correctness: output dims are max of input dims
//! 11. Commutativity of add broadcast shapes
//! 12. Commutativity of mul broadcast shapes
//! 13. Incompatible trailing dims rejected for sub
//! 14. Incompatible trailing dims rejected for div
//! 15. Broadcast output rank is max of input ranks
//! 16. Self-broadcast is identity (same shape broadcast with itself)
//! 17. All-ones 2D with arbitrary 2D broadcast
//! 18. Rank 0 + rank 0 broadcast produces rank 0
//! 19. Partial broadcast: [1,M] + [N,1] -> [N,M]
//! 20. Broadcast shape element count is product of output dims
//!
//! These harnesses operate on the pure `broadcast_output_shape` function
//! and on shape arithmetic only -- no ndarray, no GPU, no tensor allocation.
//! This makes them tractable for CBMC symbolic execution.

use crate::dyn_tensor::ops::broadcast_output_shape;

// ---------------------------------------------------------------------------
// 1. Same-shape add produces identical output shape
// ---------------------------------------------------------------------------

/// Prove: when both operands have the same 3D shape, broadcast_output_shape
/// returns that exact shape. This is the no-broadcast fast path used by
/// strict_add.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_same_shape_add_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 4);
    kani::assume(d1 >= 1 && d1 <= 4);
    kani::assume(d2 >= 1 && d2 <= 4);

    let shape = [d0 as usize, d1 as usize, d2 as usize];
    let result = broadcast_output_shape(&shape, &shape);
    assert!(result.is_ok(), "same-shape broadcast must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 3, "rank must be preserved");
    assert_eq!(out[0], d0 as usize, "dim 0 unchanged");
    assert_eq!(out[1], d1 as usize, "dim 1 unchanged");
    assert_eq!(out[2], d2 as usize, "dim 2 unchanged");
}

// ---------------------------------------------------------------------------
// 2. Same-shape mul produces identical output shape
// ---------------------------------------------------------------------------

/// Prove: same-shape broadcasting for a 2D shape preserves the shape exactly.
/// Mul uses the same broadcast_output_shape as add -- this verifies the
/// shape function is op-independent.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_same_shape_mul_2d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 4);
    kani::assume(d1 >= 1 && d1 <= 4);

    let shape = [d0 as usize, d1 as usize];
    let result = broadcast_output_shape(&shape, &shape);
    assert!(result.is_ok(), "same-shape 2D broadcast must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 2, "rank must be 2");
    assert_eq!(out[0], d0 as usize, "dim 0 unchanged");
    assert_eq!(out[1], d1 as usize, "dim 1 unchanged");
}

// ---------------------------------------------------------------------------
// 3. Scalar-tensor broadcast for add (rank 0 + rank 3)
// ---------------------------------------------------------------------------

/// Prove: broadcasting a rank-0 scalar with a rank-3 tensor produces the
/// tensor's shape. This is the mechanism behind `tensor.add_scalar()`.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_scalar_tensor_add_rank0_rank3() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 4);
    kani::assume(d1 >= 1 && d1 <= 4);
    kani::assume(d2 >= 1 && d2 <= 4);

    let scalar: [usize; 0] = [];
    let tensor = [d0 as usize, d1 as usize, d2 as usize];

    let result = broadcast_output_shape(&scalar, &tensor);
    assert!(result.is_ok(), "rank-0 + rank-3 must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 3, "output rank must be 3");
    assert_eq!(out[0], d0 as usize, "dim 0 from tensor");
    assert_eq!(out[1], d1 as usize, "dim 1 from tensor");
    assert_eq!(out[2], d2 as usize, "dim 2 from tensor");
}

// ---------------------------------------------------------------------------
// 4. Scalar-tensor broadcast for div (rank 0 + rank 2)
// ---------------------------------------------------------------------------

/// Prove: broadcasting a rank-0 scalar with a rank-2 tensor produces the
/// tensor's shape. Division uses the same shape logic as addition.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_scalar_tensor_div_rank0_rank2() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 4);
    kani::assume(d1 >= 1 && d1 <= 4);

    let scalar: [usize; 0] = [];
    let tensor = [d0 as usize, d1 as usize];

    // scalar / tensor shape
    let result = broadcast_output_shape(&scalar, &tensor);
    assert!(result.is_ok(), "rank-0 / rank-2 shape must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 2, "output rank must be 2");
    assert_eq!(out[0], d0 as usize, "dim 0 from tensor");
    assert_eq!(out[1], d1 as usize, "dim 1 from tensor");

    // tensor / scalar shape (commutative for shape computation)
    let rev = broadcast_output_shape(&tensor, &scalar);
    assert!(rev.is_ok(), "rank-2 / rank-0 shape must succeed");
    let rev_out = rev.unwrap();
    assert_eq!(out, rev_out, "div shape must be commutative");
}

// ---------------------------------------------------------------------------
// 5. Trailing dimension broadcast: [N,1] + [N,M] -> [N,M]
// ---------------------------------------------------------------------------

/// Prove: trailing dim broadcast works. [N,1] + [N,M] must produce [N,M].
/// This is the pattern for per-row bias addition in neural networks.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_trailing_dim_n1_nm() {
    let n: u8 = kani::any();
    let m: u8 = kani::any();

    kani::assume(n >= 1 && n <= 4);
    kani::assume(m >= 1 && m <= 4);

    let lhs = [n as usize, 1usize];
    let rhs = [n as usize, m as usize];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(result.is_ok(), "[N,1] + [N,M] must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 2, "output rank must be 2");
    assert_eq!(out[0], n as usize, "dim 0 must be N");
    assert_eq!(out[1], m as usize, "dim 1 must be M");
}

// ---------------------------------------------------------------------------
// 6. Leading dimension broadcast: [1,M] + [N,M] -> [N,M]
// ---------------------------------------------------------------------------

/// Prove: leading dim broadcast works. [1,M] + [N,M] must produce [N,M].
/// This is the pattern for per-column scaling in layer normalization.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_leading_dim_1m_nm() {
    let n: u8 = kani::any();
    let m: u8 = kani::any();

    kani::assume(n >= 1 && n <= 4);
    kani::assume(m >= 1 && m <= 4);

    let lhs = [1usize, m as usize];
    let rhs = [n as usize, m as usize];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(result.is_ok(), "[1,M] + [N,M] must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 2, "output rank must be 2");
    assert_eq!(out[0], n as usize, "dim 0 must be N");
    assert_eq!(out[1], m as usize, "dim 1 must be M");
}

// ---------------------------------------------------------------------------
// 7. Multi-dimension broadcast: [1,N,1] + [M,N,P] -> [M,N,P]
// ---------------------------------------------------------------------------

/// Prove: multi-dimension broadcast where multiple dims are 1.
/// [1,N,1] + [M,N,P] must produce [M,N,P]. This tests that broadcasting
/// handles multiple expansion dimensions simultaneously.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_multi_dim_1n1_mnp() {
    let m: u8 = kani::any();
    let n: u8 = kani::any();
    let p: u8 = kani::any();

    kani::assume(m >= 1 && m <= 4);
    kani::assume(n >= 1 && n <= 4);
    kani::assume(p >= 1 && p <= 4);

    let lhs = [1usize, n as usize, 1usize];
    let rhs = [m as usize, n as usize, p as usize];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(result.is_ok(), "[1,N,1] + [M,N,P] must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 3, "output rank must be 3");
    assert_eq!(out[0], m as usize, "dim 0 must be M");
    assert_eq!(out[1], n as usize, "dim 1 must be N");
    assert_eq!(out[2], p as usize, "dim 2 must be P");
}

// ---------------------------------------------------------------------------
// 8. Rank mismatch broadcast: [M] + [N,M] -> [N,M]
// ---------------------------------------------------------------------------

/// Prove: rank-mismatch broadcast. [M] + [N,M] must produce [N,M].
/// The 1D shape is right-aligned to [_, M]. This is the standard bias-add
/// pattern in linear layers.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_rank_mismatch_1d_2d() {
    let n: u8 = kani::any();
    let m: u8 = kani::any();

    kani::assume(n >= 1 && n <= 4);
    kani::assume(m >= 1 && m <= 4);

    let lhs = [m as usize];
    let rhs = [n as usize, m as usize];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(result.is_ok(), "[M] + [N,M] must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 2, "output rank must be max(1,2) = 2");
    assert_eq!(out[0], n as usize, "dim 0 must be N from rhs");
    assert_eq!(out[1], m as usize, "dim 1 must be M");
}

// ---------------------------------------------------------------------------
// 9. Division shape safety: broadcast_output_shape succeeds for compatible
//    shapes regardless of values (division by zero is a value error, not shape)
// ---------------------------------------------------------------------------

/// Prove: broadcast_output_shape is purely a shape computation and always
/// succeeds for compatible shapes. Division by zero is caught at the value
/// level (check_div_result_finite), not at the shape level.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_div_shape_always_succeeds_for_compatible() {
    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();

    kani::assume(a0 >= 1 && a0 <= 4);
    kani::assume(a1 >= 1 && a1 <= 4);
    kani::assume(b0 >= 1 && b0 <= 4);
    kani::assume(b1 >= 1 && b1 <= 4);

    // Ensure compatible: each dim is either equal or one is 1
    kani::assume(a0 == b0 || a0 == 1 || b0 == 1);
    kani::assume(a1 == b1 || a1 == 1 || b1 == 1);

    let lhs = [a0 as usize, a1 as usize];
    let rhs = [b0 as usize, b1 as usize];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(
        result.is_ok(),
        "compatible shapes must always produce a valid broadcast shape"
    );
}

// ---------------------------------------------------------------------------
// 10. Output shape correctness: output dims are max of input dims
// ---------------------------------------------------------------------------

/// Prove: for compatible 2D shapes, each output dimension is exactly
/// max(lhs_dim, rhs_dim). This is the fundamental broadcast rule.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_output_dim_is_max_2d() {
    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();

    kani::assume(a0 >= 1 && a0 <= 4);
    kani::assume(a1 >= 1 && a1 <= 4);
    kani::assume(b0 >= 1 && b0 <= 4);
    kani::assume(b1 >= 1 && b1 <= 4);

    let lhs = [a0 as usize, a1 as usize];
    let rhs = [b0 as usize, b1 as usize];

    if let Ok(out) = broadcast_output_shape(&lhs, &rhs) {
        assert_eq!(
            out[0],
            (a0 as usize).max(b0 as usize),
            "dim 0 must be max(lhs, rhs)"
        );
        assert_eq!(
            out[1],
            (a1 as usize).max(b1 as usize),
            "dim 1 must be max(lhs, rhs)"
        );
    }
}

// ---------------------------------------------------------------------------
// 11. Commutativity of add broadcast shapes
// ---------------------------------------------------------------------------

/// Prove: broadcast shape computation is commutative for 2D shapes.
/// broadcast_output_shape(a, b) == broadcast_output_shape(b, a).
/// This ensures a + b and b + a have the same output shape.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_add_commutativity_2d() {
    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();

    kani::assume(a0 >= 1 && a0 <= 4);
    kani::assume(a1 >= 1 && a1 <= 4);
    kani::assume(b0 >= 1 && b0 <= 4);
    kani::assume(b1 >= 1 && b1 <= 4);

    let lhs = [a0 as usize, a1 as usize];
    let rhs = [b0 as usize, b1 as usize];

    let forward = broadcast_output_shape(&lhs, &rhs);
    let reverse = broadcast_output_shape(&rhs, &lhs);

    match (forward, reverse) {
        (Ok(f), Ok(r)) => {
            assert_eq!(f.len(), r.len(), "commutative rank");
            assert_eq!(f[0], r[0], "commutative dim 0");
            assert_eq!(f[1], r[1], "commutative dim 1");
        }
        (Err(_), Err(_)) => {
            // Both fail -- consistent
        }
        _ => {
            panic!("add broadcast commutativity violated");
        }
    }
}

// ---------------------------------------------------------------------------
// 12. Commutativity of mul broadcast shapes
// ---------------------------------------------------------------------------

/// Prove: broadcast shape computation is commutative for 3D shapes.
/// This verifies that the shape function behaves identically for mul.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_mul_commutativity_3d() {
    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let a2: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();
    let b2: u8 = kani::any();

    kani::assume(a0 >= 1 && a0 <= 4);
    kani::assume(a1 >= 1 && a1 <= 4);
    kani::assume(a2 >= 1 && a2 <= 4);
    kani::assume(b0 >= 1 && b0 <= 4);
    kani::assume(b1 >= 1 && b1 <= 4);
    kani::assume(b2 >= 1 && b2 <= 4);

    let lhs = [a0 as usize, a1 as usize, a2 as usize];
    let rhs = [b0 as usize, b1 as usize, b2 as usize];

    let forward = broadcast_output_shape(&lhs, &rhs);
    let reverse = broadcast_output_shape(&rhs, &lhs);

    match (forward, reverse) {
        (Ok(f), Ok(r)) => {
            assert_eq!(f.len(), r.len(), "commutative rank");
            assert_eq!(f[0], r[0], "commutative dim 0");
            assert_eq!(f[1], r[1], "commutative dim 1");
            assert_eq!(f[2], r[2], "commutative dim 2");
        }
        (Err(_), Err(_)) => {}
        _ => {
            panic!("mul broadcast commutativity violated");
        }
    }
}

// ---------------------------------------------------------------------------
// 13. Incompatible trailing dims rejected for sub
// ---------------------------------------------------------------------------

/// Prove: shapes with incompatible trailing dimensions (both > 1 and
/// different) are rejected. This verifies the error path for sub.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_sub_rejects_incompatible_trailing() {
    let n: u8 = kani::any();
    let a: u8 = kani::any();
    let b: u8 = kani::any();

    kani::assume(n >= 1 && n <= 4);
    kani::assume(a >= 2 && a <= 4);
    kani::assume(b >= 2 && b <= 4);
    kani::assume(a != b);

    let lhs = [n as usize, a as usize];
    let rhs = [n as usize, b as usize];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(
        result.is_err(),
        "incompatible trailing dims (both > 1 and different) must fail"
    );
}

// ---------------------------------------------------------------------------
// 14. Incompatible trailing dims rejected for div
// ---------------------------------------------------------------------------

/// Prove: shapes with incompatible leading dimensions (both > 1 and
/// different) are rejected. Even when trailing dims match, a single
/// incompatible dim causes rejection.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_div_rejects_incompatible_leading() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    let m: u8 = kani::any();

    kani::assume(a >= 2 && a <= 4);
    kani::assume(b >= 2 && b <= 4);
    kani::assume(m >= 1 && m <= 4);
    kani::assume(a != b);

    let lhs = [a as usize, m as usize];
    let rhs = [b as usize, m as usize];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(
        result.is_err(),
        "incompatible leading dims (both > 1 and different) must fail"
    );
}

// ---------------------------------------------------------------------------
// 15. Broadcast output rank is max of input ranks
// ---------------------------------------------------------------------------

/// Prove: the output rank from broadcasting is always max(lhs_rank, rhs_rank).
/// Tests 1D vs 3D broadcasting: output must always be rank 3.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_output_rank_is_max_1d_3d() {
    let a: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();
    let b2: u8 = kani::any();

    kani::assume(a >= 1 && a <= 4);
    kani::assume(b0 >= 1 && b0 <= 4);
    kani::assume(b1 >= 1 && b1 <= 4);
    kani::assume(b2 >= 1 && b2 <= 4);

    // Ensure compatible: a must be 1 or equal to b2 (right-aligned)
    kani::assume(a == 1 || a == b2);

    let lhs = [a as usize];
    let rhs = [b0 as usize, b1 as usize, b2 as usize];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(result.is_ok(), "compatible 1D vs 3D must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 3, "output rank must be max(1, 3) = 3");
}

// ---------------------------------------------------------------------------
// 16. Self-broadcast is identity
// ---------------------------------------------------------------------------

/// Prove: broadcasting a shape with itself always succeeds and returns
/// the same shape. This is the identity property of broadcasting.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_self_is_identity_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(d0 >= 1 && d0 <= 4);
    kani::assume(d1 >= 1 && d1 <= 4);
    kani::assume(d2 >= 1 && d2 <= 4);

    let shape = [d0 as usize, d1 as usize, d2 as usize];

    let result = broadcast_output_shape(&shape, &shape);
    assert!(result.is_ok(), "self-broadcast must always succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 3, "rank preserved");
    assert_eq!(out[0], d0 as usize, "dim 0 is identity");
    assert_eq!(out[1], d1 as usize, "dim 1 is identity");
    assert_eq!(out[2], d2 as usize, "dim 2 is identity");
}

// ---------------------------------------------------------------------------
// 17. All-ones 2D with arbitrary 2D broadcast
// ---------------------------------------------------------------------------

/// Prove: [1,1] broadcast with any [N,M] produces [N,M].
/// All-ones is the broadcast identity element for same-rank shapes.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_all_ones_2d_identity() {
    let n: u8 = kani::any();
    let m: u8 = kani::any();

    kani::assume(n >= 1 && n <= 4);
    kani::assume(m >= 1 && m <= 4);

    let ones = [1usize, 1usize];
    let shape = [n as usize, m as usize];

    let result = broadcast_output_shape(&ones, &shape);
    assert!(result.is_ok(), "[1,1] + [N,M] must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 2, "rank must be 2");
    assert_eq!(out[0], n as usize, "dim 0 must be N");
    assert_eq!(out[1], m as usize, "dim 1 must be M");
}

// ---------------------------------------------------------------------------
// 18. Rank 0 + rank 0 broadcast produces rank 0
// ---------------------------------------------------------------------------

/// Prove: broadcasting two rank-0 (scalar) shapes produces a rank-0 shape.
/// This covers scalar + scalar operations.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_rank0_rank0_produces_rank0() {
    let lhs: [usize; 0] = [];
    let rhs: [usize; 0] = [];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(result.is_ok(), "rank-0 + rank-0 must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 0, "output must be rank 0");
}

// ---------------------------------------------------------------------------
// 19. Partial broadcast: [1,M] + [N,1] -> [N,M]
// ---------------------------------------------------------------------------

/// Prove: partial broadcast where BOTH operands have size-1 dims in
/// different positions. [1,M] + [N,1] must produce [N,M].
/// This is the outer-product-like broadcast pattern.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_partial_1m_n1_produces_nm() {
    let n: u8 = kani::any();
    let m: u8 = kani::any();

    kani::assume(n >= 1 && n <= 4);
    kani::assume(m >= 1 && m <= 4);

    let lhs = [1usize, m as usize];
    let rhs = [n as usize, 1usize];

    let result = broadcast_output_shape(&lhs, &rhs);
    assert!(result.is_ok(), "[1,M] + [N,1] must succeed");
    let out = result.unwrap();
    assert_eq!(out.len(), 2, "output rank must be 2");
    assert_eq!(out[0], n as usize, "dim 0 must be N");
    assert_eq!(out[1], m as usize, "dim 1 must be M");
}

// ---------------------------------------------------------------------------
// 20. Broadcast output element count >= each input element count
// ---------------------------------------------------------------------------

/// Prove: the element count of the broadcast output shape is always
/// greater than or equal to the element count of each input shape.
/// Broadcasting can only expand, never shrink.
#[kani::unwind(4)]
#[kani::proof]
fn broadcast_output_numel_geq_inputs() {
    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();

    kani::assume(a0 >= 1 && a0 <= 4);
    kani::assume(a1 >= 1 && a1 <= 4);
    kani::assume(b0 >= 1 && b0 <= 4);
    kani::assume(b1 >= 1 && b1 <= 4);

    let lhs = [a0 as usize, a1 as usize];
    let rhs = [b0 as usize, b1 as usize];

    if let Ok(out) = broadcast_output_shape(&lhs, &rhs) {
        let lhs_numel = (a0 as u64) * (a1 as u64);
        let rhs_numel = (b0 as u64) * (b1 as u64);
        let out_numel = (out[0] as u64) * (out[1] as u64);

        assert!(
            out_numel >= lhs_numel,
            "broadcast output numel must be >= lhs numel"
        );
        assert!(
            out_numel >= rhs_numel,
            "broadcast output numel must be >= rhs numel"
        );
    }
}
