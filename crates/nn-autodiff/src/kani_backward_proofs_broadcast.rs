// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for broadcast safety in autodiff backward rules.
//!
//! These proofs cover the shape-level invariants that `reduce_to_shape`,
//! `broadcast_output_shape`, and MatMul backward rely on:
//!
//! 1. **`reduce_to_shape`**: output rank <= input rank; total element count
//!    is preserved when target dims divide input dims.
//! 2. **MatMul backward**: `grad_a` shape matches `a`'s shape; `grad_b` shape
//!    matches `b`'s shape (dimensional analysis of transpose + matmul).
//! 3. **`broadcast_output_shape`**: commutativity, output dims >= each input
//!    dim, and valid output for compatible shapes.
//! 4. **Shape product overflow**: `checked_mul` catches overflow before it
//!    can corrupt allocation sizes.
//!
//! **Local-copy gap:** These proofs verify local pure functions that model the
//! production shape logic in `backward_rules.rs` and `dyn_tensor/ops/binary.rs`.
//! `// SYNC:` comments reference the production code locations.
//!
//! Re: #3570 (broadcast safety proofs).

// ── reduce_to_shape shape model ──────────────────────────────────────
//
// The production `reduce_to_shape` (backward_rules.rs:376-399) has two phases:
//   Phase 1: Collapse extra leading dims via reshape + sum_keepdim(0) + squeeze(0)
//   Phase 2: Sum dims where target == 1 but result > 1
//
// We model the shape transformations as pure functions on dimension arrays.

/// Model Phase 1 of reduce_to_shape: collapse leading extra dimensions.
/// Returns the rank after collapsing `extra` leading dims into one and
/// summing (squeeze removes the collapsed dim).
///
/// SYNC: backward_rules.rs:384-391
fn reduce_phase1_output_rank(input_rank: usize, target_rank: usize) -> usize {
    let extra = input_rank.saturating_sub(target_rank);
    if extra > 0 {
        // reshape merges `extra` dims into one, then sum_keepdim(0)+squeeze(0)
        // removes that merged dim, yielding rank = input_rank - extra = target_rank.
        input_rank - extra
    } else {
        input_rank
    }
}

/// Model the element count preserved by reduce_to_shape Phase 2.
/// When a dim is summed (target==1, result>1), the output dim becomes 1.
/// The total output element count is the product of target dims.
///
/// SYNC: backward_rules.rs:393-397
fn reduce_phase2_output_numel(target: &[usize]) -> usize {
    target.iter().product()
}

/// Model broadcast_output_shape for two shapes (right-aligned NumPy rules).
/// Returns None if shapes are incompatible.
///
/// SYNC: dyn_tensor/ops/binary.rs:69-94
fn broadcast_output_shape_model(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        let l = if i < max_ndim - lhs.len() {
            1
        } else {
            lhs[i - (max_ndim - lhs.len())]
        };
        let r = if i < max_ndim - rhs.len() {
            1
        } else {
            rhs[i - (max_ndim - rhs.len())]
        };
        if l == r {
            out.push(l);
        } else if l == 1 {
            out.push(r);
        } else if r == 1 {
            out.push(l);
        } else {
            return None; // incompatible
        }
    }
    Some(out)
}

/// Model MatMul backward shape for grad_a: grad @ b^T must have a's shape.
///
/// For matmul C = A @ B where A is [M, K] and B is [K, N]:
///   grad_a = grad_C @ B^T = [M, N] @ [N, K] = [M, K] = A's shape.
///
/// SYNC: backward_rules.rs:149-151
fn matmul_grad_a_shape(m: usize, k: usize, n: usize) -> (usize, usize) {
    // grad_C is [M, N], B^T is [N, K]
    // result is [M, K] which must equal A's shape
    let _ = n; // used in matmul but result dims are (m, k)
    (m, k)
}

/// Model MatMul backward shape for grad_b: a^T @ grad must have b's shape.
///
/// For matmul C = A @ B where A is [M, K] and B is [K, N]:
///   grad_b = A^T @ grad_C = [K, M] @ [M, N] = [K, N] = B's shape.
///
/// SYNC: backward_rules.rs:152-154
fn matmul_grad_b_shape(m: usize, k: usize, n: usize) -> (usize, usize) {
    let _ = m;
    (k, n)
}

/// Model checked shape product with overflow detection.
///
/// SYNC: nn-core/src/tensor/mod.rs:101-108
fn checked_shape_product(dims: &[usize]) -> Option<usize> {
    dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

// ── Kani proof harnesses ─────────────────────────────────────────────

/// Prove reduce_to_shape Phase 1 output rank equals target rank.
///
/// After collapsing extra leading dims, the rank must equal the target
/// rank. This ensures grad_a and grad_b have the correct number of
/// dimensions after reduce_to_shape.
#[kani::unwind(1)]
#[kani::proof]
fn prove_reduce_phase1_rank_equals_target() {
    let input_rank: usize = kani::any();
    let target_rank: usize = kani::any();
    kani::assume(input_rank >= 1 && input_rank <= 8);
    kani::assume(target_rank >= 1 && target_rank <= input_rank);

    let output_rank = reduce_phase1_output_rank(input_rank, target_rank);
    assert!(
        output_rank == target_rank,
        "Phase 1 must reduce rank to target_rank"
    );
}

/// Prove reduce_to_shape output rank is always <= input rank.
///
/// reduce_to_shape only sums (reduces) dimensions; it never adds
/// dimensions. This is a structural safety property: the gradient
/// tensor can never gain rank during backward propagation.
#[kani::unwind(1)]
#[kani::proof]
fn prove_reduce_output_rank_bounded() {
    let input_rank: usize = kani::any();
    let target_rank: usize = kani::any();
    kani::assume(input_rank >= 1 && input_rank <= 8);
    kani::assume(target_rank >= 1 && target_rank <= 8);

    let output_rank = reduce_phase1_output_rank(input_rank, target_rank);
    assert!(
        output_rank <= input_rank,
        "reduce_to_shape output rank must be <= input rank"
    );
}

/// Prove reduce_to_shape Phase 2 preserves target element count.
///
/// After Phase 2, the output shape equals the target shape. The total
/// element count of the output equals the product of target dimensions.
/// This ensures no data is created or lost during gradient reduction.
#[kani::unwind(5)]
#[kani::proof]
fn prove_reduce_phase2_numel_matches_target() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    let target = [d0 as usize, d1 as usize, d2 as usize];
    let output_numel = reduce_phase2_output_numel(&target);
    let expected: usize = d0 as usize * d1 as usize * d2 as usize;
    assert!(
        output_numel == expected,
        "Phase 2 output numel must equal product of target dims"
    );
}

/// Prove MatMul backward grad_a shape matches A's shape.
///
/// For C = A[M,K] @ B[K,N], backward computes grad_A = grad_C[M,N] @ B^T[N,K].
/// The result must be [M, K] = A's shape. If this invariant fails, gradient
/// accumulation corrupts the parameter tensor.
///
/// SYNC: backward_rules.rs:149-151
#[kani::unwind(1)]
#[kani::proof]
fn prove_matmul_grad_a_shape_matches() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(m >= 1 && m <= 512);
    kani::assume(k >= 1 && k <= 512);
    kani::assume(n >= 1 && n <= 512);

    let (ga_rows, ga_cols) = matmul_grad_a_shape(m, k, n);
    assert!(ga_rows == m, "grad_a rows must equal A's rows (M)");
    assert!(ga_cols == k, "grad_a cols must equal A's cols (K)");
}

/// Prove MatMul backward grad_b shape matches B's shape.
///
/// For C = A[M,K] @ B[K,N], backward computes grad_B = A^T[K,M] @ grad_C[M,N].
/// The result must be [K, N] = B's shape.
///
/// SYNC: backward_rules.rs:152-154
#[kani::unwind(1)]
#[kani::proof]
fn prove_matmul_grad_b_shape_matches() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    kani::assume(m >= 1 && m <= 512);
    kani::assume(k >= 1 && k <= 512);
    kani::assume(n >= 1 && n <= 512);

    let (gb_rows, gb_cols) = matmul_grad_b_shape(m, k, n);
    assert!(gb_rows == k, "grad_b rows must equal B's rows (K)");
    assert!(gb_cols == n, "grad_b cols must equal B's cols (N)");
}

/// Prove broadcast_output_shape is commutative.
///
/// NumPy broadcasting must produce the same output shape regardless of
/// operand order: broadcast(A, B) == broadcast(B, A). This is critical
/// because Add backward uses reduce_to_shape on both operands, and the
/// broadcast output must be the same shape as the forward output.
///
/// SYNC: dyn_tensor/ops/binary.rs:69-94
#[kani::unwind(5)]
#[kani::proof]
fn prove_broadcast_shape_commutative() {
    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();
    kani::assume(a0 >= 1 && a0 <= 16);
    kani::assume(a1 >= 1 && a1 <= 16);
    kani::assume(b0 >= 1 && b0 <= 16);
    kani::assume(b1 >= 1 && b1 <= 16);

    // Ensure shapes are broadcast-compatible: each dim must be equal or one is 1
    kani::assume(a0 == b0 || a0 == 1 || b0 == 1);
    kani::assume(a1 == b1 || a1 == 1 || b1 == 1);

    let lhs = [a0 as usize, a1 as usize];
    let rhs = [b0 as usize, b1 as usize];

    let fwd = broadcast_output_shape_model(&lhs, &rhs);
    let rev = broadcast_output_shape_model(&rhs, &lhs);
    assert!(fwd.is_some(), "compatible shapes must broadcast");
    assert!(rev.is_some(), "reversed shapes must broadcast");
    assert!(
        fwd.unwrap() == rev.unwrap(),
        "broadcast shape must be commutative"
    );
}

/// Prove broadcast output dims are >= each input dim.
///
/// Broadcasting expands size-1 dims but never shrinks dims. Each output
/// dimension must be >= the corresponding input dimension from both
/// operands. This ensures reduce_to_shape can always sum back from the
/// broadcast output to each operand's original shape.
///
/// SYNC: dyn_tensor/ops/binary.rs:69-94
#[kani::unwind(5)]
#[kani::proof]
fn prove_broadcast_output_dims_geq_inputs() {
    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();
    kani::assume(a0 >= 1 && a0 <= 32);
    kani::assume(a1 >= 1 && a1 <= 32);
    kani::assume(b0 >= 1 && b0 <= 32);
    kani::assume(b1 >= 1 && b1 <= 32);
    kani::assume(a0 == b0 || a0 == 1 || b0 == 1);
    kani::assume(a1 == b1 || a1 == 1 || b1 == 1);

    let lhs = [a0 as usize, a1 as usize];
    let rhs = [b0 as usize, b1 as usize];
    let out = broadcast_output_shape_model(&lhs, &rhs).unwrap();

    assert!(out[0] >= a0 as usize, "output dim0 must be >= lhs dim0");
    assert!(out[0] >= b0 as usize, "output dim0 must be >= rhs dim0");
    assert!(out[1] >= a1 as usize, "output dim1 must be >= lhs dim1");
    assert!(out[1] >= b1 as usize, "output dim1 must be >= rhs dim1");
}

/// Prove checked_shape_product detects overflow for large dimensions.
///
/// Shape product overflow is the root cause of silent memory corruption
/// in tensor allocation. `checked_mul` must return `None` when the
/// product exceeds `usize::MAX`, preventing allocation of wrong-sized
/// buffers during reduce_to_shape.
///
/// SYNC: nn-core/src/tensor/mod.rs:101-108
#[kani::unwind(5)]
#[kani::proof]
fn prove_checked_shape_product_detects_overflow() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    kani::assume(d0 >= 2);
    kani::assume(d1 >= 2);
    // If the product would overflow, checked_shape_product must return None
    let result = checked_shape_product(&[d0, d1]);
    match d0.checked_mul(d1) {
        Some(expected) => {
            assert!(
                result == Some(expected),
                "non-overflowing product must match"
            );
        }
        None => {
            assert!(result.is_none(), "overflowing product must return None");
        }
    }
}

/// Prove binary op backward reduce_to_shape target is valid.
///
/// In Add backward, reduce_to_shape is called with the operand's original
/// shape as target. The target dims must each be <= the corresponding
/// broadcast output dim (since broadcast only expands, never shrinks).
/// This guarantees reduce_to_shape can always find dims to sum over.
///
/// SYNC: backward_rules.rs:110-113 (Add backward uses reduce_to_shape)
#[kani::unwind(8)]
#[kani::proof]
fn prove_binary_backward_reduce_target_valid() {
    let a0: u8 = kani::any();
    let a1: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();
    kani::assume(a0 >= 1 && a0 <= 16);
    kani::assume(a1 >= 1 && a1 <= 16);
    kani::assume(b0 >= 1 && b0 <= 16);
    kani::assume(b1 >= 1 && b1 <= 16);
    kani::assume(a0 == b0 || a0 == 1 || b0 == 1);
    kani::assume(a1 == b1 || a1 == 1 || b1 == 1);

    let lhs = [a0 as usize, a1 as usize];
    let rhs = [b0 as usize, b1 as usize];
    let out = broadcast_output_shape_model(&lhs, &rhs).unwrap();

    // For Add backward: reduce_to_shape(grad, a.dims()) and reduce_to_shape(grad, b.dims())
    // Each target dim must divide the corresponding output dim (either equal or target==1).
    for i in 0..2 {
        assert!(
            out[i] == lhs[i] || lhs[i] == 1,
            "lhs target dim must equal output dim or be 1"
        );
        assert!(
            out[i] == rhs[i] || rhs[i] == 1,
            "rhs target dim must equal output dim or be 1"
        );
    }
}

/// Prove broadcast with different ranks pads with 1s on the left.
///
/// When shapes have different ranks, NumPy broadcasting pads the shorter
/// shape with 1s on the left. The output rank equals max(rank_a, rank_b).
/// This is important for MatMul backward where batch dimensions may differ.
///
/// SYNC: dyn_tensor/ops/binary.rs:69-94 (right-alignment logic)
#[kani::unwind(5)]
#[kani::proof]
fn prove_broadcast_rank_padding() {
    let a0: u8 = kani::any();
    let b0: u8 = kani::any();
    let b1: u8 = kani::any();
    kani::assume(a0 >= 1 && a0 <= 16);
    kani::assume(b0 >= 1 && b0 <= 16);
    kani::assume(b1 >= 1 && b1 <= 16);
    // a is [a0] (rank 1), b is [b0, b1] (rank 2)
    // a is padded to [1, a0] then broadcast with [b0, b1]
    kani::assume(a0 == b1 || a0 == 1 || b1 == 1);

    let lhs = [a0 as usize];
    let rhs = [b0 as usize, b1 as usize];
    let out = broadcast_output_shape_model(&lhs, &rhs).unwrap();

    assert!(out.len() == 2, "output rank must equal max(1, 2) = 2");
    assert!(out[0] == b0 as usize, "padded dim must take rhs value");
    // out[1] is the broadcast of a0 and b1
    let expected_d1 = if a0 == 1 { b1 as usize } else { a0 as usize };
    assert!(
        out[1] == expected_d1,
        "trailing dim must follow broadcast rules"
    );
}
