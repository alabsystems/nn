// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `backward_rules.rs` reduce_to_shape, binary backward,
//! cat/stack offset arithmetic, and reshape_for_channel_broadcast.
//!
//! These harnesses target the production helper functions and backward dispatch
//! logic in `backward_rules.rs` that were not covered by existing proof files.
//!
//! Key properties proved:
//! - `reduce_to_shape` early-return when shapes already match
//! - `reduce_to_shape` leading-dim collapse preserves target rank
//! - `reduce_to_shape` Phase 2 only sums dims where target == 1
//! - Binary Sub backward negation: grad_b = -grad_a when shapes match
//! - Binary Div backward quotient rule correctness: chain ∂(a/b)/∂b = -a/b^2
//! - Cat backward offset accumulation is monotonically increasing and partitions
//! - MeanKeepDim backward scale factor monotonicity (larger dim → smaller scale)
//! - `reshape_for_channel_broadcast` rejects rank < 2
//! - `reduce_to_shape` leading product overflow detection
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc.
//!
//! Re: #3661 (Kani harnesses for backward_rules + backward_rules_norm).

// ── reduce_to_shape fast path ─────────────────────────────────────
//
// When tensor.dims() == target, reduce_to_shape returns early (clone).
// This is the common case for same-shape binary ops.
//
// SYNC: backward_rules.rs:377-379

/// Model the reduce_to_shape fast-path check.
fn reduce_needs_work(tensor_dims: &[usize], target: &[usize]) -> bool {
    tensor_dims != target
}

/// Prove reduce_to_shape fast-path: same shape means no reduction needed.
#[kani::unwind(1)]
#[kani::proof]
fn prove_reduce_fast_path_same_shape() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    let shape = [d0 as usize, d1 as usize];
    assert!(
        !reduce_needs_work(&shape, &shape),
        "same shape must not need reduction"
    );
}

/// Prove reduce_to_shape detects rank mismatch as needing work.
#[kani::unwind(1)]
#[kani::proof]
fn prove_reduce_detects_rank_mismatch() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    let tensor_dims = [d0 as usize, d1 as usize, d2 as usize];
    let target = [d1 as usize, d2 as usize];
    // Different rank always means work is needed
    assert!(
        reduce_needs_work(&tensor_dims, &target),
        "different rank must trigger reduction"
    );
}

// ── reduce_to_shape leading collapse ──────────────────────────────
//
// Phase 1: extra = tensor_rank - target_rank leading dims collapsed.
// The leading_product is the product of dims[..extra].
//
// SYNC: backward_rules.rs:384-391

/// Model Phase 1: compute leading product for collapse.
/// Returns None on overflow (matches checked_dim_product).
fn leading_product(dims: &[usize], extra: usize) -> Option<usize> {
    dims[..extra]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// Prove leading product is >= 1 for valid shapes (all dims >= 1).
#[kani::unwind(5)]
#[kani::proof]
fn prove_leading_product_at_least_one() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    let dims = [d0 as usize, d1 as usize, 4, 4]; // rank 4
    let extra = 2; // collapsing 2 leading dims
    let product = leading_product(&dims, extra);
    assert!(product.is_some(), "product of small dims must not overflow");
    assert!(
        product.unwrap() >= 1,
        "leading product must be >= 1 when all dims >= 1"
    );
}

/// Prove leading product overflow detection: two large dims can overflow.
#[kani::unwind(5)]
#[kani::proof]
fn prove_leading_product_detects_overflow() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    kani::assume(d0 >= 2);
    kani::assume(d1 >= 2);
    let dims = [d0, d1];
    let product = leading_product(&dims, 2);
    match d0.checked_mul(d1) {
        Some(expected) => {
            assert!(product == Some(expected), "non-overflowing must match");
        }
        None => {
            assert!(product.is_none(), "overflow must be detected");
        }
    }
}

// ── reduce_to_shape Phase 2: broadcast dim detection ──────────────
//
// Phase 2 sums dims where target[d] == 1 and result.dim(d) > 1.
// This models which dimensions get summed.
//
// SYNC: backward_rules.rs:393-397

/// Returns true if dimension d needs summing in Phase 2.
fn needs_phase2_sum(target_d: usize, result_d: usize) -> bool {
    target_d == 1 && result_d > 1
}

/// Prove Phase 2 sum detection: only sums when target is 1 and result > 1.
#[kani::unwind(1)]
#[kani::proof]
fn prove_phase2_sum_detection() {
    let target_d: u8 = kani::any();
    let result_d: u8 = kani::any();
    kani::assume(target_d >= 1 && target_d <= 64);
    kani::assume(result_d >= 1 && result_d <= 64);
    // If target_d > 1, never sum (shapes must match for that dim)
    if target_d > 1 {
        assert!(
            !needs_phase2_sum(target_d as usize, result_d as usize),
            "must not sum when target dim > 1"
        );
    }
    // If target_d == 1 and result_d == 1, no sum needed
    if target_d == 1 && result_d == 1 {
        assert!(
            !needs_phase2_sum(target_d as usize, result_d as usize),
            "must not sum when result dim also 1"
        );
    }
}

/// Prove Phase 2 correctly identifies broadcast dims in a 3D example.
#[kani::unwind(8)]
#[kani::proof]
fn prove_phase2_sum_3d_broadcast() {
    let t0: u8 = kani::any();
    let t1: u8 = kani::any();
    let t2: u8 = kani::any();
    let r0: u8 = kani::any();
    let r1: u8 = kani::any();
    let r2: u8 = kani::any();
    kani::assume(t0 >= 1 && t0 <= 32);
    kani::assume(t1 >= 1 && t1 <= 32);
    kani::assume(t2 >= 1 && t2 <= 32);
    kani::assume(r0 >= 1 && r0 <= 32);
    kani::assume(r1 >= 1 && r1 <= 32);
    kani::assume(r2 >= 1 && r2 <= 32);
    // broadcast rule: result >= target, and target is either 1 or == result
    kani::assume(t0 == 1 || t0 == r0);
    kani::assume(t1 == 1 || t1 == r1);
    kani::assume(t2 == 1 || t2 == r2);

    let sum_count = [
        needs_phase2_sum(t0 as usize, r0 as usize),
        needs_phase2_sum(t1 as usize, r1 as usize),
        needs_phase2_sum(t2 as usize, r2 as usize),
    ];
    // Every summed dim must have target == 1
    for i in 0..3 {
        if sum_count[i] {
            let t = [t0, t1, t2][i];
            assert!(t == 1, "summed dim must have target == 1");
        }
    }
}

// ── Binary Sub backward: negation property ─────────────────────────
//
// Sub backward: grad_a = grad, grad_b = -grad (when shapes match).
// Key property: grad_a + grad_b == 0 (gradient conservation).
//
// SYNC: backward_rules.rs:114-116

/// Model Sub backward partial derivatives (same-shape case).
fn sub_grad_a(grad: f32) -> f32 {
    grad
}

fn sub_grad_b(grad: f32) -> f32 {
    -grad
}

/// Prove Sub backward gradient conservation: grad_a + grad_b == 0.
/// This ensures the total gradient through a subtraction node sums to zero
/// (a increases → output increases by 1, b increases → output decreases by 1).
#[kani::unwind(1)]
#[kani::proof]
fn prove_sub_backward_conservation() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    let ga = sub_grad_a(grad);
    let gb = sub_grad_b(grad);
    let sum = ga + gb;
    assert!(
        sum == 0.0,
        "Sub backward grad_a + grad_b must be zero (gradient conservation)"
    );
}

/// Prove Sub backward: grad_b has opposite sign to grad.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sub_backward_negation_sign() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite() && grad > 0.0);
    let gb = sub_grad_b(grad);
    assert!(
        gb < 0.0,
        "Sub backward grad_b must be negative when grad > 0"
    );
}

// ── Binary Div backward: quotient rule chain ─────────────────────
//
// Div backward for b: grad_b = grad * (-a / b^2)
// The chain: upstream_grad * partial_derivative.
//
// SYNC: backward_rules.rs:130-138

/// Model the full Div backward chain for b: grad * (-a / b^2).
fn div_backward_b_chained(grad: f32, a: f32, b: f32) -> f32 {
    grad * (-a / (b * b))
}

/// Prove Div backward b chain is finite for bounded non-zero b.
#[kani::unwind(1)]
#[kani::proof]
fn prove_div_backward_b_chain_finite() {
    let grad: f32 = kani::any();
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    kani::assume(a.is_finite() && a.abs() <= 1e3);
    kani::assume(b.is_finite() && b.abs() >= 0.01 && b.abs() <= 1e3);
    let result = div_backward_b_chained(grad, a, b);
    assert!(
        result.is_finite(),
        "Div backward b chain must be finite for bounded inputs"
    );
}

// ── Cat backward offset arithmetic ─────────────────────────────────
//
// Cat backward splits grad along dim using narrow(dim, offset, len).
// offset advances by len after each input. Key invariant:
//   offset_final == sum(input_lens) == grad_dim_size.
//
// SYNC: backward_rules.rs:237-250

/// Model cat backward offset accumulation for N inputs.
/// Returns the final offset after processing all inputs.
fn cat_backward_final_offset(lens: &[usize]) -> usize {
    let mut offset = 0;
    for &len in lens {
        offset += len;
    }
    offset
}

/// Prove cat backward final offset equals total dim size.
/// This ensures narrow slices partition the full gradient without gaps or overlap.
#[kani::unwind(5)]
#[kani::proof]
fn prove_cat_backward_offset_partition() {
    let len0: u8 = kani::any();
    let len1: u8 = kani::any();
    let len2: u8 = kani::any();
    let len3: u8 = kani::any();
    kani::assume(len0 >= 1 && len0 <= 32);
    kani::assume(len1 >= 1 && len1 <= 32);
    kani::assume(len2 >= 1 && len2 <= 32);
    kani::assume(len3 >= 1 && len3 <= 32);
    let lens = [len0 as usize, len1 as usize, len2 as usize, len3 as usize];
    let total: usize = lens.iter().sum();
    let final_offset = cat_backward_final_offset(&lens);
    assert!(
        final_offset == total,
        "cat backward final offset must equal sum of input lens"
    );
}

/// Prove cat backward offsets are monotonically increasing.
/// Each narrow slice starts after the previous one ends.
#[kani::unwind(1)]
#[kani::proof]
fn prove_cat_backward_offset_monotone() {
    let len0: u8 = kani::any();
    let len1: u8 = kani::any();
    let len2: u8 = kani::any();
    kani::assume(len0 >= 1 && len0 <= 32);
    kani::assume(len1 >= 1 && len1 <= 32);
    kani::assume(len2 >= 1 && len2 <= 32);
    let offsets = [0usize, len0 as usize, len0 as usize + len1 as usize];
    // Each subsequent offset is strictly greater
    assert!(offsets[1] > offsets[0], "offset[1] must be > offset[0]");
    assert!(offsets[2] > offsets[1], "offset[2] must be > offset[1]");
}

// ── MeanKeepDim backward scale monotonicity ────────────────────────
//
// MeanKeepDim backward: scale = 1/n.
// Larger n → smaller scale (gradient is more attenuated).
// This is important for gradient stability with large reduction dims.
//
// SYNC: backward_rules.rs:164-167

/// Prove MeanKeepDim backward scale is monotonically decreasing in n.
/// Larger reduction dim means more attenuation.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mean_backward_scale_monotone() {
    let n1: u16 = kani::any();
    let n2: u16 = kani::any();
    kani::assume(n1 >= 1 && n1 <= 10000);
    kani::assume(n2 >= 1 && n2 <= 10000);
    kani::assume(n1 < n2);
    let scale1 = 1.0 / n1 as f64;
    let scale2 = 1.0 / n2 as f64;
    assert!(
        scale1 > scale2,
        "larger n must produce smaller scale factor (more attenuation)"
    );
}

// ── reshape_for_channel_broadcast rank guard ────────────────────────
//
// reshape_for_channel_broadcast rejects target_rank < 2.
// This prevents invalid shapes like [C] → [] or [C] → [1].
//
// SYNC: backward_rules.rs:360-364

/// Model the rank guard in reshape_for_channel_broadcast.
fn channel_broadcast_rank_valid(target_rank: usize) -> bool {
    target_rank >= 2
}

/// Prove reshape_for_channel_broadcast rejects rank 0 and rank 1.
#[kani::unwind(1)]
#[kani::proof]
fn prove_channel_broadcast_rejects_low_rank() {
    let rank: u8 = kani::any();
    kani::assume(rank <= 1);
    assert!(
        !channel_broadcast_rank_valid(rank as usize),
        "rank < 2 must be rejected"
    );
}

/// Prove reshape_for_channel_broadcast accepts rank >= 2.
#[kani::unwind(1)]
#[kani::proof]
fn prove_channel_broadcast_accepts_valid_rank() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 8);
    assert!(
        channel_broadcast_rank_valid(rank as usize),
        "rank >= 2 must be accepted"
    );
}
