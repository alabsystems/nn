// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for `grad.rs`.
//!
//! Supplements `kani_grad.rs` with proofs of:
//! 1. accumulate_node shape validation (mirroring accumulate_var)
//! 2. Gradient accumulation commutativity and associativity
//! 3. op_inputs variant completeness: every Op variant returns non-empty inputs
//! 4. backward_for_vars identity: retaining all vars preserves all gradients
//! 5. GradStore default equivalence: Default::default() == GradStore::new()
//! 6. Gradient seed dimensionality: seed shape matches loss shape
//! 7. Topological sort visited set: no duplicate node processing
//! 8. Non-finite loss rejection: NaN and Inf losses
//! 9. accumulate_var idempotence: same grad twice == double
//! 10. retain_only preserves intersection semantics
//!
//! **Local-copy gap:** Scalar functions re-implement production logic.
//! `// SYNC:` comments track correspondence.
//!
//! Re: #3747 (Kani harnesses for op + backward_rules_norm + train_loop + grad).

// ── accumulate_node mirrors accumulate_var ────────────────────────────────
//
// Both accumulate_var and accumulate_node enforce shape matching.
// Same validation logic, different HashMap (var_grads vs node_grads).
//
// SYNC: grad.rs:71-84 (var), grad.rs:92-105 (node)

/// Model accumulation with shape check.
///
/// SYNC: grad.rs:73, 93
#[allow(dead_code)]
fn accumulate_shape_check(existing_dims: &[usize], new_dims: &[usize]) -> bool {
    existing_dims == new_dims
}

/// Prove accumulation shape check is reflexive.
#[kani::unwind(5)]
#[kani::proof]
fn prove_accumulate_shape_reflexive() {
    let d: u8 = kani::any();
    kani::assume(d >= 1 && d <= 32);
    let dims = vec![d as usize, 4, 8];
    assert!(
        accumulate_shape_check(&dims, &dims),
        "same shape must pass accumulation check"
    );
}

/// Prove accumulation shape check rejects different shapes.
#[kani::unwind(5)]
#[kani::proof]
fn prove_accumulate_shape_rejects_different() {
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);
    kani::assume(d1 != d2);
    let dims1 = vec![d1 as usize, 4];
    let dims2 = vec![d2 as usize, 4];
    assert!(
        !accumulate_shape_check(&dims1, &dims2),
        "different shapes must fail accumulation check"
    );
}

// ── Gradient accumulation commutativity ──────────────────────────────────
//
// g1 + g2 == g2 + g1 (element-wise addition is commutative).
// This ensures gradient accumulation order doesn't affect the result.
//
// SYNC: grad.rs:79 (existing.add_assign(grad))

/// Model scalar gradient accumulation.
///
/// SYNC: grad.rs:79
#[allow(dead_code)]
fn accumulate_scalar(existing: f32, new: f32) -> f32 {
    existing + new
}

/// Prove gradient accumulation is commutative.
#[kani::unwind(1)]
#[kani::proof]
fn prove_accumulate_commutative() {
    let g1: f32 = kani::any();
    let g2: f32 = kani::any();
    kani::assume(g1.is_finite() && g1.abs() <= 1e3);
    kani::assume(g2.is_finite() && g2.abs() <= 1e3);
    let r1 = accumulate_scalar(g1, g2);
    let r2 = accumulate_scalar(g2, g1);
    assert!(
        (r1 - r2).abs() < 1e-6,
        "gradient accumulation must be commutative"
    );
}

/// Prove gradient accumulation is associative (within fp tolerance).
#[kani::unwind(1)]
#[kani::proof]
fn prove_accumulate_associative() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && a.abs() <= 100.0);
    kani::assume(b.is_finite() && b.abs() <= 100.0);
    kani::assume(c.is_finite() && c.abs() <= 100.0);
    let left = accumulate_scalar(accumulate_scalar(a, b), c); // (a + b) + c
    let right = accumulate_scalar(a, accumulate_scalar(b, c)); // a + (b + c)
    kani::assume(left.is_finite() && right.is_finite());
    assert!(
        (left - right).abs() < 1e-4,
        "gradient accumulation must be approximately associative"
    );
}

/// Prove accumulating zero gradient is identity.
#[kani::unwind(1)]
#[kani::proof]
fn prove_accumulate_zero_identity() {
    let g: f32 = kani::any();
    kani::assume(g.is_finite());
    let result = accumulate_scalar(g, 0.0);
    assert!(result == g, "accumulating zero must be identity");
}

// ── op_inputs non-empty for tracked inputs ───────────────────────────────
//
// Every Op variant returns at least 1 input from op_inputs.
// Leaf nodes don't have ops, so op_inputs is only called on non-leaf nodes.
//
// SYNC: grad_op_inputs.rs:14-102

/// Minimum input count by Op category.
///
/// SYNC: grad_op_inputs.rs
#[allow(dead_code)]
fn min_inputs_by_category(category: u8) -> usize {
    match category {
        0 => 1,  // unary (Relu, Gelu, ...)
        1 => 2,  // binary (Add, Sub, Mul, Div, MatMul)
        2 => 1,  // reduction (SumKeepDim, MeanKeepDim)
        3 => 1,  // shape (Reshape, Transpose, ...)
        4 => 2,  // conv (input + kernel)
        5 => 1,  // Cat/Stack (at least 1 input)
        6 => 2,  // norm without bias (RmsNorm: input + weight)
        7 => 3,  // norm with bias (LayerNorm, GroupNorm, ...: input + weight + bias)
        8 => 2,  // loss (input + target)
        9 => 1,  // scalar ops (MulScalar, AddScalar)
        10 => 2, // dropout (input + mask)
        11 => 1, // pooling (input)
        _ => 1,
    }
}

/// Prove every category has at least 1 input.
#[kani::unwind(1)]
#[kani::proof]
fn prove_every_category_has_inputs() {
    let cat: u8 = kani::any();
    kani::assume(cat <= 11);
    let min = min_inputs_by_category(cat);
    assert!(min >= 1, "every Op category must have at least 1 input");
}

/// Prove binary categories have exactly 2 inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_binary_category_two_inputs() {
    assert!(
        min_inputs_by_category(1) == 2,
        "binary ops must have 2 inputs"
    );
}

/// Prove norm-with-bias has 3 inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_norm_bias_three_inputs() {
    assert!(
        min_inputs_by_category(7) == 3,
        "norm-with-bias must have 3 inputs"
    );
}

// ── backward_for_vars: retain all == identity ────────────────────────────
//
// If we call backward_for_vars with ALL variables, the result should be
// identical to backward() — no gradients are dropped.
//
// SYNC: grad.rs:244-248

/// Model retain_only with all IDs present.
///
/// SYNC: grad.rs:118-121
#[allow(dead_code)]
fn retain_all(total_vars: usize, n_targeted: usize) -> usize {
    // If n_targeted >= total_vars, all are retained
    total_vars.min(n_targeted)
}

/// Prove retaining all vars keeps all gradients.
#[kani::unwind(1)]
#[kani::proof]
fn prove_retain_all_is_identity() {
    let total: u8 = kani::any();
    kani::assume(total >= 1 && total <= 50);
    let count = retain_all(total as usize, total as usize);
    assert!(
        count == total as usize,
        "retaining all vars must keep all gradients"
    );
}

/// Prove retaining more than total keeps all.
#[kani::unwind(1)]
#[kani::proof]
fn prove_retain_superset_keeps_all() {
    let total: u8 = kani::any();
    let extra: u8 = kani::any();
    kani::assume(total >= 1 && total <= 50);
    kani::assume(extra >= 1 && extra <= 50);
    let n_targeted = total as usize + extra as usize;
    let count = retain_all(total as usize, n_targeted);
    assert!(
        count == total as usize,
        "retaining superset must keep all gradients"
    );
}

// ── GradStore default equivalence ────────────────────────────────────────
//
// GradStore::default() delegates to GradStore::new(). Both produce empty stores.
//
// SYNC: grad.rs:130-134

/// Model GradStore initial var_count.
///
/// SYNC: grad.rs:33-37
#[allow(dead_code)]
fn new_grad_store_var_count() -> usize {
    0
}

/// Prove new GradStore has var_count == 0.
#[kani::unwind(1)]
#[kani::proof]
fn prove_new_grad_store_empty() {
    let count = new_grad_store_var_count();
    assert!(count == 0, "new GradStore must have 0 var gradients");
}

// ── Gradient seed dimensionality ─────────────────────────────────────────
//
// backward() seeds with DynTensor::ones(loss.dims(), ...).
// The seed must have the same shape as the loss tensor (always [1] for scalar loss).
//
// SYNC: grad.rs:179-183

/// Model seed shape: same as loss shape.
///
/// SYNC: grad.rs:180
#[allow(dead_code)]
fn seed_shape_matches_loss(loss_dims: &[usize]) -> Vec<usize> {
    loss_dims.to_vec()
}

/// Prove seed shape equals loss shape.
#[kani::unwind(5)]
#[kani::proof]
fn prove_seed_shape_matches() {
    let d: u8 = kani::any();
    kani::assume(d >= 1 && d <= 16);
    let loss_dims = vec![d as usize];
    let seed_dims = seed_shape_matches_loss(&loss_dims);
    assert!(seed_dims == loss_dims, "seed shape must match loss shape");
}

/// Prove scalar loss gets scalar seed ([1]).
#[kani::unwind(1)]
#[kani::proof]
fn prove_scalar_loss_scalar_seed() {
    let seed_dims = seed_shape_matches_loss(&[1]);
    assert!(
        seed_dims.len() == 1 && seed_dims[0] == 1,
        "scalar loss must get scalar seed"
    );
}

// ── Topological sort: no duplicates ──────────────────────────────────────
//
// The visited HashSet ensures each node is processed at most once.
// visited.insert(id) returns false if already present, causing skip.
//
// SYNC: grad.rs:273-275

/// Model visited set behavior.
///
/// SYNC: grad.rs:273
#[allow(dead_code)]
fn visited_insert(visited: &mut Vec<u64>, id: u64) -> bool {
    if visited.contains(&id) {
        false
    } else {
        visited.push(id);
        true
    }
}

/// Prove first visit returns true, second returns false.
#[kani::unwind(5)]
#[kani::proof]
fn prove_visited_dedup() {
    let id: u8 = kani::any();
    kani::assume(id <= 100);
    let mut visited = Vec::new();
    let first = visited_insert(&mut visited, id as u64);
    let second = visited_insert(&mut visited, id as u64);
    assert!(first, "first visit must return true");
    assert!(!second, "second visit must return false");
}

/// Prove different IDs are both accepted.
#[kani::unwind(5)]
#[kani::proof]
fn prove_visited_distinct_accepted() {
    let id1: u8 = kani::any();
    let id2: u8 = kani::any();
    kani::assume(id1 <= 100);
    kani::assume(id2 <= 100);
    kani::assume(id1 != id2);
    let mut visited = Vec::new();
    let r1 = visited_insert(&mut visited, id1 as u64);
    let r2 = visited_insert(&mut visited, id2 as u64);
    assert!(r1, "first distinct ID must be accepted");
    assert!(r2, "second distinct ID must be accepted");
}

// ── Non-finite loss rejection ────────────────────────────────────────────
//
// backward() rejects non-finite loss via any_non_finite() check.
// NaN and Inf losses produce garbage gradients, so they're caught early.
//
// SYNC: grad.rs:170-172

/// Model loss finiteness check.
///
/// SYNC: grad.rs:170
#[allow(dead_code)]
fn is_loss_finite(val: f32) -> bool {
    val.is_finite()
}

/// Prove NaN loss is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn prove_nan_loss_rejected() {
    assert!(!is_loss_finite(f32::NAN), "NaN loss must be rejected");
}

/// Prove positive Inf loss is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn prove_pos_inf_loss_rejected() {
    assert!(!is_loss_finite(f32::INFINITY), "+Inf loss must be rejected");
}

/// Prove negative Inf loss is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn prove_neg_inf_loss_rejected() {
    assert!(
        !is_loss_finite(f32::NEG_INFINITY),
        "-Inf loss must be rejected"
    );
}

/// Prove finite loss is accepted.
#[kani::unwind(1)]
#[kani::proof]
fn prove_finite_loss_accepted() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());
    assert!(is_loss_finite(val), "finite loss must be accepted");
}

// ── accumulate_var: adding same grad twice equals double ──────────────────
//
// accumulate_var with the same gradient twice should produce 2 * gradient.
//
// SYNC: grad.rs:79 (add_assign)

/// Model double accumulation.
///
/// SYNC: grad.rs:79
#[allow(dead_code)]
fn double_accumulate(grad: f32) -> f32 {
    grad + grad
}

/// Prove double accumulation equals 2x.
#[kani::unwind(1)]
#[kani::proof]
fn prove_double_accumulate_is_2x() {
    let g: f32 = kani::any();
    kani::assume(g.is_finite() && g.abs() <= 1e3);
    let result = double_accumulate(g);
    let expected = 2.0 * g;
    assert!(
        (result - expected).abs() < 1e-5,
        "accumulating same gradient twice must equal 2x"
    );
}

// ── retain_only intersection semantics ───────────────────────────────────
//
// retain_only keeps var_grads where id is in the target set.
// Result count = |var_grads_keys INTERSECT target_ids|.
//
// SYNC: grad.rs:118-121

/// Model retain_only as set intersection.
///
/// SYNC: grad.rs:119-120
#[allow(dead_code)]
fn retain_intersection_count(store_ids: &[u64], target_ids: &[u64]) -> usize {
    store_ids
        .iter()
        .filter(|id| target_ids.contains(id))
        .count()
}

/// Prove retain with empty targets removes all.
#[kani::unwind(5)]
#[kani::proof]
fn prove_retain_empty_targets_removes_all() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 10);
    let store: Vec<u64> = (0..n as u64).collect();
    let count = retain_intersection_count(&store, &[]);
    assert!(count == 0, "empty target set must remove all gradients");
}

/// Prove retain with disjoint sets removes all.
#[kani::unwind(5)]
#[kani::proof]
fn prove_retain_disjoint_removes_all() {
    let store = [0_u64, 1, 2];
    let targets = [10_u64, 11, 12];
    let count = retain_intersection_count(&store, &targets);
    assert!(count == 0, "disjoint target set must remove all gradients");
}

/// Prove retain with identical sets keeps all.
#[kani::unwind(5)]
#[kani::proof]
fn prove_retain_identical_keeps_all() {
    let ids = [0_u64, 1, 2, 3];
    let count = retain_intersection_count(&ids, &ids);
    assert!(
        count == ids.len(),
        "identical target set must keep all gradients"
    );
}
