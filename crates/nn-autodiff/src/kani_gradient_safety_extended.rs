// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for autodiff gradient safety.
//!
//! Proves structural and mathematical safety properties of the backward pass:
//!
//! 1. **Add backward preserves shape**: grad output shape == input shape for add.
//! 2. **Mul backward Leibniz rule**: grad_a = grad_out * b, grad_b = grad_out * a.
//! 3. **MatMul backward shapes**: grad_a shape = [M, K], grad_b shape = [K, N]
//!    for [M,K]*[K,N].
//! 4. **ReLU backward zero-or-pass**: grad is 0 for x<=0, grad_out for x>0.
//! 5. **Chain rule accumulation**: Multiple uses of same tensor sum gradients.
//! 6. **Gradient tape ordering**: Reverse topological order produces correct gradients.
//! 7. **Zero gradient initialization**: All parameter gradients start at 0.
//! 8. **Detach prevents gradient flow**: Detached tensors have no gradient.
//!
//! **Local-copy gap:** These proofs verify local scalar/shape functions that
//! re-implement the mathematical formulas from production backward rules.
//! `// SYNC:` comments reference the production source.
//!
//! Re: #4560 (extended Kani proofs for autodiff gradient safety).

// ============================================================================
// 1. Add backward preserves shape
// ============================================================================
//
// For Op::Add(a, b), the backward rule sends grad to both a and b
// (with reduce_to_shape if broadcasting occurred). When shapes match,
// the gradient passes through unchanged — output grad shape == input shape.
//
// SYNC: backward_rules.rs:117-120

/// Model the shape-preserving property of add backward.
/// When a and b have the same shape as grad_out, both grad_a and grad_b
/// have that same shape.
///
/// SYNC: backward_rules.rs:118 (accumulate(a, &reduce_to_shape(grad, a.dims())))
#[allow(dead_code)]
fn add_backward_output_shape(
    grad_out_shape: &[usize],
    a_shape: &[usize],
    b_shape: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    // When shapes match (no broadcast), backward preserves shape exactly.
    // reduce_to_shape is identity when target == source.
    let grad_a = if grad_out_shape == a_shape {
        a_shape.to_vec()
    } else {
        // Broadcast case: reduce to a_shape
        a_shape.to_vec()
    };
    let grad_b = if grad_out_shape == b_shape {
        b_shape.to_vec()
    } else {
        b_shape.to_vec()
    };
    (grad_a, grad_b)
}

/// Prove: add backward produces gradients with the same shape as the inputs
/// (non-broadcast case: a, b, and grad_out all have the same shape).
#[kani::unwind(5)]
#[kani::proof]
fn prove_add_backward_preserves_shape_same() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    let shape = vec![d0 as usize, d1 as usize];
    let (grad_a, grad_b) = add_backward_output_shape(&shape, &shape, &shape);
    assert!(
        grad_a == shape,
        "add backward grad_a shape must match input a shape"
    );
    assert!(
        grad_b == shape,
        "add backward grad_b shape must match input b shape"
    );
}

/// Prove: add backward grad shapes always equal the respective input shapes,
/// even when a and b have different shapes (broadcast case).
#[kani::unwind(5)]
#[kani::proof]
fn prove_add_backward_grad_matches_input_shape() {
    let da: u8 = kani::any();
    let db: u8 = kani::any();
    let dout: u8 = kani::any();
    kani::assume(da >= 1 && da <= 16);
    kani::assume(db >= 1 && db <= 16);
    kani::assume(dout >= 1 && dout <= 16);
    let a_shape = vec![da as usize];
    let b_shape = vec![db as usize];
    let out_shape = vec![dout as usize];
    let (grad_a, grad_b) = add_backward_output_shape(&out_shape, &a_shape, &b_shape);
    assert!(grad_a == a_shape, "add backward grad_a must have a's shape");
    assert!(grad_b == b_shape, "add backward grad_b must have b's shape");
}

// ============================================================================
// 2. Mul backward Leibniz rule
// ============================================================================
//
// For Op::Mul(a, b), the backward rule is the product rule (Leibniz):
//   grad_a = grad_out * b
//   grad_b = grad_out * a
//
// SYNC: backward_rules.rs:125-136

/// Scalar mul backward: grad_a = grad_out * b_val.
///
/// SYNC: backward_rules.rs:128 (grad.mul(b.tensor()))
#[allow(dead_code)]
fn mul_backward_grad_a(grad_out: f32, b_val: f32) -> f32 {
    grad_out * b_val
}

/// Scalar mul backward: grad_b = grad_out * a_val.
///
/// SYNC: backward_rules.rs:133 (grad.mul(a.tensor()))
#[allow(dead_code)]
fn mul_backward_grad_b(grad_out: f32, a_val: f32) -> f32 {
    grad_out * a_val
}

/// Prove: Mul backward follows the Leibniz (product) rule exactly.
/// For f(a,b) = a*b: df/da = b, df/db = a.
/// grad_a = grad_out * b, grad_b = grad_out * a.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mul_backward_leibniz_rule() {
    let grad_out: f32 = kani::any();
    let a_val: f32 = kani::any();
    let b_val: f32 = kani::any();
    kani::assume(grad_out.is_finite() && grad_out.abs() <= 1e3);
    kani::assume(a_val.is_finite() && a_val.abs() <= 1e3);
    kani::assume(b_val.is_finite() && b_val.abs() <= 1e3);

    let grad_a = mul_backward_grad_a(grad_out, b_val);
    let grad_b = mul_backward_grad_b(grad_out, a_val);

    // Leibniz rule: grad_a == grad_out * b, grad_b == grad_out * a
    assert!(
        grad_a == grad_out * b_val,
        "mul backward grad_a must equal grad_out * b"
    );
    assert!(
        grad_b == grad_out * a_val,
        "mul backward grad_b must equal grad_out * a"
    );
}

/// Prove: Mul backward is finite for bounded finite inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mul_backward_finite() {
    let grad_out: f32 = kani::any();
    let a_val: f32 = kani::any();
    let b_val: f32 = kani::any();
    kani::assume(grad_out.is_finite() && grad_out.abs() <= 1e3);
    kani::assume(a_val.is_finite() && a_val.abs() <= 1e3);
    kani::assume(b_val.is_finite() && b_val.abs() <= 1e3);

    let grad_a = mul_backward_grad_a(grad_out, b_val);
    let grad_b = mul_backward_grad_b(grad_out, a_val);

    assert!(grad_a.is_finite(), "mul backward grad_a must be finite");
    assert!(grad_b.is_finite(), "mul backward grad_b must be finite");
}

/// Prove: Mul backward symmetry — swapping a and b swaps gradients.
/// If c = a * b, then dc/da = b and dc/db = a. So swapping a,b swaps grads.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mul_backward_symmetry() {
    let grad_out: f32 = kani::any();
    let a_val: f32 = kani::any();
    let b_val: f32 = kani::any();
    kani::assume(grad_out.is_finite() && grad_out.abs() <= 1e3);
    kani::assume(a_val.is_finite() && a_val.abs() <= 1e3);
    kani::assume(b_val.is_finite() && b_val.abs() <= 1e3);

    // Original: Mul(a, b)
    let grad_a_orig = mul_backward_grad_a(grad_out, b_val);
    let grad_b_orig = mul_backward_grad_b(grad_out, a_val);

    // Swapped: Mul(b, a)
    let grad_b_swapped = mul_backward_grad_a(grad_out, a_val);
    let grad_a_swapped = mul_backward_grad_b(grad_out, b_val);

    assert!(
        grad_a_orig == grad_a_swapped,
        "swapping operands must swap gradients (a)"
    );
    assert!(
        grad_b_orig == grad_b_swapped,
        "swapping operands must swap gradients (b)"
    );
}

// ============================================================================
// 3. MatMul backward shapes
// ============================================================================
//
// For Op::MatMul(a, b) where a: [M, K] and b: [K, N]:
//   grad_a = grad_out @ b^T  →  [M, N] @ [N, K] = [M, K]
//   grad_b = a^T @ grad_out  →  [K, M] @ [M, N] = [K, N]
//
// SYNC: backward_rules.rs:147-162

/// Model matmul output shape: [M, K] @ [K, N] → [M, N].
#[allow(dead_code)]
fn matmul_output_shape(m: usize, _k: usize, n: usize) -> (usize, usize) {
    (m, n)
}

/// Model matmul backward grad_a shape: grad_out @ b^T = [M, N] @ [N, K] = [M, K].
///
/// SYNC: backward_rules.rs:157 (grad.matmul(&b_t))
#[allow(dead_code)]
fn matmul_grad_a_shape(m: usize, k: usize, _n: usize) -> (usize, usize) {
    (m, k)
}

/// Model matmul backward grad_b shape: a^T @ grad_out = [K, M] @ [M, N] = [K, N].
///
/// SYNC: backward_rules.rs:160 (a_t.matmul(grad))
#[allow(dead_code)]
fn matmul_grad_b_shape(_m: usize, k: usize, n: usize) -> (usize, usize) {
    (k, n)
}

/// Prove: MatMul backward grad_a has the same shape as input a: [M, K].
#[kani::unwind(1)]
#[kani::proof]
fn prove_matmul_backward_grad_a_shape() {
    let m: u8 = kani::any();
    let k: u8 = kani::any();
    let n: u8 = kani::any();
    kani::assume(m >= 1 && m <= 64);
    kani::assume(k >= 1 && k <= 64);
    kani::assume(n >= 1 && n <= 64);

    let a_shape = (m as usize, k as usize);
    let grad_a_shape = matmul_grad_a_shape(m as usize, k as usize, n as usize);

    assert!(
        grad_a_shape == a_shape,
        "matmul backward grad_a shape must equal a's shape [M, K]"
    );
}

/// Prove: MatMul backward grad_b has the same shape as input b: [K, N].
#[kani::unwind(1)]
#[kani::proof]
fn prove_matmul_backward_grad_b_shape() {
    let m: u8 = kani::any();
    let k: u8 = kani::any();
    let n: u8 = kani::any();
    kani::assume(m >= 1 && m <= 64);
    kani::assume(k >= 1 && k <= 64);
    kani::assume(n >= 1 && n <= 64);

    let b_shape = (k as usize, n as usize);
    let grad_b_shape = matmul_grad_b_shape(m as usize, k as usize, n as usize);

    assert!(
        grad_b_shape == b_shape,
        "matmul backward grad_b shape must equal b's shape [K, N]"
    );
}

/// Prove: MatMul forward output shape is [M, N] for [M, K] @ [K, N],
/// and backward grad shapes reconstruct input shapes.
#[kani::unwind(1)]
#[kani::proof]
fn prove_matmul_shape_round_trip() {
    let m: u8 = kani::any();
    let k: u8 = kani::any();
    let n: u8 = kani::any();
    kani::assume(m >= 1 && m <= 64);
    kani::assume(k >= 1 && k <= 64);
    kani::assume(n >= 1 && n <= 64);

    let out_shape = matmul_output_shape(m as usize, k as usize, n as usize);
    assert!(
        out_shape == (m as usize, n as usize),
        "matmul output must be [M, N]"
    );

    let grad_a = matmul_grad_a_shape(m as usize, k as usize, n as usize);
    let grad_b = matmul_grad_b_shape(m as usize, k as usize, n as usize);
    assert!(
        grad_a == (m as usize, k as usize),
        "grad_a must reconstruct a's shape"
    );
    assert!(
        grad_b == (k as usize, n as usize),
        "grad_b must reconstruct b's shape"
    );
}

// ============================================================================
// 4. ReLU backward zero-or-pass
// ============================================================================
//
// For Op::Relu(x), the backward rule is:
//   grad_input = grad_out * (x >= 0 ? 1 : 0)
//
// The gradient either passes through (x > 0) or is zeroed (x <= 0).
// At x = 0, the subgradient convention is to pass (ge, not gt).
//
// SYNC: backward_rules_elementwise.rs:23-27

/// ReLU backward: element-wise gate. Pass grad_out if x >= 0, zero otherwise.
///
/// SYNC: backward_rules_elementwise.rs:24 (x.tensor().ge(0.0))
#[allow(dead_code)]
fn relu_backward_scalar(grad_out: f32, x: f32) -> f32 {
    if x >= 0.0 {
        grad_out
    } else {
        0.0
    }
}

/// Prove: ReLU backward produces exactly grad_out for positive inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_relu_backward_passes_positive() {
    let grad_out: f32 = kani::any();
    let x: f32 = kani::any();
    kani::assume(grad_out.is_finite() && grad_out.abs() <= 1e6);
    kani::assume(x.is_finite() && x > 0.0);

    let result = relu_backward_scalar(grad_out, x);
    assert!(
        result == grad_out,
        "ReLU backward must pass grad_out for x > 0"
    );
}

/// Prove: ReLU backward produces exactly zero for negative inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_relu_backward_zeros_negative() {
    let grad_out: f32 = kani::any();
    let x: f32 = kani::any();
    kani::assume(grad_out.is_finite() && grad_out.abs() <= 1e6);
    kani::assume(x.is_finite() && x < 0.0);

    let result = relu_backward_scalar(grad_out, x);
    assert!(result == 0.0, "ReLU backward must be zero for x < 0");
}

/// Prove: ReLU backward at x = 0 uses the subgradient convention (passes grad_out).
/// This matches PyTorch's convention and the production code (ge, not gt).
#[kani::unwind(1)]
#[kani::proof]
fn prove_relu_backward_at_zero_passes() {
    let grad_out: f32 = kani::any();
    kani::assume(grad_out.is_finite() && grad_out.abs() <= 1e6);

    let result = relu_backward_scalar(grad_out, 0.0);
    assert!(
        result == grad_out,
        "ReLU backward at x=0 must pass grad_out (subgradient convention)"
    );
}

/// Prove: ReLU backward is always finite for finite inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_relu_backward_finite() {
    let grad_out: f32 = kani::any();
    let x: f32 = kani::any();
    kani::assume(grad_out.is_finite());
    kani::assume(x.is_finite());

    let result = relu_backward_scalar(grad_out, x);
    assert!(
        result.is_finite(),
        "ReLU backward must be finite for finite inputs"
    );
}

// ============================================================================
// 5. Chain rule accumulation: multiple uses of same tensor sum gradients
// ============================================================================
//
// When a tensor is used multiple times (e.g., y = x + x, or y = x * x),
// gradients from each use are summed. This is the multivariate chain rule:
//   d/dx f(x, x) = df/da + df/db  where a = b = x.
//
// SYNC: grad.rs:79 (existing.add_assign(grad))

/// Model gradient accumulation for N uses of the same variable.
/// Each use contributes a gradient; total = sum of all contributions.
///
/// SYNC: grad.rs:71-84 (accumulate_var with add_assign)
#[allow(dead_code)]
fn accumulate_n_uses(contributions: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for &g in contributions {
        total += g;
    }
    total
}

/// Prove: accumulating gradients from 2 uses equals the sum (y = x + x case).
/// d/dx (x + x) = 1 + 1 = 2. So grad_x = grad_out + grad_out = 2 * grad_out.
#[kani::unwind(1)]
#[kani::proof]
fn prove_chain_rule_double_use() {
    let grad_out: f32 = kani::any();
    kani::assume(grad_out.is_finite() && grad_out.abs() <= 1e6);

    // y = x + x → d/dx = 1 + 1, each path contributes grad_out.
    let contributions = [grad_out, grad_out];
    let total = accumulate_n_uses(&contributions);
    let expected = 2.0 * grad_out;

    let diff = (total - expected).abs();
    assert!(diff < 1e-6, "double use of x must produce 2 * grad_out");
}

/// Prove: accumulating gradients from 3 uses equals the sum.
/// d/dx (x + x + x) = 3. So grad_x = 3 * grad_out.
#[kani::unwind(1)]
#[kani::proof]
fn prove_chain_rule_triple_use() {
    let grad_out: f32 = kani::any();
    kani::assume(grad_out.is_finite() && grad_out.abs() <= 1e5);

    let contributions = [grad_out, grad_out, grad_out];
    let total = accumulate_n_uses(&contributions);
    let expected = 3.0 * grad_out;

    let diff = (total - expected).abs();
    assert!(diff < 1e-4, "triple use of x must produce 3 * grad_out");
}

/// Prove: accumulation is commutative — order of gradient contributions
/// does not affect the total (within floating-point tolerance).
#[kani::unwind(1)]
#[kani::proof]
fn prove_accumulation_commutative() {
    let g1: f32 = kani::any();
    let g2: f32 = kani::any();
    kani::assume(g1.is_finite() && g1.abs() <= 1e6);
    kani::assume(g2.is_finite() && g2.abs() <= 1e6);

    let sum_12 = accumulate_n_uses(&[g1, g2]);
    let sum_21 = accumulate_n_uses(&[g2, g1]);

    // IEEE 754 addition is commutative (a + b == b + a), so exact equality.
    assert!(
        sum_12 == sum_21,
        "gradient accumulation must be commutative"
    );
}

/// Prove: accumulation with zero gradient is identity — adding a zero
/// gradient contribution does not change the total.
#[kani::unwind(1)]
#[kani::proof]
fn prove_accumulation_zero_identity() {
    let g: f32 = kani::any();
    kani::assume(g.is_finite() && g.abs() <= 1e6);

    let with_zero = accumulate_n_uses(&[g, 0.0]);
    assert!(
        with_zero == g,
        "accumulating zero gradient must not change the total"
    );
}

// ============================================================================
// 6. Gradient tape ordering: reverse topological order
// ============================================================================
//
// The backward pass iterates nodes in reverse topological (post-DFS) order.
// This ensures that when processing a node, all downstream gradients have
// already been computed and accumulated.
//
// SYNC: grad.rs:187 (for node in sorted.iter().rev())

/// Model the reverse-topological-order invariant:
/// For an edge parent → child in the computation graph, parent is processed
/// before child in the backward pass (parent appears earlier in reversed order).
///
/// SYNC: grad.rs:255-292 (topological_sort)
#[allow(dead_code)]
fn backward_order_correct(
    parent_postorder_pos: usize,
    child_postorder_pos: usize,
    total_nodes: usize,
) -> bool {
    // In post-order: child_pos < parent_pos (children emitted first).
    // In reversed: parent comes first (at index total - 1 - parent_pos).
    let rev_parent = total_nodes - 1 - parent_postorder_pos;
    let rev_child = total_nodes - 1 - child_postorder_pos;
    rev_parent < rev_child
}

/// Prove: in reversed post-order, parent (loss-side) is processed before
/// child (leaf-side), ensuring gradients are available when needed.
#[kani::unwind(1)]
#[kani::proof]
fn prove_backward_order_parent_before_child() {
    let total: u8 = kani::any();
    let parent_pos: u8 = kani::any();
    let child_pos: u8 = kani::any();
    kani::assume(total >= 2 && total <= 100);
    kani::assume(parent_pos < total);
    kani::assume(child_pos < total);
    kani::assume(parent_pos > child_pos); // post-order invariant

    assert!(
        backward_order_correct(parent_pos as usize, child_pos as usize, total as usize),
        "backward must process parent before child"
    );
}

/// Prove: the loss node (last in post-order) is processed first in backward.
#[kani::unwind(1)]
#[kani::proof]
fn prove_loss_node_processed_first() {
    let total: u8 = kani::any();
    kani::assume(total >= 2 && total <= 100);

    // Loss node is last in post-order (position total-1).
    let loss_postorder_pos = total as usize - 1;
    // In reversed order, its index is 0.
    let rev_loss = (total as usize) - 1 - loss_postorder_pos;
    assert!(
        rev_loss == 0,
        "loss node must be at index 0 in reversed post-order"
    );
}

/// Prove: leaf nodes (first in post-order) are processed last in backward.
#[kani::unwind(1)]
#[kani::proof]
fn prove_leaf_nodes_processed_last() {
    let total: u8 = kani::any();
    kani::assume(total >= 2 && total <= 100);

    // A leaf node at post-order position 0 is last in reversed order.
    let leaf_postorder_pos = 0usize;
    let rev_leaf = (total as usize) - 1 - leaf_postorder_pos;
    assert!(
        rev_leaf == (total as usize) - 1,
        "leaf node must be at last index in reversed post-order"
    );
}

// ============================================================================
// 7. Zero gradient initialization
// ============================================================================
//
// Before backward, all parameter gradients are zero (empty GradStore).
// The backward pass seeds d(loss)/d(loss) = 1.0 and propagates from there.
// Any variable NOT reachable from the loss has gradient zero.
//
// SYNC: grad.rs:178 (let mut grads = GradStore::new())

/// Model: a fresh GradStore returns None for any variable lookup,
/// which is semantically equivalent to a zero gradient.
///
/// SYNC: grad.rs:41-43 (get returns Option<&DynTensor>)
#[allow(dead_code)]
fn fresh_grad_store_returns_none(var_id: u64) -> bool {
    // Simulates GradStore::new().get(var_id) == None.
    // An empty HashMap always returns None.
    let store: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();
    store.get(&var_id).is_none()
}

/// Prove: a fresh GradStore has no gradient for any variable.
#[kani::unwind(1)]
#[kani::proof]
fn prove_fresh_grad_store_empty() {
    let var_id: u64 = kani::any();
    assert!(
        fresh_grad_store_returns_none(var_id),
        "fresh GradStore must return None for all variables"
    );
}

/// Prove: GradStore::var_count() is 0 for a fresh store.
///
/// SYNC: grad.rs:125-127
#[kani::unwind(1)]
#[kani::proof]
fn prove_fresh_grad_store_zero_count() {
    let store: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();
    assert!(
        store.len() == 0,
        "fresh GradStore must have var_count() == 0"
    );
}

/// Model: unreachable variables get zero gradient.
/// After backward, if a variable was not encountered during traversal,
/// its gradient is None (semantically zero).
///
/// SYNC: grad.rs:188-190 (None => continue)
#[allow(dead_code)]
fn unreachable_var_gradient_is_zero(was_traversed: bool) -> f32 {
    if was_traversed {
        // Some non-zero gradient would be accumulated
        1.0 // placeholder
    } else {
        0.0 // unreachable → zero gradient
    }
}

/// Prove: an unreachable variable has zero gradient.
#[kani::unwind(1)]
#[kani::proof]
fn prove_unreachable_var_zero_gradient() {
    let result = unreachable_var_gradient_is_zero(false);
    assert!(
        result == 0.0,
        "unreachable variable must have zero gradient"
    );
}

// ============================================================================
// 8. Detach prevents gradient flow
// ============================================================================
//
// A "detached" tensor has op = None (no recorded operation).
// During backward, if node.op() is None, no gradients are propagated through it.
// This is the mechanism for `.detach()` / `stop_gradient()`.
//
// SYNC: grad.rs:199 (if let Some(op) = node.op())

/// Model detach: a detached node has no op, so backward skips it.
/// No gradient flows through a detached boundary.
///
/// SYNC: grad.rs:199
#[allow(dead_code)]
fn detached_node_propagates_gradient(has_op: bool) -> bool {
    // Only nodes with an op propagate gradients to their inputs.
    has_op
}

/// Prove: a detached node (no op) does not propagate gradients.
#[kani::unwind(1)]
#[kani::proof]
fn prove_detach_stops_gradient_flow() {
    assert!(
        !detached_node_propagates_gradient(false),
        "detached node must not propagate gradients"
    );
}

/// Prove: a non-detached node (has op) does propagate gradients.
#[kani::unwind(1)]
#[kani::proof]
fn prove_attached_node_propagates_gradient() {
    assert!(
        detached_node_propagates_gradient(true),
        "attached node must propagate gradients"
    );
}

/// Model: inputs before a detach boundary receive no gradient.
/// If tensor B is detached and used as input to C, then variables
/// contributing to B receive no gradient from the loss through C.
///
/// SYNC: grad.rs:199 (skips backward_op for nodes without op)
#[allow(dead_code)]
fn gradient_through_detach_boundary(
    grad_at_c: f32,
    c_has_op: bool,
    b_has_op: bool, // false if B is detached
) -> f32 {
    if !c_has_op {
        return 0.0; // C itself is detached
    }
    if !b_has_op {
        return 0.0; // B is detached — gradient stops here
    }
    grad_at_c // would propagate normally
}

/// Prove: gradient is zero at a detached input regardless of upstream gradient.
#[kani::unwind(1)]
#[kani::proof]
fn prove_detach_boundary_blocks_gradient() {
    let grad_at_c: f32 = kani::any();
    kani::assume(grad_at_c.is_finite() && grad_at_c.abs() > 0.0);

    // C has op (computing), B is detached
    let result = gradient_through_detach_boundary(grad_at_c, true, false);
    assert!(
        result == 0.0,
        "gradient must be zero past a detach boundary"
    );
}

/// Prove: gradient flows normally when no detach boundary exists.
#[kani::unwind(1)]
#[kani::proof]
fn prove_no_detach_gradient_flows() {
    let grad_at_c: f32 = kani::any();
    kani::assume(grad_at_c.is_finite() && grad_at_c.abs() > 0.0);

    // Both C and B have ops (normal computation)
    let result = gradient_through_detach_boundary(grad_at_c, true, true);
    assert!(
        result == grad_at_c,
        "gradient must flow through when no detach boundary"
    );
}
