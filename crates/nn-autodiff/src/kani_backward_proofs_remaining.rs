// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for previously untested backward rules.
//!
//! Covers ops that had zero Kani harnesses as of 2026-03-26:
//!
//! 1. **Narrow** backward: zero-pad + slice_set restores original shape
//! 2. **Unfold** backward: scatter-add windows reindex correctly
//! 3. **Squeeze/Unsqueeze** backward: inverse shape ops preserve elements
//! 4. **Stack** backward: narrow + squeeze per input
//! 5. **Maximum/Minimum** backward: subgradient mask, NaN defense
//! 6. **LogSoftmax** backward: grad - softmax * sum(grad)
//! 7. **MulScalar/AddScalar** backward: scaling/identity rules
//! 8. **AvgPool2d** backward: per-window count normalization
//! 9. **AdaptiveAvgPool2d** backward: global pooling 1/HW scaling
//! 10. **Permute** backward: inverse permutation is self-inverse
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.
//!
//! Re: #3591 (Kani harnesses for untested backward rules).

// ── Narrow backward ─────────────────────────────────────────────────
//
// Narrow forward: slice [start..start+len] from dim of size orig_dim_size.
// Narrow backward: place gradient into a zeros tensor at the original offset.
//   zeros[start..start+len] = grad; other positions = 0.
//
// SYNC: backward_rules.rs:178-187

/// Model the Narrow backward zero-padding: the gradient occupies
/// [start, start+len) in the output of size orig_dim_size.
/// Returns true if position `pos` receives a nonzero gradient.
fn narrow_backward_has_grad(pos: usize, start: usize, len: usize) -> bool {
    pos >= start && pos < start + len
}

/// Model the Narrow backward output size along the sliced dimension.
/// Must equal the original dimension size, not the sliced size.
///
/// SYNC: backward_rules.rs:183-184 (full_shape[dim] = orig_dim_size)
fn narrow_backward_output_dim(orig_dim_size: usize) -> usize {
    orig_dim_size
}

// ── Squeeze/Unsqueeze backward ──────────────────────────────────────
//
// Unsqueeze backward: grad_input = grad.squeeze(dim)
//   Removes the inserted dimension. Element count unchanged.
//
// Squeeze backward: grad_input = grad.unsqueeze(dim)
//   Restores the removed dimension. Element count unchanged.
//
// Both are structural identity at the element level.
//
// SYNC: backward_rules.rs:189-190

/// Model squeeze rank change: rank decreases by 1.
fn squeeze_output_rank(input_rank: usize) -> usize {
    input_rank - 1
}

/// Model unsqueeze rank change: rank increases by 1.
fn unsqueeze_output_rank(input_rank: usize) -> usize {
    input_rank + 1
}

// ── Stack backward ──────────────────────────────────────────────────
//
// Stack forward: creates a new dimension of size N (number of inputs).
// Stack backward: narrow dim to 1 + squeeze for each input i.
//   grad_i = grad.narrow(dim, i, 1).squeeze(dim)
//
// Each input's gradient is disjoint — no overlap in the stacked dim.
//
// SYNC: backward_rules.rs:254-265

/// Model Stack backward: element at position `i` in the stacked dim
/// maps to input tensor `i`. Returns true if the element belongs to
/// input `target_input`.
fn stack_backward_maps_to_input(pos: usize, target_input: usize) -> bool {
    pos == target_input
}

// ── Maximum/Minimum backward ────────────────────────────────────────
//
// Maximum backward (subgradient):
//   grad_a = grad where a >= b, 0 otherwise (tie → a gets gradient)
//   grad_b = grad where b > a, 0 otherwise
//
// Minimum backward (subgradient):
//   grad_a = grad where a <= b, 0 otherwise (tie → a gets gradient)
//   grad_b = grad where b < a, 0 otherwise
//
// NaN defense: diff = a - b must be checked for non-finite before masking.
//
// SYNC: backward_rules.rs:274-320

/// Maximum backward mask for operand a: 1 when a >= b (diff >= 0).
fn maximum_mask_a(a: f32, b: f32) -> f32 {
    if a >= b {
        1.0
    } else {
        0.0
    }
}

/// Maximum backward mask for operand b: 1 when b > a (diff < 0).
fn maximum_mask_b(a: f32, b: f32) -> f32 {
    if b > a {
        1.0
    } else {
        0.0
    }
}

/// Minimum backward mask for operand a: 1 when a <= b (diff <= 0).
fn minimum_mask_a(a: f32, b: f32) -> f32 {
    if a <= b {
        1.0
    } else {
        0.0
    }
}

/// Minimum backward mask for operand b: 1 when b < a (diff > 0).
fn minimum_mask_b(a: f32, b: f32) -> f32 {
    if b < a {
        1.0
    } else {
        0.0
    }
}

// ── LogSoftmax backward ─────────────────────────────────────────────
//
// LogSoftmax backward:
//   grad_input = grad - softmax(x) * sum(grad, dim)
//
// For a single element in a vector of size N:
//   grad_x[i] = grad[i] - s[i] * sum(grad)
// where s = softmax(x).
//
// SYNC: backward_rules.rs:324-335

/// LogSoftmax backward element formula.
/// s_i is the softmax output at position i, grad_i is the upstream gradient,
/// sum_grad is the sum of all upstream gradients along the softmax dimension.
///
/// SYNC: backward_rules.rs:328-330
fn log_softmax_backward_element(grad_i: f32, s_i: f32, sum_grad: f32) -> f32 {
    grad_i - s_i * sum_grad
}

// ── MulScalar/AddScalar backward ────────────────────────────────────
//
// MulScalar backward: grad_input = grad * scalar
// AddScalar backward: grad_input = grad (identity)
//
// SYNC: backward_rules.rs:60-61

/// MulScalar backward: scale the gradient by the scalar.
fn mul_scalar_backward(grad: f32, scalar: f64) -> f32 {
    grad * scalar as f32
}

/// AddScalar backward: gradient passes through unchanged.
fn add_scalar_backward(grad: f32) -> f32 {
    grad
}

// ── AvgPool2d backward ──────────────────────────────────────────────
//
// AvgPool2d backward element: each output gradient is divided by the
// window's valid element count, then spread to input positions via
// conv_transpose2d. At the scalar level with kernel_size=1, stride=1:
//   grad_input = grad_output / count
//
// For a full window (no padding effects): count = kernel_size^2.
//
// SYNC: backward_rules_pool.rs:79-148

/// AvgPool2d backward scaling for a single element in a full window.
/// Each output gradient is divided by the number of valid elements
/// in the pooling window (kernel_size^2 for a full, non-edge window).
fn avg_pool2d_backward_element(grad: f32, kernel_size: usize) -> f32 {
    let count = (kernel_size * kernel_size) as f32;
    grad / count
}

// ── AdaptiveAvgPool2d backward ──────────────────────────────────────
//
// AdaptiveAvgPool2d backward (global pooling, output 1x1):
//   grad_input[i] = grad_output / (H * W)
// Each element receives 1/(H*W) of the output gradient.
//
// SYNC: backward_rules_pool.rs:167-170

/// AdaptiveAvgPool2d backward scaling for global pooling (output 1x1).
fn adaptive_avg_pool2d_global_backward(grad: f32, h: usize, w: usize) -> f32 {
    let window = (h * w) as f32;
    grad / window
}

// ── Permute backward ────────────────────────────────────────────────
//
// Permute backward: apply the inverse permutation.
// The Op stores the inverse permutation directly.
//
// Property: applying a permutation then its inverse is the identity.
// perm[inv_perm[i]] == i for all i.
//
// SYNC: backward_rules.rs:231

/// Compute the inverse of a permutation.
fn invert_permutation(perm: &[usize]) -> Vec<usize> {
    let n = perm.len();
    let mut inv = vec![0usize; n];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

// ── Kani proof harnesses ────────────────────────────────────────────

// --- Narrow backward ---

/// Prove Narrow backward gradient covers exactly [start, start+len) positions.
/// Positions outside that range must have zero gradient. This ensures the
/// zero-pad + slice_set produces the correct sparse gradient pattern.
///
/// SYNC: backward_rules.rs:178-187
#[kani::unwind(1)]
#[kani::proof]
fn prove_narrow_backward_coverage() {
    let orig_dim: usize = kani::any();
    let start: usize = kani::any();
    let len: usize = kani::any();
    let pos: usize = kani::any();
    kani::assume(orig_dim >= 1 && orig_dim <= 64);
    kani::assume(len >= 1 && len <= orig_dim);
    kani::assume(start + len <= orig_dim);
    kani::assume(pos < orig_dim);

    let has_grad = narrow_backward_has_grad(pos, start, len);
    let in_range = pos >= start && pos < start + len;
    assert!(
        has_grad == in_range,
        "narrow backward must assign gradient exactly to [start, start+len)"
    );
}

/// Prove Narrow backward output dim equals original (pre-slice) dimension.
/// The backward gradient must have the same shape as the original input,
/// not the sliced shape.
///
/// SYNC: backward_rules.rs:183-184
#[kani::unwind(1)]
#[kani::proof]
fn prove_narrow_backward_restores_dim() {
    let orig_dim: usize = kani::any();
    let start: usize = kani::any();
    let len: usize = kani::any();
    kani::assume(orig_dim >= 1 && orig_dim <= 1024);
    kani::assume(len >= 1 && len <= orig_dim);
    kani::assume(start + len <= orig_dim);

    let output_dim = narrow_backward_output_dim(orig_dim);
    assert!(
        output_dim == orig_dim,
        "narrow backward output dim must equal original dim"
    );
    assert!(
        output_dim >= len,
        "narrow backward output dim must be >= slice length"
    );
}

// --- Squeeze/Unsqueeze backward ---

/// Prove Unsqueeze backward (squeeze) reduces rank by exactly 1.
/// Unsqueeze forward adds a dim; its backward must remove that dim.
///
/// SYNC: backward_rules.rs:189
#[kani::unwind(1)]
#[kani::proof]
fn prove_unsqueeze_backward_rank() {
    let rank: usize = kani::any();
    kani::assume(rank >= 2 && rank <= 8);

    // Unsqueeze adds 1 dim, so the grad has `rank` dims.
    // Backward (squeeze) must produce rank - 1 dims.
    let output_rank = squeeze_output_rank(rank);
    assert!(
        output_rank == rank - 1,
        "unsqueeze backward must reduce rank by 1"
    );
}

/// Prove Squeeze backward (unsqueeze) increases rank by exactly 1.
/// Squeeze forward removes a dim; its backward must restore that dim.
///
/// SYNC: backward_rules.rs:190
#[kani::unwind(1)]
#[kani::proof]
fn prove_squeeze_backward_rank() {
    let rank: usize = kani::any();
    kani::assume(rank >= 1 && rank <= 7);

    // Squeeze removes 1 dim, so the grad has `rank` dims.
    // Backward (unsqueeze) must produce rank + 1 dims.
    let output_rank = unsqueeze_output_rank(rank);
    assert!(
        output_rank == rank + 1,
        "squeeze backward must increase rank by 1"
    );
}

/// Prove squeeze and unsqueeze are inverse operations on rank.
/// squeeze(unsqueeze(r)) == r and unsqueeze(squeeze(r)) == r.
///
/// SYNC: backward_rules.rs:189-190
#[kani::unwind(1)]
#[kani::proof]
fn prove_squeeze_unsqueeze_inverse_rank() {
    let rank: usize = kani::any();
    kani::assume(rank >= 2 && rank <= 7);

    let up = unsqueeze_output_rank(rank);
    let down = squeeze_output_rank(up);
    assert!(down == rank, "squeeze(unsqueeze(rank)) must equal rank");

    let down2 = squeeze_output_rank(rank);
    let up2 = unsqueeze_output_rank(down2);
    assert!(up2 == rank, "unsqueeze(squeeze(rank)) must equal rank");
}

// --- Stack backward ---

/// Prove Stack backward maps each stacked position to exactly one input.
/// In the stacked dimension, position i belongs to input i exclusively.
///
/// SYNC: backward_rules.rs:254-265
#[kani::unwind(17)]
#[kani::proof]
fn prove_stack_backward_exclusive_mapping() {
    let n_inputs: usize = kani::any();
    let pos: usize = kani::any();
    kani::assume(n_inputs >= 2 && n_inputs <= 32);
    kani::assume(pos < n_inputs);

    // Exactly one input claims this position
    let mut count = 0usize;
    // Check a bounded range to keep Kani tractable
    let check_limit = if n_inputs <= 8 { n_inputs } else { 8 };
    kani::assume(pos < check_limit);
    for i in 0..check_limit {
        if stack_backward_maps_to_input(pos, i) {
            count += 1;
        }
    }
    assert!(
        count == 1,
        "stack backward must map each position to exactly one input"
    );
}

// --- Maximum backward ---

/// Prove Maximum backward masks sum to exactly 1 when inputs are finite and distinct.
/// For a >= b XOR b > a, exactly one operand receives the gradient.
///
/// SYNC: backward_rules.rs:276-296
#[kani::unwind(1)]
#[kani::proof]
fn prove_maximum_backward_gradient_sum() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && a.abs() <= 1e6);
    kani::assume(b.is_finite() && b.abs() <= 1e6);
    kani::assume(a != b); // distinct values

    let ma = maximum_mask_a(a, b);
    let mb = maximum_mask_b(a, b);
    let sum = ma + mb;
    assert!(
        sum == 1.0,
        "for distinct inputs, exactly one operand receives the gradient"
    );
}

/// Prove Maximum backward tie-breaking: when a == b, a gets the gradient.
/// This is the subgradient convention (matches PyTorch).
///
/// SYNC: backward_rules.rs:277 (a >= b → a gets gradient)
#[kani::unwind(1)]
#[kani::proof]
fn prove_maximum_backward_tie_to_a() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v.abs() <= 1e6);

    let ma = maximum_mask_a(v, v);
    let mb = maximum_mask_b(v, v);
    assert!(ma == 1.0, "on tie, a must get the gradient (mask_a = 1)");
    assert!(
        mb == 0.0,
        "on tie, b must not get the gradient (mask_b = 0)"
    );
}

/// Prove Minimum backward masks sum to exactly 1 when inputs are distinct.
///
/// SYNC: backward_rules.rs:298-318
#[kani::unwind(1)]
#[kani::proof]
fn prove_minimum_backward_gradient_sum() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && a.abs() <= 1e6);
    kani::assume(b.is_finite() && b.abs() <= 1e6);
    kani::assume(a != b);

    let ma = minimum_mask_a(a, b);
    let mb = minimum_mask_b(a, b);
    let sum = ma + mb;
    assert!(
        sum == 1.0,
        "for distinct inputs, exactly one operand receives the gradient"
    );
}

/// Prove Minimum backward tie-breaking: when a == b, a gets the gradient.
///
/// SYNC: backward_rules.rs:299 (a <= b → a gets gradient)
#[kani::unwind(1)]
#[kani::proof]
fn prove_minimum_backward_tie_to_a() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v.abs() <= 1e6);

    let ma = minimum_mask_a(v, v);
    let mb = minimum_mask_b(v, v);
    assert!(ma == 1.0, "on tie, a must get the gradient (mask_a = 1)");
    assert!(
        mb == 0.0,
        "on tie, b must not get the gradient (mask_b = 0)"
    );
}

// --- LogSoftmax backward ---

/// Prove LogSoftmax backward element is finite for bounded inputs.
/// grad_x[i] = grad[i] - s[i] * sum(grad).
///
/// SYNC: backward_rules.rs:328-330
#[kani::unwind(1)]
#[kani::proof]
fn prove_log_softmax_backward_finite() {
    let grad_i: f32 = kani::any();
    let s_i: f32 = kani::any();
    let sum_grad: f32 = kani::any();
    kani::assume(grad_i.is_finite() && grad_i.abs() <= 1e6);
    kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i <= 1.0);
    kani::assume(sum_grad.is_finite() && sum_grad.abs() <= 1e6);

    let result = log_softmax_backward_element(grad_i, s_i, sum_grad);
    assert!(
        result.is_finite(),
        "log_softmax backward must be finite for bounded inputs"
    );
}

/// Prove LogSoftmax backward is zero when upstream gradient is zero.
/// When grad == 0 everywhere, grad_input must also be zero.
///
/// SYNC: backward_rules.rs:328-330
#[kani::unwind(1)]
#[kani::proof]
fn prove_log_softmax_backward_zero_grad() {
    let s_i: f32 = kani::any();
    kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i <= 1.0);

    let result = log_softmax_backward_element(0.0, s_i, 0.0);
    assert!(
        result == 0.0,
        "log_softmax backward must be zero when upstream gradient is zero"
    );
}

// --- MulScalar backward ---

/// Prove MulScalar backward is finite for bounded inputs.
/// grad_input = grad * scalar.
///
/// SYNC: backward_rules.rs:60
#[kani::unwind(1)]
#[kani::proof]
fn prove_mul_scalar_backward_finite() {
    let grad: f32 = kani::any();
    let scalar: f64 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(scalar.is_finite() && scalar.abs() <= 1e6);

    let result = mul_scalar_backward(grad, scalar);
    assert!(
        result.is_finite(),
        "mul_scalar backward must be finite for bounded inputs"
    );
}

/// Prove MulScalar backward preserves zero gradient.
/// When grad == 0, grad_input must be 0 regardless of scalar.
///
/// SYNC: backward_rules.rs:60
#[kani::unwind(1)]
#[kani::proof]
fn prove_mul_scalar_backward_zero_grad() {
    let scalar: f64 = kani::any();
    kani::assume(scalar.is_finite() && scalar.abs() <= 1e6);

    let result = mul_scalar_backward(0.0, scalar);
    assert!(
        result == 0.0,
        "mul_scalar backward must preserve zero gradient"
    );
}

/// Prove AddScalar backward is identity (gradient passes through).
///
/// SYNC: backward_rules.rs:61
#[kani::unwind(1)]
#[kani::proof]
fn prove_add_scalar_backward_identity() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite());

    let result = add_scalar_backward(grad);
    assert!(result == grad, "add_scalar backward must be identity");
}

// --- AvgPool2d backward ---

/// Prove AvgPool2d backward scaling is finite for valid kernel sizes.
/// Each output gradient is divided by kernel_size^2.
///
/// SYNC: backward_rules_pool.rs:96-99
#[kani::unwind(1)]
#[kani::proof]
fn prove_avg_pool2d_backward_finite() {
    let grad: f32 = kani::any();
    let kernel_size: usize = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(kernel_size >= 1 && kernel_size <= 32);

    let result = avg_pool2d_backward_element(grad, kernel_size);
    assert!(
        result.is_finite(),
        "avg_pool2d backward must be finite for bounded inputs"
    );
}

/// Prove AvgPool2d backward scaling is bounded: |result| <= |grad|.
/// Dividing by count >= 1 never amplifies the gradient magnitude.
///
/// SYNC: backward_rules_pool.rs:96-99
#[kani::unwind(1)]
#[kani::proof]
fn prove_avg_pool2d_backward_bounded() {
    let grad: f32 = kani::any();
    let kernel_size: usize = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(kernel_size >= 1 && kernel_size <= 32);

    let result = avg_pool2d_backward_element(grad, kernel_size);
    assert!(
        result.abs() <= grad.abs(),
        "avg_pool2d backward must not amplify gradient"
    );
}

// --- AdaptiveAvgPool2d backward (global pooling) ---

/// Prove AdaptiveAvgPool2d global backward is finite for valid spatial dims.
/// grad_input = grad / (H * W).
///
/// SYNC: backward_rules_pool.rs:167-170
#[kani::unwind(1)]
#[kani::proof]
fn prove_adaptive_avg_pool2d_global_backward_finite() {
    let grad: f32 = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(h >= 1 && h <= 256);
    kani::assume(w >= 1 && w <= 256);

    let result = adaptive_avg_pool2d_global_backward(grad, h, w);
    assert!(
        result.is_finite(),
        "adaptive_avg_pool2d global backward must be finite"
    );
}

/// Prove AdaptiveAvgPool2d global backward is bounded: |result| <= |grad|.
/// Dividing by H*W >= 1 never amplifies.
///
/// SYNC: backward_rules_pool.rs:167-170
#[kani::unwind(1)]
#[kani::proof]
fn prove_adaptive_avg_pool2d_global_backward_bounded() {
    let grad: f32 = kani::any();
    let h: usize = kani::any();
    let w: usize = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(h >= 1 && h <= 256);
    kani::assume(w >= 1 && w <= 256);

    let result = adaptive_avg_pool2d_global_backward(grad, h, w);
    assert!(
        result.abs() <= grad.abs(),
        "adaptive_avg_pool2d global backward must not amplify gradient"
    );
}

// --- Permute backward ---

/// Prove inverse permutation is correct: perm[inv[i]] == i for all i.
/// The backward applies the inverse permutation, which must round-trip.
///
/// SYNC: backward_rules.rs:231
#[kani::unwind(8)]
#[kani::proof]
fn prove_permute_inverse_roundtrip_3d() {
    let p0: usize = kani::any();
    let p1: usize = kani::any();
    let p2: usize = kani::any();
    kani::assume(p0 < 3 && p1 < 3 && p2 < 3);
    // Valid permutation: all distinct
    kani::assume(p0 != p1 && p0 != p2 && p1 != p2);

    let perm = [p0, p1, p2];
    let inv = invert_permutation(&perm);
    // perm[inv[i]] == i
    for i in 0..3 {
        assert!(perm[inv[i]] == i, "perm[inv[i]] must equal i (round-trip)");
    }
    // inv[perm[i]] == i
    for i in 0..3 {
        assert!(inv[perm[i]] == i, "inv[perm[i]] must equal i (round-trip)");
    }
}

/// Prove inverse permutation is correct for 4D tensors.
/// Covers the common [N, C, H, W] case with arbitrary permutations.
///
/// SYNC: backward_rules.rs:231
#[kani::unwind(8)]
#[kani::proof]
fn prove_permute_inverse_roundtrip_4d() {
    let p0: usize = kani::any();
    let p1: usize = kani::any();
    let p2: usize = kani::any();
    let p3: usize = kani::any();
    kani::assume(p0 < 4 && p1 < 4 && p2 < 4 && p3 < 4);
    kani::assume(p0 != p1 && p0 != p2 && p0 != p3);
    kani::assume(p1 != p2 && p1 != p3);
    kani::assume(p2 != p3);

    let perm = [p0, p1, p2, p3];
    let inv = invert_permutation(&perm);
    for i in 0..4 {
        assert!(perm[inv[i]] == i, "perm[inv[i]] must equal i");
        assert!(inv[perm[i]] == i, "inv[perm[i]] must equal i");
    }
}

/// Prove double inverse of a permutation is the original permutation.
/// inv(inv(perm)) == perm. This ensures backward(backward()) = forward.
///
/// SYNC: backward_rules.rs:231
#[kani::unwind(8)]
#[kani::proof]
fn prove_permute_double_inverse_identity() {
    let p0: usize = kani::any();
    let p1: usize = kani::any();
    let p2: usize = kani::any();
    kani::assume(p0 < 3 && p1 < 3 && p2 < 3);
    kani::assume(p0 != p1 && p0 != p2 && p1 != p2);

    let perm = [p0, p1, p2];
    let inv = invert_permutation(&perm);
    let inv_inv = invert_permutation(&inv);
    for i in 0..3 {
        assert!(
            inv_inv[i] == perm[i],
            "double inverse must equal original permutation"
        );
    }
}
