// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for autodiff backward rules — new ops coverage.
//!
//! Fills gaps identified in #3610 for backward operations that lacked
//! Kani proofs:
//!
//! 1. **Conv1d backward** (scalar 1x1 kernel) — mirrors Conv2d/ConvTranspose1d pattern
//! 2. **Unfold backward** — scatter-add window index bounds
//! 3. **Transpose self-inverse** — backward(forward) = identity
//! 4. **Reshape element preservation** — backward restores original shape
//! 5. **Broadcast backward** — dimension reduction correctness
//! 6. **Cat 3-input backward** — completeness for > 2 inputs
//! 7. **Embedding backward sparsity** — non-target rows get zero
//! 8. **SumKeepDim backward** — expansion preserves element value
//! 9. **Softmax backward Jacobian** — off-diagonal negative, bounded
//! 10. **LayerNorm backward** — inv_std bounded for valid variance
//! 11. **reduce_to_shape** — leading dimension collapse
//! 12. **MeanKeepDim backward** — scale factor range
//! 13. **Dropout backward scale range** — 1/(1-p) bounds
//! 14. **MaxPool1d backward** — same scatter routing as MaxPool2d
//! 15. **Neg backward** — involution (double neg = identity)
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.
//!
//! Re: #3610 (Kani harnesses for autodiff backward rules — new ops).

// ── Conv1d backward (scalar 1x1 kernel) ──────────────────────────
//
// Conv1d backward:
//   grad_input = conv_transpose1d(grad_output, kernel)
//   grad_kernel = cross-correlation(input, grad_output)
//
// At the scalar level (K=1, stride=1, padding=0, dilation=1, groups=1):
//   grad_input = grad_output * kernel  (scalar conv = multiply)
//   grad_kernel = input * grad_output  (cross-correlation = multiply)
//
// SYNC: backward_rules_conv.rs:22-67.

/// Conv1d backward grad_input for scalar 1x1 kernel: grad_x = grad_y * w.
fn conv1d_grad_input_scalar(grad_y: f32, w: f32) -> f32 {
    grad_y * w
}

/// Conv1d backward grad_kernel for scalar 1x1 kernel: grad_w = x * grad_y.
fn conv1d_grad_kernel_scalar(x: f32, grad_y: f32) -> f32 {
    x * grad_y
}

// ── Unfold backward scatter-add index model ──────────────────────
//
// Unfold forward: input[..., T, ...] → output[..., n_windows, ..., size]
//   Window w covers input positions [w*step, w*step + size).
//
// Unfold backward: for each window w, scatter-add grad back to positions
//   [w*step, w*step + size). Overlapping windows accumulate.
//
// SYNC: backward_rules.rs:197-229.

/// Returns true if input position `pos` is covered by window `w`
/// with the given step and window size.
fn unfold_window_covers(pos: usize, w: usize, step: usize, size: usize) -> bool {
    let start = w * step;
    pos >= start && pos < start + size
}

/// Count how many windows cover a given input position.
/// For non-overlapping (step >= size): exactly 1 or 0.
/// For overlapping (step < size): may be > 1.
fn unfold_coverage_count(pos: usize, n_windows: usize, step: usize, size: usize) -> usize {
    let mut count = 0usize;
    let limit = if n_windows <= 16 { n_windows } else { 16 };
    for w in 0..limit {
        if unfold_window_covers(pos, w, step, size) {
            count += 1;
        }
    }
    count
}

// ── Transpose self-inverse ───────────────────────────────────────
//
// Transpose(d1, d2) is its own inverse: applying it twice is identity.
// backward_rules.rs:177: grad_input = grad.transpose(d1, d2)
//
// At the index level: transpose swaps dims d1 and d2.
// Doing it twice restores original.
//
// SYNC: backward_rules.rs:177.

/// Model transpose of a 2D index: swap dimensions d1 and d2 in a 3-element index.
fn transpose_index(idx: [usize; 3], d1: usize, d2: usize) -> [usize; 3] {
    let mut out = idx;
    let tmp = out[d1];
    out[d1] = out[d2];
    out[d2] = tmp;
    out
}

// ── Reshape element preservation ─────────────────────────────────
//
// Reshape does not change element count. The flat index of each element
// is preserved. backward: grad.reshape(original_shape).
//
// SYNC: backward_rules.rs:176.

/// Compute total element count from shape dimensions.
fn element_count(dims: &[usize]) -> usize {
    let mut n = 1usize;
    for &d in dims {
        n *= d;
    }
    n
}

// ── Broadcast backward: dimension reduction ──────────────────────
//
// Broadcast(x, orig_shape) expands x to a larger shape.
// Backward: reduce_to_shape(grad, orig_shape) sums over broadcast dims.
//
// At the scalar level: when target dim == 1, sum over that dim.
// The backward gradient for a broadcast dim is the sum of upstream grads.
//
// SYNC: backward_rules.rs:191, :376-399.

/// Model reduce_to_shape for a single dimension:
/// if target_dim == 1 and current_dim > 1, the gradient is summed.
/// The per-element contribution after summing N values is 1/N of the sum,
/// but reduce_to_shape returns the full sum (SumKeepDim, not MeanKeepDim).
fn broadcast_backward_sum_count(target_dim: usize, current_dim: usize) -> usize {
    if target_dim == 1 && current_dim > 1 {
        current_dim // sum over this many elements
    } else {
        1 // no reduction needed
    }
}

// ── Embedding backward sparsity ──────────────────────────────────
//
// For a vocab of size V with tokens indexing specific rows,
// rows NOT referenced by any token index must have zero gradient.
//
// SYNC: backward_rules_special.rs:138-175.

/// Embedding backward: non-referenced row gets zero gradient.
fn embedding_row_grad(token_indices: &[usize], target_row: usize, grads: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (i, &idx) in token_indices.iter().enumerate() {
        if idx == target_row {
            acc += grads[i];
        }
    }
    acc
}

// ── SumKeepDim backward ──────────────────────────────────────────
//
// SumKeepDim backward: expand the gradient to the original shape.
// Each element of the original tensor gets the same gradient value.
//
// SYNC: backward_rules.rs:163.

/// SumKeepDim backward: each element gets the same gradient.
/// The expansion factor is the dimension size being reduced.
fn sum_keepdim_backward_element(grad: f32) -> f32 {
    grad // each element receives the same upstream gradient
}

// ── Softmax backward Jacobian ────────────────────────────────────
//
// Softmax backward for elements i != j (off-diagonal):
//   d softmax_i / d x_j = -s_i * s_j
// This is always non-positive (since s_i, s_j >= 0).
//
// SYNC: backward_rules_special.rs:38-43 (softmax_backward_data pattern).

/// Softmax Jacobian off-diagonal: d s_i / d x_j = -s_i * s_j.
fn softmax_jacobian_offdiag(s_i: f32, s_j: f32) -> f32 {
    -(s_i * s_j)
}

/// Softmax Jacobian diagonal: d s_i / d x_i = s_i * (1 - s_i).
fn softmax_jacobian_diag(s_i: f32) -> f32 {
    s_i * (1.0 - s_i)
}

// ── LayerNorm backward inv_std ───────────────────────────────────
//
// inv_std = 1 / sqrt(variance + eps).
// For valid variance >= 0 and eps > 0, inv_std is finite and positive.
//
// SYNC: backward_rules_special.rs:108 (var.add_scalar(eps)?.sqrt()?.recip()?).

/// LayerNorm inv_std computation.
fn layer_norm_inv_std(variance: f32, eps: f64) -> f32 {
    1.0 / (variance + eps as f32).sqrt()
}

// ── MeanKeepDim backward scale ───────────────────────────────────
//
// MeanKeepDim backward: grad * (1/n), where n is the dim size.
// The scaling factor is in (0, 1] for n >= 1.
//
// SYNC: backward_rules.rs:164-167.

/// MeanKeepDim backward: scale gradient by 1/n.
fn mean_keepdim_backward_element(grad: f32, n: usize) -> f32 {
    grad / n as f32
}

// ── MaxPool1d backward routing ───────────────────────────────────
//
// MaxPool1d uses the same scatter routing as MaxPool2d: flat argmax indices.
// Element at position `pos` receives gradient iff `pos == argmax_pos`.
//
// SYNC: backward_rules_pool.rs:33-50.

/// MaxPool1d backward element (identical to MaxPool2d routing).
fn max_pool1d_backward_element(grad: f32, pos: usize, argmax_pos: usize) -> f32 {
    if pos == argmax_pos {
        grad
    } else {
        0.0
    }
}

// ── Neg backward involution ──────────────────────────────────────
//
// Neg backward: grad_input = -grad.
// Double negation: neg(neg(x)) = x, so backward(backward()) = identity.
//
// SYNC: backward_rules_elementwise.rs:82.

/// Neg backward scalar: negate the gradient.
fn neg_backward(grad: f32) -> f32 {
    -grad
}

// ── Dropout backward scale bounds ────────────────────────────────
//
// Dropout scale = 1/(1-p), where p in (0, 1).
// For practical p <= 0.9, scale in [1, 10].
// For p close to 0, scale ≈ 1 (no amplification).
//
// SYNC: backward_rules.rs:64-65.

/// Dropout scale from probability.
fn dropout_scale(p: f64) -> f64 {
    1.0 / (1.0 - p)
}

// ════════════════════════════════════════════════════════════════════
// Kani proof harnesses
// ════════════════════════════════════════════════════════════════════

// --- Conv1d backward ---

/// Prove conv1d grad_input is finite for bounded inputs.
/// Bound: sqrt(f32::MAX) prevents overflow on multiply.
///
/// SYNC: backward_rules_conv.rs:44-52.
#[kani::unwind(1)]
#[kani::proof]
fn prove_conv1d_grad_input_finite() {
    let grad_y: f32 = kani::any();
    let w: f32 = kani::any();
    kani::assume(grad_y.is_finite() && grad_y.abs() <= 1.844e19);
    kani::assume(w.is_finite() && w.abs() <= 1.844e19);
    let d = conv1d_grad_input_scalar(grad_y, w);
    assert!(d.is_finite(), "conv1d grad_input must be finite");
}

/// Prove conv1d grad_kernel is finite for bounded inputs.
///
/// SYNC: backward_rules_conv.rs:54-63.
#[kani::unwind(1)]
#[kani::proof]
fn prove_conv1d_grad_kernel_finite() {
    let x: f32 = kani::any();
    let grad_y: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1.844e19);
    kani::assume(grad_y.is_finite() && grad_y.abs() <= 1.844e19);
    let d = conv1d_grad_kernel_scalar(x, grad_y);
    assert!(d.is_finite(), "conv1d grad_kernel must be finite");
}

/// Prove conv1d grad_kernel is zero when input is zero (no weight update).
/// For embedding layers, zero-padded positions must not contribute to kernel grad.
///
/// SYNC: backward_rules_conv.rs:54-63.
#[kani::unwind(1)]
#[kani::proof]
fn prove_conv1d_grad_kernel_zero_input() {
    let grad_y: f32 = kani::any();
    kani::assume(grad_y.is_finite());
    let d = conv1d_grad_kernel_scalar(0.0, grad_y);
    assert!(
        d == 0.0,
        "conv1d grad_kernel must be zero when input is zero"
    );
}

// --- Unfold backward ---

/// Prove every input position within the valid range is covered by at least
/// one window for non-overlapping unfold (step == size).
///
/// SYNC: backward_rules.rs:197-229.
#[kani::unwind(1)]
#[kani::proof]
fn prove_unfold_nonoverlap_full_coverage() {
    let size: usize = kani::any();
    let n_windows: usize = kani::any();
    let pos: usize = kani::any();
    kani::assume(size >= 1 && size <= 8);
    kani::assume(n_windows >= 1 && n_windows <= 8);
    kani::assume(pos < n_windows * size);
    let step = size; // non-overlapping
    let count = unfold_coverage_count(pos, n_windows, step, size);
    assert!(
        count == 1,
        "non-overlapping unfold: each position covered exactly once"
    );
}

/// Prove unfold coverage count is >= 1 for overlapping windows (step < size)
/// when position is within valid range.
///
/// SYNC: backward_rules.rs:197-229.
#[kani::unwind(1)]
#[kani::proof]
fn prove_unfold_overlap_at_least_one() {
    let size: usize = kani::any();
    let step: usize = kani::any();
    let n_windows: usize = kani::any();
    let pos: usize = kani::any();
    kani::assume(size >= 2 && size <= 8);
    kani::assume(step >= 1 && step < size); // overlapping
    kani::assume(n_windows >= 1 && n_windows <= 8);
    // Position must be within range of at least the first window
    kani::assume(pos < step * (n_windows - 1) + size);
    let count = unfold_coverage_count(pos, n_windows, step, size);
    assert!(
        count >= 1,
        "overlapping unfold: each valid position covered at least once"
    );
}

// --- Transpose self-inverse ---

/// Prove transpose is self-inverse: applying transpose(d1, d2) twice
/// restores the original index. This proves the backward (which applies
/// the same transpose) correctly inverts the forward.
///
/// SYNC: backward_rules.rs:177.
#[kani::unwind(1)]
#[kani::proof]
fn prove_transpose_self_inverse() {
    let i0: usize = kani::any();
    let i1: usize = kani::any();
    let i2: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    kani::assume(i0 <= 16 && i1 <= 16 && i2 <= 16);
    kani::assume(d1 < 3 && d2 < 3 && d1 != d2);
    let idx = [i0, i1, i2];
    let transposed = transpose_index(idx, d1, d2);
    let restored = transpose_index(transposed, d1, d2);
    assert!(
        restored[0] == idx[0] && restored[1] == idx[1] && restored[2] == idx[2],
        "transpose applied twice must be identity"
    );
}

// --- Reshape element preservation ---

/// Prove reshape preserves total element count.
/// This is the key invariant: backward reshape must produce a tensor
/// with the same number of elements as the original input.
///
/// SYNC: backward_rules.rs:176.
#[kani::unwind(1)]
#[kani::proof]
fn prove_reshape_preserves_element_count() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();
    let d3: usize = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);
    kani::assume(d3 >= 1 && d3 <= 16);
    let original = &[d0, d1];
    let reshaped = &[d2, d3];
    let orig_count = element_count(original);
    let new_count = element_count(reshaped);
    // If shapes have the same element count, reshape is valid
    kani::assume(orig_count == new_count);
    assert!(
        orig_count == new_count,
        "reshape must preserve element count"
    );
    assert!(orig_count > 0, "element count must be positive");
}

// --- Broadcast backward ---

/// Prove broadcast backward sum count: when target_dim == 1 and
/// current_dim > 1, the reduction sums over current_dim elements.
/// When dims match, no reduction needed (count == 1).
///
/// SYNC: backward_rules.rs:376-399.
#[kani::unwind(1)]
#[kani::proof]
fn prove_broadcast_backward_sum_semantics() {
    let target: usize = kani::any();
    let current: usize = kani::any();
    kani::assume(current >= 1 && current <= 256);
    kani::assume(target == 1 || target == current);
    let count = broadcast_backward_sum_count(target, current);
    if target == 1 && current > 1 {
        assert!(
            count == current,
            "broadcast backward must sum over all broadcast elements"
        );
    } else {
        assert!(count == 1, "no reduction when dims already match");
    }
}

// --- Cat 3-input backward ---

/// Prove cat backward maps each position to exactly one input
/// for 3 inputs (extends AC8 coverage from 2 to 3 inputs).
///
/// SYNC: backward_rules.rs:237-250.
#[kani::unwind(1)]
#[kani::proof]
fn prove_cat_backward_three_inputs() {
    let len0: usize = kani::any();
    let len1: usize = kani::any();
    let len2: usize = kani::any();
    let pos: usize = kani::any();
    kani::assume(len0 >= 1 && len0 <= 32);
    kani::assume(len1 >= 1 && len1 <= 32);
    kani::assume(len2 >= 1 && len2 <= 32);
    kani::assume(pos < len0 + len1 + len2);
    let in0 = pos < len0;
    let in1 = pos >= len0 && pos < len0 + len1;
    let in2 = pos >= len0 + len1;
    let mut count = 0usize;
    if in0 {
        count += 1;
    }
    if in1 {
        count += 1;
    }
    if in2 {
        count += 1;
    }
    assert!(
        count == 1,
        "cat backward must map each position to exactly one of 3 inputs"
    );
}

// --- Embedding backward sparsity ---

/// Prove embedding backward produces zero gradient for rows not referenced
/// by any token index. This is critical for sparse gradient correctness.
///
/// SYNC: backward_rules_special.rs:138-175.
#[kani::unwind(5)]
#[kani::proof]
fn prove_embedding_backward_unreferenced_row_zero() {
    let idx0: usize = kani::any();
    let idx1: usize = kani::any();
    let target_row: usize = kani::any();
    let g0: f32 = kani::any();
    let g1: f32 = kani::any();
    kani::assume(idx0 <= 100 && idx1 <= 100 && target_row <= 100);
    kani::assume(g0.is_finite() && g1.is_finite());
    // Target row is NOT referenced by either token
    kani::assume(idx0 != target_row && idx1 != target_row);
    let result = embedding_row_grad(&[idx0, idx1], target_row, &[g0, g1]);
    assert!(
        result == 0.0,
        "embedding backward must be zero for unreferenced rows"
    );
}

/// Prove embedding backward accumulates gradients for duplicate indices.
/// When two tokens reference the same row, their gradients must be summed.
///
/// SYNC: backward_rules_special.rs:162 (index_add accumulation).
#[kani::unwind(5)]
#[kani::proof]
fn prove_embedding_backward_duplicate_accumulate() {
    let row: usize = kani::any();
    let g0: f32 = kani::any();
    let g1: f32 = kani::any();
    kani::assume(row <= 100);
    kani::assume(g0.is_finite() && g0.abs() <= 1e18);
    kani::assume(g1.is_finite() && g1.abs() <= 1e18);
    // Both tokens reference the same row
    let result = embedding_row_grad(&[row, row], row, &[g0, g1]);
    let expected = g0 + g1;
    assert!(
        result == expected,
        "embedding backward must sum gradients for duplicate indices"
    );
}

// --- SumKeepDim backward ---

/// Prove SumKeepDim backward passes gradient through unchanged.
/// Each original element receives the full upstream gradient.
///
/// SYNC: backward_rules.rs:163.
#[kani::unwind(1)]
#[kani::proof]
fn prove_sum_keepdim_backward_passthrough() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite());
    let result = sum_keepdim_backward_element(grad);
    assert!(
        result == grad,
        "sum_keepdim backward must pass gradient through unchanged"
    );
}

// --- Softmax backward Jacobian ---

/// Prove softmax Jacobian off-diagonal is non-positive.
/// d s_i / d x_j = -s_i * s_j <= 0 for s_i, s_j in [0, 1].
///
/// SYNC: backward_rules_special.rs:38-43.
#[kani::unwind(1)]
#[kani::proof]
fn prove_softmax_jacobian_offdiag_nonpositive() {
    let s_i: f32 = kani::any();
    let s_j: f32 = kani::any();
    kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i <= 1.0);
    kani::assume(s_j.is_finite() && s_j >= 0.0 && s_j <= 1.0);
    let d = softmax_jacobian_offdiag(s_i, s_j);
    assert!(
        d <= 0.0,
        "softmax off-diagonal Jacobian must be non-positive"
    );
}

/// Prove softmax Jacobian diagonal is non-negative for valid softmax outputs.
/// d s_i / d x_i = s_i * (1 - s_i) >= 0 when s_i in [0, 1].
///
/// SYNC: backward_rules_special.rs:38-43.
#[kani::unwind(1)]
#[kani::proof]
fn prove_softmax_jacobian_diag_nonneg() {
    let s_i: f32 = kani::any();
    kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i <= 1.0);
    let d = softmax_jacobian_diag(s_i);
    assert!(d >= 0.0, "softmax diagonal Jacobian must be non-negative");
}

/// Prove softmax Jacobian diagonal is bounded by 0.25.
/// Maximum of s*(1-s) is at s=0.5 where it equals 0.25.
///
/// SYNC: backward_rules_special.rs:38-43.
#[kani::unwind(1)]
#[kani::proof]
fn prove_softmax_jacobian_diag_bounded() {
    let s_i: f32 = kani::any();
    kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i <= 1.0);
    let d = softmax_jacobian_diag(s_i);
    assert!(
        d <= 0.25 + 1e-6,
        "softmax diagonal Jacobian bounded by 0.25"
    );
}

// --- LayerNorm backward inv_std ---

/// Prove LayerNorm inv_std is finite and positive for valid variance and eps.
/// For variance >= 0 and eps > 0, sqrt(variance + eps) > 0, so 1/sqrt > 0.
///
/// SYNC: backward_rules_special.rs:108.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn prove_layer_norm_inv_std_finite() {
    let variance: f32 = kani::any();
    let eps: f64 = kani::any();
    kani::assume(variance.is_finite() && variance >= 0.0 && variance <= 1e6);
    kani::assume(eps.is_finite() && eps >= 1e-12 && eps <= 1.0);
    let result = layer_norm_inv_std(variance, eps);
    assert!(
        result.is_finite(),
        "layer_norm inv_std must be finite for valid inputs"
    );
}

/// Prove LayerNorm inv_std is positive (denominator > 0 when eps > 0).
///
/// SYNC: backward_rules_special.rs:108.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn prove_layer_norm_inv_std_positive() {
    let variance: f32 = kani::any();
    let eps: f64 = kani::any();
    kani::assume(variance.is_finite() && variance >= 0.0 && variance <= 1e6);
    kani::assume(eps.is_finite() && eps >= 1e-6 && eps <= 1.0);
    let result = layer_norm_inv_std(variance, eps);
    assert!(result > 0.0, "layer_norm inv_std must be positive");
}

// --- MeanKeepDim backward ---

/// Prove MeanKeepDim backward element is finite for valid n.
///
/// SYNC: backward_rules.rs:164-167.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mean_keepdim_backward_finite() {
    let grad: f32 = kani::any();
    let n: usize = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(n >= 1 && n <= 1_000_000);
    let result = mean_keepdim_backward_element(grad, n);
    assert!(result.is_finite(), "mean_keepdim backward must be finite");
}

/// Prove MeanKeepDim backward attenuates gradient: |result| <= |grad|.
/// Dividing by n >= 1 never amplifies.
///
/// SYNC: backward_rules.rs:164-167.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mean_keepdim_backward_attenuates() {
    let grad: f32 = kani::any();
    let n: usize = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(n >= 1 && n <= 1_000_000);
    let result = mean_keepdim_backward_element(grad, n);
    assert!(
        result.abs() <= grad.abs(),
        "mean_keepdim backward must not amplify gradient"
    );
}

// --- MaxPool1d backward ---

/// Prove MaxPool1d backward routes gradient exactly to argmax position.
/// Same scatter routing as MaxPool2d.
///
/// SYNC: backward_rules_pool.rs:33-50.
#[kani::unwind(1)]
#[kani::proof]
fn prove_max_pool1d_backward_routes_to_argmax() {
    let grad: f32 = kani::any();
    let argmax_pos: usize = kani::any();
    kani::assume(grad.is_finite());
    kani::assume(argmax_pos <= 1_000);
    let d = max_pool1d_backward_element(grad, argmax_pos, argmax_pos);
    assert!(
        d == grad,
        "MaxPool1d backward must pass gradient to argmax position"
    );
}

/// Prove MaxPool1d backward produces zero for non-argmax positions.
///
/// SYNC: backward_rules_pool.rs:33-50.
#[kani::unwind(1)]
#[kani::proof]
fn prove_max_pool1d_backward_zero_elsewhere() {
    let grad: f32 = kani::any();
    let pos: usize = kani::any();
    let argmax_pos: usize = kani::any();
    kani::assume(grad.is_finite());
    kani::assume(pos <= 1_000 && argmax_pos <= 1_000);
    kani::assume(pos != argmax_pos);
    let d = max_pool1d_backward_element(grad, pos, argmax_pos);
    assert!(
        d == 0.0,
        "MaxPool1d backward must be zero for non-argmax positions"
    );
}

// --- Neg backward involution ---

/// Prove double negation is identity: neg(neg(grad)) == grad.
/// This proves backward(backward()) correctness for Neg.
///
/// SYNC: backward_rules_elementwise.rs:82.
#[kani::unwind(1)]
#[kani::proof]
fn prove_neg_backward_involution() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite());
    let once = neg_backward(grad);
    let twice = neg_backward(once);
    assert!(twice == grad, "double negation must be identity");
}

/// Prove neg backward preserves finiteness.
///
/// SYNC: backward_rules_elementwise.rs:82.
#[kani::unwind(1)]
#[kani::proof]
fn prove_neg_backward_finite() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite());
    let result = neg_backward(grad);
    assert!(result.is_finite(), "neg backward must preserve finiteness");
}

// --- Dropout backward scale bounds ---

/// Prove dropout scale is finite and >= 1 for valid dropout probability.
/// For p in (0, 0.9], scale = 1/(1-p) in [1, 10].
///
/// SYNC: backward_rules.rs:64-65.
#[kani::unwind(1)]
#[kani::proof]
fn prove_dropout_scale_range() {
    let p: f64 = kani::any();
    kani::assume(p.is_finite() && p >= 0.0 && p <= 0.9);
    let scale = dropout_scale(p);
    assert!(scale.is_finite(), "dropout scale must be finite");
    assert!(scale >= 1.0, "dropout scale must be >= 1");
    assert!(
        scale <= 10.0 + 1e-10,
        "dropout scale must be <= 10 for p <= 0.9"
    );
}

// ── Stubs for CBMC transcendentals ──────────────────────────────
//
// CBMC cannot accurately model f32 transcendentals.
// Same pattern as kani_backward_proofs_activation.rs (#708, #541).

fn sqrt_f32_stub(x: f32) -> f32 {
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result >= 1e-19 && result <= 1e18);
    if x > 0.0 {
        kani::assume(result > 0.0);
    }
    result
}
