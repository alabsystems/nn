// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for shape ops, Cat/Stack, Conv2d, ConvTranspose1d,
//! and Embedding backward rules.
//!
//! AC7:  Shape ops backward (Reshape, Transpose, Permute — structural identity)
//! AC8:  Cat/Stack backward (split/squeeze along dim)
//! AC9:  Conv2d backward (im2col-based scalar derivative)
//! AC10: ConvTranspose1d backward (conv1d adjoint scalar derivative)
//! AC11: Embedding backward (scatter_add-based)
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.
//!
//! Re: #1603 (backward rule Kani proof coverage).

// ── AC7: Shape ops backward ──────────────────────────────────────────────
//
// Reshape backward:   grad_input = grad.reshape(original_shape)
// Transpose backward: grad_input = grad.transpose(d1, d2)  (self-inverse)
// Permute backward:   grad_input = grad.permute(inverse_perm)
//
// All three are structural identity ops on the data — they just reindex.
// At the scalar level, each element of grad passes through unchanged.
//
// SYNC: backward_rules.rs:173 (Reshape), :174 (Transpose), :190 (Permute).

// ── AC8: Cat/Stack backward ──────────────────────────────────────────────
//
// Cat backward: split gradient along cat dim into per-input slices.
//   For the i-th input with length `len_i` along `dim`, the gradient
//   is `grad.narrow(dim, offset_i, len_i)`.
//
// Stack backward: narrow + squeeze. For the i-th input:
//   grad_i = grad.narrow(dim, i, 1).squeeze(dim)
//
// At the scalar level, a gradient element from the concatenated output
// maps to exactly one input tensor's gradient at the corresponding position.
//
// SYNC: backward_rules.rs:196-210 (Cat), :213-225 (Stack).

/// Cat backward: element at position `pos` in the concatenated output
/// maps to the input whose range covers that position.
/// Returns `true` if pos falls within [offset, offset + len).
fn cat_backward_maps_to_input(pos: usize, offset: usize, len: usize) -> bool {
    pos >= offset && pos < offset + len
}

// ── AC9: Conv2d backward (scalar cross-correlation derivative) ───────────
//
// Scalar Conv2d with 1×1 kernel, stride=1, padding=0, dilation=1, groups=1:
//   y[0] = sum_c(x[c] * w[c])   (1×1 convolution = channel-wise dot product)
//
// Backward:
//   grad_x[c] = grad_y * w[c]   (conv_transpose2d with 1×1 kernel = multiply)
//   grad_w[c] = x[c] * grad_y   (im2col cross-correlation = multiply for 1×1)
//
// SYNC: backward_rules_conv2d.rs:23-92 at the scalar (1×1 kernel) level.

/// Conv2d backward grad_input for scalar 1×1 kernel: grad_x = grad_y * w.
fn conv2d_grad_input_scalar(grad_y: f32, w: f32) -> f32 {
    grad_y * w
}

/// Conv2d backward grad_kernel for scalar 1×1 kernel: grad_w = x * grad_y.
fn conv2d_grad_kernel_scalar(x: f32, grad_y: f32) -> f32 {
    x * grad_y
}

// ── AC10: ConvTranspose1d backward (conv1d adjoint scalar derivative) ────
//
// ConvTranspose1d backward:
//   grad_input = conv1d(grad_output, kernel)
//   grad_kernel = im2col(grad)^T @ input
//
// At the scalar level (K=1, stride=1, padding=0, dilation=1):
//   grad_input = grad_output * kernel  (scalar conv1d = multiply)
//   grad_kernel = input * grad_output  (cross-correlation = multiply)
//
// This is the dual of Conv1d backward — the formulas are identical at
// the scalar level because conv_transpose1d(scalar) = conv1d(scalar).
//
// SYNC: backward_rules_conv_transpose.rs:22-100.

/// ConvTranspose1d backward grad_input (scalar): grad_x = grad_y * kernel.
fn conv_transpose1d_grad_input_scalar(grad_y: f32, kernel: f32) -> f32 {
    grad_y * kernel
}

/// ConvTranspose1d backward grad_kernel (scalar): grad_w = input * grad_y.
fn conv_transpose1d_grad_kernel_scalar(input: f32, grad_y: f32) -> f32 {
    input * grad_y
}

// ── AC11: Embedding backward (scatter_add-based) ─────────────────────────
//
// Embedding forward: output[i] = weight[index[i]]  (table lookup)
// Embedding backward: grad_weight[j] = sum over {i : index[i] == j} grad[i]
//
// At the scalar level (single token, single embedding dim):
//   grad_weight[j] = grad[0] if index[0] == j, else 0
//
// For multiple occurrences of the same index, gradients accumulate (add).
//
// SYNC: backward_rules_special.rs:139-166.

/// Embedding backward for a single (token, dim) element.
/// Returns grad if the index matches, else 0.
fn embedding_backward_element(grad: f32, token_index: usize, target_row: usize) -> f32 {
    if token_index == target_row {
        grad
    } else {
        0.0
    }
}

/// Embedding backward with accumulation: two tokens mapping to the same row.
/// grad_weight = grad_0 + grad_1 (scatter_add semantics).
// SYNC: backward_rules_special.rs:162 (index_add accumulation)
fn embedding_backward_accumulate(grad_0: f32, grad_1: f32) -> f32 {
    grad_0 + grad_1
}

// ── Kani proof harnesses ────────────────────────────────────────────────

// --- AC7: Shape ops backward ---
//
// Tautological harnesses removed (#1614 AC1):
// - shape_backward_identity: proved identity fn returns input
// - shape_backward_finite: proved identity fn preserves finiteness
// - shape_backward_preserves_sign: proved identity fn preserves sign

// --- AC8: Cat/Stack backward ---

/// Prove cat backward maps each position to exactly one input.
/// For two inputs with lengths len0 and len1, positions [0, len0) map to
/// input 0, and [len0, len0+len1) map to input 1.
#[kani::unwind(1)]
#[kani::proof]
fn cat_backward_position_mapping() {
    let len0: usize = kani::any();
    let len1: usize = kani::any();
    let pos: usize = kani::any();
    kani::assume(len0 >= 1 && len0 <= 100);
    kani::assume(len1 >= 1 && len1 <= 100);
    kani::assume(pos < len0 + len1);
    let in0 = cat_backward_maps_to_input(pos, 0, len0);
    let in1 = cat_backward_maps_to_input(pos, len0, len1);
    // Exactly one input claims each position.
    assert!(
        (in0 && !in1) || (!in0 && in1),
        "cat backward must map each position to exactly one input"
    );
}

/// Prove cat backward: no position maps to two inputs simultaneously.
#[kani::unwind(1)]
#[kani::proof]
fn cat_backward_exclusive() {
    let len0: usize = kani::any();
    let len1: usize = kani::any();
    let pos: usize = kani::any();
    kani::assume(len0 >= 1 && len0 <= 100);
    kani::assume(len1 >= 1 && len1 <= 100);
    kani::assume(pos < len0 + len1);
    let in0 = cat_backward_maps_to_input(pos, 0, len0);
    let in1 = cat_backward_maps_to_input(pos, len0, len1);
    assert!(!(in0 && in1), "cat backward must not map to two inputs");
}

// Tautological harnesses removed (#1614 AC1):
// - stack_backward_identity: proved identity fn returns input

// --- AC9: Conv2d backward ---

/// Prove conv2d grad_input is finite for bounded inputs.
/// Bound: sqrt(f32::MAX) ≈ 1.844e19. At this limit, |a*b| ≤ f32::MAX.
/// Prior bound (1e4) was trivially true — 30 orders of magnitude below overflow.
#[kani::unwind(1)]
#[kani::proof]
fn conv2d_grad_input_finite() {
    let grad_y: f32 = kani::any();
    let w: f32 = kani::any();
    kani::assume(grad_y.is_finite() && grad_y.abs() <= 1.844e19);
    kani::assume(w.is_finite() && w.abs() <= 1.844e19);
    let d = conv2d_grad_input_scalar(grad_y, w);
    assert!(d.is_finite(), "conv2d grad_input must be finite");
}

/// Prove conv2d grad_kernel is finite for bounded inputs.
/// Bound: sqrt(f32::MAX) ≈ 1.844e19. See conv2d_grad_input_finite.
#[kani::unwind(1)]
#[kani::proof]
fn conv2d_grad_kernel_finite() {
    let x: f32 = kani::any();
    let grad_y: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1.844e19);
    kani::assume(grad_y.is_finite() && grad_y.abs() <= 1.844e19);
    let d = conv2d_grad_kernel_scalar(x, grad_y);
    assert!(d.is_finite(), "conv2d grad_kernel must be finite");
}

// Tautological harnesses removed (#1614 AC1):
// - conv2d_unit_kernel_passthrough: proved grad_y * 1.0 == grad_y
// - conv2d_zero_input_no_kernel_grad: proved 0.0 * grad_y == 0.0

// --- AC10: ConvTranspose1d backward ---

/// Prove conv_transpose1d grad_input is finite for bounded inputs.
/// Bound: sqrt(f32::MAX) ≈ 1.844e19. Prior bound (1e4) was trivially true.
#[kani::unwind(1)]
#[kani::proof]
fn conv_transpose1d_grad_input_finite() {
    let grad_y: f32 = kani::any();
    let kernel: f32 = kani::any();
    kani::assume(grad_y.is_finite() && grad_y.abs() <= 1.844e19);
    kani::assume(kernel.is_finite() && kernel.abs() <= 1.844e19);
    let d = conv_transpose1d_grad_input_scalar(grad_y, kernel);
    assert!(d.is_finite(), "conv_transpose1d grad_input must be finite");
}

/// Prove conv_transpose1d grad_kernel is finite for bounded inputs.
/// Bound: sqrt(f32::MAX) ≈ 1.844e19. Prior bound (1e4) was trivially true.
#[kani::unwind(1)]
#[kani::proof]
fn conv_transpose1d_grad_kernel_finite() {
    let input: f32 = kani::any();
    let grad_y: f32 = kani::any();
    kani::assume(input.is_finite() && input.abs() <= 1.844e19);
    kani::assume(grad_y.is_finite() && grad_y.abs() <= 1.844e19);
    let d = conv_transpose1d_grad_kernel_scalar(input, grad_y);
    assert!(d.is_finite(), "conv_transpose1d grad_kernel must be finite");
}

// Tautological harnesses removed (#1614 AC1):
// - conv_transpose1d_unit_kernel_passthrough: proved grad_y * 1.0 == grad_y
// - conv_transpose1d_zero_input_no_kernel_grad: proved 0.0 * grad_y == 0.0

// --- AC11: Embedding backward ---

// Tautological harnesses removed (#1614 AC1, P1-277):
// - embedding_backward_matching_index: proved if idx==idx { grad } == grad
// - embedding_backward_nonmatching_index: proved if idx!=target { 0.0 } == 0.0 when idx!=target assumed

/// Prove embedding backward accumulation is finite for finite gradients.
/// Bound: f32::MAX/2 ≈ 1.7e38. At this limit, |a+b| ≤ f32::MAX.
/// Prior bound (1e18) was trivially true — 20 orders of magnitude below overflow.
#[kani::unwind(1)]
#[kani::proof]
fn embedding_backward_accumulate_finite() {
    let g0: f32 = kani::any();
    let g1: f32 = kani::any();
    kani::assume(g0.is_finite() && g0.abs() <= 1.7e38);
    kani::assume(g1.is_finite() && g1.abs() <= 1.7e38);
    let d = embedding_backward_accumulate(g0, g1);
    assert!(
        d.is_finite(),
        "embedding backward accumulation must be finite"
    );
}

// Tautological harnesses removed (#1614 AC1):
// - embedding_backward_accumulate_commutative: proved f32 addition is commutative (hardware property)
// - embedding_backward_zero_grad: proved 0.0 + 0.0 == 0.0
