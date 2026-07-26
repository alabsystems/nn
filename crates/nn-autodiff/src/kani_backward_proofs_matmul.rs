// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for MatMul, Maximum/Minimum, LogSoftmax, and Narrow
//! backward rules.
//!
//! AC1: MatMul backward (scalar 1×1 matmul derivative)
//! AC4: Maximum/Minimum backward (subgradient with ge/lt masks)
//! AC5: LogSoftmax backward (scalar formula)
//! AC6: Narrow backward (zero-padding structure)
//! AC3: AvgPool2d backward (conv2d-based counting formula)
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.
//!
//! Re: #1603 (backward rule Kani proof coverage).

// ── MatMul scalar backward ──────────────────────────────────────────────
//
// For scalar (1×1) matmul: f(a, b) = a * b
//   grad_a = grad * b^T = grad * b   (for scalars, transpose is identity)
//   grad_b = a^T * grad = a * grad   (for scalars, transpose is identity)
//
// SYNC: backward_rules.rs:137-152 at the scalar level.

/// MatMul backward grad_a for scalar: grad_a = grad * b.
fn matmul_grad_a(grad: f32, b: f32) -> f32 {
    grad * b
}

/// MatMul backward grad_b for scalar: grad_b = a * grad.
fn matmul_grad_b(a: f32, grad: f32) -> f32 {
    a * grad
}

// ── Maximum/Minimum scalar backward ─────────────────────────────────────
//
// Maximum(a, b) backward uses subgradient:
//   grad_a = grad if a >= b, else 0
//   grad_b = grad if b > a, else 0
//
// Minimum(a, b) backward:
//   grad_a = grad if a <= b, else 0
//   grad_b = grad if b < a, else 0
//
// Tie-breaking: a gets the gradient when a == b.
// SYNC: backward_rules.rs:230-248 (Maximum), :249-266 (Minimum).

/// Maximum backward grad_a: grad if a >= b, else 0.
fn maximum_grad_a(a: f32, b: f32, grad: f32) -> f32 {
    if a >= b {
        grad
    } else {
        0.0
    }
}

/// Maximum backward grad_b: grad if b > a, else 0.
fn maximum_grad_b(a: f32, b: f32, grad: f32) -> f32 {
    if b > a {
        grad
    } else {
        0.0
    }
}

/// Minimum backward grad_a: grad if a <= b, else 0.
fn minimum_grad_a(a: f32, b: f32, grad: f32) -> f32 {
    if a <= b {
        grad
    } else {
        0.0
    }
}

/// Minimum backward grad_b: grad if b < a, else 0.
fn minimum_grad_b(a: f32, b: f32, grad: f32) -> f32 {
    if b < a {
        grad
    } else {
        0.0
    }
}

// ── LogSoftmax scalar backward ──────────────────────────────────────────
//
// For a 2-element softmax vector [s0, s1] with log_softmax output,
// the backward for element i is:
//   grad_input[i] = grad[i] - softmax[i] * sum(grad)
//
// SYNC: backward_rules.rs:272-283.

/// LogSoftmax backward for a single element.
fn log_softmax_backward_element(grad_i: f32, s_i: f32, grad_sum: f32) -> f32 {
    grad_i - s_i * grad_sum
}

// ── AvgPool2d backward scalar ───────────────────────────────────────────
//
// For AvgPool2d with a uniform window (no padding effects):
//   grad_input[i] = grad_output[window] / count
// The counting formula uses conv2d(ones) to compute per-position valid counts.
//
// For a scalar element in a uniform window:
//   backward contribution = upstream_grad / window_count
//
// SYNC: backward_rules_pool.rs:69-130 (AvgPool2d backward).

/// AvgPool2d backward contribution for a single element.
fn avgpool2d_backward_scalar(grad: f32, window_count: usize) -> f32 {
    grad / window_count as f32
}

// ── Narrow backward zero-padding structure ──────────────────────────────
//
// Narrow slices [start..start+len] from dim. Backward zero-pads back
// to original size. The gradient is placed at offset `start` with zeros
// before and after.
//
// SYNC: backward_rules.rs:175-185.

/// Narrow backward: element at position `pos` in the original tensor.
/// Returns grad if pos is in [start, start+len), else 0.
fn narrow_backward_element(pos: usize, start: usize, len: usize, grad: f32) -> f32 {
    if pos >= start && pos < start + len {
        grad
    } else {
        0.0
    }
}

// ── Kani proof harnesses ────────────────────────────────────────────────

// --- AC1: MatMul backward ---

/// Prove matmul grad_a is finite for finite inputs.
#[kani::unwind(1)]
#[kani::proof]
fn matmul_grad_a_finite() {
    let grad: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e4);
    kani::assume(b.is_finite() && b.abs() <= 1e4);
    let d = matmul_grad_a(grad, b);
    assert!(d.is_finite(), "matmul grad_a must be finite");
}

/// Prove matmul grad_b is finite for finite inputs.
#[kani::unwind(1)]
#[kani::proof]
fn matmul_grad_b_finite() {
    let a: f32 = kani::any();
    let grad: f32 = kani::any();
    kani::assume(a.is_finite() && a.abs() <= 1e4);
    kani::assume(grad.is_finite() && grad.abs() <= 1e4);
    let d = matmul_grad_b(a, grad);
    assert!(d.is_finite(), "matmul grad_b must be finite");
}

// Tautological harnesses removed (#1614 AC1):
// - matmul_grad_a_equals_b: proved 1.0 * b == b
// - matmul_grad_b_equals_a: proved a * 1.0 == a

// --- AC4: Maximum/Minimum backward ---

/// Prove maximum backward: exactly one of a or b gets the gradient.
/// The sum of grad_a + grad_b == grad (gradient conservation).
#[kani::unwind(1)]
#[kani::proof]
fn maximum_backward_gradient_conservation() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let grad: f32 = kani::any();
    kani::assume(a.is_finite() && a.abs() <= 1e4);
    kani::assume(b.is_finite() && b.abs() <= 1e4);
    kani::assume(grad.is_finite() && grad.abs() <= 1e4);
    // NaN comparison: a != b guaranteed by finiteness
    let ga = maximum_grad_a(a, b, grad);
    let gb = maximum_grad_b(a, b, grad);
    assert!(ga.is_finite(), "maximum grad_a must be finite");
    assert!(gb.is_finite(), "maximum grad_b must be finite");
    // When a != b, exactly one gets the gradient.
    // When a == b, a gets it (tie-breaking rule).
    assert!(ga + gb == grad, "maximum backward must conserve gradient");
}

/// Prove minimum backward: gradient conservation.
#[kani::unwind(1)]
#[kani::proof]
fn minimum_backward_gradient_conservation() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let grad: f32 = kani::any();
    kani::assume(a.is_finite() && a.abs() <= 1e4);
    kani::assume(b.is_finite() && b.abs() <= 1e4);
    kani::assume(grad.is_finite() && grad.abs() <= 1e4);
    let ga = minimum_grad_a(a, b, grad);
    let gb = minimum_grad_b(a, b, grad);
    assert!(ga.is_finite(), "minimum grad_a must be finite");
    assert!(gb.is_finite(), "minimum grad_b must be finite");
    assert!(ga + gb == grad, "minimum backward must conserve gradient");
}

/// Prove maximum grad_a: when a > b, grad_a == grad (a is the max).
#[kani::unwind(1)]
#[kani::proof]
fn maximum_grad_a_when_greater() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let grad: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && grad.is_finite());
    kani::assume(a > b);
    let ga = maximum_grad_a(a, b, grad);
    assert!(ga == grad, "when a > b, maximum grad_a must equal grad");
}

/// Prove minimum grad_a: when a < b, grad_a == grad (a is the min).
#[kani::unwind(1)]
#[kani::proof]
fn minimum_grad_a_when_lesser() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let grad: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && grad.is_finite());
    kani::assume(a < b);
    let ga = minimum_grad_a(a, b, grad);
    assert!(ga == grad, "when a < b, minimum grad_a must equal grad");
}

// --- AC5: LogSoftmax backward ---

/// Prove log_softmax backward is finite for valid softmax outputs.
#[kani::unwind(1)]
#[kani::proof]
fn log_softmax_backward_finite() {
    let grad_i: f32 = kani::any();
    let s_i: f32 = kani::any();
    let grad_sum: f32 = kani::any();
    kani::assume(grad_i.is_finite() && grad_i.abs() <= 1e6);
    kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i <= 1.0);
    kani::assume(grad_sum.is_finite() && grad_sum.abs() <= 1e6);
    let d = log_softmax_backward_element(grad_i, s_i, grad_sum);
    assert!(d.is_finite(), "log_softmax backward must be finite");
}

/// Prove: when s_i = 1.0 (only element in softmax), grad = grad_i - grad_sum.
/// This is the special case of a single-element softmax.
#[kani::unwind(1)]
#[kani::proof]
fn log_softmax_backward_single_element() {
    let grad_i: f32 = kani::any();
    kani::assume(grad_i.is_finite() && grad_i.abs() <= 1e6);
    // For single-element softmax: s_i = 1.0, grad_sum = grad_i
    let d = log_softmax_backward_element(grad_i, 1.0, grad_i);
    // grad_i - 1.0 * grad_i = 0.0
    assert!(d == 0.0, "single-element log_softmax backward must be zero");
}

/// Prove: when grad is uniform (all elements equal), log_softmax backward
/// produces zero for each element (sum of softmax = 1, so s_i * grad_sum = grad_i).
/// Note: f32 rounding makes (1/n)*(n*g) != g for non-power-of-2 n. The error
/// scales as n * eps * |g|. We bound |g| <= 1.0 so absolute tolerance 1e-4
/// is sufficient for n <= 100.
#[kani::unwind(1)]
#[kani::proof]
fn log_softmax_backward_uniform_grad() {
    let g: f32 = kani::any();
    let n: usize = kani::any();
    kani::assume(g.is_finite() && g.abs() <= 1.0);
    kani::assume(n >= 1 && n <= 100);
    // For uniform softmax: s_i = 1/n, grad_sum = n * g
    let s_i = 1.0 / n as f32;
    let grad_sum = n as f32 * g;
    let d = log_softmax_backward_element(g, s_i, grad_sum);
    // g - (1/n) * (n*g) = g - g = 0
    // Max error with |g|<=1, n<=100: 100 * 1.19e-7 * 1.0 ~ 1.2e-5 << 1e-4
    assert!(
        d.abs() <= 1e-4,
        "uniform grad log_softmax backward should be near zero"
    );
}

// --- AC3: AvgPool2d backward ---

/// Prove avgpool2d backward is finite for valid window counts.
#[kani::unwind(1)]
#[kani::proof]
fn avgpool2d_backward_scalar_finite() {
    let grad: f32 = kani::any();
    let count: usize = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(count >= 1 && count <= 10000);
    let d = avgpool2d_backward_scalar(grad, count);
    assert!(d.is_finite(), "avgpool2d backward must be finite");
}

/// Prove avgpool2d backward magnitude is bounded by grad magnitude.
/// Since count >= 1, |grad / count| <= |grad|.
#[kani::unwind(1)]
#[kani::proof]
fn avgpool2d_backward_bounded() {
    let grad: f32 = kani::any();
    let count: usize = kani::any();
    kani::assume(grad.is_finite() && grad.abs() <= 1e6);
    kani::assume(count >= 1 && count <= 10000);
    let d = avgpool2d_backward_scalar(grad, count);
    assert!(
        d.abs() <= grad.abs() + 1e-7,
        "avgpool2d backward must not amplify gradient"
    );
}

// Tautological harnesses removed (#1614 AC1):
// - avgpool2d_backward_count_one_passthrough: proved grad / 1.0 == grad

// --- AC6: Narrow backward ---

/// Prove narrow backward: elements inside [start, start+len) get the gradient.
#[kani::unwind(1)]
#[kani::proof]
fn narrow_backward_inside_range() {
    let pos: usize = kani::any();
    let start: usize = kani::any();
    let len: usize = kani::any();
    let grad: f32 = kani::any();
    kani::assume(start <= 1000 && len >= 1 && len <= 1000);
    kani::assume(pos >= start && pos < start + len);
    kani::assume(grad.is_finite());
    let d = narrow_backward_element(pos, start, len, grad);
    assert!(
        d == grad,
        "narrow backward must pass gradient for in-range elements"
    );
}

/// Prove narrow backward: elements outside [start, start+len) get zero.
#[kani::unwind(1)]
#[kani::proof]
fn narrow_backward_outside_range() {
    let pos: usize = kani::any();
    let start: usize = kani::any();
    let len: usize = kani::any();
    let grad: f32 = kani::any();
    kani::assume(start <= 1000 && len >= 1 && len <= 1000);
    kani::assume(pos < start || pos >= start + len);
    kani::assume(grad.is_finite());
    let d = narrow_backward_element(pos, start, len, grad);
    assert!(
        d == 0.0,
        "narrow backward must return zero for out-of-range elements"
    );
}

/// Prove narrow backward preserves dtype (finite in → finite or zero out).
#[kani::unwind(1)]
#[kani::proof]
fn narrow_backward_finiteness() {
    let pos: usize = kani::any();
    let start: usize = kani::any();
    let len: usize = kani::any();
    let grad: f32 = kani::any();
    kani::assume(start <= 1000 && len >= 1 && len <= 1000 && pos <= 2000);
    kani::assume(grad.is_finite());
    let d = narrow_backward_element(pos, start, len, grad);
    assert!(d.is_finite(), "narrow backward must produce finite output");
}

// --- AC2: Conv1d backward (scalar FIR cross-correlation) ---
//
// Scalar conv1d with kernel_size=1, stride=1, padding=0, dilation=1:
//   y[0] = sum_c(x[c] * w[c])   (dot product per output channel)
//
// Backward:
//   grad_x[c] = grad_y * w[c]             (transpose convolution)
//   grad_w[c] = x[c] * grad_y             (cross-correlation)
//
// SYNC: backward_rules_conv.rs:22-67 at the scalar (C=1, K=1) level.

/// Conv1d backward grad_input for scalar FIR: grad_x = grad_y * w.
fn conv1d_grad_input_scalar(grad_y: f32, w: f32) -> f32 {
    grad_y * w
}

/// Conv1d backward grad_kernel for scalar FIR: grad_w = x * grad_y.
fn conv1d_grad_kernel_scalar(x: f32, grad_y: f32) -> f32 {
    x * grad_y
}

/// Prove conv1d grad_input is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn conv1d_grad_input_finite() {
    let grad_y: f32 = kani::any();
    let w: f32 = kani::any();
    kani::assume(grad_y.is_finite() && grad_y.abs() <= 1e4);
    kani::assume(w.is_finite() && w.abs() <= 1e4);
    let d = conv1d_grad_input_scalar(grad_y, w);
    assert!(d.is_finite(), "conv1d grad_input must be finite");
}

/// Prove conv1d grad_kernel is finite for bounded inputs.
#[kani::unwind(1)]
#[kani::proof]
fn conv1d_grad_kernel_finite() {
    let x: f32 = kani::any();
    let grad_y: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e4);
    kani::assume(grad_y.is_finite() && grad_y.abs() <= 1e4);
    let d = conv1d_grad_kernel_scalar(x, grad_y);
    assert!(d.is_finite(), "conv1d grad_kernel must be finite");
}

// Tautological harnesses removed (#1614 AC1):
// - conv1d_unit_kernel_passthrough: proved grad_y * 1.0 == grad_y
// - conv1d_zero_input_no_kernel_grad: proved 0.0 * grad_y == 0.0
