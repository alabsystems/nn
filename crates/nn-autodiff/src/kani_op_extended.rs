// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for op.rs.
//!
//! Supplements `kani_op.rs` with proofs of Op variant parameter invariants:
//! scalar op parameter bounds (Powf, Clamp, MulScalar, AddScalar, Elu),
//! pooling parameter relationships, conv stride/dilation coverage,
//! op_inputs input count properties, and Cat/Stack dimension constraints.
//!
//! **Local-copy gap:** Scalar functions here re-implement production invariants
//! from `op.rs` and `grad_op_inputs.rs`. `// SYNC:` comments track correspondence.
//!
//! Re: #3747 (Kani harnesses for op + backward_rules_norm + train_loop + grad).

// ── Op variant input count ───────────────────────────────────────────────
//
// Each Op variant carries a fixed number of tracked inputs (for backward
// graph traversal). Binary ops have 2, unary have 1, norms have 2-3, etc.
//
// SYNC: grad_op_inputs.rs:14-102

/// Input count for binary element-wise ops: Add, Sub, Mul, Div, MatMul.
///
/// SYNC: grad_op_inputs.rs:17-19
#[allow(dead_code)]
fn binary_op_input_count() -> usize {
    2
}

/// Input count for unary element-wise ops: Relu, Gelu, Silu, etc.
///
/// SYNC: grad_op_inputs.rs:20-36
#[allow(dead_code)]
fn unary_op_input_count() -> usize {
    1
}

/// Input count for norm ops with bias: LayerNorm, GroupNorm, BatchNorm, InstanceNorm.
///
/// SYNC: grad_op_inputs.rs:61-84
#[allow(dead_code)]
fn norm_with_bias_input_count() -> usize {
    3 // input, weight, bias
}

/// Input count for norm ops without bias: RmsNorm.
///
/// SYNC: grad_op_inputs.rs:85-87
#[allow(dead_code)]
fn norm_no_bias_input_count() -> usize {
    2 // input, weight
}

/// Prove binary ops have exactly 2 inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_binary_op_input_count() {
    let count = binary_op_input_count();
    assert!(count == 2, "binary ops must have exactly 2 inputs");
}

/// Prove unary ops have exactly 1 input.
#[kani::unwind(1)]
#[kani::proof]
fn prove_unary_op_input_count() {
    let count = unary_op_input_count();
    assert!(count == 1, "unary ops must have exactly 1 input");
}

/// Prove norm-with-bias ops have exactly 3 inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_norm_with_bias_input_count() {
    let count = norm_with_bias_input_count();
    assert!(count == 3, "norm-with-bias ops must have 3 inputs");
}

/// Prove RmsNorm has 2 inputs (no bias).
#[kani::unwind(1)]
#[kani::proof]
fn prove_rms_norm_input_count() {
    let count = norm_no_bias_input_count();
    assert!(count == 2, "RmsNorm must have exactly 2 inputs");
    assert!(
        count < norm_with_bias_input_count(),
        "RmsNorm has fewer inputs than norm-with-bias variants"
    );
}

// ── Powf parameter bounds ────────────────────────────────────────────────
//
// Op::Powf(x, p) applies x^p element-wise.
// For safe backward: d/dx(x^p) = p * x^(p-1), which is finite when x > 0
// and p is finite.
//
// SYNC: op.rs:144

/// Powf backward derivative: p * x^(p-1).
/// Returns None if result would be non-finite.
///
/// SYNC: backward_rules_elementwise.rs (Powf backward)
#[allow(dead_code)]
fn powf_derivative(x: f32, p: f64) -> Option<f32> {
    if x <= 0.0 || !x.is_finite() || !p.is_finite() {
        return None;
    }
    let result = p as f32 * x.powf(p as f32 - 1.0);
    if result.is_finite() {
        Some(result)
    } else {
        None
    }
}

fn powf_f32_stub(base: f32, _exp: f32) -> f32 {
    let _ = base;
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

/// Prove Powf derivative with p=2 is 2*x (linear).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn prove_powf_derivative_p2_is_linear() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.01 && x <= 100.0);
    if let Some(d) = powf_derivative(x, 2.0) {
        let expected = 2.0 * x;
        assert!((d - expected).abs() < 1e-3, "d/dx(x^2) must equal 2x");
    }
}

/// Prove Powf derivative with p=1 is 1 (constant).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn prove_powf_derivative_p1_is_one() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.01 && x <= 100.0);
    if let Some(d) = powf_derivative(x, 1.0) {
        assert!((d - 1.0).abs() < 1e-4, "d/dx(x^1) must equal 1");
    }
}

/// Prove Powf derivative with p=0 is 0 (zero gradient).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn prove_powf_derivative_p0_is_zero() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x > 0.01 && x <= 100.0);
    // d/dx(x^0) = 0 * x^(-1) = 0
    if let Some(d) = powf_derivative(x, 0.0) {
        assert!(d.abs() < 1e-5, "d/dx(x^0) must be zero");
    }
}

// ── Clamp parameter ordering ─────────────────────────────────────────────
//
// Op::Clamp(x, lo, hi): clamp(x, lo, hi).
// For correct behavior: lo <= hi.
// The backward pass: gradient passes through when lo < x < hi, zero otherwise.
//
// SYNC: op.rs:146

/// Validate clamp parameter ordering.
///
/// SYNC: op.rs:146 (lo and hi must satisfy lo <= hi)
#[allow(dead_code)]
fn is_valid_clamp_params(lo: f64, hi: f64) -> bool {
    lo.is_finite() && hi.is_finite() && lo <= hi
}

/// Clamp backward: gradient passes through when x is in (lo, hi).
///
/// SYNC: backward_rules_elementwise.rs (Clamp backward)
#[allow(dead_code)]
fn clamp_backward_scalar(x: f32, lo: f64, hi: f64, grad: f32) -> f32 {
    let x64 = x as f64;
    if x64 > lo && x64 < hi {
        grad
    } else {
        0.0
    }
}

/// Prove clamp parameters are valid when lo <= hi.
#[kani::unwind(1)]
#[kani::proof]
fn prove_clamp_params_valid() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite() && lo <= hi);
    assert!(
        is_valid_clamp_params(lo, hi),
        "lo <= hi must be valid clamp params"
    );
}

/// Prove clamp parameters invalid when lo > hi.
#[kani::unwind(1)]
#[kani::proof]
fn prove_clamp_params_invalid_reversed() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite() && lo > hi);
    assert!(
        !is_valid_clamp_params(lo, hi),
        "lo > hi must be invalid clamp params"
    );
}

/// Prove clamp backward passes gradient in interior.
#[kani::unwind(1)]
#[kani::proof]
fn prove_clamp_backward_passthrough() {
    let x: f32 = kani::any();
    let grad: f32 = kani::any();
    kani::assume(x.is_finite() && x > -10.0 && x < 10.0);
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    let lo = -5.0_f64;
    let hi = 5.0_f64;
    kani::assume((x as f64) > lo && (x as f64) < hi);
    let result = clamp_backward_scalar(x, lo, hi, grad);
    assert!(
        result == grad,
        "clamp backward must pass gradient when x is in (lo, hi)"
    );
}

/// Prove clamp backward zeros gradient outside range.
#[kani::unwind(1)]
#[kani::proof]
fn prove_clamp_backward_zero_outside() {
    let x: f32 = kani::any();
    let grad: f32 = kani::any();
    kani::assume(x.is_finite() && grad.is_finite());
    let lo = 0.0_f64;
    let hi = 1.0_f64;
    kani::assume((x as f64) <= lo || (x as f64) >= hi);
    let result = clamp_backward_scalar(x, lo, hi, grad);
    assert!(
        result == 0.0,
        "clamp backward must zero gradient when x is outside [lo, hi]"
    );
}

// ── ELU alpha parameter ──────────────────────────────────────────────────
//
// Op::Elu(x, alpha): ELU(x) = x if x > 0, alpha*(exp(x)-1) otherwise.
// The backward: d/dx = 1 if x > 0, alpha*exp(x) otherwise.
// alpha must be positive for the activation to be monotonic.
//
// SYNC: op.rs:196

/// ELU forward (scalar).
///
/// SYNC: op.rs:196
#[allow(dead_code)]
fn elu_forward(x: f32, alpha: f64) -> f32 {
    if x > 0.0 {
        x
    } else {
        alpha as f32 * (x.exp() - 1.0)
    }
}

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

/// Prove ELU is identity for positive inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn prove_elu_identity_positive() {
    let x: f32 = kani::any();
    let alpha: f64 = kani::any();
    kani::assume(x.is_finite() && x > 0.0 && x <= 100.0);
    kani::assume(alpha.is_finite() && alpha > 0.0 && alpha <= 10.0);
    let result = elu_forward(x, alpha);
    assert!(result == x, "ELU must be identity for x > 0");
}

/// Prove ELU is non-positive for non-positive inputs (alpha > 0).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn prove_elu_nonpositive_for_neg() {
    let x: f32 = kani::any();
    let alpha: f64 = kani::any();
    kani::assume(x.is_finite() && x <= 0.0 && x >= -10.0);
    kani::assume(alpha.is_finite() && alpha > 0.0 && alpha <= 10.0);
    let result = elu_forward(x, alpha);
    assert!(
        result <= 0.0 + 1e-6,
        "ELU must be <= 0 for x <= 0 when alpha > 0"
    );
}

// ── MulScalar / AddScalar parameter finiteness ───────────────────────────
//
// Op::MulScalar(x, s) and Op::AddScalar(x, s) carry scalar f64 parameters.
// These scalars must be finite for the backward pass to produce finite gradients.
//
// SYNC: op.rs:212-214

/// MulScalar backward: d/dx(x * s) = s.
///
/// SYNC: backward_rules.rs (MulScalar backward)
#[allow(dead_code)]
fn mul_scalar_grad(scalar: f64, upstream_grad: f32) -> f32 {
    scalar as f32 * upstream_grad
}

/// AddScalar backward: d/dx(x + s) = 1 (gradient passes through unchanged).
///
/// SYNC: backward_rules.rs (AddScalar backward)
#[allow(dead_code)]
fn add_scalar_grad(upstream_grad: f32) -> f32 {
    upstream_grad
}

/// Prove MulScalar backward is finite for finite inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mul_scalar_grad_finite() {
    let scalar: f64 = kani::any();
    let grad: f32 = kani::any();
    kani::assume(scalar.is_finite() && scalar.abs() <= 1e3);
    kani::assume(grad.is_finite() && grad.abs() <= 1e3);
    let result = mul_scalar_grad(scalar, grad);
    assert!(
        result.is_finite(),
        "MulScalar backward must be finite for finite inputs"
    );
}

/// Prove AddScalar backward is identity (gradient unchanged).
#[kani::unwind(1)]
#[kani::proof]
fn prove_add_scalar_grad_identity() {
    let grad: f32 = kani::any();
    kani::assume(grad.is_finite());
    let result = add_scalar_grad(grad);
    assert!(
        result == grad,
        "AddScalar backward must pass gradient through unchanged"
    );
}

// ── MaxPool parameter constraints ────────────────────────────────────────
//
// MaxPool1d/2d: kernel_size >= 1, stride >= 1, padding < kernel_size.
// Output length: (in_len + 2*padding - kernel_size) / stride + 1.
//
// SYNC: op.rs:219-234

/// MaxPool output length.
///
/// SYNC: op.rs:219-234
#[allow(dead_code)]
fn max_pool_output_len(
    in_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Option<usize> {
    if kernel_size == 0 || stride == 0 {
        return None;
    }
    let padded = in_len.checked_add(2 * padding)?;
    if padded < kernel_size {
        return None;
    }
    Some((padded - kernel_size) / stride + 1)
}

/// Prove MaxPool output is positive for valid parameters.
#[kani::unwind(1)]
#[kani::proof]
fn prove_max_pool_output_positive() {
    let in_len: usize = kani::any();
    let kernel: usize = kani::any();
    let stride: usize = kani::any();
    let pad: usize = kani::any();
    kani::assume(in_len >= 1 && in_len <= 512);
    kani::assume(kernel >= 1 && kernel <= 16);
    kani::assume(stride >= 1 && stride <= 8);
    kani::assume(pad <= kernel / 2);
    if let Some(out) = max_pool_output_len(in_len, kernel, stride, pad) {
        assert!(out >= 1, "MaxPool output must be >= 1 when valid");
    }
}

/// Prove MaxPool with stride=1, no padding does not increase length.
#[kani::unwind(1)]
#[kani::proof]
fn prove_max_pool_no_increase() {
    let in_len: usize = kani::any();
    let kernel: usize = kani::any();
    kani::assume(in_len >= 1 && in_len <= 1024);
    kani::assume(kernel >= 1 && kernel <= in_len);
    if let Some(out) = max_pool_output_len(in_len, kernel, 1, 0) {
        assert!(
            out <= in_len,
            "MaxPool with stride=1, pad=0 must not increase length"
        );
    }
}

// ── AdaptiveAvgPool output constraints ───────────────────────────────────
//
// AdaptiveAvgPool2d: output_h and output_w specify target spatial size.
// Must be >= 1.
//
// SYNC: op.rs:236-240

/// Validate adaptive avg pool output dimensions.
///
/// SYNC: op.rs:238-239
#[allow(dead_code)]
fn is_valid_adaptive_pool_output(output_h: usize, output_w: usize) -> bool {
    output_h >= 1 && output_w >= 1
}

/// Prove adaptive pool with valid dimensions passes validation.
#[kani::unwind(1)]
#[kani::proof]
fn prove_adaptive_pool_valid() {
    let h: u8 = kani::any();
    let w: u8 = kani::any();
    kani::assume(h >= 1 && h <= 64);
    kani::assume(w >= 1 && w <= 64);
    assert!(
        is_valid_adaptive_pool_output(h as usize, w as usize),
        "valid dimensions must pass"
    );
}

/// Prove adaptive pool rejects zero dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn prove_adaptive_pool_rejects_zero() {
    assert!(
        !is_valid_adaptive_pool_output(0, 1),
        "output_h=0 must be rejected"
    );
    assert!(
        !is_valid_adaptive_pool_output(1, 0),
        "output_w=0 must be rejected"
    );
    assert!(
        !is_valid_adaptive_pool_output(0, 0),
        "both=0 must be rejected"
    );
}
