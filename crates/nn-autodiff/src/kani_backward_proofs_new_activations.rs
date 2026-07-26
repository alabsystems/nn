// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for the "new activation" backward derivatives.
//!
//! Covers backward rules for activations added in `backward_rules_elementwise.rs`
//! `backward_new_activations()` which previously had zero Kani coverage:
//!
//! 1. **HardSigmoid** — piecewise linear, derivative = 1/6 in (-3, 3), 0 elsewhere
//! 2. **HardSwish** — piecewise: 0 for x<=-3, x/3+0.5 for -3<x<3, 1 for x>=3
//! 3. **Mish** — t + x*(1-t^2)*sigmoid(x) where t = tanh(softplus(x))
//! 4. **Selu** — lambda for x>=0, lambda*alpha*exp(x) for x<0
//! 5. **Softplus** — sigmoid(x) = exp(x)/(1+exp(x))
//! 6. **Celu** — 1 for x>=0, exp(x/alpha) for x<0
//!
//! Additionally covers chain rule composition and loss backward finiteness proofs:
//! 7. **MSE backward** — 2*(x-t)/N finiteness and sign properties
//! 8. **L1 backward** — sign(x-t)/N ternary and finiteness
//! 9. **Huber backward** — piecewise continuity at delta boundary
//! 10. **Chain rule** — composition of two finite-derivative ops preserves finiteness
//!
//! **Local-copy gap:** See `kani_backward_proofs.rs` module doc. Local scalar
//! functions re-implement production formulas; `// SYNC:` comments track drift.
//!
//! Re: #4283 (Kani proof harnesses for autodiff backward rules).

// ── HardSigmoid derivative ──────────────────────────────────────────
//
// HardSigmoid(x) = clamp(x/6 + 0.5, 0, 1)
// d/dx = 1/6 for x in (-3, 3), 0 otherwise
//
// SYNC: backward_rules_elementwise.rs:106-113

/// HardSigmoid derivative: 1/6 inside (-3, 3), 0 outside.
fn hard_sigmoid_derivative(x: f32) -> f32 {
    if x > -3.0 && x < 3.0 {
        1.0 / 6.0
    } else {
        0.0
    }
}

// ── HardSwish derivative ────────────────────────────────────────────
//
// HardSwish(x) = x * HardSigmoid(x)
// d/dx: 0 for x <= -3, x/3 + 0.5 for -3 < x < 3, 1 for x >= 3
//
// SYNC: backward_rules_elementwise.rs:115-129

/// HardSwish derivative: piecewise linear.
fn hard_swish_derivative(x: f32) -> f32 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        x / 3.0 + 0.5
    }
}

// ── Mish derivative ────────────────────────────────────────────────
//
// Mish(x) = x * tanh(softplus(x))
// d/dx = t + x * (1 - t^2) * sigmoid(x)
// where t = tanh(softplus(x))
//
// SYNC: backward_rules_elementwise.rs:131-143

/// Mish derivative: t + x*(1-t^2)*sigmoid(x).
/// Uses pre-computed tanh_sp and sigmoid values for Kani tractability.
fn mish_derivative(x: f32, tanh_sp: f32, sigmoid_x: f32) -> f32 {
    let t_sq = tanh_sp * tanh_sp;
    let one_minus_t_sq = 1.0 - t_sq;
    tanh_sp + x * one_minus_t_sq * sigmoid_x
}

// ── Selu derivative ────────────────────────────────────────────────
//
// d/dx SELU(x) = lambda for x >= 0, lambda * alpha * exp(x) for x < 0
//
// SYNC: backward_rules_elementwise.rs:145-153

const SELU_ALPHA: f64 = 1.6732632423543772;
const SELU_LAMBDA: f64 = 1.0507009873554805;

/// Selu derivative: lambda for x>=0, lambda*alpha*exp(x) for x<0.
fn selu_derivative(x: f32, exp_x: f32) -> f32 {
    if x >= 0.0 {
        SELU_LAMBDA as f32
    } else {
        (SELU_LAMBDA * SELU_ALPHA) as f32 * exp_x
    }
}

// ── Softplus derivative ────────────────────────────────────────────
//
// d/dx softplus(x) = sigmoid(x) = exp(x)/(1+exp(x))
//
// SYNC: backward_rules_elementwise.rs:155-158

/// Softplus derivative: sigmoid(x).
fn softplus_derivative(x: f32) -> f32 {
    let e = (-x).exp();
    1.0 / (1.0 + e)
}

// ── Celu derivative ────────────────────────────────────────────────
//
// d/dx CELU(x, alpha) = 1 for x >= 0, exp(x/alpha) for x < 0
//
// SYNC: backward_rules_elementwise.rs:160-167

/// Celu derivative: 1 for x>=0, exp(x/alpha) for x<0.
fn celu_derivative(x: f32, alpha: f64) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        (x / alpha as f32).exp()
    }
}

// ── MSE backward scalar ────────────────────────────────────────────
//
// d/dx MSE = 2*(x - t) / N
// SYNC: backward_rules_special.rs:241-255

/// MSE backward per-element gradient with upstream scale.
fn mse_backward_element(x: f32, t: f32, n: usize, grad_val: f64) -> f32 {
    let diff = x - t;
    diff * (2.0 * grad_val / n as f64) as f32
}

// ── L1 backward scalar ────────────────────────────────────────────
//
// d/dx L1 = sign(x - t) / N
// SYNC: backward_rules_special.rs:258-279

/// L1 backward per-element gradient with upstream scale.
fn l1_backward_element(x: f32, t: f32, n: usize, grad_val: f64) -> f32 {
    let diff = x - t;
    let sign = if diff > 0.0 {
        1.0
    } else if diff < 0.0 {
        -1.0
    } else {
        0.0
    };
    sign * (grad_val / n as f64) as f32
}

// ── Huber backward scalar ──────────────────────────────────────────
//
// d/dx Huber: diff/(N*delta) if |diff|<delta, sign(diff)/N otherwise
// SYNC: backward_rules_special.rs:284-312

/// Huber backward per-element gradient with upstream scale.
fn huber_backward_element(x: f32, t: f32, delta: f64, n: usize, grad_val: f64) -> f32 {
    let diff = x - t;
    if diff.abs() < delta as f32 {
        diff * (grad_val / (n as f64 * delta)) as f32
    } else {
        let sign = if diff > 0.0 {
            1.0
        } else if diff < 0.0 {
            -1.0
        } else {
            0.0
        };
        sign * (grad_val / n as f64) as f32
    }
}

// ── Chain rule composition ──────────────────────────────────────────
//
// For two elementwise ops f and g: d/dx g(f(x)) = g'(f(x)) * f'(x).
// If both derivatives are finite and bounded, the product is finite.

/// Chain rule: product of two bounded derivatives is finite.
fn chain_rule_scalar(df: f32, dg: f32) -> f32 {
    df * dg
}

// ════════════════════════════════════════════════════════════════════
// Kani proof harnesses
// ════════════════════════════════════════════════════════════════════

// --- HardSigmoid backward ---

/// Prove HardSigmoid derivative is finite for all finite inputs.
///
/// SYNC: backward_rules_elementwise.rs:106-113
#[kani::unwind(1)]
#[kani::proof]
fn prove_hard_sigmoid_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = hard_sigmoid_derivative(x);
    assert!(d.is_finite(), "hard_sigmoid derivative must be finite");
}

/// Prove HardSigmoid derivative is non-negative (monotone increasing function).
///
/// SYNC: backward_rules_elementwise.rs:106-113
#[kani::unwind(1)]
#[kani::proof]
fn prove_hard_sigmoid_derivative_nonneg() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = hard_sigmoid_derivative(x);
    assert!(d >= 0.0, "hard_sigmoid derivative must be >= 0");
}

/// Prove HardSigmoid derivative is bounded by 1/6.
/// The maximum derivative of HardSigmoid is 1/6 (inside the linear region).
///
/// SYNC: backward_rules_elementwise.rs:106-113
#[kani::unwind(1)]
#[kani::proof]
fn prove_hard_sigmoid_derivative_bounded() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = hard_sigmoid_derivative(x);
    assert!(
        d <= 1.0 / 6.0 + 1e-7,
        "hard_sigmoid derivative must be <= 1/6"
    );
}

// --- HardSwish backward ---

/// Prove HardSwish derivative is finite for all finite inputs.
///
/// SYNC: backward_rules_elementwise.rs:115-129
#[kani::unwind(1)]
#[kani::proof]
fn prove_hard_swish_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = hard_swish_derivative(x);
    assert!(d.is_finite(), "hard_swish derivative must be finite");
}

/// Prove HardSwish derivative is in [0, 1] for all finite inputs.
/// At x=-3: d=0, at x=3: d=1. Between: x/3+0.5 goes from 0 to 1.5,
/// but the function itself is 0 at x<=-3 and 1 at x>=3, so the
/// derivative is clamped to [0, 1.5]. However, for the "middle" region
/// x/3+0.5 can exceed 1 (at x=1.5: d=1.0, at x=3: d=1.5).
/// The real bound is [-0.5, 1.5] since at x=-3: d=0 and linearly
/// goes to 1.5 at x=3 (the mid-region formula gives x/3+0.5).
///
/// Weakened bound: prove d >= -0.5 and d <= 1.5 + eps.
///
/// SYNC: backward_rules_elementwise.rs:115-129
#[kani::unwind(1)]
#[kani::proof]
fn prove_hard_swish_derivative_bounded() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    let d = hard_swish_derivative(x);
    assert!(d >= -0.5 - 1e-6, "hard_swish derivative must be >= -0.5");
    assert!(d <= 1.5 + 1e-6, "hard_swish derivative must be <= 1.5");
}

/// Prove HardSwish derivative is exactly 0 for x <= -3 and 1 for x >= 3.
///
/// SYNC: backward_rules_elementwise.rs:115-129
#[kani::unwind(1)]
#[kani::proof]
fn prove_hard_swish_derivative_saturation() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x <= -3.0 || x >= 3.0);
    let d = hard_swish_derivative(x);
    if x <= -3.0 {
        assert!(d == 0.0, "hard_swish derivative must be 0 for x <= -3");
    } else {
        assert!(d == 1.0, "hard_swish derivative must be 1 for x >= 3");
    }
}

// --- Mish backward ---

/// Prove Mish derivative is finite for bounded inputs with valid tanh and sigmoid.
/// Uses pre-computed tanh_sp (in [-1,1]) and sigmoid (in [0,1]).
///
/// SYNC: backward_rules_elementwise.rs:131-143
#[kani::unwind(1)]
#[kani::proof]
fn prove_mish_derivative_finite() {
    let x: f32 = kani::any();
    let tanh_sp: f32 = kani::any();
    let sigmoid_x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 80.0);
    kani::assume(tanh_sp.is_finite() && tanh_sp >= -1.0 && tanh_sp <= 1.0);
    kani::assume(sigmoid_x.is_finite() && sigmoid_x >= 0.0 && sigmoid_x <= 1.0);
    let d = mish_derivative(x, tanh_sp, sigmoid_x);
    assert!(d.is_finite(), "mish derivative must be finite");
}

/// Prove Mish derivative is bounded for bounded inputs.
/// With |x| <= 80, tanh in [-1,1], sigmoid in [0,1]:
/// |d| = |t + x*(1-t^2)*sig| <= 1 + 80*1*1 = 81.
///
/// SYNC: backward_rules_elementwise.rs:131-143
#[kani::unwind(1)]
#[kani::proof]
fn prove_mish_derivative_bounded() {
    let x: f32 = kani::any();
    let tanh_sp: f32 = kani::any();
    let sigmoid_x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 80.0);
    kani::assume(tanh_sp.is_finite() && tanh_sp >= -1.0 && tanh_sp <= 1.0);
    kani::assume(sigmoid_x.is_finite() && sigmoid_x >= 0.0 && sigmoid_x <= 1.0);
    let d = mish_derivative(x, tanh_sp, sigmoid_x);
    assert!(
        d.abs() <= 82.0,
        "mish derivative must be bounded for |x| <= 80"
    );
}

// --- Selu backward ---

/// Prove Selu derivative is finite for bounded inputs with finite exp(x).
///
/// SYNC: backward_rules_elementwise.rs:145-153
#[kani::unwind(1)]
#[kani::proof]
fn prove_selu_derivative_finite() {
    let x: f32 = kani::any();
    let exp_x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 100.0);
    kani::assume(exp_x.is_finite() && exp_x > 0.0 && exp_x <= 1e10);
    let d = selu_derivative(x, exp_x);
    assert!(d.is_finite(), "selu derivative must be finite");
}

/// Prove Selu derivative equals lambda for non-negative inputs.
///
/// SYNC: backward_rules_elementwise.rs:145-153
#[kani::unwind(1)]
#[kani::proof]
fn prove_selu_derivative_positive_branch() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= 0.0);
    let d = selu_derivative(x, 1.0); // exp_x unused for x >= 0
    let expected = SELU_LAMBDA as f32;
    assert!(
        (d - expected).abs() < 1e-6,
        "selu derivative must be lambda for x >= 0"
    );
}

/// Prove Selu derivative is positive for all inputs (when exp_x > 0).
/// For x >= 0: lambda > 0. For x < 0: lambda*alpha*exp(x) > 0.
///
/// SYNC: backward_rules_elementwise.rs:145-153
#[kani::unwind(1)]
#[kani::proof]
fn prove_selu_derivative_always_positive() {
    let x: f32 = kani::any();
    let exp_x: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 100.0);
    kani::assume(exp_x.is_finite() && exp_x > 0.0 && exp_x <= 1e10);
    let d = selu_derivative(x, exp_x);
    assert!(d > 0.0, "selu derivative must be positive (exp always > 0)");
}

// --- Softplus backward ---

/// Prove Softplus derivative (sigmoid) is finite for bounded inputs.
///
/// SYNC: backward_rules_elementwise.rs:155-158
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn prove_softplus_derivative_finite() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -80.0 && x <= 80.0);
    let d = softplus_derivative(x);
    assert!(d.is_finite(), "softplus derivative must be finite");
}

/// Prove Softplus derivative is in (0, 1) (sigmoid range).
/// sigmoid(x) is strictly between 0 and 1 for finite x.
///
/// SYNC: backward_rules_elementwise.rs:155-158
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn prove_softplus_derivative_bounded() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -80.0 && x <= 80.0);
    let d = softplus_derivative(x);
    assert!(d >= 0.0, "softplus derivative (sigmoid) must be >= 0");
    assert!(d <= 1.0, "softplus derivative (sigmoid) must be <= 1");
}

// --- Celu backward ---

/// Prove Celu derivative is finite for bounded inputs with alpha > 0.
///
/// SYNC: backward_rules_elementwise.rs:160-167
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn prove_celu_derivative_finite() {
    let x: f32 = kani::any();
    let alpha: f64 = kani::any();
    kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);
    kani::assume(alpha.is_finite() && alpha >= 0.01 && alpha <= 10.0);
    let d = celu_derivative(x, alpha);
    assert!(d.is_finite(), "celu derivative must be finite");
}

/// Prove Celu derivative is exactly 1 for non-negative inputs.
///
/// SYNC: backward_rules_elementwise.rs:160-167
#[kani::unwind(1)]
#[kani::proof]
fn prove_celu_derivative_one_for_positive() {
    let x: f32 = kani::any();
    let alpha: f64 = kani::any();
    kani::assume(x.is_finite() && x >= 0.0);
    kani::assume(alpha.is_finite() && alpha > 0.0);
    let d = celu_derivative(x, alpha);
    assert!(d == 1.0, "celu derivative must be 1 for x >= 0");
}

/// Prove Celu derivative is positive for negative inputs (exp always > 0).
///
/// SYNC: backward_rules_elementwise.rs:160-167
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn prove_celu_derivative_positive_for_negative() {
    let x: f32 = kani::any();
    let alpha: f64 = kani::any();
    kani::assume(x.is_finite() && x < 0.0 && x >= -100.0);
    kani::assume(alpha.is_finite() && alpha >= 0.01 && alpha <= 10.0);
    let d = celu_derivative(x, alpha);
    assert!(d > 0.0, "celu derivative must be > 0 for x < 0 (exp > 0)");
}

// --- MSE backward ---

/// Prove MSE backward is finite for bounded inputs.
///
/// SYNC: backward_rules_special.rs:241-255
#[kani::unwind(1)]
#[kani::proof]
fn prove_mse_backward_element_finite() {
    let x: f32 = kani::any();
    let t: f32 = kani::any();
    let n: usize = kani::any();
    let grad_val: f64 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);
    kani::assume(t.is_finite() && t.abs() <= 1e6);
    kani::assume(n >= 1 && n <= 1_000_000);
    kani::assume(grad_val.is_finite() && grad_val.abs() <= 1.0);
    let d = mse_backward_element(x, t, n, grad_val);
    assert!(d.is_finite(), "mse backward element must be finite");
}

/// Prove MSE backward sign: positive when x > t, negative when x < t.
///
/// SYNC: backward_rules_special.rs:241-255
#[kani::unwind(1)]
#[kani::proof]
fn prove_mse_backward_element_sign() {
    let x: f32 = kani::any();
    let t: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);
    kani::assume(t.is_finite() && t.abs() <= 1e6);
    kani::assume(x != t);
    let d = mse_backward_element(x, t, 1, 1.0);
    if x > t {
        assert!(d > 0.0, "mse backward must be positive when x > t");
    } else {
        assert!(d < 0.0, "mse backward must be negative when x < t");
    }
}

// --- L1 backward ---

/// Prove L1 backward is finite for bounded inputs.
///
/// SYNC: backward_rules_special.rs:258-279
#[kani::unwind(1)]
#[kani::proof]
fn prove_l1_backward_element_finite() {
    let x: f32 = kani::any();
    let t: f32 = kani::any();
    let n: usize = kani::any();
    let grad_val: f64 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);
    kani::assume(t.is_finite() && t.abs() <= 1e6);
    kani::assume(n >= 1 && n <= 1_000_000);
    kani::assume(grad_val.is_finite() && grad_val.abs() <= 1.0);
    let d = l1_backward_element(x, t, n, grad_val);
    assert!(d.is_finite(), "l1 backward element must be finite");
}

/// Prove L1 backward is in {-1/N, 0, 1/N} (ternary scaled by grad_val).
///
/// SYNC: backward_rules_special.rs:258-279
#[kani::unwind(1)]
#[kani::proof]
fn prove_l1_backward_element_ternary() {
    let x: f32 = kani::any();
    let t: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);
    kani::assume(t.is_finite() && t.abs() <= 1e6);
    // Use n=1, grad_val=1 to get raw sign values
    let d = l1_backward_element(x, t, 1, 1.0);
    assert!(
        d == -1.0 || d == 0.0 || d == 1.0,
        "l1 backward must produce -1, 0, or 1 for n=1, grad=1"
    );
}

// --- Huber backward ---

/// Prove Huber backward is finite for bounded inputs.
///
/// SYNC: backward_rules_special.rs:284-312
#[kani::unwind(1)]
#[kani::proof]
fn prove_huber_backward_element_finite() {
    let x: f32 = kani::any();
    let t: f32 = kani::any();
    let delta: f64 = kani::any();
    let n: usize = kani::any();
    let grad_val: f64 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);
    kani::assume(t.is_finite() && t.abs() <= 1e6);
    kani::assume(delta.is_finite() && delta >= 0.01 && delta <= 100.0);
    kani::assume(n >= 1 && n <= 1_000_000);
    kani::assume(grad_val.is_finite() && grad_val.abs() <= 1.0);
    let d = huber_backward_element(x, t, delta, n, grad_val);
    assert!(d.is_finite(), "huber backward element must be finite");
}

/// Prove Huber backward is bounded: |d| <= max(1/N, |diff|/(N*delta)).
/// In the quadratic region |d| <= |diff|/(N*delta) < delta/(N*delta) = 1/N.
/// In the linear region |d| = 1/N. So |d| <= 1/N always (for grad_val=1).
///
/// SYNC: backward_rules_special.rs:284-312
#[kani::unwind(1)]
#[kani::proof]
fn prove_huber_backward_element_bounded() {
    let x: f32 = kani::any();
    let t: f32 = kani::any();
    let delta: f64 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);
    kani::assume(t.is_finite() && t.abs() <= 1e6);
    kani::assume(delta.is_finite() && delta >= 0.01 && delta <= 100.0);
    let d = huber_backward_element(x, t, delta, 1, 1.0);
    assert!(d.is_finite(), "huber backward must be finite");
    // The gradient magnitude is bounded by 1 when n=1 and grad_val=1:
    // - Quadratic region: |diff/(1*delta)| < delta/delta = 1
    // - Linear region: |sign/1| = 1
    assert!(
        d.abs() <= 1.0 + 1e-6,
        "huber backward magnitude <= 1 for n=1, grad=1"
    );
}

// --- Chain rule composition ---

/// Prove chain rule product is finite when both derivatives are bounded.
/// If |df| <= B1 and |dg| <= B2, then |df*dg| <= B1*B2.
///
/// This proves that composing any two backward rules with finite derivatives
/// produces a finite gradient — the fundamental correctness property of
/// reverse-mode autodiff.
#[kani::unwind(1)]
#[kani::proof]
fn prove_chain_rule_finite() {
    let df: f32 = kani::any();
    let dg: f32 = kani::any();
    kani::assume(df.is_finite() && df.abs() <= 1e18);
    kani::assume(dg.is_finite() && dg.abs() <= 1e18);
    let d = chain_rule_scalar(df, dg);
    assert!(
        d.is_finite(),
        "chain rule product must be finite for bounded derivatives"
    );
}

/// Prove chain rule preserves zero: if either derivative is zero, the
/// composition is zero. This ensures gradient masking (relu, clamp)
/// stops gradient flow through dead neurons.
#[kani::unwind(1)]
#[kani::proof]
fn prove_chain_rule_zero_propagation() {
    let other: f32 = kani::any();
    kani::assume(other.is_finite());
    let d1 = chain_rule_scalar(0.0, other);
    let d2 = chain_rule_scalar(other, 0.0);
    assert!(
        d1 == 0.0,
        "chain rule must be zero when first derivative is zero"
    );
    assert!(
        d2 == 0.0,
        "chain rule must be zero when second derivative is zero"
    );
}

// ── Stubs for CBMC transcendentals ──────────────────────────────────
//
// CBMC cannot accurately model f32 transcendentals.
// Same pattern as kani_backward_proofs_activation.rs (#708, #541).

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result > 0.0);
    result
}
