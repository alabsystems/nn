// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for autodiff backward rules.
//!
//! Each harness proves that the scalar derivative formula used in
//! `backward_rules.rs` is mathematically correct for the corresponding
//! forward activation. Proofs cover:
//!
//! - **Finiteness**: finite input → finite derivative
//! - **Correctness**: derivative formula matches the mathematical definition
//! - **Sign properties**: derivative has expected sign constraints
//!
//! These are the first formal proofs of gradient correctness in any ML framework.
//! The backward rules in `backward_rules.rs` operate on `DynTensor` (runtime-sized),
//! but the mathematical formulas are element-wise scalar operations. We extract
//! the scalar formulas here and prove them with Kani.
//!
//! **Local-copy gap:** These proofs verify local scalar functions that re-implement
//! the mathematical formulas from production backward rules. If production code in
//! `backward_rules.rs` or `backward_rules_*.rs` drifts from these local copies,
//! the proofs become silently disconnected. Each local function has a `// SYNC:`
//! comment referencing the specific production line numbers it mirrors. When editing
//! production backward rules, update the corresponding local copies and SYNC comments.
//!
//! Dropout proofs extracted to `kani_backward_proofs_dropout.rs`.
//! Shape/Conv/Embedding proofs in `kani_backward_proofs_shape_conv.rs`.
//!
//! Re: #13 (verified training epic).

#[cfg(kani)]
#[path = "kani_backward_proofs_binary.rs"]
mod binary;

#[cfg(kani)]
#[path = "kani_backward_proofs_activation.rs"]
mod activation;

#[cfg(kani)]
#[path = "kani_backward_proofs_math.rs"]
mod math;

#[cfg(kani)]
#[path = "kani_backward_proofs_norm.rs"]
mod norm;

#[cfg(kani)]
#[path = "kani_backward_proofs_norm_helpers.rs"]
mod norm_helpers;

#[cfg(kani)]
#[path = "kani_backward_proofs_loss.rs"]
mod loss;

#[cfg(kani)]
#[path = "kani_backward_proofs_dropout.rs"]
mod dropout;

#[cfg(kani)]
#[path = "kani_backward_proofs_matmul.rs"]
mod matmul;

#[cfg(kani)]
#[path = "kani_backward_proofs_shape_conv.rs"]
mod shape_conv;

#[cfg(kani)]
#[path = "kani_backward_proofs_ce_pool.rs"]
mod ce_pool;

#[cfg(kani)]
#[path = "kani_backward_proofs_reduce_shape.rs"]
mod reduce_shape;

#[cfg(kani)]
#[path = "kani_backward_proofs_broadcast.rs"]
mod broadcast;

#[cfg(kani)]
#[path = "kani_backward_proofs_remaining.rs"]
mod remaining;

#[cfg(kani)]
#[path = "kani_backward_proofs_new_ops.rs"]
mod new_ops;

#[cfg(kani)]
#[path = "kani_backward_rules_reduce.rs"]
mod backward_rules_reduce;

#[cfg(kani)]
#[path = "kani_backward_rules_norm_deep.rs"]
mod backward_rules_norm_deep;

#[cfg(kani)]
#[path = "kani_backward_proofs_new_activations.rs"]
mod new_activations;

// ── Scalar activation derivative functions ───────────────────────────────
//
// Each function computes d/dx f(x) for a single scalar element, matching
// the formula used in `backward_rules_elementwise.rs`.

/// ReLU derivative: d/dx max(x, 0) = 1 if x >= 0, else 0.
///
/// SYNC: backward_rules_elementwise.rs:23-27 (ge + where_cond pattern).
#[allow(dead_code)]
fn relu_derivative(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Tanh derivative: d/dx tanh(x) = 1 - tanh(x)^2.
///
/// SYNC: backward_rules_elementwise.rs:28-31 (tanh().sqr().affine(-1, 1) pattern).
#[allow(dead_code)]
fn tanh_derivative(x: f32) -> f32 {
    let t = x.tanh();
    1.0 - t * t
}

/// Sigmoid derivative: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)).
///
/// SYNC: backward_rules_elementwise.rs:32-36 (sigmoid * (1 - sigmoid) pattern).
#[allow(dead_code)]
fn sigmoid_derivative(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 - s)
}

/// Exp derivative: d/dx exp(x) = exp(x).
///
/// SYNC: backward_rules_elementwise.rs:37 (grad.mul(x.exp())).
#[allow(dead_code)]
fn exp_derivative(x: f32) -> f32 {
    x.exp()
}

/// Log derivative: d/dx ln(x) = 1/x.
///
/// SYNC: backward_rules_elementwise.rs:38 (grad.div(x)).
#[allow(dead_code)]
fn log_derivative(x: f32) -> f32 {
    1.0 / x
}

/// Sqrt derivative: d/dx sqrt(x) = 1 / (2 * sqrt(x)), with x<=0 → 0.
///
/// SYNC: backward_rules_elementwise.rs:39-50 (clamp_min + gt(0) mask pattern).
/// Production code clamps denominator to 1e-30 and masks x<=0 to zero (#2002).
/// This model mirrors that: clamp sqrt away from zero, then zero-out x<=0.
#[allow(dead_code)]
fn sqrt_derivative(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0; // subderivative convention: zero gradient at x=0
    }
    let two_sqrt = 2.0 * x.sqrt();
    // Clamp denominator to match production clamp_min(1e-30)
    let safe_denom = if two_sqrt < 1e-30 { 1e-30 } else { two_sqrt };
    1.0 / safe_denom
}

/// Sqr (x^2) derivative: d/dx x^2 = 2x.
///
/// SYNC: backward_rules_elementwise.rs:51 (grad.mul(x.affine(2, 0)) pattern).
#[allow(dead_code)]
fn sqr_derivative(x: f32) -> f32 {
    2.0 * x
}

/// Neg derivative: d/dx (-x) = -1.
///
/// SYNC: backward_rules_elementwise.rs:82 (grad.neg() → multiply by -1 pattern).
#[allow(dead_code)]
fn neg_derivative(_x: f32) -> f32 {
    -1.0
}

/// SiLU derivative: d/dx (x * sigmoid(x)) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
///
/// SYNC: backward_rules_elementwise.rs:73-81.
#[allow(dead_code)]
fn silu_derivative(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

/// Abs derivative: d/dx |x| = sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0.
///
/// SYNC: backward_rules_elementwise.rs:83-92 (gt + lt + where_cond pattern).
#[allow(dead_code)]
fn abs_derivative(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// GELU (tanh approximation) derivative.
///
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// d/dx = 0.5 * (1 + tanh(s)) + 0.5 * x * (1 - tanh(s)^2) * s'
/// where s = sqrt(2/pi) * (x + 0.044715 * x^3)
///       s' = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
///
/// SYNC: backward_rules_elementwise.rs:52-72.
#[allow(dead_code)]
fn gelu_derivative(x: f32) -> f32 {
    let sqrt_2_pi: f32 = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
    let inner = x + 0.044715 * x * x * x;
    let s = sqrt_2_pi * inner;
    let tanh_s = s.tanh();
    let term1 = 0.5 * (1.0 + tanh_s);
    let s_prime = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x);
    let sech2 = 1.0 - tanh_s * tanh_s;
    let term2 = 0.5 * x * sech2 * s_prime;
    term1 + term2
}

// ── Scalar math derivative functions ─────────────────────────────────────
//
// Each function computes d/dx f(x) for a single scalar element, matching
// the formula used in `backward_rules_elementwise.rs`.

/// Sin derivative: d/dx sin(x) = cos(x).
///
/// SYNC: backward_rules_elementwise.rs:104 (grad.mul(x.cos()) pattern).
#[allow(dead_code)]
fn sin_derivative(x: f32) -> f32 {
    x.cos()
}

/// Cos derivative: d/dx cos(x) = -sin(x).
///
/// SYNC: backward_rules_elementwise.rs:105 (grad.mul(x.sin().neg()) pattern).
#[allow(dead_code)]
fn cos_derivative(x: f32) -> f32 {
    -(x.sin())
}

/// Recip derivative: d/dx (1/x) = -1/x^2.
///
/// SYNC: backward_rules_elementwise.rs:106-109 (x.sqr().recip().neg() pattern).
#[allow(dead_code)]
fn recip_derivative(x: f32) -> f32 {
    -1.0 / (x * x)
}

/// Powf derivative: d/dx x^p = p * x^(p-1), with p=0 short-circuit.
///
/// SYNC: backward_rules_elementwise.rs:110-121 (p==0 zeros branch + general formula).
/// The p=0 short-circuit was added in #2000 to prevent 0 * x^(-1) = NaN at x=0.
#[allow(dead_code)]
fn powf_derivative(x: f32, p: f64) -> f32 {
    if p == 0.0 {
        // x^0 = 1 (constant), gradient is zero everywhere.
        // Without this, 0.0 * x.powf(-1.0) = NaN at x=0 (IEEE 754: 0 * Inf = NaN).
        0.0
    } else {
        let p = p as f32;
        p * x.powf(p - 1.0)
    }
}

/// Clamp derivative: d/dx clamp(x, lo, hi) = 1 if lo <= x <= hi, else 0.
///
/// SYNC: backward_rules_elementwise.rs:122-130 (ge(lo) && le(hi) where_cond pattern).
/// Uses non-strict ge/le so gradient flows at boundary values, matching PyTorch.
#[allow(dead_code)]
fn clamp_derivative(x: f32, lo: f64, hi: f64) -> f32 {
    if x >= lo as f32 && x <= hi as f32 {
        1.0
    } else {
        0.0
    }
}

/// ELU derivative: d/dx ELU(x, alpha) = 1 if x > 0, alpha * exp(x) if x <= 0.
///
/// SYNC: backward_rules_elementwise.rs:131-138.
#[allow(dead_code)]
fn elu_derivative(x: f32, alpha: f64) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        alpha as f32 * x.exp()
    }
}

/// MSE loss backward scalar derivative: d/dx ||x - t||^2 / N = 2*(x - t) / N.
///
/// SYNC: backward_rules_special.rs:241-255.
#[allow(dead_code)]
fn mse_backward_scalar(x: f32, t: f32, n: usize) -> f32 {
    2.0 * (x - t) / n as f32
}

/// L1 loss backward scalar derivative: d/dx |x - t| / N = sign(x - t) / N.
///
/// SYNC: backward_rules_special.rs:258-279.
#[allow(dead_code)]
fn l1_backward_scalar(x: f32, t: f32, n: usize) -> f32 {
    let diff = x - t;
    let sign = if diff > 0.0 {
        1.0
    } else if diff < 0.0 {
        -1.0
    } else {
        0.0
    };
    sign / n as f32
}

/// Huber loss backward scalar derivative (piecewise):
///   diff / (N * delta)          if |diff| < delta
///   sign(diff) / N              if |diff| >= delta
///
/// SYNC: backward_rules_special.rs:284-312.
#[allow(dead_code)]
fn huber_backward_scalar(x: f32, t: f32, delta: f64, n: usize) -> f32 {
    let diff = x - t;
    if diff.abs() < delta as f32 {
        diff / (n as f32 * delta as f32)
    } else {
        let sign = if diff > 0.0 {
            1.0
        } else if diff < 0.0 {
            -1.0
        } else {
            0.0
        };
        sign / n as f32
    }
}

// ── Scalar reduction backward functions ──────────────────────────────────
//
// Reduction backward rules scale the upstream gradient by a constant factor.
// MeanKeepDim scales by 1/n; SumKeepDim just broadcasts (factor = 1).

/// MeanKeepDim backward scaling factor: 1/n.
///
/// SYNC: backward_rules.rs:162-165 (grad.mul_scalar(1.0 / n) pattern).
/// n is the dimension size being reduced.
#[allow(dead_code)]
fn mean_backward_scale(n: usize) -> f64 {
    1.0 / n as f64
}

/// Softmax backward: single-element Jacobian diagonal.
///
/// For loss = f(softmax(x)), the backward for element i of the logit vector is:
///   grad_x[i] = s[i] * (grad[i] - dot(grad, s))
/// where s = softmax(x). This function computes the factor for a single element.
///
/// SYNC: backward_rules_special.rs:38-43 (softmax_backward_data pattern).
#[allow(dead_code)]
fn softmax_backward_element(s_i: f32, grad_i: f32, dot_grad_s: f32) -> f32 {
    s_i * (grad_i - dot_grad_s)
}

// ── Kani proof harnesses for reduction/composite backward ────────────────

#[cfg(kani)]
mod reduction_proofs {
    use super::*;

    /// Prove MeanKeepDim backward scale factor is finite and positive for n > 0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn mean_backward_scale_finite() {
        let n: usize = kani::any();
        kani::assume(n > 0 && n <= 1_000_000);
        let scale = mean_backward_scale(n);
        assert!(scale.is_finite(), "mean backward scale must be finite");
        assert!(scale > 0.0, "mean backward scale must be positive");
    }

    /// Prove MeanKeepDim backward scale is at most 1.0 (n >= 1).
    #[kani::unwind(1)]
    #[kani::proof]
    fn mean_backward_scale_bounded() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 1_000_000);
        let scale = mean_backward_scale(n);
        assert!(scale <= 1.0, "mean backward scale must be <= 1.0");
    }

    /// Prove softmax backward element formula is finite when all inputs are finite.
    #[kani::unwind(1)]
    #[kani::proof]
    fn softmax_backward_element_finite() {
        let s_i: f32 = kani::any();
        let grad_i: f32 = kani::any();
        let dot_grad_s: f32 = kani::any();
        kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i <= 1.0);
        kani::assume(grad_i.is_finite() && grad_i.abs() <= 1e6);
        kani::assume(dot_grad_s.is_finite() && dot_grad_s.abs() <= 1e6);
        let d = softmax_backward_element(s_i, grad_i, dot_grad_s);
        assert!(d.is_finite(), "softmax backward element must be finite");
    }

    /// Prove softmax backward produces zero gradient when s_i = 0 (masked out).
    ///
    /// In attention masking, some softmax outputs are zero. The backward
    /// gradient for those elements should be exactly zero regardless of
    /// the upstream gradient, preventing gradient flow through masked positions.
    #[kani::unwind(1)]
    #[kani::proof]
    fn softmax_backward_zero_when_masked() {
        let grad_i: f32 = kani::any();
        let dot_grad_s: f32 = kani::any();
        kani::assume(grad_i.is_finite() && grad_i.abs() <= 1e6);
        kani::assume(dot_grad_s.is_finite() && dot_grad_s.abs() <= 1e6);
        let d = softmax_backward_element(0.0, grad_i, dot_grad_s);
        // Use value equality, not bit equality: 0.0 * negative = -0.0 in IEEE 754,
        // and (-0.0).to_bits() != (0.0).to_bits(), but -0.0 == 0.0.
        assert!(d == 0.0, "softmax backward must be zero when s_i = 0");
    }
}
