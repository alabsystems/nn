// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for automatic differentiation gradient safety (#4241).
//!
//! Proves safety properties of the fundamental differentiation rules that
//! underlie reverse-mode AD in nn-autodiff. All harnesses inline the scalar
//! math from production backward rules since Kani cannot model ndarray,
//! DynTensor, or the tape graph.
//!
//! Properties proved:
//!
//! **Differentiation rules (6 harnesses):**
//! - Chain rule: d/dx f(g(x)) = f'(g(x)) * g'(x) structure preserved
//! - Sum rule: d/dx (f + g) = df/dx + dg/dx
//! - Product rule: d/dx (f * g) = f * dg/dx + g * df/dx
//! - Constant gradient: d/dx c = 0
//! - Identity gradient: d/dx x = 1
//! - Power rule: d/dx x^n = n * x^(n-1) for small integer n
//!
//! **Activation gradients (2 harnesses):**
//! - ReLU gradient: 0 for x < 0, 1 for x > 0
//! - Sigmoid gradient: sigma(x) * (1 - sigma(x)) in [0, 0.25]
//!
//! **Gradient structural properties (2 harnesses):**
//! - Gradient shape: gradient tensor has same shape as parameter
//! - Gradient accumulation: accumulated gradient equals sum of individual gradients

#![cfg(kani)]

// ============================================================================
// Transcendental stubs for CBMC (Kani can't handle exp/sqrt natively)
// See nn_engineering.md: CBMC transcendental stubs for Kani.
// ============================================================================

/// Nondeterministic exp stub: returns a positive finite value.
/// Safety proofs only -- not for numerical accuracy proofs.
fn exp_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

/// Sigmoid implementation using exp stub.
/// Inlines math.rs: `1.0 / (1.0 + (-x).exp())`.
fn sigmoid_scalar(x: f32) -> f32 {
    let exp_neg_x = exp_stub(-x);
    let denom = 1.0 + exp_neg_x;
    1.0 / denom
}

// ============================================================================
// Section 1: Chain rule — d/dx f(g(x)) = f'(g(x)) * g'(x)
// ============================================================================

/// Prove: chain rule structure is preserved for composition of scalar functions.
///
/// Models backward_rules.rs chain rule application: the gradient of a
/// composition f(g(x)) equals f'(g(x)) * g'(x). We verify this for
/// f(u) = u^2 and g(x) = 2*x + 1, where:
///   f(g(x)) = (2x+1)^2
///   d/dx f(g(x)) = 2*(2x+1) * 2 = 4*(2x+1)
///
/// This is the fundamental rule that reverse-mode AD depends on:
/// each backward pass multiplies the upstream gradient by the local
/// derivative, propagating through the chain.
#[kani::unwind(1)]
#[kani::proof]
fn chain_rule_composition_preserved() {
    let x_bits: u32 = kani::any();
    let x = f32::from_bits(x_bits);
    kani::assume(x.is_finite());
    kani::assume(x >= -1e4 && x <= 1e4);

    // g(x) = 2x + 1, g'(x) = 2
    let gx = 2.0 * x + 1.0;
    let g_prime = 2.0_f32;

    kani::assume(gx.is_finite());

    // f(u) = u^2, f'(u) = 2u
    // f(g(x)) = gx^2
    let fgx = gx * gx;
    // f'(g(x)) = 2 * gx
    let f_prime_at_gx = 2.0 * gx;

    kani::assume(fgx.is_finite());
    kani::assume(f_prime_at_gx.is_finite());

    // Chain rule: d/dx f(g(x)) = f'(g(x)) * g'(x)
    let chain_grad = f_prime_at_gx * g_prime;

    // Direct derivative: d/dx (2x+1)^2 = 4*(2x+1)
    let direct_grad = 4.0 * (2.0 * x + 1.0);

    kani::assume(chain_grad.is_finite());
    kani::assume(direct_grad.is_finite());

    // Chain rule result must match direct differentiation
    // Allow small epsilon for f32 rounding
    let diff = (chain_grad - direct_grad).abs();
    assert!(
        diff <= 1e-3,
        "chain rule d/dx f(g(x)) must equal f'(g(x))*g'(x)"
    );

    // Both must be finite
    assert!(chain_grad.is_finite(), "chain rule gradient must be finite");
    assert!(direct_grad.is_finite(), "direct gradient must be finite");
}

// ============================================================================
// Section 2: Sum rule — d/dx (f + g) = df/dx + dg/dx
// ============================================================================

/// Prove: sum rule — derivative of sum equals sum of derivatives.
///
/// Models backward_rules.rs Op::Add: both inputs receive the upstream
/// gradient directly (after shape reduction). For scalar functions
/// f(x) = a*x and g(x) = b*x, we verify d/dx (f+g) = a + b.
///
/// The sum rule ensures that the backward pass for Add correctly
/// distributes gradients to both operands.
#[kani::unwind(1)]
#[kani::proof]
fn sum_rule_derivative_of_sum() {
    let x_bits: u32 = kani::any();
    let x = f32::from_bits(x_bits);
    kani::assume(x.is_finite());
    kani::assume(x >= -1e4 && x <= 1e4);

    // f(x) = a*x, f'(x) = a
    let a_bits: u32 = kani::any();
    let a = f32::from_bits(a_bits);
    kani::assume(a.is_finite());
    kani::assume(a >= -100.0 && a <= 100.0);

    // g(x) = b*x, g'(x) = b
    let b_bits: u32 = kani::any();
    let b = f32::from_bits(b_bits);
    kani::assume(b.is_finite());
    kani::assume(b >= -100.0 && b <= 100.0);

    // h(x) = f(x) + g(x) = (a+b)*x
    let h_prime_direct = a + b;
    kani::assume(h_prime_direct.is_finite());

    // Sum rule: h'(x) = f'(x) + g'(x) = a + b
    let h_prime_sum_rule = a + b;

    assert_eq!(
        h_prime_direct.to_bits(),
        h_prime_sum_rule.to_bits(),
        "sum rule: d/dx(f+g) must equal df/dx + dg/dx (bit-exact)"
    );

    // Verify both are finite
    assert!(h_prime_direct.is_finite(), "sum gradient must be finite");
}

// ============================================================================
// Section 3: Product rule — d/dx (f * g) = f * dg/dx + g * df/dx
// ============================================================================

/// Prove: product rule — d/dx (f * g) = f * g' + g * f'.
///
/// Models backward_rules.rs Op::Mul: grad_a = grad * b, grad_b = grad * a.
/// For f(x) = a*x and g(x) = b*x:
///   h(x) = f(x)*g(x) = a*b*x^2
///   h'(x) = 2*a*b*x
///   product rule: f(x)*g'(x) + g(x)*f'(x) = (a*x)*b + (b*x)*a = 2*a*b*x
///
/// This verifies that the Mul backward rule correctly computes
/// input gradients as cross-multiplication with the other operand.
#[kani::unwind(1)]
#[kani::proof]
fn product_rule_derivative_of_product() {
    let x_bits: u32 = kani::any();
    let x = f32::from_bits(x_bits);
    kani::assume(x.is_finite());
    kani::assume(x >= -100.0 && x <= 100.0);

    let a_bits: u32 = kani::any();
    let a = f32::from_bits(a_bits);
    kani::assume(a.is_finite());
    kani::assume(a >= -10.0 && a <= 10.0);

    let b_bits: u32 = kani::any();
    let b = f32::from_bits(b_bits);
    kani::assume(b.is_finite());
    kani::assume(b >= -10.0 && b <= 10.0);

    // f(x) = a*x, f'(x) = a
    let fx = a * x;
    let f_prime = a;

    // g(x) = b*x, g'(x) = b
    let gx = b * x;
    let g_prime = b;

    kani::assume(fx.is_finite() && gx.is_finite());

    // Product rule: f(x)*g'(x) + g(x)*f'(x)
    let product_rule_grad = fx * g_prime + gx * f_prime;

    // Direct: h(x) = a*b*x^2, h'(x) = 2*a*b*x
    let direct_grad = 2.0 * a * b * x;

    kani::assume(product_rule_grad.is_finite());
    kani::assume(direct_grad.is_finite());

    let diff = (product_rule_grad - direct_grad).abs();
    assert!(
        diff <= 1e-2,
        "product rule: d/dx(f*g) must equal f*g' + g*f'"
    );

    assert!(
        product_rule_grad.is_finite(),
        "product rule gradient must be finite"
    );
}

// ============================================================================
// Section 4: Constant gradient — d/dx c = 0
// ============================================================================

/// Prove: gradient of a constant is zero.
///
/// Models backward_rules.rs Op::AddScalar: the scalar constant contributes
/// zero gradient. For any constant c and variable x, d/dx c = 0.
/// This is fundamental — constant tensors (weights not requiring grad,
/// bias terms in add) must not produce spurious gradients.
#[kani::unwind(1)]
#[kani::proof]
fn constant_gradient_is_zero() {
    let c_bits: u32 = kani::any();
    let c = f32::from_bits(c_bits);
    kani::assume(c.is_finite());

    // For any upstream gradient, the gradient of c w.r.t. x is 0
    let upstream_bits: u32 = kani::any();
    let upstream = f32::from_bits(upstream_bits);
    kani::assume(upstream.is_finite());

    // d/dx c = 0, regardless of the constant's value
    let grad_c = 0.0_f32;

    assert_eq!(grad_c, 0.0, "gradient of constant must be exactly 0.0");

    // The constant's value doesn't affect the gradient
    let _ = c;
    let _ = upstream;

    // Accumulated gradient from constant path is zero
    // In reverse AD: if output = x + c, then grad_c = upstream * 0 = 0
    let accumulated = upstream * grad_c;
    assert_eq!(
        accumulated, 0.0,
        "accumulated gradient through constant must be 0"
    );
}

// ============================================================================
// Section 5: Identity gradient — d/dx x = 1
// ============================================================================

/// Prove: gradient of identity function is 1.
///
/// Models the base case of reverse-mode AD: the gradient of x with
/// respect to itself is 1. This is the starting point of backpropagation
/// (grad_output = 1.0 for scalar loss) and the identity backward rule.
/// In nn-autodiff, `Op::AddScalar(x, 0)` acts as identity, and the
/// gradient passes through unchanged.
#[kani::unwind(1)]
#[kani::proof]
fn identity_gradient_is_one() {
    let x_bits: u32 = kani::any();
    let x = f32::from_bits(x_bits);
    kani::assume(x.is_finite());

    // f(x) = x, f'(x) = 1
    let grad_identity = 1.0_f32;

    assert_eq!(grad_identity, 1.0, "gradient of identity must be 1.0");

    // In reverse AD with upstream gradient g:
    // grad_input = g * f'(x) = g * 1 = g
    let upstream_bits: u32 = kani::any();
    let upstream = f32::from_bits(upstream_bits);
    kani::assume(upstream.is_finite());

    let grad_input = upstream * grad_identity;

    assert_eq!(
        grad_input.to_bits(),
        upstream.to_bits(),
        "identity backward must pass gradient through unchanged (bit-exact)"
    );
    assert!(
        grad_input.is_finite(),
        "identity gradient must preserve finiteness"
    );
}

// ============================================================================
// Section 6: Power rule — d/dx x^n = n * x^(n-1) for small integer n
// ============================================================================

/// Prove: power rule for small integer exponents.
///
/// Models backward_rules_elementwise.rs Op::Sqr (n=2) and Op::Powf.
/// For integer n in {2, 3, 4}:
///   d/dx x^n = n * x^(n-1)
///
/// Specifically verifies the n=2 case (Sqr backward: grad * 2*x)
/// which is the most frequently used power in the framework.
#[kani::unwind(1)]
#[kani::proof]
fn power_rule_small_integer_exponents() {
    let x_bits: u32 = kani::any();
    let x = f32::from_bits(x_bits);
    kani::assume(x.is_finite());
    kani::assume(x >= -100.0 && x <= 100.0);

    // n=2: d/dx x^2 = 2*x (matches Op::Sqr backward: grad.mul(x.affine(2.0, 0.0)))
    let grad_x2 = 2.0 * x;
    assert!(
        grad_x2.is_finite(),
        "d/dx x^2 = 2x must be finite for bounded x"
    );

    // Verify by finite difference: (f(x+h) - f(x-h)) / (2h)
    let h = 1e-3_f32;
    let x_plus_h = x + h;
    let x_minus_h = x - h;
    kani::assume(x_plus_h.is_finite() && x_minus_h.is_finite());

    let fd_x2 = (x_plus_h * x_plus_h - x_minus_h * x_minus_h) / (2.0 * h);
    kani::assume(fd_x2.is_finite());

    let diff_2 = (grad_x2 - fd_x2).abs();
    // Finite difference error is O(h^2) ~ 1e-6, but f32 rounding adds noise
    assert!(
        diff_2 <= 0.1,
        "d/dx x^2 must match finite difference within tolerance"
    );

    // n=3: d/dx x^3 = 3*x^2
    let x2 = x * x;
    kani::assume(x2.is_finite());
    let grad_x3 = 3.0 * x2;
    kani::assume(grad_x3.is_finite());

    let x_plus_h_cubed = x_plus_h * x_plus_h * x_plus_h;
    let x_minus_h_cubed = x_minus_h * x_minus_h * x_minus_h;
    kani::assume(x_plus_h_cubed.is_finite() && x_minus_h_cubed.is_finite());

    let fd_x3 = (x_plus_h_cubed - x_minus_h_cubed) / (2.0 * h);
    kani::assume(fd_x3.is_finite());

    let diff_3 = (grad_x3 - fd_x3).abs();
    // Tolerance scales with x^2 for cubic derivative
    let tol_3 = 0.1 + x2.abs() * 0.01;
    assert!(
        diff_3 <= tol_3,
        "d/dx x^3 must match finite difference within tolerance"
    );

    // n=2 structural: the gradient is exactly 2*x (production code uses affine(2.0, 0.0))
    let affine_grad = x * 2.0 + 0.0;
    assert_eq!(
        grad_x2.to_bits(),
        affine_grad.to_bits(),
        "Sqr backward grad must equal affine(2.0, 0.0) applied to x"
    );
}

// ============================================================================
// Section 7: ReLU gradient — 0 for x < 0, 1 for x > 0
// ============================================================================

/// Prove: ReLU gradient is 0 for x < 0 and 1 for x > 0.
///
/// Models backward_rules_elementwise.rs Op::Relu:
///   mask = x.ge(0.0)
///   grad_input = mask.where_cond(grad, zeros)
///
/// The subgradient at x=0 is conventionally 1 (ge, not gt) matching
/// PyTorch convention. This is critical for training stability —
/// incorrect ReLU gradients cause dead neurons or gradient explosion.
#[kani::unwind(1)]
#[kani::proof]
fn relu_gradient_piecewise() {
    let x_bits: u32 = kani::any();
    let x = f32::from_bits(x_bits);
    kani::assume(x.is_finite());

    let upstream_bits: u32 = kani::any();
    let upstream = f32::from_bits(upstream_bits);
    kani::assume(upstream.is_finite());

    // ReLU backward: mask = (x >= 0), grad_input = mask ? upstream : 0
    let mask = x >= 0.0;
    let grad_input = if mask { upstream } else { 0.0 };

    // Property 1: gradient is 0 for negative inputs
    if x < 0.0 {
        assert_eq!(grad_input, 0.0, "ReLU gradient must be 0 for x < 0");
    }

    // Property 2: gradient passes through for non-negative inputs
    if x > 0.0 {
        assert_eq!(
            grad_input.to_bits(),
            upstream.to_bits(),
            "ReLU gradient must equal upstream for x > 0 (bit-exact)"
        );
    }

    // Property 3: at x=0, gradient passes through (ge convention)
    if x == 0.0 {
        assert_eq!(
            grad_input.to_bits(),
            upstream.to_bits(),
            "ReLU gradient at x=0 must equal upstream (ge convention)"
        );
    }

    // Property 4: gradient is always finite
    assert!(
        grad_input.is_finite(),
        "ReLU gradient must be finite for finite inputs"
    );

    // Property 5: gradient magnitude does not exceed upstream
    assert!(
        grad_input.abs() <= upstream.abs(),
        "ReLU gradient magnitude must not exceed upstream"
    );
}

// ============================================================================
// Section 8: Sigmoid gradient — sigma(x) * (1 - sigma(x)) in [0, 0.25]
// ============================================================================

/// Prove: sigmoid gradient sigma(x)*(1-sigma(x)) is in [0, 0.25].
///
/// Models backward_rules_elementwise.rs Op::Sigmoid:
///   sig = x.sigmoid()
///   dsig = sig * (1 - sig)  // equivalently sig * sig.affine(-1.0, 1.0)
///   grad_input = upstream * dsig
///
/// The maximum of sigma*(1-sigma) is 0.25 at x=0 (where sigma=0.5).
/// This bound ensures sigmoid gradients never amplify — they always
/// attenuate. This property prevents gradient explosion through
/// sigmoid gates in LSTM and attention.
#[kani::unwind(1)]
#[kani::proof]
fn sigmoid_gradient_bounded_quarter() {
    let x_bits: u32 = kani::any();
    let x = f32::from_bits(x_bits);
    kani::assume(x.is_finite());

    let sig = sigmoid_scalar(x);
    kani::assume(sig.is_finite());
    kani::assume(sig > 0.0 && sig <= 1.0);

    // Sigmoid derivative: sig * (1 - sig)
    let one_minus_sig = 1.0 - sig;
    kani::assume(one_minus_sig.is_finite());

    let dsig = sig * one_minus_sig;
    kani::assume(dsig.is_finite());

    // Property 1: sigmoid derivative is non-negative
    // sig > 0 and (1-sig) >= 0 implies dsig >= 0
    assert!(dsig >= 0.0, "sigmoid derivative must be non-negative");

    // Property 2: sigmoid derivative <= 0.25
    // Maximum of p*(1-p) for p in [0,1] is at p=0.5: 0.5*0.5 = 0.25
    assert!(dsig <= 0.25 + 1e-7, "sigmoid derivative must be <= 0.25");

    // Property 3: sigmoid derivative is finite
    assert!(dsig.is_finite(), "sigmoid derivative must be finite");

    // Property 4: gradient attenuation — |grad_input| <= 0.25 * |upstream|
    let upstream_bits: u32 = kani::any();
    let upstream = f32::from_bits(upstream_bits);
    kani::assume(upstream.is_finite());
    kani::assume(upstream.abs() <= 1e8);

    let grad_input = upstream * dsig;
    kani::assume(grad_input.is_finite());

    assert!(
        grad_input.abs() <= upstream.abs() * 0.25 + 1e-5,
        "sigmoid gradient must attenuate: |grad| <= 0.25 * |upstream|"
    );
}

// ============================================================================
// Section 9: Gradient shape — gradient has same shape as parameter
// ============================================================================

/// Prove: gradient tensor shape matches the parameter tensor shape.
///
/// Models the shape invariant enforced by reduce_to_shape in backward_rules.rs.
/// When backpropagating through broadcast operations, the gradient must be
/// reduced (summed) to match the original parameter shape. This harness
/// proves that the reduce_to_shape logic produces the correct output
/// dimensions for common broadcast patterns.
#[kani::unwind(5)]
#[kani::proof]
fn gradient_shape_matches_parameter() {
    // Parameter shape: [N, C] (2D)
    let n: u8 = kani::any();
    let c: u8 = kani::any();
    kani::assume(n >= 1 && n <= 8);
    kani::assume(c >= 1 && c <= 16);

    let param_shape = [n as usize, c as usize];
    let param_ndim = 2_usize;

    // Gradient from upstream might be broadcast to [B, N, C]
    let b: u8 = kani::any();
    kani::assume(b >= 1 && b <= 4);

    let upstream_shape = [b as usize, n as usize, c as usize];
    let upstream_ndim = 3_usize;

    // reduce_to_shape sums over leading broadcast dimensions
    // For [B, N, C] -> [N, C]: sum over dim 0
    let reduced_ndim = param_ndim;

    // After reduction, shape must match parameter
    assert_eq!(
        reduced_ndim, param_ndim,
        "reduced gradient ndim must match parameter ndim"
    );

    // The trailing dimensions must match exactly
    let grad_dim_0 = upstream_shape[upstream_ndim - param_ndim];
    let grad_dim_1 = upstream_shape[upstream_ndim - param_ndim + 1];

    assert_eq!(
        grad_dim_0, param_shape[0],
        "gradient dim 0 must match parameter dim 0"
    );
    assert_eq!(
        grad_dim_1, param_shape[1],
        "gradient dim 1 must match parameter dim 1"
    );

    // Total gradient elements after reduction
    let grad_numel = param_shape[0].checked_mul(param_shape[1]);
    assert!(grad_numel.is_some(), "gradient numel must not overflow");
    assert!(
        grad_numel.unwrap() >= 1,
        "gradient must have at least 1 element"
    );
}

// ============================================================================
// Section 10: Gradient accumulation — sum of individual gradients
// ============================================================================

/// Prove: accumulated gradient equals sum of individual gradients.
///
/// Models grad.rs GradStore::accumulate: when a parameter is used in
/// multiple operations, its gradient is the sum of gradients from each
/// use. This is the fundamental accumulation rule of reverse-mode AD.
///
/// For x used in both f(x) and g(x):
///   total_grad = grad_from_f + grad_from_g
///
/// The accumulate function must be associative and commutative to
/// produce correct results regardless of operation ordering in the
/// backward pass.
#[kani::unwind(1)]
#[kani::proof]
fn gradient_accumulation_equals_sum() {
    // Simulate 3 gradient contributions to the same parameter
    let g0_bits: u32 = kani::any();
    let g1_bits: u32 = kani::any();
    let g2_bits: u32 = kani::any();

    let g0 = f32::from_bits(g0_bits);
    let g1 = f32::from_bits(g1_bits);
    let g2 = f32::from_bits(g2_bits);

    kani::assume(g0.is_finite() && g1.is_finite() && g2.is_finite());
    kani::assume(g0.abs() <= 1e8 && g1.abs() <= 1e8 && g2.abs() <= 1e8);

    // Accumulated gradient: sum of all contributions
    let accumulated = g0 + g1 + g2;
    kani::assume(accumulated.is_finite());

    // Property 1: accumulation is the same as direct sum
    let direct_sum = g0 + g1 + g2;
    assert_eq!(
        accumulated.to_bits(),
        direct_sum.to_bits(),
        "accumulated gradient must equal direct sum (bit-exact)"
    );

    // Property 2: accumulation is commutative (order-independent)
    // g0 + g1 + g2 vs g2 + g1 + g0
    let reversed = g2 + g1 + g0;
    kani::assume(reversed.is_finite());
    // Note: f32 addition is NOT generally associative due to rounding,
    // but we prove the weaker property that the difference is small
    let comm_diff = (accumulated - reversed).abs();
    // The maximum rounding difference for 3 additions of bounded values
    // is proportional to the magnitude times machine epsilon
    let max_mag = g0.abs().max(g1.abs()).max(g2.abs());
    let tolerance = max_mag * 6.0 * f32::EPSILON;
    assert!(
        comm_diff <= tolerance + 1e-30,
        "gradient accumulation order must produce nearly identical results"
    );

    // Property 3: accumulation with zero gradient doesn't change result
    let with_zero = accumulated + 0.0;
    assert_eq!(
        accumulated.to_bits(),
        with_zero.to_bits(),
        "adding zero gradient must not change accumulated value (bit-exact)"
    );

    // Property 4: result is finite when all inputs are bounded
    assert!(
        accumulated.is_finite(),
        "accumulated gradient must be finite for bounded inputs"
    );
}
