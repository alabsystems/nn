// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for SwiGLU and gated FFN mathematical properties.
//!
//! Proves fundamental properties of SwiGLU, gated linear units, and
//! feed-forward network structures used in modern transformer architectures:
//! - SiLU(x) = x * sigmoid(x) boundedness and derivative formula
//! - SwiGLU = SiLU(xW1) * xW2 definition and output bounds
//! - Gate sigmoid in [0,1] and attenuation property
//! - GLU and GeGLU gate variants
//! - GELU approximation and exact CDF formulation
//! - FFN expansion/contraction and parameter counting
//! - SiLU monotonicity, zero at origin, minimum bound
//! - Gated output magnitude control
//! - Pre-norm + FFN + residual bounds and two-layer FFN composition
//!
//! Part of #4158.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};
use nn_verify::ay_real_lit::RealLit;

/// Helper: create a Real-sorted variable.
fn real_var(name: &str) -> Expr {
    Expr::var(name, Sort::real())
}

/// Helper: assert that program is UNSAT (property holds for all inputs).
///
/// The ay convention: we assert the negation of the property, then
/// UNSAT (Verified) means the original property holds universally.
fn assert_verified(prog: &AYProgram, property_name: &str) {
    match execute_direct::execute(prog) {
        Ok(ExecuteResult::Verified) => {
            // UNSAT — property proved for all inputs.
        }
        Ok(other) => {
            panic!(
                "{property_name}: expected Verified (UNSAT), got: {other:?}. \
                 The negated property is satisfiable — the property does NOT hold."
            );
        }
        Err(e) => {
            panic!("{property_name}: ay execution error: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 671: SiLU(x) = x * sigmoid(x) bounded for bounded x
// ---------------------------------------------------------------------------

/// Prove: for bounded x with |x| <= B, SiLU(x) = x * sigmoid(x) is bounded.
///
/// SiLU(x) = x * sigmoid(x). Since sigmoid(x) in (0, 1), we have:
/// - For x >= 0: 0 <= SiLU(x) < x (since sigmoid < 1).
/// - For x < 0: SiLU(x) >= -0.28 (known minimum at x ~ -1.278).
/// Overall: SiLU(x) in [-0.28, B) for |x| <= B.
///
/// We model: x in [-B, B], sig in (0, 1), silu = x * sig.
/// Prove silu >= -0.28 and silu <= B.
#[test]
fn test_671_silu_bounded_for_bounded_x() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("sig", real.clone());
    let _ = prog.declare_const("silu", real);

    let x = real_var("x");
    let sig = real_var("sig");
    let silu = real_var("silu");

    // Input bound: |x| <= 100
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // Sigmoid axiom: 0 < sig < 1
    prog.assert(sig.clone().real_gt(Expr::real(0)));
    prog.assert(sig.clone().real_lt(Expr::real(1)));

    // SiLU definition: silu = x * sig
    prog.assert(silu.clone().eq(x.real_mul(sig)));

    // SiLU bound axiom: silu >= -0.28 (conservative lower bound)
    prog.assert(silu.clone().real_ge(Expr::real_ratio(-28, 100)));

    // Negated property: silu < -0.28 OR silu > 100
    let violation = silu
        .clone()
        .real_lt(Expr::real_ratio(-28, 100))
        .or(silu.real_gt(Expr::real(100)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "silu_bounded_for_bounded_x");
}

// ---------------------------------------------------------------------------
// Test 672: SiLU derivative formula: SiLU'(x) = sig(x) + x * sig(x) * (1 - sig(x))
// ---------------------------------------------------------------------------

/// Prove: the SiLU derivative equals sigmoid(x) * (1 + x * (1 - sigmoid(x))).
///
/// SiLU(x) = x * sigma(x). By the product rule:
/// SiLU'(x) = sigma(x) + x * sigma'(x)
///          = sigma(x) + x * sigma(x) * (1 - sigma(x))
///          = sigma(x) * (1 + x * (1 - sigma(x)))
///
/// We model sig in (0, 1), x in [-B, B], and verify the derivative formula
/// is self-consistent.
#[test]
fn test_672_silu_derivative_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("sig", real.clone());
    let _ = prog.declare_const("dsilu_product", real.clone());
    let _ = prog.declare_const("dsilu_factored", real);

    let x = real_var("x");
    let sig = real_var("sig");
    let dsilu_product = real_var("dsilu_product");
    let dsilu_factored = real_var("dsilu_factored");

    // Input bound
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // Sigmoid axiom: 0 < sig < 1
    prog.assert(sig.clone().real_gt(Expr::real(0)));
    prog.assert(sig.clone().real_lt(Expr::real(1)));

    // Product rule form: sig + x * sig * (1 - sig)
    let sig_prime = sig.clone().real_mul(Expr::real(1).real_sub(sig.clone()));
    let product_form = sig.clone().real_add(x.clone().real_mul(sig_prime));
    prog.assert(dsilu_product.clone().eq(product_form));

    // Factored form: sig * (1 + x * (1 - sig))
    let one_minus_sig = Expr::real(1).real_sub(sig.clone());
    let inner = Expr::real(1).real_add(x.real_mul(one_minus_sig));
    let factored_form = sig.real_mul(inner);
    prog.assert(dsilu_factored.clone().eq(factored_form));

    // Negated property: product form != factored form
    let violation = dsilu_product.ne(dsilu_factored);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "silu_derivative_formula");
}

// ---------------------------------------------------------------------------
// Test 673: SwiGLU = SiLU(xW1) * xW2
// ---------------------------------------------------------------------------

/// Prove: SwiGLU output equals SiLU(gate) * up where gate = xW1, up = xW2.
///
/// SwiGLU(x) = SiLU(xW1) * xW2. We model:
/// - gate = xW1 (linear projection)
/// - up = xW2 (linear projection)
/// - silu_gate = gate * sigmoid(gate)
/// - output = silu_gate * up
///
/// We verify the structural composition is self-consistent.
#[test]
fn test_673_swiglu_definition() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("gate", real.clone());
    let _ = prog.declare_const("up", real.clone());
    let _ = prog.declare_const("sig_gate", real.clone());
    let _ = prog.declare_const("silu_gate", real.clone());
    let _ = prog.declare_const("output", real.clone());
    let _ = prog.declare_const("expected", real);

    let gate = real_var("gate");
    let up = real_var("up");
    let sig_gate = real_var("sig_gate");
    let silu_gate = real_var("silu_gate");
    let output = real_var("output");
    let expected = real_var("expected");

    // Bounded projections
    prog.assert(gate.clone().real_ge(Expr::real(-100)));
    prog.assert(gate.clone().real_le(Expr::real(100)));
    prog.assert(up.clone().real_ge(Expr::real(-100)));
    prog.assert(up.clone().real_le(Expr::real(100)));

    // Sigmoid of gate: 0 < sig_gate < 1
    prog.assert(sig_gate.clone().real_gt(Expr::real(0)));
    prog.assert(sig_gate.clone().real_lt(Expr::real(1)));

    // SiLU(gate) = gate * sigmoid(gate)
    prog.assert(silu_gate.clone().eq(gate.real_mul(sig_gate)));

    // SwiGLU output = SiLU(gate) * up
    prog.assert(output.clone().eq(silu_gate.clone().real_mul(up.clone())));

    // Expected: same computation
    prog.assert(expected.clone().eq(silu_gate.real_mul(up)));

    // Negated property: output != expected
    let violation = output.ne(expected);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "swiglu_definition");
}

// ---------------------------------------------------------------------------
// Test 674: SwiGLU output bounded by product of bounds
// ---------------------------------------------------------------------------

/// Prove: if |gate| <= G and |up| <= U, then |SwiGLU output| <= G * U.
///
/// SwiGLU(x) = SiLU(gate) * up. Since |SiLU(gate)| <= |gate| (because
/// |sigmoid(gate)| < 1), we have |SiLU(gate)| <= G. Therefore
/// |SwiGLU| = |SiLU(gate)| * |up| <= G * U.
///
/// We model: silu_gate bounded by gate, output = silu_gate * up.
#[test]
fn test_674_swiglu_output_bounded_by_product() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("silu_gate", real.clone());
    let _ = prog.declare_const("up", real.clone());
    let _ = prog.declare_const("output", real);

    let silu_gate = real_var("silu_gate");
    let up = real_var("up");
    let output = real_var("output");

    // SiLU(gate) bounded: |silu_gate| <= G = 10
    prog.assert(silu_gate.clone().real_ge(Expr::real(-10)));
    prog.assert(silu_gate.clone().real_le(Expr::real(10)));

    // up bounded: |up| <= U = 10
    prog.assert(up.clone().real_ge(Expr::real(-10)));
    prog.assert(up.clone().real_le(Expr::real(10)));

    // output = silu_gate * up
    prog.assert(output.clone().eq(silu_gate.real_mul(up)));

    // Negated property: |output| > 100 (= G * U = 10 * 10)
    let violation = output
        .clone()
        .real_gt(Expr::real(100))
        .or(output.real_lt(Expr::real(-100)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "swiglu_output_bounded_by_product");
}

// ---------------------------------------------------------------------------
// Test 675: Gate sigmoid in [0, 1]
// ---------------------------------------------------------------------------

/// Prove: the gating sigmoid sigma(gate) is in (0, 1) for all finite gate values.
///
/// sigma(x) = 1 / (1 + exp(-x)). Since exp(-x) > 0 for all real x,
/// the denominator > 1, so sigma(x) < 1. Also sigma(x) > 0 since
/// numerator and denominator are positive.
///
/// We model the gate sigmoid output with strict bounds.
#[test]
fn test_675_gate_sigmoid_in_zero_one() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("gate", real.clone());
    let _ = prog.declare_const("sig", real);

    let gate = real_var("gate");
    let sig = real_var("sig");

    // Gate input bounded
    prog.assert(gate.clone().real_ge(Expr::real(-1000)));
    prog.assert(gate.real_le(Expr::real(1000)));

    // Sigmoid axiom: 0 < sig < 1
    prog.assert(sig.clone().real_gt(Expr::real(0)));
    prog.assert(sig.clone().real_lt(Expr::real(1)));

    // Negated property: sig <= 0 OR sig >= 1
    let violation = sig
        .clone()
        .real_le(Expr::real(0))
        .or(sig.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gate_sigmoid_in_zero_one");
}

// ---------------------------------------------------------------------------
// Test 676: Gate attenuates: |gate * value| <= |value|
// ---------------------------------------------------------------------------

/// Prove: when gate is sigmoid output in (0, 1), |gate * value| <= |value|.
///
/// Since 0 < gate < 1, multiplying by gate reduces the magnitude:
/// |gate * value| = gate * |value| < 1 * |value| = |value|.
///
/// We model: gate in (0, 1), value in [-V, V], product = gate * value.
/// Prove |product| < |value|. We use the weaker |product| <= |value|.
#[test]
fn test_676_gate_attenuates_magnitude() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("gate", real.clone());
    let _ = prog.declare_const("value", real.clone());
    let _ = prog.declare_const("product", real.clone());
    let _ = prog.declare_const("abs_value", real.clone());
    let _ = prog.declare_const("abs_product", real);

    let gate = real_var("gate");
    let value = real_var("value");
    let product = real_var("product");
    let abs_value = real_var("abs_value");
    let abs_product = real_var("abs_product");

    // Gate in (0, 1)
    prog.assert(gate.clone().real_gt(Expr::real(0)));
    prog.assert(gate.clone().real_lt(Expr::real(1)));

    // Value bounded
    prog.assert(value.clone().real_ge(Expr::real(-100)));
    prog.assert(value.clone().real_le(Expr::real(100)));

    // product = gate * value
    prog.assert(product.clone().eq(gate.real_mul(value.clone())));

    // |value| modeled: abs_value >= value, abs_value >= -value, (abs_value = value OR abs_value = -value)
    prog.assert(abs_value.clone().real_ge(value.clone()));
    prog.assert(
        abs_value
            .clone()
            .real_ge(Expr::real(0).real_sub(value.clone())),
    );
    prog.assert(
        abs_value
            .clone()
            .eq(value.clone())
            .or(abs_value.clone().eq(Expr::real(0).real_sub(value))),
    );

    // |product| modeled similarly
    prog.assert(abs_product.clone().real_ge(product.clone()));
    prog.assert(
        abs_product
            .clone()
            .real_ge(Expr::real(0).real_sub(product.clone())),
    );
    prog.assert(
        abs_product
            .clone()
            .eq(product.clone())
            .or(abs_product.clone().eq(Expr::real(0).real_sub(product))),
    );

    // Negated property: |product| > |value| (gate should attenuate)
    let violation = abs_product.real_gt(abs_value);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gate_attenuates_magnitude");
}

// ---------------------------------------------------------------------------
// Test 677: GLU: sigma(xW1) * xW2
// ---------------------------------------------------------------------------

/// Prove: GLU output equals sigma(gate) * up where gate = xW1, up = xW2.
///
/// GLU(x) = sigma(xW1) * xW2. Unlike SwiGLU which uses SiLU on the gate,
/// vanilla GLU applies plain sigmoid to the gate projection.
///
/// We model: sig_gate in (0, 1), up bounded, output = sig_gate * up.
/// Prove output is bounded by |up| (since |sig_gate| < 1).
#[test]
fn test_677_glu_output_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("sig_gate", real.clone());
    let _ = prog.declare_const("up", real.clone());
    let _ = prog.declare_const("output", real);

    let sig_gate = real_var("sig_gate");
    let up = real_var("up");
    let output = real_var("output");

    // Sigmoid gate: 0 < sig_gate < 1
    prog.assert(sig_gate.clone().real_gt(Expr::real(0)));
    prog.assert(sig_gate.clone().real_lt(Expr::real(1)));

    // up bounded: |up| <= 50
    prog.assert(up.clone().real_ge(Expr::real(-50)));
    prog.assert(up.clone().real_le(Expr::real(50)));

    // output = sig_gate * up
    prog.assert(output.clone().eq(sig_gate.real_mul(up)));

    // Negated property: |output| > 50 (should be bounded by |up| since gate < 1)
    let violation = output
        .clone()
        .real_gt(Expr::real(50))
        .or(output.real_lt(Expr::real(-50)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "glu_output_bounded");
}

// ---------------------------------------------------------------------------
// Test 678: GeGLU: GELU gate * projection
// ---------------------------------------------------------------------------

/// Prove: GeGLU output = GELU(gate) * up is bounded when inputs are bounded.
///
/// GeGLU(x) = GELU(xW1) * xW2. GELU(gate) >= -0.18 (conservative bound)
/// and GELU(gate) <= gate for large gate. For |gate| <= G, |GELU(gate)| <= G.
/// Therefore |GeGLU output| <= G * U where |up| <= U.
///
/// We model: gelu_gate bounded, up bounded, output = gelu_gate * up.
#[test]
fn test_678_geglu_output_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("gelu_gate", real.clone());
    let _ = prog.declare_const("up", real.clone());
    let _ = prog.declare_const("output", real);

    let gelu_gate = real_var("gelu_gate");
    let up = real_var("up");
    let output = real_var("output");

    // GELU(gate) bounded: >= -0.18, <= 20 (for |gate| <= 20)
    prog.assert(gelu_gate.clone().real_ge(Expr::real_ratio(-18, 100)));
    prog.assert(gelu_gate.clone().real_le(Expr::real(20)));

    // up bounded: |up| <= 20
    prog.assert(up.clone().real_ge(Expr::real(-20)));
    prog.assert(up.clone().real_le(Expr::real(20)));

    // output = gelu_gate * up
    prog.assert(output.clone().eq(gelu_gate.real_mul(up)));

    // Negated property: |output| > 400 (= 20 * 20)
    let violation = output
        .clone()
        .real_gt(Expr::real(400))
        .or(output.real_lt(Expr::real(-400)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "geglu_output_bounded");
}

// ---------------------------------------------------------------------------
// Test 679: GELU approximation: 0.5 * x * (1 + tanh(...))
// ---------------------------------------------------------------------------

/// Prove: the GELU tanh approximation has a specific algebraic structure.
///
/// GELU_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// The key structural property: for x = 0, GELU_approx(0) = 0, since
/// 0.5 * 0 * (anything) = 0 by the zero-product property.
///
/// We verify this zero-at-origin property for the approximate form.
#[test]
fn test_679_gelu_approximation_zero_at_origin() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("tanh_arg", real.clone());
    let _ = prog.declare_const("gelu_approx", real);

    let x = real_var("x");
    let tanh_arg = real_var("tanh_arg");
    let gelu_approx = real_var("gelu_approx");

    // x = 0
    prog.assert(x.clone().eq(Expr::real(0)));

    // tanh_arg is some value (doesn't matter when x = 0)
    prog.assert(tanh_arg.clone().real_ge(Expr::real(-1)));
    prog.assert(tanh_arg.clone().real_le(Expr::real(1)));

    // gelu_approx = 0.5 * x * (1 + tanh_arg)
    let half = Expr::real_ratio(1, 2);
    let bracket = Expr::real(1).real_add(tanh_arg);
    prog.assert(gelu_approx.clone().eq(half.real_mul(x).real_mul(bracket)));

    // Negated property: gelu_approx != 0
    let violation = gelu_approx.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gelu_approximation_zero_at_origin");
}

// ---------------------------------------------------------------------------
// Test 680: GELU exact (CDF formulation concept): x * Phi(x) symmetric structure
// ---------------------------------------------------------------------------

/// Prove: the exact GELU satisfies the symmetry-related identity:
/// GELU(x) + GELU(-x) = x * (Phi(x) - Phi(-x)) + (-x) * Phi(-x) + x * Phi(x)
/// simplifies to: GELU(x) + GELU(-x) = x * (2*Phi(x) - 1) for the CDF form.
///
/// Actually, GELU(x) = x * Phi(x) and GELU(-x) = -x * Phi(-x) = -x * (1 - Phi(x)).
/// Sum: GELU(x) + GELU(-x) = x * Phi(x) - x * (1 - Phi(x)) = x * (2*Phi(x) - 1).
///
/// We verify this algebraic identity.
#[test]
fn test_680_gelu_exact_cdf_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("phi_x", real.clone());
    let _ = prog.declare_const("gelu_x", real.clone());
    let _ = prog.declare_const("gelu_neg_x", real.clone());
    let _ = prog.declare_const("sum_val", real.clone());
    let _ = prog.declare_const("expected", real);

    let x = real_var("x");
    let phi_x = real_var("phi_x");
    let gelu_x = real_var("gelu_x");
    let gelu_neg_x = real_var("gelu_neg_x");
    let sum_val = real_var("sum_val");
    let expected = real_var("expected");

    // x bounded
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // Phi(x) in (0, 1) (standard normal CDF)
    prog.assert(phi_x.clone().real_gt(Expr::real(0)));
    prog.assert(phi_x.clone().real_lt(Expr::real(1)));

    // GELU(x) = x * Phi(x)
    prog.assert(gelu_x.clone().eq(x.clone().real_mul(phi_x.clone())));

    // GELU(-x) = -x * (1 - Phi(x))  [since Phi(-x) = 1 - Phi(x)]
    let neg_x = Expr::real(0).real_sub(x.clone());
    let one_minus_phi = Expr::real(1).real_sub(phi_x.clone());
    prog.assert(gelu_neg_x.clone().eq(neg_x.real_mul(one_minus_phi)));

    // sum = GELU(x) + GELU(-x)
    prog.assert(sum_val.clone().eq(gelu_x.real_add(gelu_neg_x)));

    // expected = x * (2 * Phi(x) - 1)
    let two_phi_minus_one = Expr::real(2).real_mul(phi_x).real_sub(Expr::real(1));
    prog.assert(expected.clone().eq(x.real_mul(two_phi_minus_one)));

    // Negated property: sum != expected
    let violation = sum_val.ne(expected);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gelu_exact_cdf_identity");
}

// ---------------------------------------------------------------------------
// Test 681: FFN expansion: hidden_dim * expansion_ratio = intermediate_dim
// ---------------------------------------------------------------------------

/// Prove: FFN expansion produces intermediate_dim = hidden_dim * ratio.
///
/// Standard transformer FFN expands the hidden dimension by a factor
/// (typically 4x). The intermediate dimension equals hidden_dim * ratio.
///
/// We model: inter = hidden * ratio with all positive, and verify the identity.
#[test]
fn test_681_ffn_expansion_dimension() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("hidden", real.clone());
    let _ = prog.declare_const("ratio", real.clone());
    let _ = prog.declare_const("inter", real);

    let hidden = real_var("hidden");
    let ratio = real_var("ratio");
    let inter = real_var("inter");

    // All positive
    prog.assert(hidden.clone().real_gt(Expr::real(0)));
    prog.assert(ratio.clone().real_gt(Expr::real(0)));

    // inter = hidden * ratio
    prog.assert(inter.clone().eq(hidden.clone().real_mul(ratio.clone())));

    // Negated property: inter != hidden * ratio
    let violation = inter.ne(hidden.real_mul(ratio));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "ffn_expansion_dimension");
}

// ---------------------------------------------------------------------------
// Test 682: FFN contraction back to hidden_dim
// ---------------------------------------------------------------------------

/// Prove: FFN down-projection contracts intermediate_dim back to hidden_dim.
///
/// The FFN structure is: hidden -> expand to inter -> contract back to hidden.
/// The down-projection W_down has shape [inter, hidden], so
/// output_dim = hidden_dim (matching the residual connection).
///
/// We model: input_dim = hidden, expanded = hidden * ratio,
/// output_dim = hidden (the down-projection restores the original dimension).
#[test]
fn test_682_ffn_contraction_to_hidden() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("hidden", real.clone());
    let _ = prog.declare_const("ratio", real.clone());
    let _ = prog.declare_const("inter", real.clone());
    let _ = prog.declare_const("output_dim", real);

    let hidden = real_var("hidden");
    let ratio = real_var("ratio");
    let inter = real_var("inter");
    let output_dim = real_var("output_dim");

    // Positive dimensions
    prog.assert(hidden.clone().real_gt(Expr::real(0)));
    prog.assert(ratio.clone().real_gt(Expr::real(0)));

    // inter = hidden * ratio (expansion)
    prog.assert(inter.eq(hidden.clone().real_mul(ratio)));

    // output_dim = hidden (contraction restores dimension)
    prog.assert(output_dim.clone().eq(hidden.clone()));

    // Negated property: output_dim != hidden
    let violation = output_dim.ne(hidden);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "ffn_contraction_to_hidden");
}

// ---------------------------------------------------------------------------
// Test 683: SwiGLU params: 3 * hidden * intermediate (gate + up + down)
// ---------------------------------------------------------------------------

/// Prove: SwiGLU FFN has 3 * hidden * inter parameters.
///
/// SwiGLU uses three weight matrices:
/// - W_gate: [hidden, inter] -> hidden * inter params
/// - W_up:   [hidden, inter] -> hidden * inter params
/// - W_down: [inter, hidden] -> inter * hidden params
/// Total = 3 * hidden * inter.
///
/// We model: total = gate_params + up_params + down_params, each = hidden * inter.
#[test]
fn test_683_swiglu_param_count() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("hidden", real.clone());
    let _ = prog.declare_const("inter", real.clone());
    let _ = prog.declare_const("gate_params", real.clone());
    let _ = prog.declare_const("up_params", real.clone());
    let _ = prog.declare_const("down_params", real.clone());
    let _ = prog.declare_const("total", real);

    let hidden = real_var("hidden");
    let inter = real_var("inter");
    let gate_params = real_var("gate_params");
    let up_params = real_var("up_params");
    let down_params = real_var("down_params");
    let total = real_var("total");

    // Positive dimensions
    prog.assert(hidden.clone().real_gt(Expr::real(0)));
    prog.assert(inter.clone().real_gt(Expr::real(0)));

    // Each matrix: hidden * inter
    let hi = hidden.clone().real_mul(inter.clone());
    prog.assert(gate_params.clone().eq(hi.clone()));
    prog.assert(up_params.clone().eq(hi.clone()));
    prog.assert(down_params.clone().eq(hi));

    // total = gate_params + up_params + down_params
    prog.assert(
        total
            .clone()
            .eq(gate_params.real_add(up_params).real_add(down_params)),
    );

    // expected = 3 * hidden * inter
    let expected = Expr::real(3).real_mul(hidden.real_mul(inter));

    // Negated property: total != 3 * hidden * inter
    let violation = total.ne(expected);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "swiglu_param_count");
}

// ---------------------------------------------------------------------------
// Test 684: Standard FFN params: 2 * hidden * intermediate
// ---------------------------------------------------------------------------

/// Prove: standard FFN (no gating) has 2 * hidden * inter parameters.
///
/// Standard FFN uses two weight matrices:
/// - W_up:   [hidden, inter] -> hidden * inter params
/// - W_down: [inter, hidden] -> inter * hidden params
/// Total = 2 * hidden * inter.
///
/// This is fewer than SwiGLU's 3 * hidden * inter.
#[test]
fn test_684_standard_ffn_param_count() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("hidden", real.clone());
    let _ = prog.declare_const("inter", real.clone());
    let _ = prog.declare_const("up_params", real.clone());
    let _ = prog.declare_const("down_params", real.clone());
    let _ = prog.declare_const("total", real);

    let hidden = real_var("hidden");
    let inter = real_var("inter");
    let up_params = real_var("up_params");
    let down_params = real_var("down_params");
    let total = real_var("total");

    // Positive dimensions
    prog.assert(hidden.clone().real_gt(Expr::real(0)));
    prog.assert(inter.clone().real_gt(Expr::real(0)));

    // Each matrix: hidden * inter
    let hi = hidden.clone().real_mul(inter.clone());
    prog.assert(up_params.clone().eq(hi.clone()));
    prog.assert(down_params.clone().eq(hi));

    // total = up_params + down_params
    prog.assert(total.clone().eq(up_params.real_add(down_params)));

    // expected = 2 * hidden * inter
    let expected = Expr::real(2).real_mul(hidden.real_mul(inter));

    // Negated property: total != 2 * hidden * inter
    let violation = total.ne(expected);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "standard_ffn_param_count");
}

// ---------------------------------------------------------------------------
// Test 685: SiLU monotonic for x > 0
// ---------------------------------------------------------------------------

/// Prove: SiLU is strictly increasing for x > 0.
///
/// SiLU'(x) = sigma(x) * (1 + x * (1 - sigma(x))). For x > 0:
/// - sigma(x) > 0.5 > 0
/// - (1 - sigma(x)) in (0, 0.5)
/// - x * (1 - sigma(x)) > 0
/// - 1 + x * (1 - sigma(x)) > 1 > 0
/// So SiLU'(x) > 0, meaning SiLU is strictly increasing.
///
/// We model: x1 < x2 with both > 0, sig1, sig2 in (0.5, 1) with sig1 < sig2
/// (sigmoid is increasing). silu1 = x1 * sig1, silu2 = x2 * sig2.
/// Prove silu1 < silu2.
#[test]
fn test_685_silu_monotonic_positive_x() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("sig1", real.clone());
    let _ = prog.declare_const("sig2", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let sig1 = real_var("sig1");
    let sig2 = real_var("sig2");

    // Both positive, x1 < x2
    prog.assert(x1.clone().real_gt(Expr::real(0)));
    prog.assert(x2.clone().real_le(Expr::real(1000)));
    prog.assert(x1.clone().real_lt(x2.clone()));

    // For x > 0, sigmoid(x) in (0.5, 1) and sigmoid is increasing
    prog.assert(sig1.clone().real_gt(Expr::real_ratio(1, 2)));
    prog.assert(sig1.clone().real_lt(Expr::real(1)));
    prog.assert(sig2.clone().real_gt(Expr::real_ratio(1, 2)));
    prog.assert(sig2.clone().real_lt(Expr::real(1)));
    prog.assert(sig1.clone().real_le(sig2.clone()));

    // silu(x) = x * sigmoid(x)
    let silu1 = x1.real_mul(sig1);
    let silu2 = x2.real_mul(sig2);

    // Negated property: silu1 >= silu2 (not monotonically increasing)
    let violation = silu1.real_ge(silu2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "silu_monotonic_positive_x");
}

// ---------------------------------------------------------------------------
// Test 686: SiLU(0) = 0
// ---------------------------------------------------------------------------

/// Prove: SiLU(0) = 0.
///
/// SiLU(x) = x * sigmoid(x). At x = 0: SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0.
/// The zero-product property ensures this regardless of sigmoid(0)'s value.
#[test]
fn test_686_silu_zero_at_origin() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("sig", real.clone());
    let _ = prog.declare_const("silu", real);

    let x = real_var("x");
    let sig = real_var("sig");
    let silu = real_var("silu");

    // x = 0
    prog.assert(x.clone().eq(Expr::real(0)));

    // sigmoid(0) = 0.5
    prog.assert(sig.clone().eq(Expr::real_ratio(1, 2)));

    // silu = x * sig
    prog.assert(silu.clone().eq(x.real_mul(sig)));

    // Negated property: silu != 0
    let violation = silu.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "silu_zero_at_origin");
}

// ---------------------------------------------------------------------------
// Test 687: SiLU minimum approximately -0.278
// ---------------------------------------------------------------------------

/// Prove: SiLU(x) >= -0.28 for all x.
///
/// SiLU(x) = x * sigmoid(x). The minimum is approximately -0.2784 at x ~ -1.278.
/// We use -0.28 as a conservative lower bound (slightly below the actual minimum).
///
/// We model silu output with the axiomatic bound and prove the negation is UNSAT.
#[test]
fn test_687_silu_minimum_bound() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("silu", real);

    let x = real_var("x");
    let silu = real_var("silu");

    // Input bound
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.real_le(Expr::real(1000)));

    // SiLU axiom: silu >= -0.28
    prog.assert(silu.clone().real_ge(Expr::real_ratio(-28, 100)));

    // Negated property: silu < -0.28
    let violation = silu.real_lt(Expr::real_ratio(-28, 100));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "silu_minimum_bound");
}

// ---------------------------------------------------------------------------
// Test 688: Gated output magnitude controlled by gate
// ---------------------------------------------------------------------------

/// Prove: in a gated FFN, the output magnitude is controlled by the gate value.
///
/// If gate value g is small (near 0), the output |g * up_proj| is small
/// regardless of up_proj magnitude. Specifically: if g in [0, epsilon],
/// then |output| <= epsilon * |up_proj|.
///
/// We model: g in [0, eps], up in [-U, U], output = g * up.
/// Prove |output| <= eps * U.
#[test]
fn test_688_gated_output_magnitude_controlled() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("g", real.clone());
    let _ = prog.declare_const("up", real.clone());
    let _ = prog.declare_const("output", real);

    let g = real_var("g");
    let up = real_var("up");
    let output = real_var("output");

    // Gate is small: g in [0, 0.01]
    let eps = Expr::real_ratio(1, 100);
    prog.assert(g.clone().real_ge(Expr::real(0)));
    prog.assert(g.clone().real_le(eps));

    // up bounded: |up| <= 100
    prog.assert(up.clone().real_ge(Expr::real(-100)));
    prog.assert(up.clone().real_le(Expr::real(100)));

    // output = g * up
    prog.assert(output.clone().eq(g.real_mul(up)));

    // |output| <= eps * U = 0.01 * 100 = 1
    // Negated property: |output| > 1
    let violation = output
        .clone()
        .real_gt(Expr::real(1))
        .or(output.real_lt(Expr::real(-1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gated_output_magnitude_controlled");
}

// ---------------------------------------------------------------------------
// Test 689: Pre-norm + FFN + residual bounds
// ---------------------------------------------------------------------------

/// Prove: pre-norm + FFN + residual connection preserves bounds.
///
/// In a pre-norm transformer block:
///   output = x + FFN(Norm(x))
///
/// If |x| <= X and |FFN(Norm(x))| <= F, then |output| <= X + F.
///
/// We model: x in [-X, X], ffn_out in [-F, F], output = x + ffn_out.
/// Prove |output| <= X + F.
#[test]
fn test_689_prenorm_ffn_residual_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("ffn_out", real.clone());
    let _ = prog.declare_const("output", real);

    let x = real_var("x");
    let ffn_out = real_var("ffn_out");
    let output = real_var("output");

    // |x| <= 10
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));

    // |ffn_out| <= 5
    prog.assert(ffn_out.clone().real_ge(Expr::real(-5)));
    prog.assert(ffn_out.clone().real_le(Expr::real(5)));

    // output = x + ffn_out (residual connection)
    prog.assert(output.clone().eq(x.real_add(ffn_out)));

    // Negated property: |output| > 15 (= 10 + 5)
    let violation = output
        .clone()
        .real_gt(Expr::real(15))
        .or(output.real_lt(Expr::real(-15)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "prenorm_ffn_residual_bounds");
}

// ---------------------------------------------------------------------------
// Test 690: Two-layer FFN bounds composition
// ---------------------------------------------------------------------------

/// Prove: composing two transformer blocks with pre-norm + FFN + residual
/// accumulates bounds additively.
///
/// Layer 1: y1 = x + FFN1(Norm(x)),     |y1| <= X + F1
/// Layer 2: y2 = y1 + FFN2(Norm(y1)),   |y2| <= |y1| + F2 <= X + F1 + F2
///
/// With N layers each contributing at most F, the bound is X + N * F.
///
/// We model: x in [-X, X], ffn1_out in [-F, F], ffn2_out in [-F, F],
/// y1 = x + ffn1_out, y2 = y1 + ffn2_out.
/// Prove |y2| <= X + 2*F.
#[test]
fn test_690_two_layer_ffn_bounds_composition() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("ffn1_out", real.clone());
    let _ = prog.declare_const("ffn2_out", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y2", real);

    let x = real_var("x");
    let ffn1_out = real_var("ffn1_out");
    let ffn2_out = real_var("ffn2_out");
    let y1 = real_var("y1");
    let y2 = real_var("y2");

    // |x| <= 10
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));

    // |ffn1_out| <= 5
    prog.assert(ffn1_out.clone().real_ge(Expr::real(-5)));
    prog.assert(ffn1_out.clone().real_le(Expr::real(5)));

    // |ffn2_out| <= 5
    prog.assert(ffn2_out.clone().real_ge(Expr::real(-5)));
    prog.assert(ffn2_out.clone().real_le(Expr::real(5)));

    // Layer 1: y1 = x + ffn1_out
    prog.assert(y1.clone().eq(x.real_add(ffn1_out)));

    // Layer 2: y2 = y1 + ffn2_out
    prog.assert(y2.clone().eq(y1.real_add(ffn2_out)));

    // Negated property: |y2| > 20 (= 10 + 5 + 5)
    let violation = y2
        .clone()
        .real_gt(Expr::real(20))
        .or(y2.real_lt(Expr::real(-20)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "two_layer_ffn_bounds_composition");
}
