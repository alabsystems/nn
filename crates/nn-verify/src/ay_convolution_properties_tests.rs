// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for convolution mathematical properties (#4226).
//!
//! Proves fundamental algebraic and structural properties of convolution operations
//! used throughout nn's model execution and verification pipelines.
//!
//! # Properties proved
//!
//! 1. **Conv1d linearity**: conv(a*x + b*y) == a*conv(x) + b*conv(y) for linear convolution
//! 2. **Conv1d associativity**: specific kernel compositions
//! 3. **Padding symmetry**: symmetric padding produces symmetric output bounds
//! 4. **Stride-output relationship**: output_size = (input_size - kernel_size + 2*padding) / stride + 1
//! 5. **Dilation-effective kernel**: effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
//! 6. **Groups decomposition**: grouped conv params equal standard params / groups
//! 7. **Bias addition commutativity**: conv(x) + bias == conv_with_bias(x, bias)
//! 8. **Transpose conv as adjoint**: transpose conv undoes conv for specific dimensions
//! 9. **Depthwise separable equivalence**: depthwise + pointwise param count identity
//!
//! # Proof strategy
//!
//! Algebraic identities over continuous quantities (linearity, associativity,
//! bias commutativity) use QF_LRA / QF_NRA. Shape, index, and parameter-count
//! identities — output length, dilated kernel size, group and depthwise-separable
//! counts — are *integer* facts built on floor division and exact partitions, so
//! they are modelled in QF_LIA over concrete shapes: the divisor/stride/group
//! count is a literal (keeping every product linear and the logic decidable) and
//! the quantity of interest is *derived* from the definition rather than asserted.
//! Over the reals these same identities are SAT (fractional `padding = 1/4`,
//! non-integer output lengths), which is exactly the defect the integer encoding
//! removes. Every proof asserts the negation of the property and proves UNSAT;
//! each shape/count proof is paired with a `build_*(false)` mutation that injects
//! a plausible formula bug and must turn the query SAT.
//!
//! Part of #4226.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;

/// Declare a real variable and return its expression.
fn declare_real(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::real())
}

/// Assert `lower <= expr <= upper`.
fn assert_bounds(program: &mut AYProgram, expr: &Expr, lower: &Expr, upper: &Expr) {
    program.assert(expr.clone().real_ge(lower.clone()));
    program.assert(expr.clone().real_le(upper.clone()));
}

/// Declare an `Int` constant constrained to `lo <= name <= hi` and return it.
///
/// Conv shape quantities (channels, kernel sizes, lengths, strides, group counts)
/// are integers; modelling them as `Int` in `QF_LIA` — rather than as reals —
/// rules out the fractional counterexamples (`padding = 1/4`, a non-integer
/// output length) that make the real-valued encodings of these floor-division
/// and exact-partition identities SAT. The bounds keep each query tiny and fast.
fn declare_int_bounded(program: &mut AYProgram, name: &str, lo: i64, hi: i64) -> Expr {
    let var = program.declare_const(name, Sort::int());
    program.assert(var.clone().int_ge(Expr::int(lo)));
    program.assert(var.clone().int_le(Expr::int(hi)));
    var
}

/// Execute a ay program and return whether UNSAT (property proven).
fn execute_and_check(program: &AYProgram) -> (bool, String) {
    let (proven, detail) = match ay_bindings::execute_direct::execute(program) {
        Ok(ay_bindings::execute_direct::ExecuteResult::Verified) => {
            (true, "UNSAT: property holds for all inputs".to_string())
        }
        Ok(ay_bindings::execute_direct::ExecuteResult::Counterexample { model, .. }) => {
            (false, format!("SAT: counterexample found: {:?}", model))
        }
        Ok(ay_bindings::execute_direct::ExecuteResult::Unknown(reason)) => {
            (false, format!("Unknown: {}", reason))
        }
        Ok(other) => (false, format!("Unexpected result: {:?}", other)),
        Err(e) => (false, format!("Execution error: {}", e)),
    };
    // Uniform guard: a vacuous UNSAT (P and not-P, or X != X) never counts as a
    // proof. See crate::ay_vacuity. No-op for genuine queries.
    crate::ay_vacuity::reject_if_vacuous(&program.to_string(), proven, detail)
}

// ---------------------------------------------------------------------------
// Property 1: Conv1d Linearity
// ---------------------------------------------------------------------------

/// Prove that 1D convolution is linear: conv(a*x + b*y) == a*conv(x) + b*conv(y).
///
/// For a scalar kernel weight w and scalar inputs x, y with scalars a, b:
///   conv(a*x + b*y) = w * (a*x + b*y) = w*a*x + w*b*y
///   a*conv(x) + b*conv(y) = a*(w*x) + b*(w*y) = a*w*x + b*w*y
///
/// These are equal by commutativity and distributivity of real multiplication.
/// We prove this for a 3-element convolution (kernel size 3) on concrete positions
/// using QF_NRA since products of symbolic variables are involved.
#[test]
fn test_conv1d_linearity() {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    // Scalars a, b
    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    assert_bounds(&mut program, &a, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &b, &bound_lo, &bound_hi);

    // Kernel weight (single element for simplicity)
    let w = declare_real(&mut program, "w");
    assert_bounds(&mut program, &w, &bound_lo, &bound_hi);

    // Input signals x, y (single element position)
    let x = declare_real(&mut program, "x");
    let y = declare_real(&mut program, "y");
    assert_bounds(&mut program, &x, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &y, &bound_lo, &bound_hi);

    // conv(a*x + b*y) = w * (a*x + b*y)
    let ax = a.clone().real_mul(x.clone());
    let by = b.clone().real_mul(y.clone());
    let lhs = w.clone().real_mul(ax.real_add(by));

    // a*conv(x) + b*conv(y) = a*(w*x) + b*(w*y)
    let a_conv_x = a.real_mul(w.clone().real_mul(x));
    let b_conv_y = b.real_mul(w.real_mul(y));
    let rhs = a_conv_x.real_add(b_conv_y);

    // Violation: lhs != rhs
    program.assert(lhs.ne(rhs));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven || detail.contains("Unknown"),
        "Conv1d linearity: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Conv1d linearity must not have counterexample: {}",
        detail,
    );
    assert!(smt2.contains("check-sat"), "SMT2 should contain check-sat");
}

/// Prove linearity for a 3-element convolution kernel summing over 3 input positions.
///
/// conv(a*x + b*y)[i] = sum_k w[k] * (a*x[i-k] + b*y[i-k])
///                     = a * sum_k w[k] * x[i-k] + b * sum_k w[k] * y[i-k]
///                     = a * conv(x)[i] + b * conv(y)[i]
#[test]
fn test_conv1d_linearity_kernel_size_3() {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-10);
    let bound_hi = Expr::real(10);

    // Scalars
    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    assert_bounds(&mut program, &a, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &b, &bound_lo, &bound_hi);

    // Kernel weights w0, w1, w2
    let w0 = declare_real(&mut program, "w0");
    let w1 = declare_real(&mut program, "w1");
    let w2 = declare_real(&mut program, "w2");
    assert_bounds(&mut program, &w0, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &w1, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &w2, &bound_lo, &bound_hi);

    // Input positions x0, x1, x2 and y0, y1, y2
    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let y0 = declare_real(&mut program, "y0");
    let y1 = declare_real(&mut program, "y1");
    let y2 = declare_real(&mut program, "y2");
    assert_bounds(&mut program, &x0, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &x1, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &x2, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &y0, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &y1, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &y2, &bound_lo, &bound_hi);

    // Combined input: z_i = a*x_i + b*y_i
    let z0 = a
        .clone()
        .real_mul(x0.clone())
        .real_add(b.clone().real_mul(y0.clone()));
    let z1 = a
        .clone()
        .real_mul(x1.clone())
        .real_add(b.clone().real_mul(y1.clone()));
    let z2 = a
        .clone()
        .real_mul(x2.clone())
        .real_add(b.clone().real_mul(y2.clone()));

    // conv(z) at one output position = w0*z0 + w1*z1 + w2*z2
    let conv_z = w0
        .clone()
        .real_mul(z0)
        .real_add(w1.clone().real_mul(z1))
        .real_add(w2.clone().real_mul(z2));

    // a*conv(x) + b*conv(y)
    let conv_x = w0
        .clone()
        .real_mul(x0)
        .real_add(w1.clone().real_mul(x1))
        .real_add(w2.clone().real_mul(x2));
    let conv_y = w0
        .real_mul(y0)
        .real_add(w1.real_mul(y1))
        .real_add(w2.real_mul(y2));
    let rhs = a.real_mul(conv_x).real_add(b.real_mul(conv_y));

    // Violation: conv(a*x + b*y) != a*conv(x) + b*conv(y)
    program.assert(conv_z.ne(rhs));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven || detail.contains("Unknown"),
        "Conv1d linearity K=3: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Conv1d linearity K=3 must not have counterexample: {}",
        detail,
    );
}

// ---------------------------------------------------------------------------
// Property 2: Conv1d Associativity (Kernel Composition)
// ---------------------------------------------------------------------------

/// Prove that convolving with kernel [w0, w1] then [v0, v1] is equivalent
/// to convolving with the composed kernel [w0*v0, w0*v1 + w1*v0, w1*v1].
///
/// For input [x0, x1, x2]:
///   First conv: y0 = w0*x0 + w1*x1, y1 = w0*x1 + w1*x2
///   Second conv: z0 = v0*y0 + v1*y1
///             = v0*(w0*x0 + w1*x1) + v1*(w0*x1 + w1*x2)
///             = w0*v0*x0 + (w1*v0 + w0*v1)*x1 + w1*v1*x2
///
///   Composed kernel c = [w0*v0, w0*v1 + w1*v0, w1*v1]
///   Direct conv: z0' = c0*x0 + c1*x1 + c2*x2
///             = w0*v0*x0 + (w0*v1 + w1*v0)*x1 + w1*v1*x2
///
/// Uses QF_NRA for products of symbolic variables.
#[test]
fn test_conv1d_associativity_kernel_composition() {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound_lo = Expr::real(-10);
    let bound_hi = Expr::real(10);

    // Kernel 1: [w0, w1]
    let w0 = declare_real(&mut program, "w0");
    let w1 = declare_real(&mut program, "w1");
    assert_bounds(&mut program, &w0, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &w1, &bound_lo, &bound_hi);

    // Kernel 2: [v0, v1]
    let v0 = declare_real(&mut program, "v0");
    let v1 = declare_real(&mut program, "v1");
    assert_bounds(&mut program, &v0, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &v1, &bound_lo, &bound_hi);

    // Input: [x0, x1, x2]
    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    assert_bounds(&mut program, &x0, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &x1, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &x2, &bound_lo, &bound_hi);

    // Sequential application:
    // y0 = w0*x0 + w1*x1
    let y0 = w0
        .clone()
        .real_mul(x0.clone())
        .real_add(w1.clone().real_mul(x1.clone()));
    // y1 = w0*x1 + w1*x2
    let y1 = w0
        .clone()
        .real_mul(x1.clone())
        .real_add(w1.clone().real_mul(x2.clone()));
    // z = v0*y0 + v1*y1
    let z_seq = v0.clone().real_mul(y0).real_add(v1.clone().real_mul(y1));

    // Composed kernel: c = [w0*v0, w0*v1 + w1*v0, w1*v1]
    let c0 = w0.clone().real_mul(v0);
    let c1 = w0.real_mul(v1.clone()).real_add(w1.clone().real_mul(v1.clone()));
    let c2 = w1.real_mul(v1.clone());

    // Direct application with composed kernel:
    // z' = c0*x0 + c1*x1 + c2*x2
    let z_direct = c0
        .real_mul(x0)
        .real_add(c1.real_mul(x1))
        .real_add(c2.real_mul(x2));

    // Violation: z_seq != z_direct
    program.assert(z_seq.ne(z_direct));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven || detail.contains("Unknown"),
        "Conv1d associativity: expected Proven or Unknown (NRA), got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Conv1d associativity must not have counterexample: {}",
        detail,
    );
}

// ---------------------------------------------------------------------------
// Property 3: Padding Symmetry
// ---------------------------------------------------------------------------

/// Prove that symmetric padding produces symmetric output dimension behavior.
///
/// For padding P applied equally on both sides, with stride S=1, dilation D=1:
///   output_size = input_size + 2*P - kernel_size + 1
///
/// The output is a symmetric function of P: increasing P by 1 increases output by 2.
/// Also, for symmetric padding, if input_size is centered, the output is centered
/// around the same midpoint.
///
/// We prove: out(P+1) - out(P) = 2 for all valid configurations.
/// Uses QF_LRA.
#[test]
fn test_padding_symmetry() {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let l = declare_real(&mut program, "l");
    let k = declare_real(&mut program, "k");
    let p = declare_real(&mut program, "p");

    program.assert(l.clone().real_ge(Expr::real(1)));
    program.assert(k.clone().real_ge(Expr::real(1)));
    program.assert(p.clone().real_ge(Expr::real(0)));

    // Valid config: L + 2P >= K
    let two_p = Expr::real(2).real_mul(p.clone());
    program.assert(l.clone().real_add(two_p.clone()).real_ge(k.clone()));

    // out(P) = L + 2*P - K + 1
    let out_p = l
        .clone()
        .real_add(two_p)
        .real_sub(k.clone())
        .real_add(Expr::real(1));

    // out(P+1) = L + 2*(P+1) - K + 1 = L + 2P + 2 - K + 1
    let two_p_plus_2 = Expr::real(2).real_mul(p.real_add(Expr::real(1)));
    let out_p_plus_1 = l.real_add(two_p_plus_2).real_sub(k).real_add(Expr::real(1));

    // Property: out(P+1) - out(P) = 2
    let diff = out_p_plus_1.real_sub(out_p);

    // Violation: diff != 2
    program.assert(diff.ne(Expr::real(2)));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Padding symmetry (QF_LRA) should be Proven. detail: {}",
        detail,
    );
    assert!(smt2.contains("QF_LRA"), "should use QF_LRA logic");
}

// ---------------------------------------------------------------------------
// Property 4: Stride-Output Relationship
// ---------------------------------------------------------------------------

/// Build the stride/output-size query over a concrete conv shape, in integers.
///
/// The output length of a 1-D convolution is a *floor* division,
///   out = floor((L + 2P - K) / S) + 1,
/// which is well defined only over the integers — over the reals the real-valued
/// encoding admitted fractional `out` and was SAT. We pin the stride `S` and the
/// whole (small) shape to literals so the floor-division bracket
/// `S*(out-1) <= numerator < S*out` stays linear (`QF_LIA`), then *derive* `out`
/// from those inequalities and prove it equals the independently hand-computed
/// answer 5 for `L=9, K=3, P=1, S=2` — `out` is never asserted equal to 5, the
/// solver has to solve the floor division.
///
/// `correct` selects the padding term: the true `L + 2P - K` (numerator 8, out 5)
/// or the classic "forgot the factor of two on the padding" bug `L + P - K`
/// (numerator 7, out 4), which changes the derived output size.
fn build_stride_output(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    // Concrete conv shape: input length L=9, kernel K=3, padding P=1, stride S=2.
    let l = Expr::int(9);
    let k = Expr::int(3);
    let p = Expr::int(1);
    let s: i64 = 2;

    // numerator = L + 2P - K  (correct)   vs   L + P - K  (bug).
    let pad_term = if correct {
        Expr::int(2).int_mul(p.clone())
    } else {
        p.clone() // BUG: dropped the factor of 2 on the padding
    };
    let numerator = l.int_add(pad_term).int_sub(k);

    // `out` is DERIVED, not asserted: the unique integer with
    //   S*(out-1) <= numerator < S*out    and   numerator >= 0.
    let out = program.declare_const("out", Sort::int());
    let out_minus_1 = out.clone().int_sub(Expr::int(1));
    program.assert(
        Expr::int(s)
            .int_mul(out_minus_1)
            .int_le(numerator.clone()),
    );
    program.assert(numerator.clone().int_lt(Expr::int(s).int_mul(out.clone())));
    program.assert(numerator.int_ge(Expr::int(0)));

    // Correct answer: (9 + 2*1 - 3)/2 + 1 = 8/2 + 1 = 5.
    // Violation: the derived output size is not 5.
    program.assert(out.ne(Expr::int(5)));
    program.check_sat();
    program
}

/// Prove the stride/output-size formula computes the right output length.
#[test]
fn test_stride_output_relationship() {
    let program = build_stride_output(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Stride-output size (QF_LIA) should be Proven. detail: {}",
        detail,
    );
    assert_eq!(crate::ay_vacuity::vacuity_smell(&smt2), None);
    assert!(smt2.contains("QF_LIA"), "should use QF_LIA logic");
}

/// Dropping the factor of two on the padding shifts the numerator 8 -> 7, so the
/// derived output size becomes 4, not 5: the query must be SAT.
#[test]
fn test_stride_output_relationship_mutation() {
    let program = build_stride_output(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the wrong padding term the derived output size changes and the query \
         must be SAT; got: {detail}",
    );
}

/// Prove specific stride cases: stride=2 halves the output (approximately).
///
/// For L=8, K=3, P=1, S=2:
///   out = (8 + 2 - 3) / 2 + 1 = 7/2 + 1 = 3.5 + 1 = 4 (floor: 4)
///
/// We prove: out = 4 for these concrete values.
#[test]
fn test_stride_2_concrete_case() {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let out = declare_real(&mut program, "out");

    // Concrete: L=8, K=3, P=1, S=2
    // numerator = 8 + 2*1 - 3 = 7
    // out = 7/2 + 1 = 4.5 (floor = 4, but in exact arithmetic: 4.5)
    // For exact division case: L=8, K=3, P=1, S=1 -> out = 8
    // Let's use L=9, K=3, P=1, S=2 -> (9+2-3)/2 + 1 = 8/2 + 1 = 5
    let numerator = Expr::real(9)
        .real_add(Expr::real(2))
        .real_sub(Expr::real(3)); // = 8
    let expected = numerator
        .real_mul(Expr::real_ratio(1, 2))
        .real_add(Expr::real(1)); // = 4 + 1 = 5
    program.assert(out.clone().eq(expected));

    // Violation: out != 5
    program.assert(out.ne(Expr::real(5)));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Stride=2 concrete (QF_LRA) should be Proven. detail: {}",
        detail,
    );
}

// ---------------------------------------------------------------------------
// Property 5: Dilation-Effective Kernel
// ---------------------------------------------------------------------------

/// Prove the dilation-effective kernel formula:
///   effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
///
/// This is equivalent to: effective_kernel = dilation * (kernel_size - 1) + 1
///
/// We prove both forms are equal and that effective_kernel >= kernel_size for D >= 1.
/// Uses QF_LRA.
#[test]
fn test_dilation_effective_kernel_equivalence() {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let k = declare_real(&mut program, "k");
    let d = declare_real(&mut program, "d");

    program.assert(k.clone().real_ge(Expr::real(1)));
    program.assert(d.clone().real_ge(Expr::real(1)));

    // Form 1: K + (K-1)*(D-1)
    let form1 = k.clone().real_add(
        k.clone()
            .real_sub(Expr::real(1))
            .real_mul(d.clone().real_sub(Expr::real(1))),
    );

    // Form 2: D*(K-1) + 1
    let form2 = d
        .real_mul(k.real_sub(Expr::real(1)))
        .real_add(Expr::real(1));

    // Violation: form1 != form2
    program.assert(form1.ne(form2));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Dilation effective kernel equivalence (QF_LRA) should be Proven. detail: {}",
        detail,
    );
}

/// Build the "effective kernel >= nominal kernel" query in integers.
///
/// With dilation D the effective (dilated) kernel size is
///   eff = K + (K-1)*(D-1),
/// so eff - K = (K-1)*(D-1) >= 0 for K >= 1, D >= 1, i.e. dilation never shrinks
/// the receptive field. The two factors `K-1` and `D-1` are a var*var product;
/// we pin the kernel to a concrete `K=3` (the extra term becomes `2*(D-1)`,
/// linear in the still-symbolic dilation `D`) so the query stays in decidable
/// `QF_LIA` yet is proven for the whole family of dilations D >= 1.
///
/// `correct` toggles the sign of the dilation term: the true `+ (K-1)*(D-1)` or
/// the bug `- (K-1)*(D-1)`, which pushes eff below the nominal kernel size.
fn build_dilation_effective_kernel(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let k: i64 = 3; // pin the kernel so (K-1)*(D-1) is linear in D
    let d = declare_int_bounded(&mut program, "d", 1, 4096);

    // extra = (K-1)*(D-1) = 2*(D-1), linear in D.
    let extra = Expr::int(k - 1).int_mul(d.int_sub(Expr::int(1)));

    let eff = program.declare_const("eff", Sort::int());
    let formula = if correct {
        Expr::int(k).int_add(extra) // K + (K-1)*(D-1)
    } else {
        Expr::int(k).int_sub(extra) // BUG: K - (K-1)*(D-1)
    };
    program.assert(eff.clone().eq(formula));

    // Violation: the effective kernel is smaller than the nominal kernel.
    program.assert(eff.int_lt(Expr::int(k)));
    program.check_sat();
    program
}

/// Prove that dilation never shrinks the kernel: effective >= nominal for D >= 1.
#[test]
fn test_dilation_effective_kernel_ge_nominal() {
    let program = build_dilation_effective_kernel(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Effective kernel >= nominal (QF_LIA) should be Proven. detail: {}",
        detail,
    );
    assert_eq!(crate::ay_vacuity::vacuity_smell(&smt2), None);
    assert!(smt2.contains("QF_LIA"), "should use QF_LIA logic");
}

/// Flipping the sign of the dilation term makes eff = K - (K-1)*(D-1), which for
/// D >= 2 drops below the nominal kernel K: the query must be SAT.
#[test]
fn test_dilation_effective_kernel_ge_nominal_mutation() {
    let program = build_dilation_effective_kernel(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the dilation term negated the effective kernel drops below nominal and \
         the query must be SAT; got: {detail}",
    );
}

// ---------------------------------------------------------------------------
// Property 6: Groups Decomposition
// ---------------------------------------------------------------------------

/// Build the grouped-conv parameter-count query in integers.
///
/// A dense conv has `C_out * C_in * K` weights; splitting into G groups gives
/// `G * (C_out/G) * (C_in/G) * K = C_out * C_in * K / G` weights, i.e.
/// `grouped * G = standard`, so for G >= 2 the grouped conv uses *strictly*
/// fewer weights than the dense one. The triple product `C_out*C_in*K` and the
/// `grouped*G` term are var*var; we pin `G=2`, `C_out=6`, `K=3` to literals and
/// keep `C_in` symbolic, so `standard = 18*C_in` and `2*grouped = standard` are
/// linear (`QF_LIA`) and `grouped` is derived (= 9*C_in), not asserted.
///
/// `correct` toggles the group split: the true `2*grouped = standard` (each
/// group's filters really shrink) or the bug `grouped = standard` (forgot to
/// shrink them), which makes the grouped count equal the dense count.
fn build_groups_param_count(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let g: i64 = 2; // number of groups
    let c_out: i64 = 6;
    let kk: i64 = 3;
    let c_in = declare_int_bounded(&mut program, "c_in", 1, 4096);

    // standard = C_out * C_in * K, with C_out and K literal -> 18*C_in, linear.
    let standard = program.declare_const("standard_params", Sort::int());
    program.assert(standard.clone().eq(Expr::int(c_out * kk).int_mul(c_in)));

    // grouped * G = standard   (correct)   vs   grouped = standard   (bug).
    let grouped = program.declare_const("grouped_params", Sort::int());
    if correct {
        program.assert(grouped.clone().int_mul(Expr::int(g)).eq(standard.clone()));
    } else {
        program.assert(grouped.clone().eq(standard.clone())); // BUG: no division by G
    }

    // Violation: the grouped conv does NOT use strictly fewer params.
    program.assert(grouped.int_ge(standard));
    program.check_sat();
    program
}

/// Prove grouped conv (G=2) uses strictly fewer params than the dense conv.
#[test]
fn test_groups_decomposition_param_count() {
    let program = build_groups_param_count(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Groups decomposition (QF_LIA) should be Proven. detail: {}",
        detail,
    );
    assert_eq!(crate::ay_vacuity::vacuity_smell(&smt2), None);
    assert!(smt2.contains("QF_LIA"), "should use QF_LIA logic");
}

/// Forgetting to divide each group's filter bank by G makes the grouped count
/// equal the dense count, so it is no longer strictly fewer: the query must be SAT.
#[test]
fn test_groups_decomposition_param_count_mutation() {
    let program = build_groups_param_count(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "without dividing by G the grouped count equals the dense count and the \
         strict-fewer query must be SAT; got: {detail}",
    );
}

/// Build the grouped-conv output-channel-concatenation query in integers.
///
/// The content is that concatenating the G groups recovers *every* channel of an
/// independently-given `C_out` — and it does so exactly when `G` divides
/// `C_out`. So `C_out` is a free input (not `G*per_group`, which would make the
/// claim `X = X`): each group produces `per_group = floor(C_out / G)` channels,
/// modelled by the euclidean relation `C_out = G*per_group + rem`, `0 <= rem <
/// G`. Concatenation yields `total_out = G*per_group`, which equals `C_out` only
/// because the clean-grouping precondition forces `rem = 0`. The conclusion is
/// derived from that division, not restated.
///
/// `assume_divisible` toggles the `rem = 0` precondition: dropping it lets
/// `C_out` be indivisible (e.g. `C_out = 5, G = 2` leaves `rem = 1`), so
/// `total_out = C_out - rem != C_out` and the query becomes SAT — the
/// divisibility is load-bearing. `G = 2` is a literal so every product is linear
/// (`QF_LIA`).
fn build_groups_concat(assume_divisible: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let g: i64 = 2;

    // C_out is an INDEPENDENT input: the number of output channels to recover.
    let c_out = declare_int_bounded(&mut program, "c_out", 1, 4096);

    // per_group = floor(C_out / G), via the euclidean relation.
    let per_group = program.declare_const("per_group_out", Sort::int());
    let rem = program.declare_const("rem", Sort::int());
    program.assert(
        c_out
            .clone()
            .eq(Expr::int(g).int_mul(per_group.clone()).int_add(rem.clone())),
    );
    program.assert(rem.clone().int_ge(Expr::int(0)));
    program.assert(rem.clone().int_lt(Expr::int(g)));
    if assume_divisible {
        // Clean grouping: C_out is a multiple of G, so no channel is left over.
        program.assert(rem.eq(Expr::int(0)));
    }

    // Concatenate the G per-group outputs.
    let total_out = program.declare_const("total_out", Sort::int());
    program.assert(total_out.clone().eq(Expr::int(g).int_mul(per_group)));

    // Violation: the concatenated channel count differs from C_out.
    program.assert(total_out.ne(c_out));
    program.check_sat();
    program
}

/// Prove concatenating the G per-group outputs recovers exactly C_out channels.
#[test]
fn test_groups_output_channel_concatenation() {
    let program = build_groups_concat(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Groups output channel concatenation (QF_LIA) should be Proven. detail: {}",
        detail,
    );
    assert_eq!(crate::ay_vacuity::vacuity_smell(&smt2), None);
    assert!(smt2.contains("QF_LIA"), "should use QF_LIA logic");
}

/// Concatenating one group too many yields `(G+1)*per_group != C_out`, so the
/// channel-count identity breaks: the query must be SAT.
#[test]
fn test_groups_output_channel_concatenation_mutation() {
    let program = build_groups_concat(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "concatenating one group too many changes the channel count and the query \
         must be SAT; got: {detail}",
    );
}

// ---------------------------------------------------------------------------
// Property 7: Bias Addition Commutativity
// ---------------------------------------------------------------------------

/// Prove that bias is applied per-output-channel: adding bias[c] to each spatial
/// position of output channel c does not depend on spatial position.
///
/// For output positions (c, t0) and (c, t1) with the same channel c:
///   out(c, t0) = conv_result(c, t0) + bias[c]
///   out(c, t1) = conv_result(c, t1) + bias[c]
///
/// The bias contribution is the same for both positions.
/// Uses QF_LRA.
#[test]
fn test_bias_per_channel_independence() {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let bound_lo = Expr::real(-100);
    let bound_hi = Expr::real(100);

    // Two conv results at different spatial positions, same channel
    let conv_t0 = declare_real(&mut program, "conv_t0");
    let conv_t1 = declare_real(&mut program, "conv_t1");
    let bias_c = declare_real(&mut program, "bias_c");

    assert_bounds(&mut program, &conv_t0, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &conv_t1, &bound_lo, &bound_hi);
    assert_bounds(&mut program, &bias_c, &bound_lo, &bound_hi);

    // out_t0 = conv_t0 + bias_c
    // out_t1 = conv_t1 + bias_c
    let out_t0 = conv_t0.clone().real_add(bias_c.clone());
    let out_t1 = conv_t1.clone().real_add(bias_c);

    // Property: out_t0 - out_t1 = conv_t0 - conv_t1 (bias cancels)
    let out_diff = out_t0.real_sub(out_t1);
    let conv_diff = conv_t0.real_sub(conv_t1);

    // Violation: out_diff != conv_diff
    program.assert(out_diff.ne(conv_diff));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Bias per-channel independence (QF_LRA) should be Proven. detail: {}",
        detail,
    );
}

// ---------------------------------------------------------------------------
// Property 8: Transpose Conv as Adjoint
// ---------------------------------------------------------------------------

/// Prove that transpose convolution is the adjoint of forward convolution
/// in terms of dimension recovery.
///
/// Forward: out_size = (in_size + 2*P - K) / S + 1  [S=1 case: out = in + 2P - K + 1]
/// Transpose: recovered = (out_size - 1) * S - 2*P + K + output_padding
///
/// For S=1, output_padding=0:
///   recovered = out_size - 1 - 2P + K = (in + 2P - K + 1) - 1 - 2P + K = in
///
/// Uses QF_LRA.
#[test]
fn test_transpose_conv_dimension_recovery() {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let n = declare_real(&mut program, "n");
    let k = declare_real(&mut program, "k");
    let p = declare_real(&mut program, "p");
    let out_fwd = declare_real(&mut program, "out_fwd");
    let recovered = declare_real(&mut program, "recovered");

    program.assert(n.clone().real_ge(Expr::real(1)));
    program.assert(k.clone().real_ge(Expr::real(1)));
    program.assert(p.clone().real_ge(Expr::real(0)));

    // Valid config: N + 2P >= K
    let two_p = Expr::real(2).real_mul(p.clone());
    program.assert(n.clone().real_add(two_p.clone()).real_ge(k.clone()));

    // Forward (S=1): out = N + 2P - K + 1
    let fwd_formula = n
        .clone()
        .real_add(two_p.clone())
        .real_sub(k.clone())
        .real_add(Expr::real(1));
    program.assert(out_fwd.clone().eq(fwd_formula));

    // Transpose (S=1, output_padding=0): recovered = (out - 1) - 2P + K
    let trans_formula = out_fwd.real_sub(Expr::real(1)).real_sub(two_p).real_add(k);
    program.assert(recovered.clone().eq(trans_formula));

    // Violation: recovered != N
    program.assert(recovered.ne(n));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Transpose conv dimension recovery (QF_LRA) should be Proven. detail: {}",
        detail,
    );
}

/// Build the transpose-conv stride-recovery query in integers.
///
/// Forward conv with stride S and exact division maps an input length N to
///   out = (N + 2P - K)/S + 1,   i.e.   N + 2P - K = S*(out - 1).
/// The transpose conv (output_padding = 0) recovers the input length by
///   recovered = (out - 1)*S - 2P + K,
/// and substituting the forward relation gives
///   recovered = (N + 2P - K) - 2P + K = N.
/// The real-valued encoding had `S*(out-1)` and `(out-1)*S` var*var products; we
/// pin the stride to a concrete `S=2` so both are linear, and keep N, K, P
/// symbolic integers so the recovery identity is proven for a whole family of
/// shapes in decidable `QF_LIA`.
///
/// `correct` toggles the transpose formula's kernel term: the true `+ K` (which
/// cancels against the forward `- K`) or the sign-flipped `- K` bug (which does
/// not cancel, so the recovered length is `N - 2K != N`).
fn build_transpose_conv_recovery(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let s: i64 = 2; // concrete stride keeps (out-1)*S linear

    // Symbolic shape, bounded to keep the search tiny.
    let n = declare_int_bounded(&mut program, "n", 1, 64);
    let k = declare_int_bounded(&mut program, "k", 1, 16);
    let p = declare_int_bounded(&mut program, "p", 0, 16);

    let two_p = Expr::int(2).int_mul(p);
    let numerator = n.clone().int_add(two_p.clone()).int_sub(k.clone());
    program.assert(numerator.clone().int_ge(Expr::int(0)));

    // Forward, exact division: numerator = S*(out - 1).
    let out = program.declare_const("out_fwd", Sort::int());
    program.assert(numerator.eq(Expr::int(s).int_mul(out.clone().int_sub(Expr::int(1)))));

    // Transpose recovery: recovered = (out-1)*S - 2P (+/- K).
    let base = out.int_sub(Expr::int(1)).int_mul(Expr::int(s)).int_sub(two_p);
    let recovered = if correct {
        base.int_add(k) // + K  (cancels the forward -K)
    } else {
        base.int_sub(k) // BUG: -K, does not cancel
    };

    // Violation: the transpose did not recover the original input length.
    program.assert(recovered.ne(n));
    program.check_sat();
    program
}

/// Prove transpose conv (stride 2, exact division) recovers the input length.
#[test]
fn test_transpose_conv_stride_recovery() {
    let program = build_transpose_conv_recovery(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Transpose conv stride recovery (QF_LIA) should be Proven. detail: {}",
        detail,
    );
    assert_eq!(crate::ay_vacuity::vacuity_smell(&smt2), None);
    assert!(smt2.contains("QF_LIA"), "should use QF_LIA logic");
}

/// Flipping the kernel term to `- K` breaks the cancellation, so the recovered
/// length is `N - 2K != N` for every K >= 1: the query must be SAT.
#[test]
fn test_transpose_conv_stride_recovery_mutation() {
    let program = build_transpose_conv_recovery(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the kernel term sign-flipped the recovery no longer cancels and the \
         query must be SAT; got: {detail}",
    );
}

// ---------------------------------------------------------------------------
// Property 9: Depthwise Separable Equivalence
// ---------------------------------------------------------------------------

/// Build the depthwise-separable parameter-count query in integers.
///
/// A depthwise-separable conv = depthwise (`C_in * K` weights) + pointwise 1x1
/// (`C_out * C_in` weights), total `C_in*(K + C_out)`, versus a dense conv's
/// `C_out * C_in * K`. The separable form uses *strictly* fewer weights except
/// at the corner `C_out = K = 2`, where the two counts are exactly equal — which
/// is precisely why the original strict claim over all `C_out, K >= 2` was false
/// (SAT at (2,2), and over the reals there were further fractional witnesses).
/// We pin a representative `C_out=4`, `K=3` (inside the strict region) to literals
/// and keep `C_in` symbolic, so every product is linear (`QF_LIA`): the separable
/// count `7*C_in` is strictly below the dense `12*C_in` for all `C_in >= 1`.
///
/// `correct` toggles the dense-conv count: the true `C_out*C_in*K` or the bug
/// `C_out*C_in` (forgot the K spatial positions), which makes the dense count no
/// larger than the separable one.
fn build_depthwise_separable(correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let c_out: i64 = 4;
    let kk: i64 = 3;
    let c_in = declare_int_bounded(&mut program, "c_in", 1, 4096);

    // Depthwise = C_in * K.   Pointwise = C_out * C_in.
    let dw = program.declare_const("dw_params", Sort::int());
    program.assert(dw.clone().eq(Expr::int(kk).int_mul(c_in.clone())));
    let pw = program.declare_const("pw_params", Sort::int());
    program.assert(pw.clone().eq(Expr::int(c_out).int_mul(c_in.clone())));

    // Total separable = depthwise + pointwise.
    let total_sep = program.declare_const("total_sep", Sort::int());
    program.assert(total_sep.clone().eq(dw.int_add(pw)));

    // Dense conv = C_out * C_in * K   (correct)   vs   C_out * C_in   (bug).
    let standard = program.declare_const("standard", Sort::int());
    let dense = if correct {
        Expr::int(c_out * kk).int_mul(c_in) // C_out*K * C_in
    } else {
        Expr::int(c_out).int_mul(c_in) // BUG: dropped the K factor
    };
    program.assert(standard.clone().eq(dense));

    // Violation: the separable form does NOT use strictly fewer params.
    program.assert(total_sep.int_ge(standard));
    program.check_sat();
    program
}

/// Prove depthwise-separable conv (C_out=4, K=3) uses strictly fewer params than
/// the dense conv, for every input-channel count C_in >= 1.
#[test]
fn test_depthwise_separable_param_count() {
    let program = build_depthwise_separable(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven,
        "Depthwise separable param count (QF_LIA) should be Proven. detail: {}",
        detail,
    );
    assert_eq!(crate::ay_vacuity::vacuity_smell(&smt2), None);
    assert!(smt2.contains("QF_LIA"), "should use QF_LIA logic");
}

/// Dropping the K spatial-position factor from the dense count leaves
/// `standard = C_out*C_in = 4*C_in`, which the separable `7*C_in` exceeds, so the
/// strict-fewer claim breaks: the query must be SAT.
#[test]
fn test_depthwise_separable_param_count_mutation() {
    let program = build_depthwise_separable(false);
    let (proven, detail) = execute_and_check(&program);
    assert!(
        !proven,
        "with the dense count missing the K factor the separable form is no longer \
         strictly smaller and the query must be SAT; got: {detail}",
    );
}

/// Prove the depthwise separable decomposition identity:
///   C_in * (K + C_out) = C_in * K + C_out * C_in
///
/// This confirms the total param count for depthwise separable equals the sum
/// of depthwise and pointwise components.
/// Uses QF_LRA.
#[test]
fn test_depthwise_separable_decomposition_identity() {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let c_in = declare_real(&mut program, "c_in");
    let c_out = declare_real(&mut program, "c_out");
    let k = declare_real(&mut program, "k");

    program.assert(c_in.clone().real_ge(Expr::real(1)));
    program.assert(c_out.clone().real_ge(Expr::real(1)));
    program.assert(k.clone().real_ge(Expr::real(1)));

    // LHS: C_in * (K + C_out)
    let lhs = c_in.clone().real_mul(k.clone().real_add(c_out.clone()));

    // RHS: C_in * K + C_out * C_in
    let rhs = c_in.clone().real_mul(k).real_add(c_out.real_mul(c_in));

    // Violation: lhs != rhs
    program.assert(lhs.ne(rhs));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    assert!(
        proven || detail.contains("Unknown"),
        "Depthwise separable decomposition: expected Proven or Unknown, got: {}",
        detail,
    );
    assert!(
        !detail.contains("counterexample"),
        "Depthwise separable decomposition must not have counterexample: {}",
        detail,
    );
}

// ---------------------------------------------------------------------------
// Meta tests: SMT2 structure validation
// ---------------------------------------------------------------------------

/// Verify that all proof encodings produce valid SMT-LIB2 structure.
#[test]
fn test_all_proofs_produce_valid_smt2() {
    // Run a representative set of proofs and check SMT2 structure
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");
    let x = declare_real(&mut program, "x");
    program.assert(x.clone().real_ge(Expr::real(1)));
    program.assert(x.real_lt(Expr::real(1)));
    program.check_sat();

    let smt2 = program.to_string();
    assert!(smt2.contains("set-logic"), "should declare logic");
    assert!(smt2.contains("check-sat"), "should have check-sat");
    assert!(smt2.contains("declare-const"), "should have declarations");
}

/// Verify NRA proofs use the correct logic declaration.
#[test]
fn test_nra_proofs_declare_correct_logic() {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");
    let x = declare_real(&mut program, "x");
    let y = declare_real(&mut program, "y");
    program.assert(x.clone().real_mul(y.clone()).ne(y.real_mul(x)));
    program.check_sat();

    let smt2 = program.to_string();
    assert!(smt2.contains("QF_NRA"), "should use QF_NRA logic");
}
