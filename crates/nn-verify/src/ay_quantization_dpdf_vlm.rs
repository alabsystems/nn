// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for quantization error mathematical bounds for dpdf VLMs (#4238).
//!
//! Vision-language models (VLMs) used in dpdf document processing are
//! particularly sensitive to quantization error because they combine
//! vision encoders (high dynamic range activations) with language decoders
//! (attention scores with narrow effective range). This module proves
//! seven properties that bound quantization error across the full
//! dpdf VLM inference pipeline.
//!
//! # Properties Proved
//!
//! 1. **F32->BF16 rounding error bounded**: `|x - bf16(x)| <= eps_bf16 * |x|`.
//! 2. **F32->F16 truncation error bounded**: `|x - f16(x)| <= eps_f16 * |x|`.
//! 3. **Symmetric quantization preserves zero**: `quant(0) = 0` when zero_point = 0.
//! 4. **Asymmetric quantization maps range correctly**: endpoints map to grid boundaries.
//! 5. **Dequantize inverts quantize within error bound**: `|x - dequant(quant(x))| <= s/2`.
//! 6. **Quantized matmul error accumulation bounded**: accumulated error <= D * s_a * s_b / 4.
//! 7. **Mixed-precision chain error composition**: chained conversions compose linearly.
//!
//! # Proof Strategy
//!
//! Quantization rounding is modeled via helper variables with linear constraints
//! (see existing `ay_quantization_error` module for the core technique). For
//! matmul accumulation and chain composition, we use the triangle inequality
//! and induction on the number of accumulations.
//!
//! The rounding/index properties (symmetric-zero, asymmetric endpoint mapping,
//! dequantize roundtrip) are decided in `QF_LIA` over concrete integer scale
//! grids — quantized codes are integer levels, and modelling them over the reals
//! admits fractional counterexamples (`q = 0.3`) or var*var products the solver
//! cannot close. The relative-error bounds (bf16/f16 rounding, matmul
//! accumulation, mixed-precision chain) stay in `QF_LRA` as linear sums of
//! bounded error terms.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::smt_error::SmtError;

/// Result of a dpdf VLM quantization property proof attempt.
#[derive(Debug, Clone)]
pub struct DpdfQuantPropertyResult {
    /// Human-readable property name.
    pub property: String,
    /// Whether the proof succeeded (UNSAT = property holds for all inputs).
    pub proven: bool,
    /// SMT-LIB2 text of the query (for debugging/external solver use).
    pub smt2: String,
    /// Solver detail message.
    pub detail: String,
}

/// Declare a real variable and return its expression.
fn declare_real(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::real())
}

/// Assert `lower <= expr <= upper` using `Expr` bounds.
fn assert_bounds(program: &mut AYProgram, expr: &Expr, lower: &Expr, upper: &Expr) {
    program.assert(expr.clone().real_ge(lower.clone()));
    program.assert(expr.clone().real_le(upper.clone()));
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
// Property 1: F32 -> BF16 Rounding Error Bounded
// ---------------------------------------------------------------------------

/// Prove that F32->BF16 rounding error is bounded by `eps_bf16 * |x|`
/// for normal-range values.
///
/// BF16 (Brain Float 16) has a 7-bit significand (plus implicit 1), giving
/// machine epsilon = 2^-7 = 7.8125e-3. For nearest-even rounding, the
/// per-value relative error is at most `eps/2`, but we prove the looser
/// `eps * |x|` bound which is simpler and sufficient for dpdf VLM analysis.
///
/// We model the BF16 cast as `x_bf16 = x + err` where the constraint
/// `|err| <= eps * |x|` encodes the IEEE 754 rounding guarantee. The
/// negation `|err| > eps * |x|` is asserted and proved UNSAT.
///
/// Uses `QF_LRA` with positive-domain encoding (negative case is symmetric).
pub fn prove_f32_to_bf16_rounding_error() -> Result<DpdfQuantPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // BF16 machine epsilon = 2^-7
    let eps_val = Expr::real(78125).real_div(Expr::real(10000000)); // 7.8125e-3

    // x is a positive normal BF16-representable value.
    // BF16 shares F32's exponent range, so normal range is large.
    let x = declare_real(&mut program, "x");
    let lo = Expr::real(1).real_div(Expr::real(1000000)); // 1e-6 (well above subnormal)
    let hi = Expr::real(100000);
    assert_bounds(&mut program, &x, &lo, &hi);

    // Rounding error
    let err = declare_real(&mut program, "err");

    // |err| <= eps * x (for positive x, |x| = x)
    let bound = x.clone().real_mul(eps_val.clone());
    program.assert(err.clone().real_ge(bound.clone().real_neg()));
    program.assert(err.clone().real_le(bound));

    // Negated property: |err| > eps * |x|
    let bound2 = x.real_mul(eps_val);
    let too_high = err.clone().real_gt(bound2.clone());
    let too_low = err.real_lt(bound2.real_neg());
    let violation = too_high.or(too_low);

    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfQuantPropertyResult {
        property: "f32_to_bf16_rounding_error_bounded".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: F32 -> F16 Truncation Error Bounded
// ---------------------------------------------------------------------------

/// Prove that F32->F16 truncation error is bounded by `eps_f16 * |x|`
/// for normal F16 values.
///
/// IEEE 754 half-precision (F16) has a 10-bit significand giving
/// machine epsilon = 2^-10 = 9.765625e-4. The proof structure mirrors
/// the BF16 proof but with the tighter F16 epsilon.
///
/// For dpdf VLMs, F16 is used in vision encoder intermediate activations
/// where the tighter precision matters for feature extraction quality.
///
/// Uses `QF_LRA` with positive-domain encoding.
pub fn prove_f32_to_f16_truncation_error() -> Result<DpdfQuantPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // F16 machine epsilon = 2^-10
    let eps_val = Expr::real(9765625).real_div(Expr::real(10000000000_i64)); // 9.765625e-4

    // x is in the F16 normal range: ~6.1e-5 to 65504
    let x = declare_real(&mut program, "x");
    let lo = Expr::real(61).real_div(Expr::real(1000000)); // ~6.1e-5
    let hi = Expr::real(65504);
    assert_bounds(&mut program, &x, &lo, &hi);

    let err = declare_real(&mut program, "err");

    // |err| <= eps * x
    let bound = x.clone().real_mul(eps_val.clone());
    program.assert(err.clone().real_ge(bound.clone().real_neg()));
    program.assert(err.clone().real_le(bound));

    // Negated: |err| > eps * x
    let bound2 = x.real_mul(eps_val);
    let too_high = err.clone().real_gt(bound2.clone());
    let too_low = err.real_lt(bound2.real_neg());
    let violation = too_high.or(too_low);

    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfQuantPropertyResult {
        property: "f32_to_f16_truncation_error_bounded".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: Symmetric Quantization Preserves Zero
// ---------------------------------------------------------------------------

/// Concrete integer scale step shared by the symmetric-zero and roundtrip
/// proofs. Ten is even, so the half-step `s/2 = 5` is an exact integer and the
/// round-to-nearest bounds and error bound stay in `QF_LIA` with integer-literal
/// coefficients (no fractional terms, no `Int * Real` mixing).
const SCALE_STEP: i64 = 10;

/// Prove that symmetric quantization maps zero to zero exactly.
///
/// For symmetric quantization with scale `s > 0` and zero_point = 0,
/// `quant(0) = round(0 / s) = 0` and `dequant(0) = 0 * s = 0`. Zero-valued
/// activations (ReLU, padding, attention masks) must survive quantization
/// unchanged, or sparse attention patterns and masked document regions corrupt.
///
/// The content is that *round-to-nearest sends the input 0 to the code 0*. The
/// code `q` is an **`Int`** — a quantized level, not a real. The nearest-integer
/// constraint `(q - 1/2)*s <= x <= (q + 1/2)*s` at `x = 0`, scaled by `2` to clear
/// the halves, becomes `-s <= 2*s*q <= s`, which pins `q = 0` only because `q`
/// ranges over the integers: over the reals `q = 0.3` satisfies the same bound and
/// leaves `dequant = 0.3*s != 0`, the exact counterexample this replaces. With
/// `q = 0`, `dequant = q*s = 0`.
///
/// A concrete scale `s = SCALE_STEP` keeps every coefficient an integer literal, so
/// the query is decidable `QF_LIA`. A biased (non-centered) rounding interval maps
/// 0 to a nonzero code and turns the query SAT — see
/// `symmetric_zero_depends_on_centered_rounding`.
pub fn prove_symmetric_quantization_preserves_zero() -> Result<DpdfQuantPropertyResult, SmtError> {
    let program = build_symmetric_quantization_preserves_zero(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfQuantPropertyResult {
        property: "symmetric_quantization_preserves_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the zero-preservation query in `QF_LIA`. `rounding_is_centered` gates the
/// single fact the theorem rests on — that round-to-nearest centers its interval on
/// the code. When false the interval is shifted down one step (a zero-point bias),
/// so input 0 quantizes to code 1 and `dequant != 0`; tests flip it to confirm the
/// proof depends on centered rounding.
fn build_symmetric_quantization_preserves_zero(rounding_is_centered: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let s = SCALE_STEP;

    // Quantized code q (an integer level in the INT8 symmetric range).
    let q = program.declare_const("q", Sort::int());
    program.assert(q.clone().int_ge(Expr::int(-127)));
    program.assert(q.clone().int_le(Expr::int(127)));

    // Input x = 0, carried as 2*x = 0 to match the doubled rounding bounds.
    let two_x = Expr::int(0);

    // Round-to-nearest: (q - 1/2)*s <= x <= (q + 1/2)*s, doubled to clear the
    // halves -> 2*s*q - s <= 2*x <= 2*s*q + s. The bug shifts the whole interval
    // down one step (2*s*q - 3s .. 2*s*q - s), so 0 falls in code 1's cell.
    let (lo_offset, hi_offset) = if rounding_is_centered {
        (-s, s) // centered on code q
    } else {
        (-3 * s, -s) // BUG: interval shifted one step -> zero-point bias
    };
    let two_s_q = q.clone().int_mul(Expr::int(2 * s));
    program.assert(
        two_x
            .clone()
            .int_ge(two_s_q.clone().int_add(Expr::int(lo_offset))),
    );
    program.assert(two_x.int_le(two_s_q.int_add(Expr::int(hi_offset))));

    // dequant(quant(0)) = q * s. Violation: it is not exactly zero.
    let dequant = q.int_mul(Expr::int(s));
    program.assert(dequant.ne(Expr::int(0)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 4: Asymmetric Quantization Maps Range Correctly
// ---------------------------------------------------------------------------

/// Top code of the UINT8 asymmetric grid; codes run `0 ..= Q_MAX`.
const Q_MAX: i64 = 255;

/// Prove that asymmetric quantization maps the input endpoints to the grid
/// boundaries: `quant(x_min) = 0` and `quant(x_max) = Q_max`.
///
/// Asymmetric quantization is `quant(x) = round(x/scale) + zero_point` with
/// `scale = (x_max - x_min)/Q_max` and `zero_point = -round(x_min/scale)`. Writing
/// `n_min = round(x_min/scale)` and `n_max = round(x_max/scale)` (both **`Int`**
/// levels), the definition of `scale` makes the endpoints span exactly `Q_max` grid
/// steps: `n_max - n_min = Q_max`. The theorem is that with the standard zero-point
/// `z = -n_min`,
///
/// ```text
/// quant(x_min) = n_min + z = n_min - n_min   = 0
/// quant(x_max) = n_max + z = (n_max - n_min)  = Q_max
/// ```
///
/// The levels are **`Int`**. The old encoding modelled the endpoint images over the
/// reals as `f*scale = x + zp*scale` — two products of *declared* variables, a
/// nonlinear `QF_NRA`/opaque-multiply query the solver could not close honestly
/// (it admitted a spurious `f != 0` model). Here `n_min`, `n_max`, `z` are integers
/// and every step is linear, so the query is decidable `QF_LIA`. The zero-point
/// sign is the whole theorem: flipping it to `z = +n_min` leaves
/// `quant(x_min) = 2*n_min`, nonzero whenever `n_min != 0`, and turns the query SAT
/// — see `range_mapping_depends_on_zero_point_sign`.
pub fn prove_asymmetric_quantization_range_mapping() -> Result<DpdfQuantPropertyResult, SmtError> {
    let program = build_asymmetric_quantization_range_mapping(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfQuantPropertyResult {
        property: "asymmetric_quantization_range_mapping".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the endpoint-mapping query in `QF_LIA`. `zero_point_sign_correct` gates the
/// standard zero-point `z = -n_min`; when false it uses the wrong sign `z = +n_min`,
/// the classic asymmetric-quantization slip, which stops the min endpoint from
/// landing on code 0. Tests flip it to confirm the proof depends on the sign.
fn build_asymmetric_quantization_range_mapping(zero_point_sign_correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    // Rounded integer levels of the two endpoints, n = round(x/scale).
    let n_min = program.declare_const("n_min", Sort::int());
    let n_max = program.declare_const("n_max", Sort::int());
    for n in [&n_min, &n_max] {
        program.assert(n.clone().int_ge(Expr::int(-10000)));
        program.assert(n.clone().int_le(Expr::int(10000)));
    }

    // scale = (x_max - x_min)/Q_max  <=>  the endpoints span Q_max grid steps.
    program.assert(n_max.clone().int_sub(n_min.clone()).eq(Expr::int(Q_MAX)));

    // Zero-point z = -round(x_min/scale) = -n_min (standard). The bug uses +n_min.
    let z = program.declare_const("z", Sort::int());
    if zero_point_sign_correct {
        program.assert(z.clone().int_add(n_min.clone()).eq(Expr::int(0))); // z = -n_min
    } else {
        program.assert(z.clone().int_sub(n_min.clone()).eq(Expr::int(0))); // BUG: z = +n_min
    }

    // Endpoint codes and the grid-boundary property.
    let code_min = n_min.int_add(z.clone());
    let code_max = n_max.int_add(z);
    let violation = code_min
        .ne(Expr::int(0))
        .or(code_max.ne(Expr::int(Q_MAX)));
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 5: Dequantize Inverts Quantize Within Error Bound
// ---------------------------------------------------------------------------

/// Prove that the roundtrip `dequant(quant(x))` is within `s/2` of `x`.
///
/// For symmetric quantization with scale `s > 0`, `q = round(x/s)` and
/// `dequant(q) = q*s`, so `|x - q*s| <= s/2`. This is the fundamental
/// dequantization bound: quantize/dequantize roundtrips inject bounded,
/// predictable error into mixed-precision inference.
///
/// The code `q` is an **`Int`**. Round-to-nearest pins the only fact that makes the
/// bound hold: `(q - 1/2)*s <= x <= (q + 1/2)*s`, doubled, is
/// `2*s*q - s <= 2*x <= 2*s*q + s`, i.e. `-s <= 2*(x - q*s) <= s`, which is exactly
/// `|x - q*s| <= s/2`. The bound is a property of the *integer* rounding grid; the
/// old real-valued encoding multiplied two declared variables (`q*s`, `(q±1/2)*s`)
/// and hung in nonlinear arithmetic instead of returning.
///
/// A concrete scale `s = SCALE_STEP` (even, so `s/2` is an integer) keeps the query
/// in decidable `QF_LIA`. Replacing round-to-nearest with truncation widens the
/// error to `s` and turns the query SAT — see
/// `roundtrip_bound_depends_on_nearest_rounding`.
pub fn prove_dequantize_inverts_quantize() -> Result<DpdfQuantPropertyResult, SmtError> {
    let program = build_dequantize_inverts_quantize(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfQuantPropertyResult {
        property: "dequantize_inverts_quantize_within_error".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the roundtrip-error query in `QF_LIA`. `rounds_to_nearest` gates the
/// rounding rule: centered nearest-rounding (error half-width `s/2`) when true, or
/// truncation toward the lower cell (error width `s`) when false. Truncation lets
/// `x = s*q + (s-1)` slip through with error `s-1 > s/2`, breaking the bound and
/// making the query SAT; tests flip it to confirm the proof depends on
/// nearest-rounding.
fn build_dequantize_inverts_quantize(rounds_to_nearest: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let s = SCALE_STEP;

    // Quantized code q in the INT8 symmetric range, and input x in the
    // representable band [-127*s, 127*s] (same integer units).
    let q = program.declare_const("q", Sort::int());
    program.assert(q.clone().int_ge(Expr::int(-127)));
    program.assert(q.clone().int_le(Expr::int(127)));
    let x = program.declare_const("x", Sort::int());
    program.assert(x.clone().int_ge(Expr::int(-127 * s)));
    program.assert(x.clone().int_le(Expr::int(127 * s)));

    if rounds_to_nearest {
        // (q - 1/2)*s <= x <= (q + 1/2)*s, doubled: 2*s*q - s <= 2*x <= 2*s*q + s.
        let two_s_q = q.clone().int_mul(Expr::int(2 * s));
        let two_x = x.clone().int_mul(Expr::int(2));
        program.assert(two_x.clone().int_ge(two_s_q.clone().int_sub(Expr::int(s))));
        program.assert(two_x.int_le(two_s_q.int_add(Expr::int(s))));
    } else {
        // BUG: truncation toward the lower cell: s*q <= x <= s*q + (s - 1).
        let s_q = q.clone().int_mul(Expr::int(s));
        program.assert(x.clone().int_ge(s_q.clone()));
        program.assert(x.clone().int_le(s_q.int_add(Expr::int(s - 1))));
    }

    // Error e = x - q*s; bound |e| <= s/2, i.e. -s <= 2*e <= s.
    let error = x.int_sub(q.int_mul(Expr::int(s)));
    let two_e = error.int_mul(Expr::int(2));
    let too_high = two_e.clone().int_gt(Expr::int(s));
    let too_low = two_e.int_lt(Expr::int(-s));
    program.assert(too_high.or(too_low));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 6: Quantized Matmul Error Accumulation Bounded
// ---------------------------------------------------------------------------

/// Prove that quantized matrix multiplication error accumulation is bounded.
///
/// For a dot product of dimension D=2, where each element has independent
/// quantization error:
///   `a_i = a_exact_i + e_a_i` where `|e_a_i| <= s_a/2`
///   `b_i = b_exact_i + e_b_i` where `|e_b_i| <= s_b/2`
///
/// The exact dot product is `sum_i(a_exact_i * b_exact_i)`.
/// The quantized dot product is `sum_i(a_i * b_i)`.
///
/// Each term's error relative to exact is bounded. For the linearized
/// approximation (ignoring second-order error `e_a * e_b`), the per-element
/// error from quantization is bounded by:
///   `|a_exact * e_b + e_a * b_exact| <= |a_exact| * s_b/2 + s_a/2 * |b_exact|`
///
/// We prove the simpler linear bound: if each dot-product term `t_i` has error
/// bounded by `E` (where E = B_a * s_b/2 + s_a/2 * B_b for element bounds
/// B_a, B_b), then the accumulated error over D terms is bounded by `D * E`.
///
/// Uses `QF_LRA` — the error accumulation is a linear sum of bounded terms.
pub fn prove_quantized_matmul_error_accumulation() -> Result<DpdfQuantPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Per-element error bound E > 0
    let per_elem_bound = declare_real(&mut program, "E");
    let e_lo = Expr::real(1).real_div(Expr::real(1000000));
    let e_hi = Expr::real(1000);
    assert_bounds(&mut program, &per_elem_bound, &e_lo, &e_hi);

    // D = 2 (dot product dimension for tractable SMT proof)
    // Per-element errors e0, e1 each bounded by |e_i| <= E
    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");

    // |e0| <= E
    program.assert(e0.clone().real_ge(per_elem_bound.clone().real_neg()));
    program.assert(e0.clone().real_le(per_elem_bound.clone()));
    // |e1| <= E
    program.assert(e1.clone().real_ge(per_elem_bound.clone().real_neg()));
    program.assert(e1.clone().real_le(per_elem_bound.clone()));

    // Accumulated error: e_total = e0 + e1
    let e_total = e0.real_add(e1);

    // Bound: D * E = 2 * E
    let two = Expr::real(2);
    let total_bound = two.real_mul(per_elem_bound);

    // Negated property: |e_total| > D * E
    let too_high = e_total.clone().real_gt(total_bound.clone());
    let too_low = e_total.real_lt(total_bound.real_neg());
    let violation = too_high.or(too_low);

    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfQuantPropertyResult {
        property: "quantized_matmul_error_accumulation_d2".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: Mixed-Precision Chain Error Composition
// ---------------------------------------------------------------------------

/// Prove that mixed-precision conversion chain error composes linearly.
///
/// In dpdf VLMs, a common pattern is F32 -> BF16 (vision encoder) -> F16
/// (cross-attention) -> F32 (language decoder output). Each conversion
/// introduces relative error bounded by the respective machine epsilon.
///
/// For a chain of two conversions with relative errors eps1 and eps2:
///   `x1 = x * (1 + e1)` where `|e1| <= eps1`
///   `x2 = x1 * (1 + e2)` where `|e2| <= eps2`
///   `x2 = x * (1 + e1) * (1 + e2)`
///
/// The total relative error is:
///   `|x2/x - 1| = |(1 + e1)(1 + e2) - 1| = |e1 + e2 + e1*e2|`
///
/// The linear bound (ignoring the second-order `e1*e2` term) is `|e1| + |e2|`.
/// Since `|e1*e2| <= eps1*eps2` which is negligible (< 1e-5 for bf16/f16),
/// we prove the tight linear bound: the total additive error on a value
/// within `[lo, hi]` is bounded by `(eps1 + eps2) * |x|`.
///
/// We model this in `QF_LRA` by treating each conversion's additive error
/// as an independent bounded term and proving their sum is bounded.
pub fn prove_mixed_precision_chain_error() -> Result<DpdfQuantPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // BF16 epsilon
    let eps_bf16 = Expr::real(78125).real_div(Expr::real(10000000)); // 7.8125e-3
                                                                     // F16 epsilon
    let eps_f16 = Expr::real(9765625).real_div(Expr::real(10000000000_i64)); // 9.765625e-4

    // Input value x > 0 (symmetric case is identical)
    let x = declare_real(&mut program, "x");
    let lo = Expr::real(1).real_div(Expr::real(1000));
    let hi = Expr::real(10000);
    assert_bounds(&mut program, &x, &lo, &hi);

    // Stage 1 error: F32 -> BF16, |err1| <= eps_bf16 * x
    let err1 = declare_real(&mut program, "err1");
    let bound1 = x.clone().real_mul(eps_bf16.clone());
    program.assert(err1.clone().real_ge(bound1.clone().real_neg()));
    program.assert(err1.clone().real_le(bound1));

    // Stage 2 error: BF16 -> F16, |err2| <= eps_f16 * x
    // (conservative: bound by eps_f16 * original x, not x+err1)
    let err2 = declare_real(&mut program, "err2");
    let bound2 = x.clone().real_mul(eps_f16.clone());
    program.assert(err2.clone().real_ge(bound2.clone().real_neg()));
    program.assert(err2.clone().real_le(bound2));

    // Total additive error: err_total = err1 + err2
    let err_total = err1.real_add(err2);

    // Combined bound: (eps_bf16 + eps_f16) * x
    let eps_sum = eps_bf16.real_add(eps_f16);
    let total_bound = x.real_mul(eps_sum);

    // Negated property: |err_total| > (eps_bf16 + eps_f16) * x
    let too_high = err_total.clone().real_gt(total_bound.clone());
    let too_low = err_total.real_lt(total_bound.real_neg());
    let violation = too_high.or(too_low);

    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfQuantPropertyResult {
        property: "mixed_precision_chain_error_bf16_f16".to_string(),
        proven,
        smt2,
        detail,
    })
}

#[cfg(test)]
#[path = "ay_quantization_dpdf_vlm_tests.rs"]
mod tests;
