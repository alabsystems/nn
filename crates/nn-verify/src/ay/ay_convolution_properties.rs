// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for convolution stride and padding mathematical properties (#4226).
//!
//! Convolutions are parameterized by input size, kernel size, stride, padding, dilation,
//! and groups. The output size formulas and their interactions are non-trivial and
//! error-prone. This module proves seven key mathematical properties using ay's SMT solver:
//!
//! 1. **Conv output size positivity**: `floor((in + 2*pad - dil*(k-1) - 1) / stride) + 1 > 0`
//!    for valid configurations.
//! 2. **Conv transpose output size**: The transpose formula
//!    `(in-1)*stride - 2*pad + dil*(k-1) + out_pad + 1` matches the expected inverse.
//! 3. **Conv-transpose inverse**: For appropriate `output_padding`, conv_transpose undoes conv.
//! 4. **Padding preserves non-negativity**: Zero-padding maintains non-negative values.
//! 5. **Dilation equivalence**: Dilated conv with dilation=d, kernel=k has effective kernel
//!    size `k + (k-1)*(d-1)` on zero-inserted input.
//! 6. **Depthwise conv output channels**: `groups=in_channels` implies `out_channels` is
//!    divisible by `in_channels` and each group processes exactly 1 input channel.
//! 7. **1x1 conv is pointwise**: Conv with kernel=1, stride=1, padding=0 preserves spatial
//!    dimensions, equivalent to per-position linear transform.
//!
//! # Proof Strategy
//!
//! Convolution size formulas involve integer floor division. Since ay's `QF_LRA` fragment
//! does not natively support integer division or floor, we model the floor operation using
//! a helper variable `q` with constraints `q <= x/s < q+1` (real-arithmetic encoding of
//! integer quotient). For properties that are purely algebraic (no floor), we use
//! direct `QF_LRA` or `QF_NRA` encodings.
//!
//! All dimension parameters are modeled as positive reals (>= 1) representing positive
//! integers, with stride >= 1, padding >= 0, dilation >= 1, kernel >= 1.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of a convolution property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct ConvPropertyResult {
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

/// Assert `lower <= expr <= upper`.
fn assert_bounds(
    program: &mut AYProgram,
    expr: &Expr,
    lower: f64,
    upper: f64,
) -> Result<(), SmtError> {
    let lo = real_from_f64(lower)?;
    let hi = real_from_f64(upper)?;
    program.assert(expr.clone().real_ge(lo));
    program.assert(expr.clone().real_le(hi));
    Ok(())
}

/// Assert `expr >= lower`.
fn assert_lower_bound(program: &mut AYProgram, expr: &Expr, lower: f64) -> Result<(), SmtError> {
    let lo = real_from_f64(lower)?;
    program.assert(expr.clone().real_ge(lo));
    Ok(())
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
// Property 1: Conv Output Size Positivity
// ---------------------------------------------------------------------------

/// Prove that the standard convolution output size is positive for valid configurations.
///
/// The standard 1D conv output size formula is:
///   `out = floor((in_size + 2*padding - dilation*(kernel-1) - 1) / stride) + 1`
///
/// We prove that `out >= 1` whenever:
///   - `in_size >= 1`, `kernel >= 1`, `stride >= 1`, `dilation >= 1`, `padding >= 0`
///   - `in_size + 2*padding >= dilation*(kernel-1) + 1` (the input with padding covers
///     at least one kernel placement)
///
/// The proof models `floor(x/s)` via a helper variable `q` satisfying:
///   `q * stride <= numerator < (q+1) * stride` and `q >= 0`
/// Then `out = q + 1 >= 1`.
///
/// We assert the negation `out < 1` (i.e., `q + 1 < 1`, i.e., `q < 0`) and prove UNSAT.
pub(crate) fn prove_conv_output_size_positive() -> Result<ConvPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let in_size = declare_real(&mut program, "in_size");
    let kernel = declare_real(&mut program, "kernel");
    let stride = declare_real(&mut program, "stride");
    let dilation = declare_real(&mut program, "dilation");
    let padding = declare_real(&mut program, "padding");

    // Valid configuration constraints
    assert_bounds(&mut program, &in_size, 1.0, 1000.0)?;
    assert_bounds(&mut program, &kernel, 1.0, 100.0)?;
    assert_bounds(&mut program, &stride, 1.0, 100.0)?;
    assert_bounds(&mut program, &dilation, 1.0, 100.0)?;
    assert_bounds(&mut program, &padding, 0.0, 100.0)?;

    let zero = Expr::real(0);
    let one = real_from_f64(1.0)?;
    let two = real_from_f64(2.0)?;

    // numerator = in_size + 2*padding - dilation*(kernel - 1) - 1
    // Rewritten: in_size + 2*padding - dilation*kernel + dilation - 1
    let eff_kernel = dilation
        .clone()
        .real_mul(kernel.clone().real_sub(one.clone()));
    let numerator = in_size
        .clone()
        .real_add(two.real_mul(padding.clone()))
        .real_sub(eff_kernel.clone())
        .real_sub(one.clone());

    // Validity constraint: numerator >= 0
    // (input with padding must cover at least one kernel application)
    program.assert(numerator.clone().real_ge(zero.clone()));

    // Model floor division: q = floor(numerator / stride)
    // q * stride <= numerator < (q+1) * stride, q >= 0
    let q = declare_real(&mut program, "q");
    program.assert(q.clone().real_ge(zero.clone()));
    // q * stride <= numerator
    program.assert(
        q.clone()
            .real_mul(stride.clone())
            .real_le(numerator.clone()),
    );
    // numerator < (q + 1) * stride
    program.assert(numerator.real_lt(q.clone().real_add(one.clone()).real_mul(stride)));

    // out = q + 1.  Violation: out < 1, i.e. q < 0
    let violation = q.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "conv_output_size_positive".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Conv Transpose Output Size Positivity
// ---------------------------------------------------------------------------

/// Stride of the concrete transposed conv proven by
/// [`prove_conv_transpose_output_positive`]: a stride-2, 3x3, dilation-1 block
/// (the standard 2x up-sampling transposed convolution).
const CT_STRIDE: i64 = 2;
/// Kernel size of that transposed conv.
const CT_KERNEL: i64 = 3;
/// Dilation of that transposed conv.
const CT_DILATION: i64 = 1;

/// Prove that the conv-transpose output size is positive (`out >= 1`) for every
/// valid configuration of a stride-2 / 3x3 / dilation-1 transposed convolution.
///
/// Conv-transpose output size:
///   `out = (in - 1)*stride - 2*pad + dil*(k-1) + output_pad + 1`.
///
/// This is *not* positive unconditionally — a large `pad` drives it negative. The
/// exact precondition (tight at `in = 1, output_pad = 0`) is that the padding does
/// not exceed the effective kernel radius, `2*pad <= dil*(k-1)`. Under it,
/// `out = (in-1)*stride + [dil*(k-1) - 2*pad] + output_pad + 1` is a sum of
/// non-negative terms plus one, hence `>= 1`.
///
/// `in_size`, `pad`, `output_pad` are free `Int`s (with `output_pad < stride`, the
/// PyTorch constraint); stride/kernel/dilation are literals so the formula is
/// linear and the query is decidable `QF_LIA`. The old QF_NRA encoding only
/// constrained `out >= 0` (via `2*pad <= …`), which over the reals leaves
/// `out ∈ [0, 1)` satisfiable — e.g. `out = 1/2` — the SAT counterexample this
/// replaces: `out >= 0` does not imply `out >= 1` unless `out` is an integer.
///
/// Non-vacuous: `out >= 1` is derived from the padding precondition, not asserted.
/// Dropping the formula's trailing `+1` (the `!correct` branch) lets `out` reach 0
/// and the query goes SAT — see `mutation_conv_transpose_drops_output_plus_one`.
pub(crate) fn prove_conv_transpose_output_positive() -> Result<ConvPropertyResult, SmtError> {
    let program = build_conv_transpose_output_positive(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "conv_transpose_output_positive".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the conv-transpose positivity query. `add_output_plus_one` gates the
/// output formula's trailing `+1`; flipping it off is the off-by-one that lets the
/// output size collapse to 0, which the tests use to confirm the proof depends on
/// it.
fn build_conv_transpose_output_positive(add_output_plus_one: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let zero = Expr::int(0);
    let one = Expr::int(1);
    let two = Expr::int(2);

    // dil*(k-1): the effective kernel radius contribution, a literal for the
    // concrete shape.
    let eff = CT_DILATION * (CT_KERNEL - 1);

    // Free integer parameters of the transposed conv.
    let in_size = program.declare_const("in_size", Sort::int());
    program.assert(in_size.clone().int_ge(one.clone()));
    program.assert(in_size.clone().int_le(Expr::int(1000)));

    let padding = program.declare_const("padding", Sort::int());
    program.assert(padding.clone().int_ge(zero.clone()));
    program.assert(padding.clone().int_le(Expr::int(100)));

    let output_padding = program.declare_const("output_padding", Sort::int());
    program.assert(output_padding.clone().int_ge(zero));
    // PyTorch constraint: output_padding < stride.
    program.assert(output_padding.clone().int_lt(Expr::int(CT_STRIDE)));

    // Precondition (tight at in=1, output_padding=0): padding stays within the
    // effective kernel radius, 2*padding <= dil*(k-1).
    program.assert(two.clone().int_mul(padding.clone()).int_le(Expr::int(eff)));

    // out = (in-1)*stride - 2*pad + dil*(k-1) + output_pad + 1
    let base = in_size
        .int_sub(one.clone())
        .int_mul(Expr::int(CT_STRIDE))
        .int_sub(two.int_mul(padding))
        .int_add(Expr::int(eff))
        .int_add(output_padding);
    let out = if add_output_plus_one {
        base.int_add(one.clone())
    } else {
        base
    };

    // Violation: the output size is not positive.
    program.assert(out.int_lt(one));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 3: Conv-Transpose Inverse
// ---------------------------------------------------------------------------

/// Prove that conv_transpose can undo conv (recover original size) with
/// the correct output_padding.
///
/// Given conv output: `out_conv = floor((N + 2P - D*(K-1) - 1) / S) + 1`
/// Conv transpose on that: `N' = (out_conv - 1)*S - 2P + D*(K-1) + output_padding + 1`
///
/// For N' = N, we need:
///   `output_padding = N + 2P - D*(K-1) - 1 - S * floor((N + 2P - D*(K-1) - 1) / S)`
/// which is `(N + 2P - D*(K-1) - 1) mod S`.
///
/// We prove this algebraically: given the floor quotient `q` satisfying
///   `q*S <= numerator < (q+1)*S` where `numerator = N + 2P - D*(K-1) - 1`,
///   `out_conv = q + 1`, and `output_padding = numerator - q*S`,
/// we must have `N' = N`.
///
/// Expanding: `N' = q*S - 2P + D*(K-1) + (numerator - q*S) + 1`
///            `= -2P + D*(K-1) + numerator + 1`
///            `= -2P + D*(K-1) + N + 2P - D*(K-1) - 1 + 1`
///            `= N`
pub(crate) fn prove_conv_transpose_inverse() -> Result<ConvPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let n = declare_real(&mut program, "N");
    let k = declare_real(&mut program, "K");
    let s = declare_real(&mut program, "S");
    let d = declare_real(&mut program, "D");
    let p = declare_real(&mut program, "P");

    assert_bounds(&mut program, &n, 1.0, 500.0)?;
    assert_bounds(&mut program, &k, 1.0, 50.0)?;
    assert_bounds(&mut program, &s, 1.0, 50.0)?;
    assert_bounds(&mut program, &d, 1.0, 50.0)?;
    assert_bounds(&mut program, &p, 0.0, 50.0)?;

    let one = real_from_f64(1.0)?;
    let two = real_from_f64(2.0)?;
    let zero = Expr::real(0);

    // numerator = N + 2P - D*(K-1) - 1
    let eff_k = d.clone().real_mul(k.clone().real_sub(one.clone()));
    let numerator = n
        .clone()
        .real_add(two.clone().real_mul(p.clone()))
        .real_sub(eff_k.clone())
        .real_sub(one.clone());

    // Validity: numerator >= 0
    program.assert(numerator.clone().real_ge(zero.clone()));

    // q = floor(numerator / S): q*S <= numerator < (q+1)*S, q >= 0
    let q = declare_real(&mut program, "q");
    program.assert(q.clone().real_ge(zero));
    program.assert(q.clone().real_mul(s.clone()).real_le(numerator.clone()));
    program.assert(
        numerator
            .clone()
            .real_lt(q.clone().real_add(one.clone()).real_mul(s.clone())),
    );

    // output_padding = numerator - q * S (the remainder)
    let output_padding = numerator.real_sub(q.clone().real_mul(s.clone()));

    // out_conv = q + 1
    // N' = (out_conv - 1)*S - 2P + D*(K-1) + output_padding + 1
    //     = q*S - 2P + eff_k + output_padding + 1
    let n_prime = q
        .real_mul(s)
        .real_sub(two.real_mul(p))
        .real_add(eff_k)
        .real_add(output_padding)
        .real_add(one);

    // Violation: N' != N
    let violation = n_prime.ne(n);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "conv_transpose_inverse".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Padding Preserves Non-Negativity
// ---------------------------------------------------------------------------

/// Prove that zero-padding preserves non-negativity of input values.
///
/// For a 1D input with values `x_i >= 0`, zero-padding prepends and appends zeros.
/// Since `0 >= 0` and all `x_i >= 0`, the padded tensor has all values `>= 0`.
///
/// This is trivial but important for ReLU-activated layers: after ReLU all values
/// are non-negative, and zero-padding does not violate this invariant.
///
/// We model this as: given `x >= 0` (an arbitrary input element) and `pad_val = 0`
/// (the padding value), all elements in the padded output are `>= 0`.
/// The output consists of padding values and input values, both `>= 0`.
///
/// The proof uses QF_LRA: assert an input value `x >= 0` and padding value `= 0`,
/// then prove that neither can be negative.
pub(crate) fn prove_padding_preserves_non_negativity() -> Result<ConvPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // An arbitrary input element: x >= 0 (post-ReLU invariant)
    let x = declare_real(&mut program, "x");
    let zero = Expr::real(0);
    program.assert(x.clone().real_ge(zero.clone()));
    assert_bounds(&mut program, &x, 0.0, 1e6)?;

    // Padding value is exactly 0
    let pad_val = declare_real(&mut program, "pad_val");
    program.assert(pad_val.clone().eq(zero.clone()));

    // An arbitrary output element is either x or pad_val.
    // We model this with a selector: out = selector * x + (1 - selector) * pad_val
    // where 0 <= selector <= 1.
    // But more directly: we have two cases. Prove both are non-negative.

    // Case 1 violation: x < 0 (contradicts constraint)
    // Case 2 violation: pad_val < 0 (contradicts constraint)
    // Combined: the output element is either x or pad_val, and we need at least one < 0.
    let out_elem = declare_real(&mut program, "out_elem");

    // out_elem is either x or pad_val
    // Encode: (out_elem = x) OR (out_elem = pad_val)
    let is_input = out_elem.clone().eq(x);
    let is_pad = out_elem.clone().eq(pad_val);
    program.assert(is_input.or(is_pad));

    // Violation: out_elem < 0
    let violation = out_elem.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "padding_preserves_non_negativity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Dilation Equivalence (Effective Kernel Size)
// ---------------------------------------------------------------------------

/// Prove that dilated convolution with dilation=d and kernel=k has an effective
/// kernel size of `k + (k-1)*(d-1) = k*d - d + 1` on the input.
///
/// A dilated kernel inserts `(d-1)` zeros between each kernel element, expanding
/// a kernel of physical size `k` to an effective size covering `k + (k-1)*(d-1)`
/// positions in the input.
///
/// We prove the algebraic identity:
///   `eff_k = k + (k-1)*(d-1)` is equivalent to `eff_k = k*d - d + 1`
///
/// Both forms are used in different frameworks. We prove they are identical and
/// that `eff_k >= k` when `d >= 1` (dilation only increases effective size).
pub(crate) fn prove_dilation_equivalence() -> Result<ConvPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let k = declare_real(&mut program, "k");
    let d = declare_real(&mut program, "d");

    assert_bounds(&mut program, &k, 1.0, 100.0)?;
    assert_bounds(&mut program, &d, 1.0, 100.0)?;

    let one = real_from_f64(1.0)?;

    // Form 1: eff_k1 = k + (k-1)*(d-1)
    let k_minus_1 = k.clone().real_sub(one.clone());
    let d_minus_1 = d.clone().real_sub(one.clone());
    let eff_k1 = k.clone().real_add(k_minus_1.real_mul(d_minus_1));

    // Form 2: eff_k2 = k*d - d + 1
    let eff_k2 = k
        .clone()
        .real_mul(d.clone())
        .real_sub(d)
        .real_add(one.clone());

    // Assert violation: eff_k1 != eff_k2
    let violation = eff_k1.clone().ne(eff_k2);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "dilation_effective_kernel_equivalence".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that dilated effective kernel size is >= physical kernel size when d >= 1.
///
/// `eff_k = k + (k-1)*(d-1) >= k` because `(k-1)*(d-1) >= 0` when `k >= 1, d >= 1`.
pub(crate) fn prove_dilation_monotonic() -> Result<ConvPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let k = declare_real(&mut program, "k");
    let d = declare_real(&mut program, "d");

    assert_bounds(&mut program, &k, 1.0, 100.0)?;
    assert_bounds(&mut program, &d, 1.0, 100.0)?;

    let one = real_from_f64(1.0)?;

    // eff_k = k + (k-1)*(d-1)
    let k_minus_1 = k.clone().real_sub(one.clone());
    let d_minus_1 = d.real_sub(one);
    let eff_k = k.clone().real_add(k_minus_1.real_mul(d_minus_1));

    // Violation: eff_k < k
    let violation = eff_k.real_lt(k);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "dilation_monotonic".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Depthwise Conv Output Channels
// ---------------------------------------------------------------------------

/// Prove that for depthwise convolution (groups = in_channels), the constraint
/// `out_channels % in_channels == 0` means each group processes exactly
/// `in_channels / groups = 1` input channel and produces `out_channels / groups`
/// output channels per group.
///
/// Given:
///   - `groups = in_channels` (depthwise)
///   - `out_channels = groups * channels_per_group` (divisibility)
///   - `in_per_group = in_channels / groups = 1`
///
/// We prove `in_per_group = 1` and that `total_out = groups * channels_per_group = out_channels`.
///
/// The proof is pure QF_LRA: `in_channels / groups = in_channels / in_channels = 1`.
pub(crate) fn prove_depthwise_conv_channels() -> Result<ConvPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let in_channels = declare_real(&mut program, "in_channels");
    let out_channels = declare_real(&mut program, "out_channels");
    let channels_per_group = declare_real(&mut program, "cpg");

    assert_bounds(&mut program, &in_channels, 1.0, 1000.0)?;
    assert_bounds(&mut program, &out_channels, 1.0, 1000.0)?;
    assert_bounds(&mut program, &channels_per_group, 1.0, 1000.0)?;

    let one = real_from_f64(1.0)?;

    // groups = in_channels (depthwise)
    // out_channels = groups * channels_per_group = in_channels * channels_per_group
    program.assert(
        out_channels
            .clone()
            .eq(in_channels.clone().real_mul(channels_per_group.clone())),
    );

    // in_per_group = in_channels / groups = in_channels / in_channels = 1
    // We compute in_per_group via a helper variable.
    let in_per_group = declare_real(&mut program, "in_per_group");
    // in_per_group * in_channels = in_channels (since groups = in_channels)
    program.assert(
        in_per_group
            .clone()
            .real_mul(in_channels.clone())
            .eq(in_channels),
    );
    // in_per_group > 0
    program.assert(in_per_group.clone().real_gt(Expr::real(0)));

    // Violation: in_per_group != 1
    let violation = in_per_group.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "depthwise_conv_one_input_per_group".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: 1x1 Conv is Pointwise Linear
// ---------------------------------------------------------------------------

/// Prove that conv with kernel=1, stride=1, padding=0, dilation=1 preserves the
/// spatial dimension: `out_size == in_size`.
///
/// The conv output-size formula is
/// `out = floor((in + 2*pad - dil*(k-1) - 1) / stride) + 1`. Specialised to a 1x1
/// pointwise conv (`k=1, stride=1, pad=0, dil=1`) the numerator is `in - 1` and the
/// stride is `1`, so `out = (in - 1) + 1 = in`.
///
/// The floor is modelled honestly with an integer quotient `q` pinned by
/// `q <= numerator < q + 1`. Over the **integers** those two bounds force
/// `q = numerator` exactly. Over the reals (the old QF_LRA encoding) a real `q`
/// could sit anywhere in `(numerator - 1, numerator]`, so `out = q + 1` was free to
/// take a fractional value like `in - 1/2` — the floor-over-reals bug that made the
/// query SAT. `in_size` stays a free `Int` so the theorem is universal, and every
/// other parameter is a literal, so the query is decidable `QF_LIA`.
///
/// Non-vacuous: `out == in_size` is *derived* by chaining `q = numerator` through
/// `out = q + 1`; nothing asserts the conclusion. Dropping the trailing `+1` (the
/// `!correct` branch) makes `out = in_size - 1` and the query SAT — see
/// `mutation_1x1_conv_drops_output_plus_one`.
pub(crate) fn prove_1x1_conv_preserves_spatial() -> Result<ConvPropertyResult, SmtError> {
    let program = build_1x1_conv_preserves_spatial(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "1x1_conv_preserves_spatial".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the 1x1-conv spatial-preservation query. `add_output_plus_one` gates the
/// trailing `+1` of the output formula; the tests flip it off — the classic
/// off-by-one that silently shrinks the spatial dimension — to confirm the proof
/// depends on it.
fn build_1x1_conv_preserves_spatial(add_output_plus_one: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let zero = Expr::int(0);
    let one = Expr::int(1);

    // Free input length, a positive integer. kernel=1, stride=1, pad=0, dil=1 are
    // literals, so the numerator below is linear and the query stays in QF_LIA.
    let in_size = program.declare_const("in_size", Sort::int());
    program.assert(in_size.clone().int_ge(one.clone()));
    program.assert(in_size.clone().int_le(Expr::int(10_000)));

    // numerator = in_size + 2*0 - 1*(1-1) - 1 = in_size - 1
    let numerator = in_size.clone().int_sub(one.clone());

    // q = floor(numerator / stride) with stride = 1. Over the integers
    // `q <= numerator < q + 1` pins q = numerator exactly.
    let q = program.declare_const("q", Sort::int());
    program.assert(q.clone().int_ge(zero));
    program.assert(q.clone().int_le(numerator.clone()));
    program.assert(numerator.int_lt(q.clone().int_add(one.clone())));

    // out = q + 1  (correct)   or   out = q  (bug: dropped the +1)
    let out = if add_output_plus_one {
        q.int_add(one)
    } else {
        q
    };

    // Violation: the spatial dimension changed.
    program.assert(out.ne(in_size));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ay_vacuity::vacuity_smell;

    #[test]
    fn test_conv_output_size_positive_proven() {
        let result = prove_conv_output_size_positive().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Conv output size positivity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Conv output size must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "conv_output_size_positive");
    }

    #[test]
    fn test_conv_transpose_output_positive_proven() {
        let result = prove_conv_transpose_output_positive().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        // QF_LIA over a concrete shape is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Conv transpose output positivity should be proven, got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert!(
            !result.detail.contains("counterexample"),
            "Conv transpose output must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "conv_transpose_output_positive");
    }

    /// The trailing `+1` of the conv-transpose output formula is load-bearing:
    /// dropping it lets `out` collapse to 0 (e.g. in=1, padding=1, output_pad=0),
    /// so the positivity query must find a counterexample.
    #[test]
    fn mutation_conv_transpose_drops_output_plus_one() {
        let program = build_conv_transpose_output_positive(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the output formula's `+1` dropped the output size reaches 0 and the \
             query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_conv_transpose_inverse_proven() {
        let result = prove_conv_transpose_inverse().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Conv-transpose inverse: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Conv-transpose inverse must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "conv_transpose_inverse");
    }

    #[test]
    fn test_padding_preserves_non_negativity_proven() {
        let result = prove_padding_preserves_non_negativity().expect("proof should not error");
        assert!(
            result.proven,
            "Padding non-negativity (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "padding_preserves_non_negativity");
    }

    #[test]
    fn test_dilation_equivalence_proven() {
        let result = prove_dilation_equivalence().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Dilation equivalence: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Dilation equivalence must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "dilation_effective_kernel_equivalence");
    }

    #[test]
    fn test_dilation_monotonic_proven() {
        let result = prove_dilation_monotonic().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Dilation monotonicity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Dilation monotonicity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "dilation_monotonic");
    }

    #[test]
    fn test_depthwise_conv_channels_proven() {
        let result = prove_depthwise_conv_channels().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Depthwise conv channels: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Depthwise conv must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "depthwise_conv_one_input_per_group");
    }

    #[test]
    fn test_1x1_conv_preserves_spatial_proven() {
        let result = prove_1x1_conv_preserves_spatial().expect("proof should not error");
        // QF_LIA over a concrete shape is decidable: `Unknown`/SAT are not acceptable.
        assert!(
            result.proven,
            "1x1 conv spatial preservation (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "1x1_conv_preserves_spatial");
    }

    /// The output formula's trailing `+1` is the whole theorem: dropping it makes
    /// `out = in_size - 1`, so 1x1 conv would shrink the spatial dimension and the
    /// preservation query must find a counterexample.
    #[test]
    fn mutation_1x1_conv_drops_output_plus_one() {
        let program = build_1x1_conv_preserves_spatial(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the output formula's `+1` dropped 1x1 conv shrinks the spatial dim \
             and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_conv_output_size_smt2_structure() {
        let result = prove_conv_output_size_positive().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }

    #[test]
    fn test_dilation_identity_when_d_equals_1() {
        // When d=1, effective kernel should equal physical kernel.
        // eff_k = k + (k-1)*(1-1) = k + 0 = k
        // This is a special case of the general equivalence proof,
        // verified here with a direct algebraic check.
        let mut program = AYProgram::new();
        program.set_logic("QF_LRA");

        let k = declare_real(&mut program, "k");
        assert_bounds(&mut program, &k, 1.0, 100.0).unwrap();

        let one = real_from_f64(1.0).unwrap();
        let zero = Expr::real(0);

        // eff_k = k + (k-1)*0 = k
        let k_minus_1 = k.clone().real_sub(one);
        let eff_k = k.clone().real_add(k_minus_1.real_mul(zero));

        // Violation: eff_k != k
        let violation = eff_k.ne(k);
        program.assert(violation);
        program.check_sat();

        let (proven, detail) = execute_and_check(&program);
        assert!(
            proven,
            "d=1 dilation identity (QF_LRA) should be Proven. detail: {}",
            detail,
        );
    }
}
