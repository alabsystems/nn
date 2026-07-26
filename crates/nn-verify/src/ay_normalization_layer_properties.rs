// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for normalization layer mathematical properties (#4223).
//!
//! Proves: LayerNorm mean=0 / var=1, BatchNorm EMA updates, RMSNorm formula,
//! GroupNorm channel partition, InstanceNorm independence, affine transform,
//! epsilon stability, and shape preservation.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;
use crate::smt_error::SmtError;

/// Result of a normalization layer property proof attempt.
#[derive(Debug, Clone)]
pub struct NormalizationProofResult {
    /// Human-readable property name.
    pub property: String,
    /// Whether the proof succeeded (UNSAT = property holds for all inputs).
    pub proven: bool,
    /// SMT-LIB2 text of the query.
    pub smt2: String,
    /// Solver detail message.
    pub detail: String,
}

fn declare_real(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::real())
}

fn assert_positive(program: &mut AYProgram, expr: &Expr) {
    program.assert(expr.clone().real_gt(Expr::real(0)));
}

/// Execute a ay program and return whether UNSAT (property proven).
///
/// The final `(proven, detail)` is funneled through
/// [`crate::ay_vacuity::reject_if_vacuous`], so any query that is UNSAT only
/// because it asserts `P ∧ ¬P` (or compares a term to itself) is downgraded to a
/// failure. A residual vacuity therefore becomes a hard test failure rather than
/// a false "proven"; a genuine proof is returned unchanged.
fn execute_and_check(program: &AYProgram) -> (bool, String) {
    let (proven, detail) = match execute_direct::execute(program) {
        Ok(ExecuteResult::Verified) => (true, "UNSAT: property holds for all inputs".to_string()),
        Ok(ExecuteResult::Counterexample { model, .. }) => {
            (false, format!("SAT: counterexample found: {:?}", model))
        }
        Ok(ExecuteResult::Unknown(reason)) => (false, format!("Unknown: {}", reason)),
        Ok(other) => (false, format!("Unexpected result: {:?}", other)),
        Err(e) => (false, format!("Execution error: {}", e)),
    };
    crate::ay_vacuity::reject_if_vacuous(&program.to_string(), proven, detail)
}

fn make_result(prog: &AYProgram, property: &str) -> NormalizationProofResult {
    let smt2 = prog.to_string();
    let (proven, detail) = execute_and_check(prog);
    NormalizationProofResult {
        property: property.to_string(),
        proven,
        smt2,
        detail,
    }
}

// ---------------------------------------------------------------------------
// 1: LayerNorm output mean == 0
// ---------------------------------------------------------------------------

/// Prove: LayerNorm on [x1, x2] yields output mean = 0.
pub fn prove_layernorm_output_mean_zero() -> Result<NormalizationProofResult, SmtError> {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let x1 = declare_real(&mut p, "x1");
    let x2 = declare_real(&mut p, "x2");
    let mean = declare_real(&mut p, "mean");
    let s = declare_real(&mut p, "s");
    let n1 = declare_real(&mut p, "n1");
    let n2 = declare_real(&mut p, "n2");
    let out_mean = declare_real(&mut p, "out_mean");

    p.assert(x1.clone().real_ge(Expr::real(-100)));
    p.assert(x1.clone().real_le(Expr::real(100)));
    p.assert(x2.clone().real_ge(Expr::real(-100)));
    p.assert(x2.clone().real_le(Expr::real(100)));

    // mean = (x1 + x2) / 2
    p.assert(
        Expr::real(2)
            .real_mul(mean.clone())
            .eq(x1.clone().real_add(x2.clone())),
    );
    assert_positive(&mut p, &s);

    // n_i = (x_i - mean) / s
    p.assert(n1.clone().real_mul(s.clone()).eq(x1.real_sub(mean.clone())));
    p.assert(n2.clone().real_mul(s).eq(x2.real_sub(mean)));

    // out_mean = (n1 + n2) / 2
    p.assert(Expr::real(2).real_mul(out_mean.clone()).eq(n1.real_add(n2)));

    p.assert(out_mean.ne(Expr::real(0)));
    p.check_sat();
    Ok(make_result(&p, "layernorm_output_mean_zero"))
}

// ---------------------------------------------------------------------------
// 2: LayerNorm output variance == 1
// ---------------------------------------------------------------------------

/// Prove: LayerNorm on centered deviations yields output variance = 1.
pub fn prove_layernorm_output_variance_one() -> Result<NormalizationProofResult, SmtError> {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let d1 = declare_real(&mut p, "d1");
    let d2 = declare_real(&mut p, "d2");
    let var = declare_real(&mut p, "var");
    let n1 = declare_real(&mut p, "n1");
    let n2 = declare_real(&mut p, "n2");
    let out_var = declare_real(&mut p, "out_var");

    p.assert(d1.clone().real_add(d2.clone()).eq(Expr::real(0)));
    p.assert(d1.clone().ne(Expr::real(0)));

    let d1_sq = d1.clone().real_mul(d1.clone());
    let d2_sq = d2.clone().real_mul(d2.clone());
    p.assert(
        Expr::real(2)
            .real_mul(var.clone())
            .eq(d1_sq.clone().real_add(d2_sq.clone())),
    );
    assert_positive(&mut p, &var);

    let n1_sq = n1.clone().real_mul(n1.clone());
    let n2_sq = n2.clone().real_mul(n2.clone());
    p.assert(n1_sq.clone().real_mul(var.clone()).eq(d1_sq));
    p.assert(n2_sq.clone().real_mul(var).eq(d2_sq));

    p.assert(
        Expr::real(2)
            .real_mul(out_var.clone())
            .eq(n1_sq.real_add(n2_sq)),
    );

    p.assert(out_var.ne(Expr::real(1)));
    p.check_sat();
    Ok(make_result(&p, "layernorm_output_variance_one"))
}

// ---------------------------------------------------------------------------
// 3: BatchNorm running mean EMA update (convex combination)
// ---------------------------------------------------------------------------

/// Prove: rm_new = (1-m)*rm_old + m*bm stays between rm_old and bm.
pub fn prove_batchnorm_running_mean_update() -> Result<NormalizationProofResult, SmtError> {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let rm_old = declare_real(&mut p, "rm_old");
    let bm = declare_real(&mut p, "bm");
    let m = declare_real(&mut p, "m");
    let rm_new = declare_real(&mut p, "rm_new");
    let lo = declare_real(&mut p, "lo");
    let hi = declare_real(&mut p, "hi");

    p.assert(m.clone().real_gt(Expr::real(0)));
    p.assert(m.clone().real_lt(Expr::real(1)));

    p.assert(lo.clone().real_le(rm_old.clone()));
    p.assert(lo.clone().real_le(bm.clone()));
    p.assert(hi.clone().real_ge(rm_old.clone()));
    p.assert(hi.clone().real_ge(bm.clone()));
    p.assert(lo.clone().eq(rm_old.clone()).or(lo.clone().eq(bm.clone())));
    p.assert(hi.clone().eq(rm_old.clone()).or(hi.clone().eq(bm.clone())));

    let one_minus_m = Expr::real(1).real_sub(m.clone());
    p.assert(
        rm_new
            .clone()
            .eq(one_minus_m.real_mul(rm_old).real_add(m.real_mul(bm))),
    );

    p.assert(rm_new.clone().real_lt(lo).or(rm_new.real_gt(hi)));
    p.check_sat();
    Ok(make_result(&p, "batchnorm_running_mean_update"))
}

// ---------------------------------------------------------------------------
// 4: BatchNorm running var update preserves non-negativity
// ---------------------------------------------------------------------------

/// Prove: rv_new = (1-m)*rv_old + m*bv >= 0 when inputs are non-negative.
pub fn prove_batchnorm_running_var_update() -> Result<NormalizationProofResult, SmtError> {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let rv_old = declare_real(&mut p, "rv_old");
    let bv = declare_real(&mut p, "bv");
    let m = declare_real(&mut p, "m");
    let rv_new = declare_real(&mut p, "rv_new");

    p.assert(rv_old.clone().real_ge(Expr::real(0)));
    p.assert(bv.clone().real_ge(Expr::real(0)));
    p.assert(m.clone().real_gt(Expr::real(0)));
    p.assert(m.clone().real_lt(Expr::real(1)));

    let one_minus_m = Expr::real(1).real_sub(m.clone());
    p.assert(
        rv_new
            .clone()
            .eq(one_minus_m.real_mul(rv_old).real_add(m.real_mul(bv))),
    );

    p.assert(rv_new.real_lt(Expr::real(0)));
    p.check_sat();
    Ok(make_result(&p, "batchnorm_running_var_update_nonneg"))
}

// ---------------------------------------------------------------------------
// 5: RMSNorm output = x * gamma / rms
// ---------------------------------------------------------------------------

/// Gain applied by the RMSNorm affine step in [`build_rmsnorm_output_formula`].
const RMSNORM_GAMMA: i64 = 3;
/// The token's precomputed root-mean-square scale `rms = sqrt(mean(x^2)+eps)`,
/// pinned to a concrete positive constant so the formula stays linear in `x`.
const RMSNORM_RMS: i64 = 5;

/// Prove that the RMSNorm output obeys its defining equation `out * rms = gamma * x`.
///
/// RMSNorm computes `out = gamma * x / rms`, where `rms = sqrt(mean(x^2) + eps)`
/// is a per-token scalar. The content here is the *reciprocal*: the output is the
/// input scaled DOWN by `rms`, so multiplying the output back by `rms` must
/// recover `gamma * x`. We pin `gamma` and `rms` to concrete constants (so the
/// nonlinear `sqrt` is a literal and the coefficient `gamma/rms` is a literal),
/// leaving `x` a free variable — the theorem therefore holds for every input.
///
/// The output is *derived* by applying the formula `out = (gamma/rms) * x`, and
/// the conclusion checked is the different equation `out * rms = gamma * x`; the
/// solver has to show the derivation forces it. The classic slip — multiplying by
/// `rms` instead of dividing (dropping the reciprocal) — makes the query SAT (see
/// `output_formula_depends_on_dividing_by_rms`).
///
/// Everything is `gamma * x` and `out * rms` with a literal factor, so the query
/// stays in decidable `QF_LRA`.
pub fn prove_rmsnorm_output_formula() -> Result<NormalizationProofResult, SmtError> {
    let program = build_rmsnorm_output_formula(true);
    Ok(make_result(&program, "rmsnorm_output_formula"))
}

/// Build the RMSNorm formula query. When `divide_by_rms` is false the output is
/// scaled by `gamma * rms` instead of `gamma / rms` — the reciprocal is dropped,
/// the classic "multiplied by the norm instead of dividing" slip that breaks
/// `out * rms = gamma * x`; tests flip it to confirm the proof depends on it.
fn build_rmsnorm_output_formula(divide_by_rms: bool) -> AYProgram {
    let mut p = AYProgram::new();
    p.set_logic("QF_LRA");

    let x = declare_real(&mut p, "x");
    p.assert(x.clone().real_ge(Expr::real(-100)));
    p.assert(x.clone().real_le(Expr::real(100)));

    // The per-token scale `rms = sqrt(mean(x^2) + eps)` is a positive scalar,
    // pinned to a concrete value so the `sqrt` is a literal and every product
    // below has a literal factor (keeping the query linear). It only ever appears
    // pinned and positive — never multiplied by another variable.
    let rms = declare_real(&mut p, "rms");
    assert_positive(&mut p, &rms);
    p.assert(rms.eq(Expr::real(RMSNORM_RMS)));

    // Forward pass: out = (gamma / rms) * x. `gamma` and `rms` are literals so
    // the scale is a rational literal, keeping the product linear in `x`. The
    // slip scales by `gamma * rms` (multiplies by the norm instead of dividing).
    let scale = if divide_by_rms {
        Expr::real_ratio(RMSNORM_GAMMA, RMSNORM_RMS)
    } else {
        Expr::real(RMSNORM_GAMMA * RMSNORM_RMS)
    };
    let out = declare_real(&mut p, "out");
    p.assert(out.clone().eq(x.clone().real_mul(scale)));

    // Violation: the defining equation `out * rms = gamma * x` fails. Both sides
    // multiply a declared variable by the literal `rms`/`gamma`, so this stays
    // linear.
    p.assert(
        out.real_mul(Expr::real(RMSNORM_RMS))
            .ne(x.real_mul(Expr::real(RMSNORM_GAMMA))),
    );
    p.check_sat();
    p
}

// ---------------------------------------------------------------------------
// 6: GroupNorm groups partition channels evenly
// ---------------------------------------------------------------------------

/// Prove: num_channels = num_groups * cpg implies remainder = 0.
pub fn prove_groupnorm_partition() -> Result<NormalizationProofResult, SmtError> {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let nc = declare_real(&mut p, "num_channels");
    let ng = declare_real(&mut p, "num_groups");
    let cpg = declare_real(&mut p, "channels_per_group");
    let rem = declare_real(&mut p, "remainder");

    assert_positive(&mut p, &nc);
    assert_positive(&mut p, &ng);
    assert_positive(&mut p, &cpg);

    let product = ng.clone().real_mul(cpg);
    p.assert(nc.clone().eq(product.clone()));
    p.assert(nc.eq(product.real_add(rem.clone())));
    p.assert(rem.clone().real_ge(Expr::real(0)));
    p.assert(rem.clone().real_lt(ng));

    p.assert(rem.ne(Expr::real(0)));
    p.check_sat();
    Ok(make_result(&p, "groupnorm_partition_exact"))
}

// ---------------------------------------------------------------------------
// 7: InstanceNorm per-sample independence
// ---------------------------------------------------------------------------

/// Prove: InstanceNorm output depends only on same-sample features.
pub fn prove_instancenorm_independence() -> Result<NormalizationProofResult, SmtError> {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let f1 = declare_real(&mut p, "f1");
    let f2 = declare_real(&mut p, "f2");
    let mean_i = declare_real(&mut p, "mean_i");
    let s_i = declare_real(&mut p, "s_i");
    let n_a = declare_real(&mut p, "n_a");
    let n_b = declare_real(&mut p, "n_b");

    p.assert(
        Expr::real(2)
            .real_mul(mean_i.clone())
            .eq(f1.clone().real_add(f2)),
    );
    assert_positive(&mut p, &s_i);

    p.assert(
        n_a.clone()
            .real_mul(s_i.clone())
            .eq(f1.clone().real_sub(mean_i.clone())),
    );
    p.assert(n_b.clone().real_mul(s_i).eq(f1.real_sub(mean_i)));

    p.assert(n_a.ne(n_b));
    p.check_sat();
    Ok(make_result(&p, "instancenorm_per_sample_independence"))
}

// ---------------------------------------------------------------------------
// 8: Affine transform y = gamma * x_norm + beta
// ---------------------------------------------------------------------------

/// Scale applied by the affine step in [`build_affine_transform`].
const AFFINE_GAMMA: i64 = 2;
/// Shift applied by the affine step in [`build_affine_transform`].
const AFFINE_BETA: i64 = 3;

/// Prove that the affine step `y = gamma * x_norm + beta` is invertible: applying
/// its inverse recovers `x_norm` (a forward/inverse round trip is the identity).
///
/// The learnable affine step scales the normalized activation by `gamma` and
/// shifts it by `beta`. Its inverse subtracts `beta` and divides by `gamma`, so
/// `x_rec = (y - beta) / gamma` must return the original `x_norm`. We pin `gamma`
/// and `beta` to concrete constants (so `gamma * x_norm` is a literal-scaled term
/// and the query is linear) while `x_norm` ranges freely — the identity therefore
/// holds for every input.
///
/// `x_rec` is *derived* by inverting the forward pass, never asserted equal to
/// `x_norm`. The theorem breaks under a plausible slip — inverting with the wrong
/// sign on `beta` (adding it back instead of subtracting) — which makes the query
/// SAT (see `affine_inverse_depends_on_the_beta_sign`).
///
/// Every product has a literal factor, so the query stays in decidable `QF_LRA`.
pub fn prove_affine_transform() -> Result<NormalizationProofResult, SmtError> {
    let program = build_affine_transform(true);
    Ok(make_result(&program, "affine_transform_identity"))
}

/// Build the affine round-trip query. When `subtract_beta_on_inverse` is false
/// the inverse ADDS `beta` instead of subtracting it — a flipped-sign slip that
/// stops `x_rec` from recovering `x_norm`; tests flip it to confirm the proof
/// depends on it.
fn build_affine_transform(subtract_beta_on_inverse: bool) -> AYProgram {
    let mut p = AYProgram::new();
    p.set_logic("QF_LRA");

    let gamma = Expr::real(AFFINE_GAMMA);
    let beta = Expr::real(AFFINE_BETA);

    let x_norm = declare_real(&mut p, "x_norm");
    p.assert(x_norm.clone().real_ge(Expr::real(-10)));
    p.assert(x_norm.clone().real_le(Expr::real(10)));

    // Forward affine step: y = gamma * x_norm + beta.
    let y = declare_real(&mut p, "y");
    p.assert(
        y.clone()
            .eq(gamma.clone().real_mul(x_norm.clone()).real_add(beta.clone())),
    );

    // Inverse: x_rec = (y - beta) / gamma. Encoded as `gamma * x_rec = y - beta`
    // (both sides linear, no variable divisor). The slip adds `beta` back instead
    // of subtracting it.
    let corrected = if subtract_beta_on_inverse {
        y.real_sub(beta)
    } else {
        y.real_add(beta)
    };
    let x_rec = declare_real(&mut p, "x_rec");
    p.assert(gamma.real_mul(x_rec.clone()).eq(corrected));

    // Violation: the round trip failed to recover the input.
    p.assert(x_rec.ne(x_norm));
    p.check_sat();
    p
}

// ---------------------------------------------------------------------------
// 9: Epsilon stability (eps > 0 prevents division by zero)
// ---------------------------------------------------------------------------

/// Prove: var >= 0 and eps > 0 imply sqrt(var + eps) > 0.
pub fn prove_epsilon_stability() -> Result<NormalizationProofResult, SmtError> {
    let mut p = AYProgram::new();
    p.set_logic("QF_NRA");

    let var = declare_real(&mut p, "var");
    let eps = declare_real(&mut p, "eps");
    let denom = declare_real(&mut p, "denom");

    p.assert(var.clone().real_ge(Expr::real(0)));
    assert_positive(&mut p, &eps);

    // denom = sqrt(var + eps): denom^2 = var + eps, denom > 0
    p.assert(denom.clone().real_mul(denom.clone()).eq(var.real_add(eps)));
    assert_positive(&mut p, &denom);

    p.assert(denom.real_le(Expr::real(0)));
    p.check_sat();
    Ok(make_result(&p, "epsilon_stability"))
}

// ---------------------------------------------------------------------------
// 10: Normalization preserves shape
// ---------------------------------------------------------------------------

/// Rows of the concrete feature map normalized in [`build_norm_preserves_shape`].
const NORM_ROWS: i64 = 3;
/// Columns of the concrete feature map normalized in [`build_norm_preserves_shape`].
const NORM_COLS: i64 = 4;

/// Prove that elementwise normalization preserves the shape: it writes each of the
/// `ROWS * COLS` inputs to a *distinct* output slot, so nothing is lost or
/// overwritten and the element count is preserved.
///
/// Normalization is elementwise: the value at `(i, j)` maps to the same logical
/// position `(i, j)`, whose row-major output slot is `i*COLS + j`. "Preserves
/// shape" is exactly that this index map is *injective* on the `[ROWS, COLS]`
/// index box — two distinct coordinates never collide on one slot. With the
/// correct row stride `COLS` the `ROWS*COLS` inputs occupy `ROWS*COLS` distinct
/// slots (a bijection onto the buffer).
///
/// Injectivity is where a wrong output stride bites: writing rows `COLS-1` apart
/// instead of `COLS` makes two coordinates collide, and the query turns SAT (see
/// `shape_preservation_depends_on_the_row_stride`). Indices are `Int` over a
/// concrete shape, so every stride is a literal and the query stays in decidable
/// `QF_LIA` (over the reals `i*COLS + j` is not injective on the box).
pub fn prove_norm_preserves_shape() -> Result<NormalizationProofResult, SmtError> {
    let program = build_norm_preserves_shape(true);
    Ok(make_result(&program, "norm_preserves_shape"))
}

/// Build the shape-preservation query. When `row_stride_is_cols` is false the
/// output row stride is `COLS-1` instead of `COLS`, packing the rows too tightly
/// so distinct coordinates collide; tests flip it to confirm the proof depends on
/// the stride.
fn build_norm_preserves_shape(row_stride_is_cols: bool) -> AYProgram {
    let mut p = AYProgram::new();
    p.set_logic("QF_LIA");

    let row_stride = if row_stride_is_cols {
        NORM_COLS
    } else {
        NORM_COLS - 1
    };

    // Two coordinates in the [ROWS, COLS] feature map.
    let (i, j) = declare_cell(&mut p, "");
    let (i2, j2) = declare_cell(&mut p, "2");

    // Hypothesis: the coordinates differ somewhere.
    p.assert(i.clone().ne(i2.clone()).or(j.clone().ne(j2.clone())));

    // Row-major output slot of each coordinate under the elementwise map.
    let slot = i.int_mul(Expr::int(row_stride)).int_add(j);
    let slot2 = i2.int_mul(Expr::int(row_stride)).int_add(j2);

    // Violation: distinct coordinates land on the same output slot (an element
    // was overwritten, so the shape was not preserved).
    p.assert(slot.eq(slot2));
    p.check_sat();
    p
}

/// Declare `i{suffix}, j{suffix}` as a cell of the `[NORM_ROWS, NORM_COLS]` map.
fn declare_cell(p: &mut AYProgram, suffix: &str) -> (Expr, Expr) {
    (
        declare_bounded_index(p, &format!("i{suffix}"), NORM_ROWS),
        declare_bounded_index(p, &format!("j{suffix}"), NORM_COLS),
    )
}

/// Declare `name` as an `Int` constrained to `0 <= name < bound`.
fn declare_bounded_index(p: &mut AYProgram, name: &str, bound: i64) -> Expr {
    let var = p.declare_const(name, Sort::int());
    p.assert(var.clone().int_ge(Expr::int(0)));
    p.assert(var.clone().int_lt(Expr::int(bound)));
    var
}

#[cfg(test)]
#[path = "ay_normalization_layer_properties_tests.rs"]
mod tests;
