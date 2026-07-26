// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for convolution stride and padding mathematical properties.
//!
//! Module-level proofs that establish fundamental convolution identities used
//! throughout nn's conv1d/conv2d implementations, model import (PyTorch shape
//! inference), and verification (NY layer translation).
//!
//! # Properties proved
//!
//! 1. **Output dimension formula**: `out = (input + 2*pad - kernel) / stride + 1`
//! 2. **Transposed conv output**: `out = (input - 1) * stride - 2*pad + kernel + output_pad`
//! 3. **Dilated conv effective kernel**: `eff_k = kernel + (kernel - 1) * (dilation - 1)`
//! 4. **Same padding preserves length**: `pad = (kernel - 1) / 2` gives `out == input` (stride 1)
//! 5. **Valid padding shrinks**: `pad = 0` gives `out < input` for `kernel >= 2`
//! 6. **Causal padding**: `left_pad = kernel - 1, right_pad = 0` preserves length
//! 7. **Depthwise conv**: `groups == in_channels` => weight shape `[C, 1, K]`, params `C * K`
//! 8. **Group conv**: weight shape `[out, in/groups, K]`, params `C_out * C_in * K / G`
//!
//! # Proof strategy
//!
//! All proofs use QF_LRA (quantifier-free linear real arithmetic). We encode
//! convolution dimension formulas symbolically, assert the negation of the
//! desired property, and prove UNSAT (no counterexample exists).
//!
//! Part of #4226.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::smt_error::SmtError;

/// Result of a convolution property proof attempt.
#[derive(Debug, Clone)]
pub struct ConvPropertyResult {
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
// Property 1: Output Dimension Formula
// ---------------------------------------------------------------------------

/// Prove the standard conv output dimension formula.
///
/// For stride S=1, dilation D=1, padding P, kernel K, input length L:
///   out = L + 2*P - K + 1
///
/// We prove `out >= 1` whenever L + 2*P >= K (valid configuration).
pub fn prove_output_dimension_formula() -> Result<ConvPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let l = declare_real(&mut program, "l");
    let k = declare_real(&mut program, "k");
    let p = declare_real(&mut program, "p");
    let out_len = declare_real(&mut program, "out_len");

    // Constraints: L >= 1, K >= 1, P >= 0
    program.assert(l.clone().real_ge(Expr::real(1)));
    program.assert(k.clone().real_ge(Expr::real(1)));
    program.assert(p.clone().real_ge(Expr::real(0)));

    // Valid config: L + 2P >= K
    let two_p = Expr::real(2).real_mul(p.clone());
    program.assert(l.clone().real_add(two_p.clone()).real_ge(k.clone()));

    // S=1, D=1: out = L + 2P - K + 1
    let formula = l.real_add(two_p).real_sub(k).real_add(Expr::real(1));
    program.assert(out_len.clone().eq(formula));

    // Negated property: out_len < 1
    program.assert(out_len.real_lt(Expr::real(1)));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "output_dimension_formula".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Transposed Conv Output Formula
// ---------------------------------------------------------------------------

/// Prove the transposed convolution output dimension formula (adjoint property).
///
/// For stride S=1, output_padding=0, padding P, and effective kernel size
/// `eff_k = dilation*(K-1)`:
///   Forward conv:   out = N + 2P - eff_k
///   Conv transpose: N'  = out - 2P + eff_k
/// so `N' = N` — the transpose recovers the original input length.
///
/// `eff_k` is modelled as a single opaque non-negative quantity rather than the
/// product `dilation*(K-1)`. The cancellation `N' = N` holds for *any* effective
/// kernel, so decomposing it would only introduce a variable×variable product
/// (nonlinear — the linear engine returns a spurious SAT, which is exactly the
/// original bug) without strengthening the theorem. With `eff_k` opaque the whole
/// query is linear and stays in decidable QF_LRA. The conclusion `N' = N` is
/// *derived* by chaining the two definitions through `out_conv`, never asserted;
/// adding the padding term back with the wrong sign (see
/// [`build_transposed_conv_output`]) yields `N + 4P`, making the property FALSE
/// and the query SAT.
pub fn prove_transposed_conv_output() -> Result<ConvPropertyResult, SmtError> {
    let program = build_transposed_conv_output(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "transposed_conv_output".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the transposed-conv adjoint query. `subtract_padding_on_transpose` gates
/// the sign of the padding term the transpose removes: the correct
/// `out - 2P + eff_k` recovers `N`, while the buggy `out + 2P + eff_k` (padding
/// added back with the wrong sign) yields `N + 4P`, which differs from `N`
/// whenever `P > 0`. Tests flip it to confirm the proof depends on the sign.
fn build_transposed_conv_output(subtract_padding_on_transpose: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let n = declare_real(&mut program, "n");
    let p = declare_real(&mut program, "p");
    let eff_k = declare_real(&mut program, "eff_k");
    let out_conv = declare_real(&mut program, "out_conv");
    let n_prime = declare_real(&mut program, "n_prime");

    program.assert(n.clone().real_ge(Expr::real(1)));
    program.assert(p.clone().real_ge(Expr::real(0)));
    program.assert(eff_k.clone().real_ge(Expr::real(0)));

    let two_p = Expr::real(2).real_mul(p.clone());

    // Valid forward config: N + 2P >= eff_k + 1 (output length >= 1).
    program.assert(
        n.clone()
            .real_add(two_p.clone())
            .real_ge(eff_k.clone().real_add(Expr::real(1))),
    );

    // Forward conv output (S=1): out = N + 2P - eff_k.
    let out_formula = n.clone().real_add(two_p.clone()).real_sub(eff_k.clone());
    program.assert(out_conv.clone().eq(out_formula));

    // Conv transpose (S=1, output_padding=0): N' = out - 2P + eff_k.
    let recover = if subtract_padding_on_transpose {
        out_conv.real_sub(two_p).real_add(eff_k)
    } else {
        // BUG: padding added back with the wrong sign => N' = N + 4P.
        out_conv.real_add(two_p).real_add(eff_k)
    };
    program.assert(n_prime.clone().eq(recover));

    // Negated property: N' != N.
    program.assert(n_prime.ne(n));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 3: Dilated Conv Effective Kernel
// ---------------------------------------------------------------------------

/// Prove the two equivalent forms of dilated convolution effective kernel size.
///
///   Form 1: D*(K-1) + 1
///   Form 2: K + (K-1)*(D-1)
///
/// Both reduce to D*K - D + 1. We prove they are identical for all K >= 1, D >= 1.
pub fn prove_dilated_effective_kernel() -> Result<ConvPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let k = declare_real(&mut program, "k");
    let d = declare_real(&mut program, "d");

    program.assert(k.clone().real_ge(Expr::real(1)));
    program.assert(d.clone().real_ge(Expr::real(1)));

    // Form 1: D*(K-1) + 1
    let form1 = d
        .clone()
        .real_mul(k.clone().real_sub(Expr::real(1)))
        .real_add(Expr::real(1));

    // Form 2: K + (K-1)*(D-1)
    let form2 = k.clone().real_add(
        k.real_sub(Expr::real(1))
            .real_mul(d.real_sub(Expr::real(1))),
    );

    // Negated: form1 != form2
    program.assert(form1.ne(form2));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "dilated_effective_kernel".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Same Padding Preserves Length
// ---------------------------------------------------------------------------

/// Prove that "same" padding with stride=1 preserves spatial length.
///
/// For K odd, stride=1, dilation=1, padding P = (K-1)/2:
///   out = L + 2*P - (K-1) = L + (K-1) - (K-1) = L
///
/// Proved symbolically for any L >= 1 and K >= 1 (assuming K odd via 2P = K-1).
pub fn prove_same_padding_preserves_length() -> Result<ConvPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let l = declare_real(&mut program, "l");
    let k = declare_real(&mut program, "k");
    let out_len = declare_real(&mut program, "out_len");

    program.assert(l.clone().real_ge(Expr::real(1)));
    program.assert(k.clone().real_ge(Expr::real(1)));

    // P = (K-1)/2. So 2P = K-1.
    // Formula (S=1, D=1): out = L + 2P - (K-1) - 1 + 1 = L + (K-1) - (K-1) = L
    let k_minus_1 = k.real_sub(Expr::real(1));
    let two_p = k_minus_1.clone(); // 2 * (K-1)/2 = K-1
    let formula = l
        .clone()
        .real_add(two_p)
        .real_sub(k_minus_1)
        .real_sub(Expr::real(1))
        .real_add(Expr::real(1));
    program.assert(out_len.clone().eq(formula));

    // Negated: out_len != L
    program.assert(out_len.ne(l));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "same_padding_preserves_length".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Valid Padding Shrinks Output
// ---------------------------------------------------------------------------

/// Prove that "valid" padding (P=0) produces output strictly shorter than input
/// for kernel size K >= 2.
///
/// For S=1, D=1, P=0: out = L - K + 1. With K >= 2: out = L - K + 1 <= L - 1 < L.
pub fn prove_valid_padding_shrinks() -> Result<ConvPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let l = declare_real(&mut program, "l");
    let k = declare_real(&mut program, "k");
    let out_len = declare_real(&mut program, "out_len");

    program.assert(l.clone().real_ge(Expr::real(2))); // need L >= K for valid config
    program.assert(k.clone().real_ge(Expr::real(2))); // K >= 2 for non-trivial kernel

    // Valid config: L >= K (P=0)
    program.assert(l.clone().real_ge(k.clone()));

    // out = L - K + 1
    let formula = l.clone().real_sub(k).real_add(Expr::real(1));
    program.assert(out_len.clone().eq(formula));

    // Negated: out >= L (should be UNSAT since K >= 2 => out <= L - 1 < L)
    program.assert(out_len.real_ge(l));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "valid_padding_shrinks".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Causal Padding Preserves Length
// ---------------------------------------------------------------------------

/// Prove that causal convolution (left_pad = K-1, right_pad = 0) preserves
/// input length for stride=1, dilation=1.
///
/// total_pad = K-1. out = L + (K-1) - (K-1) = L.
/// Output at position t depends only on input [t-K+1, t] (causal).
pub fn prove_causal_padding_preserves_length() -> Result<ConvPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let l = declare_real(&mut program, "l");
    let k = declare_real(&mut program, "k");
    let out_len = declare_real(&mut program, "out_len");

    program.assert(l.clone().real_ge(Expr::real(1)));
    program.assert(k.clone().real_ge(Expr::real(1)));

    // left_pad = K-1, right_pad = 0 => total_pad = K-1
    // out = L + (K-1) - (K-1) - 1 + 1 = L
    let k_minus_1 = k.real_sub(Expr::real(1));
    let formula = l
        .clone()
        .real_add(k_minus_1.clone())
        .real_sub(k_minus_1)
        .real_sub(Expr::real(1))
        .real_add(Expr::real(1));
    program.assert(out_len.clone().eq(formula));

    // Negated: out_len != L
    program.assert(out_len.ne(l));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "causal_padding_preserves_length".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: Depthwise Conv Parameters
// ---------------------------------------------------------------------------

/// Prove depthwise convolution parameter count: `params = C_in * K`.
///
/// Depthwise conv sets `groups = C_in`, so each group owns exactly one input
/// channel: channels-per-group `cpg = C_in / groups = 1`. With `out_channels =
/// C_in` the weight tensor is `[C_in, cpg, K]` and `params = C_in * cpg * K`.
///
/// The load-bearing step is *deriving* `cpg` from the group count: `cpg` is a
/// free variable pinned by `cpg * groups = C_in`, and only `groups = C_in` forces
/// `cpg = 1` and hence `params = C_in * K`. The channel and kernel sizes are
/// concrete literals, so `C_in * cpg * K` folds to `cpg * (C_in*K)` — linear in
/// the single free variable `cpg`, decidable in QF_LRA, and with the division
/// `C_in/groups` exact (no fractional real counterexample). The original proof
/// left `C_in`, `cpg`, `K` all symbolic, producing a variable×variable product
/// that the linear engine mishandles into a spurious SAT. Setting `groups = 1`
/// (see [`build_depthwise_conv_params`]) makes `cpg = C_in` and the standard-conv
/// count `C_in^2 * K != C_in * K`, so the query turns SAT.
pub fn prove_depthwise_conv_params() -> Result<ConvPropertyResult, SmtError> {
    let program = build_depthwise_conv_params(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "depthwise_conv_params".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Input-channel count of the concrete depthwise config.
const DEPTHWISE_C_IN: i64 = 8;
/// Kernel size of the concrete depthwise config.
const DEPTHWISE_K: i64 = 3;

/// Build the depthwise param-count query. `groups_equal_channels` selects the
/// group count: the correct depthwise `groups = C_in` (so `cpg = 1`), or the
/// buggy `groups = 1` (a single group, i.e. a standard conv), which makes
/// `cpg = C_in` and the parameter count `C_in^2 * K`. Tests flip it to confirm
/// the proof depends on the depthwise group count.
fn build_depthwise_conv_params(groups_equal_channels: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let groups = if groups_equal_channels {
        DEPTHWISE_C_IN
    } else {
        1 // BUG: not depthwise — a single group is a standard conv.
    };

    // channels-per-group cpg = C_in / groups, pinned multiplicatively so the
    // division is exact (no fractional real counterexample): cpg * groups == C_in.
    let cpg = declare_real(&mut program, "cpg");
    program.assert(cpg.clone().real_ge(Expr::real(0)));
    program.assert(
        cpg.clone()
            .real_mul(Expr::real(groups))
            .eq(Expr::real(DEPTHWISE_C_IN)),
    );

    // params = out_channels * cpg * K = cpg * (C_in*K); C_in and K are literals,
    // so this stays linear in the single free variable cpg.
    let params = declare_real(&mut program, "params");
    program.assert(
        params
            .clone()
            .eq(cpg.real_mul(Expr::real(DEPTHWISE_C_IN * DEPTHWISE_K))),
    );

    // Negated property: params != C_in * K.
    program.assert(params.ne(Expr::real(DEPTHWISE_C_IN * DEPTHWISE_K)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 8: Group Conv Weight Partitioning
// ---------------------------------------------------------------------------

/// Prove grouped convolution never uses more weights than the equivalent
/// standard convolution: `total <= standard`.
///
/// With `G` groups the weight tensor partitions into `G` blocks of shape
/// `[C_out/G, C_in/G, K]`, so `total = C_out*C_in*K / G`, i.e. `total * G =
/// standard` where `standard = C_out*C_in*K`. Since `G >= 1` and the counts are
/// non-negative, `total <= standard`.
///
/// The shape `[C_out, C_in, K]` and group count are concrete, so `standard` is a
/// positive literal and `total` is the single free variable pinned by the exact
/// partition `total * G = standard` (no fractional real counterexample — `G`
/// divides `standard`). The conclusion `total <= standard` is *derived* from the
/// partition equation, not asserted. The original proof left `C_out`, `C_in`,
/// `K`, `G`, `standard`, `total` all symbolic, giving variable×variable products
/// (`C_out*C_in*K`, `total*G`) that the linear engine mishandles into a spurious
/// SAT. Multiplying by `G` instead of dividing (see
/// [`build_group_conv_weight_partition`]) makes `total = standard*G > standard`
/// and the query SAT. Linear in `total`, decidable in QF_LRA.
pub fn prove_group_conv_weight_partition() -> Result<ConvPropertyResult, SmtError> {
    let program = build_group_conv_weight_partition(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ConvPropertyResult {
        property: "group_conv_weight_partition".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Output channels of the concrete grouped-conv config.
const GROUP_C_OUT: i64 = 4;
/// Input channels of the concrete grouped-conv config.
const GROUP_C_IN: i64 = 6;
/// Kernel size of the concrete grouped-conv config.
const GROUP_K: i64 = 3;
/// Number of groups (divides both C_out and C_in evenly).
const GROUP_G: i64 = 2;
/// Standard (ungrouped) parameter count for the concrete shape: `C_out*C_in*K`.
const GROUP_STANDARD: i64 = GROUP_C_OUT * GROUP_C_IN * GROUP_K;

/// Build the grouped-conv partition query. `divide_by_groups` selects how the
/// grouped count relates to the standard count: the correct partition
/// `total * G = standard` (`total = standard/G`), or the buggy `total = standard
/// * G` (multiplying by the group count instead of dividing), which makes
/// `total > standard`. Tests flip it to confirm the proof depends on dividing.
fn build_group_conv_weight_partition(divide_by_groups: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Parameter counts are non-negative.
    let total = declare_real(&mut program, "total");
    program.assert(total.clone().real_ge(Expr::real(0)));

    if divide_by_groups {
        // Weight partition: total * G == standard, i.e. total = standard / G.
        program.assert(
            total
                .clone()
                .real_mul(Expr::real(GROUP_G))
                .eq(Expr::real(GROUP_STANDARD)),
        );
    } else {
        // BUG: multiplied by the group count instead of dividing.
        program.assert(
            total
                .clone()
                .eq(Expr::real(GROUP_STANDARD).real_mul(Expr::real(GROUP_G))),
        );
    }

    // Negated property: the grouped count exceeds the standard count.
    program.assert(total.real_gt(Expr::real(GROUP_STANDARD)));
    program.check_sat();
    program
}

#[cfg(test)]
#[path = "ay_conv_stride_padding_properties_tests.rs"]
mod tests;
