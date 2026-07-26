// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Network synthesis with ay: solve *for* the weights, not just check them.
//!
//! Verification asks "does this network satisfy this property?". Synthesis asks
//! "given this property, produce the network". Verification is the degenerate
//! case where the network is already fixed. This module implements the first
//! milestone of `designs/architecture-certified-pipeline.md` — synthesizing a
//! single affine layer from input/output examples — with ay as the backend.
//!
//! # The polarity inverts
//!
//! In nn's verification code a proof succeeds when ay answers UNSAT, which
//! `ay_bindings` spells [`ExecuteTypedResult::Verified`]: "the negated property
//! has no model". Synthesis wants the opposite. The weights are the *unknowns*,
//! the examples are the *constraints*, and a satisfying assignment IS the
//! network. So here:
//!
//! | ay verdict | `ay_bindings` spelling | synthesis meaning |
//! |---|---|---|
//! | SAT   | `Counterexample { model, .. }` | [`SynthesisResult::Found`] — `model` holds the weights |
//! | UNSAT | `Verified`                     | [`SynthesisResult::Infeasible`] — no such layer exists |
//!
//! Reading `Counterexample` as success is jarring; it is the same verdict wearing
//! a verification-shaped name.
//!
//! # Why this is linear
//!
//! For a fixed example `x`, the constraint `Σⱼ wᵢⱼ·xⱼ + bᵢ = yᵢ` multiplies each
//! unknown `wᵢⱼ` by a *constant* `xⱼ`. Products of unknowns never appear, so the
//! whole system is QF_LRA and ay decides it completely — no nonlinear search.
//! (Synthesizing two stacked layers would multiply unknown by unknown and land in
//! QF_NRA; that is the hard case the design doc flags.)
//!
//! # Numeric encoding
//!
//! Example inputs and targets are encoded as exact rationals at a resolution of
//! [`ENCODING_DENOM`] (`v ↦ round(v·10⁶)/10⁶`), so `0.1` becomes exactly `1/10`
//! rather than the f64 that is merely near it. The returned weights satisfy the
//! *encoded* system exactly. Give a nonzero
//! [`LinearSynthesisProblem::tolerance`] when the targets are measured rather
//! than exact.

use ay_bindings::execute_direct::{self, ExecuteTypedResult, ExecuteValueMap, ModelValue};
use ay_bindings::{AYProgram, Expr, Sort};
use num_traits::ToPrimitive;

use crate::ay_real_lit::RealLit;

/// Denominator used to encode `f64` example data as exact rationals.
pub const ENCODING_DENOM: i64 = 1_000_000;

/// Errors raised while building or solving a synthesis problem.
#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    /// An example's input or output length disagreed with the declared dims.
    #[error("example {index}: expected {expected} values, got {got}")]
    ExampleShape {
        /// Index of the offending example.
        index: usize,
        /// Length implied by `in_dim` / `out_dim`.
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// A supplied value was NaN or infinite.
    #[error("non-finite value in example data: {0}")]
    NonFinite(f64),
    /// The problem declared a zero dimension or supplied no examples.
    #[error("degenerate problem: {0}")]
    Degenerate(&'static str),
    /// ay could not be invoked.
    #[error("ay execution failed: {0}")]
    Execution(String),
    /// ay returned a model that lacked a variable, or held a non-real value.
    #[error("ay model is missing or mistyped variable `{0}`")]
    MalformedModel(String),
}

/// Synthesize an affine layer `y = W·x + b` satisfying every example.
#[derive(Debug, Clone)]
pub struct LinearSynthesisProblem {
    /// Input dimension (columns of `W`).
    pub in_dim: usize,
    /// Output dimension (rows of `W`, length of `b`).
    pub out_dim: usize,
    /// `(input, target_output)` pairs the layer must reproduce.
    pub examples: Vec<(Vec<f64>, Vec<f64>)>,
    /// Every weight and bias is constrained to `[-bound, bound]`.
    ///
    /// This keeps the search box finite. A too-small bound can make a
    /// genuinely-satisfiable problem report [`SynthesisResult::Infeasible`], so
    /// treat `Infeasible` as "no layer within this box", not "no layer at all".
    pub weight_abs_bound: i64,
    /// Per-output slack: `|Σ wᵢⱼxⱼ + bᵢ − yᵢ| ≤ tolerance`. Zero means exact.
    pub tolerance: f64,
}

/// A synthesized affine layer.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesizedLinear {
    /// `weights[i][j]` is `wᵢⱼ`.
    pub weights: Vec<Vec<f64>>,
    /// `bias[i]` is `bᵢ`.
    pub bias: Vec<f64>,
}

impl SynthesizedLinear {
    /// Evaluate `W·x + b`.
    #[must_use]
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(&self.bias)
            .map(|(row, b)| row.iter().zip(x).map(|(w, xj)| w * xj).sum::<f64>() + b)
            .collect()
    }
}

/// Outcome of a synthesis query.
#[derive(Debug, Clone, PartialEq)]
pub enum SynthesisResult {
    /// ay returned SAT: the assignment is a layer meeting every constraint.
    Found(SynthesizedLinear),
    /// ay returned UNSAT: no layer inside the weight box meets the constraints.
    Infeasible,
    /// ay returned unknown, or the query needed a fallback path.
    Unknown(String),
}

fn encode_f64(v: f64) -> Result<Expr, SynthesisError> {
    if !v.is_finite() {
        return Err(SynthesisError::NonFinite(v));
    }
    let scaled = (v * ENCODING_DENOM as f64).round();
    let numer = scaled
        .to_i64()
        .ok_or(SynthesisError::NonFinite(v))?;
    Ok(Expr::real_ratio(numer, ENCODING_DENOM))
}

fn weight_name(i: usize, j: usize) -> String {
    format!("w_{i}_{j}")
}

fn bias_name(i: usize) -> String {
    format!("b_{i}")
}

/// Build the SMT-LIB2 text of the synthesis query without solving it.
///
/// Useful for debugging or handing the problem to an external solver.
///
/// # Errors
///
/// See [`SynthesisError`] for shape and finiteness failures.
pub fn synthesis_query_smt2(problem: &LinearSynthesisProblem) -> Result<String, SynthesisError> {
    Ok(build_program(problem)?.to_string())
}

fn build_program(problem: &LinearSynthesisProblem) -> Result<AYProgram, SynthesisError> {
    if problem.in_dim == 0 || problem.out_dim == 0 {
        return Err(SynthesisError::Degenerate("in_dim and out_dim must be > 0"));
    }
    if problem.examples.is_empty() {
        return Err(SynthesisError::Degenerate("at least one example required"));
    }
    if problem.weight_abs_bound <= 0 {
        return Err(SynthesisError::Degenerate("weight_abs_bound must be > 0"));
    }
    for (k, (x, y)) in problem.examples.iter().enumerate() {
        if x.len() != problem.in_dim {
            return Err(SynthesisError::ExampleShape {
                index: k,
                expected: problem.in_dim,
                got: x.len(),
            });
        }
        if y.len() != problem.out_dim {
            return Err(SynthesisError::ExampleShape {
                index: k,
                expected: problem.out_dim,
                got: y.len(),
            });
        }
    }

    let mut program = AYProgram::new();
    // Unknowns are multiplied only by example constants, never by each other.
    program.set_logic("QF_LRA");

    let lo = Expr::real(-problem.weight_abs_bound);
    let hi = Expr::real(problem.weight_abs_bound);

    let mut weights = Vec::with_capacity(problem.out_dim);
    let mut biases = Vec::with_capacity(problem.out_dim);
    for i in 0..problem.out_dim {
        let mut row = Vec::with_capacity(problem.in_dim);
        for j in 0..problem.in_dim {
            let w = program.declare_const(weight_name(i, j), Sort::real());
            program.assert(w.clone().real_ge(lo.clone()));
            program.assert(w.clone().real_le(hi.clone()));
            row.push(w);
        }
        let b = program.declare_const(bias_name(i), Sort::real());
        program.assert(b.clone().real_ge(lo.clone()));
        program.assert(b.clone().real_le(hi.clone()));
        weights.push(row);
        biases.push(b);
    }

    let tol = encode_f64(problem.tolerance.abs())?;
    let exact = problem.tolerance == 0.0;

    for (x, y) in &problem.examples {
        for i in 0..problem.out_dim {
            let mut acc = biases[i].clone();
            for j in 0..problem.in_dim {
                acc = acc.real_add(weights[i][j].clone().real_mul(encode_f64(x[j])?));
            }
            let target = encode_f64(y[i])?;
            if exact {
                program.assert(acc.eq(target));
            } else {
                program.assert(
                    acc.clone()
                        .real_ge(target.clone().real_sub(tol.clone()))
                        .and(acc.real_le(target.real_add(tol.clone()))),
                );
            }
        }
    }

    program.check_sat();
    Ok(program)
}

/// Synthesize an affine layer that reproduces every example.
///
/// Encodes the examples as linear constraints over the unknown weights and asks
/// ay for a satisfying assignment. See the module docs for the inverted verdict
/// polarity and the numeric encoding.
///
/// # Errors
///
/// See [`SynthesisError`]. A well-formed problem that simply has no solution
/// inside the weight box returns `Ok(SynthesisResult::Infeasible)`, not an error.
pub fn synthesize_linear(
    problem: &LinearSynthesisProblem,
) -> Result<SynthesisResult, SynthesisError> {
    let program = build_program(problem)?;

    match execute_direct::execute_typed(&program) {
        // SAT. The "counterexample" to "no such layer exists" is the layer.
        Ok(ExecuteTypedResult::Counterexample(cx)) => {
            let mut weights = Vec::with_capacity(problem.out_dim);
            let mut bias = Vec::with_capacity(problem.out_dim);
            for i in 0..problem.out_dim {
                let mut row = Vec::with_capacity(problem.in_dim);
                for j in 0..problem.in_dim {
                    row.push(read_real(&cx.model, &weight_name(i, j))?);
                }
                weights.push(row);
                bias.push(read_real(&cx.model, &bias_name(i))?);
            }
            Ok(SynthesisResult::Found(SynthesizedLinear { weights, bias }))
        }
        // UNSAT: the constraints admit no assignment.
        Ok(ExecuteTypedResult::Verified) => Ok(SynthesisResult::Infeasible),
        Ok(ExecuteTypedResult::Unknown(reason)) => Ok(SynthesisResult::Unknown(reason)),
        Ok(ExecuteTypedResult::NeedsFallback(reason)) => Ok(SynthesisResult::Unknown(reason)),
        // `ExecuteTypedResult` is `#[non_exhaustive]`. A verdict we do not
        // recognize must never read as `Found` or `Infeasible`.
        Ok(other) => Ok(SynthesisResult::Unknown(format!(
            "unrecognized ay verdict: {other:?}"
        ))),
        Err(e) => Err(SynthesisError::Execution(e.to_string())),
    }
}

fn read_real(model: &ExecuteValueMap<ModelValue>, name: &str) -> Result<f64, SynthesisError> {
    let value = model
        .get(name)
        .ok_or_else(|| SynthesisError::MalformedModel(name.to_string()))?;
    let rational = value
        .try_real()
        .map_err(|_| SynthesisError::MalformedModel(name.to_string()))?;
    rational
        .to_f64()
        .ok_or_else(|| SynthesisError::MalformedModel(name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem(examples: Vec<(Vec<f64>, Vec<f64>)>, in_dim: usize, out_dim: usize) -> LinearSynthesisProblem {
        LinearSynthesisProblem {
            in_dim,
            out_dim,
            examples,
            weight_abs_bound: 100,
            tolerance: 0.0,
        }
    }

    /// The sanity milestone from the design doc: recover `y = 2x + 3`.
    #[test]
    fn recovers_a_one_dimensional_affine_map() {
        let p = problem(vec![(vec![0.0], vec![3.0]), (vec![1.0], vec![5.0])], 1, 1);
        let SynthesisResult::Found(layer) = synthesize_linear(&p).expect("solve") else {
            panic!("y = 2x + 3 is realizable");
        };
        assert!((layer.weights[0][0] - 2.0).abs() < 1e-9, "{layer:?}");
        assert!((layer.bias[0] - 3.0).abs() < 1e-9, "{layer:?}");
    }

    /// Two examples pin a 2->1 map exactly; a third redundant one must not break it.
    #[test]
    fn synthesized_layer_reproduces_every_example() {
        let examples = vec![
            (vec![1.0, 0.0], vec![1.5]),
            (vec![0.0, 1.0], vec![-0.5]),
            (vec![1.0, 1.0], vec![1.0]),
        ];
        let p = problem(examples.clone(), 2, 1);
        let SynthesisResult::Found(layer) = synthesize_linear(&p).expect("solve") else {
            panic!("consistent system must be satisfiable");
        };
        // Soundness: the returned weights actually satisfy the constraints.
        for (x, y) in &examples {
            let got = layer.forward(x);
            assert!(
                (got[0] - y[0]).abs() < 1e-6,
                "layer {layer:?} maps {x:?} to {got:?}, expected {y:?}",
            );
        }
    }

    /// Contradictory examples have no affine solution: ay must say UNSAT.
    /// Without this the module could not distinguish "solved" from "always SAT".
    #[test]
    fn contradictory_examples_are_infeasible() {
        let p = problem(vec![(vec![0.0], vec![0.0]), (vec![0.0], vec![1.0])], 1, 1);
        assert_eq!(
            synthesize_linear(&p).expect("solve"),
            SynthesisResult::Infeasible,
            "x=0 cannot map to both 0 and 1",
        );
    }

    /// `y = 50x` is realizable, but not inside a weight box of |w| <= 10.
    /// `Infeasible` means "not in this box", which the doc comment promises.
    #[test]
    fn weight_box_can_make_a_realizable_map_infeasible() {
        let examples = vec![(vec![1.0], vec![50.0]), (vec![2.0], vec![100.0])];
        let mut p = problem(examples, 1, 1);
        p.weight_abs_bound = 10;
        assert_eq!(synthesize_linear(&p).expect("solve"), SynthesisResult::Infeasible);

        p.weight_abs_bound = 100;
        let SynthesisResult::Found(layer) = synthesize_linear(&p).expect("solve") else {
            panic!("realizable once the box admits w = 50");
        };
        assert!((layer.weights[0][0] - 50.0).abs() < 1e-9, "{layer:?}");
    }

    /// Tolerance absorbs targets the exact system would reject.
    #[test]
    fn tolerance_admits_slightly_inconsistent_targets() {
        let examples = vec![
            (vec![0.0], vec![0.0]),
            (vec![1.0], vec![1.0]),
            (vec![2.0], vec![2.01]), // off the line by 0.01
        ];
        let mut p = problem(examples, 1, 1);
        assert_eq!(synthesize_linear(&p).expect("solve"), SynthesisResult::Infeasible);

        p.tolerance = 0.02;
        assert!(matches!(
            synthesize_linear(&p).expect("solve"),
            SynthesisResult::Found(_),
        ));
    }

    #[test]
    fn rejects_malformed_problems() {
        let mut p = problem(vec![(vec![1.0, 2.0], vec![1.0])], 1, 1);
        assert!(matches!(
            synthesize_linear(&p),
            Err(SynthesisError::ExampleShape { index: 0, expected: 1, got: 2 }),
        ));

        p = problem(vec![], 1, 1);
        assert!(matches!(synthesize_linear(&p), Err(SynthesisError::Degenerate(_))));
    }

    #[test]
    fn query_renders_as_smt2() {
        let p = problem(vec![(vec![1.0], vec![2.0])], 1, 1);
        let smt2 = synthesis_query_smt2(&p).expect("renders");
        assert!(smt2.contains("check-sat"));
        assert!(smt2.contains("w_0_0"));
        assert!(smt2.contains("b_0"));
    }
}
