// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pipeline-level two-graph equivalence — the
//! [`ny_equivalence`](crate::ny_equivalence) adapter wired into the pipeline.
//!
//! # What is wired
//!
//! [`verify_tensor_kernels_equivalent`] proves `||f(x) - g(x)|| < eps` over an
//! input region for **any** two `TensorKernelDef`s that translate to graphs with
//! a shared input and matching output dimension. It routes through ny's curated
//! facade (difference network + CROWN-family propagation) instead of NN
//! rebuilding that sequence by hand.
//!
//! # What stays as it was
//!
//! [`crate::resblock_equivalence`] keeps its specialized FusedResBlock peephole
//! proofs (`verify_style_absorption_equivalence`,
//! `verify_batched_style_equivalence`). They pin the two peephole passes with
//! their own fixtures and are *not* deleted or re-pointed; this module is the
//! general case they are an instance of. The integration test
//! `facade_equivalence_matches_the_hand_rolled_resblock_proof` holds the routed
//! path to producing a worst-case difference bound no looser than the
//! hand-rolled proof's on a shared fixture.
//!
//! Nothing calls this implicitly: it is a distinct entry point a caller reaches
//! for deliberately.

use nn_dsl::tensor_ir::TensorKernelDef;
use ny_api::verify::{PropagationConfig, PropagationMethod};
use ny_api::{Bound, BoundedTensor};

use crate::error::VerifyError;
use crate::graph_tensor::{tensor_kernel_to_graph, TensorParamBinding};
// The structured verdict type is read from ny_api directly, as
// `ny_equivalence`'s own module docs direct; the NN-facing wrappers and
// reporting helpers still come through the adapter.
use crate::ny_equivalence::{
    equivalence_summary, is_equivalent, verify_networks_equivalent_with_config,
};
use ny_api::equivalence::EquivalenceResult;

/// Outcome of a pipeline-level equivalence proof between two tensor kernels.
///
/// Carries both ny's structured verdict and the flat numbers NN's existing
/// difference-network proofs report, so this is a drop-in replacement for a
/// hand-rolled `build_difference_network` + `propagate_*` + `bounds_min_max`
/// sequence.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EquivalenceOutcome {
    /// ny's structured verdict (`Equivalent` / `NotEquivalent` / `Unknown`).
    pub result: EquivalenceResult,
    /// Whether the two graphs were *proven* equivalent within `epsilon`.
    ///
    /// `false` is not a counterexample — it means the sound bounds were too
    /// loose (or the run was inconclusive) to prove agreement.
    pub within_epsilon: bool,
    /// Worst-case output difference the verifier proved,
    /// `max(|lower|, |upper|)` over the difference network's outputs. Same
    /// quantity as `PeepholeEquivalenceResult::max_abs_diff`.
    pub max_abs_diff: f32,
    /// Epsilon the proof was run against.
    pub epsilon: f32,
    /// One-line NN-style summary from
    /// [`ny_equivalence::equivalence_summary`](crate::ny_equivalence::equivalence_summary).
    pub summary: String,
}

/// Narrow a proven non-negative `f64` upper bound to `f32` **without ever
/// shrinking it**.
///
/// `v as f32` rounds to nearest, which can land *below* `v` — reporting a
/// tighter difference than the verifier actually proved. A bound must enclose,
/// so when the round-trip comes back smaller we step one ULP away from zero.
///
/// Inputs here are always `max(|lower|, |upper|)` over the difference network's
/// outputs, hence non-negative (or `+inf`); `+inf` round-trips exactly and NaN
/// is propagated rather than silently turned into a number.
fn narrow_bound_up(v: f64) -> f32 {
    let narrowed = v as f32;
    if f64::from(narrowed) < v {
        // `v` is finite and positive here (a smaller-than-`v` round-trip rules
        // out inf and NaN), so the next representable f32 away from zero is
        // `bits + 1`.
        f32::from_bits(narrowed.to_bits() + 1)
    } else {
        narrowed
    }
}

/// Default propagation method for pipeline-level equivalence proofs.
///
/// Plain CROWN, matching what [`crate::resblock_equivalence`] propagates
/// through its hand-built difference network. CROWN carries each output's
/// linear dependence on the *shared* input, so for affine-equivalent graphs the
/// correlated terms cancel and the proven difference is exact up to `f32`
/// round-off. IBP cannot prove equivalence of anything but constants —
/// `[a, b] - [a, b] = [a - b, b - a]` never cancels — so it is not offered as
/// the default here.
#[must_use]
pub fn default_equivalence_method() -> PropagationMethod {
    PropagationMethod::Crown
}

/// Prove that two tensor kernels are equivalent within `epsilon` over
/// `input_bounds`, routed through the [`ny_equivalence`](crate::ny_equivalence)
/// facade.
///
/// This is the **pipeline-level generalization** of the difference-network
/// proofs NN hand-rolls in [`crate::resblock_equivalence`]: any two
/// `TensorKernelDef`s that translate to graphs with the same input arity and
/// output dimension can be compared, not just the two FusedResBlock peephole
/// pairs. The specialized `resblock_equivalence` proofs are deliberately left in
/// place — they pin the peephole passes with their own fixtures, and this
/// routing is additive.
///
/// `f_bindings` and `g_bindings` are separate because the interesting cases —
/// an optimized plan against its reference lowering, a re-quantized weight set
/// against the original — differ in their *constants*. What must line up is the
/// **variable** side: both graphs consume the same shared input over
/// `input_bounds`, which is the invariant the difference network is built on. A
/// mismatch in output dimension is caught and refused by the facade.
///
/// `method` selects the propagation rung; use [`default_equivalence_method`]
/// (plain CROWN) to match `resblock_equivalence`, or
/// [`PropagationMethod::AlphaCrown`] for ny's own default, which is never
/// looser.
///
/// # Errors
///
/// Returns [`VerifyError::Ny`] if either kernel fails to translate, if the two
/// graphs' output dimensions differ, if `epsilon` is not positive, or if bound
/// propagation fails.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_tensor_kernels_equivalent(
    f: &TensorKernelDef,
    f_bindings: &[TensorParamBinding],
    g: &TensorKernelDef,
    g_bindings: &[TensorParamBinding],
    input_bounds: &BoundedTensor,
    epsilon: f32,
    method: PropagationMethod,
) -> Result<EquivalenceOutcome, VerifyError> {
    let f_graph = tensor_kernel_to_graph(f, f_bindings)?;
    let g_graph = tensor_kernel_to_graph(g, g_bindings)?;

    // The facade takes a flat per-element input region; a BoundedTensor's
    // element order is the same row-major order ny rebuilds the tensor from.
    let (lower, upper) = input_bounds.lower_upper();
    let region: Vec<Bound> = lower
        .iter()
        .zip(upper.iter())
        .map(|(&lo, &hi)| Bound::try_new(lo, hi))
        .collect::<ny_api::Result<Vec<_>>>()?;

    let cfg = PropagationConfig {
        method,
        ..PropagationConfig::default()
    };
    let result =
        verify_networks_equivalent_with_config(&f_graph, &g_graph, &region, epsilon, &cfg)?;

    // ny reports the worst-case difference in f64 for all three verdicts;
    // narrow it back to the f32 scale NN's other equivalence results use.
    let max_abs_diff = narrow_bound_up(match &result {
        EquivalenceResult::Equivalent { bound } => *bound,
        EquivalenceResult::NotEquivalent { worst_case_bound } => *worst_case_bound,
        EquivalenceResult::Unknown { best_bound } => *best_bound,
        // `EquivalenceResult` is `#[non_exhaustive]`. A variant added later
        // reads as "no difference bound proven", never as a small one:
        // guessing low here would understate the gap.
        _ => f64::INFINITY,
    });

    Ok(EquivalenceOutcome {
        within_epsilon: is_equivalent(&result),
        summary: equivalence_summary(&result),
        max_abs_diff,
        epsilon,
        result,
    })
}

#[cfg(test)]
mod tests {
    use super::narrow_bound_up;

    #[test]
    fn narrowing_a_proven_bound_never_shrinks_it() {
        // A value whose nearest f32 rounds DOWN: the enclosure must not shrink.
        let v = 0.1_f64;
        assert!(
            f64::from(narrow_bound_up(v)) >= v,
            "narrowed {} must still enclose {v}",
            narrow_bound_up(v)
        );
        // Exact and edge values round-trip unchanged.
        assert_eq!(narrow_bound_up(0.0), 0.0);
        assert_eq!(narrow_bound_up(0.5), 0.5);
        assert!(narrow_bound_up(f64::INFINITY).is_infinite());
        assert!(narrow_bound_up(f64::NAN).is_nan());

        // Sweep: no input may come back smaller than it went in.
        for i in 0..2000 {
            let v = f64::from(i) * 0.001_3;
            assert!(
                f64::from(narrow_bound_up(v)) >= v,
                "narrow_bound_up({v}) shrank the bound"
            );
        }
    }
}
