// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Complete verification via ny's MIP terminal (`ny_api::complete`, HiGHS).
//!
//! Bound propagation (IBP → CROWN → β-CROWN) is *sound but incomplete*: it can
//! answer `Verified`, but an unproven property may still be true — the relaxation
//! was simply too loose. The MIP terminal is **complete** for the fragment it
//! encodes (feed-forward ReLU networks with a linear property): it encodes the
//! network exactly as a mixed-integer program and asks for feasibility of the
//! *negated* property, so
//!
//! - `MipResult::Unsat` ⇒ no counterexample exists ⇒ the property holds;
//! - `MipResult::Sat` ⇒ a concrete counterexample input was found.
//!
//! # Why this module exists
//!
//! The `ny-complete` cargo feature forwards to `ny-api/complete`, but that alone
//! never reaches the MIP rung: [`ny_api::ladder::LadderConfig::use_complete`]
//! defaults to `false`, so the ladder stops at β-CROWN. Before this module NN
//! compiled the terminal and could not call it. [`complete_ladder_config`] sets
//! the flag; [`verify_model_complete`] is the one-call entry point.
//!
//! # Soundness
//!
//! The MIP rung only ever *strengthens* a verdict: the ladder runs it after the
//! propagation rungs left the property unproven, and a `Violated` verdict
//! short-circuits with its counterexample. A `Verified` from MIP is a complete
//! proof for the encoded fragment, not a relaxation. Networks outside that
//! fragment degrade to the propagation verdict rather than claiming completeness.
//!
//! ```rust,no_run
//! use nn_verify::ny_complete::verify_model_complete;
//! # fn demo(net: &ny_propagate::GraphNetwork, spec: &ny_core::VerificationSpec)
//! #     -> ny_core::Result<()> {
//! let laddered = verify_model_complete(net, spec)?;
//! // `laddered.method_used` is `MethodUsed::Mip` when the terminal decided it.
//! # Ok(())
//! # }
//! ```

use ny_api::ladder::{verify_model, LadderConfig, LadderedResult};

/// The raw MIP surface, for callers that want to encode and solve directly
/// rather than through the ladder.
///
/// Use [`MipSolver::check_feasibility`] to decide a property. Do **not** use
/// `MipSolver::{minimize_output, maximize_output}`: `ny-mip` lowers the
/// feasibility IR, whose objective coefficients are all zero, then reads the
/// target column at whatever feasible point the solver lands on — so they return
/// an arbitrary feasible value, not an optimum (`TODO(#1763)` in
/// `ny-mip/src/solver.rs`; they have no callers in ny). [`LpTightener`] is
/// unaffected: it builds its own LP with a real objective.
pub use ny_api::complete::{
    encode_feedforward, LpTightener, MipConfig, MipEncoder, MipError, MipParts, MipResult,
    MipSolver,
};

/// A [`LadderConfig`] that escalates all the way to the complete MIP terminal.
///
/// Identical to [`LadderConfig::default`] except `use_complete = true`. The rung
/// still runs only when the propagation rungs left the property unproven.
#[must_use]
pub fn complete_ladder_config() -> LadderConfig {
    LadderConfig {
        use_complete: true,
        ..LadderConfig::default()
    }
}

/// Verify `spec` on `net`, escalating through propagation and, if still unproven,
/// into the complete MIP terminal.
///
/// This is [`crate::ny_ladder::verify_model_laddered`] with the MIP rung enabled.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while running a rung (e.g. a
/// structural mismatch between the graph and the spec, or a MIP encoding error
/// for a network outside the feed-forward ReLU fragment).
pub fn verify_model_complete(
    net: &ny_propagate::GraphNetwork,
    spec: &ny_core::VerificationSpec,
) -> ny_core::Result<LadderedResult> {
    verify_model(net, spec, &complete_ladder_config())
}

/// Verify with an explicit [`LadderConfig`], forcing the MIP terminal on.
///
/// The caller's `max_method`, `escalation_width_threshold` and `timeout_ms` are
/// respected; only `use_complete` is overridden, so this cannot silently skip the
/// terminal the caller asked for.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while running a rung.
pub fn verify_model_complete_with_config(
    net: &ny_propagate::GraphNetwork,
    spec: &ny_core::VerificationSpec,
    cfg: &LadderConfig,
) -> ny_core::Result<LadderedResult> {
    let cfg = LadderConfig {
        use_complete: true,
        ..cfg.clone()
    };
    verify_model(net, spec, &cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::Bound;

    /// `f(x) = |x|` as `Linear(1->2) -> ReLU -> Linear(2->1)`, `x in [-1, 1]`.
    ///
    /// Pre-activations are `x` and `-x`, each in `[-1, 1]`, so interval
    /// propagation relaxes the two ReLUs independently to `[0, 1]` each and
    /// concludes `f in [0, 2]`. The true range is `[0, 1]`: the two ReLUs can
    /// never both be active. Closing that relaxation gap is what the complete
    /// terminal buys, so this net separates it from every propagation rung.
    fn abs_net_encoder() -> MipEncoder {
        let weights = vec![vec![1.0, -1.0], vec![1.0, 1.0]];
        let biases = vec![vec![0.0, 0.0], vec![0.0]];
        let layer_dims = [1usize, 2, 1];
        let input_bounds = [Bound::new(-1.0, 1.0)];
        let intermediate_bounds = vec![vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)]];

        // `encode_feedforward` finalizes the output frontier itself.
        encode_feedforward(
            &weights,
            &biases,
            &layer_dims,
            &input_bounds,
            &intermediate_bounds,
        )
        .expect("feed-forward ReLU net is encodable")
    }

    fn solve_with_output_geq(threshold: f64) -> MipResult {
        let mut enc = abs_net_encoder();
        enc.constrain_output_geq_const(0, threshold)
            .expect("output 0 exists");
        MipSolver::new(enc.into_parts(), MipConfig::default())
            .check_feasibility()
            .expect("solve should not error")
    }

    /// `Unsat` is the complete verifier's proof (no counterexample exists);
    /// `Sat` carries one. Both directions are exercised, so a solver that
    /// always answered `Unsat` could not pass.
    #[test]
    fn feasibility_separates_a_true_property_from_a_false_one() {
        // `f <= 1.5` is TRUE, so its negation `f >= 1.5` must be infeasible.
        // Note IBP's relaxed upper bound of 2.0 cannot establish this.
        match solve_with_output_geq(1.5) {
            MipResult::Unsat => {}
            other => panic!("f >= 1.5 is infeasible for |x| on [-1,1], got {other:?}"),
        }

        // `f <= 0.5` is FALSE, so its negation `f >= 0.5` must be feasible,
        // and the witness must be a real input with |x| >= 0.5.
        match solve_with_output_geq(0.5) {
            MipResult::Sat { input_values, .. } => {
                let x = input_values.first().copied().expect("one input var");
                assert!(
                    x.abs() >= 0.5 - 1e-6,
                    "counterexample x={x} does not actually violate f <= 0.5",
                );
            }
            other => panic!("f >= 0.5 is feasible for |x| on [-1,1], got {other:?}"),
        }
    }

    #[test]
    fn default_ladder_leaves_mip_off_and_ours_turns_it_on() {
        assert!(
            !LadderConfig::default().use_complete,
            "ny's default must stay incomplete; this module exists to opt in",
        );
        assert!(complete_ladder_config().use_complete);
    }

    #[test]
    fn with_config_forces_the_terminal_on_without_clobbering_other_knobs() {
        let base = LadderConfig {
            use_complete: false,
            escalation_width_threshold: 0.25,
            timeout_ms: Some(1234),
            ..LadderConfig::default()
        };
        let forced = LadderConfig {
            use_complete: true,
            ..base
        };
        assert!(forced.use_complete);
        assert_eq!(forced.escalation_width_threshold, 0.25);
        assert_eq!(forced.timeout_ms, Some(1234));
    }
}
