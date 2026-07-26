// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Laddered model verification adapter — consumes `ny_api::ladder`.
//!
//! NN already builds a [`ny_propagate::GraphNetwork`] from models (see
//! [`crate::graph_tensor`] and [`crate::trace_to_graph`]). This module wires that
//! graph into ny's escalating verification ladder so NN gets the full
//!
//! ```text
//! IBP → α-CROWN → CROWN → β-CROWN → (MIP, feature = "ny-complete")
//! ```
//!
//! escalation in a single call. The ladder is demand-driven: it starts at the
//! cheapest method (IBP) and only climbs to a tighter rung when the previous one
//! left the property unproven and its bounds remained loose. It stops as soon as
//! a rung returns a decisive verdict, and a complete method's `Violated` verdict
//! short-circuits with that counterexample.
//!
//! Soundness is preserved end-to-end by ny: the laddered result is never reported
//! as `Verified` more strongly than the underlying method achieved, and soundness
//! provenance is combined across every rung that ran.
//!
//! This adapter is additive — it does not touch NN's existing verify path. It
//! exposes thin NN-facing wrappers ([`verify_model_laddered`],
//! [`verify_model_laddered_with_config`]) plus a small reporting helper
//! ([`laddered_summary`]) and does not deep-map into nn-verify's heavier result
//! types; callers that want the rich verdict read
//! [`ny_api::ladder::LadderedResult`] directly.
//!
//! ```rust,no_run
//! use nn_verify::ny_ladder::{laddered_summary, verify_model_laddered};
//! # fn run(net: &ny_propagate::GraphNetwork, spec: &ny_core::VerificationSpec)
//! #     -> ny_core::Result<()> {
//! let laddered = verify_model_laddered(net, spec)?;
//! println!("{}", laddered_summary(&laddered));
//! # Ok(())
//! # }
//! ```

use ny_api::ladder::{verify_model, LadderConfig, LadderedResult};

/// Default laddered configuration for NN model verification.
///
/// Delegates to [`LadderConfig::default`], which runs the full propagation ladder
/// up to β-CROWN, escalates only when bounds remain loose, and leaves the
/// complete MIP terminal disabled. Exposed as a named helper so NN call sites
/// document their intent rather than reaching for `Default` inline.
#[must_use]
pub fn default_ladder_config() -> LadderConfig {
    LadderConfig::default()
}

/// Verify a model graph through ny's escalating verification ladder using a
/// sensible default configuration.
///
/// Runs IBP → α-CROWN → CROWN → β-CROWN, escalating only on demand and stopping
/// at the first decisive verdict. For control over the method ceiling, escalation
/// threshold, per-rung timeout, or the complete MIP terminal, use
/// [`verify_model_laddered_with_config`].
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while running a rung (e.g. a
/// structural mismatch between the graph and the spec).
pub fn verify_model_laddered(
    net: &ny_propagate::GraphNetwork,
    spec: &ny_core::VerificationSpec,
) -> ny_core::Result<LadderedResult> {
    verify_model(net, spec, &default_ladder_config())
}

/// Verify a model graph through ny's escalating verification ladder with an
/// explicit [`LadderConfig`].
///
/// See [`verify_model_laddered`] for the default-config convenience wrapper and
/// the ladder semantics.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while running a rung.
pub fn verify_model_laddered_with_config(
    net: &ny_propagate::GraphNetwork,
    spec: &ny_core::VerificationSpec,
    cfg: &LadderConfig,
) -> ny_core::Result<LadderedResult> {
    verify_model(net, spec, cfg)
}

/// Render an NN-friendly one-line summary of a laddered run.
///
/// Reports the deciding verdict, the method that produced it, and the per-rung
/// escalation trace (method + verified/unproven). This is intentionally a flat
/// string for NN's reporting style; callers needing structured data read the
/// [`LadderedResult`] fields directly.
#[must_use]
pub fn laddered_summary(result: &LadderedResult) -> String {
    let verdict = if result.result.is_verified() {
        "verified"
    } else {
        "unproven"
    };
    let rungs: Vec<String> = result
        .rungs
        .iter()
        .map(|rung| {
            let mark = if rung.verified { "ok" } else { "·" };
            format!("{}[{mark}]", rung.method)
        })
        .collect();
    format!(
        "ladder: {verdict} via {} ({} rung{}: {})",
        result.method_used,
        result.rungs.len(),
        if result.rungs.len() == 1 { "" } else { "s" },
        rungs.join(" → "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_api::verify::PropagationMethod;
    use ny_core::{Bound, VerificationSpec};
    use ny_propagate::layers::{Layer, LinearLayer, ReLULayer};
    use ny_propagate::{GraphNetwork, GraphNode};

    /// Tiny Linear → ReLU → Linear graph (2→2→2) for end-to-end ladder exercise.
    ///
    /// Mirrors the construction NN uses elsewhere (e.g. `dead_neuron_proof_tests`),
    /// keeping the network small and deterministic.
    fn linear_relu_linear_graph() -> GraphNetwork {
        let w1 =
            ndarray::Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).expect("2x2 w1");
        let linear1 =
            LinearLayer::new(w1, Some(ndarray::Array1::zeros(2))).expect("valid linear1");
        let w2 = ndarray::Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).expect("2x2 w2");
        let linear2 =
            LinearLayer::new(w2, Some(ndarray::Array1::zeros(2))).expect("valid linear2");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear2");
        graph
    }

    #[test]
    fn ny_ladder_verifies_satisfiable_spec() {
        let graph = linear_relu_linear_graph();
        // Input in [-1, 1]^2. The graph's outputs are comfortably bounded; a wide
        // output spec is provable by the very first (IBP) rung.
        let spec = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            vec![Bound::new(-10.0, 10.0), Bound::new(-10.0, 10.0)],
        )
        .expect("valid spec");

        let result = verify_model_laddered(&graph, &spec)
            .expect("ladder should run on linear→relu→linear graph");

        assert!(
            result.result.is_verified(),
            "wide spec should verify, got {:?}",
            result.result
        );
        // method_used + rungs must be populated by the ladder.
        assert!(
            !result.rungs.is_empty(),
            "ladder must record at least the IBP rung"
        );
        assert_eq!(
            result.rungs[0].method,
            ny_core::MethodUsed::Ibp,
            "the easy case must be proven by the first (IBP) rung"
        );
        assert_eq!(
            result.method_used,
            ny_core::MethodUsed::Ibp,
            "method_used should report the deciding (IBP) rung"
        );

        // The NN-facing summary should mention the verdict and method.
        let summary = laddered_summary(&result);
        assert!(summary.contains("verified"), "summary: {summary}");
        assert!(summary.contains("Ibp"), "summary: {summary}");
    }

    #[test]
    fn ny_ladder_escalates_past_ibp_when_needed() {
        let graph = linear_relu_linear_graph();
        // A narrow output spec that IBP's loose interval arithmetic cannot prove,
        // forcing the ladder to climb at least one rung beyond IBP.
        let spec = VerificationSpec::new(
            vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
            vec![Bound::new(-0.5, 0.5), Bound::new(-0.5, 0.5)],
        )
        .expect("valid spec");

        // Force escalation (threshold 0) but cap at α-CROWN and bound per-rung
        // time to keep the test fast and deterministic.
        let cfg = LadderConfig {
            max_method: PropagationMethod::AlphaCrown,
            escalation_width_threshold: 0.0,
            use_complete: false,
            timeout_ms: Some(5_000),
        };

        let result = verify_model_laddered_with_config(&graph, &spec, &cfg)
            .expect("ladder should run with escalation config");

        assert_eq!(
            result.rungs[0].method,
            ny_core::MethodUsed::Ibp,
            "first rung must always be IBP"
        );
        assert!(
            !result.rungs[0].verified,
            "the narrow spec must not be IBP-provable (precondition of this test)"
        );
        assert!(
            result.rungs.len() >= 2,
            "escalation should advance past the IBP rung; rungs = {:?}",
            result.rungs
        );
        // The summary should reflect the multi-rung escalation.
        let summary = laddered_summary(&result);
        assert!(summary.contains("rungs"), "summary: {summary}");
    }

    #[test]
    fn default_config_matches_ladder_default() {
        let cfg = default_ladder_config();
        assert_eq!(cfg.max_method, PropagationMethod::BetaCrown);
        assert!(!cfg.use_complete);
    }
}
