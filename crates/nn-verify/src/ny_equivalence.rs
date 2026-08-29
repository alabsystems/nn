// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Network-equivalence verification adapter — consumes `ny_api::equivalence`.
//!
//! Equivalence verification proves `||f(x) - g(x)|| < eps` for every `x` in an
//! input region by stitching both graphs onto a shared input, appending a final
//! subtraction node (the *difference network* `h(x) = f(x) - g(x)`), and bound
//! propagating `h` to show every output lies within `[-eps, eps]`.
//!
//! NN already consumes [`ny_propagate::build_difference_network`] directly in
//! [`crate::resblock_equivalence`] for the FusedResBlock peephole proofs. This
//! module exposes the *general* curated facade instead, so **any** two NN-built
//! [`ny_propagate::GraphNetwork`]s — e.g. an optimized DSL plan vs. its
//! reference lowering — can be proven equivalent in one call.
//!
//! Method choice matters: plain IBP treats the two branches as independent
//! intervals, and `[a, b] - [a, b] = [a - b, b - a]` can never cancel to zero,
//! so IBP cannot prove equivalence of anything but constants. The CROWN family
//! carries each output's linear dependence on the shared input, letting the
//! correlated terms cancel; the default configuration therefore uses ny's
//! default alpha-CROWN [`PropagationConfig`].
//!
//! This adapter is additive — it does not touch NN's existing verify path (nor
//! the specialized `resblock_equivalence` proofs). It exposes thin NN-facing
//! wrappers ([`verify_networks_equivalent`],
//! [`verify_networks_equivalent_with_config`]), a difference-network builder
//! pass-through ([`build_difference_network`]), and a small reporting helper
//! ([`equivalence_summary`]); callers that want the structured verdict read
//! [`ny_api::equivalence::EquivalenceResult`] directly.
//!
//! ```rust,no_run
//! use nn_verify::ny_equivalence::{equivalence_summary, verify_networks_equivalent};
//! # fn run(f: &ny_propagate::GraphNetwork, g: &ny_propagate::GraphNetwork)
//! #     -> ny_core::Result<()> {
//! let region = vec![ny_core::Bound::new(-1.0, 1.0), ny_core::Bound::new(-1.0, 1.0)];
//! let verdict = verify_networks_equivalent(f, g, &region, 1e-4)?;
//! println!("{}", equivalence_summary(&verdict));
//! # Ok(())
//! # }
//! ```

use ny_api::equivalence::{verify_equivalence, EquivalenceResult};
use ny_api::verify::PropagationConfig;

/// Default propagation configuration for NN equivalence proving.
///
/// Delegates to [`PropagationConfig::default`] (alpha-CROWN, CPU, McCormick
/// binary-mul relaxation). Alpha-CROWN tracks each output's linear dependence
/// on the shared input, which is what lets the two branches of the difference
/// network cancel — do not swap in `PropagationMethod::Ibp`, whose interval
/// subtraction can never prove two non-constant networks equivalent. Exposed as
/// a named helper so NN call sites document their intent rather than reaching
/// for `Default` inline.
#[must_use]
pub fn default_equivalence_config() -> PropagationConfig {
    PropagationConfig::default()
}

/// Prove that two NN-built graphs produce equivalent outputs within `epsilon`
/// over `input_bounds`, using a sensible default configuration.
///
/// Builds the difference network `h(x) = f(x) - g(x)` and checks that every
/// output of `h` lies within `[-epsilon, epsilon]` for all `x` in the region.
/// The returned [`EquivalenceResult`] carries the worst-case proven output
/// difference. Note that `NotEquivalent` is *not* a counterexample — it means
/// the verifier's sound bounds were too loose to prove agreement. For control
/// over the propagation method, use
/// [`verify_networks_equivalent_with_config`].
///
/// # Errors
///
/// Returns [`ny_core::NyError::InvalidSpec`] if `epsilon` is not positive, if
/// either graph has no output node set, or if the two graphs' output dimensions
/// differ; propagates any [`ny_core::NyError`] raised during bound propagation.
pub fn verify_networks_equivalent(
    f: &ny_propagate::GraphNetwork,
    g: &ny_propagate::GraphNetwork,
    input_bounds: &[ny_core::Bound],
    epsilon: f32,
) -> ny_core::Result<EquivalenceResult> {
    verify_equivalence(f, g, input_bounds, epsilon, default_equivalence_config())
}

/// Prove that two NN-built graphs produce equivalent outputs within `epsilon`
/// with an explicit [`PropagationConfig`].
///
/// See [`verify_networks_equivalent`] for the default-config convenience
/// wrapper and the difference-network semantics.
///
/// # Errors
///
/// Returns [`ny_core::NyError::InvalidSpec`] on a non-positive `epsilon`, a
/// missing output node, or mismatched output dimensions; propagates any
/// [`ny_core::NyError`] raised during bound propagation.
pub fn verify_networks_equivalent_with_config(
    f: &ny_propagate::GraphNetwork,
    g: &ny_propagate::GraphNetwork,
    input_bounds: &[ny_core::Bound],
    epsilon: f32,
    cfg: &PropagationConfig,
) -> ny_core::Result<EquivalenceResult> {
    verify_equivalence(f, g, input_bounds, epsilon, cfg.clone())
}

/// Build the difference network `h(x) = f(x) - g(x)` from two NN-built graphs.
///
/// Pass-through to [`ny_api::equivalence::build_difference_network`]: both
/// graphs share the same input, node names are prefixed to avoid collisions,
/// and a final subtraction node produces the element-wise output difference.
/// Useful when NN wants to inspect the merged DAG or verify it under a custom
/// spec instead of going through [`verify_networks_equivalent`].
///
/// # Errors
///
/// Returns [`ny_core::NyError::InvalidSpec`] if either graph has no output
/// node set, if the output dimensions are statically detectable as mismatched,
/// or if a node-name collision survives prefixing.
pub fn build_difference_network(
    f: &ny_propagate::GraphNetwork,
    g: &ny_propagate::GraphNetwork,
) -> ny_core::Result<ny_propagate::GraphNetwork> {
    ny_api::equivalence::build_difference_network(f, g)
}

/// Whether an equivalence run proved the two networks equivalent.
///
/// Convenience predicate over the (`#[non_exhaustive]`)
/// [`EquivalenceResult`]: `true` only for
/// [`EquivalenceResult::Equivalent`].
#[must_use]
pub fn is_equivalent(result: &EquivalenceResult) -> bool {
    matches!(result, EquivalenceResult::Equivalent { .. })
}

/// Render an NN-friendly one-line summary of an equivalence run.
///
/// Reports the verdict and the relevant worst-case output-difference bound.
/// This is intentionally a flat string for NN's reporting style; callers
/// needing structured data match on the [`EquivalenceResult`] directly.
#[must_use]
pub fn equivalence_summary(result: &EquivalenceResult) -> String {
    match result {
        EquivalenceResult::Equivalent { bound } => {
            format!("equivalence: equivalent (worst-case proven diff {bound:.3e})")
        }
        EquivalenceResult::NotEquivalent { worst_case_bound } => format!(
            "equivalence: not proven (worst-case diff bound {worst_case_bound:.3e} \
             exceeds epsilon)"
        ),
        EquivalenceResult::Unknown { best_bound } => format!(
            "equivalence: unknown (best diff bound {best_bound:.3e}; timeout or inconclusive)"
        ),
        _ => "equivalence: unrecognized verdict".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_api::verify::PropagationMethod;
    use ny_core::Bound;
    use ny_propagate::layers::{Layer, LinearLayer, ReLULayer};
    use ny_propagate::{GraphNetwork, GraphNode};

    /// Tiny Linear → ReLU → Linear graph (2→2→2), parameterized by the second
    /// linear layer's bias so two structurally identical graphs can differ by a
    /// known constant output offset.
    ///
    /// The ReLU is *stably active* over the test region `[-1, 1]²`: `w1 = I`
    /// and `b1 = [2, 2]` put every pre-activation in `[1, 3]`. A stable ReLU
    /// relaxes exactly, so CROWN-family propagation through the difference
    /// network cancels the correlated terms and the proven diff bound is exact
    /// up to `f32` round-off. Mirrors the small deterministic construction NN
    /// uses elsewhere (e.g. the `ny_ladder` tests).
    fn stable_relu_graph(bias2: [f32; 2]) -> GraphNetwork {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("2x2 w1");
        let b1 = ndarray::Array1::from_vec(vec![2.0, 2.0]);
        let linear1 = LinearLayer::new(w1, Some(b1)).expect("valid linear1");
        let w2 =
            ndarray::Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, -1.0]).expect("2x2 w2");
        let b2 = ndarray::Array1::from_vec(bias2.to_vec());
        let linear2 = LinearLayer::new(w2, Some(b2)).expect("valid linear2");

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

    /// The `[-1, 1]²` input region every test verifies over.
    fn unit_box() -> Vec<Bound> {
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)]
    }

    #[test]
    fn identical_graphs_verify_equivalent_at_tight_epsilon() {
        let f = stable_relu_graph([0.0, 0.0]);
        let g = stable_relu_graph([0.0, 0.0]);

        let result = verify_networks_equivalent(&f, &g, &unit_box(), 1e-3)
            .expect("equivalence should run on identical linear→relu→linear graphs");

        assert!(
            is_equivalent(&result),
            "identical graphs must verify equivalent at eps=1e-3, got {result:?}"
        );
        if let EquivalenceResult::Equivalent { bound } = &result {
            assert!(
                bound.abs() < 1e-3,
                "diff bound of identical graphs should be ~0 (f32 round-off), got {bound}"
            );
        } else {
            unreachable!("guarded by is_equivalent above");
        }

        // The NN-facing summary should mention the verdict.
        let summary = equivalence_summary(&result);
        assert!(summary.contains("equivalent"), "summary: {summary}");
    }

    #[test]
    fn differing_graphs_fail_at_tight_epsilon() {
        let f = stable_relu_graph([0.0, 0.0]);
        // g's second linear bias differs by 0.75 in output 0, so
        // h(x) = f(x) - g(x) = [-0.75, 0] exactly, everywhere in the region.
        let g = stable_relu_graph([0.75, 0.0]);

        // Exercise the explicit-config wrapper on the plain CROWN rung.
        let cfg = PropagationConfig {
            method: PropagationMethod::Crown,
            ..default_equivalence_config()
        };
        let result = verify_networks_equivalent_with_config(&f, &g, &unit_box(), 0.05, &cfg)
            .expect("equivalence should run on differing graphs");

        assert!(
            !is_equivalent(&result),
            "a constant 0.75 output gap cannot verify at eps=0.05, got {result:?}"
        );
        if let EquivalenceResult::NotEquivalent { worst_case_bound } = &result {
            assert!(
                (0.7..0.8).contains(worst_case_bound),
                "worst-case bound should be ~0.75 (stable ReLU => exact CROWN), \
                 got {worst_case_bound}"
            );
        } else {
            panic!("expected NotEquivalent at tight epsilon, got {result:?}");
        }

        let summary = equivalence_summary(&result);
        assert!(summary.contains("not proven"), "summary: {summary}");
    }

    #[test]
    fn differing_graphs_verify_at_loose_epsilon() {
        let f = stable_relu_graph([0.0, 0.0]);
        let g = stable_relu_graph([0.75, 0.0]);

        // The same 0.75 gap fits comfortably inside eps=2.0.
        let result = verify_networks_equivalent(&f, &g, &unit_box(), 2.0)
            .expect("equivalence should run on differing graphs at loose epsilon");

        assert!(
            is_equivalent(&result),
            "a 0.75 output gap must verify at eps=2.0, got {result:?}"
        );
        if let EquivalenceResult::Equivalent { bound } = &result {
            assert!(
                (0.7..0.8).contains(bound),
                "proven worst-case diff should be ~0.75, got {bound}"
            );
        } else {
            unreachable!("guarded by is_equivalent above");
        }
    }

    #[test]
    fn difference_network_passthrough_builds_merged_dag() {
        let f = stable_relu_graph([0.0, 0.0]);
        let g = stable_relu_graph([0.75, 0.0]);

        let h = build_difference_network(&f, &g).expect("difference network should build");

        // Three prefixed nodes per branch plus the final subtraction node.
        assert_eq!(h.num_nodes(), 7, "3 + 3 + 1 (sub) nodes expected");
        assert!(
            !h.output_name().is_empty(),
            "difference network must set its output node"
        );
    }

    #[test]
    fn non_positive_epsilon_is_rejected() {
        let f = stable_relu_graph([0.0, 0.0]);
        let g = stable_relu_graph([0.0, 0.0]);

        let result = verify_networks_equivalent(&f, &g, &unit_box(), 0.0);
        assert!(
            result.is_err(),
            "epsilon = 0 must be rejected as InvalidSpec"
        );
    }

    #[test]
    fn default_config_matches_propagation_default() {
        let cfg = default_equivalence_config();
        assert_eq!(cfg.method, PropagationMethod::AlphaCrown);
        assert!(!cfg.use_gpu);
    }
}
