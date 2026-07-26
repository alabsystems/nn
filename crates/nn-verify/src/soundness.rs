// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Soundness provenance for NY graph networks.
//!
//! Delegates layer scanning to `ny_propagate::soundness_provenance_for_graph`
//! (made `pub` in NY 497b16c7) and adds nn-specific `SqrtNegativeDomain`
//! detection when input bounds are available.

use ny_api::BoundedTensor;
use ny_core::{HeuristicUsed, SoundnessProvenance};
use ny_propagate::{GraphNetwork, Layer, PropagationMethod};

use crate::soundness_compat::VerificationSoundnessMode;
use crate::PropMethod;

/// Map nn's [`PropMethod`] to gamma-propagate's [`PropagationMethod`].
fn to_propagation_method(method: &PropMethod) -> PropagationMethod {
    match method {
        PropMethod::Ibp | PropMethod::Analytical | PropMethod::MixedIbpCrown => {
            PropagationMethod::Ibp
        }
        PropMethod::Crown => PropagationMethod::Crown,
        PropMethod::AlphaCrown => PropagationMethod::AlphaCrown,
        PropMethod::BetaCrown => PropagationMethod::BetaCrown,
    }
}

/// Compute soundness provenance for a [`GraphNetwork`].
///
/// Layer scanning (heuristic flags for LayerNorm, Softmax, GELU, Sin, Cos, etc.)
/// is handled by the upstream `ny_propagate::soundness_provenance_for_graph`.
///
/// When `input_bounds` is `Some`, also checks for `SqrtNegativeDomain` — Sqrt
/// layers whose input bounds include negative values. This requires IBP
/// propagation through the graph, so it is only performed when bounds are available.
///
/// When `uses_comparison_approximation` is `true`, the provenance is forced to
/// `Heuristic` mode because the graph models discrete comparisons (Gt, Ge, Eq,
/// Ne) as continuous approximations (Sub, Abs, MulConstant). This is a sound
/// over-approximation but can cause NY to consider both branches of a
/// Select/Where active even when only one is reachable, producing looser bounds.
pub(crate) fn soundness_for_graph(
    graph: &GraphNetwork,
    method: &PropMethod,
    input_bounds: Option<&BoundedTensor>,
    uses_comparison_approximation: bool,
) -> Result<SoundnessProvenance, crate::VerifyError> {
    let upstream =
        ny_propagate::soundness_provenance_for_graph(graph, &to_propagation_method(method));

    let mut heuristics = upstream.heuristics_used().to_vec();

    // SqrtNegativeDomain: requires input bounds to propagate through the graph.
    // Uses gamma-propagate's count_sqrt_negative_domain_graph which performs IBP
    // propagation (allowing negative sqrt inputs) and counts Sqrt nodes receiving
    // negative lower bounds. This is nn-specific — upstream only scans layer flags.
    //
    // Optimization (#593): skip the expensive IBP re-propagation when the graph
    // has no Sqrt nodes. count_sqrt_negative_domain_graph does this check
    // internally on its private `nodes` field, but calling it still enters
    // gamma-propagate and allocates. Pre-checking via the public API avoids the
    // call entirely for the common case (most kernel graphs have no Sqrt).
    if let Some(bounds) = input_bounds {
        let has_sqrt = graph
            .node_names()
            .iter()
            .filter_map(|name| graph.node(name))
            .any(|node| matches!(node.layer(), Layer::Sqrt(_)));
        if has_sqrt {
            let sqrt_neg_count = ny_propagate::count_sqrt_negative_domain_graph(graph, bounds)?;
            if sqrt_neg_count > 0 {
                heuristics.push(HeuristicUsed::SqrtNegativeDomain {
                    num_nodes: sqrt_neg_count,
                });
            }
        }
    }

    // ContinuousComparisonApproximation: flagged during IR-to-graph translation
    // when any Compare node has variable operands. The NY graph models
    // Gt/Ge as `lhs - rhs`, Eq as `-(abs(lhs - rhs))`, Ne as `abs(lhs - rhs)`.
    // These are sound but the composition loses the discrete semantics.
    //
    // Note: Uses SamplingBasedNonlinearRelaxations as a stand-in until gamma-core
    // adds a ContinuousComparisonApproximation variant. The text in the provenance
    // is informational — what matters is that the mode becomes Heuristic.
    // Tracked: NY#3141 requests ContinuousComparisonApproximation variant.
    if uses_comparison_approximation {
        heuristics.push(HeuristicUsed::SamplingBasedNonlinearRelaxations);
    }

    if heuristics.is_empty() {
        Ok(upstream)
    } else {
        Ok(SoundnessProvenance::from_heuristics(heuristics))
    }
}

/// Compute the public soundness mode for a graph/method pair.
///
/// This is the stable API for downstream crates that need to classify a
/// propagation result as `Sound` or `Heuristic` without reimplementing the
/// NY layer scan logic.
pub fn soundness_mode_for_graph(
    graph: &GraphNetwork,
    method: &PropMethod,
    input_bounds: Option<&BoundedTensor>,
) -> Result<VerificationSoundnessMode, crate::VerifyError> {
    Ok(soundness_for_graph(graph, method, input_bounds, false)?.mode())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_propagate::layers::SqrtLayer;
    use ny_propagate::{GraphNode, Layer};
    use ndarray::{ArrayD, IxDyn};

    /// When `count_sqrt_negative_domain_graph` returns Err, `soundness_for_graph`
    /// must propagate the error (fail-closed) rather than silently dropping it
    /// and returning a potentially incorrect `Sound` classification.
    #[test]
    fn test_sqrt_propagate_error_returns_err_not_silent_sound() {
        // Build a graph with a Sqrt node whose input references a node that
        // does not exist. IBP propagation will fail with InvalidSpec.
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::new(
            "sqrt_0",
            Layer::Sqrt(SqrtLayer::new()),
            vec!["nonexistent_node".to_string()],
        ));
        graph.set_output("sqrt_0");

        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), -1.0f32),
            ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        )
        .expect("valid bounds");

        let method = PropMethod::Ibp;
        let result = soundness_for_graph(&graph, &method, Some(&input), false);

        // Before the fix, this would return Ok(Sound) — silently swallowing the
        // gamma-propagate error. After the fix, it must return Err.
        assert!(
            result.is_err(),
            "soundness_for_graph must propagate gamma-propagate errors (fail-closed), \
             but returned Ok({:?}) — this is a fail-open soundness bug",
            result.unwrap()
        );
    }

    /// When no input bounds are provided, the sqrt check is skipped and the
    /// function should succeed (no error to propagate).
    #[test]
    fn test_no_input_bounds_skips_sqrt_check() {
        let graph = GraphNetwork::new();
        let method = PropMethod::Ibp;

        let result = soundness_for_graph(&graph, &method, None, false);
        assert!(result.is_ok(), "no input bounds should skip sqrt check");
    }

    /// When comparison approximation flag is set, provenance must be Heuristic.
    #[test]
    fn test_comparison_approximation_forces_heuristic() {
        let graph = GraphNetwork::new();
        let method = PropMethod::Ibp;

        let result = soundness_for_graph(&graph, &method, None, true)
            .expect("should succeed for empty graph");
        assert_eq!(
            result.mode(),
            VerificationSoundnessMode::Heuristic,
            "comparison approximation must force Heuristic mode"
        );
    }

    /// Without comparison approximation, empty graph should be Sound.
    #[test]
    fn test_no_comparison_approximation_is_sound() {
        let graph = GraphNetwork::new();
        let method = PropMethod::Ibp;

        let result = soundness_for_graph(&graph, &method, None, false)
            .expect("should succeed for empty graph");
        assert_eq!(
            result.mode(),
            VerificationSoundnessMode::Sound,
            "no heuristics means Sound"
        );
    }
}
