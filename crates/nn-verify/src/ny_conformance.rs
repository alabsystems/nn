// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Operation soundness-coverage for a built model (ny conformance gate).
//!
//! Surfaces *silent soundness holes*: walks a built [`GraphNetwork`] and asks
//! ny's conformance gate ([`ny_api::conformance`]) how soundly bound-propagation
//! handles each operation, so NN can answer "is this model verifiable, and which
//! ops aren't soundly covered" *before* trusting a verification verdict.
//!
//! # How classification works
//!
//! ny's canonical gate, [`ny_api::conformance::soundness_class`], classifies a
//! [`ny_core::LayerType`]. A built `GraphNetwork`, however, exposes each node's
//! operation as a `ny_propagate` [`Layer`](ny_propagate::Layer), whose
//! [`layer_type()`](ny_propagate::Layer::layer_type) returns the propagate-side
//! name string. We bridge the two enums via [`ny_core::LayerType`]'s `FromStr`
//! impl, which maps propagate spellings (and their cross-enum aliases, e.g.
//! `LeakyReLU`→`LeakyRelu`, `RmsNorm`→`RMSNorm`, `MaxPool2d`→`MaxPool`,
//! `InstanceNorm1d`→`InstanceNorm`, `AdaIN1d`→`AdaIN`, `MulBinary`→`Mul`,
//! `SelfAttention`→`MultiHeadAttention`) back to the core variant.
//!
//! This is the same `node.layer().layer_type()` traversal that
//! [`crate::layer_bounds`] uses, so it stays in step with how the rest of
//! nn-verify reads a `GraphNetwork`.
//!
//! ## Affine/structural propagate-only helpers
//!
//! A handful of `ny_propagate` layers are *implementation* primitives the NN→ny
//! translator emits (e.g. the output-wrapping `AddConstant` identity in
//! `graph_tensor.rs`, or `MulConstant` for a scalar `SiLU * up`). These have no
//! `ny_core::LayerType` variant, so `FromStr` resolves them to
//! [`ny_core::LayerType::Unknown`] → [`SoundnessClass::Unsupported`]. They are
//! nonetheless genuinely *exact* (affine-by-constant) or structural identities,
//! so reporting them as soundness holes would be a false positive. We therefore
//! recognize a small, explicit allow-list of these primitives and classify them
//! as [`SoundnessClass::Exact`]. Every layer string outside that allow-list that
//! still fails to resolve is reported honestly as [`SoundnessClass::Unsupported`]
//! (the real silent-hole signal), with its name captured in
//! [`ModelCoverage::unsupported_layers`].

use std::str::FromStr;

use ny_propagate::GraphNetwork;

// Re-export ny's canonical conformance gate so NN consumers classify ops through
// the facade rather than reaching past it.
pub use ny_api::conformance::{is_verifiable, soundness_class, SoundnessClass};

/// Per-model op soundness-coverage tally.
///
/// Buckets every operation in a built model by its [`SoundnessClass`] and lists
/// the operations that are not soundly covered, so a caller can both gate on
/// verifiability and report *which* ops are the holes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelCoverage {
    /// Operations with exact bounds (affine / structural / index-permutation).
    pub exact: usize,
    /// Operations with sound, reasonably tight bounds.
    pub sound: usize,
    /// Operations that are sound but may be loose.
    pub loose: usize,
    /// Operations not soundly supported for verification.
    pub unsupported: usize,
    /// `layer_type()` names of the unsupported operations, in graph order
    /// (deduplicated, preserving first-seen order). Empty iff `unsupported == 0`.
    pub unsupported_layers: Vec<String>,
}

impl ModelCoverage {
    /// Total number of classified operations.
    #[must_use]
    pub fn total(&self) -> usize {
        self.exact + self.sound + self.loose + self.unsupported
    }

    /// Whether every operation is soundly covered (no `Unsupported`).
    #[must_use]
    pub fn is_fully_verifiable(&self) -> bool {
        self.unsupported == 0
    }
}

/// Propagate-only primitives that have no `ny_core::LayerType` variant but are
/// exact affine-by-constant maps or structural identities. These are emitted by
/// the NN→ny translator (not user ops), so they must not be reported as
/// soundness holes. See the module docs.
///
/// Kept deliberately conservative: only operators whose bound propagation is
/// genuinely exact are listed. Anything not here that fails to resolve is
/// reported as `Unsupported`.
const EXACT_PROPAGATE_PRIMITIVES: &[&str] = &[
    "AddConstant",        // y = x + c  (output-identity wrapper, residual constants)
    "SubConstant",        // y = x - c
    "MulConstant",        // y = x * c  (e.g. scalar SiLU * up)
    "DivConstant",        // y = x / c
    "ExpandLikeLastAxis", // broadcast/structural reshape, value-preserving
];

/// Classify one propagate layer-type name into a [`SoundnessClass`].
///
/// Bridges the propagate-side `&str` to a [`ny_core::LayerType`] via `FromStr`,
/// then defers to ny's canonical [`soundness_class`]. Recognizes the exact
/// affine/structural propagate-only primitives (see [`EXACT_PROPAGATE_PRIMITIVES`])
/// that have no core variant, classifying them as [`SoundnessClass::Exact`]
/// rather than letting them fall through to `Unknown`/`Unsupported`.
fn classify_layer_type_name(layer_type: &str) -> SoundnessClass {
    // `FromStr` never errors (it returns `LayerType::Unknown` for unrecognized
    // strings), so the `unwrap_or` is just for total-function hygiene.
    let parsed = ny_core::LayerType::from_str(layer_type).unwrap_or(ny_core::LayerType::Unknown);
    match soundness_class(&parsed) {
        // Recover exact affine/structural primitives that have no core variant.
        SoundnessClass::Unsupported if EXACT_PROPAGATE_PRIMITIVES.contains(&layer_type) => {
            SoundnessClass::Exact
        }
        other => other,
    }
}

/// Walk a built model and tally op soundness coverage.
///
/// Classifies each node's operation via [`ny_api::conformance::soundness_class`]
/// (bridged from the propagate `Layer` name through [`ny_core::LayerType`]) and
/// counts it into the matching [`ModelCoverage`] bucket. Unsupported ops also
/// have their layer-type name recorded (deduplicated, first-seen order) in
/// [`ModelCoverage::unsupported_layers`].
///
/// Nodes are visited in the graph's insertion order
/// ([`GraphNetwork::node_names`]); coverage is order-independent, but the
/// unsupported-name list reflects this order.
#[must_use]
pub fn model_op_coverage(net: &GraphNetwork) -> ModelCoverage {
    let mut coverage = ModelCoverage::default();

    for name in net.node_names() {
        let Some(node) = net.node(name) else {
            // node_names() and the node map are kept in lockstep; skip defensively.
            continue;
        };
        let layer_type = node.layer().layer_type();
        match classify_layer_type_name(layer_type) {
            SoundnessClass::Exact => coverage.exact += 1,
            SoundnessClass::Sound => coverage.sound += 1,
            SoundnessClass::SoundButLoose => coverage.loose += 1,
            SoundnessClass::Unsupported => {
                coverage.unsupported += 1;
                let owned = layer_type.to_string();
                if !coverage.unsupported_layers.contains(&owned) {
                    coverage.unsupported_layers.push(owned);
                }
            }
        }
    }

    coverage
}

/// Whether ny can soundly verify the whole model (no `Unsupported` op).
///
/// Convenience wrapper over [`model_op_coverage`] for gate-style call sites that
/// only need a yes/no answer.
#[must_use]
pub fn is_fully_verifiable(net: &GraphNetwork) -> bool {
    model_op_coverage(net).is_fully_verifiable()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use ny_propagate::layers::{Layer, LinearLayer, ReLULayer, WhereLayer};
    use ny_propagate::{GraphNode, NETWORK_INPUT};

    /// Build a tiny Linear -> ReLU -> Linear `GraphNetwork` directly from
    /// propagate layers. Deterministic, no external model needed.
    fn fc_relu_fc_net() -> GraphNetwork {
        let mut net = GraphNetwork::new();

        // Linear: 2 -> 2
        let w1 = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("w1 shape");
        let b1 = Array1::from_vec(vec![0.0, 0.0]);
        let lin1 = LinearLayer::new(w1, Some(b1)).expect("valid linear layer");
        net.add_node(GraphNode::from_input("lin1", Layer::Linear(lin1)));

        // ReLU
        net.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["lin1".to_string()],
        ));

        // Linear: 2 -> 1
        let w2 = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).expect("w2 shape");
        let b2 = Array1::from_vec(vec![0.0]);
        let lin2 = LinearLayer::new(w2, Some(b2)).expect("valid linear layer");
        net.add_node(GraphNode::new(
            "lin2",
            Layer::Linear(lin2),
            vec!["relu".to_string()],
        ));

        net.set_output("lin2");
        net
    }

    #[test]
    fn classify_layer_type_name_bridges_propagate_to_core() {
        // Direct names.
        assert_eq!(classify_layer_type_name("Linear"), SoundnessClass::Exact);
        assert_eq!(classify_layer_type_name("ReLU"), SoundnessClass::Sound);
        assert_eq!(
            classify_layer_type_name("Softmax"),
            SoundnessClass::SoundButLoose
        );
        // Cross-enum aliases (propagate spelling -> core variant).
        assert_eq!(classify_layer_type_name("LeakyReLU"), SoundnessClass::Sound);
        assert_eq!(
            classify_layer_type_name("RmsNorm"),
            SoundnessClass::SoundButLoose
        );
        assert_eq!(classify_layer_type_name("MaxPool2d"), SoundnessClass::Sound);
        // Data-dependent op stays unsupported.
        assert_eq!(
            classify_layer_type_name("Where"),
            SoundnessClass::Unsupported
        );
    }

    #[test]
    fn affine_propagate_primitives_are_recovered_as_exact() {
        // These have no ny_core::LayerType variant; without the allow-list they
        // would resolve to Unknown -> Unsupported and be false soundness holes.
        for prim in ["AddConstant", "MulConstant", "SubConstant", "DivConstant"] {
            assert_eq!(
                classify_layer_type_name(prim),
                SoundnessClass::Exact,
                "{prim} should be recovered as Exact"
            );
        }
        // A genuinely unmapped, non-affine primitive is honestly Unsupported.
        assert_eq!(
            classify_layer_type_name("OpaqueSkip"),
            SoundnessClass::Unsupported
        );
    }

    #[test]
    fn clean_fc_relu_net_is_fully_verifiable_with_zero_unsupported() {
        let net = fc_relu_fc_net();
        let coverage = model_op_coverage(&net);

        assert_eq!(coverage.total(), 3, "Linear + ReLU + Linear = 3 nodes");
        assert_eq!(coverage.exact, 2, "two Linear layers are Exact");
        assert_eq!(coverage.sound, 1, "ReLU is Sound");
        assert_eq!(coverage.loose, 0);
        assert_eq!(coverage.unsupported, 0, "no silent soundness holes");
        assert!(coverage.unsupported_layers.is_empty());
        assert!(coverage.is_fully_verifiable());
        assert!(is_fully_verifiable(&net));
    }

    #[test]
    fn unsupported_op_is_reported_as_a_hole() {
        // Start from the clean net, then append a data-dependent op (Where) that
        // ny cannot soundly verify, to confirm it is surfaced as a hole.
        let mut net = fc_relu_fc_net();
        net.add_node(GraphNode::new(
            "where_op",
            Layer::Where(WhereLayer::default()),
            vec!["lin2".to_string(), "lin2".to_string(), "lin2".to_string()],
        ));

        let coverage = model_op_coverage(&net);
        assert_eq!(coverage.unsupported, 1);
        assert_eq!(coverage.unsupported_layers, vec!["Where".to_string()]);
        assert!(!coverage.is_fully_verifiable());
        assert!(!is_fully_verifiable(&net));
        // The clean ops are still tallied alongside the hole.
        assert_eq!(coverage.exact, 2);
        assert_eq!(coverage.sound, 1);
    }

    #[test]
    fn empty_network_is_trivially_verifiable() {
        let net = GraphNetwork::new();
        let coverage = model_op_coverage(&net);
        assert_eq!(coverage.total(), 0);
        assert!(coverage.is_fully_verifiable());
        // Sanity: NETWORK_INPUT sentinel is not a node and is never classified.
        assert!(net.node(NETWORK_INPUT).is_none());
    }
}
