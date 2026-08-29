// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Rich output-property verification adapter — consumes `ny_api::annotation`
//! and `ny_api::constraints`.
//!
//! NN's legacy verify path states only per-output box bounds
//! (`y[i] in [l, u]`). This module lets NN state and soundly verify richer
//! output properties:
//!
//! - **halfspace** — `a·y <= b` or `a·y >= b`
//!   ([`AnnotationConstraint::Linear`]);
//! - **argmax margin** — output `class` stays the strict argmax
//!   ([`AnnotationConstraint::ArgmaxMargin`]), e.g. classifier top-1
//!   robustness over an input region.
//!
//! Specs are built through [`SpecBuilder`] (see [`spec_with_linear_constraint`],
//! [`spec_with_argmax_margin`], [`spec_with_constraints`]) and verified through
//! [`ny_api::constraints::verify_with_constraints`], which reduces each rich
//! constraint to a sound `margin > 0` check: an affine margin layer is appended
//! to the network (the difference-network technique) and verification proves the
//! margin's certified lower bound strictly positive. The machinery is
//! fail-closed — unencodable or inconclusive constraints yield `Unknown`, never
//! `Verified` — and a spec carrying no rich constraints falls through to the
//! legacy per-output bounds check unchanged.
//!
//! This adapter is additive — it does not touch NN's existing verify path. It
//! exposes thin NN-facing wrappers ([`verify_constrained`],
//! [`verify_constrained_with_config`]), spec-building helpers, an
//! [`AnnotatedSpec`] metadata helper ([`annotate`]), and a small reporting
//! helper ([`constrained_summary`]); callers that want the raw margin network
//! use [`ny_api::constraints::augment_for_constraint`] directly, and callers
//! needing input shapes or timeouts reach for [`SpecBuilder`] itself.
//!
//! ```rust,no_run
//! use nn_verify::ny_constraints::{
//!     constrained_summary, spec_with_argmax_margin, verify_constrained,
//! };
//! # fn run(net: &ny_propagate::GraphNetwork) -> ny_core::Result<()> {
//! // Classifier top-1 robustness: output 0 must stay the strict argmax.
//! let spec = spec_with_argmax_margin(
//!     vec![ny_core::Bound::new(-1.0, 1.0), ny_core::Bound::new(-1.0, 1.0)],
//!     0,
//! )?;
//! let result = verify_constrained(net, &spec)?;
//! println!("{}", constrained_summary(&spec, &result));
//! # Ok(())
//! # }
//! ```

use ny_api::annotation::{AnnotatedSpec, AnnotationConstraint, SpecBuilder};
use ny_api::constraints::verify_with_constraints;
use ny_api::verify::{PropagationConfig, Verifier};
use ny_core::{
    Bound, ConstraintKind, OutputConstraint, UnknownReason, VerificationResult, VerificationSpec,
};

/// Default propagation configuration for constrained verification.
///
/// Delegates to [`PropagationConfig::default`] (α-CROWN with its stock
/// iteration/tolerance settings), a sound linear-relaxation method tight enough
/// to prove most margin properties without branch and bound. Exposed as a named
/// helper so NN call sites document their intent rather than reaching for
/// `Default` inline.
#[must_use]
pub fn default_constraints_config() -> PropagationConfig {
    PropagationConfig::default()
}

/// Trivial (fully open) output bounds for specs whose property is carried
/// entirely by rich [`OutputConstraint`]s.
///
/// [`SpecBuilder::build`] requires non-empty output bounds, but constrained
/// verification never consults them; a single unbounded interval satisfies the
/// builder without stating a box property.
fn unconstrained_output_bounds() -> Vec<Bound> {
    vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)]
}

/// Build a [`VerificationSpec`] carrying a single halfspace property
/// `sum_i coeffs[i]*y[i] {<=|>=} bias` over the given input box.
///
/// Builds through [`SpecBuilder`]; the spec's `output_bounds` are left trivial
/// (fully open) so the halfspace is the only property verified. Combine several
/// constraints with [`spec_with_constraints`].
///
/// # Errors
///
/// Returns [`ny_core::NyError::InvalidSpec`] if `input_bounds` is empty or
/// `coeffs` is empty/non-finite (validated by [`SpecBuilder::build`]).
pub fn spec_with_linear_constraint(
    input_bounds: Vec<Bound>,
    coeffs: Vec<f32>,
    bias: f32,
    kind: ConstraintKind,
) -> ny_core::Result<VerificationSpec> {
    spec_with_constraints(
        input_bounds,
        vec![AnnotationConstraint::Linear { coeffs, bias, kind }],
    )
}

/// Build a [`VerificationSpec`] carrying a single argmax-margin property —
/// output `class` must remain the strict argmax (`y[class] > y[j]` for every
/// `j != class`) — over the given input box.
///
/// Builds through [`SpecBuilder`]; the spec's `output_bounds` are left trivial
/// (fully open) so the robustness property is the only one verified. An
/// out-of-range `class` is not rejected here (the output width is a property of
/// the network, not the spec); [`verify_constrained`] fails closed to `Unknown`
/// for it.
///
/// # Errors
///
/// Returns [`ny_core::NyError::InvalidSpec`] if `input_bounds` is empty
/// (validated by [`SpecBuilder::build`]).
pub fn spec_with_argmax_margin(
    input_bounds: Vec<Bound>,
    class: usize,
) -> ny_core::Result<VerificationSpec> {
    spec_with_constraints(
        input_bounds,
        vec![AnnotationConstraint::ArgmaxMargin { class }],
    )
}

/// Build a [`VerificationSpec`] carrying an arbitrary conjunction of
/// [`AnnotationConstraint`]s over the given input box.
///
/// Constraints are applied to a [`SpecBuilder`] in order:
/// [`AnnotationConstraint::Bounds`] replaces the (initially trivial)
/// `output_bounds`, while `Linear` / `ArgmaxMargin` accumulate as rich
/// [`OutputConstraint`]s. Verification treats the constraint list
/// conjunctively — every constraint must hold.
///
/// # Errors
///
/// Returns [`ny_core::NyError::InvalidSpec`] if `input_bounds` is empty, if a
/// `Bounds` constraint supplied empty bounds, or if any constraint is
/// structurally invalid (validated by [`SpecBuilder::build`]).
pub fn spec_with_constraints(
    input_bounds: Vec<Bound>,
    constraints: Vec<AnnotationConstraint>,
) -> ny_core::Result<VerificationSpec> {
    constraints
        .into_iter()
        .fold(
            SpecBuilder::default()
                .input_bounds(input_bounds, None)
                .output_bounds(unconstrained_output_bounds()),
            SpecBuilder::output_constraint,
        )
        .build()
}

/// Attach NN-facing metadata (a name and tags) to a spec as an
/// [`AnnotatedSpec`].
///
/// Metadata never alters verifier behavior; it exists for NN's grouping and
/// reporting. Pass `annotated.spec` (or [`AnnotatedSpec::into_spec`]) to the
/// verify entry points.
#[must_use]
pub fn annotate(
    spec: VerificationSpec,
    name: impl Into<String>,
    tags: Vec<String>,
) -> AnnotatedSpec {
    let mut annotated = AnnotatedSpec::new(spec);
    annotated.name = Some(name.into());
    annotated.tags = tags;
    annotated
}

/// Verify a model graph against every rich [`OutputConstraint`] carried by
/// `spec`, using a sensible default configuration.
///
/// Each halfspace / argmax-margin constraint is reduced to a sound
/// `margin > 0` check via an appended affine margin layer; the conjunction is
/// `Verified` only if every constraint individually verifies, and unencodable
/// or inconclusive constraints fail closed to `Unknown`. A spec with no rich
/// constraints is verified through the legacy per-output bounds path,
/// preserving existing behavior exactly. For control over the propagation
/// method, use [`verify_constrained_with_config`].
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised by the underlying verifier (e.g.
/// a structural mismatch between the graph and the spec). Inconclusive or
/// unencodable constraints do not error; they downgrade the verdict to
/// `Unknown`.
pub fn verify_constrained(
    net: &ny_propagate::GraphNetwork,
    spec: &VerificationSpec,
) -> ny_core::Result<VerificationResult> {
    verify_constrained_with_config(net, spec, &default_constraints_config())
}

/// Verify a model graph against every rich [`OutputConstraint`] carried by
/// `spec` with an explicit [`PropagationConfig`].
///
/// See [`verify_constrained`] for the default-config convenience wrapper and
/// the margin-reduction semantics.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised by the underlying verifier.
pub fn verify_constrained_with_config(
    net: &ny_propagate::GraphNetwork,
    spec: &VerificationSpec,
    cfg: &PropagationConfig,
) -> ny_core::Result<VerificationResult> {
    let verifier = Verifier::new(cfg.clone());
    verify_with_constraints(&verifier, net, spec)
}

/// Render an NN-friendly one-line summary of a constrained verification run.
///
/// Reports the verdict (with the reason when inconclusive) and the constraints
/// the spec carried. This is intentionally a flat string for NN's reporting
/// style; callers needing structured data read the [`VerificationResult`]
/// variants directly.
#[must_use]
pub fn constrained_summary(spec: &VerificationSpec, result: &VerificationResult) -> String {
    let verdict = match result {
        VerificationResult::Verified { .. } => "verified".to_string(),
        VerificationResult::Violated { counterexample, .. } => {
            format!(
                "violated (counterexample over {} input{})",
                counterexample.len(),
                if counterexample.len() == 1 { "" } else { "s" },
            )
        }
        VerificationResult::Unknown { reason, .. } => {
            format!("unknown ({})", unknown_label(reason))
        }
        VerificationResult::Timeout { .. } => "timeout".to_string(),
    };
    let constraints = spec.output_constraints();
    if constraints.is_empty() {
        return format!("constraints: {verdict} (legacy output-bounds path, no rich constraints)");
    }
    let labels: Vec<String> = constraints.iter().map(constraint_label).collect();
    format!(
        "constraints: {verdict} ({} constraint{}: {})",
        constraints.len(),
        if constraints.len() == 1 { "" } else { "s" },
        labels.join(", "),
    )
}

/// Compact label for one [`OutputConstraint`] in a summary line.
fn constraint_label(constraint: &OutputConstraint) -> String {
    match constraint {
        OutputConstraint::Bounds(bounds) => format!("bounds({})", bounds.len()),
        OutputConstraint::Linear {
            coeffs,
            kind: ConstraintKind::Le,
            ..
        } => format!("a·y<=b({})", coeffs.len()),
        OutputConstraint::Linear {
            coeffs,
            kind: ConstraintKind::Ge,
            ..
        } => format!("a·y>=b({})", coeffs.len()),
        OutputConstraint::ArgmaxMargin { class } => format!("argmax={class}"),
    }
}

/// Compact label for the reason a run stayed [`VerificationResult::Unknown`].
fn unknown_label(reason: &UnknownReason) -> String {
    match reason {
        UnknownReason::BoundsTooLoose { gap: Some(gap) } => {
            format!("bounds too loose, margin short by {gap}")
        }
        UnknownReason::BoundsTooLoose { gap: None } => "bounds too loose".to_string(),
        UnknownReason::Other { message } => message.clone(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_api::verify::PropagationMethod;
    use ny_core::NyError;
    use ny_propagate::layers::{Layer, LinearLayer, ReLULayer};
    use ny_propagate::{GraphNetwork, GraphNode};

    /// Tiny Linear → ReLU → Linear graph (2→2→2) with a caller-chosen output
    /// layer, for end-to-end constrained verification.
    ///
    /// Mirrors the construction in `ny_ladder`'s tests: the first layer computes
    /// `pre = (x1 - x2, x2 - x1)`, so after ReLU `h in [0, 2]^2` over the unit
    /// input box (IBP), and `h1 + h2 = |x1 - x2| in [0, 2]` exactly.
    fn linear_relu_linear_graph(w2: Vec<f32>, b2: Vec<f32>) -> GraphNetwork {
        let w1 =
            ndarray::Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).expect("2x2 w1");
        let linear1 = LinearLayer::new(w1, Some(ndarray::Array1::zeros(2))).expect("valid linear1");
        let w2 = ndarray::Array2::from_shape_vec((2, 2), w2).expect("2x2 w2");
        let linear2 =
            LinearLayer::new(w2, Some(ndarray::Array1::from_vec(b2))).expect("valid linear2");

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

    /// Symmetric output layer: `y1 = y2 = h1 + h2` (both outputs identical).
    fn symmetric_graph() -> GraphNetwork {
        linear_relu_linear_graph(vec![1.0, 1.0, 1.0, 1.0], vec![0.0, 0.0])
    }

    fn unit_box_input() -> Vec<Bound> {
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)]
    }

    fn ibp_config() -> PropagationConfig {
        PropagationConfig {
            method: PropagationMethod::Ibp,
            ..PropagationConfig::default()
        }
    }

    #[test]
    fn ny_constraints_prove_halfspace_property() {
        // y1 = h1 + h2, IBP-bounded in [0, 4]. Halfspace y1 <= 5 has margin
        // 5 - y1 with certified lower bound >= 1 > 0: provable.
        let graph = symmetric_graph();
        let spec =
            spec_with_linear_constraint(unit_box_input(), vec![1.0, 0.0], 5.0, ConstraintKind::Le)
                .expect("valid halfspace spec");

        let result = verify_constrained(&graph, &spec)
            .expect("constrained verify should run on linear→relu→linear graph");
        assert!(
            result.is_verified(),
            "y1 <= 5 must verify over the unit box, got {result:?}"
        );

        let summary = constrained_summary(&spec, &result);
        assert!(summary.contains("verified"), "summary: {summary}");
        assert!(summary.contains("a·y<=b"), "summary: {summary}");
    }

    #[test]
    fn ny_constraints_reject_unprovable_halfspace() {
        // y1 = |x1 - x2| reaches 2 at x = (1, -1), so y1 <= 1 is genuinely
        // violated: the margin 1 - y1 must NOT be certified strictly positive.
        let graph = symmetric_graph();
        let spec =
            spec_with_linear_constraint(unit_box_input(), vec![1.0, 0.0], 1.0, ConstraintKind::Le)
                .expect("valid halfspace spec");

        let result = verify_constrained_with_config(&graph, &spec, &ibp_config())
            .expect("constrained verify should run");
        assert!(
            !result.is_verified(),
            "a violated halfspace must never verify, got {result:?}"
        );

        let summary = constrained_summary(&spec, &result);
        assert!(summary.starts_with("constraints:"), "summary: {summary}");
        assert!(!summary.contains("verified"), "summary: {summary}");
    }

    #[test]
    fn ny_constraints_prove_argmax_margin() {
        // y1 = h1 + h2 + 1 in [1, 5] (IBP), y2 = 0: class 0 strictly dominates,
        // so the appended margin y1 - y2 has certified lower bound 1 > 0.
        let graph = linear_relu_linear_graph(vec![1.0, 1.0, 0.0, 0.0], vec![1.0, 0.0]);
        let spec = spec_with_argmax_margin(unit_box_input(), 0).expect("valid argmax-margin spec");

        let result = verify_constrained_with_config(&graph, &spec, &ibp_config())
            .expect("constrained verify should run");
        assert!(
            result.is_verified(),
            "dominating class 0 must verify, got {result:?}"
        );

        let summary = constrained_summary(&spec, &result);
        assert!(summary.contains("verified"), "summary: {summary}");
        assert!(summary.contains("argmax=0"), "summary: {summary}");
    }

    #[test]
    fn ny_constraints_reject_argmax_margin_tie() {
        // y1 ≡ y2 on the symmetric graph: the argmax is never strict, so the
        // margin y1 - y2 ≡ 0 can never be proven strictly positive by any sound
        // method. Fail-closed: Unknown, never Verified (and never Violated —
        // margin 0 satisfies the non-strict >= 0 check).
        let graph = symmetric_graph();
        let spec = spec_with_argmax_margin(unit_box_input(), 0).expect("valid argmax-margin spec");

        let result = verify_constrained_with_config(&graph, &spec, &ibp_config())
            .expect("constrained verify should run");
        assert!(
            matches!(result, VerificationResult::Unknown { .. }),
            "a tied argmax must fail closed to Unknown, got {result:?}"
        );

        let summary = constrained_summary(&spec, &result);
        assert!(summary.contains("unknown"), "summary: {summary}");
    }

    #[test]
    fn ny_constraints_conjunction_all_must_hold() {
        // Two provable halfspaces: y1 <= 5 (margin >= 1) and y2 >= -5
        // (margin >= 5). The conjunction verifies only because both do.
        let graph = symmetric_graph();
        let spec = spec_with_constraints(
            unit_box_input(),
            vec![
                AnnotationConstraint::Linear {
                    coeffs: vec![1.0, 0.0],
                    bias: 5.0,
                    kind: ConstraintKind::Le,
                },
                AnnotationConstraint::Linear {
                    coeffs: vec![0.0, 1.0],
                    bias: -5.0,
                    kind: ConstraintKind::Ge,
                },
            ],
        )
        .expect("valid conjunction spec");
        assert_eq!(spec.output_constraints().len(), 2);

        let result = verify_constrained_with_config(&graph, &spec, &ibp_config())
            .expect("constrained verify should run");
        assert!(result.is_verified(), "both halfspaces hold, got {result:?}");

        let summary = constrained_summary(&spec, &result);
        assert!(summary.contains("2 constraints"), "summary: {summary}");
    }

    #[test]
    fn spec_helpers_build_through_spec_builder() {
        let spec =
            spec_with_linear_constraint(unit_box_input(), vec![1.0, -1.0], 0.5, ConstraintKind::Le)
                .expect("valid spec");
        assert_eq!(
            spec.output_constraints(),
            &[OutputConstraint::Linear {
                coeffs: vec![1.0, -1.0],
                bias: 0.5,
                kind: ConstraintKind::Le,
            }]
        );
        // output_bounds stays a single trivial interval; the halfspace is the
        // only stated property.
        assert_eq!(spec.output_bounds().len(), 1);

        let spec = spec_with_argmax_margin(unit_box_input(), 1).expect("valid spec");
        assert_eq!(
            spec.output_constraints(),
            &[OutputConstraint::ArgmaxMargin { class: 1 }]
        );
    }

    #[test]
    fn spec_helpers_reject_invalid_inputs() {
        let err = spec_with_argmax_margin(Vec::new(), 0)
            .expect_err("empty input bounds must be rejected");
        assert!(matches!(err, NyError::InvalidSpec(message) if message.contains("input bounds")));

        let err =
            spec_with_linear_constraint(unit_box_input(), Vec::new(), 0.0, ConstraintKind::Le)
                .expect_err("empty coeffs must be rejected");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    #[test]
    fn annotate_sets_metadata_without_touching_spec() {
        let spec = spec_with_argmax_margin(unit_box_input(), 0).expect("valid spec");
        let annotated = annotate(spec, "top1-robust", vec!["robustness".to_string()]);
        assert_eq!(annotated.name.as_deref(), Some("top1-robust"));
        assert_eq!(annotated.tags, vec!["robustness".to_string()]);
        assert_eq!(
            annotated.into_spec().output_constraints(),
            &[OutputConstraint::ArgmaxMargin { class: 0 }]
        );
    }

    #[test]
    fn default_config_matches_propagation_default() {
        let cfg = default_constraints_config();
        assert_eq!(cfg.method, PropagationMethod::AlphaCrown);
        assert!(!cfg.use_gpu);
    }
}
