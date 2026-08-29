// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Symmetry (invariance) verification adapter — consumes `ny_api::symmetry`.
//!
//! NN already builds a [`ny_propagate::GraphNetwork`] from models (see
//! [`crate::graph_tensor`] and [`crate::trace_to_graph`]). This module wires
//! that graph into ny's symmetry checker, which reduces an invariance property
//! to a **difference-network equivalence query**: the transform `T` is
//! prepended as an exact 0/±1 `Linear` wiring layer, the difference network
//! `h(x) = f(Tx) − f(x)` is built, and `|h| ≤ ε` is proved over the input box.
//! Two property forms are supported:
//!
//! - **Permutation invariance** — `|f(Px) − f(x)| ≤ ε` for a fixed permutation
//!   `P` of the flat input coordinates. [`block_permutation`] lifts a
//!   permutation of *points* to the flat coordinate permutation for
//!   point-cloud inputs.
//! - **Finite-rotation invariance** — `|f((Iₙ⊗R)x) − f(x)| ≤ ε` for each
//!   rotation `R` in a fixed finite set, applied blockwise per point.
//!
//! **v1 rotation scope — stated honestly:** only rotations expressible as
//! **signed permutation matrices** (entries in {0, ±1}, exactly one nonzero
//! per row and column, determinant +1 — e.g. the 24 axis-aligned 90° rotations
//! of [`octahedral_rotations`]) are accepted, so the wiring layer and the
//! input-box invariance check are float-exact. Continuous SO(3) invariance is
//! **out of scope for v1**: it is not expressible as a finite wiring, and
//! certifying it would need a quantifier over the rotation manifold.
//! Reflections (determinant −1) and general rotation matrices are rejected up
//! front, as are input boxes that are not setwise invariant under the
//! transform (the property would leave the verified region).
//!
//! Verdicts are sound in both directions, guaranteed by ny:
//! [`SymmetryOutcome::Verified`] is reported only from sound
//! bound-propagation provenance (CROWN by default — plain IBP decorrelates the
//! two copies of the shared input and cannot prove invariance even for exactly
//! symmetric networks), and [`SymmetryOutcome::Falsified`] only when a sound
//! zero-width IBP enclosure at a concrete witness point **certainly** violates
//! the tolerance, carrying that witness. Everything else is
//! [`SymmetryOutcome::Unknown`].
//!
//! This adapter is additive — it does not touch NN's existing verify path. It
//! exposes thin NN-facing wrappers ([`verify_model_permutation_invariant`],
//! [`verify_model_rotation_invariant`], and their `_with_config` variants)
//! plus flat-string reporting helpers ([`symmetry_summary`],
//! [`rotation_summary`]), and re-exports the upstream outcome/option types;
//! callers that want the rich verdict match on
//! [`ny_api::symmetry::SymmetryOutcome`] directly.
//!
//! ```rust,no_run
//! use nn_verify::ny_symmetry::{symmetry_summary, verify_model_permutation_invariant};
//! # fn run(net: &ny_propagate::GraphNetwork) -> ny_core::Result<()> {
//! // Is the model indifferent to swapping its two (flat) input coordinates?
//! let bounds = vec![ny_core::Bound::new(0.0, 1.0); 2];
//! let outcome = verify_model_permutation_invariant(net, &[1, 0], &bounds, 1e-4)?;
//! println!("{}", symmetry_summary(&outcome));
//! # Ok(())
//! # }
//! ```

pub use ny_api::symmetry::{
    block_permutation, octahedral_rotations, FiniteRotationOutcome, SymmetryOptions,
    SymmetryOutcome,
};

use ny_api::symmetry::{
    verify_permutation_invariance_with, verify_rotation_invariance_finite_with,
};

/// Default symmetry-verification options for NN model checks.
///
/// Delegates to [`SymmetryOptions::default`]: CROWN bound propagation (plain
/// IBP decorrelates the two copies of the shared input and cannot prove
/// invariance even for exactly symmetric networks) and a 5-points-per-dimension
/// witness grid for sound falsification. Exposed as a named helper so NN call
/// sites document their intent rather than reaching for `Default` inline.
#[must_use]
pub fn default_symmetry_options() -> SymmetryOptions {
    SymmetryOptions::default()
}

/// Verify that a model graph is invariant (up to `epsilon`) under a fixed
/// permutation of its flat input coordinates, over the box `input_bounds`,
/// using [`default_symmetry_options`].
///
/// `permutation[j] = i` means transformed coordinate `j` reads original
/// coordinate `i`, i.e. the property checked is
/// `∀x ∈ box: |f(x ∘ permutation) − f(x)| ≤ ε`, element-wise on the outputs.
/// For point-cloud inputs stored as consecutive point blocks, build the flat
/// permutation with [`block_permutation`]. For control over the propagation
/// method or the witness grid, use
/// [`verify_model_permutation_invariant_with_config`].
///
/// # Errors
///
/// Rejects an invalid permutation, a permutation/box dimension mismatch, an
/// `epsilon` that is not strictly positive after sound `f32` rounding, a box
/// that is not setwise invariant under the permutation, and dimensions that
/// exceed ny's dense-wiring resource budget; propagates any
/// [`ny_core::NyError`] raised while bounding the difference network.
pub fn verify_model_permutation_invariant(
    net: &ny_propagate::GraphNetwork,
    permutation: &[usize],
    input_bounds: &[ny_core::Bound],
    epsilon: f64,
) -> ny_core::Result<SymmetryOutcome> {
    verify_permutation_invariance_with(
        net,
        permutation,
        input_bounds,
        epsilon,
        &default_symmetry_options(),
    )
}

/// [`verify_model_permutation_invariant`] with explicit [`SymmetryOptions`].
///
/// See [`verify_model_permutation_invariant`] for the default-options
/// convenience wrapper and the property semantics.
///
/// # Errors
///
/// Same contract as [`verify_model_permutation_invariant`].
pub fn verify_model_permutation_invariant_with_config(
    net: &ny_propagate::GraphNetwork,
    permutation: &[usize],
    input_bounds: &[ny_core::Bound],
    epsilon: f64,
    options: &SymmetryOptions,
) -> ny_core::Result<SymmetryOutcome> {
    verify_permutation_invariance_with(net, permutation, input_bounds, epsilon, options)
}

/// Verify that a model graph is invariant (up to `epsilon`) under each
/// rotation in a fixed finite set, applied blockwise to consecutive point
/// blocks of the input, over the box `input_bounds`, using
/// [`default_symmetry_options`].
///
/// Each rotation must be a **signed permutation matrix** (entries in {0, ±1},
/// exactly one nonzero per row and column) with determinant +1 — e.g. any
/// subset of [`octahedral_rotations`]. Continuous SO(3) rotations are out of
/// scope for v1; see the module docs. For control over the propagation method
/// or the witness grid, use [`verify_model_rotation_invariant_with_config`].
///
/// # Errors
///
/// Rejects an empty rotation set, non-signed-permutation or reflection
/// (determinant −1) matrices, an input dimension that is not a positive
/// multiple of the rotation dimension, an `epsilon` that is not strictly
/// positive after sound `f32` rounding, and boxes not setwise invariant under
/// a rotation; propagates any [`ny_core::NyError`] raised while bounding a
/// difference network.
pub fn verify_model_rotation_invariant(
    net: &ny_propagate::GraphNetwork,
    rotations: &[ndarray::Array2<f32>],
    input_bounds: &[ny_core::Bound],
    epsilon: f64,
) -> ny_core::Result<FiniteRotationOutcome> {
    verify_rotation_invariance_finite_with(
        net,
        rotations,
        input_bounds,
        epsilon,
        &default_symmetry_options(),
    )
}

/// [`verify_model_rotation_invariant`] with explicit [`SymmetryOptions`].
///
/// See [`verify_model_rotation_invariant`] for the default-options convenience
/// wrapper and the v1 signed-permutation rotation scope.
///
/// # Errors
///
/// Same contract as [`verify_model_rotation_invariant`].
pub fn verify_model_rotation_invariant_with_config(
    net: &ny_propagate::GraphNetwork,
    rotations: &[ndarray::Array2<f32>],
    input_bounds: &[ny_core::Bound],
    epsilon: f64,
    options: &SymmetryOptions,
) -> ny_core::Result<FiniteRotationOutcome> {
    verify_rotation_invariance_finite_with(net, rotations, input_bounds, epsilon, options)
}

/// Render an NN-friendly one-line summary of a single symmetry query.
///
/// Reports the verdict plus the tightest available envelope on
/// `f(Tx) − f(x)`, or the concrete witness for a falsified property. This is
/// intentionally a flat string for NN's reporting style; callers needing
/// structured data match on [`SymmetryOutcome`] directly.
#[must_use]
pub fn symmetry_summary(outcome: &SymmetryOutcome) -> String {
    match outcome {
        SymmetryOutcome::Verified { difference_bounds } => {
            format!(
                "symmetry: verified ({})",
                difference_phrase(difference_bounds)
            )
        }
        SymmetryOutcome::Falsified {
            witness,
            difference,
        } => format!(
            "symmetry: falsified at witness {witness:?} (difference enclosed in [{}, {}])",
            difference.lower(),
            difference.upper()
        ),
        SymmetryOutcome::Unknown { difference_bounds } => {
            format!(
                "symmetry: unknown ({})",
                difference_phrase(difference_bounds)
            )
        }
        // `SymmetryOutcome` is #[non_exhaustive] upstream.
        _ => "symmetry: unrecognized outcome".to_string(),
    }
}

/// Render an NN-friendly one-line summary of a finite-rotation run.
///
/// Reports how many rotations verified out of the total, the falsified /
/// unknown split when not everything verified, and the index of the first
/// rotation with a concrete counterexample. Callers needing structured data
/// read [`FiniteRotationOutcome::per_rotation`] directly.
#[must_use]
pub fn rotation_summary(outcome: &FiniteRotationOutcome) -> String {
    let total = outcome.per_rotation.len();
    let verified = outcome
        .per_rotation
        .iter()
        .filter(|rotation| rotation.is_verified())
        .count();
    let falsified = outcome
        .per_rotation
        .iter()
        .filter(|rotation| matches!(rotation, SymmetryOutcome::Falsified { .. }))
        .count();
    let unknown = total - verified - falsified;
    let mut summary = format!("rotations: {verified}/{total} verified");
    if falsified > 0 || unknown > 0 {
        summary.push_str(&format!(" ({falsified} falsified, {unknown} unknown)"));
    }
    if let Some((index, _)) = outcome.first_falsified() {
        summary.push_str(&format!("; first counterexample at rotation {index}"));
    }
    summary
}

/// Envelope phrase over per-output difference bounds: output count plus the
/// widest `[min lower, max upper]` enclosure across outputs.
fn difference_phrase(bounds: &[ny_core::Bound]) -> String {
    let Some(first) = bounds.first() else {
        return "no output bounds available".to_string();
    };
    let mut lo = first.lower();
    let mut hi = first.upper();
    for bound in &bounds[1..] {
        lo = lo.min(bound.lower());
        hi = hi.max(bound.upper());
    }
    format!(
        "{} output{}, difference within [{lo}, {hi}]",
        bounds.len(),
        if bounds.len() == 1 { "" } else { "s" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array1, Array2};
    use ny_core::Bound;
    use ny_propagate::layers::{Layer, LinearLayer, ReLULayer};
    use ny_propagate::{GraphNetwork, GraphNode};

    /// Tiny Linear → ReLU → Linear graph for end-to-end symmetry exercise.
    ///
    /// Mirrors the construction NN uses elsewhere (e.g. the `ny_ladder`
    /// tests), keeping the network small and deterministic. Test weights keep
    /// every ReLU stably active on the verified box, so the difference network
    /// cancels exactly (up to sound directed rounding) and CROWN can prove
    /// genuine invariance rather than a vacuously wide tolerance.
    fn linear_relu_linear(w1: Array2<f32>, b1: Array1<f32>, w2: Array2<f32>) -> GraphNetwork {
        let linear1 = LinearLayer::new(w1, Some(b1)).expect("valid linear1");
        let linear2 = LinearLayer::new(w2, None).expect("valid linear2");
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

    fn unit_box(dim: usize) -> Vec<Bound> {
        vec![Bound::new(0.0, 1.0); dim]
    }

    fn symmetric_box(dim: usize) -> Vec<Bound> {
        vec![Bound::new(-1.0, 1.0); dim]
    }

    #[test]
    fn ny_symmetry_verifies_swap_invariant_linear_relu_linear() {
        // Symmetric weight matrix: both hidden units compute x0 + x1 + 1, so
        // f(Px) = f(x) exactly for the swap P; pre-activations lie in [1, 3]
        // on [0, 1]² (stably active), so CROWN cancels the two branches.
        let graph = linear_relu_linear(
            arr2(&[[1.0_f32, 1.0], [1.0, 1.0]]),
            arr1(&[1.0_f32, 1.0]),
            arr2(&[[1.0_f32, 1.0], [1.0, 1.0]]),
        );

        let outcome = verify_model_permutation_invariant(&graph, &[1, 0], &unit_box(2), 1e-4)
            .expect("query should run on linear→relu→linear graph");

        match &outcome {
            SymmetryOutcome::Verified { difference_bounds } => {
                assert_eq!(difference_bounds.len(), 2, "one bound per model output");
                for bound in difference_bounds {
                    assert!(
                        bound.lower() >= -1e-4 && bound.upper() <= 1e-4,
                        "certified difference must stay within ±ε, got {bound:?}"
                    );
                }
            }
            other => panic!("swap-symmetric net must verify, got {other:?}"),
        }

        let summary = symmetry_summary(&outcome);
        assert!(summary.contains("verified"), "summary: {summary}");
    }

    #[test]
    fn ny_symmetry_falsifies_asymmetric_net_with_concrete_witness() {
        // Asymmetric first layer with a large bias keeps the ReLUs stably
        // active on [0, 1]², so f(Px) − f(x) = 2·x0 − 2·x1 exactly; range
        // [−2, 2] certainly violates ε = 0.5 and the sound witness search
        // must produce a concrete counterexample.
        let graph = linear_relu_linear(
            arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]),
            arr1(&[10.0_f32, 10.0]),
            arr2(&[[1.0_f32, 1.0]]),
        );

        let outcome = verify_model_permutation_invariant_with_config(
            &graph,
            &[1, 0],
            &unit_box(2),
            0.5,
            &default_symmetry_options(),
        )
        .expect("query should run on asymmetric graph");

        match &outcome {
            SymmetryOutcome::Falsified {
                witness,
                difference,
            } => {
                assert_eq!(witness.len(), 2);
                assert!(
                    witness.iter().all(|&x| (0.0..=1.0).contains(&x)),
                    "witness must lie inside the verified box: {witness:?}"
                );
                assert!(
                    difference.lower() > 0.5 || difference.upper() < -0.5,
                    "enclosure must certainly violate ε: {difference:?}"
                );
                // Cross-check against the analytic difference 2·(x0 − x1).
                assert!(
                    (witness[0] - witness[1]).abs() * 2.0 > 0.5,
                    "witness must violate on the original net: {witness:?}"
                );
            }
            other => panic!("asymmetric net must be falsified with a witness, got {other:?}"),
        }

        let summary = symmetry_summary(&outcome);
        assert!(summary.contains("falsified"), "summary: {summary}");
    }

    #[test]
    fn ny_symmetry_verifies_z_axis_rotation_invariance() {
        // f depends only on z (both hidden units read z + 2, stably active on
        // [−1, 1]³), so it is invariant under all four 90° rotations about the
        // z-axis — each a signed permutation matrix with determinant +1.
        let graph = linear_relu_linear(
            arr2(&[[0.0_f32, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            arr1(&[2.0_f32, 2.0]),
            arr2(&[[1.0_f32, 1.0]]),
        );
        let rotations = vec![
            arr2(&[[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), // I
            arr2(&[[0.0_f32, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), // Rz(90°)
            arr2(&[[-1.0_f32, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), // Rz(180°)
            arr2(&[[0.0_f32, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), // Rz(270°)
        ];

        let outcome = verify_model_rotation_invariant(&graph, &rotations, &symmetric_box(3), 1e-4)
            .expect("query should run on z-only graph");

        assert_eq!(outcome.per_rotation.len(), 4);
        assert!(
            outcome.all_verified(),
            "z-only net must verify all four z-rotations: {:?}",
            outcome.per_rotation
        );
        assert!(outcome.first_falsified().is_none());

        let summary = rotation_summary(&outcome);
        assert!(summary.contains("4/4 verified"), "summary: {summary}");
    }

    #[test]
    fn ny_symmetry_falsifies_rotation_sensitive_net() {
        // f = 2·x + 6 (stably active on [−1, 1]³) is NOT invariant under
        // Rz(90°): the difference is −2·(x + y), reaching ±4 on the corners.
        let graph = linear_relu_linear(
            arr2(&[[1.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            arr1(&[3.0_f32, 3.0]),
            arr2(&[[1.0_f32, 1.0]]),
        );
        let rz90 = arr2(&[[0.0_f32, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let options = SymmetryOptions {
            witness_grid: 3,
            ..default_symmetry_options()
        };

        let outcome = verify_model_rotation_invariant_with_config(
            &graph,
            &[rz90],
            &symmetric_box(3),
            0.5,
            &options,
        )
        .expect("query should run on x-reading graph");

        assert!(!outcome.all_verified());
        let (index, falsified) = outcome
            .first_falsified()
            .expect("rotation-sensitive net must produce a concrete counterexample");
        assert_eq!(index, 0);
        match falsified {
            SymmetryOutcome::Falsified {
                witness,
                difference,
            } => {
                assert_eq!(witness.len(), 3);
                assert!(
                    witness.iter().all(|&x| (-1.0..=1.0).contains(&x)),
                    "witness must lie inside the verified box: {witness:?}"
                );
                assert!(
                    difference.lower() > 0.5 || difference.upper() < -0.5,
                    "enclosure must certainly violate ε: {difference:?}"
                );
                // Cross-check against the analytic difference −2·(x + y).
                assert!(
                    (witness[0] + witness[1]).abs() * 2.0 > 0.5,
                    "witness must violate on the original net: {witness:?}"
                );
            }
            other => panic!("expected Falsified, got {other:?}"),
        }

        let summary = rotation_summary(&outcome);
        assert!(summary.contains("0/1 verified"), "summary: {summary}");
        assert!(
            summary.contains("first counterexample at rotation 0"),
            "summary: {summary}"
        );
    }

    #[test]
    fn default_options_match_upstream_defaults() {
        let options = default_symmetry_options();
        assert_eq!(
            options.config.method,
            ny_propagate::PropagationMethod::Crown,
            "IBP decorrelates the shared input; the default must be CROWN"
        );
        assert_eq!(options.witness_grid, 5);
    }

    #[test]
    fn reexported_helpers_stay_wired() {
        // Point-block lifting and the finite rotation set are re-exported so
        // NN call sites need only this module.
        assert_eq!(
            block_permutation(&[1, 0], 3).expect("valid point permutation"),
            vec![3, 4, 5, 0, 1, 2]
        );
        assert_eq!(octahedral_rotations().len(), 24);
    }
}
