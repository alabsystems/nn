// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deployed-precision verification (Metal f16) via `ny_api::precision`.
//!
//! NY proves bounds under an f32 *idealization*. NN's models do NOT run in f32 on
//! the GPU — the Metal backend executes in f16 — so a bound proven in f32 is not
//! automatically valid for the bits that actually run on-device. This adapter
//! asks for the bounds at the precision NN actually deploys.
//!
//! It delegates to [`ny_api::precision::verify_with_sound_precision`], the
//! fail-closed sound entry point. **Every non-F32 policy is currently REFUSED
//! upstream with [`ny_core::NyError::UnsupportedConfiguration`]** (ny hardening
//! 25dee0c5e): the experimental widening path behind the facade does not yet
//! model every deployed store and reduction rigorously, and it must never emit
//! `Verified` with sound provenance. The refusal IS the sound posture — no
//! false verification is possible through this adapter. NY's representation-only
//! *heuristic* precision API still exists upstream; this adapter deliberately
//! does not expose it, because its verdicts carry no soundness claim.
//!
//! # Soundness direction
//!
//! While the non-F32 path is refused, the fail-closed direction is trivial:
//! `verify_metal_f16` can only error, never falsely report `Verified`. If the
//! sound widening path lands upstream, the intended monotonicity (a verdict can
//! only flip `Verified` -> `Unknown` when moving f32 -> f16, never the reverse)
//! applies, and the refusal tests below will fail loudly so the un-refusal is a
//! conscious event.
//!
//! # `F32` is a strict no-op
//!
//! Passing [`FloatPrecision::F32`] yields a policy
//! (`{ compute: F32, accumulate: F32 }`) that `ny` treats as the idealized case:
//! it returns exactly the normal f32 verdict (same engine, same provenance), so
//! [`verify_deployed`] with `F32` equals the plain verifier.
//!
//! # ADDITIVE
//!
//! This module only consumes the `ny-api` facade; it does not touch NN's existing
//! verify path or types.

use ny_core::{FloatPrecision, Result, VerificationResult, VerificationSpec};
use ny_propagate::GraphNetwork;

use ny_api::precision::MixedPrecisionPolicy;

/// Verify `net` against `spec` at NN's deployed compute precision, using a uniform
/// policy where both the per-element compute and the reduction accumulate run at
/// `precision`.
///
/// This is the fail-closed sound entry point: it never falsely reports
/// `Verified` for the deployed precision. Currently (ny hardening 25dee0c5e)
/// every non-F32 `precision` is refused upstream with
/// [`ny_core::NyError::UnsupportedConfiguration`] rather than verified.
///
/// [`FloatPrecision::F32`] is a strict no-op: the result is identical to the plain
/// verifier's f32 verdict. Use [`verify_metal_f16`] for NN's actual deployment
/// target.
///
/// # Errors
/// Returns [`ny_core::NyError::UnsupportedConfiguration`] for every non-F32
/// `precision` (the sound widening path is not yet implemented upstream).
/// For F32, propagates any error from the plain verifier
/// (layer/propagation errors and shape errors from the underlying IBP).
pub fn verify_deployed(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    precision: FloatPrecision,
) -> Result<VerificationResult> {
    let policy = MixedPrecisionPolicy::new(precision, precision);
    verify_deployed_with_policy(net, spec, &policy)
}

/// Verify `net` against `spec` under an explicit mixed-precision `policy`
/// (separate compute and accumulate precisions).
///
/// This is the general form behind [`verify_deployed`]; use it when the deployed
/// hardware multiplies in one precision but accumulates in another (e.g. f16
/// multiply with f32 accumulate). The same fail-closed posture applies: the
/// idealized all-f32 policy is a strict no-op equal to the plain verifier, and
/// every non-F32 policy is currently refused upstream (ny 25dee0c5e).
///
/// # Errors
/// Returns [`ny_core::NyError::UnsupportedConfiguration`] for every non-F32
/// policy; otherwise propagates any error from
/// [`ny_api::precision::verify_with_sound_precision`].
pub fn verify_deployed_with_policy(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    policy: &MixedPrecisionPolicy,
) -> Result<VerificationResult> {
    ny_api::precision::verify_with_sound_precision(net, spec, policy)
}

/// Verify `net` against `spec` at NN's Metal deployment precision (f16).
///
/// Convenience wrapper for `verify_deployed(net, spec, FloatPrecision::F16)`. NN's
/// Metal backend runs in f16, so a sound verdict here would have to hold for the
/// bits that actually execute on-device rather than the f32 idealization.
/// Currently (ny 25dee0c5e) that sound f16 path is refused upstream, so this
/// always errors — fail-closed: it can never falsely report `Verified` for the
/// deployed f16 compute.
///
/// # Errors
/// Always returns [`ny_core::NyError::UnsupportedConfiguration`] while the sound
/// non-F32 path is refused upstream; see [`verify_deployed`].
pub fn verify_metal_f16(net: &GraphNetwork, spec: &VerificationSpec) -> Result<VerificationResult> {
    verify_deployed(net, spec, FloatPrecision::F16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::Bound;
    use ny_propagate::layers::{LinearLayer, ReLULayer};
    use ny_propagate::{GraphNode, Layer};

    /// Build a tiny FC -> ReLU -> FC `GraphNetwork` for end-to-end exercise.
    ///
    /// Topology (single-element input/output):
    ///   input --(Linear w=2, b=0)--> fc1 --ReLU--> relu --(Linear w=1, b=0)--> fc2
    /// On `[lo, hi]` with `lo >= 0` the output bounds are `[2*lo, 2*hi]`.
    fn tiny_fc_relu_net() -> GraphNetwork {
        let mut graph = GraphNetwork::new();

        // fc1: 1x1 weight [[2.0]], no bias — consumes the network input.
        let w1 = ndarray::Array2::<f32>::from_elem((1, 1), 2.0);
        graph.add_node(GraphNode::from_input(
            "fc1",
            Layer::Linear(LinearLayer::new(w1, None).unwrap()),
        ));

        // relu: depends on fc1.
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["fc1".to_string()],
        ));

        // fc2: 1x1 weight [[1.0]], no bias — depends on relu.
        let w2 = ndarray::Array2::<f32>::from_elem((1, 1), 1.0);
        graph.add_node(GraphNode::new(
            "fc2",
            Layer::Linear(LinearLayer::new(w2, None).unwrap()),
            vec!["relu".to_string()],
        ));

        graph.set_output("fc2");
        graph
    }

    /// A spec over a single non-negative input element with a generous required
    /// output band (so the f32 verdict is comfortably `Verified`).
    fn spec_generous() -> VerificationSpec {
        // input in [1, 2] -> idealized output in [2, 4]; require [-100, 100].
        VerificationSpec::new(vec![Bound::new(1.0, 2.0)], vec![Bound::new(-100.0, 100.0)])
            .unwrap()
            .with_input_shape(vec![1])
            .unwrap()
    }

    /// The plain (f32) graph verdict — the reference `verify_deployed(.., F32)`
    /// must reproduce exactly.
    fn plain_f32_verdict(net: &GraphNetwork, spec: &VerificationSpec) -> VerificationResult {
        use ny_propagate::{PropagationConfig, Verifier};
        Verifier::new(PropagationConfig::default())
            .verify_graph(net, spec)
            .expect("plain f32 verify")
    }

    /// Shared assertion: the f16 deployed path refuses fail-closed with the
    /// upstream `UnsupportedConfiguration` (ny hardening 25dee0c5e) — the sound
    /// widening path is not yet implemented, and refusing is the only verdict
    /// that can never be a false `Verified`.
    fn assert_f16_refused(err: ny_core::NyError) {
        match err {
            ny_core::NyError::UnsupportedConfiguration(msg) => {
                assert!(
                    msg.contains("sound deployed-precision verification is not yet implemented"),
                    "refusal must state the missing sound path, got: {msg}"
                );
                assert!(
                    msg.contains("compute=F16"),
                    "refusal must name the refused policy, got: {msg}"
                );
            }
            other => panic!("expected UnsupportedConfiguration, got {other:?}"),
        }
    }

    #[test]
    fn metal_f16_refused_fail_closed_is_the_sound_verdict() {
        let net = tiny_fc_relu_net();
        let spec = spec_generous();

        // Formerly this asserted an f16 `Verified` from the sound widening
        // path; ny 25dee0c5e refuses every non-F32 policy upstream because
        // that path does not yet model every deployed store/reduction. The
        // fail-closed refusal IS the sound posture: no false verification is
        // possible through this adapter.
        let err = verify_metal_f16(&net, &spec)
            .expect_err("f16 deployed verification must refuse (ny 25dee0c5e)");
        assert_f16_refused(err);

        // The refusal is not vacuous: the same net/spec verifies under the
        // f32 idealization, so it is precisely the f16 soundness claim — not
        // the problem instance — that is being refused.
        let f32 = plain_f32_verdict(&net, &spec);
        assert!(f32.is_verified(), "f32 reference must be Verified");
    }

    #[test]
    fn f32_policy_equals_plain_verifier() {
        let net = tiny_fc_relu_net();
        let spec = spec_generous();

        let deployed_f32 =
            verify_deployed(&net, &spec, FloatPrecision::F32).expect("f32 deployed verify");
        let plain = plain_f32_verdict(&net, &spec);

        // F32 is a strict no-op: same verdict variant as the plain verifier.
        assert_eq!(
            deployed_f32.is_verified(),
            plain.is_verified(),
            "F32 deployed verdict must match the plain verifier verdict"
        );
        assert!(
            deployed_f32.is_verified(),
            "both must be Verified for this generous spec, got {deployed_f32:?}"
        );

        // Same engine, same provenance under the idealized policy.
        assert_eq!(
            deployed_f32.provenance().mode(),
            plain.provenance().mode(),
            "F32 deployed verdict must carry the plain verifier's provenance"
        );
    }

    #[test]
    fn f16_refusal_cannot_rescue_an_unsatisfiable_spec() {
        // A spec the f32 path cannot satisfy must stay non-Verified under f16.
        // With the non-F32 path refused upstream (ny 25dee0c5e) the guarantee
        // is trivial and absolute: the f16 entry errors instead of returning
        // any verdict, so no spec — satisfiable or not — can be falsely
        // verified through it.
        let net = tiny_fc_relu_net();
        // input [1,2] -> output ~[2,4]; require an impossible tight band [10, 11].
        let spec = VerificationSpec::new(vec![Bound::new(1.0, 2.0)], vec![Bound::new(10.0, 11.0)])
            .unwrap()
            .with_input_shape(vec![1])
            .unwrap();

        let f32 = plain_f32_verdict(&net, &spec);
        assert!(!f32.is_verified(), "f32 verdict must not satisfy [10,11]");

        let err = verify_metal_f16(&net, &spec)
            .expect_err("f16 deployed verification must refuse (ny 25dee0c5e)");
        assert_f16_refused(err);
    }
}
