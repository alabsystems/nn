// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deployed-precision sound verification (Metal f16) via `ny_api::precision`.
//!
//! NY proves bounds under an f32 *idealization*. NN's models do NOT run in f32 on
//! the GPU — the Metal backend executes in f16 — so a bound proven in f32 is not
//! automatically valid for the bits that actually run on-device. This adapter
//! verifies the bounds at the precision NN actually deploys.
//!
//! It delegates to [`ny_api::precision::verify_with_sound_precision`], the
//! layer-aware, SOUND path: it accounts for the accumulation rounding inside each
//! reduction/Linear layer (via direct outward-rounded interval arithmetic), so a
//! `Verified` verdict it returns is sound for the deployed precision. (This is
//! distinct from `ny`'s representation-only *heuristic* widening, which this
//! adapter deliberately does not expose.)
//!
//! # Soundness direction
//!
//! Widening to the deployed grid can only *loosen* the computed output bounds, and
//! NY's verdict rule is monotone, so a verdict can only ever flip from `Verified`
//! to `Unknown` when moving from f32 to f16 — never the reverse. The pass is
//! therefore fail-closed: it never falsely reports `Verified` at a lower
//! precision.
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
/// This is the SOUND, layer-aware path: a `Verified` verdict is valid for the bits
/// that actually execute at `precision`, not just for the f32 idealization. The
/// verdict is fail-closed — widening to the deployed grid can only flip `Verified`
/// to `Unknown`, never the reverse, so this never falsely reports `Verified`.
///
/// [`FloatPrecision::F32`] is a strict no-op: the result is identical to the plain
/// verifier's f32 verdict. Use [`verify_metal_f16`] for NN's actual deployment
/// target.
///
/// # Errors
/// Propagates any error from [`ny_api::precision::verify_with_sound_precision`]
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
/// multiply with f32 accumulate). The same soundness guarantees apply: a
/// `Verified` verdict is valid for the deployed precision, and the idealized
/// all-f32 policy is a strict no-op equal to the plain verifier.
///
/// # Errors
/// Propagates any error from [`ny_api::precision::verify_with_sound_precision`].
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
/// Metal backend runs in f16, so this verifies the bounds for the bits that
/// actually execute on-device rather than the f32 idealization. A `Verified`
/// verdict here is sound for the deployed f16 compute; the pass is fail-closed and
/// can only ever downgrade an f32 `Verified` to `Unknown`.
///
/// # Errors
/// Propagates any error from [`verify_deployed`].
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

    #[test]
    fn metal_f16_returns_sound_verdict_never_falsely_verified() {
        let net = tiny_fc_relu_net();
        let spec = spec_generous();

        let f16 = verify_metal_f16(&net, &spec).expect("f16 verify");

        // Must be a SOUND verdict: Verified or Unknown, never Violated/Timeout for
        // this finite well-formed net, and crucially never a FALSE Verified — the
        // f32 verdict is Verified with a wide margin, so f16 widening (which only
        // loosens) must keep it within the generous band.
        assert!(
            matches!(
                f16,
                VerificationResult::Verified { .. } | VerificationResult::Unknown { .. }
            ),
            "f16 deployed verdict must be Verified or Unknown, got {f16:?}"
        );

        // The f32 reference is Verified with a margin; fail-closed widening cannot
        // turn an Unknown into Verified, so an f16 Verified here is genuinely sound.
        let f32 = plain_f32_verdict(&net, &spec);
        assert!(f32.is_verified(), "f32 reference must be Verified");
        // For this generous band the f16 widening stays inside it -> still Verified.
        assert!(
            f16.is_verified(),
            "f16 verdict for a generously-satisfied spec must remain Verified, got {f16:?}"
        );
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
    fn f16_does_not_rescue_an_unsatisfiable_spec() {
        // A spec the f32 path cannot satisfy must stay non-Verified under f16
        // (widening only loosens bounds — it can never rescue a verdict).
        let net = tiny_fc_relu_net();
        // input [1,2] -> output ~[2,4]; require an impossible tight band [10, 11].
        let spec = VerificationSpec::new(vec![Bound::new(1.0, 2.0)], vec![Bound::new(10.0, 11.0)])
            .unwrap()
            .with_input_shape(vec![1])
            .unwrap();

        let f32 = plain_f32_verdict(&net, &spec);
        assert!(!f32.is_verified(), "f32 verdict must not satisfy [10,11]");

        let f16 = verify_metal_f16(&net, &spec).expect("f16 verify");
        assert!(
            !f16.is_verified(),
            "f16 must not falsely rescue an unsatisfiable spec, got {f16:?}"
        );
    }
}
