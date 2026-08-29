// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Proof-carrying, Clean-checkable certificates via `ny_api::cert`.
//!
//! This adapter is NN's real proof-carrying certificate story. It wraps ny's
//! [`ny_api::cert::certify_graph`], which runs the graph verifier and — when the
//! network and property are eligible — re-derives an EXACT-RATIONAL CROWN
//! certificate, self-checks it (entailment + Farkas replay), and attaches it to
//! the verdict's proof channel as Clean-canonical JSON.
//!
//! Eligibility is a static architecture/property gate: a sequential
//! fully-connected ReLU network (`Linear, ReLU, …, Linear`, single linear chain,
//! no fan-in) whose property is a conjunction of per-output intervals, `Linear`
//! halfspaces (`a·y ≤/≥ b`), or `ArgmaxMargin` robustness. For such models the
//! emitted certificate is a machine-checkable proof over exact rationals (no
//! floating-point round-off): an external Clean kernel-backed checker can replay
//! it.
//!
//! This is the successor to NN's previously-stubbed gamma/CROWN constructive
//! proof composition (see `certify_constructive.rs`, where composition was left
//! `None` pending NY's `proof-certificates` feature). Where that path emitted at
//! best a bounds-only enrichment, `ny_api::cert` emits an exact-rational
//! certificate that genuinely discharges every conjunct of the verified property.
//!
//! # Soundness invariants (enforced by `ny_api::cert`, surfaced here verbatim)
//!
//! - The returned [`CertifiedResult::result`] is ALWAYS the verifier's verdict;
//!   the certificate is a purely additive artifact and never alters it.
//! - A certificate is emitted (`certificate_json == Some`) ONLY when the network
//!   and property are eligible AND the verdict is `Verified` AND every conjunct
//!   was closed by exact CROWN and passed the in-tree self-check. It is never a
//!   partial certificate.
//! - When `eligible == false` or `certificate_json == None`, [`certify_to_file`]
//!   writes NOTHING and reports `false`: no certificate is ever claimed or
//!   persisted for an ineligible or uncertified network.
//!
//! # Stricter certificate semantics (upstream ny-cert hardening, 2026-07/08)
//!
//! The certificate engine behind `ny_api::cert` was hardened after this adapter
//! first landed. The changes visible through the facade — and therefore through
//! this module — are:
//!
//! - **Proof minimization.** Certificates are pruned of dead rows (premise /
//!   constraint rows whose Farkas multiplier is exactly zero) before emission,
//!   roughly halving certificate size. The pruning is fail-closed: the
//!   minimized certificate is kept only when the in-tree self-checkers accept
//!   it with bounds identical to the full one; otherwise the full certificate
//!   is emitted unchanged. Consumers must not assume a fixed row count per
//!   conjunct — only that every emitted row is live.
//! - **Fail-closed margin discipline.** Certification near a margin boundary
//!   declines more often than it used to, and that is sound by design. A
//!   strict conjunct (`ArgmaxMargin` requires `margin > 0`) is discharged only
//!   by an exactly-positive rational lower bound, so a tie (`L == 0`) — or any
//!   conjunct exact CROWN cannot close — refuses the WHOLE certificate rather
//!   than overclaim. Likewise every exact-rational read is checked: a value
//!   that cannot be serialized refuses the certificate instead of emitting a
//!   corrupt one, and certificate content is ambient-free (no
//!   environment-variable sensitivity). In all of these cases the verifier's
//!   verdict is NEVER altered — only the additive proof artifact is withheld —
//!   so treat `certificate_json == None` as "no machine-checkable proof", not
//!   as a verification failure.
//!
//! ny-cert additionally gained full Alethe proof emission (checkable by the
//! external `carcara` checker, with reserved-symbol rejection), a deployed-f32
//! rounding-margin fold for dominance thresholds (`ny_cert::fp_margin`,
//! consumed by ny-groundtruth), and conditioned-proof quarantine in the CLI's
//! beta-CROWN certificate mode. None of those are re-exported by
//! `ny_api::cert`, so this adapter — which stays behind the facade by design —
//! does not wrap them.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use nn_verify::ny_certify::certify_model;
//! # fn demo(net: &ny_propagate::GraphNetwork, spec: &ny_core::VerificationSpec)
//! #     -> ny_core::Result<()> {
//! let certified = certify_model(net, spec)?;
//! if certified.eligible {
//!     if let Some(json) = &certified.certificate_json {
//!         // Clean-canonical, exact-rational, replayable certificate.
//!         println!("{json}");
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use std::path::Path;

use ny_api::cert::{certify_graph, CertifiedResult};
use ny_core::{NyError, Result, VerificationSpec};
use ny_propagate::GraphNetwork;

/// Verify `spec` on `net` and, when sound to do so, attach an exact-rational,
/// Clean-checkable proof-carrying certificate to the verdict.
///
/// Thin pass-through to [`ny_api::cert::certify_graph`]; see this module's docs
/// for the eligibility gate and soundness invariants. The returned
/// [`CertifiedResult`] carries:
///
/// - `result`: the verifier's verdict (unchanged by certification). When a
///   certificate is emitted, this is the `Verified` verdict with its
///   [`ny_core::VerificationProof`] channel populated with the certificate bytes.
/// - `certificate_json`: `Some(json)` only when an exact-rational certificate was
///   built, self-checked, and serialized (never partial); otherwise `None`.
/// - `eligible`: whether the static architecture/property gate passed (a
///   sequential FC-ReLU net with a conjunctive / halfspace / argmax property).
/// - `note`: a human-readable explanation of the outcome.
///
/// # Errors
///
/// Returns any error raised by the underlying graph verifier (e.g. a malformed
/// network). Ineligibility and an un-emittable certificate are NOT errors — they
/// are reported via `eligible` / `certificate_json` / `note`.
pub fn certify_model(net: &GraphNetwork, spec: &VerificationSpec) -> Result<CertifiedResult> {
    certify_graph(net, spec)
}

/// Run [`certify_model`] and, when (and only when) an exact-rational certificate
/// was produced, write its Clean-canonical JSON to `path`.
///
/// Returns `Ok(true)` iff a certificate was written, `Ok(false)` otherwise.
///
/// # Soundness
///
/// This NEVER writes — and never claims — a certificate when the network is
/// ineligible (`eligible == false`) or no certificate was emitted
/// (`certificate_json == None`). In those cases `path` is left untouched and the
/// function reports `false`. A file at `path` therefore always corresponds to a
/// genuinely certified, self-checked verdict.
///
/// # Errors
///
/// Propagates verifier errors from [`certify_model`], and surfaces any I/O error
/// from writing the certificate file.
pub fn certify_to_file(
    net: &GraphNetwork,
    spec: &VerificationSpec,
    path: impl AsRef<Path>,
) -> Result<bool> {
    let certified = certify_model(net, spec)?;
    // Fail-closed: only persist a certificate that was actually emitted for an
    // eligible, certified verdict. Never write for an ineligible network even if
    // (defensively) a JSON were somehow present.
    match (certified.eligible, &certified.certificate_json) {
        (true, Some(json)) => {
            std::fs::write(path, json)
                .map_err(|e| NyError::ModelLoad(format!("failed to write certificate: {e}")))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use ny_core::{Bound, OutputConstraint, ProofFormat, VerificationResult};
    use ny_propagate::layers::{LinearLayer, ReLULayer, SiLULayer};
    use ny_propagate::{GraphNode, Layer};

    fn linear(weight: Array2<f32>, bias: Vec<f32>) -> Layer {
        Layer::Linear(LinearLayer::new(weight, Some(Array1::from(bias))).expect("valid linear"))
    }

    /// y = 2 * relu(x0) + 1/2 as a 1->1->1 FC-ReLU graph.
    /// For x0 in [1, 2] (ReLU active) -> y in [5/2, 9/2]; an eligible net.
    fn eligible_fc_relu_net() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            linear(
                Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                vec![0.0],
            ),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
                vec![0.5],
            ),
            vec!["relu".to_string()],
        ));
        g.set_output("lin2");
        g
    }

    /// 1->1->2 classifier where class 0 STRICTLY dominates: y0 = 2*relu(x0),
    /// y1 = relu(x0). Over x0 in [1, 2] the argmax margin y0 - y1 = relu(x0)
    /// is in [1, 2] — exact lower bound 1 > 0, so strict argmax holds.
    fn strict_classifier_1to2() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            linear(
                Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                vec![0.0],
            ),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((2, 1), vec![2.0, 1.0]).unwrap(),
                vec![0.0, 0.0],
            ),
            vec!["relu".to_string()],
        ));
        g.set_output("lin2");
        g
    }

    /// 1->1->2 classifier whose class outputs are IDENTICAL: y0 = y1 = relu(x0).
    /// The argmax margin is exactly 0 everywhere — a tie, the canonical
    /// near-margin case that must DECLINE certification (fail-closed).
    fn tie_classifier_1to2() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            linear(
                Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                vec![0.0],
            ),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap(),
                vec![0.0, 0.0],
            ),
            vec!["relu".to_string()],
        ));
        g.set_output("lin2");
        g
    }

    /// Same shape but with a SiLU (non-FC) activation -> ineligible for cert.
    fn ineligible_net() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "lin1",
            linear(
                Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                vec![0.0],
            ),
        ));
        g.add_node(GraphNode::new(
            "silu",
            Layer::SiLU(SiLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(
                Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
                vec![0.5],
            ),
            vec!["silu".to_string()],
        ));
        g.set_output("lin2");
        g
    }

    #[test]
    fn eligible_satisfied_net_certifies_with_proof_channel() {
        let net = eligible_fc_relu_net();
        // x0 in [1, 2] -> y in [5/2, 9/2]. Assert y >= 0 (a conjunct exact CROWN
        // closes); upper +inf so no upper conjunct.
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(0.0, f32::INFINITY)],
        )
        .expect("valid spec");

        let out = certify_model(&net, &spec).expect("certify_model runs");
        assert!(out.eligible, "FC-ReLU net must be eligible: {}", out.note);
        assert!(
            out.result.is_verified(),
            "y in [5/2, 9/2] satisfies y >= 0: {:?}",
            out.result
        );
        let json = out
            .certificate_json
            .as_ref()
            .unwrap_or_else(|| panic!("expected a certificate; note: {}", out.note));
        assert!(
            json.contains("ny-cert/crown-deep/v1"),
            "Clean-canonical certificate format"
        );

        // The proof channel must be populated with the certificate bytes.
        let VerificationResult::Verified { proof, .. } = &out.result else {
            panic!("verified result expected");
        };
        let proof = proof.as_ref().expect("proof channel populated");
        assert_eq!(proof.format(), ProofFormat::BoundTrace);
        assert_eq!(proof.as_bytes(), json.as_bytes());
    }

    #[test]
    fn ineligible_net_gets_no_certificate() {
        let net = ineligible_net();
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec");

        let out = certify_model(&net, &spec).expect("certify_model runs");
        assert!(!out.eligible, "SiLU net must be ineligible: {}", out.note);
        assert!(
            out.certificate_json.is_none(),
            "ineligible net must not get a certificate"
        );
    }

    #[test]
    fn certify_to_file_writes_only_when_certified() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();

        // Eligible + satisfied -> a certificate is written.
        let net = eligible_fc_relu_net();
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(0.0, f32::INFINITY)],
        )
        .expect("valid spec");
        let ok_path = dir.join(format!("nn_verify_ny_certify_ok_{pid}.json"));
        let _ = std::fs::remove_file(&ok_path);
        let wrote = certify_to_file(&net, &spec, &ok_path).expect("certify_to_file runs");
        assert!(wrote, "eligible+verified net must write a certificate");
        let on_disk = std::fs::read_to_string(&ok_path).expect("certificate file readable");
        assert!(
            on_disk.contains("ny-cert/crown-deep/v1"),
            "persisted file is the Clean-canonical certificate"
        );
        let _ = std::fs::remove_file(&ok_path);

        // Ineligible -> nothing is written and the file does not exist.
        let bad_net = ineligible_net();
        let bad_path = dir.join(format!("nn_verify_ny_certify_bad_{pid}.json"));
        let _ = std::fs::remove_file(&bad_path);
        let wrote_bad = certify_to_file(&bad_net, &spec, &bad_path).expect("certify_to_file runs");
        assert!(!wrote_bad, "ineligible net must NOT write a certificate");
        assert!(
            !bad_path.exists(),
            "no certificate file may be created for an ineligible net"
        );
    }

    // --- Stricter certificate semantics (2026-07/08 ny-cert hardening) -------
    //
    // These pin the two hardening behaviours that reach NN through the facade:
    // proof minimization (no dead rows in emitted certificates) and fail-closed
    // margin discipline (near-margin ties decline the certificate, soundly).

    /// Every emitted certificate row is LIVE: proof minimization prunes dead
    /// (zero-multiplier) premise/constraint rows before emission, so the
    /// certificate this adapter surfaces must carry no `"0"` multiplier.
    #[test]
    fn emitted_certificate_is_minimized_no_dead_rows() {
        let net = eligible_fc_relu_net();
        // Two conjuncts: 2 <= y <= 5, both holding with slack (y in [5/2, 9/2]).
        let spec = VerificationSpec::new(vec![Bound::new(1.0, 2.0)], vec![Bound::new(2.0, 5.0)])
            .expect("valid spec");

        let out = certify_model(&net, &spec).expect("certify_model runs");
        let json = out
            .certificate_json
            .as_ref()
            .unwrap_or_else(|| panic!("expected a certificate; note: {}", out.note));
        let parsed: serde_json::Value = serde_json::from_str(json).expect("cert JSON parses");
        let conjuncts = parsed["conjuncts"].as_array().expect("conjuncts array");
        assert_eq!(conjuncts.len(), 2, "lower and upper conjuncts discharged");
        for conjunct in conjuncts {
            for key in ["entailment", "farkas"] {
                let mults = conjunct[key]["multipliers"]
                    .as_array()
                    .unwrap_or_else(|| panic!("{key} multipliers array present"));
                assert!(!mults.is_empty(), "{key} must keep at least one live row");
                for m in mults {
                    let m = m.as_str().expect("multiplier is an exact-rational string");
                    assert_ne!(
                        m, "0",
                        "dead (zero-multiplier) row survived minimization in {key}"
                    );
                }
            }
        }
    }

    /// NEAR-MARGIN DECLINE, fail-closed: an exact argmax tie (margin == 0
    /// everywhere) must yield `certificate_json == None` — `L == 0` does not
    /// establish the strict `margin > 0` relation, so no certificate may be
    /// emitted or persisted, whatever the runtime verdict.
    #[test]
    fn near_margin_tie_declines_certificate_fail_closed() {
        let net = tie_classifier_1to2();
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::ArgmaxMargin { class: 0 }])
        .expect("valid constraints");

        let out = certify_model(&net, &spec).expect("certify_model runs");
        assert!(out.eligible, "FC-ReLU classifier is eligible: {}", out.note);
        assert!(
            out.certificate_json.is_none(),
            "a tie (margin == 0) must decline the certificate (no overclaim); note: {}",
            out.note
        );
        if let VerificationResult::Verified { proof, .. } = &out.result {
            assert!(proof.is_none(), "no proof attached on a near-margin tie");
        }

        // The decline is surfaced identically by the persistence path.
        let path = std::env::temp_dir().join(format!(
            "nn_verify_ny_certify_tie_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let wrote = certify_to_file(&net, &spec, &path).expect("certify_to_file runs");
        assert!(!wrote, "declined certificate must not be persisted");
        assert!(
            !path.exists(),
            "no certificate file may be created on a near-margin decline"
        );
    }

    /// STRICT ATTESTATION: when strict argmax genuinely holds (margin lower
    /// bound 1 > 0), the emitted certificate must attest the strict relation
    /// (`strict == true`, `relation == "margin > 0"`) with a strictly positive
    /// exact-rational lower bound — the semantics a Clean/kernel checker needs
    /// to re-derive strictness from the `>= 0` inequality plus `L > 0`.
    #[test]
    fn strict_argmax_certificate_attests_strict_relation() {
        let net = strict_classifier_1to2();
        let spec = VerificationSpec::new(
            vec![Bound::new(1.0, 2.0)],
            vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        )
        .expect("valid spec")
        .with_output_constraints(vec![OutputConstraint::ArgmaxMargin { class: 0 }])
        .expect("valid constraints");

        let out = certify_model(&net, &spec).expect("certify_model runs");
        assert!(out.eligible, "FC-ReLU classifier is eligible: {}", out.note);
        assert!(
            out.result.is_verified(),
            "class 0 strictly dominates: {:?}",
            out.result
        );
        let json = out
            .certificate_json
            .as_ref()
            .unwrap_or_else(|| panic!("expected strict argmax cert; note: {}", out.note));
        let parsed: serde_json::Value = serde_json::from_str(json).expect("cert JSON parses");
        let conjuncts = parsed["conjuncts"].as_array().expect("conjuncts array");
        assert_eq!(conjuncts.len(), 1, "single competitor margin (2-class)");
        let conjunct = &conjuncts[0];
        assert_eq!(conjunct["strict"], serde_json::Value::Bool(true));
        assert_eq!(
            conjunct["relation"],
            serde_json::Value::String("margin > 0".into())
        );
        let lb = conjunct["exact_lower_bound"]
            .as_str()
            .expect("exact bound string");
        let (n, d) = lb.split_once('/').expect("v1 exact bound uses n/d");
        let num: i128 = n.parse().expect("numerator parses");
        let den: i128 = d.parse().expect("denominator parses");
        assert!(
            num.signum() * den.signum() > 0,
            "strict conjunct's exact lower bound must be > 0, got {lb}"
        );
    }
}
