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
    use ny_core::{Bound, ProofFormat, VerificationResult};
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
            linear(Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(), vec![0.0]),
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(), vec![0.5]),
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
            linear(Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(), vec![0.0]),
        ));
        g.add_node(GraphNode::new(
            "silu",
            Layer::SiLU(SiLULayer::new()),
            vec!["lin1".to_string()],
        ));
        g.add_node(GraphNode::new(
            "lin2",
            linear(Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(), vec![0.5]),
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
}
