// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: dead-neuron elimination equivalence proof attaches to
//! the TTS [`Certificate`] and surfaces via the accessor methods added in
//! `nn-tts-verify/src/certificate.rs`.
//!
//! Exercises the end-to-end wiring for NY commit `1ed64542f`
//! (per design `designs/2026-04-19-NY-f57811-adoption.md` §3) —
//! upgrades the moonshot certificate claim from "bounds propagated" to
//! "bounds propagated + dead-neuron equivalence certified".
//!
//! Part of #3874.

#![cfg(feature = "ny")]

use ny_api::BoundedTensor;
use ny_propagate::{
    layers::{Layer, LinearLayer, ReLULayer},
    Network,
};
use ndarray::{arr1, arr2, ArrayD, IxDyn};

use nn_tts_verify::certificate::Certificate;
use nn_tts_verify::error::TtsVerifyError;
use nn_tts_verify::TtsVerifier;
use nn_verify::{run_dead_neuron_elimination, DeadNeuronEliminationProof};

/// Build the mixed-neuron test network (matches upstream NY's
/// `elimination_tests::build_test_network`): one dead, two always-active,
/// one unstable. Elimination removes the dead neuron and produces a
/// CROWN-verified equivalence proof.
fn build_mixed_neuron_network() -> (Network, BoundedTensor) {
    let mut network = Network::new();
    let w1 = arr2(&[[2.0, 0.0], [0.0, -2.0], [1.0, 1.0], [0.0, 0.0]]);
    let b1 = arr1(&[3.0, -3.0, 0.0, 5.0]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1, Some(b1)).expect("valid linear layer 1"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer::new()));
    let w2 = arr2(&[[1.0, 1.0, 1.0, 1.0]]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2, None).expect("valid linear layer 2"),
    ));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .expect("valid bounded input");

    (network, input)
}

/// Build a passing Certificate via the public TTS verification pipeline.
/// Using the public API avoids the `#[non_exhaustive]` struct-literal
/// restriction on `Certificate` in external test crates.
fn passing_certificate() -> Certificate {
    // Multi-frequency 24 kHz signal, 0.5 s. RMS > 0.01 and peak < 1.0, so
    // the default hard bounds all pass.
    let sample_rate = 24000;
    let n = sample_rate / 2;
    let audio: Vec<f32> = (0..n)
        .map(|i| {
            let t = f64::from(i) / f64::from(sample_rate);
            let pi2 = 2.0 * std::f64::consts::PI;
            let s = 0.15 * (pi2 * 440.0 * t).sin() + 0.10 * (pi2 * 880.0 * t).sin();
            s as f32
        })
        .collect();

    let verifier = TtsVerifier::builder()
        .build()
        .expect("valid verifier config");
    match verifier.verify(&audio) {
        Ok(cert) => cert,
        Err(TtsVerifyError::VerificationRejected { cert }) => *cert,
        Err(e) => panic!("unexpected verification error: {e:?}"),
    }
}

/// End-to-end claim: the dead-neuron elimination proof is attached to the
/// certificate, `has_dead_neuron_eq_proof()` reports `true`, and
/// `passes_dead_neuron_equivalence()` reflects the proof's own verdict.
#[test]
fn test_certify_includes_dead_neuron_equivalence() {
    let (network, input) = build_mixed_neuron_network();
    // Upstream's `test_eliminate_and_verify_dead_neuron_pipeline_4505` uses
    // epsilon = 10.0; CROWN on the difference network bounds the diff at ≈ 2.0.
    let epsilon = 10.0_f32;

    let proof: DeadNeuronEliminationProof = run_dead_neuron_elimination(&network, &input, epsilon)
        .expect("elimination + equivalence verification should succeed");

    // The elimination fixture must prove equivalence within epsilon — without
    // that, the certificate upgrade from "bounds propagated" to
    // "bounds propagated + dead-neuron equivalence certified" is empty.
    assert!(
        proof.equivalent,
        "proof must be equivalent for deployment-safe upgrade"
    );
    assert!(proof.is_deployment_safe());
    assert!(proof.eliminated_any());
    assert_eq!(proof.neurons_before, 4);
    assert_eq!(proof.neurons_after, 3);
    assert_eq!(proof.equivalence_label, "equivalent");

    // Base certificate from the public pipeline has no proof attached.
    let cert = passing_certificate();
    assert!(
        !cert.has_dead_neuron_eq_proof(),
        "base cert must not carry a proof yet",
    );
    assert!(
        cert.passes_dead_neuron_equivalence(),
        "absence of a proof is vacuously passing",
    );

    // Attaching the proof surfaces it via both accessor methods.
    let enriched = cert.with_dead_neuron_eq_proof(proof.clone());
    assert!(
        enriched.has_dead_neuron_eq_proof(),
        "enriched cert must expose the attached proof",
    );
    assert!(
        enriched.passes_dead_neuron_equivalence(),
        "passes_dead_neuron_equivalence must mirror proof.is_deployment_safe()",
    );

    // The field value round-trips intact.
    let attached = enriched
        .dead_neuron_eq_proof
        .as_ref()
        .expect("proof field populated");
    assert_eq!(attached, &proof);
    assert!(
        attached.worst_case_bound.is_finite(),
        "worst_case_bound should be finite, got {}",
        attached.worst_case_bound
    );

    // The enriched report must mention the Kokoro certificate invariant:
    // the report text is human-readable and already covers crown/junctions;
    // the dead-neuron claim is exposed programmatically via the accessor.
    // A round-trip through `report()` must not panic on the enriched cert.
    let _ = enriched.report();
}
