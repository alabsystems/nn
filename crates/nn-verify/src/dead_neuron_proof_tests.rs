// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the dead-neuron elimination equivalence proof wrapper
//! (`dead_neuron_proof.rs`).

use ny_api::BoundedTensor;
use ny_propagate::{
    layers::{Layer, LinearLayer, ReLULayer},
    Network,
};
use ndarray::{arr1, arr2, ArrayD, IxDyn};

use super::{run_dead_neuron_elimination, DeadNeuronEliminationProof};

/// Build a simple network with known dead/active/unstable neurons, mirroring
/// upstream gamma-propagate's `elimination_tests::build_test_network`:
///
/// Linear(2->4) -> ReLU -> Linear(4->1) with input region `[-1, 1]^2`.
///
/// Neuron 0: `w=[2, 0], b=3`  => pre-act in `[1, 5]` — always active
/// Neuron 1: `w=[0, -2], b=-3` => pre-act in `[-5, -1]` — dead
/// Neuron 2: `w=[1, 1], b=0`   => pre-act in `[-2, 2]` — unstable
/// Neuron 3: `w=[0, 0], b=5`   => pre-act in `[5, 5]` — always active
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

#[test]
fn test_run_dead_neuron_elimination_proves_equivalence_on_mixed_network() {
    let (network, input) = build_mixed_neuron_network();
    // Upstream's integration test (`test_eliminate_and_verify_dead_neuron_pipeline_4505`)
    // uses epsilon = 10.0 for this fixture: the difference-network CROWN bound
    // on the mixed-neuron graph is ≈ 2.0 under `PropagationConfig::default()`,
    // so any epsilon in (2.0, +inf) certifies equivalence.
    let epsilon = 10.0_f32;

    let proof: DeadNeuronEliminationProof = run_dead_neuron_elimination(&network, &input, epsilon)
        .expect("elimination + equivalence verification should succeed");

    // The original network has 4 neurons at the (only) ReLU layer.
    assert_eq!(proof.neurons_before, 4);
    // Dead neuron 1 must be removed; the other 3 (2 always-active + 1
    // unstable) are kept — matches the upstream fixture exactly.
    assert_eq!(proof.neurons_after, 3);
    assert!(proof.eliminated_any());
    assert!(proof.elimination_fraction > 0.0 && proof.elimination_fraction < 1.0);

    // With epsilon = 10 (well above the fixture's ~2.0 diff bound), the
    // equivalence verifier must certify the optimized network as equivalent.
    assert!(
        proof.equivalent,
        "optimized network should be proven equivalent within epsilon={epsilon}; \
         got worst_case_bound={}",
        proof.worst_case_bound
    );
    assert!(proof.is_deployment_safe());
    assert_eq!(proof.equivalence_label, "equivalent");
    assert!(
        proof.worst_case_bound.is_finite(),
        "worst_case_bound should be finite, got {}",
        proof.worst_case_bound
    );
    assert!(proof.epsilon == epsilon);
}

#[test]
fn test_run_dead_neuron_elimination_rejects_nan_epsilon() {
    let (network, input) = build_mixed_neuron_network();
    let result = run_dead_neuron_elimination(&network, &input, f32::NAN);
    assert!(result.is_err(), "NaN epsilon must be rejected");
}

#[test]
fn test_run_dead_neuron_elimination_rejects_negative_epsilon() {
    let (network, input) = build_mixed_neuron_network();
    let result = run_dead_neuron_elimination(&network, &input, -0.1);
    assert!(result.is_err(), "negative epsilon must be rejected");
}

#[test]
fn test_dead_neuron_elimination_proof_roundtrips_via_json() {
    // Verify the serde impl so callers can attach the proof to JSON
    // certificate bundles.
    let proof = DeadNeuronEliminationProof {
        neurons_before: 4,
        neurons_after: 3,
        elimination_fraction: 0.25,
        layers_before: 3,
        layers_after: 3,
        equivalent: true,
        worst_case_bound: 5.0e-7,
        epsilon: 1e-3,
        equivalence_label: "equivalent".to_string(),
    };
    let json = serde_json::to_string(&proof).expect("serialize");
    let back: DeadNeuronEliminationProof = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(proof, back);
}
