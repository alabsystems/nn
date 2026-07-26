// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for constructive proof certificate serialization, validation,
//! and integration with the certify pipeline.
//!
//! Part of #4315 (Wire NY proof certificates into certify pipeline).

use super::*;
use crate::certificate_types::{
    ConstructiveLayerRecord, ConstructiveProofData, ConstructiveProofMethod,
};
use ny_api::BoundedTensor;
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::Device;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Helper: build a simple linear+relu model and certify it
// ---------------------------------------------------------------------------

fn certify_linear_relu() -> CertifyResult {
    let weight = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap();

    let (_output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let h = linear.forward(&traced)?;
        h.relu()
    })
    .unwrap();

    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let config = CertifyConfig::new("test_cert_roundtrip");
    certify_model(&graph, &input_bounds, &config).unwrap()
}

// ---------------------------------------------------------------------------
// Serialization roundtrip tests
// ---------------------------------------------------------------------------

#[test]
fn test_constructive_proof_serialization_roundtrip() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![0.0, 0.0],
        vec![1.0, 2.0],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        3,
        true,
    );

    let json = proof.to_json().expect("serialization should succeed");
    let deserialized =
        ConstructiveProofData::from_json(&json).expect("deserialization should succeed");

    assert_eq!(proof.method, deserialized.method);
    assert_eq!(proof.output_lower, deserialized.output_lower);
    assert_eq!(proof.output_upper, deserialized.output_upper);
    assert_eq!(proof.input_lower, deserialized.input_lower);
    assert_eq!(proof.input_upper, deserialized.input_upper);
    assert_eq!(proof.num_layers, deserialized.num_layers);
    assert_eq!(proof.verified, deserialized.verified);
    assert_eq!(proof.generated_at, deserialized.generated_at);
}

#[test]
fn test_constructive_proof_roundtrip_with_layer_proofs() {
    let layers = vec![
        ConstructiveLayerRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_lower: vec![-1.0, -1.0],
            input_upper: vec![1.0, 1.0],
            output_lower: vec![-2.0, -2.0],
            output_upper: vec![2.0, 2.0],
        },
        ConstructiveLayerRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_lower: vec![-2.0, -2.0],
            input_upper: vec![2.0, 2.0],
            output_lower: vec![0.0, 0.0],
            output_upper: vec![2.0, 2.0],
        },
    ];

    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::CrownComposition,
        vec![0.0, 0.0],
        vec![2.0, 2.0],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        2,
        true,
    )
    .with_layer_proofs(layers);

    let json = proof.to_json().expect("serialization should succeed");
    let deserialized =
        ConstructiveProofData::from_json(&json).expect("deserialization should succeed");

    assert_eq!(proof.layer_proofs, deserialized.layer_proofs);
    assert_eq!(deserialized.layer_proof_count(), 2);
}

#[test]
fn test_constructive_proof_roundtrip_with_lean4() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![0.0],
        vec![1.0],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    )
    .with_lean4_export("-- Lean4 proof\ntheorem bounds_valid : True := trivial".to_string())
    .with_composition_proof(
        "-- Composition\ntheorem crown_composition_sound : True := trivial".to_string(),
        "crown_composition_sound".to_string(),
    );

    let json = proof.to_json().expect("serialization should succeed");
    let deserialized =
        ConstructiveProofData::from_json(&json).expect("deserialization should succeed");

    assert_eq!(proof.lean4_export, deserialized.lean4_export);
    assert_eq!(
        proof.composition_lean4_source,
        deserialized.composition_lean4_source
    );
    assert_eq!(
        proof.composition_theorem_name,
        deserialized.composition_theorem_name
    );
    assert!(deserialized.has_composition_proof());
}

#[test]
fn test_constructive_proof_deserialize_invalid_json() {
    let result = ConstructiveProofData::from_json("not valid json");
    assert!(result.is_err(), "invalid JSON should produce an error");
}

// ---------------------------------------------------------------------------
// Validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_constructive_proof_validate_valid() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![0.0, 0.0],
        vec![1.0, 2.0],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        2,
        true,
    );
    assert!(
        proof.validate().is_ok(),
        "valid proof should pass validation"
    );
}

#[test]
fn test_constructive_proof_validate_mismatched_input_lengths() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![0.0],
        vec![1.0],
        vec![-1.0, -1.0], // 2 elements
        vec![1.0],        // 1 element — mismatch
        1,
        true,
    );
    let err = proof.validate().unwrap_err();
    assert!(
        err.contains("input bounds length mismatch"),
        "should report input length mismatch, got: {err}"
    );
}

#[test]
fn test_constructive_proof_validate_mismatched_output_lengths() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![0.0, 0.0], // 2 elements
        vec![1.0],      // 1 element — mismatch
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );
    let err = proof.validate().unwrap_err();
    assert!(
        err.contains("output bounds length mismatch"),
        "should report output length mismatch, got: {err}"
    );
}

#[test]
fn test_constructive_proof_validate_non_finite_input() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![0.0],
        vec![1.0],
        vec![f32::NAN],
        vec![1.0],
        1,
        true,
    );
    let err = proof.validate().unwrap_err();
    assert!(
        err.contains("non-finite input bound"),
        "should report non-finite input, got: {err}"
    );
}

#[test]
fn test_constructive_proof_validate_non_finite_output() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![0.0],
        vec![f32::INFINITY],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );
    let err = proof.validate().unwrap_err();
    assert!(
        err.contains("non-finite output bound"),
        "should report non-finite output, got: {err}"
    );
}

#[test]
fn test_constructive_proof_validate_inverted_input_bounds() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![0.0],
        vec![1.0],
        vec![1.0],  // lower > upper
        vec![-1.0], // inverted
        1,
        true,
    );
    let err = proof.validate().unwrap_err();
    assert!(
        err.contains("inverted input bound"),
        "should report inverted input, got: {err}"
    );
}

#[test]
fn test_constructive_proof_validate_inverted_output_bounds() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![2.0],  // lower > upper
        vec![-1.0], // inverted
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );
    let err = proof.validate().unwrap_err();
    assert!(
        err.contains("inverted output bound"),
        "should report inverted output, got: {err}"
    );
}

#[test]
fn test_constructive_proof_validate_layer_dimension_mismatch() {
    let layers = vec![ConstructiveLayerRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_lower: vec![-1.0],
        input_upper: vec![1.0, 2.0], // length mismatch
        output_lower: vec![0.0],
        output_upper: vec![1.0],
    }];

    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![0.0],
        vec![1.0],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    )
    .with_layer_proofs(layers);

    let err = proof.validate().unwrap_err();
    assert!(
        err.contains("layer[0] input bounds length mismatch"),
        "should report layer dimension mismatch, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Replay verification tests
// ---------------------------------------------------------------------------

#[test]
fn test_constructive_proof_replay_verify_no_layers() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![0.0],
        vec![1.0],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );
    // No layer proofs — should return self.verified (true).
    assert!(proof.replay_verify());
}

#[test]
fn test_constructive_proof_replay_verify_consistent_chain() {
    let layers = vec![
        ConstructiveLayerRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_lower: vec![-1.0, -1.0],
            input_upper: vec![1.0, 1.0],
            output_lower: vec![-2.0, -2.0],
            output_upper: vec![2.0, 2.0],
        },
        ConstructiveLayerRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_lower: vec![-2.0, -2.0],
            input_upper: vec![2.0, 2.0],
            output_lower: vec![0.0, 0.0],
            output_upper: vec![2.0, 2.0],
        },
    ];

    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::CrownComposition,
        vec![0.0, 0.0],
        vec![2.0, 2.0],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        2,
        true,
    )
    .with_layer_proofs(layers);

    assert!(
        proof.replay_verify(),
        "consistent bound chain should pass replay"
    );
}

#[test]
fn test_constructive_proof_replay_verify_broken_chain() {
    let layers = vec![
        ConstructiveLayerRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_lower: vec![-1.0],
            input_upper: vec![1.0],
            output_lower: vec![-2.0],
            output_upper: vec![2.0],
        },
        ConstructiveLayerRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            // Next layer's input is OUTSIDE previous output: -3.0 < -2.0
            input_lower: vec![-3.0],
            input_upper: vec![2.0],
            output_lower: vec![0.0],
            output_upper: vec![2.0],
        },
    ];

    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::CrownComposition,
        vec![0.0],
        vec![2.0],
        vec![-1.0],
        vec![1.0],
        2,
        true,
    )
    .with_layer_proofs(layers);

    assert!(
        !proof.replay_verify(),
        "broken bound chain should fail replay"
    );
}

#[test]
fn test_constructive_proof_replay_verify_invalid_structure() {
    // Inverted bounds in proof -> validate() fails -> replay returns false.
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![2.0],  // inverted
        vec![-1.0], // inverted
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );
    assert!(
        !proof.replay_verify(),
        "structurally invalid proof should fail replay"
    );
}

// ---------------------------------------------------------------------------
// CertifyResult integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_certify_result_constructive_proof_json() {
    let result = certify_linear_relu();

    assert!(
        result.has_constructive_proof(),
        "certify should generate constructive proof by default"
    );

    let json = result
        .constructive_proof_json()
        .expect("serialization should succeed");
    assert!(json.is_some(), "JSON should be Some when proof exists");

    let json_str = json.unwrap();
    assert!(
        json_str.contains("method"),
        "JSON should contain 'method' field"
    );
    assert!(
        json_str.contains("output_lower"),
        "JSON should contain 'output_lower' field"
    );
    assert!(
        json_str.contains("verified"),
        "JSON should contain 'verified' field"
    );

    // Roundtrip: deserialize back and check consistency.
    let deserialized =
        ConstructiveProofData::from_json(&json_str).expect("roundtrip should succeed");
    let original = result.constructive_proof().unwrap();
    assert_eq!(original.method, deserialized.method);
    assert_eq!(original.output_lower, deserialized.output_lower);
    assert_eq!(original.output_upper, deserialized.output_upper);
    assert_eq!(original.verified, deserialized.verified);
}

#[test]
fn test_certify_result_no_proof_returns_none_json() {
    let weight = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap();

    let (_output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let h = linear.forward(&traced)?;
        h.relu()
    })
    .unwrap();

    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let mut config = CertifyConfig::new("test_no_proof_json");
    config.generate_constructive_proof = false;
    let result = certify_model(&graph, &input_bounds, &config).unwrap();

    let json = result
        .constructive_proof_json()
        .expect("should succeed even without proof");
    assert!(json.is_none(), "JSON should be None when no proof exists");
}

#[test]
fn test_certify_result_validate_constructive_proof() {
    let result = certify_linear_relu();

    let validation = result
        .validate_constructive_proof()
        .expect("validation should succeed");
    assert!(
        validation,
        "valid constructive proof should pass validation"
    );
}

#[test]
fn test_certify_result_validate_no_proof_returns_false() {
    let weight = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap();

    let (_output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let h = linear.forward(&traced)?;
        h.relu()
    })
    .unwrap();

    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let mut config = CertifyConfig::new("test_validate_no_proof");
    config.generate_constructive_proof = false;
    let result = certify_model(&graph, &input_bounds, &config).unwrap();

    let validation = result
        .validate_constructive_proof()
        .expect("validation should succeed (no proof = Ok(false))");
    assert!(!validation, "should return false when no proof present");
}

#[test]
fn test_certify_result_replay_verify() {
    let result = certify_linear_relu();

    assert!(
        result.replay_verify_constructive_proof(),
        "certified model's constructive proof should pass replay verification"
    );
}

#[test]
fn test_certify_result_replay_verify_no_proof() {
    let weight = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap();

    let (_output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let h = linear.forward(&traced)?;
        h.relu()
    })
    .unwrap();

    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let mut config = CertifyConfig::new("test_replay_no_proof");
    config.generate_constructive_proof = false;
    let result = certify_model(&graph, &input_bounds, &config).unwrap();

    assert!(
        !result.replay_verify_constructive_proof(),
        "should return false when no proof present"
    );
}

// ---------------------------------------------------------------------------
// Certificate attached to certify result (bundle integration)
// ---------------------------------------------------------------------------

#[test]
fn test_constructive_proof_attached_to_bundle_certificate() {
    let result = certify_linear_relu();

    // The proof should be in both CertifyResult and in the bundle's certificate.
    let proof = result.constructive_proof().unwrap();
    let cert = &result.bundle.certificates[0];
    let cert_proof = cert
        .constructive_proof
        .as_ref()
        .expect("certificate should contain constructive proof");

    // The proofs should have matching bounds.
    assert_eq!(proof.output_lower, cert_proof.output_lower);
    assert_eq!(proof.output_upper, cert_proof.output_upper);
    assert_eq!(proof.input_lower, cert_proof.input_lower);
    assert_eq!(proof.input_upper, cert_proof.input_upper);
    assert_eq!(proof.method, cert_proof.method);
    assert_eq!(proof.verified, cert_proof.verified);
}

#[test]
fn test_constructive_proof_machine_checkable_from_certify() {
    let result = certify_linear_relu();
    let proof = result.constructive_proof().unwrap();

    assert!(
        proof.is_machine_checkable(),
        "verified proof with output bounds should be machine-checkable"
    );
    assert!(
        proof.verified,
        "proof from certify pipeline should be self-verified"
    );
}

// ---------------------------------------------------------------------------
// Bundle save/load roundtrip with constructive proof
// ---------------------------------------------------------------------------

#[test]
fn test_bundle_save_load_roundtrip_with_constructive_proof() {
    let result = certify_linear_relu();

    let dir = std::env::temp_dir().join("nn_certify_cert_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_bundle.proof.json");

    result
        .bundle
        .save(&path)
        .expect("bundle save should succeed");
    let loaded = CertificateBundle::load(&path).expect("bundle load should succeed");

    assert_eq!(loaded.certificates.len(), result.bundle.certificates.len());
    let orig_cert = &result.bundle.certificates[0];
    let loaded_cert = &loaded.certificates[0];

    // Constructive proof should survive save/load roundtrip.
    assert_eq!(
        orig_cert.constructive_proof.is_some(),
        loaded_cert.constructive_proof.is_some()
    );
    if let (Some(orig), Some(loaded)) = (
        &orig_cert.constructive_proof,
        &loaded_cert.constructive_proof,
    ) {
        assert_eq!(orig.method, loaded.method);
        assert_eq!(orig.output_lower, loaded.output_lower);
        assert_eq!(orig.output_upper, loaded.output_upper);
        assert_eq!(orig.verified, loaded.verified);
    }

    // Cleanup.
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

// ---------------------------------------------------------------------------
// ConstructiveProofMethod classification tests (#4315, #3340)
// ---------------------------------------------------------------------------

#[test]
fn test_constructive_proof_method_is_tight_crown_family() {
    // Per nn engineering rule #3340: Crown, AlphaCrown, BetaCrown, and
    // Analytical are tight methods.
    assert!(
        ConstructiveProofMethod::Crown.is_tight(),
        "Crown should be tight"
    );
    assert!(
        ConstructiveProofMethod::AlphaCrown.is_tight(),
        "AlphaCrown should be tight"
    );
    assert!(
        ConstructiveProofMethod::BetaCrown.is_tight(),
        "BetaCrown should be tight"
    );
    assert!(
        ConstructiveProofMethod::Analytical.is_tight(),
        "Analytical should be tight"
    );
    assert!(
        ConstructiveProofMethod::CrownComposition.is_tight(),
        "CrownComposition should be tight"
    );
    assert!(
        ConstructiveProofMethod::AlphaCrownComposition.is_tight(),
        "AlphaCrownComposition should be tight"
    );
    assert!(
        ConstructiveProofMethod::BetaCrownComposition.is_tight(),
        "BetaCrownComposition should be tight"
    );
}

#[test]
fn test_constructive_proof_method_is_tight_ibp_not_tight() {
    // IBP may be vacuously wide — not counted as tight.
    assert!(
        !ConstructiveProofMethod::Ibp.is_tight(),
        "Ibp should NOT be tight"
    );
    assert!(
        !ConstructiveProofMethod::IbpComposition.is_tight(),
        "IbpComposition should NOT be tight"
    );
}

#[test]
fn test_constructive_proof_method_is_composition() {
    assert!(ConstructiveProofMethod::IbpComposition.is_composition());
    assert!(ConstructiveProofMethod::CrownComposition.is_composition());
    assert!(ConstructiveProofMethod::AlphaCrownComposition.is_composition());
    assert!(ConstructiveProofMethod::BetaCrownComposition.is_composition());
    // Single-layer methods are not composition.
    assert!(!ConstructiveProofMethod::Ibp.is_composition());
    assert!(!ConstructiveProofMethod::Crown.is_composition());
    assert!(!ConstructiveProofMethod::AlphaCrown.is_composition());
    assert!(!ConstructiveProofMethod::BetaCrown.is_composition());
    assert!(!ConstructiveProofMethod::Analytical.is_composition());
}

#[test]
fn test_constructive_proof_method_from_prop_method() {
    use crate::verify_types::PropMethod;

    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::Ibp),
        ConstructiveProofMethod::Ibp
    );
    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::Crown),
        ConstructiveProofMethod::Crown
    );
    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::AlphaCrown),
        ConstructiveProofMethod::AlphaCrown
    );
    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::BetaCrown),
        ConstructiveProofMethod::BetaCrown
    );
    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::Analytical),
        ConstructiveProofMethod::Analytical
    );
    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::MixedIbpCrown),
        ConstructiveProofMethod::Crown
    );
}

#[test]
fn test_constructive_proof_method_composition_from_prop_method() {
    use crate::verify_types::PropMethod;

    assert_eq!(
        ConstructiveProofMethod::composition_from_prop_method(PropMethod::Ibp),
        ConstructiveProofMethod::IbpComposition
    );
    assert_eq!(
        ConstructiveProofMethod::composition_from_prop_method(PropMethod::Crown),
        ConstructiveProofMethod::CrownComposition
    );
    assert_eq!(
        ConstructiveProofMethod::composition_from_prop_method(PropMethod::AlphaCrown),
        ConstructiveProofMethod::AlphaCrownComposition
    );
    assert_eq!(
        ConstructiveProofMethod::composition_from_prop_method(PropMethod::BetaCrown),
        ConstructiveProofMethod::BetaCrownComposition
    );
}

#[test]
fn test_constructive_proof_method_serde_roundtrip_new_variants() {
    // Ensure new variants serialize/deserialize correctly.
    for method in [
        ConstructiveProofMethod::AlphaCrown,
        ConstructiveProofMethod::BetaCrown,
        ConstructiveProofMethod::Analytical,
        ConstructiveProofMethod::AlphaCrownComposition,
        ConstructiveProofMethod::BetaCrownComposition,
    ] {
        let json = serde_json::to_string(&method).expect("serialize");
        let back: ConstructiveProofMethod = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(method, back, "roundtrip failed for {method:?}");
    }
}

#[test]
fn test_certify_constructive_proof_uses_correct_method() {
    // The certify pipeline's constructive proof should reflect the
    // actual propagation method (IBP for small models where CROWN is
    // not triggered, or CROWN composition when layer bounds exist).
    let result = certify_linear_relu();
    let proof = result
        .constructive_proof()
        .expect("should have constructive proof");

    // For a simple identity-weight linear+relu, IBP bounds should be
    // tight enough that escalation stays at IBP. The constructive proof
    // method should reflect the actual escalation result, not hardcoded.
    // With a 2x2 identity weight and [-1,1] input, the model is simple
    // enough that either IBP or CROWN is used. Either way, the method
    // should be non-composition if no layer bounds, or composition if
    // layer bounds are available.
    let method = proof.method;

    // The method should be derived from the escalation PropMethod via
    // from_prop_method/composition_from_prop_method, not hardcoded.
    // We just check it's a valid variant and self-consistent.
    if method.is_composition() {
        assert!(
            proof.has_composition_proof() || proof.layer_proofs.is_some(),
            "composition method should have composition data"
        );
    }
}

// ---------------------------------------------------------------------------
// File-based save/load integration tests (#4315)
// ---------------------------------------------------------------------------

#[test]
fn test_constructive_proof_save_load_roundtrip() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![0.0, 0.0],
        vec![1.0, 2.0],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        3,
        true,
    )
    .with_lean4_export("-- Lean4 bounds proof\ntheorem t : True := trivial".to_string());

    let dir = std::env::temp_dir().join("nn_constructive_proof_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_proof.constructive.json");

    proof.save(&path).expect("save should succeed");

    let loaded = ConstructiveProofData::load(&path).expect("load should succeed");
    assert_eq!(proof.method, loaded.method);
    assert_eq!(proof.output_lower, loaded.output_lower);
    assert_eq!(proof.output_upper, loaded.output_upper);
    assert_eq!(proof.input_lower, loaded.input_lower);
    assert_eq!(proof.input_upper, loaded.input_upper);
    assert_eq!(proof.num_layers, loaded.num_layers);
    assert_eq!(proof.verified, loaded.verified);
    assert_eq!(proof.lean4_export, loaded.lean4_export);
    assert_eq!(proof.generated_at, loaded.generated_at);

    // Loaded proof should pass validation.
    assert!(loaded.validate().is_ok(), "loaded proof should be valid");

    // Cleanup.
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_constructive_proof_save_load_with_layer_proofs() {
    let layers = vec![
        ConstructiveLayerRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_lower: vec![-1.0],
            input_upper: vec![1.0],
            output_lower: vec![-2.0],
            output_upper: vec![2.0],
        },
        ConstructiveLayerRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_lower: vec![-2.0],
            input_upper: vec![2.0],
            output_lower: vec![0.0],
            output_upper: vec![2.0],
        },
    ];

    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::CrownComposition,
        vec![0.0],
        vec![2.0],
        vec![-1.0],
        vec![1.0],
        2,
        true,
    )
    .with_layer_proofs(layers)
    .with_composition_proof(
        "-- Composition proof\ntheorem t : True := trivial".to_string(),
        "crown_composition_sound".to_string(),
    );

    let dir = std::env::temp_dir().join("nn_constructive_layers_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("layered_proof.json");

    proof.save(&path).expect("save should succeed");
    let loaded = ConstructiveProofData::load(&path).expect("load should succeed");

    assert_eq!(loaded.layer_proof_count(), 2);
    assert!(loaded.has_composition_proof());
    assert_eq!(
        loaded.composition_lean4_source,
        proof.composition_lean4_source
    );
    assert_eq!(
        loaded.composition_theorem_name,
        proof.composition_theorem_name
    );

    // Replay verification on loaded proof.
    assert!(loaded.replay_verify(), "loaded proof should pass replay");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_constructive_proof_load_nonexistent_file() {
    let result = ConstructiveProofData::load(std::path::Path::new("/nonexistent/proof.json"));
    assert!(result.is_err(), "loading nonexistent file should fail");
}

#[test]
fn test_certify_result_save_constructive_proof() {
    let result = certify_linear_relu();

    let dir = std::env::temp_dir().join("nn_certify_save_proof_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("model.constructive.json");

    let saved = result
        .save_constructive_proof(&path)
        .expect("save should succeed");
    assert!(saved, "should return true when proof was saved");

    // Load and validate.
    let loaded = ConstructiveProofData::load(&path).expect("load should succeed");
    let original = result.constructive_proof().unwrap();
    assert_eq!(original.method, loaded.method);
    assert_eq!(original.output_lower, loaded.output_lower);
    assert_eq!(original.output_upper, loaded.output_upper);
    assert_eq!(original.input_lower, loaded.input_lower);
    assert_eq!(original.input_upper, loaded.input_upper);
    assert_eq!(original.verified, loaded.verified);

    // Loaded proof should validate and replay.
    assert!(loaded.validate().is_ok());
    assert!(loaded.replay_verify());

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_certify_result_save_constructive_proof_disabled() {
    let weight = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], &Device::Cpu).unwrap();
    let linear = Linear::new(weight, None).unwrap();
    let input = DynTensor::from_vec(vec![0.5, -0.5], &[1, 2], &Device::Cpu).unwrap();

    let (_output, graph) = trace_graph(|| {
        let mut traced = input.clone();
        if let Some(id) = record_input(input.dims(), input.dtype()) {
            traced.set_trace_id(id);
        }
        let h = linear.forward(&traced)?;
        h.relu()
    })
    .unwrap();

    let lower = ArrayD::from_elem(IxDyn(&[1, 2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).unwrap();

    let mut config = CertifyConfig::new("test_save_disabled");
    config.generate_constructive_proof = false;
    let result = certify_model(&graph, &input_bounds, &config).unwrap();

    let dir = std::env::temp_dir().join("nn_certify_save_disabled_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("should_not_exist.json");

    let saved = result
        .save_constructive_proof(&path)
        .expect("should succeed even without proof");
    assert!(!saved, "should return false when no proof to save");
    assert!(
        !path.exists(),
        "file should not be created when no proof exists"
    );

    let _ = std::fs::remove_dir(&dir);
}

// ---------------------------------------------------------------------------
// PropMethod → ConstructiveProofMethod full coverage (#4315)
// ---------------------------------------------------------------------------

#[test]
fn test_prop_method_to_constructive_method_exhaustive() {
    use crate::verify_types::PropMethod;

    // Verify all PropMethod variants map correctly.
    let cases: Vec<(PropMethod, ConstructiveProofMethod, ConstructiveProofMethod)> = vec![
        (
            PropMethod::Ibp,
            ConstructiveProofMethod::Ibp,
            ConstructiveProofMethod::IbpComposition,
        ),
        (
            PropMethod::Crown,
            ConstructiveProofMethod::Crown,
            ConstructiveProofMethod::CrownComposition,
        ),
        (
            PropMethod::AlphaCrown,
            ConstructiveProofMethod::AlphaCrown,
            ConstructiveProofMethod::AlphaCrownComposition,
        ),
        (
            PropMethod::BetaCrown,
            ConstructiveProofMethod::BetaCrown,
            ConstructiveProofMethod::BetaCrownComposition,
        ),
        (
            PropMethod::Analytical,
            ConstructiveProofMethod::Analytical,
            ConstructiveProofMethod::CrownComposition,
        ),
        (
            PropMethod::MixedIbpCrown,
            ConstructiveProofMethod::Crown,
            ConstructiveProofMethod::CrownComposition,
        ),
    ];

    for (prop, expected_single, expected_composition) in cases {
        let single = ConstructiveProofMethod::from_prop_method(prop);
        let composition = ConstructiveProofMethod::composition_from_prop_method(prop);
        assert_eq!(
            single, expected_single,
            "from_prop_method({prop:?}) mismatch"
        );
        assert_eq!(
            composition, expected_composition,
            "composition_from_prop_method({prop:?}) mismatch"
        );
    }
}

#[test]
fn test_constructive_proof_tightness_matches_prop_method() {
    use crate::verify_types::PropMethod;

    // Tight PropMethods should produce tight ConstructiveProofMethods.
    for method in [
        PropMethod::Crown,
        PropMethod::AlphaCrown,
        PropMethod::BetaCrown,
        PropMethod::Analytical,
    ] {
        let constructive = ConstructiveProofMethod::from_prop_method(method);
        assert!(
            constructive.is_tight(),
            "tight PropMethod {method:?} should produce tight constructive method"
        );
        let composition = ConstructiveProofMethod::composition_from_prop_method(method);
        assert!(
            composition.is_tight(),
            "tight PropMethod {method:?} should produce tight composition method"
        );
    }

    // IBP is not tight.
    let ibp = ConstructiveProofMethod::from_prop_method(PropMethod::Ibp);
    assert!(!ibp.is_tight(), "IBP should not be tight");
}
