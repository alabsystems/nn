// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for CROWN certificate generation, validation, composition, and
//! integration with the certify pipeline.
//!
//! These tests exercise the public and crate-internal APIs from the `lib.rs`
//! scope, covering:
//!   - Certificate generation from CROWN propagation results
//!   - Certificate serialization/deserialization (JSON roundtrip)
//!   - Certificate validation (bounds soundness, finite checks)
//!   - Certificate composition (combining sub-network certificates)
//!   - Certificate metadata (model name, timestamp, soundness mode, tightness)
//!   - Integration with the `certify_model` pipeline
//!
//! Part of #4315 (Wire CROWN certificates into model verification pipelines).

use crate::certificate::{
    CertificateBundle, ProofCertificate, CERTIFICATE_VERSION,
};
use crate::certificate_types::{
    ConstructiveLayerRecord, ConstructiveProofData, ConstructiveProofMethod, TransformPass,
    TransformProofBundle, TransformProofEntry,
};
use crate::status::{InputBoundsRecord, ParamInputRecord};
use crate::verify_types::{KernelVerification, PropMethod};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal `KernelVerification` for certificate creation.
fn make_verification(name: &str, lower: f32, upper: f32, method: PropMethod) -> KernelVerification {
    KernelVerification::new(
        name.to_string(),
        method,
        lower,
        upper,
        upper - lower,
        lower.is_finite() && upper.is_finite(),
    )
}

/// Build a minimal `InputBoundsRecord`.
fn make_input_spec(lower: f32, upper: f32) -> InputBoundsRecord {
    InputBoundsRecord::new(
        &[ParamInputRecord {
            param_index: 0,
            lower,
            upper,
        }],
        &[],
    )
}

/// Build a simple two-layer `ConstructiveProofData` with Linear -> ReLU.
fn make_two_layer_proof() -> ConstructiveProofData {
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

    ConstructiveProofData::new(
        ConstructiveProofMethod::CrownComposition,
        vec![0.0, 0.0],
        vec![2.0, 2.0],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        2,
        true,
    )
    .with_layer_proofs(layers)
    .with_composition_proof(
        "-- Lean4 composition proof\ntheorem crown_sound : True := trivial".to_string(),
        "crown_sound".to_string(),
    )
}

// ===========================================================================
// 1. Certificate generation from CROWN propagation results
// ===========================================================================

#[test]
fn test_crown_certificate_from_verification_result() {
    let verif = make_verification("snake_relu", -0.5, 1.5, PropMethod::Crown);
    let input_spec = make_input_spec(-1.0, 1.0);
    let cert = ProofCertificate::from_verification(&verif, input_spec);

    assert_eq!(cert.version, CERTIFICATE_VERSION);
    assert_eq!(cert.kernel_name, "snake_relu");
    assert_eq!(cert.method, PropMethod::Crown);
    assert!(cert.is_finite);
    assert!((cert.output_width - 2.0).abs() < 1e-6);
    assert!(cert.validate().is_ok());
}

#[test]
fn test_crown_certificate_from_alpha_crown_result() {
    let verif = make_verification("norm_snake", -0.2, 0.8, PropMethod::AlphaCrown);
    let input_spec = make_input_spec(-0.5, 0.5);
    let cert = ProofCertificate::from_verification(&verif, input_spec);

    assert_eq!(cert.method, PropMethod::AlphaCrown);
    assert!(cert.is_finite);
    assert!(cert.validate().is_ok());
}

#[test]
fn test_crown_certificate_from_beta_crown_result() {
    let verif = make_verification("conv_block", -0.1, 0.3, PropMethod::BetaCrown);
    let input_spec = make_input_spec(-1.0, 1.0);
    let cert = ProofCertificate::from_verification(&verif, input_spec);

    assert_eq!(cert.method, PropMethod::BetaCrown);
    assert!(cert.is_finite);
    assert!((cert.output_width - 0.4).abs() < 1e-6);
}

#[test]
fn test_crown_certificate_with_constructive_proof_attached() {
    let verif = make_verification("linear_relu", 0.0, 2.0, PropMethod::Crown);
    let input_spec = make_input_spec(-1.0, 1.0);
    let proof = make_two_layer_proof();

    let cert = ProofCertificate::from_verification(&verif, input_spec)
        .with_constructive_proof(proof);

    assert!(cert.has_constructive_proof());
    let attached = cert.constructive_proof.as_ref().unwrap();
    assert_eq!(attached.method, ConstructiveProofMethod::CrownComposition);
    assert!(attached.has_composition_proof());
    assert_eq!(attached.layer_proof_count(), 2);
}

#[test]
fn test_crown_certificate_ibp_fallback_not_tight() {
    let verif = make_verification("wide_kernel", -100.0, 100.0, PropMethod::Ibp);
    let input_spec = make_input_spec(-10.0, 10.0);
    let cert = ProofCertificate::from_verification(&verif, input_spec);

    assert_eq!(cert.method, PropMethod::Ibp);
    // IBP method is not tight per engineering rule #3340.
    assert!(!cert.method.is_tight());
}

// ===========================================================================
// 2. Certificate serialization/deserialization
// ===========================================================================

#[test]
fn test_crown_certificate_json_roundtrip() {
    let verif = make_verification("serde_test", -0.5, 1.5, PropMethod::Crown);
    let input_spec = make_input_spec(-1.0, 1.0);
    let cert = ProofCertificate::from_verification(&verif, input_spec)
        .with_constructive_proof(make_two_layer_proof());

    let json = cert.to_json().expect("serialization should succeed");
    let deserialized: ProofCertificate =
        serde_json::from_str(&json).expect("deserialization should succeed");

    assert_eq!(cert.kernel_name, deserialized.kernel_name);
    assert_eq!(cert.method, deserialized.method);
    assert_eq!(cert.version, deserialized.version);
    assert_eq!(cert.is_finite, deserialized.is_finite);
    assert!(deserialized.constructive_proof.is_some());

    let orig_proof = cert.constructive_proof.as_ref().unwrap();
    let deser_proof = deserialized.constructive_proof.as_ref().unwrap();
    assert_eq!(orig_proof.method, deser_proof.method);
    assert_eq!(orig_proof.output_lower, deser_proof.output_lower);
    assert_eq!(orig_proof.output_upper, deser_proof.output_upper);
    assert_eq!(
        orig_proof.layer_proof_count(),
        deser_proof.layer_proof_count()
    );
}

#[test]
fn test_crown_constructive_proof_json_all_fields() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::AlphaCrown,
        vec![0.1, 0.2, 0.3],
        vec![0.9, 0.8, 0.7],
        vec![-1.0, -1.0, -1.0],
        vec![1.0, 1.0, 1.0],
        5,
        true,
    )
    .with_lean4_export("-- Alpha-CROWN proof\ntheorem alpha_sound : True := trivial".to_string())
    .with_layer_proofs(vec![ConstructiveLayerRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_lower: vec![-1.0, -1.0, -1.0],
        input_upper: vec![1.0, 1.0, 1.0],
        output_lower: vec![0.1, 0.2, 0.3],
        output_upper: vec![0.9, 0.8, 0.7],
    }])
    .with_composition_proof(
        "-- Composition\ntheorem compose : True := trivial".to_string(),
        "compose".to_string(),
    );

    let json = proof.to_json().expect("serialization");
    assert!(json.contains("AlphaCrown"));
    assert!(json.contains("alpha_sound"));
    assert!(json.contains("compose"));
    assert!(json.contains("generated_at"));

    let loaded = ConstructiveProofData::from_json(&json).expect("deserialization");
    assert_eq!(loaded.method, ConstructiveProofMethod::AlphaCrown);
    assert_eq!(loaded.lean4_export, proof.lean4_export);
    assert_eq!(
        loaded.composition_lean4_source,
        proof.composition_lean4_source
    );
    assert_eq!(
        loaded.composition_theorem_name,
        proof.composition_theorem_name
    );
    assert_eq!(loaded.num_layers, 5);
    assert!(loaded.verified);
}

#[test]
fn test_crown_certificate_bundle_json_roundtrip() {
    let verif = make_verification("bundle_test", -0.5, 1.5, PropMethod::Crown);
    let input_spec = make_input_spec(-1.0, 1.0);
    let cert = ProofCertificate::from_verification(&verif, input_spec)
        .with_constructive_proof(make_two_layer_proof());

    let bundle = CertificateBundle::new("test_model").with_certificate(cert);

    let dir = std::env::temp_dir().join("nn_crown_cert_bundle_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("bundle.proof.json");

    bundle.save(&path).expect("save");
    let loaded = CertificateBundle::load(&path).expect("load");

    assert_eq!(loaded.model_name, "test_model");
    assert_eq!(loaded.certificates.len(), 1);
    let loaded_cert = &loaded.certificates[0];
    assert!(loaded_cert.constructive_proof.is_some());
    assert_eq!(loaded_cert.method, PropMethod::Crown);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_constructive_proof_deserialize_missing_optional_fields() {
    // Simulate a v6 certificate with no optional fields set (backward compat).
    let json = r#"{
        "method": "Ibp",
        "output_lower": [0.0],
        "output_upper": [1.0],
        "input_lower": [-1.0],
        "input_upper": [1.0],
        "num_layers": 1,
        "verified": true,
        "generated_at": "2026-04-01T00:00:00Z"
    }"#;

    let proof = ConstructiveProofData::from_json(json).expect("should deserialize");
    assert_eq!(proof.method, ConstructiveProofMethod::Ibp);
    assert!(proof.lean4_export.is_none());
    assert!(proof.layer_proofs.is_none());
    assert!(proof.composition_lean4_source.is_none());
    assert!(proof.composition_theorem_name.is_none());
    assert_eq!(proof.layer_proof_count(), 0);
    assert!(!proof.has_composition_proof());
}

// ===========================================================================
// 3. Certificate validation (bounds soundness)
// ===========================================================================

#[test]
fn test_crown_proof_validate_finite_bounds() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![-0.5, 0.0],
        vec![1.5, 2.0],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        3,
        true,
    );
    assert!(proof.validate().is_ok());
}

#[test]
fn test_crown_proof_validate_rejects_nan_input_lower() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![0.0],
        vec![1.0],
        vec![f32::NAN],
        vec![1.0],
        1,
        true,
    );
    let err = proof.validate().unwrap_err();
    assert!(err.contains("non-finite input bound"));
}

#[test]
fn test_crown_proof_validate_rejects_inf_output_upper() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![0.0],
        vec![f32::INFINITY],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );
    let err = proof.validate().unwrap_err();
    assert!(err.contains("non-finite output bound"));
}

#[test]
fn test_crown_proof_validate_rejects_neg_inf_input() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![0.0],
        vec![1.0],
        vec![f32::NEG_INFINITY],
        vec![1.0],
        1,
        true,
    );
    let err = proof.validate().unwrap_err();
    assert!(err.contains("non-finite input bound"));
}

#[test]
fn test_crown_proof_validate_rejects_inverted_output() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![2.0],
        vec![0.5],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );
    let err = proof.validate().unwrap_err();
    assert!(err.contains("inverted output bound"));
}

#[test]
fn test_crown_proof_validate_rejects_inverted_input() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![-1.0],
        1,
        true,
    );
    let err = proof.validate().unwrap_err();
    assert!(err.contains("inverted input bound"));
}

#[test]
fn test_crown_proof_validate_layer_output_dim_mismatch() {
    let layers = vec![ConstructiveLayerRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_lower: vec![-1.0],
        input_upper: vec![1.0],
        output_lower: vec![0.0, 0.0], // 2 elements
        output_upper: vec![1.0],      // 1 element -- mismatch
    }];

    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![0.0],
        vec![1.0],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    )
    .with_layer_proofs(layers);

    let err = proof.validate().unwrap_err();
    assert!(err.contains("layer[0] output bounds length mismatch"));
}

#[test]
fn test_crown_proof_validate_zero_width_bounds_ok() {
    // Zero-width bounds (lower == upper) are valid (exact value known).
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![0.5],
        vec![0.5],
        vec![0.0],
        vec![0.0],
        1,
        true,
    );
    assert!(proof.validate().is_ok());
}

#[test]
fn test_crown_proof_validate_empty_bounds_ok() {
    // Empty bounds vectors are valid (scalar networks with no elements).
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![],
        vec![],
        vec![],
        vec![],
        0,
        true,
    );
    assert!(proof.validate().is_ok());
}

// ===========================================================================
// 4. Certificate composition (combining sub-network certificates)
// ===========================================================================

#[test]
fn test_crown_composition_proof_generation() {
    let proof = make_two_layer_proof();

    assert!(proof.method.is_composition());
    assert!(proof.has_composition_proof());
    assert!(proof.is_machine_checkable());
    assert_eq!(proof.layer_proof_count(), 2);

    // Composition Lean4 source should be present.
    assert!(proof.composition_lean4_source.is_some());
    assert!(proof.composition_theorem_name.is_some());
    assert_eq!(
        proof.composition_theorem_name.as_deref(),
        Some("crown_sound")
    );
}

#[test]
fn test_crown_composition_replay_verify_consistent_chain() {
    let proof = make_two_layer_proof();
    assert!(
        proof.replay_verify(),
        "consistent two-layer chain should pass replay"
    );
}

#[test]
fn test_crown_composition_replay_verify_gap_in_chain() {
    // Layer 1 output [0.0, 2.0] but layer 2 input [-3.0, 2.0] -- gap.
    let layers = vec![
        ConstructiveLayerRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_lower: vec![-1.0],
            input_upper: vec![1.0],
            output_lower: vec![0.0],
            output_upper: vec![2.0],
        },
        ConstructiveLayerRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_lower: vec![-3.0], // Below previous output lower
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
        "gap in bound chain should fail replay"
    );
}

#[test]
fn test_crown_composition_replay_verify_upper_gap() {
    // Layer 1 output upper [2.0] but layer 2 input upper [5.0] -- gap.
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
            input_upper: vec![5.0], // Above previous output upper
            output_lower: vec![0.0],
            output_upper: vec![5.0],
        },
    ];

    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::CrownComposition,
        vec![0.0],
        vec![5.0],
        vec![-1.0],
        vec![1.0],
        2,
        true,
    )
    .with_layer_proofs(layers);

    assert!(
        !proof.replay_verify(),
        "upper gap in chain should fail replay"
    );
}

#[test]
fn test_crown_composition_three_layer_chain() {
    let layers = vec![
        ConstructiveLayerRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_lower: vec![-1.0],
            input_upper: vec![1.0],
            output_lower: vec![-3.0],
            output_upper: vec![3.0],
        },
        ConstructiveLayerRecord {
            layer_index: 1,
            layer_type: "ReLU".to_string(),
            input_lower: vec![-3.0],
            input_upper: vec![3.0],
            output_lower: vec![0.0],
            output_upper: vec![3.0],
        },
        ConstructiveLayerRecord {
            layer_index: 2,
            layer_type: "Linear".to_string(),
            input_lower: vec![0.0],
            input_upper: vec![3.0],
            output_lower: vec![-1.5],
            output_upper: vec![1.5],
        },
    ];

    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::CrownComposition,
        vec![-1.5],
        vec![1.5],
        vec![-1.0],
        vec![1.0],
        3,
        true,
    )
    .with_layer_proofs(layers);

    assert!(proof.validate().is_ok());
    assert!(proof.replay_verify());
    assert_eq!(proof.layer_proof_count(), 3);
}

#[test]
fn test_crown_composition_different_dimension_layers() {
    // When consecutive layers have different output/input dimensions
    // (e.g., reshape or pooling), replay_verify skips the containment check.
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
            // After reshape: 2 -> 4 elements (dimension change).
            layer_type: "Reshape".to_string(),
            input_lower: vec![-2.0, -2.0, -2.0, -2.0],
            input_upper: vec![2.0, 2.0, 2.0, 2.0],
            output_lower: vec![-2.0, -2.0, -2.0, -2.0],
            output_upper: vec![2.0, 2.0, 2.0, 2.0],
        },
    ];

    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::CrownComposition,
        vec![-2.0, -2.0, -2.0, -2.0],
        vec![2.0, 2.0, 2.0, 2.0],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        2,
        true,
    )
    .with_layer_proofs(layers);

    // Dimension mismatch between layer 0 output (2) and layer 1 input (4)
    // should NOT cause replay failure -- skipped check.
    assert!(proof.validate().is_ok());
    assert!(proof.replay_verify());
}

// ===========================================================================
// 5. Certificate metadata (model name, timestamp, soundness, tightness)
// ===========================================================================

#[test]
fn test_crown_proof_generated_at_is_iso8601() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![0.0],
        vec![1.0],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );

    let ts = &proof.generated_at;
    assert!(ts.ends_with('Z'), "timestamp should end with Z: {ts}");
    assert_eq!(ts.len(), 20, "ISO 8601 should be 20 chars: {ts}");
    assert_eq!(&ts[4..5], "-");
    assert_eq!(&ts[7..8], "-");
    assert_eq!(&ts[10..11], "T");
}

#[test]
fn test_crown_method_tightness_classification() {
    // Per nn engineering rule #3340: Crown, AlphaCrown, BetaCrown, Analytical
    // are tight. IBP is not.
    assert!(ConstructiveProofMethod::Crown.is_tight());
    assert!(ConstructiveProofMethod::AlphaCrown.is_tight());
    assert!(ConstructiveProofMethod::BetaCrown.is_tight());
    assert!(ConstructiveProofMethod::Analytical.is_tight());
    assert!(ConstructiveProofMethod::CrownComposition.is_tight());
    assert!(ConstructiveProofMethod::AlphaCrownComposition.is_tight());
    assert!(ConstructiveProofMethod::BetaCrownComposition.is_tight());

    assert!(!ConstructiveProofMethod::Ibp.is_tight());
    assert!(!ConstructiveProofMethod::IbpComposition.is_tight());
}

#[test]
fn test_crown_method_composition_classification() {
    assert!(ConstructiveProofMethod::IbpComposition.is_composition());
    assert!(ConstructiveProofMethod::CrownComposition.is_composition());
    assert!(ConstructiveProofMethod::AlphaCrownComposition.is_composition());
    assert!(ConstructiveProofMethod::BetaCrownComposition.is_composition());

    assert!(!ConstructiveProofMethod::Ibp.is_composition());
    assert!(!ConstructiveProofMethod::Crown.is_composition());
    assert!(!ConstructiveProofMethod::AlphaCrown.is_composition());
    assert!(!ConstructiveProofMethod::BetaCrown.is_composition());
    assert!(!ConstructiveProofMethod::Analytical.is_composition());
}

#[test]
fn test_crown_machine_checkable_requires_verified_and_data() {
    // Verified + non-empty bounds = machine checkable.
    let proof_ok = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![0.0],
        vec![1.0],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );
    assert!(proof_ok.is_machine_checkable());

    // Not verified = not machine checkable.
    let proof_unverified = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![0.0],
        vec![1.0],
        vec![-1.0],
        vec![1.0],
        1,
        false,
    );
    assert!(!proof_unverified.is_machine_checkable());

    // Verified but empty output bounds and no lean4 = not machine checkable.
    let proof_empty = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![],
        vec![],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    );
    assert!(!proof_empty.is_machine_checkable());

    // Verified with lean4 but empty bounds = machine checkable.
    let proof_lean4 = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![],
        vec![],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    )
    .with_lean4_export("theorem t : True := trivial".to_string());
    assert!(proof_lean4.is_machine_checkable());
}

#[test]
fn test_crown_prop_method_mapping_consistency() {
    // All tight PropMethods should produce tight ConstructiveProofMethods.
    for method in [
        PropMethod::Crown,
        PropMethod::AlphaCrown,
        PropMethod::BetaCrown,
        PropMethod::Analytical,
    ] {
        let single = ConstructiveProofMethod::from_prop_method(method);
        assert!(
            single.is_tight(),
            "{method:?} -> {single:?} should be tight"
        );
        let comp = ConstructiveProofMethod::composition_from_prop_method(method);
        assert!(
            comp.is_tight(),
            "{method:?} -> {comp:?} composition should be tight"
        );
        assert!(
            comp.is_composition(),
            "{method:?} -> {comp:?} should be composition"
        );
    }

    // IBP single is not tight, composition is not tight.
    let ibp_single = ConstructiveProofMethod::from_prop_method(PropMethod::Ibp);
    assert!(!ibp_single.is_tight());
    let ibp_comp = ConstructiveProofMethod::composition_from_prop_method(PropMethod::Ibp);
    assert!(!ibp_comp.is_tight());
    assert!(ibp_comp.is_composition());
}

// ===========================================================================
// 6. Integration with the certify.rs pipeline
// ===========================================================================

// These tests exercise the full certify pipeline using trace_graph on
// simple models to verify end-to-end CROWN certificate generation.

#[cfg(feature = "ny")]
mod pipeline_integration {
    use super::*;
    use ny_api::BoundedTensor;
    use nn_core::dyn_tensor::trace::{record_input, trace_graph};
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::layers::{Linear, Module};
    use nn_core::Device;
    use ndarray::{ArrayD, IxDyn};

    use crate::certify::{certify_model, CertifyConfig};

    fn build_identity_model() -> (nn_core::dyn_tensor::trace::ComputationGraph, BoundedTensor) {
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

        (graph, input_bounds)
    }

    #[test]
    fn test_certify_pipeline_generates_constructive_proof() {
        let (graph, input_bounds) = build_identity_model();
        let config = CertifyConfig::new("pipeline_crown_test");
        let result = certify_model(&graph, &input_bounds, &config).unwrap();

        assert!(result.has_constructive_proof());
        let proof = result.constructive_proof().unwrap();
        assert!(proof.verified);
        assert!(proof.validate().is_ok());
        assert!(proof.is_machine_checkable());
    }

    #[test]
    fn test_certify_pipeline_proof_bounds_are_finite() {
        let (graph, input_bounds) = build_identity_model();
        let config = CertifyConfig::new("finite_bounds_test");
        let result = certify_model(&graph, &input_bounds, &config).unwrap();

        let proof = result.constructive_proof().unwrap();
        for &v in &proof.input_lower {
            assert!(v.is_finite(), "input lower should be finite: {v}");
        }
        for &v in &proof.input_upper {
            assert!(v.is_finite(), "input upper should be finite: {v}");
        }
        for &v in &proof.output_lower {
            assert!(v.is_finite(), "output lower should be finite: {v}");
        }
        for &v in &proof.output_upper {
            assert!(v.is_finite(), "output upper should be finite: {v}");
        }
    }

    #[test]
    fn test_certify_pipeline_proof_in_bundle() {
        let (graph, input_bounds) = build_identity_model();
        let config = CertifyConfig::new("bundle_integration");
        let result = certify_model(&graph, &input_bounds, &config).unwrap();

        // Proof should appear in both CertifyResult and the bundle certificate.
        let result_proof = result.constructive_proof().unwrap();
        let bundle_cert = &result.bundle.certificates[0];
        let bundle_proof = bundle_cert
            .constructive_proof
            .as_ref()
            .expect("bundle certificate should have constructive proof");

        assert_eq!(result_proof.method, bundle_proof.method);
        assert_eq!(result_proof.output_lower, bundle_proof.output_lower);
        assert_eq!(result_proof.output_upper, bundle_proof.output_upper);
        assert_eq!(result_proof.verified, bundle_proof.verified);
    }

    #[test]
    fn test_certify_pipeline_disabled_proof() {
        let (graph, input_bounds) = build_identity_model();
        let mut config = CertifyConfig::new("disabled_proof");
        config.generate_constructive_proof = false;
        let result = certify_model(&graph, &input_bounds, &config).unwrap();

        assert!(!result.has_constructive_proof());
        assert!(result.constructive_proof().is_none());

        let json = result.constructive_proof_json().unwrap();
        assert!(json.is_none());
    }

    #[test]
    fn test_certify_pipeline_replay_verification() {
        let (graph, input_bounds) = build_identity_model();
        let config = CertifyConfig::new("replay_test");
        let result = certify_model(&graph, &input_bounds, &config).unwrap();

        assert!(result.replay_verify_constructive_proof());
    }

    #[test]
    fn test_certify_pipeline_validate_constructive_proof() {
        let (graph, input_bounds) = build_identity_model();
        let config = CertifyConfig::new("validate_test");
        let result = certify_model(&graph, &input_bounds, &config).unwrap();

        let valid = result
            .validate_constructive_proof()
            .expect("validation should not error");
        assert!(valid, "pipeline proof should be valid");
    }

    #[test]
    fn test_certify_pipeline_save_constructive_proof_file() {
        let (graph, input_bounds) = build_identity_model();
        let config = CertifyConfig::new("save_file_test");
        let result = certify_model(&graph, &input_bounds, &config).unwrap();

        let dir = std::env::temp_dir().join("nn_crown_pipeline_save");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("constructive.json");

        let saved = result.save_constructive_proof(&path).unwrap();
        assert!(saved);
        assert!(path.exists());

        // Load and verify roundtrip.
        let loaded = ConstructiveProofData::load(&path).unwrap();
        let original = result.constructive_proof().unwrap();
        assert_eq!(original.method, loaded.method);
        assert_eq!(original.output_lower, loaded.output_lower);
        assert_eq!(original.output_upper, loaded.output_upper);
        assert!(loaded.validate().is_ok());
        assert!(loaded.replay_verify());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_certify_pipeline_bundle_save_load_with_proof() {
        let (graph, input_bounds) = build_identity_model();
        let config = CertifyConfig::new("full_bundle_roundtrip");
        let result = certify_model(&graph, &input_bounds, &config).unwrap();

        let dir = std::env::temp_dir().join("nn_crown_full_bundle");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bundle.proof.json");

        result.bundle.save(&path).unwrap();
        let loaded_bundle = CertificateBundle::load(&path).unwrap();

        assert_eq!(
            loaded_bundle.certificates.len(),
            result.bundle.certificates.len()
        );
        let loaded_cert = &loaded_bundle.certificates[0];
        assert!(loaded_cert.constructive_proof.is_some());

        let loaded_proof = loaded_cert.constructive_proof.as_ref().unwrap();
        let orig_proof = result.constructive_proof().unwrap();
        assert_eq!(loaded_proof.method, orig_proof.method);
        assert_eq!(loaded_proof.verified, orig_proof.verified);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}

// ===========================================================================
// TransformProofBundle integration (certifying compiler #4311)
// ===========================================================================

#[test]
fn test_transform_proof_bundle_creation() {
    let mut bundle = TransformProofBundle::new("kokoro_v1");
    assert_eq!(bundle.model_name, "kokoro_v1");
    assert_eq!(bundle.entries.len(), 0);
    assert_eq!(bundle.proved_count(), 0);
    assert!(!bundle.all_verified());

    let entry = TransformProofEntry::new(
        "NormActivConv1d fusion",
        TransformPass::NormActivConv1dFusion,
        -1e-6,
        1e-6,
        1e-5,
        PropMethod::Crown,
    );
    assert!(entry.is_proved());
    assert!(!entry.has_lean4_proof());

    bundle.push(entry);
    bundle.set_total_transforms(1);
    assert_eq!(bundle.proved_count(), 1);
    assert!(bundle.all_verified());
}

#[test]
fn test_transform_proof_bundle_json_roundtrip() {
    let mut bundle = TransformProofBundle::new("test_model");
    let entry = TransformProofEntry::new(
        "FusedResBlock",
        TransformPass::FusedResBlockWiring,
        -1e-7,
        1e-7,
        1e-5,
        PropMethod::AlphaCrown,
    )
    .with_lean4_proof("theorem t : True := trivial".to_string())
    .with_source_hash("a".repeat(64));

    bundle.push(entry);
    bundle.set_total_transforms(1);

    let json = bundle.to_json().unwrap();
    let loaded = TransformProofBundle::from_json(&json).unwrap();

    assert_eq!(loaded.model_name, "test_model");
    assert_eq!(loaded.entries.len(), 1);
    assert!(loaded.entries[0].has_lean4_proof());
    assert!(loaded.entries[0].is_proved());
    assert_eq!(loaded.proved_count(), 1);
    assert!(loaded.all_verified());
    assert_eq!(loaded.lean4_proof_count(), 1);
}

#[test]
fn test_transform_proof_entry_not_within_epsilon() {
    let entry = TransformProofEntry::new(
        "bad_fusion",
        TransformPass::Other,
        -0.1,
        0.1,
        1e-5, // epsilon much smaller than diff
        PropMethod::Ibp,
    );
    assert!(!entry.is_proved());
    assert!(!entry.within_epsilon);
    assert!((entry.max_abs_diff - 0.1).abs() < 1e-6);
}

#[test]
fn test_transform_proof_bundle_partial_verification() {
    let mut bundle = TransformProofBundle::new("partial_model");

    // One proved, one not.
    bundle.push(TransformProofEntry::new(
        "good",
        TransformPass::NamedFusion,
        -1e-7,
        1e-7,
        1e-5,
        PropMethod::Crown,
    ));
    bundle.push(TransformProofEntry::new(
        "bad",
        TransformPass::Other,
        -0.5,
        0.5,
        1e-5,
        PropMethod::Ibp,
    ));
    bundle.set_total_transforms(2);

    assert_eq!(bundle.proved_count(), 1);
    assert_eq!(bundle.unverified_count(), 1);
    assert!(!bundle.all_verified());
}
