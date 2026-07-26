// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for CROWN certificate wiring and verification pipeline
//! integration (#4315).
//!
//! Covers:
//! - ConstructiveProofData creation, validation, and replay verification
//! - ConstructiveProofMethod classification (tight vs loose, composition)
//! - JSON serialization roundtrip for constructive proof certificates
//! - Certificate composition from multi-layer pipeline stages
//! - Certificate metadata (timestamps, model hash, bound tightness)
//! - Soundness mode tracking (IBP vs CROWN vs AlphaCrown vs BetaCrown)
//! - Empty/degenerate certificates (zero-layer, single-layer)
//! - TransformProofBundle and TransformProofEntry wiring
//! - ConstructiveLayerRecord structural consistency
//! - ProofCertificate with constructive proof attachment
//! - File save/load roundtrip for constructive proofs

use crate::certificate::{
    CertificateBundle, ProofCertificate, CERTIFICATE_VERSION,
};
use crate::certificate_types::{
    ConstructiveLayerRecord, ConstructiveProofData, ConstructiveProofMethod,
    TransformPass, TransformProofBundle, TransformProofEntry,
};
use crate::status::{InputBoundsRecord, ParamInputRecord};
use crate::verify_types::{KernelVerification, PropMethod};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_verification(name: &str, lower: f32, upper: f32) -> KernelVerification {
    KernelVerification::new(
        name.to_string(),
        PropMethod::Crown,
        lower,
        upper,
        upper - lower,
        lower.is_finite() && upper.is_finite(),
    )
}

fn make_input_spec(lower: f32, upper: f32) -> InputBoundsRecord {
    InputBoundsRecord {
        variable_inputs: vec![ParamInputRecord {
            param_index: 0,
            lower,
            upper,
        }],
        constant_params: vec![1.0],
        input_shape: Some(vec![1]),
        input_range: Some((lower, upper)),
    }
}

fn make_constructive_proof(
    method: ConstructiveProofMethod,
    num_layers: usize,
) -> ConstructiveProofData {
    ConstructiveProofData::new(
        method,
        vec![-1.0, 0.0],  // output_lower
        vec![1.0, 2.0],   // output_upper
        vec![-5.0, -3.0],  // input_lower
        vec![5.0, 3.0],   // input_upper
        num_layers,
        true,
    )
}

fn make_layer_record(
    index: usize,
    layer_type: &str,
    in_lo: f32,
    in_hi: f32,
    out_lo: f32,
    out_hi: f32,
) -> ConstructiveLayerRecord {
    ConstructiveLayerRecord {
        layer_index: index,
        layer_type: layer_type.to_string(),
        input_lower: vec![in_lo],
        input_upper: vec![in_hi],
        output_lower: vec![out_lo],
        output_upper: vec![out_hi],
    }
}

// ===========================================================================
// 1. ConstructiveProofData creation and basic accessors
// ===========================================================================

#[test]
fn test_constructive_proof_new_basic_fields() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 5);
    assert_eq!(proof.method, ConstructiveProofMethod::Crown);
    assert_eq!(proof.output_lower, vec![-1.0, 0.0]);
    assert_eq!(proof.output_upper, vec![1.0, 2.0]);
    assert_eq!(proof.input_lower, vec![-5.0, -3.0]);
    assert_eq!(proof.input_upper, vec![5.0, 3.0]);
    assert_eq!(proof.num_layers, 5);
    assert!(proof.verified);
    assert!(!proof.generated_at.is_empty());
}

#[test]
fn test_constructive_proof_defaults_are_none() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Ibp, 1);
    assert!(proof.lean4_export.is_none());
    assert!(proof.layer_proofs.is_none());
    assert!(proof.composition_lean4_source.is_none());
    assert!(proof.composition_theorem_name.is_none());
}

#[test]
fn test_constructive_proof_is_machine_checkable_verified_with_bounds() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    assert!(proof.is_machine_checkable());
}

#[test]
fn test_constructive_proof_not_machine_checkable_when_unverified() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    proof.verified = false;
    assert!(!proof.is_machine_checkable());
}

#[test]
fn test_constructive_proof_machine_checkable_with_lean4() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3)
        .with_lean4_export("-- Lean4 proof".to_string());
    assert!(proof.is_machine_checkable());
    assert_eq!(proof.lean4_export.as_deref(), Some("-- Lean4 proof"));
}

#[test]
fn test_constructive_proof_machine_checkable_with_composition() {
    let proof = make_constructive_proof(ConstructiveProofMethod::CrownComposition, 3)
        .with_composition_proof(
            "theorem crown_composition_sound".to_string(),
            "crown_composition_sound".to_string(),
        );
    assert!(proof.is_machine_checkable());
    assert!(proof.has_composition_proof());
}

#[test]
fn test_constructive_proof_not_machine_checkable_empty_bounds_no_lean4() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![],  // empty output_lower
        vec![],  // empty output_upper
        vec![-1.0],
        vec![1.0],
        3,
        true,
    );
    // Empty output bounds AND no lean4 means not machine checkable.
    assert!(!proof.is_machine_checkable());
}

// ===========================================================================
// 2. ConstructiveProofMethod classification
// ===========================================================================

#[test]
fn test_constructive_method_tight_variants() {
    assert!(ConstructiveProofMethod::Crown.is_tight());
    assert!(ConstructiveProofMethod::AlphaCrown.is_tight());
    assert!(ConstructiveProofMethod::BetaCrown.is_tight());
    assert!(ConstructiveProofMethod::Analytical.is_tight());
    assert!(ConstructiveProofMethod::CrownComposition.is_tight());
    assert!(ConstructiveProofMethod::AlphaCrownComposition.is_tight());
    assert!(ConstructiveProofMethod::BetaCrownComposition.is_tight());
}

#[test]
fn test_constructive_method_loose_variants() {
    assert!(!ConstructiveProofMethod::Ibp.is_tight());
    assert!(!ConstructiveProofMethod::IbpComposition.is_tight());
}

#[test]
fn test_constructive_method_composition_variants() {
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
fn test_constructive_method_from_prop_method() {
    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::Ibp),
        ConstructiveProofMethod::Ibp,
    );
    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::Crown),
        ConstructiveProofMethod::Crown,
    );
    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::AlphaCrown),
        ConstructiveProofMethod::AlphaCrown,
    );
    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::BetaCrown),
        ConstructiveProofMethod::BetaCrown,
    );
    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::Analytical),
        ConstructiveProofMethod::Analytical,
    );
    assert_eq!(
        ConstructiveProofMethod::from_prop_method(PropMethod::MixedIbpCrown),
        ConstructiveProofMethod::Crown,
    );
}

#[test]
fn test_constructive_method_composition_from_prop_method() {
    assert_eq!(
        ConstructiveProofMethod::composition_from_prop_method(PropMethod::Ibp),
        ConstructiveProofMethod::IbpComposition,
    );
    assert_eq!(
        ConstructiveProofMethod::composition_from_prop_method(PropMethod::Crown),
        ConstructiveProofMethod::CrownComposition,
    );
    assert_eq!(
        ConstructiveProofMethod::composition_from_prop_method(PropMethod::AlphaCrown),
        ConstructiveProofMethod::AlphaCrownComposition,
    );
    assert_eq!(
        ConstructiveProofMethod::composition_from_prop_method(PropMethod::BetaCrown),
        ConstructiveProofMethod::BetaCrownComposition,
    );
    assert_eq!(
        ConstructiveProofMethod::composition_from_prop_method(PropMethod::Analytical),
        ConstructiveProofMethod::CrownComposition,
    );
    assert_eq!(
        ConstructiveProofMethod::composition_from_prop_method(PropMethod::MixedIbpCrown),
        ConstructiveProofMethod::CrownComposition,
    );
}

// ===========================================================================
// 3. ConstructiveProofData validation
// ===========================================================================

#[test]
fn test_constructive_proof_validate_valid() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    assert!(proof.validate().is_ok());
}

#[test]
fn test_constructive_proof_validate_input_length_mismatch() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    proof.input_lower = vec![-1.0, -2.0, -3.0];
    proof.input_upper = vec![1.0, 2.0]; // length mismatch
    let err = proof.validate().unwrap_err();
    assert!(err.contains("input bounds length mismatch"));
}

#[test]
fn test_constructive_proof_validate_output_length_mismatch() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    proof.output_lower = vec![-1.0];
    proof.output_upper = vec![1.0, 2.0]; // length mismatch
    let err = proof.validate().unwrap_err();
    assert!(err.contains("output bounds length mismatch"));
}

#[test]
fn test_constructive_proof_validate_non_finite_input() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    proof.input_lower[0] = f32::NAN;
    let err = proof.validate().unwrap_err();
    assert!(err.contains("non-finite input bound"));
}

#[test]
fn test_constructive_proof_validate_non_finite_output() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    proof.output_upper[1] = f32::INFINITY;
    let err = proof.validate().unwrap_err();
    assert!(err.contains("non-finite output bound"));
}

#[test]
fn test_constructive_proof_validate_inverted_input() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    proof.input_lower[0] = 10.0;
    proof.input_upper[0] = -10.0; // inverted
    let err = proof.validate().unwrap_err();
    assert!(err.contains("inverted input bound"));
}

#[test]
fn test_constructive_proof_validate_inverted_output() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    proof.output_lower[0] = 5.0;
    proof.output_upper[0] = -5.0; // inverted
    let err = proof.validate().unwrap_err();
    assert!(err.contains("inverted output bound"));
}

#[test]
fn test_constructive_proof_validate_layer_input_mismatch() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    let bad_layer = ConstructiveLayerRecord {
        layer_index: 0,
        layer_type: "Linear".to_string(),
        input_lower: vec![-1.0, -2.0],
        input_upper: vec![1.0],  // length mismatch with input_lower
        output_lower: vec![0.0],
        output_upper: vec![1.0],
    };
    proof = proof.with_layer_proofs(vec![bad_layer]);
    let err = proof.validate().unwrap_err();
    assert!(err.contains("layer[0] input bounds length mismatch"));
}

#[test]
fn test_constructive_proof_validate_layer_output_mismatch() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    let bad_layer = ConstructiveLayerRecord {
        layer_index: 0,
        layer_type: "ReLU".to_string(),
        input_lower: vec![-1.0],
        input_upper: vec![1.0],
        output_lower: vec![0.0, 0.5],
        output_upper: vec![1.0],  // length mismatch with output_lower
    };
    proof = proof.with_layer_proofs(vec![bad_layer]);
    let err = proof.validate().unwrap_err();
    assert!(err.contains("layer[0] output bounds length mismatch"));
}

// ===========================================================================
// 4. ConstructiveProofData replay verification
// ===========================================================================

#[test]
fn test_constructive_proof_replay_verify_no_layers() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    // No layer proofs, but verified=true, so replay returns self.verified.
    assert!(proof.replay_verify());
}

#[test]
fn test_constructive_proof_replay_verify_no_layers_unverified() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    proof.verified = false;
    // No layer proofs and verified=false.
    assert!(!proof.replay_verify());
}

#[test]
fn test_constructive_proof_replay_verify_consistent_chain() {
    let layers = vec![
        make_layer_record(0, "Linear", -5.0, 5.0, -2.0, 2.0),
        make_layer_record(1, "ReLU", -2.0, 2.0, 0.0, 2.0),
        make_layer_record(2, "Linear", 0.0, 2.0, -1.0, 1.0),
    ];
    let proof = make_constructive_proof(ConstructiveProofMethod::CrownComposition, 3)
        .with_layer_proofs(layers);
    assert!(proof.replay_verify());
}

#[test]
fn test_constructive_proof_replay_verify_broken_chain() {
    // Layer 1 output does not contain layer 2 input.
    let layers = vec![
        make_layer_record(0, "Linear", -5.0, 5.0, -2.0, 2.0),
        make_layer_record(1, "ReLU", -2.0, 2.0, 0.0, 1.0),
        // Input lower is -5.0 but prev output lower is 0.0 -- gap.
        make_layer_record(2, "Linear", -5.0, 1.0, -1.0, 1.0),
    ];
    let proof = make_constructive_proof(ConstructiveProofMethod::CrownComposition, 3)
        .with_layer_proofs(layers);
    assert!(!proof.replay_verify());
}

#[test]
fn test_constructive_proof_replay_verify_upper_bound_violation() {
    // Layer 1 output upper does not contain layer 2 input upper.
    let layers = vec![
        make_layer_record(0, "Linear", -1.0, 1.0, -0.5, 0.5),
        // Next layer input upper (3.0) exceeds prev output upper (0.5).
        make_layer_record(1, "ReLU", -0.5, 3.0, 0.0, 0.5),
    ];
    let proof = make_constructive_proof(ConstructiveProofMethod::CrownComposition, 2)
        .with_layer_proofs(layers);
    assert!(!proof.replay_verify());
}

#[test]
fn test_constructive_proof_replay_verify_structural_failure() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    proof.input_lower = vec![f32::NAN]; // structural validation will fail
    proof.input_upper = vec![1.0];
    assert!(!proof.replay_verify());
}

#[test]
fn test_constructive_proof_replay_verify_dimension_mismatch_skips_check() {
    // Different layer sizes are allowed (reshaping between layers).
    let layers = vec![
        ConstructiveLayerRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_lower: vec![-1.0, -1.0],
            input_upper: vec![1.0, 1.0],
            output_lower: vec![-0.5, -0.5],
            output_upper: vec![0.5, 0.5],
        },
        ConstructiveLayerRecord {
            layer_index: 1,
            layer_type: "Reshape".to_string(),
            input_lower: vec![-0.5],  // different dim from prev output
            input_upper: vec![0.5],
            output_lower: vec![-0.3],
            output_upper: vec![0.3],
        },
    ];
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 2)
        .with_layer_proofs(layers);
    // Dimension mismatch causes the chain check to be skipped, not failed.
    assert!(proof.replay_verify());
}

// ===========================================================================
// 5. JSON serialization roundtrip
// ===========================================================================

#[test]
fn test_constructive_proof_json_roundtrip_basic() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 5);
    let json = proof.to_json().expect("serialize");
    let deserialized = ConstructiveProofData::from_json(&json).expect("deserialize");
    assert_eq!(deserialized.method, ConstructiveProofMethod::Crown);
    assert_eq!(deserialized.output_lower, proof.output_lower);
    assert_eq!(deserialized.output_upper, proof.output_upper);
    assert_eq!(deserialized.input_lower, proof.input_lower);
    assert_eq!(deserialized.input_upper, proof.input_upper);
    assert_eq!(deserialized.num_layers, 5);
    assert!(deserialized.verified);
}

#[test]
fn test_constructive_proof_json_roundtrip_with_layers() {
    let layers = vec![
        make_layer_record(0, "Linear", -1.0, 1.0, -0.5, 0.5),
        make_layer_record(1, "ReLU", -0.5, 0.5, 0.0, 0.5),
    ];
    let proof = make_constructive_proof(ConstructiveProofMethod::AlphaCrown, 2)
        .with_layer_proofs(layers);
    let json = proof.to_json().expect("serialize");
    let deserialized = ConstructiveProofData::from_json(&json).expect("deserialize");
    assert_eq!(deserialized.layer_proof_count(), 2);
    let lp = deserialized.layer_proofs.as_ref().unwrap();
    assert_eq!(lp[0].layer_type, "Linear");
    assert_eq!(lp[1].layer_type, "ReLU");
    assert_eq!(lp[0].output_lower, vec![-0.5]);
    assert_eq!(lp[1].output_lower, vec![0.0]);
}

#[test]
fn test_constructive_proof_json_roundtrip_with_lean4() {
    let proof = make_constructive_proof(ConstructiveProofMethod::BetaCrown, 3)
        .with_lean4_export("-- Lean4 theorem proof_sound".to_string());
    let json = proof.to_json().expect("serialize");
    let deserialized = ConstructiveProofData::from_json(&json).expect("deserialize");
    assert_eq!(
        deserialized.lean4_export.as_deref(),
        Some("-- Lean4 theorem proof_sound"),
    );
}

#[test]
fn test_constructive_proof_json_roundtrip_with_composition() {
    let proof = make_constructive_proof(ConstructiveProofMethod::CrownComposition, 4)
        .with_composition_proof(
            "theorem crown_comp : ...".to_string(),
            "crown_comp".to_string(),
        );
    let json = proof.to_json().expect("serialize");
    let deserialized = ConstructiveProofData::from_json(&json).expect("deserialize");
    assert!(deserialized.has_composition_proof());
    assert_eq!(
        deserialized.composition_lean4_source.as_deref(),
        Some("theorem crown_comp : ..."),
    );
    assert_eq!(
        deserialized.composition_theorem_name.as_deref(),
        Some("crown_comp"),
    );
}

#[test]
fn test_constructive_proof_json_roundtrip_all_methods() {
    let methods = [
        ConstructiveProofMethod::Ibp,
        ConstructiveProofMethod::Crown,
        ConstructiveProofMethod::AlphaCrown,
        ConstructiveProofMethod::BetaCrown,
        ConstructiveProofMethod::Analytical,
        ConstructiveProofMethod::IbpComposition,
        ConstructiveProofMethod::CrownComposition,
        ConstructiveProofMethod::AlphaCrownComposition,
        ConstructiveProofMethod::BetaCrownComposition,
    ];
    for method in &methods {
        let proof = make_constructive_proof(*method, 1);
        let json = proof.to_json().expect("serialize");
        let back = ConstructiveProofData::from_json(&json).expect("deserialize");
        assert_eq!(back.method, *method, "roundtrip failed for {method:?}");
    }
}

#[test]
fn test_constructive_proof_method_serde_roundtrip() {
    let methods = [
        ConstructiveProofMethod::Ibp,
        ConstructiveProofMethod::Crown,
        ConstructiveProofMethod::AlphaCrown,
        ConstructiveProofMethod::BetaCrown,
        ConstructiveProofMethod::Analytical,
        ConstructiveProofMethod::IbpComposition,
        ConstructiveProofMethod::CrownComposition,
        ConstructiveProofMethod::AlphaCrownComposition,
        ConstructiveProofMethod::BetaCrownComposition,
    ];
    for method in &methods {
        let json = serde_json::to_string(method).expect("serialize");
        let back: ConstructiveProofMethod = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*method, back, "serde roundtrip failed for {method:?}");
    }
}

// ===========================================================================
// 6. Certificate composition from multiple pipeline stages
// ===========================================================================

#[test]
fn test_constructive_proof_with_layer_proofs_count() {
    let layers = vec![
        make_layer_record(0, "Linear", -1.0, 1.0, -0.5, 0.5),
        make_layer_record(1, "ReLU", -0.5, 0.5, 0.0, 0.5),
        make_layer_record(2, "Linear", 0.0, 0.5, -0.2, 0.3),
    ];
    let proof = make_constructive_proof(ConstructiveProofMethod::CrownComposition, 3)
        .with_layer_proofs(layers);
    assert_eq!(proof.layer_proof_count(), 3);
}

#[test]
fn test_constructive_proof_no_layer_proofs_count_zero() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Ibp, 5);
    assert_eq!(proof.layer_proof_count(), 0);
}

#[test]
fn test_constructive_proof_composition_flag_without_proof() {
    let proof = make_constructive_proof(ConstructiveProofMethod::CrownComposition, 3);
    assert!(!proof.has_composition_proof());
}

#[test]
fn test_constructive_proof_multi_stage_composition() {
    // Simulate a 5-layer pipeline with per-layer proofs and composition.
    let layers: Vec<ConstructiveLayerRecord> = (0..5)
        .map(|i| {
            let scale = 1.0 / (i as f32 + 1.0);
            make_layer_record(
                i,
                if i % 2 == 0 { "Linear" } else { "ReLU" },
                -5.0 * scale,
                5.0 * scale,
                -3.0 * scale,
                3.0 * scale,
            )
        })
        .collect();

    let proof = make_constructive_proof(ConstructiveProofMethod::CrownComposition, 5)
        .with_layer_proofs(layers)
        .with_composition_proof(
            "theorem multi_stage_sound := by ...".to_string(),
            "multi_stage_sound".to_string(),
        );

    assert_eq!(proof.layer_proof_count(), 5);
    assert!(proof.has_composition_proof());
    assert!(proof.is_machine_checkable());
    assert!(proof.validate().is_ok());
}

// ===========================================================================
// 7. Certificate metadata: timestamps and generated_at
// ===========================================================================

#[test]
fn test_constructive_proof_timestamp_is_iso8601() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 1);
    // ISO 8601 timestamps contain 'T' separator and digits.
    assert!(
        proof.generated_at.contains('T'),
        "generated_at should be ISO 8601: {}",
        proof.generated_at,
    );
    assert!(
        proof.generated_at.len() >= 19,
        "ISO 8601 minimum length is 19: {}",
        proof.generated_at,
    );
}

#[test]
fn test_constructive_proof_timestamp_preserved_in_json() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 1);
    let original_ts = proof.generated_at.clone();
    let json = proof.to_json().expect("serialize");
    let back = ConstructiveProofData::from_json(&json).expect("deserialize");
    assert_eq!(back.generated_at, original_ts);
}

// ===========================================================================
// 8. Soundness mode tracking through constructive proof method
// ===========================================================================

#[test]
fn test_ibp_method_is_not_tight() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Ibp, 3);
    assert!(!proof.method.is_tight());
}

#[test]
fn test_crown_method_is_tight() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    assert!(proof.method.is_tight());
}

#[test]
fn test_alpha_crown_method_is_tight() {
    let proof = make_constructive_proof(ConstructiveProofMethod::AlphaCrown, 3);
    assert!(proof.method.is_tight());
}

#[test]
fn test_beta_crown_method_is_tight() {
    let proof = make_constructive_proof(ConstructiveProofMethod::BetaCrown, 3);
    assert!(proof.method.is_tight());
}

#[test]
fn test_analytical_method_is_tight() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Analytical, 3);
    assert!(proof.method.is_tight());
}

#[test]
fn test_ibp_composition_method_is_not_tight() {
    let proof = make_constructive_proof(ConstructiveProofMethod::IbpComposition, 3);
    assert!(!proof.method.is_tight());
}

#[test]
fn test_crown_composition_method_is_tight() {
    let proof = make_constructive_proof(ConstructiveProofMethod::CrownComposition, 3);
    assert!(proof.method.is_tight());
}

// ===========================================================================
// 9. Empty/degenerate certificates
// ===========================================================================

#[test]
fn test_constructive_proof_zero_layers() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![0.0],
        vec![1.0],
        vec![-1.0],
        vec![1.0],
        0,  // zero layers
        true,
    );
    assert_eq!(proof.num_layers, 0);
    assert!(proof.validate().is_ok());
    // Machine checkable because verified=true and has output bounds.
    assert!(proof.is_machine_checkable());
}

#[test]
fn test_constructive_proof_single_layer() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Crown,
        vec![-0.5],
        vec![0.5],
        vec![-1.0],
        vec![1.0],
        1,
        true,
    )
    .with_layer_proofs(vec![make_layer_record(0, "Linear", -1.0, 1.0, -0.5, 0.5)]);

    assert_eq!(proof.num_layers, 1);
    assert_eq!(proof.layer_proof_count(), 1);
    assert!(proof.validate().is_ok());
    assert!(proof.replay_verify());
}

#[test]
fn test_constructive_proof_empty_bounds() {
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Ibp,
        vec![],  // empty
        vec![],
        vec![],
        vec![],
        0,
        true,
    );
    assert!(proof.validate().is_ok());
    // Not machine checkable: verified but empty bounds and no lean4.
    assert!(!proof.is_machine_checkable());
}

#[test]
fn test_constructive_proof_empty_layer_proofs() {
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3)
        .with_layer_proofs(vec![]);
    assert_eq!(proof.layer_proof_count(), 0);
    assert!(proof.validate().is_ok());
    // Replay with empty layer proofs: returns verified flag since no chain to check.
    assert!(proof.replay_verify());
}

#[test]
fn test_constructive_proof_point_bounds() {
    // Lower == upper is valid (zero-width interval).
    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::Analytical,
        vec![0.5],
        vec![0.5],  // point bound
        vec![0.0],
        vec![0.0],  // point bound
        1,
        true,
    );
    assert!(proof.validate().is_ok());
    assert!(proof.replay_verify());
}

// ===========================================================================
// 10. File save/load roundtrip
// ===========================================================================

#[test]
fn test_constructive_proof_save_load_roundtrip() {
    let layers = vec![
        make_layer_record(0, "Linear", -1.0, 1.0, -0.5, 0.5),
        make_layer_record(1, "ReLU", -0.5, 0.5, 0.0, 0.5),
    ];
    let proof = make_constructive_proof(ConstructiveProofMethod::CrownComposition, 2)
        .with_layer_proofs(layers)
        .with_lean4_export("-- proof goes here".to_string())
        .with_composition_proof("theorem t : ...".to_string(), "t".to_string());

    let dir = std::env::temp_dir().join("nn_crown_cert_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_proof.json");

    proof.save(&path).expect("save");
    let loaded = ConstructiveProofData::load(&path).expect("load");

    assert_eq!(loaded.method, proof.method);
    assert_eq!(loaded.output_lower, proof.output_lower);
    assert_eq!(loaded.output_upper, proof.output_upper);
    assert_eq!(loaded.input_lower, proof.input_lower);
    assert_eq!(loaded.input_upper, proof.input_upper);
    assert_eq!(loaded.num_layers, proof.num_layers);
    assert_eq!(loaded.verified, proof.verified);
    assert_eq!(loaded.lean4_export, proof.lean4_export);
    assert_eq!(loaded.layer_proof_count(), 2);
    assert_eq!(loaded.composition_lean4_source, proof.composition_lean4_source);
    assert_eq!(loaded.composition_theorem_name, proof.composition_theorem_name);

    // Cleanup.
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

// ===========================================================================
// 11. ProofCertificate with constructive proof attachment
// ===========================================================================

#[test]
fn test_certificate_with_constructive_proof() {
    let result = make_verification("crown_kernel", -1.0, 1.0);
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 3);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-5.0, 5.0))
        .with_constructive_proof(proof);

    assert!(cert.has_constructive_proof());
    assert!(cert.constructive_proof.is_some());
    let cp = cert.constructive_proof.as_ref().unwrap();
    assert_eq!(cp.method, ConstructiveProofMethod::Crown);
    assert_eq!(cp.num_layers, 3);
}

#[test]
fn test_certificate_without_constructive_proof() {
    let result = make_verification("plain_kernel", -1.0, 1.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-5.0, 5.0));
    assert!(!cert.has_constructive_proof());
    assert!(cert.constructive_proof.is_none());
}

#[test]
fn test_certificate_json_roundtrip_with_constructive_proof() {
    let result = make_verification("rt_kernel", -2.0, 3.0);
    let proof = make_constructive_proof(ConstructiveProofMethod::AlphaCrown, 4)
        .with_lean4_export("-- lean4 export".to_string());
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-5.0, 5.0))
        .with_constructive_proof(proof);

    let json = cert.to_json().expect("serialize");
    let back: ProofCertificate = serde_json::from_str(&json).expect("deserialize");
    assert!(back.has_constructive_proof());
    let cp = back.constructive_proof.as_ref().unwrap();
    assert_eq!(cp.method, ConstructiveProofMethod::AlphaCrown);
    assert_eq!(cp.num_layers, 4);
    assert_eq!(cp.lean4_export.as_deref(), Some("-- lean4 export"));
}

#[test]
fn test_certificate_bundle_with_constructive_proofs() {
    let c1 = ProofCertificate::from_verification(
        &make_verification("k1", -1.0, 1.0),
        make_input_spec(-1.0, 1.0),
    )
    .with_constructive_proof(make_constructive_proof(ConstructiveProofMethod::Crown, 2));

    let c2 = ProofCertificate::from_verification(
        &make_verification("k2", -2.0, 2.0),
        make_input_spec(-2.0, 2.0),
    ); // no constructive proof

    let bundle = CertificateBundle::new("mixed_model")
        .with_certificate(c1)
        .with_certificate(c2);

    assert_eq!(bundle.len(), 2);
    assert!(bundle.certificates[0].has_constructive_proof());
    assert!(!bundle.certificates[1].has_constructive_proof());
    assert!(bundle.validate_all().is_ok());
}

// ===========================================================================
// 12. TransformProofEntry wiring
// ===========================================================================

#[test]
fn test_transform_proof_entry_new() {
    let entry = TransformProofEntry::new(
        "FusedResBlock wiring",
        TransformPass::FusedResBlockWiring,
        -0.001,
        0.002,
        1e-3,
        PropMethod::Crown,
    );
    assert_eq!(entry.transform_name, "FusedResBlock wiring");
    assert_eq!(entry.pass_id, TransformPass::FusedResBlockWiring);
    assert_eq!(entry.diff_lower, -0.001);
    assert_eq!(entry.diff_upper, 0.002);
    assert!((entry.max_abs_diff - 0.002).abs() < 1e-9);
    assert_eq!(entry.epsilon, 1e-3);
    assert!(!entry.within_epsilon, "0.002 > 0.001 should not be within epsilon");
    assert!(!entry.is_proved());
    assert!(!entry.has_lean4_proof());
}

#[test]
fn test_transform_proof_entry_within_epsilon() {
    let entry = TransformProofEntry::new(
        "Style projection absorption",
        TransformPass::StyleProjectionAbsorption,
        -0.0001,
        0.0002,
        1e-3,
        PropMethod::Ibp,
    );
    assert!(entry.within_epsilon, "0.0002 <= 0.001 should be within epsilon");
    assert!(entry.is_proved());
}

#[test]
fn test_transform_proof_entry_with_lean4() {
    let entry = TransformProofEntry::new(
        "named fusion",
        TransformPass::NamedFusion,
        -0.0001,
        0.0001,
        1e-3,
        PropMethod::Crown,
    )
    .with_lean4_proof("-- lean4 proof term".to_string());

    assert!(entry.has_lean4_proof());
    assert_eq!(
        entry.lean4_proof_term.as_deref(),
        Some("-- lean4 proof term"),
    );
}

#[test]
fn test_transform_proof_entry_with_source_hash() {
    let entry = TransformProofEntry::new(
        "NormActivConv1d",
        TransformPass::NormActivConv1dFusion,
        0.0,
        0.0,
        1e-5,
        PropMethod::Analytical,
    )
    .with_source_hash("a".repeat(64));

    assert_eq!(entry.source_hash.as_deref(), Some("a".repeat(64).as_str()));
}

#[test]
fn test_transform_proof_entry_serde_roundtrip() {
    let entry = TransformProofEntry::new(
        "Batched style projection",
        TransformPass::BatchedStyleProjection,
        -1e-5,
        1e-5,
        1e-4,
        PropMethod::AlphaCrown,
    )
    .with_lean4_proof("-- lean4".to_string())
    .with_source_hash("b".repeat(64));

    let json = serde_json::to_string_pretty(&entry).expect("serialize");
    let back: TransformProofEntry = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.transform_name, "Batched style projection");
    assert_eq!(back.pass_id, TransformPass::BatchedStyleProjection);
    assert!(back.within_epsilon);
    assert!(back.has_lean4_proof());
    assert_eq!(back.source_hash, entry.source_hash);
}

// ===========================================================================
// 13. TransformProofBundle wiring
// ===========================================================================

#[test]
fn test_transform_proof_bundle_new_empty() {
    let bundle = TransformProofBundle::new("kokoro");
    assert_eq!(bundle.model_name, "kokoro");
    assert!(bundle.entries.is_empty());
    assert_eq!(bundle.total_transforms, 0);
    assert_eq!(bundle.proved_count(), 0);
    assert_eq!(bundle.unverified_count(), 0);
    assert!(!bundle.all_verified());
    assert_eq!(bundle.lean4_proof_count(), 0);
}

#[test]
fn test_transform_proof_bundle_push_and_count() {
    let mut bundle = TransformProofBundle::new("kokoro");
    bundle.set_total_transforms(3);

    bundle.push(TransformProofEntry::new(
        "pass1",
        TransformPass::NormActivConv1dFusion,
        -0.0001,
        0.0001,
        1e-3,
        PropMethod::Crown,
    ));
    bundle.push(TransformProofEntry::new(
        "pass2",
        TransformPass::FusedResBlockWiring,
        -0.0001,
        0.0001,
        1e-3,
        PropMethod::Crown,
    ));
    bundle.push(TransformProofEntry::new(
        "pass3",
        TransformPass::StyleProjectionAbsorption,
        -0.5, // too large
        0.5,
        1e-3,
        PropMethod::Ibp,
    ));

    assert_eq!(bundle.proved_count(), 2);
    assert_eq!(bundle.unverified_count(), 1);
    assert!(!bundle.all_verified());
}

#[test]
fn test_transform_proof_bundle_all_verified() {
    let mut bundle = TransformProofBundle::new("test_model");
    bundle.set_total_transforms(2);

    bundle.push(TransformProofEntry::new(
        "t1",
        TransformPass::NamedFusion,
        0.0,
        0.0,
        1e-5,
        PropMethod::Analytical,
    ));
    bundle.push(TransformProofEntry::new(
        "t2",
        TransformPass::Other,
        -1e-6,
        1e-6,
        1e-5,
        PropMethod::Crown,
    ));

    assert!(bundle.all_verified());
}

#[test]
fn test_transform_proof_bundle_lean4_count() {
    let mut bundle = TransformProofBundle::new("test");
    bundle.set_total_transforms(2);
    bundle.push(
        TransformProofEntry::new("a", TransformPass::Other, 0.0, 0.0, 1e-5, PropMethod::Crown)
            .with_lean4_proof("-- lean4".to_string()),
    );
    bundle.push(TransformProofEntry::new(
        "b",
        TransformPass::Other,
        0.0,
        0.0,
        1e-5,
        PropMethod::Ibp,
    ));
    assert_eq!(bundle.lean4_proof_count(), 1);
}

#[test]
fn test_transform_proof_bundle_json_roundtrip() {
    let mut bundle = TransformProofBundle::new("roundtrip_model");
    bundle.set_total_transforms(1);
    bundle.push(TransformProofEntry::new(
        "fusion",
        TransformPass::NamedFusion,
        -1e-6,
        1e-6,
        1e-5,
        PropMethod::Crown,
    ));

    let json = bundle.to_json().expect("serialize");
    let back = TransformProofBundle::from_json(&json).expect("deserialize");
    assert_eq!(back.model_name, "roundtrip_model");
    assert_eq!(back.total_transforms, 1);
    assert_eq!(back.entries.len(), 1);
    assert_eq!(back.entries[0].transform_name, "fusion");
    assert!(back.all_verified());
}

// ===========================================================================
// 14. TransformPass variants
// ===========================================================================

#[test]
fn test_transform_pass_serde_roundtrip() {
    let passes = [
        TransformPass::NormActivConv1dFusion,
        TransformPass::FusedResBlockWiring,
        TransformPass::StyleProjectionAbsorption,
        TransformPass::BatchedStyleProjection,
        TransformPass::NamedFusion,
        TransformPass::Other,
    ];
    for pass in &passes {
        let json = serde_json::to_string(pass).expect("serialize");
        let back: TransformPass = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*pass, back, "roundtrip failed for {pass:?}");
    }
}

// ===========================================================================
// 15. ConstructiveLayerRecord field consistency
// ===========================================================================

#[test]
fn test_constructive_layer_record_basic() {
    let rec = make_layer_record(0, "Linear", -1.0, 1.0, -0.5, 0.5);
    assert_eq!(rec.layer_index, 0);
    assert_eq!(rec.layer_type, "Linear");
    assert_eq!(rec.input_lower, vec![-1.0]);
    assert_eq!(rec.input_upper, vec![1.0]);
    assert_eq!(rec.output_lower, vec![-0.5]);
    assert_eq!(rec.output_upper, vec![0.5]);
}

#[test]
fn test_constructive_layer_record_serde_roundtrip() {
    let rec = ConstructiveLayerRecord {
        layer_index: 3,
        layer_type: "Conv1d".to_string(),
        input_lower: vec![-1.0, -2.0, -3.0],
        input_upper: vec![1.0, 2.0, 3.0],
        output_lower: vec![-0.5, -1.0],
        output_upper: vec![0.5, 1.0],
    };
    let json = serde_json::to_string(&rec).expect("serialize");
    let back: ConstructiveLayerRecord = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.layer_index, 3);
    assert_eq!(back.layer_type, "Conv1d");
    assert_eq!(back.input_lower.len(), 3);
    assert_eq!(back.output_lower.len(), 2);
}

// ===========================================================================
// 16. Edge case: NaN/Inf constructive proof bounds
// ===========================================================================

#[test]
fn test_constructive_proof_nan_in_output_lower_fails_validation() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 1);
    proof.output_lower[0] = f32::NAN;
    assert!(proof.validate().is_err());
}

#[test]
fn test_constructive_proof_inf_in_input_upper_fails_validation() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 1);
    proof.input_upper[0] = f32::INFINITY;
    assert!(proof.validate().is_err());
}

#[test]
fn test_constructive_proof_neg_inf_in_output_fails_validation() {
    let mut proof = make_constructive_proof(ConstructiveProofMethod::Crown, 1);
    proof.output_lower[0] = f32::NEG_INFINITY;
    assert!(proof.validate().is_err());
}

// ===========================================================================
// 17. Version and format consistency
// ===========================================================================

#[test]
fn test_certificate_version_supports_constructive_proof() {
    // v6 added constructive_proof field.
    assert!(CERTIFICATE_VERSION >= 6);
}

#[test]
fn test_certificate_with_constructive_proof_validates_at_current_version() {
    let result = make_verification("versioned", -1.0, 1.0);
    let proof = make_constructive_proof(ConstructiveProofMethod::Crown, 2);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0))
        .with_constructive_proof(proof);
    assert_eq!(cert.version, CERTIFICATE_VERSION);
    assert!(cert.validate().is_ok());
}

// ===========================================================================
// 18. Large constructive proof (stress test)
// ===========================================================================

#[test]
fn test_constructive_proof_many_layers() {
    let n = 100;
    let layers: Vec<ConstructiveLayerRecord> = (0..n)
        .map(|i| {
            let lo = -(n as f32 - i as f32);
            let hi = n as f32 - i as f32;
            let out_lo = lo * 0.9;
            let out_hi = hi * 0.9;
            make_layer_record(
                i,
                if i % 3 == 0 {
                    "Linear"
                } else if i % 3 == 1 {
                    "ReLU"
                } else {
                    "Conv1d"
                },
                lo,
                hi,
                out_lo,
                out_hi,
            )
        })
        .collect();

    let proof = ConstructiveProofData::new(
        ConstructiveProofMethod::CrownComposition,
        vec![-1.0],
        vec![1.0],
        vec![-(n as f32)],
        vec![n as f32],
        n,
        true,
    )
    .with_layer_proofs(layers);

    assert_eq!(proof.layer_proof_count(), n);
    assert!(proof.validate().is_ok());

    // JSON roundtrip of the large proof.
    let json = proof.to_json().expect("serialize large proof");
    let back = ConstructiveProofData::from_json(&json).expect("deserialize large proof");
    assert_eq!(back.layer_proof_count(), n);
}

// ===========================================================================
// 19. Backward compatibility: None fields deserialize cleanly
// ===========================================================================

#[test]
fn test_constructive_proof_backward_compat_missing_optional_fields() {
    // Simulate a v6 JSON without the optional composition fields.
    let json = r#"{
        "method": "Crown",
        "output_lower": [-1.0],
        "output_upper": [1.0],
        "input_lower": [-5.0],
        "input_upper": [5.0],
        "num_layers": 3,
        "verified": true,
        "generated_at": "2026-04-01T00:00:00Z"
    }"#;
    let proof = ConstructiveProofData::from_json(json).expect("deserialize partial JSON");
    assert_eq!(proof.method, ConstructiveProofMethod::Crown);
    assert!(proof.lean4_export.is_none());
    assert!(proof.layer_proofs.is_none());
    assert!(proof.composition_lean4_source.is_none());
    assert!(proof.composition_theorem_name.is_none());
    assert!(proof.validate().is_ok());
}

#[test]
fn test_proof_certificate_backward_compat_no_constructive() {
    // Simulate a pre-v6 certificate JSON without constructive_proof.
    let result = make_verification("legacy", -1.0, 1.0);
    let cert = ProofCertificate::from_verification(&result, make_input_spec(-1.0, 1.0));
    let json = cert.to_json().expect("serialize");

    // Verify constructive_proof is skipped from JSON (skip_serializing_if).
    assert!(
        !json.contains("constructive_proof"),
        "None constructive_proof should be skipped in serialization",
    );

    let back: ProofCertificate = serde_json::from_str(&json).expect("deserialize");
    assert!(!back.has_constructive_proof());
}
