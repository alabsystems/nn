// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `EditCertificate` data format.

use super::*;

const HASH_A: &str = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2";
const HASH_B: &str = "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5";

fn sample_weight() -> EditedWeight {
    EditedWeight {
        layer_name: "transformer.h.4.mlp.c_proj".to_string(),
        edit_type: EditType::Rank1Update,
        delta_norm: 0.042,
        delta_rank: Some(1),
    }
}

fn sample_cert() -> EditCertificate {
    EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Ibp)
        .with_edited_weight(sample_weight())
}

// --- Construction & builder ---

#[test]
fn test_new_certificate_defaults() {
    let cert = EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Ibp);
    assert_eq!(cert.original_model_hash, HASH_A);
    assert_eq!(cert.edited_model_hash, HASH_B);
    assert!(cert.edited_weights.is_empty());
    assert!(cert.target_bounds.is_none());
    assert!(cert.preservation_bounds.is_none());
    assert!(cert.kani_status.is_none());
    assert!(!cert.verified_at.is_empty());
    assert_eq!(cert.prop_method, PropMethod::Ibp);
    assert_eq!(cert.soundness_mode, VerificationSoundnessMode::Heuristic);
}

#[test]
fn test_with_edited_weight_appends() {
    let cert = EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Ibp)
        .with_edited_weight(sample_weight())
        .with_edited_weight(EditedWeight {
            layer_name: "transformer.h.5.attn.c_attn".to_string(),
            edit_type: EditType::LoraOverlay,
            delta_norm: 0.1,
            delta_rank: Some(4),
        });
    assert_eq!(cert.edited_weights.len(), 2);
    assert_eq!(
        cert.edited_weights[0].layer_name,
        "transformer.h.4.mlp.c_proj"
    );
    assert_eq!(
        cert.edited_weights[1].layer_name,
        "transformer.h.5.attn.c_attn"
    );
    cert.validate()
        .expect("appended weights cert should pass validate()");
}

#[test]
fn test_with_edited_weights_replaces() {
    let cert = EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Ibp)
        .with_edited_weight(sample_weight())
        .with_edited_weights(vec![EditedWeight {
            layer_name: "layer.index()".to_string(),
            edit_type: EditType::DirectWrite,
            delta_norm: 1.0,
            delta_rank: None,
        }]);
    assert_eq!(cert.edited_weights.len(), 1);
    assert_eq!(cert.edited_weights[0].layer_name, "layer.index()");
    cert.validate()
        .expect("replaced weights cert should pass validate()");
}

#[test]
fn test_with_target_bounds() {
    let bounds = OutputBoundsRecord {
        lower: -1.0,
        upper: 1.0,
        tensor_lower: None,
        tensor_upper: None,
        shape: None,
        is_infeasible: false,
    };
    let cert = sample_cert().with_target_bounds(bounds.clone());
    assert_eq!(cert.target_bounds, Some(bounds));
}

#[test]
fn test_with_preservation_bounds() {
    let bounds = OutputBoundsRecord {
        lower: 0.0,
        upper: 10.0,
        tensor_lower: None,
        tensor_upper: None,
        shape: None,
        is_infeasible: false,
    };
    let cert = sample_cert().with_preservation_bounds(bounds.clone());
    assert_eq!(cert.preservation_bounds, Some(bounds));
}

#[test]
fn test_with_soundness_mode() {
    let cert = sample_cert().with_soundness_mode(VerificationSoundnessMode::Sound);
    assert_eq!(cert.soundness_mode, VerificationSoundnessMode::Sound);
}

#[test]
fn test_crown_prop_method() {
    let cert = EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Crown)
        .with_edited_weight(sample_weight());
    assert_eq!(cert.prop_method, PropMethod::Crown);
}

// --- Validation ---

#[test]
fn test_validate_valid_cert() {
    let cert = sample_cert();
    assert!(cert.validate().is_ok());
}

#[test]
fn test_validate_invalid_original_hash() {
    let cert = EditCertificate::new("bad_hash".into(), HASH_B.into(), PropMethod::Ibp)
        .with_edited_weight(sample_weight());
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("original_model_hash"), "got: {msg}");
}

#[test]
fn test_validate_invalid_edited_hash() {
    let cert = EditCertificate::new(HASH_A.into(), "short".into(), PropMethod::Ibp)
        .with_edited_weight(sample_weight());
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("edited_model_hash"), "got: {msg}");
}

#[test]
fn test_validate_empty_weights() {
    let cert = EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Ibp);
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("empty"), "got: {msg}");
}

#[test]
fn test_validate_nan_delta_norm() {
    let cert = EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Ibp)
        .with_edited_weight(EditedWeight {
            layer_name: "layer.index()".to_string(),
            edit_type: EditType::DirectWrite,
            delta_norm: f32::NAN,
            delta_rank: None,
        });
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("delta_norm"), "got: {msg}");
}

#[test]
fn test_validate_inf_delta_norm() {
    let cert = EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Ibp)
        .with_edited_weight(EditedWeight {
            layer_name: "layer.index()".to_string(),
            edit_type: EditType::DirectWrite,
            delta_norm: f32::INFINITY,
            delta_rank: None,
        });
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("delta_norm"),
        "inf delta_norm error should mention delta_norm, got: {msg}"
    );
}

#[test]
fn test_validate_negative_delta_norm() {
    let cert = EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Ibp)
        .with_edited_weight(EditedWeight {
            layer_name: "layer.index()".to_string(),
            edit_type: EditType::DirectWrite,
            delta_norm: -0.001,
            delta_rank: None,
        });
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("delta_norm"),
        "negative delta_norm error should mention delta_norm, got: {msg}"
    );
}

#[test]
fn test_validate_zero_delta_norm_ok() {
    let cert = EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Ibp)
        .with_edited_weight(EditedWeight {
            layer_name: "layer.index()".to_string(),
            edit_type: EditType::DirectWrite,
            delta_norm: 0.0,
            delta_rank: None,
        });
    assert!(cert.validate().is_ok());
}

#[test]
fn test_validate_inverted_target_bounds() {
    let cert = sample_cert().with_target_bounds(OutputBoundsRecord {
        lower: 5.0,
        upper: 1.0,
        tensor_lower: None,
        tensor_upper: None,
        shape: None,
        is_infeasible: false,
    });
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("target_bounds"), "got: {msg}");
}

#[test]
fn test_validate_inverted_preservation_bounds() {
    let cert = sample_cert().with_preservation_bounds(OutputBoundsRecord {
        lower: 10.0,
        upper: 2.0,
        tensor_lower: None,
        tensor_upper: None,
        shape: None,
        is_infeasible: false,
    });
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("preservation_bounds"), "got: {msg}");
}

// --- IEEE 754 NaN bypass regression tests ---

#[test]
fn test_validate_nan_target_bounds_rejected() {
    // IEEE 754: NaN > NaN returns false, so without is_finite() guard,
    // NaN bounds silently pass validation. This test catches the bypass.
    let cert = sample_cert().with_target_bounds(OutputBoundsRecord {
        lower: f32::NAN,
        upper: f32::NAN,
        tensor_lower: None,
        tensor_upper: None,
        shape: None,
        is_infeasible: false,
    });
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite"),
        "NaN bounds must be rejected, got: {msg}"
    );
}

#[test]
fn test_validate_nan_preservation_bounds_rejected() {
    let cert = sample_cert().with_preservation_bounds(OutputBoundsRecord {
        lower: f32::NAN,
        upper: 1.0,
        tensor_lower: None,
        tensor_upper: None,
        shape: None,
        is_infeasible: false,
    });
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite"),
        "NaN bounds must be rejected, got: {msg}"
    );
}

#[test]
fn test_validate_inf_target_bounds_rejected() {
    let cert = sample_cert().with_target_bounds(OutputBoundsRecord {
        lower: f32::NEG_INFINITY,
        upper: f32::INFINITY,
        tensor_lower: None,
        tensor_upper: None,
        shape: None,
        is_infeasible: false,
    });
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite"),
        "Inf bounds must be rejected, got: {msg}"
    );
}

#[test]
fn test_validate_inf_preservation_bounds_rejected() {
    let cert = sample_cert().with_preservation_bounds(OutputBoundsRecord {
        lower: f32::NEG_INFINITY,
        upper: f32::INFINITY,
        tensor_lower: None,
        tensor_upper: None,
        shape: None,
        is_infeasible: false,
    });
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite"),
        "Inf bounds must be rejected, got: {msg}"
    );
}

// --- Serde round-trip ---

#[test]
fn test_json_round_trip_minimal() {
    let cert = sample_cert();
    let json = cert.to_json().expect("serialize");
    let deserialized: EditCertificate = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.original_model_hash, HASH_A);
    assert_eq!(deserialized.edited_model_hash, HASH_B);
    assert_eq!(deserialized.edited_weights.len(), 1);
    assert_eq!(deserialized.prop_method, PropMethod::Ibp);
}

#[test]
fn test_json_round_trip_with_bounds() {
    let cert = sample_cert()
        .with_target_bounds(OutputBoundsRecord {
            lower: -1.0,
            upper: 1.0,
            tensor_lower: None,
            tensor_upper: None,
            shape: None,
            is_infeasible: false,
        })
        .with_preservation_bounds(OutputBoundsRecord {
            lower: 0.0,
            upper: 5.0,
            tensor_lower: None,
            tensor_upper: None,
            shape: None,
            is_infeasible: false,
        });
    let json = cert.to_json().expect("serialize");
    let deserialized: EditCertificate = serde_json::from_str(&json).expect("deserialize");
    assert!(deserialized.target_bounds.is_some());
    assert!(deserialized.preservation_bounds.is_some());
    let tb = deserialized.target_bounds.unwrap();
    assert_eq!(tb.lower, -1.0);
    assert_eq!(tb.upper, 1.0);
}

#[test]
fn test_json_omits_none_fields() {
    let cert = sample_cert();
    let json = cert.to_json().expect("serialize");
    assert!(
        !json.contains("target_bounds"),
        "None fields should be omitted"
    );
    assert!(!json.contains("preservation_bounds"));
    assert!(!json.contains("kani_status"));
    // delta_rank is Some(1) in sample_weight(), so it IS present in JSON.
    // Only check truly-None fields here.
}

#[test]
fn test_json_includes_present_optional_fields() {
    let cert = sample_cert(); // sample_weight has delta_rank: Some(1)
    let json = cert.to_json().expect("serialize");
    assert!(json.contains("delta_rank"), "Some fields should be present");
}

#[test]
fn test_deserialized_defaults_soundness_mode() {
    // JSON without soundness_mode should default to Heuristic
    let json = r#"{
        "original_model_hash": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "edited_model_hash": "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5",
        "edited_weights": [{
            "layer_name": "layer.index()",
            "edit_type": "DirectWrite",
            "delta_norm": 1.0
        }],
        "verified_at": "12345Z",
        "prop_method": "IBP"
    }"#;
    let cert: EditCertificate = serde_json::from_str(json).expect("deserialize");
    assert_eq!(cert.soundness_mode, VerificationSoundnessMode::Heuristic);
}

// --- Display ---

#[test]
fn test_edit_type_display() {
    assert_eq!(EditType::Rank1Update.to_string(), "rank1_update");
    assert_eq!(EditType::LoraOverlay.to_string(), "lora_overlay");
    assert_eq!(EditType::DirectWrite.to_string(), "direct_write");
    assert_eq!(EditType::GradientStep.to_string(), "gradient_step");
}

// --- Edge cases ---

#[test]
fn test_multiple_weights_all_valid() {
    let cert = EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Crown)
        .with_edited_weight(EditedWeight {
            layer_name: "a".to_string(),
            edit_type: EditType::Rank1Update,
            delta_norm: 0.01,
            delta_rank: Some(1),
        })
        .with_edited_weight(EditedWeight {
            layer_name: "b".to_string(),
            edit_type: EditType::LoraOverlay,
            delta_norm: 0.5,
            delta_rank: Some(4),
        })
        .with_edited_weight(EditedWeight {
            layer_name: "c".to_string(),
            edit_type: EditType::GradientStep,
            delta_norm: 0.001,
            delta_rank: None,
        });
    assert!(cert.validate().is_ok());
}

#[test]
fn test_second_weight_invalid_catches() {
    let cert = EditCertificate::new(HASH_A.into(), HASH_B.into(), PropMethod::Ibp)
        .with_edited_weight(EditedWeight {
            layer_name: "good".to_string(),
            edit_type: EditType::DirectWrite,
            delta_norm: 1.0,
            delta_rank: None,
        })
        .with_edited_weight(EditedWeight {
            layer_name: "bad".to_string(),
            edit_type: EditType::DirectWrite,
            delta_norm: f32::NAN,
            delta_rank: None,
        });
    let err = cert.validate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("bad"), "should name the bad layer, got: {msg}");
}

#[test]
fn test_clone_equality() {
    let cert = sample_cert();
    let cloned = cert.clone();
    assert_eq!(cert, cloned);
}
