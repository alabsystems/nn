// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Core status roundtrip and record tests: serialization, IBP/Crown outcomes,
//! multi-variable inputs, crown_error JSON omission.

use super::status_test_helpers::{scalar_output_bounds, single_input_bounds};
use super::*;
use crate::verify_input::ScalarInputBounds;

#[test]
fn test_status_roundtrip_verified_ibp() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "snake".to_string(),
        KernelStatus {
            status: VerifyOutcome::Verified,
            method: PropMethod::Ibp,
            input_bounds: single_input_bounds(-10.0, 10.0, vec![1.0]),
            output_bounds: scalar_output_bounds(-10.0, 11.0),
            output_width: 21.0,
            crown_error: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: None,
            crown_ibp_ratio: None,
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength: None,
        },
    );

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.kernels.len(), 1);
    assert_eq!(
        deserialized.kernels["snake"].status,
        VerifyOutcome::Verified
    );
    assert_eq!(deserialized.kernels["snake"].method, PropMethod::Ibp);
    assert_eq!(deserialized.kernels["snake"].crown_error, None);
}

#[test]
fn test_status_roundtrip_bounds_computed_crown() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "wide_kernel".to_string(),
        KernelStatus {
            status: VerifyOutcome::BoundsComputed,
            method: PropMethod::Crown,
            input_bounds: single_input_bounds(-100.0, 100.0, vec![0.5, 2.0]),
            output_bounds: scalar_output_bounds(-1e30, 1e30),
            output_width: 2e30,
            crown_error: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: None,
            crown_ibp_ratio: None,
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength: None,
        },
    );

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(
        deserialized.kernels["wide_kernel"].status,
        VerifyOutcome::BoundsComputed
    );
    assert_eq!(
        deserialized.kernels["wide_kernel"].method,
        PropMethod::Crown
    );
}

#[test]
fn test_status_roundtrip_failed() {
    let mut status = VerifyStatus::default();
    status
        .record_failure(
            "bad_kernel",
            PropMethod::Ibp,
            ScalarInputBounds::new(-1.0, 1.0).expect("valid test bounds"),
            &[0.5],
        )
        .expect("record failure");

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(
        deserialized.kernels["bad_kernel"].status,
        VerifyOutcome::Failed
    );
    assert_eq!(deserialized.kernels["bad_kernel"].output_bounds.lower, 0.0);
    assert_eq!(deserialized.kernels["bad_kernel"].output_width, 0.0);
    assert_eq!(deserialized.kernels["bad_kernel"].crown_error, None);
}

#[test]
fn test_status_roundtrip_ibp_fallback_with_crown_error() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "tricky_kernel".to_string(),
        KernelStatus {
            status: VerifyOutcome::IbpFallback,
            method: PropMethod::Ibp,
            input_bounds: single_input_bounds(-10.0, 10.0, vec![1.0]),
            output_bounds: scalar_output_bounds(-50.0, 50.0),
            output_width: 100.0,
            crown_error: Some("unsupported layer combination".to_string()),
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: None,
            crown_ibp_ratio: None,
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength: None,
        },
    );

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    assert!(json.contains("crown_error"));
    assert!(json.contains("unsupported layer combination"));
    assert!(json.contains("ibp_fallback"));

    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");
    let k = &deserialized.kernels["tricky_kernel"];
    assert_eq!(k.status, VerifyOutcome::IbpFallback);
    assert_eq!(k.method, PropMethod::Ibp);
    assert_eq!(
        k.crown_error.as_deref(),
        Some("unsupported layer combination")
    );
}

#[test]
fn test_record_sets_ibp_fallback_when_crown_failed() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "fallback_test".to_string(),
        method: PropMethod::Ibp,
        output_lower: -5.0,
        output_upper: 5.0,
        output_width: 10.0,
        is_finite: true,
        crown_fallback_reason: Some("CROWN solver timeout".to_string()),
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[1.0],
            None,
        )
        .expect("record");

    let k = &status.kernels["fallback_test"];
    assert_eq!(k.status, VerifyOutcome::IbpFallback);
    assert_eq!(k.crown_error.as_deref(), Some("CROWN solver timeout"));
}

#[test]
fn test_record_sets_bounds_computed_when_crown_failed_and_ibp_non_finite() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "nonfinite_fallback".to_string(),
        method: PropMethod::Ibp,
        output_lower: f32::NEG_INFINITY,
        output_upper: f32::INFINITY,
        output_width: f32::INFINITY,
        is_finite: false,
        crown_fallback_reason: Some("CROWN diverged".to_string()),
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[1.0],
            None,
        )
        .expect("record");

    let k = &status.kernels["nonfinite_fallback"];
    assert_eq!(
        k.status,
        VerifyOutcome::BoundsComputed,
        "non-finite IBP fallback should be BoundsComputed, not IbpFallback"
    );
    assert_eq!(k.crown_error.as_deref(), Some("CROWN diverged"));
}

#[test]
fn test_record_sets_verified_when_no_crown_fallback() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "clean_test".to_string(),
        method: PropMethod::Ibp,
        output_lower: -5.0,
        output_upper: 5.0,
        output_width: 10.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[1.0],
            None,
        )
        .expect("record");

    let k = &status.kernels["clean_test"];
    assert_eq!(k.status, VerifyOutcome::Verified);
    assert_eq!(k.crown_error, None);
}

#[test]
fn test_status_roundtrip_multi_variable_inputs() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "add".to_string(),
        method: PropMethod::Ibp,
        output_lower: -3.0,
        output_upper: 3.0,
        output_width: 6.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };
    let variable_inputs = vec![
        ParamInputRecord {
            param_index: 0,
            lower: -1.0,
            upper: 1.0,
        },
        ParamInputRecord {
            param_index: 1,
            lower: -2.0,
            upper: 2.0,
        },
    ];

    status
        .record_with_variable_inputs(&result, &variable_inputs, &[], None, None)
        .expect("record");

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    assert!(json.contains("variable_inputs"));
    assert!(json.contains("param_index"));
    assert!(!json.contains("\"input_range\""));

    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");
    let entry = &deserialized.kernels["add"].input_bounds;
    assert_eq!(entry.variable_inputs, variable_inputs);
    assert_eq!(entry.input_shape, Some(vec![2]));
    assert_eq!(entry.input_range, None);
}

#[test]
fn test_record_single_variable_nonzero_param_omits_legacy_input_range() {
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "scaled".to_string(),
        method: PropMethod::Ibp,
        output_lower: -2.0,
        output_upper: 2.0,
        output_width: 4.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };
    let variable_inputs = vec![ParamInputRecord {
        param_index: 1,
        lower: -1.0,
        upper: 1.0,
    }];

    status
        .record_with_variable_inputs(&result, &variable_inputs, &[2.0], None, None)
        .expect("record");

    let entry = &status.kernels["scaled"].input_bounds;
    assert_eq!(entry.variable_inputs, variable_inputs);
    assert_eq!(entry.input_shape, Some(vec![1]));
    assert_eq!(entry.input_range, None);

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    assert!(!json.contains("\"input_range\""));
}

#[test]
fn test_record_failure_with_variable_inputs_roundtrip() {
    let mut status = VerifyStatus::default();
    let variable_inputs = vec![
        ParamInputRecord {
            param_index: 0,
            lower: -5.0,
            upper: 5.0,
        },
        ParamInputRecord {
            param_index: 2,
            lower: 0.0,
            upper: 1.0,
        },
    ];

    status
        .record_failure_with_variable_inputs(
            "failing_multi",
            PropMethod::Ibp,
            &variable_inputs,
            &[3.15],
            None,
        )
        .expect("record failure");

    let k = &status.kernels["failing_multi"];
    assert_eq!(k.status, VerifyOutcome::Failed);
    assert_eq!(k.method, PropMethod::Ibp);
    assert_eq!(k.output_bounds.lower, 0.0, "failure sentinel");
    assert_eq!(k.output_bounds.upper, 0.0, "failure sentinel");
    assert_eq!(k.output_width, 0.0, "failure sentinel");
    assert_eq!(k.crown_error, None);
    assert_eq!(k.input_bounds.variable_inputs, variable_inputs);
    assert_eq!(k.input_bounds.constant_params, vec![3.15]);
    assert_eq!(k.input_bounds.input_shape, Some(vec![2]));
    assert_eq!(k.input_bounds.input_range, None);

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");
    let k2 = &deserialized.kernels["failing_multi"];
    assert_eq!(k2.status, VerifyOutcome::Failed);
    assert_eq!(k2.input_bounds.variable_inputs, variable_inputs);
}

#[test]
fn test_crown_error_omitted_from_json_when_none() {
    let status_entry = KernelStatus {
        status: VerifyOutcome::Verified,
        method: PropMethod::Ibp,
        input_bounds: InputBoundsRecord {
            variable_inputs: vec![ParamInputRecord {
                param_index: 0,
                lower: -1.0,
                upper: 1.0,
            }],
            constant_params: vec![],
            input_shape: Some(vec![1]),
            input_range: Some((-1.0, 1.0)),
        },
        output_bounds: OutputBoundsRecord {
            lower: -1.0,
            upper: 1.0,
            tensor_lower: None,
            tensor_upper: None,
            shape: None,
            is_infeasible: false,
        },
        output_width: 2.0,
        crown_error: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        smt: None,
        crown_coverage: None,
        ibp_comparison_width: None,
        crown_ibp_ratio: None,
        weight_artifact: None,
        soundness_justification: None,
        stale: false,
        stale_reason: None,
        proof_strength: None,
    };

    let json = serde_json::to_string(&status_entry).expect("serialize");
    assert!(!json.contains("crown_error"));
}

#[test]
fn test_record_crown_comparison() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "crown_kernel".to_string(),
        KernelStatus {
            status: VerifyOutcome::Verified,
            method: PropMethod::Crown,
            input_bounds: single_input_bounds(-1.0, 1.0, vec![]),
            output_bounds: scalar_output_bounds(-0.5, 0.5),
            output_width: 1.0,
            crown_error: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: None,
            crown_ibp_ratio: None,
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength: None,
        },
    );
    // Add matching history entry
    status.history.insert(
        "crown_kernel".to_string(),
        vec![status.kernels["crown_kernel"].clone()],
    );

    // Record IBP comparison: IBP width=2.0, CROWN width=1.0 → ratio=0.5
    status
        .record_crown_comparison("crown_kernel", 2.0)
        .expect("record comparison");

    let entry = &status.kernels["crown_kernel"];
    assert_eq!(entry.ibp_comparison_width, Some(2.0));
    assert_eq!(entry.crown_ibp_ratio, Some(0.5));

    // History should also be updated
    let hist = &status.history["crown_kernel"];
    assert_eq!(hist.last().unwrap().ibp_comparison_width, Some(2.0));
    assert_eq!(hist.last().unwrap().crown_ibp_ratio, Some(0.5));
}

#[test]
fn test_record_crown_comparison_missing_kernel() {
    let mut status = VerifyStatus::default();
    let err = status
        .record_crown_comparison("nonexistent", 2.0)
        .unwrap_err();
    assert!(err.to_string().contains("no kernel entry"));
}

#[test]
fn test_record_crown_comparison_non_finite_ibp_width() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "k".to_string(),
        KernelStatus {
            status: VerifyOutcome::Verified,
            method: PropMethod::Crown,
            input_bounds: single_input_bounds(-1.0, 1.0, vec![]),
            output_bounds: scalar_output_bounds(-1.0, 1.0),
            output_width: 2.0,
            crown_error: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: None,
            crown_ibp_ratio: None,
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength: None,
        },
    );
    let err = status.record_crown_comparison("k", f32::NAN).unwrap_err();
    assert!(err.to_string().contains("non-finite"));
}

#[test]
fn test_record_crown_comparison_rejects_ibp_method() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "ibp_kernel".to_string(),
        KernelStatus {
            status: VerifyOutcome::Verified,
            method: PropMethod::Ibp,
            input_bounds: single_input_bounds(-1.0, 1.0, vec![]),
            output_bounds: scalar_output_bounds(-1.0, 1.0),
            output_width: 2.0,
            crown_error: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: None,
            crown_ibp_ratio: None,
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength: None,
        },
    );
    let err = status
        .record_crown_comparison("ibp_kernel", 2.0)
        .unwrap_err();
    assert!(
        err.to_string().contains("not CROWN"),
        "should reject IBP entries: {err}"
    );
}

// F9: record_crown_comparison must accept CROWN-family methods, not just Crown.
#[test]
fn test_record_crown_comparison_accepts_alpha_crown() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "alpha_kernel".to_string(),
        KernelStatus {
            status: VerifyOutcome::Verified,
            method: PropMethod::AlphaCrown,
            input_bounds: single_input_bounds(-1.0, 1.0, vec![]),
            output_bounds: scalar_output_bounds(-0.5, 0.5),
            output_width: 1.0,
            crown_error: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: None,
            crown_ibp_ratio: None,
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength: None,
        },
    );
    status
        .record_crown_comparison("alpha_kernel", 2.0)
        .expect("AlphaCrown entries should be accepted for IBP comparison");
    let entry = &status.kernels["alpha_kernel"];
    assert_eq!(entry.ibp_comparison_width, Some(2.0));
    assert!(entry.crown_ibp_ratio.is_some());
}

#[test]
fn test_record_crown_comparison_rejects_mixed_ibp_crown() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "mixed_kernel".to_string(),
        KernelStatus {
            status: VerifyOutcome::Verified,
            method: PropMethod::MixedIbpCrown,
            input_bounds: single_input_bounds(-1.0, 1.0, vec![]),
            output_bounds: scalar_output_bounds(-1.0, 1.0),
            output_width: 2.0,
            crown_error: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: None,
            crown_ibp_ratio: None,
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength: None,
        },
    );
    let err = status
        .record_crown_comparison("mixed_kernel", 2.0)
        .unwrap_err();
    assert!(
        err.to_string().contains("not CROWN"),
        "MixedIbpCrown should be rejected: {err}"
    );
}

#[test]
fn test_crown_comparison_report() {
    let mut status = VerifyStatus::default();

    // CROWN entry with comparison: ratio 0.5 (CROWN tighter)
    let mut crown_entry = KernelStatus {
        status: VerifyOutcome::Verified,
        method: PropMethod::Crown,
        input_bounds: single_input_bounds(-1.0, 1.0, vec![]),
        output_bounds: scalar_output_bounds(-0.5, 0.5),
        output_width: 1.0,
        crown_error: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        smt: None,
        crown_coverage: None,
        ibp_comparison_width: Some(2.0),
        crown_ibp_ratio: Some(0.5),
        weight_artifact: None,
        soundness_justification: None,
        stale: false,
        stale_reason: None,
        proof_strength: None,
    };
    status
        .kernels
        .insert("tight_crown".to_string(), crown_entry.clone());

    // CROWN entry with ratio 1.0 (no improvement)
    crown_entry.output_width = 2.0;
    crown_entry.ibp_comparison_width = Some(2.0);
    crown_entry.crown_ibp_ratio = Some(1.0);
    status
        .kernels
        .insert("vacuous_crown".to_string(), crown_entry);

    // IBP-only entry (no comparison data)
    status.kernels.insert(
        "ibp_only".to_string(),
        KernelStatus {
            status: VerifyOutcome::Verified,
            method: PropMethod::Ibp,
            input_bounds: single_input_bounds(-1.0, 1.0, vec![]),
            output_bounds: scalar_output_bounds(-1.0, 1.0),
            output_width: 2.0,
            crown_error: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: None,
            crown_ibp_ratio: None,
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength: None,
        },
    );

    let (crown_count, tighter_count, entries) = status.crown_comparison_report();
    assert_eq!(crown_count, 2);
    assert_eq!(tighter_count, 1); // only tight_crown has ratio < 1.0
                                  // Sorted by ratio ascending
    assert_eq!(entries[0].0, "tight_crown");
    assert!((entries[0].1 - 0.5).abs() < 1e-6);
    assert_eq!(entries[1].0, "vacuous_crown");
    assert!((entries[1].1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_crown_comparison_fields_roundtrip_json() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "k".to_string(),
        KernelStatus {
            status: VerifyOutcome::Verified,
            method: PropMethod::Crown,
            input_bounds: single_input_bounds(-1.0, 1.0, vec![]),
            output_bounds: scalar_output_bounds(-0.5, 0.5),
            output_width: 1.0,
            crown_error: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: Some(2.0),
            crown_ibp_ratio: Some(0.5),
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength: None,
        },
    );

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    assert!(json.contains("ibp_comparison_width"));
    assert!(json.contains("crown_ibp_ratio"));

    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");
    let k = &deserialized.kernels["k"];
    assert_eq!(k.ibp_comparison_width, Some(2.0));
    assert_eq!(k.crown_ibp_ratio, Some(0.5));
}

#[test]
fn test_crown_comparison_fields_omitted_when_none() {
    let entry = KernelStatus {
        status: VerifyOutcome::Verified,
        method: PropMethod::Ibp,
        input_bounds: single_input_bounds(-1.0, 1.0, vec![]),
        output_bounds: scalar_output_bounds(-1.0, 1.0),
        output_width: 2.0,
        crown_error: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        smt: None,
        crown_coverage: None,
        ibp_comparison_width: None,
        crown_ibp_ratio: None,
        weight_artifact: None,
        soundness_justification: None,
        stale: false,
        stale_reason: None,
        proof_strength: None,
    };
    let json = serde_json::to_string(&entry).expect("serialize");
    assert!(!json.contains("ibp_comparison_width"));
    assert!(!json.contains("crown_ibp_ratio"));
}

#[test]
fn test_crown_comparison_fields_deserialize_from_legacy_json() {
    // Legacy JSON without new fields should deserialize with None values
    let json = r#"{
        "status": "verified",
        "method": "IBP",
        "input_bounds": {
            "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
            "constant_params": []
        },
        "output_bounds": {"lower": -1.0, "upper": 1.0},
        "output_width": 2.0
    }"#;
    let entry: KernelStatus = serde_json::from_str(json).expect("deserialize legacy");
    assert_eq!(entry.ibp_comparison_width, None);
    assert_eq!(entry.crown_ibp_ratio, None);
}

#[test]
fn test_record_with_explicit_input_shape_records_actual_shape() {
    // #2637: when input_shape is Some, the actual tensor shape is recorded
    // instead of the degenerate [variable_inputs.len()] fallback.
    let mut status = VerifyStatus::default();
    let result = KernelVerification {
        kernel_name: "decoder".to_string(),
        method: PropMethod::Ibp,
        output_lower: -1.0,
        output_upper: 1.0,
        output_width: 2.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };
    let variable_inputs = vec![ParamInputRecord {
        param_index: 0,
        lower: -1.0,
        upper: 1.0,
    }];

    // Pass actual multi-dimensional shape [8, 128] via Some.
    status
        .record_with_variable_inputs(&result, &variable_inputs, &[], None, Some(&[8, 128]))
        .expect("record");

    let entry = &status.kernels["decoder"].input_bounds;
    assert_eq!(
        entry.input_shape,
        Some(vec![8, 128]),
        "explicit input_shape should be recorded verbatim, not as [variable_inputs.len()]"
    );

    // Roundtrip through JSON to verify serialization.
    let json = serde_json::to_string_pretty(&status).expect("serialize");
    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");
    let rt_entry = &deserialized.kernels["decoder"].input_bounds;
    assert_eq!(rt_entry.input_shape, Some(vec![8, 128]));
}
