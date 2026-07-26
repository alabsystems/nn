// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `verify_types.rs` — verification config and result types.

use super::*;
use crate::error::VerifyError;
use ny_api::BoundedTensor;
use ny_core::VerificationSoundnessMode;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// VerifyConfig::with_threshold
// ---------------------------------------------------------------------------

#[test]
fn test_with_threshold_valid() {
    let config = VerifyConfig::with_threshold(100.0).unwrap();
    assert_eq!(config.escalation_threshold(), 100.0);
    assert!(!config.require_sound());
}

#[test]
fn test_with_threshold_zero() {
    // Zero is a valid threshold (immediate escalation).
    let config = VerifyConfig::with_threshold(0.0).unwrap();
    assert_eq!(config.escalation_threshold(), 0.0);
}

#[test]
fn test_with_threshold_rejects_negative() {
    let err = VerifyConfig::with_threshold(-1.0).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidThreshold { value } if value == -1.0));
}

#[test]
fn test_with_threshold_rejects_nan() {
    let err = VerifyConfig::with_threshold(f32::NAN).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidThreshold { .. }));
}

#[test]
fn test_with_threshold_rejects_positive_infinity() {
    let err = VerifyConfig::with_threshold(f32::INFINITY).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidThreshold { .. }));
}

#[test]
fn test_with_threshold_rejects_negative_infinity() {
    let err = VerifyConfig::with_threshold(f32::NEG_INFINITY).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidThreshold { .. }));
}

// ---------------------------------------------------------------------------
// VerifyConfig::with_require_sound
// ---------------------------------------------------------------------------

#[test]
fn test_with_require_sound_true() {
    let config = VerifyConfig::with_threshold(1.0)
        .unwrap()
        .with_require_sound(true);
    assert!(config.require_sound());
    assert_eq!(config.escalation_threshold(), 1.0);
}

#[test]
fn test_with_require_sound_false() {
    let config = VerifyConfig::with_threshold(1.0)
        .unwrap()
        .with_require_sound(false);
    assert!(!config.require_sound());
}

// ---------------------------------------------------------------------------
// VerifyConfig::default
// ---------------------------------------------------------------------------

#[test]
fn test_config_default_threshold() {
    let config = VerifyConfig::default();
    assert_eq!(config.escalation_threshold(), DEFAULT_ESCALATION_THRESHOLD);
    assert!(!config.require_sound());
}

#[test]
fn test_default_escalation_threshold_is_positive() {
    let threshold = DEFAULT_ESCALATION_THRESHOLD;
    assert!(threshold > 0.0);
    assert!(threshold.is_finite());
}

// ---------------------------------------------------------------------------
// default_soundness_mode
// ---------------------------------------------------------------------------

#[test]
fn test_default_soundness_mode_is_heuristic() {
    assert_eq!(
        default_soundness_mode(),
        VerificationSoundnessMode::Heuristic
    );
}

// ---------------------------------------------------------------------------
// OutputTensorBounds::from_bounded_tensor
// ---------------------------------------------------------------------------

#[test]
fn test_output_tensor_bounds_finite_values() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let bt = BoundedTensor::new(lower, upper).unwrap();

    let otb = OutputTensorBounds::from_bounded_tensor(&bt);
    assert_eq!(otb.lower, vec![-1.0, 0.0, 0.5]);
    assert_eq!(otb.upper, vec![1.0, 2.0, 3.0]);
    assert_eq!(otb.shape, vec![3]);
    assert_eq!(otb.finite_mask, vec![true, true, true]);
}

#[test]
fn test_output_tensor_bounds_infinite_replaced_with_zero() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, -1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, f32::INFINITY]).unwrap();
    let bt = BoundedTensor::new_allow_infinite(lower, upper).unwrap();

    let otb = OutputTensorBounds::from_bounded_tensor(&bt);
    // Non-finite values replaced with 0.0 by finite_or.
    assert_eq!(otb.lower[0], 0.0);
    assert_eq!(otb.lower[1], -1.0);
    assert_eq!(otb.upper[0], 1.0);
    assert_eq!(otb.upper[1], 0.0);
    // finite_mask marks which elements were originally finite.
    assert_eq!(otb.finite_mask, vec![false, false]);
}

#[test]
fn test_output_tensor_bounds_shape_preserved() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0; 6]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
    let bt = BoundedTensor::new(lower, upper).unwrap();

    let otb = OutputTensorBounds::from_bounded_tensor(&bt);
    assert_eq!(otb.shape, vec![2, 3]);
    assert_eq!(otb.lower.len(), 6);
    assert_eq!(otb.upper.len(), 6);
    assert_eq!(otb.finite_mask.len(), 6);
}

#[test]
fn test_output_tensor_bounds_single_element() {
    let lower = ArrayD::from_elem(IxDyn(&[1]), -5.0);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0);
    let bt = BoundedTensor::new(lower, upper).unwrap();

    let otb = OutputTensorBounds::from_bounded_tensor(&bt);
    assert_eq!(otb.lower, vec![-5.0]);
    assert_eq!(otb.upper, vec![5.0]);
    assert_eq!(otb.shape, vec![1]);
    assert_eq!(otb.finite_mask, vec![true]);
}

#[test]
fn test_output_tensor_bounds_mixed_finite_infinite() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, f32::NEG_INFINITY, 0.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, f32::INFINITY]).unwrap();
    let bt = BoundedTensor::new_allow_infinite(lower, upper).unwrap();

    let otb = OutputTensorBounds::from_bounded_tensor(&bt);
    // Element 0: both finite → true
    // Element 1: lower is -inf → false
    // Element 2: upper is +inf → false
    assert_eq!(otb.finite_mask, vec![true, false, false]);
}

// ---------------------------------------------------------------------------
// PropMethod serialization roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_prop_method_serde_ibp() {
    let json = serde_json::to_string(&PropMethod::Ibp).unwrap();
    assert_eq!(json, "\"IBP\"");
    let deserialized: PropMethod = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, PropMethod::Ibp);
}

#[test]
fn test_prop_method_serde_crown() {
    let json = serde_json::to_string(&PropMethod::Crown).unwrap();
    assert_eq!(json, "\"CROWN\"");
    let deserialized: PropMethod = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, PropMethod::Crown);
}

// ---------------------------------------------------------------------------
// KernelVerification serialization
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_verification_serde_roundtrip() {
    let kv = KernelVerification {
        kernel_name: "test_kernel".to_string(),
        method: PropMethod::Crown,
        output_lower: -1.5,
        output_upper: 2.5,
        output_width: 4.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };
    let json = serde_json::to_string(&kv).unwrap();
    let deserialized: KernelVerification = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.kernel_name, "test_kernel");
    assert_eq!(deserialized.method, PropMethod::Crown);
    assert_eq!(deserialized.output_lower, -1.5);
    assert_eq!(deserialized.output_upper, 2.5);
    assert_eq!(deserialized.output_width, 4.0);
    assert!(deserialized.is_finite);
    assert!(deserialized.crown_fallback_reason.is_none());
    assert_eq!(
        deserialized.soundness_mode,
        VerificationSoundnessMode::Sound
    );
    assert!(deserialized.output_tensor.is_none());
}

#[test]
fn test_kernel_verification_crown_fallback_reason_serialized() {
    let kv = KernelVerification {
        kernel_name: "test".to_string(),
        method: PropMethod::Ibp,
        output_lower: 0.0,
        output_upper: 1.0,
        output_width: 1.0,
        is_finite: true,
        crown_fallback_reason: Some("timeout".to_string()),
        soundness_mode: VerificationSoundnessMode::Heuristic,
        output_tensor: None,
    };
    let json = serde_json::to_string(&kv).unwrap();
    assert!(json.contains("timeout"));
    let deserialized: KernelVerification = serde_json::from_str(&json).unwrap();
    assert_eq!(
        deserialized.crown_fallback_reason.as_deref(),
        Some("timeout")
    );
}

#[test]
fn test_kernel_verification_missing_soundness_mode_defaults_to_heuristic() {
    // Simulate legacy JSON without `soundness_mode` field.
    let json = r#"{"kernel_name":"k","method":"IBP","output_lower":0.0,"output_upper":1.0,"output_width":1.0,"is_finite":true}"#;
    let kv: KernelVerification = serde_json::from_str(json).unwrap();
    assert_eq!(kv.soundness_mode, VerificationSoundnessMode::Heuristic);
}

// ---------------------------------------------------------------------------
// NormBoundsMode::forward_mode
// ---------------------------------------------------------------------------

#[test]
fn test_norm_bounds_mode_conservative_not_forward() {
    assert!(!NormBoundsMode::Conservative.forward_mode());
}

#[test]
fn test_norm_bounds_mode_forward_mode_is_forward() {
    assert!(NormBoundsMode::ForwardMode.forward_mode());
}

#[test]
fn test_norm_bounds_mode_crown_sampling_is_forward() {
    assert!(NormBoundsMode::CrownSampling.forward_mode());
}

// ---------------------------------------------------------------------------
// NormBoundsMode::crown_mode (NY gated)
// ---------------------------------------------------------------------------

#[test]
fn test_norm_bounds_mode_conservative_crown_mode() {
    use ny_propagate::layers::LayerNormCrownMode;
    assert_eq!(
        NormBoundsMode::Conservative.crown_mode(),
        LayerNormCrownMode::IbpValidated
    );
}

#[test]
fn test_norm_bounds_mode_forward_crown_mode() {
    use ny_propagate::layers::LayerNormCrownMode;
    assert_eq!(
        NormBoundsMode::ForwardMode.crown_mode(),
        LayerNormCrownMode::IbpValidated
    );
}

#[test]
fn test_norm_bounds_mode_crown_sampling_crown_mode() {
    use ny_propagate::layers::LayerNormCrownMode;
    assert_eq!(
        NormBoundsMode::CrownSampling.crown_mode(),
        LayerNormCrownMode::Sampling
    );
}

// ---------------------------------------------------------------------------
// KernelVerification::new constructor
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_verification_new_constructor() {
    let kv = KernelVerification::new(
        "nn_kernel".to_string(),
        PropMethod::Crown,
        -2.0,
        3.0,
        5.0,
        true,
    );
    assert_eq!(kv.kernel_name, "nn_kernel");
    assert_eq!(kv.method, PropMethod::Crown);
    assert_eq!(kv.output_lower, -2.0);
    assert_eq!(kv.output_upper, 3.0);
    assert_eq!(kv.output_width, 5.0);
    assert!(kv.is_finite);
    // Defaults set by constructor
    assert!(kv.crown_fallback_reason.is_none());
    assert_eq!(kv.soundness_mode, VerificationSoundnessMode::Sound);
    assert!(kv.output_tensor.is_none());
}

#[test]
fn test_kernel_verification_builder_chain() {
    let kv = KernelVerification::new("k".to_string(), PropMethod::Ibp, 0.0, 1.0, 1.0, true)
        .with_crown_fallback_reason(Some("unsupported op".to_string()))
        .with_soundness_mode(VerificationSoundnessMode::Heuristic);
    assert_eq!(kv.crown_fallback_reason.as_deref(), Some("unsupported op"));
    assert_eq!(kv.soundness_mode, VerificationSoundnessMode::Heuristic);
}

// ---------------------------------------------------------------------------
// OutputTensorBounds::new constructor
// ---------------------------------------------------------------------------

#[test]
fn test_output_tensor_bounds_new_constructor() {
    let otb = OutputTensorBounds::new(vec![-1.0, 0.0], vec![1.0, 2.0], vec![2]);
    assert_eq!(otb.lower, vec![-1.0, 0.0]);
    assert_eq!(otb.upper, vec![1.0, 2.0]);
    assert_eq!(otb.shape, vec![2]);
    // Constructor sets empty finite_mask (no finiteness tracking)
    assert!(otb.finite_mask.is_empty());
}

#[test]
fn test_output_tensor_bounds_new_serde_roundtrip() {
    let otb = OutputTensorBounds::new(vec![-3.0, 0.5], vec![3.0, 1.5], vec![1, 2]);
    let json = serde_json::to_string(&otb).unwrap();
    let deser: OutputTensorBounds = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.lower, otb.lower);
    assert_eq!(deser.upper, otb.upper);
    assert_eq!(deser.shape, otb.shape);
    // finite_mask defaults to empty on deserialization (serde default)
    assert!(deser.finite_mask.is_empty());
}

// ---------------------------------------------------------------------------
// VerifyConfig::with_norm_mode and with_collect_layer_bounds
// ---------------------------------------------------------------------------

#[test]
fn test_config_with_norm_mode_conservative() {
    let config = VerifyConfig::with_threshold(1.0)
        .unwrap()
        .with_norm_mode(NormBoundsMode::Conservative);
    assert_eq!(config.norm_mode(), NormBoundsMode::Conservative);
}

#[test]
fn test_config_with_norm_mode_crown_sampling() {
    let config = VerifyConfig::with_threshold(1.0)
        .unwrap()
        .with_norm_mode(NormBoundsMode::CrownSampling);
    assert_eq!(config.norm_mode(), NormBoundsMode::CrownSampling);
}

#[test]
fn test_config_with_collect_layer_bounds_true() {
    let config = VerifyConfig::with_threshold(1.0)
        .unwrap()
        .with_collect_layer_bounds(true);
    assert!(config.collect_layer_bounds());
}

#[test]
fn test_config_with_collect_layer_bounds_false() {
    let config = VerifyConfig::with_threshold(1.0)
        .unwrap()
        .with_collect_layer_bounds(false);
    assert!(!config.collect_layer_bounds());
}

#[test]
fn test_config_default_norm_mode_is_forward() {
    let config = VerifyConfig::default();
    assert_eq!(config.norm_mode(), NormBoundsMode::ForwardMode);
}

#[test]
fn test_config_default_collect_layer_bounds_is_false() {
    let config = VerifyConfig::default();
    assert!(!config.collect_layer_bounds());
}

// ---------------------------------------------------------------------------
// PropMethod serde — remaining variants
// ---------------------------------------------------------------------------

#[test]
fn test_prop_method_serde_alpha_crown() {
    let json = serde_json::to_string(&PropMethod::AlphaCrown).unwrap();
    assert_eq!(json, "\"ALPHACROWN\"");
    let deser: PropMethod = serde_json::from_str(&json).unwrap();
    assert_eq!(deser, PropMethod::AlphaCrown);
}

#[test]
fn test_prop_method_serde_beta_crown() {
    let json = serde_json::to_string(&PropMethod::BetaCrown).unwrap();
    assert_eq!(json, "\"BETACROWN\"");
    let deser: PropMethod = serde_json::from_str(&json).unwrap();
    assert_eq!(deser, PropMethod::BetaCrown);
}

#[test]
fn test_prop_method_serde_analytical() {
    let json = serde_json::to_string(&PropMethod::Analytical).unwrap();
    assert_eq!(json, "\"ANALYTICAL\"");
    let deser: PropMethod = serde_json::from_str(&json).unwrap();
    assert_eq!(deser, PropMethod::Analytical);
}

#[test]
fn test_prop_method_serde_mixed_ibp_crown() {
    let json = serde_json::to_string(&PropMethod::MixedIbpCrown).unwrap();
    assert_eq!(json, "\"mixed_IBP_CROWN\"");
    let deser: PropMethod = serde_json::from_str(&json).unwrap();
    assert_eq!(deser, PropMethod::MixedIbpCrown);
}

// PropMethod::is_tight() — single source of truth for CROWN-family classification.
#[test]
fn test_prop_method_is_tight() {
    assert!(PropMethod::Crown.is_tight());
    assert!(PropMethod::AlphaCrown.is_tight());
    assert!(PropMethod::BetaCrown.is_tight());
    assert!(PropMethod::Analytical.is_tight());
    assert!(!PropMethod::Ibp.is_tight());
    assert!(!PropMethod::MixedIbpCrown.is_tight());
}

#[test]
fn test_prop_method_from_method_used_maps_crown_family() {
    use ny_core::MethodUsed;

    assert_eq!(
        PropMethod::from_method_used(&MethodUsed::Ibp),
        Some(PropMethod::Ibp)
    );
    assert_eq!(
        PropMethod::from_method_used(&MethodUsed::Crown),
        Some(PropMethod::Crown)
    );
    assert_eq!(
        PropMethod::from_method_used(&MethodUsed::AlphaCrown),
        Some(PropMethod::AlphaCrown)
    );
    assert_eq!(
        PropMethod::from_method_used(&MethodUsed::BetaCrown),
        Some(PropMethod::BetaCrown)
    );
}

#[test]
fn test_prop_method_from_method_used_ignores_non_propagation_tags() {
    use ny_core::MethodUsed;

    assert_eq!(PropMethod::from_method_used(&MethodUsed::SdpCrown), None);
    assert_eq!(PropMethod::from_method_used(&MethodUsed::SmtRefiner), None);
    assert_eq!(
        PropMethod::from_method_used(&MethodUsed::Other("custom".to_string())),
        None
    );
}
