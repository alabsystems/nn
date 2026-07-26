// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `bound_analysis.rs` and related modules.
//!
//! Proves correctness properties of:
//! - `max_width`: empty, single, non-finite, normal inputs
//! - `is_norm_layer` / `is_exp_family`: correct classification
//! - `AnalysisConfig::default()` field values
//! - `TighteningTarget` enum distinctness
//! - `layers_needing_crown`: extracts EscalateToCrown recommendations
//! - `estimate_norm_chain_precision_drift`: depth=0 returns 1.0
//! - Vacuity threshold / expansion ratio invariants
//! - `VerifyConfig` threshold validation
//! - `NormBoundsMode::forward_mode()` classification
//!
//! Part of #3717.

use super::{
    estimate_norm_chain_precision_drift, is_exp_family, is_norm_layer, layers_needing_crown,
    max_width, AnalysisConfig, BoundAnalysisReport, TighteningRecommendation, TighteningTarget,
};
use crate::verify_types::{NormBoundsMode, PropMethod, VerifyConfig, DEFAULT_ESCALATION_THRESHOLD};

// ===========================================================================
// max_width
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. max_width of empty slice is 0.0
// ---------------------------------------------------------------------------

/// Prove: max_width of an empty slice returns 0.0.
#[kani::unwind(1)]
#[kani::proof]
fn max_width_empty_is_zero() {
    let bounds: &[(f32, f32)] = &[];
    assert_eq!(max_width(bounds), 0.0);
}

// ---------------------------------------------------------------------------
// 2. max_width of single element
// ---------------------------------------------------------------------------

/// Prove: max_width of a single bound pair returns hi - lo.
#[kani::unwind(1)]
#[kani::proof]
fn max_width_single_element() {
    let bounds = &[(-1.0_f32, 3.0_f32)];
    let w = max_width(bounds);
    assert!((w - 4.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// 3. max_width with NaN returns INFINITY
// ---------------------------------------------------------------------------

/// Prove: max_width returns INFINITY when a bound contains NaN.
#[kani::unwind(1)]
#[kani::proof]
fn max_width_nan_returns_infinity() {
    let bounds = &[(f32::NAN, 1.0)];
    assert_eq!(max_width(bounds), f32::INFINITY);
}

// ---------------------------------------------------------------------------
// 4. max_width with +Inf returns INFINITY
// ---------------------------------------------------------------------------

/// Prove: max_width returns INFINITY when a bound contains +Inf.
#[kani::unwind(1)]
#[kani::proof]
fn max_width_pos_inf_returns_infinity() {
    let bounds = &[(0.0, f32::INFINITY)];
    assert_eq!(max_width(bounds), f32::INFINITY);
}

// ---------------------------------------------------------------------------
// 5. max_width with -Inf returns INFINITY
// ---------------------------------------------------------------------------

/// Prove: max_width returns INFINITY when a bound contains -Inf.
#[kani::unwind(1)]
#[kani::proof]
fn max_width_neg_inf_returns_infinity() {
    let bounds = &[(f32::NEG_INFINITY, 0.0)];
    assert_eq!(max_width(bounds), f32::INFINITY);
}

// ---------------------------------------------------------------------------
// 6. max_width selects the widest pair
// ---------------------------------------------------------------------------

/// Prove: max_width returns the maximum width across multiple pairs.
#[kani::unwind(1)]
#[kani::proof]
fn max_width_selects_maximum() {
    let bounds = &[(-1.0, 1.0), (-5.0, 5.0), (0.0, 3.0)];
    let w = max_width(bounds);
    // Widths: 2.0, 10.0, 3.0 — max is 10.0
    assert!((w - 10.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// 7. max_width of point interval is 0.0
// ---------------------------------------------------------------------------

/// Prove: max_width of a point interval (lo == hi) returns 0.0.
#[kani::unwind(1)]
#[kani::proof]
fn max_width_point_interval() {
    let bounds = &[(5.0, 5.0)];
    assert_eq!(max_width(bounds), 0.0);
}

// ===========================================================================
// is_norm_layer classification
// ===========================================================================

// ---------------------------------------------------------------------------
// 8. LayerNorm is a norm layer
// ---------------------------------------------------------------------------

/// Prove: "LayerNorm" is classified as a normalization layer.
#[kani::unwind(1)]
#[kani::proof]
fn is_norm_layer_layer_norm() {
    assert!(is_norm_layer("LayerNorm"));
}

// ---------------------------------------------------------------------------
// 9. RMSNorm is a norm layer
// ---------------------------------------------------------------------------

/// Prove: "RMSNorm" is classified as a normalization layer.
#[kani::unwind(1)]
#[kani::proof]
fn is_norm_layer_rms_norm() {
    assert!(is_norm_layer("RMSNorm"));
}

// ---------------------------------------------------------------------------
// 10. InstanceNorm is a norm layer
// ---------------------------------------------------------------------------

/// Prove: "InstanceNorm" is classified as a normalization layer.
#[kani::unwind(1)]
#[kani::proof]
fn is_norm_layer_instance_norm() {
    assert!(is_norm_layer("InstanceNorm"));
}

// ---------------------------------------------------------------------------
// 11. BatchNorm is a norm layer
// ---------------------------------------------------------------------------

/// Prove: "BatchNorm" is classified as a normalization layer.
#[kani::unwind(1)]
#[kani::proof]
fn is_norm_layer_batch_norm() {
    assert!(is_norm_layer("BatchNorm"));
}

// ---------------------------------------------------------------------------
// 12. GroupNorm is a norm layer
// ---------------------------------------------------------------------------

/// Prove: "GroupNorm" is classified as a normalization layer.
#[kani::unwind(1)]
#[kani::proof]
fn is_norm_layer_group_norm() {
    assert!(is_norm_layer("GroupNorm"));
}

// ---------------------------------------------------------------------------
// 13. Linear is NOT a norm layer
// ---------------------------------------------------------------------------

/// Prove: "Linear" is NOT classified as a normalization layer.
#[kani::unwind(1)]
#[kani::proof]
fn is_norm_layer_linear_false() {
    assert!(!is_norm_layer("Linear"));
}

// ---------------------------------------------------------------------------
// 14. ReLU is NOT a norm layer
// ---------------------------------------------------------------------------

/// Prove: "ReLU" is NOT classified as a normalization layer.
#[kani::unwind(1)]
#[kani::proof]
fn is_norm_layer_relu_false() {
    assert!(!is_norm_layer("ReLU"));
}

// ===========================================================================
// is_exp_family classification
// ===========================================================================

// ---------------------------------------------------------------------------
// 15. Exp is exp-family
// ---------------------------------------------------------------------------

/// Prove: "Exp" is classified as exponential-family.
#[kani::unwind(1)]
#[kani::proof]
fn is_exp_family_exp() {
    assert!(is_exp_family("Exp"));
}

// ---------------------------------------------------------------------------
// 16. Softmax is exp-family
// ---------------------------------------------------------------------------

/// Prove: "Softmax" is classified as exponential-family.
#[kani::unwind(1)]
#[kani::proof]
fn is_exp_family_softmax() {
    assert!(is_exp_family("Softmax"));
}

// ---------------------------------------------------------------------------
// 17. Linear is NOT exp-family
// ---------------------------------------------------------------------------

/// Prove: "Linear" is NOT classified as exponential-family.
#[kani::unwind(1)]
#[kani::proof]
fn is_exp_family_linear_false() {
    assert!(!is_exp_family("Linear"));
}

// ===========================================================================
// AnalysisConfig::default() field values
// ===========================================================================

// ---------------------------------------------------------------------------
// 18. default explosion_threshold is 100.0
// ---------------------------------------------------------------------------

/// Prove: default explosion_threshold is exactly 100.0.
#[kani::unwind(1)]
#[kani::proof]
fn default_config_explosion_threshold() {
    let config = AnalysisConfig::default();
    assert_eq!(config.explosion_threshold, 100.0);
}

// ---------------------------------------------------------------------------
// 19. default crown_escalation_width is 1e4
// ---------------------------------------------------------------------------

/// Prove: default crown_escalation_width is exactly 1e4.
#[kani::unwind(1)]
#[kani::proof]
fn default_config_crown_escalation_width() {
    let config = AnalysisConfig::default();
    assert_eq!(config.crown_escalation_width, 1e4);
}

// ---------------------------------------------------------------------------
// 20. default smt_max_elements is 256
// ---------------------------------------------------------------------------

/// Prove: default smt_max_elements is exactly 256.
#[kani::unwind(1)]
#[kani::proof]
fn default_config_smt_max_elements() {
    let config = AnalysisConfig::default();
    assert_eq!(config.smt_max_elements, 256);
}

// ---------------------------------------------------------------------------
// 21. default norm_chain_min_length is 5
// ---------------------------------------------------------------------------

/// Prove: default norm_chain_min_length is exactly 5.
#[kani::unwind(1)]
#[kani::proof]
fn default_config_norm_chain_min_length() {
    let config = AnalysisConfig::default();
    assert_eq!(config.norm_chain_min_length, 5);
}

// ---------------------------------------------------------------------------
// 22. default norm_chain_explosion_ratio is 10.0
// ---------------------------------------------------------------------------

/// Prove: default norm_chain_explosion_ratio is exactly 10.0.
#[kani::unwind(1)]
#[kani::proof]
fn default_config_norm_chain_explosion_ratio() {
    let config = AnalysisConfig::default();
    assert_eq!(config.norm_chain_explosion_ratio, 10.0);
}

// ===========================================================================
// TighteningTarget enum distinctness
// ===========================================================================

// ---------------------------------------------------------------------------
// 23. TighteningTarget::Model != TighteningTarget::Framework
// ---------------------------------------------------------------------------

/// Prove: Model and Framework are distinct variants.
#[kani::unwind(1)]
#[kani::proof]
fn tightening_target_model_ne_framework() {
    assert_ne!(TighteningTarget::Model, TighteningTarget::Framework);
}

// ---------------------------------------------------------------------------
// 24. TighteningTarget::Model != TighteningTarget::Verifier
// ---------------------------------------------------------------------------

/// Prove: Model and Verifier are distinct variants.
#[kani::unwind(1)]
#[kani::proof]
fn tightening_target_model_ne_verifier() {
    assert_ne!(TighteningTarget::Model, TighteningTarget::Verifier);
}

// ---------------------------------------------------------------------------
// 25. TighteningTarget::Framework != TighteningTarget::Verifier
// ---------------------------------------------------------------------------

/// Prove: Framework and Verifier are distinct variants.
#[kani::unwind(1)]
#[kani::proof]
fn tightening_target_framework_ne_verifier() {
    assert_ne!(TighteningTarget::Framework, TighteningTarget::Verifier);
}

// ===========================================================================
// estimate_norm_chain_precision_drift
// ===========================================================================

// ---------------------------------------------------------------------------
// 26. drift depth=0 returns 1.0
// ---------------------------------------------------------------------------

/// Prove: precision drift estimate for depth=0 returns exactly 1.0.
#[kani::unwind(1)]
#[kani::proof]
fn precision_drift_depth_zero() {
    let ratio = estimate_norm_chain_precision_drift(0);
    assert_eq!(ratio, 1.0);
}

// ---------------------------------------------------------------------------
// 27. drift depth=1 is in (0.0, 1.0]
// ---------------------------------------------------------------------------

/// Prove: precision drift for depth=1 is positive and at most 1.0.
#[kani::unwind(1)]
#[kani::proof]
fn precision_drift_depth_one_bounded() {
    let ratio = estimate_norm_chain_precision_drift(1);
    assert!(ratio > 0.0, "drift ratio must be positive");
    assert!(ratio <= 1.0, "drift ratio must not exceed 1.0");
}

// ===========================================================================
// VerifyConfig threshold validation
// ===========================================================================

// ---------------------------------------------------------------------------
// 28. DEFAULT_ESCALATION_THRESHOLD is 1e6
// ---------------------------------------------------------------------------

/// Prove: DEFAULT_ESCALATION_THRESHOLD is exactly 1e6.
#[kani::unwind(1)]
#[kani::proof]
fn default_escalation_threshold_value() {
    assert_eq!(DEFAULT_ESCALATION_THRESHOLD, 1e6_f32);
}

// ---------------------------------------------------------------------------
// 29. VerifyConfig::with_threshold rejects NaN
// ---------------------------------------------------------------------------

/// Prove: NaN threshold is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn verify_config_rejects_nan_threshold() {
    let result = VerifyConfig::with_threshold(f32::NAN);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// 30. VerifyConfig::with_threshold rejects Inf
// ---------------------------------------------------------------------------

/// Prove: Infinity threshold is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn verify_config_rejects_inf_threshold() {
    let result = VerifyConfig::with_threshold(f32::INFINITY);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// 31. VerifyConfig::with_threshold rejects negative
// ---------------------------------------------------------------------------

/// Prove: negative threshold is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn verify_config_rejects_negative_threshold() {
    let result = VerifyConfig::with_threshold(-1.0);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// 32. VerifyConfig::with_threshold accepts zero
// ---------------------------------------------------------------------------

/// Prove: zero threshold is accepted (valid edge case).
#[kani::unwind(1)]
#[kani::proof]
fn verify_config_accepts_zero_threshold() {
    let result = VerifyConfig::with_threshold(0.0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().escalation_threshold(), 0.0);
}

// ---------------------------------------------------------------------------
// 33. VerifyConfig::with_threshold accepts positive
// ---------------------------------------------------------------------------

/// Prove: positive finite threshold is accepted.
#[kani::unwind(1)]
#[kani::proof]
fn verify_config_accepts_positive_threshold() {
    let result = VerifyConfig::with_threshold(100.0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().escalation_threshold(), 100.0);
}

// ===========================================================================
// NormBoundsMode::forward_mode()
// ===========================================================================

// ---------------------------------------------------------------------------
// 34. Conservative is not forward_mode
// ---------------------------------------------------------------------------

/// Prove: Conservative mode does not enable forward_mode.
#[kani::unwind(1)]
#[kani::proof]
fn norm_bounds_conservative_not_forward() {
    assert!(!NormBoundsMode::Conservative.forward_mode());
}

// ---------------------------------------------------------------------------
// 35. ForwardMode is forward_mode
// ---------------------------------------------------------------------------

/// Prove: ForwardMode enables forward_mode.
#[kani::unwind(1)]
#[kani::proof]
fn norm_bounds_forward_mode_is_forward() {
    assert!(NormBoundsMode::ForwardMode.forward_mode());
}

// ---------------------------------------------------------------------------
// 36. CrownSampling is forward_mode
// ---------------------------------------------------------------------------

/// Prove: CrownSampling enables forward_mode.
#[kani::unwind(1)]
#[kani::proof]
fn norm_bounds_crown_sampling_is_forward() {
    assert!(NormBoundsMode::CrownSampling.forward_mode());
}

// ===========================================================================
// VerifyConfig defaults
// ===========================================================================

// ---------------------------------------------------------------------------
// 37. VerifyConfig::default() escalation_threshold
// ---------------------------------------------------------------------------

/// Prove: default VerifyConfig has escalation_threshold == 1e6.
#[kani::unwind(1)]
#[kani::proof]
fn verify_config_default_escalation() {
    let config = VerifyConfig::default();
    assert_eq!(config.escalation_threshold(), DEFAULT_ESCALATION_THRESHOLD);
}

// ---------------------------------------------------------------------------
// 38. VerifyConfig::default() norm_mode is ForwardMode
// ---------------------------------------------------------------------------

/// Prove: default VerifyConfig uses ForwardMode for norm layers.
#[kani::unwind(1)]
#[kani::proof]
fn verify_config_default_norm_mode() {
    let config = VerifyConfig::default();
    assert_eq!(config.norm_mode(), NormBoundsMode::ForwardMode);
}

// ---------------------------------------------------------------------------
// 39. VerifyConfig::default() collect_layer_bounds is false
// ---------------------------------------------------------------------------

/// Prove: default VerifyConfig does not collect layer bounds.
#[kani::unwind(1)]
#[kani::proof]
fn verify_config_default_no_collect_layer_bounds() {
    let config = VerifyConfig::default();
    assert!(!config.collect_layer_bounds());
}

// ---------------------------------------------------------------------------
// 40. VerifyConfig::default() require_sound is false
// ---------------------------------------------------------------------------

/// Prove: default VerifyConfig does not require strict soundness.
#[kani::unwind(1)]
#[kani::proof]
fn verify_config_default_no_require_sound() {
    let config = VerifyConfig::default();
    assert!(!config.require_sound());
}

// ===========================================================================
// layers_needing_crown: extracts from recommendations
// ===========================================================================

// ---------------------------------------------------------------------------
// 41. layers_needing_crown empty for empty recommendations
// ---------------------------------------------------------------------------

/// Prove: layers_needing_crown returns empty for report with no recommendations.
#[kani::unwind(128)]
#[kani::proof]
fn layers_needing_crown_empty_report() {
    let report = BoundAnalysisReport {
        model_name: "test".to_string(),
        total_layers: 0,
        layers: vec![],
        explosion_points: vec![],
        output_width: 0.0,
        output_is_finite: true,
        crown_coverage: 1.0,
        recommendations: vec![],
        analyzed_at: "2026-01-01T00:00:00Z".to_string(),
        chained_norm_depth: 0,
        precision_drift_ratio: None,
        drift_per_layer: None,
    };
    let result = layers_needing_crown(&report);
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// 42. layers_needing_crown extracts EscalateToCrown indices
// ---------------------------------------------------------------------------

/// Prove: layers_needing_crown extracts layer indices from EscalateToCrown.
#[kani::unwind(128)]
#[kani::proof]
fn layers_needing_crown_extracts_indices() {
    let report = BoundAnalysisReport {
        model_name: "test".to_string(),
        total_layers: 5,
        layers: vec![],
        explosion_points: vec![],
        output_width: 0.0,
        output_is_finite: true,
        crown_coverage: 0.5,
        recommendations: vec![
            TighteningRecommendation::EscalateToCrown {
                layer_index: 2,
                node_name: None,
                layer_type: "Linear".to_string(),
                ibp_width: 50000.0,
            },
            TighteningRecommendation::EscalateToCrown {
                layer_index: 7,
                node_name: None,
                layer_type: "Conv1d".to_string(),
                ibp_width: 80000.0,
            },
        ],
        analyzed_at: "2026-01-01T00:00:00Z".to_string(),
        chained_norm_depth: 0,
        precision_drift_ratio: None,
        drift_per_layer: None,
    };
    let result = layers_needing_crown(&report);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], 2);
    assert_eq!(result[1], 7);
}

// ---------------------------------------------------------------------------
// 43. layers_needing_crown skips non-EscalateToCrown recommendations
// ---------------------------------------------------------------------------

/// Prove: layers_needing_crown ignores non-EscalateToCrown recommendations.
#[kani::unwind(128)]
#[kani::proof]
fn layers_needing_crown_ignores_other_recs() {
    let report = BoundAnalysisReport {
        model_name: "test".to_string(),
        total_layers: 5,
        layers: vec![],
        explosion_points: vec![],
        output_width: 0.0,
        output_is_finite: true,
        crown_coverage: 0.5,
        recommendations: vec![TighteningRecommendation::TightenLayer {
            layer_index: 3,
            node_name: None,
            layer_type: "Exp".to_string(),
            current_width: f32::INFINITY,
            expansion_ratio: f32::INFINITY,
            target: TighteningTarget::Framework,
            suggestion: "test".to_string(),
        }],
        analyzed_at: "2026-01-01T00:00:00Z".to_string(),
        chained_norm_depth: 0,
        precision_drift_ratio: None,
        drift_per_layer: None,
    };
    let result = layers_needing_crown(&report);
    assert!(result.is_empty(), "TightenLayer should be ignored");
}

// ---------------------------------------------------------------------------
// 44. layers_needing_crown deduplicates
// ---------------------------------------------------------------------------

/// Prove: layers_needing_crown deduplicates repeated layer indices.
#[kani::unwind(128)]
#[kani::proof]
fn layers_needing_crown_deduplicates() {
    let report = BoundAnalysisReport {
        model_name: "test".to_string(),
        total_layers: 5,
        layers: vec![],
        explosion_points: vec![],
        output_width: 0.0,
        output_is_finite: true,
        crown_coverage: 0.5,
        recommendations: vec![
            TighteningRecommendation::EscalateToCrown {
                layer_index: 3,
                node_name: None,
                layer_type: "Linear".to_string(),
                ibp_width: 50000.0,
            },
            TighteningRecommendation::EscalateToCrown {
                layer_index: 3,
                node_name: Some("n3".to_string()),
                layer_type: "Linear".to_string(),
                ibp_width: 60000.0,
            },
        ],
        analyzed_at: "2026-01-01T00:00:00Z".to_string(),
        chained_norm_depth: 0,
        precision_drift_ratio: None,
        drift_per_layer: None,
    };
    let result = layers_needing_crown(&report);
    assert_eq!(result.len(), 1, "duplicate layer indices should be deduped");
    assert_eq!(result[0], 3);
}
