// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NaN/Inf rejection tests for all config struct validate() methods.
//! Part of #1766.

use crate::audio_disentanglement::DisentanglementThresholds;
use crate::bounds::SpectralCoverageConfig;
use crate::config::{HardBoundsConfig, QualityConfig};
use crate::cost_model::HardwareCostModel;
use crate::curriculum::CurriculumConfig;
use crate::error::TtsVerifyError;
use crate::fairness::FairnessConfig;
use crate::multi_res_stft::MultiResStftConfig;
use crate::phoneme::PhonemeVerifyConfig;
use crate::streaming::StreamingConfig;

// ---------------------------------------------------------------------------
// HardBoundsConfig
// ---------------------------------------------------------------------------

#[test]
fn test_hard_bounds_config_default_valid() {
    HardBoundsConfig::default().validate().unwrap();
}

#[test]
fn test_hard_bounds_config_nan_max_amplitude() {
    let cfg = HardBoundsConfig {
        max_amplitude: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_hard_bounds_config_inf_max_amplitude() {
    let cfg = HardBoundsConfig {
        max_amplitude: f64::INFINITY,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_hard_bounds_config_nan_min_duration_sec() {
    let cfg = HardBoundsConfig {
        min_duration_sec: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_hard_bounds_config_nan_max_duration_sec() {
    let cfg = HardBoundsConfig {
        max_duration_sec: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

// ---------------------------------------------------------------------------
// QualityConfig
// ---------------------------------------------------------------------------

#[test]
fn test_quality_config_default_valid() {
    QualityConfig::default().validate().unwrap();
}

#[test]
fn test_quality_config_nan_max_mcd_db() {
    let cfg = QualityConfig {
        max_mcd_db: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_quality_config_inf_min_snr_db() {
    let cfg = QualityConfig {
        min_snr_db: f64::INFINITY,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// SpectralCoverageConfig
// ---------------------------------------------------------------------------

#[test]
fn test_spectral_coverage_config_default_valid() {
    SpectralCoverageConfig::default().validate().unwrap();
}

#[test]
fn test_spectral_coverage_config_nan_min_coverage() {
    let cfg = SpectralCoverageConfig {
        min_coverage: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

// ---------------------------------------------------------------------------
// StreamingConfig
// ---------------------------------------------------------------------------

#[test]
fn test_streaming_config_default_valid() {
    StreamingConfig::default().validate().unwrap();
}

#[test]
fn test_streaming_config_nan_click_threshold() {
    let cfg = StreamingConfig {
        click_threshold: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

// ---------------------------------------------------------------------------
// DisentanglementThresholds
// ---------------------------------------------------------------------------

#[test]
fn test_disentanglement_thresholds_default_valid() {
    DisentanglementThresholds::default().validate().unwrap();
}

#[test]
fn test_disentanglement_thresholds_nan_f0_correlation() {
    let cfg = DisentanglementThresholds {
        f0_correlation_min: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_disentanglement_thresholds_nan_mcd_max() {
    let cfg = DisentanglementThresholds {
        mcd_max: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_disentanglement_thresholds_inf_duration_ratio() {
    let cfg = DisentanglementThresholds {
        duration_ratio_tolerance: f64::INFINITY,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

// ---------------------------------------------------------------------------
// MultiResStftConfig
// ---------------------------------------------------------------------------

#[test]
fn test_multi_res_stft_config_default_valid() {
    MultiResStftConfig::default().validate().unwrap();
}

#[test]
fn test_multi_res_stft_config_nan_max_loss() {
    let cfg = MultiResStftConfig {
        max_loss: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

// ---------------------------------------------------------------------------
// CurriculumConfig
// ---------------------------------------------------------------------------

#[test]
fn test_curriculum_config_default_valid() {
    CurriculumConfig::default().validate().unwrap();
}

#[test]
fn test_curriculum_config_nan_bottom_fraction() {
    let cfg = CurriculumConfig {
        bottom_fraction: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_curriculum_config_nan_quality_threshold() {
    let cfg = CurriculumConfig {
        quality_threshold: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

// ---------------------------------------------------------------------------
// FairnessConfig
// ---------------------------------------------------------------------------

#[test]
fn test_fairness_config_default_valid() {
    FairnessConfig::default().validate().unwrap();
}

#[test]
fn test_fairness_config_nan_alpha() {
    let cfg = FairnessConfig {
        alpha: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_fairness_config_nan_max_gap() {
    let cfg = FairnessConfig {
        max_gap: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

// ---------------------------------------------------------------------------
// PhonemeVerifyConfig
// ---------------------------------------------------------------------------

#[test]
fn test_phoneme_verify_config_default_valid() {
    PhonemeVerifyConfig::default().validate().unwrap();
}

#[test]
fn test_phoneme_verify_config_nan_min_duration() {
    let cfg = PhonemeVerifyConfig {
        min_duration_ms: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_phoneme_verify_config_nan_f0_range() {
    let cfg = PhonemeVerifyConfig {
        f0_range_hz: (f64::NAN, 500.0),
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_phoneme_verify_config_inf_max_mcd() {
    let cfg = PhonemeVerifyConfig {
        max_mcd_db: f64::INFINITY,
        ..Default::default()
    };
    assert!(matches!(
        cfg.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

// ---------------------------------------------------------------------------
// HardwareCostModel
// ---------------------------------------------------------------------------

#[test]
fn test_hardware_cost_model_m4_max_valid() {
    HardwareCostModel::m4_max().validate().unwrap();
}

#[test]
fn test_hardware_cost_model_nan_peak_tflops() {
    let mut model = HardwareCostModel::m4_max();
    model.peak_tflops_f32 = f64::NAN;
    assert!(matches!(
        model.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_hardware_cost_model_inf_bandwidth() {
    let mut model = HardwareCostModel::m4_max();
    model.peak_bandwidth_gbs = f64::INFINITY;
    assert!(matches!(
        model.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

#[test]
fn test_hardware_cost_model_nan_dispatch_overhead() {
    let mut model = HardwareCostModel::m4_max();
    model.dispatch_overhead_us = f64::NAN;
    assert!(matches!(
        model.validate(),
        Err(TtsVerifyError::InvalidConfig(_))
    ));
}

// ---------------------------------------------------------------------------
// Monotonicity dimension mismatch
// ---------------------------------------------------------------------------

#[test]
fn test_attention_monotonicity_dimension_mismatch() {
    let lower = [1.0f32, 2.0]; // length 2, expected 4 (2×2)
    let upper = [1.0f32, 2.0, 3.0, 4.0];
    let result =
        crate::monotonicity::interpret_attention_monotonicity(&lower, &upper, 2, 2, 1.0, "CROWN");
    assert!(matches!(
        result,
        Err(TtsVerifyError::DimensionMismatch { .. })
    ));
}

#[test]
fn test_attention_monotonicity_upper_mismatch() {
    let lower = [1.0f32, 2.0, 3.0, 4.0];
    let upper = [1.0f32, 2.0]; // length 2, expected 4 (2×2)
    let result =
        crate::monotonicity::interpret_attention_monotonicity(&lower, &upper, 2, 2, 1.0, "CROWN");
    assert!(matches!(
        result,
        Err(TtsVerifyError::DimensionMismatch { .. })
    ));
}
