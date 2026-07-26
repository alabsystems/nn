// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for model-specific tolerance presets.

use super::*;
use crate::compare::ComparisonConfig;

// ---- Const field values ----

#[test]
fn test_strict_preset_values() {
    let p = TolerancePreset::STRICT;
    assert_eq!(p.name, "strict");
    assert_eq!(p.abs_threshold, 1e-6);
    assert_eq!(p.rel_threshold, 1e-5);
    assert_eq!(p.cos_threshold, 0.999_999);
}

#[test]
fn test_standard_preset_values() {
    let p = TolerancePreset::STANDARD;
    assert_eq!(p.name, "standard");
    assert_eq!(p.abs_threshold, 1e-5);
    assert_eq!(p.rel_threshold, 1e-4);
    assert_eq!(p.cos_threshold, 0.9999);
}

#[test]
fn test_transformer_preset_values() {
    let p = TolerancePreset::TRANSFORMER;
    assert_eq!(p.name, "transformer");
    assert_eq!(p.abs_threshold, 1e-4);
    assert_eq!(p.rel_threshold, 1e-3);
    assert_eq!(p.cos_threshold, 0.999);
}

#[test]
fn test_audio_preset_values() {
    let p = TolerancePreset::AUDIO;
    assert_eq!(p.name, "audio");
    assert_eq!(p.abs_threshold, 1e-3);
    assert_eq!(p.rel_threshold, 1e-2);
    assert_eq!(p.cos_threshold, 0.99);
}

#[test]
fn test_quantized_preset_values() {
    let p = TolerancePreset::QUANTIZED;
    assert_eq!(p.name, "quantized");
    assert_eq!(p.abs_threshold, 1e-2);
    assert_eq!(p.rel_threshold, 5e-2);
    assert_eq!(p.cos_threshold, 0.95);
}

#[test]
fn test_tts_preset_values() {
    let p = TolerancePreset::TTS;
    assert_eq!(p.name, "tts");
    assert_eq!(p.abs_threshold, 5e-3);
    assert_eq!(p.rel_threshold, 1e-2);
    assert_eq!(p.cos_threshold, 0.995);
}

// ---- to_config conversion ----

#[test]
fn test_to_config_preserves_thresholds() {
    let config = TolerancePreset::TRANSFORMER.to_config();
    assert_eq!(config.abs_tolerance, 1e-4_f32);
    assert_eq!(config.rel_tolerance, 1e-3_f32);
    assert_eq!(config.cosine_threshold, 0.999_f32);
    // Optional gates should be disabled by default.
    assert!(config.rms_tolerance.is_none());
    assert!(config.peak_amplitude_limit.is_none());
}

#[test]
fn test_to_config_strict_matches_comparison_config_strict() {
    // STRICT preset should produce the same values as ComparisonConfig::strict().
    let preset_config = TolerancePreset::STRICT.to_config();
    let direct_config = ComparisonConfig::strict();
    assert_eq!(preset_config.abs_tolerance, direct_config.abs_tolerance);
    assert_eq!(preset_config.rel_tolerance, direct_config.rel_tolerance);
    assert_eq!(
        preset_config.cosine_threshold,
        direct_config.cosine_threshold
    );
}

#[test]
fn test_to_config_standard_matches_comparison_config_default() {
    // STANDARD preset should produce the same values as ComparisonConfig::default().
    let preset_config = TolerancePreset::STANDARD.to_config();
    let default_config = ComparisonConfig::default();
    assert_eq!(preset_config.abs_tolerance, default_config.abs_tolerance);
    assert_eq!(preset_config.rel_tolerance, default_config.rel_tolerance);
    assert_eq!(
        preset_config.cosine_threshold,
        default_config.cosine_threshold
    );
}

// ---- Ordering: core presets form a monotonically relaxing chain ----

#[test]
fn test_core_chain_ordered_by_strictness() {
    // The core chain STRICT < STANDARD < TRANSFORMER < QUANTIZED is totally ordered
    // on all three dimensions (abs non-decreasing, rel non-decreasing, cos non-increasing).
    // AUDIO and TTS are domain-specific and trade off differently (TTS is looser in abs
    // but stricter in cosine than AUDIO), so they are not part of the total order.
    let chain = [
        TolerancePreset::STRICT,
        TolerancePreset::STANDARD,
        TolerancePreset::TRANSFORMER,
        TolerancePreset::QUANTIZED,
    ];
    for window in chain.windows(2) {
        assert!(
            window[0].abs_threshold <= window[1].abs_threshold,
            "{} abs ({}) should be <= {} abs ({})",
            window[0].name,
            window[0].abs_threshold,
            window[1].name,
            window[1].abs_threshold,
        );
        assert!(
            window[0].rel_threshold <= window[1].rel_threshold,
            "{} rel ({}) should be <= {} rel ({})",
            window[0].name,
            window[0].rel_threshold,
            window[1].name,
            window[1].rel_threshold,
        );
        assert!(
            window[0].cos_threshold >= window[1].cos_threshold,
            "{} cos ({}) should be >= {} cos ({})",
            window[0].name,
            window[0].cos_threshold,
            window[1].name,
            window[1].cos_threshold,
        );
    }
}

#[test]
fn test_domain_presets_looser_than_standard() {
    // AUDIO and TTS should both be looser than STANDARD on all three dimensions.
    for preset in [TolerancePreset::AUDIO, TolerancePreset::TTS] {
        assert!(
            preset.abs_threshold >= TolerancePreset::STANDARD.abs_threshold,
            "{} abs should be >= standard abs",
            preset.name,
        );
        assert!(
            preset.cos_threshold <= TolerancePreset::STANDARD.cos_threshold,
            "{} cos should be <= standard cos",
            preset.name,
        );
    }
}

// ---- ALL list ----

#[test]
fn test_all_presets_list_contains_all_six() {
    assert_eq!(TolerancePreset::ALL.len(), 6);
    let names: Vec<&str> = TolerancePreset::ALL.iter().map(|p| p.name).collect();
    assert!(names.contains(&"strict"));
    assert!(names.contains(&"standard"));
    assert!(names.contains(&"transformer"));
    assert!(names.contains(&"audio"));
    assert!(names.contains(&"quantized"));
    assert!(names.contains(&"tts"));
}

#[test]
fn test_all_presets_have_nonempty_descriptions() {
    for preset in TolerancePreset::ALL {
        assert!(
            !preset.description.is_empty(),
            "preset '{}' has empty description",
            preset.name
        );
    }
}

// ---- by_name lookup ----

#[test]
fn test_by_name_exact_match() {
    let p = TolerancePreset::by_name("transformer").expect("should find transformer");
    assert_eq!(p, TolerancePreset::TRANSFORMER);
}

#[test]
fn test_by_name_case_insensitive() {
    let p = TolerancePreset::by_name("AUDIO").expect("should find audio");
    assert_eq!(p, TolerancePreset::AUDIO);

    let p2 = TolerancePreset::by_name("Quantized").expect("should find quantized");
    assert_eq!(p2, TolerancePreset::QUANTIZED);
}

#[test]
fn test_by_name_unknown_returns_none() {
    assert!(TolerancePreset::by_name("imaginary").is_none());
    assert!(TolerancePreset::by_name("").is_none());
}

// ---- Explicit pairwise comparisons ----

#[test]
fn test_strict_tighter_than_standard() {
    let strict = TolerancePreset::STRICT;
    let standard = TolerancePreset::STANDARD;
    assert!(
        strict.abs_threshold < standard.abs_threshold,
        "STRICT abs should be tighter than STANDARD"
    );
    assert!(
        strict.rel_threshold < standard.rel_threshold,
        "STRICT rel should be tighter than STANDARD"
    );
    assert!(
        strict.cos_threshold > standard.cos_threshold,
        "STRICT cos should require higher similarity than STANDARD"
    );
}

#[test]
fn test_quantized_looser_than_strict() {
    let strict = TolerancePreset::STRICT;
    let quantized = TolerancePreset::QUANTIZED;
    assert!(
        quantized.abs_threshold > strict.abs_threshold,
        "QUANTIZED abs ({}) should be looser than STRICT abs ({})",
        quantized.abs_threshold,
        strict.abs_threshold,
    );
    assert!(
        quantized.rel_threshold > strict.rel_threshold,
        "QUANTIZED rel ({}) should be looser than STRICT rel ({})",
        quantized.rel_threshold,
        strict.rel_threshold,
    );
    assert!(
        quantized.cos_threshold < strict.cos_threshold,
        "QUANTIZED cos ({}) should be looser (lower) than STRICT cos ({})",
        quantized.cos_threshold,
        strict.cos_threshold,
    );
}

// ---- Reasonable defaults: all thresholds are positive and finite ----

#[test]
fn test_all_presets_have_positive_finite_thresholds() {
    for preset in TolerancePreset::ALL {
        assert!(
            preset.abs_threshold > 0.0 && preset.abs_threshold.is_finite(),
            "preset '{}' abs_threshold should be positive finite, got {}",
            preset.name,
            preset.abs_threshold,
        );
        assert!(
            preset.rel_threshold > 0.0 && preset.rel_threshold.is_finite(),
            "preset '{}' rel_threshold should be positive finite, got {}",
            preset.name,
            preset.rel_threshold,
        );
        assert!(
            preset.cos_threshold > 0.0
                && preset.cos_threshold <= 1.0
                && preset.cos_threshold.is_finite(),
            "preset '{}' cos_threshold should be in (0, 1], got {}",
            preset.name,
            preset.cos_threshold,
        );
    }
}

// ---- Copy/Clone ----

#[test]
fn test_preset_is_copy() {
    let a = TolerancePreset::STRICT;
    let b = a; // Copy
    assert_eq!(a, b);
}

// ---- Integration: preset config actually works with compare_tensors ----

#[test]
fn test_standard_preset_passes_small_perturbation() {
    use crate::trace::NamedTensor;

    let reference = NamedTensor {
        name: "test".to_string(),
        data: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![4],
    };
    // Perturbation of ~5e-6: well within STANDARD abs=1e-5.
    let candidate = NamedTensor {
        name: "test".to_string(),
        data: vec![1.000005, 2.000005, 3.000005, 4.000005],
        shape: vec![4],
    };

    let config = TolerancePreset::STANDARD.to_config();
    let result =
        crate::compare::compare_tensors(&reference, &candidate, &config).expect("should compare");
    assert!(
        result.passed,
        "standard preset should pass for 5e-6 perturbation"
    );
}

#[test]
fn test_strict_preset_rejects_large_perturbation() {
    use crate::trace::NamedTensor;

    let reference = NamedTensor {
        name: "test".to_string(),
        data: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![4],
    };
    // Perturbation of ~1e-5: exceeds STRICT abs=1e-6.
    let candidate = NamedTensor {
        name: "test".to_string(),
        data: vec![1.00001, 2.00001, 3.00001, 4.00001],
        shape: vec![4],
    };

    let config = TolerancePreset::STRICT.to_config();
    let result =
        crate::compare::compare_tensors(&reference, &candidate, &config).expect("should compare");
    assert!(
        !result.passed,
        "strict preset should reject 1e-5 perturbation (abs limit is 1e-6)"
    );
}

#[test]
fn test_quantized_preset_passes_coarse_approximation() {
    use crate::trace::NamedTensor;

    let reference = NamedTensor {
        name: "quantized_layer".to_string(),
        data: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![4],
    };
    // Simulate int8 quantization noise.
    let candidate = NamedTensor {
        name: "quantized_layer".to_string(),
        data: vec![1.005, 2.008, 3.003, 4.009],
        shape: vec![4],
    };

    let config = TolerancePreset::QUANTIZED.to_config();
    let result =
        crate::compare::compare_tensors(&reference, &candidate, &config).expect("should compare");
    assert!(
        result.passed,
        "quantized preset should tolerate int8-level noise"
    );
}
