// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended pipeline verification tests for nn-tts-verify.
//!
//! Covers:
//! 1. Audio quality property verification (sample rate, duration, amplitude)
//! 2. Cost model accuracy (FLOP counts, roofline estimates)
//! 3. Pipeline junction contracts (stage boundary constraints)
//! 4. Certificate validation (well-formed certificates)
//! 5. Moonshot property checks P1-P8 (signatures and behavior)

use std::collections::HashMap;

use crate::bounds::{
    check_duration, check_no_clipping, check_non_silence, check_spectral_coverage,
    SpectralCoverageConfig,
};
use crate::certificate::Certificate;
use crate::cost_model::{
    profile_dispatch_plan, total_estimated_time_us, total_flops, total_memory_bytes,
    HardwareCostModel, LayerCostProfile,
};
use crate::crown_junction::{
    check_all_junction_contracts, check_junction_bound, contract_bounds_map, JunctionCheckSummary,
};
use crate::kokoro_contracts::{
    all_contracts, bounds_within_contract, contract_stage, max_contract_violation, JunctionContract,
};
use crate::moonshot::{
    artifact_registry, MoonshotCertificate, MoonshotStatus, VerificationLevel, PROPERTY_NAMES,
};
use crate::moonshot_crown::{
    check_intelligibility_proxy, check_non_clipping as crown_check_non_clipping,
    check_non_silence as crown_check_non_silence, check_streaming_safety,
    check_temporal_boundedness,
};
use crate::pipeline::{check_junction, verify_pipeline, PipelineCertificate, VerifiedStage};
use crate::test_audio_helpers::{sine_wave, sine_wave_full};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a simple verified stage with uniform bounds.
fn make_stage(
    name: &str,
    shape: &[usize],
    input_range: (f64, f64),
    output_range: (f64, f64),
    method: &str,
    is_sound: bool,
) -> VerifiedStage {
    let n: usize = shape.iter().product();
    VerifiedStage::new(
        name,
        shape.to_vec(),
        shape.to_vec(),
        vec![input_range.0; n],
        vec![input_range.1; n],
        vec![output_range.0; n],
        vec![output_range.1; n],
        method,
        is_sound,
    )
}

/// Build a simple two-stage pipeline certificate with compatible bounds.
fn make_valid_pipeline() -> PipelineCertificate {
    let s1 = make_stage("encoder", &[4], (-1.0, 1.0), (-0.5, 0.5), "CROWN", true);
    let s2 = make_stage("decoder", &[4], (-0.5, 0.5), (-1.0, 1.0), "CROWN", true);
    verify_pipeline(&[s1, s2]).expect("valid pipeline should succeed")
}

/// Generate synthetic speech-like audio.
///
/// Spreads energy across the full one-sided spectrum (up to Nyquist) so the
/// signal is genuinely broadband: the default spectral-coverage check at
/// 24 kHz divides the band into eight 1500 Hz sub-bands spanning 0-12 kHz, so
/// the tone set must reach into every sub-band for coverage to pass.
fn synthetic_speech(sample_rate: u32, duration_sec: f64) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec) as usize;
    let mut signal = vec![0.0_f32; n];
    let freqs = [220.0, 440.0, 880.0, 1760.0, 3520.0, 5000.0, 7000.0, 9000.0, 11000.0];
    let amps = [0.25, 0.20, 0.15, 0.10, 0.06, 0.05, 0.04, 0.03, 0.02];
    for (i, sample) in signal.iter_mut().enumerate() {
        let t = i as f64 / f64::from(sample_rate);
        for (&f, &a) in freqs.iter().zip(amps.iter()) {
            *sample += (a * (2.0 * std::f64::consts::PI * f * t).sin()) as f32;
        }
    }
    signal
}

// ===========================================================================
// 1. Audio quality property verification
// ===========================================================================

#[test]
fn test_audio_non_silence_on_sine_wave() {
    let samples = sine_wave(440.0, 24000, 0.5);
    let bound = check_non_silence(&samples, 0.01);
    assert!(bound.passed, "sine wave RMS should exceed 0.01");
    assert!(
        bound.value > 0.5,
        "440 Hz sine at amplitude 1.0 has RMS ~0.707"
    );
}

#[test]
fn test_audio_silence_detected() {
    let samples = vec![0.0_f32; 24000];
    let bound = check_non_silence(&samples, 0.01);
    assert!(!bound.passed, "silence should fail non-silence check");
    assert!(bound.value < 1e-10, "RMS of silence should be ~0");
}

#[test]
fn test_audio_no_clipping_on_normal_signal() {
    let samples = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let bound = check_no_clipping(&samples, 1.0);
    assert!(
        bound.passed,
        "amplitude 0.5 should not clip at threshold 1.0"
    );
    assert!(bound.value < 0.51, "peak should be close to 0.5");
}

#[test]
fn test_audio_clipping_detected() {
    let samples = sine_wave_full(440.0, 24000, 0.5, 1.5);
    let bound = check_no_clipping(&samples, 1.0);
    assert!(!bound.passed, "amplitude 1.5 should clip at threshold 1.0");
    assert!(bound.value > 1.0, "peak should exceed 1.0");
}

#[test]
fn test_audio_duration_within_range() {
    let samples = sine_wave(440.0, 24000, 1.0); // 1 second
    let bound = check_duration(&samples, 24000, 0.1, 10.0);
    assert!(bound.passed, "1s should be within [0.1, 10.0]");
    assert!((bound.value - 1.0).abs() < 0.01, "duration should be ~1.0s");
}

#[test]
fn test_audio_duration_too_short() {
    let samples = sine_wave(440.0, 24000, 0.01); // 10ms
    let bound = check_duration(&samples, 24000, 0.1, 10.0);
    assert!(!bound.passed, "10ms should fail [0.1, 10.0] range");
}

#[test]
fn test_audio_duration_too_long() {
    // 15 seconds at 24kHz = 360000 samples
    let samples = sine_wave(440.0, 24000, 15.0);
    let bound = check_duration(&samples, 24000, 0.1, 10.0);
    assert!(!bound.passed, "15s should fail max 10s bound");
}

#[test]
fn test_spectral_coverage_on_broadband_signal() {
    let samples = synthetic_speech(24000, 0.5);
    let config = SpectralCoverageConfig::default();
    let bound = check_spectral_coverage(&samples, 24000, &config).unwrap();
    assert!(
        bound.passed,
        "broadband synthetic speech should have spectral coverage"
    );
}

#[test]
fn test_spectral_coverage_narrow_signal() {
    // Pure sine wave has energy in only one frequency band
    let samples = sine_wave_full(440.0, 24000, 0.5, 0.3);
    let config = SpectralCoverageConfig {
        n_bands: 8,
        min_energy_db: -40.0,
        min_coverage: 0.9, // Require 90% of bands
    };
    let bound = check_spectral_coverage(&samples, 24000, &config).unwrap();
    assert!(
        !bound.passed,
        "pure sine should not have 90% spectral coverage"
    );
}

#[test]
fn test_audio_sample_rate_zero_handling() {
    let samples = sine_wave(440.0, 24000, 0.5);
    let bound = check_duration(&samples, 0, 0.1, 10.0);
    // With sample_rate=0, duration is 0.0, which fails the min bound
    assert!(!bound.passed, "zero sample rate should produce 0 duration");
}

// ===========================================================================
// 2. Cost model accuracy
// ===========================================================================

#[test]
fn test_hardware_cost_model_m4_max_has_positive_values() {
    let model = HardwareCostModel::m4_max();
    assert!(model.peak_tflops_f32 > 0.0);
    assert!(model.peak_bandwidth_gbs > 0.0);
    assert!(model.dispatch_overhead_us > 0.0);
    assert!(model.validate().is_ok());
}

#[test]
fn test_hardware_cost_model_conservative_slower_than_theoretical() {
    let theoretical = HardwareCostModel::m4_max();
    let conservative = HardwareCostModel::m4_max_conservative();

    // Conservative should estimate longer execution times
    let flops = 1_000_000;
    let mem_bytes = 4_000_000;
    let t_theoretical = theoretical.estimate_time_us(flops, mem_bytes);
    let t_conservative = conservative.estimate_time_us(flops, mem_bytes);
    assert!(
        t_conservative > t_theoretical,
        "conservative model ({t_conservative:.2}) should be slower than theoretical ({t_theoretical:.2})"
    );
}

#[test]
fn test_estimate_time_us_compute_bound() {
    let model = HardwareCostModel {
        peak_tflops_f32: 10.0,
        peak_bandwidth_gbs: 1000.0, // Very fast memory
        dispatch_overhead_us: 0.0,
    };
    // 10 TFLOPS = 10e12 FLOPS/s = 10e6 FLOPS/us
    // 100e6 FLOPs should take 10 us (compute bound)
    let time = model.estimate_time_us(100_000_000, 100); // 100M FLOPs, tiny memory
    assert!(
        (time - 10.0).abs() < 0.01,
        "expected ~10 us for 100M FLOPs at 10 TFLOPS, got {time}"
    );
}

#[test]
fn test_estimate_time_us_memory_bound() {
    let model = HardwareCostModel {
        peak_tflops_f32: 1000.0, // Very fast compute
        peak_bandwidth_gbs: 100.0,
        dispatch_overhead_us: 0.0,
    };
    // 100 GB/s = 100e9 bytes/s = 100e3 bytes/us
    // 10e6 bytes should take 100 us (memory bound)
    let time = model.estimate_time_us(1, 10_000_000); // tiny FLOPs, 10MB
    assert!(
        (time - 100.0).abs() < 0.1,
        "expected ~100 us for 10MB at 100 GB/s, got {time}"
    );
}

#[test]
fn test_estimate_time_includes_dispatch_overhead() {
    let model = HardwareCostModel {
        peak_tflops_f32: 100.0,
        peak_bandwidth_gbs: 1000.0,
        dispatch_overhead_us: 5.0,
    };
    let time = model.estimate_time_us(0, 0);
    assert!(
        (time - 5.0).abs() < 0.01,
        "zero work should still have dispatch overhead, got {time}"
    );
}

#[test]
fn test_cost_model_validation_rejects_negative() {
    let model = HardwareCostModel {
        peak_tflops_f32: -1.0,
        peak_bandwidth_gbs: 100.0,
        dispatch_overhead_us: 5.0,
    };
    assert!(
        model.validate().is_err(),
        "negative TFLOPS should fail validation"
    );
}

#[test]
fn test_cost_model_validation_rejects_nan() {
    let model = HardwareCostModel {
        peak_tflops_f32: 10.0,
        peak_bandwidth_gbs: f64::NAN,
        dispatch_overhead_us: 5.0,
    };
    assert!(
        model.validate().is_err(),
        "NaN bandwidth should fail validation"
    );
}

#[test]
fn test_cost_model_validation_rejects_infinity() {
    let model = HardwareCostModel {
        peak_tflops_f32: 10.0,
        peak_bandwidth_gbs: 100.0,
        dispatch_overhead_us: f64::INFINITY,
    };
    assert!(
        model.validate().is_err(),
        "infinite dispatch overhead should fail validation"
    );
}

#[test]
fn test_layer_cost_profile_accessors() {
    let profile = LayerCostProfile::new("matmul_0", 1_000_000, 500_000, 12.5, Some(15.3));
    assert_eq!(profile.layer_name, "matmul_0");
    assert_eq!(profile.flops, 1_000_000);
    assert_eq!(profile.memory_bytes, 500_000);
    assert!((profile.estimated_time_us - 12.5).abs() < 1e-10);
    assert_eq!(profile.measured_time_us, Some(15.3));
}

#[test]
fn test_total_flops_sums_profiles() {
    let profiles = vec![
        LayerCostProfile::new("a", 100, 50, 1.0, None),
        LayerCostProfile::new("b", 200, 75, 2.0, None),
        LayerCostProfile::new("c", 300, 25, 3.0, None),
    ];
    assert_eq!(total_flops(&profiles), 600);
    assert_eq!(total_memory_bytes(&profiles), 150);
    assert!((total_estimated_time_us(&profiles) - 6.0).abs() < 1e-10);
}

#[test]
fn test_profile_empty_dispatch_plan() {
    let model = HardwareCostModel::m4_max();
    let profiles = profile_dispatch_plan(&[], &model);
    assert!(profiles.is_empty());
    assert_eq!(total_flops(&profiles), 0);
    assert!((total_estimated_time_us(&profiles)).abs() < 1e-10);
}

// ===========================================================================
// 3. Pipeline junction contracts
// ===========================================================================

#[test]
fn test_all_contracts_has_six_entries() {
    let contracts = all_contracts();
    assert_eq!(
        contracts.len(),
        6,
        "Kokoro pipeline has 6 junction contracts"
    );
}

#[test]
fn test_contract_names_are_unique() {
    let contracts = all_contracts();
    let names: Vec<&str> = contracts.iter().map(|c| c.name).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        names.len(),
        sorted.len(),
        "junction contract names must be unique"
    );
}

#[test]
fn test_contract_bounds_are_ordered() {
    let contracts = all_contracts();
    for c in &contracts {
        assert!(
            c.lower < c.upper,
            "contract {} has lower ({}) >= upper ({})",
            c.name,
            c.lower,
            c.upper
        );
    }
}

#[test]
fn test_bounds_within_contract_contained() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let lower = vec![-0.5, -0.3, 0.0];
    let upper = vec![0.5, 0.3, 0.8];
    assert!(bounds_within_contract(&contract, &lower, &upper));
}

#[test]
fn test_bounds_within_contract_violation_upper() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let lower = vec![0.0, 0.0];
    let upper = vec![0.5, 1.5]; // second element exceeds upper
    assert!(!bounds_within_contract(&contract, &lower, &upper));
}

#[test]
fn test_bounds_within_contract_violation_lower() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let lower = vec![-1.5, 0.0]; // first element below lower
    let upper = vec![0.5, 0.5];
    assert!(!bounds_within_contract(&contract, &lower, &upper));
}

#[test]
fn test_bounds_within_contract_nan_is_violation() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let lower = vec![f64::NAN];
    let upper = vec![0.5];
    assert!(!bounds_within_contract(&contract, &lower, &upper));
}

#[test]
fn test_bounds_within_contract_length_mismatch() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let lower = vec![0.0, 0.0];
    let upper = vec![0.5]; // different length
    assert!(!bounds_within_contract(&contract, &lower, &upper));
}

#[test]
fn test_max_contract_violation_zero_when_contained() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let lower = vec![-0.5, -0.3];
    let upper = vec![0.5, 0.8];
    let violation = max_contract_violation(&contract, &lower, &upper);
    assert!(
        violation.abs() < 1e-10,
        "no violation expected, got {violation}"
    );
}

#[test]
fn test_max_contract_violation_positive_on_overflow() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let lower = vec![0.0];
    let upper = vec![1.3]; // exceeds by 0.3
    let violation = max_contract_violation(&contract, &lower, &upper);
    assert!(
        (violation - 0.3).abs() < 1e-10,
        "expected 0.3 violation, got {violation}"
    );
}

#[test]
fn test_max_contract_violation_nan_returns_max() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let lower = vec![f64::NAN];
    let upper = vec![0.5];
    let violation = max_contract_violation(&contract, &lower, &upper);
    assert_eq!(violation, f64::MAX, "NaN should produce MAX violation");
}

#[test]
fn test_contract_stage_creates_proper_bounds() {
    let j2 = JunctionContract::new("J2_F0", "Decoder -> Source", -5.0, 800.0);
    let j5 = JunctionContract::new("J5_AUDIO", "iSTFT output", -1.0, 1.0);
    let stage = contract_stage("vocoder", &[2, 3], &[2, 3], &j2, &j5, "CROWN", true);
    assert_eq!(stage.name, "vocoder");
    assert_eq!(stage.input_shape, vec![2, 3]);
    assert_eq!(stage.output_shape, vec![2, 3]);
    assert_eq!(stage.input_lower.len(), 6);
    assert_eq!(stage.output_lower.len(), 6);
    assert!(stage
        .input_lower
        .iter()
        .all(|&v| (v - (-5.0)).abs() < 1e-10));
    assert!(stage.input_upper.iter().all(|&v| (v - 800.0).abs() < 1e-10));
    assert!(stage
        .output_lower
        .iter()
        .all(|&v| (v - (-1.0)).abs() < 1e-10));
    assert!(stage.output_upper.iter().all(|&v| (v - 1.0).abs() < 1e-10));
    assert!(stage.is_sound);
}

#[test]
fn test_check_junction_bound_pass_and_fail() {
    // Within range
    let pass = check_junction_bound("J5_AUDIO", -1.0, 1.0, -0.9, 0.9);
    assert!(pass.passed);
    assert_eq!(pass.junction_name, "J5_AUDIO");

    // Out of range (upper)
    let fail = check_junction_bound("J5_AUDIO", -1.0, 1.0, -0.5, 1.5);
    assert!(!fail.passed);
}

#[test]
fn test_check_all_junction_contracts_with_valid_intermediates() {
    let mut intermediates = HashMap::new();
    intermediates.insert("J2_F0".to_string(), (0.0_f32, 400.0_f32));
    intermediates.insert("J2_ENERGY".to_string(), (-10.0_f32, 10.0_f32));
    intermediates.insert("J3_MAGNITUDE".to_string(), (-40.0_f32, 40.0_f32));
    intermediates.insert("J3B_PHASE".to_string(), (-3000.0_f32, 3000.0_f32));
    intermediates.insert("J4_BF16".to_string(), (-64.0_f32, 64.0_f32));
    intermediates.insert("J5_AUDIO".to_string(), (-0.9_f32, 0.9_f32));

    let checks = check_all_junction_contracts(&intermediates);
    assert_eq!(checks.len(), 6);
    assert!(
        checks.iter().all(|c| c.passed),
        "all within-range intermediates should pass"
    );
}

#[test]
fn test_check_all_junction_contracts_partial_intermediates() {
    let mut intermediates = HashMap::new();
    intermediates.insert("J5_AUDIO".to_string(), (-0.9_f32, 0.9_f32));
    let checks = check_all_junction_contracts(&intermediates);
    assert_eq!(
        checks.len(),
        1,
        "only one matching contract should be checked"
    );
}

#[test]
fn test_contract_bounds_map_returns_all_contracts() {
    let map = contract_bounds_map();
    assert_eq!(map.len(), 6);
    assert!(map.contains_key("J2_F0"));
    assert!(map.contains_key("J5_AUDIO"));
}

// ===========================================================================
// 4. Certificate validation
// ===========================================================================

#[test]
fn test_certificate_passes_hard_bounds_all_pass() {
    let cert = Certificate {
        hard_bounds: vec![
            crate::bounds::HardBound {
                name: "non_silence",
                passed: true,
                value: 0.5,
                threshold: 0.01,
            },
            crate::bounds::HardBound {
                name: "no_clipping",
                passed: true,
                value: 0.8,
                threshold: 1.0,
            },
        ],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    assert!(cert.passes_hard_bounds());
    assert!(cert.passes_quality()); // vacuously true
}

#[test]
fn test_certificate_fails_when_hard_bound_fails() {
    let cert = Certificate {
        hard_bounds: vec![crate::bounds::HardBound {
            name: "non_silence",
            passed: false,
            value: 0.001,
            threshold: 0.01,
        }],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: false,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    assert!(!cert.passes_hard_bounds());
}

#[test]
fn test_certificate_with_crown_evidence() {
    let status = MoonshotStatus::from_repo();
    let moonshot_cert =
        MoonshotCertificate::from_status(&status, "test-model", "English text", "fake-hash");

    let cert = Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };

    assert!(!cert.has_crown_evidence());
    let enriched = cert.with_crown_evidence(moonshot_cert);
    assert!(enriched.has_crown_evidence());
}

#[test]
fn test_certificate_with_junction_summary() {
    let summary = JunctionCheckSummary {
        checks: vec![],
        total_passed: 3,
        total_failed: 0,
    };

    let cert = Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };

    assert!(!cert.has_junction_summary());
    assert!(cert.passes_junction_contracts()); // vacuously true when None
    let enriched = cert.with_junction_summary(summary);
    assert!(enriched.has_junction_summary());
    assert!(enriched.passes_junction_contracts());
}

#[test]
fn test_certificate_fails_junction_contracts() {
    let summary = JunctionCheckSummary {
        checks: vec![],
        total_passed: 2,
        total_failed: 1,
    };

    let cert = Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: Some(summary),
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };

    assert!(!cert.passes_junction_contracts());
}

#[test]
fn test_certificate_report_includes_all_sections() {
    let status = MoonshotStatus::from_repo();
    let moonshot_cert =
        MoonshotCertificate::from_status(&status, "test-model", "English text", "fake-hash");

    let summary = JunctionCheckSummary {
        checks: vec![crate::crown_junction::StageBoundCheck {
            junction_name: "J5_AUDIO".to_string(),
            expected_lower: -1.0,
            expected_upper: 1.0,
            actual_lower: -0.5,
            actual_upper: 0.5,
            passed: true,
        }],
        total_passed: 1,
        total_failed: 0,
    };

    let cert = Certificate {
        hard_bounds: vec![crate::bounds::HardBound {
            name: "non_silence",
            passed: true,
            value: 0.5,
            threshold: 0.01,
        }],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: Some("abc123".to_string()),
        crown_evidence: Some(moonshot_cert),
        junction_summary: Some(summary),
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };

    let report = cert.report();
    assert!(report.contains("TTS Verification Certificate"));
    assert!(report.contains("Hard Bounds"));
    assert!(report.contains("non_silence"));
    assert!(report.contains("Deterministic Hash"));
    assert!(report.contains("abc123"));
    assert!(report.contains("CROWN Verification Evidence"));
    assert!(report.contains("Junction Contract Checks"));
    assert!(report.contains("1/1 contracts passed"));
}

#[test]
fn test_moonshot_certificate_from_status_has_8_properties() {
    let status = MoonshotStatus::from_repo();
    let cert =
        MoonshotCertificate::from_status(&status, "kokoro-v1", "English text, <=50 words", "hash");
    assert_eq!(cert.properties.len(), 8);
    assert_eq!(cert.model_name, "kokoro-v1");
    assert!(!cert.verification_date.is_empty());
    assert_eq!(
        cert.schema_version,
        crate::moonshot::CERTIFICATE_SCHEMA_VERSION
    );
}

#[test]
fn test_moonshot_certificate_property_indices_sequential() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "spec", "hash");
    for (i, prop) in cert.properties.iter().enumerate() {
        assert_eq!(
            prop.property_index, i,
            "property at position {i} should have index {i}"
        );
    }
}

#[test]
fn test_moonshot_certificate_property_names_match_constants() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "spec", "hash");
    for (i, prop) in cert.properties.iter().enumerate() {
        assert_eq!(
            prop.property_name, PROPERTY_NAMES[i],
            "property {i} name should match PROPERTY_NAMES"
        );
    }
}

// ===========================================================================
// 5. Moonshot property checks (P1-P8)
// ===========================================================================

#[test]
fn test_p1_non_silence_proven_on_valid_pipeline() {
    let cert = make_valid_pipeline();
    let result = crown_check_non_silence(&cert, 0.01);
    assert_eq!(result.property_index, 0);
    assert_eq!(result.property_name, PROPERTY_NAMES[0]);
    assert!(
        result.proven,
        "pipeline with output bounds [-1,1] should prove non-silence"
    );
    assert!(result.is_sound);
    assert!(
        result.level >= VerificationLevel::CrownPartial,
        "level should be at least CrownPartial"
    );
}

#[test]
fn test_p1_non_silence_not_proven_on_zero_bounds() {
    // Pipeline with output bounds [0.0, 0.0] -- all zeros
    let s1 = make_stage("enc", &[4], (-1.0, 1.0), (0.0, 0.0), "CROWN", true);
    let s2 = make_stage("dec", &[4], (0.0, 0.0), (0.0, 0.0), "CROWN", true);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    let result = crown_check_non_silence(&cert, 0.01);
    assert!(
        !result.proven,
        "zero output bounds should not prove non-silence"
    );
}

#[test]
fn test_p2_non_clipping_proven_on_bounded_pipeline() {
    let cert = make_valid_pipeline();
    let result = crown_check_non_clipping(&cert);
    assert_eq!(result.property_index, 1);
    assert_eq!(result.property_name, PROPERTY_NAMES[1]);
    assert!(
        result.proven,
        "pipeline with output bounds [-1,1] should prove non-clipping"
    );
}

#[test]
fn test_p2_non_clipping_fails_on_wide_bounds() {
    let s1 = make_stage("enc", &[4], (-1.0, 1.0), (-0.5, 0.5), "CROWN", true);
    let s2 = make_stage("dec", &[4], (-0.5, 0.5), (-2.0, 2.0), "CROWN", true);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    let result = crown_check_non_clipping(&cert);
    assert!(
        !result.proven,
        "output bounds [-2,2] should not prove non-clipping"
    );
}

#[test]
fn test_p3_intelligibility_proxy_with_tight_bounds() {
    let cert = make_valid_pipeline();
    let result = check_intelligibility_proxy(&cert, 10.0);
    assert_eq!(result.property_index, 2);
    assert_eq!(result.property_name, PROPERTY_NAMES[2]);
    // The proxy checks output range / input range < max_range_ratio
    // With input range 2.0 and output range 2.0, ratio = 1.0 < 10.0
    assert!(
        result.proven,
        "tight bounds should prove intelligibility proxy"
    );
}

#[test]
fn test_p5_temporal_boundedness_within_timing_bound() {
    use crate::pipeline::TimingCertificate;

    let bounds_cert = make_valid_pipeline();
    let timing_cert = TimingCertificate::new(
        bounds_cert,
        vec![],
        50_000.0,  // 50ms worst case
        1_000_000, // 1M FLOPs
        500_000,   // 500KB
        "M4 Max",
        100_000.0, // 100ms bound
        true,      // within bound
        true,      // overall passed
        None,
    );

    let result = check_temporal_boundedness(&timing_cert);
    assert_eq!(result.property_index, 4);
    assert_eq!(result.property_name, PROPERTY_NAMES[4]);
    assert!(
        result.proven,
        "50ms < 100ms should prove temporal boundedness"
    );
}

#[test]
fn test_p5_temporal_boundedness_exceeds_bound() {
    use crate::pipeline::TimingCertificate;

    let bounds_cert = make_valid_pipeline();
    let timing_cert = TimingCertificate::new(
        bounds_cert,
        vec![],
        150_000.0, // 150ms worst case (exceeds bound)
        1_000_000,
        500_000,
        "M4 Max",
        100_000.0, // 100ms bound
        false,     // NOT within bound
        false,     // overall failed
        None,
    );

    let result = check_temporal_boundedness(&timing_cert);
    assert!(
        !result.proven,
        "150ms > 100ms should not prove temporal boundedness"
    );
}

#[test]
fn test_p6_streaming_safety_with_crossfade() {
    let cert = make_valid_pipeline();
    // Output range is 2.0 (from -1.0 to 1.0), crossfade of 240 samples:
    // alpha_step = 1/239, max_click = 2.0/239 ~ 0.0084
    let result = check_streaming_safety(&cert, 240, 0.05);
    assert_eq!(result.property_index, 5);
    assert_eq!(result.property_name, PROPERTY_NAMES[5]);
    assert!(
        result.proven,
        "crossfade bound should be within click threshold"
    );
}

#[test]
fn test_p6_streaming_safety_degenerate_crossfade() {
    let cert = make_valid_pipeline();
    // crossfade_samples = 1 -> alpha_step = 1.0 -> max_click = output_range
    let result = check_streaming_safety(&cert, 1, 0.05);
    assert!(
        !result.proven,
        "degenerate crossfade (1 sample) should not prove streaming safety"
    );
}

#[test]
fn test_moonshot_property_result_has_explanation() {
    let cert = make_valid_pipeline();
    let result = crown_check_non_clipping(&cert);
    assert!(
        !result.explanation.is_empty(),
        "property result should have a non-empty explanation"
    );
    assert!(
        result.explanation.contains("PROVEN") || result.explanation.contains("NOT PROVEN"),
        "explanation should indicate proof status"
    );
}

// ===========================================================================
// Pipeline composition tests
// ===========================================================================

#[test]
fn test_pipeline_verify_valid_two_stages() {
    let cert = make_valid_pipeline();
    assert!(cert.is_valid);
    assert!(cert.is_sound);
    assert_eq!(cert.stages.len(), 2);
    assert_eq!(cert.junctions.len(), 1);
    assert!(cert.junctions[0].bounds_contained);
}

#[test]
fn test_pipeline_verify_incompatible_bounds() {
    // Stage 1 outputs [-2, 2], but stage 2 expects [-0.5, 0.5]
    let s1 = make_stage("enc", &[4], (-1.0, 1.0), (-2.0, 2.0), "CROWN", true);
    let s2 = make_stage("dec", &[4], (-0.5, 0.5), (-1.0, 1.0), "CROWN", true);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(
        !cert.is_valid,
        "incompatible bounds should invalidate pipeline"
    );
    assert!(!cert.junctions[0].bounds_contained);
    assert!(
        cert.junctions[0].max_violation > 0.0,
        "should report positive violation"
    );
}

#[test]
fn test_pipeline_verify_insufficient_stages() {
    let s1 = make_stage("enc", &[4], (-1.0, 1.0), (-0.5, 0.5), "CROWN", true);
    let result = verify_pipeline(&[s1]);
    assert!(result.is_err(), "single stage should fail");
}

#[test]
fn test_pipeline_verify_three_stages() {
    let s1 = make_stage("prosody", &[8], (-1.0, 1.0), (-0.8, 0.8), "CROWN", true);
    let s2 = make_stage("decoder", &[8], (-0.8, 0.8), (-0.5, 0.5), "CROWN", true);
    let s3 = make_stage("vocoder", &[8], (-0.5, 0.5), (-1.0, 1.0), "CROWN", true);
    let cert = verify_pipeline(&[s1, s2, s3]).unwrap();
    assert!(cert.is_valid);
    assert_eq!(cert.junctions.len(), 2);
}

#[test]
fn test_pipeline_unsound_stage_makes_pipeline_unsound() {
    let s1 = make_stage("enc", &[4], (-1.0, 1.0), (-0.5, 0.5), "CROWN", true);
    let s2 = make_stage("dec", &[4], (-0.5, 0.5), (-1.0, 1.0), "IBP", false);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(cert.is_valid);
    assert!(
        !cert.is_sound,
        "pipeline with unsound stage should be unsound"
    );
}

#[test]
fn test_pipeline_report_is_nonempty() {
    let cert = make_valid_pipeline();
    let report = cert.report();
    assert!(report.contains("Pipeline Verification Report"));
    assert!(report.contains("encoder"));
    assert!(report.contains("decoder"));
    assert!(report.contains("End-to-end bounds"));
}

#[test]
fn test_check_junction_nan_in_bounds_is_violation() {
    let s1 = VerifiedStage::new(
        "stage_a",
        vec![2],
        vec![2],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        vec![f64::NAN, 0.5],
        vec![0.5, 0.5],
        "CROWN",
        true,
    );
    let s2 = make_stage("stage_b", &[2], (-1.0, 1.0), (-1.0, 1.0), "CROWN", true);
    let junction = check_junction(&s1, &s2, 0);
    assert!(
        !junction.bounds_contained,
        "NaN in output bounds should be a violation"
    );
    assert!(junction.violation_count > 0);
}

// ===========================================================================
// Moonshot status and artifact registry
// ===========================================================================

#[test]
fn test_moonshot_status_has_8_properties() {
    let status = MoonshotStatus::from_repo();
    assert_eq!(status.properties.len(), 8);
}

#[test]
fn test_moonshot_status_property_names_match_constants() {
    let status = MoonshotStatus::from_repo();
    for (i, prop) in status.properties.iter().enumerate() {
        assert_eq!(prop.name, PROPERTY_NAMES[i]);
    }
}

#[test]
fn test_moonshot_status_level_counts_sum_to_8() {
    let status = MoonshotStatus::from_repo();
    let counts = status.level_counts();
    let total: usize = counts.iter().map(|(_, c)| c).sum();
    assert_eq!(total, 8, "level counts should sum to 8 properties");
}

#[test]
fn test_moonshot_status_report_nonempty() {
    let status = MoonshotStatus::from_repo();
    let report = status.report();
    assert!(report.contains("Moonshot Status"));
    assert!(report.contains("Property 1"));
    assert!(report.contains("Property 8"));
    assert!(report.contains("Summary"));
}

#[test]
fn test_artifact_registry_nonempty() {
    let artifacts = artifact_registry();
    assert!(
        !artifacts.is_empty(),
        "artifact registry should have verification artifacts"
    );
    // Each artifact should have a valid property index
    for a in &artifacts {
        for &idx in a.properties {
            assert!(idx < 8, "artifact property index {idx} should be < 8");
        }
        assert!(!a.description.is_empty());
        assert!(!a.file.is_empty());
    }
}

#[test]
fn test_verification_level_ordering() {
    assert!(VerificationLevel::None < VerificationLevel::Empirical);
    assert!(VerificationLevel::Empirical < VerificationLevel::CrownPartial);
    assert!(VerificationLevel::CrownPartial < VerificationLevel::CrownProbabilistic);
    assert!(VerificationLevel::CrownProbabilistic < VerificationLevel::CrownProven);
    assert!(VerificationLevel::CrownProven < VerificationLevel::KaniProven);
    assert!(VerificationLevel::KaniProven < VerificationLevel::SmtProven);
}

#[test]
fn test_verification_level_display() {
    assert_eq!(format!("{}", VerificationLevel::None), "NONE");
    assert_eq!(format!("{}", VerificationLevel::Empirical), "EMPIRICAL");
    assert_eq!(
        format!("{}", VerificationLevel::CrownPartial),
        "CROWN_PARTIAL"
    );
    assert_eq!(
        format!("{}", VerificationLevel::CrownProbabilistic),
        "CROWN_PROBABILISTIC"
    );
    assert_eq!(
        format!("{}", VerificationLevel::CrownProven),
        "CROWN_PROVEN"
    );
    assert_eq!(format!("{}", VerificationLevel::KaniProven), "KANI_PROVEN");
    assert_eq!(format!("{}", VerificationLevel::SmtProven), "SMT_PROVEN");
}
