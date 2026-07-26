// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended pipeline verification tests for nn-tts-verify.
//!
//! Covers audio quality metric computation, pipeline contract validation,
//! cost model estimates, certificate properties, stage boundary validation,
//! streaming verification, quantization certificates, quality bounds,
//! deterministic hashing, and edge cases (empty, silence, clipped, short/long).

use crate::bounds::{
    check_duration, check_no_clicks, check_no_dc_offset, check_non_silence, check_nyquist, check_tail_energy, HardBound, SpectralCoverageConfig,
};
use crate::certificate::Certificate;
use crate::cost_model::{
    total_estimated_time_us, total_flops, total_memory_bytes, HardwareCostModel, LayerCostProfile,
};
use crate::crown_junction::{
    check_junction_bound, contract_bounds_map, JunctionCheckSummary, StageBoundCheck,
};
use crate::deterministic::{pcm_sha256, DeterministicCert, DeterministicMeta};
use crate::kokoro_contracts::{
    all_contracts, bounds_within_contract, contract_stage, max_contract_violation,
    JunctionContract, J2_F0_LOWER, J2_F0_UPPER, J5_AUDIO_LOWER, J5_AUDIO_UPPER,
};
use crate::monotonicity::interpret_duration_positivity;
use crate::moonshot::{
    artifact_registry, MoonshotCertificate, MoonshotStatus, VerificationLevel, PROPERTY_NAMES,
};
use crate::moonshot_crown::{
    check_non_clipping as crown_check_non_clipping,
    check_temporal_boundedness, verify_properties_from_pipeline,
};
use crate::pipeline::{check_junction, verify_pipeline, TimingCertificate, VerifiedStage};
use crate::quality::{
    check_f0_range, compute_hnr, compute_mcd, compute_rms, compute_sdr, compute_snr,
};
use crate::quality_bound::{
    cosine_similarity_lipschitz, mcd_lipschitz, snr_lipschitz, spectral_convergence_lipschitz,
    standard_quality_specs, verify_quality_bounds, QualityMetricSpec,
};
use crate::quantization_certificate::{
    build_quantization_certificate, build_segment_result, compute_element_drift,
};
use crate::streaming::{crossfade_linear, verify_streaming, StreamingConfig};
use crate::test_audio_helpers::{sine_wave, sine_wave_full, sine_wave_samples};

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

/// Generate broadband synthetic speech-like audio with multiple harmonics.
fn synthetic_speech(sample_rate: u32, duration_sec: f64) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec) as usize;
    let mut signal = vec![0.0_f32; n];
    let freqs = [220.0, 440.0, 880.0, 1760.0, 3520.0];
    let amps = [0.25, 0.20, 0.15, 0.10, 0.06];
    for (i, sample) in signal.iter_mut().enumerate() {
        let t = i as f64 / f64::from(sample_rate);
        for (&f, &a) in freqs.iter().zip(amps.iter()) {
            *sample += (a * (2.0 * std::f64::consts::PI * f * t).sin()) as f32;
        }
    }
    signal
}

/// Generate a chunk of audio for streaming tests.
fn make_chunk(freq_hz: f64, sample_rate: u32, n_samples: usize, amplitude: f32) -> Vec<f32> {
    sine_wave_samples(freq_hz, sample_rate, n_samples)
        .into_iter()
        .map(|s| s * amplitude)
        .collect()
}

// ===========================================================================
// Section 1: Audio Quality Metric Computation
// ===========================================================================

#[test]
fn test_snr_identical_signals_very_high() {
    // Identical candidate and reference should have very high SNR
    let signal = sine_wave(440.0, 24000, 0.5);
    let result = compute_snr(&signal, &signal, 10.0).unwrap();
    assert!(result.passed, "identical signals should have high SNR");
    // SNR of identical signals is theoretically infinite, but due to float
    // representation it should at least be very high
    assert!(
        result.value > 100.0,
        "SNR should be very high, got {}",
        result.value
    );
}

#[test]
fn test_snr_noisy_signal_lower() {
    let signal = sine_wave(440.0, 24000, 0.5);
    let noisy: Vec<f32> = signal
        .iter()
        .enumerate()
        .map(|(i, &s)| s + 0.1 * ((i as f32 * 0.7).sin()))
        .collect();
    let result = compute_snr(&noisy, &signal, 5.0).unwrap();
    assert!(result.value.is_finite(), "SNR should be finite");
    assert!(
        result.value > 5.0,
        "noisy signal should still have decent SNR"
    );
}

#[test]
fn test_rms_of_sine_wave_amplitude() {
    let samples = sine_wave_full(440.0, 24000, 1.0, 0.5);
    let result = compute_rms(&samples, 0.01).unwrap();
    assert!(result.passed);
    // RMS of sine wave with amplitude A is A/sqrt(2) ~ 0.354
    assert!(
        (result.value - 0.354).abs() < 0.02,
        "RMS of 0.5 amplitude sine should be ~0.354, got {}",
        result.value
    );
}

#[test]
fn test_rms_empty_input_error() {
    let result = compute_rms(&[], 0.01);
    assert!(result.is_err(), "empty input should return error");
}

#[test]
fn test_mcd_same_signal_near_zero() {
    let signal = synthetic_speech(24000, 0.5);
    let result = compute_mcd(&signal, &signal, 24000, 6.0).unwrap();
    assert!(
        result.value < 0.01,
        "MCD of identical signals should be near zero, got {}",
        result.value
    );
    assert!(result.passed);
}

#[test]
fn test_sdr_identical_signals_high() {
    let signal = sine_wave(440.0, 24000, 0.5);
    let result = compute_sdr(&signal, &signal, 5.0).unwrap();
    assert!(result.passed);
    assert!(
        result.value > 100.0,
        "SDR of identical signals should be very high"
    );
}

#[test]
fn test_hnr_on_periodic_signal() {
    let signal = sine_wave(220.0, 24000, 0.5);
    let result = compute_hnr(&signal, 24000, 5.0).unwrap();
    // Pure sine wave is highly periodic, so HNR should be high
    assert!(result.value.is_finite(), "HNR should be finite");
    assert!(result.passed, "periodic signal should have HNR > 5 dB");
}

#[test]
fn test_f0_range_all_voiced_in_range() {
    let contour = vec![120.0, 130.0, 140.0, 150.0, 160.0];
    let result = check_f0_range(&contour, 80.0, 400.0);
    assert!(
        result.passed,
        "all F0 values within speech range should pass"
    );
    assert!(
        (result.value - 1.0).abs() < 1e-10,
        "all voiced frames in range"
    );
}

#[test]
fn test_f0_range_out_of_range() {
    // All values outside normal speech range
    let contour = vec![20.0, 25.0, 30.0]; // Below 80 Hz
    let result = check_f0_range(&contour, 80.0, 400.0);
    assert!(!result.passed, "F0 below speech range should fail");
}

#[test]
fn test_f0_range_unvoiced_frames_excluded() {
    // 0.0 = unvoiced, should be excluded from ratio
    let contour = vec![0.0, 0.0, 120.0, 0.0, 130.0];
    let result = check_f0_range(&contour, 80.0, 400.0);
    assert!(result.passed, "unvoiced frames (0.0) should be excluded");
    assert!((result.value - 1.0).abs() < 1e-10);
}

#[test]
fn test_f0_range_all_unvoiced_fails() {
    let contour = vec![0.0, 0.0, 0.0];
    let result = check_f0_range(&contour, 80.0, 400.0);
    assert!(!result.passed, "all unvoiced should fail");
}

// ===========================================================================
// Section 2: Hard Bounds Edge Cases
// ===========================================================================

#[test]
fn test_dc_offset_on_zero_mean_sine() {
    let samples = sine_wave(440.0, 24000, 1.0);
    let bound = check_no_dc_offset(&samples, 0.01);
    assert!(bound.passed, "sine wave should have near-zero DC offset");
    assert!(bound.value < 0.001, "DC offset should be very small");
}

#[test]
fn test_dc_offset_on_biased_signal() {
    let samples: Vec<f32> = sine_wave(440.0, 24000, 1.0)
        .into_iter()
        .map(|s| s + 0.5)
        .collect();
    let bound = check_no_dc_offset(&samples, 0.1);
    assert!(!bound.passed, "biased signal should fail DC offset check");
    assert!((bound.value - 0.5).abs() < 0.01, "DC offset should be ~0.5");
}

#[test]
fn test_click_detection_on_smooth_signal() {
    let samples = sine_wave_full(200.0, 24000, 0.5, 0.5);
    let bound = check_no_clicks(&samples, 0.5);
    assert!(bound.passed, "smooth sine should have no clicks");
}

#[test]
fn test_click_detection_on_discontinuous_signal() {
    let mut samples = vec![0.0_f32; 1000];
    samples[500] = 1.0; // Sudden spike
    let bound = check_no_clicks(&samples, 0.5);
    assert!(!bound.passed, "sudden spike should be detected as click");
    assert!(bound.value >= 1.0, "max diff should be >= 1.0");
}

#[test]
fn test_tail_energy_normal_speech() {
    let samples = synthetic_speech(24000, 1.0);
    let bound = check_tail_energy(&samples, 24000, 50.0, 500.0, 3.0);
    assert!(
        bound.passed,
        "normal speech should not have tail energy spike"
    );
}

#[test]
fn test_tail_energy_with_spike() {
    let mut samples = synthetic_speech(24000, 1.0);
    let n = samples.len();
    // Add a huge spike at the tail
    for s in samples[n - 500..].iter_mut() {
        *s *= 10.0;
    }
    let bound = check_tail_energy(&samples, 24000, 50.0, 500.0, 2.0);
    assert!(!bound.passed, "tail spike should fail energy check");
}

#[test]
fn test_nyquist_on_low_freq_signal() {
    let samples = synthetic_speech(24000, 0.5);
    let result = check_nyquist(&samples, 24000);
    assert!(result.is_ok());
    let bound = result.unwrap();
    assert!(
        bound.passed,
        "speech-like signal should not have excessive Nyquist energy"
    );
}

#[test]
fn test_spectral_coverage_config_validation() {
    let config = SpectralCoverageConfig {
        n_bands: 0,
        min_energy_db: -60.0,
        min_coverage: 0.5,
    };
    assert!(
        config.validate().is_err(),
        "zero bands should fail validation"
    );
}

#[test]
fn test_duration_with_very_long_audio() {
    // 120 seconds
    let samples = vec![0.1_f32; 24000 * 120];
    let bound = check_duration(&samples, 24000, 0.1, 100.0);
    assert!(!bound.passed, "120s exceeds 100s max");
}

#[test]
fn test_non_silence_very_quiet_signal() {
    // Very quiet signal: amplitude 0.001
    let samples = sine_wave_full(440.0, 24000, 0.5, 0.001);
    let bound = check_non_silence(&samples, 0.01);
    assert!(
        !bound.passed,
        "very quiet signal should fail non-silence at 0.01 threshold"
    );
    let bound_low = check_non_silence(&samples, 0.0001);
    assert!(
        bound_low.passed,
        "very quiet signal should pass with lower threshold"
    );
}

// ===========================================================================
// Section 3: Cost Model Estimates
// ===========================================================================

#[test]
fn test_cost_model_m4_max_conservative_validation() {
    let model = HardwareCostModel::m4_max_conservative();
    assert!(model.validate().is_ok());
    assert!(model.peak_tflops_f32 > 0.0);
    assert!(model.peak_tflops_f32 < HardwareCostModel::m4_max().peak_tflops_f32);
}

#[test]
fn test_cost_model_zero_flops_zero_mem() {
    let model = HardwareCostModel::m4_max();
    let time = model.estimate_time_us(0, 0);
    assert!(
        (time - model.dispatch_overhead_us).abs() < 1e-10,
        "zero work should just be dispatch overhead"
    );
}

#[test]
fn test_cost_model_large_matmul_dominated_by_compute() {
    let model = HardwareCostModel::m4_max();
    // Large matmul: 10 billion FLOPs, small memory
    let time = model.estimate_time_us(10_000_000_000, 1_000);
    // Compute time = 10e9 / (14.2e6) ~ 704 us
    let compute_only = 10_000_000_000_f64 / (14.2 * 1e6);
    assert!(
        time >= compute_only,
        "time should be >= compute time, got {time} vs {compute_only}"
    );
}

#[test]
fn test_layer_cost_profile_without_measured() {
    let profile = LayerCostProfile::new("conv1d", 500_000, 100_000, 3.5, None);
    assert_eq!(profile.measured_time_us, None);
    assert_eq!(profile.flops, 500_000);
}

#[test]
fn test_total_flops_empty_profiles() {
    let profiles: Vec<LayerCostProfile> = vec![];
    assert_eq!(total_flops(&profiles), 0);
    assert_eq!(total_memory_bytes(&profiles), 0);
    assert!((total_estimated_time_us(&profiles)).abs() < 1e-10);
}

#[test]
fn test_total_estimated_time_accumulates() {
    let profiles = vec![
        LayerCostProfile::new("a", 100, 50, 10.0, None),
        LayerCostProfile::new("b", 200, 75, 20.0, None),
        LayerCostProfile::new("c", 300, 25, 30.0, None),
    ];
    assert!((total_estimated_time_us(&profiles) - 60.0).abs() < 1e-10);
}

#[test]
fn test_cost_model_validation_rejects_zero_bandwidth() {
    let model = HardwareCostModel {
        peak_tflops_f32: 10.0,
        peak_bandwidth_gbs: 0.0,
        dispatch_overhead_us: 5.0,
    };
    assert!(model.validate().is_err(), "zero bandwidth should fail");
}

// ===========================================================================
// Section 4: Pipeline Contract Validation
// ===========================================================================

#[test]
fn test_all_kokoro_contracts_have_finite_bounds() {
    for c in &all_contracts() {
        assert!(
            c.lower.is_finite(),
            "contract {} lower is not finite",
            c.name
        );
        assert!(
            c.upper.is_finite(),
            "contract {} upper is not finite",
            c.name
        );
        assert!(c.lower < c.upper, "contract {} lower >= upper", c.name);
    }
}

#[test]
fn test_j2_f0_contract_values() {
    assert!((J2_F0_LOWER - (-5.0)).abs() < 1e-10);
    assert!((J2_F0_UPPER - 800.0).abs() < 1e-10);
}

#[test]
fn test_j5_audio_contract_is_pcm_range() {
    assert!((J5_AUDIO_LOWER - (-1.0)).abs() < 1e-10);
    assert!((J5_AUDIO_UPPER - 1.0).abs() < 1e-10);
}

#[test]
fn test_bounds_within_contract_empty_arrays() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    assert!(bounds_within_contract(&contract, &[], &[]));
}

#[test]
fn test_bounds_within_contract_exact_boundary() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let lower = vec![-1.0, -1.0];
    let upper = vec![1.0, 1.0];
    assert!(bounds_within_contract(&contract, &lower, &upper));
}

#[test]
fn test_bounds_within_contract_infinity_is_violation() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let lower = vec![f64::NEG_INFINITY];
    let upper = vec![0.5];
    assert!(!bounds_within_contract(&contract, &lower, &upper));
}

#[test]
fn test_max_violation_underflow() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let lower = vec![-1.5]; // underflows by 0.5
    let upper = vec![0.5];
    let violation = max_contract_violation(&contract, &lower, &upper);
    assert!(
        (violation - 0.5).abs() < 1e-10,
        "expected 0.5, got {violation}"
    );
}

#[test]
fn test_contract_stage_different_input_output_shapes() {
    let j_in = JunctionContract::new("in", "zone", -5.0, 5.0);
    let j_out = JunctionContract::new("out", "zone", -1.0, 1.0);
    let stage = contract_stage("mixer", &[2, 4], &[3, 2], &j_in, &j_out, "IBP", false);
    assert_eq!(stage.input_lower.len(), 8); // 2*4
    assert_eq!(stage.output_lower.len(), 6); // 3*2
    assert!(!stage.is_sound);
    assert_eq!(stage.method, "IBP");
}

#[test]
fn test_check_junction_bound_exact_boundary_passes() {
    let result = check_junction_bound("test", -1.0, 1.0, -1.0, 1.0);
    assert!(result.passed, "exact boundary values should pass");
}

#[test]
fn test_check_junction_bound_epsilon_violation_fails() {
    let result = check_junction_bound("test", -1.0, 1.0, -1.0, 1.001);
    assert!(!result.passed, "slight overflow should fail");
}

// ===========================================================================
// Section 5: Pipeline Composition Verification
// ===========================================================================

#[test]
fn test_pipeline_five_stage_valid_chain() {
    let s1 = make_stage("s1", &[8], (-1.0, 1.0), (-0.9, 0.9), "CROWN", true);
    let s2 = make_stage("s2", &[8], (-0.9, 0.9), (-0.7, 0.7), "CROWN", true);
    let s3 = make_stage("s3", &[8], (-0.7, 0.7), (-0.5, 0.5), "CROWN", true);
    let s4 = make_stage("s4", &[8], (-0.5, 0.5), (-0.3, 0.3), "CROWN", true);
    let s5 = make_stage("s5", &[8], (-0.3, 0.3), (-1.0, 1.0), "CROWN", true);
    let cert = verify_pipeline(&[s1, s2, s3, s4, s5]).unwrap();
    assert!(cert.is_valid);
    assert!(cert.is_sound);
    assert_eq!(cert.junctions.len(), 4);
    assert_eq!(cert.stages.len(), 5);
}

#[test]
fn test_pipeline_middle_junction_fails() {
    let s1 = make_stage("s1", &[4], (-1.0, 1.0), (-0.5, 0.5), "CROWN", true);
    let s2 = make_stage("s2", &[4], (-0.5, 0.5), (-2.0, 2.0), "CROWN", true); // outputs wide
    let s3 = make_stage("s3", &[4], (-0.3, 0.3), (-1.0, 1.0), "CROWN", true); // expects narrow
    let cert = verify_pipeline(&[s1, s2, s3]).unwrap();
    assert!(!cert.is_valid, "junction between s2 and s3 should fail");
    assert!(cert.junctions[0].bounds_contained, "s1->s2 should be ok");
    assert!(!cert.junctions[1].bounds_contained, "s2->s3 should fail");
}

#[test]
fn test_pipeline_shape_incompatibility() {
    let s1 = VerifiedStage::new(
        "s1",
        vec![4],
        vec![6], // output shape 6 elements
        vec![-1.0; 4],
        vec![1.0; 4],
        vec![-0.5; 6],
        vec![0.5; 6],
        "CROWN",
        true,
    );
    let s2 = VerifiedStage::new(
        "s2",
        vec![4], // input shape 4 elements -- mismatch!
        vec![4],
        vec![-0.5; 4],
        vec![0.5; 4],
        vec![-1.0; 4],
        vec![1.0; 4],
        "CROWN",
        true,
    );
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(!cert.junctions[0].shape_compatible);
}

#[test]
fn test_pipeline_e2e_bounds_match_first_and_last() {
    let s1 = make_stage("enc", &[4], (-2.0, 2.0), (-1.0, 1.0), "CROWN", true);
    let s2 = make_stage("dec", &[4], (-1.0, 1.0), (-0.5, 0.5), "CROWN", true);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(cert
        .e2e_input_lower
        .iter()
        .all(|&v| (v - (-2.0)).abs() < 1e-10));
    assert!(cert
        .e2e_output_upper
        .iter()
        .all(|&v| (v - 0.5).abs() < 1e-10));
}

#[test]
fn test_pipeline_display_trait() {
    let s1 = make_stage("a", &[4], (-1.0, 1.0), (-0.5, 0.5), "CROWN", true);
    let s2 = make_stage("b", &[4], (-0.5, 0.5), (-1.0, 1.0), "CROWN", true);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    let display = format!("{cert}");
    assert!(display.contains("2 stages"));
    assert!(display.contains("valid=true"));
}

// ===========================================================================
// Section 6: Moonshot Certificate Properties (P1-P8)
// ===========================================================================

#[test]
fn test_moonshot_status_all_have_evidence() {
    let status = MoonshotStatus::from_repo();
    // The artifact registry should populate evidence for all properties
    assert!(
        status.all_have_evidence(),
        "all properties should have evidence"
    );
}

#[test]
fn test_moonshot_status_at_least_crown_partial() {
    let status = MoonshotStatus::from_repo();
    // Given the state of the project, most properties should be at least CrownPartial
    assert!(
        status.all_at_least_crown_partial(),
        "all properties should be at least CrownPartial"
    );
}

#[test]
fn test_moonshot_certificate_serialization_roundtrip() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "kokoro-v1", "test spec", "hash123");
    let json = cert.to_json_serde().expect("serialize moonshot cert");
    let deserialized = MoonshotCertificate::from_json(&json).expect("deserialize moonshot cert");
    assert_eq!(deserialized.model_name, cert.model_name);
    assert_eq!(deserialized.properties.len(), 8);
    assert_eq!(deserialized.schema_version, cert.schema_version);
}

#[test]
fn test_moonshot_property_result_proven_has_crown_level() {
    let s1 = make_stage("enc", &[4], (-1.0, 1.0), (-0.5, 0.5), "CROWN", true);
    let s2 = make_stage("dec", &[4], (-0.5, 0.5), (-0.8, 0.8), "CROWN", true);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    let result = crown_check_non_clipping(&cert);
    assert!(result.proven);
    assert!(result.level >= VerificationLevel::CrownPartial);
    assert!(result.is_sound);
}

#[test]
fn test_moonshot_all_properties_from_pipeline() {
    let s1 = make_stage("enc", &[4], (-1.0, 1.0), (-0.5, 0.5), "CROWN", true);
    let s2 = make_stage("dec", &[4], (-0.5, 0.5), (-0.9, 0.9), "CROWN", true);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    let bundle = verify_properties_from_pipeline(&cert, 4);
    // Should have results for P1-P3, P5-P6 (properties verifiable from bounds)
    assert!(
        !bundle.results.is_empty(),
        "should produce property results"
    );
    for r in &bundle.results {
        assert!(r.property_index < 8);
        assert!(!r.explanation.is_empty());
    }
}

#[test]
fn test_p5_temporal_boundedness_tight_bound() {
    let bounds_cert = {
        let s1 = make_stage("enc", &[4], (-1.0, 1.0), (-0.5, 0.5), "CROWN", true);
        let s2 = make_stage("dec", &[4], (-0.5, 0.5), (-1.0, 1.0), "CROWN", true);
        verify_pipeline(&[s1, s2]).unwrap()
    };
    // Exactly at the boundary
    let timing_cert = TimingCertificate::new(
        bounds_cert,
        vec![],
        100_000.0, // 100ms exactly == bound
        1_000_000,
        500_000,
        "M4 Max",
        100_000.0, // 100ms bound
        true,
        true,
        None,
    );
    let result = check_temporal_boundedness(&timing_cert);
    assert!(result.proven, "exactly at bound should still be proven");
}

// ===========================================================================
// Section 7: Quality Bound Verification (Lipschitz)
// ===========================================================================

#[test]
fn test_snr_lipschitz_computation() {
    // signal_rms = 1.0, baseline_snr = 20 dB
    // noise_rms = 1.0 * 10^(-20/20) = 0.1
    // L = 20 / (ln(10) * 0.1) ~ 86.86
    let l = snr_lipschitz(1.0, 20.0).unwrap();
    assert!(
        (l - 86.86).abs() < 1.0,
        "SNR Lipschitz for 20 dB should be ~86.86, got {l}"
    );
}

#[test]
fn test_snr_lipschitz_rejects_zero_rms() {
    assert!(snr_lipschitz(0.0, 20.0).is_err());
}

#[test]
fn test_snr_lipschitz_rejects_nan() {
    assert!(snr_lipschitz(f64::NAN, 20.0).is_err());
}

#[test]
fn test_spectral_convergence_lipschitz() {
    let l = spectral_convergence_lipschitz(10.0).unwrap();
    assert!((l - 0.1).abs() < 1e-10, "L = 1/10 = 0.1, got {l}");
}

#[test]
fn test_mcd_lipschitz_single_frame() {
    let l = mcd_lipschitz(1).unwrap();
    let expected = 10.0 * 2.0_f64.sqrt() / 10.0_f64.ln();
    assert!(
        (l - expected).abs() < 0.01,
        "single frame MCD Lipschitz should be {expected}, got {l}"
    );
}

#[test]
fn test_mcd_lipschitz_decreases_with_frames() {
    let l1 = mcd_lipschitz(1).unwrap();
    let l100 = mcd_lipschitz(100).unwrap();
    assert!(
        l100 < l1,
        "more frames should give smaller Lipschitz constant"
    );
}

#[test]
fn test_mcd_lipschitz_zero_frames_error() {
    assert!(mcd_lipschitz(0).is_err());
}

#[test]
fn test_cosine_similarity_lipschitz() {
    let l = cosine_similarity_lipschitz(5.0).unwrap();
    assert!((l - 0.2).abs() < 1e-10, "L = 1/5 = 0.2, got {l}");
}

#[test]
fn test_verify_quality_bounds_guaranteed() {
    let metrics = vec![QualityMetricSpec {
        name: "SNR".into(),
        lipschitz_constant: 10.0,
        baseline_value: 25.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    // delta = 0.5, max_change = 10 * 0.5 = 5.0
    // worst_case = 25.0 - 5.0 = 20.0 > 10.0 threshold
    let cert = verify_quality_bounds(0.5, &metrics).unwrap();
    assert!(cert.all_guaranteed);
    assert!((cert.metric_results[0].margin - 10.0).abs() < 1e-10);
}

#[test]
fn test_verify_quality_bounds_not_guaranteed() {
    let metrics = vec![QualityMetricSpec {
        name: "SNR".into(),
        lipschitz_constant: 100.0,
        baseline_value: 15.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    // delta = 0.5, max_change = 100 * 0.5 = 50.0
    // worst_case = 15.0 - 50.0 = -35.0 < 10.0 threshold
    let cert = verify_quality_bounds(0.5, &metrics).unwrap();
    assert!(!cert.all_guaranteed);
    assert!(cert.metric_results[0].margin < 0.0);
}

#[test]
fn test_verify_quality_bounds_negative_width_error() {
    let metrics = vec![QualityMetricSpec {
        name: "test".into(),
        lipschitz_constant: 1.0,
        baseline_value: 10.0,
        threshold: 5.0,
        higher_is_better: true,
        citation: "test",
    }];
    assert!(verify_quality_bounds(-1.0, &metrics).is_err());
}

#[test]
fn test_standard_quality_specs_returns_four() {
    let specs = standard_quality_specs(0.5, 10.0, 100.0, 50, 25.0, 0.1, 3.0, 0.95).unwrap();
    assert_eq!(
        specs.len(),
        4,
        "standard specs should have SNR, SC, MCD, cosine"
    );
    let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"SNR"));
    assert!(names.contains(&"spectral_convergence"));
    assert!(names.contains(&"MCD"));
    assert!(names.contains(&"cosine_similarity"));
}

// ===========================================================================
// Section 8: Streaming Verification
// ===========================================================================

#[test]
fn test_crossfade_linear_basic() {
    let tail = vec![1.0_f32, 1.0, 1.0, 1.0];
    let head = vec![0.0_f32, 0.0, 0.0, 0.0];
    let blended = crossfade_linear(&tail, &head).unwrap();
    assert_eq!(blended.len(), 4);
    // First sample should be dominated by tail, last by head
    assert!(blended[0] > 0.5, "first sample should favor tail");
    assert!(blended[3] < 0.5, "last sample should favor head");
}

#[test]
fn test_crossfade_linear_length_mismatch_error() {
    assert!(crossfade_linear(&[1.0, 2.0], &[1.0]).is_err());
}

#[test]
fn test_crossfade_linear_single_sample() {
    let blended = crossfade_linear(&[1.0], &[0.0]).unwrap();
    assert_eq!(blended.len(), 1);
}

#[test]
fn test_streaming_config_default_valid() {
    let config = StreamingConfig::default();
    assert!(config.validate().is_ok());
    assert_eq!(config.sample_rate, 24000);
    assert_eq!(config.crossfade_samples, 960);
}

#[test]
fn test_streaming_config_invalid_margin() {
    let config = StreamingConfig {
        margin_samples: 100, // less than crossfade_samples
        ..StreamingConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_streaming_verify_two_smooth_chunks() {
    let config = StreamingConfig {
        sample_rate: 24000,
        crossfade_samples: 480,
        margin_samples: 960,
        click_threshold: 0.5,
        energy_lo: 0.3,
        energy_hi: 3.0,
        spectral_threshold: 0.5,
    };
    let chunk1 = make_chunk(440.0, 24000, 4800, 0.5); // 200ms
    let chunk2 = make_chunk(440.0, 24000, 4800, 0.5); // 200ms
    let cert = verify_streaming(&[&chunk1, &chunk2], &config).unwrap();
    assert_eq!(cert.n_chunks, 2);
    assert_eq!(cert.boundaries.len(), 1);
}

#[test]
fn test_streaming_verify_single_chunk_error() {
    let config = StreamingConfig::default();
    let chunk = make_chunk(440.0, 24000, 4800, 0.5);
    assert!(verify_streaming(&[&chunk[..]], &config).is_err());
}

// ===========================================================================
// Section 9: Quantization Certificate
// ===========================================================================

#[test]
fn test_compute_element_drift_identical() {
    let lower = vec![0.0_f32, -1.0, 0.5];
    let upper = vec![1.0_f32, 0.0, 1.5];
    let (max_drift, mean_drift, n) = compute_element_drift(&lower, &upper, &lower, &upper).unwrap();
    assert_eq!(n, 3);
    assert!(
        max_drift.abs() < 1e-10,
        "identical bounds should have zero drift"
    );
    assert!(mean_drift.abs() < 1e-10);
}

#[test]
fn test_compute_element_drift_with_shift() {
    let f32_lower = vec![0.0_f32, 0.0];
    let f32_upper = vec![1.0_f32, 1.0];
    let q_lower = vec![0.1_f32, 0.0]; // shifted by 0.1
    let q_upper = vec![1.0_f32, 1.2]; // shifted by 0.2
    let (max_drift, _mean_drift, _n) =
        compute_element_drift(&f32_lower, &f32_upper, &q_lower, &q_upper).unwrap();
    assert!(
        (max_drift - 0.2).abs() < 1e-5,
        "max drift should be 0.2, got {max_drift}"
    );
}

#[test]
fn test_compute_element_drift_length_mismatch() {
    assert!(compute_element_drift(&[0.0], &[1.0], &[0.0, 0.0], &[1.0, 1.0]).is_err());
}

#[test]
fn test_compute_element_drift_empty_error() {
    assert!(compute_element_drift(&[], &[], &[], &[]).is_err());
}

#[test]
fn test_compute_element_drift_nan_error() {
    assert!(
        compute_element_drift(&[f32::NAN], &[1.0], &[0.0], &[1.0]).is_err(),
        "NaN in bounds should error"
    );
}

#[test]
fn test_build_segment_result_basic() {
    let f32_l = vec![-1.0_f32, -0.5];
    let f32_u = vec![1.0_f32, 0.5];
    let q_l = vec![-1.1_f32, -0.5];
    let q_u = vec![1.0_f32, 0.6];
    let result = build_segment_result("encoder", &f32_l, &f32_u, &q_l, &q_u).unwrap();
    assert_eq!(result.segment_name, "encoder");
    assert_eq!(result.num_elements, 2);
    assert!(result.max_element_drift > 0.0);
}

#[test]
fn test_build_quantization_certificate_quality_preserved() {
    let f32_l = vec![0.0_f32; 10];
    let f32_u = vec![1.0_f32; 10];
    let q_l = vec![0.001_f32; 10]; // tiny drift
    let q_u = vec![1.001_f32; 10];

    let segment = build_segment_result("test_seg", &f32_l, &f32_u, &q_l, &q_u).unwrap();

    let quality_specs = vec![QualityMetricSpec {
        name: "SNR".into(),
        lipschitz_constant: 10.0,
        baseline_value: 30.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];

    let cert =
        build_quantization_certificate("F32", "BF16", vec![segment], &quality_specs).unwrap();
    assert!(cert.quality_preserved, "tiny drift should preserve quality");
    assert_eq!(cert.source_dtype, "F32");
    assert_eq!(cert.target_dtype, "BF16");
}

// ===========================================================================
// Section 10: Deterministic Hashing
// ===========================================================================

#[test]
fn test_pcm_sha256_deterministic() {
    let audio = vec![0.1_f32, -0.3, 0.5, 0.0, 1.0];
    let h1 = pcm_sha256(&audio);
    let h2 = pcm_sha256(&audio);
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 64, "SHA-256 hex should be 64 chars");
}

#[test]
fn test_deterministic_cert_verification() {
    let audio = sine_wave(440.0, 24000, 0.1);
    let cert = DeterministicCert::from_audio(
        &audio,
        DeterministicMeta {
            input_text: Some("hello".into()),
            voice_id: None,
            seed: Some(42),
        },
    );
    assert!(cert.verify(&audio));
    // Modified audio should not verify
    let mut modified = audio;
    modified[0] += 0.001;
    assert!(!cert.verify(&modified));
}

#[test]
fn test_pcm_sha256_endianness_consistent() {
    // Verify the hash uses LE encoding consistently
    let samples = vec![1.0_f32, -1.0, 0.0];
    let h = pcm_sha256(&samples);
    // Just verify it is a valid hex string
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
}

// ===========================================================================
// Section 11: Duration Positivity Certificate
// ===========================================================================

#[test]
fn test_duration_positivity_proven() {
    let cert = interpret_duration_positivity(-5.0, -10.0, 1.0, 1.0, 1, "CROWN");
    assert!(cert.is_proven, "-5 > -10 should be proven");
    assert!((cert.lower_bound - (-5.0)).abs() < 1e-10);
    assert_eq!(cert.propagation_mode, "CROWN");
}

#[test]
fn test_duration_positivity_not_proven() {
    let cert = interpret_duration_positivity(-15.0, -10.0, 1.0, 1.0, 1, "IBP");
    assert!(!cert.is_proven, "-15 < -10 should not be proven");
}

#[test]
fn test_duration_positivity_sequence_length_metadata() {
    let cert = interpret_duration_positivity(-5.0, -10.0, 2.0, 0.5, 4, "alpha-CROWN");
    assert_eq!(cert.sequence_length, 4);
    assert!((cert.input_bound - 2.0).abs() < 1e-10);
    assert!((cert.style_bound - 0.5).abs() < 1e-10);
}

// ===========================================================================
// Section 12: Certificate Composition and Enrichment
// ===========================================================================

#[test]
fn test_certificate_enrichment_chain() {
    let status = MoonshotStatus::from_repo();
    let moonshot = MoonshotCertificate::from_status(&status, "test", "spec", "hash");
    let junction_summary = JunctionCheckSummary {
        checks: vec![StageBoundCheck {
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
        hard_bounds: vec![HardBound {
            name: "non_silence",
            passed: true,
            value: 0.5,
            threshold: 0.01,
        }],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: Some("abc".to_string()),
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };

    // Chain enrichment
    let enriched = cert
        .with_crown_evidence(moonshot)
        .with_junction_summary(junction_summary);
    assert!(enriched.has_crown_evidence());
    assert!(enriched.has_junction_summary());
    assert!(enriched.passes_hard_bounds());
    assert!(enriched.passes_junction_contracts());

    // Report should include all sections
    let report = enriched.report();
    assert!(report.contains("TTS Verification Certificate"));
    assert!(report.contains("Hard Bounds"));
    assert!(report.contains("CROWN Verification Evidence"));
    assert!(report.contains("Junction Contract Checks"));
    assert!(report.contains("Deterministic Hash"));
}

#[test]
fn test_certificate_quality_metrics_fail() {
    let cert = Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![crate::quality::QualityMetric {
            name: "snr",
            value: 5.0,
            threshold: 10.0,
            passed: false,
            citation: "test",
        }],
        phoneme_results: None,
        overall_passed: false,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };
    assert!(!cert.passes_quality());
    assert!(cert.passes_hard_bounds()); // vacuously true (no hard bounds)
}

// ===========================================================================
// Section 13: Verification Level Semantics
// ===========================================================================

#[test]
fn test_verification_level_total_ordering() {
    let levels = [
        VerificationLevel::None,
        VerificationLevel::Empirical,
        VerificationLevel::CrownPartial,
        VerificationLevel::CrownProbabilistic,
        VerificationLevel::CrownProven,
        VerificationLevel::KaniProven,
        VerificationLevel::SmtProven,
    ];
    for i in 0..levels.len() - 1 {
        assert!(
            levels[i] < levels[i + 1],
            "{:?} should be less than {:?}",
            levels[i],
            levels[i + 1]
        );
    }
}

#[test]
fn test_verification_level_serde_roundtrip() {
    let level = VerificationLevel::CrownProbabilistic;
    let json = serde_json::to_string(&level).unwrap();
    let deserialized: VerificationLevel = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, level);
}

// ===========================================================================
// Section 14: Artifact Registry
// ===========================================================================

#[test]
fn test_artifact_registry_all_property_indices_valid() {
    let artifacts = artifact_registry();
    for a in &artifacts {
        for &idx in a.properties {
            assert!(
                idx < 8,
                "property index {idx} out of range in artifact: {}",
                a.description
            );
        }
    }
}

#[test]
fn test_artifact_registry_covers_all_properties() {
    let artifacts = artifact_registry();
    let mut covered = [false; 8];
    for a in &artifacts {
        for &idx in a.properties {
            covered[idx] = true;
        }
    }
    for (i, &c) in covered.iter().enumerate() {
        assert!(c, "property {i} ({}) has no artifacts", PROPERTY_NAMES[i]);
    }
}

#[test]
fn test_artifact_registry_no_empty_descriptions() {
    let artifacts = artifact_registry();
    for a in &artifacts {
        assert!(!a.description.is_empty(), "artifact has empty description");
        assert!(!a.file.is_empty(), "artifact has empty file path");
    }
}

// ===========================================================================
// Section 15: Edge Cases and Error Handling
// ===========================================================================

#[test]
fn test_pipeline_report_with_nan_bounds() {
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
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    // Pipeline should still produce a report without panicking
    let report = cert.report();
    assert!(report.contains("Pipeline Verification Report"));
    assert!(!cert.is_valid, "NaN bounds should invalidate pipeline");
}

#[test]
fn test_junction_with_inf_bounds() {
    let s1 = VerifiedStage::new(
        "s1",
        vec![1],
        vec![1],
        vec![-1.0],
        vec![1.0],
        vec![f64::NEG_INFINITY],
        vec![f64::INFINITY],
        "IBP",
        false,
    );
    let s2 = make_stage("s2", &[1], (-1.0, 1.0), (-1.0, 1.0), "CROWN", true);
    let junction = check_junction(&s1, &s2, 0);
    assert!(
        !junction.bounds_contained,
        "infinite bounds should be a violation"
    );
    assert_eq!(junction.max_violation, f64::MAX);
}

#[test]
fn test_contract_bounds_map_values_match_constants() {
    let map = contract_bounds_map();
    let j2_f0 = map.get("J2_F0").unwrap();
    assert!((j2_f0.0 - J2_F0_LOWER as f32).abs() < 1e-5);
    assert!((j2_f0.1 - J2_F0_UPPER as f32).abs() < 1e-5);
    let j5 = map.get("J5_AUDIO").unwrap();
    assert!((j5.0 - J5_AUDIO_LOWER as f32).abs() < 1e-5);
    assert!((j5.1 - J5_AUDIO_UPPER as f32).abs() < 1e-5);
}
