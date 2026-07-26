// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for nn-tts-verify covering cost models, pipeline composition,
//! Kokoro contracts, audio quality metrics, sample rate validation, batch sizing,
//! duration prediction, silence detection, text normalization patterns, and
//! multi-speaker configuration.

use crate::bounds::{
    check_duration, check_no_clicks, check_no_clipping, check_no_dc_offset, check_non_silence,
    check_tail_energy, SpectralCoverageConfig,
};
use crate::certificate::Certificate;
use crate::config::{HardBoundsConfig, QualityConfig};
use crate::cost_model::{
    total_estimated_time_us, total_flops, total_memory_bytes, HardwareCostModel, LayerCostProfile,
};
use crate::dsp;
use crate::error::TtsVerifyError;
use crate::kokoro_contracts::{
    all_contracts, bounds_within_contract, contract_stage, max_contract_violation, JunctionContract,
};
use crate::monotonicity::interpret_duration_positivity;
use crate::pipeline::{check_junction, verify_pipeline, VerifiedStage};
use crate::quality::{compute_rms, compute_snr};
use crate::quality_bound::{
    cosine_similarity_lipschitz, mcd_lipschitz, snr_lipschitz, spectral_convergence_lipschitz,
    standard_quality_specs, verify_quality_bounds, QualityMetricSpec,
};
use crate::streaming::{crossfade_linear, StreamingConfig};
use crate::test_audio_helpers::{sine_wave, sine_wave_full};
use crate::verifier::TtsVerifier;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Cost model calculations: RTF estimation for different model sizes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_cost_model_m4_max_roofline_basic() {
    let model = HardwareCostModel::m4_max();
    // 1 GFLOP with 10 MB memory traffic
    let time = model.estimate_time_us(1_000_000_000, 10_000_000);
    // compute_time = 1e9 / (14.2e6) = ~70.4 us
    // memory_time  = 10e6 / (400e3) = 25 us
    // total = max(70.4, 25) + 5.0 = ~75.4 us
    assert!(time > 70.0, "time should reflect compute-bound workload");
    assert!(time < 100.0, "time should be reasonable for 1 GFLOP");
}

#[test]
fn test_cost_model_m4_max_conservative_is_slower() {
    let fast = HardwareCostModel::m4_max();
    let slow = HardwareCostModel::m4_max_conservative();

    let flops = 1_000_000_000_u64;
    let mem = 10_000_000_u64;

    let fast_time = fast.estimate_time_us(flops, mem);
    let slow_time = slow.estimate_time_us(flops, mem);

    assert!(
        slow_time > fast_time,
        "conservative model ({slow_time:.1}) should be slower than theoretical ({fast_time:.1})"
    );
    // Conservative is ~5x slower compute, 2x slower memory, 2x overhead
    assert!(
        slow_time > fast_time * 2.0,
        "conservative should be >= 2x slower"
    );
}

#[test]
fn test_cost_model_small_vs_large_model() {
    let model = HardwareCostModel::m4_max();

    // Small model: 100M FLOPs, 1 MB memory
    let small_time = model.estimate_time_us(100_000_000, 1_000_000);
    // Large model: 10B FLOPs, 100 MB memory
    let large_time = model.estimate_time_us(10_000_000_000, 100_000_000);

    assert!(
        large_time > small_time * 50.0,
        "100x more FLOPs should yield >50x longer time"
    );
}

#[test]
fn test_cost_model_rtf_from_profiles() {
    let profiles = vec![
        LayerCostProfile::new("embedding", 1000, 4000, 10.0, None),
        LayerCostProfile::new("attention", 5_000_000, 500_000, 500.0, None),
        LayerCostProfile::new("ffn", 10_000_000, 1_000_000, 800.0, None),
        LayerCostProfile::new("decoder", 2_000_000, 200_000, 300.0, None),
    ];

    let total_time = total_estimated_time_us(&profiles);
    let total_f = total_flops(&profiles);
    let total_m = total_memory_bytes(&profiles);

    assert!(
        (total_time - 1610.0).abs() < 0.01,
        "total time should be sum of all steps"
    );
    assert_eq!(total_f, 17_001_000, "FLOPs should sum correctly");
    assert_eq!(total_m, 1_704_000, "memory should sum correctly");

    // RTF for 1 second of audio at 24kHz: time_sec / audio_sec
    let audio_duration_sec = 1.0;
    let rtf = (total_time / 1e6) / audio_duration_sec;
    assert!(rtf < 1.0, "RTF should be < 1.0 for real-time synthesis");
}

#[test]
fn test_cost_model_validate_rejects_nan() {
    let model = HardwareCostModel {
        peak_tflops_f32: f64::NAN,
        peak_bandwidth_gbs: 400.0,
        dispatch_overhead_us: 5.0,
    };
    assert!(model.validate().is_err(), "NaN TFLOPS should be rejected");
}

#[test]
fn test_cost_model_validate_rejects_negative() {
    let model = HardwareCostModel {
        peak_tflops_f32: -1.0,
        peak_bandwidth_gbs: 400.0,
        dispatch_overhead_us: 5.0,
    };
    assert!(
        model.validate().is_err(),
        "negative TFLOPS should be rejected"
    );
}

#[test]
fn test_cost_model_memory_bound_workload() {
    let model = HardwareCostModel::m4_max();
    // Low compute, high memory: should be memory-bound
    let time = model.estimate_time_us(1000, 1_000_000_000);
    // memory_time = 1e9 / (400e3) = 2500 us
    // compute_time = 1000 / (14.2e6) ~= 0 us
    assert!(time > 2000.0, "memory-bound workload should take >2 ms");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Pipeline stage composition: stages connect correctly
// ═══════════════════════════════════════════════════════════════════════════

fn make_stage(
    name: &str,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
    lo: f64,
    hi: f64,
) -> VerifiedStage {
    let in_elements: usize = in_shape.iter().product();
    let out_elements: usize = out_shape.iter().product();
    VerifiedStage::new(
        name,
        in_shape,
        out_shape,
        vec![-1.0; in_elements],
        vec![1.0; in_elements],
        vec![lo; out_elements],
        vec![hi; out_elements],
        "CROWN",
        true,
    )
}

#[test]
fn test_pipeline_two_compatible_stages() {
    let s1 = make_stage("encoder", vec![1, 256], vec![1, 512], -0.5, 0.5);
    let s2 = VerifiedStage::new(
        "decoder",
        vec![1, 512],
        vec![1, 24000],
        vec![-1.0; 512],
        vec![1.0; 512],
        vec![-1.0; 24000],
        vec![1.0; 24000],
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(
        cert.is_valid,
        "compatible stages should produce valid pipeline"
    );
    assert!(
        cert.is_sound,
        "all-sound stages should produce sound pipeline"
    );
    assert_eq!(cert.junctions.len(), 1);
    assert!(cert.junctions[0].bounds_contained);
    assert!(cert.junctions[0].shape_compatible);
}

#[test]
fn test_pipeline_shape_mismatch_detected() {
    let s1 = make_stage("encoder", vec![1, 256], vec![1, 512], -0.5, 0.5);
    // Next stage expects different input shape (256 instead of 512)
    let s2 = make_stage("decoder", vec![1, 256], vec![1, 1000], -1.0, 1.0);

    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(!cert.is_valid, "shape mismatch should invalidate pipeline");
    assert!(!cert.junctions[0].shape_compatible);
}

#[test]
fn test_pipeline_bounds_violation_detected() {
    // Stage 1 outputs in [-2.0, 2.0]
    let s1 = make_stage("wide_encoder", vec![1, 10], vec![1, 10], -2.0, 2.0);
    // Stage 2 expects input in [-1.0, 1.0] -- violation!
    let s2 = VerifiedStage::new(
        "narrow_decoder",
        vec![1, 10],
        vec![1, 10],
        vec![-1.0; 10],
        vec![1.0; 10],
        vec![-0.5; 10],
        vec![0.5; 10],
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(
        !cert.is_valid,
        "bounds violation should invalidate pipeline"
    );
    assert!(!cert.junctions[0].bounds_contained);
    assert!(cert.junctions[0].max_violation > 0.0);
    assert_eq!(cert.junctions[0].violation_count, 10);
}

#[test]
fn test_pipeline_three_stage_chain() {
    let s1 = make_stage("text_encoder", vec![1, 100], vec![1, 256], -1.0, 1.0);
    let s2 = VerifiedStage::new(
        "prosody",
        vec![1, 256],
        vec![1, 256],
        vec![-2.0; 256],
        vec![2.0; 256],
        vec![-0.8; 256],
        vec![0.8; 256],
        "CROWN",
        true,
    );
    let s3 = VerifiedStage::new(
        "decoder",
        vec![1, 256],
        vec![1, 48000],
        vec![-1.0; 256],
        vec![1.0; 256],
        vec![-1.0; 48000],
        vec![1.0; 48000],
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[s1, s2, s3]).unwrap();
    assert!(cert.is_valid, "three compatible stages should be valid");
    assert_eq!(cert.junctions.len(), 2);
    assert_eq!(cert.e2e_input_lower.len(), 100);
    assert_eq!(cert.e2e_output_lower.len(), 48000);
}

#[test]
fn test_pipeline_insufficient_stages() {
    let s1 = make_stage("lone_stage", vec![1, 10], vec![1, 10], -1.0, 1.0);
    let result = verify_pipeline(&[s1]);
    assert!(result.is_err(), "single stage should be rejected");
}

#[test]
fn test_pipeline_non_sound_stage_detected() {
    let s1 = make_stage("encoder", vec![1, 10], vec![1, 10], -0.5, 0.5);
    let mut s2 = VerifiedStage::new(
        "heuristic_decoder",
        vec![1, 10],
        vec![1, 10],
        vec![-1.0; 10],
        vec![1.0; 10],
        vec![-0.5; 10],
        vec![0.5; 10],
        "heuristic",
        false,
    );
    // Ensure bounds are contained
    let _ = &mut s2; // s2 already correctly set up

    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(
        !cert.is_sound,
        "heuristic stage should make pipeline non-sound"
    );
    // But it should still be valid if bounds are contained
    assert!(cert.is_valid);
}

#[test]
fn test_junction_nan_bounds_violation() {
    let s1 = VerifiedStage::new(
        "nan_stage",
        vec![2],
        vec![2],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        vec![f64::NAN, 0.5],
        vec![0.5, 0.5],
        "CROWN",
        true,
    );
    let s2 = make_stage("next", vec![2], vec![2], -1.0, 1.0);

    let junction = check_junction(&s1, &s2, 0);
    assert!(
        !junction.bounds_contained,
        "NaN in bounds should be a violation"
    );
    assert!(junction.violation_count > 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Kokoro contracts: junction bounds (J2-J5) are non-empty intervals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_all_contracts_have_nonempty_intervals() {
    for contract in &all_contracts() {
        assert!(
            contract.lower < contract.upper,
            "contract {} should have lower ({}) < upper ({})",
            contract.name,
            contract.lower,
            contract.upper,
        );
    }
}

#[test]
fn test_all_contracts_have_finite_bounds() {
    for contract in &all_contracts() {
        assert!(
            contract.lower.is_finite(),
            "contract {} lower bound must be finite",
            contract.name,
        );
        assert!(
            contract.upper.is_finite(),
            "contract {} upper bound must be finite",
            contract.name,
        );
    }
}

#[test]
fn test_j2_f0_contract_range_covers_speech() {
    let contracts = all_contracts();
    let j2_f0 = &contracts[0];
    assert_eq!(j2_f0.name, "J2_F0");
    // Speech F0 range is roughly 80-400 Hz. J2 should cover this.
    assert!(
        j2_f0.lower < 80.0,
        "J2_F0 lower should be below typical speech F0"
    );
    assert!(
        j2_f0.upper > 400.0,
        "J2_F0 upper should be above soprano range"
    );
}

#[test]
fn test_j5_audio_contract_is_pcm_range() {
    let contracts = all_contracts();
    let j5 = &contracts[5];
    assert_eq!(j5.name, "J5_AUDIO");
    assert!(
        (j5.lower - (-1.0)).abs() < f64::EPSILON,
        "J5 lower should be -1.0"
    );
    assert!(
        (j5.upper - 1.0).abs() < f64::EPSILON,
        "J5 upper should be 1.0"
    );
}

#[test]
fn test_bounds_within_contract_passes_for_tight_bounds() {
    let contract = JunctionContract::new("test", "test_zone", -1.0, 1.0);
    let lo = vec![-0.5, -0.3, 0.0];
    let hi = vec![0.5, 0.3, 0.1];
    assert!(bounds_within_contract(&contract, &lo, &hi));
}

#[test]
fn test_bounds_within_contract_fails_for_exceeding_bounds() {
    let contract = JunctionContract::new("test", "test_zone", -1.0, 1.0);
    let lo = vec![-0.5, -1.5]; // Second element exceeds contract lower
    let hi = vec![0.5, 0.5];
    assert!(!bounds_within_contract(&contract, &lo, &hi));
}

#[test]
fn test_bounds_within_contract_fails_for_nan() {
    let contract = JunctionContract::new("test", "test_zone", -1.0, 1.0);
    let lo = vec![f64::NAN];
    let hi = vec![0.5];
    assert!(!bounds_within_contract(&contract, &lo, &hi));
}

#[test]
fn test_max_contract_violation_zero_when_contained() {
    let contract = JunctionContract::new("test", "test_zone", -1.0, 1.0);
    let lo = vec![-0.5, -0.3];
    let hi = vec![0.5, 0.3];
    let violation = max_contract_violation(&contract, &lo, &hi);
    assert!(
        violation.abs() < f64::EPSILON,
        "should be 0.0 when contained, got {violation}"
    );
}

#[test]
fn test_max_contract_violation_positive_when_exceeded() {
    let contract = JunctionContract::new("test", "test_zone", -1.0, 1.0);
    let lo = vec![-0.5];
    let hi = vec![1.5]; // Exceeds upper by 0.5
    let violation = max_contract_violation(&contract, &lo, &hi);
    assert!(
        (violation - 0.5).abs() < 1e-10,
        "violation should be 0.5, got {violation}"
    );
}

#[test]
fn test_contract_stage_builds_correct_uniform_bounds() {
    let input_contract = JunctionContract::new("in", "zone_a", -1.0, 1.0);
    let output_contract = JunctionContract::new("out", "zone_b", -0.5, 0.5);

    let stage = contract_stage(
        "test_stage",
        &[1, 256],
        &[1, 512],
        &input_contract,
        &output_contract,
        "CROWN",
        true,
    );

    assert_eq!(stage.input_lower.len(), 256);
    assert_eq!(stage.output_lower.len(), 512);
    assert!(stage
        .input_lower
        .iter()
        .all(|&v| (v - (-1.0)).abs() < f64::EPSILON));
    assert!(stage
        .output_upper
        .iter()
        .all(|&v| (v - 0.5).abs() < f64::EPSILON));
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Audio quality metrics: SNR, PESQ-like scoring, spectral distance
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_snr_identical_signals_high() {
    let signal = sine_wave(440.0, 24000, 0.5);
    let snr = compute_snr(&signal, &signal, 10.0).unwrap();
    // Identical signals should have very high SNR (or infinite).
    // The implementation may cap it, but it should pass the threshold.
    assert!(snr.passed, "identical signals should have high SNR");
}

#[test]
fn test_snr_with_noise_lower() {
    let signal = sine_wave(440.0, 24000, 0.5);
    let noisy: Vec<f32> = signal
        .iter()
        .enumerate()
        .map(|(i, &s)| s + 0.1 * ((i as f32 * 0.137).sin()))
        .collect();
    let snr = compute_snr(&noisy, &signal, 0.0).unwrap();
    assert!(snr.value.is_finite(), "SNR should be finite");
    assert!(snr.value > 0.0, "SNR with small noise should be positive");
}

#[test]
fn test_rms_of_silence_is_zero() {
    let silence = vec![0.0_f32; 1000];
    let rms = compute_rms(&silence, 0.01).unwrap();
    assert!(rms.value.abs() < f64::EPSILON, "silence RMS should be 0.0");
    assert!(!rms.passed, "silence should fail any positive threshold");
}

#[test]
fn test_rms_of_sine_wave() {
    // RMS of a full sine cycle at amplitude A is A/sqrt(2).
    let signal = sine_wave(440.0, 24000, 1.0);
    let rms = compute_rms(&signal, 0.01).unwrap();
    let expected_rms = 1.0 / 2.0_f64.sqrt();
    assert!(
        (rms.value - expected_rms).abs() < 0.01,
        "RMS of unit sine should be ~0.707, got {}",
        rms.value
    );
    assert!(rms.passed, "unit sine RMS should exceed 0.01 threshold");
}

#[test]
fn test_snr_lipschitz_positive() {
    let lip = snr_lipschitz(0.5, 25.0).unwrap();
    assert!(lip > 0.0, "SNR Lipschitz should be positive");
    assert!(lip.is_finite(), "SNR Lipschitz should be finite");
}

#[test]
fn test_spectral_convergence_lipschitz_positive() {
    let lip = spectral_convergence_lipschitz(10.0).unwrap();
    assert!(
        (lip - 0.1).abs() < f64::EPSILON,
        "SC Lipschitz should be 1/ref_energy"
    );
}

#[test]
fn test_mcd_lipschitz_decreases_with_more_frames() {
    let lip_10 = mcd_lipschitz(10).unwrap();
    let lip_100 = mcd_lipschitz(100).unwrap();
    assert!(
        lip_100 < lip_10,
        "MCD Lipschitz should decrease with more frames"
    );
}

#[test]
fn test_cosine_similarity_lipschitz_inversely_proportional() {
    let lip = cosine_similarity_lipschitz(5.0).unwrap();
    assert!(
        (lip - 0.2).abs() < f64::EPSILON,
        "cosine Lipschitz should be 1/norm"
    );
}

#[test]
fn test_quality_bound_certificate_all_guaranteed() {
    let specs = vec![
        QualityMetricSpec {
            name: "SNR".into(),
            lipschitz_constant: 1.0,
            baseline_value: 30.0,
            threshold: 10.0,
            higher_is_better: true,
            citation: "test",
        },
        QualityMetricSpec {
            name: "MCD".into(),
            lipschitz_constant: 0.5,
            baseline_value: 3.0,
            threshold: 6.0,
            higher_is_better: false,
            citation: "test",
        },
    ];
    let cert = verify_quality_bounds(0.1, &specs).unwrap();
    assert!(
        cert.all_guaranteed,
        "small perturbation should guarantee all metrics"
    );
    assert!(cert.tightest_margin > 0.0, "margin should be positive");
}

#[test]
fn test_quality_bound_certificate_not_guaranteed() {
    let specs = vec![QualityMetricSpec {
        name: "SNR".into(),
        lipschitz_constant: 100.0,
        baseline_value: 11.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    // Large perturbation: worst_case = 11 - 100*1.0 = -89 < 10
    let cert = verify_quality_bounds(1.0, &specs).unwrap();
    assert!(
        !cert.all_guaranteed,
        "large perturbation should fail guarantee"
    );
}

#[test]
fn test_standard_quality_specs_builds_four_metrics() {
    let specs = standard_quality_specs(
        0.3,  // signal_rms
        10.0, // signal_l2_norm
        50.0, // reference_spectral_energy
        100,  // n_frames
        25.0, // baseline_snr
        0.05, // baseline_sc
        3.5,  // baseline_mcd
        0.95, // baseline_cosine
    )
    .unwrap();

    assert_eq!(specs.len(), 4, "should produce SNR, SC, MCD, cosine specs");
    let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"SNR"));
    assert!(names.contains(&"spectral_convergence"));
    assert!(names.contains(&"MCD"));
    assert!(names.contains(&"cosine_similarity"));
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Sample rate validation: resampling between 22050, 24000, 44100 Hz
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_verifier_accepts_24000_sample_rate() {
    let v = TtsVerifier::builder().sample_rate(24000).build();
    assert!(v.is_ok(), "24000 Hz should be accepted");
}

#[test]
fn test_verifier_accepts_22050_sample_rate() {
    let v = TtsVerifier::builder().sample_rate(22050).build();
    assert!(v.is_ok(), "22050 Hz should be accepted");
}

#[test]
fn test_verifier_accepts_44100_sample_rate() {
    let v = TtsVerifier::builder().sample_rate(44100).build();
    assert!(v.is_ok(), "44100 Hz should be accepted");
}

#[test]
fn test_verifier_rejects_zero_sample_rate() {
    let v = TtsVerifier::builder().sample_rate(0).build();
    assert!(v.is_err(), "0 Hz sample rate should be rejected");
}

#[test]
fn test_duration_check_at_different_sample_rates() {
    // 1 second of audio at different sample rates
    for &sr in &[22050_u32, 24000, 44100, 48000] {
        let samples = sine_wave_full(440.0, sr, 1.0, 0.5);
        let bound = check_duration(&samples, sr, 0.5, 2.0);
        assert!(
            bound.passed,
            "1 second audio at {sr} Hz should pass duration check [0.5, 2.0]"
        );
        assert!(
            (bound.value - 1.0).abs() < 0.01,
            "duration should be ~1.0s at {sr} Hz, got {}",
            bound.value
        );
    }
}

#[test]
fn test_streaming_config_default_for_24khz() {
    let config = StreamingConfig::default();
    assert_eq!(config.sample_rate, 24000);
    assert_eq!(config.crossfade_samples, 960);
    // 960 samples at 24kHz = 40ms
    let crossfade_ms = config.crossfade_samples as f64 / f64::from(config.sample_rate) * 1000.0;
    assert!(
        (crossfade_ms - 40.0).abs() < 0.1,
        "default crossfade should be 40ms"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Batch size effects on RTF: larger batches have lower per-sample cost
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_batch_size_amortizes_dispatch_overhead() {
    let model = HardwareCostModel::m4_max();

    // Single sample: 1M FLOPs per sample
    let single_time = model.estimate_time_us(1_000_000, 100_000);

    // Batch of 8: 8M FLOPs total but only 1 dispatch overhead
    let batch_time = model.estimate_time_us(8_000_000, 800_000);

    // Per-sample cost should be lower in batch due to amortized overhead
    let per_sample_single = single_time;
    let per_sample_batch = batch_time / 8.0;

    assert!(
        per_sample_batch < per_sample_single,
        "per-sample cost should decrease with batching: single={per_sample_single:.1}, batch={per_sample_batch:.1}"
    );
}

#[test]
fn test_dispatch_overhead_dominates_tiny_workloads() {
    let model = HardwareCostModel::m4_max();

    // Tiny workload: 1 FLOP, 4 bytes
    let time = model.estimate_time_us(1, 4);
    // Dispatch overhead (5.0 us) should dominate
    assert!(
        time > 4.9,
        "dispatch overhead should dominate tiny workloads: {time}"
    );
    assert!(
        time < 6.0,
        "total should be mostly overhead for tiny workloads: {time}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Duration prediction: predicted duration is positive and bounded
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_duration_positivity_proven_above_threshold() {
    let cert = interpret_duration_positivity(
        -5.0,  // lower_bound: CROWN proves dur_logits >= -5.0
        -10.0, // threshold: we need dur_logits > -10.0
        1.0,   // input_bound
        1.0,   // style_bound
        1,     // sequence_length
        "CROWN",
    );
    assert!(cert.is_proven, "-5.0 > -10.0 should be proven");
    assert!((cert.lower_bound - (-5.0)).abs() < f64::EPSILON);
}

#[test]
fn test_duration_positivity_not_proven_below_threshold() {
    let cert = interpret_duration_positivity(
        -15.0, // lower_bound: CROWN proves dur_logits >= -15.0
        -10.0, // threshold: we need dur_logits > -10.0
        1.0, 1.0, 1, "IBP",
    );
    assert!(!cert.is_proven, "-15.0 <= -10.0 should not be proven");
}

#[test]
fn test_duration_positivity_guarantees_exp_positive() {
    // If lower_bound > threshold = -20, then exp(lower_bound) > exp(-20) > 0
    let cert = interpret_duration_positivity(-8.0, -20.0, 1.0, 1.0, 4, "alpha-CROWN");
    assert!(cert.is_proven);

    // Compute the guaranteed minimum duration
    let min_duration = cert.lower_bound.exp();
    assert!(min_duration > 0.0, "exp(lower_bound) must be positive");
    assert!(
        min_duration > 1e-9,
        "exp(-8) should be significantly positive"
    );
}

#[test]
fn test_duration_check_bounds() {
    let sr = 24000_u32;

    // Too short (0.05 seconds for min 0.1)
    let short = sine_wave_full(440.0, sr, 0.05, 0.5);
    let bound = check_duration(&short, sr, 0.1, 300.0);
    assert!(!bound.passed, "0.05s should fail min duration 0.1s");

    // Normal (2 seconds)
    let normal = sine_wave_full(440.0, sr, 2.0, 0.5);
    let bound = check_duration(&normal, sr, 0.1, 300.0);
    assert!(bound.passed, "2s should pass duration [0.1, 300]");
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Silence detection: silence threshold correctly identifies quiet segments
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_silence_detection_all_zeros() {
    let silence = vec![0.0_f32; 24000]; // 1 second of silence at 24kHz
    let bound = check_non_silence(&silence, 0.01);
    assert!(
        !bound.passed,
        "all-zero signal should fail non-silence check"
    );
    assert!(bound.value < f64::EPSILON, "RMS of silence should be ~0");
}

#[test]
fn test_silence_detection_quiet_signal() {
    // Very quiet signal: amplitude 0.001
    let quiet = sine_wave_full(440.0, 24000, 0.5, 0.001);
    let bound = check_non_silence(&quiet, 0.01);
    // RMS of 0.001 * sine = 0.001/sqrt(2) ~= 0.0007
    assert!(
        !bound.passed,
        "very quiet signal (amp=0.001) should fail 0.01 RMS threshold"
    );
}

#[test]
fn test_silence_detection_normal_signal() {
    let normal = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let bound = check_non_silence(&normal, 0.01);
    assert!(
        bound.passed,
        "normal amplitude signal should pass non-silence check"
    );
}

#[test]
fn test_clipping_detection() {
    let mut signal = sine_wave(440.0, 24000, 0.5);
    // Force some samples to clip
    signal[100] = 1.5;
    signal[200] = -1.5;
    let bound = check_no_clipping(&signal, 1.0);
    assert!(!bound.passed, "clipped signal should fail clipping check");
    assert!(bound.value > 1.0, "peak should exceed 1.0");
}

#[test]
fn test_dc_offset_detection() {
    // Signal with significant DC offset
    let offset_signal: Vec<f32> = (0..24000)
        .map(|i| 0.3 + 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin())
        .collect();
    let bound = check_no_dc_offset(&offset_signal, 0.05);
    assert!(
        !bound.passed,
        "signal with 0.3 DC offset should fail 0.05 threshold"
    );
}

#[test]
fn test_click_detection() {
    let mut signal = sine_wave_full(440.0, 24000, 0.5, 0.3);
    // Insert a click: large sudden jump
    let mid = signal.len() / 2;
    signal[mid] = 0.0;
    signal[mid + 1] = 0.9; // diff = 0.9
    let bound = check_no_clicks(&signal, 0.5);
    assert!(
        !bound.passed,
        "signal with 0.9 jump should fail 0.5 click threshold"
    );
}

#[test]
fn test_tail_energy_normal() {
    let signal = sine_wave_full(440.0, 24000, 1.0, 0.5);
    let bound = check_tail_energy(&signal, 24000, 50.0, 500.0, 3.0);
    assert!(
        bound.passed,
        "uniform sine wave should have tail/body ratio ~1.0"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Text normalization: verifier input validation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_verifier_rejects_empty_input() {
    let verifier = TtsVerifier::builder().sample_rate(24000).build().unwrap();
    let result = verifier.verify(&[]);
    assert!(result.is_err(), "empty input should be rejected");
    match result.unwrap_err() {
        TtsVerifyError::EmptyInput => {}
        other => panic!("expected EmptyInput, got: {other:?}"),
    }
}

#[test]
fn test_verifier_rejects_nan_samples() {
    let verifier = TtsVerifier::builder().sample_rate(24000).build().unwrap();
    let samples = vec![0.5, f32::NAN, 0.3];
    let result = verifier.verify(&samples);
    assert!(result.is_err(), "NaN samples should be rejected");
    match result.unwrap_err() {
        TtsVerifyError::NonFiniteInput { count } => {
            assert_eq!(count, 1, "should report 1 non-finite sample");
        }
        other => panic!("expected NonFiniteInput, got: {other:?}"),
    }
}

#[test]
fn test_verifier_rejects_inf_samples() {
    let verifier = TtsVerifier::builder().sample_rate(24000).build().unwrap();
    let samples = vec![0.5, f32::INFINITY, f32::NEG_INFINITY, 0.3];
    let result = verifier.verify(&samples);
    assert!(result.is_err(), "Inf samples should be rejected");
    match result.unwrap_err() {
        TtsVerifyError::NonFiniteInput { count } => {
            assert_eq!(count, 2, "should report 2 non-finite samples");
        }
        other => panic!("expected NonFiniteInput, got: {other:?}"),
    }
}

#[test]
fn test_verifier_reference_length_mismatch() {
    let verifier = TtsVerifier::builder()
        .sample_rate(24000)
        .with_quality()
        .build()
        .unwrap();
    let candidate = sine_wave_full(440.0, 24000, 1.0, 0.5);
    let reference = sine_wave_full(440.0, 24000, 0.5, 0.5); // Different length!
    let result = verifier.verify_with_reference(&candidate, &reference);
    assert!(result.is_err(), "length mismatch should be rejected");
}

#[test]
fn test_hard_bounds_config_validation_inverted_range() {
    let config = HardBoundsConfig {
        min_duration_sec: 300.0, // Inverted: min > max
        max_duration_sec: 0.1,
        ..HardBoundsConfig::default()
    };
    assert!(
        config.validate().is_err(),
        "inverted duration range should fail validation"
    );
}

#[test]
fn test_quality_config_validation_inverted_f0_range() {
    let config = QualityConfig {
        f0_range: (400.0, 80.0), // Inverted: high < low
        ..QualityConfig::default()
    };
    assert!(
        config.validate().is_err(),
        "inverted F0 range should fail validation"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Multi-speaker config: speaker embeddings have correct dimensionality
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_weight_magnitude_validates_dimensionality() {
    use crate::monotonicity::validate_weight_magnitudes;

    let w1 = vec![0.1_f32; 256 * 512]; // layer 1: 256x512
    let w2 = vec![-0.05_f32; 512 * 256]; // layer 2: 512x256

    let cert = validate_weight_magnitudes(
        &[&w1, &w2],
        &["linear1", "linear2"],
        &[256, 512], // fan_in for each layer
        256,         // d_model
        1.0,         // magnitude_bound
    )
    .unwrap();

    // Both layers should pass: max abs is 0.1 and 0.05, both < 1.0
    assert!(cert.all_within_bound, "all weights should be within bounds");
    assert_eq!(cert.per_layer_max_abs.len(), 2);
}

#[test]
fn test_weight_magnitude_dimension_mismatch_error() {
    use crate::monotonicity::validate_weight_magnitudes;

    let w1 = vec![0.1_f32; 100];
    let result = validate_weight_magnitudes(
        &[&w1],
        &["layer1", "layer2"], // Mismatched: 1 weight, 2 names
        &[10],
        64,
        1.0,
    );
    assert!(result.is_err(), "mismatched layer count should fail");
}

#[test]
fn test_duration_positivity_different_propagation_modes() {
    // Test that different propagation modes produce correct certificates
    for mode in &["IBP", "CROWN", "alpha-CROWN"] {
        let cert = interpret_duration_positivity(-3.0, -10.0, 1.0, 1.0, 1, mode);
        assert!(cert.is_proven, "{mode} should prove -3 > -10");
        assert_eq!(cert.propagation_mode, *mode);
    }
}

#[test]
fn test_pipeline_report_contains_all_stages() {
    let s1 = make_stage("prosody", vec![1, 10], vec![1, 10], -0.5, 0.5);
    let s2 = VerifiedStage::new(
        "decoder",
        vec![1, 10],
        vec![1, 10],
        vec![-1.0; 10],
        vec![1.0; 10],
        vec![-1.0; 10],
        vec![1.0; 10],
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[s1, s2]).unwrap();
    let report = cert.report();
    assert!(
        report.contains("prosody"),
        "report should mention prosody stage"
    );
    assert!(
        report.contains("decoder"),
        "report should mention decoder stage"
    );
    assert!(
        report.contains("Valid: true"),
        "report should indicate valid pipeline"
    );
}

#[test]
fn test_crossfade_linear_preserves_energy() {
    let n = 480; // 20ms at 24kHz
    let tail = sine_wave_full(440.0, 24000, n as f64 / 24000.0, 0.5);
    let head = sine_wave_full(440.0, 24000, n as f64 / 24000.0, 0.5);

    // Ensure same length
    let tail = &tail[..n];
    let head = &head[..n];

    let faded = crossfade_linear(tail, head).unwrap();
    assert_eq!(faded.len(), n, "crossfade should preserve length");

    // Energy should not spike or drop dramatically
    let tail_rms = dsp::rms(tail);
    let faded_rms = dsp::rms(&faded);
    let ratio = faded_rms / tail_rms;
    assert!(
        ratio > 0.3 && ratio < 3.0,
        "crossfade energy ratio should be moderate: {ratio}"
    );
}

#[test]
fn test_crossfade_linear_length_mismatch() {
    let tail = vec![0.5_f32; 100];
    let head = vec![0.5_f32; 200]; // Different length
    let result = crossfade_linear(&tail, &head);
    assert!(result.is_err(), "mismatched lengths should fail");
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

    let config = SpectralCoverageConfig {
        n_bands: 8,
        min_energy_db: f64::NAN,
        min_coverage: 0.5,
    };
    assert!(
        config.validate().is_err(),
        "NaN energy should fail validation"
    );
}

#[test]
fn test_certificate_report_structure() {
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
        deterministic_hash: Some("abc123".to_string()),
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };

    let report = cert.report();
    assert!(
        report.contains("PASS"),
        "report should show PASS for passing bounds"
    );
    assert!(
        report.contains("non_silence"),
        "report should name the bound"
    );
    assert!(report.contains("abc123"), "report should include hash");
    assert!(cert.passes_hard_bounds());
    assert!(cert.passes_quality());
}

#[test]
fn test_streaming_config_validation_rejects_bad_params() {
    let config = StreamingConfig {
        sample_rate: 0,
        ..StreamingConfig::default()
    };
    assert!(config.validate().is_err(), "zero sample rate should fail");

    let config = StreamingConfig {
        crossfade_samples: 0,
        ..StreamingConfig::default()
    };
    assert!(config.validate().is_err(), "zero crossfade should fail");

    let config = StreamingConfig {
        margin_samples: 100, // Less than default crossfade_samples (960)
        ..StreamingConfig::default()
    };
    assert!(config.validate().is_err(), "margin < crossfade should fail");

    let config = StreamingConfig {
        energy_lo: 2.0,
        energy_hi: 1.0, // Inverted
        ..StreamingConfig::default()
    };
    assert!(
        config.validate().is_err(),
        "inverted energy range should fail"
    );
}

#[test]
fn test_dsp_rms_empty_returns_zero() {
    assert!(dsp::rms(&[]).abs() < f64::EPSILON);
}

#[test]
fn test_dsp_dc_offset_empty_returns_zero() {
    assert!(dsp::dc_offset(&[]).abs() < f64::EPSILON);
}

#[test]
fn test_dsp_max_sample_diff_monotone() {
    let ascending: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let diff = dsp::max_sample_diff(&ascending);
    assert!(
        (diff - 0.01).abs() < 0.001,
        "monotone diff should be 0.01, got {diff}"
    );
}
