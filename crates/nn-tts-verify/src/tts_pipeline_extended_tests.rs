// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for nn-tts-verify covering TTS pipeline composition,
//! cost model edge cases, quantization certificates, deterministic hashing,
//! moonshot status tracking, singing verification, Kokoro contract composition,
//! quality bound boundary conditions, streaming boundary verification,
//! and attention monotonicity certificates.

use crate::bounds::HardBound;
use crate::certificate::Certificate;
use crate::config::{CheckOverrides, HardBoundsConfig, QualityConfig};
use crate::cost_model::{
    total_estimated_time_us, total_flops, total_memory_bytes, HardwareCostModel, LayerCostProfile,
};
use crate::deterministic::{pcm_sha256, DeterministicCert, DeterministicMeta};
use crate::dsp;
use crate::kokoro_contracts::{
    all_contracts, bounds_within_contract, contract_stage, max_contract_violation,
    JunctionContract, VerifiedJunctionContract,
};
use crate::monotonicity::{
    from_multi_head_weight_margins, interpret_attention_monotonicity, interpret_duration_positivity,
};
use crate::moonshot::{MoonshotStatus, VerificationLevel};
use crate::pipeline::{check_junction, verify_pipeline, VerifiedStage};
use crate::quality_bound::{
    mcd_lipschitz, snr_lipschitz, verify_quality_bounds, QualityMetricSpec,
};
use crate::quantization_certificate::{build_segment_result, compute_element_drift};
use crate::singing::{hz_to_cents, midi_to_hz, MusicalScore, ScoreNote};
use crate::streaming::{crossfade_linear, verify_streaming, StreamingConfig};
use crate::test_audio_helpers::{sine_wave, sine_wave_full, sine_wave_samples};

// ============================================================================
// 1. Cost model edge cases and roofline model properties
// ============================================================================

#[test]
fn test_cost_model_zero_flops_zero_memory() {
    let model = HardwareCostModel::m4_max();
    let time = model.estimate_time_us(0, 0);
    // Only dispatch overhead remains
    assert!(
        (time - model.dispatch_overhead_us).abs() < f64::EPSILON,
        "zero work should yield only dispatch overhead: expected {}, got {time}",
        model.dispatch_overhead_us
    );
}

#[test]
fn test_cost_model_monotonic_in_flops() {
    let model = HardwareCostModel::m4_max();
    let t1 = model.estimate_time_us(1_000_000, 1_000);
    let t2 = model.estimate_time_us(10_000_000, 1_000);
    let t3 = model.estimate_time_us(100_000_000, 1_000);
    assert!(t2 >= t1, "more FLOPs should take >= time");
    assert!(t3 >= t2, "even more FLOPs should take >= time");
}

#[test]
fn test_cost_model_monotonic_in_memory() {
    let model = HardwareCostModel::m4_max();
    let t1 = model.estimate_time_us(1_000, 1_000_000);
    let t2 = model.estimate_time_us(1_000, 10_000_000);
    let t3 = model.estimate_time_us(1_000, 100_000_000);
    assert!(t2 >= t1, "more memory traffic should take >= time");
    assert!(t3 >= t2, "even more memory traffic should take >= time");
}

#[test]
fn test_cost_model_validate_rejects_infinity() {
    let model = HardwareCostModel {
        peak_tflops_f32: f64::INFINITY,
        peak_bandwidth_gbs: 400.0,
        dispatch_overhead_us: 5.0,
    };
    assert!(model.validate().is_err(), "Inf TFLOPS should be rejected");
}

#[test]
fn test_cost_model_validate_rejects_zero() {
    let model = HardwareCostModel {
        peak_tflops_f32: 0.0,
        peak_bandwidth_gbs: 400.0,
        dispatch_overhead_us: 5.0,
    };
    assert!(model.validate().is_err(), "zero TFLOPS should be rejected");
}

#[test]
fn test_cost_model_conservative_fields_are_derated() {
    let peak = HardwareCostModel::m4_max();
    let cons = HardwareCostModel::m4_max_conservative();
    assert!(
        cons.peak_tflops_f32 < peak.peak_tflops_f32,
        "conservative TFLOPS should be lower"
    );
    assert!(
        cons.peak_bandwidth_gbs < peak.peak_bandwidth_gbs,
        "conservative bandwidth should be lower"
    );
    assert!(
        cons.dispatch_overhead_us > peak.dispatch_overhead_us,
        "conservative overhead should be higher"
    );
}

#[test]
fn test_layer_cost_profile_new_constructor() {
    let profile = LayerCostProfile::new("matmul", 1_000_000, 500_000, 42.5, Some(38.0));
    assert_eq!(profile.layer_name, "matmul");
    assert_eq!(profile.flops, 1_000_000);
    assert_eq!(profile.memory_bytes, 500_000);
    assert!((profile.estimated_time_us - 42.5).abs() < f64::EPSILON);
    assert_eq!(profile.measured_time_us, Some(38.0));
}

#[test]
fn test_total_flops_empty_profiles() {
    let profiles: Vec<LayerCostProfile> = vec![];
    assert_eq!(total_flops(&profiles), 0);
    assert_eq!(total_memory_bytes(&profiles), 0);
    assert!((total_estimated_time_us(&profiles)).abs() < f64::EPSILON);
}

// ============================================================================
// 2. Pipeline composition: edge cases and multi-stage chains
// ============================================================================

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
fn test_pipeline_five_stage_chain() {
    let stages: Vec<VerifiedStage> = (0..5)
        .map(|i| make_stage(&format!("stage_{i}"), vec![1, 64], vec![1, 64], -0.5, 0.5))
        .collect();

    let cert = verify_pipeline(&stages).unwrap();
    assert!(
        cert.is_valid,
        "5-stage chain with compatible bounds should be valid"
    );
    assert_eq!(cert.junctions.len(), 4);
    assert!(cert.is_sound);
}

#[test]
fn test_pipeline_empty_stages_rejected() {
    let result = verify_pipeline(&[]);
    assert!(result.is_err(), "empty stages should be rejected");
}

#[test]
fn test_pipeline_display_format() {
    let s1 = make_stage("enc", vec![1, 10], vec![1, 10], -0.5, 0.5);
    let s2 = make_stage("dec", vec![1, 10], vec![1, 10], -0.5, 0.5);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    let display = format!("{cert}");
    assert!(
        display.contains("2 stages"),
        "display should show stage count"
    );
    assert!(
        display.contains("valid=true"),
        "display should show validity"
    );
}

#[test]
fn test_junction_inf_bounds_violation() {
    let s1 = VerifiedStage::new(
        "inf_stage",
        vec![2],
        vec![2],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        vec![f64::INFINITY, 0.5],
        vec![0.5, 0.5],
        "CROWN",
        true,
    );
    let s2 = make_stage("next", vec![2], vec![2], -1.0, 1.0);
    let junction = check_junction(&s1, &s2, 0);
    assert!(
        !junction.bounds_contained,
        "Infinity in bounds should be a violation"
    );
    assert!(junction.violation_count > 0);
}

#[test]
fn test_junction_length_mismatch_is_violation() {
    let s1 = VerifiedStage::new(
        "short_out",
        vec![2],
        vec![2],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        vec![-0.5], // Only 1 element, but shape says 2
        vec![0.5],
        "CROWN",
        true,
    );
    let s2 = VerifiedStage::new(
        "long_in",
        vec![2],
        vec![2],
        vec![-1.0, -1.0],
        vec![1.0, 1.0],
        vec![-0.5, -0.5],
        vec![0.5, 0.5],
        "CROWN",
        true,
    );
    let junction = check_junction(&s1, &s2, 0);
    // Length mismatch in bounds vectors counts as violation
    assert!(
        junction.violation_count > 0,
        "length mismatch in bounds should be a violation"
    );
}

#[test]
fn test_pipeline_report_contains_junction_info() {
    let s1 = make_stage("encoder", vec![1, 10], vec![1, 10], -0.5, 0.5);
    let s2 = make_stage("decoder", vec![1, 10], vec![1, 10], -0.5, 0.5);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    let report = cert.report();
    assert!(
        report.contains("Junction 0"),
        "report should have junction info"
    );
    assert!(report.contains("encoder"), "report should name from_stage");
    assert!(report.contains("decoder"), "report should name to_stage");
}

// ============================================================================
// 3. Kokoro contracts: verified junction contracts and composition proofs
// ============================================================================

#[test]
fn test_verified_junction_contract_initially_unverified() {
    let contract = JunctionContract::new("J_TEST", "test zone", -1.0, 1.0);
    let verified = VerifiedJunctionContract::new(contract);
    assert!(!verified.bounds_verified);
    assert!(!verified.has_composition_proof());
    assert!(verified.composition_proof_lean4.is_none());
    assert!(verified.composition_theorem_name.is_none());
}

#[test]
fn test_verified_junction_contract_with_proof() {
    let contract = JunctionContract::new("J_TEST", "test zone", -1.0, 1.0);
    let verified = VerifiedJunctionContract::new(contract).with_composition_proof(
        "theorem j_test_contained : ...".to_string(),
        "j_test_contained".to_string(),
    );
    assert!(verified.bounds_verified);
    assert!(verified.has_composition_proof());
    assert_eq!(
        verified.composition_theorem_name.as_deref(),
        Some("j_test_contained")
    );
}

#[test]
fn test_all_six_kokoro_contracts() {
    let contracts = all_contracts();
    assert_eq!(
        contracts.len(),
        6,
        "should have exactly 6 junction contracts"
    );

    let names: Vec<&str> = contracts.iter().map(|c| c.name).collect();
    assert!(names.contains(&"J2_F0"));
    assert!(names.contains(&"J2_ENERGY"));
    assert!(names.contains(&"J3_MAGNITUDE"));
    assert!(names.contains(&"J3B_PHASE"));
    assert!(names.contains(&"J4_BF16"));
    assert!(names.contains(&"J5_AUDIO"));
}

#[test]
fn test_j3_magnitude_contract_allows_exp_range() {
    let contracts = all_contracts();
    let j3 = contracts.iter().find(|c| c.name == "J3_MAGNITUDE").unwrap();
    // exp(80) ~ 5.5e34 is within f32 range
    assert!(j3.upper <= 80.0, "J3 upper should be <= 80 for exp safety");
    assert!(j3.lower >= -80.0, "J3 lower should be >= -80");
}

#[test]
fn test_j4_bf16_contract_within_safe_range() {
    let contracts = all_contracts();
    let j4 = contracts.iter().find(|c| c.name == "J4_BF16").unwrap();
    // BF16 safe range: |x| < 128 for ULP < 1.0
    assert!(
        j4.upper.abs() <= 128.0,
        "J4 upper should be within BF16 safe range"
    );
    assert!(
        j4.lower.abs() <= 128.0,
        "J4 lower should be within BF16 safe range"
    );
}

#[test]
fn test_bounds_within_contract_empty_slices() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    assert!(
        bounds_within_contract(&contract, &[], &[]),
        "empty bounds should be vacuously within contract"
    );
}

#[test]
fn test_bounds_within_contract_mismatched_lengths_fails() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    assert!(
        !bounds_within_contract(&contract, &[0.0], &[0.0, 0.5]),
        "mismatched lower/upper lengths should fail"
    );
}

#[test]
fn test_max_contract_violation_nan_returns_max() {
    let contract = JunctionContract::new("test", "zone", -1.0, 1.0);
    let violation = max_contract_violation(&contract, &[f64::NAN], &[0.5]);
    assert_eq!(violation, f64::MAX, "NaN should produce MAX violation");
}

#[test]
fn test_contract_stage_pipelines_compose() {
    let contracts = all_contracts();
    let j2_f0 = &contracts[0]; // J2_F0
    let j5_audio = &contracts[5]; // J5_AUDIO

    let stage = contract_stage(
        "decoder",
        &[1, 10],
        &[1, 24000],
        j2_f0,
        j5_audio,
        "CROWN",
        true,
    );

    assert_eq!(stage.name, "decoder");
    assert_eq!(stage.input_shape, vec![1, 10]);
    assert_eq!(stage.output_shape, vec![1, 24000]);
    assert!(stage.is_sound);
}

// ============================================================================
// 4. Quantization certificate: drift computation and certificate building
// ============================================================================

#[test]
fn test_compute_element_drift_identical_bounds() {
    let lo = vec![0.0_f32; 10];
    let hi = vec![1.0_f32; 10];
    let (max_drift, mean_drift, n) = compute_element_drift(&lo, &hi, &lo, &hi).unwrap();
    assert!(
        max_drift.abs() < f64::EPSILON,
        "identical bounds should have zero drift"
    );
    assert!(mean_drift.abs() < f64::EPSILON);
    assert_eq!(n, 10);
}

#[test]
fn test_compute_element_drift_uniform_shift() {
    let f32_lo = vec![0.0_f32; 5];
    let f32_hi = vec![1.0_f32; 5];
    let q_lo = vec![0.1_f32; 5]; // shifted by 0.1
    let q_hi = vec![1.1_f32; 5]; // shifted by 0.1
    let (max_drift, mean_drift, _) = compute_element_drift(&f32_lo, &f32_hi, &q_lo, &q_hi).unwrap();
    assert!(
        (max_drift - 0.1).abs() < 1e-6,
        "uniform 0.1 shift should yield 0.1 drift, got {max_drift}"
    );
    assert!((mean_drift - 0.1).abs() < 1e-6);
}

#[test]
fn test_compute_element_drift_mismatched_lengths() {
    let result = compute_element_drift(&[0.0_f32; 3], &[1.0; 3], &[0.0; 5], &[1.0; 5]);
    assert!(result.is_err(), "mismatched lengths should fail");
}

#[test]
fn test_compute_element_drift_empty_arrays() {
    let result = compute_element_drift(&[], &[], &[], &[]);
    assert!(result.is_err(), "empty arrays should fail");
}

#[test]
fn test_compute_element_drift_non_finite_rejected() {
    let result = compute_element_drift(&[f32::NAN], &[1.0], &[0.0], &[1.0]);
    assert!(result.is_err(), "NaN in bounds should fail");
}

#[test]
fn test_build_segment_result_basic() {
    let f32_lo = vec![-0.5_f32; 4];
    let f32_hi = vec![0.5_f32; 4];
    let q_lo = vec![-0.52_f32; 4];
    let q_hi = vec![0.52_f32; 4];

    let result = build_segment_result("test_segment", &f32_lo, &f32_hi, &q_lo, &q_hi).unwrap();

    assert_eq!(result.segment_name, "test_segment");
    assert_eq!(result.num_elements, 4);
    assert!(result.max_element_drift > 0.0);
    assert!(result.mean_element_drift > 0.0);
    assert!(result.f32_output_width > 0.0);
    assert!(result.quantized_output_width > 0.0);
}

// ============================================================================
// 5. Deterministic hashing: reproducibility and regression detection
// ============================================================================

#[test]
fn test_pcm_sha256_deterministic_across_calls() {
    let audio = sine_wave(440.0, 24000, 0.1);
    let h1 = pcm_sha256(&audio);
    let h2 = pcm_sha256(&audio);
    assert_eq!(h1, h2, "identical audio should produce identical hashes");
}

#[test]
fn test_pcm_sha256_different_for_different_audio() {
    let a = sine_wave(440.0, 24000, 0.1);
    let b = sine_wave(880.0, 24000, 0.1);
    assert_ne!(
        pcm_sha256(&a),
        pcm_sha256(&b),
        "different audio should produce different hashes"
    );
}

#[test]
fn test_deterministic_cert_from_audio() {
    let audio = sine_wave_full(440.0, 24000, 0.5, 0.5);
    let meta = DeterministicMeta {
        input_text: Some("test".to_string()),
        voice_id: Some("spk1".to_string()),
        seed: Some(42),
    };
    let cert = DeterministicCert::from_audio(&audio, meta);
    assert!(cert.verify(&audio), "cert should verify against same audio");
    assert!(
        !cert.verify(&sine_wave(880.0, 24000, 0.5)),
        "cert should fail against different audio"
    );
}

#[test]
fn test_deterministic_meta_default() {
    let meta = DeterministicMeta::default();
    assert!(meta.input_text.is_none());
    assert!(meta.voice_id.is_none());
    assert!(meta.seed.is_none());
}

// ============================================================================
// 6. Moonshot status tracking and verification levels
// ============================================================================

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
    assert_eq!(
        format!("{}", VerificationLevel::CrownProven),
        "CROWN_PROVEN"
    );
    assert_eq!(format!("{}", VerificationLevel::SmtProven), "SMT_PROVEN");
}

#[test]
fn test_moonshot_status_from_repo_has_8_properties() {
    let status = MoonshotStatus::from_repo();
    assert_eq!(
        status.properties.len(),
        8,
        "moonshot should track exactly 8 properties"
    );
}

#[test]
fn test_moonshot_status_report_contains_all_properties() {
    let status = MoonshotStatus::from_repo();
    let report = status.report();
    assert!(report.contains("Non-silent"), "report should mention P1");
    assert!(report.contains("Non-clipping"), "report should mention P2");
    assert!(report.contains("Intelligible"), "report should mention P3");
    assert!(
        report.contains("Speaker-consistent"),
        "report should mention P4"
    );
    assert!(
        report.contains("Temporally bounded"),
        "report should mention P5"
    );
    assert!(
        report.contains("Streaming-safe"),
        "report should mention P6"
    );
    assert!(report.contains("Memory-safe"), "report should mention P7");
    assert!(
        report.contains("Correct implementation"),
        "report should mention P8"
    );
}

#[test]
fn test_moonshot_status_display_format() {
    let status = MoonshotStatus::from_repo();
    let display = format!("{status}");
    assert!(
        display.contains("P1:"),
        "display should show property indices"
    );
    assert!(display.contains("P8:"), "display should show all 8");
}

#[test]
fn test_moonshot_level_counts_sum_to_eight() {
    let status = MoonshotStatus::from_repo();
    let counts = status.level_counts();
    let total: usize = counts.iter().map(|(_, c)| c).sum();
    assert_eq!(total, 8, "level counts should sum to 8");
}

// ============================================================================
// 7. Attention monotonicity certificates
// ============================================================================

#[test]
fn test_attention_monotonicity_diagonal_dominant() {
    // 3x3 score matrix where diagonal is clearly dominant
    // Row 0: diag=5.0, off=1.0,1.0
    // Row 1: diag=5.0, off=1.0,1.0
    // Row 2: diag=5.0, off=1.0,1.0
    let lower = vec![5.0_f32, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 5.0];
    let upper = vec![5.0_f32, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 5.0];
    let cert = interpret_attention_monotonicity(&lower, &upper, 3, 3, 1.0, "CROWN").unwrap();
    assert!(
        cert.is_proven,
        "diagonal dominant matrix should be proven monotonic"
    );
    assert!(cert.min_margin > 0.0);
    assert_eq!(cert.row_margins.len(), 3);
}

#[test]
fn test_attention_monotonicity_not_dominant() {
    // 2x2 where off-diagonal upper exceeds diagonal lower
    let lower = vec![1.0_f32, 2.0, 2.0, 1.0];
    let upper = vec![1.5_f32, 3.0, 3.0, 1.5];
    let cert = interpret_attention_monotonicity(&lower, &upper, 2, 2, 1.0, "IBP").unwrap();
    assert!(
        !cert.is_proven,
        "off-diagonal dominant matrix should not be proven"
    );
}

#[test]
fn test_attention_monotonicity_dimension_mismatch() {
    let lower = vec![1.0_f32; 4]; // 2x2
    let upper = vec![1.0_f32; 6]; // wrong size
    let result = interpret_attention_monotonicity(&lower, &upper, 2, 2, 1.0, "CROWN");
    assert!(result.is_err(), "dimension mismatch should fail");
}

#[test]
fn test_attention_monotonicity_nan_rejected() {
    let lower = vec![1.0_f32, f32::NAN, 1.0, 1.0];
    let upper = vec![2.0_f32, 2.0, 2.0, 2.0];
    let result = interpret_attention_monotonicity(&lower, &upper, 2, 2, 1.0, "CROWN");
    assert!(result.is_err(), "NaN in scores should fail");
}

#[test]
fn test_attention_monotonicity_single_column_trivial() {
    // Single encoder position: trivially monotonic
    let lower = vec![1.0_f32, 2.0, 3.0];
    let upper = vec![1.5_f32, 2.5, 3.5];
    let cert = interpret_attention_monotonicity(&lower, &upper, 3, 1, 1.0, "CROWN").unwrap();
    assert!(
        cert.is_proven,
        "single encoder position should be trivially monotonic"
    );
}

#[test]
fn test_multi_head_weight_margins_all_positive() {
    let head_margins = vec![vec![0.5, 0.3, 0.8], vec![0.2, 0.6, 0.4]];
    let cert = from_multi_head_weight_margins(&head_margins, 3, 3, 1.0, "CROWN").unwrap();
    assert!(
        cert.is_proven,
        "all-positive margins should prove monotonicity"
    );
    // Per-step minimum across heads: [0.2, 0.3, 0.4]
    assert!((cert.row_margins[0] - 0.2).abs() < f64::EPSILON);
    assert!((cert.row_margins[1] - 0.3).abs() < f64::EPSILON);
    assert!((cert.row_margins[2] - 0.4).abs() < f64::EPSILON);
}

#[test]
fn test_multi_head_weight_margins_one_negative() {
    let head_margins = vec![
        vec![0.5, -0.1, 0.8], // head 0 fails at step 1
        vec![0.2, 0.6, 0.4],
    ];
    let cert = from_multi_head_weight_margins(&head_margins, 3, 3, 1.0, "CROWN").unwrap();
    assert!(
        !cert.is_proven,
        "negative margin in any head should prevent proof"
    );
}

#[test]
fn test_multi_head_weight_margins_short_vector_rejected() {
    let head_margins = vec![
        vec![0.5, 0.3], // only 2 elements for 3-step problem
        vec![0.2, 0.6, 0.4],
    ];
    let result = from_multi_head_weight_margins(&head_margins, 3, 3, 1.0, "CROWN");
    assert!(result.is_err(), "short margin vector should fail");
}

// ============================================================================
// 8. Duration positivity certificates: edge cases
// ============================================================================

#[test]
fn test_duration_positivity_at_exact_threshold() {
    // lower_bound == threshold: NOT proven (need strict >)
    let cert = interpret_duration_positivity(-10.0, -10.0, 1.0, 1.0, 1, "CROWN");
    assert!(
        !cert.is_proven,
        "exact equality should not be proven (need strict >)"
    );
}

#[test]
fn test_duration_positivity_long_sequence() {
    let cert = interpret_duration_positivity(-2.0, -10.0, 1.0, 1.0, 100, "alpha-CROWN");
    assert!(cert.is_proven);
    assert_eq!(cert.sequence_length, 100);
    assert_eq!(cert.propagation_mode, "alpha-CROWN");
}

#[test]
fn test_duration_positivity_large_input_bound() {
    let cert = interpret_duration_positivity(-5.0, -10.0, 10.0, 5.0, 4, "CROWN");
    assert!(cert.is_proven);
    assert!((cert.input_bound - 10.0).abs() < f64::EPSILON);
    assert!((cert.style_bound - 5.0).abs() < f64::EPSILON);
}

// ============================================================================
// 9. Quality bound verification: boundary and stress conditions
// ============================================================================

#[test]
fn test_quality_bound_zero_perturbation() {
    let specs = vec![QualityMetricSpec {
        name: "SNR".into(),
        lipschitz_constant: 100.0,
        baseline_value: 11.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    let cert = verify_quality_bounds(0.0, &specs).unwrap();
    assert!(
        cert.all_guaranteed,
        "zero perturbation should guarantee all metrics"
    );
    assert!((cert.tightest_margin - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_quality_bound_negative_perturbation_rejected() {
    let specs = vec![QualityMetricSpec {
        name: "SNR".into(),
        lipschitz_constant: 1.0,
        baseline_value: 20.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    let result = verify_quality_bounds(-1.0, &specs);
    assert!(result.is_err(), "negative perturbation should be rejected");
}

#[test]
fn test_quality_bound_empty_metrics_rejected() {
    let result = verify_quality_bounds(0.1, &[]);
    assert!(result.is_err(), "empty metrics should be rejected");
}

#[test]
fn test_quality_bound_nan_lipschitz_rejected() {
    let specs = vec![QualityMetricSpec {
        name: "bad".into(),
        lipschitz_constant: f64::NAN,
        baseline_value: 20.0,
        threshold: 10.0,
        higher_is_better: true,
        citation: "test",
    }];
    let result = verify_quality_bounds(0.1, &specs);
    assert!(result.is_err(), "NaN Lipschitz should be rejected");
}

#[test]
fn test_quality_bound_lower_is_better_metric() {
    let specs = vec![QualityMetricSpec {
        name: "MCD".into(),
        lipschitz_constant: 1.0,
        baseline_value: 3.0,
        threshold: 6.0,
        higher_is_better: false,
        citation: "test",
    }];
    // Perturbation 0.5: worst_case = 3.0 + 1.0*0.5 = 3.5 < 6.0 (guaranteed)
    let cert = verify_quality_bounds(0.5, &specs).unwrap();
    assert!(cert.all_guaranteed);
    // margin = threshold - worst_case = 6.0 - 3.5 = 2.5
    assert!((cert.tightest_margin - 2.5).abs() < f64::EPSILON);
}

#[test]
fn test_snr_lipschitz_zero_signal_rejected() {
    let result = snr_lipschitz(0.0, 25.0);
    assert!(result.is_err(), "zero signal RMS should be rejected");
}

#[test]
fn test_snr_lipschitz_nan_snr_rejected() {
    let result = snr_lipschitz(0.5, f64::NAN);
    assert!(result.is_err(), "NaN SNR should be rejected");
}

#[test]
fn test_mcd_lipschitz_zero_frames_rejected() {
    let result = mcd_lipschitz(0);
    assert!(result.is_err(), "zero frames should be rejected");
}

// ============================================================================
// 10. Streaming boundary verification
// ============================================================================

#[test]
fn test_streaming_two_identical_chunks() {
    let config = StreamingConfig::default();
    let chunk = sine_wave_full(440.0, 24000, 0.1, 0.5); // 100ms, 2400 samples
    let chunks: Vec<&[f32]> = vec![&chunk, &chunk];
    let cert = verify_streaming(&chunks, &config).unwrap();
    assert_eq!(cert.n_chunks, 2);
    assert_eq!(cert.boundaries.len(), 1);
}

#[test]
fn test_streaming_three_chunks() {
    let config = StreamingConfig::default();
    let chunk = sine_wave_full(440.0, 24000, 0.1, 0.5);
    let chunks: Vec<&[f32]> = vec![&chunk, &chunk, &chunk];
    let cert = verify_streaming(&chunks, &config).unwrap();
    assert_eq!(cert.n_chunks, 3);
    assert_eq!(cert.boundaries.len(), 2);
}

#[test]
fn test_streaming_single_chunk_rejected() {
    let config = StreamingConfig::default();
    let chunk = sine_wave_full(440.0, 24000, 0.1, 0.5);
    let chunks: Vec<&[f32]> = vec![&chunk];
    let result = verify_streaming(&chunks, &config);
    assert!(result.is_err(), "single chunk should be rejected");
}

#[test]
fn test_crossfade_linear_empty_slices() {
    let result = crossfade_linear(&[], &[]);
    assert!(result.is_ok(), "empty slices should succeed");
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_crossfade_linear_single_sample() {
    let result = crossfade_linear(&[0.5], &[1.0]);
    assert!(result.is_ok());
    let faded = result.unwrap();
    assert_eq!(faded.len(), 1);
    // Single sample returns head value
    assert!((faded[0] - 1.0).abs() < f64::EPSILON as f32);
}

#[test]
fn test_crossfade_linear_midpoint() {
    let tail = vec![1.0_f32; 3];
    let head = vec![0.0_f32; 3];
    let faded = crossfade_linear(&tail, &head).unwrap();
    assert_eq!(faded.len(), 3);
    // At midpoint (index 1), alpha = 0.5, value = 1.0*0.5 + 0.0*0.5 = 0.5
    assert!(
        (faded[1] - 0.5).abs() < 0.01,
        "midpoint should be ~0.5, got {}",
        faded[1]
    );
}

// ============================================================================
// 11. Singing verification: pitch conversion utilities
// ============================================================================

#[test]
fn test_midi_to_hz_a4() {
    let hz = midi_to_hz(69);
    assert!(
        (hz - 440.0).abs() < 0.01,
        "MIDI 69 (A4) should be 440 Hz, got {hz}"
    );
}

#[test]
fn test_midi_to_hz_c4() {
    let hz = midi_to_hz(60);
    assert!(
        (hz - 261.63).abs() < 0.1,
        "MIDI 60 (C4) should be ~261.63 Hz, got {hz}"
    );
}

#[test]
fn test_midi_to_hz_octave_doubles_frequency() {
    let hz_a3 = midi_to_hz(57);
    let hz_a4 = midi_to_hz(69);
    assert!(
        (hz_a4 / hz_a3 - 2.0).abs() < 0.01,
        "octave should double frequency"
    );
}

#[test]
fn test_hz_to_cents_unison() {
    let cents = hz_to_cents(440.0, 440.0);
    assert!(
        cents.abs() < 0.01,
        "same frequency should give 0 cents, got {cents}"
    );
}

#[test]
fn test_hz_to_cents_octave() {
    let cents = hz_to_cents(880.0, 440.0);
    assert!(
        (cents - 1200.0).abs() < 0.1,
        "octave should be 1200 cents, got {cents}"
    );
}

#[test]
fn test_hz_to_cents_semitone() {
    let hz_a4 = 440.0;
    let hz_bb4 = hz_a4 * 2.0_f64.powf(1.0 / 12.0); // A#4/Bb4
    let cents = hz_to_cents(hz_bb4, hz_a4);
    assert!(
        (cents - 100.0).abs() < 0.1,
        "semitone should be 100 cents, got {cents}"
    );
}

#[test]
fn test_musical_score_validation_empty() {
    let score = MusicalScore {
        notes: vec![],
        tempo_bpm: 120.0,
    };
    assert!(
        score.validate().is_err(),
        "empty score should fail validation"
    );
}

#[test]
fn test_musical_score_validation_zero_tempo() {
    let score = MusicalScore {
        notes: vec![ScoreNote {
            midi_note: 69,
            onset_sec: 0.0,
            duration_sec: 1.0,
            is_rest: false,
        }],
        tempo_bpm: 0.0,
    };
    assert!(
        score.validate().is_err(),
        "zero tempo should fail validation"
    );
}

#[test]
fn test_musical_score_validation_valid() {
    let score = MusicalScore {
        notes: vec![
            ScoreNote {
                midi_note: 60,
                onset_sec: 0.0,
                duration_sec: 0.5,
                is_rest: false,
            },
            ScoreNote {
                midi_note: 64,
                onset_sec: 0.5,
                duration_sec: 0.5,
                is_rest: false,
            },
        ],
        tempo_bpm: 120.0,
    };
    assert!(
        score.validate().is_ok(),
        "valid score should pass validation"
    );
}

// ============================================================================
// 12. Config validation: CheckOverrides and cross-field constraints
// ============================================================================

#[test]
fn test_check_overrides_empty_valid() {
    let overrides = CheckOverrides::new();
    assert!(
        overrides.validate().is_ok(),
        "empty overrides should be valid"
    );
}

#[test]
fn test_check_overrides_nan_min_rms_rejected() {
    let overrides = CheckOverrides {
        min_rms: Some(f64::NAN),
        ..CheckOverrides::default()
    };
    assert!(overrides.validate().is_err(), "NaN min_rms should fail");
}

#[test]
fn test_check_overrides_inverted_duration_rejected() {
    let overrides = CheckOverrides {
        min_duration_sec: Some(100.0),
        max_duration_sec: Some(1.0), // Inverted: min > max
        ..CheckOverrides::default()
    };
    assert!(
        overrides.validate().is_err(),
        "inverted duration override should fail"
    );
}

#[test]
fn test_hard_bounds_config_effective_uses_override() {
    let config = HardBoundsConfig {
        min_rms: 0.01,
        overrides: CheckOverrides {
            min_rms: Some(0.05),
            ..CheckOverrides::default()
        },
        ..HardBoundsConfig::default()
    };
    assert!(
        (config.effective_min_rms() - 0.05).abs() < f64::EPSILON,
        "effective_min_rms should use override"
    );
}

#[test]
fn test_hard_bounds_config_effective_uses_default_when_no_override() {
    let config = HardBoundsConfig::default();
    assert!(
        (config.effective_min_rms() - 0.01).abs() < f64::EPSILON,
        "effective_min_rms should use default when no override"
    );
}

#[test]
fn test_quality_config_defaults() {
    let config = QualityConfig::default();
    assert!((config.max_mcd_db - 6.0).abs() < f64::EPSILON);
    assert!((config.min_hnr_db - 15.0).abs() < f64::EPSILON);
    assert!((config.f0_range.0 - 80.0).abs() < f64::EPSILON);
    assert!((config.f0_range.1 - 400.0).abs() < f64::EPSILON);
    assert!(config.multi_res_stft.is_none());
}

// ============================================================================
// 13. Certificate: report generation and method helpers
// ============================================================================

#[test]
fn test_certificate_passes_hard_bounds_all_pass() {
    let cert = Certificate {
        hard_bounds: vec![
            HardBound {
                name: "test_a",
                passed: true,
                value: 0.5,
                threshold: 0.01,
            },
            HardBound {
                name: "test_b",
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
    assert!(!cert.has_crown_evidence());
    assert!(!cert.has_junction_summary());
    assert!(cert.passes_junction_contracts()); // vacuously true
}

#[test]
fn test_certificate_report_shows_fail() {
    let cert = Certificate {
        hard_bounds: vec![HardBound {
            name: "no_clipping",
            passed: false,
            value: 1.5,
            threshold: 1.0,
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
    let report = cert.report();
    assert!(report.contains("FAIL"), "report should show FAIL");
    assert!(
        report.contains("no_clipping"),
        "report should name the check"
    );
    assert!(report.contains("FAILED"), "overall should say FAILED");
}

// ============================================================================
// 14. DSP utilities: edge cases
// ============================================================================

#[test]
fn test_dsp_rms_single_sample() {
    assert!((dsp::rms(&[0.5]) - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_dsp_dc_offset_symmetric_signal() {
    let signal = sine_wave(440.0, 24000, 1.0);
    let offset = dsp::dc_offset(&signal);
    assert!(
        offset.abs() < 0.01,
        "symmetric sine wave should have near-zero DC offset, got {offset}"
    );
}

#[test]
fn test_dsp_max_sample_diff_single_sample() {
    assert!(dsp::max_sample_diff(&[1.0]).abs() < f64::EPSILON);
}

#[test]
fn test_dsp_max_sample_diff_constant_signal() {
    let constant = vec![0.5_f32; 100];
    assert!(
        dsp::max_sample_diff(&constant).abs() < f64::EPSILON,
        "constant signal should have zero diff"
    );
}

// ============================================================================
// 15. Test audio helpers: sine wave generation
// ============================================================================

#[test]
fn test_sine_wave_correct_length() {
    let wave = sine_wave(440.0, 24000, 1.0);
    assert_eq!(wave.len(), 24000, "1s at 24kHz should be 24000 samples");
}

#[test]
fn test_sine_wave_full_amplitude_scaling() {
    let wave = sine_wave_full(440.0, 24000, 0.5, 0.3);
    let max_abs = wave.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    assert!(
        (max_abs - 0.3).abs() < 0.01,
        "amplitude should be ~0.3, got {max_abs}"
    );
}

#[test]
fn test_sine_wave_samples_exact_count() {
    let wave = sine_wave_samples(440.0, 24000, 1000);
    assert_eq!(wave.len(), 1000, "should produce exactly 1000 samples");
}

#[test]
fn test_sine_wave_values_in_range() {
    let wave = sine_wave(440.0, 24000, 1.0);
    for (i, &v) in wave.iter().enumerate() {
        assert!(
            (-1.001..=1.001).contains(&v),
            "sample {i} = {v} out of range [-1, 1]"
        );
    }
}
