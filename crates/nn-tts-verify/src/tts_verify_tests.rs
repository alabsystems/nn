// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for nn-tts-verify covering cost models, junction
//! contracts, moonshot properties (P1-P8), pipeline composition, and
//! certificate generation.

use crate::cost_model::{
    total_estimated_time_us, total_flops, total_memory_bytes, HardwareCostModel, LayerCostProfile,
};
use crate::cost_propagation::{CoupledLayerResult, CoupledTimingCertificate};
use crate::kokoro_contracts::{
    all_contracts, bounds_within_contract, contract_stage, max_contract_violation,
    JunctionContract, J2_ENERGY_LOWER, J2_ENERGY_UPPER, J2_F0_LOWER, J2_F0_UPPER, J3B_PHASE_LOWER,
    J3B_PHASE_UPPER, J3_MAGNITUDE_LOWER, J3_MAGNITUDE_UPPER, J4_BF16_LOWER, J4_BF16_UPPER,
    J5_AUDIO_LOWER, J5_AUDIO_UPPER,
};
use crate::moonshot::{
    MoonshotCertificate, MoonshotStatus, VerificationLevel, CERTIFICATE_SCHEMA_VERSION,
};
use crate::moonshot_crown::{
    check_non_clipping, check_non_silence, check_streaming_safety, check_temporal_boundedness,
    verify_properties_from_pipeline,
};
use crate::pipeline::{
    check_junction, verify_pipeline, PipelineCertificate, TimingCertificate, VerifiedStage,
};

// ============================================================================
// Cost model tests
// ============================================================================

#[test]
fn test_hardware_cost_model_m4_max_fields() {
    let m = HardwareCostModel::m4_max();
    assert!((m.peak_tflops_f32 - 14.2).abs() < 1e-10);
    assert!((m.peak_bandwidth_gbs - 400.0).abs() < 1e-10);
    assert!((m.dispatch_overhead_us - 5.0).abs() < 1e-10);
}

#[test]
fn test_hardware_cost_model_conservative_has_lower_throughput() {
    let standard = HardwareCostModel::m4_max();
    let conservative = HardwareCostModel::m4_max_conservative();
    // Conservative model has reduced throughput (higher time estimates).
    assert!(conservative.peak_tflops_f32 < standard.peak_tflops_f32);
    assert!(conservative.peak_bandwidth_gbs < standard.peak_bandwidth_gbs);
    assert!(conservative.dispatch_overhead_us > standard.dispatch_overhead_us);
}

#[test]
fn test_hardware_cost_model_validate_valid() {
    let m = HardwareCostModel::m4_max();
    assert!(m.validate().is_ok());
}

#[test]
fn test_hardware_cost_model_validate_rejects_zero() {
    let m = HardwareCostModel {
        peak_tflops_f32: 0.0,
        peak_bandwidth_gbs: 400.0,
        dispatch_overhead_us: 5.0,
    };
    assert!(m.validate().is_err());
}

#[test]
fn test_hardware_cost_model_validate_rejects_nan() {
    let m = HardwareCostModel {
        peak_tflops_f32: f64::NAN,
        peak_bandwidth_gbs: 400.0,
        dispatch_overhead_us: 5.0,
    };
    assert!(m.validate().is_err());
}

#[test]
fn test_hardware_cost_model_validate_rejects_negative() {
    let m = HardwareCostModel {
        peak_tflops_f32: 14.2,
        peak_bandwidth_gbs: -1.0,
        dispatch_overhead_us: 5.0,
    };
    assert!(m.validate().is_err());
}

#[test]
fn test_hardware_cost_model_validate_rejects_infinity() {
    let m = HardwareCostModel {
        peak_tflops_f32: 14.2,
        peak_bandwidth_gbs: 400.0,
        dispatch_overhead_us: f64::INFINITY,
    };
    assert!(m.validate().is_err());
}

#[test]
fn test_estimate_time_us_compute_bound() {
    let m = HardwareCostModel::m4_max();
    // 14.2 TFLOPS = 14.2e6 FLOPS/us
    // 1e9 FLOPs / 14.2e6 = ~70.4 us compute time
    // 0 bytes → 0 memory time
    // Total = 70.4 + 5.0 = 75.4 us
    let time = m.estimate_time_us(1_000_000_000, 0);
    let expected_compute = 1e9 / (14.2 * 1e6);
    assert!((time - (expected_compute + 5.0)).abs() < 0.1);
}

#[test]
fn test_estimate_time_us_memory_bound() {
    let m = HardwareCostModel::m4_max();
    // 400 GB/s = 400e3 bytes/us
    // 1e9 bytes / 400e3 = 2500 us memory time
    // 0 FLOPs → 0 compute time
    // Total = 2500 + 5.0 = 2505 us
    let time = m.estimate_time_us(0, 1_000_000_000);
    let expected_memory = 1e9 / (400.0 * 1e3);
    assert!((time - (expected_memory + 5.0)).abs() < 0.1);
}

#[test]
fn test_estimate_time_us_roofline_takes_max() {
    let m = HardwareCostModel::m4_max();
    // Both compute and memory non-zero: roofline takes max.
    let time = m.estimate_time_us(1_000_000_000, 1_000_000_000);
    let compute = 1e9 / (14.2 * 1e6);
    let memory = 1e9 / (400.0 * 1e3);
    let expected = f64::max(compute, memory) + 5.0;
    assert!((time - expected).abs() < 0.1);
}

#[test]
fn test_conservative_gives_higher_time_estimate() {
    let standard = HardwareCostModel::m4_max();
    let conservative = HardwareCostModel::m4_max_conservative();
    let t_standard = standard.estimate_time_us(10_000_000, 5_000_000);
    let t_conservative = conservative.estimate_time_us(10_000_000, 5_000_000);
    assert!(
        t_conservative > t_standard,
        "conservative ({t_conservative}) should exceed standard ({t_standard})"
    );
}

#[test]
fn test_layer_cost_profile_new() {
    let p = LayerCostProfile::new("test_layer", 1000, 2000, 3.0, Some(2.5));
    assert_eq!(p.layer_name, "test_layer");
    assert_eq!(p.flops, 1000);
    assert_eq!(p.memory_bytes, 2000);
    assert!((p.estimated_time_us - 3.0).abs() < 1e-10);
    assert_eq!(p.measured_time_us, Some(2.5));
}

#[test]
fn test_total_estimated_time_us_sums_profiles() {
    let profiles = vec![
        LayerCostProfile::new("a", 0, 0, 10.0, None),
        LayerCostProfile::new("b", 0, 0, 20.0, None),
        LayerCostProfile::new("c", 0, 0, 30.0, None),
    ];
    assert!((total_estimated_time_us(&profiles) - 60.0).abs() < 1e-10);
}

#[test]
fn test_total_flops_sums_profiles() {
    let profiles = vec![
        LayerCostProfile::new("a", 100, 0, 0.0, None),
        LayerCostProfile::new("b", 200, 0, 0.0, None),
    ];
    assert_eq!(total_flops(&profiles), 300);
}

#[test]
fn test_total_memory_bytes_sums_profiles() {
    let profiles = vec![
        LayerCostProfile::new("a", 0, 1000, 0.0, None),
        LayerCostProfile::new("b", 0, 2000, 0.0, None),
    ];
    assert_eq!(total_memory_bytes(&profiles), 3000);
}

#[test]
fn test_empty_profiles_return_zero() {
    let profiles: Vec<LayerCostProfile> = vec![];
    assert!((total_estimated_time_us(&profiles)).abs() < 1e-10);
    assert_eq!(total_flops(&profiles), 0);
    assert_eq!(total_memory_bytes(&profiles), 0);
}

// ============================================================================
// Junction contract tests
// ============================================================================

#[test]
fn test_junction_contract_new() {
    let c = JunctionContract::new("test", "zone A -> B", -10.0, 10.0);
    assert_eq!(c.name, "test");
    assert_eq!(c.zone, "zone A -> B");
    assert!((c.lower - (-10.0)).abs() < 1e-10);
    assert!((c.upper - 10.0).abs() < 1e-10);
}

#[test]
fn test_all_contracts_symmetric_pairs() {
    let contracts = all_contracts();
    // J2_F0 and J2_ENERGY share the same zone.
    assert_eq!(contracts[0].zone, contracts[1].zone);
    // J3 and J3B share the generator zone.
    assert_eq!(contracts[2].zone, contracts[3].zone);
}

#[test]
fn test_j4_bf16_symmetric() {
    // BF16 bounds are symmetric around zero.
    assert!((J4_BF16_LOWER + J4_BF16_UPPER).abs() < 1e-10);
}

#[test]
fn test_j5_audio_symmetric() {
    assert!((J5_AUDIO_LOWER + J5_AUDIO_UPPER).abs() < 1e-10);
}

#[test]
fn test_j2_energy_symmetric() {
    assert!((J2_ENERGY_LOWER + J2_ENERGY_UPPER).abs() < 1e-10);
}

#[test]
fn test_j3_magnitude_symmetric() {
    assert!((J3_MAGNITUDE_LOWER + J3_MAGNITUDE_UPPER).abs() < 1e-10);
}

#[test]
fn test_j3b_phase_symmetric() {
    assert!((J3B_PHASE_LOWER + J3B_PHASE_UPPER).abs() < 1e-10);
}

#[test]
fn test_bounds_within_contract_empty_vectors() {
    let c = JunctionContract::new("test", "zone", -1.0, 1.0);
    // Empty slices: trivially contained (no elements to violate).
    assert!(bounds_within_contract(&c, &[], &[]));
}

#[test]
fn test_bounds_within_contract_infinity_rejected() {
    let c = JunctionContract::new("test", "zone", -1.0, 1.0);
    assert!(!bounds_within_contract(&c, &[f64::NEG_INFINITY], &[0.5]));
    assert!(!bounds_within_contract(&c, &[-0.5], &[f64::INFINITY]));
}

#[test]
fn test_max_contract_violation_empty_vectors() {
    let c = JunctionContract::new("test", "zone", -1.0, 1.0);
    let v = max_contract_violation(&c, &[], &[]);
    assert!((v - 0.0).abs() < 1e-10);
}

#[test]
fn test_max_contract_violation_both_sides_breached() {
    let c = JunctionContract::new("test", "zone", -1.0, 1.0);
    // Lower breach: -2.0 is 1.0 below -1.0. Upper breach: 3.0 is 2.0 above 1.0.
    let v = max_contract_violation(&c, &[-2.0], &[3.0]);
    assert!((v - 2.0).abs() < 1e-10);
}

#[test]
fn test_max_contract_violation_infinity_rejected() {
    let c = JunctionContract::new("test", "zone", -1.0, 1.0);
    let v = max_contract_violation(&c, &[f64::INFINITY], &[0.5]);
    assert_eq!(v, f64::MAX);
}

#[test]
fn test_contract_stage_bounds_fill() {
    let contracts = all_contracts();
    let j2_f0 = &contracts[0];
    let j5 = &contracts[5];
    let stage = contract_stage("test", &[2, 4], &[2, 8], j2_f0, j5, "IBP", false);
    // 2*4 = 8 input elements, all at J2_F0 bounds.
    assert_eq!(stage.input_lower.len(), 8);
    assert_eq!(stage.input_upper.len(), 8);
    assert!(stage.input_lower.iter().all(|&v| v == J2_F0_LOWER));
    assert!(stage.input_upper.iter().all(|&v| v == J2_F0_UPPER));
    // 2*8 = 16 output elements, all at J5_AUDIO bounds.
    assert_eq!(stage.output_lower.len(), 16);
    assert_eq!(stage.output_upper.len(), 16);
    assert!(stage.output_lower.iter().all(|&v| v == J5_AUDIO_LOWER));
    assert!(stage.output_upper.iter().all(|&v| v == J5_AUDIO_UPPER));
    assert!(!stage.is_sound);
    assert_eq!(stage.method, "IBP");
}

// ============================================================================
// Pipeline composition tests
// ============================================================================

fn make_stage(
    name: &str,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
    in_lo: f64,
    in_hi: f64,
    out_lo: f64,
    out_hi: f64,
    is_sound: bool,
) -> VerifiedStage {
    let in_elements: usize = in_shape.iter().product();
    let out_elements: usize = out_shape.iter().product();
    VerifiedStage::new(
        name,
        in_shape,
        out_shape,
        vec![in_lo; in_elements],
        vec![in_hi; in_elements],
        vec![out_lo; out_elements],
        vec![out_hi; out_elements],
        "CROWN",
        is_sound,
    )
}

#[test]
fn test_verify_pipeline_two_compatible_stages() {
    let s1 = make_stage("s1", vec![4], vec![4], -1.0, 1.0, -0.5, 0.5, true);
    let s2 = make_stage("s2", vec![4], vec![4], -0.6, 0.6, -0.3, 0.3, true);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(cert.is_valid);
    assert!(cert.is_sound);
    assert_eq!(cert.junctions.len(), 1);
    assert!(cert.junctions[0].bounds_contained);
    assert!(cert.junctions[0].shape_compatible);
}

#[test]
fn test_verify_pipeline_three_stages_valid() {
    let s1 = make_stage("s1", vec![4], vec![8], -1.0, 1.0, -0.5, 0.5, true);
    let s2 = make_stage("s2", vec![8], vec![8], -1.0, 1.0, -0.3, 0.3, true);
    let s3 = make_stage("s3", vec![8], vec![4], -0.5, 0.5, -0.1, 0.1, true);
    let cert = verify_pipeline(&[s1, s2, s3]).unwrap();
    assert!(cert.is_valid);
    assert!(cert.is_sound);
    assert_eq!(cert.junctions.len(), 2);
    // E2e bounds propagate from first input to last output.
    assert!(cert
        .e2e_input_lower
        .iter()
        .all(|&v| (v - (-1.0)).abs() < 1e-10));
    assert!(cert
        .e2e_output_upper
        .iter()
        .all(|&v| (v - 0.1).abs() < 1e-10));
}

#[test]
fn test_verify_pipeline_detects_bound_violation() {
    let s1 = make_stage("s1", vec![4], vec![4], -1.0, 1.0, -2.0, 2.0, true);
    let s2 = make_stage("s2", vec![4], vec![4], -1.0, 1.0, -0.5, 0.5, true);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(!cert.is_valid);
    assert!(!cert.junctions[0].bounds_contained);
    assert!(cert.junctions[0].max_violation > 0.0);
}

#[test]
fn test_verify_pipeline_detects_shape_incompatibility() {
    let s1 = make_stage("s1", vec![4], vec![8], -1.0, 1.0, -0.5, 0.5, true);
    let s2 = make_stage("s2", vec![4], vec![4], -1.0, 1.0, -0.3, 0.3, true);
    // s1 output elements=8, s2 input elements=4 → shape incompatible.
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(!cert.is_valid);
    assert!(!cert.junctions[0].shape_compatible);
}

#[test]
fn test_verify_pipeline_insufficient_stages() {
    let s = make_stage("s1", vec![4], vec![4], -1.0, 1.0, -0.5, 0.5, true);
    let err = verify_pipeline(&[s]);
    assert!(err.is_err());
}

#[test]
fn test_verify_pipeline_empty_stages() {
    let err = verify_pipeline(&[]);
    assert!(err.is_err());
}

#[test]
fn test_verify_pipeline_unsound_stage_propagates() {
    let s1 = make_stage("s1", vec![4], vec![4], -1.0, 1.0, -0.5, 0.5, true);
    let s2 = make_stage("s2", vec![4], vec![4], -0.6, 0.6, -0.3, 0.3, false);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(cert.is_valid);
    assert!(
        !cert.is_sound,
        "one unsound stage should make pipeline unsound"
    );
}

#[test]
fn test_check_junction_nan_bounds_are_violations() {
    let from = VerifiedStage::new(
        "from",
        vec![2],
        vec![2],
        vec![-1.0; 2],
        vec![1.0; 2],
        vec![f64::NAN, 0.5],
        vec![0.5, 0.5],
        "CROWN",
        true,
    );
    let to = VerifiedStage::new(
        "to",
        vec![2],
        vec![2],
        vec![-1.0; 2],
        vec![1.0; 2],
        vec![-0.5; 2],
        vec![0.5; 2],
        "CROWN",
        true,
    );
    let junction = check_junction(&from, &to, 0);
    assert!(!junction.bounds_contained);
    assert!(junction.violation_count > 0);
    assert_eq!(junction.max_violation, f64::MAX);
}

#[test]
fn test_check_junction_length_mismatch_counts_as_violation() {
    let from = VerifiedStage::new(
        "from",
        vec![2],
        vec![2],
        vec![-1.0; 2],
        vec![1.0; 2],
        vec![-0.5; 2],
        vec![0.5; 2],
        "CROWN",
        true,
    );
    let to = VerifiedStage::new(
        "to",
        vec![3],
        vec![3],
        vec![-1.0; 3],
        vec![1.0; 3],
        vec![-0.5; 3],
        vec![0.5; 3],
        "CROWN",
        true,
    );
    let junction = check_junction(&from, &to, 0);
    // Bounds vectors have different lengths: 2 vs 3.
    assert!(!junction.bounds_contained);
    assert!(junction.violation_count >= 1);
}

#[test]
fn test_pipeline_certificate_report_contains_stage_info() {
    let s1 = make_stage("encoder", vec![4], vec![4], -1.0, 1.0, -0.5, 0.5, true);
    let s2 = make_stage("decoder", vec![4], vec![4], -0.6, 0.6, -0.3, 0.3, true);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    let report = cert.report();
    assert!(report.contains("encoder"));
    assert!(report.contains("decoder"));
    assert!(report.contains("Pipeline Verification Report"));
    assert!(report.contains("Valid: true"));
    assert!(report.contains("Sound: true"));
}

#[test]
fn test_pipeline_certificate_display() {
    let s1 = make_stage("s1", vec![4], vec![4], -1.0, 1.0, -0.5, 0.5, true);
    let s2 = make_stage("s2", vec![4], vec![4], -0.6, 0.6, -0.3, 0.3, true);
    let cert = verify_pipeline(&[s1, s2]).unwrap();
    let display = format!("{cert}");
    assert!(display.contains("2 stages"));
    assert!(display.contains("valid=true"));
    assert!(display.contains("sound=true"));
}

// ============================================================================
// Moonshot property check tests (P1-P6 via moonshot_crown)
// ============================================================================

fn make_valid_pipeline_cert(out_lo: f64, out_hi: f64, is_sound: bool) -> PipelineCertificate {
    PipelineCertificate {
        e2e_input_lower: vec![-1.0; 8],
        e2e_input_upper: vec![1.0; 8],
        e2e_output_lower: vec![out_lo; 8],
        e2e_output_upper: vec![out_hi; 8],
        junctions: vec![],
        stages: vec![],
        is_valid: true,
        is_sound,
    }
}

// -- P1: Non-silence --

#[test]
fn test_p1_non_silence_proven_when_output_away_from_zero() {
    let cert = make_valid_pipeline_cert(-0.8, 0.8, true);
    let result = check_non_silence(&cert, 0.01);
    assert!(result.proven);
    assert_eq!(result.property_index, 0);
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.bound_value > 0.01);
}

#[test]
fn test_p1_non_silence_not_proven_when_output_near_zero() {
    let cert = make_valid_pipeline_cert(-0.005, 0.005, true);
    let result = check_non_silence(&cert, 0.01);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
}

#[test]
fn test_p1_non_silence_partial_when_unsound() {
    let cert = make_valid_pipeline_cert(-0.8, 0.8, false);
    let result = check_non_silence(&cert, 0.01);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

#[test]
fn test_p1_non_silence_fails_when_pipeline_invalid() {
    let mut cert = make_valid_pipeline_cert(-0.8, 0.8, true);
    cert.is_valid = false;
    let result = check_non_silence(&cert, 0.01);
    assert!(!result.proven);
}

// -- P2: Non-clipping --

#[test]
fn test_p2_non_clipping_proven_within_unit_range() {
    let cert = make_valid_pipeline_cert(-0.9, 0.9, true);
    let result = check_non_clipping(&cert);
    assert!(result.proven);
    assert_eq!(result.property_index, 1);
    assert_eq!(result.level, VerificationLevel::CrownProven);
}

#[test]
fn test_p2_non_clipping_proven_exact_boundary() {
    let cert = make_valid_pipeline_cert(-1.0, 1.0, true);
    let result = check_non_clipping(&cert);
    assert!(result.proven);
}

#[test]
fn test_p2_non_clipping_not_proven_when_exceeding_range() {
    let cert = make_valid_pipeline_cert(-1.1, 0.9, true);
    let result = check_non_clipping(&cert);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
}

#[test]
fn test_p2_non_clipping_not_proven_with_nan_bounds() {
    let mut cert = make_valid_pipeline_cert(-0.9, 0.9, true);
    cert.e2e_output_lower[0] = f64::NAN;
    let result = check_non_clipping(&cert);
    assert!(!result.proven);
}

#[test]
fn test_p2_non_clipping_partial_when_unsound() {
    let cert = make_valid_pipeline_cert(-0.9, 0.9, false);
    let result = check_non_clipping(&cert);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

// -- P5: Temporal boundedness --

fn make_timing_cert(worst_case_us: f64, bound_us: f64, is_sound: bool) -> TimingCertificate {
    let bounds_cert = make_valid_pipeline_cert(-0.5, 0.5, is_sound);
    TimingCertificate {
        bounds_cert,
        cost_profiles: vec![LayerCostProfile::new("test", 100, 200, worst_case_us, None)],
        worst_case_time_us: worst_case_us,
        total_flops: 100,
        total_memory_bytes: 200,
        hardware_name: "Apple M4 Max".to_string(),
        timing_bound_us: bound_us,
        timing_bound_met: worst_case_us <= bound_us,
        overall_passed: worst_case_us <= bound_us,
        peak_memory: None,
    }
}

#[test]
fn test_p5_temporal_boundedness_proven_within_budget() {
    let tc = make_timing_cert(50_000.0, 100_000.0, true);
    let result = check_temporal_boundedness(&tc);
    assert!(result.proven);
    assert_eq!(result.property_index, 4);
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!((result.bound_value - 50_000.0).abs() < 1e-10);
    assert!((result.threshold - 100_000.0).abs() < 1e-10);
}

#[test]
fn test_p5_temporal_boundedness_not_proven_exceeds_budget() {
    let tc = make_timing_cert(150_000.0, 100_000.0, true);
    let result = check_temporal_boundedness(&tc);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
}

#[test]
fn test_p5_temporal_boundedness_partial_when_unsound() {
    let tc = make_timing_cert(50_000.0, 100_000.0, false);
    let result = check_temporal_boundedness(&tc);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
}

// -- P6: Streaming safety --

#[test]
fn test_p6_streaming_safety_proven_tight_bounds() {
    // Output range [-0.5, 0.5] → range = 1.0.
    // crossfade_samples=480 → alpha_step = 1/479 ≈ 0.002088.
    // max_click_bound = 1.0 * 0.002088 ≈ 0.002088, well below 0.3.
    let cert = make_valid_pipeline_cert(-0.5, 0.5, true);
    let result = check_streaming_safety(&cert, 480, 0.3);
    assert!(result.proven);
    assert_eq!(result.property_index, 5);
    assert_eq!(result.level, VerificationLevel::CrownProven);
}

#[test]
fn test_p6_streaming_safety_not_proven_wide_bounds() {
    // Output range [-100.0, 100.0] → range = 200.0.
    // alpha_step = 1/479 ≈ 0.002088.
    // max_click_bound = 200.0 * 0.002088 ≈ 0.4175, exceeds 0.3.
    let cert = make_valid_pipeline_cert(-100.0, 100.0, true);
    let result = check_streaming_safety(&cert, 480, 0.3);
    assert!(!result.proven);
}

#[test]
fn test_p6_streaming_safety_degenerate_single_sample_crossfade() {
    // crossfade_samples=1 → alpha_step = 1.0 (full discontinuity possible).
    let cert = make_valid_pipeline_cert(-0.5, 0.5, true);
    let result = check_streaming_safety(&cert, 1, 0.3);
    // range = 1.0, alpha_step = 1.0, max_click_bound = 1.0 > 0.3.
    assert!(!result.proven);
}

#[test]
fn test_p6_streaming_safety_with_nan_bounds_not_proven() {
    let mut cert = make_valid_pipeline_cert(-0.5, 0.5, true);
    cert.e2e_output_upper[0] = f64::NAN;
    let result = check_streaming_safety(&cert, 480, 0.3);
    assert!(!result.proven);
}

// ============================================================================
// Moonshot property bundle tests
// ============================================================================

#[test]
fn test_verify_properties_from_pipeline_returns_4_results() {
    let cert = make_valid_pipeline_cert(-0.5, 0.5, true);
    let bundle = verify_properties_from_pipeline(&cert, 64);
    // P1 (non-silence), P2 (non-clipping), P3 (intelligibility), P6 (streaming).
    assert_eq!(bundle.results.len(), 4);
    assert_eq!(bundle.verification_dim, 64);
}

#[test]
fn test_verify_properties_from_pipeline_all_proven_tight_bounds() {
    let cert = make_valid_pipeline_cert(-0.5, 0.5, true);
    let bundle = verify_properties_from_pipeline(&cert, 64);
    assert!(bundle.all_proven);
    for r in &bundle.results {
        assert!(r.proven, "property {} should be proven", r.property_name);
    }
}

#[test]
fn test_verify_properties_from_pipeline_not_all_proven_wide_bounds() {
    // Wide bounds will fail P6 (streaming safety).
    let cert = make_valid_pipeline_cert(-200.0, 200.0, true);
    let bundle = verify_properties_from_pipeline(&cert, 64);
    // P2 (non-clipping) fails because bounds exceed [-1, 1].
    assert!(!bundle.all_proven);
}

// ============================================================================
// Moonshot status and certificate tests
// ============================================================================

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

#[test]
fn test_verification_level_ordering_is_total() {
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
            "{:?} should be < {:?}",
            levels[i],
            levels[i + 1]
        );
    }
}

#[test]
fn test_verification_level_crown_probabilistic_ordering() {
    assert!(VerificationLevel::CrownPartial < VerificationLevel::CrownProbabilistic);
    assert!(VerificationLevel::CrownProbabilistic < VerificationLevel::CrownProven);
}

#[test]
fn test_moonshot_status_from_repo_has_8_properties() {
    let status = MoonshotStatus::from_repo();
    assert_eq!(status.properties.len(), 8);
}

#[test]
fn test_moonshot_status_level_counts_sum_to_8() {
    let status = MoonshotStatus::from_repo();
    let counts = status.level_counts();
    let total: usize = counts.iter().map(|(_, c)| c).sum();
    assert_eq!(total, 8);
}

#[test]
fn test_moonshot_status_all_have_evidence() {
    let status = MoonshotStatus::from_repo();
    assert!(
        status.all_have_evidence(),
        "All 8 properties should have at least some verification evidence"
    );
}

#[test]
fn test_certificate_schema_version() {
    assert_eq!(CERTIFICATE_SCHEMA_VERSION, 4);
}

#[test]
fn test_certificate_from_status_populates_all_fields() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(
        &status,
        "test-model",
        "English text, max 50 words",
        "deadbeef1234",
    );
    assert_eq!(cert.model_name, "test-model");
    assert_eq!(cert.input_specification, "English text, max 50 words");
    assert_eq!(cert.source_hash, "deadbeef1234");
    assert_eq!(cert.schema_version, CERTIFICATE_SCHEMA_VERSION);
    assert_eq!(cert.properties.len(), 8);
    assert!(cert.verification_dim.is_none());
    // Verification date should be non-empty.
    assert!(!cert.verification_date.is_empty());
}

#[test]
fn test_certificate_property_indices_are_sequential() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");
    for (i, prop) in cert.properties.iter().enumerate() {
        assert_eq!(prop.property_index, i);
    }
}

#[test]
fn test_certificate_crown_partial_properties_have_formal_assumptions() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");
    for prop in &cert.properties {
        if prop.level >= VerificationLevel::CrownPartial {
            assert!(
                prop.assumptions
                    .iter()
                    .any(|a| a.contains("Input within specified bounds")),
                "CrownPartial+ property {} should have formal assumption",
                prop.property_name
            );
        }
    }
}

#[test]
fn test_certificate_to_json_roundtrip_contains_all_keys() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "roundtrip-test", "English", "abc123");
    let json = cert.to_json();
    // Verify key structural fields are present.
    assert!(json.contains("\"model_name\""));
    assert!(json.contains("\"schema_version\""));
    assert!(json.contains("\"properties\""));
    assert!(json.contains("\"source_hash\""));
    assert!(json.contains("\"verification_date\""));
    assert!(json.contains("\"all_at_least_partial\""));
    assert!(json.contains("\"all_proven\""));
}

// ============================================================================
// CoupledTimingCertificate tests
// ============================================================================

#[test]
fn test_coupled_timing_certificate_all_layers_coupled() {
    let stage = make_stage("s1", vec![4], vec![4], -1.0, 1.0, -0.5, 0.5, true);
    let profile = LayerCostProfile::new("s1", 1000, 2000, 10.0, None);
    let coupled = CoupledLayerResult {
        stage,
        cost_profile: profile,
        dispatch_step_count: 3,
    };
    let tc = make_timing_cert(10.0, 100.0, true);
    let cert = CoupledTimingCertificate {
        timing: tc,
        coupled_layers: vec![coupled],
        total_dispatch_steps: 3,
    };
    assert!(cert.all_layers_coupled());
}

#[test]
fn test_coupled_timing_certificate_not_coupled_zero_dispatch() {
    let stage = make_stage("s1", vec![4], vec![4], -1.0, 1.0, -0.5, 0.5, true);
    let profile = LayerCostProfile::new("s1", 1000, 2000, 10.0, None);
    let coupled = CoupledLayerResult {
        stage,
        cost_profile: profile,
        dispatch_step_count: 0, // No dispatch steps.
    };
    let tc = make_timing_cert(10.0, 100.0, true);
    let cert = CoupledTimingCertificate {
        timing: tc,
        coupled_layers: vec![coupled],
        total_dispatch_steps: 0,
    };
    assert!(!cert.all_layers_coupled());
}

#[test]
fn test_coupled_timing_certificate_not_coupled_empty() {
    let tc = make_timing_cert(10.0, 100.0, true);
    let cert = CoupledTimingCertificate {
        timing: tc,
        coupled_layers: vec![],
        total_dispatch_steps: 0,
    };
    assert!(!cert.all_layers_coupled());
}

#[test]
fn test_coupled_timing_certificate_report_includes_layer_info() {
    let stage = make_stage("test_layer", vec![4], vec![4], -1.0, 1.0, -0.5, 0.5, true);
    let profile = LayerCostProfile::new("test_layer", 1000, 2000, 10.0, None);
    let coupled = CoupledLayerResult {
        stage,
        cost_profile: profile,
        dispatch_step_count: 2,
    };
    let tc = make_timing_cert(10.0, 100.0, true);
    let cert = CoupledTimingCertificate {
        timing: tc,
        coupled_layers: vec![coupled],
        total_dispatch_steps: 2,
    };
    let report = cert.report();
    assert!(report.contains("test_layer"));
    assert!(report.contains("Per-Layer Coupled Verification"));
}

// ============================================================================
// Moonshot property result field verification
// ============================================================================

#[test]
fn test_moonshot_property_result_explanation_contains_status() {
    let cert = make_valid_pipeline_cert(-0.5, 0.5, true);
    let result = check_non_silence(&cert, 0.01);
    assert!(
        result.explanation.contains("PROVEN") || result.explanation.contains("NOT PROVEN"),
        "explanation should contain proof status"
    );
}

#[test]
fn test_moonshot_property_result_threshold_matches_input() {
    let cert = make_valid_pipeline_cert(-0.5, 0.5, true);
    let result = check_non_silence(&cert, 0.05);
    assert!((result.threshold - 0.05).abs() < 1e-10);
}

#[test]
fn test_p2_non_clipping_threshold_is_one() {
    let cert = make_valid_pipeline_cert(-0.5, 0.5, true);
    let result = check_non_clipping(&cert);
    assert!((result.threshold - 1.0).abs() < 1e-10);
}

// ============================================================================
// Pipeline with Kokoro junction contracts — integration
// ============================================================================

#[test]
fn test_kokoro_full_pipeline_compose_5_stages() {
    let contracts = all_contracts();
    let j4_bf16 = &contracts[4];
    let j2_f0 = &contracts[0];
    let j2_energy = &contracts[1];
    let j3_mag = &contracts[2];
    let j3b_phase = &contracts[3];
    let j5_audio = &contracts[5];

    // 5-stage pipeline simulating Kokoro TTS:
    // 1. PLBert: BF16 -> F0
    let plbert = contract_stage("plbert", &[1, 64], &[1, 64], j4_bf16, j2_f0, "CROWN", true);
    // 2. Prosody: F0 -> Energy (narrower output fits Magnitude input)
    let prosody = contract_stage(
        "prosody",
        &[1, 64],
        &[1, 64],
        j2_f0,
        j2_energy,
        "CROWN",
        true,
    );
    // 3. Decoder: Energy -> Magnitude (energy [-50,50] fits magnitude [-80,80])
    let decoder = contract_stage(
        "decoder",
        &[1, 64],
        &[1, 64],
        j2_energy,
        j3_mag,
        "CROWN",
        true,
    );
    // 4. Generator: Magnitude -> Phase
    let generator = contract_stage(
        "generator",
        &[1, 64],
        &[1, 64],
        j3_mag,
        j3b_phase,
        "CROWN",
        true,
    );
    // 5. iSTFT: Phase -> Audio
    let istft = contract_stage(
        "istft",
        &[1, 64],
        &[1, 24000],
        j3b_phase,
        j5_audio,
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[plbert, prosody, decoder, generator, istft]).unwrap();
    assert_eq!(cert.junctions.len(), 4);

    // Junction 0: PLBert F0 output [-5, 800] fits Prosody F0 input [-5, 800] exactly.
    assert!(cert.junctions[0].bounds_contained);

    // Junction 1: Prosody Energy [-50, 50] fits Decoder Energy [-50, 50] exactly.
    assert!(cert.junctions[1].bounds_contained);

    // Junction 2: Decoder Magnitude [-80, 80] matches Generator Magnitude [-80, 80].
    assert!(cert.junctions[2].bounds_contained);

    // Junction 3: Generator Phase output [-6283.2, 6283.2] matches iSTFT input.
    assert!(cert.junctions[3].bounds_contained);

    assert!(cert.is_valid);
    assert!(cert.is_sound);
    assert!(cert.e2e_input_lower.iter().all(|&v| v == J4_BF16_LOWER));
    assert!(cert.e2e_output_upper.iter().all(|&v| v == J5_AUDIO_UPPER));
}

#[test]
fn test_moonshot_p2_on_kokoro_pipeline_output() {
    let contracts = all_contracts();
    let j3b_phase = &contracts[3];
    let j5_audio = &contracts[5];

    // Build a minimal 2-stage pipeline ending in audio output [-1, 1].
    let stage1 = contract_stage(
        "generator",
        &[1, 16],
        &[1, 16],
        j3b_phase,
        j3b_phase,
        "CROWN",
        true,
    );
    let stage2 = contract_stage(
        "istft",
        &[1, 16],
        &[1, 16],
        j3b_phase,
        j5_audio,
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[stage1, stage2]).unwrap();
    assert!(cert.is_valid);

    // P2: audio output is [-1, 1], so non-clipping is proven.
    let p2 = check_non_clipping(&cert);
    assert!(p2.proven);
    assert_eq!(p2.level, VerificationLevel::CrownProven);
}
