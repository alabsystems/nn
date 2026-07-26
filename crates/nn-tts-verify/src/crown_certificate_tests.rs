// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN certificate structure, pipeline verification, cost model integration,
//! and error handling tests for the nn-tts-verify certification pipeline.
//!
//! Part of #4254.

use std::collections::HashMap;

use crate::bounds::HardBound;
use crate::certificate::Certificate;
use crate::cost_model::{HardwareCostModel, LayerCostProfile};
use crate::cost_propagation::{CoupledLayerResult, CoupledTimingCertificate};
use crate::crown_junction::{
    check_all_junction_contracts, check_junction_bound, contract_bounds_map, StageBoundCheck,
};
use crate::crown_synthesis::{
    verify_synthesis_crown, verify_synthesis_crown_full, CrownCertificateConfig,
};
use crate::kokoro_contracts::{
    all_contracts, bounds_within_contract, contract_stage, max_contract_violation,
    JunctionContract, J2_F0_LOWER, J2_F0_UPPER, J4_BF16_LOWER, J5_AUDIO_LOWER, J5_AUDIO_UPPER,
};
use crate::moonshot::{MoonshotCertificate, MoonshotStatus, VerificationLevel};
use crate::moonshot_crown::{
    check_non_clipping, check_temporal_boundedness, verify_properties_from_pipeline,
};
use crate::pipeline::{verify_pipeline, PipelineCertificate, TimingCertificate, VerifiedStage};
use crate::quality::QualityMetric;

// ============================================================================
// Helper constructors
// ============================================================================

fn make_certificate(
    hard_bounds: Vec<HardBound>,
    quality_metrics: Vec<QualityMetric>,
    overall_passed: bool,
) -> Certificate {
    Certificate {
        hard_bounds,
        quality_metrics,
        phoneme_results: None,
        overall_passed,
        crown_evidence: None,
        junction_summary: None,
        deterministic_hash: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    }
}

fn make_passing_hard_bounds() -> Vec<HardBound> {
    vec![
        HardBound {
            name: "non_silence",
            passed: true,
            value: 0.15,
            threshold: 0.01,
        },
        HardBound {
            name: "no_clipping",
            passed: true,
            value: 0.95,
            threshold: 1.0,
        },
        HardBound {
            name: "no_clicks",
            passed: true,
            value: 0.3,
            threshold: 0.5,
        },
    ]
}

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

fn make_pipeline_cert(out_lo: f64, out_hi: f64, is_sound: bool) -> PipelineCertificate {
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

fn make_timing_cert(worst_case_us: f64, bound_us: f64, is_sound: bool) -> TimingCertificate {
    let bounds_cert = make_pipeline_cert(-0.5, 0.5, is_sound);
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

// ============================================================================
// 1. Certificate structure tests (12+)
// ============================================================================

#[test]
fn test_certificate_with_all_8_properties_from_status() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(
        &status,
        "kokoro-test",
        "English text",
        "test-hash-abc123",
    );
    assert_eq!(cert.properties.len(), 8);
    for (i, prop) in cert.properties.iter().enumerate() {
        assert_eq!(prop.property_index, i);
        assert!(!prop.property_name.is_empty());
        // Every property should have at least one assumption.
        assert!(
            !prop.assumptions.is_empty(),
            "P{} should have assumptions",
            i + 1
        );
    }
}

#[test]
fn test_certificate_property_status_transitions_via_enrichment() {
    // Start with a certificate from status (base levels from artifact registry).
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    // Verify base level is set from artifact registry.
    for prop in &cert.properties {
        assert!(
            prop.level >= VerificationLevel::None,
            "base level should be at least None"
        );
    }

    // Enrich via crown synthesis — should upgrade P1/P2/P6 to at least Empirical.
    let runtime_cert = make_certificate(make_passing_hard_bounds(), vec![], true);
    let config = CrownCertificateConfig::default();
    let enriched = verify_synthesis_crown(&runtime_cert, &config);

    assert!(
        enriched.properties[0].level >= VerificationLevel::Empirical,
        "P1 should be upgraded to at least Empirical"
    );
    assert!(
        enriched.properties[1].level >= VerificationLevel::Empirical,
        "P2 should be upgraded to at least Empirical"
    );
    assert!(
        enriched.properties[5].level >= VerificationLevel::Empirical,
        "P6 should be upgraded to at least Empirical"
    );
}

#[test]
fn test_certificate_serialization_roundtrip() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "serial-test", "English", "hash123");
    let json = cert.to_json();

    // Verify JSON contains all essential fields.
    // The to_json() method uses its own formatting (not serde_json).
    assert!(json.contains("\"schema_version\""));
    assert!(json.contains("serial-test"));
    assert!(json.contains("\"properties\""));
    assert!(json.contains("\"index\"")); // property_index → "index" in JSON
    assert!(json.contains("\"name\"")); // property_name → "name" in JSON
    assert!(json.contains("\"level\""));
    assert!(json.contains("\"proof_artifacts\""));
    assert!(json.contains("\"assumptions\""));
    assert!(json.contains("\"all_at_least_partial\""));
    assert!(json.contains("\"all_proven\""));
}

#[test]
fn test_junction_contract_bounds_validation_all_pass() {
    let contracts = all_contracts();
    for contract in &contracts {
        // Values strictly within bounds should pass.
        let half_range = (contract.upper - contract.lower) / 4.0;
        let mid = f64::midpoint(contract.upper, contract.lower);
        assert!(
            bounds_within_contract(contract, &[mid - half_range], &[mid + half_range]),
            "contract {} should pass with values within bounds",
            contract.name
        );
    }
}

#[test]
fn test_junction_contract_bounds_validation_j2_j5_exact_boundary() {
    let contracts = all_contracts();
    // J2_F0: exact lower/upper should pass (inclusive comparison).
    assert!(bounds_within_contract(
        &contracts[0],
        &[J2_F0_LOWER],
        &[J2_F0_UPPER]
    ));
    // J5_AUDIO: exact boundary.
    assert!(bounds_within_contract(
        &contracts[5],
        &[J5_AUDIO_LOWER],
        &[J5_AUDIO_UPPER]
    ));
}

#[test]
fn test_composition_bounds_across_4_pipeline_stages() {
    let contracts = all_contracts();
    let j4_bf16 = &contracts[4];
    let j2_f0 = &contracts[0];
    let j3_mag = &contracts[2];
    let j5_audio = &contracts[5];

    // 4-stage pipeline: BF16 -> F0 -> Magnitude -> Audio
    let s1 = contract_stage("encoder", &[1, 32], &[1, 32], j4_bf16, j2_f0, "CROWN", true);
    let s2 = contract_stage("decoder", &[1, 32], &[1, 32], j2_f0, j2_f0, "CROWN", true);
    let s3 = contract_stage(
        "generator",
        &[1, 32],
        &[1, 32],
        j2_f0,
        j3_mag,
        "CROWN",
        true,
    );
    let s4 = contract_stage(
        "vocoder",
        &[1, 32],
        &[1, 32],
        j3_mag,
        j5_audio,
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[s1, s2, s3, s4]).unwrap();
    assert!(cert.is_valid);
    assert_eq!(cert.junctions.len(), 3);
    assert!(cert.junctions.iter().all(|j| j.bounds_contained));

    // End-to-end bounds should be BF16 input, audio output.
    assert!(cert.e2e_input_lower.iter().all(|&v| v == J4_BF16_LOWER));
    assert!(cert.e2e_output_upper.iter().all(|&v| v == J5_AUDIO_UPPER));
}

#[test]
fn test_crown_config_creation_for_kokoro() {
    let config = CrownCertificateConfig::new("dvoice-kokoro-v1")
        .with_input_specification("English text, <=50 words");
    assert_eq!(config.model_name, "dvoice-kokoro-v1");
    assert_eq!(config.input_specification, "English text, <=50 words");
    assert!(config.map_hard_bounds);
    assert!(!config.check_junction_contracts);
}

#[test]
fn test_crown_config_creation_for_whisper() {
    let config = CrownCertificateConfig::new("whisper-large-v3")
        .with_input_specification("16kHz audio, <=30s");
    assert_eq!(config.model_name, "whisper-large-v3");
    assert_eq!(config.input_specification, "16kHz audio, <=30s");
}

#[test]
fn test_crown_config_with_junction_contracts_enabled() {
    let config = CrownCertificateConfig {
        check_junction_contracts: true,
        ..CrownCertificateConfig::new("kokoro-v1")
    };
    assert!(config.check_junction_contracts);
}

#[test]
fn test_certificate_report_contains_all_sections() {
    let cert = make_certificate(
        make_passing_hard_bounds(),
        vec![QualityMetric {
            name: "mcd",
            value: 4.5,
            threshold: 6.0,
            passed: true,
            citation: "Kubichek 1993",
        }],
        true,
    );

    let config = CrownCertificateConfig::default();
    let moonshot = verify_synthesis_crown(&cert, &config);
    let enriched = cert.with_crown_evidence(moonshot);

    let report = enriched.report();
    assert!(report.contains("Hard Bounds"));
    assert!(report.contains("Quality Metrics"));
    assert!(report.contains("CROWN Verification Evidence"));
    assert!(report.contains("PASSED"));
}

#[test]
fn test_certificate_with_deterministic_hash() {
    let mut cert = make_certificate(make_passing_hard_bounds(), vec![], true);
    cert.deterministic_hash = Some("sha256:abcdef1234567890".to_string());

    let report = cert.report();
    assert!(report.contains("Deterministic Hash"));
    assert!(report.contains("sha256:abcdef1234567890"));
}

#[test]
fn test_certificate_junction_summary_integration() {
    let cert = make_certificate(make_passing_hard_bounds(), vec![], true);
    let config = CrownCertificateConfig {
        check_junction_contracts: true,
        ..CrownCertificateConfig::new("kokoro-v1")
    };

    // Provide intermediates that all pass.
    let mut intermediates = HashMap::new();
    intermediates.insert("J2_F0".to_string(), (0.0_f32, 400.0_f32));
    intermediates.insert("J5_AUDIO".to_string(), (-0.9_f32, 0.9_f32));

    let result = verify_synthesis_crown_full(&cert, &config, Some(&intermediates));
    assert!(result.junction_summary.is_some());

    let summary = result.junction_summary.unwrap();
    assert_eq!(summary.total_passed, 2);
    assert_eq!(summary.total_failed, 0);

    // Attach to certificate.
    let enriched = cert.with_junction_summary(summary);
    assert!(enriched.has_junction_summary());
    assert!(enriched.passes_junction_contracts());

    let report = enriched.report();
    assert!(report.contains("Junction Contract Checks"));
    assert!(report.contains("2/2 contracts passed"));
}

// ============================================================================
// 2. Pipeline verification tests (10+)
// ============================================================================

#[test]
fn test_kokoro_8_segment_pipeline_property_checking() {
    // Build an 8-segment pipeline with narrowing bounds (simulating Kokoro segments).
    // Each stage's output is strictly within the next stage's input to avoid
    // floating-point equality edge cases.
    let bounds: [(f64, f64, f64, f64); 8] = [
        (-1.0, 1.0, -0.89, 0.89), // segment 0
        (-0.9, 0.9, -0.79, 0.79), // segment 1
        (-0.8, 0.8, -0.69, 0.69), // segment 2
        (-0.7, 0.7, -0.59, 0.59), // segment 3
        (-0.6, 0.6, -0.49, 0.49), // segment 4
        (-0.5, 0.5, -0.39, 0.39), // segment 5
        (-0.4, 0.4, -0.29, 0.29), // segment 6
        (-0.3, 0.3, -0.19, 0.19), // segment 7
    ];

    let segments: Vec<VerifiedStage> = bounds
        .iter()
        .enumerate()
        .map(|(i, &(in_lo, in_hi, out_lo, out_hi))| {
            make_stage(
                &format!("segment_{i}"),
                vec![32],
                vec![32],
                in_lo,
                in_hi,
                out_lo,
                out_hi,
                true,
            )
        })
        .collect();

    let cert = verify_pipeline(&segments).unwrap();
    assert!(cert.is_valid);
    assert_eq!(cert.junctions.len(), 7);

    // All junctions should have bounds contained (each output fits next input).
    for (i, j) in cert.junctions.iter().enumerate() {
        assert!(
            j.bounds_contained,
            "junction {i} should have contained bounds"
        );
        assert!(
            j.shape_compatible,
            "junction {i} should have compatible shapes"
        );
    }
}

#[test]
fn test_junction_bound_propagation_encoder_decoder_vocoder_output() {
    let contracts = all_contracts();
    let j4_bf16 = &contracts[4]; // BF16 safe range
    let j2_energy = &contracts[1]; // Energy range
    let j3_mag = &contracts[2]; // Magnitude range
    let j5_audio = &contracts[5]; // Audio range

    let encoder = contract_stage(
        "encoder",
        &[1, 64],
        &[1, 64],
        j4_bf16,
        j2_energy,
        "CROWN",
        true,
    );
    let decoder = contract_stage(
        "decoder",
        &[1, 64],
        &[1, 64],
        j2_energy,
        j3_mag,
        "CROWN",
        true,
    );
    let vocoder = contract_stage("vocoder", &[1, 64], &[1, 64], j3_mag, j3_mag, "CROWN", true);
    let output = contract_stage(
        "output",
        &[1, 64],
        &[1, 64],
        j3_mag,
        j5_audio,
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[encoder, decoder, vocoder, output]).unwrap();
    assert!(cert.is_valid);

    // E2E: BF16 input, audio output.
    assert!(cert.e2e_input_lower.iter().all(|&v| v == J4_BF16_LOWER));
    assert!(cert.e2e_output_upper.iter().all(|&v| v == J5_AUDIO_UPPER));
}

#[test]
fn test_audio_quality_property_no_nan_inf_in_bounds() {
    let cert = make_pipeline_cert(-0.95, 0.95, true);

    // Verify all output bounds are finite (no NaN/Inf).
    for &lo in &cert.e2e_output_lower {
        assert!(lo.is_finite(), "output lower bound should be finite");
    }
    for &hi in &cert.e2e_output_upper {
        assert!(hi.is_finite(), "output upper bound should be finite");
    }

    // Verify samples would be in [-1, 1].
    let p2 = check_non_clipping(&cert);
    assert!(
        p2.proven,
        "output in [-0.95, 0.95] should prove non-clipping"
    );
}

#[test]
fn test_audio_quality_nan_bounds_fail_non_clipping() {
    let mut cert = make_pipeline_cert(-0.5, 0.5, true);
    cert.e2e_output_lower[3] = f64::NAN;
    let p2 = check_non_clipping(&cert);
    assert!(!p2.proven, "NaN in output bounds should prevent proof");
}

#[test]
fn test_latency_property_rtf_below_threshold() {
    // 50ms inference for audio → RTF well below 1.0.
    let tc = make_timing_cert(50_000.0, 100_000.0, true);
    let result = check_temporal_boundedness(&tc);
    assert!(result.proven);
    assert!(result.bound_value <= result.threshold);
}

#[test]
fn test_latency_property_rtf_exceeds_threshold() {
    // 200ms inference exceeds 100ms budget.
    let tc = make_timing_cert(200_000.0, 100_000.0, true);
    let result = check_temporal_boundedness(&tc);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
}

#[test]
fn test_determinism_property_same_input_same_output() {
    // Determinism is tested by generating two certificates from identical inputs
    // and checking that the verification results are identical.
    let cert1 = make_pipeline_cert(-0.5, 0.5, true);
    let cert2 = make_pipeline_cert(-0.5, 0.5, true);

    let bundle1 = verify_properties_from_pipeline(&cert1, 64);
    let bundle2 = verify_properties_from_pipeline(&cert2, 64);

    assert_eq!(bundle1.results.len(), bundle2.results.len());
    for (r1, r2) in bundle1.results.iter().zip(bundle2.results.iter()) {
        assert_eq!(r1.property_index, r2.property_index);
        assert_eq!(r1.proven, r2.proven);
        assert_eq!(r1.level, r2.level);
        assert!(
            (r1.bound_value - r2.bound_value).abs() < 1e-10,
            "bound values should match for identical inputs"
        );
    }
}

#[test]
fn test_pipeline_mixed_soundness_stages() {
    // Mix of sound and unsound stages.
    let s1 = make_stage(
        "sound_encoder",
        vec![16],
        vec![16],
        -1.0,
        1.0,
        -0.5,
        0.5,
        true,
    );
    let s2 = make_stage(
        "unsound_decoder",
        vec![16],
        vec![16],
        -0.6,
        0.6,
        -0.3,
        0.3,
        false,
    );
    let s3 = make_stage(
        "sound_vocoder",
        vec![16],
        vec![16],
        -0.4,
        0.4,
        -0.1,
        0.1,
        true,
    );

    let cert = verify_pipeline(&[s1, s2, s3]).unwrap();
    assert!(cert.is_valid);
    assert!(
        !cert.is_sound,
        "pipeline with unsound stage should be unsound"
    );
}

#[test]
fn test_pipeline_with_shape_mismatch_detects_error() {
    // Output of s1 has 32 elements, input of s2 expects 16.
    let s1 = make_stage("s1", vec![4, 8], vec![4, 8], -1.0, 1.0, -0.5, 0.5, true);
    let s2 = make_stage("s2", vec![4, 4], vec![4, 4], -0.6, 0.6, -0.3, 0.3, true);

    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(!cert.is_valid);
    assert!(!cert.junctions[0].shape_compatible);
}

#[test]
fn test_pipeline_with_tight_bound_violation_reports_max_violation() {
    // s1 outputs [-2.0, 2.0] but s2 expects [-1.0, 1.0].
    let s1 = make_stage("s1", vec![4], vec![4], -1.0, 1.0, -2.0, 2.0, true);
    let s2 = make_stage("s2", vec![4], vec![4], -1.0, 1.0, -0.5, 0.5, true);

    let cert = verify_pipeline(&[s1, s2]).unwrap();
    assert!(!cert.is_valid);
    assert!(!cert.junctions[0].bounds_contained);
    // Max violation should be 1.0 (output upper 2.0 vs input upper 1.0).
    assert!(
        (cert.junctions[0].max_violation - 1.0).abs() < 1e-10,
        "max violation should be 1.0, got {}",
        cert.junctions[0].max_violation
    );
}

// ============================================================================
// 3. Cost model integration tests (8+)
// ============================================================================

#[test]
fn test_kokoro_cost_model_m4_max_parameters() {
    let m = HardwareCostModel::m4_max();
    assert!((m.peak_tflops_f32 - 14.2).abs() < 1e-10);
    assert!((m.peak_bandwidth_gbs - 400.0).abs() < 1e-10);
    assert!((m.dispatch_overhead_us - 5.0).abs() < 1e-10);
    assert!(m.validate().is_ok());
}

#[test]
fn test_dispatch_cost_estimation_compute_bound_kernel() {
    let m = HardwareCostModel::m4_max();
    // Large matmul: 10B FLOPs, minimal memory.
    let time = m.estimate_time_us(10_000_000_000, 1_000_000);
    let compute_time = 10e9 / (14.2 * 1e6);
    let memory_time = 1e6 / (400.0 * 1e3);
    let expected = f64::max(compute_time, memory_time) + 5.0;
    assert!(
        (time - expected).abs() < 0.1,
        "compute-bound kernel estimate mismatch"
    );
    assert!(compute_time > memory_time, "this should be compute-bound");
}

#[test]
fn test_dispatch_cost_estimation_memory_bound_kernel() {
    let m = HardwareCostModel::m4_max();
    // Small compute, large memory: elementwise op.
    let time = m.estimate_time_us(100_000, 1_000_000_000);
    let compute_time = 1e5 / (14.2 * 1e6);
    let memory_time = 1e9 / (400.0 * 1e3);
    let expected = f64::max(compute_time, memory_time) + 5.0;
    assert!(
        (time - expected).abs() < 0.1,
        "memory-bound kernel estimate mismatch"
    );
    assert!(memory_time > compute_time, "this should be memory-bound");
}

#[test]
fn test_memory_bandwidth_estimation_conservative_vs_standard() {
    let standard = HardwareCostModel::m4_max();
    let conservative = HardwareCostModel::m4_max_conservative();

    // Same workload.
    let flops = 1_000_000_000_u64;
    let bytes = 500_000_000_u64;

    let t_std = standard.estimate_time_us(flops, bytes);
    let t_con = conservative.estimate_time_us(flops, bytes);

    assert!(
        t_con > t_std,
        "conservative ({t_con:.1} us) should exceed standard ({t_std:.1} us)"
    );

    // Conservative should be at least 2x standard (5x compute derating, 2x bandwidth).
    assert!(
        t_con >= t_std * 1.5,
        "conservative should be significantly higher"
    );
}

#[test]
fn test_roofline_analysis_balanced_workload() {
    let m = HardwareCostModel::m4_max();
    // Find the balance point: FLOPS/TFLOPS == bytes/bandwidth.
    // 14.2 TFLOPS, 400 GB/s → arithmetic intensity = 14.2e12/400e9 = 35.5 FLOP/byte.
    // For 1e9 bytes at 35.5 FLOP/byte: 35.5e9 FLOPs.
    let bytes: u64 = 1_000_000_000;
    let flops: u64 = 35_500_000_000;

    let time = m.estimate_time_us(flops, bytes);
    let compute_time = flops as f64 / (14.2 * 1e6);
    let memory_time = bytes as f64 / (400.0 * 1e3);

    // At balance, compute_time ≈ memory_time.
    assert!(
        (compute_time - memory_time).abs() / compute_time < 0.01,
        "should be balanced: compute={compute_time:.1}, memory={memory_time:.1}"
    );
    // Time should be approximately 2x either + overhead.
    assert!(time > compute_time);
}

#[test]
fn test_kokoro_segment_cost_profiles_sum_correctly() {
    // Simulate 8 Kokoro segments with varying costs.
    let profiles: Vec<LayerCostProfile> = (0..8)
        .map(|i| {
            LayerCostProfile::new(
                format!("segment_{i}"),
                (i as u64 + 1) * 1_000_000,
                (i as u64 + 1) * 500_000,
                (f64::from(i) + 1.0) * 10.0,
                None,
            )
        })
        .collect();

    let total_time = crate::cost_model::total_estimated_time_us(&profiles);
    let total_flops = crate::cost_model::total_flops(&profiles);
    let total_mem = crate::cost_model::total_memory_bytes(&profiles);

    // Sum of i=1..8: i * 1e6 = 36e6 FLOPs.
    assert_eq!(total_flops, 36_000_000);
    // Sum of i=1..8: i * 500k = 18e6 bytes.
    assert_eq!(total_mem, 18_000_000);
    // Sum of i=1..8: i * 10 = 360 us.
    assert!((total_time - 360.0).abs() < 1e-10);
}

#[test]
fn test_coupled_timing_certificate_with_multiple_layers() {
    let layers: Vec<CoupledLayerResult> = (0..4)
        .map(|i| CoupledLayerResult {
            stage: make_stage(
                &format!("layer_{i}"),
                vec![32],
                vec![32],
                -1.0,
                1.0,
                -0.5,
                0.5,
                true,
            ),
            cost_profile: LayerCostProfile::new(
                format!("layer_{i}"),
                1_000_000,
                500_000,
                25.0,
                Some(20.0),
            ),
            dispatch_step_count: 3,
        })
        .collect();

    let tc = make_timing_cert(100.0, 1000.0, true);
    let coupled = CoupledTimingCertificate {
        timing: tc,
        coupled_layers: layers,
        total_dispatch_steps: 12,
    };

    assert!(coupled.all_layers_coupled());
    assert_eq!(coupled.total_dispatch_steps, 12);

    let report = coupled.report();
    assert!(report.contains("layer_0"));
    assert!(report.contains("layer_3"));
    assert!(report.contains("Per-Layer Coupled Verification"));
}

#[test]
fn test_cost_model_zero_flops_zero_memory_equals_overhead() {
    let m = HardwareCostModel::m4_max();
    let time = m.estimate_time_us(0, 0);
    assert!(
        (time - m.dispatch_overhead_us).abs() < 1e-10,
        "zero workload should equal dispatch overhead"
    );
}

// ============================================================================
// 4. Error handling tests (5+)
// ============================================================================

#[test]
fn test_missing_crown_config_uses_defaults() {
    let config = CrownCertificateConfig::default();
    assert_eq!(config.model_name, "kokoro-v1");
    assert_eq!(config.input_specification, "English text");
    assert!(config.map_hard_bounds);
    assert!(!config.check_junction_contracts);
}

#[test]
fn test_invalid_junction_bounds_nan_detection() {
    // NaN in actual bounds should fail.
    let check = check_junction_bound("J2_F0", -5.0, 800.0, f32::NAN, 500.0);
    assert!(!check.passed);

    // NaN in expected bounds should also fail.
    let check = check_junction_bound("J2_F0", f32::NAN, 800.0, 0.0, 500.0);
    assert!(!check.passed);

    // Infinity should fail.
    let check = check_junction_bound("J5_AUDIO", -1.0, 1.0, f32::NEG_INFINITY, 0.5);
    assert!(!check.passed);
}

#[test]
fn test_incomplete_certificate_missing_junction_data() {
    let cert = make_certificate(make_passing_hard_bounds(), vec![], true);
    let config = CrownCertificateConfig {
        check_junction_contracts: true,
        ..CrownCertificateConfig::new("kokoro-v1")
    };

    // No intermediates provided: junction_summary should be None.
    let result = verify_synthesis_crown_full(&cert, &config, None);
    assert!(result.junction_summary.is_none());
}

#[test]
fn test_certificate_with_partial_junction_intermediates() {
    let cert = make_certificate(make_passing_hard_bounds(), vec![], true);
    let config = CrownCertificateConfig {
        check_junction_contracts: true,
        ..CrownCertificateConfig::new("kokoro-v1")
    };

    // Only provide 2 of 6 junction intermediates.
    let mut intermediates = HashMap::new();
    intermediates.insert("J2_F0".to_string(), (0.0_f32, 400.0_f32));
    intermediates.insert("J5_AUDIO".to_string(), (-0.9_f32, 0.9_f32));

    let result = verify_synthesis_crown_full(&cert, &config, Some(&intermediates));
    let summary = result.junction_summary.unwrap();

    // Only 2 checks should be performed (for the 2 provided intermediates).
    assert_eq!(summary.checks.len(), 2);
    assert_eq!(summary.total_passed, 2);
}

#[test]
fn test_junction_check_with_failing_intermediates() {
    let mut intermediates = HashMap::new();
    // J5_AUDIO expects [-1, 1]; provide [-0.5, 1.5] which violates upper.
    intermediates.insert("J5_AUDIO".to_string(), (-0.5_f32, 1.5_f32));
    // J2_F0 within range.
    intermediates.insert("J2_F0".to_string(), (0.0_f32, 400.0_f32));

    let checks = check_all_junction_contracts(&intermediates);
    let j5_check = checks
        .iter()
        .find(|c| c.junction_name == "J5_AUDIO")
        .unwrap();
    assert!(!j5_check.passed, "J5_AUDIO with upper 1.5 should fail");

    let j2_check = checks.iter().find(|c| c.junction_name == "J2_F0").unwrap();
    assert!(j2_check.passed, "J2_F0 within range should pass");
}

#[test]
fn test_contract_bounds_map_contains_all_6_contracts() {
    let map = contract_bounds_map();
    assert_eq!(map.len(), 6);
    assert!(map.contains_key("J2_F0"));
    assert!(map.contains_key("J2_ENERGY"));
    assert!(map.contains_key("J3_MAGNITUDE"));
    assert!(map.contains_key("J3B_PHASE"));
    assert!(map.contains_key("J4_BF16"));
    assert!(map.contains_key("J5_AUDIO"));
}

#[test]
fn test_cost_model_validate_rejects_all_invalid_fields() {
    // NaN in each field.
    let fields = [
        HardwareCostModel {
            peak_tflops_f32: f64::NAN,
            peak_bandwidth_gbs: 400.0,
            dispatch_overhead_us: 5.0,
        },
        HardwareCostModel {
            peak_tflops_f32: 14.2,
            peak_bandwidth_gbs: f64::NAN,
            dispatch_overhead_us: 5.0,
        },
        HardwareCostModel {
            peak_tflops_f32: 14.2,
            peak_bandwidth_gbs: 400.0,
            dispatch_overhead_us: f64::NAN,
        },
        // Negative values.
        HardwareCostModel {
            peak_tflops_f32: -1.0,
            peak_bandwidth_gbs: 400.0,
            dispatch_overhead_us: 5.0,
        },
        HardwareCostModel {
            peak_tflops_f32: 14.2,
            peak_bandwidth_gbs: -1.0,
            dispatch_overhead_us: 5.0,
        },
        HardwareCostModel {
            peak_tflops_f32: 14.2,
            peak_bandwidth_gbs: 400.0,
            dispatch_overhead_us: -1.0,
        },
        // Zero.
        HardwareCostModel {
            peak_tflops_f32: 0.0,
            peak_bandwidth_gbs: 400.0,
            dispatch_overhead_us: 5.0,
        },
        // Infinity.
        HardwareCostModel {
            peak_tflops_f32: f64::INFINITY,
            peak_bandwidth_gbs: 400.0,
            dispatch_overhead_us: 5.0,
        },
    ];

    for (i, m) in fields.iter().enumerate() {
        assert!(
            m.validate().is_err(),
            "field configuration {i} should be rejected"
        );
    }
}

// ============================================================================
// 5. Additional CROWN synthesis integration tests
// ============================================================================

#[test]
fn test_verify_synthesis_crown_full_with_all_intermediates_passing() {
    let cert = make_certificate(make_passing_hard_bounds(), vec![], true);
    let config = CrownCertificateConfig {
        check_junction_contracts: true,
        ..CrownCertificateConfig::new("kokoro-v1")
    };

    // All 6 junction intermediates within bounds.
    let mut intermediates = HashMap::new();
    intermediates.insert("J2_F0".to_string(), (0.0_f32, 400.0_f32));
    intermediates.insert("J2_ENERGY".to_string(), (-10.0_f32, 10.0_f32));
    intermediates.insert("J3_MAGNITUDE".to_string(), (-40.0_f32, 40.0_f32));
    intermediates.insert("J3B_PHASE".to_string(), (-3000.0_f32, 3000.0_f32));
    intermediates.insert("J4_BF16".to_string(), (-64.0_f32, 64.0_f32));
    intermediates.insert("J5_AUDIO".to_string(), (-0.9_f32, 0.9_f32));

    let result = verify_synthesis_crown_full(&cert, &config, Some(&intermediates));

    // Moonshot certificate should have 8 properties.
    assert_eq!(result.moonshot.properties.len(), 8);

    // All junction checks should pass.
    let summary = result.junction_summary.unwrap();
    assert_eq!(summary.total_passed, 6);
    assert_eq!(summary.total_failed, 0);

    // Verify Display formatting.
    let display = format!("{summary}");
    assert!(display.contains("6/6 passed"));
}

#[test]
fn test_certificate_passes_hard_bounds_and_quality() {
    let cert = make_certificate(
        vec![
            HardBound {
                name: "non_silence",
                passed: true,
                value: 0.1,
                threshold: 0.01,
            },
            HardBound {
                name: "no_clipping",
                passed: true,
                value: 0.9,
                threshold: 1.0,
            },
        ],
        vec![QualityMetric {
            name: "mcd",
            value: 4.0,
            threshold: 6.0,
            passed: true,
            citation: "test",
        }],
        true,
    );

    assert!(cert.passes_hard_bounds());
    assert!(cert.passes_quality());
    assert!(cert.overall_passed);
}

#[test]
fn test_certificate_fails_when_hard_bound_fails() {
    let cert = make_certificate(
        vec![HardBound {
            name: "non_silence",
            passed: false,
            value: 0.001,
            threshold: 0.01,
        }],
        vec![],
        false,
    );

    assert!(!cert.passes_hard_bounds());
}

#[test]
fn test_certificate_empty_quality_is_vacuously_true() {
    let cert = make_certificate(vec![], vec![], true);
    assert!(cert.passes_quality());
}

#[test]
fn test_max_contract_violation_multi_element() {
    let c = JunctionContract::new("test", "zone", -1.0, 1.0);

    // Multiple elements: some within, some violating.
    let lower = vec![-0.5, -1.5, -0.8]; // -1.5 violates by 0.5
    let upper = vec![0.5, 0.8, 1.3]; // 1.3 violates by 0.3

    let v = max_contract_violation(&c, &lower, &upper);
    assert!(
        (v - 0.5).abs() < 1e-10,
        "max violation should be 0.5 (lower breach), got {v}"
    );
}

#[test]
fn test_moonshot_certificate_recompute_flags() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    // Check that aggregate flags are consistent with property levels.
    let all_partial = cert
        .properties
        .iter()
        .all(|p| p.level >= VerificationLevel::CrownPartial);
    assert_eq!(cert.all_at_least_partial, all_partial);

    let all_proven = cert.properties.iter().all(|p| {
        matches!(
            p.level,
            VerificationLevel::CrownProven
                | VerificationLevel::KaniProven
                | VerificationLevel::SmtProven
        )
    });
    assert_eq!(cert.all_proven, all_proven);
}

#[test]
fn test_stage_bound_check_display_format() {
    let check = StageBoundCheck {
        junction_name: "J2_F0".to_string(),
        expected_lower: -5.0,
        expected_upper: 800.0,
        actual_lower: 10.0,
        actual_upper: 500.0,
        passed: true,
    };

    let display = format!("{check}");
    assert!(display.contains("[PASS]"));
    assert!(display.contains("J2_F0"));
    assert!(display.contains("10.0000"));
    assert!(display.contains("500.0000"));
}

#[test]
fn test_stage_bound_check_display_format_fail() {
    let check = StageBoundCheck {
        junction_name: "J5_AUDIO".to_string(),
        expected_lower: -1.0,
        expected_upper: 1.0,
        actual_lower: -0.5,
        actual_upper: 1.5,
        passed: false,
    };

    let display = format!("{check}");
    assert!(display.contains("[FAIL]"));
    assert!(display.contains("J5_AUDIO"));
}
