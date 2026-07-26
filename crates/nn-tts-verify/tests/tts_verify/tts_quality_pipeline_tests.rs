// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for TTS quality verification pipeline.
//!
//! Covers cost model validation, pipeline dispatch, junction contract bounds,
//! moonshot property verification, quality metric computation (PESQ/STOI/MCD),
//! and audio format handling (PCM validation).
//!
//! Part of #3942.

use nn_tts_verify::cost_model::{self, HardwareCostModel, LayerCostProfile};
use nn_tts_verify::error::TtsVerifyError;
use nn_tts_verify::kokoro_contracts::{
    all_contracts, bounds_within_contract, contract_stage, max_contract_violation, J2_F0_LOWER,
    J2_F0_UPPER, J5_AUDIO_LOWER, J5_AUDIO_UPPER,
};
use nn_tts_verify::kokoro_dispatch::{
    build_kokoro_dispatch_plan, build_kokoro_dispatch_plan_default,
};
use nn_tts_verify::moonshot::{artifact_registry, MoonshotStatus, VerificationLevel};
use nn_tts_verify::pipeline::{verify_pipeline, VerifiedStage};
use nn_tts_verify::streaming::{crossfade_linear, verify_streaming, StreamingConfig};
use nn_tts_verify::{HardBoundsConfig, QualityConfig, RejectionPolicy, TtsVerifier};

// ---------------------------------------------------------------------------
// Test audio generation helpers
// ---------------------------------------------------------------------------

/// Generate a sine wave at given frequency, sample rate, duration, and amplitude.
fn sine_wave(freq_hz: f64, sample_rate: u32, duration_sec: f64, amplitude: f32) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec).ceil() as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            amplitude * (2.0 * std::f64::consts::PI * freq_hz * t).sin() as f32
        })
        .collect()
}

/// Generate a rich harmonic signal suitable for passing spectral coverage checks.
fn harmonic_signal(sample_rate: u32, duration_sec: f64, amplitude: f32) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec).ceil() as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let pi2 = 2.0 * std::f32::consts::PI;
            let mut s = 0.0_f32;
            for k in 1..=50 {
                let freq = 200.0 * k as f32;
                if freq > sample_rate as f32 / 2.0 {
                    break;
                }
                s += (1.0 / k as f32) * (pi2 * freq * t).sin();
            }
            s * amplitude
        })
        .collect()
}

// ===========================================================================
// Section A: Cost model validation
// ===========================================================================

#[test]
fn test_cost_model_m4_max_fields_finite_positive() {
    let model = HardwareCostModel::m4_max();
    assert!(model.peak_tflops_f32 > 0.0);
    assert!(model.peak_tflops_f32.is_finite());
    assert!(model.peak_bandwidth_gbs > 0.0);
    assert!(model.peak_bandwidth_gbs.is_finite());
    assert!(model.dispatch_overhead_us > 0.0);
    assert!(model.dispatch_overhead_us.is_finite());
    model.validate().expect("M4 Max model should validate");
}

#[test]
fn test_cost_model_m4_max_conservative_fields_finite_positive() {
    let model = HardwareCostModel::m4_max_conservative();
    assert!(model.peak_tflops_f32 > 0.0);
    assert!(model.peak_tflops_f32.is_finite());
    assert!(model.peak_bandwidth_gbs > 0.0);
    assert!(model.peak_bandwidth_gbs.is_finite());
    assert!(model.dispatch_overhead_us > 0.0);
    assert!(model.dispatch_overhead_us.is_finite());
    model
        .validate()
        .expect("M4 Max conservative model should validate");
}

#[test]
fn test_cost_model_conservative_is_slower_than_theoretical() {
    let theoretical = HardwareCostModel::m4_max();
    let conservative = HardwareCostModel::m4_max_conservative();

    // Conservative has lower throughput (slower estimates).
    assert!(conservative.peak_tflops_f32 < theoretical.peak_tflops_f32);
    assert!(conservative.peak_bandwidth_gbs < theoretical.peak_bandwidth_gbs);
    assert!(conservative.dispatch_overhead_us > theoretical.dispatch_overhead_us);

    // Estimate time should be higher for conservative.
    let flops = 1_000_000_u64;
    let bytes = 500_000_u64;
    let t_theoretical = theoretical.estimate_time_us(flops, bytes);
    let t_conservative = conservative.estimate_time_us(flops, bytes);
    assert!(
        t_conservative > t_theoretical,
        "conservative ({t_conservative}) should be slower than theoretical ({t_theoretical})"
    );
}

#[test]
fn test_cost_model_estimate_time_non_negative() {
    let model = HardwareCostModel::m4_max();
    let time = model.estimate_time_us(0, 0);
    assert!(
        time >= 0.0,
        "time estimate should be non-negative, got {time}"
    );
    assert!(time.is_finite(), "time estimate should be finite");

    // Even with zero work, dispatch overhead applies.
    assert!(
        time >= model.dispatch_overhead_us,
        "time ({time}) should be >= dispatch_overhead ({})",
        model.dispatch_overhead_us
    );
}

#[test]
fn test_cost_model_estimate_time_finite_for_large_inputs() {
    let model = HardwareCostModel::m4_max();
    let time = model.estimate_time_us(u64::MAX / 2, u64::MAX / 2);
    assert!(
        time.is_finite(),
        "time estimate should be finite even for large inputs"
    );
    assert!(time > 0.0, "time for large inputs should be positive");
}

#[test]
fn test_cost_model_validate_rejects_zero_tflops() {
    let model = HardwareCostModel {
        peak_tflops_f32: 0.0,
        peak_bandwidth_gbs: 400.0,
        dispatch_overhead_us: 5.0,
    };
    assert!(
        model.validate().is_err(),
        "zero tflops should fail validation"
    );
}

#[test]
fn test_cost_model_validate_rejects_nan_bandwidth() {
    let model = HardwareCostModel {
        peak_tflops_f32: 14.2,
        peak_bandwidth_gbs: f64::NAN,
        dispatch_overhead_us: 5.0,
    };
    assert!(
        model.validate().is_err(),
        "NaN bandwidth should fail validation"
    );
}

#[test]
fn test_cost_model_validate_rejects_negative_overhead() {
    let model = HardwareCostModel {
        peak_tflops_f32: 14.2,
        peak_bandwidth_gbs: 400.0,
        dispatch_overhead_us: -1.0,
    };
    assert!(
        model.validate().is_err(),
        "negative overhead should fail validation"
    );
}

#[test]
fn test_cost_model_validate_rejects_inf() {
    let model = HardwareCostModel {
        peak_tflops_f32: f64::INFINITY,
        peak_bandwidth_gbs: 400.0,
        dispatch_overhead_us: 5.0,
    };
    assert!(
        model.validate().is_err(),
        "infinite tflops should fail validation"
    );
}

#[test]
fn test_layer_cost_profile_construction() {
    let profile = LayerCostProfile::new("conv1d", 1000, 2000, 5.0, Some(4.5));
    assert_eq!(profile.layer_name, "conv1d");
    assert_eq!(profile.flops, 1000);
    assert_eq!(profile.memory_bytes, 2000);
    assert!((profile.estimated_time_us - 5.0).abs() < 1e-12);
    assert_eq!(profile.measured_time_us, Some(4.5));
}

#[test]
fn test_total_estimated_time_sums_correctly() {
    let profiles = vec![
        LayerCostProfile::new("a", 100, 200, 10.0, None),
        LayerCostProfile::new("b", 300, 400, 20.0, None),
        LayerCostProfile::new("c", 500, 600, 30.0, None),
    ];
    let total = cost_model::total_estimated_time_us(&profiles);
    assert!((total - 60.0).abs() < 1e-12, "expected 60.0, got {total}");
}

#[test]
fn test_total_flops_sums_correctly() {
    let profiles = vec![
        LayerCostProfile::new("a", 100, 200, 10.0, None),
        LayerCostProfile::new("b", 300, 400, 20.0, None),
    ];
    assert_eq!(cost_model::total_flops(&profiles), 400);
}

#[test]
fn test_total_memory_bytes_sums_correctly() {
    let profiles = vec![
        LayerCostProfile::new("a", 100, 200, 10.0, None),
        LayerCostProfile::new("b", 300, 400, 20.0, None),
    ];
    assert_eq!(cost_model::total_memory_bytes(&profiles), 600);
}

// ===========================================================================
// Section B: Pipeline dispatch — Kokoro dispatch plan
// ===========================================================================

#[test]
fn test_kokoro_dispatch_plan_default_produces_steps() {
    let (plan, t_final) = build_kokoro_dispatch_plan_default();
    assert!(
        !plan.is_empty(),
        "default Kokoro dispatch plan should have steps"
    );
    // 100 tokens * 10 * 6 = 6000 frames.
    assert_eq!(t_final, 6000, "100 tokens should produce 6000 frames");
}

#[test]
fn test_kokoro_dispatch_plan_step_count_reasonable() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
    // The Kokoro vocoder has many steps (conv, activations, resblocks).
    // Expect at least 50 steps.
    assert!(
        plan.len() >= 50,
        "Kokoro plan should have >= 50 steps, got {}",
        plan.len()
    );
}

#[test]
fn test_kokoro_dispatch_plan_temporal_scaling() {
    let (_, t50) = build_kokoro_dispatch_plan(50);
    let (_, t100) = build_kokoro_dispatch_plan(100);
    let (_, t200) = build_kokoro_dispatch_plan(200);

    // Temporal dim scales linearly with input tokens.
    assert_eq!(t50 * 2, t100, "temporal dim should scale linearly");
    assert_eq!(t100 * 2, t200, "temporal dim should scale linearly");
}

#[test]
fn test_kokoro_dispatch_plan_profiling_produces_non_negative_times() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
    let model = HardwareCostModel::m4_max();
    let profiles = cost_model::profile_dispatch_plan(&plan, &model);

    assert_eq!(profiles.len(), plan.len());
    for p in &profiles {
        assert!(
            p.estimated_time_us >= 0.0,
            "layer {} has negative time: {}",
            p.layer_name,
            p.estimated_time_us
        );
        assert!(
            p.estimated_time_us.is_finite(),
            "layer {} has non-finite time",
            p.layer_name
        );
    }

    let total_time = cost_model::total_estimated_time_us(&profiles);
    assert!(total_time > 0.0, "total time should be positive");
    assert!(total_time.is_finite(), "total time should be finite");
}

// ===========================================================================
// Section C: Junction contract bounds (J2-J5)
// ===========================================================================

#[test]
fn test_all_contract_bounds_are_finite_and_ordered() {
    for c in &all_contracts() {
        assert!(c.lower.is_finite(), "{}: lower must be finite", c.name);
        assert!(c.upper.is_finite(), "{}: upper must be finite", c.name);
        assert!(
            c.lower < c.upper,
            "{}: lower ({}) must be < upper ({})",
            c.name,
            c.lower,
            c.upper
        );
    }
}

#[test]
fn test_j5_audio_contract_pcm_range() {
    // PCM audio must be in [-1, 1].
    assert_eq!(J5_AUDIO_LOWER, -1.0);
    assert_eq!(J5_AUDIO_UPPER, 1.0);
}

#[test]
fn test_bounds_within_contract_empty_bounds() {
    let j5 = &all_contracts()[5];
    // Empty bounds vectors: vacuously true.
    assert!(bounds_within_contract(j5, &[], &[]));
}

#[test]
fn test_bounds_within_contract_inf_rejected() {
    let j5 = &all_contracts()[5];
    assert!(!bounds_within_contract(j5, &[f64::NEG_INFINITY], &[0.5]));
    assert!(!bounds_within_contract(j5, &[-0.5], &[f64::INFINITY]));
}

#[test]
fn test_max_contract_violation_empty_bounds() {
    let j5 = &all_contracts()[5];
    let v = max_contract_violation(j5, &[], &[]);
    assert_eq!(v, 0.0, "empty bounds should have zero violation");
}

#[test]
fn test_max_contract_violation_inf_returns_max() {
    let j5 = &all_contracts()[5];
    let v = max_contract_violation(j5, &[f64::INFINITY], &[0.5]);
    assert_eq!(v, f64::MAX);
}

#[test]
fn test_contract_stage_creates_uniform_bounds() {
    let contracts = all_contracts();
    let j2_f0 = &contracts[0];
    let j5 = &contracts[5];
    let stage = contract_stage("test_stage", &[1, 10], &[1, 20], j2_f0, j5, "IBP", false);

    // Input elements = 1*10 = 10, all uniform.
    assert_eq!(stage.input_lower.len(), 10);
    assert_eq!(stage.input_upper.len(), 10);
    assert!(stage.input_lower.iter().all(|&v| v == J2_F0_LOWER));
    assert!(stage.input_upper.iter().all(|&v| v == J2_F0_UPPER));

    // Output elements = 1*20 = 20, all uniform.
    assert_eq!(stage.output_lower.len(), 20);
    assert_eq!(stage.output_upper.len(), 20);
    assert!(stage.output_lower.iter().all(|&v| v == J5_AUDIO_LOWER));
    assert!(stage.output_upper.iter().all(|&v| v == J5_AUDIO_UPPER));

    assert_eq!(stage.method, "IBP");
    assert!(!stage.is_sound);
}

// ===========================================================================
// Section D: Pipeline composition verification
// ===========================================================================

#[test]
fn test_verify_pipeline_requires_at_least_two_stages() {
    let stage = VerifiedStage::new(
        "solo",
        vec![4],
        vec![4],
        vec![-1.0; 4],
        vec![1.0; 4],
        vec![-0.5; 4],
        vec![0.5; 4],
        "CROWN",
        true,
    );
    let result = verify_pipeline(&[stage]);
    assert!(result.is_err(), "pipeline with 1 stage should fail");
}

#[test]
fn test_verify_pipeline_compatible_stages_valid() {
    let stage_a = VerifiedStage::new(
        "encoder",
        vec![8],
        vec![8],
        vec![-1.0; 8],
        vec![1.0; 8],
        vec![-0.5; 8],
        vec![0.5; 8],
        "CROWN",
        true,
    );
    let stage_b = VerifiedStage::new(
        "decoder",
        vec![8],
        vec![8],
        vec![-0.5; 8],
        vec![0.5; 8],
        vec![-0.3; 8],
        vec![0.3; 8],
        "CROWN",
        true,
    );
    let cert = verify_pipeline(&[stage_a, stage_b]).expect("valid pipeline");
    assert!(cert.is_valid);
    assert!(cert.is_sound);
    assert_eq!(cert.junctions.len(), 1);
    assert!(cert.junctions[0].bounds_contained);
    assert_eq!(cert.junctions[0].max_violation, 0.0);
}

#[test]
fn test_verify_pipeline_detects_bound_violation() {
    // stage_a outputs [-2, 2], stage_b expects input [-1, 1].
    let stage_a = VerifiedStage::new(
        "wide_output",
        vec![4],
        vec![4],
        vec![-1.0; 4],
        vec![1.0; 4],
        vec![-2.0; 4],
        vec![2.0; 4],
        "CROWN",
        true,
    );
    let stage_b = VerifiedStage::new(
        "narrow_input",
        vec![4],
        vec![4],
        vec![-1.0; 4],
        vec![1.0; 4],
        vec![-0.5; 4],
        vec![0.5; 4],
        "CROWN",
        true,
    );
    let cert = verify_pipeline(&[stage_a, stage_b]).expect("computed pipeline");
    assert!(!cert.is_valid);
    assert!(!cert.junctions[0].bounds_contained);
    assert!(cert.junctions[0].max_violation > 0.0);
    // Violation = 2.0 - 1.0 = 1.0.
    assert!(
        (cert.junctions[0].max_violation - 1.0).abs() < 1e-10,
        "expected violation 1.0, got {}",
        cert.junctions[0].max_violation
    );
}

#[test]
fn test_verify_pipeline_shape_mismatch_detected() {
    let stage_a = VerifiedStage::new(
        "output_4",
        vec![4],
        vec![4],
        vec![-1.0; 4],
        vec![1.0; 4],
        vec![-0.5; 4],
        vec![0.5; 4],
        "CROWN",
        true,
    );
    let stage_b = VerifiedStage::new(
        "input_8",
        vec![8],
        vec![8],
        vec![-1.0; 8],
        vec![1.0; 8],
        vec![-0.5; 8],
        vec![0.5; 8],
        "CROWN",
        true,
    );
    let cert = verify_pipeline(&[stage_a, stage_b]).expect("computed pipeline");
    assert!(!cert.junctions[0].shape_compatible);
    assert!(!cert.is_valid);
}

#[test]
fn test_verify_pipeline_nan_in_bounds_is_violation() {
    let stage_a = VerifiedStage::new(
        "nan_output",
        vec![2],
        vec![2],
        vec![-1.0; 2],
        vec![1.0; 2],
        vec![f64::NAN, -0.5],
        vec![0.5, 0.5],
        "CROWN",
        true,
    );
    let stage_b = VerifiedStage::new(
        "normal_input",
        vec![2],
        vec![2],
        vec![-1.0; 2],
        vec![1.0; 2],
        vec![-0.5; 2],
        vec![0.5; 2],
        "CROWN",
        true,
    );
    let cert = verify_pipeline(&[stage_a, stage_b]).expect("computed pipeline");
    assert!(!cert.junctions[0].bounds_contained);
    assert_eq!(cert.junctions[0].max_violation, f64::MAX);
}

#[test]
fn test_verify_pipeline_soundness_propagation() {
    let sound = VerifiedStage::new(
        "sound",
        vec![4],
        vec![4],
        vec![-1.0; 4],
        vec![1.0; 4],
        vec![-0.5; 4],
        vec![0.5; 4],
        "CROWN",
        true,
    );
    let unsound = VerifiedStage::new(
        "unsound",
        vec![4],
        vec![4],
        vec![-0.5; 4],
        vec![0.5; 4],
        vec![-0.3; 4],
        vec![0.3; 4],
        "IBP",
        false,
    );
    let cert = verify_pipeline(&[sound, unsound]).expect("valid pipeline");
    assert!(cert.is_valid);
    assert!(
        !cert.is_sound,
        "pipeline with unsound stage should be unsound"
    );
}

#[test]
fn test_pipeline_certificate_report_contains_stage_info() {
    let stage_a = VerifiedStage::new(
        "alpha",
        vec![4],
        vec![4],
        vec![-1.0; 4],
        vec![1.0; 4],
        vec![-0.5; 4],
        vec![0.5; 4],
        "CROWN",
        true,
    );
    let stage_b = VerifiedStage::new(
        "beta",
        vec![4],
        vec![4],
        vec![-0.5; 4],
        vec![0.5; 4],
        vec![-0.3; 4],
        vec![0.3; 4],
        "CROWN",
        true,
    );
    let cert = verify_pipeline(&[stage_a, stage_b]).unwrap();
    let report = cert.report();
    assert!(
        report.contains("alpha"),
        "report should mention stage alpha"
    );
    assert!(report.contains("beta"), "report should mention stage beta");
    assert!(report.contains("Valid: true"));
    assert!(report.contains("Sound: true"));
}

// ===========================================================================
// Section E: Moonshot property checks (P1-P8)
// ===========================================================================

#[test]
fn test_moonshot_status_has_8_properties() {
    let status = MoonshotStatus::from_repo();
    assert_eq!(status.properties.len(), 8);
}

#[test]
fn test_moonshot_all_property_names_non_empty() {
    let status = MoonshotStatus::from_repo();
    for (i, prop) in status.properties.iter().enumerate() {
        assert!(
            !prop.name.is_empty(),
            "Property {} should have a non-empty name",
            i + 1
        );
    }
}

#[test]
fn test_moonshot_verification_level_display() {
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
fn test_moonshot_verification_level_strict_ordering() {
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
fn test_moonshot_level_counts_sum_to_8() {
    let status = MoonshotStatus::from_repo();
    let counts = status.level_counts();
    let total: usize = counts.iter().map(|(_, c)| c).sum();
    assert_eq!(total, 8, "Level counts should sum to 8");
}

#[test]
fn test_moonshot_all_have_evidence() {
    let status = MoonshotStatus::from_repo();
    assert!(
        status.all_have_evidence(),
        "all 8 properties should have verification evidence"
    );
}

#[test]
fn test_artifact_registry_non_empty() {
    let artifacts = artifact_registry();
    assert!(
        !artifacts.is_empty(),
        "artifact registry should not be empty"
    );
    for artifact in &artifacts {
        assert!(!artifact.description.is_empty());
        assert!(!artifact.file.is_empty());
        // All property indices should be < 8.
        for &idx in artifact.properties {
            assert!(idx < 8, "property index {idx} out of range");
        }
    }
}

#[test]
fn test_artifact_registry_covers_every_property() {
    let artifacts = artifact_registry();
    let mut covered = [false; 8];
    for artifact in &artifacts {
        for &idx in artifact.properties {
            if idx < 8 {
                covered[idx] = true;
            }
        }
    }
    for (i, &c) in covered.iter().enumerate() {
        assert!(
            c,
            "Property {} should be covered by at least one artifact",
            i + 1
        );
    }
}

// ===========================================================================
// Section F: Quality metric computation (PESQ/STOI/MCD)
// ===========================================================================

#[test]
fn test_pesq_identical_signals_high_score() {
    // PESQ of identical signals should be near maximum (4.5).
    let signal = harmonic_signal(16000, 1.0, 0.3);
    let result = nn_tts_verify::compute_pesq(&signal, &signal, 16000, 1.0)
        .expect("PESQ should succeed for valid inputs");
    assert!(
        result.value > 3.0,
        "PESQ of identical signals should be high, got {}",
        result.value
    );
    assert!(result.passed, "PESQ should pass with threshold 1.0");
    assert!(result.value.is_finite(), "PESQ value should be finite");
    assert_eq!(result.name, "pesq");
}

#[test]
fn test_pesq_different_signals_lower_score() {
    let a = sine_wave(200.0, 16000, 1.0, 0.3);
    let b = sine_wave(800.0, 16000, 1.0, 0.3);
    let result = nn_tts_verify::compute_pesq(&a, &b, 16000, 1.0).expect("PESQ should succeed");
    // Different signals should produce lower (but valid) PESQ.
    assert!(result.value.is_finite(), "PESQ value should be finite");
    assert!(result.value <= 4.5, "PESQ should be at most 4.5");
    assert!(result.value >= -0.5, "PESQ should be at least -0.5");
}

#[test]
fn test_pesq_empty_signals_error() {
    let result = nn_tts_verify::compute_pesq(&[], &[], 16000, 1.0);
    assert!(result.is_err(), "PESQ should fail for empty signals");
}

#[test]
fn test_pesq_length_mismatch_error() {
    let a = sine_wave(200.0, 16000, 1.0, 0.3);
    let b = sine_wave(200.0, 16000, 2.0, 0.3);
    let result = nn_tts_verify::compute_pesq(&a, &b, 16000, 1.0);
    assert!(result.is_err(), "PESQ should fail for mismatched lengths");
}

#[test]
fn test_pesq_zero_sample_rate_error() {
    let a = sine_wave(200.0, 16000, 1.0, 0.3);
    let result = nn_tts_verify::compute_pesq(&a, &a, 0, 1.0);
    assert!(result.is_err(), "PESQ should fail for zero sample rate");
}

#[test]
fn test_stoi_identical_signals_high_score() {
    // STOI of identical signals should be near 1.0.
    let signal = harmonic_signal(16000, 2.0, 0.3);
    let result =
        nn_tts_verify::compute_stoi(&signal, &signal, 16000, 0.5).expect("STOI should succeed");
    assert!(
        result.value > 0.8,
        "STOI of identical signals should be high, got {}",
        result.value
    );
    assert!(result.passed);
    assert!(result.value >= 0.0 && result.value <= 1.0);
    assert_eq!(result.name, "stoi");
}

#[test]
fn test_stoi_empty_signals_error() {
    let result = nn_tts_verify::compute_stoi(&[], &[], 16000, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_stoi_length_mismatch_error() {
    let a = sine_wave(200.0, 16000, 2.0, 0.3);
    let b = sine_wave(200.0, 16000, 3.0, 0.3);
    let result = nn_tts_verify::compute_stoi(&a, &b, 16000, 0.5);
    assert!(result.is_err(), "STOI should fail for mismatched lengths");
}

#[test]
fn test_mcd_identical_signals_near_zero() {
    let signal = harmonic_signal(16000, 1.0, 0.3);
    let result = nn_tts_verify::quality::compute_mcd(&signal, &signal, 16000, 6.0)
        .expect("MCD should succeed");
    assert!(
        result.value < 1.0,
        "MCD of identical signals should be near 0, got {}",
        result.value
    );
    assert!(result.passed);
    assert!(result.value.is_finite());
}

#[test]
fn test_mcd_different_signals_measurable() {
    let a = sine_wave(200.0, 16000, 0.5, 0.3);
    let b = sine_wave(800.0, 16000, 0.5, 0.3);
    let result =
        nn_tts_verify::quality::compute_mcd(&a, &b, 16000, 20.0).expect("MCD should succeed");
    assert!(result.value > 0.0, "MCD of different signals should be > 0");
    assert!(result.value.is_finite());
}

#[test]
fn test_rms_metric_non_negative() {
    let signal = sine_wave(440.0, 16000, 0.5, 0.5);
    let result = nn_tts_verify::quality::compute_rms(&signal, 0.01).expect("RMS should succeed");
    assert!(result.value >= 0.0, "RMS should be non-negative");
    assert!(result.value.is_finite());
    assert!(result.passed, "tone should pass min_rms=0.01");
}

#[test]
fn test_rms_metric_empty_error() {
    let result = nn_tts_verify::quality::compute_rms(&[], 0.01);
    assert!(result.is_err());
}

// ===========================================================================
// Section G: Audio format handling (PCM validation)
// ===========================================================================

#[test]
fn test_verifier_builder_default_builds() {
    let verifier = TtsVerifier::builder()
        .build()
        .expect("builder should succeed with defaults");
    // Verify default builder produces a valid verifier by running it on valid audio.
    let signal = harmonic_signal(24000, 0.5, 0.15);
    let cert = verifier
        .verify(&signal)
        .expect("valid audio should pass default verifier");
    assert!(cert.overall_passed);
}

#[test]
fn test_verifier_builder_zero_sample_rate_rejected() {
    let result = TtsVerifier::builder().sample_rate(0).build();
    assert!(result.is_err(), "zero sample rate should be rejected");
}

#[test]
fn test_verifier_rejects_empty_input() {
    let verifier = TtsVerifier::builder().build().unwrap();
    let result = verifier.verify(&[]);
    assert!(result.is_err(), "empty input should be rejected");
}

#[test]
fn test_verifier_rejects_nan_samples() {
    let verifier = TtsVerifier::builder().build().unwrap();
    let samples = vec![0.1, f32::NAN, 0.2];
    let result = verifier.verify(&samples);
    assert!(result.is_err(), "NaN samples should be rejected");
    match result.unwrap_err() {
        TtsVerifyError::NonFiniteInput { count } => {
            assert_eq!(count, 1, "should detect 1 NaN");
        }
        other => panic!("expected NonFiniteInput, got {other:?}"),
    }
}

#[test]
fn test_verifier_rejects_inf_samples() {
    let verifier = TtsVerifier::builder().build().unwrap();
    let samples = vec![0.1, f32::INFINITY, 0.2, f32::NEG_INFINITY];
    let result = verifier.verify(&samples);
    assert!(result.is_err(), "Inf samples should be rejected");
    match result.unwrap_err() {
        TtsVerifyError::NonFiniteInput { count } => {
            assert_eq!(count, 2, "should detect 2 infinite values");
        }
        other => panic!("expected NonFiniteInput, got {other:?}"),
    }
}

#[test]
fn test_verifier_valid_audio_passes() {
    let verifier = TtsVerifier::builder().sample_rate(24000).build().unwrap();
    let signal = harmonic_signal(24000, 1.0, 0.15);
    let cert = verifier.verify(&signal).expect("valid audio should pass");
    assert!(cert.overall_passed);
    assert!(cert.passes_hard_bounds());
    assert!(cert.deterministic_hash.is_some());
}

#[test]
fn test_verifier_with_reference_length_mismatch_error() {
    let verifier = TtsVerifier::builder().sample_rate(24000).build().unwrap();
    let a = harmonic_signal(24000, 1.0, 0.15);
    let b = harmonic_signal(24000, 2.0, 0.15);
    let result = verifier.verify_with_reference(&a, &b);
    assert!(result.is_err(), "mismatched lengths should fail");
}

#[test]
fn test_certificate_report_format() {
    let verifier = TtsVerifier::builder().sample_rate(24000).build().unwrap();
    let signal = harmonic_signal(24000, 1.0, 0.15);
    let cert = verifier.verify(&signal).unwrap();
    let report = cert.report();
    assert!(report.contains("TTS Verification Certificate"));
    assert!(report.contains("Hard Bounds"));
    assert!(report.contains("non_silence"));
    assert!(report.contains("no_clipping"));
    assert!(report.contains("Overall: PASSED"));
}

#[test]
fn test_certificate_hard_bounds_include_all_checks() {
    let verifier = TtsVerifier::builder().sample_rate(24000).build().unwrap();
    let signal = harmonic_signal(24000, 1.0, 0.15);
    let cert = verifier.verify(&signal).unwrap();
    let names: Vec<&str> = cert.hard_bounds.iter().map(|b| b.name).collect();
    assert!(names.contains(&"non_silence"));
    assert!(names.contains(&"no_clipping"));
    assert!(names.contains(&"no_dc_offset"));
    assert!(names.contains(&"no_clicks"));
    assert!(names.contains(&"duration"));
    assert!(names.contains(&"tail_energy"));
    assert!(names.contains(&"spectral_coverage"));
    assert!(names.contains(&"nyquist"));
}

#[test]
fn test_hard_bound_values_are_finite() {
    let verifier = TtsVerifier::builder().sample_rate(24000).build().unwrap();
    let signal = harmonic_signal(24000, 1.0, 0.15);
    let cert = verifier.verify(&signal).unwrap();
    for b in &cert.hard_bounds {
        assert!(b.value.is_finite(), "{}: value should be finite", b.name);
        assert!(
            b.threshold.is_finite(),
            "{}: threshold should be finite",
            b.name
        );
    }
}

// ===========================================================================
// Section H: Streaming verification
// ===========================================================================

#[test]
fn test_crossfade_linear_equal_length() {
    let tail = vec![1.0_f32; 10];
    let head = vec![0.0_f32; 10];
    let blended = crossfade_linear(&tail, &head).unwrap();
    assert_eq!(blended.len(), 10);
    // First sample: alpha=0 -> tail[0]*1.0 + head[0]*0.0 = 1.0.
    assert!((blended[0] - 1.0).abs() < 1e-6);
    // Last sample: alpha=1 -> tail[9]*0.0 + head[9]*1.0 = 0.0.
    assert!((blended[9] - 0.0).abs() < 1e-6);
    // Middle sample: alpha=0.5 -> 0.5.
    let mid = blended.len() / 2;
    assert!(
        (blended[mid] - 0.5).abs() < 0.15,
        "midpoint should be near 0.5, got {}",
        blended[mid]
    );
}

#[test]
fn test_crossfade_linear_mismatched_length_error() {
    let result = crossfade_linear(&[1.0; 5], &[0.0; 10]);
    assert!(result.is_err(), "mismatched lengths should fail");
}

#[test]
fn test_streaming_config_default_valid() {
    let config = StreamingConfig::default();
    config.validate().expect("default config should be valid");
    assert_eq!(config.sample_rate, 24000);
    assert_eq!(config.crossfade_samples, 960);
}

#[test]
fn test_streaming_config_zero_sample_rate_rejected() {
    let mut config = StreamingConfig::default();
    config.sample_rate = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_streaming_config_margin_less_than_crossfade_rejected() {
    let mut config = StreamingConfig::default();
    config.margin_samples = 100;
    config.crossfade_samples = 200;
    assert!(config.validate().is_err());
}

#[test]
fn test_verify_streaming_less_than_2_chunks_error() {
    let config = StreamingConfig::default();
    let chunk = harmonic_signal(24000, 0.1, 0.3);
    let result = verify_streaming(&[&chunk], &config);
    assert!(
        result.is_err(),
        "1 chunk should fail streaming verification"
    );
}

#[test]
fn test_verify_streaming_two_identical_chunks_passes() {
    let config = StreamingConfig::default();
    // Create chunks long enough for margin analysis.
    //
    // Use a band-limited harmonic chunk (fundamental + low harmonics, all
    // periodic over the 0.2 s window). The general `harmonic_signal` helper
    // packs energy up to ~10 kHz, where a single-sample step at near-Nyquist
    // frequencies already exceeds the default 0.3 click threshold regardless
    // of how smooth the seam is — so it can never pass the click check. A
    // band-limited variant has small adjacent-sample steps, so two identical
    // copies genuinely join smoothly at the boundary.
    let sample_rate = 24000u32;
    let n = (f64::from(sample_rate) * 0.2).ceil() as usize;
    let chunk: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let pi2 = 2.0 * std::f32::consts::PI;
            let mut s = 0.0_f32;
            for k in 1..=10 {
                s += (1.0 / k as f32) * (pi2 * 200.0 * k as f32 * t).sin();
            }
            s * 0.3
        })
        .collect();
    let cert =
        verify_streaming(&[&chunk, &chunk], &config).expect("two identical chunks should succeed");
    assert_eq!(cert.n_chunks, 2);
    assert_eq!(cert.boundaries.len(), 1);
    // Identical chunks at boundary should be smooth.
    assert!(
        cert.overall_passed,
        "identical chunks should pass streaming verification"
    );
}

// ===========================================================================
// Section I: Config validation
// ===========================================================================

#[test]
fn test_hard_bounds_config_default_validates() {
    let config = HardBoundsConfig::default();
    config
        .validate()
        .expect("default hard bounds config should validate");
}

#[test]
fn test_quality_config_default_validates() {
    let config = QualityConfig::default();
    config
        .validate()
        .expect("default quality config should validate");
}

#[test]
fn test_hard_bounds_config_rejects_nan_min_rms() {
    let mut config = HardBoundsConfig::default();
    config.min_rms = f64::NAN;
    assert!(config.validate().is_err());
}

#[test]
fn test_hard_bounds_config_rejects_inverted_duration_range() {
    let mut config = HardBoundsConfig::default();
    config.min_duration_sec = 100.0;
    config.max_duration_sec = 10.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_quality_config_rejects_inverted_f0_range() {
    let mut config = QualityConfig::default();
    config.f0_range = (400.0, 80.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_rejection_policy_default_is_reject() {
    assert_eq!(RejectionPolicy::default(), RejectionPolicy::Reject);
}
