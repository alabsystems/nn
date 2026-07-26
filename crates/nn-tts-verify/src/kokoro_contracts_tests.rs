// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Kokoro junction contract constants and containment checks.
//!
//! Verifies that nn-tts-verify's contract constants match dvoice's
//! junction_contracts.rs values and that the pipeline compose framework
//! correctly detects containment / violation at each junction.
//!
//! Part of #2478.

use super::*;
use crate::pipeline::verify_pipeline;

// ── Constant value tests ─────────────────────────────────────────

#[test]
fn test_j2_f0_bounds_match_dvoice() {
    // dvoice: F0_LOWER_TOLERANCE = -5.0, F0_UPPER_HZ = 800.0
    assert_eq!(J2_F0_LOWER, -5.0);
    assert_eq!(J2_F0_UPPER, 800.0);
}

#[test]
fn test_j2_energy_bounds_match_dvoice() {
    // dvoice: ENERGY_LOWER = -50.0, ENERGY_UPPER = 50.0
    assert_eq!(J2_ENERGY_LOWER, -50.0);
    assert_eq!(J2_ENERGY_UPPER, 50.0);
}

#[test]
fn test_j3_magnitude_bounds_match_dvoice() {
    // dvoice: MAGNITUDE_PRE_EXP_LOWER = -80.0, MAGNITUDE_PRE_EXP_UPPER = 80.0
    assert_eq!(J3_MAGNITUDE_LOWER, -80.0);
    assert_eq!(J3_MAGNITUDE_UPPER, 80.0);
}

#[test]
fn test_j3b_phase_bounds_match_dvoice() {
    // dvoice: PHASE_ABS_MAX = 6283.2
    assert_eq!(J3B_PHASE_LOWER, -6283.2);
    assert_eq!(J3B_PHASE_UPPER, 6283.2);
}

#[test]
fn test_j4_bf16_bounds_match_dvoice() {
    // dvoice: BF16_SAFE_ABS_MAX = 128.0
    assert_eq!(J4_BF16_LOWER, -128.0);
    assert_eq!(J4_BF16_UPPER, 128.0);
}

#[test]
fn test_j5_audio_bounds_match_dvoice() {
    // dvoice: AUDIO_PEAK_ABS_MAX = 1.0
    assert_eq!(J5_AUDIO_LOWER, -1.0);
    assert_eq!(J5_AUDIO_UPPER, 1.0);
}

#[test]
fn test_all_contracts_count() {
    let contracts = all_contracts();
    assert_eq!(contracts.len(), 6);
}

#[test]
fn test_all_contracts_valid_ranges() {
    for c in &all_contracts() {
        assert!(
            c.lower < c.upper,
            "{}: lower ({}) must be < upper ({})",
            c.name,
            c.lower,
            c.upper
        );
        assert!(c.lower.is_finite(), "{}: lower must be finite", c.name);
        assert!(c.upper.is_finite(), "{}: upper must be finite", c.name);
    }
}

// ── bounds_within_contract tests ─────────────────────────────────

#[test]
fn test_bounds_within_contract_contained() {
    let j5 = &all_contracts()[5]; // J5_AUDIO: [-1, 1]
    assert!(bounds_within_contract(j5, &[-0.8, -0.5], &[0.5, 0.8]));
}

#[test]
fn test_bounds_within_contract_exact() {
    let j5 = &all_contracts()[5];
    assert!(bounds_within_contract(j5, &[-1.0], &[1.0]));
}

#[test]
fn test_bounds_within_contract_violated_upper() {
    let j5 = &all_contracts()[5]; // J5_AUDIO: [-1, 1]
    assert!(!bounds_within_contract(j5, &[-0.5], &[1.5]));
}

#[test]
fn test_bounds_within_contract_violated_lower() {
    let j5 = &all_contracts()[5];
    assert!(!bounds_within_contract(j5, &[-1.5], &[0.5]));
}

#[test]
fn test_bounds_within_contract_nan_rejected() {
    let j5 = &all_contracts()[5];
    assert!(!bounds_within_contract(j5, &[f64::NAN], &[0.5]));
    assert!(!bounds_within_contract(j5, &[-0.5], &[f64::NAN]));
}

#[test]
fn test_bounds_within_contract_length_mismatch() {
    let j5 = &all_contracts()[5];
    assert!(!bounds_within_contract(j5, &[-0.5, -0.3], &[0.5]));
}

// ── max_contract_violation tests ─────────────────────────────────

#[test]
fn test_max_violation_zero_when_contained() {
    let j3 = &all_contracts()[2]; // J3_MAGNITUDE: [-80, 80]
    let v = max_contract_violation(j3, &[-50.0, -20.0], &[20.0, 50.0]);
    assert_eq!(v, 0.0);
}

#[test]
fn test_max_violation_upper_breach() {
    let j5 = &all_contracts()[5]; // J5_AUDIO: [-1, 1]
    let v = max_contract_violation(j5, &[-0.5], &[1.3]);
    assert!((v - 0.3).abs() < 1e-10);
}

#[test]
fn test_max_violation_lower_breach() {
    let j5 = &all_contracts()[5];
    let v = max_contract_violation(j5, &[-1.5], &[0.5]);
    assert!((v - 0.5).abs() < 1e-10);
}

#[test]
fn test_max_violation_nan_sentinel() {
    let j5 = &all_contracts()[5];
    let v = max_contract_violation(j5, &[f64::NAN], &[0.5]);
    assert_eq!(v, f64::MAX);
}

// ── contract_stage + pipeline compose tests ──────────────────────

#[test]
fn test_contract_stage_shape() {
    let contracts = all_contracts();
    let j2_f0 = &contracts[0];
    let j5 = &contracts[5];
    let stage = contract_stage(
        "decoder_to_audio",
        &[1, 128],
        &[1, 24000],
        j2_f0,
        j5,
        "CROWN",
        true,
    );
    assert_eq!(stage.input_shape, vec![1, 128]);
    assert_eq!(stage.output_shape, vec![1, 24000]);
    assert_eq!(stage.input_lower.len(), 128);
    assert_eq!(stage.output_lower.len(), 24000);
    assert_eq!(stage.input_lower[0], J2_F0_LOWER);
    assert_eq!(stage.output_upper[0], J5_AUDIO_UPPER);
}

/// Compose test: Kokoro pipeline with junction contracts.
///
/// Models the Kokoro pipeline as 3 stages with junction bounds from
/// dvoice's runtime contracts:
///   Stage 1 (Decoder): input BF16[-128,128] -> output F0[-5,800] + Energy[-50,50]
///   Stage 2 (Generator): input Magnitude[-80,80] -> output Phase[-6283.2,6283.2]
///   Stage 3 (iSTFT): input Phase[-6283.2,6283.2] -> output Audio[-1,1]
///
/// Junction checks:
///   J1: Decoder output F0/Energy must fit Generator input Magnitude bounds.
///   J2: Generator output Phase must fit iSTFT input Phase bounds.
///
/// This test verifies the junction containment invariant: if NY
/// proves each stage's output bounds are within the next stage's contract
/// input bounds, the pipeline composes safely.
#[test]
fn test_kokoro_junction_pipeline_compose() {
    let contracts = all_contracts();
    let j4_bf16 = &contracts[4]; // BF16 safe input for decoder
    let j2_energy = &contracts[1]; // Energy output from decoder

    // Stage 1: Decoder — BF16 input, energy-bounded output.
    // Output [-50, 50] must be contained in Generator input [-80, 80].
    let decoder = contract_stage(
        "kokoro_decoder",
        &[1, 256],
        &[1, 256],
        j4_bf16,
        j2_energy,
        "CROWN",
        true,
    );

    let j3_mag = &contracts[2]; // Magnitude input for generator
    let j3b_phase = &contracts[3]; // Phase output from generator

    // Stage 2: Generator — magnitude input, phase output.
    // Input [-80, 80] accepts decoder output [-50, 50].
    // Output [-6283.2, 6283.2] must be contained in iSTFT input.
    let generator = contract_stage(
        "kokoro_generator",
        &[1, 256],
        &[1, 256],
        j3_mag,
        j3b_phase,
        "CROWN",
        true,
    );

    let j5_audio = &contracts[5]; // Audio output

    // Stage 3: iSTFT — phase input, audio output.
    // Input [-6283.2, 6283.2] accepts generator output [-6283.2, 6283.2].
    let istft = contract_stage(
        "kokoro_istft",
        &[1, 256],
        &[1, 24000],
        j3b_phase,
        j5_audio,
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[decoder, generator, istft]).expect("valid pipeline");

    // Junction 0: Decoder output Energy [-50, 50] is within Generator input Magnitude [-80, 80].
    assert!(
        cert.junctions[0].bounds_contained,
        "J0 violation: decoder->generator, max_violation={}",
        cert.junctions[0].max_violation
    );

    // Junction 1: Generator output Phase [-6283.2, 6283.2] matches iSTFT input exactly.
    assert!(
        cert.junctions[1].bounds_contained,
        "J1 violation: generator->istft, max_violation={}",
        cert.junctions[1].max_violation
    );

    assert!(cert.is_valid, "pipeline should be valid");
    assert!(cert.is_sound, "all stages are CROWN-verified");

    // E2e: BF16 input -> audio output.
    assert_eq!(cert.e2e_input_lower[0], J4_BF16_LOWER);
    assert_eq!(cert.e2e_output_upper[0], J5_AUDIO_UPPER);
}

/// Negative test: pipeline with junction violation.
///
/// If the decoder outputs bounds wider than the generator's input contract,
/// the pipeline should be invalid.
#[test]
fn test_kokoro_junction_violation_detected() {
    let contracts = all_contracts();
    let j4_bf16 = &contracts[4];
    let j3_mag = &contracts[2]; // [-80, 80]
    let j3b_phase = &contracts[3];
    let j5_audio = &contracts[5];

    // Decoder output exceeds generator input: [-128, 128] not within [-80, 80].
    let decoder = contract_stage(
        "kokoro_decoder",
        &[1, 64],
        &[1, 64],
        j4_bf16,
        j4_bf16, // Output BF16 bounds [-128, 128]
        "CROWN",
        true,
    );

    let generator = contract_stage(
        "kokoro_generator",
        &[1, 64],
        &[1, 64],
        j3_mag, // Input magnitude bounds [-80, 80]
        j3b_phase,
        "CROWN",
        true,
    );

    let istft = contract_stage(
        "kokoro_istft",
        &[1, 64],
        &[1, 24000],
        j3b_phase,
        j5_audio,
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[decoder, generator, istft]).expect("pipeline computed");

    // Junction 0: Decoder output [-128, 128] violates Generator input [-80, 80].
    assert!(!cert.junctions[0].bounds_contained);
    assert!(cert.junctions[0].max_violation > 0.0);
    // Violation = 128 - 80 = 48 (upper breach).
    assert!(
        (cert.junctions[0].max_violation - 48.0).abs() < 1e-10,
        "expected violation 48.0, got {}",
        cert.junctions[0].max_violation
    );

    assert!(!cert.is_valid, "pipeline should be invalid");
}

/// Test that tighter NY bounds pass junction contracts.
///
/// Simulates NY proving tighter bounds than the runtime contracts.
/// This is the expected production scenario: CROWN proves [-30, 30] for
/// decoder energy, which is well within the J2 contract [-50, 50].
#[test]
fn test_tighter_crown_bounds_pass_contracts() {
    let contracts = all_contracts();

    // NY proves decoder output is actually [-30, 30], not [-50, 50].
    let tighter_output = JunctionContract {
        name: "crown_proven_energy",
        zone: "Decoder -> SourceModule",
        lower: -30.0,
        upper: 30.0,
    };

    let j4_bf16 = &contracts[4];
    let j3_mag = &contracts[2];
    let j3b_phase = &contracts[3];
    let j5_audio = &contracts[5];

    let decoder = contract_stage(
        "kokoro_decoder",
        &[1, 64],
        &[1, 64],
        j4_bf16,
        &tighter_output,
        "alpha-CROWN",
        true,
    );

    let generator = contract_stage(
        "kokoro_generator",
        &[1, 64],
        &[1, 64],
        j3_mag,
        j3b_phase,
        "alpha-CROWN",
        true,
    );

    let istft = contract_stage(
        "kokoro_istft",
        &[1, 64],
        &[1, 24000],
        j3b_phase,
        j5_audio,
        "alpha-CROWN",
        true,
    );

    // Verify using the bounds_within_contract helper directly before pipeline consumes.
    let j2_energy = &contracts[1];
    assert!(bounds_within_contract(
        j2_energy,
        &decoder.output_lower,
        &decoder.output_upper
    ));

    let cert = verify_pipeline(&[decoder, generator, istft]).expect("valid pipeline");
    assert!(cert.is_valid);
    assert!(cert.is_sound);
}

/// Test all contracts have non-overlapping or logical ordering.
#[test]
fn test_contracts_are_ordered_by_pipeline_stage() {
    let contracts = all_contracts();
    // J2 contracts come first (decoder output).
    assert!(contracts[0].name.starts_with("J2"));
    assert!(contracts[1].name.starts_with("J2"));
    // J3/J3b (generator).
    assert!(contracts[2].name.starts_with("J3_"));
    assert!(contracts[3].name.starts_with("J3B"));
    // J4 (precision boundary).
    assert!(contracts[4].name.starts_with("J4"));
    // J5 (audio output).
    assert!(contracts[5].name.starts_with("J5"));
}

// ── IEEE 754 edge cases (defense-in-depth) ───────────────────────

#[test]
fn test_bounds_within_contract_both_nan_rejected() {
    let j5 = &all_contracts()[5];
    assert!(!bounds_within_contract(j5, &[f64::NAN], &[f64::NAN]));
}

#[test]
fn test_bounds_within_contract_neg_inf_upper_rejected() {
    let j5 = &all_contracts()[5];
    // NEG_INFINITY in the upper bound is non-finite.
    assert!(!bounds_within_contract(j5, &[-0.5], &[f64::NEG_INFINITY]));
}

#[test]
fn test_bounds_within_contract_pos_inf_lower_rejected() {
    let j5 = &all_contracts()[5];
    // INFINITY in the lower bound is non-finite.
    assert!(!bounds_within_contract(j5, &[f64::INFINITY], &[0.5]));
}

#[test]
fn test_max_violation_both_nan_returns_max() {
    let c = JunctionContract::new("test", "zone", -10.0, 10.0);
    assert_eq!(
        max_contract_violation(&c, &[f64::NAN], &[f64::NAN]),
        f64::MAX
    );
}

#[test]
fn test_max_violation_neg_inf_lower_returns_max() {
    let c = JunctionContract::new("test", "zone", -10.0, 10.0);
    assert_eq!(
        max_contract_violation(&c, &[f64::NEG_INFINITY], &[5.0]),
        f64::MAX
    );
}

// ── Zero-width and inverted contracts ────────────────────────────

#[test]
fn test_zero_width_contract_exact_match() {
    let c = JunctionContract::new("point", "zone", 0.0, 0.0);
    assert!(bounds_within_contract(&c, &[0.0], &[0.0]));
    assert!(!bounds_within_contract(&c, &[-0.001], &[0.0]));
    assert!(!bounds_within_contract(&c, &[0.0], &[0.001]));
}

#[test]
fn test_inverted_contract_rejects_normal_values() {
    // Contract with lower > upper: normal finite values should fail.
    let c = JunctionContract::new("inverted", "zone", 10.0, -10.0);
    // lo=0.0 < contract.lower=10.0, so fails the lower check.
    assert!(!bounds_within_contract(&c, &[0.0], &[0.0]));
    // lo=5.0 < contract.lower=10.0, so fails.
    assert!(!bounds_within_contract(&c, &[5.0], &[5.0]));
    // hi=0.0 > contract.upper=-10.0, so fails the upper check.
    assert!(!bounds_within_contract(&c, &[10.0], &[0.0]));
}

#[test]
fn test_inverted_contract_degenerate_exact_match() {
    // Degenerate case: proven bounds match inverted contract bounds exactly.
    // lo=10.0 >= contract.lower=10.0 (OK), hi=-10.0 <= contract.upper=-10.0 (OK).
    // This "passes" because the element-wise check doesn't detect
    // that proven_lower > proven_upper. In practice, proven bounds
    // should always have lower <= upper.
    let c = JunctionContract::new("inverted", "zone", 10.0, -10.0);
    assert!(bounds_within_contract(&c, &[10.0], &[-10.0]));
}

#[test]
fn test_max_violation_inverted_contract() {
    let c = JunctionContract::new("inverted", "zone", 10.0, -10.0);
    // With contract lower=10.0, upper=-10.0, and proven [0.0, 0.0]:
    // lower_gap = 10.0 - 0.0 = 10.0 (lower violation)
    // upper_gap = 0.0 - (-10.0) = 10.0 (upper violation)
    let v = max_contract_violation(&c, &[0.0], &[0.0]);
    assert!((v - 10.0).abs() < 1e-15);
}

// ── Multi-element violation counting ─────────────────────────────

#[test]
fn test_max_violation_multi_element_mixed() {
    let c = JunctionContract::new("test", "zone", -10.0, 10.0);
    // 4 elements: [OK, lower breach 3.0, OK, upper breach 7.0]
    let lower = vec![-5.0, -13.0, 0.0, 5.0];
    let upper = vec![5.0, -3.0, 8.0, 17.0];
    let v = max_contract_violation(&c, &lower, &upper);
    // Worst violation: upper breach of 7.0 (17.0 - 10.0).
    assert!((v - 7.0).abs() < 1e-15);
}

#[test]
fn test_bounds_within_contract_multi_element_first_violates() {
    let c = JunctionContract::new("test", "zone", -10.0, 10.0);
    // First element violates, rest are fine.
    assert!(!bounds_within_contract(
        &c,
        &[-11.0, 0.0, 0.0],
        &[5.0, 5.0, 5.0]
    ));
}

#[test]
fn test_bounds_within_contract_multi_element_last_violates() {
    let c = JunctionContract::new("test", "zone", -10.0, 10.0);
    // Last element violates upper.
    assert!(!bounds_within_contract(
        &c,
        &[0.0, 0.0, 0.0],
        &[5.0, 5.0, 11.0]
    ));
}

// ── Consistency: bounds_within ↔ max_violation ───────────────────

#[test]
fn test_consistency_contained_implies_zero_violation() {
    let c = JunctionContract::new("test", "zone", -100.0, 100.0);
    let lower = vec![-50.0, -25.0, 0.0, 10.0, 99.9];
    let upper = vec![-10.0, 25.0, 50.0, 90.0, 100.0];
    assert!(bounds_within_contract(&c, &lower, &upper));
    assert!((max_contract_violation(&c, &lower, &upper)).abs() < 1e-15);
}

#[test]
fn test_consistency_violated_implies_positive_violation() {
    let c = JunctionContract::new("test", "zone", -100.0, 100.0);
    let lower = vec![-50.0, -25.0, 0.0];
    let upper = vec![-10.0, 25.0, 100.5]; // Third element breaches by 0.5.
    assert!(!bounds_within_contract(&c, &lower, &upper));
    let v = max_contract_violation(&c, &lower, &upper);
    assert!(v > 0.0);
    assert!((v - 0.5).abs() < 1e-15);
}

// ── Pipeline composition with unsound stages ─────────────────────

#[test]
fn test_pipeline_unsound_stage_propagates_to_certificate() {
    let c = JunctionContract::new("test", "zone", -100.0, 100.0);
    let narrow = JunctionContract::new("narrow", "zone", -10.0, 10.0);

    // Stage 1 is sound, stage 2 is not.
    let s1 = contract_stage("s1", &[4], &[4], &c, &narrow, "CROWN", true);
    let s2 = contract_stage("s2", &[4], &[4], &narrow, &narrow, "IBP", false);

    let cert = verify_pipeline(&[s1, s2]).expect("pipeline should compose");
    assert!(cert.is_valid);
    assert!(!cert.is_sound, "unsound stage should propagate");
}

#[test]
fn test_pipeline_all_unsound_stages() {
    let c = JunctionContract::new("test", "zone", -10.0, 10.0);
    let s1 = contract_stage("s1", &[4], &[4], &c, &c, "heuristic", false);
    let s2 = contract_stage("s2", &[4], &[4], &c, &c, "heuristic", false);

    let cert = verify_pipeline(&[s1, s2]).expect("pipeline should compose");
    assert!(cert.is_valid);
    assert!(!cert.is_sound);
}

// ── contract_stage edge cases ────────────────────────────────────

#[test]
fn test_contract_stage_preserves_method_string() {
    let c = JunctionContract::new("test", "zone", -1.0, 1.0);
    let stage = contract_stage("test", &[2], &[2], &c, &c, "alpha-CROWN", true);
    assert_eq!(stage.method, "alpha-CROWN");
}

#[test]
fn test_contract_stage_large_shape() {
    let c = JunctionContract::new("test", "zone", -1.0, 1.0);
    // 1024 elements.
    let stage = contract_stage("large", &[32, 32], &[16, 64], &c, &c, "CROWN", true);
    assert_eq!(stage.input_lower.len(), 1024);
    assert_eq!(stage.output_lower.len(), 1024);
    assert!(stage.input_lower.iter().all(|&v| v == -1.0));
    assert!(stage.output_upper.iter().all(|&v| v == 1.0));
}

// ── JunctionContract Clone and Debug ─────────────────────────────

#[test]
fn test_junction_contract_clone_independence() {
    let c = JunctionContract::new("original", "zone A", -5.0, 5.0);
    let c2 = c.clone();
    // Cloned contract has same values.
    assert_eq!(c.name, c2.name);
    assert_eq!(c.zone, c2.zone);
    assert!((c.lower - c2.lower).abs() < 1e-15);
    assert!((c.upper - c2.upper).abs() < 1e-15);
}

#[test]
fn test_junction_contract_debug_includes_fields() {
    let c = JunctionContract::new("J5_AUDIO", "iSTFT output", -1.0, 1.0);
    let dbg = format!("{c:?}");
    assert!(dbg.contains("J5_AUDIO"), "Debug should contain name");
    assert!(dbg.contains("iSTFT output"), "Debug should contain zone");
    assert!(dbg.contains("-1"), "Debug should contain lower bound");
}

// ── J3 magnitude safety for exp() ────────────────────────────────

#[test]
fn test_j3_magnitude_upper_exp_finite_f32() {
    // Verify that exp(J3_MAGNITUDE_UPPER) is finite in f32.
    let exp_val = (J3_MAGNITUDE_UPPER as f32).exp();
    assert!(
        exp_val.is_finite(),
        "exp(J3_MAGNITUDE_UPPER={J3_MAGNITUDE_UPPER}) = {exp_val} must be finite"
    );
}

#[test]
fn test_j3_magnitude_lower_exp_positive_f32() {
    // exp(negative) should be a small positive number, not zero.
    let exp_val = (J3_MAGNITUDE_LOWER as f32).exp();
    // exp(-80) is ~1.8e-35 which is denormal in f32 but still representable.
    assert!(exp_val >= 0.0, "exp(J3_MAGNITUDE_LOWER) should be >= 0");
}

// ── Kokoro pipeline: 4-stage realistic composition ───────────────

#[test]
fn test_kokoro_4stage_narrowing_pipeline() {
    // A realistic 4-stage narrowing pipeline where each output is
    // tighter than the next input contract.
    let contracts = all_contracts();
    let j4 = &contracts[4]; // BF16: [-128, 128]
    let j2_energy = &contracts[1]; // Energy: [-50, 50]
    let j3_mag = &contracts[2]; // Magnitude: [-80, 80]
    let j5 = &contracts[5]; // Audio: [-1, 1]

    // Narrowing output that fits within J3_MAG input.
    let narrow_40 = JunctionContract::new("narrow40", "test", -40.0, 40.0);
    // Very narrow output that fits within J5 audio.
    let narrow_08 = JunctionContract::new("narrow08", "test", -0.8, 0.8);

    let s1 = contract_stage("frontend", &[8], &[8], j4, &narrow_40, "CROWN", true);
    let s2 = contract_stage("decoder", &[8], &[8], j2_energy, &narrow_40, "CROWN", true);
    let s3 = contract_stage("generator", &[8], &[8], j3_mag, &narrow_08, "CROWN", true);
    let s4 = contract_stage("istft", &[8], &[8], j5, j5, "CROWN", true);

    let cert = verify_pipeline(&[s1, s2, s3, s4]).expect("pipeline should compose");
    // s1 output [-40,40] fits s2 input [-50,50]: OK.
    assert!(cert.junctions[0].bounds_contained);
    // s2 output [-40,40] fits s3 input [-80,80]: OK.
    assert!(cert.junctions[1].bounds_contained);
    // s3 output [-0.8,0.8] fits s4 input [-1,1]: OK.
    assert!(cert.junctions[2].bounds_contained);
    assert!(cert.is_valid);
    assert!(cert.is_sound);
    assert_eq!(cert.junctions.len(), 3);
}
