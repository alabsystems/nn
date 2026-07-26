// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for [`CompiledKokoro::new_with_hard_bounds`] (#3785).
//!
//! Validates that custom [`HardBoundsConfig`] propagates correctly to the
//! embedded [`TtsVerifier`] in the pipeline. Tests use miniaturized synthetic
//! weights (D=8) -- no production assets required.
//!
//! Test cases:
//! - Construction with custom config succeeds.
//! - Default constructor still works identically.
//! - Warn policy produces `overall_passed == true` despite hard bound failures.
//! - Reject policy (default) correctly reflects actual hard bound failures.
//! - HardBoundsConfig::validate() catches invalid configs.
//! - Relaxed amplitude override is at least as permissive as default.
//!
//! Part of #3785, #3780, #3525.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};
use nn_metal::compiled_kokoro::CompiledKokoro;
use nn_tts_verify::{HardBoundsConfig, RejectionPolicy};

use super::kokoro_test_weights as kw;

fn cpu() -> Device {
    Device::Cpu
}

fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

/// Must match `kw::mini_test_config().style_dim`.
const STYLE_DIM: usize = 4;

/// Build test input tensors for the mini Kokoro model.
fn test_inputs() -> (DynTensor, DynTensor) {
    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(200, 2 * STYLE_DIM, -0.1, 0.1),
        &[1, 2 * STYLE_DIM],
        &cpu(),
    )
    .unwrap();
    (input_ids, style)
}

/// Build a `KokoroModel` from synthetic weights for the mini config.
fn build_model() -> nn_models::KokoroModel {
    let config = kw::mini_test_config();
    let weights = kw::all_weights(&config);
    let vb = VarBuilder::from_tensors(weights, DType::F32, &gpu());
    nn_models::KokoroModel::load(&vb, &config).expect("KokoroModel::load synthetic weights")
}

/// Create a `HardBoundsConfig` with Warn policy and relaxed max_amplitude.
fn warn_config_with_relaxed_amplitude() -> HardBoundsConfig {
    let mut hb = HardBoundsConfig::default();
    hb.rejection_policy = RejectionPolicy::Warn;
    hb.overrides.max_amplitude = Some(2.0);
    hb
}

/// Create a `HardBoundsConfig` with Warn policy and impossible min_rms.
fn warn_config_with_impossible_rms() -> HardBoundsConfig {
    let mut hb = HardBoundsConfig::default();
    hb.rejection_policy = RejectionPolicy::Warn;
    hb.overrides.min_rms = Some(100.0);
    hb
}

/// Create a `HardBoundsConfig` with Reject policy and impossible min_rms.
fn reject_config_with_impossible_rms() -> HardBoundsConfig {
    let mut hb = HardBoundsConfig::default();
    hb.rejection_policy = RejectionPolicy::Reject;
    hb.overrides.min_rms = Some(100.0);
    hb
}

/// Create a `HardBoundsConfig` with Reject policy and relaxed max_amplitude.
fn reject_config_with_relaxed_amplitude() -> HardBoundsConfig {
    let mut hb = HardBoundsConfig::default();
    hb.rejection_policy = RejectionPolicy::Reject;
    hb.overrides.max_amplitude = Some(10.0);
    hb
}

// =============================================================================
// Test: Construction with custom HardBoundsConfig
// =============================================================================

/// `new_with_hard_bounds` succeeds with a valid custom config.
#[test]
fn test_compiled_kokoro_hard_bounds_construction() {
    super::test_utils::gpu_init();

    let hb = warn_config_with_relaxed_amplitude();
    let kokoro = CompiledKokoro::new_with_hard_bounds(build_model(), hb);
    assert!(
        kokoro.is_ok(),
        "new_with_hard_bounds should succeed: {:?}",
        kokoro.err()
    );
}

// =============================================================================
// Test: Default constructor unchanged
// =============================================================================

/// Default `new()` still works identically after adding `new_with_hard_bounds`.
#[test]
fn test_compiled_kokoro_default_constructor_unchanged() {
    let (kokoro, _cache) = kw::build_kokoro_mini();
    assert_eq!(kokoro.config().d_en, 8, "mini config d_en must be 8");
    assert!(
        !kokoro.weights_released(),
        "default constructor should not release weights"
    );
}

// =============================================================================
// Test: Warn policy overall_passed behavior
// =============================================================================

/// Warn policy makes `overall_passed == true` even when hard bounds fail.
///
/// Miniaturized zero-weight synthesis produces near-silent audio, which
/// fails the `min_rms` hard bound when set to an impossible threshold.
/// With `RejectionPolicy::Warn`, this failure is recorded in the certificate
/// but does not block `overall_passed`.
#[test]
fn test_hard_bounds_warn_policy_overall_passed() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let hb = warn_config_with_impossible_rms();
    let mut kokoro =
        CompiledKokoro::new_with_hard_bounds(build_model(), hb).expect("new_with_hard_bounds");

    let (input_ids, style) = test_inputs();
    let result = kokoro.synthesize(&input_ids, &style, 1.0, &cache);
    assert!(
        result.is_ok(),
        "synthesis should succeed: {:?}",
        result.err()
    );

    let (_audio, cert) = result.unwrap();
    // With Warn policy, overall_passed should be true regardless of hard bound failures.
    assert!(
        cert.overall_passed,
        "Warn policy should make overall_passed true even with hard bound failures"
    );
    // But individual hard bounds should reflect the actual result.
    let any_failed = cert.hard_bounds.iter().any(|b| !b.passed);
    assert!(
        any_failed,
        "with min_rms=100.0, at least one hard bound should have actually failed"
    );
}

// =============================================================================
// Test: Reject policy blocks on hard bound failure
// =============================================================================

/// Reject policy (default) correctly surfaces hard bound failures in `overall_passed`.
#[test]
fn test_hard_bounds_reject_policy_blocks() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let hb = reject_config_with_impossible_rms();
    let mut kokoro =
        CompiledKokoro::new_with_hard_bounds(build_model(), hb).expect("new_with_hard_bounds");

    let (input_ids, style) = test_inputs();
    let result = kokoro.synthesize(&input_ids, &style, 1.0, &cache);
    assert!(
        result.is_ok(),
        "synthesis should succeed: {:?}",
        result.err()
    );

    let (_audio, cert) = result.unwrap();
    // With Reject policy and impossible min_rms, overall_passed should be false.
    assert!(
        !cert.overall_passed,
        "Reject policy should make overall_passed false when hard bounds fail"
    );
}

// =============================================================================
// Test: HardBoundsConfig::validate() catches invalid configs
// =============================================================================

/// HardBoundsConfig validation catches NaN thresholds.
///
/// Note: `new_with_hard_bounds` does not validate the HardBoundsConfig at
/// construction time (the TtsVerifierBuilder only checks sample_rate).
/// This test verifies that `HardBoundsConfig::validate()` correctly rejects
/// invalid configs, which callers should use before passing to
/// `new_with_hard_bounds`.
#[test]
fn test_hard_bounds_config_validate_rejects_nan() {
    let mut hb = HardBoundsConfig::default();
    hb.max_amplitude = f64::NAN;
    assert!(
        hb.validate().is_err(),
        "HardBoundsConfig::validate() should reject NaN max_amplitude"
    );
}

/// HardBoundsConfig validation catches inverted duration range.
#[test]
fn test_hard_bounds_config_validate_rejects_inverted_range() {
    let mut hb = HardBoundsConfig::default();
    hb.min_duration_sec = 10.0;
    hb.max_duration_sec = 1.0;
    assert!(
        hb.validate().is_err(),
        "HardBoundsConfig::validate() should reject min_duration > max_duration"
    );
}

// =============================================================================
// Test: Relaxed amplitude override
// =============================================================================

/// Relaxed max_amplitude override is at least as permissive as default.
///
/// Constructs two pipelines: default and relaxed (max_amplitude=10.0).
/// The relaxed pipeline should pass at least as many hard bounds.
#[test]
fn test_hard_bounds_relaxed_amplitude_override() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Default pipeline.
    let config = kw::mini_test_config();
    let (mut default_kokoro, _) = kw::build_kokoro_with_config(&config);
    let (input_ids, style) = test_inputs();
    let (_, default_cert) = default_kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("default synthesis");

    // Relaxed pipeline: max_amplitude = 10.0 (very permissive).
    let hb = reject_config_with_relaxed_amplitude();
    let mut relaxed_kokoro = CompiledKokoro::new_with_hard_bounds(build_model(), hb)
        .expect("relaxed new_with_hard_bounds");
    let (_, relaxed_cert) = relaxed_kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("relaxed synthesis");

    // Relaxed pipeline should pass at least as many hard bounds as default.
    let default_passes = default_cert.hard_bounds.iter().filter(|b| b.passed).count();
    let relaxed_passes = relaxed_cert.hard_bounds.iter().filter(|b| b.passed).count();
    assert!(
        relaxed_passes >= default_passes,
        "relaxed config should pass at least as many hard bounds as default \
         (relaxed: {relaxed_passes}, default: {default_passes})"
    );
}
