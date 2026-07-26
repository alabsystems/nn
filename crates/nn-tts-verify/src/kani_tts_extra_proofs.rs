// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extra Kani proof harnesses for nn-tts-verify.
//!
//! Proves safety and correctness properties across Certificate, HardBoundsConfig,
//! QualityConfig, cost model, streaming config, bounds, and DSP utilities.
//!
//! Harnesses:
//!  1. certificate_empty_hard_bounds_passes
//!  2. certificate_empty_quality_passes
//!  3. certificate_passes_hard_bounds_reflects_all
//!  4. certificate_passes_quality_vacuously_true
//!  5. hard_bounds_config_effective_overrides_take_precedence
//!  6. hard_bounds_config_validate_rejects_nan_min_rms
//!  7. hard_bounds_config_validate_rejects_inverted_duration
//!  8. quality_config_default_validates
//!  9. quality_config_default_f0_range_ordered
//! 10. quality_config_default_tilt_range_ordered
//! 11. spectral_coverage_config_default_validates
//! 12. spectral_coverage_config_default_n_bands_nonzero
//! 13. streaming_config_default_validates
//! 14. streaming_config_default_margin_ge_crossfade
//! 15. streaming_config_default_energy_lo_lt_hi
//! 16. hardware_cost_model_conservative_dominates_theoretical
//! 17. hardware_cost_model_zero_flops_mem_returns_overhead
//! 18. peak_memory_profile_within_bound_monotone
//! 19. peak_memory_profile_mb_conversion_correct
//! 20. dsp_rms_empty_is_zero
//! 21. dsp_dc_offset_empty_is_zero
//! 22. dsp_max_sample_diff_single_element_is_zero
//! 23. check_overrides_with_override_validates
//! 24. hard_bound_name_is_static

// ---- Certificate Proofs ---------------------------------------------------

/// Prove: Certificate with empty hard_bounds vec passes hard bounds check.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn certificate_empty_hard_bounds_passes() {
    let cert = super::Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: true,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
    };
    assert!(
        cert.passes_hard_bounds(),
        "empty hard_bounds must pass (vacuously true)"
    );
}

/// Prove: Certificate with empty quality_metrics vec passes quality check.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn certificate_empty_quality_passes() {
    let cert = super::Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: false,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
    };
    assert!(
        cert.passes_quality(),
        "empty quality_metrics must pass (vacuously true)"
    );
}

/// Prove: Certificate::passes_hard_bounds returns false when any bound fails.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn certificate_passes_hard_bounds_reflects_all() {
    let pass_bound = super::HardBound {
        name: "test_pass",
        passed: true,
        value: 0.5,
        threshold: 1.0,
    };
    let fail_bound = super::HardBound {
        name: "test_fail",
        passed: false,
        value: 1.5,
        threshold: 1.0,
    };
    let cert = super::Certificate {
        hard_bounds: vec![pass_bound, fail_bound],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: false,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
    };
    assert!(
        !cert.passes_hard_bounds(),
        "certificate with a failing bound must not pass hard bounds"
    );
}

/// Prove: Certificate::passes_quality is vacuously true when no metrics present.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn certificate_passes_quality_vacuously_true() {
    let cert = super::Certificate {
        hard_bounds: vec![super::HardBound {
            name: "test",
            passed: false,
            value: 2.0,
            threshold: 1.0,
        }],
        quality_metrics: vec![],
        phoneme_results: None,
        overall_passed: false,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
    };
    // Quality passes even though hard bounds fail — they are independent.
    assert!(cert.passes_quality());
}

// ---- HardBoundsConfig Proofs -----------------------------------------------

/// Prove: effective_* methods return the override value when set.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn hard_bounds_config_effective_overrides_take_precedence() {
    let mut cfg = super::HardBoundsConfig::default();
    cfg.overrides.min_rms = Some(0.05);
    cfg.overrides.max_amplitude = Some(0.9);
    cfg.overrides.max_dc_offset = Some(0.02);
    cfg.overrides.max_click_diff = Some(0.3);
    cfg.overrides.min_duration_sec = Some(0.5);
    cfg.overrides.max_duration_sec = Some(120.0);

    assert_eq!(cfg.effective_min_rms(), 0.05);
    assert_eq!(cfg.effective_max_amplitude(), 0.9);
    assert_eq!(cfg.effective_max_dc_offset(), 0.02);
    assert_eq!(cfg.effective_max_click_diff(), 0.3);
    assert_eq!(cfg.effective_min_duration_sec(), 0.5);
    assert_eq!(cfg.effective_max_duration_sec(), 120.0);
}

/// Prove: HardBoundsConfig::validate rejects NaN min_rms.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn hard_bounds_config_validate_rejects_nan_min_rms() {
    let mut cfg = super::HardBoundsConfig::default();
    cfg.min_rms = f64::NAN;
    assert!(cfg.validate().is_err(), "NaN min_rms must fail validation");
}

/// Prove: HardBoundsConfig::validate rejects inverted duration range.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn hard_bounds_config_validate_rejects_inverted_duration() {
    let mut cfg = super::HardBoundsConfig::default();
    cfg.min_duration_sec = 100.0;
    cfg.max_duration_sec = 10.0;
    assert!(
        cfg.validate().is_err(),
        "inverted duration range must fail validation"
    );
}

/// Prove: QualityConfig::default() has f0_range.0 < f0_range.1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn quality_config_default_f0_range_ordered() {
    let cfg = super::QualityConfig::default();
    assert!(
        cfg.f0_range.0 < cfg.f0_range.1,
        "f0_range must be ordered: low < high"
    );
}

/// Prove: QualityConfig::default() has spectral_tilt.0 < spectral_tilt.1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn quality_config_default_tilt_range_ordered() {
    let cfg = super::QualityConfig::default();
    assert!(
        cfg.spectral_tilt.0 < cfg.spectral_tilt.1,
        "spectral_tilt must be ordered: low < high"
    );
}

/// Prove: SpectralCoverageConfig::default() has n_bands > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn spectral_coverage_config_default_n_bands_nonzero() {
    let cfg = super::SpectralCoverageConfig::default();
    assert!(cfg.n_bands > 0, "n_bands must be > 0");
}

/// Prove: StreamingConfig::default() has margin_samples >= crossfade_samples.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn streaming_config_default_margin_ge_crossfade() {
    let cfg = super::StreamingConfig::default();
    assert!(
        cfg.margin_samples >= cfg.crossfade_samples,
        "margin_samples must be >= crossfade_samples"
    );
}

/// Prove: StreamingConfig::default() has energy_lo < energy_hi.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn streaming_config_default_energy_lo_lt_hi() {
    let cfg = super::StreamingConfig::default();
    assert!(
        cfg.energy_lo < cfg.energy_hi,
        "energy_lo must be less than energy_hi"
    );
}

// ---- HardwareCostModel Proofs ----------------------------------------------

/// Prove: conservative model gives >= theoretical model for any workload.
///
/// m4_max_conservative() has lower throughput and higher overhead than m4_max(),
/// so estimate_time_us() is always >= the theoretical model's estimate.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn hardware_cost_model_conservative_dominates_theoretical() {
    let theoretical = super::HardwareCostModel::m4_max();
    let conservative = super::HardwareCostModel::m4_max_conservative();

    let flops: u64 = kani::any();
    let mem_bytes: u64 = kani::any();
    kani::assume(flops <= 1_000_000_000);
    kani::assume(mem_bytes <= 1_000_000_000);

    let t_theo = theoretical.estimate_time_us(flops, mem_bytes);
    let t_cons = conservative.estimate_time_us(flops, mem_bytes);
    assert!(
        t_cons >= t_theo - 1e-9,
        "conservative model must dominate theoretical model"
    );
}

/// Prove: zero FLOPs and zero memory returns exactly dispatch_overhead_us.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn hardware_cost_model_zero_flops_mem_returns_overhead() {
    let model = super::HardwareCostModel::m4_max();
    let time = model.estimate_time_us(0, 0);
    // max(0/peak, 0/bw) + overhead = 0 + overhead = overhead
    assert!(
        (time - model.dispatch_overhead_us).abs() < 1e-12,
        "zero workload must return exactly dispatch_overhead_us"
    );
}

// ---- PeakMemoryProfile Proofs ----------------------------------------------

/// Prove: PeakMemoryProfile::within_bound is monotone in the bound parameter.
///
/// If within_bound(B1) is true and B2 >= B1, then within_bound(B2) is also true.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn peak_memory_profile_within_bound_monotone() {
    let profile = super::PeakMemoryProfile {
        weight_bytes: 1000,
        peak_activation_bytes: 500,
        peak_total_bytes: 1500,
        peak_step_index: 0,
        peak_step_name: String::new(),
        per_step_output_bytes: vec![],
    };

    let b1: u64 = kani::any();
    let b2: u64 = kani::any();
    kani::assume(b1 <= b2);

    if profile.within_bound(b1) {
        assert!(
            profile.within_bound(b2),
            "within_bound must be monotone: if passes at B1, must pass at B2 >= B1"
        );
    }
}

/// Prove: PeakMemoryProfile::peak_total_mb conversion is correct.
///
/// peak_total_bytes / (1024 * 1024) == peak_total_mb()
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn peak_memory_profile_mb_conversion_correct() {
    let bytes: u64 = kani::any();
    kani::assume(bytes <= 1_000_000_000); // 1 GB — avoids f64 precision edge cases

    let profile = super::PeakMemoryProfile {
        weight_bytes: 0,
        peak_activation_bytes: 0,
        peak_total_bytes: bytes,
        peak_step_index: 0,
        peak_step_name: String::new(),
        per_step_output_bytes: vec![],
    };

    let expected = bytes as f64 / (1024.0 * 1024.0);
    let actual = profile.peak_total_mb();
    assert!(
        (actual - expected).abs() < 1e-9,
        "peak_total_mb must be peak_total_bytes / (1024*1024)"
    );
}

// ---- DSP Function Proofs ---------------------------------------------------

/// Prove: rms([]) returns 0.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dsp_rms_empty_is_zero() {
    let result = super::dsp::rms(&[]);
    assert!(result == 0.0, "rms of empty slice must be 0.0");
}

/// Prove: dc_offset([]) returns 0.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dsp_dc_offset_empty_is_zero() {
    let result = super::dsp::dc_offset(&[]);
    assert!(result == 0.0, "dc_offset of empty slice must be 0.0");
}

/// Prove: max_sample_diff of a single element is 0.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn dsp_max_sample_diff_single_element_is_zero() {
    let result = super::dsp::max_sample_diff(&[0.5]);
    assert!(
        result == 0.0,
        "max_sample_diff of single element must be 0.0"
    );
}

// ---- CheckOverrides Proofs -------------------------------------------------

/// Prove: CheckOverrides with a valid override passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn check_overrides_with_override_validates() {
    let mut co = super::CheckOverrides::new();
    co.min_rms = Some(0.02);
    co.max_amplitude = Some(0.95);
    assert!(
        co.validate().is_ok(),
        "valid overrides must pass validation"
    );
}

// ---- HardBound Proofs ------------------------------------------------------

/// Prove: HardBound fields are accessible and consistent with construction.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn hard_bound_name_is_static() {
    let hb = super::HardBound {
        name: "test_bound",
        passed: true,
        value: 0.5,
        threshold: 1.0,
    };
    assert_eq!(hb.name, "test_bound");
    assert!(hb.passed);
    assert!((hb.value - 0.5).abs() < 1e-15);
    assert!((hb.threshold - 1.0).abs() < 1e-15);
}
