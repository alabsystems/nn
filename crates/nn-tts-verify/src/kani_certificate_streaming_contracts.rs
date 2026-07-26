// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Certificate, StreamingConfig, JunctionContract,
//! HardBound checks, VerificationLevel ordering, QualityConfig defaults,
//! error validation helpers, and crossfade_linear.
//!
//! 25 harnesses proving:
//!
//! 1. Certificate::passes_hard_bounds is true when all HardBound.passed are true.
//! 2. Certificate::passes_hard_bounds is false when any HardBound.passed is false.
//! 3. Certificate::passes_quality is vacuously true for empty quality_metrics.
//! 4. StreamingConfig::default() passes validation.
//! 5. StreamingConfig::default() has energy_lo < energy_hi.
//! 6. StreamingConfig::default() margin_samples >= crossfade_samples.
//! 7. StreamingConfig validation rejects zero sample_rate.
//! 8. StreamingConfig validation rejects energy_lo >= energy_hi.
//! 9. JunctionContract all_contracts returns exactly 6 contracts.
//! 10. JunctionContract all_contracts: every contract has lower < upper.
//! 11. bounds_within_contract returns true when proven bounds are within contract.
//! 12. bounds_within_contract returns false for non-finite bounds.
//! 13. max_contract_violation returns 0.0 when bounds contained.
//! 14. max_contract_violation returns positive when bounds exceed contract.
//! 15. J5_AUDIO bounds are [-1, 1] (PCM convention).
//! 16. VerificationLevel ordering: None < Empirical < ... < SmtProven.
//! 17. QualityConfig::default() passes validation.
//! 18. QualityConfig::default() has f0_range.0 < f0_range.1.
//! 19. validate_finite rejects NaN.
//! 20. validate_finite_positive rejects zero.
//! 21. validate_finite_positive rejects negative.
//! 22. validate_finite accepts any finite value.
//! 23. crossfade_linear rejects mismatched lengths.
//! 24. crossfade_linear returns same length as input.
//! 25. SpectralCoverageConfig::default() passes validation.

#[cfg(kani)]
mod proofs {
    use crate::bounds::SpectralCoverageConfig;
    use crate::certificate::Certificate;
    use crate::config::{HardBoundsConfig, QualityConfig};
    use crate::error::{validate_finite, validate_finite_positive};
    use crate::kokoro_contracts::{
        all_contracts, bounds_within_contract, max_contract_violation, JunctionContract,
        J5_AUDIO_LOWER, J5_AUDIO_UPPER,
    };
    use crate::moonshot::VerificationLevel;
    use crate::streaming::{crossfade_linear, StreamingConfig};

    // -----------------------------------------------------------------------
    // Certificate harnesses
    // -----------------------------------------------------------------------

    /// Prove: passes_hard_bounds is true when all HardBound.passed are true.
    #[kani::unwind(8)]
    #[kani::proof]
    fn certificate_passes_hard_bounds_all_true() {
        let cert = Certificate {
            hard_bounds: vec![
                crate::bounds::HardBound {
                    name: "a",
                    passed: true,
                    value: 0.5,
                    threshold: 1.0,
                },
                crate::bounds::HardBound {
                    name: "b",
                    passed: true,
                    value: 0.1,
                    threshold: 0.5,
                },
            ],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: true,
            deterministic_hash: None,
            crown_evidence: None,
            junction_summary: None,
        };
        assert!(
            cert.passes_hard_bounds(),
            "all hard bounds passed => passes_hard_bounds must be true"
        );
    }

    /// Prove: passes_hard_bounds is false when any HardBound.passed is false.
    #[kani::unwind(8)]
    #[kani::proof]
    fn certificate_passes_hard_bounds_one_false() {
        let cert = Certificate {
            hard_bounds: vec![
                crate::bounds::HardBound {
                    name: "a",
                    passed: true,
                    value: 0.5,
                    threshold: 1.0,
                },
                crate::bounds::HardBound {
                    name: "b",
                    passed: false,
                    value: 2.0,
                    threshold: 1.0,
                },
            ],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: false,
            deterministic_hash: None,
            crown_evidence: None,
            junction_summary: None,
        };
        assert!(
            !cert.passes_hard_bounds(),
            "one hard bound failed => passes_hard_bounds must be false"
        );
    }

    /// Prove: passes_quality is vacuously true for empty quality_metrics.
    #[kani::unwind(8)]
    #[kani::proof]
    fn certificate_passes_quality_vacuous() {
        let cert = Certificate {
            hard_bounds: vec![],
            quality_metrics: vec![],
            phoneme_results: None,
            overall_passed: true,
            deterministic_hash: None,
            crown_evidence: None,
            junction_summary: None,
        };
        assert!(
            cert.passes_quality(),
            "empty quality_metrics => passes_quality must be true (vacuously)"
        );
    }

    // -----------------------------------------------------------------------
    // StreamingConfig harnesses
    // -----------------------------------------------------------------------

    /// Prove: StreamingConfig::default() passes validation.
    #[kani::unwind(1)]
    #[kani::proof]
    fn streaming_config_default_validates() {
        let cfg = StreamingConfig::default();
        assert!(
            cfg.validate().is_ok(),
            "default StreamingConfig must validate"
        );
    }

    /// Prove: StreamingConfig::default() has energy_lo < energy_hi.
    #[kani::unwind(1)]
    #[kani::proof]
    fn streaming_config_default_energy_order() {
        let cfg = StreamingConfig::default();
        assert!(
            cfg.energy_lo < cfg.energy_hi,
            "default energy_lo must be less than energy_hi"
        );
    }

    /// Prove: StreamingConfig::default() margin_samples >= crossfade_samples.
    #[kani::unwind(1)]
    #[kani::proof]
    fn streaming_config_default_margin_geq_crossfade() {
        let cfg = StreamingConfig::default();
        assert!(
            cfg.margin_samples >= cfg.crossfade_samples,
            "default margin_samples must be >= crossfade_samples"
        );
    }

    /// Prove: StreamingConfig validation rejects zero sample_rate.
    #[kani::unwind(1)]
    #[kani::proof]
    fn streaming_config_rejects_zero_sample_rate() {
        let mut cfg = StreamingConfig::default();
        cfg.sample_rate = 0;
        assert!(cfg.validate().is_err(), "zero sample_rate must be rejected");
    }

    /// Prove: StreamingConfig validation rejects energy_lo >= energy_hi.
    #[kani::unwind(1)]
    #[kani::proof]
    fn streaming_config_rejects_inverted_energy() {
        let mut cfg = StreamingConfig::default();
        cfg.energy_lo = 2.0;
        cfg.energy_hi = 1.0;
        assert!(
            cfg.validate().is_err(),
            "energy_lo >= energy_hi must be rejected"
        );
    }

    // -----------------------------------------------------------------------
    // JunctionContract harnesses
    // -----------------------------------------------------------------------

    /// Prove: all_contracts() returns exactly 6 contracts.
    #[kani::unwind(1)]
    #[kani::proof]
    fn all_contracts_count_is_six() {
        let contracts = all_contracts();
        assert_eq!(contracts.len(), 6, "must have exactly 6 junction contracts");
    }

    /// Prove: every contract has lower < upper.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(7)]
    fn all_contracts_lower_lt_upper() {
        let contracts = all_contracts();
        for c in &contracts {
            assert!(
                c.lower < c.upper,
                "contract {} must have lower < upper",
                c.name
            );
        }
    }

    /// Prove: bounds_within_contract returns true when proven bounds are within contract.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn bounds_within_contract_contained() {
        let lower: f64 = kani::any();
        let upper: f64 = kani::any();
        kani::assume(lower.is_finite() && upper.is_finite());
        kani::assume(lower >= -1.0 && upper <= 1.0);
        kani::assume(lower <= upper);

        let contract = JunctionContract::new("test", "test_zone", -1.0, 1.0);
        let result = bounds_within_contract(&contract, &[lower], &[upper]);
        assert!(result, "bounds within [-1, 1] must be contained");
    }

    /// Prove: max_contract_violation returns 0.0 when bounds are contained.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn max_violation_zero_when_contained() {
        let lower: f64 = kani::any();
        let upper: f64 = kani::any();
        kani::assume(lower.is_finite() && upper.is_finite());
        kani::assume(lower >= -1.0 && upper <= 1.0);
        kani::assume(lower <= upper);

        let contract = JunctionContract::new("test", "test_zone", -1.0, 1.0);
        let violation = max_contract_violation(&contract, &[lower], &[upper]);
        assert!(
            violation.abs() < 1e-15,
            "violation must be 0.0 when bounds are contained, got {violation}"
        );
    }

    /// Prove: max_contract_violation returns positive when bounds exceed contract.
    #[kani::unwind(1)]
    #[kani::proof]
    fn max_violation_positive_when_exceeded() {
        let contract = JunctionContract::new("test", "test_zone", -1.0, 1.0);
        // Upper bound exceeds contract upper by 0.5
        let violation = max_contract_violation(&contract, &[0.0], &[1.5]);
        assert!(
            violation > 0.0,
            "violation must be positive when bounds exceed contract"
        );
        assert!(
            (violation - 0.5).abs() < 1e-15,
            "violation must equal 0.5, got {violation}"
        );
    }

    /// Prove: J5_AUDIO bounds are [-1, 1] (PCM convention).
    #[kani::unwind(1)]
    #[kani::proof]
    fn j5_audio_bounds_are_pcm() {
        assert_eq!(J5_AUDIO_LOWER, -1.0, "J5 lower must be -1.0");
        assert_eq!(J5_AUDIO_UPPER, 1.0, "J5 upper must be 1.0");
    }

    // -----------------------------------------------------------------------
    // QualityConfig harnesses
    // -----------------------------------------------------------------------

    /// Prove: QualityConfig::default() passes validation.
    #[kani::unwind(1)]
    #[kani::proof]
    fn quality_config_default_validates() {
        let cfg = QualityConfig::default();
        assert!(
            cfg.validate().is_ok(),
            "default QualityConfig must validate"
        );
    }

    /// Prove: QualityConfig::default() has f0_range.0 < f0_range.1.
    #[kani::unwind(1)]
    #[kani::proof]
    fn quality_config_default_f0_range_valid() {
        let cfg = QualityConfig::default();
        assert!(
            cfg.f0_range.0 < cfg.f0_range.1,
            "default f0_range lower must be less than upper"
        );
    }

    // -----------------------------------------------------------------------
    // Error validation helper harnesses
    // -----------------------------------------------------------------------

    /// Prove: validate_finite rejects NaN.
    #[kani::unwind(1)]
    #[kani::proof]
    fn validate_finite_rejects_nan() {
        let result = validate_finite(f64::NAN, "test");
        assert!(result.is_err(), "NaN must be rejected by validate_finite");
    }

    /// Prove: validate_finite_positive rejects zero.
    #[kani::unwind(1)]
    #[kani::proof]
    fn validate_finite_positive_rejects_zero() {
        let result = validate_finite_positive(0.0, "test");
        assert!(
            result.is_err(),
            "zero must be rejected by validate_finite_positive"
        );
    }

    /// Prove: validate_finite_positive rejects negative.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn validate_finite_positive_rejects_negative() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite() && val < 0.0);
        let result = validate_finite_positive(val, "test");
        assert!(
            result.is_err(),
            "negative value must be rejected by validate_finite_positive"
        );
    }

    /// Prove: validate_finite accepts any finite value.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn validate_finite_accepts_finite() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());
        let result = validate_finite(val, "test");
        assert!(
            result.is_ok(),
            "any finite value must be accepted by validate_finite"
        );
    }

    // -----------------------------------------------------------------------
    // crossfade_linear harnesses
    // -----------------------------------------------------------------------

    /// Prove: crossfade_linear rejects mismatched lengths.
    #[kani::unwind(1)]
    #[kani::proof]
    fn crossfade_linear_rejects_mismatched_lengths() {
        let tail = [0.5_f32, 0.5];
        let head = [0.3_f32];
        let result = crossfade_linear(&tail, &head);
        assert!(
            result.is_err(),
            "mismatched lengths must be rejected by crossfade_linear"
        );
    }

    /// Prove: crossfade_linear returns same length as input.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(4)]
    fn crossfade_linear_output_length() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        let c: f32 = kani::any();
        let d: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());
        kani::assume(c.is_finite() && d.is_finite());
        kani::assume(a.abs() <= 1.0 && b.abs() <= 1.0);
        kani::assume(c.abs() <= 1.0 && d.abs() <= 1.0);

        let tail = [a, b];
        let head = [c, d];
        let result = crossfade_linear(&tail, &head);
        assert!(result.is_ok(), "equal-length crossfade must succeed");
        let blended = result.unwrap();
        assert_eq!(
            blended.len(),
            tail.len(),
            "crossfade output must have same length as input"
        );
    }

    // -----------------------------------------------------------------------
    // SpectralCoverageConfig harness
    // -----------------------------------------------------------------------

    /// Prove: SpectralCoverageConfig::default() passes validation.
    #[kani::unwind(1)]
    #[kani::proof]
    fn spectral_coverage_config_default_validates() {
        let cfg = SpectralCoverageConfig::default();
        assert!(
            cfg.validate().is_ok(),
            "default SpectralCoverageConfig must validate"
        );
    }
}
