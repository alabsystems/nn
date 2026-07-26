// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani harnesses for config types: HardBoundsConfig, CheckOverrides, RejectionPolicy.

#[cfg(kani)]
mod proofs {
    use crate::config::{CheckOverrides, HardBoundsConfig, RejectionPolicy};

    // -----------------------------------------------------------------------
    // RejectionPolicy harnesses
    // -----------------------------------------------------------------------

    /// Prove RejectionPolicy::default() returns Reject.
    #[kani::unwind(1)]
    #[kani::proof]
    fn rejection_policy_default_is_reject() {
        let policy = RejectionPolicy::default();
        assert!(
            matches!(policy, RejectionPolicy::Reject),
            "default RejectionPolicy must be Reject"
        );
    }

    /// Prove all three RejectionPolicy variants are distinct.
    #[kani::unwind(1)]
    #[kani::proof]
    fn rejection_policy_variants_are_distinct() {
        let reject = RejectionPolicy::Reject;
        let warn = RejectionPolicy::Warn;
        let remediate = RejectionPolicy::Remediate;

        // Each pair is distinct.
        assert_ne!(reject, warn);
        assert_ne!(reject, remediate);
        assert_ne!(warn, remediate);
    }

    /// Prove every RejectionPolicy variant equals itself (reflexivity).
    #[kani::unwind(1)]
    #[kani::proof]
    fn rejection_policy_eq_reflexive() {
        let reject = RejectionPolicy::Reject;
        let warn = RejectionPolicy::Warn;
        let remediate = RejectionPolicy::Remediate;

        assert_eq!(reject, reject);
        assert_eq!(warn, warn);
        assert_eq!(remediate, remediate);
    }

    // -----------------------------------------------------------------------
    // CheckOverrides harnesses
    // -----------------------------------------------------------------------

    /// Prove CheckOverrides::default() has all None fields.
    #[kani::unwind(1)]
    #[kani::proof]
    fn check_overrides_default_all_none() {
        let co = CheckOverrides::default();
        assert!(co.min_rms.is_none(), "default min_rms must be None");
        assert!(
            co.max_amplitude.is_none(),
            "default max_amplitude must be None"
        );
        assert!(
            co.max_dc_offset.is_none(),
            "default max_dc_offset must be None"
        );
        assert!(
            co.max_click_diff.is_none(),
            "default max_click_diff must be None"
        );
        assert!(
            co.min_duration_sec.is_none(),
            "default min_duration_sec must be None"
        );
        assert!(
            co.max_duration_sec.is_none(),
            "default max_duration_sec must be None"
        );
    }

    /// Prove CheckOverrides::new() produces the same state as default().
    #[kani::unwind(1)]
    #[kani::proof]
    fn check_overrides_new_eq_default() {
        let from_new = CheckOverrides::new();
        let from_default = CheckOverrides::default();

        // All fields must match (all None).
        assert!(from_new.min_rms == from_default.min_rms);
        assert!(from_new.max_amplitude == from_default.max_amplitude);
        assert!(from_new.max_dc_offset == from_default.max_dc_offset);
        assert!(from_new.max_click_diff == from_default.max_click_diff);
        assert!(from_new.min_duration_sec == from_default.min_duration_sec);
        assert!(from_new.max_duration_sec == from_default.max_duration_sec);
    }

    /// Prove default CheckOverrides validates successfully.
    #[kani::unwind(1)]
    #[kani::proof]
    fn check_overrides_default_validates() {
        let co = CheckOverrides::default();
        assert!(
            co.validate().is_ok(),
            "default CheckOverrides must validate"
        );
    }

    // -----------------------------------------------------------------------
    // HardBoundsConfig harnesses
    // -----------------------------------------------------------------------

    /// Prove HardBoundsConfig::default() has sensible bounds:
    /// - rejection_policy == Reject (the default)
    /// - all f64 fields are finite and positive where expected
    /// - min_duration_sec < max_duration_sec
    #[kani::unwind(1)]
    #[kani::proof]
    fn hard_bounds_config_default_sensible() {
        let cfg = HardBoundsConfig::default();

        // Default rejection policy is Reject.
        assert!(
            matches!(cfg.rejection_policy, RejectionPolicy::Reject),
            "default rejection_policy must be Reject"
        );

        // All numeric defaults are finite.
        assert!(cfg.min_rms.is_finite(), "min_rms must be finite");
        assert!(
            cfg.max_amplitude.is_finite(),
            "max_amplitude must be finite"
        );
        assert!(
            cfg.max_dc_offset.is_finite(),
            "max_dc_offset must be finite"
        );
        assert!(
            cfg.max_click_diff.is_finite(),
            "max_click_diff must be finite"
        );
        assert!(
            cfg.min_duration_sec.is_finite(),
            "min_duration_sec must be finite"
        );
        assert!(
            cfg.max_duration_sec.is_finite(),
            "max_duration_sec must be finite"
        );

        // Positive where required.
        assert!(cfg.min_rms > 0.0, "min_rms must be positive");
        assert!(cfg.max_amplitude > 0.0, "max_amplitude must be positive");
        assert!(cfg.max_click_diff > 0.0, "max_click_diff must be positive");
        assert!(
            cfg.max_duration_sec > 0.0,
            "max_duration_sec must be positive"
        );

        // Duration range is valid.
        assert!(
            cfg.min_duration_sec < cfg.max_duration_sec,
            "min_duration_sec must be less than max_duration_sec"
        );
    }

    /// Prove HardBoundsConfig::default() passes its own validate().
    #[kani::unwind(1)]
    #[kani::proof]
    fn hard_bounds_config_default_validates() {
        let cfg = HardBoundsConfig::default();
        assert!(
            cfg.validate().is_ok(),
            "default HardBoundsConfig must validate"
        );
    }

    /// Prove that effective_* methods return the base value when overrides are None.
    #[kani::unwind(1)]
    #[kani::proof]
    fn hard_bounds_config_effective_without_overrides() {
        let cfg = HardBoundsConfig::default();

        // With default (empty) overrides, effective == base.
        assert_eq!(cfg.effective_min_rms(), cfg.min_rms);
        assert_eq!(cfg.effective_max_amplitude(), cfg.max_amplitude);
        assert_eq!(cfg.effective_max_dc_offset(), cfg.max_dc_offset);
        assert_eq!(cfg.effective_max_click_diff(), cfg.max_click_diff);
        assert_eq!(cfg.effective_min_duration_sec(), cfg.min_duration_sec);
        assert_eq!(cfg.effective_max_duration_sec(), cfg.max_duration_sec);
    }
}
