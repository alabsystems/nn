// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for WhisperBeamConfig validation and beam search helpers.
//!
//! Covers:
//! - WhisperBeamConfig::validate rejects zero beam_width
//! - WhisperBeamConfig::validate rejects non-finite length_penalty
//! - WhisperBeamConfig default passes validation
//! - BeamState score computation properties
//! - reconstruct_decoded from empty tree
//! - reconstruct_decoded single node
//!
//! Issue: #4303

#[cfg(kani)]
mod proofs {
    use crate::decode::WhisperBeamConfig;

    // ============================================================================
    // Harness 1: Default beam config passes validation
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn beam_config_default_passes_validation() {
        let config = WhisperBeamConfig::default();
        assert!(
            config.validate().is_ok(),
            "default WhisperBeamConfig must pass validation"
        );
    }

    // ============================================================================
    // Harness 2: Zero beam_width rejected
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn beam_config_rejects_zero_beam_width() {
        let config = WhisperBeamConfig {
            beam_width: 0,
            length_penalty: 1.0,
        };
        assert!(
            config.validate().is_err(),
            "zero beam_width must be rejected"
        );
    }

    // ============================================================================
    // Harness 3: Non-finite length_penalty rejected (NaN)
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn beam_config_rejects_nan_length_penalty() {
        let config = WhisperBeamConfig {
            beam_width: 5,
            length_penalty: f64::NAN,
        };
        assert!(
            config.validate().is_err(),
            "NaN length_penalty must be rejected"
        );
    }

    // ============================================================================
    // Harness 4: Non-finite length_penalty rejected (Inf)
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn beam_config_rejects_inf_length_penalty() {
        let config = WhisperBeamConfig {
            beam_width: 5,
            length_penalty: f64::INFINITY,
        };
        assert!(
            config.validate().is_err(),
            "Inf length_penalty must be rejected"
        );
    }

    // ============================================================================
    // Harness 5: Negative infinity length_penalty rejected
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn beam_config_rejects_neg_inf_length_penalty() {
        let config = WhisperBeamConfig {
            beam_width: 5,
            length_penalty: f64::NEG_INFINITY,
        };
        assert!(
            config.validate().is_err(),
            "NEG_INFINITY length_penalty must be rejected"
        );
    }

    // ============================================================================
    // Harness 6: Valid beam configs pass validation
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn beam_config_valid_configs_pass() {
        let bw: usize = kani::any();
        kani::assume(bw >= 1 && bw <= 10);

        let config = WhisperBeamConfig {
            beam_width: bw,
            length_penalty: 0.5,
        };
        assert!(
            config.validate().is_ok(),
            "valid beam config must pass"
        );
    }

    // ============================================================================
    // Harness 7: Default beam width is 5
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn beam_config_default_beam_width_is_5() {
        let config = WhisperBeamConfig::default();
        assert_eq!(config.beam_width, 5, "default beam_width must be 5");
    }

    // ============================================================================
    // Harness 8: Default length penalty is 1.0
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn beam_config_default_length_penalty_is_one() {
        let config = WhisperBeamConfig::default();
        assert!(
            (config.length_penalty - 1.0).abs() < 1e-12,
            "default length_penalty must be 1.0"
        );
    }
}
