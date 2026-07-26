// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for cross-cutting verification properties.
//!
//! Proves correctness of:
//! - `compute_proof_strength`: ProofStrength classification from soundness/method/width
//! - `PropMethod::is_tight`: CROWN-family method classification
//! - `NormBoundsMode::forward_mode`: mode flag consistency
//! - `VerifyConfig::with_threshold`: threshold validation (IEEE 754 defense)
//! - `finite_or` / `sanitize_tensor_bounds`: NaN/Inf sanitization
//! - `ContractProperty::new`: behavioral contract construction
//! - `ContractValidation::passing`: default validation invariant
//! - `AnalysisConfig::default`: config invariant bounds
//! - `KernelVerification::new`: constructor field preservation
//! - `OutputTensorBounds::new`: constructor field preservation
//! - `VerifyConfig::default`: default config invariants
//! - `ParamInputRecord::new`: constructor field preservation
//!
//! Part of #4295.

#[cfg(kani)]
mod proofs {
    // ========================================================================
    // compute_proof_strength proofs
    // ========================================================================

    use crate::soundness_compat::VerificationSoundnessMode;
    use crate::status_proof_strength::{
        compute_proof_strength, ProofStrength, VACUOUS_WIDTH_THRESHOLD,
    };
    use crate::verify_types::PropMethod;

    /// Prove: any output_width > VACUOUS_WIDTH_THRESHOLD yields Vacuous,
    /// regardless of soundness mode or method.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_strength_wide_bounds_always_vacuous() {
        let mode_idx: u8 = kani::any();
        kani::assume(mode_idx < 2);
        let mode = if mode_idx == 0 {
            VerificationSoundnessMode::Sound
        } else {
            VerificationSoundnessMode::Heuristic
        };

        let method_idx: u8 = kani::any();
        kani::assume(method_idx < 6);
        let method = match method_idx {
            0 => PropMethod::Ibp,
            1 => PropMethod::Crown,
            2 => PropMethod::AlphaCrown,
            3 => PropMethod::BetaCrown,
            4 => PropMethod::Analytical,
            _ => PropMethod::MixedIbpCrown,
        };

        // Any width strictly above the threshold
        let width: f32 = kani::any();
        kani::assume(width > VACUOUS_WIDTH_THRESHOLD);
        kani::assume(width.is_finite());

        let strength = compute_proof_strength(mode, method, width);
        assert_eq!(strength, ProofStrength::Vacuous);
    }

    /// Prove: Sound mode + tight method => SoundCrown (when not vacuous).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_strength_sound_tight_is_sound_crown() {
        let method_idx: u8 = kani::any();
        kani::assume(method_idx < 4);
        let method = match method_idx {
            0 => PropMethod::Crown,
            1 => PropMethod::AlphaCrown,
            2 => PropMethod::BetaCrown,
            _ => PropMethod::Analytical,
        };

        let width: f32 = kani::any();
        kani::assume(width.is_finite());
        kani::assume(width <= VACUOUS_WIDTH_THRESHOLD);

        let strength = compute_proof_strength(VerificationSoundnessMode::Sound, method, width);
        assert_eq!(strength, ProofStrength::SoundCrown);
    }

    /// Prove: Sound mode + IBP (non-tight, non-mixed) => SoundIbp.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_strength_sound_ibp_is_sound_ibp() {
        let width: f32 = kani::any();
        kani::assume(width.is_finite());
        kani::assume(width <= VACUOUS_WIDTH_THRESHOLD);

        let strength =
            compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Ibp, width);
        assert_eq!(strength, ProofStrength::SoundIbp);
    }

    /// Prove: Sound mode + MixedIbpCrown => SoundMixed.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_strength_sound_mixed_is_sound_mixed() {
        let width: f32 = kani::any();
        kani::assume(width.is_finite());
        kani::assume(width <= VACUOUS_WIDTH_THRESHOLD);

        let strength = compute_proof_strength(
            VerificationSoundnessMode::Sound,
            PropMethod::MixedIbpCrown,
            width,
        );
        assert_eq!(strength, ProofStrength::SoundMixed);
    }

    /// Prove: Heuristic mode => Heuristic (when not vacuous).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_strength_heuristic_mode_is_heuristic() {
        let method_idx: u8 = kani::any();
        kani::assume(method_idx < 6);
        let method = match method_idx {
            0 => PropMethod::Ibp,
            1 => PropMethod::Crown,
            2 => PropMethod::AlphaCrown,
            3 => PropMethod::BetaCrown,
            4 => PropMethod::Analytical,
            _ => PropMethod::MixedIbpCrown,
        };

        let width: f32 = kani::any();
        kani::assume(width.is_finite());
        kani::assume(width <= VACUOUS_WIDTH_THRESHOLD);

        let strength = compute_proof_strength(VerificationSoundnessMode::Heuristic, method, width);
        assert_eq!(strength, ProofStrength::Heuristic);
    }

    /// Prove: VACUOUS_WIDTH_THRESHOLD is exactly 100.0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn vacuous_threshold_is_100() {
        assert_eq!(VACUOUS_WIDTH_THRESHOLD, 100.0f32);
    }

    // ========================================================================
    // PropMethod::is_tight proofs
    // ========================================================================

    /// Prove: Crown, AlphaCrown, BetaCrown, Analytical are tight.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prop_method_tight_variants_are_tight() {
        assert!(PropMethod::Crown.is_tight());
        assert!(PropMethod::AlphaCrown.is_tight());
        assert!(PropMethod::BetaCrown.is_tight());
        assert!(PropMethod::Analytical.is_tight());
    }

    /// Prove: IBP and MixedIbpCrown are NOT tight.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prop_method_loose_variants_are_not_tight() {
        assert!(!PropMethod::Ibp.is_tight());
        assert!(!PropMethod::MixedIbpCrown.is_tight());
    }

    /// Prove: is_tight partitions all 6 variants — exactly 4 tight, 2 not tight.
    #[kani::unwind(1)]
    #[kani::proof]
    fn prop_method_is_tight_partition() {
        let method_idx: u8 = kani::any();
        kani::assume(method_idx < 6);
        let method = match method_idx {
            0 => PropMethod::Ibp,
            1 => PropMethod::Crown,
            2 => PropMethod::AlphaCrown,
            3 => PropMethod::BetaCrown,
            4 => PropMethod::Analytical,
            _ => PropMethod::MixedIbpCrown,
        };

        // Every variant returns a defined bool (no panic)
        let result = method.is_tight();
        // Tight iff the variant is one of Crown/AlphaCrown/BetaCrown/Analytical
        let expected_tight = matches!(method_idx, 1 | 2 | 3 | 4);
        assert_eq!(result, expected_tight);
    }

    // ========================================================================
    // NormBoundsMode::forward_mode proofs
    // ========================================================================

    use crate::verify_types::NormBoundsMode;

    /// Prove: Conservative does NOT use forward mode.
    #[kani::unwind(1)]
    #[kani::proof]
    fn norm_mode_conservative_not_forward() {
        assert!(!NormBoundsMode::Conservative.forward_mode());
    }

    /// Prove: ForwardMode uses forward mode.
    #[kani::unwind(1)]
    #[kani::proof]
    fn norm_mode_forward_uses_forward() {
        assert!(NormBoundsMode::ForwardMode.forward_mode());
    }

    /// Prove: CrownSampling uses forward mode.
    #[kani::unwind(1)]
    #[kani::proof]
    fn norm_mode_crown_sampling_uses_forward() {
        assert!(NormBoundsMode::CrownSampling.forward_mode());
    }

    // ========================================================================
    // VerifyConfig::with_threshold proofs
    // ========================================================================

    /// Prove: with_threshold rejects NaN.
    #[kani::unwind(1)]
    #[kani::proof]
    fn verify_config_rejects_nan_threshold() {
        let result = crate::verify_types::VerifyConfig::with_threshold(f32::NAN);
        assert!(result.is_err());
    }

    /// Prove: with_threshold rejects positive infinity.
    #[kani::unwind(1)]
    #[kani::proof]
    fn verify_config_rejects_inf_threshold() {
        let result = crate::verify_types::VerifyConfig::with_threshold(f32::INFINITY);
        assert!(result.is_err());
    }

    /// Prove: with_threshold rejects negative infinity.
    #[kani::unwind(1)]
    #[kani::proof]
    fn verify_config_rejects_neg_inf_threshold() {
        let result = crate::verify_types::VerifyConfig::with_threshold(f32::NEG_INFINITY);
        assert!(result.is_err());
    }

    /// Prove: with_threshold rejects negative values.
    #[kani::unwind(1)]
    #[kani::proof]
    fn verify_config_rejects_negative_threshold() {
        let result = crate::verify_types::VerifyConfig::with_threshold(-1.0);
        assert!(result.is_err());
    }

    /// Prove: with_threshold accepts zero.
    #[kani::unwind(1)]
    #[kani::proof]
    fn verify_config_accepts_zero_threshold() {
        let result = crate::verify_types::VerifyConfig::with_threshold(0.0);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.escalation_threshold(), 0.0);
    }

    /// Prove: with_threshold accepts any finite non-negative value and preserves it.
    #[kani::unwind(1)]
    #[kani::proof]
    fn verify_config_accepts_valid_threshold() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val >= 0.0);

        let result = crate::verify_types::VerifyConfig::with_threshold(val);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.escalation_threshold(), val);
    }

    // ========================================================================
    // finite_or proofs
    // ========================================================================

    /// Prove: finite_or returns val when val is finite.
    #[kani::unwind(1)]
    #[kani::proof]
    fn finite_or_returns_val_when_finite() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        let fallback: f32 = kani::any();

        let result = crate::util::finite_or(val, fallback);
        assert_eq!(result, val);
    }

    /// Prove: finite_or returns fallback for NaN.
    #[kani::unwind(1)]
    #[kani::proof]
    fn finite_or_returns_fallback_for_nan() {
        let fallback: f32 = kani::any();
        let result = crate::util::finite_or(f32::NAN, fallback);
        // NaN != NaN, so check via bit pattern
        assert_eq!(result.to_bits(), fallback.to_bits());
    }

    /// Prove: finite_or returns fallback for positive infinity.
    #[kani::unwind(1)]
    #[kani::proof]
    fn finite_or_returns_fallback_for_inf() {
        let fallback: f32 = kani::any();
        let result = crate::util::finite_or(f32::INFINITY, fallback);
        assert_eq!(result.to_bits(), fallback.to_bits());
    }

    /// Prove: finite_or returns fallback for negative infinity.
    #[kani::unwind(1)]
    #[kani::proof]
    fn finite_or_returns_fallback_for_neg_inf() {
        let fallback: f32 = kani::any();
        let result = crate::util::finite_or(f32::NEG_INFINITY, fallback);
        assert_eq!(result.to_bits(), fallback.to_bits());
    }

    /// Prove: finite_or output is finite when fallback is finite.
    #[kani::unwind(1)]
    #[kani::proof]
    fn finite_or_output_finite_when_fallback_finite() {
        let val: f32 = kani::any();
        let fallback: f32 = kani::any();
        kani::assume(fallback.is_finite());

        let result = crate::util::finite_or(val, fallback);
        assert!(result.is_finite());
    }

    // ========================================================================
    // sanitize_tensor_bounds proofs
    // ========================================================================

    /// Prove: sanitize_tensor_bounds preserves length.
    #[kani::unwind(5)]
    #[kani::proof]
    fn sanitize_tensor_bounds_preserves_length() {
        let len: usize = kani::any();
        kani::assume(len <= 3);

        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(kani::any::<f32>());
        }

        let sanitized = crate::util::sanitize_tensor_bounds(&values);
        assert_eq!(sanitized.len(), values.len());
    }

    /// Prove: all outputs of sanitize_tensor_bounds are finite (fallback=0.0 is finite).
    #[kani::unwind(5)]
    #[kani::proof]
    fn sanitize_tensor_bounds_all_outputs_finite() {
        let len: usize = kani::any();
        kani::assume(len <= 3);

        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(kani::any::<f32>());
        }

        let sanitized = crate::util::sanitize_tensor_bounds(&values);
        for &v in &sanitized {
            assert!(
                v.is_finite(),
                "sanitize_tensor_bounds must produce finite values"
            );
        }
    }

    /// Prove: sanitize_tensor_bounds preserves finite values unchanged.
    #[kani::unwind(5)]
    #[kani::proof]
    fn sanitize_tensor_bounds_preserves_finite_values() {
        let len: usize = kani::any();
        kani::assume(len <= 3);

        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            let v: f32 = kani::any();
            kani::assume(v.is_finite());
            values.push(v);
        }

        let sanitized = crate::util::sanitize_tensor_bounds(&values);
        for (i, &v) in sanitized.iter().enumerate() {
            assert_eq!(v, values[i]);
        }
    }

    // ========================================================================
    // ContractProperty / ContractValidation proofs
    // ========================================================================

    use crate::behavioral_contract::{ContractProperty, ContractValidation};

    /// Prove: ContractProperty::new preserves name, bound_value, threshold.
    #[kani::unwind(1)]
    #[kani::proof]
    fn contract_property_new_preserves_fields() {
        let bound_value: f64 = 42.0;
        let threshold: f64 = 100.0;
        let prop = ContractProperty::new("test_prop", bound_value, threshold);

        assert_eq!(prop.name, "test_prop");
        assert_eq!(prop.bound_value, bound_value);
        assert_eq!(prop.threshold, threshold);
    }

    /// Prove: ContractValidation::passing() has all_satisfied=true and empty violations.
    #[kani::unwind(1)]
    #[kani::proof]
    fn contract_validation_passing_invariants() {
        let v = ContractValidation::passing();
        assert!(v.all_satisfied);
        assert!(v.violations.is_empty());
        assert!(v.tightened.is_empty());
    }

    // ========================================================================
    // AnalysisConfig::default proofs
    // ========================================================================

    use crate::bound_analysis_types::AnalysisConfig;

    /// Prove: default AnalysisConfig has positive explosion_threshold.
    #[kani::unwind(1)]
    #[kani::proof]
    fn analysis_config_default_explosion_threshold_positive() {
        let config = AnalysisConfig::default();
        assert!(config.explosion_threshold > 0.0);
        assert!(config.explosion_threshold.is_finite());
    }

    /// Prove: default AnalysisConfig has positive crown_escalation_width.
    #[kani::unwind(1)]
    #[kani::proof]
    fn analysis_config_default_crown_escalation_positive() {
        let config = AnalysisConfig::default();
        assert!(config.crown_escalation_width > 0.0);
        assert!(config.crown_escalation_width.is_finite());
    }

    /// Prove: default AnalysisConfig has smt_max_elements > 0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn analysis_config_default_smt_max_elements_positive() {
        let config = AnalysisConfig::default();
        assert!(config.smt_max_elements > 0);
    }

    /// Prove: default AnalysisConfig norm_chain_min_length > 0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn analysis_config_default_norm_chain_min_length_positive() {
        let config = AnalysisConfig::default();
        assert!(config.norm_chain_min_length > 0);
    }

    /// Prove: default AnalysisConfig precision_risk_drift_threshold is in (0, 1].
    #[kani::unwind(1)]
    #[kani::proof]
    fn analysis_config_default_precision_drift_in_valid_range() {
        let config = AnalysisConfig::default();
        assert!(config.precision_risk_drift_threshold > 0.0);
        assert!(config.precision_risk_drift_threshold <= 1.0);
    }

    /// Prove: explosion_threshold < crown_escalation_width (layered escalation).
    #[kani::unwind(1)]
    #[kani::proof]
    fn analysis_config_default_escalation_ordering() {
        let config = AnalysisConfig::default();
        assert!(
            config.explosion_threshold < config.crown_escalation_width,
            "explosion threshold must be below crown escalation width"
        );
    }

    // ========================================================================
    // KernelVerification::new proofs
    // ========================================================================

    use crate::verify_types::KernelVerification;

    /// Prove: KernelVerification::new preserves kernel_name.
    #[kani::unwind(1)]
    #[kani::proof]
    fn kernel_verification_new_preserves_name() {
        let kv =
            KernelVerification::new("test".to_string(), PropMethod::Crown, -1.0, 1.0, 2.0, true);
        assert_eq!(kv.kernel_name, "test");
    }

    /// Prove: KernelVerification::new preserves method.
    #[kani::unwind(1)]
    #[kani::proof]
    fn kernel_verification_new_preserves_method() {
        let kv =
            KernelVerification::new("k".to_string(), PropMethod::AlphaCrown, 0.0, 1.0, 1.0, true);
        assert_eq!(kv.method, PropMethod::AlphaCrown);
    }

    /// Prove: KernelVerification::new preserves bounds and width.
    #[kani::unwind(1)]
    #[kani::proof]
    fn kernel_verification_new_preserves_bounds() {
        let lo: f32 = kani::any();
        let hi: f32 = kani::any();
        let width: f32 = kani::any();
        kani::assume(lo.is_finite());
        kani::assume(hi.is_finite());
        kani::assume(width.is_finite());

        let kv = KernelVerification::new("k".to_string(), PropMethod::Ibp, lo, hi, width, true);
        assert_eq!(kv.output_lower, lo);
        assert_eq!(kv.output_upper, hi);
        assert_eq!(kv.output_width, width);
        assert!(kv.is_finite);
    }

    /// Prove: KernelVerification::new defaults crown_fallback_reason to None.
    #[kani::unwind(1)]
    #[kani::proof]
    fn kernel_verification_new_default_crown_fallback_none() {
        let kv = KernelVerification::new("k".to_string(), PropMethod::Crown, 0.0, 1.0, 1.0, true);
        assert!(kv.crown_fallback_reason.is_none());
    }

    /// Prove: KernelVerification::new defaults soundness_mode to Sound.
    #[kani::unwind(1)]
    #[kani::proof]
    fn kernel_verification_new_default_soundness_sound() {
        let kv = KernelVerification::new("k".to_string(), PropMethod::Crown, 0.0, 1.0, 1.0, true);
        assert_eq!(kv.soundness_mode, VerificationSoundnessMode::Sound);
    }

    /// Prove: with_soundness_mode builder correctly updates the mode.
    #[kani::unwind(1)]
    #[kani::proof]
    fn kernel_verification_with_soundness_mode() {
        let kv = KernelVerification::new("k".to_string(), PropMethod::Crown, 0.0, 1.0, 1.0, true)
            .with_soundness_mode(VerificationSoundnessMode::Heuristic);
        assert_eq!(kv.soundness_mode, VerificationSoundnessMode::Heuristic);
    }

    // ========================================================================
    // OutputTensorBounds::new proofs
    // ========================================================================

    use crate::verify_types::OutputTensorBounds;

    /// Prove: OutputTensorBounds::new preserves lower, upper, shape.
    #[kani::unwind(1)]
    #[kani::proof]
    fn output_tensor_bounds_new_preserves_fields() {
        let lower = vec![0.0f32, 1.0];
        let upper = vec![2.0f32, 3.0];
        let shape = vec![2usize];

        let bounds = OutputTensorBounds::new(lower.clone(), upper.clone(), shape.clone());
        assert_eq!(bounds.lower, lower);
        assert_eq!(bounds.upper, upper);
        assert_eq!(bounds.shape, shape);
    }

    /// Prove: OutputTensorBounds::new initializes finite_mask to empty.
    #[kani::unwind(1)]
    #[kani::proof]
    fn output_tensor_bounds_new_empty_finite_mask() {
        let bounds = OutputTensorBounds::new(vec![], vec![], vec![]);
        assert!(bounds.finite_mask.is_empty());
    }

    // ========================================================================
    // VerifyConfig::default proofs
    // ========================================================================

    /// Prove: VerifyConfig::default has positive finite escalation threshold.
    #[kani::unwind(1)]
    #[kani::proof]
    fn verify_config_default_threshold_positive_finite() {
        let config = crate::verify_types::VerifyConfig::default();
        let threshold = config.escalation_threshold();
        assert!(threshold > 0.0);
        assert!(threshold.is_finite());
    }

    /// Prove: VerifyConfig::default does not require sound.
    #[kani::unwind(1)]
    #[kani::proof]
    fn verify_config_default_not_require_sound() {
        let config = crate::verify_types::VerifyConfig::default();
        assert!(!config.require_sound());
    }

    /// Prove: VerifyConfig::default has ForwardMode norm mode.
    #[kani::unwind(1)]
    #[kani::proof]
    fn verify_config_default_norm_mode_forward() {
        let config = crate::verify_types::VerifyConfig::default();
        assert_eq!(config.norm_mode(), NormBoundsMode::ForwardMode);
    }

    /// Prove: VerifyConfig::default does not collect layer bounds.
    #[kani::unwind(1)]
    #[kani::proof]
    fn verify_config_default_no_collect_layer_bounds() {
        let config = crate::verify_types::VerifyConfig::default();
        assert!(!config.collect_layer_bounds());
    }

    // ========================================================================
    // ParamInputRecord::new proofs
    // ========================================================================

    use crate::status::ParamInputRecord;

    /// Prove: ParamInputRecord::new preserves all fields.
    #[kani::unwind(1)]
    #[kani::proof]
    fn param_input_record_new_preserves_fields() {
        let idx: usize = kani::any();
        kani::assume(idx < 1000); // bound for tractability
        let lo: f32 = kani::any();
        let hi: f32 = kani::any();
        kani::assume(lo.is_finite());
        kani::assume(hi.is_finite());

        let record = ParamInputRecord::new(idx, lo, hi);
        assert_eq!(record.param_index, idx);
        assert_eq!(record.lower, lo);
        assert_eq!(record.upper, hi);
    }

    // ========================================================================
    // DEFAULT_ESCALATION_THRESHOLD proof
    // ========================================================================

    /// Prove: DEFAULT_ESCALATION_THRESHOLD equals VerifyConfig::default().escalation_threshold().
    #[kani::unwind(1)]
    #[kani::proof]
    fn default_escalation_threshold_matches_config() {
        let config = crate::verify_types::VerifyConfig::default();
        assert_eq!(
            config.escalation_threshold(),
            crate::verify_types::DEFAULT_ESCALATION_THRESHOLD,
        );
    }
}
