// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fusion specification types and parameter validation.
//!
//! Pure data types and validation logic for fusion equivalence checking.
//! The core fusion algorithm lives in [`super::fusion`].

use ny_core::VerificationSoundnessMode;
use nn_dsl::ir::KernelDef;

use crate::error::{StructuralError, VerifyError};
use crate::verify::PropMethod;

/// Specification for a fusion equivalence check: the three kernels and their
/// parameter mapping.
///
/// Bundles the kernel-triple and index arrays that describe how a fused kernel
/// relates to its sequential components, reducing `verify_fusion_equivalence`
/// from 9 parameters to 4.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FusionSpec<'a> {
    /// The fused kernel (N params, all become variables in the diamond DAG).
    pub fused: &'a KernelDef,
    /// First kernel in the sequential path.
    pub first: &'a KernelDef,
    /// Second kernel in the sequential path.
    pub second: &'a KernelDef,
    /// Total number of shared input variables.
    pub num_shared_inputs: usize,
    /// Maps each param of `first` to a shared input index.
    pub first_param_indices: &'a [usize],
    /// Maps each param of `second` to a shared input index
    /// (the entry at `second_input_from_first` is ignored).
    pub second_param_indices: &'a [usize],
    /// Which param of `second` receives `first`'s output.
    pub second_input_from_first: usize,
}

impl<'a> FusionSpec<'a> {
    /// Construct a validated `FusionSpec`.
    ///
    /// Checks length and index-range consistency of parameter mappings.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralError::FusionParam`] (via [`VerifyError::Structural`]) if any mapping is inconsistent.
    #[must_use = "returns a Result that may contain an error"]
    pub fn new(
        fused: &'a KernelDef,
        first: &'a KernelDef,
        second: &'a KernelDef,
        num_shared_inputs: usize,
        first_param_indices: &'a [usize],
        second_param_indices: &'a [usize],
        second_input_from_first: usize,
    ) -> Result<Self, VerifyError> {
        let spec = Self {
            fused,
            first,
            second,
            num_shared_inputs,
            first_param_indices,
            second_param_indices,
            second_input_from_first,
        };
        validate_fusion_params(&spec)?;
        Ok(spec)
    }
}

/// Result of fusion equivalence verification.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[must_use]
pub struct FusionVerification {
    /// Name of the fused kernel.
    pub fused_kernel_name: String,
    /// Lower bound on (fused - sequential) output.
    pub diff_lower: f32,
    /// Upper bound on (fused - sequential) output.
    pub diff_upper: f32,
    /// Maximum absolute difference bound: max(|diff_lower|, |diff_upper|).
    pub max_abs_diff: f32,
    /// Whether the diff is provably within the epsilon budget.
    pub within_epsilon: bool,
    /// The epsilon threshold used.
    pub epsilon: f32,
    /// Propagation method that produced these bounds.
    pub method: PropMethod,
    /// If CROWN failed and we fell back to IBP, the error reason.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crown_fallback_reason: Option<String>,
    /// Soundness classification of the propagation result.
    ///
    /// Per design doc #377: soundness provenance must be persisted end-to-end.
    /// `Sound` means all ops in the fusion graph used exact/sound approximations.
    /// `Heuristic` means at least one op used a heuristic (e.g., SqrtNegativeDomain).
    /// Defaults to `Heuristic` for legacy deserialization (fail-closed per #377).
    #[serde(default = "default_heuristic")]
    pub soundness_mode: VerificationSoundnessMode,
}

fn default_heuristic() -> VerificationSoundnessMode {
    VerificationSoundnessMode::Heuristic
}

impl FusionVerification {
    /// Whether this verification result is conclusive.
    ///
    /// IBP produces arbitrarily wide Minkowski-difference bounds for diff
    /// graphs (see `fusion.rs` diamond DAG comment), so `within_epsilon`
    /// can be `false` even for a correct fusion. Only CROWN results are
    /// tight enough to be conclusive.
    ///
    /// Callers should use `result.is_conclusive() && result.within_epsilon`
    /// to avoid false alarms from IBP fallback results.
    pub fn is_conclusive(&self) -> bool {
        self.method.is_tight()
    }
}

/// Validate fusion parameter arrays for length and index-range consistency.
pub(crate) fn validate_fusion_params(spec: &FusionSpec<'_>) -> Result<(), VerifyError> {
    let fusion_err = |msg: String| VerifyError::from(StructuralError::FusionParam { context: msg });
    if spec.first_param_indices.len() != spec.first.params.len() {
        return Err(fusion_err(format!(
            "first_param_indices length ({}) != first kernel param count ({})",
            spec.first_param_indices.len(),
            spec.first.params.len(),
        )));
    }
    if spec.second_param_indices.len() != spec.second.params.len() {
        return Err(fusion_err(format!(
            "second_param_indices length ({}) != second kernel param count ({})",
            spec.second_param_indices.len(),
            spec.second.params.len(),
        )));
    }
    if spec.second_input_from_first >= spec.second.params.len() {
        return Err(fusion_err(format!(
            "second_input_from_first ({}) >= second param count ({})",
            spec.second_input_from_first,
            spec.second.params.len(),
        )));
    }
    if spec.fused.params.len() != spec.num_shared_inputs {
        return Err(fusion_err(format!(
            "fused param count ({}) != num_shared_inputs ({})",
            spec.fused.params.len(),
            spec.num_shared_inputs,
        )));
    }
    for (i, &idx) in spec.first_param_indices.iter().enumerate() {
        if idx >= spec.num_shared_inputs {
            return Err(fusion_err(format!(
                "first_param_indices[{i}] = {idx} >= num_shared_inputs ({})",
                spec.num_shared_inputs,
            )));
        }
    }
    for (i, &idx) in spec.second_param_indices.iter().enumerate() {
        if i != spec.second_input_from_first && idx >= spec.num_shared_inputs {
            return Err(fusion_err(format!(
                "second_param_indices[{i}] = {idx} >= num_shared_inputs ({})",
                spec.num_shared_inputs,
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::PropMethod;

    /// Helper to build a FusionVerification with the given method.
    fn make_verification(method: PropMethod) -> FusionVerification {
        FusionVerification {
            fused_kernel_name: "test".to_string(),
            diff_lower: -0.01,
            diff_upper: 0.01,
            max_abs_diff: 0.01,
            within_epsilon: true,
            epsilon: 0.05,
            method,
            crown_fallback_reason: None,
            soundness_mode: VerificationSoundnessMode::Sound,
        }
    }

    #[test]
    fn test_is_conclusive_crown_returns_true() {
        let v = make_verification(PropMethod::Crown);
        assert!(v.is_conclusive(), "CROWN results should be conclusive");
    }

    #[test]
    fn test_is_conclusive_ibp_returns_false() {
        let v = make_verification(PropMethod::Ibp);
        assert!(
            !v.is_conclusive(),
            "IBP results should NOT be conclusive (Minkowski-difference bounds are vacuously wide)"
        );
    }

    #[test]
    fn test_ibp_within_epsilon_true_but_not_conclusive() {
        // An IBP result can say within_epsilon: true, but is_conclusive
        // is still false. Callers should check BOTH.
        let v = make_verification(PropMethod::Ibp);
        assert!(
            v.within_epsilon,
            "test setup: within_epsilon should be true"
        );
        assert!(
            !v.is_conclusive(),
            "IBP within_epsilon should not be trusted as conclusive"
        );
    }

    // F9: is_conclusive must accept all CROWN-family methods, not just Crown.
    // AlphaCrown/BetaCrown are tighter than Crown; Analytical is exact.
    #[test]
    fn test_is_conclusive_alpha_crown() {
        let v = make_verification(PropMethod::AlphaCrown);
        assert!(v.is_conclusive(), "AlphaCrown results should be conclusive");
    }

    #[test]
    fn test_is_conclusive_beta_crown() {
        let v = make_verification(PropMethod::BetaCrown);
        assert!(v.is_conclusive(), "BetaCrown results should be conclusive");
    }

    #[test]
    fn test_is_conclusive_analytical() {
        let v = make_verification(PropMethod::Analytical);
        assert!(v.is_conclusive(), "Analytical results should be conclusive");
    }

    #[test]
    fn test_is_conclusive_mixed_ibp_crown_not_conclusive() {
        let v = make_verification(PropMethod::MixedIbpCrown);
        assert!(!v.is_conclusive(), "MixedIbpCrown should NOT be conclusive");
    }

    #[test]
    fn test_soundness_mode_preserved_in_struct() {
        let v = make_verification(PropMethod::Crown);
        assert_eq!(
            v.soundness_mode,
            VerificationSoundnessMode::Sound,
            "soundness_mode should be preserved from construction"
        );

        // Heuristic mode should also be preserved.
        let mut v_heuristic = make_verification(PropMethod::Crown);
        v_heuristic.soundness_mode = VerificationSoundnessMode::Heuristic;
        assert_eq!(
            v_heuristic.soundness_mode,
            VerificationSoundnessMode::Heuristic,
            "Heuristic soundness_mode should be preserved"
        );
    }

    #[test]
    fn test_soundness_mode_serde_roundtrip() {
        let v = make_verification(PropMethod::Crown);
        let json = serde_json::to_string(&v).expect("serialize");
        let v2: FusionVerification = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            v.soundness_mode, v2.soundness_mode,
            "soundness_mode survives serde roundtrip"
        );
    }
}
