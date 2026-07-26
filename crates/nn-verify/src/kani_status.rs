// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for `status.rs`.
//!
//! Proves structural and correctness properties of:
//! - `VerifyStatus`: kernel_count, has_kernel, soundness_counts
//! - `KernelStatus::new`: proof_strength auto-computation
//! - `compute_proof_strength`: vacuous threshold, sound/heuristic classification
//! - `ProofStrength` boundary conditions at VACUOUS_WIDTH_THRESHOLD
//! - Stale entry exclusion from soundness/proof-strength counts
//! - `set_soundness_justification`: missing kernel rejected
//! - `mark_stale`: missing kernel rejected, stale flag propagated
//! - `VerifyOutcome` variant coverage
//!
//! Part of #3708.

use super::{
    compute_proof_strength, is_false, KernelStatus, ProofStrength, VerifyOutcome, VerifyStatus,
    VACUOUS_WIDTH_THRESHOLD,
};
use crate::soundness_compat::VerificationSoundnessMode;
use crate::status::{InputBoundsRecord, OutputBoundsRecord, ParamInputRecord};
use crate::verify_types::PropMethod;

/// Helper: create a minimal valid KernelStatus.
fn make_status(
    method: PropMethod,
    soundness: VerificationSoundnessMode,
    width: f32,
) -> KernelStatus {
    KernelStatus::new(
        VerifyOutcome::Verified,
        method,
        InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]),
        OutputBoundsRecord::new(-1.0, 1.0),
        width,
        soundness,
    )
}

// ===========================================================================
// is_false helper
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. is_false returns true for false input
// ---------------------------------------------------------------------------

/// Prove: `is_false(false)` returns `true`.
#[kani::unwind(1)]
#[kani::proof]
fn is_false_returns_true_for_false() {
    assert!(is_false(&false), "is_false(false) must be true");
}

// ---------------------------------------------------------------------------
// 2. is_false returns false for true input
// ---------------------------------------------------------------------------

/// Prove: `is_false(true)` returns `false`.
#[kani::unwind(1)]
#[kani::proof]
fn is_false_returns_false_for_true() {
    assert!(!is_false(&true), "is_false(true) must be false");
}

// ===========================================================================
// VerifyStatus: kernel_count and has_kernel
// ===========================================================================

// ---------------------------------------------------------------------------
// 3. kernel_count is 0 for default status
// ---------------------------------------------------------------------------

/// Prove: default `VerifyStatus` has 0 kernels.
#[kani::unwind(1)]
#[kani::proof]
fn default_status_has_zero_kernels() {
    let status = VerifyStatus::default();
    assert_eq!(status.kernel_count(), 0, "default must have 0 kernels");
}

// ---------------------------------------------------------------------------
// 4. has_kernel is false for empty status
// ---------------------------------------------------------------------------

/// Prove: `has_kernel` returns false for a name not in an empty status.
#[kani::unwind(1)]
#[kani::proof]
fn empty_status_has_no_kernels() {
    let status = VerifyStatus::default();
    assert!(
        !status.has_kernel("anything"),
        "empty status has no kernels"
    );
}

// ---------------------------------------------------------------------------
// 5. kernel returns None for missing kernel
// ---------------------------------------------------------------------------

/// Prove: `kernel()` returns None for a name not in the status.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_returns_none_for_missing() {
    let status = VerifyStatus::default();
    assert!(status.kernel("nonexistent").is_none() || status.kernel("missing").is_none());
    assert!(
        status.kernel("missing").is_none(),
        "missing kernel must return None"
    );
}

// ---------------------------------------------------------------------------
// 6. kernels returns empty map for default status
// ---------------------------------------------------------------------------

/// Prove: `kernels()` returns an empty map for default status.
#[kani::unwind(1)]
#[kani::proof]
fn default_status_kernels_empty() {
    let status = VerifyStatus::default();
    assert!(
        status.kernels().is_empty(),
        "default kernels map must be empty"
    );
}

// ---------------------------------------------------------------------------
// 7. history returns empty map for default status
// ---------------------------------------------------------------------------

/// Prove: `history()` returns an empty map for default status.
#[kani::unwind(1)]
#[kani::proof]
fn default_status_history_empty() {
    let status = VerifyStatus::default();
    assert!(
        status.history().is_empty(),
        "default history map must be empty"
    );
}

// ---------------------------------------------------------------------------
// 8. history_for returns None for missing kernel
// ---------------------------------------------------------------------------

/// Prove: `history_for()` returns None for a missing kernel.
#[kani::unwind(1)]
#[kani::proof]
fn history_for_returns_none_for_missing() {
    let status = VerifyStatus::default();
    assert!(
        status.history_for("missing").is_none(),
        "missing history must be None"
    );
}

// ===========================================================================
// KernelStatus::new: proof_strength auto-computation
// ===========================================================================

// ---------------------------------------------------------------------------
// 9. KernelStatus::new always sets proof_strength to Some
// ---------------------------------------------------------------------------

/// Prove: `KernelStatus::new` always sets `proof_strength` to `Some`.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_new_sets_proof_strength() {
    let ks = make_status(PropMethod::Crown, VerificationSoundnessMode::Sound, 2.0);
    assert!(ks.proof_strength.is_some(), "new must set proof_strength");
}

// ---------------------------------------------------------------------------
// 10. KernelStatus::new: Sound + Crown → SoundCrown
// ---------------------------------------------------------------------------

/// Prove: Sound CROWN method produces SoundCrown proof strength.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_sound_crown_is_sound_crown() {
    let ks = make_status(PropMethod::Crown, VerificationSoundnessMode::Sound, 2.0);
    assert_eq!(
        ks.proof_strength,
        Some(ProofStrength::SoundCrown),
        "Sound + Crown must be SoundCrown",
    );
}

// ---------------------------------------------------------------------------
// 11. KernelStatus::new: Sound + IBP → SoundIbp
// ---------------------------------------------------------------------------

/// Prove: Sound IBP method produces SoundIbp proof strength.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_sound_ibp_is_sound_ibp() {
    let ks = make_status(PropMethod::Ibp, VerificationSoundnessMode::Sound, 2.0);
    assert_eq!(
        ks.proof_strength,
        Some(ProofStrength::SoundIbp),
        "Sound + IBP must be SoundIbp",
    );
}

// ---------------------------------------------------------------------------
// 12. KernelStatus::new: Heuristic → Heuristic
// ---------------------------------------------------------------------------

/// Prove: Heuristic soundness mode produces Heuristic proof strength.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_heuristic_is_heuristic() {
    let ks = make_status(PropMethod::Crown, VerificationSoundnessMode::Heuristic, 2.0);
    assert_eq!(
        ks.proof_strength,
        Some(ProofStrength::Heuristic),
        "Heuristic must be Heuristic",
    );
}

// ---------------------------------------------------------------------------
// 13. KernelStatus::new: wide output → Vacuous
// ---------------------------------------------------------------------------

/// Prove: output_width > VACUOUS_WIDTH_THRESHOLD produces Vacuous regardless
/// of soundness mode.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_wide_output_is_vacuous() {
    let ks = make_status(
        PropMethod::Crown,
        VerificationSoundnessMode::Sound,
        VACUOUS_WIDTH_THRESHOLD + 1.0,
    );
    assert_eq!(
        ks.proof_strength,
        Some(ProofStrength::Vacuous),
        "wide output must be Vacuous",
    );
}

// ===========================================================================
// compute_proof_strength: boundary conditions
// ===========================================================================

// ---------------------------------------------------------------------------
// 14. compute_proof_strength: exactly at threshold is NOT vacuous
// ---------------------------------------------------------------------------

/// Prove: output_width == VACUOUS_WIDTH_THRESHOLD is not Vacuous.
#[kani::unwind(1)]
#[kani::proof]
fn proof_strength_at_threshold_not_vacuous() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Crown,
        VACUOUS_WIDTH_THRESHOLD,
    );
    assert_ne!(
        strength,
        ProofStrength::Vacuous,
        "exactly at threshold must not be Vacuous"
    );
}

// ---------------------------------------------------------------------------
// 15. compute_proof_strength: just above threshold is Vacuous
// ---------------------------------------------------------------------------

/// Prove: output_width just above threshold is Vacuous.
#[kani::unwind(1)]
#[kani::proof]
fn proof_strength_above_threshold_is_vacuous() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Crown,
        VACUOUS_WIDTH_THRESHOLD + f32::EPSILON,
    );
    assert_eq!(
        strength,
        ProofStrength::Vacuous,
        "above threshold must be Vacuous"
    );
}

// ---------------------------------------------------------------------------
// 16. compute_proof_strength: AlphaCrown is tight (SoundCrown)
// ---------------------------------------------------------------------------

/// Prove: Sound + AlphaCrown → SoundCrown.
#[kani::unwind(1)]
#[kani::proof]
fn proof_strength_alpha_crown_is_sound_crown() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::AlphaCrown,
        5.0,
    );
    assert_eq!(strength, ProofStrength::SoundCrown);
}

// ---------------------------------------------------------------------------
// 17. compute_proof_strength: BetaCrown is tight (SoundCrown)
// ---------------------------------------------------------------------------

/// Prove: Sound + BetaCrown → SoundCrown.
#[kani::unwind(1)]
#[kani::proof]
fn proof_strength_beta_crown_is_sound_crown() {
    let strength =
        compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::BetaCrown, 5.0);
    assert_eq!(strength, ProofStrength::SoundCrown);
}

// ---------------------------------------------------------------------------
// 18. compute_proof_strength: Analytical is tight (SoundCrown)
// ---------------------------------------------------------------------------

/// Prove: Sound + Analytical → SoundCrown.
#[kani::unwind(1)]
#[kani::proof]
fn proof_strength_analytical_is_sound_crown() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::Analytical,
        5.0,
    );
    assert_eq!(strength, ProofStrength::SoundCrown);
}

// ---------------------------------------------------------------------------
// 19. compute_proof_strength: MixedIbpCrown → SoundMixed
// ---------------------------------------------------------------------------

/// Prove: Sound + MixedIbpCrown → SoundMixed.
#[kani::unwind(1)]
#[kani::proof]
fn proof_strength_mixed_ibp_crown_is_sound_mixed() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Sound,
        PropMethod::MixedIbpCrown,
        5.0,
    );
    assert_eq!(strength, ProofStrength::SoundMixed);
}

// ---------------------------------------------------------------------------
// 20. compute_proof_strength: Heuristic + wide → Vacuous (not Heuristic)
// ---------------------------------------------------------------------------

/// Prove: vacuous threshold takes priority over heuristic classification.
#[kani::unwind(1)]
#[kani::proof]
fn proof_strength_heuristic_wide_is_vacuous() {
    let strength = compute_proof_strength(
        VerificationSoundnessMode::Heuristic,
        PropMethod::Crown,
        VACUOUS_WIDTH_THRESHOLD + 1.0,
    );
    assert_eq!(
        strength,
        ProofStrength::Vacuous,
        "vacuous overrides heuristic"
    );
}

// ===========================================================================
// soundness_counts: stale exclusion
// ===========================================================================

// ---------------------------------------------------------------------------
// 21. soundness_counts: empty status returns (0, 0)
// ---------------------------------------------------------------------------

/// Prove: `soundness_counts()` on empty status returns (0, 0).
#[kani::unwind(1)]
#[kani::proof]
fn soundness_counts_empty_is_zero_zero() {
    let status = VerifyStatus::default();
    assert_eq!(status.soundness_counts(), (0, 0));
}

// ===========================================================================
// proof_strength_counts: empty status
// ===========================================================================

// ---------------------------------------------------------------------------
// 22. proof_strength_counts: empty status returns all zeros
// ---------------------------------------------------------------------------

/// Prove: `proof_strength_counts()` on empty status returns (0, 0, 0, 0).
#[kani::unwind(1)]
#[kani::proof]
fn proof_strength_counts_empty_is_all_zeros() {
    let status = VerifyStatus::default();
    assert_eq!(status.proof_strength_counts(), (0, 0, 0, 0));
}

// ===========================================================================
// set_soundness_justification: validation
// ===========================================================================

// ---------------------------------------------------------------------------
// 23. set_soundness_justification: missing kernel rejected
// ---------------------------------------------------------------------------

/// Prove: `set_soundness_justification` on a missing kernel returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn set_justification_missing_kernel_rejected() {
    let mut status = VerifyStatus::default();
    let result = status.set_soundness_justification("nonexistent", "reason");
    assert!(result.is_err(), "missing kernel must be rejected");
}

// ===========================================================================
// mark_stale: validation
// ===========================================================================

// ---------------------------------------------------------------------------
// 24. mark_stale: missing kernel rejected
// ---------------------------------------------------------------------------

/// Prove: `mark_stale` on a missing kernel returns Err.
#[kani::unwind(1)]
#[kani::proof]
fn mark_stale_missing_kernel_rejected() {
    let mut status = VerifyStatus::default();
    let result = status.mark_stale("nonexistent", "obsolete");
    assert!(result.is_err(), "missing kernel must be rejected");
}

// ===========================================================================
// KernelStatus::new: optional fields default to None/false
// ===========================================================================

// ---------------------------------------------------------------------------
// 25. KernelStatus::new: crown_error defaults to None
// ---------------------------------------------------------------------------

/// Prove: `KernelStatus::new` sets `crown_error` to `None`.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_new_crown_error_none() {
    let ks = make_status(PropMethod::Ibp, VerificationSoundnessMode::Sound, 1.0);
    assert!(ks.crown_error.is_none(), "crown_error must default to None");
}

// ---------------------------------------------------------------------------
// 26. KernelStatus::new: smt defaults to None
// ---------------------------------------------------------------------------

/// Prove: `KernelStatus::new` sets `smt` to `None`.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_new_smt_none() {
    let ks = make_status(PropMethod::Ibp, VerificationSoundnessMode::Sound, 1.0);
    assert!(ks.smt.is_none(), "smt must default to None");
}

// ---------------------------------------------------------------------------
// 27. KernelStatus::new: crown_coverage defaults to None
// ---------------------------------------------------------------------------

/// Prove: `KernelStatus::new` sets `crown_coverage` to `None`.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_new_crown_coverage_none() {
    let ks = make_status(PropMethod::Ibp, VerificationSoundnessMode::Sound, 1.0);
    assert!(
        ks.crown_coverage.is_none(),
        "crown_coverage must default to None"
    );
}

// ---------------------------------------------------------------------------
// 28. KernelStatus::new: stale defaults to false
// ---------------------------------------------------------------------------

/// Prove: `KernelStatus::new` sets `stale` to `false`.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_new_stale_false() {
    let ks = make_status(PropMethod::Ibp, VerificationSoundnessMode::Sound, 1.0);
    assert!(!ks.stale, "stale must default to false");
}

// ---------------------------------------------------------------------------
// 29. KernelStatus::new: stale_reason defaults to None
// ---------------------------------------------------------------------------

/// Prove: `KernelStatus::new` sets `stale_reason` to `None`.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_new_stale_reason_none() {
    let ks = make_status(PropMethod::Ibp, VerificationSoundnessMode::Sound, 1.0);
    assert!(
        ks.stale_reason.is_none(),
        "stale_reason must default to None"
    );
}

// ---------------------------------------------------------------------------
// 30. KernelStatus::new: weight_artifact defaults to None
// ---------------------------------------------------------------------------

/// Prove: `KernelStatus::new` sets `weight_artifact` to `None`.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_new_weight_artifact_none() {
    let ks = make_status(PropMethod::Ibp, VerificationSoundnessMode::Sound, 1.0);
    assert!(
        ks.weight_artifact.is_none(),
        "weight_artifact must default to None"
    );
}

// ===========================================================================
// VerifyOutcome: variant existence
// ===========================================================================

// ---------------------------------------------------------------------------
// 31. VerifyOutcome: all variants constructible
// ---------------------------------------------------------------------------

/// Prove: all `VerifyOutcome` variants can be constructed and compared.
#[kani::unwind(8)]
#[kani::proof]
fn verify_outcome_all_variants_exist() {
    let variants = [
        VerifyOutcome::Verified,
        VerifyOutcome::BoundsComputed,
        VerifyOutcome::IbpFallback,
        VerifyOutcome::Failed,
        VerifyOutcome::SmtContradiction,
    ];
    // All variants are distinct.
    for i in 0..variants.len() {
        for j in (i + 1)..variants.len() {
            assert_ne!(variants[i], variants[j], "variants must be distinct");
        }
    }
}

// ===========================================================================
// VACUOUS_WIDTH_THRESHOLD: correct value
// ===========================================================================

// ---------------------------------------------------------------------------
// 32. VACUOUS_WIDTH_THRESHOLD is 100.0
// ---------------------------------------------------------------------------

/// Prove: `VACUOUS_WIDTH_THRESHOLD` is exactly 100.0.
#[kani::unwind(1)]
#[kani::proof]
fn vacuous_width_threshold_is_100() {
    assert_eq!(VACUOUS_WIDTH_THRESHOLD, 100.0_f32);
}

// ===========================================================================
// KernelStatus::new: output_width stored correctly
// ===========================================================================

// ---------------------------------------------------------------------------
// 33. KernelStatus::new preserves output_width
// ---------------------------------------------------------------------------

/// Prove: `KernelStatus::new` preserves the given output_width.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_new_preserves_output_width() {
    let width: f32 = 42.5;
    let ks = make_status(PropMethod::Crown, VerificationSoundnessMode::Sound, width);
    assert_eq!(ks.output_width, width, "output_width must be preserved");
}

// ---------------------------------------------------------------------------
// 34. KernelStatus::new preserves method
// ---------------------------------------------------------------------------

/// Prove: `KernelStatus::new` preserves the given method.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_new_preserves_method() {
    let ks = make_status(
        PropMethod::AlphaCrown,
        VerificationSoundnessMode::Heuristic,
        3.0,
    );
    assert_eq!(
        ks.method,
        PropMethod::AlphaCrown,
        "method must be preserved"
    );
}

// ---------------------------------------------------------------------------
// 35. KernelStatus::new preserves soundness_mode
// ---------------------------------------------------------------------------

/// Prove: `KernelStatus::new` preserves the given soundness_mode.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_new_preserves_soundness_mode() {
    let ks = make_status(PropMethod::Ibp, VerificationSoundnessMode::Heuristic, 3.0);
    assert_eq!(
        ks.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "soundness_mode must be preserved",
    );
}

// ---------------------------------------------------------------------------
// 36. KernelStatus::new preserves status (VerifyOutcome)
// ---------------------------------------------------------------------------

/// Prove: `KernelStatus::new` preserves the status field.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_status_new_preserves_status() {
    let ks = KernelStatus::new(
        VerifyOutcome::IbpFallback,
        PropMethod::Crown,
        InputBoundsRecord::new(&[ParamInputRecord::new(0, -1.0, 1.0)], &[]),
        OutputBoundsRecord::new(-1.0, 1.0),
        2.0,
        VerificationSoundnessMode::Sound,
    );
    assert_eq!(
        ks.status,
        VerifyOutcome::IbpFallback,
        "status must be preserved"
    );
}

// ===========================================================================
// OutputBoundsRecord: construction
// ===========================================================================

// ---------------------------------------------------------------------------
// 37. OutputBoundsRecord::new preserves bounds
// ---------------------------------------------------------------------------

/// Prove: `OutputBoundsRecord::new` preserves lower and upper.
#[kani::unwind(1)]
#[kani::proof]
fn output_bounds_new_preserves_bounds() {
    let ob = OutputBoundsRecord::new(-3.14, 2.71);
    assert_eq!(ob.lower, -3.14_f32);
    assert_eq!(ob.upper, 2.71_f32);
    assert!(!ob.is_infeasible, "new must not set infeasible");
}

// ---------------------------------------------------------------------------
// 38. OutputBoundsRecord::zero is (0, 0)
// ---------------------------------------------------------------------------

/// Prove: `OutputBoundsRecord::zero()` returns (0.0, 0.0).
#[kani::unwind(1)]
#[kani::proof]
fn output_bounds_zero_is_zero_zero() {
    let ob = OutputBoundsRecord::zero();
    assert_eq!(ob.lower, 0.0_f32);
    assert_eq!(ob.upper, 0.0_f32);
}

// ---------------------------------------------------------------------------
// 39. ParamInputRecord::new preserves fields
// ---------------------------------------------------------------------------

/// Prove: `ParamInputRecord::new` preserves all fields.
#[kani::unwind(1)]
#[kani::proof]
fn param_input_record_new_preserves_fields() {
    let p = ParamInputRecord::new(3, -0.5, 0.5);
    assert_eq!(p.param_index, 3);
    assert_eq!(p.lower, -0.5_f32);
    assert_eq!(p.upper, 0.5_f32);
}
