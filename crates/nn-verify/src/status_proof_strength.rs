// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Proof strength classification for verification status entries.
//!
//! Extracted from `status.rs` to keep the parent file under 450 lines.

use serde::{Deserialize, Serialize};

use crate::soundness_compat::VerificationSoundnessMode;
use crate::verify_types::PropMethod;

/// Proof strength classification combining soundness mode and output width.
///
/// Distinguishes entries by practical verification quality (#2650):
/// - `SoundCrown`: sound CROWN bounds (no heuristics, tight bounds)
/// - `SoundIbp`: sound IBP bounds (no heuristics, but may be wider than CROWN)
/// - `Heuristic`: heuristic bounds with reasonable width (output_width <= 100)
/// - `Vacuous`: bounds too wide for practical use (output_width > 100)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ProofStrength {
    /// Sound CROWN verification — no heuristics used, CROWN method.
    SoundCrown,
    /// Sound IBP verification — no heuristics used, IBP-only method.
    SoundIbp,
    /// Heuristic bounds with reasonable width (output_width <= 100).
    Heuristic,
    /// Bounds too wide for practical use (output_width > 100).
    Vacuous,
    /// Sound mixed-mode: IBP for intractable layers, CROWN for tractable ones.
    /// Both components are sound (no heuristic normalization approximations).
    SoundMixed,
}

/// Threshold for vacuous bounds: entries with output_width > this are `Vacuous`.
pub const VACUOUS_WIDTH_THRESHOLD: f32 = 100.0;

/// Compute proof strength from soundness mode, method, and output width.
#[must_use]
pub fn compute_proof_strength(
    soundness_mode: VerificationSoundnessMode,
    method: PropMethod,
    output_width: f32,
) -> ProofStrength {
    if output_width > VACUOUS_WIDTH_THRESHOLD {
        return ProofStrength::Vacuous;
    }
    match soundness_mode {
        VerificationSoundnessMode::Sound => {
            if method.is_tight() {
                ProofStrength::SoundCrown
            } else if method == PropMethod::MixedIbpCrown {
                ProofStrength::SoundMixed
            } else {
                ProofStrength::SoundIbp
            }
        }
        VerificationSoundnessMode::Heuristic => ProofStrength::Heuristic,
    }
}
