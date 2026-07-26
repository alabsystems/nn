// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compatibility shim for NY types used in always-available modules.
//!
//! When the `NY` feature is enabled, re-exports from gamma crates.
//! When disabled, provides local definitions with identical serde layout
//! so that `status.rs`, `certificate.rs`, and `error.rs` can function
//! without the NY dependency.
//!
//! Part of #864 — NY optional feature flag.

// ---------------------------------------------------------------------------
// VerificationSoundnessMode
// ---------------------------------------------------------------------------

#[cfg(feature = "ny")]
pub use ny_core::VerificationSoundnessMode;

#[cfg(not(feature = "ny"))]
mod soundness_local {
    use serde::{Deserialize, Serialize};

    /// Whether the verification run used only sound techniques.
    ///
    /// Local mirror of `ny_core::VerificationSoundnessMode` for use
    /// when the `NY` feature is disabled. Serde layout is
    /// identical (`rename_all = "snake_case"`) so status JSON files
    /// round-trip correctly.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    #[non_exhaustive]
    pub enum VerificationSoundnessMode {
        /// No known heuristic/unsound switches were enabled.
        Sound,
        /// At least one heuristic/approximation that weakens proof semantics was used.
        Heuristic,
    }
}

#[cfg(not(feature = "ny"))]
pub use soundness_local::VerificationSoundnessMode;

/// Default soundness mode for legacy JSON without the field.
///
/// Returns `Heuristic` (fail-closed): old results that were truly sound
/// can be re-verified to restore `Sound`.
pub(crate) fn default_soundness_mode() -> VerificationSoundnessMode {
    VerificationSoundnessMode::Heuristic
}
