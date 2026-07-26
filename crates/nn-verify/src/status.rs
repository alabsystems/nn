// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verification status persistence to `nn_verify_status.json`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::AtomicU64;

use serde::{Deserialize, Serialize};

use crate::soundness_compat::VerificationSoundnessMode;

use crate::error::VerifyError;
pub use crate::status_smt::{
    BoundsSource, SmtEncodingKind, SmtOutcome, SmtProofVerdict, SmtStatusRecord,
};
#[cfg(feature = "ny")]
use crate::verify_input::ScalarInputBounds;
use crate::verify_types::{KernelVerification, PropMethod};

#[path = "status_helpers.rs"]
mod status_helpers;
use status_helpers::{atomic_tmp_path, sync_directory, validate_input_metadata, StatusFileLock};

#[path = "status_recording.rs"]
mod status_recording;

#[path = "status_crown_comparison.rs"]
mod status_crown_comparison;

#[path = "status_per_model.rs"]
mod status_per_model;
pub use status_per_model::{model_for_kernel, model_status_path, MODEL_CATEGORIES};

static SAVE_NONCE: AtomicU64 = AtomicU64::new(0);

/// Helper for `#[serde(skip_serializing_if = "is_false")]`.
fn is_false(v: &bool) -> bool {
    !*v
}

/// Top-level verification status file. `kernels` holds the latest result per
/// kernel (overwrites on record). `history` holds recent runs (max 10 per kernel).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VerifyStatus {
    /// Latest verification result per kernel name.
    kernels: BTreeMap<String, KernelStatus>,
    /// Recent verification history per kernel name (max 10 entries per kernel).
    /// Older entries are discarded on record and load (#538).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    history: BTreeMap<String, Vec<KernelStatus>>,
}

/// Verification status for a single kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KernelStatus {
    pub status: VerifyOutcome,
    pub method: PropMethod,
    pub input_bounds: InputBoundsRecord,
    pub output_bounds: OutputBoundsRecord,
    pub output_width: f32,
    /// If CROWN was attempted but failed, the error text is preserved here.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crown_error: Option<String>,
    /// Soundness classification: `Sound` if no heuristics were used,
    /// `Heuristic` if approximations weakened proof semantics.
    #[serde(default = "crate::soundness_compat::default_soundness_mode")]
    pub soundness_mode: VerificationSoundnessMode,
    /// Optional SMT verification result (ay backend); absent for legacy/NY-only runs.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub smt: Option<SmtStatusRecord>,
    /// CROWN coverage ratio (fraction of layers verified with CROWN).
    /// None for scalar kernels or entries without layer-level data.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crown_coverage: Option<f32>,
    /// IBP output width for the same input bounds, when method is CROWN/AlphaCrown/BetaCrown.
    /// Enables comparison of CROWN vs IBP tightening. `None` for IBP-only entries
    /// or when IBP comparison was not performed.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ibp_comparison_width: Option<f32>,
    /// Ratio of CROWN output width to IBP output width (`output_width / ibp_comparison_width`).
    /// Values < 1.0 indicate CROWN produced tighter bounds than IBP.
    /// `None` for IBP-only entries or when IBP comparison was not performed.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crown_ibp_ratio: Option<f32>,
    /// When CROWN/IBP ratio is vacuous due to synthetic weight structure,
    /// this field records the artifact type (e.g., "uniform_positive_synthetic").
    /// Entries with this field should not be counted as evidence of CROWN
    /// tightening capability. See #2615.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub weight_artifact: Option<String>,
    /// Human-readable justification for the `soundness_mode` classification.
    /// Required for `Heuristic` entries to document which approximation was used
    /// (e.g., "InstanceNorm forward-pass midpoint statistics", "GroupNorm linearization").
    /// `None` for `Sound` entries or legacy entries without justification (#2635).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub soundness_justification: Option<String>,
    /// `true` when this entry verifies against an outdated model architecture.
    /// Stale entries should not be counted toward verification coverage (#2508).
    #[serde(default, skip_serializing_if = "is_false")]
    pub stale: bool,
    /// Human-readable reason why this entry is stale.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub stale_reason: Option<String>,
    /// Combined proof strength classification derived from `soundness_mode`,
    /// `method`, and `output_width`. Computed automatically; legacy entries
    /// without this field get it set on load (#2650).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub proof_strength: Option<ProofStrength>,
}

impl KernelStatus {
    /// Construct a new `KernelStatus` with required fields.
    ///
    /// Optional fields (`crown_error`, `smt`, `crown_coverage`, etc.) default
    /// to `None` / `false`. `proof_strength` is computed automatically from
    /// `soundness_mode`, `method`, and `output_width`.
    #[must_use]
    pub fn new(
        status: VerifyOutcome,
        method: PropMethod,
        input_bounds: InputBoundsRecord,
        output_bounds: OutputBoundsRecord,
        output_width: f32,
        soundness_mode: VerificationSoundnessMode,
    ) -> Self {
        let proof_strength = Some(compute_proof_strength(soundness_mode, method, output_width));
        Self {
            status,
            method,
            input_bounds,
            output_bounds,
            output_width,
            crown_error: None,
            soundness_mode,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: None,
            crown_ibp_ratio: None,
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength,
        }
    }
}

// ProofStrength, compute_proof_strength, and VACUOUS_WIDTH_THRESHOLD extracted
// to status_proof_strength.rs to keep this file under 450 lines.
#[path = "status_proof_strength.rs"]
mod status_proof_strength;
pub use status_proof_strength::{compute_proof_strength, ProofStrength, VACUOUS_WIDTH_THRESHOLD};

/// High-level verification outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum VerifyOutcome {
    /// IBP or CROWN produced finite bounds directly.
    Verified,
    /// Bounds were computed but are not conclusively useful: either non-finite,
    /// or finite but vacuously wide (output width exceeds `VACUOUS_PIPELINE_WIDTH`).
    /// For fusion results: CROWN succeeded but diff exceeds epsilon.
    BoundsComputed,
    /// CROWN was attempted, failed, and IBP fallback produced finite bounds.
    /// The CROWN failure reason is in `KernelStatus::crown_error`.
    IbpFallback,
    /// Verification produced an error or degenerate bounds.
    Failed,
    /// NY reported `Verified` but ay SMT found a counterexample.
    /// The `smt` field on `KernelStatus` contains the counterexample detail.
    SmtContradiction,
}

// Record types (ParamInputRecord, InputBoundsRecord, OutputBoundsRecord)
// extracted to status_record_types.rs to keep this file under 450 lines (#2575).
#[path = "status_record_types.rs"]
mod status_record_types;
pub use status_record_types::{InputBoundsRecord, OutputBoundsRecord, ParamInputRecord};

impl VerifyStatus {
    /// Load from file, or return default if file doesn't exist.
    ///
    /// For concurrent access, prefer [`load_locked`](Self::load_locked) which
    /// holds an advisory lock across the full load-modify-save cycle.
    #[must_use = "returns a Result that may contain an error"]
    pub fn load(path: &Path) -> Result<Self, VerifyError> {
        match std::fs::read_to_string(path) {
            Ok(contents) => {
                let mut status: Self = serde_json::from_str(&contents)?;
                status.normalize_legacy_input_bounds();
                status.normalize_proof_strength();
                status.truncate_history();
                Ok(status)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(VerifyError::Io(e)),
        }
    }

    /// Load from file with an exclusive advisory lock held for the duration.
    ///
    /// Returns a [`LockedStatus`] guard that holds the lock until dropped.
    /// Use this for `load-modify-save` sequences where concurrent writers
    /// must be excluded for the entire operation, not just during `save()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock cannot be acquired or the file cannot
    /// be read.
    #[must_use = "returns a Result that may contain an error"]
    pub fn load_locked(path: &Path) -> Result<LockedStatus, VerifyError> {
        let lock = StatusFileLock::acquire(path)?;
        let status = Self::load(path)?;
        Ok(LockedStatus {
            status,
            _lock: lock,
            path: path.to_path_buf(),
        })
    }

    // --- Read-only accessors ---

    /// Read-only view of the latest verification results per kernel name.
    #[must_use]
    pub fn kernels(&self) -> &BTreeMap<String, KernelStatus> {
        &self.kernels
    }

    /// Look up the latest verification result for a kernel by name.
    #[must_use]
    pub fn kernel(&self, name: &str) -> Option<&KernelStatus> {
        self.kernels.get(name)
    }

    /// Number of distinct kernels with verification results.
    #[must_use]
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// Check whether a kernel has any verification result.
    #[must_use]
    pub fn has_kernel(&self, name: &str) -> bool {
        self.kernels.contains_key(name)
    }

    /// Count entries by soundness mode: `(sound_count, heuristic_count)`.
    ///
    /// Excludes stale entries from counts (#2635).
    #[must_use]
    pub fn soundness_counts(&self) -> (usize, usize) {
        let mut sound = 0usize;
        let mut heuristic = 0usize;
        for entry in self.kernels.values() {
            if entry.stale {
                continue;
            }
            match entry.soundness_mode {
                VerificationSoundnessMode::Sound => sound += 1,
                VerificationSoundnessMode::Heuristic => heuristic += 1,
            }
        }
        (sound, heuristic)
    }

    /// Count entries by proof strength: `(sound_crown, sound_ibp, heuristic, vacuous)`.
    ///
    /// Excludes stale entries from counts (#2650).
    #[must_use]
    pub fn proof_strength_counts(&self) -> (usize, usize, usize, usize) {
        let (mut sc, mut si, mut h, mut v) = (0, 0, 0, 0);
        for entry in self.kernels.values() {
            if entry.stale {
                continue;
            }
            let strength = entry.proof_strength.unwrap_or_else(|| {
                compute_proof_strength(entry.soundness_mode, entry.method, entry.output_width)
            });
            match strength {
                ProofStrength::SoundCrown | ProofStrength::SoundMixed => sc += 1,
                ProofStrength::SoundIbp => si += 1,
                ProofStrength::Heuristic => h += 1,
                ProofStrength::Vacuous => v += 1,
            }
        }
        (sc, si, h, v)
    }

    /// Set the soundness justification for an existing kernel entry.
    ///
    /// Documents why the entry has its current `soundness_mode` classification
    /// (e.g., "InstanceNorm forward-pass midpoint statistics" for Heuristic entries).
    /// Returns `Err` if the kernel entry does not exist (#2635).
    pub fn set_soundness_justification(
        &mut self,
        name: &str,
        justification: &str,
    ) -> Result<(), VerifyError> {
        let entry = self
            .kernels
            .get_mut(name)
            .ok_or_else(|| VerifyError::InvalidInput(format!("kernel '{name}' not found")))?;
        entry.soundness_justification = Some(justification.to_string());
        Ok(())
    }

    /// Mark an existing kernel entry as stale and attach a reason.
    ///
    /// Stale entries remain in the status file for audit/history purposes but
    /// are excluded from soundness/proof-strength coverage counts.
    pub fn mark_stale(&mut self, name: &str, reason: &str) -> Result<(), VerifyError> {
        let entry = self
            .kernels
            .get_mut(name)
            .ok_or_else(|| VerifyError::InvalidInput(format!("kernel '{name}' not found")))?;
        entry.stale = true;
        entry.stale_reason = Some(reason.to_string());

        if let Some(last) = self.history.get_mut(name).and_then(|h| h.last_mut()) {
            last.stale = true;
            last.stale_reason = Some(reason.to_string());
        }

        Ok(())
    }

    /// Read-only view of the full verification history.
    #[must_use]
    pub fn history(&self) -> &BTreeMap<String, Vec<KernelStatus>> {
        &self.history
    }

    /// Return the history of verification runs for a specific kernel.
    #[must_use]
    pub fn history_for(&self, name: &str) -> Option<&[KernelStatus]> {
        self.history.get(name).map(Vec::as_slice)
    }

    /// Save to file using atomic write semantics (temp file + fsync + rename)
    /// with advisory file locking to prevent concurrent write races.
    ///
    /// Acquires an advisory lock (`.lock` file), writes to a temporary file in
    /// the same directory, calls `fsync`, then renames to the target path. The
    /// lock is released on completion (or on error via `Drop`).
    ///
    /// The lock prevents concurrent `load-modify-save` sequences from silently
    /// dropping results when multiple processes write to the same status file.
    /// Stale locks from crashed processes are auto-cleaned after 5 minutes.
    #[must_use = "returns a Result that may contain an error"]
    pub fn save(&self, path: &Path) -> Result<(), VerifyError> {
        use std::io::Write;

        let _lock = StatusFileLock::acquire(path)?;

        let json = serde_json::to_string_pretty(self)?;
        let dir = path.parent().unwrap_or(Path::new("."));
        let tmp_path = atomic_tmp_path(path);
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp_path)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        drop(file);

        if let Err(e) = std::fs::rename(&tmp_path, path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(VerifyError::Io(e));
        }
        sync_directory(dir)?;
        Ok(())
    }

    // Recording methods (record, record_failure, record_smt, run_count) live in
    // status_recording.rs to keep this file under 500 lines.

    fn normalize_legacy_input_bounds(&mut self) {
        for kernel_status in self.kernels.values_mut() {
            kernel_status.input_bounds.normalize_legacy_fields();
        }
        for entries in self.history.values_mut() {
            for entry in entries.iter_mut() {
                entry.input_bounds.normalize_legacy_fields();
            }
        }
    }

    /// Compute and set `proof_strength` for entries missing it (legacy data, #2650).
    fn normalize_proof_strength(&mut self) {
        for entry in self.kernels.values_mut() {
            if entry.proof_strength.is_none() {
                entry.proof_strength = Some(compute_proof_strength(
                    entry.soundness_mode,
                    entry.method,
                    entry.output_width,
                ));
            }
        }
    }

    /// Truncate history to `MAX_HISTORY_PER_KERNEL` most recent entries per kernel.
    /// Called on `load()` to compact legacy data (#538).
    fn truncate_history(&mut self) {
        use status_recording::MAX_HISTORY_PER_KERNEL;
        for entries in self.history.values_mut() {
            if entries.len() > MAX_HISTORY_PER_KERNEL {
                let excess = entries.len() - MAX_HISTORY_PER_KERNEL;
                entries.drain(..excess);
            }
        }
    }
}

// LockedStatus, InputBoundsRecord impls, and OutputBoundsRecord impls
// extracted to status_types.rs.
#[path = "status_types.rs"]
mod status_types;
pub use status_types::LockedStatus;

#[cfg(kani)]
#[path = "kani_status.rs"]
mod kani_status;

#[cfg(all(test, feature = "ny"))]
#[path = "status_recording_tests.rs"]
mod status_recording_tests;
#[cfg(test)]
#[path = "status_test_helpers.rs"]
mod status_test_helpers;
#[cfg(all(test, feature = "ny"))]
#[path = "status_tests.rs"]
mod status_tests;
#[cfg(all(test, feature = "ny"))]
#[path = "status_tests_history.rs"]
mod status_tests_history;
#[cfg(all(test, feature = "ny"))]
#[path = "status_tests_infeasible.rs"]
mod status_tests_infeasible;
#[cfg(all(test, feature = "ny"))]
#[path = "status_tests_legacy.rs"]
mod status_tests_legacy;
#[cfg(all(test, feature = "ny"))]
#[path = "status_tests_lock.rs"]
mod status_tests_lock;
#[cfg(all(test, feature = "ny"))]
#[path = "status_tests_smt.rs"]
mod status_tests_smt;
#[cfg(all(test, feature = "ny"))]
#[path = "status_tests_validation.rs"]
mod status_tests_validation;
