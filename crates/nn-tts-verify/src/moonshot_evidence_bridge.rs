// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bridge functions connecting verification infrastructure to moonshot
//! evidence types (`SmtVerificationEvidence`, `KaniVerificationEvidence`).
//!
//! - P8 bridge: scans `nn_verify_status.json` via `VerifyStatus` for real
//!   SMT outcomes (requires `NY` feature for `nn-verify` dep).
//! - P7 bridge (preferred): reads `kani_status.json` for actual pass/fail
//!   results per harness via `from_kani_status_file()`.
//! - P7 bridge (fallback): scans workspace `.rs` files for `#[kani::proof]`
//!   annotations via `from_workspace_scan()`.
//! - P7 bridge (combined): `from_kani_status_or_scan()` tries the status
//!   file first, falls back to workspace scan.
//!
//! These replace hardcoded kernel names and counts with real verification
//! results in the moonshot certificate pipeline.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::KaniVerificationEvidence;
#[cfg(feature = "ny")]
use super::SmtVerificationEvidence;

// -- P8: SmtVerificationEvidence from VerifyStatus ----------------------------

#[cfg(feature = "ny")]
impl SmtVerificationEvidence {
    /// Construct P8 evidence from the persisted `VerifyStatus`.
    ///
    /// Scans all kernel entries (latest and history) for SMT results with
    /// `SmtOutcome::Proven`. A kernel counts as "proven" if its latest entry
    /// has `smt.outcome == Proven`, or (if the latest entry has `smt: None`)
    /// its most recent history entry has `smt.outcome == Proven`.
    ///
    /// `kernels_total` is the number of distinct kernels that have any SMT
    /// result (Proven, Counterexample, Unknown, or ExecutionFailed).
    /// `Unexecuted` entries are excluded from the total since the solver
    /// was never invoked.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let status = VerifyStatus::load(Path::new("nn_verify_status.json"))?;
    /// let evidence = SmtVerificationEvidence::from_verify_status(&status);
    /// let cert = cert.with_smt_results(&evidence);
    /// ```
    #[must_use]
    pub fn from_verify_status(status: &nn_verify::VerifyStatus) -> Self {
        use nn_verify::SmtOutcome;

        let mut proven_names: Vec<String> = Vec::new();
        let mut total_with_smt = 0usize;

        // Collect all distinct kernel names from both latest and history.
        let mut all_kernel_names: Vec<&String> = status.kernels().keys().collect();
        for key in status.history().keys() {
            if !status.kernels().contains_key(key) {
                all_kernel_names.push(key);
            }
        }

        for name in &all_kernel_names {
            // Try latest kernel entry first.
            let latest_smt = status.kernel(name).and_then(|ks| ks.smt.as_ref());

            // If latest has no SMT, fall back to most recent history entry with SMT.
            let effective_smt = if latest_smt.is_some() {
                latest_smt
            } else {
                status
                    .history_for(name)
                    .and_then(|entries| entries.iter().rev().find_map(|ks| ks.smt.as_ref()))
            };

            if let Some(smt_record) = effective_smt {
                // Skip Unexecuted — solver was never invoked, not real evidence.
                if smt_record.outcome == SmtOutcome::Unexecuted {
                    continue;
                }

                total_with_smt += 1;

                if smt_record.outcome == SmtOutcome::Proven {
                    proven_names.push((*name).clone());
                }
            }
        }

        proven_names.sort();

        let kernels_proven = proven_names.len();
        Self {
            kernels_proven,
            kernels_total: total_with_smt,
            proven_kernel_names: proven_names,
            all_proven: kernels_proven == total_with_smt && total_with_smt > 0,
        }
    }
}

// -- P7: KaniVerificationEvidence from kani_status.json -----------------------

/// A single harness entry deserialized from `kani_status.json`.
///
/// Only the `status` field is needed for pass/fail determination.
/// All other fields (duration_sec, commit, error_message, etc.) are
/// ignored for certificate purposes.
#[derive(serde::Deserialize)]
struct KaniStatusHarnessEntry {
    status: String,
}

/// Top-level `kani_status.json` structure.
///
/// The Python `kani_runner` tool generates this file with per-harness
/// results. Only the `harnesses` map is needed for certificate evidence.
#[derive(serde::Deserialize)]
struct KaniStatusFileSchema {
    harnesses: BTreeMap<String, KaniStatusHarnessEntry>,
}

impl KaniVerificationEvidence {
    /// Construct P7 evidence from a `kani_status.json` file.
    ///
    /// This reads actual Kani pass/fail results rather than assuming all
    /// harnesses pass. Each harness entry has a `status` field:
    /// - `"passed"` → counts toward `harnesses_passed`
    /// - `"failed"`, `"timeout"`, `"not_run"`, or unknown → not passed
    ///
    /// Also cross-references the workspace source scan (via `crates_dir`)
    /// to include harness file paths in the evidence.
    ///
    /// Returns `None` if `kani_status.json` cannot be read or parsed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let evidence = KaniVerificationEvidence::from_kani_status_file(
    ///     Path::new("kani_status.json"),
    ///     Path::new("crates/"),
    /// );
    /// if let Some(evidence) = evidence {
    ///     let cert = cert.with_kani_results(&evidence);
    /// }
    /// ```
    #[must_use]
    pub fn from_kani_status_file(kani_status_path: &Path, crates_dir: &Path) -> Option<Self> {
        let contents = std::fs::read_to_string(kani_status_path).ok()?;
        let status_file: KaniStatusFileSchema = serde_json::from_str(&contents).ok()?;

        let total = status_file.harnesses.len();
        let passed = status_file
            .harnesses
            .values()
            .filter(|entry| entry.status == "passed")
            .count();

        // Collect harness file paths from workspace scan for artifact tracking.
        let harness_files = if crates_dir.is_dir() {
            collect_harness_files(crates_dir)
        } else {
            Vec::new()
        };

        Some(Self {
            harnesses_passed: passed,
            harnesses_total: total,
            harness_files,
            all_passed: passed == total && total > 0,
        })
    }

    /// Construct P7 evidence by combining `kani_status.json` pass/fail data
    /// with workspace source scan.
    ///
    /// Falls back to `from_workspace_scan(assume_all_pass)` if the status
    /// file cannot be read. This is the recommended entry point — it uses
    /// real Kani results when available and degrades gracefully.
    #[must_use]
    pub fn from_kani_status_or_scan(
        kani_status_path: &Path,
        crates_dir: &Path,
        assume_all_pass_fallback: bool,
    ) -> Self {
        if let Some(evidence) = Self::from_kani_status_file(kani_status_path, crates_dir) {
            evidence
        } else {
            Self::from_workspace_scan(crates_dir, assume_all_pass_fallback)
        }
    }
}

/// Collect harness file paths by scanning .rs files for `#[kani::proof]`.
fn collect_harness_files(crates_dir: &Path) -> Vec<String> {
    let mut harness_files = Vec::new();
    if let Ok(entries) = walk_rs_files(crates_dir) {
        for path in entries {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                let count = contents
                    .lines()
                    .filter(|line| line.contains("#[kani::proof]"))
                    .count();
                if count > 0 {
                    if let Ok(relative) =
                        path.strip_prefix(crates_dir.parent().unwrap_or(crates_dir))
                    {
                        harness_files.push(relative.to_string_lossy().to_string());
                    } else {
                        harness_files.push(path.to_string_lossy().to_string());
                    }
                }
            }
        }
    }
    harness_files.sort();
    harness_files
}

// -- P7: KaniVerificationEvidence from workspace scan -------------------------

impl KaniVerificationEvidence {
    /// Construct P7 evidence by scanning workspace source files for
    /// `#[kani::proof]` harness annotations.
    ///
    /// `crates_dir` should be the path to the workspace `crates/` directory.
    /// Recursively searches `.rs` files for lines containing `#[kani::proof]`.
    ///
    /// Since Kani is run externally (not from `nn_verify_status.json`), this
    /// function does a simple source scan. When `assume_all_pass` is true,
    /// `harnesses_passed` equals `harnesses_total` — the caller vouches that
    /// all harnesses pass based on external Kani execution (e.g.,
    /// `kani_status.json` or `cargo kani` run).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let evidence = KaniVerificationEvidence::from_workspace_scan(
    ///     Path::new("crates/"),
    ///     true, // assume all pass (based on kani_status.json)
    /// );
    /// let cert = cert.with_kani_results(&evidence);
    /// ```
    #[must_use]
    pub fn from_workspace_scan(crates_dir: &Path, assume_all_pass: bool) -> Self {
        let mut harness_count = 0usize;
        let mut harness_files: Vec<String> = Vec::new();

        if let Ok(entries) = walk_rs_files(crates_dir) {
            for path in entries {
                if let Ok(contents) = std::fs::read_to_string(&path) {
                    let count = contents
                        .lines()
                        .filter(|line| line.contains("#[kani::proof]"))
                        .count();
                    if count > 0 {
                        harness_count += count;
                        // Store path relative to workspace root (strip up to crates/).
                        if let Ok(relative) =
                            path.strip_prefix(crates_dir.parent().unwrap_or(crates_dir))
                        {
                            harness_files.push(relative.to_string_lossy().to_string());
                        } else {
                            harness_files.push(path.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }

        harness_files.sort();

        let passed = if assume_all_pass { harness_count } else { 0 };

        Self {
            harnesses_passed: passed,
            harnesses_total: harness_count,
            harness_files,
            all_passed: assume_all_pass && harness_count > 0,
        }
    }
}

/// Recursively walk a directory collecting `.rs` file paths.
fn walk_rs_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    walk_rs_files_inner(dir, &mut result)?;
    Ok(result)
}

fn walk_rs_files_inner(dir: &Path, result: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files_inner(&path, result)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            result.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "moonshot_evidence_bridge_tests.rs"]
mod tests;
