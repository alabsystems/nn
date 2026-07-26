// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bridge between `kani_status.json` and proof certificates.
//!
//! `kani_status.json` is maintained by the Python `kani_runner` tool and
//! tracks per-harness Kani verification results. This module reads that file
//! and produces a [`KaniProofRecord`] suitable for embedding in a
//! [`ProofCertificate`](crate::certificate::ProofCertificate).
//!
//! Harness names follow the convention `<crate>::<harness_name>`. A kernel
//! lookup matches harnesses whose name (after the `::` separator) starts with
//! the kernel name. For example, kernel `"snake"` matches
//! `"nn-dsl::snake_scalar_finite_for_bounded_inputs"`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::certificate_types::{KaniOutcome, KaniProofRecord};
use crate::error::VerifyError;

// ---------------------------------------------------------------------------
// Serde types matching kani_status.json schema
// ---------------------------------------------------------------------------

/// Top-level `kani_status.json` structure.
#[derive(Debug, Deserialize)]
pub(crate) struct KaniStatusFile {
    pub harnesses: BTreeMap<String, HarnessEntry>,
    // `summary`, `last_updated`, `validation_rules` are present but not needed
    // for certificate generation.
}

/// A single harness entry in `kani_status.json`.
#[derive(Debug, Deserialize)]
pub(crate) struct HarnessEntry {
    pub status: String,
    // All other fields are optional/unused for certificate purposes.
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load and parse `kani_status.json` from the given path.
///
/// Returns an error if the file cannot be read or is malformed JSON.
pub(crate) fn load_kani_status(path: &Path) -> Result<KaniStatusFile, VerifyError> {
    let contents = std::fs::read_to_string(path)?;
    let status: KaniStatusFile = serde_json::from_str(&contents)?;
    Ok(status)
}

// ---------------------------------------------------------------------------
// Kernel → KaniProofRecord extraction
// ---------------------------------------------------------------------------

/// Extract a [`KaniProofRecord`] for a specific kernel from a loaded status file.
///
/// Harness matching: a harness named `"<crate>::<name>"` matches `kernel_name`
/// if `<name>` starts with `kernel_name`. This follows the naming convention
/// where kernel-specific harnesses are prefixed with the kernel name
/// (e.g., `snake_scalar_finite_for_bounded_inputs` for kernel `"snake"`).
///
/// The aggregate `status` is the worst outcome across all matching harnesses:
/// `Failed` > `Timeout` > `NotRun` > `Passed`.
///
/// Returns `None` if no harnesses match the kernel name.
pub(crate) fn kani_record_for_kernel(
    status: &KaniStatusFile,
    kernel_name: &str,
) -> Option<KaniProofRecord> {
    let mut matching: Vec<(&String, &HarnessEntry)> = Vec::new();

    for (name, entry) in &status.harnesses {
        // Extract the harness name after "::" separator.
        let harness_name = match name.find("::") {
            Some(pos) => &name[pos + 2..],
            None => name.as_str(),
        };
        if let Some(rest) = harness_name.strip_prefix(kernel_name) {
            // Ensure we match at a word boundary: the character after the
            // kernel name prefix must be '_', end-of-string, or non-alphanumeric.
            if rest.is_empty() || rest.starts_with('_') {
                matching.push((name, entry));
            }
        }
    }

    if matching.is_empty() {
        return None;
    }

    let harness_count = matching.len();

    // Compute aggregate status (worst outcome wins).
    let mut has_failed = false;
    let mut has_timeout = false;
    let mut has_not_run = false;

    for (_, entry) in &matching {
        match entry.status.as_str() {
            "failed" => has_failed = true,
            "timeout" => has_timeout = true,
            "not_run" => has_not_run = true,
            "passed" => {}
            _ => has_not_run = true, // Unknown statuses treated as not_run
        }
    }

    let status = if has_failed {
        KaniOutcome::Failed
    } else if has_timeout {
        KaniOutcome::Timeout
    } else if has_not_run {
        KaniOutcome::NotRun
    } else {
        KaniOutcome::Passed
    };

    // Infer properties from harness names.
    let mut properties = Vec::new();
    for (name, _) in &matching {
        if name.contains("no_overflow") || name.contains("finite") {
            if !properties.contains(&"no_overflow".to_string()) {
                properties.push("no_overflow".to_string());
            }
        }
        if name.contains("no_nan") || name.contains("nan") {
            if !properties.contains(&"no_nan".to_string()) {
                properties.push("no_nan".to_string());
            }
        }
        if name.contains("bounds") || name.contains("sound") || name.contains("monotone") {
            if !properties.contains(&"bounds_preservation".to_string()) {
                properties.push("bounds_preservation".to_string());
            }
        }
        if name.contains("safe") || name.contains("safety") {
            if !properties.contains(&"safety".to_string()) {
                properties.push("safety".to_string());
            }
        }
    }
    properties.sort();

    Some(KaniProofRecord {
        harness_count,
        status,
        properties,
        cbmc_version: None, // Not recorded in kani_status.json per-harness
    })
}

/// Convenience: load `kani_status.json` and extract a record for one kernel.
///
/// Returns `Ok(None)` if the file exists but has no matching harnesses.
/// Returns `Err` if the file cannot be read.
pub fn kani_record_from_file(
    kani_status_path: &Path,
    kernel_name: &str,
) -> Result<Option<KaniProofRecord>, VerifyError> {
    let status = load_kani_status(kani_status_path)?;
    Ok(kani_record_for_kernel(&status, kernel_name))
}

#[cfg(test)]
#[path = "kani_bridge_tests.rs"]
mod tests;
