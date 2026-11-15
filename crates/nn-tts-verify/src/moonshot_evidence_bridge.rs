// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bridge from the verification infrastructure to the moonshot P8 evidence
//! type (`SmtVerificationEvidence`): scans `nn_verify_status.json` via
//! `VerifyStatus` for real SMT outcomes (requires the `ny` feature for the
//! `nn-verify` dependency). This replaces hardcoded kernel names and counts
//! with real verification results in the moonshot certificate pipeline.

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

#[cfg(all(test, feature = "ny"))]
#[path = "moonshot_evidence_bridge_smt_tests.rs"]
mod smt_bridge_tests;
