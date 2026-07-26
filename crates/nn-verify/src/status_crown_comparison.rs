// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN vs IBP comparison recording and reporting (#2578).
//!
//! Extracted from `status_recording.rs` to keep it under 450 lines.

use super::*;

impl VerifyStatus {
    /// Attach IBP comparison data to an existing CROWN kernel entry.
    ///
    /// When a kernel was verified with CROWN, this records the IBP output width
    /// for the same input bounds and computes the CROWN/IBP ratio. Ratio < 1.0
    /// means CROWN produced tighter bounds than IBP.
    ///
    /// Updates both the latest status entry and the most recent history entry.
    ///
    /// # Errors
    ///
    /// Returns an error if no entry exists for the kernel, if `ibp_width`
    /// is non-finite, or if the entry's method is not CROWN.
    #[must_use = "returns a Result that may contain an error"]
    pub fn record_crown_comparison(
        &mut self,
        kernel_name: &str,
        ibp_width: f32,
    ) -> Result<(), VerifyError> {
        if !ibp_width.is_finite() {
            return Err(VerifyError::InvalidInput(format!(
                "record_crown_comparison: non-finite ibp_width for '{kernel_name}'"
            )));
        }

        let Some(entry) = self.kernels.get_mut(kernel_name) else {
            return Err(VerifyError::InvalidInput(format!(
                "record_crown_comparison: no kernel entry for '{kernel_name}' — \
                 record() must be called before record_crown_comparison()"
            )));
        };

        // Defense-in-depth: only CROWN-family entries should have IBP comparison data.
        if !entry.method.is_tight() {
            return Err(VerifyError::InvalidInput(format!(
                "record_crown_comparison: '{kernel_name}' uses {:?}, not CROWN",
                entry.method
            )));
        }

        entry.ibp_comparison_width = Some(ibp_width);
        let ratio = if ibp_width > 1e-10 {
            entry.output_width / ibp_width
        } else {
            1.0
        };
        entry.crown_ibp_ratio = Some(ratio);

        // Update most recent history entry
        if let Some(last) = self.history.get_mut(kernel_name).and_then(|h| h.last_mut()) {
            last.ibp_comparison_width = Some(ibp_width);
            last.crown_ibp_ratio = Some(ratio);
        }
        Ok(())
    }

    /// Return a summary of CROWN vs IBP effectiveness across all kernels.
    ///
    /// Returns `(crown_count, tighter_count, entries)` where:
    /// - `crown_count` is the number of CROWN entries with comparison data
    /// - `tighter_count` is the number where CROWN produced tighter bounds (ratio < 1.0)
    /// - `entries` is a Vec of `(kernel_name, ratio)` sorted by ratio ascending
    #[must_use]
    pub fn crown_comparison_report(&self) -> (usize, usize, Vec<(String, f32)>) {
        let mut entries: Vec<(String, f32)> = self
            .kernels
            .iter()
            .filter_map(|(name, ks)| ks.crown_ibp_ratio.map(|ratio| (name.clone(), ratio)))
            .collect();
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let crown_count = entries.len();
        let tighter_count = entries.iter().filter(|(_, r)| *r < 1.0).count();
        (crown_count, tighter_count, entries)
    }
}
