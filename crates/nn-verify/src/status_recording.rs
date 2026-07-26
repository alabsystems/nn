// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Write-path functions for `VerifyStatus`: `record`, `record_failure`,
//! `record_smt`, `record_fusion`.
//!
//! Extracted from `status.rs` to keep it under 500 lines (#359).

use crate::soundness_compat::VerificationSoundnessMode;

use super::*;
#[cfg(feature = "ny")]
use crate::fusion_spec::FusionVerification;
use crate::util::finite_or;

/// Maximum number of history records retained per kernel.
/// Older records are discarded on each `record*` call to prevent unbounded
/// growth of `nn_verify_status.json` (#538).
pub(super) const MAX_HISTORY_PER_KERNEL: usize = 10;

/// Pipeline output width above which bounds are vacuously wide and the
/// status is downgraded from `Verified` to `BoundsComputed`.
///
/// A width of 1e6 means the bounds constrain the output to a range of
/// ±500,000 — effectively unconstrained for any neural network output.
/// Matches the threshold used in `assert_bounds_width` in compose tests.
const VACUOUS_PIPELINE_WIDTH: f32 = 1e6;

#[cfg(feature = "ny")]
impl VerifyStatus {
    /// Record a kernel verification result.
    ///
    /// `status_key` overrides the entry key in the status file. When `None`,
    /// `result.kernel_name` is used. This supports distinct config names in
    /// `verify_all` without mutating `kernel.name` (#521).
    ///
    /// Returns an error if any constant parameter is non-finite
    /// (NaN or ±Infinity). Input bounds are validated at construction.
    #[must_use = "returns a Result that may contain an error"]
    pub fn record(
        &mut self,
        result: &KernelVerification,
        bounds: ScalarInputBounds,
        constant_params: &[f32],
        status_key: Option<&str>,
    ) -> Result<(), VerifyError> {
        let variable_inputs = [ParamInputRecord {
            param_index: 0,
            lower: bounds.lower(),
            upper: bounds.upper(),
        }];
        self.record_with_variable_inputs(
            result,
            &variable_inputs,
            constant_params,
            status_key,
            None, // scalar kernel — no tensor shape
        )
    }

    /// Record a verification failure for a kernel.
    ///
    /// Use this when verification returned an error or produced degenerate
    /// bounds (e.g. both NaN). The method and input bounds are recorded so
    /// the failure context is preserved.
    ///
    /// Returns an error if any constant parameter is non-finite. Input bounds
    /// are validated at construction.
    #[must_use = "returns a Result that may contain an error"]
    pub fn record_failure(
        &mut self,
        kernel_name: &str,
        method: PropMethod,
        bounds: ScalarInputBounds,
        constant_params: &[f32],
    ) -> Result<(), VerifyError> {
        let variable_inputs = [ParamInputRecord {
            param_index: 0,
            lower: bounds.lower(),
            upper: bounds.upper(),
        }];
        self.record_failure_with_variable_inputs(
            kernel_name,
            method,
            &variable_inputs,
            constant_params,
            None, // scalar kernel — no tensor shape
        )
    }
}

impl VerifyStatus {
    /// Insert a `KernelStatus` entry and update history, truncating old records.
    fn record_and_track(&mut self, key: &str, entry: KernelStatus) {
        let hist = self.history.entry(key.to_string()).or_default();
        hist.push(entry.clone());
        if hist.len() > MAX_HISTORY_PER_KERNEL {
            let excess = hist.len() - MAX_HISTORY_PER_KERNEL;
            hist.drain(..excess);
        }
        self.kernels.insert(key.to_string(), entry);
    }

    /// Record a kernel verification result with explicit per-variable metadata.
    ///
    /// `status_key` overrides the entry key in the status file (#521).
    /// When `None`, `result.kernel_name` is used. `input_shape` records the
    /// actual tensor shape used for verification (pass `None` for scalar kernels).
    ///
    /// Returns an error if any variable input bound or constant parameter is
    /// non-finite (NaN or ±Infinity).
    #[must_use = "returns a Result that may contain an error"]
    pub fn record_with_variable_inputs(
        &mut self,
        result: &KernelVerification,
        variable_inputs: &[ParamInputRecord],
        constant_params: &[f32],
        status_key: Option<&str>,
        input_shape: Option<&[usize]>,
    ) -> Result<(), VerifyError> {
        let key = status_key.unwrap_or(&result.kernel_name);
        validate_input_metadata(variable_inputs, constant_params)?;

        let outcome = if result.crown_fallback_reason.is_some() {
            if result.is_finite {
                VerifyOutcome::IbpFallback
            } else {
                VerifyOutcome::BoundsComputed
            }
        } else if result.is_finite {
            VerifyOutcome::Verified
        } else {
            VerifyOutcome::BoundsComputed
        };

        let mut entry = KernelStatus::new(
            outcome,
            result.method,
            InputBoundsRecord::from_variable_inputs(variable_inputs, constant_params, input_shape),
            OutputBoundsRecord::from_verification(result),
            finite_or(result.output_width, f32::MAX),
            result.soundness_mode,
        );
        entry.crown_error = result.crown_fallback_reason.clone();
        self.record_and_track(key, entry);
        Ok(())
    }

    /// Record a verification failure with explicit per-variable metadata.
    ///
    /// Returns an error if any variable input bound or constant parameter is
    /// non-finite (NaN or ±Infinity).
    #[must_use = "returns a Result that may contain an error"]
    pub fn record_failure_with_variable_inputs(
        &mut self,
        kernel_name: &str,
        method: PropMethod,
        variable_inputs: &[ParamInputRecord],
        constant_params: &[f32],
        input_shape: Option<&[usize]>,
    ) -> Result<(), VerifyError> {
        validate_input_metadata(variable_inputs, constant_params)?;

        let entry = KernelStatus::new(
            VerifyOutcome::Failed,
            method,
            InputBoundsRecord::from_variable_inputs(variable_inputs, constant_params, input_shape),
            OutputBoundsRecord::zero(),
            0.0,
            VerificationSoundnessMode::Heuristic,
        );
        self.record_and_track(kernel_name, entry);
        Ok(())
    }

    /// Attach an SMT verification result to an existing kernel entry.
    ///
    /// Updates the latest status entry and the most recent history entry
    /// for the given kernel. Returns `Err` if no entry exists for
    /// the kernel (caller should `record()` first).
    ///
    /// Logs a warning to stderr if the kernel entry was updated but the
    /// corresponding history entry is missing or empty (indicates a
    /// structural inconsistency in the status tracker).
    #[must_use = "returns a Result that may contain an error"]
    pub fn record_smt(
        &mut self,
        kernel_name: &str,
        smt: SmtStatusRecord,
    ) -> Result<(), VerifyError> {
        let Some(entry) = self.kernels.get_mut(kernel_name) else {
            return Err(VerifyError::InvalidInput(format!(
                "record_smt: no kernel entry for '{kernel_name}' — \
                 record() must be called before record_smt()"
            )));
        };
        entry.smt = Some(smt.clone());

        // If ay found a counterexample, the NY Verified/IbpFallback
        // status is contradicted. Downgrade to SmtContradiction so consumers
        // checking `status` alone cannot mistake the kernel as verified (#393).
        if smt.outcome == SmtOutcome::Counterexample {
            entry.status = VerifyOutcome::SmtContradiction;
        }

        let history_updated = self
            .history
            .get_mut(kernel_name)
            .and_then(|h| h.last_mut())
            .map(|last| {
                if smt.outcome == SmtOutcome::Counterexample {
                    last.status = VerifyOutcome::SmtContradiction;
                }
                last.smt = Some(smt);
                true
            })
            .unwrap_or(false);

        if !history_updated {
            let _ = std::io::Write::write_fmt(
                &mut std::io::stderr(),
                format_args!(
                    "nn-verify: record_smt: kernel '{kernel_name}' has a status entry \
                     but no history entry — SMT result recorded to latest status only\n"
                ),
            );
        }
        Ok(())
    }

    /// Return the number of historical runs for a kernel.
    #[must_use]
    pub fn run_count(&self, kernel_name: &str) -> usize {
        self.history.get(kernel_name).map_or(0, Vec::len)
    }

    /// Record a pipeline-level verification result (e.g., per-layer CROWN composition).
    ///
    /// Creates a `KernelStatus` entry from pipeline-level data without requiring
    /// NY types. Input/output bounds are scalar summaries (global
    /// min of lower, global max of upper) of the end-to-end pipeline bounds.
    ///
    /// `output_shape` records the shape of the pipeline output tensor.
    /// `input_shape` records the actual tensor shape used for verification input.
    ///
    /// # Errors
    ///
    /// Returns an error if any bound value is non-finite.
    #[must_use = "returns a Result that may contain an error"]
    pub fn record_pipeline(
        &mut self,
        status_key: &str,
        method: PropMethod,
        input_lower: f32,
        input_upper: f32,
        output_lower: f32,
        output_upper: f32,
        output_shape: &[usize],
        soundness_mode: VerificationSoundnessMode,
        input_shape: Option<&[usize]>,
    ) -> Result<(), VerifyError> {
        if !input_lower.is_finite()
            || !input_upper.is_finite()
            || !output_lower.is_finite()
            || !output_upper.is_finite()
        {
            return Err(VerifyError::InvalidInput(format!(
                "record_pipeline: non-finite bounds for '{status_key}'"
            )));
        }

        let variable_inputs = vec![ParamInputRecord {
            param_index: 0,
            lower: input_lower,
            upper: input_upper,
        }];
        let output_width = output_upper - output_lower;
        // Vacuously wide bounds provide no useful verification constraint.
        // Downgrade to BoundsComputed so status consumers don't mistake
        // a 2e10-wide range as meaningful verification (#2218).
        let outcome = if output_width > VACUOUS_PIPELINE_WIDTH {
            VerifyOutcome::BoundsComputed
        } else {
            VerifyOutcome::Verified
        };
        let entry = KernelStatus::new(
            outcome,
            method,
            InputBoundsRecord::from_variable_inputs(&variable_inputs, &[], input_shape),
            OutputBoundsRecord::with_shape(output_lower, output_upper, output_shape.to_vec()),
            output_width,
            soundness_mode,
        );
        self.record_and_track(status_key, entry);
        Ok(())
    }
}

#[cfg(feature = "ny")]
impl VerifyStatus {
    /// Record a fusion equivalence verification result.
    ///
    /// Maps `FusionVerification` fields to a `KernelStatus` entry using
    /// the `fusion_` prefix convention. The status key defaults to
    /// `"fusion_{fused_kernel_name}"` unless overridden via `status_key`.
    ///
    /// Status mapping:
    /// - `Verified` if `within_epsilon && is_conclusive()` (CROWN proved equivalence)
    /// - `IbpFallback` if `within_epsilon && !is_conclusive()` (IBP — may be vacuously wide)
    /// - `Failed` if `!within_epsilon`
    ///
    /// The output bounds record stores the diff bounds (`diff_lower`, `diff_upper`),
    /// and `output_width` stores `max_abs_diff`. The `input_bounds` stores the
    /// variable bounds used for the fusion check.
    ///
    /// # Errors
    ///
    /// Returns an error if any variable bound is non-finite.
    #[must_use = "returns a Result that may contain an error"]
    pub fn record_fusion(
        &mut self,
        result: &FusionVerification,
        variable_bounds: &[(f32, f32)],
        status_key: Option<&str>,
    ) -> Result<(), VerifyError> {
        let default_key = format!("fusion_{}", result.fused_kernel_name);
        let key = status_key.unwrap_or(&default_key);

        let variable_inputs: Vec<ParamInputRecord> = variable_bounds
            .iter()
            .enumerate()
            .map(|(i, &(lo, hi))| ParamInputRecord {
                param_index: i,
                lower: lo,
                upper: hi,
            })
            .collect();
        validate_input_metadata(&variable_inputs, &[])?;

        // Fusion outcomes distinguish CROWN-conclusive results from IBP fallback.
        // BoundsComputed: CROWN succeeded with finite bounds but diff exceeds epsilon.
        // Failed: IBP fallback or degenerate — vacuously wide, not conclusive (#2225).
        let outcome = if result.within_epsilon && result.is_conclusive() {
            VerifyOutcome::Verified
        } else if result.within_epsilon {
            VerifyOutcome::IbpFallback
        } else if result.is_conclusive() {
            VerifyOutcome::BoundsComputed
        } else {
            VerifyOutcome::Failed
        };

        // Infeasible bounds: lower > upper (mark_infeasible pattern is +Inf/-Inf),
        // or both non-finite (NaN/NaN from failed verification).
        // Matches the pattern in OutputBoundsRecord::from_verification (#1692 F3).
        let is_infeasible = (!result.diff_lower.is_finite() && !result.diff_upper.is_finite())
            || (result.diff_lower.is_finite()
                && result.diff_upper.is_finite()
                && result.diff_lower > result.diff_upper);
        let diff_lo = finite_or(result.diff_lower, 0.0);
        let diff_hi = finite_or(result.diff_upper, 0.0);
        let mut output_bounds = OutputBoundsRecord::with_shape(diff_lo, diff_hi, vec![1]);
        output_bounds.tensor_lower = Some(vec![diff_lo]);
        output_bounds.tensor_upper = Some(vec![diff_hi]);
        output_bounds.is_infeasible = is_infeasible;

        let mut entry = KernelStatus::new(
            outcome,
            result.method,
            InputBoundsRecord::from_variable_inputs(&variable_inputs, &[], None),
            output_bounds,
            finite_or(result.max_abs_diff, f32::MAX),
            result.soundness_mode,
        );
        entry.crown_error = result.crown_fallback_reason.clone();
        self.record_and_track(key, entry);
        Ok(())
    }
}

// CROWN comparison recording and reporting extracted to status_crown_comparison.rs
// to keep this file under 450 lines.
