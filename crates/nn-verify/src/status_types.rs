// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Record type impls and `LockedStatus` guard for verification status.
//!
//! Extracted from `status.rs` for 500-line compliance.

use std::path::Path;

use crate::error::VerifyError;
use crate::util::{finite_or, sanitize_tensor_bounds};
use crate::verify_types::KernelVerification;

use super::status_helpers::{atomic_tmp_path, sync_directory, StatusFileLock};
use super::{InputBoundsRecord, OutputBoundsRecord, ParamInputRecord, VerifyStatus};

/// Guard holding both a [`VerifyStatus`] and its advisory file lock.
///
/// Created by [`VerifyStatus::load_locked`]. Provides mutable access to the
/// status and a [`save`](Self::save) method that writes without re-acquiring
/// the lock (since the lock is already held). The lock is released when this
/// guard is dropped.
pub struct LockedStatus {
    /// The loaded verification status. Mutate freely; call [`save`](Self::save)
    /// to persist.
    pub status: VerifyStatus,
    pub(super) _lock: StatusFileLock,
    pub(super) path: std::path::PathBuf,
}

impl LockedStatus {
    /// Save the status to disk without re-acquiring the lock.
    ///
    /// Uses the same atomic write (temp + fsync + rename) as
    /// [`VerifyStatus::save`], but skips lock acquisition since this guard
    /// already holds the lock.
    ///
    /// # Errors
    ///
    /// Returns an error if the write or rename fails.
    #[must_use = "returns a Result that may contain an error"]
    pub fn save(&self) -> Result<(), VerifyError> {
        use std::io::Write;

        let json = serde_json::to_string_pretty(&self.status)?;
        let dir = self.path.parent().unwrap_or(Path::new("."));
        let tmp_path = atomic_tmp_path(&self.path);
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp_path)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        drop(file);

        if let Err(e) = std::fs::rename(&tmp_path, &self.path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(VerifyError::Io(e));
        }
        sync_directory(dir)?;
        Ok(())
    }
}

impl InputBoundsRecord {
    /// Construct from variable inputs and constant parameters.
    ///
    /// When `input_shape` is `Some`, records the actual tensor shape used for
    /// verification. When `None`, falls back to `[variable_inputs.len()]` for
    /// backward compatibility with scalar kernel recording paths.
    pub fn from_variable_inputs(
        variable_inputs: &[ParamInputRecord],
        constant_params: &[f32],
        input_shape: Option<&[usize]>,
    ) -> Self {
        let shape = if let Some(s) = input_shape {
            Some(s.to_vec())
        } else if variable_inputs.is_empty() {
            None
        } else {
            Some(vec![variable_inputs.len()])
        };
        Self {
            variable_inputs: variable_inputs.to_vec(),
            constant_params: constant_params.to_vec(),
            input_shape: shape,
            input_range: Self::legacy_input_range_for_single_param_zero(variable_inputs),
        }
    }

    pub(super) fn normalize_legacy_fields(&mut self) {
        if self.variable_inputs.is_empty() {
            if let Some((lower, upper)) = self.input_range {
                // Skip non-finite legacy values (NaN/Inf from corrupted JSON).
                if lower.is_finite() && upper.is_finite() {
                    self.variable_inputs.push(ParamInputRecord {
                        param_index: 0,
                        lower,
                        upper,
                    });
                }
            }
        }

        if self.input_shape.is_none() && !self.variable_inputs.is_empty() {
            self.input_shape = Some(vec![self.variable_inputs.len()]);
        }

        self.input_range = Self::legacy_input_range_for_single_param_zero(&self.variable_inputs);
    }

    fn legacy_input_range_for_single_param_zero(
        variable_inputs: &[ParamInputRecord],
    ) -> Option<(f32, f32)> {
        if variable_inputs.len() == 1 && variable_inputs[0].param_index == 0 {
            let only = &variable_inputs[0];
            Some((only.lower, only.upper))
        } else {
            None
        }
    }
}

impl OutputBoundsRecord {
    /// Build from a `KernelVerification`, extracting tensor data when present.
    /// Non-finite bounds use `0.0` sentinels (matching the failure path) to
    /// prevent `serde_json` errors. Consumers check `is_finite`/`status`.
    ///
    /// Detects infeasible bounds (e.g., `lower=+Inf, upper=-Inf` from
    /// `mark_infeasible_all()`) and sets `is_infeasible = true` so consumers
    /// don't misinterpret `(0.0, 0.0)` as a verified tight bound (#1692 F3).
    pub fn from_verification(result: &KernelVerification) -> Self {
        let (tensor_lower, tensor_upper, shape) = match &result.output_tensor {
            Some(tensor) => (
                Some(sanitize_tensor_bounds(&tensor.lower)),
                Some(sanitize_tensor_bounds(&tensor.upper)),
                Some(tensor.shape.clone()),
            ),
            None => (None, None, None),
        };
        // Infeasible bounds: lower > upper (the mark_infeasible pattern is
        // +Inf/-Inf), or both non-finite (NaN/NaN from failed verification).
        // IEEE 754: NaN > NaN is false, so check non-finite explicitly.
        let is_infeasible = (!result.output_lower.is_finite() && !result.output_upper.is_finite())
            || (result.output_lower.is_finite()
                && result.output_upper.is_finite()
                && result.output_lower > result.output_upper);
        Self {
            lower: finite_or(result.output_lower, 0.0),
            upper: finite_or(result.output_upper, 0.0),
            tensor_lower,
            tensor_upper,
            shape,
            is_infeasible,
        }
    }
}
