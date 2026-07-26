// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Runtime dtype tracking for compiled model execution.
//!
//! Encapsulates the `buffer_dtypes` vector and eliminates the empty-vec
//! sentinel pattern from `run_steps_inner`. Always initialized from
//! `step_scalar_types` so `get()` returns correct dtypes even when
//! mutations are disabled.
//!
//! Part of #2981.

use nn_dsl::ir::ScalarType;

use super::StepMeta;

/// Tracks actual runtime dtypes for each buffer in the execution loop.
///
/// The 300-byte clone (300 steps × 1 byte) is negligible vs megabytes
/// of GPU allocation per forward pass.
pub(super) struct DtypeTracker {
    dtypes: Vec<ScalarType>,
    active: bool,
}

impl DtypeTracker {
    pub(super) fn new(step_metas: &[StepMeta], active: bool) -> Self {
        Self {
            // Always extract so get() returns correct base dtypes
            // even when set()/propagate() are disabled.
            dtypes: step_metas.iter().map(|m| m.scalar_type).collect(),
            active,
        }
    }

    /// Get the runtime dtype for a step. Returns step_scalar_types[step_idx]
    /// when tracking is off, or the dynamically updated dtype when on.
    pub(super) fn get(&self, step_idx: usize) -> ScalarType {
        self.dtypes
            .get(step_idx)
            .copied()
            .unwrap_or(ScalarType::F32)
    }

    /// Set the runtime dtype for a step. No-op if tracking is off.
    pub(super) fn set(&mut self, step_idx: usize, dtype: ScalarType) {
        if self.active {
            self.dtypes[step_idx] = dtype;
        }
    }

    /// Propagate dtype from source step (first edge) to destination step.
    /// Replaces the 3× duplicated IdentityPassthrough/Passthrough/NarrowView block.
    pub(super) fn propagate_from_source(&mut self, step_idx: usize, step_metas: &[StepMeta]) {
        if self.active {
            if let Some(&src_step) = step_metas.get(step_idx).and_then(|m| m.edges.first()) {
                self.dtypes[step_idx] = self.dtypes[src_step];
            }
        }
    }

    /// NarrowView byte offset scaling: F16/BF16 halves the F32-assumed offset.
    pub(super) fn narrow_byte_offset(
        &self,
        step_idx: usize,
        step_metas: &[StepMeta],
        f32_offset: usize,
    ) -> usize {
        if !self.active {
            return f32_offset;
        }
        if let Some(&src_step) = step_metas.get(step_idx).and_then(|m| m.edges.first()) {
            // Scale offset from F32-assumed bytes to actual element bytes.
            f32_offset * self.dtypes[src_step].byte_size() / ScalarType::F32.byte_size()
        } else {
            f32_offset
        }
    }

    /// Element byte size of the source step feeding into `step_idx`.
    ///
    /// Returns the source step's dtype byte size (e.g. 2 for F16, 4 for F32).
    /// Falls back to F32 (4 bytes) when tracking is inactive or the edge
    /// map has no entry. Used by NarrowView upper-bound validation (#3266).
    pub(super) fn source_byte_size(&self, step_idx: usize, step_metas: &[StepMeta]) -> usize {
        if let Some(&src_step) = step_metas.get(step_idx).and_then(|m| m.edges.first()) {
            self.dtypes[src_step].byte_size()
        } else {
            ScalarType::F32.byte_size()
        }
    }

    /// Immutable slice access for helpers that take `&[ScalarType]`.
    pub(super) fn as_slice(&self) -> &[ScalarType] {
        &self.dtypes
    }

    /// Mutable slice access for helpers that take `&mut [ScalarType]`.
    /// Only called when active (mixed_precision path).
    ///
    /// Uses a runtime assert (not debug_assert) because debug_assert is
    /// stripped in release builds, which would silently allow mutation of
    /// an inactive tracker. See design doc: "Never debug_assert for
    /// production validation."
    pub(super) fn as_mut_slice(&mut self) -> &mut [ScalarType] {
        assert!(
            self.active,
            "DtypeTracker::as_mut_slice on inactive tracker"
        );
        &mut self.dtypes
    }

    /// Consume into inner Vec for return value.
    pub(super) fn into_inner(self) -> Vec<ScalarType> {
        self.dtypes
    }
}
