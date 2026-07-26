// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pipeline-level ICB replay wiring for [`CompiledKokoro`].
//!
//! Bridges the shape-keyed [`IcbReplayBuffer`] with the Kokoro forward
//! execution loop. The pipeline is split at the `regulate_scalar_readback`
//! sync point into two independently replayable phases:
//!
//! - **Pre-readback** (steps 1-4): PlBert + TextEncoder + Prosody + Regulate
//!   Shape key: `seq_len`
//! - **Post-readback** (steps 5-8): F0Energy + HarmonicSource + Generator + iSTFT
//!   Shape key: `(t_mel, total_samples)`
//!
//! On the first forward pass for a given shape, dispatches execute normally
//! and the `IcbReplayBuffer` remains empty (recording requires a separate
//! warmup pass to establish deterministic arena offsets). On the second pass,
//! the replay buffer is populated. On subsequent passes, the pre-encoded ICB
//! is replayed via `executeCommandsInBuffer`.
//!
//! # Feature gate
//!
//! The `icb-replay` feature must be enabled AND `with_icb_replay()` must be
//! called on `CompiledKokoro`. When disabled, all methods in this module
//! are no-ops (the `IcbReplayBuffer` is created with `use_icb_replay: false`).
//!
//! # Arena determinism requirement
//!
//! ICB replay encodes fixed buffer offsets. Arena offsets must be identical
//! across forward passes with the same input shape. This is guaranteed after
//! warmup because:
//! - Arena capacity is pre-sized via `ensure_capacity()`.
//! - Bump allocator resets to offset 0 at each pass.
//! - Allocation order and sizes are deterministic for same shapes.
//!
//! Part of #4264.

use crate::compiled_model::icb::replay::{IcbReplayBuffer, ReplayPhase, ShapeKey};

use super::CompiledKokoro;

impl CompiledKokoro {
    /// Build the pre-readback shape key from sequence length.
    ///
    /// Pre-readback segments (PlBert, TextEncoder, Prosody, Regulate) all
    /// depend on `seq_len` — the input token count determines dispatch
    /// geometry for all 4 segments.
    pub(super) fn pre_readback_shape_key(&self, seq_len: usize) -> ShapeKey {
        ShapeKey::from_single(seq_len)
    }

    /// Build the post-readback shape key from mel frames and total samples.
    ///
    /// Post-readback segments (F0Energy, HarmonicSource, Generator, iSTFT)
    /// depend on `t_mel` (derived from `total_repeats` after regulate) and
    /// `total_samples` (derived from `t_mel`).
    pub(super) fn post_readback_shape_key(
        &self,
        t_mel: usize,
        total_samples: usize,
    ) -> ShapeKey {
        ShapeKey::from_pair(t_mel, total_samples)
    }

    /// Check whether a pre-readback ICB replay is available for this shape.
    ///
    /// Returns `true` if the replay buffer has a cached ICB for the given
    /// `seq_len`. When `true`, the caller can skip normal dispatch and call
    /// [`try_replay_pre_readback`] instead.
    pub(super) fn has_pre_readback_replay(&self, seq_len: usize) -> bool {
        let key = self.pre_readback_shape_key(seq_len);
        self.icb_replay.has_cached(ReplayPhase::PreReadback, key)
    }

    /// Check whether a post-readback ICB replay is available for this shape.
    pub(super) fn has_post_readback_replay(
        &self,
        t_mel: usize,
        total_samples: usize,
    ) -> bool {
        let key = self.post_readback_shape_key(t_mel, total_samples);
        self.icb_replay.has_cached(ReplayPhase::PostReadback, key)
    }

    /// Notify the replay buffer that a forward pass completed for the given
    /// pre-readback shape. On the recording pass (second pass for a shape),
    /// this would record the ICB. Currently a no-op placeholder — actual
    /// recording requires capturing dispatch commands during execution,
    /// which will be wired when per-segment ICB recording is integrated
    /// with the `SegmentCache` dispatch path.
    ///
    /// The replay buffer's `record_segments` API expects pre-encoded
    /// `IcbReplaySegment` objects. These are produced by the per-`CompiledModel`
    /// ICB infrastructure (see `compiled_model_execute_icb_replay.rs`).
    /// This method bridges the pipeline-level shape tracking with the
    /// per-segment ICB encoding.
    pub(super) fn notify_pre_readback_complete(&mut self, seq_len: usize) {
        if !self.icb_replay.is_enabled() {
            return;
        }
        let _key = self.pre_readback_shape_key(seq_len);
        // Phase 1: Shape tracking only. ICB recording will be wired in
        // Phase 2 when per-segment dispatch command capture is integrated
        // with SegmentCache::execute_dyn_no_fence.
        //
        // The per-CompiledModel ICB replay (compiled_model_execute_icb_replay.rs)
        // handles ICB encoding at the individual step level within each
        // CompiledModel. This pipeline-level replay will aggregate those
        // per-segment ICBs into phase-level replay entries.
    }

    /// Notify the replay buffer that a forward pass completed for the given
    /// post-readback shape. See `notify_pre_readback_complete` for details.
    pub(super) fn notify_post_readback_complete(
        &mut self,
        t_mel: usize,
        total_samples: usize,
    ) {
        if !self.icb_replay.is_enabled() {
            return;
        }
        let _key = self.post_readback_shape_key(t_mel, total_samples);
        // Phase 1: Shape tracking only. See notify_pre_readback_complete.
    }

    /// Access the mutable replay buffer for direct manipulation.
    ///
    /// Used by the pipeline orchestrator to check cache state and record
    /// new entries after warmup.
    pub(super) fn icb_replay_mut(&mut self) -> &mut IcbReplayBuffer {
        &mut self.icb_replay
    }
}
