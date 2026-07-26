// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)] // ICB replay wiring in progress (#4264)

//! ICB replay infrastructure for Kokoro dispatch optimization.
//!
//! Pre-encodes Metal dispatch commands into Indirect Command Buffers (ICBs)
//! and caches them by input shape. On subsequent forward passes with the
//! same shape, the entire dispatch sequence is replayed via a single
//! `executeCommandsInBuffer` call, eliminating CPU-side pipeline lookup,
//! threadgroup calculation, and buffer binding per dispatch (~192 commands
//! per Kokoro forward pass).
//!
//! # Architecture
//!
//! The Kokoro pipeline has a CPU readback point in `step_regulate` where
//! a 4-byte scalar (`total_repeats`) is read from GPU to CPU. This breaks
//! ICB continuity because the GPU must complete all prior work before the
//! CPU can read the result. The replay infrastructure handles this by
//! splitting the dispatch sequence into segments:
//!
//! - **Pre-readback segments** (segments 0-2): PlBert, TextEncoder,
//!   ProsodyPredictor, and the regulate elementwise chain.
//! - **Post-readback segments** (segments 3-4): F0EnergyPredictor,
//!   SineGen, Generator, iSTFT.
//!
//! Each segment is independently cached and replayed. The readback point
//! forces a `submit()+sync()` between the two groups, but within each
//! group all dispatches are replayed as a single ICB execution.
//!
//! # Shape-dependent caching
//!
//! ICBs encode fixed buffer offsets and dispatch grid sizes, so they are
//! only valid for the exact input shape they were recorded with. The cache
//! key is a `ShapeKey` tuple of dimensions that determine the dispatch
//! geometry (e.g., `seq_len` for pre-readback, `t_mel` for post-readback).
//!
//! # F16 autocast limitation
//!
//! When F16 autocast is active, buffer byte widths change dynamically
//! (F32 for accumulate ops, F16 for compute ops). Pre-encoded ICB commands
//! have fixed buffer bindings with fixed byte offsets. If the autocast
//! dtype plan changes (e.g., a new op is classified differently), cached
//! ICBs become invalid. The replay infrastructure handles this by:
//!
//! 1. Storing the `IcbAutocastPlan` hash alongside each cached ICB.
//! 2. Invalidating the cache entry if the autocast plan changes.
//!
//! In practice, the autocast plan is deterministic for a given model
//! configuration, so invalidation only occurs on model reconfiguration.
//!
//! # Arena offset determinism
//!
//! ICB replay requires deterministic arena offsets: the same input shape
//! must produce the same arena allocation pattern. After warmup (one
//! forward pass per shape), the arena layout is fixed because:
//!
//! - Arena capacity is pre-sized via `ensure_capacity()` after warmup.
//! - The bump allocator is reset to offset 0 at the start of each pass.
//! - Allocation order is deterministic (same compiled step sequence).
//! - Allocation sizes are deterministic (same shapes → same byte counts).
//!
//! Part of #4264.

use std::collections::HashMap;

use crate::buffer::MetalBuffer;
use crate::dispatch::CommandBatch;
use crate::error::MetalError;

use super::IndirectCommandBuffer;

/// Shape key for ICB cache lookup.
///
/// Encodes the dimensions that determine dispatch geometry and buffer
/// offsets. Different shape keys produce different ICBs. Using a
/// fixed-size array avoids heap allocation on the lookup hot path.
///
/// For Kokoro:
/// - Pre-readback key: `[seq_len, 0, 0, 0]`
/// - Post-readback key: `[t_mel, total_samples, 0, 0]`
///
/// The trailing zeros are padding for future use (e.g., batch size
/// dimension when batch > 1 is supported).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ShapeKey([usize; 4]);

impl ShapeKey {
    /// Create a shape key from a single dimension (e.g., seq_len).
    pub(crate) fn from_single(dim: usize) -> Self {
        Self([dim, 0, 0, 0])
    }

    /// Create a shape key from two dimensions (e.g., t_mel, total_samples).
    pub(crate) fn from_pair(dim0: usize, dim1: usize) -> Self {
        Self([dim0, dim1, 0, 0])
    }

    /// Create a shape key from an arbitrary slice (up to 4 dims).
    /// Extra dims beyond 4 are silently ignored.
    pub(crate) fn from_dims(dims: &[usize]) -> Self {
        let mut key = [0usize; 4];
        for (i, &d) in dims.iter().take(4).enumerate() {
            key[i] = d;
        }
        Self(key)
    }

    /// Returns the underlying dimension array.
    #[cfg(test)]
    pub(crate) fn dims(&self) -> &[usize; 4] {
        &self.0
    }
}

/// A single recorded ICB segment with its associated metadata.
///
/// Stores a pre-encoded `IndirectCommandBuffer` plus the resource
/// buffers it references. On replay, the ICB is executed via a single
/// `executeCommandsInBuffer` call with the resource list declared for
/// GPU access.
pub(crate) struct IcbReplaySegment {
    /// The pre-encoded indirect command buffer.
    icb: IndirectCommandBuffer,
    /// Resource buffers the ICB reads from or writes to.
    /// Must be declared via `useResource:usage:` before ICB execution.
    /// Stored as owned `MetalBuffer`s to keep them alive (ObjC ARC).
    resource_buffers: Vec<MetalBuffer>,
    /// Number of dispatch commands in this segment.
    command_count: usize,
    /// Human-readable label for diagnostics (e.g., "pre_readback_0").
    label: String,
    /// Hash of the autocast plan at recording time, for invalidation.
    /// 0 when autocast is not active.
    autocast_plan_hash: u64,
}

impl IcbReplaySegment {
    /// Execute this segment's ICB on the given command batch.
    ///
    /// Declares all resource buffers for GPU read/write access, then
    /// replays the pre-encoded commands via `executeCommandsInBuffer`.
    pub(crate) fn replay(&self, batch: &CommandBatch) -> Result<(), MetalError> {
        if self.command_count == 0 {
            return Ok(());
        }
        let refs: Vec<&MetalBuffer> = self.resource_buffers.iter().collect();
        self.icb.execute(batch, &refs)
    }

    /// Number of pre-encoded dispatch commands.
    pub(crate) fn command_count(&self) -> usize {
        self.command_count
    }

    /// Diagnostic label.
    pub(crate) fn label(&self) -> &str {
        &self.label
    }

    /// Autocast plan hash at recording time.
    pub(crate) fn autocast_plan_hash(&self) -> u64 {
        self.autocast_plan_hash
    }
}

impl std::fmt::Debug for IcbReplaySegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcbReplaySegment")
            .field("command_count", &self.command_count)
            .field("resource_buffers", &self.resource_buffers.len())
            .field("label", &self.label)
            .field("autocast_plan_hash", &self.autocast_plan_hash)
            .finish()
    }
}

/// Configuration for ICB replay behavior.
///
/// Controls whether ICB replay is enabled, cache capacity limits,
/// and invalidation policy. Designed for safe rollout: disabled by
/// default, opt-in via `use_icb_replay`.
#[derive(Debug, Clone)]
pub(crate) struct IcbReplayConfig {
    /// Master switch: when false, all record/replay calls are no-ops.
    /// Default: false (safe rollout).
    pub(crate) use_icb_replay: bool,
    /// Maximum number of shape keys to cache per segment group.
    /// Eviction is LRU. Default: 8 (covers common Kokoro seq_lens).
    pub(crate) max_cached_shapes: usize,
    /// Minimum number of dispatch commands for a segment to be worth
    /// recording. Below this, the ICB creation overhead exceeds the
    /// per-dispatch savings. Default: 4.
    pub(crate) min_commands_per_segment: usize,
    /// When true, validate arena offsets on replay by comparing against
    /// the recorded offsets. Catches arena non-determinism bugs at the
    /// cost of a small CPU-side check per replay. Default: true in debug
    /// builds, false in release.
    pub(crate) validate_arena_offsets: bool,
}

impl Default for IcbReplayConfig {
    fn default() -> Self {
        Self {
            use_icb_replay: false,
            max_cached_shapes: 8,
            min_commands_per_segment: 4,
            validate_arena_offsets: cfg!(debug_assertions),
        }
    }
}

impl IcbReplayConfig {
    /// Create a config with ICB replay enabled.
    pub(crate) fn enabled() -> Self {
        Self {
            use_icb_replay: true,
            ..Default::default()
        }
    }

    /// Create a config with ICB replay enabled and arena validation on.
    #[cfg(test)]
    pub(crate) fn enabled_with_validation() -> Self {
        Self {
            use_icb_replay: true,
            validate_arena_offsets: true,
            ..Default::default()
        }
    }
}

/// Phase of the Kokoro pipeline relative to the CPU readback point.
///
/// The `step_regulate` scalar readback splits the pipeline into two
/// independently replayable halves. Each half can have multiple ICB
/// segments if non-ICB-eligible steps (NativeOps, RuntimeOps) break
/// continuity within the half.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ReplayPhase {
    /// Segments 0-2 + regulate elementwise chain, before the
    /// `total_repeats` scalar readback in `step_regulate`.
    PreReadback,
    /// Segments 3-4 + SineGen + Generator + iSTFT, after the
    /// `total_repeats` scalar readback.
    PostReadback,
}

impl std::fmt::Display for ReplayPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreReadback => write!(f, "pre_readback"),
            Self::PostReadback => write!(f, "post_readback"),
        }
    }
}

/// Cache entry: one or more ICB replay segments for a specific shape.
struct ShapeCacheEntry {
    /// Ordered replay segments for this shape.
    segments: Vec<IcbReplaySegment>,
    /// Arena byte offsets recorded during the recording pass.
    /// Used for validation on replay when `validate_arena_offsets` is true.
    recorded_arena_offsets: Vec<usize>,
    /// Number of times this entry has been replayed (for LRU eviction).
    replay_count: u64,
    /// Total dispatch commands across all segments.
    total_commands: usize,
}

impl std::fmt::Debug for ShapeCacheEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShapeCacheEntry")
            .field("segments", &self.segments.len())
            .field("total_commands", &self.total_commands)
            .field("replay_count", &self.replay_count)
            .field(
                "recorded_arena_offsets",
                &self.recorded_arena_offsets.len(),
            )
            .finish()
    }
}

/// Per-phase ICB replay cache.
///
/// Maps `ShapeKey -> ShapeCacheEntry` for one pipeline phase.
/// Entries are evicted LRU when the cache exceeds `max_cached_shapes`.
struct PhaseCache {
    entries: HashMap<ShapeKey, ShapeCacheEntry>,
    max_entries: usize,
    /// Total replays served from this cache (for diagnostics).
    total_replays: u64,
    /// Total recordings into this cache.
    total_recordings: u64,
    /// Total cache misses (shape not found).
    total_misses: u64,
}

impl PhaseCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            total_replays: 0,
            total_recordings: 0,
            total_misses: 0,
        }
    }

    /// Evict the least-recently-used entry if at capacity.
    fn maybe_evict(&mut self) {
        if self.entries.len() < self.max_entries {
            return;
        }
        // Find the entry with the lowest replay_count.
        let evict_key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.replay_count)
            .map(|(key, _)| *key);
        if let Some(key) = evict_key {
            self.entries.remove(&key);
        }
    }
}

impl std::fmt::Debug for PhaseCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhaseCache")
            .field("entries", &self.entries.len())
            .field("max_entries", &self.max_entries)
            .field("total_replays", &self.total_replays)
            .field("total_recordings", &self.total_recordings)
            .field("total_misses", &self.total_misses)
            .finish()
    }
}

/// ICB replay buffer for Kokoro dispatch optimization.
///
/// Caches pre-encoded Metal Indirect Command Buffers by input shape,
/// split into pre-readback and post-readback phases around the
/// `step_regulate` scalar readback point.
///
/// # Usage
///
/// ```rust,ignore
/// let mut replay = IcbReplayBuffer::new(IcbReplayConfig::enabled());
///
/// // First pass: record
/// let shape_key = ShapeKey::from_single(seq_len);
/// if !replay.has_cached(ReplayPhase::PreReadback, shape_key) {
///     // Run dispatches normally, collecting recording data...
///     replay.record_segment(
///         ReplayPhase::PreReadback,
///         shape_key,
///         segments,
///         arena_offsets,
///     )?;
/// }
///
/// // Subsequent passes: replay
/// if let Some(stats) = replay.try_replay(
///     ReplayPhase::PreReadback,
///     shape_key,
///     &batch,
/// )? {
///     // Skipped N dispatches via ICB replay.
/// }
/// ```
///
/// # Thread safety
///
/// `IcbReplayBuffer` is NOT `Sync`. Each `CompiledKokoro` instance
/// owns its own replay buffer. For multi-voice sharing, each voice
/// clone gets its own replay buffer (lightweight: only caches, no
/// model weights).
///
/// Part of #4264.
pub(crate) struct IcbReplayBuffer {
    config: IcbReplayConfig,
    pre_readback: PhaseCache,
    post_readback: PhaseCache,
}

impl IcbReplayBuffer {
    /// Create a new replay buffer with the given configuration.
    pub(crate) fn new(config: IcbReplayConfig) -> Self {
        let max = config.max_cached_shapes;
        Self {
            config,
            pre_readback: PhaseCache::new(max),
            post_readback: PhaseCache::new(max),
        }
    }

    /// Whether ICB replay is enabled.
    pub(crate) fn is_enabled(&self) -> bool {
        self.config.use_icb_replay
    }

    /// Check whether a cached ICB exists for the given phase and shape.
    pub(crate) fn has_cached(&self, phase: ReplayPhase, key: ShapeKey) -> bool {
        if !self.config.use_icb_replay {
            return false;
        }
        self.phase_cache(phase).entries.contains_key(&key)
    }

    /// Record a set of ICB replay segments for the given phase and shape.
    ///
    /// Evicts the LRU entry if the cache is at capacity. Overwrites any
    /// existing entry for the same shape key (handles autocast plan changes).
    ///
    /// # Arguments
    ///
    /// * `phase` - Pipeline phase (pre-readback or post-readback).
    /// * `key` - Shape key identifying the input dimensions.
    /// * `segments` - Pre-encoded ICB replay segments in execution order.
    /// * `arena_offsets` - Arena byte offsets recorded during this pass.
    ///   Used for validation on subsequent replays.
    pub(crate) fn record_segments(
        &mut self,
        phase: ReplayPhase,
        key: ShapeKey,
        segments: Vec<IcbReplaySegment>,
        arena_offsets: Vec<usize>,
    ) {
        if !self.config.use_icb_replay {
            return;
        }
        if segments.is_empty() {
            return;
        }

        let total_commands: usize = segments.iter().map(|s| s.command_count).sum();
        if total_commands < self.config.min_commands_per_segment {
            return;
        }

        let cache = self.phase_cache_mut(phase);
        cache.maybe_evict();
        cache.total_recordings += 1;
        cache.entries.insert(
            key,
            ShapeCacheEntry {
                segments,
                recorded_arena_offsets: arena_offsets,
                replay_count: 0,
                total_commands,
            },
        );
    }

    /// Attempt to replay cached ICBs for the given phase and shape.
    ///
    /// Returns `Ok(Some(stats))` if replay succeeded (all segments
    /// replayed), `Ok(None)` if no cache entry exists (caller should
    /// fall back to direct dispatch), or `Err` on Metal API failure.
    ///
    /// # Arena offset validation
    ///
    /// When `validate_arena_offsets` is true, compares the current
    /// arena offsets against the recorded offsets. A mismatch indicates
    /// arena non-determinism and the entry is invalidated (removed from
    /// cache), returning `Ok(None)` so the caller falls back to direct
    /// dispatch.
    pub(crate) fn try_replay(
        &mut self,
        phase: ReplayPhase,
        key: ShapeKey,
        batch: &CommandBatch,
        current_arena_offsets: Option<&[usize]>,
    ) -> Result<Option<ReplayStats>, MetalError> {
        if !self.config.use_icb_replay {
            return Ok(None);
        }

        // Read config flags before taking mutable borrows on the cache.
        let validate_offsets = self.config.validate_arena_offsets;

        let cache = self.phase_cache_mut(phase);
        let entry = match cache.entries.get_mut(&key) {
            Some(e) => e,
            None => {
                cache.total_misses += 1;
                return Ok(None);
            }
        };

        // Validate arena offsets if configured and provided.
        if validate_offsets {
            if let Some(current) = current_arena_offsets {
                if current != entry.recorded_arena_offsets.as_slice() {
                    // Arena layout changed — ICB bindings are invalid.
                    // Remove the stale entry and fall back to direct dispatch.
                    cache.entries.remove(&key);
                    cache.total_misses += 1;
                    return Ok(None);
                }
            }
        }

        // Replay all segments in order.
        let mut total_commands = 0;
        let segments_count = entry.segments.len();
        for segment in &entry.segments {
            segment.replay(batch)?;
            total_commands += segment.command_count();
        }
        entry.replay_count += 1;
        let replay_count = entry.replay_count;

        cache.total_replays += 1;

        Ok(Some(ReplayStats {
            commands_replayed: total_commands,
            segments_replayed: segments_count,
            cumulative_replays: replay_count,
        }))
    }

    /// Remove all cached entries for a specific shape key across both phases.
    ///
    /// Use when the autocast plan changes or the model is reconfigured.
    pub(crate) fn invalidate_shape(&mut self, key: ShapeKey) {
        self.pre_readback.entries.remove(&key);
        self.post_readback.entries.remove(&key);
    }

    /// Remove all cached entries across all phases.
    ///
    /// Use on model reconfiguration, autocast policy change, or
    /// precision contract change.
    pub(crate) fn invalidate_all(&mut self) {
        self.pre_readback.entries.clear();
        self.post_readback.entries.clear();
    }

    /// Current reference to the replay configuration.
    pub(crate) fn config(&self) -> &IcbReplayConfig {
        &self.config
    }

    /// Diagnostic summary of cache state.
    pub(crate) fn stats(&self) -> IcbReplayBufferStats {
        IcbReplayBufferStats {
            enabled: self.config.use_icb_replay,
            pre_readback_entries: self.pre_readback.entries.len(),
            post_readback_entries: self.post_readback.entries.len(),
            pre_readback_total_replays: self.pre_readback.total_replays,
            post_readback_total_replays: self.post_readback.total_replays,
            pre_readback_total_recordings: self.pre_readback.total_recordings,
            post_readback_total_recordings: self.post_readback.total_recordings,
            pre_readback_total_misses: self.pre_readback.total_misses,
            post_readback_total_misses: self.post_readback.total_misses,
            total_cached_commands: self.total_cached_commands(),
        }
    }

    /// Total dispatch commands cached across all phases and shapes.
    fn total_cached_commands(&self) -> usize {
        let pre: usize = self
            .pre_readback
            .entries
            .values()
            .map(|e| e.total_commands)
            .sum();
        let post: usize = self
            .post_readback
            .entries
            .values()
            .map(|e| e.total_commands)
            .sum();
        pre + post
    }

    fn phase_cache(&self, phase: ReplayPhase) -> &PhaseCache {
        match phase {
            ReplayPhase::PreReadback => &self.pre_readback,
            ReplayPhase::PostReadback => &self.post_readback,
        }
    }

    fn phase_cache_mut(&mut self, phase: ReplayPhase) -> &mut PhaseCache {
        match phase {
            ReplayPhase::PreReadback => &mut self.pre_readback,
            ReplayPhase::PostReadback => &mut self.post_readback,
        }
    }
}

impl std::fmt::Debug for IcbReplayBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcbReplayBuffer")
            .field("config", &self.config)
            .field("pre_readback", &self.pre_readback)
            .field("post_readback", &self.post_readback)
            .finish()
    }
}

/// Statistics from a successful ICB replay.
#[derive(Debug, Clone)]
pub(crate) struct ReplayStats {
    /// Number of Metal dispatch commands replayed via ICB.
    pub(crate) commands_replayed: usize,
    /// Number of ICB segments executed.
    pub(crate) segments_replayed: usize,
    /// Cumulative replay count for this shape key.
    pub(crate) cumulative_replays: u64,
}

/// Diagnostic summary of the replay buffer state.
#[derive(Debug, Clone)]
pub(crate) struct IcbReplayBufferStats {
    /// Whether ICB replay is enabled.
    pub(crate) enabled: bool,
    /// Number of cached shape entries in pre-readback phase.
    pub(crate) pre_readback_entries: usize,
    /// Number of cached shape entries in post-readback phase.
    pub(crate) post_readback_entries: usize,
    /// Total successful replays from pre-readback cache.
    pub(crate) pre_readback_total_replays: u64,
    /// Total successful replays from post-readback cache.
    pub(crate) post_readback_total_replays: u64,
    /// Total recordings into pre-readback cache.
    pub(crate) pre_readback_total_recordings: u64,
    /// Total recordings into post-readback cache.
    pub(crate) post_readback_total_recordings: u64,
    /// Total cache misses from pre-readback phase.
    pub(crate) pre_readback_total_misses: u64,
    /// Total cache misses from post-readback phase.
    pub(crate) post_readback_total_misses: u64,
    /// Total dispatch commands cached across all entries.
    pub(crate) total_cached_commands: usize,
}

impl std::fmt::Display for IcbReplayBufferStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IcbReplay(enabled={}, pre={}/{} hit/miss, post={}/{} hit/miss, cached_cmds={})",
            self.enabled,
            self.pre_readback_total_replays,
            self.pre_readback_total_misses,
            self.post_readback_total_replays,
            self.post_readback_total_misses,
            self.total_cached_commands,
        )
    }
}

/// Builder for constructing `IcbReplaySegment` from recorded dispatch data.
///
/// Collects buffer bindings and dispatch commands during a forward pass,
/// then finalizes into an `IcbReplaySegment` that can be cached for
/// future replay.
pub(crate) struct IcbReplayRecorder {
    /// Collected resource buffers (deduped by pointer identity).
    resource_buffers: Vec<MetalBuffer>,
    /// Set of buffer pointers already added (for deduplication).
    seen_buffer_ptrs: std::collections::HashSet<usize>,
    /// Number of commands recorded so far.
    command_count: usize,
    /// Label for the segment being recorded.
    label: String,
    /// Autocast plan hash at recording time.
    autocast_plan_hash: u64,
}

impl IcbReplayRecorder {
    /// Start recording a new segment.
    pub(crate) fn new(label: impl Into<String>, autocast_plan_hash: u64) -> Self {
        Self {
            resource_buffers: Vec::new(),
            seen_buffer_ptrs: std::collections::HashSet::new(),
            command_count: 0,
            label: label.into(),
            autocast_plan_hash,
        }
    }

    /// Register a buffer as a resource for the ICB.
    ///
    /// Deduplicates by pointer identity: if the same `MetalBuffer` is
    /// registered multiple times (common for weight buffers shared across
    /// steps), only one copy is retained.
    pub(crate) fn add_resource_buffer(&mut self, buffer: &MetalBuffer) {
        let ptr = std::ptr::from_ref(buffer.inner()) as usize;
        if self.seen_buffer_ptrs.insert(ptr) {
            self.resource_buffers.push(buffer.alias());
        }
    }

    /// Increment the recorded command count.
    pub(crate) fn add_commands(&mut self, count: usize) {
        self.command_count += count;
    }

    /// Finalize the recording into an `IcbReplaySegment`.
    ///
    /// Consumes the recorder and wraps the collected data with the
    /// provided pre-encoded `IndirectCommandBuffer`.
    pub(crate) fn finalize(self, icb: IndirectCommandBuffer) -> IcbReplaySegment {
        IcbReplaySegment {
            icb,
            resource_buffers: self.resource_buffers,
            command_count: self.command_count,
            label: self.label,
            autocast_plan_hash: self.autocast_plan_hash,
        }
    }

    /// Current command count.
    pub(crate) fn command_count(&self) -> usize {
        self.command_count
    }
}

#[cfg(test)]
#[path = "compiled_model_icb_replay_tests.rs"]
mod tests;
