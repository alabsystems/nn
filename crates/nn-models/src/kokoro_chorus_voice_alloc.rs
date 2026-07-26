// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Voice allocation and load balancing for the Kokoro TTS chorus system.
//!
//! When rendering multi-voice TTS chorus, voices must be distributed across
//! the stereo field and processing resources. This module handles:
//!
//! - **Slot management** -- fixed-size pool of voice slots with active/inactive
//!   tracking. Allocation returns a slot index; release frees it.
//! - **Pan distribution** -- auto-assigns pan positions spread evenly across
//!   the stereo field (left=0.0, center=0.5, right=1.0).
//! - **Voice stealing** -- when all slots are occupied, the allocator can steal
//!   the oldest or quietest voice to make room for a new request.
//! - **Gain ramping** -- smooth fade-in/fade-out on voice add/remove to avoid
//!   clicks. Applied as a per-sample linear ramp multiplied into the voice buf.
//! - **Rebalancing** -- when voices are added/removed, pan positions can be
//!   redistributed evenly to maintain stereo image symmetry.
//!
//! # References
//!
//! - Pirkle, "Designing Audio Effect Plugins in C++", 2nd ed., Ch. 10
//!   (voice management and polyphony).
//! - Farnell, "Designing Sound", MIT Press, 2010 (pan-law and stereo field).

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Strategy for distributing voices across the stereo field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum AllocationStrategy {
    /// Voices assigned to slots in sequential order.
    RoundRobin,
    /// Voices spread evenly across the stereo field on allocation.
    #[default]
    SpreadPan,
    /// Higher-priority voices get preferred (center) pan positions.
    PriorityBased,
}


/// Policy for voice stealing when all slots are occupied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum VoiceStealPolicy {
    /// Steal the voice that was allocated earliest.
    #[default]
    StealOldest,
    /// Steal the voice with the lowest current gain.
    StealQuietest,
    /// Never steal -- allocation fails if no free slot exists.
    NoStealing,
}


// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the voice allocator.
///
/// Use builder methods or preset constructors. `#[non_exhaustive]` ensures
/// forward compatibility when new fields are added.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VoiceAllocConfig {
    /// Maximum number of simultaneous voice slots. Range: [1, 64]. Default: 16.
    pub max_voices: usize,
    /// How voices are distributed across the stereo field.
    pub allocation_strategy: AllocationStrategy,
    /// Whether to automatically assign pan positions on allocation.
    pub auto_pan: bool,
    /// Policy when a voice is requested but all slots are occupied.
    pub voice_stealing: VoiceStealPolicy,
    /// Duration of gain ramp in samples for voice add/remove. Default: 256.
    pub ramp_samples: usize,
}

impl Default for VoiceAllocConfig {
    fn default() -> Self {
        Self {
            max_voices: 16,
            allocation_strategy: AllocationStrategy::SpreadPan,
            auto_pan: true,
            voice_stealing: VoiceStealPolicy::StealOldest,
            ramp_samples: 256,
        }
    }
}

impl VoiceAllocConfig {
    /// Create a validated config with all parameters.
    pub fn new(
        max_voices: usize,
        allocation_strategy: AllocationStrategy,
        auto_pan: bool,
        voice_stealing: VoiceStealPolicy,
        ramp_samples: usize,
    ) -> Result<Self, KokoroError> {
        let config = Self {
            max_voices,
            allocation_strategy,
            auto_pan,
            voice_stealing,
            ramp_samples,
        };
        config.validate()?;
        Ok(config)
    }

    #[must_use]
    pub fn with_max_voices(mut self, n: usize) -> Self {
        self.max_voices = n;
        self
    }

    #[must_use]
    pub fn with_allocation_strategy(mut self, s: AllocationStrategy) -> Self {
        self.allocation_strategy = s;
        self
    }

    #[must_use]
    pub fn with_auto_pan(mut self, v: bool) -> Self {
        self.auto_pan = v;
        self
    }

    #[must_use]
    pub fn with_voice_stealing(mut self, p: VoiceStealPolicy) -> Self {
        self.voice_stealing = p;
        self
    }

    #[must_use]
    pub fn with_ramp_samples(mut self, n: usize) -> Self {
        self.ramp_samples = n;
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.max_voices < 1 || self.max_voices > 64 {
            return Err(KokoroError::InvalidConfig {
                field: "max_voices",
                reason: format!("must be in [1, 64], got {}", self.max_voices),
            });
        }
        if self.ramp_samples > 48000 {
            return Err(KokoroError::InvalidConfig {
                field: "ramp_samples",
                reason: format!("must be <= 48000, got {}", self.ramp_samples),
            });
        }
        Ok(())
    }

    // -- Presets --------------------------------------------------------------

    /// Fixed 8-voice chorus with even stereo spread.
    #[must_use]
    pub fn fixed_8() -> Self {
        Self {
            max_voices: 8,
            allocation_strategy: AllocationStrategy::SpreadPan,
            auto_pan: true,
            voice_stealing: VoiceStealPolicy::NoStealing,
            ramp_samples: 256,
        }
    }

    /// Dynamic 16-voice pool with oldest-steal policy.
    #[must_use]
    pub fn dynamic_16() -> Self {
        Self::default()
    }

    /// Solo lead: 1 primary voice + N background at reduced gain.
    ///
    /// Uses priority-based allocation so the lead voice (highest priority)
    /// gets center pan. Background voices spread to the sides.
    #[must_use]
    pub fn solo_lead() -> Self {
        Self {
            max_voices: 8,
            allocation_strategy: AllocationStrategy::PriorityBased,
            auto_pan: true,
            voice_stealing: VoiceStealPolicy::StealQuietest,
            ramp_samples: 512,
        }
    }
}

// ---------------------------------------------------------------------------
// Voice slot
// ---------------------------------------------------------------------------

/// A single voice slot in the allocator pool.
#[derive(Debug, Clone)]
pub struct VoiceSlot {
    /// Index of this slot in the pool.
    pub voice_index: usize,
    /// Stereo pan position: 0.0 = hard left, 0.5 = center, 1.0 = hard right.
    pub pan_position: f32,
    /// Priority of this voice (higher = more important). Range: [0.0, 1.0].
    pub priority: f32,
    /// Whether this slot is currently active (has a voice assigned).
    pub active: bool,
    /// Current linear gain applied to this voice. Range: [0.0, 1.0].
    pub gain: f32,
    /// Whether this voice is muted (gain applied but slot kept active).
    pub muted: bool,
    /// Allocation order (monotonically increasing counter for steal-oldest).
    alloc_order: u64,
    /// Remaining ramp samples (for fade-in on allocate, fade-out on release).
    ramp_remaining: usize,
    /// Whether currently ramping up (true) or down (false).
    ramp_up: bool,
}

impl VoiceSlot {
    fn new(index: usize) -> Self {
        Self {
            voice_index: index,
            pan_position: 0.5,
            priority: 0.0,
            active: false,
            gain: 0.0,
            muted: false,
            alloc_order: 0,
            ramp_remaining: 0,
            ramp_up: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Voice allocator
// ---------------------------------------------------------------------------

/// Voice allocator and load balancer for the Kokoro chorus system.
///
/// Manages a fixed-size pool of [`VoiceSlot`]s. Voices are allocated with
/// a priority, auto-panned across the stereo field, and gain-ramped on
/// transitions to prevent clicks.
pub struct VoiceAllocator {
    config: VoiceAllocConfig,
    slots: Vec<VoiceSlot>,
    /// Monotonic counter for allocation ordering (steal-oldest).
    alloc_counter: u64,
}

impl VoiceAllocator {
    /// Create a new allocator from a validated config.
    pub fn new(config: &VoiceAllocConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let slots = (0..config.max_voices).map(VoiceSlot::new).collect();
        Ok(Self {
            config: config.clone(),
            slots,
            alloc_counter: 0,
        })
    }

    /// Create an allocator with default configuration.
    pub fn with_defaults() -> Result<Self, KokoroError> {
        Self::new(&VoiceAllocConfig::default())
    }

    /// Allocate a voice slot with the given priority.
    ///
    /// Returns the slot index on success, or `None` if no slot is available
    /// and voice stealing is disabled or no stealable candidate exists.
    ///
    /// When `auto_pan` is enabled, the newly allocated voice receives a pan
    /// position computed from the allocation strategy. The voice starts with
    /// gain 0.0 and ramps up over `ramp_samples`.
    pub fn allocate_voice(&mut self, priority: f32) -> Option<usize> {
        let priority = priority.clamp(0.0, 1.0);

        // Try to find a free (inactive) slot.
        let free_idx = self.slots.iter().position(|s| !s.active);

        let idx = if let Some(i) = free_idx {
            i
        } else {
            // All slots occupied -- attempt voice stealing.
            self.find_steal_candidate(priority)?
        };

        self.alloc_counter += 1;
        let slot = &mut self.slots[idx];
        slot.active = true;
        slot.priority = priority;
        slot.gain = 0.0;
        slot.muted = false;
        slot.alloc_order = self.alloc_counter;
        slot.ramp_remaining = self.config.ramp_samples;
        slot.ramp_up = true;

        if self.config.auto_pan {
            self.assign_pan_positions();
        }

        Some(idx)
    }

    /// Release a voice slot, triggering a gain ramp-down.
    ///
    /// After the ramp completes (via `apply_gains`), the slot becomes
    /// inactive and available for reuse.
    pub fn release_voice(&mut self, slot: usize) {
        if slot >= self.slots.len() {
            return;
        }
        let s = &mut self.slots[slot];
        if !s.active {
            return;
        }
        // Start fade-out ramp. The slot remains active until ramp completes.
        s.ramp_remaining = self.config.ramp_samples;
        s.ramp_up = false;
    }

    /// Get references to all currently active voice slots.
    #[must_use]
    pub fn get_active_slots(&self) -> Vec<&VoiceSlot> {
        self.slots.iter().filter(|s| s.active).collect()
    }

    /// Number of currently active voices.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.active).count()
    }

    /// Total number of slots (active + inactive).
    #[must_use]
    pub fn total_slots(&self) -> usize {
        self.slots.len()
    }

    /// Redistribute pan positions evenly across all active voices.
    ///
    /// Active voices are spread from 0.0 (hard left) to 1.0 (hard right).
    /// A single active voice is panned center (0.5).
    pub fn rebalance(&mut self) {
        self.assign_pan_positions();
    }

    /// Apply per-voice gain with ramping to a set of voice buffers.
    ///
    /// `voices` must have length equal to `total_slots()`. Each inner `Vec<f32>`
    /// is the audio buffer for that voice slot. Inactive slots are zeroed.
    /// Active slots have their gain applied sample-by-sample, with linear
    /// ramp transitions for smooth fade-in/fade-out.
    ///
    /// After a fade-out ramp completes, the slot is automatically deactivated.
    pub fn apply_gains(&mut self, voices: &mut [Vec<f32>]) {
        let n_slots = self.slots.len().min(voices.len());

        for slot_idx in 0..n_slots {
            let buf = &mut voices[slot_idx];
            let slot = &mut self.slots[slot_idx];

            if !slot.active {
                // Inactive slot: zero the buffer.
                for s in buf.iter_mut() {
                    *s = 0.0;
                }
                continue;
            }

            if slot.muted {
                for s in buf.iter_mut() {
                    *s = 0.0;
                }
                continue;
            }

            let ramp_len = self.config.ramp_samples.max(1);

            for sample in buf.iter_mut() {
                // Advance ramp state.
                if slot.ramp_remaining > 0 {
                    let progress = 1.0 - (slot.ramp_remaining as f32 / ramp_len as f32);
                    slot.gain = if slot.ramp_up {
                        progress.clamp(0.0, 1.0)
                    } else {
                        (1.0 - progress).clamp(0.0, 1.0)
                    };
                    slot.ramp_remaining -= 1;

                    // Fade-out complete: deactivate.
                    if slot.ramp_remaining == 0 && !slot.ramp_up {
                        slot.active = false;
                        slot.gain = 0.0;
                    }
                }

                if !sample.is_finite() {
                    *sample = 0.0;
                } else {
                    *sample *= slot.gain;
                }
            }
        }
    }

    /// Mute a voice slot (gain goes to 0 but slot stays active).
    pub fn mute_voice(&mut self, slot: usize) {
        if slot < self.slots.len() {
            self.slots[slot].muted = true;
        }
    }

    /// Unmute a voice slot.
    pub fn unmute_voice(&mut self, slot: usize) {
        if slot < self.slots.len() {
            self.slots[slot].muted = false;
        }
    }

    /// Set the priority of an active voice slot.
    pub fn set_priority(&mut self, slot: usize, priority: f32) {
        if slot < self.slots.len() && self.slots[slot].active {
            self.slots[slot].priority = priority.clamp(0.0, 1.0);
        }
    }

    /// Reset the allocator: deactivate all slots, reset counters.
    pub fn reset(&mut self) {
        self.alloc_counter = 0;
        for (i, slot) in self.slots.iter_mut().enumerate() {
            *slot = VoiceSlot::new(i);
        }
    }

    /// Get the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &VoiceAllocConfig {
        &self.config
    }

    // -- Internal helpers -----------------------------------------------------

    /// Find a slot to steal based on the configured policy.
    fn find_steal_candidate(&self, new_priority: f32) -> Option<usize> {
        match self.config.voice_stealing {
            VoiceStealPolicy::NoStealing => None,
            VoiceStealPolicy::StealOldest => {
                // Steal the slot with the lowest alloc_order (oldest).
                self.slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.active)
                    .min_by_key(|(_, s)| s.alloc_order)
                    .map(|(i, _)| i)
            }
            VoiceStealPolicy::StealQuietest => {
                // Steal the active slot with the lowest gain, but only if
                // the new voice has higher priority.
                self.slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.active && s.priority < new_priority)
                    .min_by(|(_, a), (_, b)| {
                        a.gain
                            .partial_cmp(&b.gain)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
            }
        }
    }

    /// Assign pan positions to all active voices based on the strategy.
    fn assign_pan_positions(&mut self) {
        let active_indices: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.active)
            .map(|(i, _)| i)
            .collect();

        let n = active_indices.len();
        if n == 0 {
            return;
        }

        match self.config.allocation_strategy {
            AllocationStrategy::RoundRobin => {
                // Spread evenly by slot index.
                for (pos, &idx) in active_indices.iter().enumerate() {
                    self.slots[idx].pan_position = if n == 1 {
                        0.5
                    } else {
                        pos as f32 / (n - 1) as f32
                    };
                }
            }
            AllocationStrategy::SpreadPan => {
                // Even spread: 0.0 .. 1.0.
                for (pos, &idx) in active_indices.iter().enumerate() {
                    self.slots[idx].pan_position = if n == 1 {
                        0.5
                    } else {
                        pos as f32 / (n - 1) as f32
                    };
                }
            }
            AllocationStrategy::PriorityBased => {
                // Sort active voices by priority (highest first).
                // Highest priority gets center; lower priorities spread outward.
                let mut sorted: Vec<usize> = active_indices;
                sorted.sort_by(|&a, &b| {
                    self.slots[b]
                        .priority
                        .partial_cmp(&self.slots[a].priority)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // First voice (highest priority) = center.
                // Remaining voices alternate left/right, spreading outward.
                for (rank, &idx) in sorted.iter().enumerate() {
                    self.slots[idx].pan_position = if n == 1 { 0.5 } else { priority_pan(rank, n) };
                }
            }
        }
    }
}

/// Compute pan position for priority-based allocation.
///
/// Rank 0 (highest priority) = center (0.5).
/// Subsequent ranks alternate left/right, spreading outward.
fn priority_pan(rank: usize, total: usize) -> f32 {
    if total <= 1 {
        return 0.5;
    }
    if rank == 0 {
        return 0.5;
    }
    // Spread remaining voices symmetrically: odd ranks go left, even go right.
    let spread = rank as f32 / total as f32;
    if rank % 2 == 1 {
        (0.5 - spread * 0.5).clamp(0.0, 1.0)
    } else {
        (0.5 + spread * 0.5).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests (extracted to stay under 500-line limit)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_voice_alloc_tests.rs"]
mod tests;
