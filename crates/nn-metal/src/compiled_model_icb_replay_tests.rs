// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ICB replay infrastructure.
//!
//! These are unit tests for the cache, shape key, and configuration logic.
//! Integration tests with actual Metal ICBs require GPU access and live
//! in the Kokoro gate tests.

use super::*;

// ---------------------------------------------------------------------------
// ShapeKey tests
// ---------------------------------------------------------------------------

#[test]
fn test_shape_key_from_single() {
    let key = ShapeKey::from_single(128);
    assert_eq!(key.dims(), &[128, 0, 0, 0]);
}

#[test]
fn test_shape_key_from_pair() {
    let key = ShapeKey::from_pair(256, 44100);
    assert_eq!(key.dims(), &[256, 44100, 0, 0]);
}

#[test]
fn test_shape_key_from_dims_empty() {
    let key = ShapeKey::from_dims(&[]);
    assert_eq!(key.dims(), &[0, 0, 0, 0]);
}

#[test]
fn test_shape_key_from_dims_overflow() {
    // More than 4 dims: extras silently ignored.
    let key = ShapeKey::from_dims(&[1, 2, 3, 4, 5, 6]);
    assert_eq!(key.dims(), &[1, 2, 3, 4]);
}

#[test]
fn test_shape_key_equality() {
    let a = ShapeKey::from_single(42);
    let b = ShapeKey::from_dims(&[42]);
    assert_eq!(a, b);
}

#[test]
fn test_shape_key_hash_consistency() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let a = ShapeKey::from_single(100);
    let b = ShapeKey::from_single(100);
    let mut ha = DefaultHasher::new();
    let mut hb = DefaultHasher::new();
    a.hash(&mut ha);
    b.hash(&mut hb);
    assert_eq!(ha.finish(), hb.finish());
}

// ---------------------------------------------------------------------------
// IcbReplayConfig tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_default_disabled() {
    let config = IcbReplayConfig::default();
    assert!(!config.use_icb_replay);
    assert_eq!(config.max_cached_shapes, 8);
    assert_eq!(config.min_commands_per_segment, 4);
}

#[test]
fn test_config_enabled() {
    let config = IcbReplayConfig::enabled();
    assert!(config.use_icb_replay);
}

#[test]
fn test_config_enabled_with_validation() {
    let config = IcbReplayConfig::enabled_with_validation();
    assert!(config.use_icb_replay);
    assert!(config.validate_arena_offsets);
}

// ---------------------------------------------------------------------------
// IcbReplayBuffer tests (no GPU required — tests cache logic only)
// ---------------------------------------------------------------------------

#[test]
fn test_buffer_disabled_has_cached_returns_false() {
    let buf = IcbReplayBuffer::new(IcbReplayConfig::default());
    let key = ShapeKey::from_single(64);
    assert!(!buf.has_cached(ReplayPhase::PreReadback, key));
}

#[test]
fn test_buffer_enabled_empty_has_cached_returns_false() {
    let buf = IcbReplayBuffer::new(IcbReplayConfig::enabled());
    let key = ShapeKey::from_single(64);
    assert!(!buf.has_cached(ReplayPhase::PreReadback, key));
    assert!(!buf.has_cached(ReplayPhase::PostReadback, key));
}

#[test]
fn test_buffer_disabled_record_is_noop() {
    let mut buf = IcbReplayBuffer::new(IcbReplayConfig::default());
    let key = ShapeKey::from_single(64);
    // record_segments does nothing when disabled.
    buf.record_segments(ReplayPhase::PreReadback, key, Vec::new(), Vec::new());
    assert!(!buf.has_cached(ReplayPhase::PreReadback, key));
}

#[test]
fn test_buffer_stats_initial() {
    let buf = IcbReplayBuffer::new(IcbReplayConfig::enabled());
    let stats = buf.stats();
    assert!(stats.enabled);
    assert_eq!(stats.pre_readback_entries, 0);
    assert_eq!(stats.post_readback_entries, 0);
    assert_eq!(stats.total_cached_commands, 0);
}

#[test]
fn test_buffer_stats_display() {
    let buf = IcbReplayBuffer::new(IcbReplayConfig::enabled());
    let display = format!("{}", buf.stats());
    assert!(display.contains("IcbReplay"));
    assert!(display.contains("enabled=true"));
}

#[test]
fn test_buffer_invalidate_all() {
    let mut buf = IcbReplayBuffer::new(IcbReplayConfig::enabled());
    // No entries to clear, but should not panic.
    buf.invalidate_all();
    assert_eq!(buf.stats().pre_readback_entries, 0);
}

#[test]
fn test_buffer_invalidate_shape() {
    let mut buf = IcbReplayBuffer::new(IcbReplayConfig::enabled());
    let key = ShapeKey::from_single(64);
    // No entry exists, should not panic.
    buf.invalidate_shape(key);
}

// ---------------------------------------------------------------------------
// ReplayPhase tests
// ---------------------------------------------------------------------------

#[test]
fn test_replay_phase_display() {
    assert_eq!(format!("{}", ReplayPhase::PreReadback), "pre_readback");
    assert_eq!(format!("{}", ReplayPhase::PostReadback), "post_readback");
}

#[test]
fn test_replay_phase_equality() {
    assert_eq!(ReplayPhase::PreReadback, ReplayPhase::PreReadback);
    assert_ne!(ReplayPhase::PreReadback, ReplayPhase::PostReadback);
}

// ---------------------------------------------------------------------------
// IcbReplayRecorder tests (no GPU required)
// ---------------------------------------------------------------------------

#[test]
fn test_recorder_initial_state() {
    let recorder = IcbReplayRecorder::new("test_segment", 0);
    assert_eq!(recorder.command_count(), 0);
}

#[test]
fn test_recorder_add_commands() {
    let mut recorder = IcbReplayRecorder::new("test_segment", 42);
    recorder.add_commands(5);
    recorder.add_commands(3);
    assert_eq!(recorder.command_count(), 8);
}

// ---------------------------------------------------------------------------
// PhaseCache eviction tests
// ---------------------------------------------------------------------------

#[test]
fn test_phase_cache_eviction_lru() {
    let mut cache = PhaseCache::new(2);

    // Insert two entries.
    cache.entries.insert(
        ShapeKey::from_single(64),
        ShapeCacheEntry {
            segments: Vec::new(),
            recorded_arena_offsets: Vec::new(),
            replay_count: 1,
            total_commands: 10,
        },
    );
    cache.entries.insert(
        ShapeKey::from_single(128),
        ShapeCacheEntry {
            segments: Vec::new(),
            recorded_arena_offsets: Vec::new(),
            replay_count: 5,
            total_commands: 20,
        },
    );
    assert_eq!(cache.entries.len(), 2);

    // Eviction should remove the entry with replay_count=1 (key=64).
    cache.maybe_evict();
    assert_eq!(cache.entries.len(), 1);
    assert!(cache.entries.contains_key(&ShapeKey::from_single(128)));
    assert!(!cache.entries.contains_key(&ShapeKey::from_single(64)));
}

#[test]
fn test_phase_cache_no_eviction_below_capacity() {
    let mut cache = PhaseCache::new(5);
    cache.entries.insert(
        ShapeKey::from_single(64),
        ShapeCacheEntry {
            segments: Vec::new(),
            recorded_arena_offsets: Vec::new(),
            replay_count: 0,
            total_commands: 0,
        },
    );
    cache.maybe_evict();
    assert_eq!(cache.entries.len(), 1);
}
