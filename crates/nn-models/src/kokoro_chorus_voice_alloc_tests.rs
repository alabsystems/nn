// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `kokoro_chorus_voice_alloc` -- voice allocation and load balancing.

use super::*;

#[test]
fn test_config_default_is_valid() {
    let config = VoiceAllocConfig::default();
    config.validate().expect("default config should be valid");
    assert_eq!(config.max_voices, 16);
    assert!(config.auto_pan);
    assert_eq!(config.allocation_strategy, AllocationStrategy::SpreadPan);
    assert_eq!(config.voice_stealing, VoiceStealPolicy::StealOldest);
}

#[test]
fn test_config_validation_bounds() {
    assert!(VoiceAllocConfig::new(
        0,
        AllocationStrategy::SpreadPan,
        true,
        VoiceStealPolicy::StealOldest,
        256
    )
    .is_err());
    assert!(VoiceAllocConfig::new(
        65,
        AllocationStrategy::SpreadPan,
        true,
        VoiceStealPolicy::StealOldest,
        256
    )
    .is_err());
    assert!(VoiceAllocConfig::new(
        1,
        AllocationStrategy::SpreadPan,
        true,
        VoiceStealPolicy::StealOldest,
        256
    )
    .is_ok());
    assert!(VoiceAllocConfig::new(
        64,
        AllocationStrategy::SpreadPan,
        true,
        VoiceStealPolicy::StealOldest,
        256
    )
    .is_ok());
    assert!(VoiceAllocConfig::new(
        8,
        AllocationStrategy::SpreadPan,
        true,
        VoiceStealPolicy::StealOldest,
        50000
    )
    .is_err());
}

#[test]
fn test_config_builder() {
    let config = VoiceAllocConfig::default()
        .with_max_voices(8)
        .with_allocation_strategy(AllocationStrategy::PriorityBased)
        .with_auto_pan(false)
        .with_voice_stealing(VoiceStealPolicy::NoStealing)
        .with_ramp_samples(128);
    config.validate().expect("builder config should be valid");
    assert_eq!(config.max_voices, 8);
    assert!(!config.auto_pan);
}

#[test]
fn test_preset_fixed_8() {
    let config = VoiceAllocConfig::fixed_8();
    config.validate().expect("fixed_8 preset should be valid");
    assert_eq!(config.max_voices, 8);
    assert_eq!(config.voice_stealing, VoiceStealPolicy::NoStealing);
}

#[test]
fn test_preset_dynamic_16() {
    let config = VoiceAllocConfig::dynamic_16();
    config
        .validate()
        .expect("dynamic_16 preset should be valid");
    assert_eq!(config.max_voices, 16);
}

#[test]
fn test_preset_solo_lead() {
    let config = VoiceAllocConfig::solo_lead();
    config.validate().expect("solo_lead preset should be valid");
    assert_eq!(
        config.allocation_strategy,
        AllocationStrategy::PriorityBased
    );
    assert_eq!(config.voice_stealing, VoiceStealPolicy::StealQuietest);
}

#[test]
fn test_allocate_and_release() {
    let config = VoiceAllocConfig::default().with_max_voices(4);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    assert_eq!(alloc.active_count(), 0);

    let s0 = alloc.allocate_voice(0.5).expect("should allocate");
    assert_eq!(alloc.active_count(), 1);
    assert!(alloc.slots[s0].active);

    let s1 = alloc.allocate_voice(0.8).expect("should allocate");
    assert_eq!(alloc.active_count(), 2);

    alloc.release_voice(s0);
    // Slot still active until ramp completes.
    assert!(alloc.slots[s0].active);
    assert!(!alloc.slots[s0].ramp_up);

    // Simulate ramp completion via apply_gains.
    let mut voices: Vec<Vec<f32>> = vec![vec![0.1; 512]; 4];
    alloc.apply_gains(&mut voices);

    // After ramp-down, slot should be inactive.
    assert!(!alloc.slots[s0].active);
    assert!(alloc.slots[s1].active);
}

#[test]
fn test_allocate_fills_all_slots() {
    let config = VoiceAllocConfig::default()
        .with_max_voices(3)
        .with_voice_stealing(VoiceStealPolicy::NoStealing);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    assert!(alloc.allocate_voice(0.5).is_some());
    assert!(alloc.allocate_voice(0.5).is_some());
    assert!(alloc.allocate_voice(0.5).is_some());
    // No more slots, no stealing.
    assert!(alloc.allocate_voice(0.5).is_none());
}

#[test]
fn test_voice_stealing_oldest() {
    let config = VoiceAllocConfig::default()
        .with_max_voices(2)
        .with_voice_stealing(VoiceStealPolicy::StealOldest);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    let s0 = alloc.allocate_voice(0.5).expect("slot 0");
    let _s1 = alloc.allocate_voice(0.5).expect("slot 1");

    // All full; allocate should steal the oldest (s0).
    let stolen = alloc.allocate_voice(0.9).expect("should steal oldest");
    assert_eq!(stolen, s0);
}

#[test]
fn test_voice_stealing_quietest() {
    let config = VoiceAllocConfig::default()
        .with_max_voices(2)
        .with_voice_stealing(VoiceStealPolicy::StealQuietest)
        .with_ramp_samples(0);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    alloc.allocate_voice(0.3).expect("slot 0");
    alloc.allocate_voice(0.3).expect("slot 1");

    // Set gains manually (simulating ramp completion).
    alloc.slots[0].gain = 0.8;
    alloc.slots[1].gain = 0.2;

    // New voice with higher priority should steal the quietest.
    let stolen = alloc.allocate_voice(0.9).expect("should steal quietest");
    assert_eq!(stolen, 1); // slot 1 had lower gain.
}

#[test]
fn test_steal_quietest_respects_priority() {
    let config = VoiceAllocConfig::default()
        .with_max_voices(2)
        .with_voice_stealing(VoiceStealPolicy::StealQuietest);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    alloc.allocate_voice(0.9).expect("slot 0 high priority");
    alloc.allocate_voice(0.9).expect("slot 1 high priority");

    // Low-priority voice cannot steal high-priority voices.
    assert!(alloc.allocate_voice(0.1).is_none());
}

#[test]
fn test_pan_spread_single_voice() {
    let config = VoiceAllocConfig::default().with_max_voices(4);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    let s = alloc.allocate_voice(0.5).expect("slot");
    assert!((alloc.slots[s].pan_position - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_pan_spread_two_voices() {
    let config = VoiceAllocConfig::default().with_max_voices(4);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    let s0 = alloc.allocate_voice(0.5).expect("slot 0");
    let s1 = alloc.allocate_voice(0.5).expect("slot 1");

    // Two voices should be panned hard left (0.0) and hard right (1.0).
    let pan0 = alloc.slots[s0].pan_position;
    let pan1 = alloc.slots[s1].pan_position;
    assert!((pan0 - 0.0).abs() < f32::EPSILON, "first voice pan: {pan0}");
    assert!(
        (pan1 - 1.0).abs() < f32::EPSILON,
        "second voice pan: {pan1}"
    );
}

#[test]
fn test_rebalance_after_release() {
    let config = VoiceAllocConfig::default()
        .with_max_voices(4)
        .with_ramp_samples(0);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    alloc.allocate_voice(0.5).expect("slot 0");
    alloc.allocate_voice(0.5).expect("slot 1");
    let s2 = alloc.allocate_voice(0.5).expect("slot 2");

    // Release middle voice and complete ramp immediately.
    alloc.release_voice(s2);
    let mut voices: Vec<Vec<f32>> = vec![vec![0.1; 1]; 4];
    alloc.apply_gains(&mut voices);

    alloc.rebalance();

    // After rebalance, remaining 2 active voices should be at 0.0 and 1.0.
    let active = alloc.get_active_slots();
    assert_eq!(active.len(), 2);
    let pans: Vec<f32> = active.iter().map(|s| s.pan_position).collect();
    assert!((pans[0] - 0.0).abs() < f32::EPSILON || (pans[0] - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_apply_gains_ramp_up() {
    let config = VoiceAllocConfig::default()
        .with_max_voices(2)
        .with_ramp_samples(4);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    alloc.allocate_voice(0.5).expect("slot");

    let mut voices = vec![vec![1.0; 8], vec![1.0; 8]];
    alloc.apply_gains(&mut voices);

    // First 4 samples should ramp from 0 to ~1.
    let buf = &voices[0];
    assert!(
        buf[0] < 0.5,
        "sample 0 should be low (ramping), got {}",
        buf[0]
    );
    assert!(
        buf[3] > 0.5,
        "sample 3 should be high (near end of ramp), got {}",
        buf[3]
    );
    // After ramp, gain should be 1.0.
    assert!(
        (buf[7] - 1.0).abs() < f32::EPSILON,
        "sample 7 should be 1.0, got {}",
        buf[7]
    );
}

#[test]
fn test_apply_gains_zeros_inactive() {
    let config = VoiceAllocConfig::default().with_max_voices(2);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    let mut voices = vec![vec![0.5; 10], vec![0.5; 10]];
    alloc.apply_gains(&mut voices);

    // Both slots inactive: all zeros.
    assert!(voices[0].iter().all(|&s| s == 0.0));
    assert!(voices[1].iter().all(|&s| s == 0.0));
}

#[test]
fn test_mute_unmute() {
    let config = VoiceAllocConfig::default()
        .with_max_voices(2)
        .with_ramp_samples(0);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    let s = alloc.allocate_voice(0.5).expect("slot");
    alloc.slots[s].gain = 1.0; // Manually set gain (no ramp).

    alloc.mute_voice(s);
    assert!(alloc.slots[s].muted);
    assert!(alloc.slots[s].active); // Still active, just muted.

    let mut voices = vec![vec![1.0; 4], vec![1.0; 4]];
    alloc.apply_gains(&mut voices);
    assert!(
        voices[s].iter().all(|&v| v == 0.0),
        "muted voice should be zeroed"
    );

    alloc.unmute_voice(s);
    assert!(!alloc.slots[s].muted);
}

#[test]
fn test_reset() {
    let config = VoiceAllocConfig::default().with_max_voices(4);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");

    alloc.allocate_voice(0.5);
    alloc.allocate_voice(0.8);
    assert_eq!(alloc.active_count(), 2);

    alloc.reset();
    assert_eq!(alloc.active_count(), 0);
    assert_eq!(alloc.alloc_counter, 0);
}

#[test]
fn test_nan_samples_zeroed_by_apply_gains() {
    let config = VoiceAllocConfig::default()
        .with_max_voices(1)
        .with_ramp_samples(0);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");
    alloc.allocate_voice(0.5);
    alloc.slots[0].gain = 1.0;

    let mut voices = vec![vec![f32::NAN, f32::INFINITY, 0.5, f32::NEG_INFINITY]];
    alloc.apply_gains(&mut voices);

    for (i, &s) in voices[0].iter().enumerate() {
        assert!(s.is_finite(), "sample {i} should be finite, got {s}");
    }
}

#[test]
fn test_priority_pan_center_for_lead() {
    assert!((priority_pan(0, 4) - 0.5).abs() < f32::EPSILON);
    // Rank 1 goes left, rank 2 goes right.
    assert!(priority_pan(1, 4) < 0.5);
    assert!(priority_pan(2, 4) > 0.5);
}

#[test]
fn test_set_priority() {
    let config = VoiceAllocConfig::default().with_max_voices(2);
    let mut alloc = VoiceAllocator::new(&config).expect("valid config");
    let s = alloc.allocate_voice(0.3).expect("slot");
    assert!((alloc.slots[s].priority - 0.3).abs() < f32::EPSILON);

    alloc.set_priority(s, 0.9);
    assert!((alloc.slots[s].priority - 0.9).abs() < f32::EPSILON);

    // Clamped to [0.0, 1.0].
    alloc.set_priority(s, 2.0);
    assert!((alloc.slots[s].priority - 1.0).abs() < f32::EPSILON);
}
