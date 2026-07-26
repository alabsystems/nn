// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for side-chain ducker kernel.
//!
//! Part of #956 D2 (Audio DSP kernel support).

use super::*;

fn test_config() -> DuckerCoeffs {
    DuckerCoeffs {
        attack_coeff: 0.1,
        release_coeff: 0.01,
        threshold: 0.5,
        ratio: 0.2,
    }
}

#[test]
fn test_no_ducking_below_threshold() {
    let state = DuckerState::new();
    let config = test_config();
    // Sidechain below threshold → gain = 1.0
    let out = ducker_process_sample_scalar(0.8, 0.1, &state, &config).unwrap();
    assert_eq!(out.gain, 1.0, "no ducking below threshold");
    assert!((out.y - 0.8).abs() < 1e-6, "passthrough below threshold");
}

#[test]
fn test_ducking_above_threshold() {
    let state = DuckerState {
        envelope: 1.0, // already above threshold=0.5
        gain: 1.0,
    };
    let config = test_config();
    let out = ducker_process_sample_scalar(1.0, 2.0, &state, &config).unwrap();
    assert!(out.gain < 1.0, "gain should be reduced above threshold");
    assert!(out.y.abs() <= 1.0, "output should not exceed input");
}

#[test]
fn test_gain_at_threshold_boundary() {
    let state = DuckerState {
        envelope: 0.5, // exactly at threshold
        gain: 1.0,
    };
    let config = test_config();
    // Feed sidechain = 0.5 (at threshold) with attack_coeff envelope tracking
    let out = ducker_process_sample_scalar(1.0, 0.5, &state, &config).unwrap();
    // Envelope should be near 0.5, gain should be near 1.0
    assert!(
        out.gain >= 0.99,
        "gain at threshold should be near 1.0, got {}",
        out.gain
    );
}

#[test]
fn test_envelope_tracks_sidechain() {
    let state = DuckerState::new(); // envelope = 0
    let config = test_config();
    // Large sidechain should increase envelope
    let out1 = ducker_process_sample_scalar(0.0, 5.0, &state, &config).unwrap();
    assert!(
        out1.envelope > 0.0,
        "envelope should increase with sidechain"
    );
    // Feed more
    let state2 = DuckerState {
        envelope: out1.envelope,
        gain: out1.gain,
    };
    let out2 = ducker_process_sample_scalar(0.0, 5.0, &state2, &config).unwrap();
    assert!(
        out2.envelope > out1.envelope,
        "envelope should keep increasing"
    );
}

#[test]
fn test_never_amplifies() {
    let config = test_config();
    for &x in &[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
        for &sc in &[0.0, 0.5, 1.0, 5.0] {
            for &env in &[0.0, 0.5, 1.0, 5.0] {
                let state = DuckerState {
                    envelope: env,
                    gain: 1.0,
                };
                let out = ducker_process_sample_scalar(x, sc, &state, &config).unwrap();
                assert!(
                    out.y.abs() <= x.abs() + 1e-6,
                    "x={x}, sc={sc}, env={env}: |y|={} > |x|={}",
                    out.y.abs(),
                    x.abs()
                );
            }
        }
    }
}

#[test]
fn test_envelope_non_negative() {
    let state = DuckerState::new();
    let config = test_config();
    let out = ducker_process_sample_scalar(0.0, -3.0, &state, &config).unwrap();
    assert!(out.envelope >= 0.0, "envelope must be non-negative");
}

#[test]
fn test_gain_in_range() {
    let state = DuckerState {
        envelope: 10.0,
        gain: 1.0,
    };
    let config = test_config();
    let out = ducker_process_sample_scalar(1.0, 10.0, &state, &config).unwrap();
    assert!(
        out.gain >= 0.0 && out.gain <= 1.0,
        "gain out of [0,1]: {}",
        out.gain
    );
}

// --- Config validation tests ---

#[test]
fn test_validate_config_valid() {
    assert!(validate_ducker_config(&test_config()).is_ok());
}

#[test]
fn test_validate_config_bad_attack() {
    let mut config = test_config();
    config.attack_coeff = 0.0;
    assert!(validate_ducker_config(&config).is_err());
}

#[test]
fn test_validate_config_bad_threshold() {
    let mut config = test_config();
    config.threshold = -1.0;
    assert!(validate_ducker_config(&config).is_err());
}

#[test]
fn test_validate_config_bad_ratio() {
    let mut config = test_config();
    config.ratio = 1.5;
    assert!(validate_ducker_config(&config).is_err());
}

// --- Error handling ---

#[test]
fn test_reject_nan_sidechain() {
    let state = DuckerState::new();
    let config = test_config();
    assert!(ducker_process_sample_scalar(0.5, f32::NAN, &state, &config).is_err());
}

#[test]
fn test_reject_inf_input() {
    let state = DuckerState::new();
    let config = test_config();
    assert!(ducker_process_sample_scalar(f32::INFINITY, 0.5, &state, &config).is_err());
}

#[test]
fn test_default_state() {
    let state = DuckerState::default();
    assert_eq!(state.envelope, 0.0);
    assert_eq!(state.gain, 1.0);
}
