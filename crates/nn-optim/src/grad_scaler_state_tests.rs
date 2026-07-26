// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GradScaler state persistence and checkpoint tests.
//!
//! Extracted from `grad_scaler_tests.rs` to keep each file under 500 lines.

use super::*;

#[test]
fn test_save_state_round_trip() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 512.0,
        growth_interval: 10,
        ..Default::default()
    })
    .unwrap();

    // Advance some state
    for _ in 0..5 {
        scaler.found_inf = false;
        scaler.update();
    }

    let state = scaler.save_state();
    assert!((state.scale - 512.0).abs() < f64::EPSILON);
    assert_eq!(state.growth_tracker, 5);

    // Restore into a fresh scaler with the same config
    let mut scaler2 = GradScaler::new(GradScalerConfig {
        init_scale: 1.0, // different initial
        growth_interval: 10,
        ..Default::default()
    })
    .unwrap();
    scaler2.load_state(&state).unwrap();
    assert!((scaler2.scale_factor() - 512.0).abs() < f64::EPSILON);
}

#[test]
fn test_save_state_after_backoff() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        backoff_factor: 0.5,
        ..Default::default()
    })
    .unwrap();

    // Trigger backoff
    scaler.found_inf = true;
    scaler.update();

    let state = scaler.save_state();
    assert!((state.scale - 512.0).abs() < f64::EPSILON);
    assert_eq!(state.growth_tracker, 0);
}

#[test]
fn test_load_state_clamps_to_bounds() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 100.0,
        min_scale: 10.0,
        max_scale: 1000.0,
        ..Default::default()
    })
    .unwrap();

    // Scale above max — should clamp to max
    let state = crate::checkpoint::GradScalerState {
        scale: 5000.0,
        growth_tracker: 0,
    };
    scaler.load_state(&state).unwrap();
    assert!((scaler.scale_factor() - 1000.0).abs() < f64::EPSILON);

    // Scale below min — should clamp to min
    let state2 = crate::checkpoint::GradScalerState {
        scale: 1.0,
        growth_tracker: 0,
    };
    scaler.load_state(&state2).unwrap();
    assert!((scaler.scale_factor() - 10.0).abs() < f64::EPSILON);
}

#[test]
fn test_load_state_caps_growth_tracker() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 100.0,
        growth_interval: 5,
        ..Default::default()
    })
    .unwrap();

    let state = crate::checkpoint::GradScalerState {
        scale: 100.0,
        growth_tracker: 999,
    };
    scaler.load_state(&state).unwrap();

    // growth_tracker capped to growth_interval - 1 = 4 (prevents immediate growth).
    // One clean step should NOT trigger growth (tracker goes 4 → 5, which equals
    // growth_interval, so it DOES trigger). Verify at least one step is needed.
    assert!(
        (scaler.scale_factor() - 100.0).abs() < f64::EPSILON,
        "scale should not have grown immediately after load"
    );

    // One clean step: tracker 4 → 5 >= 5, triggers growth
    scaler.found_inf = false;
    scaler.update();
    assert!(
        (scaler.scale_factor() - 200.0).abs() < f64::EPSILON,
        "scale should grow after one clean step, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_load_state_exact_interval_requires_one_step() {
    // Verify that loading with growth_tracker == growth_interval still
    // requires at least one clean step before growth (saturating_sub guard).
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 100.0,
        growth_interval: 3,
        ..Default::default()
    })
    .unwrap();

    let state = crate::checkpoint::GradScalerState {
        scale: 100.0,
        growth_tracker: 3, // exactly growth_interval
    };
    scaler.load_state(&state).unwrap();

    // Should be capped to 2 (growth_interval - 1), not 3
    // So we need 1 clean step to reach 3 and trigger growth
    scaler.found_inf = false;
    scaler.update(); // tracker 2 → 3 >= 3, triggers growth
    assert!(
        (scaler.scale_factor() - 200.0).abs() < f64::EPSILON,
        "expected 200.0, got {}",
        scaler.scale_factor()
    );
}

#[test]
fn test_load_state_rejects_nan_scale() {
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let state = crate::checkpoint::GradScalerState {
        scale: f64::NAN,
        growth_tracker: 0,
    };
    let err = scaler.load_state(&state).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("finite"), "error should mention finite: {msg}");
}

#[test]
fn test_load_state_rejects_inf_scale() {
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let state = crate::checkpoint::GradScalerState {
        scale: f64::INFINITY,
        growth_tracker: 0,
    };
    let err = scaler.load_state(&state).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("finite"), "error should mention finite: {msg}");
}

#[test]
fn test_load_state_rejects_zero_scale() {
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let state = crate::checkpoint::GradScalerState {
        scale: 0.0,
        growth_tracker: 0,
    };
    let err = scaler.load_state(&state).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("positive"),
        "error should mention positive: {msg}"
    );
}

#[test]
fn test_load_state_rejects_negative_scale() {
    let mut scaler = GradScaler::new(GradScalerConfig::default()).unwrap();
    let state = crate::checkpoint::GradScalerState {
        scale: -1.0,
        growth_tracker: 0,
    };
    let err = scaler.load_state(&state).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("positive"),
        "error should mention positive: {msg}"
    );
}

#[test]
fn test_invalid_config_growth_interval_zero() {
    let err = GradScaler::new(GradScalerConfig {
        growth_interval: 0,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("growth_interval"),
        "error should mention growth_interval: {msg}"
    );
}

#[test]
fn test_invalid_config_init_scale_below_min() {
    let err = GradScaler::new(GradScalerConfig {
        init_scale: 0.5,
        min_scale: 1.0,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("init_scale"),
        "error should mention init_scale: {msg}"
    );
}

#[test]
fn test_invalid_config_init_scale_above_max() {
    let err = GradScaler::new(GradScalerConfig {
        init_scale: 1e20,
        max_scale: 1e10,
        ..Default::default()
    })
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("init_scale"),
        "error should mention init_scale: {msg}"
    );
}
