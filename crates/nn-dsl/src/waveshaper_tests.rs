// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for tanh waveshaper kernel.
//!
//! Part of #956 D2 (Audio DSP kernel support).

use super::*;

#[test]
fn test_passthrough_at_zero_drive() {
    let y = tanh_waveshaper_scalar(0.5, 0.0).unwrap();
    assert_eq!(y, 0.5);
}

#[test]
fn test_passthrough_at_negative_drive() {
    let y = tanh_waveshaper_scalar(0.75, -1.0).unwrap();
    assert_eq!(y, 0.75);
}

#[test]
fn test_output_bounded_positive_drive() {
    // With drive > 0 and |x| ≤ 1, output must be in [-1, 1].
    // tanh is monotone → |tanh(d*x)| ≤ tanh(d) when |x| ≤ 1, so ratio ≤ 1.
    // For |x| > 1, output can exceed ±1 (approaches 1/tanh(drive) at x→∞).
    for drive in [0.1, 1.0, 5.0, 10.0, 50.0] {
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let y = tanh_waveshaper_scalar(x, drive).unwrap();
            assert!(
                (-1.001..=1.001).contains(&y),
                "drive={drive}, x={x}: output {y} out of [-1, 1]"
            );
        }
    }
}

#[test]
fn test_output_exceeds_one_for_large_input() {
    // For |x| > 1 and small drive, output can exceed ±1
    let y = tanh_waveshaper_scalar(10.0, 0.1).unwrap();
    assert!(y > 1.0, "large x + small drive → output > 1, got {y}");
}

#[test]
fn test_zero_input_zero_output() {
    let y = tanh_waveshaper_scalar(0.0, 5.0).unwrap();
    assert!(y.abs() < 1e-7, "tanh(0)/tanh(d) = 0, got {y}");
}

#[test]
fn test_monotone_increasing() {
    let drive = 2.0;
    let y_neg = tanh_waveshaper_scalar(-1.0, drive).unwrap();
    let y_zero = tanh_waveshaper_scalar(0.0, drive).unwrap();
    let y_pos = tanh_waveshaper_scalar(1.0, drive).unwrap();
    assert!(y_neg < y_zero, "monotone: y(-1) < y(0)");
    assert!(y_zero < y_pos, "monotone: y(0) < y(1)");
}

#[test]
fn test_symmetry() {
    let drive = 3.0;
    let y_pos = tanh_waveshaper_scalar(0.7, drive).unwrap();
    let y_neg = tanh_waveshaper_scalar(-0.7, drive).unwrap();
    assert!((y_pos + y_neg).abs() < 1e-6, "odd symmetry: f(-x) = -f(x)");
}

#[test]
fn test_unity_at_x1_drive_inf_limit() {
    // As drive → ∞, tanh(drive*x)/tanh(drive) → sign(x) for |x|>0
    // At finite high drive, tanh(50*1)/tanh(50) ≈ 1.0/1.0 = 1.0
    let y = tanh_waveshaper_scalar(1.0, 50.0).unwrap();
    assert!(
        (y - 1.0).abs() < 1e-5,
        "high drive at x=1 should be ~1.0, got {y}"
    );
}

#[test]
fn test_reject_nan_input() {
    assert!(tanh_waveshaper_scalar(f32::NAN, 1.0).is_err());
}

#[test]
fn test_reject_inf_drive() {
    assert!(tanh_waveshaper_scalar(0.5, f32::INFINITY).is_err());
}

// --- Bounds tests ---

#[test]
fn test_bounds_positive_drive_unit_input() {
    // |x| ≤ 1 with drive > 0 → bounds are [-1, 1]
    let (lo, hi) = tanh_waveshaper_scalar_bounds(-1.0, 1.0, 1.0, 10.0).unwrap();
    assert_eq!(lo, -1.0);
    assert_eq!(hi, 1.0);
}

#[test]
fn test_bounds_positive_drive_large_input() {
    // |x| > 1 with drive > 0 → bounds widen to cover passthrough at low drive
    let (lo, hi) = tanh_waveshaper_scalar_bounds(-5.0, 5.0, 1.0, 10.0).unwrap();
    assert!(lo <= -5.0, "lower bound must cover x_lo=-5, got {lo}");
    assert!(hi >= 5.0, "upper bound must cover x_hi=5, got {hi}");
}

#[test]
fn test_bounds_mixed_drive() {
    let (lo, hi) = tanh_waveshaper_scalar_bounds(-2.0, 2.0, -1.0, 5.0).unwrap();
    assert!(lo <= -1.0);
    assert!(hi >= 1.0);
    // Conservative: includes both passthrough and waveshaper ranges
    assert!(lo <= -2.0);
    assert!(hi >= 2.0);
}
