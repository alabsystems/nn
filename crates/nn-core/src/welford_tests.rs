// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_welford_single_sample() {
    let s = welford_update(WelfordState::ZERO, 5.0);
    assert_eq!(s.n, 1.0);
    assert_eq!(s.mean, 5.0);
    assert_eq!(s.m2, 0.0);
    assert_eq!(s.variance(), 0.0);
}

#[test]
fn test_welford_two_samples() {
    let s = welford_update(WelfordState::ZERO, 3.0);
    let s = welford_update(s, 7.0);
    assert_eq!(s.n, 2.0);
    assert_eq!(s.mean, 5.0);
    // m2 = (3-5)^2 + (7-5)^2 = 4 + 4 = 8
    assert!((s.m2 - 8.0).abs() < 1e-5);
    // variance = m2 / n = 4.0
    assert!((s.variance() - 4.0).abs() < 1e-5);
}

#[test]
fn test_welford_merge_identity() {
    // Merging with zero state is identity
    let s = welford_update(WelfordState::ZERO, 10.0);
    let merged = welford_merge(s, WelfordState::ZERO);
    assert_eq!(merged.n, s.n);
    assert_eq!(merged.mean, s.mean);
    assert_eq!(merged.m2, s.m2);
}

#[test]
fn test_welford_merge_two_halves() {
    // Split [1, 2, 3, 4] into [1, 2] and [3, 4], merge
    let mut a = WelfordState::ZERO;
    a = welford_update(a, 1.0);
    a = welford_update(a, 2.0);

    let mut b = WelfordState::ZERO;
    b = welford_update(b, 3.0);
    b = welford_update(b, 4.0);

    let merged = welford_merge(a, b);
    assert_eq!(merged.n, 4.0);
    assert!((merged.mean - 2.5).abs() < 1e-5);
    // Variance of [1,2,3,4] = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4
    // = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 5.0 / 4 = 1.25
    // m2 = sum of squared deviations = 5.0
    assert!((merged.m2 - 5.0).abs() < 1e-4);
}

#[test]
fn test_welford_variance_known_sequence() {
    // [2, 4, 4, 4, 5, 5, 7, 9] — mean=5, var=4
    let samples = [2.0_f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let mut s = WelfordState::ZERO;
    for &x in &samples {
        s = welford_update(s, x);
    }
    assert_eq!(s.n, 8.0);
    assert!((s.mean - 5.0).abs() < 1e-5);
    // m2 = sum of squared deviations = 32.0
    assert!((s.m2 - 32.0).abs() < 1e-3);
    // Population variance = m2/n = 32/8 = 4.0
    assert!((s.variance() - 4.0).abs() < 1e-4);
}

#[test]
fn test_compensation_vs_uncompensated_large_offset() {
    // Large offset + small variations: catastrophic cancellation scenario.
    // Values near 1e5 with tiny differences.
    let base = 1.0e5_f32;
    let samples: Vec<f32> = (0..100).map(|i| base + (i as f32) * 0.001).collect();

    let mut comp = WelfordState::ZERO;
    let mut uncomp = WelfordState::ZERO;
    for &x in &samples {
        comp = welford_update(comp, x);
        uncomp = welford_update_uncompensated(uncomp, x);
    }

    // f64 reference
    let f64_samples: Vec<f64> = samples.iter().map(|&x| f64::from(x)).collect();
    let f64_mean = f64_samples.iter().sum::<f64>() / f64_samples.len() as f64;
    let f64_m2: f64 = f64_samples.iter().map(|&x| (x - f64_mean).powi(2)).sum();

    let comp_err = (f64::from(comp.m2) - f64_m2).abs();
    let uncomp_err = (f64::from(uncomp.m2) - f64_m2).abs();

    // Compensated should be closer to the f64 reference
    assert!(
        comp_err <= uncomp_err,
        "compensated error ({comp_err}) should be <= uncompensated ({uncomp_err})"
    );
}
