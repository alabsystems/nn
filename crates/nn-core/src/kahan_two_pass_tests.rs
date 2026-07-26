// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Kahan two-pass mean/variance computation.
//! Part of #2735.

use super::*;

// -- Helper: f64 reference computation (ground truth) --

fn f64_mean_var(data: &[f32]) -> (f64, f64) {
    let n = data.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let mean = data.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let var = data
        .iter()
        .map(|&x| {
            let d = f64::from(x) - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean, var)
}

// -- Welford reference for cross-algorithm tests --

fn welford_mean_var(data: &[f32]) -> (f32, f32) {
    use crate::welford::{welford_update, WelfordState};
    let mut state = WelfordState::ZERO;
    for &x in data {
        state = welford_update(state, x);
    }
    let mean = state.mean;
    let var = if state.n > 0.0 {
        state.m2 / state.n
    } else {
        0.0
    };
    (mean, var)
}

// -- Basic correctness --

#[test]
fn test_kahan_acc_single_value() {
    let acc = KahanAcc::ZERO.add(5.0);
    assert_eq!(acc.sum, 5.0);
}

#[test]
fn test_kahan_acc_two_values() {
    let acc = KahanAcc::ZERO.add(3.0).add(7.0);
    assert_eq!(acc.sum, 10.0);
}

#[test]
fn test_kahan_sum_small() {
    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let sum = kahan_sum(&data);
    assert!((sum - 15.0).abs() < 1e-6);
}

#[test]
fn test_kahan_merge_identity() {
    let acc = KahanAcc::ZERO.add(10.0).add(20.0);
    let merged = kahan_merge(acc, KahanAcc::ZERO);
    assert_eq!(merged.sum, acc.sum);
}

#[test]
fn test_kahan_merge_two_halves() {
    let a = KahanAcc::ZERO.add(1.0).add(2.0);
    let b = KahanAcc::ZERO.add(3.0).add(4.0);
    let merged = kahan_merge(a, b);
    assert!((merged.sum - 10.0).abs() < 1e-5);
}

#[test]
fn test_two_pass_basic() {
    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let (mean, var) = kahan_two_pass_mean_var(&data);
    assert!((mean - 3.0).abs() < 1e-5, "mean={mean}");
    // Population variance of [1,2,3,4,5] = 2.0
    assert!((var - 2.0).abs() < 1e-4, "var={var}");
}

#[test]
fn test_two_pass_empty() {
    let (mean, var) = kahan_two_pass_mean_var(&[]);
    assert_eq!(mean, 0.0);
    assert_eq!(var, 0.0);
}

#[test]
fn test_two_pass_single_element() {
    let (mean, var) = kahan_two_pass_mean_var(&[42.0]);
    assert_eq!(mean, 42.0);
    assert_eq!(var, 0.0);
}

#[test]
fn test_two_pass_constant_input() {
    let data = [7.0_f32; 100];
    let (mean, var) = kahan_two_pass_mean_var(&data);
    assert!((mean - 7.0).abs() < 1e-6, "mean={mean}");
    assert!(var.abs() < 1e-6, "var={var}");
}

#[test]
fn test_two_pass_known_sequence() {
    // [2, 4, 4, 4, 5, 5, 7, 9] — mean=5, pop_var=4.0
    let data = [2.0_f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let (mean, var) = kahan_two_pass_mean_var(&data);
    assert!((mean - 5.0).abs() < 1e-5, "mean={mean}");
    assert!((var - 4.0).abs() < 1e-4, "var={var}");
}

// -- Naive sum comparison (compensation benefit) --

#[test]
fn test_kahan_vs_naive_large_offset() {
    // Large offset + small variations: catastrophic cancellation scenario.
    let base = 1.0e5_f32;
    let data: Vec<f32> = (0..1000).map(|i| base + (i as f32) * 0.001).collect();

    let kahan_result = kahan_sum(&data);
    let naive_result = naive_sum(&data);

    // f64 reference
    let ref_sum: f64 = data.iter().map(|&x| f64::from(x)).sum();

    let kahan_err = (f64::from(kahan_result) - ref_sum).abs();
    let naive_err = (f64::from(naive_result) - ref_sum).abs();

    assert!(
        kahan_err <= naive_err,
        "Kahan err ({kahan_err}) should be <= naive err ({naive_err})"
    );
}

// -- Cross-algorithm equivalence: Welford vs KahanTwoPass vs f64 --

#[test]
fn test_cross_algo_small_input() {
    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let (ref_mean, ref_var) = f64_mean_var(&data);
    let (ktp_mean, ktp_var) = kahan_two_pass_mean_var(&data);
    let (w_mean, w_var) = welford_mean_var(&data);

    let eps = 1e-5;
    assert!(
        (f64::from(ktp_mean) - ref_mean).abs() < eps,
        "KTP mean {ktp_mean} vs ref {ref_mean}"
    );
    assert!(
        (f64::from(ktp_var) - ref_var).abs() < eps,
        "KTP var {ktp_var} vs ref {ref_var}"
    );
    assert!(
        (f64::from(w_mean) - ref_mean).abs() < eps,
        "Welford mean {w_mean} vs ref {ref_mean}"
    );
    assert!(
        (f64::from(w_var) - ref_var).abs() < eps,
        "Welford var {w_var} vs ref {ref_var}"
    );
}

#[test]
fn test_cross_algo_large_offset() {
    // Catastrophic cancellation: values near 1e3 with tiny variations.
    let base = 1.0e3_f32;
    let data: Vec<f32> = (0..256).map(|i| base + (i as f32) * 0.01).collect();

    let (ref_mean, ref_var) = f64_mean_var(&data);
    let (ktp_mean, ktp_var) = kahan_two_pass_mean_var(&data);
    let (w_mean, w_var) = welford_mean_var(&data);

    // Both algorithms should be within tolerance of f64 reference
    let mean_tol = 0.01;
    let var_tol = 0.01;

    assert!(
        (f64::from(ktp_mean) - ref_mean).abs() < mean_tol,
        "KTP mean {ktp_mean} vs ref {ref_mean}"
    );
    assert!(
        (f64::from(ktp_var) - ref_var).abs() < var_tol,
        "KTP var {ktp_var} vs ref {ref_var}"
    );
    assert!(
        (f64::from(w_mean) - ref_mean).abs() < mean_tol,
        "Welford mean {w_mean} vs ref {ref_mean}"
    );
    assert!(
        (f64::from(w_var) - ref_var).abs() < var_tol,
        "Welford var {w_var} vs ref {ref_var}"
    );
}

#[test]
fn test_cross_algo_sinusoidal_kokoro_like() {
    // Simulates Kokoro-like audio feature data: sinusoidal + small noise.
    let data: Vec<f32> = (0..256)
        .map(|i| {
            let t = i as f32 / 256.0;
            (t * std::f32::consts::TAU).sin() * 0.5
        })
        .collect();

    let (ref_mean, ref_var) = f64_mean_var(&data);
    let (ktp_mean, ktp_var) = kahan_two_pass_mean_var(&data);
    let (w_mean, w_var) = welford_mean_var(&data);

    let tol = 1e-5;
    assert!(
        (f64::from(ktp_mean) - ref_mean).abs() < tol,
        "KTP mean {ktp_mean} vs ref {ref_mean}"
    );
    assert!(
        (f64::from(ktp_var) - ref_var).abs() < tol,
        "KTP var {ktp_var} vs ref {ref_var}"
    );
    assert!(
        (f64::from(w_mean) - ref_mean).abs() < tol,
        "Welford mean {w_mean} vs ref {ref_mean}"
    );
    assert!(
        (f64::from(w_var) - ref_var).abs() < tol,
        "Welford var {w_var} vs ref {ref_var}"
    );
}

#[test]
fn test_cross_algo_extreme_offset() {
    // Extreme offset 1e6: where Kahan compensation matters most.
    let base = 1.0e6_f32;
    let data: Vec<f32> = (0..100).map(|i| base + (i as f32) * 0.001).collect();

    let (ref_mean, _ref_var) = f64_mean_var(&data);
    let (ktp_mean, ktp_var) = kahan_two_pass_mean_var(&data);
    let (w_mean, w_var) = welford_mean_var(&data);

    // Both should produce finite results
    assert!(ktp_mean.is_finite(), "KTP mean must be finite");
    assert!(ktp_var.is_finite(), "KTP var must be finite");
    assert!(w_mean.is_finite(), "Welford mean must be finite");
    assert!(w_var.is_finite(), "Welford var must be finite");

    // KTP should be closer to f64 reference than Welford for mean
    // (two-pass Kahan sum is optimal for summation)
    let ktp_mean_err = (f64::from(ktp_mean) - ref_mean).abs();
    let w_mean_err = (f64::from(w_mean) - ref_mean).abs();
    assert!(
        ktp_mean_err <= w_mean_err + 1e-10,
        "KTP mean err ({ktp_mean_err}) should be <= Welford mean err ({w_mean_err})"
    );
}
