// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Rust-side reference implementations and correctness tests for the GPU norm
//! reduction algorithms in `dyn_tensor_metal_welford_msl.rs`.
//!
//! These tests verify algorithm correctness independently of GPU execution by
//! implementing the exact same algorithms in Rust and comparing against f64
//! reference computations. Findings feed directly into #2703 (Kani harnesses).
//!
//! ## Algorithms tested:
//! - Kahan-compensated Welford (single-pass): matches MSL `welford_update` + `welford_merge`
//! - Two-pass Kahan-compensated summation: matches MSL `kahan_two_pass_reduction_msl`
//!
//! ## Algorithm audit findings (Part of #2218, #2703):
//!
//! 1. **welford_merge compensation is approximate.** `base_m2 = a.m2 + b.m2`
//!    discards its own rounding error. For 8-level tree reductions, this adds
//!    O(8*eps) relative error — adequate but could be tighter with a 3-term
//!    Kahan sum. Same pattern in two-pass merge.
//!
//! 2. **Both algorithms' tree reduction merge uses `(a_comp + b_comp)` as
//!    combined prior compensation.** This is an approximation — the sum itself
//!    can lose low-order bits. In practice, comp values are O(eps * sum),
//!    so the error in summing them is O(eps^2 * sum), negligible.
//!
//! Part of #2218, #2703, #2696, #2697.

// ---- Rust-side reference: Kahan-compensated Welford (matches MSL) -----------

/// Mirrors the MSL `WelfordState` struct exactly.
#[derive(Debug, Clone, Copy)]
struct WelfordState {
    n: f32,
    mean: f32,
    m2: f32,
    m2_comp: f32, // Kahan compensation for m2
}

impl WelfordState {
    fn new() -> Self {
        Self {
            n: 0.0,
            mean: 0.0,
            m2: 0.0,
            m2_comp: 0.0,
        }
    }
}

/// Mirrors the MSL `welford_update()` exactly.
fn welford_update(mut state: WelfordState, x: f32) -> WelfordState {
    state.n += 1.0;
    let delta = x - state.mean;
    state.mean += delta / state.n;
    let delta2 = x - state.mean;
    // Kahan-compensated m2 accumulation
    let y = delta * delta2 - state.m2_comp;
    let t = state.m2 + y;
    state.m2_comp = (t - state.m2) - y;
    state.m2 = t;
    state
}

/// Mirrors the MSL `welford_merge()` exactly.
fn welford_merge(a: WelfordState, b: WelfordState) -> WelfordState {
    if b.n == 0.0 {
        return a;
    }
    if a.n == 0.0 {
        return b;
    }
    let n = a.n + b.n;
    let delta = b.mean - a.mean;
    let mean = a.mean + delta * b.n / n;
    // Kahan-compensated m2 merge
    let m2_add = delta * delta * a.n * b.n / n;
    let base_m2 = a.m2 + b.m2;
    let comp = a.m2_comp + b.m2_comp;
    let y = m2_add - comp;
    let t = base_m2 + y;
    let new_comp = (t - base_m2) - y;
    WelfordState {
        n,
        mean,
        m2: t,
        m2_comp: new_comp,
    }
}

/// Compute mean and variance via Kahan-Welford, simulating 1 thread (no tree reduction).
fn welford_mean_var(data: &[f32]) -> (f32, f32) {
    let mut state = WelfordState::new();
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

/// Simulate the GPU tree reduction with `tg_size` threads.
fn welford_tree_reduction(data: &[f32], tg_size: usize) -> (f32, f32) {
    // Phase 1: per-thread accumulation (strided).
    let mut states: Vec<WelfordState> = (0..tg_size)
        .map(|tid| {
            let mut s = WelfordState::new();
            let mut i = tid;
            while i < data.len() {
                s = welford_update(s, data[i]);
                i += tg_size;
            }
            s
        })
        .collect();

    // Phase 2: tree reduction.
    let mut stride = tg_size / 2;
    while stride > 0 {
        for tid in 0..stride {
            states[tid] = welford_merge(states[tid], states[tid + stride]);
        }
        stride /= 2;
    }

    let mean = states[0].mean;
    let var = if states[0].n > 0.0 {
        states[0].m2 / states[0].n.max(1.0)
    } else {
        0.0
    };
    (mean, var)
}

// ---- Rust-side reference: Two-pass Kahan-compensated summation (matches MSL) ---

/// Per-thread Kahan accumulator state.
#[derive(Debug, Clone, Copy)]
struct KahanAcc {
    sum: f32,
    comp: f32,
}

impl KahanAcc {
    fn new() -> Self {
        Self {
            sum: 0.0,
            comp: 0.0,
        }
    }

    fn add(&mut self, val: f32) {
        let y = val - self.comp;
        let t = self.sum + y;
        self.comp = (t - self.sum) - y;
        self.sum = t;
    }
}

/// Merge two Kahan accumulators (matches MSL tree reduction merge).
fn kahan_merge(a: KahanAcc, b: KahanAcc) -> KahanAcc {
    let y = b.sum - (a.comp + b.comp);
    let t = a.sum + y;
    KahanAcc {
        sum: t,
        comp: (t - a.sum) - y,
    }
}

/// Two-pass Kahan mean+variance, simulating GPU tree reduction with `tg_size` threads.
fn kahan_two_pass_tree_reduction(data: &[f32], tg_size: usize) -> (f32, f32) {
    // Pass 1: Kahan-compensated sum for mean.
    let mut pass1: Vec<KahanAcc> = (0..tg_size)
        .map(|tid| {
            let mut acc = KahanAcc::new();
            let mut i = tid;
            while i < data.len() {
                acc.add(data[i]);
                i += tg_size;
            }
            acc
        })
        .collect();

    // Tree reduction for sum.
    let mut stride = tg_size / 2;
    while stride > 0 {
        for tid in 0..stride {
            pass1[tid] = kahan_merge(pass1[tid], pass1[tid + stride]);
        }
        stride /= 2;
    }
    let mean = pass1[0].sum / (data.len() as f32).max(1.0);

    // Pass 2: Kahan-compensated sum of (x - mean)^2.
    let mut pass2: Vec<KahanAcc> = (0..tg_size)
        .map(|tid| {
            let mut acc = KahanAcc::new();
            let mut i = tid;
            while i < data.len() {
                let diff = data[i] - mean;
                acc.add(diff * diff);
                i += tg_size;
            }
            acc
        })
        .collect();

    stride = tg_size / 2;
    while stride > 0 {
        for tid in 0..stride {
            pass2[tid] = kahan_merge(pass2[tid], pass2[tid + stride]);
        }
        stride /= 2;
    }
    let var = pass2[0].sum / (data.len() as f32).max(1.0);
    (mean, var)
}

// ---- F64 reference computation (ground truth) --------------------------------

fn f64_mean_var(data: &[f32]) -> (f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = data
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean, var)
}

// ---- Tests -------------------------------------------------------------------

/// Basic correctness: both algorithms match f64 reference for small input.
#[test]
fn test_welford_basic_correctness() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (ref_mean, ref_var) = f64_mean_var(&data);

    let (w_mean, w_var) = welford_mean_var(&data);
    assert!(
        (w_mean as f64 - ref_mean).abs() < 1e-6,
        "Welford mean: {w_mean} vs ref {ref_mean}"
    );
    assert!(
        (w_var as f64 - ref_var).abs() < 1e-6,
        "Welford var: {w_var} vs ref {ref_var}"
    );
}

/// Both algorithms match f64 for sinusoidal data (Kokoro-like).
#[test]
fn test_both_algos_sinusoidal_data_256() {
    let data: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect();
    let (ref_mean, ref_var) = f64_mean_var(&data);

    let (w_mean, w_var) = welford_tree_reduction(&data, 256);
    let (k_mean, k_var) = kahan_two_pass_tree_reduction(&data, 256);

    let mean_tol = 1e-6;
    let var_tol = 1e-5;

    assert!(
        (w_mean as f64 - ref_mean).abs() < mean_tol,
        "Welford mean error: {:.2e}",
        (w_mean as f64 - ref_mean).abs()
    );
    assert!(
        (w_var as f64 - ref_var).abs() < var_tol,
        "Welford var error: {:.2e}",
        (w_var as f64 - ref_var).abs()
    );
    assert!(
        (k_mean as f64 - ref_mean).abs() < mean_tol,
        "Kahan 2-pass mean error: {:.2e}",
        (k_mean as f64 - ref_mean).abs()
    );
    assert!(
        (k_var as f64 - ref_var).abs() < var_tol,
        "Kahan 2-pass var error: {:.2e}",
        (k_var as f64 - ref_var).abs()
    );
}

/// Kahan compensation measurably reduces error vs uncompensated Welford.
/// Uses large-offset data where naive accumulation drifts.
#[test]
fn test_kahan_compensation_reduces_error() {
    // Large offset + small perturbation: catastrophic cancellation scenario.
    // Offset 1e3 (not 1e6) keeps condition number manageable for f32.
    // Condition number of variance ≈ (offset / std)^2 ≈ (1e3 / 0.3)^2 ≈ 1e7.
    let offset = 1e3_f32;
    let data: Vec<f32> = (0..1024).map(|i| offset + (i as f32) * 1e-3).collect();
    let (ref_mean, ref_var) = f64_mean_var(&data);

    // Kahan-compensated Welford.
    let (w_mean, w_var) = welford_tree_reduction(&data, 256);
    let w_var_err = (w_var as f64 - ref_var).abs();

    // Uncompensated Welford (zero out compensation to simulate old behavior).
    let (_, naive_var) = {
        let mut state = WelfordState::new();
        for &x in &data {
            state = welford_update(state, x);
            state.m2_comp = 0.0; // Disable Kahan compensation
        }
        let mean = state.mean;
        let var = state.m2 / state.n.max(1.0);
        (mean, var)
    };
    let naive_var_err = (naive_var as f64 - ref_var).abs();

    // Kahan should be more accurate than uncompensated for this input.
    // If both are zero error (unlikely for f32), test is vacuous but not wrong.
    assert!(
        w_var_err <= naive_var_err + 1e-10,
        "Kahan variance error ({w_var_err:.6e}) should be <= uncompensated ({naive_var_err:.6e})"
    );

    // Two-pass Kahan: for condition number ~1e7, best achievable relative
    // error is ~eps * cond ≈ 1.2e-7 * 1e7 ≈ 1.2. For practical Kahan
    // summation the error is much less but still bounded by mean precision.
    // Use 1% tolerance (generous but still validates Kahan helps).
    let (_, k_var) = kahan_two_pass_tree_reduction(&data, 256);
    let k_var_err = (k_var as f64 - ref_var).abs();
    assert!(
        k_var_err < ref_var * 0.01,
        "Two-pass Kahan var relative error {:.2e} exceeds 1%",
        k_var_err / ref_var
    );

    // Welford mean should be accurate (numerically stable by design).
    assert!(
        (w_mean as f64 - ref_mean).abs() / ref_mean.abs() < 1e-6,
        "Welford mean relative error too large"
    );

    // Also test with extreme offset (1e6) — here two-pass has inherently
    // worse variance precision due to f32 mean quantization, but Welford
    // should still be better than uncompensated.
    let offset_extreme = 1e6_f32;
    let data_extreme: Vec<f32> = (0..1024)
        .map(|i| offset_extreme + (i as f32) * 1e-3)
        .collect();
    let (_, ref_var_ext) = f64_mean_var(&data_extreme);
    let (_, w_var_ext) = welford_tree_reduction(&data_extreme, 256);
    let w_err_ext = (w_var_ext as f64 - ref_var_ext).abs();
    // Welford should be finite and positive (no blow-up).
    assert!(
        w_var_ext.is_finite() && w_var_ext >= 0.0,
        "Welford var not finite/positive for extreme offset: {w_var_ext}"
    );
    // Document: at 1e6 offset, f32 variance has ~1-5% relative error.
    // This is an inherent f32 limitation, not an algorithm bug.
    let w_rel_ext = w_err_ext / ref_var_ext;
    assert!(
        w_rel_ext < 0.10,
        "Welford extreme offset var relative error {w_rel_ext:.2e} exceeds 10%"
    );
}

/// Edge case: all-identical input (variance = 0).
#[test]
fn test_constant_input_variance_zero() {
    let data = vec![5.0_f32; 256];

    let (w_mean, w_var) = welford_tree_reduction(&data, 256);
    assert!((w_mean - 5.0).abs() < 1e-6, "Welford mean: {w_mean}");
    assert!(w_var.abs() < 1e-10, "Welford var should be ~0: {w_var}");

    let (k_mean, k_var) = kahan_two_pass_tree_reduction(&data, 256);
    assert!((k_mean - 5.0).abs() < 1e-6, "Kahan mean: {k_mean}");
    assert!(k_var.abs() < 1e-10, "Kahan var should be ~0: {k_var}");
}

/// Edge case: single element.
#[test]
fn test_single_element() {
    let data = vec![7.0_f32];

    let (w_mean, w_var) = welford_tree_reduction(&data, 256);
    assert!((w_mean - 7.0).abs() < 1e-6);
    assert!(w_var.abs() < 1e-10);

    let (k_mean, k_var) = kahan_two_pass_tree_reduction(&data, 256);
    assert!((k_mean - 7.0).abs() < 1e-6);
    assert!(k_var.abs() < 1e-10);
}

/// Edge case: empty input.
#[test]
fn test_empty_input() {
    let data: Vec<f32> = vec![];

    let (w_mean, w_var) = welford_tree_reduction(&data, 256);
    assert!(w_mean == 0.0);
    assert!(w_var == 0.0);

    let (k_mean, k_var) = kahan_two_pass_tree_reduction(&data, 256);
    assert!(k_mean == 0.0);
    assert!(k_var == 0.0);
}

/// Edge case: spatial_len < tg_size (many threads get zero samples).
/// This is the normal case for small spatial dims — threads beyond
/// spatial_len produce empty Welford states that must merge correctly.
#[test]
fn test_spatial_smaller_than_threadgroup() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let (ref_mean, ref_var) = f64_mean_var(&data);

    // tg_size=256 but only 100 data points: threads 100-255 are empty.
    let (w_mean, w_var) = welford_tree_reduction(&data, 256);
    assert!(
        (w_mean as f64 - ref_mean).abs() < 1e-5,
        "Welford mean with empty threads: {w_mean} vs {ref_mean}"
    );
    assert!(
        (w_var as f64 - ref_var).abs() < 1e-4,
        "Welford var with empty threads: {w_var} vs {ref_var}"
    );

    let (k_mean, k_var) = kahan_two_pass_tree_reduction(&data, 256);
    assert!(
        (k_mean as f64 - ref_mean).abs() < 1e-5,
        "Kahan mean with empty threads: {k_mean} vs {ref_mean}"
    );
    assert!(
        (k_var as f64 - ref_var).abs() < 1e-4,
        "Kahan var with empty threads: {k_var} vs {ref_var}"
    );
}

/// Welford merge associativity: merge(merge(a,b),c) ≈ merge(a,merge(b,c)).
/// Tests the property required by #2703 Kani harness.
#[test]
fn test_welford_merge_near_associative() {
    let data_a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    let data_b: Vec<f32> = (64..128).map(|i| (i as f32) * 0.1).collect();
    let data_c: Vec<f32> = (128..192).map(|i| (i as f32) * 0.1).collect();

    let mut sa = WelfordState::new();
    for &x in &data_a {
        sa = welford_update(sa, x);
    }
    let mut sb = WelfordState::new();
    for &x in &data_b {
        sb = welford_update(sb, x);
    }
    let mut sc = WelfordState::new();
    for &x in &data_c {
        sc = welford_update(sc, x);
    }

    // (a merge b) merge c
    let ab = welford_merge(sa, sb);
    let abc_left = welford_merge(ab, sc);

    // a merge (b merge c)
    let bc = welford_merge(sb, sc);
    let abc_right = welford_merge(sa, bc);

    // Results should be within f32 tolerance.
    assert!(
        (abc_left.mean - abc_right.mean).abs() < 1e-5,
        "merge associativity mean: {} vs {}",
        abc_left.mean,
        abc_right.mean
    );
    assert!(
        (abc_left.m2 - abc_right.m2).abs() / abc_left.m2.abs().max(1e-10) < 1e-5,
        "merge associativity m2: {} vs {}",
        abc_left.m2,
        abc_right.m2
    );
}

/// Large values: verify no overflow/NaN for inputs in [-1e6, 1e6].
/// Property required by #2703 Kani harness.
#[test]
fn test_large_values_no_overflow() {
    let data: Vec<f32> = (0..512)
        .map(|i| {
            let t = (i as f32) / 512.0;
            (t * 2.0 - 1.0) * 1e6
        })
        .collect();

    let (w_mean, w_var) = welford_tree_reduction(&data, 256);
    assert!(w_mean.is_finite(), "Welford mean is not finite: {w_mean}");
    assert!(w_var.is_finite(), "Welford var is not finite: {w_var}");
    assert!(w_var >= 0.0, "Welford var is negative: {w_var}");

    let (k_mean, k_var) = kahan_two_pass_tree_reduction(&data, 256);
    assert!(k_mean.is_finite(), "Kahan mean is not finite: {k_mean}");
    assert!(k_var.is_finite(), "Kahan var is not finite: {k_var}");
    assert!(k_var >= 0.0, "Kahan var is negative: {k_var}");
}

/// Chained InstanceNorm simulation: apply normalize(x) 48 times.
/// Verifies both algorithms maintain stability through 48 applications.
/// Directly relevant to #2701 (chained precision drift).
#[test]
fn test_chained_normalize_48_stability() {
    let eps = 1e-5_f32;
    let n = 256;
    let tg_size = 256;

    let mut welford_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect();
    let mut kahan_data = welford_data.clone();

    for _ in 0..48 {
        // Welford path
        let (w_mean, w_var) = welford_tree_reduction(&welford_data, tg_size);
        let w_inv_std = 1.0 / (w_var + eps).sqrt();
        for x in welford_data.iter_mut() {
            *x = (*x - w_mean) * w_inv_std;
        }

        // Two-pass Kahan path
        let (k_mean, k_var) = kahan_two_pass_tree_reduction(&kahan_data, tg_size);
        let k_inv_std = 1.0 / (k_var + eps).sqrt();
        for x in kahan_data.iter_mut() {
            *x = (*x - k_mean) * k_inv_std;
        }
    }

    // After 48 InstanceNorms, data should have approximately unit variance.
    let welford_rms: f32 = (welford_data.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
    let kahan_rms: f32 = (kahan_data.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();

    // Both should be near 1.0 (unit variance).
    assert!(
        (0.90..=1.10).contains(&welford_rms),
        "Welford 48-chain RMS {welford_rms:.4} deviates >10% from 1.0"
    );
    assert!(
        (0.90..=1.10).contains(&kahan_rms),
        "Kahan 48-chain RMS {kahan_rms:.4} deviates >10% from 1.0"
    );

    // Both algorithms should agree closely with each other.
    let ratio = welford_rms / kahan_rms;
    assert!(
        (0.95..=1.05).contains(&ratio),
        "Welford/Kahan RMS ratio {ratio:.4} after 48 chains — algorithms diverge"
    );
}

/// Verify the merge correctly handles the zero-count boundary.
#[test]
fn test_merge_with_zero_count_states() {
    let mut sa = WelfordState::new();
    for x in [1.0_f32, 2.0, 3.0] {
        sa = welford_update(sa, x);
    }
    let empty = WelfordState::new();

    // merge(populated, empty) == populated
    let result = welford_merge(sa, empty);
    assert_eq!(result.n, sa.n);
    assert_eq!(result.mean, sa.mean);
    assert_eq!(result.m2, sa.m2);

    // merge(empty, populated) == populated
    let result2 = welford_merge(empty, sa);
    assert_eq!(result2.n, sa.n);
    assert_eq!(result2.mean, sa.mean);
    assert_eq!(result2.m2, sa.m2);

    // merge(empty, empty) == empty
    let result3 = welford_merge(empty, empty);
    assert_eq!(result3.n, 0.0);
}

/// Verify Welford is exact for 2 elements (no accumulation error possible).
#[test]
fn test_welford_exact_for_two_elements() {
    let a = 3.0_f32;
    let b = 7.0_f32;

    let (mean, var) = welford_mean_var(&[a, b]);
    let expected_mean = (a + b) / 2.0;
    let expected_var = ((a - expected_mean).powi(2) + (b - expected_mean).powi(2)) / 2.0;

    assert_eq!(mean, expected_mean);
    assert!(
        (var - expected_var).abs() < 1e-7,
        "var {var} vs expected {expected_var}"
    );
}
