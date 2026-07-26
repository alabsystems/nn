// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Reference implementation for Kani verification — production code uses MSL.
#![allow(dead_code)]

//! Rust reference implementation of the two-pass Kahan-compensated reduction.
//!
//! Mirrors the MSL code in `nn_metal::dyn_tensor_metal_welford_msl`
//! (`kahan_two_pass_reduction_msl`) exactly, enabling Kani verification of
//! numerical properties. Lives in nn-core so harnesses compile without Metal.
//!
//! ## Algorithm
//!
//! Two-pass approach:
//! - **Pass 1:** Kahan-compensated sum → divide by N → mean.
//! - **Pass 2:** Kahan-compensated sum of (x − mean)² → divide by N → variance.
//!
//! Each pass uses [`KahanAcc`] for compensated summation and [`kahan_merge`]
//! for parallel tree reduction.
//!
//! ## Kani Harnesses
//!
//! 5 harnesses prove:
//! 1. `kahan_sum` produces finite output for bounded inputs
//! 2. Kahan sum has less error than naive sum (compensation reduces error)
//! 3. `kahan_two_pass_mean_var` produces non-negative variance
//! 4. Mean stays within convex hull [min(data), max(data)]
//! 5. `kahan_merge` produces finite output for valid states (tree reduction)
//!
//! Part of #2735.

/// Per-element Kahan accumulator state.
///
/// Matches the MSL `KahanAcc` struct in `kahan_two_pass_reduction_msl`.
/// Fields are f32 to match GPU precision exactly.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KahanAcc {
    /// Running sum.
    pub(crate) sum: f32,
    /// Kahan compensation term.
    pub(crate) comp: f32,
}

impl KahanAcc {
    /// Zero-initialized accumulator.
    pub(crate) const ZERO: Self = Self {
        sum: 0.0,
        comp: 0.0,
    };

    /// Returns true when both fields are finite (no NaN, no Inf).
    pub(crate) fn is_finite(&self) -> bool {
        self.sum.is_finite() && self.comp.is_finite()
    }

    /// Accumulate one value using Kahan-compensated summation.
    ///
    /// Exact Rust translation of the MSL `kahan_add()` logic.
    #[must_use]
    pub(crate) fn add(self, val: f32) -> Self {
        let y = val - self.comp;
        let t = self.sum + y;
        let comp = (t - self.sum) - y;
        Self { sum: t, comp }
    }
}

/// Merge two Kahan accumulators (for parallel tree reduction).
///
/// Exact Rust translation of the MSL tree reduction merge step.
#[must_use]
pub(crate) fn kahan_merge(a: KahanAcc, b: KahanAcc) -> KahanAcc {
    let y = b.sum - (a.comp + b.comp);
    let t = a.sum + y;
    KahanAcc {
        sum: t,
        comp: (t - a.sum) - y,
    }
}

/// Kahan-compensated sum of a slice.
///
/// Sequential version (no tree reduction). For Kani verification of the
/// core summation property.
#[must_use]
pub(crate) fn kahan_sum(data: &[f32]) -> f32 {
    let mut acc = KahanAcc::ZERO;
    for &x in data {
        acc = acc.add(x);
    }
    acc.sum
}

/// Two-pass Kahan-compensated mean and variance (sequential).
///
/// Pass 1: Kahan sum / N → mean.
/// Pass 2: Kahan sum of (x − mean)² / N → variance.
///
/// This is the sequential equivalent of `kahan_two_pass_tree_reduction` in the
/// GPU algorithm tests. The tree-reduction version distributes work across
/// threads but computes the same result (modulo merge-order rounding).
#[must_use]
pub(crate) fn kahan_two_pass_mean_var(data: &[f32]) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let n = data.len() as f32;

    // Pass 1: Kahan-compensated sum for mean.
    let sum = kahan_sum(data);
    let mean = sum / n;

    // Pass 2: Kahan-compensated sum of (x - mean)^2.
    let mut acc = KahanAcc::ZERO;
    for &x in data {
        let diff = x - mean;
        acc = acc.add(diff * diff);
    }
    let var = acc.sum / n;

    (mean, var)
}

/// Naive (non-compensated) sum for differential proofs.
///
/// Used by Kani harnesses to prove Kahan compensation reduces error.
#[must_use]
pub(crate) fn naive_sum(data: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for &x in data {
        sum += x;
    }
    sum
}

// ---------------------------------------------------------------------------
// Kani proof harnesses (#2735)
// ---------------------------------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // -- Harness 1: kahan_sum produces finite output for bounded inputs --

    /// Proves `KahanAcc::add` returns finite state for bounded inputs.
    ///
    /// Domain: values in [-1e6, 1e6], up to 4 accumulations (enough to
    /// exercise the compensation path). Covers the production input range
    /// for InstanceNorm/AdaIN/AdaLayerNorm audio features.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn kahan_acc_finite_for_bounded_inputs() {
        let v0: f32 = kani::any();
        let v1: f32 = kani::any();
        let v2: f32 = kani::any();
        let v3: f32 = kani::any();

        kani::assume(v0.is_finite() && v0 >= -1.0e6 && v0 <= 1.0e6);
        kani::assume(v1.is_finite() && v1 >= -1.0e6 && v1 <= 1.0e6);
        kani::assume(v2.is_finite() && v2 >= -1.0e6 && v2 <= 1.0e6);
        kani::assume(v3.is_finite() && v3 >= -1.0e6 && v3 <= 1.0e6);

        let acc = KahanAcc::ZERO;
        let acc = acc.add(v0);
        assert!(acc.is_finite(), "acc must be finite after 1 add");
        let acc = acc.add(v1);
        assert!(acc.is_finite(), "acc must be finite after 2 adds");
        let acc = acc.add(v2);
        assert!(acc.is_finite(), "acc must be finite after 3 adds");
        let acc = acc.add(v3);
        assert!(acc.is_finite(), "acc must be finite after 4 adds");
    }

    // -- Harness 2: Kahan compensation reduces error vs naive --

    /// Proves Kahan-compensated sum is no worse than naive sum for 3 values.
    ///
    /// Uses f64 arithmetic as ground truth. For N=3, Kahan compensation
    /// starts showing its benefit (compensation term from step 2 corrects
    /// step 3). The harness proves |kahan_sum - ref| <= |naive_sum - ref|.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn kahan_sum_no_worse_than_naive() {
        let v0: f32 = kani::any();
        let v1: f32 = kani::any();
        let v2: f32 = kani::any();

        kani::assume(v0.is_finite() && v0 >= -1.0e4 && v0 <= 1.0e4);
        kani::assume(v1.is_finite() && v1 >= -1.0e4 && v1 <= 1.0e4);
        kani::assume(v2.is_finite() && v2 >= -1.0e4 && v2 <= 1.0e4);

        // Kahan path
        let acc = KahanAcc::ZERO.add(v0).add(v1).add(v2);
        let kahan_result = acc.sum;

        // Naive path
        let naive_result = ((v0 + v1) + v2) as f32;

        // f64 reference (exact for f32-range inputs)
        let ref_sum = v0 as f64 + v1 as f64 + v2 as f64;

        let kahan_err = ((kahan_result as f64) - ref_sum).abs();
        let naive_err = ((naive_result as f64) - ref_sum).abs();

        assert!(
            kahan_err <= naive_err,
            "Kahan sum must not have greater error than naive sum"
        );
    }

    // -- Harness 3: variance is non-negative --

    /// Proves the two-pass variance computation produces non-negative results.
    ///
    /// The sum of (x - mean)^2 is algebraically non-negative. Floating-point
    /// rounding could theoretically produce a tiny negative, but Kahan
    /// compensation prevents this for bounded inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn kahan_two_pass_variance_non_negative() {
        let v0: f32 = kani::any();
        let v1: f32 = kani::any();

        kani::assume(v0.is_finite() && v0 >= -1.0e4 && v0 <= 1.0e4);
        kani::assume(v1.is_finite() && v1 >= -1.0e4 && v1 <= 1.0e4);

        let n = 2.0_f32;

        // Pass 1: mean via Kahan sum
        let sum_acc = KahanAcc::ZERO.add(v0).add(v1);
        let mean = sum_acc.sum / n;

        // Pass 2: sum of (x - mean)^2
        let d0 = v0 - mean;
        let d1 = v1 - mean;
        let var_acc = KahanAcc::ZERO.add(d0 * d0).add(d1 * d1);
        let var = var_acc.sum / n;

        assert!(mean.is_finite(), "mean must be finite");
        assert!(var.is_finite(), "variance must be finite");
        assert!(var >= 0.0, "variance must be non-negative");
    }

    // -- Harness 5: kahan_merge produces finite output for valid states --

    /// Proves `kahan_merge` returns finite state when both inputs are valid.
    ///
    /// This covers the GPU tree reduction merge step in
    /// `kahan_two_pass_reduction_msl`. Each merge combines two per-thread
    /// accumulators — finiteness must be preserved through all log2(tg_size)
    /// merge levels.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn kahan_merge_finite_for_valid_states() {
        let a = KahanAcc {
            sum: {
                let v: f32 = kani::any();
                kani::assume(v.is_finite() && v >= -1.0e9 && v <= 1.0e9);
                v
            },
            comp: {
                let v: f32 = kani::any();
                kani::assume(v.is_finite() && v >= -1.0e3 && v <= 1.0e3);
                v
            },
        };
        let b = KahanAcc {
            sum: {
                let v: f32 = kani::any();
                kani::assume(v.is_finite() && v >= -1.0e9 && v <= 1.0e9);
                v
            },
            comp: {
                let v: f32 = kani::any();
                kani::assume(v.is_finite() && v >= -1.0e3 && v <= 1.0e3);
                v
            },
        };

        let merged = kahan_merge(a, b);

        assert!(merged.is_finite(), "merged accumulator must be finite");
    }

    // -- Harness 4: mean in convex hull --

    /// Proves the mean is between min and max of the input data.
    ///
    /// For N=2: mean = (v0 + v1) / 2, which is always in [min(v0,v1), max(v0,v1)].
    /// Floating-point rounding of the Kahan sum and division could
    /// theoretically push outside the hull, but bounded inputs prevent this.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn kahan_two_pass_mean_in_convex_hull() {
        let v0: f32 = kani::any();
        let v1: f32 = kani::any();

        kani::assume(v0.is_finite() && v0 >= -1.0e4 && v0 <= 1.0e4);
        kani::assume(v1.is_finite() && v1 >= -1.0e4 && v1 <= 1.0e4);

        let n = 2.0_f32;
        let sum_acc = KahanAcc::ZERO.add(v0).add(v1);
        let mean = sum_acc.sum / n;

        assert!(mean.is_finite(), "mean must be finite");

        let lo = f32::min(v0, v1);
        let hi = f32::max(v0, v1);
        assert!(
            mean >= lo && mean <= hi,
            "mean must be in [min(data), max(data)]"
        );
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kahan_two_pass_tests.rs"]
mod tests;
