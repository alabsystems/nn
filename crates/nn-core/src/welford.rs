// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Reference implementation for Kani verification — production code uses MSL.
#![allow(dead_code)]

//! Rust reference implementation of the Kahan-compensated Welford algorithm.
//!
//! Mirrors the MSL code in `nn_metal::dyn_tensor_metal_welford_msl` exactly,
//! enabling Kani verification of numerical properties that cannot be verified
//! on the GPU side directly. Lives in nn-core so unit tests compile without
//! Metal dependencies.
//!
//! ## Algorithms
//!
//! - [`welford_update`]: Accumulate one sample. Kahan-compensated m2 (#2696).
//! - [`welford_merge`]: Merge two accumulators (parallel tree reduction).
//! - [`welford_update_uncompensated`]: Same without Kahan compensation (for
//!   differential proofs that compensation does not increase error).
//!
//! ## Kani Harnesses
//!
//! 7 harnesses prove:
//! 1. `welford_update` produces finite output for bounded inputs
//! 2. `welford_merge` produces finite output for valid states
//! 3. Count increments exactly by 1.0 per update
//! 4. Merged count equals sum of input counts
//! 5. Mean stays within convex hull of samples
//!    6a. First update from ZERO produces m2 == 0, m2_comp == 0
//!    6b. Compensation is harmless when m2 and m2_comp start at zero
//!
//! Part of #2703.

/// Online mean/variance accumulator state (Kahan-compensated).
///
/// Matches the MSL `WelfordState` struct in `welford_msl_preamble()`.
/// Fields are f32 to match GPU precision exactly.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WelfordState {
    /// Sample count (stored as float to match MSL).
    pub(crate) n: f32,
    /// Running mean.
    pub(crate) mean: f32,
    /// Running sum of squared deviations from the mean (M2).
    pub(crate) m2: f32,
    /// Kahan compensation term for m2 accumulation.
    pub(crate) m2_comp: f32,
}

impl WelfordState {
    /// Zero-initialized state (no samples seen).
    pub(crate) const ZERO: Self = Self {
        n: 0.0,
        mean: 0.0,
        m2: 0.0,
        m2_comp: 0.0,
    };

    /// Returns true when all fields are finite (no NaN, no Inf).
    pub(crate) fn is_finite(&self) -> bool {
        self.n.is_finite()
            && self.mean.is_finite()
            && self.m2.is_finite()
            && self.m2_comp.is_finite()
    }

    /// Variance = m2 / max(n, 1).
    pub(crate) fn variance(&self) -> f32 {
        self.m2 / f32::max(self.n, 1.0)
    }
}

/// Accumulate a single sample into a Welford accumulator.
///
/// Exact Rust translation of the MSL `welford_update()` function.
/// m2 uses Kahan-compensated summation to prevent systematic drift.
#[must_use]
pub(crate) fn welford_update(state: WelfordState, x: f32) -> WelfordState {
    let n = state.n + 1.0;
    let delta = x - state.mean;
    let mean = state.mean + delta / n;
    let delta2 = x - mean;
    // Kahan-compensated m2 accumulation (#2696)
    let y = delta * delta2 - state.m2_comp;
    let t = state.m2 + y;
    let m2_comp = (t - state.m2) - y;
    let m2 = t;
    WelfordState {
        n,
        mean,
        m2,
        m2_comp,
    }
}

/// Merge two Welford accumulators (for parallel tree reduction).
///
/// Exact Rust translation of the MSL `welford_merge()` function.
/// m2 merge uses Kahan compensation to prevent systematic drift.
#[must_use]
pub(crate) fn welford_merge(a: WelfordState, b: WelfordState) -> WelfordState {
    if b.n == 0.0 {
        return a;
    }
    if a.n == 0.0 {
        return b;
    }
    let n = a.n + b.n;
    let delta = b.mean - a.mean;
    let mean = a.mean + delta * b.n / n;
    // Kahan-compensated m2 merge (#2696)
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

/// Welford update WITHOUT Kahan compensation (for differential proofs).
///
/// Same algorithm as `welford_update` but accumulates m2 directly.
/// Used by Kani harnesses to prove compensation does not increase error.
#[must_use]
pub(crate) fn welford_update_uncompensated(state: WelfordState, x: f32) -> WelfordState {
    let n = state.n + 1.0;
    let delta = x - state.mean;
    let mean = state.mean + delta / n;
    let delta2 = x - mean;
    let m2 = state.m2 + delta * delta2;
    WelfordState {
        n,
        mean,
        m2,
        m2_comp: 0.0,
    }
}

// ---------------------------------------------------------------------------
// Kani proof harnesses (#2703)
// ---------------------------------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // -- Harness 1: welford_update produces finite output for bounded inputs --

    /// Proves `welford_update` returns finite state for any bounded input.
    ///
    /// Domain: x in [-1e6, 1e6], prior state has n in [0, 1e6] and all fields
    /// finite and bounded. This covers the production input range for
    /// InstanceNorm/AdaIN/AdaLayerNorm kernels (audio features).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn welford_update_finite_for_bounded_inputs() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite());
        kani::assume(x >= -1.0e6 && x <= 1.0e6);

        let state = WelfordState {
            n: {
                let v: f32 = kani::any();
                kani::assume(v.is_finite() && v >= 0.0 && v < 1.0e6);
                v
            },
            mean: {
                let v: f32 = kani::any();
                kani::assume(v.is_finite() && v >= -1.0e6 && v <= 1.0e6);
                v
            },
            m2: {
                let v: f32 = kani::any();
                kani::assume(v.is_finite() && v >= 0.0 && v <= 1.0e12);
                v
            },
            m2_comp: {
                let v: f32 = kani::any();
                kani::assume(v.is_finite() && v >= -1.0e6 && v <= 1.0e6);
                v
            },
        };

        let result = welford_update(state, x);

        assert!(result.n.is_finite(), "n must be finite after update");
        assert!(result.mean.is_finite(), "mean must be finite after update");
        assert!(result.m2.is_finite(), "m2 must be finite after update");
        assert!(
            result.m2_comp.is_finite(),
            "m2_comp must be finite after update"
        );
    }

    // -- Harness 2: welford_merge produces finite output for valid states --

    /// Proves `welford_merge` returns finite state when both inputs are valid.
    ///
    /// Valid = all fields finite, n >= 0, m2 >= 0. Covers the parallel tree
    /// reduction in `welford_reduction_msl`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn welford_merge_finite_for_valid_states() {
        let make_state = || -> WelfordState {
            let n: f32 = kani::any();
            let mean: f32 = kani::any();
            let m2: f32 = kani::any();
            let m2_comp: f32 = kani::any();
            kani::assume(n.is_finite() && n >= 0.0 && n <= 1.0e6);
            kani::assume(mean.is_finite() && mean >= -1.0e6 && mean <= 1.0e6);
            kani::assume(m2.is_finite() && m2 >= 0.0 && m2 <= 1.0e12);
            kani::assume(m2_comp.is_finite() && m2_comp >= -1.0e6 && m2_comp <= 1.0e6);
            WelfordState {
                n,
                mean,
                m2,
                m2_comp,
            }
        };

        let a = make_state();
        let b = make_state();

        let result = welford_merge(a, b);

        assert!(result.n.is_finite(), "merged n must be finite");
        assert!(result.mean.is_finite(), "merged mean must be finite");
        assert!(result.m2.is_finite(), "merged m2 must be finite");
        assert!(result.m2_comp.is_finite(), "merged m2_comp must be finite");
    }

    // -- Harness 3: count increments exactly by 1.0 --

    /// Proves n increases by exactly 1.0 per `welford_update` call.
    ///
    /// Since n starts from 0 and increments by 1.0, for n < 2^24 (f32 mantissa
    /// precision for integers), this is exact. Production kernels run with
    /// n up to ~300k (spatial dimensions), well within this range.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn welford_update_count_exact() {
        let n_old: f32 = kani::any();
        kani::assume(n_old.is_finite() && n_old >= 0.0 && n_old < 16_777_216.0);

        let x: f32 = kani::any();
        kani::assume(x.is_finite() && x >= -1.0e6 && x <= 1.0e6);

        let state = WelfordState {
            n: n_old,
            mean: 0.0,
            m2: 0.0,
            m2_comp: 0.0,
        };

        let result = welford_update(state, x);
        assert!(
            result.n == n_old + 1.0,
            "count must increment by exactly 1.0"
        );
    }

    // -- Harness 4: merged count equals sum of input counts --

    /// Proves `merge(a, b).n == a.n + b.n` for valid integer-valued counts.
    ///
    /// Counts in production are always exact integers (threadgroup lane counts
    /// from the parallel reduction). This is exact for n < 2^24.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn welford_merge_count_additive() {
        let n_a: f32 = kani::any();
        let n_b: f32 = kani::any();
        kani::assume(n_a.is_finite() && n_a >= 1.0 && n_a < 8_388_608.0);
        kani::assume(n_b.is_finite() && n_b >= 1.0 && n_b < 8_388_608.0);

        let a = WelfordState {
            n: n_a,
            mean: 0.0,
            m2: 0.0,
            m2_comp: 0.0,
        };
        let b = WelfordState {
            n: n_b,
            mean: 0.0,
            m2: 0.0,
            m2_comp: 0.0,
        };

        let result = welford_merge(a, b);
        assert!(result.n == n_a + n_b, "merged count must equal a.n + b.n");
    }

    // -- Harness 5: mean stays in convex hull of samples --

    /// Proves the running mean after update stays between the old mean and the
    /// new sample (convex hull property).
    ///
    /// For n >= 1, mean is a weighted average of old data and new sample,
    /// so `min(old_mean, x) <= new_mean <= max(old_mean, x)`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn welford_update_mean_bounded() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite() && x >= -1.0e4 && x <= 1.0e4);

        let n_old: f32 = kani::any();
        kani::assume(n_old.is_finite() && n_old >= 1.0 && n_old < 1.0e6);

        let mean_old: f32 = kani::any();
        kani::assume(mean_old.is_finite() && mean_old >= -1.0e4 && mean_old <= 1.0e4);

        let state = WelfordState {
            n: n_old,
            mean: mean_old,
            m2: 0.0,
            m2_comp: 0.0,
        };

        let result = welford_update(state, x);

        // Mean must be finite (subset of harness 1 but stated explicitly)
        assert!(result.mean.is_finite(), "mean must be finite");

        // Convex hull: new mean is between old mean and new sample.
        let lo = f32::min(mean_old, x);
        let hi = f32::max(mean_old, x);
        assert!(
            result.mean >= lo && result.mean <= hi,
            "mean must stay in convex hull [min(old_mean, x), max(old_mean, x)]"
        );
    }

    // -- Harness 6a: first update from ZERO zeroes m2 and m2_comp --

    /// Proves that `welford_update(ZERO, x)` produces `m2 == 0` and
    /// `m2_comp == 0` for any bounded input.
    ///
    /// After the first sample, delta2 = x - mean = x - x = 0, so
    /// delta * delta2 = 0 and both m2 and m2_comp remain zero.
    /// This is the base case for harness 6b.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn welford_first_update_zeroes_m2() {
        let x: f32 = kani::any();
        kani::assume(x.is_finite() && x >= -1.0e4 && x <= 1.0e4);

        let s = welford_update(WelfordState::ZERO, x);

        assert!(s.is_finite(), "state must be finite after first update");
        assert!(s.n == 1.0, "count must be 1 after first update");
        assert!(s.mean == x, "mean must equal the single sample");
        assert!(s.m2 == 0.0, "m2 must be 0 after first sample");
        assert!(s.m2_comp == 0.0, "m2_comp must be 0 after first sample");
    }

    // -- Harness 6b: compensation is harmless when m2 and m2_comp are zero --

    /// Proves Kahan compensation produces m2_comp == 0 (and thus identical
    /// m2 to uncompensated) for a single update from a state with m2 == 0
    /// and m2_comp == 0.
    ///
    /// Combined with harness 6a, this proves that for any 2-sample sequence
    /// from ZERO, compensated and uncompensated Welford produce identical m2.
    ///
    /// Decomposed from the original `welford_compensation_no_worse_than_uncompensated`
    /// harness which timed out with 4 chained function calls (#2742).
    /// Each sub-harness has only 1 call, making CBMC's SAT formula tractable.
    ///
    /// The strict advantage of compensation manifests at N >= 3 where
    /// accumulated m2_comp corrections diverge. Proving the N >= 3 case
    /// requires f64 ground truth which CBMC cannot model tractably.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn welford_compensation_harmless_from_zero_m2() {
        let n_old: f32 = kani::any();
        let mean_old: f32 = kani::any();
        let x: f32 = kani::any();
        kani::assume(n_old.is_finite() && n_old >= 1.0 && n_old < 1.0e6);
        kani::assume(mean_old.is_finite() && mean_old >= -1.0e4 && mean_old <= 1.0e4);
        kani::assume(x.is_finite() && x >= -1.0e4 && x <= 1.0e4);

        // State after first update: m2 and m2_comp are zero (proved by 6a)
        let state = WelfordState {
            n: n_old,
            mean: mean_old,
            m2: 0.0,
            m2_comp: 0.0,
        };

        let comp = welford_update(state, x);
        let uncomp = welford_update_uncompensated(state, x);

        assert!(comp.is_finite(), "compensated state must be finite");
        assert!(uncomp.is_finite(), "uncompensated state must be finite");

        // When m2_comp == 0 and m2 == 0:
        //   Compensated: y = delta*delta2 - 0, t = 0 + y, m2 = y
        //                m2_comp = (t - 0) - y = y - y = 0
        //   Uncompensated: m2 = 0 + delta*delta2
        // Both produce m2 = delta * delta2. Identical f32 arithmetic.
        assert!(
            comp.m2 == uncomp.m2,
            "compensated m2 must equal uncompensated when starting from zero m2"
        );

        // m2 must be non-negative (sum of squared deviations)
        assert!(comp.m2 >= 0.0, "m2 must be non-negative");

        // Compensation term stays zero (no accumulated error to correct)
        assert!(
            comp.m2_comp == 0.0,
            "m2_comp must remain 0 when starting from zero m2"
        );
    }
}

#[cfg(test)]
#[path = "welford_tests.rs"]
mod tests;
