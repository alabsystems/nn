// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for probabilistic confidence arithmetic.

#[cfg(kani)]
mod proofs {
    fn normalize_failure_budget(confidence: f64) -> f64 {
        1.0 - confidence
    }

    fn confidence_interval(mean: f64, epsilon: f64) -> (f64, f64) {
        (mean - epsilon, mean + epsilon)
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn confidence_and_failure_budget_are_normalized() {
        let confidence: f64 = kani::any();
        kani::assume(confidence.is_finite());
        kani::assume((0.0..=1.0).contains(&confidence));

        let delta = normalize_failure_budget(confidence);

        assert!((0.0..=1.0).contains(&delta));
        assert!((confidence + delta - 1.0).abs() <= 1e-12);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn bonferroni_split_preserves_probability_mass() {
        let confidence: f64 = kani::any();
        let dims: usize = kani::any();
        kani::assume(confidence.is_finite());
        kani::assume((0.0..=1.0).contains(&confidence));
        kani::assume(dims >= 1 && dims <= 128);

        let delta_total = normalize_failure_budget(confidence);
        let delta_per_dim = delta_total / dims as f64;

        assert!(delta_per_dim >= 0.0);
        assert!(delta_per_dim <= delta_total + 1e-12);
        assert!(((delta_per_dim * dims as f64) - delta_total).abs() <= 1e-12);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn larger_epsilon_widens_confidence_bounds() {
        let mean: f64 = kani::any();
        let eps_small: f64 = kani::any();
        let eps_large: f64 = kani::any();
        kani::assume(mean.is_finite() && eps_small.is_finite() && eps_large.is_finite());
        kani::assume(mean.abs() <= 1_000.0);
        kani::assume(eps_small >= 0.0 && eps_large >= eps_small);
        kani::assume(eps_large <= 1_000.0);

        let (lo_small, hi_small) = confidence_interval(mean, eps_small);
        let (lo_large, hi_large) = confidence_interval(mean, eps_large);

        assert!(lo_large <= lo_small);
        assert!(hi_small <= hi_large);
        assert!(lo_large <= mean && mean <= hi_large);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn tighter_min_epsilon_interval_is_nested() {
        let mean: f64 = kani::any();
        let hoeffding_eps: f64 = kani::any();
        let mcdiarmid_eps: f64 = kani::any();
        kani::assume(mean.is_finite());
        kani::assume(hoeffding_eps.is_finite() && mcdiarmid_eps.is_finite());
        kani::assume(mean.abs() <= 1_000.0);
        kani::assume(hoeffding_eps >= 0.0 && mcdiarmid_eps >= 0.0);
        kani::assume(hoeffding_eps <= 1_000.0 && mcdiarmid_eps <= 1_000.0);

        let tight_eps = hoeffding_eps.min(mcdiarmid_eps);
        let (tight_lo, tight_hi) = confidence_interval(mean, tight_eps);
        let (h_lo, h_hi) = confidence_interval(mean, hoeffding_eps);
        let (m_lo, m_hi) = confidence_interval(mean, mcdiarmid_eps);

        assert!(h_lo <= tight_lo && tight_hi <= h_hi);
        assert!(m_lo <= tight_lo && tight_hi <= m_hi);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn confidence_interval_width_is_twice_epsilon() {
        let mean: f64 = kani::any();
        let epsilon: f64 = kani::any();
        kani::assume(mean.is_finite() && epsilon.is_finite());
        kani::assume(mean.abs() <= 1_000.0);
        kani::assume(epsilon >= 0.0 && epsilon <= 1_000.0);

        let (lower, upper) = confidence_interval(mean, epsilon);

        assert!(((upper - lower) - (2.0 * epsilon)).abs() <= 1e-12);
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn confidence_interval_center_is_the_empirical_mean() {
        let mean: f64 = kani::any();
        let epsilon: f64 = kani::any();
        kani::assume(mean.is_finite() && epsilon.is_finite());
        kani::assume(mean.abs() <= 1_000.0);
        kani::assume(epsilon >= 0.0 && epsilon <= 1_000.0);

        let (lower, upper) = confidence_interval(mean, epsilon);
        let midpoint = (lower + upper) / 2.0;

        assert!((midpoint - mean).abs() <= 1e-12);
        assert!(lower <= mean && mean <= upper);
    }
}
