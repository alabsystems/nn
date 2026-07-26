// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Analytical output bounds for harmonic_source (cumsum → sin pattern).
//!
//! Kokoro's `harmonic_source()` uses `cumsum(dim)` on the time dimension
//! to accumulate phase, then applies `sin()` to produce a harmonic signal.
//! At 24kHz, 1 second of audio has T=24000 samples — far exceeding the
//! verify-path cumsum decomposition cap (`MAX_DECOMPOSE_DIM = 2048`).
//!
//! This module provides analytical bounds that allow the verify pipeline
//! to handle cumsum on arbitrarily large dimensions by recognizing that:
//!
//! 1. `cumsum` of any real-valued input produces unbounded accumulated values
//! 2. `sin(cumsum(...))` is always in `[-1, 1]` regardless of accumulated value
//! 3. For the Kokoro SourceModule path: `SineGen → Linear → tanh` bounds to `(-1, 1)`
//!
//! When the cumsum dimension exceeds `MAX_DECOMPOSE_DIM`, the trace-to-graph
//! translator emits an identity pass-through (`AddConstant(0.0)`) instead of
//! decomposing into O(N) slice+add nodes. The downstream `sin()` layer in
//! NY's IBP naturally produces `[-1, 1]` bounds. This is sound
//! because `sin` is bounded regardless of its input magnitude.
//!
//! # Mathematical justification
//!
//! For `harmonic_source(f0, sr)`:
//! - `phase_inc = f0 * (2π / sr)` — per-sample phase increment
//! - `phase = cumsum(phase_inc, dim=2)` — accumulated phase
//! - `output = sin(phase)` — harmonic signal
//!
//! **Bound:** `∀ phase ∈ ℝ: sin(phase) ∈ [-1, 1]`
//!
//! This is exact (not an approximation). The sin function is bounded by
//! definition, independent of its input range. IBP through
//! `Identity → Sin` produces `[-1, 1]` because NY's `Sin` layer
//! propagates `lower = -1, upper = 1` for any input range exceeding `2π`.
//!
//! For the full SourceModule path (with learned weights):
//! - `SineGen: sin(phase) ∈ [-1, 1]` for each of 9 harmonics
//! - `l_linear: Linear([1, 9])` — weight-dependent, but followed by:
//! - `tanh(projected) ∈ (-1, 1)` — hard analytical bound
//!
//! Part of #2411 (cumsum 2048 verification cap).

use crate::error::VerifyError;

/// Analytical bounds for the harmonic_source cumsum→sin pattern.
///
/// Provides verified output bounds that bypass the O(N) cumsum decomposition
/// when the time dimension exceeds `MAX_DECOMPOSE_DIM`.
///
/// # Soundness
///
/// The bounds are exact (not heuristic):
/// - `sin(x) ∈ [-1, 1]` for all real x (mathematical identity)
/// - `tanh(x) ∈ (-1, 1)` for all real x (monotone sigmoid)
/// - These hold regardless of cumsum accumulation depth
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicSourceBounds {
    /// Lower bound on sin(cumsum(...)) output.
    pub sin_lower: f64,
    /// Upper bound on sin(cumsum(...)) output.
    pub sin_upper: f64,
    /// Lower bound on tanh(linear(sin(cumsum(...)))) output (SourceModule path).
    pub tanh_lower: f64,
    /// Upper bound on tanh(linear(sin(cumsum(...)))) output (SourceModule path).
    pub tanh_upper: f64,
    /// The cumsum dimension size that triggered analytical bypass.
    pub cumsum_dim_size: usize,
}

impl HarmonicSourceBounds {
    /// Compute analytical bounds for a cumsum dimension that exceeds the
    /// decomposition cap.
    ///
    /// # Arguments
    /// * `dim_size` — The size of the cumsum dimension (e.g., T=24000)
    ///
    /// # Errors
    /// Returns `VerifyError` if `dim_size` is 0.
    pub fn new(dim_size: usize) -> Result<Self, VerifyError> {
        if dim_size == 0 {
            return Err(VerifyError::UnsupportedOp(
                "HarmonicSourceBounds: dim_size must be > 0".into(),
            ));
        }
        Ok(Self {
            sin_lower: -1.0,
            sin_upper: 1.0,
            tanh_lower: -1.0,
            tanh_upper: 1.0,
            cumsum_dim_size: dim_size,
        })
    }

    /// Width of the sin output bounds (always 2.0).
    #[must_use]
    pub fn sin_width(&self) -> f64 {
        self.sin_upper - self.sin_lower
    }

    /// Width of the tanh output bounds (always 2.0).
    #[must_use]
    pub fn tanh_width(&self) -> f64 {
        self.tanh_upper - self.tanh_lower
    }

    /// Verify that the bounds are mathematically valid.
    ///
    /// Checks the fundamental mathematical identities:
    /// - sin(x) ∈ [-1, 1] for all real x
    /// - tanh(x) ∈ (-1, 1) for all real x
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.sin_lower == -1.0
            && self.sin_upper == 1.0
            && self.tanh_lower >= -1.0
            && self.tanh_upper <= 1.0
            && self.cumsum_dim_size > 0
    }

    /// Maximum cumsum dimension for which O(N) decomposition is tractable.
    ///
    /// Beyond this, the analytical bypass is used. Matches the cumsum
    /// decomposition limit in the NY-owned translator
    /// (ny-trace-bridge `translate/ops_misc.rs`).
    pub const MAX_DECOMPOSE_DIM: usize = 2048;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_source_bounds_basic() {
        let bounds = HarmonicSourceBounds::new(24000).expect("valid bounds");
        assert_eq!(bounds.sin_lower, -1.0);
        assert_eq!(bounds.sin_upper, 1.0);
        assert_eq!(bounds.cumsum_dim_size, 24000);
        assert!(bounds.is_valid());
    }

    #[test]
    fn test_harmonic_source_bounds_width() {
        let bounds = HarmonicSourceBounds::new(100).expect("valid bounds");
        assert!((bounds.sin_width() - 2.0).abs() < 1e-12);
        assert!((bounds.tanh_width() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_harmonic_source_bounds_zero_dim_errors() {
        let err = HarmonicSourceBounds::new(0);
        assert!(err.is_err());
    }

    #[test]
    fn test_harmonic_source_bounds_at_cap() {
        // At exactly MAX_DECOMPOSE_DIM, normal decomposition should still work.
        // Analytical bypass is for dim > MAX_DECOMPOSE_DIM.
        let bounds = HarmonicSourceBounds::new(2048).expect("valid bounds");
        assert!(bounds.is_valid());
    }

    #[test]
    fn test_harmonic_source_bounds_large_dim() {
        // 10 seconds of 24kHz audio = 240,000 samples.
        let bounds = HarmonicSourceBounds::new(240_000).expect("valid bounds");
        assert!(bounds.is_valid());
        assert_eq!(bounds.cumsum_dim_size, 240_000);
        // sin bounds are always [-1, 1] regardless of dimension size.
        assert_eq!(bounds.sin_lower, -1.0);
        assert_eq!(bounds.sin_upper, 1.0);
    }

    /// Verify the mathematical identity: sin(x) ∈ [-1, 1] for all x.
    ///
    /// Tests at extreme accumulated phase values that would occur with
    /// large cumsum dimensions.
    #[test]
    fn test_sin_bounds_hold_for_large_accumulated_phase() {
        // Kokoro: f0 up to 1600 Hz, sr = 24000, T = 24000 (1 second)
        // Max phase = 2π × 1600/24000 × 24000 = 2π × 1600 ≈ 10053 radians
        let max_phase = 2.0 * std::f64::consts::PI * 1600.0;
        let sin_val = max_phase.sin();
        assert!(
            (-1.0..=1.0).contains(&sin_val),
            "sin({max_phase}) = {sin_val} must be in [-1, 1]"
        );

        // Even larger: 10 seconds of audio
        let huge_phase = 2.0 * std::f64::consts::PI * 1600.0 * 10.0;
        let sin_val2 = huge_phase.sin();
        assert!(
            (-1.0..=1.0).contains(&sin_val2),
            "sin({huge_phase}) = {sin_val2} must be in [-1, 1]"
        );
    }

    /// Verify the mathematical identity: tanh(x) ∈ [-1, 1] for all finite x.
    ///
    /// Note: mathematically tanh(x) is in the open interval (-1, 1), but
    /// IEEE 754 f64 saturates to exactly -1.0/1.0 for extreme inputs
    /// (e.g., tanh(1e6) == 1.0 in f64). The bounds [-1, 1] are still
    /// correct and useful for verification.
    #[test]
    fn test_tanh_bounds_hold_for_arbitrary_input() {
        for &x in &[-1e6_f64, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6] {
            let t = x.tanh();
            assert!((-1.0..=1.0).contains(&t), "tanh({x}) = {t} must be in [-1, 1]");
        }
    }
}
