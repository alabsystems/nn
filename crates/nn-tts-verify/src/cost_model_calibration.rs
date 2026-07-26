// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Roofline calibration — comparing predicted vs measured GPU timing.
//!
//! Validates that the roofline cost model produces conservative (safe) but
//! not vacuously wide timing estimates. A calibration result is "conservative"
//! if `estimated >= measured` and "non-vacuous" if `estimated <= K * measured`
//! for some conservatism factor K (typically 2-10x).
//!
//! # Usage
//!
//! ```text
//! let measurements = vec![
//!     Measurement { step_name: "matmul".into(), measured_time_us: 784.0 },
//! ];
//! let report = calibrate_profiles(&profiles, &measurements);
//! assert!(report.all_conservative());
//! ```
//!
//! Part of #1739 Phase 2 — AC4: Roofline estimates are conservative (>= measured).

use super::LayerCostProfile;

/// A single measured timing observation for a dispatch step.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// Name matching `LayerCostProfile.layer_name`.
    pub step_name: String,
    /// Measured wall-clock time in microseconds (GPU execution).
    pub measured_time_us: f64,
}

/// Per-step calibration result comparing predicted vs measured.
#[derive(Debug, Clone)]
pub struct StepCalibration {
    /// Layer name from the cost profile.
    pub layer_name: String,
    /// Estimated time from roofline model (μs).
    pub estimated_time_us: f64,
    /// Measured time from GPU execution (μs).
    pub measured_time_us: f64,
    /// Conservatism ratio: estimated / measured.
    /// Values >= 1.0 mean the estimate is conservative (safe).
    /// Values < 1.0 mean the estimate is an underestimate (unsafe).
    pub conservatism_ratio: f64,
    /// Whether the estimate is conservative (>= measured).
    pub is_conservative: bool,
}

/// Aggregate calibration report for a dispatch plan.
#[derive(Debug, Clone)]
pub struct CalibrationReport {
    /// Per-step calibration results.
    pub steps: Vec<StepCalibration>,
    /// Steps in the profile that had no matching measurement.
    pub unmatched_steps: Vec<String>,
    /// Number of conservative estimates (estimated >= measured).
    pub conservative_count: usize,
    /// Number of underestimates (estimated < measured).
    pub underestimate_count: usize,
    /// Maximum conservatism ratio across all matched steps.
    pub max_conservatism: f64,
    /// Minimum conservatism ratio across all matched steps.
    pub min_conservatism: f64,
    /// Mean conservatism ratio across all matched steps.
    pub mean_conservatism: f64,
}

impl CalibrationReport {
    /// True if all matched steps have conservative estimates (estimated >= measured).
    pub fn all_conservative(&self) -> bool {
        self.underestimate_count == 0 && !self.steps.is_empty()
    }

    /// True if all estimates are within `max_factor` of measured.
    ///
    /// A vacuously wide estimate (e.g., 100x measured) is not useful even if
    /// conservative. This checks that the conservatism doesn't exceed the
    /// given factor.
    pub fn within_factor(&self, max_factor: f64) -> bool {
        !self.steps.is_empty() && self.max_conservatism <= max_factor
    }

    /// Generate a human-readable calibration report.
    pub fn report(&self) -> String {
        let mut out = String::with_capacity(1024);
        out.push_str("=== Roofline Calibration Report ===\n\n");
        out.push_str(&format!("Steps matched: {}\n", self.steps.len()));
        out.push_str(&format!(
            "Conservative: {} / {}\n",
            self.conservative_count,
            self.steps.len()
        ));
        out.push_str(&format!("Underestimates: {}\n", self.underestimate_count));

        if !self.steps.is_empty() {
            out.push_str(&format!(
                "Conservatism ratio: min={:.2}x, max={:.2}x, mean={:.2}x\n\n",
                self.min_conservatism, self.max_conservatism, self.mean_conservatism,
            ));
        }

        for step in &self.steps {
            out.push_str(&format!(
                "  {}: est={:.1}μs, meas={:.1}μs, ratio={:.2}x {}\n",
                step.layer_name,
                step.estimated_time_us,
                step.measured_time_us,
                step.conservatism_ratio,
                if step.is_conservative { "✓" } else { "✗" },
            ));
        }

        if !self.unmatched_steps.is_empty() {
            out.push_str(&format!(
                "\nUnmatched steps (no measurement): {}\n",
                self.unmatched_steps.join(", "),
            ));
        }

        out
    }
}

/// Compare roofline predictions against measured GPU timing.
///
/// For each step in `profiles` that has a matching entry in `measurements`
/// (by `step_name` == `layer_name`), computes the conservatism ratio
/// `estimated / measured`. Steps without measurements are listed in
/// `unmatched_steps`.
///
/// # Arguments
///
/// * `profiles` - Cost profiles from [`profile_dispatch_plan`](super::profile_dispatch_plan).
/// * `measurements` - Actual GPU timing measurements.
pub fn calibrate_profiles(
    profiles: &[LayerCostProfile],
    measurements: &[Measurement],
) -> CalibrationReport {
    let mut steps = Vec::new();
    let mut unmatched_steps = Vec::new();

    for profile in profiles {
        let measurement = measurements
            .iter()
            .find(|m| m.step_name == profile.layer_name);

        match measurement {
            Some(m) if m.measured_time_us > 0.0 && m.measured_time_us.is_finite() => {
                let ratio = profile.estimated_time_us / m.measured_time_us;
                steps.push(StepCalibration {
                    layer_name: profile.layer_name.clone(),
                    estimated_time_us: profile.estimated_time_us,
                    measured_time_us: m.measured_time_us,
                    conservatism_ratio: ratio,
                    is_conservative: ratio >= 1.0,
                });
            }
            _ => {
                unmatched_steps.push(profile.layer_name.clone());
            }
        }
    }

    let conservative_count = steps.iter().filter(|s| s.is_conservative).count();
    let underestimate_count = steps.len() - conservative_count;

    let (max_conservatism, min_conservatism, mean_conservatism) = if steps.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let max = crate::stats::fold_max_propagate_nan(
            steps.iter().map(|s| s.conservatism_ratio),
            f64::NEG_INFINITY,
        );
        let min = crate::stats::fold_min_propagate_nan(
            steps.iter().map(|s| s.conservatism_ratio),
            f64::INFINITY,
        );
        let sum: f64 = steps.iter().map(|s| s.conservatism_ratio).sum();
        let mean = sum / steps.len() as f64;
        (max, min, mean)
    };

    CalibrationReport {
        steps,
        unmatched_steps,
        conservative_count,
        underestimate_count,
        max_conservatism,
        min_conservatism,
        mean_conservatism,
    }
}

/// Fill `measured_time_us` in profiles from matching measurements.
///
/// Returns a new vector of profiles with measured values filled in where
/// available. This is a non-destructive operation — original profiles are
/// cloned.
pub fn fill_measured(
    profiles: &[LayerCostProfile],
    measurements: &[Measurement],
) -> Vec<LayerCostProfile> {
    profiles
        .iter()
        .map(|p| {
            let measured = measurements
                .iter()
                .find(|m| m.step_name == p.layer_name)
                .map(|m| m.measured_time_us);
            LayerCostProfile {
                layer_name: p.layer_name.clone(),
                flops: p.flops,
                memory_bytes: p.memory_bytes,
                estimated_time_us: p.estimated_time_us,
                measured_time_us: measured.or(p.measured_time_us),
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "cost_model_calibration_tests.rs"]
mod tests;
