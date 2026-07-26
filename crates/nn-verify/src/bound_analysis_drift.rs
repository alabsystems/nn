// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! F32 vs F64 precision drift tracking for [`BoundAnalysisReport`].
//!
//! Extracted from `bound_analysis.rs` to keep it under 500 lines.
//!
//! Part of #2705, Part of #2218.

use super::{AnalysisConfig, BoundAnalysisReport, TighteningRecommendation};

/// Estimate F32/F64 precision drift through `depth` chained InstanceNorm layers.
///
/// Runs a synthetic forward pass through `depth` InstanceNorm layers in both
/// F64 (reference) and **naive** F32 (uncompensated summation). Returns
/// `RMS(f32_output) / RMS(f64_output)` — a ratio of 1.0 means perfect
/// precision; lower values indicate amplitude attenuation from F32 rounding.
///
/// Uses naive F32 (not Kahan) to model worst-case drift for models without
/// compensation. If a model uses Kahan, actual drift will be better than
/// estimated — making this a conservative bound.
///
/// Uses 256 channels (matching Kokoro) with per-channel affine gamma/beta
/// parameters that vary in magnitude, stressing F32 summation of non-uniform
/// values across channels. This is more realistic than constant shift because
/// real InstanceNorm layers have learned per-channel parameters.
///
/// Returns 1.0 for `depth == 0`.
///
/// Part of #2705.
#[must_use]
pub fn estimate_norm_chain_precision_drift(depth: usize) -> f32 {
    if depth == 0 {
        return 1.0;
    }
    let channels = 256; // Kokoro Generator channel count
    let time = 256;
    let eps = 1e-5;
    let n = channels * time;

    // Per-channel affine parameters: gamma ∈ [0.5, 2.5], beta ∈ [-1000, +1000].
    // Varying magnitudes stress F32 summation of non-uniform values.
    let gammas: Vec<f64> = (0..channels)
        .map(|c| 0.5 + 2.0 * ((c as f64 * 0.7).sin() * 0.5 + 0.5))
        .collect();
    let betas: Vec<f64> = (0..channels)
        .map(|c| 1000.0 * (c as f64 * 0.3).cos())
        .collect();

    // Seed with deterministic sinusoidal pattern.
    let seed: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.137).sin() * (1.0 + (i as f64 * 0.03)))
        .collect();

    let mut f64_data = seed.clone();
    let mut f32_data: Vec<f32> = seed.iter().map(|&x| x as f32).collect();

    for _ in 0..depth {
        instance_norm_f64(&mut f64_data, channels, time, eps);
        instance_norm_f32_naive(&mut f32_data, channels, time, eps as f32);
        // Per-channel affine: gamma[c] * x + beta[c]
        for c in 0..channels {
            let start = c * time;
            let end = start + time;
            let g = gammas[c];
            let b = betas[c];
            for t in start..end {
                f64_data[t] = f64_data[t] * g + b;
                f32_data[t] = f32_data[t] * (g as f32) + (b as f32);
            }
        }
    }

    let f64_rms = (f64_data.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
    let f32_rms = (f32_data
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        / n as f64)
        .sqrt();

    if f64_rms > 1e-10 {
        (f32_rms / f64_rms) as f32
    } else {
        1.0
    }
}

fn instance_norm_f64(data: &mut [f64], channels: usize, time: usize, eps: f64) {
    for c in 0..channels {
        let start = c * time;
        let end = start + time;
        let mean: f64 = data[start..end].iter().sum::<f64>() / time as f64;
        let var: f64 = data[start..end]
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
            / time as f64;
        let inv = 1.0 / (var + eps).sqrt();
        for item in &mut data[start..end] {
            *item = (*item - mean) * inv;
        }
    }
}

fn instance_norm_f32_naive(data: &mut [f32], channels: usize, time: usize, eps: f32) {
    for c in 0..channels {
        let start = c * time;
        let end = start + time;
        let mean: f32 = data[start..end].iter().sum::<f32>() / time as f32;
        let var: f32 = data[start..end]
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>()
            / time as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for item in &mut data[start..end] {
            *item = (*item - mean) * inv;
        }
    }
}

impl BoundAnalysisReport {
    /// Auto-estimate and populate F32/F64 precision drift from `chained_norm_depth`.
    ///
    /// Calls [`estimate_norm_chain_precision_drift`] with the report's
    /// `chained_norm_depth`, then [`set_precision_drift`] with the result.
    /// No-op if `chained_norm_depth == 0`.
    pub fn estimate_and_set_precision_drift(&mut self, config: &AnalysisConfig) {
        if self.chained_norm_depth > 0 {
            let ratio = estimate_norm_chain_precision_drift(self.chained_norm_depth);
            self.set_precision_drift(ratio, config);
        }
    }

    /// Populate F32 vs F64 precision drift from a measured ratio.
    pub fn set_precision_drift(&mut self, ratio: f32, config: &AnalysisConfig) {
        if !ratio.is_finite() {
            return;
        }
        self.precision_drift_ratio = Some(ratio);

        let depth = self.chained_norm_depth;
        self.drift_per_layer = if depth > 0 && ratio > 0.0 {
            // drift_per_layer = 1 - ratio^(1/depth)
            let per_layer = 1.0 - ratio.powf(1.0 / depth as f32);
            Some(per_layer)
        } else {
            Some(0.0)
        };

        // Flag PRECISION_RISK when depth exceeds threshold AND ratio is below threshold.
        if depth >= config.precision_risk_depth_threshold
            && ratio < config.precision_risk_drift_threshold
        {
            self.recommendations
                .push(TighteningRecommendation::PrecisionRisk {
                    chained_norm_depth: depth,
                    precision_drift_ratio: ratio,
                    drift_per_layer: self.drift_per_layer.unwrap_or(0.0),
                });
        }
    }
}
