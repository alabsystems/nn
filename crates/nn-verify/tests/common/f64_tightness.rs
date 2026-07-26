// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! f64 tightness validation helpers for compose tests.
//!
//! Measures the precision gap between f32 IBP/CROWN bounds and f64 concrete
//! evaluation. For applicable sequential subgraphs (Linear+ReLU), this provides
//! a ground-truth reference: f64 evaluation at boundary input points gives the
//! actual output range, and the gap between that and f32 bounds quantifies the
//! overapproximation from both bound propagation and f32 rounding.
//!
//! Part of #4316: f64 evaluation for bound tightness.

use nn_verify::{
    convert_network_to_f64, evaluate_network_f64, BoundedTensor, Network, SequentialLayerF64,
};
use ndarray::ArrayD;

/// Result of an f64 tightness measurement.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct F64TightnessResult {
    /// f32 IBP lower bounds.
    pub f32_ibp_lower: Vec<f32>,
    /// f32 IBP upper bounds.
    pub f32_ibp_upper: Vec<f32>,
    /// f64 concrete evaluation at the lower input corner.
    pub f64_at_lower: Vec<f64>,
    /// f64 concrete evaluation at the upper input corner.
    pub f64_at_upper: Vec<f64>,
    /// f64 concrete evaluation at the midpoint.
    pub f64_at_mid: Vec<f64>,
    /// Observed f64 output range: (min across all eval points, max across all eval points).
    pub f64_range: Vec<(f64, f64)>,
    /// Per-element gap: f32 IBP width minus f64 observed range.
    /// Positive means f32 is wider (expected). Negative would indicate a soundness issue.
    pub gap: Vec<f64>,
    /// Mean gap across all output elements.
    pub mean_gap: f64,
    /// Max gap across all output elements.
    pub max_gap: f64,
}

/// Measure the precision gap between f32 IBP bounds and f64 concrete evaluation.
///
/// Builds a sequential `Network`, runs f32 IBP to get bounds, then evaluates
/// the same network in f64 at the input lower corner, upper corner, and midpoint.
/// Returns the per-element gap between the f32 bound width and the f64 observed range.
///
/// Requirements:
/// - `network` must be a sequential network (Linear+ReLU only for f64 path).
/// - `input` must be a `BoundedTensor` with matching input dimension.
///
/// Panics if f64 conversion fails (unsupported layer types).
#[allow(dead_code)]
pub(crate) fn measure_f64_tightness(
    network: &Network,
    input: &BoundedTensor,
) -> F64TightnessResult {
    // Step 1: f32 IBP
    let ibp_output = network.propagate_ibp(input).expect("f32 IBP propagation");
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
    let f32_ibp_lower: Vec<f32> = ibp_lo.iter().copied().collect();
    let f32_ibp_upper: Vec<f32> = ibp_hi.iter().copied().collect();

    // Step 2: Convert to f64 layers
    let f64_layers: Vec<SequentialLayerF64> =
        convert_network_to_f64(network.layers()).expect("f64 conversion");

    // Step 3: Evaluate at corner and midpoint inputs in f64
    let (in_lo, in_hi) = input.lower_upper();
    let lower_f64: ArrayD<f64> = in_lo.mapv(f64::from);
    let upper_f64: ArrayD<f64> = in_hi.mapv(f64::from);
    let mid_f64: ArrayD<f64> = (&lower_f64 + &upper_f64) / 2.0;

    let out_at_lower =
        evaluate_network_f64(&f64_layers, &lower_f64).expect("f64 eval at lower corner");
    let out_at_upper =
        evaluate_network_f64(&f64_layers, &upper_f64).expect("f64 eval at upper corner");
    let out_at_mid = evaluate_network_f64(&f64_layers, &mid_f64).expect("f64 eval at midpoint");

    let f64_at_lower: Vec<f64> = out_at_lower.iter().copied().collect();
    let f64_at_upper: Vec<f64> = out_at_upper.iter().copied().collect();
    let f64_at_mid: Vec<f64> = out_at_mid.iter().copied().collect();

    // Step 4: Compute observed f64 range and gap
    let n = f32_ibp_lower.len();
    let mut f64_range = Vec::with_capacity(n);
    let mut gap = Vec::with_capacity(n);

    for i in 0..n {
        let obs_min = f64_at_lower[i].min(f64_at_upper[i]).min(f64_at_mid[i]);
        let obs_max = f64_at_lower[i].max(f64_at_upper[i]).max(f64_at_mid[i]);
        f64_range.push((obs_min, obs_max));

        let f32_width = f64::from(f32_ibp_upper[i] - f32_ibp_lower[i]);
        let f64_obs_width = obs_max - obs_min;
        gap.push(f32_width - f64_obs_width);
    }

    let mean_gap = if n > 0 {
        gap.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };
    let max_gap = gap.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    F64TightnessResult {
        f32_ibp_lower,
        f32_ibp_upper,
        f64_at_lower,
        f64_at_upper,
        f64_at_mid,
        f64_range,
        gap,
        mean_gap,
        max_gap,
    }
}

/// Assert that f32 IBP bounds contain all f64 concrete evaluations (soundness).
///
/// This verifies that the f32 overapproximation does not miss any actual
/// network output — every f64 evaluation at boundary points must lie within
/// the f32 IBP bounds (with tolerance for f32/f64 rounding).
#[allow(dead_code)]
pub(crate) fn assert_f64_contained_in_f32_bounds(result: &F64TightnessResult) {
    let eps = 1e-4_f64; // tolerance for f32/f64 rounding differences

    for (i, &(obs_min, obs_max)) in result.f64_range.iter().enumerate() {
        let ibp_lo = f64::from(result.f32_ibp_lower[i]);
        let ibp_hi = f64::from(result.f32_ibp_upper[i]);

        assert!(
            obs_min >= ibp_lo - eps,
            "Soundness: f64 output[{i}] min {obs_min} below f32 IBP lower {ibp_lo} \
             (gap={}, eps={eps})",
            ibp_lo - obs_min
        );
        assert!(
            obs_max <= ibp_hi + eps,
            "Soundness: f64 output[{i}] max {obs_max} above f32 IBP upper {ibp_hi} \
             (gap={}, eps={eps})",
            obs_max - ibp_hi
        );
    }
}

/// Log precision gap metrics for a tightness result.
///
/// Emits structured output for compose test `--nocapture` runs, documenting
/// the f32-vs-f64 precision gap for the subgraph.
#[allow(dead_code)]
pub(crate) fn log_precision_gap(label: &str, result: &F64TightnessResult) {
    let n = result.gap.len();
    let f32_widths: Vec<f64> = result
        .f32_ibp_lower
        .iter()
        .zip(&result.f32_ibp_upper)
        .map(|(&lo, &hi)| f64::from(hi - lo))
        .collect();
    let f64_obs_widths: Vec<f64> = result.f64_range.iter().map(|&(lo, hi)| hi - lo).collect();

    let avg_f32_width: f64 = f32_widths.iter().sum::<f64>() / n as f64;
    let avg_f64_width: f64 = f64_obs_widths.iter().sum::<f64>() / n as f64;
    let tightness_ratio = if avg_f64_width > 0.0 {
        avg_f32_width / avg_f64_width
    } else {
        f64::INFINITY
    };

    eprintln!("=== f64 Tightness Report: {label} ===");
    eprintln!("  Output dimension: {n}");
    eprintln!("  Avg f32 IBP width:     {avg_f32_width:.6}");
    eprintln!("  Avg f64 observed width: {avg_f64_width:.6}");
    eprintln!("  Overapprox ratio:      {tightness_ratio:.2}x");
    eprintln!("  Mean gap:              {:.6}", result.mean_gap);
    eprintln!("  Max gap:               {:.6}", result.max_gap);
    eprintln!("===");
}
