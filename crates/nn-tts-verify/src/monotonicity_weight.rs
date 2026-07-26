// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight magnitude validation for attention monotonicity proofs.
//!
//! Validates that actual model weights satisfy the magnitude assumptions
//! required by the Phase 44-46 attention monotonicity proof.

/// Certificate validating that model weights satisfy the magnitude assumptions
/// required by the attention monotonicity proof.
///
/// Phase 44-46 of the attention monotonicity work proved that IBP bounds are
/// tight enough to prove diagonal dominance when weight magnitudes are small
/// (trained model magnitudes ≈ 0.001-0.005). Specifically:
///
/// - With uniform weight magnitude `mag` and symmetric input `[-ib, ib]`,
///   the IBP lower bound for a linear layer with `D` inputs is `-D * mag * ib`.
/// - With Xavier/trained `mag ≈ 1/sqrt(D)`, the bound is `-sqrt(D) * ib`.
/// - Provability requires `D * mag * ib ≤ margin_budget`.
///
/// This certificate validates that actual loaded weights satisfy the magnitude
/// assumptions, bridging the gap between synthetic-weight proofs and production.
#[derive(Debug, Clone)]
pub struct WeightMagnitudeCertificate {
    /// Per-layer maximum absolute weight values.
    pub per_layer_max_abs: Vec<f64>,
    /// Per-layer names (for diagnostics).
    pub layer_names: Vec<String>,
    /// Model dimension (D) used in scaling analysis.
    pub d_model: usize,
    /// Maximum allowed per-element magnitude (the proof assumption).
    pub magnitude_bound: f64,
    /// Whether ALL layers satisfy `max_abs <= magnitude_bound`.
    pub all_within_bound: bool,
    /// Number of layers that exceed the bound.
    pub violating_layers: usize,
    /// The Xavier-normalized effective magnitude: `max_abs * sqrt(fan_in)`.
    /// If this is ≤ 1.0 for all layers, IBP provability follows from Phase 44.
    pub max_normalized_magnitude: f64,
}

/// Validate weight magnitudes against the monotonicity proof assumptions.
///
/// For each weight tensor, computes the maximum absolute element value and
/// checks it against the given `magnitude_bound`. This connects the Phase 44-46
/// proof (which assumes controlled weight magnitudes) to actual model weights.
///
/// # Arguments
///
/// * `weights` — per-layer weight data as flat `f32` slices
/// * `layer_names` — human-readable names for each layer (must match `weights` length)
/// * `fan_ins` — input dimensionality for each layer (for Xavier normalization)
/// * `d_model` — model dimension for scaling analysis
/// * `magnitude_bound` — maximum allowed per-element weight magnitude
///
/// # Errors
///
/// Returns [`TtsVerifyError::DimensionMismatch`] if slice counts don't match.
pub fn validate_weight_magnitudes(
    weights: &[&[f32]],
    layer_names: &[&str],
    fan_ins: &[usize],
    d_model: usize,
    magnitude_bound: f64,
) -> Result<WeightMagnitudeCertificate, crate::error::TtsVerifyError> {
    if weights.len() != layer_names.len() {
        return Err(crate::error::TtsVerifyError::DimensionMismatch {
            expected: weights.len(),
            actual: layer_names.len(),
            context: "layer_names vs weights count",
        });
    }
    if weights.len() != fan_ins.len() {
        return Err(crate::error::TtsVerifyError::DimensionMismatch {
            expected: weights.len(),
            actual: fan_ins.len(),
            context: "fan_ins vs weights count",
        });
    }

    let mut per_layer_max_abs = Vec::with_capacity(weights.len());
    let mut names = Vec::with_capacity(weights.len());
    let mut violating = 0;
    let mut max_normalized = 0.0_f64;

    for (i, (w, name)) in weights.iter().zip(layer_names.iter()).enumerate() {
        // Defense-in-depth: check for NaN/Inf in raw weight data.
        // f64::max swallows NaN (IEEE 754-2008 maxNum: max(x, NaN) = x),
        // so fold(0.0, f64::max) would silently skip NaN elements.
        // Check elements directly before computing the aggregate.
        if w.iter().any(|x| !x.is_finite()) {
            return Err(crate::error::TtsVerifyError::InvalidConfig(
                crate::error::InvalidConfigKind::NonFinite {
                    param: "weight values",
                },
            ));
        }

        let max_abs = w.iter().map(|x| f64::from(x.abs())).fold(0.0_f64, f64::max);

        per_layer_max_abs.push(max_abs);
        names.push(name.to_string());

        if max_abs > magnitude_bound {
            violating += 1;
        }

        // Xavier-normalized magnitude: max_abs * sqrt(fan_in).
        // If this is ≤ 1.0, the weight scaling matches Xavier initialization
        // and Phase 44's 1/sqrt(D) assumption holds.
        let fan_in = fan_ins[i].max(1) as f64;
        let normalized = max_abs * fan_in.sqrt();
        if normalized > max_normalized {
            max_normalized = normalized;
        }
    }

    let all_within = violating == 0;

    Ok(WeightMagnitudeCertificate {
        per_layer_max_abs,
        layer_names: names,
        d_model,
        magnitude_bound,
        all_within_bound: all_within,
        violating_layers: violating,
        max_normalized_magnitude: max_normalized,
    })
}

/// Compute the provable input bound given weight magnitudes.
///
/// Phase 44 formula: IBP lower bound for linear layer = `-D * mag * ib`.
/// The margin budget from positional encoding constant term = `PE_margin`.
/// Provability requires `D * mag * ib ≤ PE_margin`, so:
///
/// ```text
/// max_provable_ib = PE_margin / (D * max_weight_mag)
/// ```
///
/// Returns the maximum symmetric input bound `ib` for which diagonal
/// dominance is provable via IBP.
pub fn max_provable_input_bound(weight_cert: &WeightMagnitudeCertificate, pe_margin: f64) -> f64 {
    // Defense-in-depth: NaN pe_margin would silently produce NaN result.
    if !pe_margin.is_finite() || pe_margin < 0.0 {
        return 0.0; // Non-finite or negative margin → not provable.
    }

    let d = weight_cert.d_model.max(1) as f64;
    // Use the maximum per-element magnitude across all layers.
    let max_mag = crate::stats::fold_max_propagate_nan(
        weight_cert.per_layer_max_abs.iter().copied(),
        0.0_f64,
    );

    if max_mag <= 0.0 || d <= 0.0 {
        return f64::INFINITY; // Zero weights → trivially provable at any ib.
    }

    pe_margin / (d * max_mag)
}
