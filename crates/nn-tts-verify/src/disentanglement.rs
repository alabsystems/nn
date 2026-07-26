// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compositional verification of prosody control disentanglement.
//!
//! Proves that TTS control knobs (rate, pitch, emotion, speaker, style) are
//! independent by computing CROWN sensitivity bounds. When varying one control
//! dimension produces small output bound widths on another dimension's acoustic
//! correlate, the two are formally disentangled.
//!
//! The key idea: output bound width when varying one input subspace is a
//! **CROWN-verified upper bound on the output range** (image diameter) for that
//! subspace. This is a valid sensitivity metric: small CROWN width ⇒ small
//! worst-case output variation under perturbation. Note: this is NOT a Jacobian
//! norm bound — CROWN width bounds the output range, which can be both larger
//! (over-approximation) or smaller (oscillating functions) than Jacobian-derived
//! Lipschitz bounds.
//!
//! # Usage
//!
//! ```text
//! // Define control dimensions and acoustic properties
//! let controls = vec![
//!     ControlDimension::new("prosody_style", 0, 0, 128),
//!     ControlDimension::new("decoder_style", 0, 128, 256),
//! ];
//! let properties = vec![
//!     AcousticProperty::new("f0", 0, 64),
//!     AcousticProperty::new("duration", 64, 80),
//! ];
//!
//! // Compute NxM sensitivity matrix via CROWN
//! let cert = verify_disentanglement(
//!     &graph, &controls, &properties,
//!     1.0,        // input_bound
//!     &midpoint,  // fixed operating point
//!     0.3,        // max 30% cross-influence
//! )?;
//! assert!(cert.is_disentangled);
//! ```
//!
//! Part of #1738: Compositional Verification of Prosody Controls.

use crate::error::{DspErrorKind, TtsVerifyError};
use nn_verify::{propagate_with_crown_fallback, BoundedTensor, GraphNetwork};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Defines one control dimension with its acoustic correlate.
///
/// Maps a named control (e.g., "prosody_style") to a contiguous slice within
/// the flat variable input vector used by NY graphs.
#[derive(Debug, Clone)]
pub struct ControlDimension {
    /// Human-readable name (e.g., "prosody_style", "decoder_style").
    pub name: String,
    /// Which variable input this maps to (index in multi-variable graph).
    /// For single-variable packed inputs, this is always 0.
    pub variable_index: usize,
    /// Start index within the variable (inclusive).
    pub slice_start: usize,
    /// End index within the variable (exclusive).
    pub slice_end: usize,
}

impl ControlDimension {
    /// Create a new control dimension.
    pub fn new(name: &str, variable_index: usize, slice_start: usize, slice_end: usize) -> Self {
        Self {
            name: name.to_string(),
            variable_index,
            slice_start,
            slice_end,
        }
    }

    /// Number of elements in this control dimension.
    pub fn dim(&self) -> usize {
        self.slice_end.saturating_sub(self.slice_start)
    }
}

/// Defines one acoustic property to measure disentanglement against.
///
/// Maps a named property (e.g., "F0", "duration") to a contiguous slice of the
/// graph output. The output bound width for this slice measures how much an
/// input perturbation affects this property.
#[derive(Debug, Clone)]
pub struct AcousticProperty {
    /// Human-readable name (e.g., "F0", "duration", "spectral_envelope").
    pub name: String,
    /// Start index in the output (inclusive).
    pub output_start: usize,
    /// End index in the output (exclusive).
    pub output_end: usize,
}

impl AcousticProperty {
    /// Create a new acoustic property.
    pub fn new(name: &str, output_start: usize, output_end: usize) -> Self {
        Self {
            name: name.to_string(),
            output_start,
            output_end,
        }
    }

    /// Number of output elements for this property.
    pub fn dim(&self) -> usize {
        self.output_end.saturating_sub(self.output_start)
    }
}

/// Result of a disentanglement test: how much does varying control C
/// affect acoustic property P?
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    /// Control dimension varied.
    pub control: String,
    /// Acoustic property measured.
    pub property: String,
    /// Mean output bound width when this control varies (others fixed).
    /// Smaller = less influence = more disentangled.
    pub bound_width: f64,
    /// Propagation mode used ("CROWN" or "IBP").
    pub propagation_mode: String,
}

/// Full disentanglement certificate: NxM matrix of sensitivities
/// for N control dimensions and M acoustic properties.
#[derive(Debug, Clone)]
pub struct DisentanglementCertificate {
    /// All sensitivity results (N controls × M properties).
    pub sensitivities: Vec<SensitivityResult>,
    /// Number of control dimensions.
    pub n_controls: usize,
    /// Number of acoustic properties.
    pub n_properties: usize,
    /// Input bound magnitude used for variable regions.
    pub input_bound: f64,
    /// Is the sensitivity matrix approximately diagonal? (disentangled)
    pub is_disentangled: bool,
    /// Maximum off-diagonal / on-diagonal ratio.
    /// Lower is better. 0.0 = perfectly disentangled.
    pub max_cross_influence: f64,
}

// ---------------------------------------------------------------------------
// Sensitivity measurement
// ---------------------------------------------------------------------------

/// Compute CROWN sensitivity of an acoustic property to a control dimension.
///
/// Methodology:
/// 1. Set ALL variable inputs to zero-width (constant at midpoint)
/// 2. Expand ONLY the target control dimension to `[mid - bound, mid + bound]`
/// 3. Run CROWN propagation
/// 4. Measure output bound width for the target acoustic property
///
/// The output bound width is an upper bound on the output range:
/// `sup f(x) - inf f(x) ≤ bound_width` for all x in the perturbed region.
/// Equivalently: `max|f(x) - f(y)| ≤ bound_width` for x, y in the region.
pub fn measure_sensitivity(
    graph: &GraphNetwork,
    control: &ControlDimension,
    property: &AcousticProperty,
    input_bound: f64,
    fixed_midpoint: &[f64],
) -> Result<SensitivityResult, TtsVerifyError> {
    if !input_bound.is_finite() || input_bound <= 0.0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "input_bound must be finite and positive",
        }));
    }

    let n = fixed_midpoint.len();
    if control.slice_end > n || control.slice_start >= control.slice_end {
        return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "control slice out of range for input size",
        }));
    }

    // Build input bounds: all elements at midpoint (zero-width) except the
    // target control, which gets [-bound, +bound] around the midpoint.
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);

    for (i, &mid) in fixed_midpoint.iter().enumerate() {
        let mid_f32 = mid as f32;
        if i >= control.slice_start && i < control.slice_end {
            // Expand this control dimension
            lower.push(mid_f32 - input_bound as f32);
            upper.push(mid_f32 + input_bound as f32);
        } else {
            // Fix at midpoint (zero-width)
            lower.push(mid_f32);
            upper.push(mid_f32);
        }
    }

    let lower_arr = ArrayD::from_shape_vec(IxDyn(&[n]), lower).map_err(|e| {
        TtsVerifyError::OperationFailed {
            context: "build lower bounds array",
            source: Box::new(e),
        }
    })?;
    let upper_arr = ArrayD::from_shape_vec(IxDyn(&[n]), upper).map_err(|e| {
        TtsVerifyError::OperationFailed {
            context: "build upper bounds array",
            source: Box::new(e),
        }
    })?;

    let input_bounds =
        BoundedTensor::new(lower_arr, upper_arr).map_err(|e| TtsVerifyError::OperationFailed {
            context: "create BoundedTensor",
            source: Box::new(e),
        })?;

    // Propagate through the graph
    let (method, output, _fallback_reason) = propagate_with_crown_fallback(graph, &input_bounds)
        .map_err(|e| TtsVerifyError::OperationFailed {
            context: "CROWN propagation",
            source: Box::new(e),
        })?;

    // Extract output bounds for the target property
    let (lo, hi) = output.lower_upper();
    let lo_slice = lo.as_slice().ok_or_else(|| {
        TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "output lower bounds not contiguous",
        })
    })?;
    let hi_slice = hi.as_slice().ok_or_else(|| {
        TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "output upper bounds not contiguous",
        })
    })?;

    if property.output_end > lo_slice.len() || property.output_start >= property.output_end {
        return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "property slice out of range for output size",
        }));
    }

    // Compute mean bound width for the target property
    let mut total_width = 0.0_f64;
    let count = property.output_end - property.output_start;
    for i in property.output_start..property.output_end {
        let width = f64::from(hi_slice[i] - lo_slice[i]);
        if !width.is_finite() {
            return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
                what: "non-finite bound width at output index",
            }));
        }
        total_width += width;
    }
    let mean_width = if count > 0 {
        total_width / count as f64
    } else {
        0.0
    };

    Ok(SensitivityResult {
        control: control.name.clone(),
        property: property.name.clone(),
        bound_width: mean_width,
        propagation_mode: format!("{method:?}"),
    })
}

// ---------------------------------------------------------------------------
// Disentanglement certificate
// ---------------------------------------------------------------------------

/// Compute the full NxM sensitivity matrix and assess disentanglement.
///
/// A model is "disentangled" if the sensitivity matrix is approximately
/// diagonal: each control primarily affects its designated property, with
/// bounded cross-influence.
///
/// # Arguments
///
/// * `graph` — NY GraphNetwork of the TTS submodel
/// * `controls` — N control dimensions to vary
/// * `properties` — M acoustic properties to measure
/// * `input_bound` — symmetric perturbation magnitude for each control
/// * `fixed_midpoint` — operating point for fixed controls
/// * `max_cross_influence_ratio` — maximum allowed off-diagonal/on-diagonal ratio
///
/// # Returns
///
/// A `DisentanglementCertificate` with the full NxM sensitivity matrix.
/// `is_disentangled` is true when the maximum cross-influence ratio is below
/// the threshold.
pub fn verify_disentanglement(
    graph: &GraphNetwork,
    controls: &[ControlDimension],
    properties: &[AcousticProperty],
    input_bound: f64,
    fixed_midpoint: &[f64],
    max_cross_influence_ratio: f64,
) -> Result<DisentanglementCertificate, TtsVerifyError> {
    if controls.is_empty() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::EmptyInput {
            what: "at least one control dimension required",
        }));
    }
    if properties.is_empty() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::EmptyInput {
            what: "at least one acoustic property required",
        }));
    }

    let n_controls = controls.len();
    let n_properties = properties.len();

    // Compute all NxM sensitivities
    let mut sensitivities = Vec::with_capacity(n_controls * n_properties);
    for control in controls {
        for property in properties {
            let result =
                measure_sensitivity(graph, control, property, input_bound, fixed_midpoint)?;
            sensitivities.push(result);
        }
    }

    // Compute cross-influence ratio.
    // For each control, find its "primary" property (highest sensitivity)
    // and then compute cross-influence = max(other_props) / primary.
    let mut max_cross = 0.0_f64;
    for (ci, _control) in controls.iter().enumerate() {
        let row_start = ci * n_properties;
        let row_end = row_start + n_properties;
        let row = &sensitivities[row_start..row_end];

        // Find primary index (property with highest sensitivity).
        // Use index-based exclusion, not value-based, to correctly handle
        // multiple properties with identical sensitivity values.
        let primary_idx = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.bound_width.total_cmp(&b.bound_width))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let primary_width = row[primary_idx].bound_width;

        if primary_width > 0.0 {
            // Cross-influence: max of non-primary sensitivities / primary
            for (j, s) in row.iter().enumerate() {
                if j != primary_idx {
                    let ratio = s.bound_width / primary_width;
                    if ratio > max_cross {
                        max_cross = ratio;
                    }
                }
            }
        }
    }

    Ok(DisentanglementCertificate {
        sensitivities,
        n_controls,
        n_properties,
        input_bound,
        is_disentangled: max_cross < max_cross_influence_ratio,
        max_cross_influence: max_cross,
    })
}

#[cfg(test)]
#[path = "disentanglement_tests.rs"]
mod tests;
