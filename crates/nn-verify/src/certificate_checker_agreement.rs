// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Output agreement checking for proof certificates.
//!
//! Extracted from `certificate_checker.rs` per 500-line limit.

use super::CheckIssue;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::status::InputBoundsRecord;

/// Check that the last layer's output agrees with the certificate's output_bounds.
pub(super) fn check_output_agreement(
    cert: &ProofCertificate,
    bounds: &[LayerBoundRecord],
    issues: &mut Vec<CheckIssue>,
) {
    if let Some(last) = bounds.last() {
        // F4 (#1692): Empty output_bounds must not silently pass — a layer
        // with no output elements cannot validate the certificate's claims.
        if last.output_bounds.is_empty() {
            issues.push(CheckIssue::EmptyOutputBounds {
                layer_index: last.layer_index,
            });
            return;
        }

        // F1 (#1692): Check individual elements for NaN/non-finite BEFORE
        // reducing. IEEE 754: `NaN < x` is false, so the reduce
        // `if a < b { a } else { b }` silently drops NaN values among valid
        // elements. Checking each element first ensures NaN is never lost.
        for (i, (lo, hi)) in last.output_bounds.iter().enumerate() {
            if !lo.is_finite() || !hi.is_finite() {
                issues.push(CheckIssue::NanOutputBounds);
                // Report the specific element index for diagnostic purposes
                // if there are other valid elements.
                if last.output_bounds.len() > 1 {
                    issues.push(CheckIssue::NonFiniteElement {
                        layer_index: last.layer_index,
                        element_index: i,
                        lower: *lo,
                        upper: *hi,
                    });
                }
                return;
            }
        }

        // Safe to reduce: all elements are finite.
        let trace_lower = last
            .output_bounds
            .iter()
            .map(|(lo, _)| *lo)
            .fold(f32::INFINITY, f32::min);
        let trace_upper = last
            .output_bounds
            .iter()
            .map(|(_, hi)| *hi)
            .fold(f32::NEG_INFINITY, f32::max);

        let cert_lower = cert.output_bounds.lower;
        let cert_upper = cert.output_bounds.upper;

        // Guard: certificate-level bounds must also be finite.
        if !cert_lower.is_finite() || !cert_upper.is_finite() {
            issues.push(CheckIssue::NanOutputBounds);
            return;
        }

        // Allow small floating-point tolerance.
        let eps = 1e-6;
        if (cert_lower - trace_lower).abs() > eps || (cert_upper - trace_upper).abs() > eps {
            issues.push(CheckIssue::OutputMismatch {
                certificate_lower: cert_lower,
                certificate_upper: cert_upper,
                trace_lower,
                trace_upper,
            });
        }
    }
}

/// #3020, #3322: Validate that ALL network input layers' input_bounds are
/// consistent with the certificate's input_spec.
///
/// The trace continuity checks verify layer[i].output == layer[i+1].input, but
/// never anchor network input layers' bounds against the declared input range.
/// A forged certificate could claim arbitrary input_bounds while having a
/// valid-looking input_spec.
///
/// For multi-input models (e.g., ProsodyPredictor with bert_output + style),
/// spec.variable_inputs is a flat list covering ALL inputs. We consume spec
/// elements in layer_index order, matching each network input layer's element
/// count against the corresponding slice of spec_bounds.
pub(super) fn check_first_layer_input_spec(
    spec: &InputBoundsRecord,
    bounds: &[LayerBoundRecord],
    issues: &mut Vec<CheckIssue>,
) {
    // Collect ALL network input layers, sorted by layer_index (the iteration
    // order of bounds preserves insertion order, but sort for safety).
    let mut input_layers: Vec<&LayerBoundRecord> = bounds
        .iter()
        .filter(|r| {
            matches!(&r.input_sources, Some(s) if s.is_empty()) || r.input_sources.is_none()
        })
        .collect();
    input_layers.sort_by_key(|r| r.layer_index);

    if input_layers.is_empty() {
        return; // No network input layer found — other checks will catch this.
    }

    // F4/F5: check ALL network input layers' input_bounds for NaN/Inf/inverted.
    for layer in &input_layers {
        for (j, (lo, hi)) in layer.input_bounds.iter().enumerate() {
            if !lo.is_finite() || !hi.is_finite() {
                issues.push(CheckIssue::NonFiniteElement {
                    layer_index: layer.layer_index,
                    element_index: j,
                    lower: *lo,
                    upper: *hi,
                });
            } else if lo > hi {
                issues.push(CheckIssue::InvertedElementBounds {
                    layer_index: layer.layer_index,
                    element_index: j,
                    lower: *lo,
                    upper: *hi,
                });
            }
        }
    }

    // Build expected bounds from input_spec variable_inputs.
    let spec_bounds: Vec<(f32, f32)> = spec
        .variable_inputs
        .iter()
        .map(|p| (p.lower, p.upper))
        .collect();

    if spec_bounds.is_empty() {
        return; // check_input_spec already flags empty variable_inputs.
    }

    // Total spec elements should equal the sum of all input layer elements.
    // If not, we may have broadcast semantics — compare what we can.
    let total_input_elements: usize = input_layers.iter().map(|l| l.input_bounds.len()).sum();

    if spec_bounds.len() != total_input_elements {
        // Element count mismatch (broadcast semantics?) — can't compare slices.
        return;
    }

    // Consume spec elements in layer_index order, comparing each input layer's
    // bounds against its corresponding slice.
    let mut offset = 0;
    for layer in &input_layers {
        let n = layer.input_bounds.len();
        let expected = &spec_bounds[offset..offset + n];
        if layer.input_bounds != expected {
            issues.push(CheckIssue::InputBoundsSpecMismatch {
                layer_index: layer.layer_index,
                spec_bounds: expected.to_vec(),
                layer_input_bounds: layer.input_bounds.clone(),
            });
        }
        offset += n;
    }
}
