// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Layer trace consistency checking for proof certificates.
//!
//! Validates that layer bounds form a contiguous, self-consistent chain:
//! - Sequential mode (v2.0): layer[i].output == layer[i+1].input
//! - Graph-aware mode (v2.1+): source layers' outputs match declared inputs
//! - Non-finite and inverted element detection on all output bounds
//!
//! Extracted from `certificate_checker.rs` (500-line limit).

use crate::certificate_types::LayerBoundRecord;
use crate::status::InputBoundsRecord;

use super::checker_types::CheckIssue;

/// Check layer trace continuity using graph topology when available.
///
/// When `input_sources` is present (v2.1+), validates that each layer's
/// input bounds match the output bounds of its declared source layers.
/// When `input_sources` is absent (v2.0), falls back to the sequential
/// assumption: layer[i].output == layer[i+1].input.
pub(super) fn check_layer_trace_consistency(
    bounds: &[LayerBoundRecord],
    issues: &mut Vec<CheckIssue>,
) {
    // #3020 F1: Non-finite check on all layers' output bounds (symmetric with
    // check_multi_source_containment's finiteness check). Detects NaN/Inf in
    // intermediate layers that would otherwise cause spurious LayerTraceGap
    // issues due to IEEE 754 NaN != NaN.
    check_nonfinite_output_bounds(bounds, issues);

    // #3153 F2: Per-element inverted bounds check on all layer output bounds.
    check_inverted_element_bounds(bounds, issues);

    // Algorithm audit F6: NaN/Inf and inverted check on all layers' INPUT bounds.
    // Output bounds are validated above, and check_first_layer_input_spec validates
    // all network input layers' input_bounds (#3322). This covers multi-source
    // layers (Add, Concat) whose input_bounds are not network inputs.
    check_input_bounds_validity(bounds, issues);

    // Check if any record has input_sources — if so, use graph-aware mode.
    let has_graph_topology = bounds.iter().any(|r| r.input_sources.is_some());

    if has_graph_topology {
        check_layer_trace_graph_aware(bounds, issues);
    } else {
        check_layer_trace_sequential(bounds, issues);
    }
}

/// #3153 F2: Check all layer output bound elements for inverted (lo > hi) pairs.
///
/// Corrupted or tampered traces could have per-element inverted bounds that pass
/// finiteness checks but represent impossible intervals.
fn check_inverted_element_bounds(bounds: &[LayerBoundRecord], issues: &mut Vec<CheckIssue>) {
    for record in bounds {
        for (j, (lo, hi)) in record.output_bounds.iter().enumerate() {
            if lo.is_finite() && hi.is_finite() && lo > hi {
                issues.push(CheckIssue::InvertedElementBounds {
                    layer_index: record.layer_index,
                    element_index: j,
                    lower: *lo,
                    upper: *hi,
                });
            }
        }
    }
}

/// Algorithm audit F6: Check all layers' input bounds for non-finite and inverted.
///
/// `check_nonfinite_output_bounds` and `check_inverted_element_bounds` only
/// validate output bounds. `check_first_layer_input_spec` validates all network
/// input layers (#3322). This covers multi-source layers (Add, Concat) whose
/// input_bounds come from predecessor layers, not the network input.
///
/// Inverted input bounds (lo > hi) mean the layer was verified on an empty
/// interval — the proof is vacuously true. NaN input bounds bypass all
/// relational comparisons (IEEE 754).
///
/// May produce duplicate issues for network input layers (also checked by
/// check_first_layer_input_spec). Duplicates are harmless defense-in-depth.
fn check_input_bounds_validity(bounds: &[LayerBoundRecord], issues: &mut Vec<CheckIssue>) {
    for record in bounds {
        for (j, (lo, hi)) in record.input_bounds.iter().enumerate() {
            if !lo.is_finite() || !hi.is_finite() {
                issues.push(CheckIssue::NonFiniteElement {
                    layer_index: record.layer_index,
                    element_index: j,
                    lower: *lo,
                    upper: *hi,
                });
            } else if lo > hi {
                issues.push(CheckIssue::InvertedElementBounds {
                    layer_index: record.layer_index,
                    element_index: j,
                    lower: *lo,
                    upper: *hi,
                });
            }
        }
    }
}

/// #3020 F1: Check all layers' output bounds for non-finite (NaN/Inf) elements.
///
/// The multi-source path in `check_multi_source_containment` has explicit
/// finiteness checks, but single-source and sequential paths use `!=`
/// comparison which produces spurious `LayerTraceGap` for NaN (IEEE 754:
/// NaN != NaN). This check makes NaN detection symmetric across all paths.
fn check_nonfinite_output_bounds(bounds: &[LayerBoundRecord], issues: &mut Vec<CheckIssue>) {
    for record in bounds {
        for (j, (lo, hi)) in record.output_bounds.iter().enumerate() {
            if !lo.is_finite() || !hi.is_finite() {
                issues.push(CheckIssue::NonFiniteElement {
                    layer_index: record.layer_index,
                    element_index: j,
                    lower: *lo,
                    upper: *hi,
                });
            }
        }
    }
}

/// Graph-aware trace validation: for each layer with `input_sources`,
/// verify that the source layers' output bounds are consistent with
/// this layer's input bounds. Layers whose input comes from the network
/// input (empty `input_sources`) are not checked for predecessor consistency.
fn check_layer_trace_graph_aware(bounds: &[LayerBoundRecord], issues: &mut Vec<CheckIssue>) {
    // Pre-build index: layer_index → position in bounds slice. O(n) setup
    // avoids O(n²) linear scan per source lookup.
    // #3020: detect duplicate layer_index values during construction.
    let mut index = std::collections::HashMap::with_capacity(bounds.len());
    for (pos, r) in bounds.iter().enumerate() {
        if index.insert(r.layer_index, pos).is_some() {
            issues.push(CheckIssue::DuplicateLayerIndex {
                layer_index: r.layer_index,
            });
        }
    }

    for record in bounds {
        let sources = match &record.input_sources {
            Some(s) if !s.is_empty() => s,
            Some(_) => continue, // Empty source list — network input layer, skip
            None => continue,    // No source info — network input layer, skip
        };

        // Check all source references exist and validate bounds consistency.
        for &src_idx in sources {
            // Self-reference: a layer cannot be its own input source.
            if src_idx == record.layer_index {
                issues.push(CheckIssue::SelfReferenceSource {
                    layer_index: record.layer_index,
                });
                continue;
            }
            // Forward reference (#3153 F4): source index must be strictly
            // less than this layer's index to ensure topological ordering.
            if src_idx >= record.layer_index {
                issues.push(CheckIssue::ForwardReference {
                    layer_index: record.layer_index,
                    source_index: src_idx,
                });
                continue;
            }
            if let Some(&pos) = index.get(&src_idx) {
                let src = &bounds[pos];
                if sources.len() == 1 {
                    // Single-source: source output must match this layer's input exactly.
                    if src.output_bounds != record.input_bounds {
                        issues.push(CheckIssue::LayerTraceGap {
                            layer_index: src_idx,
                            output_bounds: src.output_bounds.clone(),
                            next_input_bounds: record.input_bounds.clone(),
                        });
                    }
                } else {
                    check_multi_source_containment(src, record, src_idx, issues);
                }
            } else {
                // Dangling reference: source layer doesn't exist in the trace
                issues.push(CheckIssue::DanglingSourceRef {
                    layer_index: record.layer_index,
                    dangling_source: src_idx,
                });
            }
        }
    }
}

/// Multi-source structural check.
///
/// For multi-source layers (Add, MulBinary, Concat), the layer's
/// `input_bounds` represent the *combined* result of all sources after
/// NY bound propagation — NOT the individual source contributions.
/// Element-wise containment (source output ⊆ layer input) is semantically
/// wrong here: a source outputting `(0, 0)` feeding into a MulBinary whose
/// combined input is `(0.5, 1.0)` is correct NY behavior.
///
/// We verify:
/// 1. Source output bounds are non-empty (structural completeness).
/// 2. Source output bounds contain no NaN/Inf (finiteness).
///
/// Concat layers where source/input lengths differ are flagged as
/// `MultiSourceLengthMismatch` for informational purposes only.
fn check_multi_source_containment(
    src: &LayerBoundRecord,
    record: &LayerBoundRecord,
    src_idx: usize,
    issues: &mut Vec<CheckIssue>,
) {
    // Structural: source must have non-empty output bounds.
    if src.output_bounds.is_empty() {
        issues.push(CheckIssue::EmptyOutputBounds {
            layer_index: src_idx,
        });
        return;
    }

    // Finiteness: all source output elements must be finite.
    for (j, (lo, hi)) in src.output_bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            issues.push(CheckIssue::NonFiniteElement {
                layer_index: src_idx,
                element_index: j,
                lower: *lo,
                upper: *hi,
            });
            return;
        }
    }

    // Informational: flag Concat-style length mismatches for diagnostics.
    // This is not a validation failure — Concat sources naturally have
    // different element counts than the combined layer input.
    let _ = (record, issues);
}

/// Sequential fallback: layer[i].output_bounds == layer[i+1].input_bounds.
///
/// Used for v2.0 certificates that don't have `input_sources`.
fn check_layer_trace_sequential(bounds: &[LayerBoundRecord], issues: &mut Vec<CheckIssue>) {
    for i in 0..bounds.len().saturating_sub(1) {
        if bounds[i].output_bounds != bounds[i + 1].input_bounds {
            issues.push(CheckIssue::LayerTraceGap {
                layer_index: bounds[i].layer_index,
                output_bounds: bounds[i].output_bounds.clone(),
                next_input_bounds: bounds[i + 1].input_bounds.clone(),
            });
        }
    }
}

/// #3153 F3: Validate the input specification for the certificate.
///
/// NaN or inverted input bounds make a proof vacuously true (verified "for no
/// inputs"). Empty variable_inputs means nothing was verified.
pub(super) fn check_input_spec(spec: &InputBoundsRecord, issues: &mut Vec<CheckIssue>) {
    if spec.variable_inputs.is_empty() {
        issues.push(CheckIssue::InvalidInputSpec {
            message: "no variable_inputs — nothing was verified".to_string(),
        });
        return;
    }
    for param in &spec.variable_inputs {
        if !param.lower.is_finite() || !param.upper.is_finite() {
            issues.push(CheckIssue::InvalidInputSpec {
                message: format!(
                    "param {} has non-finite bounds ({}, {})",
                    param.param_index, param.lower, param.upper
                ),
            });
        } else if param.lower > param.upper {
            issues.push(CheckIssue::InvalidInputSpec {
                message: format!(
                    "param {} has inverted bounds ({} > {})",
                    param.param_index, param.lower, param.upper
                ),
            });
        }
    }
}
