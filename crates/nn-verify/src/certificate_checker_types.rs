// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Types for certificate checking results.
//!
//! Extracted from `certificate_checker.rs` for the 500-line limit.

/// Result of checking a single certificate.
#[derive(Debug)]
#[non_exhaustive]
pub struct CheckResult {
    /// Kernel name from the certificate.
    pub kernel_name: String,
    /// List of issues found. Empty means the certificate passed all checks.
    pub issues: Vec<CheckIssue>,
    /// Bound quality assessment (None if no layer bounds present).
    pub vacuity: Option<VacuityAssessment>,
}

/// Certificate bound quality assessment.
///
/// A certificate is **non-vacuous** when at least half of layers used CROWN
/// propagation (not IBP fallback) and the output interval width is practically
/// meaningful (below a configurable threshold).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VacuityAssessment {
    /// Fraction of layers using CROWN bounds (0.0 = all IBP, 1.0 = all CROWN).
    pub crown_coverage: f32,
    /// Total number of layers in the certificate trace.
    pub total_layers: usize,
    /// Number of layers that used CROWN propagation.
    pub crown_layers: usize,
    /// Number of layers that fell back to IBP.
    pub ibp_layers: usize,
    /// Output interval width from the certificate.
    pub output_width: f32,
    /// Whether the certificate is considered non-vacuous.
    pub is_non_vacuous: bool,
}

impl CheckResult {
    /// Whether the certificate passed all checks.
    ///
    /// `VacuousBounds` is informational and does NOT cause this to return `false`.
    /// `SmtProofMissing` is NOT informational — a "Proven" claim without an
    /// Alethe proof artifact is unverifiable and fails validation (#3221).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.issues
            .iter()
            .all(|i| matches!(i, CheckIssue::VacuousBounds { .. }))
    }
}

/// A specific issue found during certificate checking.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CheckIssue {
    /// Structural validation failed.
    StructuralError {
        /// Human-readable validation failure message.
        message: String,
    },
    /// Layer trace has a gap: layer i's output != layer i+1's input.
    LayerTraceGap {
        layer_index: usize,
        output_bounds: Vec<(f32, f32)>,
        next_input_bounds: Vec<(f32, f32)>,
    },
    /// Final layer output bounds don't match certificate's claimed output.
    OutputMismatch {
        certificate_lower: f32,
        certificate_upper: f32,
        trace_lower: f32,
        trace_upper: f32,
    },
    /// Weight hash doesn't match the file on disk.
    WeightHashMismatch { expected: String, actual: String },
    /// Source hash doesn't match the file on disk.
    SourceHashMismatch { expected: String, actual: String },
    /// File referenced by hash could not be read.
    HashFileError { field: String, error: String },
    /// No layer bounds present (cannot verify trace).
    NoLayerBounds,
    /// Certificate is missing a required hash (weight or source).
    MissingHash {
        /// Which hash field is absent: "weight_hash" or "source_hash".
        field: String,
    },
    /// Layer references a non-existent source layer index.
    DanglingSourceRef {
        layer_index: usize,
        dangling_source: usize,
    },
    /// Output bounds contain non-finite (NaN or Inf) values.
    NanOutputBounds,
    /// Last layer has empty output_bounds — cannot verify agreement (#1692 F4).
    EmptyOutputBounds { layer_index: usize },
    /// A specific output bound element is non-finite (#1692 F1).
    NonFiniteElement {
        layer_index: usize,
        element_index: usize,
        lower: f32,
        upper: f32,
    },
    /// Multi-source layer has mismatched element counts (#1692 F2).
    MultiSourceLengthMismatch {
        layer_index: usize,
        source_index: usize,
        source_len: usize,
        input_len: usize,
    },
    /// Layer declares itself as its own input source (cycle).
    SelfReferenceSource { layer_index: usize },
    /// Certificate bounds are vacuously wide (IBP-only or very wide output).
    /// Informational — does not cause `is_valid()` to return `false`.
    VacuousBounds {
        crown_coverage: f32,
        output_width: f32,
    },
    /// Certificate output_bounds.is_infeasible is true — the proof failed and
    /// (0.0, 0.0) are sentinel values, not verified bounds (#3153 F1).
    InfeasibleBounds,
    /// Per-element inverted bounds: lower > upper in a layer bound record (#3153 F2).
    InvertedElementBounds {
        layer_index: usize,
        element_index: usize,
        lower: f32,
        upper: f32,
    },
    /// Input specification has invalid bounds (NaN, inverted, or empty) (#3153 F3).
    InvalidInputSpec { message: String },
    /// Forward reference in graph topology: source index >= layer index (#3153 F4).
    ForwardReference {
        layer_index: usize,
        source_index: usize,
    },
    /// SMT outcome is Proven but no Alethe proof artifact attached (#3095).
    SmtProofMissing,
    /// SMT proof artifact is present but verdict is Invalid (#3095).
    SmtProofInvalid,
    /// Content hash does not match recomputed hash (#3222).
    ContentHashMismatch { expected: String, actual: String },
    /// HMAC signature verification failed (#3222).
    SignatureInvalid { message: String },
    /// Signing key error (invalid length, missing content hash for verification) (#3325).
    SignatureKeyError { message: String },
    /// Duplicate layer_index in layer bounds — HashMap silently drops earlier record (#3020).
    DuplicateLayerIndex { layer_index: usize },
    /// First layer's input_bounds are inconsistent with input_spec (#3020).
    InputBoundsSpecMismatch {
        layer_index: usize,
        spec_bounds: Vec<(f32, f32)>,
        layer_input_bounds: Vec<(f32, f32)>,
    },
}

impl std::fmt::Display for CheckIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StructuralError { message } => write!(f, "structural: {message}"),
            Self::LayerTraceGap {
                layer_index,
                output_bounds,
                next_input_bounds,
            } => write!(
                f,
                "layer trace gap at layer {layer_index}: output {output_bounds:?} != next input {next_input_bounds:?}"
            ),
            Self::OutputMismatch {
                certificate_lower,
                certificate_upper,
                trace_lower,
                trace_upper,
            } => write!(
                f,
                "output mismatch: certificate [{certificate_lower}, {certificate_upper}] vs trace [{trace_lower}, {trace_upper}]"
            ),
            Self::WeightHashMismatch { expected, actual } => {
                write!(f, "weight hash mismatch: expected {expected}, got {actual}")
            }
            Self::SourceHashMismatch { expected, actual } => {
                write!(f, "source hash mismatch: expected {expected}, got {actual}")
            }
            Self::HashFileError { field, error } => {
                write!(f, "{field} file error: {error}")
            }
            Self::NoLayerBounds => write!(f, "no layer bounds in certificate"),
            Self::MissingHash { field } => {
                write!(f, "missing {field} in certificate")
            }
            Self::DanglingSourceRef {
                layer_index,
                dangling_source,
            } => write!(
                f,
                "layer {layer_index} references non-existent source layer {dangling_source}"
            ),
            Self::NanOutputBounds => {
                write!(f, "output bounds contain non-finite values (NaN or Inf)")
            }
            Self::EmptyOutputBounds { layer_index } => {
                write!(f, "layer {layer_index} has empty output_bounds")
            }
            Self::NonFiniteElement {
                layer_index,
                element_index,
                lower,
                upper,
            } => write!(
                f,
                "layer {layer_index} element {element_index} has non-finite bounds ({lower}, {upper})"
            ),
            Self::MultiSourceLengthMismatch {
                layer_index,
                source_index,
                source_len,
                input_len,
            } => write!(
                f,
                "layer {layer_index}: source {source_index} has {source_len} elements but layer input has {input_len}"
            ),
            Self::SelfReferenceSource { layer_index } => write!(
                f,
                "layer {layer_index} declares itself as its own input source (cycle)"
            ),
            Self::VacuousBounds {
                crown_coverage,
                output_width,
            } => write!(
                f,
                "vacuous bounds: crown_coverage={crown_coverage:.2}, output_width={output_width:.2}"
            ),
            Self::InfeasibleBounds => write!(
                f,
                "certificate output_bounds.is_infeasible is true — proof failed, bounds are sentinel values"
            ),
            Self::InvertedElementBounds {
                layer_index,
                element_index,
                lower,
                upper,
            } => write!(
                f,
                "layer {layer_index} element {element_index} has inverted bounds ({lower} > {upper})"
            ),
            Self::InvalidInputSpec { message } => {
                write!(f, "invalid input_spec: {message}")
            }
            Self::ForwardReference {
                layer_index,
                source_index,
            } => write!(
                f,
                "layer {layer_index} references forward/equal source {source_index} (must be < layer_index)"
            ),
            Self::SmtProofMissing => write!(
                f,
                "SMT outcome is Proven but no Alethe proof artifact attached"
            ),
            Self::SmtProofInvalid => write!(
                f,
                "SMT proof artifact present but verdict is Invalid"
            ),
            Self::ContentHashMismatch { expected, actual } => write!(
                f,
                "content hash mismatch: stored {expected}, computed {actual}"
            ),
            Self::SignatureInvalid { message } => write!(
                f,
                "HMAC signature invalid: {message}"
            ),
            Self::SignatureKeyError { message } => write!(
                f,
                "signing key error: {message}"
            ),
            Self::DuplicateLayerIndex { layer_index } => write!(
                f,
                "duplicate layer_index {layer_index} in layer bounds (earlier record silently dropped)"
            ),
            Self::InputBoundsSpecMismatch {
                layer_index,
                spec_bounds,
                layer_input_bounds,
            } => write!(
                f,
                "layer {layer_index} input_bounds {layer_input_bounds:?} inconsistent with input_spec {spec_bounds:?}"
            ),
        }
    }
}
