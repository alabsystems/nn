// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Types and utilities for v2 proof certificates.
//!
//! Extracted from `certificate.rs` to keep it under 500 lines.
//! Contains: `LayerBoundRecord`, `KaniProofRecord`, `KaniOutcome`,
//! and SHA-256 fingerprinting functions.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::VerifyError;
use crate::verify_types::PropMethod;

/// How F16/BF16 precision loss was modeled during verification.
///
/// Distinguishes "proved for F32 algorithm" from "proved for F16 execution."
/// The latter is strictly stronger: it accounts for overflow at F16 boundaries
/// (values > 65504 become Inf) and ULP rounding error at each dtype cast.
///
/// Default: `F32Only` for backward compatibility with pre-v5 certificates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[derive(Default)]
pub enum PrecisionModel {
    /// Verification assumed F32 computation throughout.
    /// The certificate proves the F32 algorithm, not the F16 execution.
    #[default]
    F32Only,
    /// Verification modeled F16/BF16 precision loss at dtype cast points.
    F16Aware {
        /// Number of F16/BF16 downcast points modeled in the graph.
        cast_count: usize,
        /// Total accumulated epsilon budget from all dtype casts.
        /// Sum of per-cast ULP widening applied to bounds.
        total_epsilon: f32,
    },
}

/// Per-layer bound propagation record for independent checking.
///
/// Records the input and output interval bounds at each layer of the network
/// during verification. An independent checker can re-derive the output bounds
/// from the input bounds using IBP (simple interval arithmetic) without needing
/// access to NY.
///
/// The `input_sources` field (v2.1) records the actual graph topology so the
/// checker can validate trace consistency for non-sequential (branching) graphs
/// without assuming layer[i] feeds directly into layer[i+1].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerBoundRecord {
    /// Index of this layer in the network graph (0-based).
    pub layer_index: usize,
    /// Layer type name (e.g. "Linear", "ReLU", "Conv1d", "Snake").
    pub layer_type: String,
    /// Per-element input bounds as (lower, upper) pairs.
    pub input_bounds: Vec<(f32, f32)>,
    /// Per-element output bounds as (lower, upper) pairs.
    pub output_bounds: Vec<(f32, f32)>,
    /// Propagation method used for this specific layer.
    pub method: PropMethod,
    /// Graph node name (e.g. "trace_5", "n3") for mapping back to the
    /// NY `GraphNetwork`.
    ///
    /// `None` for certificates generated before this field was added.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub node_name: Option<String>,
    /// Indices of layers whose outputs feed into this layer's input.
    ///
    /// `None` for v2.0 certificates (backward compatibility). When present,
    /// the checker validates that each source layer's output bounds are
    /// consistent with this layer's input bounds, replacing the fragile
    /// "layer[i].output == layer[i+1].input" sequential assumption.
    ///
    /// An empty list means the layer takes input directly from the network
    /// input (no predecessor layers in the graph).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub input_sources: Option<Vec<usize>>,
}

/// Kani formal verification outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum KaniOutcome {
    /// All harnesses passed verification.
    Passed,
    /// At least one harness failed.
    Failed,
    /// Harnesses exist but have not been run.
    NotRun,
    /// Verification timed out.
    Timeout,
}

/// Kani proof status for a kernel.
///
/// Records how many Kani harnesses exist for this kernel, whether they passed,
/// and what properties they verify. Populated from `kani_status.json` during
/// certificate generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KaniProofRecord {
    /// Number of Kani proof harnesses for this kernel.
    pub harness_count: usize,
    /// Overall Kani verification outcome.
    pub status: KaniOutcome,
    /// Properties verified by the harnesses (e.g. "no_overflow", "no_nan").
    pub properties: Vec<String>,
    /// CBMC version used for verification, if available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cbmc_version: Option<String>,
}

// ---------------------------------------------------------------------------
// SHA-256 fingerprinting
// ---------------------------------------------------------------------------

/// Compute the SHA-256 hex digest of a file at the given path.
///
/// Reads the file in 8 KiB chunks to avoid loading large weight files into memory.
///
/// # Errors
///
/// Returns `VerifyError::Io` if the file cannot be read.
pub fn compute_file_hash(path: &Path) -> Result<String, VerifyError> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(crate::signing_config::hex_encode(&hasher.finalize()))
}

/// Compute the SHA-256 hex digest of a byte slice.
///
/// Useful for hashing in-memory content (e.g. serialized weights, source strings).
#[must_use]
pub fn compute_bytes_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    crate::signing_config::hex_encode(&hasher.finalize())
}

/// Per-layer constructive proof record for independent bound recomputation.
///
/// Each record captures the input/output bounds and layer type for one layer
/// of the verified network. An auditor can recompute interval arithmetic
/// layer-by-layer to confirm the claimed bounds without running NY.
///
/// Part of #4315 (Wire NY proof certificates into certify pipeline).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstructiveLayerRecord {
    /// Layer index in the network (0-based).
    pub layer_index: usize,
    /// Layer type description (e.g., "Linear", "ReLU", "Linear+ReLU").
    pub layer_type: String,
    /// Per-element input lower bounds for this layer.
    pub input_lower: Vec<f32>,
    /// Per-element input upper bounds for this layer.
    pub input_upper: Vec<f32>,
    /// Per-element output lower bounds for this layer.
    pub output_lower: Vec<f32>,
    /// Per-element output upper bounds for this layer.
    pub output_upper: Vec<f32>,
}

/// Constructive proof certificate data from NY.
///
/// Contains machine-checkable proof artifacts generated by NY's
/// constructive proof pipeline. The IBP certificate can be independently
/// verified by recomputing interval arithmetic; the Lean4 export text
/// can be checked by the Lean4 proof checker.
///
/// Part of #4315 (Wire NY proof certificates into certify pipeline).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ConstructiveProofData {
    /// The method used to generate the constructive proof.
    pub method: ConstructiveProofMethod,
    /// Per-element output lower bounds from the constructive certificate.
    pub output_lower: Vec<f32>,
    /// Per-element output upper bounds from the constructive certificate.
    pub output_upper: Vec<f32>,
    /// Per-element input lower bounds used in the constructive certificate.
    pub input_lower: Vec<f32>,
    /// Per-element input upper bounds used in the constructive certificate.
    pub input_upper: Vec<f32>,
    /// Number of layers in the verified network.
    pub num_layers: usize,
    /// Whether the constructive certificate passed self-verification.
    /// `true` means `IbpCertificate::verify()` or equivalent succeeded.
    pub verified: bool,
    /// Optional Lean4 proof export text for machine-checkable verification.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub lean4_export: Option<String>,
    /// ISO 8601 timestamp of constructive proof generation.
    pub generated_at: String,
    /// Per-layer constructive proof records from `compose_crown_proofs()`.
    ///
    /// When present, each record contains the input/output bounds at one layer
    /// of the network. An auditor can recompute IBP layer-by-layer to confirm
    /// the end-to-end bounds independently.
    ///
    /// `None` for IBP-only certificates (method == `Ibp`) that lack per-layer
    /// granularity, or when the NY composition pipeline was not run.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub layer_proofs: Option<Vec<ConstructiveLayerRecord>>,
    /// Lean4 source from `CrownCompositionProofExport` for multi-layer
    /// composition soundness.
    ///
    /// When present, this is a self-contained Lean4 module that proves
    /// end-to-end bounds by chaining per-layer CROWN proofs via transitivity.
    /// An auditor can check this with the Lean4 proof checker independently.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub composition_lean4_source: Option<String>,
    /// Theorem name from the composition proof (e.g., "crown_composition_sound").
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub composition_theorem_name: Option<String>,
}

impl ConstructiveProofData {
    /// Create a new constructive proof data instance.
    #[must_use]
    pub fn new(
        method: ConstructiveProofMethod,
        output_lower: Vec<f32>,
        output_upper: Vec<f32>,
        input_lower: Vec<f32>,
        input_upper: Vec<f32>,
        num_layers: usize,
        verified: bool,
    ) -> Self {
        Self {
            method,
            output_lower,
            output_upper,
            input_lower,
            input_upper,
            num_layers,
            verified,
            lean4_export: None,
            generated_at: crate::certificate::now_iso8601(),
            layer_proofs: None,
            composition_lean4_source: None,
            composition_theorem_name: None,
        }
    }

    /// Attach Lean4 proof export text.
    #[must_use]
    pub fn with_lean4_export(mut self, lean4: String) -> Self {
        self.lean4_export = Some(lean4);
        self
    }

    /// Attach per-layer constructive proof records from NY composition.
    #[must_use]
    pub fn with_layer_proofs(mut self, proofs: Vec<ConstructiveLayerRecord>) -> Self {
        self.layer_proofs = Some(proofs);
        self
    }

    /// Attach a composition proof Lean4 source and theorem name.
    ///
    /// The source is a self-contained Lean4 module that proves end-to-end
    /// bounds by chaining per-layer CROWN proofs via transitivity.
    #[must_use]
    pub fn with_composition_proof(mut self, lean4_source: String, theorem_name: String) -> Self {
        self.composition_lean4_source = Some(lean4_source);
        self.composition_theorem_name = Some(theorem_name);
        self
    }

    /// Whether this constructive proof contains independently verifiable data.
    ///
    /// Returns `true` when the certificate was self-verified (recomputed bounds
    /// matched) AND contains either IBP layer data or Lean4 export text.
    #[must_use]
    pub fn is_machine_checkable(&self) -> bool {
        self.verified
            && (self.lean4_export.is_some()
                || self.composition_lean4_source.is_some()
                || !self.output_lower.is_empty())
    }

    /// Whether this constructive proof has a multi-layer composition certificate.
    #[must_use]
    pub fn has_composition_proof(&self) -> bool {
        self.composition_lean4_source.is_some()
    }

    /// Number of per-layer proof records, if available.
    #[must_use]
    pub fn layer_proof_count(&self) -> usize {
        self.layer_proofs.as_ref().map_or(0, Vec::len)
    }

    /// Serialize this constructive proof to a JSON string for deployment.
    ///
    /// The JSON can be stored alongside the model binary and later loaded
    /// by an auditor using [`ConstructiveProofData::from_json`].
    ///
    /// # Errors
    ///
    /// Returns `VerifyError::Serialization` if serialization fails.
    pub fn to_json(&self) -> Result<String, VerifyError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a constructive proof from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError::Serialization` if the JSON is malformed.
    pub fn from_json(json: &str) -> Result<Self, VerifyError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Save the constructive proof certificate to a JSON file.
    ///
    /// Uses atomic write semantics (write to .tmp, then rename) to prevent
    /// partial writes from corrupting the proof artifact. The file can be
    /// loaded later by [`ConstructiveProofData::load`] for independent
    /// auditor verification.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError::Serialization` if serialization fails, or
    /// `VerifyError::Io` if the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<(), VerifyError> {
        use std::io::Write;

        let json = self.to_json()?;
        let tmp_path = {
            let mut s = path.as_os_str().to_owned();
            s.push(".tmp");
            std::path::PathBuf::from(s)
        };
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp_path)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        drop(file);

        if let Err(e) = std::fs::rename(&tmp_path, path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(VerifyError::Io(e));
        }
        Ok(())
    }

    /// Load a constructive proof certificate from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError::Io` if the file cannot be read, or
    /// `VerifyError::Serialization` if the JSON is malformed.
    pub fn load(path: &Path) -> Result<Self, VerifyError> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_json(&contents)
    }

    /// Validate structural consistency of this constructive proof.
    ///
    /// Checks:
    /// - Input/output bounds have matching lengths (lower.len() == upper.len())
    /// - All bounds are finite (constructive proofs require finite bounds)
    /// - Output bounds are not inverted (lower <= upper) per element
    /// - Input bounds are not inverted (lower <= upper) per element
    /// - Layer proofs (if present) have consistent input/output dimensions
    ///
    /// # Errors
    ///
    /// Returns a descriptive error message on the first validation failure.
    pub fn validate(&self) -> Result<(), String> {
        // Check input bounds consistency.
        if self.input_lower.len() != self.input_upper.len() {
            return Err(format!(
                "input bounds length mismatch: lower={} vs upper={}",
                self.input_lower.len(),
                self.input_upper.len()
            ));
        }
        // Check output bounds consistency.
        if self.output_lower.len() != self.output_upper.len() {
            return Err(format!(
                "output bounds length mismatch: lower={} vs upper={}",
                self.output_lower.len(),
                self.output_upper.len()
            ));
        }
        // Check all input bounds are finite.
        for (i, (lo, hi)) in self.input_lower.iter().zip(&self.input_upper).enumerate() {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(format!(
                    "non-finite input bound at index {i}: lower={lo}, upper={hi}"
                ));
            }
            if *lo > *hi {
                return Err(format!(
                    "inverted input bound at index {i}: lower={lo} > upper={hi}"
                ));
            }
        }
        // Check all output bounds are finite and non-inverted.
        for (i, (lo, hi)) in self.output_lower.iter().zip(&self.output_upper).enumerate() {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(format!(
                    "non-finite output bound at index {i}: lower={lo}, upper={hi}"
                ));
            }
            if *lo > *hi {
                return Err(format!(
                    "inverted output bound at index {i}: lower={lo} > upper={hi}"
                ));
            }
        }
        // Validate layer proofs consistency.
        if let Some(ref layers) = self.layer_proofs {
            for (i, layer) in layers.iter().enumerate() {
                if layer.input_lower.len() != layer.input_upper.len() {
                    return Err(format!(
                        "layer[{i}] input bounds length mismatch: lower={} vs upper={}",
                        layer.input_lower.len(),
                        layer.input_upper.len()
                    ));
                }
                if layer.output_lower.len() != layer.output_upper.len() {
                    return Err(format!(
                        "layer[{i}] output bounds length mismatch: lower={} vs upper={}",
                        layer.output_lower.len(),
                        layer.output_upper.len()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Replay-verify this constructive proof by checking bound containment.
    ///
    /// Validates that the claimed output bounds are structurally consistent
    /// and that per-layer proofs (if present) form a valid chain where each
    /// layer's output bounds contain the next layer's input bounds.
    ///
    /// Returns `true` if the replay verification passes. Returns `false` if
    /// the proof has structural issues or bound chain inconsistencies.
    ///
    /// This does NOT re-run NY; it checks the proof's internal
    /// consistency. For full independent verification, use the Lean4 export.
    #[must_use]
    pub fn replay_verify(&self) -> bool {
        // Structural validation must pass first.
        if self.validate().is_err() {
            return false;
        }
        // If no layer proofs, we can only confirm structural validity.
        let layers = match &self.layer_proofs {
            Some(l) if !l.is_empty() => l,
            _ => return self.verified,
        };
        // Check bound chain: layer[i].output should contain layer[i+1].input
        // within floating-point tolerance.
        let eps = 1e-6_f32;
        for window in layers.windows(2) {
            let prev = &window[0];
            let next = &window[1];
            // The previous layer's output should cover the next layer's input.
            // We check: prev.output_lower[j] - eps <= next.input_lower[j]
            //           prev.output_upper[j] + eps >= next.input_upper[j]
            // But only when dimensions match (different layers may have
            // different sizes due to reshaping, pooling, etc.).
            if prev.output_lower.len() == next.input_lower.len() {
                for j in 0..prev.output_lower.len() {
                    if prev.output_lower[j] - eps > next.input_lower[j] {
                        return false;
                    }
                    if prev.output_upper[j] + eps < next.input_upper[j] {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Method used to generate the constructive proof certificate.
///
/// Per nn engineering rule (#3340): AlphaCrown, BetaCrown, and Analytical
/// are counted as tight methods alongside Crown. Use [`is_tight()`](Self::is_tight)
/// to classify certificate vacuity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ConstructiveProofMethod {
    /// IBP (Interval Bound Propagation) constructive certificate.
    /// Contains per-layer weight/bias data enabling independent recomputation.
    Ibp,
    /// CROWN linear relaxation constructive certificate.
    /// Contains linear bound matrices and Lean4 export.
    Crown,
    /// alpha-CROWN optimized linear relaxation constructive certificate.
    /// Tighter than CROWN for normalization layers (InstanceNorm, AdaIN).
    AlphaCrown,
    /// beta-CROWN branch-and-bound constructive certificate.
    /// Most precise: uses alpha-CROWN with branching on ReLU/activation splits.
    BetaCrown,
    /// Analytical closed-form bounds (e.g., linear transforms, known ranges).
    Analytical,
    /// IBP composition proof for multi-layer networks.
    IbpComposition,
    /// CROWN composition proof for multi-layer networks.
    /// Generated by `ny_propagate::proof_certificate::compose_crown_proofs()`.
    /// Contains per-layer CROWN proofs chained via transitivity with Lean4 export.
    CrownComposition,
    /// alpha-CROWN composition proof for multi-layer networks.
    AlphaCrownComposition,
    /// beta-CROWN composition proof for multi-layer networks.
    BetaCrownComposition,
}

impl ConstructiveProofMethod {
    /// Whether this method produces tight (non-vacuous) bounds.
    ///
    /// Per nn engineering rule (#3340): Crown, AlphaCrown, BetaCrown, and
    /// Analytical are tight methods. IBP alone may be vacuously wide.
    /// Composition variants inherit tightness from their base method.
    #[must_use]
    pub fn is_tight(self) -> bool {
        matches!(
            self,
            Self::Crown
                | Self::AlphaCrown
                | Self::BetaCrown
                | Self::Analytical
                | Self::CrownComposition
                | Self::AlphaCrownComposition
                | Self::BetaCrownComposition
        )
    }

    /// Whether this is a composition proof (multi-layer).
    #[must_use]
    pub fn is_composition(self) -> bool {
        matches!(
            self,
            Self::IbpComposition
                | Self::CrownComposition
                | Self::AlphaCrownComposition
                | Self::BetaCrownComposition
        )
    }

    /// Convert a [`PropMethod`] to the corresponding single-layer constructive
    /// proof method.
    ///
    /// [`PropMethod`]: crate::verify_types::PropMethod
    #[must_use]
    pub fn from_prop_method(method: PropMethod) -> Self {
        use crate::verify_types::PropMethod;
        match method {
            PropMethod::Ibp => Self::Ibp,
            PropMethod::Crown => Self::Crown,
            PropMethod::AlphaCrown => Self::AlphaCrown,
            PropMethod::BetaCrown => Self::BetaCrown,
            PropMethod::Analytical => Self::Analytical,
            PropMethod::MixedIbpCrown => Self::Crown,
            // Forward compat: unknown methods fall back to IBP (conservative).
            _ => Self::Ibp,
        }
    }

    /// Convert a [`PropMethod`] to the corresponding composition constructive
    /// proof method.
    ///
    /// [`PropMethod`]: crate::verify_types::PropMethod
    #[must_use]
    pub fn composition_from_prop_method(method: PropMethod) -> Self {
        use crate::verify_types::PropMethod;
        match method {
            PropMethod::Ibp => Self::IbpComposition,
            PropMethod::Crown | PropMethod::MixedIbpCrown => Self::CrownComposition,
            PropMethod::AlphaCrown => Self::AlphaCrownComposition,
            PropMethod::BetaCrown => Self::BetaCrownComposition,
            PropMethod::Analytical => Self::CrownComposition,
            // Forward compat: unknown methods fall back to IBP composition.
            _ => Self::IbpComposition,
        }
    }
}

// ---------------------------------------------------------------------------
// Transform proof entries for certifying compiler (#4311)
// ---------------------------------------------------------------------------

/// Proof entry for a single compilation transform in the certifying compiler
/// pipeline.
///
/// Each peephole pass that modifies the compiled model during the Kokoro
/// compilation pipeline produces a `TransformProofEntry` proving that the
/// transformation preserves output equivalence. The collection of all entries
/// forms the `TransformProofBundle` that ships with the certified model.
///
/// Part of #4311: Verification gaps for Milestone 1 Kokoro certifying compiler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TransformProofEntry {
    /// Human-readable name of the transform (e.g., "FusedResBlock wiring",
    /// "Style projection absorption", "Batched style projection").
    pub transform_name: String,

    /// Which compilation pass produced this transform.
    pub pass_id: TransformPass,

    /// Lower bound on the diff (fused - sequential) output.
    pub diff_lower: f32,

    /// Upper bound on the diff (fused - sequential) output.
    pub diff_upper: f32,

    /// Maximum absolute difference proved by the verification.
    pub max_abs_diff: f32,

    /// Epsilon threshold used for the proof.
    pub epsilon: f32,

    /// Whether the proof confirmed `max_abs_diff <= epsilon`.
    pub within_epsilon: bool,

    /// Propagation method (IBP, CROWN, or Analytical).
    pub method: PropMethod,

    /// Optional Lean4 proof term emitted by NY for this transform.
    /// When present, can be checked by the Lean4 proof checker independently.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub lean4_proof_term: Option<String>,

    /// Optional SHA-256 hash of the transform's source code for traceability.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub source_hash: Option<String>,

    /// ISO 8601 timestamp of proof generation.
    pub generated_at: String,
}

impl TransformProofEntry {
    /// Create a new transform proof entry from verification results.
    #[must_use]
    pub fn new(
        transform_name: &str,
        pass_id: TransformPass,
        diff_lower: f32,
        diff_upper: f32,
        epsilon: f32,
        method: PropMethod,
    ) -> Self {
        let max_abs_diff = diff_lower.abs().max(diff_upper.abs());
        Self {
            transform_name: transform_name.to_string(),
            pass_id,
            diff_lower,
            diff_upper,
            max_abs_diff,
            epsilon,
            within_epsilon: max_abs_diff <= epsilon,
            method,
            lean4_proof_term: None,
            source_hash: None,
            generated_at: crate::certificate::now_iso8601(),
        }
    }

    /// Attach a Lean4 proof term for this transform.
    #[must_use]
    pub fn with_lean4_proof(mut self, lean4: String) -> Self {
        self.lean4_proof_term = Some(lean4);
        self
    }

    /// Attach a source hash for traceability.
    #[must_use]
    pub fn with_source_hash(mut self, hash: String) -> Self {
        self.source_hash = Some(hash);
        self
    }

    /// Whether this transform proof is sound (within epsilon).
    #[must_use]
    pub fn is_proved(&self) -> bool {
        self.within_epsilon
    }

    /// Whether this proof has a Lean4 machine-checkable certificate.
    #[must_use]
    pub fn has_lean4_proof(&self) -> bool {
        self.lean4_proof_term.is_some()
    }
}

/// Identifies which compilation pass produced a transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TransformPass {
    /// Pass 1: NormActivConv1d per-tap fusion.
    NormActivConv1dFusion,
    /// Pass 2: FusedResBlock wiring (residual + shortcut + scale).
    FusedResBlockWiring,
    /// Pass 3: Style projection absorption into FusedResBlock.
    StyleProjectionAbsorption,
    /// Pass 4: Batched style projection across FusedResBlocks.
    BatchedStyleProjection,
    /// Named fusions (AdaIN+Snake, LayerNorm+GELU, etc.)
    NamedFusion,
    /// Other/custom transforms.
    Other,
}

/// Bundle of transform proof entries for a certified compiled model.
///
/// Collects all `TransformProofEntry` instances from the compilation
/// pipeline. The success criterion for #4311 Milestone 1 is: every peephole
/// pass applied during compilation has a corresponding entry in this bundle
/// with `within_epsilon == true`.
///
/// Part of #4311.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TransformProofBundle {
    /// Model name this bundle covers.
    pub model_name: String,

    /// Per-transform proof entries.
    pub entries: Vec<TransformProofEntry>,

    /// Number of transforms applied during compilation.
    pub total_transforms: usize,

    /// ISO 8601 timestamp of bundle generation.
    pub generated_at: String,
}

impl TransformProofBundle {
    /// Create a new empty transform proof bundle.
    #[must_use]
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            entries: Vec::new(),
            total_transforms: 0,
            generated_at: crate::certificate::now_iso8601(),
        }
    }

    /// Add a transform proof entry.
    pub fn push(&mut self, entry: TransformProofEntry) {
        self.entries.push(entry);
    }

    /// Set the total number of transforms applied during compilation.
    pub fn set_total_transforms(&mut self, count: usize) {
        self.total_transforms = count;
    }

    /// Number of verified (proved) entries.
    #[must_use]
    pub fn proved_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_proved()).count()
    }

    /// Number of unverified entries.
    #[must_use]
    pub fn unverified_count(&self) -> usize {
        self.total_transforms.saturating_sub(self.proved_count())
    }

    /// Whether all transforms are verified.
    #[must_use]
    pub fn all_verified(&self) -> bool {
        self.proved_count() == self.total_transforms && self.total_transforms > 0
    }

    /// Number of entries with Lean4 machine-checkable proofs.
    #[must_use]
    pub fn lean4_proof_count(&self) -> usize {
        self.entries.iter().filter(|e| e.has_lean4_proof()).count()
    }

    /// Serialize to JSON.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if serialization fails.
    pub fn to_json(&self) -> Result<String, VerifyError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize from JSON.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if the JSON is malformed.
    pub fn from_json(json: &str) -> Result<Self, VerifyError> {
        Ok(serde_json::from_str(json)?)
    }
}

// ---------------------------------------------------------------------------
// Constructive proof summary — aggregated proof data from status entries (#4315)
// ---------------------------------------------------------------------------

/// Aggregated summary of constructive proof data across all status entries.
///
/// When a status file path is provided to [`CertifyConfig`], the certify
/// pipeline iterates over active (non-stale) status entries, extracts their
/// proof strength and method, and aggregates them into this summary. This
/// lets certificates report not just "verified" but "constructively proved
/// with method X, tightness Y" at the model level.
///
/// Part of #4315 (Wire NY proof certificates into certify pipeline).
///
/// [`CertifyConfig`]: crate::certify::CertifyConfig
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ConstructiveProofSummary {
    /// Total number of active (non-stale) status entries inspected.
    pub total_entries: usize,
    /// Number of entries classified as sound (SoundCrown, SoundIbp, SoundMixed).
    pub sound_count: usize,
    /// Number of entries classified as heuristic.
    pub heuristic_count: usize,
    /// Number of entries classified as vacuous.
    pub vacuous_count: usize,
    /// Breakdown of proof methods used across all active entries.
    /// Keys are method names (e.g., "Crown", "Ibp", "AlphaCrown").
    pub method_distribution: std::collections::BTreeMap<String, usize>,
    /// Ratio of sound entries to total active entries (0.0..=1.0).
    pub sound_ratio: f64,
    /// Whether all active entries are constructively proved (sound + non-vacuous).
    pub all_constructive: bool,
    /// Tightest output width across all active entries (smallest width = best).
    /// `None` if no active entries exist.
    pub tightest_width: Option<f32>,
    /// Widest output width across all active entries.
    /// `None` if no active entries exist.
    pub widest_width: Option<f32>,
    /// Number of entries with CROWN-family methods (Crown, AlphaCrown, BetaCrown,
    /// MixedIbpCrown).
    pub crown_method_count: usize,
    /// ISO 8601 timestamp of summary generation.
    pub generated_at: String,
}

impl ConstructiveProofSummary {
    /// Whether the summary indicates deployment-ready proof quality.
    ///
    /// Returns `true` when there is at least one entry, all entries are sound,
    /// and none are vacuous.
    #[must_use]
    pub fn is_deployment_ready(&self) -> bool {
        self.total_entries > 0 && self.all_constructive
    }

    /// Fraction of entries using CROWN-family methods.
    #[must_use]
    pub fn crown_coverage_ratio(&self) -> f64 {
        if self.total_entries == 0 {
            0.0
        } else {
            self.crown_method_count as f64 / self.total_entries as f64
        }
    }
}

/// Validate that a string is a valid SHA-256 hex digest (64 hex chars).
pub(crate) fn validate_sha256_hex(s: &str) -> Result<(), ()> {
    if s.len() != 64 {
        return Err(());
    }
    if !s.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(());
    }
    Ok(())
}
