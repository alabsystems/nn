// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Formal verification re-exports for `nn::verify`.
//!
//! IBP/CROWN bound propagation, proof certificates, fusion verification,
//! and verified model wrappers. Extracted from `lib.rs` for 450-line compliance.
//!
//! The trace-to-graph helpers in this module operate on an in-memory
//! `ComputationGraph` captured from DynTensor tracing or produced by earlier
//! graph-building steps. They are not direct `graph.json`, ONNX, or raw
//! PyTorch compiler entry points.
//!
//! For traced graphs, callers that need reusable NY build state
//! should prefer the `*_with_boundary` helpers below. They return a
//! `TraceGraphBoundaryResult` that owns the traced `GraphModel`, exposes
//! borrowed `GraphBuildInputs` via `graph_build_inputs()`, and can rebuild a
//! fresh `GraphNetwork` later via `build_graph_network()` without retracing.
//! This is a reusable verification boundary, not a full NN proof-powered
//! compiler.

// ---------------------------------------------------------------------------
// Core verification
// ---------------------------------------------------------------------------

pub use nn_verify::VerificationSoundnessMode;
pub use nn_verify::{verify_compiled, verify_trace, VerifiedModel, VerifyTraceResult};
pub use nn_verify::{KernelVerification, OutputTensorBounds, PropMethod};
pub use nn_verify::{NormBoundsMode, SpecVerification};
pub use nn_verify::{VerifyConfig, VerifyRequest};

// Error types
pub use nn_verify::{SmtError, StructuralError, VerifyError};

// ---------------------------------------------------------------------------
// Certification
// ---------------------------------------------------------------------------

pub use nn_verify::{certificate_from_pipeline, certificate_from_pipeline_enriched};
pub use nn_verify::{certify_model, CertifyConfig, CertifyError, CertifyResult};

// Proof certificates
pub use nn_verify::{
    CertificateBundle, CertificateEnrichment, CertificateError, ConstructiveLayerRecord,
    ConstructiveProofData, ConstructiveProofMethod, KaniOutcome, KaniProofRecord, LayerBoundRecord,
    PrecisionModel, ProofCertificate, CERTIFICATE_VERSION,
};

// Certificate checking
pub use nn_verify::{
    check_bundle, check_bundle_file, check_certificate, CheckIssue, CheckResult, VacuityAssessment,
};

// Edit certificates
pub use nn_verify::{EditCertificate, EditType, EditedWeight};

// ---------------------------------------------------------------------------
// Bounds computation
// ---------------------------------------------------------------------------

pub use nn_verify::{
    multi_scalar_input_bounds, scalar_input_bounds, uniform_bounds, ScalarInputBounds,
};
pub use nn_verify::{quick_bounds, quick_bounds_multi_input};
pub use nn_verify::{to_bounded_tensor, to_interval_bounds};
pub use nn_verify::{with_bounds, with_bounds_multi_input, BoundsPolicy, WithBoundsResult};

// Bound types (re-exported from NY for API stability)
pub use nn_verify::{BoundedTensor, VerificationResult, VerificationSpec};

// ---------------------------------------------------------------------------
// Verification status persistence
// ---------------------------------------------------------------------------

pub use nn_verify::{KernelStatus, VerifyOutcome, VerifyStatus};

// Status record types for verification pipeline output.
pub use nn_verify::{
    LockedStatus, OutputBoundsRecord, ParamInputRecord, SmtEncodingKind, SmtOutcome,
};

// ---------------------------------------------------------------------------
// Trace-to-graph translation
// ---------------------------------------------------------------------------

pub use nn_verify::extract_layer_bounds;

/// Translate a single-input traced graph directly to verification-ready
/// output. Use `trace_to_graph_model_with_boundary()` when the caller also
/// needs the reusable owned producer artifact instead of only the built graph
/// plus translation metadata.
pub use nn_verify::trace_to_graph_model;

/// Translate a multi-input traced graph directly to verification-ready
/// output. Use `trace_to_graph_model_multi_input_with_boundary()` when the
/// caller also needs the reusable owned producer artifact instead of only the
/// built graph plus translation metadata.
pub use nn_verify::trace_to_graph_model_multi_input;

/// Translate a single-input traced graph and keep the owned NY
/// producer boundary. The returned `TraceGraphBoundaryResult` lets callers
/// inspect the traced `GraphModel`, borrow `GraphBuildInputs`, and rebuild
/// the graph later without retracing. This remains a reusable verification
/// boundary over traced graph-build state, not a proof-powered compiler.
pub use nn_verify::trace_to_graph_model_with_boundary;

/// Multi-input variant of `trace_to_graph_model_with_boundary()`. Preserves
/// the stacked traced-input contract while returning the same reusable owned
/// producer boundary surface.
pub use nn_verify::trace_to_graph_model_multi_input_with_boundary;

/// Owned traced producer boundary returned by the `*_with_boundary` helpers.
/// Contains both the reusable `GraphModel` and the eagerly built graph so
/// downstream callers can inspect or rebuild the traced producer state via
/// `nn::verify` alone.
pub use nn_verify::TraceGraphBoundaryResult;

/// Owned gamma-build producer artifact captured from traced translation.
/// Keeps `LayerSpec` / `TensorSpec` / weight / provenance state together so
/// callers can inspect or rebuild later without retracing and without taking a
/// direct dependency on `nn-verify` or `gamma-build`.
pub use nn_verify::GraphModel;

/// Borrowed graph-build view over a traced `GraphModel`. Useful when a caller
/// needs to inspect the traced producer inputs/layers/outputs without
/// reconstructing them manually. This is a reusable build-boundary view, not
/// a higher-level compiler artifact.
pub use nn_verify::GraphBuildInputs;

/// Narrower traced-translation result for callers that only need the built
/// graph plus translation metadata, not the reusable owned producer boundary.
pub use nn_verify::TraceTranslateResult;

// ---------------------------------------------------------------------------
// Fusion verification
// ---------------------------------------------------------------------------

pub use nn_verify::verify_fusion_wiring;
pub use nn_verify::{generate_fusion_specs, AutoFusionSpec};
pub use nn_verify::{
    verify_fusion_equivalence, FusionSpec, FusionVerification, NamedFusionBounds,
};

// ---------------------------------------------------------------------------
// Composition and decomposition
// ---------------------------------------------------------------------------

pub use nn_verify::{compose_sequential, SequentialSpec};
pub use nn_verify::{decompose, decompose_at_norms, DecompositionResult, SubBlock};

// ---------------------------------------------------------------------------
// Pipeline verification (verify-and-record + tensor-level pipelines)
// ---------------------------------------------------------------------------

pub use nn_verify::{
    verify_and_record_full, verify_and_record_full_multi, verify_and_record_full_multi_with_config,
    verify_and_record_full_with_config,
};
pub use nn_verify::{
    verify_fusion_and_record, verify_fusion_and_record_with_config, verify_fusion_certificate,
    AutoFusionPipelineResult, CertifiedFusionResult,
};
pub use nn_verify::{verify_tensor_and_record, verify_tensor_and_record_with_config};

// ---------------------------------------------------------------------------
// Edit verification
// ---------------------------------------------------------------------------

pub use nn_verify::{verify_edit, EditVerification, EditVerificationSpec};

// ---------------------------------------------------------------------------
// Graph building
// ---------------------------------------------------------------------------

pub use nn_verify::{kernel_to_graph, kernel_to_graph_multi, ParamBinding};

// Tensor-level graph building for verification pipelines.
pub use nn_verify::{
    tensor_kernel_to_graph, tensor_kernel_to_graph_with_norm_mode, TensorParamBinding,
};

// ---------------------------------------------------------------------------
// Kani bridge
// ---------------------------------------------------------------------------

pub use nn_verify::kani_record_from_file;

// ---------------------------------------------------------------------------
// NY re-exports for API stability
// ---------------------------------------------------------------------------

pub use nn_verify::{GemmEngine, HeuristicUsed, NaiveCpuGemmEngine, SoundnessProvenance};
