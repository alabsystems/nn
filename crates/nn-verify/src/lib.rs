// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verification integration — NY bounds propagation + ay SMT verification.
//!
//! Two verification backends:
//!
//! - **NY**: Sound overapproximation via IBP/CROWN bound propagation.
//!   Fast, scales to large networks. Translates kernel IR to `GraphNetwork`.
//! - **ay**: SMT verification via `ay-bindings`. Phase A (current default): generates
//!   SMT-LIB2 with UF approximation (`SmtOutcome::Unexecuted`). Phase B: direct
//!   in-process solving (requires `ay-smt` feature flag, tracked in #29).
//!   Exact for pure arithmetic kernels; uses uninterpreted function (UF)
//!   approximation for transcendental ops (sin, cos, exp). See [`ay`] module.
//!
//! Results from both backends are persisted to per-model status files
//! (`nn_verify_status_*.json`: demucs, kokoro, qwen3, shared, silero, whisper).
//!
//! # Additional modules (NY feature)
//!
//! - **`fusion`**: Fusion equivalence verification — proves fused kernel produces
//!   same bounds as sequential composition via IBP diff.
//! - **`compose`**: Sequential model composition — chains sub-networks for
//!   end-to-end bound propagation.
//! - **`pipeline`**: Verify-and-record pipeline — runs verification and persists
//!   results to per-model status files in a single call.
//! - **`trace_to_graph`**: Translates `DynTensor` computation traces to NY
//!   `GraphNetwork` for IBP/CROWN verification of imperative model code.
//! - **`certificate`**: Proof certificates — `ProofCertificate` with source hash,
//!   Kani records, and layer bound records for deployment.
//! - **`overlay_composition`**: Overlay interaction verification — proves composed
//!   model segments satisfy cross-boundary bound specifications.
//!
//! # Quick start
//!
//! Requires the `NY` feature (enabled by default).
//!
//! ```rust,no_run
//! use nn_verify::{scalar_input_bounds, VerifyRequest};
//! use nn_dsl::lower::Lowerer;
//!
//! let func: syn::ItemFn = syn::parse_str(
//!     "fn snake(x: f32, alpha: f32) -> f32 { x + (1.0 / alpha) * (alpha * x).sin().powi(2) }"
//! ).expect("valid Rust");
//! let kernel = Lowerer::lower_fn(&func).expect("valid kernel");
//! let bounds = scalar_input_bounds(-10.0, 10.0).expect("input bounds");
//! let result = VerifyRequest::new(&kernel)
//!     .constant_params(&[1.0])
//!     .input_bounds(&bounds)
//!     .verify_bounds()
//!     .expect("verification");
//! assert!(result.is_finite);
//! ```
//!
//! # VerifyStatus retention
//!
//! `VerifyStatus` keeps two views of verification results:
//! - `kernels()`: latest run per kernel name (updated on every record)
//! - `history()`: append-only list of all runs per kernel name
//!
//! Both maps are private; read access is through `kernels()`, `kernel()`,
//! `history()`, `history_for()`, etc. Mutation is only through `record*` APIs
//! which enforce the latest/history coupling invariant.

// Compatibility shim: provides VerificationSoundnessMode and default_soundness_mode
// from ny_core when the feature is on, or a local mirror when off.
pub(crate) mod soundness_compat;

// --- verify_model! macro (always available, expansion is NY gated) ---
#[cfg(feature = "ny")]
mod verify_model_macro;

// --- ny-api capability adapters: consume ny's curated facade so NN gains the
// laddered driver, deployed-precision (Metal f16) soundness, op-coverage
// conformance, and Clean-checkable certificates. Additive; the legacy verify
// path is untouched. ---
/// Laddered model verification (IBP -> alpha -> CROWN -> beta -> MIP) via ny_api::ladder.
#[cfg(feature = "ny")]
pub mod ny_ladder;
/// Deployed-precision (f16/bf16) sound verification via ny_api::precision.
#[cfg(feature = "ny")]
pub mod ny_precision;
/// Operation soundness-coverage report via ny_api::conformance.
#[cfg(feature = "ny")]
pub mod ny_conformance;
/// Load-once / verify-many sessions with a sound verdict cache via ny_api::session.
#[cfg(feature = "ny")]
pub mod ny_session;
/// Proof-carrying, Clean-checkable certificates via ny_api::cert (opt-in).
#[cfg(feature = "ny-cert")]
pub mod ny_certify;
/// Sound exact-rational global Lipschitz certification via ny_api::lipschitz (opt-in).
#[cfg(feature = "ny-cert")]
pub mod ny_lipschitz;
/// Complete verification terminal (HiGHS MIP) via ny_api::complete (opt-in).
#[cfg(feature = "ny-complete")]
pub mod ny_complete;

/// Sound beta-CROWN (GenBaB branch-and-bound) verification path via
/// `ny_propagate::BetaCrownVerifier`. Proves lower/upper-bound box properties by
/// splitting multiplicative nodes that IBP/alpha-CROWN decorrelate.
#[cfg(feature = "ny")]
pub mod beta_crown_verify;

// --- Always-available modules (no NY dependency) ---
pub mod behavioral_contract;
pub mod bound_analysis;
pub(crate) mod bounds;
pub mod certificate;
pub mod certificate_checker;
pub(crate) mod certificate_types;
pub mod dpdf_certify;
pub mod edit_certificate;
pub mod error;
pub mod external_bounds;
pub mod fingerprint;
pub mod gap_detector;
pub mod istft_linear_matrix;
pub mod kokoro_certificate;
pub mod kokoro_certificate_bundle;
pub mod kokoro_crown_certificate;
pub mod tighten;

pub mod junction_contract;
#[cfg(kani)]
mod kani_attention_safety;
pub(crate) mod kani_bridge;
#[cfg(kani)]
#[path = "kani_dpdf_certify_proofs.rs"]
mod kani_dpdf_certify_proofs;
#[cfg(kani)]
mod kani_dpdf_kernels;
#[cfg(kani)]
mod kani_dpdf_kernels_edge;
#[cfg(kani)]
#[path = "kani_fingerprint.rs"]
mod kani_fingerprint;
#[cfg(kani)]
mod kani_fingerprint_issue3729;
#[cfg(kani)]
#[path = "kani_fingerprint_subgraph.rs"]
mod kani_fingerprint_subgraph;
#[cfg(kani)]
#[path = "kani_fusion_adain.rs"]
mod kani_fusion_adain;
#[cfg(kani)]
mod kani_fusion_equivalence;
#[cfg(kani)]
mod kani_gap_detector;
#[cfg(kani)]
mod kani_graph_construction_proofs;
#[cfg(kani)]
mod kani_istft_matrix;
#[cfg(kani)]
mod kani_model_properties;
#[cfg(kani)]
mod kani_quantization_proofs;
#[cfg(kani)]
#[path = "kani_snake_bounds.rs"]
mod kani_snake_bounds;
#[cfg(kani)]
#[path = "kani_subgraph_extract.rs"]
mod kani_subgraph_extract;
#[cfg(kani)]
#[path = "kani_subgraph_extract_3743.rs"]
mod kani_subgraph_extract_3743;
#[cfg(kani)]
mod kani_subgraph_extract_issue3729;
#[cfg(kani)]
mod kani_verify_properties;
#[cfg(kani)]
#[path = "kani_ay_fp_snake_extended.rs"]
mod kani_ay_fp_snake_extended;
#[cfg(kani)]
#[path = "kani_ay_snake_decompose.rs"]
mod kani_ay_snake_decompose;
pub mod proof_bundle;
pub mod selective_crown;
pub mod signing_config;
pub(crate) mod smt_error;
pub mod status;
pub mod status_report;
pub(crate) mod status_smt;
pub mod subgraph_extract;
pub(crate) mod util;
pub(crate) mod verify_types;
#[cfg(feature = "ay-smt")]
pub mod ay;
#[cfg(feature = "ay-smt")]
pub mod ay_attention_mask_position_bias_dpdf;
#[cfg(feature = "ay-smt")]
pub mod ay_real_lit;
/// Network synthesis with ay: solve for weights from constraints (ay as backend).
#[cfg(feature = "ay-smt")]
pub mod ay_synthesize;
#[cfg(feature = "ay-smt")]
pub mod ay_attention_mask_properties;
#[cfg(feature = "ay-smt")]
pub mod ay_conv_stride_padding_properties;
#[cfg(feature = "ay-smt")]
pub mod ay_gradient_computation_proofs;
#[cfg(feature = "ay-smt")]
pub mod ay_gradient_computation_properties;
#[cfg(feature = "ay-smt")]
pub mod ay_matrix_decomposition;
#[cfg(feature = "ay-smt")]
pub mod ay_matrix_decomposition_properties;
#[cfg(feature = "ay-smt")]
pub mod ay_normalization_layer_properties;
#[cfg(feature = "ay-smt")]
pub mod ay_quantization_dpdf_vlm;
#[cfg(feature = "ay-smt")]
pub mod ay_quantization_error_bounds;
#[cfg(feature = "ay-smt")]
pub mod ay_reshape_view_properties;
#[cfg(feature = "ay-smt")]
pub mod ay_rope_position_properties;
#[cfg(feature = "ay-smt")]
pub mod ay_softmax_cross_entropy;
#[cfg(feature = "ay-smt")]
pub mod ay_softmax_cross_entropy_dpdf;
#[cfg(feature = "ay-smt")]
pub mod ay_softmax_cross_entropy_extended;
/// Detect SMT queries that are UNSAT for reasons unrelated to the property.
#[cfg(feature = "ay-smt")]
pub(crate) mod ay_vacuity;
#[cfg(feature = "ay-smt")]
pub mod ay_weight_init_constraints;

// --- NY-gated modules ---
#[cfg(feature = "ny")]
pub(crate) mod bounds_bridge;
#[cfg(feature = "ny")]
pub mod certify;
#[cfg(feature = "ny")]
pub mod compile_bridge;
#[cfg(feature = "ny")]
pub(crate) mod compose;
#[cfg(feature = "ny")]
pub mod dead_neuron_proof;
#[cfg(feature = "ny")]
pub mod edit_verify;
#[cfg(feature = "ny")]
pub mod fusion;
#[cfg(feature = "ny")]
pub(crate) mod fusion_adain;
#[cfg(feature = "ny")]
pub mod fusion_auto;
#[cfg(feature = "ny")]
pub mod fusion_auto_verify;
#[cfg(feature = "ny")]
pub mod fusion_certificate;
#[cfg(feature = "ny")]
pub(crate) mod fusion_norm_activ_conv;
#[cfg(feature = "ny")]
pub(crate) mod fusion_spec;
#[cfg(feature = "ny")]
pub mod fusion_wiring;
#[cfg(feature = "ny")]
pub(crate) mod graph;
#[cfg(feature = "ny")]
pub(crate) mod graph_ops;
#[cfg(feature = "ny")]
pub(crate) mod graph_tensor;
#[cfg(feature = "ny")]
pub mod layer_bounds;
#[cfg(feature = "ny")]
pub mod overlay_composition;
#[cfg(feature = "ny")]
pub mod parallel;
#[cfg(feature = "ny")]
pub mod pipeline;
#[cfg(feature = "ny")]
pub mod quick_bounds;
#[cfg(feature = "ny")]
pub mod resblock_equivalence;
#[cfg(feature = "ny")]
pub(crate) mod soundness;
#[cfg(feature = "ny")]
pub mod subblock_decompose;
#[cfg(feature = "ny")]
pub mod tightening_loop;
#[cfg(feature = "ny")]
pub(crate) mod trace_to_graph;
// Serialize an nn-core ComputationGraph into the NY bridge schema consumed by
// ny_trace_bridge::translate — nn-verify's (only) trace translation path.
// The converter is total over all TraceOp variants; the bridge translator
// refuses unsupported/unsound ops fail-closed.
#[cfg(feature = "ny")]
pub(crate) mod trace_to_schema;
#[cfg(feature = "ny")]
pub mod verify;
#[cfg(feature = "ny")]
pub(crate) mod verify_input;
#[cfg(feature = "ny")]
pub mod verify_request;
#[cfg(feature = "ny")]
pub mod verify_trace;
#[cfg(feature = "ny")]
pub mod with_bounds;

// --- Always-available re-exports ---
pub use behavioral_contract::{
    BehavioralContract, ContractError, ContractProperty, ContractValidation,
};
pub use certificate::integrity::{
    compute_content_hash, sign_bundle, sign_certificate, verify_bundle_signatures,
    verify_bundle_signatures_strict, verify_content_hash, verify_signature, BundleIntegrityError,
    IntegrityError, Signable,
};
pub use certificate::{
    compute_bytes_hash, compute_file_hash, CertificateBundle, CertificateEnrichment,
    CertificateError, KaniOutcome, KaniProofRecord, LayerBoundRecord, PrecisionModel,
    ProofCertificate, CERTIFICATE_VERSION,
};
pub use certificate_checker::{
    check_bundle, check_bundle_file, check_bundle_file_with_key, check_bundle_with_key,
    check_certificate, check_certificate_with_key, CheckIssue, CheckResult, VacuityAssessment,
    DEFAULT_VACUITY_THRESHOLD,
};
pub use certificate_types::{
    ConstructiveLayerRecord, ConstructiveProofData, ConstructiveProofMethod,
    ConstructiveProofSummary, TransformPass, TransformProofBundle, TransformProofEntry,
};
pub use edit_certificate::{EditCertificate, EditType, EditedWeight};
pub use error::{StructuralError, VerifyError};
pub use external_bounds::{
    load_external_bounds, load_external_bounds_from_bytes, verification_from_external,
    verify_and_record_external, verify_and_record_external_from_loaded, ExternalBounds,
    ExternalBoundsSource, ExternalLayerBounds,
};
pub use fingerprint::{
    diff_fingerprints, fingerprint_graph, fingerprint_graph_with_weights, fingerprint_trace,
    fingerprint_trace_with_weights, ChangeReason, ChangedRegion, SubgraphFingerprint,
};
pub use junction_contract::{
    verify_junction, verify_junctions, JunctionProof, JunctionVerification, SubBlockBounds,
    JUNCTION_MARGIN,
};
pub use kani_bridge::kani_record_from_file;
pub use kokoro_certificate::{
    generate_kokoro_certificate, verify_certificate as verify_kokoro_certificate,
    CertificateConfig as KokoroCertificateConfig, CertificateVerdict as KokoroCertificateVerdict,
    KokoroCertificate, KOKORO_CERTIFICATE_VERSION,
};
pub use kokoro_certificate_bundle::{
    BundleConfig, DeploymentThresholds, EntrySoundness, EntrySoundnessRecord,
    KokoroCertificateBundle, SoundnessBreakdown,
};
pub use kokoro_crown_certificate::{
    generate_deployment_certificate, DeploymentConfig, DeploymentGate, KokoroCrownCertificate,
    StageCrownCoverage,
};
pub use proof_bundle::{
    BoundCertificate, CrownSummary, KaniSummary, ProofBundle, ProofBundleBuilder, ProofBundleError,
    VerificationMethod,
};
pub use selective_crown::{
    analyze_and_select, analyze_selective_crown, select_crown_layers, select_from_recommendations,
    simulate_crown_tightening, EscalationStrategy, SelectiveCrownAnalysis, SelectiveCrownConfig,
};
pub use signing_config::SigningKey;
pub use smt_error::SmtError;
pub use soundness_compat::VerificationSoundnessMode;
pub use status::{
    model_for_kernel, model_status_path, BoundsSource, InputBoundsRecord, KernelStatus,
    LockedStatus, OutputBoundsRecord, ParamInputRecord, SmtEncodingKind, SmtOutcome,
    SmtProofVerdict, SmtStatusRecord, VerifyOutcome, VerifyStatus, MODEL_CATEGORIES,
};
pub use verify_types::{KernelVerification, OutputTensorBounds, PropMethod, VerifyConfig};

// Verification status report generator (#3942).
pub use status_report::{GapSummary, ModelSummary, StatusReport, Trend, VerificationBreakdown};

// Subgraph extraction for ay SMT verification (#2455).
pub use subgraph_extract::{
    extract_subgraph, find_ay_candidates, is_ay_compatible_op, validate_subgraph,
    ExtractedSubgraph, SubgraphSpec, AYCandidateRegion,
};
pub use tighten::{
    classify_tightening, compare_report_paths, compare_reports, format_tightening_diff,
    load_bound_analysis_report, TighteningDiff, TighteningOutcome,
};

// Analytical bounds for harmonic_source cumsum→sin pattern (#2411).
pub use bounds::harmonic_source::HarmonicSourceBounds;

// --- NY-gated re-exports ---
#[cfg(feature = "ny")]
pub use bounds_bridge::{to_bounded_tensor, to_interval_bounds};
#[cfg(feature = "ny")]
pub use certificate::certificate_from_pipeline;
#[cfg(feature = "ny")]
pub use certificate::certificate_from_pipeline_enriched;
#[cfg(feature = "ny")]
pub use certify::{
    certify_model, verify_and_certify, CertifyConfig, CertifyError, CertifyResult,
    ProofStrengthClassification, VerifyAndCertifyResult,
};
#[cfg(feature = "ny")]
pub use compile_bridge::{verify_compiled, verify_compiled_with_transforms, VerifiedModel};
#[cfg(feature = "ny")]
pub use compose::{compose_sequential, SequentialSpec};
#[cfg(feature = "ny")]
pub use dead_neuron_proof::{run_dead_neuron_elimination, DeadNeuronEliminationProof};
#[cfg(feature = "ny")]
pub use edit_verify::{
    verify_edit, verify_edit_with_elimination, EditVerification, EditVerificationSpec,
};
#[cfg(feature = "ny")]
pub use fusion::{
    build_fusion_diff_graph, propagate_with_crown_fallback, verify_ada_layer_norm_fusion,
    verify_ada_layer_norm_fusion_with_config, verify_adain_leaky_relu_fusion,
    verify_adain_leaky_relu_fusion_with_config, verify_adain_snake_fusion,
    verify_adain_snake_fusion_with_config, verify_all_named_fusions, verify_fusion_equivalence,
    verify_fusion_equivalence_with_config, verify_layer_norm_gelu_fusion,
    verify_layer_norm_gelu_fusion_with_config, verify_rms_norm_silu_mul_fusion,
    verify_rms_norm_silu_mul_fusion_with_config, FusionSpec, FusionVerification, NamedFusionBounds,
};
#[cfg(feature = "ny")]
pub use fusion_auto::{generate_fusion_specs, AutoFusionSpec};
#[cfg(feature = "ny")]
pub use fusion_auto_verify::{
    verify_all_fusion_specs, verify_and_record_auto_fusion, AutoFusionResult,
};
#[cfg(feature = "ny")]
pub use fusion_certificate::{
    AnalyticalFusionBound, FusionEquivalenceCertificate, FUSION_CERTIFICATE_VERSION,
};
#[cfg(feature = "ny")]
pub use fusion_norm_activ_conv::{
    verify_norm_activ_conv1d_leaky_relu_fusion,
    verify_norm_activ_conv1d_leaky_relu_fusion_with_config, verify_norm_activ_conv1d_snake_fusion,
    verify_norm_activ_conv1d_snake_fusion_with_config,
};
#[cfg(feature = "ny")]
pub use fusion_wiring::verify_fusion_wiring;
#[cfg(feature = "ny")]
pub use graph::{kernel_to_graph, kernel_to_graph_multi, ParamBinding};
#[cfg(feature = "ny")]
pub use graph_tensor::{
    chain_graphs, model_to_graph_network, model_to_graph_network_with_norm_mode,
    tensor_kernel_to_graph, tensor_kernel_to_graph_with_norm_mode, tensor_kernels_to_grouped_graph,
    TensorParamBinding,
};
#[cfg(feature = "ny")]
pub use layer_bounds::extract_layer_bounds;
#[cfg(feature = "ny")]
pub use overlay_composition::{
    overlay_interaction_matrix, verify_overlay_composition, BoundSpec, CompositionCertificate,
    SpecResult, VerifiedOverlay,
};
#[cfg(feature = "ny")]
pub use parallel::{
    parallel_verify_positions, parallel_verify_with_method, ParallelVerifyBackend,
    ParallelVerifyConfig,
};
#[cfg(feature = "ny")]
pub use pipeline::{
    certify_auto_fusion_from_graph, verify_and_record_auto_fusion_from_graph,
    verify_and_record_full, verify_and_record_full_multi, verify_and_record_full_multi_with_config,
    verify_and_record_full_with_config, verify_auto_fusion_from_graph, verify_fusion_and_record,
    verify_fusion_and_record_with_config, verify_fusion_certificate, verify_tensor_and_record,
    verify_tensor_and_record_with_config, AutoFusionPipelineResult, CertifiedFusionResult,
    FusionPipelineResult, PipelineResult, TensorPipelineResult,
};
#[cfg(feature = "ny")]
pub use quick_bounds::{quick_bounds, quick_bounds_multi_input};
#[cfg(feature = "ny")]
pub use soundness::soundness_mode_for_graph;
#[cfg(feature = "ny")]
pub use subblock_decompose::{decompose, decompose_at_norms, DecompositionResult, SubBlock};
#[cfg(feature = "ny")]
pub use tightening_loop::{
    run_tightening_loop, TighteningConfig, TighteningResult, TighteningStep, AYCandidateRange,
};
#[cfg(feature = "ny")]
pub use trace_to_graph::{
    trace_to_graph_model, trace_to_graph_model_multi_input,
    trace_to_graph_model_multi_input_with_boundary, trace_to_graph_model_with_boundary,
    trace_to_graph_segmented, SegmentTranslation, SegmentedTranslateResult,
    TraceGraphBoundaryResult, TraceTranslateResult,
};
#[cfg(feature = "ny")]
pub use trace_to_schema::{segmented_to_schema, to_schema};
#[cfg(feature = "ny")]
pub use verify::{NormBoundsMode, SpecVerification};
#[cfg(feature = "ny")]
pub use verify_input::{
    multi_scalar_input_bounds, scalar_input_bounds, uniform_bounds, ScalarInputBounds,
};
#[cfg(feature = "ny")]
pub use verify_request::VerifyRequest;
#[cfg(feature = "ny")]
pub use verify_trace::{verify_trace, VerifyTraceResult};
#[cfg(feature = "ny")]
pub use with_bounds::{with_bounds, with_bounds_multi_input, BoundsPolicy, WithBoundsResult};
#[cfg(feature = "ay-smt")]
pub use ay::{
    kernel_to_smt2, kernel_to_smt2_with_bounds, verify_kernel_smt, verify_kernel_smt_multi,
    verify_kernel_smt_with_bounds, TranslatedKernel,
};

// Re-export NY types when the feature is enabled.
// Consumers (e.g., dvoice) should use these re-exports rather than depending
// on NY crates directly, to avoid version divergence (#1091).
#[cfg(feature = "ny")]
pub use ny_api::{Bound, BoundedTensor, VerificationResult, VerificationSpec};
#[cfg(feature = "ny")]
pub use ny_build::{GraphBuildInputs, GraphModel};
#[cfg(feature = "ny")]
pub use ny_core::{GemmEngine, HeuristicUsed, NaiveCpuGemmEngine, SoundnessProvenance};
#[cfg(feature = "ny")]
pub use ny_propagate::{
    layers::{Layer, LinearLayer, ReLULayer, SigmoidLayer},
    GraphNetwork, Network,
};

/// Double-precision (f64) propagation types for bound tightness validation.
///
/// `evaluate_network_f64` evaluates a sequential network (Linear+Conv2d+ReLU) in f64
/// precision — useful for differential testing and bound tightness measurement.
/// `convert_network_to_f64` converts a `Network`'s layers to f64 format.
///
/// Part of #4316: f64 evaluation for bound tightness.
#[cfg(feature = "ny")]
pub use ny_propagate::{
    convert_network_to_f64, evaluate_network_f64, F64PropagationMode, SequentialLayerF64,
};

/// Hidden re-exports for `verify_model!` macro hygiene.
///
/// The macro uses `$crate::__macro_internals::trace_graph` etc. so that
/// consumers don't need to import `nn_core::dyn_tensor::trace` directly.
#[cfg(feature = "ny")]
#[doc(hidden)]
pub mod __macro_internals {
    pub use nn_core::dyn_tensor::trace::{record_input, trace_graph, ComputationGraph};
    pub use nn_core::dyn_tensor::DynTensor;
}

/// Shared test helpers for unit test modules across the crate (#505, #605).
/// Only available when NY is enabled (uses graph translation and
/// bounds propagation).
#[cfg(all(test, feature = "ny"))]
pub(crate) mod test_helpers {
    use crate::graph::{kernel_to_graph, kernel_to_graph_multi, ParamBinding};
    use crate::verify_input::ScalarInputBounds;

    pub(crate) fn bounds(lo: f32, hi: f32) -> ScalarInputBounds {
        ScalarInputBounds::new(lo, hi).expect("valid test bounds")
    }

    /// Parse a Rust function string into a `KernelDef`.
    pub(crate) fn parse_kernel(src: &str) -> nn_dsl::ir::KernelDef {
        let func: syn::ItemFn = syn::parse_str(src).expect("parse Rust source");
        nn_dsl::lower::Lowerer::lower_fn(&func).expect("lower to KernelDef")
    }

    /// Translate a single-variable kernel and propagate IBP bounds.
    /// Returns (lower, upper) of the output.
    pub(crate) fn propagate_single(
        src: &str,
        constants: &[f32],
        input_lo: f32,
        input_hi: f32,
    ) -> (f32, f32) {
        let kernel = parse_kernel(src);
        let graph = kernel_to_graph(&kernel, constants).expect("translate to graph");
        let input =
            crate::verify_input::scalar_input_bounds(input_lo, input_hi).expect("input bounds");
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        let lower = output.lower().as_slice().expect("lower slice")[0];
        let upper = output.upper().as_slice().expect("upper slice")[0];
        (lower, upper)
    }

    /// Translate a multi-variable kernel and propagate IBP bounds.
    /// Returns (lower, upper) of the output.
    pub(crate) fn propagate_multi(
        src: &str,
        bindings: &[ParamBinding],
        variable_bounds: &[(f32, f32)],
    ) -> (f32, f32) {
        let kernel = parse_kernel(src);
        let graph = kernel_to_graph_multi(&kernel, bindings).expect("translate to graph");
        let input = crate::verify_input::multi_scalar_input_bounds(variable_bounds)
            .expect("multi input bounds");
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        let lower = output.lower().as_slice().expect("lower slice")[0];
        let upper = output.upper().as_slice().expect("upper slice")[0];
        (lower, upper)
    }
}

#[cfg(test)]
#[path = "crown_certificate_tests.rs"]
mod crown_certificate_tests;

#[cfg(test)]
#[path = "certification_extended_tests2.rs"]
mod certification_extended_tests2;

#[cfg(test)]
#[path = "verification_infrastructure_extended_tests.rs"]
mod verification_infrastructure_extended_tests;

#[cfg(all(test, feature = "ay-smt"))]
#[path = "ay_convolution_properties_tests.rs"]
mod ay_convolution_properties_tests;

#[cfg(all(test, feature = "ay-smt"))]
#[path = "ay_normalization_properties_tests.rs"]
mod ay_normalization_properties_tests;

#[cfg(all(test, feature = "ay-smt"))]
#[path = "ay_matrix_decomposition_tests.rs"]
mod ay_matrix_decomposition_tests;
