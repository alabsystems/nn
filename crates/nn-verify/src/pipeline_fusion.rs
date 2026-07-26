// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fusion equivalence verification pipeline (#803).
//!
//! Extracted from `pipeline.rs` to stay within the 500-line file limit.

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_dsl::detect_fusion_chains;

use crate::error::VerifyError;
use crate::fusion_auto::generate_fusion_specs;
use crate::fusion_auto_verify::{verify_all_fusion_specs, AutoFusionResult};
use crate::fusion_certificate::{AnalyticalFusionBound, FusionEquivalenceCertificate};
use crate::fusion_spec::{FusionSpec, FusionVerification};
use crate::status::VerifyStatus;

/// Result of a fusion equivalence verification pipeline run.
#[derive(Debug)]
#[non_exhaustive]
pub struct FusionPipelineResult {
    /// Fusion equivalence verification result.
    pub fusion: FusionVerification,
}

/// Verify that a fused kernel is equivalent to its sequential components
/// and record the result to the status file.
///
/// Chains `verify_fusion_equivalence` → `record_fusion` into a single call,
/// preventing the gap where fusion results can be computed but not persisted.
///
/// Uses `VerifyConfig::default()`. For custom configuration (soundness mode,
/// normalization bounds mode), use [`verify_fusion_and_record_with_config`].
///
/// `variable_bounds` provides `(lower, upper)` for each shared input variable.
/// `epsilon` is the maximum allowed absolute difference between fused and
/// sequential outputs.
///
/// `status_key` overrides the entry key in `nn_verify_status.json`. When
/// `None`, the default `"fusion_{fused_kernel_name}"` is used.
///
/// # Errors
///
/// Returns an error if fusion verification or status recording fails.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_fusion_and_record(
    status: &mut VerifyStatus,
    spec: &FusionSpec<'_>,
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    status_key: Option<&str>,
) -> Result<FusionPipelineResult, VerifyError> {
    // 1. Run fusion equivalence verification via NY diamond DAG.
    let fusion = crate::fusion::verify_fusion_equivalence(spec, variable_bounds, epsilon)?;

    // 2. Record result to status file.
    status.record_fusion(&fusion, variable_bounds, status_key)?;

    Ok(FusionPipelineResult { fusion })
}

/// Verify that a fused kernel is equivalent to its sequential components
/// with explicit [`VerifyConfig`] and record the result to the status file.
///
/// Same as [`verify_fusion_and_record`] but accepts a [`VerifyConfig`] to
/// control soundness mode and normalization bounds mode (#2225).
///
/// # Errors
///
/// Returns an error if fusion verification or status recording fails.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_fusion_and_record_with_config(
    status: &mut VerifyStatus,
    spec: &FusionSpec<'_>,
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    status_key: Option<&str>,
    config: &crate::verify::VerifyConfig,
) -> Result<FusionPipelineResult, VerifyError> {
    let fusion = crate::fusion::verify_fusion_equivalence_with_config(
        spec,
        variable_bounds,
        epsilon,
        config,
    )?;

    status.record_fusion(&fusion, variable_bounds, status_key)?;

    Ok(FusionPipelineResult { fusion })
}

/// Generate a `FusionEquivalenceCertificate` from fusion verification.
///
/// Runs NY diamond DAG verification and wraps the result in a
/// certificate that can be serialized and shipped with the compiled model.
/// Optionally includes an analytical ULP bound.
///
/// # Arguments
///
/// * `spec` — Fusion specification (fused kernel + sequential pair)
/// * `first_name` — name of the first sequential kernel
/// * `second_name` — name of the second sequential kernel
/// * `variable_bounds` — per-variable input bounds
/// * `epsilon` — maximum tolerable absolute difference
/// * `production_dim` — production tensor dimension (e.g. 512)
/// * `analytical` — optional pre-computed analytical ULP bound
///
/// # Errors
///
/// Returns `VerifyError` if verification or certificate validation fails.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_fusion_certificate(
    spec: &FusionSpec<'_>,
    first_name: &str,
    second_name: &str,
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    production_dim: usize,
    analytical: Option<AnalyticalFusionBound>,
) -> Result<FusionEquivalenceCertificate, VerifyError> {
    let verification = crate::fusion::verify_fusion_equivalence(spec, variable_bounds, epsilon)?;

    let mut cert = FusionEquivalenceCertificate::from_verification(
        &verification,
        first_name,
        second_name,
        production_dim,
        variable_bounds,
    );
    if let Some(bound) = analytical {
        cert = cert.with_analytical_bound(bound);
    }
    cert.validate()?;
    Ok(cert)
}

/// Result of auto-fusion verification from a computation graph.
#[derive(Debug)]
#[non_exhaustive]
pub struct AutoFusionPipelineResult {
    /// Number of fusion chains detected in the graph.
    pub chains_detected: usize,
    /// Number of pairwise specs generated (N-1 per N-op chain).
    pub specs_generated: usize,
    /// Per-spec verification results.
    pub results: Vec<AutoFusionResult>,
    /// Number of conclusive CROWN proofs (tight bounds).
    pub conclusive_count: usize,
}

/// Detect fusible elementwise chains in a computation graph, generate
/// fusion specs, and verify each with NY.
///
/// Single-call pipeline: `ComputationGraph` → chain detection → spec
/// generation → CROWN verification. Returns results for all detected
/// fusion pairs. This is the production entry point for auto-fusion
/// verification — call it alongside `CompiledModel::builder().build()` to
/// verify that all fused GPU kernels are equivalent to their unfused
/// counterparts.
///
/// `variable_bounds` provides `(lower, upper)` for shared inputs. When
/// the slice length doesn't match a spec's input count, defaults to
/// `[-3.0, 3.0]` per variable.
///
/// `epsilon` is the maximum allowed absolute difference between fused
/// and sequential outputs.
///
/// # Errors
///
/// Returns `VerifyError` if chain detection fails (malformed graph).
/// Individual verification failures are captured in each result entry.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_auto_fusion_from_graph(
    graph: &ComputationGraph,
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
) -> Result<AutoFusionPipelineResult, VerifyError> {
    let chains = detect_fusion_chains(graph)?;
    let chains_detected = chains.len();
    let specs = generate_fusion_specs(&chains);
    let specs_generated = specs.len();
    let results = verify_all_fusion_specs(&specs, variable_bounds, epsilon);

    let conclusive_count = results
        .iter()
        .filter(|r| {
            r.result
                .as_ref()
                .map(|v| v.is_conclusive() && v.within_epsilon)
                .unwrap_or(false)
        })
        .count();

    Ok(AutoFusionPipelineResult {
        chains_detected,
        specs_generated,
        results,
        conclusive_count,
    })
}

/// Result of auto-fusion verification with proof certificates.
#[derive(Debug)]
#[non_exhaustive]
pub struct CertifiedFusionResult {
    /// Verification pipeline result.
    pub verification: AutoFusionPipelineResult,
    /// Certificates for CROWN-conclusive fusion pairs.
    ///
    /// One certificate per fusion pair that was proved equivalent within
    /// epsilon using CROWN (not IBP fallback). These can be serialized
    /// to JSON and shipped alongside the compiled model for offline auditing.
    pub certificates: Vec<FusionEquivalenceCertificate>,
}

/// Detect, verify, and certify fusion equivalence from a computation graph.
///
/// Same as [`verify_auto_fusion_from_graph`] but additionally generates
/// [`FusionEquivalenceCertificate`]s for each CROWN-conclusive fusion pair.
/// Certificates can be serialized and shipped with the compiled model for
/// offline auditing.
///
/// `production_dim` is the production tensor dimension (e.g. 512 for
/// hidden_dim). This is recorded in each certificate for provenance.
///
/// # Errors
///
/// Returns `VerifyError` if chain detection fails (malformed graph).
/// Individual verification failures are captured in each result entry.
#[must_use = "returns a Result that may contain an error"]
pub fn certify_auto_fusion_from_graph(
    graph: &ComputationGraph,
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    production_dim: usize,
) -> Result<CertifiedFusionResult, VerifyError> {
    let chains = detect_fusion_chains(graph)?;
    let chains_detected = chains.len();
    let specs = generate_fusion_specs(&chains);
    let specs_generated = specs.len();
    let results = verify_all_fusion_specs(&specs, variable_bounds, epsilon);

    // Generate certificates for conclusive CROWN proofs.
    let mut certificates = Vec::new();
    for (result, spec) in results.iter().zip(specs.iter()) {
        if let Ok(ref verification) = result.result {
            if verification.is_conclusive() && verification.within_epsilon {
                let bounds = if variable_bounds.len() == spec.num_shared_inputs() {
                    variable_bounds.to_vec()
                } else if variable_bounds.len() == 1 {
                    vec![variable_bounds[0]; spec.num_shared_inputs()]
                } else {
                    vec![(-3.0, 3.0); spec.num_shared_inputs()]
                };
                let cert = FusionEquivalenceCertificate::from_verification(
                    verification,
                    &spec.first.name,
                    &spec.second.name,
                    production_dim,
                    &bounds,
                );
                if cert.validate().is_ok() {
                    certificates.push(cert);
                }
            }
        }
    }

    let conclusive_count = results
        .iter()
        .filter(|r| {
            r.result
                .as_ref()
                .map(|v| v.is_conclusive() && v.within_epsilon)
                .unwrap_or(false)
        })
        .count();

    Ok(CertifiedFusionResult {
        verification: AutoFusionPipelineResult {
            chains_detected,
            specs_generated,
            results,
            conclusive_count,
        },
        certificates,
    })
}

/// Detect, verify, and record auto-fusion results from a computation graph.
///
/// Same as [`verify_auto_fusion_from_graph`] but additionally records each
/// successful verification to `nn_verify_status.json`.
///
/// # Errors
///
/// Returns `VerifyError` if chain detection or status recording fails.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_and_record_auto_fusion_from_graph(
    graph: &ComputationGraph,
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    status: &mut VerifyStatus,
) -> Result<AutoFusionPipelineResult, VerifyError> {
    let chains = detect_fusion_chains(graph)?;
    let chains_detected = chains.len();
    let specs = generate_fusion_specs(&chains);
    let specs_generated = specs.len();
    let results = crate::fusion_auto_verify::verify_and_record_auto_fusion(
        &specs,
        variable_bounds,
        epsilon,
        status,
    )?;

    let conclusive_count = results
        .iter()
        .filter(|r| {
            r.result
                .as_ref()
                .map(|v| v.is_conclusive() && v.within_epsilon)
                .unwrap_or(false)
        })
        .count();

    Ok(AutoFusionPipelineResult {
        chains_detected,
        specs_generated,
        results,
        conclusive_count,
    })
}
