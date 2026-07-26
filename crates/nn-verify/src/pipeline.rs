// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unified verify-and-record pipeline for NY + ay SMT.
//!
//! Chains NY bounds verification, status recording, ay SMT
//! cross-verification, and SMT result recording into a single call.
//! This prevents the decoupled-recording gap where ay results can be
//! silently dropped if the caller forgets `record_smt()`.

use nn_dsl::ir::KernelDef;

use crate::error::VerifyError;
use crate::graph::ParamBinding;
use crate::status::{ParamInputRecord, SmtStatusRecord, VerifyStatus};
use crate::verify::{KernelVerification, VerifyConfig};
use crate::verify_input::ScalarInputBounds;
use crate::verify_request::VerifyRequest;
#[cfg(feature = "ay-smt")]
use crate::ay::{verify_kernel_smt_multi, verify_kernel_smt_with_bounds, TranslatedKernel};

#[path = "pipeline_tensor.rs"]
pub mod tensor;
pub use tensor::{
    verify_tensor_and_record, verify_tensor_and_record_with_config, verify_with_selective_crown,
    SelectiveCrownResult, TensorPipelineResult,
};

#[path = "pipeline_fusion.rs"]
pub mod fusion;
pub use fusion::{
    certify_auto_fusion_from_graph, verify_and_record_auto_fusion_from_graph,
    verify_auto_fusion_from_graph, verify_fusion_and_record, verify_fusion_and_record_with_config,
    verify_fusion_certificate, AutoFusionPipelineResult, CertifiedFusionResult,
    FusionPipelineResult,
};

/// Result of a full verification pipeline run (NY + ay).
#[derive(Debug)]
#[non_exhaustive]
pub struct PipelineResult {
    /// NY bounds verification result.
    pub gamma_crown: KernelVerification,
    /// ay SMT cross-verification result.
    pub smt: SmtStatusRecord,
    /// Whether ay received NY output bounds for cross-verification.
    /// `false` when NY produced non-finite bounds (`is_finite == false`),
    /// meaning ay ran without bounds to check against (#428).
    pub cross_verified: bool,
}

/// Run the full verification pipeline for a single-variable kernel:
/// NY bounds → record → ay SMT (with cross-verification) → record_smt.
///
/// Uses `VerifyConfig::default()`. For custom configuration (escalation
/// thresholds, soundness mode), use [`verify_and_record_full_with_config`].
///
/// `status_key` overrides the key used in `nn_verify_status.json`. When
/// `None`, `kernel.name` is used. This allows `verify_all` to record
/// distinct config names (e.g. `"adain_scaled"`) without mutating
/// `kernel.name`, which must remain the base name for BOUNDS_REGISTRY
/// dispatch (#521).
///
/// # Errors
///
/// Returns an error if NY verification, ay encoding, or status
/// recording fails. Both stages propagate errors immediately.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_and_record_full(
    status: &mut VerifyStatus,
    kernel: &KernelDef,
    constant_params: &[f32],
    bounds: ScalarInputBounds,
    status_key: Option<&str>,
) -> Result<PipelineResult, VerifyError> {
    verify_and_record_full_with_config(
        status,
        kernel,
        constant_params,
        bounds,
        status_key,
        &VerifyConfig::default(),
    )
}

/// Run the full verification pipeline for a single-variable kernel with
/// explicit [`VerifyConfig`].
///
/// Same as [`verify_and_record_full`] but accepts a [`VerifyConfig`] to
/// control escalation thresholds, soundness mode, and normalization bounds
/// mode. This is the scalar equivalent of
/// [`verify_tensor_and_record_with_config`].
///
/// # Errors
///
/// Returns an error if NY verification, ay encoding, or status
/// recording fails. Both stages propagate errors immediately.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_and_record_full_with_config(
    status: &mut VerifyStatus,
    kernel: &KernelDef,
    constant_params: &[f32],
    bounds: ScalarInputBounds,
    status_key: Option<&str>,
    config: &VerifyConfig,
) -> Result<PipelineResult, VerifyError> {
    let key = status_key.unwrap_or(&kernel.name);
    let input_bounds = crate::verify_input::scalar_input_bounds(bounds.lower(), bounds.upper())?;

    // 1. NY bounds verification
    let gamma_crown = VerifyRequest::new(kernel)
        .config(config.clone())
        .constant_params(constant_params)
        .input_bounds(&input_bounds)
        .verify_bounds()?;

    // 2. Record NY result (using status_key for the entry name)
    status.record(&gamma_crown, bounds, constant_params, Some(key))?;

    // 3. ay SMT cross-verification via TranslatedKernel (#719) — pass
    //    NY output bounds so ay can check consistency.
    //    Uses TranslatedKernel for the ay step; the translated program
    //    supports push/pop incremental solving for future multi-property
    //    verification without re-translating.
    #[cfg(feature = "ay-smt")]
    let (smt, cross_verified) = {
        let cross_verified = gamma_crown.is_finite;
        let smt = if cross_verified {
            let expected = (
                f64::from(gamma_crown.output_lower),
                f64::from(gamma_crown.output_upper),
            );
            let mut tk = TranslatedKernel::from_kernel(kernel, constant_params, bounds)?;
            tk.check_output_bounded(expected)?
        } else {
            // No finite output bounds from NY — fall back to single-shot
            // path with heuristic bounds. TranslatedKernel requires explicit
            // caller-provided bounds, so the heuristic path stays on the old API.
            verify_kernel_smt_with_bounds(kernel, constant_params, bounds, None)?
        };

        // 4. Record SMT result (attaches to the NY entry, using status_key)
        status.record_smt(key, smt.clone())?;
        (smt, cross_verified)
    };

    #[cfg(not(feature = "ay-smt"))]
    let (smt, cross_verified) = (
        SmtStatusRecord::execution_failed("ay-smt feature disabled"),
        false,
    );

    Ok(PipelineResult {
        gamma_crown,
        smt,
        cross_verified,
    })
}

/// Run the full verification pipeline for a multi-variable kernel (#411):
/// NY bounds → record_with_variable_inputs → ay SMT → record_smt.
///
/// Uses `VerifyConfig::default()`. For custom configuration (escalation
/// thresholds, soundness mode), use [`verify_and_record_full_multi_with_config`].
///
/// `bindings` maps each kernel parameter to `Variable` or `Constant(val)`.
/// `variable_bounds` provides `(lower, upper)` for each `Variable` binding,
/// in the order they appear in `bindings`.
///
/// `status_key` overrides the key used in `nn_verify_status.json` (#521).
/// When `None`, `kernel.name` is used.
///
/// # Errors
///
/// Returns an error if NY verification, ay encoding, or status
/// recording fails. Both stages propagate errors immediately.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_and_record_full_multi(
    status: &mut VerifyStatus,
    kernel: &KernelDef,
    bindings: &[ParamBinding],
    variable_bounds: &[(f32, f32)],
    status_key: Option<&str>,
) -> Result<PipelineResult, VerifyError> {
    verify_and_record_full_multi_with_config(
        status,
        kernel,
        bindings,
        variable_bounds,
        status_key,
        &VerifyConfig::default(),
    )
}

/// Run the full verification pipeline for a multi-variable kernel with
/// explicit [`VerifyConfig`].
///
/// Same as [`verify_and_record_full_multi`] but accepts a [`VerifyConfig`] to
/// control escalation thresholds, soundness mode, and normalization bounds
/// mode. This is the scalar multi-variable equivalent of
/// [`verify_tensor_and_record_with_config`].
///
/// # Errors
///
/// Returns an error if NY verification, ay encoding, or status
/// recording fails. Both stages propagate errors immediately.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_and_record_full_multi_with_config(
    status: &mut VerifyStatus,
    kernel: &KernelDef,
    bindings: &[ParamBinding],
    variable_bounds: &[(f32, f32)],
    status_key: Option<&str>,
    config: &VerifyConfig,
) -> Result<PipelineResult, VerifyError> {
    let key = status_key.unwrap_or(&kernel.name);

    // 1. NY bounds verification (multi-variable path)
    let gamma_crown = VerifyRequest::new(kernel)
        .config(config.clone())
        .bindings(bindings)
        .variable_bounds(variable_bounds)
        .verify_bounds()?;

    // 2. Record NY result with per-variable metadata
    let variable_inputs: Vec<ParamInputRecord> = bindings
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b, ParamBinding::Variable))
        .zip(variable_bounds.iter())
        .map(|((idx, _), &(lo, hi))| ParamInputRecord {
            param_index: idx,
            lower: lo,
            upper: hi,
        })
        .collect();

    let constant_params: Vec<f32> = bindings
        .iter()
        .filter_map(|b| match b {
            ParamBinding::Constant(v) => Some(*v),
            ParamBinding::Variable => None,
        })
        .collect();

    status.record_with_variable_inputs(
        &gamma_crown,
        &variable_inputs,
        &constant_params,
        Some(key),
        None, // scalar kernel — no tensor shape
    )?;

    // 3. ay SMT cross-verification — pass NY output bounds
    #[cfg(feature = "ay-smt")]
    let (smt, cross_verified) = {
        let expected_output_bounds = if gamma_crown.is_finite {
            Some((
                f64::from(gamma_crown.output_lower),
                f64::from(gamma_crown.output_upper),
            ))
        } else {
            None
        };
        let cross_verified = expected_output_bounds.is_some();
        let smt =
            verify_kernel_smt_multi(kernel, bindings, variable_bounds, expected_output_bounds)?;

        // 4. Record SMT result (using status_key)
        status.record_smt(key, smt.clone())?;
        (smt, cross_verified)
    };

    #[cfg(not(feature = "ay-smt"))]
    let (smt, cross_verified) = (
        SmtStatusRecord::execution_failed("ay-smt feature disabled"),
        false,
    );

    Ok(PipelineResult {
        gamma_crown,
        smt,
        cross_verified,
    })
}
