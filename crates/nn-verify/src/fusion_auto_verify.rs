// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Batch verification of auto-generated fusion specs.
//!
//! Runs [`verify_fusion_equivalence`] on each [`AutoFusionSpec`] and
//! optionally records results in `nn_verify_status.json`.
//!
//! Results use `is_conclusive()` to distinguish CROWN proofs (tight) from
//! IBP fallback (vacuously wide for diamond DAGs). Only CROWN-conclusive
//! results should be treated as fusion equivalence evidence.

use crate::error::VerifyError;
use crate::fusion::{verify_fusion_equivalence, verify_fusion_equivalence_with_config};
use crate::fusion_auto::AutoFusionSpec;
use crate::fusion_spec::FusionVerification;
use crate::status::VerifyStatus;
use crate::verify::VerifyConfig;

/// Result of verifying a single auto-generated fusion spec.
#[derive(Debug)]
pub struct AutoFusionResult {
    /// The chain/pair name for status tracking.
    pub name: String,
    /// The verification result (or error).
    pub result: Result<FusionVerification, VerifyError>,
}

/// Verify all auto-generated fusion specs.
///
/// Returns a result for each spec. Uses CROWN for tight bounds with
/// IBP fallback if CROWN fails.
///
/// If `variable_bounds` length doesn't match a spec's input count,
/// default bounds of [-3.0, 3.0] are used for that spec.
pub fn verify_all_fusion_specs(
    specs: &[AutoFusionSpec],
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
) -> Vec<AutoFusionResult> {
    specs
        .iter()
        .map(|spec| {
            let bounds = resolve_bounds(spec, variable_bounds);
            let result = verify_fusion_equivalence(&spec.as_fusion_spec(), &bounds, epsilon);
            AutoFusionResult {
                name: spec.chain_name.clone(),
                result,
            }
        })
        .collect()
}

/// Verify all fusion specs with custom configuration.
pub fn verify_all_fusion_specs_with_config(
    specs: &[AutoFusionSpec],
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    config: &VerifyConfig,
) -> Vec<AutoFusionResult> {
    specs
        .iter()
        .map(|spec| {
            let bounds = resolve_bounds(spec, variable_bounds);
            let result = verify_fusion_equivalence_with_config(
                &spec.as_fusion_spec(),
                &bounds,
                epsilon,
                config,
            );
            AutoFusionResult {
                name: spec.chain_name.clone(),
                result,
            }
        })
        .collect()
}

/// Verify all fusion specs and record results in the status file.
///
/// Each result is recorded with key `"fusion_auto_{chain_name}"`.
/// Uses `record_fusion()` for consistent outcome classification
/// (Verified, IbpFallback, BoundsComputed, Failed).
///
/// # Errors
///
/// Individual verification errors are captured in each `AutoFusionResult`.
/// Status recording errors are propagated.
pub fn verify_and_record_auto_fusion(
    specs: &[AutoFusionSpec],
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    status: &mut VerifyStatus,
) -> Result<Vec<AutoFusionResult>, VerifyError> {
    let results = verify_all_fusion_specs(specs, variable_bounds, epsilon);

    for result in &results {
        if let Ok(ref verification) = result.result {
            let bounds = resolve_bounds_for_name(specs, &result.name, variable_bounds);
            let status_key = format!("fusion_auto_{}", result.name);
            status.record_fusion(verification, &bounds, Some(&status_key))?;
        }
    }

    Ok(results)
}

/// Resolve bounds for a spec: use provided bounds if length matches,
/// broadcast single-element bounds, otherwise generate defaults.
fn resolve_bounds(spec: &AutoFusionSpec, variable_bounds: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if variable_bounds.len() == spec.num_shared_inputs() {
        variable_bounds.to_vec()
    } else if variable_bounds.len() == 1 {
        // Broadcast single bounds to all inputs (certify_model path).
        vec![variable_bounds[0]; spec.num_shared_inputs()]
    } else {
        vec![(-3.0, 3.0); spec.num_shared_inputs()]
    }
}

/// Find the spec with the given name and resolve its bounds.
fn resolve_bounds_for_name(
    specs: &[AutoFusionSpec],
    name: &str,
    variable_bounds: &[(f32, f32)],
) -> Vec<(f32, f32)> {
    specs
        .iter()
        .find(|s| s.chain_name == name)
        .map(|s| resolve_bounds(s, variable_bounds))
        .unwrap_or_else(|| variable_bounds.to_vec())
}
