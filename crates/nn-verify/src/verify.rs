// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kernel bounds verification via NY propagation.
//!
//! Primary entry point: [`VerifyRequest`](crate::VerifyRequest) builder. Set
//! parameters via chained setters, then call `verify_bounds()` or `verify_spec()`.
//!
//! Types (`PropMethod`, `KernelVerification`, `SpecVerification`, `VerifyConfig`)
//! live in [`verify_types`](crate::verify_types).

use ny_api::{Bound, BoundedTensor, NyError, VerificationResult, VerificationSpec};
use ny_core::{nan_propagating_max, VerificationSoundnessMode};
use ny_propagate::{PropagationConfig, PropagationMethod, Verifier};

use crate::error::VerifyError;
use crate::graph::ParamBinding;
use crate::soundness::soundness_for_graph;
use crate::util::finite_or;
use crate::verify_input::verification_spec_from_tensors;

// Re-export types so `use crate::verify::PropMethod` etc. still works.
pub use crate::verify_types::{
    KernelVerification, NormBoundsMode, OutputTensorBounds, PropMethod, SpecVerification,
    VerifyConfig,
};

use ny_propagate::bounds::AlphaCrownConfig;
use std::time::{Duration, Instant};

/// Default per-propagation alpha-CROWN wall-clock budget (seconds). The
/// 100-iteration optimizer checks this deadline each iteration and bails to its
/// best-bounds-so-far — which is sound — when exceeded, so deep document models
/// terminate cleanly instead of running the optimizer unbounded (>240s test
/// budget). Override with `NN_VERIFY_CROWN_DEADLINE_SECS` (0 disables it).
const DEFAULT_CROWN_DEADLINE_SECS: u64 = 20;

/// Wall-clock deadline for one alpha-CROWN propagation, or `None` to run all
/// iterations (when `NN_VERIFY_CROWN_DEADLINE_SECS=0`).
pub(crate) fn crown_deadline() -> Option<Instant> {
    let secs = std::env::var("NN_VERIFY_CROWN_DEADLINE_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_CROWN_DEADLINE_SECS);
    (secs > 0).then(|| Instant::now() + Duration::from_secs(secs))
}

/// `AlphaCrownConfig::default()` with [`crown_deadline`] applied. Bounds returned
/// after the deadline are the optimizer's best-so-far, hence still sound.
#[allow(clippy::field_reassign_with_default)]
pub(crate) fn alpha_config_with_deadline() -> AlphaCrownConfig {
    let mut config = AlphaCrownConfig::default();
    config.deadline = crown_deadline();
    config
}

/// Build bindings for the single-variable API: param 0 = Variable, rest = Constant.
pub(crate) fn single_variable_bindings(constant_params: &[f32]) -> Vec<ParamBinding> {
    let mut bindings = vec![ParamBinding::Variable];
    for &val in constant_params {
        bindings.push(ParamBinding::Constant(val));
    }
    bindings
}

/// Run IBP with optional CROWN escalation on a pre-built graph, returning
/// a `KernelVerification`. Shared implementation for single- and multi-variable paths.
///
/// ## CROWN escalation for single-kernel graphs (#489)
///
/// For kernels with native layers (e.g., Snake via [`SnakeLayer`]), IBP already
/// computes exact bounds via monotonicity, so CROWN provides no tightening.
/// CROWN is most valuable for multi-layer or multi-path graphs (fusion diamond
/// DAGs) where IBP loses inter-path input correlation. The default escalation
/// threshold (1e6) avoids unnecessary CROWN calls for already-tight IBP results.
///
/// `uses_comparison_approximation`: whether the graph was translated from IR
/// containing variable-operand Compare nodes (continuous approximation).
pub(crate) fn run_escalation(
    graph: &ny_propagate::GraphNetwork,
    input_bounds: &BoundedTensor,
    kernel_name: &str,
    config: &VerifyConfig,
    uses_comparison_approximation: bool,
) -> Result<(KernelVerification, BoundedTensor), VerifyError> {
    // Phase 1: IBP (fast, sound, potentially loose)
    let ibp_output = graph.propagate_ibp(input_bounds)?;
    let ibp_width = ibp_output.max_width();

    let (method, output_bounds, crown_fallback_reason) =
        if ibp_width <= config.escalation_threshold() {
            (PropMethod::Ibp, ibp_output, None)
        } else {
            // Phase 2: CROWN-family escalation (slower, tighter)
            //
            // GraphNetwork::propagate_crown() already tries alpha-CROWN first,
            // but it only returns bounds, not the verifier-level method that
            // actually succeeded. We mirror that upgrade path explicitly here so
            // KernelVerification.method can report AlphaCrown/Crown/Ibp
            // accurately instead of collapsing every successful upgrade to Crown.
            match graph.propagate_alpha_crown_with_config(input_bounds, &alpha_config_with_deadline()) {
                Ok(alpha_output) => (PropMethod::AlphaCrown, alpha_output, None),
                Err(alpha_err) => {
                    if crown_error_must_propagate(&alpha_err) {
                        return Err(alpha_err.into());
                    }

                    match graph.propagate_crown_with_provenance(input_bounds) {
                        Ok(crown_result) => {
                            if crown_result.is_fallback() {
                                let reason = format!(
                                    "alpha-CROWN failed: {alpha_err}; fixed-slope CROWN fell \
                                     back to IBP internally"
                                );
                                (PropMethod::Ibp, crown_result.bounds, Some(reason))
                            } else {
                                (PropMethod::Crown, crown_result.bounds, None)
                            }
                        }
                        Err(crown_err) => {
                            if crown_error_must_propagate(&crown_err) {
                                return Err(crown_err.into());
                            }

                            // CROWN-family propagation failed — fall back to IBP but record the
                            // failure reason. This surfaces the distinction between
                            // "escalation not attempted" and "escalation attempted, IBP used."
                            let reason = format!(
                                "alpha-CROWN failed: {alpha_err}; fixed-slope CROWN failed: \
                                 {crown_err}"
                            );
                            (PropMethod::Ibp, ibp_output, Some(reason))
                        }
                    }
                }
            }
        };

    let (out_lower, out_upper) = crate::util::bounds_min_max(&output_bounds);
    let raw_width = out_upper - out_lower;
    // Guard: serde_json cannot serialize NaN/Infinity. Check width too —
    // f32::MAX - (-f32::MAX) overflows to Inf even when both bounds are finite.
    let is_finite = out_lower.is_finite() && out_upper.is_finite() && raw_width.is_finite();
    let out_width = if is_finite { raw_width } else { f32::MAX };

    let provenance = soundness_for_graph(
        graph,
        &method,
        Some(input_bounds),
        uses_comparison_approximation,
    )?;
    let soundness_mode = provenance.mode();

    if config.require_sound() && soundness_mode == VerificationSoundnessMode::Heuristic {
        return Err(VerifyError::SoundnessRequired {
            kernel_name: kernel_name.to_string(),
        });
    }

    let verification = KernelVerification {
        kernel_name: kernel_name.to_string(),
        method,
        output_lower: finite_or(out_lower, 0.0),
        output_upper: finite_or(out_upper, 0.0),
        output_width: out_width,
        is_finite,
        output_tensor: Some(OutputTensorBounds::from_bounded_tensor(&output_bounds)),
        crown_fallback_reason,
        soundness_mode,
    };
    Ok((verification, output_bounds))
}

pub(crate) fn verify_graph_against_spec_with_config(
    graph: &ny_propagate::GraphNetwork,
    input_bounds: &BoundedTensor,
    required_output_bounds: &[Bound],
    config: &VerifyConfig,
    kernel_name: &str,
) -> Result<SpecVerification, VerifyError> {
    let spec = verification_spec_from_tensors(input_bounds, required_output_bounds)?;

    let ibp_result = run_graph_verification(graph, &spec, PropagationMethod::Ibp)?;
    let ibp_method = resolve_result_method(&ibp_result, PropMethod::Ibp);
    let (method, result, crown_fallback_reason) =
        if should_escalate_to_crown(&ibp_result, config.escalation_threshold()) {
            match run_graph_verification(graph, &spec, PropagationMethod::Crown) {
                Ok(crown_result) => (
                    resolve_result_method(&crown_result, PropMethod::Crown),
                    crown_result,
                    None,
                ),
                Err(crown_err) => (ibp_method, ibp_result, Some(crown_err.to_string())),
            }
        } else {
            (ibp_method, ibp_result, None)
        };

    if config.require_sound() && result.provenance().mode() == VerificationSoundnessMode::Heuristic
    {
        return Err(VerifyError::SoundnessRequired {
            kernel_name: kernel_name.to_string(),
        });
    }

    Ok(SpecVerification {
        result,
        method,
        crown_fallback_reason,
    })
}

fn run_graph_verification(
    graph: &ny_propagate::GraphNetwork,
    spec: &VerificationSpec,
    method: PropagationMethod,
) -> Result<VerificationResult, VerifyError> {
    let verifier = Verifier::new(PropagationConfig {
        method,
        ..Default::default()
    });
    Ok(verifier.verify_graph(graph, spec)?)
}

fn resolve_result_method(result: &VerificationResult, requested: PropMethod) -> PropMethod {
    result
        .actual_method_tag()
        .and_then(PropMethod::from_method_used)
        .unwrap_or(requested)
}

fn crown_error_must_propagate(error: &NyError) -> bool {
    matches!(
        error,
        NyError::SoundnessRefusal(_) | NyError::InternalError(_)
    )
}

fn should_escalate_to_crown(result: &VerificationResult, threshold: f32) -> bool {
    // NaN max-width conservatively triggers escalation (IEEE 754: NaN > x is false,
    // so without this guard, all-NaN bounds would silently skip escalation).
    let exceeds_threshold = |max_width: f32| max_width.is_nan() || max_width > threshold;

    match result {
        VerificationResult::Unknown { bounds, .. } => {
            let max_width = bounds
                .iter()
                .map(Bound::width)
                .fold(0.0f32, nan_propagating_max);
            exceeds_threshold(max_width)
        }
        VerificationResult::Timeout { partial_bounds, .. } => {
            // Timeouts should escalate — the initial method couldn't complete.
            // If partial bounds exist, check width; otherwise always escalate.
            partial_bounds.as_ref().map_or(true, |bounds| {
                let max_width = bounds
                    .iter()
                    .map(Bound::width)
                    .fold(0.0f32, nan_propagating_max);
                exceeds_threshold(max_width)
            })
        }
        VerificationResult::Verified { .. } | VerificationResult::Violated { .. } => false,
    }
}

#[cfg(test)]
#[path = "verify_tests.rs"]
mod tests;
