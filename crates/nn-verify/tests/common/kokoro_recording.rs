// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared Kokoro verification recording helpers.
//!
//! Replaces 5 identical `record_traced_*` functions across compose test files
//! with a single `record_ibp_result`. All traced Kokoro tests use IBP-only
//! propagation (`propagate_multi_input_ibp`), so PropMethod is always IBP and
//! soundness is Heuristic (normalization layers use heuristic approximation).
//!
//! Part of #2623, Part of #2218.

use nn_tts_verify::PipelineCertificate;
use nn_verify::{
    model_for_kernel, model_status_path, BoundedTensor, PropMethod, VerificationSoundnessMode,
    VerifyStatus,
};
use std::path::Path;

/// Record IBP propagation result to the per-model status file.
///
/// Shared helper for all traced Kokoro compose tests that use
/// `propagate_multi_input_ibp`. Hardcodes `PropMethod::Ibp` and
/// `VerificationSoundnessMode::Heuristic` because:
/// - All callers use IBP-only propagation (multi-input graph, no CROWN).
/// - All Kokoro sub-graphs contain normalization layers (InstanceNorm/AdaIN),
///   which force heuristic approximation in NY.
///
/// If a caller upgrades to CROWN propagation, it should use `verify_and_assert`
/// from `common/mod.rs` instead of this helper.
#[allow(dead_code)]
pub(crate) fn record_ibp_result(
    status_key: &str,
    input_bounds: &BoundedTensor,
    output: &BoundedTensor,
) {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let model = model_for_kernel(status_key);
    let model_path = model_status_path(ws, model);
    let mut locked = VerifyStatus::load_locked(&model_path).expect("load_locked");

    let (in_lo, in_hi) = super::bounds_min_max(input_bounds);
    let (out_lo, out_hi) = super::bounds_min_max(output);
    let (lo_arr, _) = output.lower_upper();
    let out_shape = [lo_arr.len()];

    locked
        .status
        .record_pipeline(
            status_key,
            PropMethod::Ibp,
            in_lo,
            in_hi,
            out_lo,
            out_hi,
            &out_shape,
            VerificationSoundnessMode::Heuristic,
            Some(input_bounds.shape()),
        )
        .expect("record_pipeline");
    locked
        .status
        .set_soundness_justification(
            status_key,
            "IBP propagation with heuristic normalization approximation",
        )
        .expect("set justification");
    locked.save().expect("save status");
    eprintln!("Recorded {status_key} to status file (stale=false)");
}

/// Record IBP propagation result with an explicit soundness mode.
///
/// Like `record_ibp_result`, but allows the caller to specify soundness.
/// Use for Conservative-mode re-verification where normalization layers
/// use `forward_mode: false` and produce provably sound bounds.
///
/// Part of #3422 D3, #3351 T3.1.
#[allow(dead_code)]
pub(crate) fn record_ibp_result_with_soundness(
    status_key: &str,
    input_bounds: &BoundedTensor,
    output: &BoundedTensor,
    soundness: VerificationSoundnessMode,
    justification: &str,
) {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let model = model_for_kernel(status_key);
    let model_path = model_status_path(ws, model);
    let mut locked = VerifyStatus::load_locked(&model_path).expect("load_locked");

    let (in_lo, in_hi) = super::bounds_min_max(input_bounds);
    let (out_lo, out_hi) = super::bounds_min_max(output);
    let (lo_arr, _) = output.lower_upper();
    let out_shape = [lo_arr.len()];

    locked
        .status
        .record_pipeline(
            status_key,
            PropMethod::Ibp,
            in_lo,
            in_hi,
            out_lo,
            out_hi,
            &out_shape,
            soundness,
            Some(input_bounds.shape()),
        )
        .expect("record_pipeline");
    locked
        .status
        .set_soundness_justification(status_key, justification)
        .expect("set justification");
    locked.save().expect("save status");
    eprintln!("Recorded {status_key} to status file (soundness={soundness:?})");
}

fn stage_method_is_tight(method: &str) -> bool {
    matches!(
        method.trim().to_ascii_uppercase().as_str(),
        "CROWN" | "ALPHACROWN" | "ALPHA-CROWN" | "BETACROWN" | "BETA-CROWN" | "ANALYTICAL"
    )
}

#[allow(dead_code)]
pub(crate) fn pipeline_tight_stage_count(cert: &PipelineCertificate) -> usize {
    cert.stages
        .iter()
        .filter(|stage| stage_method_is_tight(&stage.method))
        .count()
}

#[allow(dead_code)]
pub(crate) fn pipeline_crown_coverage(cert: &PipelineCertificate) -> f32 {
    if cert.stages.is_empty() {
        0.0
    } else {
        pipeline_tight_stage_count(cert) as f32 / cert.stages.len() as f32
    }
}

/// Collapse a pipeline certificate's per-stage method provenance into one
/// status-level propagation method.
///
/// Pure tight-stage pipelines are recorded as `CROWN`, pure IBP pipelines as
/// `IBP`, and mixed pipelines as `mixed_IBP_CROWN` so status entries cannot
/// silently overstate end-to-end CROWN coverage.
#[allow(dead_code)]
pub(crate) fn pipeline_prop_method(cert: &PipelineCertificate) -> PropMethod {
    let tight_stages = pipeline_tight_stage_count(cert);
    if tight_stages == 0 {
        PropMethod::Ibp
    } else if tight_stages == cert.stages.len() {
        PropMethod::Crown
    } else {
        PropMethod::MixedIbpCrown
    }
}

/// Record a pipeline certificate to status using the actual stage-method mix.
#[allow(dead_code)]
pub(crate) fn record_pipeline_certificate(
    status: &mut VerifyStatus,
    status_key: &str,
    cert: &PipelineCertificate,
    output_shape: &[usize],
    input_shape: Option<&[usize]>,
) -> PropMethod {
    let method = pipeline_prop_method(cert);
    let soundness = if cert.is_sound {
        VerificationSoundnessMode::Sound
    } else {
        VerificationSoundnessMode::Heuristic
    };
    let in_lo = cert
        .e2e_input_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min) as f32;
    let in_hi = cert
        .e2e_input_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max) as f32;
    let out_lo = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min) as f32;
    let out_hi = cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max) as f32;
    status
        .record_pipeline(
            status_key,
            method,
            in_lo,
            in_hi,
            out_lo,
            out_hi,
            output_shape,
            soundness,
            input_shape,
        )
        .expect("record_pipeline");
    method
}
