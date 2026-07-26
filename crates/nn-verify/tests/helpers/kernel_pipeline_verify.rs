// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! Integration test: verify all 10 kernel types through the NY + ay
//! unified pipeline and persist results to `nn_verify_status.json`.
//!
//! Coverage: IBP-only, full pipeline (NY + ay), CROWN escalation,
//! multi-variable, status_key deduplication, and cross_verified invariant.
//!
//! Split into sub-modules to stay under 500 lines (#533):
//! - `kernel_pipeline_individual.rs` — per-kernel IBP and full pipeline tests
//! - `kernel_pipeline_advanced.rs` — CROWN, multi-variable, status_key, cross_verified

use nn_dsl::adain::{build_adain_scalar_kernel, build_adain_snake_fused_kernel};
use nn_dsl::instance_norm::{
    build_instance_norm_affine_scalar_kernel, build_instance_norm_scalar_kernel,
};
use nn_dsl::layer_norm::build_layer_norm_scalar_kernel;
use nn_dsl::rms_norm::build_rms_norm_scalar_kernel;
use nn_dsl::rope::{build_rope_cos_kernel, build_rope_sin_kernel};
use nn_dsl::silu_mul::build_silu_mul_kernel;
use nn_verify::{
    scalar_input_bounds, verify_and_record_full, verify_and_record_full_multi,
    verify_and_record_full_multi_with_config, verify_and_record_full_with_config,
    KernelVerification, ParamBinding, PropMethod, ScalarInputBounds, VerifyConfig, VerifyRequest,
    VerifyStatus,
};

use super::common;

/// Verify a single-variable kernel and record to status.
/// Returns the verification result for further assertions.
fn verify_and_record(
    status: &mut VerifyStatus,
    kernel: &nn_dsl::ir::KernelDef,
    constant_params: &[f32],
    lo: f32,
    hi: f32,
) -> KernelVerification {
    let input_bounds = scalar_input_bounds(lo, hi).expect("input bounds");
    let result = VerifyRequest::new(kernel)
        .constant_params(constant_params)
        .input_bounds(&input_bounds)
        .verify_bounds()
        .unwrap_or_else(|e| panic!("IBP failed for {}: {e}", kernel.name));
    assert!(
        result.is_finite,
        "{}: IBP bounds not finite (lo={:?}, hi={:?})",
        kernel.name, result.output_lower, result.output_upper
    );
    status
        .record(
            &result,
            ScalarInputBounds::new(lo, hi).unwrap(),
            constant_params,
            None,
        )
        .unwrap_or_else(|e| panic!("record {} failed: {e}", kernel.name));
    result
}

/// Assert that IBP bounds are sound: they contain the expected analytical range.
///
/// `expected_lo` / `expected_hi` are the true min/max of the kernel over the input
/// domain (computed analytically). IBP bounds must contain this range (soundness)
/// and should not be absurdly wide (max_width).
fn assert_sound_bounds(
    result: &KernelVerification,
    expected_lo: f32,
    expected_hi: f32,
    max_width: f32,
) {
    assert!(
        result.output_lower <= expected_lo,
        "{}: IBP lower ({}) must be <= true min ({}) for soundness",
        result.kernel_name,
        result.output_lower,
        expected_lo
    );
    assert!(
        result.output_upper >= expected_hi,
        "{}: IBP upper ({}) must be >= true max ({}) for soundness",
        result.kernel_name,
        result.output_upper,
        expected_hi
    );
    assert!(
        result.output_width <= max_width,
        "{}: IBP width ({}) exceeds sanity threshold ({})",
        result.kernel_name,
        result.output_width,
        max_width
    );
    assert_eq!(
        result.method,
        PropMethod::Ibp,
        "{}: expected IBP method for these simple inputs",
        result.kernel_name
    );
}

// --- Combined pipeline: verify all and persist ---

/// Verify all 10 kernel types and record into `status` (#450).
/// Constant params match `verify_all.rs`.
fn verify_all_kernels(status: &mut VerifyStatus) {
    let s = &common::snake_kernel();
    let _ = verify_and_record(status, s, &[1.0], -10.0, 10.0);
    let k = build_silu_mul_kernel().expect("build silu_mul");
    let _ = verify_and_record(status, &k, &[2.0], -5.0, 5.0);
    let k = build_adain_scalar_kernel().expect("build adain");
    let _ = verify_and_record(status, &k, &[0.0, 1.0, 1.0, 0.0, 1e-5], -5.0, 5.0);
    let k = build_adain_snake_fused_kernel().expect("build adain_snake");
    let _ = verify_and_record(status, &k, &[0.0, 1.0, 1.0, 0.0, 1.0, 1e-5], -5.0, 5.0);
    let k = build_rope_cos_kernel().expect("build rope_cos");
    let _ = verify_and_record(status, &k, &[1.0, 0.5], -10.0, 10.0);
    let k = build_rope_sin_kernel().expect("build rope_sin");
    let _ = verify_and_record(status, &k, &[1.0, 0.5], -10.0, 10.0);
    let k = build_layer_norm_scalar_kernel().expect("build layer_norm");
    let _ = verify_and_record(status, &k, &[0.0, 1.0, 1e-5, 1.0, 0.0], -5.0, 5.0);
    let k = build_rms_norm_scalar_kernel().expect("build rms_norm");
    let _ = verify_and_record(status, &k, &[1.0, 1.0], -5.0, 5.0);
    let k = build_instance_norm_scalar_kernel().expect("build instance_norm");
    let _ = verify_and_record(status, &k, &[0.0, 1.0, 1e-5], -5.0, 5.0);
    let k = build_instance_norm_affine_scalar_kernel().expect("build instance_norm_affine");
    let _ = verify_and_record(status, &k, &[0.0, 1.0, 1e-5, 1.0, 0.0], -5.0, 5.0);
}

#[test]
fn test_pipeline_verify_all_and_persist() {
    // Lock held across full load-modify-save cycle (#482, #530 TOCTOU fix).
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    // Per-model status files (#2577): shared kernels go to shared model file.
    let status_path = nn_verify::model_status_path(workspace_root, "shared");
    let mut locked = VerifyStatus::load_locked(&status_path).expect("load_locked");
    let pre_existing = locked.status.kernel_count();

    verify_all_kernels(&mut locked.status);

    // Persist (merge, not overwrite; lock held across full cycle)
    locked.save().expect("save per-model status");
    drop(locked);

    // Validate: all 10 pipeline kernels present (#450, #476).
    // Locked read to avoid TOCTOU race with parallel tests.
    let validation = VerifyStatus::load_locked(&status_path).expect("load_locked validation");
    assert!(
        validation.status.kernel_count() >= 10,
        "expected at least 10 kernels, got {}",
        validation.status.kernel_count()
    );
    // Pre-existing entries must not have been destroyed
    assert!(
        validation.status.kernel_count() >= pre_existing,
        "load-then-save must not destroy pre-existing entries: had {pre_existing}, now {}",
        validation.status.kernel_count()
    );
    for name in &[
        "snake",
        "silu_mul",
        "adain",
        "adain_snake",
        "rope_cos",
        "rope_sin",
        "layer_norm_scalar",
        "rms_norm_scalar",
        "instance_norm_scalar",
        "instance_norm_affine_scalar",
    ] {
        assert!(
            validation.status.has_kernel(name),
            "{name} missing from status"
        );
        let entry = validation
            .status
            .kernel(name)
            .unwrap_or_else(|| panic!("{name} should exist"));
        assert_eq!(
            entry.status,
            nn_verify::VerifyOutcome::Verified,
            "{name} should be Verified"
        );
    }
}

// Per-kernel individual tests (IBP-only + full pipeline)
#[path = "../kernel_pipeline/individual.rs"]
mod individual;

// Advanced tests (CROWN, multi-variable, status_key, cross_verified)
#[path = "../kernel_pipeline/advanced.rs"]
mod advanced;
