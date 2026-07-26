// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: verify Snake kernel bounds (IBP + CROWN escalation) and
//! persist to nn_verify_status.json.

use nn_verify::{
    scalar_input_bounds, ScalarInputBounds, VerifyConfig, VerifyRequest, VerifyStatus,
};

use super::common;

fn verify_ibp_alphas(
    kernel: &nn_dsl::ir::KernelDef,
    status: &mut VerifyStatus,
    x_lo: f32,
    x_hi: f32,
) {
    let alphas: &[f32] = &[0.01, 0.1, 1.0, 10.0, 100.0];
    let input_bounds = scalar_input_bounds(x_lo, x_hi).expect("input bounds");
    for alpha in alphas {
        let result = VerifyRequest::new(kernel)
            .constant_params(&[*alpha])
            .input_bounds(&input_bounds)
            .verify_bounds()
            .unwrap_or_else(|e| panic!("IBP verification failed for alpha={alpha}: {e}"));
        assert!(result.is_finite, "IBP bounds not finite for alpha={alpha}");

        let key = format!("snake_alpha_{alpha}");
        status
            .record(
                &result,
                ScalarInputBounds::new(x_lo, x_hi).expect("valid test bounds"),
                &[*alpha],
                Some(&key),
            )
            .expect("record");
    }

    // Canonical entry with alpha=1.0
    let canonical = VerifyRequest::new(kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .verify_bounds()
        .expect("canonical IBP verification");
    status
        .record(
            &canonical,
            ScalarInputBounds::new(x_lo, x_hi).expect("valid test bounds"),
            &[1.0],
            None,
        )
        .expect("record");
}

fn verify_crown_alphas(
    kernel: &nn_dsl::ir::KernelDef,
    status: &mut VerifyStatus,
    x_lo: f32,
    x_hi: f32,
) {
    let crown_config = VerifyConfig::with_threshold(5.0).expect("valid threshold");
    let input_bounds = scalar_input_bounds(x_lo, x_hi).expect("input bounds");

    for alpha in &[0.1f32, 1.0, 10.0] {
        let result = VerifyRequest::new(kernel)
            .constant_params(&[*alpha])
            .input_bounds(&input_bounds)
            .config(crown_config.clone())
            .verify_bounds()
            .unwrap_or_else(|e| panic!("CROWN verification failed for alpha={alpha}: {e}"));
        assert!(
            result.is_finite,
            "CROWN bounds not finite for alpha={alpha}"
        );

        let key = format!("snake_alpha_{alpha}_crown");
        status
            .record(
                &result,
                ScalarInputBounds::new(x_lo, x_hi).expect("valid test bounds"),
                &[*alpha],
                Some(&key),
            )
            .expect("record");
    }
}

/// Per-model status file path for snake kernel persistence (#2577).
fn status_file_path() -> std::path::PathBuf {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    nn_verify::model_status_path(workspace_root, "shared")
}

#[test]
fn test_snake_verify_and_persist() {
    let kernel = common::snake_kernel();
    let (x_lo, x_hi) = (-10.0f32, 10.0f32);

    // Lock held across full load-modify-save cycle (#482 TOCTOU fix).
    let status_path = status_file_path();
    let mut locked = VerifyStatus::load_locked(&status_path).expect("load_locked");
    let pre_existing = locked.status.kernel_count();

    verify_ibp_alphas(&kernel, &mut locked.status, x_lo, x_hi);
    verify_crown_alphas(&kernel, &mut locked.status, x_lo, x_hi);

    // Persist to workspace root (merge, not overwrite; lock held)
    locked.save().expect("save nn_verify_status.json");
    drop(locked);

    // Validate persisted file (locked read to avoid TOCTOU race with parallel tests).
    let validation = VerifyStatus::load_locked(&status_path).expect("load_locked validation");
    // 9 snake entries (6 IBP + 3 CROWN) plus any pre-existing entries
    assert!(
        validation.status.kernel_count() >= 9,
        "expected at least 9 entries (6 IBP + 3 CROWN), got {}",
        validation.status.kernel_count()
    );
    assert!(
        validation.status.kernel_count() >= pre_existing,
        "must not destroy pre-existing entries: had {pre_existing}, now {}",
        validation.status.kernel_count()
    );
    assert!(validation.status.has_kernel("snake"));
    assert!(validation.status.has_kernel("snake_alpha_1_crown"));

    // Only assert on snake entries written by this test — fusion and adain
    // entries from other tests may have different statuses (#450).
    // Use starts_with("snake") to exclude fusion_adain_snake and adain_snake_* entries.
    for (name, entry) in validation.status.kernels() {
        if name.starts_with("snake") {
            assert_eq!(
                entry.status,
                nn_verify::VerifyOutcome::Verified,
                "snake kernel {name} should be Verified"
            );
        }
    }
}
