// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for kani_status.json → KaniProofRecord bridge.

use super::*;
use crate::certificate_types::KaniOutcome;
use std::collections::BTreeMap;

fn make_entry(status: &str) -> HarnessEntry {
    HarnessEntry {
        status: status.to_string(),
    }
}

fn sample_status() -> KaniStatusFile {
    let mut harnesses = BTreeMap::new();
    // Snake kernel: 5 passed, 1 timeout, 1 not_run
    harnesses.insert(
        "nn-dsl::snake_scalar_finite_for_bounded_inputs".to_string(),
        make_entry("passed"),
    );
    harnesses.insert(
        "nn-dsl::snake_scalar_safe_at_zero_alpha".to_string(),
        make_entry("passed"),
    );
    harnesses.insert(
        "nn-dsl::snake_scalar_monotone_lower_bound".to_string(),
        make_entry("passed"),
    );
    harnesses.insert(
        "nn-dsl::snake_scalar_bounds_are_sound".to_string(),
        make_entry("passed"),
    );
    harnesses.insert(
        "nn-dsl::snake_safety_relaxed_alpha_domain".to_string(),
        make_entry("passed"),
    );
    harnesses.insert(
        "nn-dsl::snake_scalar_build_no_panic".to_string(),
        make_entry("not_run"),
    );
    harnesses.insert(
        "nn-dsl::snake_tensor_build_no_panic".to_string(),
        make_entry("timeout"),
    );

    // SiLU kernel: all passed
    harnesses.insert(
        "nn-dsl::silu_mul_output_finite".to_string(),
        make_entry("passed"),
    );
    harnesses.insert(
        "nn-dsl::silu_mul_bounds_check".to_string(),
        make_entry("passed"),
    );

    // Unrelated harness (no kernel prefix match)
    harnesses.insert(
        "nn-metal::plan_elementwise_output_equals_total".to_string(),
        make_entry("passed"),
    );

    KaniStatusFile { harnesses }
}

#[test]
fn test_kani_record_for_kernel_snake() {
    let status = sample_status();
    let record = kani_record_for_kernel(&status, "snake").expect("should match snake harnesses");

    assert_eq!(record.harness_count, 7);
    // Has timeout and not_run, so aggregate is Timeout (worse than NotRun).
    // Actually: Failed > Timeout > NotRun > Passed. We have timeout + not_run → Timeout.
    assert_eq!(record.status, KaniOutcome::Timeout);
    assert!(record
        .properties
        .contains(&"bounds_preservation".to_string()));
    assert!(record.properties.contains(&"safety".to_string()));
    assert!(record.cbmc_version.is_none());
}

#[test]
fn test_kani_record_for_kernel_silu_mul_all_passed() {
    let status = sample_status();
    let record =
        kani_record_for_kernel(&status, "silu_mul").expect("should match silu_mul harnesses");

    assert_eq!(record.harness_count, 2);
    assert_eq!(record.status, KaniOutcome::Passed);
    assert!(record
        .properties
        .contains(&"bounds_preservation".to_string()));
    assert!(record.properties.contains(&"no_overflow".to_string()));
}

#[test]
fn test_kani_record_for_kernel_no_match() {
    let status = sample_status();
    let record = kani_record_for_kernel(&status, "nonexistent_kernel");
    assert!(record.is_none());
}

#[test]
fn test_kani_record_for_kernel_failed_overrides_all() {
    let mut status = sample_status();
    status.harnesses.insert(
        "nn-dsl::silu_mul_crash_test".to_string(),
        make_entry("failed"),
    );

    let record =
        kani_record_for_kernel(&status, "silu_mul").expect("should match silu_mul harnesses");
    assert_eq!(record.harness_count, 3);
    assert_eq!(record.status, KaniOutcome::Failed);
}

#[test]
fn test_kani_record_word_boundary_no_false_match() {
    // "snake" should NOT match "snake2_something" but SHOULD match "snake_something"
    let mut harnesses = BTreeMap::new();
    harnesses.insert("nn-dsl::snake_test".to_string(), make_entry("passed"));
    // This should NOT match "snake" because "2" follows without underscore
    harnesses.insert("nn-dsl::snake2_test".to_string(), make_entry("passed"));
    let status = KaniStatusFile { harnesses };

    let record = kani_record_for_kernel(&status, "snake").expect("should match");
    assert_eq!(record.harness_count, 1); // Only snake_test, not snake2_test
}

#[test]
fn test_kani_record_exact_name_match() {
    // A harness whose name exactly equals the kernel name (no suffix)
    let mut harnesses = BTreeMap::new();
    harnesses.insert("nn-dsl::relu".to_string(), make_entry("passed"));
    harnesses.insert("nn-dsl::relu_bounds".to_string(), make_entry("passed"));
    let status = KaniStatusFile { harnesses };

    let record = kani_record_for_kernel(&status, "relu").expect("should match");
    assert_eq!(record.harness_count, 2);
    assert_eq!(record.status, KaniOutcome::Passed);
}

#[test]
fn test_kani_record_properties_inferred_from_names() {
    let mut harnesses = BTreeMap::new();
    harnesses.insert(
        "nn-dsl::test_no_overflow_check".to_string(),
        make_entry("passed"),
    );
    harnesses.insert(
        "nn-dsl::test_no_nan_guard".to_string(),
        make_entry("passed"),
    );
    harnesses.insert(
        "nn-dsl::test_bounds_valid".to_string(),
        make_entry("passed"),
    );
    harnesses.insert(
        "nn-dsl::test_safety_proof".to_string(),
        make_entry("passed"),
    );
    let status = KaniStatusFile { harnesses };

    let record = kani_record_for_kernel(&status, "test").expect("should match");
    assert_eq!(record.harness_count, 4);
    let props = &record.properties;
    assert!(props.contains(&"no_overflow".to_string()));
    assert!(props.contains(&"no_nan".to_string()));
    assert!(props.contains(&"bounds_preservation".to_string()));
    assert!(props.contains(&"safety".to_string()));
}

#[test]
fn test_load_kani_status_real_file() {
    // Load the actual kani_status.json from the repo root.
    let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let path = repo_root.join("kani_status.json");
    if !path.exists() {
        // Skip if running in CI without the file.
        return;
    }

    let status = load_kani_status(&path).expect("load real kani_status.json");
    assert!(!status.harnesses.is_empty());

    // Snake should have at least 5 harnesses in the real file.
    let record = kani_record_for_kernel(&status, "snake");
    assert!(record.is_some(), "snake harnesses should exist");
    let record = record.unwrap();
    assert!(record.harness_count >= 5);
}

#[test]
fn test_load_kani_status_nonexistent() {
    let result = load_kani_status(Path::new("/nonexistent/kani_status.json"));
    assert!(result.is_err());
}

#[test]
fn test_kani_record_from_file_convenience() {
    let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let path = repo_root.join("kani_status.json");
    if !path.exists() {
        return;
    }

    let record = kani_record_from_file(&path, "snake").expect("should load");
    assert!(record.is_some());

    let record = kani_record_from_file(&path, "nonexistent_kernel_xyz").expect("should load");
    assert!(record.is_none());
}

#[test]
fn test_kani_record_unknown_status_treated_as_not_run() {
    let mut harnesses = BTreeMap::new();
    harnesses.insert("nn-dsl::foo_test".to_string(), make_entry("passed"));
    harnesses.insert(
        "nn-dsl::foo_bar".to_string(),
        make_entry("some_unknown_status"),
    );
    let status = KaniStatusFile { harnesses };

    let record = kani_record_for_kernel(&status, "foo").expect("should match");
    assert_eq!(record.harness_count, 2);
    assert_eq!(record.status, KaniOutcome::NotRun);
}
