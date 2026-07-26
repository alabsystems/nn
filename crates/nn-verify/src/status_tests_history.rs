// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! History retention and atomic persistence tests.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::test_helpers::bounds;

fn make_result(name: &str, method: PropMethod, lo: f32, hi: f32) -> KernelVerification {
    KernelVerification {
        kernel_name: name.to_string(),
        method,
        output_lower: lo,
        output_upper: hi,
        output_width: hi - lo,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "nn_verify_{}_{}_{}",
        prefix,
        std::process::id(),
        unique
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[test]
fn test_repeated_record_preserves_history() {
    let mut status = VerifyStatus::default();

    let result1 = make_result("snake", PropMethod::Ibp, -5.0, 5.0);
    status
        .record(&result1, bounds(-1.0, 1.0), &[1.0], None)
        .expect("record");

    let result2 = make_result("snake", PropMethod::Crown, -50.0, 50.0);
    status
        .record(&result2, bounds(-10.0, 10.0), &[1.0], None)
        .expect("record");

    assert_eq!(status.kernels["snake"].method, PropMethod::Crown);
    assert_eq!(status.kernels["snake"].output_width, 100.0);
    assert_eq!(status.run_count("snake"), 2);

    let runs = &status.history["snake"];
    assert_eq!(runs[0].method, PropMethod::Ibp);
    assert_eq!(runs[0].output_width, 10.0);
    assert_eq!(runs[1].method, PropMethod::Crown);
    assert_eq!(runs[1].output_width, 100.0);
}

#[test]
fn test_repeated_record_failure_preserves_history() {
    let mut status = VerifyStatus::default();

    let result = make_result("unstable", PropMethod::Ibp, -1.0, 1.0);
    status
        .record(&result, bounds(-1.0, 1.0), &[], None)
        .expect("record");
    status
        .record_failure("unstable", PropMethod::Ibp, bounds(-100.0, 100.0), &[])
        .expect("record failure");

    assert_eq!(status.kernels["unstable"].status, VerifyOutcome::Failed);
    assert_eq!(status.run_count("unstable"), 2);
    assert_eq!(
        status.history["unstable"][0].status,
        VerifyOutcome::Verified
    );
    assert_eq!(status.history["unstable"][1].status, VerifyOutcome::Failed);
}

#[test]
fn test_history_serializes_and_deserializes() {
    let mut status = VerifyStatus::default();
    let result = make_result("test_k", PropMethod::Ibp, -1.0, 1.0);
    status
        .record(&result, bounds(0.0, 1.0), &[], None)
        .expect("record");
    status
        .record(&result, bounds(0.0, 2.0), &[], None)
        .expect("record");

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    assert!(json.contains("\"history\""));

    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.run_count("test_k"), 2);
}

#[test]
fn test_legacy_status_without_history_loads_with_empty_history() {
    let json = r#"{
        "kernels": {
            "snake": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {
                    "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
                    "constant_params": [1.0],
                    "input_shape": [1],
                    "input_range": [-1.0, 1.0]
                },
                "output_bounds": {"lower": -5.0, "upper": 5.0},
                "output_width": 10.0
            }
        }
    }"#;

    let status: VerifyStatus = serde_json::from_str(json).expect("deserialize legacy");
    assert_eq!(status.kernels.len(), 1);
    assert!(
        status.history.is_empty(),
        "legacy files should have empty history"
    );
    assert_eq!(status.run_count("snake"), 0);
}

#[test]
fn test_atomic_save_produces_valid_file() {
    let tmp_dir = unique_temp_dir("atomic_save");
    let path = tmp_dir.join("test_status.json");

    let mut status = VerifyStatus::default();
    let result = make_result("atomic_test", PropMethod::Ibp, -1.0, 1.0);
    status
        .record(&result, bounds(0.0, 1.0), &[], None)
        .expect("record");
    status.save(&path).expect("atomic save");

    let loaded = VerifyStatus::load(&path).expect("load after atomic save");
    assert_eq!(loaded.kernels.len(), 1);
    assert_eq!(
        loaded.kernels["atomic_test"].status,
        VerifyOutcome::Verified
    );
    assert_eq!(loaded.run_count("atomic_test"), 1);

    std::fs::remove_dir_all(&tmp_dir).expect("remove temp dir");
}

#[test]
fn test_atomic_save_no_temp_file_left_behind() {
    let tmp_dir = unique_temp_dir("atomic_no_leftover");
    let path = tmp_dir.join("status.json");

    let status = VerifyStatus::default();
    status.save(&path).expect("save");

    let entries: Vec<_> = std::fs::read_dir(&tmp_dir)
        .expect("read dir")
        .filter_map(Result::ok)
        .collect();
    assert_eq!(entries.len(), 1, "only the target file should exist");
    assert_eq!(entries[0].file_name().to_string_lossy(), "status.json");

    std::fs::remove_dir_all(&tmp_dir).expect("remove temp dir");
}

#[test]
fn test_load_ignores_interrupted_atomic_temp_file_and_keeps_last_good_snapshot() {
    let tmp_dir = unique_temp_dir("atomic_interrupted");
    let path = tmp_dir.join("status.json");

    let mut status = VerifyStatus::default();
    let result = make_result("stable_kernel", PropMethod::Ibp, -2.0, 2.0);
    status
        .record(&result, bounds(-1.0, 1.0), &[], None)
        .expect("record");
    status.save(&path).expect("initial save");

    // Simulate interrupted save: temp file was written but rename never happened.
    let interrupted_tmp = tmp_dir.join(".status.json.interrupted.tmp");
    std::fs::write(&interrupted_tmp, "{\"kernels\":").expect("write partial temp file");

    let loaded = VerifyStatus::load(&path).expect("load should ignore stale temp file");
    assert_eq!(
        loaded.kernels["stable_kernel"].status,
        VerifyOutcome::Verified
    );
    assert_eq!(loaded.run_count("stable_kernel"), 1);
    assert!(
        interrupted_tmp.exists(),
        "load must not touch temp artifacts"
    );

    std::fs::remove_dir_all(&tmp_dir).expect("remove temp dir");
}

#[test]
fn test_roundtrip_preserves_invariants_via_record_api() {
    let tmp_dir = unique_temp_dir("roundtrip_invariant");
    let path = tmp_dir.join("status.json");

    // Build status through record* API only.
    let mut status = VerifyStatus::default();
    let result = make_result("inv_kernel", PropMethod::Ibp, -3.0, 3.0);
    status
        .record(&result, bounds(-1.0, 1.0), &[], None)
        .expect("record");
    status
        .record_failure("inv_kernel", PropMethod::Crown, bounds(-2.0, 2.0), &[])
        .expect("record failure");

    // Invariant: kernels has latest (failure), history has both runs.
    assert_eq!(status.kernels["inv_kernel"].status, VerifyOutcome::Failed);
    assert_eq!(status.run_count("inv_kernel"), 2);
    assert_eq!(
        status.history["inv_kernel"][0].status,
        VerifyOutcome::Verified
    );
    assert_eq!(
        status.history["inv_kernel"][1].status,
        VerifyOutcome::Failed
    );

    // Roundtrip through save/load.
    status.save(&path).expect("save");
    let loaded = VerifyStatus::load(&path).expect("load");

    // Invariants survive serialization.
    assert_eq!(loaded.kernels["inv_kernel"].status, VerifyOutcome::Failed);
    assert_eq!(loaded.run_count("inv_kernel"), 2);
    assert_eq!(
        loaded.history["inv_kernel"][0].status,
        VerifyOutcome::Verified
    );
    assert_eq!(
        loaded.history["inv_kernel"][1].status,
        VerifyOutcome::Failed
    );
    assert_eq!(loaded, status);

    // Further record* after load still maintains invariants.
    let mut loaded = loaded;
    let result3 = make_result("inv_kernel", PropMethod::Crown, -2.0, 2.0);
    loaded
        .record(&result3, bounds(-1.0, 1.0), &[], None)
        .expect("record");
    assert_eq!(loaded.kernels["inv_kernel"].status, VerifyOutcome::Verified);
    assert_eq!(loaded.run_count("inv_kernel"), 3);

    std::fs::remove_dir_all(&tmp_dir).expect("remove temp dir");
}

#[test]
fn test_load_returns_serialization_error_for_partial_target_file() {
    let tmp_dir = unique_temp_dir("partial_target");
    let path = tmp_dir.join("status.json");
    std::fs::write(&path, "{\"kernels\":").expect("write partial target");

    let err = VerifyStatus::load(&path).expect_err("partial JSON should fail to parse");
    assert!(
        matches!(err, VerifyError::Serialization(_)),
        "expected serialization error for partial JSON target, got {err:?}"
    );

    std::fs::remove_dir_all(&tmp_dir).expect("remove temp dir");
}

#[test]
fn test_record_smt_with_missing_history_still_updates_kernel() {
    // Simulate structural inconsistency: kernel entry exists but history is empty.
    // This can happen when loading legacy status files or if a code path
    // populates `kernels` without appending to `history`.
    let mut status = VerifyStatus::default();
    let result = make_result("orphan_kernel", PropMethod::Ibp, -1.0, 1.0);
    // Record normally first (creates both kernel + history entries).
    status
        .record(&result, bounds(-1.0, 1.0), &[], None)
        .expect("record");
    assert_eq!(status.run_count("orphan_kernel"), 1);

    // Remove the history entry to simulate the inconsistency.
    status.history.clear();
    assert_eq!(status.run_count("orphan_kernel"), 0);

    let smt = SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::Exact,
        property: "output_finite".to_string(),
        outcome: SmtOutcome::Proven,
        detail: None,
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };

    // record_smt succeeds (kernel entry updated). Partial write (history
    // missing) is logged to stderr, not signaled via return value (#669).
    status
        .record_smt("orphan_kernel", smt)
        .expect("record_smt should not error");

    // Kernel entry should have the SMT result.
    let kernel = status.kernel("orphan_kernel").expect("kernel should exist");
    assert_eq!(
        kernel.smt.as_ref().expect("smt should be set").outcome,
        SmtOutcome::Proven
    );

    // History should still be empty (no history entry to update).
    assert_eq!(status.run_count("orphan_kernel"), 0);
}

#[test]
fn test_record_smt_with_no_kernel_entry_returns_error() {
    let mut status = VerifyStatus::default();
    let smt = SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::Exact,
        property: "output_finite".to_string(),
        outcome: SmtOutcome::Proven,
        detail: None,
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };

    // Per #669: record_smt returns Err when no kernel entry exists.
    let err = status
        .record_smt("nonexistent", smt)
        .expect_err("record_smt should error for missing kernel");
    assert!(err.to_string().contains("no kernel entry"));
}

#[test]
fn test_record_smt_normal_path_updates_both_kernel_and_history() {
    let mut status = VerifyStatus::default();
    let result = make_result("smt_test", PropMethod::Ibp, -1.0, 1.0);
    status
        .record(&result, bounds(-1.0, 1.0), &[], None)
        .expect("record");

    let smt = SmtStatusRecord {
        solver: "ay".to_string(),
        encoding: SmtEncodingKind::UfApprox,
        property: "bound_check".to_string(),
        outcome: SmtOutcome::Unexecuted,
        detail: None,
        bounds_source: BoundsSource::Heuristic,
        expected_bounds: None,
        proof_alethe: None,
        proof_verdict: None,
    };

    status
        .record_smt("smt_test", smt)
        .expect("record_smt should not error");

    // Both kernel and history should have the SMT result.
    let kernel = status.kernel("smt_test").expect("kernel");
    assert_eq!(
        kernel.smt.as_ref().expect("smt").outcome,
        SmtOutcome::Unexecuted
    );

    let history = status.history_for("smt_test").expect("history");
    assert_eq!(history.len(), 1);
    assert_eq!(
        history[0].smt.as_ref().expect("history smt").outcome,
        SmtOutcome::Unexecuted
    );
}
