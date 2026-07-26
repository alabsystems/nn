// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! P8 (ay SMT) bridge tests for [`SmtVerificationEvidence::from_verify_status`].
//!
//! These tests use `serde_json::from_str` to construct synthetic `VerifyStatus`
//! objects (required because `KernelStatus` and `SmtStatusRecord` are
//! `#[non_exhaustive]`).
//!
//! Extracted from `moonshot_evidence_bridge_tests.rs` (Phase 37 of #1741).

use super::super::SmtVerificationEvidence;

/// Helper: build a `VerifyStatus` from JSON for test isolation.
///
/// Uses `serde_json::from_str` because `KernelStatus` and `SmtStatusRecord`
/// are `#[non_exhaustive]` and cannot be constructed from outside nn-verify.
fn status_from_json(json: &str) -> nn_verify::VerifyStatus {
    serde_json::from_str(json).expect("valid test JSON")
}

/// All kernels proven → `all_proven` = true.
#[test]
fn test_smt_bridge_all_proven() {
    let json = r#"{
        "kernels": {
            "snake": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -10.0, "upper": 10.0}], "constant_params": [1.0]},
                "output_bounds": {"lower": -10.0, "upper": 11.0, "is_infeasible": false},
                "output_width": 21.0,
                "soundness_mode": "sound",
                "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "proven", "bounds_source": "analytical"}
            },
            "silu_mul": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -5.0, "upper": 5.0}], "constant_params": []},
                "output_bounds": {"lower": -5.0, "upper": 5.0, "is_infeasible": false},
                "output_width": 10.0,
                "soundness_mode": "sound",
                "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "proven", "bounds_source": "analytical"}
            }
        }
    }"#;

    let status = status_from_json(json);
    let evidence = SmtVerificationEvidence::from_verify_status(&status);

    assert_eq!(evidence.kernels_proven, 2);
    assert_eq!(evidence.kernels_total, 2);
    assert!(evidence.all_proven);
    assert_eq!(evidence.proven_kernel_names, vec!["silu_mul", "snake"]);
}

/// Mixed outcomes: one Proven, one Counterexample → not all_proven.
#[test]
fn test_smt_bridge_mixed_outcomes() {
    let json = r#"{
        "kernels": {
            "snake": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -10.0, "upper": 10.0}], "constant_params": [1.0]},
                "output_bounds": {"lower": -10.0, "upper": 11.0, "is_infeasible": false},
                "output_width": 21.0,
                "soundness_mode": "sound",
                "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "proven", "bounds_source": "analytical"}
            },
            "gelu": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -5.0, "upper": 5.0}], "constant_params": []},
                "output_bounds": {"lower": -5.0, "upper": 5.0, "is_infeasible": false},
                "output_width": 10.0,
                "soundness_mode": "heuristic",
                "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "counterexample", "bounds_source": "analytical"}
            }
        }
    }"#;

    let status = status_from_json(json);
    let evidence = SmtVerificationEvidence::from_verify_status(&status);

    assert_eq!(evidence.kernels_proven, 1);
    assert_eq!(evidence.kernels_total, 2);
    assert!(!evidence.all_proven);
    assert_eq!(evidence.proven_kernel_names, vec!["snake"]);
}

/// Unexecuted outcomes are excluded from total.
#[test]
fn test_smt_bridge_unexecuted_excluded() {
    let json = r#"{
        "kernels": {
            "snake": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -10.0, "upper": 10.0}], "constant_params": [1.0]},
                "output_bounds": {"lower": -10.0, "upper": 11.0, "is_infeasible": false},
                "output_width": 21.0,
                "soundness_mode": "sound",
                "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "proven", "bounds_source": "analytical"}
            },
            "pending_kernel": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": []},
                "output_bounds": {"lower": -1.0, "upper": 1.0, "is_infeasible": false},
                "output_width": 2.0,
                "soundness_mode": "heuristic",
                "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "unexecuted", "bounds_source": "heuristic"}
            }
        }
    }"#;

    let status = status_from_json(json);
    let evidence = SmtVerificationEvidence::from_verify_status(&status);

    assert_eq!(evidence.kernels_proven, 1, "only snake is proven");
    assert_eq!(evidence.kernels_total, 1, "unexecuted excluded from total");
    assert!(evidence.all_proven, "1/1 proven → all_proven");
}

/// Kernel with `smt: null` in latest but `Proven` in history → counts.
#[test]
fn test_smt_bridge_history_fallback() {
    let json = r#"{
        "kernels": {
            "rope_cos": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -3.14, "upper": 3.14}], "constant_params": []},
                "output_bounds": {"lower": -1.0, "upper": 1.0, "is_infeasible": false},
                "output_width": 2.0,
                "soundness_mode": "sound"
            }
        },
        "history": {
            "rope_cos": [
                {
                    "status": "verified",
                    "method": "IBP",
                    "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -3.14, "upper": 3.14}], "constant_params": []},
                    "output_bounds": {"lower": -1.0, "upper": 1.0, "is_infeasible": false},
                    "output_width": 2.0,
                    "soundness_mode": "sound",
                    "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "proven", "bounds_source": "analytical"}
                }
            ]
        }
    }"#;

    let status = status_from_json(json);
    let evidence = SmtVerificationEvidence::from_verify_status(&status);

    assert_eq!(evidence.kernels_proven, 1, "history fallback → proven");
    assert_eq!(evidence.kernels_total, 1);
    assert!(evidence.all_proven);
    assert_eq!(evidence.proven_kernel_names, vec!["rope_cos"]);
}

/// Empty status → zero evidence (not all_proven because total is 0).
#[test]
fn test_smt_bridge_empty_status() {
    let status = nn_verify::VerifyStatus::default();
    let evidence = SmtVerificationEvidence::from_verify_status(&status);

    assert_eq!(evidence.kernels_proven, 0);
    assert_eq!(evidence.kernels_total, 0);
    assert!(!evidence.all_proven, "zero kernels → not all_proven");
    assert!(evidence.proven_kernel_names.is_empty());
}

/// Kernels only in history (not in latest) are discovered.
#[test]
fn test_smt_bridge_history_only_kernel() {
    let json = r#"{
        "kernels": {},
        "history": {
            "tanh_act": [
                {
                    "status": "verified",
                    "method": "IBP",
                    "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -5.0, "upper": 5.0}], "constant_params": []},
                    "output_bounds": {"lower": -1.0, "upper": 1.0, "is_infeasible": false},
                    "output_width": 2.0,
                    "soundness_mode": "sound",
                    "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "proven", "bounds_source": "analytical"}
                }
            ]
        }
    }"#;

    let status = status_from_json(json);
    let evidence = SmtVerificationEvidence::from_verify_status(&status);

    assert_eq!(evidence.kernels_proven, 1);
    assert_eq!(evidence.kernels_total, 1);
    assert!(evidence.all_proven);
    assert_eq!(evidence.proven_kernel_names, vec!["tanh_act"]);
}

/// ExecutionFailed counts toward total but not proven.
#[test]
fn test_smt_bridge_execution_failed() {
    let json = r#"{
        "kernels": {
            "broken_kernel": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": []},
                "output_bounds": {"lower": -1.0, "upper": 1.0, "is_infeasible": false},
                "output_width": 2.0,
                "soundness_mode": "heuristic",
                "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "execution_failed", "detail": "solver crash", "bounds_source": "heuristic"}
            }
        }
    }"#;

    let status = status_from_json(json);
    let evidence = SmtVerificationEvidence::from_verify_status(&status);

    assert_eq!(evidence.kernels_proven, 0);
    assert_eq!(evidence.kernels_total, 1, "ExecutionFailed counts in total");
    assert!(!evidence.all_proven);
}

/// Unknown outcome counts toward total but not proven.
#[test]
fn test_smt_bridge_unknown_outcome() {
    let json = r#"{
        "kernels": {
            "complex_kernel": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}], "constant_params": []},
                "output_bounds": {"lower": -1.0, "upper": 1.0, "is_infeasible": false},
                "output_width": 2.0,
                "soundness_mode": "heuristic",
                "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "unknown", "bounds_source": "analytical"}
            }
        }
    }"#;

    let status = status_from_json(json);
    let evidence = SmtVerificationEvidence::from_verify_status(&status);

    assert_eq!(evidence.kernels_proven, 0);
    assert_eq!(evidence.kernels_total, 1, "Unknown counts in total");
    assert!(!evidence.all_proven);
}

/// proven_kernel_names is sorted alphabetically.
#[test]
fn test_smt_bridge_names_sorted() {
    let json = r#"{
        "kernels": {
            "sigmoid": {
                "status": "verified", "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -5.0, "upper": 5.0}], "constant_params": []},
                "output_bounds": {"lower": 0.0, "upper": 1.0, "is_infeasible": false},
                "output_width": 1.0, "soundness_mode": "sound",
                "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "proven", "bounds_source": "analytical"}
            },
            "gelu": {
                "status": "verified", "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -5.0, "upper": 5.0}], "constant_params": []},
                "output_bounds": {"lower": -0.2, "upper": 5.0, "is_infeasible": false},
                "output_width": 5.2, "soundness_mode": "sound",
                "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "proven", "bounds_source": "analytical"}
            },
            "relu": {
                "status": "verified", "method": "IBP",
                "input_bounds": {"variable_inputs": [{"param_index": 0, "lower": -5.0, "upper": 5.0}], "constant_params": []},
                "output_bounds": {"lower": 0.0, "upper": 5.0, "is_infeasible": false},
                "output_width": 5.0, "soundness_mode": "sound",
                "smt": {"solver": "ay", "encoding": "uf_approx", "property": "bound_check", "outcome": "proven", "bounds_source": "analytical"}
            }
        }
    }"#;

    let status = status_from_json(json);
    let evidence = SmtVerificationEvidence::from_verify_status(&status);

    assert_eq!(evidence.kernels_proven, 3);
    assert_eq!(
        evidence.proven_kernel_names,
        vec!["gelu", "relu", "sigmoid"],
        "names must be sorted alphabetically"
    );
}
