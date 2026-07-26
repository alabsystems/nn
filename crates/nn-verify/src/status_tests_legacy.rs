// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Legacy JSON loading, tensor output bounds, and soundness mode tests.

use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn test_load_legacy_input_range_synthesizes_variable_inputs() {
    let legacy_json = r#"{
  "kernels": {
    "legacy_snake": {
      "status": "verified",
      "method": "IBP",
      "input_bounds": {
        "input_range": [-10.0, 10.0],
        "constant_params": [1.0]
      },
      "output_bounds": {
        "lower": -10.0,
        "upper": 11.0
      },
      "output_width": 21.0
    }
  }
}"#;

    let mut path = std::env::temp_dir();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    path.push(format!(
        "nn_verify_legacy_status_{}_{}.json",
        std::process::id(),
        unique
    ));
    std::fs::write(&path, legacy_json).expect("write legacy json");

    let loaded = VerifyStatus::load(&path).expect("load status");
    let input = &loaded.kernels["legacy_snake"].input_bounds;

    assert_eq!(input.input_range, Some((-10.0, 10.0)));
    assert_eq!(
        input.variable_inputs,
        vec![ParamInputRecord {
            param_index: 0,
            lower: -10.0,
            upper: 10.0,
        }]
    );
    assert_eq!(input.input_shape, Some(vec![1]));

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_load_legacy_status_defaults_missing_soundness_mode_to_heuristic() {
    let legacy_json = r#"{
  "kernels": {
    "legacy_snake": {
      "status": "verified",
      "method": "IBP",
      "input_bounds": {
        "input_range": [-10.0, 10.0],
        "constant_params": [1.0]
      },
      "output_bounds": {
        "lower": -10.0,
        "upper": 11.0
      },
      "output_width": 21.0
    }
  }
}"#;

    let mut path = std::env::temp_dir();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    path.push(format!(
        "nn_verify_legacy_soundness_status_{}_{}.json",
        std::process::id(),
        unique
    ));
    std::fs::write(&path, legacy_json).expect("write legacy json");

    let loaded = VerifyStatus::load(&path).expect("load status");
    assert_eq!(
        loaded.kernels["legacy_snake"].soundness_mode,
        VerificationSoundnessMode::Heuristic
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_load_drops_legacy_input_range_for_nonzero_param_index() {
    let malformed_json = r#"{
  "kernels": {
    "scaled": {
      "status": "verified",
      "method": "IBP",
      "input_bounds": {
        "variable_inputs": [{"param_index": 1, "lower": -1.0, "upper": 1.0}],
        "constant_params": [2.0],
        "input_shape": [1],
        "input_range": [-1.0, 1.0]
      },
      "output_bounds": {
        "lower": -2.0,
        "upper": 2.0
      },
      "output_width": 4.0
    }
  }
}"#;

    let mut path = std::env::temp_dir();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    path.push(format!(
        "nn_verify_nonzero_param_status_{}_{}.json",
        std::process::id(),
        unique
    ));
    std::fs::write(&path, malformed_json).expect("write malformed json");

    let loaded = VerifyStatus::load(&path).expect("load status");
    let input = &loaded.kernels["scaled"].input_bounds;
    assert_eq!(input.input_range, None);
    assert_eq!(input.input_shape, Some(vec![1]));
    assert_eq!(
        input.variable_inputs,
        vec![ParamInputRecord {
            param_index: 1,
            lower: -1.0,
            upper: 1.0,
        }]
    );

    let _ = std::fs::remove_file(path);
}

// --- Legacy NaN guard tests (#214 AC4) ---

#[test]
fn test_normalize_legacy_fields_rejects_nan_input_range() {
    let mut record = InputBoundsRecord {
        variable_inputs: vec![],
        constant_params: vec![1.0],
        input_shape: None,
        input_range: Some((f32::NAN, 10.0)),
    };
    record.normalize_legacy_fields();

    // NaN lower should prevent promotion to variable_inputs.
    assert!(record.variable_inputs.is_empty());
    assert_eq!(record.input_shape, None);
    assert_eq!(record.input_range, None);
}

#[test]
fn test_normalize_legacy_fields_rejects_nan_upper_input_range() {
    let mut record = InputBoundsRecord {
        variable_inputs: vec![],
        constant_params: vec![1.0],
        input_shape: None,
        input_range: Some((0.0, f32::NAN)),
    };
    record.normalize_legacy_fields();

    // NaN upper (with finite lower) should also prevent promotion.
    assert!(record.variable_inputs.is_empty());
    assert_eq!(record.input_shape, None);
    assert_eq!(record.input_range, None);
}

#[test]
fn test_normalize_legacy_fields_rejects_infinity_input_range() {
    let mut record = InputBoundsRecord {
        variable_inputs: vec![],
        constant_params: vec![],
        input_shape: None,
        input_range: Some((f32::NEG_INFINITY, f32::INFINITY)),
    };
    record.normalize_legacy_fields();

    // Infinite bounds should not be promoted to variable_inputs.
    assert!(record.variable_inputs.is_empty());
    assert_eq!(record.input_shape, None);
    assert_eq!(record.input_range, None);
}

#[test]
fn test_normalize_legacy_fields_accepts_finite_input_range() {
    let mut record = InputBoundsRecord {
        variable_inputs: vec![],
        constant_params: vec![1.0],
        input_shape: None,
        input_range: Some((-5.0, 5.0)),
    };
    record.normalize_legacy_fields();

    // Finite bounds should be promoted normally.
    assert_eq!(record.variable_inputs.len(), 1);
    assert_eq!(record.variable_inputs[0].param_index, 0);
    assert_eq!(record.variable_inputs[0].lower, -5.0);
    assert_eq!(record.variable_inputs[0].upper, 5.0);
    assert_eq!(record.input_shape, Some(vec![1]));
    assert_eq!(record.input_range, Some((-5.0, 5.0)));
}

// --- Tensor output bounds tests (#65) ---

#[test]
fn test_tensor_output_bounds_roundtrip() {
    let record = OutputBoundsRecord {
        lower: -3.0,
        upper: 5.0,
        tensor_lower: Some(vec![-3.0, -1.0, 0.0, 2.0]),
        tensor_upper: Some(vec![1.0, 3.0, 4.0, 5.0]),
        shape: Some(vec![2, 2]),
        is_infeasible: false,
    };

    let json = serde_json::to_string_pretty(&record).expect("serialize");
    assert!(json.contains("tensor_lower"));
    assert!(json.contains("tensor_upper"));
    assert!(json.contains("\"shape\""));

    let deserialized: OutputBoundsRecord = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.lower, -3.0);
    assert_eq!(deserialized.upper, 5.0);
    assert_eq!(deserialized.tensor_lower, Some(vec![-3.0, -1.0, 0.0, 2.0]));
    assert_eq!(deserialized.tensor_upper, Some(vec![1.0, 3.0, 4.0, 5.0]));
    assert_eq!(deserialized.shape, Some(vec![2, 2]));
}

#[test]
fn test_legacy_scalar_output_bounds_deserializes_without_tensor_fields() {
    let legacy_json = r#"{"lower": -10.0, "upper": 11.0}"#;
    let record: OutputBoundsRecord = serde_json::from_str(legacy_json).expect("deserialize legacy");
    assert_eq!(record.lower, -10.0);
    assert_eq!(record.upper, 11.0);
    assert_eq!(record.tensor_lower, None);
    assert_eq!(record.tensor_upper, None);
    assert_eq!(record.shape, None);
    assert!(
        !record.is_infeasible,
        "legacy JSON without is_infeasible defaults to false"
    );
}

#[test]
fn test_scalar_output_bounds_omits_tensor_fields_in_json() {
    let record = OutputBoundsRecord {
        lower: -5.0,
        upper: 5.0,
        tensor_lower: None,
        tensor_upper: None,
        shape: None,
        is_infeasible: false,
    };
    let json = serde_json::to_string(&record).expect("serialize");
    assert!(!json.contains("tensor_lower"));
    assert!(!json.contains("tensor_upper"));
    assert!(!json.contains("shape"));
}

#[test]
fn test_tensor_status_full_roundtrip_through_verify_status() {
    let mut status = VerifyStatus::default();
    status.kernels.insert(
        "tensor_kernel".to_string(),
        KernelStatus {
            status: VerifyOutcome::Verified,
            method: PropMethod::Ibp,
            input_bounds: InputBoundsRecord {
                variable_inputs: vec![ParamInputRecord {
                    param_index: 0,
                    lower: -1.0,
                    upper: 1.0,
                }],
                constant_params: vec![],
                input_shape: Some(vec![1]),
                input_range: Some((-1.0, 1.0)),
            },
            output_bounds: OutputBoundsRecord {
                lower: -2.0,
                upper: 3.0,
                tensor_lower: Some(vec![-2.0, -1.0, 0.5]),
                tensor_upper: Some(vec![1.0, 2.0, 3.0]),
                shape: Some(vec![3]),
                is_infeasible: false,
            },
            output_width: 5.0,
            crown_error: None,
            soundness_mode: VerificationSoundnessMode::Sound,
            smt: None,
            crown_coverage: None,
            ibp_comparison_width: None,
            crown_ibp_ratio: None,
            weight_artifact: None,
            soundness_justification: None,
            stale: false,
            stale_reason: None,
            proof_strength: None,
        },
    );

    let json = serde_json::to_string_pretty(&status).expect("serialize");
    let deserialized: VerifyStatus = serde_json::from_str(&json).expect("deserialize");

    let k = &deserialized.kernels["tensor_kernel"];
    assert_eq!(k.output_bounds.tensor_lower, Some(vec![-2.0, -1.0, 0.5]));
    assert_eq!(k.output_bounds.tensor_upper, Some(vec![1.0, 2.0, 3.0]));
    assert_eq!(k.output_bounds.shape, Some(vec![3]));
    assert_eq!(k.output_bounds.lower, -2.0);
    assert_eq!(k.output_bounds.upper, 3.0);
}

#[test]
fn test_normalize_legacy_input_bounds_also_normalizes_history() {
    // Legacy JSON with history entries containing old-format input_range
    // but no variable_inputs — normalization should synthesize variable_inputs
    // in both kernels AND history.
    let legacy_json = r#"{
  "kernels": {
    "snake": {
      "status": "verified",
      "method": "IBP",
      "input_bounds": {
        "input_range": [-5.0, 5.0],
        "constant_params": [1.0]
      },
      "output_bounds": { "lower": -5.0, "upper": 6.0 },
      "output_width": 11.0
    }
  },
  "history": {
    "snake": [
      {
        "status": "verified",
        "method": "IBP",
        "input_bounds": {
          "input_range": [-3.0, 3.0],
          "constant_params": [0.5]
        },
        "output_bounds": { "lower": -3.0, "upper": 4.0 },
        "output_width": 7.0
      }
    ]
  }
}"#;

    let mut path = std::env::temp_dir();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    path.push(format!(
        "nn_verify_history_norm_{}_{}.json",
        std::process::id(),
        unique
    ));
    std::fs::write(&path, legacy_json).expect("write legacy json");

    let loaded = VerifyStatus::load(&path).expect("load status");

    // AC2: kernel() and history_for() should return consistent representations.
    let kernel_bounds = &loaded.kernel("snake").expect("kernel exists").input_bounds;
    let history_bounds = &loaded.history_for("snake").expect("history exists")[0].input_bounds;

    // Both should have synthesized variable_inputs from legacy input_range.
    assert_eq!(kernel_bounds.variable_inputs.len(), 1);
    assert_eq!(history_bounds.variable_inputs.len(), 1);

    // History entry should have the correct bounds from its input_range.
    assert_eq!(history_bounds.variable_inputs[0].lower, -3.0);
    assert_eq!(history_bounds.variable_inputs[0].upper, 3.0);
    assert_eq!(history_bounds.variable_inputs[0].param_index, 0);
    assert_eq!(history_bounds.input_shape, Some(vec![1]));

    let _ = std::fs::remove_file(path);
}
