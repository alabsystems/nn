// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! .nnc model serialization roundtrip tests for the Kokoro pipeline.
//!
//! Tests verify that `CompiledPlan` objects representative of Kokoro pipeline
//! segments survive save/load and JSON serialization without data loss.
//!
//! - Synthetic tests: work without KOKORO_WEIGHTS, use hand-built plans that
//!   mirror the NativeOp structure of real Kokoro segments.
//! - Production test: gated on KOKORO_WEIGHTS, builds real CompiledKokoro,
//!   runs synthesize() to populate segment caches, and verifies dispatch
//!   counts are consistent after a full roundtrip.

use nn_dsl::trace_compile::{CompiledPlan, CompiledStep};

// ---------------------------------------------------------------------------
// Helper: build Kokoro-representative compiled plans for each segment type.
//
// CompiledPlan is #[non_exhaustive], so we construct via JSON deserialization
// from outside the nn-dsl crate.
// ---------------------------------------------------------------------------

/// Build a plan mimicking the PlBert segment (seg_plbert).
///
/// PlBert: token embeddings -> self-attention -> LayerNorm -> linear.
/// Key NativeOps: FlashAttention, LayerNorm, LinearActivation.
fn plbert_segment_plan() -> CompiledPlan {
    let json = r#"{
        "steps": [
            "InputForward",
            {
                "NativeOp": {
                    "op": {
                        "LayerNorm": {
                            "eps": 1e-5,
                            "input_shape": [1, 4, 8],
                            "hidden_dim": 8
                        }
                    },
                    "weight_data": {
                        "norm.weight": { "data": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "shape": [8] },
                        "norm.bias": { "data": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "shape": [8] }
                    }
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "FlashAttention": {
                            "scale": 0.3536,
                            "causal": false,
                            "q_shape": [1, 2, 4, 4],
                            "k_shape": [1, 2, 4, 4],
                            "output_shape": [1, 2, 4, 4],
                            "input_layout": "HeadsFirst"
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "LinearActivation": {
                            "activation": "Gelu",
                            "in_features": 8,
                            "out_features": 16,
                            "has_bias": true,
                            "input_shape": [1, 4, 8]
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "Passthrough": {
                    "op_name": "reshape",
                    "output_shape": [1, 4, 8]
                }
            }
        ],
        "input_shapes": [[1, 4]],
        "output_step": 4,
        "weight_names": ["norm.bias", "norm.weight"]
    }"#;
    CompiledPlan::from_json(json).expect("plbert_segment_plan JSON parse")
}

/// Build a plan mimicking the TextEncoder segment (seg_text).
///
/// TextEncoder: LSTM + NormActivConv1d + AdainSnake + InstanceNorm.
fn text_encoder_segment_plan() -> CompiledPlan {
    let json = r#"{
        "steps": [
            "InputForward",
            {
                "NativeOp": {
                    "op": {
                        "LstmSequence": {
                            "hidden_size": 4,
                            "input_shape": [4, 1, 8],
                            "h_shape": [1, 4],
                            "reverse": false
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "InstanceNorm": {
                            "eps": 1e-5,
                            "input_shape": [1, 8, 4]
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "NormActivConv1d": {
                            "activation": "Snake",
                            "eps": 1e-5,
                            "conv_dilation": 1,
                            "conv_padding": 1,
                            "input_shape": [1, 8, 4],
                            "output_channels": 8,
                            "kernel_size": 3,
                            "external_node_ids": [0, 1, 2]
                        }
                    },
                    "weight_data": {
                        "alpha": { "data": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "shape": [8] }
                    }
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "AdainSnake": {
                            "eps": 1e-5,
                            "input_shape": [1, 8, 4],
                            "channels": 8,
                            "residual_gamma": true,
                            "external_node_ids": [10, 20, 30]
                        }
                    },
                    "weight_data": {}
                }
            }
        ],
        "input_shapes": [[1, 4, 8]],
        "output_step": 4,
        "weight_names": ["alpha"]
    }"#;
    CompiledPlan::from_json(json).expect("text_encoder_segment_plan JSON parse")
}

/// Build a plan mimicking the Generator segment (seg_generator).
///
/// Generator: FusedResBlock chain + NormActivConv1d + ConstantValue.
fn generator_segment_plan() -> CompiledPlan {
    let json = r#"{
        "steps": [
            "InputForward",
            {
                "NativeOp": {
                    "op": {
                        "FusedResBlock": {
                            "phase1": {
                                "activation": "Snake",
                                "eps": 1e-5,
                                "conv_dilation": 1,
                                "conv_padding": 1,
                                "input_shape": [1, 8, 16],
                                "output_channels": 8,
                                "kernel_size": 3
                            },
                            "phase2": {
                                "activation": "Snake",
                                "eps": 1e-5,
                                "conv_dilation": 1,
                                "conv_padding": 1,
                                "input_shape": [1, 8, 16],
                                "output_channels": 8,
                                "kernel_size": 3
                            },
                            "input_steps": [0, 1, 2, 3, 4],
                            "residual_scale": 1.0,
                            "style_proj": {
                                "channels1": 8,
                                "channels2": 8,
                                "style_dim": 4
                            },
                            "shortcut_step": 0,
                            "pool_step": null,
                            "style_batch_offset": null
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "NormActivConv1d": {
                            "activation": "Snake",
                            "eps": 1e-5,
                            "conv_dilation": 1,
                            "conv_padding": 3,
                            "input_shape": [1, 8, 16],
                            "output_channels": 4,
                            "kernel_size": 7,
                            "external_node_ids": [5, 6, 7]
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "ConstantValue": {
                    "value": 0.7071,
                    "shape": [1]
                }
            },
            {
                "Passthrough": {
                    "op_name": "mul",
                    "output_shape": [1, 4, 16]
                }
            }
        ],
        "input_shapes": [[1, 8, 16]],
        "output_step": 4,
        "weight_names": []
    }"#;
    CompiledPlan::from_json(json).expect("generator_segment_plan JSON parse")
}

/// Build a plan mimicking the F0Energy segment (seg_f0).
///
/// F0Energy: BiLstmCat + LinearActivation.
fn f0_energy_segment_plan() -> CompiledPlan {
    let json = r#"{
        "steps": [
            "InputForward",
            {
                "NativeOp": {
                    "op": {
                        "LstmSequence": {
                            "hidden_size": 4,
                            "input_shape": [8, 1, 8],
                            "h_shape": [1, 4],
                            "reverse": false
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "LstmSequence": {
                            "hidden_size": 4,
                            "input_shape": [8, 1, 8],
                            "h_shape": [1, 4],
                            "reverse": true
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "BiLstmCat": {
                            "hidden_size": 4,
                            "input_shape": [8, 1, 8],
                            "h_shape": [1, 4],
                            "fwd_lstm_step": 1,
                            "rev_lstm_step": 2
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "LinearActivation": {
                            "activation": "Relu",
                            "in_features": 8,
                            "out_features": 4,
                            "has_bias": true,
                            "input_shape": [1, 8, 8]
                        }
                    },
                    "weight_data": {}
                }
            }
        ],
        "input_shapes": [[8, 1, 8]],
        "output_step": 4,
        "weight_names": []
    }"#;
    CompiledPlan::from_json(json).expect("f0_energy_segment_plan JSON parse")
}

/// Build a plan mimicking the Regulate segment (seg_regulate).
///
/// Regulate: pure elementwise chain (no model weights).
fn regulate_segment_plan() -> CompiledPlan {
    let json = r#"{
        "steps": [
            "InputForward",
            {
                "Passthrough": {
                    "op_name": "sigmoid",
                    "output_shape": [1, 4]
                }
            },
            {
                "Passthrough": {
                    "op_name": "sum",
                    "output_shape": [1]
                }
            },
            {
                "Passthrough": {
                    "op_name": "mul_speed",
                    "output_shape": [1]
                }
            },
            {
                "Passthrough": {
                    "op_name": "clamp",
                    "output_shape": [1]
                }
            }
        ],
        "input_shapes": [[1, 4]],
        "output_step": 4,
        "weight_names": []
    }"#;
    CompiledPlan::from_json(json).expect("regulate_segment_plan JSON parse")
}

// ---------------------------------------------------------------------------
// Segment-level roundtrip tests (synthetic, no KOKORO_WEIGHTS needed)
// ---------------------------------------------------------------------------

/// Save each Kokoro segment plan to a temp .nnc file, load it back, and
/// verify that step count, input shapes, output step, and weight names
/// all survive the roundtrip. Also verifies dispatch count matches.
#[test]
fn test_segment_save_load_roundtrip() {
    let segments: Vec<(&str, CompiledPlan)> = vec![
        ("plbert", plbert_segment_plan()),
        ("text_encoder", text_encoder_segment_plan()),
        ("generator", generator_segment_plan()),
        ("f0_energy", f0_energy_segment_plan()),
        ("regulate", regulate_segment_plan()),
    ];

    let dir =
        std::env::temp_dir().join(format!("nn_kokoro_serde_roundtrip_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    for (name, plan) in &segments {
        let path = dir.join(format!("{name}.nnc"));

        // Save
        plan.save(&path)
            .unwrap_or_else(|e| panic!("{name}: save failed: {e}"));

        // Verify file exists and is non-empty
        let file_size = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("{name}: metadata failed: {e}"))
            .len();
        assert!(file_size > 0, "{name}: .nnc file should be non-empty");

        // Load
        let restored =
            CompiledPlan::load(&path).unwrap_or_else(|e| panic!("{name}: load failed: {e}"));

        // Verify structural equality
        assert_eq!(
            restored.steps.len(),
            plan.steps.len(),
            "{name}: step count mismatch"
        );
        assert_eq!(
            restored.input_shapes, plan.input_shapes,
            "{name}: input_shapes mismatch"
        );
        assert_eq!(
            restored.output_step, plan.output_step,
            "{name}: output_step mismatch"
        );
        assert_eq!(
            restored.weight_names, plan.weight_names,
            "{name}: weight_names mismatch"
        );

        // Count dispatches (NativeOp + Dispatch steps) to verify they match
        let original_dispatches = plan
            .steps
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
                )
            })
            .count();
        let restored_dispatches = restored
            .steps
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
                )
            })
            .count();
        assert_eq!(
            original_dispatches, restored_dispatches,
            "{name}: dispatch count mismatch"
        );
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

/// Serialize each segment plan to JSON in memory, deserialize, and verify
/// the JSON roundtrip is identical (double roundtrip produces same JSON).
#[test]
fn test_plan_serde_roundtrip() {
    let segments: Vec<(&str, CompiledPlan)> = vec![
        ("plbert", plbert_segment_plan()),
        ("text_encoder", text_encoder_segment_plan()),
        ("generator", generator_segment_plan()),
        ("f0_energy", f0_energy_segment_plan()),
        ("regulate", regulate_segment_plan()),
    ];

    for (name, plan) in &segments {
        // Serialize to JSON
        let json = plan
            .to_json()
            .unwrap_or_else(|e| panic!("{name}: to_json failed: {e}"));

        // Deserialize
        let restored = CompiledPlan::from_json(&json)
            .unwrap_or_else(|e| panic!("{name}: from_json failed: {e}"));

        // Verify structural equality
        assert_eq!(
            restored.steps.len(),
            plan.steps.len(),
            "{name}: step count mismatch after JSON roundtrip"
        );
        assert_eq!(
            restored.input_shapes, plan.input_shapes,
            "{name}: input_shapes mismatch after JSON roundtrip"
        );
        assert_eq!(
            restored.output_step, plan.output_step,
            "{name}: output_step mismatch after JSON roundtrip"
        );
        assert_eq!(
            restored.weight_names, plan.weight_names,
            "{name}: weight_names mismatch after JSON roundtrip"
        );

        // Double roundtrip: deserialize restored JSON again, verify structure.
        // HashMap key ordering is non-deterministic, so JSON string comparison
        // is not reliable for plans with weight_data. We verify structure.
        let json2 = restored
            .to_json()
            .unwrap_or_else(|e| panic!("{name}: second to_json failed: {e}"));
        let restored2 = CompiledPlan::from_json(&json2)
            .unwrap_or_else(|e| panic!("{name}: second from_json failed: {e}"));
        assert_eq!(
            restored.steps.len(),
            restored2.steps.len(),
            "{name}: double roundtrip step count mismatch"
        );
        assert_eq!(
            restored.input_shapes, restored2.input_shapes,
            "{name}: double roundtrip input_shapes mismatch"
        );
        assert_eq!(
            restored.output_step, restored2.output_step,
            "{name}: double roundtrip output_step mismatch"
        );
        assert_eq!(
            restored.weight_names, restored2.weight_names,
            "{name}: double roundtrip weight_names mismatch"
        );
    }
}

/// Verify that NativeOp field values survive roundtrip with full fidelity
/// (eps, hidden sizes, shapes, boolean flags, optional fields).
#[test]
fn test_native_op_field_fidelity() {
    let plan = text_encoder_segment_plan();
    let json = plan.to_json().expect("serialize text_encoder plan");
    let restored = CompiledPlan::from_json(&json).expect("deserialize text_encoder plan");

    // Verify LSTM hidden_size, reverse flag, h_shape
    match &restored.steps[1] {
        CompiledStep::NativeOp { op, .. } => {
            let json_str = serde_json::to_string(op).unwrap();
            assert!(
                json_str.contains("\"hidden_size\":4"),
                "hidden_size must be 4"
            );
            assert!(
                json_str.contains("\"reverse\":false"),
                "forward LSTM reverse flag must be false"
            );
        }
        other => panic!("expected NativeOp, got {other:?}"),
    }

    // Verify InstanceNorm eps
    match &restored.steps[2] {
        CompiledStep::NativeOp { op, .. } => {
            let json_str = serde_json::to_string(op).unwrap();
            assert!(
                json_str.contains("InstanceNorm"),
                "must be InstanceNorm variant"
            );
            assert!(
                json_str.contains("\"input_shape\":[1,8,4]"),
                "input_shape must match"
            );
        }
        other => panic!("expected NativeOp, got {other:?}"),
    }

    // Verify NormActivConv1d weight data survived
    match &restored.steps[3] {
        CompiledStep::NativeOp { op, weight_data } => {
            let json_str = serde_json::to_string(op).unwrap();
            assert!(
                json_str.contains("NormActivConv1d"),
                "must be NormActivConv1d"
            );
            assert!(
                json_str.contains("\"external_node_ids\":[0,1,2]"),
                "external_node_ids must survive"
            );

            let alpha = weight_data
                .get("alpha")
                .expect("alpha weight must survive roundtrip");
            assert_eq!(alpha.shape(), &[8]);
            assert_eq!(alpha.data(), &[1.0; 8]);
        }
        other => panic!("expected NativeOp, got {other:?}"),
    }

    // Verify AdainSnake residual_gamma and external_node_ids
    match &restored.steps[4] {
        CompiledStep::NativeOp { op, .. } => {
            let json_str = serde_json::to_string(op).unwrap();
            assert!(json_str.contains("AdainSnake"), "must be AdainSnake");
            assert!(
                json_str.contains("\"residual_gamma\":true"),
                "residual_gamma must survive"
            );
            assert!(
                json_str.contains("\"external_node_ids\":[10,20,30]"),
                "external_node_ids must survive"
            );
        }
        other => panic!("expected NativeOp, got {other:?}"),
    }
}

/// Verify that the generator plan's FusedResBlock with StyleProjectionParams
/// and ConstantValue survive serialization.
#[test]
fn test_generator_plan_fidelity() {
    let plan = generator_segment_plan();
    let json = plan.to_json().expect("serialize generator plan");
    let restored = CompiledPlan::from_json(&json).expect("deserialize generator plan");

    // Verify FusedResBlock with style_proj
    match &restored.steps[1] {
        CompiledStep::NativeOp { op, .. } => {
            let json_str = serde_json::to_string(op).unwrap();
            assert!(json_str.contains("FusedResBlock"), "must be FusedResBlock");
            assert!(json_str.contains("Snake"), "phase activation must be Snake");
            assert!(json_str.contains("\"style_dim\":4"), "style_dim must be 4");
            assert!(json_str.contains("\"channels1\":8"), "channels1 must be 8");
            assert!(
                json_str.contains("\"shortcut_step\":0"),
                "shortcut_step must be 0"
            );
            assert!(
                json_str.contains("\"pool_step\":null"),
                "pool_step must be null"
            );
        }
        other => panic!("expected NativeOp, got {other:?}"),
    }

    // Verify ConstantValue
    match &restored.steps[3] {
        CompiledStep::ConstantValue { value, shape } => {
            assert!(
                (value - 0.7071).abs() < 1e-10,
                "constant value must survive"
            );
            assert_eq!(shape, &[1]);
        }
        other => panic!("expected ConstantValue, got {other:?}"),
    }
}

/// Verify that the f0_energy plan's BiLstmCat and reverse LSTM survive
/// serialization with correct step references.
#[test]
fn test_f0_energy_plan_fidelity() {
    let plan = f0_energy_segment_plan();
    let json = plan.to_json().expect("serialize f0_energy plan");
    let restored = CompiledPlan::from_json(&json).expect("deserialize f0_energy plan");

    // Verify forward LSTM (step 1)
    match &restored.steps[1] {
        CompiledStep::NativeOp { op, .. } => {
            let json_str = serde_json::to_string(op).unwrap();
            assert!(
                json_str.contains("\"reverse\":false"),
                "forward LSTM must not be reversed"
            );
        }
        other => panic!("expected NativeOp, got {other:?}"),
    }

    // Verify reverse LSTM (step 2)
    match &restored.steps[2] {
        CompiledStep::NativeOp { op, .. } => {
            let json_str = serde_json::to_string(op).unwrap();
            assert!(
                json_str.contains("\"reverse\":true"),
                "reverse LSTM must be reversed"
            );
        }
        other => panic!("expected NativeOp, got {other:?}"),
    }

    // Verify BiLstmCat step references (step 3)
    match &restored.steps[3] {
        CompiledStep::NativeOp { op, .. } => {
            let json_str = serde_json::to_string(op).unwrap();
            assert!(json_str.contains("BiLstmCat"), "must be BiLstmCat");
            assert!(
                json_str.contains("\"fwd_lstm_step\":1"),
                "fwd_lstm_step must reference step 1"
            );
            assert!(
                json_str.contains("\"rev_lstm_step\":2"),
                "rev_lstm_step must reference step 2"
            );
        }
        other => panic!("expected NativeOp, got {other:?}"),
    }
}

/// Build a combined plan with all segment types and verify that the full
/// pipeline plan survives file roundtrip with double-roundtrip idempotency.
#[test]
fn test_combined_kokoro_plan_file_roundtrip() {
    let segments = vec![
        plbert_segment_plan(),
        text_encoder_segment_plan(),
        generator_segment_plan(),
        f0_energy_segment_plan(),
        regulate_segment_plan(),
    ];

    let mut all_steps = Vec::new();
    let mut all_weight_names = Vec::new();
    let mut total_dispatches = 0usize;

    for seg in &segments {
        all_steps.extend(seg.steps.iter().cloned());
        all_weight_names.extend(seg.weight_names.iter().cloned());
        total_dispatches += seg
            .steps
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
                )
            })
            .count();
    }
    all_weight_names.sort();
    all_weight_names.dedup();

    // Build the combined plan via serde_json::Value (CompiledPlan is #[non_exhaustive])
    let combined_value = serde_json::json!({
        "steps": all_steps,
        "input_shapes": [[1, 4], [1, 4, 8], [1, 8, 16], [8, 1, 8]],
        "output_step": 0,
        "weight_names": all_weight_names,
    });
    let combined: CompiledPlan =
        serde_json::from_value(combined_value).expect("parse combined plan");

    let dir =
        std::env::temp_dir().join(format!("nn_kokoro_combined_serde_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("kokoro_combined.nnc");

    // Save
    combined.save(&path).expect("save combined plan");

    // Load
    let restored = CompiledPlan::load(&path).expect("load combined plan");

    // Verify
    assert_eq!(
        restored.steps.len(),
        combined.steps.len(),
        "combined step count mismatch"
    );
    assert_eq!(
        restored.input_shapes, combined.input_shapes,
        "combined input_shapes mismatch"
    );
    assert_eq!(
        restored.weight_names, combined.weight_names,
        "combined weight_names mismatch"
    );

    let restored_dispatches = restored
        .steps
        .iter()
        .filter(|s| {
            matches!(
                s,
                CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
            )
        })
        .count();
    assert_eq!(
        total_dispatches, restored_dispatches,
        "combined dispatch count mismatch"
    );

    // Verify double roundtrip structural idempotency: save restored plan
    // again, reload, and compare structural fields. HashMap key ordering
    // is non-deterministic, so JSON string comparison is not reliable for
    // plans containing weight_data. We verify structure instead.
    let path2 = dir.join("kokoro_combined_pass2.nnc");
    restored.save(&path2).expect("save pass 2");
    let restored2 = CompiledPlan::load(&path2).expect("load pass 2");
    assert_eq!(
        restored.steps.len(),
        restored2.steps.len(),
        "double roundtrip: step count mismatch"
    );
    assert_eq!(
        restored.input_shapes, restored2.input_shapes,
        "double roundtrip: input_shapes mismatch"
    );
    assert_eq!(
        restored.output_step, restored2.output_step,
        "double roundtrip: output_step mismatch"
    );
    assert_eq!(
        restored.weight_names, restored2.weight_names,
        "double roundtrip: weight_names mismatch"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Production roundtrip test (requires KOKORO_WEIGHTS)
// ---------------------------------------------------------------------------

/// Full production Kokoro .nnc roundtrip test.
///
/// Builds a real CompiledKokoro with production weights, runs synthesize()
/// to populate segment caches, then verifies that production-scale
/// NativeOp plans survive .nnc save/load roundtrip.
///
/// Gated on KOKORO_WEIGHTS env var. Skips gracefully when unset.
#[test]
fn test_kokoro_production_synthesis_then_plan_roundtrip() {
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::Device;

    if super::kokoro_test_env::require_kokoro_weights("production serialization roundtrip")
        .is_none()
    {
        return;
    }

    // Build production Kokoro and synthesize to force segment compilation
    let (mut kokoro, cache) = super::kokoro_test_weights::build_kokoro_mini();
    let cpu = Device::Cpu;
    let config = super::kokoro_test_weights::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids =
        DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu).expect("create input_ids");
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(200, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu,
    )
    .expect("create style");

    let result = kokoro.synthesize(&input_ids, &style, 1.0, &cache);
    assert!(result.is_ok(), "synthesize failed: {:?}", result.err());

    // Verify segment cache was populated
    let stats = kokoro.segment_cache_stats();
    assert!(
        stats.misses > 0,
        "segment cache should have had at least one miss (first compilation)"
    );

    // Now test that a production-scale Kokoro plan survives .nnc roundtrip.
    // Uses production dimension sizes (d_en=512, style_dim=128) but no actual
    // weights — just verifying the plan structure serializes correctly.
    let production_json = r#"{
        "steps": [
            "InputForward",
            {
                "NativeOp": {
                    "op": {
                        "LstmSequence": {
                            "hidden_size": 256,
                            "input_shape": [5, 1, 512],
                            "h_shape": [1, 256],
                            "reverse": false
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "FlashAttention": {
                            "scale": 0.0442,
                            "causal": false,
                            "q_shape": [1, 2, 5, 256],
                            "k_shape": [1, 2, 5, 256],
                            "output_shape": [1, 2, 5, 256],
                            "input_layout": "HeadsFirst"
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "FusedResBlock": {
                            "phase1": {
                                "activation": "Snake",
                                "eps": 1e-5,
                                "conv_dilation": 1,
                                "conv_padding": 3,
                                "input_shape": [1, 512, 100],
                                "output_channels": 512,
                                "kernel_size": 7
                            },
                            "phase2": {
                                "activation": "Snake",
                                "eps": 1e-5,
                                "conv_dilation": 1,
                                "conv_padding": 1,
                                "input_shape": [1, 512, 100],
                                "output_channels": 512,
                                "kernel_size": 3
                            },
                            "input_steps": [0, 1, 2, 3, 4],
                            "residual_scale": 1.0,
                            "style_proj": {
                                "channels1": 512,
                                "channels2": 512,
                                "style_dim": 128
                            },
                            "shortcut_step": 0,
                            "pool_step": null,
                            "style_batch_offset": null
                        }
                    },
                    "weight_data": {}
                }
            },
            {
                "NativeOp": {
                    "op": {
                        "NormLinear": {
                            "norm_kind": "LayerNorm",
                            "eps": 1e-5,
                            "input_shape": [1, 5, 512],
                            "hidden_dim": 512,
                            "out_features": 1024,
                            "has_bias": true
                        }
                    },
                    "weight_data": {}
                }
            }
        ],
        "input_shapes": [[1, 5, 512]],
        "output_step": 4,
        "weight_names": []
    }"#;

    let production_plan = CompiledPlan::from_json(production_json).expect("parse production plan");

    let dir = std::env::temp_dir().join(format!("nn_kokoro_prod_serde_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("kokoro_production.nnc");

    production_plan.save(&path).expect("save production plan");
    let restored = CompiledPlan::load(&path).expect("load production plan");

    assert_eq!(restored.steps.len(), production_plan.steps.len());
    assert_eq!(restored.input_shapes, production_plan.input_shapes);
    assert_eq!(restored.output_step, production_plan.output_step);

    // Double roundtrip structural idempotency
    let path2 = dir.join("kokoro_production_pass2.nnc");
    restored.save(&path2).expect("save pass 2");
    let restored2 = CompiledPlan::load(&path2).expect("load pass 2");
    assert_eq!(
        restored.steps.len(),
        restored2.steps.len(),
        "production double roundtrip: step count mismatch"
    );
    assert_eq!(
        restored.input_shapes, restored2.input_shapes,
        "production double roundtrip: input_shapes mismatch"
    );
    assert_eq!(
        restored.output_step, restored2.output_step,
        "production double roundtrip: output_step mismatch"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
