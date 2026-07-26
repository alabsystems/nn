// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Free-function convenience API for saving and loading [`CompiledPlan`]s.
//!
//! Gated behind the `plan-serde` feature flag. Delegates to
//! [`CompiledPlan::save`] / [`CompiledPlan::load`] from `compiled_plan_serde`.

use std::path::Path;

use crate::trace_compile::{CompiledPlan, CompiledPlanSerdeError};

/// Save a [`CompiledPlan`] to a JSON file at `path`.
///
/// Creates or overwrites the file. Uses pretty-printed JSON for
/// debuggability.
///
/// # Errors
///
/// Returns [`CompiledPlanSerdeError`] on I/O or serialization failure.
pub fn save_plan(
    plan: &CompiledPlan,
    path: impl AsRef<Path>,
) -> Result<(), CompiledPlanSerdeError> {
    plan.save(path)
}

/// Load a [`CompiledPlan`] from a JSON file at `path`.
///
/// # Errors
///
/// Returns [`CompiledPlanSerdeError`] on I/O or deserialization failure.
pub fn load_plan(path: impl AsRef<Path>) -> Result<CompiledPlan, CompiledPlanSerdeError> {
    CompiledPlan::load(path)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nn_core::dyn_tensor::trace::WeightRef;

    use crate::ir::{IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType};
    use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
    use crate::trace_compile::{CompiledKernel, CompiledPlan, CompiledStep};

    use super::{load_plan, save_plan};

    /// Create a temporary directory unique to this test run.
    fn test_dir(suffix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nn_plan_io_{}_{}_{}",
            suffix,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Build a minimal relu TensorKernelDef for use in Dispatch steps.
    fn make_relu_kernel(shape: Vec<usize>) -> CompiledKernel {
        let relu_kernel = KernelDef::new(
            "relu",
            vec![Param::new("x", ScalarType::F32)],
            ScalarType::F32,
            vec![
                IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
                IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
                IRNode::new(
                    NodeId::new(2),
                    IRNodeKind::MinMax {
                        op: MinMaxKind::Max,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );

        let tensor_def = TensorKernelDef::new(
            "elementwise_relu",
            vec![
                TensorNode::new(
                    TensorNodeId::new(0),
                    TensorOpKind::Input {
                        name: "x".to_string(),
                        shape: shape.clone(),
                    },
                    shape.clone(),
                ),
                TensorNode::new(
                    TensorNodeId::new(1),
                    TensorOpKind::Elementwise {
                        kernel: relu_kernel,
                        inputs: vec![TensorNodeId::new(0)],
                    },
                    shape,
                ),
            ],
            TensorNodeId::new(1),
        );

        CompiledKernel::new(tensor_def)
    }

    fn make_test_plan() -> CompiledPlan {
        let weight = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mut weight_data = HashMap::new();
        weight_data.insert("layer.weight".to_string(), weight);

        CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::Dispatch {
                    kernel: make_relu_kernel(vec![2, 3]),
                    weight_data,
                    external_node_ids: None,
                },
            ],
            input_shapes: vec![vec![2, 3]],
            output_step: 1,
            weight_names: vec!["layer.weight".to_string()],
        }
    }

    #[test]
    fn test_save_load_plan_round_trip() {
        let plan = make_test_plan();
        let dir = test_dir("basic");
        let path = dir.join("test_plan.json");

        save_plan(&plan, &path).expect("save_plan should succeed");
        let restored = load_plan(&path).expect("load_plan should succeed");

        assert_eq!(restored.steps.len(), plan.steps.len());
        assert_eq!(restored.input_shapes, plan.input_shapes);
        assert_eq!(restored.output_step, plan.output_step);
        assert_eq!(restored.weight_names, plan.weight_names);

        // Verify weight data survived the round-trip
        match &restored.steps[1] {
            CompiledStep::Dispatch { weight_data, .. } => {
                let w = weight_data.get("layer.weight").expect("weight present");
                assert_eq!(w.data(), &[1.0, 2.0, 3.0, 4.0]);
                assert_eq!(w.shape(), &[2, 2]);
            }
            other => panic!("expected Dispatch, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Empty plan: no steps, no weights, no inputs.
    #[test]
    fn test_save_load_empty_plan() {
        let plan = CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        };

        let dir = test_dir("empty");
        let path = dir.join("empty.json");

        save_plan(&plan, &path).expect("save empty plan");
        let restored = load_plan(&path).expect("load empty plan");

        assert!(restored.steps.is_empty());
        assert!(restored.input_shapes.is_empty());
        assert_eq!(restored.output_step, 0);
        assert!(restored.weight_names.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Plan with all basic CompiledStep variants (InputForward, Passthrough,
    /// IdentityPassthrough, Dispatch, ConstantValue, NarrowView, RuntimeOp).
    #[test]
    fn test_save_load_all_compiled_step_variants() {
        use crate::trace_compile::RuntimeOpKind;

        let weight = WeightRef::new(vec![0.5; 6], vec![2, 3]).unwrap();
        let mut weight_data = HashMap::new();
        weight_data.insert("w".to_string(), weight);

        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::Dispatch {
                    kernel: make_relu_kernel(vec![2, 3]),
                    weight_data,
                    external_node_ids: Some(vec![0, 1]),
                },
                CompiledStep::Passthrough {
                    op_name: "reshape".to_string(),
                    output_shape: vec![6],
                },
                CompiledStep::IdentityPassthrough,
                CompiledStep::ConstantValue {
                    value: 3.14159,
                    shape: vec![1, 1],
                },
                CompiledStep::NarrowView {
                    byte_offset: 512,
                    output_shape: vec![1, 3],
                    source_step: Some(1),
                },
                CompiledStep::NarrowView {
                    byte_offset: 0,
                    output_shape: vec![2, 3],
                    source_step: None,
                },
                CompiledStep::RuntimeOp {
                    op: RuntimeOpKind::RepeatInterleave {
                        dim: 0,
                        input_shape: vec![4, 8],
                        counts_shape: vec![4],
                    },
                },
            ],
            input_shapes: vec![vec![2, 3]],
            output_step: 2,
            weight_names: vec!["w".to_string()],
        };

        let dir = test_dir("all_variants");
        let path = dir.join("all_variants.json");

        save_plan(&plan, &path).expect("save all-variants plan");
        let restored = load_plan(&path).expect("load all-variants plan");

        assert_eq!(restored.steps.len(), 8);
        assert_eq!(restored.input_shapes, vec![vec![2, 3]]);
        assert_eq!(restored.output_step, 2);
        assert_eq!(restored.weight_names, vec!["w".to_string()]);

        // Verify InputForward (step 0)
        assert!(matches!(restored.steps[0], CompiledStep::InputForward));

        // Verify Dispatch with external_node_ids (step 1)
        match &restored.steps[1] {
            CompiledStep::Dispatch {
                kernel,
                weight_data,
                external_node_ids,
            } => {
                assert_eq!(kernel.name(), "elementwise_relu");
                assert_eq!(external_node_ids.as_deref(), Some(&[0u64, 1][..]));
                let w = weight_data.get("w").expect("weight present");
                assert_eq!(w.data(), &[0.5; 6]);
                assert_eq!(w.shape(), &[2, 3]);
            }
            other => panic!("expected Dispatch, got {other:?}"),
        }

        // Verify Passthrough (step 2)
        match &restored.steps[2] {
            CompiledStep::Passthrough {
                op_name,
                output_shape,
            } => {
                assert_eq!(op_name, "reshape");
                assert_eq!(output_shape, &[6]);
            }
            other => panic!("expected Passthrough, got {other:?}"),
        }

        // Verify IdentityPassthrough (step 3)
        assert!(matches!(
            restored.steps[3],
            CompiledStep::IdentityPassthrough
        ));

        // Verify ConstantValue (step 4)
        match &restored.steps[4] {
            CompiledStep::ConstantValue { value, shape } => {
                assert!((value - 3.14159).abs() < 1e-10);
                assert_eq!(shape, &[1, 1]);
            }
            other => panic!("expected ConstantValue, got {other:?}"),
        }

        // Verify NarrowView with source_step (step 5)
        match &restored.steps[5] {
            CompiledStep::NarrowView {
                byte_offset,
                output_shape,
                source_step,
            } => {
                assert_eq!(*byte_offset, 512);
                assert_eq!(output_shape, &[1, 3]);
                assert_eq!(*source_step, Some(1));
            }
            other => panic!("expected NarrowView, got {other:?}"),
        }

        // Verify NarrowView without source_step (step 6)
        match &restored.steps[6] {
            CompiledStep::NarrowView {
                byte_offset,
                output_shape,
                source_step,
            } => {
                assert_eq!(*byte_offset, 0);
                assert_eq!(output_shape, &[2, 3]);
                assert_eq!(*source_step, None);
            }
            other => panic!("expected NarrowView, got {other:?}"),
        }

        // Verify RuntimeOp (step 7)
        match &restored.steps[7] {
            CompiledStep::RuntimeOp { op } => match op {
                RuntimeOpKind::RepeatInterleave {
                    dim,
                    input_shape,
                    counts_shape,
                } => {
                    assert_eq!(*dim, 0);
                    assert_eq!(input_shape, &[4, 8]);
                    assert_eq!(counts_shape, &[4]);
                }
            },
            other => panic!("expected RuntimeOp, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Plan with NativeOp steps covering several representative variants.
    #[test]
    fn test_save_load_native_op_plan() {
        use crate::trace_compile::{AttentionLayout, NativeOpKind, NormActivation};

        let alpha_weight = WeightRef::new(vec![1.0; 64], vec![64]).unwrap();
        let mut weight_data = HashMap::new();
        weight_data.insert("alpha".to_string(), alpha_weight);

        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::NativeOp {
                    op: NativeOpKind::InstanceNorm {
                        eps: 1e-5,
                        input_shape: vec![1, 64, 128],
                    },
                    weight_data: HashMap::new(),
                },
                CompiledStep::NativeOp {
                    op: NativeOpKind::FlashAttention {
                        scale: 0.125,
                        causal: true,
                        q_shape: vec![1, 8, 32, 64],
                        k_shape: vec![1, 8, 32, 64],
                        output_shape: vec![1, 8, 32, 64],
                        input_layout: AttentionLayout::SeqFirst,
                    },
                    weight_data: HashMap::new(),
                },
                CompiledStep::NativeOp {
                    op: NativeOpKind::NormActivConv1d {
                        activation: NormActivation::Snake,
                        eps: 1e-5,
                        conv_dilation: 3,
                        conv_padding: 3,
                        input_shape: vec![1, 64, 100],
                        output_channels: 128,
                        kernel_size: 3,
                        external_node_ids: Some(vec![10, 20, 30]),
                    },
                    weight_data,
                },
                CompiledStep::NativeOp {
                    op: NativeOpKind::LstmSequence {
                        hidden_size: 256,
                        input_shape: vec![50, 1, 512],
                        h_shape: vec![1, 256],
                        reverse: true,
                    },
                    weight_data: HashMap::new(),
                },
            ],
            input_shapes: vec![vec![1, 64, 128]],
            output_step: 4,
            weight_names: vec!["alpha".to_string()],
        };

        let dir = test_dir("native_ops");
        let path = dir.join("native.json");

        save_plan(&plan, &path).expect("save NativeOp plan");
        let restored = load_plan(&path).expect("load NativeOp plan");

        assert_eq!(restored.steps.len(), 5);
        assert_eq!(restored.output_step, 4);
        assert_eq!(restored.weight_names, vec!["alpha"]);

        // Verify InstanceNorm eps
        match &restored.steps[1] {
            CompiledStep::NativeOp { op, .. } => match op {
                NativeOpKind::InstanceNorm { eps, input_shape } => {
                    assert!((eps - 1e-5).abs() < 1e-10);
                    assert_eq!(input_shape, &[1, 64, 128]);
                }
                other => panic!("expected InstanceNorm, got {other:?}"),
            },
            other => panic!("expected NativeOp, got {other:?}"),
        }

        // Verify FlashAttention SeqFirst layout
        match &restored.steps[2] {
            CompiledStep::NativeOp { op, .. } => match op {
                NativeOpKind::FlashAttention {
                    scale,
                    causal,
                    input_layout,
                    ..
                } => {
                    assert!((scale - 0.125).abs() < 1e-10);
                    assert!(*causal);
                    assert!(matches!(input_layout, AttentionLayout::SeqFirst));
                }
                other => panic!("expected FlashAttention, got {other:?}"),
            },
            other => panic!("expected NativeOp, got {other:?}"),
        }

        // Verify NormActivConv1d alpha weight + external_node_ids
        match &restored.steps[3] {
            CompiledStep::NativeOp { op, weight_data } => {
                match op {
                    NativeOpKind::NormActivConv1d {
                        activation,
                        conv_dilation,
                        conv_padding,
                        output_channels,
                        kernel_size,
                        external_node_ids,
                        ..
                    } => {
                        assert!(matches!(activation, NormActivation::Snake));
                        assert_eq!(*conv_dilation, 3);
                        assert_eq!(*conv_padding, 3);
                        assert_eq!(*output_channels, 128);
                        assert_eq!(*kernel_size, 3);
                        assert_eq!(external_node_ids.as_deref(), Some(&[10u64, 20, 30][..]));
                    }
                    other => panic!("expected NormActivConv1d, got {other:?}"),
                }
                let alpha = weight_data.get("alpha").expect("alpha weight");
                assert_eq!(alpha.shape(), &[64]);
                assert_eq!(alpha.data().len(), 64);
            }
            other => panic!("expected NativeOp, got {other:?}"),
        }

        // Verify LstmSequence reverse flag
        match &restored.steps[4] {
            CompiledStep::NativeOp { op, .. } => match op {
                NativeOpKind::LstmSequence {
                    hidden_size,
                    reverse,
                    ..
                } => {
                    assert_eq!(*hidden_size, 256);
                    assert!(*reverse);
                }
                other => panic!("expected LstmSequence, got {other:?}"),
            },
            other => panic!("expected NativeOp, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Plans with empty weight_names and multiple input shapes.
    #[test]
    fn test_save_load_empty_weights_multiple_inputs() {
        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::InputForward,
                CompiledStep::InputForward,
                CompiledStep::Passthrough {
                    op_name: "cat".to_string(),
                    output_shape: vec![3, 100],
                },
            ],
            input_shapes: vec![vec![1, 100], vec![1, 100], vec![1, 100]],
            output_step: 3,
            weight_names: vec![],
        };

        let dir = test_dir("multi_input");
        let path = dir.join("multi.json");

        save_plan(&plan, &path).expect("save multi-input plan");
        let restored = load_plan(&path).expect("load multi-input plan");

        assert_eq!(restored.input_shapes.len(), 3);
        assert!(restored.weight_names.is_empty());
        assert_eq!(restored.output_step, 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Plan with large input shapes (high rank, large dimensions).
    #[test]
    fn test_save_load_large_input_shapes() {
        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::Passthrough {
                    op_name: "identity".to_string(),
                    output_shape: vec![4, 128, 256, 256, 3],
                },
            ],
            input_shapes: vec![vec![4, 128, 256, 256, 3]],
            output_step: 1,
            weight_names: vec![],
        };

        let dir = test_dir("large_shapes");
        let path = dir.join("large.json");

        save_plan(&plan, &path).expect("save large-shape plan");
        let restored = load_plan(&path).expect("load large-shape plan");

        assert_eq!(restored.input_shapes, vec![vec![4, 128, 256, 256, 3]]);
        match &restored.steps[1] {
            CompiledStep::Passthrough { output_shape, .. } => {
                assert_eq!(output_shape, &[4, 128, 256, 256, 3]);
            }
            other => panic!("expected Passthrough, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Verify save_plan overwrites existing files and load reads the latest.
    #[test]
    fn test_save_overwrite_and_reload() {
        let plan1 = CompiledPlan {
            steps: vec![CompiledStep::InputForward],
            input_shapes: vec![vec![1]],
            output_step: 0,
            weight_names: vec![],
        };
        let plan2 = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::ConstantValue {
                    value: 42.0,
                    shape: vec![1],
                },
            ],
            input_shapes: vec![vec![1]],
            output_step: 1,
            weight_names: vec![],
        };

        let dir = test_dir("overwrite");
        let path = dir.join("plan.json");

        save_plan(&plan1, &path).expect("save plan1");
        let restored1 = load_plan(&path).expect("load plan1");
        assert_eq!(restored1.steps.len(), 1);

        save_plan(&plan2, &path).expect("save plan2 overwrite");
        let restored2 = load_plan(&path).expect("load plan2");
        assert_eq!(restored2.steps.len(), 2);
        match &restored2.steps[1] {
            CompiledStep::ConstantValue { value, .. } => {
                assert!((value - 42.0).abs() < 1e-10);
            }
            other => panic!("expected ConstantValue, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Load from non-existent path returns an I/O error.
    #[test]
    fn test_load_nonexistent_path_returns_error() {
        let result = load_plan("/tmp/nn_nonexistent_plan_12345.json");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, crate::trace_compile::CompiledPlanSerdeError::Io(_)),
            "expected Io error, got {err:?}"
        );
    }

    /// Load from file with invalid JSON returns a Json error.
    #[test]
    fn test_load_invalid_json_returns_error() {
        let dir = test_dir("invalid_json");
        let path = dir.join("bad.json");
        std::fs::write(&path, "{ not valid json !!!").unwrap();

        let result = load_plan(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, crate::trace_compile::CompiledPlanSerdeError::Json(_)),
            "expected Json error, got {err:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Verify to_json/from_json in-memory matches save/load file roundtrip.
    #[test]
    fn test_json_string_matches_file_roundtrip() {
        let plan = make_test_plan();

        let json_str = plan.to_json().expect("to_json");
        let from_str = CompiledPlan::from_json(&json_str).expect("from_json");

        let dir = test_dir("json_match");
        let path = dir.join("plan.json");
        save_plan(&plan, &path).expect("save");
        let from_file = load_plan(&path).expect("load");

        assert_eq!(from_str.steps.len(), from_file.steps.len());
        assert_eq!(from_str.input_shapes, from_file.input_shapes);
        assert_eq!(from_str.output_step, from_file.output_step);
        assert_eq!(from_str.weight_names, from_file.weight_names);

        // `save_plan` writes a versioned `.nnc` file: a 16-byte binary
        // NncHeader followed by the pretty-printed JSON payload. `to_json`
        // returns just that JSON payload (no header). So the file's JSON
        // payload (after the header) must equal the in-memory JSON string.
        let file_bytes = std::fs::read(&path).unwrap();
        let payload = std::str::from_utf8(&file_bytes[crate::nnc_header::NNC_HEADER_SIZE..])
            .expect("JSON payload after header should be valid UTF-8");
        assert_eq!(payload, json_str);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Plan with ConstantValue steps containing edge-case float values.
    #[test]
    fn test_save_load_constant_value_edge_cases() {
        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::ConstantValue {
                    value: 0.0,
                    shape: vec![],
                },
                CompiledStep::ConstantValue {
                    value: -0.0,
                    shape: vec![1],
                },
                CompiledStep::ConstantValue {
                    value: 1e-38,
                    shape: vec![1, 1],
                },
                CompiledStep::ConstantValue {
                    value: 1e38,
                    shape: vec![2],
                },
                CompiledStep::ConstantValue {
                    value: -1.0,
                    shape: vec![3, 3, 3],
                },
            ],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        };

        let dir = test_dir("const_edge");
        let path = dir.join("const.json");

        save_plan(&plan, &path).expect("save constant edge cases");
        let restored = load_plan(&path).expect("load constant edge cases");

        assert_eq!(restored.steps.len(), 5);

        match &restored.steps[0] {
            CompiledStep::ConstantValue { value, shape } => {
                assert!((value - 0.0).abs() < 1e-40);
                assert!(shape.is_empty());
            }
            other => panic!("expected ConstantValue, got {other:?}"),
        }

        match &restored.steps[2] {
            CompiledStep::ConstantValue { value, .. } => {
                assert!((*value - 1e-38).abs() < 1e-45);
            }
            other => panic!("expected ConstantValue, got {other:?}"),
        }

        match &restored.steps[3] {
            CompiledStep::ConstantValue { value, .. } => {
                assert!((*value - 1e38).abs() / 1e38 < 1e-10);
            }
            other => panic!("expected ConstantValue, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Plan with many weight names (stress the weight inventory).
    #[test]
    fn test_save_load_many_weights() {
        let mut weight_data = HashMap::new();
        let mut weight_names = Vec::new();
        for i in 0..50 {
            let name = format!("layer_{i}.weight");
            weight_data.insert(
                name.clone(),
                WeightRef::new(vec![i as f32; 4], vec![2, 2]).unwrap(),
            );
            weight_names.push(name);
        }
        weight_names.sort();

        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::Dispatch {
                    kernel: make_relu_kernel(vec![2, 2]),
                    weight_data,
                    external_node_ids: None,
                },
            ],
            input_shapes: vec![vec![2, 2]],
            output_step: 1,
            weight_names,
        };

        let dir = test_dir("many_weights");
        let path = dir.join("weights.json");

        save_plan(&plan, &path).expect("save many-weights plan");
        let restored = load_plan(&path).expect("load many-weights plan");

        assert_eq!(restored.weight_names.len(), 50);
        for pair in restored.weight_names.windows(2) {
            assert!(pair[0] <= pair[1], "weight_names must be sorted");
        }

        match &restored.steps[1] {
            CompiledStep::Dispatch { weight_data, .. } => {
                let w = weight_data.get("layer_25.weight").expect("w25");
                assert_eq!(w.data(), &[25.0; 4]);
            }
            other => panic!("expected Dispatch, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Double roundtrip: save -> load -> save -> load should be idempotent.
    #[test]
    fn test_double_roundtrip_idempotent() {
        use crate::trace_compile::NativeOpKind;

        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::NativeOp {
                    op: NativeOpKind::SiluMul {
                        input_shape: vec![1, 16, 512],
                    },
                    weight_data: HashMap::new(),
                },
                CompiledStep::ConstantValue {
                    value: 0.7071,
                    shape: vec![1],
                },
                CompiledStep::Passthrough {
                    op_name: "mul".to_string(),
                    output_shape: vec![1, 16, 512],
                },
            ],
            input_shapes: vec![vec![1, 16, 512]],
            output_step: 3,
            weight_names: vec![],
        };

        let dir = test_dir("double_rt");
        let path1 = dir.join("pass1.json");
        let path2 = dir.join("pass2.json");

        save_plan(&plan, &path1).expect("save pass1");
        let restored1 = load_plan(&path1).expect("load pass1");

        save_plan(&restored1, &path2).expect("save pass2");
        let restored2 = load_plan(&path2).expect("load pass2");

        // `.nnc` files start with a 16-byte binary header (including a
        // timestamp), so read raw bytes and compare the JSON payload after
        // the header. The header's timestamp differs between saves; the JSON
        // payload must be byte-identical across roundtrips.
        let bytes1 = std::fs::read(&path1).unwrap();
        let bytes2 = std::fs::read(&path2).unwrap();
        let json1 = std::str::from_utf8(&bytes1[crate::nnc_header::NNC_HEADER_SIZE..])
            .expect("pass1 JSON payload should be valid UTF-8");
        let json2 = std::str::from_utf8(&bytes2[crate::nnc_header::NNC_HEADER_SIZE..])
            .expect("pass2 JSON payload should be valid UTF-8");
        assert_eq!(json1, json2, "double roundtrip must produce identical JSON");

        assert_eq!(restored2.steps.len(), 4);
        assert_eq!(restored2.output_step, 3);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
