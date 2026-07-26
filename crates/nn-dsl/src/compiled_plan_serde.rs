// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! JSON serialization for [`CompiledPlan`].
//!
//! Enables saving compiled plans to disk and loading them back. The JSON
//! format is chosen for debuggability; binary formats (bincode) are a
//! future optimization.

use std::path::Path;

use super::CompiledPlan;

/// Errors from compiled plan serialization/deserialization.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CompiledPlanSerdeError {
    /// I/O error reading or writing the plan file.
    #[error("plan I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization or deserialization error.
    #[error("plan JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// `.nnc` header validation error (bad magic, unsupported version, truncated).
    #[error(".nnc header error: {0}")]
    Header(#[from] crate::nnc_header::NncError),
}

impl CompiledPlan {
    /// Serialize this plan to a versioned `.nnc` file at `path`.
    ///
    /// The file begins with a 16-byte [`NncHeader`](crate::nnc_header::NncHeader)
    /// (magic + version + timestamp) followed by pretty-printed JSON.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), CompiledPlanSerdeError> {
        use crate::nnc_header::NncHeader;

        let header = NncHeader::now();
        let json = serde_json::to_string_pretty(self)?;

        let mut buf = Vec::with_capacity(crate::nnc_header::NNC_HEADER_SIZE + json.len());
        buf.extend_from_slice(&header.to_bytes());
        buf.extend_from_slice(json.as_bytes());

        std::fs::write(path, buf)?;
        Ok(())
    }

    /// Deserialize a plan from a versioned `.nnc` file at `path`.
    ///
    /// Validates the [`NncHeader`](crate::nnc_header::NncHeader) (magic
    /// bytes, version range) before parsing the JSON payload. Returns
    /// [`CompiledPlanSerdeError::Header`] for header issues.
    ///
    /// Also supports legacy header-less JSON files for backward
    /// compatibility: if the first non-whitespace byte is `{`, the entire
    /// file is parsed as plain JSON.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CompiledPlanSerdeError> {
        use crate::nnc_header::{NncHeader, NNC_HEADER_SIZE, NNC_MAGIC};

        let data = std::fs::read(path)?;

        // Detect legacy (header-less) JSON files: starts with optional
        // whitespace then `{`.
        let is_legacy = data.iter().find(|b| !b.is_ascii_whitespace()).copied() == Some(b'{');

        if data.len() >= 4 && data[0..4] == NNC_MAGIC {
            // Versioned format.
            let header = NncHeader::from_bytes(&data)?;
            header.validate()?;
            let json = std::str::from_utf8(&data[NNC_HEADER_SIZE..])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let plan: Self = serde_json::from_str(json)?;
            Ok(plan)
        } else if is_legacy {
            // Legacy format: plain JSON without header.
            let json = std::str::from_utf8(&data)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let plan: Self = serde_json::from_str(json)?;
            Ok(plan)
        } else {
            // Neither versioned nor legacy JSON — try to parse the header
            // for a diagnostic error.
            let header = NncHeader::from_bytes(&data)?;
            header.validate()?;
            // If we reach here the header was fine but the payload is bad.
            Err(CompiledPlanSerdeError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file is neither a versioned .nnc nor a legacy JSON plan",
            )))
        }
    }

    /// Serialize this plan to a JSON string (in-memory, no header).
    pub fn to_json(&self) -> Result<String, CompiledPlanSerdeError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a plan from a JSON string (in-memory, no header).
    pub fn from_json(json: &str) -> Result<Self, CompiledPlanSerdeError> {
        Ok(serde_json::from_str(json)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nn_core::dyn_tensor::trace::WeightRef;

    use crate::ir::{IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};
    use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
    use crate::trace_compile::{CompiledKernel, CompiledPlan, CompiledStep};

    /// Build a minimal CompiledPlan for round-trip testing.
    fn make_test_plan() -> CompiledPlan {
        // Scalar kernel: relu(x) = max(x, 0.0)
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
                        op: crate::ir::MinMaxKind::Max,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        );

        // Tensor kernel wrapping the scalar relu
        let tensor_def = TensorKernelDef::new(
            "elementwise_relu",
            vec![
                TensorNode::new(
                    TensorNodeId::new(0),
                    TensorOpKind::Input {
                        name: "x".to_string(),
                        shape: vec![2, 3],
                    },
                    vec![2, 3],
                ),
                TensorNode::new(
                    TensorNodeId::new(1),
                    TensorOpKind::Elementwise {
                        kernel: relu_kernel,
                        inputs: vec![TensorNodeId::new(0)],
                    },
                    vec![2, 3],
                ),
            ],
            TensorNodeId::new(1),
        );

        let weight = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mut weight_data = HashMap::new();
        weight_data.insert("layer.weight".to_string(), weight);

        CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::Dispatch {
                    kernel: CompiledKernel::new(tensor_def),
                    weight_data,
                    external_node_ids: None,
                },
                CompiledStep::Passthrough {
                    op_name: "reshape".to_string(),
                    output_shape: vec![6],
                },
                CompiledStep::IdentityPassthrough,
            ],
            input_shapes: vec![vec![2, 3]],
            output_step: 2,
            weight_names: vec!["layer.weight".to_string()],
        }
    }

    #[test]
    fn test_compiled_plan_json_round_trip() {
        let plan = make_test_plan();
        let json = plan.to_json().expect("serialize");
        let restored = CompiledPlan::from_json(&json).expect("deserialize");

        // Verify structural equality
        assert_eq!(restored.steps.len(), plan.steps.len());
        assert_eq!(restored.input_shapes, plan.input_shapes);
        assert_eq!(restored.output_step, plan.output_step);
        assert_eq!(restored.weight_names, plan.weight_names);

        // Verify Dispatch step preserved weight data
        match &restored.steps[1] {
            CompiledStep::Dispatch { weight_data, .. } => {
                let w = weight_data.get("layer.weight").expect("weight present");
                assert_eq!(w.data(), &[1.0, 2.0, 3.0, 4.0]);
                assert_eq!(w.shape(), &[2, 2]);
            }
            other => panic!("expected Dispatch, got {other:?}"),
        }

        // Verify Passthrough preserved
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
    }

    #[test]
    fn test_compiled_plan_file_round_trip() {
        let plan = make_test_plan();
        let dir = std::env::temp_dir().join(format!("nn_plan_serde_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_plan.json");

        plan.save(&path).expect("save");
        let restored = CompiledPlan::load(&path).expect("load");

        assert_eq!(restored.steps.len(), plan.steps.len());
        assert_eq!(restored.output_step, plan.output_step);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compiled_plan_json_round_trip_with_runtime_op() {
        use crate::trace_compile::RuntimeOpKind;

        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::InputForward,
                CompiledStep::RuntimeOp {
                    op: RuntimeOpKind::RepeatInterleave {
                        dim: 0,
                        input_shape: vec![3, 16],
                        counts_shape: vec![3],
                    },
                },
                CompiledStep::Passthrough {
                    op_name: "reshape".to_string(),
                    output_shape: vec![48],
                },
            ],
            input_shapes: vec![vec![3, 16], vec![3]],
            output_step: 2,
            weight_names: vec![],
        };

        let json = plan.to_json().expect("serialize");
        let restored = CompiledPlan::from_json(&json).expect("deserialize");

        assert_eq!(restored.steps.len(), 4);
        assert_eq!(restored.weight_names.len(), 0);

        // Verify RuntimeOp preserved
        match &restored.steps[2] {
            CompiledStep::RuntimeOp { op } => match op {
                RuntimeOpKind::RepeatInterleave {
                    dim,
                    input_shape,
                    counts_shape,
                } => {
                    assert_eq!(*dim, 0);
                    assert_eq!(input_shape, &[3, 16]);
                    assert_eq!(counts_shape, &[3]);
                }
            },
            other => panic!("expected RuntimeOp, got {other:?}"),
        }

        // JSON should contain RuntimeOp marker
        assert!(json.contains("RuntimeOp"));
        assert!(json.contains("RepeatInterleave"));
    }

    #[test]
    fn test_compiled_plan_json_is_readable() {
        let plan = make_test_plan();
        let json = plan.to_json().expect("serialize");

        // JSON should contain human-readable field names
        assert!(json.contains("\"steps\""));
        assert!(json.contains("\"input_shapes\""));
        assert!(json.contains("\"weight_names\""));
        assert!(json.contains("\"layer.weight\""));
        assert!(json.contains("\"relu\""));
    }

    #[test]
    fn test_compiled_plan_json_round_trip_with_native_ops() {
        use crate::trace_compile::{AttentionLayout, NativeOpKind, NormActivation};

        let mut weight_data = HashMap::new();
        weight_data.insert(
            "alpha".to_string(),
            WeightRef::new(vec![1.0; 64], vec![64]).unwrap(),
        );
        weight_data.insert(
            "conv_weight".to_string(),
            WeightRef::new(vec![0.1; 64 * 64 * 3], vec![64, 64, 3]).unwrap(),
        );

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
                    op: NativeOpKind::NormActivConv1d {
                        activation: NormActivation::Snake,
                        eps: 1e-5,
                        conv_dilation: 1,
                        conv_padding: 1,
                        input_shape: vec![1, 64, 128],
                        output_channels: 64,
                        kernel_size: 3,
                        external_node_ids: Some(vec![0, 1, 2]),
                    },
                    weight_data: weight_data.clone(),
                },
                CompiledStep::NativeOp {
                    op: NativeOpKind::FlashAttention {
                        scale: 0.125,
                        causal: true,
                        q_shape: vec![1, 8, 16, 64],
                        k_shape: vec![1, 8, 16, 64],
                        output_shape: vec![1, 8, 16, 64],
                        input_layout: AttentionLayout::SeqFirst,
                    },
                    weight_data: HashMap::new(),
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
                CompiledStep::NativeOp {
                    op: NativeOpKind::AdainSnake {
                        eps: 1e-5,
                        input_shape: vec![1, 256, 100],
                        channels: 256,
                        residual_gamma: true,
                        external_node_ids: Some(vec![10, 20, 30]),
                    },
                    weight_data: HashMap::new(),
                },
                CompiledStep::ConstantValue {
                    value: 0.7071,
                    shape: vec![1],
                },
                CompiledStep::NarrowView {
                    byte_offset: 1024,
                    output_shape: vec![1, 64, 64],
                    source_step: Some(3),
                },
            ],
            input_shapes: vec![vec![1, 64, 128]],
            output_step: 7,
            weight_names: vec!["alpha".into(), "conv_weight".into()],
        };

        // Serialize
        let json = plan.to_json().expect("serialize NativeOp plan");

        // Verify JSON contains NativeOp variant names
        assert!(json.contains("InstanceNorm"), "must contain InstanceNorm");
        assert!(
            json.contains("NormActivConv1d"),
            "must contain NormActivConv1d"
        );
        assert!(
            json.contains("FlashAttention"),
            "must contain FlashAttention"
        );
        assert!(json.contains("LstmSequence"), "must contain LstmSequence");
        assert!(json.contains("AdainSnake"), "must contain AdainSnake");

        // Deserialize
        let restored = CompiledPlan::from_json(&json).expect("deserialize NativeOp plan");

        // Verify structural equality
        assert_eq!(restored.steps.len(), plan.steps.len());
        assert_eq!(restored.input_shapes, plan.input_shapes);
        assert_eq!(restored.output_step, plan.output_step);
        assert_eq!(restored.weight_names, plan.weight_names);

        // Verify NativeOp fields survived round-trip
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

        // Verify NormActivConv1d with external_node_ids
        match &restored.steps[2] {
            CompiledStep::NativeOp { op, weight_data } => {
                match op {
                    NativeOpKind::NormActivConv1d {
                        activation,
                        external_node_ids,
                        output_channels,
                        kernel_size,
                        ..
                    } => {
                        assert!(
                            matches!(activation, NormActivation::Snake),
                            "activation must be Snake"
                        );
                        assert_eq!(*output_channels, 64);
                        assert_eq!(*kernel_size, 3);
                        assert_eq!(external_node_ids.as_deref(), Some(&[0u64, 1, 2][..]));
                    }
                    other => panic!("expected NormActivConv1d, got {other:?}"),
                }
                assert!(weight_data.contains_key("alpha"));
                assert!(weight_data.contains_key("conv_weight"));
            }
            other => panic!("expected NativeOp, got {other:?}"),
        }

        // Verify LSTM reverse flag round-trip
        match &restored.steps[4] {
            CompiledStep::NativeOp { op, .. } => match op {
                NativeOpKind::LstmSequence { reverse, .. } => {
                    assert!(*reverse, "reverse flag must survive round-trip");
                }
                other => panic!("expected LstmSequence, got {other:?}"),
            },
            other => panic!("expected NativeOp, got {other:?}"),
        }

        // Verify ConstantValue
        match &restored.steps[6] {
            CompiledStep::ConstantValue { value, shape } => {
                assert!((value - 0.7071).abs() < 1e-10);
                assert_eq!(shape, &[1]);
            }
            other => panic!("expected ConstantValue, got {other:?}"),
        }

        // Verify NarrowView with source_step
        match &restored.steps[7] {
            CompiledStep::NarrowView {
                byte_offset,
                output_shape,
                source_step,
            } => {
                assert_eq!(*byte_offset, 1024);
                assert_eq!(output_shape, &[1, 64, 64]);
                assert_eq!(*source_step, Some(3));
            }
            other => panic!("expected NarrowView, got {other:?}"),
        }
    }

    /// Round-trip test covering all NativeOpKind variants to verify complete serde
    /// derive coverage. Each variant must serialize and deserialize without error.
    /// Part of #3772, #3764.
    #[test]
    fn test_compiled_plan_json_round_trip_all_native_op_variants() {
        use crate::trace_compile::{
            AttentionLayout, FusedNormKind, GemmActivation, NativeOpKind, NormActivConv1dParams,
            NormActivation, StyleBatchOffset, StyleProjectionParams,
        };

        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                // LstmSequence
                CompiledStep::NativeOp {
                    op: NativeOpKind::LstmSequence {
                        hidden_size: 128,
                        input_shape: vec![10, 1, 256],
                        h_shape: vec![1, 128],
                        reverse: false,
                    },
                    weight_data: HashMap::new(),
                },
                // Cumsum
                CompiledStep::NativeOp {
                    op: NativeOpKind::Cumsum {
                        dim: 1,
                        input_shape: vec![1, 64],
                    },
                    weight_data: HashMap::new(),
                },
                // InstanceNorm
                CompiledStep::NativeOp {
                    op: NativeOpKind::InstanceNorm {
                        eps: 1e-5,
                        input_shape: vec![1, 32, 100],
                    },
                    weight_data: HashMap::new(),
                },
                // LayerNorm
                CompiledStep::NativeOp {
                    op: NativeOpKind::LayerNorm {
                        eps: 1e-5,
                        input_shape: vec![1, 16, 256],
                        hidden_dim: 256,
                    },
                    weight_data: HashMap::new(),
                },
                // AddLayerNorm
                CompiledStep::NativeOp {
                    op: NativeOpKind::AddLayerNorm {
                        eps: 1e-5,
                        input_shape: vec![1, 16, 256],
                        hidden_dim: 256,
                    },
                    weight_data: HashMap::new(),
                },
                // AdainSnake
                CompiledStep::NativeOp {
                    op: NativeOpKind::AdainSnake {
                        eps: 1e-5,
                        input_shape: vec![1, 64, 100],
                        channels: 64,
                        residual_gamma: true,
                        external_node_ids: Some(vec![1, 2, 3]),
                    },
                    weight_data: HashMap::new(),
                },
                // AdainLeakyRelu
                CompiledStep::NativeOp {
                    op: NativeOpKind::AdainLeakyRelu {
                        eps: 1e-5,
                        slope: 0.2,
                        input_shape: vec![1, 64, 100],
                        external_node_ids: None,
                    },
                    weight_data: HashMap::new(),
                },
                // AdaLayerNorm
                CompiledStep::NativeOp {
                    op: NativeOpKind::AdaLayerNorm {
                        eps: 1e-5,
                        input_shape: vec![1, 16, 256],
                        hidden_dim: 256,
                    },
                    weight_data: HashMap::new(),
                },
                // FlashAttention (HeadsFirst default)
                CompiledStep::NativeOp {
                    op: NativeOpKind::FlashAttention {
                        scale: 0.125,
                        causal: false,
                        q_shape: vec![1, 4, 8, 64],
                        k_shape: vec![1, 4, 8, 64],
                        output_shape: vec![1, 4, 8, 64],
                        input_layout: AttentionLayout::HeadsFirst,
                    },
                    weight_data: HashMap::new(),
                },
                // MaxPool1d
                CompiledStep::NativeOp {
                    op: NativeOpKind::MaxPool1d {
                        kernel_size: 3,
                        stride: 2,
                        padding: 1,
                        input_shape: vec![1, 32, 100],
                    },
                    weight_data: HashMap::new(),
                },
                // ConstantWeight
                CompiledStep::NativeOp {
                    op: NativeOpKind::ConstantWeight {
                        name: "arange_data".to_string(),
                        shape: vec![64],
                    },
                    weight_data: HashMap::new(),
                },
                // FusedResBlock (with style_proj)
                CompiledStep::NativeOp {
                    op: NativeOpKind::FusedResBlock {
                        phase1: NormActivConv1dParams::new(
                            NormActivation::Snake,
                            1e-5,
                            3,
                            3,
                            vec![1, 64, 100],
                            64,
                            7,
                        ),
                        phase2: NormActivConv1dParams::new(
                            NormActivation::LeakyRelu { slope: 0.2 },
                            1e-5,
                            1,
                            1,
                            vec![1, 64, 100],
                            64,
                            3,
                        ),
                        input_steps: vec![0, 1, 2, 3, 4],
                        residual_scale: 1.0,
                        style_proj: Some(StyleProjectionParams::new(64, 64, 128)),
                        shortcut_step: Some(5),
                        pool_step: None,
                        style_batch_offset: None,
                    },
                    weight_data: HashMap::new(),
                },
                // FusedResBlock (with style_batch_offset)
                CompiledStep::NativeOp {
                    op: NativeOpKind::FusedResBlock {
                        phase1: NormActivConv1dParams::new(
                            NormActivation::Snake,
                            1e-5,
                            1,
                            1,
                            vec![1, 64, 100],
                            64,
                            3,
                        ),
                        phase2: NormActivConv1dParams::new(
                            NormActivation::Snake,
                            1e-5,
                            1,
                            1,
                            vec![1, 64, 100],
                            64,
                            3,
                        ),
                        input_steps: vec![0, 1],
                        residual_scale: 0.7071,
                        style_proj: None,
                        shortcut_step: None,
                        pool_step: Some(2),
                        style_batch_offset: Some(StyleBatchOffset::new(0, 64, 64)),
                    },
                    weight_data: HashMap::new(),
                },
                // BatchedStyleProjection
                CompiledStep::NativeOp {
                    op: NativeOpKind::BatchedStyleProjection {
                        blocks: vec![
                            StyleBatchOffset::new(0, 64, 64),
                            StyleBatchOffset::new(256, 128, 128),
                        ],
                        style_dim: 128,
                        total_out: 768,
                        style_step: 0,
                    },
                    weight_data: HashMap::new(),
                },
                // NormActivConv1d
                CompiledStep::NativeOp {
                    op: NativeOpKind::NormActivConv1d {
                        activation: NormActivation::Snake,
                        eps: 1e-5,
                        conv_dilation: 1,
                        conv_padding: 1,
                        input_shape: vec![1, 64, 128],
                        output_channels: 64,
                        kernel_size: 3,
                        external_node_ids: Some(vec![10, 20, 30]),
                    },
                    weight_data: HashMap::new(),
                },
                // LinearActivation (all GemmActivation variants)
                CompiledStep::NativeOp {
                    op: NativeOpKind::LinearActivation {
                        activation: GemmActivation::Gelu,
                        in_features: 256,
                        out_features: 512,
                        has_bias: true,
                        input_shape: vec![1, 16, 256],
                    },
                    weight_data: HashMap::new(),
                },
                // BatchedLinearProjection
                CompiledStep::NativeOp {
                    op: NativeOpKind::BatchedLinearProjection {
                        in_features: 256,
                        total_out_features: 768,
                        projection_sizes: vec![256, 256, 256],
                        has_bias: true,
                        input_shape: vec![1, 16, 256],
                    },
                    weight_data: HashMap::new(),
                },
                // ProjectionSlice
                CompiledStep::NativeOp {
                    op: NativeOpKind::ProjectionSlice {
                        source_step: 17,
                        dim: 2,
                        start: 256,
                        length: 256,
                        output_shape: vec![1, 16, 256],
                    },
                    weight_data: HashMap::new(),
                },
                // NormLinear (LayerNorm)
                CompiledStep::NativeOp {
                    op: NativeOpKind::NormLinear {
                        norm_kind: FusedNormKind::LayerNorm,
                        eps: 1e-5,
                        input_shape: vec![1, 16, 256],
                        hidden_dim: 256,
                        out_features: 512,
                        has_bias: true,
                    },
                    weight_data: HashMap::new(),
                },
                // NormLinear (RmsNorm)
                CompiledStep::NativeOp {
                    op: NativeOpKind::NormLinear {
                        norm_kind: FusedNormKind::RmsNorm,
                        eps: 1e-5,
                        input_shape: vec![1, 16, 256],
                        hidden_dim: 256,
                        out_features: 512,
                        has_bias: false,
                    },
                    weight_data: HashMap::new(),
                },
                // ChannelsFirstLayerNorm (with LeakyRelu fusion)
                CompiledStep::NativeOp {
                    op: NativeOpKind::ChannelsFirstLayerNorm {
                        eps: 1e-5,
                        input_shape: vec![1, 128, 64],
                        channels: 128,
                        leaky_relu_slope: Some(0.2),
                    },
                    weight_data: HashMap::new(),
                },
                // Int8Gemm
                CompiledStep::NativeOp {
                    op: NativeOpKind::Int8Gemm {
                        in_features: 256,
                        out_features: 512,
                        has_bias: true,
                        input_shape: vec![1, 16, 256],
                    },
                    weight_data: HashMap::new(),
                },
                // Conv1dGemm
                CompiledStep::NativeOp {
                    op: NativeOpKind::Conv1dGemm {
                        input_shape: vec![1, 256, 512],
                        out_channels: 256,
                        kernel_size: 3,
                        stride: 1,
                        padding: 1,
                        dilation: 1,
                        groups: 1,
                        has_bias: true,
                    },
                    weight_data: HashMap::new(),
                },
                // SiluMul
                CompiledStep::NativeOp {
                    op: NativeOpKind::SiluMul {
                        input_shape: vec![1, 16, 512],
                    },
                    weight_data: HashMap::new(),
                },
                // RotaryEmbedding
                CompiledStep::NativeOp {
                    op: NativeOpKind::RotaryEmbedding {
                        head_dim: 64,
                        input_shape: vec![1, 8, 16, 64],
                    },
                    weight_data: HashMap::new(),
                },
                // MoeGating
                CompiledStep::NativeOp {
                    op: NativeOpKind::MoeGating {
                        num_experts: 8,
                        top_k: 2,
                        input_shape: vec![1, 16, 256],
                    },
                    weight_data: HashMap::new(),
                },
                // AddNormLinear
                CompiledStep::NativeOp {
                    op: NativeOpKind::AddNormLinear {
                        eps: 1e-5,
                        input_shape: vec![1, 16, 256],
                        hidden_dim: 256,
                        out_features: 512,
                        has_bias: true,
                    },
                    weight_data: HashMap::new(),
                },
                // BiLstmCat
                CompiledStep::NativeOp {
                    op: NativeOpKind::BiLstmCat {
                        hidden_size: 128,
                        input_shape: vec![50, 1, 256],
                        h_shape: vec![1, 128],
                        fwd_lstm_step: 3,
                        rev_lstm_step: 5,
                    },
                    weight_data: HashMap::new(),
                },
            ],
            input_shapes: vec![vec![1, 256, 512]],
            output_step: 1,
            weight_names: vec![],
        };

        // Serialize
        let json = plan.to_json().expect("serialize all-variants plan");

        // Verify every NativeOpKind variant name appears in JSON
        let expected_variants = [
            "LstmSequence",
            "Cumsum",
            "InstanceNorm",
            "LayerNorm",
            "AddLayerNorm",
            "AdainSnake",
            "AdainLeakyRelu",
            "AdaLayerNorm",
            "FlashAttention",
            "MaxPool1d",
            "ConstantWeight",
            "FusedResBlock",
            "BatchedStyleProjection",
            "NormActivConv1d",
            "LinearActivation",
            "BatchedLinearProjection",
            "ProjectionSlice",
            "NormLinear",
            "ChannelsFirstLayerNorm",
            "Int8Gemm",
            "Conv1dGemm",
            "SiluMul",
            "RotaryEmbedding",
            "MoeGating",
            "AddNormLinear",
            "BiLstmCat",
        ];
        for variant in &expected_variants {
            assert!(
                json.contains(variant),
                "JSON must contain variant {variant}"
            );
        }

        // Deserialize
        let restored = CompiledPlan::from_json(&json).expect("deserialize all-variants plan");

        // Verify structural equality
        assert_eq!(restored.steps.len(), plan.steps.len());
        assert_eq!(restored.input_shapes, plan.input_shapes);
        assert_eq!(restored.output_step, plan.output_step);

        // Spot-check a few restored fields to verify deep round-trip fidelity
        // FusedResBlock with style_proj (step 12)
        match &restored.steps[12] {
            CompiledStep::NativeOp { op, .. } => match op {
                NativeOpKind::FusedResBlock {
                    style_proj,
                    shortcut_step,
                    residual_scale,
                    ..
                } => {
                    let proj = style_proj.as_ref().expect("style_proj must survive");
                    assert_eq!(proj.channels1, 64);
                    assert_eq!(proj.style_dim, 128);
                    assert_eq!(*shortcut_step, Some(5));
                    assert!((residual_scale - 1.0).abs() < 1e-10);
                }
                other => panic!("expected FusedResBlock, got {other:?}"),
            },
            other => panic!("expected NativeOp, got {other:?}"),
        }

        // FusedResBlock with style_batch_offset (step 13)
        match &restored.steps[13] {
            CompiledStep::NativeOp { op, .. } => match op {
                NativeOpKind::FusedResBlock {
                    pool_step,
                    style_batch_offset,
                    ..
                } => {
                    assert_eq!(*pool_step, Some(2));
                    let sbo = style_batch_offset
                        .as_ref()
                        .expect("style_batch_offset must survive");
                    assert_eq!(sbo.offset, 0);
                    assert_eq!(sbo.channels1, 64);
                }
                other => panic!("expected FusedResBlock, got {other:?}"),
            },
            other => panic!("expected NativeOp, got {other:?}"),
        }

        // NormLinear RmsNorm (step 20)
        match &restored.steps[20] {
            CompiledStep::NativeOp { op, .. } => match op {
                NativeOpKind::NormLinear {
                    norm_kind,
                    has_bias,
                    ..
                } => {
                    assert_eq!(*norm_kind, FusedNormKind::RmsNorm);
                    assert!(!has_bias);
                }
                other => panic!("expected NormLinear, got {other:?}"),
            },
            other => panic!("expected NativeOp, got {other:?}"),
        }

        // MoeGating (step 26)
        match &restored.steps[26] {
            CompiledStep::NativeOp { op, .. } => match op {
                NativeOpKind::MoeGating {
                    num_experts, top_k, ..
                } => {
                    assert_eq!(*num_experts, 8);
                    assert_eq!(*top_k, 2);
                }
                other => panic!("expected MoeGating, got {other:?}"),
            },
            other => panic!("expected NativeOp, got {other:?}"),
        }

        // AddNormLinear (step 27)
        match &restored.steps[27] {
            CompiledStep::NativeOp { op, .. } => match op {
                NativeOpKind::AddNormLinear {
                    eps,
                    hidden_dim,
                    out_features,
                    has_bias,
                    ..
                } => {
                    assert!((eps - 1e-5).abs() < 1e-10);
                    assert_eq!(*hidden_dim, 256);
                    assert_eq!(*out_features, 512);
                    assert!(*has_bias);
                }
                other => panic!("expected AddNormLinear, got {other:?}"),
            },
            other => panic!("expected NativeOp, got {other:?}"),
        }

        // BiLstmCat (step 28, last step)
        match restored.steps.last().unwrap() {
            CompiledStep::NativeOp { op, .. } => match op {
                NativeOpKind::BiLstmCat {
                    hidden_size,
                    input_shape,
                    h_shape,
                    fwd_lstm_step,
                    rev_lstm_step,
                } => {
                    assert_eq!(*hidden_size, 128);
                    assert_eq!(input_shape, &[50, 1, 256]);
                    assert_eq!(h_shape, &[1, 128]);
                    assert_eq!(*fwd_lstm_step, 3);
                    assert_eq!(*rev_lstm_step, 5);
                }
                other => panic!("expected BiLstmCat, got {other:?}"),
            },
            other => panic!("expected NativeOp, got {other:?}"),
        }
    }

    /// Verify all GemmActivation variants serialize/deserialize correctly.
    #[test]
    fn test_compiled_plan_json_round_trip_gemm_activations() {
        use crate::trace_compile::{GemmActivation, NativeOpKind};

        let activations = [
            GemmActivation::Relu,
            GemmActivation::Gelu,
            GemmActivation::GeluErf,
            GemmActivation::Sigmoid,
            GemmActivation::Silu,
            GemmActivation::Tanh,
        ];

        for activation in activations {
            let plan = CompiledPlan {
                steps: vec![
                    CompiledStep::InputForward,
                    CompiledStep::NativeOp {
                        op: NativeOpKind::LinearActivation {
                            activation,
                            in_features: 64,
                            out_features: 128,
                            has_bias: true,
                            input_shape: vec![1, 64],
                        },
                        weight_data: HashMap::new(),
                    },
                ],
                input_shapes: vec![vec![1, 64]],
                output_step: 1,
                weight_names: vec![],
            };

            let json = plan
                .to_json()
                .unwrap_or_else(|e| panic!("serialize {activation:?}: {e}"));
            let restored = CompiledPlan::from_json(&json)
                .unwrap_or_else(|e| panic!("deserialize {activation:?}: {e}"));

            match &restored.steps[1] {
                CompiledStep::NativeOp { op, .. } => match op {
                    NativeOpKind::LinearActivation {
                        activation: restored_act,
                        ..
                    } => {
                        assert_eq!(
                            *restored_act, activation,
                            "GemmActivation {activation:?} must round-trip"
                        );
                    }
                    other => panic!("expected LinearActivation, got {other:?}"),
                },
                other => panic!("expected NativeOp, got {other:?}"),
            }
        }
    }
}
