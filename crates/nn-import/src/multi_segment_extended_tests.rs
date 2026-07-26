// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for multi-segment model import and quantization detection.
//!
//! Covers:
//! - Multi-segment model construction, lookup, and ordering
//! - Segment boundary validation (duplicate names, empty input, missing segments)
//! - Weight mapping across segments (shared and isolated weights)
//! - Single-segment backward-compatible import
//! - Quantization detection edge cases (F8, I16/U16 merging, F64 recommendations)
//! - QuantizationReport arithmetic consistency
//! - MultiSegmentError formatting and variant coverage
//!
//! Part of #4548.

use std::collections::HashMap;
use std::path::Path;

use crate::multi_segment::{
    convert_multi_segment, convert_single_segment, MultiSegmentError, MultiSegmentModel,
};
use crate::quantization::{
    detect_quantization_from_bytes, DetectedDtype, DtypeBreakdown, QuantRecommendation,
    TensorQuantInfo,
};

// ---------------------------------------------------------------------------
// Helper: build safetensors bytes for quantization tests
// ---------------------------------------------------------------------------

fn build_st(tensors: &[(&str, safetensors::Dtype, &[usize])]) -> Vec<u8> {
    use safetensors::tensor::TensorView;

    let owned_data: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_name, dtype, shape)| {
            let num_elements: usize = shape.iter().product();
            let bytes_per_elem = match dtype {
                safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
                safetensors::Dtype::F16
                | safetensors::Dtype::BF16
                | safetensors::Dtype::I16
                | safetensors::Dtype::U16 => 2,
                safetensors::Dtype::I8 | safetensors::Dtype::U8 | safetensors::Dtype::BOOL => 1,
                safetensors::Dtype::F64
                | safetensors::Dtype::I64
                | safetensors::Dtype::U64
                | safetensors::Dtype::C64 => 8,
                _ => 4,
            };
            vec![0u8; num_elements * bytes_per_elem]
        })
        .collect();

    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .zip(owned_data.iter())
        .map(|((name, dtype, shape), data)| {
            let view = TensorView::new(*dtype, shape.to_vec(), data).unwrap();
            (*name, view)
        })
        .collect();

    safetensors::serialize(views.iter().map(|(n, v)| (*n, v)), None).unwrap()
}

// ---------------------------------------------------------------------------
// Helper: MLP graph JSON fixtures (same as multi_segment_tests.rs)
// ---------------------------------------------------------------------------

fn mlp_graph_json() -> serde_json::Value {
    serde_json::from_str(include_str!("../test_data/e2e_mlp.json")).unwrap()
}

fn mlp2_graph_json() -> serde_json::Value {
    serde_json::from_str(
        r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "p_fc3_weight"}},
                    {"as_tensor": {"name": "p_fc3_bias"}},
                    {"as_tensor": {"name": "p_fc4_weight"}},
                    {"as_tensor": {"name": "p_fc4_bias"}},
                    {"as_tensor": {"name": "y"}}
                ],
                "outputs": [{"as_tensor": {"name": "linear_3"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.linear.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "y"}}, "kind": 1},
                            {"name": "weight", "arg": {"as_tensor": {"name": "p_fc3_weight"}}, "kind": 1},
                            {"name": "bias", "arg": {"as_tensor": {"name": "p_fc3_bias"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "linear_2"}}],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "linear_2"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "relu_1"}}],
                        "metadata": {}
                    },
                    {
                        "target": "torch.ops.aten.linear.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "relu_1"}}, "kind": 1},
                            {"name": "weight", "arg": {"as_tensor": {"name": "p_fc4_weight"}}, "kind": 1},
                            {"name": "bias", "arg": {"as_tensor": {"name": "p_fc4_bias"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "linear_3"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "y": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_fc3_weight": {"dtype": 7, "sizes": [{"as_int": 6}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_fc3_bias": {"dtype": 7, "sizes": [{"as_int": 6}], "requires_grad": true, "strides": [{"as_int": 1}]},
                    "p_fc4_weight": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 6}], "requires_grad": true, "strides": [{"as_int": 6}, {"as_int": 1}]},
                    "p_fc4_bias": {"dtype": 7, "sizes": [{"as_int": 2}], "requires_grad": true, "strides": [{"as_int": 1}]},
                    "linear_2": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 6}], "requires_grad": false, "strides": [{"as_int": 6}, {"as_int": 1}]},
                    "relu_1": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 6}], "requires_grad": false, "strides": [{"as_int": 6}, {"as_int": 1}]},
                    "linear_3": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 2}], "requires_grad": false, "strides": [{"as_int": 2}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_fc3_weight"}, "parameter_name": "fc3.weight"}},
                    {"parameter": {"arg": {"name": "p_fc3_bias"}, "parameter_name": "fc3.bias"}},
                    {"parameter": {"arg": {"name": "p_fc4_weight"}, "parameter_name": "fc4.weight"}},
                    {"parameter": {"arg": {"name": "p_fc4_bias"}, "parameter_name": "fc4.bias"}},
                    {"user_input": {"arg": {"as_tensor": {"name": "y"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "linear_3"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "opset_version": {"aten": 10},
        "range_constraints": {}
    }"#,
    )
    .unwrap()
}

fn shared_weight_graph_json() -> serde_json::Value {
    serde_json::from_str(
        r#"{
        "graph_module": {
            "graph": {
                "inputs": [
                    {"as_tensor": {"name": "p_fc1_weight"}},
                    {"as_tensor": {"name": "p_fc1_bias"}},
                    {"as_tensor": {"name": "z"}}
                ],
                "outputs": [{"as_tensor": {"name": "head_out"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.linear.default",
                        "inputs": [
                            {"name": "input", "arg": {"as_tensor": {"name": "z"}}, "kind": 1},
                            {"name": "weight", "arg": {"as_tensor": {"name": "p_fc1_weight"}}, "kind": 1},
                            {"name": "bias", "arg": {"as_tensor": {"name": "p_fc1_bias"}}, "kind": 1}
                        ],
                        "outputs": [{"as_tensor": {"name": "head_out"}}],
                        "metadata": {}
                    }
                ],
                "tensor_values": {
                    "z": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 4}], "requires_grad": false, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_fc1_weight": {"dtype": 7, "sizes": [{"as_int": 8}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]},
                    "p_fc1_bias": {"dtype": 7, "sizes": [{"as_int": 8}], "requires_grad": true, "strides": [{"as_int": 1}]},
                    "head_out": {"dtype": 7, "sizes": [{"as_int": 1}, {"as_int": 8}], "requires_grad": false, "strides": [{"as_int": 8}, {"as_int": 1}]}
                },
                "is_single_tensor_return": true
            },
            "signature": {
                "input_specs": [
                    {"parameter": {"arg": {"name": "p_fc1_weight"}, "parameter_name": "fc1.weight"}},
                    {"parameter": {"arg": {"name": "p_fc1_bias"}, "parameter_name": "fc1.bias"}},
                    {"user_input": {"arg": {"as_tensor": {"name": "z"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "head_out"}}}}
                ]
            },
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 15},
        "opset_version": {"aten": 10},
        "range_constraints": {}
    }"#,
    )
    .unwrap()
}

fn test_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_mseg_ext_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_combined_weights(dir: &Path) -> std::path::PathBuf {
    let fc1_w: Vec<u8> = (0..32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc1_b: Vec<u8> = [0.0f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc2_w: Vec<u8> = (0..24)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc2_b: Vec<u8> = [0.0f32; 3].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc3_w: Vec<u8> = (0..24)
        .flat_map(|i| ((i as f32) * 0.02).to_le_bytes())
        .collect();
    let fc3_b: Vec<u8> = [0.0f32; 6].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc4_w: Vec<u8> = (0..12)
        .flat_map(|i| ((i as f32) * 0.03).to_le_bytes())
        .collect();
    let fc4_b: Vec<u8> = [0.0f32; 2].iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "fc1.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8, 4], &fc1_w).unwrap(),
    );
    tensors.insert(
        "fc1.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &fc1_b).unwrap(),
    );
    tensors.insert(
        "fc2.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 8], &fc2_w).unwrap(),
    );
    tensors.insert(
        "fc2.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3], &fc2_b).unwrap(),
    );
    tensors.insert(
        "fc3.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![6, 4], &fc3_w).unwrap(),
    );
    tensors.insert(
        "fc3.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![6], &fc3_b).unwrap(),
    );
    tensors.insert(
        "fc4.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 6], &fc4_w).unwrap(),
    );
    tensors.insert(
        "fc4.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &fc4_b).unwrap(),
    );

    let weights_path = dir.join("combined_weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

fn write_mlp1_weights(dir: &Path) -> std::path::PathBuf {
    let fc1_w: Vec<u8> = (0..32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc1_b: Vec<u8> = [0.0f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc2_w: Vec<u8> = (0..24)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc2_b: Vec<u8> = [0.0f32; 3].iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "fc1.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8, 4], &fc1_w).unwrap(),
    );
    tensors.insert(
        "fc1.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &fc1_b).unwrap(),
    );
    tensors.insert(
        "fc2.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 8], &fc2_w).unwrap(),
    );
    tensors.insert(
        "fc2.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3], &fc2_b).unwrap(),
    );

    let weights_path = dir.join("mlp1_weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

// ===========================================================================
// 1. MultiSegmentModel construction and API
// ===========================================================================

#[test]
fn test_multi_segment_model_new_preserves_order() {
    let dir = test_dir("new_order");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("decoder".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    assert_eq!(model.segment_order, vec!["encoder", "decoder"]);
    assert_eq!(model.num_segments(), 2);
}

#[test]
fn test_multi_segment_model_get_segment_by_name() {
    let dir = test_dir("get_seg");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![
        ("alpha".to_string(), mlp_graph_json()),
        ("beta".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    assert!(model.get_segment("alpha").is_some());
    assert!(model.get_segment("beta").is_some());
    assert!(model.get_segment("gamma").is_none());
}

#[test]
fn test_multi_segment_model_graph_lookup() {
    let dir = test_dir("graph_lookup");
    let wpath = write_mlp1_weights(&dir);
    let graphs = vec![("main".to_string(), mlp_graph_json())];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    let graph = model.graph("main");
    assert!(graph.is_some());
    let g = graph.unwrap();
    assert!(!g.is_empty(), "graph should have nodes");
}

#[test]
fn test_multi_segment_model_graph_returns_none_for_missing() {
    let dir = test_dir("graph_none");
    let wpath = write_mlp1_weights(&dir);
    let graphs = vec![("seg1".to_string(), mlp_graph_json())];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    assert!(model.graph("nonexistent").is_none());
}

#[test]
fn test_multi_segment_model_num_segments_single() {
    let dir = test_dir("num_seg1");
    let wpath = write_mlp1_weights(&dir);
    let graphs = vec![("only".to_string(), mlp_graph_json())];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    assert_eq!(model.num_segments(), 1);
}

#[test]
fn test_multi_segment_model_segments_vec_matches_order() {
    let dir = test_dir("seg_vec");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![
        ("first".to_string(), mlp_graph_json()),
        ("second".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    let names_from_segments: Vec<&str> = model.segments.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names_from_segments, vec!["first", "second"]);
    assert_eq!(model.segment_order, names_from_segments);
}

#[test]
fn test_multi_segment_model_ordered_helpers_follow_declared_order() {
    let dir = test_dir("ordered_helpers");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![
        ("z_last".to_string(), mlp_graph_json()),
        ("a_first".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    let ordered_names: Vec<&str> = model.ordered_segment_names().collect();
    assert_eq!(ordered_names, vec!["z_last", "a_first"]);

    let ordered_segments: Vec<(&str, *const _)> = model
        .ordered_segments()
        .map(|(name, segment)| (name, std::ptr::from_ref(segment)))
        .collect();
    assert_eq!(
        ordered_segments
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>(),
        vec!["z_last", "a_first"]
    );
    assert_eq!(
        ordered_segments[0].1,
        std::ptr::from_ref(model.get_segment("z_last").unwrap())
    );
    assert_eq!(
        ordered_segments[1].1,
        std::ptr::from_ref(model.get_segment("a_first").unwrap())
    );
}

// ===========================================================================
// 2. Segment boundary validation (error cases)
// ===========================================================================

#[test]
fn test_convert_multi_segment_empty_input() {
    let dir = test_dir("empty_input");
    let wpath = write_mlp1_weights(&dir);
    let result = convert_multi_segment(&[], &wpath);
    assert!(result.is_err());
    match result.unwrap_err() {
        MultiSegmentError::EmptyInput => {}
        other => panic!("expected EmptyInput, got {other}"),
    }
}

#[test]
fn test_convert_multi_segment_duplicate_name() {
    let dir = test_dir("dup_name");
    let wpath = write_mlp1_weights(&dir);
    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("encoder".to_string(), mlp_graph_json()),
    ];
    let result = convert_multi_segment(&graphs, &wpath);
    assert!(result.is_err());
    match result.unwrap_err() {
        MultiSegmentError::DuplicateSegment { name } => {
            assert_eq!(name, "encoder");
        }
        other => panic!("expected DuplicateSegment, got {other}"),
    }
}

#[test]
fn test_convert_multi_segment_missing_weights_file() {
    let fake_path = Path::new("/tmp/nonexistent_weights_4548.safetensors");
    let graphs = vec![("seg".to_string(), mlp_graph_json())];
    let result = convert_multi_segment(&graphs, fake_path);
    assert!(result.is_err());
    match result.unwrap_err() {
        MultiSegmentError::Io { path, .. } => {
            assert!(path.contains("nonexistent_weights_4548"));
        }
        other => panic!("expected Io error, got {other}"),
    }
}

#[test]
fn test_convert_multi_segment_invalid_graph_json() {
    let dir = test_dir("invalid_json");
    let wpath = write_mlp1_weights(&dir);
    let bad_json = serde_json::json!({"not": "a valid graph"});
    let graphs = vec![("bad".to_string(), bad_json)];
    let result = convert_multi_segment(&graphs, &wpath);
    assert!(result.is_err());
    match result.unwrap_err() {
        MultiSegmentError::SegmentImport { segment, .. } => {
            assert_eq!(segment, "bad");
        }
        other => panic!("expected SegmentImport, got {other}"),
    }
}

// ===========================================================================
// 3. MultiSegmentError formatting
// ===========================================================================

#[test]
fn test_error_empty_input_display() {
    let err = MultiSegmentError::EmptyInput;
    let msg = format!("{err}");
    assert!(msg.contains("at least one graph segment"));
}

#[test]
fn test_error_duplicate_segment_display() {
    let err = MultiSegmentError::DuplicateSegment {
        name: "encoder".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("duplicate segment name"));
    assert!(msg.contains("encoder"));
}

#[test]
fn test_error_missing_segment_display() {
    let err = MultiSegmentError::MissingSegment {
        name: "decoder".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("missing segment"));
    assert!(msg.contains("decoder"));
}

#[test]
fn test_error_io_display() {
    let err = MultiSegmentError::Io {
        path: "/tmp/test.safetensors".to_string(),
        detail: "file not found".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("/tmp/test.safetensors"));
    assert!(msg.contains("file not found"));
}

#[test]
fn test_error_segment_import_display() {
    let err = MultiSegmentError::SegmentImport {
        segment: "vocoder".to_string(),
        source: Box::new(crate::error::ImportError::UnsupportedOp {
            target: "torch.ops.aten.fake_op".to_string(),
        }),
    };
    let msg = format!("{err}");
    assert!(msg.contains("vocoder"));
    assert!(msg.contains("import error"));
}

#[test]
fn test_error_debug_format() {
    let err = MultiSegmentError::EmptyInput;
    let dbg = format!("{err:?}");
    assert!(dbg.contains("EmptyInput"));
}

// ===========================================================================
// 4. Shared weight detection
// ===========================================================================

#[test]
fn test_shared_weights_detected_across_segments() {
    let dir = test_dir("shared_w");
    let wpath = write_combined_weights(&dir);
    // mlp_graph_json uses fc1.weight/fc1.bias, shared_weight_graph_json also uses fc1.weight/fc1.bias.
    let graphs = vec![
        ("mlp".to_string(), mlp_graph_json()),
        ("head".to_string(), shared_weight_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    assert!(
        !model.shared_weights.is_empty(),
        "shared weights should be detected"
    );
    assert!(model.shared_weights.contains(&"fc1.weight".to_string()));
    assert!(model.shared_weights.contains(&"fc1.bias".to_string()));
}

#[test]
fn test_no_shared_weights_for_isolated_segments() {
    let dir = test_dir("no_shared");
    let wpath = write_combined_weights(&dir);
    // mlp_graph_json uses fc1/fc2, mlp2_graph_json uses fc3/fc4 - no overlap.
    let graphs = vec![
        ("seg_a".to_string(), mlp_graph_json()),
        ("seg_b".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    assert!(
        model.shared_weights.is_empty(),
        "isolated segments should have no shared weights, got {:?}",
        model.shared_weights
    );
}

#[test]
fn test_shared_weights_sorted_alphabetically() {
    let dir = test_dir("shared_sorted");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![
        ("mlp".to_string(), mlp_graph_json()),
        ("head".to_string(), shared_weight_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    let sorted = {
        let mut v = model.shared_weights.clone();
        v.sort();
        v
    };
    assert_eq!(
        model.shared_weights, sorted,
        "shared_weights should be sorted"
    );
}

// ===========================================================================
// 5. Single-segment backward-compatible import
// ===========================================================================

#[test]
fn test_convert_single_segment_creates_main_segment() {
    let dir = test_dir("single_main");
    let wpath = write_mlp1_weights(&dir);
    let model = convert_single_segment(&mlp_graph_json(), &wpath).unwrap();

    assert_eq!(model.num_segments(), 1);
    assert_eq!(model.segment_order, vec!["main"]);
    assert!(model.get_segment("main").is_some());
}

#[test]
fn test_convert_single_segment_graph_accessible() {
    let dir = test_dir("single_graph");
    let wpath = write_mlp1_weights(&dir);
    let model = convert_single_segment(&mlp_graph_json(), &wpath).unwrap();

    let g = model.graph("main").unwrap();
    assert!(!g.is_empty());
}

#[test]
fn test_convert_single_segment_no_shared_weights() {
    let dir = test_dir("single_shared");
    let wpath = write_mlp1_weights(&dir);
    let model = convert_single_segment(&mlp_graph_json(), &wpath).unwrap();

    assert!(
        model.shared_weights.is_empty(),
        "single-segment model should have no shared weights"
    );
}

#[test]
fn test_convert_single_segment_imported_graph_fields() {
    let dir = test_dir("single_fields");
    let wpath = write_mlp1_weights(&dir);
    let model = convert_single_segment(&mlp_graph_json(), &wpath).unwrap();
    let ig = model.get_segment("main").unwrap();

    assert!(ig.num_user_inputs > 0, "should have user inputs");
    assert!(!ig.user_input_names.is_empty(), "should have input names");
    assert!(!ig.output_names.is_empty(), "should have output names");
}

// ===========================================================================
// 6. Multi-segment with 3+ segments
// ===========================================================================

#[test]
fn test_three_segments_all_accessible() {
    let dir = test_dir("three_seg");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("decoder".to_string(), mlp2_graph_json()),
        ("head".to_string(), shared_weight_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    assert_eq!(model.num_segments(), 3);
    assert!(model.get_segment("encoder").is_some());
    assert!(model.get_segment("decoder").is_some());
    assert!(model.get_segment("head").is_some());
    assert_eq!(model.segment_order, vec!["encoder", "decoder", "head"]);
}

#[test]
fn test_three_segments_shared_weights_between_first_and_third() {
    let dir = test_dir("three_shared");
    let wpath = write_combined_weights(&dir);
    // mlp_graph uses fc1/fc2, shared_weight_graph uses fc1 -> shared fc1.weight/fc1.bias.
    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("decoder".to_string(), mlp2_graph_json()),
        ("head".to_string(), shared_weight_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    assert!(model.shared_weights.contains(&"fc1.weight".to_string()));
    assert!(model.shared_weights.contains(&"fc1.bias".to_string()));
    // fc3/fc4 weights used only by decoder, fc2 only by encoder -- not shared.
    assert!(!model.shared_weights.contains(&"fc3.weight".to_string()));
}

// ===========================================================================
// 7. MultiSegmentModel::new direct construction
// ===========================================================================

#[test]
fn test_multi_segment_model_new_direct() {
    let model = MultiSegmentModel::new(
        vec![],
        vec!["a".to_string(), "b".to_string()],
        vec!["shared.w".to_string()],
    );
    assert_eq!(model.num_segments(), 0);
    assert_eq!(model.segment_order.len(), 2);
    assert_eq!(model.shared_weights.len(), 1);
}

#[test]
fn test_multi_segment_model_get_segment_empty() {
    let model = MultiSegmentModel::new(vec![], vec![], vec![]);
    assert!(model.get_segment("anything").is_none());
    assert!(model.graph("anything").is_none());
    assert_eq!(model.num_segments(), 0);
}

// ===========================================================================
// 8. Quantization detection: F8 variants mapping
// ===========================================================================

#[test]
fn test_detected_dtype_f8_e5m2() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::F8_E5M2),
        DetectedDtype::F8
    );
}

#[test]
fn test_detected_dtype_f8_e4m3() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::F8_E4M3),
        DetectedDtype::F8
    );
}

#[test]
fn test_detected_dtype_f8_e8m0() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::F8_E8M0),
        DetectedDtype::F8
    );
}

// ===========================================================================
// 9. Quantization detection: I32/U32 and I64/U64 merging
// ===========================================================================

#[test]
fn test_i32_u32_merge_to_single_bucket() {
    let bytes = build_st(&[
        ("signed32", safetensors::Dtype::I32, &[100]),
        ("unsigned32", safetensors::Dtype::U32, &[200]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::I32);
    assert_eq!(report.dtype_breakdown[0].tensor_count, 2);
    assert_eq!(report.dtype_breakdown[0].total_parameters, 300);
}

#[test]
fn test_i64_u64_merge_to_single_bucket() {
    let bytes = build_st(&[
        ("signed64", safetensors::Dtype::I64, &[50]),
        ("unsigned64", safetensors::Dtype::U64, &[75]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.dtype_breakdown.len(), 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::I64);
    assert_eq!(report.dtype_breakdown[0].tensor_count, 2);
    assert_eq!(report.dtype_breakdown[0].total_parameters, 125);
    assert_eq!(report.dtype_breakdown[0].total_bytes, 125 * 8);
}

// ===========================================================================
// 10. Quantization detection: F64 recommendation generation
// ===========================================================================

#[test]
fn test_f64_even_single_element_gets_recommendation() {
    // F64 with num_elements >= 1 triggers recommendation.
    let bytes = build_st(&[("scale", safetensors::Dtype::F64, &[1])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.recommendations.len(), 1);
    assert_eq!(report.recommendations[0].target_dtype, DetectedDtype::F32);
}

#[test]
fn test_f64_large_tensor_savings_calculation() {
    let bytes = build_st(&[("big_double", safetensors::Dtype::F64, &[4096])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let rec = &report.recommendations[0];
    assert_eq!(rec.target_dtype, DetectedDtype::F32);
    assert_eq!(rec.current_bytes, 4096 * 8);
    assert_eq!(rec.projected_bytes, 4096 * 8 / 2);
    assert_eq!(rec.savings_bytes, 4096 * 8 / 2);
}

// ===========================================================================
// 11. QuantizationReport: is_mixed_precision edge cases
// ===========================================================================

#[test]
fn test_is_mixed_precision_empty_model() {
    let bytes = build_st(&[]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert!(!report.is_mixed_precision());
}

#[test]
fn test_is_mixed_precision_five_dtypes() {
    let bytes = build_st(&[
        ("f32_w", safetensors::Dtype::F32, &[10]),
        ("f16_w", safetensors::Dtype::F16, &[10]),
        ("bf16_w", safetensors::Dtype::BF16, &[10]),
        ("i8_w", safetensors::Dtype::I8, &[10]),
        ("i64_w", safetensors::Dtype::I64, &[10]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 5);
}

// ===========================================================================
// 12. QuantizationReport: dtype_fraction consistency
// ===========================================================================

#[test]
fn test_dtype_fraction_all_dtypes_sum_to_one() {
    let bytes = build_st(&[
        ("f32", safetensors::Dtype::F32, &[1000]),
        ("bf16", safetensors::Dtype::BF16, &[2000]),
        ("i8", safetensors::Dtype::I8, &[500]),
        ("f64", safetensors::Dtype::F64, &[100]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let sum = report.dtype_fraction(DetectedDtype::F32)
        + report.dtype_fraction(DetectedDtype::BF16)
        + report.dtype_fraction(DetectedDtype::I8)
        + report.dtype_fraction(DetectedDtype::F64);
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "dtype fractions should sum to 1.0, got {sum}"
    );
}

#[test]
fn test_dtype_fraction_absent_dtype_returns_zero() {
    let bytes = build_st(&[("w", safetensors::Dtype::F32, &[100])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.dtype_fraction(DetectedDtype::C64), 0.0);
    assert_eq!(report.dtype_fraction(DetectedDtype::SubByte), 0.0);
    assert_eq!(report.dtype_fraction(DetectedDtype::Other), 0.0);
}

// ===========================================================================
// 13. QuantizationReport: total_savings_bytes
// ===========================================================================

#[test]
fn test_total_savings_bytes_matches_individual_recommendations() {
    let bytes = build_st(&[
        ("f32_big", safetensors::Dtype::F32, &[4096]),
        ("f64_big", safetensors::Dtype::F64, &[2048]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let manual: usize = report.recommendations.iter().map(|r| r.savings_bytes).sum();
    assert_eq!(report.total_savings_bytes(), manual);
    assert!(manual > 0);
}

#[test]
fn test_total_savings_bytes_zero_for_compact_model() {
    let bytes = build_st(&[("w", safetensors::Dtype::I8, &[4096])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.total_savings_bytes(), 0);
}

// ===========================================================================
// 14. QuantizationReport: summary formatting
// ===========================================================================

#[test]
fn test_summary_contains_gb_for_large_model() {
    // 256M F32 parameters = ~1 GB.
    let bytes = build_st(&[("huge", safetensors::Dtype::F32, &[16384, 16384])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let summary = report.summary();
    assert!(
        summary.contains("GB") || summary.contains("MB"),
        "large model summary should mention GB or MB: {summary}"
    );
}

#[test]
fn test_summary_contains_recommendations_section() {
    let bytes = build_st(&[
        ("w1", safetensors::Dtype::F32, &[2048]),
        ("w2", safetensors::Dtype::F64, &[512]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let summary = report.summary();

    assert!(summary.contains("Recommendations:"));
    assert!(summary.contains("Quantize"));
    assert!(summary.contains("Total potential savings"));
}

#[test]
fn test_summary_display_trait_equals_summary() {
    let bytes = build_st(&[
        ("a", safetensors::Dtype::F32, &[4096]),
        ("b", safetensors::Dtype::BF16, &[2048]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(format!("{report}"), report.summary());
}

// ===========================================================================
// 15. TensorQuantInfo edge cases
// ===========================================================================

#[test]
fn test_tensor_quant_info_zero_dim_shape() {
    let info = TensorQuantInfo {
        name: "empty_dim".to_string(),
        dtype: DetectedDtype::F32,
        shape: vec![0],
        num_elements: 0,
        size_bytes: 0,
    };
    assert_eq!(info.num_elements, 0);
    assert_eq!(info.size_bytes, 0);
}

#[test]
fn test_tensor_quant_info_high_rank_tensor() {
    let info = TensorQuantInfo {
        name: "high_rank".to_string(),
        dtype: DetectedDtype::BF16,
        shape: vec![2, 3, 4, 5, 6],
        num_elements: 2 * 3 * 4 * 5 * 6,
        size_bytes: 2 * 3 * 4 * 5 * 6 * 2,
    };
    assert_eq!(info.shape.len(), 5);
    assert_eq!(info.num_elements, 720);
    assert_eq!(info.size_bytes, 1440);
}

// ===========================================================================
// 16. DtypeBreakdown direct construction
// ===========================================================================

#[test]
fn test_dtype_breakdown_clone_and_debug() {
    let bd = DtypeBreakdown {
        dtype: DetectedDtype::F16,
        tensor_count: 5,
        total_parameters: 1000,
        total_bytes: 2000,
    };
    let cloned = bd.clone();
    assert_eq!(cloned.dtype, DetectedDtype::F16);
    assert_eq!(cloned.tensor_count, 5);
    let dbg = format!("{bd:?}");
    assert!(dbg.contains("F16"));
}

// ===========================================================================
// 17. QuantRecommendation direct construction
// ===========================================================================

#[test]
fn test_quant_recommendation_clone() {
    let rec = QuantRecommendation {
        target_dtype: DetectedDtype::I8,
        tensor_names: vec!["w1".to_string()],
        current_bytes: 4096,
        projected_bytes: 1024,
        savings_bytes: 3072,
    };
    let cloned = rec;
    assert_eq!(cloned.target_dtype, DetectedDtype::I8);
    assert_eq!(cloned.tensor_names, vec!["w1"]);
    assert_eq!(cloned.savings_bytes, 3072);
}

#[test]
fn test_quant_recommendation_savings_invariant() {
    let rec = QuantRecommendation {
        target_dtype: DetectedDtype::F16,
        tensor_names: vec!["a".to_string(), "b".to_string()],
        current_bytes: 10000,
        projected_bytes: 5000,
        savings_bytes: 5000,
    };
    assert_eq!(rec.savings_bytes, rec.current_bytes - rec.projected_bytes);
}

// ===========================================================================
// 18. DetectedDtype: bytes_per_element boundary checks
// ===========================================================================

#[test]
fn test_detected_dtype_bytes_per_element_sub_byte_none() {
    assert_eq!(DetectedDtype::SubByte.bytes_per_element(), None);
}

#[test]
fn test_detected_dtype_bytes_per_element_other_none() {
    assert_eq!(DetectedDtype::Other.bytes_per_element(), None);
}

#[test]
fn test_detected_dtype_bytes_per_element_bool_one() {
    assert_eq!(DetectedDtype::Bool.bytes_per_element(), Some(1));
}

// ===========================================================================
// 19. DetectedDtype: label and Display consistency
// ===========================================================================

#[test]
fn test_detected_dtype_label_unique_per_variant() {
    let all = [
        DetectedDtype::F32,
        DetectedDtype::F16,
        DetectedDtype::BF16,
        DetectedDtype::F64,
        DetectedDtype::I8,
        DetectedDtype::U8,
        DetectedDtype::F8,
        DetectedDtype::SubByte,
        DetectedDtype::I16,
        DetectedDtype::I32,
        DetectedDtype::I64,
        DetectedDtype::Bool,
        DetectedDtype::C64,
        DetectedDtype::Other,
    ];
    let labels: Vec<&str> = all.iter().map(DetectedDtype::label).collect();
    let mut deduped = labels.clone();
    deduped.sort_unstable();
    deduped.dedup();
    assert_eq!(
        labels.len(),
        deduped.len(),
        "each DetectedDtype variant should have a unique label"
    );
}

// ===========================================================================
// 20. Quantization detection with C64 tensors
// ===========================================================================

#[test]
fn test_detect_c64_tensors() {
    let bytes = build_st(&[("complex_w", safetensors::Dtype::C64, &[64, 64])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::C64);
    assert_eq!(report.total_bytes, 64 * 64 * 8);
    // No quantization recommendations for C64.
    assert!(report.recommendations.is_empty());
}

// ===========================================================================
// 21. Quantization detection with BOOL tensors
// ===========================================================================

#[test]
fn test_detect_bool_no_recommendations() {
    let bytes = build_st(&[("mask", safetensors::Dtype::BOOL, &[1024, 1024])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 1);
    assert_eq!(report.dtype_breakdown[0].dtype, DetectedDtype::Bool);
    assert!(report.recommendations.is_empty());
}

// ===========================================================================
// 22. Mixed F32 and F64 recommendations coexist
// ===========================================================================

#[test]
fn test_mixed_f32_f64_three_recommendations() {
    let bytes = build_st(&[
        ("f32_large", safetensors::Dtype::F32, &[2048]),
        ("f64_large", safetensors::Dtype::F64, &[1024]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    // F32->F16, F32->I8, F64->F32 = 3 recommendations.
    assert_eq!(report.recommendations.len(), 3);
    let targets: Vec<DetectedDtype> = report
        .recommendations
        .iter()
        .map(|r| r.target_dtype)
        .collect();
    assert!(targets.contains(&DetectedDtype::F16));
    assert!(targets.contains(&DetectedDtype::I8));
    assert!(targets.contains(&DetectedDtype::F32));
}

// ===========================================================================
// 23. F32 recommendation threshold boundary
// ===========================================================================

#[test]
fn test_f32_1023_elements_no_recommendations() {
    let bytes = build_st(&[("under", safetensors::Dtype::F32, &[1023])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_f32_1024_elements_gets_recommendations() {
    let bytes = build_st(&[("exact", safetensors::Dtype::F32, &[1024])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.recommendations.len(), 2);
}

#[test]
fn test_f32_1025_elements_gets_recommendations() {
    let bytes = build_st(&[("over", safetensors::Dtype::F32, &[1025])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    assert_eq!(report.recommendations.len(), 2);
}

// ===========================================================================
// 24. Multi-segment: segment graph isolation
// ===========================================================================

#[test]
fn test_segment_graphs_independent() {
    let dir = test_dir("seg_indep");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![
        ("seg_a".to_string(), mlp_graph_json()),
        ("seg_b".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    let ga = model.graph("seg_a").unwrap();
    let gb = model.graph("seg_b").unwrap();
    // Both graphs should have nodes but be independently constructed.
    assert!(!ga.is_empty());
    assert!(!gb.is_empty());
}

// ===========================================================================
// 25. Multi-segment: imported graph metadata
// ===========================================================================

#[test]
fn test_imported_graph_has_user_inputs() {
    let dir = test_dir("user_inp");
    let wpath = write_mlp1_weights(&dir);
    let graphs = vec![("main".to_string(), mlp_graph_json())];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();
    let ig = model.get_segment("main").unwrap();

    assert_eq!(ig.num_user_inputs, 1, "MLP graph has 1 user input ('x')");
}

#[test]
fn test_imported_graph_output_names() {
    let dir = test_dir("out_names");
    let wpath = write_mlp1_weights(&dir);
    let graphs = vec![("main".to_string(), mlp_graph_json())];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();
    let ig = model.get_segment("main").unwrap();

    assert!(!ig.output_names.is_empty(), "should have output names");
}

// ===========================================================================
// 26. Quantization: detect from truncated safetensors
// ===========================================================================

#[test]
fn test_detect_from_single_byte() {
    let result = detect_quantization_from_bytes(&[0x00]);
    assert!(result.is_err());
}

#[test]
fn test_detect_from_seven_bytes() {
    // safetensors header length is 8 bytes, so 7 is truncated.
    let result = detect_quantization_from_bytes(&[0; 7]);
    assert!(result.is_err());
}

// ===========================================================================
// 27. Quantization: report with only small F32 tensors
// ===========================================================================

#[test]
fn test_only_small_f32_tensors_no_recs() {
    let bytes = build_st(&[
        ("bias_a", safetensors::Dtype::F32, &[64]),
        ("bias_b", safetensors::Dtype::F32, &[128]),
        ("bias_c", safetensors::Dtype::F32, &[512]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 3);
    assert!(
        report.recommendations.is_empty(),
        "all tensors below 1024 elements should have no recommendations"
    );
}

// ===========================================================================
// 28. Multi-segment: duplicate detection is name-based
// ===========================================================================

#[test]
fn test_duplicate_detection_same_name_different_json() {
    let dir = test_dir("dup_diff_json");
    let wpath = write_combined_weights(&dir);
    // Same name "seg", different graph content.
    let graphs = vec![
        ("seg".to_string(), mlp_graph_json()),
        ("seg".to_string(), mlp2_graph_json()),
    ];
    let result = convert_multi_segment(&graphs, &wpath);
    assert!(result.is_err());
    match result.unwrap_err() {
        MultiSegmentError::DuplicateSegment { name } => assert_eq!(name, "seg"),
        other => panic!("expected DuplicateSegment, got {other}"),
    }
}

// ===========================================================================
// 29. Multi-segment: segment order preserved for many segments
// ===========================================================================

#[test]
fn test_segment_order_preserved_for_two_same_graph() {
    let dir = test_dir("order_two");
    let wpath = write_mlp1_weights(&dir);
    let graphs = vec![
        ("z_last".to_string(), mlp_graph_json()),
        ("a_first".to_string(), mlp_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    // Order should be insertion order, not alphabetical.
    assert_eq!(model.segment_order, vec!["z_last", "a_first"]);
}

// ===========================================================================
// 30. QuantizationReport: empty model methods
// ===========================================================================

#[test]
fn test_empty_model_dtype_fraction_all_zero() {
    let bytes = build_st(&[]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    for dt in &[
        DetectedDtype::F32,
        DetectedDtype::F16,
        DetectedDtype::BF16,
        DetectedDtype::I8,
        DetectedDtype::F64,
    ] {
        assert_eq!(report.dtype_fraction(*dt), 0.0);
    }
}

// ===========================================================================
// 31. DetectedDtype: Ord/Hash/Eq trait consistency
// ===========================================================================

#[test]
fn test_detected_dtype_ord_total() {
    use std::cmp::Ordering;
    let a = DetectedDtype::F32;
    let b = DetectedDtype::F16;
    // Ord should be consistent: a.cmp(&a) == Equal, a.cmp(&b) != Equal.
    assert_eq!(a.cmp(&a), Ordering::Equal);
    assert_ne!(a.cmp(&b), Ordering::Equal);
}

#[test]
fn test_detected_dtype_hash_all_unique() {
    use std::collections::HashSet;
    let all = [
        DetectedDtype::F32,
        DetectedDtype::F16,
        DetectedDtype::BF16,
        DetectedDtype::F64,
        DetectedDtype::I8,
        DetectedDtype::U8,
        DetectedDtype::F8,
        DetectedDtype::SubByte,
        DetectedDtype::I16,
        DetectedDtype::I32,
        DetectedDtype::I64,
        DetectedDtype::Bool,
        DetectedDtype::C64,
        DetectedDtype::Other,
    ];
    let set: HashSet<DetectedDtype> = all.iter().copied().collect();
    assert_eq!(set.len(), 14, "all 14 variants should hash uniquely");
}

// ===========================================================================
// 32. Quantization: report with many dtypes together
// ===========================================================================

#[test]
fn test_report_seven_dtype_model() {
    let bytes = build_st(&[
        ("f32", safetensors::Dtype::F32, &[100]),
        ("f16", safetensors::Dtype::F16, &[100]),
        ("bf16", safetensors::Dtype::BF16, &[100]),
        ("i8", safetensors::Dtype::I8, &[100]),
        ("u8", safetensors::Dtype::U8, &[100]),
        ("i32", safetensors::Dtype::I32, &[100]),
        ("i64", safetensors::Dtype::I64, &[100]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 7);
    assert!(report.is_mixed_precision());
    assert_eq!(report.dtype_breakdown.len(), 7);

    // Breakdown bytes should sum to total.
    let sum_bytes: usize = report.dtype_breakdown.iter().map(|b| b.total_bytes).sum();
    assert_eq!(sum_bytes, report.total_bytes);
}

// ===========================================================================
// 33. Quantization: QuantRecommendation debug format
// ===========================================================================

#[test]
fn test_quant_recommendation_debug() {
    let rec = QuantRecommendation {
        target_dtype: DetectedDtype::F16,
        tensor_names: vec!["w1".to_string()],
        current_bytes: 1000,
        projected_bytes: 500,
        savings_bytes: 500,
    };
    let dbg = format!("{rec:?}");
    assert!(dbg.contains("F16"));
    assert!(dbg.contains("w1"));
}

// ===========================================================================
// 34. QuantizationReport: clone behavior
// ===========================================================================

#[test]
fn test_quantization_report_clone() {
    let bytes = build_st(&[
        ("w", safetensors::Dtype::F32, &[2048]),
        ("b", safetensors::Dtype::BF16, &[64]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let cloned = report.clone();

    assert_eq!(cloned.total_tensors, report.total_tensors);
    assert_eq!(cloned.total_parameters, report.total_parameters);
    assert_eq!(cloned.total_bytes, report.total_bytes);
    assert_eq!(cloned.dtype_breakdown.len(), report.dtype_breakdown.len());
    assert_eq!(cloned.recommendations.len(), report.recommendations.len());
}

// ===========================================================================
// 35. Multi-segment: convert_single_segment missing weights
// ===========================================================================

#[test]
fn test_convert_single_segment_missing_weights() {
    let fake_path = Path::new("/tmp/nonexistent_single_4548.safetensors");
    let result = convert_single_segment(&mlp_graph_json(), fake_path);
    assert!(result.is_err());
}

// ===========================================================================
// 36. Multi-segment: convert_single_segment bad JSON
// ===========================================================================

#[test]
fn test_convert_single_segment_invalid_json() {
    let dir = test_dir("single_bad");
    let wpath = write_mlp1_weights(&dir);
    let bad = serde_json::json!({"invalid": true});
    let result = convert_single_segment(&bad, &wpath);
    assert!(result.is_err());
}

// ===========================================================================
// 37. Quantization: F32 recommendation tensor names match
// ===========================================================================

#[test]
fn test_f32_recommendation_lists_correct_tensor_names() {
    let bytes = build_st(&[
        ("encoder.weight", safetensors::Dtype::F32, &[2048, 1024]),
        ("decoder.weight", safetensors::Dtype::F32, &[1024, 512]),
        ("tiny_bias", safetensors::Dtype::F32, &[16]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let f16_rec = report
        .recommendations
        .iter()
        .find(|r| r.target_dtype == DetectedDtype::F16)
        .unwrap();
    // Only large tensors (>= 1024 elements) should be recommended.
    assert!(f16_rec.tensor_names.contains(&"encoder.weight".to_string()));
    assert!(f16_rec.tensor_names.contains(&"decoder.weight".to_string()));
    assert!(!f16_rec.tensor_names.contains(&"tiny_bias".to_string()));
}

// ===========================================================================
// 38. DetectedDtype: from_safetensors I16 and U16
// ===========================================================================

#[test]
fn test_detected_dtype_from_safetensors_i16() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::I16),
        DetectedDtype::I16
    );
}

#[test]
fn test_detected_dtype_from_safetensors_u16() {
    assert_eq!(
        DetectedDtype::from_safetensors(safetensors::Dtype::U16),
        DetectedDtype::I16
    );
}

// ===========================================================================
// 39. Multi-segment: segment_order has correct length
// ===========================================================================

#[test]
fn test_segment_order_length_matches_segments() {
    let dir = test_dir("order_len");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![
        ("a".to_string(), mlp_graph_json()),
        ("b".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    assert_eq!(model.segment_order.len(), model.num_segments());
    assert_eq!(model.segments.len(), model.num_segments());
}

// ===========================================================================
// 40. Quantization: detect from safetensors with nested names
// ===========================================================================

#[test]
fn test_detect_nested_tensor_names() {
    let bytes = build_st(&[
        (
            "model.layers.0.attn.qkv.weight",
            safetensors::Dtype::F16,
            &[768, 2304],
        ),
        (
            "model.layers.0.attn.out.weight",
            safetensors::Dtype::F16,
            &[768, 768],
        ),
        (
            "model.layers.0.mlp.fc1.weight",
            safetensors::Dtype::F16,
            &[768, 3072],
        ),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    assert_eq!(report.total_tensors, 3);
    // Tensors should be sorted alphabetically by full name.
    let names: Vec<&str> = report.tensors.iter().map(|t| t.name.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted, "tensors should be sorted by name");
}

// ===========================================================================
// 41. Quantization: report summary format for KB range
// ===========================================================================

#[test]
fn test_summary_shows_kb_for_medium_size() {
    // 256 F32 elements = 1024 bytes = 1 KB
    let bytes = build_st(&[("small", safetensors::Dtype::F32, &[256])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let summary = report.summary();
    assert!(
        summary.contains("KB") || summary.contains("B"),
        "1024-byte model should show KB: {summary}"
    );
}

// ===========================================================================
// 42. Quantization: F32 and BF16 mixed - only F32 gets recs
// ===========================================================================

#[test]
fn test_mixed_f32_bf16_only_f32_recommended() {
    let bytes = build_st(&[
        ("f32_w", safetensors::Dtype::F32, &[4096]),
        ("bf16_w", safetensors::Dtype::BF16, &[8192]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    // F32 should have F16 and I8 recs; BF16 should have none.
    for rec in &report.recommendations {
        for name in &rec.tensor_names {
            assert!(
                name.starts_with("f32"),
                "only F32 tensors should be in recommendations, got {name}"
            );
        }
    }
}

// ===========================================================================
// 43. Multi-segment: segments field directly accessible
// ===========================================================================

#[test]
fn test_segments_field_names_and_graphs() {
    let dir = test_dir("seg_field");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![
        ("alpha".to_string(), mlp_graph_json()),
        ("beta".to_string(), mlp2_graph_json()),
    ];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();

    assert_eq!(model.segments[0].0, "alpha");
    assert_eq!(model.segments[1].0, "beta");
    assert!(!model.segments[0].1.graph.is_empty());
    assert!(!model.segments[1].1.graph.is_empty());
}

// ===========================================================================
// 44. Quantization: detect_quantization_from_bytes error message
// ===========================================================================

#[test]
fn test_detect_error_contains_safetensors_mention() {
    let result = detect_quantization_from_bytes(&[0xFF, 0xFF]);
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("safetensors") || msg.contains("I/O"),
        "error should reference safetensors: {msg}"
    );
}

// ===========================================================================
// 45. Multi-segment: segment imported graph user_input_names
// ===========================================================================

#[test]
fn test_segment_user_input_names_non_empty() {
    let dir = test_dir("inp_names");
    let wpath = write_mlp1_weights(&dir);
    let model = convert_single_segment(&mlp_graph_json(), &wpath).unwrap();
    let ig = model.get_segment("main").unwrap();

    assert!(
        !ig.user_input_names.is_empty(),
        "user_input_names should contain the runtime input name"
    );
}

// ===========================================================================
// 46. Quantization: breakdown sorted by BTreeMap key
// ===========================================================================

#[test]
fn test_breakdown_sorted_by_dtype_ord() {
    let bytes = build_st(&[
        ("i64", safetensors::Dtype::I64, &[10]),
        ("f32", safetensors::Dtype::F32, &[10]),
        ("bool", safetensors::Dtype::BOOL, &[10]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    // BTreeMap produces entries sorted by Ord of DetectedDtype.
    let dtypes: Vec<DetectedDtype> = report.dtype_breakdown.iter().map(|b| b.dtype).collect();
    let mut sorted = dtypes.clone();
    sorted.sort();
    assert_eq!(
        dtypes, sorted,
        "breakdown should be sorted by DetectedDtype Ord"
    );
}

// ===========================================================================
// 47. Multi-segment: empty shared_weights for single segment
// ===========================================================================

#[test]
fn test_single_segment_always_empty_shared_weights() {
    let dir = test_dir("single_no_sw");
    let wpath = write_mlp1_weights(&dir);
    let model = convert_single_segment(&mlp_graph_json(), &wpath).unwrap();
    assert!(model.shared_weights.is_empty());
}

// ===========================================================================
// 48. Quantization: large mixed model has correct total parameters
// ===========================================================================

#[test]
fn test_large_mixed_model_total_parameters() {
    let bytes = build_st(&[
        ("emb", safetensors::Dtype::F32, &[50000, 768]),
        ("attn_w", safetensors::Dtype::F16, &[768, 768]),
        ("mlp_w", safetensors::Dtype::BF16, &[768, 3072]),
        ("ln_w", safetensors::Dtype::F32, &[768]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    let expected = 50000 * 768 + 768 * 768 + 768 * 3072 + 768;
    assert_eq!(report.total_parameters, expected);
    assert_eq!(report.total_tensors, 4);
}

// ===========================================================================
// 49. Multi-segment: imported graph has correct num_user_inputs for mlp2
// ===========================================================================

#[test]
fn test_mlp2_segment_user_inputs() {
    let dir = test_dir("mlp2_inp");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![("decoder".to_string(), mlp2_graph_json())];
    let model = convert_multi_segment(&graphs, &wpath).unwrap();
    let ig = model.get_segment("decoder").unwrap();

    assert_eq!(ig.num_user_inputs, 1, "mlp2 has 1 user input ('y')");
}

// ===========================================================================
// 50. Quantization: QuantizationReport debug format
// ===========================================================================

#[test]
fn test_quantization_report_debug() {
    let bytes = build_st(&[("w", safetensors::Dtype::F32, &[100])]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();
    let dbg = format!("{report:?}");
    assert!(dbg.contains("QuantizationReport"));
    assert!(dbg.contains("total_tensors"));
}

// ===========================================================================
// 51. Multi-segment model debug format
// ===========================================================================

#[test]
fn test_multi_segment_model_debug() {
    let model = MultiSegmentModel::new(vec![], vec!["main".to_string()], vec![]);
    let dbg = format!("{model:?}");
    assert!(dbg.contains("MultiSegmentModel"));
    assert!(dbg.contains("main"));
}

// ===========================================================================
// 52. MultiSegmentError: all variants implement std::error::Error
// ===========================================================================

#[test]
fn test_multi_segment_error_is_error_trait() {
    fn assert_is_error<E: std::error::Error>(_: &E) {}

    assert_is_error(&MultiSegmentError::EmptyInput);
    assert_is_error(&MultiSegmentError::DuplicateSegment {
        name: "x".to_string(),
    });
    assert_is_error(&MultiSegmentError::MissingSegment {
        name: "y".to_string(),
    });
    assert_is_error(&MultiSegmentError::Io {
        path: "p".to_string(),
        detail: "d".to_string(),
    });
}

// ===========================================================================
// 53. Quantization: recommendation current_bytes > projected_bytes invariant
// ===========================================================================

#[test]
fn test_all_recommendations_positive_savings() {
    let bytes = build_st(&[
        ("f32_w", safetensors::Dtype::F32, &[8192]),
        ("f64_w", safetensors::Dtype::F64, &[4096]),
    ]);
    let report = detect_quantization_from_bytes(&bytes).unwrap();

    for rec in &report.recommendations {
        assert!(
            rec.current_bytes > rec.projected_bytes,
            "current_bytes ({}) should exceed projected_bytes ({}) for {:?}",
            rec.current_bytes,
            rec.projected_bytes,
            rec.target_dtype
        );
        assert!(rec.savings_bytes > 0, "savings should be positive");
    }
}

// ===========================================================================
// 54. Metal: compiled multi-segment surface
// ===========================================================================

#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_metal_cache() -> nn_metal::PipelineCache {
    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    nn_metal::PipelineCache::new(nn_metal::MetalContext::new().expect("Metal device required"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_buffer_ptr(buffer: &nn_metal::MetalBuffer) -> usize {
    buffer
        .contents::<u8>()
        .expect("weight buffers should be CPU-readable for test inspection")
        .as_ptr() as usize
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn find_weight_alias<'a>(
    aliases: &'a HashMap<(usize, String), nn_metal::MetalBuffer>,
    weight_name: &str,
    expected: &[f32],
) -> &'a nn_metal::MetalBuffer {
    aliases
        .iter()
        .find_map(|((_, name), buffer)| {
            if name != weight_name {
                return None;
            }
            let contents = buffer
                .contents::<f32>()
                .expect("weight buffer readback should succeed");
            if contents == expected {
                Some(buffer)
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            panic!(
                "missing alias entry for weight '{weight_name}' with {} values",
                expected.len()
            )
        })
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_multi_segment_to_metal_preserves_order() {
    let dir = test_dir("compiled_multi_segment");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("decoder".to_string(), mlp2_graph_json()),
    ];
    let cache = test_metal_cache();

    let compiled = crate::multi_segment::convert_multi_segment_to_metal(&graphs, &wpath, &cache)
        .expect("multi-segment Metal compile should succeed");

    assert_eq!(compiled.num_segments(), 2);
    assert_eq!(compiled.segment_order, vec!["encoder", "decoder"]);
    assert!(
        compiled.shared_weights.is_empty(),
        "disjoint MLP fixtures should not report shared weights"
    );

    let encoder = compiled
        .get_segment("encoder")
        .expect("encoder segment should compile");
    let decoder = compiled
        .get_segment("decoder")
        .expect("decoder segment should compile");
    assert!(
        encoder.num_steps() > 0,
        "encoder should have compiled steps"
    );
    assert!(
        decoder.num_steps() > 0,
        "decoder should have compiled steps"
    );
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_compile_multi_segment_preserves_shared_weight_metadata() {
    let dir = test_dir("compiled_multi_segment_shared");
    let wpath = write_mlp1_weights(&dir);
    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("head".to_string(), shared_weight_graph_json()),
    ];
    let imported = convert_multi_segment(&graphs, &wpath).expect("import should succeed");
    let cache = test_metal_cache();

    let compiled = crate::multi_segment::compile_multi_segment(&imported, &cache)
        .expect("compiling imported multi-segment model should succeed");

    assert_eq!(compiled.num_segments(), 2);
    assert_eq!(compiled.segment_order, imported.segment_order);
    assert_eq!(compiled.shared_weights, imported.shared_weights);
    assert!(
        compiled.shared_weights.contains(&"fc1.weight".to_string()),
        "shared weight metadata should preserve fc1.weight"
    );
    assert!(
        compiled.shared_weights.contains(&"fc1.bias".to_string()),
        "shared weight metadata should preserve fc1.bias"
    );
    assert!(
        compiled
            .get_segment("head")
            .expect("head segment should compile")
            .num_dispatches()
            > 0,
        "compiled head segment should expose compiled dispatches"
    );
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_compile_multi_segment_reuses_shared_weight_buffers_across_segments() {
    let dir = test_dir("compiled_multi_segment_aliases");
    let wpath = write_mlp1_weights(&dir);
    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("head".to_string(), shared_weight_graph_json()),
    ];
    let imported = convert_multi_segment(&graphs, &wpath).expect("import should succeed");
    let cache = test_metal_cache();

    let compiled = crate::multi_segment::compile_multi_segment(&imported, &cache)
        .expect("compiling imported multi-segment model should succeed");

    let encoder_aliases = compiled
        .get_segment("encoder")
        .expect("encoder segment should compile")
        .weight_buffer_aliases();
    let head_aliases = compiled
        .get_segment("head")
        .expect("head segment should compile")
        .weight_buffer_aliases();

    let fc1_weight: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
    let fc1_bias = vec![0.0f32; 8];
    let fc2_weight: Vec<f32> = (0..24).map(|i| (i as f32) * 0.01).collect();

    let encoder_fc1_weight = find_weight_alias(&encoder_aliases, "weight", &fc1_weight);
    let encoder_fc1_bias = find_weight_alias(&encoder_aliases, "bias", &fc1_bias);
    let encoder_fc2_weight = find_weight_alias(&encoder_aliases, "weight", &fc2_weight);
    let head_fc1_weight = find_weight_alias(&head_aliases, "weight", &fc1_weight);
    let head_fc1_bias = find_weight_alias(&head_aliases, "bias", &fc1_bias);

    assert_eq!(
        metal_buffer_ptr(encoder_fc1_weight),
        metal_buffer_ptr(head_fc1_weight),
        "shared fc1.weight should reuse the same Metal allocation across segments"
    );
    assert_eq!(
        metal_buffer_ptr(encoder_fc1_bias),
        metal_buffer_ptr(head_fc1_bias),
        "shared fc1.bias should reuse the same Metal allocation across segments"
    );
    assert_ne!(
        metal_buffer_ptr(encoder_fc2_weight),
        metal_buffer_ptr(head_fc1_weight),
        "non-shared encoder weights must not alias the head segment's shared buffer"
    );
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_compile_multi_segment_shared_aliasing_preserves_segment_outputs() {
    use nn_core::Device;

    let dir = test_dir("compiled_multi_segment_exec_parity");
    let wpath = write_mlp1_weights(&dir);
    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("head".to_string(), shared_weight_graph_json()),
    ];
    let imported = convert_multi_segment(&graphs, &wpath).expect("import should succeed");
    let cache = test_metal_cache();

    let compiled_multi = crate::multi_segment::compile_multi_segment(&imported, &cache)
        .expect("compiling imported multi-segment model should succeed");

    let standalone_encoder = nn_metal::compiled_model::CompiledModel::builder(
        &imported
            .get_segment("encoder")
            .expect("encoder segment should import")
            .graph,
        &cache,
    )
    .build()
    .expect("standalone encoder compile should succeed");
    let standalone_head = nn_metal::compiled_model::CompiledModel::builder(
        &imported
            .get_segment("head")
            .expect("head segment should import")
            .graph,
        &cache,
    )
    .build()
    .expect("standalone head compile should succeed");

    let encoder_input_cpu =
        nn_core::DynTensor::from_vec(vec![0.25, -0.5, 0.75, 1.25], &[1, 4], &Device::Cpu)
            .expect("encoder input");
    let encoder_input_gpu = encoder_input_cpu
        .to_device(&Device::metal())
        .expect("encoder input -> metal");
    let head_input_cpu =
        nn_core::DynTensor::from_vec(vec![-1.0, 0.5, 1.5, -0.25], &[1, 4], &Device::Cpu)
            .expect("head input");
    let head_input_gpu = head_input_cpu
        .to_device(&Device::metal())
        .expect("head input -> metal");

    let multi_encoder_output = compiled_multi
        .get_segment("encoder")
        .expect("encoder segment should compile")
        .execute_dyn(&cache, &[&encoder_input_gpu])
        .expect("multi-segment encoder execution");
    let standalone_encoder_output = standalone_encoder
        .execute_dyn(&cache, &[&encoder_input_gpu])
        .expect("standalone encoder execution");
    let multi_head_output = compiled_multi
        .get_segment("head")
        .expect("head segment should compile")
        .execute_dyn(&cache, &[&head_input_gpu])
        .expect("multi-segment head execution");
    let standalone_head_output = standalone_head
        .execute_dyn(&cache, &[&head_input_gpu])
        .expect("standalone head execution");

    let multi_encoder_vals = multi_encoder_output
        .to_device(&Device::Cpu)
        .expect("multi encoder -> cpu")
        .to_flat_vec::<f32>()
        .expect("multi encoder values");
    let standalone_encoder_vals = standalone_encoder_output
        .to_device(&Device::Cpu)
        .expect("standalone encoder -> cpu")
        .to_flat_vec::<f32>()
        .expect("standalone encoder values");
    let multi_head_vals = multi_head_output
        .to_device(&Device::Cpu)
        .expect("multi head -> cpu")
        .to_flat_vec::<f32>()
        .expect("multi head values");
    let standalone_head_vals = standalone_head_output
        .to_device(&Device::Cpu)
        .expect("standalone head -> cpu")
        .to_flat_vec::<f32>()
        .expect("standalone head values");

    assert_eq!(
        multi_encoder_vals.len(),
        standalone_encoder_vals.len(),
        "encoder output length should be stable with shared aliasing"
    );
    assert_eq!(
        multi_head_vals.len(),
        standalone_head_vals.len(),
        "head output length should be stable with shared aliasing"
    );

    let encoder_max_diff = multi_encoder_vals
        .iter()
        .zip(standalone_encoder_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let head_max_diff = multi_head_vals
        .iter()
        .zip(standalone_head_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        encoder_max_diff < 1e-6,
        "encoder output drifted after shared alias seeding: max diff {encoder_max_diff:.6e}"
    );
    assert!(
        head_max_diff < 1e-6,
        "head output drifted after shared alias seeding: max diff {head_max_diff:.6e}"
    );
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_compile_multi_segment_ordered_helpers_support_explicit_execution() {
    use nn_core::Device;

    let dir = test_dir("compiled_multi_segment_ordered_helpers");
    let wpath = write_combined_weights(&dir);
    let graphs = vec![
        ("encoder".to_string(), mlp_graph_json()),
        ("decoder".to_string(), mlp2_graph_json()),
    ];
    let imported = convert_multi_segment(&graphs, &wpath).expect("import should succeed");
    let cache = test_metal_cache();

    let compiled = crate::multi_segment::compile_multi_segment(&imported, &cache)
        .expect("compiling imported multi-segment model should succeed");

    let ordered_names: Vec<&str> = compiled.ordered_segment_names().collect();
    assert_eq!(ordered_names, vec!["encoder", "decoder"]);

    let encoder_input_cpu =
        nn_core::DynTensor::from_vec(vec![0.25, -0.5, 0.75, 1.25], &[1, 4], &Device::Cpu)
            .expect("encoder input");
    let encoder_input_gpu = encoder_input_cpu
        .to_device(&Device::metal())
        .expect("encoder input -> metal");
    let decoder_input_cpu =
        nn_core::DynTensor::from_vec(vec![-1.0, 0.5, 1.5, -0.25], &[1, 4], &Device::Cpu)
            .expect("decoder input");
    let decoder_input_gpu = decoder_input_cpu
        .to_device(&Device::metal())
        .expect("decoder input -> metal");

    let expected_encoder = compiled
        .get_segment("encoder")
        .expect("encoder segment should compile")
        .execute_dyn(&cache, &[&encoder_input_gpu])
        .expect("direct encoder execution")
        .to_device(&Device::Cpu)
        .expect("direct encoder -> cpu")
        .to_flat_vec::<f32>()
        .expect("direct encoder values");
    let expected_decoder = compiled
        .get_segment("decoder")
        .expect("decoder segment should compile")
        .execute_dyn(&cache, &[&decoder_input_gpu])
        .expect("direct decoder execution")
        .to_device(&Device::Cpu)
        .expect("direct decoder -> cpu")
        .to_flat_vec::<f32>()
        .expect("direct decoder values");

    let ordered_outputs: Vec<(String, Vec<f32>)> = compiled
        .ordered_segments()
        .map(|(name, segment)| {
            let input = match name {
                "encoder" => &encoder_input_gpu,
                "decoder" => &decoder_input_gpu,
                other => panic!("unexpected compiled segment '{other}'"),
            };
            let output = segment
                .execute_dyn(&cache, &[input])
                .expect("ordered segment execution")
                .to_device(&Device::Cpu)
                .expect("ordered output -> cpu")
                .to_flat_vec::<f32>()
                .expect("ordered output values");
            (name.to_string(), output)
        })
        .collect();

    assert_eq!(
        ordered_outputs
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["encoder", "decoder"]
    );
    assert_eq!(
        ordered_outputs[0].1, expected_encoder,
        "ordered encoder execution should match direct segment access"
    );
    assert_eq!(
        ordered_outputs[1].1, expected_decoder,
        "ordered decoder execution should match direct segment access"
    );
}
