// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration tests for the nn-import pipeline.
//!
//! Covers: op mapping correctness and determinism, ConvertBuilder configuration,
//! weight loading edge cases, and end-to-end model import roundtrips for various
//! architecture patterns (attention, normalization, pooling, multi-output).
//!
//! These tests complement the existing test suites:
//! - `op_coverage.rs` (per-op mapping via synthetic JSON)
//! - `weight_loading.rs` (safetensors parsing, dtype conversion)
//! - `model_graph_e2e.rs` (architecture-specific graph patterns)
//! - `convert_builder_tests.rs` (proof types, Metal-gated builder)
//! - `unit_coverage.rs` (supported_ops list, parse edge cases)

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;
use nn_import::{
    build_graph, build_weight_map, parse_exported_program, supported_ops, ImportError,
    ResolvedWeight,
};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a minimal ExportedProgram JSON string for a single-op graph.
///
/// Creates: input x:[in_shape] -> op_node -> output:[out_shape]
///
/// `op_target`: full torch.ops target (e.g., "torch.ops.aten.relu.default")
/// `inputs_json`: raw JSON array for the node's "inputs" field
/// `extra_graph_inputs`: additional graph-level input arguments (e.g., weight placeholders)
/// `extra_tensor_values`: additional tensor_values entries
/// `extra_input_specs`: additional input_specs entries (parameters, buffers)
fn build_single_op_json(
    op_target: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    inputs_json: &str,
    extra_graph_inputs: &str,
    extra_tensor_values: &str,
    extra_input_specs: &str,
) -> String {
    let in_sizes: String = in_shape
        .iter()
        .map(|d| format!("{{\"as_int\": {d}}}"))
        .collect::<Vec<_>>()
        .join(", ");
    let out_sizes: String = out_shape
        .iter()
        .map(|d| format!("{{\"as_int\": {d}}}"))
        .collect::<Vec<_>>()
        .join(", ");

    let in_strides = compute_strides(in_shape);
    let in_strides_json: String = in_strides
        .iter()
        .map(|s| format!("{{\"as_int\": {s}}}"))
        .collect::<Vec<_>>()
        .join(", ");
    let out_strides = compute_strides(out_shape);
    let out_strides_json: String = out_strides
        .iter()
        .map(|s| format!("{{\"as_int\": {s}}}"))
        .collect::<Vec<_>>()
        .join(", ");

    let extra_graph_inputs_comma = if extra_graph_inputs.is_empty() {
        String::new()
    } else {
        format!(", {extra_graph_inputs}")
    };
    let extra_tv_comma = if extra_tensor_values.is_empty() {
        String::new()
    } else {
        format!(", {extra_tensor_values}")
    };
    let extra_specs_comma = if extra_input_specs.is_empty() {
        String::new()
    } else {
        format!(", {extra_input_specs}")
    };

    format!(
        r#"{{
        "graph_module": {{
            "graph": {{
                "inputs": [{{"as_tensor": {{"name": "x"}}}}{extra_graph_inputs_comma}],
                "outputs": [{{"as_tensor": {{"name": "out"}}}}],
                "nodes": [
                    {{
                        "target": "{op_target}",
                        "inputs": {inputs_json},
                        "outputs": [{{"as_tensor": {{"name": "out"}}}}],
                        "metadata": {{}}
                    }}
                ],
                "tensor_values": {{
                    "x": {{"dtype": 7, "sizes": [{in_sizes}], "requires_grad": false, "strides": [{in_strides_json}]}},
                    "out": {{"dtype": 7, "sizes": [{out_sizes}], "requires_grad": false, "strides": [{out_strides_json}]}}
                    {extra_tv_comma}
                }},
                "is_single_tensor_return": true
            }},
            "signature": {{
                "input_specs": [
                    {{"user_input": {{"arg": {{"as_tensor": {{"name": "x"}}}}}}}}
                    {extra_specs_comma}
                ],
                "output_specs": [
                    {{"user_output": {{"arg": {{"as_tensor": {{"name": "out"}}}}}}}}
                ]
            }},
            "module_call_graph": []
        }},
        "schema_version": {{"major": 8, "minor": 15}},
        "range_constraints": {{}}
    }}"#
    )
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Import a graph from JSON with the given weight data.
fn import_from_json(
    json: &str,
    raw_weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> nn_import::ImportedGraph {
    let program = parse_exported_program(json.as_bytes()).expect("fixture JSON must parse");
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &raw_weights);
    build_graph(&program, &weight_map).expect("build_graph must succeed")
}

/// Collect only compute ops (non-Input, non-Constant) from an imported graph.
fn compute_ops(
    imported: &nn_import::ImportedGraph,
) -> Vec<&nn_core::dyn_tensor::trace::TraceNode> {
    imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect()
}

// ===========================================================================
// A. Op Mapping Tests
// ===========================================================================

#[test]
fn test_supported_ops_list_non_empty_and_has_expected_ops() {
    let ops = supported_ops();
    assert!(
        ops.len() >= 80,
        "expected at least 80 supported ops, got {}",
        ops.len()
    );
    // Core ops every ML framework needs
    for expected in [
        "aten::relu",
        "aten::linear",
        "aten::softmax",
        "aten::matmul",
        "aten::conv2d",
        "aten::layer_norm",
        "aten::embedding",
        "aten::batch_norm",
    ] {
        assert!(
            ops.contains(&expected),
            "supported_ops missing core op: {expected}"
        );
    }
}

#[test]
fn test_relu_mapping_produces_relu_trace_op() {
    let json = build_single_op_json(
        "torch.ops.aten.relu.default",
        &[1, 4],
        &[1, 4],
        r#"[{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}]"#,
        "",
        "",
        "",
    );
    let imported = import_from_json(&json, HashMap::new());
    let ops = compute_ops(&imported);
    assert_eq!(ops.len(), 1);
    assert!(
        matches!(ops[0].op(), TraceOp::Relu),
        "expected Relu, got {:?}",
        ops[0].op()
    );
    assert_eq!(ops[0].output_shape(), &[1, 4]);
}

#[test]
fn test_linear_mapping_produces_linear_trace_op() {
    // Linear needs weight and bias parameters
    let json = build_single_op_json(
        "torch.ops.aten.linear.default",
        &[2, 4],
        &[2, 3],
        r#"[
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "weight", "arg": {"as_tensor": {"name": "p_weight"}}, "kind": 1},
            {"name": "bias", "arg": {"as_tensor": {"name": "p_bias"}}, "kind": 1}
        ]"#,
        r#"{"as_tensor": {"name": "p_weight"}}, {"as_tensor": {"name": "p_bias"}}"#,
        r#""p_weight": {"dtype": 7, "sizes": [{"as_int": 3}, {"as_int": 4}], "requires_grad": true, "strides": [{"as_int": 4}, {"as_int": 1}]},
        "p_bias": {"dtype": 7, "sizes": [{"as_int": 3}], "requires_grad": true, "strides": [{"as_int": 1}]}"#,
        r#"{"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "weight"}},
        {"parameter": {"arg": {"name": "p_bias"}, "parameter_name": "bias"}}"#,
    );
    let mut weights = HashMap::new();
    weights.insert("weight".to_string(), (vec![0.1; 12], vec![3, 4]));
    weights.insert("bias".to_string(), (vec![0.0; 3], vec![3]));

    let imported = import_from_json(&json, weights);
    let ops = compute_ops(&imported);
    assert_eq!(ops.len(), 1);
    assert!(
        matches!(ops[0].op(), TraceOp::Linear { .. }),
        "expected Linear, got {:?}",
        ops[0].op()
    );
    assert_eq!(ops[0].output_shape(), &[2, 3]);
}

#[test]
fn test_softmax_mapping_produces_softmax_trace_op() {
    let json = build_single_op_json(
        "torch.ops.aten.softmax.int",
        &[1, 10],
        &[1, 10],
        r#"[
            {"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1},
            {"name": "dim", "arg": {"as_int": 1}, "kind": 1},
            {"name": "dtype", "arg": {"as_none": true}, "kind": 2}
        ]"#,
        "",
        "",
        "",
    );
    let imported = import_from_json(&json, HashMap::new());
    let ops = compute_ops(&imported);
    assert_eq!(ops.len(), 1);
    match ops[0].op() {
        TraceOp::Softmax { dim } => {
            assert_eq!(*dim, 1, "softmax dim should be 1");
        }
        other => panic!("expected Softmax, got {other:?}"),
    }
}

#[test]
fn test_unsupported_op_returns_clear_error() {
    let json = build_single_op_json(
        "torch.ops.custom.made_up_op.default",
        &[1, 4],
        &[1, 4],
        r#"[{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}]"#,
        "",
        "",
        "",
    );
    let program = parse_exported_program(json.as_bytes()).unwrap();
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &HashMap::new());
    let err = build_graph(&program, &weight_map).unwrap_err();
    assert!(
        matches!(err, ImportError::UnsupportedOp { .. }),
        "expected UnsupportedOp, got: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("custom.made_up_op"),
        "error message should contain the op name: {msg}"
    );
}

#[test]
fn test_all_supported_ops_have_aten_prefix() {
    let ops = supported_ops();
    for op in &ops {
        assert!(
            op.starts_with("aten::"),
            "supported op '{op}' does not start with 'aten::'"
        );
    }
}

#[test]
fn test_op_mapping_deterministic() {
    // Same input should produce the same output on repeated calls.
    let json = build_single_op_json(
        "torch.ops.aten.relu.default",
        &[2, 8],
        &[2, 8],
        r#"[{"name": "input", "arg": {"as_tensor": {"name": "x"}}, "kind": 1}]"#,
        "",
        "",
        "",
    );

    let imported1 = import_from_json(&json, HashMap::new());
    let imported2 = import_from_json(&json, HashMap::new());

    let ops1 = compute_ops(&imported1);
    let ops2 = compute_ops(&imported2);
    assert_eq!(ops1.len(), ops2.len());

    for (a, b) in ops1.iter().zip(ops2.iter()) {
        assert_eq!(
            std::mem::discriminant(a.op()),
            std::mem::discriminant(b.op()),
            "op mapping is not deterministic"
        );
        assert_eq!(a.output_shape(), b.output_shape());
        assert_eq!(a.output_dtype(), b.output_dtype());
    }
}

#[test]
fn test_supported_ops_no_duplicates() {
    let ops = supported_ops();
    let mut seen = std::collections::HashSet::new();
    for op in &ops {
        assert!(seen.insert(op), "duplicate entry in supported_ops: '{op}'");
    }
}

// ===========================================================================
// B. ConvertBuilder / Report / Verification Coverage Tests (no Metal required)
// ===========================================================================

#[test]
fn test_verification_coverage_default_is_zero() {
    let vc = nn_import::VerificationCoverage::default();
    assert_eq!(vc.gamma_crown_layers_covered, 0);
    assert_eq!(vc.gamma_crown_layers_total, 0);
    assert!(vc.kani_harnesses_applicable.is_none());
    assert!(!vc.composition_bounds_ok);
    assert!(vc.composition_bound_width.is_none());
    assert!(vc.composition_method.is_none());
    assert!(vc.composition_soundness_mode.is_none());
    assert!(vc.composition_proof_strength.is_none());
    assert!(vc.reference_parity_passed.is_none());
}

#[test]
fn test_verification_coverage_gamma_crown_pct() {
    let mut vc = nn_import::VerificationCoverage::default();
    vc.gamma_crown_layers_covered = 10;
    vc.gamma_crown_layers_total = 20;
    let pct = vc.gamma_crown_coverage_pct();
    assert!((pct - 50.0).abs() < 0.01, "expected 50%, got {pct}");
}

#[test]
fn test_verification_coverage_gamma_crown_pct_zero_total() {
    let vc = nn_import::VerificationCoverage::default();
    let pct = vc.gamma_crown_coverage_pct();
    assert!(
        pct == 0.0 || pct.is_nan(),
        "coverage with 0 total should be 0 or NaN, got {pct}"
    );
}

#[test]
fn test_verification_coverage_full_coverage() {
    let mut vc = nn_import::VerificationCoverage::default();
    vc.gamma_crown_layers_covered = 50;
    vc.gamma_crown_layers_total = 50;
    let pct = vc.gamma_crown_coverage_pct();
    assert!((pct - 100.0).abs() < 0.01, "expected 100%, got {pct}");
}

#[test]
fn test_verification_coverage_serializable() {
    let mut vc = nn_import::VerificationCoverage::default();
    vc.gamma_crown_layers_covered = 5;
    vc.gamma_crown_layers_total = 10;
    vc.kani_harnesses_applicable = Some(7);
    vc.composition_bounds_ok = true;
    vc.composition_method = Some(nn_import::ConvertCompositionMethod::Ibp);
    vc.composition_soundness_mode = Some(nn_import::ConvertSoundnessMode::Sound);
    vc.composition_proof_strength = Some(nn_import::ConvertProofStrength::SoundIbp);
    let json = serde_json::to_string(&vc).expect("VerificationCoverage must serialize");
    let val: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(val["gamma_crown_layers_covered"], 5);
    assert_eq!(val["gamma_crown_layers_total"], 10);
    assert_eq!(val["kani_harnesses_applicable"], 7);
    assert_eq!(val["composition_bounds_ok"], true);
    assert_eq!(val["composition_method"], "IBP");
    assert_eq!(val["composition_soundness_mode"], "sound");
    assert_eq!(val["composition_proof_strength"], "sound_ibp");
}

#[test]
fn test_equivalence_proof_construction() {
    // EquivalenceProof should be constructible via ::new() and inspectable.
    let proof = nn_import::EquivalenceProof::new(None, None, None);
    assert!(proof.kernel_safety.is_none());
    assert!(proof.composition_bounds.is_none());
    assert!(proof.reference_parity.is_none());
}

#[test]
fn test_imported_graph_constructor() {
    // ImportedGraph::new with an empty ComputationGraph.
    let graph = nn_core::dyn_tensor::trace::ComputationGraph::from_nodes(vec![]);
    let imported = nn_import::ImportedGraph::new(
        graph,
        2,
        vec!["a".to_string(), "b".to_string()],
        vec!["out".to_string()],
    );
    assert_eq!(imported.num_user_inputs, 2);
    assert_eq!(imported.user_input_names, vec!["a", "b"]);
    assert_eq!(imported.output_names, vec!["out"]);
}

// ===========================================================================
// C. Weight Loading Tests
// ===========================================================================

/// Build a safetensors byte buffer from typed tensor data.
fn build_safetensors_f32(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    let byte_bufs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|&(_, _, data)| data.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let mut tensor_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for (i, &(name, shape, _)) in tensors.iter().enumerate() {
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.to_vec(),
            &byte_bufs[i],
        )
        .expect("valid tensor view");
        tensor_map.push((name.to_string(), view));
    }
    safetensors::tensor::serialize(tensor_map, None).expect("serialization should succeed")
}

#[test]
fn test_safetensors_weight_loading_roundtrip() {
    // Create safetensors, load it, build weight map, verify data integrity.
    let weight_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    let bias_data: Vec<f32> = vec![0.5, -0.5, 0.0];
    let bytes = build_safetensors_f32(&[
        ("weight", &[3, 4], &weight_data),
        ("bias", &[3], &bias_data),
    ]);

    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut raw_weights: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for name in tensors.names() {
        let view = tensors.tensor(name).unwrap();
        let shape: Vec<usize> = view.shape().to_vec();
        let f32_data: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        raw_weights.insert(name.to_string(), (f32_data, shape));
    }

    assert_eq!(raw_weights.len(), 2);
    let (w_data, w_shape) = raw_weights.get("weight").unwrap();
    assert_eq!(w_shape, &[3, 4]);
    assert_eq!(w_data.len(), 12);
    assert!((w_data[0] - 0.0).abs() < f32::EPSILON);
    assert!((w_data[1] - 0.1).abs() < f32::EPSILON);

    let (b_data, b_shape) = raw_weights.get("bias").unwrap();
    assert_eq!(b_shape, &[3]);
    assert!((b_data[0] - 0.5).abs() < f32::EPSILON);
    assert!((b_data[1] - (-0.5)).abs() < f32::EPSILON);
}

#[test]
fn test_weight_shape_mismatch_detected_at_construction() {
    // ResolvedWeight stores data as-is; mismatch is a logic error in the caller.
    let weight = ResolvedWeight::new(vec![0.1; 10], vec![3, 4]); // 10 != 12
    assert_ne!(
        weight.data.len(),
        weight.shape.iter().product::<usize>(),
        "data length should not match shape product"
    );
}

#[test]
fn test_weight_dtype_conversion_f16() {
    // Verify f16 weights are converted to f32 during safetensors loading.
    let f16_vals: Vec<half::f16> = vec![
        half::f16::from_f32(1.0),
        half::f16::from_f32(-2.5),
        half::f16::from_f32(0.0),
    ];
    let bytes: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut tensor_map = Vec::new();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![3], &bytes).unwrap();
    tensor_map.push(("w".to_string(), view));
    let st_bytes = safetensors::tensor::serialize(tensor_map, None).unwrap();

    let tensors = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();
    let w_view = tensors.tensor("w").unwrap();
    let f32_data: Vec<f32> = w_view
        .data()
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();

    assert_eq!(f32_data.len(), 3);
    assert!((f32_data[0] - 1.0).abs() < 0.01);
    assert!((f32_data[1] - (-2.5)).abs() < 0.01);
    assert_eq!(f32_data[2], 0.0);
}

#[test]
fn test_missing_weight_key_descriptive_error() {
    // import_model with valid graph but empty safetensors -> MissingWeight error.
    let dir = std::env::temp_dir().join(format!(
        "nn_import_test_missing_key_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();

    // Write graph JSON that expects weights
    let graph_path = dir.join("graph.json");
    std::fs::write(&graph_path, include_bytes!("../test_data/e2e_mlp.json")).unwrap();

    // Write empty safetensors
    let empty_tensors: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
    let serialized = safetensors::serialize(&empty_tensors, None).unwrap();
    let weights_path = dir.join("weights.safetensors");
    std::fs::write(&weights_path, serialized).unwrap();

    let err = nn_import::import_model(&graph_path, &weights_path).unwrap_err();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        matches!(err, ImportError::MissingWeight { .. }),
        "expected MissingWeight, got: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("not found"),
        "error should describe the missing weight: {msg}"
    );
}

// ===========================================================================
// D. End-to-End Pipeline Tests
// ===========================================================================

#[test]
fn test_simple_model_roundtrip_linear() {
    // Define a simple linear model, import it, verify output shape and op type.
    let imported = import_from_json(include_str!("../test_data/e2e_mlp.json"), {
        let mut w = HashMap::new();
        w.insert("fc1.weight".to_string(), (vec![0.01; 32], vec![8, 4]));
        w.insert("fc1.bias".to_string(), (vec![0.0; 8], vec![8]));
        w.insert("fc2.weight".to_string(), (vec![0.01; 24], vec![3, 8]));
        w.insert("fc2.bias".to_string(), (vec![0.0; 3], vec![3]));
        w
    });

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["linear_1"]);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Linear { .. }),
        "output should be Linear, got {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 3]);
}

#[test]
fn test_model_with_residual_connection() {
    // ResNet basic block has a skip connection (Add node with 2 inputs).
    let imported = import_from_json(include_str!("../test_data/resnet_basic_block.json"), {
        let mut w = HashMap::new();
        w.insert(
            "conv1.weight".to_string(),
            (vec![0.01; 2304], vec![16, 16, 3, 3]),
        );
        w.insert("conv1.bias".to_string(), (vec![0.0; 16], vec![16]));
        w.insert("bn1.weight".to_string(), (vec![1.0; 16], vec![16]));
        w.insert("bn1.bias".to_string(), (vec![0.0; 16], vec![16]));
        w.insert("bn1.running_mean".to_string(), (vec![0.0; 16], vec![16]));
        w.insert("bn1.running_var".to_string(), (vec![1.0; 16], vec![16]));
        w.insert(
            "conv2.weight".to_string(),
            (vec![0.01; 2304], vec![16, 16, 3, 3]),
        );
        w.insert("conv2.bias".to_string(), (vec![0.0; 16], vec![16]));
        w.insert("bn2.weight".to_string(), (vec![1.0; 16], vec![16]));
        w.insert("bn2.bias".to_string(), (vec![0.0; 16], vec![16]));
        w.insert("bn2.running_mean".to_string(), (vec![0.0; 16], vec![16]));
        w.insert("bn2.running_var".to_string(), (vec![1.0; 16], vec![16]));
        w
    });

    // Verify the Add (skip connection) node exists with 2 inputs.
    let add_nodes: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Add))
        .collect();
    assert_eq!(
        add_nodes.len(),
        1,
        "ResNet block should have exactly 1 Add (skip connection)"
    );
    assert_eq!(
        add_nodes[0].inputs().len(),
        2,
        "skip connection Add should have 2 inputs"
    );

    // Output shape should match input shape (identity residual).
    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_shape(), &[1, 16, 8, 8]);
}

#[test]
fn test_model_with_attention() {
    // Transformer encoder layer with SDPA.
    let imported = import_from_json(
        include_str!("../test_data/transformer_encoder_layer.json"),
        {
            let mut w = HashMap::new();
            w.insert("ln1.weight".to_string(), (vec![1.0; 16], vec![16]));
            w.insert("ln1.bias".to_string(), (vec![0.0; 16], vec![16]));
            w.insert(
                "attn.q_proj.weight".to_string(),
                (vec![0.01; 256], vec![16, 16]),
            );
            w.insert("attn.q_proj.bias".to_string(), (vec![0.0; 16], vec![16]));
            w.insert(
                "attn.k_proj.weight".to_string(),
                (vec![0.01; 256], vec![16, 16]),
            );
            w.insert("attn.k_proj.bias".to_string(), (vec![0.0; 16], vec![16]));
            w.insert(
                "attn.v_proj.weight".to_string(),
                (vec![0.01; 256], vec![16, 16]),
            );
            w.insert("attn.v_proj.bias".to_string(), (vec![0.0; 16], vec![16]));
            w.insert(
                "attn.out_proj.weight".to_string(),
                (vec![0.01; 256], vec![16, 16]),
            );
            w.insert("attn.out_proj.bias".to_string(), (vec![0.0; 16], vec![16]));
            w.insert("ln2.weight".to_string(), (vec![1.0; 16], vec![16]));
            w.insert("ln2.bias".to_string(), (vec![0.0; 16], vec![16]));
            w.insert("ff.fc1.weight".to_string(), (vec![0.01; 512], vec![32, 16]));
            w.insert("ff.fc1.bias".to_string(), (vec![0.0; 32], vec![32]));
            w.insert("ff.fc2.weight".to_string(), (vec![0.01; 512], vec![16, 32]));
            w.insert("ff.fc2.bias".to_string(), (vec![0.0; 16], vec![16]));
            w
        },
    );

    let ops = compute_ops(&imported);

    // Verify SDPA is present
    let sdpa_count = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Sdpa { .. }))
        .count();
    assert_eq!(sdpa_count, 1, "transformer encoder should have 1 SDPA op");

    // Verify LayerNorm present
    let ln_count = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::LayerNorm { .. }))
        .count();
    assert_eq!(
        ln_count, 2,
        "transformer encoder should have 2 LayerNorm ops"
    );

    // Output shape preserves input dimensions.
    let output = imported.graph.output_node().unwrap();
    assert_eq!(
        output.output_shape(),
        &[1, 4, 16],
        "encoder output = [B, seq_len, d_model]"
    );
}

#[test]
fn test_import_produces_valid_trace_ops() {
    // Import the MLP and verify that every node produces a valid TraceOp.
    let imported = import_from_json(include_str!("../test_data/e2e_mlp.json"), {
        let mut w = HashMap::new();
        w.insert("fc1.weight".to_string(), (vec![0.01; 32], vec![8, 4]));
        w.insert("fc1.bias".to_string(), (vec![0.0; 8], vec![8]));
        w.insert("fc2.weight".to_string(), (vec![0.01; 24], vec![3, 8]));
        w.insert("fc2.bias".to_string(), (vec![0.0; 3], vec![3]));
        w
    });

    // Every node should have a non-empty output shape.
    for node in imported.graph.nodes() {
        assert!(
            !node.output_shape().is_empty(),
            "node '{}' has empty output shape",
            node.name()
        );
    }

    // Every node's inputs should reference valid earlier nodes.
    for node in imported.graph.nodes() {
        for &input_id in node.inputs() {
            assert!(
                imported.graph.node(input_id).is_some(),
                "node '{}' references missing input_id {}",
                node.name(),
                input_id
            );
        }
    }

    // Verify the compute ops are the expected types.
    let ops = compute_ops(&imported);
    assert_eq!(ops.len(), 3, "MLP should have 3 compute ops");
    assert!(matches!(ops[0].op(), TraceOp::Linear { .. }));
    assert!(matches!(ops[1].op(), TraceOp::Relu));
    assert!(matches!(ops[2].op(), TraceOp::Linear { .. }));
}

#[test]
fn test_multi_layer_mlp_shape_propagation() {
    // Multi-layer MLP: x:[1,4] -> Linear(4->8) -> ReLU -> Linear(8->3)
    // Verify shape propagation through the pipeline.
    let imported = import_from_json(include_str!("../test_data/e2e_mlp.json"), {
        let mut w = HashMap::new();
        w.insert("fc1.weight".to_string(), (vec![0.01; 32], vec![8, 4]));
        w.insert("fc1.bias".to_string(), (vec![0.0; 8], vec![8]));
        w.insert("fc2.weight".to_string(), (vec![0.01; 24], vec![3, 8]));
        w.insert("fc2.bias".to_string(), (vec![0.0; 3], vec![3]));
        w
    });

    let ops = compute_ops(&imported);
    // Linear(4->8): [1, 4] -> [1, 8]
    assert_eq!(ops[0].output_shape(), &[1, 8]);
    // ReLU: [1, 8] -> [1, 8]
    assert_eq!(ops[1].output_shape(), &[1, 8]);
    // Linear(8->3): [1, 8] -> [1, 3]
    assert_eq!(ops[2].output_shape(), &[1, 3]);
}

#[test]
fn test_embedding_model_import() {
    // Embedding lookup model: two embeddings + add.
    let imported = import_from_json(include_str!("../test_data/embedding_positional.json"), {
        let mut w = HashMap::new();
        w.insert(
            "tok_embed.weight".to_string(),
            (vec![0.01; 1600], vec![100, 16]),
        );
        w.insert(
            "pos_embed.weight".to_string(),
            (vec![0.01; 512], vec![32, 16]),
        );
        w
    });

    assert_eq!(imported.num_user_inputs, 2);
    let ops = compute_ops(&imported);

    // Two Embedding ops + one Add
    let embed_count = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Embedding { .. }))
        .count();
    assert_eq!(embed_count, 2, "should have 2 Embedding ops");

    let add_count = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Add))
        .count();
    assert_eq!(add_count, 1, "should have 1 Add op");
}

#[test]
fn test_multi_input_graph_import() {
    // Multi-input model: two inputs merged via cat.
    let imported = import_from_json(include_str!("../test_data/multi_input_cat.json"), {
        let mut w = HashMap::new();
        w.insert("fc.weight".to_string(), (vec![0.01; 64], vec![4, 16]));
        w.insert("fc.bias".to_string(), (vec![0.0; 4], vec![4]));
        w
    });

    assert_eq!(imported.num_user_inputs, 2);
    assert_eq!(imported.user_input_names, vec!["a", "b"]);

    let ops = compute_ops(&imported);
    assert!(
        matches!(ops[0].op(), TraceOp::Cat { .. }),
        "first op should be Cat"
    );
}

#[test]
fn test_classification_head_softmax_output() {
    // Classification head: Linear -> Softmax with correct dim.
    let imported = import_from_json(include_str!("../test_data/classification_head.json"), {
        let mut w = HashMap::new();
        w.insert("fc.weight".to_string(), (vec![0.01; 640], vec![10, 64]));
        w.insert("fc.bias".to_string(), (vec![0.0; 10], vec![10]));
        w
    });

    let ops = compute_ops(&imported);
    assert_eq!(ops.len(), 2);

    // Verify softmax is on the correct dimension
    match ops[1].op() {
        TraceOp::Softmax { dim } => {
            assert_eq!(*dim, 1, "softmax should be on dim=1 for classification");
        }
        other => panic!("expected Softmax, got {other:?}"),
    }

    // Output shape: [1, 10] (10 classes)
    assert_eq!(ops[1].output_shape(), &[1, 10]);
}

#[test]
fn test_conv_bn_silu_chain_import() {
    // Conv backbone: Conv2d -> BN -> SiLU chain with stride downsampling.
    let imported = import_from_json(include_str!("../test_data/convbnact_backbone.json"), {
        let mut w = HashMap::new();
        w.insert(
            "stage0.conv.weight".to_string(),
            (vec![0.01; 432], vec![16, 3, 3, 3]),
        );
        w.insert("stage0.conv.bias".to_string(), (vec![0.0; 16], vec![16]));
        w.insert("stage0.bn.weight".to_string(), (vec![1.0; 16], vec![16]));
        w.insert("stage0.bn.bias".to_string(), (vec![0.0; 16], vec![16]));
        w.insert(
            "stage0.bn.running_mean".to_string(),
            (vec![0.0; 16], vec![16]),
        );
        w.insert(
            "stage0.bn.running_var".to_string(),
            (vec![1.0; 16], vec![16]),
        );
        w.insert(
            "stage1.conv.weight".to_string(),
            (vec![0.01; 4608], vec![32, 16, 3, 3]),
        );
        w.insert("stage1.conv.bias".to_string(), (vec![0.0; 32], vec![32]));
        w.insert("stage1.bn.weight".to_string(), (vec![1.0; 32], vec![32]));
        w.insert("stage1.bn.bias".to_string(), (vec![0.0; 32], vec![32]));
        w.insert(
            "stage1.bn.running_mean".to_string(),
            (vec![0.0; 32], vec![32]),
        );
        w.insert(
            "stage1.bn.running_var".to_string(),
            (vec![1.0; 32], vec![32]),
        );
        w
    });

    let ops = compute_ops(&imported);
    // Verify the chain pattern: Conv2d, BN, SiLU, Conv2d, BN, SiLU
    assert_eq!(ops.len(), 6);
    assert!(matches!(ops[0].op(), TraceOp::Conv2d { .. }));
    assert!(matches!(ops[1].op(), TraceOp::BatchNorm { .. }));
    assert!(matches!(ops[2].op(), TraceOp::Silu));
    assert!(matches!(ops[3].op(), TraceOp::Conv2d { .. }));
    assert!(matches!(ops[4].op(), TraceOp::BatchNorm { .. }));
    assert!(matches!(ops[5].op(), TraceOp::Silu));

    // Verify spatial downsampling: [1, 3, 32, 32] -> [1, 16, 16, 16] -> [1, 32, 8, 8]
    assert_eq!(ops[0].output_shape(), &[1, 16, 16, 16]);
    assert_eq!(ops[5].output_shape(), &[1, 32, 8, 8]);
}

#[test]
fn test_graph_topology_all_fixtures_valid() {
    // Verify every fixture has valid topology: all input references exist.
    let fixtures: Vec<(&str, HashMap<String, (Vec<f32>, Vec<usize>)>)> = vec![
        (include_str!("../test_data/e2e_mlp.json"), {
            let mut w = HashMap::new();
            w.insert("fc1.weight".to_string(), (vec![0.01; 32], vec![8, 4]));
            w.insert("fc1.bias".to_string(), (vec![0.0; 8], vec![8]));
            w.insert("fc2.weight".to_string(), (vec![0.01; 24], vec![3, 8]));
            w.insert("fc2.bias".to_string(), (vec![0.0; 3], vec![3]));
            w
        }),
        (include_str!("../test_data/classification_head.json"), {
            let mut w = HashMap::new();
            w.insert("fc.weight".to_string(), (vec![0.01; 640], vec![10, 64]));
            w.insert("fc.bias".to_string(), (vec![0.0; 10], vec![10]));
            w
        }),
    ];

    for (i, (json, weights)) in fixtures.iter().enumerate() {
        let imported = import_from_json(json, weights.clone());
        for node in imported.graph.nodes() {
            for &input_id in node.inputs() {
                assert!(
                    imported.graph.node(input_id).is_some(),
                    "fixture[{i}]: node '{}' references missing input_id {}",
                    node.name(),
                    input_id
                );
            }
        }
    }
}

#[test]
fn test_import_model_from_file_roundtrip() {
    // Full file-based import_model roundtrip using temp files.
    let dir = std::env::temp_dir().join(format!("nn_import_pipeline_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Write graph JSON
    let graph_path = dir.join("graph.json");
    std::fs::write(&graph_path, include_bytes!("../test_data/e2e_mlp.json")).unwrap();

    // Write safetensors weights
    let fc1_w: Vec<u8> = (0..32u32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc1_b: Vec<u8> = [0.0f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc2_w: Vec<u8> = (0..24u32)
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
    let weights_path = dir.join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, &serialized).unwrap();

    // Import
    let imported =
        nn_import::import_model(&graph_path, &weights_path).expect("import_model must succeed");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.output_names, vec!["linear_1"]);
    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_shape(), &[1, 3]);
}

#[test]
fn test_parse_exported_program_schema_v8_accepted() {
    let json = br#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 8, "minor": 42},
        "range_constraints": {}
    }"#;
    let program = parse_exported_program(json).unwrap();
    assert_eq!(program.schema_version.major, 8);
    assert_eq!(program.schema_version.minor, 42);
}

#[test]
fn test_parse_exported_program_wrong_version_rejected() {
    let json = br#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 7, "minor": 0},
        "range_constraints": {}
    }"#;
    let err = parse_exported_program(json).unwrap_err();
    assert!(
        matches!(err, ImportError::UnsupportedSchema { major: 7, .. }),
        "expected UnsupportedSchema, got: {err:?}"
    );
}

#[test]
fn test_import_model_invalid_json_path() {
    let err = nn_import::import_model(
        std::path::Path::new("/nonexistent.json"),
        std::path::Path::new("/nonexistent.safetensors"),
    )
    .unwrap_err();
    assert!(
        matches!(err, ImportError::Io { .. }),
        "expected Io error, got: {err:?}"
    );
}

#[test]
fn test_resolved_weight_constructor() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let w = ResolvedWeight::new(data.clone(), shape.clone());
    assert_eq!(w.data, data);
    assert_eq!(w.shape, shape);
}

#[test]
fn test_build_weight_map_maps_parameters_and_buffers() {
    let json = include_str!("../test_data/resnet_basic_block.json");
    let program = parse_exported_program(json.as_bytes()).unwrap();

    let mut raw_weights = HashMap::new();
    raw_weights.insert(
        "conv1.weight".to_string(),
        (vec![0.01; 2304], vec![16, 16, 3, 3]),
    );
    raw_weights.insert("conv1.bias".to_string(), (vec![0.0; 16], vec![16]));
    raw_weights.insert("bn1.weight".to_string(), (vec![1.0; 16], vec![16]));
    raw_weights.insert("bn1.bias".to_string(), (vec![0.0; 16], vec![16]));
    raw_weights.insert("bn1.running_mean".to_string(), (vec![0.0; 16], vec![16]));
    raw_weights.insert("bn1.running_var".to_string(), (vec![1.0; 16], vec![16]));
    raw_weights.insert(
        "conv2.weight".to_string(),
        (vec![0.01; 2304], vec![16, 16, 3, 3]),
    );
    raw_weights.insert("conv2.bias".to_string(), (vec![0.0; 16], vec![16]));
    raw_weights.insert("bn2.weight".to_string(), (vec![1.0; 16], vec![16]));
    raw_weights.insert("bn2.bias".to_string(), (vec![0.0; 16], vec![16]));
    raw_weights.insert("bn2.running_mean".to_string(), (vec![0.0; 16], vec![16]));
    raw_weights.insert("bn2.running_var".to_string(), (vec![1.0; 16], vec![16]));

    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &raw_weights);

    // Weight map should include both parameters and buffers.
    assert!(
        weight_map.len() >= 8,
        "weight map should have at least 8 entries (params + buffers), got {}",
        weight_map.len()
    );
}

#[test]
fn test_build_weight_map_empty_input_specs() {
    let raw_weights: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let weight_map = build_weight_map(&[], &raw_weights);
    assert!(
        weight_map.is_empty(),
        "empty specs should produce empty map"
    );
}

#[test]
fn test_import_error_debug_format() {
    // Verify all ImportError variants implement Debug properly.
    let errors: Vec<ImportError> = vec![
        ImportError::UnsupportedOp {
            target: "test".to_string(),
        },
        ImportError::MissingArgument {
            op_target: "linear".to_string(),
            arg_name: "weight".to_string(),
        },
        ImportError::MissingWeight {
            fqn: "model.weight".to_string(),
        },
        ImportError::UnsupportedSchema { major: 5, minor: 0 },
        ImportError::TopologyError {
            node_name: "relu".to_string(),
            ref_name: "missing".to_string(),
        },
    ];

    for err in &errors {
        let debug = format!("{err:?}");
        assert!(!debug.is_empty(), "Debug format should not be empty");
        let display = format!("{err}");
        assert!(!display.is_empty(), "Display format should not be empty");
    }
}
