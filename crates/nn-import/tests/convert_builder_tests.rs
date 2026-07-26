// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the `nn::convert()` builder pipeline.
//!
//! Tests the full parse -> graph -> weight map -> convert flow with synthetic
//! data. Exercises `ConvertBuilder` with different `OptLevel`/`VerifyLevel`
//! combinations, verifies `EquivalenceProof` field population, and covers
//! error handling for invalid inputs.
//!
//! Tests that do NOT require Metal are ungated and run on all platforms.
//! Metal-gated tests use `#[cfg(all(feature = "metal", target_os = "macos"))]`.

use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Helpers: synthetic graph JSON and safetensors weight generation
// ---------------------------------------------------------------------------

/// Minimal MLP graph JSON: x:[1,4] -> fc1(4->8) -> relu -> fc2(8->3) -> output:[1,3]
fn mlp_graph_json() -> &'static [u8] {
    include_bytes!("../test_data/e2e_mlp.json")
}

/// Write the MLP graph JSON to a temp directory, returning the path.
fn write_graph_json(dir: &Path) -> std::path::PathBuf {
    let graph_path = dir.join("graph.json");
    std::fs::write(&graph_path, mlp_graph_json()).unwrap();
    graph_path
}

/// Write synthetic MLP safetensors weights (fc1: 4->8, fc2: 8->3).
fn write_mlp_weights(dir: &Path) -> std::path::PathBuf {
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
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

/// Create a unique temp directory for each test to avoid conflicts.
fn make_temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_cb_test_{}_{}", name, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ===========================================================================
// 1. EquivalenceProof construction and field population (no Metal)
// ===========================================================================

#[test]
fn test_equivalence_proof_all_none() {
    let proof = nn_import::EquivalenceProof::new(None, None, None);
    assert!(proof.kernel_safety.is_none());
    assert!(proof.composition_bounds.is_none());
    assert!(proof.reference_parity.is_none());
}

#[test]
fn test_equivalence_proof_with_kani_safety() {
    let kani = nn_import::KaniSafetyReport::new(100, 98, 2);
    let proof = nn_import::EquivalenceProof::new(Some(kani), None, None);
    let ks = proof.kernel_safety.as_ref().unwrap();
    assert_eq!(ks.harness_count, 100);
    assert_eq!(ks.passed, 98);
    assert_eq!(ks.failed, 2);
    assert!(proof.composition_bounds.is_none());
    assert!(proof.reference_parity.is_none());
}

#[test]
fn test_equivalence_proof_with_composition_bounds() {
    let bounds = nn_import::CompositionBoundsReport::new(true, Some(1.5));
    let proof = nn_import::EquivalenceProof::new(None, Some(bounds), None);
    assert!(proof.kernel_safety.is_none());
    let cb = proof.composition_bounds.as_ref().unwrap();
    assert!(cb.propagation_ok);
    assert!((cb.output_width.unwrap() - 1.5).abs() < f32::EPSILON);
    assert!(proof.reference_parity.is_none());
}

#[test]
fn test_equivalence_proof_kani_and_bounds_populated() {
    let kani = nn_import::KaniSafetyReport::new(50, 50, 0);
    let bounds = nn_import::CompositionBoundsReport::new(true, Some(0.8));
    let proof = nn_import::EquivalenceProof::new(Some(kani), Some(bounds), None);
    assert!(proof.kernel_safety.is_some());
    assert!(proof.composition_bounds.is_some());
    // reference_parity requires reftest feature, tested separately
}

#[test]
fn test_kani_safety_report_zero_failures() {
    let report = nn_import::KaniSafetyReport::new(754, 754, 0);
    assert_eq!(report.harness_count, 754);
    assert_eq!(report.passed, 754);
    assert_eq!(report.failed, 0);
}

#[test]
fn test_composition_bounds_report_failed_propagation() {
    let report = nn_import::CompositionBoundsReport::new(false, None);
    assert!(!report.propagation_ok);
    assert!(report.output_width.is_none());
}

#[test]
fn test_composition_bounds_report_infinite_width_stored_as_none() {
    // When output width is infinite, it should still be stored (the check is
    // done at the call site). Here we verify the report stores what it is given.
    let report = nn_import::CompositionBoundsReport::new(true, None);
    assert!(report.propagation_ok);
    assert!(report.output_width.is_none());
}

// ===========================================================================
// 2. Parse -> graph build -> weight map flow (no Metal)
// ===========================================================================

#[test]
fn test_parse_graph_build_mlp_synthetic() {
    use nn_core::dyn_tensor::trace::TraceOp;

    let dir = make_temp_dir("parse_graph_build");
    let graph_path = write_graph_json(&dir);
    let weights_path = write_mlp_weights(&dir);

    let imported = nn_import::import_model(&graph_path, &weights_path)
        .expect("import_model must succeed for MLP fixture");

    let _ = std::fs::remove_dir_all(&dir);

    // Verify imported graph structure.
    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["linear_1"]);

    // 1 input + 4 param placeholders + 3 compute ops = 8 nodes
    assert_eq!(imported.graph.len(), 8);

    // Output node should be a Linear op producing [1, 3].
    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Linear { .. }),
        "expected Linear output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 3]);
}

#[test]
fn test_parse_graph_build_op_sequence() {
    use nn_core::dyn_tensor::trace::TraceOp;

    let dir = make_temp_dir("op_sequence");
    let graph_path = write_graph_json(&dir);
    let weights_path = write_mlp_weights(&dir);

    let imported = nn_import::import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    // Extract only compute ops (not Input or Constant placeholders).
    let compute_ops: Vec<_> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect();

    assert_eq!(compute_ops.len(), 3, "expected 3 compute ops");
    assert!(matches!(compute_ops[0].op(), TraceOp::Linear { .. }));
    assert!(matches!(compute_ops[1].op(), TraceOp::Relu));
    assert!(matches!(compute_ops[2].op(), TraceOp::Linear { .. }));

    // Shape propagation: [1,4] -> Linear(4->8) -> [1,8] -> ReLU -> [1,8] -> Linear(8->3) -> [1,3]
    assert_eq!(compute_ops[0].output_shape(), &[1, 8]);
    assert_eq!(compute_ops[1].output_shape(), &[1, 8]);
    assert_eq!(compute_ops[2].output_shape(), &[1, 3]);
}

#[test]
fn test_build_weight_map_maps_params_correctly() {
    let program = nn_import::parse_exported_program(mlp_graph_json()).unwrap();

    // Construct in-memory weight data matching parameter FQNs.
    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("fc1.weight".to_string(), (vec![0.1; 32], vec![8, 4]));
    weight_data.insert("fc1.bias".to_string(), (vec![0.0; 8], vec![8]));
    weight_data.insert("fc2.weight".to_string(), (vec![0.1; 24], vec![3, 8]));
    weight_data.insert("fc2.bias".to_string(), (vec![0.0; 3], vec![3]));

    let weight_map =
        nn_import::build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    // Weight map should have 4 entries (graph placeholder names, not FQNs).
    assert_eq!(weight_map.len(), 4);
    assert!(weight_map.contains_key("p_fc1_weight"));
    assert!(weight_map.contains_key("p_fc1_bias"));
    assert!(weight_map.contains_key("p_fc2_weight"));
    assert!(weight_map.contains_key("p_fc2_bias"));

    // Verify shapes are correctly propagated.
    assert_eq!(weight_map["p_fc1_weight"].shape, vec![8, 4]);
    assert_eq!(weight_map["p_fc1_weight"].data.len(), 32);
    assert_eq!(weight_map["p_fc2_bias"].shape, vec![3]);
    assert_eq!(weight_map["p_fc2_bias"].data.len(), 3);
}

#[test]
fn test_build_weight_map_missing_fqn_silently_skipped() {
    let program = nn_import::parse_exported_program(mlp_graph_json()).unwrap();

    // Provide only one of the four expected weights. Missing FQNs are silently
    // skipped by build_weight_map (the error surfaces later in build_graph).
    let mut weight_data: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    weight_data.insert("fc1.weight".to_string(), (vec![0.1; 32], vec![8, 4]));

    let weight_map =
        nn_import::build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    // Only 1 of the 4 parameters matched.
    assert_eq!(weight_map.len(), 1);
    assert!(weight_map.contains_key("p_fc1_weight"));
}

#[test]
fn test_parse_exported_program_roundtrip() {
    let program = nn_import::parse_exported_program(mlp_graph_json()).unwrap();
    assert_eq!(program.schema_version.major, 8);
    assert_eq!(program.schema_version.minor, 15);
    assert_eq!(program.graph_module.graph.nodes.len(), 3);
    assert_eq!(program.graph_module.signature.input_specs.len(), 5);
    assert_eq!(program.graph_module.signature.output_specs.len(), 1);
}

#[test]
fn test_parse_exported_program_tensor_values() {
    let program = nn_import::parse_exported_program(mlp_graph_json()).unwrap();
    let tv = &program.graph_module.graph.tensor_values;

    // The MLP fixture declares 8 tensors in tensor_values.
    assert!(tv.contains_key("x"), "should have input tensor 'x'");
    assert!(
        tv.contains_key("linear_0"),
        "should have intermediate 'linear_0'"
    );
    assert!(tv.contains_key("relu"), "should have 'relu'");
    assert!(tv.contains_key("linear_1"), "should have output 'linear_1'");

    // Verify concrete shapes.
    let x_meta = tv.get("x").unwrap();
    let x_shape = x_meta.concrete_shape().unwrap();
    assert_eq!(x_shape, vec![1, 4]);

    let out_meta = tv.get("linear_1").unwrap();
    let out_shape = out_meta.concrete_shape().unwrap();
    assert_eq!(out_shape, vec![1, 3]);
}

#[test]
fn test_imported_graph_topology_valid() {
    let dir = make_temp_dir("topology");
    let graph_path = write_graph_json(&dir);
    let weights_path = write_mlp_weights(&dir);

    let imported = nn_import::import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    // Every node's inputs must reference existing earlier nodes.
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
}

// ===========================================================================
// 3. Error handling: invalid inputs (no Metal)
// ===========================================================================

#[test]
fn test_import_model_missing_graph_json() {
    let err = nn_import::import_model(
        Path::new("/nonexistent/graph.json"),
        Path::new("/nonexistent/weights.safetensors"),
    )
    .unwrap_err();
    assert!(
        matches!(err, nn_import::ImportError::Io { .. }),
        "expected Io error for missing file, got: {err:?}"
    );
}

#[test]
fn test_import_model_missing_weights_file() {
    let dir = make_temp_dir("missing_weights");
    let graph_path = write_graph_json(&dir);

    let err = nn_import::import_model(&graph_path, Path::new("/nonexistent/weights.safetensors"))
        .unwrap_err();

    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        matches!(err, nn_import::ImportError::Io { .. }),
        "expected Io error for missing weights, got: {err:?}"
    );
}

#[test]
fn test_parse_invalid_json_returns_error() {
    let invalid_json = b"{ not valid json at all }";
    let err = nn_import::parse_exported_program(invalid_json).unwrap_err();
    assert!(
        matches!(err, nn_import::ImportError::JsonParse(_)),
        "expected JsonParse for invalid JSON, got: {err:?}"
    );
}

#[test]
fn test_parse_wrong_schema_version_returns_unsupported() {
    let bad_version_json = br#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 5, "minor": 0},
        "range_constraints": {}
    }"#;
    let err = nn_import::parse_exported_program(bad_version_json).unwrap_err();
    assert!(
        matches!(
            err,
            nn_import::ImportError::UnsupportedSchema { major: 5, .. }
        ),
        "expected UnsupportedSchema with major=5, got: {err:?}"
    );
}

#[test]
fn test_parse_schema_version_9_returns_unsupported() {
    let future_version_json = br#"{
        "graph_module": {
            "graph": {"inputs": [], "outputs": [], "nodes": [], "tensor_values": {}},
            "signature": {"input_specs": [], "output_specs": []},
            "module_call_graph": []
        },
        "schema_version": {"major": 9, "minor": 0},
        "range_constraints": {}
    }"#;
    let err = nn_import::parse_exported_program(future_version_json).unwrap_err();
    assert!(
        matches!(
            err,
            nn_import::ImportError::UnsupportedSchema { major: 9, .. }
        ),
        "expected UnsupportedSchema with major=9, got: {err:?}"
    );
}

#[test]
fn test_parse_empty_json_object_returns_error() {
    let empty_json = b"{}";
    let err = nn_import::parse_exported_program(empty_json).unwrap_err();
    assert!(
        matches!(err, nn_import::ImportError::JsonParse(_)),
        "expected JsonParse for missing fields, got: {err:?}"
    );
}

#[test]
fn test_import_model_corrupted_safetensors() {
    let dir = make_temp_dir("corrupted_weights");
    let graph_path = write_graph_json(&dir);
    let weights_path = dir.join("weights.safetensors");
    std::fs::write(&weights_path, b"this is not a valid safetensors file").unwrap();

    let err = nn_import::import_model(&graph_path, &weights_path).unwrap_err();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        matches!(err, nn_import::ImportError::Io { .. }),
        "expected Io error for corrupted safetensors, got: {err:?}"
    );
}

#[test]
fn test_import_model_empty_weight_map_returns_missing_weight() {
    let dir = make_temp_dir("empty_weights");
    let graph_path = write_graph_json(&dir);

    // Write a valid but empty safetensors file.
    let tensors: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    let weights_path = dir.join("weights.safetensors");
    std::fs::write(&weights_path, serialized).unwrap();

    let err = nn_import::import_model(&graph_path, &weights_path).unwrap_err();
    let _ = std::fs::remove_dir_all(&dir);

    // The graph expects 4 weight parameters but safetensors is empty.
    assert!(
        matches!(err, nn_import::ImportError::MissingWeight { .. }),
        "expected MissingWeight for empty weight map, got: {err:?}"
    );
}

#[test]
fn test_import_error_display_is_informative() {
    // Verify that ImportError Display trait provides useful diagnostic info.
    let err = nn_import::ImportError::UnsupportedSchema { major: 5, minor: 3 };
    let msg = format!("{err}");
    assert!(
        msg.contains("5.3"),
        "error message should contain version: {msg}"
    );
    assert!(
        msg.contains("expected major=8"),
        "error message should mention expected version: {msg}"
    );
}

#[test]
fn test_import_error_io_display() {
    let err = nn_import::ImportError::Io {
        path: "/foo/bar.json".to_string(),
        detail: "No such file".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("/foo/bar.json"), "should contain path: {msg}");
    assert!(msg.contains("No such file"), "should contain detail: {msg}");
}

// ===========================================================================
// 4. VerificationCoverage (no Metal)
// ===========================================================================

#[test]
fn test_verification_coverage_default_values() {
    let coverage = nn_import::VerificationCoverage::default();
    assert_eq!(coverage.gamma_crown_layers_covered, 0);
    assert_eq!(coverage.gamma_crown_layers_total, 0);
    assert!(!coverage.composition_bounds_ok);
    assert!(coverage.composition_bound_width.is_none());
    assert!(coverage.composition_method.is_none());
    assert!(coverage.composition_soundness_mode.is_none());
    assert!(coverage.composition_proof_strength.is_none());
    assert!(coverage.kani_harnesses_applicable.is_none());
    assert!(coverage.reference_parity_passed.is_none());
}

#[test]
fn test_verification_coverage_pct_zero_total_returns_zero() {
    let coverage = nn_import::VerificationCoverage::default();
    assert!((coverage.gamma_crown_coverage_pct()).abs() < f32::EPSILON);
}

#[test]
fn test_verification_coverage_pct_partial_coverage() {
    let mut coverage = nn_import::VerificationCoverage::default();
    coverage.gamma_crown_layers_covered = 30;
    coverage.gamma_crown_layers_total = 50;
    let pct = coverage.gamma_crown_coverage_pct();
    assert!((pct - 60.0).abs() < 0.01, "expected 60%, got {pct}");
}

#[test]
fn test_verification_coverage_pct_full_coverage() {
    let mut coverage = nn_import::VerificationCoverage::default();
    coverage.gamma_crown_layers_covered = 100;
    coverage.gamma_crown_layers_total = 100;
    let pct = coverage.gamma_crown_coverage_pct();
    assert!((pct - 100.0).abs() < 0.01, "expected 100%, got {pct}");
}

// ===========================================================================
// 5. ConvertBuilder with different OptLevel/VerifyLevel (Metal-gated)
// ===========================================================================

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_tests {
    use super::*;

    fn init_metal() -> nn_metal::PipelineCache {
        let _ = nn_metal::MetalBackend::init();
        nn_metal::register_metal_dyn_backend();
        nn_metal::PipelineCache::new(
            nn_metal::MetalContext::new().expect("Metal device required"),
        )
    }

    // -- OptLevel / VerifyLevel enum tests (require metal feature for export) --

    #[test]
    fn test_optlevel_default_is_full() {
        assert_eq!(nn_import::OptLevel::default(), nn_import::OptLevel::Full);
    }

    #[test]
    fn test_verifylevel_default_is_bounds() {
        assert_eq!(
            nn_import::VerifyLevel::default(),
            nn_import::VerifyLevel::Bounds
        );
    }

    #[test]
    fn test_optlevel_variants_are_distinct() {
        assert_ne!(nn_import::OptLevel::None, nn_import::OptLevel::Full);
        assert_ne!(nn_import::OptLevel::Full, nn_import::OptLevel::Aggressive);
        assert_ne!(nn_import::OptLevel::None, nn_import::OptLevel::Aggressive);
    }

    #[test]
    fn test_verifylevel_variants_are_distinct() {
        assert_ne!(
            nn_import::VerifyLevel::None,
            nn_import::VerifyLevel::Bounds
        );
        assert_ne!(
            nn_import::VerifyLevel::Bounds,
            nn_import::VerifyLevel::Full
        );
        assert_ne!(nn_import::VerifyLevel::None, nn_import::VerifyLevel::Full);
    }

    #[test]
    fn test_optlevel_debug_format() {
        let dbg = format!("{:?}", nn_import::OptLevel::Aggressive);
        assert_eq!(dbg, "Aggressive");
    }

    #[test]
    fn test_verifylevel_debug_format() {
        let dbg = format!("{:?}", nn_import::VerifyLevel::Full);
        assert_eq!(dbg, "Full");
    }

    // -- ConvertBuilder pipeline tests --

    #[test]
    fn test_convert_builder_optlevel_none_verify_none() {
        let cache = init_metal();
        let dir = make_temp_dir("opt_none_verify_none");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::None)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .expect("build with OptLevel::None + VerifyLevel::None must succeed");

        let _ = std::fs::remove_dir_all(&dir);

        // Proof: Kani is never inline, composition bounds skipped at VerifyLevel::None.
        assert!(result.result.proof.kernel_safety.is_none());
        assert!(result.result.proof.composition_bounds.is_none());
        assert!(result.result.proof.reference_parity.is_none());
        // Report should have valid import metrics.
        assert_eq!(result.report.num_user_inputs, 1);
        assert_eq!(result.report.op_count, 3);
        assert!(result.report.dispatch_count > 0);
    }

    #[test]
    fn test_convert_builder_optlevel_full_verify_none() {
        let cache = init_metal();
        let dir = make_temp_dir("opt_full_verify_none");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Full)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .expect("build with OptLevel::Full + VerifyLevel::None must succeed");

        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(result.report.op_count, 3);
        assert!(result.report.dispatch_count > 0);
        // VerifyLevel::None => no composition bounds in proof.
        assert!(result.result.proof.composition_bounds.is_none());
    }

    #[test]
    fn test_convert_builder_optlevel_aggressive_verify_none() {
        let cache = init_metal();
        let dir = make_temp_dir("opt_aggressive_verify_none");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Aggressive)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .expect("build with OptLevel::Aggressive must succeed");

        let _ = std::fs::remove_dir_all(&dir);

        // Aggressive is currently equivalent to Full.
        assert_eq!(result.report.op_count, 3);
        assert!(result.report.dispatch_count > 0);
    }

    #[test]
    #[cfg(feature = "verify")]
    fn test_convert_builder_verify_bounds_populates_verification_coverage() {
        let cache = init_metal();
        let dir = make_temp_dir("verify_bounds");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Full)
            .verify(nn_import::VerifyLevel::Bounds)
            .build()
            .expect("build with VerifyLevel::Bounds must succeed");

        let _ = std::fs::remove_dir_all(&dir);

        // VerifyLevel::Bounds triggers NY IBP (requires verify feature).
        let vc = &result.report.verification;
        assert!(
            vc.gamma_crown_layers_total > 0,
            "VerifyLevel::Bounds should populate NY total layers"
        );
        assert_eq!(
            vc.composition_method,
            Some(nn_import::ConvertCompositionMethod::Ibp)
        );
        assert_eq!(
            vc.composition_soundness_mode,
            Some(nn_import::ConvertSoundnessMode::Sound)
        );
        assert_eq!(
            vc.composition_proof_strength,
            Some(nn_import::ConvertProofStrength::SoundIbp)
        );
    }

    #[test]
    #[cfg(feature = "verify")]
    fn test_convert_builder_verify_full_populates_verification_coverage() {
        let cache = init_metal();
        let dir = make_temp_dir("verify_full");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Full)
            .verify(nn_import::VerifyLevel::Full)
            .build()
            .expect("build with VerifyLevel::Full must succeed");

        let _ = std::fs::remove_dir_all(&dir);

        // VerifyLevel::Full also runs composition bounds (requires verify feature).
        let vc = &result.report.verification;
        assert!(
            vc.gamma_crown_layers_total > 0,
            "VerifyLevel::Full should populate NY total layers"
        );
        assert_eq!(
            vc.composition_method,
            Some(nn_import::ConvertCompositionMethod::Ibp)
        );
        assert_eq!(
            vc.composition_soundness_mode,
            Some(nn_import::ConvertSoundnessMode::Sound)
        );
        assert_eq!(
            vc.composition_proof_strength,
            Some(nn_import::ConvertProofStrength::SoundIbp)
        );
    }

    #[test]
    fn test_convert_builder_verify_none_skips_verification() {
        let cache = init_metal();
        let dir = make_temp_dir("verify_skip");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Full)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .expect("build with VerifyLevel::None must succeed");

        let _ = std::fs::remove_dir_all(&dir);

        // VerifyLevel::None skips NY entirely.
        let vc = &result.report.verification;
        assert_eq!(
            vc.gamma_crown_layers_total, 0,
            "VerifyLevel::None should leave NY counts at zero"
        );
        assert!(!vc.composition_bounds_ok);
        assert!(vc.composition_bound_width.is_none());
        assert!(vc.composition_method.is_none());
        assert!(vc.composition_soundness_mode.is_none());
        assert!(vc.composition_proof_strength.is_none());
    }

    #[test]
    fn test_convert_builder_equivalence_proof_l1_always_none() {
        let cache = init_metal();
        let dir = make_temp_dir("proof_l1_none");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        // L1 (Kani) is always None in the pipeline (run offline by Prover).
        for verify_level in [
            nn_import::VerifyLevel::None,
            nn_import::VerifyLevel::Bounds,
            nn_import::VerifyLevel::Full,
        ] {
            let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
                .optimize(nn_import::OptLevel::Full)
                .verify(verify_level)
                .build()
                .unwrap_or_else(|e| panic!("build with VerifyLevel::{verify_level:?} failed: {e}"));

            assert!(
                result.result.proof.kernel_safety.is_none(),
                "L1 Kani should always be None at VerifyLevel::{verify_level:?}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_convert_builder_equivalence_proof_l3_none_without_ref_trace() {
        let cache = init_metal();
        let dir = make_temp_dir("proof_l3_none");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        // L3 (reference parity) should be None when no reference trace is provided.
        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Full)
            .verify(nn_import::VerifyLevel::Full)
            .build()
            .expect("build must succeed");

        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            result.result.proof.reference_parity.is_none(),
            "L3 should be None without reference_trace"
        );
    }

    #[test]
    fn test_convert_builder_report_fields_stable_across_opt_levels() {
        let cache = init_metal();
        let dir = make_temp_dir("opt_level_compare");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result_none = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::None)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .unwrap();

        let result_full = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Full)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .unwrap();

        let result_aggressive = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Aggressive)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .unwrap();

        let _ = std::fs::remove_dir_all(&dir);

        // Import metrics are independent of optimization level.
        assert_eq!(result_none.report.op_count, result_full.report.op_count);
        assert_eq!(
            result_full.report.op_count,
            result_aggressive.report.op_count
        );
        assert_eq!(
            result_none.report.num_user_inputs,
            result_full.report.num_user_inputs
        );
        assert_eq!(
            result_none.report.num_weights_loaded,
            result_full.report.num_weights_loaded
        );
        assert_eq!(
            result_none.report.mapped_ops_count(),
            result_full.report.mapped_ops_count()
        );

        // Optimized dispatches should be <= unoptimized.
        assert!(
            result_full.report.dispatch_count <= result_none.report.dispatch_count,
            "Full ({}) should not exceed None ({})",
            result_full.report.dispatch_count,
            result_none.report.dispatch_count
        );
    }

    #[test]
    fn test_convert_builder_compiled_model_metadata() {
        use nn_core::DType;

        let cache = init_metal();
        let dir = make_temp_dir("model_metadata");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Full)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .unwrap();

        let _ = std::fs::remove_dir_all(&dir);

        let model = &result.result.model;
        assert_eq!(model.num_inputs(), 1);
        assert_eq!(model.output_shape(), &[1, 3]);
        assert_eq!(model.output_dtype(), DType::F32);
        assert!(model.num_dispatches() > 0);
        assert!(model.num_steps() > 0);
    }

    #[test]
    fn test_convert_builder_gpu_execution_produces_finite_output() {
        use nn_core::{DType, Device};

        let cache = init_metal();
        let dir = make_temp_dir("gpu_exec");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Full)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .unwrap();

        let _ = std::fs::remove_dir_all(&dir);

        // Execute on GPU with deterministic input.
        let input_data = vec![0.5_f32, -0.3, 0.8, -0.1];
        let input_cpu = nn_core::DynTensor::from_vec(input_data, &[1, 4], &Device::Cpu).unwrap();
        let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

        let output = result
            .result
            .model
            .execute_dyn(&cache, &[&input_gpu])
            .expect("GPU execution must succeed");

        assert_eq!(output.dims(), &[1, 3]);
        assert_eq!(output.dtype(), DType::F32);

        let output_cpu = output.to_device(&Device::Cpu).unwrap();
        let vals = output_cpu.to_flat_vec::<f32>().unwrap();
        assert_eq!(vals.len(), 3);
        for (i, &v) in vals.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
        // Weights are 0.01*i, so output should be non-trivial.
        let any_nonzero = vals.iter().any(|&v| v.abs() > 1e-10);
        assert!(any_nonzero, "GPU output is all zeros");
    }

    #[test]
    fn test_convert_builder_missing_graph_returns_convert_error() {
        let cache = init_metal();

        let result = nn_import::convert_build(
            Path::new("/nonexistent/graph.json"),
            Path::new("/nonexistent/weights.safetensors"),
            &cache,
        )
        .optimize(nn_import::OptLevel::Full)
        .verify(nn_import::VerifyLevel::None)
        .build();

        match result {
            Err(err) => {
                assert!(
                    matches!(err, nn_import::ConvertError::Import(_)),
                    "expected ConvertError::Import for missing files, got: {err:?}"
                );
            }
            Ok(_) => panic!("expected error for nonexistent files, got Ok"),
        }
    }

    #[test]
    fn test_convert_builder_report_rtf_populated() {
        let cache = init_metal();
        let dir = make_temp_dir("rtf_check");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Full)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .unwrap();

        let _ = std::fs::remove_dir_all(&dir);

        // RTF estimate should be populated for a compiled model with dispatches.
        if result.report.metal_dispatches > 0 {
            assert!(
                result.report.estimated_rtf.is_some(),
                "RTF should be populated when metal_dispatches > 0"
            );
            let rtf = result.report.estimated_rtf.unwrap();
            assert!(
                rtf > 0.0 && rtf.is_finite(),
                "RTF should be positive and finite, got {rtf}"
            );
        }
    }

    #[test]
    fn test_convert_builder_report_display_and_json() {
        let cache = init_metal();
        let dir = make_temp_dir("report_display");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .optimize(nn_import::OptLevel::Full)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .unwrap();

        let _ = std::fs::remove_dir_all(&dir);

        // Display should produce a human-readable report without panicking.
        let text = format!("{}", result.report);
        assert!(text.contains("Conversion complete:"));
        assert!(text.contains("Intake path:   exported artifacts"));
        assert!(text.contains("Artifact kind: compiled Metal artifact"));
        assert!(text.contains("Compiled Metal artifact ready for GPU execution."));
        assert_eq!(
            result.report.provenance_summary(),
            "exported artifacts -> compiled Metal artifact"
        );

        // JSON serialization should produce valid JSON.
        let json = result.report.to_json();
        let val: serde_json::Value =
            serde_json::from_str(&json).expect("ConvertReport JSON must be valid");
        assert_eq!(val["intake_path"], "exported_artifacts");
        assert_eq!(val["artifact_kind"], "compiled_metal_artifact");
        assert_eq!(val["num_user_inputs"], 1);
        assert_eq!(val["op_count"], 3);
        assert!(val["dispatch_count"].as_u64().unwrap() > 0);

        // Summary table should have correct structure.
        let table = result.report.summary_table();
        assert!(table.contains("| Metric | Value |"));
        assert!(table.contains("| Provenance | exported artifacts -> compiled Metal artifact |"));
        assert!(table.contains("| Intake path | exported artifacts |"));
        assert!(table.contains("| Artifact kind | compiled Metal artifact |"));
        assert!(table.contains("| Input ops | 3 |"));
    }

    #[test]
    fn test_convert_builder_report_cli_exported_pytorch_provenance() {
        let cache = init_metal();
        let dir = make_temp_dir("report_cli_exported_pytorch");
        let graph_path = write_graph_json(&dir);
        let weights_path = write_mlp_weights(&dir);

        let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
            .cli_exported_from_pytorch()
            .optimize(nn_import::OptLevel::Full)
            .verify(nn_import::VerifyLevel::None)
            .build()
            .unwrap();

        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(
            result.report.intake_path,
            nn_import::ConvertIntakePath::CliExportedPytorch
        );
        assert_eq!(
            result.report.artifact_kind,
            nn_import::ConvertArtifactKind::CompiledMetalArtifact
        );
        assert_eq!(
            result.report.provenance_summary(),
            "CLI-exported PyTorch -> compiled Metal artifact"
        );

        let json = result.report.to_json();
        let val: serde_json::Value =
            serde_json::from_str(&json).expect("ConvertReport JSON must be valid");
        assert_eq!(val["intake_path"], "cli_exported_pytorch");
        assert_eq!(val["artifact_kind"], "compiled_metal_artifact");

        let table = result.report.summary_table();
        assert!(table.contains("| Provenance | CLI-exported PyTorch -> compiled Metal artifact |"));
    }
}
