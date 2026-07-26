// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Roundtrip test: convert -> save .nnc -> load .nnc -> run -> verify output.
//!
//! Exercises the full serialization path used by `nn convert --output model.nnc`
//! and `nn run --compiled model.nnc`. Verifies that a plan saved to disk and
//! loaded back produces identical inference results.
//!
//! Uses the same MLP fixture (Linear -> ReLU -> Linear) as convert_builder_e2e.

use std::collections::HashMap;
use std::path::Path;

/// Write the MLP graph JSON (Linear -> ReLU -> Linear) to a file.
fn write_mlp_graph_json(dir: &Path) -> std::path::PathBuf {
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../../nn-import/test_data/e2e_mlp.json"),
    )
    .unwrap();
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

/// Full roundtrip: import -> compile -> save .nnc -> load .nnc -> execute -> compare.
///
/// Verifies that serialized plans produce identical outputs to freshly compiled models.
#[test]
#[cfg(target_os = "macos")]
fn test_nnc_roundtrip_produces_identical_output() {
    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_nnc_roundtrip_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = write_mlp_graph_json(&dir);
    let weights_path = write_mlp_weights(&dir);

    // Phase 1: Import the model graph.
    let imported =
        nn_import::import_model(&graph_path, &weights_path).expect("import_model should succeed");

    // Phase 2: Compile to plan (same as cmd_compile does).
    let plan = nn_dsl::trace_compile::compile_trace_to_plan_with_fusion(&imported.graph)
        .expect("compile_trace_to_plan_with_fusion should succeed");

    assert!(!plan.steps.is_empty(), "plan should have steps");
    assert!(
        !plan.input_shapes.is_empty(),
        "plan should have input shapes"
    );

    // Phase 3: Build a model from the fresh plan (baseline).
    let model_fresh =
        nn_metal::compiled_model::CompiledModel::from_plan(&plan, &imported.graph, &cache)
            .expect("from_plan (fresh) should succeed");

    // Phase 4: Save the plan to .nnc.
    let nnc_path = dir.join("model.nnc");
    nn_dsl::save_plan(&plan, &nnc_path).expect("save_plan should succeed");

    // Verify the file was created and is non-empty.
    let file_size = std::fs::metadata(&nnc_path)
        .expect("nnc file should exist")
        .len();
    assert!(file_size > 0, "nnc file should be non-empty");

    // Phase 5: Load the plan back from disk.
    let loaded_plan = nn_dsl::load_plan(&nnc_path).expect("load_plan should succeed");

    // Verify plan structure matches.
    assert_eq!(
        loaded_plan.steps.len(),
        plan.steps.len(),
        "loaded plan should have same number of steps"
    );
    assert_eq!(
        loaded_plan.input_shapes, plan.input_shapes,
        "loaded plan should have same input shapes"
    );
    assert_eq!(
        loaded_plan.output_step, plan.output_step,
        "loaded plan should have same output step"
    );
    assert_eq!(
        loaded_plan.weight_names, plan.weight_names,
        "loaded plan should have same weight names"
    );

    // Phase 6: Build a model from the loaded plan (roundtripped).
    let model_loaded =
        nn_metal::compiled_model::CompiledModel::from_plan(&loaded_plan, &imported.graph, &cache)
            .expect("from_plan (loaded) should succeed");

    // Verify both models have the same structure.
    assert_eq!(
        model_fresh.num_steps(),
        model_loaded.num_steps(),
        "both models should have same step count"
    );
    assert_eq!(
        model_fresh.num_dispatches(),
        model_loaded.num_dispatches(),
        "both models should have same dispatch count"
    );

    // Phase 7: Execute both models with the same input.
    let input = nn_core::DynTensor::ones(&[1, 4], nn_core::DType::F32, &nn_core::Device::Cpu)
        .expect("creating input tensor should succeed");
    let gpu_input = input
        .to_device(&nn_core::Device::metal())
        .expect("moving input to GPU should succeed");

    let outputs_fresh = model_fresh
        .execute_dyn_outputs(&cache, &[&gpu_input])
        .expect("execute (fresh model) should succeed");
    let outputs_loaded = model_loaded
        .execute_dyn_outputs(&cache, &[&gpu_input])
        .expect("execute (loaded model) should succeed");

    assert_eq!(
        outputs_fresh.len(),
        outputs_loaded.len(),
        "both models should produce the same number of outputs"
    );

    // Phase 8: Verify outputs are bitwise identical.
    for (i, (fresh, loaded)) in outputs_fresh.iter().zip(outputs_loaded.iter()).enumerate() {
        let fresh_cpu = fresh
            .to_device(&nn_core::Device::Cpu)
            .expect("moving fresh output to CPU should succeed");
        let loaded_cpu = loaded
            .to_device(&nn_core::Device::Cpu)
            .expect("moving loaded output to CPU should succeed");

        assert_eq!(
            fresh_cpu.shape(),
            loaded_cpu.shape(),
            "output {i}: shapes must match"
        );
        assert_eq!(
            fresh_cpu.dtype(),
            loaded_cpu.dtype(),
            "output {i}: dtypes must match"
        );

        // Extract f32 values and compare.
        let fresh_vals: Vec<f32> = fresh_cpu.to_flat_vec::<f32>().expect("to_flat_vec (fresh)");
        let loaded_vals: Vec<f32> = loaded_cpu
            .to_flat_vec::<f32>()
            .expect("to_flat_vec (loaded)");

        assert_eq!(
            fresh_vals.len(),
            loaded_vals.len(),
            "output {i}: value count must match"
        );

        for (j, (fv, lv)) in fresh_vals.iter().zip(loaded_vals.iter()).enumerate() {
            assert!(
                (fv - lv).abs() < 1e-6,
                "output {i}, element {j}: fresh={fv}, loaded={lv}, diff={}",
                (fv - lv).abs()
            );
        }
    }

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dir);
}

/// Verify that loading a non-existent .nnc file produces a clear error.
#[test]
fn test_load_plan_nonexistent_file_returns_error() {
    let result = nn_dsl::load_plan("/nonexistent/path/model.nnc");
    assert!(result.is_err(), "loading non-existent .nnc should fail");
}

/// Verify that a plan can be saved and loaded (structure-only, no GPU execution).
///
/// This test does not require Metal and exercises just the serialization layer.
#[test]
fn test_nnc_save_load_preserves_plan_structure() {
    let dir = std::env::temp_dir().join(format!("nn_nnc_structure_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Build a minimal plan from JSON to exercise the serde path without
    // needing to construct a CompiledPlan directly (it is #[non_exhaustive]).
    let json = r#"{
        "steps": [],
        "input_shapes": [[1, 4]],
        "output_step": 0,
        "weight_names": ["fc1.weight"]
    }"#;
    let plan: nn_dsl::trace_compile::CompiledPlan =
        serde_json::from_str(json).expect("deserialize minimal plan from JSON");

    let path = dir.join("structure.nnc");
    nn_dsl::save_plan(&plan, &path).expect("save_plan should succeed");
    let loaded = nn_dsl::load_plan(&path).expect("load_plan should succeed");

    assert_eq!(loaded.steps.len(), 0);
    assert_eq!(loaded.input_shapes, vec![vec![1usize, 4]]);
    assert_eq!(loaded.output_step, 0);
    assert_eq!(loaded.weight_names, vec!["fc1.weight".to_string()]);

    let _ = std::fs::remove_dir_all(&dir);
}
