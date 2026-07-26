// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use nn_dsl::trace_compile::{CompiledPlan, CompiledStep};
use nn_import::ConvertReport;

fn write_mlp_graph_json(dir: &Path) -> PathBuf {
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../../nn-import/test_data/e2e_mlp.json"),
    )
    .unwrap();
    graph_path
}

fn write_mlp_weights(dir: &Path) -> PathBuf {
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

fn write_mlp_input(dir: &Path) -> PathBuf {
    let input_x: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "x".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1, 4], &input_x)
            .unwrap(),
    );

    let input_path = dir.join("input.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&input_path, serialized).unwrap();
    input_path
}

fn load_report_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read report file"))
        .expect("report file should be valid JSON")
}

fn build_default_compile_report(graph_path: &Path, weights_path: &Path) -> ConvertReport {
    let _backend = nn_metal::MetalBackend::init().expect("initialize Metal backend");
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new_global().expect("create pipeline cache");
    nn_import::convert_build(graph_path, weights_path, &cache)
        .build()
        .expect("build default exported-artifact report")
        .report
}

fn assert_matches_default_builder_report_contract(
    report_json: &serde_json::Value,
    graph_path: &Path,
    weights_path: &Path,
) {
    let expected = build_default_compile_report(graph_path, weights_path);

    assert_eq!(
        report_json["intake_path"],
        serde_json::to_value(expected.intake_path).expect("serialize intake path"),
        "compile report intake path should match the default builder contract"
    );
    assert_eq!(
        report_json["artifact_kind"],
        serde_json::to_value(expected.artifact_kind).expect("serialize artifact kind"),
        "compile report artifact kind should match the default builder contract"
    );
    assert_eq!(
        report_json["num_user_inputs"],
        serde_json::to_value(expected.num_user_inputs).expect("serialize num_user_inputs"),
        "compile report input arity should match the default builder contract"
    );
    assert_eq!(
        report_json["num_weights_loaded"],
        serde_json::to_value(expected.num_weights_loaded).expect("serialize num_weights_loaded"),
        "compile report weight count should match the default builder contract"
    );
    assert_eq!(
        report_json["dispatch_count"],
        serde_json::to_value(expected.dispatch_count).expect("serialize dispatch_count"),
        "compile report dispatch count should match the default builder contract"
    );
    assert_eq!(
        report_json["total_steps"],
        serde_json::to_value(expected.total_steps).expect("serialize total_steps"),
        "compile report total steps should match the default builder contract"
    );
    assert_eq!(
        report_json["metal_dispatches"],
        serde_json::to_value(expected.metal_dispatches).expect("serialize metal_dispatches"),
        "compile report Metal dispatch count should match the default builder contract"
    );
    assert_eq!(
        report_json["verification"],
        serde_json::to_value(&expected.verification).expect("serialize verification coverage"),
        "compile report verification coverage should match the default builder contract"
    );
}

fn dispatch_step_count(plan: &CompiledPlan) -> usize {
    plan.steps
        .iter()
        .filter(|step| {
            matches!(
                step,
                CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
            )
        })
        .count()
}

fn assert_report_describes_saved_plan(report_json: &serde_json::Value, plan_path: &Path) {
    let plan = nn_dsl::load_plan(plan_path).expect("saved .nnc plan should load");

    assert!(
        !plan.steps.is_empty(),
        "saved .nnc plan should contain compiled steps"
    );
    assert!(
        plan.output_step < plan.steps.len(),
        "saved .nnc output_step should index into the compiled steps"
    );
    assert_eq!(
        plan.steps.len() as u64,
        report_json["total_steps"]
            .as_u64()
            .expect("report total_steps should be a u64"),
        "report total_steps should match the saved .nnc step count"
    );
    assert_eq!(
        dispatch_step_count(&plan) as u64,
        report_json["dispatch_count"]
            .as_u64()
            .expect("report dispatch_count should be a u64"),
        "report dispatch_count should match GPU-dispatching steps in the saved .nnc plan"
    );
    assert_eq!(
        plan.input_shapes.len() as u64,
        report_json["num_user_inputs"]
            .as_u64()
            .expect("report num_user_inputs should be a u64"),
        "report num_user_inputs should match saved .nnc input arity"
    );
    assert!(
        !plan.weight_names.is_empty(),
        "saved .nnc plan should retain a serialized weight inventory"
    );
    assert!(
        plan.weight_names.len() as u64
            <= report_json["num_weights_loaded"]
                .as_u64()
                .expect("report num_weights_loaded should be a u64"),
        "saved .nnc weight inventory should not exceed the import-side weight placeholder count recorded in the report"
    );
}

fn assert_compiled_plan_runs_via_cli(
    graph_path: &Path,
    weights_path: &Path,
    compiled_path: &Path,
    dir: &Path,
) {
    let input_path = write_mlp_input(dir);
    let output_path = dir.join("outputs.safetensors");

    let output = Command::new(env!("CARGO_BIN_EXE_nn"))
        .arg("run")
        .arg(graph_path)
        .arg(weights_path)
        .arg("--compiled")
        .arg(compiled_path)
        .arg("--input")
        .arg(&input_path)
        .arg("--output")
        .arg(&output_path)
        .output()
        .expect("run nn run --compiled");

    assert!(
        output.status.success(),
        "nn run --compiled failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Loading pre-compiled plan"),
        "run stderr should prove the persisted .nnc artifact was consumed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output_path.exists(),
        "nn run --compiled should write output safetensors"
    );

    let outputs = nn_core::load_safetensors(&output_path).expect("load run outputs");
    assert_eq!(
        outputs.len(),
        1,
        "the saved outputs should contain exactly the exported single model output"
    );
    let output_tensor = outputs
        .get("linear_1")
        .expect("saved outputs should preserve the exported output tensor name");
    assert_eq!(
        output_tensor.dims(),
        &[1, 3],
        "the compiled artifact should execute to the expected output shape"
    );
}

#[test]
fn test_compile_persists_structured_report_and_plan() {
    let dir = tempfile::tempdir().expect("tempdir");
    let graph_path = write_mlp_graph_json(dir.path());
    let weights_path = write_mlp_weights(dir.path());
    let output_path = dir.path().join("artifacts/model.nnc");
    let report_path = dir.path().join("reports/model.compile.json");
    let default_report_path = dir.path().join("artifacts/model.compile.json");

    let output = Command::new(env!("CARGO_BIN_EXE_nn"))
        .arg("compile")
        .arg(&graph_path)
        .arg(&weights_path)
        .arg("--output")
        .arg(&output_path)
        .arg("--report-output")
        .arg(&report_path)
        .arg("--json")
        .output()
        .expect("run nn compile");

    assert!(
        output.status.success(),
        "nn compile failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(output_path.exists(), "compiled plan should be written");
    assert!(report_path.exists(), "report JSON should be written");
    assert!(
        !default_report_path.exists(),
        "explicit --report-output should suppress the default sibling report path"
    );
    assert!(
        std::fs::metadata(&output_path).unwrap().len() > 0,
        ".nnc output should be non-empty"
    );
    assert!(
        std::fs::metadata(&report_path).unwrap().len() > 0,
        "report JSON should be non-empty"
    );

    let stdout_json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout should be valid JSON");
    let report_json = load_report_json(&report_path);

    assert_eq!(
        stdout_json, report_json,
        "--json stdout and --report-output should persist the same structured report"
    );
    assert_report_describes_saved_plan(&stdout_json, &output_path);
    assert_eq!(stdout_json["intake_path"], "exported_artifacts");
    assert_eq!(stdout_json["artifact_kind"], "compiled_metal_artifact");
    assert_eq!(stdout_json["num_user_inputs"], 1);
    assert_eq!(stdout_json["num_weights_loaded"], 4);
    assert!(
        stdout_json["dispatch_count"].as_u64().unwrap_or(0) > 0,
        "dispatch_count should be populated"
    );
    assert!(
        stdout_json["total_steps"].as_u64().unwrap_or(0) > 0,
        "total_steps should be populated"
    );
    assert_matches_default_builder_report_contract(&stdout_json, &graph_path, &weights_path);
    assert_compiled_plan_runs_via_cli(&graph_path, &weights_path, &output_path, dir.path());
}

#[test]
fn test_compile_persists_default_sibling_report_when_not_explicitly_requested() {
    let dir = tempfile::tempdir().expect("tempdir");
    let graph_path = write_mlp_graph_json(dir.path());
    let weights_path = write_mlp_weights(dir.path());
    let output_path = dir.path().join("artifacts/model.nnc");
    let report_path = dir.path().join("artifacts/model.compile.json");

    let output = Command::new(env!("CARGO_BIN_EXE_nn"))
        .arg("compile")
        .arg(&graph_path)
        .arg(&weights_path)
        .arg("--output")
        .arg(&output_path)
        .arg("--json")
        .output()
        .expect("run nn compile");

    assert!(
        output.status.success(),
        "nn compile failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(output_path.exists(), "compiled plan should be written");
    assert!(
        report_path.exists(),
        "default sibling report JSON should be written next to the .nnc output"
    );

    let stdout_json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout should be valid JSON");
    let report_json = load_report_json(&report_path);

    assert_eq!(
        stdout_json, report_json,
        "default persisted report should match --json stdout"
    );
    assert_report_describes_saved_plan(&stdout_json, &output_path);
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("default sibling report"),
        "stderr should explain the default compile report location:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(report_path.to_string_lossy().as_ref()),
        "stderr should print the default report path:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
