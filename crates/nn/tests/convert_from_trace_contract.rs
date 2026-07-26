// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "convert-model")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use nn::{convert_from_trace, ConvertConfig};
use nn_models::convert::DpdfModelType;

struct ExportFixtureDir {
    path: PathBuf,
}

impl ExportFixtureDir {
    fn new(name: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nn_{name}_{}_{}_{}",
            std::process::id(),
            unique,
            std::thread::current().name().unwrap_or("unnamed"),
        ));
        std::fs::create_dir_all(&path).expect("create export fixture dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ExportFixtureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn write_mlp_export_fixture() -> (ExportFixtureDir, PathBuf, PathBuf) {
    let dir = ExportFixtureDir::new("convert_from_trace_contract");

    let graph_path = dir.path().join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../../nn-import/test_data/e2e_mlp.json"),
    )
    .expect("write graph fixture");

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
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8, 4], &fc1_w)
            .expect("fc1.weight tensor view"),
    );
    tensors.insert(
        "fc1.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &fc1_b)
            .expect("fc1.bias tensor view"),
    );
    tensors.insert(
        "fc2.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 8], &fc2_w)
            .expect("fc2.weight tensor view"),
    );
    tensors.insert(
        "fc2.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3], &fc2_b)
            .expect("fc2.bias tensor view"),
    );

    let weights_path = dir.path().join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).expect("serialize safetensors");
    std::fs::write(&weights_path, serialized).expect("write weight fixture");

    (dir, graph_path, weights_path)
}

fn write_mlp_export_fixture_with_layoutlmv3_extra_weight() -> (ExportFixtureDir, PathBuf, PathBuf) {
    let dir = ExportFixtureDir::new("convert_from_trace_contract_layoutlmv3_weight");

    let graph_path = dir.path().join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../../nn-import/test_data/e2e_mlp.json"),
    )
    .expect("write graph fixture");

    let fc1_w: Vec<u8> = (0..32u32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc1_b: Vec<u8> = [0.0f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc2_w: Vec<u8> = (0..24u32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let fc2_b: Vec<u8> = [0.0f32; 3].iter().flat_map(|f| f.to_le_bytes()).collect();
    let extra_layoutlmv3_weight: Vec<u8> = (0..6u32)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "fc1.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8, 4], &fc1_w)
            .expect("fc1.weight tensor view"),
    );
    tensors.insert(
        "fc1.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &fc1_b)
            .expect("fc1.bias tensor view"),
    );
    tensors.insert(
        "fc2.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 8], &fc2_w)
            .expect("fc2.weight tensor view"),
    );
    tensors.insert(
        "fc2.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3], &fc2_b)
            .expect("fc2.bias tensor view"),
    );
    tensors.insert(
        "model.embeddings.word_embeddings.weight".to_string(),
        safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![2, 3],
            &extra_layoutlmv3_weight,
        )
        .expect("layoutlmv3 extra tensor view"),
    );

    let weights_path = dir.path().join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).expect("serialize safetensors");
    std::fs::write(&weights_path, serialized).expect("write weight fixture");

    (dir, graph_path, weights_path)
}

#[test]
fn convert_from_trace_returns_imported_converted_model() {
    let (_dir, graph_path, weights_path) = write_mlp_export_fixture();
    let config = ConvertConfig::new("test-mlp");

    let model = convert_from_trace(&graph_path, &weights_path, &config)
        .expect("convert exported artifacts into ConvertedModel");

    assert_eq!(model.model_name, "test-mlp");
    assert_eq!(model.num_ops(), 8);
    assert_eq!(model.num_weights(), 4);
    assert_eq!(model.input_names(), &["x"]);
    assert_eq!(model.output_names(), &["linear_1"]);
}

#[test]
fn convert_from_trace_reserved_config_knobs_do_not_change_import_contract() {
    let (_dir, graph_path, weights_path) = write_mlp_export_fixture();
    let baseline_config = ConvertConfig::new("test-mlp");
    let reserved_knobs_config = ConvertConfig::new("test-mlp")
        .with_validate_weights(false)
        .with_constant_fold(false);

    let baseline = convert_from_trace(&graph_path, &weights_path, &baseline_config)
        .expect("convert exported artifacts with default config");
    let reserved = convert_from_trace(&graph_path, &weights_path, &reserved_knobs_config)
        .expect("convert exported artifacts with reserved config knobs");

    assert_eq!(reserved.model_name, baseline.model_name);
    assert_eq!(reserved.num_ops(), baseline.num_ops());
    assert_eq!(reserved.num_inputs(), baseline.num_inputs());
    assert_eq!(reserved.num_weights(), baseline.num_weights());
    assert_eq!(reserved.total_params(), baseline.total_params());
    assert_eq!(reserved.input_names(), baseline.input_names());
    assert_eq!(reserved.output_names(), baseline.output_names());

    let mut baseline_weight_names: Vec<_> = baseline.weights.keys().cloned().collect();
    let mut reserved_weight_names: Vec<_> = reserved.weights.keys().cloned().collect();
    baseline_weight_names.sort();
    reserved_weight_names.sort();
    assert_eq!(reserved_weight_names, baseline_weight_names);
}

#[test]
fn convert_from_trace_accepts_model_type_on_converted_model_surface() {
    let (_dir, graph_path, weights_path) = write_mlp_export_fixture_with_layoutlmv3_extra_weight();
    let baseline_config = ConvertConfig::new("test-mlp");
    let model_type_config =
        ConvertConfig::new("test-mlp").with_model_type(DpdfModelType::LayoutLMv3);

    let baseline = convert_from_trace(&graph_path, &weights_path, &baseline_config)
        .expect("convert exported artifacts with default config");
    let with_model_type = convert_from_trace(&graph_path, &weights_path, &model_type_config)
        .expect("convert exported artifacts with model_type on ConvertedModel surface");

    assert_eq!(with_model_type.model_name, baseline.model_name);
    assert_eq!(with_model_type.num_ops(), baseline.num_ops());
    assert_eq!(with_model_type.num_inputs(), baseline.num_inputs());
    assert_eq!(with_model_type.num_weights(), baseline.num_weights());
    assert_eq!(with_model_type.total_params(), baseline.total_params());
    assert_eq!(with_model_type.input_names(), baseline.input_names());
    assert_eq!(with_model_type.output_names(), baseline.output_names());

    assert!(
        baseline
            .weights
            .contains_key("model.embeddings.word_embeddings.weight"),
        "baseline ConvertedModel should keep the original safetensors key"
    );
    assert!(
        !baseline
            .weights
            .contains_key("text_embeddings.word_embeddings.weight"),
        "baseline ConvertedModel should not expose the remapped LayoutLMv3 key"
    );
    assert!(
        with_model_type
            .weights
            .contains_key("text_embeddings.word_embeddings.weight"),
        "model_type should remap matching weight keys on the ConvertedModel surface"
    );
    assert!(
        !with_model_type
            .weights
            .contains_key("model.embeddings.word_embeddings.weight"),
        "remapped ConvertedModel should not retain the original matching key"
    );
}
