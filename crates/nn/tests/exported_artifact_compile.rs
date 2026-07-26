// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(feature = "metal", feature = "convert-model"))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use nn::metal::{CompiledModel, MetalBackend, MetalElement, PipelineCache};
use nn::{
    compile_exported_artifacts, compile_metal_from_exported_artifacts,
    compile_metal_from_exported_artifacts_with_report, convert_from_trace, ConvertConfig,
    ConvertError, ConvertedModelMetadata, ExportedArtifactCompileError,
};
use nn_import::VerifyLevel;
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

fn write_mlp_export_fixture_impl(
    include_extra_unused_tensor: bool,
) -> (ExportFixtureDir, PathBuf, PathBuf) {
    let dir = ExportFixtureDir::new("exported_artifact_compile");

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
    let extra_unused = include_extra_unused_tensor.then(|| {
        (0..5u32)
            .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
            .collect::<Vec<u8>>()
    });

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
    if let Some(extra_unused) = extra_unused.as_ref() {
        tensors.insert(
            "extra_unused.weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![5], extra_unused)
                .expect("extra unused tensor view"),
        );
    }

    let weights_path = dir.path().join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).expect("serialize safetensors");
    std::fs::write(&weights_path, serialized).expect("write weight fixture");

    (dir, graph_path, weights_path)
}

fn write_mlp_export_fixture() -> (ExportFixtureDir, PathBuf, PathBuf) {
    write_mlp_export_fixture_impl(false)
}

fn write_mlp_export_fixture_with_extra_unused_tensor() -> (ExportFixtureDir, PathBuf, PathBuf) {
    write_mlp_export_fixture_impl(true)
}

fn metal_cache() -> Option<PipelineCache> {
    let _backend = MetalBackend::init().ok()?;
    PipelineCache::new_global().ok()
}

fn assert_mlp_metadata(metadata: &ConvertedModelMetadata) {
    assert_eq!(metadata.model_name, "test-mlp");
    assert_eq!(metadata.num_ops, 8);
    assert_eq!(metadata.num_inputs, 1);
    assert_eq!(metadata.num_weights, 4);
    assert_eq!(metadata.total_params, 67);
    assert_eq!(metadata.input_names, ["x"]);
    assert_eq!(metadata.output_names, ["linear_1"]);
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: output length mismatch"
    );
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "{label}[{idx}]: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
fn compile_metal_from_exported_artifacts_retains_metadata() {
    let Some(cache) = metal_cache() else {
        return;
    };

    let (_dir, graph_path, weights_path) = write_mlp_export_fixture();
    let config = ConvertConfig::new("test-mlp");

    let compiled =
        compile_metal_from_exported_artifacts(&graph_path, &weights_path, &config, &cache)
            .expect("compile exported artifacts to metal");

    assert_mlp_metadata(&compiled.metadata);

    assert!(
        compiled.model.num_steps() > 0,
        "compiled model should have steps"
    );
    assert!(
        compiled.model.num_dispatches() > 0,
        "compiled model should have dispatches"
    );
}

#[test]
fn compile_metal_from_exported_artifacts_with_report_preserves_report_contents() {
    let Some(cache) = metal_cache() else {
        return;
    };

    let (_dir, graph_path, weights_path) = write_mlp_export_fixture();
    let config = ConvertConfig::new("test-mlp");

    let compiled = compile_metal_from_exported_artifacts_with_report(
        &graph_path,
        &weights_path,
        &config,
        &cache,
    )
    .expect("compile exported artifacts to metal with report");

    assert_mlp_metadata(&compiled.metadata);

    let report = &compiled.report;
    assert_eq!(report.op_count, 3);
    assert_eq!(report.mapped_ops_count(), 3);
    assert!(report.unmapped_ops.is_empty(), "all ops should map");
    assert_eq!(report.num_user_inputs, compiled.metadata.num_inputs);
    assert_eq!(report.num_weights_loaded, compiled.metadata.num_weights);
    assert_eq!(report.total_ops_imported, compiled.metadata.num_ops);
    assert_eq!(report.dispatch_count, compiled.model.num_dispatches());
    assert_eq!(report.total_steps, compiled.model.num_steps());
    assert_eq!(
        report.metal_dispatches,
        compiled.model.num_metal_dispatches()
    );
    assert!(
        report.dispatch_count_before_fusion >= report.dispatch_count,
        "fusion should not increase dispatch count"
    );
    assert!(
        report.estimated_rtf.is_some(),
        "report should retain the estimated RTF"
    );

    let mapped_names: Vec<&str> = report
        .mapped_ops
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(
        mapped_names.contains(&"torch.ops.aten.linear.default"),
        "report should retain linear mapping stats: {mapped_names:?}"
    );
    assert!(
        mapped_names.contains(&"torch.ops.aten.relu.default"),
        "report should retain relu mapping stats: {mapped_names:?}"
    );
    assert!(
        report.verification.reference_parity_passed.is_none(),
        "reference parity should remain unset because this helper accepts no reference trace"
    );
}

#[test]
fn compile_exported_artifacts_matches_explicit_report_helper() {
    let Some(cache) = metal_cache() else {
        return;
    };

    let (_dir, graph_path, weights_path) = write_mlp_export_fixture();
    let config = ConvertConfig::new("test-mlp");

    let preferred = compile_exported_artifacts(&graph_path, &weights_path, &config, &cache)
        .expect("compile exported artifacts through preferred one-call helper");
    let explicit = compile_metal_from_exported_artifacts_with_report(
        &graph_path,
        &weights_path,
        &config,
        &cache,
    )
    .expect("compile exported artifacts through explicit report helper");

    assert_eq!(
        preferred.metadata, explicit.metadata,
        "preferred helper should preserve the same metadata"
    );
    assert_eq!(
        serde_json::to_value(&preferred.report).expect("serialize preferred report"),
        serde_json::to_value(&explicit.report).expect("serialize explicit report"),
        "preferred helper should preserve the same structured report"
    );
    assert_eq!(
        preferred.model.num_steps(),
        explicit.model.num_steps(),
        "preferred helper should preserve compiled step count"
    );
    assert_eq!(
        preferred.model.num_dispatches(),
        explicit.model.num_dispatches(),
        "preferred helper should preserve dispatch count"
    );
    assert_eq!(
        preferred.model.output_shape(),
        explicit.model.output_shape(),
        "preferred helper should preserve output shape"
    );
    assert_eq!(
        preferred.model.output_dtype(),
        explicit.model.output_dtype(),
        "preferred helper should preserve output dtype"
    );
}

#[test]
fn compile_metal_from_exported_artifacts_with_report_matches_manual_path_with_extra_unused_weights()
{
    let Some(cache) = metal_cache() else {
        return;
    };

    let (_dir, graph_path, weights_path) = write_mlp_export_fixture_with_extra_unused_tensor();
    let config = ConvertConfig::new("test-mlp-nondefault")
        .with_validate_weights(false)
        .with_constant_fold(false);

    let helper = compile_metal_from_exported_artifacts_with_report(
        &graph_path,
        &weights_path,
        &config,
        &cache,
    )
    .expect("compile exported artifacts to metal with report");

    let converted = convert_from_trace(&graph_path, &weights_path, &config)
        .expect("convert exported artifacts for manual metal compile");
    let manual_metadata = ConvertedModelMetadata::from(&converted);
    let manual = CompiledModel::builder(&converted.graph, &cache)
        .build()
        .expect("compile manual metal model from ConvertedModel graph");

    assert_eq!(
        helper.metadata, manual_metadata,
        "helper metadata should mirror the established convert_from_trace path"
    );
    assert_eq!(
        helper.metadata.num_weights, 5,
        "metadata should retain the extra safetensors tensor like ConvertedModel does"
    );
    assert_eq!(
        helper.metadata.total_params, 72,
        "metadata parameter count should include the extra safetensors tensor"
    );
    assert_eq!(
        helper.report.num_weights_loaded, 4,
        "report should still count only graph-used weights compiled to Metal"
    );

    assert_eq!(
        helper.model.output_shape(),
        manual.output_shape(),
        "helper and manual paths should produce the same output shape"
    );
    assert_eq!(
        helper.model.output_dtype(),
        manual.output_dtype(),
        "helper and manual paths should produce the same output dtype"
    );

    let input = [2.0f32, -1.0, -0.5, -0.25];
    let input_buffer =
        f32::create_buffer(cache.context(), &input).expect("create exported-artifact input buffer");
    let output_numel: usize = helper.model.output_shape().iter().product();

    let helper_output = helper
        .model
        .execute(&cache, &[&input_buffer])
        .expect("execute helper-compiled metal model");
    let manual_output = manual
        .execute(&cache, &[&input_buffer])
        .expect("execute manually compiled metal model");

    let helper_values =
        f32::read_buffer_at_offset(&helper_output, 0, output_numel).expect("read helper output");
    let manual_values =
        f32::read_buffer_at_offset(&manual_output, 0, output_numel).expect("read manual output");

    assert_close(
        "helper vs manual execution",
        &helper_values,
        &manual_values,
        1e-5,
    );
}

#[test]
fn compile_metal_from_exported_artifacts_with_report_matches_default_builder_verification_path() {
    let Some(cache) = metal_cache() else {
        return;
    };

    let (_dir, graph_path, weights_path) = write_mlp_export_fixture();
    let config = ConvertConfig::new("test-mlp");

    let helper = compile_metal_from_exported_artifacts_with_report(
        &graph_path,
        &weights_path,
        &config,
        &cache,
    )
    .expect("compile exported artifacts through explicit helper");
    let default_builder = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .build()
        .expect("build exported artifacts with default report request");
    let none_builder = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .verify(VerifyLevel::None)
        .build()
        .expect("build exported artifacts with verification disabled");
    let helper_verification =
        serde_json::to_value(&helper.report.verification).expect("serialize helper verification");
    let default_verification = serde_json::to_value(&default_builder.report.verification)
        .expect("serialize default builder verification");
    let none_verification = serde_json::to_value(&none_builder.report.verification)
        .expect("serialize no-verify builder verification");

    assert_eq!(
        helper_verification, default_verification,
        "helper report should follow the builder's default verification/report request"
    );
    if default_verification != none_verification {
        assert_ne!(
            helper_verification, none_verification,
            "helper report should not silently fall back to VerifyLevel::None"
        );
    }
}

#[test]
fn compile_metal_from_exported_artifacts_with_report_rejects_model_type_config() {
    let Some(cache) = metal_cache() else {
        return;
    };

    let (_dir, graph_path, weights_path) = write_mlp_export_fixture();
    let config = ConvertConfig::new("test-mlp").with_model_type(DpdfModelType::LayoutLMv3);

    let err = compile_metal_from_exported_artifacts_with_report(
        &graph_path,
        &weights_path,
        &config,
        &cache,
    )
    .expect_err("model_type should be rejected on the Metal/report helper");

    match err {
        ExportedArtifactCompileError::Convert(ConvertError::WeightLoad(detail)) => {
            assert!(
                detail.contains("ConvertConfig::model_type"),
                "error should explain the rejected config knob: {detail}"
            );
            assert!(
                detail.contains("convert_from_trace"),
                "error should point callers at the accepted remap surface: {detail}"
            );
        }
        other => panic!("expected model_type contract error, got {other:?}"),
    }
}
