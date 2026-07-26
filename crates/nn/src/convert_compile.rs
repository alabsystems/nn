// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exported-artifact bridge from `torch.export` traces to a compiled Metal model.
//!
//! This module intentionally stays honest about scope: it consumes exported
//! artifacts (`torch.export` graph JSON + `safetensors` weights) and compiles
//! them for Metal. It is not a raw PyTorch or ONNX intake API.

use std::path::Path;

use crate::convert_model::{ConvertConfig, ConvertError, ConvertedModel};

fn unsupported_model_type_for_metal_helpers() -> ExportedArtifactCompileError {
    ConvertError::WeightLoad(
        "ConvertConfig::model_type is not supported by the exported-artifact Metal/report helpers; \
         use convert_from_trace() if you need a ConvertedModel with remapped weight keys, or omit \
         model_type for Metal/report compilation"
            .into(),
    )
    .into()
}

/// Retained metadata from a [`ConvertedModel`] after Metal compilation.
///
/// This keeps the user-visible import details around without retaining the full
/// CPU-side converted model and weight map in the returned helper result.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConvertedModelMetadata {
    /// Model name from the convert configuration.
    pub model_name: String,
    /// Number of graph nodes in the imported computation graph.
    pub num_ops: usize,
    /// Number of runtime inputs.
    pub num_inputs: usize,
    /// Number of weight tensors retained by `convert_from_trace()`.
    ///
    /// This mirrors the established converted-model surface and may include
    /// extra tensors present in the `safetensors` file even if the imported
    /// graph does not reference them.
    pub num_weights: usize,
    /// Total scalar parameter count across all retained weights.
    ///
    /// This mirrors the established converted-model surface and may therefore
    /// exceed the count of weights actually referenced by the imported graph.
    pub total_params: usize,
    /// Ordered user input names from the exported program signature.
    pub input_names: Vec<String>,
    /// Ordered output names from the exported program signature.
    pub output_names: Vec<String>,
}

impl From<&ConvertedModel> for ConvertedModelMetadata {
    fn from(model: &ConvertedModel) -> Self {
        Self {
            model_name: model.model_name.clone(),
            num_ops: model.num_ops(),
            num_inputs: model.num_inputs(),
            num_weights: model.num_weights(),
            total_params: model.total_params(),
            input_names: model.input_names().to_vec(),
            output_names: model.output_names().to_vec(),
        }
    }
}

/// Compiled Metal model plus retained converted-model metadata.
#[non_exhaustive]
pub struct ExportedArtifactMetalModel {
    /// Compiled Metal execution plan built from the imported graph.
    pub model: crate::metal::CompiledModel,
    /// Retained metadata from the intermediate converted-model representation.
    pub metadata: ConvertedModelMetadata,
}

impl std::fmt::Debug for ExportedArtifactMetalModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportedArtifactMetalModel")
            .field("num_steps", &self.model.num_steps())
            .field("num_dispatches", &self.model.num_dispatches())
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Error from [`compile_metal_from_exported_artifacts()`].
#[derive(Debug)]
#[non_exhaustive]
pub enum ExportedArtifactCompileError {
    /// Exported-artifact import/conversion failed before Metal compilation.
    Convert(ConvertError),
    /// Metal compilation of the imported graph failed.
    MetalCompile(crate::TensorError),
}

impl std::fmt::Display for ExportedArtifactCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Convert(err) => write!(f, "{err}"),
            Self::MetalCompile(err) => write!(f, "metal compilation failed: {err}"),
        }
    }
}

impl std::error::Error for ExportedArtifactCompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Convert(err) => Some(err),
            Self::MetalCompile(err) => Some(err),
        }
    }
}

impl From<ConvertError> for ExportedArtifactCompileError {
    fn from(err: ConvertError) -> Self {
        Self::Convert(err)
    }
}

impl From<crate::TensorError> for ExportedArtifactCompileError {
    fn from(err: crate::TensorError) -> Self {
        Self::MetalCompile(err)
    }
}

#[derive(Debug)]
struct PreparedExportedArtifacts {
    converted: ConvertedModel,
    metadata: ConvertedModelMetadata,
}

fn map_import_error(err: nn_import::ImportError) -> ConvertError {
    match err {
        nn_import::ImportError::Io { path, detail } => ConvertError::Io { path, detail },
        other => ConvertError::GraphParse(other.to_string()),
    }
}

fn prepare_exported_artifacts_for_metal_compile(
    trace_path: &Path,
    weights_path: &Path,
    config: &ConvertConfig,
) -> Result<PreparedExportedArtifacts, ExportedArtifactCompileError> {
    if config.model_type.is_some() {
        return Err(unsupported_model_type_for_metal_helpers());
    }

    let converted = crate::convert_from_trace(trace_path, weights_path, config)?;
    let metadata = ConvertedModelMetadata::from(&converted);
    Ok(PreparedExportedArtifacts {
        converted,
        metadata,
    })
}

fn map_reported_build_error(err: nn_import::ConvertError) -> ExportedArtifactCompileError {
    match err {
        nn_import::ConvertError::Import(err) => map_import_error(err).into(),
        nn_import::ConvertError::Compile(message) => crate::TensorError::backend_failure(
            crate::BackendDomain::Metal,
            crate::BackendErrorKind::KernelCompile,
            message,
        )
        .into(),
        // This helper does not accept a reference trace, so reftest failures
        // are unexpected and treated as a convert-surface contract error.
        nn_import::ConvertError::Reftest(message) => {
            ConvertError::GraphParse(format!("unexpected reftest failure: {message}")).into()
        }
        other => {
            ConvertError::GraphParse(format!("unexpected convert builder failure: {other}")).into()
        }
    }
}

/// Compile exported `torch.export` artifacts into a Metal model and keep the
/// converted-model metadata.
///
/// This helper is intentionally limited to exported artifacts: a graph JSON
/// produced by `torch.export` tooling plus `safetensors` weights. It does not
/// ingest raw PyTorch modules or ONNX files.
///
/// This helper shares the exported-artifact import contract with
/// [`crate::convert_from_trace()`] only for the supported config subset:
/// `model_name` is retained in metadata, while `validate_weights` and
/// `constant_fold` are still accepted but not applied. Unlike
/// [`crate::convert_from_trace()`], this Metal/report path rejects
/// `ConvertConfig::model_type` and returns a convert error before touching
/// artifacts or Metal, because the compile path does not apply remapped weight
/// keys end-to-end. The retained weight counts may include extra safetensors
/// tensors not referenced by the imported graph.
pub fn compile_metal_from_exported_artifacts(
    trace_path: &Path,
    weights_path: &Path,
    config: &ConvertConfig,
    cache: &crate::metal::PipelineCache,
) -> Result<ExportedArtifactMetalModel, ExportedArtifactCompileError> {
    let prepared = prepare_exported_artifacts_for_metal_compile(trace_path, weights_path, config)?;
    let model = crate::metal::CompiledModel::builder(&prepared.converted.graph, cache).build()?;
    Ok(ExportedArtifactMetalModel {
        model,
        metadata: prepared.metadata,
    })
}

/// Compiled Metal model, retained metadata, and detailed convert report.
///
/// The returned report mirrors the default [`nn_import::ConvertBuilder`]
/// import+compile path used internally by
/// [`compile_metal_from_exported_artifacts_with_report()`]. It exposes report
/// metrics and verification coverage counters, but this helper does not return
/// the underlying [`nn_import::EquivalenceProof`] object and does not accept a
/// reference trace for L3 parity.
#[non_exhaustive]
pub struct ExportedArtifactMetalModelWithReport {
    /// Compiled Metal execution plan built from the imported graph.
    pub model: crate::metal::CompiledModel,
    /// Retained metadata from the exported-artifact import.
    pub metadata: ConvertedModelMetadata,
    /// Detailed import/compile report from the exported-artifact bridge.
    ///
    /// `report.verification` reflects only the verification layers that the
    /// internal default builder run actually covered. In particular, reference
    /// parity remains `None` on this helper surface because no reference trace
    /// is accepted here.
    pub report: nn_import::ConvertReport,
}

impl std::fmt::Debug for ExportedArtifactMetalModelWithReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportedArtifactMetalModelWithReport")
            .field("num_steps", &self.model.num_steps())
            .field("num_dispatches", &self.model.num_dispatches())
            .field("metadata", &self.metadata)
            .field("report", &self.report)
            .finish()
    }
}

/// Preferred one-call exported-artifact compile helper.
///
/// This façade composes exported-artifact intake, Metal compilation, retained
/// converted-model metadata, and the machine-readable [`nn_import::ConvertReport`]
/// in a single call. It is intentionally scoped to already-exported
/// `torch.export` graph JSON plus `safetensors` weights, and does not ingest
/// raw PyTorch modules or ONNX files.
///
/// This is a thin convenience wrapper around
/// [`compile_metal_from_exported_artifacts_with_report()`]. It keeps the
/// existing explicit helper available for callers that want the longer name to
/// mirror the underlying implementation contract. The returned report follows
/// the same default [`nn_import::ConvertBuilder::build()`] verification/report
/// request as the explicit helper; it does not silently force
/// `VerifyLevel::None`.
#[doc(alias = "compile_metal_from_exported_artifacts_with_report")]
pub fn compile_exported_artifacts(
    trace_path: &Path,
    weights_path: &Path,
    config: &ConvertConfig,
    cache: &crate::metal::PipelineCache,
) -> Result<ExportedArtifactMetalModelWithReport, ExportedArtifactCompileError> {
    compile_metal_from_exported_artifacts_with_report(trace_path, weights_path, config, cache)
}

/// Compile exported `torch.export` artifacts to Metal and return the compiled
/// model, retained metadata, and detailed convert report.
///
/// This helper is intentionally limited to exported artifacts: a graph JSON
/// produced by `torch.export` tooling plus `safetensors` weights. It does not
/// ingest raw PyTorch modules or ONNX files.
///
/// Retained metadata follows [`crate::convert_from_trace()`] for the supported
/// shared config subset only: `model_name` is retained, while
/// `validate_weights` and `constant_fold` are accepted but not applied.
/// Unlike [`crate::convert_from_trace()`], this helper rejects
/// `ConvertConfig::model_type` and returns a convert error before touching
/// artifacts or Metal, because the builder/report path does not apply remapped
/// weight keys end-to-end. The returned [`nn_import::ConvertReport`] still
/// reflects graph-used weights from the Metal builder path, so
/// `report.num_weights_loaded` can be lower than `metadata.num_weights` when
/// the safetensors file contains extra unused tensors. The helper uses the
/// builder's default report and verification request, so report coverage stays
/// aligned with a plain [`nn_import::convert_build()`] call instead of forcing
/// verification off. It returns only the report, not the underlying
/// [`nn_import::EquivalenceProof`]; callers that need configurable proof
/// layers or direct proof access should use
/// [`nn_import::convert_build()`] directly.
pub fn compile_metal_from_exported_artifacts_with_report(
    trace_path: &Path,
    weights_path: &Path,
    config: &ConvertConfig,
    cache: &crate::metal::PipelineCache,
) -> Result<ExportedArtifactMetalModelWithReport, ExportedArtifactCompileError> {
    let metadata =
        prepare_exported_artifacts_for_metal_compile(trace_path, weights_path, config)?.metadata;
    let result = nn_import::convert_build(trace_path, weights_path, cache)
        .build()
        .map_err(map_reported_build_error)?;

    Ok(ExportedArtifactMetalModelWithReport {
        model: result.result.model,
        metadata,
        report: result.report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::error::Error;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn missing_artifact_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("nn_{name}_{}_{}", std::process::id(), unique))
    }

    fn assert_missing_graph_maps_to_convert_error() {
        let graph_path = missing_artifact_path("missing_exported_graph.json");
        let weights_path = missing_artifact_path("unused_exported_weights.safetensors");
        let config = ConvertConfig::new("test-mlp");

        let err = prepare_exported_artifacts_for_metal_compile(&graph_path, &weights_path, &config)
            .expect_err("missing artifacts should fail during convert");

        match &err {
            ExportedArtifactCompileError::Convert(ConvertError::Io { path, .. }) => {
                assert_eq!(path, &graph_path.display().to_string());
            }
            other => panic!("expected convert I/O error, got {other:?}"),
        }

        let display = err.to_string();
        assert!(
            display.contains(&graph_path.display().to_string()),
            "error should mention missing graph path: {display}"
        );
        assert_eq!(
            err.source()
                .expect("convert errors should preserve their source")
                .to_string(),
            display
        );
    }

    fn assert_model_type_rejected_before_touching_artifacts_or_metal() {
        let graph_path = missing_artifact_path("model_type_should_fail_before_graph_io.json");
        let weights_path = missing_artifact_path("model_type_should_fail_before_weight_io.st");
        let config = ConvertConfig::new("test-mlp")
            .with_model_type(nn_models::convert::DpdfModelType::LayoutLMv3);

        let err = prepare_exported_artifacts_for_metal_compile(&graph_path, &weights_path, &config)
            .expect_err("model_type should be rejected before artifact import");

        match &err {
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

    #[test]
    fn compile_metal_from_exported_artifacts_maps_convert_errors_before_touching_metal() {
        assert_missing_graph_maps_to_convert_error();
    }

    #[test]
    fn compile_metal_from_exported_artifacts_rejects_model_type_before_touching_artifacts_or_metal()
    {
        assert_model_type_rejected_before_touching_artifacts_or_metal();
    }

    #[test]
    fn compile_metal_from_exported_artifacts_with_report_maps_convert_errors_before_touching_metal()
    {
        assert_missing_graph_maps_to_convert_error();
    }

    #[test]
    fn compile_metal_from_exported_artifacts_with_report_rejects_model_type_before_touching_artifacts_or_metal(
    ) {
        assert_model_type_rejected_before_touching_artifacts_or_metal();
    }

    #[test]
    fn compile_exported_artifacts_maps_convert_errors_before_touching_metal() {
        assert_missing_graph_maps_to_convert_error();
    }

    #[test]
    fn compile_exported_artifacts_rejects_model_type_before_touching_artifacts_or_metal() {
        assert_model_type_rejected_before_touching_artifacts_or_metal();
    }
}
