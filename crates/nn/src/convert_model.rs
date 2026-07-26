// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Import `torch.export` JSON + `safetensors` artifacts into a
//! backend-agnostic [`ConvertedModel`].
//!
//! `nn-models` cannot depend on `nn-import` (dependency cycle via
//! nn-metal), so the weight-only [`nn_models::convert::convert_from_trace()`]
//! leaves the `ComputationGraph` empty. This module lives in the top-level
//! `nn` crate which can see both `nn-import` and `nn-models`, imports the
//! graph with `nn-import`, loads the weights, and assembles the
//! [`ConvertedModel`].
//!
//! This is the current reliable one-function intake for exported artifacts in
//! `nn`: `torch.export` graph JSON plus `safetensors` weights. Richer compile
//! reports and verification only appear on downstream, feature-gated surfaces
//! such as the `metal` and `verify` helpers; this module itself stops at a
//! populated [`ConvertedModel`].
//!
//! It is not yet the raw PyTorch/ONNX proof-powered compiler front door. Call
//! this when you already have exported artifacts and want a backend-agnostic
//! graph + weights bundle for inspection or later compilation. In particular,
//! optional `ConvertConfig::model_type` remapping is a `ConvertedModel`
//! weight-map concern on this surface; the one-function Metal/report helpers
//! intentionally reject that knob instead of silently compiling with
//! non-remapped graph weights.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn::{convert_from_trace, ConvertConfig};
//!
//! let config = ConvertConfig::new("nn-model");
//! let model = convert_from_trace(
//!     &Path::new("graph.json"),
//!     &Path::new("weights.safetensors"),
//!     &config,
//! )?;
//! assert!(model.num_ops() > 0); // Graph is populated!
//! ```

use std::collections::HashMap;
use std::path::Path;

use nn_core::{Device, DynTensor};
pub use nn_models::convert::{ConvertConfig, ConvertError, ConvertedModel};

/// Import exported `torch.export` artifacts into a populated
/// [`ConvertedModel`].
///
/// This is the current reliable one-function conversion surface for exported
/// `torch.export` JSON + `safetensors`. It intentionally stops at
/// [`ConvertedModel`]: compile-report detail and verification depth depend on
/// downstream feature-gated surfaces, and raw PyTorch/ONNX ingestion is out of
/// scope for this API today.
///
/// This function parses the graph JSON with `nn-import`, loads `safetensors`
/// weights as `DynTensor` values, applies optional dpdf weight-key remapping
/// via `config.model_type`, and assembles the result into a backend-agnostic
/// [`ConvertedModel`]. `config.validate_weights` and
/// `config.constant_fold` are currently accepted but not applied on this
/// bridge. The optional `model_type` remap affects only the returned
/// [`ConvertedModel`] weight map; the imported computation graph itself still
/// comes from the exported artifacts as-is.
///
/// It does not compile the graph, produce a `ConvertReport`, or run
/// proof/verification pipelines; those are separate surfaces. Unlike this
/// import-only bridge, the one-function Metal/report helpers reject
/// `config.model_type` rather than silently compiling through a graph-only path
/// that does not consume remapped weight keys.
///
/// # Arguments
///
/// - `trace_path` — torch.export graph JSON (produced by `nn_export.py`)
/// - `weights_path` — safetensors weight file
/// - `config` — conversion options. This bridge currently honors
///   `model_name` and optional `model_type` remapping on the returned
///   [`ConvertedModel`] weights; `validate_weights` and `constant_fold` are
///   accepted but not applied.
///
/// # Errors
///
/// Returns [`ConvertError`] on I/O failure, graph parse failure, unsupported
/// weight dtype, or tensor materialization failure while loading safetensors.
pub fn convert_from_trace(
    trace_path: &Path,
    weights_path: &Path,
    config: &ConvertConfig,
) -> Result<ConvertedModel, ConvertError> {
    // Phase 1: Parse graph JSON and build ComputationGraph via nn-import.
    let imported = nn_import::import_model(trace_path, weights_path).map_err(map_import_error)?;

    // Phase 2: Load safetensors weights as DynTensor on CPU.
    let weights = load_weights_as_dyntensor(weights_path, config)?;

    // Phase 3: Assemble into ConvertedModel.
    Ok(ConvertedModel::from_imported(
        imported.graph,
        imported.num_user_inputs,
        imported.user_input_names,
        imported.output_names,
        weights,
        config.model_name.clone(),
    ))
}

/// Load all tensors from a safetensors file as `DynTensor` on CPU.
///
/// Applies dpdf weight key remapping when `config.model_type` is set.
/// This only changes the keys exposed through the returned
/// [`ConvertedModel::weights`] map; the imported graph is built before this
/// remap is applied.
fn load_weights_as_dyntensor(
    weights_path: &Path,
    config: &ConvertConfig,
) -> Result<HashMap<String, DynTensor>, ConvertError> {
    let weights_data = std::fs::read(weights_path).map_err(|e| ConvertError::Io {
        path: weights_path.display().to_string(),
        detail: e.to_string(),
    })?;

    let safetensors = safetensors::SafeTensors::deserialize(&weights_data).map_err(|e| {
        ConvertError::WeightLoad(format!(
            "safetensors parse '{}': {e}",
            weights_path.display()
        ))
    })?;

    let mut weights = HashMap::new();
    for (name, view) in safetensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let f32_data = tensor_view_to_f32(&view, &name)?;
        let tensor =
            DynTensor::from_vec(f32_data, &shape, &Device::Cpu).map_err(ConvertError::Tensor)?;
        weights.insert(name, tensor);
    }

    // Apply dpdf model-specific weight key remapping.
    let weights = if let Some(ref model_type) = config.model_type {
        nn_models::convert::remap_weight_keys(model_type, weights)
    } else {
        weights
    };

    Ok(weights)
}

/// Convert a safetensors tensor view to f32 data.
///
/// Supports F32, F16, BF16, F64, I64, U8, I8 dtypes.
fn tensor_view_to_f32(
    view: &safetensors::tensor::TensorView<'_>,
    name: &str,
) -> Result<Vec<f32>, ConvertError> {
    use safetensors::Dtype;
    let raw = view.data();
    match view.dtype() {
        Dtype::F32 => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Dtype::F16 => Ok(raw
            .chunks_exact(2)
            .map(|c| nn_core::half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        Dtype::BF16 => Ok(raw
            .chunks_exact(2)
            .map(|c| nn_core::half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        Dtype::F64 => Ok(raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect()),
        Dtype::I64 => Ok(raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect()),
        Dtype::U8 => Ok(raw.iter().map(|&b| f32::from(b)).collect()),
        Dtype::I8 => Ok(raw.iter().map(|&b| f32::from(b as i8)).collect()),
        other => Err(ConvertError::WeightLoad(format!(
            "unsupported dtype {other:?} for weight '{name}'"
        ))),
    }
}

/// Map `nn_import::ImportError` to `nn_models::convert::ConvertError`.
fn map_import_error(e: nn_import::ImportError) -> ConvertError {
    match e {
        nn_import::ImportError::Io { path, detail } => ConvertError::Io { path, detail },
        other => ConvertError::GraphParse(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the bridge module's `convert_from_trace()` produces a
    /// `ConvertedModel` with a populated graph (not empty like the stub).
    #[test]
    fn test_convert_from_trace_populates_graph() {
        let dir =
            std::env::temp_dir().join(format!("nn_convert_model_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Write the MLP graph JSON fixture.
        let graph_path = dir.join("graph.json");
        std::fs::write(
            &graph_path,
            include_str!("../../nn-import/test_data/e2e_mlp.json"),
        )
        .unwrap();

        // Write synthetic MLP safetensors weights (fc1: 4->8, fc2: 8->3).
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
                .unwrap(),
        );
        tensors.insert(
            "fc1.bias".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &fc1_b).unwrap(),
        );
        tensors.insert(
            "fc2.weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 8], &fc2_w)
                .unwrap(),
        );
        tensors.insert(
            "fc2.bias".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3], &fc2_b).unwrap(),
        );
        let weights_path = dir.join("weights.safetensors");
        let serialized = safetensors::serialize(&tensors, None).unwrap();
        std::fs::write(&weights_path, serialized).unwrap();

        // Run the full bridge converter.
        let config = ConvertConfig::new("test-mlp");
        let model = convert_from_trace(&graph_path, &weights_path, &config)
            .expect("convert_from_trace must succeed for MLP fixture");

        let _ = std::fs::remove_dir_all(&dir);

        // The graph MUST be populated (not empty like the stub).
        assert!(
            model.num_ops() > 0,
            "graph should have ops, got {}",
            model.num_ops()
        );

        // MLP fixture: 1 input + 4 param placeholders + 3 compute ops = 8 nodes.
        assert_eq!(model.num_ops(), 8, "MLP fixture should have 8 graph nodes");

        // Input/output metadata.
        assert_eq!(model.input_names(), &["x"]);
        assert_eq!(model.output_names(), &["linear_1"]);

        // Weights loaded.
        assert!(
            model.num_weights() > 0,
            "weights should be loaded, got {}",
            model.num_weights()
        );
        assert_eq!(model.num_weights(), 4, "4 weight tensors for MLP");

        // Model name propagated.
        assert_eq!(model.model_name, "test-mlp");

        eprintln!(
            "[convert_model bridge] ops={}, weights={}, inputs={:?}, outputs={:?}",
            model.num_ops(),
            model.num_weights(),
            model.input_names(),
            model.output_names(),
        );
    }
}
