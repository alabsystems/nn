// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic converted model representation for `nn::convert()`.
//!
//! `ConvertedModel` is the portable output of importing a PyTorch model via
//! `torch.export` JSON + safetensors weights. It owns the computation graph
//! (as `ComputationGraph`), the weight tensors (as `DynTensor`), and metadata
//! (input/output names, parameter counts).
//!
//! # Architecture
//!
//! Due to a cyclic dependency (nn-import -> nn-metal -> nn-models), this
//! crate cannot depend on `nn-import` directly. The full import pipeline is
//! bridged through the top-level `nn` crate:
//!
//! - **`nn::convert_model::convert_from_trace()`** — full import (graph + weights)
//!   via `nn-import`. This is the preferred entry point when you want a
//!   populated `ConvertedModel`; it also shares this module's `model_type`
//!   weight-key remapping behavior on the returned weight map.
//! - **`ConvertedModel::from_imported()`** — wraps a pre-built `ImportedGraph`
//!   and weight map into a `ConvertedModel`. Called by the bridge.
//! - **`convert_from_trace()` (this module)** — weight-only stub for cases
//!   where full graph import is not needed.
//!
//! The one-function Metal/report helpers in the top-level `nn` crate are a
//! separate surface: they compile exported artifacts directly and intentionally
//! reject `ConvertConfig::model_type` rather than silently relying on this
//! module's weight-key remap semantics.
//!
//! # Usage (Recommended Path)
//!
//! ```rust,ignore
//! // Via the nn crate bridge (full graph + weights):
//! use nn::convert_model::convert_from_trace;
//! use nn_models::convert::ConvertConfig;
//!
//! let config = ConvertConfig::new("wavlm-base");
//! let model = convert_from_trace(&graph_path, &weights_path, &config)?;
//! assert!(model.num_ops() > 0); // Graph is populated
//! ```
//!
//! Or, using `nn-import` directly:
//!
//! ```rust,ignore
//! use nn_import::import_model;
//! use nn_models::convert::ConvertedModel;
//!
//! let imported = import_model(&graph_path, &weights_path)?;
//! let model = ConvertedModel::from_imported(
//!     imported.graph, imported.num_user_inputs,
//!     imported.user_input_names, imported.output_names,
//!     weights, "wavlm-base",
//! );
//! ```

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::{Device, DynTensor};

// ---------------------------------------------------------------------------
// dpdf model type + weight mapping
// ---------------------------------------------------------------------------

/// Known dpdf model architectures for weight name mapping during import.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DpdfModelType {
    /// Granite-Docling-258M: SigLIP2 vision encoder + Granite-165M decoder.
    GraniteDocling,
    /// DocLayout-YOLO: YOLOv8-nano backbone + PAN neck + detect head.
    DocLayoutYolo,
    /// Qwen3-VL: Conv3d-patched ViT + Qwen3 decoder.
    Qwen3VL,
    /// Table Transformer: ResNet-18 backbone + DETR encoder/decoder.
    TableTransformer,
    /// UniTable: linear patch projection + transformer encoder/decoder.
    UniTable,
    /// LayoutLMv3: text + layout + image transformer for forms.
    LayoutLMv3,
    /// Sprint document understanding model.
    Sprint,
    /// GLM-OCR: SigLIP2 vision + GLM decoder + MTP heads.
    GlmOcr,
    /// PaddleOCR-VL-1.5: SigLIP ViT vision encoder + ERNIE-4.5 GQA decoder.
    PaddleOcr,
    /// FireRed-OCR: Qwen3-VL-2B fine-tune with CTC head + line detector.
    FireRedOcr,
    /// RT-DETRv2 (Heron): ResNet-18 backbone + hybrid encoder + transformer decoder.
    RtDetr,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the `convert_from_trace()` exported-artifact bridge.
///
/// Today this surface reliably controls model naming and optional model-type
/// weight remapping on `ConvertedModel`-producing exported-artifact paths.
/// `validate_weights` and `constant_fold` remain part of the public config
/// shape for future expansion, but they are not yet exercised by the
/// exported-artifact path.
///
/// `model_type` is only honored on surfaces that actually return a
/// `ConvertedModel` weight map, such as this module's `convert_from_trace()`
/// and the top-level `nn::convert_model::convert_from_trace()` bridge. The
/// top-level Metal/report helpers intentionally reject `model_type` because
/// they do not consume remapped weight keys end-to-end.
///
/// # Examples
///
/// ```rust
/// use nn_models::convert::ConvertConfig;
///
/// // Minimal config with model name.
/// let config = ConvertConfig::new("wavlm-base");
///
/// // Customized config.
/// let config = ConvertConfig::new("nn-model")
///     .with_model_type(nn_models::convert::DpdfModelType::LayoutLMv3);
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ConvertConfig {
    /// Model name for diagnostics and logging.
    pub model_name: String,
    /// Reserved weight-validation knob for future exported-artifact checks.
    ///
    /// The current exported-artifact path preserves this value on the public
    /// config type but does not yet implement additional validation behavior.
    /// Defaults to `true`.
    pub validate_weights: bool,
    /// Reserved constant-folding knob for future exported-artifact lowering.
    ///
    /// The current exported-artifact path preserves this value on the public
    /// config type but does not yet run an extra constant-folding pass here.
    /// Defaults to `true`.
    pub constant_fold: bool,
    /// Optional dpdf model type for weight name mapping.
    ///
    /// When set, `convert_from_trace()` applies model-specific HuggingFace
    /// to nn weight key translation on the returned `ConvertedModel.weights`
    /// map. `None` = no remapping.
    ///
    /// This does not rewrite the exported graph itself, and the top-level
    /// Metal/report helpers reject this knob rather than compiling through a
    /// graph-only path with remapped weights that they would not consume.
    pub model_type: Option<DpdfModelType>,
}

impl ConvertConfig {
    /// Create a new config with the given model name and current defaults.
    #[must_use]
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            validate_weights: true,
            constant_fold: true,
            model_type: None,
        }
    }

    /// Set the reserved weight-validation preference.
    ///
    /// Exported-artifact conversion currently accepts but ignores this knob.
    #[must_use]
    pub fn with_validate_weights(mut self, validate: bool) -> Self {
        self.validate_weights = validate;
        self
    }

    /// Set the reserved constant-folding preference.
    ///
    /// Exported-artifact conversion currently accepts but ignores this knob.
    #[must_use]
    pub fn with_constant_fold(mut self, fold: bool) -> Self {
        self.constant_fold = fold;
        self
    }

    /// Set the dpdf model type for weight name mapping on returned
    /// `ConvertedModel` weights.
    ///
    /// This knob is for the weight-loading / `ConvertedModel` surfaces only;
    /// the top-level Metal/report helpers in `nn` reject it.
    #[must_use]
    pub fn with_model_type(mut self, model_type: DpdfModelType) -> Self {
        self.model_type = Some(model_type);
        self
    }

    /// Detect `DpdfModelType` from a HuggingFace model identifier string.
    ///
    /// Returns `None` if the identifier does not match a known dpdf model.
    #[must_use]
    pub fn detect_model_type(hf_model_id: &str) -> Option<DpdfModelType> {
        let lower = hf_model_id.to_lowercase();
        if lower.contains("granite") && lower.contains("docling") {
            Some(DpdfModelType::GraniteDocling)
        } else if lower.contains("doclayout") && lower.contains("yolo") {
            Some(DpdfModelType::DocLayoutYolo)
        } else if lower.contains("firered") && lower.contains("ocr") {
            Some(DpdfModelType::FireRedOcr)
        } else if lower.contains("qwen3") && lower.contains("vl") {
            Some(DpdfModelType::Qwen3VL)
        } else if lower.contains("table") && lower.contains("transformer") {
            Some(DpdfModelType::TableTransformer)
        } else if lower.contains("unitable") {
            Some(DpdfModelType::UniTable)
        } else if lower.contains("layoutlmv3")
            || (lower.contains("layoutlm") && lower.contains("v3"))
        {
            Some(DpdfModelType::LayoutLMv3)
        } else if lower.contains("sprint") {
            Some(DpdfModelType::Sprint)
        } else if lower.contains("glm") && lower.contains("ocr") {
            Some(DpdfModelType::GlmOcr)
        } else if lower.contains("paddle") && lower.contains("ocr") {
            Some(DpdfModelType::PaddleOcr)
        } else {
            None
        }
    }
}

impl Default for ConvertConfig {
    fn default() -> Self {
        Self::new("unnamed")
    }
}

// ---------------------------------------------------------------------------
// ConvertedModel
// ---------------------------------------------------------------------------

/// A backend-agnostic converted model ready for compilation or inspection.
///
/// This is the portable representation of an imported PyTorch model. It owns
/// the computation graph (`ComputationGraph`), weight tensors (`DynTensor`),
/// and metadata. It can be:
///
/// - **Inspected:** op counts, parameter counts, input/output shapes
/// - **Compiled:** pass `self.graph` to any backend's `CompiledModel::builder()`
/// - **Verified:** pass to `check_composition_bounds()` for NY IBP
///
/// # Examples
///
/// ```rust,ignore
/// let model = convert_from_trace(&graph_path, &weights_path, &config)?;
/// assert_eq!(model.num_inputs(), 1);
/// assert!(model.num_ops() > 0);
///
/// // Access the computation graph for compilation.
/// let graph = &model.graph;
/// ```
#[non_exhaustive]
pub struct ConvertedModel {
    /// The computation graph (sequence of `TraceOp` nodes).
    ///
    /// This is the same `ComputationGraph` type used by `CompiledModel::builder()`
    /// and `trace_to_graph_model()` for verification.
    pub graph: ComputationGraph,

    /// All model weights keyed by fully-qualified parameter name.
    ///
    /// By default these keys match the original safetensors FQNs (for example
    /// `"encoder.layers.0.weight"`), not the torch.export placeholder names.
    /// When `ConvertConfig::model_type` remapping is enabled, the returned map
    /// instead uses the remapped model-specific keys from that conversion
    /// surface. This only changes the keys exposed through this
    /// `ConvertedModel`; it does not imply that other exported-artifact
    /// surfaces, such as the top-level Metal/report helpers, accept or consume
    /// the same remapped keys. Values are CPU `DynTensor`s.
    pub weights: HashMap<String, DynTensor>,

    /// Number of runtime inputs (user inputs, not parameters/buffers).
    pub num_inputs: usize,

    /// Ordered input tensor names from the torch.export signature.
    pub input_names: Vec<String>,

    /// Ordered output tensor names from the torch.export signature.
    pub output_names: Vec<String>,

    /// Model name from config (for diagnostics).
    pub model_name: String,
}

impl ConvertedModel {
    /// Create a new `ConvertedModel` from its components.
    pub fn new(
        graph: ComputationGraph,
        weights: HashMap<String, DynTensor>,
        num_inputs: usize,
        input_names: Vec<String>,
        output_names: Vec<String>,
        model_name: String,
    ) -> Self {
        Self {
            graph,
            weights,
            num_inputs,
            input_names,
            output_names,
            model_name,
        }
    }

    /// Number of computation graph nodes (all ops including Input/Constant).
    #[must_use]
    pub fn num_ops(&self) -> usize {
        self.graph.len()
    }

    /// Number of runtime inputs.
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Number of weight tensors loaded from safetensors.
    #[must_use]
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }

    /// Total number of scalar parameters across all weight tensors.
    #[must_use]
    pub fn total_params(&self) -> usize {
        self.weights.values().map(DynTensor::elem_count).sum()
    }

    /// Create a `ConvertedModel` from a pre-built imported graph and weights.
    ///
    /// This is the preferred entry point when using `nn-import` directly.
    /// The caller builds an `ImportedGraph` via `nn_import::import_model()`
    /// and passes it here along with the loaded weights.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use nn_import::import_model;
    /// use nn_models::convert::ConvertedModel;
    ///
    /// let imported = import_model(&graph_path, &weights_path)?;
    /// let model = ConvertedModel::from_imported(imported.graph, imported.weights, "nn-model");
    /// ```
    pub fn from_imported(
        graph: ComputationGraph,
        num_user_inputs: usize,
        user_input_names: Vec<String>,
        output_names: Vec<String>,
        weights: HashMap<String, DynTensor>,
        model_name: impl Into<String>,
    ) -> Self {
        Self {
            graph,
            weights,
            num_inputs: num_user_inputs,
            input_names: user_input_names,
            output_names,
            model_name: model_name.into(),
        }
    }

    /// Compile the computation graph into a [`CompiledPlan`] with fusion.
    ///
    /// Runs the full compilation pipeline: constant folding, sequential
    /// elementwise chain fusion, partition-driven fusion, and peephole
    /// optimization passes. The result is ready to be passed to
    /// `CompiledModel::from_plan()` on any GPU backend.
    ///
    /// Returns `None` if the graph is empty (weight-only `ConvertedModel`
    /// from the stub `convert_from_trace()`). Use the `nn` crate bridge
    /// `nn::convert_model::convert_from_trace()` to get a populated graph.
    ///
    /// # Errors
    ///
    /// Returns `ConvertError::GraphParse` if graph compilation fails
    /// (unsupported op, topology error, etc.).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let model = nn::convert_model::convert_from_trace(&graph, &weights, &config)?;
    /// let plan = model.compile_graph()?;
    /// if let Some(plan) = plan {
    ///     let compiled = CompiledModel::from_plan(&plan, &model.graph, &cache)?;
    /// }
    /// ```
    pub fn compile_graph(
        &self,
    ) -> Result<Option<nn_dsl::trace_compile::CompiledPlan>, ConvertError> {
        if self.graph.is_empty() {
            return Ok(None);
        }
        let plan = nn_dsl::compile_trace_to_plan_with_fusion(&self.graph)
            .map_err(|e| ConvertError::GraphParse(format!("graph compilation failed: {e}")))?;
        Ok(Some(plan))
    }

    /// Compile the computation graph with custom peephole configuration.
    ///
    /// Same as [`compile_graph()`](Self::compile_graph) but allows disabling
    /// individual peephole passes via [`PeepholeConfig`](nn_dsl::PeepholeConfig).
    ///
    /// # Errors
    ///
    /// Returns `ConvertError::GraphParse` if graph compilation fails.
    pub fn compile_graph_configured(
        &self,
        peephole_config: &nn_dsl::PeepholeConfig,
    ) -> Result<Option<nn_dsl::trace_compile::CompiledPlan>, ConvertError> {
        if self.graph.is_empty() {
            return Ok(None);
        }
        let plan = nn_dsl::compile_trace_to_plan_configured(&self.graph, peephole_config)
            .map_err(|e| ConvertError::GraphParse(format!("graph compilation failed: {e}")))?;
        Ok(Some(plan))
    }

    /// Get a weight tensor by its fully-qualified name.
    #[must_use]
    pub fn weight(&self, name: &str) -> Option<&DynTensor> {
        self.weights.get(name)
    }

    /// Ordered input names.
    #[must_use]
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Ordered output names.
    #[must_use]
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

impl std::fmt::Debug for ConvertedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvertedModel")
            .field("model_name", &self.model_name)
            .field("num_ops", &self.num_ops())
            .field("num_inputs", &self.num_inputs)
            .field("num_weights", &self.num_weights())
            .field("total_params", &self.total_params())
            .field("input_names", &self.input_names)
            .field("output_names", &self.output_names)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error from the backend-agnostic convert pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConvertError {
    /// File I/O error.
    #[error("I/O error reading '{path}': {detail}")]
    Io { path: String, detail: String },

    /// JSON deserialization of the torch.export graph failed.
    #[error("graph parse error: {0}")]
    GraphParse(String),

    /// Safetensors weight loading failed.
    #[error("weight load error: {0}")]
    WeightLoad(String),

    /// Weight shape validation failed.
    #[error("weight '{name}' shape mismatch: expected {expected} elements, got {actual}")]
    WeightShapeMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    /// Tensor creation error from nn-core.
    #[error("tensor error: {0}")]
    Tensor(#[from] nn_core::TensorError),
}

// ---------------------------------------------------------------------------
// convert_from_trace()
// ---------------------------------------------------------------------------

/// Load safetensors weights into a `ConvertedModel`.
///
/// **Weight-only.** This function loads and remaps safetensors weights but
/// produces an empty `ComputationGraph`. The `nn-models` crate cannot depend
/// on `nn-import` (cyclic: nn-import -> nn-metal -> nn-models), so graph
/// translation is not available here.
///
/// For **full import** (graph + weights), use the bridge in the top-level `nn`
/// crate: [`nn::convert_model::convert_from_trace()`], which calls
/// `nn_import::import_model()` and wraps the result via
/// [`ConvertedModel::from_imported()`].
///
/// # Arguments
///
/// - `trace_path`: Path to the torch.export graph JSON (validated but not translated)
/// - `weights_path`: Path to the safetensors weight file
/// - `config`: Import configuration. This weight-only path currently honors
///   `model_name` and optional `model_type` remapping on the returned weight
///   map; `validate_weights` and `constant_fold` are accepted but not applied.
///   The top-level Metal/report helpers are a different surface and reject
///   `model_type`.
///
/// # Errors
///
/// Returns `ConvertError` if weight loading or JSON validation fails.
///
/// # Examples
///
/// ```rust,ignore
/// // For full graph import, prefer the nn crate bridge:
/// // use nn::convert_model::convert_from_trace;
///
/// // This function is for weight-only loading:
/// use nn_models::convert::{ConvertConfig, convert_from_trace};
///
/// let config = ConvertConfig::new("wavlm-base");
/// let model = convert_from_trace(
///     Path::new("exports/wavlm/graph.json"),
///     Path::new("exports/wavlm/weights.safetensors"),
///     &config,
/// )?;
/// assert_eq!(model.num_ops(), 0); // Graph empty — use nn::convert_model for full import
/// assert!(model.num_weights() > 0); // Weights loaded
/// ```
pub fn convert_from_trace(
    trace_path: &Path,
    weights_path: &Path,
    config: &ConvertConfig,
) -> Result<ConvertedModel, ConvertError> {
    // Phase 1: Read and validate the torch.export graph JSON.
    let json_bytes = std::fs::read(trace_path).map_err(|e| ConvertError::Io {
        path: trace_path.display().to_string(),
        detail: e.to_string(),
    })?;
    let _value: serde_json::Value =
        serde_json::from_slice(&json_bytes).map_err(|e| ConvertError::GraphParse(e.to_string()))?;

    // Phase 2: Load safetensors weights, with optional model-type key remap.
    let weights = load_weights(weights_path, config)?;

    // Phase 3: Return weight-only model with empty graph.
    // For full graph import, callers should use nn::convert_model::convert_from_trace()
    // which bridges nn-import and nn-models.
    Ok(ConvertedModel::new(
        ComputationGraph::from_nodes(vec![]),
        weights,
        0,
        Vec::new(),
        Vec::new(),
        config.model_name.clone(),
    ))
}

/// Load safetensors weights as `DynTensor` on CPU with optional dpdf remapping.
///
/// Any `ConvertConfig::model_type` remap changes only the keys in the returned
/// weight map.
fn load_weights(
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
        remap_weight_keys(model_type, weights)
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
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        Dtype::BF16 => Ok(raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        Dtype::F64 => Ok(raw
            .chunks_exact(8)
            .map(|c| {
                let bytes: [u8; 8] = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                f64::from_le_bytes(bytes) as f32
            })
            .collect()),
        Dtype::I64 => Ok(raw
            .chunks_exact(8)
            .map(|c| {
                let bytes: [u8; 8] = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                i64::from_le_bytes(bytes) as f32
            })
            .collect()),
        Dtype::U8 => Ok(raw.iter().map(|&b| f32::from(b)).collect()),
        Dtype::I8 => Ok(raw.iter().map(|&b| f32::from(b as i8)).collect()),
        other => Err(ConvertError::WeightLoad(format!(
            "unsupported dtype {other:?} for weight '{name}'"
        ))),
    }
}

// ---------------------------------------------------------------------------
// dpdf weight name mapping (extracted to convert_dpdf.rs for 500-line rule)
// ---------------------------------------------------------------------------

#[path = "convert_dpdf.rs"]
mod convert_dpdf;
pub use convert_dpdf::map_weight_key;
pub use convert_dpdf::remap_weight_keys;

// ---------------------------------------------------------------------------
// Parity verification (Phase 1: L0 structure + L3 numerical)
// ---------------------------------------------------------------------------

/// Structured parity diagnostic report from `verify_parity()`.
#[derive(Debug)]
#[non_exhaustive]
pub struct ParityReport {
    /// Model name for diagnostics.
    pub model_name: String,
    /// Whether all checks passed.
    pub overall_pass: bool,
    /// Individual check results.
    pub checks: Vec<ParityCheck>,
}

impl ParityReport {
    /// Create a new `ParityReport` with no checks yet.
    fn new(model_name: String) -> Self {
        Self {
            model_name,
            overall_pass: true,
            checks: Vec::new(),
        }
    }

    /// Add a check and update `overall_pass`.
    fn push(&mut self, check: ParityCheck) {
        if !check.passed {
            self.overall_pass = false;
        }
        self.checks.push(check);
    }
}

/// A single parity check result.
#[derive(Debug)]
#[non_exhaustive]
pub struct ParityCheck {
    /// Human-readable check name.
    pub name: String,
    /// Equivalence proof level this check belongs to.
    pub level: ParityLevel,
    /// Whether the check passed.
    pub passed: bool,
    /// Numerical metrics (L3 checks only).
    pub metric: Option<ParityMetric>,
    /// Error message if failed, or additional context.
    pub error: Option<String>,
}

impl ParityCheck {
    /// Create a passing structural check.
    fn structure_pass(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            level: ParityLevel::Structure,
            passed: true,
            metric: None,
            error: None,
        }
    }

    /// Create a failing structural check.
    fn structure_fail(name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            level: ParityLevel::Structure,
            passed: false,
            metric: None,
            error: Some(error.into()),
        }
    }
}

/// Which level of the equivalence proof this check belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityLevel {
    /// L0: graph structure (non-empty, inputs/outputs present).
    Structure,
    /// L1: kernel safety (Kani harnesses).
    KernelSafety,
    /// L2: bounds propagation (NY).
    Bounds,
    /// L3: numerical parity against reference tensors.
    NumericalParity,
}

/// Numerical metrics from a parity comparison.
#[derive(Debug, Clone)]
pub struct ParityMetric {
    /// Cosine similarity between reference and actual (1.0 = identical direction).
    pub cosine_similarity: f64,
    /// Maximum absolute element-wise difference.
    pub max_abs_diff: f64,
    /// Root mean square of element-wise differences.
    pub rms_diff: f64,
    /// Total number of elements compared.
    pub element_count: usize,
}

/// Configurable thresholds for parity checks.
#[derive(Debug, Clone)]
pub struct ParityThresholds {
    /// Minimum cosine similarity (default: 0.999).
    pub cosine_min: f64,
    /// Maximum absolute difference (default: 0.02).
    pub max_abs_max: f64,
    /// Maximum RMS difference (default: 0.001).
    pub rms_max: f64,
    /// Maximum bounds width ratio (default: 1.2).
    pub bounds_width_ratio: f64,
}

impl Default for ParityThresholds {
    fn default() -> Self {
        Self {
            cosine_min: 0.999,
            max_abs_max: 0.02,
            rms_max: 0.001,
            bounds_width_ratio: 1.2,
        }
    }
}

impl ConvertedModel {
    /// Run parity verification and return a structured diagnostic report.
    ///
    /// Phase 1 checks:
    /// - **L0 (Structure):** graph has >0 ops, num_inputs > 0, output_names non-empty
    /// - **L3 (Numerical):** if `reference_outputs` is provided, computes cosine similarity,
    ///   max absolute difference, and RMS difference against each output tensor
    ///
    /// # Arguments
    ///
    /// - `reference_outputs`: Optional map from output name to reference f32 data.
    ///   When provided, L3 numerical parity checks are run.
    /// - `actual_outputs`: Optional map from output name to actual f32 data.
    ///   Required when `reference_outputs` is provided.
    /// - `thresholds`: Parity thresholds. Pass `None` to use defaults.
    #[must_use]
    pub fn verify_parity(
        &self,
        reference_outputs: Option<&HashMap<String, Vec<f32>>>,
        actual_outputs: Option<&HashMap<String, Vec<f32>>>,
        thresholds: Option<&ParityThresholds>,
    ) -> ParityReport {
        let defaults = ParityThresholds::default();
        let thresholds = thresholds.unwrap_or(&defaults);
        let mut report = ParityReport::new(self.model_name.clone());

        // L0: graph has operations
        if self.graph.is_empty() {
            report.push(ParityCheck::structure_fail(
                "graph_non_empty",
                "computation graph has 0 ops",
            ));
        } else {
            report.push(ParityCheck::structure_pass("graph_non_empty"));
        }

        // L0: at least one input
        if self.num_inputs == 0 {
            report.push(ParityCheck::structure_fail(
                "has_inputs",
                "model has 0 runtime inputs",
            ));
        } else {
            report.push(ParityCheck::structure_pass("has_inputs"));
        }

        // L0: output names present
        if self.output_names.is_empty() {
            report.push(ParityCheck::structure_fail(
                "has_outputs",
                "output_names is empty",
            ));
        } else {
            report.push(ParityCheck::structure_pass("has_outputs"));
        }

        // L3: numerical parity (only when both reference and actual are provided)
        if let (Some(refs), Some(actuals)) = (reference_outputs, actual_outputs) {
            for name in &self.output_names {
                let ref_data = refs.get(name);
                let actual_data = actuals.get(name);

                match (ref_data, actual_data) {
                    (Some(r), Some(a)) => {
                        if r.len() != a.len() {
                            report.push(ParityCheck {
                                name: format!("numerical_{name}"),
                                level: ParityLevel::NumericalParity,
                                passed: false,
                                metric: None,
                                error: Some(format!(
                                    "length mismatch: reference={}, actual={}",
                                    r.len(),
                                    a.len()
                                )),
                            });
                            continue;
                        }
                        if r.is_empty() {
                            report.push(ParityCheck {
                                name: format!("numerical_{name}"),
                                level: ParityLevel::NumericalParity,
                                passed: true,
                                metric: Some(ParityMetric {
                                    cosine_similarity: 1.0,
                                    max_abs_diff: 0.0,
                                    rms_diff: 0.0,
                                    element_count: 0,
                                }),
                                error: None,
                            });
                            continue;
                        }

                        let cos = cosine_similarity(r, a);
                        let max_abs = max_abs_diff(r, a);
                        let rms = rms_diff(r, a);

                        let passed = cos >= thresholds.cosine_min
                            && max_abs <= thresholds.max_abs_max
                            && rms <= thresholds.rms_max;

                        let error = if !passed {
                            Some(format!(
                                "cosine={cos:.6} (min {}), max_abs={max_abs:.6} (max {}), rms={rms:.6} (max {})",
                                thresholds.cosine_min, thresholds.max_abs_max, thresholds.rms_max,
                            ))
                        } else {
                            None
                        };

                        report.push(ParityCheck {
                            name: format!("numerical_{name}"),
                            level: ParityLevel::NumericalParity,
                            passed,
                            metric: Some(ParityMetric {
                                cosine_similarity: cos,
                                max_abs_diff: max_abs,
                                rms_diff: rms,
                                element_count: r.len(),
                            }),
                            error,
                        });
                    }
                    (None, _) => {
                        report.push(ParityCheck {
                            name: format!("numerical_{name}"),
                            level: ParityLevel::NumericalParity,
                            passed: false,
                            metric: None,
                            error: Some(format!("missing reference data for output '{name}'")),
                        });
                    }
                    (_, None) => {
                        report.push(ParityCheck {
                            name: format!("numerical_{name}"),
                            level: ParityLevel::NumericalParity,
                            passed: false,
                            metric: None,
                            error: Some(format!("missing actual data for output '{name}'")),
                        });
                    }
                }
            }
        }

        report
    }
}

/// Cosine similarity between two f32 slices.
///
/// Returns 1.0 for identical direction, 0.0 for orthogonal, -1.0 for opposite.
/// Returns 0.0 if either vector has zero magnitude (avoids division by zero).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    dot / denom
}

/// Maximum absolute element-wise difference between two f32 slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0_f64, f64::max)
}

/// Root mean square of element-wise differences between two f32 slices.
fn rms_diff(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum();
    (sum_sq / a.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_config_defaults() {
        let config = ConvertConfig::new("test-model");
        assert_eq!(config.model_name, "test-model");
        assert!(config.validate_weights);
        assert!(config.constant_fold);
    }

    #[test]
    fn test_convert_config_builder() {
        let config = ConvertConfig::new("custom")
            .with_validate_weights(false)
            .with_constant_fold(false);
        assert_eq!(config.model_name, "custom");
        assert!(!config.validate_weights);
        assert!(!config.constant_fold);
    }

    #[test]
    fn test_convert_config_model_type_builder() {
        let config = ConvertConfig::new("layout-model").with_model_type(DpdfModelType::LayoutLMv3);
        assert_eq!(config.model_name, "layout-model");
        assert_eq!(config.model_type, Some(DpdfModelType::LayoutLMv3));
        assert!(config.validate_weights);
        assert!(config.constant_fold);
    }

    #[test]
    fn test_convert_config_default_trait() {
        let config = ConvertConfig::default();
        assert_eq!(config.model_name, "unnamed");
    }

    #[test]
    fn test_converted_model_accessors() {
        let model = ConvertedModel::new(
            ComputationGraph::from_nodes(vec![]),
            HashMap::new(),
            2,
            vec!["audio".to_string(), "mask".to_string()],
            vec!["output".to_string()],
            "test".to_string(),
        );
        assert_eq!(model.num_ops(), 0);
        assert_eq!(model.num_inputs(), 2);
        assert_eq!(model.num_weights(), 0);
        assert_eq!(model.total_params(), 0);
        assert!(model.weight("nonexistent").is_none());
        assert_eq!(model.input_names(), &["audio", "mask"]);
        assert_eq!(model.output_names(), &["output"]);
    }

    #[test]
    fn test_converted_model_with_weights() {
        let mut weights = HashMap::new();
        let w =
            DynTensor::from_vec(vec![1.0_f32; 12], &[3, 4], &Device::Cpu).expect("tensor creation");
        weights.insert("layer.weight".to_string(), w);

        let model = ConvertedModel::new(
            ComputationGraph::from_nodes(vec![]),
            weights,
            1,
            vec!["input".to_string()],
            vec!["output".to_string()],
            "test".to_string(),
        );
        assert_eq!(model.num_weights(), 1);
        assert_eq!(model.total_params(), 12);
        assert!(model.weight("layer.weight").is_some());
    }

    #[test]
    fn test_converted_model_debug() {
        let model = ConvertedModel::new(
            ComputationGraph::from_nodes(vec![]),
            HashMap::new(),
            1,
            vec!["x".to_string()],
            vec!["y".to_string()],
            "debug-test".to_string(),
        );
        let debug = format!("{model:?}");
        assert!(debug.contains("debug-test"));
        assert!(debug.contains("num_ops"));
    }

    #[test]
    fn test_convert_error_display() {
        let err = ConvertError::Io {
            path: "/tmp/test".to_string(),
            detail: "not found".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("/tmp/test"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_convert_from_trace_missing_file() {
        let config = ConvertConfig::new("test");
        let result = convert_from_trace(
            Path::new("/nonexistent/graph.json"),
            Path::new("/nonexistent/weights.safetensors"),
            &config,
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ConvertError::Io { .. }));
    }

    #[test]
    fn test_from_imported() {
        let graph = ComputationGraph::from_nodes(vec![]);
        let weights = HashMap::new();
        let model = ConvertedModel::from_imported(
            graph,
            2,
            vec!["x".to_string(), "y".to_string()],
            vec!["out".to_string()],
            weights,
            "test-imported",
        );
        assert_eq!(model.num_inputs(), 2);
        assert_eq!(model.input_names(), &["x", "y"]);
        assert_eq!(model.output_names(), &["out"]);
        assert_eq!(model.model_name, "test-imported");
    }

    #[test]
    fn test_compile_graph_empty_returns_none() {
        let model = ConvertedModel::new(
            ComputationGraph::from_nodes(vec![]),
            HashMap::new(),
            0,
            vec![],
            vec![],
            "empty".to_string(),
        );
        let result = model
            .compile_graph()
            .expect("compile_graph should not error");
        assert!(result.is_none(), "empty graph should return None");
    }

    #[test]
    fn test_compile_graph_with_traced_ops() {
        use nn_core::dyn_tensor::trace::trace_graph;

        // Trace a simple add-then-relu graph to get a real ComputationGraph.
        let (_, graph) = trace_graph(|| {
            let a = DynTensor::zeros(&[2, 4], nn_core::DType::F32, &Device::Cpu)?;
            let b = DynTensor::ones(&[2, 4], nn_core::DType::F32, &Device::Cpu)?;
            let c = a.add(&b)?;
            let d = c.relu()?;
            Ok(d)
        })
        .expect("trace_graph should succeed");

        assert!(!graph.is_empty(), "traced graph should have nodes");

        // Wrap in ConvertedModel (Path B pattern).
        let model = ConvertedModel::from_imported(
            graph,
            2,
            vec!["a".to_string(), "b".to_string()],
            vec!["out".to_string()],
            HashMap::new(),
            "test-compile",
        );

        let plan = model
            .compile_graph()
            .expect("compile_graph should succeed")
            .expect("non-empty graph should produce Some plan");

        assert!(!plan.steps.is_empty(), "compiled plan should have steps");
        // Note: input_shapes may be empty when all inputs are traced as
        // Constant nodes (DynTensor::zeros/ones). The plan is still valid.
    }

    /// Path A vs Path B compilation parity:
    /// The same ComputationGraph compiled directly via nn-dsl vs through
    /// ConvertedModel::compile_graph() must produce identical CompiledPlans.
    #[test]
    fn test_compile_graph_parity_with_direct_compilation() {
        use nn_core::dyn_tensor::trace::trace_graph;

        // Trace a more complex graph: matmul + bias add + relu.
        let (_, graph) = trace_graph(|| {
            let x = DynTensor::zeros(&[1, 4], nn_core::DType::F32, &Device::Cpu)?;
            let w = DynTensor::ones(&[4, 8], nn_core::DType::F32, &Device::Cpu)?;
            let b = DynTensor::zeros(&[1, 8], nn_core::DType::F32, &Device::Cpu)?;
            let mm = x.matmul(&w)?;
            let biased = mm.add(&b)?;
            let out = biased.relu()?;
            Ok(out)
        })
        .expect("trace_graph should succeed");

        // Path A: compile directly via nn-dsl.
        let plan_a = nn_dsl::compile_trace_to_plan_with_fusion(&graph)
            .expect("direct compilation should succeed");

        // Path B: compile through ConvertedModel.
        let model = ConvertedModel::from_imported(
            graph,
            1,
            vec!["x".to_string()],
            vec!["out".to_string()],
            HashMap::new(),
            "parity-test",
        );
        let plan_b = model
            .compile_graph()
            .expect("compile_graph should succeed")
            .expect("non-empty graph should produce Some plan");

        // The two plans must be structurally identical.
        assert_eq!(
            plan_a.steps.len(),
            plan_b.steps.len(),
            "Path A ({}) and Path B ({}) step counts differ",
            plan_a.steps.len(),
            plan_b.steps.len(),
        );
        assert_eq!(
            plan_a.input_shapes, plan_b.input_shapes,
            "input shapes differ between paths"
        );
        assert_eq!(
            plan_a.output_step, plan_b.output_step,
            "output step index differs between paths"
        );
        assert_eq!(
            plan_a.weight_names, plan_b.weight_names,
            "weight names differ between paths"
        );
    }

    #[test]
    fn test_compile_graph_configured_disables_passes() {
        use nn_core::dyn_tensor::trace::trace_graph;

        let (_, graph) = trace_graph(|| {
            let a = DynTensor::zeros(&[2, 4], nn_core::DType::F32, &Device::Cpu)?;
            let b = DynTensor::ones(&[2, 4], nn_core::DType::F32, &Device::Cpu)?;
            let c = a.add(&b)?;
            Ok(c)
        })
        .expect("trace_graph");

        let model = ConvertedModel::from_imported(
            graph,
            2,
            vec!["a".to_string(), "b".to_string()],
            vec!["out".to_string()],
            HashMap::new(),
            "config-test",
        );

        // Compile with all peephole passes disabled.
        let config = nn_dsl::PeepholeConfig {
            norm_activ_conv1d: false,
            fused_resblock: false,
            linear_activation: false,
            ..Default::default()
        };

        let plan = model
            .compile_graph_configured(&config)
            .expect("compile_graph_configured should succeed")
            .expect("non-empty graph should produce Some plan");

        assert!(!plan.steps.is_empty(), "plan should have steps");
    }
}

#[cfg(test)]
#[path = "convert_dpdf_tests.rs"]
mod convert_dpdf_tests;

#[cfg(test)]
#[path = "convert_parity_tests.rs"]
mod parity_tests;
