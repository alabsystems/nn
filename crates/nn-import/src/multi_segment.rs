// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-segment model import for models composed of multiple computation graphs.
//!
//! Models like Kokoro have 8 separate segments (plbert, text, prosody,
//! f0_energy, generator, regulate, sinegen_pre, sinegen_post) that are
//! exported as independent torch.export graphs sharing weights from a
//! single safetensors file.
//!
//! This module provides [`MultiSegmentModel`] and [`convert_multi_segment`]
//! for importing multiple graphs into a unified structure. When the `metal`
//! feature is enabled, [`CompiledMultiSegmentModel`] and
//! [`convert_multi_segment_to_metal`] extend that path to one-shot Metal
//! compilation for already-segmented exported-artifact bundles.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::error::ImportError;
use crate::graph_build::{build_graph, build_weight_map, ImportedGraph};
use crate::parse::{parse_exported_program, InputSpec};
use nn_core::dyn_tensor::trace::{ComputationGraph, WeightRef};
#[cfg(feature = "metal")]
use nn_core::{BackendDomain, BackendErrorKind, TensorError};
#[cfg(feature = "metal")]
use nn_dsl::trace_compile::{compile_trace_to_plan_with_fusion, CompiledStep, NativeOpKind};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A model composed of multiple named computation graph segments.
///
/// Each segment is an independently-compiled graph (e.g., encoder, decoder,
/// vocoder) that shares weights from a common safetensors file. The segments
/// execute in the order specified by `segment_order`.
///
/// # Examples
///
/// ```rust,ignore
/// use nn_import::multi_segment::convert_multi_segment;
///
/// let graphs = vec![
///     ("encoder".to_string(), serde_json::from_slice(&encoder_json)?),
///     ("decoder".to_string(), serde_json::from_slice(&decoder_json)?),
/// ];
/// let model = convert_multi_segment(&graphs, weights_path)?;
/// assert_eq!(model.segments.len(), 2);
/// assert_eq!(model.segment_order, ["encoder", "decoder"]);
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct MultiSegmentModel {
    /// Named segments, each containing a computation graph and import metadata.
    pub segments: Vec<(String, ImportedGraph)>,
    /// Execution order of segments by name.
    pub segment_order: Vec<String>,
    /// Weight tensor names that appear in more than one segment.
    pub shared_weights: Vec<String>,
}

impl MultiSegmentModel {
    /// Create a new multi-segment model.
    pub fn new(
        segments: Vec<(String, ImportedGraph)>,
        segment_order: Vec<String>,
        shared_weights: Vec<String>,
    ) -> Self {
        Self {
            segments,
            segment_order,
            shared_weights,
        }
    }

    /// Look up a segment by name.
    pub fn get_segment(&self, name: &str) -> Option<&ImportedGraph> {
        self.segments
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, g)| g)
    }

    /// Number of segments.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Iterate segment names in the declared execution order.
    ///
    /// This is a convenience for explicit caller-managed orchestration. It
    /// preserves the imported segment order but does not execute segments or
    /// route tensors across dynamic boundaries.
    pub fn ordered_segment_names(&self) -> impl Iterator<Item = &str> {
        self.segment_order.iter().map(String::as_str)
    }

    /// Iterate imported segments in the declared execution order.
    ///
    /// This provides ordered access for callers that need to drive segment
    /// execution or inspection themselves. It does not add runtime
    /// orchestration across segments.
    pub fn ordered_segments(&self) -> impl Iterator<Item = (&str, &ImportedGraph)> {
        self.segment_order.iter().map(|name| {
            let segment = self.get_segment(name).expect(
                "MultiSegmentModel invariant violated: segment_order references missing segment",
            );
            (name.as_str(), segment)
        })
    }

    /// Retrieve the computation graph for a segment by name.
    pub fn graph(&self, name: &str) -> Option<&ComputationGraph> {
        self.get_segment(name).map(|ig| &ig.graph)
    }
}

/// A Metal-compiled model composed of multiple named segments.
#[cfg(feature = "metal")]
#[non_exhaustive]
pub struct CompiledMultiSegmentModel {
    /// Named compiled segments in the same order as `segment_order`.
    pub segments: Vec<(String, nn_metal::compiled_model::CompiledModel)>,
    /// Execution order of segments by name.
    pub segment_order: Vec<String>,
    /// Weight tensor names that appear in more than one imported segment.
    ///
    /// This remains imported-artifact metadata from [`MultiSegmentModel`]. The
    /// actual GPU buffer aliasing performed by [`compile_multi_segment()`]
    /// matches identical compiled weight tensors across segments, so this list
    /// is not itself the alias map.
    pub shared_weights: Vec<String>,
}

#[cfg(feature = "metal")]
impl CompiledMultiSegmentModel {
    /// Create a new compiled multi-segment model.
    pub fn new(
        segments: Vec<(String, nn_metal::compiled_model::CompiledModel)>,
        segment_order: Vec<String>,
        shared_weights: Vec<String>,
    ) -> Self {
        Self {
            segments,
            segment_order,
            shared_weights,
        }
    }

    /// Look up a compiled segment by name.
    pub fn get_segment(&self, name: &str) -> Option<&nn_metal::compiled_model::CompiledModel> {
        self.segments
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, model)| model)
    }

    /// Number of compiled segments.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Iterate compiled segment names in the declared execution order.
    ///
    /// This is a convenience for explicit caller-managed orchestration. It
    /// preserves the compiled segment order but does not execute segments or
    /// wire outputs into later segments automatically.
    pub fn ordered_segment_names(&self) -> impl Iterator<Item = &str> {
        self.segment_order.iter().map(String::as_str)
    }

    /// Iterate compiled segments in the declared execution order.
    ///
    /// Callers can use this to drive their own per-segment execution loops.
    /// It does not provide dynamic runtime orchestration across segment
    /// boundaries such as Kokoro's `length_regulate`.
    pub fn ordered_segments(
        &self,
    ) -> impl Iterator<Item = (&str, &nn_metal::compiled_model::CompiledModel)> {
        self.segment_order.iter().map(|name| {
            let segment = self.get_segment(name).expect(
                "CompiledMultiSegmentModel invariant violated: segment_order references missing segment",
            );
            (name.as_str(), segment)
        })
    }
}

#[cfg(feature = "metal")]
impl std::fmt::Debug for CompiledMultiSegmentModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let segment_summaries: Vec<_> = self
            .segments
            .iter()
            .map(|(name, model)| {
                (
                    name,
                    model.num_steps(),
                    model.num_dispatches(),
                    model.num_inputs(),
                )
            })
            .collect();
        f.debug_struct("CompiledMultiSegmentModel")
            .field("segments", &segment_summaries)
            .field("segment_order", &self.segment_order)
            .field("shared_weights", &self.shared_weights)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to multi-segment model import.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MultiSegmentError {
    /// A segment name appears more than once.
    #[error("duplicate segment name: '{name}'")]
    DuplicateSegment { name: String },

    /// The segment order references a segment that was not provided.
    #[error("segment order references missing segment: '{name}'")]
    MissingSegment { name: String },

    /// No graphs were provided.
    #[error("at least one graph segment is required")]
    EmptyInput,

    /// Import error from a specific segment.
    #[error("import error in segment '{segment}': {source}")]
    SegmentImport {
        segment: String,
        source: Box<ImportError>,
    },

    /// File I/O error.
    #[error("I/O error reading '{path}': {detail}")]
    Io { path: String, detail: String },
}

/// Errors specific to multi-segment Metal compilation.
#[cfg(feature = "metal")]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MultiSegmentCompileError {
    /// Import failed before Metal compilation.
    #[error("{0}")]
    Import(#[from] MultiSegmentError),

    /// Metal compilation failed for a specific segment.
    #[error("metal compilation failed in segment '{segment}': {source}")]
    SegmentCompile {
        segment: String,
        source: TensorError,
    },
}

// ---------------------------------------------------------------------------
// Core import function
// ---------------------------------------------------------------------------

/// Import multiple torch.export graphs from a single weights file into a
/// [`MultiSegmentModel`].
///
/// Each entry in `graphs` is `(segment_name, parsed_json_value)`. The graphs
/// are imported in order, and the segment order is preserved from the input.
/// Shared weights are automatically detected by scanning for weight names
/// that appear in more than one segment's parameter list.
///
/// # Arguments
///
/// * `graphs` — Named graph JSON values, parsed from torch.export output.
/// * `weights_path` — Path to the shared safetensors weights file.
///
/// # Errors
///
/// Returns [`MultiSegmentError`] if:
/// - `graphs` is empty
/// - A segment name is duplicated
/// - A segment's graph JSON fails to parse or build
/// - The weights file cannot be read
pub fn convert_multi_segment(
    graphs: &[(String, serde_json::Value)],
    weights_path: &Path,
) -> Result<MultiSegmentModel, MultiSegmentError> {
    if graphs.is_empty() {
        return Err(MultiSegmentError::EmptyInput);
    }

    // Check for duplicate segment names.
    let mut seen_names: HashSet<&str> = HashSet::new();
    for (name, _) in graphs {
        if !seen_names.insert(name.as_str()) {
            return Err(MultiSegmentError::DuplicateSegment { name: name.clone() });
        }
    }

    // Load weights once (shared across all segments).
    let weight_data = crate::convert::load_safetensors_weights_pub(weights_path).map_err(|e| {
        MultiSegmentError::Io {
            path: weights_path.display().to_string(),
            detail: e.to_string(),
        }
    })?;

    // Track weight usage per segment for shared-weight detection.
    let mut weight_usage: HashMap<String, Vec<String>> = HashMap::new();

    let mut segments: Vec<(String, ImportedGraph)> = Vec::with_capacity(graphs.len());
    let mut segment_order: Vec<String> = Vec::with_capacity(graphs.len());

    for (name, json_value) in graphs {
        // Serialize the JSON value back to bytes for parsing.
        // This is slightly redundant but keeps the pipeline uniform —
        // parse_exported_program expects raw bytes.
        let json_bytes =
            serde_json::to_vec(json_value).map_err(|e| MultiSegmentError::SegmentImport {
                segment: name.clone(),
                source: Box::new(ImportError::JsonParse(e)),
            })?;

        let program =
            parse_exported_program(&json_bytes).map_err(|e| MultiSegmentError::SegmentImport {
                segment: name.clone(),
                source: Box::new(e),
            })?;

        let segment_weight_map =
            build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

        // Track which real weight/buffer names this segment uses, not the
        // placeholder tensor arg names from the exported graph.
        for spec in &program.graph_module.signature.input_specs {
            match spec {
                InputSpec::Parameter(p)
                    if weight_data.contains_key(&p.parameter.parameter_name) =>
                {
                    weight_usage
                        .entry(p.parameter.parameter_name.clone())
                        .or_default()
                        .push(name.clone());
                }
                InputSpec::Buffer(b) if weight_data.contains_key(&b.buffer.buffer_name) => {
                    weight_usage
                        .entry(b.buffer.buffer_name.clone())
                        .or_default()
                        .push(name.clone());
                }
                _ => {}
            }
        }

        let imported = build_graph(&program, &segment_weight_map).map_err(|e| {
            MultiSegmentError::SegmentImport {
                segment: name.clone(),
                source: Box::new(e),
            }
        })?;

        segment_order.push(name.clone());
        segments.push((name.clone(), imported));
    }

    // Detect shared weights: weight names used by more than one segment.
    let mut shared_weights: Vec<String> = weight_usage
        .into_iter()
        .filter(|(_, users)| users.len() > 1)
        .map(|(name, _)| name)
        .collect();
    shared_weights.sort();

    Ok(MultiSegmentModel::new(
        segments,
        segment_order,
        shared_weights,
    ))
}

/// Import a single torch.export graph as a [`MultiSegmentModel`] with one segment.
///
/// This provides backward-compatible single-segment import through the
/// multi-segment API. The segment is named `"main"`.
pub fn convert_single_segment(
    graph_json: &serde_json::Value,
    weights_path: &Path,
) -> Result<MultiSegmentModel, MultiSegmentError> {
    convert_multi_segment(&[("main".to_string(), graph_json.clone())], weights_path)
}

/// Compile an already-imported multi-segment model into Metal segments.
///
/// This preserves segment order and imported shared-weight metadata while
/// compiling each segment's `ImportedGraph` independently. When later segments
/// capture weight tensors with the same concrete shape and f32 payload as ones
/// already compiled, the Metal build reuses those GPU buffers through
/// `CompiledModel::builder(...).shared_weights(...)` instead of uploading
/// duplicate copies.
#[cfg(feature = "metal")]
pub fn compile_multi_segment(
    model: &MultiSegmentModel,
    cache: &nn_metal::PipelineCache,
) -> Result<CompiledMultiSegmentModel, MultiSegmentCompileError> {
    let mut shared_weight_store: HashMap<OwnedWeightKey, nn_metal::MetalBuffer> = HashMap::new();
    let mut compiled_segments = Vec::with_capacity(model.segments.len());
    for (name, imported) in &model.segments {
        let plan = compile_trace_to_plan_with_fusion(&imported.graph)
            .map_err(|source| segment_plan_error(name, source))?;
        let shared = build_segment_shared_aliases(&plan.steps, &shared_weight_store);
        let mut builder = nn_metal::compiled_model::CompiledModel::builder(&imported.graph, cache);
        if !shared.is_empty() {
            builder = builder.shared_weights(&shared);
        }
        let compiled =
            builder
                .build()
                .map_err(|source| MultiSegmentCompileError::SegmentCompile {
                    segment: name.clone(),
                    source,
                })?;
        seed_shared_weight_store(&plan.steps, &compiled, &mut shared_weight_store);
        compiled_segments.push((name.clone(), compiled));
    }

    Ok(CompiledMultiSegmentModel::new(
        compiled_segments,
        model.segment_order.clone(),
        model.shared_weights.clone(),
    ))
}

#[cfg(feature = "metal")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct OwnedWeightKey {
    shape: Box<[usize]>,
    bits: Box<[u32]>,
}

#[cfg(feature = "metal")]
fn segment_plan_error(
    segment: &str,
    source: nn_dsl::tensor_ir::TensorIRError,
) -> MultiSegmentCompileError {
    MultiSegmentCompileError::SegmentCompile {
        segment: segment.to_string(),
        source: TensorError::backend_failure_with_source(
            BackendDomain::Metal,
            BackendErrorKind::KernelCompile,
            format!("segment trace planning failed: {source}"),
            source,
        ),
    }
}

#[cfg(feature = "metal")]
fn compiled_step_weights(step: &CompiledStep) -> Option<&HashMap<String, WeightRef>> {
    match step {
        CompiledStep::Dispatch { weight_data, .. } | CompiledStep::NativeOp { weight_data, .. } => {
            Some(weight_data)
        }
        _ => None,
    }
}

#[cfg(feature = "metal")]
fn step_shares_weight_buffers(step: &CompiledStep) -> bool {
    !matches!(
        step,
        CompiledStep::NativeOp {
            op: NativeOpKind::ConstantWeight { .. },
            ..
        }
    )
}

#[cfg(feature = "metal")]
fn owned_weight_key(weight: &WeightRef) -> Option<OwnedWeightKey> {
    if weight.is_placeholder() {
        return None;
    }

    Some(OwnedWeightKey {
        shape: weight.shape().to_vec().into_boxed_slice(),
        bits: weight
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    })
}

#[cfg(feature = "metal")]
fn build_segment_shared_aliases(
    steps: &[CompiledStep],
    shared_weight_store: &HashMap<OwnedWeightKey, nn_metal::MetalBuffer>,
) -> HashMap<(usize, String), nn_metal::MetalBuffer> {
    let mut shared = HashMap::new();
    for (step_idx, step) in steps.iter().enumerate() {
        if !step_shares_weight_buffers(step) {
            continue;
        }
        let Some(weight_data) = compiled_step_weights(step) else {
            continue;
        };
        for (name, weight) in weight_data {
            let Some(key) = owned_weight_key(weight) else {
                continue;
            };
            if let Some(buffer) = shared_weight_store.get(&key) {
                shared.insert((step_idx, name.clone()), buffer.alias());
            }
        }
    }
    shared
}

#[cfg(feature = "metal")]
fn seed_shared_weight_store(
    steps: &[CompiledStep],
    compiled: &nn_metal::compiled_model::CompiledModel,
    shared_weight_store: &mut HashMap<OwnedWeightKey, nn_metal::MetalBuffer>,
) {
    let aliases = compiled.weight_buffer_aliases();
    for (step_idx, step) in steps.iter().enumerate() {
        if !step_shares_weight_buffers(step) {
            continue;
        }
        let Some(weight_data) = compiled_step_weights(step) else {
            continue;
        };
        for (name, weight) in weight_data {
            let Some(weight_key) = owned_weight_key(weight) else {
                continue;
            };
            let alias_key = (step_idx, name.clone());
            if let Some(buffer) = aliases.get(&alias_key) {
                shared_weight_store
                    .entry(weight_key)
                    .or_insert_with(|| buffer.alias());
            }
        }
    }
}

/// Import and compile multiple exported-artifact segments to Metal in one call.
///
/// This is still an exported-artifact bridge for already-segmented graph JSON
/// values plus weights; it is not a raw PyTorch/ONNX compiler or an end-to-end
/// runtime orchestrator for dynamic cross-segment control flow.
#[cfg(feature = "metal")]
pub fn convert_multi_segment_to_metal(
    graphs: &[(String, serde_json::Value)],
    weights_path: &Path,
    cache: &nn_metal::PipelineCache,
) -> Result<CompiledMultiSegmentModel, MultiSegmentCompileError> {
    let imported = convert_multi_segment(graphs, weights_path)?;
    compile_multi_segment(&imported, cache)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "multi_segment_tests.rs"]
mod tests;
