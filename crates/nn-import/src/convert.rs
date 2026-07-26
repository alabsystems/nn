// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `nn_import::convert()` — one-shot Metal wrapper for already-exported
//! artifacts.
//!
//! Takes exported `torch.export` JSON + `safetensors` weights + optional
//! reference trace, then imports and compiles them to a Metal `CompiledModel`
//! while assembling the current report scaffold when available.
//!
//! This module is intentionally narrower than a fully general PyTorch/ONNX
//! compiler: it does not ingest raw ONNX or raw PyTorch modules, and it does
//! not provide a complete proof-powered compiler certificate by itself.
//! For structured provenance wording at this boundary, prefer
//! `convert_build()` / `ConvertBuilder::build()`, which returns `ConvertReport`
//! with `provenance_summary()` and `artifact_readiness_note()`.
//!
//! # Feature gates
//!
//! - `metal`: enables the one-shot wrapper returning a Metal `CompiledModel`
//! - `verify`: enables optional NY composition-bounds reporting
//! - `reftest`: enables optional reference parity reporting against a
//!   provided trace

#[cfg(test)]
use std::collections::HashMap;
use std::path::Path;

use crate::error::ImportError;
use crate::graph_build::{build_graph, build_weight_map, ImportedGraph};
use crate::parse::parse_exported_program;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Result of `convert()` — compiled Metal model plus the current verification
/// and reporting scaffold.
#[cfg(feature = "metal")]
pub struct ConvertResult {
    /// Compiled model ready for GPU execution.
    pub model: nn_metal::compiled_model::CompiledModel,
    /// Current verification/report summary. Individual layers are optional and
    /// only reflect checks that actually ran.
    pub proof: EquivalenceProof,
    /// Imported graph metadata and trace used for compilation.
    pub graph: ImportedGraph,
}

/// Historical name for the current three-layer report scaffold returned by
/// `convert()`.
///
/// Present fields summarize checks that ran; `None` means that layer was not
/// executed or did not produce a report.
#[non_exhaustive]
pub struct EquivalenceProof {
    /// L1: Offline Kani/prover kernel-safety summary, if one was attached.
    ///
    /// `convert()` / `ConvertBuilder::build()` do not run Kani inline today,
    /// so this is typically `None` unless another tool populates it.
    pub kernel_safety: Option<KaniSafetyReport>,
    /// L2: NY composition-bounds summary when `verify` ran.
    /// `None` if NY was not run.
    pub composition_bounds: Option<CompositionBoundsReport>,
    /// L3: Numerical parity with a provided reference trace.
    /// `None` if no reference trace was provided or parity did not run.
    pub reference_parity: Option<ParityReport>,
}

impl EquivalenceProof {
    /// Create a new equivalence proof.
    pub fn new(
        kernel_safety: Option<KaniSafetyReport>,
        composition_bounds: Option<CompositionBoundsReport>,
        reference_parity: Option<ParityReport>,
    ) -> Self {
        Self {
            kernel_safety,
            composition_bounds,
            reference_parity,
        }
    }
}

/// L1: Summary of externally produced Kani/prover kernel checks.
///
/// This is not generated inline by `convert()` today.
#[non_exhaustive]
pub struct KaniSafetyReport {
    /// Total harnesses checked.
    pub harness_count: usize,
    /// Number that passed.
    pub passed: usize,
    /// Number that failed or timed out.
    pub failed: usize,
}

impl KaniSafetyReport {
    /// Create a new Kani safety report.
    pub fn new(harness_count: usize, passed: usize, failed: usize) -> Self {
        Self {
            harness_count,
            passed,
            failed,
        }
    }
}

/// L2: Summary of NY bound propagation.
#[non_exhaustive]
pub struct CompositionBoundsReport {
    /// Whether bounds propagation succeeded.
    pub propagation_ok: bool,
    /// Output bound width (lower to upper), if available.
    pub output_width: Option<f32>,
    /// Propagation method recorded for this composition-bounds run.
    ///
    /// Today `check_composition_bounds()` only populates this for the current
    /// NY composition-bounds pass. It does not imply broader proof
    /// integration beyond this report entry.
    pub composition_method: Option<report::ConvertCompositionMethod>,
    /// Soundness mode recorded for this composition-bounds run.
    pub composition_soundness_mode: Option<report::ConvertSoundnessMode>,
    /// Proof-strength classification recorded for this composition-bounds run.
    pub composition_proof_strength: Option<report::ConvertProofStrength>,
}

impl CompositionBoundsReport {
    /// Create a new composition bounds report.
    pub fn new(propagation_ok: bool, output_width: Option<f32>) -> Self {
        Self {
            propagation_ok,
            output_width,
            composition_method: None,
            composition_soundness_mode: None,
            composition_proof_strength: None,
        }
    }

    /// Attach the current verifier classification to this composition-bounds report.
    ///
    /// This is scoped to the current composition-bounds run only.
    #[must_use]
    pub fn with_verifier_classification(
        mut self,
        method: report::ConvertCompositionMethod,
        soundness_mode: Option<report::ConvertSoundnessMode>,
        proof_strength: Option<report::ConvertProofStrength>,
    ) -> Self {
        self.composition_method = Some(method);
        self.composition_soundness_mode = soundness_mode;
        self.composition_proof_strength = proof_strength;
        self
    }
}

/// L3: Summary of numerical parity with PyTorch.
#[cfg(feature = "reftest")]
pub struct ParityReport {
    /// Per-layer comparison results.
    pub divergence: nn_reftest::DivergenceReport,
}

/// L3: Summary of numerical parity with PyTorch (without reftest feature).
#[cfg(not(feature = "reftest"))]
pub struct ParityReport {
    _private: (),
}

// -- Report and builder extracted to keep convert.rs focused --

#[path = "convert_report.rs"]
pub(crate) mod report;

#[path = "convert_builder.rs"]
pub(crate) mod builder;

// -- Weight loading extracted to convert_weights.rs (Wave 4 D3a) --

#[path = "convert_weights.rs"]
mod weights;
use weights::load_safetensors_weights;
pub(crate) use weights::load_safetensors_weights as load_safetensors_weights_pub;
#[cfg(test)]
use weights::tensor_view_to_f32;

// ---------------------------------------------------------------------------
// Core import pipeline (no Metal required)
// ---------------------------------------------------------------------------

/// Import a torch.export model graph + safetensors weights into a
/// `ComputationGraph` ready for compilation.
///
/// This is the Metal-free entry point. Returns an `ImportedGraph` that can be
/// passed to `CompiledModel::builder().build()` for GPU compilation. It still
/// accepts exported artifacts only: `torch.export` JSON + `safetensors`, not
/// raw PyTorch or raw ONNX input.
pub fn import_model(graph_json: &Path, weights: &Path) -> Result<ImportedGraph, ImportError> {
    let json_bytes = std::fs::read(graph_json).map_err(|e| ImportError::Io {
        path: graph_json.display().to_string(),
        detail: e.to_string(),
    })?;
    let program = parse_exported_program(&json_bytes)?;

    let weight_data = load_safetensors_weights(weights)?;
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &weight_data);

    build_graph(&program, &weight_map)
}

// ---------------------------------------------------------------------------
// Full convert pipeline (Metal required)
// ---------------------------------------------------------------------------

/// Convert already-exported `torch.export` artifacts to a compiled Metal model
/// and the current report scaffold.
///
/// Takes:
/// - `graph_json`: Path to torch.export JSON (from `nn_export.py`)
/// - `weights`: Path to safetensors weights (from `nn_export.py`)
/// - `reference_trace`: Optional path to reference activations (from `nn_export.py`)
/// - `cache`: Metal pipeline cache for GPU compilation
///
/// When a reference trace is provided, the graph's node shapes are overridden
/// with the reference tensor shapes before compilation. This ensures that GPU
/// buffers are allocated for the reference input sizes (which may differ from
/// the shapes recorded during `torch.export` tracing).
///
/// Returns a `ConvertResult` with the compiled model plus currently available
/// report fields:
/// - L1 stays `None`; inline Kani is not run here.
/// - L2 is populated only when built with `verify`.
/// - L3 is populated only when built with `reftest` and a reference trace.
///
/// For structured provenance reporting, prefer
/// `convert_build()` / `ConvertBuilder::build()` and inspect the returned
/// `ConvertReport`. This remains an exported-artifact intake path, not a raw
/// PyTorch or raw ONNX compiler, and it is not a complete proof-powered
/// compiler by itself.
#[cfg(feature = "metal")]
pub fn convert(
    graph_json: &Path,
    weights: &Path,
    reference_trace: Option<&Path>,
    cache: &nn_metal::PipelineCache,
) -> Result<ConvertResult, ConvertError> {
    // Phase 1: Import (parse + weight load + graph build)
    #[allow(unused_mut)]
    let mut imported = import_model(graph_json, weights).map_err(ConvertError::Import)?;

    // Phase 1b: If reference trace is available, override graph node shapes
    // with reference tensor shapes BEFORE compilation. The graph.json records
    // shapes from a specific torch.export tracing run, but the reference may
    // have been generated with different input sizes (e.g., longer sequences).
    // Without this override, compiled GPU buffers are too small for reference
    // inputs, causing "buffer capacity < required" errors.
    #[cfg(feature = "reftest")]
    if let Some(ref_path) = reference_trace {
        override_graph_shapes_from_reference(&mut imported, ref_path);
    }

    // Phase 2: Compile to Metal GPU
    let model = nn_metal::compiled_model::CompiledModel::builder(&imported.graph, cache)
        .build()
        .map_err(|e| ConvertError::Compile(format!("{e}")))?;

    // Phase 3: Assemble the current verification/report scaffold
    let reference_parity = match reference_trace {
        #[cfg(feature = "reftest")]
        Some(ref_path) => {
            match check_reference_parity(&model, cache, &imported, ref_path) {
                Ok(report) => Some(report),
                Err(_) => None, // Parity check failure is non-fatal
            }
        }
        #[cfg(not(feature = "reftest"))]
        Some(_) => None,
        None => None,
    };

    // L2: NY composition bounds (IBP propagation).
    let composition_bounds = check_composition_bounds(&imported);

    let proof = EquivalenceProof::new(
        None, // L1: Populated by Prover via Kani
        composition_bounds,
        reference_parity,
    );

    Ok(ConvertResult {
        model,
        proof,
        graph: imported,
    })
}

/// Override graph node shapes from a reference trace.
///
/// Two-phase approach:
/// 1. Override input node shapes from the reference trace (name matching).
/// 2. Propagate shapes through the graph using each op's deterministic
///    shape rules, so ALL intermediate node shapes are consistent with
///    the new inputs.
///
/// This solves the name mismatch problem: torch.export node names (e.g.,
/// `conv1d`, `add_1`) don't match PyTorch module-level reference tensor
/// names (e.g., `noise_res.0.adain1.0.fc`). Instead of trying to match
/// 1164 intermediate names, we set the 3 input shapes and propagate.
#[cfg(feature = "reftest")]
fn override_graph_shapes_from_reference(imported: &mut ImportedGraph, ref_path: &Path) {
    let reference = match nn_reftest::load_safetensors(ref_path) {
        Ok(r) => r,
        Err(_) => return, // Non-fatal: reference loading may fail
    };

    // Phase 1: Build input shape overrides from the reference trace.
    // Only override input nodes — intermediate shapes will be propagated.
    let mut shape_overrides = std::collections::HashMap::new();

    let ref_input_keys: Vec<String> = reference
        .names()
        .filter(|k| k.starts_with("input_"))
        .map(|s| s.to_string())
        .collect();

    for (i, graph_name) in imported.user_input_names.iter().enumerate() {
        let fallback_idx = format!("input_{i}");
        let fallback_name = format!("input_{graph_name}");
        let ref_tensor = reference
            .get_by_name(graph_name)
            .or_else(|| reference.get_by_name(&fallback_idx))
            .or_else(|| reference.get_by_name(&fallback_name))
            .or_else(|| ref_input_keys.get(i).and_then(|k| reference.get_by_name(k)));

        if let Some(tensor) = ref_tensor {
            shape_overrides.insert(graph_name.clone(), tensor.shape.clone());
        }
    }

    if shape_overrides.is_empty() {
        return;
    }

    let input_updated = imported.graph.override_node_shapes(&shape_overrides);

    // Phase 2: Propagate shapes through the entire graph.
    // This recomputes all intermediate shapes using each op's deterministic
    // shape rules (conv output formula, element-wise passthrough, etc.).
    let propagated = imported.graph.propagate_shapes();

    let total_updated = input_updated + propagated;
    if total_updated > 0 {
        eprintln!(
            "convert: updated {total_updated} node shapes ({input_updated} inputs overridden, \
             {propagated} intermediates propagated, {} total nodes)",
            imported.graph.len()
        );
    }
}

/// Check numerical parity between the compiled model and PyTorch reference.
#[cfg(all(feature = "metal", feature = "reftest"))]
pub(crate) fn check_reference_parity(
    model: &nn_metal::compiled_model::CompiledModel,
    cache: &nn_metal::PipelineCache,
    imported: &ImportedGraph,
    ref_path: &Path,
) -> Result<ParityReport, ConvertError> {
    use nn_core::{Device, DynTensor};

    let reference = nn_reftest::load_safetensors(ref_path)
        .map_err(|e| ConvertError::Reftest(format!("{e}")))?;

    // Find input tensors in the reference trace by matching user_input_names.
    // Falls back to `input_{i}` naming (common in PyTorch reference exports)
    // when the graph-declared name isn't found.
    let mut inputs = Vec::new();
    for (i, name) in imported.user_input_names.iter().enumerate() {
        let fallback = format!("input_{i}");
        let tensor = reference
            .get_by_name(name)
            .or_else(|| reference.get_by_name(&fallback))
            .ok_or_else(|| {
                ConvertError::Reftest(format!(
                    "input '{name}' (or '{fallback}') not found in reference trace"
                ))
            })?;
        let cpu = DynTensor::from_vec(tensor.data.clone(), &tensor.shape, &Device::Cpu)
            .map_err(|e| ConvertError::Reftest(format!("tensor creation: {e}")))?;
        let gpu = cpu
            .to_device(&Device::metal())
            .map_err(|e| ConvertError::Reftest(format!("GPU transfer '{name}': {e}")))?;
        inputs.push(gpu);
    }

    let input_refs: Vec<&DynTensor> = inputs.iter().collect();
    let outputs = model
        .execute_dyn_outputs(cache, &input_refs)
        .map_err(|e| ConvertError::Reftest(format!("model execution: {e}")))?;

    // Build candidate and reference traces for all outputs.
    let output_names: Vec<String> = if imported.output_names.is_empty() {
        vec!["output".to_string()]
    } else {
        imported.output_names.clone()
    };

    // Validate output count: silent mismatch would cause incomplete parity checks.
    if outputs.len() != output_names.len() {
        return Err(ConvertError::Reftest(format!(
            "model produced {} outputs but graph declares {}",
            outputs.len(),
            output_names.len()
        )));
    }

    let mut candidate_trace = nn_reftest::ReferenceTrace::new();
    let mut ref_trace = nn_reftest::ReferenceTrace::new();

    for (i, name) in output_names.iter().enumerate() {
        // SAFETY: bounds already checked above.
        let out = &outputs[i];
        let flat = out
            .to_flat_vec::<f32>()
            .map_err(|e| ConvertError::Reftest(format!("output '{name}' extraction: {e}")))?;
        candidate_trace
            .checkpoint(name, &flat, out.dims())
            .map_err(|e| ConvertError::Reftest(format!("candidate trace '{name}': {e}")))?;

        // Falls back to positional naming (common in PyTorch reference exports)
        // when the graph-declared name isn't found.
        // Single-output: try "output". Multi-output: try "output_{i}".
        let fallback_out = if output_names.len() == 1 {
            "output".to_string()
        } else {
            format!("output_{i}")
        };
        let ref_output = reference
            .get_by_name(name)
            .or_else(|| reference.get_by_name(&fallback_out))
            .ok_or_else(|| {
                ConvertError::Reftest(format!(
                    "output '{name}' (or '{fallback_out}') not found in reference trace"
                ))
            })?;
        ref_trace
            .checkpoint(&ref_output.name, &ref_output.data, &ref_output.shape)
            .map_err(|e| ConvertError::Reftest(format!("reference trace '{name}': {e}")))?;
    }

    // Real models accumulate float32 rounding errors through deep compute chains
    // (Conv→Norm→Pool→LSTM→Linear). Use tolerances appropriate for GPU parity,
    // not the tight defaults (1e-5 abs) designed for element-wise op tests.
    let config = nn_reftest::ComparisonConfig::new(0.02, 0.02, 0.999);
    let divergence = nn_reftest::compare_traces(&ref_trace, &candidate_trace, &config)
        .map_err(|e| ConvertError::Reftest(format!("comparison: {e}")))?;

    Ok(ParityReport { divergence })
}

// ---------------------------------------------------------------------------
// L2: Composition bounds via NY IBP
// ---------------------------------------------------------------------------

/// Check composition bounds by translating the graph to NY and
/// running IBP (Interval Bound Propagation).
///
/// Uses uniform input bounds of `[-1, 1]` over the first user input shape.
/// Returns `None` if translation or propagation fails (non-fatal).
///
/// The current machine-readable classification reflects only this
/// composition-bounds pass. It does not claim inline Kani coverage or a
/// complete proof-powered compiler certificate.
#[cfg(feature = "verify")]
pub fn check_composition_bounds(imported: &ImportedGraph) -> Option<CompositionBoundsReport> {
    use ndarray::{ArrayD, IxDyn};

    // Count variable inputs. Multi-input models (e.g., Kokoro decoder with 3 inputs)
    // require trace_to_graph_model_multi_input which stacks inputs into a flat 1D tensor.
    let variable_input_count = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
        .count();

    let is_multi_input = variable_input_count > 1;

    let gn = if is_multi_input {
        nn_verify::trace_to_graph_model_multi_input(&imported.graph)
            .ok()
            .map(|r| r.graph)?
    } else {
        nn_verify::trace_to_graph_model(&imported.graph)
            .ok()
            .map(|r| r.graph)?
    };

    if is_multi_input {
        // Multi-input: stacked flat 1D bounds. Sum element counts across all inputs.
        let total_elements: usize = imported
            .graph
            .nodes()
            .iter()
            .filter(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
            .map(|n| n.output_shape().iter().product::<usize>())
            .sum();
        if total_elements == 0 {
            return None;
        }
        let lower = ArrayD::from_elem(IxDyn(&[total_elements]), -1.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[total_elements]), 1.0_f32);
        let input_bounds = nn_verify::BoundedTensor::new(lower, upper).ok()?;
        propagate_and_report(&gn, &input_bounds)
    } else {
        // Single-input: shaped bounds matching the declared shape.
        let input_node = imported
            .graph
            .nodes()
            .iter()
            .find(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))?;
        let shape = input_node.output_shape();
        if shape.iter().any(|&d| d == 0) {
            return None;
        }
        let lower = ArrayD::from_elem(IxDyn(shape), -1.0_f32);
        let upper = ArrayD::from_elem(IxDyn(shape), 1.0_f32);
        let input_bounds = nn_verify::BoundedTensor::new(lower, upper).ok()?;
        propagate_and_report(&gn, &input_bounds)
    }
}

/// Run IBP propagation and produce a composition bounds report.
#[cfg(feature = "verify")]
fn propagate_and_report(
    gn: &nn_verify::GraphNetwork,
    input_bounds: &nn_verify::BoundedTensor,
) -> Option<CompositionBoundsReport> {
    let method = nn_verify::PropMethod::Ibp;
    let output = gn.propagate_ibp(input_bounds).ok()?;
    let (out_lower, out_upper) = output.lower_upper();

    let width = out_upper
        .iter()
        .zip(out_lower.iter())
        .map(|(hi, lo)| hi - lo)
        .fold(0.0_f32, f32::max);
    let finite_width = if width.is_finite() { Some(width) } else { None };
    let verify_soundness_mode =
        nn_verify::soundness_mode_for_graph(gn, &method, Some(input_bounds)).ok();
    let soundness_mode =
        verify_soundness_mode.and_then(report::ConvertSoundnessMode::from_verify_soundness_mode);
    let proof_strength = finite_width.zip(verify_soundness_mode).and_then(
        |(finite_width, verify_soundness_mode)| {
            report::ConvertProofStrength::from_verify_proof_strength(
                nn_verify::status::compute_proof_strength(
                    verify_soundness_mode,
                    method,
                    finite_width,
                ),
            )
        },
    );

    Some(
        CompositionBoundsReport::new(true, finite_width).with_verifier_classification(
            report::ConvertCompositionMethod::from_verify_method(method)
                .unwrap_or(report::ConvertCompositionMethod::Ibp),
            soundness_mode,
            proof_strength,
        ),
    )
}

/// Stub when verify feature is not enabled.
#[cfg(not(feature = "verify"))]
pub fn check_composition_bounds(_imported: &ImportedGraph) -> Option<CompositionBoundsReport> {
    None
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error from `convert()` pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConvertError {
    /// Import phase failed (parse, weight load, or graph build).
    #[error("import error: {0}")]
    Import(#[from] ImportError),

    /// Compilation to GPU failed.
    #[error("compilation error: {0}")]
    Compile(String),

    /// Reference parity check failed.
    #[error("reftest error: {0}")]
    Reftest(String),
}

#[cfg(test)]
#[path = "convert_tests.rs"]
mod tests;
