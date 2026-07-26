// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder pattern for the exported-artifact `convert()` pipeline.
//!
//! The recommended entry point is [`convert()`], which returns a
//! [`ConvertBuilder`] for already-exported `torch.export` JSON +
//! `safetensors` weights. Chain `.reference_trace()`, `.optimize()`, and
//! `.verify()` before `.build()` to customize the current import -> compile ->
//! report pipeline.
//!
//! [`VerifyLevel::Full`] requests the fullest report this builder can assemble
//! today. That means NY bounds reporting when `verify` is enabled,
//! including the current composition-bounds method/soundness/proof-strength
//! classification when it is available, plus optional reference parity when
//! `reftest` is enabled and a reference trace is supplied. It does not run
//! Kani inline, does not accept raw ONNX or raw PyTorch input, and does not
//! turn `build()` into a complete proof-powered compiler.
//!
//! # Examples
//!
//! ```rust,ignore
//! use nn_import::{convert, OptLevel, VerifyLevel};
//!
//! let result = convert(&graph_json, &weights, &cache)
//!     .reference_trace(&ref_path)
//!     .optimize(OptLevel::Aggressive)
//!     .verify(VerifyLevel::Bounds)
//!     .build()?;
//!
//! result.report.print();  // human-readable summary
//! // result.result.model  -- CompiledModel
//! // result.result.proof  -- EquivalenceProof
//! ```
//!
//! # Feature gates
//!
//! - `metal`: required for compilation and GPU execution
//! - `verify`: enables NY composition-bounds reporting
//! - `reftest`: enables optional reference parity against a provided trace

use super::report::VerificationCoverage;
#[cfg(feature = "metal")]
use super::report::{
    ConvertArtifactKind, ConvertIntakePath, ConvertReport, FusionReport, PeepholeReport,
};

/// Optimization level for the convert pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum OptLevel {
    /// No fusion or peephole optimization. Fastest compile, most dispatches.
    None,
    /// Full optimization: constant folding + elementwise fusion + peephole.
    /// This is the default and recommended level.
    #[default]
    Full,
    /// Aggressive optimization: `Full` plus profile-guided optimization rounds.
    ///
    /// Currently equivalent to `Full` (profile-guided optimization is planned
    /// for a follow-up). Selecting `Aggressive` opts in to future PGO when it
    /// lands, without API changes.
    Aggressive,
}

/// Verification level for the convert pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum VerifyLevel {
    /// No verification. Fastest pipeline.
    None,
    /// Request NY composition-bounds reporting.
    ///
    /// When the crate is built with the `verify` feature, `build()` populates
    /// the composition-bounds fields in the report, including the current
    /// method/soundness/proof-strength classification when that verifier path
    /// can supply it. Without that feature, the request is accepted but no
    /// NY report is produced.
    #[default]
    Bounds,
    /// Request the fullest report path currently available from `build()`.
    ///
    /// Today this expands to:
    /// - NY composition bounds when `verify` is enabled
    /// - optional reference parity when `reftest` is enabled and a
    ///   [`ConvertBuilder::reference_trace`] is provided
    ///
    /// It does not run Kani inline, does not ingest raw PyTorch or ONNX
    /// inputs, and does not by itself provide a complete proof-powered
    /// compiler.
    Full,
}

/// Create a [`ConvertBuilder`] for the given model.
///
/// This is the builder entry point for already-exported artifacts. Returns a
/// builder with sensible defaults (`OptLevel::Full`,
/// `VerifyLevel::Bounds`). Bounds reporting appears only when the crate is
/// built with `verify`, and reference parity appears only when a reference
/// trace is provided and `reftest` is enabled.
///
/// # Examples
///
/// ```rust,ignore
/// use nn_import::{convert, OptLevel, VerifyLevel};
///
/// // Minimal usage:
/// let result = convert(&graph_json, &weights, &cache).build()?;
///
/// // Full usage:
/// let result = convert(&graph_json, &weights, &cache)
///     .reference_trace(&ref_path)
///     .optimize(OptLevel::Aggressive)
///     .verify(VerifyLevel::Bounds)
///     .build()?;
///
/// result.report.print();
/// let model = result.result.model;
/// let proof = result.result.proof;
/// ```
#[cfg(feature = "metal")]
pub fn convert<'a>(
    graph_json: &'a std::path::Path,
    weights: &'a std::path::Path,
    cache: &'a nn_metal::PipelineCache,
) -> ConvertBuilder<'a> {
    ConvertBuilder::new(graph_json, weights, cache)
}

/// Builder for the exported-artifact `convert()` pipeline.
///
/// Provides fine-grained control over import, compilation, and report layers
/// for already-exported `torch.export` JSON + `safetensors` weights, and
/// produces a [`ConvertReport`] with detailed metrics. Provenance setters such
/// as [`ConvertBuilder::intake_path`] annotate the report; they do not widen
/// the accepted input format beyond exported artifacts.
///
/// # Examples
///
/// ```rust,ignore
/// use nn_import::{ConvertBuilder, VerifyLevel};
///
/// let result = ConvertBuilder::new(&graph_json, &weights, &cache)
///     .reference_trace(&ref_path)
///     .verify(VerifyLevel::Bounds)
///     .build()?;
///
/// println!("{}", result.report);
/// ```
#[cfg(feature = "metal")]
pub struct ConvertBuilder<'a> {
    graph_json: &'a std::path::Path,
    weights: &'a std::path::Path,
    cache: &'a nn_metal::PipelineCache,
    reference_trace: Option<std::path::PathBuf>,
    intake_path: ConvertIntakePath,
    opt_level: OptLevel,
    verify_level: VerifyLevel,
}

#[cfg(feature = "metal")]
impl<'a> ConvertBuilder<'a> {
    /// Create a new builder with required parameters.
    pub fn new(
        graph_json: &'a std::path::Path,
        weights: &'a std::path::Path,
        cache: &'a nn_metal::PipelineCache,
    ) -> Self {
        Self {
            graph_json,
            weights,
            cache,
            reference_trace: None,
            intake_path: ConvertIntakePath::ExportedArtifacts,
            opt_level: OptLevel::default(),
            verify_level: VerifyLevel::default(),
        }
    }

    /// Set the path to PyTorch reference activations for L3 parity checking.
    #[must_use]
    pub fn reference_trace(mut self, path: &std::path::Path) -> Self {
        self.reference_trace = Some(path.to_path_buf());
        self
    }

    /// Record the provenance of the exported artifacts consumed by this build.
    ///
    /// This annotates the resulting [`ConvertReport`]. It does not change what
    /// inputs `build()` accepts: the builder still consumes exported
    /// `torch.export` JSON + `safetensors`, not raw PyTorch or raw ONNX.
    #[must_use]
    pub fn intake_path(mut self, path: ConvertIntakePath) -> Self {
        self.intake_path = path;
        self
    }

    /// Mark the report intake as artifacts exported by
    /// `nn convert --from-pytorch`.
    ///
    /// This is a report-provenance shortcut for
    /// [`ConvertBuilder::intake_path`]. It does not make the builder ingest raw
    /// PyTorch directly.
    #[must_use]
    pub fn cli_exported_from_pytorch(self) -> Self {
        self.intake_path(ConvertIntakePath::CliExportedPytorch)
    }

    /// Set the optimization level.
    #[must_use]
    pub fn optimize(mut self, level: OptLevel) -> Self {
        self.opt_level = level;
        self
    }

    /// Set the verification level.
    #[must_use]
    pub fn verify(mut self, level: VerifyLevel) -> Self {
        self.verify_level = level;
        self
    }

    /// Build the converted model with a detailed report.
    ///
    /// Runs the current exported-artifact pipeline: import -> compile ->
    /// available reporting -> report.
    ///
    /// The returned [`ConvertResultWithReport`] contains both the compiled
    /// model (same as the legacy one-shot `convert()`) and the detailed
    /// [`ConvertReport`]. Report fields are populated only for checks that
    /// actually ran: NY bounds require `verify`, reference parity
    /// requires `reftest` plus [`ConvertBuilder::reference_trace`], and L1
    /// Kani remains an external/offline input today. Successful builds record
    /// a compiled Metal artifact in the report; the provenance setters describe
    /// how the exported-artifact intake was obtained.
    pub fn build(self) -> Result<ConvertResultWithReport, super::ConvertError> {
        use super::{check_composition_bounds, ConvertError, ConvertResult, EquivalenceProof};
        use crate::error::ImportError;

        let mut report = ConvertReport::new();
        report.intake_path = self.intake_path;

        // Phase 1a: Parse the graph JSON and load weights.
        let json_bytes = std::fs::read(self.graph_json).map_err(|e| {
            ConvertError::Import(ImportError::Io {
                path: self.graph_json.display().to_string(),
                detail: e.to_string(),
            })
        })?;
        let program =
            crate::parse::parse_exported_program(&json_bytes).map_err(ConvertError::Import)?;
        let weight_data =
            super::weights::load_safetensors_weights(self.weights).map_err(ConvertError::Import)?;
        let weight_map = crate::graph_build::build_weight_map(
            &program.graph_module.signature.input_specs,
            &weight_data,
        );

        // Phase 1b: Collect op mapping statistics from parsed graph nodes.
        collect_op_stats(&program.graph_module.graph.nodes, &mut report);

        // Phase 1c: Build the computation graph.
        let imported =
            crate::graph_build::build_graph(&program, &weight_map).map_err(ConvertError::Import)?;

        // Phase 1d: Override graph shapes from reference trace if available.
        // The graph.json records shapes from a specific torch.export tracing run,
        // but the reference may have been generated with different input sizes.
        // Without shape override + propagation, intermediate buffer sizes are wrong
        // and produce NaN. This mirrors the logic in convert::override_graph_shapes_from_reference.
        #[cfg(feature = "reftest")]
        if let Some(ref_path) = self.reference_trace.as_deref() {
            super::override_graph_shapes_from_reference(&mut imported, ref_path);
        }

        // Populate import metrics.
        report.num_user_inputs = imported.num_user_inputs;
        report.total_ops_imported = imported.graph.len();
        // Count Constant nodes as weight placeholders (parameters + buffers).
        let constant_count = imported
            .graph
            .nodes()
            .iter()
            .filter(|n| {
                matches!(
                    n.op(),
                    nn_core::dyn_tensor::trace::TraceOp::Constant { .. }
                )
            })
            .count();
        report.num_weights_loaded = constant_count;

        // Estimate pre-fusion dispatch count from the graph.
        // Count non-Input, non-Constant nodes as potential dispatches.
        report.dispatch_count_before_fusion = imported
            .graph
            .nodes()
            .iter()
            .filter(|n| {
                !matches!(
                    n.op(),
                    nn_core::dyn_tensor::trace::TraceOp::Input
                        | nn_core::dyn_tensor::trace::TraceOp::Constant { .. }
                )
            })
            .count();

        // Phase 2: Compile to Metal GPU (timed).
        let compile_start = std::time::Instant::now();
        let model = nn_metal::compiled_model::CompiledModel::builder(&imported.graph, self.cache)
            .build()
            .map_err(|e| ConvertError::Compile(format!("{e}")))?;
        report.compile_time_ms = compile_start.elapsed().as_millis() as u64;
        report.artifact_kind = ConvertArtifactKind::CompiledMetalArtifact;

        // Populate compilation metrics from the CompiledModel.
        report.dispatch_count = model.num_dispatches();
        report.total_steps = model.num_steps();
        report.metal_dispatches = model.num_metal_dispatches();

        // Extract peephole + fusion stats from the compiled plan steps.
        populate_compilation_stats(&model, &mut report);

        // Populate fusion_count and native_op_count from compiled stats.
        report.native_op_count = report.peephole_stats.native_ops;
        report.fusion_count =
            report.fusion_stats.fused_ops + report.peephole_stats.passthrough_count;

        // Estimate RTF from dispatch count.
        report.estimate_rtf();

        // Phase 3: Verification.
        let mut composition_bounds = None;
        if self.verify_level != VerifyLevel::None {
            composition_bounds = check_composition_bounds(&imported);
            if let Some(ref cb) = composition_bounds {
                report.verification.composition_bounds_ok = cb.propagation_ok;
                report.verification.composition_bound_width = cb.output_width;
                report.verification.composition_method = cb.composition_method;
                report.verification.composition_soundness_mode = cb.composition_soundness_mode;
                report.verification.composition_proof_strength = cb.composition_proof_strength;
            }
            // Count NY layer coverage.
            populate_verification_coverage(&imported, &mut report.verification);
        }

        // Phase 3b: Reference parity (L3).
        let reference_parity = match self.reference_trace.as_deref() {
            #[cfg(feature = "reftest")]
            Some(ref_path) => {
                match super::check_reference_parity(&model, self.cache, &imported, ref_path) {
                    Ok(parity) => {
                        report.verification.reference_parity_passed =
                            Some(parity.divergence.all_passed);
                        Some(parity)
                    }
                    Err(_) => {
                        report.verification.reference_parity_passed = Some(false);
                        None
                    }
                }
            }
            #[cfg(not(feature = "reftest"))]
            Some(_) => None,
            None => None,
        };

        let proof = EquivalenceProof::new(
            None, // L1: Populated by Prover via Kani
            composition_bounds,
            reference_parity,
        );

        let convert_result = ConvertResult {
            model,
            proof,
            graph: imported,
        };

        Ok(ConvertResultWithReport {
            result: convert_result,
            report,
        })
    }
}

/// Result of [`ConvertBuilder::build()`] -- includes both the convert result
/// and a detailed report.
#[cfg(feature = "metal")]
pub struct ConvertResultWithReport {
    /// The compiled model, proof, and imported graph (same as `convert()`).
    pub result: super::ConvertResult,
    /// Detailed optimization and verification report.
    pub report: ConvertReport,
}

/// Populate peephole and fusion stats from compiled model steps.
#[cfg(feature = "metal")]
fn populate_compilation_stats(
    model: &nn_metal::compiled_model::CompiledModel,
    report: &mut ConvertReport,
) {
    use nn_dsl::trace_compile::CompiledStep;

    let steps = model.steps();

    let mut native_ops = 0usize;
    let mut native_dispatches = 0usize;
    let mut passthrough_count = 0usize;
    let mut variant_map: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut fused_chains = 0usize;
    let mut fused_ops = 0usize;

    for step in steps {
        match step {
            CompiledStep::NativeOp { op, .. } => {
                native_ops += 1;
                native_dispatches += op.estimated_metal_dispatches();
                *variant_map
                    .entry(op.variant_name().to_string())
                    .or_default() += 1;
            }
            CompiledStep::IdentityPassthrough => {
                passthrough_count += 1;
            }
            CompiledStep::Dispatch { kernel, .. } => {
                let name = kernel.name();
                if let Some(rest) = name.strip_prefix("fused_") {
                    if let Some(x_pos) = rest.rfind("_x") {
                        if let Ok(chain_len) = rest[x_pos + 2..].parse::<usize>() {
                            fused_chains += 1;
                            fused_ops += chain_len;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut by_variant: Vec<_> = variant_map.into_iter().collect();
    by_variant.sort_by_key(|x| std::cmp::Reverse(x.1));

    report.peephole_stats = PeepholeReport {
        native_ops,
        native_dispatches,
        passthrough_count,
        by_variant,
    };

    report.fusion_stats = FusionReport {
        fused_chains,
        fused_ops,
        dispatches_saved: fused_ops.saturating_sub(fused_chains),
    };
}

/// Collect op mapping statistics from the parsed graph nodes.
///
/// Scans each computation node's target against the op_map dispatch table and
/// classifies it as mapped or unmapped. Populates `report.op_count`,
/// `report.mapped_ops`, and `report.unmapped_ops`.
#[cfg(feature = "metal")]
fn collect_op_stats(nodes: &[crate::parse::Node], report: &mut ConvertReport) {
    use std::collections::HashMap;

    let mut mapped: HashMap<String, usize> = HashMap::new();
    let mut unmapped: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;

    for node in nodes {
        let target = &node.target;
        // Skip getitem nodes (Python tuple unpacking, not real ops).
        if target.contains("getitem") {
            continue;
        }
        total += 1;
        if is_supported_target(target) {
            *mapped.entry(target.clone()).or_default() += 1;
        } else {
            *unmapped.entry(target.clone()).or_default() += 1;
        }
    }

    report.op_count = total;

    let mut mapped_vec: Vec<_> = mapped.into_iter().collect();
    mapped_vec.sort_by_key(|x| std::cmp::Reverse(x.1));
    report.mapped_ops = mapped_vec;

    let mut unmapped_vec: Vec<_> = unmapped.into_iter().collect();
    unmapped_vec.sort_by_key(|x| std::cmp::Reverse(x.1));
    report.unmapped_ops = unmapped_vec;
}

/// Check if a target string is in the op_map dispatch table.
///
/// Returns `true` for all aten targets that `map_node_to_trace_op` and
/// `try_expand_node` can handle. This is a lightweight check that does not
/// require weight data or tensor metadata — it only inspects the target name.
#[cfg(feature = "metal")]
fn is_supported_target(target: &str) -> bool {
    matches!(
        target,
        // Unary element-wise
        "torch.ops.aten.relu.default"
        | "torch.ops.aten.gelu.default"
        | "torch.ops.aten.silu.default"
        | "torch.ops.aten.tanh.default"
        | "torch.ops.aten.sigmoid.default"
        | "torch.ops.aten.exp.default"
        | "torch.ops.aten.log.default"
        | "torch.ops.aten.sqrt.default"
        | "torch.ops.aten.abs.default"
        | "torch.ops.aten.neg.default"
        | "torch.ops.aten.reciprocal.default"
        | "torch.ops.aten.sin.default"
        | "torch.ops.aten.cos.default"
        | "torch.ops.aten.floor.default"
        | "torch.ops.aten.round.default"
        // Binary element-wise
        | "torch.ops.aten.add.Tensor"
        | "torch.ops.aten.add_.Tensor"
        | "torch.ops.aten.sub.Tensor"
        | "torch.ops.aten.mul.Tensor"
        | "torch.ops.aten.div.Tensor"
        | "torch.ops.aten.maximum.default"
        | "torch.ops.aten.minimum.default"
        // Scalar binary (expanded)
        | "torch.ops.aten.add.Scalar"
        | "torch.ops.aten.sub.Scalar"
        | "torch.ops.aten.mul.Scalar"
        | "torch.ops.aten.div.Scalar"
        // Matrix multiply
        | "torch.ops.aten.mm.default"
        | "torch.ops.aten.bmm.default"
        | "torch.ops.aten.matmul.default"
        // Linear
        | "torch.ops.aten.linear.default"
        // Convolution
        | "torch.ops.aten.convolution.default"
        // Normalization
        | "torch.ops.aten.layer_norm.default"
        | "torch.ops.aten.group_norm.default"
        | "torch.ops.aten.native_batch_norm.default"
        | "torch.ops.aten._native_batch_norm_legit_no_training.default"
        | "torch.ops.aten.instance_norm.default"
        // Attention / softmax
        | "torch.ops.aten.softmax.int"
        | "torch.ops.aten._softmax.default"
        | "torch.ops.aten.log_softmax.int"
        | "torch.ops.aten._log_softmax.default"
        | "torch.ops.aten.scaled_dot_product_attention.default"
        // Embedding
        | "torch.ops.aten.embedding.default"
        // Reductions
        | "torch.ops.aten.sum.dim_IntList"
        | "torch.ops.aten.mean.dim"
        | "torch.ops.aten.amax.default"
        | "torch.ops.aten.amin.default"
        // Shape operations
        | "torch.ops.aten.view.default"
        | "torch.ops.aten.reshape.default"
        | "torch.ops.aten._unsafe_view.default"
        | "torch.ops.aten.transpose.int"
        | "torch.ops.aten.permute.default"
        | "torch.ops.aten.unsqueeze.default"
        | "torch.ops.aten.squeeze.dim"
        | "torch.ops.aten.squeeze.default"
        | "torch.ops.aten.cat.default"
        | "torch.ops.aten.slice.Tensor"
        | "torch.ops.aten.expand.default"
        | "torch.ops.aten.flip.default"
        | "torch.ops.aten.select.int"
        | "torch.ops.aten.chunk.default"
        // Pooling
        | "torch.ops.aten.max_pool1d.default"
        | "torch.ops.aten.max_pool1d_with_indices.default"
        | "torch.ops.aten.avg_pool2d.default"
        | "torch.ops.aten.max_pool2d_with_indices.default"
        | "torch.ops.aten.adaptive_avg_pool2d.default"
        // Activation (ext)
        | "torch.ops.aten.elu.default"
        | "torch.ops.aten.leaky_relu.default"
        | "torch.ops.aten.dropout.default"
        // Comparison / Selection
        | "torch.ops.aten.where.self"
        | "torch.ops.aten.clamp.default"
        | "torch.ops.aten.clamp_min.default"
        // Type conversion
        | "torch.ops.aten.to.dtype"
        | "torch.ops.aten._to_copy.default"
        // Power
        | "torch.ops.aten.pow.Tensor_Scalar"
        // Recurrent
        | "torch.ops.aten.lstm.input"
        // Misc
        | "torch.ops.aten.cumsum.default"
        | "torch.ops.aten.repeat_interleave.self_Tensor"
        // Zero tensor creation
        | "torch.ops.aten.zeros.default"
        | "torch.ops.aten.zeros_like.default"
        // Standalone conv1d
        | "torch.ops.aten.conv1d.default"
        // ConvTranspose1d
        | "torch.ops.aten.conv_transpose1d.default"
        // Padding
        | "torch.ops.aten.reflection_pad1d.default"
        | "torch.ops.aten.constant_pad_nd.default"
        | "torch.ops.aten.pad.default"
        // Upsampling
        | "torch.ops.aten.upsample_nearest1d.default"
        | "torch.ops.aten.upsample_nearest1d.vec"
        // Indexing
        | "torch.ops.aten.index_select.default"
        // Scalar comparison
        | "torch.ops.aten.gt.Scalar"
        | "torch.ops.aten.lt.Scalar"
        | "torch.ops.aten.ge.Scalar"
        | "torch.ops.aten.le.Scalar"
        | "torch.ops.aten.eq.Scalar"
        | "torch.ops.aten.ne.Scalar"
        // Tensor comparison
        | "torch.ops.aten.gt.Tensor"
        | "torch.ops.aten.lt.Tensor"
        | "torch.ops.aten.eq.Tensor"
        // Trigonometric extended
        | "torch.ops.aten.atan2.default"
        // Tensor creation
        | "torch.ops.aten.ones.default"
        | "torch.ops.aten.ones_like.default"
        | "torch.ops.aten.full.default"
        | "torch.ops.aten.full_like.default"
        | "torch.ops.aten.arange.default"
        | "torch.ops.aten.arange.start_step"
        // Identity / memory layout
        | "torch.ops.aten.contiguous.default"
        | "torch.ops.aten.clone.default"
        | "torch.ops.aten._copy.default"
        // dpdf model ops: upsampling 2D
        | "torch.ops.aten.upsample_nearest2d.default"
        | "torch.ops.aten.upsample_nearest2d.vec"
        | "torch.ops.aten.upsample_bilinear2d.default"
        | "torch.ops.aten.upsample_bilinear2d.vec"
        // dpdf model ops: normalization
        | "torch.ops.aten.rms_norm.default"
        // dpdf model ops: activation
        | "torch.ops.aten.hardswish.default"
        | "torch.ops.aten.hardswish_.default"
        | "torch.ops.aten.hardsigmoid.default"
        | "torch.ops.aten.mish.default"
        | "torch.ops.aten.softplus.default"
        | "torch.ops.aten.selu.default"
        // dpdf model ops: mask
        | "torch.ops.aten.triu.default"
        | "torch.ops.aten.tril.default"
        // dpdf model ops: selection / indexing
        | "torch.ops.aten.gather.default"
        | "torch.ops.aten.argmax.default"
        | "torch.ops.aten.argmin.default"
        // dpdf model ops: vision
        | "torch.ops.aten.pixel_shuffle.default"
        | "torch.ops.aten.pixel_unshuffle.default"
        // dpdf model ops: split / unbind / repeat
        | "torch.ops.aten.split.Tensor"
        | "torch.ops.aten.split_with_sizes.default"
        | "torch.ops.aten.unbind.int"
        | "torch.ops.aten.repeat.default"
        // Wave 6: interpolate, scatter, padding, clamp
        | "torch.ops.aten.interpolate.default"
        | "torch.ops.aten.interpolate.vec"
        | "torch.ops.aten.scatter.src"
        | "torch.ops.aten.scatter.value"
        | "torch.ops.aten.scatter_add.default"
        | "torch.ops.aten.reflection_pad2d.default"
        | "torch.ops.aten.clamp_max.default"
        | "torch.ops.aten.narrow.default"
        | "torch.ops.aten.narrow.Tensor"
        | "torch.ops.aten.topk.default"
        | "torch.ops.aten.sort.default"
        | "torch.ops.aten.sort.stable"
        | "torch.ops.aten.roll.default"
        | "torch.ops.aten.masked_fill.Scalar"
        | "torch.ops.aten.masked_fill_.Scalar"
        | "torch.ops.aten.index.Tensor"
        | "torch.ops.aten.stack.default"
        // Conv / pool extended
        | "torch.ops.aten.conv2d.default"
        | "torch.ops.aten.conv3d.default"
        | "torch.ops.aten.conv_transpose2d.input"
        | "torch.ops.aten.conv_transpose2d.default"
        | "torch.ops.aten.batch_norm.default"
        | "torch.ops.aten.max_pool2d.default"
        | "torch.ops.aten.avg_pool1d.default"
        | "torch.ops.aten.adaptive_avg_pool1d.default"
        | "torch.ops.aten.adaptive_max_pool2d.default"
        | "torch.ops.aten.grid_sample.default"
        // Wave 7: transformer / audio model ops
        | "torch.ops.aten.tan.default"
        | "torch.ops.aten.ceil.default"
        | "torch.ops.aten.sign.default"
        | "torch.ops.aten.sgn.default"
        | "torch.ops.aten.frac.default"
        | "torch.ops.aten.log2.default"
        | "torch.ops.aten.log10.default"
        | "torch.ops.aten.exp2.default"
        | "torch.ops.aten.erf.default"
        | "torch.ops.aten.rsqrt.default"
        // Activation extended
        | "torch.ops.aten.softsign.default"
        | "torch.ops.aten.prelu.default"
        | "torch.ops.aten.log_sigmoid.default"
        | "torch.ops.aten.log_sigmoid_forward.default"
        | "torch.ops.aten.glu.default"
        | "torch.ops.aten.celu.default"
        | "torch.ops.aten.celu_.default"
        | "torch.ops.aten.selu_.default"
        | "torch.ops.aten.hardtanh.default"
        | "torch.ops.aten.hardtanh_.default"
        // Tensor comparison extended
        | "torch.ops.aten.ge.Tensor"
        | "torch.ops.aten.le.Tensor"
        | "torch.ops.aten.ne.Tensor"
        // Matrix ops
        | "torch.ops.aten.addmm.default"
        | "torch.ops.aten.baddbmm.default"
        // Index ops
        | "torch.ops.aten.index_add.default"
        | "torch.ops.aten.index_add_.default"
        | "torch.ops.aten.index_put.default"
        | "torch.ops.aten.index_put_.default"
        | "torch.ops.aten.unfold.default"
        // Tensor creation extended
        | "torch.ops.aten.empty.memory_format"
        | "torch.ops.aten.empty.default"
        | "torch.ops.aten.empty_like.default"
        | "torch.ops.aten.new_zeros.default"
        | "torch.ops.aten.new_ones.default"
        | "torch.ops.aten.linspace.default"
        | "torch.ops.aten.scalar_tensor.default"
        | "torch.ops.aten.fill.Scalar"
        | "torch.ops.aten.fill_.Scalar"
        | "torch.ops.aten.zero.default"
        | "torch.ops.aten.zero_.default"
        // Shape ops extended
        | "torch.ops.aten.t.default"
        | "torch.ops.aten.movedim.int"
        | "torch.ops.aten.flatten.using_ints"
        // Power extended
        | "torch.ops.aten.pow.Tensor_Tensor"
        | "torch.ops.aten.pow.Scalar"
        // Reductions extended
        | "torch.ops.aten.sum.default"
        | "torch.ops.aten.mean.default"
        | "torch.ops.aten.prod.default"
        | "torch.ops.aten.prod.dim_int"
        | "torch.ops.aten.var.default"
        | "torch.ops.aten.var.correction"
        | "torch.ops.aten.std.default"
        | "torch.ops.aten.std.correction"
        | "torch.ops.aten.any.default"
        | "torch.ops.aten.any.dim"
        | "torch.ops.aten.all.default"
        | "torch.ops.aten.all.dim"
        // Boolean / logical
        | "torch.ops.aten.logical_not.default"
        | "torch.ops.aten.logical_and.default"
        | "torch.ops.aten.logical_or.default"
        // Miscellaneous
        | "torch.ops.aten.remainder.Scalar"
        | "torch.ops.aten.remainder.Tensor"
        | "torch.ops.aten.fmod.Scalar"
        | "torch.ops.aten.fmod.Tensor"
        | "torch.ops.aten.slice_scatter.default"
        | "torch.ops.aten.copy.default"
        | "torch.ops.aten.copy_.default"
        // In-place variants
        | "torch.ops.aten.add_.Scalar"
        | "torch.ops.aten.sub_.Scalar"
        | "torch.ops.aten.sub_.Tensor"
        | "torch.ops.aten.mul_.Scalar"
        | "torch.ops.aten.mul_.Tensor"
        | "torch.ops.aten.div_.Scalar"
        | "torch.ops.aten.div_.Tensor"
        // Meshgrid (decomposed via try_expand_node)
        | "torch.ops.aten.meshgrid.default"
        | "torch.ops.aten.meshgrid.indexing"
        // Wave 8+: synced from op_map dispatch table
        // Batch norm variants
        | "torch.ops.aten._native_batch_norm_legit.default"
        | "torch.ops.aten._native_batch_norm_legit.no_stats"
        | "torch.ops.aten.cudnn_batch_norm.default"
        // Normalization variants (no-affine)
        | "torch.ops.aten.layer_norm.no_affine"
        | "torch.ops.aten.group_norm.no_affine"
        | "torch.ops.aten.instance_norm.affine"
        // Attention variants
        | "torch.ops.aten._scaled_dot_product_efficient_attention.default"
        | "torch.ops.aten._scaled_dot_product_flash_attention.default"
        | "torch.ops.aten.multi_head_attention_forward.default"
        // Embedding variants
        | "torch.ops.aten.embedding.padding_idx"
        | "torch.ops.aten.embedding_bag.default"
        | "torch.ops.aten._embedding_bag.default"
        // Trigonometric / transcendental
        | "torch.ops.aten.asin.default"
        | "torch.ops.aten.acos.default"
        | "torch.ops.aten.atan.default"
        | "torch.ops.aten.sinh.default"
        | "torch.ops.aten.cosh.default"
        | "torch.ops.aten.expm1.default"
        | "torch.ops.aten.log1p.default"
        | "torch.ops.aten.trunc.default"
        // Bitwise / logical
        | "torch.ops.aten.bitwise_and.Tensor"
        | "torch.ops.aten.bitwise_or.Tensor"
        | "torch.ops.aten.bitwise_not.default"
        // NaN / Inf predicates
        | "torch.ops.aten.isnan.default"
        | "torch.ops.aten.isinf.default"
        | "torch.ops.aten.isfinite.default"
        // Clamp tensor variants
        | "torch.ops.aten.clamp_min.Tensor"
        | "torch.ops.aten.clamp_max.Tensor"
        // Masked fill / scatter extended
        | "torch.ops.aten.masked_fill.Tensor"
        | "torch.ops.aten.masked_fill_.Tensor"
        | "torch.ops.aten.masked_scatter.default"
        | "torch.ops.aten.masked_scatter_.default"
        | "torch.ops.aten.scatter_.src"
        | "torch.ops.aten.scatter_.reduce"
        | "torch.ops.aten.scatter_reduce.two"
        // Index extended
        | "torch.ops.aten.index_copy.default"
        | "torch.ops.aten.index_copy_.default"
        | "torch.ops.aten.index_fill.int_Scalar"
        | "torch.ops.aten.index_fill_.int_Scalar"
        // Where variants
        | "torch.ops.aten.where.ScalarOther"
        | "torch.ops.aten.where.ScalarSelf"
        // Shape ops extended
        | "torch.ops.aten.broadcast_to.default"
        | "torch.ops.aten.expand_as.default"
        | "torch.ops.aten.diagonal.default"
        | "torch.ops.aten.tile.default"
        | "torch.ops.aten.rot90.default"
        | "torch.ops.aten.channel_shuffle.default"
        // Pooling extended
        | "torch.ops.aten.adaptive_max_pool1d.default"
        // Tensor creation extended
        | "torch.ops.aten.arange.start"
        | "torch.ops.aten.arange.start_stop"
        | "torch.ops.aten.linspace.out"
        | "torch.ops.aten.eye.default"
        | "torch.ops.aten.eye.m"
        | "torch.ops.aten.affine_grid_generator.default"
        // Repeat interleave (int variant)
        | "torch.ops.aten.repeat_interleave.self_int"
        // Padding extended
        | "torch.ops.aten.replication_pad1d.default"
        | "torch.ops.aten.replication_pad2d.default"
        // Upsampling extended
        | "torch.ops.aten.upsample_bicubic2d.default"
        | "torch.ops.aten.upsample_bicubic2d.vec"
        // In-place mask / triu / tril
        | "torch.ops.aten.triu_.default"
        | "torch.ops.aten.tril_.default"
        // Loss functions (for training import)
        | "torch.ops.aten.cross_entropy_loss.default"
        | "torch.ops.aten.cross_entropy_loss.label_smoothing"
        | "torch.ops.aten.nll_loss.default"
        | "torch.ops.aten.nll_loss_nd.default"
        | "torch.ops.aten.nll_loss_forward.default"
        | "torch.ops.aten.nll_loss2d_forward.default"
        | "torch.ops.aten.mse_loss.default"
        | "torch.ops.aten.mse_loss_backward.default"
        | "torch.ops.aten.l1_loss.default"
        | "torch.ops.aten.l1_loss_backward.default"
        | "torch.ops.aten.smooth_l1_loss.default"
        | "torch.ops.aten.smooth_l1_loss_backward.default"
        | "torch.ops.aten.huber_loss.default"
        | "torch.ops.aten.binary_cross_entropy.default"
        | "torch.ops.aten.binary_cross_entropy.weight"
        | "torch.ops.aten.binary_cross_entropy_with_logits.default"
        | "torch.ops.aten.kl_div.default"
        | "torch.ops.aten.kl_div_backward.default"
        // Wave 13: advanced indexing, scatter, gather, masking, sort/unique variants
        | "torch.ops.aten.index_put.hacked_twin"
        | "torch.ops.aten.index_put_.hacked_twin"
        | "torch.ops.aten.index_put.accumulate"
        | "torch.ops.aten.index_put_.accumulate"
        | "torch.ops.aten.scatter_.value_reduce"
        | "torch.ops.aten.scatter_add_.default"
        | "torch.ops.aten.gather.out"
        | "torch.ops.aten.index_select.out"
        | "torch.ops.aten.masked_fill.Tensor_Scalar"
        | "torch.ops.aten.masked_select.default"
        | "torch.ops.aten.masked_select.out"
        | "torch.ops.aten.nonzero.default"
        | "torch.ops.aten.nonzero.out"
        | "torch.ops.aten.topk.values"
        | "torch.ops.aten.sort.values"
        | "torch.ops.aten.sort.values_stable"
        | "torch.ops.aten._unique2.default"
        | "torch.ops.aten.unique_dim.default"
        | "torch.ops.aten.unique_consecutive.default"
        // Wave 14: elementwise ternary, norms, sampling, scan, one-hot, threshold
        | "torch.ops.aten.lerp.Scalar"
        | "torch.ops.aten.lerp.Tensor"
        | "torch.ops.aten.addcmul.default"
        | "torch.ops.aten.addcmul_.default"
        | "torch.ops.aten.addcdiv.default"
        | "torch.ops.aten.addcdiv_.default"
        | "torch.ops.aten.linalg_vector_norm.default"
        | "torch.ops.aten.cdist.default"
        | "torch.ops.aten.multinomial.default"
        | "torch.ops.aten.searchsorted.Tensor"
        | "torch.ops.aten.bucketize.Tensor"
        | "torch.ops.aten.count_nonzero.default"
        | "torch.ops.aten.count_nonzero.dim_IntList"
        | "torch.ops.aten.cumprod.default"
        | "torch.ops.aten.cumprod.int"
        | "torch.ops.aten.cummax.default"
        | "torch.ops.aten.cummin.default"
        | "torch.ops.aten.one_hot.default"
        | "torch.ops.aten.threshold.default"
        | "torch.ops.aten.threshold_.default"
    )
}

/// Populate verification coverage from the imported graph.
#[cfg(feature = "verify")]
fn populate_verification_coverage(
    imported: &crate::graph_build::ImportedGraph,
    coverage: &mut VerificationCoverage,
) {
    use nn_core::dyn_tensor::trace::TraceOp;

    // Count total compute layers (non-Input, non-Constant).
    let total = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .count();
    coverage.gamma_crown_layers_total = total;

    // Try translating to NY to count covered layers.
    let variable_input_count = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Input))
        .count();

    let translation_result = if variable_input_count > 1 {
        nn_verify::trace_to_graph_model_multi_input(&imported.graph).ok()
    } else {
        nn_verify::trace_to_graph_model(&imported.graph).ok()
    };

    if let Some(result) = translation_result {
        // Count the nodes in the translated NY graph.
        coverage.gamma_crown_layers_covered = result.graph.num_nodes();
    }
}

/// Stub when verify feature is not enabled.
#[cfg(not(feature = "verify"))]
fn populate_verification_coverage(
    _imported: &crate::graph_build::ImportedGraph,
    _coverage: &mut VerificationCoverage,
) {
    // No verification available without the feature.
}
