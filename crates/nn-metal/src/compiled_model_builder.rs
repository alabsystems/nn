// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder pattern for [`CompiledModel`] construction.
//!
//! Replaces 6+ `from_trace*` combinatorial constructors with a single
//! configurable builder. Part of #1815, #2218.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_core::{DType, Result, TensorError};
use nn_dsl::buffer_planner::plan_buffers_with_dtypes;
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::{
    compile_trace_to_plan_with_fusion, CompiledPlan, CompiledStep,
};

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;

use super::{build, CompiledModel, CompiledModelError};

#[path = "compiled_model_builder_classify.rs"]
mod classify;
use classify::{
    is_bandwidth_bound_dispatch, is_bandwidth_bound_linear, is_compute_native_op,
    is_non_gemm_compute_dispatch, is_passthrough_safe,
};

/// Builder for [`CompiledModel`]. Replaces 6+ `from_trace*` constructors.
///
/// # Examples
///
/// ```rust,no_run
/// // Simple (replaces from_trace):
/// let model = CompiledModel::builder(&graph, &cache).build()?;
///
/// // With shared weights (replaces from_trace_with_shared_weights):
/// let model = CompiledModel::builder(&graph, &cache)
///     .shared_weights(&weights)
///     .build()?;
///
/// // Per-op autocast + shared (replaces from_trace_autocast_with_shared_weights):
/// let model = CompiledModel::builder(&graph, &cache)
///     .shared_weights(&weights)
///     .autocast(policy)
///     .build()?;
/// ```
pub struct CompiledModelBuilder<'a> {
    graph: &'a ComputationGraph,
    cache: &'a PipelineCache,
    shared_weights: Option<&'a HashMap<(usize, String), MetalBuffer>>,
    mixed_precision: Option<ScalarType>,
    autocast_policy: Option<MixedPrecisionPolicy>,
    peephole_config: Option<nn_dsl::PeepholeConfig>,
    optimization_budget: Option<Duration>,
    shape_policy: super::ShapePolicy,
}

impl<'a> CompiledModelBuilder<'a> {
    pub(crate) fn new(graph: &'a ComputationGraph, cache: &'a PipelineCache) -> Self {
        Self {
            graph,
            cache,
            shared_weights: None,
            mixed_precision: None,
            autocast_policy: None,
            peephole_config: None,
            optimization_budget: None,
            shape_policy: super::ShapePolicy::default(),
        }
    }

    #[must_use]
    pub fn shared_weights(mut self, w: &'a HashMap<(usize, String), MetalBuffer>) -> Self {
        self.shared_weights = Some(w);
        self
    }

    /// Apply uniform dtype override to all non-LSTM, non-RuntimeOp steps.
    ///
    /// This is NOT mixed precision — it forces ALL eligible steps to `dtype`.
    /// For per-op mixed precision (some F16, some F32), use [`autocast()`](Self::autocast).
    ///
    /// Only GPU-compilable float dtypes (F32, F16, BF16) are accepted.
    /// Returns `Err` for non-compilable dtypes (U8, U32, I64, etc.).
    pub fn force_dtype(mut self, dtype: DType) -> Result<Self> {
        let scalar = ScalarType::try_from(dtype).map_err(|_| {
            TensorError::from(CompiledModelError::InvalidConfig {
                reason: format!("{dtype} is not a GPU-compilable dtype"),
            })
        })?;
        self.mixed_precision = Some(scalar);
        Ok(self)
    }

    #[deprecated(
        since = "0.1.0",
        note = "Renamed to force_dtype() — this mode applies uniform dtype, not mixed precision"
    )]
    pub fn mixed_precision(self, dtype: DType) -> Result<Self> {
        self.force_dtype(dtype)
    }

    #[must_use]
    pub fn autocast(mut self, policy: MixedPrecisionPolicy) -> Self {
        // f32_only() policy is a no-op — skip to prevent is_autocast()
        // returning true when nothing was actually autocasted. (#2981 D5)
        if policy.compute_dtype != DType::F32 {
            self.autocast_policy = Some(policy);
        }
        self
    }

    /// Compile with a specific [`PeepholeConfig`](nn_dsl::PeepholeConfig) instead of the default.
    /// Mutually exclusive with [`optimize()`](Self::optimize).
    #[must_use]
    pub fn with_peephole_config(mut self, config: nn_dsl::PeepholeConfig) -> Self {
        self.peephole_config = Some(config);
        self
    }

    /// Run exhaustive [`PeepholeConfig`](nn_dsl::PeepholeConfig) search within the given
    /// time budget. Finds the config that minimizes dispatch count + estimated cost.
    /// Mutually exclusive with [`with_peephole_config()`](Self::with_peephole_config).
    #[must_use]
    pub fn optimize(mut self, budget: Duration) -> Self {
        self.optimization_budget = Some(budget);
        self
    }

    /// Set the shape policy for this compiled model.
    ///
    /// `ShapePolicy::Fixed` (default): shapes baked at compile time.
    /// `ShapePolicy::Polymorphic { .. }`: sequence dimensions resolved at
    /// runtime, eliminating recompilation for variable-length TTS inputs.
    ///
    /// Part of #3873.
    #[must_use]
    pub fn shape_policy(mut self, policy: super::ShapePolicy) -> Self {
        self.shape_policy = policy;
        self
    }

    pub fn build(self) -> Result<CompiledModel> {
        let plan = if let Some(budget) = self.optimization_budget {
            // Run exhaustive search over 2048 PeepholeConfig combinations.
            let result = nn_dsl::optimize_plan_with_cost(
                self.graph,
                &nn_dsl::CostModel::apple_m4(),
                budget,
            )
            .map_err(|e| TensorError::from(CompiledModelError::CompileFailed(e)))?;
            result.plan
        } else if let Some(ref config) = self.peephole_config {
            // Compile with the specified PeepholeConfig.
            nn_dsl::compile_trace_to_plan_configured(self.graph, config)
                .map_err(|e| TensorError::from(CompiledModelError::CompileFailed(e)))?
        } else {
            // Default: all peephole passes enabled + fusion.
            compile_trace_to_plan_with_fusion(self.graph)
                .map_err(|e| TensorError::from(CompiledModelError::CompileFailed(e)))?
        };
        from_plan_inner(
            &plan,
            self.graph,
            self.cache,
            self.shared_weights,
            self.mixed_precision,
            self.autocast_policy,
            self.shape_policy,
        )
    }

    /// Build from a pre-compiled plan (skips trace compilation).
    pub(super) fn build_from_plan(self, plan: &CompiledPlan) -> Result<CompiledModel> {
        from_plan_inner(
            plan,
            self.graph,
            self.cache,
            self.shared_weights,
            self.mixed_precision,
            self.autocast_policy,
            self.shape_policy,
        )
    }
}

/// Internal builder shared by all construction paths.
fn from_plan_inner(
    plan: &CompiledPlan,
    graph: &ComputationGraph,
    cache: &PipelineCache,
    shared_weights: Option<&HashMap<(usize, String), MetalBuffer>>,
    mixed_precision: Option<ScalarType>,
    autocast_policy: Option<MixedPrecisionPolicy>,
    shape_policy: super::ShapePolicy,
) -> Result<CompiledModel> {
    if plan.steps.is_empty() {
        return Ok(CompiledModel::empty());
    }

    let mixed_precision_active = mixed_precision.is_some();
    let edge_map = build::build_edge_map(graph, &plan.steps)?;
    let num_inputs = plan.input_shapes.len();
    let input_specs: Vec<(Vec<usize>, DType)> = graph
        .input_nodes()
        .iter()
        .map(|n| (n.output_shape().to_vec(), n.output_dtype()))
        .collect();

    // Derive per-step ScalarType from graph node dtypes BEFORE upload,
    // so weight and constant buffers are created in the correct dtype.
    // Non-float dtypes (U32, I64) default to F32 since GPU dispatch
    // only handles float types. (#2339, #2273)
    let nodes = graph.nodes();
    let mut step_scalar_types: Vec<ScalarType> = nodes
        .iter()
        .map(|n| ScalarType::try_from(n.output_dtype()).unwrap_or(ScalarType::F32))
        .collect();
    // Invariant: plan.steps and graph.nodes() are 1:1 (maintained by
    // compile_trace, fusion IdentityPassthrough placeholders, and peephole).
    if plan.steps.len() != step_scalar_types.len() {
        return Err(TensorError::from(CompiledModelError::DispatchFailed {
            step_idx: 0,
            reason: format!(
                "plan.steps ({}) and graph.nodes() ({}) length mismatch",
                plan.steps.len(),
                step_scalar_types.len(),
            ),
        }));
    }

    // Mixed-precision: override steps to target dtype (F16).
    // LSTM stays F32 (sigmoid/tanh saturation at F16 range). D6.
    // RuntimeOp stays F32 (execute_runtime_op always produces F32). #3122.
    // All other NativeOps now have parameterized MSL (D4) and accept
    // F16 input (D5a), so they participate in mixed precision.
    if let Some(target) = mixed_precision {
        for (i, step) in plan.steps.iter().enumerate() {
            if matches!(step, CompiledStep::NativeOp { op, .. }
                if matches!(op, nn_dsl::NativeOpKind::LstmSequence { .. }))
            {
                continue; // LSTM stays F32
            }
            if matches!(step, CompiledStep::RuntimeOp { .. }) {
                continue; // RuntimeOp always produces F32 (#3122)
            }
            step_scalar_types[i] = target;
        }
    }

    // Per-op autocast (#3085, #2981): Compute-dominant steps get F16, but
    // GEMM ops (Linear/MatMul) only when simdgroup-eligible AND above the
    // TG threshold — low-occupancy GEMMs regress in F16. Non-GEMM compute
    // ops (Conv/Embedding/Attention) always benefit from F16. #3112 mutual
    // exclusivity: autocast converts only Compute steps; mixed_precision
    // converts ALL intermediates.
    if autocast_policy.is_some() && mixed_precision.is_some() {
        return Err(CompiledModelError::InvalidConfig {
            reason: "autocast_policy and mixed_precision are mutually exclusive".into(),
        }
        .into());
    }
    let (autocast_active, mixed_gemm_infos) =
        if autocast_policy.is_some() && mixed_precision.is_none() {
            let target = autocast_policy
                .as_ref()
                .and_then(|p| ScalarType::try_from(p.compute_dtype).ok())
                .unwrap_or(ScalarType::F16);
            // Extract mixed GEMM infos first — includes simdgroup eligibility
            // AND TG count threshold. Used to gate GEMM autocast below.
            let infos = build::extract_mixed_gemm_infos(&plan.steps);
            for (i, step) in plan.steps.iter().enumerate() {
                match step {
                    CompiledStep::Dispatch { kernel, .. } => {
                        // GEMM (Linear/MatMul): F16 if mixed GEMM eligible (simdgroup)
                        // OR bandwidth-bound (small M, large weights).
                        // Non-GEMM compute (Conv/Embedding/Attention): always F16.
                        // Part of #4264.
                        if infos[i].is_some()
                            || is_non_gemm_compute_dispatch(kernel.def())
                            || is_bandwidth_bound_dispatch(kernel.def())
                        {
                            step_scalar_types[i] = target;
                        }
                    }
                    CompiledStep::NativeOp { op, .. }
                        // LinearActivation: F16 if mixed GEMM eligible (simdgroup)
                        // OR bandwidth-bound (small M, large weights). The naive
                        // kernel uses F32 accumulators, so F16 weights are safe.
                        // FlashAttention: always F16 (F32 accumulators).
                        // Part of #4264.
                        if (infos[i].is_some()
                            || is_compute_native_op(op)
                            || is_bandwidth_bound_linear(op))
                        => {
                            step_scalar_types[i] = target;
                        }
                    _ => {}
                }
            }
            (true, infos)
        } else {
            (false, vec![None; plan.steps.len()])
        };

    // Forward-propagate F16 through passthrough-safe ops (activations,
    // binary elementwise, data-movement). Avoids F16→F32→F16 casts between
    // conv layers. Matches PyTorch's "implicit" autocast category. #2981.
    if autocast_active {
        let target = autocast_policy
            .as_ref()
            .and_then(|p| ScalarType::try_from(p.compute_dtype).ok())
            .unwrap_or(ScalarType::F16);
        for i in 0..plan.steps.len() {
            if step_scalar_types[i] != ScalarType::F32 {
                continue;
            }
            if !is_passthrough_safe(&plan.steps[i]) {
                continue;
            }
            let all_inputs_target = !edge_map[i].is_empty()
                && edge_map[i].iter().all(|&src| {
                    // Mixed GEMM steps have step_scalar_types = F16 but actually
                    // produce F32 output at runtime. Do not propagate from them.
                    step_scalar_types[src] == target && mixed_gemm_infos[src].is_none()
                });
            if all_inputs_target {
                step_scalar_types[i] = target;
            }
        }
    }

    let mut weight_buffers_flat = build::upload_weights(
        &plan.steps,
        &step_scalar_types,
        cache.context(),
        shared_weights,
    )?;
    // Pre-compute combined LSTM biases at build time. Saves 1 GPU dispatch
    // per LSTM step per forward pass (2 total in Kokoro). Part of #3291.
    build::precompute_lstm_combined_biases(
        &plan.steps,
        &step_scalar_types,
        &mut weight_buffers_flat,
        cache.context(),
    )?;
    // Convert flat HashMap<(step_idx, name), buf> to Vec<HashMap<name, buf>>
    // for O(1) step-indexed lookup without String allocation on the hot path.
    let weight_buffers = build::flat_weights_to_indexed(weight_buffers_flat, plan.steps.len());

    // Pre-upload constant values to GPU once at construction time.
    // Without this, ConstantValue steps allocate fresh CPU→GPU buffers
    // on every forward pass (#2338).
    let constant_buffers =
        build::upload_constants(&plan.steps, &step_scalar_types, cache.context())?;

    // Strip CPU-side weight data from cloned steps. Weight data has been
    // uploaded to GPU in weight_buffers; retaining it in steps wastes
    // memory proportional to total model parameters (~4 bytes per param).
    // For a 100M-param model this saves ~400MB.
    let mut steps = plan.steps.clone();
    for step in &mut steps {
        match step {
            CompiledStep::Dispatch { weight_data, .. }
            | CompiledStep::NativeOp { weight_data, .. } => {
                weight_data.clear();
            }
            _ => {}
        }
    }

    // Resolve output nodes to step indices and metadata.
    // `nodes` already bound above for step_scalar_types.
    let (output_step_indices, output_metas) = {
        let out_nodes = graph.output_nodes();
        if out_nodes.is_empty() {
            // No explicit mark_output -- fall back to last node.
            let last = nodes
                .last()
                .ok_or_else(|| TensorError::from(CompiledModelError::MissingOutputNode))?;
            (
                vec![nodes.len() - 1],
                vec![(last.output_shape().to_vec(), last.output_dtype())],
            )
        } else {
            let id_to_idx: HashMap<u64, usize> =
                nodes.iter().enumerate().map(|(i, n)| (n.id(), i)).collect();
            let mut indices = Vec::with_capacity(out_nodes.len());
            let mut metas = Vec::with_capacity(out_nodes.len());
            for n in &out_nodes {
                let &idx = id_to_idx
                    .get(&n.id())
                    .ok_or_else(|| TensorError::from(CompiledModelError::MissingOutputNode))?;
                indices.push(idx);
                metas.push((n.output_shape().to_vec(), n.output_dtype()));
            }
            (indices, metas)
        }
    };

    // Buffer planner sizes by stored dtype. LSTM exception: step keeps F32
    // for compute, but planner uses target (F16) for downstream sizing.
    let buffer_plan = if let Some(target) = mixed_precision {
        let planner_dtypes: Vec<ScalarType> = vec![target; plan.steps.len()];
        plan_buffers_with_dtypes(plan, graph, &planner_dtypes)
    } else if autocast_active {
        // Mixed GEMM produces F32 output despite F16 step type — override planner (#2981).
        let mut planner_dtypes = step_scalar_types.clone();
        for (i, info) in mixed_gemm_infos.iter().enumerate() {
            if info.is_some() {
                planner_dtypes[i] = ScalarType::F32;
            }
        }
        plan_buffers_with_dtypes(plan, graph, &planner_dtypes)
    } else {
        plan_buffers_with_dtypes(plan, graph, &step_scalar_types)
    };

    // Validate that the buffer planner's last_use keeps buffers alive for all
    // direct-access NativeOp dependencies (FusedResBlock). Catches divergence
    // between the two edge_map builders at construction time. Part of #3117.
    build::validate_buffer_plan_edges(&plan.steps, &buffer_plan.last_use)?;

    // Pre-compute input names once at build time (#2501).
    let input_name_cache: Vec<Vec<String>> = steps
        .iter()
        .map(|step| match step {
            CompiledStep::Dispatch { kernel, .. } => build::def_input_names(kernel.def()),
            _ => Vec::new(),
        })
        .collect();

    // Pre-compute release map once at construction time. `release_at[j]`
    // lists step indices whose last consumer is step j (excluding output
    // steps which must be preserved). Eliminates ~300 Vec allocations
    // + 1 HashSet per forward pass. (#2944)
    let release_at = {
        let n = buffer_plan.last_use.len();
        let mut map: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
        for (step, &consumer) in buffer_plan.last_use.iter().enumerate() {
            if consumer > step && consumer < n && !output_step_indices.contains(&step) {
                map[consumer].push(step);
            }
        }
        map
    };

    // Per-step element counts for F16↔F32 casts (planned buffer has shared allocation size).
    let step_numels: Vec<usize> = nodes
        .iter()
        .enumerate()
        .map(|(step_idx, node)| {
            node.output_shape()
                .iter()
                .try_fold(1usize, |acc, &dim| {
                    acc.checked_mul(dim).ok_or_else(|| {
                        TensorError::from(CompiledModelError::InvalidConfig {
                            reason: format!(
                                "output shape overflow at step {step_idx} node '{}' (id {}): {:?}",
                                node.name(),
                                node.id(),
                                node.output_shape(),
                            ),
                        })
                    })
                })
        })
        .collect::<Result<Vec<_>>>()?;

    // Consolidated per-step metadata (#1815, #3295).
    let step_metas: Vec<super::StepMeta> = (0..steps.len())
        .map(|i| super::StepMeta {
            edges: edge_map[i].clone(),
            scalar_type: step_scalar_types[i],
            numel: step_numels[i],
        })
        .collect();

    // ICB eligibility + concurrent barriers (Part of #3259 D1, #3426 autocast).
    let icb_eligible = super::icb::analyze_icb_eligibility(
        &steps,
        &step_metas,
        &mixed_gemm_infos,
        autocast_active,
        mixed_precision_active,
    );
    let gpu_dispatches = super::icb::analyze_gpu_dispatch_steps(&steps);
    let concurrent_barriers = super::icb::compute_concurrent_barriers(
        &edge_map,
        &buffer_plan.step_offsets,
        &gpu_dispatches,
    );

    // ICB pre-compilation: cache codegen outputs for eligible segments.
    // Autocast models have eligible steps via static dtype analysis (#3426).
    // Segment starts wired to execution loop for ICB replay.
    let has_eligible = icb_eligible.iter().any(|&e| e);
    let (icb_segments, icb_segment_starts) = if has_eligible && !mixed_precision_active {
        super::icb::pre_compile_icb_segments(&steps, &icb_eligible, &step_scalar_types, None)
    } else {
        (Vec::new(), HashMap::new())
    };

    // Verified by default: attempt verification if feature is enabled.
    // Operates on the F32 computation graph regardless of mixed_precision flag.
    // The certificate proves F32 bounds; F16 quantization gap is tracked by #3023.
    #[cfg(feature = "verify")]
    let proof_certificate = CompiledModel::try_auto_verify(graph);
    #[cfg(not(feature = "verify"))]
    let proof_certificate = None;

    let num_icb_segments = icb_segments.len();
    let def = super::CompiledModelDef {
        steps,
        step_metas,
        weight_buffers,
        constant_buffers,
        num_inputs,
        input_specs,
        output_step_indices,
        output_metas,
        buffer_plan,
        precision: None,
        input_name_cache,
        release_at,
        mixed_precision_active,
        autocast_policy,
        autocast_active,
        mixed_gemm_infos,
        proof_certificate,
        shape_policy,
        icb_eligible,
        icb_segments,
        icb_segment_starts,
        concurrent_barriers,
    };
    Ok(CompiledModel {
        def: Arc::new(def),
        cached_planned_buf: std::cell::RefCell::new(None),
        cached_icbs: std::cell::RefCell::new((0..num_icb_segments).map(|_| None).collect()),
    })
}

#[cfg(test)]
#[path = "compiled_model_builder_tests.rs"]
mod compiled_model_builder_tests;
