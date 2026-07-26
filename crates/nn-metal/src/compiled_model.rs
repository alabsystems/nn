// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pre-compiled model executor for trace-based graph execution.
//!
//! Takes a `ComputationGraph` from `trace_graph()`, compiles it into
//! `Vec<CompiledStep>` via `compile_trace_with_fusion()` (elementwise
//! chain fusion enabled), uploads weight data to GPU, and executes the
//! plan with minimal per-forward overhead.
//!
//! # Usage
//!
//! ```rust,no_run
//! use nn_metal::compiled_model::CompiledModel;
//!
//! // First execution: trace and compile
//! let (output, graph) = nn_core::dyn_tensor::trace::trace_graph(|| {
//!     model.forward(&input)
//! })?;
//! let compiled = CompiledModel::builder(&graph, &cache).build()?;
//!
//! // Subsequent executions: DynTensor in, DynTensor out
//! let result = compiled.execute_dyn(&cache, &[&input])?;
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_core::{DType, Result};
use nn_dsl::buffer_planner::BufferPlan;
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::CompiledStep;
use nn_dsl::PrecisionContract;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;

#[path = "compiled_model_build.rs"]
pub(crate) mod build;
#[path = "compiled_model_builder.rs"]
mod builder;
pub use builder::CompiledModelBuilder;
#[path = "compiled_model_shape_policy.rs"]
mod shape_policy;
pub use shape_policy::{ShapePolicy, ShapePolicyError};
#[path = "compiled_model_dyn.rs"]
mod dyn_iface;
#[path = "compiled_model_error.rs"]
pub(crate) mod error;
pub use error::CompiledModelError;
#[path = "compiled_model_convenience.rs"]
mod convenience;
#[path = "compiled_model_dtype_tracker.rs"]
pub(crate) mod dtype_tracker;
#[path = "compiled_model_execute.rs"]
mod execute;
#[path = "compiled_model_icb.rs"]
pub(crate) mod icb;
#[path = "mixed_gemm_msl.rs"]
pub(crate) mod mixed_gemm_msl;
#[path = "int8_gemm_msl.rs"]
pub(crate) mod int8_gemm_msl;
#[path = "compiled_model_profile.rs"]
pub mod profile;
#[path = "compiled_model_query.rs"]
mod query;
#[cfg(feature = "verify")]
#[path = "compiled_model_verify.rs"]
mod verify;

/// Pre-computed parameters for mixed-precision simdgroup GEMM dispatch.
///
/// When a compiled step is simdgroup-eligible in autocast mode, this struct
/// holds the M, K, N dimensions and layout flags needed to generate the
/// `simd_gemm_mixed` MSL kernel (F32 activations × F16 weights).
/// Populated at build time for Dispatch (Linear/MatMul) and NativeOp
/// (LinearActivation) steps.
///
/// Part of #3085, #2981.
#[derive(Debug, Clone)]
pub struct MixedGemmInfo {
    /// Rows in the activation matrix (product of leading batch dims).
    pub m: usize,
    /// Contracted dimension (input features for Linear, inner dim for MatMul).
    pub k: usize,
    /// Columns in the output (output features for Linear).
    pub n: usize,
    /// Number of batches (for batched matmul; 1 for simple linear).
    pub batch_count: usize,
    /// Whether weight B is stored transposed (true for Linear: `[N, K]`).
    pub transpose_b: bool,
    /// Whether weight B is broadcast across batches.
    pub broadcast_b: bool,
    /// Whether the step has a bias vector.
    pub has_bias: bool,
    /// Optional fused activation epilogue (from `NativeOp::LinearActivation`).
    /// `None` for plain Linear/MatMul Dispatch steps.
    pub activation: Option<nn_dsl::GemmActivation>,
}

/// Per-step metadata, built once at construction time.
///
/// Consolidates parallel per-step vectors from `CompiledModel` into a
/// single struct indexed by `step_idx`. Fields are migrated here
/// incrementally from the parallel vecs on `CompiledModel`.
///
/// See `designs/2026-03-22-compiled-model-step-metadata-consolidation.md`.
/// Part of #1815, #3295.
#[derive(Debug, Clone)]
pub(crate) struct StepMeta {
    /// Graph edges: step indices of this step's inputs.
    pub edges: Vec<usize>,
    /// Scalar type (F32/F16/BF16) derived from graph node dtype.
    pub scalar_type: ScalarType,
    /// Element count (product of output shape dimensions).
    pub numel: usize,
}

/// Immutable, shareable model definition.
///
/// Contains all the data that can be shared across multiple execution
/// instances via `Arc`. Separated from execution-time mutable caches
/// (`RefCell` fields) to enable `Arc<CompiledModelDef>` sharing in the
/// chorus system.
///
/// Not `pub` — callers interact through [`CompiledModel`] which wraps
/// this in an `Arc` alongside per-instance execution state.
#[derive(Debug)]
pub(crate) struct CompiledModelDef {
    /// Compiled steps in topological order.
    pub(crate) steps: Vec<CompiledStep>,
    /// Per-step metadata consolidating edges, scalar_type, numel.
    /// Part of #1815, #3295.
    pub(crate) step_metas: Vec<StepMeta>,
    /// Pre-uploaded weight buffers indexed by step. `weight_buffers[step_idx]`
    /// is a map from weight name to GPU buffer. Indexed access eliminates
    /// `(usize, String)` key allocation on the hot path (~1200 String clones
    /// per Kokoro forward pass). Construction-time only: populated once from
    /// `upload_weights()`. (#2501)
    pub(crate) weight_buffers: Vec<HashMap<String, MetalBuffer>>,
    /// Pre-uploaded constant buffers keyed by step index.
    /// Uploaded once at construction; reused (aliased) every forward pass.
    pub(crate) constant_buffers: HashMap<usize, MetalBuffer>,
    /// Number of input nodes (TraceOp::Input) in the graph.
    pub(crate) num_inputs: usize,
    /// Expected shape and dtype for each input, in order.
    pub(crate) input_specs: Vec<(Vec<usize>, DType)>,
    /// Step indices of output nodes in the compiled plan.
    /// For single-output models, this has one entry (the last step).
    pub(crate) output_step_indices: Vec<usize>,
    /// Shape and dtype for each output, ordered to match `output_step_indices`.
    pub(crate) output_metas: Vec<(Vec<usize>, DType)>,
    /// Static buffer allocation plan for memory estimation.
    pub(crate) buffer_plan: BufferPlan,
    /// Optional precision contract for all dispatch steps.
    /// When `Some`, uses Kahan-compensated reductions for normalization ops.
    /// Set via [`with_precision()`](CompiledModel::with_precision).
    pub(crate) precision: Option<PrecisionContract>,
    /// Cached input names per dispatch step, computed once at build time.
    /// Indexed by step_idx; non-Dispatch steps have empty vecs.
    /// Eliminates per-forward-pass IR node scanning and ~1200 heap
    /// allocations per Kokoro inference (~300 steps × 4 allocs/step). (#2501)
    pub(crate) input_name_cache: Vec<Vec<String>>,
    /// Pre-computed release map: `release_at[j]` lists step indices whose
    /// last consumer is step `j` and are not output steps. Built once at
    /// construction from `buffer_plan.last_use` and `output_step_indices`.
    /// Eliminates ~300 Vec allocations + 1 HashSet per forward pass. (#2944)
    pub(crate) release_at: Vec<Vec<usize>>,
    /// When true, F16 mixed-precision is active: Dispatch steps use F16,
    /// NativeOp steps use F32 with auto-casting at boundaries.
    /// Set by `builder().force_dtype()` (formerly `from_trace_f16()`).
    pub(crate) mixed_precision_active: bool,
    /// Per-op autocast policy. When `Some`, each op's dtype is determined
    /// by [`OpDTypeCategory`](nn_core::mixed_precision::OpDTypeCategory):
    /// Compute ops use F16 buffers, passthrough ops inherit F16 from predecessors,
    /// Accumulate ops (softmax, norms) stay F32. Mixed GEMM steps use F16 weights
    /// but produce F32 output (F32 accumulators). Part of #3085.
    pub(crate) autocast_policy: Option<MixedPrecisionPolicy>,
    /// When true, per-op autocast is active: Compute and passthrough Dispatch
    /// steps use F16, Accumulate steps (softmax, norms) stay F32.
    /// Boundary casts inserted automatically. Part of #3085.
    pub(crate) autocast_active: bool,
    /// Per-step mixed GEMM info for autocast Phase 2. When `Some`, the step
    /// bypasses normal IR dispatch and uses the `simd_gemm_mixed` kernel with
    /// F32 activations × F16 weights → F32 output. `None` for non-GEMM steps
    /// or steps below simdgroup eligibility threshold. Part of #3085.
    pub(crate) mixed_gemm_infos: Vec<Option<MixedGemmInfo>>,
    /// Opaque proof certificate JSON, populated automatically when the `verify`
    /// feature is enabled. `None` when verification is disabled, unsupported,
    /// or fails. Check with `.proof_certificate_json()`. Part of #3042.
    pub(crate) proof_certificate: Option<String>,
    /// Per-step ICB eligibility. `true` when the step can be pre-encoded
    /// into a Metal Indirect Command Buffer for batch replay.
    /// Computed once at build time. Part of #3206.
    #[allow(dead_code)] // ICB wiring in progress (#3259)
    pub(crate) icb_eligible: Vec<bool>,
    /// Shape policy controlling fixed vs polymorphic shape dispatch.
    /// When `Polymorphic`, input validation allows sequence dimension
    /// variance and output shapes are computed from actual inputs.
    /// Part of #3873.
    pub(crate) shape_policy: ShapePolicy,
    /// Pre-encoded ICB segments. Each segment covers a contiguous run of
    /// ICB-eligible steps. Empty when autocast/mixed-precision is active
    /// or when no eligible segments exist. Part of #3259.
    pub(crate) icb_segments: Vec<icb::IcbSegment>,
    /// O(1) lookup: step_idx → index into `icb_segments`.
    /// Only populated for the first step of each segment.
    pub(crate) icb_segment_starts: HashMap<usize, usize>,
    /// Per-step barrier requirement for concurrent dispatch.
    /// `true` when a memory barrier must be inserted before this step.
    /// Computed once at build time from `edge_map` + `buffer_plan.step_offsets`.
    /// Part of #3258.
    #[allow(dead_code)] // ICB wiring in progress (#3259)
    pub(crate) concurrent_barriers: Vec<bool>,
}

/// A pre-compiled model execution plan.
///
/// Holds the compiled dispatch steps and pre-uploaded GPU weight buffers.
/// Weight data is uploaded to GPU once at construction; subsequent
/// `execute()` calls reuse the GPU buffers with zero IR rebuild.
///
/// Internally split into a shareable [`CompiledModelDef`] (immutable model
/// definition) and per-instance execution caches (`RefCell` fields).
/// Use [`share_def()`](Self::share_def) to obtain an `Arc` for cross-instance
/// sharing, and [`from_shared()`](Self::from_shared) to create new execution
/// instances from a shared definition.
pub struct CompiledModel {
    /// Shareable model definition (immutable after construction).
    pub(crate) def: Arc<CompiledModelDef>,
    /// Cached contiguous GPU buffer for `BufferPlan` sub-allocation.
    /// Allocated once on first `run_steps()` call and reused on subsequent
    /// calls. Output data is always blit-copied out by
    /// `normalize_output_to_offset_zero` before `run_steps` returns, so
    /// reusing the buffer across forward passes is safe.
    cached_planned_buf: RefCell<Option<MetalBuffer>>,
    /// Lazily created ICBs, one per segment. `None` until first forward pass.
    /// Part of #3259 (D3).
    cached_icbs: RefCell<Vec<Option<icb::IndirectCommandBuffer>>>,
}

impl CompiledModel {
    /// Create an empty model with no steps, used for empty computation graphs.
    pub(crate) fn empty() -> Self {
        let def = CompiledModelDef {
            steps: Vec::new(),
            step_metas: Vec::new(),
            weight_buffers: Vec::new(),
            constant_buffers: HashMap::new(),
            num_inputs: 0,
            input_specs: Vec::new(),
            output_step_indices: Vec::new(),
            output_metas: Vec::new(),
            buffer_plan: BufferPlan {
                total_bytes: 0,
                step_offsets: Vec::new(),
                step_sizes: Vec::new(),
                naive_total: 0,
                last_use: Vec::new(),
            },
            precision: None,
            input_name_cache: Vec::new(),
            release_at: Vec::new(),
            mixed_precision_active: false,
            autocast_policy: None,
            autocast_active: false,
            mixed_gemm_infos: Vec::new(),
            proof_certificate: None,
            shape_policy: ShapePolicy::Fixed,
            icb_eligible: Vec::new(),
            icb_segments: Vec::new(),
            icb_segment_starts: HashMap::new(),
            concurrent_barriers: Vec::new(),
        };
        Self {
            def: Arc::new(def),
            cached_planned_buf: RefCell::new(None),
            cached_icbs: RefCell::new(Vec::new()),
        }
    }

    /// Returns a clone of the shared model definition `Arc`.
    ///
    /// Use this to share the immutable model definition across multiple
    /// execution instances (e.g., in the chorus system). Each instance
    /// maintains its own execution caches (planned buffer, ICBs).
    #[must_use]
    pub(crate) fn share_def(&self) -> Arc<CompiledModelDef> {
        Arc::clone(&self.def)
    }

    /// Create a new execution instance from a shared model definition.
    ///
    /// The new instance shares all immutable state (steps, weights, buffers)
    /// with other instances created from the same `Arc<CompiledModelDef>`,
    /// but has its own execution caches (planned buffer, ICBs).
    #[must_use]
    pub(crate) fn from_shared(def: Arc<CompiledModelDef>) -> Self {
        let num_icb_segments = def.icb_segments.len();
        Self {
            def,
            cached_planned_buf: RefCell::new(None),
            cached_icbs: RefCell::new((0..num_icb_segments).map(|_| None).collect()),
        }
    }

    /// Create a builder for configuring model compilation options.
    pub fn builder<'a>(
        graph: &'a ComputationGraph,
        cache: &'a PipelineCache,
    ) -> CompiledModelBuilder<'a> {
        CompiledModelBuilder::new(graph, cache)
    }

    /// Compile a traced computation graph into a pre-built execution plan.
    ///
    /// Convenience wrapper: calls [`compile_trace_to_plan_with_fusion`] then
    /// builds via [`builder`](Self::builder).
    #[deprecated(
        since = "0.1.0",
        note = "Use CompiledModel::builder(graph, cache).build() instead"
    )]
    pub fn from_trace(graph: &ComputationGraph, cache: &PipelineCache) -> Result<Self> {
        Self::builder(graph, cache).build()
    }

    /// Compile a traced graph, reusing pre-uploaded GPU weight buffers (#2630).
    /// Aliases buffers from `shared`; uploads fresh for weights not in `shared`.
    #[deprecated(
        since = "0.1.0",
        note = "Use CompiledModel::builder(graph, cache).shared_weights(w).build() instead"
    )]
    pub fn from_trace_with_shared_weights(
        graph: &ComputationGraph,
        cache: &PipelineCache,
        shared: &HashMap<(usize, String), MetalBuffer>,
    ) -> Result<Self> {
        Self::builder(graph, cache).shared_weights(shared).build()
    }

    /// Compile with shared weights and mixed-precision dtype override.
    /// Non-NativeOp steps use `precision_dtype`; NativeOps stay F32.
    #[deprecated(
        since = "0.1.0",
        note = "Use CompiledModel::builder(..).shared_weights(w).force_dtype(dt).build() instead"
    )]
    pub fn from_trace_with_shared_weights_f16(
        graph: &ComputationGraph,
        cache: &PipelineCache,
        shared: &HashMap<(usize, String), MetalBuffer>,
        precision_dtype: DType,
    ) -> Result<Self> {
        Self::builder(graph, cache)
            .shared_weights(shared)
            .force_dtype(precision_dtype)?
            .build()
    }

    /// Compile a traced graph with mixed-precision dtype override.
    /// Non-NativeOp steps use `precision_dtype`; NativeOps stay F32.
    #[deprecated(
        since = "0.1.0",
        note = "Use CompiledModel::builder(graph, cache).force_dtype(dt).build() instead"
    )]
    pub fn from_trace_f16(
        graph: &ComputationGraph,
        cache: &PipelineCache,
        precision_dtype: DType,
    ) -> Result<Self> {
        Self::builder(graph, cache)
            .force_dtype(precision_dtype)?
            .build()
    }

    /// Compile with per-op autocast mixed precision.
    ///
    /// Compute-dominant ops (matmul, linear, conv, embedding) dispatch with
    /// F16 weights for bandwidth savings; numerically sensitive ops (softmax,
    /// norms, reductions) stay F32 for correctness. The `policy` controls
    /// dtype selection per op category. Part of #2981.
    #[deprecated(
        since = "0.1.0",
        note = "Use CompiledModel::builder(graph, cache).autocast(policy).build() instead"
    )]
    pub fn from_trace_autocast(
        graph: &ComputationGraph,
        cache: &PipelineCache,
        policy: MixedPrecisionPolicy,
    ) -> Result<Self> {
        Self::builder(graph, cache).autocast(policy).build()
    }

    /// Compile with per-op autocast and shared weight buffers.
    #[deprecated(
        since = "0.1.0",
        note = "Use CompiledModel::builder(..).shared_weights(w).autocast(policy).build() instead"
    )]
    pub fn from_trace_autocast_with_shared_weights(
        graph: &ComputationGraph,
        cache: &PipelineCache,
        shared: &HashMap<(usize, String), MetalBuffer>,
        policy: MixedPrecisionPolicy,
    ) -> Result<Self> {
        Self::builder(graph, cache)
            .shared_weights(shared)
            .autocast(policy)
            .build()
    }

    /// Build a `CompiledModel` from a pre-compiled [`CompiledPlan`] and graph.
    ///
    /// The `graph` is still needed for edge_map construction and output metadata.
    /// The `plan.steps` are used directly (including any fusion already applied).
    pub fn from_plan(
        plan: &nn_dsl::trace_compile::CompiledPlan,
        graph: &ComputationGraph,
        cache: &PipelineCache,
    ) -> Result<Self> {
        Self::builder(graph, cache).build_from_plan(plan)
    }

    // -- Verify/certify constructors extracted to `compiled_model_verify.rs` --

    /// Create aliased copies of all pre-uploaded weight buffers.
    ///
    /// Each alias shares the same GPU allocation via ARC — zero-copy.
    /// Used by [`SegmentCache`](super::compiled_kokoro::segment_cache) to
    /// share weights across shape variants of the same model segment (#2630).
    /// Excludes traced `ConstantWeight` buffers because those may encode
    /// shape-dependent helper tensors (#3507).
    pub fn weight_buffer_aliases(&self) -> HashMap<(usize, String), MetalBuffer> {
        let mut out = HashMap::new();
        for (step_idx, (step, step_weights)) in self
            .def
            .steps
            .iter()
            .zip(self.def.weight_buffers.iter())
            .enumerate()
        {
            if !build::shares_weight_buffers(step) {
                continue;
            }
            for (name, buf) in step_weights {
                out.insert((step_idx, name.clone()), buf.alias());
            }
        }
        out
    }

    // -- Convenience constructors and cert access in `compiled_model_convenience.rs` --
    // -- Execution methods extracted to `compiled_model_execute.rs` --
    // -- Query/inspection methods extracted to `compiled_model_query.rs` --
}

#[cfg(test)]
#[path = "compiled_model_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "compiled_model_def_tests.rs"]
mod def_tests;
