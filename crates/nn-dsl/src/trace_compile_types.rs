// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Type definitions for compiled trace execution plans.
//!
//! Extracted from `trace_compile.rs` to keep the parent module under 450 lines.
//! Contains: [`CompiledKernel`], [`RuntimeOpKind`], [`CompiledStep`], and
//! [`CompiledPlan`]. Native op types (`NativeOpKind`, `NormActivConv1dParams`,
//! `NormActivation`, `StyleProjectionParams`) are in `trace_compile_native_ops.rs`.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{NodeId, WeightRef};

use crate::tensor_ir::{TensorKernelDef, TensorOpKind};

use super::NativeOpKind;

/// Opaque wrapper around [`TensorKernelDef`] for `CompiledStep::Dispatch`.
///
/// Provides a stable API surface for downstream consumers (e.g., dvoice)
/// without exposing the internal IR representation. Internal dispatch code
/// can access the underlying definition via [`def()`](Self::def).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompiledKernel {
    inner: TensorKernelDef,
}

impl CompiledKernel {
    /// Create a new `CompiledKernel` wrapping a `TensorKernelDef`.
    #[must_use]
    pub fn new(def: TensorKernelDef) -> Self {
        Self { inner: def }
    }

    /// Human-readable kernel name (e.g., "linear", "conv1d", "gelu").
    #[must_use]
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Extract the input tensor names from the kernel IR.
    ///
    /// Returns the names of `TensorOpKind::Input` nodes in the order they
    /// appear in the IR graph.
    #[must_use]
    pub fn input_names(&self) -> Vec<String> {
        self.inner
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                TensorOpKind::Input { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// Output shape of the kernel (shape of the output node).
    #[must_use]
    pub fn output_shape(&self) -> Option<&[usize]> {
        let out_id = self.inner.output;
        self.inner
            .nodes
            .iter()
            .find(|n| n.id == out_id)
            .map(|n| n.shape.as_slice())
    }

    /// Access the underlying `TensorKernelDef` for GPU dispatch.
    ///
    /// This is the escape hatch for the executor -- it needs the full IR
    /// to build dispatch plans and generate MSL. Downstream consumers
    /// (dvoice) should prefer the stable accessor methods above.
    #[must_use]
    pub fn def(&self) -> &TensorKernelDef {
        &self.inner
    }
}

/// Runtime operations whose output shape depends on input data.
///
/// These are executed eagerly by the compiled model executor because
/// the output size cannot be determined at compile time (e.g., Kokoro's
/// `length_regulate` uses RepeatInterleave with duration-predictor counts).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum RuntimeOpKind {
    /// RepeatInterleave with data-dependent repeat counts.
    ///
    /// Input 0: the tensor to repeat (shape from `input_shape`).
    /// Input 1: the repeat counts (1D f32 tensor, shape from `counts_shape`).
    /// Output shape is `[..., sum(counts), ...]` along `dim` — data-dependent.
    RepeatInterleave {
        /// Dimension along which to repeat.
        dim: usize,
        /// Shape of input 0 (the tensor being repeated).
        input_shape: Vec<usize>,
        /// Shape of input 1 (the repeat counts tensor).
        counts_shape: Vec<usize>,
    },
}

/// A single step in a compiled model execution plan.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum CompiledStep {
    /// A GPU-dispatchable kernel with pre-built dispatch plan.
    Dispatch {
        /// The compiled kernel (opaque wrapper around `TensorKernelDef`).
        kernel: CompiledKernel,
        /// Weight name → captured weight data from the trace.
        /// The executor binds these to Metal buffers at runtime.
        weight_data: HashMap<String, WeightRef>,
        /// For fused steps: the graph `NodeId`s of external inputs, in the
        /// order the kernel expects them. `None` for non-fused dispatches
        /// (the executor falls back to the edge_map).
        external_node_ids: Option<Vec<NodeId>>,
    },
    /// A shape-only operation (reshape, unsqueeze, squeeze) handled by
    /// metadata manipulation, not GPU dispatch.
    Passthrough {
        /// Human-readable op name for diagnostics.
        op_name: String,
        /// Output shape after the passthrough.
        output_shape: Vec<usize>,
    },
    /// A zero-copy narrow view: byte offset into the input buffer.
    ///
    /// The buffer planner resolves this as zero-allocation — the output
    /// buffer starts at `input_byte_offset + narrow_byte_offset`. Part of #2780.
    NarrowView {
        /// Byte offset from the start of the input buffer to the narrow slice.
        byte_offset: usize,
        /// Output shape after narrowing.
        output_shape: Vec<usize>,
        /// Optional explicit source step index for batched projection narrow.
        ///
        /// When `Some(idx)`, the executor reads from `buffers[idx]` directly
        /// instead of resolving via edge_map. Used by peephole pass 10
        /// (BatchedLinearProjection) to narrow individual Q/K/V slices from
        /// a concatenated matmul output. Part of #3269.
        #[serde(default)]
        source_step: Option<usize>,
    },
    /// A graph input node — consumes the next entry from the external
    /// `inputs[]` array provided by the executor.
    InputForward,
    /// An identity pass-through (e.g., Dropout at inference, or a fusion
    /// placeholder). Resolves its buffer from the edge_map, NOT from the
    /// external inputs array.
    IdentityPassthrough,
    /// A constant value (scalar fill) that the executor materializes as a
    /// GPU buffer at runtime. Produced by `TraceOp::Constant` (from
    /// `full()`, `scalar_like()`, etc. during tracing).
    ///
    /// Unlike `InputForward`, this does NOT consume an external input slot.
    /// The executor creates a buffer filled with `value` in the given `shape`.
    ConstantValue {
        /// The scalar fill value.
        value: f64,
        /// Output shape of the constant tensor.
        shape: Vec<usize>,
    },
    /// A native (pre-compiled) operation that delegates to an existing
    /// fused Metal kernel, bypassing the IR → MSL code-generation path.
    ///
    /// Used for operations where the fused kernel significantly outperforms
    /// the decomposed IR expansion (e.g., sequence LSTM: one dispatch vs
    /// O(seq_len) dispatches from IR unrolling).
    NativeOp {
        /// Which native operation to execute.
        op: NativeOpKind,
        /// Weight name → captured weight data from the trace.
        weight_data: HashMap<String, WeightRef>,
    },
    /// A runtime operation whose output shape depends on input data.
    ///
    /// Executed eagerly at inference time because the output size cannot
    /// be determined at compile time. Used for data-dependent ops like
    /// variable-length RepeatInterleave (Kokoro `length_regulate`).
    ///
    /// No weight data — all inputs come from graph edges. The executor
    /// allocates the output buffer at runtime after computing actual sizes.
    RuntimeOp {
        /// Which runtime operation to execute.
        op: RuntimeOpKind,
    },
}

/// A compiled execution plan for a traced model.
///
/// Wraps the step sequence with metadata needed by the downstream executor
/// (input shapes, output step index, weight inventory).
///
/// Created by [`compile_trace_to_plan`](super::compile_trace_to_plan).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CompiledPlan {
    /// Pre-compiled steps in topological (execution) order.
    pub steps: Vec<CompiledStep>,
    /// Shapes of graph input tensors, in the order they appear.
    pub input_shapes: Vec<Vec<usize>>,
    /// Index into `steps` of the final output.
    pub output_step: usize,
    /// All weight names referenced across Dispatch steps.
    pub weight_names: Vec<String>,
}

/// Fusion statistics from a compiled plan.
///
/// Counts elementwise chain fusions and dispatch savings. Used by
/// diagnostics and integration tests to measure fusion effectiveness.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FusionStats {
    /// Number of fused dispatch steps (kernel name starts with "fused_").
    pub fused_chains: usize,
    /// Total ops absorbed into fused chains (sum of chain lengths).
    pub fused_ops: usize,
    /// Dispatch savings: fused_ops - fused_chains (dispatches eliminated).
    pub dispatches_saved: usize,
}

/// Peephole fusion statistics from NativeOp steps.
///
/// Complements [`FusionStats`] (which counts elementwise chain fusions)
/// with NativeOp-level peephole pass statistics. Part of #1815.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PeepholeStats {
    /// Number of NativeOp steps (each replaces 2+ dispatches).
    pub native_ops: usize,
    /// Total Metal dispatches from NativeOps (sum of estimated_metal_dispatches).
    pub native_dispatches: usize,
    /// Number of IdentityPassthrough steps (fusion placeholders).
    pub passthrough_count: usize,
    /// Per-variant breakdown: variant name -> count (sorted descending).
    pub by_variant: Vec<(String, usize)>,
}

impl CompiledPlan {
    /// Count elementwise chain fusion statistics in this compiled plan.
    ///
    /// Identifies fused dispatch steps by their "fused_*_xN" kernel name
    /// convention set by [`compile_trace_with_fusion`]. Returns the number
    /// of fused chains, total fused ops, and dispatch savings.
    #[must_use]
    pub fn fusion_stats(&self) -> FusionStats {
        let mut stats = FusionStats::default();
        for step in &self.steps {
            if let CompiledStep::Dispatch { kernel, .. } = step {
                let name = kernel.name();
                if let Some(rest) = name.strip_prefix("fused_") {
                    if let Some(x_pos) = rest.rfind("_x") {
                        if let Ok(chain_len) = rest[x_pos + 2..].parse::<usize>() {
                            stats.fused_chains += 1;
                            stats.fused_ops += chain_len;
                        }
                    }
                }
            }
        }
        stats.dispatches_saved = stats.fused_ops.saturating_sub(stats.fused_chains);
        stats
    }

    /// Count peephole NativeOp fusion statistics in this compiled plan.
    ///
    /// Counts NativeOp steps and their estimated Metal dispatch cost,
    /// plus IdentityPassthrough steps (fusion placeholders that replaced
    /// absorbed steps). Part of #1815.
    #[must_use]
    pub fn peephole_stats(&self) -> PeepholeStats {
        let mut stats = PeepholeStats::default();
        let mut variant_map = HashMap::new();
        for step in &self.steps {
            match step {
                CompiledStep::NativeOp { op, .. } => {
                    stats.native_ops += 1;
                    stats.native_dispatches += op.estimated_metal_dispatches();
                    *variant_map.entry(op.variant_name()).or_insert(0usize) += 1;
                }
                CompiledStep::IdentityPassthrough => {
                    stats.passthrough_count += 1;
                }
                _ => {}
            }
        }
        let mut by_variant: Vec<(String, usize)> = variant_map
            .into_iter()
            .map(|(name, count)| (name.to_string(), count))
            .collect();
        by_variant.sort_by_key(|x| std::cmp::Reverse(x.1));
        stats.by_variant = by_variant;
        stats
    }
}

#[cfg(test)]
#[path = "trace_compile_types_tests.rs"]
mod tests;
