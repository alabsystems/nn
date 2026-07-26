// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Computation graph tracing for DynTensor operations.
//!
//! Records the computation graph as operations execute (like PyTorch
//! `torch.fx.symbolic_trace`). The captured graph feeds NY for
//! verification. Zero-cost when inactive (single thread-local bool check).
//! See [`trace_graph`] for the entry point.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{DType, Device, Result, TensorError};

#[path = "trace_weight_ref.rs"]
mod weight_ref;
pub use weight_ref::WeightRef;

#[path = "trace_node.rs"]
mod node;
pub use node::{NodeId, TraceNode};

#[path = "trace_resblock_activation.rs"]
mod resblock_activation;
pub use resblock_activation::ResBlockActivation;

#[path = "trace_op_support_types.rs"]
mod support_types;
pub use support_types::{TraceActivation, TraceUpsampleMode};

#[path = "trace_op_kokoro_fused.rs"]
mod kokoro_fused;
pub use kokoro_fused::KokoroFusedOp;

#[path = "trace_types.rs"]
mod types;
pub use types::*;

#[path = "trace_op_class.rs"]
mod op_class;
pub use op_class::TraceOpClass;

#[path = "trace_op_names.rs"]
mod op_names;

#[path = "trace_graph.rs"]
mod graph;
pub use graph::{ComputationGraph, GraphSegment, SegmentedGraph};

#[path = "trace_shape_propagate.rs"]
mod shape_propagate;

// -- Thread-local trace recorder ----------------------------------------------

/// Global counter for generating unique node IDs across trace sessions.
static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);

fn next_node_id() -> NodeId {
    NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Per-operation-type counter for generating readable names.
struct NameCounter {
    counts: HashMap<String, usize>,
}

impl NameCounter {
    fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    fn next_name(&mut self, prefix: &str) -> String {
        let count = self.counts.entry(prefix.to_string()).or_insert(0);
        let name = format!("{prefix}_{count}");
        *count += 1;
        name
    }
}

/// Thread-local trace recorder state.
struct TraceRecorder {
    nodes: Vec<TraceNode>,
    id_to_index: HashMap<NodeId, usize>,
    names: NameCounter,
    last_output: Option<NodeId>,
}

impl TraceRecorder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            id_to_index: HashMap::new(),
            names: NameCounter::new(),
            last_output: None,
        }
    }

    fn add_node(
        &mut self,
        op: TraceOp,
        inputs: Vec<NodeId>,
        output_shape: Vec<usize>,
        output_dtype: DType,
    ) -> NodeId {
        let prefix = op_prefix(&op);
        let name = self.names.next_name(prefix);
        let id = next_node_id();
        let idx = self.nodes.len();
        self.nodes.push(TraceNode::new(
            id,
            name,
            op,
            inputs,
            output_shape,
            output_dtype,
        ));
        self.id_to_index.insert(id, idx);
        self.last_output = Some(id);
        id
    }

    fn into_graph(self) -> ComputationGraph {
        let output_nodes: Vec<NodeId> = self.last_output.into_iter().collect();
        ComputationGraph {
            nodes: self.nodes,
            id_to_index: self.id_to_index,
            output_nodes,
        }
    }
}

#[path = "trace_op_prefix.rs"]
mod op_prefix_mod;
use op_prefix_mod::op_prefix;

// -- Thread-local storage -----------------------------------------------------

thread_local! {
    static TRACE_RECORDER: RefCell<Option<TraceRecorder>> = const { RefCell::new(None) };
    static TRACE_SUPPRESSED: Cell<bool> = const { Cell::new(false) };
}

/// Returns true if tracing is currently active on this thread.
pub fn is_tracing() -> bool {
    TRACE_RECORDER.with(|r| r.borrow().is_some()) && !TRACE_SUPPRESSED.with(Cell::get)
}

/// RAII guard that restores the prior trace-suppression state on drop, even on panic.
struct TraceSuppressGuard {
    prev: bool,
}

impl Drop for TraceSuppressGuard {
    fn drop(&mut self) {
        TRACE_SUPPRESSED.with(|s| s.set(self.prev));
    }
}

/// Suppress trace recording for the duration of a closure.
///
/// Composite ops (Linear, Conv, etc.) use this to prevent their internal
/// decomposed ops (matmul, broadcast_add) from being recorded separately.
/// Only the composite op itself is recorded after the closure returns.
///
/// Uses an RAII guard for panic safety — if `f()` panics, the prior
/// suppression state is restored during stack unwinding.
pub(crate) fn with_trace_suppressed<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = TraceSuppressGuard {
        prev: TRACE_SUPPRESSED.with(Cell::get),
    };
    TRACE_SUPPRESSED.with(|s| s.set(true));
    f()
}

/// Record a traced operation. Returns the node ID if tracing is active.
///
/// This is called from DynTensor dispatch points. When tracing is inactive
/// (the common case), this is a single thread-local bool check — zero cost.
pub(crate) fn record_op(
    op: TraceOp,
    inputs: &[NodeId],
    output_shape: &[usize],
    output_dtype: DType,
) -> Option<NodeId> {
    if !is_tracing() {
        return None;
    }
    TRACE_RECORDER.with(|r| {
        let mut recorder = r.borrow_mut();
        recorder
            .as_mut()
            .map(|rec| rec.add_node(op, inputs.to_vec(), output_shape.to_vec(), output_dtype))
    })
}

/// Register an input tensor and return its node ID.
///
/// Called when a tensor enters the tracing scope (model inputs, constants).
pub fn record_input(shape: &[usize], dtype: DType) -> Option<NodeId> {
    record_op(TraceOp::Input, &[], shape, dtype)
}

/// Record a segment boundary marker on a tensor (#2378).
///
/// Called after data-dependent operations (e.g., `length_regulate`) whose
/// output shape depends on tensor *values*. The verify path (nn-verify)
/// splits graphs at these markers and verifies each segment independently.
///
/// The compile path (nn-dsl) preserves actual ops (RepeatInterleave)
/// and ignores segment boundaries — they are only meaningful for verification.
///
/// This function is a no-op when tracing is not active.
///
/// # Arguments
/// * `tensor` — the output tensor of the data-dependent op (mutated to update trace ID)
/// * `reason` — human-readable label (e.g., `"length_regulate"`)
/// * `input_bounds` — optional (lower, upper) bounds hint for the segment output
pub fn record_segment_boundary(
    tensor: &mut super::DynTensor,
    reason: String,
    input_bounds: Option<(f32, f32)>,
) {
    if !is_tracing() {
        return;
    }
    if let Some(input_id) = tensor.trace_id() {
        if let Some(id) = record_op(
            TraceOp::SegmentBoundary {
                reason,
                input_bounds,
            },
            &[input_id],
            tensor.dims(),
            tensor.dtype(),
        ) {
            tensor.set_trace_id(id);
        }
    }
}

/// Run a composite `Module::forward()` with trace suppression and recording.
///
/// Handles the three-step boilerplate shared by all traced nn layers:
/// 1. Suppress tracing during `compute` (so decomposed ops aren't recorded)
/// 2. Execute `compute` to get the result tensor
/// 3. Record the composite `op` in the trace graph with the result's metadata
///
/// The `op` closure is lazy to avoid building `TraceOp` (which may call
/// `to_weight_ref()`) when tracing is inactive.
///
/// LSTM is intentionally excluded — it returns `(DynTensor, LstmState)`.
///
/// Public so that cross-crate model implementations (e.g., `nn-models`)
/// can record fused composite ops like `FusedAdainResBlock` (#2459).
pub fn traced_forward(
    inputs: &[&super::DynTensor],
    op: impl FnOnce() -> Result<TraceOp>,
    compute: impl FnOnce() -> Result<super::DynTensor>,
) -> Result<super::DynTensor> {
    let tracing = is_tracing();
    let mut result = if tracing {
        with_trace_suppressed(compute)?
    } else {
        compute()?
    };
    if tracing {
        let input_ids = super::DynTensor::trace_input_ids(inputs)?;
        if let Some(id) = record_op(op()?, &input_ids, result.dims(), result.dtype()) {
            result.set_trace_id(id);
        }
    }
    Ok(result)
}

// -- Public API: trace_graph --------------------------------------------------

/// Trace a computation, capturing the operation graph.
///
/// Runs the closure with tracing enabled. The closure executes normally
/// (all operations produce real results) AND records the computation graph.
///
/// # Example
///
/// ```rust,ignore
/// use nn_core::dyn_tensor::trace::trace_graph;
///
/// let graph = trace_graph(|| {
///     let output = model.forward(&input)?;
///     Ok(output)
/// })?;
/// ```
///
/// # Errors
///
/// Returns an error if tracing is already active (nested tracing is not supported)
/// or if the closure returns an error.
///
/// # Panics
///
/// Does not panic. Trace state is cleaned up even if the closure panics.
pub fn trace_graph<F, T>(f: F) -> Result<(T, ComputationGraph)>
where
    F: FnOnce() -> Result<T>,
{
    // Check for nested tracing
    if is_tracing() {
        return Err(TensorError::Unsupported(
            "nested tracing is not supported".into(),
        ));
    }

    // Install recorder
    TRACE_RECORDER.with(|r| {
        *r.borrow_mut() = Some(TraceRecorder::new());
    });

    // Run the computation (with cleanup on panic)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

    // Extract the recorder (always, even on panic)
    let recorder = TRACE_RECORDER.with(|r| r.borrow_mut().take());

    match result {
        Ok(Ok(value)) => {
            let graph =
                recorder
                    .map(TraceRecorder::into_graph)
                    .unwrap_or_else(|| ComputationGraph {
                        nodes: Vec::new(),
                        id_to_index: HashMap::new(),
                        output_nodes: Vec::new(),
                    });
            Ok((value, graph))
        }
        Ok(Err(e)) => Err(e),
        Err(panic_payload) => std::panic::resume_unwind(panic_payload),
    }
}

// -- DynTensor trace ID storage -----------------------------------------------

/// Trait for storing a trace node ID on a tensor.
///
/// When tracing is active, each DynTensor result gets a `trace_id` that
/// identifies it in the computation graph. This ID is stored alongside the
/// tensor and used as input references for subsequent operations.
impl super::DynTensor {
    /// Returns the trace node ID if this tensor was created during tracing.
    pub fn trace_id(&self) -> Option<NodeId> {
        self.trace_node_id
    }

    /// Set the trace node ID. Called internally after recording an op.
    ///
    /// Also available to integration tests in other crates that need to
    /// associate a DynTensor with its traced input node.
    pub fn set_trace_id(&mut self, id: NodeId) {
        self.trace_node_id = Some(id);
    }

    /// Collect trace IDs from input tensors for recording.
    ///
    /// Collects trace node IDs for each input tensor.
    ///
    /// If an input tensor has no trace ID during active tracing (e.g., a
    /// weight tensor created outside the trace scope), it is automatically
    /// registered as a `ConstantWeight` node with the tensor's actual data.
    /// This allows weight parameters to participate in traced binary ops
    /// without explicit trace registration (#2987).
    pub(crate) fn trace_input_ids(inputs: &[&Self]) -> Result<Vec<NodeId>> {
        inputs
            .iter()
            .enumerate()
            .map(|(i, t)| {
                if let Some(id) = t.trace_node_id {
                    Ok(id)
                } else {
                    // Auto-register untraced tensor as a constant weight node.
                    let weight = t.to_weight_ref()?;
                    record_op(TraceOp::ConstantWeight { weight }, &[], t.dims(), t.dtype())
                        .ok_or_else(|| {
                            TensorError::InvalidShape(format!(
                                "trace_input_ids: input {i} of {} has no trace ID and \
                             auto-registration as ConstantWeight failed \
                             (shape={:?}, dtype={:?})",
                                inputs.len(),
                                t.dims(),
                                t.dtype(),
                            ))
                        })
                }
            })
            .collect()
    }

    /// Create a `WeightRef` that captures the actual tensor data (as f32).
    ///
    /// Used during tracing to store weight values in the computation graph
    /// so the NY translator can build `GraphNetwork` layers.
    /// For GPU tensors, transfers to CPU first (one-time cost during tracing).
    ///
    /// Returns `Err(WeightConversionFailed)` if data cannot be extracted.
    /// Use `WeightRef::from_shape()` explicitly when shape-only is intended.
    ///
    /// Public so that cross-crate model implementations (e.g., `nn-models`)
    /// can build fused `TraceOp` variants with weight data (#2459).
    pub fn to_weight_ref(&self) -> Result<WeightRef> {
        // Fast path: CPU or GPU tensor with f32 storage.
        // GPU tensors: to_f32_array() calls to_device(Cpu) → gpu_to_cpu() internally.
        if let Ok(arr) = self.to_f32_array() {
            let shape = arr.shape().to_vec();
            let (data, _offset) = arr.into_raw_vec_and_offset();
            return Ok(WeightRef::new_unchecked(data, shape));
        }

        // U32 CPU path: convert index tensors to f32 weight data.
        // Needed for index_select ConstantWeight capture during trace compilation.
        // The compiled pipeline stores indices as f32 and converts f32→u32 at
        // dispatch time (codegen_msl_tensor_emit_index.rs). Round-trip is lossless
        // for indices < 2^24 (f32 mantissa). Max Kokoro index ≈ 37,800.
        if self.dtype() == DType::U32 && !self.device().is_gpu() {
            if let Ok(arr) = self.as_cpu_u32() {
                let shape = arr.shape().to_vec();
                let data: Vec<f32> = arr.iter().map(|&v| v as f32).collect();
                return Ok(WeightRef::new_unchecked(data, shape));
            }
        }

        // GPU tensor: transfer to CPU first, then extract data.
        if self.device().is_gpu() {
            if let Ok(cpu_tensor) = self.to_device(&Device::Cpu) {
                // Recurse on the CPU copy — hits f32 or U32 path above.
                return cpu_tensor.to_weight_ref();
            }
        }

        Err(TensorError::WeightConversionFailed {
            dtype: self.dtype(),
            device: self.device(),
        })
    }
}

#[cfg(test)]
#[path = "trace_test_index.rs"]
mod test_index;

#[cfg(kani)]
#[path = "kani_trace_types_proofs.rs"]
mod kani_trace_types_proofs;

#[cfg(kani)]
#[path = "kani_trace_op_class_proofs.rs"]
mod kani_trace_op_class_proofs;

#[cfg(kani)]
#[path = "kani_trace.rs"]
mod kani_trace;

#[cfg(kani)]
#[path = "kani_trace_types_extended.rs"]
mod kani_trace_types_extended;

#[cfg(kani)]
#[path = "kani_trace_graph_proofs.rs"]
mod kani_trace_graph_proofs;

#[cfg(kani)]
#[path = "kani_trace_variants_proofs.rs"]
mod kani_trace_variants_proofs;

#[cfg(kani)]
#[path = "kani_trace_recorder_proofs.rs"]
mod kani_trace_recorder_proofs;
