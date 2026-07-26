// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Subgraph fingerprinting for incremental re-verification.
//!
//! Computes structural and parametric fingerprints of `ComputationGraph` nodes
//! so that when a model changes (fine-tuning, layer swap, architecture edit),
//! only the changed subgraphs need re-verification. Unchanged subgraphs retain
//! their prior proof certificates.
//!
//! # Two fingerprint modes
//!
//! - **Structural** (`fingerprint_trace`): Hashes op type, shapes, and
//!   hyperparameters. Detects architecture changes. Cheap.
//! - **Parametric** (`fingerprint_trace_with_weights`): Also hashes weight
//!   content. Detects fine-tuning changes. Expensive for large models.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_verify::fingerprint::{fingerprint_trace, diff_fingerprints};
//!
//! let old_fps = fingerprint_trace(old_graph.nodes());
//! let new_fps = fingerprint_trace(new_graph.nodes());
//! let changes = diff_fingerprints(&old_fps, &new_fps);
//! // changes tells you which regions need re-verification
//! ```
//!
//! Part of #2457.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};

/// Fingerprint of a contiguous subgraph region.
///
/// Each fingerprint identifies a set of computation graph nodes by their
/// structural and (optionally) parametric identity. Two fingerprints with
/// the same `hash` guarantee that the nodes have identical op types, shapes,
/// hyperparameters, and (if parametric mode was used) weight content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubgraphFingerprint {
    /// Indices of the nodes in the `ComputationGraph::nodes()` slice.
    pub node_indices: Vec<usize>,
    /// SHA-256 hash of the fingerprinted content.
    pub hash: [u8; 32],
    /// Human-readable summary of the ops (e.g., "linear,relu,layer_norm").
    pub op_summary: String,
}

/// A contiguous region of the computation graph that changed between two
/// fingerprint sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangedRegion {
    /// Start index (inclusive) in the new graph's node list.
    pub start: usize,
    /// End index (exclusive) in the new graph's node list.
    pub end: usize,
    /// Why this region was flagged as changed.
    pub reason: ChangeReason,
}

/// Reason a subgraph region was flagged as changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ChangeReason {
    /// The operation type or hyperparameters differ.
    OpChanged,
    /// The output shape differs.
    ShapeChanged,
    /// The weight content differs (parametric fingerprint only).
    WeightChanged,
    /// Nodes were inserted (new graph is longer).
    Inserted,
    /// Nodes were removed (new graph is shorter).
    Removed,
}

impl std::fmt::Display for ChangeReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpChanged => write!(f, "op_changed"),
            Self::ShapeChanged => write!(f, "shape_changed"),
            Self::WeightChanged => write!(f, "weight_changed"),
            Self::Inserted => write!(f, "inserted"),
            Self::Removed => write!(f, "removed"),
        }
    }
}

/// Compute per-node structural fingerprints for a computation graph.
///
/// Hashes op type, output shape, and hyperparameters for each node.
/// Weight shapes are included but weight *content* is not (use
/// [`fingerprint_trace_with_weights`] for content-aware fingerprints).
///
/// Returns one `SubgraphFingerprint` per node, with `node_indices` containing
/// a single index each.
pub fn fingerprint_trace(nodes: &[TraceNode]) -> Vec<SubgraphFingerprint> {
    nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let hash = hash_node(node, false);
            SubgraphFingerprint {
                node_indices: vec![i],
                hash,
                op_summary: node.op().canonical_name().to_string(),
            }
        })
        .collect()
}

/// Compute per-node parametric fingerprints (includes weight content).
///
/// Like [`fingerprint_trace`] but also hashes the flat f32 weight data for
/// each `WeightRef` in the operation. This detects fine-tuning changes where
/// architecture is identical but weights differ.
///
/// More expensive than structural-only fingerprinting for large weight tensors.
pub fn fingerprint_trace_with_weights(nodes: &[TraceNode]) -> Vec<SubgraphFingerprint> {
    nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let hash = hash_node(node, true);
            SubgraphFingerprint {
                node_indices: vec![i],
                hash,
                op_summary: node.op().canonical_name().to_string(),
            }
        })
        .collect()
}

/// Convenience: fingerprint an entire `ComputationGraph`.
pub fn fingerprint_graph(graph: &ComputationGraph) -> Vec<SubgraphFingerprint> {
    fingerprint_trace(graph.nodes())
}

/// Convenience: fingerprint an entire `ComputationGraph` with weight content.
pub fn fingerprint_graph_with_weights(graph: &ComputationGraph) -> Vec<SubgraphFingerprint> {
    fingerprint_trace_with_weights(graph.nodes())
}

/// Compare two fingerprint sets and identify changed regions.
///
/// Uses sequential comparison: walks both lists in parallel and groups
/// contiguous mismatches into `ChangedRegion` values. When lengths differ,
/// trailing nodes are reported as `Inserted` or `Removed`.
///
/// This is intentionally simple (O(max(n,m))) rather than using LCS/edit-distance,
/// because:
/// 1. Computation graphs are topologically ordered — insertions shift all
///    subsequent indices, making LCS less useful than it seems.
/// 2. The primary use case is fine-tuning (same structure, different weights)
///    or small edits (swap one layer), where sequential comparison suffices.
pub fn diff_fingerprints(
    old: &[SubgraphFingerprint],
    new: &[SubgraphFingerprint],
) -> Vec<ChangedRegion> {
    let mut regions = Vec::new();
    let common_len = old.len().min(new.len());

    // Compare overlapping portion.
    let mut i = 0;
    while i < common_len {
        if old[i].hash == new[i].hash {
            i += 1;
            continue;
        }
        // Start of a changed region.
        let start = i;
        let reason = classify_change(&old[i], &new[i]);
        // Extend as long as consecutive nodes differ.
        while i < common_len && old[i].hash != new[i].hash {
            i += 1;
        }
        regions.push(ChangedRegion {
            start,
            end: i,
            reason,
        });
    }

    // Handle length differences.
    if new.len() > old.len() {
        regions.push(ChangedRegion {
            start: old.len(),
            end: new.len(),
            reason: ChangeReason::Inserted,
        });
    } else if old.len() > new.len() {
        regions.push(ChangedRegion {
            start: new.len(),
            end: old.len(),
            reason: ChangeReason::Removed,
        });
    }

    regions
}

// ---------------------------------------------------------------------------
// Internal hashing
// ---------------------------------------------------------------------------

/// Hash a single trace node into a 32-byte SHA-256 digest.
///
/// When `include_weight_content` is true, the flat f32 weight data is included
/// in the hash (detects fine-tuning). When false, only weight shapes are hashed
/// (detects architecture changes).
fn hash_node(node: &TraceNode, include_weight_content: bool) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // 1. Op type
    hasher.update(node.op().canonical_name().as_bytes());
    hasher.update(b"|");

    // 2. Output shape
    for &dim in node.output_shape() {
        hasher.update(dim.to_le_bytes());
    }
    hasher.update(b"|");

    // 3. Op-specific hyperparameters and weight shapes
    hash_op_params(&mut hasher, node.op(), include_weight_content);

    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Hash operation-specific hyperparameters and weight references.
///
/// Uses a domain separator for each op variant so that different ops with
/// the same numeric parameters hash differently.
#[allow(clippy::too_many_lines)]
fn hash_op_params(hasher: &mut Sha256, op: &TraceOp, include_weight_content: bool) {
    // Helper closures.
    let hash_f64 = |h: &mut Sha256, v: f64| h.update(v.to_le_bytes());
    let hash_usize = |h: &mut Sha256, v: usize| h.update(v.to_le_bytes());

    match op {
        // Leaf / no-param ops
        TraceOp::Input
        | TraceOp::Add
        | TraceOp::Sub
        | TraceOp::Mul
        | TraceOp::Div
        | TraceOp::Maximum
        | TraceOp::Minimum
        | TraceOp::MatMul
        | TraceOp::Relu
        | TraceOp::Gelu
        | TraceOp::GeluErf
        | TraceOp::Silu
        | TraceOp::Tanh
        | TraceOp::Sigmoid
        | TraceOp::Exp
        | TraceOp::Log
        | TraceOp::Sqrt
        | TraceOp::Sqr
        | TraceOp::Abs
        | TraceOp::Neg
        | TraceOp::Recip
        | TraceOp::Sin
        | TraceOp::Cos
        | TraceOp::Floor
        | TraceOp::Round
        | TraceOp::Fract
        | TraceOp::Softplus
        | TraceOp::SwiGlu
        | TraceOp::Dropout
        | TraceOp::WhereCond
        | TraceOp::Atan2 => {}

        // Ops with simple scalar/usize params
        TraceOp::ReduceSum { dim, keepdim }
        | TraceOp::ReduceMean { dim, keepdim }
        | TraceOp::ReduceMax { dim, keepdim }
        | TraceOp::ReduceMin { dim, keepdim } => {
            hash_usize(hasher, *dim);
            hasher.update([u8::from(*keepdim)]);
        }

        TraceOp::Reshape { target_shape } => {
            for &d in target_shape {
                hash_usize(hasher, d);
            }
        }
        TraceOp::Transpose { dim0, dim1 } => {
            hash_usize(hasher, *dim0);
            hash_usize(hasher, *dim1);
        }
        TraceOp::Narrow { dim, start, length } => {
            hash_usize(hasher, *dim);
            hash_usize(hasher, *start);
            hash_usize(hasher, *length);
        }
        TraceOp::Unsqueeze { dim } | TraceOp::Squeeze { dim } => {
            hash_usize(hasher, *dim);
        }
        TraceOp::Permute { axes } => {
            for &a in axes {
                hash_usize(hasher, a);
            }
        }
        TraceOp::Cat { dim, num_inputs } => {
            hash_usize(hasher, *dim);
            hash_usize(hasher, *num_inputs);
        }

        // Normalization ops (have eps + weights)
        TraceOp::LayerNorm { eps, weight, bias } => {
            hash_f64(hasher, *eps);
            hash_weight_ref(hasher, weight, include_weight_content);
            hash_weight_ref(hasher, bias, include_weight_content);
        }
        TraceOp::RmsNorm { eps, weight } => {
            hash_f64(hasher, *eps);
            hash_weight_ref(hasher, weight, include_weight_content);
        }
        TraceOp::GroupNorm {
            num_groups,
            eps,
            weight,
            bias,
        } => {
            hash_usize(hasher, *num_groups);
            hash_f64(hasher, *eps);
            hash_weight_ref(hasher, weight, include_weight_content);
            hash_weight_ref(hasher, bias, include_weight_content);
        }
        TraceOp::InstanceNorm { eps } => {
            hash_f64(hasher, *eps);
        }
        TraceOp::BatchNorm {
            eps,
            weight,
            bias,
            running_mean,
            running_var,
        } => {
            hash_f64(hasher, *eps);
            hash_weight_ref(hasher, weight, include_weight_content);
            hash_weight_ref(hasher, bias, include_weight_content);
            hash_weight_ref(hasher, running_mean, include_weight_content);
            hash_weight_ref(hasher, running_var, include_weight_content);
        }

        // Linear/Conv ops
        TraceOp::Linear { weight, bias } | TraceOp::QLinear { weight, bias } => {
            hash_weight_ref(hasher, weight, include_weight_content);
            if let Some(b) = bias {
                hash_weight_ref(hasher, b, include_weight_content);
            }
        }
        TraceOp::Conv1d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => {
            hash_weight_ref(hasher, weight, include_weight_content);
            if let Some(b) = bias {
                hash_weight_ref(hasher, b, include_weight_content);
            }
            hash_usize(hasher, *padding);
            hash_usize(hasher, *stride);
            hash_usize(hasher, *dilation);
            hash_usize(hasher, *groups);
        }
        TraceOp::Conv2d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => {
            hash_weight_ref(hasher, weight, include_weight_content);
            if let Some(b) = bias {
                hash_weight_ref(hasher, b, include_weight_content);
            }
            for &p in padding {
                hash_usize(hasher, p);
            }
            for &s in stride {
                hash_usize(hasher, s);
            }
            for &d in dilation {
                hash_usize(hasher, d);
            }
            hash_usize(hasher, *groups);
        }
        TraceOp::Conv3d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => {
            hash_weight_ref(hasher, weight, include_weight_content);
            if let Some(b) = bias {
                hash_weight_ref(hasher, b, include_weight_content);
            }
            for &p in padding {
                hash_usize(hasher, p);
            }
            for &s in stride {
                hash_usize(hasher, s);
            }
            for &d in dilation {
                hash_usize(hasher, d);
            }
            hash_usize(hasher, *groups);
        }
        TraceOp::ConvTranspose1d {
            weight,
            bias,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        } => {
            hash_weight_ref(hasher, weight, include_weight_content);
            if let Some(b) = bias {
                hash_weight_ref(hasher, b, include_weight_content);
            }
            hash_usize(hasher, *padding);
            hash_usize(hasher, *output_padding);
            hash_usize(hasher, *stride);
            hash_usize(hasher, *dilation);
            hash_usize(hasher, *groups);
        }
        TraceOp::ConvTranspose2d {
            weight,
            bias,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        } => {
            hash_weight_ref(hasher, weight, include_weight_content);
            if let Some(b) = bias {
                hash_weight_ref(hasher, b, include_weight_content);
            }
            for &p in padding {
                hash_usize(hasher, p);
            }
            for &p in output_padding {
                hash_usize(hasher, p);
            }
            for &s in stride {
                hash_usize(hasher, s);
            }
            for &d in dilation {
                hash_usize(hasher, d);
            }
            hash_usize(hasher, *groups);
        }

        // Attention ops
        TraceOp::Softmax { dim } | TraceOp::LogSoftmax { dim } => {
            hash_usize(hasher, *dim);
        }
        TraceOp::Sdpa { scale } | TraceOp::SdpaCausal { scale } => {
            hash_f64(hasher, *scale);
        }
        TraceOp::RotaryEmbedding {
            head_dim,
            offset,
            cos_cache,
            sin_cache,
        } => {
            hash_usize(hasher, *head_dim);
            hash_usize(hasher, *offset);
            hash_weight_ref(hasher, cos_cache, include_weight_content);
            hash_weight_ref(hasher, sin_cache, include_weight_content);
        }
        TraceOp::MultiHeadAttention {
            num_heads,
            num_kv_heads,
            head_dim,
        } => {
            hash_usize(hasher, *num_heads);
            hash_usize(hasher, *num_kv_heads);
            hash_usize(hasher, *head_dim);
        }
        TraceOp::Embedding { weight } => {
            hash_weight_ref(hasher, weight, include_weight_content);
        }

        // Recurrent
        TraceOp::Lstm {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
            ..
        } => {
            hash_weight_ref(hasher, weight_ih, include_weight_content);
            hash_weight_ref(hasher, weight_hh, include_weight_content);
            if let Some(b) = bias_ih {
                hash_weight_ref(hasher, b, include_weight_content);
            }
            if let Some(b) = bias_hh {
                hash_weight_ref(hasher, b, include_weight_content);
            }
            hash_usize(hasher, *hidden_size);
        }

        // Pooling
        TraceOp::MaxPool1d {
            kernel_size,
            stride,
            padding,
        } => {
            hash_usize(hasher, *kernel_size);
            hash_usize(hasher, *stride);
            hash_usize(hasher, *padding);
        }
        TraceOp::AvgPool2d {
            kernel_size,
            stride,
            padding,
        }
        | TraceOp::MaxPool2d {
            kernel_size,
            stride,
            padding,
        } => {
            for &k in kernel_size {
                hash_usize(hasher, k);
            }
            for &s in stride {
                hash_usize(hasher, s);
            }
            for &p in padding {
                hash_usize(hasher, p);
            }
        }
        TraceOp::AdaptiveAvgPool2d { output_size } => {
            for &s in output_size {
                hash_usize(hasher, s);
            }
        }

        // Activation ops
        TraceOp::Activation { kind } => {
            hasher.update(format!("{kind:?}").as_bytes());
        }
        TraceOp::Elu { alpha } => {
            hash_f64(hasher, *alpha);
        }
        TraceOp::LeakyRelu { slope } => {
            hash_f64(hasher, *slope);
        }
        TraceOp::KokoroFused(fused) => {
            hasher.update(format!("{fused:?}").as_bytes());
        }

        // Vision ops
        TraceOp::PixelShuffle { upscale_factor } => {
            hash_usize(hasher, *upscale_factor);
        }
        TraceOp::PixelUnshuffle { downscale_factor } => {
            hash_usize(hasher, *downscale_factor);
        }
        TraceOp::Upsample1d { factor } => {
            hash_usize(hasher, *factor);
        }
        TraceOp::Upsample2d {
            mode,
            scale_h,
            scale_w,
        } => {
            hasher.update(format!("{mode:?}").as_bytes());
            hash_f64(hasher, *scale_h);
            hash_f64(hasher, *scale_w);
        }

        // Spatial mask / sampling
        TraceOp::Triu { diagonal } | TraceOp::Tril { diagonal } => {
            hasher.update(diagonal.to_le_bytes());
        }
        TraceOp::GridSample {
            padding_mode,
            align_corners,
        } => {
            hasher.update(format!("{padding_mode:?}").as_bytes());
            hasher.update([u8::from(*align_corners)]);
        }

        // Selection / indexing
        TraceOp::Topk { k, dim } => {
            hash_usize(hasher, *k);
            hash_usize(hasher, *dim);
        }
        TraceOp::Argmax { dim }
        | TraceOp::Argmin { dim }
        | TraceOp::IndexSelect { dim }
        | TraceOp::Gather { dim }
        | TraceOp::Cumsum { dim }
        | TraceOp::RepeatInterleave { dim }
        | TraceOp::ScatterAdd { dim }
        | TraceOp::IndexAdd { dim }
        | TraceOp::IndexPut { dim } => {
            hash_usize(hasher, *dim);
        }
        TraceOp::ArgSort { dim, descending } => {
            hash_usize(hasher, *dim);
            hasher.update([u8::from(*descending)]);
        }
        TraceOp::Expand { target_shape } => {
            for &d in target_shape {
                hash_usize(hasher, d);
            }
        }
        TraceOp::Compare { op, value } => {
            hasher.update(format!("{op:?}").as_bytes());
            hash_f64(hasher, *value);
        }
        TraceOp::CompareTensor { op } => {
            hasher.update(format!("{op:?}").as_bytes());
        }
        TraceOp::Powf { exponent } => {
            hash_f64(hasher, *exponent);
        }
        TraceOp::ToDtype { target_dtype } => {
            hasher.update(format!("{target_dtype:?}").as_bytes());
        }

        // Shape ops (extended)
        TraceOp::Flip { dim } | TraceOp::SliceSet { dim, .. } => {
            hash_usize(hasher, *dim);
        }
        TraceOp::Unfold { dim, size, step } => {
            hash_usize(hasher, *dim);
            hash_usize(hasher, *size);
            hash_usize(hasher, *step);
        }
        TraceOp::Clamp { min, max } => {
            if let Some(v) = min {
                hash_f64(hasher, *v);
            }
            if let Some(v) = max {
                hash_f64(hasher, *v);
            }
        }
        TraceOp::Constant { value } => {
            hash_f64(hasher, *value);
        }
        TraceOp::ConstantWeight { weight } => {
            hash_weight_ref(hasher, weight, include_weight_content);
        }

        // Padding
        TraceOp::ReflectionPad1d {
            pad_left,
            pad_right,
        } => {
            hash_usize(hasher, *pad_left);
            hash_usize(hasher, *pad_right);
        }
        TraceOp::ConstantPadNd { padding, value } => {
            for &p in padding {
                hash_usize(hasher, p);
            }
            hash_f64(hasher, *value);
        }

        // Tensor creation
        TraceOp::Arange { start, end, step } => {
            hash_f64(hasher, *start);
            hash_f64(hasher, *end);
            hash_f64(hasher, *step);
        }

        // Segment boundary
        TraceOp::SegmentBoundary {
            reason,
            input_bounds,
        } => {
            hasher.update(reason.as_bytes());
            if let Some((lo, hi)) = input_bounds {
                hasher.update(lo.to_le_bytes());
                hasher.update(hi.to_le_bytes());
            }
        }

        // Custom
        TraceOp::Custom { name } => {
            hasher.update(name.as_bytes());
        }

        // Catch-all for future #[non_exhaustive] variants.
        // Conservative: hash the Debug representation.
        _ => {
            hasher.update(format!("{op:?}").as_bytes());
        }
    }
}

/// Hash a `WeightRef` into the running hasher.
///
/// Always includes the shape. Includes flat f32 data only when
/// `include_content` is true.
fn hash_weight_ref(
    hasher: &mut Sha256,
    weight: &nn_core::dyn_tensor::trace::WeightRef,
    include_content: bool,
) {
    // Shape (always).
    for &dim in weight.shape() {
        hasher.update(dim.to_le_bytes());
    }
    hasher.update(b":");
    // Content (optional).
    if include_content {
        for &val in weight.data() {
            hasher.update(val.to_le_bytes());
        }
    }
    hasher.update(b";");
}

/// Classify the reason for a fingerprint mismatch.
///
/// Compares op_summary (op type) first, then falls back to generic "changed".
fn classify_change(old: &SubgraphFingerprint, new: &SubgraphFingerprint) -> ChangeReason {
    if old.op_summary != new.op_summary {
        ChangeReason::OpChanged
    } else {
        // Same op type but different hash — could be shape or weight change.
        // Without deeper inspection, we report the most common case.
        ChangeReason::WeightChanged
    }
}

#[cfg(test)]
#[path = "fingerprint_tests.rs"]
mod tests;
