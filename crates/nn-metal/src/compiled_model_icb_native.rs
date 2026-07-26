// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ICB encoding for `NativeOp` steps in `CompiledModel`.
//!
//! NativeOps bypass the standard IR → MSL codegen pipeline, using hand-tuned
//! Metal kernels dispatched through existing eager-mode paths. This module
//! bridges NativeOps into the ICB replay system by classifying which variants
//! can be pre-encoded and providing the metadata needed for ICB command
//! construction.
//!
//! # Design
//!
//! NativeOps fall into three categories for ICB eligibility:
//!
//! 1. **ICB-compatible** — single Metal dispatch with static shapes and
//!    deterministic buffer bindings. Can be pre-encoded into an ICB command.
//!    Examples: `SiluMul`, `RotaryEmbedding`, `MaxPool1d`.
//!
//! 2. **Multi-dispatch** — internally issue multiple Metal dispatches or use
//!    DynTensor bridges with intermediate allocations. Cannot be pre-encoded
//!    as a single ICB command without decomposition.
//!    Examples: `LstmSequence`, `FusedResBlock`, `FlashAttention`.
//!
//! 3. **CPU-materialized** — produce output from pre-computed data or use
//!    CPU readback paths. No GPU dispatch to encode.
//!    Examples: `ConstantWeight`.
//!
//! The `NativeOpIcbEncoder` provides build-time classification via
//! `try_encode_native_op_icb()`. When a NativeOp is ICB-compatible, it
//! returns an `IcbNativeCommand` describing the Metal dispatch parameters.
//! When not, it returns `None` and the execution loop falls back to
//! direct dispatch via `execute_native_op()`.
//!
//! Part of #3458.

use nn_dsl::NativeOpKind;

/// Result of attempting to classify a NativeOp for ICB encoding.
///
/// When `Some`, the NativeOp can be pre-encoded into an ICB command
/// using the returned metadata. When `None`, the NativeOp requires
/// direct dispatch (multi-dispatch, CPU readback, or unsupported).
#[derive(Debug, Clone)]
#[allow(dead_code)] // ICB NativeOp wiring in progress (#3458)
pub(crate) struct IcbNativeCommand {
    /// Number of GPU buffer bindings required (inputs + weights + output).
    pub(crate) buffer_binding_count: usize,
    /// Total output elements (for elementwise dispatch planning).
    pub(crate) total_output_elements: usize,
    /// Human-readable tag for diagnostics and logging.
    pub(crate) op_tag: &'static str,
    /// Dispatch geometry classification.
    pub(crate) dispatch_kind: IcbNativeDispatchKind,
}

/// How the NativeOp's Metal dispatch should be configured in the ICB.
///
/// Different NativeOps use different dispatch patterns:
/// - Elementwise: 1D grid over total elements
/// - Tiled GEMM: 2D threadgroup grid for matmul
/// - Custom: op-specific grid dimensions
#[derive(Debug, Clone)]
#[allow(dead_code)] // ICB NativeOp wiring in progress (#3458)
pub(crate) enum IcbNativeDispatchKind {
    /// 1D elementwise dispatch: `total_elements` threads with standard
    /// threadgroup size (256). Used for SiluMul, RotaryEmbedding.
    Elementwise,
    /// 2D tiled simdgroup GEMM dispatch. Used for LinearActivation,
    /// Conv1dGemm, BatchedLinearProjection, Int8Gemm.
    TiledGemm {
        /// Rows in the activation matrix.
        m: usize,
        /// Contracted dimension.
        k: usize,
        /// Output columns.
        n: usize,
    },
    /// 3D dispatch with custom grid. Used for MaxPool1d and similar
    /// ops with spatial dimensions.
    Grid3D {
        /// Grid dimensions `[x, y, z]` in threadgroups.
        grid: [u32; 3],
        /// Threads per threadgroup `[x, y, z]`.
        threads: [u32; 3],
    },
}

/// Attempt to classify a NativeOp for ICB pre-encoding.
///
/// Returns `Some(IcbNativeCommand)` if the operation can be pre-encoded
/// into a single ICB command with static dispatch parameters.
/// Returns `None` if the operation requires direct dispatch.
///
/// # Unsupported categories
///
/// - **Multi-dispatch ops** (LstmSequence, FusedResBlock, FlashAttention,
///   NormActivConv1d, AdainSnake, AdainLeakyRelu, AdaLayerNorm, NormLinear):
///   These issue multiple Metal dispatches or use DynTensor bridges internally.
///
/// - **CPU-materialized ops** (ConstantWeight): No GPU dispatch to encode.
///
/// - **Reduction ops** (InstanceNorm, LayerNorm, AddLayerNorm,
///   ChannelsFirstLayerNorm, Cumsum): Use threadgroup reductions with
///   complex dispatch geometry that varies by input shape and typically
///   use DynTensor bridge paths.
///
/// - **Batched projection ops** (BatchedLinearProjection, ProjectionSlice,
///   BatchedStyleProjection): Use thread-local temp storage or narrow
///   dispatch paths that are incompatible with static ICB encoding.
#[allow(dead_code)] // ICB NativeOp wiring in progress (#3458)
pub(crate) fn try_encode_native_op_icb(
    op: &NativeOpKind,
    _step_idx: usize,
) -> Option<IcbNativeCommand> {
    match op {
        // ── ICB-compatible: single elementwise dispatch ─────────────
        NativeOpKind::SiluMul { input_shape } => {
            let total: usize = input_shape.iter().product();
            if total == 0 {
                return None;
            }
            Some(IcbNativeCommand {
                // gate (input 0) + up (input 1) + output = 3 buffers
                buffer_binding_count: 3,
                total_output_elements: total,
                op_tag: "SiluMul",
                dispatch_kind: IcbNativeDispatchKind::Elementwise,
            })
        }

        NativeOpKind::RotaryEmbedding {
            head_dim,
            input_shape,
        } => {
            let total: usize = input_shape.iter().product();
            if total == 0 || *head_dim == 0 {
                return None;
            }
            Some(IcbNativeCommand {
                // input + cos_cache + sin_cache + output = 4 buffers
                buffer_binding_count: 4,
                total_output_elements: total,
                op_tag: "RotaryEmbedding",
                dispatch_kind: IcbNativeDispatchKind::Elementwise,
            })
        }

        NativeOpKind::MaxPool1d {
            kernel_size,
            stride,
            padding,
            input_shape,
        } => {
            // Output shape: [B, C, L_out]
            if input_shape.len() < 3 {
                return None;
            }
            let batch = input_shape[0];
            let channels = input_shape[1];
            let l_in = input_shape[2];
            let effective_k = *kernel_size;
            let padded = l_in + 2 * padding;
            if padded < effective_k {
                return None;
            }
            let l_out = (padded - effective_k) / stride + 1;
            let total = batch * channels * l_out;
            if total == 0 {
                return None;
            }
            Some(IcbNativeCommand {
                // input + output = 2 buffers
                buffer_binding_count: 2,
                total_output_elements: total,
                op_tag: "MaxPool1d",
                dispatch_kind: IcbNativeDispatchKind::Elementwise,
            })
        }

        NativeOpKind::LinearActivation {
            in_features,
            out_features,
            has_bias,
            input_shape,
            ..
        } => {
            let batch_size: usize = input_shape.iter().rev().skip(1).product();
            let total = batch_size.checked_mul(*out_features)?;
            if total == 0 || *in_features == 0 {
                return None;
            }
            // input + weight + [bias] + output
            let binding_count = if *has_bias { 4 } else { 3 };
            Some(IcbNativeCommand {
                buffer_binding_count: binding_count,
                total_output_elements: total,
                op_tag: "LinearActivation",
                dispatch_kind: IcbNativeDispatchKind::TiledGemm {
                    m: batch_size,
                    k: *in_features,
                    n: *out_features,
                },
            })
        }

        NativeOpKind::Conv1dGemm {
            input_shape,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            ..
        } => {
            if input_shape.len() < 3 {
                return None;
            }
            let batch = input_shape[0];
            let c_in = input_shape[1];
            let l_in = input_shape[2];
            let effective_k = (kernel_size - 1) * dilation + 1;
            let padded = l_in + 2 * padding;
            if padded < effective_k {
                return None;
            }
            let l_out = (padded - effective_k) / stride + 1;
            let m = batch * l_out;
            let k = c_in * kernel_size;
            let n = *out_channels;
            let total = batch * n * l_out;
            if total == 0 {
                return None;
            }
            // input + weight + [bias] + output (+ im2col intermediate)
            // Note: Conv1dGemm uses im2col + GEMM internally, which is
            // 2 dispatches. Not ICB-compatible as a single command.
            // Future: decompose into im2col ICB + GEMM ICB.
            let _ = (m, k, n);
            None
        }

        NativeOpKind::Int8Gemm {
            in_features,
            out_features,
            has_bias,
            input_shape,
        } => {
            let batch_size: usize = input_shape.iter().rev().skip(1).product();
            let total = batch_size.checked_mul(*out_features)?;
            if total == 0 || *in_features == 0 {
                return None;
            }
            // input + weight_int8 + scale + zero_point + [bias] + output
            let binding_count = if *has_bias { 6 } else { 5 };
            Some(IcbNativeCommand {
                buffer_binding_count: binding_count,
                total_output_elements: total,
                op_tag: "Int8Gemm",
                dispatch_kind: IcbNativeDispatchKind::TiledGemm {
                    m: batch_size,
                    k: *in_features,
                    n: *out_features,
                },
            })
        }

        // ── Multi-dispatch: require direct dispatch ─────────────────

        // LSTM: sequential recurrence + optional precomputed GEMM path.
        // 2+ dispatches with data-dependent routing.
        NativeOpKind::LstmSequence { .. } => None,

        // Cumsum: parallel prefix scan, 1-3 dispatches depending on size.
        NativeOpKind::Cumsum { .. } => None,

        // Norm ops: use DynTensor bridge with threadgroup reductions.
        NativeOpKind::InstanceNorm { .. } => None,
        NativeOpKind::LayerNorm { .. } => None,
        NativeOpKind::AddLayerNorm { .. } => None,
        NativeOpKind::ChannelsFirstLayerNorm { .. } => None,

        // Fused AdaIN/Ada variants: multi-phase norm+affine+activation.
        NativeOpKind::AdainSnake { .. } => None,
        NativeOpKind::AdainLeakyRelu { .. } => None,
        NativeOpKind::FusedAdainSnake { .. } => None,
        // Fused InstanceNorm + Mul + Add: multi-phase norm+affine.
        NativeOpKind::FusedInstanceNormMulAdd { .. } => None,
        // Fused Snake + InstanceNorm: two-pass kernel, not ICB-encodable.
        NativeOpKind::FusedSnakeInstanceNorm { .. } => None,
        // Fused upsample + conv: multi-phase, not ICB-encodable.
        NativeOpKind::FusedUpsampleConv1d { .. } => None,
        // Fused conv + activation: multi-phase, not ICB-encodable.
        NativeOpKind::FusedConv1dActivation { .. } => None,
        NativeOpKind::AdaLayerNorm { .. } => None,

        // Flash attention: tiled with online softmax, complex dispatch.
        NativeOpKind::FlashAttention { .. } => None,

        // FusedResBlock: 2x NormActivConv1d + residual, multiple dispatches.
        NativeOpKind::FusedResBlock { .. } => None,

        // NormActivConv1d: fused norm+activation+conv, multi-phase.
        NativeOpKind::NormActivConv1d { .. } => None,

        // NormLinear: fused norm+GEMM, uses threadgroup memory.
        NativeOpKind::NormLinear { .. } => None,

        // Batched projection ops: thread-local temp storage.
        NativeOpKind::BatchedLinearProjection { .. } => None,
        NativeOpKind::ProjectionSlice { .. } => None,
        NativeOpKind::BatchedStyleProjection { .. } => None,

        // CPU-materialized: no GPU dispatch.
        NativeOpKind::ConstantWeight { .. } => None,

        // BiLstmCat: 2 LSTM dispatches + 1 cat, multi-dispatch.
        NativeOpKind::BiLstmCat { .. } => None,

        // AddNormLinear: fused norm+GEMM, uses threadgroup memory.
        NativeOpKind::AddNormLinear { .. } => None,

        // MoeGating: multi-dispatch (softmax + topk + routing).
        NativeOpKind::MoeGating { .. } => None,

        // Conservative default for future variants.
        _ => None,
    }
}

/// Check whether a NativeOp variant is ICB-eligible.
///
/// Convenience wrapper around [`try_encode_native_op_icb`] that returns
/// a boolean. Used by eligibility analysis to extend ICB segments
/// across NativeOp steps where possible.
#[allow(dead_code)] // ICB NativeOp wiring in progress (#3458)
pub(crate) fn is_native_op_icb_eligible(op: &NativeOpKind) -> bool {
    try_encode_native_op_icb(op, 0).is_some()
}

/// Count the number of ICB-eligible NativeOp steps in a compiled model.
///
/// Returns `(eligible, total)` where `eligible` is the count of NativeOp
/// steps that can be pre-encoded into ICB commands, and `total` is the
/// count of all NativeOp steps.
#[allow(dead_code)] // ICB NativeOp wiring in progress (#3458)
pub(crate) fn count_icb_eligible_native_ops(
    steps: &[nn_dsl::trace_compile::CompiledStep],
) -> (usize, usize) {
    use nn_dsl::trace_compile::CompiledStep;

    let mut eligible = 0;
    let mut total = 0;

    for step in steps {
        if let CompiledStep::NativeOp { op, .. } = step {
            total += 1;
            if is_native_op_icb_eligible(op) {
                eligible += 1;
            }
        }
    }

    (eligible, total)
}

/// Compute the dispatch grid and threadgroup size for an ICB native command.
///
/// Translates the abstract `IcbNativeDispatchKind` into concrete Metal
/// dispatch parameters suitable for `IndirectCommandBuffer::encode_command`.
///
/// Returns `(grid_size, threadgroup_size)` as `[u32; 3]` pairs.
#[allow(dead_code)] // ICB NativeOp wiring in progress (#3458)
pub(crate) fn compute_native_dispatch_geometry(
    cmd: &IcbNativeCommand,
) -> Option<([u32; 3], [u32; 3])> {
    match &cmd.dispatch_kind {
        IcbNativeDispatchKind::Elementwise => {
            let total = u32::try_from(cmd.total_output_elements).ok()?;
            let threads_per_tg: u32 = 256;
            let num_tgs = total.div_ceil(threads_per_tg);
            Some(([num_tgs, 1, 1], [threads_per_tg, 1, 1]))
        }
        IcbNativeDispatchKind::TiledGemm { m, n, .. } => {
            let m_u32 = u32::try_from(*m).ok()?;
            let n_u32 = u32::try_from(*n).ok()?;
            // Simdgroup tiling: 32x32 tiles, 32x4 threads per threadgroup.
            let grid = [n_u32.div_ceil(32), m_u32.div_ceil(32), 1];
            let threads = [32, 4, 1];
            Some((grid, threads))
        }
        IcbNativeDispatchKind::Grid3D { grid, threads } => Some((*grid, *threads)),
    }
}

#[cfg(test)]
#[path = "compiled_model_icb_native_tests.rs"]
mod tests;
