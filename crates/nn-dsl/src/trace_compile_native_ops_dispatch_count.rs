// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal dispatch count estimation for [`NativeOpKind`].

impl super::NativeOpKind {
    /// Returns the variant name as a static string (e.g., "FlashAttention").
    ///
    /// Used by dispatch diagnostics instead of Debug format parsing.
    #[must_use]
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::LstmSequence { .. } => "LstmSequence",
            Self::Cumsum { .. } => "Cumsum",
            Self::InstanceNorm { .. } => "InstanceNorm",
            Self::LayerNorm { .. } => "LayerNorm",
            Self::ChannelsFirstLayerNorm { .. } => "ChannelsFirstLayerNorm",
            Self::AddLayerNorm { .. } => "AddLayerNorm",
            Self::AdainSnake { .. } => "AdainSnake",
            Self::AdainLeakyRelu { .. } => "AdainLeakyRelu",
            Self::AdaLayerNorm { .. } => "AdaLayerNorm",
            Self::FlashAttention { .. } => "FlashAttention",
            Self::MaxPool1d { .. } => "MaxPool1d",
            Self::ConstantWeight { .. } => "ConstantWeight",
            Self::FusedResBlock { .. } => "FusedResBlock",
            Self::NormActivConv1d { .. } => "NormActivConv1d",
            Self::LinearActivation { .. } => "LinearActivation",
            Self::NormLinear { .. } => "NormLinear",
            Self::BatchedLinearProjection { .. } => "BatchedLinearProjection",
            Self::ProjectionSlice { .. } => "ProjectionSlice",
            Self::BatchedStyleProjection { .. } => "BatchedStyleProjection",
            Self::Int8Gemm { .. } => "Int8Gemm",
            Self::Conv1dGemm { .. } => "Conv1dGemm",
            Self::SiluMul { .. } => "SiluMul",
            Self::RotaryEmbedding { .. } => "RotaryEmbedding",
            Self::AddNormLinear { .. } => "AddNormLinear",
            Self::MoeGating { .. } => "MoeGating",
            Self::FusedAdainSnake { .. } => "FusedAdainSnake",
            Self::FusedUpsampleConv1d { .. } => "FusedUpsampleConv1d",
            Self::BiLstmCat { .. } => "BiLstmCat",
            Self::FusedMulAdd { .. } => "FusedMulAdd",
            Self::FusedSiGLU { .. } => "FusedSiGLU",
            Self::FusedGeGLU { .. } => "FusedGeGLU",
            Self::FusedLayerNormLinear { .. } => "FusedLayerNormLinear",
            Self::BatchNorm2d { .. } => "BatchNorm2d",
            Self::FusedInstanceNormMulAdd { .. } => "FusedInstanceNormMulAdd",
            Self::FusedSnakeInstanceNorm { .. } => "FusedSnakeInstanceNorm",
            Self::FusedConv1dActivation { .. } => "FusedConv1dActivation",
            Self::FusedConv1dSnakeNorm { .. } => "FusedConv1dSnakeNorm",
            Self::FusedAddInstanceNormConv1x1 { .. } => "FusedAddInstanceNormConv1x1",
            Self::FusedConvTranspose1dActivation { .. } => "FusedConvTranspose1dActivation",
            Self::FusedConv1dSnakeNormResBlock { .. } => "FusedConv1dSnakeNormResBlock",
            Self::NormActivConvTranspose1d { .. } => "NormActivConvTranspose1d",
            Self::FusedInstanceNormConv1d { .. } => "FusedInstanceNormConv1d",
            Self::FusedConv1dInstanceNorm { .. } => "FusedConv1dInstanceNorm",
            Self::FusedLinearLayerNorm { .. } => "FusedLinearLayerNorm",
            Self::FusedResBlockChain { .. } => "FusedResBlockChain",
        }
    }

    /// Estimated number of actual Metal kernel launches for this NativeOp.
    ///
    /// Most fused NativeOps (AdainSnake, FlashAttention, etc.) are single
    /// Metal dispatches. Some, like `LayerNorm` (decomposed IR graph) and
    /// `FusedResBlock` (sequenced sub-ops), launch multiple kernels internally.
    ///
    /// Used by `CompiledModel::num_metal_dispatches()` for accurate dispatch
    /// counting. Part of #2937.
    pub fn estimated_metal_dispatches(&self) -> usize {
        match self {
            // Single fused Metal kernel.
            Self::LstmSequence { .. } => 1,
            Self::InstanceNorm { .. } => 1,
            Self::AdainSnake { .. } => 1,
            Self::AdainLeakyRelu { .. } => 1,
            Self::AdaLayerNorm { .. } => 1,
            Self::FlashAttention { .. } => 1,
            Self::MaxPool1d { .. } => 1,
            Self::LinearActivation { .. } => 1,
            Self::Int8Gemm { .. } => 1,

            // Fused AdaIN+Snake: single Metal dispatch (#4252).
            Self::FusedAdainSnake { .. } => 1,

            // Fused Upsample1d + Conv1d: single Metal dispatch (#4310).
            // True single-kernel fusion — upsample + conv in one MSL kernel.
            Self::FusedUpsampleConv1d { .. } => 1,

            // Fused norm + GEMM: single dispatch (scalar GEMM fallback) or
            // two dispatches (norm-only → simdgroup GEMM) when dims qualify.
            // Use compile-time shape to predict the runtime routing.
            Self::NormLinear {
                input_shape,
                hidden_dim,
                out_features,
                ..
            } => norm_linear_dispatches(input_shape, *hidden_dim, *out_features),

            // Fused add + LayerNorm (single dispatch, #1815 Tier 5 D2).
            Self::AddLayerNorm { .. } => 1,

            // No GPU computation — returns a pre-uploaded buffer.
            Self::ConstantWeight { .. } => 0,

            // Blelloch prefix sum: single-pass (axis <= 256) = 1 dispatch,
            // multi-pass (axis > 256) = 3 dispatches.
            Self::Cumsum { dim, input_shape } => {
                let axis_size = input_shape.get(*dim).copied().unwrap_or(1);
                if axis_size <= 256 {
                    1
                } else {
                    3
                }
            }

            // Fused single-dispatch kernel (#2937).
            Self::LayerNorm { .. } => 1,
            Self::ChannelsFirstLayerNorm { .. } => 1,

            // Both LeakyRelu and Snake: 2 dispatches (stats + fused_norm_conv).
            Self::NormActivConv1d { .. } => 2,

            // Conv-stats fusion (#1815 Tier 2): phase 1 conv writes output
            // stats in its epilogue, phase 2 conv uses precomputed stats.
            // 3 dispatches: p1_stats + p1_conv_with_stats + p2_conv.
            // Style proj adds 2 Linear projections × (1 matmul + 1 bias_add) = 4.
            // Batched style_batch_offset: 0 (zero-copy narrow from batch output).
            Self::FusedResBlock {
                style_proj,
                style_batch_offset,
                ..
            } => {
                let base = 3; // stats + conv_with_stats + conv_precomputed
                let proj = match (style_proj, style_batch_offset) {
                    (Some(_), _) => 4, // unbatched: 2 projections × 2 dispatches
                    (_, Some(_)) => 0, // batched: zero-copy narrow from batch output
                    (None, None) => 0, // pre-computed gamma/beta
                };
                base + proj
            }

            // Batched linear projection: 1 fused matmul+bias + 1 narrow (#3269, #3351 D5).
            // When simdgroup qualifies (typical for attention dims), matmul+bias is
            // fused into a single dispatch. Fallback: 3 (matmul + bias_add + narrow).
            Self::BatchedLinearProjection { .. } => 2,

            // GPU narrow dispatch from batched projection temp (#3269).
            Self::ProjectionSlice { .. } => 1,

            // Batched style projection: 1 matmul + 1 bias_add (#1815 Tier 1).
            Self::BatchedStyleProjection { .. } => 2,

            // Conv1d GEMM: dispatch count depends on kernel size and stride.
            //
            // K=3, stride=1, dilation=1: direct sliding-window kernel (#4264).
            //   1 dispatch (conv) + 1 optional (bias add) = 1 or 2.
            //
            // Other shapes: im2col + GEMM (+ optional bias broadcast add).
            //   2 dispatches (im2col + GEMM) + 1 optional (bias add) = 2 or 3.
            //
            // Part of #3351 Tier 1, #4264.
            Self::Conv1dGemm {
                has_bias,
                kernel_size,
                stride,
                dilation,
                groups,
                ..
            } => {
                let uses_direct_k3 =
                    *kernel_size == 3 && *stride == 1 && *dilation == 1 && *groups == 1;
                if uses_direct_k3 {
                    // Direct K=3 path: 1 conv dispatch + optional bias add.
                    if *has_bias {
                        2
                    } else {
                        1
                    }
                } else {
                    // im2col + GEMM path: 2 dispatches + optional bias add.
                    if *has_bias {
                        3
                    } else {
                        2
                    }
                }
            }

            // Fused SiLU-Mul: single elementwise kernel dispatch (#3521).
            Self::SiluMul { .. } => 1,

            // Fused RoPE: single dispatch graph (reshape + narrow + mul + sub/add + concat).
            // Part of #3526.
            Self::RotaryEmbedding { .. } => 1,

            // Fused Add + LayerNorm + Linear: same dispatch count as NormLinear
            // (scalar fallback = 1, simdgroup = 2). Part of #3351 T2.1.
            Self::AddNormLinear {
                input_shape,
                hidden_dim,
                out_features,
                ..
            } => norm_linear_dispatches(input_shape, *hidden_dim, *out_features),

            // MoE gating: composite CPU dispatch via DynTensor ops.
            // Gating (5): linear + softmax + topk + sum_keepdim + div.
            // Per active expert (~5): index_select + expert_fwd(3) + broadcast_mul + index_add.
            // Estimate: 5 + top_k * 5. Part of #3542.
            Self::MoeGating { top_k, .. } => 5 + top_k * 5,

            // BiLstmCat: forward LSTM + reverse LSTM + concatenation = 3 dispatches.
            // Each LSTM direction is a single fused kernel; cat is elementwise.
            Self::BiLstmCat { .. } => 3,

            // Fused MulAdd: single elementwise kernel dispatch (hardware FMA).
            Self::FusedMulAdd { .. } => 1,

            // Fused SiGLU (Swish): single elementwise kernel dispatch.
            Self::FusedSiGLU { .. } => 1,

            // Fused GeGLU: single elementwise kernel dispatch.
            Self::FusedGeGLU { .. } => 1,

            // Fused LayerNorm + Linear: same dispatch count as NormLinear
            // (scalar fallback = 1, simdgroup = 2). Part of #4252.
            Self::FusedLayerNormLinear {
                input_shape,
                hidden_dim,
                out_features,
                ..
            } => norm_linear_dispatches(input_shape, *hidden_dim, *out_features),

            // Fused BatchNorm2d: single Metal dispatch (#4324).
            // Per-element with per-channel parameters, no reduction.
            Self::BatchNorm2d { .. } => 1,

            // Fused InstanceNorm + Mul + Add: single Metal dispatch (#4252).
            Self::FusedInstanceNormMulAdd { .. } => 1,

            // Fused Snake + InstanceNorm: single Metal dispatch (#4264).
            // Snake activation + Welford reduction in one kernel.
            Self::FusedSnakeInstanceNorm { .. } => 1,

            // Fused Conv1d + Activation: single Metal dispatch (#4264).
            // Conv1d accumulation + activation epilogue in one kernel.
            Self::FusedConv1dActivation { .. } => 1,

            // Fused Add + InstanceNorm + Conv1d(K=1): single logical dispatch.
            // Sequences add + instance_norm + conv1d(1x1) via lazy command buffer.
            // Part of #4264.
            Self::FusedAddInstanceNormConv1x1 { .. } => 1,

            // Fused Conv1d + Snake + InstanceNorm: conv1d → snake → instance_norm
            // in a single logical NativeOp. Batched via lazy command buffer.
            // Part of #4264.
            Self::FusedConv1dSnakeNorm { .. } => 1,

            // Fused ConvTranspose1d + Activation: single logical dispatch.
            // Conv + activation batched via lazy command buffer. Part of #4264.
            Self::FusedConvTranspose1dActivation { .. } => 1,

            // Fused 2x (Conv1d + Snake + InstanceNorm) + residual add:
            // Internally sequences 2x FusedConv1dSnakeNorm + add. Each phase
            // is 1 logical dispatch, plus 1 for the residual add = 3 total.
            // But these batch into the lazy command buffer. Part of #4264.
            Self::FusedConv1dSnakeNormResBlock { .. } => 3,

            // Fused NormActiv + ConvTranspose1d: same structure as NormActivConv1d
            // but with transposed convolution. 2 dispatches (stats + fused conv).
            // Part of #4264.
            Self::NormActivConvTranspose1d { .. } => 2,

            // Fused InstanceNorm + Conv1d: 2 dispatches (stats + fused norm-conv).
            // Part of #4264.
            Self::FusedInstanceNormConv1d { .. } => 2,

            // Fused Conv1d + InstanceNorm: 2 dispatches (conv + fused stats-norm).
            // Part of #4264.
            Self::FusedConv1dInstanceNorm { .. } => 2,

            // Fused Linear + LayerNorm: 2 dispatches (matmul + fused stats-norm).
            // Part of #4264.
            Self::FusedLinearLayerNorm { .. } => 2,

            // Chained FusedResBlocks: N blocks × 3 dispatches each.
            // The savings come from reduced NativeOp dispatch overhead, not
            // fewer Metal kernel launches. Part of #4264.
            Self::FusedResBlockChain { blocks, .. } => blocks.len() * 3,
        }
    }

    /// Estimated number of `get_or_create_batch()` calls (encoding events).
    ///
    /// Unlike [`estimated_metal_dispatches()`] which counts internal Metal
    /// kernel launches (including sub-encoders within a single batch),
    /// this method counts the number of distinct command buffer batch
    /// creations — the metric that tracks `TOTAL_ENCODINGS + TOTAL_BLITS`
    /// at runtime. Part of #1815 D5.2.
    ///
    /// Key differences from `estimated_metal_dispatches()`:
    /// - LSTM: +1 for bias_ih + bias_hh combine (DynTensor GPU add)
    /// - Cumsum (multi-pass): 1 batch with 3 sub-encoders, not 3 batches
    /// - NormActivConv1d: 1 batch with 2 sub-encoders, not 2 batches
    /// - FusedResBlock: 2 batches (phase 1 + phase 2), not 3
    /// - MaxPool1d: 0 (CPU roundtrip, no compute dispatch)
    pub fn estimated_encoding_events(&self) -> usize {
        match self {
            // LSTM fused kernel = 1 encoding event.
            // PyTorch LSTMs always have separate bias_ih + bias_hh, requiring
            // a GPU add (bih.add(&bhh)) before the kernel = +1 encoding event.
            Self::LstmSequence { .. } => 2,

            // Single fused Metal kernel = 1 encoding event each.
            Self::InstanceNorm { .. } => 1,
            Self::AdainSnake { .. } => 1,
            Self::AdainLeakyRelu { .. } => 1,
            Self::AdaLayerNorm { .. } => 1,
            Self::FlashAttention { .. } => 1,
            Self::LinearActivation { .. } => 1,
            Self::AddLayerNorm { .. } => 1,
            Self::LayerNorm { .. } => 1,
            Self::ChannelsFirstLayerNorm { .. } => 1,
            Self::Int8Gemm { .. } => 1,

            // Fused AdaIN+Snake: single encoding event (#4252).
            Self::FusedAdainSnake { .. } => 1,

            // Fused Upsample1d + Conv1d: single encoding event (#4310).
            Self::FusedUpsampleConv1d { .. } => 1,

            // Fused path: 1 encoding. Simdgroup path: 2 (norm + GEMM).
            // Use compile-time shape to predict the runtime routing.
            Self::NormLinear {
                input_shape,
                hidden_dim,
                out_features,
                ..
            } => norm_linear_dispatches(input_shape, *hidden_dim, *out_features),

            // No GPU computation — returns a pre-uploaded buffer.
            Self::ConstantWeight { .. } => 0,

            // CPU roundtrip (GPU→CPU→GPU via to_device). No compute dispatch.
            Self::MaxPool1d { .. } => 0,

            // Blelloch prefix sum: single get_or_create_batch() regardless of
            // pass count. Multi-pass (axis > 256) uses 3 sub-encoders in 1 batch.
            Self::Cumsum { .. } => 1,

            // 1 get_or_create_batch() with 2 sub-encoders (stats + conv).
            Self::NormActivConv1d { .. } => 1,

            // Phase 1 (conv-with-stats): 1 get_or_create_batch.
            // Phase 2 (conv-precomputed): 1 get_or_create_batch.
            // Style proj adds DynTensor matmul + bias_add per projection (2 each).
            Self::FusedResBlock {
                style_proj,
                style_batch_offset,
                ..
            } => {
                let base = 2; // phase 1 + phase 2
                let proj = match (style_proj, style_batch_offset) {
                    (Some(_), _) => 4, // unbatched: 2 projections × 2 encoding events
                    (_, Some(_)) => 0, // batched: zero-copy narrow
                    (None, None) => 0, // pre-computed gamma/beta
                };
                base + proj
            }

            // Batched linear projection: 1 fused matmul+bias + 1 narrow (#3269, #3351 D5).
            // Simdgroup path fuses matmul+bias into 1 encoding event.
            // Fallback: 3 (matmul + bias_add + narrow).
            Self::BatchedLinearProjection { .. } => 2,

            // GPU narrow dispatch from batched projection temp (#3269).
            Self::ProjectionSlice { .. } => 1,

            // 1 matmul + 1 bias_add = 2 encoding events.
            Self::BatchedStyleProjection { .. } => 2,

            // Conv1d GEMM: encoding events depend on kernel size.
            // K=3 direct path: 1 conv + optional 1 bias = 1 or 2.
            // im2col path: im2col + GEMM + optional bias = 2 or 3.
            // Part of #3351 Tier 1, #4264.
            Self::Conv1dGemm {
                has_bias,
                kernel_size,
                stride,
                dilation,
                groups,
                ..
            } => {
                let uses_direct_k3 =
                    *kernel_size == 3 && *stride == 1 && *dilation == 1 && *groups == 1;
                if uses_direct_k3 {
                    if *has_bias {
                        2
                    } else {
                        1
                    }
                } else if *has_bias {
                    3
                } else {
                    2
                }
            }

            // Fused SiLU-Mul: single encoding event (#3521).
            Self::SiluMul { .. } => 1,

            // Fused RoPE: single encoding event (#3526).
            Self::RotaryEmbedding { .. } => 1,

            // Fused Add + LayerNorm + Linear: same encoding events as NormLinear.
            Self::AddNormLinear {
                input_shape,
                hidden_dim,
                out_features,
                ..
            } => norm_linear_dispatches(input_shape, *hidden_dim, *out_features),

            // MoE gating: composite CPU dispatch, encoding events match metal dispatches.
            Self::MoeGating { top_k, .. } => 5 + top_k * 5,

            // BiLstmCat: fwd LSTM bias combine + fwd kernel + rev LSTM bias combine +
            // rev kernel + cat = 5 encoding events.
            Self::BiLstmCat { .. } => 5,

            // Fused MulAdd: single encoding event.
            Self::FusedMulAdd { .. } => 1,

            // Fused SiGLU (Swish): single encoding event.
            Self::FusedSiGLU { .. } => 1,

            // Fused GeGLU: single encoding event.
            Self::FusedGeGLU { .. } => 1,

            // Fused LayerNorm + Linear: same encoding events as NormLinear.
            Self::FusedLayerNormLinear {
                input_shape,
                hidden_dim,
                out_features,
                ..
            } => norm_linear_dispatches(input_shape, *hidden_dim, *out_features),

            // Fused BatchNorm2d: single encoding event (#4324).
            Self::BatchNorm2d { .. } => 1,

            // Fused InstanceNorm + Mul + Add: single encoding event (#4252).
            Self::FusedInstanceNormMulAdd { .. } => 1,

            // Fused Snake + InstanceNorm: single encoding event (#4264).
            Self::FusedSnakeInstanceNorm { .. } => 1,

            // Fused Conv1d + Activation: single encoding event (#4264).
            Self::FusedConv1dActivation { .. } => 1,

            // Fused Add + InstanceNorm + Conv1d(K=1): single encoding event (#4264).
            Self::FusedAddInstanceNormConv1x1 { .. } => 1,

            // Fused Conv1d + Snake + InstanceNorm: single encoding event (#4264).
            Self::FusedConv1dSnakeNorm { .. } => 1,

            // Fused ConvTranspose1d + Activation: single encoding event (#4264).
            Self::FusedConvTranspose1dActivation { .. } => 1,

            // Fused 2x (Conv1d + Snake + InstanceNorm) + add: 2 encoding events
            // (phase 1 batch + phase 2 batch; add batches with phase 2).
            Self::FusedConv1dSnakeNormResBlock { .. } => 2,

            // Fused NormActiv + ConvTranspose1d: 1 encoding event (batched norm + conv).
            // Same as NormActivConv1d. Part of #4264.
            Self::NormActivConvTranspose1d { .. } => 1,

            // Fused InstanceNorm + Conv1d: 1 encoding event. Part of #4264.
            Self::FusedInstanceNormConv1d { .. } => 1,

            // Fused Conv1d + InstanceNorm: 1 encoding event. Part of #4264.
            Self::FusedConv1dInstanceNorm { .. } => 1,

            // Fused Linear + LayerNorm: 1 encoding event. Part of #4264.
            Self::FusedLinearLayerNorm { .. } => 1,

            // Chained FusedResBlocks: N blocks × 2 encoding events each
            // (phase 1 + phase 2 per block). Part of #4264.
            Self::FusedResBlockChain { blocks, .. } => blocks.len() * 2,
        }
    }

    /// Graph `NodeId`s of external inputs set at creation time.
    ///
    /// Returns `Some(&[NodeId])` for NativeOps that carry their own
    /// edge dependencies (overriding the graph-topology-based edge_map).
    /// The edge_map builder uses this to resolve edges generically,
    /// eliminating per-NativeOp patches. Part of #3261.
    #[must_use]
    pub fn external_node_ids(&self) -> Option<&[u64]> {
        match self {
            Self::NormActivConv1d {
                external_node_ids: Some(ids),
                ..
            }
            | Self::AdainSnake {
                external_node_ids: Some(ids),
                ..
            }
            | Self::AdainLeakyRelu {
                external_node_ids: Some(ids),
                ..
            }
            | Self::FusedInstanceNormMulAdd {
                external_node_ids: Some(ids),
                ..
            }
            | Self::FusedAdainSnake {
                external_node_ids: Some(ids),
                ..
            }
            | Self::NormActivConvTranspose1d {
                external_node_ids: Some(ids),
                ..
            } => Some(ids),
            _ => None,
        }
    }

    /// Collect step indices that this NativeOp reads directly (bypassing edge_map).
    ///
    /// Used by the D4 elementwise fusion pass to prevent fusing steps that
    /// are consumed by NativeOps via direct buffer access. Without this,
    /// `effective_counts` misses these consumers and may fuse a step whose
    /// buffer is read by a FusedResBlock/BatchedStyleProjection. #3385.
    pub fn collect_direct_step_deps(&self, out: &mut Vec<usize>) {
        match self {
            Self::FusedResBlock {
                input_steps,
                shortcut_step,
                pool_step,
                ..
            } => {
                out.extend_from_slice(input_steps);
                if let Some(sc) = shortcut_step {
                    out.push(*sc);
                }
                if let Some(ps) = pool_step {
                    out.push(*ps);
                }
            }
            Self::BatchedStyleProjection { style_step, .. } => {
                out.push(*style_step);
            }
            Self::ProjectionSlice { source_step, .. } => {
                out.push(*source_step);
            }
            Self::FusedConv1dSnakeNormResBlock { x_step, .. } => {
                out.push(*x_step);
            }
            Self::FusedResBlockChain {
                input_steps,
                first_shortcut_step,
                ..
            } => {
                out.extend_from_slice(input_steps);
                if let Some(sc) = first_shortcut_step {
                    out.push(*sc);
                }
            }
            _ => {}
        }
    }
}

/// Predict NormLinear/AddNormLinear dispatch count from compile-time shapes.
///
/// Returns 2 when dimensions qualify for the simdgroup GEMM path (norm-only
/// dispatch + simdgroup GEMM dispatch), 1 when falling back to the scalar
/// fused kernel. Mirrors `dyn_tensor_metal_matmul_simd::should_use_simdgroup`.
fn norm_linear_dispatches(input_shape: &[usize], hidden_dim: usize, out_features: usize) -> usize {
    let flat_rows = input_shape.iter().rev().skip(1).product::<usize>().max(1);
    let m = flat_rows;
    let k = hidden_dim;
    let n = out_features;
    if m.is_multiple_of(8)
        && k.is_multiple_of(8)
        && n.is_multiple_of(8)
        && m * n >= 16_384
        && k >= 128
    {
        2
    } else {
        1
    }
}

/// Number of [`NativeOpKind`] variants. Update when adding a new variant.
///
/// Adding a NativeOp requires updating:
/// 1. `variant_name()` and `estimated_metal_dispatches()` (compile error, no catch-all)
/// 2. `execute_native_op` in nn-metal (has `_ =>` catch-all — silent runtime failure!)
/// 3. `is_compute_native_op` / `is_passthrough_safe` (autocast classification)
/// 4. This constant (test failure reminds you of items 2-3)
///
/// Part of design `2026-03-23-api-health-execute-dispatch-consistency.md` F2.
#[allow(dead_code)] // Used by #[cfg(test)] and #[cfg(kani)] harnesses.
pub(crate) const KNOWN_NATIVE_OP_COUNT: usize = 45;

#[cfg(test)]
mod tests {
    use super::super::NativeOpKind;

    fn make_batched_proj(has_bias: bool) -> NativeOpKind {
        NativeOpKind::BatchedLinearProjection {
            in_features: 768,
            total_out_features: 768 * 3,
            projection_sizes: vec![768, 768, 768],
            has_bias,
            input_shape: vec![2, 16, 768],
        }
    }

    #[test]
    fn test_batched_linear_projection_dispatch_count_with_bias() {
        let op = make_batched_proj(true);
        // 1 fused matmul+bias + 1 narrow = 2 dispatches (#3351 D5).
        assert_eq!(op.estimated_metal_dispatches(), 2);
        assert_eq!(op.estimated_encoding_events(), 2);
    }

    #[test]
    fn test_batched_linear_projection_dispatch_count_no_bias() {
        let op = make_batched_proj(false);
        // 1 matmul + 1 narrow = 2 dispatches.
        assert_eq!(op.estimated_metal_dispatches(), 2);
        assert_eq!(op.estimated_encoding_events(), 2);
    }

    #[test]
    fn test_projection_slice_dispatch_count() {
        let op = NativeOpKind::ProjectionSlice {
            source_step: 0,
            dim: 2,
            start: 768,
            length: 768,
            output_shape: vec![2, 16, 768],
        };
        assert_eq!(op.estimated_metal_dispatches(), 1);
        assert_eq!(op.estimated_encoding_events(), 1);
    }

    // -- NormLinear / AddNormLinear shape-aware dispatch counts --

    #[test]
    fn test_norm_linear_simdgroup_eligible() {
        // Kokoro PLBert dims: [2, 16, 768] → m=32, k=768, n=768.
        // All multiples of 8, m*n=24576 >= 16384, k=768 >= 128 → simdgroup → 2.
        let op = NativeOpKind::NormLinear {
            norm_kind: super::super::FusedNormKind::LayerNorm,
            eps: 1e-5,
            input_shape: vec![2, 16, 768],
            hidden_dim: 768,
            out_features: 768,
            has_bias: true,
        };
        assert_eq!(op.estimated_metal_dispatches(), 2);
        assert_eq!(op.estimated_encoding_events(), 2);
    }

    #[test]
    fn test_norm_linear_scalar_fallback_small_k() {
        // k=64 < 128 → scalar fallback → 1 dispatch.
        let op = NativeOpKind::NormLinear {
            norm_kind: super::super::FusedNormKind::LayerNorm,
            eps: 1e-5,
            input_shape: vec![1, 8, 64],
            hidden_dim: 64,
            out_features: 64,
            has_bias: false,
        };
        assert_eq!(op.estimated_metal_dispatches(), 1);
        assert_eq!(op.estimated_encoding_events(), 1);
    }

    #[test]
    fn test_norm_linear_scalar_fallback_non_aligned() {
        // n=100 not multiple of 8 → scalar fallback → 1 dispatch.
        let op = NativeOpKind::NormLinear {
            norm_kind: super::super::FusedNormKind::LayerNorm,
            eps: 1e-5,
            input_shape: vec![1, 8, 256],
            hidden_dim: 256,
            out_features: 100,
            has_bias: true,
        };
        assert_eq!(op.estimated_metal_dispatches(), 1);
        assert_eq!(op.estimated_encoding_events(), 1);
    }

    // -- Variant count safety gate (F2 of execute-dispatch-consistency design) --

    /// All known [`NativeOpKind`] variant names, in enum declaration order.
    ///
    /// When adding a new variant, this list and [`KNOWN_NATIVE_OP_COUNT`]
    /// must be updated. The test failure message reminds you to also update:
    /// - `execute_native_op` match arms in nn-metal
    /// - `is_compute_native_op` / `is_passthrough_safe` (autocast classification)
    const ALL_VARIANT_NAMES: [&str; super::KNOWN_NATIVE_OP_COUNT] = [
        "LstmSequence",
        "Cumsum",
        "InstanceNorm",
        "LayerNorm",
        "ChannelsFirstLayerNorm",
        "AddLayerNorm",
        "AdainSnake",
        "AdainLeakyRelu",
        "AdaLayerNorm",
        "FlashAttention",
        "MaxPool1d",
        "ConstantWeight",
        "FusedResBlock",
        "NormActivConv1d",
        "LinearActivation",
        "NormLinear",
        "BatchedLinearProjection",
        "ProjectionSlice",
        "BatchedStyleProjection",
        "Int8Gemm",
        "Conv1dGemm",
        "SiluMul",
        "RotaryEmbedding",
        "AddNormLinear",
        "MoeGating",
        "FusedAdainSnake",
        "FusedUpsampleConv1d",
        "BiLstmCat",
        "FusedMulAdd",
        "FusedSiGLU",
        "FusedGeGLU",
        "BatchNorm2d",
        "FusedLayerNormLinear",
        "FusedInstanceNormMulAdd",
        "FusedSnakeInstanceNorm",
        "FusedConv1dActivation",
        "FusedAddInstanceNormConv1x1",
        "FusedConv1dSnakeNorm",
        "FusedConvTranspose1dActivation",
        "FusedConv1dSnakeNormResBlock",
        "NormActivConvTranspose1d",
        "FusedInstanceNormConv1d",
        "FusedConv1dInstanceNorm",
        "FusedLinearLayerNorm",
        "FusedResBlockChain",
    ];

    #[test]
    fn test_native_op_variant_count() {
        // Compile-time: array size must equal KNOWN_NATIVE_OP_COUNT.
        // Runtime: variant_name() on each minimal instance must match.
        //
        // If a new variant is added without updating:
        // - variant_name() match → compile error (no catch-all)
        // - ALL_VARIANT_NAMES array size → compile error (size mismatch)
        // - This test → reminds to update execute_native_op (nn-metal)
        let instances: Vec<NativeOpKind> = vec![
            NativeOpKind::LstmSequence {
                hidden_size: 64,
                input_shape: vec![1, 1, 64],
                h_shape: vec![1, 64],
                reverse: false,
            },
            NativeOpKind::Cumsum {
                dim: 0,
                input_shape: vec![4],
            },
            NativeOpKind::InstanceNorm {
                eps: 1e-5,
                input_shape: vec![1, 2, 4],
            },
            NativeOpKind::LayerNorm {
                eps: 1e-5,
                input_shape: vec![1, 4],
                hidden_dim: 4,
            },
            NativeOpKind::ChannelsFirstLayerNorm {
                eps: 1e-5,
                input_shape: vec![1, 4, 8],
                channels: 4,
                leaky_relu_slope: None,
            },
            NativeOpKind::AddLayerNorm {
                eps: 1e-5,
                input_shape: vec![1, 4],
                hidden_dim: 4,
            },
            NativeOpKind::AdainSnake {
                eps: 1e-5,
                input_shape: vec![1, 2, 4],
                channels: 2,
                residual_gamma: false,
                external_node_ids: None,
            },
            NativeOpKind::AdainLeakyRelu {
                eps: 1e-5,
                slope: 0.01,
                input_shape: vec![1, 2, 4],
                external_node_ids: None,
            },
            NativeOpKind::AdaLayerNorm {
                eps: 1e-5,
                input_shape: vec![1, 4, 8],
                hidden_dim: 8,
            },
            NativeOpKind::FlashAttention {
                scale: 0.125,
                causal: false,
                q_shape: vec![1, 1, 4, 8],
                k_shape: vec![1, 1, 4, 8],
                output_shape: vec![1, 1, 4, 8],
                input_layout: Default::default(),
            },
            NativeOpKind::MaxPool1d {
                kernel_size: 3,
                stride: 1,
                padding: 1,
                input_shape: vec![1, 2, 8],
            },
            NativeOpKind::ConstantWeight {
                name: "test".into(),
                shape: vec![4],
            },
            NativeOpKind::FusedResBlock {
                phase1: super::super::NormActivConv1dParams {
                    activation: super::super::NormActivation::Snake,
                    eps: 1e-5,
                    conv_dilation: 1,
                    conv_padding: 1,
                    input_shape: vec![1, 4, 8],
                    output_channels: 4,
                    kernel_size: 3,
                },
                phase2: super::super::NormActivConv1dParams {
                    activation: super::super::NormActivation::Snake,
                    eps: 1e-5,
                    conv_dilation: 1,
                    conv_padding: 1,
                    input_shape: vec![1, 4, 8],
                    output_channels: 4,
                    kernel_size: 3,
                },
                input_steps: vec![0, 1, 2, 3, 4],
                residual_scale: 1.0,
                style_proj: None,
                shortcut_step: None,
                pool_step: None,
                style_batch_offset: None,
            },
            NativeOpKind::NormActivConv1d {
                activation: super::super::NormActivation::Snake,
                eps: 1e-5,
                conv_dilation: 1,
                conv_padding: 1,
                input_shape: vec![1, 4, 8],
                output_channels: 4,
                kernel_size: 3,
                external_node_ids: None,
            },
            NativeOpKind::LinearActivation {
                activation: super::super::GemmActivation::Relu,
                in_features: 4,
                out_features: 8,
                has_bias: true,
                input_shape: vec![1, 4],
            },
            NativeOpKind::NormLinear {
                norm_kind: super::super::FusedNormKind::LayerNorm,
                eps: 1e-5,
                input_shape: vec![1, 4],
                hidden_dim: 4,
                out_features: 8,
                has_bias: true,
            },
            NativeOpKind::BatchedLinearProjection {
                in_features: 64,
                total_out_features: 192,
                projection_sizes: vec![64, 64, 64],
                has_bias: true,
                input_shape: vec![1, 4, 64],
            },
            NativeOpKind::ProjectionSlice {
                source_step: 0,
                dim: 2,
                start: 64,
                length: 64,
                output_shape: vec![1, 4, 64],
            },
            NativeOpKind::BatchedStyleProjection {
                blocks: vec![],
                style_dim: 128,
                total_out: 256,
                style_step: 0,
            },
            NativeOpKind::Int8Gemm {
                in_features: 64,
                out_features: 128,
                has_bias: true,
                input_shape: vec![1, 4, 64],
            },
            NativeOpKind::Conv1dGemm {
                input_shape: vec![1, 4, 16],
                out_channels: 8,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
                has_bias: true,
            },
            NativeOpKind::SiluMul {
                input_shape: vec![1, 8, 256],
            },
            NativeOpKind::RotaryEmbedding {
                head_dim: 64,
                input_shape: vec![1, 8, 16, 64],
            },
            NativeOpKind::AddNormLinear {
                eps: 1e-5,
                input_shape: vec![1, 4],
                hidden_dim: 4,
                out_features: 8,
                has_bias: true,
            },
            NativeOpKind::MoeGating {
                num_experts: 8,
                top_k: 2,
                input_shape: vec![1, 4, 64],
            },
            NativeOpKind::FusedAdainSnake {
                eps: 1e-5,
                input_shape: vec![1, 4, 16],
                channels: 4,
                external_node_ids: None,
            },
            NativeOpKind::FusedUpsampleConv1d {
                upsample_factor: 2,
                in_channels: 4,
                out_channels: 8,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                input_shape: vec![1, 4, 16],
            },
            NativeOpKind::BiLstmCat {
                hidden_size: 64,
                input_shape: vec![4, 1, 128],
                h_shape: vec![1, 64],
                fwd_lstm_step: 0,
                rev_lstm_step: 1,
            },
            NativeOpKind::FusedMulAdd {
                input_shape: vec![1, 8, 256],
            },
            NativeOpKind::FusedSiGLU {
                input_shape: vec![1, 8, 256],
            },
            NativeOpKind::FusedGeGLU {
                input_shape: vec![1, 8, 256],
            },
            NativeOpKind::BatchNorm2d {
                eps: 1e-5,
                num_channels: 4,
                input_shape: vec![1, 4, 8, 8],
                has_weight: true,
                has_bias: true,
            },
            NativeOpKind::FusedLayerNormLinear {
                eps: 1e-5,
                input_shape: vec![1, 4],
                hidden_dim: 4,
                out_features: 8,
                has_bias: true,
            },
            NativeOpKind::FusedInstanceNormMulAdd {
                eps: 1e-5,
                input_shape: vec![1, 4, 16],
                channels: 4,
                external_node_ids: None,
            },
            NativeOpKind::FusedSnakeInstanceNorm {
                eps: 1e-5,
                input_shape: vec![1, 4, 16],
                channels: 4,
            },
            NativeOpKind::FusedConv1dActivation {
                activation: super::super::ConvActivation::Relu,
                out_channels: 8,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
                has_bias: true,
                input_shape: vec![1, 4, 16],
                pre_activation: false,
            },
            NativeOpKind::FusedAddInstanceNormConv1x1 {
                eps: 1e-5,
                input_shape: vec![1, 4, 16],
                in_channels: 4,
                out_channels: 8,
                has_bias: true,
            },
            NativeOpKind::FusedConv1dSnakeNorm {
                out_channels: 8,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
                has_bias: true,
                eps: 1e-5,
                input_shape: vec![1, 4, 16],
            },
            NativeOpKind::FusedConvTranspose1dActivation {
                activation: super::super::ConvActivation::LeakyRelu { slope: 0.2 },
                out_channels: 8,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                dilation: 1,
                groups: 1,
                output_padding: 1,
                has_bias: true,
                input_shape: vec![1, 4, 16],
            },
            NativeOpKind::FusedConv1dSnakeNormResBlock {
                phase1_out_channels: 8,
                phase1_kernel_size: 3,
                phase1_padding: 1,
                phase1_dilation: 1,
                phase1_has_bias: true,
                phase2_out_channels: 8,
                phase2_kernel_size: 3,
                phase2_padding: 1,
                phase2_dilation: 1,
                phase2_has_bias: true,
                eps: 1e-5,
                residual_scale: 1.0,
                input_shape: vec![1, 4, 16],
                x_step: 0,
            },
            NativeOpKind::NormActivConvTranspose1d {
                activation: super::super::NormActivation::LeakyRelu { slope: 0.2 },
                eps: 1e-5,
                kernel_size: 4,
                stride: 2,
                padding: 1,
                dilation: 1,
                groups: 1,
                output_padding: 1,
                output_channels: 8,
                input_shape: vec![1, 4, 16],
                external_node_ids: None,
            },
            NativeOpKind::FusedInstanceNormConv1d {
                eps: 1e-5,
                out_channels: 8,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
                has_bias: true,
                input_shape: vec![1, 4, 16],
            },
            NativeOpKind::FusedConv1dInstanceNorm {
                eps: 1e-5,
                out_channels: 8,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
                has_bias: true,
                input_shape: vec![1, 4, 16],
            },
            NativeOpKind::FusedLinearLayerNorm {
                in_features: 64,
                out_features: 32,
                has_bias: true,
                eps: 1e-5,
                input_shape: vec![1, 64],
            },
            NativeOpKind::FusedResBlockChain {
                blocks: vec![
                    super::super::ResBlockChainEntry::new(
                        super::super::NormActivConv1dParams::new(
                            super::super::NormActivation::Snake,
                            1e-5,
                            1,
                            1,
                            vec![1, 4, 8],
                            4,
                            3,
                        ),
                        super::super::NormActivConv1dParams::new(
                            super::super::NormActivation::Snake,
                            1e-5,
                            1,
                            1,
                            vec![1, 4, 8],
                            4,
                            3,
                        ),
                        1.0,
                    ),
                    super::super::ResBlockChainEntry::new(
                        super::super::NormActivConv1dParams::new(
                            super::super::NormActivation::Snake,
                            1e-5,
                            3,
                            3,
                            vec![1, 4, 8],
                            4,
                            3,
                        ),
                        super::super::NormActivConv1dParams::new(
                            super::super::NormActivation::Snake,
                            1e-5,
                            1,
                            1,
                            vec![1, 4, 8],
                            4,
                            3,
                        ),
                        1.0,
                    ),
                ],
                input_steps: vec![0, 1],
                style_batch_offsets: vec![
                    super::super::StyleBatchOffset::new(0, 4, 4),
                    super::super::StyleBatchOffset::new(16, 4, 4),
                ],
                first_shortcut_step: None,
            },
        ];

        assert_eq!(
            instances.len(),
            super::KNOWN_NATIVE_OP_COUNT,
            "instance count must match KNOWN_NATIVE_OP_COUNT"
        );

        for (instance, expected_name) in instances.iter().zip(ALL_VARIANT_NAMES.iter()) {
            assert_eq!(
                instance.variant_name(),
                *expected_name,
                "variant_name() mismatch for {expected_name}"
            );
        }
    }

    #[test]
    fn test_moe_gating_dispatch_count() {
        // 8 experts, top-2: 5 gating + 2*5 expert = 15 dispatches.
        let op = NativeOpKind::MoeGating {
            num_experts: 8,
            top_k: 2,
            input_shape: vec![4, 64],
        };
        assert_eq!(op.estimated_metal_dispatches(), 15);
        assert_eq!(op.estimated_encoding_events(), 15);
        assert_eq!(op.variant_name(), "MoeGating");
    }

    #[test]
    fn test_fused_adain_snake_dispatch_count() {
        let op = NativeOpKind::FusedAdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 256, 512],
            channels: 256,
            external_node_ids: None,
        };
        assert_eq!(op.estimated_metal_dispatches(), 1);
        assert_eq!(op.estimated_encoding_events(), 1);
        assert_eq!(op.variant_name(), "FusedAdainSnake");
    }

    // -- FusedResBlock dispatch counts (#3554) --

    fn make_kokoro_resblock_params(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> super::super::NormActivConv1dParams {
        super::super::NormActivConv1dParams {
            activation: super::super::NormActivation::Snake,
            eps: 1e-5,
            conv_dilation: dilation,
            conv_padding: dilation * (kernel_size - 1) / 2,
            input_shape: vec![1, channels, 256],
            output_channels: channels,
            kernel_size,
        }
    }

    /// Direct buffer path (no style_proj, no batch_offset): 3 dispatches.
    #[test]
    fn test_fused_resblock_direct_path_dispatch_count() {
        let op = NativeOpKind::FusedResBlock {
            phase1: make_kokoro_resblock_params(256, 3, 1),
            phase2: make_kokoro_resblock_params(256, 3, 1),
            input_steps: vec![0, 1, 2, 3, 4],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        };
        assert_eq!(op.estimated_metal_dispatches(), 3);
        assert_eq!(op.estimated_encoding_events(), 2);
        assert_eq!(op.variant_name(), "FusedResBlock");
    }

    /// Style projection path (unbatched): 3 + 4 = 7 dispatches.
    #[test]
    fn test_fused_resblock_style_proj_dispatch_count() {
        let op = NativeOpKind::FusedResBlock {
            phase1: make_kokoro_resblock_params(256, 3, 1),
            phase2: make_kokoro_resblock_params(256, 3, 1),
            input_steps: vec![0, 1],
            residual_scale: 1.0,
            style_proj: Some(super::super::StyleProjectionParams::new(256, 256, 128)),
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        };
        assert_eq!(op.estimated_metal_dispatches(), 7);
        assert_eq!(op.estimated_encoding_events(), 6);
    }

    /// Batched style_batch_offset path: 3 + 0 = 3 dispatches.
    #[test]
    fn test_fused_resblock_batch_offset_dispatch_count() {
        let op = NativeOpKind::FusedResBlock {
            phase1: make_kokoro_resblock_params(256, 3, 1),
            phase2: make_kokoro_resblock_params(256, 3, 1),
            input_steps: vec![0, 1],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: Some(super::super::StyleBatchOffset::new(0, 256, 256)),
        };
        assert_eq!(op.estimated_metal_dispatches(), 3);
        assert_eq!(op.estimated_encoding_events(), 2);
    }

    /// style_proj takes priority over style_batch_offset: 7 dispatches.
    #[test]
    fn test_fused_resblock_style_proj_priority_over_batch_offset() {
        let op = NativeOpKind::FusedResBlock {
            phase1: make_kokoro_resblock_params(256, 3, 1),
            phase2: make_kokoro_resblock_params(256, 3, 1),
            input_steps: vec![0, 1],
            residual_scale: 1.0,
            style_proj: Some(super::super::StyleProjectionParams::new(256, 256, 128)),
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: Some(super::super::StyleBatchOffset::new(0, 256, 256)),
        };
        // style_proj takes priority: 3 + 4 = 7.
        assert_eq!(op.estimated_metal_dispatches(), 7);
        assert_eq!(op.estimated_encoding_events(), 6);
    }

    /// F0 energy predictor: LeakyRelu with residual_scale = 1/sqrt(2).
    #[test]
    fn test_fused_resblock_f0_leaky_relu_dispatch_count() {
        let params = super::super::NormActivConv1dParams {
            activation: super::super::NormActivation::LeakyRelu { slope: 0.2 },
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 1,
            input_shape: vec![1, 256, 64],
            output_channels: 256,
            kernel_size: 3,
        };
        let op = NativeOpKind::FusedResBlock {
            phase1: params.clone(),
            phase2: params,
            input_steps: vec![0, 1],
            residual_scale: 1.0 / 2.0_f32.sqrt(),
            style_proj: Some(super::super::StyleProjectionParams::new(256, 256, 128)),
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        };
        assert_eq!(op.estimated_metal_dispatches(), 7);
        assert_eq!(op.estimated_encoding_events(), 6);
    }

    /// Representative Kokoro generator block shapes (Snake, channels=512).
    #[test]
    fn test_fused_resblock_kokoro_generator_shapes() {
        let op = NativeOpKind::FusedResBlock {
            phase1: make_kokoro_resblock_params(512, 7, 1),
            phase2: make_kokoro_resblock_params(512, 7, 3),
            input_steps: vec![0, 1],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: Some(super::super::StyleBatchOffset::new(0, 512, 512)),
        };
        assert_eq!(op.estimated_metal_dispatches(), 3);
        assert_eq!(op.estimated_encoding_events(), 2);
    }

    // -- Fused elementwise NativeOp dispatch counts --

    #[test]
    fn test_fused_mul_add_dispatch_count() {
        let op = NativeOpKind::FusedMulAdd {
            input_shape: vec![1, 8, 256],
        };
        assert_eq!(op.estimated_metal_dispatches(), 1);
        assert_eq!(op.estimated_encoding_events(), 1);
        assert_eq!(op.variant_name(), "FusedMulAdd");
    }

    #[test]
    fn test_fused_siglu_dispatch_count() {
        let op = NativeOpKind::FusedSiGLU {
            input_shape: vec![1, 8, 256],
        };
        assert_eq!(op.estimated_metal_dispatches(), 1);
        assert_eq!(op.estimated_encoding_events(), 1);
        assert_eq!(op.variant_name(), "FusedSiGLU");
    }

    #[test]
    fn test_fused_geglu_dispatch_count() {
        let op = NativeOpKind::FusedGeGLU {
            input_shape: vec![1, 8, 256],
        };
        assert_eq!(op.estimated_metal_dispatches(), 1);
        assert_eq!(op.estimated_encoding_events(), 1);
        assert_eq!(op.variant_name(), "FusedGeGLU");
    }
}
