// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Native GPU operation types for compiled trace execution.
//!
//! Contains [`NativeOpKind`], [`NormActivConv1dParams`], [`NormActivation`],
//! and [`StyleProjectionParams`] — extracted from `trace_compile_types.rs`.

/// Native operations that delegate to existing fused Metal kernels,
/// bypassing the IR → MSL code-generation path.
///
/// Each variant maps to a high-performance kernel implementation that
/// already exists in the eager execution path. The compiled model
/// executor dispatches to these kernels directly.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum NativeOpKind {
    /// Full-sequence LSTM: delegates to `gpu_lstm_sequence()`.
    ///
    /// Input 0: `[seq_len, batch, input_size]`
    /// Weights: `weight_ih`, `weight_hh`, optionally `bias_ih`+`bias_hh`, `h0`, `c0`
    LstmSequence {
        /// LSTM hidden size (determines threadgroup memory).
        hidden_size: usize,
        /// Input tensor shape `[seq_len, batch, input_size]`.
        input_shape: Vec<usize>,
        /// Hidden/cell state shape `[batch, hidden_size]`.
        h_shape: Vec<usize>,
        /// Process sequence in reverse order (for BiLSTM backward direction).
        ///
        /// When `true`, the GPU kernel reads timesteps from `seq_len-1` down
        /// to 0 and writes output in reverse order. Eliminates 2 external
        /// `flip(dim=0)` dispatches per backward LSTM layer (#1815).
        #[serde(default)]
        reverse: bool,
    },
    /// GPU-native cumulative sum (Blelloch parallel prefix scan).
    ///
    /// Delegates to `gpu_cumsum()` — single-pass for axis <= 256,
    /// three-pass for axis <= 65536. No weights.
    ///
    /// Input 0: tensor of any rank (F32).
    Cumsum {
        /// Axis along which to compute the prefix sum.
        dim: usize,
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
    },
    /// Fused InstanceNorm using a single Metal dispatch with threadgroup
    /// parallel reduction (#2472).
    ///
    /// Replaces the 7-dispatch IR decomposition (mean → sub → sqr → mean →
    /// add_eps → rsqrt → mul) with a single compute kernel.
    ///
    /// Input 0: `[B, C, *spatial]` (F32). No learnable affine parameters.
    /// Delegates to `gpu_instance_norm_fused()`.
    InstanceNorm {
        /// Epsilon for numerical stability.
        eps: f32,
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
    },
    /// GPU LayerNorm via decomposed dispatch path.
    ///
    /// Routes through `gpu_layer_norm()` which decomposes into standard
    /// GPU ops (mean → sub → sqr → mean → add_eps → rsqrt → mul → weight → bias).
    ///
    /// Input 0: any rank >= 2 (F32). Normalizes over the last dimension.
    /// Weights: `weight` `[hidden_dim]`, `bias` `[hidden_dim]`.
    LayerNorm {
        /// Epsilon for numerical stability.
        eps: f32,
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
        /// Hidden dimension (last dim of input). Normalization axis.
        hidden_dim: usize,
    },
    /// Fused residual-add + LayerNorm in a single Metal dispatch.
    ///
    /// Computes `LayerNorm(a + b, weight, bias, eps)` without materializing
    /// the intermediate `a + b` tensor. Saves 1 dispatch per fusion site.
    /// Part of #1815 Tier 5 D2.
    ///
    /// Input 0: residual `[*, hidden_dim]` (a).
    /// Input 1: new value `[*, hidden_dim]` (b).
    /// Weights: `weight` `[hidden_dim]`, `bias` `[hidden_dim]`.
    AddLayerNorm {
        /// Epsilon for LayerNorm numerical stability.
        eps: f32,
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
        /// Hidden dimension (last dim). Normalization axis.
        hidden_dim: usize,
    },
    /// Fused AdaIN+Snake: InstanceNorm → affine(gamma, beta) → Snake(alpha)
    /// in a single Metal dispatch (#2472).
    ///
    /// Replaces ~20 dispatches per AdaIN+Snake call. Used in Kokoro Generator
    /// ResBlocks (36 invocations per forward).
    ///
    /// Input 0: `[B, C, *spatial]` (x). Input 1: `[B, C, 1]` (gamma).
    /// Input 2: `[B, C, 1]` (beta). Weight: `alpha` `[C]`.
    AdainSnake {
        /// Epsilon for InstanceNorm numerical stability.
        eps: f32,
        /// Input x shape `[B, C, *spatial]`.
        input_shape: Vec<usize>,
        /// Number of channels (for alpha indexing).
        channels: usize,
        /// Whether to use Kokoro residual gamma convention `(1+gamma)*normed+beta`
        /// (true) or standard AdaIN `gamma*normed+beta` (false). Part of #3251.
        residual_gamma: bool,
        /// Graph `NodeId`s of the 3 external inputs `[x, gamma, beta]`.
        ///
        /// Set at creation time so the edge_map builder can resolve
        /// edges generically without per-NativeOp patches. Part of #3261.
        #[serde(default)]
        external_node_ids: Option<Vec<u64>>,
    },
    /// Fused AdaIN+LeakyRelu: InstanceNorm → affine(gamma, beta) → LeakyRelu(slope)
    /// in a single Metal dispatch (#2472).
    ///
    /// Replaces ~20 dispatches per AdaIN+LeakyRelu call. Used in Kokoro
    /// F0EnergyPredictor AdainResBlk1d blocks (12 invocations per forward).
    ///
    /// Input 0: `[B, C, *spatial]` (x). Input 1: `[B, C, 1]` (gamma).
    /// Input 2: `[B, C, 1]` (beta). No per-channel weights.
    AdainLeakyRelu {
        /// Epsilon for InstanceNorm numerical stability.
        eps: f32,
        /// LeakyRelu negative slope.
        slope: f32,
        /// Input x shape `[B, C, *spatial]`.
        input_shape: Vec<usize>,
        /// Graph `NodeId`s of the 3 external inputs `[x, gamma, beta]`.
        ///
        /// Set at creation time so the edge_map builder can resolve
        /// edges generically without per-NativeOp patches. Part of #3261.
        #[serde(default)]
        external_node_ids: Option<Vec<u64>>,
    },
    /// Fused AdaLayerNorm: LayerNorm(x, w, b) → `(1+gamma) * normed + beta`
    /// in a single Metal dispatch (#2482).
    ///
    /// Replaces ~6-7 dispatches per AdaLayerNorm call. Used in Kokoro
    /// ProsodyPredictor ProsodyBlock (2 blocks × 1 call = 2 per forward).
    ///
    /// Input 0: `[B, T, C]` (x). Input 1: `[B, 1, C]` (gamma).
    /// Input 2: `[B, 1, C]` (beta).
    /// Weights: `norm_weight` `[C]`, `norm_bias` `[C]`.
    AdaLayerNorm {
        /// Epsilon for LayerNorm numerical stability.
        eps: f32,
        /// Input x shape `[B, T, C]`.
        input_shape: Vec<usize>,
        /// Hidden dimension (last dim of x, = C). Normalization axis.
        hidden_dim: usize,
    },
    /// Fused Flash Attention: `softmax(Q @ K^T * scale) @ V` in a single
    /// Metal dispatch using online softmax (Tri Dao, arXiv:2307.08691).
    /// Part of #2434.
    ///
    /// Avoids materializing the O(S_q × S_kv) attention matrix. Supports
    /// GQA (H_q must be a multiple of H_kv) and optional causal masking.
    ///
    /// Input 0: Q `[B, H_q, S_q, D]`.
    /// Input 1: K `[B, H_kv, S_kv, D]`.
    /// Input 2: V `[B, H_kv, S_kv, D]`.
    /// No weights.
    FlashAttention {
        /// Scaling factor (typically `1/sqrt(D)`).
        scale: f32,
        /// Whether to apply causal masking (k_col > q_row → masked).
        causal: bool,
        /// Q shape `[B, H_q, S_q, D]`.
        q_shape: Vec<usize>,
        /// K shape `[B, H_kv, S_kv, D]`.
        k_shape: Vec<usize>,
        /// Output shape `[B, H_q, S_q, D]` (same as Q shape).
        output_shape: Vec<usize>,
        /// Memory layout of Q/K/V/O tensors.
        #[serde(default)]
        input_layout: AttentionLayout,
    },
    /// 1-D max pooling: delegates to `DynTensor::max_pool1d()`.
    ///
    /// Input 0: `[batch, channels, length]` (F32).
    /// Output: `[batch, channels, out_length]`.
    /// No weights. Part of #2295 (PyanNet import).
    MaxPool1d {
        /// Pooling window size.
        kernel_size: usize,
        /// Stride between pooling windows.
        stride: usize,
        /// Zero-padding on each side of the input.
        padding: usize,
        /// Input (and source for output shape computation) tensor shape.
        input_shape: Vec<usize>,
    },
    /// A pre-computed constant tensor (e.g., from `arange`).
    ///
    /// The executor materializes the weight data (`weight_data["arange_data"]`)
    /// as a GPU buffer. No computation needed — the output is fully determined
    /// at compile time.
    ConstantWeight {
        /// Human-readable name for diagnostics.
        name: String,
        /// Output shape of the constant tensor.
        shape: Vec<usize>,
    },
    /// Full AdainResBlock: 2× (InstanceNorm + affine + activation + Conv1d) +
    /// residual add + optional scale. Part of #2218.
    ///
    /// Peephole-fused from graph topology (not just consecutive steps):
    ///   NormActivConv1d(x, γ1, β1) → NormActivConv1d(h, γ2, β2) → add(x, h) [→ mul(scale)]
    ///
    /// Handles both consecutive step layouts (Generator ResBlocks) and
    /// non-consecutive layouts (F0 ResBlocks with intervening style projections).
    ///
    /// The executor sequences existing fused kernels (2× NormActivConv1d + add)
    /// with pre-resolved buffers, eliminating per-step overhead.
    ///
    /// Graph inputs via `input_steps`: x, γ1, β1, γ2, β2.
    /// Weights: `p1_conv_weight`, `p1_conv_bias`, `p1_alpha` (Snake),
    ///          `p2_conv_weight`, `p2_conv_bias`, `p2_alpha` (Snake).
    FusedResBlock {
        /// First NormActivConv1d (norm + activation + dilated conv).
        phase1: NormActivConv1dParams,
        /// Second NormActivConv1d (norm + activation + stride-1 conv).
        phase2: NormActivConv1dParams,
        /// Step indices for direct buffer access by the executor:
        /// `[x_step, γ1_step, β1_step, γ2_step, β2_step]`.
        ///
        /// Encoded at peephole time from graph topology. The executor uses
        /// these to resolve input buffers directly instead of edge_map.
        input_steps: Vec<usize>,
        /// Post-add residual scale factor. 1.0 for Generator (no-op),
        /// `1/√2` for F0EnergyPredictor. Absorbed from the post-add
        /// `ConstantValue + Dispatch "mul"` pattern at peephole time.
        residual_scale: f32,
        /// Absorbed style projection (peephole pass 3, #2780).
        ///
        /// When `Some`, the executor takes `style_embed` as additional input
        /// and runs linear projections to produce gamma/beta pairs. Input steps
        /// become `[x, style_embed]` instead of `[x, γ1, β1, γ2, β2]`.
        /// When `None`, `input_steps` is `[x, γ1, β1, γ2, β2]` and
        /// gamma/beta come from pre-computed buffers. Part of #2780.
        style_proj: Option<StyleProjectionParams>,
        /// Optional conv1x1 shortcut step for blocks where `dim_in != dim_out`.
        ///
        /// When `Some(step_idx)`, the executor uses `buffers[step_idx]` as the
        /// residual for the add instead of `buffers[input_steps[0]]`. The
        /// conv1x1 step executes normally before FusedResBlock; this field
        /// tells the executor which buffer holds the shortcut output.
        ///
        /// `None` means identity shortcut (`x + h(x)`).
        #[serde(default)]
        shortcut_step: Option<usize>,
        /// Optional pool (ConvTranspose1d) step between phase1's norm+activation
        /// and phase1's Conv1d, for upsample ResBlocks (#3510).
        ///
        /// When `Some(step_idx)`, the executor splits phase1:
        /// 1. Run norm+activation on x → activated
        /// 2. Read pool output from `buffers[pool_step]` (pool already executed)
        /// 3. Run Conv1d on pool output → phase1_output
        ///    When `None`, phase1 is a single fused norm+activation+conv dispatch.
        ///
        /// The AdainLeakyRelu/AdainSnake and ConvTranspose1d steps remain in
        /// the compiled plan (NOT absorbed). Only conv1, adain2/conv2, and
        /// add/mul steps are replaced with IdentityPassthrough.
        #[serde(default)]
        pool_step: Option<usize>,
        /// Batched style projection (peephole pass 4, #1815 Tier 1).
        ///
        /// When `Some`, the executor narrows gamma/beta from a pre-computed
        /// batched projection buffer (`input_steps[1]` → `BatchedStyleProjection`
        /// output) instead of running per-block Linear projections.
        /// Mutually exclusive with `style_proj`.
        #[serde(default)]
        style_batch_offset: Option<StyleBatchOffset>,
    },
    /// Batched style projection: one matmul for all FusedResBlocks in a segment.
    ///
    /// Concatenates per-block style projection weights along dim 0 and runs a
    /// single `[B, style_dim] × [style_dim, total_out]` matmul + bias_add.
    /// Each FusedResBlock then narrows its gamma/beta from the output (zero-copy).
    ///
    /// Saves ~136 Metal dispatches for Kokoro (35 blocks × 4 style dispatches → 2).
    /// Part of #1815 Tier 1.
    ///
    /// Weights: `"weight"` `[total_out, style_dim]`, `"bias"` `[total_out]`.
    BatchedStyleProjection {
        /// Per-block narrow offsets into the concatenated output.
        blocks: Vec<StyleBatchOffset>,
        /// Style embedding dimension (128 for Kokoro).
        style_dim: usize,
        /// Total output dimension (sum of 2*(C1+C2) across all blocks).
        total_out: usize,
        /// Step index of the style embedding input.
        style_step: usize,
    },
    /// Fused InstanceNorm + style affine + activation + Conv1d in a single
    /// Metal dispatch. Replaces 2 separate steps (AdainLeakyRelu/AdainSnake
    /// NativeOp + Conv1d Dispatch). Part of #2780.
    ///
    /// Peephole-fused from adjacent `AdainLeakyRelu(x, gamma, beta)` →
    /// `Conv1d(result, weight, bias)` patterns in the compiled plan.
    ///
    /// Input 0: `[B, C_in, T]` (x). Input 1: `[B, C_in, 1]` (gamma).
    /// Input 2: `[B, C_in, 1]` (beta).
    /// Weights: `conv_weight` `[C_out, C_in, K]`, `conv_bias` `[C_out]`.
    /// Optional weight: `alpha` `[C_in]` (for Snake activation only).
    NormActivConv1d {
        /// Which activation to apply after InstanceNorm + style affine.
        activation: NormActivation,
        /// Epsilon for InstanceNorm numerical stability.
        eps: f32,
        /// Conv1d dilation factor.
        conv_dilation: usize,
        /// Conv1d padding (symmetric).
        conv_padding: usize,
        /// Input x shape `[B, C_in, T]`.
        input_shape: Vec<usize>,
        /// Number of output channels (Conv1d weight shape\[0\]).
        output_channels: usize,
        /// Convolution kernel size (Conv1d weight shape\[2\]).
        kernel_size: usize,
        /// Graph `NodeId`s of the 3 external inputs `[x, gamma, beta]`.
        ///
        /// Set at peephole creation time so the edge_map builder can resolve
        /// edges generically without per-NativeOp patches. Part of #3261.
        #[serde(default)]
        external_node_ids: Option<Vec<u64>>,
    },
    /// Fused Linear + Activation: applies activation in the GEMM epilogue.
    ///
    /// Reduces `Linear → Activation` from 2 dispatches to 1 by applying the
    /// activation function inside the matmul write-back, avoiding a full
    /// buffer round-trip through global memory.
    ///
    /// Peephole-fused from `Dispatch{linear}` → `Dispatch{activation}`.
    /// Part of #2256.
    ///
    /// Input 0: `[..batch, in_features]`. Weights: `"weight"` `[out_features, in_features]`,
    /// optional `"bias"` `[out_features]`.
    LinearActivation {
        /// Which activation to apply after matmul+bias.
        activation: GemmActivation,
        /// Number of input features (last dim of input, weight dim 1).
        in_features: usize,
        /// Number of output features (weight dim 0, bias length).
        out_features: usize,
        /// Whether the linear has a bias term.
        has_bias: bool,
        /// Input tensor shape (all dimensions).
        input_shape: Vec<usize>,
    },
    /// Batched linear projection: N parallel Linear dispatches fused into
    /// a single matmul with concatenated weights.
    ///
    /// Used for Q/K/V attention projections sharing the same input tensor.
    /// Saves N-1 matmul dispatches per attention block. The executor
    /// performs the full matmul, narrows the first projection as the step
    /// output, and stashes the full `[..batch, total_out]` intermediate in
    /// a thread-local temp for `ProjectionSlice` steps to read. Part of #3269.
    ///
    /// Input 0: `[..batch, in_features]` (shared hidden state).
    /// Weights: `"weight_t"` `[in_features, total_out]`, optional `"bias"` `[total_out]`.
    BatchedLinearProjection {
        /// Number of input features (shared across all projections).
        in_features: usize,
        /// Total output features (sum of all projection out_features).
        total_out_features: usize,
        /// Per-projection output sizes, in concatenation order.
        projection_sizes: Vec<usize>,
        /// Whether the projection has a bias term.
        has_bias: bool,
        /// Input tensor shape (all dimensions).
        input_shape: Vec<usize>,
    },
    /// GPU narrow dispatch from a batched projection output.
    ///
    /// Reads the stashed full `[..batch, total_out]` tensor from
    /// `BatchedLinearProjection` and narrows on the specified dimension
    /// to extract one projection's slice. Unlike `NarrowView` (zero-copy
    /// byte-offset alias), this dispatches a GPU narrow because last-dim
    /// narrow on multi-dimensional tensors produces non-contiguous data.
    /// Part of #3269.
    ///
    /// No weights. No graph-level inputs (reads from thread-local temp).
    ProjectionSlice {
        /// Step index of the `BatchedLinearProjection` that produced the
        /// full output (used as key into the projection temp map).
        source_step: usize,
        /// Dimension to narrow on (typically the last dimension).
        dim: usize,
        /// Start index within the narrow dimension.
        start: usize,
        /// Number of elements to take from the narrow dimension.
        length: usize,
        /// Output shape after narrowing.
        output_shape: Vec<usize>,
    },
    /// Fused Norm + Linear: normalizes the last dimension, then applies
    /// a dense linear projection in a single Metal dispatch.
    ///
    /// Avoids materializing the intermediate normalized tensor. Uses threadgroup
    /// memory to hold normalized values between the norm reduction and GEMM
    /// phases. Part of #3089.
    ///
    /// Supports both LayerNorm (`(x-mean)/std * w + b`) and RmsNorm
    /// (`x * rsqrt(mean(x²)+eps) * w`) via the `norm_kind` discriminator.
    ///
    /// Peephole-fused from `NativeOp{LayerNorm}` or `Dispatch{rms_norm}`
    /// followed by `Dispatch{linear}`.
    ///
    /// Input 0: `[..batch, hidden_dim]`.
    /// Weights: `"norm_weight"` `[hidden_dim]`, `"norm_bias"` `[hidden_dim]`
    ///          (LayerNorm only), `"weight"` `[out_features, hidden_dim]`,
    ///          optional `"bias"` `[out_features]`.
    NormLinear {
        /// Which normalization to apply before GEMM.
        norm_kind: FusedNormKind,
        /// Epsilon for numerical stability.
        eps: f32,
        /// Input tensor shape (all dimensions).
        input_shape: Vec<usize>,
        /// Hidden dimension (last dim of input). Normalization axis size.
        hidden_dim: usize,
        /// Number of output features (linear weight dim 0).
        out_features: usize,
        /// Whether the linear has a bias term.
        has_bias: bool,
    },
    /// Channels-first LayerNorm: normalizes over dim 1 (channel dimension)
    /// of a `[B, C, T]` tensor.
    ///
    /// Semantically equivalent to `Transpose(1,2) → LayerNorm → Transpose(1,2)`
    /// but eliminates two data-copy transpose dispatches. The kernel reduces
    /// over the C elements at stride T for each `(b, t)` position.
    ///
    /// Input 0: `[B, C, T]` (F32).
    /// Weights: `weight` `[C]`, `bias` `[C]`.
    /// Part of #3457.
    ChannelsFirstLayerNorm {
        /// Epsilon for LayerNorm numerical stability.
        eps: f32,
        /// Input (and output) tensor shape `[B, C, T]`.
        input_shape: Vec<usize>,
        /// Channel dimension size (dim 1 of input). Normalization axis.
        channels: usize,
        /// Optional post-norm LeakyReLU slope. When set, the kernel fuses
        /// `LayerNorm → LeakyRelu(slope)` in a single dispatch.
        leaky_relu_slope: Option<f32>,
    },
    /// INT8 W8A16 quantized matmul: INT8 weights, F32 activations, F32 output.
    ///
    /// Dequantizes INT8 weights on-the-fly inside the tiled GEMM kernel.
    /// Per-channel scale and zero_point are applied during the tile load phase.
    /// Provides ~4x memory reduction vs F32 weights with minimal accuracy loss.
    ///
    /// Input 0: `[..batch, in_features]` (F32 activations).
    /// Weights: `"weight_int8"` `[out_features, in_features]` (U8, storing i8 values),
    ///          `"scale"` `[out_features]` (F32 per-channel scale),
    ///          `"zero_point"` `[out_features]` (I32 per-channel zero point).
    /// Optional: `"bias"` `[out_features]` (F32).
    ///
    /// Part of #3522.
    Int8Gemm {
        /// Number of input features (last dim of input, weight dim 1).
        in_features: usize,
        /// Number of output features (weight dim 0).
        out_features: usize,
        /// Whether the linear has a bias term.
        has_bias: bool,
        /// Input tensor shape (all dimensions).
        input_shape: Vec<usize>,
    },
    /// Conv1d via im2col + simdgroup GEMM for large channel counts.
    ///
    /// Replaces the naive per-element Conv1d MSL kernel with a two-phase
    /// approach: im2col unfold → simdgroup matrix multiply. Provides ~2-4x
    /// throughput for Kokoro's dominant shapes (C=128-512, K=3-7).
    ///
    /// Routing condition: `c_out * (c_in * k_size) * l_out >= 2_000_000`
    /// (same threshold as `MetalDynBackend::should_use_conv1d_gemm`).
    ///
    /// Input 0: `[B, C_in, L_in]`.
    /// Weights: `"weight"` `[C_out, C_in, K]`, optional `"bias"` `[C_out]`.
    /// Part of #3390.
    Conv1dGemm {
        /// Input tensor shape `[B, C_in, L_in]`.
        input_shape: Vec<usize>,
        /// Number of output channels.
        out_channels: usize,
        /// Convolution kernel size.
        kernel_size: usize,
        /// Convolution stride.
        stride: usize,
        /// Zero-padding on each side.
        padding: usize,
        /// Dilation factor.
        dilation: usize,
        /// Number of channel groups (must be 1 for GEMM path).
        groups: usize,
        /// Whether the conv has a bias term.
        has_bias: bool,
    },
    /// Fused SiLU-Mul: `silu(gate) * up` in a single Metal dispatch.
    ///
    /// Replaces the 2-dispatch `Silu → Mul` pattern from SwiGLU MLP blocks
    /// (Qwen3, GLM5, etc.). Purely elementwise on two same-shape inputs,
    /// no weights.
    ///
    /// Input 0: `gate` (any shape, flattened).
    /// Input 1: `up` (same shape as gate).
    /// Output: `silu(gate) * up` (same shape).
    /// No weights.
    /// Part of #3521.
    SiluMul {
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
    },
    /// Rotary Position Embedding (RoPE): applies position-dependent rotation
    /// to Q/K tensors in transformer attention layers.
    ///
    /// Delegates to the fused GPU RoPE kernel (`MetalDynBackend::gpu_rope`)
    /// which applies the rotation in a single dispatch graph:
    /// ```text
    /// y[..., 2i]   = x[..., 2i] * cos[..., i] - x[..., 2i+1] * sin[..., i]
    /// y[..., 2i+1] = x[..., 2i] * sin[..., i] + x[..., 2i+1] * cos[..., i]
    /// ```
    ///
    /// Input 0: `[..., S, D]` where D = head_dim (must be even).
    /// Weights: `"cos_cache"` `[S, D/2]`, `"sin_cache"` `[S, D/2]`.
    /// Output: same shape as input.
    ///
    /// Part of #3526.
    RotaryEmbedding {
        /// Attention head dimension (last dim of input, must be even).
        head_dim: usize,
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
    },
    /// Fused residual-Add + LayerNorm + Linear in a single Metal dispatch.
    ///
    /// Combines AddLayerNorm and NormLinear: computes `Linear(LayerNorm(a + b, w, b))`
    /// without materializing the intermediate sum or normalized tensor. Uses
    /// threadgroup memory to hold normalized values between the reduction phase
    /// and GEMM phase.
    ///
    /// In PlBert transformer layers, the post-attention and post-FFN residual
    /// connections produce `Add + LayerNorm` (fused to AddLayerNorm by pass 6)
    /// followed immediately by a `Linear` projection. This variant fuses all three
    /// into a single NativeOp, saving 1 dispatch per site (2 per transformer layer).
    ///
    /// When `should_use_simdgroup(flat_rows, hidden_dim, out_features)` is true,
    /// splits into two Metal dispatches (add-norm → simdgroup GEMM).
    ///
    /// Input 0: residual `[*, hidden_dim]` (a).
    /// Input 1: new value `[*, hidden_dim]` (b).
    /// Weights: `"norm_weight"` `[hidden_dim]`, `"norm_bias"` `[hidden_dim]`,
    ///          `"weight"` `[out_features, hidden_dim]`, optional `"bias"` `[out_features]`.
    ///
    /// Part of #3351 T2.1 (dispatch reduction).
    AddNormLinear {
        /// Epsilon for LayerNorm numerical stability.
        eps: f32,
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
        /// Hidden dimension (last dim of input). Normalization axis.
        hidden_dim: usize,
        /// Number of output features (linear weight dim 0).
        out_features: usize,
        /// Whether the linear has a bias term.
        has_bias: bool,
    },
    /// MoE top-k gating: softmax over experts + top-k selection + renormalization.
    ///
    /// CPU dispatch: computes routing weights and expert indices, then runs
    /// each selected expert's FFN on the gathered tokens and scatter-adds
    /// weighted results back. The gating itself (softmax + topk) decomposes
    /// to existing DynTensor ops; this NativeOp marks the composite for
    /// dispatch counting and buffer planning.
    ///
    /// Input 0: `[..., model_dim]` hidden states.
    /// Output: `[..., model_dim]` (same shape, routed through experts).
    /// No weights (gate weights are in the Linear sub-op).
    ///
    /// Part of #3542.
    MoeGating {
        /// Total number of experts.
        num_experts: usize,
        /// Number of experts selected per token.
        top_k: usize,
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
    },
    /// Fused AdaIN (instance_norm + affine) + Snake activation in a single dispatch.
    ///
    /// Equivalent to: `snake(instance_norm(x) * gamma + beta, alpha)`
    /// where `snake(x, alpha) = x + (1/alpha) * sin(alpha * x)^2`.
    ///
    /// Detected by peephole from the trace-level pattern:
    /// `InstanceNorm(x)` → `Mul(gamma)` → `Add(beta)` → `Snake(alpha)`.
    /// Replaces 3 separate dispatches (instance_norm, scale+shift, snake)
    /// with a single Metal compute kernel.
    ///
    /// Used in Kokoro Generator blocks that aren't captured by the deeper
    /// FusedResBlock or NormActivConv1d fusion passes. 12 blocks in the
    /// generator segment = 36 dispatches reduced to 12. Part of #4252.
    ///
    /// Input 0: `[B, C, T]` (x).
    /// Input 1: `[B, C, 1]` (gamma from style projection).
    /// Input 2: `[B, C, 1]` (beta from style projection).
    /// Weight: `alpha` `[C]`.
    FusedAdainSnake {
        /// Epsilon for InstanceNorm numerical stability.
        eps: f32,
        /// Input x shape `[B, C, T]`.
        input_shape: Vec<usize>,
        /// Number of channels (C dimension).
        channels: usize,
        /// Graph node IDs for external edge resolution: [x, gamma, beta].
        external_node_ids: Option<Vec<u64>>,
    },
    /// Fused upsample1d (nearest-neighbor) + conv1d in a single Metal dispatch.
    ///
    /// The f0_energy Kokoro segment has 6 pairs of upsample1d followed by
    /// conv1d. Fusing them saves 6 dispatches by reducing two plan steps
    /// into one.
    ///
    /// A single MSL kernel reads `[B, C_in, T]`, computes nearest-neighbor
    /// upsample inline during Conv1d accumulation (no intermediate buffer),
    /// and writes `[B, C_out, T_out]` directly. F16 inputs accumulate in
    /// F32 for precision.
    ///
    /// Part of #4310.
    ///
    /// Input 0: `[B, C_in, T]` (x).
    /// Weights: `weight` `[C_out, C_in, K]`, `bias` `[C_out]`.
    FusedUpsampleConv1d {
        /// Nearest-neighbor upsample factor along time dimension.
        upsample_factor: usize,
        /// Number of input channels.
        in_channels: usize,
        /// Number of output channels.
        out_channels: usize,
        /// Convolution kernel size.
        kernel_size: usize,
        /// Convolution stride.
        stride: usize,
        /// Convolution padding.
        padding: usize,
        /// Input tensor shape `[B, C_in, T]` (before upsampling).
        input_shape: Vec<usize>,
    },
    /// Fused bidirectional LSTM + concatenation.
    ///
    /// Runs forward and reverse LSTMs and concatenates their outputs along
    /// the hidden dimension. Replaces 2 separate LSTM dispatches + a cat.
    ///
    /// Part of #4252.
    BiLstmCat {
        /// LSTM hidden size per direction.
        hidden_size: usize,
        /// Input tensor shape `[seq_len, batch, input_size]`.
        input_shape: Vec<usize>,
        /// Hidden/cell state shape `[batch, hidden_size]`.
        h_shape: Vec<usize>,
        /// Index of the forward LSTM step (for weight access).
        fwd_lstm_step: usize,
        /// Index of the reverse LSTM step (for weight access).
        rev_lstm_step: usize,
    },
    /// Fused multiply-add (FMA): `a * b + c` in a single Metal dispatch.
    ///
    /// Replaces the 2-dispatch `Mul -> Add` pattern. Maps to hardware FMA.
    FusedMulAdd {
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
    },
    /// Fused SiGLU: `sigmoid(x) * x` (SiLU/Swish) in a single Metal dispatch.
    ///
    /// Replaces the 2-dispatch `Sigmoid -> Mul` pattern on a single input.
    FusedSiGLU {
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
    },
    /// Fused GeGLU: `gelu(gate) * up` in a single Metal dispatch.
    ///
    /// Replaces the 2-dispatch `GELU -> Mul` pattern. Used in Qwen3/GLM5.
    FusedGeGLU {
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
    },
    /// Fused LayerNorm + Linear: normalizes the last dimension, then applies
    /// a dense linear projection. Semantically identical to `NormLinear` with
    /// `norm_kind = LayerNorm`, but detected from a different peephole pattern
    /// (standalone LayerNorm NativeOp followed by a Dispatch{linear}).
    ///
    /// Delegates to the same executor as `NormLinear` with
    /// `FusedNormKind::LayerNorm`. Saves ~12 dispatches in PlBert encoder
    /// segments by fusing LayerNorm + Linear attention projections.
    ///
    /// Input 0: `[..batch, hidden_dim]`.
    /// Weights: `"norm_weight"` `[hidden_dim]`, `"norm_bias"` `[hidden_dim]`,
    ///          `"weight"` `[out_features, hidden_dim]`,
    ///          optional `"bias"` `[out_features]`.
    ///
    /// Part of #4252.
    FusedLayerNormLinear {
        /// Epsilon for LayerNorm numerical stability.
        eps: f32,
        /// Input tensor shape (all dimensions).
        input_shape: Vec<usize>,
        /// Hidden dimension (last dim of input). Normalization axis size.
        hidden_dim: usize,
        /// Number of output features (linear weight dim 0).
        out_features: usize,
        /// Whether the linear has a bias term.
        has_bias: bool,
    },
    /// Fused BatchNorm2d inference: `(x - mean) / sqrt(var + eps) * weight + bias`
    /// in a single Metal dispatch.
    ///
    /// Uses precomputed running statistics -- no reduction needed, purely
    /// per-element with per-channel parameters. Replaces ~6 separate GPU
    /// dispatches (reshape + broadcast_sub + add_scalar + sqrt + recip +
    /// broadcast_mul + broadcast_add).
    ///
    /// Used by ResNet, Table Transformer, YOLO, and any CNN model with
    /// BatchNorm2d layers. Part of #4324.
    ///
    /// Input 0: `[N, C, *spatial]` (rank >= 2).
    /// Weights: `running_mean` `[C]`, `running_var` `[C]`,
    ///          optional `weight` `[C]`, optional `bias` `[C]`.
    BatchNorm2d {
        /// Epsilon for numerical stability.
        eps: f32,
        /// Number of channels (C dimension).
        num_channels: usize,
        /// Input (and output) tensor shape.
        input_shape: Vec<usize>,
        /// Whether the layer has a learnable weight (gamma).
        has_weight: bool,
        /// Whether the layer has a learnable bias (beta).
        has_bias: bool,
    },
    /// Fused InstanceNorm + Mul + Add: `instance_norm(x) * gamma + beta`
    /// in a single Metal dispatch.
    ///
    /// Detected by peephole from the trace-level pattern:
    /// `InstanceNorm(x)` -> `Mul(gamma)` -> `Add(beta)`.
    /// Replaces 3 separate dispatches (instance_norm, mul, add) with 1.
    ///
    /// This pattern appears in every Kokoro AdaIN block. 24 blocks in the
    /// generator segment = 72 dispatches reduced to 24. Part of #4252.
    ///
    /// Input 0: `[B, C, T]` (x).
    /// Input 1: `[B, C, 1]` (gamma from style projection).
    /// Input 2: `[B, C, 1]` (beta from style projection).
    /// No static weights.
    FusedInstanceNormMulAdd {
        /// Epsilon for InstanceNorm numerical stability.
        eps: f32,
        /// Input x shape `[B, C, T]`.
        input_shape: Vec<usize>,
        /// Number of channels (C dimension).
        channels: usize,
        /// Graph node IDs for external edge resolution: [x, gamma, beta].
        external_node_ids: Option<Vec<u64>>,
    },
    /// Fused Snake activation + InstanceNorm in a single Metal dispatch.
    ///
    /// Combines per-element Snake activation `x + (1/alpha) * sin(alpha*x)^2`
    /// with per-channel InstanceNorm `(y - mean(y)) / sqrt(var(y) + eps)` into
    /// a single compute kernel. The Snake output is computed on-the-fly and
    /// fed directly into the Welford reduction for mean/variance, avoiding a
    /// full buffer round-trip through global memory.
    ///
    /// Detected by peephole from the trace-level pattern:
    /// `Dispatch{snake_tensor}(x, alpha)` → `NativeOp{InstanceNorm}(result, eps)`.
    /// Replaces 2 dispatches (snake + instance_norm) with 1.
    ///
    /// The Kokoro generator has `snake → instance_norm` sequences in blocks
    /// where AdaIN affine is applied separately or where the pattern falls
    /// outside deeper FusedResBlock/NormActivConv1d captures. Part of #4264.
    ///
    /// Input 0: `[B, C, T]` (x).
    /// Weight: `alpha` `[C]` (per-channel Snake parameter).
    FusedSnakeInstanceNorm {
        /// Epsilon for InstanceNorm numerical stability.
        eps: f32,
        /// Input x shape `[B, C, T]`.
        input_shape: Vec<usize>,
        /// Number of channels (C dimension).
        channels: usize,
    },
    /// Fused Conv1d + Activation in a single Metal dispatch.
    ///
    /// Replaces the 2-dispatch `Conv1d → Activation` pattern by applying the
    /// activation function inside the Conv1d kernel's write-back phase,
    /// avoiding a full buffer round-trip through global memory.
    ///
    /// Peephole-fused from `Dispatch{conv1d}` → `Dispatch{activation}`.
    /// The Kokoro generator has many Conv1d → Snake/LeakyReLU sequences;
    /// fusing them saves 4-8 dispatches per synthesis call.
    /// Part of #4264.
    ///
    /// Input 0: `[B, C_in, L_in]`.
    /// Weights: `"weight"` `[C_out, C_in, K]`, optional `"bias"` `[C_out]`.
    /// Optional weight: `"alpha"` `[C_out]` (for Snake activation only).
    FusedConv1dActivation {
        /// Which activation to apply (before or after conv1d, see `pre_activation`).
        activation: ConvActivation,
        /// Number of output channels.
        out_channels: usize,
        /// Convolution kernel size.
        kernel_size: usize,
        /// Convolution stride.
        stride: usize,
        /// Zero-padding on each side.
        padding: usize,
        /// Dilation factor.
        dilation: usize,
        /// Number of channel groups (must be 1 for this fusion path).
        groups: usize,
        /// Whether the conv has a bias term.
        has_bias: bool,
        /// Input tensor shape `[B, C_in, L_in]`.
        input_shape: Vec<usize>,
        /// When `true`, activation is applied BEFORE conv1d (Activation -> Conv1d).
        /// When `false` (default), activation is applied AFTER conv1d (Conv1d -> Activation).
        pre_activation: bool,
    },
    /// Fused Conv1d + Snake activation + InstanceNorm in a single logical
    /// dispatch.
    ///
    /// Replaces the 3-step pattern `Conv1d → Snake → InstanceNorm` by
    /// executing conv1d, applying Snake activation, then normalizing
    /// per-channel — all within one NativeOp. The Kokoro generator has
    /// repeating `conv1d → snake → instance_norm` chains in blocks that
    /// fall outside the deeper FusedResBlock / NormActivConv1d patterns.
    ///
    /// Saves 1-2 dispatches per site (eliminates intermediate buffer
    /// materialization between conv → snake → norm).
    /// Part of #4264.
    ///
    /// Input 0: `[B, C_in, L_in]`.
    /// Weights: `"conv_weight"` `[C_out, C_in, K]`, optional `"conv_bias"` `[C_out]`,
    ///          `"alpha"` `[C_out]` (Snake per-channel parameter).
    FusedConv1dSnakeNorm {
        /// Number of output channels (from conv1d weight shape[0]).
        out_channels: usize,
        /// Convolution kernel size.
        kernel_size: usize,
        /// Convolution stride.
        stride: usize,
        /// Zero-padding on each side.
        padding: usize,
        /// Dilation factor.
        dilation: usize,
        /// Number of channel groups.
        groups: usize,
        /// Whether the conv has a bias term.
        has_bias: bool,
        /// Epsilon for InstanceNorm numerical stability.
        eps: f32,
        /// Input tensor shape `[B, C_in, L_in]`.
        input_shape: Vec<usize>,
    },
    /// Fused 2x (Conv1d + Snake + InstanceNorm) + residual add in a single
    /// logical NativeOp.
    ///
    /// Captures the Kokoro Generator ResBlock pattern WITHOUT AdaIN style
    /// projection (no gamma/beta affine):
    /// ```text
    /// phase1: conv1d(x) -> snake(alpha1) -> instance_norm(eps)
    /// phase2: conv1d(phase1) -> snake(alpha2) -> instance_norm(eps)
    /// output: x + phase2  [* optional residual_scale]
    /// ```
    ///
    /// This is DIFFERENT from `FusedResBlock` which uses the AdaIN order:
    /// `instance_norm -> affine(gamma,beta) -> activation -> conv1d`.
    ///
    /// Detected by peephole from: 2x `FusedConv1dSnakeNorm` + `Dispatch{add}`.
    /// Reduces 3 plan steps to 1, and 7 original dispatches to 1 logical op
    /// (internally sequences conv+snake+norm per phase + add).
    /// Part of #4264.
    ///
    /// Input 0: `[B, C_in, L_in]` (x, also residual).
    /// Weights: `p1_conv_weight`, `p1_conv_bias`, `p1_alpha`,
    ///          `p2_conv_weight`, `p2_conv_bias`, `p2_alpha`.
    /// Fused Add + InstanceNorm + Conv1d(K=1) in a single logical NativeOp.
    ///
    /// Captures the Kokoro decoder pattern:
    /// ```text
    /// residual = x + h(x)        // residual add
    /// normed   = instance_norm(residual, eps)
    /// output   = conv1d(normed, weight_1x1, bias)  // 1x1 channel projection
    /// ```
    ///
    /// Replaces 3 separate dispatches (add + instance_norm + conv1d) with 1
    /// logical NativeOp. The executor sequences these as DynTensor ops and
    /// the lazy command buffer batches them.
    ///
    /// Used in Kokoro Generator decoder blocks where a residual connection
    /// feeds into instance normalization followed by a 1x1 convolution for
    /// channel dimension changes. Part of #4264.
    ///
    /// Input 0: `[B, C_in, T]` (x, one operand of the add).
    /// Input 1: `[B, C_in, T]` (h, other operand of the add).
    /// Weights: `"weight"` `[C_out, C_in, 1]`, optional `"bias"` `[C_out]`.
    FusedAddInstanceNormConv1x1 {
        /// Epsilon for InstanceNorm numerical stability.
        eps: f32,
        /// Input tensor shape `[B, C_in, T]`.
        input_shape: Vec<usize>,
        /// Number of input channels (C_in = input_shape[1]).
        in_channels: usize,
        /// Number of output channels (C_out, from conv1d weight shape[0]).
        out_channels: usize,
        /// Whether the conv1d has a bias term.
        has_bias: bool,
    },
    /// Fused ConvTranspose1d + Activation in a single logical dispatch.
    ///
    /// Captures `ConvTranspose1d → Activation` (LeakyReLU, Snake, ReLU, SiLU,
    /// GELU, Tanh) patterns in the Kokoro Generator upsample stages and
    /// F0EnergyPredictor upsampling blocks. These ConvTranspose1d steps are
    /// NOT captured by any existing peephole pass (FusedResBlock handles
    /// them via `pool_step` but the ConvTranspose1d + Activation pair
    /// remains as separate dispatches).
    ///
    /// Saves 1 dispatch per pair by merging into a single NativeOp that
    /// the lazy command buffer can batch.
    /// Part of #4264.
    ///
    /// Input 0: `[B, C_in, L_in]`.
    /// Weights: `"weight"` `[C_in, C_out, K]`, optional `"bias"` `[C_out]`.
    /// Optional weight: `"alpha"` `[C_out]` (for Snake activation only).
    FusedConvTranspose1dActivation {
        /// Which activation to apply after conv_transpose1d.
        activation: ConvActivation,
        /// Number of output channels (C_out from weight shape[1]).
        out_channels: usize,
        /// Convolution kernel size.
        kernel_size: usize,
        /// Convolution stride.
        stride: usize,
        /// Zero-padding on each side.
        padding: usize,
        /// Dilation factor.
        dilation: usize,
        /// Number of channel groups.
        groups: usize,
        /// Extra output padding (must be < stride).
        output_padding: usize,
        /// Whether the conv has a bias term.
        has_bias: bool,
        /// Input tensor shape `[B, C_in, L_in]`.
        input_shape: Vec<usize>,
    },
    FusedConv1dSnakeNormResBlock {
        /// Phase 1 output channels.
        phase1_out_channels: usize,
        /// Phase 1 conv kernel size.
        phase1_kernel_size: usize,
        /// Phase 1 conv padding.
        phase1_padding: usize,
        /// Phase 1 conv dilation.
        phase1_dilation: usize,
        /// Phase 1 has conv bias.
        phase1_has_bias: bool,
        /// Phase 2 output channels.
        phase2_out_channels: usize,
        /// Phase 2 conv kernel size.
        phase2_kernel_size: usize,
        /// Phase 2 conv padding.
        phase2_padding: usize,
        /// Phase 2 conv dilation.
        phase2_dilation: usize,
        /// Phase 2 has conv bias.
        phase2_has_bias: bool,
        /// Epsilon for InstanceNorm in both phases.
        eps: f32,
        /// Post-add residual scale factor. 1.0 for standard (no-op),
        /// `1/sqrt(2)` for some variants. Absorbed from post-add multiply.
        residual_scale: f32,
        /// Input tensor shape `[B, C_in, L_in]`.
        input_shape: Vec<usize>,
        /// Step index of the residual input (x).
        x_step: usize,
    },
    /// Fused InstanceNorm + style affine + activation + ConvTranspose1d in a
    /// single Metal dispatch. The transposed-conv dual of `NormActivConv1d`.
    ///
    /// Replaces 2 separate steps (AdainLeakyRelu/AdainSnake NativeOp +
    /// ConvTranspose1d Dispatch). Detected by peephole from adjacent
    /// `AdainLeakyRelu(x, gamma, beta)` → `ConvTranspose1d(result, weight, bias)`
    /// patterns in the compiled plan.
    ///
    /// In Kokoro, the Generator and F0EnergyPredictor upsample stages use
    /// AdainLeakyRelu/AdainSnake followed by ConvTranspose1d (stride>1) for
    /// upsampling. These fall outside the `NormActivConv1d` peephole (which
    /// only matches regular Conv1d). Part of #4264.
    ///
    /// Input 0: `[B, C_in, T]` (x). Input 1: `[B, C_in, 1]` (gamma).
    /// Input 2: `[B, C_in, 1]` (beta).
    /// Weights: `conv_weight` `[C_in, C_out/groups, K]`,
    ///          `conv_bias` `[C_out]` (optional).
    /// Optional weight: `alpha` `[C_in]` (for Snake activation only).
    NormActivConvTranspose1d {
        /// Which activation to apply after InstanceNorm + style affine.
        activation: NormActivation,
        /// Epsilon for InstanceNorm numerical stability.
        eps: f32,
        /// ConvTranspose1d kernel size.
        kernel_size: usize,
        /// ConvTranspose1d stride.
        stride: usize,
        /// ConvTranspose1d padding.
        padding: usize,
        /// ConvTranspose1d dilation.
        dilation: usize,
        /// ConvTranspose1d groups.
        groups: usize,
        /// ConvTranspose1d output padding (must be < stride).
        output_padding: usize,
        /// Number of output channels (ConvTranspose1d weight shape[1] * groups).
        output_channels: usize,
        /// Input x shape `[B, C_in, T]`.
        input_shape: Vec<usize>,
        /// Graph `NodeId`s of the 3 external inputs `[x, gamma, beta]`.
        #[serde(default)]
        external_node_ids: Option<Vec<u64>>,
    },
    /// Fused InstanceNorm + Conv1d in a single logical NativeOp.
    ///
    /// Captures the pattern: `InstanceNorm(x, eps)` → `Conv1d(normed, weight, bias)`.
    /// This is the simpler counterpart of NormActivConv1d: no style affine
    /// projection (gamma/beta), no activation between norm and conv. Appears in
    /// Kokoro generator/decoder blocks where normalization precedes a 1×1 or 3×3
    /// convolution for channel projection or feature extraction.
    ///
    /// Saves 1 dispatch per site by eliminating the intermediate buffer between
    /// InstanceNorm and Conv1d.
    /// Part of #4264.
    ///
    /// Input 0: `[B, C_in, T]` (x).
    /// Weights: `"conv_weight"` `[C_out, C_in/groups, K]`, optional `"conv_bias"` `[C_out]`.
    FusedInstanceNormConv1d {
        /// Epsilon for InstanceNorm numerical stability.
        eps: f32,
        /// Number of output channels (Conv1d weight shape[0]).
        out_channels: usize,
        /// Convolution kernel size.
        kernel_size: usize,
        /// Convolution stride.
        stride: usize,
        /// Zero-padding on each side.
        padding: usize,
        /// Dilation factor.
        dilation: usize,
        /// Number of channel groups.
        groups: usize,
        /// Whether the conv has a bias term.
        has_bias: bool,
        /// Input tensor shape `[B, C_in, T]`.
        input_shape: Vec<usize>,
    },
    /// Fused Conv1d + InstanceNorm in a single logical NativeOp.
    ///
    /// Captures the pattern: `Conv1d(x, weight, bias)` → `InstanceNorm(result, eps)`.
    /// This is the simplest conv→norm pattern, without any activation in between
    /// (unlike FusedConv1dSnakeNorm which includes Snake activation).
    ///
    /// Appears in Kokoro generator blocks where convolution is followed directly
    /// by normalization, particularly in initial/final projection layers.
    ///
    /// Saves 1 dispatch per site.
    /// Part of #4264.
    ///
    /// Input 0: `[B, C_in, L_in]`.
    /// Weights: `"conv_weight"` `[C_out, C_in/groups, K]`, optional `"conv_bias"` `[C_out]`.
    FusedConv1dInstanceNorm {
        /// Epsilon for InstanceNorm numerical stability.
        eps: f32,
        /// Number of output channels (Conv1d weight shape[0]).
        out_channels: usize,
        /// Convolution kernel size.
        kernel_size: usize,
        /// Convolution stride.
        stride: usize,
        /// Zero-padding on each side.
        padding: usize,
        /// Dilation factor.
        dilation: usize,
        /// Number of channel groups.
        groups: usize,
        /// Whether the conv has a bias term.
        has_bias: bool,
        /// Input tensor shape `[B, C_in, L_in]`.
        input_shape: Vec<usize>,
    },
    /// Fused Linear + LayerNorm in a single logical NativeOp.
    ///
    /// Captures the pattern: `Linear(x, weight, bias)` → `LayerNorm(result, w, b, eps)`.
    /// The reverse of NormLinear. In PlBert transformer layers, the attention
    /// output projection and FFN output projection feed into LayerNorm before
    /// residual addition. Fusing eliminates the intermediate buffer.
    ///
    /// Saves 1 dispatch per site. In PlBert with 3 layers, up to 6 fusion sites
    /// (2 per layer: post-attention and post-FFN).
    /// Part of #4264.
    ///
    /// Input 0: `[..batch, in_features]`.
    /// Weights: `"weight"` `[out_features, in_features]`, optional `"bias"` `[out_features]`,
    ///          `"norm_weight"` `[out_features]`, `"norm_bias"` `[out_features]`.
    FusedLinearLayerNorm {
        /// Number of input features (last dim of input).
        in_features: usize,
        /// Number of output features (linear weight dim 0, also norm dim).
        out_features: usize,
        /// Whether the linear has a bias term.
        has_bias: bool,
        /// Epsilon for LayerNorm numerical stability.
        eps: f32,
        /// Input tensor shape (all dimensions).
        input_shape: Vec<usize>,
    },
    /// Chain of 2-4 consecutive FusedResBlocks executed as a single NativeOp.
    ///
    /// The Kokoro generator has 24 FusedResBlocks (6 upsample stages x 3-4 blocks
    /// per stage). Each FusedResBlock is currently a separate NativeOp with its own
    /// dispatch overhead. By chaining N consecutive ResBlocks into one NativeOp,
    /// we eliminate N-1 inter-block dispatch transitions — the output of block i
    /// feeds directly into block i+1 without writing to and reading from device
    /// memory through the compiled plan's buffer machinery.
    ///
    /// Each block in the chain shares the same activation type (all Snake or all
    /// LeakyRelu) and uses the batched style projection path for gamma/beta.
    ///
    /// Estimated dispatch reduction: 24 FusedResBlock NativeOps (each 3 dispatches
    /// = 72 total) → 6-12 FusedResBlockChain NativeOps (each N*3 dispatches but
    /// with reduced inter-op overhead). The real savings come from:
    /// 1. Fewer NativeOp dispatch entries in the compiled plan
    /// 2. Intermediate tensors stay in GPU registers/L1 between blocks
    /// 3. Single weight-resolve phase for the entire chain
    ///
    /// Part of #4264.
    ///
    /// Input 0: `[B, C, T]` (x, input to first block in chain).
    /// Input 1: style embed or batched projection output (shared across all blocks).
    /// Weights: per-block `"block{i}_p{j}_conv_weight"`, `"block{i}_p{j}_conv_bias"`,
    ///          `"block{i}_p{j}_alpha"` (Snake only).
    FusedResBlockChain {
        /// Parameters for each block in the chain (2-4 blocks).
        /// Each entry contains phase1 and phase2 NormActivConv1dParams.
        blocks: Vec<ResBlockChainEntry>,
        /// Step indices for direct buffer access: `[x_step, style_step]`.
        /// `x_step` is the input to the first block.
        /// `style_step` is the batched style projection output (shared).
        input_steps: Vec<usize>,
        /// Batched style projection offsets for each block in the chain.
        /// Each block narrows its gamma/beta from the batched projection output.
        style_batch_offsets: Vec<StyleBatchOffset>,
        /// Optional shortcut step for the first block (conv1x1 when dim_in != dim_out).
        #[serde(default)]
        first_shortcut_step: Option<usize>,
    },
}

#[path = "trace_compile_native_ops_types.rs"]
mod native_ops_types;

pub use native_ops_types::{
    AttentionLayout, ConvActivation, FusedNormKind, GemmActivation, NormActivConv1dParams,
    NormActivation, ResBlockChainEntry, StyleBatchOffset, StyleProjectionParams,
};

#[path = "trace_compile_native_ops_dispatch_count.rs"]
mod dispatch_count;

#[cfg(kani)]
#[path = "kani_native_ops_dispatch_count.rs"]
mod kani_native_ops_dispatch_count;

#[cfg(kani)]
#[path = "kani_trace_compile_native_ops_dispatch_count.rs"]
mod kani_trace_compile_native_ops_dispatch_count;

#[cfg(kani)]
#[path = "kani_trace_compile_native_ops.rs"]
mod kani_trace_compile_native_ops;

#[cfg(kani)]
#[path = "kani_trace_compile_native_ops_advanced.rs"]
mod kani_trace_compile_native_ops_advanced;

#[cfg(kani)]
#[path = "kani_dispatch_count_3738.rs"]
mod kani_dispatch_count_3738;
