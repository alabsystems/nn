// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Type definitions for computation graph tracing: `TraceOp`, `WeightRef`,
//! `TraceNode`, and `NodeId`.

use crate::dyn_tensor::{CompareOp, GridSamplePaddingMode};
use crate::DType;

use super::{KokoroFusedOp, NodeId, TraceActivation, TraceUpsampleMode, WeightRef};

/// An operation recorded during tracing.
///
/// Maps to NY `Layer` variants. Weight data is flat f32 with shape.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[allow(clippy::large_enum_variant)] // FusedAdainResBlock carries 8 WeightRefs by design
pub enum TraceOp {
    /// Network input placeholder.
    Input,

    /// Constant weight tensor auto-registered during tracing.
    ///
    /// When a tensor without a trace ID is used in a binary op during active
    /// tracing, it is automatically registered as a `ConstantWeight` node.
    /// This happens for model weight tensors (e.g., Snake alpha parameters)
    /// that are created outside the trace scope but participate in traced ops.
    ConstantWeight {
        weight: WeightRef,
    },

    // -- Binary element-wise --
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,

    // -- Matrix multiply --
    /// Raw variable-variable matmul. Compilable but **not verifiable** via
    /// NY. Use `Linear` for verifiable weight-input matmul.
    MatMul,

    // -- Unary element-wise --
    Relu,
    Gelu,
    GeluErf,
    Silu,
    Tanh,
    Sigmoid,
    Exp,
    Log,
    Sqrt,
    Sqr,
    Abs,
    Neg,
    Recip,
    Sin,
    Cos,
    /// Tangent function: y = tan(x).
    Tan,
    Floor,
    /// Ceiling function: y = ceil(x).
    Ceil,
    Round,
    /// Sign function: y = -1 if x < 0, 0 if x == 0, 1 if x > 0.
    Sign,
    Fract,

    // -- Reductions --
    ReduceSum {
        dim: usize,
        keepdim: bool,
    },
    ReduceMean {
        dim: usize,
        keepdim: bool,
    },
    ReduceMax {
        dim: usize,
        keepdim: bool,
    },
    ReduceMin {
        dim: usize,
        keepdim: bool,
    },

    // -- Shape operations --
    Reshape {
        target_shape: Vec<usize>,
    },
    Transpose {
        dim0: usize,
        dim1: usize,
    },
    Narrow {
        dim: usize,
        start: usize,
        length: usize,
    },
    Unsqueeze {
        dim: usize,
    },
    Squeeze {
        dim: usize,
    },
    Permute {
        axes: Vec<usize>,
    },
    Cat {
        dim: usize,
        num_inputs: usize,
    },

    // -- Normalization --
    /// LayerNorm with weight parameters.
    LayerNorm {
        eps: f64,
        weight: WeightRef,
        bias: WeightRef,
    },
    /// RMSNorm with weight parameters.
    RmsNorm {
        eps: f64,
        weight: WeightRef,
    },
    /// GroupNorm with weight parameters.
    GroupNorm {
        num_groups: usize,
        eps: f64,
        weight: WeightRef,
        bias: WeightRef,
    },
    /// Instance normalization.
    InstanceNorm {
        eps: f64,
    },
    /// Batch normalization.
    BatchNorm {
        eps: f64,
        weight: WeightRef,
        bias: WeightRef,
        running_mean: WeightRef,
        running_var: WeightRef,
    },

    // -- Linear / Conv --
    /// Linear layer: y = x @ weight^T + bias.
    Linear {
        weight: WeightRef,
        bias: Option<WeightRef>,
    },
    /// 1D convolution.
    Conv1d {
        weight: WeightRef,
        bias: Option<WeightRef>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    },
    /// 2D convolution.
    Conv2d {
        weight: WeightRef,
        bias: Option<WeightRef>,
        padding: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    },
    /// 3D convolution.
    Conv3d {
        weight: WeightRef,
        bias: Option<WeightRef>,
        padding: [usize; 3],
        stride: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
    },
    /// 1D transposed convolution.
    ConvTranspose1d {
        weight: WeightRef,
        bias: Option<WeightRef>,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    },
    /// 2D transposed convolution.
    ConvTranspose2d {
        weight: WeightRef,
        bias: Option<WeightRef>,
        padding: [usize; 2],
        output_padding: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    },

    // -- Attention --
    Softmax {
        dim: usize,
    },
    LogSoftmax {
        dim: usize,
    },
    /// Scaled dot-product attention: `softmax(Q @ K^T * scale + mask) @ V`.
    Sdpa {
        scale: f64,
    },
    /// Causal SDPA (no mask tensor). Inputs: Q, K, V. S_q must equal S_kv.
    SdpaCausal {
        scale: f64,
    },
    /// Rotary position embedding applied to Q or K tensor.
    ///
    /// `cos_cache` and `sin_cache` hold the narrowed cos/sin frequency vectors
    /// for the positions `[offset .. offset+seq_len]`. These are captured at
    /// trace time so the NY RoPE layer can access frequency data.
    RotaryEmbedding {
        head_dim: usize,
        offset: usize,
        /// Narrowed cos frequencies: shape `[seq_len, head_dim/2]`.
        cos_cache: WeightRef,
        /// Narrowed sin frequencies: shape `[seq_len, head_dim/2]`.
        sin_cache: WeightRef,
    },
    /// Multi-head attention composite: Q/K/V proj -> optional RoPE -> GQA
    /// repeat -> SDPA -> output proj.
    ///
    /// Inputs, in fixed order: `x` (always), `kv_input` (when
    /// `cross_attention`), attention mask (when `has_mask`, always last).
    /// The mask is a real dataflow edge — auto-registered as a
    /// `ConstantWeight` node when the tensor carries no trace id — never
    /// frozen into the op. `cross_attention`/`has_mask` disambiguate a
    /// 2-input node: without them `[x, kv]` and `[x, mask]` are
    /// indistinguishable on the wire.
    ///
    /// The four projections are captured as data exactly as `Linear` does;
    /// `scale` is the multiplier actually applied to `Q @ K^T` before the
    /// softmax, recorded rather than re-derived. `rope` is `Some` exactly
    /// when RoPE was applied to Q and K inside the fused kernel.
    MultiHeadAttention {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        /// Softmax scale actually applied to `Q @ K^T` (`1/sqrt(head_dim)`).
        scale: f64,
        /// Q projection weight `[num_heads * head_dim, D]`.
        q_weight: WeightRef,
        /// Q projection bias `[num_heads * head_dim]`, or `None`.
        q_bias: Option<WeightRef>,
        /// K projection weight `[num_kv_heads * head_dim, D]`.
        k_weight: WeightRef,
        /// K projection bias `[num_kv_heads * head_dim]`, or `None`.
        k_bias: Option<WeightRef>,
        /// V projection weight `[num_kv_heads * head_dim, D]`.
        v_weight: WeightRef,
        /// V projection bias `[num_kv_heads * head_dim]`, or `None`.
        v_bias: Option<WeightRef>,
        /// Output projection weight `[D_out, num_heads * head_dim]`.
        out_weight: WeightRef,
        /// Output projection bias `[D_out]`, or `None`.
        out_bias: Option<WeightRef>,
        /// RoPE applied to Q and K inside the fused kernel; `None` = no RoPE.
        rope: Option<Box<TraceMhaRope>>,
        /// `true` when input 1 is a separate K/V source (cross-attention).
        cross_attention: bool,
        /// `true` when the last input is an additive attention mask.
        has_mask: bool,
    },
    Embedding {
        weight: WeightRef,
    },

    // -- Recurrent --
    /// LSTM cell: `h_new, c_new = lstm_cell(x, h, c, w_ih, w_hh, b_ih, b_hh)`.
    ///
    /// When `initial_hidden`/`initial_cell` are `None`, zero-initialized states
    /// are used. This is sound for first-timestep verification (h_0=0, c_0=0)
    /// but NOT for warm-start or multi-timestep — see #2401.
    Lstm {
        weight_ih: WeightRef,
        weight_hh: WeightRef,
        bias_ih: Option<WeightRef>,
        bias_hh: Option<WeightRef>,
        hidden_size: usize,
        /// Optional initial hidden state tensor from trace graph.
        /// `None` means zero-initialized (first-timestep assumption).
        initial_hidden: Option<NodeId>,
        /// Optional initial cell state tensor from trace graph.
        /// `None` means zero-initialized (first-timestep assumption).
        initial_cell: Option<NodeId>,
    },

    // -- Pooling --
    MaxPool1d {
        kernel_size: usize,
        stride: usize,
        padding: usize,
    },
    AvgPool2d {
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    },
    MaxPool2d {
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    },
    AdaptiveAvgPool2d {
        output_size: [usize; 2],
    },
    /// 1-D average pooling.
    AvgPool1d {
        kernel_size: usize,
        stride: usize,
        padding: usize,
    },
    /// Adaptive 1-D average pooling: output spatial dim is fixed.
    AdaptiveAvgPool1d {
        output_size: usize,
    },
    /// Adaptive 2-D max pooling: output spatial dims are fixed.
    AdaptiveMaxPool2d {
        output_size: [usize; 2],
    },

    // -- Activation --
    /// Named activation function (Relu, Gelu, Silu, Sigmoid, Tanh, etc.).
    Activation {
        kind: TraceActivation,
    },
    /// Exponential Linear Unit: `x if x > 0, alpha * (exp(x) - 1) otherwise`.
    Elu {
        alpha: f64,
    },
    /// Leaky ReLU: `x if x > 0, slope * x otherwise`.
    LeakyRelu {
        slope: f64,
    },
    /// Softplus: `log(1 + exp(x))`. Smooth approximation of ReLU.
    Softplus,
    /// SELU (Scaled ELU): `lambda * (x if x >= 0, else alpha * (exp(x) - 1))`.
    /// Uses fixed constants: alpha ~= 1.6733, lambda ~= 1.0507.
    Selu,
    /// CELU (Continuous ELU): `max(0,x) + min(0, alpha*(exp(x/alpha)-1))`.
    Celu {
        alpha: f64,
    },
    /// Mish activation: `x * tanh(softplus(x))`.
    Mish,
    /// HardSigmoid: `max(0, min(1, alpha*x + beta))`.
    HardSigmoid,
    /// HardSwish: `x * HardSigmoid(x)`.
    HardSwish,
    /// Softsign: `x / (1 + |x|)`. Output range (-1, 1).
    Softsign,
    /// PReLU (Parametric ReLU): `x if x >= 0, else slope * x`.
    PRelu {
        slope: WeightRef,
    },
    /// Kokoro-specific fused activation/normalization ops.
    /// See [`KokoroFusedOp`] for variant details.
    KokoroFused(KokoroFusedOp),
    /// SwiGLU gated feedforward: `w_down(silu(w_gate(x)) * w_up(x))`.
    SwiGlu,
    /// Dropout (identity at inference).
    Dropout,

    // -- Vision --
    /// PixelShuffle: `[B, C*r², H, W] → [B, C, H*r, W*r]`.
    PixelShuffle {
        upscale_factor: usize,
    },
    /// PixelUnshuffle: `[B, C, H*r, W*r] → [B, C*r², H, W]`.
    PixelUnshuffle {
        downscale_factor: usize,
    },
    /// 1-D nearest-neighbor upsampling: `[..., T] → [..., T * factor]`.
    Upsample1d {
        factor: usize,
    },
    /// 2-D upsampling (nearest or bilinear).
    ///
    /// `align_corners` selects the coordinate mapping used by the
    /// interpolating modes, and is what makes two otherwise identical
    /// `Bilinear` nodes compute different functions:
    ///
    /// * `true` — `src = dst * (in_size - 1) / (out_size - 1)`
    /// * `false` — `src = (dst + 0.5) * in_size / out_size - 0.5`
    ///
    /// For [`TraceUpsampleMode::Nearest`] the mapping is `src = dst / scale`
    /// regardless, so the flag carries no semantics there; the nearest
    /// record site always records `false`.
    Upsample2d {
        mode: TraceUpsampleMode,
        scale_h: f64,
        scale_w: f64,
        /// Corner-alignment convention. Meaningful for `Bilinear`; always
        /// `false` for `Nearest`, whose mapping does not depend on it.
        align_corners: bool,
    },
    /// Bilinear interpolation resize to absolute target dimensions.
    ResizeBilinear {
        target_h: usize,
        target_w: usize,
    },

    // -- Spatial mask / sampling --
    /// Upper-triangular mask: zero out elements below the k-th diagonal.
    Triu {
        diagonal: i64,
    },
    /// Lower-triangular mask: zero out elements above the k-th diagonal.
    Tril {
        diagonal: i64,
    },
    /// Bilinear grid sampling at arbitrary 2D coordinates.
    GridSample {
        padding_mode: GridSamplePaddingMode,
        align_corners: bool,
    },
    /// Quantized linear layer.
    QLinear {
        weight: WeightRef,
        bias: Option<WeightRef>,
    },

    // -- Selection / indexing --
    /// Select the top-k values and their indices along a dimension.
    ///
    /// `largest` and `sorted` are not presentation details — together with
    /// `k`/`dim` they pick out one of four different functions, and bounds
    /// derived for one do not enclose another (the k largest and the k
    /// smallest of a lane share no value in general).
    Topk {
        k: usize,
        dim: usize,
        /// `true` selects the k largest values, `false` the k smallest.
        largest: bool,
        /// `true` when output order along `dim` follows the selection order
        /// (descending for `largest`, ascending otherwise); `false` leaves
        /// the order of the selected set unspecified.
        sorted: bool,
    },
    /// Index of the maximum value along a dimension.
    Argmax {
        dim: usize,
    },
    /// Index of the minimum value along a dimension.
    Argmin {
        dim: usize,
    },
    /// Indices that would sort along a dimension.
    ArgSort {
        dim: usize,
        descending: bool,
    },
    /// Sort values and return indices along a dimension.
    Sort {
        dim: usize,
        descending: bool,
    },
    /// Select elements along `dim` using 1-D index tensor.
    IndexSelect {
        dim: usize,
    },
    /// Gather elements using N-D index tensor along `dim`.
    Gather {
        dim: usize,
    },
    /// Element-wise conditional select (ternary).
    WhereCond,
    /// Broadcast-expand tensor to larger shape.
    Expand {
        target_shape: Vec<usize>,
    },
    /// Element-wise scalar comparison producing a mask tensor.
    Compare {
        op: CompareOp,
        value: f64,
    },
    /// Element-wise tensor-vs-tensor comparison producing a mask tensor.
    CompareTensor {
        op: CompareOp,
    },
    /// Cumulative sum along a dimension.
    Cumsum {
        dim: usize,
    },
    /// Repeat each element along `dim` by variable counts.
    RepeatInterleave {
        dim: usize,
    },
    /// Element-wise power: `x^exponent`.
    Powf {
        exponent: f64,
    },
    /// Convert tensor to a different dtype.
    ToDtype {
        target_dtype: DType,
    },

    // -- Shape operations (extended) --
    /// Reverse elements along a dimension.
    Flip {
        dim: usize,
    },
    /// Circular shift along specified dimensions.
    Roll {
        shifts: Vec<i64>,
        dims: Vec<usize>,
    },
    /// Sliding window extraction (e.g. STFT framing).
    Unfold {
        dim: usize,
        size: usize,
        step: usize,
    },
    /// Write `src` into `self` at a slice along `dim` (KV cache updates).
    SliceSet {
        dim: usize,
        start: usize,
    },
    /// Scatter `src` into `self` along `dim` using `index` (overwrite).
    Scatter {
        dim: usize,
    },
    /// Scatter-add `src` into `self` along `dim` using `index`.
    ScatterAdd {
        dim: usize,
    },
    /// Index-add `src` into `self` along `dim` using `index`.
    IndexAdd {
        dim: usize,
    },
    /// Non-mutating index-put: write src into self at positions along dim.
    IndexPut {
        dim: usize,
    },
    /// Clamp values to [min, max] range.
    Clamp {
        min: Option<f64>,
        max: Option<f64>,
    },
    /// Constant value (scalar or filled tensor) injected during tracing.
    ///
    /// Registered automatically when `full()` or `scalar_like()` is called
    /// during active tracing, so that downstream binary ops can reference
    /// this tensor's trace node ID instead of failing with "no trace ID".
    Constant {
        value: f64,
    },
    // -- Padding --
    /// 1-D reflection padding: pad left and right by reflecting boundary values.
    ReflectionPad1d {
        pad_left: usize,
        pad_right: usize,
    },
    /// 2-D reflection padding: pad left, right, top, bottom by reflecting boundary values.
    ReflectionPad2d {
        pad_left: usize,
        pad_right: usize,
        pad_top: usize,
        pad_bottom: usize,
    },
    /// N-D constant padding: pad with a constant value.
    ConstantPadNd {
        padding: Vec<usize>,
        value: f64,
    },

    /// Two-argument arctangent: `atan2(y, x)`. Binary op: inputs are (y, x).
    Atan2,

    // -- Tensor creation --
    /// Monotonic integer range `[start, end)` with step.
    Arange {
        start: f64,
        end: f64,
        step: f64,
    },

    /// Pipeline segment boundary marker for data-dependent ops (#2378).
    ///
    /// Inserted after operations whose output shape depends on tensor *values*
    /// (e.g., `length_regulate` with data-dependent `repeat_interleave`).
    /// The verify path (nn-verify) splits graphs at these markers and
    /// verifies each segment independently. The compile path (nn-dsl)
    /// ignores this variant — actual ops (RepeatInterleave) are preserved.
    SegmentBoundary {
        /// Human-readable reason (e.g., "length_regulate").
        reason: String,
        /// Optional (lower, upper) bounds hint for the segment output.
        /// When `None`, the verify path uses conservative defaults.
        input_bounds: Option<(f32, f32)>,
    },

    /// MoE gating: softmax + top-k expert routing.
    ///
    /// Input: `[..., model_dim]` hidden states.
    /// Produces routing weights and expert indices for top-k experts.
    /// Used by [`MoeRouter`] and [`MoeLayer`] for mixture-of-experts dispatch.
    MoeGating {
        /// Total number of experts in the MoE layer.
        num_experts: usize,
        /// Number of experts selected per token.
        top_k: usize,
    },

    /// Custom/unknown operation (for extensibility).
    Custom {
        name: String,
    },
}

/// RoPE capture fused inside a [`TraceOp::MultiHeadAttention`] node.
///
/// Q is rotated over `S_q` and K over `S_kv` at the same `offset`; in
/// cross-attention those lengths differ, which is why BOTH windows are
/// captured rather than one shared pair.
///
/// Like every frozen capture in [`TraceOp`], the windows are valid only for
/// the traced sequence length and position offset.
#[derive(Debug, Clone)]
pub struct TraceMhaRope {
    /// Head dimension the rotation was applied over (even).
    pub head_dim: usize,
    /// Starting position of the rotation (0 for prefill).
    pub offset: usize,
    /// Cos window applied to Q: shape `[S_q, head_dim / 2]`.
    pub q_cos_cache: WeightRef,
    /// Sin window applied to Q: shape `[S_q, head_dim / 2]`.
    pub q_sin_cache: WeightRef,
    /// Cos window applied to K: shape `[S_kv, head_dim / 2]`.
    pub k_cos_cache: WeightRef,
    /// Sin window applied to K: shape `[S_kv, head_dim / 2]`.
    pub k_sin_cache: WeightRef,
}
