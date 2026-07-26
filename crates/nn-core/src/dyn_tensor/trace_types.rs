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
    /// Multi-head attention composite: Q/K/V proj -> SDPA -> output proj.
    MultiHeadAttention {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
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
    Upsample2d {
        mode: TraceUpsampleMode,
        scale_h: f64,
        scale_w: f64,
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
    Topk {
        k: usize,
        dim: usize,
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
