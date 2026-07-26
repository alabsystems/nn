// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor IR operation types — `TensorOpKind` enum and display impls.
//! Supporting types live in `tensor_ir_types.rs`; all re-exported from `tensor_ir`.

use super::types::{AttentionMask, BroadcastAlignment, Pool2dParams, ReduceOp, TensorNodeId};
use crate::ir::KernelDef;

/// A tensor-level operation node.
///
/// Unlike scalar `IRNodeKind`, tensor ops consume and produce multi-element tensors
/// with explicit shape information. The graph is topologically ordered: each node
/// references only earlier nodes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum TensorOpKind {
    /// Kernel input tensor with a named binding and static shape.
    Input { name: String, shape: Vec<usize> },

    /// Reshape a tensor without changing its element count.
    ///
    /// Validation enforces that `product(input.shape) == product(target_shape)`.
    Reshape {
        input: TensorNodeId,
        target_shape: Vec<usize>,
    },

    /// Select one index along an axis and remove that axis from the output shape.
    ///
    /// Axis 0 is reserved in tensor verification paths and rejected at validation.
    AxisSelect {
        input: TensorNodeId,
        axis: usize,
        index: usize,
    },

    /// Stack same-shaped tensors by inserting a new axis.
    ///
    /// Axis 0 is reserved in tensor verification paths and rejected at validation.
    Stack {
        inputs: Vec<TensorNodeId>,
        axis: usize,
    },

    /// Concatenate tensors along an existing axis.
    ///
    /// All inputs must have identical shapes except at `axis`, where the output
    /// dimension equals the sum of input dimensions. Unlike `Stack`, no new axis
    /// is inserted. Used for head merging in multi-head attention, KV cache
    /// appending, and sequence concatenation.
    Concat {
        inputs: Vec<TensorNodeId>,
        axis: usize,
    },

    /// Reduce over a single axis.
    ///
    /// When `keepdim` is `false`, the `axis` dimension is removed from the output.
    /// When `keepdim` is `true`, the `axis` dimension is retained with size 1.
    Reduce {
        op: ReduceOp,
        input: TensorNodeId,
        axis: usize,
        /// Whether to keep the reduced axis as a size-1 dimension.
        keepdim: bool,
    },

    /// Element-wise operation using a scalar `KernelDef`.
    ///
    /// Each scalar parameter in the kernel maps to one tensor input.
    /// All input tensors must be broadcast-compatible. The scalar kernel
    /// is applied element-wise over the broadcast shape.
    Elementwise {
        kernel: KernelDef,
        inputs: Vec<TensorNodeId>,
    },

    /// Broadcast a tensor to a target shape with explicit alignment.
    Broadcast {
        input: TensorNodeId,
        target_shape: Vec<usize>,
        alignment: BroadcastAlignment,
    },

    /// 1D convolution with weight and optional bias tensors.
    Conv1d {
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride: usize,
        /// Zero-padding applied to both sides of the input.
        padding: usize,
        /// Dilation factor (must be >= 1). Default 1.
        dilation: usize,
        /// Number of input channel groups (must be >= 1). Default 1.
        groups: usize,
    },

    /// 2D convolution with weight and optional bias tensors.
    ///
    /// Input `[C_in, H, W]`, weight `[C_out, C_in/groups, kH, kW]`, output `[C_out, oH, oW]`.
    /// `oH = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1` (same for W).
    Conv2d {
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride_h: usize,
        stride_w: usize,
        /// Zero-padding applied to both sides of each spatial dimension.
        padding_h: usize,
        padding_w: usize,
        /// Dilation factor per dimension (must be >= 1). Default 1.
        dilation_h: usize,
        dilation_w: usize,
        /// Number of input channel groups (must be >= 1). Default 1.
        groups: usize,
    },

    /// Transposed 1D convolution (upsampling).
    ///
    /// Weight `[C_in, C_out/groups, K]`, output `[C_out, L_out]`
    /// where `L_out = (L_in-1)*stride - 2*pad + dilation*(K-1) + output_padding + 1`.
    ConvTranspose1d {
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride: usize,
        /// Zero-padding applied to both sides.
        padding: usize,
        /// Dilation (spacing between kernel elements).
        dilation: usize,
        groups: usize,
        /// Extra output padding (must be < stride).
        output_padding: usize,
    },

    /// Transposed 2D convolution (upsampling).
    ///
    /// Weight `[C_in, C_out/groups, kH, kW]`, output `[C_out, H_out, W_out]`
    /// where `H_out = (H_in-1)*stride_h - 2*pad_h + dilation_h*(kH-1) + output_padding_h + 1`.
    ConvTranspose2d {
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        groups: usize,
        output_padding_h: usize,
        output_padding_w: usize,
    },

    /// Instance normalization with optional affine transform.
    ///
    /// `y[c,t] = gamma[c] * (x[c,t] - mean_c) / sqrt(var_c + eps) + beta[c]`.
    /// Input `[B, C, T]` or `[C, T]`. Output shape matches input.
    /// When gamma/beta are `None`, equivalent to gamma=1, beta=0.
    InstanceNorm1d {
        /// Input tensor node (shape `[B, C, T]` or `[C, T]`).
        input: TensorNodeId,
        /// Epsilon constant node (scalar).
        eps: TensorNodeId,
        /// The time/spatial axis to normalize over (typically the last axis).
        axis: usize,
        /// Optional scale parameter per channel (shape `[C]`).
        gamma: Option<TensorNodeId>,
        /// Optional shift parameter per channel (shape `[C]`).
        beta: Option<TensorNodeId>,
    },

    /// RMS normalization: `x * rsqrt(mean(x²) + eps) * weight`.
    ///
    /// Input `[N, hidden]`, weight `[hidden]`. Output shape matches input.
    RmsNorm {
        input: TensorNodeId,
        /// Scalar constant, shape `[1]`.
        eps: TensorNodeId,
        /// Must be the last axis.
        axis: usize,
        /// Per-feature weight, shape `[hidden]`.
        weight: TensorNodeId,
    },

    /// Element-wise binary addition: `output[i] = left[i] + right[i]`.
    /// Both inputs must have identical shapes. Output shape matches input.
    BinaryAdd {
        left: TensorNodeId,
        right: TensorNodeId,
    },

    /// Element-wise binary multiplication: `output[i] = left[i] * right[i]`.
    /// Both inputs must have identical shapes. Output shape matches input.
    BinaryMul {
        left: TensorNodeId,
        right: TensorNodeId,
    },

    /// Elementwise sigmoid: `1 / (1 + exp(-x))`. Output shape matches input.
    Sigmoid { input: TensorNodeId },
    /// Elementwise SiLU / swish: `x * sigmoid(x) = x / (1 + exp(-x))`.
    /// Output shape matches input.
    ///
    /// A *fused* node (one op) rather than a `Sigmoid` + `BinaryMul`
    /// decomposition. The fusion is what lets the verifier (ny) recognize the
    /// SwiGLU `MulBinary(SiLU(gate), up)` pattern and apply its up/gate
    /// correlation-aware zonotope tightening instead of decorrelating the
    /// product. Translates to ny `Layer::SiLU`.
    Silu { input: TensorNodeId },
    /// Elementwise GELU (tanh approx). Output shape matches input.
    Gelu { input: TensorNodeId },
    /// Elementwise GELU (exact erf). Output shape matches input.
    ///
    /// `0.5 * x * (1 + erf(x / sqrt(2)))`. More precise than the tanh
    /// approximation (`Gelu`) but slower. Used by models that trace
    /// `gelu_erf()` rather than `gelu()`.
    GeluErf { input: TensorNodeId },
    /// Elementwise ReLU: `max(x, 0.0)`. Output shape matches input.
    Relu { input: TensorNodeId },

    /// Elementwise LeakyReLU: `x if x >= 0, else negative_slope * x`.
    /// Output shape matches input. Used in Kokoro decoder (ISTFTNet vocoder).
    LeakyRelu {
        input: TensorNodeId,
        /// Negative slope (e.g., 0.01 or 0.1). Must be finite.
        negative_slope: f32,
    },

    /// Elementwise ELU: `x if x >= 0, else alpha * (exp(x) - 1)`.
    /// Output shape matches input. Single-dispatch kernel — replaces the
    /// previous ~7 dispatch decomposition. Part of #3230 (Gap 3).
    Elu {
        input: TensorNodeId,
        /// Alpha scale for negative region (e.g., 1.0). Must be finite.
        alpha: f32,
    },

    /// Elementwise tanh. Output shape matches input.
    Tanh { input: TensorNodeId },

    /// Elementwise softplus: `ln(1 + exp(x))`.
    Softplus { input: TensorNodeId },
    /// Elementwise exponential: `exp(x)`.
    Exp { input: TensorNodeId },

    /// Adaptive instance normalization: `style_gamma * InstanceNorm(x) + style_beta`.
    /// Input `[C, T]` or `[B, C, T]`. Output shape matches input.
    AdaIN1d {
        /// Input tensor node (shape `[C, T]` or `[B, C, T]`).
        input: TensorNodeId,
        /// Epsilon constant node (scalar, shape `[1]`).
        eps: TensorNodeId,
        /// The time/spatial axis to normalize over (must be the last axis).
        axis: usize,
        /// Style scale parameter per channel (shape `[C]`).
        style_gamma: TensorNodeId,
        /// Style shift parameter per channel (shape `[C]`).
        style_beta: TensorNodeId,
    },

    /// Extract a contiguous slice `[start, start+length)` along one axis.
    /// Output shape: input shape with `shape[axis]` replaced by `length`.
    /// Unlike `AxisSelect`, preserves the sliced axis.
    Narrow {
        input: TensorNodeId,
        axis: usize,
        start: usize,
        length: usize,
    },

    /// Fully-connected linear layer: `y = x @ W^T + b`.
    ///
    /// Input `[*, in_features]`, weight `[out_features, in_features]`, output `[*, out_features]`.
    /// Unary case (fixed weight). For binary case use `MatMul`.
    Linear {
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
    },

    /// Binary matrix multiplication: `y = left @ right * scale`.
    ///
    /// Left `[*, M, K]`, right `[*, K, N]` (or `[*, N, K]` if `transpose_right`).
    /// Output `[*, M, N]`. Binary case (both inputs bounded); for unary see `Linear`.
    MatMul {
        left: TensorNodeId,
        right: TensorNodeId,
        transpose_right: bool,
        /// Scaling factor applied to the result (e.g., `1/sqrt(d_k)`).
        scale: Option<f32>,
    },

    /// Softmax: `exp(x[i]) / sum(exp(x), axis)`. Output shape matches input.
    /// Axis uses Python-style negative indexing: -1 = last. Range `[-rank, rank)`.
    Softmax {
        input: TensorNodeId,
        /// Supports negative indexing: -1 = last.
        axis: i32,
    },

    /// LogSoftmax: `log(softmax(x, axis))` with numerical stability.
    /// Decomposed to `x - logsumexp(x, axis)` at the kernel level.
    /// Axis uses Python-style negative indexing: -1 = last. Range `[-rank, rank)`.
    LogSoftmax {
        input: TensorNodeId,
        /// Supports negative indexing: -1 = last.
        axis: i32,
    },

    /// Zero-pad a 1-D time axis: `shape[last] += pad_left + pad_right`.
    /// Padded elements are exactly 0.0 with bounds `[0.0, 0.0]`.
    ZeroPad1d {
        input: TensorNodeId,
        pad_left: usize,
        pad_right: usize,
    },

    /// Embedding lookup: `output[*] = weight[input[*]]`.
    /// Input `[*]` (indices), weight `[V, D]`, output `[*, D]`.
    Embedding {
        input: TensorNodeId,
        weight: TensorNodeId,
    },

    /// LayerNorm: normalize along `axis`, scale by `weight` (gamma), shift by `bias` (beta).
    LayerNorm {
        input: TensorNodeId,
        eps: TensorNodeId,
        axis: usize,
        weight: TensorNodeId,
        bias: TensorNodeId,
    },

    /// Monolithic self-attention: `softmax(Q @ K^T * scale, mask) @ V`.
    /// Q `[*, T, D]`, K `[*, T_kv, D]`, V `[*, T_kv, D_v]`, output `[*, T, D_v]`.
    Attention {
        q: TensorNodeId,
        k: TensorNodeId,
        v: TensorNodeId,
        mask: AttentionMask,
        scale: Option<f32>,
    },

    /// Permute tensor dimensions. `axes` is a valid permutation of `[0..rank)`.
    /// Example: `[1, 0, 2]` on `[A, B, C]` → `[B, A, C]`.
    Transpose {
        input: TensorNodeId,
        /// Permutation of `[0..rank)`. Length must equal input rank.
        axes: Vec<usize>,
    },

    /// LSTM cell (single time-step). PyTorch `nn.LSTMCell` convention.
    /// Decomposes to Linear+Sigmoid+Tanh+BinaryMul+BinaryAdd (21 gc nodes).
    Lstm {
        input: TensorNodeId,
        hidden_state: TensorNodeId,
        cell_state: TensorNodeId,
        weight_ih: TensorNodeId,
        weight_hh: TensorNodeId,
        bias: Option<TensorNodeId>,
    },

    /// 2D average pooling: reduces each spatial window to its mean.
    AvgPool2d {
        input: TensorNodeId,
        params: Pool2dParams,
    },

    /// 2D max pooling: reduces each spatial window to its maximum.
    MaxPool2d {
        input: TensorNodeId,
        params: Pool2dParams,
    },

    /// Batch normalization using frozen running statistics (inference mode).
    ///
    /// `y[c, ...] = gamma[c] * (x[c, ...] - mean[c]) / sqrt(var[c] + eps) + beta[c]`.
    /// Input `[B, C, ...]` or `[C, ...]`. Output shape matches input.
    /// NY pre-computes `scale = gamma / sqrt(var + eps)` and
    /// `bias = beta - mean * scale` internally.
    BatchNorm {
        input: TensorNodeId,
        /// Per-channel running mean from training, shape `[C]`.
        running_mean: TensorNodeId,
        /// Per-channel running variance from training, shape `[C]`.
        running_var: TensorNodeId,
        /// Per-channel scale (gamma), shape `[C]`.
        weight: TensorNodeId,
        /// Per-channel shift (beta), shape `[C]`.
        bias: TensorNodeId,
        /// Scalar constant node, shape `[1]`.
        eps: TensorNodeId,
    },

    /// Gated DeltaNet recurrent cell (single time-step, arXiv 2412.06464).
    ///
    /// `S_t = gate * S_{t-1} + k^T @ (beta * (v - gate * S_{t-1} @ k))`
    /// `o_t = scale * q @ S_t`
    ///
    /// Decomposes to MatMul+BinaryMul+BinaryAdd (see `gated_delta_net.rs`).
    GatedDeltaNet {
        /// Query tensor `[*, H, K]` (per-head, L2-normalized).
        q: TensorNodeId,
        /// Key tensor `[*, H, K]` (per-head, L2-normalized).
        k: TensorNodeId,
        /// Value tensor `[*, H, V]`.
        v: TensorNodeId,
        /// Recurrent state from previous timestep `[*, H, K, V]`.
        state: TensorNodeId,
        /// Decay gate `exp(g_t)` in `(0, 1)`, shape `[*, H]` or `[*, H, 1, 1]`.
        gate: TensorNodeId,
        /// Write strength `beta_t` in `(0, 1)`, shape `[*, H]` or `[*, H, 1]`.
        beta: TensorNodeId,
        /// Scale factor (typically `1/sqrt(K)`).
        scale: f32,
    },

    /// Index-select: gather slices from `input` along `dim` using 1-D `indices`.
    ///
    /// `output[..., i, ...] = input[..., indices[i], ...]` where dim `dim`
    /// is replaced by the index lookup. Generalizes Embedding to arbitrary
    /// dimensions. Indices are stored as f32 (cast to uint in MSL).
    IndexSelect {
        input: TensorNodeId,
        /// 1-D index tensor (stored as f32, cast to uint in MSL).
        indices: TensorNodeId,
        dim: usize,
    },

    /// Gather: like IndexSelect but the index tensor has the same rank as input.
    ///
    /// `output[i][j][k] = input[i][index[i][j][k]][k]` for dim=1 (example).
    /// The index tensor has the same shape as the output.
    Gather {
        input: TensorNodeId,
        /// Same rank as input (stored as f32, cast to uint in MSL).
        indices: TensorNodeId,
        dim: usize,
    },
}
