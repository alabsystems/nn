// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! The `DispatchStep` enum — one step in a tensor kernel dispatch plan.
//!
//! Extracted from `codegen_msl_tensor.rs` to keep that file under 500 lines.
//! Each variant corresponds to a single Metal (or CPU) kernel dispatch.

use crate::ir::{KernelDef, ScalarType};
use crate::tensor_ir::{BroadcastAlignment, ReduceOp, TensorNodeId};

#[path = "dispatch_step_broadcast.rs"]
mod broadcast;
#[allow(unreachable_pub)]
pub use broadcast::{BinaryBroadcastInfo, BroadcastSide};

#[path = "dispatch_step_conv.rs"]
mod conv;
pub use conv::{Conv1dParams, Conv2dParams, ConvTranspose1dParams};

#[path = "dispatch_step_simdgroup.rs"]
mod simdgroup;
pub use simdgroup::{SimdgroupLinearParams, SimdgroupMatMulParams};

#[path = "dispatch_step_tiled.rs"]
mod tiled;
pub use tiled::{TiledLinearParams, TiledMatMulParams, TILED_GEMM_TILE};

#[path = "dispatch_step_query.rs"]
mod query;
pub use query::{tiled_transpose_2d_params, TILED_TRANSPOSE_TILE_SIZE};

/// A single step in a tensor kernel dispatch plan.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DispatchStep {
    /// Launch a reduction kernel.
    Reduce {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Which reduce operation (Sum or Mean).
        op: ReduceOp,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Tensor node that provides the input buffer.
        input: TensorNodeId,
        /// Tensor node that receives the output buffer.
        output: TensorNodeId,
        /// Size of the dimension being reduced.
        reduce_dim: usize,
        /// Number of independent reduction slices (product of non-reduced dims).
        outer_size: usize,
        /// Whether to keep the reduced dimension as size 1 in the output shape.
        keepdim: bool,
    },
    /// Launch an element-wise kernel using the scalar codegen infrastructure.
    ///
    /// The `scalar_kernel` field carries the full `KernelDef` IR so that
    /// `emit_tensor_msl` can emit a self-contained MSL string without
    /// requiring the caller to separately compose scalar kernel bodies.
    Elementwise {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar `KernelDef` IR for the element-wise operation.
        scalar_kernel: KernelDef,
        /// Tensor node IDs for input buffers.
        inputs: Vec<TensorNodeId>,
        /// Tensor node that receives the output buffer.
        output: TensorNodeId,
        /// Total number of elements to process.
        total_elements: usize,
    },
    /// Broadcast: copy or alias a smaller buffer into a larger shape.
    ///
    /// For MVP, broadcast is implemented as a simple element-wise copy with
    /// modular indexing. Buffer aliasing optimization is deferred.
    Broadcast {
        kernel_name: String,
        dtype: ScalarType,
        input: TensorNodeId,
        output: TensorNodeId,
        /// Shape of the input tensor.
        input_shape: Vec<usize>,
        /// Shape of the output (broadcast target) tensor.
        output_shape: Vec<usize>,
        /// Total elements in the output.
        total_elements: usize,
        /// How input dims map to output dims (left-aligned vs right-aligned).
        alignment: BroadcastAlignment,
    },
    /// Launch a Conv1d kernel (strided sliding-window weighted sum).
    Conv1d(Conv1dParams),
    /// Launch a Conv2d kernel (2D convolution).
    Conv2d(Conv2dParams),
    /// Launch a ConvTranspose1d kernel (transposed convolution / upsampling).
    ConvTranspose1d(ConvTranspose1dParams),
    /// Linear layer (fully-connected): `out[row, col] = dot(input[row], weight[col]) + bias[col]`.
    ///
    /// Naive row-major GEMV/GEMM dispatch. Each output element is an independent
    /// dot product — suitable for correctness testing. Tiled GEMM optimization deferred.
    Linear {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Tensor node for input data (shape `[*, in_features]`).
        input: TensorNodeId,
        /// Tensor node for weight matrix (shape `[out_features, in_features]`).
        weight: TensorNodeId,
        /// Optional tensor node for bias vector (shape `[out_features]`).
        bias: Option<TensorNodeId>,
        /// Tensor node for output (shape `[*, out_features]`).
        output: TensorNodeId,
        /// Size of the contracted dimension.
        in_features: usize,
        /// Output feature dimension.
        out_features: usize,
        /// Product of all leading (batch) dimensions.
        batch_size: usize,
        /// Total output elements (batch_size * out_features).
        total_elements: usize,
    },
    /// Binary matrix multiplication: `out[b,i,j] = sum_k(left[b,i,k] * right[b,k,j]) * scale`.
    ///
    /// Both inputs are runtime buffers (bounded variables). Naive GEMM dispatch.
    /// Used in transformer attention: `Q @ K^T / sqrt(d_k)` and `attn_weights @ V`.
    MatMul {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Left input tensor node (shape `[*, M, K]`).
        left: TensorNodeId,
        /// Right input tensor node (shape `[*, K, N]` or `[*, N, K]`).
        right: TensorNodeId,
        /// Output tensor node (shape `[*, M, N]`).
        output: TensorNodeId,
        /// Number of rows in left matrix (M).
        m: usize,
        /// Contracted dimension (K).
        k: usize,
        /// Number of columns in output (N).
        n: usize,
        /// Product of all leading batch dimensions.
        batch_size: usize,
        /// Whether right is transposed before multiplication.
        transpose_right: bool,
        /// Whether right tensor has fewer batch dims than left (broadcast right across batches).
        broadcast_right: bool,
        /// Optional scaling factor applied post-multiply.
        scale: Option<f32>,
        /// Total output elements (batch_size * M * N).
        total_elements: usize,
    },
    /// Element-wise binary addition: `out[tid] = left[tid] + right[tid]`.
    ///
    /// When `broadcast` is `Some`, one operand uses modular indexing
    /// (fused Broadcast+BinaryAdd from peephole pass).
    BinaryAdd {
        kernel_name: String,
        dtype: ScalarType,
        left: TensorNodeId,
        right: TensorNodeId,
        output: TensorNodeId,
        total_elements: usize,
        /// If set, one operand is broadcast from a smaller shape.
        broadcast: Option<BinaryBroadcastInfo>,
    },
    /// Element-wise binary multiplication: `out[tid] = left[tid] * right[tid]`.
    ///
    /// When `broadcast` is `Some`, one operand uses modular indexing
    /// (fused Broadcast+BinaryMul from peephole pass).
    BinaryMul {
        kernel_name: String,
        dtype: ScalarType,
        left: TensorNodeId,
        right: TensorNodeId,
        output: TensorNodeId,
        total_elements: usize,
        /// If set, one operand is broadcast from a smaller shape.
        broadcast: Option<BinaryBroadcastInfo>,
    },
    /// Element-wise sigmoid activation: `out[tid] = 1 / (1 + exp(-in[tid]))`.
    Sigmoid {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Input tensor node.
        input: TensorNodeId,
        /// Output tensor node.
        output: TensorNodeId,
        /// Total number of elements to process.
        total_elements: usize,
    },
    /// Element-wise GELU activation (tanh approximation):
    /// `out[tid] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    Gelu {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Input tensor node.
        input: TensorNodeId,
        /// Output tensor node.
        output: TensorNodeId,
        /// Total number of elements to process.
        total_elements: usize,
    },
    /// Element-wise GELU activation (exact erf):
    /// `out[tid] = 0.5 * x * (1 + erf(x / sqrt(2)))`.
    GeluErf {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Input tensor node.
        input: TensorNodeId,
        /// Output tensor node.
        output: TensorNodeId,
        /// Total number of elements to process.
        total_elements: usize,
    },
    /// Element-wise ReLU activation: `out[tid] = max(in[tid], 0)`.
    Relu {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Input tensor node.
        input: TensorNodeId,
        /// Output tensor node.
        output: TensorNodeId,
        /// Total number of elements to process.
        total_elements: usize,
    },
    /// Element-wise tanh activation: `out[tid] = tanh(in[tid])`.
    Tanh {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Input tensor node.
        input: TensorNodeId,
        /// Output tensor node.
        output: TensorNodeId,
        /// Total number of elements to process.
        total_elements: usize,
    },
    /// Element-wise LeakyReLU activation: `out[tid] = select(x, slope*x, x < 0)`.
    ///
    /// Negative slope is baked into the MSL kernel as a compile-time constant —
    /// no extra buffer binding needed. Same 2-buffer dispatch as Relu/Sigmoid/etc.
    /// Part of #3230 (Gap 3).
    LeakyRelu {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Input tensor node.
        input: TensorNodeId,
        /// Output tensor node.
        output: TensorNodeId,
        /// Total number of elements to process.
        total_elements: usize,
        /// Negative slope baked into MSL kernel as constant.
        negative_slope: f32,
    },
    /// Element-wise ELU activation: `out[tid] = select(x, alpha*(exp(x)-1), x < 0)`.
    ///
    /// Alpha is baked into the MSL kernel as a compile-time constant —
    /// no extra buffer binding needed. Same 2-buffer dispatch as LeakyRelu.
    /// Replaces ~7 dispatch decomposition. Part of #3230 (Gap 3).
    Elu {
        kernel_name: String,
        dtype: ScalarType,
        input: TensorNodeId,
        output: TensorNodeId,
        total_elements: usize,
        /// Alpha scale for negative region, baked into MSL kernel.
        alpha: f32,
    },
    /// Element-wise exp activation: `out[tid] = exp(in[tid])`.
    Exp {
        kernel_name: String,
        dtype: ScalarType,
        input: TensorNodeId,
        output: TensorNodeId,
        total_elements: usize,
    },
    /// Element-wise softplus activation: `out[tid] = log(1 + exp(in[tid]))`.
    Softplus {
        kernel_name: String,
        dtype: ScalarType,
        input: TensorNodeId,
        output: TensorNodeId,
        total_elements: usize,
    },
    /// Zero-copy reshape: buffer alias with new shape interpretation.
    Reshape {
        input: TensorNodeId,
        output: TensorNodeId,
    },
    /// Select a single index along an axis (strided copy).
    AxisSelect {
        kernel_name: String,
        dtype: ScalarType,
        input: TensorNodeId,
        output: TensorNodeId,
        input_shape: Vec<usize>,
        axis: usize,
        index: usize,
    },
    /// Stack multiple tensors along a new axis (interleaved write).
    Stack {
        kernel_name: String,
        dtype: ScalarType,
        inputs: Vec<TensorNodeId>,
        output: TensorNodeId,
        input_shape: Vec<usize>,
        axis: usize,
    },
    /// Extract a contiguous slice along one axis (strided copy).
    Narrow {
        kernel_name: String,
        dtype: ScalarType,
        input: TensorNodeId,
        output: TensorNodeId,
        input_shape: Vec<usize>,
        axis: usize,
        start: usize,
        length: usize,
    },
    /// Softmax along an axis: two-pass (max-subtract, exp, sum, normalize).
    ///
    /// Dispatch is per-reduction-slice (one threadgroup per slice along the
    /// softmax axis), similar to Reduce but with a multi-pass kernel.
    Softmax {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Input tensor node.
        input: TensorNodeId,
        /// Output tensor node.
        output: TensorNodeId,
        /// Normalized axis index (non-negative, resolved from potentially negative IR axis).
        axis: usize,
        /// Size of the dimension being softmax'd.
        axis_size: usize,
        /// Number of independent softmax slices (product of all other dims).
        outer_size: usize,
    },
    /// Zero-pad a 1-D time axis: copy input elements and fill padding with 0.0.
    ZeroPad1d {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Input tensor node.
        input: TensorNodeId,
        /// Output tensor node.
        output: TensorNodeId,
        /// Number of channels (product of all dims except the last).
        channels: usize,
        /// Input length (last axis size).
        in_length: usize,
        /// Left (start) zero-padding.
        pad_left: usize,
        /// Output length (in_length + pad_left + pad_right).
        out_length: usize,
    },
    /// Transpose (axis permutation): reorder elements by dimension permutation.
    ///
    /// Output coordinate `c[k]` maps to input axis `axes[k]`. Formally:
    /// `output[c_0, ..., c_{n-1}] = input[d_0, ..., d_{n-1}]` where
    /// `d_{axes[k]} = c_k` for all k.
    Transpose {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Input tensor node.
        input: TensorNodeId,
        /// Output tensor node.
        output: TensorNodeId,
        /// Input tensor shape (pre-transpose).
        input_shape: Vec<usize>,
        /// Axes permutation.
        axes: Vec<usize>,
        /// Total number of elements to process.
        total_elements: usize,
    },
    /// Embedding table lookup: `output[i] = weight[indices[i]]`.
    Embedding {
        /// Name of the generated MSL kernel function.
        kernel_name: String,
        /// Scalar type (f32 or f16).
        dtype: ScalarType,
        /// Input (indices) tensor node.
        input: TensorNodeId,
        /// Weight (embedding table) tensor node.
        weight: TensorNodeId,
        /// Output tensor node.
        output: TensorNodeId,
        /// Embedding dimension (columns in weight table).
        embedding_dim: usize,
        /// Number of index elements (product of input shape).
        num_indices: usize,
        /// Total output elements (num_indices * embedding_dim).
        total_elements: usize,
    },
    /// Concatenate multiple tensors along an existing axis.
    ///
    /// Each output element reads from the correct input buffer at the correct
    /// offset, determined by the cumulative sizes along the concat axis.
    /// Unlike `Stack`, no new axis is inserted — the concat axis in each input
    /// contributes its full extent to the output.
    ///
    /// Part of #810.
    Concat {
        kernel_name: String,
        dtype: ScalarType,
        inputs: Vec<TensorNodeId>,
        output: TensorNodeId,
        /// Shape of the first input (all inputs share shape except at `axis`).
        first_input_shape: Vec<usize>,
        /// Per-input size along the concat axis.
        input_axis_sizes: Vec<usize>,
        axis: usize,
    },
    /// Index-select: gather slices along one axis using 1-D integer indices.
    ///
    /// `output[..., i, ...] = input[..., indices[i], ...]` where `dim` is the
    /// selection axis. Generalizes `Embedding` (which is index_select on dim 0
    /// of a 2-D table). Indices stored as f32, cast to uint in MSL.
    IndexSelect {
        kernel_name: String,
        dtype: ScalarType,
        input: TensorNodeId,
        indices: TensorNodeId,
        output: TensorNodeId,
        dim: usize,
        input_shape: Vec<usize>,
        num_indices: usize,
        total_elements: usize,
    },
    /// Gather: N-D index lookup along one axis.
    ///
    /// `output[i][j][k] = input[i][index[i][j][k]][k]` for dim=1 (example).
    /// Index tensor has the same rank as input; output shape == index shape.
    Gather {
        kernel_name: String,
        dtype: ScalarType,
        input: TensorNodeId,
        indices: TensorNodeId,
        output: TensorNodeId,
        dim: usize,
        input_shape: Vec<usize>,
        total_elements: usize,
    },
    /// Simdgroup-tiled linear layer (simdgroup-conforming shapes). Part of #2275.
    SimdgroupLinear(SimdgroupLinearParams),
    /// Simdgroup-tiled matrix multiplication (simdgroup-conforming shapes). Part of #2275.
    SimdgroupMatMul(SimdgroupMatMulParams),
    /// Tiled shared-memory linear layer (non-simdgroup shapes). Part of #3230 (Gap 1).
    TiledLinear(TiledLinearParams),
    /// Tiled shared-memory matrix multiplication (non-simdgroup shapes). Part of #3230 (Gap 1).
    TiledMatMul(TiledMatMulParams),
}
