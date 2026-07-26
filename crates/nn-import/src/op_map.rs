// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Maps torch.export aten op target strings to nn `TraceOp` variants.
//!
//! Each aten op (e.g., `"torch.ops.aten.linear.default"`) maps to a `TraceOp`
//! with arguments extracted from the node's named input list.
//!
//! The dispatch table is here; individual mapper functions are in `op_map_impl.rs`.

use nn_core::dyn_tensor::trace::TraceOp;
use nn_core::DType;

use crate::error::ImportError;
use crate::parse::{Argument, Node, TensorMeta};

#[path = "op_map_args.rs"]
mod args;
use args::*;
#[path = "op_map_expand.rs"]
mod expand;
#[path = "op_map_impl.rs"]
mod impls;
#[path = "op_map_impl_dpdf.rs"]
mod impls_dpdf;
#[path = "op_map_impl_ext.rs"]
mod impls_ext;
#[path = "op_map_impl_kokoro.rs"]
mod impls_kokoro;
#[path = "op_map_impl_wave10.rs"]
mod impls_w10;
#[path = "op_map_impl_wave11.rs"]
mod impls_w11;
#[path = "op_map_impl_wave12.rs"]
mod impls_w12;
#[path = "op_map_impl_wave13.rs"]
mod impls_w13;
#[path = "op_map_impl_wave14.rs"]
mod impls_w14;
#[path = "op_map_impl_wave15.rs"]
mod impls_w15;
#[path = "op_map_impl_wave16.rs"]
mod impls_w16;
#[path = "op_map_impl_wave9.rs"]
mod impls_w9;
#[path = "op_map_impl_transformer.rs"]
mod impls_xfmr;

/// Resolved weight data for a parameter placeholder.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ResolvedWeight {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl ResolvedWeight {
    /// Create a new resolved weight.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

/// A single node produced by multi-node decomposition (e.g., BiLSTM).
///
/// Used by `try_expand_node` to return multiple TraceNodes from one aten node.
#[derive(Debug)]
pub(crate) struct ExpandedNode {
    pub name: String,
    pub op: TraceOp,
    pub input_names: Vec<String>,
    pub output_shape: Vec<usize>,
    pub output_dtype: DType,
}

/// Context provided to the op mapper for resolving arguments.
pub struct OpMapContext<'a> {
    /// Maps graph tensor name -> TensorMeta (from `graph.tensor_values`).
    pub tensor_meta: &'a std::collections::HashMap<String, TensorMeta>,
    /// Maps graph placeholder name -> resolved weight data.
    pub weights: &'a std::collections::HashMap<String, ResolvedWeight>,
}

/// Returns a sorted, deduplicated list of all aten operations supported by the
/// import pipeline.
///
/// Each entry is a short aten name (e.g. `"aten::linear"`) derived from the
/// `torch.ops.aten.<name>.<overload>` target strings in the dispatch table.
/// Both the primary `map_node_to_trace_op` match arms and the `try_expand_node`
/// decomposition paths are included.
pub fn supported_ops() -> Vec<&'static str> {
    let mut ops: Vec<&str> = SUPPORTED_ATEN_OPS.to_vec();
    ops.sort_unstable();
    ops.dedup();
    ops
}

/// Static table of all aten operations supported by the import pipeline.
///
/// Derived from `map_node_to_trace_op` match arms plus `try_expand_node`
/// decomposition paths. Grouped by category for maintainability.
const SUPPORTED_ATEN_OPS: &[&str] = &[
    // -- Unary element-wise --
    "aten::relu",
    "aten::gelu",
    "aten::silu",
    "aten::tanh",
    "aten::sigmoid",
    "aten::exp",
    "aten::log",
    "aten::sqrt",
    "aten::abs",
    "aten::neg",
    "aten::reciprocal",
    "aten::sin",
    "aten::cos",
    "aten::floor",
    "aten::round",
    "aten::rsqrt",
    // -- Binary element-wise --
    "aten::add",
    "aten::sub",
    "aten::mul",
    "aten::div",
    "aten::maximum",
    "aten::minimum",
    // -- Matrix multiply --
    "aten::mm",
    "aten::bmm",
    "aten::matmul",
    // -- Linear --
    "aten::linear",
    // -- Convolution --
    "aten::convolution",
    "aten::conv1d",
    "aten::conv2d",
    "aten::conv_transpose1d",
    // -- Normalization --
    "aten::layer_norm",
    "aten::group_norm",
    "aten::batch_norm",
    "aten::native_batch_norm",
    "aten::_native_batch_norm_legit_no_training",
    "aten::instance_norm",
    // -- Attention --
    "aten::softmax",
    "aten::_softmax",
    "aten::log_softmax",
    "aten::_log_softmax",
    "aten::scaled_dot_product_attention",
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::multi_head_attention_forward",
    // -- Embedding --
    "aten::embedding",
    // -- Reductions --
    "aten::sum",
    "aten::mean",
    "aten::amax",
    "aten::amin",
    // -- Shape operations --
    "aten::view",
    "aten::reshape",
    "aten::_unsafe_view",
    "aten::transpose",
    "aten::permute",
    "aten::flatten",
    "aten::unsqueeze",
    "aten::squeeze",
    "aten::cat",
    "aten::slice",
    "aten::expand",
    "aten::flip",
    "aten::chunk",
    "aten::select",
    // -- Pooling --
    "aten::max_pool1d",
    "aten::max_pool1d_with_indices",
    "aten::avg_pool2d",
    "aten::max_pool2d_with_indices",
    "aten::adaptive_avg_pool2d",
    // -- Activation --
    "aten::elu",
    "aten::leaky_relu",
    "aten::dropout",
    "aten::hardtanh",
    "aten::hardsigmoid",
    "aten::hardswish",
    "aten::selu",
    "aten::softplus",
    "aten::mish",
    "aten::celu",
    // -- Comparison / Selection --
    "aten::where",
    "aten::clamp",
    "aten::clamp_min",
    "aten::gt",
    "aten::lt",
    "aten::ge",
    "aten::le",
    "aten::eq",
    "aten::ne",
    // -- Type conversion --
    "aten::to",
    "aten::_to_copy",
    // -- Power --
    "aten::pow",
    // -- Recurrent --
    "aten::lstm",
    // -- Misc --
    "aten::cumsum",
    "aten::repeat_interleave",
    // -- Tensor creation --
    "aten::zeros",
    "aten::zeros_like",
    "aten::ones",
    "aten::ones_like",
    "aten::full",
    "aten::full_like",
    "aten::arange",
    // -- Padding --
    "aten::reflection_pad1d",
    "aten::constant_pad_nd",
    "aten::pad",
    // -- Upsampling --
    "aten::upsample_nearest1d",
    // -- Indexing --
    "aten::index_select",
    // -- Trigonometric --
    "aten::atan2",
    // -- Identity / memory layout --
    "aten::contiguous",
    "aten::clone",
    "aten::_copy",
    // -- dpdf model ops --
    "aten::upsample_nearest2d",
    "aten::upsample_bilinear2d",
    "aten::rms_norm",
    "aten::hardswish",
    "aten::hardsigmoid",
    "aten::mish",
    "aten::softplus",
    "aten::selu",
    "aten::triu",
    "aten::tril",
    "aten::gather",
    "aten::argmax",
    "aten::argmin",
    "aten::pixel_shuffle",
    "aten::pixel_unshuffle",
    "aten::split",
    "aten::split_with_sizes",
    "aten::unbind",
    "aten::repeat",
    // Wave 6: interpolate, scatter, reflection_pad2d, clamp_max
    "aten::interpolate",
    "aten::scatter",
    "aten::reflection_pad2d",
    "aten::clamp_max",
    // -- Advanced indexing / shape ops --
    "aten::stack",
    "aten::narrow",
    "aten::topk",
    "aten::sort",
    "aten::scatter_add",
    "aten::roll",
    // -- Vision model ops (conv3d, grid_sample, meshgrid, index, masked_fill) --
    "aten::conv3d",
    "aten::grid_sample",
    "aten::meshgrid",
    "aten::index",
    "aten::masked_fill",
    // -- Transformer / CNN / audio model ops (Wave 7) --
    // Unary math
    "aten::tan",
    "aten::ceil",
    "aten::sign",
    "aten::sgn",
    "aten::frac",
    "aten::log2",
    "aten::log10",
    "aten::exp2",
    "aten::erf",
    // Activation
    "aten::softsign",
    "aten::prelu",
    "aten::log_sigmoid",
    "aten::log_sigmoid_forward",
    "aten::glu",
    // Missing tensor comparisons
    "aten::ge_tensor",
    "aten::le_tensor",
    "aten::ne_tensor",
    // Conv transpose 2D (standalone)
    "aten::conv_transpose2d",
    // Matrix ops (addmm, baddbmm)
    "aten::addmm",
    "aten::baddbmm",
    // Index ops
    "aten::index_add",
    "aten::index_put",
    "aten::unfold",
    // Tensor creation
    "aten::empty",
    "aten::empty_like",
    "aten::new_zeros",
    "aten::new_ones",
    "aten::linspace",
    "aten::scalar_tensor",
    "aten::fill",
    "aten::zero",
    // Shape ops
    "aten::t",
    "aten::movedim",
    // Power
    "aten::pow_tensor_tensor",
    "aten::pow_scalar",
    // Reductions (no dim)
    "aten::sum_no_dim",
    "aten::mean_no_dim",
    "aten::prod",
    "aten::var",
    "aten::std",
    "aten::any",
    "aten::all",
    // Boolean / logical
    "aten::logical_not",
    "aten::logical_and",
    "aten::logical_or",
    // Miscellaneous
    "aten::remainder",
    "aten::fmod",
    "aten::slice_scatter",
    "aten::copy",
    // -- Vision model ops (pooling, bicubic interpolate) --
    "aten::max_pool2d",
    "aten::avg_pool1d",
    "aten::adaptive_avg_pool1d",
    "aten::adaptive_max_pool2d",
    // -- Vision / audio model ops (Wave 8) --
    "aten::upsample_bicubic2d",
    "aten::replication_pad1d",
    "aten::replication_pad2d",
    "aten::channel_shuffle",
    "aten::adaptive_max_pool1d",
    "aten::nll_loss_forward",
    "aten::mse_loss",
    "aten::l1_loss",
    "aten::smooth_l1_loss",
    "aten::huber_loss",
    "aten::binary_cross_entropy",
    // -- Wave 9: commonly missing model patterns --
    // Unary math
    "aten::trunc",
    "aten::expm1",
    "aten::log1p",
    "aten::acos",
    "aten::asin",
    "aten::atan",
    "aten::cosh",
    "aten::sinh",
    // Value testing
    "aten::isinf",
    "aten::isnan",
    "aten::isfinite",
    // Bitwise
    "aten::bitwise_not",
    "aten::bitwise_and",
    "aten::bitwise_or",
    // Tensor-arg clamp variants
    "aten::clamp_min_tensor",
    "aten::clamp_max_tensor",
    // Tensor creation
    "aten::tile",
    "aten::eye",
    // Expand variants
    "aten::expand_as",
    "aten::broadcast_to",
    // Loss functions
    "aten::binary_cross_entropy_with_logits",
    "aten::cross_entropy_loss",
    // Indexing
    "aten::index_fill",
    "aten::index_copy",
    "aten::scatter_reduce",
    // Repeat
    "aten::repeat_interleave_int",
    // Conditional / where variants
    "aten::masked_scatter",
    // -- Wave 10: additional transformer and training ops --
    "aten::diagonal",
    "aten::rot90",
    "aten::nll_loss",
    "aten::kl_div",
    // -- Wave 11: spatial transformer, grid, and shape ops --
    "aten::affine_grid_generator",
    "aten::triu_",
    "aten::tril_",
    // -- Wave 12: normalization, embedding, and loss overloads --
    "aten::_native_batch_norm_legit",
    "aten::cudnn_batch_norm",
    "aten::embedding_bag",
    "aten::nll_loss_nd",
    "aten::nll_loss2d_forward",
    "aten::mse_loss_backward",
    "aten::l1_loss_backward",
    "aten::smooth_l1_loss_backward",
    "aten::kl_div_backward",
    // -- Wave 13: advanced tensor manipulation and control flow ops --
    "aten::index_put_hacked_twin",
    "aten::masked_select",
    "aten::nonzero",
    "aten::unique",
    "aten::unique_consecutive",
    // -- Wave 14: common missing PyTorch ops --
    "aten::lerp",
    "aten::addcmul",
    "aten::addcdiv",
    "aten::linalg_vector_norm",
    "aten::cdist",
    "aten::multinomial",
    "aten::searchsorted",
    "aten::bucketize",
    "aten::count_nonzero",
    "aten::cumprod",
    "aten::cummax",
    "aten::cummin",
    "aten::one_hot",
    "aten::threshold",
    // -- Wave 15: matrix ops, sampling, creation, and strided views --
    "aten::clamp_tensor",
    "aten::norm",
    "aten::einsum",
    "aten::as_strided",
    "aten::addmv",
    "aten::addr",
    "aten::outer",
    "aten::bernoulli",
    "aten::randn",
    "aten::cross",
    // -- Wave 16: in-place activations, native norms, GRU, complex/FFT, dropout variants --
    "aten::relu_",
    "aten::sigmoid_",
    "aten::tanh_",
    "aten::silu_",
    "aten::gelu_",
    "aten::native_layer_norm",
    "aten::native_group_norm",
    "aten::gru",
    "aten::view_as_real",
    "aten::view_as_complex",
    "aten::fft_rfft",
    "aten::fft_irfft",
    "aten::feature_dropout",
    "aten::alpha_dropout",
];

/// Map a parsed node to a `TraceOp`.
///
/// Returns the `TraceOp` and the list of input tensor names (in dependency order).
///
/// `input_ndim` is the rank of the first input tensor (0 if unknown).
/// Used to resolve negative dimension indices (e.g., `dim=-1` → last dim).
pub fn map_node_to_trace_op(
    node: &Node,
    ctx: &OpMapContext<'_>,
    input_ndim: usize,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let target = node.target.as_str();
    match target {
        // -- Unary element-wise --
        "torch.ops.aten.relu.default" => impls::unary_op(node, TraceOp::Relu),
        "torch.ops.aten.gelu.default" => impls::map_gelu(node),
        "torch.ops.aten.silu.default" => impls::unary_op(node, TraceOp::Silu),
        "torch.ops.aten.tanh.default" => impls::unary_op(node, TraceOp::Tanh),
        "torch.ops.aten.sigmoid.default" => impls::unary_op(node, TraceOp::Sigmoid),
        "torch.ops.aten.exp.default" => impls::unary_op(node, TraceOp::Exp),
        "torch.ops.aten.log.default" => impls::unary_op(node, TraceOp::Log),
        "torch.ops.aten.sqrt.default" => impls::unary_op(node, TraceOp::Sqrt),
        "torch.ops.aten.abs.default" => impls::unary_op(node, TraceOp::Abs),
        "torch.ops.aten.neg.default" => impls::unary_op(node, TraceOp::Neg),
        "torch.ops.aten.reciprocal.default" => impls::unary_op(node, TraceOp::Recip),
        "torch.ops.aten.sin.default" => impls::unary_op(node, TraceOp::Sin),
        "torch.ops.aten.cos.default" => impls::unary_op(node, TraceOp::Cos),
        "torch.ops.aten.floor.default" => impls::unary_op(node, TraceOp::Floor),
        "torch.ops.aten.round.default" => impls::unary_op(node, TraceOp::Round),
        // rsqrt(x) = x^(-0.5) -- no dedicated TraceOp variant, decompose to Powf.
        "torch.ops.aten.rsqrt.default" => impls::unary_op(node, TraceOp::Powf { exponent: -0.5 }),

        // -- Binary element-wise (aten schema: self, other) --
        "torch.ops.aten.add.Tensor" | "torch.ops.aten.add_.Tensor" => {
            impls::binary_op(node, TraceOp::Add, "other")
        }
        "torch.ops.aten.sub.Tensor" => impls::binary_op(node, TraceOp::Sub, "other"),
        "torch.ops.aten.mul.Tensor" => impls::binary_op(node, TraceOp::Mul, "other"),
        "torch.ops.aten.div.Tensor" => impls::binary_op(node, TraceOp::Div, "other"),
        "torch.ops.aten.maximum.default" => impls::binary_op(node, TraceOp::Maximum, "other"),
        "torch.ops.aten.minimum.default" => impls::binary_op(node, TraceOp::Minimum, "other"),

        // -- Matrix multiply (mm/bmm schema: self, mat2; matmul schema: self, other) --
        "torch.ops.aten.mm.default" | "torch.ops.aten.bmm.default" => {
            impls::binary_op(node, TraceOp::MatMul, "mat2")
        }
        "torch.ops.aten.matmul.default" => impls::binary_op(node, TraceOp::MatMul, "other"),

        // -- Linear --
        "torch.ops.aten.linear.default" => impls::map_linear(node, ctx),

        // -- Convolution (unified aten.convolution.default) --
        "torch.ops.aten.convolution.default" => impls::map_convolution(node, ctx),

        // -- Normalization --
        "torch.ops.aten.layer_norm.default" => impls::map_layer_norm(node, ctx),
        "torch.ops.aten.group_norm.default" => impls::map_group_norm(node, ctx),
        "torch.ops.aten.native_batch_norm.default"
        | "torch.ops.aten._native_batch_norm_legit_no_training.default" => {
            impls::map_batch_norm(node, ctx)
        }
        "torch.ops.aten.instance_norm.default" => impls::map_instance_norm(node),

        // -- Attention --
        "torch.ops.aten.softmax.int" | "torch.ops.aten._softmax.default" => {
            impls::map_softmax(node, input_ndim)
        }
        "torch.ops.aten.log_softmax.int" | "torch.ops.aten._log_softmax.default" => {
            impls::map_log_softmax(node, input_ndim)
        }
        "torch.ops.aten.scaled_dot_product_attention.default" => impls::map_sdpa(node, ctx),

        // Flash attention (PyTorch internal CUDA dispatch target)
        "torch.ops.aten._scaled_dot_product_flash_attention.default" => {
            impls_xfmr::map_flash_attention(node, ctx)
        }
        // Efficient attention / xformers (PyTorch internal dispatch target)
        "torch.ops.aten._scaled_dot_product_efficient_attention.default" => {
            impls_xfmr::map_efficient_attention(node, ctx)
        }
        // Full multi-head attention forward
        "torch.ops.aten.multi_head_attention_forward.default" => {
            impls_xfmr::map_multi_head_attention_forward(node, ctx)
        }

        // -- Embedding --
        "torch.ops.aten.embedding.default" => impls::map_embedding(node, ctx),

        // -- Reductions --
        "torch.ops.aten.sum.dim_IntList" => impls::map_reduce_sum(node),
        "torch.ops.aten.mean.dim" => impls::map_reduce_mean(node),
        "torch.ops.aten.amax.default" => impls::map_reduce_max(node),
        "torch.ops.aten.amin.default" => impls::map_reduce_min(node),

        // -- Shape operations --
        "torch.ops.aten.view.default"
        | "torch.ops.aten.reshape.default"
        | "torch.ops.aten._unsafe_view.default" => impls::map_reshape(node),
        "torch.ops.aten.transpose.int" => impls::map_transpose(node),
        "torch.ops.aten.permute.default" => impls::map_permute(node),
        "torch.ops.aten.flatten.using_ints" => {
            // Flatten is handled by try_expand_node when input_shape is available.
            // This fallback errors because we need the input shape to compute
            // the flattened target shape for Reshape.
            Err(ImportError::UnsupportedOp {
                target: format!(
                    "{target} (flatten needs input shape metadata for Reshape decomposition)"
                ),
            })
        }
        "torch.ops.aten.unsqueeze.default" => impls::map_unsqueeze(node),
        "torch.ops.aten.squeeze.dim" => impls::map_squeeze(node),
        "torch.ops.aten.squeeze.default" => impls::map_squeeze_default(node),
        "torch.ops.aten.cat.default" => impls::map_cat(node),
        "torch.ops.aten.slice.Tensor" => impls::map_slice(node),
        "torch.ops.aten.expand.default" => impls::map_expand(node),
        "torch.ops.aten.flip.default" => impls::map_flip(node),

        // -- Pooling (op_map_impl_ext.rs) --
        "torch.ops.aten.max_pool1d.default" | "torch.ops.aten.max_pool1d_with_indices.default" => {
            impls_ext::map_max_pool1d(node)
        }
        "torch.ops.aten.avg_pool2d.default" => impls_ext::map_avg_pool2d(node),
        "torch.ops.aten.max_pool2d_with_indices.default" => impls_ext::map_max_pool2d(node),
        "torch.ops.aten.adaptive_avg_pool2d.default" => impls_ext::map_adaptive_avg_pool2d(node),

        // -- Activation (op_map_impl_ext.rs) --
        "torch.ops.aten.elu.default" => impls_ext::map_elu(node),
        "torch.ops.aten.leaky_relu.default" => impls_ext::map_leaky_relu(node),
        "torch.ops.aten.dropout.default" => impls::unary_op(node, TraceOp::Dropout),
        "torch.ops.aten.hardtanh.default" | "torch.ops.aten.hardtanh_.default" => {
            impls_ext::map_hardtanh(node)
        }
        "torch.ops.aten.hardsigmoid.default" => impls::unary_op(node, TraceOp::HardSigmoid),
        "torch.ops.aten.hardswish.default" | "torch.ops.aten.hardswish_.default" => {
            impls::unary_op(node, TraceOp::HardSwish)
        }
        "torch.ops.aten.selu.default" | "torch.ops.aten.selu_.default" => {
            impls::unary_op(node, TraceOp::Selu)
        }
        "torch.ops.aten.softplus.default" => impls_ext::map_softplus(node),
        "torch.ops.aten.mish.default" => impls::unary_op(node, TraceOp::Mish),
        "torch.ops.aten.celu.default" | "torch.ops.aten.celu_.default" => impls_ext::map_celu(node),

        // -- Comparison / Selection (op_map_impl_ext.rs) --
        "torch.ops.aten.where.self" => impls_ext::map_where_cond(node),
        "torch.ops.aten.clamp.default" | "torch.ops.aten.clamp_min.default" => {
            impls_ext::map_clamp(node)
        }

        // -- Type conversion (op_map_impl_ext.rs) --
        "torch.ops.aten.to.dtype" | "torch.ops.aten._to_copy.default" => {
            impls_ext::map_to_dtype(node)
        }

        // -- Power (op_map_impl_ext.rs) --
        "torch.ops.aten.pow.Tensor_Scalar" => impls_ext::map_powf(node),

        // -- Recurrent (op_map_impl_ext.rs) --
        "torch.ops.aten.lstm.input" => impls_ext::map_lstm(node, ctx),

        // -- Misc (op_map_impl_ext.rs) --
        "torch.ops.aten.cumsum.default" => impls_ext::map_cumsum(node),
        "torch.ops.aten.repeat_interleave.self_Tensor" => impls_ext::map_repeat_interleave(node),

        // -- Zero tensor creation (op_map_impl_ext.rs) --
        "torch.ops.aten.zeros.default" => impls_ext::map_zeros(node),
        "torch.ops.aten.zeros_like.default" => impls_ext::map_zeros_like(node),

        // -- Standalone conv1d / conv2d (op_map_impl_ext.rs) --
        "torch.ops.aten.conv1d.default" => impls_ext::map_conv1d(node, ctx),
        "torch.ops.aten.conv2d.default" => impls_ext::map_conv2d(node, ctx),

        // -- Standalone batch_norm (op_map_impl_ext.rs) --
        "torch.ops.aten.batch_norm.default" => impls_ext::map_batch_norm_standalone(node, ctx),

        // -- Standalone ConvTranspose1d (op_map_impl_kokoro.rs) --
        "torch.ops.aten.conv_transpose1d.default" => impls_kokoro::map_conv_transpose1d(node, ctx),

        // -- Padding (op_map_impl_kokoro.rs) --
        "torch.ops.aten.reflection_pad1d.default" => impls_kokoro::map_reflection_pad1d(node),
        "torch.ops.aten.constant_pad_nd.default" => impls_kokoro::map_constant_pad_nd(node),
        "torch.ops.aten.pad.default" => impls_kokoro::map_pad(node),

        // -- Upsampling (op_map_impl_kokoro.rs) --
        "torch.ops.aten.upsample_nearest1d.default" | "torch.ops.aten.upsample_nearest1d.vec" => {
            impls_kokoro::map_upsample_nearest1d(node)
        }

        // -- Indexing (op_map_impl_kokoro.rs) --
        "torch.ops.aten.index_select.default" => impls_kokoro::map_index_select(node),

        // -- Scalar comparison (op_map_impl_kokoro.rs) --
        "torch.ops.aten.gt.Scalar" => impls_kokoro::map_gt_scalar(node),
        "torch.ops.aten.lt.Scalar" => impls_kokoro::map_lt_scalar(node),
        "torch.ops.aten.ge.Scalar" => impls_kokoro::map_ge_scalar(node),
        "torch.ops.aten.le.Scalar" => impls_kokoro::map_le_scalar(node),
        "torch.ops.aten.eq.Scalar" => impls_kokoro::map_eq_scalar(node),
        "torch.ops.aten.ne.Scalar" => impls_kokoro::map_ne_scalar(node),

        // -- Tensor comparison (op_map_impl_kokoro.rs) --
        "torch.ops.aten.gt.Tensor" => {
            impls_kokoro::map_compare_tensor(node, nn_core::dyn_tensor::CompareOp::Gt)
        }
        "torch.ops.aten.lt.Tensor" => {
            impls_kokoro::map_compare_tensor(node, nn_core::dyn_tensor::CompareOp::Lt)
        }
        "torch.ops.aten.eq.Tensor" => {
            impls_kokoro::map_compare_tensor(node, nn_core::dyn_tensor::CompareOp::Eq)
        }

        // -- Trigonometric extended (op_map_impl_kokoro.rs) --
        "torch.ops.aten.atan2.default" => impls_kokoro::map_atan2(node),

        // -- Tensor creation (op_map_impl_kokoro.rs) --
        "torch.ops.aten.ones.default" | "torch.ops.aten.ones_like.default" => {
            impls_kokoro::map_ones(node)
        }
        "torch.ops.aten.full.default" | "torch.ops.aten.full_like.default" => {
            impls_kokoro::map_full(node)
        }
        "torch.ops.aten.arange.default" | "torch.ops.aten.arange.start_step" => {
            impls_kokoro::map_arange(node)
        }

        // -- Identity / memory layout (op_map_impl_kokoro.rs) --
        "torch.ops.aten.contiguous.default"
        | "torch.ops.aten.clone.default"
        | "torch.ops.aten._copy.default" => impls_kokoro::map_identity(node),

        // -- dpdf model ops (op_map_impl_dpdf.rs) --
        // Upsampling 2D
        "torch.ops.aten.upsample_nearest2d.default" | "torch.ops.aten.upsample_nearest2d.vec" => {
            impls_dpdf::map_upsample_nearest2d(node)
        }
        "torch.ops.aten.upsample_bilinear2d.default" | "torch.ops.aten.upsample_bilinear2d.vec" => {
            impls_dpdf::map_upsample_bilinear2d(node)
        }

        // Normalization
        "torch.ops.aten.rms_norm.default" => impls_dpdf::map_rms_norm(node, ctx),

        // Activation ops (hardswish, hardsigmoid, mish, softplus, selu)
        // already matched above in the main activation section.

        // Mask ops
        "torch.ops.aten.triu.default" => impls_dpdf::map_triu(node),
        "torch.ops.aten.tril.default" => impls_dpdf::map_tril(node),

        // Selection / Indexing
        "torch.ops.aten.gather.default" => impls_dpdf::map_gather(node),
        "torch.ops.aten.argmax.default" => impls_dpdf::map_argmax(node),
        "torch.ops.aten.argmin.default" => impls_dpdf::map_argmin(node),

        // Vision
        "torch.ops.aten.pixel_shuffle.default" => impls_dpdf::map_pixel_shuffle(node),
        "torch.ops.aten.pixel_unshuffle.default" => impls_dpdf::map_pixel_unshuffle(node),

        // Repeat
        "torch.ops.aten.repeat.default" => impls_dpdf::map_repeat(node),

        // Wave 6: interpolate, scatter, reflection_pad2d, clamp_max
        "torch.ops.aten.interpolate.default" | "torch.ops.aten.interpolate.vec" => {
            impls_dpdf::map_interpolate(node)
        }
        "torch.ops.aten.scatter.src" => impls_dpdf::map_scatter(node),
        "torch.ops.aten.reflection_pad2d.default" => impls_dpdf::map_reflection_pad2d(node),
        "torch.ops.aten.clamp_max.default" => impls_dpdf::map_clamp_max(node),

        // -- Advanced indexing / shape ops (op_map_impl_dpdf.rs) --
        // Narrow (slice with size)
        "torch.ops.aten.narrow.default" | "torch.ops.aten.narrow.Tensor" => {
            impls_dpdf::map_narrow(node)
        }
        // TopK
        "torch.ops.aten.topk.default" => impls_dpdf::map_topk(node),
        // Sort
        "torch.ops.aten.sort.default" | "torch.ops.aten.sort.stable" => impls_dpdf::map_sort(node),
        // Scatter (value + src variants)
        "torch.ops.aten.scatter.value" => impls_dpdf::map_scatter_value(node),
        "torch.ops.aten.scatter_add.default" => impls_dpdf::map_scatter_add(node),
        // Roll
        "torch.ops.aten.roll.default" => impls_dpdf::map_roll(node),

        // -- Conv3d (standalone, op_map_impl_dpdf.rs) --
        "torch.ops.aten.conv3d.default" => impls_dpdf::map_conv3d(node, ctx),

        // -- ConvTranspose2d (standalone, op_map_impl_dpdf.rs) --
        "torch.ops.aten.conv_transpose2d.input" => impls_dpdf::map_conv_transpose2d(node, ctx),

        // -- Pooling (vision models, op_map_impl_dpdf.rs) --
        "torch.ops.aten.max_pool2d.default" => impls_dpdf::map_max_pool2d_plain(node),
        "torch.ops.aten.avg_pool1d.default" => impls_dpdf::map_avg_pool1d(node),
        "torch.ops.aten.adaptive_avg_pool1d.default" => impls_dpdf::map_adaptive_avg_pool1d(node),
        "torch.ops.aten.adaptive_max_pool2d.default" => impls_dpdf::map_adaptive_max_pool2d(node),

        // -- Grid sample (op_map_impl_dpdf.rs) --
        "torch.ops.aten.grid_sample.default" => impls_dpdf::map_grid_sample(node),

        // -- Masked fill (decomposed via try_expand_node, w11 direct fallback) --
        "torch.ops.aten.masked_fill.Scalar" | "torch.ops.aten.masked_fill_.Scalar" => {
            impls_w11::map_masked_fill_scalar(node)
        }

        // -- Index.Tensor (decomposed via try_expand_node) --
        "torch.ops.aten.index.Tensor" => impls_dpdf::map_index_tensor_fallback(node),

        // -- Meshgrid (decomposed via try_expand_node, w11 direct fallback) --
        "torch.ops.aten.meshgrid.default" | "torch.ops.aten.meshgrid.indexing" => {
            impls_w11::map_meshgrid(node)
        }

        // Stack (decomposed via try_expand_node, w11 direct fallback)
        "torch.ops.aten.stack.default" => impls_w11::map_stack(node),

        // Split/unbind fallback (w11 direct mappers + dpdf unbind fallback)
        "torch.ops.aten.split.Tensor" => impls_w11::map_split(node),
        "torch.ops.aten.split_with_sizes.default" => impls_w11::map_split_with_sizes(node),
        "torch.ops.aten.unbind.int" => impls_dpdf::map_unbind_fallback(node),

        // ====================================================================
        // Wave 7: Transformer / CNN / audio model ops
        // ====================================================================

        // -- Unary math --
        "torch.ops.aten.tan.default" => impls_xfmr::map_tan(node),
        "torch.ops.aten.ceil.default" => impls_xfmr::map_ceil(node),
        "torch.ops.aten.sign.default" | "torch.ops.aten.sgn.default" => impls_xfmr::map_sign(node),
        "torch.ops.aten.frac.default" => impls_xfmr::map_frac(node),
        "torch.ops.aten.log2.default" => impls_xfmr::map_log2(node),
        "torch.ops.aten.log10.default" => impls_xfmr::map_log10(node),
        "torch.ops.aten.exp2.default" => impls_xfmr::map_exp2(node),
        "torch.ops.aten.erf.default" => impls_xfmr::map_erf(node),

        // -- Activation --
        "torch.ops.aten.softsign.default" => impls_xfmr::map_softsign(node),
        "torch.ops.aten.prelu.default" => impls_xfmr::map_prelu(node, ctx),
        "torch.ops.aten.log_sigmoid.default" | "torch.ops.aten.log_sigmoid_forward.default" => {
            impls_xfmr::map_log_sigmoid(node)
        }
        "torch.ops.aten.glu.default" => impls_xfmr::map_glu(node),

        // -- Missing tensor comparisons --
        "torch.ops.aten.ge.Tensor" => impls_xfmr::map_ge_tensor(node),
        "torch.ops.aten.le.Tensor" => impls_xfmr::map_le_tensor(node),
        "torch.ops.aten.ne.Tensor" => impls_xfmr::map_ne_tensor(node),

        // -- Conv transpose 2D (standalone) --
        "torch.ops.aten.conv_transpose2d.default" => {
            impls_xfmr::map_conv_transpose2d(node, ctx)
        }

        // -- Matrix ops (decompose via try_expand_node when shape available) --
        "torch.ops.aten.addmm.default" => impls_xfmr::map_addmm_fallback(node),
        "torch.ops.aten.baddbmm.default" => impls_xfmr::map_baddbmm(node),

        // -- Index ops --
        "torch.ops.aten.index_add.default" | "torch.ops.aten.index_add_.default" => {
            impls_xfmr::map_index_add(node)
        }
        "torch.ops.aten.index_put.default" | "torch.ops.aten.index_put_.default" => {
            impls_xfmr::map_index_put(node)
        }
        "torch.ops.aten.unfold.default" => impls_xfmr::map_unfold(node),

        // -- Tensor creation --
        "torch.ops.aten.empty.memory_format"
        | "torch.ops.aten.empty.default"
        | "torch.ops.aten.empty_like.default" => impls_xfmr::map_empty(node),
        "torch.ops.aten.new_zeros.default" => impls_xfmr::map_new_zeros(node),
        "torch.ops.aten.new_ones.default" => impls_xfmr::map_new_ones(node),
        "torch.ops.aten.linspace.default" => impls_xfmr::map_linspace(node),
        "torch.ops.aten.scalar_tensor.default" => impls_xfmr::map_scalar_tensor(node),
        "torch.ops.aten.fill.Scalar" | "torch.ops.aten.fill_.Scalar" => impls_xfmr::map_fill(node),
        "torch.ops.aten.zero.default" | "torch.ops.aten.zero_.default" => {
            impls_xfmr::map_zero(node)
        }

        // -- Shape ops --
        "torch.ops.aten.t.default" => impls_xfmr::map_t(node),
        "torch.ops.aten.movedim.int" => impls_xfmr::map_movedim(node),

        // -- Power --
        "torch.ops.aten.pow.Tensor_Tensor" => impls_xfmr::map_pow_tensor_tensor(node),
        "torch.ops.aten.pow.Scalar" => impls_xfmr::map_pow_scalar(node),

        // -- Reductions (no dim) --
        "torch.ops.aten.sum.default" => impls_xfmr::map_sum_no_dim(node),
        "torch.ops.aten.mean.default" => impls_xfmr::map_mean_no_dim(node),
        "torch.ops.aten.prod.default" | "torch.ops.aten.prod.dim_int" => impls_xfmr::map_prod(node),
        "torch.ops.aten.var.default" | "torch.ops.aten.var.correction" => impls_xfmr::map_var(node),
        "torch.ops.aten.std.default" | "torch.ops.aten.std.correction" => impls_xfmr::map_std(node),
        "torch.ops.aten.any.default" | "torch.ops.aten.any.dim" => impls_xfmr::map_any(node),
        "torch.ops.aten.all.default" | "torch.ops.aten.all.dim" => impls_xfmr::map_all(node),

        // -- Boolean / logical --
        "torch.ops.aten.logical_not.default" => impls_xfmr::map_logical_not(node),
        "torch.ops.aten.logical_and.default" => impls_xfmr::map_logical_and(node),
        "torch.ops.aten.logical_or.default" => impls_xfmr::map_logical_or(node),

        // -- Miscellaneous --
        "torch.ops.aten.remainder.Scalar" | "torch.ops.aten.fmod.Scalar" => {
            impls_xfmr::map_remainder_scalar(node)
        }
        "torch.ops.aten.remainder.Tensor" | "torch.ops.aten.fmod.Tensor" => {
            impls_xfmr::map_remainder_tensor(node)
        }
        "torch.ops.aten.slice_scatter.default" => impls_xfmr::map_slice_scatter(node),
        "torch.ops.aten.copy.default" | "torch.ops.aten.copy_.default" => {
            impls_xfmr::map_copy(node)
        }
        // Wave 8: Vision and audio model ops
        "torch.ops.aten.upsample_bicubic2d.default" | "torch.ops.aten.upsample_bicubic2d.vec" => {
            impls_xfmr::map_upsample_bicubic2d(node)
        }
        "torch.ops.aten.replication_pad1d.default" => impls_xfmr::map_replication_pad1d(node),
        "torch.ops.aten.replication_pad2d.default" => impls_xfmr::map_replication_pad2d(node),
        "torch.ops.aten.channel_shuffle.default" => impls_xfmr::map_channel_shuffle(node),
        "torch.ops.aten.adaptive_max_pool1d.default" => impls_xfmr::map_adaptive_max_pool1d(node),
        "torch.ops.aten.nll_loss_forward.default" => impls_xfmr::map_nll_loss_forward(node),
        "torch.ops.aten.mse_loss.default" => impls_xfmr::map_mse_loss(node),
        "torch.ops.aten.l1_loss.default" => impls_xfmr::map_l1_loss(node),
        "torch.ops.aten.smooth_l1_loss.default" | "torch.ops.aten.huber_loss.default" => {
            impls_xfmr::map_smooth_l1_loss(node)
        }
        "torch.ops.aten.binary_cross_entropy.default" => impls_xfmr::map_binary_cross_entropy(node),

        // ====================================================================
        // Wave 9: commonly missing model patterns
        // ====================================================================

        // -- Unary math --
        "torch.ops.aten.trunc.default" => impls_w9::map_trunc(node),
        "torch.ops.aten.expm1.default" => impls_w9::map_expm1(node),
        "torch.ops.aten.log1p.default" => impls_w9::map_log1p(node),
        "torch.ops.aten.acos.default" => impls_w9::map_acos(node),
        "torch.ops.aten.asin.default" => impls_w9::map_asin(node),
        "torch.ops.aten.atan.default" => impls_w9::map_atan(node),
        "torch.ops.aten.cosh.default" => impls_w9::map_cosh(node),
        "torch.ops.aten.sinh.default" => impls_w9::map_sinh(node),

        // -- Value testing --
        "torch.ops.aten.isinf.default" => impls_w9::map_isinf(node),
        "torch.ops.aten.isnan.default" => impls_w9::map_isnan(node),
        "torch.ops.aten.isfinite.default" => impls_w9::map_isfinite(node),

        // -- Bitwise --
        "torch.ops.aten.bitwise_not.default" => impls_w9::map_bitwise_not(node),
        "torch.ops.aten.bitwise_and.Tensor" => impls_w9::map_bitwise_and(node),
        "torch.ops.aten.bitwise_or.Tensor" => impls_w9::map_bitwise_or(node),

        // -- Tensor-arg clamp variants --
        "torch.ops.aten.clamp_min.Tensor" => impls_w9::map_clamp_min_tensor(node),
        "torch.ops.aten.clamp_max.Tensor" => impls_w9::map_clamp_max_tensor(node),

        // -- Tensor creation --
        "torch.ops.aten.tile.default" => impls_w9::map_tile(node),
        "torch.ops.aten.arange.start" => impls_w9::map_arange_start(node),
        "torch.ops.aten.eye.default" | "torch.ops.aten.eye.m" => impls_w9::map_eye(node),

        // -- Expand variants --
        "torch.ops.aten.expand_as.default" => impls_w9::map_expand_as(node),
        "torch.ops.aten.broadcast_to.default" => impls_w9::map_broadcast_to(node),

        // -- Loss functions --
        "torch.ops.aten.binary_cross_entropy_with_logits.default" => {
            impls_w9::map_bce_with_logits(node)
        }
        "torch.ops.aten.cross_entropy_loss.default" => impls_w9::map_cross_entropy_loss(node),

        // -- Indexing --
        "torch.ops.aten.index_fill.int_Scalar" | "torch.ops.aten.index_fill_.int_Scalar" => {
            impls_w9::map_index_fill(node)
        }
        "torch.ops.aten.index_copy.default" | "torch.ops.aten.index_copy_.default" => {
            impls_w9::map_index_copy(node)
        }
        "torch.ops.aten.scatter_reduce.two" => impls_w9::map_scatter_reduce(node),

        // -- Repeat (scalar count) --
        "torch.ops.aten.repeat_interleave.self_int" => impls_w9::map_repeat_interleave_int(node),

        // -- Conditional / where variants --
        "torch.ops.aten.where.ScalarOther" => impls_w9::map_where_scalar_other(node),
        "torch.ops.aten.where.ScalarSelf" => impls_w9::map_where_scalar_self(node),
        "torch.ops.aten.masked_scatter.default" | "torch.ops.aten.masked_scatter_.default" => {
            impls_w9::map_masked_scatter(node)
        }

        // ====================================================================
        // Wave 10: additional transformer and training ops
        // ====================================================================

        // -- Diagonal extraction --
        "torch.ops.aten.diagonal.default" => impls_w10::map_diagonal(node),

        // -- 90-degree rotation --
        "torch.ops.aten.rot90.default" => impls_w10::map_rot90(node),

        // -- Loss functions --
        "torch.ops.aten.nll_loss.default" => impls_w10::map_nll_loss(node),
        "torch.ops.aten.kl_div.default" => impls_w10::map_kl_div(node),

        // -- Masked fill with tensor value --
        "torch.ops.aten.masked_fill.Tensor" | "torch.ops.aten.masked_fill_.Tensor" => {
            impls_w10::map_masked_fill_tensor(node)
        }

        // -- In-place scatter --
        "torch.ops.aten.scatter_.src" | "torch.ops.aten.scatter_.reduce" => {
            impls_w10::map_scatter_inplace(node)
        }

        // ====================================================================
        // Wave 11: spatial transformer, grid, and shape ops
        // ====================================================================

        // -- Affine grid generator (Spatial Transformer Networks) --
        "torch.ops.aten.affine_grid_generator.default" => {
            impls_w11::map_affine_grid_generator(node)
        }

        // -- Triu/Tril in-place overloads --
        "torch.ops.aten.triu_.default" => impls_w11::map_triu_inplace(node),
        "torch.ops.aten.tril_.default" => impls_w11::map_tril_inplace(node),

        // -- Arange start_stop overload --
        "torch.ops.aten.arange.start_stop" => impls_w11::map_arange_start_stop(node),

        // -- Linspace out overload --
        "torch.ops.aten.linspace.out" => impls_w11::map_linspace_out(node),

        // -- Chunk direct mapper (fallback when shape unavailable) --
        "torch.ops.aten.chunk.default" => impls_w11::map_chunk(node),

        // ====================================================================
        // Wave 12: normalization, embedding, and loss overloads
        // ====================================================================

        // -- Batch norm: training variant (_native_batch_norm_legit) --
        "torch.ops.aten._native_batch_norm_legit.default" => {
            impls_w12::map_native_batch_norm_legit(node, ctx)
        }
        "torch.ops.aten._native_batch_norm_legit.no_stats" => {
            impls_w12::map_native_batch_norm_legit_no_stats(node)
        }
        "torch.ops.aten.cudnn_batch_norm.default" => impls_w12::map_cudnn_batch_norm(node, ctx),

        // -- Layer norm: optional affine overload --
        "torch.ops.aten.layer_norm.no_affine" => {
            impls_w12::map_layer_norm_optional_affine(node, ctx)
        }

        // -- Group norm: optional affine overload --
        "torch.ops.aten.group_norm.no_affine" => {
            impls_w12::map_group_norm_optional_affine(node, ctx)
        }

        // -- Instance norm: affine variant --
        "torch.ops.aten.instance_norm.affine" => impls_w12::map_instance_norm_affine(node, ctx),

        // -- Embedding: padding_idx overload --
        "torch.ops.aten.embedding.padding_idx" => impls_w12::map_embedding_padding_idx(node, ctx),

        // -- Embedding bag --
        "torch.ops.aten._embedding_bag.default" | "torch.ops.aten.embedding_bag.default" => {
            impls_w12::map_embedding_bag(node, ctx)
        }

        // -- Cross-entropy loss: label_smoothing overload --
        "torch.ops.aten.cross_entropy_loss.label_smoothing" => {
            impls_w12::map_cross_entropy_loss_full(node)
        }

        // -- NLL loss: N-dim and 2D variants --
        "torch.ops.aten.nll_loss_nd.default" => impls_w12::map_nll_loss_nd(node),
        "torch.ops.aten.nll_loss2d_forward.default" => impls_w12::map_nll_loss2d_forward(node),

        // -- Binary cross-entropy: weight variant --
        "torch.ops.aten.binary_cross_entropy.weight" => {
            impls_w12::map_binary_cross_entropy_weighted(node)
        }

        // -- Loss backward variants (training graph exports) --
        "torch.ops.aten.mse_loss_backward.default" => impls_w12::map_mse_loss_backward(node),
        "torch.ops.aten.l1_loss_backward.default" => impls_w12::map_l1_loss_backward(node),
        "torch.ops.aten.smooth_l1_loss_backward.default" => {
            impls_w12::map_smooth_l1_loss_backward(node)
        }
        "torch.ops.aten.kl_div_backward.default" => impls_w12::map_kl_div_backward(node),

        // ====================================================================
        // Wave 13: advanced tensor manipulation and control flow ops
        // ====================================================================

        // -- Index put: hacked_twin overloads --
        "torch.ops.aten.index_put.hacked_twin" | "torch.ops.aten.index_put_.hacked_twin" => {
            impls_w13::map_index_put_hacked_twin(node)
        }

        // -- Index put: accumulate overload --
        "torch.ops.aten.index_put.accumulate" | "torch.ops.aten.index_put_.accumulate" => {
            impls_w13::map_index_put_accumulate(node)
        }

        // -- Scatter: value_reduce variant --
        "torch.ops.aten.scatter_.value_reduce" => impls_w13::map_scatter_value_reduce(node),

        // -- Scatter add: in-place variant --
        "torch.ops.aten.scatter_add_.default" => impls_w13::map_scatter_add_inplace(node),

        // -- Gather: out variant --
        "torch.ops.aten.gather.out" => impls_w13::map_gather_out(node),

        // -- Index select: out variant --
        "torch.ops.aten.index_select.out" => impls_w13::map_index_select_out(node),

        // -- Masked fill: Tensor_Scalar variant --
        "torch.ops.aten.masked_fill.Tensor_Scalar" => {
            impls_w13::map_masked_fill_tensor_scalar(node)
        }

        // -- Masked select --
        "torch.ops.aten.masked_select.default" => impls_w13::map_masked_select(node),
        "torch.ops.aten.masked_select.out" => impls_w13::map_masked_select_out(node),

        // -- Nonzero --
        "torch.ops.aten.nonzero.default" => impls_w13::map_nonzero(node),
        "torch.ops.aten.nonzero.out" => impls_w13::map_nonzero_out(node),

        // -- Topk: values variant --
        "torch.ops.aten.topk.values" => impls_w13::map_topk_values(node),

        // -- Sort: values variants --
        "torch.ops.aten.sort.values" | "torch.ops.aten.sort.values_stable" => {
            impls_w13::map_sort_values(node)
        }

        // -- Unique --
        "torch.ops.aten._unique2.default" => impls_w13::map_unique2(node),
        "torch.ops.aten.unique_dim.default" => impls_w13::map_unique_dim(node),

        // -- Unique consecutive --
        "torch.ops.aten.unique_consecutive.default" => impls_w13::map_unique_consecutive(node),

        // ====================================================================
        // Wave 14: common missing PyTorch ops
        // ====================================================================

        // -- Lerp (linear interpolation) --
        "torch.ops.aten.lerp.Scalar" => impls_w14::map_lerp_scalar(node),
        "torch.ops.aten.lerp.Tensor" => impls_w14::map_lerp_tensor(node),

        // -- Fused mul-add / div --
        "torch.ops.aten.addcmul.default" | "torch.ops.aten.addcmul_.default" => {
            impls_w14::map_addcmul(node)
        }
        "torch.ops.aten.addcdiv.default" | "torch.ops.aten.addcdiv_.default" => {
            impls_w14::map_addcdiv(node)
        }

        // -- Norm --
        "torch.ops.aten.linalg_vector_norm.default" => impls_w14::map_linalg_vector_norm(node),

        // -- Pairwise distance --
        "torch.ops.aten.cdist.default" => impls_w14::map_cdist(node),

        // -- Sampling --
        "torch.ops.aten.multinomial.default" => impls_w14::map_multinomial(node),

        // -- Search --
        "torch.ops.aten.searchsorted.Tensor" => impls_w14::map_searchsorted(node),
        "torch.ops.aten.bucketize.Tensor" => impls_w14::map_bucketize(node),

        // -- Counting --
        "torch.ops.aten.count_nonzero.default" => impls_w14::map_count_nonzero(node),
        "torch.ops.aten.count_nonzero.dim_IntList" => impls_w14::map_count_nonzero_dims(node),

        // -- Cumulative --
        "torch.ops.aten.cumprod.default" | "torch.ops.aten.cumprod.int" => {
            impls_w14::map_cumprod(node)
        }
        "torch.ops.aten.cummax.default" => impls_w14::map_cummax(node),
        "torch.ops.aten.cummin.default" => impls_w14::map_cummin(node),

        // -- One-hot encoding --
        "torch.ops.aten.one_hot.default" => impls_w14::map_one_hot(node),

        // -- Threshold activation --
        "torch.ops.aten.threshold.default" | "torch.ops.aten.threshold_.default" => {
            impls_w14::map_threshold(node)
        }

        // ====================================================================
        // Wave 15: matrix ops, sampling, creation, strided views
        // ====================================================================

        // -- Clamp with tensor bounds --
        "torch.ops.aten.clamp.Tensor" => impls_w15::map_clamp_tensor(node),

        // -- Norm (Lp norm along dims) --
        "torch.ops.aten.norm.ScalarOpt_dim" | "torch.ops.aten.norm.Scalar" => {
            impls_w15::map_norm(node)
        }

        // -- Einsum (Einstein summation) --
        "torch.ops.aten.einsum.default" => impls_w15::map_einsum(node),

        // -- Strided view --
        "torch.ops.aten.as_strided.default" => impls_w15::map_as_strided(node),

        // -- Matrix-vector multiply-add --
        "torch.ops.aten.addmv.default" => impls_w15::map_addmv(node),

        // -- Additive outer product --
        "torch.ops.aten.addr.default" => impls_w15::map_addr(node),

        // -- Outer product --
        "torch.ops.aten.outer.default" => impls_w15::map_outer(node),

        // -- Bernoulli sampling --
        "torch.ops.aten.bernoulli.default" => impls_w15::map_bernoulli(node),
        "torch.ops.aten.bernoulli_.float" => impls_w15::map_bernoulli_float(node),

        // -- Random normal creation --
        "torch.ops.aten.randn.default" => impls_w15::map_randn(node),

        // -- Cross product --
        "torch.ops.aten.cross.default" => impls_w15::map_cross(node),

        // ====================================================================
        // Wave 16: in-place activations, native norms, GRU, complex/FFT,
        //          dropout variants
        // ====================================================================

        // -- In-place activations (semantically identical to out-of-place) --
        "torch.ops.aten.relu_.default" => impls_w16::map_relu_inplace(node),
        "torch.ops.aten.sigmoid_.default" => impls_w16::map_sigmoid_inplace(node),
        "torch.ops.aten.tanh_.default" => impls_w16::map_tanh_inplace(node),
        "torch.ops.aten.silu_.default" => impls_w16::map_silu_inplace(node),
        "torch.ops.aten.gelu_.default" => impls_w16::map_gelu_inplace(node),

        // -- Native normalization (torch.export internal decomposition targets) --
        "torch.ops.aten.native_layer_norm.default" => impls_w16::map_native_layer_norm(node, ctx),
        "torch.ops.aten.native_group_norm.default" => impls_w16::map_native_group_norm(node, ctx),

        // -- GRU recurrent --
        "torch.ops.aten.gru.input" => impls_w16::map_gru(node),

        // -- Complex tensor views --
        "torch.ops.aten.view_as_real.default" => impls_w16::map_view_as_real(node),
        "torch.ops.aten.view_as_complex.default" => impls_w16::map_view_as_complex(node),

        // -- FFT operations (audio/signal processing) --
        "torch.ops.aten.fft_rfft.default" => impls_w16::map_fft_rfft(node),
        "torch.ops.aten.fft_irfft.default" => impls_w16::map_fft_irfft(node),

        // -- Dropout variants (all identity at inference) --
        "torch.ops.aten.feature_dropout.default" => impls_w16::map_feature_dropout(node),
        "torch.ops.aten.alpha_dropout.default" => impls_w16::map_alpha_dropout(node),

        _ => Err(ImportError::UnsupportedOp {
            target: target.to_string(),
        }),
    }
}

/// Check if a node requires multi-node expansion (e.g., bidirectional LSTM).
///
/// Returns `Some(expanded_nodes)` if the node decomposes into multiple
/// TraceNodes, or `None` if the standard single-op path should be used.
pub(crate) fn try_expand_node(
    node: &Node,
    ctx: &OpMapContext<'_>,
    output_name: &str,
    input_shape: &[usize],
) -> Result<Option<Vec<ExpandedNode>>, ImportError> {
    if node.target == "torch.ops.aten.lstm.input" && optional_bool(node, "bidirectional", false) {
        return impls_ext::expand_bilstm(node, ctx, output_name, input_shape).map(Some);
    }
    // chunk: split into N Narrow ops using the multi-output tensor names.
    if node.target == "torch.ops.aten.chunk.default" && !input_shape.is_empty() {
        return expand::expand_chunk(node, output_name, input_shape).map(Some);
    }
    // Scalar binary ops: create Constant node + binary op node.
    // Also handles `.Tensor` variants where "other" is actually a scalar (common in torch.export).
    if let Some(op) = match node.target.as_str() {
        "torch.ops.aten.add.Scalar"
        | "torch.ops.aten.add_.Scalar"
        | "torch.ops.aten.sub.Scalar"
        | "torch.ops.aten.sub_.Scalar"
        | "torch.ops.aten.mul.Scalar"
        | "torch.ops.aten.mul_.Scalar"
        | "torch.ops.aten.div.Scalar"
        | "torch.ops.aten.div_.Scalar" => Some(match node.target.as_str() {
            t if t.contains("sub") => TraceOp::Sub,
            t if t.contains("mul") => TraceOp::Mul,
            t if t.contains("div") => TraceOp::Div,
            _ => TraceOp::Add,
        }),
        "torch.ops.aten.add.Tensor"
        | "torch.ops.aten.add_.Tensor"
        | "torch.ops.aten.sub.Tensor"
        | "torch.ops.aten.sub_.Tensor"
        | "torch.ops.aten.mul.Tensor"
        | "torch.ops.aten.mul_.Tensor"
        | "torch.ops.aten.div.Tensor"
        | "torch.ops.aten.div_.Tensor" => {
            // Check if "other" is a scalar rather than a tensor.
            let other_is_scalar = get_arg(node, "other")
                .ok()
                .is_some_and(|a| a.as_tensor_name().is_none() && !a.is_none());
            if other_is_scalar {
                Some(match node.target.as_str() {
                    t if t.contains("sub") => TraceOp::Sub,
                    t if t.contains("mul") => TraceOp::Mul,
                    t if t.contains("div") => TraceOp::Div,
                    _ => TraceOp::Add,
                })
            } else {
                None
            }
        }
        _ => None,
    } {
        return expand::expand_scalar_binary(node, output_name, input_shape, op).map(Some);
    }
    // flatten.using_ints: decompose to Reshape with computed target shape.
    if node.target == "torch.ops.aten.flatten.using_ints" && !input_shape.is_empty() {
        return expand::expand_flatten(node, output_name, input_shape).map(Some);
    }
    // squeeze.default (squeeze all size-1 dims): decompose to Reshape.
    if node.target == "torch.ops.aten.squeeze.default" && !input_shape.is_empty() {
        return expand::expand_squeeze_default(node, output_name, input_shape).map(Some);
    }
    // select.int (select single index along dim): decompose to Narrow + Reshape.
    if node.target == "torch.ops.aten.select.int" && !input_shape.is_empty() {
        return expand::expand_select_int(node, output_name, input_shape).map(Some);
    }
    // Multi-axis reductions: decompose to sequential single-dim reduces.
    if let Some(make_op) = match node.target.as_str() {
        "torch.ops.aten.sum.dim_IntList" => {
            Some(expand::make_reduce_sum as fn(usize, bool) -> TraceOp)
        }
        "torch.ops.aten.mean.dim" => Some(expand::make_reduce_mean as fn(usize, bool) -> TraceOp),
        "torch.ops.aten.amax.default" => {
            Some(expand::make_reduce_max as fn(usize, bool) -> TraceOp)
        }
        "torch.ops.aten.amin.default" => {
            Some(expand::make_reduce_min as fn(usize, bool) -> TraceOp)
        }
        _ => None,
    } {
        let dims = require_ints(node, "dim")?;
        if dims.len() > 1 && !input_shape.is_empty() {
            let keepdim = optional_bool(node, "keepdim", false);
            return expand::expand_multi_axis_reduce(
                node,
                output_name,
                input_shape,
                make_op,
                &dims,
                keepdim,
            )
            .map(Some);
        }
    }
    // split.Tensor / split_with_sizes: decompose into N Narrow ops.
    if (node.target == "torch.ops.aten.split.Tensor"
        || node.target == "torch.ops.aten.split_with_sizes.default")
        && !input_shape.is_empty()
    {
        return expand::expand_split(node, output_name, input_shape).map(Some);
    }
    // unbind.int: decompose into N (Narrow + Reshape) pairs.
    if node.target == "torch.ops.aten.unbind.int" && !input_shape.is_empty() {
        return expand::expand_unbind(node, output_name, input_shape).map(Some);
    }
    // stack.default: decompose to N Unsqueeze ops + 1 Cat op.
    if node.target == "torch.ops.aten.stack.default" && !input_shape.is_empty() {
        return expand::expand_stack(node, output_name, input_shape).map(Some);
    }
    // masked_fill.Scalar / masked_fill_.Scalar: decompose to Constant + WhereCond.
    if (node.target == "torch.ops.aten.masked_fill.Scalar"
        || node.target == "torch.ops.aten.masked_fill_.Scalar")
        && !input_shape.is_empty()
    {
        return expand::expand_masked_fill(node, output_name, input_shape).map(Some);
    }
    // index.Tensor: decompose to IndexSelect for single-index case.
    if node.target == "torch.ops.aten.index.Tensor" && !input_shape.is_empty() {
        return expand::expand_index_tensor(node, output_name, input_shape).map(Some);
    }
    // meshgrid: decompose to Reshape + Expand for each input.
    if (node.target == "torch.ops.aten.meshgrid.default"
        || node.target == "torch.ops.aten.meshgrid.indexing")
        && !input_shape.is_empty()
    {
        return expand::expand_meshgrid(node, output_name, input_shape).map(Some);
    }
    // addmm: decompose to MatMul + Add.
    if node.target == "torch.ops.aten.addmm.default" && !input_shape.is_empty() {
        return expand::expand_addmm(node, output_name, input_shape).map(Some);
    }
    // baddbmm: decompose to MatMul + scale + Add.
    if node.target == "torch.ops.aten.baddbmm.default" && !input_shape.is_empty() {
        return expand::expand_baddbmm(node, output_name, input_shape).map(Some);
    }
    Ok(None)
}

#[cfg(test)]
#[path = "op_map_tests.rs"]
mod tests;
