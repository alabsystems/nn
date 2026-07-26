// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Neural network layer abstractions for [`DynTensor`].
//!
//! Provides candle-nn compatible layer types and operations (44 Module impls):
//! - [`Module`] / [`ModuleT`] traits for forward pass
//! - Core layers: [`Linear`], [`Conv1d`], [`Conv2d`], [`Conv3d`],
//!   [`ConvTranspose1d`], [`ConvTranspose2d`], [`Embedding`], [`WeightNormConv1d`]
//! - Normalization: [`LayerNorm`], [`GroupNorm`], [`BatchNorm`], [`BatchNorm2d`],
//!   [`RmsNorm`], [`InstanceNorm`], [`AdaIn`]
//! - Pooling: [`MaxPool2d`], [`AvgPool2d`], [`AdaptiveAvgPool2d`]
//! - Spatial: [`Upsample2d`], [`PixelShuffle`], [`PixelUnshuffle`]
//! - Recurrent: [`Lstm`], [`BiLstm`]
//! - Attention: [`MultiHeadAttention`], [`JointAttention`], [`RotaryEmbedding`],
//!   [`KvCache`], [`causal_mask`], [`alibi_bias`]
//! - Vision: [`VitEncoder`], [`PatchEmbedding`], [`VitEncoderBlock`],
//!   [`SqueezeExcitation`], [`MBConv`]
//! - Detection: [`ConvBnAct`], [`Sppf`], [`C2f`], [`Bottleneck`], [`Detection`],
//!   [`nms_filter`]
//! - Advanced: [`GatedDeltaNet`], [`SwiGlu`], [`MoeLayer`], [`DiTBlock`],
//!   [`AdaLnZero`], [`Rvq`], [`Dropout`]
//! - Quantized: [`QLinear`], [`BlockQ4K`]
//! - Containers: [`Sequential`], [`Activation`]
//! - Ops: [`softmax`], [`log_softmax`], [`sigmoid`], [`rope`]
//! - Generation: [`generate`] (greedy/top-k autoregressive decoding)
//! - Weight loading: [`VarBuilder`] with `linear()`, `conv1d()`, `embedding()`, etc.
//!
//! These enable find-and-replace migration from candle-nn to nn.
//!
//! # GPU dispatch coverage
//!
//! When a layer's input tensors are on GPU (Metal), operations dispatch natively
//! unless noted otherwise. Layers fall into three tiers:
//!
//! **GPU-native** (all ops dispatch on device):
//! [`Linear`], [`Conv1d`], [`Conv2d`], [`ConvTranspose1d`], [`Embedding`],
//! [`LayerNorm`], [`GroupNorm`], [`RmsNorm`], [`Lstm`], [`SwiGlu`],
//! [`MoeLayer`] (expert forward + weighted scatter-add stay on device;
//! only O(N*k) routing indices transfer to CPU for grouping),
//! [`Upsample2d`] (nearest mode), [`Activation`], [`Dropout`] (no-op in eval),
//! [`WeightNormConv1d`] (forward delegates to Conv1d),
//! [`PatchEmbedding`] (Conv2d + reshape + transpose),
//! [`VqCodebook`] (Embedding-based lookup).
//!
//! **CPU round-trip** (GPU→CPU→GPU for some internal ops):
//! [`BatchNorm`], [`InstanceNorm`], [`AdaIn`] (decomposed norm ops),
//! [`MoeRouter`] (softmax + topk are GPU-native, but topk falls back to CPU
//! when k>64), [`MaxPool2d`], [`AvgPool2d`], [`AdaptiveAvgPool2d`],
//! [`ConvTranspose2d`], [`Upsample2d`] (bilinear mode),
//! [`PixelShuffle`], [`PixelUnshuffle`],
//! [`Res2NetBlock`] (uses BatchNorm internally).
//!
//! **CPU-only** (no GPU dispatch, always on CPU):
//! [`Rvq`], [`BiLstm`], [`SqueezeExcitation`], [`MBConv`], [`DiTBlock`],
//! selection ops ([`DynTensor::gather`], [`DynTensor::scatter_add`] are
//! GPU-native, but model-level pipelines may still force CPU at boundaries).

use crate::dyn_tensor::DynTensor;
use crate::Result;

// -- Infrastructure submodules ------------------------------------------------
mod nan_check;
pub use nan_check::{check_output_finite, nan_check_policy, with_nan_check_policy, NanCheckPolicy};

pub(crate) mod validation;
pub(crate) use validation::{validate_divisible, validate_eps, validate_heads, CpuRoundTrip};

pub(crate) mod trace_helper;
pub(crate) use trace_helper::traced_forward;

// -- Module trait -------------------------------------------------------------

/// Layer abstraction matching candle's `Module` trait.
///
/// Any struct implementing `Module` can be used with `DynTensor::apply()`.
pub trait Module {
    /// Forward pass: compute output from input tensor.
    fn forward(&self, x: &DynTensor) -> Result<DynTensor>;
}

// Re-export ModuleT so `use nn::layers::{Module, ModuleT}` works.
pub use crate::module::ModuleT;

/// Blanket impl: closures are modules (enables `Sequential::add_fn`).
impl<T: Fn(&DynTensor) -> Result<DynTensor>> Module for T {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        self(x)
    }
}

impl DynTensor {
    /// Apply a module to this tensor (syntactic sugar for `m.forward(self)`).
    pub fn apply<M: Module>(&self, m: &M) -> Result<Self> {
        m.forward(self)
    }
}

// -- Core layers (Linear, LayerNorm) -------------------------------------------
mod core_layers;
pub use core_layers::{LayerNorm, Linear};
// -- GroupNorm ----------------------------------------------------------------
mod group_norm;
pub use group_norm::GroupNorm;
mod embedding;
pub use embedding::Embedding;

// -- BatchNorm ----------------------------------------------------------------
mod batch_norm;
pub use batch_norm::{BatchNorm, BatchNorm2d, BatchNormConfig};

// -- Conv1d / Conv2d / ConvTranspose1d / ConvTranspose2d ----------------------
mod conv;
pub use conv::{
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Conv3d, Conv3dConfig, ConvTranspose1d,
    ConvTranspose1dConfig, ConvTranspose2d, ConvTranspose2dConfig, WeightNormConv1d,
};

// -- Pooling layers (MaxPool1d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d) -------
mod pool;
pub use pool::{AdaptiveAvgPool2d, AvgPool2d, MaxPool1d, MaxPool2d, Pool1dConfig, Pool2dConfig};

// -- Vision layers (ViT, Upsample2d, PixelShuffle, SE, MBConv) (nn/vision/) ---
pub mod vision;
pub use vision::{
    AttentiveStatisticsPooling, BasicBlock, Bottleneck, C2f, ConvBnAct, DeepStackFusion,
    DetectHead, Detection, DetrDecoder, DetrDecoderLayer, DetrOutput, ImagePreprocessor,
    ImageProcessor, MBConv, MBConvConfig, PanNeck, PatchEmbedding, PixelShuffle, PixelUnshuffle,
    PoolingStrategy, Qwen2VLVitConfig, Qwen3VLVitConfig, Res2NetBlock, ResNet18, ScaleOutput,
    SigLip2Config, SigLip2VisionEncoder, Sppf, SqueezeExcitation, SqueezeExcitation1d, Upsample2d,
    Upsample2dToSize, UpsampleMode, VitConfig, VitEncoder, VitEncoderBlock,
};
// -- Activation enum ----------------------------------------------------------
mod activation;
pub use activation::Activation;

// -- RmsNorm ------------------------------------------------------------------
mod rms_norm;
pub use rms_norm::RmsNorm;

// -- Sequential container -----------------------------------------------------
mod sequential;
pub use sequential::Sequential;

// -- Attention mechanisms + positional encodings (nn/attention/) ---------------
pub mod attention;
pub use attention::{
    alibi_bias, alibi_bias_scaled, alibi_slopes, causal_mask, causal_mask_dtype,
    causal_mask_with_offset, repeat_kv, rope, sdpa, sdpa_causal, sinusoidal_2d,
    sliding_window_mask, window_partition, window_unpartition, AttentionMode, DeformableAttention,
    DeformableAttentionConfig, HalfRotaryEmbedding, InterleavedMRoPE, InterleavedMRoPEConfig,
    JointAttention, MultiHeadAttention, MultimodalRoPE, RotaryEmbedding, RotaryEmbedding2d,
    SlidingWindowAttention, WindowAttentionConfig, WindowMultiHeadAttention, YarnScaling,
};
/// candle_nn::rotary_emb compatibility — enables `nn::layers::rotary_emb::rope`.
///
/// Prefer `nn::layers::RotaryEmbedding` and `nn::layers::rope` directly.
/// This submodule exists only for candle_nn import path compatibility.
pub mod rotary_emb {
    pub use super::attention::{rope, HalfRotaryEmbedding, RotaryEmbedding, YarnScaling};
}
// -- Softmax / log-softmax with dim parameter ---------------------------------
mod ops;
pub use ops::{log_softmax, sigmoid, softmax};
// Re-export softmax_last_dim from dyn_tensor_ops for candle_nn::ops compatibility.
#[allow(deprecated)]
pub use crate::dyn_tensor::softmax_last_dim;
// -- Generation (KV cache, autoregressive, beam search, CTC) (nn/generation/) -
pub mod generation;
/// `layers::autoregressive` path for internal consumers.
pub use generation::autoregressive;
/// `layers::kv_cache` path for candle compatibility (`candle_nn::kv_cache::KvCache`).
pub use generation::kv_cache;
pub use generation::{
    beam_search, ctc_beam_decode, ctc_greedy_decode, decode_generate, decode_step, generate,
    prefill, BeamHypothesis, BeamSearchConfig, BeamSearchOutput, CtcBeamHypothesis, CtcConfig,
    DecodeContext, GenerationConfig, GenerationOutput, KvCache, KvCacheBackend, KvCacheLayer,
    KvCacheLayerBackend, MtpHead, MtpHeadConfig, PreallocKvCache, PreallocKvCacheLayer,
};
// -- LSTM + InstanceNorm + AdaIN -----------------------------------------------
mod lstm;
pub use lstm::{BiLstm, Lstm, LstmCell, LstmState};
mod instance_norm;
pub use instance_norm::{InstanceNorm, InstanceNormPrecision};
mod adain;
pub use adain::AdaIn;
// -- Adaptive Layer Normalization (AdaLN-Zero, DiT models) --------------------
mod adaln;
pub use adaln::{apply_adaln_modulation, AdaLnParams, AdaLnZero, AdaLnZeroDual, LowRankAdaLn};
// -- Gated DeltaNet (Qwen3.5 linear attention) --------------------------------
mod gated_delta_net;
pub use gated_delta_net::{GatedDeltaNet, GatedDeltaNetState};
// -- SwiGLU Feed-Forward Network (Shazeer 2020) -------------------------------
mod swiglu;
pub use swiglu::SwiGlu;
// -- Mixture-of-Experts (MoE routing + dispatch) ------------------------------
mod moe;
pub use moe::{MoeRouter, MoeRoutingOutput, SwiGluExpert};
mod moe_dispatch;
pub use moe_dispatch::{MoeDispatch, MoeDispatchConfig, MoeDispatchOutput};
mod moe_experts;
mod moe_layer;
pub use moe_layer::{ExpertFFN, ExpertMlp, MoeLayer, MoeLayerConfig, MoeOutput};
mod moe_mlp_layer;
pub use moe_mlp_layer::{MoeMlpConfig, MoeMlpLayer};
// -- Dropout (inference-mode no-op) -------------------------------------------
mod dropout;
pub use dropout::Dropout;
// -- DiT Block composites (Diffusion Transformer) -----------------------------
mod dit_block;
pub use dit_block::{DiTBlock, DiTBlockDual};
// -- Quantized layers (QLinear, Q4K, RVQ) (nn/quantized/) ---------------------
pub mod quantized;
pub use quantized::{
    dequantize_per_channel, max_quantization_error, quantize_per_channel, quantized_matmul,
    weight_dequantize, weight_quantize_per_group, BlockQ4K, GgmlDType, Int8Linear, Int8Mode,
    Int8QuantParams, QLinear, QuantDtype, QuantizationConfig, QuantizedTensor, QuantizedWeight,
    Rvq, VqCodebook,
};
// -- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning -----------
#[cfg(feature = "training")]
pub mod lora;
#[cfg(feature = "training")]
pub use lora::{LoraConfig, LoraLinear};
// -- VarBuilder load() constructors for nn layers -----------------------------
mod var_builder_loaders;
pub use var_builder_loaders::{
    batch_norm, conv1d, conv1d_no_bias, conv2d, conv2d_no_bias, conv3d, conv3d_no_bias,
    conv_transpose1d, conv_transpose1d_no_bias, conv_transpose2d, conv_transpose2d_no_bias,
    embedding, group_norm, layer_norm, linear, linear_no_bias, lstm, rms_norm, LayerNormConfig,
};
#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "layer_coverage_tests.rs"]
mod layer_coverage_tests;

#[cfg(test)]
#[path = "nn_layer_tests.rs"]
mod nn_layer_tests;

#[cfg(test)]
#[path = "integration_tests.rs"]
mod integration_tests;

#[cfg(test)]
#[path = "advanced_layers_tests.rs"]
mod advanced_layers_tests;

#[cfg(kani)]
#[path = "kani_moe_layer.rs"]
mod kani_moe_layer;

#[cfg(kani)]
mod kani_moe_layer_issue_3730;

#[cfg(kani)]
#[path = "kani_moe_dispatch.rs"]
mod kani_moe_dispatch;

#[cfg(kani)]
mod kani_moe_dispatch_issue_3730;

#[cfg(kani)]
#[path = "kani_moe_mlp_layer.rs"]
mod kani_moe_mlp_layer;

#[cfg(kani)]
#[path = "kani_embedding.rs"]
mod kani_embedding;

#[cfg(kani)]
#[path = "kani_batch_norm.rs"]
mod kani_batch_norm;

#[cfg(kani)]
#[path = "kani_instance_norm.rs"]
mod kani_instance_norm;

#[cfg(kani)]
#[path = "kani_adain.rs"]
mod kani_adain;

#[cfg(kani)]
#[path = "kani_dropout.rs"]
mod kani_dropout;

#[cfg(kani)]
#[path = "kani_sequential.rs"]
mod kani_sequential;

#[cfg(kani)]
#[path = "kani_moe_layer_advanced.rs"]
mod kani_moe_layer_advanced;

#[cfg(kani)]
#[path = "kani_moe_dispatch_advanced.rs"]
mod kani_moe_dispatch_advanced;

#[cfg(kani)]
#[path = "kani_bilstm_dit_moe_proofs.rs"]
mod kani_bilstm_dit_moe_proofs;
#[cfg(kani)]
#[path = "kani_conv_proofs.rs"]
mod kani_conv_proofs;
#[cfg(kani)]
#[path = "kani_lstm_gpu_varbuilder_proofs.rs"]
mod kani_lstm_gpu_varbuilder_proofs;
#[cfg(kani)]
#[path = "kani_norm_activation_proofs.rs"]
mod kani_norm_activation_proofs;

#[cfg(kani)]
#[path = "kani_layers_dyntensor_proofs.rs"]
mod kani_layers_dyntensor_proofs;

#[cfg(kani)]
#[path = "kani_embedding_weight_proofs.rs"]
mod kani_embedding_weight_proofs;
#[cfg(kani)]
#[path = "kani_linear_weight_proofs.rs"]
mod kani_linear_weight_proofs;

#[cfg(kani)]
#[path = "kani_conv_shape_proofs.rs"]
mod kani_conv_shape_proofs;
#[cfg(kani)]
#[path = "kani_group_norm_proofs.rs"]
mod kani_group_norm_proofs;
#[cfg(kani)]
#[path = "kani_lstm_gate_proofs.rs"]
mod kani_lstm_gate_proofs;

#[cfg(kani)]
#[path = "kani_rms_norm_proofs.rs"]
mod kani_rms_norm_proofs;

#[cfg(kani)]
#[path = "kani_var_builder_proofs.rs"]
mod kani_var_builder_proofs;

#[cfg(kani)]
#[path = "kani_activation_enum_proofs.rs"]
mod kani_activation_enum_proofs;
#[cfg(kani)]
#[path = "kani_conv2d_pool_matmul_proofs.rs"]
mod kani_conv2d_pool_matmul_proofs;
#[cfg(kani)]
#[path = "kani_conv_transpose_proofs.rs"]
mod kani_conv_transpose_proofs;
#[cfg(kani)]
#[path = "kani_dpdf_batch_norm_proofs.rs"]
mod kani_dpdf_batch_norm_proofs;
#[cfg(kani)]
#[path = "kani_layer_norm_proofs.rs"]
mod kani_layer_norm_proofs;
#[cfg(kani)]
#[path = "kani_validation_proofs.rs"]
mod kani_validation_proofs;
#[cfg(kani)]
#[path = "kani_weight_norm_proofs.rs"]
mod kani_weight_norm_proofs;
