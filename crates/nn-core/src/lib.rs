// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// rust-lang/rust-clippy#13774: libtest harness generates a span-less
// [TestDescAndFn; N] array that exceeds 16384 bytes when a crate has
// >2000 tests. Suppress until clippy fix (PR #16347) lands in stable.
#![cfg_attr(test, allow(clippy::large_stack_arrays))]

//! nn-core — tensor types, neural network layers, and model execution
//!
//! This crate provides:
//! - Rank-typed `Tensor<D>` with compile-time dimensionality checking
//! - [`DynTensor`] imperative API (dynamic-rank, CPU + GPU via [`GpuBackend`])
//! - [`nn`] module with 44 Module impls (Linear, Conv1d/2d, LSTM, Attention, Norms, etc.)
//! - [`VarBuilder`] hierarchical weight loader for safetensors/mmap
//! - [`IntervalBounds`] for verification support
//! - Computation graph tracing via `DynTensor` for verification and compilation
//! - Device-agnostic backends via [`GpuBackend`] trait (Metal in nn-metal)

// Force linker to include Accelerate framework symbols when workspace feature
// unification activates ndarray/blas (via NY). Without this, downstream
// test binaries fail to link with "undefined _cblas_sgemm".
extern crate accelerate_src;
extern crate blas_src;

pub mod audio;
pub mod backend;
pub mod bounds;
pub mod device;
pub mod dtype;
pub mod dyn_tensor;
pub mod error;
pub(crate) mod kahan_two_pass;
#[cfg(kani)]
mod kani_attention;
#[cfg(kani)]
mod kani_attention_extended_safety;
#[cfg(kani)]
mod kani_autodiff_gradient_extended;
#[cfg(kani)]
mod kani_autodiff_gradient_safety;
#[cfg(kani)]
mod kani_beam_search_decoding;
#[cfg(kani)]
mod kani_beam_search_safety;
#[cfg(kani)]
mod kani_bounds;
#[cfg(kani)]
mod kani_concat_split_safety;
#[cfg(kani)]
mod kani_conv;
#[cfg(kani)]
mod kani_conv_extended_safety;
#[cfg(kani)]
mod kani_conv_pool;
#[cfg(kani)]
mod kani_device_transfer_safety;
#[cfg(kani)]
mod kani_dpdf_vlm_gpu_transfer_proofs;
#[cfg(kani)]
mod kani_dpdf_vlm_kv_cache_proofs;
#[cfg(kani)]
mod kani_dpdf_vlm_kv_cache_proofs_ext;
#[cfg(kani)]
mod kani_dpdf_vlm_memory_layout;
#[cfg(kani)]
mod kani_dpdf_vlm_memory_layout_ext;
#[cfg(kani)]
mod kani_dpdf_vlm_safetensors_ext_proofs;
#[cfg(kani)]
mod kani_dpdf_vlm_safetensors_proofs;
#[cfg(kani)]
mod kani_dtype_cast_extended;
#[cfg(kani)]
mod kani_dtype_convert;
#[cfg(kani)]
mod kani_dtype_shape_proofs;
#[cfg(kani)]
mod kani_dyn_tensor;
#[cfg(kani)]
mod kani_dyn_tensor_broadcast;
#[cfg(kani)]
mod kani_dyn_tensor_shape;
#[cfg(kani)]
mod kani_elementwise;
#[cfg(kani)]
mod kani_embedding_dpdf_vlm_extended;
#[cfg(kani)]
mod kani_embedding_dpdf_vlm_proofs;
#[cfg(kani)]
mod kani_embedding_safety;
#[cfg(kani)]
mod kani_embedding_vlm_safety;
#[cfg(kani)]
mod kani_image_preprocess_safety;
#[cfg(kani)]
mod kani_indexing_gather_scatter;
#[cfg(kani)]
mod kani_kv_cache_dpdf_extended2;
#[cfg(kani)]
mod kani_kv_cache_safety;
#[cfg(kani)]
mod kani_lstm;
#[cfg(kani)]
mod kani_math_ops;
#[cfg(kani)]
mod kani_mha_attention_safety_proofs;
#[cfg(kani)]
mod kani_mha_dpdf_vlm_extended2;
#[cfg(kani)]
mod kani_mha_dpdf_vlm_proofs;
#[cfg(kani)]
mod kani_mha_dpdf_vlm_proofs_ext;
#[cfg(kani)]
mod kani_mha_safety;
#[cfg(kani)]
mod kani_nn_config;
#[cfg(kani)]
mod kani_nn_forward_shape_proofs;
#[cfg(kani)]
mod kani_nn_pooling_safety;
#[cfg(kani)]
mod kani_nn_shape_consistency;
#[cfg(kani)]
mod kani_ode;
#[cfg(kani)]
mod kani_ode_extended_safety;
#[cfg(kani)]
mod kani_pool;
#[cfg(kani)]
mod kani_quantization_extended;
#[cfg(kani)]
mod kani_quantization_safety;
#[cfg(kani)]
mod kani_quantized;
#[cfg(kani)]
mod kani_recurrent_extended_safety;
#[cfg(kani)]
mod kani_reshape_view_safety;
#[cfg(kani)]
mod kani_safetensors_dpdf_extended;
#[cfg(kani)]
mod kani_safetensors_extended_safety;
#[cfg(kani)]
mod kani_safetensors_safety;
#[cfg(kani)]
mod kani_shape_invariants;
#[cfg(kani)]
mod kani_stride_layout_proofs;
#[cfg(kani)]
mod kani_tensor_bounds;
#[cfg(kani)]
mod kani_tensor_indexing_safety;
#[cfg(kani)]
mod kani_tensor_memory_layout_stride;
#[cfg(kani)]
mod kani_tensor_memory_safety;
#[cfg(kani)]
mod kani_trace_op_bounds;
#[cfg(kani)]
mod kani_trace_op_class;
#[cfg(kani)]
mod kani_trace_types_moe;
#[cfg(kani)]
mod kani_traceop_nn_proofs;
pub mod mixed_precision;
pub mod model_manifest;
pub mod module;
pub mod layers;
pub mod ode;
pub mod tensor;
#[doc(hidden)]
pub mod test_prng;
#[cfg(any(test, feature = "test-helpers"))]
pub mod test_utils;
pub mod var_builder;
pub(crate) mod welford;

pub use backend::{Backend, CpuBackend};
pub use bounds::{next_down_f32, next_up_f32, IntervalBounds};
pub use device::Device;
pub use dtype::DType;
#[allow(deprecated)]
pub use dyn_tensor::softmax_last_dim;
pub use dyn_tensor::{
    conv1d_out_len, conv2d_out_len, conv3d_out_len, conv_transpose1d_out_len,
    conv_transpose2d_out_len, gpu_backend_flush, load_safetensors, load_safetensors_from_bytes,
    register_gpu_backend, save_safetensors, tensors_to_safetensors_bytes, BinaryOp, CompareOp,
    Conv1dParams, Conv2dParams, Conv3dParams, ConvTranspose1dParams, ConvTranspose2dParams, Dim,
    DynTensor, GpuBackend, GpuFullBackend, GpuNnOps, GpuSelectionOps, GpuShapeOps,
    GridSamplePaddingMode, IndexOp, QuantType, QuantizedStorage, ReduceOp, Shape, TensorIndexer,
    UnaryOp, WithDType, D,
};
pub use error::{check_dim, BackendDomain, BackendErrorKind, Result, TensorError};
pub use mixed_precision::{default_op_category, MixedPrecisionPolicy, OpDTypeCategory};
pub use module::ModuleT;
pub use layers::{
    alibi_bias, alibi_bias_scaled, alibi_slopes, apply_adaln_modulation, batch_norm, beam_search,
    causal_mask, causal_mask_dtype, causal_mask_with_offset, check_output_finite, conv1d,
    conv1d_no_bias, conv2d, conv2d_no_bias, conv3d, conv3d_no_bias, conv_transpose1d,
    conv_transpose1d_no_bias, conv_transpose2d, conv_transpose2d_no_bias, ctc_beam_decode,
    ctc_greedy_decode, embedding, generate, group_norm, layer_norm, linear, linear_no_bias,
    log_softmax, lstm, nan_check_policy, repeat_kv, rms_norm, rope, sdpa, sdpa_causal, sigmoid,
    sinusoidal_2d, softmax, window_partition, window_unpartition, with_nan_check_policy,
    Activation, AdaIn, AdaLnParams, AdaLnZero, AdaLnZeroDual, AdaptiveAvgPool2d, AttentionMode,
    AttentiveStatisticsPooling, AvgPool2d, BatchNorm, BatchNorm2d, BatchNormConfig, BeamHypothesis,
    BeamSearchConfig, BeamSearchOutput, BiLstm, BlockQ4K, Conv1d, Conv1dConfig, Conv2d,
    Conv2dConfig, Conv3d, Conv3dConfig, ConvTranspose1d, ConvTranspose1dConfig, ConvTranspose2d,
    ConvTranspose2dConfig, CtcBeamHypothesis, CtcConfig, DeformableAttention,
    DeformableAttentionConfig, DiTBlock, DiTBlockDual, Dropout, Embedding, ExpertFFN,
    GatedDeltaNet, GatedDeltaNetState, GenerationConfig, GenerationOutput, GgmlDType, GroupNorm,
    HalfRotaryEmbedding, InstanceNorm, InstanceNormPrecision, InterleavedMRoPE,
    InterleavedMRoPEConfig, JointAttention, KvCache, KvCacheBackend, KvCacheLayer,
    KvCacheLayerBackend, LayerNorm, LayerNormConfig, Linear, LowRankAdaLn, Lstm, LstmCell,
    LstmState, MBConv, MBConvConfig, MaxPool1d, MaxPool2d, Module, MoeDispatch, MoeDispatchConfig,
    MoeDispatchOutput, MoeLayer, MoeLayerConfig, MoeOutput, MoeRouter, MoeRoutingOutput, MtpHead,
    MtpHeadConfig, MultiHeadAttention, MultimodalRoPE, NanCheckPolicy, PatchEmbedding,
    PixelShuffle, PixelUnshuffle, Pool1dConfig, Pool2dConfig, PoolingStrategy, PreallocKvCache,
    PreallocKvCacheLayer, QLinear, QuantizedWeight, Qwen2VLVitConfig, Res2NetBlock, RmsNorm,
    RotaryEmbedding, RotaryEmbedding2d, Rvq, Sequential, SqueezeExcitation, SqueezeExcitation1d,
    SwiGlu, SwiGluExpert, Upsample2d, UpsampleMode, VitConfig, VitEncoder, VitEncoderBlock,
    VqCodebook, WeightNormConv1d, WindowAttentionConfig, WindowMultiHeadAttention, YarnScaling,
};
// INT8 quantization (W8A16 per-channel) -- Part of #3522
pub use layers::{
    dequantize_per_channel, max_quantization_error, quantize_per_channel, Int8Linear, Int8Mode,
    Int8QuantParams,
};
pub use ode::{euler_solve, euler_solve_cfg, TimeSchedule, VelocityField};
pub use tensor::{Tensor, TensorElement};
pub use var_builder::VarBuilder;
// VarBuilder backend traits for custom weight loaders
pub use var_builder::{NameMapFn, TensorBackend, TensorMapBackend, ZerosBackend};
// Weight name mapper for HF-to-NN import
pub use var_builder::{verify_mapper_coverage, HfToNnMapper, WeightNameMapper};

/// Re-export half for f16/bf16 tensor element types
pub use half;

#[cfg(test)]
#[path = "model_manifest_tests.rs"]
mod model_manifest_tests;

#[cfg(test)]
#[path = "error_dtype_tests.rs"]
mod error_dtype_tests;

#[cfg(test)]
#[path = "var_builder_tests.rs"]
mod var_builder_tests;

#[cfg(test)]
#[path = "var_builder_extended_tests.rs"]
mod var_builder_extended_tests;

#[cfg(test)]
#[path = "audio_tests.rs"]
mod audio_tests;

#[cfg(test)]
#[path = "safetensors_roundtrip_tests.rs"]
mod safetensors_roundtrip_tests;

#[cfg(test)]
#[path = "var_builder_hierarchy_tests.rs"]
mod var_builder_hierarchy_tests;

#[cfg(test)]
#[path = "nn_attention_extended_tests.rs"]
mod nn_attention_extended_tests;

#[cfg(test)]
#[path = "nn_conv_extended_tests.rs"]
mod nn_conv_extended_tests;

#[cfg(test)]
#[path = "nn_generation_tests.rs"]
mod nn_generation_tests;

#[cfg(test)]
#[path = "dyn_tensor_ops_extended_tests.rs"]
mod dyn_tensor_ops_extended_tests;

#[cfg(test)]
#[path = "dyn_tensor_shape_extended_tests.rs"]
mod dyn_tensor_shape_extended_tests;

#[cfg(test)]
#[path = "mixed_precision_extended_tests.rs"]
mod mixed_precision_extended_tests;

#[cfg(test)]
#[path = "nn_normalization_extended_tests.rs"]
mod nn_normalization_extended_tests;

#[cfg(test)]
#[path = "nn_pooling_embedding_tests.rs"]
mod nn_pooling_embedding_tests;

#[cfg(test)]
#[path = "nn_layer_config_tests.rs"]
mod nn_layer_config_tests;

#[cfg(test)]
#[path = "var_builder_weight_tests.rs"]
mod var_builder_weight_tests;

#[cfg(test)]
#[path = "var_builder_name_mapping_tests.rs"]
mod var_builder_name_mapping_tests;

#[cfg(test)]
#[path = "error_dtype_extended_tests.rs"]
mod error_dtype_extended_tests;

#[cfg(test)]
#[path = "dtype_mixed_precision_tests.rs"]
mod dtype_mixed_precision_tests;

#[cfg(test)]
#[path = "nn_attention_sdpa_tests.rs"]
mod nn_attention_sdpa_tests;

#[cfg(test)]
#[path = "dtype_manifest_extended_tests.rs"]
mod dtype_manifest_extended_tests;

#[cfg(test)]
#[path = "nn_vision_config_tests.rs"]
mod nn_vision_config_tests;

#[cfg(test)]
#[path = "nn_vision_extended_tests.rs"]
mod nn_vision_extended_tests;

#[cfg(test)]
#[path = "nn_layer_extended_tests.rs"]
mod nn_layer_extended_tests;

#[cfg(test)]
#[path = "model_import_weight_loading_extended_tests.rs"]
mod model_import_weight_loading_extended_tests;
