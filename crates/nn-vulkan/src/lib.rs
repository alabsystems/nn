// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vulkan GPU backend for nn: SPIR-V compute pipeline for cross-platform inference.
//!
//! Cross-platform GPU backend targeting AMD, Intel, mobile, and any Vulkan-capable
//! device. Complements the Metal backend (Apple) and CUDA/HIP backend (NVIDIA/AMD).
//! All three backends share the same `TensorOpKind` IR from `nn-dsl` and produce
//! identical numerical results for the same kernel + inputs.
//!
//! # Architecture
//!
//! ```text
//! TensorOpKind IR (nn-dsl, backend-agnostic)
//!   ├─ GLSL text emission  → glslangValidator → .spv  → VulkanDispatcher (Vulkan 1.2+)
//!   └─ Direct SPIR-V binary construction (future: precise control for tiled matmul)
//! ```
//!
//! # Modules
//!
//! - **[`error`]**: `VulkanError` enum with `thiserror`.
//! - **[`device`]**: Physical device discovery, queue family selection, memory types.
//! - **[`buffer`]**: `VulkanBuffer` (device-local) and `StagingBuffer` (host-visible).
//! - **[`buffer_pool`]**: Size-class buffer reuse pool (mirrors Metal `BufferPool`).
//! - **[`spirv_emit`]**: GLSL compute shader generation and SPIR-V binary helpers.
//! - **[`spirv_binary`]**: Direct SPIR-V binary generation for ML ops (add, mul, ReLU, scalar mul, transpose).
//! - **[`spirv_matmul`]**: Direct SPIR-V binary generation for matrix multiplication (naive and tiled).
//! - **[`spirv_reduction`]**: Direct SPIR-V binary generation for reduction ops (sum, max, mean, softmax).
//! - **[`spirv_transpose`]**: Direct SPIR-V binary generation for 2D and batched transpose.
//! - **[`spirv_argreduce`]**: Direct SPIR-V binary generation for argmax, argmin, and top-k.
//! - **[`spirv_quantized`]**: Direct SPIR-V binary generation for INT8 quantize/dequantize and dtype casting.
//! - **[`spirv_gather`]**: Direct SPIR-V binary generation for gather and scatter operations.
//! - **[`spirv_clamp`]**: Direct SPIR-V binary generation for clamp operations.
//! - **[`spirv_upsample`]**: Direct SPIR-V binary generation for nearest-neighbor upsampling (1D and 2D).
//! - **[`spirv_where`]**: Direct SPIR-V binary generation for where/select and absolute value operations.
//! - **[`spirv_tensor_ops`]**: Direct SPIR-V binary generation for tensor manipulation (concat, slice, repeat, fill).
//! - **[`dispatch`]**: Compute pipeline creation, descriptor sets, single-dispatch command buffer.
//! - **[`command_batch`]**: Multi-dispatch command batch with memory barriers.
//! - **[`pipeline_cache`]**: Two-tier (L1 thread-local + L2 shared) pipeline cache.
//! - **[`workgroup`]**: Workgroup size calculation and dispatch validation utilities.
//! - **[`kernels`]**: Pre-built GLSL shader strings for activation, reduction, matmul.
//!
//! # Platform support
//!
//! SPIR-V code generation works on all platforms (no Vulkan runtime needed).
//! Runtime dispatch requires a Vulkan-capable GPU and driver. On systems
//! without Vulkan, [`device::is_vulkan_available`] returns `false` and
//! [`device::VulkanDevice::new`] returns [`VulkanError::NotAvailable`].
//!
//! # Status
//!
//! Phase 4: types, GLSL emission, dispatch infrastructure, pipeline cache,
//! command batch, buffer pool, workgroup utilities. Runtime FFI integration
//! (via `ash` or `wgpu`) is deferred to follow-up iterations.
//!
//! Deferred:
//! - Vulkan FFI bindings (ash/wgpu runtime)
//! - DynTensor backend registration
//! - End-to-end integration tests on GPU hardware
//! - Subgroup operations for tiled matmul
//! - bf16 emulation via uint16 pack/unpack

pub mod buffer;
pub mod buffer_pool;
pub mod command_batch;
pub mod compute_pipeline;
pub mod device;
pub mod dispatch;
pub mod error;
pub mod kernels;
pub mod pipeline_cache;
pub mod spirv_activations;
pub mod spirv_argreduce;
pub mod spirv_attention;
pub mod spirv_binary;
pub mod spirv_cast;
pub mod spirv_clamp;
pub mod spirv_conv;
pub mod spirv_conv1d;
pub mod spirv_conv2d;
pub mod spirv_depthwise_conv;
pub mod spirv_embedding;
pub mod spirv_emit;
pub mod spirv_fused_linear_act;
pub mod spirv_fused_residual;
pub mod spirv_gather;
pub mod spirv_gemv;
pub mod spirv_layernorm;
pub mod spirv_linear;
pub mod spirv_matmul;
pub mod spirv_norms;
pub mod spirv_pad;
pub mod spirv_pool2d;
pub mod spirv_quantized;
pub mod spirv_reduction;
pub mod spirv_rmsnorm;
pub mod spirv_rope;
pub mod spirv_softmax;
pub mod spirv_tensor_ops;
pub mod spirv_transpose;
pub mod spirv_upsample;
pub mod spirv_where;
pub mod workgroup;

// ---- Public re-exports ----
pub use buffer::{BufferUsage, StagingBuffer, VulkanBuffer};
pub use buffer_pool::{BufferPool, BufferPoolStats, PoolStats, SizeClassStats};
pub use command_batch::{BarrierStrategy, CommandBatch, PendingBatch};
pub use compute_pipeline::{
    compute_grid_dims, spirv_words_to_bytes, BufferBinding, CompiledShader, DispatchConfig,
    PushConstants, VulkanComputeConfig, VulkanPipelineError,
};
pub use device::{is_vulkan_available, MemoryPropertyFlags, QueueFamilyInfo, VulkanDevice};
pub use dispatch::{
    ComputePipeline, DescriptorBinding, DescriptorSetLayout, DescriptorType, PipelineLayout,
    PushConstantRange, VulkanDispatcher,
};
pub use error::VulkanError;
pub use pipeline_cache::{PipelineCache, PipelineCacheStats};
pub use spirv_activations::{
    fused_adain_snake_reference, gelu_reference, generate_fused_adain_snake_spirv,
    generate_gelu_spirv, generate_silu_spirv, generate_snake_spirv, silu_reference,
    snake_reference, ACTIVATION_WORKGROUP_SIZE,
};
pub use spirv_argreduce::{
    argmax_reference, argmin_reference, generate_argmax_spirv, generate_argmin_spirv,
    generate_topk_spirv, ARGREDUCE_WORKGROUP_SIZE,
};
pub use spirv_attention::{generate_attention_spirv, ATTENTION_WORKGROUP_SIZE};
pub use spirv_binary::{
    emit_add_spirv, emit_mul_spirv, emit_relu_spirv, emit_scalar_mul_spirv, emit_transpose_spirv,
    find_entry_point_name, find_workgroup_size, BINARY_WORKGROUP_SIZE,
};
pub use spirv_cast::{
    generate_bf16_to_f32_spirv, generate_f16_to_f32_spirv, generate_f32_to_bf16_spirv,
    generate_f32_to_f16_spirv, CAST_WORKGROUP_SIZE,
};
pub use spirv_clamp::{clamp_reference, generate_clamp_spirv, CLAMP_WORKGROUP_SIZE};
pub use spirv_conv::{
    conv1d_output_length, generate_avg_pool1d_spirv, generate_conv1d_spirv,
    generate_max_pool1d_spirv, pool1d_output_length, CONV_WORKGROUP_SIZE,
};
pub use spirv_conv1d::{
    conv1d_reference, generate_conv1d_grouped_spirv, Conv1dConfig, CONV1D_WORKGROUP_SIZE,
};
pub use spirv_conv2d::{
    conv2d_output_size, conv2d_reference, generate_conv2d_spirv, Conv2dConfig,
    CONV2D_WORKGROUP_SIZE,
};
pub use spirv_depthwise_conv::{
    depthwise_conv1d_reference, generate_depthwise_conv1d_spirv, DEPTHWISE_CONV_WORKGROUP_SIZE,
};
pub use spirv_embedding::{generate_embedding_spirv, EMBEDDING_WORKGROUP_SIZE};
pub use spirv_emit::{
    emit_elementwise_glsl, emit_matmul_glsl, emit_reduction_glsl, emit_softmax_glsl, glsl_type,
    spirv_type_bytes, ReductionOp, DEFAULT_WORKGROUP_SIZE, GLSL_COMPUTE_VERSION, SPIRV_MAGIC,
    SPIRV_VERSION_1_5,
};
pub use spirv_fused_linear_act::{
    fused_linear_gelu_reference, fused_linear_relu_reference, fused_linear_silu_reference,
    generate_fused_linear_gelu_spirv, generate_fused_linear_relu_spirv,
    generate_fused_linear_silu_spirv, FUSED_LINEAR_ACT_WORKGROUP_SIZE,
};
pub use spirv_fused_residual::{
    bias_residual_add_reference, generate_bias_residual_add_spirv,
    generate_residual_add_gelu_spirv, generate_residual_add_relu_spirv,
    generate_residual_add_spirv, residual_add_gelu_reference, residual_add_reference,
    residual_add_relu_reference, FUSED_RESIDUAL_WORKGROUP_SIZE,
};
pub use spirv_gather::{
    gather_reference, generate_gather_spirv, generate_scatter_spirv, scatter_reference,
    GATHER_WORKGROUP_SIZE,
};
pub use spirv_gemv::{
    dot_reference, gemv_reference, generate_dot_spirv, generate_gemv_spirv, generate_outer_spirv,
    outer_reference, GEMV_WORKGROUP_SIZE,
};
pub use spirv_layernorm::{
    generate_layernorm_spirv, generate_rmsnorm_spirv, LAYERNORM_WORKGROUP_SIZE,
};
pub use spirv_linear::{
    generate_linear_no_bias_spirv, generate_linear_spirv, linear_reference, LINEAR_WORKGROUP_SIZE,
};
pub use spirv_matmul::{generate_matmul_spirv, generate_matmul_spirv_naive, MATMUL_TILE_SIZE};
pub use spirv_norms::{
    batchnorm_reference, generate_batchnorm_spirv, generate_groupnorm_spirv,
    generate_instancenorm_spirv, groupnorm_reference, instancenorm_reference, NORM_WORKGROUP_SIZE,
};
pub use spirv_pad::{
    generate_pad2d_spirv, generate_pad_spirv, pad2d_reference, pad_reference, PAD_WORKGROUP_SIZE,
};
pub use spirv_pool2d::{
    avg_pool2d_reference, generate_avg_pool2d_spirv, generate_max_pool2d_spirv,
    max_pool2d_reference, pool2d_output_size, Pool2dConfig, POOL2D_WORKGROUP_SIZE,
};
pub use spirv_quantized::{
    dequantize_reference, generate_dequantize_int8_spirv, generate_quantize_f32_to_int8_spirv,
    quantize_reference, QUANTIZED_WORKGROUP_SIZE,
};
pub use spirv_reduction::{
    generate_max_spirv, generate_mean_spirv, generate_softmax_spirv, generate_sum_spirv,
    REDUCTION_WORKGROUP_SIZE,
};
pub use spirv_rmsnorm::{
    generate_rmsnorm_separate_io_spirv, rmsnorm_reference, RmsNormConfig, RMSNORM_WORKGROUP_SIZE,
};
pub use spirv_rope::{
    generate_rope_neox_spirv, generate_rope_spirv, rope_reference, ROPE_WORKGROUP_SIZE,
};
pub use spirv_softmax::{
    generate_softmax_separate_io_spirv, reference_softmax, SOFTMAX_WORKGROUP_SIZE,
};
pub use spirv_tensor_ops::{
    concat_reference, fill_reference, generate_concat_spirv, generate_fill_spirv,
    generate_repeat_spirv, generate_slice_spirv, repeat_reference, slice_reference,
    TENSOR_OPS_WORKGROUP_SIZE,
};
pub use spirv_transpose::{
    generate_batch_transpose_spirv, generate_transpose_spirv, transpose_reference,
    TRANSPOSE_WORKGROUP_SIZE,
};
pub use spirv_upsample::{
    generate_upsample_nearest1d_spirv, generate_upsample_nearest2d_spirv,
    upsample_nearest1d_reference, upsample_nearest2d_reference, UPSAMPLE_WORKGROUP_SIZE,
};
pub use spirv_where::{
    abs_reference, generate_abs_spirv, generate_where_spirv, where_reference, WHERE_WORKGROUP_SIZE,
};
pub use workgroup::{
    optimal_elementwise_workgroup, push_constants_1d, push_constants_matmul,
    push_constants_reduction, validate_dispatch, workgroup_count_1d, workgroup_count_2d,
    workgroup_count_row_reduce,
};

#[cfg(test)]
#[path = "spirv_reduction_tests.rs"]
mod spirv_reduction_tests;

#[cfg(test)]
#[path = "spirv_transpose_tests.rs"]
mod spirv_transpose_tests;

#[cfg(test)]
#[path = "spirv_argreduce_tests.rs"]
mod spirv_argreduce_tests;

#[cfg(test)]
#[path = "spirv_quantized_tests.rs"]
mod spirv_quantized_tests;

#[cfg(test)]
#[path = "spirv_gather_tests.rs"]
mod spirv_gather_tests;

#[cfg(test)]
#[path = "spirv_clamp_tests.rs"]
mod spirv_clamp_tests;

#[cfg(test)]
#[path = "spirv_tensor_ops_tests.rs"]
mod spirv_tensor_ops_tests;

#[cfg(test)]
#[path = "spirv_binary_tests.rs"]
mod spirv_binary_tests;

#[cfg(test)]
#[path = "spirv_emit_tests.rs"]
mod spirv_emit_tests;

#[cfg(test)]
#[path = "workgroup_extended_tests.rs"]
mod workgroup_extended_tests;

#[cfg(test)]
#[path = "spirv_norms_extended_tests.rs"]
mod spirv_norms_extended_tests;

#[cfg(test)]
#[path = "vulkan_kernel_extended_tests.rs"]
mod vulkan_kernel_extended_tests;

#[cfg(test)]
#[path = "vulkan_spirv_generation_tests.rs"]
mod vulkan_spirv_generation_tests;
