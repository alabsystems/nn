// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CUDA/HIP GPU backend for nn.
//!
//! Dual-target GPU backend supporting both NVIDIA CUDA (via PTX/cubin) and
//! AMD ROCm (via HIP C++/hsaco). Both targets share the same `TensorOpKind`
//! IR and `DispatchStep` plan from `nn-dsl`; only the emission and
//! compilation stages are target-specific.
//!
//! # Architecture
//!
//! ```text
//! TensorOpKind IR (nn-dsl, backend-agnostic)
//!   ├─ HIP C++ emission  → hipcc → .hsaco  → HipRuntime dispatch (AMD)
//!   └─ CUDA C++ emission → nvcc  → .ptx    → CudaRuntime dispatch (NVIDIA)
//!                                 → ptxas → .cubin (optional, avoids JIT)
//! ```
//!
//! # HIP backend (AMD)
//!
//! Phases 1-9: Full HIP C++ codegen for GEMM, elementwise, softmax, reduce,
//! conv1d/2d, structural ops, MoE, MXFP4, MLA decode, rocWMMA tiled GEMM.
//! Runtime dispatch via `hipModuleLaunchKernel`. See `codegen_hip*` modules.
//!
//! # CUDA/PTX backend (NVIDIA)
//!
//! PTX generation infrastructure for NVIDIA GPUs:
//! - PTX type mapping and assembly helpers (`codegen_ptx`)
//! - CUDA C++ `CodegenSyntax` impl for shared codegen (`codegen_syntax_ptx`)
//! - CUDA C++ kernel emission: elementwise, softmax, reduction, tiled matmul (`ptx_emit`)
//! - `nvcc`/`ptxas` compilation pipeline with cache (`compile_ptx`)
//! - CUDA Driver API FFI types (`cuda_ffi`)
//! - Safe runtime wrappers: `CudaRuntime`, `CudaBuffer`, `CudaStream` (`cuda_runtime`)
//! - Platform-gated: real FFI on Linux, graceful `NotAvailable` on macOS
//!
//! Deferred to follow-up iterations:
//! - End-to-end integration tests on both AMD and NVIDIA hardware
//! - Tensor core (wmma) PTX emission for NVIDIA
//! - Native PTX kernel dispatch for DynTensor ops (currently CPU round-trip)

// ---- DynTensor CUDA backend (GpuBackend impl) ----
pub mod dyn_tensor_cuda;
pub use dyn_tensor_cuda::{init_cuda_runtime, register_cuda_dyn_backend, CudaTensorData};

// ---- HIP (AMD ROCm) modules ----
pub mod codegen_hip;
pub mod codegen_hip_mla_decode;
pub mod codegen_hip_moe;
pub mod codegen_hip_mxfp4;
pub mod codegen_hip_mxfp4_gemm;
pub mod codegen_hip_tensor;
pub mod codegen_hip_tensor_emit_complex;
pub mod codegen_hip_tensor_emit_conv;
pub mod codegen_hip_tensor_emit_elementwise;
pub mod codegen_hip_tensor_emit_gemm;
pub mod codegen_hip_tensor_emit_index;
pub mod codegen_hip_tensor_emit_ops;
pub mod codegen_hip_tensor_emit_select;
pub mod codegen_hip_tensor_emit_step;
pub mod codegen_hip_tensor_emit_structural;
pub mod compile_hip;
mod error;
pub mod hip_cache;
pub mod hip_dispatch;
pub mod hip_ffi;
pub mod hip_runtime;

// ---- CUDA/PTX (NVIDIA) modules ----
pub mod codegen_ptx;
pub mod codegen_syntax_ptx;
pub mod compile_ptx;
pub mod cuda_ffi;
pub mod cuda_runtime;
pub mod cuda_validation;
pub mod ptx_activations;
pub mod ptx_attention;
pub mod ptx_attention_multihead;
pub mod ptx_batchnorm;
pub mod ptx_cast;
pub mod ptx_conv1d;
pub mod ptx_conv2d;
pub mod ptx_depthwise_conv;
pub mod ptx_elementwise;
pub mod ptx_embedding;
pub mod ptx_emit;
pub mod ptx_gather;
pub mod ptx_gemv;
pub mod ptx_groupnorm;
pub mod ptx_instancenorm;
pub mod ptx_layernorm;
pub mod ptx_linear;
pub mod ptx_matmul;
pub mod ptx_pad;
pub mod ptx_pooling;
pub mod ptx_quantize;
pub mod ptx_reduce;
pub mod ptx_residual;
pub mod ptx_rmsnorm;
pub mod ptx_rope;
pub mod ptx_softmax;
pub mod ptx_tensor_ops;
pub mod ptx_transpose;
pub mod ptx_upsample;
pub mod ptx_where;

// ---- HIP re-exports ----
pub use codegen_hip::{hip_accumulator_type, hip_type, HIP_BLOCK_SIZE, HIP_PRELUDE};
pub use codegen_hip_tensor::{emit_gemm_hip, emit_tensor_hip, emit_tensor_hip_with_plan};
pub use codegen_hip_tensor_emit_gemm::{
    emit_rocwmma_linear_kernel, emit_rocwmma_matmul_kernel, should_use_rocwmma, ROCWMMA_INCLUDE,
};
pub use compile_hip::{check_hipcc, compile_hip_source, hipcc_command, HipCompileError, HipModule};
pub use error::HipCodegenError;
pub use hip_cache::HipCache;
pub use hip_dispatch::{launch_config_for_step, HipDispatchError, HipDispatcher, PreparedKernel};
pub use hip_ffi::{Dim3, LaunchConfig};
pub use hip_runtime::{
    is_hip_available, HipBuffer, HipKernel, HipRuntime, HipRuntimeError, HipStream,
};

// ---- CUDA/PTX re-exports ----
pub use codegen_ptx::{
    cuda_type, format_ptx_float, ptx_prelude, ptx_type, PtxCodegenError, PTX_BLOCK_SIZE,
    PTX_VERSION, WARP_SIZE,
};
pub use codegen_syntax_ptx::CudaSyntax;
pub use compile_ptx::{
    assemble_ptx_to_cubin, check_nvcc, check_ptxas, compile_cuda_to_ptx, nvcc_command,
    ptxas_command, PtxCompileError, PtxModule,
};
pub use cuda_ffi::{CudaDim3, CudaLaunchConfig, CudaMemcpyKind};
pub use cuda_runtime::{
    is_cuda_available, CudaBuffer, CudaKernel, CudaRuntime, CudaRuntimeError, CudaStream,
};
pub use ptx_activations::{
    emit_ptx_activation, emit_ptx_activation_default, gelu_fast_reference, gelu_reference,
    generate_all_activation_ptx, mish_reference, ptx_activation_launch_config, silu_reference,
    snake_reference, PtxActivation, PtxActivationConfig,
};
pub use ptx_attention::{
    emit_ptx_attention, emit_ptx_attention_default, generate_sdpa_causal_ptx, generate_sdpa_ptx,
    ptx_attention_launch_config, sdpa_reference, PtxAttentionConfig, ATTENTION_BLOCK_SIZE,
};
pub use ptx_attention_multihead::{
    attention_reference, generate_multihead_attention_ptx, multihead_attention_launch_config,
    PtxMultiHeadAttentionConfig,
};
pub use ptx_batchnorm::{
    batchnorm_reference, emit_ptx_batchnorm, emit_ptx_batchnorm_default, generate_batchnorm_ptx,
    ptx_batchnorm_launch_config, PtxBatchNormConfig,
};
pub use ptx_cast::{
    generate_bf16_to_f32_ptx, generate_f16_to_f32_ptx, generate_f32_to_bf16_ptx,
    generate_f32_to_f16_ptx, CAST_BLOCK_SIZE,
};
pub use ptx_conv1d::{
    conv1d_output_length, emit_ptx_conv1d, emit_ptx_conv1d_default, ptx_conv1d_launch_config,
    PtxConv1dConfig, PTX_CONV1D_BLOCK_SIZE, PTX_CONV1D_MAX_KERNEL,
};
pub use ptx_conv2d::{
    conv2d_output_size, conv2d_reference, emit_ptx_conv2d, emit_ptx_conv2d_default,
    ptx_conv2d_launch_config, PtxConv2dConfig, PTX_CONV2D_BLOCK_H, PTX_CONV2D_BLOCK_W,
    PTX_CONV2D_MAX_BLOCK, PTX_CONV2D_MIN_BLOCK,
};
pub use ptx_depthwise_conv::{
    depthwise_conv2d_output_size, depthwise_conv2d_reference, generate_depthwise_conv2d_ptx,
    ptx_depthwise_conv2d_launch_config, PtxDepthwiseConv2dConfig, PTX_DEPTHWISE_CONV2D_BLOCK_SIZE,
};
pub use ptx_elementwise::{
    add_reference, div_reference, exp_reference, generate_add_ptx, generate_div_ptx,
    generate_exp_ptx, generate_log_ptx, generate_mul_ptx, generate_neg_ptx,
    generate_scalar_mul_ptx, generate_sqrt_ptx, generate_sub_ptx, log_reference, mul_reference,
    neg_reference, ptx_elementwise_launch_config, scalar_mul_reference, sqrt_reference,
    sub_reference, ELEMENTWISE_BLOCK_SIZE,
};
pub use ptx_embedding::{
    embedding_reference, generate_embedding_ptx, ptx_embedding_launch_config, PtxEmbeddingConfig,
    EMBEDDING_BLOCK_SIZE,
};
pub use ptx_emit::{
    emit_activation_kernels, emit_elementwise_kernel, emit_matmul_kernel, emit_reduction_kernel,
    emit_softmax_kernel, ReductionOp,
};
pub use ptx_gather::{
    gather_reference, generate_gather_ptx, generate_scatter_add_ptx, ptx_gather_launch_config,
    scatter_add_reference, GATHER_BLOCK_SIZE,
};
pub use ptx_gemv::{
    dot_reference, gemv_reference, generate_dot_ptx, generate_gemv_ptx, generate_outer_ptx,
    outer_reference, GEMV_BLOCK_SIZE,
};
pub use ptx_groupnorm::{
    emit_ptx_groupnorm, emit_ptx_groupnorm_default, generate_groupnorm_ptx, groupnorm_reference,
    ptx_groupnorm_launch_config, PtxGroupNormConfig,
};
pub use ptx_instancenorm::{
    generate_instancenorm_ptx, instancenorm_reference, INSTANCENORM_BLOCK_SIZE,
};
pub use ptx_layernorm::{
    emit_ptx_layernorm, emit_ptx_layernorm_default, generate_layernorm_ptx, layernorm_reference,
    ptx_layernorm_launch_config, PtxLayerNormConfig,
};
pub use ptx_linear::{
    generate_linear_no_bias_ptx, generate_linear_ptx, generate_linear_relu_ptx, linear_reference,
    ptx_linear_launch_config, LINEAR_BLOCK_SIZE,
};
pub use ptx_matmul::{
    emit_ptx_matmul, emit_ptx_matmul_default, generate_matmul_ptx, generate_matmul_tiled_ptx,
    matmul_reference, ptx_matmul_launch_config, PtxMatmulConfig, MATMUL_BLOCK_SIZE,
    PTX_MATMUL_MAX_TILE, PTX_MATMUL_MIN_TILE, PTX_MATMUL_TILE_SIZE,
};
pub use ptx_pad::{
    generate_pad1d_ptx, generate_reflect_pad1d_ptx, pad1d_reference, reflect_pad1d_reference,
    PAD_BLOCK_SIZE,
};
pub use ptx_pooling::{
    adaptive_avg_pool2d_reference, avg_pool2d_reference, generate_adaptive_avg_pool2d_ptx,
    generate_avg_pool2d_ptx, generate_max_pool2d_ptx, max_pool2d_reference, pool2d_output_size,
    ptx_pool2d_launch_config, PtxAdaptiveAvgPool2dConfig, PtxPool2dConfig, PTX_POOL2D_BLOCK_SIZE,
};
pub use ptx_quantize::{
    dequantize_reference, generate_dequantize_int8_to_f32_ptx, generate_quantize_f32_to_int8_ptx,
    quantize_reference, QUANTIZE_BLOCK_SIZE,
};
pub use ptx_reduce::{
    argmax_reference, argmin_reference, generate_argmax_ptx, generate_argmin_ptx, generate_max_ptx,
    generate_mean_ptx, generate_sum_ptx, max_reference, mean_reference, ptx_reduce_launch_config,
    sum_reference, REDUCE_BLOCK_SIZE,
};
pub use ptx_residual::{
    generate_residual_add_layernorm_ptx, generate_residual_add_ptx, generate_residual_add_relu_ptx,
    residual_add_launch_config, residual_add_layernorm_launch_config,
    residual_add_layernorm_reference, residual_add_reference, residual_add_relu_launch_config,
    residual_add_relu_reference, RESIDUAL_BLOCK_SIZE,
};
pub use ptx_rmsnorm::{
    emit_ptx_rmsnorm, emit_ptx_rmsnorm_default, generate_rmsnorm_ptx, ptx_rmsnorm_launch_config,
    rmsnorm_reference, PtxRmsNormConfig,
};
pub use ptx_rope::{
    generate_rope_cached_ptx, generate_rope_ptx, ptx_rope_launch_config, rope_reference,
    rope_reference_with_base, PtxRopeConfig, ROPE_BLOCK_SIZE,
};
pub use ptx_softmax::{
    emit_ptx_softmax, emit_ptx_softmax_default, generate_log_softmax_ptx, generate_softmax_ptx,
    log_softmax_reference, ptx_softmax_launch_config, softmax_reference, PtxSoftmaxConfig,
    SOFTMAX_BLOCK_SIZE,
};
pub use ptx_tensor_ops::{
    concat_reference, fill_reference, generate_concat_ptx, generate_fill_ptx, generate_repeat_ptx,
    generate_slice_ptx, ptx_tensor_ops_launch_config, repeat_reference, slice_reference,
    TENSOR_OPS_BLOCK_SIZE,
};
pub use ptx_transpose::{
    batch_transpose_reference, generate_batch_transpose_ptx, generate_transpose_ptx,
    ptx_batch_transpose_launch_config, ptx_transpose_launch_config, transpose_reference,
    TRANSPOSE_BLOCK_SIZE,
};
pub use ptx_upsample::{
    generate_upsample_nearest1d_ptx, generate_upsample_nearest2d_ptx, upsample_nearest1d_reference,
    upsample_nearest2d_reference, UPSAMPLE_BLOCK_SIZE,
};
pub use ptx_where::{
    clamp_reference, generate_clamp_ptx, generate_where_ptx, ptx_where_launch_config,
    where_reference, WHERE_BLOCK_SIZE,
};

#[cfg(test)]
#[path = "cuda_kernel_extended_tests.rs"]
mod cuda_kernel_extended_tests;

#[cfg(test)]
#[path = "cuda_kernel_config_extended_tests.rs"]
mod cuda_kernel_config_extended_tests;

#[cfg(test)]
#[path = "ptx_kernel_extended_tests.rs"]
mod ptx_kernel_extended_tests;

#[cfg(test)]
#[path = "ptx_kernel_extended_tests2.rs"]
mod ptx_kernel_extended_tests2;

#[cfg(test)]
#[path = "ptx_attention_extended_tests.rs"]
mod ptx_attention_extended_tests;

#[cfg(test)]
#[path = "ptx_quantize_tests.rs"]
mod ptx_quantize_tests;

#[cfg(test)]
#[path = "ptx_depthwise_conv_extended_tests.rs"]
mod ptx_depthwise_conv_extended_tests;

#[cfg(test)]
#[path = "ptx_reduce_tests.rs"]
mod ptx_reduce_tests;

#[cfg(test)]
#[path = "ptx_gather_tests.rs"]
mod ptx_gather_tests;

#[cfg(test)]
#[path = "ptx_residual_tests.rs"]
mod ptx_residual_tests;

#[cfg(test)]
#[path = "ptx_emit_tests.rs"]
mod ptx_emit_tests;

#[cfg(test)]
#[path = "ptx_integration_tests.rs"]
mod ptx_integration_tests;

#[cfg(test)]
#[path = "cuda_dispatch_plan_tests.rs"]
mod cuda_dispatch_plan_tests;

#[cfg(test)]
#[path = "cuda_config_validation_tests.rs"]
mod cuda_config_validation_tests;

#[cfg(test)]
#[path = "cuda_error_extended_tests.rs"]
mod cuda_error_extended_tests;

#[cfg(test)]
#[path = "cuda_e2e_validation_tests.rs"]
mod cuda_e2e_validation_tests;

#[cfg(test)]
#[path = "cuda_ptx_generation_tests.rs"]
mod cuda_ptx_generation_tests;

#[cfg(test)]
#[path = "dyn_tensor_cuda_tests.rs"]
mod dyn_tensor_cuda_tests;

#[cfg(kani)]
#[path = "kani_dispatch_coverage.rs"]
mod kani_dispatch_coverage;

#[cfg(kani)]
#[path = "kani_codegen_hip_emit_step.rs"]
mod kani_codegen_hip_emit_step;

#[cfg(kani)]
#[path = "kani_codegen_hip_emit_conv.rs"]
mod kani_codegen_hip_emit_conv;

#[cfg(kani)]
#[path = "kani_codegen_hip_emit_structural.rs"]
mod kani_codegen_hip_emit_structural;

#[cfg(kani)]
mod kani_codegen_hip_moe;
#[cfg(kani)]
mod kani_hip_dispatch;
#[cfg(kani)]
mod kani_hip_runtime;

#[cfg(kani)]
#[path = "kani_mla_mxfp4_cache.rs"]
mod kani_mla_mxfp4_cache;
