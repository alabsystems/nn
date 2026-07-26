// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CPU backend with ARM NEON / x86 AVX2 SIMD for nn.
//!
//! Provides SIMD-optimized elementwise activations, reductions, and matmul.
//! Falls back to scalar implementations on unsupported architectures.
//! Serves as the testing baseline and differential test target for GPU backends.

pub mod simd_detect;

pub mod attention;
pub mod conv1d;
pub mod elementwise;
pub mod layernorm;
pub mod matmul;
pub mod reduce;
pub mod reduction;
pub mod simd_attention;
pub mod simd_batchnorm;
pub mod simd_cast;
pub mod simd_conv1d;
pub mod simd_conv2d;
pub mod simd_elementwise;
pub mod simd_embedding;
pub mod simd_gather;
pub mod simd_gemv;
pub mod simd_groupnorm;
pub mod simd_instance_norm;
pub mod simd_layernorm;
pub mod simd_linear;
pub mod simd_matmul;
pub mod simd_normalize;
pub mod simd_pooling;
pub mod simd_quantize;
pub mod simd_reduce;
pub mod simd_rmsnorm;
pub mod simd_rope;
pub mod simd_sdpa;
pub mod simd_softmax;
pub mod simd_transpose;
pub mod softmax;

// Re-exports for conv1d full API (groups + dilation).
pub use simd_conv1d::{
    conv1d_full, conv1d_full_reference, conv1d_grouped, Conv1dConfig, Conv1dError,
};

// Re-exports for embedding API.
pub use simd_embedding::{
    embedding, embedding_lookup, embedding_reference, embedding_scalar, EmbeddingError,
    EMBEDDING_BLOCK_SIZE,
};

// Re-exports for gather/scatter API.
pub use simd_gather::{
    gather_1d, gather_1d_reference, gather_1d_scalar, scatter_add_1d, scatter_add_1d_reference,
    scatter_add_1d_scalar, GatherError, GATHER_BLOCK_SIZE,
};

// Re-exports for SIMD softmax API.
pub use simd_softmax::{
    softmax_f32, softmax_f32_avx2, softmax_f32_neon, softmax_f32_reference, softmax_f32_scalar,
};

// Re-exports for SIMD layer_norm API.
pub use simd_layernorm::{
    layer_norm_f32, layer_norm_f32_avx2, layer_norm_f32_neon, layer_norm_f32_reference,
    layer_norm_f32_scalar,
};

// Re-exports for SIMD instance_norm API.
pub use simd_instance_norm::{
    instance_norm_f32, instance_norm_f32_avx2, instance_norm_f32_neon, instance_norm_f32_reference,
    instance_norm_f32_scalar,
};

// Re-exports for SIMD batchnorm API.
pub use simd_batchnorm::{
    batchnorm_f32, batchnorm_f32_avx2, batchnorm_f32_neon, batchnorm_f32_scalar,
    batchnorm_reference, BATCHNORM_SIMD_THRESHOLD,
};

// Re-exports for SIMD groupnorm API.
pub use simd_groupnorm::{
    groupnorm_f32, groupnorm_f32_avx2, groupnorm_f32_neon, groupnorm_f32_scalar,
    groupnorm_reference,
};

// Re-exports for SIMD matmul API.
pub use simd_matmul::{
    matmul_f32, matmul_f32_avx2, matmul_f32_neon, matmul_f32_scalar, matmul_reference,
};

// Re-exports for SIMD GEMV API.
pub use simd_gemv::{
    gemv_bias_f32, gemv_f32, gemv_f32_avx2, gemv_f32_neon, gemv_f32_scalar, gemv_reference,
};

// Re-exports for SIMD elementwise API.
pub use simd_elementwise::{
    add_f32, add_f32_avx2, add_f32_neon, add_f32_scalar, fma_f32, fma_f32_avx2, fma_f32_neon,
    fma_f32_scalar, gelu_f32, gelu_f32_avx2, gelu_f32_neon, gelu_f32_scalar, mul_f32, mul_f32_avx2,
    mul_f32_neon, mul_f32_scalar, relu_f32, relu_f32_avx2, relu_f32_neon, relu_f32_scalar,
    scalar_mul_f32, scalar_mul_f32_avx2, scalar_mul_f32_neon, scalar_mul_f32_scalar, silu_f32,
    silu_f32_avx2, silu_f32_neon, silu_f32_scalar,
};

// Re-exports for SIMD RoPE API.
pub use simd_rope::{rope_apply, rope_reference, ROPE_CHUNK_SIZE};

// Re-exports for SIMD transpose API.
pub use simd_transpose::{transpose_2d, transpose_reference};

// Re-exports for SIMD RMSNorm API.
pub use simd_rmsnorm::{rmsnorm, rmsnorm_batch, rmsnorm_reference};

// Re-exports for SIMD pooling API.
pub use simd_pooling::{
    avg_pool1d, avg_pool1d_reference, avg_pool2d, avg_pool2d_reference, max_pool1d,
    max_pool1d_reference, max_pool2d, max_pool2d_reference,
};

// Re-exports for SIMD quantize API.
pub use simd_quantize::{
    dequantize_i8_to_f32, dequantize_i8_to_f32_reference, quantize_f32_to_i8,
    quantize_f32_to_i8_reference, quantize_per_channel, quantize_per_channel_reference,
};

// Re-exports for SIMD cast API.
pub use simd_cast::{
    bf16_to_f32, bf16_to_f32_avx2, bf16_to_f32_neon, bf16_to_f32_scalar, f16_to_f32,
    f16_to_f32_avx2, f16_to_f32_neon, f16_to_f32_scalar, f32_to_bf16, f32_to_bf16_avx2,
    f32_to_bf16_neon, f32_to_bf16_scalar, f32_to_f16, f32_to_f16_avx2, f32_to_f16_neon,
    f32_to_f16_scalar,
};

// Re-exports for SIMD linear API.
pub use simd_linear::{
    linear, linear_batched, linear_batched_reference, linear_no_bias, linear_no_bias_reference,
    linear_reference,
};

// Re-exports for SIMD conv2d API.
pub use simd_conv2d::{conv2d, conv2d_reference, Conv2dError};

// Re-exports for SIMD SDPA API.
pub use simd_sdpa::{sdpa, sdpa_reference, SdpaError};

// Re-exports for SIMD normalize API.
pub use simd_normalize::{
    l1_normalize, l1_normalize_avx2, l1_normalize_neon, l1_normalize_reference,
    l1_normalize_scalar, l2_normalize, l2_normalize_avx2, l2_normalize_neon,
    l2_normalize_reference, l2_normalize_scalar, min_max_normalize, min_max_normalize_avx2,
    min_max_normalize_neon, min_max_normalize_reference, min_max_normalize_scalar,
};

// Re-exports for SIMD reduce API.
pub use simd_reduce::{
    dot_f32 as simd_dot_f32, dot_f32_avx2 as simd_dot_f32_avx2, dot_f32_neon as simd_dot_f32_neon,
    dot_f32_scalar as simd_dot_f32_scalar, l2_norm_f32, l2_norm_f32_avx2, l2_norm_f32_neon,
    l2_norm_f32_scalar, max_f32 as simd_max_f32, max_f32_avx2 as simd_max_f32_avx2,
    max_f32_neon as simd_max_f32_neon, max_f32_scalar as simd_max_f32_scalar,
    min_f32 as simd_min_f32, min_f32_avx2 as simd_min_f32_avx2, min_f32_neon as simd_min_f32_neon,
    min_f32_scalar as simd_min_f32_scalar, sum_f32 as simd_sum_f32,
    sum_f32_avx2 as simd_sum_f32_avx2, sum_f32_neon as simd_sum_f32_neon,
    sum_f32_scalar as simd_sum_f32_scalar,
};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "simd_gather_tests.rs"]
mod simd_gather_tests;

#[cfg(test)]
#[path = "simd_kernel_extended_tests.rs"]
mod simd_kernel_extended_tests;

#[cfg(test)]
#[path = "simd_backend_extended_tests.rs"]
mod simd_backend_extended_tests;

#[cfg(test)]
#[path = "cpu_backend_extended_tests.rs"]
mod cpu_backend_extended_tests;
