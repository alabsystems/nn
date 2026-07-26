// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal GPU backend for [`DynTensor`] operations.
//!
//! Implements [`GpuBackend`] so that `DynTensor` operations dispatch to Metal
//! when the tensor lives on a GPU device. Registration is one-shot via
//! [`register_metal_backend`] (typically called at application startup).
//!
//! **Phase 2 (#1022):** All binary ops (Add, Sub, Mul, Div), all unary ops
//! (Relu, Gelu, Sigmoid, Tanh, Silu, Exp, Sqrt, Sqr, Abs, Neg, Recip, Sin,
//! Cos, Log), all reduce ops (Sum, Mean, Max, Min), and MatMul dispatch
//! directly on GPU via `execute_tensor_dispatch_to_buffer` with
//! `DispatchInput::Gpu`, eliminating CPU round-trips for core tensor ops.
//!
//! See `designs/2026-03-04-dyntensor-metal-phase2.md`.

use nn_core::dyn_tensor::register_gpu_backend;
use nn_core::{Result, TensorError};

use crate::context::MetalContext;
use crate::metal_backend::global_metal_context;

#[path = "dyn_tensor_metal_helpers.rs"]
mod helpers;
#[path = "dyn_tensor_metal_kernels.rs"]
mod kernels;

#[path = "dyn_tensor_metal_ops.rs"]
mod ops;
#[path = "dyn_tensor_metal_ops_compare.rs"]
mod ops_compare;
#[path = "dyn_tensor_metal_ops_reduce.rs"]
mod ops_reduce;
#[path = "dyn_tensor_metal_ops_reduce_compensated.rs"]
mod ops_reduce_compensated;

#[path = "dyn_tensor_metal_adain_fused.rs"]
mod adain_fused;
#[path = "dyn_tensor_metal_adaln_fused.rs"]
mod adaln_fused;
#[path = "dyn_tensor_metal_batch_norm_fused.rs"]
mod batch_norm_fused;
#[path = "dyn_tensor_metal_argreduce_ops.rs"]
mod argreduce_ops;
#[path = "dyn_tensor_metal_cat.rs"]
mod cat_ops;
#[path = "dyn_tensor_metal_conv_gemm.rs"]
mod conv_gemm;
#[path = "dyn_tensor_metal_conv_ops.rs"]
mod conv_ops;
#[path = "dyn_tensor_metal_conv3d_ops.rs"]
mod conv3d_ops;
#[path = "dyn_tensor_metal_cumsum_ops.rs"]
mod cumsum_ops;
#[path = "dyn_tensor_metal_fused_helpers.rs"]
mod fused_helpers;
pub(crate) use cumsum_ops::{
    cumsum_block_scan_msl_source, cumsum_propagate_msl_source, cumsum_scan_block_sums_msl_source,
    cumsum_single_pass_msl_source, CUMSUM_BLOCK_SIZE, CUMSUM_MAX_AXIS,
};
#[path = "dyn_tensor_metal_add_layer_norm.rs"]
mod add_layer_norm;
#[path = "dyn_tensor_metal_cast_dtype.rs"]
mod cast_dtype;
#[path = "dyn_tensor_metal_data_ops.rs"]
mod data_ops;
#[path = "dyn_tensor_metal_flash_attn.rs"]
mod flash_attn;
#[path = "dyn_tensor_metal_sage_attn.rs"]
pub(crate) mod sage_attn;
#[path = "dyn_tensor_metal_group_norm_fused.rs"]
mod group_norm_fused;
#[path = "dyn_tensor_metal_instance_norm_fused.rs"]
mod instance_norm_fused;
#[path = "dyn_tensor_metal_layer_norm_fused.rs"]
mod layer_norm_fused;
#[path = "dyn_tensor_metal_channels_first_ln_fused.rs"]
mod channels_first_ln_fused;
#[path = "dyn_tensor_metal_lstm_ops.rs"]
mod lstm_ops;
#[path = "dyn_tensor_metal_lstm_sequence.rs"]
mod lstm_sequence;
#[path = "dyn_tensor_metal_matmul.rs"]
mod matmul;
#[path = "dyn_tensor_metal_matmul_simd.rs"]
pub(crate) mod matmul_simd;
#[path = "dyn_tensor_metal_quantized_matmul.rs"]
mod quantized_matmul;
pub(crate) use matmul_simd::encode_simdgroup_matmul_into_batch;
pub(crate) use lstm_sequence::dispatch_lstm_precomputed;
pub(crate) use matmul_simd::should_use_f16_simdgroup;
pub(crate) use matmul_simd::should_use_simdgroup;
#[cfg(any(test, kani))]
pub(crate) use matmul_simd::{select_tile_config, GemmTileConfig};
// MSL source re-exports for KernelSpec builders (#3503).
pub(crate) use instance_norm_fused::{instance_norm_msl_source, instance_norm_f16_msl_source};
pub(crate) use layer_norm_fused::{layer_norm_msl_source, layer_norm_f16_msl_source};
pub(crate) use norm_conv_fused::stats_kernel_msl_source;
pub(crate) use add_layer_norm::{add_layer_norm_msl_source, add_layer_norm_f16_msl_source};
pub(crate) use channels_first_ln_fused::{
    channels_first_layer_norm_msl_source, channels_first_layer_norm_f16_msl_source,
    channels_first_ln_leaky_relu_msl_source, channels_first_ln_leaky_relu_f16_msl_source,
};
pub(crate) use adain_fused::{
    adain_snake_msl_source, adain_snake_f16_msl_source,
    adain_leaky_relu_msl_source, adain_leaky_relu_f16_msl_source,
};
pub(crate) use adaln_fused::{ada_layer_norm_msl_source, ada_layer_norm_f16_msl_source};
pub(crate) use batch_norm_fused::{batch_norm_msl_source, batch_norm_f16_msl_source};
pub(crate) use group_norm_fused::{group_norm_msl_source, group_norm_f16_msl_source};
pub(crate) use rms_norm_fused::{rms_norm_msl_source, rms_norm_f16_msl_source};
pub(crate) use snake_fused::{snake_msl_source, snake_f16_msl_source};
pub(crate) use flash_attn::{
    flash_attn_f32_msl_source, flash_attn_f16_msl_source,
    flash_attn_f32_seq_first_msl_source, flash_attn_f16_seq_first_msl_source,
};
#[path = "dyn_tensor_metal_norm_ops.rs"]
mod norm_ops;
#[path = "dyn_tensor_metal_polar_to_rect.rs"]
mod polar_to_rect;
#[path = "dyn_tensor_metal_repeat_interleave_gpu.rs"]
mod repeat_interleave_gpu;
#[path = "dyn_tensor_metal_rms_norm_fused.rs"]
mod rms_norm_fused;
#[path = "dyn_tensor_metal_snake_fused.rs"]
mod snake_fused;
// gpu_prefix_sum_offsets, gpu_scatter_with_offsets, MAX_GPU_PREFIX_SUM
// re-exported via native_bridges below (which delegates to repeat_interleave_gpu).
#[path = "dyn_tensor_metal_resize_bilinear.rs"]
mod resize_bilinear;
#[path = "dyn_tensor_metal_rope_ops.rs"]
mod rope_ops;
#[path = "dyn_tensor_metal_moe_ops.rs"]
mod moe_ops;
#[path = "dyn_tensor_metal_pad_ops.rs"]
mod pad_ops;
#[path = "dyn_tensor_metal_pool_ops.rs"]
mod pool_ops;
#[path = "dyn_tensor_metal_scatter_ops.rs"]
mod scatter_ops;
#[path = "dyn_tensor_metal_select_ops.rs"]
mod select_ops;
#[path = "dyn_tensor_metal_shape_ops.rs"]
mod shape_ops;
#[path = "dyn_tensor_metal_sort_ops.rs"]
mod sort_ops;
#[path = "dyn_tensor_metal_topk_ops.rs"]
mod topk_ops;
#[path = "dyn_tensor_metal_welford_msl.rs"]
pub(crate) mod welford_msl;

#[path = "dyn_tensor_metal_dispatch.rs"]
mod dispatch;
// Re-export with_pipeline_cache so sibling submodules can use it.
pub(crate) use dispatch::with_pipeline_cache;

#[path = "dyn_tensor_metal_backend_impl.rs"]
mod backend_impl;

#[path = "dyn_tensor_metal_norm_conv_fused.rs"]
mod norm_conv_fused;
pub(crate) use norm_conv_fused::ResidualParams;

#[path = "dyn_tensor_metal_upsample_conv1d_fused.rs"]
pub(crate) mod upsample_conv1d_fused;

#[path = "dyn_tensor_metal_norm_conv_stats.rs"]
mod norm_conv_stats;
pub(crate) use norm_conv_stats::PrecomputedStats;
pub(crate) use norm_conv_stats::with_fast_half_scope;

#[path = "dyn_tensor_metal_native_bridges.rs"]
mod native_bridges;
pub(crate) use native_bridges::{
    dispatch_prefix_sum_only, gpu_scatter_with_offsets, native_ada_layer_norm,
    native_adain_leaky_relu, native_adain_snake, native_adain_snake_precise, native_add_layer_norm,
    native_channels_first_layer_norm_with_activation,
    native_flash_attention, native_flash_attention_seq_first, native_instance_norm,
    native_instance_norm_precise, native_layer_norm, native_lstm_sequence,
    native_lstm_sequence_reverse, native_max_pool1d, native_norm_activ_conv1d,
    native_norm_activ_conv1d_snake,
    native_norm_activ_conv1d_snake_with_output_stats,
    native_norm_activ_conv1d_snake_with_precomputed_stats,
    native_norm_activ_conv1d_with_output_stats, native_norm_activ_conv1d_with_precomputed_stats,
    native_conv1d, native_conv1d_gemm, native_fused_upsample_conv1d,
    native_batch_norm_2d,
    read_prefix_sum_total, MAX_GPU_PREFIX_SUM,
};
#[cfg(test)]
pub(crate) use native_bridges::gpu_polar_to_rect;

#[path = "dyn_tensor_metal_storage.rs"]
mod storage;
pub use storage::MetalTensorData;

// -- Backend implementation ---------------------------------------------------

/// Metal GPU backend for `DynTensor` dispatch.
///
/// Stateless — all state comes from the global `MetalContext` (initialized
/// via `MetalBackend::init()`). The struct exists to implement the `GpuBackend`
/// trait, which is stored as `Box<dyn GpuBackend>` in the global registry.
///
/// A thread-local `PipelineCache` caches compiled Metal pipelines across calls.
/// See `dyn_tensor_metal_dispatch.rs` for the `dispatch_def*` methods.
pub(crate) struct MetalDynBackend;

impl MetalDynBackend {
    /// Get the global Metal context or convert error to TensorError.
    fn ctx() -> Result<&'static MetalContext> {
        global_metal_context().map_err(|e| {
            TensorError::backend_failure(
                nn_core::BackendDomain::Metal,
                nn_core::BackendErrorKind::Other,
                e.to_string(),
            )
        })
    }
}

// -- Registration -------------------------------------------------------------

/// Register the Metal backend for `DynTensor` GPU dispatch.
///
/// Must be called after `MetalBackend::init()` (which initializes the global
/// Metal context). Subsequent calls are no-ops (`OnceLock` semantics).
///
/// # Example
///
/// ```no_run
/// use nn_metal::MetalBackend;
/// use nn_metal::register_metal_dyn_backend;
///
/// MetalBackend::init().expect("Metal init");
/// register_metal_dyn_backend();
/// ```
pub fn register_metal_dyn_backend() {
    register_gpu_backend(Box::new(MetalDynBackend));
}

/// Collect all native kernel MSL sources for pre-compilation to `.metallib`.
///
/// Returns `(entry_point_name, msl_source)` pairs for all fixed-name Metal
/// kernels used by the Kokoro pipeline (and general DynTensor dispatch).
/// These kernels have static MSL — their source is the same regardless of
/// input shapes or model dimensions.
///
/// Used by `precompile::collect_native_kernel_sources()`.
pub(crate) fn collect_native_msl_sources() -> Vec<(&'static str, String)> {
    let mut sources = vec![
        // BatchNorm fused (F32 + F16) — #4324
        (
            "fused_batch_norm_float",
            batch_norm_msl_source(),
        ),
        (
            "fused_batch_norm_half",
            batch_norm_f16_msl_source(),
        ),
        // AdaIN fused (F32 + F16)
        (
            "fused_adain_snake_float",
            adain_snake_msl_source(),
        ),
        (
            "fused_adain_snake_half",
            adain_snake_f16_msl_source(),
        ),
        (
            "fused_adain_leaky_relu_float",
            adain_leaky_relu_msl_source(),
        ),
        (
            "fused_adain_leaky_relu_half",
            adain_leaky_relu_f16_msl_source(),
        ),
        // AdaLayerNorm fused (F32 + F16)
        (
            "fused_ada_layer_norm_float",
            ada_layer_norm_msl_source(),
        ),
        (
            "fused_ada_layer_norm_half",
            ada_layer_norm_f16_msl_source(),
        ),
        // InstanceNorm fused (F32 + F16)
        (
            "fused_instance_norm_float",
            instance_norm_msl_source(),
        ),
        (
            "fused_instance_norm_half",
            instance_norm_f16_msl_source(),
        ),
        // GroupNorm fused (F32 + F16)
        (
            "fused_group_norm_float",
            group_norm_msl_source(),
        ),
        (
            "fused_group_norm_half",
            group_norm_f16_msl_source(),
        ),
        // LayerNorm fused (F32 + F16)
        (
            "fused_layer_norm_float",
            layer_norm_msl_source(),
        ),
        (
            "fused_layer_norm_half",
            layer_norm_f16_msl_source(),
        ),
        // Channels-first LayerNorm fused (F32 + F16) — #3457
        (
            "fused_channels_first_layer_norm_float",
            channels_first_layer_norm_msl_source(),
        ),
        (
            "fused_channels_first_layer_norm_half",
            channels_first_layer_norm_f16_msl_source(),
        ),
        // RmsNorm fused (F32 + F16)
        (
            "fused_rms_norm_float",
            rms_norm_msl_source(),
        ),
        (
            "fused_rms_norm_half",
            rms_norm_f16_msl_source(),
        ),
        // Snake fused (F32 + F16)
        ("fused_snake_float", snake_msl_source()),
        ("fused_snake_half", snake_f16_msl_source()),
        // Flash attention (F32 + F16, HeadsFirst + SeqFirst)
        (
            "flash_attn_f32",
            flash_attn_f32_msl_source().to_owned(),
        ),
        (
            "flash_attn_f16",
            flash_attn_f16_msl_source().to_owned(),
        ),
        (
            "flash_attn_f32_seq_first",
            flash_attn_f32_seq_first_msl_source().to_owned(),
        ),
        (
            "flash_attn_f16_seq_first",
            flash_attn_f16_seq_first_msl_source().to_owned(),
        ),
        // SIMD GEMM: 32x32 tiles (F32 + F16)
        (
            "simd_gemm_f32",
            matmul_simd::simd_gemm_f32_msl_source().to_owned(),
        ),
        (
            "simd_gemm_f16",
            matmul_simd::simd_gemm_f16_msl_source().to_owned(),
        ),
        // SIMD GEMM: 64x64 tiles (#3479, F32 + F16)
        (
            "simd_gemm_64_f32",
            matmul_simd::simd_gemm_64_f32_msl_source().to_owned(),
        ),
        (
            "simd_gemm_64_f16",
            matmul_simd::simd_gemm_64_f16_msl_source().to_owned(),
        ),
        // Cumsum + polar-to-rect (F32 only — cumsum uses f64 accumulator)
        ("cumsum_f32", cumsum_single_pass_msl_source()),
        ("cumsum_propagate", cumsum_propagate_msl_source().to_owned()),
        (
            "fused_polar_to_rect_f32",
            polar_to_rect::polar_to_rect_msl_source().to_owned(),
        ),
    ];
    // NormActivConv1d fused (stats + leaky_relu + snake, each F32 + F16)
    sources.extend(norm_conv_fused::collect_msl_sources());
    // Conv-with-output-stats variants (#1815 Tier 2)
    sources.extend(norm_conv_stats::collect_msl_sources());
    sources
}

#[cfg(test)]
#[path = "dyn_tensor_metal_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_tests_validation.rs"]
mod tests_validation;

#[cfg(test)]
#[path = "dyn_tensor_metal_nn_tests.rs"]
mod nn_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_shape_ops_tests.rs"]
mod shape_ops_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_shape_ops_narrow_view_tests.rs"]
mod shape_ops_narrow_view_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_narrow_view_kernel_tests.rs"]
mod narrow_view_kernel_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_shape_ops_softmax_tests.rs"]
mod shape_ops_softmax_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_shape_ops_softmax_tests_dtype.rs"]
mod shape_ops_softmax_tests_dtype;

#[cfg(test)]
#[path = "dyn_tensor_metal_shape_ops_index_tests.rs"]
mod shape_ops_index_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_ops_tests.rs"]
mod ops_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_reduce_tests.rs"]
mod reduce_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_reduce_extended_tests.rs"]
mod reduce_extended_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_data_ops_tests.rs"]
mod data_ops_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_index_add_tests.rs"]
mod index_add_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_with_dtype_tests.rs"]
mod with_dtype_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_rope_tests.rs"]
mod rope_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_argreduce_tests.rs"]
mod argreduce_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_cumsum_tests.rs"]
mod cumsum_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_triu_tril_tests.rs"]
mod triu_tril_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_norm_tests.rs"]
mod norm_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_norm_tests_edge.rs"]
mod norm_tests_edge;

#[cfg(test)]
#[path = "dyn_tensor_metal_norm_tests_albert.rs"]
mod norm_tests_albert;

#[cfg(test)]
#[path = "dyn_tensor_metal_norm_tests_adain_snake.rs"]
mod norm_tests_adain_snake;

#[cfg(test)]
#[path = "compiled_model_execute_native_fused_adain_direct_tests.rs"]
mod fused_adain_snake_direct_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_norm_tests_precision.rs"]
mod norm_tests_precision;

#[cfg(test)]
#[path = "dyn_tensor_metal_norm_tests_adain_leaky_relu.rs"]
mod norm_tests_adain_leaky_relu;

#[cfg(test)]
#[path = "dyn_tensor_metal_snake_tests.rs"]
mod snake_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_conv2d_tests.rs"]
mod conv2d_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_comparison_tests.rs"]
mod comparison_tests;
#[cfg(all(test, feature = "bench"))]
#[path = "dyn_tensor_metal_dispatch_bench.rs"]
mod dispatch_bench;
#[cfg(all(test, feature = "bench"))]
#[path = "dyn_tensor_metal_matmul_bench.rs"]
mod matmul_bench;
#[cfg(test)]
#[path = "dyn_tensor_metal_matmul_simd_tests.rs"]
mod matmul_simd_tests;
#[cfg(test)]
#[path = "dyn_tensor_metal_matmul_tests.rs"]
mod matmul_tests;
#[cfg(test)]
#[path = "dyn_tensor_metal_pool_tests.rs"]
mod pool_tests;
#[cfg(test)]
#[path = "dyn_tensor_metal_upsample_tests.rs"]
mod upsample_tests;
#[cfg(test)]
#[path = "dyn_tensor_metal_resize_bilinear_tests.rs"]
mod resize_bilinear_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_shape_ops_conv_tests.rs"]
mod shape_ops_conv_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_shape_ops_validation_tests.rs"]
mod shape_ops_validation_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_topk_tests.rs"]
mod topk_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_lstm_tests.rs"]
mod lstm_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_rope_error_tests.rs"]
mod rope_error_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_lstm_error_tests.rs"]
mod lstm_error_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_ops_validation_extended_tests.rs"]
mod ops_validation_extended_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_bf16_integration_tests.rs"]
mod bf16_integration_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_lstm_sequence_tests.rs"]
mod lstm_sequence_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_compensated_reduce_tests.rs"]
mod compensated_reduce_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_unfold_tests.rs"]
mod unfold_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_moe_tests.rs"]
mod moe_tests;

#[cfg(test)]
#[path = "dyn_tensor_metal_batch_norm_tests.rs"]
mod batch_norm_tests;
