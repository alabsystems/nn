// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal backend for nn: Apple Silicon GPU inference via Metal Shading Language.
//!
//! This crate provides the GPU compute layer for nn on Apple platforms. It is
//! consumed by dvoice via `nn::metal::*` re-exports (feature-gated).
//!
//! # Major Subsystems
//!
//! - **GPU DynTensor backend** — [`MetalBackend`] registers with `nn-core`'s
//!   `GpuBackend` trait, providing Metal dispatch for matmul, conv, softmax,
//!   normalization, RoPE, topk, and element-wise ops.
//! - **Model inference** — [`HTDemucs`] (source separation) and [`SileroVad`]
//!   (voice activity detection) with persistent GPU weight buffers and full
//!   CPU/GPU forward paths.
//! - **Fused GPU kernels** — Single-dispatch Metal kernels for LayerNorm,
//!   RmsNorm, GroupNorm, InstanceNorm, Snake, AdaIN+Snake, LSTM cell/sequence,
//!   RoPE, and iSTFT via `TensorBlockBuilder` IR graphs compiled to MSL.
//! - **Lazy command buffer batching** — Always-on GPU command batching via
//!   `begin_batch()` / `commit_and_wait()`. Reduces Metal API overhead.
//! - **Weight loading** — [`WeightMap`] for mmap-backed safetensors,
//!   [`SafeTensorsBackend`] for heap-backed loading, and [`PipelineCache`]
//!   with LRU eviction for compiled MSL pipelines.
//! - **bf16/f16 dispatch** — [`execute_tensor_dispatch`] is generic over
//!   [`MetalElement`] (f32, f16, bf16) with MSL codegen emitting correct
//!   scalar types per `ScalarType`.
//! - **Signal processing** — STFT magnitude decomposition
//!   ([`compute_stft_magnitude`]) and GPU-accelerated iSTFT
//!   ([`IstftBasis`]) for audio model pipelines.
//! - **Compiled model executor** — [`compiled_model::CompiledModel`] compiles
//!   a traced `ComputationGraph` into pre-built dispatch plans for repeated
//!   GPU execution with zero IR rebuild overhead.
//! - **Arena allocation** — [`ActivationArena`] (`with_arena()`) reuses GPU
//!   buffers across forward passes, reducing allocation overhead.
//! - **Weight surgery** — [`live_edit`] for GPU-resident weight editing,
//!   [`lora_overlay`] for GPU-resident LoRA overlays, and [`weight_edit`]
//!   for the weight editing pipeline.
//! - **GPU buffer slicing** — [`GpuSlice`] newtype for `(MetalBuffer, byte_offset)`
//!   pairs, supporting arena-allocated interior buffer references.
//!
//! # GPU Dispatch Architecture
//!
//! Two frontends share a common executor (`dispatch_inner` in `tensor_dispatch.rs`):
//!
//! - **Pre-built path** (SileroVad, HTDemucs): `TensorKernelDef` IR built once
//!   at model construction via `TensorBlockBuilder`, dispatched repeatedly at
//!   inference time. Used for models with fixed shapes and persistent GPU weights.
//! - **DynTensor path** (Whisper, Qwen3, nn layers): IR built per-op through
//!   `GpuBackend` trait methods, cached by [`kernel_def_cache::KernelDefCache`]
//!   to eliminate redundant IR construction on repeat calls with the same shapes.
//!
//! Both produce identical MSL and use the same [`PipelineCache`] for compiled
//! Metal pipelines. Design: `designs/2026-03-07-gpu-dispatch-unification.md`.

pub mod arena;
mod blit_copy_analysis;
pub(crate) mod buffer;
pub(crate) mod cache;
pub mod cache_stats;
pub mod compiled_model;
pub(crate) mod context;
/// Demucs shared helpers re-exported from nn-models (backend-agnostic).
pub(crate) use nn_models::demucs_shared;
#[allow(unreachable_pub)]
pub(crate) mod demucs_spectral_decoder;
#[allow(unreachable_pub)]
pub(crate) mod demucs_spectral_encoder;
#[allow(unreachable_pub)]
pub(crate) mod demucs_temporal_decoder;
#[allow(unreachable_pub)]
pub(crate) mod demucs_temporal_encoder;
#[allow(unreachable_pub)]
pub(crate) mod demucs_transformer;
pub(crate) mod dispatch;
pub(crate) mod dispatch_plan;
pub(crate) mod dispatch_plan_profiler;
pub mod dispatch_profiler;
pub(crate) mod dispatch_stats;
pub mod dpdf_image_preprocess_metal;
pub mod dpdf_pipeline_metal;
pub(crate) mod dyn_tensor_metal;
pub(crate) mod element;
pub(crate) mod error;
pub mod gpu_audio_handle;
pub mod gpu_fence;
pub mod gpu_future;
pub(crate) mod gpu_scope;
pub mod gpu_slice;
pub(crate) mod htdemucs;
/// STFT/iSTFT re-exported from nn-models (backend-agnostic signal processing).
pub use nn_models::istft;
pub mod compiled_kokoro;
pub(crate) mod istft_gpu;
pub mod kokoro_audio_gpu;
pub mod segment_cache;
pub mod shared_segment_store;
pub(crate) mod stft_gpu;
pub(crate) use nn_models::stft;
#[cfg(kani)]
mod kani_buffer_capacity;
#[cfg(kani)]
mod kani_buffer_planner;
#[cfg(kani)]
mod kani_channels_first_ln;
#[cfg(kani)]
mod kani_dispatch_plan;
#[cfg(kani)]
mod kani_dispatch_plan_builder;
#[cfg(kani)]
mod kani_dispatch_plan_extra;
#[cfg(kani)]
mod kani_dispatch_plan_ordering;
#[cfg(kani)]
mod kani_fused_resblock;
#[cfg(kani)]
mod kani_fused_resblock_routing;
#[cfg(kani)]
mod kani_compiled_kokoro_dispatch;
#[cfg(kani)]
mod kani_gemm_tile_select;
#[cfg(kani)]
mod kani_buffer_safety;
#[cfg(kani)]
mod kani_fused_kernel_helpers;
#[cfg(kani)]
mod kani_native_encoding_plan;
#[cfg(kani)]
mod kani_trace_compiler;
#[cfg(kani)]
mod kani_compiled_execute_proofs;
#[cfg(kani)]
mod kani_matmul_lstm_gpu_proofs;
#[cfg(kani)]
mod kani_istft_arena_scope;
#[cfg(kani)]
mod kani_kernel_spec_gemm_resblock;
#[cfg(kani)]
mod kani_dyn_tensor_metal_bridges;
#[cfg(kani)]
mod kani_compiled_kokoro_chorus;
#[cfg(kani)]
mod kani_compiled_kokoro_segments;
#[cfg(kani)]
mod kani_norm_conv_stats;
#[cfg(kani)]
mod kani_kernel_spec_norm_ops;
#[cfg(kani)]
mod kani_dispatch_arena_dyn;
#[cfg(kani)]
mod kani_compiled_model_execute_native_simple;
#[cfg(kani)]
mod kani_compiled_model_execute_native;
#[cfg(kani)]
mod kani_compiled_model_execute_native_step_bounds;
#[cfg(kani)]
mod kani_dyn_tensor_metal_matmul_simd_msl;
#[cfg(kani)]
mod kani_compiled_model_kernel_spec_gemm;
#[cfg(kani)]
mod kani_compiled_model_kernel_spec_gemm_tile_alignment;
#[cfg(kani)]
mod kani_dyn_tensor_metal_lstm_sequence;
#[cfg(kani)]
mod kani_dyn_tensor_metal_lstm_sequence_bounds;
#[cfg(kani)]
mod kani_istft_gpu;
#[cfg(kani)]
mod kani_native_resblock_dispatch;
#[cfg(kani)]
mod kani_dyn_tensor_metal_module;
#[cfg(kani)]
mod kani_native_bridges;
#[cfg(kani)]
mod kani_arena_scope;
#[cfg(kani)]
mod kani_execute_native_resblock;
#[cfg(kani)]
mod kani_kokoro_gpu_synth;
#[cfg(kani)]
mod kani_compiled_model_builder;
#[cfg(kani)]
mod kani_execute_native_fused_batched;
#[cfg(kani)]
mod kani_matmul_simd_msl_extended;
#[cfg(kani)]
mod kani_native_simple_extended;
#[cfg(kani)]
mod kani_lstm_sequence_extended;
#[cfg(kani)]
mod kani_istft_gpu_extended;
#[cfg(kani)]
mod kani_kernel_spec_gemm_extended;
#[cfg(kani)]
mod kani_execute_native_resblock_extended;
#[cfg(kani)]
mod kani_dispatch_helpers;
#[cfg(kani)]
#[path = "kani_weight_edit.rs"]
mod kani_weight_edit;
#[cfg(kani)]
#[path = "kani_compiled_kokoro_config.rs"]
mod kani_compiled_kokoro_config;
#[cfg(kani)]
#[path = "kani_dpdf_preprocess_proofs.rs"]
mod kani_dpdf_preprocess_proofs;
#[cfg(kani)]
mod kani_stats_dispatch_proofs;
#[cfg(kani)]
#[path = "kani_dispatch_safety_proofs.rs"]
mod kani_dispatch_safety_proofs;
#[cfg(kani)]
#[path = "kani_kernel_source_proofs.rs"]
mod kani_kernel_source_proofs;
#[cfg(kani)]
#[path = "kani_tile_config_proofs.rs"]
mod kani_tile_config_proofs;
#[cfg(kani)]
#[path = "kani_dispatch_plan_builder_proofs.rs"]
mod kani_dispatch_plan_builder_proofs;
#[cfg(kani)]
#[path = "kani_kernel_spec_binding_proofs.rs"]
mod kani_kernel_spec_binding_proofs;
#[cfg(kani)]
#[path = "kani_lib_helpers_proofs.rs"]
mod kani_lib_helpers_proofs;
#[cfg(kani)]
#[path = "kani_gpu_fence.rs"]
mod kani_gpu_fence;
#[cfg(kani)]
#[path = "kani_segment_cache_eviction.rs"]
mod kani_segment_cache_eviction;
#[cfg(kani)]
#[path = "kani_compiled_kokoro_segment_budget.rs"]
mod kani_compiled_kokoro_segment_budget;
#[cfg(kani)]
#[path = "kani_crossfade_blend.rs"]
mod kani_crossfade_blend;
#[cfg(kani)]
#[path = "kani_compiled_model_builder_proofs.rs"]
mod kani_compiled_model_builder_proofs;
#[cfg(kani)]
#[path = "kani_streaming_chorus_proofs.rs"]
mod kani_streaming_chorus_proofs;
#[cfg(kani)]
#[path = "kani_gpu_audio_handle.rs"]
mod kani_gpu_audio_handle;
#[cfg(kani)]
mod kani_arena_reuse_proofs;
#[cfg(kani)]
#[path = "kani_kernel_def_cache_proofs.rs"]
mod kani_kernel_def_cache_proofs;
#[cfg(kani)]
#[path = "kani_weight_map_safety.rs"]
mod kani_weight_map_safety;
#[cfg(kani)]
#[path = "kani_metal_context_proofs.rs"]
mod kani_metal_context_proofs;
#[cfg(kani)]
#[path = "kani_pipeline_cache_proofs.rs"]
mod kani_pipeline_cache_proofs;
#[cfg(kani)]
#[path = "kani_compute_dispatch_proofs.rs"]
mod kani_compute_dispatch_proofs;
#[cfg(kani)]
#[path = "kani_compiled_model_execute_fused.rs"]
mod kani_compiled_model_execute_fused;
#[cfg(kani)]
#[path = "kani_msl_codegen_cache_proofs.rs"]
mod kani_msl_codegen_cache_proofs;
#[cfg(kani)]
#[path = "kani_streaming_kokoro_session_proofs.rs"]
mod kani_streaming_kokoro_session_proofs;
#[cfg(kani)]
#[path = "kani_safetensors_backend_proofs.rs"]
mod kani_safetensors_backend_proofs;
#[cfg(kani)]
#[path = "kani_arena_buffer_planner_proofs.rs"]
mod kani_arena_buffer_planner_proofs;
#[cfg(kani)]
#[path = "kani_simdgroup_tile_extended.rs"]
mod kani_simdgroup_tile_extended;
#[cfg(kani)]
#[path = "kani_gpu_weight_cache_proofs.rs"]
mod kani_gpu_weight_cache_proofs;
pub(crate) mod kernel_def_cache;
#[cfg(all(test, feature = "bench"))]
#[path = "kernel_def_cache_bench.rs"]
mod kernel_def_cache_bench;
pub(crate) mod kernel_dispatch;
pub(crate) mod kernel_source;
pub mod live_edit;
pub mod lora_overlay;
pub mod memory_profiler;
pub mod compiled_model_memory_report;
pub mod compiled_model_optimizer_report;
pub(crate) mod metal_backend;
pub mod metallib_loader;
pub(crate) mod msl_codegen_cache;
pub mod optimization_report;
pub mod optimization_report_diff;
pub(crate) mod pipeline;
pub mod precompile;
pub mod rss;
pub(crate) mod safetensors;
pub(crate) mod silero_vad;
pub(crate) mod simdgroup_tile_select;
pub(crate) mod tensor_dispatch;
pub(crate) mod var_builder_safetensors;
pub mod weight_edit;

pub use arena::{
    arena_capacity, arena_stats, default_arena_total_growth_count, ensure_default_arena_capacity,
    estimate_arena_peak_bytes, estimate_arena_peak_from_shapes, reset_arena_stats,
    try_reset_active_arena, with_arena, with_decode_scope, without_arena, ActivationArena,
    ArenaEstimate, ArenaStats, PoolStats,
};
pub use buffer::MetalBuffer;
pub use cache::{precompiled_pipeline_count, PipelineCache};
pub use cache_stats::{CacheStats, CacheStatsSnapshot};
pub use context::MetalContext;
pub use gpu_slice::GpuSlice;
// Demucs weight types are public because HTDemucsWeights has pub fields
// referencing them. These are defined in nn-models and re-exported here
// for consumer convenience.
pub use demucs_spectral_decoder::DemucsSpectralDecoderWeights;
pub use demucs_spectral_encoder::DemucsSpectralEncoderWeights;
pub use demucs_temporal_decoder::DemucsTemporalDecoderWeights;
pub use demucs_temporal_encoder::DemucsTemporalEncoderWeights;
pub use demucs_transformer::DemucsTransformerWeights;
// Advanced/extension API — for custom kernel authoring and low-level Metal
// dispatch. Most consumers should use model-level APIs (HTDemucs, SileroVad)
// or the DynTensor GPU backend (register_metal_dyn_backend) instead.
pub use compiled_kokoro::chorus::KokoroChorus;
pub use compiled_kokoro::gpu_synth::{ChorusGpuSynth, GpuSynth};
// Re-export ChorusConfig alongside KokoroChorus — KokoroChorus::new() takes ChorusConfig.
pub use nn_models::kokoro_chorus::ChorusConfig;
// Re-export HumanizeConfig — KokoroChorus::with_humanize() takes HumanizeConfig.
pub use nn_models::kokoro_chorus_humanize::HumanizeConfig;
pub use compiled_kokoro::precompile::{
    precompile_kokoro_msl, PrecompileResult, PrecompileShapes, SegmentPeepholeConfigs,
};
#[cfg(feature = "plan-serde")]
pub use compiled_kokoro::precompile::OptimizerWarmupResult;
pub use compiled_kokoro::{
    BatchStats, BottleneckKind, ChannelStreamingSession, ChorusChunkMode, CompiledKokoro,
    CompiledKokoroError, CompiledKokoroStreamingSession, Conv1dBatchGroup, ConvBatchAnalysis,
    ConvBatchOptimizer, DiagnosticOutput, DispatchCensus, DispatchSummary, EncoderBatchPlanner,
    EncoderGroup, F16AutocastConfig, FusedGroup, FusionPlan, GpuTimingReport, KokoroArenaReport,
    auto_precision_config, format_precision_report, AutoPrecisionResult, SegmentPrecisionDecision,
    LazyBufferPool, LazyPoolStats, MemoryBreakdown, PipelineConvBatchSummary, PipelineMode,
    PipelineProfile, PipelineTensor, SegmentCensus, SegmentFusionPlanner, SegmentGapAnalysis,
    RtfBottleneck, RtfOptimizer, RtfReport, SegmentInfo, SegmentOptimizerResult, SegmentProfile,
    SegmentReport, SharedSegmentCache, StepEncodeResult,
    StepF0EnergyResult, StepGeneratorResult, StepProsodyResult, StepRegulateResult, StreamChunk,
    StreamReceiver, StreamingChorusSession, StreamingKokoroSession, StyleSplit,
    SynthesisIntermediates, TimingReport, TypedEncodeResult, TypedF0EnergyResult,
    TypedGeneratorResult, TypedProsodyResult, TypedRegulateResult, can_fuse,
    format_profile_report, identify_bottleneck, plan_segment_fusion,
};
#[cfg(feature = "plan-serde")]
pub use compiled_kokoro::{load_peephole_configs, save_peephole_configs};
#[cfg(feature = "plan-serde")]
pub use compiled_kokoro::{
    load_optimal_configs, load_optimal_configs_if_exists, save_optimal_configs,
    KokoroOptimalConfigs, SegmentOptimalConfig,
};
pub use segment_cache::{EvictionPolicy, SegmentCacheConfig, SegmentCacheStats, ShapeKeyedCache};
pub use shared_segment_store::{SegmentKey, SharedSegmentStats, SharedSegmentStore};
pub use dispatch::{BatchEncoder, CommandBatch, ComputeDispatch, PendingBatch};
pub use dispatch_plan::{DispatchMode, DispatchPlan};
pub use dispatch_profiler::{
    DispatchProfileEntry, DispatchProfileReport, DispatchProfiler, DispatchType,
    FusionOpportunity, TopEntry, TypeBreakdown,
};
pub use dispatch_stats::{dispatch_stats, reset_counters, DispatchStats};
pub use dpdf_image_preprocess_metal::DpdfImagePreprocessMetal;
pub use dpdf_pipeline_metal::DpdfPipelineMetal;
pub use dyn_tensor_metal::{register_metal_dyn_backend, MetalTensorData};
pub use element::MetalElement;
pub use error::MetalError;
pub use gpu_audio_handle::GpuAudioHandle;
pub use gpu_fence::GpuFence;
pub use gpu_future::{AsyncGpuResult, GpuFuture};
pub use gpu_scope::{
    encode_custom_dispatch, flush, submit, sync, with_gpu_scope, with_scope_exit_mode,
    ScopeExitMode,
};
pub use htdemucs::{HTDemucs, HTDemucsError, HTDemucsWeights, WeightLoadError};
pub use istft::{IstftBasis, IstftError, IstftParams};
pub use istft_gpu::IstftGpuBasis;
pub use kernel_dispatch::{BufferAccess, BufferBinding, KernelPipeline};
pub use kernel_source::KernelSource;
pub use kokoro_audio_gpu::kokoro_forward_audio_gpu;
pub use live_edit::{ApplyReceipt, DeltaApplyReceipt, LiveEditApply, LiveEditError};
pub use lora_overlay::LoraGpuOverlay;
pub use metal_backend::{
    from_metal_buffer, MetalBackend, MetalInitOptions, MetalTensorExt, MetalTensorStorage,
    RUNTIME_METALLIB_ENV_GUARD,
};
pub use nn_models::kokoro_tts::KokoroConfig;
pub use nn_tts_verify::Certificate;
pub use optimization_report::{ContractStatus, OptimizationReport, ReportError};
pub use optimization_report_diff::{diff_reports, IterationVerdict, ReportDelta};
pub use pipeline::ComputePipeline;
pub use rss::{
    metal_allocated_bytes, metal_allocated_mb, metal_budget_bytes, rss_bytes, rss_mb, RssSnapshot,
    RssTracker,
};
pub use safetensors::{TensorInfo, WeightError, WeightMap};
pub use silero_vad::{
    SegmentConfig, SileroVad, SileroVadError, SileroVadOutput, SileroVadState, SileroVadWeights,
    SpeechSegment,
};
pub use stft::{compute_stft_magnitude, StftError, StftParams};
pub use tensor_dispatch::{
    execute_tensor_dispatch, execute_tensor_dispatch_batched, execute_tensor_dispatch_readback,
    execute_tensor_dispatch_to_buffer, execute_tensor_dispatch_to_buffer_with_contract,
    DispatchInput, TensorDispatchError,
};
// Precision types needed by execute_tensor_dispatch_to_buffer_with_contract consumers.
pub use nn_dsl::{PrecisionContract, PrecisionTier};
pub use var_builder_safetensors::{
    from_mmaped_safetensors, from_mmaped_safetensors_with_ctx, var_builder_from_weight_map,
    MetalVarBuilderExt, SafeTensorsBackend,
};
pub use to_standalone::to_standalone;
pub use memory_profiler::{
    BufferCategory, GpuMemoryProfiler, GpuMemorySnapshot, MemoryBreakdownByCategory,
};
pub use compiled_model_memory_report::{
    bytes_to_human, format_memory_report, MemoryReport, StepMemoryReport,
};
pub use compiled_model_optimizer_report::{
    diff_peephole_configs, format_optimizer_report,
    generate_optimizer_report_with_metrics, OptimizerReport,
};
pub use buffer_pool_size_class::{
    AllocResult, BufferPoolSizeClassStats, SizeClassAllocator, SizeClassStats,
};
pub use weight_edit::{apply_weight_edit, WeightEditError, WeightEditResult, WeightEditSpec};

/// Count non-finite (NaN or Inf) values in a float slice — single pass.
///
/// Replaces the double-scan pattern `if data.iter().any(|v| !v.is_finite()) { count = ... }`
/// with a single `filter().count()` pass. Callers check `if count > 0 { ... }`.
pub(crate) fn count_non_finite(data: &[f32]) -> usize {
    data.iter().filter(|v| !v.is_finite()).count()
}

/// Build a [`TensorKernelDef`] from a builder, mapping errors to [`TensorError::InvalidShape`].
///
/// Replaces the repeated `.build(out).map_err(|e| TensorError::InvalidShape(...))` pattern
/// found across 19 GPU dispatch call sites.
pub(crate) fn build_kernel(
    builder: nn_dsl::TensorBlockBuilder,
    out: nn_dsl::TensorNodeId,
) -> nn_core::Result<nn_dsl::TensorKernelDef> {
    builder
        .build(out)
        .map_err(|e| nn_core::TensorError::InvalidShape(format!("kernel build: {e}")))
}

/// Check for non-finite values in a float slice and return a model-specific error if found.
///
/// Replaces inline `count_non_finite` + error-return blocks across model forward files.
/// The `make_err` closure constructs the model-specific error type given the non-finite
/// count. Each model has its own error variant (e.g., `NonFiniteInput { block, count }`,
/// `NonFiniteOutput { count }`, `NonFiniteIntermediate { stage, count }`).
///
/// Respects the thread-local [`NanCheckPolicy`](nn_core::layers::NanCheckPolicy): when
/// `Skip` is active (set via [`with_nan_check_policy`](nn_core::layers::with_nan_check_policy)),
/// returns `Ok(())` without scanning. This allows GPU forward paths to skip per-stage
/// finiteness checks while retaining model-boundary validation outside the skip scope.
pub(crate) fn check_non_finite_err<E>(
    data: &[f32],
    make_err: impl FnOnce(usize) -> E,
) -> Result<(), E> {
    if nn_core::layers::nan_check_policy() == nn_core::layers::NanCheckPolicy::Skip {
        return Ok(());
    }
    let count = count_non_finite(data);
    if count > 0 {
        return Err(make_err(count));
    }
    Ok(())
}

/// Global counter of GPU-to-CPU fallback events.
///
/// Incremented atomically on every `gpu_fallback()` call in both debug and
/// release builds. Consumers can check this before/after a forward pass to
/// detect unintended fallbacks:
///
/// ```no_run
/// use nn_metal::GPU_FALLBACK_COUNT;
/// use std::sync::atomic::Ordering;
///
/// let before = GPU_FALLBACK_COUNT.load(Ordering::Relaxed);
/// // ... run forward pass ...
/// let fallbacks = GPU_FALLBACK_COUNT.load(Ordering::Relaxed) - before;
/// assert_eq!(fallbacks, 0, "unexpected GPU→CPU fallbacks");
/// ```
pub static GPU_FALLBACK_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Log a GPU-to-CPU fallback and return `None`.
///
/// In debug builds, emits an `eprintln!` diagnostic so developers can identify
/// which GPU ops silently fall back to CPU. In all builds, increments
/// [`GPU_FALLBACK_COUNT`] so consumers can detect fallbacks programmatically.
///
/// Usage: `return gpu_fallback("softmax", "non-last axis not supported on Metal");`
#[cold]
pub(crate) fn gpu_fallback<T>(op: &str, reason: &str) -> Option<T> {
    GPU_FALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    #[cfg(debug_assertions)]
    eprintln!("[nn-metal] GPU fallback: {op} -> CPU ({reason})");
    let _ = (op, reason); // suppress unused-variable warning in release
    None
}

/// Convert a `usize` to `u32` for Metal GPU dispatch parameters, returning an
/// error instead of silently truncating if the value exceeds `u32::MAX`.
///
/// Metal compute shaders use `uint` (32-bit) for grid dimensions and kernel
/// constants. Bare `as u32` casts silently truncate on overflow, producing
/// incorrect dispatch parameters. This helper makes the conversion explicit.
/// Convert a `DType` to its MSL scalar type string and byte size.
///
/// Returns `("float", 4)` for F32, `("half", 2)` for F16/BF16.
/// Used by fused MSL kernel dispatchers that emit dtype-parameterized kernels.
pub(crate) fn dtype_to_msl(dtype: nn_core::DType) -> nn_core::Result<(&'static str, usize)> {
    let st = nn_dsl::ir::ScalarType::try_from(dtype)
        .map_err(|_| nn_core::TensorError::dtype_mismatch(nn_core::DType::F32, dtype))?;
    Ok((st.msl_str(), st.byte_size()))
}

pub(crate) fn to_u32(val: usize, _name: &str) -> nn_core::Result<u32> {
    u32::try_from(val).map_err(|_| nn_core::TensorError::ValueOutOfRange {
        description: "exceeds u32::MAX for Metal dispatch",
    })
}

pub mod to_standalone;
pub mod buffer_pool_size_class;
pub(crate) mod native_op_direct;
pub(crate) mod gpu_weight_cache;
pub(crate) use gpu_weight_cache::{GpuWeightCache, GpuWeightRef};

#[cfg(test)]
pub(crate) mod test_common;

#[cfg(test)]
pub(crate) mod demucs_test_common;

#[cfg(test)]
#[path = "error_conversion_tests.rs"]
mod error_conversion_tests;

#[cfg(test)]
#[path = "count_non_finite_tests.rs"]
mod count_non_finite_tests;

#[cfg(test)]
#[path = "to_u32_tests.rs"]
mod to_u32_tests;

#[cfg(test)]
#[path = "streaming_assembler_tests.rs"]
mod streaming_assembler_tests;

#[cfg(test)]
#[path = "dispatch_infrastructure_tests.rs"]
mod dispatch_infrastructure_tests;

#[cfg(test)]
#[path = "compiled_model_extended_tests.rs"]
mod compiled_model_extended_tests;

#[cfg(test)]
#[path = "dispatch_plan_extended_tests.rs"]
mod dispatch_plan_extended_tests;

#[cfg(test)]
#[path = "compiled_model_builder_extended_tests.rs"]
mod compiled_model_builder_extended_tests;

#[cfg(test)]
#[path = "fused_op_extended_tests.rs"]
mod fused_op_extended_tests;

#[cfg(test)]
#[path = "fused_op_gpu_correctness_tests.rs"]
mod fused_op_gpu_correctness_tests;

#[cfg(test)]
#[path = "metal_dispatch_extended_tests.rs"]
mod metal_dispatch_extended_tests;

#[cfg(test)]
#[path = "metal_native_op_extended_tests.rs"]
mod metal_native_op_extended_tests;

#[cfg(test)]
#[path = "metal_buffer_safety_tests.rs"]
mod metal_buffer_safety_tests;

#[cfg(test)]
#[path = "metal_dispatch_plan_extended_tests.rs"]
mod metal_dispatch_plan_extended_tests;

#[cfg(test)]
#[path = "metal_pool2d_extended_tests.rs"]
mod metal_pool2d_extended_tests;

#[cfg(test)]
#[path = "metal_batchnorm2d_extended_tests.rs"]
mod metal_batchnorm2d_extended_tests;

#[cfg(test)]
#[path = "native_op_fusion_extended_tests.rs"]
mod native_op_fusion_extended_tests;

/// Compile MSL source into a [`ComputePipeline`] using a fresh context.
///
/// Convenience wrapper for simple one-shot compilation. For repeated
/// compilations, prefer [`PipelineCache`] to avoid redundant work.
#[cfg(target_os = "macos")]
#[must_use = "returns a Result that may contain an error"]
pub fn compile_msl_pipeline(
    msl_source: &str,
    entry_point: &str,
    fast_math: bool,
) -> Result<(MetalContext, ComputePipeline), MetalError> {
    let context = MetalContext::new()?;
    let source = KernelSource::new(msl_source, entry_point).with_fast_math(fast_math);
    let pipeline = context.compile_pipeline(&source)?;
    Ok((context, pipeline))
}

/// Compile MSL source — returns `Err` on non-macOS platforms.
#[cfg(not(target_os = "macos"))]
#[must_use = "returns a Result that may contain an error"]
pub fn compile_msl_pipeline(
    _msl_source: &str,
    _entry_point: &str,
    _fast_math: bool,
) -> Result<(MetalContext, ComputePipeline), MetalError> {
    Err(MetalError::UnsupportedPlatform)
}
