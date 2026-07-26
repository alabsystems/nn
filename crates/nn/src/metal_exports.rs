// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal GPU backend re-exports for `nn::metal`.
//!
//! This module provides everything needed to run ML models on Apple Silicon
//! GPUs: weight loading, kernel dispatch, pre-built model pipelines (Kokoro TTS,
//! Silero VAD, HTDemucs), and GPU memory management. It is the primary
//! integration surface for downstream consumers like dvoice.
//!
//! # Getting started
//!
//! Add nn with the `metal` feature:
//!
//! ```toml
//! [dependencies]
//! nn = { git = "https://github.com/alabsystems/nn", features = ["metal", "models"] }
//! ```
//!
//! # Single-voice Kokoro TTS synthesis
//!
//! The simplest path: load weights, build the model, synthesize audio.
//!
//! ```rust,ignore
//! use nn::metal::{
//!     CompiledKokoro, PipelineCache, register_metal_dyn_backend, from_mmaped_safetensors,
//! };
//! use nn::models::{KokoroConfig, KokoroModel, KokoroTokenizer, VoicePack};
//! use nn::{DType, Device, VarBuilder};
//!
//! // 1. Initialize Metal backend (once per process).
//! register_metal_dyn_backend();
//! let cache = PipelineCache::new();
//!
//! // 2. Load weights via mmap (zero-copy into Metal buffers).
//! let vb = unsafe {
//!     from_mmaped_safetensors(
//!         &[std::path::Path::new("kokoro_v1_0.safetensors")],
//!         DType::F32,
//!         &Device::Gpu,
//!     )?
//! };
//!
//! // 3. Build model and compile to GPU.
//! let config = KokoroConfig::default();
//! let model = KokoroModel::new(&config, vb)?;
//! let mut kokoro = CompiledKokoro::new(model)?
//!     .with_autocast();  // F16 mixed precision -- 2x throughput on Apple Silicon
//!
//! // 4. Tokenize and synthesize.
//! let tokenizer = KokoroTokenizer::new()?;
//! let input_ids = tokenizer.encode("Hello from nn.")?;
//! let voice = VoicePack::load("af_heart", &vb)?;
//! let (audio, certificate) = kokoro.synthesize(&input_ids, &voice.style, 1.0, &cache)?;
//!
//! // 5. Check the quality certificate.
//! assert!(certificate.overall_passed);
//! ```
//!
//! # Multi-voice chorus synthesis
//!
//! [`KokoroChorus`] creates N voice instances that share GPU weight buffers
//! with a primary [`CompiledKokoro`]. Each voice synthesizes the same text
//! with a different style, and the outputs are mixed together.
//!
//! ```rust,ignore
//! use nn::metal::{CompiledKokoro, KokoroChorus, PipelineCache};
//! use nn::models::kokoro_chorus::ChorusConfig;
//!
//! // Warm up the primary instance first (compiles GPU segments).
//! let cache = PipelineCache::new();
//! kokoro.synthesize(&input_ids, &styles[0], 1.0, &cache)?;
//!
//! // Create an 8-voice chorus -- shares weights, independent segment caches.
//! let chorus_config = ChorusConfig::equal_gain(8)?;
//! let mut chorus = KokoroChorus::new(&kokoro, chorus_config)?;
//!
//! // Synthesize all voices and mix.
//! let mixed_audio = chorus.synthesize_chorus(&input_ids, &styles, 1.0, &cache)?;
//! ```
//!
//! # Streaming synthesis
//!
//! For low-latency playback, use pull-based streaming sessions. Audio is
//! produced chunk-by-chunk with crossfade between chunks.
//!
//! ## Single-voice streaming
//!
//! ```rust,ignore
//! use nn::metal::{CompiledKokoro, StreamingKokoroSession, PipelineCache};
//!
//! // Pre-chunk text into token tensors.
//! let chunks: Vec<(DynTensor, DynTensor)> = /* tokenized chunks */;
//! let mut session = StreamingKokoroSession::new(chunks, /*speed=*/1.0);
//!
//! // Pull chunks until exhausted.
//! while let Some(result) = session.next_chunk(&mut kokoro, &cache) {
//!     let (audio_chunk, cert) = result?;
//!     // Feed audio_chunk to audio output / playback buffer.
//! }
//! ```
//!
//! ## Multi-voice streaming chorus
//!
//! ```rust,ignore
//! use nn::metal::{StreamingChorusSession, KokoroChorus, PipelineCache};
//! use nn::models::kokoro_streaming::KokoroStreamConfig;
//!
//! let stream_config = KokoroStreamConfig::default();
//! let mut session = StreamingChorusSession::new(
//!     chunks,         // shared text chunks
//!     styles,         // per-voice style embeddings
//!     1.0,            // speed
//!     stream_config,
//! )?;
//!
//! // Pull mixed chorus chunks.
//! while let Some(result) = session.next_chunk(&mut chorus, &cache) {
//!     let audio_chunk = result?;
//!     // audio_chunk.samples contains crossfaded multi-voice audio.
//! }
//! ```
//!
//! # Non-blocking GPU synthesis with GpuFence
//!
//! [`GpuFence`] enables pipelining: encode the next segment on CPU while the
//! previous one executes on GPU. [`GpuAudioHandle`] defers the GPU-to-CPU
//! transfer so the caller controls when the readback happens.
//!
//! ```rust,ignore
//! use nn::metal::{CompiledKokoro, GpuFence, GpuAudioHandle, PipelineCache};
//!
//! // synthesize_gpu() returns a GPU-resident handle -- no flush.
//! let (handle, cert) = kokoro.synthesize_gpu(&input_ids, &style, 1.0, &cache)?;
//!
//! // Do other work while audio sits on GPU (encode next utterance, etc.).
//! prepare_next_utterance();
//!
//! // Transfer to CPU only when needed.
//! let pcm: Vec<f32> = handle.to_cpu()?;
//! ```
//!
//! For segment-level pipelining with explicit fences:
//!
//! ```rust,ignore
//! use nn::metal::GpuFence;
//!
//! // Submit segment 1 GPU work.
//! let fence1 = GpuFence::submit_current()?;
//!
//! // Encode segment 2 on CPU while segment 1 runs on GPU.
//! encode_segment_2();
//! let fence2 = GpuFence::submit_current()?;
//!
//! // Wait for results in order.
//! if let Some(f) = fence1 { f.wait()?; }
//! if let Some(f) = fence2 { f.wait()?; }
//! ```
//!
//! # Compiled model builder
//!
//! [`CompiledModelBuilder`] provides fluent configuration for compiling
//! arbitrary traced computation graphs to GPU dispatch plans. This is the
//! generic counterpart to [`CompiledKokoro`] for non-Kokoro models.
//!
//! ```rust,ignore
//! use nn::metal::{CompiledModel, CompiledModelBuilder, PipelineCache};
//! use nn::MixedPrecisionPolicy;
//!
//! let compiled = CompiledModel::builder(&graph, &cache)
//!     .autocast(MixedPrecisionPolicy::apple_silicon_default())
//!     .build()?;
//! let output = compiled.execute_dyn(&cache, &[&input])?;
//! ```
//!
//! # Segment cache configuration
//!
//! [`SegmentCacheConfig`] controls how many compiled GPU dispatch plans are
//! cached per pipeline step. Larger caches avoid recompilation when input
//! lengths vary; the byte budget caps total GPU memory for cached plans.
//!
//! ```rust,ignore
//! use nn::metal::{CompiledKokoro, SegmentCacheConfig, EvictionPolicy};
//!
//! let cache_config = SegmentCacheConfig {
//!     max_segments_per_step: 8,       // cache up to 8 shapes per step
//!     eviction: EvictionPolicy::Lru,  // evict least-recently-used
//!     byte_budget: Some(1024 * 1024 * 1024),  // 1 GB max for cached plans
//! };
//! let kokoro = CompiledKokoro::new(model)?
//!     .with_segment_cache_config(cache_config);
//! ```
//!
//! # F16 autocast (recommended for Apple Silicon)
//!
//! F16 mixed precision doubles Metal ALU throughput on Apple Silicon. Enable
//! it uniformly with [`with_autocast()`](CompiledKokoro::with_autocast), or
//! use [`F16AutocastConfig`] for per-segment control.
//!
//! ```rust,ignore
//! use nn::metal::{CompiledKokoro, F16AutocastConfig};
//! use nn::MixedPrecisionPolicy;
//!
//! // Uniform: all segments use F16.
//! let kokoro = CompiledKokoro::new(model)?.with_autocast();
//!
//! // Per-segment: disable autocast for the regulate stage (pure elementwise).
//! let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default())
//!     .with_regulate(false);
//! let kokoro = CompiledKokoro::new(model)?.with_segment_autocast(config);
//! ```
//!
//! # GPU memory management
//!
//! Use [`ActivationArena`] for bump-allocated intermediate GPU buffers (avoids
//! per-op allocation), and [`with_gpu_scope`] to batch command buffer encoding
//! into a single GPU submit.
//!
//! ```rust,ignore
//! use nn::metal::{with_arena, with_gpu_scope, flush, ActivationArena};
//!
//! let mut arena = ActivationArena::new();
//! let audio = with_arena(&mut arena, || {
//!     with_gpu_scope(|| {
//!         kokoro.synthesize(&input_ids, &style, 1.0, &cache)
//!     })
//! })?;
//!
//! // flush() commits any remaining GPU work before CPU readback.
//! flush()?;
//! ```
//!
//! # Memory monitoring
//!
//! Track process RSS and Metal GPU allocation for capacity planning:
//!
//! ```rust,ignore
//! use nn::metal::{rss_mb, metal_allocated_mb, RssTracker};
//!
//! println!("RSS: {:.1} MB, Metal: {:.1} MB", rss_mb(), metal_allocated_mb());
//!
//! let tracker = RssTracker::new();
//! // ... run workload ...
//! let snapshot = tracker.snapshot();
//! println!("Peak RSS: {:.1} MB", snapshot.peak_rss_mb);
//! ```
//!
//! # Optimization reports
//!
//! [`OptimizationReport`] captures per-step dispatch counts, precision
//! contracts, and fusion status after compilation. Use [`diff_reports`]
//! to compare reports across optimization iterations.
//!
//! ```rust,ignore
//! use nn::metal::{OptimizationReport, diff_reports, IterationVerdict};
//!
//! let report_a = compiled_a.optimization_report();
//! let report_b = compiled_b.optimization_report();
//! let delta = diff_reports(&report_a, &report_b);
//! match delta.verdict {
//!     IterationVerdict::Improved => println!("Optimization helped!"),
//!     IterationVerdict::Regressed => println!("Optimization regressed."),
//!     IterationVerdict::Neutral => println!("No change."),
//! }
//! ```

// ---------------------------------------------------------------------------
// Backend infrastructure
// ---------------------------------------------------------------------------

/// Global counter of GPU-to-CPU fallback events. Incremented on every
/// `gpu_fallback()` call; check before/after a forward pass to detect
/// unintended CPU fallbacks during GPU inference.
pub use nn_metal::GPU_FALLBACK_COUNT;

/// Core Metal backend types.
///
/// - [`MetalBackend`] -- typed backend for `Tensor<D, T, Metal>`.
/// - [`MetalContext`] -- Metal device handle and command queue.
/// - [`MetalElement`] -- trait for types storable in Metal buffers (f32, f16, bf16).
/// - [`MetalError`] -- error type for Metal operations.
/// - [`MetalInitOptions`] -- options for `MetalBackend::init_with`; the
///   default is proof-closed (compile-time embedded shaders only, no runtime
///   filesystem loading).
/// - [`PipelineCache`] -- LRU cache of compiled Metal compute pipelines.
///   Shared across dispatch calls to avoid redundant MSL compilation.
pub use nn_metal::{
    MetalBackend, MetalContext, MetalElement, MetalError, MetalInitOptions, PipelineCache,
};

/// Number of pipelines currently cached in the global shared pipeline cache.
pub use nn_metal::precompiled_pipeline_count;

// ---------------------------------------------------------------------------
// Low-level compute primitives
// ---------------------------------------------------------------------------

/// MSL compilation, Metal buffer management, and raw compute dispatch.
///
/// Most consumers use higher-level APIs ([`CompiledKokoro`], DynTensor ops)
/// instead of these primitives directly.
///
/// - [`compile_msl_pipeline`] -- one-shot MSL source to Metal pipeline.
/// - [`MetalBuffer`] -- GPU-resident buffer handle (reference-counted).
pub use nn_metal::{compile_msl_pipeline, MetalBuffer};

/// Command buffer batching primitives.
///
/// - [`BatchEncoder`] -- accumulates GPU commands into a [`CommandBatch`].
/// - [`CommandBatch`] -- a batch of Metal commands committed as a unit.
/// - [`ComputeDispatch`] -- a single compute kernel dispatch specification.
/// - [`PendingBatch`] -- a submitted but not yet committed command batch.
/// - [`ComputePipeline`] -- a compiled Metal compute pipeline state object.
/// - [`KernelSource`] -- MSL source text paired with an entry point name.
pub use nn_metal::{
    BatchEncoder, CommandBatch, ComputeDispatch, ComputePipeline, KernelSource, PendingBatch,
};

/// Kernel dispatch binding types for manual Metal kernel invocation.
///
/// - [`BufferAccess`] -- read or read-write access mode for a buffer binding.
/// - [`BufferBinding`] -- binds a Metal buffer to a kernel argument index.
/// - [`KernelPipeline`] -- a pipeline paired with threadgroup dimensions.
pub use nn_metal::{BufferAccess, BufferBinding, KernelPipeline};

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

/// Safetensors weight loading with mmap-backed Metal buffers.
///
/// [`from_mmaped_safetensors`] is the primary entry point: it mmaps
/// safetensors files and creates GPU-resident weight buffers with zero copy.
/// [`WeightMap`] provides direct tensor lookup by name.
/// [`var_builder_from_weight_map`] wraps a `WeightMap` as a hierarchical
/// `VarBuilder` for model constructors that expect candle-compatible loading.
/// [`SafeTensorsBackend`] is the backend adapter for `VarBuilder`.
pub use nn_metal::{
    from_mmaped_safetensors, from_mmaped_safetensors_with_ctx, var_builder_from_weight_map,
    SafeTensorsBackend,
};

/// Weight map types for direct tensor access.
///
/// - [`WeightMap`] -- mmap-backed tensor storage; lookup tensors by name.
/// - [`WeightError`] -- errors from weight loading (missing key, shape mismatch).
/// - [`TensorInfo`] -- metadata for a single tensor (dtype, shape, byte offsets).
/// - [`MetalVarBuilderExt`] -- extension trait for constructing Metal-backed VarBuilders.
pub use nn_metal::{MetalVarBuilderExt, TensorInfo, WeightError, WeightMap};

// ---------------------------------------------------------------------------
// Tensor dispatch
// ---------------------------------------------------------------------------

/// Execute pre-planned GPU dispatch operations on DynTensors.
///
/// These functions run compiled dispatch plans against input tensors.
/// [`execute_tensor_dispatch`] is the standard path; batched and readback
/// variants exist for bulk operations and CPU-transfer patterns.
///
/// - [`execute_tensor_dispatch`] -- dispatch a plan with DynTensor inputs.
/// - [`execute_tensor_dispatch_batched`] -- dispatch multiple plans in a single batch.
/// - [`execute_tensor_dispatch_readback`] -- dispatch and read result back to CPU.
/// - [`execute_tensor_dispatch_to_buffer`] -- dispatch into a pre-allocated Metal buffer.
/// - [`execute_tensor_dispatch_to_buffer_with_contract`] -- dispatch with precision contract.
/// - [`DispatchInput`] -- tagged input for dispatch (tensor or constant).
/// - [`TensorDispatchError`] -- errors from tensor dispatch execution.
pub use nn_metal::{
    execute_tensor_dispatch, execute_tensor_dispatch_batched, execute_tensor_dispatch_readback,
    execute_tensor_dispatch_to_buffer, execute_tensor_dispatch_to_buffer_with_contract,
    DispatchInput, TensorDispatchError,
};

/// Dispatch planning types.
///
/// - [`DispatchPlan`] -- a sequence of GPU operations compiled from a traced graph.
/// - [`DispatchMode`] -- controls how dispatch plans are executed (immediate vs batched).
pub use nn_metal::{DispatchMode, DispatchPlan};

/// Precision types for tensor dispatch with explicit precision contracts.
///
/// - [`PrecisionContract`] -- declares the numeric precision guarantee of a dispatch.
/// - [`PrecisionTier`] -- coarse precision classification (Exact, Approximate, etc.).
pub use nn_metal::{PrecisionContract, PrecisionTier};

// ---------------------------------------------------------------------------
// Pre-built models
// ---------------------------------------------------------------------------

/// Silero VAD (Voice Activity Detection) -- streaming speech/silence
/// segmentation on Metal GPU.
///
/// - [`SileroVad`] -- the pre-compiled VAD model with persistent GPU weights.
/// - [`SileroVadState`] -- streaming state (h/c LSTM hidden, sample count).
/// - [`SileroVadOutput`] -- per-frame speech probability + timestamp.
/// - [`SileroVadWeights`] -- weight container for the Silero ONNX model.
/// - [`SileroVadError`] -- errors from VAD inference.
/// - [`SegmentConfig`] -- configuration for segment detection thresholds.
/// - [`SpeechSegment`] -- a detected speech region (start/end samples).
pub use nn_metal::{
    SegmentConfig, SileroVad, SileroVadError, SileroVadOutput, SileroVadState, SileroVadWeights,
    SpeechSegment,
};

/// HTDemucs music source separation model -- separate vocals, drums, bass,
/// and other instruments from mixed audio.
///
/// - [`HTDemucs`] -- the pre-compiled model with GPU weights and forward pass.
/// - [`HTDemucsError`] -- errors from source separation inference.
/// - [`HTDemucsWeights`] -- weight container composed of sub-component weights.
/// - [`WeightLoadError`] -- errors from weight file loading.
pub use nn_metal::{HTDemucs, HTDemucsError, HTDemucsWeights, WeightLoadError};

/// HTDemucs sub-component weight types (needed to construct [`HTDemucsWeights`]).
/// Consumers construct these from weight files, then assemble into [`HTDemucsWeights`].
pub use nn_metal::{
    DemucsSpectralDecoderWeights, DemucsSpectralEncoderWeights, DemucsTemporalDecoderWeights,
    DemucsTemporalEncoderWeights, DemucsTransformerWeights,
};

// ---------------------------------------------------------------------------
// Signal processing
// ---------------------------------------------------------------------------

/// STFT / iSTFT utilities for time-frequency analysis and audio
/// reconstruction on GPU.
///
/// - [`compute_stft_magnitude`] -- forward STFT to magnitude spectrum.
/// - [`StftParams`] -- STFT window size, hop size, FFT size.
/// - [`StftError`] -- errors from STFT computation.
/// - [`IstftBasis`] -- CPU-side inverse STFT basis matrices.
/// - [`IstftGpuBasis`] -- GPU-uploaded iSTFT basis for Metal dispatch.
/// - [`IstftParams`] -- iSTFT window/hop configuration.
/// - [`IstftError`] -- errors from inverse STFT.
pub use nn_metal::{compute_stft_magnitude, StftError, StftParams};
pub use nn_metal::{IstftBasis, IstftError, IstftGpuBasis, IstftParams};

// ---------------------------------------------------------------------------
// Tensor bridge and backend registration
// ---------------------------------------------------------------------------

/// GPU buffer slice handle (zero-copy view into Metal buffer regions).
///
/// Represents a `(MetalBuffer, byte_offset)` pair for referencing interior
/// regions of arena-allocated GPU buffers.
pub use nn_metal::GpuSlice;

/// Typed tensor bridge -- convert between `Tensor<D, T, Metal>` and Metal buffers.
///
/// - [`from_metal_buffer`] -- create a typed `Tensor` from a raw Metal buffer.
/// - [`MetalTensorExt`] -- extension trait for DynTensor Metal operations
///   (GPU buffer access, Metal-specific dispatch).
/// - [`MetalTensorStorage`] -- underlying GPU buffer storage for Metal tensors.
pub use nn_metal::{from_metal_buffer, MetalTensorExt, MetalTensorStorage};

/// Register the Metal backend for DynTensor GPU dispatch.
///
/// Call [`register_metal_dyn_backend()`] once at startup before using
/// `Device::Gpu` with DynTensor operations. This installs the Metal
/// implementation of `GpuBackend` into the global registry.
///
/// [`MetalTensorData`] is the per-tensor GPU storage handle (buffer + offset + dtype).
pub use nn_metal::{register_metal_dyn_backend, MetalTensorData};

// ---------------------------------------------------------------------------
// GPU command buffer batching and scope
// ---------------------------------------------------------------------------

/// GPU command buffer batching -- wrap model forward passes in
/// [`with_gpu_scope`] to reduce per-op `commit_and_wait` barriers to a
/// single barrier at scope exit.
///
/// - [`with_gpu_scope`] -- batch GPU commands within a closure.
/// - [`flush()`] -- commit pending GPU work before CPU readback.
/// - [`encode_custom_dispatch`] -- inject a custom dispatch into the current batch.
/// - [`submit`] -- submit the current command buffer without waiting.
/// - [`sync`] -- submit and wait for the current command buffer.
/// - [`with_scope_exit_mode`] -- override whether scope exit flushes or submits.
/// - [`ScopeExitMode`] -- controls scope exit behavior (Flush, Submit, None).
pub use nn_metal::{
    encode_custom_dispatch, flush, submit, sync, with_gpu_scope, with_scope_exit_mode,
    ScopeExitMode,
};

/// GPU dispatch profiling counters -- track Metal dispatch count, encoding
/// events, and fallback stats for performance tuning.
///
/// - [`dispatch_stats()`] -- snapshot current dispatch counters.
/// - [`reset_counters()`] -- zero all dispatch counters.
/// - [`DispatchStats`] -- snapshot struct with dispatch count, encoding events, fallbacks.
pub use nn_metal::{dispatch_stats, reset_counters, DispatchStats};

/// Dispatch profiler -- detailed per-op profiling for optimization.
///
/// - [`DispatchProfiler`] -- collects per-op timing and type information.
/// - [`DispatchProfileReport`] -- aggregated profiling report.
/// - [`DispatchProfileEntry`] -- a single profiled dispatch operation.
/// - [`DispatchType`] -- classification of dispatch (native, fused, fallback).
/// - [`FusionOpportunity`] -- detected fusion opportunity from profiling.
/// - [`TopEntry`] -- top-N dispatch entries by cost.
/// - [`TypeBreakdown`] -- dispatch count per type category.
pub use nn_metal::{
    DispatchProfileEntry, DispatchProfileReport, DispatchProfiler, DispatchType, FusionOpportunity,
    TopEntry, TypeBreakdown,
};

// ---------------------------------------------------------------------------
// Activation arena and memory management
// ---------------------------------------------------------------------------

/// Activation arena -- bump-allocator for intermediate GPU buffers.
///
/// [`with_arena`] scopes arena allocation around a closure. Intermediate
/// buffers are reused across forward passes without per-op allocation.
/// [`with_decode_scope`] is a specialized scope for autoregressive decoding.
/// [`without_arena`] disables arena allocation within a nested scope.
/// [`try_reset_active_arena`] resets the active arena without reallocating.
pub use nn_metal::{
    try_reset_active_arena, with_arena, with_decode_scope, without_arena, ActivationArena,
};

/// Arena statistics -- introspection for arena buffer reuse metrics.
///
/// - [`arena_stats()`] -- snapshot current arena usage counters.
/// - [`reset_arena_stats()`] -- zero all arena counters.
/// - [`ArenaStats`] -- aggregate arena metrics (allocations, reuse, bytes).
/// - [`PoolStats`] -- per-size-class pool metrics within the arena.
pub use nn_metal::{arena_stats, reset_arena_stats, ArenaStats, PoolStats};

/// Arena estimation -- predict peak memory usage from step byte sizes.
///
/// - [`estimate_arena_peak_bytes`] -- estimate peak from a list of step sizes.
/// - [`ArenaEstimate`] -- result with peak bytes and per-step breakdown.
pub use nn_metal::{estimate_arena_peak_bytes, ArenaEstimate};

/// Arena capacity management.
///
/// - [`arena_capacity`] -- query the current arena capacity.
/// - [`ensure_default_arena_capacity`] -- pre-allocate arena to a minimum size.
/// - [`default_arena_total_growth_count`] -- query how many times the arena grew.
/// - [`estimate_arena_peak_from_shapes`] -- estimate peak from tensor shapes.
pub use nn_metal::{
    arena_capacity, default_arena_total_growth_count, ensure_default_arena_capacity,
    estimate_arena_peak_from_shapes,
};

// ---------------------------------------------------------------------------
// Live weight editing and LoRA
// ---------------------------------------------------------------------------

/// GPU-resident LoRA overlay -- apply/remove low-rank adapters on Metal
/// without mutating base weights.
pub use nn_metal::LoraGpuOverlay;

/// Live weight editing -- verified weight surgery with bounds checking.
///
/// - [`apply_weight_edit`] -- apply a weight edit specification to GPU buffers.
/// - [`WeightEditSpec`] -- describes which weight to edit and the new values.
/// - [`WeightEditResult`] -- outcome of the edit (success, partial, error).
/// - [`WeightEditError`] -- errors from weight editing.
pub use nn_metal::{apply_weight_edit, WeightEditError, WeightEditResult, WeightEditSpec};

/// Live edit apply -- GPU-resident weight surgery with receipts.
///
/// - [`LiveEditApply`] -- orchestrates delta application to GPU weights.
/// - [`ApplyReceipt`] -- receipt for a completed weight edit.
/// - [`DeltaApplyReceipt`] -- receipt for a delta (incremental) weight edit.
/// - [`LiveEditError`] -- errors from live edit operations.
pub use nn_metal::{ApplyReceipt, DeltaApplyReceipt, LiveEditApply, LiveEditError};

// ---------------------------------------------------------------------------
// Compiled model (generic trace-based execution)
// ---------------------------------------------------------------------------

/// Compiled model -- pre-compiled trace-based graph execution for arbitrary
/// models. See [`CompiledKokoro`] for the Kokoro-specific pipeline.
///
/// - [`CompiledModel`] -- holds a compiled dispatch plan and GPU weight buffers.
///   Executes a traced computation graph with zero IR rebuild overhead.
/// - [`CompiledModelError`] -- errors from model compilation and execution.
pub use nn_metal::compiled_model::{CompiledModel, CompiledModelError};

/// Compiled model builder -- fluent API for configuring model compilation.
///
/// [`CompiledModelBuilder`] is created via [`CompiledModel::builder()`] and
/// supports shared weights, forced dtype, autocast policy, peephole
/// optimization, and optimization budget before calling `.build()`.
///
/// ```rust,ignore
/// use nn::metal::{CompiledModel, CompiledModelBuilder, PipelineCache};
/// use nn::MixedPrecisionPolicy;
///
/// let compiled = CompiledModel::builder(&graph, &cache)
///     .autocast(MixedPrecisionPolicy::apple_silicon_default())
///     .build()?;
/// ```
pub use nn_metal::compiled_model::CompiledModelBuilder;

/// Metallib loading -- load precompiled Metal libraries.
///
/// The default shader-delivery path is proof-closed: the metallib bytes are
/// embedded into the binary at compile time ([`embedded_metallib`]). Loading
/// a `.metallib` from the filesystem at runtime requires the explicit double
/// opt-in (`MetalInitOptions::allow_runtime_metallib(true)` plus
/// `NN_ALLOW_RUNTIME_METALLIB=1`) and is loud, never silent.
///
/// - [`embedded_metallib`] -- the compile-time embedded metallib bytes.
/// - [`load_metallib`] -- create a Metal library from caller-provided metallib bytes.
/// - [`pipelines_from_metallib`] -- extract compute pipelines from a loaded metallib.
/// - [`precompiled_metallib_path`] -- the build-time metallib path (informational).
pub use nn_metal::metallib_loader::{
    embedded_metallib, load_metallib, pipelines_from_metallib, precompiled_metallib_path,
};

// ---------------------------------------------------------------------------
// Optimization reports
// ---------------------------------------------------------------------------

/// Optimization report -- per-step dispatch analysis after compilation.
///
/// - [`OptimizationReport`] -- captures dispatch count, precision contracts,
///   and fusion status for each compiled step.
/// - [`ContractStatus`] -- per-step precision contract fulfillment status.
/// - [`ReportError`] -- errors from report generation.
pub use nn_metal::{ContractStatus, OptimizationReport, ReportError};

/// Optimization report diff -- compare reports across optimization iterations.
///
/// - [`diff_reports`] -- compute the delta between two optimization reports.
/// - [`ReportDelta`] -- the difference between two reports.
/// - [`IterationVerdict`] -- whether an optimization iteration improved,
///   regressed, or had no effect.
pub use nn_metal::{diff_reports, IterationVerdict, ReportDelta};

// ---------------------------------------------------------------------------
// Kokoro TTS pipeline
// ---------------------------------------------------------------------------

/// Kokoro compiled TTS pipeline -- the primary synthesis API.
///
/// [`CompiledKokoro`] wraps a `KokoroModel` with 8 compiled GPU segments,
/// segment caching, F16 autocast, and quality verification. The main entry
/// points are:
///
/// - [`synthesize()`](CompiledKokoro::synthesize) -- full synthesis with CPU readback
/// - [`synthesize_gpu()`](CompiledKokoro::synthesize_gpu) -- GPU-resident result (no flush)
/// - [`with_autocast()`](CompiledKokoro::with_autocast) -- enable F16 mixed precision
/// - [`with_segment_autocast()`](CompiledKokoro::with_segment_autocast) -- per-segment F16 control
/// - [`clone_dispatch()`](CompiledKokoro::clone_dispatch) -- share weights for multi-voice
/// - [`clone_dispatch_warm()`](CompiledKokoro::clone_dispatch_warm) -- share weights + compiled pipelines
/// - [`warmup()`](CompiledKokoro::warmup) -- pre-compile segments for known input shapes
/// - [`release_model_weights()`](CompiledKokoro::release_model_weights) -- free CPU weights (~320 MB)
/// - [`segment_cache_stats()`](CompiledKokoro::segment_cache_stats) -- cache hit/miss metrics
pub use nn_metal::precompile_kokoro_msl;
pub use nn_metal::{
    CompiledKokoro, CompiledKokoroError, DiagnosticOutput, DispatchSummary, F16AutocastConfig,
    MemoryBreakdown, PrecompileResult, PrecompileShapes, SegmentPeepholeConfigs, StepEncodeResult,
    StepF0EnergyResult, StepGeneratorResult, StepProsodyResult, StepRegulateResult, StyleSplit,
    SynthesisIntermediates, TimingReport,
};

/// Serializable optimizer configuration types (requires `plan-serde` feature).
///
/// - [`OptimizerWarmupResult`] -- result of precompile optimizer warmup.
/// - [`load_peephole_configs`] / [`save_peephole_configs`] -- persist peephole configs.
/// - [`load_optimal_configs`] / [`save_optimal_configs`] -- persist optimal configs.
/// - [`KokoroOptimalConfigs`] -- per-segment optimal configuration set.
/// - [`SegmentOptimalConfig`] -- optimal config for a single segment.
#[cfg(feature = "plan-serde")]
pub use nn_metal::OptimizerWarmupResult;
#[cfg(feature = "plan-serde")]
pub use nn_metal::{
    load_optimal_configs, load_optimal_configs_if_exists, save_optimal_configs,
    KokoroOptimalConfigs, SegmentOptimalConfig,
};
#[cfg(feature = "plan-serde")]
pub use nn_metal::{load_peephole_configs, save_peephole_configs};

/// TTS quality certificate -- hard-bound verification results from synthesis.
///
/// Each [`CompiledKokoro::synthesize()`] call returns a [`Certificate`]
/// alongside the audio. The certificate contains hard-bound check results
/// (amplitude, silence, clipping) and quality metrics. Check
/// `certificate.overall_passed` to verify audio quality.
pub use nn_metal::Certificate;

/// Kokoro chorus -- multi-voice TTS synthesis pool.
///
/// [`KokoroChorus`] creates N voice instances sharing GPU weights with a
/// primary [`CompiledKokoro`]. Call
/// [`synthesize_chorus()`](KokoroChorus::synthesize_chorus) to synthesize
/// all voices and mix them together. [`ChorusConfig`] controls voice count,
/// per-voice gains, and clipping behavior.
///
/// - [`KokoroChorus`] -- the chorus manager (owns N `CompiledKokoro` clones).
/// - [`ChorusConfig`] -- voice count, per-voice gain weights, clipping.
/// - [`ChorusGpuSynth`] -- GPU synthesis backend for chorus pipeline.
/// - [`GpuSynth`] -- trait for GPU synthesis backends.
pub use nn_metal::{ChorusConfig, ChorusGpuSynth, GpuSynth, KokoroChorus};

/// Kokoro streaming sessions -- pull-based and push-based streaming.
///
/// - [`StreamingKokoroSession`] -- single-voice pull-based streaming
/// - [`StreamingChorusSession`] -- multi-voice pull-based streaming (shared or per-voice text)
/// - [`CompiledKokoroStreamingSession`] -- push-based streaming with crossfade assembly
/// - [`ChannelStreamingSession`] -- channel-based streaming with background synthesis
/// - [`ChorusChunkMode`] -- shared text vs. per-voice independent text
/// - [`StreamChunk`] -- a single audio chunk from a streaming session
/// - [`StreamReceiver`] -- receiving end of a channel streaming session
///
/// All streaming sessions produce audio chunk-by-chunk via `next_chunk()`,
/// suitable for real-time playback with low first-chunk latency.
pub use nn_metal::{
    ChannelStreamingSession, ChorusChunkMode, CompiledKokoroStreamingSession, StreamChunk,
    StreamReceiver, StreamingChorusSession, StreamingKokoroSession,
};

/// Phantom-typed pipeline tensor shapes for compile-time shape verification.
///
/// These types tag synthesis intermediate results with phantom type markers
/// so shape mismatches between pipeline stages are caught at compile time.
///
/// - [`PipelineTensor`] -- a DynTensor tagged with a phantom pipeline stage marker.
/// - [`TypedEncodeResult`] -- typed output of the encode stage.
/// - [`TypedProsodyResult`] -- typed output of the prosody stage.
/// - [`TypedRegulateResult`] -- typed output of the regulate stage.
/// - [`TypedF0EnergyResult`] -- typed output of the F0/energy prediction stage.
/// - [`TypedGeneratorResult`] -- typed output of the generator stage.
pub use nn_metal::{
    PipelineTensor, TypedEncodeResult, TypedF0EnergyResult, TypedGeneratorResult,
    TypedProsodyResult, TypedRegulateResult,
};

// ---------------------------------------------------------------------------
// Non-blocking GPU execution
// ---------------------------------------------------------------------------

/// Non-blocking GPU submit -- fence-based async GPU execution.
///
/// [`GpuFence`] lets callers hold multiple outstanding GPU submissions and
/// wait on them individually, enabling CPU/GPU pipelining. [`GpuFuture`]
/// wraps a fence with a typed result. [`AsyncGpuResult`] is the type alias
/// for the result of an async GPU operation.
pub use nn_metal::{AsyncGpuResult, GpuFence, GpuFuture};

/// GPU-resident audio handle -- deferred CPU transfer.
///
/// Returned by [`CompiledKokoro::synthesize_gpu()`]. Call
/// [`to_cpu()`](GpuAudioHandle::to_cpu) to transfer audio from GPU to CPU
/// when ready. This avoids blocking the GPU pipeline with an immediate readback.
pub use nn_metal::GpuAudioHandle;

// ---------------------------------------------------------------------------
// Segment cache configuration
// ---------------------------------------------------------------------------

/// Segment cache configuration -- LRU capacity, byte budget, eviction policy.
///
/// Controls how many compiled GPU dispatch plans are cached per Kokoro
/// pipeline step. Tune `max_segments_per_step` for workloads with varying
/// text lengths; set `byte_budget` to cap GPU memory for cached plans.
///
/// - [`SegmentCacheConfig`] -- capacity, eviction policy, byte budget.
/// - [`EvictionPolicy`] -- LRU or FIFO eviction strategy.
/// - [`ShapeKeyedCache`] -- generic shape-keyed LRU cache.
/// - [`SegmentCacheStats`] -- hit/miss/eviction/byte counters.
pub use nn_metal::{EvictionPolicy, SegmentCacheConfig, SegmentCacheStats, ShapeKeyedCache};

/// Shared segment store -- cross-session segment sharing for multi-voice.
///
/// - [`SharedSegmentStore`] -- thread-safe shared compiled segment cache.
/// - [`SegmentKey`] -- key type for segment lookup (step index + shape hash).
/// - [`SharedSegmentStats`] -- hit/miss counters for shared segments.
pub use nn_metal::{SegmentKey, SharedSegmentStats, SharedSegmentStore};

// ---------------------------------------------------------------------------
// GPU buffer utilities
// ---------------------------------------------------------------------------

/// GPU blit copy -- create a standalone buffer from an arena-resident tensor.
///
/// Arena buffers are invalidated when the arena scope exits. Call
/// [`to_standalone`] to copy an arena tensor into a persistent Metal buffer
/// that survives beyond the arena scope.
pub use nn_metal::to_standalone;

// ---------------------------------------------------------------------------
// Cache statistics
// ---------------------------------------------------------------------------

/// Pipeline cache and kernel def cache statistics.
///
/// - [`CacheStats`] -- hit/miss/eviction counters for the pipeline cache.
/// - [`CacheStatsSnapshot`] -- immutable snapshot of cache statistics.
pub use nn_metal::{CacheStats, CacheStatsSnapshot};

// ---------------------------------------------------------------------------
// Memory monitoring
// ---------------------------------------------------------------------------

/// Process RSS and Metal GPU allocation monitoring.
///
/// - [`rss_bytes()`] / [`rss_mb()`] -- current process resident set size.
/// - [`metal_allocated_bytes()`] / [`metal_allocated_mb()`] -- total Metal buffer allocation.
/// - [`metal_budget_bytes()`] -- Metal recommended working set size.
/// - [`RssSnapshot`] -- point-in-time snapshot of RSS and Metal allocation.
/// - [`RssTracker`] -- tracks peak RSS over the lifetime of the tracker.
pub use nn_metal::{
    metal_allocated_bytes, metal_allocated_mb, metal_budget_bytes, rss_bytes, rss_mb, RssSnapshot,
    RssTracker,
};

/// GPU memory profiler -- detailed per-category GPU memory breakdown.
///
/// - [`GpuMemoryProfiler`] -- tracks Metal buffer allocations by category.
/// - [`GpuMemorySnapshot`] -- point-in-time GPU memory snapshot.
/// - [`BufferCategory`] -- classification of a GPU buffer (weights, activations, etc.).
/// - [`MemoryBreakdownByCategory`] -- bytes per category.
pub use nn_metal::{
    BufferCategory, GpuMemoryProfiler, GpuMemorySnapshot, MemoryBreakdownByCategory,
};

// ---------------------------------------------------------------------------
// Document processing (dpdf)
// ---------------------------------------------------------------------------

/// Metal-accelerated document image preprocessing for the dpdf pipeline.
///
/// - [`DpdfImagePreprocessMetal`] -- GPU image preprocessing (resize, normalize).
/// - [`DpdfPipelineMetal`] -- end-to-end document processing pipeline on Metal.
pub use nn_metal::DpdfImagePreprocessMetal;
pub use nn_metal::DpdfPipelineMetal;

// ---------------------------------------------------------------------------
// Kokoro audio GPU helper
// ---------------------------------------------------------------------------

/// Kokoro forward audio GPU helper -- runs the audio reconstruction
/// (magnitude + phase to PCM via iSTFT) on GPU.
pub use nn_metal::kokoro_forward_audio_gpu;

// ---------------------------------------------------------------------------
// Buffer pool size class allocator
// ---------------------------------------------------------------------------

/// Size-class buffer pool -- reduces Metal buffer allocation fragmentation.
///
/// - [`SizeClassAllocator`] -- pool of Metal buffers bucketed by size class.
/// - [`AllocResult`] -- result of a pool allocation (new or reused buffer).
/// - [`SizeClassStats`] -- per-size-class hit/miss/total counters.
/// - [`BufferPoolSizeClassStats`] -- aggregate pool statistics.
pub use nn_metal::{AllocResult, BufferPoolSizeClassStats, SizeClassAllocator, SizeClassStats};

// ---------------------------------------------------------------------------
// Compiled model memory and optimizer reports
// ---------------------------------------------------------------------------

/// Compiled model memory report -- per-step GPU memory breakdown.
///
/// - [`MemoryReport`] -- full model memory breakdown.
/// - [`StepMemoryReport`] -- per-step memory usage.
/// - [`bytes_to_human`] -- format byte count as human-readable string.
/// - [`format_memory_report`] -- format a MemoryReport for display.
pub use nn_metal::{bytes_to_human, format_memory_report, MemoryReport, StepMemoryReport};

/// Compiled model optimizer report -- peephole optimization analysis.
///
/// - [`OptimizerReport`] -- optimization summary for a compiled model.
/// - [`diff_peephole_configs`] -- compare two peephole configurations.
/// - [`format_optimizer_report`] -- format an OptimizerReport for display.
/// - [`generate_optimizer_report_with_metrics`] -- generate report with metric data.
pub use nn_metal::{
    diff_peephole_configs, format_optimizer_report, generate_optimizer_report_with_metrics,
    OptimizerReport,
};
