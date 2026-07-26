// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive smoke test for ALL `nn::metal` re-exports.
//!
//! Every `pub use` in `crates/nn/src/metal_exports.rs` must have a
//! corresponding compile-time check here. If a re-export is removed or
//! renamed, this test fails to compile, preventing silent API breakage.
//!
//! Run: `cargo test -p nn --features metal --test metal_api_smoke`

#![allow(dead_code, unused_imports, clippy::let_unit_value)]

// Only compile when the metal feature is active.
#[cfg(feature = "metal")]
mod smoke {
    /// Helper: proves a concrete type is importable. Optimized away.
    fn check_type<T>() {}

    // =================================================================
    // 1. Backend infrastructure
    // =================================================================

    #[test]
    fn backend_infrastructure_types() {
        use nn::metal::{MetalBackend, MetalContext, MetalElement, MetalError, PipelineCache};

        check_type::<MetalBackend>();
        check_type::<MetalContext>();
        check_type::<MetalError>();
        check_type::<PipelineCache>();

        // MetalElement is a trait — verify f32 implements it.
        fn check_trait<T: MetalElement>() {}
        check_trait::<f32>();
    }

    #[test]
    fn gpu_fallback_count_constant() {
        use nn::metal::GPU_FALLBACK_COUNT;
        // GPU_FALLBACK_COUNT is AtomicU64.
        let _: &std::sync::atomic::AtomicU64 = &GPU_FALLBACK_COUNT;
    }

    // =================================================================
    // 2. Low-level compute primitives
    // =================================================================

    #[test]
    fn compute_primitives() {
        use nn::metal::{
            compile_msl_pipeline, BatchEncoder, CommandBatch, ComputeDispatch, ComputePipeline,
            KernelSource, MetalBuffer,
        };

        check_type::<MetalBuffer>();
        check_type::<ComputePipeline>();
        check_type::<ComputeDispatch>();
        check_type::<CommandBatch>();
        check_type::<BatchEncoder>();
        check_type::<KernelSource>();

        // compile_msl_pipeline is a function — import above is the assertion.
    }

    // =================================================================
    // 3. Kernel dispatch primitives
    // =================================================================

    #[test]
    fn kernel_dispatch_primitives() {
        use nn::metal::{BufferAccess, BufferBinding, KernelPipeline};

        check_type::<KernelPipeline>();
        check_type::<BufferAccess>();
        check_type::<BufferBinding<'_>>();
    }

    // =================================================================
    // 4. Weight loading
    // =================================================================

    #[test]
    fn weight_loading_functions() {
        // Generic functions — import alone is the compile-time assertion.
        use nn::metal::{
            from_mmaped_safetensors, from_mmaped_safetensors_with_ctx, var_builder_from_weight_map,
        };
    }

    #[test]
    fn weight_loading_types() {
        use nn::metal::{
            MetalVarBuilderExt, SafeTensorsBackend, TensorInfo, WeightError, WeightMap,
        };

        check_type::<SafeTensorsBackend>();
        check_type::<TensorInfo>();
        check_type::<WeightError>();
        check_type::<WeightMap>();
        // MetalVarBuilderExt is a trait — import is the assertion.
    }

    // =================================================================
    // 5. Tensor dispatch
    // =================================================================

    #[test]
    fn tensor_dispatch_functions() {
        // Generic functions — import alone is the compile-time assertion.
        use nn::metal::{
            execute_tensor_dispatch, execute_tensor_dispatch_batched,
            execute_tensor_dispatch_readback, execute_tensor_dispatch_to_buffer,
            execute_tensor_dispatch_to_buffer_with_contract,
        };
    }

    #[test]
    fn tensor_dispatch_types() {
        use nn::metal::{DispatchInput, TensorDispatchError};

        check_type::<TensorDispatchError>();
        // DispatchInput<'a, E: MetalElement> — verify with concrete type.
        check_type::<DispatchInput<'_, f32>>();
    }

    // =================================================================
    // 6. Dispatch planning
    // =================================================================

    #[test]
    fn dispatch_planning_types() {
        use nn::metal::{DispatchMode, DispatchPlan};

        check_type::<DispatchPlan>();
        let _: Option<DispatchMode> = None;
    }

    // =================================================================
    // 7. Precision types
    // =================================================================

    #[test]
    fn precision_types() {
        use nn::metal::{PrecisionContract, PrecisionTier};

        check_type::<PrecisionContract>();
        check_type::<PrecisionTier>();
    }

    // =================================================================
    // 8. Silero VAD model
    // =================================================================

    #[test]
    fn silero_vad_types() {
        use nn::metal::{
            SegmentConfig, SileroVad, SileroVadError, SileroVadOutput, SileroVadState,
            SileroVadWeights, SpeechSegment,
        };

        check_type::<SegmentConfig>();
        check_type::<SileroVad>();
        check_type::<SileroVadError>();
        check_type::<SileroVadOutput>();
        check_type::<SileroVadState>();
        check_type::<SileroVadWeights>();
        check_type::<SpeechSegment>();
    }

    // =================================================================
    // 9. HTDemucs model
    // =================================================================

    #[test]
    fn htdemucs_types() {
        use nn::metal::{HTDemucs, HTDemucsError, HTDemucsWeights, WeightLoadError};

        check_type::<HTDemucs>();
        check_type::<HTDemucsError>();
        check_type::<HTDemucsWeights>();
        check_type::<WeightLoadError>();
    }

    #[test]
    fn htdemucs_sub_component_weights() {
        use nn::metal::{
            DemucsSpectralDecoderWeights, DemucsSpectralEncoderWeights,
            DemucsTemporalDecoderWeights, DemucsTemporalEncoderWeights, DemucsTransformerWeights,
        };

        check_type::<DemucsSpectralDecoderWeights>();
        check_type::<DemucsSpectralEncoderWeights>();
        check_type::<DemucsTemporalDecoderWeights>();
        check_type::<DemucsTemporalEncoderWeights>();
        check_type::<DemucsTransformerWeights>();
    }

    // =================================================================
    // 10. STFT / iSTFT utilities
    // =================================================================

    #[test]
    fn stft_types_and_functions() {
        use nn::metal::{compute_stft_magnitude, StftError, StftParams};

        check_type::<StftError>();
        check_type::<StftParams>();
        let _ = compute_stft_magnitude;
    }

    #[test]
    fn istft_types() {
        use nn::metal::{IstftBasis, IstftError, IstftGpuBasis, IstftParams};

        check_type::<IstftBasis>();
        check_type::<IstftError>();
        check_type::<IstftGpuBasis>();
        check_type::<IstftParams>();
    }

    // =================================================================
    // 11. GPU buffer slice handle
    // =================================================================

    #[test]
    fn gpu_slice_type() {
        use nn::metal::GpuSlice;

        check_type::<GpuSlice>();
    }

    // =================================================================
    // 12. Typed tensor bridge
    // =================================================================

    #[test]
    fn tensor_bridge_types() {
        use nn::metal::{from_metal_buffer, MetalTensorExt, MetalTensorStorage};

        check_type::<MetalTensorStorage>();
        // from_metal_buffer has const generic D — import is the assertion.
        // MetalTensorExt is a trait — import is the assertion.
    }

    // =================================================================
    // 13. DynTensor GPU backend registration
    // =================================================================

    #[test]
    fn dyn_backend_registration() {
        use nn::metal::{register_metal_dyn_backend, MetalTensorData};

        check_type::<MetalTensorData>();
        let _ = register_metal_dyn_backend;
    }

    // =================================================================
    // 14. GPU command buffer batching
    // =================================================================

    #[test]
    fn gpu_scope_functions() {
        // Generic closures — import alone is the compile-time assertion.
        use nn::metal::{encode_custom_dispatch, flush, with_gpu_scope};

        // flush is not generic — can verify directly.
        let _ = flush;
    }

    // =================================================================
    // 15. GPU dispatch profiling counters
    // =================================================================

    #[test]
    fn dispatch_stats_types_and_functions() {
        use nn::metal::{dispatch_stats, reset_counters, DispatchStats};

        check_type::<DispatchStats>();
        let _ = dispatch_stats;
        let _ = reset_counters;
    }

    // =================================================================
    // 16. Activation arena
    // =================================================================

    #[test]
    fn activation_arena_types_and_functions() {
        // Generic closure functions — import alone is the compile-time assertion.
        use nn::metal::{
            try_reset_active_arena, with_arena, with_decode_scope, without_arena, ActivationArena,
        };

        check_type::<ActivationArena>();
        let _ = try_reset_active_arena;
    }

    // =================================================================
    // 17. Arena statistics
    // =================================================================

    #[test]
    fn arena_stats_types_and_functions() {
        use nn::metal::{arena_stats, reset_arena_stats, ArenaStats, PoolStats};

        check_type::<ArenaStats>();
        check_type::<PoolStats>();
        let _ = arena_stats;
        let _ = reset_arena_stats;
    }

    // =================================================================
    // 18. GPU-resident LoRA overlay
    // =================================================================

    #[test]
    fn lora_gpu_overlay_type() {
        use nn::metal::LoraGpuOverlay;

        check_type::<LoraGpuOverlay>();
    }

    // =================================================================
    // 19. Live weight editing
    // =================================================================

    #[test]
    fn weight_edit_types() {
        use nn::metal::{apply_weight_edit, WeightEditError, WeightEditResult, WeightEditSpec};

        check_type::<WeightEditError>();
        check_type::<WeightEditResult>();
        // WeightEditSpec<'a> has a lifetime parameter.
        check_type::<WeightEditSpec<'_>>();
        let _ = apply_weight_edit;
    }

    // =================================================================
    // 20. Live edit apply
    // =================================================================

    #[test]
    fn live_edit_types() {
        use nn::metal::{ApplyReceipt, DeltaApplyReceipt, LiveEditApply, LiveEditError};

        check_type::<ApplyReceipt>();
        check_type::<DeltaApplyReceipt>();
        check_type::<LiveEditApply>();
        check_type::<LiveEditError>();
    }

    // =================================================================
    // 21. Compiled model
    // =================================================================

    #[test]
    fn compiled_model_types() {
        use nn::metal::{CompiledModel, CompiledModelError};

        check_type::<CompiledModel>();
        check_type::<CompiledModelError>();
    }

    // =================================================================
    // 22. Metallib loading
    // =================================================================

    #[test]
    fn metallib_loader_functions() {
        use nn::metal::{load_metallib, pipelines_from_metallib, precompiled_metallib_path};

        let _ = load_metallib;
        let _ = pipelines_from_metallib;
        let _ = precompiled_metallib_path;
    }

    // =================================================================
    // 23. Kokoro compiled TTS pipeline
    // =================================================================

    #[test]
    fn kokoro_precompile_function() {
        // precompile_kokoro_msl has an impl AsRef<Path> generic — import is assertion.
        use nn::metal::precompile_kokoro_msl;
    }

    #[test]
    fn kokoro_compiled_types() {
        use nn::metal::{
            Certificate, CompiledKokoro, CompiledKokoroError, DiagnosticOutput, DispatchSummary,
            F16AutocastConfig, MemoryBreakdown, PrecompileResult, PrecompileShapes,
            StepEncodeResult, StepF0EnergyResult, StepGeneratorResult, StepProsodyResult,
            StepRegulateResult, StyleSplit, SynthesisIntermediates, TimingReport,
        };

        check_type::<Certificate>();
        check_type::<CompiledKokoro>();
        check_type::<CompiledKokoroError>();
        check_type::<DiagnosticOutput>();
        check_type::<DispatchSummary>();
        check_type::<F16AutocastConfig>();
        check_type::<MemoryBreakdown>();
        check_type::<PrecompileResult>();
        check_type::<PrecompileShapes>();
        check_type::<StepEncodeResult>();
        check_type::<StepF0EnergyResult>();
        check_type::<StepGeneratorResult>();
        check_type::<StepProsodyResult>();
        check_type::<StepRegulateResult>();
        check_type::<StyleSplit>();
        check_type::<SynthesisIntermediates>();
        check_type::<TimingReport>();
    }

    // =================================================================
    // 24. Kokoro chorus (multi-voice synthesis)
    // =================================================================

    #[test]
    fn kokoro_chorus_types() {
        use nn::metal::{ChorusConfig, ChorusGpuSynth, GpuSynth, KokoroChorus};

        check_type::<KokoroChorus>();
        check_type::<ChorusConfig>();
        check_type::<ChorusGpuSynth<'_>>();
        check_type::<GpuSynth<'_>>();
    }

    // =================================================================
    // 25. Kokoro streaming sessions
    // =================================================================

    #[test]
    fn kokoro_streaming_types() {
        use nn::metal::{
            ChorusChunkMode, CompiledKokoroStreamingSession, StreamingChorusSession,
            StreamingKokoroSession,
        };

        check_type::<StreamingKokoroSession>();
        check_type::<StreamingChorusSession>();
        check_type::<CompiledKokoroStreamingSession>();
        check_type::<ChorusChunkMode>();
    }

    // =================================================================
    // 26. Non-blocking GPU execution (GpuFence, GpuAudioHandle)
    // =================================================================

    #[test]
    fn gpu_fence_types() {
        use nn::metal::{AsyncGpuResult, GpuFence, GpuFuture};

        check_type::<GpuFence>();
        check_type::<GpuFuture>();
        // AsyncGpuResult is generic — verify with concrete type.
        check_type::<AsyncGpuResult<Vec<f32>>>();
    }

    #[test]
    fn gpu_audio_handle_type() {
        use nn::metal::GpuAudioHandle;

        check_type::<GpuAudioHandle>();
    }

    // =================================================================
    // 27. Segment cache configuration
    // =================================================================

    #[test]
    fn segment_cache_types() {
        use nn::metal::{EvictionPolicy, SegmentCacheConfig, SegmentCacheStats, ShapeKeyedCache};

        check_type::<SegmentCacheConfig>();
        check_type::<EvictionPolicy>();
        check_type::<SegmentCacheStats>();
        // ShapeKeyedCache is generic — verify with concrete types.
        check_type::<ShapeKeyedCache<String>>();
    }

    // =================================================================
    // 28. Compiled model builder
    // =================================================================

    #[test]
    fn compiled_model_builder_type() {
        use nn::metal::CompiledModelBuilder;

        check_type::<CompiledModelBuilder<'_>>();
    }

    // =================================================================
    // 29. Phantom-typed pipeline tensors
    // =================================================================

    #[test]
    fn typed_pipeline_tensors() {
        use nn::metal::{
            PipelineTensor, TypedEncodeResult, TypedF0EnergyResult, TypedGeneratorResult,
            TypedProsodyResult, TypedRegulateResult,
        };

        // These are phantom-typed wrappers — import is the assertion.
        let _: Option<TypedEncodeResult> = None;
        let _: Option<TypedProsodyResult> = None;
        let _: Option<TypedRegulateResult> = None;
        let _: Option<TypedF0EnergyResult> = None;
        let _: Option<TypedGeneratorResult> = None;
    }

    // =================================================================
    // 30. Optimization reports
    // =================================================================

    #[test]
    fn optimization_report_types() {
        use nn::metal::{
            diff_reports, ContractStatus, IterationVerdict, OptimizationReport, ReportDelta,
            ReportError,
        };

        check_type::<OptimizationReport>();
        check_type::<ContractStatus>();
        check_type::<ReportError>();
        check_type::<ReportDelta>();
        check_type::<IterationVerdict>();
        let _ = diff_reports;
    }

    // =================================================================
    // 31. GPU scope extended (submit, sync, ScopeExitMode)
    // =================================================================

    #[test]
    fn gpu_scope_extended_types() {
        use nn::metal::{submit, sync, with_scope_exit_mode, ScopeExitMode};

        check_type::<ScopeExitMode>();
        let _ = submit;
        let _ = sync;
    }

    // =================================================================
    // 32. Memory monitoring
    // =================================================================

    #[test]
    fn memory_monitoring_types() {
        use nn::metal::{
            metal_allocated_bytes, metal_allocated_mb, metal_budget_bytes, rss_bytes, rss_mb,
            RssSnapshot, RssTracker,
        };

        check_type::<RssSnapshot>();
        check_type::<RssTracker>();
    }

    #[test]
    fn memory_profiler_types() {
        use nn::metal::{
            BufferCategory, GpuMemoryProfiler, GpuMemorySnapshot, MemoryBreakdownByCategory,
        };

        check_type::<GpuMemoryProfiler>();
        check_type::<GpuMemorySnapshot>();
        check_type::<BufferCategory>();
        check_type::<MemoryBreakdownByCategory>();
    }

    // =================================================================
    // 33. Arena estimation and capacity
    // =================================================================

    #[test]
    fn arena_estimation_types() {
        use nn::metal::{estimate_arena_peak_bytes, ArenaEstimate};

        check_type::<ArenaEstimate>();
        // estimate_arena_peak_bytes takes impl IntoIterator — call with concrete type.
        let _result = estimate_arena_peak_bytes(std::iter::empty::<usize>());
    }

    #[test]
    fn arena_capacity_functions() {
        use nn::metal::{
            arena_capacity, default_arena_total_growth_count, ensure_default_arena_capacity,
            estimate_arena_peak_from_shapes,
        };

        let _ = arena_capacity;
        let _ = default_arena_total_growth_count;
        let _ = ensure_default_arena_capacity;
        let _ = estimate_arena_peak_from_shapes;
    }

    // =================================================================
    // 34. Cache statistics
    // =================================================================

    #[test]
    fn cache_stats_types() {
        use nn::metal::{CacheStats, CacheStatsSnapshot};

        check_type::<CacheStats>();
        check_type::<CacheStatsSnapshot>();
    }

    // =================================================================
    // 35. GPU buffer utilities
    // =================================================================

    #[test]
    fn to_standalone_function() {
        use nn::metal::to_standalone;
        let _ = to_standalone;
    }

    // =================================================================
    // 36. Document processing
    // =================================================================

    #[test]
    fn dpdf_types() {
        use nn::metal::{DpdfImagePreprocessMetal, DpdfPipelineMetal};

        check_type::<DpdfImagePreprocessMetal>();
        check_type::<DpdfPipelineMetal>();
    }

    // =================================================================
    // 37. Kokoro audio GPU helper
    // =================================================================

    #[test]
    fn kokoro_audio_gpu_function() {
        use nn::metal::kokoro_forward_audio_gpu;
        let _ = kokoro_forward_audio_gpu;
    }

    // =================================================================
    // 38. Pipeline cache count
    // =================================================================

    #[test]
    fn precompiled_pipeline_count_function() {
        use nn::metal::precompiled_pipeline_count;
        let _ = precompiled_pipeline_count;
    }
}
