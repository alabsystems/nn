// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: verify `nn::metal` re-exports are wired correctly.
//!
//! Run with: `cargo test -p nn --features metal --test metal_facade`
//!
//! This test catches facade wiring regressions — missing `pub use`, feature
//! gate typos, or type visibility changes that would silently break consumers
//! who depend on `nn::metal::SileroVad` et al.

// All assertions are compile-time: if this file compiles with `--features metal`,
// the re-export chain is correct.

#[cfg(feature = "metal")]
mod facade_tests {
    // ── Model types ──────────────────────────────────────────────────────

    #[test]
    fn silero_vad_types_accessible() {
        use nn::metal::SileroVadWeights;
        use nn::metal::{SileroVad, SileroVadError, SileroVadOutput, SileroVadState};

        let _: Option<SileroVad> = None;
        let _: Option<SileroVadError> = None;
        let _: Option<SileroVadOutput> = None;
        let _: Option<SileroVadWeights> = None;

        // SileroVadState::zero() returns valid initial state.
        let state = SileroVadState::zero();
        assert_eq!(state.h_state.len(), 128);
        assert_eq!(state.c_state.len(), 128);
    }

    #[test]
    fn htdemucs_types_accessible() {
        use nn::metal::{HTDemucs, HTDemucsError, HTDemucsWeights, WeightLoadError};

        let _: Option<HTDemucs> = None;
        let _: Option<HTDemucsError> = None;
        let _: Option<HTDemucsWeights> = None;
        let _: Option<WeightLoadError> = None;
    }

    // ── Infrastructure types ─────────────────────────────────────────────

    #[test]
    fn backend_types_accessible() {
        use nn::metal::{MetalBackend, MetalContext, MetalElement, MetalError, PipelineCache};

        let _: Option<MetalBackend> = None;
        let _: Option<MetalContext> = None;
        let _: Option<MetalError> = None;
        let _: Option<PipelineCache> = None;

        // MetalElement is a trait — verify f32 implements it.
        fn _check<T: MetalElement>() {}
        _check::<f32>();
    }

    #[test]
    fn weight_loading_types_accessible() {
        use nn::metal::{TensorInfo, WeightError, WeightMap};

        let _: Option<WeightMap> = None;
        let _: Option<WeightError> = None;
        let _: Option<TensorInfo> = None;
    }

    #[test]
    fn dispatch_types_accessible() {
        use nn::metal::TensorDispatchError;

        // Verify dispatch functions are re-exported (import is the assertion).
        #[allow(unused_imports)]
        use nn::metal::{
            execute_tensor_dispatch, execute_tensor_dispatch_batched,
            execute_tensor_dispatch_readback,
        };

        #[allow(unused_imports)]
        use nn::metal::{DispatchMode, DispatchPlan};

        let _: Option<TensorDispatchError> = None;
        let _: Option<DispatchPlan> = None;
    }

    #[test]
    fn stft_types_accessible() {
        use nn::metal::{StftError, StftParams};

        #[allow(unused_imports)]
        use nn::metal::compute_stft_magnitude;

        let _: Option<StftError> = None;
        let _: Option<StftParams> = None;
    }

    #[test]
    fn tensor_bridge_types_accessible() {
        use nn::metal::MetalTensorStorage;

        #[allow(unused_imports)]
        use nn::metal::{from_metal_buffer, MetalTensorExt};

        let _: Option<MetalTensorStorage> = None;
    }

    #[test]
    fn compute_primitives_accessible() {
        use nn::metal::{
            BatchEncoder, CommandBatch, ComputeDispatch, ComputePipeline, KernelSource, MetalBuffer,
        };

        #[allow(unused_imports)]
        use nn::metal::compile_msl_pipeline;

        // Kernel dispatch primitives (#2289)
        use nn::metal::{BufferAccess, BufferBinding, KernelPipeline};

        let _: Option<MetalBuffer> = None;
        let _: Option<ComputePipeline> = None;
        let _: Option<ComputeDispatch> = None;
        let _: Option<CommandBatch> = None;
        let _: Option<BatchEncoder> = None;
        let _: Option<KernelSource> = None;
        let _: Option<KernelPipeline> = None;
        let _: Option<BufferAccess> = None;
        let _: Option<BufferBinding<'_>> = None;
    }

    // ── Segment detection types ─────────────────────────────────────────

    #[test]
    fn segment_types_accessible() {
        use nn::metal::{SegmentConfig, SpeechSegment};

        // SegmentConfig::default() returns dvoice-compatible thresholds.
        let config = SegmentConfig::default();
        assert!(
            (0.0..=1.0).contains(&config.threshold),
            "threshold should be a probability: {}",
            config.threshold,
        );
        assert!(
            config.min_speech_duration_ms > 0,
            "min_speech_duration_ms should be positive",
        );
        assert!(
            config.min_silence_duration_ms > 0,
            "min_silence_duration_ms should be positive",
        );

        // SpeechSegment is #[non_exhaustive] — constructed by get_speech_segments(),
        // consumed by reading public fields. Verify the type and field accessors
        // are importable (compile-time check + function signature acceptance).
        fn _accepts_segment(seg: &SpeechSegment) -> f32 {
            assert!(seg.end_sample >= seg.start_sample);
            assert!(seg.end_time >= seg.start_time);
            seg.duration()
        }
    }

    // ── Cross-type integration ───────────────────────────────────────────

    #[test]
    fn error_conversion_from_model_to_tensor() {
        // Verify From<SileroVadError> for TensorError works through the facade.
        use nn::metal::SileroVadError;
        use nn::TensorError;

        let model_err = SileroVadError::AudioLength {
            actual: 10,
            expected: 512,
        };
        let tensor_err: TensorError = model_err.into();
        let msg = format!("{tensor_err}");
        assert!(
            msg.contains("10"),
            "should contain actual length in message: {msg}"
        );
    }

    #[test]
    fn error_conversion_from_htdemucs_to_tensor() {
        use nn::metal::HTDemucsError;
        use nn::TensorError;

        let model_err = HTDemucsError::AudioTooShort {
            actual: 0,
            minimum: 1,
        };
        let tensor_err: TensorError = model_err.into();
        let msg = format!("{tensor_err}");
        assert!(
            msg.contains("minimum"),
            "should describe the minimum requirement: {msg}",
        );
    }

    #[test]
    fn core_types_alongside_metal() {
        // Verify core types and metal types can coexist in the same scope,
        // and that ? propagation from model errors to TensorError works.
        use nn::metal::SileroVadError;
        use nn::{Result, TensorError};

        fn _example() -> Result<()> {
            let err = SileroVadError::AudioLength {
                actual: 0,
                expected: 512,
            };
            Err(err)?
        }

        let result = _example();
        assert!(result.is_err());
        match result.unwrap_err() {
            TensorError::BackendFailure {
                domain, message, ..
            } => {
                assert_eq!(format!("{domain:?}"), "Metal");
                assert!(message.contains("512"));
            }
            other => panic!("expected BackendFailure, got: {other:?}"),
        }
    }

    // ── Re-exports added in #2289 ───────────────────────────────────────

    #[test]
    fn arena_stats_accessible() {
        use nn::metal::ArenaStats;

        #[allow(unused_imports)]
        use nn::metal::{arena_stats, reset_arena_stats};

        let _: Option<ArenaStats> = None;
    }

    #[test]
    fn metallib_loader_accessible() {
        #[allow(unused_imports)]
        use nn::metal::{load_metallib, pipelines_from_metallib, precompiled_metallib_path};
    }
}
