// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Public API surface test for dvoice integration types.
//!
//! Verifies that key types needed by dvoice are importable from the `nn`
//! crate root and from `nn::metal`. This test catches silent API breakage
//! when re-exports are removed or renamed.
//!
//! Run: `cargo test -p nn --features metal --test public_api_surface`

#![allow(dead_code, unused_imports)]

// =============================================================================
// Root-level convenience re-exports (nn::*)
// =============================================================================

#[cfg(feature = "metal")]
mod root_reexports {
    /// Helper: proves a concrete type is importable. Optimized away.
    fn check_type<T>() {}

    #[test]
    fn kokoro_chorus_at_root() {
        use nn::KokoroChorus;
        check_type::<KokoroChorus>();
    }

    #[test]
    fn gpu_fence_at_root() {
        use nn::GpuFence;
        check_type::<GpuFence>();
    }

    #[test]
    fn gpu_audio_handle_at_root() {
        use nn::GpuAudioHandle;
        check_type::<GpuAudioHandle>();
    }

    #[test]
    fn streaming_kokoro_session_at_root() {
        use nn::StreamingKokoroSession;
        check_type::<StreamingKokoroSession>();
    }

    #[test]
    fn streaming_chorus_session_at_root() {
        use nn::StreamingChorusSession;
        check_type::<StreamingChorusSession>();
    }
}

// =============================================================================
// Module-level re-exports (nn::metal::*)
// =============================================================================

#[cfg(feature = "metal")]
mod metal_reexports {
    fn check_type<T>() {}

    #[test]
    fn kokoro_chorus_under_metal() {
        use nn::metal::KokoroChorus;
        check_type::<KokoroChorus>();
    }

    #[test]
    fn gpu_fence_under_metal() {
        use nn::metal::GpuFence;
        check_type::<GpuFence>();
    }

    #[test]
    fn gpu_audio_handle_under_metal() {
        use nn::metal::GpuAudioHandle;
        check_type::<GpuAudioHandle>();
    }

    #[test]
    fn streaming_sessions_under_metal() {
        use nn::metal::{
            ChannelStreamingSession, CompiledKokoroStreamingSession, StreamingChorusSession,
            StreamingKokoroSession,
        };
        check_type::<StreamingKokoroSession>();
        check_type::<StreamingChorusSession>();
        check_type::<CompiledKokoroStreamingSession>();
        check_type::<ChannelStreamingSession>();
    }

    #[test]
    fn chorus_config_types_under_metal() {
        use nn::metal::{ChorusConfig, ChorusGpuSynth, GpuSynth};
        check_type::<ChorusConfig>();
        check_type::<ChorusGpuSynth<'_>>();
        check_type::<GpuSynth<'_>>();
    }

    #[test]
    fn gpu_future_types_under_metal() {
        use nn::metal::{AsyncGpuResult, GpuFuture};
        check_type::<GpuFuture>();
        check_type::<AsyncGpuResult<Vec<f32>>>();
    }
}

// =============================================================================
// Path equivalence: root re-export is the same type as metal module re-export
// =============================================================================

#[cfg(feature = "metal")]
mod path_equivalence {
    #[test]
    fn kokoro_chorus_same_type() {
        // If these are different types, this function would fail to compile.
        fn accept_root(_: nn::KokoroChorus) {}
        fn produce_metal() -> Option<nn::metal::KokoroChorus> {
            None
        }
        if let Some(v) = produce_metal() {
            accept_root(v);
        }
    }

    #[test]
    fn gpu_fence_same_type() {
        fn accept_root(_: nn::GpuFence) {}
        fn produce_metal() -> Option<nn::metal::GpuFence> {
            None
        }
        if let Some(v) = produce_metal() {
            accept_root(v);
        }
    }

    #[test]
    fn gpu_audio_handle_same_type() {
        fn accept_root(_: nn::GpuAudioHandle) {}
        fn produce_metal() -> Option<nn::metal::GpuAudioHandle> {
            None
        }
        if let Some(v) = produce_metal() {
            accept_root(v);
        }
    }

    #[test]
    fn streaming_kokoro_session_same_type() {
        fn accept_root(_: nn::StreamingKokoroSession) {}
        fn produce_metal() -> Option<nn::metal::StreamingKokoroSession> {
            None
        }
        if let Some(v) = produce_metal() {
            accept_root(v);
        }
    }

    #[test]
    fn streaming_chorus_session_same_type() {
        fn accept_root(_: nn::StreamingChorusSession) {}
        fn produce_metal() -> Option<nn::metal::StreamingChorusSession> {
            None
        }
        if let Some(v) = produce_metal() {
            accept_root(v);
        }
    }
}

// =============================================================================
// Root convert-model surface (torch.export + safetensors)
// =============================================================================

#[cfg(feature = "convert-model")]
mod convert_model_surface {
    fn check_type<T>() {}

    #[test]
    fn convert_config_available_at_root() {
        use nn::ConvertConfig;
        check_type::<ConvertConfig>();
    }

    #[test]
    fn converted_model_available_at_root() {
        use nn::ConvertedModel;
        check_type::<ConvertedModel>();
    }

    #[test]
    fn convert_error_available_at_root() {
        use nn::ConvertError;
        check_type::<ConvertError>();
    }

    #[test]
    fn convert_from_trace_available_at_root() {
        use nn::{convert_from_trace, ConvertConfig, ConvertError, ConvertedModel};
        let _: fn(
            &std::path::Path,
            &std::path::Path,
            &ConvertConfig,
        ) -> Result<ConvertedModel, ConvertError> = convert_from_trace;
    }

    #[test]
    fn convert_model_module_reexports_match_root() {
        use nn::convert_model::{convert_from_trace, ConvertConfig, ConvertError, ConvertedModel};
        let _: fn(
            &std::path::Path,
            &std::path::Path,
            &ConvertConfig,
        ) -> Result<ConvertedModel, ConvertError> = convert_from_trace;
    }
}

// =============================================================================
// Root import/report provenance surface
// =============================================================================

#[cfg(feature = "import")]
mod import_report_surface {
    fn check_type<T>() {}

    #[test]
    fn report_types_available_at_root() {
        use nn::{
            ConvertArtifactKind, ConvertCompositionMethod, ConvertIntakePath, ConvertProofStrength,
            ConvertReport, ConvertSoundnessMode, VerificationCoverage,
        };

        check_type::<ConvertReport>();
        check_type::<ConvertIntakePath>();
        check_type::<ConvertArtifactKind>();
        check_type::<ConvertCompositionMethod>();
        check_type::<ConvertSoundnessMode>();
        check_type::<ConvertProofStrength>();
        check_type::<VerificationCoverage>();

        let _: fn(&ConvertReport) -> String = ConvertReport::provenance_summary;
        let _: fn(&ConvertReport) -> &'static str = ConvertReport::artifact_readiness_note;
        let _: fn(ConvertIntakePath) -> &'static str = ConvertIntakePath::label;
        let _: fn(ConvertArtifactKind) -> &'static str = ConvertArtifactKind::label;
        let _: fn(ConvertCompositionMethod) -> &'static str = ConvertCompositionMethod::label;
        let _: fn(ConvertSoundnessMode) -> &'static str = ConvertSoundnessMode::label;
        let _: fn(ConvertProofStrength) -> &'static str = ConvertProofStrength::label;
    }

    #[test]
    fn verification_coverage_exposes_current_composition_classification_fields() {
        use nn::{
            ConvertCompositionMethod, ConvertProofStrength, ConvertSoundnessMode,
            VerificationCoverage,
        };

        let coverage = VerificationCoverage::default();
        assert!(coverage.composition_method.is_none());
        assert!(coverage.composition_soundness_mode.is_none());
        assert!(coverage.composition_proof_strength.is_none());

        let _: Option<ConvertCompositionMethod> = coverage.composition_method;
        let _: Option<ConvertSoundnessMode> = coverage.composition_soundness_mode;
        let _: Option<ConvertProofStrength> = coverage.composition_proof_strength;
    }

    #[test]
    fn report_root_reexports_match_import_module() {
        fn accept_root_report(_: nn::ConvertReport) {}
        fn accept_root_intake(_: nn::ConvertIntakePath) {}
        fn accept_root_artifact(_: nn::ConvertArtifactKind) {}
        fn accept_root_method(_: nn::ConvertCompositionMethod) {}
        fn accept_root_soundness(_: nn::ConvertSoundnessMode) {}
        fn accept_root_strength(_: nn::ConvertProofStrength) {}
        fn accept_root_coverage(_: nn::VerificationCoverage) {}

        fn produce_import_report() -> Option<nn::import::ConvertReport> {
            None
        }
        fn produce_import_intake() -> Option<nn::import::ConvertIntakePath> {
            None
        }
        fn produce_import_artifact() -> Option<nn::import::ConvertArtifactKind> {
            None
        }
        fn produce_import_method() -> Option<nn::import::ConvertCompositionMethod> {
            None
        }
        fn produce_import_soundness() -> Option<nn::import::ConvertSoundnessMode> {
            None
        }
        fn produce_import_strength() -> Option<nn::import::ConvertProofStrength> {
            None
        }
        fn produce_import_coverage() -> Option<nn::import::VerificationCoverage> {
            None
        }

        if let Some(report) = produce_import_report() {
            accept_root_report(report);
        }
        if let Some(intake) = produce_import_intake() {
            accept_root_intake(intake);
        }
        if let Some(artifact) = produce_import_artifact() {
            accept_root_artifact(artifact);
        }
        if let Some(method) = produce_import_method() {
            accept_root_method(method);
        }
        if let Some(soundness) = produce_import_soundness() {
            accept_root_soundness(soundness);
        }
        if let Some(strength) = produce_import_strength() {
            accept_root_strength(strength);
        }
        if let Some(coverage) = produce_import_coverage() {
            accept_root_coverage(coverage);
        }
    }
}

// =============================================================================
// Root import-metal convert builder surface
// =============================================================================

#[cfg(feature = "import-metal")]
mod import_metal_convert_surface {
    #[test]
    fn root_convert_returns_builder() {
        use nn::{convert, ConvertBuilder};

        let _: for<'graph> fn(
            &'graph std::path::Path,
            &'graph std::path::Path,
            &'graph nn::metal::PipelineCache,
        ) -> ConvertBuilder<'graph> = convert;
    }
}

// =============================================================================
// Root multi-segment exported-artifact import surface
// =============================================================================

#[cfg(feature = "import")]
mod multi_segment_import_surface {
    fn check_type<T>() {}

    #[test]
    fn multi_segment_import_types_available_at_root() {
        use nn::{
            convert_multi_segment, convert_single_segment, MultiSegmentError, MultiSegmentModel,
        };

        check_type::<MultiSegmentModel>();
        check_type::<MultiSegmentError>();
        let _: fn(
            &[(String, serde_json::Value)],
            &std::path::Path,
        ) -> Result<MultiSegmentModel, MultiSegmentError> = convert_multi_segment;
        let _: fn(
            &serde_json::Value,
            &std::path::Path,
        ) -> Result<MultiSegmentModel, MultiSegmentError> = convert_single_segment;
    }

    #[test]
    fn multi_segment_import_module_reexports_match_root() {
        use nn::import::{
            convert_multi_segment, convert_single_segment, MultiSegmentError, MultiSegmentModel,
        };

        check_type::<MultiSegmentModel>();
        check_type::<MultiSegmentError>();
        let _: fn(
            &[(String, serde_json::Value)],
            &std::path::Path,
        ) -> Result<MultiSegmentModel, MultiSegmentError> = convert_multi_segment;
        let _: fn(
            &serde_json::Value,
            &std::path::Path,
        ) -> Result<MultiSegmentModel, MultiSegmentError> = convert_single_segment;
    }
}

// =============================================================================
// Root multi-segment exported-artifact Metal surface
// =============================================================================

#[cfg(feature = "import-metal")]
mod multi_segment_metal_surface {
    fn check_type<T>() {}

    #[test]
    fn multi_segment_metal_types_available_at_root() {
        use nn::{
            compile_multi_segment, convert_multi_segment_to_metal, CompiledMultiSegmentModel,
            MultiSegmentCompileError, MultiSegmentModel,
        };

        check_type::<CompiledMultiSegmentModel>();
        check_type::<MultiSegmentCompileError>();
        let _: fn(
            &MultiSegmentModel,
            &nn::metal::PipelineCache,
        ) -> Result<CompiledMultiSegmentModel, MultiSegmentCompileError> = compile_multi_segment;
        let _: fn(
            &[(String, serde_json::Value)],
            &std::path::Path,
            &nn::metal::PipelineCache,
        ) -> Result<CompiledMultiSegmentModel, MultiSegmentCompileError> =
            convert_multi_segment_to_metal;
    }

    #[test]
    fn multi_segment_metal_module_reexports_match_root() {
        use nn::import::{
            compile_multi_segment, convert_multi_segment_to_metal, CompiledMultiSegmentModel,
            MultiSegmentCompileError, MultiSegmentModel,
        };

        check_type::<CompiledMultiSegmentModel>();
        check_type::<MultiSegmentCompileError>();
        let _: fn(
            &MultiSegmentModel,
            &nn::metal::PipelineCache,
        ) -> Result<CompiledMultiSegmentModel, MultiSegmentCompileError> = compile_multi_segment;
        let _: fn(
            &[(String, serde_json::Value)],
            &std::path::Path,
            &nn::metal::PipelineCache,
        ) -> Result<CompiledMultiSegmentModel, MultiSegmentCompileError> =
            convert_multi_segment_to_metal;
    }
}

// =============================================================================
// Exported-artifact Metal bridge surface
// =============================================================================

#[cfg(all(feature = "metal", feature = "convert-model"))]
mod convert_compile_surface {
    fn check_type<T>() {}

    #[test]
    fn exported_artifact_bridge_types_available_at_root() {
        use nn::{
            compile_metal_from_exported_artifacts, ConvertConfig, ConvertedModelMetadata,
            ExportedArtifactCompileError, ExportedArtifactMetalModel,
        };

        check_type::<ConvertedModelMetadata>();
        check_type::<ExportedArtifactCompileError>();
        check_type::<ExportedArtifactMetalModel>();
        let _: fn(
            &std::path::Path,
            &std::path::Path,
            &ConvertConfig,
            &nn::metal::PipelineCache,
        ) -> Result<ExportedArtifactMetalModel, ExportedArtifactCompileError> =
            compile_metal_from_exported_artifacts;
    }

    #[test]
    fn exported_artifact_bridge_module_reexports_match_root() {
        use nn::convert_compile::{
            compile_metal_from_exported_artifacts, ConvertedModelMetadata,
            ExportedArtifactCompileError, ExportedArtifactMetalModel,
        };

        check_type::<ConvertedModelMetadata>();
        check_type::<ExportedArtifactCompileError>();
        check_type::<ExportedArtifactMetalModel>();
        let _: fn(
            &std::path::Path,
            &std::path::Path,
            &nn::ConvertConfig,
            &nn::metal::PipelineCache,
        ) -> Result<ExportedArtifactMetalModel, ExportedArtifactCompileError> =
            compile_metal_from_exported_artifacts;
    }

    #[test]
    fn exported_artifact_report_bridge_types_available_at_root() {
        use nn::{
            compile_exported_artifacts, compile_metal_from_exported_artifacts_with_report,
            ConvertConfig, ExportedArtifactCompileError, ExportedArtifactMetalModelWithReport,
        };

        check_type::<ExportedArtifactMetalModelWithReport>();
        let _: fn(
            &std::path::Path,
            &std::path::Path,
            &ConvertConfig,
            &nn::metal::PipelineCache,
        )
            -> Result<ExportedArtifactMetalModelWithReport, ExportedArtifactCompileError> =
            compile_exported_artifacts;
        let _: fn(
            &std::path::Path,
            &std::path::Path,
            &ConvertConfig,
            &nn::metal::PipelineCache,
        )
            -> Result<ExportedArtifactMetalModelWithReport, ExportedArtifactCompileError> =
            compile_metal_from_exported_artifacts_with_report;
    }

    #[test]
    fn exported_artifact_report_bridge_module_reexports_match_root() {
        use nn::convert_compile::{
            compile_exported_artifacts, compile_metal_from_exported_artifacts_with_report,
            ExportedArtifactCompileError, ExportedArtifactMetalModelWithReport,
        };

        check_type::<ExportedArtifactMetalModelWithReport>();
        let _: fn(
            &std::path::Path,
            &std::path::Path,
            &nn::ConvertConfig,
            &nn::metal::PipelineCache,
        )
            -> Result<ExportedArtifactMetalModelWithReport, ExportedArtifactCompileError> =
            compile_exported_artifacts;
        let _: fn(
            &std::path::Path,
            &std::path::Path,
            &nn::ConvertConfig,
            &nn::metal::PipelineCache,
        )
            -> Result<ExportedArtifactMetalModelWithReport, ExportedArtifactCompileError> =
            compile_metal_from_exported_artifacts_with_report;
    }
}
