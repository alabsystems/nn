// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic model definitions, builders, and signal processing.
//!
//! This crate contains model-specific code that depends only on nn-core
//! and nn-dsl — not on any GPU backend (Metal, CUDA, Vulkan). Model
//! builders produce `TensorKernelDef` (from nn-dsl) and can be consumed
//! by any backend for dispatch, or by nn-verify for composition testing.
//!
//! # Contents
//!
//! - **stft** — CPU-side Short-Time Fourier Transform (magnitude spectrogram)
//! - **istft** — Inverse STFT via overlap-add (time-domain reconstruction)
//! - **demucs_shared** — Shared Demucs architecture constants and builder helpers
//! - **demucs_transformer_weights** — Demucs transformer bottleneck weight types
//! - **demucs_temporal_weights** — Demucs temporal encoder/decoder weight types
//! - **demucs_spectral_weights** — Demucs spectral encoder/decoder weight types
//! - **demucs_temporal_encoder_builders** — Temporal encoder TensorKernelDef builders
//! - **demucs_temporal_decoder_builders** — Temporal decoder TensorKernelDef builders
//! - **demucs_spectral_encoder_builders** — Spectral encoder TensorKernelDef builders
//! - **demucs_spectral_decoder_builders** — Spectral decoder TensorKernelDef builders
//! - **demucs_transformer_helpers** — CPU-side transpose and positional embedding helpers
//! - **demucs_transformer_constants** — HTDemucs transformer architecture constants
//! - **demucs_transformer_builders** — Transformer TensorKernelDef builders
//! - **demucs_transformer_validate** — Transformer weight validation and weight map builders
//! - **silero_vad_builders** — Silero VAD encoder block configs and TensorKernelDef builders
//! - **kokoro_tts** — Kokoro-82M text-to-speech model (config, prosody, text encoder, signal via `#[path]`)
//! - **kokoro_decoder** — Kokoro ISTFTNet generator (waveform decoder)
//! - **kokoro_resblock** — Residual block for Kokoro ISTFTNet decoder (AdaIN+Snake+Conv1d)
//! - **kokoro_error** — Kokoro error types and NaN/Inf validation
//! - **kokoro_f0** — Kokoro F0/energy prediction
//! - **kokoro_forward_stft** — Forward STFT for Kokoro harmonic source generation
//! - **kokoro_full_decoder** — Kokoro Stage 1 preprocessing + ISTFTNet Generator
//! - **kokoro_audio** — Kokoro iSTFT audio reconstruction (forward_audio, private)
//! - **kokoro_istft** — Kokoro iSTFT waveform synthesis (pub(crate))
//! - **kokoro_source** — SineGen + SourceModule harmonic excitation source
//! - **kokoro_chorus** — Multi-voice chorus types and audio mixing (ChorusConfig, mix_voices)
//! - **kokoro_chorus_crossfade** — Streaming crossfade optimizer for chunk boundaries (adaptive window, phase-aligned)
//! - **kokoro_chorus_ensemble** — Classic ensemble chorus effect (modulated delay lines, LFO, stereo)
//! - **kokoro_chorus_saturation** — Harmonic warmth and saturation (Tape/Tube/Console/Warm modes)
//! - **kokoro_streaming** — Streaming synthesis API contract (AudioChunk, crossfade, assembly)
//! - **kokoro_tokenizer** — Phoneme string → token IDs (data-driven vocab, chunking)
//! - **kokoro_voice_pack** — Voice pack loading (safetensors → named style embeddings)
//! - **kokoro_g2p** — Espeak IPA → Kokoro phoneme remapping (misaki port)
//! - **plbert** — PlBert (ALBERT) text encoder for Kokoro TTS
//! - **ecapa_tdnn** — ECAPA-TDNN speaker encoder for speaker verification
//! - **ecapa_tdnn_block** — SE-Res2 block for ECAPA-TDNN speaker verification
//! - **wespeaker_resnet34** — WeSpeaker ResNet34 speaker embeddings
//! - **convert** — Backend-agnostic PyTorch model import (`ConvertConfig`, `ConvertedModel`)

use std::borrow::Cow;

// -- Build-time model manifest validation (generated from configs/*.toml) ----
#[allow(dead_code)]
mod manifest_checks;

pub mod demucs_shared;
pub mod demucs_spectral_decoder_builders;
pub mod demucs_spectral_encoder_builders;
pub mod demucs_spectral_weights;
pub mod demucs_temporal_decoder_builders;
pub mod demucs_temporal_encoder_builders;
pub mod demucs_temporal_weights;
pub mod demucs_transformer_builders;
pub mod demucs_transformer_constants;
pub mod demucs_transformer_helpers;
pub mod demucs_transformer_validate;
pub mod demucs_transformer_weights;
pub mod istft;
pub mod silero_vad_builders;
pub mod stft;

// -- espeak-ng FFI binding (feature-gated) ------------------------------------
#[cfg(feature = "espeak")]
pub mod espeak_ffi;

// -- Kokoro TTS model (extracted from nn-core) --------------------------------
mod kokoro_audio;
pub mod kokoro_chorus;
pub mod kokoro_chorus_adaptive_dynamics;
pub mod kokoro_chorus_air_absorption;
pub mod kokoro_chorus_alignment;
pub mod kokoro_chorus_auto_eq;
pub mod kokoro_chorus_auto_mix;
pub mod kokoro_chorus_automation;
pub mod kokoro_chorus_bass_management;
pub mod kokoro_chorus_bleed;
pub mod kokoro_chorus_blend;
pub mod kokoro_chorus_breath;
pub mod kokoro_chorus_character;
pub mod kokoro_chorus_convolution;
pub mod kokoro_chorus_crossfade;
pub mod kokoro_chorus_decorrelation;
pub mod kokoro_chorus_delay;
pub mod kokoro_chorus_depth_staging;
pub mod kokoro_chorus_detune;
pub mod kokoro_chorus_dither;
pub mod kokoro_chorus_doubler;
pub mod kokoro_chorus_ducking;
pub mod kokoro_chorus_dynamic_eq;
pub mod kokoro_chorus_dynamics;
pub mod kokoro_chorus_ensemble;
pub mod kokoro_chorus_eq;
pub mod kokoro_chorus_exciter;
pub mod kokoro_chorus_formant;
pub mod kokoro_chorus_formant_tune;
pub mod kokoro_chorus_freeze;
pub mod kokoro_chorus_gain_staging;
pub mod kokoro_chorus_gate;
pub mod kokoro_chorus_harmonic_tuner;
pub mod kokoro_chorus_hrtf;
pub mod kokoro_chorus_humanize;
pub mod kokoro_chorus_intelligibility;
pub mod kokoro_chorus_intonation;
pub mod kokoro_chorus_limiter;
pub mod kokoro_chorus_loudness;
pub mod kokoro_chorus_loudness_curve;
pub mod kokoro_chorus_masking_compensator;
pub mod kokoro_chorus_mic_model;
pub mod kokoro_chorus_micro_pitch;
pub mod kokoro_chorus_mix_analyzer;
pub mod kokoro_chorus_multiband_stereo;
pub mod kokoro_chorus_onset_sync;
pub mod kokoro_chorus_output;
pub mod kokoro_chorus_oversample;
pub mod kokoro_chorus_pipeline;
pub mod kokoro_chorus_pitch_correct;
pub mod kokoro_chorus_presence;
pub mod kokoro_chorus_preset_library;
pub mod kokoro_chorus_reverb;
pub mod kokoro_chorus_reverb_streaming;
pub mod kokoro_chorus_room;
pub mod kokoro_chorus_saturation;
pub mod kokoro_chorus_shimmer;
pub mod kokoro_chorus_sibilance;
pub mod kokoro_chorus_spatial;
pub mod kokoro_chorus_spectral_fill;
pub mod kokoro_chorus_spectral_match;
pub mod kokoro_chorus_stage;
pub mod kokoro_chorus_stereo;
pub mod kokoro_chorus_stereo_analysis;
pub mod kokoro_chorus_stereo_optimizer;
pub mod kokoro_chorus_streaming;
pub mod kokoro_chorus_sub_bass;
pub mod kokoro_chorus_thickener;
pub mod kokoro_chorus_tilt;
pub mod kokoro_chorus_transient;
pub mod kokoro_chorus_transient_align;
pub mod kokoro_chorus_vibrato;
pub mod kokoro_chorus_vocal_chain;
pub mod kokoro_chorus_vocal_tract;
pub mod kokoro_chorus_voice_alloc;
pub mod kokoro_chorus_vowel_align;
pub mod kokoro_chorus_warmth;
pub mod kokoro_chorus_width;
pub mod kokoro_decoder;
pub mod kokoro_error;
pub mod kokoro_f0;
pub mod kokoro_forward_stft;
pub mod kokoro_full_decoder;
pub mod kokoro_g2p;
pub(crate) mod kokoro_istft;
pub(crate) mod kokoro_number_words;
pub mod kokoro_pipeline;
pub mod kokoro_resblock;
pub mod kokoro_source;
pub mod kokoro_streaming;
pub mod kokoro_text_preprocess;
pub mod kokoro_tokenizer;
#[cfg(feature = "training")]
pub mod kokoro_trainable_decoder;
pub mod kokoro_tts;
pub mod kokoro_vocab;
pub mod kokoro_voice_pack;
pub mod plbert;

// -- ECAPA-TDNN speaker encoder (speaker verification / P4 moonshot) ---------
mod ecapa_tdnn;
pub use ecapa_tdnn::EcapaTdnn;
mod ecapa_tdnn_block;
pub use ecapa_tdnn_block::SERes2Block;

// -- WeSpeaker ResNet34 speaker embeddings (#2294) ----------------------------
mod wespeaker_resnet34;
pub use wespeaker_resnet34::WeSpeakerResNet34;

// -- Granite-Docling-258M VLM (SigLIP2 + Granite-165M decoder, #3864) ----------
pub mod granite_docling;

// -- Qwen3-VL multimodal model (#3868) -----------------------------------------
pub mod qwen3_vl;

// -- Qwen3-VL quantized weight loading (GPTQ/AWQ) (#3897) --------------------
pub mod qwen3_vl_quantized;
pub use qwen3_vl_quantized::{
    estimate_memory_bytes, QuantMethod, QuantizedLayerError, QuantizedLinearLayer,
    Qwen3VLQuantConfig,
};

// -- GLM-OCR 0.9B with Multi-Token Prediction (#3876) -------------------------
pub mod glm_ocr;

// -- FireRed-OCR (Qwen3-VL-2B) document OCR (#3899) --------------------------
pub mod firered_ocr;

// -- DocLayout-YOLO document layout detection (#3865) -------------------------
pub mod doclayout_yolo;
pub use doclayout_yolo::{DocLayoutYolo, DocLayoutYoloConfig};

// -- RT-DETRv2 document layout detection (#4189, docling_rs) ------------------
pub mod rt_detr;
pub use rt_detr::{RtDetr, RtDetrBackboneVariant, RtDetrConfig};

// -- Weighted Box Fusion for ensemble detection merging (#4189, docling_rs) ----
pub mod weighted_box_fusion;
pub use weighted_box_fusion::{ScoredBox, WbfConfig, WeightedBoxFusion};

// -- dpdf end-to-end document inference pipeline (#3879) ----------------------
pub mod dpdf_pipeline;
pub use dpdf_pipeline::{DocumentOutput, DocumentRegion, DpdfPipeline, PageOutput, PipelineConfig};

// -- dpdf pipeline DynTensor forward passes (#3903) ---------------------------
pub mod dpdf_pipeline_forward;
pub use dpdf_pipeline_forward::{DpdfInferencePipeline, DpdfModelWeights};

// -- dpdf image preprocessing pipeline (#3900) --------------------------------
pub mod dpdf_image_preprocess;

// -- dpdf pipeline benchmark infrastructure (#3930) ---------------------------
pub mod dpdf_benchmark;

// -- dpdf document output export (JSON, HTML, Markdown, CSV) (#3922) ---------
pub mod dpdf_export;
pub use dpdf_export::{
    CsvTableExporter, DocumentExporter, ExportError, HtmlExporter, JsonExporter, MarkdownExporter,
};

// -- dpdf model registry and dispatch routing (#3937) -------------------------
pub mod dpdf_registry;
pub use dpdf_registry::{DpdfModelRegistry, ModelEntry, ModelType};

// -- dpdf streaming (chunked) document processing (#3932) --------------------
pub mod dpdf_streaming;
pub use dpdf_streaming::{ChunkOutput, StreamingConfig, StreamingError, StreamingPipeline};

// -- dpdf document region post-processing (#3889) ----------------------------
pub mod dpdf_postprocess;
pub use dpdf_postprocess::{FusionPriority, PostProcessConfig};

// -- Table Transformer (DETR) for table detection/structure (#3875) -----------
pub mod table_transformer;

// -- UniTable linear-projection table structure model (#4320) -----------------
pub mod unitable;

// -- LayoutLMv3 multi-modal form entity labeling (#4320) ----------------------
pub mod layoutlmv3;

// -- Bipartite matching (Hungarian algorithm) for DETR post-processing (#4320) -
pub mod bipartite_matching;

// -- Table cell detection post-processing: NMS, box decoding (#4320) ----------
pub mod table_cell_postprocess;
pub use table_cell_postprocess::{postprocess_table_detections, TableCellPostProcessConfig};

// -- Form field association: key-value pairing from LayoutLMv3 (#4320) --------
pub mod form_field_association;
pub use form_field_association::{
    extract_form_fields, EntityTag, EntityType, FormAssociationConfig, FormExtractionResult,
    FormField,
};

// -- Table span recognition: row/column span inference (#4320) ----------------
pub mod table_span_recognition;
pub use table_span_recognition::{
    recognize_spans, validate_span_coverage, GridBoundaries, RawCellDetection,
    SpanRecognitionConfig,
};

// -- Table structure recognition model config (#4320) -------------------------
pub mod table_structure_model_config;
pub use table_structure_model_config::{
    BackboneVariant, TableBackboneConfig, TableDecoderConfig, TablePostProcessConfig,
    TableStructureModelConfig,
};

// -- Form field detection model config (#4320) --------------------------------
pub mod form_field_model_config;
pub use form_field_model_config::{
    FieldDetectionHeadConfig, FormFieldModelConfig, ValueExtractionHeadConfig,
};

// -- Layout analysis model config (#4320) -------------------------------------
pub mod layout_analysis_model_config;
pub use layout_analysis_model_config::{
    DocumentPreprocessConfig, LayoutAnalysisModelConfig, MultiScaleDetectionConfig, PanNeckConfig,
};

// -- Table + form integration pipeline (#4320) --------------------------------
pub mod table_form_integration;
pub use table_form_integration::{
    classify_regions, merge_results, summarize, ClassifiedRegion, ExtractionSummary,
    PageExtractionResult, RegionKind, TableExtractionResult, TableFormConfig,
};

// -- Table structure recognition: cell-level parsing (#3888) ------------------
pub mod table_structure;
pub use table_structure::{StructuredTable, TableCell, TableStructureConfig};

// -- PaddleOCR-VL-1.5 vision-language model (#3881) ---------------------------
pub mod paddle_ocr;
pub mod paddle_ocr_vision;
pub use paddle_ocr::{PaddleOcrVl, PaddleOcrVlConfig};

// -- nn::convert() backend-agnostic model import (#2293) ---------------------
pub mod convert;

// ---------------------------------------------------------------------------
// Backend-agnostic transformer build error
// ---------------------------------------------------------------------------

/// Errors from Demucs transformer construction (backend-agnostic).
///
/// Covers weight validation and IR construction. GPU dispatch errors are
/// backend-specific and live in the backend crate (e.g., `DemucsTransformerError`
/// in nn-metal wraps this via `#[from]`).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TransformerBuildError {
    /// Weight tensor has wrong element count.
    #[error("weight '{name}' expected {expected} elements, got {actual}")]
    WeightSize {
        name: Cow<'static, str>,
        expected: usize,
        actual: usize,
    },

    /// Dimension mismatch during construction.
    #[error("'{stage}' expected {expected} elements, got {actual}")]
    DimMismatch {
        stage: String,
        expected: usize,
        actual: usize,
    },

    /// Tensor IR construction error.
    #[error("tensor IR error: {0}")]
    TensorIr(#[from] nn_dsl::TensorIRError),
}

impl From<DemucsBuilderError> for TransformerBuildError {
    fn from(e: DemucsBuilderError) -> Self {
        match e {
            DemucsBuilderError::WeightSize {
                name,
                expected,
                actual,
            } => Self::WeightSize {
                name,
                expected,
                actual,
            },
            other => Self::DimMismatch {
                stage: other.to_string(),
                expected: 0,
                actual: 0,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Backend-agnostic Demucs builder error (shared across enc/dec/spectral)
// ---------------------------------------------------------------------------

/// Errors from Demucs builder construction (backend-agnostic).
///
/// Covers weight validation and conv dimension computation for the shared
/// builder functions in `demucs_shared` and the spectral encoder/decoder
/// builders. Backend-specific error types (e.g., `DemucsSpectralDecoderError`)
/// convert via `From<DemucsBuilderError>`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DemucsBuilderError {
    /// Weight tensor has wrong element count.
    #[error("weight '{name}' expected {expected} elements, got {actual}")]
    WeightSize {
        name: Cow<'static, str>,
        expected: usize,
        actual: usize,
    },

    /// Conv dimension computation failed (e.g., zero stride, padded < kernel).
    #[error("{msg}")]
    InvalidConvDim { msg: Cow<'static, str> },

    /// Block count mismatch in weight struct.
    #[error("{context}: expected {expected}, got {actual}")]
    BlockCountMismatch {
        context: Cow<'static, str>,
        expected: usize,
        actual: usize,
    },

    /// Conv1d output length computation failed.
    #[error("conv1d output length: {0}")]
    Conv1dOutLen(#[from] nn_core::TensorError),

    /// Tensor IR construction error.
    #[error("tensor IR: {0}")]
    TensorIr(#[from] nn_dsl::TensorIRError),
}

pub use demucs_shared::{
    build_dconv_sublayer, channels_at_depth, conv1d_output_len, validate_weight_size,
    DConvSubLayerInputs, AUDIO_CHANNELS, BASE_CHANNELS, DCONV_COMPRESS, DCONV_DEPTH, DCONV_KERNEL,
    DECODER_OUTPUT_CHANNELS, DECODER_REWRITE_KERNEL, DECODER_REWRITE_PADDING, GROUP_NORM_EPS,
    GROWTH, SPECTRAL_BASIC_DEPTH, SPECTRAL_CONV_PADDING, SPECTRAL_CONV_TR_PADDING, SPECTRAL_DEPTH,
    SPECTRAL_FREQ_EMB_DIM, SPECTRAL_FREQ_EMB_FEATURES, SPECTRAL_INPUT_CHANNELS,
    SPECTRAL_KERNEL_SIZE, SPECTRAL_OUTPUT_CHANNELS, SPECTRAL_REWRITE_KERNEL,
    SPECTRAL_REWRITE_PADDING, SPECTRAL_STRIDE, TEMPORAL_BASIC_DEPTH, TEMPORAL_CONV_PADDING,
    TEMPORAL_CONV_TR_PADDING, TEMPORAL_DEPTH, TEMPORAL_KERNEL_SIZE, TEMPORAL_STRIDE,
};
pub use istft::{IstftBasis, IstftError, IstftParams};
pub use stft::{compute_stft_magnitude, StftError, StftParams};

#[cfg(feature = "espeak")]
pub use espeak_ffi::{EspeakEngine, EspeakError};
pub use kokoro_chorus::{
    mix_voices, mix_voices_from_refs, mix_voices_stereo, mix_voices_with_config,
    pitch_shift_factor, ChorusConfig, VoiceInput, VoiceMix,
};
pub use kokoro_chorus_adaptive_dynamics::{AdaptiveDynamicsConfig, AdaptiveDynamicsProcessor};
pub use kokoro_chorus_alignment::{
    align_voices, apply_shift, cross_correlate, AlignmentConfig, CorrelationResult,
};
pub use kokoro_chorus_auto_eq::{AutoEqConfig, AutoEqProcessor, TargetCurve};
pub use kokoro_chorus_automation::{
    build_to_chorus, dynamic_swell, fade_to_intimate, AutomationConfig, AutomationTimeline,
    CrossfadeCurve, EffectEnables, MixAutomator, MixParams, SceneSnapshot, TimelineKeyframe,
};
pub use kokoro_chorus_bass_management::{BassManagementConfig, BassManager};
pub use kokoro_chorus_blend::{blend_voices, EnsembleBlendConfig, FormantShift, SpectralAlignment};
pub use kokoro_chorus_character::{
    apply_character_variation, CharacterConfig, CharacterPreset, VoiceCharacter,
};
pub use kokoro_chorus_crossfade::{
    compute_energy_envelope, find_zero_crossings, generate_adaptive_window, CrossfadeAnalysis,
    CrossfadeOptimizer, CrossfadeOptimizerConfig, CrossfadeOptimizerConfigBuilder,
};
pub use kokoro_chorus_detune::{
    apply_detune, cents_to_rate, AllpassInterpolator, DetuneConfig, DetuneDistribution,
    VoiceDetuner,
};
pub use kokoro_chorus_dither::{apply_dither, DitherConfig, DitherProcessor, DitherType};
pub use kokoro_chorus_doubler::{apply_doubler_per_voice, DoublerConfig, VocalDoubler};
pub use kokoro_chorus_ducking::{apply_sidechain, DuckingConfig, SidechainConfig, SpectralDucker};
pub use kokoro_chorus_dynamics::{
    BandCompressor, BandCompressorConfig, BusLimiter, DynamicsPreset, MultibandCompressor,
    MultibandCompressorConfig,
};
pub use kokoro_chorus_ensemble::{EnsembleConfig, EnsembleMode, EnsembleProcessor};
pub use kokoro_chorus_eq::{
    ChorusEQ, DeEsser, DeEsserConfig, EqConfig, EqPreset, MixBusConfig, MixBusProcessor,
};
pub use kokoro_chorus_formant::{
    shift_pitch_preserve_formant, simple_pitch_shift as formant_simple_pitch_shift,
    FormantPreserveConfig, FormantShifter,
};
pub use kokoro_chorus_formant_tune::{FormantBand, FormantTuneConfig, FormantTuner};
pub use kokoro_chorus_freeze::{FreezeConfig, SpectralFreezer};
pub use kokoro_chorus_gate::{apply_noise_gate, GateConfig, GateState, NoiseGate};
pub use kokoro_chorus_harmonic_tuner::{
    HarmonicAnalysis, HarmonicTunerConfig, HarmonicTunerProcessor,
};
pub use kokoro_chorus_hrtf::{
    arc as hrtf_arc, semicircle as hrtf_semicircle, surround as hrtf_surround, HrtfConfig,
    HrtfModel, HrtfPosition, HrtfProcessor,
};
pub use kokoro_chorus_humanize::{
    apply_humanize, apply_jitter, AmplitudeEnvelope, BreathPattern, HumanizeConfig, MicroTiming,
};
pub use kokoro_chorus_intonation::{
    correct_intonation, IntonationConfig, IntonationTracker, PitchInfo,
};
pub use kokoro_chorus_loudness::{
    bark_band_loudness, measure_true_peak, LoudnessConfig, LoudnessMeter, LoudnessWeighting,
};
pub use kokoro_chorus_micro_pitch::{MicroPitchConfig, MicroPitchProcessor};
pub use kokoro_chorus_multiband_stereo::{MultibandStereoConfig, MultibandStereoProcessor};
pub use kokoro_chorus_onset_sync::{OnsetSyncConfig, OnsetSynchronizer};
pub use kokoro_chorus_pipeline::{process_chorus, ChorusMasterConfig, ChorusMasterPipeline};
pub use kokoro_chorus_pitch_correct::{
    apply_pitch_correction, detect_pitch, MusicalScale, PitchCorrectConfig, PitchCorrector,
};
pub use kokoro_chorus_preset_library::{
    validate_preset, ChorusPreset, ConfigDifference, PresetComparison,
};
pub use kokoro_chorus_reverb::{ReverbConfig, StereoReverb};
pub use kokoro_chorus_reverb_streaming::StreamingReverb;
pub use kokoro_chorus_room::{EarlyReflections, RoomConfig};
pub use kokoro_chorus_saturation::{
    db_to_linear, linear_to_db, process_saturation, SaturationConfig, SaturationMode,
    SaturationProcessor,
};
pub use kokoro_chorus_sibilance::{
    aggressive_deess, align_sibilants, broadcast_balanced, gentle_deess, presence_enhance,
    SibilanceConfig, SibilanceMode, SibilanceProcessor,
};
pub use kokoro_chorus_spatial::{
    auto_layout_spatial, process_voice_spatial, SpatialConfig, SpatialProcessor,
    VoiceSpatialPosition,
};
pub use kokoro_chorus_spectral_match::{SpectralMatchConfig, SpectralMatcher};
pub use kokoro_chorus_stage::{
    default_order as chorus_default_order, format_pipeline as format_chorus_pipeline,
    validate_order as validate_chorus_order, ChorusPipelineStage,
};
pub use kokoro_chorus_stereo::{
    apply_stereo_mix, apply_stereo_mix_refs, constant_power_pan, default_voice_layout,
    interleave_stereo, StereoChorusConfig, StereoPanner, StereoPosition,
};
pub use kokoro_chorus_stereo_analysis::{StereoAnalysisConfig, StereoAnalyzer, StereoMetrics};
pub use kokoro_chorus_streaming::StreamingChorusSession;
pub use kokoro_chorus_sub_bass::{SubBassConfig, SubBassEnhancer};
pub use kokoro_chorus_tilt::{TiltConfig, TiltProcessor};
pub use kokoro_chorus_vocal_chain::{
    validate_chain, ChainWarning, VocalChain, VocalChainPreset, WarningSeverity,
};
pub use kokoro_chorus_vowel_align::{FormantTrack, VowelAlignConfig, VowelAligner};
pub use kokoro_chorus_warmth::{WarmthConfig, WarmthMode, WarmthProcessor};
pub use kokoro_error::KokoroError;
pub use kokoro_f0::{AdainResBlk1d, F0EnergyPredictor};
pub use kokoro_forward_stft::KokoroForwardStft;
pub use kokoro_full_decoder::{FullDecoder, Stage1ResBlk};
pub use kokoro_g2p::{EspeakRemapper, PhonemeLexicon, RemapTable};
pub use kokoro_pipeline::{chunks_to_tensors, KokoroSynth, KokoroTextPipeline, PipelineError};
pub use kokoro_source::{interp_downsample_gpu, interp_upsample_gpu, SineGen, SourceModule};
pub use kokoro_streaming::{
    assemble_streaming_chorus, assemble_streaming_chunks, concatenate_chunks, crossfade_chunks,
    crossfade_chunks_windowed, AudioChunk, CrossfadeWindow, KokoroStreamConfig, StreamingAssembler,
    StreamingKokoroSession,
};
pub use kokoro_text_preprocess::TextPreprocessor;
pub use kokoro_tokenizer::{KokoroTokenizer, KokoroVocab, MAX_PHONEME_TOKENS, PAD_TOKEN_ID};
#[cfg(feature = "training")]
pub use kokoro_trainable_decoder::{
    LoraConv1dAdapter, LoraGenerator, LoraResBlockPair, LoraStage1Block, MergedDecoderWeights,
    SingingLoraConfig, SingingStage, TrainableKokoroDecoder,
};
pub use kokoro_tts::{
    length_regulate, AdaLayerNorm, EncoderFeaturesResult, KokoroConfig, KokoroModel,
    ProsodyPredictor, TextEncoder, TextPipelineResult,
};
pub use kokoro_voice_pack::VoicePack;
pub use plbert::{PlBert, PlbertConfig};

// -- Kani proof harnesses for tokenizer + convert safety (#3630) -----------
#[cfg(kani)]
#[path = "kani_tokenizer_convert_proofs.rs"]
mod kani_tokenizer_convert_proofs;

// -- Kani proof harnesses for pipeline + chorus safety (#3642) -------------
#[cfg(kani)]
#[path = "kani_pipeline_chorus_proofs.rs"]
mod kani_pipeline_chorus_proofs;

// -- Kani proof harnesses for text preprocess safety (#3663) ----------------
#[cfg(kani)]
#[path = "kani_text_preprocess_proofs.rs"]
mod kani_text_preprocess_proofs;

// -- Kani proof harnesses for streaming assembly safety (#3663) -------------
#[cfg(kani)]
#[path = "kani_streaming_assembly_proofs.rs"]
mod kani_streaming_assembly_proofs;

// -- Kani proof harnesses for kokoro_tokenizer deep invariants (#3686) ------
#[cfg(kani)]
#[path = "kani_kokoro_tokenizer.rs"]
mod kani_kokoro_tokenizer;

// -- Kani proof harnesses for convert deep invariants (#3686) ---------------
#[cfg(kani)]
#[path = "kani_convert.rs"]
mod kani_convert;

// -- Kani proof harnesses for kokoro_pipeline deep invariants (#3701) -------
#[cfg(kani)]
#[path = "kani_kokoro_pipeline.rs"]
mod kani_kokoro_pipeline;

// -- Kani proof harnesses for tokenizer bounds + vocab validation (#3732) ---
#[cfg(kani)]
#[path = "kani_kokoro_tokenizer_bounds_proofs.rs"]
mod kani_kokoro_tokenizer_bounds_proofs;

// -- Kani proof harnesses for convert validation + layer consistency (#3732) -
#[cfg(kani)]
#[path = "kani_convert_validation_proofs.rs"]
mod kani_convert_validation_proofs;

// -- Kani proof harnesses for pipeline ordering + shape propagation (#3732) --
#[cfg(kani)]
#[path = "kani_kokoro_pipeline_shape_proofs.rs"]
mod kani_kokoro_pipeline_shape_proofs;

// -- Kani proof harnesses for kokoro_chorus deep invariants (#3701) ---------
#[cfg(kani)]
#[path = "kani_kokoro_chorus.rs"]
mod kani_kokoro_chorus;

#[cfg(kani)]
mod kani_demucs_spectral_decoder_builder_proofs;
#[cfg(kani)]
mod kani_demucs_spectral_decoder_builders;
#[cfg(kani)]
mod kani_istft;
#[cfg(kani)]
mod kani_istft_overlap_add_proofs;
#[cfg(kani)]
mod kani_kokoro_text_preprocess;
#[cfg(kani)]
mod kani_text_preprocess_unicode_proofs;

// -- Deep Kani harnesses for tokenizer, convert, pipeline (#3732) ----------
#[cfg(kani)]
#[path = "kani_convert_deep.rs"]
mod kani_convert_deep;
#[cfg(kani)]
#[path = "kani_kokoro_pipeline_deep.rs"]
mod kani_kokoro_pipeline_deep;
#[cfg(kani)]
#[path = "kani_kokoro_tokenizer_deep.rs"]
mod kani_kokoro_tokenizer_deep;

// -- Deep Kani harnesses for kokoro_chorus, plbert, stft (#3739) ------------
#[cfg(kani)]
#[path = "kani_kokoro_chorus_deep.rs"]
mod kani_kokoro_chorus_deep;
#[cfg(kani)]
#[path = "kani_plbert_deep.rs"]
mod kani_plbert_deep;
#[cfg(kani)]
#[path = "kani_stft_deep.rs"]
mod kani_stft_deep;

// -- Kani harnesses for under-covered areas (#3793) ---------------------------
#[cfg(kani)]
#[path = "kani_demucs_shared_proofs.rs"]
mod kani_demucs_shared_proofs;
#[cfg(kani)]
#[path = "kani_demucs_transformer_proofs.rs"]
mod kani_demucs_transformer_proofs;
#[cfg(kani)]
#[path = "kani_ecapa_tdnn_proofs.rs"]
mod kani_ecapa_tdnn_proofs;
#[cfg(kani)]
#[path = "kani_kokoro_config_proofs.rs"]
mod kani_kokoro_config_proofs;
#[cfg(kani)]
#[path = "kani_kokoro_error_proofs.rs"]
mod kani_kokoro_error_proofs;
#[cfg(kani)]
#[path = "kani_kokoro_number_words_proofs.rs"]
mod kani_kokoro_number_words_proofs;
#[cfg(kani)]
#[path = "kani_kokoro_signal_proofs.rs"]
mod kani_kokoro_signal_proofs;
#[cfg(kani)]
#[path = "kani_kokoro_streaming_proofs.rs"]
mod kani_kokoro_streaming_proofs;
#[cfg(kani)]
#[path = "kani_plbert_config_proofs.rs"]
mod kani_plbert_config_proofs;
#[cfg(kani)]
#[path = "kani_silero_vad_proofs.rs"]
mod kani_silero_vad_proofs;

#[cfg(kani)]
#[path = "kani_models_wave11.rs"]
mod kani_models_wave11;

// -- Kani proof harnesses for dpdf model builder configs (#3880) ------------
#[cfg(kani)]
#[path = "kani_dpdf_model_proofs.rs"]
mod kani_dpdf_model_proofs;

// -- Kani proof harnesses for Table Transformer + GLM-OCR configs (#3882) ---
#[cfg(kani)]
#[path = "kani_table_transformer_glm_ocr_proofs.rs"]
mod kani_table_transformer_glm_ocr_proofs;

// -- Kani proof harnesses for DpdfPipeline + PaddleOCR configs (#3887) ------
#[cfg(kani)]
#[path = "kani_dpdf_pipeline_paddle_ocr_proofs.rs"]
mod kani_dpdf_pipeline_paddle_ocr_proofs;

// -- Kani proof harnesses for dpdf_postprocess + table_structure (#3892) -----
#[cfg(kani)]
#[path = "kani_dpdf_postprocess_table_structure_proofs.rs"]
mod kani_dpdf_postprocess_table_structure_proofs;

// -- Kani proof harnesses for dpdf weight key mapping (#3901) ----------------
#[cfg(kani)]
#[path = "kani_convert_dpdf_proofs.rs"]
mod kani_convert_dpdf_proofs;

// -- Kani proof harnesses for FireRed-OCR, Qwen3-VL Quantized, preprocess (#3904) --
#[cfg(kani)]
#[path = "kani_dpdf_new_modules_proofs.rs"]
mod kani_dpdf_new_modules_proofs;

// -- Kani proof harnesses for FireRed-OCR config + dpdf pipeline forward (#3917) --
#[cfg(kani)]
#[path = "kani_dpdf_firered_forward_proofs.rs"]
mod kani_dpdf_firered_forward_proofs;

// -- Kani proof harnesses for dpdf_export module correctness (#3927) ----------
#[cfg(kani)]
#[path = "kani_dpdf_export_proofs.rs"]
mod kani_dpdf_export_proofs;

// -- Kani proof harnesses for dpdf_streaming + dpdf_benchmark safety (#3933) --
#[cfg(kani)]
#[path = "kani_dpdf_streaming_benchmark_proofs.rs"]
mod kani_dpdf_streaming_benchmark_proofs;

// -- Kani proof harnesses for dpdf_registry dispatch safety (#3939) -----------
#[cfg(kani)]
#[path = "kani_dpdf_registry_proofs.rs"]
mod kani_dpdf_registry_proofs;

// -- Kani proof harnesses for dpdf_export + dpdf_image_preprocess safety (#3946) --
#[cfg(kani)]
#[path = "kani_dpdf_export_preprocess_proofs.rs"]
mod kani_dpdf_export_preprocess_proofs;

// -- Kani deep proof harnesses for dpdf_streaming + dpdf_benchmark edge cases (#3954) --
#[cfg(kani)]
#[path = "kani_dpdf_streaming_deep_proofs.rs"]
mod kani_dpdf_streaming_deep_proofs;

// -- Kani proof harnesses for dpdf_registry dispatch + dpdf_postprocess NMS (#3958) ---
#[cfg(kani)]
#[path = "kani_dpdf_registry_postprocess_deep_proofs.rs"]
mod kani_dpdf_registry_postprocess_deep_proofs;

// -- Kani proof harnesses for dpdf_pipeline_forward DynTensor dispatch safety (#3963) --
#[cfg(kani)]
#[path = "kani_dpdf_pipeline_forward_proofs.rs"]
mod kani_dpdf_pipeline_forward_proofs;

// -- Kani proof harnesses for dpdf_pipeline end-to-end document processing invariants (#3970) --
#[cfg(kani)]
#[path = "kani_dpdf_pipeline_deep_proofs.rs"]
mod kani_dpdf_pipeline_deep_proofs;

// -- Kani proof harnesses for dpdf_export format correctness and round-trip safety (#3976) --
#[cfg(kani)]
#[path = "kani_dpdf_export_deep_proofs.rs"]
mod kani_dpdf_export_deep_proofs;

// -- Kani proof harnesses for dpdf_image_preprocess safety and numerical invariants (#3982) --
#[cfg(kani)]
#[path = "kani_dpdf_image_preprocess_deep_proofs.rs"]
mod kani_dpdf_image_preprocess_deep_proofs;

// -- Kani proof harnesses for dpdf_postprocess NMS, dedup, and fusion safety (#3988) --
#[cfg(kani)]
#[path = "kani_dpdf_postprocess_deep_proofs.rs"]
mod kani_dpdf_postprocess_deep_proofs;

// -- Kani proof harnesses for dpdf_streaming chunked processing safety (#3993) --
#[cfg(kani)]
#[path = "kani_dpdf_streaming_safety_proofs.rs"]
mod kani_dpdf_streaming_safety_proofs;

// -- Kani proof harnesses for dpdf_registry model dispatch routing + type safety (#3999) --
#[cfg(kani)]
#[path = "kani_dpdf_registry_safety_proofs.rs"]
mod kani_dpdf_registry_safety_proofs;

// -- Kani proof harnesses for dpdf_export format correctness (#4005) ----------
#[cfg(kani)]
#[path = "kani_dpdf_export_format_proofs.rs"]
mod kani_dpdf_export_format_proofs;

// -- Kani proof harnesses for dpdf image preprocessing dimension safety (#4017) --
#[cfg(kani)]
#[path = "kani_dpdf_image_preprocess_proofs.rs"]
mod kani_dpdf_image_preprocess_proofs;

// -- Kani proof harnesses for dpdf attention mask construction safety (#4023) --
#[cfg(kani)]
#[path = "kani_dpdf_attention_mask_proofs.rs"]
mod kani_dpdf_attention_mask_proofs;

// -- Kani proof harnesses for dpdf tokenizer safety (#4029) --
#[cfg(kani)]
#[path = "kani_dpdf_tokenizer_safety_proofs.rs"]
mod kani_dpdf_tokenizer_safety_proofs;

// -- Kani proof harnesses for dpdf image preprocessing safety (#4041) --
#[cfg(kani)]
#[path = "kani_dpdf_image_safety_proofs.rs"]
mod kani_dpdf_image_safety_proofs;

// -- Kani proof harnesses for NMS and detection postprocessing safety (#4037) --
#[cfg(kani)]
#[path = "kani_dpdf_nms_postprocess_proofs.rs"]
mod kani_dpdf_nms_postprocess_proofs;

// -- Kani proof harnesses for tensor shape and broadcast safety (#4044) --------
#[cfg(kani)]
#[path = "kani_dpdf_shape_broadcast_proofs.rs"]
mod kani_dpdf_shape_broadcast_proofs;

// -- Kani proof harnesses for quantization rounding and overflow safety (#4050) --
#[cfg(kani)]
#[path = "kani_dpdf_quantization_safety_proofs.rs"]
mod kani_dpdf_quantization_safety_proofs;

// -- Kani proof harnesses for output decoding safety (#4057) -------------------
#[cfg(kani)]
#[path = "kani_dpdf_output_decode_proofs.rs"]
mod kani_dpdf_output_decode_proofs;

// -- Kani proof harnesses for DocLayout-YOLO detection head invariants (#4145) --
#[cfg(kani)]
#[path = "kani_doclayout_yolo_proofs.rs"]
mod kani_doclayout_yolo_proofs;

// -- Kani proof harnesses for Granite-Docling ResNet18 backbone shape invariants (#4149) --
#[cfg(kani)]
#[path = "kani_granite_docling_proofs.rs"]
mod kani_granite_docling_proofs;
// NOTE: kani_paddle_ocr_svtr_proofs.rs removed -- SVTR architecture replaced by
// PaddleOCR-VL-1.5 vision-language model. Old proofs for DB+SVTR+CTC pipeline
// no longer apply to the new SigLIP ViT + ERNIE-4.5 GQA architecture.

// -- Kani proof harnesses for Wave 37 cross-module invariants (#4298) ------
#[cfg(kani)]
#[path = "kani_wave37_proofs.rs"]
mod kani_wave37_proofs;

// -- HTDemucs integration-level model tests (#4297) -------------------------
#[cfg(test)]
#[path = "demucs_htdemucs_model_tests.rs"]
mod demucs_htdemucs_model_tests;

// -- Comprehensive STFT/iSTFT signal processing tests (#3351) ---------------
#[cfg(test)]
#[path = "stft_istft_signal_tests.rs"]
mod stft_istft_signal_tests;

// -- Kokoro config pipeline integration tests (#4186) -------------------------
#[cfg(test)]
#[path = "kokoro_config_pipeline_tests.rs"]
mod kokoro_config_pipeline_tests;

// -- Signal processing tests: STFT, iSTFT, Hann window, DFT basis (#4186) ----
#[cfg(test)]
#[path = "signal_processing_tests.rs"]
mod signal_processing_tests;

// -- HTDemucs integration tests: enc/dec shapes, skip connections (#4186) -----
#[cfg(test)]
#[path = "htdemucs_integration_tests.rs"]
mod htdemucs_integration_tests;

// -- Silero VAD integration tests: frame size, LSTM, encoder chain (#4186) ----
#[cfg(test)]
#[path = "silero_vad_integration_tests.rs"]
mod silero_vad_integration_tests;

// -- STFT/iSTFT roundtrip and signal processing tests (#4186) -----------------
#[cfg(test)]
#[path = "stft_roundtrip_tests.rs"]
mod stft_roundtrip_tests;

// -- Kokoro tokenizer and phoneme vocabulary tests (#4186) --------------------
#[cfg(test)]
#[path = "kokoro_tokenizer_roundtrip_tests.rs"]
mod kokoro_tokenizer_roundtrip_tests;

// -- Extended signal processing tests: DFT basis, phase, harmonics (#4186) ----
#[cfg(test)]
#[path = "signal_processing_extended_tests.rs"]
mod signal_processing_extended_tests;

// -- Extended STFT/iSTFT signal processing tests (#4186) ----------------------
#[cfg(test)]
#[path = "stft_signal_extended_tests.rs"]
mod stft_signal_extended_tests;

// -- Extended Kokoro configuration tests (#4186) ------------------------------
#[cfg(test)]
#[path = "kokoro_config_extended_tests.rs"]
mod kokoro_config_extended_tests;

// -- Extended model builder and signal processing tests (#4186) ---------------
#[cfg(test)]
#[path = "models_extended_tests.rs"]
mod models_extended_tests;

// -- Extended STFT, dispatch, and model config tests (#4495) ------------------
#[cfg(test)]
#[path = "stft_dispatch_extended_tests.rs"]
mod stft_dispatch_extended_tests;

// -- Kokoro auto-converter parity tests (#4276) ------------------------------
#[cfg(test)]
#[path = "kokoro_convert_parity_tests.rs"]
mod kokoro_convert_parity_tests;

// -- Extended Kokoro auto-converter parity tests (#4276) ---------------------
#[cfg(test)]
#[path = "kokoro_convert_parity_tests_extended.rs"]
mod kokoro_convert_parity_tests_extended;

// -- Kokoro auto-converter parity scaffolding: dtype, config, architecture (#4276)
#[cfg(test)]
#[path = "kokoro_converter_parity_tests.rs"]
mod kokoro_converter_parity_tests;

// -- Streaming quality verification tests for Wave 22 Kokoro improvements (#4560)
#[cfg(test)]
#[path = "kokoro_streaming_quality_tests.rs"]
mod kokoro_streaming_quality_tests;

// -- Audio quality gate tests for Kokoro chorus processing modules (#4264)
#[cfg(test)]
#[path = "kokoro_chorus_quality_gates.rs"]
mod kokoro_chorus_quality_gates;

// -- Production integration tests for full Kokoro chorus pipeline (#4264)
#[cfg(test)]
#[path = "kokoro_chorus_production_tests.rs"]
mod kokoro_chorus_production_tests;

// -- End-to-end audio correctness tests for Kokoro chorus pipeline (#3351)
#[cfg(test)]
#[path = "kokoro_chorus_e2e_correctness_tests.rs"]
mod kokoro_chorus_e2e_correctness_tests;
