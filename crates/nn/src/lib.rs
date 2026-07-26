// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Top-level nn crate — verified Rust ML framework.
//!
//! # Kernel authors
//!
//! Re-exports the proc-macro attributes so users can write:
//!
//! ```text
//! use nn::Tensor;
//!
//! #[nn::kernel]
//! fn snake(x: f32, alpha: f32) -> f32 {
//!     x + (1.0 / alpha) * (alpha * x).sin().powi(2)
//! }
//! ```
//!
//! instead of `#[nn_macros::kernel]`.
//!
//! # Inference consumers
//!
//! Enable the `metal` feature to access the Metal GPU backend and pre-built
//! model inference types:
//!
//! ```toml
//! [dependencies]
//! nn = { git = "...", features = ["metal"] }
//! ```
//!
//! ```text
//! use nn::metal::{MetalBackend, PipelineCache, SileroVad, WeightMap};
//! ```

// Proc-macro re-exports: `#[nn::kernel]` and `#[nn::model]`
// Available when `dsl` feature is enabled (inference consumers don't need syn/proc-macro).
#[cfg(feature = "dsl")]
pub use nn_macros::{kernel, model};

// Core types
pub use nn_core::{Backend, CpuBackend, Tensor, TensorElement};

// Dynamic-rank tensor API for imperative model code (candle-compatible)
#[allow(deprecated)]
pub use nn_core::softmax_last_dim;
pub use nn_core::{
    conv1d_out_len, conv2d_out_len, conv_transpose1d_out_len, conv_transpose2d_out_len,
    load_safetensors, load_safetensors_from_bytes, register_gpu_backend, save_safetensors,
    tensors_to_safetensors_bytes, BinaryOp, CompareOp, Conv1dParams, Conv2dParams,
    ConvTranspose1dParams, ConvTranspose2dParams, Dim, DynTensor, GpuBackend, GpuFullBackend,
    GpuNnOps, GpuSelectionOps, GpuShapeOps, GridSamplePaddingMode, IndexOp, ReduceOp, Shape,
    TensorIndexer, UnaryOp, WithDType, D,
};

// Neural network layers (candle-nn compatible)
pub use nn_core::{
    alibi_bias, alibi_bias_scaled, alibi_slopes, apply_adaln_modulation, batch_norm, beam_search,
    causal_mask, causal_mask_dtype, causal_mask_with_offset, check_output_finite, conv1d,
    conv1d_no_bias, conv2d, conv2d_no_bias, conv3d, conv3d_no_bias, conv_transpose1d,
    conv_transpose1d_no_bias, conv_transpose2d, conv_transpose2d_no_bias, ctc_beam_decode,
    ctc_greedy_decode, embedding, generate, group_norm, layer_norm, linear, linear_no_bias,
    log_softmax, lstm, nan_check_policy, repeat_kv, rms_norm, rope, sdpa, sdpa_causal, sigmoid,
    sinusoidal_2d, softmax, window_partition, window_unpartition, with_nan_check_policy,
    Activation, AdaIn, AdaLnParams, AdaLnZero, AdaLnZeroDual, AdaptiveAvgPool2d,
    AttentiveStatisticsPooling, AvgPool2d, BatchNorm, BatchNorm2d, BatchNormConfig, BeamHypothesis,
    BeamSearchConfig, BeamSearchOutput, BiLstm, BlockQ4K, Conv1d, Conv1dConfig, Conv2d,
    Conv2dConfig, Conv3d, Conv3dConfig, ConvTranspose1d, ConvTranspose1dConfig, ConvTranspose2d,
    ConvTranspose2dConfig, CtcBeamHypothesis, CtcConfig, DeformableAttention,
    DeformableAttentionConfig, DiTBlock, DiTBlockDual, Dropout, Embedding, ExpertFFN,
    GatedDeltaNet, GatedDeltaNetState, GenerationConfig, GenerationOutput, GgmlDType, GroupNorm,
    HalfRotaryEmbedding, InstanceNorm, InstanceNormPrecision, InterleavedMRoPE,
    InterleavedMRoPEConfig, JointAttention, KvCache, KvCacheBackend, KvCacheLayer,
    KvCacheLayerBackend, LayerNorm, LayerNormConfig, Linear, LowRankAdaLn, Lstm, LstmCell,
    LstmState, MBConv, MBConvConfig, MaxPool1d, MaxPool2d, Module, ModuleT, MoeDispatch,
    MoeDispatchConfig, MoeDispatchOutput, MoeLayer, MoeLayerConfig, MoeOutput, MoeRouter,
    MoeRoutingOutput, MtpHead, MtpHeadConfig, MultiHeadAttention, MultimodalRoPE, NanCheckPolicy,
    PatchEmbedding, PixelShuffle, PixelUnshuffle, Pool1dConfig, Pool2dConfig, PoolingStrategy,
    PreallocKvCache, PreallocKvCacheLayer, QLinear, QuantizedWeight, Res2NetBlock, RmsNorm,
    RotaryEmbedding, RotaryEmbedding2d, Rvq, Sequential, SqueezeExcitation, SqueezeExcitation1d,
    SwiGlu, SwiGluExpert, Upsample2d, UpsampleMode, VitConfig, VitEncoder, VitEncoderBlock,
    VqCodebook, WeightNormConv1d, YarnScaling,
};
// INT8 quantization (W8A16 per-channel) -- Part of #3522
pub use nn_core::{
    dequantize_per_channel, max_quantization_error, quantize_per_channel, Int8Linear, Int8Mode,
    Int8QuantParams,
};

// ODE solvers for flow matching / rectified flow inference
pub use nn_core::{euler_solve, euler_solve_cfg, TimeSchedule, VelocityField};

/// Whisper STT model — speech recognition, transcription, and language detection.
///
/// Available when the `whisper` feature is enabled.
///
/// ```text
/// use nn::whisper::{WhisperModel, WhisperConfig, greedy_decode, pcm_to_mel};
/// ```
#[cfg(feature = "whisper")]
pub mod whisper {
    // Model
    pub use nn_whisper::WhisperModel;
    pub use nn_whisper::{AudioEncoder, TextDecoder};
    pub use nn_whisper::{WhisperConfig, WhisperError};

    // Decoding
    pub use nn_whisper::{
        beam_search_decode, compression_ratio, decode_with_temperature, detect_language,
        greedy_decode, passes_quality_check, temperature_fallback_decode, transcribe,
        transcribe_long, transcribe_with_fallback,
    };
    pub use nn_whisper::{
        DecodeConfig, DecodingResult, LanguageDetectionResult, LongFormConfig, LongFormResult,
        LongFormSegment, TranscriptionResult, WhisperBeamConfig,
    };
    pub use nn_whisper::{
        DEFAULT_AVG_LOGPROB_THRESHOLD, DEFAULT_COMPRESSION_RATIO_THRESHOLD, DEFAULT_TEMPERATURES,
        MAX_DECODE_LENGTH,
    };

    // Audio / mel
    pub use nn_whisper::{
        mel_filterbank, pcm_to_mel, whisper_mel_spectrogram, whisper_mel_spectrogram_for_config,
    };
    pub use nn_whisper::{
        CHUNK_LENGTH, HOP_LENGTH, NUM_MEL_BINS, N_FFT, N_FRAMES, N_SAMPLES, SAMPLE_RATE,
    };

    // Tokenizer
    pub use nn_whisper::{DecodedSegment, WhisperTokenizer};
    pub use nn_whisper::{
        DEFAULT_NO_SPEECH_THRESHOLD, EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START,
        NO_SPEECH_TOKEN, SOT_TOKEN,
    };

    // Quality
    pub use nn_whisper::word_error_rate;
}

// Qwen3 LLM model (backend-agnostic, uses DynTensor + Module API)
// Note: causal_mask and causal_mask_with_offset are now in nn-core::layers, not re-exported from qwen3.
#[cfg(feature = "qwen3")]
pub use nn_qwen3::{Qwen3Config, Qwen3Error, Qwen3Model, Qwen3MoeConfig, Qwen3MoeModel};

// GLM-4/5 (ChatGLM) LLM model (backend-agnostic, uses DynTensor + Module API)
#[cfg(feature = "glm5")]
pub use nn_glm5::{Glm5Config, Glm5Error, Glm5Model};

/// Backend-agnostic model definitions, builders, and signal processing.
///
/// Available when the `models` feature is enabled. Provides Kokoro TTS,
/// PlBert, ECAPA-TDNN, Demucs builder helpers, STFT/iSTFT, and model
/// weight types that any GPU backend can consume.
///
/// ```toml
/// [dependencies]
/// nn = { git = "...", features = ["models"] }
/// ```
///
/// ```text
/// use nn::models::{KokoroConfig, KokoroModel, PlBert, EcapaTdnn};
/// ```
#[cfg(feature = "models")]
pub mod models {
    // Explicit re-exports replace wildcard (#2288). Import from nn-models
    // directly for items not listed here.

    // Kokoro TTS
    #[cfg(feature = "espeak")]
    pub use nn_models::espeak_ffi::{EspeakEngine, EspeakError};
    pub use nn_models::kokoro_chorus::{
        self, mix_voices, mix_voices_from_refs, mix_voices_stereo, mix_voices_with_config,
        ChorusConfig, VoiceInput, VoiceMix,
    };
    pub use nn_models::kokoro_decoder;
    pub use nn_models::kokoro_error::KokoroError;
    pub use nn_models::kokoro_f0::{self, AdainResBlk1d, F0EnergyPredictor};
    pub use nn_models::kokoro_forward_stft::KokoroForwardStft;
    pub use nn_models::kokoro_full_decoder::{self, FullDecoder, Stage1ResBlk};
    pub use nn_models::kokoro_g2p::{self, EspeakRemapper, PhonemeLexicon, RemapTable};
    pub use nn_models::kokoro_pipeline::{
        self, chunks_to_tensors, KokoroSynth, KokoroTextPipeline, PipelineError,
    };
    pub use nn_models::kokoro_resblock;
    pub use nn_models::kokoro_source::{self, SineGen, SourceModule};
    pub use nn_models::kokoro_streaming::{
        self, assemble_streaming_chorus, assemble_streaming_chunks, concatenate_chunks,
        crossfade_chunks, crossfade_chunks_windowed, AudioChunk, CrossfadeWindow,
        KokoroStreamConfig, StreamingAssembler, HANN_CROSSFADE_THRESHOLD,
    };
    pub use nn_models::kokoro_text_preprocess::{self, TextPreprocessor};
    pub use nn_models::kokoro_tokenizer::{self, KokoroTokenizer, KokoroVocab};
    pub use nn_models::kokoro_tts::{self, KokoroConfig, KokoroModel, TextEncoder};
    pub use nn_models::kokoro_voice_pack::{self, VoicePack};
    pub use nn_models::plbert::{PlBert, PlbertConfig};

    // STFT/iSTFT
    pub use nn_models::istft::{self, IstftBasis, IstftError, IstftParams};
    pub use nn_models::stft::{self, compute_stft_magnitude, StftError, StftParams};

    // HTDemucs
    pub use nn_models::demucs_shared::{self, channels_at_depth, conv1d_output_len};
    pub use nn_models::{
        demucs_spectral_decoder_builders, demucs_spectral_encoder_builders,
        demucs_spectral_weights, demucs_temporal_decoder_builders,
        demucs_temporal_encoder_builders, demucs_temporal_weights, demucs_transformer_builders,
        demucs_transformer_constants, demucs_transformer_helpers, demucs_transformer_validate,
        demucs_transformer_weights,
    };
    pub use nn_models::{DemucsBuilderError, TransformerBuildError};

    // Speaker verification
    pub use nn_models::EcapaTdnn;
    pub use nn_models::SERes2Block;
    pub use nn_models::WeSpeakerResNet34;

    // Silero VAD
    pub use nn_models::silero_vad_builders;
}

/// Neural network layers — Linear, Conv1d, LayerNorm, etc.
///
/// Re-exports `nn_core::layers` for consumers who prefer `nn::layers::Linear` style.
pub mod layers {
    pub use nn_core::layers::*;
}

// Hierarchical weight loading (candle-nn VarBuilder compatible)
pub use nn_core::VarBuilder;

// Types commonly needed by consumers but previously required direct nn-core import
pub use nn_core::{check_dim, Result, TensorError};
pub use nn_core::{BackendDomain, BackendErrorKind, DType, Device, IntervalBounds};

/// Computation graph tracing — captures DynTensor operations for verification.
///
/// Re-exports `nn_core::dyn_tensor::trace` so downstream consumers (dvoice)
/// can import `nn::trace::{TraceOp, KokoroFusedOp, traced_forward}` instead of
/// depending on `nn-core` directly.
pub mod trace {
    pub use nn_core::dyn_tensor::trace::*;
}

// ULP rounding utilities for sound bounds computation
pub use nn_core::{next_down_f32, next_up_f32};

// Mixed-precision policy for graph-level dtype management
pub use nn_core::{default_op_category, MixedPrecisionPolicy, OpDTypeCategory};

// VarBuilder backend traits for custom weight loaders
pub use nn_core::var_builder::{TensorBackend, TensorMapBackend, ZerosBackend};

// Audio DSP utilities (mel filterbank, Hann window, Hz↔Mel conversion)
pub use nn_core::audio;

// Third-party re-exports for tensor interop
pub use nn_core::half;

// DSL types commonly needed alongside `#[kernel]`
#[cfg(feature = "dsl")]
pub use nn_dsl::{KernelDescriptor, PrecisionContract, PrecisionTier};

// ---------------------------------------------------------------------------
// DSL — kernel IR, reference implementations, and codegen (feature-gated)
// ---------------------------------------------------------------------------

/// Kernel DSL — reference implementations, IR builders, and codegen.
///
/// Available when the `dsl` feature is enabled. Provides reference scalar
/// implementations, IR builders, and MSL/PTX codegen.
///
/// ```toml
/// [dependencies]
/// nn = { git = "...", features = ["dsl"] }
/// ```
#[cfg(feature = "dsl")]
pub mod dsl {
    // Explicit re-exports replace wildcard (#2288). Import from nn-dsl
    // directly for items not listed here.

    // Core types
    pub use nn_dsl::{KernelDescriptor, PrecisionContract, PrecisionTier};

    // IR and builders
    pub use nn_dsl::input_names;
    pub use nn_dsl::ir;
    pub use nn_dsl::model_ir;
    pub use nn_dsl::tensor_block_builder::{self, TensorBlockBuilder};
    pub use nn_dsl::tensor_ir;

    // Kernel modules (reference implementations)
    pub use nn_dsl::{
        ada_layer_norm, adain, biquad, causal_conv1d, conv1d, conv2d, conv_transpose_1d, ducker,
        gated_delta_net, gelu, instance_norm, layer_norm, linear, lstm_decomposed, relu, reverb,
        rms_norm, rope, sigmoid, silu_mul, snake, softmax, tanh_kernel, waveshaper, weight_norm,
    };

    // Lowering and codegen
    pub use nn_dsl::kernel_error;
    pub use nn_dsl::lower;
    pub use nn_dsl::precision;

    // Compiled model support
    pub use nn_dsl::buffer_planner;
    pub use nn_dsl::norm_activ_conv_kernels;
    pub use nn_dsl::performance_report;
    pub use nn_dsl::trace_compile;
    pub use nn_dsl::verifiability;

    // Standalone functions
    pub use nn_dsl::sum_reduce;
}

// ---------------------------------------------------------------------------
// Training — autodiff, optimizers, LoRA (feature-gated)
// ---------------------------------------------------------------------------

/// Automatic differentiation and training utilities.
///
/// Available when the `training` feature is enabled. Provides:
/// - `Var` / `TrackedTensor` / `backward()` for reverse-mode AD
/// - `AdamW` / `Sgd` optimizers with LR scheduling
/// - `LoraLinear` for parameter-efficient fine-tuning
///
/// ```toml
/// [dependencies]
/// nn = { git = "...", features = ["training"] }
/// ```
///
/// ```text
/// use nn::training::{Var, TrackedTensor, backward, AdamW, LoraLinear};
/// ```
#[cfg(feature = "training")]
#[path = "training_exports.rs"]
pub mod training;

// ---------------------------------------------------------------------------
// Metal backend (feature-gated)
// ---------------------------------------------------------------------------

/// Metal GPU backend — model inference, weight loading, and kernel dispatch.
///
/// Available when the `metal` feature is enabled. Provides everything needed
/// to run pre-built models (Silero VAD, HTDemucs) on Apple Silicon GPUs.
#[cfg(feature = "metal")]
#[path = "metal_exports.rs"]
pub mod metal;

// ---------------------------------------------------------------------------
// Root-level convenience re-exports for dvoice integration (feature-gated)
// ---------------------------------------------------------------------------

/// Multi-voice Kokoro TTS chorus pool — shares GPU weights across N voice
/// instances for efficient multi-voice synthesis.
///
/// This is a convenience re-export of [`metal::KokoroChorus`]. Available when
/// the `metal` feature is enabled.
///
/// ```text
/// use nn::KokoroChorus;
/// // equivalent to: use nn::metal::KokoroChorus;
/// ```
#[cfg(feature = "metal")]
pub use nn_metal::KokoroChorus;

/// Caller-held fence for submitted GPU work — enables CPU/GPU pipelining by
/// letting callers hold multiple outstanding GPU submissions and wait on them
/// individually.
///
/// This is a convenience re-export of [`metal::GpuFence`]. Available when
/// the `metal` feature is enabled.
///
/// ```text
/// use nn::GpuFence;
/// // equivalent to: use nn::metal::GpuFence;
/// ```
#[cfg(feature = "metal")]
pub use nn_metal::GpuFence;

/// GPU-resident audio handle — defers GPU-to-CPU transfer so the caller
/// controls when the readback happens. Returned by
/// [`CompiledKokoro::synthesize_gpu()`](metal::CompiledKokoro).
///
/// This is a convenience re-export of [`metal::GpuAudioHandle`]. Available when
/// the `metal` feature is enabled.
///
/// ```text
/// use nn::GpuAudioHandle;
/// // equivalent to: use nn::metal::GpuAudioHandle;
/// ```
#[cfg(feature = "metal")]
pub use nn_metal::GpuAudioHandle;

/// Single-voice pull-based streaming synthesis session — yields one audio
/// chunk per `next_chunk()` call with automatic crossfade between chunks.
///
/// This is a convenience re-export of [`metal::StreamingKokoroSession`].
/// Available when the `metal` feature is enabled.
///
/// ```text
/// use nn::StreamingKokoroSession;
/// // equivalent to: use nn::metal::StreamingKokoroSession;
/// ```
#[cfg(feature = "metal")]
pub use nn_metal::StreamingKokoroSession;

/// Multi-voice pull-based streaming chorus session — shared or per-voice
/// text synthesis with crossfade and voice mixing.
///
/// This is a convenience re-export of [`metal::StreamingChorusSession`].
/// Available when the `metal` feature is enabled.
///
/// ```text
/// use nn::StreamingChorusSession;
/// // equivalent to: use nn::metal::StreamingChorusSession;
/// ```
#[cfg(feature = "metal")]
pub use nn_metal::StreamingChorusSession;

// ---------------------------------------------------------------------------
// Reftest — reference tensor comparison (feature-gated)
// ---------------------------------------------------------------------------

/// Reference tensor comparison — load, compare, and assert parity between
/// Python and Rust model implementations.
///
/// Available when the `reftest` feature is enabled.
#[cfg(feature = "reftest")]
pub mod reftest {
    // Types
    pub use nn_reftest::{
        ComparisonConfig, DivergenceReport, LayerComparison, NamedTensor, ReferenceTrace,
        ReftestError,
    };
    // Functions
    pub use nn_reftest::{
        compare_tensors, compare_traces, load_npy, load_npy_dir, load_npy_from_bytes,
        load_safetensors, load_safetensors_from_bytes,
    };
    // Macros
    pub use nn_reftest::assert_traces_match;

    // Spectral comparison (feature-gated in nn-reftest)
    #[cfg(feature = "spectral")]
    pub use nn_reftest::assert_spectral_match;
    #[cfg(feature = "spectral")]
    pub use nn_reftest::{compare_spectral, SpectralComparison, SpectralConfig};
}

// ---------------------------------------------------------------------------
// TTS verification — audio quality checks and proof certificates (feature-gated)
// ---------------------------------------------------------------------------

/// TTS output verification — hard bounds, quality metrics, and CROWN bridge.
///
/// Available when the `tts-verify` feature is enabled. Provides:
/// - `TtsVerifier` builder for configuring audio quality checks
/// - `Certificate` with hard bounds and quality metrics
/// - DSP utilities (RMS, FFT, mel filterbank, F0 extraction)
///
/// ```toml
/// [dependencies]
/// nn = { git = "...", features = ["tts-verify"] }
/// ```
///
/// ```text
/// use nn::tts_verify::{TtsVerifier, Certificate};
/// ```
#[cfg(feature = "tts-verify")]
pub mod tts_verify {
    // Explicit re-exports replace wildcard (#2288). Import from nn-tts-verify
    // directly for items not listed here.

    // Core API (documented in module doc)
    pub use nn_tts_verify::Certificate;
    pub use nn_tts_verify::{HardBound, SpectralCoverageConfig};
    pub use nn_tts_verify::{QualityMetric, TtsVerifyError};
    pub use nn_tts_verify::{TtsVerifier, TtsVerifierBuilder};

    // Configuration
    pub use nn_tts_verify::{CheckOverrides, HardBoundsConfig, QualityConfig, RejectionPolicy};

    // DSP utilities
    pub use nn_tts_verify::{compute_pesq, compute_stoi};
    pub use nn_tts_verify::{dsp, f0_contour, multi_res_stft, stats};

    // Kokoro dispatch
    pub use nn_tts_verify::{kokoro_contracts, kokoro_dispatch, kokoro_encoder_dispatch};

    // Verification sub-modules
    pub use nn_tts_verify::{
        adversarial, bounds, cost_model, cost_propagation, deterministic, error, monotonicity,
        moonshot, pipeline, quality_bound, streaming, unicode_safety,
    };

    // NY gated sub-modules (requires `tts-verify-crown` feature)
    #[cfg(feature = "tts-verify-crown")]
    pub use nn_tts_verify::crown;
}

// ---------------------------------------------------------------------------
// Verification — NY bounds, proof certificates (feature-gated)
// ---------------------------------------------------------------------------

/// Formal verification — IBP/CROWN bound propagation, proof certificates,
/// and verified model wrappers.
///
/// Available when the `verify` feature is enabled. Provides:
/// - `verify_trace` for quick IBP verification of traced graphs
/// - `VerifiedModel<M>` wrapper pairing a model with its certificate
/// - `ProofCertificate` / `CertificateBundle` for deployment auditing
/// - `verify_model!` macro for declarative verification test generation
///
/// ```toml
/// [dependencies]
/// nn = { git = "...", features = ["verify"] }
/// ```
///
/// ```text
/// use nn::verify::{verify_trace, VerifiedModel, ProofCertificate};
/// ```
#[cfg(feature = "verify")]
#[path = "verify_exports.rs"]
pub mod verify;

// Re-export verify_model! macro at crate root so `nn::verify_model!` works.
// $crate in the macro body resolves to nn_verify (the defining crate),
// so __macro_internals and other internal references remain correct.
#[cfg(feature = "verify")]
pub use nn_verify::verify_model;

// ---------------------------------------------------------------------------
// Import — torch.export graph import and model conversion (feature-gated)
// ---------------------------------------------------------------------------

/// PyTorch export import — parse `torch.export` graphs, map aten ops, and
/// build intermediate graph data for downstream conversion and compilation
/// surfaces.
///
/// Available when the `import` feature is enabled.
///
/// ```toml
/// [dependencies]
/// nn = { git = "...", features = ["import"] }
/// ```
///
/// ```text
/// use nn::import::{ExportedProgram, build_graph, map_node_to_trace_op};
/// ```
#[cfg(feature = "import")]
#[path = "import_exports.rs"]
pub mod import;

/// Root conversion-report provenance and current composition-bounds
/// classification surface for exported artifacts.
///
/// Available when the `import` feature is enabled. `ConvertReport` exposes
/// `provenance_summary()` and `artifact_readiness_note()` so downstream callers
/// can summarize which exported-artifact intake path produced a report and what
/// artifact kind it covers. When the current composition-bounds verifier path
/// runs, `VerificationCoverage` may also record
/// `ConvertCompositionMethod`, `ConvertSoundnessMode`, and
/// `ConvertProofStrength` for that run. This is not a raw PyTorch or ONNX
/// intake API.
#[cfg(feature = "import")]
pub use nn_import::{
    ConvertArtifactKind, ConvertCompositionMethod, ConvertIntakePath, ConvertProofStrength,
    ConvertReport, ConvertSoundnessMode, VerificationCoverage,
};

/// Root multi-segment exported-artifact import surface.
///
/// Available when the `import` feature is enabled. Imports multiple named
/// `torch.export` JSON values that share one `safetensors` file into a
/// [`MultiSegmentModel`]. This is an already-exported-artifact bridge for
/// segmented bundles such as Kokoro-style pipelines; it is not raw
/// PyTorch/ONNX intake or runtime orchestration across segments.
#[cfg(feature = "import")]
#[doc(inline)]
pub use nn_import::{
    convert_multi_segment, convert_single_segment, MultiSegmentError, MultiSegmentModel,
};

// Root builder surface for already-exported `torch.export` + `safetensors`
// artifacts. `nn::convert()` returns a ConvertBuilder for fluent
// configuration; `VerifyLevel::Full` requests the fullest currently available
// report path, but does not run Kani inline or turn this into a complete
// proof-powered compiler.
//
// ```rust,ignore
// let result = nn::convert(&graph_json, &weights, &cache)
//     .reference_trace(&ref_path)
//     .optimize(OptLevel::Aggressive)
//     .verify(VerifyLevel::Bounds)
//     .build()?;
// let provenance = result.report.provenance_summary();
// let readiness = result.report.artifact_readiness_note();
// result.report.print();
// ```
#[cfg(feature = "import-metal")]
pub use nn_import::{
    convert_build as convert, ConvertBuilder, ConvertResultWithReport, OptLevel, VerifyLevel,
};

/// Root multi-segment exported-artifact Metal compile surface.
///
/// Available when the `import-metal` feature is enabled. Compiles an imported
/// [`MultiSegmentModel`] or directly supplied named exported-artifact JSON
/// segments to Metal while preserving segment order and current shared-weight
/// aliasing metadata. This remains an exported-artifact-only bridge; it does
/// not claim raw PyTorch/ONNX intake or a runtime scheduler for cross-segment
/// control flow.
#[cfg(feature = "import-metal")]
#[doc(inline)]
pub use nn_import::{
    compile_multi_segment, convert_multi_segment_to_metal, CompiledMultiSegmentModel,
    MultiSegmentCompileError,
};

// ---------------------------------------------------------------------------
// Exported-artifact conversion: nn-import graph + nn-models ConvertedModel
// ---------------------------------------------------------------------------

/// Exported-artifact conversion bridge between `nn-import` and `nn-models`.
///
/// Available when the `convert-model` feature is enabled (implies `import` +
/// `models`). See [`convert_from_trace()`] for the entry point.
#[cfg(feature = "convert-model")]
#[path = "convert_model.rs"]
pub mod convert_model;

/// Root `torch.export` + `safetensors` conversion surface.
///
/// Available when the `convert-model` feature is enabled. Imports exported
/// artifacts into a backend-agnostic [`ConvertedModel`]. Backend compilation
/// remains a separate surface. This is also the only current one-function
/// exported-artifact surface that accepts optional `ConvertConfig::model_type`
/// remapping on the returned weight map.
#[cfg(feature = "convert-model")]
#[doc(inline)]
pub use convert_model::{convert_from_trace, ConvertConfig, ConvertError, ConvertedModel};

/// Exported-artifact Metal bridge.
///
/// Available when both `metal` and `convert-model` are enabled. This helper
/// compiles already-exported `torch.export` JSON + `safetensors` artifacts to
/// Metal, with optional detailed report generation, and retains converted-model
/// metadata. The report-returning helper exposes `ConvertReport` coverage data,
/// including the current composition-bounds classification when that verifier
/// path runs, but not the underlying `EquivalenceProof`, and it is
/// intentionally not a raw PyTorch or ONNX intake API. Unlike
/// [`convert_from_trace()`], these helpers reject `ConvertConfig::model_type`
/// rather than silently compiling without end-to-end remapped weight-key
/// support.
#[cfg(all(feature = "metal", feature = "convert-model"))]
#[path = "convert_compile.rs"]
pub mod convert_compile;

/// Exported-artifact Metal bridge helper and result types.
///
/// Available when both `metal` and `convert-model` are enabled.
/// [`compile_exported_artifacts()`] is the preferred one-call façade for
/// exported `torch.export` JSON + `safetensors` intake, Metal compilation, and
/// `ConvertReport` generation. The longer helper names remain available for
/// callers that want to opt into the metadata-only or explicitly named report
/// path. This is still an exported-artifact-only API, not raw PyTorch or ONNX
/// ingestion.
#[cfg(all(feature = "metal", feature = "convert-model"))]
#[doc(inline)]
pub use convert_compile::{
    compile_exported_artifacts, compile_metal_from_exported_artifacts,
    compile_metal_from_exported_artifacts_with_report, ConvertedModelMetadata,
    ExportedArtifactCompileError, ExportedArtifactMetalModel, ExportedArtifactMetalModelWithReport,
};

// ---------------------------------------------------------------------------
// GGUF — llama.cpp model format parser (feature-gated)
// ---------------------------------------------------------------------------

/// GGUF model format parser — load llama.cpp ecosystem models.
///
/// Available when the `gguf` feature is enabled. Provides:
/// - `GgufFile::read_from()` — parse GGUF headers, metadata, tensor info
/// - `GgufFile::to_computation_graph()` — build computation graph from architecture
/// - `GgufFile::read_tensor_f32()` — dequantize Q4_0/Q8_0/F16/F32 tensors
/// - `LlamaConfig` / `build_llama_graph()` — Llama architecture builder
///
/// ```toml
/// [dependencies]
/// nn = { git = "...", features = ["gguf"] }
/// ```
///
/// ```text
/// use nn::gguf::{GgufFile, LlamaConfig};
/// let file = GgufFile::read_from(&mut reader)?;
/// let graph = file.to_computation_graph()?;
/// ```
#[cfg(feature = "gguf")]
pub mod gguf {
    pub use nn_gguf::{
        build_llama_graph, build_llama_graph_with_weights, GgufDType, GgufError, GgufFile,
        GgufHeader, GgufMetadata, GgufMetadataValue, GgufTensorInfo, LlamaConfig,
    };
}
