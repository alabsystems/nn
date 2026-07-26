// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pre-compiled Kokoro TTS pipeline for GPU-accelerated inference.
//!
//! [`CompiledKokoro`] wraps the full Kokoro-82M pipeline as 5 GPU-compiled
//! segments with GPU-native bridges. Each segment is a [`CompiledModel`]
//! with fused dispatch plans and pre-uploaded GPU weights.
//!
//! ```text
//! Segment 0 [compiled GPU]: PlBert+bert_encoder(input_ids, pos_emb, type_emb)
//!                           → bert_features [B, d_en, T]  (#2744)
//! Segment 1 [compiled GPU]: TextEncoder(input_ids) → text_features [B, d_en, T]
//! Segment 2 [compiled GPU]: ProsodyPredictor(bert_features, style) → (dur_logits, features)
//! GPU bridge: length_regulate ×2 (sigmoid+sum on GPU, GPU repeat_interleave)
//!   → aligned_dur [B, d_en+style_dim, T_mel]  (prosody features → F0EnergyPredictor)
//!   → regulated   [B, d_en, T_mel] (text features → FullDecoder)
//! Segment 3 [compiled GPU]: F0EnergyPredictor(aligned_dur, style) → (f0, energy)
//! GPU bridge: harmonic_source (fully GPU-native: Kahan cumsum #2909) + expand + cat + pad
//! Segment 4 [compiled GPU]: Generator(regulated, f0, energy, style, har) → (mag, phase)
//! GPU iSTFT: magnitude + phase → PCM audio
//! ```
//!
//! Segments 0-2 are compiled per input sequence length. Segment 3 is
//! compiled per T_mel. Segment 4 is compiled per total_samples (derived
//! from T_mel). Each segment has an LRU cache
//! (default capacity 4) so frequently-seen shapes reuse compiled models
//! instead of recompiling on every new text length (#2626).
//! The harmonic_source and length_regulate stages run on GPU (#2487, #2493)
//! — only small counts vectors are read to CPU for prefix-sum computation.
//!
//! Part of #2465, #2218.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_core::{DType, VarBuilder};

use nn_models::kokoro_tts::{KokoroConfig, KokoroModel};

use nn_tts_verify::{Certificate, HardBoundsConfig};

use crate::cache::PipelineCache;
use crate::compiled_model::ShapePolicy;
use crate::MetalVarBuilderExt;

/// Controls the GPU dispatch strategy for single-voice synthesis.
///
/// The two-phase pipeline (default) splits the forward pass at the regulate
/// sync point (step 4) and uses [`GpuFence`](crate::gpu_fence::GpuFence)
/// submissions between phases to overlap CPU dispatch encoding with GPU
/// execution. The sequential path dispatches all steps in a single lazy
/// batch, which is simpler to reason about for debugging.
///
/// Part of #4264.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PipelineMode {
    /// Two-phase CPU-GPU pipelining (default).
    ///
    /// Phase 1 (encode + prosody + regulate) contains the only hard GPU sync
    /// point. After regulate returns, Phase 1 GPU work is submitted via
    /// `GpuFence` non-blocking. Phase 2 (f0 + harmonic + generator + iSTFT)
    /// is entirely sync-free GPU work encoded while Phase 1 executes on GPU.
    ///
    /// Production path: uses `_production` step variants that skip
    /// `to_standalone()` blits for ~5 fewer GPU dispatches.
    #[default]
    TwoPhase,

    /// Sequential dispatch without explicit pipelining.
    ///
    /// All steps are dispatched in order with `GpuFence` submissions at step
    /// boundaries for CPU-GPU overlap, but without the explicit Phase 1/Phase 2
    /// structural split. Useful for debugging pipeline issues — the execution
    /// order matches the logical step sequence exactly.
    Sequential,
}

#[path = "compiled_kokoro_error.rs"]
mod error;
pub use error::CompiledKokoroError;

#[path = "compiled_kokoro_autocast_config.rs"]
mod autocast_config;
pub use autocast_config::F16AutocastConfig;

#[path = "compiled_kokoro_auto_precision.rs"]
mod auto_precision;
pub use auto_precision::{
    auto_precision_config, format_precision_report, AutoPrecisionResult, SegmentPrecisionDecision,
};

#[path = "compiled_kokoro_segment_cache.rs"]
mod segment_cache;
use segment_cache::SegmentCache;
use crate::segment_cache::SegmentCacheConfig;

#[path = "compiled_kokoro_shared.rs"]
mod shared;
use shared::SharedKokoroState;

/// Pre-compiled Kokoro TTS pipeline.
///
/// Holds the model for tracing reference and caches compiled GPU segments.
/// Segments are compiled lazily on first use and cached by input shape.
/// Subsequent calls with the same shapes reuse cached compiled segments.
///
/// # Multi-voice sharing (#2740)
///
/// Model weights, verifier, and iSTFT basis are held in
/// `Arc<SharedKokoroState>`. Use [`clone_dispatch()`](Self::clone_dispatch)
/// to create lightweight instances for multi-voice synthesis pools.
/// 7 voices share ~400MB weights (1.02x overhead vs 7x without sharing).
///
/// # Example
///
/// ```rust,no_run
/// use nn_metal::compiled_kokoro::CompiledKokoro;
///
/// let mut kokoro = CompiledKokoro::new(model)?;
/// let (audio, cert) = kokoro.synthesize(&input_ids, &style, 1.0, &cache)?;
/// assert!(cert.overall_passed);
///
/// // Multi-voice: share weights, separate dispatch state.
/// let mut voice2 = kokoro.clone_dispatch();
/// ```
pub struct CompiledKokoro {
    /// Shared state: model weights, verifier, iSTFT basis.
    /// Arc-wrapped for multi-voice sharing (#2740).
    pub(super) shared: Arc<SharedKokoroState>,
    /// Cached segment 0: PlBert + bert_encoder (token IDs → bert_features). Key: seq_len.
    /// Eliminates ~187 eager GPU dispatch encodings (35% of pipeline total). (#2744)
    pub(super) seg_plbert: SegmentCache,
    /// Cached segment 1: TextEncoder (token IDs → features). Key: seq_len.
    /// LRU cache keeps last N compiled models to avoid recompilation on
    /// every new text length (#2626).
    pub(super) seg_text: SegmentCache,
    /// Cached segment 2: ProsodyPredictor. Key: seq_len.
    pub(super) seg_prosody: SegmentCache,
    /// Cached segment 3: F0EnergyPredictor. Key: t_mel.
    pub(super) seg_f0: SegmentCache,
    /// Cached segment 4: Generator. Key: total_samples.
    pub(super) seg_generator: SegmentCache,
    /// Cached segment 5: Regulate (elementwise chain: sigmoid→sum→mul_speed→clamp→
    /// squeeze→add→floor→clamp_min). Key: seq_len.
    /// No model weights — pure elementwise ops. Part of #1815 Tier 6 D2b.
    pub(super) seg_regulate: SegmentCache,
    /// Cached segment 5a: SineGen pre-cumsum (f0 → rad_frames + voiced).
    /// Key: t_frames. Multi-output. Part of #1815 Tier 6 D2.
    pub(super) seg_sinegen_pre: SegmentCache,
    /// Cached segment 5b: SineGen post-cumsum (cum_gpu + voiced → excitation).
    /// Key: t_frames. Single-output. Part of #1815 Tier 6 D3.
    pub(super) seg_sinegen_post: SegmentCache,
    /// Cached PlBert position/type embeddings per seq_len (#2912).
    /// Both embeddings are deterministic given seq_len, so we cache the
    /// GPU-resident tensors to eliminate 2 CPU→GPU transfers per call.
    pub(super) plbert_emb_cache: HashMap<usize, (DynTensor, DynTensor)>,
    /// When true, compile segments with F16 mixed-precision.
    /// Dispatch steps use F16 for 2x Metal ALU throughput; NativeOps stay F32.
    pub(super) mixed_precision: bool,
    /// Per-op autocast policy. When `Some`, segments use `builder().autocast()`
    /// — Compute/passthrough steps use F16, Accumulate ops stay F32. Part of #3085.
    pub(super) autocast_policy: Option<MixedPrecisionPolicy>,
    /// When true, auto-release CPU model weights after first successful
    /// synthesis. Saves ~320 MB RSS for Kokoro-82M. New input shapes
    /// cannot be compiled after release — call `precompile_shapes()` first
    /// if multiple shapes are expected. Part of #3079.
    pub(super) auto_release: bool,
    /// Per-shape segment cache configuration. Controls capacity and eviction
    /// policy for all 8 segment caches. Part of #3634.
    pub(super) segment_cache_config: SegmentCacheConfig,
    /// Per-segment PeepholeConfig overrides for selective fusion pass control.
    /// Keys are segment names: "plbert", "text", "prosody", "f0",
    /// "generator", "regulate", "sinegen_pre", "sinegen_post".
    /// Segments without a config use the default (all passes enabled).
    /// Part of #3828 Phase 2B.
    pub(super) peephole_configs: HashMap<String, nn_dsl::PeepholeConfig>,
    /// Per-segment autocast configuration. When `Some`, overrides the uniform
    /// `autocast_policy` with per-segment F16 enable/disable control.
    /// Part of #4269.
    pub(super) segment_autocast: Option<F16AutocastConfig>,
    /// When true, generate CROWN verification evidence after synthesis.
    ///
    /// Post-synthesis, the runtime Certificate's hard-bound results are mapped
    /// to moonshot property evidence (P1, P2, P6) and attached as a
    /// [`MoonshotCertificate`](nn_tts_verify::MoonshotCertificate) on the
    /// returned Certificate.
    ///
    /// Part of #4254, #3874.
    pub(super) crown_verification: bool,
    /// Configuration for CROWN certificate generation.
    ///
    /// Controls model name, input specification, and which hard bounds are
    /// mapped to moonshot properties. Only used when `crown_verification`
    /// is true.
    ///
    /// Part of #4254, #3874.
    pub(super) crown_config: nn_tts_verify::CrownCertificateConfig,
    /// Cached `(seq_len, speed_bits) -> total_repeats` from prior
    /// `step_regulate` calls. When a cache hit occurs, the `submit()+sync()`
    /// GPU pipeline stall for the 4-byte prefix-sum readback is eliminated.
    ///
    /// **Correctness:** Given identical `dur_logits` (deterministic from
    /// compiled segment + seq_len) and identical speed, `total_repeats` is
    /// deterministic. The GPU prefix-sum dispatch still runs (compiled into
    /// the segment); only the CPU readback is skipped.
    ///
    /// Speed is stored as `f32::to_bits()` for exact float key matching —
    /// NaN and +-0 have distinct bit patterns, which is correct here because
    /// we validate speed earlier (positive, finite).
    ///
    /// Part of #4264.
    pub(super) regulate_total_cache: HashMap<(usize, u32), usize>,
    /// Pipeline-level ICB replay buffer for Kokoro dispatch optimization.
    ///
    /// Caches pre-encoded Metal Indirect Command Buffers by input shape,
    /// split into pre-readback (steps 1-4) and post-readback (steps 5-8)
    /// phases around the `step_regulate` scalar readback sync point.
    ///
    /// Only active when the `icb-replay` feature is enabled AND
    /// `with_icb_replay()` has been called. Otherwise, the buffer exists
    /// but is disabled (all record/replay calls are no-ops).
    ///
    /// Part of #4264.
    pub(super) icb_replay: crate::compiled_model::icb::replay::IcbReplayBuffer,
    /// Terminal cumulative phase from the previous streaming chunk (GPU path).
    ///
    /// Shape: `[1, 1, n_channels]` (one value per harmonic). When `Some`,
    /// `build_harmonic_source()` adds this offset to the Kahan cumulative sum
    /// before scaling by `2*pi*upp`, ensuring SineGen phase continuity
    /// across streaming chunk boundaries. Reset to `None` at the start of
    /// each new streaming session via `reset_sinegen_phase()`.
    pub(super) sinegen_last_cumphase: Option<DynTensor>,
    /// Shape policy applied to all compiled segments.
    ///
    /// When `Polymorphic`, compiled segments accept variable sequence lengths
    /// without recompilation. Buffers are pre-allocated at max dimensions and
    /// threadgroup grids are computed from actual input sizes at dispatch time.
    ///
    /// Part of #3873.
    pub(super) shape_policy: ShapePolicy,
    /// GPU dispatch strategy for single-voice synthesis.
    ///
    /// Default: [`PipelineMode::TwoPhase`] for maximum CPU-GPU overlap.
    /// Set to [`PipelineMode::Sequential`] for debugging.
    ///
    /// Part of #4264.
    pub(super) pipeline_mode: PipelineMode,
    /// Per-segment optimization results from the last
    /// [`warmup_with_optimizer()`](Self::warmup_with_optimizer) call.
    ///
    /// Populated when `warmup_with_optimizer()` runs (either from cache or
    /// fresh optimizer search). Use [`optimization_summary()`](Self::optimization_summary)
    /// for a human-readable report.
    ///
    /// Part of #3828.
    #[cfg(feature = "plan-serde")]
    pub(super) optimization_results: Option<Vec<(String, nn_dsl::OptimizationResult)>>,
}

impl CompiledKokoro {
    /// Load Kokoro from a safetensors file with default configuration.
    ///
    /// Uses mmap for zero-copy weight loading. VarBuilder paths match
    /// PyTorch `state_dict()` key names (the standard convention).
    ///
    /// # Safety
    ///
    /// The safetensors file must not be modified or truncated while the
    /// returned `CompiledKokoro` is alive (standard mmap contract).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut kokoro = unsafe { CompiledKokoro::load("kokoro.safetensors")? };
    /// let (audio, cert) = kokoro.synthesize(&input_ids, &style, 1.0, &cache)?;
    /// assert!(cert.overall_passed);
    /// ```
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self, CompiledKokoroError> {
        // SAFETY: Caller guarantees safetensors file not modified while alive.
        unsafe { Self::load_with_config(path, &KokoroConfig::default()) }
    }

    /// Load Kokoro from a safetensors file with a custom configuration.
    ///
    /// # Safety
    ///
    /// Same as [`Self::load`]: safetensors file must not be modified while alive.
    pub unsafe fn load_with_config(
        path: impl AsRef<Path>,
        config: &KokoroConfig,
    ) -> Result<Self, CompiledKokoroError> {
        let path = path.as_ref();
        // Load weights to CPU — not GPU (#3079 RSS optimization).
        // Model weights are only used for tracing (which extracts f32 data)
        // and SourceModule bridge (moved to GPU by ensure_source_device).
        // Compiled segments upload their own GPU weight buffers.
        // This eliminates ~492 MB of duplicate GPU-resident model weights
        // (328 MB base + 164 MB Linear pre-transpose).
        let device = cpu();

        // SAFETY: Caller guarantees safetensors file not modified while alive.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device) }
            .map_err(|e| CompiledKokoroError::WeightLoadFailed {
                source: Box::new(e.into()),
            })?;

        let model =
            KokoroModel::load(&vb, config).map_err(|e| CompiledKokoroError::WeightLoadFailed {
                source: Box::new(e),
            })?;

        Self::new(model)
    }

    /// Load Kokoro from a safetensors file with custom hard bounds config.
    ///
    /// Combines [`load`](Self::load) weight loading with custom verification
    /// thresholds and rejection policy. Useful for benchmarks that need synthesis
    /// to succeed regardless of audio quality (e.g., `RejectionPolicy::Warn`).
    ///
    /// # Safety
    ///
    /// Same as [`Self::load`]: safetensors file must not be modified while alive.
    ///
    /// Part of #4262.
    pub unsafe fn load_with_hard_bounds(
        path: impl AsRef<Path>,
        hard_bounds: HardBoundsConfig,
    ) -> Result<Self, CompiledKokoroError> {
        let path = path.as_ref();
        let device = cpu();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device) }
            .map_err(|e| CompiledKokoroError::WeightLoadFailed {
                source: Box::new(e.into()),
            })?;
        let model = KokoroModel::load(&vb, &KokoroConfig::default())
            .map_err(|e| CompiledKokoroError::WeightLoadFailed {
                source: Box::new(e),
            })?;
        Self::new_with_hard_bounds(model, hard_bounds)
    }

    /// Create a new `CompiledKokoro` from a loaded model.
    ///
    /// Transfers SourceModule weights to GPU. Returns error if GPU transfer fails.
    ///
    /// No compilation happens here — segments are compiled lazily at the
    /// first `synthesize()` call and cached for reuse.
    pub fn new(model: KokoroModel) -> Result<Self, CompiledKokoroError> {
        model.config().validate()?;
        Ok(Self {
            shared: SharedKokoroState::new(model)?,
            seg_plbert: SegmentCache::new(),
            seg_text: SegmentCache::new(),
            seg_prosody: SegmentCache::new(),
            seg_f0: SegmentCache::new(),
            seg_generator: SegmentCache::new(),
            seg_regulate: SegmentCache::new(),
            seg_sinegen_pre: SegmentCache::new(),
            seg_sinegen_post: SegmentCache::new(),
            plbert_emb_cache: HashMap::new(),
            regulate_total_cache: HashMap::new(),
            // ICB replay enabled by default: shape tracking is lightweight
            // and the buffer's record/replay methods exit early when no
            // segments are cached. Phase 2 will wire actual ICB recording.
            // Part of #4264.
            icb_replay: crate::compiled_model::icb::replay::IcbReplayBuffer::new(
                crate::compiled_model::icb::replay::IcbReplayConfig::enabled(),
            ),
            sinegen_last_cumphase: None,
            shape_policy: ShapePolicy::Fixed,
            mixed_precision: false,
            autocast_policy: None,
            auto_release: false,
            segment_cache_config: SegmentCacheConfig::default(),
            peephole_configs: HashMap::new(),
            segment_autocast: None,
            crown_verification: false,
            crown_config: nn_tts_verify::CrownCertificateConfig::default(),
            pipeline_mode: PipelineMode::default(),
            #[cfg(feature = "plan-serde")]
            optimization_results: None,
        })
    }

    /// Create a new `CompiledKokoro` with custom hard bounds verification config.
    ///
    /// Same as [`new`](Self::new) but configures the embedded [`TtsVerifier`]
    /// with the given [`HardBoundsConfig`], allowing per-check threshold
    /// overrides and a custom rejection policy (Warn, Reject, Remediate).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nn_tts_verify::{HardBoundsConfig, RejectionPolicy, CheckOverrides};
    ///
    /// let hb = HardBoundsConfig {
    ///     rejection_policy: RejectionPolicy::Warn,
    ///     overrides: CheckOverrides {
    ///         max_amplitude: Some(1.5),
    ///         ..Default::default()
    ///     },
    ///     ..Default::default()
    /// };
    /// let mut kokoro = CompiledKokoro::new_with_hard_bounds(model, hb)?;
    /// ```
    ///
    /// Part of #3780, #3758, #3760.
    pub fn new_with_hard_bounds(
        model: KokoroModel,
        hard_bounds: HardBoundsConfig,
    ) -> Result<Self, CompiledKokoroError> {
        model.config().validate()?;
        Ok(Self {
            shared: SharedKokoroState::with_hard_bounds(model, hard_bounds)?,
            seg_plbert: SegmentCache::new(),
            seg_text: SegmentCache::new(),
            seg_prosody: SegmentCache::new(),
            seg_f0: SegmentCache::new(),
            seg_generator: SegmentCache::new(),
            seg_regulate: SegmentCache::new(),
            seg_sinegen_pre: SegmentCache::new(),
            seg_sinegen_post: SegmentCache::new(),
            plbert_emb_cache: HashMap::new(),
            regulate_total_cache: HashMap::new(),
            // ICB replay enabled by default: see new() comment. Part of #4264.
            icb_replay: crate::compiled_model::icb::replay::IcbReplayBuffer::new(
                crate::compiled_model::icb::replay::IcbReplayConfig::enabled(),
            ),
            sinegen_last_cumphase: None,
            shape_policy: ShapePolicy::Fixed,
            mixed_precision: false,
            autocast_policy: None,
            auto_release: false,
            segment_cache_config: SegmentCacheConfig::default(),
            peephole_configs: HashMap::new(),
            segment_autocast: None,
            crown_verification: false,
            crown_config: nn_tts_verify::CrownCertificateConfig::default(),
            pipeline_mode: PipelineMode::default(),
            #[cfg(feature = "plan-serde")]
            optimization_results: None,
        })
    }

    /// Enable F16 mixed-precision for 2x Metal ALU throughput.
    ///
    /// Dispatch steps (elementwise, matmul, conv, etc.) use F16. NativeOps
    /// (LSTM, fused ResBlock, fused norm) stay F32. Boundary casts are
    /// inserted automatically. Call before the first `synthesize()`.
    ///
    /// **Warning:** This path stores intermediate buffers in F16. Production
    /// Kokoro weights produce activations that overflow F16 range (±65504),
    /// causing NaN. Use [`with_autocast()`](Self::with_autocast) instead.
    #[deprecated(
        since = "0.1.0",
        note = "Causes NaN with production weights. Use with_autocast() instead"
    )]
    #[must_use]
    pub fn with_mixed_precision(mut self) -> Self {
        self.mixed_precision = true;
        self
    }

    /// Enable per-op autocast mixed precision with a custom policy.
    ///
    /// Compute/passthrough steps use F16 buffers, Accumulate ops (softmax,
    /// norms) stay F32. Mixed GEMM steps use F16 weights with F32
    /// accumulators. Safe with production weights — no NaN.
    /// Call before the first `synthesize()`.
    ///
    /// Invalidates ICB replay cache entries (pre-encoded dispatch commands
    /// have fixed buffer bindings with dtype-specific byte widths).
    ///
    /// Part of #3085, #2981, #4264.
    #[must_use]
    pub fn with_autocast_policy(mut self, policy: MixedPrecisionPolicy) -> Self {
        self.autocast_policy = Some(policy);
        // ICB replay encodes fixed buffer byte widths. Changing the autocast
        // policy changes which buffers are F16 vs F32, invalidating all cached
        // ICBs. Part of #4264 (F16 autocast + ICB replay interaction).
        self.icb_replay.invalidate_all();
        self
    }

    /// Enable per-op autocast with Apple Silicon defaults (BF16 weights, F16 compute).
    ///
    /// Convenience wrapper around [`with_autocast_policy()`](Self::with_autocast_policy).
    #[must_use]
    pub fn with_autocast(self) -> Self {
        self.with_autocast_policy(MixedPrecisionPolicy::apple_silicon_default())
    }

    /// Enable per-segment F16 autocast with granular control.
    ///
    /// When set, overrides the uniform `autocast_policy` on a per-segment
    /// basis. Segments where `config.policy_for_segment(name)` returns
    /// `Some(policy)` use that policy for autocast; segments returning `None`
    /// compile without autocast (F32).
    ///
    /// Also sets the uniform `autocast_policy` to the config's `base_policy`
    /// as a fallback for any code paths that read `autocast_policy` directly.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nn_metal::F16AutocastConfig;
    /// use nn_core::mixed_precision::MixedPrecisionPolicy;
    ///
    /// // All segments F16 except regulate (elementwise, no benefit).
    /// let config = F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default())
    ///     .with_regulate(false);
    /// let kokoro = CompiledKokoro::new(model)?.with_segment_autocast(config);
    /// ```
    ///
    /// Part of #4269, #4264.
    #[must_use]
    pub fn with_segment_autocast(mut self, config: F16AutocastConfig) -> Self {
        // Set uniform policy as fallback for code that reads autocast_policy directly.
        self.autocast_policy = Some(config.base_policy.clone());
        self.segment_autocast = Some(config);
        // ICB replay encodes fixed buffer byte widths per segment. Changing
        // per-segment autocast config changes which segments use F16 vs F32,
        // invalidating all cached ICBs. Part of #4264.
        self.icb_replay.invalidate_all();
        self
    }

    /// Enable recommended per-segment F16 autocast for maximum throughput.
    ///
    /// Enables F16 for all compute-heavy segments (PlBert, TextEncoder,
    /// ProsodyPredictor, F0EnergyPredictor, Generator, SineGen post-cumsum)
    /// while keeping lightweight elementwise segments in F32.
    ///
    /// Part of #4269.
    #[must_use]
    pub fn with_recommended_autocast(self) -> Self {
        self.with_segment_autocast(
            F16AutocastConfig::recommended(MixedPrecisionPolicy::apple_silicon_default()),
        )
    }

    /// Enable fast half-precision accumulators in FusedResBlock conv kernels.
    ///
    /// When `enabled` is `true` and per-segment autocast is configured with
    /// the generator segment enabled, the 24 FusedResBlocks switch from
    /// float-accumulator F16 (~1.36x) to half-accumulator F16 (~2x).
    ///
    /// **No-op if `segment_autocast` is not set.** Call
    /// [`with_segment_autocast`](Self::with_segment_autocast) or
    /// [`with_recommended_autocast`](Self::with_recommended_autocast) first.
    ///
    /// Default: `false` (safe, opt-in only).
    #[must_use]
    pub fn with_fast_half_accumulator(mut self, enabled: bool) -> Self {
        if let Some(ref mut config) = self.segment_autocast {
            config.use_fast_half_accumulator = enabled;
        }
        self
    }

    /// Returns the per-segment autocast configuration, if set.
    #[must_use]
    pub fn segment_autocast(&self) -> Option<&F16AutocastConfig> {
        self.segment_autocast.as_ref()
    }

    /// Auto-release CPU model weights after first successful synthesis.
    ///
    /// Saves ~320 MB RSS for Kokoro-82M. After release, compiled segments
    /// continue working (GPU buffers unaffected), but new input shapes
    /// **cannot** be compiled. Call `precompile_shapes()` first if multiple
    /// shapes are expected.
    ///
    /// Part of #3079.
    #[must_use]
    pub fn with_auto_release_weights(mut self) -> Self {
        self.auto_release = true;
        self
    }

    /// Configure per-shape segment cache capacity and eviction policy.
    ///
    /// Replaces all 8 segment caches with new caches using the given config's
    /// `max_segments_per_step` as capacity (clamped to minimum 1). Call before
    /// the first `synthesize()` — existing cached segments are discarded.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nn_metal::segment_cache::{SegmentCacheConfig, EvictionPolicy};
    ///
    /// let config = SegmentCacheConfig {
    ///     max_segments_per_step: 8,
    ///     eviction: EvictionPolicy::Lru,
    ///     ..SegmentCacheConfig::default()
    /// };
    /// let kokoro = CompiledKokoro::new(model)?
    ///     .with_segment_cache_config(config);
    /// ```
    ///
    /// Part of #3634.
    #[must_use]
    pub fn with_segment_cache_config(mut self, config: SegmentCacheConfig) -> Self {
        self.seg_plbert = SegmentCache::with_config(&config);
        self.seg_text = SegmentCache::with_config(&config);
        self.seg_prosody = SegmentCache::with_config(&config);
        self.seg_f0 = SegmentCache::with_config(&config);
        self.seg_generator = SegmentCache::with_config(&config);
        self.seg_regulate = SegmentCache::with_config(&config);
        self.seg_sinegen_pre = SegmentCache::with_config(&config);
        self.seg_sinegen_post = SegmentCache::with_config(&config);
        self.segment_cache_config = config;
        self
    }

    /// Returns the current segment cache configuration.
    #[must_use]
    pub fn segment_cache_config(&self) -> &SegmentCacheConfig {
        &self.segment_cache_config
    }

    /// Returns aggregate cache statistics across all 8 segment caches.
    ///
    /// Sums hits, misses, evictions, and total_bytes from all segment caches.
    /// Use this for monitoring cache efficiency and tuning `SegmentCacheConfig`.
    #[must_use]
    pub fn segment_cache_stats(&self) -> crate::segment_cache::SegmentCacheStats {
        let caches = [
            self.seg_plbert.stats(),
            self.seg_text.stats(),
            self.seg_prosody.stats(),
            self.seg_f0.stats(),
            self.seg_generator.stats(),
            self.seg_regulate.stats(),
            self.seg_sinegen_pre.stats(),
            self.seg_sinegen_post.stats(),
        ];
        let mut agg = crate::segment_cache::SegmentCacheStats::default();
        for s in &caches {
            agg.hits += s.hits;
            agg.misses += s.misses;
            agg.evictions += s.evictions;
            agg.total_bytes += s.total_bytes;
        }
        agg
    }

    /// Reset cache statistics counters across all 8 segment caches.
    pub fn reset_segment_cache_stats(&mut self) {
        self.seg_plbert.reset_stats();
        self.seg_text.reset_stats();
        self.seg_prosody.reset_stats();
        self.seg_f0.reset_stats();
        self.seg_generator.reset_stats();
        self.seg_regulate.reset_stats();
        self.seg_sinegen_pre.reset_stats();
        self.seg_sinegen_post.reset_stats();
    }

    /// Set per-segment [`PeepholeConfig`](nn_dsl::PeepholeConfig) overrides
    /// for selective fusion pass control.
    ///
    /// Keys are segment names: `"plbert"`, `"text"`, `"prosody"`, `"f0"`,
    /// `"generator"`, `"regulate"`, `"sinegen_pre"`, `"sinegen_post"`.
    /// Segments without a config use the default (all passes enabled).
    /// Call before the first `synthesize()`.
    ///
    /// Part of #3828 Phase 2B.
    #[must_use]
    pub fn with_peephole_configs(
        mut self,
        configs: HashMap<String, nn_dsl::PeepholeConfig>,
    ) -> Self {
        self.peephole_configs = configs;
        self
    }

    /// Returns the current per-segment peephole configurations.
    #[must_use]
    pub fn peephole_configs(&self) -> &HashMap<String, nn_dsl::PeepholeConfig> {
        &self.peephole_configs
    }

    /// Load per-segment optimal [`PeepholeConfig`](nn_dsl::PeepholeConfig)
    /// from a persisted [`KokoroOptimalConfigs`](optimal_configs::KokoroOptimalConfigs) file.
    ///
    /// If the file exists and the configs are valid for the current nn
    /// version, applies per-segment configs. If the file is missing or
    /// stale (version mismatch), falls back to default configs (all passes
    /// enabled) without error.
    ///
    /// Call before the first `synthesize()`.
    ///
    /// Requires the `plan-serde` feature.
    ///
    /// Part of #3828.
    #[cfg(feature = "plan-serde")]
    #[must_use]
    pub fn with_optimal_configs(mut self, path: &Path) -> Self {
        match optimal_configs::load_optimal_configs_if_exists(path) {
            Ok(Some(configs)) if configs.is_current() => {
                self.peephole_configs = configs.to_peephole_map();
            }
            _ => {
                // File missing, stale, or error — use default configs silently.
            }
        }
        self
    }

    /// Enable post-synthesis CROWN certificate generation.
    ///
    /// When enabled, each synthesis call maps the runtime Certificate's
    /// hard-bound results to moonshot property evidence (P1 non-silence,
    /// P2 non-clipping, P6 streaming safety) and attaches a
    /// [`MoonshotCertificate`](nn_tts_verify::MoonshotCertificate) to the
    /// returned Certificate.
    ///
    /// Call before the first `synthesize()`.
    ///
    /// Part of #4254, #3874.
    #[must_use]
    pub fn with_crown_verification(mut self, enabled: bool) -> Self {
        self.crown_verification = enabled;
        self
    }

    /// Set the CROWN certificate configuration.
    ///
    /// Controls model name, input specification, and hard-bound mapping
    /// for the post-synthesis CROWN certificate. Implicitly enables
    /// CROWN verification.
    ///
    /// Part of #4254, #3874.
    #[must_use]
    pub fn with_crown_config(mut self, config: nn_tts_verify::CrownCertificateConfig) -> Self {
        self.crown_config = config;
        self.crown_verification = true;
        self
    }

    /// Returns whether CROWN verification is enabled.
    #[must_use]
    pub fn crown_verification_enabled(&self) -> bool {
        self.crown_verification
    }

    /// Set the GPU dispatch strategy for single-voice synthesis.
    ///
    /// Default is [`PipelineMode::TwoPhase`], which splits the forward pass at
    /// the regulate sync point and uses explicit Phase 1/Phase 2 GPU fencing
    /// for maximum CPU-GPU overlap. Set to [`PipelineMode::Sequential`] for
    /// debugging — the sequential path dispatches steps in order with per-step
    /// fence submissions but without the structural two-phase split.
    ///
    /// Call before the first `synthesize()`.
    ///
    /// Part of #4264.
    #[must_use]
    pub fn with_pipeline_mode(mut self, mode: PipelineMode) -> Self {
        self.pipeline_mode = mode;
        self
    }

    /// Returns the current pipeline mode.
    #[must_use]
    pub fn pipeline_mode(&self) -> PipelineMode {
        self.pipeline_mode
    }

    /// Set the shape policy for all compiled segments.
    ///
    /// `ShapePolicy::Fixed` (default): shapes baked at compile time. Each new
    /// sequence length triggers a full recompile (trace + compile + GPU upload).
    ///
    /// `ShapePolicy::Polymorphic { max_seq_len, max_t_mel }`: sequence
    /// dimensions resolved at runtime. Buffers are pre-allocated at max
    /// dimensions; Metal pipelines are compiled once and reused for any input
    /// shape within the max bounds. Eliminates recompilation for variable-length
    /// TTS inputs.
    ///
    /// Call before the first `synthesize()`.
    ///
    /// Part of #3873.
    #[must_use]
    pub fn with_shape_policy(mut self, policy: ShapePolicy) -> Self {
        self.shape_policy = policy;
        self
    }

    /// Enable zero-recompilation mode for TTS with variable input lengths.
    ///
    /// Convenience wrapper around [`with_shape_policy()`](Self::with_shape_policy)
    /// that sets `ShapePolicy::Polymorphic` with the given maximum dimensions.
    /// After one warmup compilation at max dimensions, all subsequent inputs
    /// with smaller sequence lengths dispatch without recompilation.
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` -- Maximum token sequence length. Must be >= the largest
    ///   `seq_len` that will be passed at runtime. Typical values: 128 (chat),
    ///   512 (narration).
    /// * `max_t_mel` -- Maximum mel frame count. Must be >= the largest `t_mel`
    ///   at runtime. Typical values: 320 (chat), 1024 (narration).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let kokoro = CompiledKokoro::new(model)?
    ///     .with_zero_recompilation(512, 1024)
    ///     .with_autocast();
    /// // First synthesize() compiles at max dims. All subsequent calls
    /// // with seq_len <= 512 and t_mel <= 1024 reuse compiled segments.
    /// ```
    ///
    /// Part of #3873.
    #[must_use]
    pub fn with_zero_recompilation(self, max_seq_len: usize, max_t_mel: usize) -> Self {
        self.with_shape_policy(ShapePolicy::Polymorphic {
            max_seq_len,
            max_t_mel,
        })
    }

    /// Returns the shape policy for this pipeline.
    #[must_use]
    pub fn shape_policy(&self) -> ShapePolicy {
        self.shape_policy
    }

    /// Enable pipeline-level ICB replay for Kokoro dispatch optimization.
    ///
    /// When enabled, the first forward pass for a given input shape records
    /// GPU dispatch commands into Metal Indirect Command Buffers (ICBs).
    /// Subsequent passes with the same shape replay the pre-encoded ICBs
    /// via a single `executeCommandsInBuffer` call per phase, eliminating
    /// CPU-side pipeline lookup, threadgroup calculation, and buffer binding
    /// for each of the ~192 dispatches in a Kokoro forward pass.
    ///
    /// The pipeline is split at the `regulate_scalar_readback` sync point:
    /// - **Pre-readback** (steps 1-4): PlBert, TextEncoder, Prosody, Regulate
    /// - **Post-readback** (steps 5-8): F0Energy, HarmonicSource, Generator, iSTFT
    ///
    /// Each phase is independently cached and replayed. Requires deterministic
    /// arena offsets (guaranteed after warmup).
    ///
    /// Gated by the `icb-replay` feature flag. Call before `synthesize()`.
    ///
    /// Part of #4264.
    #[must_use]
    pub fn with_icb_replay(mut self) -> Self {
        self.icb_replay = crate::compiled_model::icb::replay::IcbReplayBuffer::new(
            crate::compiled_model::icb::replay::IcbReplayConfig::enabled(),
        );
        self
    }

    /// Enable pipeline-level ICB replay with custom configuration.
    ///
    /// See [`with_icb_replay()`](Self::with_icb_replay) for details.
    /// The config controls cache capacity, minimum segment size, and
    /// arena offset validation.
    ///
    /// Part of #4264.
    #[must_use]
    pub fn with_icb_replay_config(
        mut self,
        config: crate::compiled_model::icb::replay::IcbReplayConfig,
    ) -> Self {
        self.icb_replay = crate::compiled_model::icb::replay::IcbReplayBuffer::new(config);
        self
    }

    /// Returns whether pipeline-level ICB replay is enabled.
    #[must_use]
    pub fn icb_replay_enabled(&self) -> bool {
        self.icb_replay.is_enabled()
    }

    /// Returns diagnostic statistics for the ICB replay buffer.
    #[must_use]
    pub fn icb_replay_stats(&self) -> crate::compiled_model::icb::replay::IcbReplayBufferStats {
        self.icb_replay.stats()
    }

    /// Invalidate all cached ICB replays.
    ///
    /// Use when the model configuration changes (e.g., autocast policy,
    /// peephole configs) that would make pre-encoded ICBs invalid.
    pub fn invalidate_icb_replay(&mut self) {
        self.icb_replay.invalidate_all();
    }

    /// Generate a deployment certificate from the verification status file.
    ///
    /// Reads `nn_verify_status_kokoro.json`, aggregates per-entry verification
    /// status (sound/heuristic/vacuous), includes all 6 junction contract bounds,
    /// and produces a [`KokoroCertificate`](nn_verify::KokoroCertificate) that
    /// can be serialized to JSON and shipped alongside the deployed model.
    ///
    /// The certificate includes a content hash for tamper detection and can be
    /// independently verified using
    /// [`verify_kokoro_certificate()`](nn_verify::verify_kokoro_certificate)
    /// without running the model.
    ///
    /// # Arguments
    ///
    /// * `model_hash` -- SHA-256 hash of the model weights file.
    /// * `status_path` -- Path to `nn_verify_status_kokoro.json`.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::VerificationFailed` if the status file
    /// cannot be read or parsed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::path::Path;
    ///
    /// let mut kokoro = CompiledKokoro::new(model)?;
    /// let cert = kokoro.generate_deployment_certificate(
    ///     "sha256_of_weights",
    ///     Path::new("nn_verify_status_kokoro.json"),
    /// )?;
    /// cert.save(Path::new("kokoro.proof.json"))?;
    /// ```
    ///
    /// Part of #4254, #3874.
    #[cfg(feature = "verify")]
    pub fn generate_deployment_certificate(
        &self,
        model_hash: &str,
        status_path: &Path,
    ) -> Result<nn_verify::KokoroCertificate, CompiledKokoroError> {
        let config = nn_verify::KokoroCertificateConfig::new(model_hash, status_path);
        nn_verify::generate_kokoro_certificate(&config).map_err(|e| {
            CompiledKokoroError::CertificateGenerationFailed {
                reason: e.to_string(),
            }
        })
    }

    /// Returns the per-segment optimization results from the last
    /// [`warmup_with_optimizer()`](Self::warmup_with_optimizer) call, if any.
    ///
    /// Each entry is `(segment_name, OptimizationResult)` with the best
    /// config, dispatch count, baseline comparison, and cost estimates.
    ///
    /// Returns `None` if `warmup_with_optimizer()` has not been called yet.
    ///
    /// Part of #3828.
    #[cfg(feature = "plan-serde")]
    #[must_use]
    pub fn optimization_results(&self) -> Option<&[(String, nn_dsl::OptimizationResult)]> {
        self.optimization_results.as_deref()
    }

    /// Human-readable summary of per-segment optimization results.
    ///
    /// Shows baseline vs optimal dispatch count and cost reduction for each
    /// segment. Returns a placeholder message if `warmup_with_optimizer()`
    /// has not been called.
    ///
    /// # Example output
    ///
    /// ```text
    /// === Kokoro Segment Optimization Summary ===
    /// plbert:       baseline  25 -> optimal  18 dispatches (-28.0%), cost 150.2 -> 102.3 us (-31.9%)
    /// text:         baseline  12 -> optimal  10 dispatches (-16.7%), cost  80.1 ->  72.5 us ( -9.5%)
    /// generator:    baseline  45 -> optimal  32 dispatches (-28.9%), cost 450.0 -> 310.2 us (-31.1%)
    /// Total:        baseline  82 -> optimal  60 dispatches (-26.8%)
    /// ```
    ///
    /// Part of #3828.
    #[cfg(feature = "plan-serde")]
    #[must_use]
    pub fn optimization_summary(&self) -> String {
        let results = match &self.optimization_results {
            Some(r) if !r.is_empty() => r,
            _ => return "No optimization results available. Call warmup_with_optimizer() first.".to_string(),
        };

        let mut lines = Vec::with_capacity(results.len() + 3);
        lines.push("=== Kokoro Segment Optimization Summary ===".to_string());

        let mut total_baseline = 0usize;
        let mut total_optimal = 0usize;

        for (name, result) in results {
            let baseline = result.baseline_dispatch_count;
            let optimal = result.dispatch_count;
            total_baseline += baseline;
            total_optimal += optimal;

            let dispatch_pct = if baseline > 0 {
                let saved = baseline.saturating_sub(optimal);
                (saved as f64 / baseline as f64) * -100.0
            } else {
                0.0
            };

            let cost_info = if result.baseline_cost_ns > 0.0 {
                let cost_pct =
                    (result.baseline_cost_ns - result.best_cost_ns) / result.baseline_cost_ns
                        * -100.0;
                format!(
                    ", cost {:.1} -> {:.1} us ({:+.1}%)",
                    result.baseline_cost_ns / 1e3,
                    result.best_cost_ns / 1e3,
                    cost_pct,
                )
            } else {
                String::new()
            };

            lines.push(format!(
                "{name:<14} baseline {baseline:>3} -> optimal {optimal:>3} dispatches ({dispatch_pct:+.1}%){cost_info}",
            ));
        }

        let total_pct = if total_baseline > 0 {
            let saved = total_baseline.saturating_sub(total_optimal);
            (saved as f64 / total_baseline as f64) * -100.0
        } else {
            0.0
        };
        lines.push(format!(
            "Total:         baseline {total_baseline:>3} -> optimal {total_optimal:>3} dispatches ({total_pct:+.1}%)",
        ));

        lines.join("\n")
    }

    /// Clear all segment caches, forcing recompilation on next use.
    ///
    /// Shared GPU weight buffers are preserved so recompilation aliases
    /// existing weights (zero-copy) instead of re-uploading. This is used
    /// when peephole configs change: segments compiled with the old config
    /// must be evicted so they recompile with the new optimal config.
    ///
    /// Part of #3828.
    pub(crate) fn clear_segment_caches(&mut self) {
        self.seg_plbert.clear();
        self.seg_text.clear();
        self.seg_prosody.clear();
        self.seg_f0.clear();
        self.seg_generator.clear();
        self.seg_regulate.clear();
        self.seg_sinegen_pre.clear();
        self.seg_sinegen_post.clear();
    }

    /// Create a lightweight dispatch instance sharing compiled state.
    ///
    /// The new instance shares model weights, verifier, and iSTFT basis
    /// with the original via `Arc`. Segment caches are seeded with the
    /// parent's shared GPU weight buffers (if any segments have been
    /// compiled), so the first compilation on the new instance aliases
    /// existing GPU buffers instead of re-uploading weights.
    ///
    /// Use this for multi-voice synthesis pools where N voices share one
    /// set of weights. Memory: 7 `clone_dispatch()` instances use ~1.02x
    /// the memory of 1 instance (vs 7x without sharing).
    ///
    /// Part of #2740.
    #[must_use]
    pub fn clone_dispatch(&self) -> Self {
        /// Seed a new cache from the parent's shared weights and config, or
        /// create with config only.
        fn seed(parent: &SegmentCache, config: &SegmentCacheConfig) -> SegmentCache {
            match parent.shared_weights() {
                Some(w) => {
                    // Alias each buffer (ARC ref count bump, zero-copy).
                    let aliases = w.iter().map(|(k, b)| (k.clone(), b.alias())).collect();
                    SegmentCache::with_config_and_shared_weights(config, aliases)
                }
                None => SegmentCache::with_config(config),
            }
        }
        let cfg = &self.segment_cache_config;
        Self {
            shared: Arc::clone(&self.shared),
            seg_plbert: seed(&self.seg_plbert, cfg),
            seg_text: seed(&self.seg_text, cfg),
            seg_prosody: seed(&self.seg_prosody, cfg),
            seg_f0: seed(&self.seg_f0, cfg),
            seg_generator: seed(&self.seg_generator, cfg),
            seg_regulate: seed(&self.seg_regulate, cfg),
            seg_sinegen_pre: seed(&self.seg_sinegen_pre, cfg),
            seg_sinegen_post: seed(&self.seg_sinegen_post, cfg),
            plbert_emb_cache: HashMap::new(),
            regulate_total_cache: HashMap::new(),
            // Each clone gets its own replay buffer — lightweight (cache only).
            icb_replay: crate::compiled_model::icb::replay::IcbReplayBuffer::new(
                self.icb_replay.config().clone(),
            ),
            sinegen_last_cumphase: None,
            shape_policy: self.shape_policy,
            mixed_precision: self.mixed_precision,
            autocast_policy: self.autocast_policy.clone(),
            auto_release: false, // clones share Arc — can't auto-release
            segment_cache_config: self.segment_cache_config.clone(),
            peephole_configs: self.peephole_configs.clone(),
            segment_autocast: self.segment_autocast.clone(),
            crown_verification: self.crown_verification,
            crown_config: self.crown_config.clone(),
            pipeline_mode: self.pipeline_mode,
            #[cfg(feature = "plan-serde")]
            optimization_results: self.optimization_results.clone(),
        }
    }

    /// Create a lightweight dispatch instance sharing compiled segments.
    ///
    /// Like [`clone_dispatch()`](Self::clone_dispatch), shares model weights,
    /// verifier, and iSTFT basis via `Arc`. Additionally, **shares compiled
    /// Metal pipelines** from all segment caches via `Arc<CompiledModelDef>`,
    /// so the new instance can dispatch immediately for any shape the parent
    /// has compiled — no recompilation needed.
    ///
    /// Each clone gets independent execution buffers (`cached_planned_buf`,
    /// `cached_icbs`). Compilation cost (~1.2s per segment per shape) is
    /// eliminated; only buffer allocation (~1ms) happens on first dispatch.
    ///
    /// For a 4-voice chorus with 8 segments compiled for one shape each,
    /// this saves ~30s of compilation overhead compared to `clone_dispatch()`.
    ///
    /// Part of #4104.
    #[must_use]
    pub fn clone_dispatch_warm(&self) -> Self {
        let cfg = &self.segment_cache_config;
        Self {
            shared: Arc::clone(&self.shared),
            seg_plbert: self.seg_plbert.clone_warm(cfg),
            seg_text: self.seg_text.clone_warm(cfg),
            seg_prosody: self.seg_prosody.clone_warm(cfg),
            seg_f0: self.seg_f0.clone_warm(cfg),
            seg_generator: self.seg_generator.clone_warm(cfg),
            seg_regulate: self.seg_regulate.clone_warm(cfg),
            seg_sinegen_pre: self.seg_sinegen_pre.clone_warm(cfg),
            seg_sinegen_post: self.seg_sinegen_post.clone_warm(cfg),
            plbert_emb_cache: HashMap::new(),
            regulate_total_cache: HashMap::new(),
            icb_replay: crate::compiled_model::icb::replay::IcbReplayBuffer::new(
                self.icb_replay.config().clone(),
            ),
            sinegen_last_cumphase: None,
            shape_policy: self.shape_policy,
            mixed_precision: self.mixed_precision,
            autocast_policy: self.autocast_policy.clone(),
            auto_release: false, // clones share Arc — can't auto-release
            segment_cache_config: self.segment_cache_config.clone(),
            peephole_configs: self.peephole_configs.clone(),
            segment_autocast: self.segment_autocast.clone(),
            crown_verification: self.crown_verification,
            crown_config: self.crown_config.clone(),
            pipeline_mode: self.pipeline_mode,
            #[cfg(feature = "plan-serde")]
            optimization_results: self.optimization_results.clone(),
        }
    }

    /// Access the underlying model configuration.
    ///
    /// Always available, even after [`release_model_weights()`](Self::release_model_weights).
    #[must_use]
    pub fn config(&self) -> &KokoroConfig {
        &self.shared.config
    }

    /// Reset the SineGen cumulative phase for a new utterance.
    ///
    /// Call at the start of each streaming session to ensure the first chunk
    /// begins with zero phase. Within a session, phase continuity across
    /// chunk boundaries is maintained automatically by `build_harmonic_source`.
    pub fn reset_sinegen_phase(&mut self) {
        self.sinegen_last_cumphase = None;
    }

    /// Release CPU model weights to reduce RSS memory (~320 MB for Kokoro-82M).
    ///
    /// After all segments have been compiled (either via `synthesize()` or
    /// `precompile_shapes()`), the CPU model weights are no longer needed —
    /// compiled segments have their own GPU `MetalBuffer` copies. This method
    /// drops the `KokoroModel` to free the CPU `ArrayD<f32>` weight data.
    ///
    /// After release:
    /// - Existing compiled segments continue to work (GPU buffers unaffected).
    /// - `config()` and `source_module` remain available.
    /// - New input shapes **cannot** be compiled (returns `WeightsReleased`).
    ///   Call `precompile_shapes()` for all expected shapes before releasing.
    ///
    /// Requires sole ownership of `SharedKokoroState` — all `clone_dispatch()`
    /// instances must be dropped first. Returns `SharedOwnership` if other
    /// instances hold references.
    ///
    /// Part of #3079.
    pub fn release_model_weights(&mut self) -> Result<(), CompiledKokoroError> {
        let shared = Arc::get_mut(&mut self.shared).ok_or(CompiledKokoroError::SharedOwnership)?;
        shared.model = None;
        // Clear embedding cache — references model data no longer available.
        self.plbert_emb_cache.clear();
        Ok(())
    }

    /// Returns `true` if model weights have been released via
    /// [`release_model_weights()`](Self::release_model_weights).
    #[must_use]
    pub fn weights_released(&self) -> bool {
        self.shared.model.is_none()
    }
}

// -- Synthesize pipeline extracted to compiled_kokoro_pipeline.rs (#2575) --

#[path = "compiled_kokoro_pipeline.rs"]
mod pipeline;

// -- Pipeline-level ICB replay wiring (#4264) --

#[path = "compiled_kokoro_icb_replay.rs"]
mod icb_replay_wiring;

// -- Two-phase CPU-GPU segment pipelining (#4264) --

#[path = "compiled_kokoro_segment_pipeline.rs"]
mod segment_pipeline;

// -- Non-blocking synthesis via GpuFence (#4251) --

#[path = "compiled_kokoro_async.rs"]
mod async_synth;

// -- Per-synthesis arena utilization report (#4264) --

#[path = "compiled_kokoro_arena_report.rs"]
mod arena_report;
pub use arena_report::KokoroArenaReport;

// -- Lazy buffer allocation with size-class pooling (#4264) --

#[path = "compiled_kokoro_lazy_alloc.rs"]
pub mod lazy_alloc;
pub use lazy_alloc::{LazyBufferPool, LazyPoolStats};

// -- Diagnostic accessors extracted to compiled_kokoro_diagnostics.rs --

#[path = "compiled_kokoro_diagnostics.rs"]
mod diagnostics;
pub use diagnostics::{
    DiagnosticOutput, DispatchCensus, DispatchSummary, GpuTimingReport, MemoryBreakdown,
    SegmentCensus, TimingReport,
};

// -- Performance profiling infrastructure (#4264) --

#[path = "compiled_kokoro_perf_profile.rs"]
pub mod perf_profile;
pub use perf_profile::{
    format_profile_report, identify_bottleneck, BottleneckKind, PipelineProfile, SegmentProfile,
};

// -- Segment compilation extracted to compiled_kokoro_segments.rs --

#[path = "compiled_kokoro_segments.rs"]
mod segments;

// -- Shared trace functions for segments + precompile (#2218 trace-dedup) --

#[path = "compiled_kokoro_trace_fns.rs"]
mod trace_fns;

// -- GPU-native bridge helpers extracted to compiled_kokoro_bridges.rs (#2744, #2785) --

#[path = "compiled_kokoro_bridges.rs"]
mod bridges;

// -- Step result types extracted to compiled_kokoro_steps_types.rs --

#[path = "compiled_kokoro_steps_types.rs"]
mod step_types;
pub use step_types::{
    StepEncodeResult, StepF0EnergyResult, StepGeneratorResult, StepProsodyResult,
    StepRegulateResult, StyleSplit, SynthesisIntermediates,
};

// -- Step-by-step execution API for dvoice integration --

#[path = "compiled_kokoro_steps.rs"]
mod steps;

#[path = "compiled_kokoro_step_regulate.rs"]
mod step_regulate;

// -- Shared helpers extracted to compiled_kokoro_helpers.rs (Wave 4 D2, #2575) --

#[path = "compiled_kokoro_helpers.rs"]
mod helpers;
use helpers::{
    check_multi_output, cpu, generator_total_samples, gpu, model_device, prepare_synthesis_inputs,
    seg_cache_miss, seg_compile_err, set_last_output, trace_input, validate_input_ids,
};

// -- Build-time MSL pre-compilation for representative shapes (#2218) --

#[path = "compiled_kokoro_precompile.rs"]
pub mod precompile;

// -- Component registry for pipeline discoverability (#2923) --

#[path = "compiled_kokoro_registry.rs"]
mod registry;

// -- Pre-warmed shared segment cache (#4104) --

#[path = "compiled_kokoro_shared_segment_cache.rs"]
mod shared_segment_cache;
pub use shared_segment_cache::SharedSegmentCache;

// -- Multi-voice chorus pipeline (#3355, #3351, #2740) --

#[path = "compiled_kokoro_chorus.rs"]
pub mod chorus;
#[path = "compiled_kokoro_chorus_streaming.rs"]
mod chorus_streaming;
pub use chorus::KokoroChorus;

// -- GPU synthesis backend for KokoroTextPipeline (#3351 Step 2) --

#[path = "compiled_kokoro_gpu_synth.rs"]
pub mod gpu_synth;
pub use gpu_synth::{ChorusGpuSynth, GpuSynth};

// -- Streaming synthesis with crossfade (#3355, #2918) --

#[path = "compiled_kokoro_streaming.rs"]
mod streaming;

// -- Pull-based streaming session for single-voice (#4105) --

#[path = "compiled_kokoro_pull_streaming.rs"]
mod pull_streaming;
pub use pull_streaming::StreamingKokoroSession;

// -- Callback-driven streaming session (#4105) --

#[path = "compiled_kokoro_streaming_session.rs"]
mod streaming_session;
pub use streaming_session::CompiledKokoroStreamingSession;

// -- Pull-based streaming session for chorus (#4105) --

#[path = "compiled_kokoro_chorus_pull_streaming.rs"]
mod chorus_pull_streaming;
pub use chorus_pull_streaming::{ChorusChunkMode, StreamingChorusSession};

// -- Channel-based pull streaming session (#4105) --

#[path = "compiled_kokoro_channel_streaming.rs"]
mod channel_streaming;
pub use channel_streaming::{ChannelStreamingSession, StreamChunk, StreamReceiver};

// -- Segment fusion planner for Metal submit reduction (#4264) --

#[path = "compiled_kokoro_segment_fusion.rs"]
pub mod segment_fusion;
pub use segment_fusion::{
    FusedGroup, FusionPlan, SegmentFusionPlanner, SegmentInfo,
    can_fuse, plan_segment_fusion,
};

// -- Conv1d dispatch batching optimizer (#4264) --

#[path = "compiled_kokoro_conv_batch.rs"]
pub mod conv_batch;
pub use conv_batch::{Conv1dBatchGroup, ConvBatchAnalysis, ConvBatchOptimizer, PipelineConvBatchSummary};

// -- Fusion gap analysis for all 8 segments (#3836) --

#[path = "compiled_kokoro_gap_analysis.rs"]
mod gap_analysis;
pub use gap_analysis::SegmentGapAnalysis;

// -- PeepholeConfig optimizer search for all segments (#3828 Phase 2C) --

#[path = "compiled_kokoro_optimizer.rs"]
mod optimizer;
pub use optimizer::SegmentOptimizerResult;

// -- Closed-loop RTF optimizer (#4264) --

#[path = "compiled_kokoro_rtf_optimizer.rs"]
pub mod rtf_optimizer;
pub use rtf_optimizer::{RtfBottleneck, RtfOptimizer, RtfReport, SegmentReport};

// -- Metal command encoder batching for dispatch reduction (#4264) --

#[path = "compiled_kokoro_encoder_batch.rs"]
pub mod encoder_batch;
pub use encoder_batch::{BatchStats, EncoderBatchPlanner, EncoderGroup};

// -- Phantom type-tagged pipeline shape verification (#3635) --

#[path = "compiled_kokoro_shapes.rs"]
pub mod compiled_kokoro_shapes;
pub use compiled_kokoro_shapes::{
    PipelineTensor, TypedEncodeResult, TypedF0EnergyResult, TypedGeneratorResult,
    TypedProsodyResult, TypedRegulateResult,
};

#[path = "compiled_kokoro_typed_steps.rs"]
mod typed_steps;

// -- PeepholeConfig JSON loading/saving utility (#3828 Phase 2B) --

/// Load per-segment [`PeepholeConfig`](nn_dsl::PeepholeConfig) overrides
/// from a JSON file.
///
/// Expected format:
/// ```json
/// {
///   "plbert": { "norm_activ_conv1d": true, "fused_resblock": true, ... },
///   "generator": { "norm_activ_conv1d": true, ... }
/// }
/// ```
///
/// Only segments with entries are overridden; segments absent from the file
/// use the default config (all passes enabled).
///
/// Requires the `plan-serde` feature (for PeepholeConfig serde derives).
///
/// Part of #3828 Phase 2B.
#[cfg(feature = "plan-serde")]
pub fn load_peephole_configs(
    path: &Path,
) -> Result<HashMap<String, nn_dsl::PeepholeConfig>, CompiledKokoroError> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| CompiledKokoroError::ConfigLoad(format!("read {}: {e}", path.display())))?;
    let configs: HashMap<String, nn_dsl::PeepholeConfig> = serde_json::from_str(&data)
        .map_err(|e| CompiledKokoroError::ConfigLoad(format!("parse {}: {e}", path.display())))?;
    Ok(configs)
}

/// Save per-segment [`PeepholeConfig`](nn_dsl::PeepholeConfig) overrides
/// to a JSON file.
///
/// Uses pretty-printed JSON for debuggability and diffing. Creates or
/// overwrites the file at `path`.
///
/// Requires the `plan-serde` feature (for PeepholeConfig serde derives).
///
/// Part of #3828.
#[cfg(feature = "plan-serde")]
pub fn save_peephole_configs(
    configs: &HashMap<String, nn_dsl::PeepholeConfig>,
    path: &Path,
) -> Result<(), CompiledKokoroError> {
    let json = serde_json::to_string_pretty(configs)
        .map_err(|e| CompiledKokoroError::ConfigLoad(format!("serialize: {e}")))?;
    std::fs::write(path, json)
        .map_err(|e| CompiledKokoroError::ConfigLoad(format!("write {}: {e}", path.display())))?;
    Ok(())
}

// -- Per-segment optimal PeepholeConfig persistence (#3828 Phase 4) --

#[cfg(feature = "plan-serde")]
#[path = "compiled_kokoro_optimal_configs.rs"]
pub mod optimal_configs;
#[cfg(feature = "plan-serde")]
pub use optimal_configs::{
    load_optimal_configs, load_optimal_configs_if_exists, save_optimal_configs,
    KokoroOptimalConfigs, SegmentOptimalConfig,
};

#[cfg(all(test, feature = "plan-serde"))]
#[path = "compiled_kokoro_optimal_configs_tests.rs"]
mod optimal_configs_tests;

#[cfg(kani)]
#[path = "kani_compiled_kokoro_streaming.rs"]
mod kani_compiled_kokoro_streaming;

#[cfg(test)]
#[path = "compiled_kokoro_tests.rs"]
mod tests;
