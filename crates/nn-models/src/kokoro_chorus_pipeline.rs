// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integrated chorus master pipeline chaining all processing stages.
//!
//! Orchestrates the full per-voice + bus processing chain:
//!
//! ```text
//! Per-voice:  vibrato → detuning → EQ → de-essing → humanize
//! Mix:        blend voices → stereo imaging
//! Bus:        dynamics compression → reverb → final limiter
//! ```
//!
//! Each stage is optional (controlled via `Option<T>` sub-configs in
//! [`ChorusMasterConfig`]). Missing stages are skipped with zero cost.
//!
//! # Presets
//!
//! **Basic presets:**
//! - [`ChorusMasterConfig::minimal`] — just mix (blend + stereo)
//! - [`ChorusMasterConfig::standard`] — EQ + stereo + compression
//! - [`ChorusMasterConfig::full`] — every stage enabled
//!
//! **Production presets (use-case tuned):**
//! - [`ChorusMasterConfig::singing_chorus`] — choir/singing (wide stereo, vibrato, reverb)
//! - [`ChorusMasterConfig::speaking_chorus`] — speech (narrow stereo, no vibrato, tight dynamics)
//! - [`ChorusMasterConfig::intimate`] — close-mic (no reverb, tight stereo, warm EQ)
//! - [`ChorusMasterConfig::cathedral`] — large hall (huge reverb, wide stereo, deep vibrato)
//! - [`ChorusMasterConfig::broadcast`] — radio/podcast (tight dynamics, de-essing, no reverb)
//!
//! Part of #4264, #3351.

use crate::kokoro_chorus_adaptive_dynamics::{AdaptiveDynamicsConfig, AdaptiveDynamicsProcessor};
use crate::kokoro_chorus_air_absorption::{AirAbsorptionConfig, AirAbsorptionProcessor};
use crate::kokoro_chorus_alignment::{align_voices, AlignmentConfig};
use crate::kokoro_chorus_auto_eq::{AutoEqConfig, AutoEqProcessor};
use crate::kokoro_chorus_auto_mix::{AutoMixConfig, AutoMixer};
use crate::kokoro_chorus_bass_management::{BassManagementConfig, BassManager};
use crate::kokoro_chorus_bleed::{apply_voice_bleed, BleedConfig};
use crate::kokoro_chorus_blend::EnsembleBlendConfig;
use crate::kokoro_chorus_breath::{
    detect_pauses, insert_breath_sounds, BreathConfig, BreathGenerator,
};
use crate::kokoro_chorus_character::{apply_character_variation, CharacterConfig};
use crate::kokoro_chorus_convolution::{
    generate_synthetic_ir, ConvolutionConfig, ConvolutionReverb, SyntheticRoom,
};
use crate::kokoro_chorus_decorrelation::{DecorrelationConfig, DecorrelationProcessor};
use crate::kokoro_chorus_delay::{DelayConfig, MultiTapDelay};
use crate::kokoro_chorus_depth_staging::{DepthStagingConfig, DepthStagingProcessor};
use crate::kokoro_chorus_detune::{apply_detune, cents_to_rate, DetuneConfig};
use crate::kokoro_chorus_dither::{DitherConfig, DitherProcessor};
use crate::kokoro_chorus_doubler::{apply_doubler_per_voice, DoublerConfig};
use crate::kokoro_chorus_ducking::{DuckingConfig, SpectralDucker};
use crate::kokoro_chorus_dynamic_eq::{DynamicEqConfig, DynamicEqProcessor};
use crate::kokoro_chorus_dynamics::{
    BusLimiter, DynamicsPreset, MultibandCompressor, MultibandCompressorConfig,
};
use crate::kokoro_chorus_ensemble::{EnsembleConfig, EnsembleProcessor};
use crate::kokoro_chorus_eq::{DeEsserConfig, EqConfig, EqPreset, MixBusConfig, MixBusProcessor};
use crate::kokoro_chorus_exciter::{ExciterConfig, HarmonicExciter};
use crate::kokoro_chorus_formant::{
    shift_pitch_preserve_formant, simple_pitch_shift, FormantPreserveConfig,
};
use crate::kokoro_chorus_formant_tune::{FormantTuneConfig, FormantTuner};
use crate::kokoro_chorus_freeze::{FreezeConfig, SpectralFreezer};
use crate::kokoro_chorus_gain_staging::{GainStager, GainStagingConfig};
use crate::kokoro_chorus_gate::{apply_noise_gate, GateConfig};
use crate::kokoro_chorus_harmonic_tuner::{HarmonicTunerConfig, HarmonicTunerProcessor};
use crate::kokoro_chorus_hrtf::{HrtfConfig, HrtfProcessor};
use crate::kokoro_chorus_humanize::{apply_humanize, HumanizeConfig};
use crate::kokoro_chorus_intelligibility::{IntelligibilityConfig, IntelligibilityOptimizer};
use crate::kokoro_chorus_intonation::{IntonationConfig, IntonationTracker};
use crate::kokoro_chorus_limiter::{LimiterConfig, LimiterProcessor};
use crate::kokoro_chorus_loudness::{LoudnessConfig, LoudnessMeter};
use crate::kokoro_chorus_loudness_curve::{LoudnessCurveConfig, LoudnessCurveProcessor};
use crate::kokoro_chorus_masking_compensator::{MaskingCompensator, MaskingCompensatorConfig};
use crate::kokoro_chorus_mic_model::{MicModelConfig, MicModelProcessor};
use crate::kokoro_chorus_micro_pitch::{MicroPitchConfig, MicroPitchProcessor};
use crate::kokoro_chorus_mix_analyzer::{MixAnalyzerConfig, MixAnalyzerProcessor};
use crate::kokoro_chorus_multiband_stereo::{MultibandStereoConfig, MultibandStereoProcessor};
use crate::kokoro_chorus_onset_sync::{OnsetSyncConfig, OnsetSynchronizer};
use crate::kokoro_chorus_output::{FormattedOutput, OutputConfig, OutputFormatter};
use crate::kokoro_chorus_oversample::{OversampleConfig, Oversampler};
use crate::kokoro_chorus_pitch_correct::{
    apply_pitch_correction, PitchCorrectConfig,
};
use crate::kokoro_chorus_presence::{PresenceConfig, PresenceProcessor};
use crate::kokoro_chorus_reverb::ReverbConfig;
use crate::kokoro_chorus_room::{EarlyReflections, RoomConfig};
use crate::kokoro_chorus_saturation::{SaturationConfig, SaturationMode, SaturationProcessor};
use crate::kokoro_chorus_shimmer::{ShimmerConfig, ShimmerProcessor};
use crate::kokoro_chorus_sibilance::{align_sibilants, SibilanceConfig, SibilanceProcessor};
use crate::kokoro_chorus_spatial::{
    auto_layout_spatial, SpatialConfig, SpatialProcessor, VoiceSpatialPosition,
};
use crate::kokoro_chorus_spectral_fill::{SpectralFillConfig, SpectralFillProcessor};
use crate::kokoro_chorus_spectral_match::{SpectralMatchConfig, SpectralMatcher};
use crate::kokoro_chorus_stereo::{apply_stereo_mix, StereoChorusConfig};
use crate::kokoro_chorus_stereo_analysis::{StereoAnalysisConfig, StereoAnalyzer};
use crate::kokoro_chorus_stereo_optimizer::{StereoOptimizer, StereoOptimizerConfig};
use crate::kokoro_chorus_sub_bass::{SubBassConfig, SubBassEnhancer};
use crate::kokoro_chorus_thickener::{ThickenerConfig, ThickenerProcessor};
use crate::kokoro_chorus_tilt::{TiltConfig, TiltProcessor};
use crate::kokoro_chorus_transient::{apply_transient_shaping, TransientConfig};
use crate::kokoro_chorus_transient_align::{TransientAlignConfig, TransientAligner};
use crate::kokoro_chorus_vibrato::{apply_vibrato, VibratoConfig};
use crate::kokoro_chorus_vocal_tract::{VocalTractConfig, VocalTractProcessor};
use crate::kokoro_chorus_voice_alloc::{VoiceAllocConfig, VoiceAllocator};
use crate::kokoro_chorus_vowel_align::{VowelAlignConfig, VowelAligner};
use crate::kokoro_chorus_warmth::{WarmthConfig, WarmthProcessor};
use crate::kokoro_chorus_width::{StereoWidener, StereoWidthConfig};
use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Master configuration for the full chorus processing pipeline.
///
/// Each `Option<T>` field enables/disables a processing stage. When `None`,
/// that stage is skipped entirely (zero cost). Use the builder methods
/// (`with_eq`, `with_detune`, etc.) or the preset constructors
/// (`minimal`, `standard`, `full`).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ChorusMasterConfig {
    /// Number of voices in the chorus (1-32).
    pub n_voices: usize,
    /// Adaptive voice alignment (cross-correlation temporal sync). `None` = skip.
    ///
    /// When set, alignment is applied FIRST in the pipeline, before any other
    /// processing. Timing must be corrected before detuning, EQ, or vibrato
    /// to avoid misaligning the corrections themselves.
    pub alignment: Option<AlignmentConfig>,
    /// Per-voice EQ (biquad parametric). `None` = skip.
    pub eq: Option<EqConfig>,
    /// Per-voice de-esser (sibilance reduction). `None` = skip.
    pub deesser: Option<DeEsserConfig>,
    /// Per-voice LFO vibrato (F0 pitch modulation). `None` = skip.
    pub vibrato: Option<VibratoConfig>,
    /// Per-voice allpass detuning (±cents spread). `None` = skip.
    pub detune: Option<DetuneConfig>,
    /// Per-voice humanization (breathing, micro-timing, envelope). `None` = skip.
    pub humanize: Option<HumanizeConfig>,
    /// Ensemble blending (PSOLA formant-preserving pitch). `None` = skip.
    pub blend: Option<EnsembleBlendConfig>,
    /// Stereo imaging (constant-power pan law). `None` = mono mix.
    pub stereo: Option<StereoChorusConfig>,
    /// Bus multiband dynamics compressor. `None` = skip.
    pub dynamics: Option<MultibandCompressorConfig>,
    /// Bus saturation (harmonic warmth). `None` = skip.
    /// Applied after dynamics and before reverb so that saturation harmonics
    /// are colored by the reverb tail.
    pub saturation: Option<SaturationConfig>,
    /// Bus reverb (Schroeder). `None` = skip.
    pub reverb: Option<ReverbConfig>,
    /// Per-voice breath noise at pauses. `None` = skip.
    ///
    /// When enabled, synthetic breath sounds are inserted at detected pause
    /// regions with per-voice timing stagger. Applied after humanization and
    /// before stereo mixing.
    pub breath: Option<BreathConfig>,
    /// Per-voice spatial depth processing (distance attenuation, air absorption,
    /// propagation delay, ILD stereo panning). `None` = skip.
    /// When enabled, spatial processing replaces the default stereo panning
    /// stage since it provides its own stereo output per voice.
    pub spatial: Option<SpatialConfig>,
    /// Per-voice transient shaper (attack/sustain control). `None` = skip.
    ///
    /// When enabled, transient shaping is applied per-voice BEFORE EQ so
    /// that consonant/sustain balance is set before tonal processing. Placed
    /// after vibrato, before detuning.
    pub transient: Option<TransientConfig>,
    /// Per-voice bleed (microphone crosstalk simulation). `None` = skip.
    ///
    /// When enabled, voice bleed is applied as the last per-voice step,
    /// after humanize and breath, before stereo/spatial mixing. This adds
    /// subtle crosstalk between neighboring voices for cohesive ensemble feel.
    pub bleed: Option<BleedConfig>,
    /// Stereo width enhancement (mid/side, bass mono, Haas). `None` = skip.
    ///
    /// When enabled, stereo width processing is applied AFTER stereo/spatial
    /// mix and BEFORE bus dynamics. Enhances the stereo image with mid/side
    /// widening, bass mono filtering, and optional Haas effect delay.
    pub width: Option<StereoWidthConfig>,
    /// FFT convolution reverb with impulse responses. `None` = skip.
    ///
    /// When enabled, convolution reverb is applied AFTER Schroeder reverb
    /// (or as an alternative) and BEFORE the final limiter. Provides more
    /// realistic room simulation using real or synthetic impulse responses.
    pub convolution: Option<ConvolutionConfig>,
    /// Per-voice pitch correction (auto-tune / scale snapping). `None` = skip.
    ///
    /// When enabled, pitch correction is applied per-voice AFTER vibrato
    /// and BEFORE transient shaping. Snaps detected pitch toward the nearest
    /// note in the configured musical scale.
    pub pitch_correct: Option<PitchCorrectConfig>,
    /// Per-voice harmonic exciter (presence/air enhancement). `None` = skip.
    ///
    /// When enabled, the exciter is applied per-voice AFTER EQ to add
    /// harmonics and air-band shimmer after tonal shaping.
    pub exciter: Option<ExciterConfig>,
    /// Per-voice ADT vocal doubler. `None` = skip.
    ///
    /// When enabled, automatic double tracking is applied per-voice AFTER
    /// humanize to create doubled copies with timing/pitch variation.
    pub doubler: Option<DoublerConfig>,
    /// Per-voice spectral ducking (lead voice prominence). `None` = skip.
    ///
    /// When enabled, ducking is applied per-voice AFTER bleed to reduce
    /// non-lead voices when the lead is active.
    pub ducking: Option<DuckingConfig>,
    /// Bus gain staging (LUFS targeting + peak normalization). `None` = skip.
    ///
    /// When enabled, gain staging is applied on the bus BEFORE the limiter
    /// to auto-level the mix to a target LUFS.
    pub gain_staging: Option<GainStagingConfig>,
    /// Bus dithering (TPDF + noise shaping). `None` = skip.
    ///
    /// When enabled, dithering is applied as the absolute LAST bus step,
    /// AFTER the limiter, to decorrelate quantization noise before output.
    pub dither: Option<DitherConfig>,
    /// Bus limiter (peak limiting to -0.1 dBFS). `false` = skip.
    pub limiter_enabled: bool,
    /// Formant-preserving pitch shift for detuning. `None` = use basic
    /// allpass detuning only. When set AND `detune` is also configured,
    /// voices with >10 cents of detuning use the PSOLA formant-preserving
    /// algorithm, while voices with <=10 cents use the fast
    /// `simple_pitch_shift` path. This prevents the "chipmunk effect"
    /// where large pitch shifts distort vocal formant structure.
    pub formant_preserve: Option<FormantPreserveConfig>,
    /// Per-voice noise gate (silence cleanup). `None` = skip.
    ///
    /// When enabled, the noise gate is applied per-voice FIRST in the
    /// pipeline, before alignment, removing noise floor before any other
    /// processing. This prevents stacked noise from N voices amplifying
    /// the noise floor.
    pub gate: Option<GateConfig>,
    /// Per-voice timbral character variation. `None` = skip.
    ///
    /// When enabled, character variation is applied per-voice AFTER
    /// alignment, BEFORE vibrato. Assigns deterministic vocal tract
    /// scaling, breathiness, and brightness differences to each voice
    /// so they sound like distinct singers rather than clones.
    pub character: Option<CharacterConfig>,
    /// Bus early reflections room simulation. `None` = skip.
    ///
    /// When enabled, image-source early reflections are applied on the
    /// bus BEFORE Schroeder reverb. Early reflections provide spatial cues
    /// about room size and shape, complementing the late reverb tail.
    pub room: Option<RoomConfig>,
    /// Bus frequency-dependent stereo imaging. `None` = skip.
    ///
    /// When enabled, multi-band stereo is applied on the bus AFTER stereo
    /// width processing and BEFORE dynamics. Splits the stereo signal into
    /// low/mid/high bands with independent width control for each.
    pub multiband_stereo: Option<MultibandStereoConfig>,
    /// Bus spectral freeze effect. `None` = skip.
    ///
    /// When enabled, spectral freeze is applied on the bus AFTER
    /// reverb/convolution and BEFORE gain staging. Captures and sustains
    /// the current spectrum for drone/pad textures.
    pub freeze: Option<FreezeConfig>,
    /// HRTF binaural spatial processing. `None` = skip.
    ///
    /// When enabled, HRTF replaces the default stereo/spatial mix stage
    /// with head-related transfer function processing (ITD + ILD + head
    /// shadow) for immersive binaural output. Provides a third arm in the
    /// stereo section alongside spatial processors and stereo config.
    pub hrtf: Option<HrtfConfig>,
    /// Per-voice auto-EQ (spectral correction). `None` = skip.
    ///
    /// When enabled, auto-EQ is applied per-voice AFTER EQ/de-essing and
    /// BEFORE exciter. Analyzes each voice's spectrum and applies corrective
    /// filtering toward a target curve.
    pub auto_eq: Option<AutoEqConfig>,
    /// Bus loudness normalization (LUFS metering + normalization). `None` = skip.
    ///
    /// When enabled, loudness normalization is applied on the bus AFTER gain
    /// staging and BEFORE the limiter. Measures integrated loudness and
    /// normalizes to a target LUFS.
    pub loudness: Option<LoudnessConfig>,
    /// Per-voice sibilance processor (frequency-domain de-essing). `None` = skip.
    ///
    /// When enabled, sibilance processing is applied per-voice AFTER
    /// auto-EQ and BEFORE humanize. Provides more precise sibilance control
    /// than the basic de-esser, including cross-voice sibilant alignment.
    pub sibilance: Option<SibilanceConfig>,
    /// Bus ensemble processor (stereo modulation/diffusion). `None` = skip.
    ///
    /// When enabled, the ensemble processor is applied on the bus AFTER
    /// convolution reverb/freeze and BEFORE gain staging. Adds stereo
    /// modulation, chorus, and diffusion for a wider, richer sound.
    pub ensemble: Option<EnsembleConfig>,
    /// Per-voice vocal warmth and presence. `None` = skip.
    ///
    /// When enabled, warmth processing is applied per-voice AFTER exciter
    /// and BEFORE humanize. Adds analog-style body saturation and presence
    /// clarity without full-band saturation or exciter harmonics.
    pub warmth: Option<WarmthConfig>,
    /// Bus stereo correlation analysis and correction. `None` = skip.
    ///
    /// When enabled, stereo analysis is applied on the bus AFTER stereo
    /// width processing and BEFORE dynamics. Monitors and optionally
    /// corrects phase coherence issues for mono compatibility.
    pub stereo_analysis: Option<StereoAnalysisConfig>,
    /// Per-voice formant resonance tuning. `None` = skip.
    ///
    /// When enabled, formant tuning is applied per-voice AFTER character
    /// variation and BEFORE vibrato. Detects and reshapes formant frequencies
    /// to create timbre variation between voices.
    pub formant_tune: Option<FormantTuneConfig>,
    /// Per-voice micro-pitch drift (shimmer). `None` = skip.
    ///
    /// When enabled, micro-pitch drift is applied per-voice AFTER detuning
    /// and BEFORE EQ. Adds slow 1/f-like pitch wandering for natural
    /// ensemble shimmer.
    pub micro_pitch: Option<MicroPitchConfig>,
    /// Per-voice intonation correction. `None` = skip.
    ///
    /// When enabled, intonation tracking is applied per-voice AFTER vibrato
    /// and BEFORE pitch correction. Gently pulls voices toward a shared
    /// reference pitch to prevent inter-voice drift.
    pub intonation: Option<IntonationConfig>,
    /// Per-voice spectral envelope matching. `None` = skip.
    ///
    /// When enabled, spectral matching is applied per-voice AFTER intonation
    /// and BEFORE detuning. Aligns each voice's spectral envelope toward
    /// voice 0 so detuned voices retain the same timbral color.
    pub spectral_match: Option<SpectralMatchConfig>,
    /// Bus sub-harmonic bass enhancement. `None` = skip.
    ///
    /// When enabled, sub-bass enhancement is applied on the bus AFTER
    /// dynamics compression and BEFORE saturation.
    pub sub_bass: Option<SubBassConfig>,
    /// Bus adaptive dynamics with masking. `None` = skip.
    ///
    /// When enabled AND regular dynamics is also set, adaptive dynamics
    /// replaces regular dynamics. When only regular dynamics is set,
    /// regular dynamics is used.
    pub adaptive_dynamics: Option<AdaptiveDynamicsConfig>,
    /// Bus spectral tilt. `None` = skip.
    ///
    /// When enabled, spectral tilt is applied on the bus AFTER gain
    /// staging and BEFORE the limiter.
    pub tilt: Option<TiltConfig>,
    /// Per-voice onset synchronization. `None` = skip.
    ///
    /// When enabled, onset sync is applied per-voice AFTER alignment
    /// and BEFORE formant_tune.
    pub onset_sync: Option<OnsetSyncConfig>,
    /// Oversampling for saturation/exciter stages. `None` = skip.
    ///
    /// When enabled, wraps the saturation processing with 2x or 4x
    /// oversampling for anti-aliased waveshaping.
    pub oversample: Option<OversampleConfig>,
    /// Per-voice front-to-back depth staging. `None` = skip.
    ///
    /// When enabled, depth staging is applied per-voice AFTER alignment
    /// and BEFORE character. Positions voices in a front-to-back depth
    /// field with distance attenuation, air absorption LPF, pre-delay,
    /// and early reflections.
    pub depth_staging: Option<DepthStagingConfig>,
    /// Per-voice vocal tract resonance modeling. `None` = skip.
    ///
    /// When enabled, vocal tract processing is applied per-voice AFTER
    /// character and BEFORE formant_tune. Models cascaded formant resonators
    /// to give each voice a unique virtual singer body.
    pub vocal_tract: Option<VocalTractConfig>,
    /// Per-voice vocal shimmer and air harmonics. `None` = skip.
    ///
    /// When enabled, shimmer is applied per-voice AFTER warmth and BEFORE
    /// sibilance. Adds airy high-frequency harmonics and subtle brightness.
    pub shimmer: Option<ShimmerConfig>,
    /// Per-voice vocal intelligibility optimizer. `None` = skip.
    ///
    /// When enabled, intelligibility optimization is applied per-voice
    /// AFTER sibilance and BEFORE humanize. Protects critical speech
    /// frequency bands from masking.
    pub intelligibility: Option<IntelligibilityConfig>,
    /// Per-voice gain and pan allocation. `None` = skip.
    ///
    /// When enabled, voice allocation gains are applied per-voice AFTER
    /// ducking and BEFORE the mix stage.
    pub voice_alloc: Option<VoiceAllocConfig>,
    /// Bus frequency-dependent dynamic EQ. `None` = skip.
    ///
    /// When enabled, dynamic EQ is applied on the bus AFTER
    /// dynamics/adaptive_dynamics and BEFORE sub_bass.
    pub dynamic_eq: Option<DynamicEqConfig>,
    /// Bus psychoacoustic bass management. `None` = skip.
    ///
    /// When enabled, bass management is applied on the bus AFTER sub_bass
    /// and BEFORE saturation.
    pub bass_management: Option<BassManagementConfig>,
    /// Bus multi-tap delay/echo. `None` = skip.
    ///
    /// When enabled, delay is applied on the bus AFTER reverb/convolution
    /// and BEFORE freeze.
    pub delay: Option<DelayConfig>,
    /// Pre-mix spectral balance auto-mixer. `None` = skip.
    ///
    /// When enabled, auto-mix analyzes and adjusts per-voice gains
    /// BEFORE the blend/stereo mix stage.
    pub auto_mix: Option<AutoMixConfig>,
    /// Bus mix analyzer and auto-correction. `None` = skip.
    ///
    /// When enabled, the mix analyzer is applied on the bus as the
    /// absolute LAST step, AFTER dither.
    pub mix_analyzer: Option<MixAnalyzerConfig>,
    /// Per-voice allpass diffusion de-correlation. `None` = skip.
    ///
    /// When enabled, decorrelation is applied per-voice AFTER pitch
    /// correction and BEFORE spatial processing. Randomizes phase response
    /// per voice via cascaded allpass filters so voices sound distinct.
    pub decorrelation: Option<DecorrelationConfig>,
    /// Per-voice harmonic series tuning. `None` = skip.
    ///
    /// When enabled, harmonic tuning is applied per-voice AFTER vowel
    /// alignment and BEFORE vibrato. Analyzes and adjusts individual
    /// harmonics for clean voice stacking.
    pub harmonic_tuner: Option<HarmonicTunerConfig>,
    /// Bus equal-loudness contour compensation. `None` = skip.
    ///
    /// When enabled, loudness curve correction is applied on the bus
    /// AFTER loudness normalization and BEFORE the limiter. Compensates
    /// for Fletcher-Munson curves at the playback level.
    pub loudness_curve: Option<LoudnessCurveConfig>,
    /// Cross-voice psychoacoustic masking compensator. `None` = skip.
    ///
    /// When enabled, masking compensation is applied AFTER all per-voice
    /// processing and spectral fill, BEFORE the mix stage. Boosts masked
    /// spectral content using a simplified Zwicker model on the Bark scale.
    pub masking_compensator: Option<MaskingCompensatorConfig>,
    /// Per-voice microphone model and proximity effect. `None` = skip.
    ///
    /// When enabled, mic modeling is applied per-voice near the end of
    /// per-voice processing, AFTER masking compensation and BEFORE the
    /// mix stage. Simulates different mic types and proximity effect.
    pub mic_model: Option<MicModelConfig>,
    /// Bus spectral density and fullness optimizer. `None` = skip.
    ///
    /// When enabled, spectral fill is applied on the bus AFTER ensemble
    /// and BEFORE gain staging. Detects spectral gaps and generates
    /// complementary fill material.
    pub spectral_fill: Option<SpectralFillConfig>,
    /// Per-voice vowel formant alignment. `None` = skip.
    ///
    /// When enabled, vowel alignment is applied per-voice AFTER formant
    /// tuning and BEFORE harmonic tuning. Tracks and aligns formant
    /// frequencies toward a reference voice for better blending.
    pub vowel_align: Option<VowelAlignConfig>,
    /// Post-pipeline output formatting. `None` = skip.
    ///
    /// Not wired into `process()`. Use the separate
    /// [`ChorusMasterPipeline::format_output`] method after `process()`.
    pub output: Option<OutputConfig>,
    /// Per-voice frequency-dependent air absorption. `None` = skip.
    ///
    /// When enabled, air absorption is applied per-voice AFTER depth staging
    /// and BEFORE character variation. Models how HF content attenuates with
    /// distance, providing a natural depth cue (distant voices sound darker).
    pub air_absorption: Option<AirAbsorptionConfig>,
    /// Per-voice vocal presence enhancer. `None` = skip.
    ///
    /// When enabled, presence enhancement is applied per-voice AFTER shimmer
    /// and BEFORE sibilance processing. Dynamically boosts the 2-5 kHz
    /// presence band with sibilance-aware gain reduction.
    pub presence: Option<PresenceConfig>,
    /// Per-voice micro-modulation thickener. `None` = skip.
    ///
    /// When enabled, the thickener is applied per-voice AFTER humanize
    /// and BEFORE doubler. Adds subtle LFO-driven pitch, timing, and
    /// amplitude variations for a thicker, lusher chorus sound.
    pub thickener: Option<ThickenerConfig>,
    /// Bus stereo image optimizer. `None` = skip.
    ///
    /// When enabled, the stereo optimizer is applied on the bus AFTER
    /// stereo analysis and BEFORE dynamics. Monitors L/R correlation and
    /// automatically narrows the image when correlation drops, plus
    /// forces bass mono for playback compatibility.
    pub stereo_optimizer: Option<StereoOptimizerConfig>,
    /// Per-voice transient attack alignment. `None` = skip.
    ///
    /// When enabled, transient alignment is applied per-voice AFTER onset
    /// sync and BEFORE depth staging. Micro-shifts transient attacks so
    /// consonant onsets align across voices for tight ensemble cohesion.
    pub transient_align: Option<TransientAlignConfig>,
    /// Bus true peak limiter with oversampled detection. `None` = skip.
    ///
    /// When enabled AND the basic `limiter_enabled` is also true, the true
    /// peak limiter replaces the basic bus limiter for intersample-accurate
    /// peak control. When only basic limiter is enabled, it is used alone.
    pub true_peak_limiter: Option<LimiterConfig>,
}

impl ChorusMasterConfig {
    /// Create a new config with all stages disabled.
    ///
    /// Use builder methods to enable stages, or use a preset constructor.
    pub fn new(n_voices: usize) -> Result<Self, KokoroError> {
        if n_voices == 0 || n_voices > 32 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: format!("must be 1..=32, got {n_voices}"),
            });
        }
        Ok(Self {
            n_voices,
            alignment: None,
            eq: None,
            deesser: None,
            vibrato: None,
            detune: None,
            humanize: None,
            blend: None,
            stereo: None,
            dynamics: None,
            saturation: None,
            reverb: None,
            breath: None,
            spatial: None,
            transient: None,
            bleed: None,
            width: None,
            convolution: None,
            pitch_correct: None,
            exciter: None,
            doubler: None,
            ducking: None,
            gain_staging: None,
            dither: None,
            limiter_enabled: false,
            formant_preserve: None,
            gate: None,
            character: None,
            room: None,
            multiband_stereo: None,
            freeze: None,
            hrtf: None,
            auto_eq: None,
            loudness: None,
            sibilance: None,
            ensemble: None,
            warmth: None,
            stereo_analysis: None,
            formant_tune: None,
            micro_pitch: None,
            intonation: None,
            spectral_match: None,
            sub_bass: None,
            adaptive_dynamics: None,
            tilt: None,
            onset_sync: None,
            oversample: None,
            depth_staging: None,
            vocal_tract: None,
            shimmer: None,
            intelligibility: None,
            voice_alloc: None,
            dynamic_eq: None,
            bass_management: None,
            delay: None,
            auto_mix: None,
            mix_analyzer: None,
            decorrelation: None,
            harmonic_tuner: None,
            loudness_curve: None,
            masking_compensator: None,
            mic_model: None,
            spectral_fill: None,
            vowel_align: None,
            output: None,
            air_absorption: None,
            presence: None,
            thickener: None,
            stereo_optimizer: None,
            transient_align: None,
            true_peak_limiter: None,
        })
    }

    /// Minimal preset: just blend + stereo with auto layout.
    ///
    /// No per-voice processing, no bus processing. Cheapest option.
    pub fn minimal(n_voices: usize) -> Result<Self, KokoroError> {
        let mut config = Self::new(n_voices)?;
        config.blend = Some(EnsembleBlendConfig::default());
        config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?);
        Ok(config)
    }

    /// Standard preset: EQ + de-esser + stereo + dynamics + limiter.
    ///
    /// Good balance of quality and CPU cost.
    pub fn standard(n_voices: usize) -> Result<Self, KokoroError> {
        let mut config = Self::new(n_voices)?;
        config.eq = Some(EqPreset::Natural.to_config());
        config.deesser = Some(DeEsserConfig::default());
        config.blend = Some(EnsembleBlendConfig::default());
        config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?);
        config.dynamics = Some(DynamicsPreset::Gentle.to_config());
        config.limiter_enabled = true;
        Ok(config)
    }

    /// Full preset: every processing stage enabled.
    ///
    /// Maximum quality. Includes detuning, humanization, reverb.
    pub fn full(n_voices: usize) -> Result<Self, KokoroError> {
        let mut config = Self::new(n_voices)?;
        config.eq = Some(EqPreset::Natural.to_config());
        config.deesser = Some(DeEsserConfig::default());
        config.vibrato = Some(VibratoConfig::default());
        config.detune = Some(DetuneConfig::default());
        config.humanize = Some(HumanizeConfig::default());
        config.blend = Some(EnsembleBlendConfig::default());
        config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?);
        config.dynamics = Some(DynamicsPreset::Gentle.to_config());
        config.saturation = Some(
            SaturationConfig::new()
                .with_drive(0.15)
                .with_mix(0.5)
                .with_mode(SaturationMode::Warm)
                .with_output_gain_db(-0.5),
        );
        config.reverb = Some(ReverbConfig::default());
        // Spatial depth: moderate room with close listener.
        config.spatial = Some(
            SpatialConfig::new()
                .with_room_size(10.0)
                .with_listener_distance(3.0),
        );
        // Transient shaping: subtle attack boost for consonant clarity.
        config.transient = Some(
            TransientConfig::new()
                .with_attack(2.0)
                .with_sustain(-1.0)
                .with_sensitivity(1.5),
        );
        // Voice bleed: subtle crosstalk for cohesive ensemble.
        config.bleed = Some(
            BleedConfig::new()
                .with_bleed_amount(0.04)
                .with_proximity_rolloff(2.0),
        );
        // Stereo width: moderate mid/side enhancement with bass mono.
        config.width = Some(
            StereoWidthConfig::new()
                .with_width(1.2)
                .with_bass_mono_freq(80.0),
        );
        // Convolution reverb: medium hall IR for natural room ambience.
        config.convolution = Some(
            ConvolutionConfig::new()
                .with_wet_mix(0.15)
                .with_pre_delay_ms(10.0),
        );
        // Pitch correction: gentle chromatic correction for ensemble tuning.
        config.pitch_correct = Some(
            PitchCorrectConfig::default()
                .with_strength(0.3)
                .with_speed_ms(80.0),
        );
        // Exciter: subtle presence and air band enhancement.
        config.exciter = Some(
            ExciterConfig::new()
                .with_harmonics_mix(0.2)
                .with_air_gain_db(1.0),
        );
        // Doubler: subtle ADT for vocal thickness.
        config.doubler = Some(DoublerConfig::tight());
        // Ducking: gentle lead voice prominence.
        config.ducking = Some(DuckingConfig::default());
        // Gain staging: auto-level to -16 LUFS with headroom.
        config.gain_staging = Some(GainStagingConfig::default());
        // Dither: 24-bit TPDF with noise shaping for clean output.
        config.dither = Some(DitherConfig::default());
        config.limiter_enabled = true;
        Ok(config)
    }

    /// Singing chorus preset: optimized for choir/singing voice.
    ///
    /// Wide stereo imaging, deep vibrato for choir shimmer, moderate reverb
    /// for spatial depth, gentle dynamics to preserve vocal expression, warm
    /// EQ with slight presence boost for intelligibility, and full
    /// humanization (breathing, micro-timing, envelope) for a natural
    /// ensemble feel. De-tuning uses Gaussian distribution to concentrate
    /// voices near unison with a few wider outliers, mimicking real choir
    /// intonation spread.
    pub fn singing_chorus(n_voices: usize) -> Result<Self, KokoroError> {
        let mut config = Self::new(n_voices)?;
        // Moderate alignment tightness for natural choir feel: voices snap to
        // stressed syllables while allowing loose timing on unstressed ones.
        config.alignment = Some(AlignmentConfig::new(0.6)?);
        // Warm EQ: gentle low boost for body, slight presence lift at 3kHz,
        // rolled-off highs to avoid harshness when stacking voices.
        config.eq = Some(EqConfig {
            low_freq: 180.0,
            low_gain_db: 1.5,
            mid_freq: 3000.0,
            mid_gain_db: 1.0,
            mid_q: 0.8,
            high_freq: 9000.0,
            high_gain_db: -1.5,
        });
        // Light de-essing: chorus stacking amplifies sibilance.
        config.deesser = Some(DeEsserConfig {
            center_freq_hz: 6500.0,
            q: 1.2,
            threshold_db: -18.0,
            max_reduction_db: -8.0,
            attack_sec: 0.001,
            release_sec: 0.060,
        });
        // Natural singing vibrato: 5.5 Hz at 35 cents with per-voice spread,
        // creating the characteristic choir shimmer.
        config.vibrato = Some(VibratoConfig {
            rate_hz: 5.5,
            depth_cents: 35.0,
            rate_spread_hz: 0.5,
            depth_spread_cents: 10.0,
            onset_sec: 0.20,
        });
        // Gaussian detuning: concentrates voices near unison with a few wider
        // outliers, more realistic than uniform spread for choir intonation.
        config.detune = Some(DetuneConfig {
            cents_spread: 12.0,
            distribution: crate::kokoro_chorus_detune::DetuneDistribution::Gaussian,
            seed: 0,
        });
        // Full humanization: breathing, micro-timing, envelope.
        config.humanize = Some(HumanizeConfig::default());
        // Moderate ensemble blending for cohesive pitch.
        config.blend = Some(EnsembleBlendConfig {
            blend_strength: 0.4,
            formant_preservation: true,
            harmonic_alignment: true,
            min_period: 30,
            max_period: 300,
        });
        // Wide stereo: full pan spread for spacious choir image.
        config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(0.85));
        // Gentle dynamics: preserve vocal expression, don't squash dynamics.
        config.dynamics = Some(DynamicsPreset::Gentle.to_config());
        // Tape saturation: classic analog warmth to bind voices together.
        config.saturation = Some(
            SaturationConfig::new()
                .with_drive(0.2)
                .with_mix(0.4)
                .with_mode(SaturationMode::Tape)
                .with_output_gain_db(-1.0),
        );
        // Moderate reverb: spatial depth without muddying the voices.
        config.reverb = Some(ReverbConfig {
            reverb_mix: 0.18,
            room_size: 0.4,
            early_reflections: true,
            damping: 0.45,
        });
        // Formant preservation: critical for singing, where large detuning
        // (12 cents) would otherwise shift formants and produce an unnatural
        // "chipmunk" effect on voices with significant pitch offsets.
        config.formant_preserve = Some(FormantPreserveConfig::default());
        // Natural breathing: default breath noise fills pauses with subtle
        // per-voice staggered breath sounds for a realistic ensemble feel.
        config.breath = Some(BreathConfig::default());
        // Gentle chromatic pitch correction: subtle nudge toward semitones
        // to keep the choir's intonation cohesive without sounding robotic.
        config.pitch_correct = Some(
            PitchCorrectConfig::default()
                .with_strength(0.3)
                .with_speed_ms(120.0)
                .with_scale(crate::kokoro_chorus_pitch_correct::MusicalScale::Chromatic),
        );
        config.limiter_enabled = true;
        Ok(config)
    }

    /// Speaking chorus preset: optimized for speech/narration.
    ///
    /// Narrow stereo to maintain speech intelligibility, no vibrato (speech
    /// does not have natural vibrato), stronger dynamics for even loudness,
    /// broadcast EQ for clear articulation, aggressive de-essing since
    /// stacked speech creates harsh sibilance, and no reverb (reverb
    /// degrades speech clarity in multi-voice mixes).
    pub fn speaking_chorus(n_voices: usize) -> Result<Self, KokoroError> {
        let mut config = Self::new(n_voices)?;
        // Broadcast EQ: low cut for clarity, presence boost at 3.5kHz.
        config.eq = Some(EqPreset::Broadcast.to_config());
        // Aggressive de-essing: stacked speech creates harsh sibilance.
        config.deesser = Some(DeEsserConfig {
            center_freq_hz: 5500.0,
            q: 1.5,
            threshold_db: -24.0,
            max_reduction_db: -14.0,
            attack_sec: 0.001,
            release_sec: 0.040,
        });
        // No vibrato: speech does not have natural vibrato.
        config.vibrato = None;
        // Subtle detuning: just enough for voice differentiation, not obvious.
        config.detune = Some(DetuneConfig {
            cents_spread: 5.0,
            distribution: crate::kokoro_chorus_detune::DetuneDistribution::Uniform,
            seed: 0,
        });
        // Minimal humanization: envelope shaping only, no breathing/timing
        // (those artifacts are distracting in speech).
        config.humanize = Some(HumanizeConfig {
            enable_breath: false,
            enable_timing: false,
            enable_envelope: true,
            ..HumanizeConfig::default()
        });
        // Light blending for consistent speech timbre.
        config.blend = Some(EnsembleBlendConfig {
            blend_strength: 0.3,
            formant_preservation: true,
            harmonic_alignment: false,
            min_period: 30,
            max_period: 300,
        });
        // Narrow stereo: keeps speech intelligible and center-focused.
        config.stereo = Some(
            StereoChorusConfig::auto_layout(n_voices)?
                .with_stereo_width(0.4)
                .with_mono_compatible(true),
        );
        // Broadcast dynamics: medium compression for even loudness.
        config.dynamics = Some(DynamicsPreset::Broadcast.to_config());
        // No reverb: reverb degrades speech clarity.
        config.reverb = None;
        config.limiter_enabled = true;
        Ok(config)
    }

    /// Intimate preset: close-mic feel for warm, personal narration.
    ///
    /// No reverb, tight stereo, subtle detuning, warm EQ with low-end
    /// body and rolled-off highs. Creates the feeling of a small group
    /// of voices in a close, dry acoustic space. Ideal for audiobooks,
    /// podcasts, and whispered/gentle speech synthesis.
    pub fn intimate(n_voices: usize) -> Result<Self, KokoroError> {
        let mut config = Self::new(n_voices)?;
        // Warm EQ: low boost for proximity warmth, mid dip to reduce
        // nasal buildup, high roll-off for smooth texture.
        config.eq = Some(EqConfig {
            low_freq: 200.0,
            low_gain_db: 2.5,
            mid_freq: 2000.0,
            mid_gain_db: -1.5,
            mid_q: 0.7,
            high_freq: 7000.0,
            high_gain_db: -3.0,
        });
        // Gentle de-essing: close-mic captures more sibilance.
        config.deesser = Some(DeEsserConfig {
            center_freq_hz: 6000.0,
            q: 1.0,
            threshold_db: -22.0,
            max_reduction_db: -10.0,
            attack_sec: 0.001,
            release_sec: 0.050,
        });
        // No vibrato: intimate setting, voices are straight and close.
        config.vibrato = None;
        // Subtle detuning: just enough to prevent phasing artifacts
        // without obvious pitch spread.
        config.detune = Some(DetuneConfig {
            cents_spread: 4.0,
            distribution: crate::kokoro_chorus_detune::DetuneDistribution::Uniform,
            seed: 0,
        });
        // Envelope-only humanization: gentle attack/release shaping.
        config.humanize = Some(HumanizeConfig {
            enable_breath: false,
            enable_timing: false,
            enable_envelope: true,
            envelope: crate::kokoro_chorus_humanize::AmplitudeEnvelope {
                attack_sec: 0.040,
                hold_sec: 0.0,
                decay_sec: 0.030,
                sustain_level: 1.0,
                release_sec: 0.120,
            },
            ..HumanizeConfig::default()
        });
        // Light blending.
        config.blend = Some(EnsembleBlendConfig {
            blend_strength: 0.3,
            formant_preservation: true,
            harmonic_alignment: false,
            min_period: 30,
            max_period: 300,
        });
        // Tight stereo: voices close to center for intimate feel.
        config.stereo = Some(
            StereoChorusConfig::auto_layout(n_voices)?
                .with_stereo_width(0.3)
                .with_mono_compatible(true),
        );
        // Mastering-grade transparent dynamics: don't squash the intimacy.
        config.dynamics = Some(DynamicsPreset::Mastering.to_config());
        // No reverb: dry, close-mic sound.
        config.reverb = None;
        config.limiter_enabled = true;
        Ok(config)
    }

    /// Cathedral preset: huge reverb, wide stereo, dramatic chorus.
    ///
    /// Maximum spatial depth with large-hall reverb, wide stereo field,
    /// deep vibrato, generous detuning, and full humanization. Creates the
    /// impression of a choir singing in a large stone cathedral with long
    /// reverb tails. Dynamics are compressed more aggressively to keep the
    /// reverb tails from becoming muddy.
    pub fn cathedral(n_voices: usize) -> Result<Self, KokoroError> {
        let mut config = Self::new(n_voices)?;
        // Natural EQ: flat with slight presence dip to prevent harshness
        // in the reverb tail.
        config.eq = Some(EqConfig {
            low_freq: 200.0,
            low_gain_db: 0.5,
            mid_freq: 2500.0,
            mid_gain_db: -2.0,
            mid_q: 0.6,
            high_freq: 8000.0,
            high_gain_db: -1.0,
        });
        // De-essing: reverb amplifies sibilance, so moderate reduction.
        config.deesser = Some(DeEsserConfig {
            center_freq_hz: 6000.0,
            q: 1.0,
            threshold_db: -20.0,
            max_reduction_db: -10.0,
            attack_sec: 0.001,
            release_sec: 0.060,
        });
        // Deep vibrato: dramatic choir effect with wide per-voice spread.
        config.vibrato = Some(VibratoConfig {
            rate_hz: 5.0,
            depth_cents: 50.0,
            rate_spread_hz: 0.6,
            depth_spread_cents: 15.0,
            onset_sec: 0.25,
        });
        // Wide Gaussian detuning for thick, lush ensemble texture.
        config.detune = Some(DetuneConfig {
            cents_spread: 18.0,
            distribution: crate::kokoro_chorus_detune::DetuneDistribution::Gaussian,
            seed: 0,
        });
        // Full humanization: breathing, timing, and envelope for realism.
        config.humanize = Some(HumanizeConfig::default());
        // Strong ensemble blending: align the choir into a cohesive unit.
        config.blend = Some(EnsembleBlendConfig {
            blend_strength: 0.6,
            formant_preservation: true,
            harmonic_alignment: true,
            min_period: 30,
            max_period: 300,
        });
        // Maximum stereo width for immersive spatial image.
        config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(1.0));
        // Broadcast dynamics to keep reverb tails controlled.
        config.dynamics = Some(DynamicsPreset::Broadcast.to_config());
        // Console saturation: subtle cohesion from symmetric odd-harmonic
        // coloring, mimicking transformer-coupled mixing desks.
        config.saturation = Some(
            SaturationConfig::new()
                .with_drive(0.1)
                .with_mix(0.3)
                .with_mode(SaturationMode::Console)
                .with_output_gain_db(-0.5),
        );
        // Large-hall reverb: high wet mix, large room, warm damping.
        config.reverb = Some(ReverbConfig {
            reverb_mix: 0.35,
            room_size: 0.85,
            early_reflections: true,
            damping: 0.6,
        });
        // Formant preservation: essential for cathedral with 18-cent detuning,
        // where voices at the extremes would otherwise sound obviously pitch-
        // shifted. Preserving formants keeps all voices sounding like the same
        // vocal tract regardless of fundamental frequency offset.
        config.formant_preserve = Some(FormantPreserveConfig::default());
        // Cathedral breath: louder and longer breath sounds to fill the
        // large reverberant space, with wider stagger for spatial depth.
        config.breath = Some(
            BreathConfig::new()
                .with_noise_level(0.04)
                .with_duration_ms(140.0)
                .with_stagger_ms(50.0),
        );
        // Spatial depth: large cathedral space with distant listener.
        config.spatial = Some(
            SpatialConfig::new()
                .with_room_size(20.0)
                .with_listener_distance(5.0),
        );
        // Convolution reverb: Cathedral IR for authentic stone-hall ambience.
        // Higher wet mix than default to complement the Schroeder reverb tail.
        config.convolution = Some(
            ConvolutionConfig::new()
                .with_wet_mix(0.25)
                .with_pre_delay_ms(20.0),
        );
        config.limiter_enabled = true;
        Ok(config)
    }

    /// Broadcast preset: radio/podcast optimized.
    ///
    /// Tight dynamics for consistent loudness, aggressive de-essing,
    /// moderate stereo width (mono-compatible), broadcast EQ with
    /// presence boost and low cut, no reverb. Designed for radio,
    /// podcast, and streaming playback where loudness consistency,
    /// sibilance control, and mono compatibility are critical.
    pub fn broadcast(n_voices: usize) -> Result<Self, KokoroError> {
        let mut config = Self::new(n_voices)?;
        // Tight alignment for professional broadcast sync: voices lock to
        // the reference timing with minimal drift.
        config.alignment = Some(AlignmentConfig::new(0.8)?);
        // Broadcast EQ: cut below 100Hz, boost presence at 3.5kHz, slight
        // high roll-off to reduce hiss/noise buildup.
        config.eq = Some(EqConfig {
            low_freq: 100.0,
            low_gain_db: -4.0,
            mid_freq: 3500.0,
            mid_gain_db: 3.0,
            mid_q: 1.0,
            high_freq: 10000.0,
            high_gain_db: -1.5,
        });
        // Aggressive de-essing: broadcast standards require sibilance control.
        config.deesser = Some(DeEsserConfig {
            center_freq_hz: 5500.0,
            q: 1.3,
            threshold_db: -22.0,
            max_reduction_db: -14.0,
            attack_sec: 0.001,
            release_sec: 0.035,
        });
        // No vibrato: broadcast speech is straight-toned.
        config.vibrato = None;
        // Minimal detuning: just enough to avoid comb filtering artifacts.
        config.detune = Some(DetuneConfig {
            cents_spread: 3.0,
            distribution: crate::kokoro_chorus_detune::DetuneDistribution::Uniform,
            seed: 0,
        });
        // Envelope-only humanization: clean attack/release.
        config.humanize = Some(HumanizeConfig {
            enable_breath: false,
            enable_timing: false,
            enable_envelope: true,
            ..HumanizeConfig::default()
        });
        // Light blending: preserve each voice's character.
        config.blend = Some(EnsembleBlendConfig {
            blend_strength: 0.2,
            formant_preservation: true,
            harmonic_alignment: false,
            min_period: 30,
            max_period: 300,
        });
        // Moderate stereo width with mono compatibility for radio playback.
        config.stereo = Some(
            StereoChorusConfig::auto_layout(n_voices)?
                .with_stereo_width(0.5)
                .with_mono_compatible(true),
        );
        // Aggressive dynamics: tight compression for consistent loudness.
        config.dynamics = Some(DynamicsPreset::Aggressive.to_config());
        // No reverb: clean, dry broadcast sound.
        config.reverb = None;
        config.limiter_enabled = true;
        Ok(config)
    }

    // -- Builder methods --

    /// Enable adaptive voice alignment (cross-correlation temporal sync).
    ///
    /// When set, alignment is applied FIRST in the pipeline, before any other
    /// per-voice processing. Timing must be corrected before detuning, EQ, or
    /// vibrato to avoid misaligning the corrections themselves.
    #[must_use]
    pub fn with_alignment(mut self, alignment: AlignmentConfig) -> Self {
        self.alignment = Some(alignment);
        self
    }

    /// Enable per-voice EQ with the given config.
    #[must_use]
    pub fn with_eq(mut self, eq: EqConfig) -> Self {
        self.eq = Some(eq);
        self
    }

    /// Enable per-voice de-esser with the given config.
    #[must_use]
    pub fn with_deesser(mut self, deesser: DeEsserConfig) -> Self {
        self.deesser = Some(deesser);
        self
    }

    /// Enable per-voice LFO vibrato (F0 pitch modulation).
    #[must_use]
    pub fn with_vibrato(mut self, vibrato: VibratoConfig) -> Self {
        self.vibrato = Some(vibrato);
        self
    }

    /// Enable per-voice allpass detuning.
    #[must_use]
    pub fn with_detune(mut self, detune: DetuneConfig) -> Self {
        self.detune = Some(detune);
        self
    }

    /// Enable per-voice humanization.
    #[must_use]
    pub fn with_humanize(mut self, humanize: HumanizeConfig) -> Self {
        self.humanize = Some(humanize);
        self
    }

    /// Enable ensemble blending.
    #[must_use]
    pub fn with_blend(mut self, blend: EnsembleBlendConfig) -> Self {
        self.blend = Some(blend);
        self
    }

    /// Enable stereo imaging with the given config.
    #[must_use]
    pub fn with_stereo(mut self, stereo: StereoChorusConfig) -> Self {
        self.stereo = Some(stereo);
        self
    }

    /// Enable bus multiband dynamics compressor.
    #[must_use]
    pub fn with_dynamics(mut self, dynamics: MultibandCompressorConfig) -> Self {
        self.dynamics = Some(dynamics);
        self
    }

    /// Enable bus saturation (harmonic warmth).
    ///
    /// Applied after dynamics and before reverb so that saturation harmonics
    /// are colored by the reverb tail.
    #[must_use]
    pub fn with_saturation(mut self, saturation: SaturationConfig) -> Self {
        self.saturation = Some(saturation);
        self
    }

    /// Enable bus reverb.
    #[must_use]
    pub fn with_reverb(mut self, reverb: ReverbConfig) -> Self {
        self.reverb = Some(reverb);
        self
    }

    /// Enable per-voice breath noise at pauses.
    ///
    /// Synthetic breath sounds are detected at pause regions and inserted
    /// with per-voice timing stagger. Applied after humanization and before
    /// stereo mixing.
    #[must_use]
    pub fn with_breath(mut self, breath: BreathConfig) -> Self {
        self.breath = Some(breath);
        self
    }

    /// Enable per-voice spatial depth processing.
    ///
    /// When spatial is enabled, each voice is processed through distance-based
    /// attenuation, air absorption lowpass, propagation delay, and ILD stereo
    /// panning. Spatial processing replaces the default stereo panning stage
    /// since it provides its own stereo output per voice.
    #[must_use]
    pub fn with_spatial(mut self, spatial: SpatialConfig) -> Self {
        self.spatial = Some(spatial);
        self
    }

    /// Enable or disable the bus limiter.
    #[must_use]
    pub fn with_limiter(mut self, enabled: bool) -> Self {
        self.limiter_enabled = enabled;
        self
    }

    /// Enable formant-preserving pitch shift for detuning.
    ///
    /// When set alongside `detune`, voices with large detuning (>10 cents)
    /// use the PSOLA formant-preserving algorithm instead of basic allpass
    /// resampling. This prevents formant distortion ("chipmunk effect") on
    /// voices with significant pitch offsets.
    #[must_use]
    pub fn with_formant_preserve(mut self, config: FormantPreserveConfig) -> Self {
        self.formant_preserve = Some(config);
        self
    }

    /// Enable per-voice transient shaping (attack/sustain control).
    ///
    /// Transient shaping is applied per-voice BEFORE EQ, after vibrato,
    /// before detuning. Shapes consonant attacks and vowel sustain before
    /// tonal processing.
    #[must_use]
    pub fn with_transient(mut self, config: TransientConfig) -> Self {
        self.transient = Some(config);
        self
    }

    /// Enable per-voice bleed (microphone crosstalk simulation).
    ///
    /// Bleed is applied as the last per-voice step, after humanize and
    /// breath, before stereo/spatial mixing. Adds subtle crosstalk between
    /// neighboring voices for a cohesive ensemble feel.
    #[must_use]
    pub fn with_bleed(mut self, config: BleedConfig) -> Self {
        self.bleed = Some(config);
        self
    }

    /// Enable stereo width enhancement (mid/side processing).
    ///
    /// Width processing is applied AFTER stereo/spatial mix and BEFORE bus
    /// dynamics. Enhances the stereo image with mid/side widening, bass mono
    /// filtering, and optional Haas effect delay.
    #[must_use]
    pub fn with_width(mut self, config: StereoWidthConfig) -> Self {
        self.width = Some(config);
        self
    }

    /// Enable FFT convolution reverb with impulse responses.
    ///
    /// Convolution reverb is applied AFTER Schroeder reverb (or as an
    /// alternative) and BEFORE the final limiter. Provides more realistic
    /// room simulation using real or synthetic impulse responses.
    #[must_use]
    pub fn with_convolution(mut self, config: ConvolutionConfig) -> Self {
        self.convolution = Some(config);
        self
    }

    /// Enable per-voice pitch correction (auto-tune / scale snapping).
    ///
    /// Applied per-voice AFTER vibrato and BEFORE transient shaping.
    /// Snaps detected pitch toward the nearest note in the configured scale.
    #[must_use]
    pub fn with_pitch_correct(mut self, config: PitchCorrectConfig) -> Self {
        self.pitch_correct = Some(config);
        self
    }

    /// Enable per-voice harmonic exciter (presence/air enhancement).
    ///
    /// Applied per-voice AFTER EQ to add harmonics and air-band shimmer.
    #[must_use]
    pub fn with_exciter(mut self, config: ExciterConfig) -> Self {
        self.exciter = Some(config);
        self
    }

    /// Enable per-voice ADT vocal doubler.
    ///
    /// Applied per-voice AFTER humanize to create doubled copies with
    /// timing and pitch variation.
    #[must_use]
    pub fn with_doubler(mut self, config: DoublerConfig) -> Self {
        self.doubler = Some(config);
        self
    }

    /// Enable per-voice spectral ducking (lead voice prominence).
    ///
    /// Applied per-voice AFTER bleed to reduce non-lead voices when the
    /// lead voice is active.
    #[must_use]
    pub fn with_ducking(mut self, config: DuckingConfig) -> Self {
        self.ducking = Some(config);
        self
    }

    /// Enable bus gain staging (LUFS targeting + peak normalization).
    ///
    /// Applied on the bus BEFORE the limiter to auto-level the mix to
    /// a target LUFS while respecting a peak ceiling.
    #[must_use]
    pub fn with_gain_staging(mut self, config: GainStagingConfig) -> Self {
        self.gain_staging = Some(config);
        self
    }

    /// Enable bus dithering (TPDF + noise shaping).
    ///
    /// Applied as the absolute LAST bus step, AFTER the limiter, to
    /// decorrelate quantization noise before output.
    #[must_use]
    pub fn with_dither(mut self, config: DitherConfig) -> Self {
        self.dither = Some(config);
        self
    }

    /// Enable per-voice noise gate (silence cleanup).
    ///
    /// Applied per-voice FIRST in the pipeline, before alignment. Removes
    /// noise floor from silent/quiet sections so stacked voices don't
    /// amplify noise proportionally to voice count.
    #[must_use]
    pub fn with_gate(mut self, config: GateConfig) -> Self {
        self.gate = Some(config);
        self
    }

    /// Enable per-voice timbral character variation.
    ///
    /// Applied per-voice AFTER alignment, BEFORE vibrato. Assigns
    /// deterministic vocal tract scaling, breathiness, and brightness
    /// differences to each voice for distinct singer character.
    #[must_use]
    pub fn with_character(mut self, config: CharacterConfig) -> Self {
        self.character = Some(config);
        self
    }

    /// Enable bus early reflections room simulation.
    ///
    /// Applied on the bus BEFORE Schroeder reverb. Image-source early
    /// reflections provide spatial cues about room size and shape.
    #[must_use]
    pub fn with_room(mut self, config: RoomConfig) -> Self {
        self.room = Some(config);
        self
    }

    /// Enable bus frequency-dependent multi-band stereo imaging.
    ///
    /// Applied on the bus AFTER stereo width and BEFORE dynamics. Splits
    /// stereo into low/mid/high bands with independent width control.
    #[must_use]
    pub fn with_multiband_stereo(mut self, config: MultibandStereoConfig) -> Self {
        self.multiband_stereo = Some(config);
        self
    }

    /// Enable bus spectral freeze effect (drone/pad textures).
    ///
    /// Applied on the bus AFTER reverb/convolution and BEFORE gain staging.
    /// Captures the current spectrum and sustains it indefinitely.
    #[must_use]
    pub fn with_freeze(mut self, config: FreezeConfig) -> Self {
        self.freeze = Some(config);
        self
    }

    /// Enable HRTF binaural spatial processing.
    ///
    /// When set, HRTF replaces the default stereo/spatial mix stage with
    /// head-related transfer function processing for immersive binaural output.
    #[must_use]
    pub fn with_hrtf(mut self, config: HrtfConfig) -> Self {
        self.hrtf = Some(config);
        self
    }

    /// Enable per-voice auto-EQ (spectral correction).
    ///
    /// Applied per-voice AFTER EQ/de-essing and BEFORE exciter. Analyzes
    /// each voice's spectrum and applies corrective filtering.
    #[must_use]
    pub fn with_auto_eq(mut self, config: AutoEqConfig) -> Self {
        self.auto_eq = Some(config);
        self
    }

    /// Enable bus loudness normalization (LUFS metering).
    ///
    /// Applied on the bus AFTER gain staging and BEFORE the limiter.
    /// Measures integrated loudness and normalizes to a target LUFS.
    #[must_use]
    pub fn with_loudness(mut self, config: LoudnessConfig) -> Self {
        self.loudness = Some(config);
        self
    }

    /// Enable per-voice sibilance processing (frequency-domain de-essing).
    ///
    /// Applied per-voice AFTER auto-EQ and BEFORE humanize. More precise
    /// than the basic de-esser, with cross-voice sibilant alignment.
    #[must_use]
    pub fn with_sibilance(mut self, config: SibilanceConfig) -> Self {
        self.sibilance = Some(config);
        self
    }

    /// Enable bus ensemble processor (stereo modulation/diffusion).
    ///
    /// Applied on the bus AFTER convolution reverb/freeze and BEFORE gain
    /// staging. Adds stereo modulation and diffusion for a wider sound.
    #[must_use]
    pub fn with_ensemble(mut self, config: EnsembleConfig) -> Self {
        self.ensemble = Some(config);
        self
    }

    /// Enable per-voice vocal warmth and presence processing.
    ///
    /// Applied per-voice AFTER exciter and BEFORE humanize. Adds analog-style
    /// body saturation and presence clarity.
    #[must_use]
    pub fn with_warmth(mut self, config: WarmthConfig) -> Self {
        self.warmth = Some(config);
        self
    }

    /// Enable bus stereo correlation analysis and correction.
    ///
    /// Applied on the bus AFTER stereo width processing and BEFORE dynamics.
    /// Monitors and optionally corrects phase coherence issues.
    #[must_use]
    pub fn with_stereo_analysis(mut self, config: StereoAnalysisConfig) -> Self {
        self.stereo_analysis = Some(config);
        self
    }

    /// Enable per-voice formant resonance tuning.
    ///
    /// Applied per-voice AFTER character variation and BEFORE vibrato.
    /// Detects and reshapes formant frequencies for timbre variation.
    #[must_use]
    pub fn with_formant_tune(mut self, config: FormantTuneConfig) -> Self {
        self.formant_tune = Some(config);
        self
    }

    /// Enable per-voice micro-pitch drift (shimmer).
    ///
    /// Applied per-voice AFTER detuning and BEFORE EQ. Adds slow random
    /// pitch wandering for natural ensemble shimmer.
    #[must_use]
    pub fn with_micro_pitch(mut self, config: MicroPitchConfig) -> Self {
        self.micro_pitch = Some(config);
        self
    }

    /// Enable per-voice intonation correction.
    ///
    /// Applied per-voice AFTER vibrato and BEFORE pitch correction.
    /// Gently pulls voices toward a shared reference pitch.
    #[must_use]
    pub fn with_intonation(mut self, config: IntonationConfig) -> Self {
        self.intonation = Some(config);
        self
    }

    /// Enable per-voice spectral envelope matching.
    ///
    /// Applied per-voice AFTER intonation and BEFORE detuning. Aligns
    /// each voice's spectral envelope toward voice 0.
    #[must_use]
    pub fn with_spectral_match(mut self, config: SpectralMatchConfig) -> Self {
        self.spectral_match = Some(config);
        self
    }

    /// Enable bus sub-harmonic bass enhancement.
    ///
    /// Applied on the bus AFTER dynamics and BEFORE saturation.
    #[must_use]
    pub fn with_sub_bass(mut self, config: SubBassConfig) -> Self {
        self.sub_bass = Some(config);
        self
    }

    /// Enable bus adaptive dynamics with masking.
    ///
    /// When set alongside regular dynamics, adaptive dynamics replaces it.
    #[must_use]
    pub fn with_adaptive_dynamics(mut self, config: AdaptiveDynamicsConfig) -> Self {
        self.adaptive_dynamics = Some(config);
        self
    }

    /// Enable bus spectral tilt.
    ///
    /// Applied on the bus AFTER gain staging and BEFORE the limiter.
    #[must_use]
    pub fn with_tilt(mut self, config: TiltConfig) -> Self {
        self.tilt = Some(config);
        self
    }

    /// Enable per-voice onset synchronization.
    ///
    /// Applied per-voice AFTER alignment and BEFORE formant_tune.
    #[must_use]
    pub fn with_onset_sync(mut self, config: OnsetSyncConfig) -> Self {
        self.onset_sync = Some(config);
        self
    }

    /// Enable oversampling for saturation/exciter stages.
    ///
    /// Wraps saturation with 2x or 4x oversampling for anti-aliased
    /// waveshaping.
    #[must_use]
    pub fn with_oversample(mut self, config: OversampleConfig) -> Self {
        self.oversample = Some(config);
        self
    }

    /// Enable per-voice front-to-back depth staging.
    ///
    /// Applied per-voice AFTER alignment and BEFORE character. Positions
    /// voices in a front-to-back depth field with distance attenuation,
    /// air absorption LPF, pre-delay, and early reflections.
    #[must_use]
    pub fn with_depth_staging(mut self, config: DepthStagingConfig) -> Self {
        self.depth_staging = Some(config);
        self
    }

    /// Enable per-voice vocal tract resonance modeling.
    ///
    /// Applied per-voice AFTER character and BEFORE formant_tune. Models
    /// cascaded formant resonators to give each voice a unique virtual
    /// singer body.
    #[must_use]
    pub fn with_vocal_tract(mut self, config: VocalTractConfig) -> Self {
        self.vocal_tract = Some(config);
        self
    }

    /// Enable per-voice vocal shimmer and air harmonics.
    ///
    /// Applied per-voice AFTER warmth and BEFORE sibilance. Adds airy
    /// high-frequency harmonics and subtle brightness.
    #[must_use]
    pub fn with_shimmer(mut self, config: ShimmerConfig) -> Self {
        self.shimmer = Some(config);
        self
    }

    /// Enable per-voice vocal intelligibility optimizer.
    ///
    /// Applied per-voice AFTER sibilance and BEFORE humanize. Protects
    /// critical speech frequency bands from masking.
    #[must_use]
    pub fn with_intelligibility(mut self, config: IntelligibilityConfig) -> Self {
        self.intelligibility = Some(config);
        self
    }

    /// Enable per-voice gain and pan allocation.
    ///
    /// Applied per-voice AFTER ducking and BEFORE the mix stage.
    #[must_use]
    pub fn with_voice_alloc(mut self, config: VoiceAllocConfig) -> Self {
        self.voice_alloc = Some(config);
        self
    }

    /// Enable bus frequency-dependent dynamic EQ.
    ///
    /// Applied on the bus AFTER dynamics/adaptive_dynamics and BEFORE sub_bass.
    #[must_use]
    pub fn with_dynamic_eq(mut self, config: DynamicEqConfig) -> Self {
        self.dynamic_eq = Some(config);
        self
    }

    /// Enable bus psychoacoustic bass management.
    ///
    /// Applied on the bus AFTER sub_bass and BEFORE saturation.
    #[must_use]
    pub fn with_bass_management(mut self, config: BassManagementConfig) -> Self {
        self.bass_management = Some(config);
        self
    }

    /// Enable bus multi-tap delay/echo.
    ///
    /// Applied on the bus AFTER reverb/convolution and BEFORE freeze.
    #[must_use]
    pub fn with_delay(mut self, config: DelayConfig) -> Self {
        self.delay = Some(config);
        self
    }

    /// Enable pre-mix spectral balance auto-mixer.
    ///
    /// Analyzes and adjusts per-voice gains BEFORE the blend/stereo mix stage.
    #[must_use]
    pub fn with_auto_mix(mut self, config: AutoMixConfig) -> Self {
        self.auto_mix = Some(config);
        self
    }

    /// Enable bus mix analyzer and auto-correction.
    ///
    /// Applied on the bus as the absolute LAST step, AFTER dither.
    #[must_use]
    pub fn with_mix_analyzer(mut self, config: MixAnalyzerConfig) -> Self {
        self.mix_analyzer = Some(config);
        self
    }

    /// Enable per-voice allpass diffusion de-correlation.
    ///
    /// Randomizes phase response per voice to break perceptual fusion.
    #[must_use]
    pub fn with_decorrelation(mut self, config: DecorrelationConfig) -> Self {
        self.decorrelation = Some(config);
        self
    }

    /// Enable per-voice harmonic series tuning.
    ///
    /// Analyzes and adjusts individual harmonics for clean voice stacking.
    #[must_use]
    pub fn with_harmonic_tuner(mut self, config: HarmonicTunerConfig) -> Self {
        self.harmonic_tuner = Some(config);
        self
    }

    /// Enable bus equal-loudness contour compensation.
    ///
    /// Compensates for Fletcher-Munson curves at the playback level.
    #[must_use]
    pub fn with_loudness_curve(mut self, config: LoudnessCurveConfig) -> Self {
        self.loudness_curve = Some(config);
        self
    }

    /// Enable cross-voice psychoacoustic masking compensation.
    ///
    /// Boosts masked spectral content for cross-voice clarity.
    #[must_use]
    pub fn with_masking_compensator(mut self, config: MaskingCompensatorConfig) -> Self {
        self.masking_compensator = Some(config);
        self
    }

    /// Enable per-voice microphone modeling and proximity effect.
    ///
    /// Simulates different mic types and distance-based characteristics.
    #[must_use]
    pub fn with_mic_model(mut self, config: MicModelConfig) -> Self {
        self.mic_model = Some(config);
        self
    }

    /// Enable bus spectral density and fullness optimizer.
    ///
    /// Detects spectral gaps and generates complementary fill material.
    #[must_use]
    pub fn with_spectral_fill(mut self, config: SpectralFillConfig) -> Self {
        self.spectral_fill = Some(config);
        self
    }

    /// Enable per-voice vowel formant alignment.
    ///
    /// Tracks and aligns formant frequencies toward a reference voice.
    #[must_use]
    pub fn with_vowel_align(mut self, config: VowelAlignConfig) -> Self {
        self.vowel_align = Some(config);
        self
    }

    /// Enable post-pipeline output formatting.
    ///
    /// Not wired into `process()`. Use the separate
    /// [`ChorusMasterPipeline::format_output`] method after `process()`.
    #[must_use]
    pub fn with_output(mut self, config: OutputConfig) -> Self {
        self.output = Some(config);
        self
    }

    /// Enable per-voice air absorption (HF distance attenuation).
    #[must_use]
    pub fn with_air_absorption(mut self, config: AirAbsorptionConfig) -> Self {
        self.air_absorption = Some(config);
        self
    }

    /// Enable per-voice vocal presence enhancement (2-5 kHz boost).
    #[must_use]
    pub fn with_presence(mut self, config: PresenceConfig) -> Self {
        self.presence = Some(config);
        self
    }

    /// Enable per-voice micro-modulation thickener.
    #[must_use]
    pub fn with_thickener(mut self, config: ThickenerConfig) -> Self {
        self.thickener = Some(config);
        self
    }

    /// Enable bus stereo image optimizer (correlation monitoring + bass mono).
    #[must_use]
    pub fn with_stereo_optimizer(mut self, config: StereoOptimizerConfig) -> Self {
        self.stereo_optimizer = Some(config);
        self
    }

    /// Enable per-voice transient attack alignment.
    #[must_use]
    pub fn with_transient_align(mut self, config: TransientAlignConfig) -> Self {
        self.transient_align = Some(config);
        self
    }

    /// Enable bus true peak limiter with oversampled detection.
    #[must_use]
    pub fn with_true_peak_limiter(mut self, config: LimiterConfig) -> Self {
        self.true_peak_limiter = Some(config);
        self
    }

    /// Validate all sub-configs.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.n_voices == 0 || self.n_voices > 32 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: format!("must be 1..=32, got {}", self.n_voices),
            });
        }
        if let Some(ref alignment) = self.alignment {
            alignment.validate()?;
        }
        if let Some(ref eq) = self.eq {
            eq.validate()?;
        }
        if let Some(ref deesser) = self.deesser {
            deesser.validate()?;
        }
        if let Some(ref vibrato) = self.vibrato {
            vibrato.validate()?;
        }
        if let Some(ref detune) = self.detune {
            detune.validate()?;
        }
        if let Some(ref humanize) = self.humanize {
            humanize.validate()?;
        }
        if let Some(ref blend) = self.blend {
            blend.validate()?;
        }
        if let Some(ref stereo) = self.stereo {
            stereo.validate()?;
        }
        if let Some(ref dynamics) = self.dynamics {
            dynamics.validate()?;
        }
        if let Some(ref saturation) = self.saturation {
            saturation.validate()?;
        }
        if let Some(ref reverb) = self.reverb {
            reverb.validate()?;
        }
        if let Some(ref formant) = self.formant_preserve {
            formant.validate()?;
        }
        if let Some(ref breath) = self.breath {
            breath.validate()?;
        }
        if let Some(ref spatial) = self.spatial {
            spatial.validate()?;
        }
        if let Some(ref transient) = self.transient {
            transient.validate()?;
        }
        if let Some(ref bleed) = self.bleed {
            bleed.validate()?;
        }
        if let Some(ref width) = self.width {
            width.validate()?;
        }
        if let Some(ref convolution) = self.convolution {
            convolution.validate()?;
        }
        if let Some(ref pitch_correct) = self.pitch_correct {
            pitch_correct.validate()?;
        }
        if let Some(ref exciter) = self.exciter {
            exciter.validate()?;
        }
        if let Some(ref doubler) = self.doubler {
            doubler.validate()?;
        }
        if let Some(ref ducking) = self.ducking {
            ducking.validate()?;
        }
        if let Some(ref gain_staging) = self.gain_staging {
            gain_staging.validate()?;
        }
        if let Some(ref dither) = self.dither {
            dither.validate()?;
        }
        if let Some(ref gate) = self.gate {
            gate.validate()?;
        }
        if let Some(ref character) = self.character {
            character.validate()?;
        }
        if let Some(ref room) = self.room {
            room.validate()?;
        }
        if let Some(ref multiband_stereo) = self.multiband_stereo {
            multiband_stereo.validate()?;
        }
        if let Some(ref freeze) = self.freeze {
            freeze.validate()?;
        }
        if let Some(ref hrtf) = self.hrtf {
            hrtf.validate()?;
        }
        if let Some(ref auto_eq) = self.auto_eq {
            auto_eq.validate()?;
        }
        if let Some(ref loudness) = self.loudness {
            loudness.validate()?;
        }
        if let Some(ref sibilance) = self.sibilance {
            sibilance.validate()?;
        }
        if let Some(ref ensemble) = self.ensemble {
            ensemble.validate()?;
        }
        if let Some(ref warmth) = self.warmth {
            warmth.validate()?;
        }
        if let Some(ref stereo_analysis) = self.stereo_analysis {
            stereo_analysis.validate()?;
        }
        if let Some(ref formant_tune) = self.formant_tune {
            formant_tune.validate()?;
        }
        if let Some(ref micro_pitch) = self.micro_pitch {
            micro_pitch.validate()?;
        }
        if let Some(ref intonation) = self.intonation {
            intonation.validate()?;
        }
        if let Some(ref spectral_match) = self.spectral_match {
            spectral_match.validate()?;
        }
        if let Some(ref sub_bass) = self.sub_bass {
            sub_bass.validate()?;
        }
        if let Some(ref adaptive_dynamics) = self.adaptive_dynamics {
            adaptive_dynamics.validate()?;
        }
        if let Some(ref tilt) = self.tilt {
            tilt.validate()?;
        }
        if let Some(ref onset_sync) = self.onset_sync {
            onset_sync.validate()?;
        }
        if let Some(ref oversample) = self.oversample {
            oversample.validate()?;
        }
        if let Some(ref depth_staging) = self.depth_staging {
            depth_staging.validate()?;
        }
        if let Some(ref vocal_tract) = self.vocal_tract {
            vocal_tract.validate()?;
        }
        if let Some(ref shimmer) = self.shimmer {
            shimmer.validate()?;
        }
        if let Some(ref intelligibility) = self.intelligibility {
            intelligibility.validate()?;
        }
        if let Some(ref voice_alloc) = self.voice_alloc {
            voice_alloc.validate()?;
        }
        if let Some(ref dynamic_eq) = self.dynamic_eq {
            dynamic_eq.validate()?;
        }
        if let Some(ref bass_management) = self.bass_management {
            bass_management.validate()?;
        }
        if let Some(ref delay) = self.delay {
            delay.validate()?;
        }
        if let Some(ref auto_mix) = self.auto_mix {
            auto_mix.validate()?;
        }
        if let Some(ref mix_analyzer) = self.mix_analyzer {
            mix_analyzer.validate()?;
        }
        if let Some(ref decorrelation) = self.decorrelation {
            decorrelation.validate()?;
        }
        if let Some(ref harmonic_tuner) = self.harmonic_tuner {
            harmonic_tuner.validate()?;
        }
        if let Some(ref loudness_curve) = self.loudness_curve {
            loudness_curve.validate()?;
        }
        if let Some(ref masking_compensator) = self.masking_compensator {
            masking_compensator.validate()?;
        }
        if let Some(ref mic_model) = self.mic_model {
            mic_model.validate()?;
        }
        if let Some(ref spectral_fill) = self.spectral_fill {
            spectral_fill.validate()?;
        }
        if let Some(ref vowel_align) = self.vowel_align {
            vowel_align.validate()?;
        }
        if let Some(ref output) = self.output {
            output.validate()?;
        }
        if let Some(ref air_absorption) = self.air_absorption {
            air_absorption.validate()?;
        }
        if let Some(ref presence) = self.presence {
            presence.validate()?;
        }
        if let Some(ref thickener) = self.thickener {
            thickener.validate()?;
        }
        if let Some(ref stereo_optimizer) = self.stereo_optimizer {
            stereo_optimizer.validate()?;
        }
        if let Some(ref transient_align) = self.transient_align {
            transient_align.validate()?;
        }
        if let Some(ref true_peak_limiter) = self.true_peak_limiter {
            true_peak_limiter.validate()?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Stateful pipeline
// ---------------------------------------------------------------------------

/// Stateful chorus pipeline with pre-allocated filter state.
///
/// Holds mutable processor state (EQ filters, compressors, limiter) so
/// they can be reused across multiple calls without re-allocation.
/// Call [`reset`](Self::reset) between unrelated audio segments.
pub struct ChorusMasterPipeline {
    config: ChorusMasterConfig,
    mix_bus: Option<MixBusProcessor>,
    dynamics: Option<MultibandCompressor>,
    saturation_processor: Option<SaturationProcessor>,
    limiter: Option<BusLimiter>,
    breath_generator: Option<BreathGenerator>,
    /// Per-voice spatial processors, one per voice. Created from
    /// `config.spatial` + `auto_layout_spatial()` in the constructor.
    spatial_processors: Option<Vec<SpatialProcessor>>,
    /// Per-voice spatial positions (kept for diagnostics / reset).
    spatial_positions: Option<Vec<VoiceSpatialPosition>>,
    /// Stereo width enhancer (stateful: crossover filter + Haas delay line).
    stereo_widener: Option<StereoWidener>,
    /// FFT convolution reverb processor (stateful: overlap buffer + pre-delay).
    convolution_reverb: Option<ConvolutionReverb>,
    /// Per-voice harmonic exciter processors (stateful: filter state per voice).
    exciter_processors: Option<Vec<HarmonicExciter>>,
    /// Spectral ducker (stateful: envelope followers and band splitters).
    ducker: Option<SpectralDucker>,
    /// Bus gain stager (stateless per-call, but holds config-derived ceiling).
    gain_stager: Option<GainStager>,
    /// Bus dither processor (stateful: PRNG, error feedback, DC blocker).
    dither_processor: Option<DitherProcessor>,
    /// Bus multi-band stereo processor (stateful: crossover filter state).
    multiband_stereo_processor: Option<MultibandStereoProcessor>,
    /// Bus spectral freezer (stateful: FFT overlap buffer, frozen magnitudes).
    spectral_freezer: Option<SpectralFreezer>,
    /// Bus early reflections processor (stateful: delay line).
    early_reflections: Option<EarlyReflections>,
    /// HRTF binaural spatial processor (stateful: delay lines, shadow filters).
    hrtf_processor: Option<HrtfProcessor>,
    /// Per-voice auto-EQ processors (stateful: analysis + correction filters).
    auto_eq_processors: Option<Vec<AutoEqProcessor>>,
    /// Bus loudness meter/normalizer (stateful: integrated loudness history).
    loudness_meter: Option<LoudnessMeter>,
    /// Per-voice sibilance processors (stateful: band-split filters).
    sibilance_processors: Option<Vec<SibilanceProcessor>>,
    /// Bus ensemble processor (stateful: modulation LFOs, allpass diffusers).
    ensemble_processor: Option<EnsembleProcessor>,
    /// Per-voice warmth processors (stateful: bandpass + peaking EQ filters).
    warmth_processors: Option<Vec<WarmthProcessor>>,
    /// Bus stereo correlation analyzer and corrector (stateful: bass filter).
    stereo_analyzer: Option<StereoAnalyzer>,
    /// Formant resonance tuner (stateful: Hann window, LPC analysis).
    formant_tuner: Option<FormantTuner>,
    /// Per-voice micro-pitch drift processor (stateful: noise generators, delay lines).
    micro_pitch_processor: Option<MicroPitchProcessor>,
    /// Per-voice intonation tracker and corrector (stateful: pitch estimates, smoothing).
    intonation_tracker: Option<IntonationTracker>,
    /// Per-voice spectral envelope matcher (stateful: analysis buffers).
    spectral_matcher: Option<SpectralMatcher>,
    /// Bus sub-bass enhancer (stateful: filter state + phase toggle).
    sub_bass_processor: Option<SubBassEnhancer>,
    /// Bus adaptive dynamics processor (stateful: envelope follower).
    adaptive_dynamics_processor: Option<AdaptiveDynamicsProcessor>,
    /// Bus spectral tilt processor (stateful: one-pole filter).
    tilt_processor: Option<TiltProcessor>,
    /// Per-voice onset synchronizer (stateless per-call).
    onset_synchronizer: Option<OnsetSynchronizer>,
    /// Oversampler for saturation/exciter stages (stateful: FIR filter state).
    oversampler: Option<Oversampler>,
    /// Per-voice depth staging processor (stateful: LPF, delay lines).
    depth_staging_processor: Option<DepthStagingProcessor>,
    /// Per-voice vocal tract processor (stateful: cascaded formant resonators).
    vocal_tract_processor: Option<VocalTractProcessor>,
    /// Per-voice shimmer processor (stateful: allpass + comb filters).
    shimmer_processor: Option<ShimmerProcessor>,
    /// Per-voice intelligibility optimizer (stateful: band-split filters).
    intelligibility_optimizer: Option<IntelligibilityOptimizer>,
    /// Per-voice gain/pan allocator (stateful: gain smoothing state).
    voice_allocator: Option<VoiceAllocator>,
    /// Bus dynamic EQ processor (stateful: per-band envelope followers).
    dynamic_eq_processor: Option<DynamicEqProcessor>,
    /// Bus bass manager (stateful: crossover filters, sub synthesis).
    bass_manager: Option<BassManager>,
    /// Bus multi-tap delay (stateful: delay lines per tap).
    multi_tap_delay: Option<MultiTapDelay>,
    /// Pre-mix auto-mixer (stateful: gain smoothing, level tracking).
    auto_mixer: Option<AutoMixer>,
    /// Bus mix analyzer (stateful: level meters, correlation tracking).
    mix_analyzer_processor: Option<MixAnalyzerProcessor>,
    /// Per-voice allpass diffusion de-correlation processor (stateful: allpass chains).
    decorrelation_processor: Option<DecorrelationProcessor>,
    /// Per-voice harmonic series tuner (stateful: FFT window, analysis buffers).
    harmonic_tuner_processor: Option<HarmonicTunerProcessor>,
    /// Bus equal-loudness contour compensation (stateful: biquad filter bank).
    loudness_curve_processor: Option<LoudnessCurveProcessor>,
    /// Cross-voice psychoacoustic masking compensator (stateful: analysis + EQ filters).
    masking_compensator_processor: Option<MaskingCompensator>,
    /// Per-voice microphone model (stateful: per-voice filter chains).
    mic_model_processor: Option<MicModelProcessor>,
    /// Bus spectral density and fullness optimizer (stateful: oscillator phases, noise).
    spectral_fill_processor: Option<SpectralFillProcessor>,
    /// Per-voice vowel formant aligner (stateful: LPC analysis, formant tracking).
    vowel_aligner: Option<VowelAligner>,
    /// Post-pipeline output formatter (stateful: sample rate converter).
    output_formatter: Option<OutputFormatter>,
    /// Per-voice air absorption processor (stateful: cascaded one-pole LPF chains).
    air_absorption_processor: Option<AirAbsorptionProcessor>,
    /// Per-voice presence enhancer (stateful: sibilance detector, envelope, EQ filters).
    presence_processors: Option<Vec<PresenceProcessor>>,
    /// Per-voice thickener (stateful: LFO state, modulated delay lines).
    thickener_processor: Option<ThickenerProcessor>,
    /// Bus stereo image optimizer (stateful: correlation tracker, bass filter).
    stereo_optimizer_processor: Option<StereoOptimizer>,
    /// Per-voice transient attack aligner (stateful: lookahead buffers, onset detectors).
    transient_aligner: Option<TransientAligner>,
    /// Bus true peak limiter (stateful: lookahead delay, gain smoothing).
    true_peak_limiter_processor: Option<LimiterProcessor>,
}

impl ChorusMasterPipeline {
    /// Create a new pipeline from the given config.
    ///
    /// Pre-allocates all filter state.
    pub fn new(config: ChorusMasterConfig) -> Result<Self, KokoroError> {
        config.validate()?;

        let mix_bus = if config.eq.is_some() || config.deesser.is_some() {
            let bus_config = MixBusConfig {
                voice_eq: config.eq.clone(),
                deesser: config.deesser.clone(),
                bus_eq: None,
                deesser_enabled: config.deesser.is_some(),
            };
            Some(MixBusProcessor::new(config.n_voices, &bus_config)?)
        } else {
            None
        };

        let dynamics = config
            .dynamics
            .as_ref()
            .map(MultibandCompressor::new)
            .transpose()?;

        let saturation_processor = config
            .saturation
            .map(SaturationProcessor::new_kokoro)
            .transpose()?;

        let limiter = if config.limiter_enabled {
            Some(BusLimiter::new())
        } else {
            None
        };

        let breath_generator = config
            .breath
            .as_ref()
            .map(|bc| BreathGenerator::new(bc, config.n_voices))
            .transpose()?;

        // Pre-allocate per-voice spatial processors if spatial is enabled.
        let (spatial_processors, spatial_positions) = if let Some(ref spatial_cfg) = config.spatial
        {
            let positions = auto_layout_spatial(config.n_voices, spatial_cfg)?;
            let processors: Result<Vec<SpatialProcessor>, _> = positions
                .iter()
                .map(|pos| SpatialProcessor::new(spatial_cfg, pos))
                .collect();
            (Some(processors?), Some(positions))
        } else {
            (None, None)
        };

        // Pre-allocate stereo widener if width is enabled.
        let stereo_widener = config.width.as_ref().map(StereoWidener::new).transpose()?;

        // Pre-allocate convolution reverb if convolution is enabled.
        let convolution_reverb = if let Some(ref conv_cfg) = config.convolution {
            let mut reverb = ConvolutionReverb::new(conv_cfg)?;
            // Load a default medium hall IR when convolution is enabled.
            let ir = generate_synthetic_ir(SyntheticRoom::MediumHall);
            reverb.load_ir(&ir, conv_cfg);
            Some(reverb)
        } else {
            None
        };

        // Pre-allocate per-voice exciter processors if exciter is enabled.
        let exciter_processors = if let Some(ref exciter_cfg) = config.exciter {
            let sr = KOKORO_SAMPLE_RATE as f32;
            let processors: Result<Vec<HarmonicExciter>, _> = (0..config.n_voices)
                .map(|_| HarmonicExciter::new(exciter_cfg, sr))
                .collect();
            Some(processors?)
        } else {
            None
        };

        // Pre-allocate spectral ducker if ducking is enabled.
        let ducker = if let Some(ref ducking_cfg) = config.ducking {
            Some(SpectralDucker::new(ducking_cfg, KOKORO_SAMPLE_RATE as f32)?)
        } else {
            None
        };

        // Pre-allocate gain stager if gain staging is enabled.
        let gain_stager = config
            .gain_staging
            .as_ref()
            .map(GainStager::new)
            .transpose()?;

        // Pre-allocate dither processor if dither is enabled.
        let dither_processor = config
            .dither
            .as_ref()
            .map(DitherProcessor::new)
            .transpose()?;

        // Pre-allocate multi-band stereo processor if configured.
        let multiband_stereo_processor = config
            .multiband_stereo
            .as_ref()
            .map(|cfg| MultibandStereoProcessor::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate spectral freezer if freeze is enabled.
        let spectral_freezer = config
            .freeze
            .as_ref()
            .map(|cfg| SpectralFreezer::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate early reflections processor if room is enabled.
        let early_reflections = config
            .room
            .as_ref()
            .map(|cfg| EarlyReflections::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate HRTF processor if hrtf is enabled.
        let hrtf_processor = config
            .hrtf
            .as_ref()
            .map(|cfg| HrtfProcessor::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate per-voice auto-EQ processors if auto_eq is enabled.
        let auto_eq_processors = if let Some(ref auto_eq_cfg) = config.auto_eq {
            let sr = KOKORO_SAMPLE_RATE as f32;
            let processors: Result<Vec<AutoEqProcessor>, _> = (0..config.n_voices)
                .map(|_| AutoEqProcessor::new(auto_eq_cfg, sr))
                .collect();
            Some(processors?)
        } else {
            None
        };

        // Pre-allocate loudness meter if loudness is enabled.
        let loudness_meter = config
            .loudness
            .as_ref()
            .map(|cfg| LoudnessMeter::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate per-voice sibilance processors if sibilance is enabled.
        let sibilance_processors = if let Some(ref sib_cfg) = config.sibilance {
            let processors: Result<Vec<SibilanceProcessor>, _> = (0..config.n_voices)
                .map(|_| SibilanceProcessor::new_kokoro(*sib_cfg))
                .collect();
            Some(processors?)
        } else {
            None
        };

        // Pre-allocate ensemble processor if ensemble is enabled.
        let ensemble_processor = config
            .ensemble
            .as_ref()
            .map(|cfg| EnsembleProcessor::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate per-voice warmth processors if warmth is enabled.
        let warmth_processors = if let Some(ref warmth_cfg) = config.warmth {
            let processors: Result<Vec<WarmthProcessor>, _> = (0..config.n_voices)
                .map(|_| WarmthProcessor::new_kokoro(*warmth_cfg))
                .collect();
            Some(processors?)
        } else {
            None
        };

        // Pre-allocate stereo analyzer if stereo_analysis is enabled.
        let stereo_analyzer = config
            .stereo_analysis
            .as_ref()
            .map(|c| StereoAnalyzer::new(c.clone()))
            .transpose()?;

        // Pre-allocate formant tuner if formant_tune is enabled.
        let formant_tuner = config
            .formant_tune
            .as_ref()
            .map(|cfg| FormantTuner::new(cfg.clone()))
            .transpose()?;

        // Pre-allocate micro-pitch processor if micro_pitch is enabled.
        let micro_pitch_processor = config
            .micro_pitch
            .as_ref()
            .map(|cfg| MicroPitchProcessor::new(cfg, config.n_voices, KOKORO_SAMPLE_RATE as u32))
            .transpose()?;

        // Pre-allocate intonation tracker if intonation is enabled.
        let intonation_tracker = config
            .intonation
            .as_ref()
            .map(|cfg| IntonationTracker::new(cfg, config.n_voices, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate spectral matcher if spectral_match is enabled.
        let spectral_matcher = config
            .spectral_match
            .as_ref()
            .map(SpectralMatcher::new)
            .transpose()?;

        // Pre-allocate sub-bass processor if sub_bass is enabled.
        let sub_bass_processor = config
            .sub_bass
            .as_ref()
            .map(|cfg| SubBassEnhancer::new(*cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate adaptive dynamics processor if adaptive_dynamics is enabled.
        let adaptive_dynamics_processor = config
            .adaptive_dynamics
            .as_ref()
            .map(AdaptiveDynamicsProcessor::new)
            .transpose()?;

        // Pre-allocate tilt processor if tilt is enabled.
        let tilt_processor = config
            .tilt
            .as_ref()
            .map(|cfg| TiltProcessor::new(*cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate onset synchronizer if onset_sync is enabled.
        let onset_synchronizer = config
            .onset_sync
            .as_ref()
            .map(|cfg| OnsetSynchronizer::new(cfg.clone()))
            .transpose()?;

        // Pre-allocate oversampler if oversample is enabled.
        let oversampler = config
            .oversample
            .as_ref()
            .map(Oversampler::new)
            .transpose()?;

        // Pre-allocate depth staging processor if depth_staging is enabled.
        let depth_staging_processor = config
            .depth_staging
            .as_ref()
            .map(|cfg| DepthStagingProcessor::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate vocal tract processor if vocal_tract is enabled.
        let vocal_tract_processor = config
            .vocal_tract
            .as_ref()
            .map(|cfg| VocalTractProcessor::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate shimmer processor if shimmer is enabled.
        let shimmer_processor = config
            .shimmer
            .as_ref()
            .map(|cfg| ShimmerProcessor::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate intelligibility optimizer if intelligibility is enabled.
        let intelligibility_optimizer = config
            .intelligibility
            .as_ref()
            .map(|cfg| IntelligibilityOptimizer::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate voice allocator if voice_alloc is enabled.
        let voice_allocator = config
            .voice_alloc
            .as_ref()
            .map(VoiceAllocator::new)
            .transpose()?;

        // Pre-allocate dynamic EQ processor if dynamic_eq is enabled.
        let dynamic_eq_processor = config
            .dynamic_eq
            .as_ref()
            .map(|cfg| DynamicEqProcessor::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate bass manager if bass_management is enabled.
        let bass_manager = if let Some(ref bm_cfg) = config.bass_management {
            Some(BassManager::new(*bm_cfg, KOKORO_SAMPLE_RATE as f32)?)
        } else {
            None
        };

        // Pre-allocate multi-tap delay if delay is enabled.
        let multi_tap_delay = config.delay.as_ref().map(MultiTapDelay::new).transpose()?;

        // Pre-allocate auto-mixer if auto_mix is enabled.
        let auto_mixer = config.auto_mix.as_ref().map(AutoMixer::new).transpose()?;

        // Pre-allocate mix analyzer if mix_analyzer is enabled.
        let mix_analyzer_processor = config
            .mix_analyzer
            .as_ref()
            .map(|cfg| MixAnalyzerProcessor::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate decorrelation processor if decorrelation is enabled.
        let decorrelation_processor = config
            .decorrelation
            .as_ref()
            .map(|cfg| DecorrelationProcessor::new(cfg, config.n_voices, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate harmonic tuner processor if harmonic_tuner is enabled.
        let harmonic_tuner_processor = config
            .harmonic_tuner
            .clone()
            .map(HarmonicTunerProcessor::new)
            .transpose()?;

        // Pre-allocate loudness curve processor if loudness_curve is enabled.
        let loudness_curve_processor = config
            .loudness_curve
            .as_ref()
            .map(|cfg| LoudnessCurveProcessor::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate masking compensator if masking_compensator is enabled.
        let masking_compensator_processor = config
            .masking_compensator
            .as_ref()
            .map(MaskingCompensator::new)
            .transpose()?;

        // Pre-allocate mic model processor if mic_model is enabled.
        let mic_model_processor = if let Some(ref mic_cfg) = config.mic_model {
            Some(MicModelProcessor::new(*mic_cfg, config.n_voices)?)
        } else {
            None
        };

        // Pre-allocate spectral fill processor if spectral_fill is enabled.
        let spectral_fill_processor = config
            .spectral_fill
            .clone()
            .map(SpectralFillProcessor::new)
            .transpose()?;

        // Pre-allocate vowel aligner if vowel_align is enabled.
        let vowel_aligner = config
            .vowel_align
            .clone()
            .map(VowelAligner::new)
            .transpose()?;

        // Pre-allocate output formatter if output is enabled.
        let output_formatter = config
            .output
            .as_ref()
            .map(OutputFormatter::new)
            .transpose()?;

        // Pre-allocate air absorption processor if air_absorption is enabled.
        let air_absorption_processor = config
            .air_absorption
            .as_ref()
            .map(|cfg| AirAbsorptionProcessor::new(cfg, config.n_voices))
            .transpose()?;

        // Pre-allocate per-voice presence processors if presence is enabled.
        let presence_processors = if let Some(ref pres_cfg) = config.presence {
            let processors: Result<Vec<PresenceProcessor>, _> = (0..config.n_voices)
                .map(|_| PresenceProcessor::new_kokoro(pres_cfg))
                .collect();
            Some(processors?)
        } else {
            None
        };

        // Pre-allocate thickener processor if thickener is enabled.
        let thickener_processor = config
            .thickener
            .as_ref()
            .map(|cfg| ThickenerProcessor::new(cfg, config.n_voices, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate stereo optimizer if stereo_optimizer is enabled.
        let stereo_optimizer_processor = config
            .stereo_optimizer
            .as_ref()
            .map(|cfg| StereoOptimizer::new(cfg, KOKORO_SAMPLE_RATE as u32))
            .transpose()?;

        // Pre-allocate transient aligner if transient_align is enabled.
        let transient_aligner = config
            .transient_align
            .as_ref()
            .map(|cfg| TransientAligner::new(cfg, config.n_voices, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        // Pre-allocate true peak limiter if true_peak_limiter is enabled.
        let true_peak_limiter_processor = config
            .true_peak_limiter
            .as_ref()
            .map(|cfg| LimiterProcessor::new(cfg, KOKORO_SAMPLE_RATE as f32))
            .transpose()?;

        Ok(Self {
            config,
            mix_bus,
            dynamics,
            saturation_processor,
            limiter,
            breath_generator,
            spatial_processors,
            spatial_positions,
            stereo_widener,
            convolution_reverb,
            exciter_processors,
            ducker,
            gain_stager,
            dither_processor,
            multiband_stereo_processor,
            spectral_freezer,
            early_reflections,
            hrtf_processor,
            auto_eq_processors,
            loudness_meter,
            sibilance_processors,
            ensemble_processor,
            warmth_processors,
            stereo_analyzer,
            formant_tuner,
            micro_pitch_processor,
            intonation_tracker,
            spectral_matcher,
            sub_bass_processor,
            adaptive_dynamics_processor,
            tilt_processor,
            onset_synchronizer,
            oversampler,
            depth_staging_processor,
            vocal_tract_processor,
            shimmer_processor,
            intelligibility_optimizer,
            voice_allocator,
            dynamic_eq_processor,
            bass_manager,
            multi_tap_delay,
            auto_mixer,
            mix_analyzer_processor,
            decorrelation_processor,
            harmonic_tuner_processor,
            loudness_curve_processor,
            masking_compensator_processor,
            mic_model_processor,
            spectral_fill_processor,
            vowel_aligner,
            output_formatter,
            air_absorption_processor,
            presence_processors,
            thickener_processor,
            stereo_optimizer_processor,
            transient_aligner,
            true_peak_limiter_processor,
        })
    }

    /// Process voices through the full pipeline, returning stereo (L, R).
    ///
    /// The canonical stage ordering is defined by
    /// [`ChorusPipelineStage`](crate::kokoro_chorus_stage::ChorusPipelineStage).
    /// Stages that are disabled (`None` in the config) are skipped at zero cost.
    ///
    /// ```text
    /// Per-voice (individual voice buffers):
    ///   [0] Alignment     — cross-correlation time-alignment
    ///   [1] Detune        — allpass fractional-delay detuning
    ///   [2] FormantPitch  — formant-preserving pitch shift (PSOLA)
    ///   [3] Vibrato       — LFO-based F0 pitch modulation
    ///   [4] Breath        — breath insertion at pause regions
    ///   [5] Humanize      — micro-timing, envelope, breathing
    ///   [6] EnsembleBlend — PSOLA ensemble blending
    ///   [7] PerVoiceEq    — parametric biquad EQ
    ///   [8] DeEss         — sibilance reduction (de-esser)
    ///
    /// Mix (voice combination):
    ///   [9]  StereoMix    — constant-power stereo pan-law
    ///   [10] SpatialDepth — 3D spatial positioning
    ///
    /// Bus (stereo master):
    ///   [11] BusEq        — bus-level parametric EQ
    ///   [12] Dynamics     — multiband compression
    ///   [13] Saturation   — harmonic warmth (tape/tube/console)
    ///   [14] Reverb       — Schroeder reverb
    ///   [15] Limiter      — peak limiter to -0.1 dBFS (always last)
    /// ```
    ///
    /// See [`crate::kokoro_chorus_stage::default_order`] for the full
    /// `Vec<ChorusPipelineStage>` and
    /// [`crate::kokoro_chorus_stage::validate_order`] to check custom orderings.
    ///
    /// Voices are cloned internally to avoid mutating the input.
    pub fn process(&mut self, voices: &[Vec<f32>]) -> Result<(Vec<f32>, Vec<f32>), KokoroError> {
        if voices.len() != self.config.n_voices {
            return Err(KokoroError::InvalidInput(format!(
                "expected {} voices, got {}",
                self.config.n_voices,
                voices.len(),
            )));
        }

        // Clone to avoid mutating caller's data.
        let mut work = voices.to_vec();

        // -- Per-voice processing --

        let sr = KOKORO_SAMPLE_RATE as u32;

        // -1. Gate (noise gate per-voice, FIRST before everything).
        // Removes noise floor from silent/quiet sections so stacked voices
        // don't amplify noise proportionally to voice count.
        if let Some(ref gate_cfg) = self.config.gate {
            apply_noise_gate(&mut work, gate_cfg, KOKORO_SAMPLE_RATE as f32)?;
        }

        // 0. Alignment (cross-correlation temporal sync).
        // Applied FIRST: timing must be corrected before detuning, EQ, or
        // vibrato to avoid misaligning the corrections themselves.
        if let Some(ref alignment_cfg) = self.config.alignment {
            let voice_slices: Vec<Vec<f32>> = work.clone();
            let aligned = align_voices(&voice_slices, alignment_cfg)?;
            for (i, v) in aligned.into_iter().enumerate() {
                work[i] = v;
            }
        }

        // 0a2. Onset synchronization (per-voice, after alignment, before formant_tune).
        // Micro-shifts voices so transient attacks align with the reference voice.
        if let Some(ref mut onset_sync) = self.onset_synchronizer {
            onset_sync.synchronize(&mut work)?;
        }

        // 0a2b. Transient alignment (per-voice, after onset_sync, before depth_staging).
        // Micro-shifts transient attacks so consonant onsets align across voices
        // for tight ensemble cohesion without affecting sustain/release.
        if let Some(ref mut aligner) = self.transient_aligner {
            aligner.process_voices(&mut work)?;
        }

        // 0a3. Depth staging (per-voice, after alignment, before character).
        // Positions voices in a front-to-back depth field with distance
        // attenuation, air absorption LPF, pre-delay, and early reflections.
        if let Some(ref mut depth_staging) = self.depth_staging_processor {
            depth_staging.process_voices(&mut work);
        }

        // 0a4. Air absorption (per-voice, after depth staging, before character).
        // Models frequency-dependent HF attenuation over distance: distant voices
        // sound darker and less present, providing a natural depth cue.
        if let Some(ref mut air_abs) = self.air_absorption_processor {
            air_abs.process_voices(&mut work);
        }

        // 0b. Character variation (per-voice timbre, after alignment, before vibrato).
        // Assigns deterministic vocal tract scaling, breathiness, and brightness
        // differences to each voice for distinct singer character.
        if let Some(ref character_cfg) = self.config.character {
            apply_character_variation(&mut work, character_cfg, KOKORO_SAMPLE_RATE as f32)?;
        }

        // 0b2. Vocal tract (per-voice, after character, before formant_tune).
        // Models cascaded formant resonators for unique virtual singer body.
        if let Some(ref mut vocal_tract) = self.vocal_tract_processor {
            vocal_tract.process_voices(&mut work)?;
        }

        // 0c. Formant tuning (per-voice, after character, before vibrato).
        // Detects and reshapes formant frequencies to create timbre variation.
        if let Some(ref mut tuner) = self.formant_tuner {
            for (i, voice) in work.iter_mut().enumerate() {
                tuner.process_voice(voice, i);
            }
        }

        // 0c2. Vowel alignment (per-voice, after formant_tune, before harmonic_tuner).
        // Tracks and aligns formant frequencies toward a reference voice
        // for better vowel blending in the ensemble.
        if let Some(ref mut aligner) = self.vowel_aligner {
            let _tracks = aligner.process_voices(&mut work);
        }

        // 0c3. Harmonic tuner (per-voice, after vowel_align, before vibrato).
        // Analyzes harmonic series and adjusts for clean voice stacking.
        if let Some(ref mut harm_tuner) = self.harmonic_tuner_processor {
            harm_tuner.process_voices(&mut work);
        }

        // 1. Vibrato (LFO-based F0 pitch modulation).
        if let Some(ref vibrato) = self.config.vibrato {
            apply_vibrato(&mut work, vibrato, sr)?;
        }

        // 1a. Intonation (per-voice, after vibrato, before pitch_correct).
        // Gently pulls voices toward a shared reference pitch to prevent
        // inter-voice drift while preserving natural micro-variation.
        if let Some(ref mut tracker) = self.intonation_tracker {
            tracker.process_voices(&mut work)?;
        }

        // 1a2. Spectral match (per-voice, after intonation, before detuning).
        // Aligns spectral envelopes toward voice 0 so detuned voices retain
        // the same timbral color.
        if let Some(ref mut matcher) = self.spectral_matcher {
            matcher.match_to_reference(&mut work)?;
        }

        // 1b. Pitch correction (per-voice, after intonation, before transient).
        // Snaps detected pitch toward the nearest scale note.
        if let Some(ref pitch_correct_cfg) = self.config.pitch_correct {
            apply_pitch_correction(&mut work, pitch_correct_cfg, KOKORO_SAMPLE_RATE as f32)?;
        }

        // 1c. Transient shaping (per-voice, after pitch correct, before detuning).
        // Shapes consonant attacks and vowel sustain before tonal/pitch processing.
        if let Some(ref transient_cfg) = self.config.transient {
            apply_transient_shaping(&mut work, transient_cfg, KOKORO_SAMPLE_RATE as f32)?;
        }

        // 2. Detuning: formant-preserving for large shifts, fast path for small.
        if let Some(ref detune) = self.config.detune {
            if let Some(ref formant_cfg) = self.config.formant_preserve {
                // Formant-preserving path: per-voice pitch shift using the
                // cents assigned by the detune config. Voices with >10 cents
                // use PSOLA formant preservation; <=10 cents use fast simple
                // pitch shift. Voice 0 is always the anchor (0 cents).
                let voice_cents = detune.voice_cents(work.len());
                for (i, cents) in voice_cents.iter().enumerate() {
                    if cents.abs() < 1e-6 {
                        continue; // anchor voice or zero offset
                    }
                    let rate = cents_to_rate(*cents);
                    if cents.abs() > 10.0 {
                        // Large detuning: full formant-preserving PSOLA.
                        work[i] = shift_pitch_preserve_formant(&work[i], rate, Some(formant_cfg))?;
                    } else {
                        // Small detuning: fast linear-interpolation resampling.
                        work[i] = simple_pitch_shift(&work[i], rate);
                    }
                }
            } else {
                // Default path: allpass fractional-delay resampling.
                apply_detune(&mut work, detune, sr)?;
            }
        }

        // 2a. Micro-pitch drift (per-voice, after detuning, before EQ).
        // Adds slow 1/f-like pitch wandering for natural ensemble shimmer.
        if let Some(ref mut micro) = self.micro_pitch_processor {
            micro.process_voices(&mut work)?;
        }

        // 2. EQ + de-essing via MixBusProcessor.
        if let Some(ref mut bus) = self.mix_bus {
            for i in 0..work.len() {
                bus.process_voice(i, &mut work[i]);
            }
        }

        // 2b. Auto-EQ (per-voice, after EQ/de-ess, before exciter).
        // Analyzes each voice's spectrum and applies corrective filtering
        // toward a target curve.
        if let Some(ref mut auto_eq_procs) = self.auto_eq_processors {
            let sr_f32 = KOKORO_SAMPLE_RATE as f32;
            for (i, voice) in work.iter_mut().enumerate() {
                if let Some(aeq) = auto_eq_procs.get_mut(i) {
                    aeq.analyze_and_correct(voice, sr_f32);
                }
            }
        }

        // 2c. Exciter (per-voice, after auto-EQ, enhances harmonics).
        if let Some(ref mut exciter_procs) = self.exciter_processors {
            for (i, voice) in work.iter_mut().enumerate() {
                if let Some(exciter) = exciter_procs.get_mut(i) {
                    exciter.process(voice);
                }
            }
        }

        // 2c2. Warmth (per-voice, after exciter, before humanize).
        // Adds analog-style body saturation and presence clarity.
        if let Some(ref mut warmth_procs) = self.warmth_processors {
            for (i, voice) in work.iter_mut().enumerate() {
                if let Some(wp) = warmth_procs.get_mut(i) {
                    wp.process_voice(voice);
                }
            }
        }

        // 2c3. Shimmer (per-voice, after warmth, before presence).
        // Adds airy high-frequency harmonics and subtle brightness.
        if let Some(ref mut shimmer) = self.shimmer_processor {
            for voice in work.iter_mut() {
                shimmer.process_voice(voice);
            }
        }

        // 2c4. Presence (per-voice, after shimmer, before sibilance).
        // Dynamic 2-5 kHz presence boost with sibilance-aware gain reduction
        // and optional air band shelf for sparkle.
        if let Some(ref mut pres_procs) = self.presence_processors {
            for (i, voice) in work.iter_mut().enumerate() {
                if let Some(pres) = pres_procs.get_mut(i) {
                    pres.process(voice);
                }
            }
        }

        // 2d. Sibilance (per-voice, after presence, before humanize).
        // Precise frequency-domain sibilance control with cross-voice alignment.
        if let Some(ref mut sib_procs) = self.sibilance_processors {
            // Cross-voice sibilant alignment first (modifies in-place).
            if let Some(ref sib_cfg) = self.config.sibilance {
                align_sibilants(&mut work, sib_cfg, KOKORO_SAMPLE_RATE as f32);
            }
            // Per-voice sibilance processing.
            for (i, voice) in work.iter_mut().enumerate() {
                if let Some(sib) = sib_procs.get_mut(i) {
                    sib.process_voice(voice);
                }
            }
        }

        // 2e. Intelligibility (per-voice, after sibilance, before humanize).
        // Protects critical speech frequency bands from masking.
        if let Some(ref mut intelli) = self.intelligibility_optimizer {
            for voice in work.iter_mut() {
                intelli.process_voice(voice);
            }
        }

        // 3. Humanization (breathing, micro-timing, envelope).
        if let Some(ref humanize) = self.config.humanize {
            for (i, voice) in work.iter_mut().enumerate() {
                apply_humanize(voice, humanize, i, sr)?;
            }
        }

        // 3a. Thickener (per-voice, after humanize, before doubler).
        // Adds subtle LFO-driven pitch, timing, and amplitude micro-modulation
        // to each voice for a thicker, lusher chorus without adding new voices.
        if let Some(ref mut thick) = self.thickener_processor {
            thick.process_voices(&mut work);
        }

        // 3b. Doubler / ADT (per-voice, after thickener, creates doubled copies).
        if let Some(ref doubler_cfg) = self.config.doubler {
            apply_doubler_per_voice(&mut work, doubler_cfg, KOKORO_SAMPLE_RATE as f32)?;
        }

        // 3c. Breath insertion (per-voice, after humanize + doubler, before mix).
        // Detect pauses from voice 0 as the reference (all voices share the
        // same text and similar timing). Each voice gets staggered breath
        // placement via the BreathGenerator's per-voice PRNG offset.
        if let (Some(ref breath_config), Some(ref mut bgen)) =
            (&self.config.breath, &mut self.breath_generator)
        {
            if let Some(reference_voice) = work.first() {
                let pauses = detect_pauses(reference_voice, breath_config);
                if !pauses.is_empty() {
                    insert_breath_sounds(&mut work, &pauses, bgen, breath_config)?;
                }
            }
        }

        // 3c. Voice bleed (per-voice crosstalk, last per-voice step).
        // Applied after humanize and breath, before stereo/spatial mix.
        if let Some(ref bleed_cfg) = self.config.bleed {
            apply_voice_bleed(&mut work, bleed_cfg, sr)?;
        }

        // 3e. Ducking (per-voice, after bleed, lead voice prominence).
        if let (Some(ref ducking_cfg), Some(ref mut ducker)) =
            (&self.config.ducking, &mut self.ducker)
        {
            ducker.process(&mut work, ducking_cfg)?;
        }

        // 3f. Voice allocation (per-voice, after ducking, before mix).
        // Applies per-voice gain and pan adjustments for ensemble balance.
        // Must allocate voice slots before apply_gains, since the allocator
        // starts with all slots inactive (gain=0) and zeroes inactive buffers.
        // Fix for #4337: without allocation, apply_gains zeroes all voices.
        if let Some(ref mut allocator) = self.voice_allocator {
            // Ensure each voice has an active slot so apply_gains doesn't
            // zero the buffers. Active count may be 0 on first call or after
            // a reset().
            if allocator.active_count() < work.len() {
                allocator.reset();
                for i in 0..work.len() {
                    // Priority decreases with voice index (first voice = lead).
                    let priority = 1.0 - (i as f32 / work.len().max(1) as f32);
                    allocator.allocate_voice(priority);
                }
            }
            allocator.apply_gains(&mut work);
        }

        // 3g. Auto-mix (pre-mix, before blend/stereo, analyzes and adjusts levels).
        // Spectral balance auto-mixer for per-voice gain optimization.
        if let Some(ref mut auto_mixer) = self.auto_mixer {
            let _analysis = auto_mixer.analyze_and_adjust(&mut work);
        }

        // 3h. Decorrelation (per-voice, after auto-mix, before masking compensation).
        // Randomizes phase response per voice via cascaded allpass diffusion
        // so voices sound perceptually distinct rather than comb-filtered clones.
        if let Some(ref mut decorr) = self.decorrelation_processor {
            decorr.process_voices(&mut work);
        }

        // 3i. Masking compensation (cross-voice, after decorrelation, before mic model).
        // Detects simultaneous masking interactions and boosts masked spectral
        // components to maintain cross-voice clarity.
        if let Some(ref mut masking) = self.masking_compensator_processor {
            let _analyses = masking.process_voices(&mut work);
        }

        // 3j. Mic model (per-voice, after masking compensation, before mix).
        // Simulates different microphone types and proximity effect for
        // realistic depth variety across voices.
        if let Some(ref mut mic) = self.mic_model_processor {
            mic.process_voices(&mut work);
        }

        // -- Mix stage --

        // 4. Ensemble blending (PSOLA formant-preserving pitch correction).
        if let Some(ref blend) = self.config.blend {
            crate::kokoro_chorus_blend::blend_voices(&mut work, blend, sr)?;
        }

        // 5. Stereo imaging (HRTF, spatial, or default panning).
        // Priority: HRTF > spatial > stereo config > mono fallback.
        // When HRTF is enabled, binaural processing replaces all other
        // stereo paths (ITD + ILD + head shadow for immersive output).
        // When spatial is enabled, each voice is processed through
        // distance attenuation + ILD stereo panning, then summed.
        // This replaces the default constant-power pan law.
        let (mut left, mut right) = if let Some(ref mut hrtf) = self.hrtf_processor {
            hrtf.process_voices(&work)?
        } else if let Some(ref mut processors) = self.spatial_processors {
            let max_len = work.iter().map(Vec::len).max().unwrap_or(0);
            let mut left = vec![0.0f32; max_len];
            let mut right = vec![0.0f32; max_len];
            for (voice, proc) in work.iter().zip(processors.iter_mut()) {
                let (vl, vr) = proc.process(voice);
                for (i, (&sl, &sr)) in vl.iter().zip(vr.iter()).enumerate() {
                    left[i] += sl;
                    right[i] += sr;
                }
            }
            (left, right)
        } else if let Some(ref stereo) = self.config.stereo {
            apply_stereo_mix(&work, stereo)?
        } else {
            // Mono fallback: sum voices with equal gain.
            let n = work.len() as f32;
            let gain = if n > 0.0 { 1.0 / n } else { 1.0 };
            let max_len = work.iter().map(Vec::len).max().unwrap_or(0);
            let mut mono = vec![0.0f32; max_len];
            for voice in &work {
                for (i, &s) in voice.iter().enumerate() {
                    mono[i] += s * gain;
                }
            }
            (mono.clone(), mono)
        };

        // -- Bus processing --

        // 5b. Stereo width enhancement (after stereo/spatial mix, before dynamics).
        // Enhances the stereo image with mid/side widening, bass mono
        // filtering, and optional Haas effect delay.
        if let Some(ref mut widener) = self.stereo_widener {
            widener.process(&mut left, &mut right);
        }

        // 5c. Multi-band stereo (after stereo width, before dynamics).
        // Frequency-dependent stereo imaging: narrow bass, moderate mids,
        // wide highs.
        if let Some(ref mut mb_stereo) = self.multiband_stereo_processor {
            mb_stereo.process(&mut left, &mut right);
        }

        // 5d. Stereo analysis (bus, after width/multiband, before stereo optimizer).
        // Monitors and optionally corrects phase coherence for mono compatibility.
        if let Some(ref mut analyzer) = self.stereo_analyzer {
            analyzer.process(&mut left, &mut right);
        }

        // 5e. Stereo optimizer (bus, after stereo analysis, before dynamics).
        // Monitors L/R correlation and automatically narrows the stereo image
        // when correlation drops below threshold to prevent mono cancellation.
        // Also forces bass frequencies to mono for playback compatibility.
        if let Some(ref mut optimizer) = self.stereo_optimizer_processor {
            optimizer.process_stereo(&mut left, &mut right);
        }

        // 6. Dynamics compression.
        // When adaptive dynamics is configured AND regular dynamics is also
        // set, adaptive dynamics replaces the regular compressor. When only
        // regular dynamics is set, use regular. When only adaptive is set,
        // use adaptive.
        if let Some(ref mut adaptive) = self.adaptive_dynamics_processor {
            // Adaptive dynamics replaces regular when both are present,
            // or is used alone.
            adaptive.process(&mut left);
            adaptive.reset();
            adaptive.process(&mut right);
        } else if let Some(ref mut comp) = self.dynamics {
            comp.process(&mut left);
            comp.reset(); // Reset state between L/R.
            comp.process(&mut right);
        }

        // 6a. Dynamic EQ (bus, after dynamics, before sub_bass).
        // Frequency-dependent dynamic EQ with per-band envelope followers.
        // Mono bus effect: must reset between L/R channel processing.
        if let Some(ref mut deq) = self.dynamic_eq_processor {
            deq.process(&mut left);
            deq.reset();
            deq.process(&mut right);
        }

        // 6a2. Sub-bass enhancement (bus, after dynamic_eq, before bass_management).
        // Generates a sub-harmonic of the low-frequency content for depth.
        if let Some(ref mut sub_bass) = self.sub_bass_processor {
            sub_bass.process(&mut left, &mut right);
        }

        // 6a3. Bass management (bus, after sub_bass, before saturation).
        // Psychoacoustic bass management with crossover and sub synthesis.
        if let Some(ref mut bm) = self.bass_manager {
            bm.process(&mut left, &mut right);
        }

        // 6b. Saturation (harmonic warmth). Applied after dynamics so the
        // compressor tames peaks before the waveshaper, and before reverb
        // so saturation harmonics get spatial coloring from the reverb tail.
        // When oversampling is configured, saturation runs at 2x/4x rate
        // for anti-aliased waveshaping.
        if let Some(ref mut sat) = self.saturation_processor {
            if let Some(ref mut os) = self.oversampler {
                os.process_oversampled(&mut left, |buf| sat.process(buf));
                sat.reset();
                os.reset();
                os.process_oversampled(&mut right, |buf| sat.process(buf));
            } else {
                sat.process(&mut left);
                sat.reset(); // Reset filter state between L/R.
                sat.process(&mut right);
            }
        }

        // 6c. Room (image-source early reflections, BEFORE Schroeder reverb).
        // Early reflections provide spatial cues about room size/shape that
        // complement the late reverb tail. Process a mono downmix through the
        // room simulation and add the stereo reflections to the bus.
        if let Some(ref mut er) = self.early_reflections {
            // Create mono downmix for early reflections input.
            let len = left.len().min(right.len());
            let mono: Vec<f32> = (0..len).map(|i| (left[i] + right[i]) * 0.5).collect();
            let (er_l, er_r) = er.process(&mono);
            for (i, (&el, &er_r)) in er_l.iter().zip(er_r.iter()).enumerate() {
                if i < left.len() {
                    left[i] += el;
                }
                if i < right.len() {
                    right[i] += er_r;
                }
            }
        }

        // 7. Reverb.
        if let Some(ref reverb_config) = self.config.reverb {
            crate::kokoro_chorus_reverb::apply_reverb(
                &mut left,
                reverb_config,
                false,
                None,
                None,
                None,
            )?;
            crate::kokoro_chorus_reverb::apply_reverb(
                &mut right,
                reverb_config,
                false,
                None,
                None,
                None,
            )?;
        }

        // 7b. Convolution reverb (after Schroeder reverb, before limiter).
        // Provides more realistic room simulation using FFT convolution with
        // real or synthetic impulse responses.
        if let Some(ref mut conv) = self.convolution_reverb {
            left = conv.process(&left);
            conv.reset(); // Reset overlap state between L/R.
            right = conv.process(&right);
        }

        // 7b2. Delay (bus, after reverb/convolution, before freeze).
        // Multi-tap stereo delay/echo with per-tap pan and feedback.
        if let Some(ref mut delay) = self.multi_tap_delay {
            delay.process_stereo(&mut left, &mut right)?;
        }

        // 7c. Spectral freeze (bus, after delay, before ensemble).
        // Captures and sustains the current spectrum for drone/pad textures.
        if let Some(ref mut freezer) = self.spectral_freezer {
            freezer.process(&mut left);
            freezer.reset();
            freezer.process(&mut right);
        }

        // 7d. Ensemble (bus, after convolution/freeze, before spectral fill).
        // Stereo modulation, chorus, and diffusion for a wider, richer sound.
        if let Some(ref mut ensemble_proc) = self.ensemble_processor {
            ensemble_proc.process_stereo(&mut left, &mut right);
        }

        // 7d2. Spectral fill (bus, after ensemble, before gain staging).
        // Detects spectral gaps and generates complementary fill material
        // (harmonics, sub-harmonics, shaped noise) for a full, lush sound.
        if let Some(ref mut filler) = self.spectral_fill_processor {
            filler.process(&mut left, &mut right);
        }

        // 7e. Gain staging (bus, before loudness/limiter, auto-level to target LUFS).
        if let Some(ref gain_stager) = self.gain_stager {
            let sr = KOKORO_SAMPLE_RATE as f32;
            gain_stager.auto_level(&mut left, sr);
            gain_stager.auto_level(&mut right, sr);
        }

        // 7f. Loudness normalization (bus, after gain staging, before limiter).
        // Measures integrated loudness and normalizes to a target LUFS.
        if let Some(ref mut meter) = self.loudness_meter {
            meter.normalize_to_target(&mut left);
            meter.reset();
            meter.normalize_to_target(&mut right);
        }

        // 7f2. Loudness curve (bus, after loudness normalization, before tilt/limiter).
        // Fletcher-Munson equal-loudness contour compensation. Adjusts spectral
        // balance so the chorus sounds correct at the listener's playback level.
        if let Some(ref mut lc) = self.loudness_curve_processor {
            lc.process(&mut left);
            lc.reset();
            lc.process(&mut right);
        }

        // 7g. Spectral tilt (bus, after loudness curve, before limiter).
        // Adjusts overall tonal balance (brighter or darker).
        if let Some(ref mut tilt) = self.tilt_processor {
            tilt.process(&mut left);
            tilt.reset();
            tilt.process(&mut right);
        }

        // 8. Final limiter.
        // When true peak limiter is configured, it replaces the basic bus limiter
        // for intersample-accurate peak control. Otherwise fall back to basic.
        if let Some(ref mut tp_lim) = self.true_peak_limiter_processor {
            tp_lim.process_stereo(&mut left, &mut right)?;
        } else if let Some(ref mut lim) = self.limiter {
            lim.process(&mut left);
            lim.reset();
            lim.process(&mut right);
        }

        // 8b. Dither (bus, after limiter, before mix_analyzer).
        if let Some(ref mut dither_proc) = self.dither_processor {
            dither_proc.process(&mut left);
            dither_proc.reset();
            dither_proc.process(&mut right);
        }

        // 8c. Mix analyzer (bus, absolute LAST step, after dither).
        // Monitors spectral balance, stereo correlation, and applies
        // auto-correction for final quality assurance.
        if let Some(ref mut analyzer) = self.mix_analyzer_processor {
            analyzer.process(&mut left, &mut right);
        }

        // 9. Final safety hard-clip.
        //
        // Post-limiter stages (dither noise shaping, mix analyzer loudness
        // targeting) can push samples above the limiter ceiling. Dither
        // noise shaping accumulates error feedback that can overshoot by
        // >0.2 on transient-heavy material. The mix analyzer's LUFS gain
        // stage also applies gain after its own true-peak limiter.
        //
        // This hard-clip is the absolute last line of defense: no sample
        // leaves the pipeline above [-1.0, 1.0]. Standard mastering
        // practice per AES-6id-2006 (AES Information Document for Digital
        // Audio - Personal Computer Audio Quality Measurements).
        let ceiling = self
            .limiter
            .as_ref()
            .map(BusLimiter::ceiling_linear)
            .unwrap_or(1.0);
        for s in left.iter_mut() {
            *s = s.clamp(-ceiling, ceiling);
        }
        for s in right.iter_mut() {
            *s = s.clamp(-ceiling, ceiling);
        }

        Ok((left, right))
    }

    /// Format the processed output for final delivery.
    ///
    /// Call AFTER [`process`](Self::process) to convert the stereo (L, R)
    /// output into a [`FormattedOutput`] with optional sample rate conversion,
    /// bit depth quantization, and interleaving. Requires the `output` config
    /// to be set; returns `None` if output formatting is not configured.
    pub fn format_output(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Option<Result<FormattedOutput, KokoroError>> {
        self.output_formatter
            .as_mut()
            .map(|fmt| fmt.format_output(left, right, KOKORO_SAMPLE_RATE as u32))
    }

    /// Reset all stateful processors (filters, compressors, limiter).
    ///
    /// Call between unrelated audio segments to prevent state leakage.
    pub fn reset(&mut self) {
        if let Some(ref mut comp) = self.dynamics {
            comp.reset();
        }
        if let Some(ref mut sat) = self.saturation_processor {
            sat.reset();
        }
        if let Some(ref mut lim) = self.limiter {
            lim.reset();
        }
        if let Some(ref mut bgen) = self.breath_generator {
            bgen.reset();
        }
        if let Some(ref mut processors) = self.spatial_processors {
            for proc in processors.iter_mut() {
                proc.reset();
            }
        }
        if let Some(ref mut widener) = self.stereo_widener {
            widener.reset();
        }
        if let Some(ref mut conv) = self.convolution_reverb {
            conv.reset();
        }
        if let Some(ref mut exciter_procs) = self.exciter_processors {
            for proc in exciter_procs.iter_mut() {
                proc.reset();
            }
        }
        if let Some(ref mut ducker) = self.ducker {
            ducker.reset();
        }
        if let Some(ref mut dither_proc) = self.dither_processor {
            dither_proc.reset();
        }
        if let Some(ref mut mb_stereo) = self.multiband_stereo_processor {
            mb_stereo.reset();
        }
        if let Some(ref mut freezer) = self.spectral_freezer {
            freezer.reset();
        }
        if let Some(ref mut er) = self.early_reflections {
            er.reset();
        }
        if let Some(ref mut hrtf) = self.hrtf_processor {
            hrtf.reset();
        }
        if let Some(ref mut auto_eq_procs) = self.auto_eq_processors {
            for proc in auto_eq_procs.iter_mut() {
                proc.reset();
            }
        }
        if let Some(ref mut meter) = self.loudness_meter {
            meter.reset();
        }
        if let Some(ref mut sib_procs) = self.sibilance_processors {
            for proc in sib_procs.iter_mut() {
                proc.reset();
            }
        }
        if let Some(ref mut ensemble_proc) = self.ensemble_processor {
            ensemble_proc.reset();
        }
        if let Some(ref mut warmth_procs) = self.warmth_processors {
            for proc in warmth_procs.iter_mut() {
                proc.reset();
            }
        }
        if let Some(ref mut analyzer) = self.stereo_analyzer {
            analyzer.reset();
        }
        if let Some(ref mut tuner) = self.formant_tuner {
            tuner.reset();
        }
        if let Some(ref mut mp_proc) = self.micro_pitch_processor {
            mp_proc.reset();
        }
        if let Some(ref mut inton) = self.intonation_tracker {
            inton.reset();
        }
        if let Some(ref mut matcher) = self.spectral_matcher {
            matcher.reset();
        }
        if let Some(ref mut sub_bass) = self.sub_bass_processor {
            sub_bass.reset();
        }
        if let Some(ref mut adaptive) = self.adaptive_dynamics_processor {
            adaptive.reset();
        }
        if let Some(ref mut tilt) = self.tilt_processor {
            tilt.reset();
        }
        if let Some(ref mut onset_sync) = self.onset_synchronizer {
            onset_sync.reset();
        }
        if let Some(ref mut os) = self.oversampler {
            os.reset();
        }
        if let Some(ref mut depth_staging) = self.depth_staging_processor {
            depth_staging.reset();
        }
        if let Some(ref mut vocal_tract) = self.vocal_tract_processor {
            vocal_tract.reset();
        }
        if let Some(ref mut shimmer) = self.shimmer_processor {
            shimmer.reset();
        }
        if let Some(ref mut intelli) = self.intelligibility_optimizer {
            intelli.reset();
        }
        if let Some(ref mut allocator) = self.voice_allocator {
            allocator.reset();
        }
        if let Some(ref mut deq) = self.dynamic_eq_processor {
            deq.reset();
        }
        if let Some(ref mut bm) = self.bass_manager {
            bm.reset();
        }
        if let Some(ref mut delay) = self.multi_tap_delay {
            delay.reset();
        }
        if let Some(ref mut auto_mixer) = self.auto_mixer {
            auto_mixer.reset();
        }
        if let Some(ref mut mix_an) = self.mix_analyzer_processor {
            mix_an.reset();
        }
        if let Some(ref mut output_fmt) = self.output_formatter {
            output_fmt.reset();
        }
        if let Some(ref mut decorr) = self.decorrelation_processor {
            decorr.reset();
        }
        if let Some(ref mut harm_tuner) = self.harmonic_tuner_processor {
            harm_tuner.reset();
        }
        if let Some(ref mut lc) = self.loudness_curve_processor {
            lc.reset();
        }
        if let Some(ref mut masking) = self.masking_compensator_processor {
            masking.reset();
        }
        if let Some(ref mut mic) = self.mic_model_processor {
            mic.reset();
        }
        if let Some(ref mut filler) = self.spectral_fill_processor {
            filler.reset();
        }
        if let Some(ref mut aligner) = self.vowel_aligner {
            aligner.reset();
        }
        if let Some(ref mut air_abs) = self.air_absorption_processor {
            air_abs.reset();
        }
        if let Some(ref mut pres_procs) = self.presence_processors {
            for proc in pres_procs.iter_mut() {
                proc.reset();
            }
        }
        if let Some(ref mut thick) = self.thickener_processor {
            thick.reset();
        }
        if let Some(ref mut optimizer) = self.stereo_optimizer_processor {
            optimizer.reset();
        }
        if let Some(ref mut ta) = self.transient_aligner {
            ta.reset();
        }
        if let Some(ref mut tp_lim) = self.true_peak_limiter_processor {
            tp_lim.reset();
        }
        // MixBusProcessor does not expose a reset — it's stateless per-call
        // for the biquad filters, which reset on the next process_voice call.
    }

    /// Get a reference to the pipeline config.
    #[must_use]
    pub fn config(&self) -> &ChorusMasterConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Stateless convenience function
// ---------------------------------------------------------------------------

/// Process voices through the full chorus pipeline (stateless convenience).
///
/// Creates a temporary [`ChorusMasterPipeline`] and runs a single pass.
/// For repeated use, prefer constructing a pipeline once and calling
/// [`ChorusMasterPipeline::process`] to reuse filter state.
pub fn process_chorus(
    voices: &[Vec<f32>],
    config: &ChorusMasterConfig,
) -> Result<(Vec<f32>, Vec<f32>), KokoroError> {
    let mut pipeline = ChorusMasterPipeline::new(config.clone())?;
    pipeline.process(voices)
}

#[cfg(test)]
#[path = "kokoro_chorus_pipeline_tests.rs"]
mod tests;
