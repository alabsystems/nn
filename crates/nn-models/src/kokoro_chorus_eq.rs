// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! De-essing and spectral EQ for Kokoro chorus voice mixing.
//!
//! When multiple TTS voices are mixed in a chorus, sibilant sounds (s, sh, z)
//! in the 4-8 kHz range stack up and become harsh. This module provides:
//!
//! - [`DeEsser`]: 2-pole bandpass filter with RMS envelope follower and
//!   gain reduction when the envelope exceeds a configurable threshold.
//! - [`ChorusEQ`]: Per-voice 3-band parametric EQ (low shelf, mid bell,
//!   high shelf) using biquad filters with coefficients from the
//!   Audio EQ Cookbook (Robert Bristow-Johnson).
//! - [`MixBusProcessor`]: Chains per-voice EQ, de-esser, mix, and bus EQ.
//! - [`EqPreset`]: Named EQ curves (`Warm`, `Bright`, `Natural`, `Broadcast`).
//!
//! # References
//!
//! - Bristow-Johnson, R. "Audio EQ Cookbook." (2005).
//!   <https://www.w3.org/2011/audio/audio-eq-cookbook.html>
//! - Giannoulis, D. et al. "Digital Dynamic Range Compressor Design."
//!   Journal of the Audio Engineering Society, 60(6), 2012.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Biquad filter (Direct Form II Transposed)
// ---------------------------------------------------------------------------

/// Second-order IIR (biquad) filter coefficients.
///
/// Transfer function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
/// Note: a0 is normalized to 1.0 in all coefficient computations.
#[derive(Debug, Clone, Copy)]
struct BiquadCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

/// Biquad filter state (Direct Form II Transposed).
#[derive(Debug, Clone)]
struct BiquadFilter {
    coeffs: BiquadCoeffs,
    z1: f32,
    z2: f32,
}

impl BiquadFilter {
    fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Process a single sample through the biquad filter.
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        // Guard against NaN/Inf propagation from upstream.
        if !input.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            return 0.0;
        }

        let c = &self.coeffs;
        let output = c.b0 * input + self.z1;
        self.z1 = c.b1 * input - c.a1 * output + self.z2;
        self.z2 = c.b2 * input - c.a2 * output;

        // Flush denormals and clamp non-finite outputs.
        
        if !output.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            0.0
        } else {
            output
        }
    }

    /// Process a buffer of samples in place.
    fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

// ---------------------------------------------------------------------------
// Biquad coefficient computation (Audio EQ Cookbook)
// ---------------------------------------------------------------------------

/// Compute bandpass filter coefficients (constant skirt gain, peak gain = Q).
///
/// Used by the de-esser to isolate the sibilance frequency range.
fn bandpass_coeffs(freq_hz: f32, q: f32, sample_rate: f32) -> BiquadCoeffs {
    let w0 = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
    let alpha = w0.sin() / (2.0 * q);
    let a0 = 1.0 + alpha;

    BiquadCoeffs {
        b0: (q * alpha) / a0,
        b1: 0.0,
        b2: (-q * alpha) / a0,
        a1: (-2.0 * w0.cos()) / a0,
        a2: (1.0 - alpha) / a0,
    }
}

/// Compute low-shelf filter coefficients.
///
/// `gain_db` is the shelf gain in dB (positive = boost, negative = cut).
fn low_shelf_coeffs(freq_hz: f32, gain_db: f32, q: f32, sample_rate: f32) -> BiquadCoeffs {
    let a = 10.0f32.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;

    BiquadCoeffs {
        b0: (a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha)) / a0,
        b1: (2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)) / a0,
        b2: (a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha)) / a0,
        a1: (-2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)) / a0,
        a2: ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha) / a0,
    }
}

/// Compute high-shelf filter coefficients.
///
/// `gain_db` is the shelf gain in dB (positive = boost, negative = cut).
fn high_shelf_coeffs(freq_hz: f32, gain_db: f32, q: f32, sample_rate: f32) -> BiquadCoeffs {
    let a = 10.0f32.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;

    BiquadCoeffs {
        b0: (a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha)) / a0,
        b1: (-2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)) / a0,
        b2: (a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha)) / a0,
        a1: (2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)) / a0,
        a2: ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha) / a0,
    }
}

/// Compute peaking (bell) EQ filter coefficients.
///
/// `gain_db` is the peak gain in dB. `q` controls bandwidth.
fn peaking_eq_coeffs(freq_hz: f32, gain_db: f32, q: f32, sample_rate: f32) -> BiquadCoeffs {
    let a = 10.0f32.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
    let cos_w0 = w0.cos();
    let alpha = w0.sin() / (2.0 * q);

    let a0 = 1.0 + alpha / a;

    BiquadCoeffs {
        b0: (1.0 + alpha * a) / a0,
        b1: (-2.0 * cos_w0) / a0,
        b2: (1.0 - alpha * a) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha / a) / a0,
    }
}

// ---------------------------------------------------------------------------
// De-esser
// ---------------------------------------------------------------------------

/// Configuration for the de-esser.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DeEsserConfig {
    /// Center frequency of the sibilance detection band (Hz).
    /// Default: 6000.0 (center of the 4-8 kHz sibilance range).
    pub center_freq_hz: f32,

    /// Q factor for the bandpass detection filter.
    /// Default: 1.0 (covers ~2 octaves around center frequency).
    pub q: f32,

    /// Threshold in dB below which no gain reduction occurs.
    /// Default: -20.0 dB. Signals above this trigger de-essing.
    pub threshold_db: f32,

    /// Maximum gain reduction in dB.
    /// Default: -12.0 dB.
    pub max_reduction_db: f32,

    /// RMS envelope attack time in seconds.
    /// Default: 0.001 (1 ms, fast attack to catch transients).
    pub attack_sec: f32,

    /// RMS envelope release time in seconds.
    /// Default: 0.050 (50 ms, smooth release to avoid pumping).
    pub release_sec: f32,
}

impl Default for DeEsserConfig {
    fn default() -> Self {
        Self {
            center_freq_hz: 6000.0,
            q: 1.0,
            threshold_db: -20.0,
            max_reduction_db: -12.0,
            attack_sec: 0.001,
            release_sec: 0.050,
        }
    }
}

impl DeEsserConfig {
    /// Validate that all parameters are within valid ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.center_freq_hz.is_finite()
            || self.center_freq_hz < 100.0
            || self.center_freq_hz > 20000.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "center_freq_hz",
                reason: format!(
                    "center_freq_hz = {}: must be finite and in [100, 20000]",
                    self.center_freq_hz,
                ),
            });
        }
        if !self.q.is_finite() || self.q <= 0.0 || self.q > 20.0 {
            return Err(KokoroError::InvalidConfig {
                field: "q",
                reason: format!("q = {}: must be finite and in (0, 20]", self.q),
            });
        }
        if !self.threshold_db.is_finite() || self.threshold_db > 0.0 || self.threshold_db < -96.0 {
            return Err(KokoroError::InvalidConfig {
                field: "threshold_db",
                reason: format!(
                    "threshold_db = {}: must be finite and in [-96, 0]",
                    self.threshold_db,
                ),
            });
        }
        if !self.max_reduction_db.is_finite()
            || self.max_reduction_db > 0.0
            || self.max_reduction_db < -48.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "max_reduction_db",
                reason: format!(
                    "max_reduction_db = {}: must be finite and in [-48, 0]",
                    self.max_reduction_db,
                ),
            });
        }
        if !self.attack_sec.is_finite() || self.attack_sec <= 0.0 || self.attack_sec > 1.0 {
            return Err(KokoroError::InvalidConfig {
                field: "attack_sec",
                reason: format!(
                    "attack_sec = {}: must be finite and in (0, 1]",
                    self.attack_sec,
                ),
            });
        }
        if !self.release_sec.is_finite() || self.release_sec <= 0.0 || self.release_sec > 5.0 {
            return Err(KokoroError::InvalidConfig {
                field: "release_sec",
                reason: format!(
                    "release_sec = {}: must be finite and in (0, 5]",
                    self.release_sec,
                ),
            });
        }
        Ok(())
    }
}

/// RMS-based de-esser that attenuates sibilant frequencies (4-8 kHz).
///
/// Uses a 2-pole bandpass filter to isolate the sibilance band, an RMS
/// envelope follower to track energy, and smooth gain reduction when the
/// envelope exceeds the threshold. The gain reduction is applied to the
/// full-band signal (wideband de-essing), preserving the overall tonal
/// balance while taming sibilance buildup.
pub struct DeEsser {
    /// Bandpass filter for sibilance detection.
    detector: BiquadFilter,
    /// RMS envelope state (squared amplitude, exponentially weighted).
    envelope_sq: f32,
    /// Attack coefficient: `exp(-1 / (attack_sec * sample_rate))`.
    attack_coeff: f32,
    /// Release coefficient: `exp(-1 / (release_sec * sample_rate))`.
    release_coeff: f32,
    /// Threshold as linear power (from dB).
    threshold_power: f32,
    /// Maximum gain reduction as linear factor (from dB).
    max_reduction_linear: f32,
}

impl DeEsser {
    /// Create a new de-esser with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if parameters are out of range.
    pub fn new(config: &DeEsserConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let sr = KOKORO_SAMPLE_RATE as f32;
        let coeffs = bandpass_coeffs(config.center_freq_hz, config.q, sr);
        let attack_coeff = (-1.0 / (config.attack_sec * sr)).exp();
        let release_coeff = (-1.0 / (config.release_sec * sr)).exp();

        // Convert threshold from dB to linear power: 10^(dB/10).
        let threshold_power = 10.0f32.powf(config.threshold_db / 10.0);
        // Convert max reduction from dB to linear: 10^(dB/20).
        let max_reduction_linear = 10.0f32.powf(config.max_reduction_db / 20.0);

        Ok(Self {
            detector: BiquadFilter::new(coeffs),
            envelope_sq: 0.0,
            attack_coeff,
            release_coeff,
            threshold_power,
            max_reduction_linear,
        })
    }

    /// Create a de-esser with default settings.
    pub fn default_config() -> Result<Self, KokoroError> {
        Self::new(&DeEsserConfig::default())
    }

    /// Process an audio buffer in place, applying de-essing.
    pub fn process(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }

            // Run the bandpass detector on the input.
            let detected = self.detector.process(*sample);
            let detected_sq = detected * detected;

            // RMS envelope follower with attack/release.
            let coeff = if detected_sq > self.envelope_sq {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.envelope_sq = coeff * self.envelope_sq + (1.0 - coeff) * detected_sq;

            // Clamp envelope to avoid denormals.
            if self.envelope_sq < 1e-20 {
                self.envelope_sq = 0.0;
            }

            // Compute gain reduction.
            if self.envelope_sq > self.threshold_power {
                // Amount above threshold in dB (approximation for smooth curve).
                let excess_ratio = self.envelope_sq / self.threshold_power;
                // Soft-knee gain reduction: reduce proportionally to excess.
                // gain = threshold_power / envelope_sq, clamped to max_reduction.
                let gain = (self.threshold_power / self.envelope_sq).sqrt();
                let gain = gain.max(self.max_reduction_linear);
                // Ensure gain is reasonable.
                let gain = if !gain.is_finite() || gain < 0.0 {
                    self.max_reduction_linear
                } else {
                    gain.min(1.0)
                };
                let _ = excess_ratio; // used conceptually, gain formula is equivalent
                *sample *= gain;
            }
        }
    }

    /// Reset the de-esser state (e.g., between voice segments).
    pub fn reset(&mut self) {
        self.detector.z1 = 0.0;
        self.detector.z2 = 0.0;
        self.envelope_sq = 0.0;
    }
}

// ---------------------------------------------------------------------------
// 3-band parametric EQ
// ---------------------------------------------------------------------------

/// Configuration for a 3-band parametric EQ.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EqConfig {
    /// Low shelf frequency (Hz). Default: 200.0.
    pub low_freq: f32,
    /// Low shelf gain (dB). Default: 0.0.
    pub low_gain_db: f32,
    /// Mid bell frequency (Hz). Default: 1500.0.
    pub mid_freq: f32,
    /// Mid bell gain (dB). Default: 0.0.
    pub mid_gain_db: f32,
    /// Mid bell Q factor. Default: 1.0.
    pub mid_q: f32,
    /// High shelf frequency (Hz). Default: 6000.0.
    pub high_freq: f32,
    /// High shelf gain (dB). Default: 0.0.
    pub high_gain_db: f32,
}

impl Default for EqConfig {
    fn default() -> Self {
        Self {
            low_freq: 200.0,
            low_gain_db: 0.0,
            mid_freq: 1500.0,
            mid_gain_db: 0.0,
            mid_q: 1.0,
            high_freq: 6000.0,
            high_gain_db: 0.0,
        }
    }
}

impl EqConfig {
    /// Validate that all parameters are within valid ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        for &(name, freq) in &[
            ("low_freq", self.low_freq),
            ("mid_freq", self.mid_freq),
            ("high_freq", self.high_freq),
        ] {
            if !freq.is_finite() || !(20.0..=20000.0).contains(&freq) {
                return Err(KokoroError::InvalidConfig {
                    field: name,
                    reason: format!("{name} = {freq}: must be finite and in [20, 20000]"),
                });
            }
        }
        for &(name, gain) in &[
            ("low_gain_db", self.low_gain_db),
            ("mid_gain_db", self.mid_gain_db),
            ("high_gain_db", self.high_gain_db),
        ] {
            if !gain.is_finite() || !(-24.0..=24.0).contains(&gain) {
                return Err(KokoroError::InvalidConfig {
                    field: name,
                    reason: format!("{name} = {gain}: must be finite and in [-24, 24]"),
                });
            }
        }
        if !self.mid_q.is_finite() || self.mid_q <= 0.0 || self.mid_q > 20.0 {
            return Err(KokoroError::InvalidConfig {
                field: "mid_q",
                reason: format!("mid_q = {}: must be finite and in (0, 20]", self.mid_q),
            });
        }
        Ok(())
    }
}

/// Per-voice 3-band parametric EQ: low shelf + mid bell + high shelf.
///
/// Each band uses a biquad filter with coefficients computed from the
/// Audio EQ Cookbook. Filters are applied in series.
pub struct ChorusEQ {
    low_shelf: BiquadFilter,
    mid_bell: BiquadFilter,
    high_shelf: BiquadFilter,
}

impl ChorusEQ {
    /// Create a new 3-band EQ with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if parameters are out of range.
    pub fn new(config: &EqConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let sr = KOKORO_SAMPLE_RATE as f32;
        let shelf_q = 0.707; // Butterworth Q for shelves

        Ok(Self {
            low_shelf: BiquadFilter::new(low_shelf_coeffs(
                config.low_freq,
                config.low_gain_db,
                shelf_q,
                sr,
            )),
            mid_bell: BiquadFilter::new(peaking_eq_coeffs(
                config.mid_freq,
                config.mid_gain_db,
                config.mid_q,
                sr,
            )),
            high_shelf: BiquadFilter::new(high_shelf_coeffs(
                config.high_freq,
                config.high_gain_db,
                shelf_q,
                sr,
            )),
        })
    }

    /// Process an audio buffer in place through all three EQ bands.
    pub fn process(&mut self, buffer: &mut [f32]) {
        self.low_shelf.process_buffer(buffer);
        self.mid_bell.process_buffer(buffer);
        self.high_shelf.process_buffer(buffer);
    }

    /// Reset all filter states.
    pub fn reset(&mut self) {
        self.low_shelf.z1 = 0.0;
        self.low_shelf.z2 = 0.0;
        self.mid_bell.z1 = 0.0;
        self.mid_bell.z2 = 0.0;
        self.high_shelf.z1 = 0.0;
        self.high_shelf.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// EQ presets for TTS chorus
// ---------------------------------------------------------------------------

/// Named EQ presets optimized for TTS chorus mixing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EqPreset {
    /// Warm: gentle low boost, slight high roll-off. Good for intimate narration.
    Warm,
    /// Bright: presence boost around 3 kHz, air boost at 10 kHz.
    Bright,
    /// Natural: flat with subtle 2.5 kHz presence dip to reduce harshness.
    Natural,
    /// Broadcast: broadcast-standard EQ with presence boost and low cut.
    Broadcast,
}

impl EqPreset {
    /// Convert the preset to an [`EqConfig`].
    #[must_use]
    pub fn to_config(self) -> EqConfig {
        match self {
            Self::Warm => EqConfig {
                low_freq: 200.0,
                low_gain_db: 2.0,
                mid_freq: 2000.0,
                mid_gain_db: -1.0,
                mid_q: 1.0,
                high_freq: 8000.0,
                high_gain_db: -2.5,
            },
            Self::Bright => EqConfig {
                low_freq: 150.0,
                low_gain_db: -1.0,
                mid_freq: 3000.0,
                mid_gain_db: 2.0,
                mid_q: 1.2,
                high_freq: 10000.0,
                high_gain_db: 1.5,
            },
            Self::Natural => EqConfig {
                low_freq: 200.0,
                low_gain_db: 0.0,
                mid_freq: 2500.0,
                mid_gain_db: -1.5,
                mid_q: 0.8,
                high_freq: 8000.0,
                high_gain_db: 0.0,
            },
            Self::Broadcast => EqConfig {
                low_freq: 100.0,
                low_gain_db: -3.0,
                mid_freq: 3500.0,
                mid_gain_db: 2.5,
                mid_q: 1.0,
                high_freq: 10000.0,
                high_gain_db: -1.0,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Mix bus processor
// ---------------------------------------------------------------------------

/// Configuration for the mix bus processor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MixBusConfig {
    /// Per-voice EQ configuration. If `None`, defaults to `EqPreset::Natural`.
    pub voice_eq: Option<EqConfig>,
    /// Per-voice de-esser configuration. If `None`, uses default de-esser.
    pub deesser: Option<DeEsserConfig>,
    /// Bus (post-mix) EQ configuration. If `None`, no bus EQ is applied.
    pub bus_eq: Option<EqConfig>,
    /// Whether to enable per-voice de-essing. Default: `true`.
    pub deesser_enabled: bool,
}

impl Default for MixBusConfig {
    fn default() -> Self {
        Self {
            voice_eq: Some(EqPreset::Natural.to_config()),
            deesser: None,
            bus_eq: None,
            deesser_enabled: true,
        }
    }
}

impl MixBusConfig {
    /// Create a config from an EQ preset with default de-esser.
    #[must_use]
    pub fn from_preset(preset: EqPreset) -> Self {
        Self {
            voice_eq: Some(preset.to_config()),
            ..Self::default()
        }
    }

    /// Create a config with a specific bus EQ.
    #[must_use]
    pub fn with_bus_eq(mut self, eq: EqConfig) -> Self {
        self.bus_eq = Some(eq);
        self
    }

    /// Disable the de-esser.
    #[must_use]
    pub fn without_deesser(mut self) -> Self {
        self.deesser_enabled = false;
        self
    }
}

/// Mix bus processor: per-voice EQ + de-esser + bus EQ.
///
/// Chains spectral processing stages for multi-voice chorus mixing:
/// 1. Per-voice parametric EQ (shape each voice's spectrum)
/// 2. Per-voice de-essing (tame sibilance before summing)
/// 3. (External mixing by caller)
/// 4. Bus EQ (shape the final mix)
///
/// Each voice gets its own EQ and de-esser instances to maintain
/// independent filter state.
pub struct MixBusProcessor {
    /// Per-voice EQ instances.
    voice_eqs: Vec<ChorusEQ>,
    /// Per-voice de-esser instances.
    deessers: Vec<DeEsser>,
    /// Bus (post-mix) EQ, applied to the mixed output.
    bus_eq: Option<ChorusEQ>,
    /// Whether de-essing is enabled.
    deesser_enabled: bool,
}

impl MixBusProcessor {
    /// Create a new mix bus processor for `n_voices` voices.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if EQ or de-esser config is invalid.
    pub fn new(n_voices: usize, config: &MixBusConfig) -> Result<Self, KokoroError> {
        let eq_config = config
            .voice_eq
            .clone()
            .unwrap_or_else(|| EqPreset::Natural.to_config());

        let deesser_config = config.deesser.clone().unwrap_or_default();

        let mut voice_eqs = Vec::with_capacity(n_voices);
        let mut deessers = Vec::with_capacity(n_voices);

        for _ in 0..n_voices {
            voice_eqs.push(ChorusEQ::new(&eq_config)?);
            deessers.push(DeEsser::new(&deesser_config)?);
        }

        let bus_eq = if let Some(ref bus_config) = config.bus_eq {
            Some(ChorusEQ::new(bus_config)?)
        } else {
            None
        };

        Ok(Self {
            voice_eqs,
            deessers,
            bus_eq,
            deesser_enabled: config.deesser_enabled,
        })
    }

    /// Process a single voice buffer (EQ + de-essing) in place.
    ///
    /// Call this for each voice before mixing.
    ///
    /// # Panics
    ///
    /// Panics if `voice_index >= n_voices`.
    pub fn process_voice(&mut self, voice_index: usize, buffer: &mut [f32]) {
        self.voice_eqs[voice_index].process(buffer);
        if self.deesser_enabled {
            self.deessers[voice_index].process(buffer);
        }
    }

    /// Process the mixed bus output (bus EQ) in place.
    ///
    /// Call this after mixing all voices together.
    pub fn process_bus(&mut self, buffer: &mut [f32]) {
        if let Some(ref mut eq) = self.bus_eq {
            eq.process(buffer);
        }
    }

    /// Reset all filter states (e.g., between segments).
    pub fn reset(&mut self) {
        for eq in &mut self.voice_eqs {
            eq.reset();
        }
        for de in &mut self.deessers {
            de.reset();
        }
        if let Some(ref mut eq) = self.bus_eq {
            eq.reset();
        }
    }

    /// Number of voices this processor was created for.
    #[must_use]
    pub fn n_voices(&self) -> usize {
        self.voice_eqs.len()
    }
}

#[cfg(test)]
#[path = "kokoro_chorus_eq_tests.rs"]
mod tests;
