// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dynamic EQ for Kokoro chorus mix control.
//!
//! Dynamic EQ applies frequency-dependent gain reduction only when specific
//! bands exceed their thresholds. Unlike static EQ (constant boost/cut),
//! dynamic EQ acts as a frequency-selective compressor: the peaking filter
//! gain is modulated by a per-band envelope follower, so untriggered bands
//! pass through unaltered. This is essential for controlling problem
//! frequencies in multi-voice chorus without dulling the overall mix.
//!
//! # Architecture
//!
//! ```text
//! Input ─┬──[Band 1 sidechain BPF]──[Envelope]──[Gain computer]──┐
//!        ├──[Band 2 sidechain BPF]──[Envelope]──[Gain computer]──┤
//!        ├──[Band 3 sidechain BPF]──[Envelope]──[Gain computer]──┤
//!        └──[Band 4 sidechain BPF]──[Envelope]──[Gain computer]──┤
//!                                                                  │
//!        Input ──[Sum of per-band peaking EQ w/ dynamic gain]──────┘
//!                                                                  │
//!        Dry/Wet mix ──────────────────────────────────────── Output
//! ```
//!
//! Per-band processing:
//! 1. Sidechain bandpass isolates the detection frequency range.
//! 2. Log-domain envelope follower tracks RMS level with separate
//!    attack/release, providing musical ballistics.
//! 3. Gain computer maps level above threshold through the compression
//!    ratio, yielding gain reduction in dB.
//! 4. The gain reduction drives a peaking EQ cut applied to the main
//!    signal at that band's frequency.
//!
//! # Default vocal chorus preset
//!
//! - Band 1: 250 Hz, Q=1.0, threshold=-18 dB, ratio=2:1 (mud control)
//! - Band 2: 800 Hz, Q=1.2, threshold=-16 dB, ratio=2:1 (boxiness)
//! - Band 3: 3000 Hz, Q=1.5, threshold=-20 dB, ratio=3:1 (harshness)
//! - Band 4: 6500 Hz, Q=1.0, threshold=-22 dB, ratio=2.5:1 (sibilance)
//!
//! # References
//!
//! - Bristow-Johnson, R. "Audio EQ Cookbook." (2005).
//!   <https://www.w3.org/2011/audio/audio-eq-cookbook.html>
//! - Giannoulis, D. et al. "Digital Dynamic Range Compressor Design."
//!   Journal of the Audio Engineering Society, 60(6), 2012.
//! - McNally, G. "Dynamic Range Control of Digital Audio Signals."
//!   Journal of the Audio Engineering Society, 32(5), 1984.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Biquad filter (Direct Form II Transposed)
// ---------------------------------------------------------------------------

/// Second-order IIR (biquad) filter coefficients.
///
/// Transfer function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
/// a0 is normalized to 1.0 in all coefficient computations.
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

    /// Process a single sample through the biquad.
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        if !input.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            return 0.0;
        }

        let c = &self.coeffs;
        let output = c.b0 * input + self.z1;
        self.z1 = c.b1 * input - c.a1 * output + self.z2;
        self.z2 = c.b2 * input - c.a2 * output;

        if !output.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            0.0
        } else {
            output
        }
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Biquad coefficient computation (Robert Bristow-Johnson Audio EQ Cookbook)
// ---------------------------------------------------------------------------

/// Compute bandpass filter coefficients (constant skirt gain, peak gain = Q).
///
/// Used for the sidechain detection path to isolate each band's frequency.
fn bandpass_coefficients(freq: f32, q: f32, sample_rate: f32) -> BiquadCoeffs {
    let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let a0 = 1.0 + alpha;

    BiquadCoeffs {
        b0: (q * alpha) / a0,
        b1: 0.0,
        b2: (-q * alpha) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha) / a0,
    }
}

/// Compute peaking (bell) EQ filter coefficients.
///
/// Bristow-Johnson peaking EQ: boosts or cuts at `freq` with bandwidth
/// controlled by `q`, and peak amplitude `gain_db` in dB.
///
/// This is used for the processing path: the dynamic gain reduction is
/// applied as a negative `gain_db` to create a frequency-selective cut.
pub fn peaking_eq_coefficients(freq: f32, q: f32, gain_db: f32, sample_rate: f32) -> BiquadCoeffs {
    let a = 10.0f32.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate;
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

/// Compute highpass filter coefficients (sidechain HPF).
///
/// 2nd-order Butterworth HPF for removing low-frequency content from
/// the sidechain detection path, preventing bass energy from triggering
/// mid/high band compression.
fn highpass_coefficients(freq: f32, sample_rate: f32) -> BiquadCoeffs {
    let q = std::f32::consts::FRAC_1_SQRT_2; // Butterworth Q = 0.7071
    let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate;
    let cos_w0 = w0.cos();
    let alpha = w0.sin() / (2.0 * q);
    let a0 = 1.0 + alpha;

    BiquadCoeffs {
        b0: f32::midpoint(1.0, cos_w0) / a0,
        b1: (-(1.0 + cos_w0)) / a0,
        b2: f32::midpoint(1.0, cos_w0) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha) / a0,
    }
}

// ---------------------------------------------------------------------------
// Per-band configuration
// ---------------------------------------------------------------------------

/// Configuration for a single dynamic EQ band.
#[derive(Debug, Clone, Copy)]
pub struct DynamicEqBandConfig {
    /// Center frequency in Hz for both detection and processing.
    pub frequency_hz: f32,
    /// Q factor (bandwidth control). Higher Q = narrower band.
    /// Typical range: 0.5 (wide) to 8.0 (surgical).
    pub q: f32,
    /// Threshold in dBFS. The band only compresses when the detected
    /// level in that frequency range exceeds this threshold.
    pub threshold_db: f32,
    /// Compression ratio (e.g. 2.0 means 2:1). 1.0 = no compression.
    pub ratio: f32,
    /// Attack time in milliseconds. How quickly the compressor reacts
    /// to transients above threshold.
    pub attack_ms: f32,
    /// Release time in milliseconds. How quickly the compressor releases
    /// after the signal drops below threshold.
    pub release_ms: f32,
    /// Makeup gain in dB applied after dynamic processing.
    /// Typically 0.0 for dynamic EQ (we only cut, not boost+cut).
    pub gain_db: f32,
}

impl DynamicEqBandConfig {
    /// Validate band parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let nyquist = KOKORO_SAMPLE_RATE as f32 / 2.0;
        if !self.frequency_hz.is_finite()
            || self.frequency_hz < 20.0
            || self.frequency_hz >= nyquist
        {
            return Err(KokoroError::InvalidConfig {
                field: "frequency_hz",
                reason: format!(
                    "frequency_hz = {}: must be finite and in [20, {})",
                    self.frequency_hz, nyquist,
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
        if !self.ratio.is_finite() || self.ratio < 1.0 || self.ratio > 100.0 {
            return Err(KokoroError::InvalidConfig {
                field: "ratio",
                reason: format!("ratio = {}: must be finite and in [1, 100]", self.ratio),
            });
        }
        if !self.attack_ms.is_finite() || self.attack_ms < 0.01 || self.attack_ms > 1000.0 {
            return Err(KokoroError::InvalidConfig {
                field: "attack_ms",
                reason: format!(
                    "attack_ms = {}: must be finite and in [0.01, 1000]",
                    self.attack_ms,
                ),
            });
        }
        if !self.release_ms.is_finite() || self.release_ms < 1.0 || self.release_ms > 5000.0 {
            return Err(KokoroError::InvalidConfig {
                field: "release_ms",
                reason: format!(
                    "release_ms = {}: must be finite and in [1, 5000]",
                    self.release_ms,
                ),
            });
        }
        if !self.gain_db.is_finite() || self.gain_db < -24.0 || self.gain_db > 24.0 {
            return Err(KokoroError::InvalidConfig {
                field: "gain_db",
                reason: format!(
                    "gain_db = {}: must be finite and in [-24, 24]",
                    self.gain_db,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Sidechain filter option
// ---------------------------------------------------------------------------

/// Sidechain filter mode for dynamic EQ detection.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
#[derive(Default)]
pub enum SidechainFilter {
    /// No sidechain filter -- raw signal feeds the detector.
    #[default]
    None,
    /// Internal highpass filter on the sidechain at the specified
    /// frequency (Hz). Removes low-frequency content from detection,
    /// preventing bass energy from triggering mid/high bands.
    HighPass(f32),
}


// ---------------------------------------------------------------------------
// Top-level dynamic EQ config
// ---------------------------------------------------------------------------

/// Configuration for the multi-band dynamic EQ processor.
///
/// Each band is an independent frequency-selective compressor with its own
/// detection filter, envelope follower, gain computer, and peaking EQ.
///
/// # Vocal chorus default preset
///
/// The default preset is tuned for multi-voice TTS chorus mixing:
/// - Band 1 (250 Hz): controls low-mid muddiness from stacking voices
/// - Band 2 (800 Hz): tames boxy resonance
/// - Band 3 (3 kHz): reduces harshness in the presence range
/// - Band 4 (6.5 kHz): controls sibilance buildup
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DynamicEqConfig {
    /// Per-band configurations. Supports 1-6 bands.
    pub bands: Vec<DynamicEqBandConfig>,
    /// Global dry/wet mix (0.0 = fully dry, 1.0 = fully wet).
    /// Default: 1.0 (fully processed).
    pub mix: f32,
    /// Sidechain filter option for the detection path.
    /// Default: `SidechainFilter::None`.
    pub sidechain_filter: SidechainFilter,
}

impl DynamicEqConfig {
    /// Create a new dynamic EQ config with the given bands.
    pub fn new(bands: Vec<DynamicEqBandConfig>) -> Self {
        Self {
            bands,
            mix: 1.0,
            sidechain_filter: SidechainFilter::None,
        }
    }

    /// Builder: set the dry/wet mix.
    #[must_use]
    pub fn with_mix(mut self, mix: f32) -> Self {
        self.mix = mix;
        self
    }

    /// Builder: set the sidechain filter.
    #[must_use]
    pub fn with_sidechain_filter(mut self, filter: SidechainFilter) -> Self {
        self.sidechain_filter = filter;
        self
    }

    /// Vocal chorus preset: 4 bands tuned for multi-voice TTS.
    ///
    /// - 250 Hz: tame low-mid muddiness (gentle 2:1)
    /// - 800 Hz: reduce boxiness (gentle 2:1)
    /// - 3 kHz: control harshness (tighter 3:1)
    /// - 6.5 kHz: manage sibilance buildup (moderate 2.5:1)
    #[must_use]
    pub fn vocal_chorus() -> Self {
        Self {
            bands: vec![
                DynamicEqBandConfig {
                    frequency_hz: 250.0,
                    q: 1.0,
                    threshold_db: -18.0,
                    ratio: 2.0,
                    attack_ms: 15.0,
                    release_ms: 150.0,
                    gain_db: 0.0,
                },
                DynamicEqBandConfig {
                    frequency_hz: 800.0,
                    q: 1.2,
                    threshold_db: -16.0,
                    ratio: 2.0,
                    attack_ms: 10.0,
                    release_ms: 120.0,
                    gain_db: 0.0,
                },
                DynamicEqBandConfig {
                    frequency_hz: 3000.0,
                    q: 1.5,
                    threshold_db: -20.0,
                    ratio: 3.0,
                    attack_ms: 5.0,
                    release_ms: 80.0,
                    gain_db: 0.0,
                },
                DynamicEqBandConfig {
                    frequency_hz: 6500.0,
                    q: 1.0,
                    threshold_db: -22.0,
                    ratio: 2.5,
                    attack_ms: 3.0,
                    release_ms: 60.0,
                    gain_db: 0.0,
                },
            ],
            mix: 1.0,
            sidechain_filter: SidechainFilter::HighPass(80.0),
        }
    }

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.bands.is_empty() || self.bands.len() > 6 {
            return Err(KokoroError::InvalidConfig {
                field: "bands",
                reason: format!("bands.len() = {}: must be in [1, 6]", self.bands.len()),
            });
        }
        if !self.mix.is_finite() || self.mix < 0.0 || self.mix > 1.0 {
            return Err(KokoroError::InvalidConfig {
                field: "mix",
                reason: format!("mix = {}: must be finite and in [0, 1]", self.mix),
            });
        }
        if let SidechainFilter::HighPass(freq) = self.sidechain_filter {
            let nyquist = KOKORO_SAMPLE_RATE as f32 / 2.0;
            if !freq.is_finite() || freq < 20.0 || freq >= nyquist {
                return Err(KokoroError::InvalidConfig {
                    field: "sidechain_filter",
                    reason: format!(
                        "HPF frequency = {freq}: must be finite and in [20, {nyquist})",
                    ),
                });
            }
        }
        for (i, band) in self.bands.iter().enumerate() {
            band.validate().map_err(|e| KokoroError::InvalidConfig {
                field: "bands",
                reason: format!("band[{i}]: {e}"),
            })?;
        }
        Ok(())
    }
}

impl Default for DynamicEqConfig {
    fn default() -> Self {
        Self::vocal_chorus()
    }
}

// ---------------------------------------------------------------------------
// Per-band runtime state
// ---------------------------------------------------------------------------

/// Runtime state for a single dynamic EQ band.
///
/// Contains the sidechain bandpass filter for detection, the processing
/// peaking EQ (whose gain is modulated dynamically), and the log-domain
/// envelope follower state.
struct DynamicEqBand {
    /// Bandpass filter for sidechain level detection.
    detector_bpf: BiquadFilter,
    /// Current peaking EQ filter for applying gain reduction.
    /// Re-computed each sample when gain reduction changes.
    processing_eq: BiquadFilter,
    /// Band center frequency (Hz).
    frequency_hz: f32,
    /// Band Q factor.
    q: f32,
    /// Threshold in dB.
    threshold_db: f32,
    /// Compression ratio.
    ratio: f32,
    /// Makeup gain in dB.
    gain_db: f32,
    /// Attack coefficient for envelope follower.
    attack_coeff: f32,
    /// Release coefficient for envelope follower.
    release_coeff: f32,
    /// Log-domain envelope state (dB).
    /// Using log-domain for musical attack/release behavior.
    envelope_db: f32,
    /// Current gain reduction in dB (always >= 0, applied as negative).
    current_gr_db: f32,
    /// Sample rate for coefficient recomputation.
    sample_rate: f32,
}

impl DynamicEqBand {
    fn new(config: &DynamicEqBandConfig, sample_rate: f32) -> Self {
        let sr = f64::from(sample_rate);
        let attack_coeff = (-1.0 / (f64::from(config.attack_ms) * 0.001 * sr)).exp() as f32;
        let release_coeff = (-1.0 / (f64::from(config.release_ms) * 0.001 * sr)).exp() as f32;

        let detector_bpf = BiquadFilter::new(bandpass_coefficients(
            config.frequency_hz,
            config.q,
            sample_rate,
        ));

        // Start with unity gain (0 dB reduction).
        let processing_eq = BiquadFilter::new(peaking_eq_coefficients(
            config.frequency_hz,
            config.q,
            0.0,
            sample_rate,
        ));

        Self {
            detector_bpf,
            processing_eq,
            frequency_hz: config.frequency_hz,
            q: config.q,
            threshold_db: config.threshold_db,
            ratio: config.ratio,
            gain_db: config.gain_db,
            attack_coeff,
            release_coeff,
            envelope_db: -96.0,
            current_gr_db: 0.0,
            sample_rate,
        }
    }

    /// Detect level from the sidechain-filtered sample and compute
    /// gain reduction. Returns the gain reduction in dB (positive value).
    #[inline]
    fn detect_and_compute_gr(&mut self, sidechain_sample: f32) -> f32 {
        // Run bandpass detection filter.
        let detected = self.detector_bpf.process(sidechain_sample);
        let abs_detected = detected.abs();

        // Convert to dB, clamping to floor.
        let level_db = if abs_detected > 1e-10 {
            20.0 * abs_detected.log10()
        } else {
            -96.0
        };

        // Guard against non-finite level.
        let level_db = if level_db.is_finite() {
            level_db
        } else {
            -96.0
        };

        // Log-domain envelope follower: smooth in dB for musical behavior.
        // This is the key difference from linear-domain smoothing: attack
        // and release times are perceptually consistent regardless of level.
        let coeff = if level_db > self.envelope_db {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope_db = coeff * self.envelope_db + (1.0 - coeff) * level_db;

        // Clamp envelope floor.
        if self.envelope_db < -96.0 {
            self.envelope_db = -96.0;
        }

        // Gain computer: only reduce when above threshold.
        let over_db = self.envelope_db - self.threshold_db;
        if over_db <= 0.0 {
            0.0
        } else {
            // Gain reduction = overshoot * (1 - 1/ratio).
            // For ratio=2:1, signal 6 dB over threshold gets 3 dB reduction.
            over_db * (1.0 - 1.0 / self.ratio)
        }
    }

    /// Apply the processing peaking EQ with the current dynamic gain to
    /// a sample of the main (dry) signal.
    ///
    /// We update the peaking EQ coefficients only when the gain reduction
    /// changes by more than 0.1 dB to avoid per-sample coefficient
    /// recomputation while still tracking dynamics accurately.
    #[inline]
    fn apply(&mut self, input: f32, gr_db: f32) -> f32 {
        // Only recompute coefficients when gain reduction changes
        // significantly (>0.1 dB delta). This is the hot path.
        let total_gain_db = self.gain_db - gr_db;
        if (total_gain_db - (-self.current_gr_db + self.gain_db)).abs() > 0.1
            || (self.current_gr_db == 0.0 && gr_db > 0.0)
        {
            self.current_gr_db = gr_db;
            self.processing_eq.coeffs =
                peaking_eq_coefficients(self.frequency_hz, self.q, total_gain_db, self.sample_rate);
        }

        self.processing_eq.process(input)
    }

    fn reset(&mut self) {
        self.detector_bpf.reset();
        self.processing_eq.reset();
        self.envelope_db = -96.0;
        self.current_gr_db = 0.0;
        // Reset processing EQ to unity.
        self.processing_eq.coeffs =
            peaking_eq_coefficients(self.frequency_hz, self.q, self.gain_db, self.sample_rate);
    }

    /// Get current gain reduction in dB (positive = reducing).
    fn gain_reduction_db(&self) -> f32 {
        self.current_gr_db
    }
}

// ---------------------------------------------------------------------------
// Dynamic EQ processor
// ---------------------------------------------------------------------------

/// Multi-band dynamic EQ processor.
///
/// Each band independently detects energy via a sidechain bandpass filter,
/// tracks the level with a log-domain envelope follower, computes gain
/// reduction when the level exceeds the band's threshold, and applies
/// the reduction via a peaking EQ filter. Bands operate in series on
/// the main signal, allowing multiple problem frequencies to be
/// controlled simultaneously.
///
/// # Example
///
/// ```rust,no_run
/// use nn_models::kokoro_chorus_dynamic_eq::{DynamicEqConfig, DynamicEqProcessor};
///
/// let config = DynamicEqConfig::vocal_chorus();
/// let mut processor = DynamicEqProcessor::new(&config, 24000.0).unwrap();
///
/// let mut audio = vec![0.0f32; 1024];
/// // ... fill audio with samples ...
/// processor.process(&mut audio);
///
/// let gr = processor.get_gain_reduction();
/// // gr[0] = gain reduction in dB for band 1 (250 Hz mud)
/// // gr[1] = gain reduction in dB for band 2 (800 Hz boxiness)
/// // gr[2] = gain reduction in dB for band 3 (3 kHz harshness)
/// // gr[3] = gain reduction in dB for band 4 (6.5 kHz sibilance)
/// ```
pub struct DynamicEqProcessor {
    bands: Vec<DynamicEqBand>,
    sidechain_hpf: Option<BiquadFilter>,
    mix: f32,
}

impl DynamicEqProcessor {
    /// Create a new dynamic EQ processor.
    ///
    /// # Arguments
    ///
    /// * `config` - Dynamic EQ configuration with band definitions.
    /// * `sample_rate` - Audio sample rate in Hz.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is invalid.
    pub fn new(config: &DynamicEqConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;

        if !sample_rate.is_finite() || !(8000.0..=192000.0).contains(&sample_rate) {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!(
                    "sample_rate = {sample_rate}: must be finite and in [8000, 192000]",
                ),
            });
        }

        let bands = config
            .bands
            .iter()
            .map(|bc| DynamicEqBand::new(bc, sample_rate))
            .collect();

        let sidechain_hpf = match config.sidechain_filter {
            SidechainFilter::HighPass(freq) => {
                Some(BiquadFilter::new(highpass_coefficients(freq, sample_rate)))
            }
            SidechainFilter::None => None,
        };

        Ok(Self {
            bands,
            sidechain_hpf,
            mix: config.mix,
        })
    }

    /// Process an audio buffer in place.
    ///
    /// For each sample:
    /// 1. Optionally apply sidechain HPF to the detection copy.
    /// 2. Each band detects level and computes gain reduction from the
    ///    sidechain signal.
    /// 3. Each band applies its peaking EQ (with dynamic gain) to the
    ///    main signal in series.
    /// 4. Dry/wet mix blends the result with the original.
    pub fn process(&mut self, audio: &mut [f32]) {
        // Fast path: fully dry means no processing.
        if self.mix < 1e-6 {
            return;
        }

        let fully_wet = self.mix > (1.0 - 1e-6);

        for sample in audio.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                continue;
            }

            let dry = *sample;

            // Sidechain path: optionally HPF the detection signal.
            let sidechain = if let Some(ref mut hpf) = self.sidechain_hpf {
                hpf.process(dry)
            } else {
                dry
            };

            // Process through each band in series.
            let mut wet = dry;
            for band in &mut self.bands {
                let gr_db = band.detect_and_compute_gr(sidechain);
                wet = band.apply(wet, gr_db);
            }

            // NaN/Inf safety.
            if !wet.is_finite() {
                wet = 0.0;
            }

            // Dry/wet mix.
            *sample = if fully_wet {
                wet
            } else {
                dry * (1.0 - self.mix) + wet * self.mix
            };
        }
    }

    /// Reset all filter and envelope state.
    ///
    /// Call between audio segments to prevent state leakage.
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
        if let Some(ref mut hpf) = self.sidechain_hpf {
            hpf.reset();
        }
    }

    /// Get per-band gain reduction in dB.
    ///
    /// Returns a vector with one entry per band. Values are positive
    /// (e.g. 3.0 means 3 dB of gain reduction is being applied).
    /// Useful for metering and visualization.
    #[must_use]
    pub fn get_gain_reduction(&self) -> Vec<f32> {
        self.bands.iter().map(DynamicEqBand::gain_reduction_db).collect()
    }

    /// Get the number of bands.
    #[must_use]
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// Get the current dry/wet mix.
    #[must_use]
    pub fn mix(&self) -> f32 {
        self.mix
    }

    /// Set the dry/wet mix (0.0 = dry, 1.0 = wet).
    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        let config = DynamicEqConfig::default();
        config.validate().expect("default config should validate");
    }

    #[test]
    fn test_vocal_chorus_preset_creates_processor() {
        let config = DynamicEqConfig::vocal_chorus();
        let proc = DynamicEqProcessor::new(&config, KOKORO_SAMPLE_RATE as f32);
        assert!(proc.is_ok());
        let proc = proc.unwrap();
        assert_eq!(proc.num_bands(), 4);
    }

    #[test]
    fn test_silence_passes_through_unchanged() {
        let config = DynamicEqConfig::vocal_chorus();
        let mut proc =
            DynamicEqProcessor::new(&config, KOKORO_SAMPLE_RATE as f32).expect("valid config");
        let mut audio = vec![0.0f32; 512];
        proc.process(&mut audio);
        for &s in &audio {
            assert!(s.abs() < 1e-10, "silence should remain silent");
        }
    }

    #[test]
    fn test_no_gain_reduction_below_threshold() {
        // Very quiet signal should not trigger any gain reduction.
        let config = DynamicEqConfig::vocal_chorus();
        let mut proc =
            DynamicEqProcessor::new(&config, KOKORO_SAMPLE_RATE as f32).expect("valid config");

        // -60 dBFS signal: well below all thresholds.
        let amplitude = 10.0f32.powf(-60.0 / 20.0); // 0.001
        let mut audio: Vec<f32> = (0..2048)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 24000.0).sin())
            .collect();
        proc.process(&mut audio);

        let gr = proc.get_gain_reduction();
        for &g in &gr {
            assert!(g < 0.5, "quiet signal should have minimal GR, got {g} dB");
        }
    }

    #[test]
    fn test_loud_signal_triggers_gain_reduction() {
        // Loud 3 kHz tone should trigger band 3 (harshness control).
        let config = DynamicEqConfig::vocal_chorus();
        let mut proc =
            DynamicEqProcessor::new(&config, KOKORO_SAMPLE_RATE as f32).expect("valid config");

        // -6 dBFS at 3 kHz: well above threshold of -20 dB.
        let amplitude = 10.0f32.powf(-6.0 / 20.0); // ~0.5
        let mut audio: Vec<f32> = (0..4096)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * 3000.0 * i as f32 / 24000.0).sin())
            .collect();
        proc.process(&mut audio);

        let gr = proc.get_gain_reduction();
        // Band 3 (index 2, 3 kHz) should have significant GR.
        assert!(
            gr[2] > 1.0,
            "loud 3 kHz tone should trigger band 3 GR, got {} dB",
            gr[2],
        );
    }

    #[test]
    fn test_nan_inf_handling() {
        let config = DynamicEqConfig::vocal_chorus();
        let mut proc =
            DynamicEqProcessor::new(&config, KOKORO_SAMPLE_RATE as f32).expect("valid config");
        let mut audio = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5, -0.3];
        proc.process(&mut audio);
        for &s in &audio {
            assert!(s.is_finite(), "output must be finite, got {s}");
        }
    }

    #[test]
    fn test_dry_wet_mix_zero_passes_dry() {
        let config = DynamicEqConfig::vocal_chorus().with_mix(0.0);
        let mut proc =
            DynamicEqProcessor::new(&config, KOKORO_SAMPLE_RATE as f32).expect("valid config");

        let original: Vec<f32> = (0..256).map(|i| 0.5 * (i as f32 * 0.1).sin()).collect();
        let mut audio = original.clone();
        proc.process(&mut audio);

        for (o, p) in original.iter().zip(audio.iter()) {
            assert!((o - p).abs() < 1e-6, "mix=0 should pass through unchanged");
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let config = DynamicEqConfig::vocal_chorus();
        let mut proc =
            DynamicEqProcessor::new(&config, KOKORO_SAMPLE_RATE as f32).expect("valid config");

        // Process some loud audio.
        let mut audio: Vec<f32> = (0..2048)
            .map(|i| 0.8 * (2.0 * std::f32::consts::PI * 3000.0 * i as f32 / 24000.0).sin())
            .collect();
        proc.process(&mut audio);

        // Reset should clear all GR.
        proc.reset();
        let gr = proc.get_gain_reduction();
        for &g in &gr {
            assert!(g.abs() < 1e-6, "GR should be zero after reset, got {g}");
        }
    }

    #[test]
    fn test_invalid_config_rejected() {
        // Empty bands.
        let config = DynamicEqConfig::new(vec![]);
        assert!(config.validate().is_err());

        // Too many bands.
        let band = DynamicEqBandConfig {
            frequency_hz: 1000.0,
            q: 1.0,
            threshold_db: -20.0,
            ratio: 2.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            gain_db: 0.0,
        };
        let config = DynamicEqConfig::new(vec![band; 7]);
        assert!(config.validate().is_err());

        // Invalid mix.
        let config = DynamicEqConfig::vocal_chorus().with_mix(1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_single_band_config() {
        let config = DynamicEqConfig::new(vec![DynamicEqBandConfig {
            frequency_hz: 3000.0,
            q: 2.0,
            threshold_db: -15.0,
            ratio: 4.0,
            attack_ms: 2.0,
            release_ms: 50.0,
            gain_db: 0.0,
        }]);
        config.validate().expect("single band should validate");
        let proc = DynamicEqProcessor::new(&config, KOKORO_SAMPLE_RATE as f32);
        assert!(proc.is_ok());
        assert_eq!(proc.unwrap().num_bands(), 1);
    }

    #[test]
    fn test_peaking_eq_coefficients_unity_at_zero_db() {
        // 0 dB gain should produce unity filter (b0~1, b1~..., etc.).
        let coeffs = peaking_eq_coefficients(1000.0, 1.0, 0.0, 24000.0);
        // At 0 dB, A=1, so alpha*A = alpha/A, meaning b0=a0/a0=1 (after norm).
        assert!(
            (coeffs.b0 - 1.0).abs() < 0.01,
            "0 dB peaking EQ b0 should be ~1.0, got {}",
            coeffs.b0,
        );
    }

    #[test]
    fn test_sidechain_hpf_option() {
        // With HPF sidechain.
        let config =
            DynamicEqConfig::vocal_chorus().with_sidechain_filter(SidechainFilter::HighPass(100.0));
        let proc = DynamicEqProcessor::new(&config, KOKORO_SAMPLE_RATE as f32);
        assert!(proc.is_ok());

        // Without sidechain.
        let config = DynamicEqConfig::vocal_chorus().with_sidechain_filter(SidechainFilter::None);
        let proc = DynamicEqProcessor::new(&config, KOKORO_SAMPLE_RATE as f32);
        assert!(proc.is_ok());
    }
}
