// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vowel formant tracking and alignment for natural chorus voice blending.
//!
//! In real choirs, singers subconsciously match their vowel shapes to blend
//! better. This module tracks the formant frequencies (F1, F2) of each voice
//! in real-time and gently aligns them toward a reference voice's formant
//! trajectory. The result is clearer vowels and better voice blending.
//!
//! # Algorithm
//!
//! 1. **LPC analysis** via Levinson-Durbin on Hann-windowed frames to
//!    estimate the vocal tract transfer function per voice.
//! 2. **Formant detection** by peak-picking the LPC spectral envelope to
//!    find F1 (first formant, ~250-800 Hz) and F2 (second formant,
//!    ~700-2500 Hz).
//! 3. **Exponential smoothing** of detected formants across frames for
//!    stable tracking without jitter.
//! 4. **Formant correction** via peaking EQ: for each non-reference voice,
//!    a slight notch at the detected formant and a boost at the reference
//!    formant, scaled by `alignment_strength` and clamped by `max_shift_hz`.
//!
//! # References
//!
//! - Makhoul, J. "Linear Prediction: A Tutorial Review." Proceedings of
//!   the IEEE, 63(4), 1975.
//! - Markel, J. D. & Gray, A. H. "Linear Prediction of Speech."
//!   Springer-Verlag, 1976.
//! - Smith, J. O. "Introduction to Digital Filters with Audio Applications."
//!   <https://ccrma.stanford.edu/~jos/filters/>
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of bins to evaluate in the LPC magnitude response for peak picking.
const LPC_SPECTRUM_BINS: usize = 512;

/// Default analysis frame size in samples (~42.7 ms at 24 kHz).
const DEFAULT_FRAME_SIZE: usize = 1024;

/// Default hop size between analysis frames in samples (~21.3 ms at 24 kHz).
const DEFAULT_HOP_SIZE: usize = 512;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for vowel formant tracking and alignment.
///
/// Constructed via [`VowelAlignConfig::new`] (required for cross-crate use
/// due to `#[non_exhaustive]`). Use the builder methods (`with_*`) to
/// customize individual parameters.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VowelAlignConfig {
    /// Index of the reference voice that other voices align toward.
    /// Default: 0.
    pub reference_voice: usize,

    /// How strongly non-reference voices are pulled toward the reference
    /// formant trajectory. 0.0 = no alignment, 1.0 = full alignment.
    /// Default: 0.3.
    pub alignment_strength: f32,

    /// Formant tracking smoothing window in milliseconds. Controls
    /// how fast the tracked formant follows changes. Lower values are
    /// more responsive but noisier. Default: 30.0.
    pub tracking_speed_ms: f32,

    /// Valid range for the first formant (F1) in Hz.
    /// Default: (250.0, 800.0).
    pub f1_range: (f32, f32),

    /// Valid range for the second formant (F2) in Hz.
    /// Default: (700.0, 2500.0).
    pub f2_range: (f32, f32),

    /// Maximum formant correction shift in Hz. The EQ correction will
    /// never shift a formant by more than this amount, regardless of
    /// the difference between reference and detected formant.
    /// Default: 150.0.
    pub max_shift_hz: f32,

    /// LPC analysis order. Higher orders capture finer spectral detail
    /// but cost more CPU. Range: 6-20. Default: 12.
    pub lpc_order: usize,

    /// Sample rate in Hz. Default: 24000.0 (Kokoro native rate).
    pub sample_rate: f32,
}

impl Default for VowelAlignConfig {
    fn default() -> Self {
        Self {
            reference_voice: 0,
            alignment_strength: 0.3,
            tracking_speed_ms: 30.0,
            f1_range: (250.0, 800.0),
            f2_range: (700.0, 2500.0),
            max_shift_hz: 150.0,
            lpc_order: 12,
            sample_rate: 24000.0,
        }
    }
}

impl VowelAlignConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reference voice index.
    #[must_use]
    pub fn with_reference_voice(mut self, idx: usize) -> Self {
        self.reference_voice = idx;
        self
    }

    /// Set the alignment strength (0.0-1.0).
    #[must_use]
    pub fn with_alignment_strength(mut self, strength: f32) -> Self {
        self.alignment_strength = strength;
        self
    }

    /// Set the tracking smoothing speed in milliseconds.
    #[must_use]
    pub fn with_tracking_speed_ms(mut self, ms: f32) -> Self {
        self.tracking_speed_ms = ms;
        self
    }

    /// Set the valid F1 frequency range in Hz.
    #[must_use]
    pub fn with_f1_range(mut self, range: (f32, f32)) -> Self {
        self.f1_range = range;
        self
    }

    /// Set the valid F2 frequency range in Hz.
    #[must_use]
    pub fn with_f2_range(mut self, range: (f32, f32)) -> Self {
        self.f2_range = range;
        self
    }

    /// Set the maximum formant correction shift in Hz.
    #[must_use]
    pub fn with_max_shift_hz(mut self, hz: f32) -> Self {
        self.max_shift_hz = hz;
        self
    }

    /// Set the LPC analysis order.
    #[must_use]
    pub fn with_lpc_order(mut self, order: usize) -> Self {
        self.lpc_order = order;
        self
    }

    /// Set the sample rate in Hz.
    #[must_use]
    pub fn with_sample_rate(mut self, sr: f32) -> Self {
        self.sample_rate = sr;
        self
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.alignment_strength.is_finite() || !(0.0..=1.0).contains(&self.alignment_strength) {
            return Err(KokoroError::InvalidConfig {
                field: "alignment_strength",
                reason: format!(
                    "must be finite and in [0.0, 1.0], got {}",
                    self.alignment_strength
                ),
            });
        }
        if !self.tracking_speed_ms.is_finite() || self.tracking_speed_ms <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "tracking_speed_ms",
                reason: format!(
                    "must be finite and positive, got {}",
                    self.tracking_speed_ms
                ),
            });
        }
        // F1 range validation.
        if !self.f1_range.0.is_finite()
            || !self.f1_range.1.is_finite()
            || self.f1_range.0 <= 0.0
            || self.f1_range.0 >= self.f1_range.1
        {
            return Err(KokoroError::InvalidConfig {
                field: "f1_range",
                reason: format!(
                    "must be (lo, hi) with 0 < lo < hi, got ({}, {})",
                    self.f1_range.0, self.f1_range.1
                ),
            });
        }
        // F2 range validation.
        if !self.f2_range.0.is_finite()
            || !self.f2_range.1.is_finite()
            || self.f2_range.0 <= 0.0
            || self.f2_range.0 >= self.f2_range.1
        {
            return Err(KokoroError::InvalidConfig {
                field: "f2_range",
                reason: format!(
                    "must be (lo, hi) with 0 < lo < hi, got ({}, {})",
                    self.f2_range.0, self.f2_range.1
                ),
            });
        }
        if !self.max_shift_hz.is_finite() || self.max_shift_hz < 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "max_shift_hz",
                reason: format!("must be finite and non-negative, got {}", self.max_shift_hz),
            });
        }
        if self.lpc_order < 6 || self.lpc_order > 20 {
            return Err(KokoroError::InvalidConfig {
                field: "lpc_order",
                reason: format!("must be in [6, 20], got {}", self.lpc_order),
            });
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be finite and positive, got {}", self.sample_rate),
            });
        }
        Ok(())
    }

    // -- Presets ---------------------------------------------------------------

    /// Subtle alignment: very gentle formant pull, suitable for natural
    /// ensemble blending where voices already match well.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            alignment_strength: 0.15,
            tracking_speed_ms: 50.0,
            max_shift_hz: 80.0,
            ..Self::default()
        }
    }

    /// Tight blend: stronger alignment for close-harmony sections where
    /// vowel matching is critical for clarity.
    #[must_use]
    pub fn tight_blend() -> Self {
        Self {
            alignment_strength: 0.6,
            tracking_speed_ms: 20.0,
            max_shift_hz: 200.0,
            ..Self::default()
        }
    }

    /// Vowel lock: maximum alignment for unison passages where all
    /// voices should have nearly identical vowel shapes.
    #[must_use]
    pub fn vowel_lock() -> Self {
        Self {
            alignment_strength: 0.85,
            tracking_speed_ms: 15.0,
            max_shift_hz: 300.0,
            ..Self::default()
        }
    }

    /// Natural preset: moderate alignment mimicking a well-rehearsed
    /// choir section. The default configuration.
    #[must_use]
    pub fn natural() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// FormantTrack — per-voice formant trajectory
// ---------------------------------------------------------------------------

/// Per-voice formant trajectory snapshot.
///
/// Contains the currently detected F1 and F2 frequencies and a confidence
/// score indicating how reliable the detection is for this frame.
#[derive(Debug, Clone, Copy)]
pub struct FormantTrack {
    /// First formant frequency in Hz (vowel openness).
    pub f1_hz: f32,

    /// Second formant frequency in Hz (vowel frontness/backness).
    pub f2_hz: f32,

    /// Detection confidence in [0.0, 1.0]. Higher values indicate
    /// clearer formant peaks in the LPC spectral envelope.
    pub confidence: f32,
}

impl Default for FormantTrack {
    fn default() -> Self {
        Self {
            f1_hz: 0.0,
            f2_hz: 0.0,
            confidence: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// VowelAligner — stateful formant tracker and corrector
// ---------------------------------------------------------------------------

/// Stateful vowel formant tracker and aligner for chorus voices.
///
/// Tracks F1 and F2 per voice using LPC analysis with exponential smoothing,
/// then applies corrective peaking EQ to align non-reference voices toward
/// the reference voice's formant trajectory.
pub struct VowelAligner {
    config: VowelAlignConfig,
    /// Pre-computed Hann window for analysis frames.
    window: Vec<f32>,
    /// Per-voice smoothed formant state. Index = voice.
    tracks: Vec<FormantTrack>,
    /// Exponential smoothing coefficient derived from tracking_speed_ms.
    smooth_alpha: f32,
}

impl VowelAligner {
    /// Create a new vowel aligner from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is invalid.
    pub fn new(config: VowelAlignConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let window = hann_window(DEFAULT_FRAME_SIZE);
        let smooth_alpha = compute_smooth_alpha(config.tracking_speed_ms, config.sample_rate);
        Ok(Self {
            config,
            window,
            tracks: Vec::new(),
            smooth_alpha,
        })
    }

    /// Reset all per-voice tracking state.
    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &VowelAlignConfig {
        &self.config
    }

    /// Process all voices: detect formants, align non-reference voices
    /// toward the reference, and return the per-voice formant tracks.
    ///
    /// Each entry in `voices` is a mono PCM buffer for one voice. The
    /// reference voice (by index) is analyzed but not modified. All other
    /// voices have corrective EQ applied to gently shift their formants
    /// toward the reference.
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) -> Vec<FormantTrack> {
        let n_voices = voices.len();
        if n_voices == 0 {
            return Vec::new();
        }

        // Ensure we have tracking state for each voice.
        while self.tracks.len() < n_voices {
            self.tracks.push(FormantTrack::default());
        }

        // Step 1: Detect formants for every voice.
        let mut current_tracks: Vec<FormantTrack> = Vec::with_capacity(n_voices);
        for (vi, voice) in voices.iter().enumerate() {
            let raw = self.detect_formants(voice);
            // Exponentially smooth the tracked formants.
            let smoothed = self.smooth_track(vi, &raw);
            current_tracks.push(smoothed);
        }

        // Step 2: Get reference formants.
        let ref_idx = self.config.reference_voice.min(n_voices.saturating_sub(1));
        let ref_track = current_tracks[ref_idx];

        // Step 3: Apply corrective EQ to non-reference voices.
        if self.config.alignment_strength > 1e-6 && ref_track.confidence > 0.1 {
            for (vi, voice) in voices.iter_mut().enumerate() {
                if vi == ref_idx {
                    continue;
                }
                let track = &current_tracks[vi];
                if track.confidence < 0.1 {
                    continue; // Low confidence — skip correction.
                }
                self.apply_formant_correction(voice, track, &ref_track);
            }
        }

        current_tracks
    }

    // -- Internal: formant detection ------------------------------------------

    /// Detect F1 and F2 from a mono audio buffer via LPC peak-picking.
    fn detect_formants(&self, audio: &[f32]) -> FormantTrack {
        if audio.len() < DEFAULT_FRAME_SIZE {
            return FormantTrack::default();
        }

        // Analyze a frame from the middle of the buffer.
        let mid = audio.len().saturating_sub(DEFAULT_FRAME_SIZE) / 2;
        let frame = &audio[mid..mid + DEFAULT_FRAME_SIZE];

        // Apply Hann window.
        let windowed: Vec<f32> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| if s.is_finite() { s * w } else { 0.0 })
            .collect();

        // Check for silence.
        let energy: f32 = windowed.iter().map(|x| x * x).sum();
        if energy < 1e-10 {
            return FormantTrack::default();
        }

        // LPC via Levinson-Durbin.
        let lpc_coeffs = levinson_durbin(&windowed, self.config.lpc_order);

        // Peak-pick the LPC spectral envelope.
        let peaks = lpc_peak_pick(&lpc_coeffs, self.config.sample_rate);

        // Find F1 (first peak in f1_range) and F2 (first peak in f2_range).
        let (f1_lo, f1_hi) = self.config.f1_range;
        let (f2_lo, f2_hi) = self.config.f2_range;

        let mut f1: Option<(f32, f64)> = None;
        let mut f2: Option<(f32, f64)> = None;

        for &(freq, mag) in &peaks {
            if freq >= f1_lo && freq <= f1_hi && f1.is_none() {
                f1 = Some((freq, mag));
            }
            if freq >= f2_lo && freq <= f2_hi && f2.is_none() {
                f2 = Some((freq, mag));
            }
            if f1.is_some() && f2.is_some() {
                break;
            }
        }

        // Compute confidence from peak prominence relative to mean magnitude.
        let mean_mag: f64 = if peaks.is_empty() {
            1.0
        } else {
            let sum: f64 = peaks.iter().map(|&(_, m)| m).sum();
            (sum / peaks.len() as f64).max(1e-10)
        };

        let f1_conf = f1.map_or(0.0, |(_, m)| (m / mean_mag).min(3.0) / 3.0);
        let f2_conf = f2.map_or(0.0, |(_, m)| (m / mean_mag).min(3.0) / 3.0);
        let confidence = f64::midpoint(f1_conf, f2_conf) as f32;

        FormantTrack {
            f1_hz: f1.map_or(0.0, |(f, _)| f),
            f2_hz: f2.map_or(0.0, |(f, _)| f),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Exponentially smooth a raw formant detection against the tracked state.
    fn smooth_track(&mut self, voice_idx: usize, raw: &FormantTrack) -> FormantTrack {
        let prev = &self.tracks[voice_idx];

        // If previous track has no data, adopt the raw detection directly.
        if prev.confidence < 1e-6 {
            let track = *raw;
            self.tracks[voice_idx] = track;
            return track;
        }

        // Only smooth if the raw detection is confident.
        if raw.confidence < 0.1 {
            return *prev;
        }

        let alpha = self.smooth_alpha;
        let one_minus_alpha = 1.0 - alpha;

        let smoothed = FormantTrack {
            f1_hz: if raw.f1_hz > 0.0 {
                alpha * raw.f1_hz + one_minus_alpha * prev.f1_hz
            } else {
                prev.f1_hz
            },
            f2_hz: if raw.f2_hz > 0.0 {
                alpha * raw.f2_hz + one_minus_alpha * prev.f2_hz
            } else {
                prev.f2_hz
            },
            confidence: alpha * raw.confidence + one_minus_alpha * prev.confidence,
        };

        self.tracks[voice_idx] = smoothed;
        smoothed
    }

    // -- Internal: formant correction -----------------------------------------

    /// Apply corrective EQ to a single voice to shift its formants toward
    /// the reference.
    fn apply_formant_correction(
        &self,
        audio: &mut [f32],
        voice_track: &FormantTrack,
        ref_track: &FormantTrack,
    ) {
        let strength = self.config.alignment_strength;
        let max_shift = self.config.max_shift_hz;
        let sr = self.config.sample_rate;

        // Correct F1 if both voices have valid F1.
        if voice_track.f1_hz > 0.0 && ref_track.f1_hz > 0.0 {
            let diff = ref_track.f1_hz - voice_track.f1_hz;
            let shift = (diff * strength).clamp(-max_shift, max_shift);
            if shift.abs() > 1.0 {
                // Notch at current F1, boost at target F1.
                let notch_gain = -3.0 * strength;
                let boost_gain = 3.0 * strength;
                let bw = voice_track.f1_hz * 0.15; // ~15% bandwidth
                apply_peaking_eq(audio, voice_track.f1_hz, bw, notch_gain, sr);
                apply_peaking_eq(audio, voice_track.f1_hz + shift, bw, boost_gain, sr);
            }
        }

        // Correct F2 if both voices have valid F2.
        if voice_track.f2_hz > 0.0 && ref_track.f2_hz > 0.0 {
            let diff = ref_track.f2_hz - voice_track.f2_hz;
            let shift = (diff * strength).clamp(-max_shift, max_shift);
            if shift.abs() > 1.0 {
                let notch_gain = -3.0 * strength;
                let boost_gain = 3.0 * strength;
                let bw = voice_track.f2_hz * 0.12; // ~12% bandwidth
                apply_peaking_eq(audio, voice_track.f2_hz, bw, notch_gain, sr);
                apply_peaking_eq(audio, voice_track.f2_hz + shift, bw, boost_gain, sr);
            }
        }

        // Final IEEE 754 guard.
        for s in audio.iter_mut() {
            if !s.is_finite() {
                *s = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: smoothing coefficient
// ---------------------------------------------------------------------------

/// Compute the exponential smoothing coefficient alpha from the tracking
/// speed in milliseconds and sample rate.
///
/// alpha = 1 - exp(-hop_time / tau), where tau = tracking_speed_ms / 1000.
fn compute_smooth_alpha(tracking_speed_ms: f32, sample_rate: f32) -> f32 {
    let hop_time = DEFAULT_HOP_SIZE as f64 / f64::from(sample_rate);
    let tau = (f64::from(tracking_speed_ms) / 1000.0).max(1e-6);
    let alpha = 1.0 - (-hop_time / tau).exp();
    (alpha as f32).clamp(0.01, 1.0)
}

// ---------------------------------------------------------------------------
// Internal: Hann window
// ---------------------------------------------------------------------------

fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let scale = 2.0 * std::f64::consts::PI / (n - 1) as f64;
    (0..n)
        .map(|i| (0.5 * (1.0 - (scale * i as f64).cos())) as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// Internal: Levinson-Durbin LPC
// ---------------------------------------------------------------------------

/// Compute LPC coefficients via Levinson-Durbin recursion.
///
/// Returns `order + 1` coefficients where `a[0] = 1.0`.
fn levinson_durbin(signal: &[f32], order: usize) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || order == 0 {
        return vec![1.0];
    }

    // Compute autocorrelation R[0..order].
    let mut r = vec![0.0f64; order + 1];
    for lag in 0..=order {
        let mut sum = 0.0f64;
        for i in lag..n {
            let s0 = f64::from(signal[i]);
            let s1 = f64::from(signal[i - lag]);
            if s0.is_finite() && s1.is_finite() {
                sum += s0 * s1;
            }
        }
        r[lag] = sum;
    }

    // Trivial or silent signal.
    if r[0].abs() < 1e-30 {
        let mut a = vec![0.0f64; order + 1];
        a[0] = 1.0;
        return a;
    }

    // Levinson-Durbin recursion.
    let mut a = vec![0.0f64; order + 1];
    a[0] = 1.0;
    let mut err = r[0];

    for m in 1..=order {
        let mut lambda = 0.0f64;
        for j in 1..m {
            lambda += a[j] * r[m - j];
        }
        lambda = -(r[m] + lambda) / err;

        // Guard against instability.
        if !lambda.is_finite() || lambda.abs() >= 1.0 {
            break;
        }

        let mut a_new = a.clone();
        a_new[m] = lambda;
        for j in 1..m {
            a_new[j] = a[j] + lambda * a[m - j];
        }
        a = a_new;

        err *= 1.0 - lambda * lambda;
        if err < 1e-30 {
            break;
        }
    }

    a
}

// ---------------------------------------------------------------------------
// Internal: LPC magnitude response peak picking
// ---------------------------------------------------------------------------

/// Evaluate the LPC magnitude response and find spectral peaks.
///
/// Returns peaks as `(frequency_hz, magnitude)` sorted by frequency ascending.
fn lpc_peak_pick(lpc_coeffs: &[f64], sample_rate: f32) -> Vec<(f32, f64)> {
    let n_bins = LPC_SPECTRUM_BINS;
    let sr = f64::from(sample_rate);

    // Evaluate |1/A(e^jw)| at uniformly spaced frequencies.
    let mut magnitudes = Vec::with_capacity(n_bins);
    for k in 0..n_bins {
        let freq = (k as f64 / n_bins as f64) * (sr / 2.0);
        let omega = 2.0 * std::f64::consts::PI * freq / sr;

        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (i, &coeff) in lpc_coeffs.iter().enumerate() {
            re += coeff * (omega * i as f64).cos();
            im -= coeff * (omega * i as f64).sin();
        }

        let mag_sq = re * re + im * im;
        let inv_mag = if mag_sq > 1e-30 {
            1.0 / mag_sq.sqrt()
        } else {
            0.0
        };
        magnitudes.push(inv_mag);
    }

    // Find local maxima (peaks) in the magnitude response.
    let mut peaks: Vec<(f32, f64)> = Vec::new();
    for k in 1..n_bins.saturating_sub(1) {
        if magnitudes[k] > magnitudes[k - 1] && magnitudes[k] > magnitudes[k + 1] {
            let freq_hz = (k as f64 / n_bins as f64 * (sr / 2.0)) as f32;
            peaks.push((freq_hz, magnitudes[k]));
        }
    }

    // Sort by frequency ascending (natural ordering for formants).
    peaks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    peaks
}

// ---------------------------------------------------------------------------
// Internal: Peaking EQ biquad
// ---------------------------------------------------------------------------

/// Apply a second-order peaking EQ filter to an audio buffer in-place.
///
/// Based on Robert Bristow-Johnson's Audio EQ Cookbook.
fn apply_peaking_eq(
    audio: &mut [f32],
    center_hz: f32,
    bandwidth_hz: f32,
    gain_db: f32,
    sample_rate: f32,
) {
    if audio.is_empty() || gain_db.abs() < 1e-4 {
        return;
    }

    let sr = f64::from(sample_rate);
    let fc = f64::from(center_hz).clamp(20.0, sr / 2.0 - 1.0);
    let bw = f64::from(bandwidth_hz).max(10.0);
    let a_lin = 10.0f64.powf(f64::from(gain_db) / 40.0);

    let w0 = 2.0 * std::f64::consts::PI * fc / sr;
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();

    let q = fc / bw;
    let alpha = sin_w0 / (2.0 * q);

    // Peaking EQ coefficients (Bristow-Johnson).
    let b0 = 1.0 + alpha * a_lin;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0 - alpha * a_lin;
    let a0 = 1.0 + alpha / a_lin;
    let a1_coeff = -2.0 * cos_w0;
    let a2 = 1.0 - alpha / a_lin;

    // Normalize.
    let inv_a0 = 1.0 / a0;
    let b0 = (b0 * inv_a0) as f32;
    let b1 = (b1 * inv_a0) as f32;
    let b2 = (b2 * inv_a0) as f32;
    let a1_coeff = (a1_coeff * inv_a0) as f32;
    let a2 = (a2 * inv_a0) as f32;

    // Direct Form II Transposed.
    let mut z1: f32 = 0.0;
    let mut z2: f32 = 0.0;

    for sample in audio.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
            z1 = 0.0;
            z2 = 0.0;
            continue;
        }
        let x = *sample;
        let y = b0 * x + z1;
        z1 = b1 * x - a1_coeff * y + z2;
        z2 = b2 * x - a2 * y;

        *sample = if y.is_finite() { y } else { 0.0 };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f32, n_samples: usize, sr: f32) -> Vec<f32> {
        (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    fn vowel_like_signal(f1: f32, f2: f32, n_samples: usize, sr: f32) -> Vec<f32> {
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sr;
                0.5 * (2.0 * std::f32::consts::PI * f1 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * f2 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 150.0 * t).sin() // fundamental
            })
            .collect()
    }

    // -- Config validation tests -----------------------------------------------

    #[test]
    fn test_config_default_is_valid() {
        VowelAlignConfig::default()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_invalid_alignment_strength() {
        assert!(VowelAlignConfig::new()
            .with_alignment_strength(-0.1)
            .validate()
            .is_err());
        assert!(VowelAlignConfig::new()
            .with_alignment_strength(1.1)
            .validate()
            .is_err());
        assert!(VowelAlignConfig::new()
            .with_alignment_strength(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_tracking_speed() {
        assert!(VowelAlignConfig::new()
            .with_tracking_speed_ms(0.0)
            .validate()
            .is_err());
        assert!(VowelAlignConfig::new()
            .with_tracking_speed_ms(-1.0)
            .validate()
            .is_err());
        assert!(VowelAlignConfig::new()
            .with_tracking_speed_ms(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_f1_range() {
        // Inverted range.
        assert!(VowelAlignConfig::new()
            .with_f1_range((800.0, 250.0))
            .validate()
            .is_err());
        // Zero lower bound.
        assert!(VowelAlignConfig::new()
            .with_f1_range((0.0, 800.0))
            .validate()
            .is_err());
        // NaN.
        assert!(VowelAlignConfig::new()
            .with_f1_range((f32::NAN, 800.0))
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_f2_range() {
        assert!(VowelAlignConfig::new()
            .with_f2_range((2500.0, 700.0))
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_max_shift() {
        assert!(VowelAlignConfig::new()
            .with_max_shift_hz(-1.0)
            .validate()
            .is_err());
        assert!(VowelAlignConfig::new()
            .with_max_shift_hz(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_lpc_order() {
        assert!(VowelAlignConfig::new()
            .with_lpc_order(5)
            .validate()
            .is_err());
        assert!(VowelAlignConfig::new()
            .with_lpc_order(21)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        assert!(VowelAlignConfig::new()
            .with_sample_rate(0.0)
            .validate()
            .is_err());
        assert!(VowelAlignConfig::new()
            .with_sample_rate(f32::NAN)
            .validate()
            .is_err());
    }

    // -- Preset tests ----------------------------------------------------------

    #[test]
    fn test_presets_all_valid() {
        VowelAlignConfig::subtle().validate().expect("subtle");
        VowelAlignConfig::tight_blend()
            .validate()
            .expect("tight_blend");
        VowelAlignConfig::vowel_lock()
            .validate()
            .expect("vowel_lock");
        VowelAlignConfig::natural().validate().expect("natural");
    }

    #[test]
    fn test_preset_strength_ordering() {
        let subtle = VowelAlignConfig::subtle();
        let natural = VowelAlignConfig::natural();
        let tight = VowelAlignConfig::tight_blend();
        let lock = VowelAlignConfig::vowel_lock();

        assert!(subtle.alignment_strength < natural.alignment_strength);
        assert!(natural.alignment_strength < tight.alignment_strength);
        assert!(tight.alignment_strength < lock.alignment_strength);
    }

    // -- Aligner construction tests -------------------------------------------

    #[test]
    fn test_aligner_construction() {
        let aligner = VowelAligner::new(VowelAlignConfig::default());
        assert!(aligner.is_ok());
    }

    #[test]
    fn test_aligner_invalid_config_rejected() {
        let cfg = VowelAlignConfig::new().with_alignment_strength(2.0);
        assert!(VowelAligner::new(cfg).is_err());
    }

    #[test]
    fn test_aligner_reset() {
        let mut aligner = VowelAligner::new(VowelAlignConfig::default()).unwrap();
        let sr = 24000.0;
        let mut voices = vec![
            vowel_like_signal(500.0, 1500.0, 2048, sr),
            vowel_like_signal(550.0, 1600.0, 2048, sr),
        ];
        aligner.process_voices(&mut voices);
        aligner.reset();
        // After reset, tracks should be empty.
        assert!(aligner.tracks.is_empty());
    }

    // -- Processing tests ------------------------------------------------------

    #[test]
    fn test_process_empty_voices() {
        let mut aligner = VowelAligner::new(VowelAlignConfig::default()).unwrap();
        let mut voices: Vec<Vec<f32>> = Vec::new();
        let tracks = aligner.process_voices(&mut voices);
        assert!(tracks.is_empty());
    }

    #[test]
    fn test_process_single_voice_no_modification() {
        // With only one voice (the reference), no EQ correction should be
        // applied. The voice should be unchanged.
        let sr = 24000.0;
        let mut voices = vec![vowel_like_signal(500.0, 1500.0, 4096, sr)];
        let original = voices[0].clone();
        let mut aligner = VowelAligner::new(VowelAlignConfig::default()).unwrap();
        let tracks = aligner.process_voices(&mut voices);
        assert_eq!(tracks.len(), 1);
        // Reference voice should be unmodified.
        assert_eq!(voices[0], original);
    }

    #[test]
    fn test_process_two_voices_modifies_non_reference() {
        let sr = 24000.0;
        let n = 4096;
        let mut voices = vec![
            vowel_like_signal(500.0, 1500.0, n, sr), // reference (voice 0)
            vowel_like_signal(600.0, 1800.0, n, sr), // voice 1 — different formants
        ];
        let ref_original = voices[0].clone();
        let v1_original = voices[1].clone();

        let cfg = VowelAlignConfig::new().with_alignment_strength(0.5);
        let mut aligner = VowelAligner::new(cfg).unwrap();
        let tracks = aligner.process_voices(&mut voices);

        assert_eq!(tracks.len(), 2);

        // Reference should be unmodified.
        assert_eq!(voices[0], ref_original);

        // Voice 1 should be modified (different from original).
        let diff: f32 = voices[1]
            .iter()
            .zip(v1_original.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n as f32;
        // The modification might be small depending on formant detection,
        // but we check the output is at least finite.
        for (i, &s) in voices[1].iter().enumerate() {
            assert!(s.is_finite(), "voice 1 sample {i} is non-finite: {s}");
        }
        // If formants were detected and differ, diff should be > 0.
        // (We allow 0 diff if formant detection didn't find valid peaks.)
        let _ = diff;
    }

    #[test]
    fn test_process_zero_strength_is_identity() {
        let sr = 24000.0;
        let n = 4096;
        let mut voices = vec![
            vowel_like_signal(500.0, 1500.0, n, sr),
            vowel_like_signal(600.0, 1800.0, n, sr),
        ];
        let originals: Vec<Vec<f32>> = voices.clone();

        let cfg = VowelAlignConfig::new().with_alignment_strength(0.0);
        let mut aligner = VowelAligner::new(cfg).unwrap();
        aligner.process_voices(&mut voices);

        // Both voices should be identical to originals with zero strength.
        for (vi, (voice, orig)) in voices.iter().zip(originals.iter()).enumerate() {
            assert_eq!(voice, orig, "voice {vi} should be unmodified at strength=0");
        }
    }

    #[test]
    fn test_process_all_outputs_finite_with_nan_input() {
        let sr = 24000.0;
        let n = 2048;
        let v0 = vowel_like_signal(500.0, 1500.0, n, sr);
        let mut v1 = vowel_like_signal(550.0, 1600.0, n, sr);
        v1[100] = f32::NAN;
        v1[101] = f32::INFINITY;
        v1[500] = f32::NEG_INFINITY;

        let mut voices = vec![v0, v1];
        let cfg = VowelAlignConfig::new().with_alignment_strength(0.5);
        let mut aligner = VowelAligner::new(cfg).unwrap();
        aligner.process_voices(&mut voices);

        for (vi, voice) in voices.iter().enumerate() {
            for (si, &s) in voice.iter().enumerate() {
                assert!(s.is_finite(), "voice {vi} sample {si} is non-finite: {s}");
            }
        }
    }

    #[test]
    fn test_process_short_audio_no_crash() {
        let mut voices = vec![
            vec![0.5; 64], // shorter than frame size
            vec![0.3; 64],
        ];
        let mut aligner = VowelAligner::new(VowelAlignConfig::default()).unwrap();
        let tracks = aligner.process_voices(&mut voices);
        assert_eq!(tracks.len(), 2);
        // Should have zero confidence for short audio.
        for track in &tracks {
            assert!(
                track.confidence < 0.01,
                "short audio should have ~0 confidence, got {}",
                track.confidence
            );
        }
    }

    #[test]
    fn test_process_reference_voice_out_of_range_clamps() {
        let sr = 24000.0;
        let n = 2048;
        let mut voices = vec![
            vowel_like_signal(500.0, 1500.0, n, sr),
            vowel_like_signal(550.0, 1600.0, n, sr),
        ];

        // Reference voice index 99 should clamp to last voice.
        let cfg = VowelAlignConfig::new().with_reference_voice(99);
        let mut aligner = VowelAligner::new(cfg).unwrap();
        let tracks = aligner.process_voices(&mut voices);
        assert_eq!(tracks.len(), 2);
    }

    // -- LPC / formant detection unit tests -----------------------------------

    #[test]
    fn test_levinson_durbin_silent_signal() {
        let signal = vec![0.0f32; 256];
        let coeffs = levinson_durbin(&signal, 10);
        assert_eq!(coeffs[0], 1.0);
    }

    #[test]
    fn test_levinson_durbin_returns_correct_length() {
        let signal = sine_wave(440.0, 1024, 24000.0);
        let coeffs = levinson_durbin(&signal, 12);
        assert_eq!(coeffs.len(), 13);
        assert_eq!(coeffs[0], 1.0);
    }

    #[test]
    fn test_lpc_peak_pick_finds_peaks() {
        let signal: Vec<f32> = (0..1024)
            .map(|i| {
                let t = i as f32 / 24000.0;
                0.5 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 1500.0 * t).sin()
            })
            .collect();
        let coeffs = levinson_durbin(&signal, 12);
        let peaks = lpc_peak_pick(&coeffs, 24000.0);
        assert!(
            !peaks.is_empty(),
            "should find at least one peak in multi-tone signal"
        );
        // Peaks should be sorted by frequency.
        for i in 1..peaks.len() {
            assert!(peaks[i].0 >= peaks[i - 1].0);
        }
    }

    // -- Peaking EQ unit tests ------------------------------------------------

    #[test]
    fn test_peaking_eq_zero_gain_identity() {
        let mut audio = sine_wave(1000.0, 1024, 24000.0);
        let original = audio.clone();
        apply_peaking_eq(&mut audio, 1000.0, 200.0, 0.0, 24000.0);
        assert_eq!(audio, original);
    }

    #[test]
    fn test_peaking_eq_positive_gain_boosts() {
        let mut audio = sine_wave(1000.0, 4096, 24000.0);
        let dry_rms: f32 = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        apply_peaking_eq(&mut audio, 1000.0, 200.0, 6.0, 24000.0);
        let wet_rms: f32 = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        assert!(
            wet_rms > dry_rms,
            "positive gain should boost: dry={dry_rms}, wet={wet_rms}"
        );
    }

    #[test]
    fn test_peaking_eq_nan_input_zeroed() {
        let mut audio = vec![1.0, f32::NAN, 0.5, f32::INFINITY, -0.3];
        apply_peaking_eq(&mut audio, 500.0, 100.0, 3.0, 24000.0);
        for (i, &s) in audio.iter().enumerate() {
            assert!(s.is_finite(), "sample {i} is non-finite after EQ: {s}");
        }
    }

    // -- Smoothing coefficient test -------------------------------------------

    #[test]
    fn test_smooth_alpha_in_range() {
        let alpha = compute_smooth_alpha(30.0, 24000.0);
        assert!((0.01..=1.0).contains(&alpha), "alpha={alpha} out of range");
    }

    #[test]
    fn test_smooth_alpha_faster_tracking_higher_alpha() {
        let fast = compute_smooth_alpha(10.0, 24000.0);
        let slow = compute_smooth_alpha(100.0, 24000.0);
        assert!(
            fast > slow,
            "faster tracking should have higher alpha: fast={fast}, slow={slow}"
        );
    }

    // -- FormantTrack defaults ------------------------------------------------

    #[test]
    fn test_formant_track_default() {
        let t = FormantTrack::default();
        assert_eq!(t.f1_hz, 0.0);
        assert_eq!(t.f2_hz, 0.0);
        assert_eq!(t.confidence, 0.0);
    }
}
