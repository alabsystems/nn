// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Harmonic series analyzer and tuner for clean voice stacking in chorus.
//!
//! When multiple chorus voices sing the same note, their harmonics should
//! reinforce cleanly rather than create muddy interference. This module
//! analyzes the harmonic series of each voice and gently adjusts individual
//! harmonics for constructive stacking.
//!
//! # Algorithm
//!
//! 1. **F0 detection** via normalized autocorrelation with parabolic
//!    interpolation to find the fundamental frequency of each voice.
//! 2. **Harmonic extraction** using windowed FFT: for each harmonic
//!    k * F0, read magnitude and phase from the nearest DFT bin.
//! 3. **Phase alignment**: when `fundamental_lock` is true, align all
//!    voices' fundamental phases. For higher harmonics, introduce
//!    controlled phase offsets based on `harmonic_spread` to create
//!    width without destructive cancellation.
//! 4. **Odd/even balance**: gently boost odd harmonics (square-wave-like,
//!    bright) or even harmonics (octave-like, warm) via a balance knob.
//! 5. **Resynthesis** via overlap-add of the modified spectrum.
//!
//! # References
//!
//! - de Cheveigne, A. & Kawahara, H. "YIN, a fundamental frequency
//!   estimator for speech and music." JASA, 111(4), 2002.
//! - Serra, X. & Smith, J. O. "Spectral Modeling Synthesis: A Sound
//!   Analysis/Synthesis System Based on a Deterministic plus Stochastic
//!   Decomposition." Computer Music Journal, 14(4), 1990.
//! - Smith, J. O. "Mathematics of the DFT."
//!   <https://ccrma.stanford.edu/~jos/mdft/>
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum harmonics we support analyzing.
const MAX_HARMONICS: usize = 16;

/// Minimum plausible F0 for speech (Hz). Below this is likely noise.
const MIN_F0_HZ: f32 = 50.0;

/// Maximum plausible F0 for speech (Hz). Above this is unlikely fundamental.
const MAX_F0_HZ: f32 = 1000.0;

/// Autocorrelation voicing threshold. Below this the signal is unvoiced
/// and we skip harmonic processing.
const VOICING_THRESHOLD: f32 = 0.3;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the harmonic series analyzer and tuner.
///
/// Constructed via [`HarmonicTunerConfig::new`] and builder methods.
/// `#[non_exhaustive]` allows adding fields without breaking downstream.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HarmonicTunerConfig {
    /// Number of harmonics to analyze (1-16). Default: 8.
    pub n_harmonics: usize,

    /// Correction strength: 0.0 = no effect, 1.0 = full correction.
    /// Controls how strongly phase alignment and harmonic adjustments
    /// are applied. Default: 0.3.
    pub correction_strength: f32,

    /// Per-voice harmonic detuning for stereo width: 0.0 = all voices
    /// phase-locked, 1.0 = maximum spread across harmonics.
    /// Higher harmonics get proportionally more spread. Default: 0.1.
    pub harmonic_spread: f32,

    /// Odd/even harmonic balance: -1.0 = boost odd harmonics only
    /// (square-wave-like, bright), 1.0 = boost even harmonics only
    /// (octave-like, warm), 0.0 = natural (no rebalancing). Default: 0.0.
    pub odd_even_balance: f32,

    /// When true, align all voices' fundamental (1st harmonic) phases
    /// exactly. Higher harmonics get controlled offsets via
    /// `harmonic_spread`. Default: true.
    pub fundamental_lock: bool,

    /// FFT window size for harmonic analysis. Must be a power of 2,
    /// minimum 256. Default: 2048.
    pub window_size: usize,

    /// Sample rate in Hz. Default: 24000.0 (Kokoro native rate).
    pub sample_rate: f32,
}

impl Default for HarmonicTunerConfig {
    fn default() -> Self {
        Self {
            n_harmonics: 8,
            correction_strength: 0.3,
            harmonic_spread: 0.1,
            odd_even_balance: 0.0,
            fundamental_lock: true,
            window_size: 2048,
            sample_rate: KOKORO_SAMPLE_RATE as f32,
        }
    }
}

impl HarmonicTunerConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of harmonics to analyze.
    #[must_use]
    pub fn with_n_harmonics(mut self, n: usize) -> Self {
        self.n_harmonics = n;
        self
    }

    /// Set the correction strength (0.0 - 1.0).
    #[must_use]
    pub fn with_correction_strength(mut self, s: f32) -> Self {
        self.correction_strength = s;
        self
    }

    /// Set the harmonic spread for per-voice width (0.0 - 1.0).
    #[must_use]
    pub fn with_harmonic_spread(mut self, s: f32) -> Self {
        self.harmonic_spread = s;
        self
    }

    /// Set the odd/even harmonic balance (-1.0 to 1.0).
    #[must_use]
    pub fn with_odd_even_balance(mut self, b: f32) -> Self {
        self.odd_even_balance = b;
        self
    }

    /// Set fundamental lock on or off.
    #[must_use]
    pub fn with_fundamental_lock(mut self, lock: bool) -> Self {
        self.fundamental_lock = lock;
        self
    }

    /// Set the FFT window size (power of 2, >= 256).
    #[must_use]
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
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
        if self.n_harmonics == 0 || self.n_harmonics > MAX_HARMONICS {
            return Err(KokoroError::InvalidConfig {
                field: "n_harmonics",
                reason: format!("must be in [1, {MAX_HARMONICS}], got {}", self.n_harmonics),
            });
        }
        check_finite_range("correction_strength", self.correction_strength, 0.0, 1.0)?;
        check_finite_range("harmonic_spread", self.harmonic_spread, 0.0, 1.0)?;
        check_finite_range("odd_even_balance", self.odd_even_balance, -1.0, 1.0)?;
        if self.window_size < 256 || !self.window_size.is_power_of_two() {
            return Err(KokoroError::InvalidConfig {
                field: "window_size",
                reason: format!("must be a power of 2 and >= 256, got {}", self.window_size),
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

    /// Natural preset: gentle phase alignment with minimal coloring.
    #[must_use]
    pub fn natural() -> Self {
        Self {
            n_harmonics: 8,
            correction_strength: 0.2,
            harmonic_spread: 0.05,
            odd_even_balance: 0.0,
            fundamental_lock: true,
            window_size: 2048,
            sample_rate: KOKORO_SAMPLE_RATE as f32,
        }
    }

    /// Bright stack: boost odd harmonics for added presence and clarity.
    #[must_use]
    pub fn bright_stack() -> Self {
        Self {
            n_harmonics: 12,
            correction_strength: 0.5,
            harmonic_spread: 0.15,
            odd_even_balance: -0.4,
            fundamental_lock: true,
            window_size: 2048,
            sample_rate: KOKORO_SAMPLE_RATE as f32,
        }
    }

    /// Warm stack: boost even harmonics for octave-rich warmth.
    #[must_use]
    pub fn warm_stack() -> Self {
        Self {
            n_harmonics: 10,
            correction_strength: 0.4,
            harmonic_spread: 0.08,
            odd_even_balance: 0.5,
            fundamental_lock: true,
            window_size: 2048,
            sample_rate: KOKORO_SAMPLE_RATE as f32,
        }
    }

    /// Wide harmonics: maximum spread for spacious stereo image.
    #[must_use]
    pub fn wide_harmonics() -> Self {
        Self {
            n_harmonics: 12,
            correction_strength: 0.3,
            harmonic_spread: 0.6,
            odd_even_balance: 0.0,
            fundamental_lock: true,
            window_size: 2048,
            sample_rate: KOKORO_SAMPLE_RATE as f32,
        }
    }
}

fn check_finite_range(field: &'static str, val: f32, lo: f32, hi: f32) -> Result<(), KokoroError> {
    if !val.is_finite() || val < lo || val > hi {
        Err(KokoroError::InvalidConfig {
            field,
            reason: format!("must be finite and in [{lo}, {hi}], got {val}"),
        })
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Harmonic analysis result
// ---------------------------------------------------------------------------

/// Per-voice harmonic analysis result.
///
/// Contains the detected fundamental frequency and a list of harmonics
/// with their frequency, magnitude, and phase.
#[derive(Debug, Clone)]
pub struct HarmonicAnalysis {
    /// Detected fundamental frequency in Hz. 0.0 if unvoiced.
    pub fundamental_hz: f32,

    /// Harmonic series: `(frequency_hz, magnitude, phase_radians)`.
    /// Index 0 is the fundamental, index 1 is the 2nd harmonic, etc.
    /// Phase is in [-pi, pi].
    pub harmonics: Vec<(f32, f32, f32)>,
}

impl HarmonicAnalysis {
    /// True if the signal was detected as voiced (has a fundamental).
    #[must_use]
    pub fn is_voiced(&self) -> bool {
        self.fundamental_hz > 0.0
    }
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

/// Harmonic series analyzer and tuner for clean voice stacking.
///
/// Analyzes each chorus voice's harmonic content and adjusts phases
/// and magnitudes so that stacked voices reinforce cleanly.
pub struct HarmonicTunerProcessor {
    config: HarmonicTunerConfig,
    /// Pre-computed Hann window for FFT frames.
    window: Vec<f32>,
    /// Hop size for overlap-add (window_size / 4).
    hop_size: usize,
}

impl HarmonicTunerProcessor {
    /// Create a new harmonic tuner from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is invalid.
    pub fn new(config: HarmonicTunerConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let window = hann_window(config.window_size);
        let hop_size = config.window_size / 4;
        Ok(Self {
            config,
            window,
            hop_size,
        })
    }

    /// Create a harmonic tuner with default configuration.
    pub fn with_defaults() -> Result<Self, KokoroError> {
        Self::new(HarmonicTunerConfig::default())
    }

    /// Reset internal state. The harmonic tuner is stateless per-call,
    /// so this is provided for API consistency with other chorus processors.
    pub fn reset(&mut self) {
        // Harmonic analysis is frame-by-frame with no inter-frame state.
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &HarmonicTunerConfig {
        &self.config
    }

    /// Analyze the harmonic content of a single voice signal.
    ///
    /// Detects the fundamental via autocorrelation, then extracts harmonic
    /// magnitudes and phases from a windowed FFT of the signal center.
    #[must_use]
    pub fn analyze_voice(&self, audio: &[f32]) -> HarmonicAnalysis {
        if audio.len() < self.config.window_size {
            return HarmonicAnalysis {
                fundamental_hz: 0.0,
                harmonics: Vec::new(),
            };
        }

        // Take a frame from the center of the signal.
        let mid = audio.len().saturating_sub(self.config.window_size) / 2;
        let frame = &audio[mid..mid + self.config.window_size];

        // Detect F0 via autocorrelation.
        let f0 = detect_f0_autocorrelation(frame, self.config.sample_rate);
        if f0 < MIN_F0_HZ {
            return HarmonicAnalysis {
                fundamental_hz: 0.0,
                harmonics: Vec::new(),
            };
        }

        // Apply Hann window and compute DFT magnitudes/phases.
        let windowed: Vec<f32> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| if s.is_finite() { s * w } else { 0.0 })
            .collect();

        let (mags, phases) = compute_dft_mag_phase(&windowed);
        let bin_hz = self.config.sample_rate / self.config.window_size as f32;

        // Extract harmonic magnitudes and phases.
        let mut harmonics = Vec::with_capacity(self.config.n_harmonics);
        for k in 1..=self.config.n_harmonics {
            let freq = f0 * k as f32;
            if freq >= self.config.sample_rate / 2.0 {
                break;
            }
            let bin = (freq / bin_hz).round() as usize;
            if bin < mags.len() {
                harmonics.push((freq, mags[bin], phases[bin]));
            }
        }

        HarmonicAnalysis {
            fundamental_hz: f0,
            harmonics,
        }
    }

    /// Process all chorus voices for clean harmonic stacking.
    ///
    /// For each voice:
    /// 1. Detect F0 via autocorrelation.
    /// 2. Extract harmonic magnitudes via FFT.
    /// 3. Align fundamental phases across voices (if `fundamental_lock`).
    /// 4. Apply controlled phase offsets for width (`harmonic_spread`).
    /// 5. Rebalance odd/even harmonics (`odd_even_balance`).
    /// 6. Resynthesize via overlap-add.
    pub fn process_voices(&mut self, voices: &mut [Vec<f32>]) {
        if voices.is_empty() || self.config.correction_strength < 1e-6 {
            return;
        }

        let n_voices = voices.len();
        let ws = self.config.window_size;

        // Find the minimum length across all voices.
        let min_len = voices.iter().map(Vec::len).min().unwrap_or(0);
        if min_len < ws {
            return;
        }

        // Analyze all voices to find their fundamentals.
        let analyses: Vec<HarmonicAnalysis> =
            voices.iter().map(|v| self.analyze_voice(v)).collect();

        // Find the reference phase from the first voiced signal.
        let ref_analysis = analyses.iter().find(|a| a.is_voiced());
        let ref_phases: Option<Vec<f32>> =
            ref_analysis.map(|a| a.harmonics.iter().map(|&(_, _, p)| p).collect());

        // Process each voice frame-by-frame with overlap-add.
        let strength = self.config.correction_strength;

        for (voice_idx, voice) in voices.iter_mut().enumerate() {
            let analysis = &analyses[voice_idx];
            if !analysis.is_voiced() {
                continue;
            }

            let f0 = analysis.fundamental_hz;
            let bin_hz = self.config.sample_rate / ws as f32;

            // Process in overlapping frames.
            let n_frames = (voice.len().saturating_sub(ws)) / self.hop_size + 1;
            let mut output = vec![0.0f32; voice.len()];
            let mut norm = vec![0.0f32; voice.len()];

            for frame_idx in 0..n_frames {
                let start = frame_idx * self.hop_size;
                let end = (start + ws).min(voice.len());
                if end - start < ws {
                    break;
                }

                // Window the frame.
                let mut re: Vec<f32> = voice[start..end]
                    .iter()
                    .zip(self.window.iter())
                    .map(|(&s, &w)| if s.is_finite() { s * w } else { 0.0 })
                    .collect();
                let mut im = vec![0.0f32; ws];

                // Forward DFT (in-place).
                dft_forward(&mut re, &mut im);

                // Modify harmonic bins.
                for k in 1..=self.config.n_harmonics {
                    let freq = f0 * k as f32;
                    if freq >= self.config.sample_rate / 2.0 {
                        break;
                    }
                    let bin = (freq / bin_hz).round() as usize;
                    if bin >= ws / 2 {
                        break;
                    }

                    let mag = re[bin].hypot(im[bin]);
                    let phase = im[bin].atan2(re[bin]);

                    // --- Phase alignment ---
                    let mut target_phase = phase;
                    if let Some(ref rp) = ref_phases {
                        if k <= rp.len() {
                            let ref_p = rp[k - 1];
                            if self.config.fundamental_lock && k == 1 {
                                // Lock fundamental exactly to reference.
                                target_phase = ref_p;
                            } else {
                                // Blend toward reference with controlled spread.
                                let voice_offset = voice_idx as f32
                                    * self.config.harmonic_spread
                                    * k as f32
                                    * std::f32::consts::PI
                                    / n_voices.max(1) as f32;
                                target_phase = ref_p + voice_offset;
                            }
                        }
                    }

                    // Interpolate between original and target phase.
                    let new_phase = lerp_angle(phase, target_phase, strength);

                    // --- Odd/even balance ---
                    let mut new_mag = mag;
                    let balance = self.config.odd_even_balance;
                    if balance.abs() > 1e-6 {
                        let is_odd = k % 2 == 1;
                        // Odd harmonics: k=1,3,5,... Even: k=2,4,6,...
                        // balance < 0 boosts odd, balance > 0 boosts even.
                        let gain_db = if is_odd {
                            -balance * 3.0 * strength
                        } else {
                            balance * 3.0 * strength
                        };
                        new_mag *= db_to_linear(gain_db);
                    }

                    // Write back modified bin.
                    re[bin] = new_mag * new_phase.cos();
                    im[bin] = new_mag * new_phase.sin();

                    // Mirror for negative frequencies.
                    let mirror = ws - bin;
                    if mirror < ws && mirror != bin {
                        re[mirror] = re[bin];
                        im[mirror] = -im[bin];
                    }
                }

                // Inverse DFT.
                dft_inverse(&mut re, &mut im);

                // Overlap-add with synthesis window.
                for i in 0..ws {
                    let idx = start + i;
                    if idx < output.len() {
                        output[idx] += re[i] * self.window[i];
                        norm[idx] += self.window[i] * self.window[i];
                    }
                }
            }

            // Normalize overlap-add and blend with dry signal.
            for i in 0..voice.len() {
                let wet = if norm[i] > 1e-8 {
                    output[i] / norm[i]
                } else {
                    0.0
                };
                // Blend: voice = dry * (1 - strength) + wet * strength.
                voice[i] = voice[i] * (1.0 - strength) + wet * strength;
                if !voice[i].is_finite() {
                    voice[i] = 0.0;
                }
            }
        }
    }
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
// Internal: F0 detection via normalized autocorrelation
// ---------------------------------------------------------------------------

/// Detect fundamental frequency via normalized autocorrelation.
///
/// Uses a simplified YIN-like approach: compute the normalized
/// autocorrelation, find the first dip-then-peak pattern in the
/// plausible pitch range, and refine with parabolic interpolation.
fn detect_f0_autocorrelation(frame: &[f32], sample_rate: f32) -> f32 {
    let n = frame.len();
    if n < 4 {
        return 0.0;
    }

    // Lag range corresponding to [MIN_F0_HZ, MAX_F0_HZ].
    let min_lag = (sample_rate / MAX_F0_HZ).floor() as usize;
    let max_lag = ((sample_rate / MIN_F0_HZ).ceil() as usize).min(n / 2);
    if min_lag >= max_lag || max_lag >= n {
        return 0.0;
    }

    // Compute energy of the frame.
    let energy: f32 = frame
        .iter()
        .map(|&s| {
            let v = if s.is_finite() { s } else { 0.0 };
            v * v
        })
        .sum();
    if energy < 1e-10 {
        return 0.0;
    }

    // Normalized autocorrelation for each lag.
    let mut best_lag = 0usize;
    let mut best_corr = -1.0f32;

    for lag in min_lag..=max_lag {
        let mut num = 0.0f32;
        let mut den_a = 0.0f32;
        let mut den_b = 0.0f32;

        let count = n - lag;
        for i in 0..count {
            let a = if frame[i].is_finite() { frame[i] } else { 0.0 };
            let b = if frame[i + lag].is_finite() {
                frame[i + lag]
            } else {
                0.0
            };
            num += a * b;
            den_a += a * a;
            den_b += b * b;
        }

        let den = (den_a * den_b).sqrt();
        let corr = if den > 1e-10 { num / den } else { 0.0 };

        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    // Reject unvoiced signals.
    if best_corr < VOICING_THRESHOLD || best_lag == 0 {
        return 0.0;
    }

    // Parabolic interpolation for sub-sample accuracy.
    let refined_lag = if best_lag > min_lag && best_lag < max_lag {
        let prev = normalized_autocorr_at(frame, best_lag - 1);
        let curr = best_corr;
        let next = normalized_autocorr_at(frame, best_lag + 1);
        let denom = 2.0 * (2.0 * curr - prev - next);
        if denom.abs() > 1e-10 {
            best_lag as f32 + (prev - next) / denom
        } else {
            best_lag as f32
        }
    } else {
        best_lag as f32
    };

    if refined_lag > 0.0 {
        sample_rate / refined_lag
    } else {
        0.0
    }
}

/// Compute normalized autocorrelation at a specific lag.
fn normalized_autocorr_at(frame: &[f32], lag: usize) -> f32 {
    let n = frame.len();
    if lag >= n {
        return 0.0;
    }
    let mut num = 0.0f32;
    let mut den_a = 0.0f32;
    let mut den_b = 0.0f32;
    let count = n - lag;
    for i in 0..count {
        let a = if frame[i].is_finite() { frame[i] } else { 0.0 };
        let b = if frame[i + lag].is_finite() {
            frame[i + lag]
        } else {
            0.0
        };
        num += a * b;
        den_a += a * a;
        den_b += b * b;
    }
    let den = (den_a * den_b).sqrt();
    if den > 1e-10 {
        num / den
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Internal: DFT (real, in-place, O(N^2) reference implementation)
// ---------------------------------------------------------------------------

/// Compute DFT magnitudes and phases from a real signal.
///
/// Returns `(magnitudes, phases)` for bins 0..N/2+1.
fn compute_dft_mag_phase(signal: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = signal.len();
    let n_bins = n / 2 + 1;
    let mut mags = Vec::with_capacity(n_bins);
    let mut phases = Vec::with_capacity(n_bins);

    let inv_n = 1.0 / n as f64;
    let tau = 2.0 * std::f64::consts::PI;

    for k in 0..n_bins {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        let omega = tau * k as f64 * inv_n;
        for (i, &s) in signal.iter().enumerate() {
            let v = if s.is_finite() { f64::from(s) } else { 0.0 };
            let angle = omega * i as f64;
            re += v * angle.cos();
            im -= v * angle.sin();
        }
        mags.push(re.hypot(im) as f32);
        phases.push((im.atan2(re)) as f32);
    }

    (mags, phases)
}

/// Forward DFT: real/imaginary arrays, in-place, O(N^2).
///
/// On entry `re` contains the signal, `im` is zero.
/// On exit both contain the DFT coefficients.
fn dft_forward(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let tau = 2.0 * std::f64::consts::PI / n as f64;

    let input_re: Vec<f64> = re
        .iter()
        .map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 })
        .collect();

    for k in 0..n {
        let mut sum_re = 0.0f64;
        let mut sum_im = 0.0f64;
        let omega = tau * k as f64;
        for (i, &s) in input_re.iter().enumerate() {
            let angle = omega * i as f64;
            sum_re += s * angle.cos();
            sum_im -= s * angle.sin();
        }
        re[k] = sum_re as f32;
        im[k] = sum_im as f32;
    }
}

/// Inverse DFT: in-place, O(N^2).
///
/// On entry `re`/`im` contain DFT coefficients.
/// On exit `re` contains the reconstructed signal.
fn dft_inverse(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    if n == 0 {
        return;
    }
    let tau = 2.0 * std::f64::consts::PI / n as f64;
    let inv_n = 1.0 / n as f64;

    let spec_re: Vec<f64> = re
        .iter()
        .map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 })
        .collect();
    let spec_im: Vec<f64> = im
        .iter()
        .map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 })
        .collect();

    for i in 0..n {
        let mut sum = 0.0f64;
        for k in 0..n {
            let angle = tau * k as f64 * i as f64;
            sum += spec_re[k] * angle.cos() - spec_im[k] * angle.sin();
        }
        re[i] = (sum * inv_n) as f32;
        if !re[i].is_finite() {
            re[i] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: utilities
// ---------------------------------------------------------------------------

/// Linearly interpolate between two angles (radians), taking the short
/// arc. Returns angle in [-pi, pi].
fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut diff = b - a;
    // Wrap to [-pi, pi].
    while diff > std::f32::consts::PI {
        diff -= std::f32::consts::TAU;
    }
    while diff < -std::f32::consts::PI {
        diff += std::f32::consts::TAU;
    }
    let result = a + diff * t;
    // Normalize to [-pi, pi].
    let mut r = result % std::f32::consts::TAU;
    if r > std::f32::consts::PI {
        r -= std::f32::consts::TAU;
    }
    if r < -std::f32::consts::PI {
        r += std::f32::consts::TAU;
    }
    r
}

/// Convert decibels to linear amplitude.
#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = KOKORO_SAMPLE_RATE as f32;

    fn sine_wave(freq: f32, n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / SR).sin())
            .collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
        (sum_sq / buf.len().max(1) as f32).sqrt()
    }

    // -- Config validation --

    #[test]
    fn test_config_default_valid() {
        HarmonicTunerConfig::default()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_invalid_n_harmonics() {
        assert!(HarmonicTunerConfig::new()
            .with_n_harmonics(0)
            .validate()
            .is_err());
        assert!(HarmonicTunerConfig::new()
            .with_n_harmonics(17)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_correction_strength() {
        assert!(HarmonicTunerConfig::new()
            .with_correction_strength(-0.1)
            .validate()
            .is_err());
        assert!(HarmonicTunerConfig::new()
            .with_correction_strength(1.1)
            .validate()
            .is_err());
        assert!(HarmonicTunerConfig::new()
            .with_correction_strength(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_harmonic_spread() {
        assert!(HarmonicTunerConfig::new()
            .with_harmonic_spread(-0.1)
            .validate()
            .is_err());
        assert!(HarmonicTunerConfig::new()
            .with_harmonic_spread(1.1)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_odd_even_balance() {
        assert!(HarmonicTunerConfig::new()
            .with_odd_even_balance(-1.1)
            .validate()
            .is_err());
        assert!(HarmonicTunerConfig::new()
            .with_odd_even_balance(1.1)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_window_size() {
        assert!(HarmonicTunerConfig::new()
            .with_window_size(128)
            .validate()
            .is_err());
        assert!(HarmonicTunerConfig::new()
            .with_window_size(1000)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        assert!(HarmonicTunerConfig::new()
            .with_sample_rate(0.0)
            .validate()
            .is_err());
        assert!(HarmonicTunerConfig::new()
            .with_sample_rate(-1.0)
            .validate()
            .is_err());
        assert!(HarmonicTunerConfig::new()
            .with_sample_rate(f32::NAN)
            .validate()
            .is_err());
    }

    // -- Presets --

    #[test]
    fn test_presets_validate() {
        HarmonicTunerConfig::natural().validate().expect("natural");
        HarmonicTunerConfig::bright_stack()
            .validate()
            .expect("bright_stack");
        HarmonicTunerConfig::warm_stack()
            .validate()
            .expect("warm_stack");
        HarmonicTunerConfig::wide_harmonics()
            .validate()
            .expect("wide_harmonics");
    }

    // -- F0 detection --

    #[test]
    fn test_f0_detection_sine_440() {
        let audio = sine_wave(440.0, 4096);
        let f0 = detect_f0_autocorrelation(&audio, SR);
        assert!((f0 - 440.0).abs() < 10.0, "expected ~440 Hz, got {f0}");
    }

    #[test]
    fn test_f0_detection_sine_220() {
        let audio = sine_wave(220.0, 4096);
        let f0 = detect_f0_autocorrelation(&audio, SR);
        assert!((f0 - 220.0).abs() < 10.0, "expected ~220 Hz, got {f0}");
    }

    #[test]
    fn test_f0_detection_silence_returns_zero() {
        let audio = vec![0.0f32; 4096];
        let f0 = detect_f0_autocorrelation(&audio, SR);
        assert_eq!(f0, 0.0, "silence should return 0 Hz");
    }

    #[test]
    fn test_f0_detection_short_signal() {
        let audio = vec![0.5; 2];
        let f0 = detect_f0_autocorrelation(&audio, SR);
        assert_eq!(f0, 0.0);
    }

    // -- Harmonic analysis --

    #[test]
    fn test_analyze_voice_sine() {
        let audio = sine_wave(440.0, 4096);
        let tuner = HarmonicTunerProcessor::with_defaults().unwrap();
        let analysis = tuner.analyze_voice(&audio);
        assert!(analysis.is_voiced(), "sine wave should be voiced");
        assert!(
            (analysis.fundamental_hz - 440.0).abs() < 15.0,
            "expected ~440 Hz fundamental, got {}",
            analysis.fundamental_hz
        );
        assert!(
            !analysis.harmonics.is_empty(),
            "should have at least one harmonic"
        );
        // Fundamental should have the largest magnitude.
        if analysis.harmonics.len() > 1 {
            assert!(
                analysis.harmonics[0].1 >= analysis.harmonics[1].1 * 0.5,
                "fundamental should be strong"
            );
        }
    }

    #[test]
    fn test_analyze_voice_short_audio() {
        let audio = vec![0.5; 64];
        let tuner = HarmonicTunerProcessor::with_defaults().unwrap();
        let analysis = tuner.analyze_voice(&audio);
        assert!(!analysis.is_voiced(), "too-short audio should be unvoiced");
    }

    #[test]
    fn test_analyze_voice_with_harmonics() {
        // Create a signal with fundamental + 2nd + 3rd harmonics.
        let f0 = 300.0;
        let n = 4096;
        let audio: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SR;
                0.5 * (std::f32::consts::TAU * f0 * t).sin()
                    + 0.3 * (std::f32::consts::TAU * 2.0 * f0 * t).sin()
                    + 0.2 * (std::f32::consts::TAU * 3.0 * f0 * t).sin()
            })
            .collect();
        let tuner = HarmonicTunerProcessor::with_defaults().unwrap();
        let analysis = tuner.analyze_voice(&audio);
        assert!(analysis.is_voiced());
        assert!(
            analysis.harmonics.len() >= 3,
            "should detect at least 3 harmonics, got {}",
            analysis.harmonics.len()
        );
    }

    // -- process_voices --

    #[test]
    fn test_process_voices_empty() {
        let mut tuner = HarmonicTunerProcessor::with_defaults().unwrap();
        let mut voices: Vec<Vec<f32>> = vec![];
        tuner.process_voices(&mut voices);
        // Should not panic.
    }

    #[test]
    fn test_process_voices_zero_strength_is_identity() {
        let config = HarmonicTunerConfig::new().with_correction_strength(0.0);
        let mut tuner = HarmonicTunerProcessor::new(config).unwrap();
        let original = sine_wave(440.0, 4096);
        let mut voices = vec![original.clone()];
        tuner.process_voices(&mut voices);
        assert_eq!(voices[0], original, "zero strength should be identity");
    }

    #[test]
    fn test_process_voices_all_outputs_finite() {
        let mut tuner =
            HarmonicTunerProcessor::new(HarmonicTunerConfig::new().with_correction_strength(0.8))
                .unwrap();

        let voice_a = sine_wave(440.0, 4096);
        let mut voice_b = sine_wave(440.0, 4096);
        // Inject NaN into voice_b.
        voice_b[100] = f32::NAN;
        voice_b[200] = f32::INFINITY;

        let mut voices = vec![voice_a, voice_b];
        tuner.process_voices(&mut voices);

        for (vi, voice) in voices.iter().enumerate() {
            for (i, &s) in voice.iter().enumerate() {
                assert!(s.is_finite(), "voice[{vi}][{i}] is non-finite: {s}");
            }
        }
    }

    #[test]
    fn test_process_voices_modifies_with_spread() {
        let config = HarmonicTunerConfig::new()
            .with_correction_strength(0.8)
            .with_harmonic_spread(0.5);
        let mut tuner = HarmonicTunerProcessor::new(config).unwrap();

        let voice_a = sine_wave(440.0, 4096);
        let voice_b = sine_wave(440.0, 4096);
        let original_b = voice_b.clone();
        let mut voices = vec![voice_a, voice_b];
        tuner.process_voices(&mut voices);

        // With spread, voice_b should differ from original.
        let diff: f32 = voices[1]
            .iter()
            .zip(original_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / voices[1].len() as f32;
        assert!(diff > 1e-5, "spread should modify voice, mean_diff={diff}");
    }

    #[test]
    fn test_process_voices_preserves_energy() {
        let config = HarmonicTunerConfig::new()
            .with_correction_strength(0.5)
            .with_odd_even_balance(0.0);
        let mut tuner = HarmonicTunerProcessor::new(config).unwrap();

        let voice = sine_wave(440.0, 4096);
        let original_rms = rms(&voice);
        let mut voices = vec![voice];
        tuner.process_voices(&mut voices);
        let new_rms = rms(&voices[0]);

        // Energy should be roughly preserved (within 6 dB).
        let ratio = new_rms / original_rms.max(1e-10);
        assert!(
            ratio > 0.25 && ratio < 4.0,
            "energy should be roughly preserved, ratio={ratio}"
        );
    }

    // -- Odd/even balance --

    #[test]
    fn test_odd_even_balance_changes_spectrum() {
        // Create a signal with both odd and even harmonics.
        let f0 = 300.0;
        let n = 4096;
        let make_signal = || -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR;
                    0.4 * (std::f32::consts::TAU * f0 * t).sin()
                        + 0.3 * (std::f32::consts::TAU * 2.0 * f0 * t).sin()
                        + 0.2 * (std::f32::consts::TAU * 3.0 * f0 * t).sin()
                        + 0.1 * (std::f32::consts::TAU * 4.0 * f0 * t).sin()
                })
                .collect()
        };

        // Process with strong odd boost.
        let config_odd = HarmonicTunerConfig::new()
            .with_correction_strength(0.8)
            .with_odd_even_balance(-0.8)
            .with_harmonic_spread(0.0);
        let mut tuner_odd = HarmonicTunerProcessor::new(config_odd).unwrap();
        let mut voices_odd = vec![make_signal()];
        tuner_odd.process_voices(&mut voices_odd);

        // Process with strong even boost.
        let config_even = HarmonicTunerConfig::new()
            .with_correction_strength(0.8)
            .with_odd_even_balance(0.8)
            .with_harmonic_spread(0.0);
        let mut tuner_even = HarmonicTunerProcessor::new(config_even).unwrap();
        let mut voices_even = vec![make_signal()];
        tuner_even.process_voices(&mut voices_even);

        // The two results should differ from each other.
        let diff: f32 = voices_odd[0]
            .iter()
            .zip(voices_even[0].iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n as f32;
        assert!(
            diff > 1e-5,
            "odd and even balance should produce different outputs, mean_diff={diff}"
        );
    }

    // -- lerp_angle --

    #[test]
    fn test_lerp_angle_same() {
        let r = lerp_angle(1.0, 1.0, 0.5);
        assert!((r - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lerp_angle_zero_t() {
        let r = lerp_angle(0.5, 2.0, 0.0);
        assert!((r - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_lerp_angle_one_t() {
        let r = lerp_angle(0.5, 2.0, 1.0);
        assert!((r - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_lerp_angle_wraps() {
        // Lerp from near +pi to near -pi should go the short way.
        let r = lerp_angle(3.0, -3.0, 0.5);
        assert!(r.abs() > 2.5, "should stay near +/-pi, got {r}");
    }

    // -- db_to_linear --

    #[test]
    fn test_db_to_linear_zero() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_db_to_linear_plus6() {
        // +6 dB ~= 2.0x amplitude.
        let r = db_to_linear(6.0);
        assert!((r - 2.0).abs() < 0.1, "6 dB should be ~2.0, got {r}");
    }

    #[test]
    fn test_db_to_linear_minus6() {
        let r = db_to_linear(-6.0);
        assert!((r - 0.5).abs() < 0.05, "-6 dB should be ~0.5, got {r}");
    }

    // -- DFT round-trip --

    #[test]
    fn test_dft_roundtrip() {
        let n = 64;
        let signal: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * 5.0 * i as f32 / n as f32).sin())
            .collect();
        let mut re = signal.clone();
        let mut im = vec![0.0f32; n];
        dft_forward(&mut re, &mut im);
        dft_inverse(&mut re, &mut im);

        for (i, (&orig, &recon)) in signal.iter().zip(re.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-3,
                "DFT roundtrip mismatch at {i}: orig={orig}, recon={recon}"
            );
        }
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut tuner = HarmonicTunerProcessor::with_defaults().unwrap();
        tuner.reset(); // Should not panic.
    }

    // -- Constructor with defaults --

    #[test]
    fn test_with_defaults() {
        let tuner = HarmonicTunerProcessor::with_defaults().unwrap();
        assert_eq!(tuner.config().n_harmonics, 8);
        assert_eq!(tuner.config().window_size, 2048);
    }
}
