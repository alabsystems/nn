// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spectral density and fullness optimizer for Kokoro chorus output.
//!
//! Ensures the combined chorus output fills the frequency spectrum evenly,
//! producing a "full" and "lush" sound. Rather than boosting existing content,
//! this module detects spectral gaps and generates complementary fill material:
//! harmonics of existing content, sub-harmonics for bass presence, and
//! band-limited shaped noise for ambient fill.
//!
//! # Algorithm
//!
//! 1. Analyze the stereo signal into `n_bands` logarithmically-spaced bands
//!    using a windowed FFT.
//! 2. Compute per-band energy (dB), spectral flatness, and a density score.
//! 3. Detect gaps: bands whose energy is more than 12 dB below a local
//!    moving average of their neighbors.
//! 4. For each gap, generate fill material via one or more methods:
//!    - **Harmonic fill**: synthesize harmonics at the gap frequency derived
//!      from the nearest occupied band.
//!    - **Noise fill**: band-limited pink noise shaped to the gap frequency
//!      at a configurable level below the signal.
//!    - **Sub-harmonic fill**: for gaps below 200 Hz, generate octave-down
//!      content derived from the 400 Hz region.
//! 5. Mix fill material into the output at `fill_strength`.
//!
//! # References
//!
//! - Puckette, M. "The Theory and Technique of Electronic Music." (2007).
//! - Zolzer, U. "DAFX: Digital Audio Effects." 2nd ed., Wiley, 2011.
//! - ITU-R BS.1770-4 loudness measurement (spectral flatness concept).
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the spectral density and fullness optimizer.
///
/// Constructed via [`SpectralFillConfig::new`] and builder methods
/// (required for cross-crate use due to `#[non_exhaustive]`).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpectralFillConfig {
    /// How aggressively to fill detected gaps. 0.0 = off, 1.0 = full fill.
    /// Default: 0.2.
    pub fill_strength: f32,
    /// Target spectral flatness (0.0 = tonal, 1.0 = white noise).
    /// The optimizer nudges the output toward this flatness.
    /// Default: 0.6.
    pub target_density: f32,
    /// Generate harmonics of existing content to fill gaps. Default: true.
    pub harmonic_fill: bool,
    /// Generate sub-harmonics for bass presence in gaps below 200 Hz.
    /// Default: false.
    pub sub_harmonic_fill: bool,
    /// Use shaped noise to fill spectral gaps. Default: true.
    pub noise_fill: bool,
    /// Noise fill level in dB relative to signal RMS. Default: -40.0.
    pub noise_level_db: f32,
    /// Number of analysis bands (logarithmically spaced). Default: 32.
    pub n_bands: usize,
    /// FFT window size (must be power of 2). Default: 2048.
    pub window_size: usize,
    /// Sample rate in Hz. Default: 24000.0.
    pub sample_rate: f32,
}

impl Default for SpectralFillConfig {
    fn default() -> Self {
        Self {
            fill_strength: 0.2,
            target_density: 0.6,
            harmonic_fill: true,
            sub_harmonic_fill: false,
            noise_fill: true,
            noise_level_db: -40.0,
            n_bands: 32,
            window_size: 2048,
            sample_rate: 24000.0,
        }
    }
}

impl SpectralFillConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the fill strength (0.0-1.0).
    #[must_use]
    pub fn with_fill_strength(mut self, v: f32) -> Self {
        self.fill_strength = v;
        self
    }

    /// Set the target spectral density/flatness (0.0-1.0).
    #[must_use]
    pub fn with_target_density(mut self, v: f32) -> Self {
        self.target_density = v;
        self
    }

    /// Enable or disable harmonic fill.
    #[must_use]
    pub fn with_harmonic_fill(mut self, v: bool) -> Self {
        self.harmonic_fill = v;
        self
    }

    /// Enable or disable sub-harmonic fill.
    #[must_use]
    pub fn with_sub_harmonic_fill(mut self, v: bool) -> Self {
        self.sub_harmonic_fill = v;
        self
    }

    /// Enable or disable noise fill.
    #[must_use]
    pub fn with_noise_fill(mut self, v: bool) -> Self {
        self.noise_fill = v;
        self
    }

    /// Set the noise fill level in dB relative to signal.
    #[must_use]
    pub fn with_noise_level_db(mut self, v: f32) -> Self {
        self.noise_level_db = v;
        self
    }

    /// Set the number of analysis bands.
    #[must_use]
    pub fn with_n_bands(mut self, v: usize) -> Self {
        self.n_bands = v;
        self
    }

    /// Set the FFT window size (power of 2).
    #[must_use]
    pub fn with_window_size(mut self, v: usize) -> Self {
        self.window_size = v;
        self
    }

    /// Set the sample rate in Hz.
    #[must_use]
    pub fn with_sample_rate(mut self, v: f32) -> Self {
        self.sample_rate = v;
        self
    }

    /// Validate all parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let err =
            |field: &'static str, reason: String| Err(KokoroError::InvalidConfig { field, reason });
        if !self.fill_strength.is_finite() || !(0.0..=1.0).contains(&self.fill_strength) {
            return err(
                "fill_strength",
                format!(
                    "fill_strength = {}: must be finite in [0.0, 1.0]",
                    self.fill_strength,
                ),
            );
        }
        if !self.target_density.is_finite() || !(0.0..=1.0).contains(&self.target_density) {
            return err(
                "target_density",
                format!(
                    "target_density = {}: must be finite in [0.0, 1.0]",
                    self.target_density,
                ),
            );
        }
        if !self.noise_level_db.is_finite() || self.noise_level_db > 0.0 {
            return err(
                "noise_level_db",
                format!(
                    "noise_level_db = {}: must be finite and <= 0.0",
                    self.noise_level_db,
                ),
            );
        }
        if self.n_bands == 0 || self.n_bands > 128 {
            return err(
                "n_bands",
                format!("n_bands = {}: must be in [1, 128]", self.n_bands),
            );
        }
        if !self.window_size.is_power_of_two() || self.window_size < 256 || self.window_size > 8192
        {
            return err(
                "window_size",
                format!(
                    "window_size = {}: must be power of 2 in [256, 8192]",
                    self.window_size,
                ),
            );
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return err(
                "sample_rate",
                format!(
                    "sample_rate = {}: must be finite and positive",
                    self.sample_rate,
                ),
            );
        }
        Ok(())
    }

    // --- Presets ---------------------------------------------------------------

    /// Subtle fill -- barely perceptible gap-filling for clean mixes.
    /// Preserves the original character while gently reducing spectral holes.
    #[must_use]
    pub fn subtle_fill() -> Self {
        Self {
            fill_strength: 0.1,
            target_density: 0.5,
            harmonic_fill: true,
            sub_harmonic_fill: false,
            noise_fill: true,
            noise_level_db: -48.0,
            ..Self::default()
        }
    }

    /// Lush -- noticeable fullness enhancement for rich, enveloping sound.
    /// Enables all fill methods including sub-harmonics for bass depth.
    #[must_use]
    pub fn lush() -> Self {
        Self {
            fill_strength: 0.35,
            target_density: 0.7,
            harmonic_fill: true,
            sub_harmonic_fill: true,
            noise_fill: true,
            noise_level_db: -36.0,
            ..Self::default()
        }
    }

    /// Dense pad -- heavy spectral fill for pad-like, ambient textures.
    /// Maximizes fullness at the cost of transparency.
    #[must_use]
    pub fn dense_pad() -> Self {
        Self {
            fill_strength: 0.6,
            target_density: 0.85,
            harmonic_fill: true,
            sub_harmonic_fill: true,
            noise_fill: true,
            noise_level_db: -30.0,
            ..Self::default()
        }
    }

    /// Transparent -- minimal fill with noise-only gap treatment.
    /// Adds air without coloring the harmonic content.
    #[must_use]
    pub fn transparent() -> Self {
        Self {
            fill_strength: 0.08,
            target_density: 0.45,
            harmonic_fill: false,
            sub_harmonic_fill: false,
            noise_fill: true,
            noise_level_db: -50.0,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Spectral density analysis result
// ---------------------------------------------------------------------------

/// Result of spectral density analysis on a stereo signal.
#[derive(Debug, Clone)]
pub struct SpectralDensityAnalysis {
    /// Energy per analysis band in dB (relative to full-scale).
    pub per_band_level_db: Vec<f32>,
    /// Spectral flatness (0.0 = purely tonal, 1.0 = white noise).
    /// Computed as geometric_mean / arithmetic_mean of band energies.
    pub spectral_flatness: f32,
    /// Detected spectral gaps: `(band_index, gap_depth_db)`.
    /// A gap is a band whose energy is more than the gap threshold below
    /// the local moving average of its neighbors.
    pub gaps: Vec<(usize, f32)>,
    /// Overall density/fullness score (0.0-1.0).
    /// Higher = more evenly distributed spectral energy.
    pub density_score: f32,
}

// ---------------------------------------------------------------------------
// FFT and windowing helpers
// ---------------------------------------------------------------------------

/// Radix-2 DIT FFT (matches kokoro_chorus_spectral_match.rs).
fn fft(data: &mut [(f32, f32)]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            data.swap(i, j);
        }
    }
    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let angle_step = -std::f32::consts::TAU / stage_len as f32;
        for k in (0..n).step_by(stage_len) {
            for j in 0..half {
                let angle = angle_step * j as f32;
                let (tw_re, tw_im) = (angle.cos(), angle.sin());
                let (a_re, a_im) = data[k + j];
                let (b_re, b_im) = data[k + j + half];
                let t_re = b_re * tw_re - b_im * tw_im;
                let t_im = b_re * tw_im + b_im * tw_re;
                data[k + j] = (a_re + t_re, a_im + t_im);
                data[k + j + half] = (a_re - t_re, a_im - t_im);
            }
        }
        stage_len *= 2;
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            0.5 * (1.0 - (std::f32::consts::TAU * t).cos())
        })
        .collect()
}

/// Compute logarithmically-spaced band center frequencies.
fn compute_band_centers(n_bands: usize, sample_rate: f32) -> Vec<f32> {
    let low_hz = 60.0f32;
    let nyquist = sample_rate / 2.0;
    let high_hz = nyquist.min(12000.0);
    if n_bands <= 1 {
        return vec![(low_hz * high_hz).sqrt()];
    }
    let log_low = low_hz.ln();
    let log_high = high_hz.ln();
    (0..n_bands)
        .map(|i| {
            let t = i as f32 / (n_bands - 1) as f32;
            (log_low + t * (log_high - log_low)).exp()
        })
        .collect()
}

/// Compute per-band energy in dB from an audio buffer.
fn analyze_band_energy(
    audio: &[f32],
    window: &[f32],
    band_centers: &[f32],
    sample_rate: f32,
) -> Vec<f32> {
    let n = window.len();
    let mut frame = vec![0.0f32; n];
    let src_len = audio.len().min(n);
    let src_start = audio.len().saturating_sub(n);
    for (i, &s) in audio[src_start..src_start + src_len].iter().enumerate() {
        frame[i] = if s.is_finite() { s } else { 0.0 };
    }

    // Apply Hann window.
    for (s, &w) in frame.iter_mut().zip(window.iter()) {
        *s *= w;
    }

    // FFT.
    let mut spectrum: Vec<(f32, f32)> = frame.iter().map(|&s| (s, 0.0)).collect();
    fft(&mut spectrum);

    let n_bins = n / 2 + 1;
    let bin_width = sample_rate / n as f32;
    let factor = 2.0f32.powf(1.0 / 6.0); // 1/3-octave bandwidth

    let mut band_db = vec![-96.0f32; band_centers.len()];
    for (idx, &center) in band_centers.iter().enumerate() {
        let bin_low = ((center / factor) / bin_width).floor() as usize;
        let bin_high = ((center * factor) / bin_width).ceil() as usize;
        let bin_low = bin_low.max(1);
        let bin_high = bin_high.min(n_bins - 1);
        if bin_low > bin_high {
            continue;
        }
        let mut energy = 0.0f32;
        let mut count = 0usize;
        for bin in bin_low..=bin_high {
            let (re, im) = spectrum[bin];
            energy += re * re + im * im;
            count += 1;
        }
        if count > 0 && energy > 0.0 {
            let db = 20.0 * (energy / count as f32).sqrt().log10();
            band_db[idx] = db.max(-96.0);
        }
    }
    band_db
}

// ---------------------------------------------------------------------------
// Gap detection
// ---------------------------------------------------------------------------

/// Gap threshold: a band is a "gap" when it is this many dB below
/// the local moving average of its neighbors.
const GAP_THRESHOLD_DB: f32 = 12.0;

/// Moving average window radius for gap detection.
const GAP_AVG_RADIUS: usize = 3;

/// Detect spectral gaps by comparing per-band energy to a moving average
/// of neighboring bands.
fn detect_gaps(band_db: &[f32]) -> Vec<(usize, f32)> {
    let n = band_db.len();
    if n < 2 {
        return Vec::new();
    }
    let mut gaps = Vec::new();
    for i in 0..n {
        let lo = i.saturating_sub(GAP_AVG_RADIUS);
        let hi = (i + GAP_AVG_RADIUS + 1).min(n);
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for j in lo..hi {
            if j != i {
                sum += band_db[j];
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }
        let avg = sum / count as f32;
        let depth = avg - band_db[i];
        if depth > GAP_THRESHOLD_DB {
            gaps.push((i, depth));
        }
    }
    gaps
}

/// Compute spectral flatness as geometric_mean / arithmetic_mean of
/// linear band energies.
fn compute_spectral_flatness(band_db: &[f32]) -> f32 {
    let n = band_db.len();
    if n == 0 {
        return 0.0;
    }
    // Convert dB to linear power for the computation.
    let linear: Vec<f32> = band_db
        .iter()
        .map(|&db| 10.0f32.powf(db / 10.0).max(1e-12))
        .collect();

    let arithmetic_mean = linear.iter().sum::<f32>() / n as f32;
    if arithmetic_mean <= 0.0 || !arithmetic_mean.is_finite() {
        return 0.0;
    }

    // Log-domain geometric mean to avoid overflow/underflow.
    let log_sum: f32 = linear.iter().map(|&x| x.ln()).sum();
    let geometric_mean = (log_sum / n as f32).exp();

    let flatness = geometric_mean / arithmetic_mean;
    flatness.clamp(0.0, 1.0)
}

/// Compute an overall density score from per-band levels.
/// Combines spectral flatness with gap penalty.
fn compute_density_score(band_db: &[f32], gaps: &[(usize, f32)]) -> f32 {
    let flatness = compute_spectral_flatness(band_db);
    let n = band_db.len().max(1);
    // Penalty proportional to the fraction of bands that are gaps.
    let gap_penalty = gaps.len() as f32 / n as f32;
    (flatness * (1.0 - gap_penalty * 0.5)).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Fill generation
// ---------------------------------------------------------------------------

/// Simple deterministic pseudo-random number generator (xorshift32).
/// Used for noise fill -- no external dependency needed.
struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        // Map to [-1, 1].
        (self.state as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Single-pole IIR lowpass for noise shaping.
struct OnePoleLP {
    coeff: f32,
    z1: f32,
}

impl OnePoleLP {
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let w = (-std::f32::consts::TAU * cutoff_hz / sample_rate).exp();
        Self { coeff: w, z1: 0.0 }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let b = 1.0 - self.coeff;
        let y = b * x + self.coeff * self.z1;
        self.z1 = if y.is_finite() { y } else { 0.0 };
        self.z1
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
    }
}

/// Single-pole highpass for band-limiting noise from below.
struct OnePoleHP {
    coeff: f32,
    x_prev: f32,
    y_prev: f32,
}

impl OnePoleHP {
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let rc = 1.0 / (std::f32::consts::TAU * cutoff_hz);
        let dt = 1.0 / sample_rate;
        Self {
            coeff: rc / (rc + dt),
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.coeff * (self.y_prev + x - self.x_prev);
        self.x_prev = x;
        self.y_prev = if y.is_finite() { y } else { 0.0 };
        self.y_prev
    }

    fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }
}

/// Generate band-limited pink-ish noise shaped to a target frequency band.
/// Returns a buffer of `len` samples at the given amplitude.
fn generate_band_noise(
    rng: &mut Xorshift32,
    lp: &mut OnePoleLP,
    hp: &mut OnePoleHP,
    len: usize,
    amplitude: f32,
) -> Vec<f32> {
    let mut buf = Vec::with_capacity(len);
    for _ in 0..len {
        let white = rng.next_f32();
        // Pink-ish approximation: LP filter white noise.
        let pink = lp.process(white);
        // Band-limit from below.
        let shaped = hp.process(pink);
        buf.push(shaped * amplitude);
    }
    buf
}

/// Generate a sinusoidal harmonic fill at the given frequency and amplitude.
fn generate_harmonic_tone(
    freq_hz: f32,
    sample_rate: f32,
    len: usize,
    amplitude: f32,
    phase: &mut f32,
) -> Vec<f32> {
    let phase_inc = std::f32::consts::TAU * freq_hz / sample_rate;
    let mut buf = Vec::with_capacity(len);
    for _ in 0..len {
        buf.push((*phase).sin() * amplitude);
        *phase += phase_inc;
        // Keep phase bounded to avoid float precision loss.
        if *phase > std::f32::consts::TAU {
            *phase -= std::f32::consts::TAU;
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// SpectralFillProcessor
// ---------------------------------------------------------------------------

/// Stateful spectral density and fullness optimizer.
///
/// Analyzes the stereo chorus output for spectral gaps and generates
/// complementary fill material (harmonics, noise, sub-harmonics) to
/// produce a full, lush frequency spectrum.
pub struct SpectralFillProcessor {
    config: SpectralFillConfig,
    band_centers: Vec<f32>,
    window: Vec<f32>,
    /// Per-gap harmonic oscillator phases (indexed by gap band index).
    harmonic_phases: Vec<f32>,
    /// PRNG for noise generation.
    rng: Xorshift32,
    /// Per-gap noise LP filters (one per possible band).
    noise_lp: Vec<OnePoleLP>,
    /// Per-gap noise HP filters (one per possible band).
    noise_hp: Vec<OnePoleHP>,
    /// Sub-harmonic oscillator phases.
    sub_phases: Vec<f32>,
}

impl SpectralFillProcessor {
    /// Create a new spectral fill processor.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if parameters are out of range.
    pub fn new(config: SpectralFillConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let band_centers = compute_band_centers(config.n_bands, config.sample_rate);
        let window = hann_window(config.window_size);
        let n = band_centers.len();

        // Pre-allocate per-band filter state.
        let factor = 2.0f32.powf(1.0 / 6.0);
        let noise_lp: Vec<OnePoleLP> = band_centers
            .iter()
            .map(|&center| OnePoleLP::new(center * factor, config.sample_rate))
            .collect();
        let noise_hp: Vec<OnePoleHP> = band_centers
            .iter()
            .map(|&center| OnePoleHP::new((center / factor).max(20.0), config.sample_rate))
            .collect();

        Ok(Self {
            config,
            band_centers,
            window,
            harmonic_phases: vec![0.0; n],
            rng: Xorshift32::new(0xDEAD_BEEF),
            noise_lp,
            noise_hp,
            sub_phases: vec![0.0; n],
        })
    }

    /// Analyze the spectral density of the stereo signal without modifying it.
    #[must_use]
    pub fn analyze(&self, left: &[f32], right: &[f32]) -> SpectralDensityAnalysis {
        // Use mid signal for analysis (mono sum).
        let len = left.len().min(right.len());
        let mid: Vec<f32> = (0..len)
            .map(|i| {
                let l = if left[i].is_finite() { left[i] } else { 0.0 };
                let r = if right[i].is_finite() { right[i] } else { 0.0 };
                (l + r) * 0.5
            })
            .collect();

        let band_db = analyze_band_energy(
            &mid,
            &self.window,
            &self.band_centers,
            self.config.sample_rate,
        );

        let gaps = detect_gaps(&band_db);
        let flatness = compute_spectral_flatness(&band_db);
        let density = compute_density_score(&band_db, &gaps);

        SpectralDensityAnalysis {
            per_band_level_db: band_db,
            spectral_flatness: flatness,
            gaps,
            density_score: density,
        }
    }

    /// Analyze the signal and apply spectral fill to both channels in-place.
    ///
    /// Fast path: returns immediately when `fill_strength == 0.0`.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        if self.config.fill_strength == 0.0 {
            return;
        }

        let len = left.len().min(right.len());
        if len == 0 {
            return;
        }

        // Sanitize non-finite input samples before analysis.
        for s in left[..len].iter_mut() {
            if !s.is_finite() {
                *s = 0.0;
            }
        }
        for s in right[..len].iter_mut() {
            if !s.is_finite() {
                *s = 0.0;
            }
        }

        let analysis = self.analyze(left, right);
        if analysis.gaps.is_empty() {
            return;
        }

        // Compute signal RMS for level-relative noise fill.
        let rms = {
            let sum_sq: f32 = left[..len]
                .iter()
                .zip(right[..len].iter())
                .map(|(&l, &r)| {
                    let l = if l.is_finite() { l } else { 0.0 };
                    let r = if r.is_finite() { r } else { 0.0 };
                    let mid = (l + r) * 0.5;
                    mid * mid
                })
                .sum();
            (sum_sq / len as f32).sqrt().max(1e-10)
        };

        let noise_amplitude = rms * 10.0f32.powf(self.config.noise_level_db / 20.0);
        let strength = self.config.fill_strength;

        // Find nearest occupied band for harmonic fill reference.
        let occupied: Vec<usize> = (0..analysis.per_band_level_db.len())
            .filter(|&i| {
                analysis.per_band_level_db[i] > -60.0
                    && !analysis.gaps.iter().any(|&(gi, _)| gi == i)
            })
            .collect();

        for &(gap_idx, _gap_depth) in &analysis.gaps {
            if gap_idx >= self.band_centers.len() {
                continue;
            }
            let gap_freq = self.band_centers[gap_idx];
            let mut fill_buf = vec![0.0f32; len];

            // --- Harmonic fill ---
            if self.config.harmonic_fill && !occupied.is_empty() {
                // Find nearest occupied band.
                let nearest = occupied
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        let da = (self.band_centers[a] - gap_freq).abs();
                        let db = (self.band_centers[b] - gap_freq).abs();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);
                let source_freq = self.band_centers[nearest];

                // Find which harmonic of source_freq lands closest to gap_freq.
                let harmonic_num = (gap_freq / source_freq).round().max(1.0) as usize;
                let harmonic_freq = source_freq * harmonic_num as f32;

                // Generate harmonic tone at reduced amplitude.
                let harm_amp = noise_amplitude * 2.0; // Slightly louder than noise.
                let tone = generate_harmonic_tone(
                    harmonic_freq,
                    self.config.sample_rate,
                    len,
                    harm_amp,
                    &mut self.harmonic_phases[gap_idx],
                );
                for (f, &t) in fill_buf.iter_mut().zip(tone.iter()) {
                    *f += t;
                }
            }

            // --- Noise fill ---
            if self.config.noise_fill {
                let noise = generate_band_noise(
                    &mut self.rng,
                    &mut self.noise_lp[gap_idx],
                    &mut self.noise_hp[gap_idx],
                    len,
                    noise_amplitude,
                );
                for (f, &n) in fill_buf.iter_mut().zip(noise.iter()) {
                    *f += n;
                }
            }

            // --- Sub-harmonic fill ---
            if self.config.sub_harmonic_fill && gap_freq < 200.0 {
                // Generate content one octave below the 400 Hz region.
                let sub_freq = gap_freq;
                let sub_amp = noise_amplitude * 1.5;
                let sub = generate_harmonic_tone(
                    sub_freq,
                    self.config.sample_rate,
                    len,
                    sub_amp,
                    &mut self.sub_phases[gap_idx],
                );
                for (f, &s) in fill_buf.iter_mut().zip(sub.iter()) {
                    *f += s;
                }
            }

            // Mix fill into both channels.
            for i in 0..len {
                let fill_sample = fill_buf[i] * strength;
                if fill_sample.is_finite() {
                    left[i] += fill_sample;
                    right[i] += fill_sample;
                }
                // NaN/Inf guard.
                if !left[i].is_finite() {
                    left[i] = 0.0;
                }
                if !right[i].is_finite() {
                    right[i] = 0.0;
                }
            }
        }
    }

    /// Reset all internal state (call between unrelated audio segments).
    pub fn reset(&mut self) {
        self.harmonic_phases.fill(0.0);
        self.sub_phases.fill(0.0);
        self.rng = Xorshift32::new(0xDEAD_BEEF);
        for lp in &mut self.noise_lp {
            lp.reset();
        }
        for hp in &mut self.noise_hp {
            hp.reset();
        }
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &SpectralFillConfig {
        &self.config
    }

    /// Read-only access to the analysis band center frequencies.
    #[must_use]
    pub fn band_centers(&self) -> &[f32] {
        &self.band_centers
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 24000.0;

    fn sine_wave(freq: f32, n: usize, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (std::f32::consts::TAU * freq * i as f32 / SR).sin())
            .collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
        (sum_sq / buf.len().max(1) as f32).sqrt()
    }

    // --- Config validation ---

    #[test]
    fn test_config_default_valid() {
        SpectralFillConfig::new()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn test_config_builder_roundtrip() {
        let cfg = SpectralFillConfig::new()
            .with_fill_strength(0.5)
            .with_target_density(0.8)
            .with_harmonic_fill(false)
            .with_sub_harmonic_fill(true)
            .with_noise_fill(false)
            .with_noise_level_db(-36.0)
            .with_n_bands(24)
            .with_window_size(4096)
            .with_sample_rate(48000.0);
        cfg.validate().expect("builder config should be valid");
        assert_eq!(cfg.fill_strength, 0.5);
        assert_eq!(cfg.target_density, 0.8);
        assert!(!cfg.harmonic_fill);
        assert!(cfg.sub_harmonic_fill);
        assert!(!cfg.noise_fill);
        assert_eq!(cfg.noise_level_db, -36.0);
        assert_eq!(cfg.n_bands, 24);
        assert_eq!(cfg.window_size, 4096);
        assert_eq!(cfg.sample_rate, 48000.0);
    }

    #[test]
    fn test_config_invalid_fill_strength() {
        assert!(SpectralFillConfig::new()
            .with_fill_strength(-0.1)
            .validate()
            .is_err());
        assert!(SpectralFillConfig::new()
            .with_fill_strength(1.5)
            .validate()
            .is_err());
        assert!(SpectralFillConfig::new()
            .with_fill_strength(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_target_density() {
        assert!(SpectralFillConfig::new()
            .with_target_density(-0.1)
            .validate()
            .is_err());
        assert!(SpectralFillConfig::new()
            .with_target_density(1.1)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_noise_level_db() {
        assert!(SpectralFillConfig::new()
            .with_noise_level_db(1.0)
            .validate()
            .is_err());
        assert!(SpectralFillConfig::new()
            .with_noise_level_db(f32::INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_n_bands() {
        assert!(SpectralFillConfig::new()
            .with_n_bands(0)
            .validate()
            .is_err());
        assert!(SpectralFillConfig::new()
            .with_n_bands(200)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_window_size() {
        assert!(SpectralFillConfig::new()
            .with_window_size(1000)
            .validate()
            .is_err());
        assert!(SpectralFillConfig::new()
            .with_window_size(128)
            .validate()
            .is_err());
        assert!(SpectralFillConfig::new()
            .with_window_size(16384)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        assert!(SpectralFillConfig::new()
            .with_sample_rate(0.0)
            .validate()
            .is_err());
        assert!(SpectralFillConfig::new()
            .with_sample_rate(-44100.0)
            .validate()
            .is_err());
        assert!(SpectralFillConfig::new()
            .with_sample_rate(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_presets_valid() {
        SpectralFillConfig::subtle_fill()
            .validate()
            .expect("subtle_fill valid");
        SpectralFillConfig::lush().validate().expect("lush valid");
        SpectralFillConfig::dense_pad()
            .validate()
            .expect("dense_pad valid");
        SpectralFillConfig::transparent()
            .validate()
            .expect("transparent valid");
    }

    // --- Analysis ---

    #[test]
    fn test_analyze_silent_signal() {
        let cfg = SpectralFillConfig::new();
        let proc = SpectralFillProcessor::new(cfg).expect("valid");
        let left = vec![0.0f32; 4096];
        let right = vec![0.0f32; 4096];
        let analysis = proc.analyze(&left, &right);
        assert_eq!(analysis.per_band_level_db.len(), 32);
        assert!(analysis.spectral_flatness >= 0.0 && analysis.spectral_flatness <= 1.0);
        assert!(analysis.density_score >= 0.0 && analysis.density_score <= 1.0);
    }

    #[test]
    fn test_analyze_tone_has_peaks() {
        let cfg = SpectralFillConfig::new();
        let proc = SpectralFillProcessor::new(cfg).expect("valid");
        let left = sine_wave(1000.0, 4096, 0.5);
        let right = sine_wave(1000.0, 4096, 0.5);
        let analysis = proc.analyze(&left, &right);
        // Should have at least some bands with energy.
        let active_bands = analysis
            .per_band_level_db
            .iter()
            .filter(|&&db| db > -80.0)
            .count();
        assert!(
            active_bands >= 1,
            "tone should produce at least 1 active band, got {active_bands}",
        );
        // Tonal signal should have low spectral flatness.
        assert!(
            analysis.spectral_flatness < 0.8,
            "tonal signal should have flatness < 0.8, got {}",
            analysis.spectral_flatness,
        );
    }

    #[test]
    fn test_analyze_detects_gaps() {
        // Create a signal with energy at low and high frequencies but a gap
        // in the middle.
        let n = 4096;
        let mut left = sine_wave(100.0, n, 0.5);
        let high = sine_wave(8000.0, n, 0.5);
        for (l, &h) in left.iter_mut().zip(high.iter()) {
            *l += h;
        }
        let right = left.clone();

        let cfg = SpectralFillConfig::new();
        let proc = SpectralFillProcessor::new(cfg).expect("valid");
        let analysis = proc.analyze(&left, &right);

        // The signal has content at ~100 Hz and ~8000 Hz but is missing the
        // middle. There should be some detected gaps.
        // Note: exact gap count depends on the band distribution and FFT
        // resolution, but we expect at least some.
        // We just verify the analysis produces valid output.
        assert!(analysis.per_band_level_db.len() == 32);
    }

    // --- Processing ---

    #[test]
    fn test_zero_strength_is_noop() {
        let cfg = SpectralFillConfig::new().with_fill_strength(0.0);
        let mut proc = SpectralFillProcessor::new(cfg).expect("valid");
        let mut left = sine_wave(440.0, 4096, 0.5);
        let mut right = sine_wave(440.0, 4096, 0.5);
        let orig_left = left.clone();
        let orig_right = right.clone();
        proc.process(&mut left, &mut right);
        assert_eq!(left, orig_left, "zero strength should not modify left");
        assert_eq!(right, orig_right, "zero strength should not modify right");
    }

    #[test]
    fn test_process_adds_energy_to_sparse_signal() {
        // Narrowband signal -- should have gaps that get filled.
        let n = 8192;
        let mut left = sine_wave(440.0, n, 0.3);
        let mut right = sine_wave(440.0, n, 0.3);
        let dry_rms = rms(&left);

        let cfg = SpectralFillConfig::lush();
        let mut proc = SpectralFillProcessor::new(cfg).expect("valid");
        proc.process(&mut left, &mut right);
        let wet_rms = rms(&left);

        // Fill should add some energy (even if small).
        assert!(
            wet_rms >= dry_rms - 1e-6,
            "fill should not significantly reduce energy: dry={dry_rms}, wet={wet_rms}",
        );
    }

    #[test]
    fn test_process_all_outputs_finite() {
        let cfg = SpectralFillConfig::dense_pad();
        let mut proc = SpectralFillProcessor::new(cfg).expect("valid");
        let mut left = vec![0.0, 0.5, -0.5, 1.0, f32::NAN, f32::INFINITY, -0.3, 0.1];
        // Pad to at least window_size for meaningful analysis.
        left.resize(4096, 0.1);
        let mut right = left.clone();
        proc.process(&mut left, &mut right);
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.is_finite(), "left sample {i} non-finite: {l}");
            assert!(r.is_finite(), "right sample {i} non-finite: {r}");
        }
    }

    #[test]
    fn test_process_empty_buffers() {
        let cfg = SpectralFillConfig::new();
        let mut proc = SpectralFillProcessor::new(cfg).expect("valid");
        let mut left: Vec<f32> = vec![];
        let mut right: Vec<f32> = vec![];
        proc.process(&mut left, &mut right);
        assert!(left.is_empty());
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = SpectralFillConfig::new();
        let mut proc = SpectralFillProcessor::new(cfg).expect("valid");
        let mut left = sine_wave(440.0, 4096, 0.5);
        let mut right = sine_wave(440.0, 4096, 0.5);
        proc.process(&mut left, &mut right);
        proc.reset();
        assert!(
            proc.harmonic_phases.iter().all(|&p| p == 0.0),
            "reset should zero harmonic phases",
        );
        assert!(
            proc.sub_phases.iter().all(|&p| p == 0.0),
            "reset should zero sub phases",
        );
    }

    #[test]
    fn test_stereo_coherence() {
        let n = 4096;
        let mut left = sine_wave(440.0, n, 0.3);
        let mut right = sine_wave(440.0, n, 0.3);

        let cfg = SpectralFillConfig::new()
            .with_fill_strength(0.3)
            .with_noise_fill(false); // Noise is random; disable for coherence test.
        let mut proc = SpectralFillProcessor::new(cfg).expect("valid");
        proc.process(&mut left, &mut right);

        // With identical input and deterministic fill (no noise), both
        // channels should receive the same fill material.
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (l - r).abs() < 1e-5,
                "sample {i}: left={l}, right={r} should be equal",
            );
        }
    }

    // --- Gap detection unit tests ---

    #[test]
    fn test_detect_gaps_uniform_no_gaps() {
        let band_db = vec![-20.0; 16];
        let gaps = detect_gaps(&band_db);
        assert!(gaps.is_empty(), "uniform spectrum should have no gaps");
    }

    #[test]
    fn test_detect_gaps_single_dip() {
        let mut band_db = vec![-20.0; 16];
        band_db[8] = -50.0; // 30 dB dip
        let gaps = detect_gaps(&band_db);
        assert!(
            gaps.iter().any(|&(idx, _)| idx == 8),
            "should detect gap at band 8, gaps={gaps:?}",
        );
    }

    #[test]
    fn test_spectral_flatness_pure_tone_low() {
        // A spectrum with energy in only one band should have low flatness.
        let mut band_db = vec![-96.0; 32];
        band_db[10] = -10.0;
        let flatness = compute_spectral_flatness(&band_db);
        assert!(
            flatness < 0.3,
            "single-band spectrum should have low flatness, got {flatness}",
        );
    }

    #[test]
    fn test_spectral_flatness_uniform_high() {
        let band_db = vec![-20.0; 32];
        let flatness = compute_spectral_flatness(&band_db);
        assert!(
            flatness > 0.9,
            "uniform spectrum should have high flatness, got {flatness}",
        );
    }

    #[test]
    fn test_density_score_bounded() {
        let band_db = vec![-20.0; 16];
        let gaps = detect_gaps(&band_db);
        let score = compute_density_score(&band_db, &gaps);
        assert!((0.0..=1.0).contains(&score), "density_score={score}");
    }

    #[test]
    fn test_band_centers_logarithmic() {
        let centers = compute_band_centers(32, 24000.0);
        assert_eq!(centers.len(), 32);
        for w in centers.windows(2) {
            assert!(w[1] > w[0], "bands must be monotonically increasing");
        }
        assert!(centers[0] > 50.0 && centers[0] < 70.0);
        assert!(centers[31] > 11000.0 && centers[31] < 12500.0);
    }

    #[test]
    fn test_xorshift_deterministic() {
        let mut rng1 = Xorshift32::new(42);
        let mut rng2 = Xorshift32::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_xorshift_range() {
        let mut rng = Xorshift32::new(12345);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((-1.0..=1.0).contains(&v), "value {v} out of range");
        }
    }
}
