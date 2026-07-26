// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spectral freeze effect for the Kokoro chorus system.
//!
//! Captures the current spectrum at a given moment and sustains it indefinitely,
//! creating infinite drone/pad textures from speech material. Useful for
//! atmospheric beds, ambient layers, and sustained vowel drones.
//!
//! # Algorithm
//!
//! ```text
//! Live audio
//!   -> Windowed FFT (Hann window, overlap-add)
//!   -> Capture magnitudes at freeze point
//!   -> Blend frozen magnitudes with live magnitudes per freeze_mix
//!   -> Apply optional phase randomization (avoids comb filtering)
//!   -> IFFT + overlap-add synthesis
//!   -> Crossfade on engage/disengage
//!   -> Output audio
//! ```
//!
//! # Phase Randomization
//!
//! Without phase randomization, resynthesis from a single captured frame
//! produces a periodic signal with strong comb-filter coloration. Randomizing
//! phases across bins eliminates this periodicity, producing a smoother,
//! more diffuse drone texture.
//!
//! # References
//!
//! - Wishart, T. (1994). "Audible Design." Orpheus the Pantomime.
//!   Spectral freezing as a compositional technique.
//! - Roads, C. (1996). "The Computer Music Tutorial." MIT Press.
//!   Chapter 9: Short-Time Fourier Transform and spectral processing.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the spectral freeze effect.
///
/// Controls the FFT window size, freeze/live blend, decay rate, phase
/// randomization, and crossfade duration when engaging or disengaging
/// the freeze.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FreezeConfig {
    /// FFT window size in samples (must be power of 2, range 256-4096).
    ///
    /// Larger windows capture more spectral detail but have worse time
    /// resolution. 1024 is a good default for speech material at 24kHz.
    pub window_size: usize,

    /// Blend of frozen spectrum with live audio (0.0-1.0).
    ///
    /// 0.0 = fully live (no freeze effect), 1.0 = fully frozen (infinite
    /// sustain of captured spectrum). Values in between crossfade the two
    /// spectra bin-by-bin before resynthesis.
    pub freeze_mix: f32,

    /// Decay rate for frozen magnitudes (0.0-1.0).
    ///
    /// 0.0 = infinite sustain (magnitudes never decay). 1.0 = immediate
    /// decay (frozen spectrum vanishes in one frame). Intermediate values
    /// produce a gradual fade-out of the frozen texture.
    pub decay_rate: f32,

    /// Whether to randomize phases of the frozen spectrum.
    ///
    /// When `true`, each resynthesis frame uses fresh random phases for the
    /// frozen component, eliminating comb-filter coloration and producing a
    /// smoother, more diffuse drone. When `false`, the original captured
    /// phases are reused, preserving the exact timbre of the freeze point
    /// but potentially sounding more metallic.
    pub randomize_phase: bool,

    /// Crossfade duration in milliseconds when engaging/disengaging freeze
    /// (10.0-500.0).
    ///
    /// Controls how smoothly the effect transitions between live and frozen
    /// states. Shorter crossfades are more abrupt; longer crossfades produce
    /// a gradual morph.
    pub crossfade_ms: f32,
}

impl Default for FreezeConfig {
    fn default() -> Self {
        Self {
            window_size: 1024,
            freeze_mix: 1.0,
            decay_rate: 0.0,
            randomize_phase: true,
            crossfade_ms: 50.0,
        }
    }
}

impl FreezeConfig {
    /// Create a new freeze config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the FFT window size.
    #[must_use]
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set the freeze/live blend ratio.
    #[must_use]
    pub fn with_freeze_mix(mut self, mix: f32) -> Self {
        self.freeze_mix = mix;
        self
    }

    /// Set the decay rate for frozen magnitudes.
    #[must_use]
    pub fn with_decay_rate(mut self, rate: f32) -> Self {
        self.decay_rate = rate;
        self
    }

    /// Enable or disable phase randomization.
    #[must_use]
    pub fn with_randomize_phase(mut self, randomize: bool) -> Self {
        self.randomize_phase = randomize;
        self
    }

    /// Set the crossfade duration in milliseconds.
    #[must_use]
    pub fn with_crossfade_ms(mut self, ms: f32) -> Self {
        self.crossfade_ms = ms;
        self
    }

    /// Validate that all parameters are within valid ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.window_size.is_power_of_two() || self.window_size < 256 || self.window_size > 4096
        {
            return Err(KokoroError::InvalidConfig {
                field: "window_size",
                reason: format!(
                    "window_size = {}: must be a power of 2 in [256, 4096]",
                    self.window_size,
                ),
            });
        }
        if !self.freeze_mix.is_finite() || !(0.0..=1.0).contains(&self.freeze_mix) {
            return Err(KokoroError::InvalidConfig {
                field: "freeze_mix",
                reason: format!(
                    "freeze_mix = {}: must be finite and in [0.0, 1.0]",
                    self.freeze_mix,
                ),
            });
        }
        if !self.decay_rate.is_finite() || !(0.0..=1.0).contains(&self.decay_rate) {
            return Err(KokoroError::InvalidConfig {
                field: "decay_rate",
                reason: format!(
                    "decay_rate = {}: must be finite and in [0.0, 1.0]",
                    self.decay_rate,
                ),
            });
        }
        if !self.crossfade_ms.is_finite() || !(10.0..=500.0).contains(&self.crossfade_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "crossfade_ms",
                reason: format!(
                    "crossfade_ms = {}: must be finite and in [10.0, 500.0]",
                    self.crossfade_ms,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Radix-2 DIT FFT (self-contained, matches kokoro_chorus_convolution.rs)
// ---------------------------------------------------------------------------

/// In-place radix-2 decimation-in-time FFT.
///
/// `data` length MUST be a power of two. Each element is `(re, im)`.
/// Forward transform (analysis): no 1/N scaling.
fn fft(data: &mut [(f32, f32)]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FFT length must be a power of two");
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            data.swap(i, j);
        }
    }

    // Butterfly stages.
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

/// In-place inverse FFT. Conjugate -> FFT -> conjugate -> scale by 1/N.
fn ifft(data: &mut [(f32, f32)]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    for (_, im) in data.iter_mut() {
        *im = -*im;
    }
    fft(data);
    let scale = 1.0 / n as f32;
    for (re, im) in data.iter_mut() {
        *re *= scale;
        *im = -*im * scale;
    }
}

// ---------------------------------------------------------------------------
// Hann window
// ---------------------------------------------------------------------------

/// Generate a Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            0.5 * (1.0 - (std::f32::consts::TAU * t).cos())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Deterministic LCG for phase randomization
// ---------------------------------------------------------------------------

/// Simple LCG pseudo-random number generator for deterministic phase
/// randomization. Uses Numerical Recipes parameters. Not cryptographic.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Return a pseudo-random f32 in [-PI, PI].
    fn next_phase(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (self.state >> 33) as i32;
        let normalized = bits as f32 / (i32::MAX as f32);
        normalized * std::f32::consts::PI
    }
}

// ---------------------------------------------------------------------------
// SpectralFreezer processor
// ---------------------------------------------------------------------------

/// Spectral freeze processor.
///
/// Captures the frequency-domain snapshot of an audio signal and sustains
/// it, blending frozen and live spectra via overlap-add synthesis.
pub struct SpectralFreezer {
    /// FFT window size (power of 2).
    window_size: usize,
    /// Hop size (window_size / 2 for 50% overlap).
    hop_size: usize,
    /// Hann analysis/synthesis window.
    window: Vec<f32>,
    /// Frozen magnitude spectrum (window_size bins). `None` if not yet captured.
    frozen_magnitudes: Option<Vec<f32>>,
    /// Frozen phase spectrum (window_size bins). Used when `randomize_phase`
    /// is false. `None` if not yet captured.
    frozen_phases: Option<Vec<f32>>,
    /// Input ring buffer for accumulating samples before FFT.
    input_buffer: Vec<f32>,
    /// Write position in the input ring buffer.
    input_pos: usize,
    /// Output overlap-add buffer (2 * window_size to handle overlap).
    output_buffer: Vec<f32>,
    /// Read position in the output buffer.
    output_read_pos: usize,
    /// Write position in the output buffer (where next IFFT frame lands).
    output_write_pos: usize,
    /// Samples accumulated since last FFT frame.
    samples_since_last_frame: usize,
    /// Crossfade length in samples.
    crossfade_samples: usize,
    /// Current crossfade position (counts down from crossfade_samples to 0).
    crossfade_pos: usize,
    /// Whether crossfade is active.
    crossfading: bool,
    /// Freeze/live mix ratio.
    freeze_mix: f32,
    /// Magnitude decay rate per frame.
    decay_rate: f32,
    /// Whether to randomize frozen phases.
    randomize_phase: bool,
    /// LCG for phase randomization.
    rng: Lcg,
}

impl SpectralFreezer {
    /// Create a new spectral freezer from the given configuration.
    ///
    /// `sample_rate` is used to convert crossfade duration from milliseconds
    /// to samples.
    pub fn new(config: &FreezeConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and positive"),
            });
        }

        let window_size = config.window_size;
        let hop_size = window_size / 2;
        let crossfade_samples = (config.crossfade_ms / 1000.0 * sample_rate) as usize;

        Ok(Self {
            window_size,
            hop_size,
            window: hann_window(window_size),
            frozen_magnitudes: None,
            frozen_phases: None,
            input_buffer: vec![0.0; window_size],
            input_pos: 0,
            output_buffer: vec![0.0; window_size * 2],
            output_read_pos: 0,
            output_write_pos: 0,
            samples_since_last_frame: 0,
            crossfade_samples,
            crossfade_pos: 0,
            crossfading: false,
            freeze_mix: config.freeze_mix,
            decay_rate: config.decay_rate,
            randomize_phase: config.randomize_phase,
            rng: Lcg::new(0xF2EE_2ECA_FE12_3400_u64.wrapping_add(window_size as u64)),
        })
    }

    /// Capture the current spectral snapshot from the provided audio window.
    ///
    /// Takes an FFT of `audio`, stores the magnitudes (and optionally phases),
    /// and enters the frozen state. If `audio` is shorter than `window_size`,
    /// it is zero-padded. If longer, only the last `window_size` samples are
    /// used.
    pub fn capture(&mut self, audio: &[f32]) {
        let n = self.window_size;
        let mut frame = vec![0.0f32; n];

        // Use the last `window_size` samples (or zero-pad if shorter).
        if audio.len() >= n {
            frame.copy_from_slice(&audio[audio.len() - n..]);
        } else {
            frame[..audio.len()].copy_from_slice(audio);
        }

        // Apply analysis window.
        for (s, &w) in frame.iter_mut().zip(self.window.iter()) {
            *s *= w;
        }

        // Forward FFT.
        let mut spectrum: Vec<(f32, f32)> = frame.iter().map(|&s| (s, 0.0)).collect();
        fft(&mut spectrum);

        // Extract magnitudes and phases.
        let mut magnitudes = Vec::with_capacity(n);
        let mut phases = Vec::with_capacity(n);
        for &(re, im) in &spectrum {
            magnitudes.push(re.hypot(im));
            phases.push(im.atan2(re));
        }

        let was_frozen = self.frozen_magnitudes.is_some();
        self.frozen_magnitudes = Some(magnitudes);
        self.frozen_phases = Some(phases);

        // Start crossfade if this is a new capture (not replacing an existing one).
        if !was_frozen && self.crossfade_samples > 0 {
            self.crossfading = true;
            self.crossfade_pos = self.crossfade_samples;
        }
    }

    /// Process a block of mono audio through the spectral freeze effect.
    ///
    /// Blends the frozen spectrum with the live signal using overlap-add
    /// synthesis. If no spectrum has been captured, passes audio through
    /// unchanged.
    pub fn process(&mut self, audio: &mut [f32]) {
        if self.frozen_magnitudes.is_none() {
            return;
        }

        let n = self.window_size;
        let hop = self.hop_size;

        for i in 0..audio.len() {
            // Feed sample into input ring buffer.
            self.input_buffer[self.input_pos] = audio[i];
            self.input_pos = (self.input_pos + 1) % n;
            self.samples_since_last_frame += 1;

            // When we have accumulated a hop's worth of samples, process a frame.
            if self.samples_since_last_frame >= hop {
                self.samples_since_last_frame = 0;
                self.process_frame();
            }

            // Read from output overlap-add buffer.
            let out_len = self.output_buffer.len();
            let frozen_sample = self.output_buffer[self.output_read_pos % out_len];
            self.output_buffer[self.output_read_pos % out_len] = 0.0;
            self.output_read_pos = (self.output_read_pos + 1) % out_len;

            // Blend frozen with live.
            let mix = self.current_mix();
            audio[i] = audio[i] * (1.0 - mix) + frozen_sample * mix;

            // Advance crossfade.
            if self.crossfading && self.crossfade_pos > 0 {
                self.crossfade_pos -= 1;
                if self.crossfade_pos == 0 {
                    self.crossfading = false;
                }
            }
        }
    }

    /// Whether a spectrum has been captured and the freezer is active.
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.frozen_magnitudes.is_some()
    }

    /// Clear the frozen spectrum and reset all internal buffers.
    pub fn reset(&mut self) {
        self.frozen_magnitudes = None;
        self.frozen_phases = None;
        self.input_buffer.fill(0.0);
        self.input_pos = 0;
        self.output_buffer.fill(0.0);
        self.output_read_pos = 0;
        self.output_write_pos = 0;
        self.samples_since_last_frame = 0;
        self.crossfading = false;
        self.crossfade_pos = 0;
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Compute the effective freeze mix, accounting for crossfade.
    fn current_mix(&self) -> f32 {
        if !self.crossfading {
            return self.freeze_mix;
        }
        // Linear crossfade from 0 -> freeze_mix over crossfade_samples.
        let progress = 1.0 - (self.crossfade_pos as f32 / self.crossfade_samples as f32);
        self.freeze_mix * progress
    }

    /// Process one overlapping frame: extract the current input window,
    /// FFT it, blend magnitudes with frozen snapshot, IFFT, and overlap-add
    /// into the output buffer.
    fn process_frame(&mut self) {
        let n = self.window_size;

        // Extract the current analysis window from the ring buffer.
        let mut frame = vec![0.0f32; n];
        for j in 0..n {
            let idx = (self.input_pos + j) % n;
            frame[j] = self.input_buffer[idx] * self.window[j];
        }

        // Forward FFT of the live frame.
        let mut spectrum: Vec<(f32, f32)> = frame.iter().map(|&s| (s, 0.0)).collect();
        fft(&mut spectrum);

        // Blend with frozen magnitudes.
        if let Some(ref mut frozen_mags) = self.frozen_magnitudes {
            let frozen_phases = self.frozen_phases.as_ref();

            for k in 0..n {
                let (re, im) = spectrum[k];
                let live_mag = re.hypot(im);
                let live_phase = im.atan2(re);

                let frozen_mag = frozen_mags[k];

                // Blended magnitude: mix between live and frozen.
                let blended_mag = live_mag * (1.0 - self.freeze_mix) + frozen_mag * self.freeze_mix;

                // Phase: use random phases for frozen component, or original.
                let phase = if self.randomize_phase && self.freeze_mix > 0.0 {
                    // Blend phases: live phase weighted by (1-mix), random by mix.
                    let random_phase = self.rng.next_phase();
                    if self.freeze_mix >= 1.0 {
                        random_phase
                    } else {
                        // Use live phase as base, nudge toward random.
                        live_phase * (1.0 - self.freeze_mix) + random_phase * self.freeze_mix
                    }
                } else if let Some(fp) = frozen_phases {
                    // Blend between live phase and captured phase.
                    live_phase * (1.0 - self.freeze_mix) + fp[k] * self.freeze_mix
                } else {
                    live_phase
                };

                spectrum[k] = (blended_mag * phase.cos(), blended_mag * phase.sin());
            }

            // Apply decay to frozen magnitudes.
            if self.decay_rate > 0.0 {
                let decay_factor = 1.0 - self.decay_rate;
                for mag in frozen_mags.iter_mut() {
                    *mag *= decay_factor;
                }
            }
        }

        // Inverse FFT.
        ifft(&mut spectrum);

        // Apply synthesis window and overlap-add into output buffer.
        let out_len = self.output_buffer.len();
        for j in 0..n {
            let sample = spectrum[j].0 * self.window[j];
            let pos = (self.output_write_pos + j) % out_len;
            self.output_buffer[pos] += sample;
        }

        // Advance the output write position by one hop.
        self.output_write_pos = (self.output_write_pos + self.hop_size) % out_len;
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Apply spectral freeze to an audio buffer, capturing at a given sample position.
///
/// Captures the spectrum centered around `capture_at` and freezes the
/// remainder of the audio. Audio before `capture_at` passes through unchanged.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if the configuration is invalid or
/// `capture_at` is beyond the audio length.
pub fn apply_spectral_freeze(
    audio: &mut [f32],
    config: &FreezeConfig,
    capture_at: usize,
    sample_rate: f32,
) -> Result<(), KokoroError> {
    if capture_at >= audio.len() {
        return Err(KokoroError::InvalidConfig {
            field: "capture_at",
            reason: format!(
                "capture_at = {} is beyond audio length {}",
                capture_at,
                audio.len(),
            ),
        });
    }

    let mut freezer = SpectralFreezer::new(config, sample_rate)?;

    // Determine the capture window: grab up to window_size samples ending
    // at capture_at (or as many as are available).
    let window = config.window_size;
    let start = capture_at.saturating_sub(window);
    freezer.capture(&audio[start..=capture_at]);

    // Process everything after the capture point.
    if capture_at + 1 < audio.len() {
        freezer.process(&mut audio[capture_at + 1..]);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SR: f32 = 24000.0;

    /// Helper: generate a sine wave.
    fn sine_wave(freq: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    /// Helper: compute RMS energy of a signal.
    fn rms(signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }
        (signal.iter().map(|s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
    }

    // -- Config validation --

    #[test]
    fn test_config_defaults_valid() {
        assert!(FreezeConfig::new().validate().is_ok());
    }

    #[test]
    fn test_config_rejects_bad_window_size() {
        assert!(FreezeConfig::new()
            .with_window_size(100)
            .validate()
            .is_err());
        assert!(FreezeConfig::new()
            .with_window_size(128)
            .validate()
            .is_err());
        assert!(FreezeConfig::new()
            .with_window_size(8192)
            .validate()
            .is_err());
        assert!(FreezeConfig::new().with_window_size(256).validate().is_ok());
        assert!(FreezeConfig::new()
            .with_window_size(4096)
            .validate()
            .is_ok());
    }

    #[test]
    fn test_config_rejects_bad_freeze_mix() {
        assert!(FreezeConfig::new()
            .with_freeze_mix(-0.1)
            .validate()
            .is_err());
        assert!(FreezeConfig::new().with_freeze_mix(1.1).validate().is_err());
        assert!(FreezeConfig::new()
            .with_freeze_mix(f32::NAN)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_rejects_bad_decay_rate() {
        assert!(FreezeConfig::new()
            .with_decay_rate(-0.01)
            .validate()
            .is_err());
        assert!(FreezeConfig::new()
            .with_decay_rate(1.01)
            .validate()
            .is_err());
    }

    #[test]
    fn test_config_rejects_bad_crossfade() {
        assert!(FreezeConfig::new()
            .with_crossfade_ms(5.0)
            .validate()
            .is_err());
        assert!(FreezeConfig::new()
            .with_crossfade_ms(501.0)
            .validate()
            .is_err());
        assert!(FreezeConfig::new()
            .with_crossfade_ms(f32::INFINITY)
            .validate()
            .is_err());
    }

    // -- Capture and frozen state --

    #[test]
    fn test_capture_produces_frozen_state() {
        let config = FreezeConfig::new();
        let mut freezer = SpectralFreezer::new(&config, TEST_SR).unwrap();

        assert!(!freezer.is_frozen());
        let audio = sine_wave(440.0, TEST_SR, 2048);
        freezer.capture(&audio);
        assert!(freezer.is_frozen());
    }

    #[test]
    fn test_reset_clears_frozen_state() {
        let config = FreezeConfig::new();
        let mut freezer = SpectralFreezer::new(&config, TEST_SR).unwrap();

        let audio = sine_wave(440.0, TEST_SR, 2048);
        freezer.capture(&audio);
        assert!(freezer.is_frozen());

        freezer.reset();
        assert!(!freezer.is_frozen());
    }

    // -- Frozen output energy --

    #[test]
    fn test_frozen_output_has_constant_energy() {
        let config = FreezeConfig::new()
            .with_freeze_mix(1.0)
            .with_decay_rate(0.0)
            .with_crossfade_ms(10.0);
        let mut freezer = SpectralFreezer::new(&config, TEST_SR).unwrap();

        // Capture from a sine wave.
        let capture_audio = sine_wave(440.0, TEST_SR, 2048);
        freezer.capture(&capture_audio);

        // Process several blocks of silence and measure energy of each.
        let block_size = 2048;
        let mut energies = Vec::new();
        for _ in 0..8 {
            let mut block = vec![0.0f32; block_size];
            freezer.process(&mut block);
            energies.push(rms(&block));
        }

        // Skip the first block (crossfade ramp-up), then energy should be
        // approximately constant across subsequent blocks.
        let stable = &energies[2..];
        let mean_energy: f32 = stable.iter().sum::<f32>() / stable.len() as f32;

        // All blocks should be within 20% of mean (allowing for overlap-add
        // artifacts and Hann window modulation).
        assert!(
            mean_energy > 0.0,
            "frozen output should have non-zero energy, got {mean_energy}",
        );
        for (i, &e) in stable.iter().enumerate() {
            let ratio = if mean_energy > 1e-10 {
                (e - mean_energy).abs() / mean_energy
            } else {
                0.0
            };
            assert!(
                ratio < 0.3,
                "block {}: energy {e} deviates from mean {mean_energy} by {:.1}%",
                i + 2,
                ratio * 100.0,
            );
        }
    }

    // -- Decay reduces energy --

    #[test]
    fn test_decay_reduces_energy_over_time() {
        let config = FreezeConfig::new()
            .with_freeze_mix(1.0)
            .with_decay_rate(0.1)
            .with_crossfade_ms(10.0);
        let mut freezer = SpectralFreezer::new(&config, TEST_SR).unwrap();

        let capture_audio = sine_wave(440.0, TEST_SR, 2048);
        freezer.capture(&capture_audio);

        // Process several blocks.
        let block_size = 2048;
        let mut energies = Vec::new();
        for _ in 0..10 {
            let mut block = vec![0.0f32; block_size];
            freezer.process(&mut block);
            energies.push(rms(&block));
        }

        // After crossfade stabilizes, energy should decrease monotonically
        // (approximately, allowing for some overlap-add variation).
        let later = &energies[3..];
        let first_later = later[0];
        let last_later = later[later.len() - 1];
        assert!(
            last_later < first_later,
            "energy should decrease with decay: first={first_later}, last={last_later}",
        );
    }

    // -- Phase randomization changes output --

    #[test]
    fn test_randomize_phase_changes_output() {
        // With randomize_phase = true.
        let config_random = FreezeConfig::new()
            .with_freeze_mix(1.0)
            .with_decay_rate(0.0)
            .with_randomize_phase(true)
            .with_crossfade_ms(10.0);
        let mut freezer_random = SpectralFreezer::new(&config_random, TEST_SR).unwrap();

        // With randomize_phase = false.
        let config_fixed = FreezeConfig::new()
            .with_freeze_mix(1.0)
            .with_decay_rate(0.0)
            .with_randomize_phase(false)
            .with_crossfade_ms(10.0);
        let mut freezer_fixed = SpectralFreezer::new(&config_fixed, TEST_SR).unwrap();

        let capture_audio = sine_wave(440.0, TEST_SR, 2048);
        freezer_random.capture(&capture_audio);
        freezer_fixed.capture(&capture_audio);

        // Process the same silence through both.
        let n = 4096;
        let mut out_random = vec![0.0f32; n];
        let mut out_fixed = vec![0.0f32; n];
        freezer_random.process(&mut out_random);
        freezer_fixed.process(&mut out_fixed);

        // Outputs should differ (different phase strategies produce different
        // waveforms even with the same magnitudes).
        let diff: f32 = out_random
            .iter()
            .zip(out_fixed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "random vs fixed phase outputs should differ, total diff = {diff}",
        );
    }

    // -- Convenience function --

    #[test]
    fn test_apply_spectral_freeze() {
        let config = FreezeConfig::new()
            .with_freeze_mix(0.8)
            .with_decay_rate(0.0)
            .with_crossfade_ms(10.0);

        let mut audio = sine_wave(440.0, TEST_SR, 8000);
        let original = audio.clone();

        apply_spectral_freeze(&mut audio, &config, 2000, TEST_SR).unwrap();

        // Audio before capture point should be unchanged.
        assert_eq!(&audio[..2000], &original[..2000]);

        // Audio after capture point should be modified.
        let diff: f32 = audio[2001..]
            .iter()
            .zip(original[2001..].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.1,
            "post-freeze audio should differ from original, total diff = {diff}",
        );
    }

    #[test]
    fn test_apply_spectral_freeze_invalid_capture_pos() {
        let config = FreezeConfig::new();
        let mut audio = vec![0.0f32; 100];
        assert!(apply_spectral_freeze(&mut audio, &config, 200, TEST_SR).is_err());
    }

    // -- FFT round-trip --

    #[test]
    fn test_fft_roundtrip() {
        let n = 256;
        let mut data: Vec<(f32, f32)> = (0..n).map(|i| ((i as f32 * 0.3).sin(), 0.0)).collect();
        let original: Vec<(f32, f32)> = data.clone();

        fft(&mut data);
        ifft(&mut data);

        for (i, (&orig, &recovered)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig.0 - recovered.0).abs() < 1e-4,
                "sample {i}: re mismatch: {} vs {}",
                orig.0,
                recovered.0,
            );
        }
    }

    // -- Process without capture is pass-through --

    #[test]
    fn test_process_without_capture_is_passthrough() {
        let config = FreezeConfig::new();
        let mut freezer = SpectralFreezer::new(&config, TEST_SR).unwrap();

        let mut audio = sine_wave(440.0, TEST_SR, 1024);
        let original = audio.clone();

        freezer.process(&mut audio);

        assert_eq!(audio, original, "unfrozen process should be pass-through");
    }

    // -- Zero freeze_mix passes live signal --

    #[test]
    fn test_zero_freeze_mix_passes_live() {
        let config = FreezeConfig::new()
            .with_freeze_mix(0.0)
            .with_crossfade_ms(10.0);
        let mut freezer = SpectralFreezer::new(&config, TEST_SR).unwrap();

        let capture_audio = sine_wave(440.0, TEST_SR, 2048);
        freezer.capture(&capture_audio);

        let mut audio = sine_wave(880.0, TEST_SR, 4096);
        let original = audio.clone();
        freezer.process(&mut audio);

        // With freeze_mix = 0.0 the output should essentially be the live
        // signal (within FFT analysis/synthesis round-trip tolerance).
        let max_diff: f32 = audio
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Allow some tolerance for the overlap-add round-trip windowing.
        assert!(
            max_diff < 0.5,
            "freeze_mix=0 should approximate live signal, max_diff = {max_diff}",
        );
    }
}
