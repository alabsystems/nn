// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Noise gate for Kokoro chorus voice cleanup.
//!
//! When stacking multiple TTS voices in a chorus, the noise floor is amplified
//! proportionally to the number of voices. A noise gate cuts signal below a
//! configurable threshold to clean up silent and quiet sections, producing a
//! tighter, more professional mix.
//!
//! # Architecture
//!
//! ```text
//! Input --> [Optional HP sidechain filter] --> Envelope follower (peak)
//!                                                     |
//!                                           State machine (4-state):
//!                                           Closed --> Attack --> Open
//!                                                                  |
//!                                           Closed <-- Release <-- Hold
//!                                                     |
//!                                           Gain envelope (smoothed)
//!                                                     |
//! Input --> [Lookahead delay] ----------> * Gain ----> Output
//! ```
//!
//! # Sidechain highpass filter
//!
//! An optional first-order highpass on the sidechain signal prevents
//! low-frequency content (room rumble, bass harmonics) from keeping the gate
//! open. This is common in vocal processing where only mid/high-frequency
//! transients should trigger the gate.
//!
//! # Lookahead
//!
//! A circular delay buffer on the audio path lets the gate "look ahead" of
//! the current sample position. The envelope detector sees the signal before
//! the gate applies gain, so transient onsets are not clipped by a slow
//! attack.
//!
//! # References
//!
//! - Giannoulis, D. et al. "Digital Dynamic Range Compressor Design."
//!   Journal of the Audio Engineering Society, 60(6), 2012.
//! - Zolzer, U. "DAFX: Digital Audio Effects." 2nd ed., Wiley, 2011.
//!
//! Part of #4582, #3351.

use crate::kokoro_chorus_saturation::db_to_linear;
use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Gate state
// ---------------------------------------------------------------------------

/// Current state of the noise gate state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    /// Gate is closed — signal is attenuated by `range_db`.
    Closed,
    /// Gate is opening — gain is ramping from closed to open.
    Attack,
    /// Gate is fully open — signal passes unattenuated.
    Open,
    /// Signal dropped below threshold but gate stays open for `hold_ms`.
    Hold,
    /// Gate is closing — gain is ramping from open to closed.
    Release,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the noise gate.
///
/// Constructed via [`GateConfig::new`] with builder methods.
/// Required for cross-crate use due to `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct GateConfig {
    /// Signal level (dBFS) below which the gate closes.
    /// Range: -80.0 to -20.0. Default: -50.0.
    pub threshold_db: f32,
    /// How quickly (ms) the gate opens when signal exceeds threshold.
    /// Range: 0.01 to 10.0. Default: 0.5.
    pub attack_ms: f32,
    /// How long (ms) the gate stays open after signal drops below threshold.
    /// Range: 0.0 to 500.0. Default: 50.0.
    pub hold_ms: f32,
    /// How slowly (ms) the gate closes after hold expires.
    /// Range: 5.0 to 500.0. Default: 50.0.
    pub release_ms: f32,
    /// How much (dB) to reduce signal when gate is closed.
    /// 0.0 = pass through (gate disabled), -80.0 = full mute.
    /// Range: -80.0 to 0.0. Default: -80.0.
    pub range_db: f32,
    /// Lookahead (ms) to catch transients before they are gated.
    /// Range: 0.0 to 10.0. Default: 1.0.
    pub lookahead_ms: f32,
    /// Highpass filter cutoff (Hz) on sidechain signal.
    /// 0.0 = disabled. Prevents bass from opening gate.
    /// Range: 0.0 to 500.0. Default: 0.0.
    pub sidechain_filter_hz: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            threshold_db: -50.0,
            attack_ms: 0.5,
            hold_ms: 50.0,
            release_ms: 50.0,
            range_db: -80.0,
            lookahead_ms: 1.0,
            sidechain_filter_hz: 0.0,
        }
    }
}

impl GateConfig {
    /// Create a new gate config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the threshold in dBFS.
    #[must_use]
    pub fn with_threshold(mut self, db: f32) -> Self {
        self.threshold_db = db;
        self
    }

    /// Set the attack time in milliseconds.
    #[must_use]
    pub fn with_attack(mut self, ms: f32) -> Self {
        self.attack_ms = ms;
        self
    }

    /// Set the hold time in milliseconds.
    #[must_use]
    pub fn with_hold(mut self, ms: f32) -> Self {
        self.hold_ms = ms;
        self
    }

    /// Set the release time in milliseconds.
    #[must_use]
    pub fn with_release(mut self, ms: f32) -> Self {
        self.release_ms = ms;
        self
    }

    /// Set the range (attenuation depth) in dB.
    #[must_use]
    pub fn with_range(mut self, db: f32) -> Self {
        self.range_db = db;
        self
    }

    /// Set the lookahead time in milliseconds.
    #[must_use]
    pub fn with_lookahead(mut self, ms: f32) -> Self {
        self.lookahead_ms = ms;
        self
    }

    /// Set the sidechain highpass filter cutoff in Hz.
    #[must_use]
    pub fn with_sidechain_filter(mut self, hz: f32) -> Self {
        self.sidechain_filter_hz = hz;
        self
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.threshold_db.is_finite() || self.threshold_db < -80.0 || self.threshold_db > -20.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "threshold_db",
                reason: format!(
                    "threshold_db = {}: must be finite and in [-80, -20]",
                    self.threshold_db,
                ),
            });
        }
        if !self.attack_ms.is_finite() || self.attack_ms < 0.01 || self.attack_ms > 10.0 {
            return Err(KokoroError::InvalidConfig {
                field: "attack_ms",
                reason: format!(
                    "attack_ms = {}: must be finite and in [0.01, 10]",
                    self.attack_ms,
                ),
            });
        }
        if !self.hold_ms.is_finite() || self.hold_ms < 0.0 || self.hold_ms > 500.0 {
            return Err(KokoroError::InvalidConfig {
                field: "hold_ms",
                reason: format!("hold_ms = {}: must be finite and in [0, 500]", self.hold_ms),
            });
        }
        if !self.release_ms.is_finite() || self.release_ms < 5.0 || self.release_ms > 500.0 {
            return Err(KokoroError::InvalidConfig {
                field: "release_ms",
                reason: format!(
                    "release_ms = {}: must be finite and in [5, 500]",
                    self.release_ms,
                ),
            });
        }
        if !self.range_db.is_finite() || self.range_db < -80.0 || self.range_db > 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "range_db",
                reason: format!(
                    "range_db = {}: must be finite and in [-80, 0]",
                    self.range_db,
                ),
            });
        }
        if !self.lookahead_ms.is_finite() || self.lookahead_ms < 0.0 || self.lookahead_ms > 10.0 {
            return Err(KokoroError::InvalidConfig {
                field: "lookahead_ms",
                reason: format!(
                    "lookahead_ms = {}: must be finite and in [0, 10]",
                    self.lookahead_ms,
                ),
            });
        }
        if !self.sidechain_filter_hz.is_finite()
            || self.sidechain_filter_hz < 0.0
            || self.sidechain_filter_hz > 500.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "sidechain_filter_hz",
                reason: format!(
                    "sidechain_filter_hz = {}: must be finite and in [0, 500]",
                    self.sidechain_filter_hz,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// One-pole highpass filter for sidechain
// ---------------------------------------------------------------------------

/// First-order highpass filter for sidechain signal conditioning.
///
/// Prevents low-frequency content (bass, room rumble) from opening the gate.
/// Uses a simple one-pole design: `y[n] = alpha * (y[n-1] + x[n] - x[n-1])`.
#[derive(Debug, Clone)]
struct SidechainHighpass {
    alpha: f32,
    prev_input: f32,
    prev_output: f32,
}

impl SidechainHighpass {
    /// Create a new highpass filter at the given cutoff frequency.
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        // RC = 1 / (2 * pi * cutoff)
        // alpha = RC / (RC + dt)   where dt = 1/sample_rate
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);
        Self {
            alpha,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }

    /// Process a single sample through the highpass filter.
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let output = self.alpha * (self.prev_output + input - self.prev_input);
        self.prev_input = input;
        self.prev_output = output;
        output
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Circular delay buffer for lookahead
// ---------------------------------------------------------------------------

/// Fixed-size circular buffer for lookahead delay.
///
/// The audio signal is delayed by `lookahead_samples` so the envelope
/// detector can see upcoming transients before the gate applies gain.
#[derive(Debug, Clone)]
struct DelayBuffer {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl DelayBuffer {
    /// Create a delay buffer with the given length in samples.
    ///
    /// A length of 0 results in a passthrough (no delay).
    fn new(length_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; length_samples.max(1)],
            write_pos: 0,
        }
    }

    /// Push a new sample and return the oldest sample.
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        if self.buffer.len() <= 1 {
            return input;
        }
        let output = self.buffer[self.write_pos];
        self.buffer[self.write_pos] = input;
        self.write_pos += 1;
        if self.write_pos >= self.buffer.len() {
            self.write_pos = 0;
        }
        output
    }

    /// Reset the delay buffer to silence.
    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// Noise gate processor
// ---------------------------------------------------------------------------

/// Noise gate processor with lookahead, hold, and optional sidechain filtering.
///
/// Implements a 4-state gate (Closed, Attack, Open, Hold, Release) with
/// smoothed gain transitions and optional lookahead delay.
#[derive(Debug, Clone)]
pub struct NoiseGate {
    /// Linear threshold (amplitude).
    threshold_lin: f32,
    /// Linear range gain when gate is fully closed.
    range_gain: f32,
    /// Attack coefficient per sample (one-pole smoothing).
    attack_coeff: f32,
    /// Release coefficient per sample (one-pole smoothing).
    release_coeff: f32,
    /// Hold duration in samples.
    hold_samples: usize,
    /// Current gate state.
    state: GateState,
    /// Current smoothed gain (0.0 = fully closed, 1.0 = open).
    gain: f32,
    /// Peak envelope follower state.
    envelope: f32,
    /// Hold counter: samples remaining in hold state.
    hold_counter: usize,
    /// Lookahead delay buffer (audio path).
    delay: DelayBuffer,
    /// Optional sidechain highpass filter.
    sidechain_hp: Option<SidechainHighpass>,
    /// Envelope detector attack coefficient (fast peak follower).
    env_attack_coeff: f32,
    /// Envelope detector release coefficient.
    env_release_coeff: f32,
}

impl NoiseGate {
    /// Create a new noise gate from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range,
    /// or if `sample_rate` is not positive and finite.
    pub fn new(config: &GateConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("sample_rate = {sample_rate}: must be finite and > 0"),
            });
        }

        let threshold_lin = db_to_linear(config.threshold_db);
        let range_gain = db_to_linear(config.range_db);

        // Time constant conversion: coeff = 1 - exp(-1 / (time_sec * sample_rate))
        // For attack/release envelope smoothing.
        let attack_coeff = time_to_coeff(config.attack_ms, sample_rate);
        let release_coeff = time_to_coeff(config.release_ms, sample_rate);

        let hold_samples = ms_to_samples(config.hold_ms, sample_rate);
        let lookahead_samples = ms_to_samples(config.lookahead_ms, sample_rate);

        // Envelope detector: fast attack (0.1ms), moderate release (5ms).
        // This is the level detector, separate from the gate smoothing.
        let env_attack_coeff = time_to_coeff(0.1, sample_rate);
        let env_release_coeff = time_to_coeff(5.0, sample_rate);

        let sidechain_hp = if config.sidechain_filter_hz > 0.0 {
            Some(SidechainHighpass::new(
                config.sidechain_filter_hz,
                sample_rate,
            ))
        } else {
            None
        };

        Ok(Self {
            threshold_lin,
            range_gain,
            attack_coeff,
            release_coeff,
            hold_samples,
            state: GateState::Closed,
            gain: 0.0,
            envelope: 0.0,
            hold_counter: 0,
            delay: DelayBuffer::new(lookahead_samples),
            sidechain_hp,
            env_attack_coeff,
            env_release_coeff,
        })
    }

    /// Process audio in-place through the noise gate.
    ///
    /// The gate operates sample-by-sample:
    /// 1. Optionally highpass-filter the sidechain signal.
    /// 2. Update the peak envelope follower.
    /// 3. Advance the state machine (Closed/Attack/Open/Hold/Release).
    /// 4. Smooth the gain envelope.
    /// 5. Apply gain to the (optionally delayed) audio.
    pub fn process(&mut self, audio: &mut [f32]) {
        for i in 0..audio.len() {
            let input = audio[i];

            // Sidechain: optionally filter, then take absolute value.
            let sidechain = match &mut self.sidechain_hp {
                Some(hp) => hp.process(input).abs(),
                None => input.abs(),
            };

            // Peak envelope follower (asymmetric attack/release).
            if sidechain > self.envelope {
                self.envelope += self.env_attack_coeff * (sidechain - self.envelope);
            } else {
                self.envelope += self.env_release_coeff * (sidechain - self.envelope);
            }

            // State machine transitions.
            let above_threshold = self.envelope >= self.threshold_lin;

            match self.state {
                GateState::Closed => {
                    if above_threshold {
                        self.state = GateState::Attack;
                    }
                }
                GateState::Attack => {
                    if !above_threshold {
                        // Signal dropped during attack — go to release.
                        self.state = GateState::Release;
                    } else if self.gain >= 1.0 - 1e-6 {
                        self.state = GateState::Open;
                        self.gain = 1.0;
                    }
                }
                GateState::Open => {
                    if !above_threshold {
                        if self.hold_samples > 0 {
                            self.state = GateState::Hold;
                            self.hold_counter = self.hold_samples;
                        } else {
                            self.state = GateState::Release;
                        }
                    }
                }
                GateState::Hold => {
                    if above_threshold {
                        self.state = GateState::Open;
                    } else if self.hold_counter == 0 {
                        self.state = GateState::Release;
                    } else {
                        self.hold_counter -= 1;
                    }
                }
                GateState::Release => {
                    if above_threshold {
                        self.state = GateState::Attack;
                    } else if self.gain <= 1e-6 {
                        self.state = GateState::Closed;
                        self.gain = 0.0;
                    }
                }
            }

            // Target gain based on state.
            let target_gain = match self.state {
                GateState::Closed => 0.0,
                GateState::Attack | GateState::Open | GateState::Hold => 1.0,
                GateState::Release => 0.0,
            };

            // Smooth gain envelope (attack when opening, release when closing).
            let coeff = if target_gain > self.gain {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.gain += coeff * (target_gain - self.gain);

            // Apply gain: interpolate between range_gain (closed) and 1.0 (open).
            let applied_gain = self.range_gain + self.gain * (1.0 - self.range_gain);

            // Delay the audio for lookahead, then apply gain.
            let delayed = self.delay.process(input);
            audio[i] = delayed * applied_gain;
        }
    }

    /// Return the current gate state.
    #[must_use]
    pub fn gate_state(&self) -> GateState {
        self.state
    }

    /// Return the current smoothed gain value (0.0 = fully closed, 1.0 = open).
    #[must_use]
    pub fn current_gain(&self) -> f32 {
        self.gain
    }

    /// Reset the gate state machine, envelope, delay buffer, and filters.
    pub fn reset(&mut self) {
        self.state = GateState::Closed;
        self.gain = 0.0;
        self.envelope = 0.0;
        self.hold_counter = 0;
        self.delay.reset();
        if let Some(hp) = &mut self.sidechain_hp {
            hp.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Per-voice convenience function
// ---------------------------------------------------------------------------

/// Apply noise gating to each voice independently.
///
/// Each voice gets its own `NoiseGate` instance so per-voice silence is
/// cleaned up without cross-voice interference.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if the gate configuration is invalid.
pub fn apply_noise_gate(
    voices: &mut [Vec<f32>],
    config: &GateConfig,
    sample_rate: f32,
) -> Result<(), KokoroError> {
    for voice in voices.iter_mut() {
        let mut gate = NoiseGate::new(config, sample_rate)?;
        gate.process(voice.as_mut_slice());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a time constant in milliseconds to a one-pole coefficient.
///
/// `coeff = 1 - exp(-1 / (time_sec * sample_rate))`
///
/// When `time_ms` is very small (< 0.001), returns 1.0 (instant response).
#[inline]
fn time_to_coeff(time_ms: f32, sample_rate: f32) -> f32 {
    if time_ms < 0.001 {
        return 1.0;
    }
    let time_sec = time_ms / 1000.0;
    let samples = time_sec * sample_rate;
    1.0 - (-1.0 / samples).exp()
}

/// Convert milliseconds to a sample count (rounded to nearest integer).
#[inline]
fn ms_to_samples(ms: f32, sample_rate: f32) -> usize {
    ((ms / 1000.0) * sample_rate).round() as usize
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 24000.0;

    /// Generate a silent buffer of given length in samples.
    fn silence(len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    /// Generate a constant-amplitude tone (sine wave).
    fn tone(amplitude: f32, freq_hz: f32, duration_samples: usize, sample_rate: f32) -> Vec<f32> {
        (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                amplitude * (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect()
    }

    // -- Config validation tests -----------------------------------------------

    #[test]
    fn test_default_config_is_valid() {
        let config = GateConfig::new();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn test_threshold_out_of_range() {
        let config = GateConfig::new().with_threshold(-90.0);
        assert!(config.validate().is_err());

        let config = GateConfig::new().with_threshold(-10.0);
        assert!(config.validate().is_err());

        let config = GateConfig::new().with_threshold(f32::NAN);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_attack_out_of_range() {
        let config = GateConfig::new().with_attack(0.0);
        assert!(config.validate().is_err());

        let config = GateConfig::new().with_attack(20.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_release_out_of_range() {
        let config = GateConfig::new().with_release(1.0);
        assert!(config.validate().is_err());

        let config = GateConfig::new().with_release(1000.0);
        assert!(config.validate().is_err());
    }

    // -- Gate behavior tests ---------------------------------------------------

    #[test]
    fn test_gate_closes_on_silence() {
        let config = GateConfig::new().with_threshold(-40.0).with_range(-80.0);
        let mut gate = NoiseGate::new(&config, SR).expect("valid config");

        let mut audio = silence(4800); // 200ms of silence
        gate.process(&mut audio);

        // All samples should be near-zero (gate closed, -80dB attenuation).
        let max_sample = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            max_sample < 1e-4,
            "silence should remain near-zero, got max={max_sample}",
        );
        assert_eq!(gate.gate_state(), GateState::Closed);
    }

    #[test]
    fn test_gate_opens_on_signal() {
        let config = GateConfig::new()
            .with_threshold(-40.0)
            .with_attack(0.5)
            .with_range(-80.0);
        let mut gate = NoiseGate::new(&config, SR).expect("valid config");

        // 0.1 amplitude ~ -20 dBFS, well above -40 dB threshold.
        let mut audio = tone(0.1, 440.0, 4800, SR);
        gate.process(&mut audio);

        // After processing, the tail of the buffer should have signal close to
        // the original amplitude (gate fully open).
        let tail_start = audio.len() - 480; // last 20ms
        let tail_rms: f32 = (audio[tail_start..].iter().map(|s| s * s).sum::<f32>() / 480.0).sqrt();

        // Original RMS of 0.1 * sin ~ 0.0707. Gate open should pass most of it.
        assert!(
            tail_rms > 0.03,
            "gate should open for loud signal, got tail_rms={tail_rms}",
        );
    }

    #[test]
    fn test_hold_keeps_gate_open() {
        // Signal above threshold for 100ms, then silence. Hold = 100ms.
        let config = GateConfig::new()
            .with_threshold(-40.0)
            .with_hold(100.0)
            .with_release(50.0)
            .with_attack(0.5)
            .with_range(-80.0)
            .with_lookahead(0.0);
        let mut gate = NoiseGate::new(&config, SR).expect("valid config");

        let signal_len = 2400; // 100ms at 24kHz
        let silence_len = 4800; // 200ms
        let mut audio: Vec<f32> = tone(0.1, 440.0, signal_len, SR);
        audio.extend(silence(silence_len));

        gate.process(&mut audio);

        // During the hold period (100ms after signal ends = samples 2400..4800),
        // the gate should still be at least partially open. Check that the gain
        // at sample 3000 (about 25ms into hold) is meaningful.
        // We test indirectly: the first few silence samples after signal should
        // be near-zero (input is zero), but the gate gain should be ~1.0.
        // Since input is zero, output is zero regardless of gain.
        // Better test: check gate_state after processing a partial buffer.
        let config2 = config;
        let mut gate2 = NoiseGate::new(&config2, SR).expect("valid config");

        // First: feed the signal portion.
        let mut signal_part = tone(0.1, 440.0, signal_len, SR);
        gate2.process(&mut signal_part);
        assert!(
            gate2.gate_state() == GateState::Open || gate2.gate_state() == GateState::Attack,
            "gate should be open/attacking after signal, got {:?}",
            gate2.gate_state(),
        );

        // Then: feed a short silence (25ms = 600 samples), should be in Hold.
        let mut short_silence = silence(600);
        gate2.process(&mut short_silence);
        assert!(
            gate2.gate_state() == GateState::Hold || gate2.gate_state() == GateState::Open,
            "gate should be in hold after short silence, got {:?}",
            gate2.gate_state(),
        );
        // Gain should still be high during hold.
        assert!(
            gate2.current_gain() > 0.5,
            "gain should remain high during hold, got {}",
            gate2.current_gain(),
        );
    }

    #[test]
    fn test_range_controls_depth() {
        // With range = -6dB, closed gate should still pass ~50% amplitude.
        let config = GateConfig::new()
            .with_threshold(-20.0)
            .with_range(-6.0)
            .with_lookahead(0.0);
        let mut gate = NoiseGate::new(&config, SR).expect("valid config");

        // Low amplitude signal below threshold.
        let amp = 0.01; // ~ -40 dBFS, below -20 threshold
        let mut audio = tone(amp, 440.0, 4800, SR);
        gate.process(&mut audio);

        // -6dB range means closed gain ~ 0.501. Signal should be attenuated
        // but not fully muted.
        let max_sample = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let range_gain = db_to_linear(-6.0);

        // The output should be approximately amp * range_gain.
        assert!(
            max_sample > amp * range_gain * 0.5,
            "range=-6dB should still pass signal, max={}, expected > {}",
            max_sample,
            amp * range_gain * 0.5,
        );
        assert!(
            max_sample < amp * 1.1,
            "output should be attenuated, max={max_sample}, input amp={amp}",
        );
    }

    #[test]
    fn test_lookahead_catches_transients() {
        // Without lookahead: the first few samples of a transient are gated.
        // With lookahead: the gate opens early so the transient passes.
        let config_no_la = GateConfig::new()
            .with_threshold(-40.0)
            .with_attack(0.5)
            .with_range(-80.0)
            .with_lookahead(0.0);

        let config_la = GateConfig::new()
            .with_threshold(-40.0)
            .with_attack(0.5)
            .with_range(-80.0)
            .with_lookahead(2.0); // 2ms lookahead

        let mut gate_no_la = NoiseGate::new(&config_no_la, SR).expect("valid");
        let mut gate_la = NoiseGate::new(&config_la, SR).expect("valid");

        // Start with 50ms silence, then a transient burst.
        let pre_silence = 1200; // 50ms
        let burst_len = 2400; // 100ms
        let mut audio_no_la: Vec<f32> = silence(pre_silence);
        audio_no_la.extend(tone(0.1, 1000.0, burst_len, SR));
        let mut audio_la = audio_no_la.clone();

        gate_no_la.process(&mut audio_no_la);
        gate_la.process(&mut audio_la);

        // Sum energy in the first 2ms of the burst (samples 1200..1248).
        let burst_start = pre_silence;
        let check_len = 48; // 2ms at 24kHz
        let energy_no_la: f32 = audio_no_la[burst_start..burst_start + check_len]
            .iter()
            .map(|s| s * s)
            .sum();
        let _energy_la: f32 = audio_la[burst_start..burst_start + check_len]
            .iter()
            .map(|s| s * s)
            .sum();

        // With lookahead, the gate should have opened earlier, preserving
        // more transient energy. Note: with lookahead the audio is delayed,
        // so the transient arrives later. We compare energy at the same
        // absolute position — the lookahead version has less energy here
        // because the signal is delayed. The real benefit is that when the
        // transient arrives in the lookahead version, the gate is already open.
        //
        // To properly test: check energy in the delayed transient window.
        let la_samples = ms_to_samples(2.0, SR);
        let la_burst_start = burst_start + la_samples;
        if la_burst_start + check_len <= audio_la.len() {
            let energy_la_shifted: f32 = audio_la[la_burst_start..la_burst_start + check_len]
                .iter()
                .map(|s| s * s)
                .sum();
            // The lookahead version should preserve transient energy better
            // at its delayed position than the no-lookahead version at its
            // immediate position (where attack is still ramping).
            assert!(
                energy_la_shifted >= energy_no_la * 0.8,
                "lookahead should preserve transient energy: la={energy_la_shifted}, no_la={energy_no_la}",
            );
        }
    }

    #[test]
    fn test_sidechain_filter_ignores_bass() {
        // A pure bass tone (60 Hz) below the sidechain HP cutoff should not
        // open the gate when the sidechain filter is enabled.
        let config_no_sc = GateConfig::new()
            .with_threshold(-30.0)
            .with_range(-80.0)
            .with_sidechain_filter(0.0)
            .with_lookahead(0.0);

        let config_sc = GateConfig::new()
            .with_threshold(-30.0)
            .with_range(-80.0)
            .with_sidechain_filter(200.0) // 200 Hz HP on sidechain
            .with_lookahead(0.0);

        let mut gate_no_sc = NoiseGate::new(&config_no_sc, SR).expect("valid");
        let mut gate_sc = NoiseGate::new(&config_sc, SR).expect("valid");

        // -20 dBFS 60 Hz bass tone (above the -30 dB threshold in amplitude).
        let mut audio_no_sc = tone(0.1, 60.0, 4800, SR);
        let mut audio_sc = audio_no_sc.clone();

        gate_no_sc.process(&mut audio_no_sc);
        gate_sc.process(&mut audio_sc);

        // Without sidechain filter: gate should open (bass opens gate).
        let rms_no_sc: f32 =
            (audio_no_sc.iter().map(|s| s * s).sum::<f32>() / audio_no_sc.len() as f32).sqrt();

        // With sidechain filter: gate should stay more closed (bass filtered out).
        let rms_sc: f32 =
            (audio_sc.iter().map(|s| s * s).sum::<f32>() / audio_sc.len() as f32).sqrt();

        // The sidechain-filtered version should have significantly lower RMS
        // because the 60 Hz tone is filtered out of the detector path.
        assert!(
            rms_sc < rms_no_sc * 0.7,
            "sidechain HP should attenuate bass detection: sc_rms={rms_sc}, no_sc_rms={rms_no_sc}",
        );
    }

    #[test]
    fn test_apply_noise_gate_per_voice() {
        let config = GateConfig::new()
            .with_threshold(-40.0)
            .with_range(-80.0)
            .with_lookahead(0.0);

        let mut voices = vec![
            silence(2400),              // voice 0: silence, should be gated
            tone(0.1, 440.0, 2400, SR), // voice 1: signal, should pass
        ];

        apply_noise_gate(&mut voices, &config, SR).expect("valid config");

        let rms_0: f32 =
            (voices[0].iter().map(|s| s * s).sum::<f32>() / voices[0].len() as f32).sqrt();
        let rms_1: f32 =
            (voices[1].iter().map(|s| s * s).sum::<f32>() / voices[1].len() as f32).sqrt();

        assert!(
            rms_0 < 1e-4,
            "silent voice should remain gated, rms={rms_0}"
        );
        assert!(rms_1 > 0.01, "signal voice should pass, rms={rms_1}");
    }

    #[test]
    fn test_reset_clears_state() {
        let config = GateConfig::new()
            .with_threshold(-40.0)
            .with_hold(100.0)
            .with_lookahead(1.0);
        let mut gate = NoiseGate::new(&config, SR).expect("valid config");

        // Process some signal to put the gate into a non-default state.
        let mut audio = tone(0.1, 440.0, 2400, SR);
        gate.process(&mut audio);
        assert_ne!(gate.gate_state(), GateState::Closed);

        // Reset should bring everything back to initial state.
        gate.reset();
        assert_eq!(gate.gate_state(), GateState::Closed);
        assert!((gate.current_gain() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_sample_rate() {
        let config = GateConfig::new();
        assert!(NoiseGate::new(&config, 0.0).is_err());
        assert!(NoiseGate::new(&config, -1.0).is_err());
        assert!(NoiseGate::new(&config, f32::NAN).is_err());
        assert!(NoiseGate::new(&config, f32::INFINITY).is_err());
    }

    #[test]
    fn test_range_zero_is_passthrough() {
        // range_db = 0 means even when closed, gain = 1.0 (no attenuation).
        let config = GateConfig::new()
            .with_threshold(-20.0)
            .with_range(0.0)
            .with_lookahead(0.0);
        let mut gate = NoiseGate::new(&config, SR).expect("valid config");

        let original = tone(0.01, 440.0, 2400, SR);
        let mut audio = original.clone();
        gate.process(&mut audio);

        // Output should be essentially unchanged (range=0 means no attenuation).
        let diff_rms: f32 = (original
            .iter()
            .zip(audio.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / original.len() as f32)
            .sqrt();

        assert!(
            diff_rms < 1e-4,
            "range=0 should be passthrough, diff_rms={diff_rms}",
        );
    }
}
