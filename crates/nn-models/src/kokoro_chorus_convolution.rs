// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! FFT-based convolution reverb for the Kokoro chorus system.
//!
//! Convolves audio with an impulse response (IR) to simulate real acoustic
//! spaces. Uses overlap-add FFT convolution for O(N log N) efficiency instead
//! of O(N*M) direct convolution.
//!
//! # Architecture
//!
//! ```text
//! Input audio
//!   → Pre-delay (circular buffer, 0-100ms)
//!   → Overlap-add FFT convolution with IR
//!   → Wet/dry mix
//!   → Output audio
//! ```
//!
//! # Synthetic Impulse Responses
//!
//! [`generate_synthetic_ir`] creates IRs for common acoustic spaces without
//! requiring external WAV files. Synthesized via exponentially decaying noise
//! with early reflections modeled as discrete delay taps.
//!
//! # References
//!
//! - Smith, J.O. (2007). "Mathematics of the Discrete Fourier Transform (DFT)."
//!   W3K Publishing. Chapter on fast convolution via overlap-add.
//! - Moorer, J.A. (1979). "About This Reverberation Business."
//!   Computer Music Journal, 3(2), 13-28.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for FFT convolution reverb.
///
/// Controls the wet/dry balance, impulse response length limit, pre-delay,
/// and IR normalization. Built via method chaining.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ConvolutionConfig {
    /// Wet/dry ratio: 0.0 = fully dry, 1.0 = fully wet.
    ///
    /// Default: `0.20`. Convolution reverbs are denser than Schroeder reverbs,
    /// so a slightly higher default than the Schroeder `reverb_mix` (0.15)
    /// produces a comparable perceived depth.
    pub wet_mix: f32,

    /// Maximum impulse response length in samples.
    ///
    /// Default: `48000` (2 seconds at 24kHz). IRs longer than this are
    /// truncated with a fade-out to avoid clicks. Caps memory usage for
    /// the FFT overlap buffer.
    pub ir_length_limit: usize,

    /// Pre-delay in milliseconds before reverb onset (0.0-100.0).
    ///
    /// Default: `0.0`. Adds a gap between the dry signal and the reverb
    /// tail, preserving transient clarity. 10-30ms is typical for vocals.
    pub pre_delay_ms: f32,

    /// Whether to normalize the IR energy to prevent level changes.
    ///
    /// Default: `true`. Scales the IR so its RMS energy equals 1.0,
    /// ensuring the wet signal has approximately the same loudness as the
    /// dry signal regardless of the IR's original level.
    pub normalize_ir: bool,
}

impl Default for ConvolutionConfig {
    fn default() -> Self {
        Self {
            wet_mix: 0.20,
            ir_length_limit: 48000,
            pre_delay_ms: 0.0,
            normalize_ir: true,
        }
    }
}

impl ConvolutionConfig {
    /// Create a new convolution config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the wet/dry mix ratio.
    #[must_use]
    pub fn with_wet_mix(mut self, mix: f32) -> Self {
        self.wet_mix = mix;
        self
    }

    /// Set the maximum IR length in samples.
    #[must_use]
    pub fn with_ir_length_limit(mut self, limit: usize) -> Self {
        self.ir_length_limit = limit;
        self
    }

    /// Set the pre-delay in milliseconds.
    #[must_use]
    pub fn with_pre_delay_ms(mut self, ms: f32) -> Self {
        self.pre_delay_ms = ms;
        self
    }

    /// Enable or disable IR normalization.
    #[must_use]
    pub fn with_normalize_ir(mut self, normalize: bool) -> Self {
        self.normalize_ir = normalize;
        self
    }

    /// Validate that all parameters are within valid ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.wet_mix.is_finite() || !(0.0..=1.0).contains(&self.wet_mix) {
            return Err(KokoroError::InvalidConfig {
                field: "wet_mix",
                reason: format!(
                    "wet_mix = {}: must be finite and in [0.0, 1.0]",
                    self.wet_mix,
                ),
            });
        }
        if self.ir_length_limit == 0 {
            return Err(KokoroError::InvalidConfig {
                field: "ir_length_limit",
                reason: "ir_length_limit must be > 0".to_string(),
            });
        }
        if !self.pre_delay_ms.is_finite() || !(0.0..=100.0).contains(&self.pre_delay_ms) {
            return Err(KokoroError::InvalidConfig {
                field: "pre_delay_ms",
                reason: format!(
                    "pre_delay_ms = {}: must be finite and in [0.0, 100.0]",
                    self.pre_delay_ms,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Synthetic room types
// ---------------------------------------------------------------------------

/// Predefined acoustic spaces for synthetic impulse response generation.
///
/// Each variant models a different room geometry and material absorption
/// profile. Generated via exponentially decaying noise with discrete
/// early-reflection taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SyntheticRoom {
    /// Small rehearsal room. RT60 ~ 0.3s. Tight, controlled reflections.
    SmallRoom,
    /// Medium concert hall. RT60 ~ 1.2s. Balanced early/late energy.
    MediumHall,
    /// Large cathedral. RT60 ~ 3.0s. Long diffuse tail with sparse early reflections.
    Cathedral,
    /// Metallic plate reverb. RT60 ~ 1.5s. Dense, bright, no early reflections.
    Plate,
}

// ---------------------------------------------------------------------------
// Simple radix-2 DIT FFT (self-contained, no external dependency)
// ---------------------------------------------------------------------------

/// Next power of two >= n.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n.next_power_of_two()
}

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

/// In-place inverse FFT. Conjugate → FFT → conjugate → scale by 1/N.
fn ifft(data: &mut [(f32, f32)]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    // Conjugate.
    for (_, im) in data.iter_mut() {
        *im = -*im;
    }
    fft(data);
    // Conjugate and scale.
    let scale = 1.0 / n as f32;
    for (re, im) in data.iter_mut() {
        *re *= scale;
        *im = -*im * scale;
    }
}

// ---------------------------------------------------------------------------
// Pre-delay circular buffer
// ---------------------------------------------------------------------------

/// Simple circular buffer for pre-delay.
struct PreDelayBuffer {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl PreDelayBuffer {
    fn new(delay_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
        }
    }

    /// Push a sample and return the delayed sample.
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.buffer[self.write_pos];
        self.buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        delayed
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// Convolution reverb processor
// ---------------------------------------------------------------------------

/// FFT-based convolution reverb processor.
///
/// Uses overlap-add to convolve audio with an impulse response in O(N log N)
/// time. Supports pre-delay and wet/dry mixing.
pub struct ConvolutionReverb {
    /// FFT of the padded impulse response (length = fft_size).
    ir_fft: Vec<(f32, f32)>,
    /// FFT block size (next power of 2 >= 2 * ir_len, for linear convolution).
    fft_size: usize,
    /// Original IR length before padding.
    ir_len: usize,
    /// Overlap buffer from previous block (length = ir_len - 1).
    overlap: Vec<f32>,
    /// Pre-delay circular buffer.
    pre_delay: PreDelayBuffer,
    /// Wet/dry mix ratio.
    wet_mix: f32,
}

impl ConvolutionReverb {
    /// Create a new convolution reverb processor.
    ///
    /// Starts with an empty (pass-through) IR. Call [`load_ir`](Self::load_ir)
    /// or load a [`generate_synthetic_ir`] result before processing.
    pub fn new(config: &ConvolutionConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let delay_samples = (config.pre_delay_ms / 1000.0 * KOKORO_SAMPLE_RATE as f32) as usize;
        Ok(Self {
            ir_fft: Vec::new(),
            fft_size: 0,
            ir_len: 0,
            overlap: Vec::new(),
            pre_delay: PreDelayBuffer::new(delay_samples),
            wet_mix: config.wet_mix,
        })
    }

    /// Load an impulse response.
    ///
    /// Truncates to `ir_length_limit` (from the config used at construction)
    /// with a short fade-out to avoid clicks. Optionally normalizes IR energy.
    pub fn load_ir(&mut self, ir: &[f32], config: &ConvolutionConfig) {
        if ir.is_empty() {
            self.ir_fft.clear();
            self.fft_size = 0;
            self.ir_len = 0;
            self.overlap.clear();
            return;
        }

        // Truncate with fade-out.
        let limit = config.ir_length_limit.min(ir.len());
        let mut truncated: Vec<f32> = ir[..limit].to_vec();
        // Apply 64-sample fade-out at the end to prevent clicks.
        let fade_len = 64.min(truncated.len());
        let start = truncated.len() - fade_len;
        for i in 0..fade_len {
            let t = i as f32 / fade_len as f32;
            // Cosine fade: 1 → 0.
            let gain = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
            truncated[start + i] *= gain;
        }

        // Normalize IR energy if requested.
        if config.normalize_ir {
            let rms =
                (truncated.iter().map(|s| s * s).sum::<f32>() / truncated.len() as f32).sqrt();
            if rms > 1e-10 {
                let scale = 1.0 / rms;
                for s in &mut truncated {
                    *s *= scale;
                }
            }
        }

        self.ir_len = truncated.len();
        // FFT size must be >= ir_len + block_size - 1 for linear convolution.
        // We use block_size = ir_len, so fft_size >= 2 * ir_len - 1.
        self.fft_size = next_power_of_two(2 * self.ir_len);

        // Pre-compute IR FFT.
        let mut ir_buf: Vec<(f32, f32)> = vec![(0.0, 0.0); self.fft_size];
        for (i, &s) in truncated.iter().enumerate() {
            ir_buf[i] = (s, 0.0);
        }
        fft(&mut ir_buf);
        self.ir_fft = ir_buf;

        // Reset overlap buffer.
        self.overlap = vec![0.0; self.ir_len - 1];
    }

    /// Process a block of mono audio through the convolution reverb.
    ///
    /// Returns a new buffer of the same length as the input. The output
    /// is the wet/dry mix of the original signal and the convolved signal.
    /// Internal state (overlap buffer, pre-delay) is maintained across calls
    /// for seamless streaming.
    pub fn process(&mut self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() || self.ir_fft.is_empty() {
            return audio.to_vec();
        }

        let input_len = audio.len();
        let block_size = self.ir_len;
        let fft_size = self.fft_size;

        // Accumulate full output (input_len samples).
        let mut output = vec![0.0f32; input_len];

        // Process in blocks of `block_size`.
        let mut pos = 0;
        while pos < input_len {
            let end = (pos + block_size).min(input_len);
            let chunk_len = end - pos;

            // Zero-pad chunk to fft_size.
            let mut buf: Vec<(f32, f32)> = vec![(0.0, 0.0); fft_size];
            for i in 0..chunk_len {
                buf[i] = (audio[pos + i], 0.0);
            }

            // Forward FFT.
            fft(&mut buf);

            // Pointwise multiply with IR FFT.
            for i in 0..fft_size {
                let (a_re, a_im) = buf[i];
                let (b_re, b_im) = self.ir_fft[i];
                buf[i] = (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re);
            }

            // Inverse FFT.
            ifft(&mut buf);

            // Overlap-add: first part gets added the previous overlap.
            let conv_len = chunk_len + self.ir_len - 1;
            let overlap_len = self.overlap.len(); // ir_len - 1

            for i in 0..chunk_len.min(input_len - pos) {
                let mut val = buf[i].0;
                if i < overlap_len {
                    val += self.overlap[i];
                }
                output[pos + i] = val;
            }

            // Save new overlap.
            let new_overlap_start = chunk_len;
            let new_overlap_end = conv_len.min(fft_size);
            let new_overlap_len = new_overlap_end.saturating_sub(new_overlap_start);

            // Reset overlap buffer and fill with tail samples.
            self.overlap.fill(0.0);
            // Ensure overlap is large enough.
            if self.overlap.len() < new_overlap_len {
                self.overlap.resize(new_overlap_len, 0.0);
            }
            for i in 0..new_overlap_len.min(self.overlap.len()) {
                self.overlap[i] = buf[new_overlap_start + i].0;
            }

            pos += block_size;
        }

        // Apply pre-delay and wet/dry mix.
        let dry_mix = 1.0 - self.wet_mix;
        for i in 0..input_len {
            let wet = self.pre_delay.process(output[i]);
            output[i] = audio[i] * dry_mix + wet * self.wet_mix;
        }

        output
    }

    /// Clear all internal state (overlap buffer, pre-delay).
    ///
    /// Call this when switching to a new audio stream to avoid artifacts
    /// from the previous stream's tail.
    pub fn reset(&mut self) {
        self.overlap.fill(0.0);
        self.pre_delay.reset();
    }
}

// ---------------------------------------------------------------------------
// Synthetic impulse response generation
// ---------------------------------------------------------------------------

/// Generate a synthetic impulse response for a given room type.
///
/// Creates an IR at `KOKORO_SAMPLE_RATE` (24kHz) using exponentially decaying
/// noise with discrete early-reflection taps. The result can be loaded via
/// [`ConvolutionReverb::load_ir`].
///
/// # Algorithm
///
/// 1. Generate white noise of length `RT60 * sample_rate`.
/// 2. Apply exponential decay envelope: `exp(-6.9 * t / RT60)` (60dB decay).
/// 3. Add early reflections as discrete attenuated taps at physically-motivated
///    delay times (wall distances → delay → attenuation).
/// 4. Plate reverb uses denser noise with a brighter decay (less HF damping).
pub fn generate_synthetic_ir(room_type: SyntheticRoom) -> Vec<f32> {
    let sr = KOKORO_SAMPLE_RATE as f32;

    let (rt60, early_taps, high_shelf_damping) = match room_type {
        SyntheticRoom::SmallRoom => (
            0.3,
            // (delay_ms, gain) — short delays from nearby walls.
            vec![(5.0, 0.8), (11.0, 0.6), (17.0, 0.4), (23.0, 0.25)],
            0.4,
        ),
        SyntheticRoom::MediumHall => (
            1.2,
            vec![
                (12.0, 0.7),
                (25.0, 0.55),
                (38.0, 0.4),
                (52.0, 0.3),
                (68.0, 0.2),
                (85.0, 0.12),
            ],
            0.3,
        ),
        SyntheticRoom::Cathedral => (
            3.0,
            vec![
                (20.0, 0.65),
                (45.0, 0.5),
                (72.0, 0.38),
                (105.0, 0.28),
                (145.0, 0.18),
                (190.0, 0.12),
                (250.0, 0.07),
            ],
            0.25,
        ),
        SyntheticRoom::Plate => (
            1.5,
            // No early reflections — plate reverbs are all diffuse tail.
            vec![],
            0.1, // Bright: minimal HF damping.
        ),
    };

    let ir_len = (rt60 * sr) as usize;
    let mut ir = vec![0.0f32; ir_len];

    // 1. Deterministic pseudo-random noise (LCG for reproducibility).
    //    Using a simple linear congruential generator so synthetic IRs are
    //    identical across runs (no dependency on thread_rng or OsRng).
    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;
    for sample in ir.iter_mut() {
        // LCG: Numerical Recipes parameters.
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Convert to [-1.0, 1.0].
        let noise = ((rng_state >> 33) as i32) as f32 / (i32::MAX as f32);
        *sample = noise;
    }

    // 2. Exponential decay envelope: -60dB at RT60.
    //    decay_rate = ln(1000) / RT60 ≈ 6.908 / RT60.
    let decay_rate = 6.908 / rt60;
    for (i, sample) in ir.iter_mut().enumerate() {
        let t = i as f32 / sr;
        let envelope = (-decay_rate * t).exp();
        *sample *= envelope;
    }

    // 3. Simple high-frequency damping via one-pole lowpass on the noise tail.
    //    Simulates air absorption and surface material absorption.
    if high_shelf_damping > 0.0 {
        let alpha = high_shelf_damping;
        let mut prev = 0.0f32;
        for sample in ir.iter_mut() {
            *sample = *sample * (1.0 - alpha) + prev * alpha;
            prev = *sample;
        }
    }

    // 4. Add early reflections as discrete impulses.
    for (delay_ms, gain) in &early_taps {
        let delay_samples = (*delay_ms / 1000.0 * sr) as usize;
        if delay_samples < ir_len {
            ir[delay_samples] += gain;
        }
    }

    // 5. Ensure the very first sample is 1.0 (direct sound component).
    //    The convolution reverb's wet path includes this; the dry path
    //    in process() handles the original signal separately.
    if !ir.is_empty() {
        ir[0] = 1.0;
    }

    ir
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// An impulse IR (single sample = 1.0) should reproduce the input exactly
    /// when wet_mix = 1.0 and pre_delay = 0.
    #[test]
    fn test_identity_ir_reproduces_input() {
        let config = ConvolutionConfig::new()
            .with_wet_mix(1.0)
            .with_normalize_ir(false)
            .with_pre_delay_ms(0.0);
        let mut reverb = ConvolutionReverb::new(&config).unwrap();
        reverb.load_ir(&[1.0], &config);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = reverb.process(&input);

        assert_eq!(output.len(), input.len());
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (inp - out).abs() < 1e-4,
                "sample {i}: input={inp}, output={out}, diff={}",
                (inp - out).abs(),
            );
        }
    }

    /// wet_mix = 0.0 should produce the dry signal unchanged.
    #[test]
    fn test_wet_zero_is_dry() {
        let config = ConvolutionConfig::new()
            .with_wet_mix(0.0)
            .with_normalize_ir(false);
        let mut reverb = ConvolutionReverb::new(&config).unwrap();

        let ir = generate_synthetic_ir(SyntheticRoom::MediumHall);
        reverb.load_ir(&ir, &config);

        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin()).collect();
        let output = reverb.process(&input);

        assert_eq!(output.len(), input.len());
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (inp - out).abs() < 1e-6,
                "sample {i}: input={inp}, output={out}",
            );
        }
    }

    /// Synthetic IRs should have reasonable lengths.
    #[test]
    fn test_synthetic_ir_lengths() {
        let sr = KOKORO_SAMPLE_RATE as f32;

        let small = generate_synthetic_ir(SyntheticRoom::SmallRoom);
        assert_eq!(small.len(), (0.3 * sr) as usize);

        let hall = generate_synthetic_ir(SyntheticRoom::MediumHall);
        assert_eq!(hall.len(), (1.2 * sr) as usize);

        let cathedral = generate_synthetic_ir(SyntheticRoom::Cathedral);
        assert_eq!(cathedral.len(), (3.0 * sr) as usize);

        let plate = generate_synthetic_ir(SyntheticRoom::Plate);
        assert_eq!(plate.len(), (1.5 * sr) as usize);
    }

    /// Synthetic IRs should start with 1.0 (direct sound).
    #[test]
    fn test_synthetic_ir_starts_with_direct_sound() {
        for room in [
            SyntheticRoom::SmallRoom,
            SyntheticRoom::MediumHall,
            SyntheticRoom::Cathedral,
            SyntheticRoom::Plate,
        ] {
            let ir = generate_synthetic_ir(room);
            assert!(
                (ir[0] - 1.0).abs() < 1e-6,
                "{room:?}: first sample should be 1.0, got {}",
                ir[0],
            );
        }
    }

    /// Synthetic IRs should decay — the tail should be quieter than the head.
    #[test]
    fn test_synthetic_ir_decays() {
        for room in [
            SyntheticRoom::SmallRoom,
            SyntheticRoom::MediumHall,
            SyntheticRoom::Cathedral,
            SyntheticRoom::Plate,
        ] {
            let ir = generate_synthetic_ir(room);
            let head_energy: f32 = ir[..ir.len() / 4].iter().map(|s| s * s).sum();
            let tail_energy: f32 = ir[3 * ir.len() / 4..].iter().map(|s| s * s).sum();
            assert!(
                tail_energy < head_energy,
                "{room:?}: tail_energy ({tail_energy}) should be < head_energy ({head_energy})",
            );
        }
    }

    /// Convolution output with a multi-sample IR should be non-trivially
    /// different from the input (wet signal adds reverb energy).
    #[test]
    fn test_convolution_adds_energy() {
        let config = ConvolutionConfig::new()
            .with_wet_mix(0.5)
            .with_normalize_ir(true);
        let mut reverb = ConvolutionReverb::new(&config).unwrap();

        let ir = generate_synthetic_ir(SyntheticRoom::SmallRoom);
        reverb.load_ir(&ir, &config);

        // Short impulse as input: one sample = 1.0, rest = 0.0.
        let mut input = vec![0.0f32; 1024];
        input[0] = 1.0;

        let output = reverb.process(&input);
        assert_eq!(output.len(), input.len());

        // The output should have energy spread beyond just the first sample.
        let non_zero_count = output.iter().filter(|&&s| s.abs() > 1e-8).count();
        assert!(
            non_zero_count > 1,
            "expected reverb energy spread, got {non_zero_count} non-zero samples",
        );
    }

    /// Config validation rejects out-of-range values.
    #[test]
    fn test_config_validation() {
        assert!(ConvolutionConfig::new()
            .with_wet_mix(-0.1)
            .validate()
            .is_err());
        assert!(ConvolutionConfig::new()
            .with_wet_mix(1.1)
            .validate()
            .is_err());
        assert!(ConvolutionConfig::new()
            .with_wet_mix(f32::NAN)
            .validate()
            .is_err());
        assert!(ConvolutionConfig::new()
            .with_pre_delay_ms(-1.0)
            .validate()
            .is_err());
        assert!(ConvolutionConfig::new()
            .with_pre_delay_ms(101.0)
            .validate()
            .is_err());
        assert!(ConvolutionConfig::new()
            .with_ir_length_limit(0)
            .validate()
            .is_err());
        assert!(ConvolutionConfig::new().validate().is_ok());
    }

    /// FFT → IFFT round-trip preserves the signal.
    #[test]
    fn test_fft_roundtrip() {
        let n = 64;
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
            assert!(
                (orig.1 - recovered.1).abs() < 1e-4,
                "sample {i}: im mismatch: {} vs {}",
                orig.1,
                recovered.1,
            );
        }
    }

    /// reset() clears internal state so a second pass produces identical output.
    #[test]
    fn test_reset_clears_state() {
        let config = ConvolutionConfig::new()
            .with_wet_mix(0.5)
            .with_normalize_ir(false);
        let mut reverb = ConvolutionReverb::new(&config).unwrap();
        let ir = generate_synthetic_ir(SyntheticRoom::SmallRoom);
        reverb.load_ir(&ir, &config);

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();

        let output1 = reverb.process(&input);
        reverb.reset();
        let output2 = reverb.process(&input);

        for (i, (&a, &b)) in output1.iter().zip(output2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "sample {i}: outputs differ after reset: {a} vs {b}",
            );
        }
    }

    /// IR truncation respects ir_length_limit.
    #[test]
    fn test_ir_truncation() {
        let config = ConvolutionConfig::new()
            .with_ir_length_limit(100)
            .with_normalize_ir(false);
        let mut reverb = ConvolutionReverb::new(&config).unwrap();

        let long_ir = vec![0.5f32; 500];
        reverb.load_ir(&long_ir, &config);

        assert_eq!(reverb.ir_len, 100, "IR should be truncated to limit");
    }

    /// Pre-delay shifts the wet signal in time.
    #[test]
    fn test_pre_delay_shifts_signal() {
        // No pre-delay.
        let config_no_delay = ConvolutionConfig::new()
            .with_wet_mix(1.0)
            .with_normalize_ir(false)
            .with_pre_delay_ms(0.0);
        let mut reverb_no_delay = ConvolutionReverb::new(&config_no_delay).unwrap();
        reverb_no_delay.load_ir(&[1.0], &config_no_delay);

        // With pre-delay.
        let config_delay = ConvolutionConfig::new()
            .with_wet_mix(1.0)
            .with_normalize_ir(false)
            .with_pre_delay_ms(10.0);
        let mut reverb_delay = ConvolutionReverb::new(&config_delay).unwrap();
        reverb_delay.load_ir(&[1.0], &config_delay);

        let input: Vec<f32> = (0..1024).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

        let out_no_delay = reverb_no_delay.process(&input);
        let out_delay = reverb_delay.process(&input);

        // The no-delay output should have energy at sample 0.
        assert!(
            out_no_delay[0].abs() > 0.5,
            "no-delay should have immediate output"
        );

        // The delayed output should have near-zero at sample 0 and energy later.
        assert!(
            out_delay[0].abs() < 1e-6,
            "delayed output should be silent at sample 0, got {}",
            out_delay[0],
        );
        // Pre-delay of 10ms at 24kHz = 240 samples.
        let delay_samples = (10.0 / 1000.0 * KOKORO_SAMPLE_RATE as f32) as usize;
        assert!(
            out_delay[delay_samples].abs() > 0.5,
            "delayed output should have energy at sample {delay_samples}, got {}",
            out_delay[delay_samples],
        );
    }
}
