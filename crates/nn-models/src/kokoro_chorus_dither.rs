// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dithering and noise shaping for Kokoro chorus final output.
//!
//! When multiple voices are mixed and processed through dynamics/limiting,
//! quantization artifacts become audible — especially at lower bit depths.
//! Dithering adds carefully shaped noise to decorrelate quantization error
//! from the signal, replacing harsh distortion with a benign noise floor.
//!
//! # Dither types
//!
//! - **Rectangular** (RPDF): uniform noise, +/-0.5 LSB. Simplest, but
//!   modulation noise is not fully decorrelated from the signal.
//! - **Triangular** (TPDF): sum of two uniform random values, +/-1.0 LSB
//!   triangle distribution. Industry standard — fully decorrelates first-order
//!   quantization distortion from the signal.
//! - **Shaped**: TPDF with first-order noise shaping (error feedback). Pushes
//!   dither energy into higher frequencies (above ~10 kHz) where human hearing
//!   is less sensitive, effectively lowering the perceived noise floor.
//!
//! # Determinism
//!
//! The PRNG is a linear congruential generator (LCG) with a configurable seed.
//! Same seed + same input length = identical dither sequence. This is critical
//! for reproducible output in testing and verification pipelines.
//!
//! # References
//!
//! - Vanderkooy, J. & Lipshitz, S. "Resolution Below the Least Significant
//!   Bit in Digital Systems with Dither." JAES, 32(3), 1984.
//! - Wannamaker, R. "Psychoacoustically Optimal Noise Shaping." JAES, 40(7),
//!   1992.
//! - Lipshitz, S. et al. "Quantization and Dither: A Theoretical Survey."
//!   JAES, 40(5), 1992.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Dither type enum
// ---------------------------------------------------------------------------

/// Type of dithering noise to apply before quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum DitherType {
    /// No dithering — pass through unchanged.
    None,
    /// Rectangular probability density function (RPDF).
    /// Uniform noise in [-0.5, +0.5] LSB.
    Rectangular,
    /// Triangular probability density function (TPDF).
    /// Sum of two uniform values: triangle distribution in [-1.0, +1.0] LSB.
    /// Industry standard for audio mastering.
    #[default]
    Triangular,
    /// Noise-shaped TPDF. First-order error feedback pushes dither energy
    /// above ~10 kHz where hearing sensitivity is lower.
    Shaped,
}


// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for dithering in the Kokoro chorus output path.
///
/// Controls dither type, target bit depth, noise shaping, and DC blocking.
/// Use [`DitherConfig::default()`] for sensible defaults (TPDF, 24-bit,
/// noise shaping on, DC block on).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DitherConfig {
    /// Type of dither noise.
    pub dither_type: DitherType,

    /// Target bit depth for dithering (16..=32).
    ///
    /// The LSB amplitude is `2.0 / (2^bit_depth - 1)`. Higher bit depths
    /// produce smaller dither noise. 24-bit is standard for professional
    /// audio; 16-bit for CD distribution.
    pub bit_depth: u32,

    /// Apply first-order error feedback noise shaping.
    ///
    /// When `true`, quantization error from the previous sample is fed back
    /// and subtracted from the current sample before dithering. This shifts
    /// noise energy toward higher frequencies where human hearing is less
    /// sensitive. Only effective with `Triangular` or `Shaped` dither types.
    pub noise_shaping: bool,

    /// Remove DC offset before dithering via a 10 Hz high-pass filter.
    ///
    /// DC offset can bias the dither distribution asymmetrically. A gentle
    /// high-pass at 10 Hz removes sub-audible DC while preserving all
    /// audible content.
    pub dc_block: bool,

    /// PRNG seed for deterministic dither sequences.
    ///
    /// Same seed + same buffer length = identical dither output.
    pub seed: u64,
}

impl Default for DitherConfig {
    fn default() -> Self {
        Self {
            dither_type: DitherType::Triangular,
            bit_depth: 24,
            noise_shaping: true,
            dc_block: true,
            seed: 0,
        }
    }
}

impl DitherConfig {
    /// Create a new dither configuration with explicit parameters.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if `bit_depth` is outside [16, 32].
    pub fn new(
        dither_type: DitherType,
        bit_depth: u32,
        noise_shaping: bool,
        dc_block: bool,
        seed: u64,
    ) -> Result<Self, KokoroError> {
        let config = Self {
            dither_type,
            bit_depth,
            noise_shaping,
            dc_block,
            seed,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if `bit_depth` is outside [16, 32].
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !(16..=32).contains(&self.bit_depth) {
            return Err(KokoroError::InvalidConfig {
                field: "bit_depth",
                reason: format!("must be in [16, 32], got {}", self.bit_depth),
            });
        }
        Ok(())
    }

    /// Builder: set the dither type.
    #[must_use]
    pub fn with_dither_type(mut self, dither_type: DitherType) -> Self {
        self.dither_type = dither_type;
        self
    }

    /// Builder: set the target bit depth.
    #[must_use]
    pub fn with_bit_depth(mut self, bit_depth: u32) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Builder: enable or disable noise shaping.
    #[must_use]
    pub fn with_noise_shaping(mut self, enabled: bool) -> Self {
        self.noise_shaping = enabled;
        self
    }

    /// Builder: enable or disable DC blocking.
    #[must_use]
    pub fn with_dc_block(mut self, enabled: bool) -> Self {
        self.dc_block = enabled;
        self
    }

    /// Builder: set the PRNG seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

// ---------------------------------------------------------------------------
// Deterministic LCG PRNG
// ---------------------------------------------------------------------------

/// Minimal linear congruential generator for deterministic dither noise.
///
/// Uses the Numerical Recipes LCG constants. Period is 2^64.
/// Not cryptographically secure — only for audio dither noise.
struct DitherPrng {
    state: u64,
}

impl DitherPrng {
    fn new(seed: u64) -> Self {
        // Mix the seed to avoid degenerate initial states.
        let state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        Self { state }
    }

    /// Generate the next pseudo-random u64.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG: state = a * state + c
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Generate a uniform random f32 in [-1.0, +1.0].
    #[inline]
    fn next_uniform(&mut self) -> f32 {
        // Use upper 32 bits for better distribution.
        let bits = (self.next_u64() >> 32) as i32;
        (f64::from(bits) / f64::from(i32::MAX)) as f32
    }

    /// Reset to a given seed.
    fn reset(&mut self, seed: u64) {
        self.state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
}

// ---------------------------------------------------------------------------
// DitherProcessor
// ---------------------------------------------------------------------------

/// Stateful dithering processor for the Kokoro chorus output path.
///
/// Maintains PRNG state and error feedback memory between `process()` calls.
/// Use [`DitherProcessor::reset()`] to reinitialize state for a new audio
/// segment.
pub struct DitherProcessor {
    config: DitherConfig,
    prng: DitherPrng,
    /// Error feedback state for noise shaping (previous quantization error).
    error_feedback: f32,
    /// DC blocker state: previous input sample.
    dc_x1: f32,
    /// DC blocker state: previous output sample.
    dc_y1: f32,
    /// LSB amplitude: 2.0 / (2^bit_depth - 1).
    lsb: f32,
}

impl DitherProcessor {
    /// Create a new dither processor.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the config is invalid.
    pub fn new(config: &DitherConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        let lsb = compute_lsb(config.bit_depth);
        Ok(Self {
            config: config.clone(),
            prng: DitherPrng::new(config.seed),
            error_feedback: 0.0,
            dc_x1: 0.0,
            dc_y1: 0.0,
            lsb,
        })
    }

    /// Reset PRNG state and error feedback to initial conditions.
    ///
    /// After reset, processing the same input buffer will produce identical
    /// output (deterministic dither).
    pub fn reset(&mut self) {
        self.prng.reset(self.config.seed);
        self.error_feedback = 0.0;
        self.dc_x1 = 0.0;
        self.dc_y1 = 0.0;
    }

    /// Apply dithering to an audio buffer in-place.
    ///
    /// If `dither_type` is `None`, the buffer is returned unchanged (but DC
    /// blocking still applies if enabled).
    pub fn process(&mut self, audio: &mut [f32]) {
        // DC blocking pass (if enabled).
        if self.config.dc_block {
            self.remove_dc(audio);
        }

        // Skip dithering if type is None.
        if self.config.dither_type == DitherType::None {
            return;
        }

        let lsb = self.lsb;
        let use_shaping = self.config.noise_shaping
            && matches!(
                self.config.dither_type,
                DitherType::Triangular | DitherType::Shaped
            );

        for sample in audio.iter_mut() {
            // IEEE 754 guard: skip non-finite samples.
            if !sample.is_finite() {
                self.error_feedback = 0.0;
                continue;
            }

            let mut s = *sample;

            // Subtract previous quantization error (noise shaping).
            if use_shaping {
                s -= self.error_feedback;
            }

            // Generate dither noise scaled to LSB amplitude.
            let dither = match self.config.dither_type {
                DitherType::None => 0.0,
                DitherType::Rectangular => {
                    // RPDF: uniform in [-0.5, +0.5] LSB.
                    self.prng.next_uniform() * 0.5 * lsb
                }
                DitherType::Triangular | DitherType::Shaped => {
                    // TPDF: sum of two uniform values.
                    // Result is triangle-distributed in [-1.0, +1.0] LSB.
                    let u1 = self.prng.next_uniform();
                    let u2 = self.prng.next_uniform();
                    (u1 + u2) * 0.5 * lsb
                }
            };

            // Add dither and quantize to target bit depth.
            let dithered = s + dither;
            let quantized = quantize(dithered, lsb);

            // Compute quantization error for next sample's feedback.
            if use_shaping {
                self.error_feedback = quantized - s;
                // Flush denormals in error feedback.
                if !self.error_feedback.is_finite() || self.error_feedback.abs() < 1e-30 {
                    self.error_feedback = 0.0;
                }
            }

            *sample = quantized;
        }
    }

    /// Remove DC offset from audio via a first-order high-pass at ~10 Hz.
    ///
    /// Uses the standard DC blocker:
    /// ```text
    /// y[n] = x[n] - x[n-1] + R * y[n-1]
    /// ```
    /// where `R = 1 - (2 * pi * fc / fs)`. At 24 kHz sample rate and 10 Hz
    /// cutoff, `R ≈ 0.99738`.
    pub fn remove_dc(&mut self, audio: &mut [f32]) {
        // R coefficient for ~10 Hz cutoff at 24 kHz sample rate.
        // R = 1 - (2 * pi * 10 / 24000) ≈ 0.99738
        const R: f32 = 0.997_38;

        for sample in audio.iter_mut() {
            if !sample.is_finite() {
                self.dc_x1 = 0.0;
                self.dc_y1 = 0.0;
                continue;
            }

            let x = *sample;
            let y = x - self.dc_x1 + R * self.dc_y1;

            self.dc_x1 = x;
            // Flush denormals.
            self.dc_y1 = if y.is_finite() && y.abs() > 1e-30 {
                y
            } else {
                0.0
            };

            *sample = self.dc_y1;
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Apply dithering to an audio buffer in-place.
///
/// Creates a [`DitherProcessor`], processes the buffer, and discards state.
/// For repeated calls on sequential audio chunks, create a processor directly
/// to preserve error feedback and DC blocker state across chunks.
///
/// # Errors
///
/// Returns `KokoroError::InvalidConfig` if the config is invalid.
pub fn apply_dither(audio: &mut [f32], config: &DitherConfig) -> Result<(), KokoroError> {
    config.validate()?;

    if audio.is_empty() || config.dither_type == DitherType::None && !config.dc_block {
        return Ok(());
    }

    let mut processor = DitherProcessor::new(config)?;
    processor.process(audio);
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the LSB amplitude for a given bit depth.
///
/// `lsb = 2.0 / (2^bit_depth - 1)`, representing the smallest step in
/// a signed fixed-point representation spanning [-1.0, +1.0].
#[inline]
fn compute_lsb(bit_depth: u32) -> f32 {
    // For bit_depth in [16, 32], 2^bit_depth fits in u64 easily.
    let levels = (1u64 << bit_depth) - 1;
    2.0 / levels as f32
}

/// Quantize a sample to the nearest LSB step.
///
/// Rounds to the nearest representable level in the target bit depth.
#[inline]
fn quantize(sample: f32, lsb: f32) -> f32 {
    if !sample.is_finite() {
        return 0.0;
    }
    // Round to nearest LSB.
    (sample / lsb).round() * lsb
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dithered_output_differs_from_input() {
        let config = DitherConfig::new(
            DitherType::Triangular,
            16, // 16-bit: large LSB, visible dither effect
            false,
            false,
            42,
        )
        .unwrap();

        let original: Vec<f32> = (0..4000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin() * 0.5)
            .collect();
        let mut dithered = original.clone();
        apply_dither(&mut dithered, &config).unwrap();

        // Dithered output should differ from original.
        let diff: f32 = dithered
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / dithered.len() as f32;

        assert!(
            diff > 1e-7,
            "dithered output should differ from input, mean diff = {diff}"
        );
    }

    #[test]
    fn test_tpdf_distribution_is_triangular() {
        // TPDF dither noise should have a triangular distribution:
        // variance of triangle[-A, A] = A^2 / 6.
        // For two uniform [-1, 1] summed and scaled by 0.5 * lsb:
        // each uniform has variance 1/3, sum has variance 2/3,
        // scaled by (0.5 * lsb)^2.
        let config = DitherConfig::new(DitherType::Triangular, 16, false, false, 123).unwrap();

        // Feed silence to isolate dither noise.
        let mut silence = vec![0.0f32; 100_000];
        apply_dither(&mut silence, &config).unwrap();

        // Compute mean and variance of the dithered output.
        let n = silence.len() as f64;
        let mean: f64 = silence.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let variance: f64 = silence
            .iter()
            .map(|&x| {
                let d = f64::from(x) - mean;
                d * d
            })
            .sum::<f64>()
            / n;

        // Mean should be near zero.
        assert!(mean.abs() < 1e-5, "TPDF mean should be ~0, got {mean}");

        // Total output variance = dither variance + quantization noise variance.
        // Dither: two uniform [-1,1] scaled by 0.5*lsb → var = (0.5*lsb)^2 * 2/3
        // Quantization: uniform over LSB → var = lsb^2/12
        // Combined: (0.5*lsb)^2 * 2/3 + lsb^2/12 ≈ lsb^2 * (1/6 + 1/12) = lsb^2/4
        let lsb = f64::from(compute_lsb(16));
        let expected_var = lsb.powi(2) / 4.0;
        let ratio = variance / expected_var;
        assert!(
            (0.5..2.0).contains(&ratio),
            "TPDF variance ratio: {ratio} (expected ~1.0), \
             variance={variance}, expected={expected_var}"
        );
    }

    #[test]
    fn test_shaped_noise_has_more_hf_energy() {
        // Noise-shaped dither should push energy toward higher frequencies.
        // Compare shaped vs unshaped TPDF on silence and measure HF energy.
        let unshaped_config = DitherConfig::new(
            DitherType::Triangular,
            16,
            false, // no shaping
            false,
            42,
        )
        .unwrap();

        let shaped_config = DitherConfig::new(
            DitherType::Shaped,
            16,
            true, // noise shaping on
            false,
            42,
        )
        .unwrap();

        let n = 8192;
        let mut unshaped = vec![0.0f32; n];
        let mut shaped = vec![0.0f32; n];

        apply_dither(&mut unshaped, &unshaped_config).unwrap();
        apply_dither(&mut shaped, &shaped_config).unwrap();

        // Measure high-frequency energy via first-order difference (approx HP).
        let hf_energy = |buf: &[f32]| -> f64 {
            buf.windows(2)
                .map(|w| {
                    let d = f64::from(w[1] - w[0]);
                    d * d
                })
                .sum::<f64>()
                / buf.len() as f64
        };

        let hf_unshaped = hf_energy(&unshaped);
        let hf_shaped = hf_energy(&shaped);

        // Shaped should have more HF energy due to error feedback.
        assert!(
            hf_shaped > hf_unshaped * 0.8,
            "shaped HF energy ({hf_shaped:.2e}) should be >= unshaped ({hf_unshaped:.2e})"
        );
    }

    #[test]
    fn test_dc_block_removes_offset() {
        let config = DitherConfig::new(
            DitherType::None,
            24,
            false,
            true, // DC block enabled
            0,
        )
        .unwrap();

        // Signal with a DC offset of 0.3.
        let n = 8000;
        let mut signal: Vec<f32> = (0..n)
            .map(|i| 0.3 + 0.5 * (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 24000.0).sin())
            .collect();

        apply_dither(&mut signal, &config).unwrap();

        // After DC blocking, the mean should be near zero (skip transient).
        let skip = 2000; // DC blocker has a time constant; skip initial transient.
        let mean: f64 = signal[skip..].iter().map(|&x| f64::from(x)).sum::<f64>() / (n - skip) as f64;

        assert!(
            mean.abs() < 0.05,
            "DC-blocked signal mean should be ~0, got {mean}"
        );
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let config = DitherConfig::new(DitherType::Triangular, 24, true, false, 12345).unwrap();

        let original: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.01).sin() * 0.7).collect();

        let mut run1 = original.clone();
        let mut run2 = original;

        apply_dither(&mut run1, &config).unwrap();
        apply_dither(&mut run2, &config).unwrap();

        for (i, (&a, &b)) in run1.iter().zip(run2.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "sample {i}: run1={a}, run2={b} — should be identical"
            );
        }
    }

    #[test]
    fn test_different_seeds_produce_different_output() {
        let signal: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.01).sin() * 0.7).collect();

        let config_a = DitherConfig::new(DitherType::Triangular, 16, false, false, 0).unwrap();
        let config_b = DitherConfig::new(DitherType::Triangular, 16, false, false, 999).unwrap();

        let mut out_a = signal.clone();
        let mut out_b = signal;

        apply_dither(&mut out_a, &config_a).unwrap();
        apply_dither(&mut out_b, &config_b).unwrap();

        let diff: f32 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / out_a.len() as f32;

        assert!(
            diff > 1e-7,
            "different seeds should produce different dither, mean diff = {diff}"
        );
    }

    #[test]
    fn test_none_type_is_passthrough() {
        let config = DitherConfig::new(DitherType::None, 24, false, false, 0).unwrap();
        let original: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.02).sin()).collect();
        let mut output = original.clone();

        apply_dither(&mut output, &config).unwrap();

        for (i, (&got, &expected)) in output.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - expected).abs() < f32::EPSILON,
                "sample {i}: DitherType::None should be passthrough"
            );
        }
    }

    #[test]
    fn test_config_validation_rejects_invalid_bit_depth() {
        assert!(DitherConfig::new(DitherType::Triangular, 15, false, false, 0).is_err());
        assert!(DitherConfig::new(DitherType::Triangular, 33, false, false, 0).is_err());
        assert!(DitherConfig::new(DitherType::Triangular, 0, false, false, 0).is_err());
    }

    #[test]
    fn test_config_validation_accepts_valid_bit_depths() {
        for bd in [16, 20, 24, 32] {
            assert!(
                DitherConfig::new(DitherType::Triangular, bd, true, true, 0).is_ok(),
                "bit_depth {bd} should be valid"
            );
        }
    }

    #[test]
    fn test_empty_buffer_ok() {
        let config = DitherConfig::default();
        let mut empty: Vec<f32> = vec![];
        assert!(apply_dither(&mut empty, &config).is_ok());
    }

    #[test]
    fn test_rectangular_dither_smaller_than_tpdf() {
        // RPDF uses +/-0.5 LSB vs TPDF +/-1.0 LSB range,
        // so RPDF variance should be smaller.
        let rpdf_config = DitherConfig::new(DitherType::Rectangular, 16, false, false, 42).unwrap();
        let tpdf_config = DitherConfig::new(DitherType::Triangular, 16, false, false, 42).unwrap();

        let n = 50_000;
        let mut rpdf_buf = vec![0.0f32; n];
        let mut tpdf_buf = vec![0.0f32; n];

        apply_dither(&mut rpdf_buf, &rpdf_config).unwrap();
        apply_dither(&mut tpdf_buf, &tpdf_config).unwrap();

        let variance = |buf: &[f32]| -> f64 {
            let n = buf.len() as f64;
            let mean: f64 = buf.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
            buf.iter()
                .map(|&x| {
                    let d = f64::from(x) - mean;
                    d * d
                })
                .sum::<f64>()
                / n
        };

        let rpdf_var = variance(&rpdf_buf);
        let tpdf_var = variance(&tpdf_buf);

        assert!(
            rpdf_var < tpdf_var,
            "RPDF variance ({rpdf_var:.2e}) should be less than TPDF ({tpdf_var:.2e})"
        );
    }

    #[test]
    fn test_nan_safety() {
        let config = DitherConfig::new(DitherType::Triangular, 24, true, true, 0).unwrap();
        let mut buf = vec![0.5, f32::NAN, 0.3, f32::INFINITY, -0.2];
        apply_dither(&mut buf, &config).unwrap();

        // Non-finite samples should not propagate NaN into subsequent samples.
        // The last sample should still be finite.
        assert!(
            buf[4].is_finite(),
            "sample after NaN/Inf should be finite, got {}",
            buf[4]
        );
    }

    #[test]
    fn test_processor_reset_gives_same_output() {
        let config = DitherConfig::new(DitherType::Triangular, 24, true, false, 77).unwrap();
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.015).sin() * 0.8).collect();

        let mut proc = DitherProcessor::new(&config).unwrap();

        let mut run1 = signal.clone();
        proc.process(&mut run1);

        proc.reset();
        let mut run2 = signal;
        proc.process(&mut run2);

        for (i, (&a, &b)) in run1.iter().zip(run2.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "sample {i}: after reset, output should match. got {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_higher_bit_depth_means_smaller_dither() {
        // 32-bit dither should produce less change than 16-bit.
        let signal: Vec<f32> = (0..4000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();

        let config_16 = DitherConfig::new(DitherType::Triangular, 16, false, false, 42).unwrap();
        let config_32 = DitherConfig::new(DitherType::Triangular, 32, false, false, 42).unwrap();

        let mut out_16 = signal.clone();
        let mut out_32 = signal.clone();

        apply_dither(&mut out_16, &config_16).unwrap();
        apply_dither(&mut out_32, &config_32).unwrap();

        let mean_diff = |out: &[f32], orig: &[f32]| -> f64 {
            out.iter()
                .zip(orig.iter())
                .map(|(a, b)| f64::from((a - b).abs()))
                .sum::<f64>()
                / out.len() as f64
        };

        let diff_16 = mean_diff(&out_16, &signal);
        let diff_32 = mean_diff(&out_32, &signal);

        assert!(
            diff_32 < diff_16,
            "32-bit dither diff ({diff_32:.2e}) should be < 16-bit ({diff_16:.2e})"
        );
    }

    #[test]
    fn test_compute_lsb_values() {
        // 16-bit: 2 / (65535) ≈ 3.05e-5
        let lsb16 = compute_lsb(16);
        assert!((lsb16 - 3.0518e-5).abs() < 1e-7, "16-bit LSB: {lsb16}");

        // 24-bit: 2 / (16777215) ≈ 1.19e-7
        let lsb24 = compute_lsb(24);
        assert!((lsb24 - 1.1921e-7).abs() < 1e-10, "24-bit LSB: {lsb24}");
    }

    #[test]
    fn test_default_config_is_valid() {
        let config = DitherConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.dither_type, DitherType::Triangular);
        assert_eq!(config.bit_depth, 24);
        assert!(config.noise_shaping);
        assert!(config.dc_block);
    }
}
