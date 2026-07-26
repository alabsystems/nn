// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro TTS audio reconstruction: iSTFT waveform synthesis from spectrogram.
//!
//! Extends `KokoroModel` with `forward_audio()` that produces PCM audio directly,
//! completing the end-to-end pipeline: token IDs → audio waveform.

use crate::kokoro_error::{check_tensor_finite, KokoroError};
use crate::kokoro_istft::{kokoro_istft, KokoroIstftParams};
use crate::kokoro_tts::KokoroModel;
use nn_core::dyn_tensor::DynTensor;

impl KokoroModel {
    /// Full forward: token IDs → audio waveform via iSTFT.
    ///
    /// Runs `forward()` to produce (magnitude, phase), then reconstructs audio
    /// via CPU iSTFT (DFT matmul + Hann window overlap-add).
    ///
    /// Returns `[1, 1, T_audio]` PCM audio at 24kHz.
    pub fn forward_audio(
        &self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
    ) -> Result<DynTensor, KokoroError> {
        let (magnitude, phase) = self.forward(input_ids, style, speed)?;
        self.audio_from_spectrogram(&magnitude, &phase)
    }

    /// Reconstruct audio from (magnitude, phase) spectrogram via iSTFT.
    ///
    /// `magnitude`: `[B, n_bins, T_out]` — non-negative magnitude.
    /// `phase`: `[B, n_bins, T_out]` — phase in radians (direct network output,
    ///   NOT scaled by π).
    ///
    /// Returns `[1, 1, T_audio]` PCM audio with center trim matching
    /// `torch.istft(center=True)`.
    ///
    /// Used by `forward_audio()` and the har_source bypass test (D1 in
    /// `designs/2026-03-19-kokoro-ac1-decomposed-correctness.md`).
    pub fn audio_from_spectrogram(
        &self,
        magnitude: &DynTensor,
        phase: &DynTensor,
    ) -> Result<DynTensor, KokoroError> {
        let cos_phase = phase.cos()?;
        let sin_phase = phase.sin()?;
        let real_spec = magnitude.mul(&cos_phase)?;
        let imag_spec = magnitude.mul(&sin_phase)?;

        let n_bins = magnitude.dim(1)?;
        let n_frames = magnitude.dim(2)?;

        let real_cpu = real_spec.to_device(&nn_core::Device::Cpu)?;
        let real_arr = real_cpu.to_f32_array()?;
        let real_std = real_arr.as_standard_layout();
        let real_flat = real_std.as_slice().ok_or(KokoroError::IstftArrayLayout)?;
        let imag_cpu = imag_spec.to_device(&nn_core::Device::Cpu)?;
        let imag_arr = imag_cpu.to_f32_array()?;
        let imag_std = imag_arr.as_standard_layout();
        let imag_flat = imag_std.as_slice().ok_or(KokoroError::IstftArrayLayout)?;

        let n_fft = self.config().n_fft;
        let hop = n_fft / 4;
        let output_length = n_fft + n_frames.saturating_sub(1) * hop;

        let istft_params = KokoroIstftParams {
            n_fft,
            hop_length: hop,
        };

        let expected_bins = n_fft / 2 + 1;
        if n_bins != expected_bins {
            return Err(KokoroError::IstftBinMismatch {
                actual: n_bins,
                expected: expected_bins,
                n_fft,
            });
        }

        let audio_pcm = kokoro_istft(&istft_params, real_flat, imag_flat, n_frames, output_length)
            .map_err(KokoroError::IstftFailed)?;

        // Center trim: remove n_fft/2 from each side to match torch.istft(center=True).
        let center_pad = n_fft / 2;
        let trim_end = audio_pcm.len().saturating_sub(center_pad);
        let trimmed = if center_pad < trim_end {
            &audio_pcm[center_pad..trim_end]
        } else {
            &audio_pcm[..]
        };

        let audio_len = trimmed.len();
        let audio = DynTensor::new(trimmed, &[1, 1, audio_len], &magnitude.device())?;
        check_tensor_finite(&audio, "istft_output")?;
        let audio = audio.clamp(-1.0, 1.0)?;

        Ok(audio)
    }
}

#[cfg(kani)]
#[path = "kokoro_audio_kani_tests.rs"]
mod kani_proofs;

#[cfg(test)]
mod tests {
    use crate::kokoro_istft::{kokoro_istft, KokoroIstftParams};

    #[test]
    fn test_kokoro_istft_pure_tone_reconstruction() {
        // Generate a simple pure-tone STFT and verify iSTFT produces finite audio.
        let n_fft = 20;
        let hop = 5;
        let n_bins = n_fft / 2 + 1; // 11
        let n_frames = 8;
        let output_length = n_frames * hop; // 40

        // Put energy only in the DC bin (f=0): a constant signal
        let mut real = vec![0.0f32; n_bins * n_frames];
        let imag = vec![0.0f32; n_bins * n_frames];
        for v in real.iter_mut().take(n_frames) {
            *v = 1.0; // DC real = 1.0 for all frames
        }

        let params = KokoroIstftParams {
            n_fft,
            hop_length: hop,
        };
        let audio = kokoro_istft(&params, &real, &imag, n_frames, output_length).unwrap();
        assert_eq!(audio.len(), output_length);
        // All values should be finite
        for v in &audio {
            assert!(v.is_finite(), "non-finite value in iSTFT output: {v}");
        }
        // DC-only signal should produce roughly constant output after windowing
        let mid = &audio[n_fft / 2..audio.len() - n_fft / 2];
        if !mid.is_empty() {
            let mean: f32 = mid.iter().sum::<f32>() / mid.len() as f32;
            for v in mid {
                assert!(
                    (v - mean).abs() < 0.1,
                    "DC signal not constant: {v} vs mean {mean}"
                );
            }
        }
    }

    #[test]
    fn test_kokoro_istft_all_zero_produces_zero_audio() {
        let n_fft = 20;
        let hop = 5;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 4;
        let output_length = n_frames * hop;

        let real = vec![0.0f32; n_bins * n_frames];
        let imag = vec![0.0f32; n_bins * n_frames];

        let params = KokoroIstftParams {
            n_fft,
            hop_length: hop,
        };
        let audio = kokoro_istft(&params, &real, &imag, n_frames, output_length).unwrap();
        for v in &audio {
            assert!(*v == 0.0 || v.abs() < 1e-10, "expected zero, got {v}");
        }
    }
}
